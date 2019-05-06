'''
latent_ode.py

Provides experiment() and experiment_1(), experiment_2(), experiment_3(), experiment_1_small().

See README for command line usage.
'''
import os
import argparse
import logging
import time
import numpy as np
import numpy.random as npr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from time import time

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--exp1',    action='store_true')
parser.add_argument('--exp1s',    action='store_true')
parser.add_argument('--exp2',    action='store_true')
parser.add_argument('--exp3',    action='store_true')
parser.add_argument('--device',  type=str, default="cuda:0") # "cpu", "cuda:0", "cuda:1", ...
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

from parametric_dataset import generate_spirals_nonaugmented, generate_spirals_augmented, generate_parametric, generate_spirals_nonaugmented_small
# returns orig_trajs, samp_trajs, orig_ts, samp_ts labels
# replaces generate_spiral2d

class LatentODEfunc(nn.Module):
    def __init__(self, latent_dim=4, nhidden=20, depth = 1):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fcH = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(depth)])
        self.fc_out = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        for fch in self.fcH:
            out = fch(out)
            out = self.elu(out)
        out = self.fc_out(out)
        return out


class RecognitionRNN(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=25, nbatch=1):
        super(RecognitionRNN, self).__init__()
        self.nhidden = nhidden
        self.nbatch = nbatch
        self.i2h = nn.Linear(obs_dim + nhidden, nhidden)
        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def forward(self, x, h):
        combined = torch.cat((x, h), dim=1)
        h = torch.tanh(self.i2h(combined))
        out = self.h2o(h)
        return out, h

    def initHidden(self):
        return torch.zeros(self.nbatch, self.nhidden)


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, obs_dim=2, nhidden=20, depth=0):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(latent_dim, nhidden)
        self.fcH = nn.ModuleList([nn.Linear(nhidden, nhidden) for _ in range(depth)])
        self.fc2 = nn.Linear(nhidden, obs_dim)

    def forward(self, z):
        out = self.fc1(z)
        out = self.relu(out)
        for fch in self.fcH:
            out = fch(out)
            out = self.relu(out)
        out = self.fc2(out)
        return out


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def experiment(generate_data = generate_spirals_nonaugmented, 
               latent_dim = 4,
               nhidden = 20,
               rnn_nhidden = 25,
               hidden_depth = 2,
               obs_dim = 2,
               noise_std = 2**-7,
               epochs = 1,
               load_from_fn = None,
               save_to_fn = "ckpt_{}.pth",
               vis_fn = "vis_{}.png",
               viscount = 1,
               lr = 0.001):
    '''A more generic form of the experiment from 5.1 of the paper.
    
    # Arguments:
    latent_dim: Int, The size of the latent space
    nhidden: Int, The width of the models used.
    rnn_nhidden: Int, width of the RNN
    hidden_depth: Int, the number of hidden layers to use in the neural ODE and in the decoder
    obs_dim: Int, how many dimensions we work in.
    noise_std: Real, The std of Gaussian noise to add per observation.
    epochs: Int, Number of iterations to train for
    load_from_fn: String; reload a saved state from here.
            Alternatively, None.
    save_to_fn: String with one formattable field. Location to save model state to.
            Alternatively, None.
    vis_fn: String with one formattable field. Location to save visualizations to.
            Alternatively, None.
    viscount: Int >= 0; number of images to save when visualizing.
    lr: Real; learning rate.
    
    # Returns:
    func (PyTorch neural ODE model), rec (PyTorch RNN model), dec (PyTorch decoding MLP),
    params (list of parameters to func, dec, RNN),
    optimizer (PyTorch optim.Adam), loss_meter (PyTorch RunningAverageMeter)
    '''
    device = torch.device(args.device)

    # generate toy spiral data
    orig_trajs, samp_trajs, orig_ts, samp_ts, labels = generate_data()
    orig_trajs = torch.from_numpy(orig_trajs).float().to(device)
    samp_trajs = torch.from_numpy(samp_trajs).float().to(device)
    samp_ts = torch.from_numpy(samp_ts).float().to(device)

    # model
    NUMSAMPLES = orig_trajs.shape[0]
    
    func = LatentODEfunc(latent_dim, nhidden, hidden_depth).to(device)
    rec = RecognitionRNN(latent_dim, obs_dim, rnn_nhidden, NUMSAMPLES).to(device)
    dec = Decoder(latent_dim, obs_dim, nhidden, hidden_depth).to(device)
    params = (list(func.parameters()) + list(dec.parameters()) + list(rec.parameters()))
    optimizer = optim.Adam(params, lr=lr)
    loss_meter = RunningAverageMeter()

    # Load data
    if load_from_fn and os.path.exists(load_from_fn):
        checkpoint = torch.load(load_from_fn)
        func.load_state_dict(checkpoint['func_state_dict'])
        rec.load_state_dict(checkpoint['rec_state_dict'])
        dec.load_state_dict(checkpoint['dec_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        orig_trajs = checkpoint['orig_trajs']
        samp_trajs = checkpoint['samp_trajs']
        orig_ts = checkpoint['orig_ts']
        samp_ts = checkpoint['samp_ts']
        print('Loaded ckpt from {}'.format(load_from_fn))

    # Training
    for itr in range(1, epochs + 1):
        optimizer.zero_grad()
        # backward in time to infer q(z_0)
        h = rec.initHidden().to(device)
        for t in reversed(range(samp_trajs.size(1))):
            obs = samp_trajs[:, t, :]
            out, h = rec.forward(obs, h)
        qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
        epsilon = torch.randn(qz0_mean.size()).to(device)
        z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

        # forward in time and solve ode for reconstructions
        pred_z = odeint(func, z0, samp_ts).permute(1, 0, 2)
        pred_x = dec(pred_z)

        # compute loss
        noise_std_ = torch.zeros(pred_x.size()).to(device) + noise_std
        noise_logvar = 2. * torch.log(noise_std_).to(device)
        logpx = log_normal_pdf(
            samp_trajs, pred_x, noise_logvar).sum(-1).sum(-1)
        pz0_mean = pz0_logvar = torch.zeros(z0.size()).to(device)
        analytic_kl = normal_kl(qz0_mean, qz0_logvar,
                                pz0_mean, pz0_logvar).sum(-1)
        loss = torch.mean(-logpx + analytic_kl, dim=0)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item())

        print('Iter: {}, running avg elbo: {:.4f}'.format(itr, -loss_meter.avg))

    # Save
    if save_to_fn:
        torch.save({
            'func_state_dict': func.state_dict(),
            'rec_state_dict': rec.state_dict(),
            'dec_state_dict': dec.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'orig_trajs': orig_trajs,
            'samp_trajs': samp_trajs,
            'orig_ts': orig_ts,
            'samp_ts': samp_ts,
        }, save_to_fn)
        print('Stored ckpt at {}'.format(save_to_fn))
    print('Training complete after {} iters.'.format(itr))

    # Visualize
    if vis_fn:
        orig_ts = torch.from_numpy(orig_ts).float().to(device)
        for ii in range(viscount):
            with torch.no_grad():
                # sample from trajectorys' approx. posterior
                h = rec.initHidden().to(device)
                for t in reversed(range(samp_trajs.size(1))):
                    obs = samp_trajs[:, t, :]
                    out, h = rec.forward(obs, h)
                qz0_mean, qz0_logvar = out[:, :latent_dim], out[:, latent_dim:]
                epsilon = torch.randn(qz0_mean.size()).to(device)
                z0 = epsilon * torch.exp(.5 * qz0_logvar) + qz0_mean

                # take iith trajectory for visualization
                z0 = z0[ii]

                ts_pos = np.linspace(0., 2. * np.pi, num=2000)
                ts_neg = np.linspace(-np.pi, 0., num=2000)[::-1].copy()
                ts_pos = torch.from_numpy(ts_pos).float().to(device)
                ts_neg = torch.from_numpy(ts_neg).float().to(device)

                zs_pos = odeint(func, z0, ts_pos)
                zs_neg = odeint(func, z0, ts_neg)

                xs_pos = dec(zs_pos)
                xs_neg = torch.flip(dec(zs_neg), dims=[0])

            xs_pos = xs_pos.cpu().numpy()
            xs_neg = xs_neg.cpu().numpy()
            orig_traj = orig_trajs[ii].cpu().numpy()
            samp_traj = samp_trajs[ii].cpu().numpy()

            plt.figure()
            plt.plot(orig_traj[:, 0], orig_traj[:, 1],  'g', label='true trajectory')
            plt.plot(xs_pos[:, 0],    xs_pos[:, 1],     'r', label='learned trajectory (t>0)')
            plt.plot(xs_neg[:, 0],    xs_neg[:, 1],     'c', label='learned trajectory (t<0)')
            plt.scatter(samp_traj[:, 0], samp_traj[:, 1],    label='sampled data', s=3)
            plt.legend()
            
            vis_i_fn = vis_fn.format(ii)
            plt.savefig(vis_i_fn, dpi=200)
            print('Saved visualization figure at {}'.format(vis_i_fn))

    return func, rec, dec, params, optimizer, loss_meter



def experiment_1():
    return experiment(generate_data = generate_spirals_nonaugmented,
                      latent_dim    = 8,
                      nhidden       = 40,
                      rnn_nhidden   = 50,
                      hidden_depth  = 2,
                      epochs        = 1000,
                      save_to_fn    = "./exp1/ckpt_{}.pth".format(round(time())),
                      vis_fn        = "./exp1/vis_{}.png",
                      viscount      = 10)

def experiment_2():
    return experiment(generate_data = generate_spirals_augmented,
                      latent_dim    = 12,
                      nhidden       = 60,
                      rnn_nhidden   = 75,
                      hidden_depth  = 2,
                      epochs        = 4000,
                      save_to_fn    = "./exp2/ckpt_{}.pth".format(round(time())),
                      vis_fn        = "./exp2/vis_{}.png",
                      viscount      = 20)

def experiment_3():
    return experiment(generate_data = generate_parametric,
                      latent_dim    = 24,
                      nhidden       = 120,
                      rnn_nhidden   = 150,
                      hidden_depth  = 3,
                      epochs        = 8000,
                      save_to_fn    = "./exp3/ckpt_{}.pth".format(round(time())),
                      vis_fn        = "./exp3/vis_{}.png",
                      viscount      = 50)

def experiment_1_small():
    return experiment(generate_data = generate_spirals_nonaugmented_small,
                      latent_dim    = 8,
                      nhidden       = 40,
                      rnn_nhidden   = 50,
                      hidden_depth  = 2,
                      epochs        = 1000,
                      save_to_fn    = "./exp1/ckpt_{}.pth".format(round(time())),
                      vis_fn        = "./exp1/vis_{}.png",
                      viscount      = 10)

if __name__ == "__main__":
    if args.exp1:
        exp1 = experiment_1()
    if args.exp1s:
        exp1s = experiment_1_small()
    if args.exp2:
        exp2 = experiment_2()
    if args.exp3:
        exp3 = experiment_3()
