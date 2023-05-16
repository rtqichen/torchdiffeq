import os
import argparse
import glob
import random

from PIL import Image
import numpy as np
import matplotlib

from examples.torch_rbf import basis_func_dict

matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--width', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

# SEED = 42
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# RBF Layer
# https://github.com/mlguy101/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py
class RBF_CNF(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        n_centres: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, n_centres, basis_func, device):
        super(RBF_CNF, self).__init__()
        self.in_features = in_features
        self.n_centers = n_centres
        self.centres = nn.Parameter(torch.Tensor(n_centres, in_features).to(device))
        self.log_sigmas = nn.Parameter(torch.Tensor(n_centres).to(device))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, mean=0, std=1)
        nn.init.constant_(self.log_sigmas, val=0)

    def forward(self, input):
        size = (input.size(0), self.n_centers, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)


class RBFN_CNF(torch.nn.Module):
    # https://en.wikipedia.org/wiki/Radial_basis_function_network
    def __init__(self, in_dim, out_dim, n_centres_t, n_centres_z, basis_fn_str, device):
        super().__init__()
        self.basis_fn_str = basis_fn_str
        self.out_dim = out_dim
        self.in_dim = in_dim
        basis_fn = basis_func_dict()[basis_fn_str]
        self.n_centres_t = n_centres_t
        self.n_centres_z = n_centres_z
        self.rbf_module_z = RBF_CNF(in_features=in_dim, n_centres=self.n_centres_z,
                                    basis_func=basis_fn, device=device)
        self.rbf_module_t = RBF_CNF(in_features=1, n_centres=self.n_centres_t,
                                    basis_func=basis_fn, device=device)
        # rbf inited by its own reset fn
        # self.input_norm_module = \
        #     torch.nn.BatchNorm1d(num_features=in_dim, affine=True) if input_batch_norm \
        #         else torch.nn.Identity()
        # self.linear_module = torch.nn.Linear(in_features=n_centres, out_features=n_centres).to(device)
        self.W = torch.nn.Parameter(torch.empty(out_dim, self.n_centres_t, self.n_centres_z + 1)).to(device)
        # TODO revisit theory for batch-norm
        #   ref : https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        #   ref : https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        #   ref-paper : https://arxiv.org/abs/1502.03167
        #   Note : batch-norm layer is before the non-linearity
        # self.net = torch.nn.Sequential(self.rbf_module, self.linear_module)
        # init
        torch.nn.init.normal_(self.W, mean=0, std=0.01)
        # torch.nn.init.constant_(self.linear_module.bias, val=0)
        # get numel learnable
        self.numel_learnable = 0
        param_list = list(self.named_parameters())
        for name, param in param_list:
            self.numel_learnable += torch.numel(param)

    def forward(self, t: float, states: torch.Tensor):
        z = states[0]
        logp_z = states[1]
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            batchsize = z.shape[0]
            t_tensor = torch.tensor(t).view(1, 1).to(device)
            Phi_t = self.rbf_module_t(t_tensor).flatten()
            W_t = torch.einsum('dij,i->dj', self.W, Phi_t)
            Phi_z = self.rbf_module_z(z)
            Phi_z = torch.cat([Phi_z, torch.ones(batchsize, 1).to(device)], dim=1)
            dz_dt = torch.einsum('dj,bj->bd', W_t, Phi_z)
            # dz_dt = torch.einsum('bi')
            # t_tensor = torch.tensor(t).repeat(batchsize, 1).to(device)
            # z_aug = torch.cat([z, t_tensor], dim=1)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]

        batchsize = z.shape[0]

        with torch.set_grad_enabled(True):
            z.requires_grad_(True)

            W, B, U = self.hyper_net(t)

            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)

            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)

            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)

        return (dz_dt, dlogp_z_dt)


def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """
    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()

    return sum_diag.contiguous()


class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """

    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()

        blocksize = width * in_out_dim

        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)

        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):
        # predict params
        params = t.reshape(1, 1)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        # restructure
        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)

        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)

        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)

        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]


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


def get_batch(num_samples):
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.001)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)

    return (x, logp_diff_t1)


if __name__ == '__main__':
    t0 = 0
    t1 = 10
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')


    # model
    func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
    #func = RBFN_CNF(in_dim=2, out_dim=2, n_centres_z=80, n_centres_t=80,
    #                basis_fn_str="gaussian", device=device)
    # n_scalars = 0
    # for param in list(func.parameters()):
    #     n_scalars += param.numel()
    # print(f'func = {type(func).__name__}\n'
    #       f'n_scalars = {n_scalars}')
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
    )
    loss_meter = RunningAverageMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            x, logp_diff_t1 = get_batch(args.num_samples)

            z_t, logp_diff_t = odeint(
                func,
                (x, logp_diff_t1),
                torch.tensor([t1, t0]).type(torch.float32).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
            # TODO, replace first term with least squares
            #   or calculate .log_prob via least squares
            logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
            loss = -logp_x.mean(0)
            # TODO separate opt loss from loss to add to meter
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())

            print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.viz:
        viz_samples = 30000
        viz_timesteps = 41
        target_sample, _ = get_batch(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)

            z_t_samples, _ = odeint(
                func,
                (z_t0, logp_diff_t0),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Generate evolution of density
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)

            z_t_density, logp_diff_t = odeint(
                func,
                (z_t1, logp_diff_t1),
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                ax1.get_xaxis().set_ticks([])
                ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                ax2.get_xaxis().set_ticks([])
                ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                ax3.get_xaxis().set_ticks([])
                ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t * 1000):05d}.jpg"),
                            pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))