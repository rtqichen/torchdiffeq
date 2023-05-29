"""
The idea of hybrid ALS-GD-Adjoint for Neural-ODE
http://www1.ece.neu.edu/~erdogmus/publications/C041_ICANN2003_LinearLSlearning_Oscar.pdf
https://papers.nips.cc/paper_files/paper/2017/file/393c55aea738548df743a186d15f3bef-Paper.pdf
http://www1.ece.neu.edu/~erdogmus/publications/C034_ESANN2003_Accelerating_Oscar.pdf
https://www.jmlr.org/papers/volume7/castillo06a/castillo06a.pdf
"""
import random

# TODO
#   1. Recursive-Least-Squares
#   (the good tutorial)
#   http://pfister.ee.duke.edu/courses/ece586/ex_proj_2008.pdf
#   https://faculty.sites.iastate.edu/jia/files/inline-files/recursive-least-squares.pdf
#   1.1 Dynamic Linear Regression
#   https://openreview.net/pdf?id=zBhwgP7kt4
#   2. IRLS https://en.wikipedia.org/wiki/Iteratively_reweighted_least_squares
#   https://anilkeshwani.github.io/files/iterative-reweighted-least-squares-12.pdf
#   https://arxiv.org/pdf/1411.5057.pdf
#   http://users.stat.umn.edu/~sandy/courses/8053/handouts/robust.pdf (section 2.1)
#   https://stats.stackexchange.com/questions/36250/definition-and-convergence-of-iteratively-reweighted-least-squares
#   https://mediatum.ub.tum.de/doc/1401694/document.pdf
#

"""
Refs
RBFN vs NN
https://www.lsoptsupport.com/faqs/algorithms/what-are-the-fundamental-differences-between-rbf-and-ffnn

"""
import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from examples.torch_rbf import basis_func_dict

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
##
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true', default=True)
parser.add_argument('--opt-method', type=str, choices=['gd', 'lstsq-gd'], required=True)
parser.add_argument('--lstsq-itr', type=int, default=100)
# parser.add_argument('--model', type=str, choices=['nn', 'rbfn'], required=True)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

# device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch():
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
                     'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
                     pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class RBF_ode(nn.Module):
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

    def __init__(self, in_features_1, in_features_2, n_centres, basis_func, device):
        super(RBF_ode, self).__init__()
        self.in_features_1 = in_features_1
        self.in_features_2 = in_features_2
        self.n_centers = n_centres
        self.centres = nn.Parameter(torch.Tensor(n_centres, in_features_1 * in_features_2).to(device))
        self.log_sigmas = nn.Parameter(torch.Tensor(n_centres).to(device))
        self.basis_func = basis_func
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.centres, mean=0, std=1)
        nn.init.constant_(self.log_sigmas, val=0)

    def forward(self, input):
        size = (input.size(0), self.n_centers, self.in_features_1 * self.in_features_2)
        x = input.expand(size)
        c = self.centres.expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)


class RBFN_ode(torch.nn.Module):
    # https://en.wikipedia.org/wiki/Radial_basis_function_network
    def __init__(self, in_dim_1, in_dim_2, out_dim, n_centres, basis_fn_str, device):
        super().__init__()
        self.basis_fn_str = basis_fn_str
        self.n_centres = n_centres
        self.out_dim = out_dim
        self.in_dim_1 = in_dim_1
        self.in_dim_2 = in_dim_2
        basis_fn = basis_func_dict()[basis_fn_str]
        self.rbf_module = RBF_ode(in_features_1=in_dim_1, in_features_2=in_dim_2, n_centres=n_centres,
                                  basis_func=basis_fn, device=device)
        # rbf inited by its own reset fn
        # self.input_norm_module = \
        #     torch.nn.BatchNorm1d(num_features=in_dim, affine=True) if input_batch_norm \
        #         else torch.nn.Identity()
        self.linear_module = torch.nn.Linear(in_features=n_centres, out_features=out_dim).to(device)
        # TODO revisit theory for batch-norm
        #   ref : https://machinelearningmastery.com/how-to-improve-neural-network-stability-and-modeling-performance-with-data-scaling/
        #   ref : https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
        #   ref-paper : https://arxiv.org/abs/1502.03167
        #   Note : batch-norm layer is before the non-linearity
        self.net = torch.nn.Sequential(self.rbf_module, self.linear_module)
        # init
        torch.nn.init.normal_(self.linear_module.weight, mean=0, std=0.01)
        torch.nn.init.constant_(self.linear_module.bias, val=0)
        # get numel learnable
        self.numel_learnable = 0
        param_list = list(self.named_parameters())
        for name, param in param_list:
            self.numel_learnable += torch.numel(param)

    def forward(self, t: float, y: torch.Tensor):
        y_hat = self.net(y).unsqueeze(1)
        return y_hat

    def forward2(self, t, y):
        y_hat = self.rbf_module(y).unsqueeze(1)
        return y_hat

    def __str__(self):
        return f"\n***\n" \
               f"RBFN\n" \
               f"in_dim={self.in_dim}\nn_centres={self.n_centres}\n" \
               f"out_dim={self.out_dim}\nbasis_fn={self.basis_fn_str}\n" \
               f"numel_learnable={self.numel_learnable}" \
               f"\n***\n"


class NeuralNetOdeFunc(nn.Module):

    def __init__(self, input_dim: int, out_dim: int, hidden_dim: int):
        super(NeuralNetOdeFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, out_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y_hat = self.net(y**3)
        return y_hat


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


if __name__ == '__main__':
    print(f'args = \n{args}')
    # Params
    lr = 1e-3
    ##
    ii = 0
    # if args.model == 'rbfn':
    #     learnable_ode_func = RBFN_ode(in_dim_1=1, in_dim_2=2, out_dim=2, n_centres=50, basis_fn_str="gaussian",
    #                                   device=device)
    # elif args.model == 'nn':
    #     learnable_ode_func = NeuralNetOdeFunc(input_dim=2, out_dim=2, hidden_dim=100).to(device)
    nn_model = NeuralNetOdeFunc(input_dim=2, out_dim=2, hidden_dim=5).to(device)
    rbfn_model = RBFN_ode(in_dim_1=1, in_dim_2=2, out_dim=2, n_centres=200, basis_fn_str="gaussian",
                          device=device)

    n_scalar_params = 0
    for name, param in nn_model.named_parameters():
        n_scalar_params += param.numel()
    for name, param in rbfn_model.named_parameters():
        n_scalar_params += param.numel()
    print(f'nn-rbfn numscaler = {n_scalar_params}')
    # for name, param in learnable_ode_func.named_parameters():
    #     n_scalar_params += param.numel()
    # print(f'learnable_ode_func = {type(learnable_ode_func).__name__} \n'
    #       f'# of scalar params = {n_scalar_params}')
    # params = list(nn_model.parameters())
    # params.extend(list(rbfn_model.parameters()))
    optimizer_rbfn = optim.RMSprop(rbfn_model.parameters(), lr=1e-3)
    optimizer_nn = optim.RMSprop(nn_model.parameters(), lr=1e-3)
    # optimizer = optim.Adam(learnable_ode_func.parameters(), lr=lr)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    W_lstsq_prev = None
    b_lstsq_prev = None
    for itr in range(1, args.niters + 1):
        # forward
        delta_t_mean = 0.1
        delta_t_half_window = 0.01
        delta_t = torch.distributions. \
            Uniform(delta_t_mean - delta_t_half_window, delta_t_mean + delta_t_half_window).sample()
        batch_y0, batch_t, batch_y = get_batch()
        batch_y = batch_y[-1]
        # first segment of het. model
        tN_minus = batch_t[-1] - delta_t
        batch_t_minus = torch.tensor([batch_t[0], tN_minus])
        pred_y_minus = odeint(nn_model, batch_y0, batch_t_minus).to(device)
        pred_y_minus = pred_y_minus[-1]
        # second segment of het. model
        dydt = rbfn_model(tN_minus, pred_y_minus)
        pred_y = pred_y_minus + dydt * delta_t
        loss = torch.mean(torch.abs(pred_y - batch_y))

        # lstsq
        if args.opt_method == 'lstsq-gd' and itr > args.lstsq_itr:
            with torch.no_grad():
                # save current model state ( got by GD)
                W_gd = rbfn_model.state_dict()['linear_module.weight']
                b_gd = rbfn_model.state_dict()['linear_module.bias']
                # start lstsq calc.
                pred_y_minus_2 = odeint(nn_model, batch_y0, batch_t_minus).to(device)[-1]
                xx = rbfn_model.forward2(tN_minus, pred_y_minus_2).squeeze()
                bsize = xx.size()[0]
                xx = torch.cat([xx, torch.ones(bsize, device=device).view(-1, 1)], dim=1)
                yy = ((batch_y - pred_y_minus_2) / delta_t).squeeze()
                Wnb = torch.linalg.lstsq(xx, yy).solution.T
                W_dim = Wnb.size()[1] - 1
                b_lstsq = Wnb[:, -1]
                W_lstsq = Wnb[:, :(W_dim)]
                alpha = 0.9
                if W_lstsq_prev is not None:
                    W_lstsq_new = alpha * W_lstsq + (1 - alpha) * W_lstsq_prev
                    b_lstsq_new = alpha * b_lstsq + (1 - alpha) * b_lstsq_prev
                else:
                    W_lstsq_new = W_lstsq
                    b_lstsq_new = b_lstsq

                W_lstsq_prev = W_lstsq
                b_lstsq_prev = b_lstsq

                rbfn_model.state_dict()['linear_module.weight'].data.copy_(W_lstsq_new.data)
                rbfn_model.state_dict()['linear_module.bias'].data.copy_(b_lstsq_new.data)
                dydt_lstsq = rbfn_model(tN_minus, pred_y_minus)
                pred_y_lstsq = pred_y_minus + dydt_lstsq * delta_t
                loss_lstsq = torch.mean(torch.abs(pred_y_lstsq - batch_y))
                loss_lstsq.requires_grad = True
                if loss_lstsq.item() < loss.item():
                    pass  # loss = loss_lstsq
        optimizer_nn.zero_grad()
        loss.backward()
        optimizer_nn.step()
        if itr <= args.lstsq_itr:
            optimizer_rbfn.zero_grad()
            optimizer_rbfn.step()
        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        if itr % args.test_freq == 0:
            with torch.no_grad():
                # fixme, fix the odeint call for heterogeneous model
                # pred_y = odeint(nn_model, true_y0, t)
                # loss = torch.mean(torch.abs(pred_y - true_y))
                # print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if args.opt_method == 'lstsq-gd' and itr > args.lstsq_itr:
                    opt_method_flag = "lstsq-gd"
                else:
                    opt_method_flag = "gd"
                print(
                    'Opt-Method : {} - Iter {:04d} | Running Loss {:.6f}'
                    .format(opt_method_flag, itr, loss_meter.avg))
                visualize(true_y, pred_y, nn_model, ii)
                ii += 1

        end = time.time()
