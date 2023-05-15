"""
The idea of hybrid ALS-GD-Adjoint for Neural-ODE
http://www1.ece.neu.edu/~erdogmus/publications/C041_ICANN2003_LinearLSlearning_Oscar.pdf
https://papers.nips.cc/paper_files/paper/2017/file/393c55aea738548df743a186d15f3bef-Paper.pdf
http://www1.ece.neu.edu/~erdogmus/publications/C034_ESANN2003_Accelerating_Oscar.pdf
https://www.jmlr.org/papers/volume7/castillo06a/castillo06a.pdf
"""

# TODO
#   1. read the two papers
#   - http://www1.ece.neu.edu/~erdogmus/publications/C041_ICANN2003_LinearLSlearning_Oscar.pdf
#   - https://www.jmlr.org/papers/volume7/castillo06a/castillo06a.pdf
#   2- quick idea of paper https://papers.nips.cc/paper_files/paper/2017/file/393c55aea738548df743a186d15f3bef-Paper.pdf
#   3 -Finalize the EM-code for ode and documentation
#       3.1 - finalize the update method
#       3.2 - Apply Dopri step (Explicit RK Step)
#       https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge.E2.80.93Kutta_methods
#       https://numerary.readthedocs.io/en/latest/dormand-prince-method.html
#       https://core.ac.uk/download/pdf/237206461.pdf
#   4 - visualize the loss landscape
#   https://medium.com/mlearning-ai/visualising-the-loss-landscape-3a7bfa1c6fdf
#   5 - ALS-only neural ODE coding starting
#   6- Hybridize ALS-Adjoint based on Last-Layer- LS idea
#   7- make last step as RK45 or dopri5

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

from examples.torch_rbf import basis_func_dict, RBF

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true',default=True)
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

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


class RBFN(torch.nn.Module):
    # https://en.wikipedia.org/wiki/Radial_basis_function_network
    def __init__(self, in_dim_1, in_dim_2, out_dim, n_centres, basis_fn_str, device):
        super().__init__()
        self.basis_fn_str = basis_fn_str
        self.n_centres = n_centres
        self.out_dim = out_dim
        self.in_dim_1 = in_dim_1
        self.in_dim_2 = in_dim_2
        basis_fn = basis_func_dict()[basis_fn_str]
        self.rbf_module = RBF(in_features_1=in_dim_1, in_features_2=in_dim_2, n_centres=n_centres, basis_func=basis_fn,
                              device=device)
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
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y_hat = self.net(y)
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
    # Params
    lr = 1e-3
    ##
    ii = 0

    # learnable_ode_func = NeuralNetOdeFunc(input_dim=2,out_dim=2,hidden_dim=100).to(device)
    learnable_ode_func = RBFN(in_dim_1=1, in_dim_2=2, out_dim=2, n_centres=50, basis_fn_str="gaussian", device=device)
    n_scalar_params = 0
    param_list = list(learnable_ode_func.parameters())
    for param in param_list:
        n_scalar_params += param.numel()
    print(f'learnable_ode_func = {type(learnable_ode_func).__name__} \n'
          f'# of scalar params = {n_scalar_params}')
    optimizer = optim.RMSprop(learnable_ode_func.parameters(), lr=1e-3)
    # optimizer = optim.Adam(learnable_ode_func.parameters(), lr=lr)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        # forward
        batch_y0, batch_t, batch_y = get_batch()
        pred_y = odeint(learnable_ode_func, batch_y0, batch_t).to(device)
        # fixme, focus on last time-point
        batch_y = batch_y[-1]
        pred_y = pred_y[-1]
        loss = torch.mean(torch.abs(pred_y - batch_y))
        # fixme , just a dummy to test manual ode func update
        #   https://discuss.pytorch.org/t/updatation-of-parameters-without-using-optimizer-step/34244/3
        #   https://discuss.pytorch.org/t/what-is-the-recommended-way-to-re-assign-update-values-in-a-variable-or-tensor/6125/12
        with torch.no_grad():
            # TODO place holder for LS
            for p in learnable_ode_func.parameters():
                # TODO
                #   https://discuss.pytorch.org/t/what-is-the-recommended-way-to-re-assign-update-values-in-a-variable-or-tensor/6125/13
                #   look deeper
                p.data.copy_(p.data)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(learnable_ode_func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                print('Iter {:04d} | Running Loss {:.6f}'.format(itr, loss_meter.avg))
                visualize(true_y, pred_y, learnable_ode_func, ii)
                ii += 1

        end = time.time()
