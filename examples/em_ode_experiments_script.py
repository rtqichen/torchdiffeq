import os
import argparse
import time
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
###
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=100000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--nn-hidden-dim', type=int, default=100)  # for NN-ODE
# FIXME , now I disable adjoint completely to make sure the opt in this script is based on Gradient Descent Only
parser.add_argument('--adjoint', default=False, action='store_true')

# FIXME , find a better way to activate adjoint method
# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
from torchdiffeq import odeint

"""
Notation : 
M : is the size of the data from sampling perspective 
N : is the size of the data from time perspective 
D : is the dimension of the input/output/latent variable
T : is batch size from temporal perspective 
"""


class Constants:
    def __init__(self, device):
        self.true_ode_method = 'dopri5'
        self.true_t0 = 0.0
        self.true_t_T = 25.0
        self.true_y0 = torch.tensor([[2., 0.]]).to(device)
        self.true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class LambdaAmulty3(nn.Module):
    def __init__(self, true_A: torch.Tensor, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_A = true_A

    def forward(self, t, y):
        return torch.mm(y ** 3, self.true_A)
        # vdp
        mio = 1.0


class LambdaFVDP(nn.Module):
    def __init__(self, *args, **kwargs):
        # https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
        self.mio = 1.0  # 8.53
        self.a = 1.2  # 1.2
        self.omega = 2.0 * torch.pi / 10
        super().__init__(*args, **kwargs)

    def forward(self, t, y):
        y1 = y[:, 0]
        y2 = y[:, 1]
        y1_dot = y2.view(-1, 1)
        y2_dot = (self.mio * (1 - y1 ** 2) * y2 - y1 + self.a * torch.sin(self.omega * t)).view(-1, 1)
        y_dot = torch.cat([y1_dot, y2_dot], dim=1)
        return y_dot


class LambdaLorenz(nn.Module):
    def __init__(self, sigma, rho, beta, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.rho = rho
        self.sigma = sigma
        #
        # https://en.wikipedia.org/wiki/Lorenz_system
        self.beta = 8.0 / 3
        self.rho = 28
        self.sigma = 10

    def forward(self, t, y):
        y1 = y[0]
        y2 = y[1]
        y3 = y[2]
        #
        y1_dot = self.sigma * (y2 - y1)
        y2_dot = y1 * (self.rho - y3) - y2
        y3_dot = y1 * y2 - self.beta * y3
        y_dot = torch.cat([y1_dot, y2_dot, y3_dot], dim=1)
        return y_dot


def get_true_y(true_ode_model: torch.nn.Module, true_t0: float, true_t_T: float, true_y0: torch.Tensor, method: str,
               args):
    t = torch.linspace(true_t0, true_t_T, args.data_size).to(device)
    with torch.no_grad():
        true_y = odeint(true_ode_model, true_y0, t, method=method)
        print(f'Generating true_y of size = {true_y.size()}')
        return true_y, t


def get_batch(true_y: torch.Tensor, t: torch.Tensor, args):
    N = 100
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - N-1, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    # batch_t = t[:args.batch_time]  # (T)

    batch_t = torch.tensor([t[0], t[N]])
    # batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (M, N, D)
    batch_y = true_y[s+N]
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# if args.viz:
#     makedirs('png')
#     import matplotlib.pyplot as plt
#
#     fig = plt.figure(figsize=(12, 4), facecolor='white')
#     ax_traj = fig.add_subplot(131, frameon=False)
#     ax_phase = fig.add_subplot(132, frameon=False)
#     ax_vecfield = fig.add_subplot(133, frameon=False)
#     plt.show(block=False)


# def visualize(true_y, pred_y, odefunc, itr, t):
#     if args.viz:
#         ax_traj.cla()
#         ax_traj.set_title('Trajectories')
#         ax_traj.set_xlabel('t')
#         ax_traj.set_ylabel('x,y')
#         ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0], t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1],
#                      'g-')
#         ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--', t.cpu().numpy(),
#                      pred_y.cpu().numpy()[:, 0, 1], 'b--')
#         ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
#         ax_traj.set_ylim(-2, 2)
#         ax_traj.legend()
#
#         ax_phase.cla()
#         ax_phase.set_title('Phase Portrait')
#         ax_phase.set_xlabel('x')
#         ax_phase.set_ylabel('y')
#         ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
#         ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
#         ax_phase.set_xlim(-2, 2)
#         ax_phase.set_ylim(-2, 2)
#
#         ax_vecfield.cla()
#         ax_vecfield.set_title('Learned Vector Field')
#         ax_vecfield.set_xlabel('x')
#         ax_vecfield.set_ylabel('y')
#
#         y, x = np.mgrid[-2:2:21j, -2:2:21j]
#         dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2)).to(device)).cpu().detach().numpy()
#         mag = np.sqrt(dydt[:, 0] ** 2 + dydt[:, 1] ** 2).reshape(-1, 1)
#         dydt = (dydt / mag)
#         dydt = dydt.reshape(21, 21, 2)
#
#         ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
#         ax_vecfield.set_xlim(-2, 2)
#         ax_vecfield.set_ylim(-2, 2)
#
#         fig.tight_layout()
#         plt.savefig('png/{:03d}'.format(itr))
#         plt.draw()
#         plt.pause(0.001)


class NeuralNetOdeFunc(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim):
        super(NeuralNetOdeFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y ** 3)


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


def em_euler_ode_forward(func, batch_y0, batch_t):
    t0 = batch_t[0]
    pred_y_list = [batch_y0]
    for i in range(1, batch_t.size()[0]):
        t_T = batch_t[i]
        t_T_1 = batch_t[i - 1]
        delta_t_T = t_T - t_T_1
        with torch.no_grad():
            if i == 1:
                zT_1 = batch_y0
            else:
                traj = odeint(func, batch_y0, torch.tensor([t0, t_T_1])).to(device)
                zT_1 = traj[-1]
        assert zT_1.requires_grad == False
        zT_1_stoch = torch.distributions.Normal(loc=zT_1, scale=0.1).sample()
        dzdt = func(t_T_1, zT_1)
        assert dzdt.requires_grad == True
        zT = zT_1 + dzdt * delta_t_T  # grad only comes from the dzdt part
        pred_y_list.append(zT)
    pred_y_tensor = torch.stack(pred_y_list, dim=0)
    return pred_y_tensor


# TODO
#   0. re-debug the code with the two fw methods and 3 datasets to make sure exec path is as expected
#   1. quick experiment with TT-RBF
#   2. add configs to select ode_forward and true_y0
#   3. epochs plots
#   4. document work
#   5. test with hidden_dim 50,100,200
#   6. trace computational graph to make sure we have computation graph of the shape zT = C + dzdt*delta_t
#   7. test with lorenz systems
#   8. test with other "stiff odes" ???
if __name__ == '__main__':
    args = parser.parse_args()
    # device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
    # fixme
    device = torch.device('cpu')
    constants = Constants(device=device)
    ##
    ii = 0
    # generate true_y
    true_ode_model = LambdaFVDP()
    if isinstance(true_ode_model, (LambdaAmulty3, LambdaFVDP)):
        Dy = 2
    elif isinstance(true_ode_model, LambdaLorenz):
        Dy = 3
    else:
        raise ValueError(f'Unknown true-ode-model = {type(true_ode_model)}')
    print(f'Generating true_y using true_ode_trajectory = {type(true_ode_model).__name__}')
    true_y, true_t = get_true_y(true_ode_model=true_ode_model, true_y0=constants.true_y0, true_t0=constants.true_t0,
                                true_t_T=constants.true_t_T, method=constants.true_ode_method, args=args)
    # set forward method
    forward_fn = em_euler_ode_forward
    # forward_fn = odeint
    print(f'forward_fn = {forward_fn.__name__}')
    learnable_ode_func = NeuralNetOdeFunc(input_dim=Dy, output_dim=Dy, hidden_dim=args.nn_hidden_dim).to(device)

    # optimizer = optim.RMSprop(func.parameters(), lr=1e-6)
    # optimizer = optim.SGD(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(learnable_ode_func.parameters(), lr=1e-4)
    start_time = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)
    for epoch in range(1, args.epochs + 1):  # itr = epoch
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(true_y=true_y, t=true_t, args=args)
        # pedict via odeint or my em_euler_ode_forward
        pred_y = forward_fn(learnable_ode_func, batch_y0, batch_t)[-1]
        # batch_y = batch_y[-1]  # fixme consider only the last time-point
        # pred_y = pred_y[-1]
        loss = torch.mean(torch.abs(pred_y - batch_y))
        loss.backward()
        optimizer.step()

        time_meter.update(time.time() - start_time)
        loss_meter.update(loss.item())

        if epoch % args.test_freq == 0:
            with torch.no_grad():
                print(f'epoch = {epoch} - loss_meter_avg = {loss_meter.avg}')
                # pred_y = odeint(func, true_y0, t)
                # loss = torch.mean(torch.abs(pred_y - true_y))
                # print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                # visualize(true_y, pred_y, learnable_ode_func, ii)
                ii += 1
"""
Experiment results : 

# 1)
ode_forward = odeint
ode_true_model = dydt = A.y3
results
epoch = 950 - loss_meter_avg = 0.005418554378761679
epoch = 960 - loss_meter_avg = 0.004940435532788371
epoch = 970 - loss_meter_avg = 0.005352401817800845
epoch = 980 - loss_meter_avg = 0.005208418145054498
epoch = 990 - loss_meter_avg = 0.005534352723540507
epoch = 1000 - loss_meter_avg = 0.005642940935778827

***
# 2)
ode_forward = odeint
ode_true_model = dydt = FVDP
epoch = 970 - loss_meter_avg = 0.05246230135725554
epoch = 980 - loss_meter_avg = 0.05145772834002182
epoch = 990 - loss_meter_avg = 0.05153626088939964
epoch = 1000 - loss_meter_avg = 0.050862263443920246

***

# 3) 
ode_forward = em_euler_forward
ode_true_model = dydt = FVDP
results
epoch = 980 - loss_meter_avg = 0.05184645471956136
epoch = 990 - loss_meter_avg = 0.05193931148298325
epoch = 1000 - loss_meter_avg = 0.051310242529871264

***
# 4) 
ode_forward = em_euler_forward
ode_true_model = dydt = A.y3
epoch = 980 - loss_meter_avg = 0.008169971514184712
epoch = 990 - loss_meter_avg = 0.008515986481298773
epoch = 1000 - loss_meter_avg = 0.008567934482237062

"""
