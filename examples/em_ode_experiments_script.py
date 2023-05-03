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
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', default=False, action='store_true')
args = parser.parse_args()


# FIXME , find a better way to activate adjoint method
# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)
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
    pass


with torch.no_grad():
    true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    # true_y = odeint(LambdaFVDP(), true_y0, t, method='dopri5')


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


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()
        hidden_dim = 200
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
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
        T = batch_t[i]
        T_1 = batch_t[i - 1]
        delta_t_T_1 = T - T_1
        with torch.no_grad():
            traj = odeint(func, batch_y0, torch.tensor([t0, T])).to(device)
            zT_1 = traj[-1]
        dzdt = func(T_1, zT_1)
        zT = zT_1 + dzdt * delta_t_T_1
        pred_y_list.append(zT)
    pred_y_tensor = torch.stack(pred_y_list, dim=0)
    return pred_y_tensor


# TODO
#   1. quick experiment with TT-RBF
#   2. add configs to select ode_forward and true_y0
#   3. epochs plots
#   4. document work
#   5. test with hidden_dim 50,100,200
if __name__ == '__main__':

    ii = 0

    func = ODEFunc().to(device)

    # optimizer = optim.RMSprop(func.parameters(), lr=1e-6)
    # optimizer = optim.SGD(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(func.parameters(), lr=1e-4)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)

    loss_meter = RunningAverageMeter(0.97)
    for itr in range(1, args.niters + 1):  # itr = epoch
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        repeats = 1  # for em opt only
        for i in range(repeats):
            # pred_y = odeint(func, batch_y0, batch_t).to(device)
            pred_y = em_euler_ode_forward(func, batch_y0, batch_t)
            # batch_y = batch_y[-1]  # fixme consider only the last time-point
            # pred_y = pred_y[-1]
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
            optimizer.step()

            time_meter.update(time.time() - end)
            loss_meter.update(loss.item())

            if itr % args.test_freq == 0:
                with torch.no_grad():
                    # pred_y = odeint(func, true_y0, t)
                    # loss = torch.mean(torch.abs(pred_y - true_y))
                    # print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                    print(f'epoch = {itr} - loss_meter_avg = {loss_meter.avg}')
                    visualize(true_y, pred_y, func, ii)
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