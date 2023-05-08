import logging
import os
import argparse
import random
import time
from typing import Tuple

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=10000)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# fixme disable adjoint, focus on bp-gd
# parser.add_argument('--adjoint', default=False,action='store_true')
args = parser.parse_args()

# fixme
# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
from torchdiffeq import odeint

# set random seed
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
###
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)


class Lambda(nn.Module):

    def forward(self, t, y):
        return torch.mm(y ** 3, true_A)


with torch.no_grad():
    true_y_trajectory = odeint(Lambda(), true_y0, t, method='dopri5')


def get_batch(true_y_trajectory: torch.Tensor, batch_time: int, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y_trajectory[s]  # (M, D)
    batch_t = t[:batch_time]  # (T)
    batch_ytN_true = torch.stack([true_y_trajectory[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_ytN_true.to(device)


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

        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
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


def total_trajectory_odeint_opt_block(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
                                      batch_ytN_true: torch.Tensor,
                                      batch_t: torch.Tensor, optimizer: torch.optim.Optimizer,
                                      loss_meter: RunningAverageMeter, time_meter: RunningAverageMeter,
                                      start_timestamp: datetime) -> None:
    # typical training steps
    optimizer.zero_grad()
    # fixme, I am doing it pairwaise ( t0->ti where i = 1 ... N-1 ) to be comparable with EM-ODE
    #   it is not the most efficient way to do it.
    t0 = batch_t[0]
    N = len(batch_t)
    for i in range(1, N):
        ti = batch_t[i]
        batch_t_pair = torch.tensor([t0, ti]).to(t0.get_device())
        # now I am interested only in the final time point prediction, not the trajectory
        batch_ytN_pred = odeint(func=learnable_ode_func, y0=batch_y0, t=batch_t_pair)[-1]
        batch_ytN_true_i = batch_ytN_true[i]
        loss_pred = torch.mean(torch.abs(batch_ytN_pred - batch_ytN_true_i))
        loss_pred.backward()
        optimizer.step()
        # update training loss and running time stats
        time_meter.update((datetime.now() - start_timestamp).seconds)
        loss_meter.update(loss_pred.item())


def em_odeint_opt_block_v2(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
                           batch_ytN_true: torch.Tensor,
                           batch_t: torch.Tensor, optimizer: torch.optim.Optimizer,
                           loss_meter: RunningAverageMeter, time_meter: RunningAverageMeter,
                           start_timestamp: datetime) -> None:
    # typical training steps
    optimizer.zero_grad()
    # fixme, I am doing it pairwise ( t0->ti where i = 1 ... N-1 ) to be comparable with EM-ODE
    #   it is not the most efficient way to do it.
    t0 = batch_t[0]
    zt0 = batch_y0
    N = len(batch_t)
    b = 0.001
    a = 0.002
    for i in range(4, N):
        ti = batch_t[i]
        zti = batch_ytN_true[i]
        tj = torch.distributions.Uniform(low=t0 + a, high=ti - a).sample()
        tj_plus = tj + b
        tj_minus = tj - b
        # E1
        with torch.no_grad():
            ztj_minus = odeint(func=learnable_ode_func, y0=zt0,
                               t=torch.tensor([t0, tj_minus]).to(t0.get_device()))
            ztj_plus = odeint(func=learnable_ode_func, y0=zti,
                              t=torch.tensor([ti, tj_plus]).to(t0.get_device()))
        assert ztj_minus.requires_grad == False
        assert ztj_plus.requires_grad == False
        ztj_plus_hat = ztj_minus + learnable_ode_func(tj_minus, ztj_minus) * 2 * b
        pred_loss = torch.mean(torch.abs(ztj_plus_hat - ztj_plus))
        # M1
        # pred_loss.backward()
        # optimizer.step()

        # E2
        ti_minus = ti - b
        with torch.no_grad():
            zti_minus = odeint(func=learnable_ode_func, y0=zt0,
                               t=torch.tensor([t0, ti_minus]).to(t0.get_device()))
        assert zti_minus.requires_grad == False
        zti_hat = zti_minus + learnable_ode_func(ti_minus, zti_minus) * b
        pred_loss = torch.mean(torch.abs(zti_hat - zti))
        pred_loss.backward()
        optimizer.step()
        loss_meter.update(pred_loss.item())
        time_meter.update((datetime.now() - start_timestamp).seconds)


def em_odeint_opt_block_v1(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
                           batch_ytN_true: torch.Tensor,
                           batch_t: torch.Tensor, optimizer: torch.optim.Optimizer,
                           loss_meter: RunningAverageMeter, time_meter: RunningAverageMeter,
                           start_timestamp: datetime) -> None:
    # typical training steps
    optimizer.zero_grad()
    # fixme, I am doing it pairwise ( t0->ti where i = 1 ... N-1 ) to be comparable with EM-ODE
    #   it is not the most efficient way to do it.
    t0 = batch_t[0]
    N = len(batch_t)
    slack = 0.05
    for i in range(1, N):
        zt0 = batch_y0
        ztN = batch_ytN_true[i]
        ti = batch_t[i]
        # E1
        ti_minus = ti - slack

        batch_t_pair = torch.tensor([t0, ti_minus]).to(t0.get_device())
        with torch.no_grad():
            ztN_minus = odeint(func=learnable_ode_func, y0=zt0, t=batch_t_pair)[-1]
        assert ztN_minus.requires_grad == False
        ztN_hat = ztN_minus + learnable_ode_func(ti_minus, ztN_minus) * slack
        # M1
        pred_loss = torch.mean(torch.abs(ztN - ztN_hat))
        pred_loss.backward()
        optimizer.step()
        loss_meter.update(pred_loss.item())

        # E2
        t0_plus = t0 + slack
        batch_t_pair = torch.tensor([ti, t0_plus]).to(t0.get_device())
        with torch.no_grad():
            zt0_plus = odeint(func=learnable_ode_func, y0=ztN, t=batch_t_pair)
        assert zt0_plus.requires_grad == False
        zt0_hat = zt0_plus - learnable_ode_func(t0_plus, zt0_plus) * slack
        pred_loss = torch.mean(torch.abs(zt0 - zt0_hat))
        # M2
        pred_loss.backward()
        optimizer.step()
        time_meter.update((datetime.now() - start_timestamp).seconds)


def get_ode_opt_block_fn(opt_method: str):
    if opt_method == 'bp':
        return total_trajectory_odeint_opt_block
    elif opt_method == 'em1':
        return em_odeint_opt_block_v1
    elif opt_method == 'em2':
        return em_odeint_opt_block_v2
    else:
        raise ValueError(f"Unknown ode-opt-method = {opt_method}")


# TODO Debug
if __name__ == '__main__':
    # params
    ode_opt_method = "em1"
    #
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    #
    ii = 0

    func = ODEFunc().to(device)

    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(func.parameters(), lr=1e-4)

    start_time = datetime.now()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    #
    ode_opt_block_fn = get_ode_opt_block_fn(opt_method=ode_opt_method)
    logger.info(f'Using ode-opt-block : {ode_opt_block_fn.__name__}')
    logger.info(f'Using optimizer = {optimizer}')
    for epoch in range(1, args.epochs + 1):
        batch_y0, batch_t, batch_ytN_true = \
            get_batch(true_y_trajectory=true_y_trajectory,
                      batch_time=args.batch_time, device=device)
        # start of opt-block
        ode_opt_block_fn(learnable_ode_func=func, batch_y0=batch_y0,
                         batch_ytN_true=batch_ytN_true, batch_t=batch_t,
                         optimizer=optimizer, loss_meter=loss_meter,
                         time_meter=time_meter, start_timestamp=start_time)
        # end of opt block

        if epoch % args.test_freq == 0:
            with torch.no_grad():
                y_trajectory_pred_test = odeint(func, true_y0, t)
                loss_to_report = torch.mean(torch.abs(y_trajectory_pred_test - true_y_trajectory))
                logger.info('epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_to_report.item()))
                logger.info('epoch {:04d} | Running Loss {:.6f}'.format(epoch, loss_meter.avg))
                logger.info('---')
                # visualize(true_y, pred_y, func, ii)
                ii += 1

        # end = time.time()
