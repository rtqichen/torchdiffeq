import logging
import os
import argparse
import pickle

import time
from typing import Tuple

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=60)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--epochs', type=int, default=1e7)  # inf, make it time based
parser.add_argument('--max_time_sec', type=int, default=60)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
# fixme disable adjoint, focus on bp-gd
# parser.add_argument('--adjoint', default=False,action='store_true')
args = parser.parse_args()
"""
Related Refrences
http://www1.ece.neu.edu/~erdogmus/publications/C041_ICANN2003_LinearLSlearning_Oscar.pdf 
https://medium.com/@ngamzedeniz/how-to-avoid-local-minima-in-gradient-descent-neural-network-d0798f55e9ed 

"""
# fixme
# if args.adjoint:
#     from torchdiffeq import odeint_adjoint as odeint
# else:
from torchdiffeq import odeint

# set random seed
# SEED = 42
# torch.manual_seed(SEED)
# random.seed(SEED)
# np.random.seed(SEED)
###
# fixme focus first on cpu
device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


# device = torch.device('cpu')


class PolyOde3(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

    def forward(self, t, z):
        return torch.mm(z ** 3, self.true_A)


class Lorenz5D(nn.Module):
    # https://iopscience.iop.org/article/10.1088/2399-6528/aaa955/pdf
    pass


class Lorenz3D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sigma = 10
        self.rho = 28
        self.beta = 8.0 / 3

    def forward(self, t, y):
        y1 = y[:, 0]
        y2 = y[:, 1]
        y3 = y[:, 2]
        y1_dot = (self.sigma * (y2 - y1)).view(-1, 1)
        y2_dot = (y1 * (self.rho - y3) - y2).view(-1, 1)
        y3_dot = (y1 * y2 - self.beta * y3).view(-1, 1)
        y_dot = torch.cat(tensors=[y1_dot, y2_dot, y3_dot], dim=1)
        return y_dot


class FVDP(nn.Module):
    def __init__(self, *args, **kwargs):
        # https://www.johndcook.com/blog/2019/12/22/van-der-pol/
        super().__init__(*args, **kwargs)
        # fixme a = 0.0, no forced term
        self.a = 0.5  # 1.2
        self.omega = 2.0 * torch.pi / 10
        self.mio = 0.8  # 8.53

    def forward(self, t: float, z: torch.Tensor):
        z1 = z[:, 0]
        z2 = z[:, 1]
        z1_dot = z2.view(1, -1)
        z2_dot = (self.mio * (1 - z1 ** 2) * z2 - z1 +
                  self.a * torch.sin(self.omega * torch.tensor(t))).view(-1, 1)
        z_dot = torch.cat([z1_dot, z2_dot], dim=1)
        return z_dot


def get_true_y_trajectory(true_ode_func: torch.nn.Module, true_y0: torch.Tensor, true_t: torch.Tensor, method: str):
    return odeint(func=true_ode_func, y0=true_y0, method=method, t=true_t)


def get_batch(true_y_trajectory: torch.Tensor, true_t: torch.Tensor, batch_time: int, device: torch.device) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    s = torch.from_numpy(
        np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64), args.batch_size, replace=False))
    batch_y0 = true_y_trajectory[s]  # (M, D)
    batch_t = true_t[:batch_time]  # (T)
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


# fixme no viz for now
#
# def visualize(true_y, pred_y, odefunc, itr):
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

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int, input_power: int):
        super(NeuralNetOdeFunc, self).__init__()
        self.input_power = input_power
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t: torch.Tensor, y: torch.Tensor):
        # https://github.com/rtqichen/torchdiffeq/blob/master/examples/ode_demo.py#L128
        y = torch.pow(y, self.input_power)
        if len(y.size()) == 2:
            y_aug = torch.cat([y, t.view(1, 1)], dim=1)
        elif len(y.size()) == 3:
            b = y.size()[0]
            if len(t.size()) == 0:
                t_tensor = torch.tensor(t).repeat(b, 1, 1)
            else:
                t_tensor = t
            y_aug = torch.cat([y, t_tensor], dim=2)
        else:
            raise ValueError(f'Unsupported y.size() = {y.size()}')
        return self.net(y_aug)


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

# TODO line by line debug
def vanilla_odeint_opt_block(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
                             batch_ytN_true: torch.Tensor,
                             batch_t: torch.Tensor,
                             optimizer: torch.optim.Optimizer) -> torch.Tensor:

    # fixme, I am doing it pairwaise ( t0->ti where i = 1 ... N-1 ) to be comparable with EM-ODE
    #   it is not the most efficient way to do it.
    t0 = batch_t[0]
    N = len(batch_t)
    block_losses = []
    N_start = 40
    assert (N-N_start) > 10
    # TODO , insert block timing
    for i in range(N_start, N):
        optimizer.zero_grad()
        ti = batch_t[i]
        batch_t_pair = torch.tensor([t0, ti]).to(t0.get_device())
        # now I am interested only in the final time point prediction, not the trajectory
        batch_ytN_pred = odeint(func=learnable_ode_func, y0=batch_y0, t=batch_t_pair)[-1]
        batch_ytN_true_i = batch_ytN_true[i]
        loss = torch.mean(torch.abs(batch_ytN_pred - batch_ytN_true_i))
        block_losses.append(loss.item())
        loss.backward()
        optimizer.step()
    return torch.mean(torch.tensor(block_losses)).item()


# FIXME
# def em_odeint_opt_block_v3(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
#                            batch_ytN_true: torch.Tensor,
#                            batch_t: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
#     """
#         fixme, the version that diverges. I dunno why yet
#         """
#     # typical training steps
#     ode_solver_method = 'euler'
#     optimizer.zero_grad()
#     # fixme, I am doing it pairwise ( t0->ti where i = 1 ... N-1 ) to be comparable with EM-ODE
#     #   it is not the most efficient way to do it.
#     # t0 = batch_t[0]
#     zt0 = batch_y0
#     N = len(batch_t)
#     em3_block_losses = []
#     for i in range(1, N):
#         # ti = batch_t[i]
#         ztN = batch_ytN_true[i]
#         t_vals_forward = batch_t[:(i + 1)]
#         t_vals_backward = torch.flip(t_vals_forward, dims=[0])
#         with torch.no_grad():
#             z_trajectory_forward = odeint(func=learnable_ode_func, y0=zt0, t=t_vals_forward, method=ode_solver_method)
#             z_trajectory_backward = odeint(func=learnable_ode_func, y0=ztN, t=t_vals_backward, method=ode_solver_method)
#             ztN_hat = t_vals_forward[-1]
#             em3_block_loss = torch.mean(torch.abs(ztN_hat - ztN)).item()
#             em3_block_losses.append(em3_block_loss)
#         xx = z_trajectory_forward[:-1]
#         yy = z_trajectory_backward[1:]
#         traj_len = xx.size()[0]
#         opt_loss = None  # torch.tensor([0])
#         for j in range(traj_len):
#             tj = t_vals_forward[j]
#             tj_plus_1 = t_vals_forward[j + 1]
#             delta_t_j = tj_plus_1 - tj
#             xx_j = xx[j]
#             yy_j = yy[j]
#             yy_j_hat = xx_j + learnable_ode_func(tj, xx_j) * delta_t_j
#             if opt_loss:
#                 opt_loss += torch.sum((yy_j - yy_j_hat) ** 2)
#             else:
#                 opt_loss = torch.sum((yy_j - yy_j_hat) ** 2)
#         opt_loss.backward()
#         optimizer.step()
#
#     return torch.mean(torch.tensor(em3_block_losses)).item()

# FIXME, to remove this code, now we focus on the last-layer idea
#   this code is kept for learning only, no use
# def em_odeint_opt_block_v2(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
#                            batch_ytN_true: torch.Tensor,
#                            batch_t: torch.Tensor, optimizer: torch.optim.Optimizer,
#                            loss_meter: RunningAverageMeter, time_meter: RunningAverageMeter,
#                            start_timestamp: datetime) -> None:
#     """
#     fixme, the version that diverges. I dunno why yet
#     """
#     # typical training steps
#     optimizer.zero_grad()
#     # fixme, I am doing it pairwise ( t0->ti where i = 1 ... N-1 ) to be comparable with EM-ODE
#     #   it is not the most efficient way to do it.
#     t0 = batch_t[0]
#     zt0 = batch_y0
#     N = len(batch_t)
#     b = 0.001
#     a = 0.002
#     for i in range(4, N):
#         ti = batch_t[i]
#         zti = batch_ytN_true[i]
#         tj = torch.distributions.Uniform(low=t0 + a, high=ti - a).sample()
#         tj_plus = tj + b
#         tj_minus = tj - b
#         # E1
#         with torch.no_grad():
#             ztj_minus = odeint(func=learnable_ode_func, y0=zt0,
#                                t=torch.tensor([t0, tj_minus]).to(t0.get_device()))
#             ztj_plus = odeint(func=learnable_ode_func, y0=zti,
#                               t=torch.tensor([ti, tj_plus]).to(t0.get_device()))
#         assert ztj_minus.requires_grad == False
#         assert ztj_plus.requires_grad == False
#         ztj_plus_hat = ztj_minus + learnable_ode_func(tj_minus, ztj_minus) * 2 * b
#         pred_loss = torch.mean(torch.abs(ztj_plus_hat - ztj_plus))
#         # M1
#         # pred_loss.backward()
#         # optimizer.step()
#
#         # E2
#         ti_minus = ti - b
#         with torch.no_grad():
#             zti_minus = odeint(func=learnable_ode_func, y0=zt0,
#                                t=torch.tensor([t0, ti_minus]).to(t0.get_device()))
#         assert zti_minus.requires_grad == False
#         zti_hat = zti_minus + learnable_ode_func(ti_minus, zti_minus) * b
#         pred_loss = torch.mean(torch.abs(zti_hat - zti))
#         pred_loss.backward()
#         optimizer.step()

# TODO , line by line debug and insepction
def ls_odeint_opt_block(learnable_ode_func: torch.nn.Module, batch_y0: torch.Tensor,
                        batch_ytN_true: torch.Tensor,
                        batch_t: torch.Tensor, optimizer: torch.optim.Optimizer) -> torch.Tensor:
    # TODO , insert block timing
    """
    The version that converges
    """
    # fixme, I am doing it pairwise ( t0->ti where i = 1 ... N-1 ) to be comparable with LS-GD-ODE
    #   it is not the most efficient way to do it.
    t0 = batch_t[0]
    N = len(batch_t)
    # slack_mean = 0.1
    # slack_half_window = 0.001
    # slack = torch.distributions.Uniform(slack_mean - slack_half_window, slack_mean + slack_half_window).sample()
    slack = 0.05
    block_losses = []
    #opt_block_losses = torch.tensor(0.0).to(device)
    N_start = 40
    assert (N - N_start) >10
    iter_times = []
    for i in range(N_start, N):
        itr_start_time = time.time()
        optimizer.zero_grad()
        zt0 = batch_y0
        ztN = batch_ytN_true[i]
        ti = torch.distributions.Uniform(batch_t[i] - 0.01, batch_t[i] + 0.01).sample()
        ti_minus = ti - slack
        batch_t_pair = torch.tensor([t0, ti_minus]).to(t0.get_device())
        with torch.no_grad():
            ztN_minus = odeint(func=learnable_ode_func, y0=zt0, t=batch_t_pair)[-1]
        assert ztN_minus.requires_grad == False
        # E1
        ztN_hat = ztN_minus + learnable_ode_func(ti_minus, ztN_minus) * slack
        # loss to report per block
        block_losses.append(torch.mean(torch.abs(ztN - ztN_hat)))
        # M1
        opt_loss = torch.sum((ztN - ztN_hat) ** 2)
        opt_loss.backward()
        optimizer.step()
        # opt_block_losses += torch.sum((ztN - ztN_hat) ** 2)

        # E2
        t0_plus = t0 + slack
        batch_t_pair = torch.tensor([ti, t0_plus]).to(t0.get_device())
        with torch.no_grad():
            zt0_plus = odeint(func=learnable_ode_func, y0=ztN, t=batch_t_pair)[-1]
        assert zt0_plus.requires_grad == False
        zt0_hat = zt0_plus - learnable_ode_func(t0_plus, zt0_plus) * slack
        # M2
        opt_loss = torch.sum((zt0 - zt0_hat) ** 2)
        opt_loss.backward()
        optimizer.step()
        iter_time = time.time()-itr_start_time
        iter_times.append(iter_time)
    
    return torch.mean(torch.tensor(block_losses)).item()


def get_ode_opt_block_fn(opt_method: str):
    if opt_method == 'vanilla':
        return vanilla_odeint_opt_block
    elif opt_method == 'ls':
        return ls_odeint_opt_block
    # em2 not working
    # fixme , test em2 later or remove at all
    # elif opt_method == 'em2':
    #     return em_odeint_opt_block_v2
    # elif opt_method == 'em3':
    #     return em_odeint_opt_block_v3
    else:
        raise ValueError(f"Unknown ode-opt-method = {opt_method}")


if __name__ == '__main__':
    # params
    ode_opt_block_fn_name = "ls"
    true_ode_method = 'dopri5'
    nn_hidden_dim = 50
    init_lr = 1e-3
    min_lr = 1e-6
    scheduler_factor = 0.8
    #
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    # Generate true y trajectory
    true_t = torch.linspace(0., 25., args.data_size).to(device)
    # i) dzdt = Ay**3bat
    true_y0 = torch.tensor([[2., 0.]]).to(device)
    true_ode_func = PolyOde3()
    # ii) dzdt = fvdp
    # true_y0 = torch.tensor([[1., 0.]]).to(device)
    # true_ode_func = FVDP()
    # ii) dzdt = lorenz
    # true_y0 = torch.tensor([[1., 0., 0.0]]).to(device)
    # true_ode_func = Lorenz3D()

    true_y_trajectory = get_true_y_trajectory(true_ode_func=true_ode_func,
                                              true_y0=true_y0, true_t=true_t,
                                              method=true_ode_method)
    if isinstance(true_ode_func, (PolyOde3, FVDP)):
        D = 2
    elif isinstance(true_ode_func, Lorenz3D):
        D = 3
    else:
        raise ValueError(f'Unknown true-ode model {true_ode_func}')
    if isinstance(true_ode_func, (FVDP, Lorenz3D)):
        input_power = 1
    elif isinstance(true_ode_func, PolyOde3):
        input_power = 3
    logger.info(f'Data set = {type(true_ode_func).__name__}')
    LearnableOdeFunc = NeuralNetOdeFunc(input_dim=D, output_dim=D, hidden_dim=nn_hidden_dim,
                                        input_power=input_power).to(device)

    # optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    optimizer = optim.Adam(LearnableOdeFunc.parameters(), lr=init_lr)
    # FIXME , SGD doesn't work
    # optimizer = optim.SGD(LearnableOdeFunc.parameters(),lr=init_lr)
    scheduler = ReduceLROnPlateau(optimizer=optimizer, factor=scheduler_factor, min_lr=min_lr)
    start_time = time.time()
    time_meter = RunningAverageMeter(0.97)
    loss_meter = RunningAverageMeter(0.97)
    loss_time_tracker = []
    #
    ode_opt_block_fn = get_ode_opt_block_fn(opt_method=ode_opt_block_fn_name)
    logger.info(f'Using ode-opt-block : {ode_opt_block_fn.__name__}')
    logger.info(f'Using optimizer = {optimizer}')
    logger.info(f'arg.data_size = {args}')
    for epoch in range(1, int(args.epochs) + 1):
        batch_y0, batch_t, batch_ytN_true = \
            get_batch(true_y_trajectory=true_y_trajectory, true_t=true_t,
                      batch_time=args.batch_time, device=device)
        # start of opt-block
        block_loss = ode_opt_block_fn(learnable_ode_func=LearnableOdeFunc, batch_y0=batch_y0,
                                      batch_ytN_true=batch_ytN_true, batch_t=batch_t,
                                      optimizer=optimizer)
        loss_meter.update(block_loss)
        time_meter.update(time.time() - start_time)
        scheduler.step(block_loss)

        # end of opt block

        if epoch % args.test_freq == 0:
            with torch.no_grad():
                y_trajectory_pred_test = odeint(LearnableOdeFunc, true_y0, true_t)
                loss_to_report = torch.mean(torch.abs(y_trajectory_pred_test - true_y_trajectory))
                logger.info('epoch {:04d} | Total Loss {:.6f}'.format(epoch, loss_to_report.item()))
                logger.info('epoch {:04d} | Running Loss {:.6f}'.format(epoch, loss_meter.avg))
                logger.info('epoch {:04d} | Running time {:.6f}'.format(epoch, time_meter.avg))
                # fixme
                if time_meter.avg > args.max_time_sec:
                    break
                logger.info(f'optimizer current lr = {optimizer.param_groups[0]["lr"]}')
                loss_time_tracker.append((time.time() - start_time, loss_meter.avg))
                logger.info('---')
                # visualize(true_y, pred_y, func, ii)
                # ii += 1
    logger.info('Training Finished')
    logger.info(f'{loss_time_tracker}')
    pickle.dump(obj=loss_time_tracker,
                file=open(f"{ode_opt_block_fn_name}_loss_time_tracker_{true_ode_func.__class__.__name__}.pkl", "wb"))
    # end = time.time()
