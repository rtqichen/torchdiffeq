#!/usr/bin/env python3

import argparse
import os
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchdiffeq import odeint, odeint_event

from bouncing_ball import BouncingBallExample


class HamiltonianDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.dvel = nn.Linear(1, 1)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def forward(self, t, state):
        pos, vel, *rest = state
        dpos = vel
        dvel = torch.tanh(self.dvel(torch.zeros_like(vel))) * self.scale
        return (dpos, dvel, *[torch.zeros_like(r) for r in rest])


class EventFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.radius = nn.Parameter(torch.rand(1))

    def parameters(self):
        return [self.radius]

    def forward(self, t, state):
        # IMPORTANT: event computation must use variables from the state.
        pos, _, radius = state
        return pos - radius.reshape_as(pos) ** 2


class InstantaneousStateChange(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1, 1)

    def forward(self, t, state):
        pos, vel, *rest = state
        vel = -torch.sigmoid(self.net(torch.ones_like(vel))) * vel
        return (pos, vel, *rest)


class NeuralPhysics(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_pos = nn.Parameter(torch.tensor([10.0]))
        self.initial_vel = nn.Parameter(torch.tensor([0.0]))
        self.dynamics_fn = HamiltonianDynamics()
        self.event_fn = EventFn()
        self.inst_update = InstantaneousStateChange()

    def simulate(self, times):

        t0 = torch.tensor([0.0]).to(times)

        # Add a terminal time to the event function.
        def event_fn(t, state):
            if t > times[-1] + 1e-7:
                return torch.zeros_like(t)
            event_fval = self.event_fn(t, state)
            return event_fval

        # IMPORTANT: for gradients of odeint_event to be computed, parameters of the event function
        # must appear in the state in the current implementation.
        state = (self.initial_pos, self.initial_vel, *self.event_fn.parameters())

        event_times = []

        trajectory = [state[0][None]]

        n_events = 0
        max_events = 20

        while t0 < times[-1] and n_events < max_events:
            last = n_events == max_events - 1

            if not last:
                event_t, solution = odeint_event(
                    self.dynamics_fn,
                    state,
                    t0,
                    event_fn=event_fn,
                    atol=1e-8,
                    rtol=1e-8,
                    method="dopri5",
                )
            else:
                event_t = times[-1]

            interval_ts = times[times > t0]
            interval_ts = interval_ts[interval_ts <= event_t]
            interval_ts = torch.cat([t0.reshape(-1), interval_ts.reshape(-1)])

            solution_ = odeint(
                self.dynamics_fn, state, interval_ts, atol=1e-8, rtol=1e-8
            )
            traj_ = solution_[0][1:]  # [0] for position; [1:] to remove intial state.
            trajectory.append(traj_)

            if event_t < times[-1]:
                state = tuple(s[-1] for s in solution)

                # update velocity instantaneously.
                state = self.inst_update(event_t, state)

                # advance the position a little bit to avoid re-triggering the event fn.
                pos, *rest = state
                pos = pos + 1e-7 * self.dynamics_fn(event_t, state)[0]
                state = pos, *rest

            event_times.append(event_t)
            t0 = event_t

            n_events += 1

            # print(event_t.item(), state[0].item(), state[1].item(), self.event_fn.mod(pos).item())

        trajectory = torch.cat(trajectory, dim=0).reshape(-1)
        return trajectory, event_times


class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


class NeuralODE(nn.Module):
    def __init__(self, aug_dim=2):
        super().__init__()
        self.initial_pos = nn.Parameter(torch.tensor([10.0]))
        self.initial_aug = nn.Parameter(torch.zeros(aug_dim))
        self.odefunc = mlp(
            input_dim=1 + aug_dim,
            hidden_dim=64,
            output_dim=1 + aug_dim,
            hidden_depth=2,
            act=Sine,
        )

        def init(m):
            if isinstance(m, nn.Linear):
                std = 1.0 / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-2.0 * std, 2.0 * std)
                m.bias.data.zero_()

        self.odefunc.apply(init)

    def forward(self, t, state):
        return self.odefunc(state)

    def simulate(self, times):
        x0 = torch.cat([self.initial_pos, self.initial_aug]).reshape(-1)
        solution = odeint(self, x0, times, atol=1e-8, rtol=1e-8, method="dopri5")
        trajectory = solution[:, 0]
        return trajectory, []


def mlp(input_dim, hidden_dim, output_dim, hidden_depth, output_mod=None, act=nn.ReLU):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), act()]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), act()]
        mods.append(nn.Linear(hidden_dim, output_dim))
    if output_mod is not None:
        mods.append(output_mod)
    trunk = nn.Sequential(*mods)
    return trunk


def cosine_decay(learning_rate, global_step, decay_steps, alpha=0.0):
    global_step = min(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * global_step / decay_steps))
    decayed = (1 - alpha) * cosine_decay + alpha
    return learning_rate * decayed


def learning_rate_schedule(
    global_step, warmup_steps, base_learning_rate, lr_scaling, train_steps
):
    warmup_steps = int(round(warmup_steps))
    scaled_lr = base_learning_rate * lr_scaling
    if warmup_steps:
        learning_rate = global_step / warmup_steps * scaled_lr
    else:
        learning_rate = scaled_lr

    if global_step < warmup_steps:
        learning_rate = learning_rate
    else:
        learning_rate = cosine_decay(
            scaled_lr, global_step - warmup_steps, train_steps - warmup_steps
        )
    return learning_rate


def set_learning_rate(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_lr", type=float, default=0.1)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--no_events", action="store_true")
    parser.add_argument("--save", type=str, default="figs")
    args = parser.parse_args()

    torch.manual_seed(0)

    torch.set_default_dtype(torch.float64)

    with torch.no_grad():
        system = BouncingBallExample()
        obs_times, gt_trajectory, _, _ = system.simulate(nbounces=4)

    obs_times = obs_times[:300]
    gt_trajectory = gt_trajectory[:300]

    if args.no_events:
        model = NeuralODE()
    else:
        model = NeuralPhysics()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)

    decay = 1.0

    model.train()
    for itr in range(args.num_iterations):
        optimizer.zero_grad()
        trajectory, event_times = model.simulate(obs_times)
        weights = decay**obs_times
        loss = (
            ((trajectory - gt_trajectory) / (gt_trajectory + 1e-3))
            .abs()
            .mul(weights)
            .mean()
        )
        loss.backward()

        lr = learning_rate_schedule(itr, 0, args.base_lr, 1.0, args.num_iterations)
        set_learning_rate(optimizer, lr)
        optimizer.step()

        if itr % 10 == 0:
            print(itr, loss.item(), len(event_times))

        if itr % 10 == 0:
            plt.figure()
            plt.plot(
                obs_times.detach().cpu().numpy(),
                gt_trajectory.detach().cpu().numpy(),
                label="Target",
            )
            plt.plot(
                obs_times.detach().cpu().numpy(),
                trajectory.detach().cpu().numpy(),
                label="Learned",
            )
            plt.tight_layout()
            os.makedirs(args.save, exist_ok=True)
            plt.savefig(f"{args.save}/{itr:05d}.png")
            plt.close()

        if (itr + 1) % 100 == 0:
            torch.save(
                {
                    "state_dict": model.state_dict(),
                },
                f"{args.save}/model.pt",
            )

        del trajectory, loss
