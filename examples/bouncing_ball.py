import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchdiffeq import odeint, odeint_adjoint
from torchdiffeq import odeint_event

torch.set_default_dtype(torch.float64)


class BouncingBallExample(nn.Module):

    def __init__(self, radius=0.2, gravity=9.8, adjoint=False):
        super().__init__()
        self.gravity = nn.Parameter(torch.as_tensor([gravity]))
        self.log_radius = nn.Parameter(torch.log(torch.as_tensor([radius])))
        self.t0 = nn.Parameter(torch.tensor([0.0]))
        self.init_pos = nn.Parameter(torch.tensor([10.0]))
        self.init_vel = nn.Parameter(torch.tensor([0.0]))
        self.absorption = nn.Parameter(torch.tensor([0.2]))
        self.odeint = odeint_adjoint if adjoint else odeint

    def forward(self, t, state):
        pos, vel, log_radius = state
        dpos = vel
        dvel = -self.gravity
        return dpos, dvel, torch.zeros_like(log_radius)

    def event_fn(self, t, state):
        # positive if ball in mid-air, negative if ball within ground.
        pos, _, log_radius = state
        return pos - torch.exp(log_radius)

    def get_initial_state(self):
        state = (self.init_pos, self.init_vel, self.log_radius)
        return self.t0, state

    def state_update(self, state):
        """ Updates state based on an event (collision)."""
        pos, vel, log_radius = state
        pos = pos + 1e-7  # need to add a small eps so as not to trigger the event function immediately.
        vel = -vel * (1 - self.absorption)
        return (pos, vel, log_radius)

    def get_collision_times(self, nbounces=1):

        event_times = []

        t0, state = self.get_initial_state()

        for i in range(nbounces):
            event_t, solution = odeint_event(self, state, t0, event_fn=self.event_fn, reverse_time=False, atol=1e-8, rtol=1e-8, odeint_interface=self.odeint)
            event_times.append(event_t)

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return event_times

    def simulate(self, nbounces=1):
        event_times = self.get_collision_times(nbounces)

        # get dense path
        t0, state = self.get_initial_state()
        trajectory = [state[0][None]]
        times = [t0.reshape(-1)]
        for event_t in event_times:
            tt = torch.linspace(float(t0), float(event_t), 100)[1:-1]
            tt = torch.cat([t0.reshape(-1), tt, event_t.reshape(-1)])
            solution = odeint(self, state, tt, atol=1e-8, rtol=1e-8)

            trajectory.append(solution[0][1:])
            times.append(tt[1:])

            state = self.state_update(tuple(s[-1] for s in solution))
            t0 = event_t

        return torch.cat(times), torch.cat(trajectory, dim=0).reshape(-1)


def gradcheck(nbounces):

    system = BouncingBallExample()

    variables = {
        "init_pos": system.init_pos,
        "init_vel": system.init_vel,
        "t0": system.t0,
        "gravity": system.gravity,
        "log_radius": system.log_radius,
    }

    event_t = system.get_collision_times(nbounces)[-1]
    event_t.backward()

    analytical_grads = {}
    for name, p in system.named_parameters():
        for var in variables.keys():
            if var in name:
                analytical_grads[var] = p.grad

    eps = 1e-3

    fd_grads = {}

    for var, param in variables.items():
        orig = param.data
        param.data = orig - eps
        f_meps = system.get_collision_times(nbounces)[-1]
        param.data = orig + eps
        f_peps = system.get_collision_times(nbounces)[-1]
        param.data = orig
        fd = (f_peps - f_meps) / (2 * eps)
        fd_grads[var] = fd

    success = True
    for var in variables.keys():
        analytical = analytical_grads[var]
        fd = fd_grads[var]
        if torch.norm(analytical - fd) > 1e-4:
            success = False
            print(f"Got analytical grad {analytical.item()} for {var} param but finite difference is {fd.item()}")

    if not success:
        raise Exception("Gradient check failed.")

    print("Gradient check passed.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("nbounces", type=int, nargs="?", default=1)
    parser.add_argument("--adjoint", action="store_true")
    args = parser.parse_args()

    gradcheck(args.nbounces)

    system = BouncingBallExample()
    times, trajectory = system.simulate(nbounces=args.nbounces)
    times = times.detach().cpu().numpy()
    trajectory = trajectory.detach().cpu().numpy()

    plt.plot(times, trajectory)
    plt.savefig("bouncing_ball.png")
