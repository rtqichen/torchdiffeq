import torch
from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func


class Euler(FixedGridODESolver):
    order = 1

    def __init__(self, eps=0., **kwargs):
        super(Euler, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_func(self, func, t, dt, y):
        f0 = func(t + self.eps, y)
        return dt * f0, f0


class Midpoint(FixedGridODESolver):
    order = 2

    def __init__(self, eps=0., **kwargs):
        super(Midpoint, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_func(self, func, t, dt, y):
        half_dt = 0.5 * dt
        f0 = func(t + self.eps, y)
        y_mid = y + f0 * half_dt
        return dt * func(t + half_dt, y_mid), f0


class RK4(FixedGridODESolver):
    order = 4

    def __init__(self, eps=0., **kwargs):
        super(RK4, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_func(self, func, t, dt, y):
        f0 = func(t + self.eps, y)
        return rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y, k1=f0), f0
