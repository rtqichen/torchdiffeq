from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):
    order = 1

    def step_func(self, func, t, dt, y):
        return tuple(dt * f_ for f_ in func(t + self.eps, y))


class Midpoint(FixedGridODESolver):
    order = 2

    def step_func(self, func, t, dt, y):
        y_mid = tuple(y_ + f_ * dt / 2 for y_, f_ in zip(y, func(t + self.eps, y)))
        return tuple(dt * f_ for f_ in func(t + dt / 2, y_mid))


class RK4(FixedGridODESolver):
    order = 4

    def step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)
