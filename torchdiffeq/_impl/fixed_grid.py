from .solvers import FixedGridODESolver
from . import rk_common


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t, dt, y):
        return dt * func(t + self.eps, y)


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t, dt, y):
        half_dt = 0.5 * dt
        y_mid = y + func(t + self.eps, y) * half_dt
        return dt * func(t + half_dt, y_mid)


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t, dt, y):
        return rk_common.rk4_alt_step_func(func, t + self.eps, dt - 2 * self.eps, y)
