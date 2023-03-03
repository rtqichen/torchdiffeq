from .solvers import FixedGridODESolver
from .rk_common import rk4_step_func, rk4_alt_step_func
from .misc import Perturb


class Euler(FixedGridODESolver):
    order = 1

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return dt * f0, f0


class Heun(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        f1 = func(t0 + dt, y0 + dt * f0)
        return half_dt * (f0 + f1), f0


class Midpoint(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        y_mid = y0 + f0 * half_dt
        return dt * func(t0 + half_dt, y_mid), f0


class Ralston(FixedGridODESolver):
    order = 2

    def _step_func(self, func, t0, dt, t1, y0):
        fourth_dt = 0.25 * dt
        double_dt = 2 * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        f1 = func(t0 + double_dt / 3, y0 + double_dt * f0 / 3)
        return fourth_dt * (f0 + 3 * f1), f0


class RK3(FixedGridODESolver):
    order = 3

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        double_dt = 2 * dt
        sixth_dt = (1 / 6) * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        f1 = func(t0 + half_dt, y0 + half_dt * f0)
        f2 = func(t0 + dt, y0 - dt * f0 + double_dt * f1)
        return sixth_dt * (f0 + 4 * f1 + f2), f0


class SSPRK3(FixedGridODESolver):
    order = 3

    def _step_func(self, func, t0, dt, t1, y0):
        fourth_dt = 0.25 * dt
        sixth_dt = (1 / 6) * dt
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        f1 = func(t0 + dt, y0 + dt * f0)
        f2 = func(t0 + fourth_dt, y0 + fourth_dt * (f0 + f1))
        return sixth_dt * (f0 + f1 + 4 * f2), f0


class RK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_alt_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0


class ClassicRK4(FixedGridODESolver):
    order = 4

    def _step_func(self, func, t0, dt, t1, y0):
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)
        return rk4_step_func(func, t0, dt, t1, y0, f0=f0, perturb=self.perturb), f0
