from .misc import _handle_deprecated_kwargs
from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func


_eps_message = """`eps` has been deprecated; use `perturb` instead."""


class Euler(FixedGridODESolver):
    order = 1

    def __init__(self, perturb=False, **kwargs):
        try:
            perturb = kwargs['eps'] > 0
        except KeyError:
            pass
        _handle_deprecated_kwargs(self, kwargs, 'eps', _eps_message)
        super(Euler, self).__init__(**kwargs)
        self.perturb = perturb

    def _step_func(self, func, t0, dt, t1, y0):
        return dt * func(t0, y0, True if self.perturb else None)


class Midpoint(FixedGridODESolver):
    order = 2

    def __init__(self, perturb=False, **kwargs):
        try:
            perturb = kwargs['eps'] > 0
        except KeyError:
            pass
        _handle_deprecated_kwargs(self, kwargs, 'eps', _eps_message)
        super(Midpoint, self).__init__(**kwargs)
        self.perturb = perturb

    def _step_func(self, func, t0, dt, t1, y0):
        half_dt = 0.5 * dt
        y_mid = y0 + func(t0, y0, True if self.perturb else None) * half_dt
        return dt * func(t0 + half_dt, y_mid)


class RK4(FixedGridODESolver):
    order = 4

    def __init__(self, perturb=False, **kwargs):
        try:
            perturb = kwargs['eps'] > 0
        except KeyError:
            pass
        _handle_deprecated_kwargs(self, kwargs, 'eps', _eps_message)
        super(RK4, self).__init__(**kwargs)
        self.perturb = perturb

    def _step_func(self, func, t0, dt, t1, y0):
        return rk4_alt_step_func(func, t0, dt, t1, y0, perturb=self.perturb)
