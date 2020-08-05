import bisect
import collections
import torch
from .interp import _interp_evaluate, _interp_fit
from .misc import (_compute_error_ratio,
                   _select_initial_step,
                   _optimal_step_size)
from .solvers import AdaptiveStepsizeODESolver


_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')


_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float64 Tensor giving start of the last time step.
#     t1: scalar float64 Tensor giving end of the last time step.
#     dt: scalar float64 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.


class _UncheckedAssign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scratch, value, index):
        ctx.index = index
        scratch.data[index] = value  # sneak past the version checker
        return scratch

    @staticmethod
    def backward(ctx, grad_scratch):
        return grad_scratch, grad_scratch[ctx.index], None


def _runge_kutta_step(func, y0, f0, t0, dt, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.

    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.

    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t0 = t0.type_as(y0)
    dt = dt.type_as(y0)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=y0.dtype, device=y0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        ti = t0 + alpha_i * dt
        yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        f = func(ti, yi)
        k = _UncheckedAssign.apply(k, f, (..., i + 1))

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + k.matmul(dt * tableau.c_sol).view_as(f0)

    y1 = yi
    f1 = k[..., -1]
    y1_error = k.matmul(dt * tableau.c_error)
    return y1, f1, y1_error, k


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


def rk4_step_func(func, t, dt, y, k1=None):
    if k1 is None:
        k1 = func(t, y)
    half_dt = dt * 0.5
    k2 = func(t + half_dt, y + half_dt * k1)
    k3 = func(t + half_dt, y + half_dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t, dt, y, k1=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t, y)
    k2 = func(t + dt * _one_third, y + dt * k1 * _one_third)
    k3 = func(t + dt * _two_thirds, y + dt * (k2 - k1 * _one_third))
    k4 = func(t + dt, y + dt * (k1 - k2 + k3))
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2,
                 max_num_steps=2 ** 31 - 1, grid_points=None, eps=0., dtype=torch.float64, **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device

        self.func = lambda t, y: func(t.type_as(y), y)
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        grid_points = torch.tensor([], dtype=dtype, device=device) if grid_points is None else grid_points.to(dtype)
        self.grid_points = grid_points
        self.eps = torch.as_tensor(eps, dtype=dtype, device=device)
        self.dtype = dtype

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.mid = self.mid.to(device=device, dtype=y0.dtype)

    def _before_integrate(self, t):
        f0 = self.func(t[0], self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                              self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)
        self.next_grid_index = min(bisect.bisect(self.grid_points.tolist(), t[0]), len(self.grid_points) - 1)

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        # dtypes: self.y0.dtype (probably float32); self.dtype (probably float64)
        # used for state and timelike objects respectively.
        # Then:
        # y0.dtype == self.y0.dtype
        # f0.dtype == self.y0.dtype
        # t0.dtype == self.dtype
        # dt.dtype == self.dtype
        # for coeff in interp_coeff: coeff.dtype == self.y0.dtype


        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)

        ########################################################
        #     Make step, respecting prescribed grid points     #
        ########################################################
        on_grid = len(self.grid_points) and t0 < self.grid_points[self.next_grid_index] < t0 + dt
        if on_grid:
            dt = self.grid_points[self.next_grid_index] - t0
            eps = min(0.5 * dt, self.eps)
            dt = dt - eps
        else:
            eps = 0

        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=self.tableau)
        # dtypes:
        # y1.dtype == self.y0.dtype
        # f1.dtype == self.y0.dtype
        # y1_error.dtype == self.dtype
        # k.dtype == self.y0.dtype

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        accept_step = error_ratio <= 1
        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        t_next = t0 + dt + 2 * eps if accept_step else t0
        y_next = y1 if accept_step else y0
        if on_grid and accept_step:
            # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity we're
            # now on.
            if eps != 0:
                f1 = self.func(t_next, y_next)
            if self.next_grid_index != len(self.grid_points) - 1:
                self.next_grid_index += 1
        f_next = f1 if accept_step else f0
        interp_coeff = self._interp_fit(y0, y1, k, dt) if accept_step else interp_coeff
        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(y0)
        y_mid = y0 + k.matmul(dt * self.mid).view_as(y0)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)
