import bisect
import collections
import torch
from .event_handling import find_event
from .interp import _interp_evaluate, _interp_fit
from .misc import (_compute_error_ratio,
                   _select_initial_step,
                   _optimal_step_size,
                   _handle_unused_kwargs)
from .misc import Perturb
from .solvers import AdaptiveStepsizeEventODESolver, FixedGridODESolver
import warnings


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


def _runge_kutta_step(func, y0, f0, t0, dt, t1, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.
    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        t1: float64 scalar Tensor giving the end time; equal to t0 + dt. This is used (rather than t0 + dt) to ensure
            floating point accuracy when needed.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.
    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    t_dtype = y0.abs().dtype

    t0 = t0.to(t_dtype)
    dt = dt.to(t_dtype)
    t1 = t1.to(t_dtype)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=y0.dtype, device=y0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
            perturb = Perturb.PREV
        else:
            ti = t0 + alpha_i * dt
            perturb = Perturb.NONE
        yi = y0 + torch.sum(k[..., :i + 1] * (beta_i * dt), dim=-1).view_as(f0)
        f = func(ti, yi, perturb=perturb)
        k = _UncheckedAssign.apply(k, f, (..., i + 1))

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + torch.sum(k * (dt * tableau.c_sol), dim=-1).view_as(f0)

    y1 = yi
    f1 = k[..., -1]
    y1_error = torch.sum(k * (dt * tableau.c_error), dim=-1)
    return y1, f1, y1_error, k


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


def rk4_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    half_dt = dt * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t1, y0 + dt * k3, perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t0, dt, t1, y0, f0=None, perturb=False):
    """Smaller error with slightly more compute."""
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), perturb=Perturb.PREV if perturb else Perturb.NONE)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


def rk3_step_func(func, t0, dt, t1, y0, butcher_tableu=None, f0=None, perturb=False):
    """butcher_tableu should be of the form
    
    [
        [0  , 0     , 0     ,   0],
        [c_2, a_{21}, 0     ,   0],
        [c_3, a_{31}, a_{32},   0],
        [0  , b_1   , b_2   , b_3],
    ]

    https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods
    """
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)

    k2 = func(t0 + dt * butcher_tableu[1][0], y0 + dt * k1 * butcher_tableu[1][1])
    k3 = func(t0 + dt * butcher_tableu[2][0], y0 + dt * (k1 * butcher_tableu[2][1] + k2 * butcher_tableu[2][2]))
    return dt * (k1 * butcher_tableu[3][1] + k2 * butcher_tableu[3][2] + k3 * butcher_tableu[3][3])


def rk2_step_func(func, t0, dt, t1, y0, butcher_tableu=None, f0=None, perturb=False):
    """butcher_tableu should be of the form

    [
        [0  , 0     , 0     ],
        [c_2, a_{21}, 0     ],
        [0  , b_1   , b_2   ],
    ]

    https://en.wikipedia.org/wiki/List_of_Runge-Kutta_methods
    """
    k1 = f0
    if k1 is None:
        k1 = func(t0, y0, perturb=Perturb.NEXT if perturb else Perturb.NONE)

    k2 = func(t0 + dt * butcher_tableu[1][0], y0 + dt * k1 * butcher_tableu[1][1], perturb=Perturb.PREV if perturb else Perturb.NONE)
    return dt * (k1 * butcher_tableu[2][1] + k2 * butcher_tableu[2][2])


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeEventODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, func, y0, rtol, atol,
                 min_step=0,
                 max_step=float('inf'),
                 first_step=None,
                 step_t=None,
                 jump_t=None,
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 dtype=torch.float64,
                 **kwargs):
        super(RKAdaptiveStepsizeODESolver, self).__init__(dtype=dtype, y0=y0, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0.abs().dtype)
        device = y0.device

        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.min_step = torch.as_tensor(min_step, dtype=dtype, device=device)
        self.max_step = torch.as_tensor(max_step, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.dtype = dtype

        self.step_t = None if step_t is None else torch.as_tensor(step_t, dtype=dtype, device=device)
        self.jump_t = None if jump_t is None else torch.as_tensor(jump_t, dtype=dtype, device=device)

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.mid = self.mid.to(device=device, dtype=y0.dtype)

    @classmethod
    def valid_callbacks(cls):
        return super(RKAdaptiveStepsizeODESolver, cls).valid_callbacks() | {'callback_step',
                                                                            'callback_accept_step',
                                                                            'callback_reject_step'}

    def _before_integrate(self, t):
        t0 = t[0]
        f0 = self.func(t[0], self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                              self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)

        # Handle step_t and jump_t arguments.
        if self.step_t is None:
            step_t = torch.tensor([], dtype=self.dtype, device=self.y0.device)
        else:
            step_t = _sort_tvals(self.step_t, t0)
            step_t = step_t.to(self.dtype)
        if self.jump_t is None:
            jump_t = torch.tensor([], dtype=self.dtype, device=self.y0.device)
        else:
            jump_t = _sort_tvals(self.jump_t, t0)
            jump_t = jump_t.to(self.dtype)
        counts = torch.cat([step_t, jump_t]).unique(return_counts=True)[1]
        if (counts > 1).any():
            raise ValueError("`step_t` and `jump_t` must not have any repeated elements between them.")

        self.step_t = step_t
        self.jump_t = jump_t
        self.next_step_index = min(bisect.bisect(self.step_t.tolist(), t[0]), len(self.step_t) - 1)
        self.next_jump_index = min(bisect.bisect(self.jump_t.tolist(), t[0]), len(self.jump_t) - 1)

    def _advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        return _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, next_t)

    def _advance_until_event(self, event_fn):
        """Returns t, state(t) such that event_fn(t, state(t)) == 0."""
        if event_fn(self.rk_state.t1, self.rk_state.y1) == 0:
            return (self.rk_state.t1, self.rk_state.y1)

        n_steps = 0
        sign0 = torch.sign(event_fn(self.rk_state.t1, self.rk_state.y1))
        while sign0 == torch.sign(event_fn(self.rk_state.t1, self.rk_state.y1)):
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_step(self.rk_state)
            n_steps += 1
        interp_fn = lambda t: _interp_evaluate(self.rk_state.interp_coeff, self.rk_state.t0, self.rk_state.t1, t)
        return find_event(interp_fn, sign0, self.rk_state.t0, self.rk_state.t1, event_fn, self.atol)

    def _adaptive_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, interp_coeff = rk_state
        if not torch.isfinite(dt):
            dt = self.min_step
        dt = dt.clamp(self.min_step, self.max_step)
        self.func.callback_step(t0, y0, dt)
        t1 = t0 + dt
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

        on_step_t = False
        if len(self.step_t):
            next_step_t = self.step_t[self.next_step_index]
            on_step_t = t0 < next_step_t < t0 + dt
            if on_step_t:
                t1 = next_step_t
                dt = t1 - t0

        on_jump_t = False
        if len(self.jump_t):
            next_jump_t = self.jump_t[self.next_jump_index]
            on_jump_t = t0 < next_jump_t < t0 + dt
            if on_jump_t:
                on_step_t = False
                t1 = next_jump_t
                dt = t1 - t0

        # Must be arranged as doing all the step_t handling, then all the jump_t handling, in case we
        # trigger both. (i.e. interleaving them would be wrong.)

        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, t1, tableau=self.tableau)
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

        # Handle min max stepping
        if dt > self.max_step:
            accept_step = False
        if dt <= self.min_step:
            accept_step = True

        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        if accept_step:
            self.func.callback_accept_step(t0, y0, dt)
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            if on_step_t:
                if self.next_step_index != len(self.step_t) - 1:
                    self.next_step_index += 1
            if on_jump_t:
                if self.next_jump_index != len(self.jump_t) - 1:
                    self.next_jump_index += 1
                # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity
                # we're now on.
                f1 = self.func(t_next, y_next, perturb=Perturb.NEXT)
            f_next = f1
        else:
            self.func.callback_reject_step(t0, y0, dt)
            t_next = t0
            y_next = y0
            f_next = f0
        dt_next = _optimal_step_size(dt, error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        dt_next = dt_next.clamp(self.min_step, self.max_step)
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(y0)
        y_mid = y0 + torch.sum(k * (dt * self.mid), dim=-1).view_as(y0)
        f0 = k[..., 0]
        f1 = k[..., -1]
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)


def _sort_tvals(tvals, t0):
    # TODO: add warning if tvals come before t0?
    tvals = tvals[tvals >= t0]
    return torch.sort(tvals).values


class FixedGridImplicitODESolver(FixedGridODESolver):
    """Common machinery for the fixed-grid implicit Runge-Kutta solvers.

    Every step requires solving a system of nonlinear equations for the stage
    derivatives `k`. For stage `i` the residual is

        F(k) = k_i - f(t_i, y0 + dt * sum_j a_ij k_j)

    and its Jacobian is `I - dt * a_ii * df/dy` (FIRK: `I - dt * (A kron df/dy)`).

    This is solved by a damped Newton iteration using that Jacobian, evaluated at the
    *current iterate*. Both of those matter on stiff problems:

    * Seeding the iteration with the identity instead reduces the first step to a
      plain functional (fixed point) iteration, which diverges whenever
      `|dt * lambda| > 1` -- precisely the regime these solvers exist to handle.
    * Freezing `df/dy` at `y0` is not enough either: for problems such as Robertson
      the stiffness is invisible at the initial point (the stiff term is multiplied
      by a component that starts at zero), so the Jacobian must be refreshed as the
      iterate moves.
    """

    order: int
    tableau: _ButcherTableau

    # Number of step halvings the line search may try before declaring no descent.
    _max_backtracks = 30

    def __init__(self, func, y0, step_size=None, grid_constructor=None, interp='linear', perturb=False,
                 max_iters=100, **unused_kwargs):

        self.max_iters = max_iters
        self.rtol = unused_kwargs.pop('rtol', 1e-7)
        super(FixedGridImplicitODESolver, self).__init__(func=func, y0=y0, step_size=step_size,
                                                        grid_constructor=grid_constructor, interp=interp,
                                                        perturb=perturb, **unused_kwargs)

        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=self.device, dtype=y0.dtype),
                                       beta=[b.to(device=self.device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=self.device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=self.device, dtype=y0.dtype))

    # ------------------------------------------------------------------ helpers

    def _newton_tol(self, y0, dt):
        """Tolerance for the nonlinear solve, expressed in units of the residual.

        Derived from the user's `atol`/`rtol` rather than hardcoded, so it tracks the
        accuracy actually being asked for, and floored at the level of floating point
        round-off so that it always remains attainable.

        The residual has the units of `dy/dt`, whereas `atol`/`rtol` describe `y`. The
        two are reconciled by dividing by `dt`: an error of `tol` in `y` corresponds to
        one of `tol / dt` in the stage derivatives. Comparing an unscaled residual
        against a `y`-scale tolerance is what makes an otherwise perfectly converged
        solve report failure on problems where `f` is large -- which is every stiff
        problem.
        """
        dtype = y0.abs().dtype
        scale = float(y0.abs().max().detach())
        atol = float(self.atol.max() if torch.is_tensor(self.atol) else self.atol)
        rtol = float(self.rtol.max() if torch.is_tensor(self.rtol) else self.rtol)
        # Ask for rather more than the discretisation error we are aiming for,
        # otherwise the nonlinear solve caps the accuracy of the high order methods.
        tol_y = 1e-3 * (atol + rtol * scale)
        tol_y = max(tol_y, 100 * torch.finfo(dtype).eps * max(scale, 1.0))
        return tol_y / max(abs(float(dt)), torch.finfo(dtype).tiny)

    def _jacobian(self, func, t, y, perturb=Perturb.NONE):
        """(n, n) Jacobian of the flattened `func(t, .)` at `y`. Detached."""
        n = y.numel()
        y_ = y.detach()

        def _g(z):
            return func(t, z.view_as(y_), perturb=perturb).reshape(-1)

        if torch.isfinite(y_).all():
            try:
                with torch.enable_grad():
                    J = torch.autograd.functional.jacobian(_g, y_.reshape(-1), vectorize=True)
                J = J.reshape(n, n).detach()
                if torch.isfinite(J).all():
                    return J
            except Exception:
                pass
            try:
                return self._fd_jacobian(_g, y_.reshape(-1), n)
            except Exception:
                pass
        return torch.zeros(n, n, dtype=y_.dtype, device=y_.device)

    @staticmethod
    def _fd_jacobian(g, y_flat, n):
        """Finite difference fallback, for functions autograd cannot differentiate."""
        with torch.no_grad():
            f0 = g(y_flat)
            eps = torch.finfo(y_flat.abs().dtype).eps ** 0.5
            J = torch.empty(n, n, dtype=f0.dtype, device=f0.device)
            for i in range(n):
                h = eps * max(float(y_flat[i].abs()), 1.0)
                y_pert = y_flat.clone()
                y_pert[i] = y_pert[i] + h
                J[:, i] = (g(y_pert) - f0) / h
        return J

    @torch.no_grad()
    def _solve_newton(self, residual_fn, jac_fn, x, tol):
        """Damped Newton iteration on `residual_fn(x) == 0`, starting from `x`.

        `jac_fn(x)` returns the Jacobian of the residual at `x`. It is refreshed
        whenever the iteration stops converging quickly, and refined by Broyden's
        "good" update in between. A backtracking line search keeps the residual
        monotonically decreasing, so a bad step can never blow the iterate up to
        inf/NaN and poison the rest of the integration.

        Returns `(x, converged)`.
        """
        f = residual_fn(x)
        if not torch.isfinite(f).all():
            return x, False
        f_norm = torch.linalg.norm(f, 2)
        if f_norm < tol:
            return x, True

        J = jac_fn(x)
        fresh = True

        for _ in range(self.max_iters):
            try:
                step = -torch.linalg.solve(J, f)
            except Exception:
                # Singular Jacobian: nudge the diagonal and retry once.
                eye = torch.eye(J.shape[0], dtype=J.dtype, device=J.device)
                scale_ = torch.linalg.matrix_norm(J, ord=float('inf')).clamp(min=1.0)
                try:
                    step = -torch.linalg.solve(J + 1e-10 * scale_ * eye, f)
                except Exception:
                    return x, False
            if not torch.isfinite(step).all():
                if fresh:
                    return x, False
                J, fresh = jac_fn(x), True
                continue

            if torch.linalg.norm(step, 2) <= tol:
                # Newton increment test. The step is an estimate of how far the current
                # iterate still is from the root, so this says the remaining error is
                # already below tolerance. It is the criterion that actually matters:
                # the residual itself stalls at round-off (relative to `f`, which is
                # huge on a stiff problem) and may never reach an absolute threshold.
                return x + step, True

            # Backtracking line search: never accept a step that makes things worse.
            # With an exact Jacobian the Newton direction always reduces ||F|| for a
            # small enough step, but "small enough" can be very small indeed on a
            # strongly nonlinear problem (Robertson's 3e7 y^2 term, say), so the
            # backtracking has to be allowed a wide range before giving up.
            damp = 1.0
            accepted = False
            for _ in range(self._max_backtracks):
                x_new = x + damp * step
                f_new = residual_fn(x_new)
                if torch.isfinite(f_new).all():
                    f_new_norm = torch.linalg.norm(f_new, 2)
                    if f_new_norm < f_norm:
                        accepted = True
                        break
                damp *= 0.5

            if not accepted:
                if fresh:
                    # Exact Jacobian and still no descent: we are as close as we get.
                    return x, bool(f_norm < tol)
                J, fresh = jac_fn(x), True
                continue

            s = damp * step
            z = f_new - f
            rate = f_new_norm / f_norm
            x, f, f_norm = x_new, f_new, f_new_norm

            if f_norm < tol:
                return x, True

            if damp < 1.0 or rate > 0.5:
                # Converging slowly: the Jacobian is no longer a good model.
                J, fresh = jac_fn(x), True
            else:
                # vecdot conjugates its first argument, so this is ||s||^2 and the
                # outer product uses s-conjugate: both reduce to the usual Broyden
                # update for real tensors, and stay correct for complex states.
                denom = torch.linalg.vecdot(s, s).real
                if denom > 0:
                    J = J + torch.outer(z - torch.mv(J, s), s.conj()) / denom
                    fresh = False

        return x, bool(f_norm < tol)

    @staticmethod
    def _reattach(residual_fn, jac_fn, x):
        """Re-attach the autograd graph with one differentiable Newton step from `x`.

        The Newton iteration above runs under `no_grad`, so `x` carries no history.
        Since the residual vanishes at `x`, `x - J^{-1} F(x)` has the same value but
        the correct derivative with respect to `y0` and any parameters of `func` --
        this is the implicit function theorem, and it is both cheaper and more
        accurate than backpropagating through every iteration.
        """
        if not torch.is_grad_enabled():
            return x
        f = residual_fn(x)
        if not f.requires_grad or not torch.isfinite(f).all():
            return x
        try:
            return x - torch.linalg.solve(jac_fn(x), f)
        except Exception:
            return x

    def _check(self, converged, dy, method_name, t0):
        if not torch.isfinite(dy).all():
            raise RuntimeError(
                '{}: the implicit step at t={:.6g} produced non-finite values -- the nonlinear solve '
                'diverged. Reduce `step_size`, or use an adaptive method such as `dopri5`.'
                .format(method_name, float(t0)))
        if not converged:
            warnings.warn(
                "{}: the nonlinear solve did not converge at t={:.6g}. Treat the returned solution as "
                "wrong rather than merely imprecise -- a step that fails here can settle on a different "
                "branch of the solution entirely. Reduce `step_size` until this warning stops. Note that "
                "`max_iters` rarely helps (the iteration stalls rather than running out of steps), and "
                "`rtol`/`atol` only set how tightly this solve must converge, not the accuracy of the "
                "integration itself -- for a fixed-grid method that is governed by `step_size`."
                .format(method_name, float(t0)),
                RuntimeWarning)


class FixedGridFIRKODESolver(FixedGridImplicitODESolver):
    """Fully implicit Runge-Kutta: all stages are solved as one coupled system."""

    def _step_func(self, func, t0, dt, t1, y0):
        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt)
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1)
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)

        t_dtype = y0.abs().dtype
        t0 = t0.to(t_dtype)
        dt = dt.to(t_dtype)
        t1 = t1.to(t_dtype)

        n = y0.numel()
        s = len(self.tableau.alpha)
        # beta_T[j, i] == a_ij, so `k @ beta_T` forms sum_j a_ij k_j for every stage i.
        beta_T = torch.stack(self.tableau.beta, -1)
        A = torch.stack(self.tableau.beta, 0)  # A[i, j] == a_ij
        dt_y = dt.to(y0.dtype)

        stages = self._stage_times(t0, dt, t1)

        def _stage_states(x):
            k = x.reshape(*f0.shape, s)
            return k, torch.matmul(k, beta_T * dt).add(y0.unsqueeze(-1)).movedim(-1, 0)

        def residual_fn(x):
            k, y = _stage_states(x)
            return self._residual(func, k, y, stages)

        def jac_fn(x):
            # J[(a,i),(b,j)] = delta_ab delta_ij - dt * a_ij * (df/dy at stage i)_ab,
            # with the unknown flattened as k[state, stage].
            _, y = _stage_states(x)
            J = torch.eye(n * s, dtype=y0.dtype, device=y0.device)
            for i, (ti, perturb, skip) in enumerate(stages):
                if skip:
                    continue
                Jf = self._jacobian(func, ti, y[i], perturb=perturb).to(y0.dtype)
                for j in range(s):
                    a_ij = A[i, j]
                    if a_ij != 0:
                        J[i::s, j::s] -= (dt_y * a_ij) * Jf
            return J

        has_explicit = any(skip for _, _, skip in stages)

        def _restore_explicit(x):
            # An explicit stage is not part of the nonlinear system, so the Newton
            # iteration never gives it an autograd history. Put `f0` itself back, both
            # so that its own contribution to `dy` carries a gradient and so that the
            # other stages' residuals see that `f0` depends on `y0` and on `func`'s
            # parameters. Numerically this is a no-op: k_i already equals f0 there.
            if not has_explicit:
                return x
            k = x.reshape(*f0.shape, s)
            return torch.stack([f0 if stages[i][2] else k[..., i] for i in range(s)], dim=-1).reshape(-1)

        k0 = f0.detach().reshape(-1, 1).tile(s).reshape(-1)
        x, converged = self._solve_newton(residual_fn, jac_fn, k0, self._newton_tol(y0, dt))
        x = self._reattach(residual_fn, jac_fn, _restore_explicit(x))
        k = _restore_explicit(x).reshape(*f0.shape, s)

        dy = torch.matmul(k, dt * self.tableau.c_sol)
        self._check(converged, dy, type(self).__name__, t0)
        return dy, f0

    def _stage_times(self, t0, dt, t1):
        """Per-stage `(t_i, perturb, skip)`. `skip` marks an explicit first stage."""
        out = []
        for i, alpha_i in enumerate(self.tableau.alpha):
            if alpha_i == 1.:
                out.append((t1, Perturb.PREV, False))
            elif alpha_i == 0.:
                # An explicit stage: y_i == y0 and k_i == f0, which is how it is
                # initialised, so its residual is identically zero.
                out.append((t0, Perturb.NONE, not bool(torch.any(self.tableau.beta[i]))))
            else:
                out.append((t0 + alpha_i * dt, Perturb.NONE, False))
        return out

    def _residual(self, func, K, y, stages):
        res = torch.zeros_like(K)
        for i, (ti, perturb, skip) in enumerate(stages):
            if skip:
                continue
            res[..., i] = K[..., i] - func(ti, y[i], perturb=perturb)
        return res.flatten()


class FixedGridDIRKODESolver(FixedGridImplicitODESolver):
    """Diagonally implicit Runge-Kutta: the stages are solved one after another."""

    def _step_func(self, func, t0, dt, t1, y0):
        if not isinstance(t0, torch.Tensor):
            t0 = torch.tensor(t0)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt)
        if not isinstance(t1, torch.Tensor):
            t1 = torch.tensor(t1)
        f0 = func(t0, y0, perturb=Perturb.NEXT if self.perturb else Perturb.NONE)

        t_dtype = y0.abs().dtype
        t0 = t0.to(t_dtype)
        dt = dt.to(t_dtype)
        t1 = t1.to(t_dtype)

        n = y0.numel()
        tol = self._newton_tol(y0, dt)
        eye = torch.eye(n, dtype=y0.dtype, device=y0.device)
        dt_y = dt.to(y0.dtype)

        k = [f0] * len(self.tableau.alpha)
        all_converged = True

        for i, (alpha_i, beta_i) in enumerate(zip(self.tableau.alpha, self.tableau.beta)):
            perturb = Perturb.NONE
            if alpha_i == 1.:
                ti = t1
                perturb = Perturb.PREV
            elif alpha_i == 0.:
                if not torch.any(self.tableau.beta[i]):
                    # Explicit first stage: k_i == f0 already.
                    continue
                ti = t0
            else:
                ti = t0 + alpha_i * dt

            prev = k[:i]
            a_ii = beta_i[-1]

            def _stage_state(x, _prev=prev, _beta_i=beta_i):
                k_i = torch.stack(_prev + [x.view_as(f0)], -1)
                return k_i, torch.matmul(k_i, _beta_i * dt).add(y0)

            def residual_fn(x, _ti=ti, _perturb=perturb):
                k_i, y_i = _stage_state(x)
                return (k_i[..., -1] - func(_ti, y_i, perturb=_perturb)).flatten()

            def jac_fn(x, _ti=ti, _perturb=perturb, _a_ii=a_ii):
                # dF/dk_i = I - dt * a_ii * df/dy, at the current stage state.
                _, y_i = _stage_state(x)
                if _a_ii == 0:
                    return eye
                Jf = self._jacobian(func, _ti, y_i, perturb=_perturb).to(y0.dtype)
                return eye - (dt_y * _a_ii) * Jf

            x, converged = self._solve_newton(residual_fn, jac_fn, k[i].detach().reshape(-1).clone(), tol)
            k[i] = self._reattach(residual_fn, jac_fn, x).view_as(f0)
            all_converged = all_converged and converged

        dy = torch.matmul(torch.stack(k, -1), dt * self.tableau.c_sol)
        self._check(all_converged, dy, type(self).__name__, t0)
        return dy, f0
