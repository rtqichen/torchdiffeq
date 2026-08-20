"""Adaptive-step Radau IIA (3 stage, order 5).

An L-stable, stiffly accurate implicit Runge-Kutta method with an embedded error
estimator, so it chooses its own step sizes. Unlike the `scipy_solver` wrapper it
runs natively in torch: it supports GPU tensors, batched states, and gradients by
direct backpropagation as well as through the adjoint.

The method, its error estimator and its dense output follow

    E. Hairer, G. Wanner, *Solving Ordinary Differential Equations II: Stiff and
    Differential-Algebraic Problems*, 2nd ed., Section IV.8.

Being stiffly accurate (the last row of `A` equals the weights `b`), the step is
just `y1 = y0 + Z_s`, where `Z` are the stage increments solved for below.
"""
import torch

from .event_handling import find_event
from .interp import _interp_evaluate
from .misc import Perturb, _compute_error_ratio, _select_initial_step, _handle_unused_kwargs
from .solvers import AdaptiveStepsizeEventODESolver


_S6 = 6 ** 0.5

# Collocation points.
_C = [(4 - _S6) / 10, (4 + _S6) / 10, 1.0]

# Coefficient matrix.
_A = [[(88 - 7 * _S6) / 360, (296 - 169 * _S6) / 1800, (-2 + 3 * _S6) / 225],
      [(296 + 169 * _S6) / 1800, (88 + 7 * _S6) / 360, (-2 - 3 * _S6) / 225],
      [(16 - _S6) / 36, (16 + _S6) / 36, 1 / 9]]

# Error estimator weights, and the real eigenvalue of inv(A) that scales it.
_E = [(-13 - 7 * _S6) / 3, (-13 + 7 * _S6) / 3, -1 / 3]
_MU_REAL = 3 + 3 ** (2 / 3) - 3 ** (1 / 3)

# Dense output: y(t0 + x*h) = y0 + (Z^T P) @ [x, x^2, x^3].
_P = [[13 / 3 + 7 * _S6 / 3, -23 / 3 - 22 * _S6 / 3, 10 / 3 + 5 * _S6],
      [13 / 3 - 7 * _S6 / 3, -23 / 3 + 22 * _S6 / 3, 10 / 3 - 5 * _S6],
      [1 / 3, -8 / 3, 10 / 3]]

# The error estimate behaves like a 4th order quantity for step size control.
_ERROR_ORDER = 4


def _params_require_grad(func):
    """Whether `func` carries any parameter we would need a gradient for."""
    try:
        return any(p.requires_grad for p in func.parameters())
    except AttributeError:
        return False


class RadauIIA5Adaptive(AdaptiveStepsizeEventODESolver):
    order = 5

    def __init__(self, func, y0, rtol, atol,
                 first_step=None,
                 min_step=0.,
                 max_step=float('inf'),
                 safety=0.9,
                 ifactor=10.0,
                 dfactor=0.2,
                 max_num_steps=2 ** 31 - 1,
                 newton_iters=12,
                 grad_jacobian='exact',
                 grad_iters=4,
                 dtype=torch.float64,
                 **kwargs):
        super(RadauIIA5Adaptive, self).__init__(dtype=dtype, y0=y0, **kwargs)

        dtype = torch.promote_types(dtype, y0.abs().dtype)
        device = y0.device

        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.min_step = torch.as_tensor(min_step, dtype=dtype, device=device)
        self.max_step = torch.as_tensor(max_step, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = float(safety)
        self.ifactor = float(ifactor)
        self.dfactor = float(dfactor)
        self.max_num_steps = int(max_num_steps)
        self.newton_iters = int(newton_iters)
        if grad_jacobian not in ('iterative', 'exact'):
            raise ValueError("`grad_jacobian` must be 'iterative' or 'exact', got {!r}".format(grad_jacobian))
        self.grad_jacobian = grad_jacobian
        self.grad_iters = int(grad_iters)
        self.dtype = dtype

        y_dtype = y0.dtype
        self.C = torch.tensor(_C, dtype=dtype, device=device)
        self.A = torch.tensor(_A, dtype=y_dtype, device=device)
        self.E = torch.tensor(_E, dtype=y_dtype, device=device)
        self.P = torch.tensor(_P, dtype=y_dtype, device=device)

    @classmethod
    def valid_callbacks(cls):
        return super(RadauIIA5Adaptive, cls).valid_callbacks() | {'callback_step',
                                                                  'callback_accept_step',
                                                                  'callback_reject_step'}

    # ------------------------------------------------------------------ set-up

    def _before_integrate(self, t):
        self.n = self.y0.numel()
        self.t = t[0].to(self.dtype)
        self.y = self.y0
        self.f = self.func(self.t, self.y)

        if self.first_step is None:
            h = _select_initial_step(self.func, t[0], self.y0, self.order - 1,
                                     self.rtol, self.atol, self.norm, f0=self.f)
        else:
            h = self.first_step
        self.h = h.abs().clamp(self.min_step, self.max_step).to(self.dtype)

        # Newton tolerance, following the usual "solve rather tighter than the
        # local error we are targeting" rule.
        eps = torch.finfo(self.y0.abs().dtype).eps
        rtol = float(self.rtol.max())
        self.newton_tol = max(10 * eps / max(rtol, eps), min(0.03, rtol ** 0.5))

        self.J = None            # cached Jacobian, refreshed when Newton struggles
        self.J_current = False
        self.Z = None            # previous stage increments, for extrapolation
        self.t_old = self.t
        self.y_old = self.y
        self.h_old = self.h
        self.Q = None            # dense output coefficients for [t_old, t]

    # ------------------------------------------------------------------ pieces

    def _jacobian(self, t, y):
        """(n, n) Jacobian of the flattened `func(t, .)` at `y`, detached."""
        y_ = y.detach()

        def _g(z):
            return self.func(t, z.view_as(y_)).reshape(-1)

        try:
            with torch.enable_grad():
                J = torch.autograd.functional.jacobian(_g, y_.reshape(-1), vectorize=True)
            J = J.reshape(self.n, self.n).detach()
            if torch.isfinite(J).all():
                return J
        except Exception:
            pass
        return self._fd_jacobian(_g, y_.reshape(-1))

    def _fd_jacobian(self, g, y_flat):
        with torch.no_grad():
            f0 = g(y_flat)
            eps = torch.finfo(y_flat.abs().dtype).eps ** 0.5
            J = torch.empty(self.n, self.n, dtype=f0.dtype, device=f0.device)
            for i in range(self.n):
                h = eps * max(float(y_flat[i].abs()), 1.0)
                y_pert = y_flat.clone()
                y_pert[i] = y_pert[i] + h
                J[:, i] = (g(y_pert) - f0) / h
        return J

    def _stage_residual(self, t, y, h, Z):
        """G(Z) = Z - h (A kron I) F(Z), with Z of shape (3, n)."""
        F = torch.stack([
            self.func(t + self.C[i] * h,
                      (y.reshape(-1) + Z[i]).view_as(y),
                      perturb=Perturb.PREV if self.C[i] == 1. else Perturb.NONE).reshape(-1)
            for i in range(3)
        ])
        return Z - h.to(Z.dtype) * torch.matmul(self.A, F)

    @torch.no_grad()
    def _solve_stages(self, t, y, h, Z, lu):
        """Simplified Newton on the stage system. Returns (Z, converged, n_iter)."""
        dZ_norm_old = None
        rate = None
        for n_iter in range(1, self.newton_iters + 1):
            G = self._stage_residual(t, y, h, Z)
            if not torch.isfinite(G).all():
                return Z, False, n_iter
            dZ = torch.linalg.lu_solve(lu[0], lu[1], (-G).reshape(-1, 1)).reshape(3, self.n)
            dZ_norm = self._scaled_norm(dZ, y)

            if dZ_norm_old is not None and dZ_norm_old > 0:
                rate = dZ_norm / dZ_norm_old
                if rate >= 1:
                    # Diverging. The step will be rejected and h reduced.
                    return Z, False, n_iter

            Z = Z + dZ

            if dZ_norm == 0:
                return Z, True, n_iter
            if rate is not None and rate / (1 - rate) * dZ_norm < self.newton_tol:
                return Z, True, n_iter
            dZ_norm_old = dZ_norm

        return Z, False, self.newton_iters

    def _scaled_norm(self, dZ, y):
        scale = self.atol + self.rtol * y.abs().reshape(-1)
        return float((dZ / scale).pow(2).mean().sqrt())

    def _lu_stage_matrix(self, h):
        """LU of I - h (A kron J), the Jacobian of the stage residual."""
        eye = torch.eye(3 * self.n, dtype=self.y0.dtype, device=self.y0.device)
        M = eye - torch.kron((self.A * h.to(self.y0.dtype)), self.J)
        return torch.linalg.lu_factor(M)

    # ------------------------------------------------------------------ step

    def _step(self):
        """Take one accepted step. Advances t, y, f, h and the dense output."""
        t, y, f = self.t, self.y, self.f
        y_flat = y.reshape(-1)

        h = self.h.clamp(self.min_step, self.max_step)
        self.func.callback_step(t, y, h)

        while True:
            assert t + h > t, 'underflow in dt {}'.format(h.item())

            if self.J is None:
                self.J = self._jacobian(t, y)
                self.J_current = True

            lu = self._lu_stage_matrix(h)

            # Initial guess: extrapolate the previous collocation polynomial, which
            # is far better than zero once the integration is under way.
            if self.Q is None:
                Z0 = torch.zeros(3, self.n, dtype=y.dtype, device=y.device)
            else:
                x = ((t + self.C * h - self.t_old) / self.h_old).to(y.dtype)
                powers = torch.stack([x, x * x, x * x * x])          # (3 powers, 3 stages)
                Z0 = torch.matmul(self.Q, powers).T + (self.y_old.reshape(-1) - y_flat)

            Z, converged, n_iter = self._solve_stages(t, y, h, Z0, lu)

            if not converged:
                if not self.J_current:
                    # Stale Jacobian is the likeliest cause; refresh and retry at
                    # the same step size before giving up on it.
                    self.J = self._jacobian(t, y)
                    self.J_current = True
                    continue
                self.func.callback_reject_step(t, y, h)
                h = (h * 0.5).clamp(self.min_step, self.max_step)
                continue

            # Error estimate: (mu/h I - J)^-1 (f + Z^T E / h), see Hairer & Wanner.
            err = self._error_estimate(t, y, h, Z, f)
            y_new_flat = y_flat + Z[2]
            y_new = y_new_flat.view_as(y)
            error_ratio = _compute_error_ratio(err.view_as(y), self.rtol, self.atol, y, y_new, self.norm)

            if error_ratio <= 1 or h <= self.min_step:
                self.func.callback_accept_step(t, y, h)
                break

            self.func.callback_reject_step(t, y, h)
            h = (h * self._step_factor(error_ratio, n_iter)).clamp(self.min_step, self.max_step)
            self.J_current = False  # allow a refresh if the retry also struggles

        # Re-attach the autograd graph. The Newton loop above ran under `no_grad`,
        # so `Z` carries no history; differentiable Newton steps from the converged
        # point leave the value alone (the residual vanishes there) while building
        # the derivative.
        #
        # One step is not enough here. The step uses the *simplified* Jacobian `M`
        # (frozen, and possibly several steps stale), which is fine for converging
        # the value but leaves the derivative short by a factor. Iterating drives it
        # to the true implicit-function-theorem derivative: the error after k steps
        # is (I - M^-1 dG/dZ)^k, i.e. it contracts at exactly the rate the simplified
        # Newton iteration itself converged at.
        if torch.is_grad_enabled() and (y.requires_grad or _params_require_grad(self.func)):
            Z = self._reattach(t, y, h, Z, lu)

        y_new = (y.reshape(-1) + Z[2]).view_as(y)
        t_new = t + h

        self.Q = torch.matmul(Z.T, self.P)          # (n, 3) dense output coefficients
        self.t_old, self.y_old, self.h_old = t, y, h
        self.t, self.y = t_new, y_new
        self.f = self.func(t_new, y_new)
        self.J_current = False
        self.h = (h * self._step_factor(error_ratio, n_iter)).clamp(self.min_step, self.max_step)

    def _reattach(self, t, y, h, Z, lu):
        """Give the converged stages an autograd history.

        `Z` comes out of `_solve_stages` detached. A Newton step from a converged
        point leaves the value alone (the residual vanishes there) while attaching
        the derivative -- the implicit function theorem, without backpropagating
        through every Newton iteration.

        Two ways to do it, selected by `grad_jacobian`:

        `'exact'` (default) builds the true dG/dZ, which makes a single step exact.
        It costs three Jacobian evaluations per step, but measured against the
        alternative below it is both more accurate and faster at small to moderate
        state sizes, because what dominates is graph construction rather than the
        Jacobians.

        `'iterative'` instead reuses the simplified matrix already factorised for
        the forward solve. One step with that matrix under-counts the derivative by
        a factor of (I - M^-1 dG/dZ); iterating contracts that error at the rate the
        simplified Newton iteration itself converged at, needing about four passes
        to match `'exact'`. Its cost is independent of the state size, so it becomes
        the cheaper option for a large state, where an exact Jacobian means a
        backward pass per component.
        """
        if self.grad_jacobian == 'exact':
            try:
                M_lu = torch.linalg.lu_factor(self._exact_stage_jacobian(t, y, h, Z))
                G = self._stage_residual(t, y, h, Z)
                return Z - torch.linalg.lu_solve(M_lu[0], M_lu[1], G.reshape(-1, 1)).reshape(3, self.n)
            except Exception:
                pass
        for _ in range(self.grad_iters):
            G = self._stage_residual(t, y, h, Z)
            Z = Z - torch.linalg.lu_solve(lu[0], lu[1], G.reshape(-1, 1)).reshape(3, self.n)
        return Z

    def _exact_stage_jacobian(self, t, y, h, Z):
        """dG/dZ at the converged stages, with df/dy taken at each stage point.

        The Newton iteration above deliberately uses one frozen, possibly stale
        Jacobian for every stage -- that converges the value perfectly well. The
        derivative is less forgiving: the implicit function theorem needs the true
        dG/dZ, so a re-attach through the simplified matrix comes out short by a
        factor of (I - M^-1 dG/dZ). Building the exact matrix here makes the single
        re-attach step exact.
        """
        n = self.n
        h_y = h.to(y.dtype)
        y_flat = y.reshape(-1)

        # G[i] = Z[i] - h * sum_j A[i,j] * f(t_j, y + Z[j])
        # so  dG[i]/dZ[j] = delta_ij I - h A[i,j] * (df/dy at stage j).
        # Note the Jacobian is taken at stage j, not stage i: in this formulation
        # the unknown Z[j] only enters through the j-th function evaluation. (The
        # fixed-grid solvers in rk_common solve for the stage *derivatives* K
        # instead, where the roles are genuinely reversed and J_i is correct.)
        Js = [self._jacobian(t + self.C[k] * h, (y_flat + Z[k]).view_as(y)) for k in range(3)]

        M = torch.eye(3 * n, dtype=y.dtype, device=y.device)
        for i in range(3):
            for j in range(3):
                a_ij = self.A[i, j]
                if a_ij != 0:
                    M[i * n:(i + 1) * n, j * n:(j + 1) * n] -= (h_y * a_ij) * Js[j]
        return M

    def _error_estimate(self, t, y, h, Z, f):
        h_y = h.to(y.dtype)
        eye = torch.eye(self.n, dtype=y.dtype, device=y.device)
        M = (_MU_REAL / h_y) * eye - self.J
        rhs = f.reshape(-1) + torch.matmul(self.E, Z) / h_y
        return torch.linalg.solve(M, rhs)

    def _step_factor(self, error_ratio, n_iter):
        """Step size multiplier, damped when the Newton iteration was slow.

        The `n_iter` term is the RADAU5 heuristic: a step whose nonlinear solve
        needed many iterations is charged for it, because growing the step would
        make the next solve harder still.
        """
        error_ratio = float(error_ratio)
        m = 2 * self.newton_iters
        safety = self.safety * (m + 1) / (m + n_iter)
        if error_ratio == 0:
            return self.ifactor
        factor = safety * error_ratio ** (-1.0 / _ERROR_ORDER)
        return min(self.ifactor, max(self.dfactor, factor))

    # ------------------------------------------------------------------ driver

    def _advance(self, next_t):
        n_steps = 0
        while next_t > self.t:
            assert n_steps < self.max_num_steps, \
                'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self._step()
            n_steps += 1
        return self._dense(next_t)

    def _advance_until_event(self, event_fn):
        if event_fn(self.t, self.y) == 0:
            return self.t, self.y

        n_steps = 0
        sign0 = torch.sign(event_fn(self.t, self.y))
        while sign0 == torch.sign(event_fn(self.t, self.y)):
            assert n_steps < self.max_num_steps, \
                'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self._step()
            n_steps += 1
        return find_event(self._dense, sign0, self.t_old, self.t, event_fn, self.atol)

    def _dense(self, t_eval):
        """Evaluate the collocation polynomial covering [t_old, t]."""
        if self.Q is None or t_eval == self.t:
            return self.y
        x = ((t_eval - self.t_old) / self.h_old).to(self.y.dtype)
        powers = torch.stack([x, x * x, x * x * x])
        return (self.y_old.reshape(-1) + torch.matmul(self.Q, powers)).view_as(self.y)
