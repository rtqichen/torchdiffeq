import collections
import torch
from .solvers import AdaptiveStepsizeODESolver
from .misc import (_compute_error_ratio,
                   _error_tol,
                   _handle_unused_kwargs,
                   _optimal_step_size,
                   _select_initial_step)

_MIN_ORDER = 1
_MAX_ORDER = 12

_gamma_star = [
    1, -1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160, -863 / 60480, -275 / 24192, -33953 / 3628800, -0.00789255,
    -0.00678585, -0.00592406, -0.00523669, -0.0046775, -0.00421495, -0.0038269
]


def _possibly_nonzero(x):
    return isinstance(x, torch.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    # Using _possibly_nonzero lets us avoid wasted computation.
    return sum([(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) or _possibly_nonzero(y)])


class _VCABMState(collections.namedtuple('_VCABMState', 'y_n, prev_f, prev_t, next_t, phi, order')):
    """Saved state of the variable step size Adams-Bashforth-Moulton solver as described in

        Solving Ordinary Differential Equations I - Nonstiff Problems III.5
        by Ernst Hairer, Gerhard Wanner, and Syvert P Norsett.
    """


def g_and_explicit_phi(prev_t, next_t, implicit_phi, k):
    curr_t = prev_t[0]
    dt = next_t - prev_t[0]
    dtype = curr_t.dtype
    device = curr_t.device

    g = torch.empty(k + 1, dtype=dtype, device=device)
    explicit_phi = []
    beta = torch.tensor(1, dtype=dtype, device=device)

    g[0] = 1
    c = torch.arange(1, k + 2, dtype=dtype, device=device).reciprocal()
    explicit_phi.append(implicit_phi[0])

    for j in range(1, k):
        beta = ((next_t - prev_t[j - 1]) / (curr_t - prev_t[j])) * beta
        beta_cast = beta.type_as(implicit_phi[j])
        explicit_phi.append(implicit_phi[j] * beta_cast)

        c = c[:-1] - c[1:] if j == 1 else c[:-1] - c[1:] * dt / (next_t - prev_t[j - 1])
        g[j] = c[0]

    c = c[:-1] - c[1:] * dt / (next_t - prev_t[k - 1])
    g[k] = c[0]

    return g, explicit_phi


def compute_implicit_phi(explicit_phi, f_n, k):
    k = min(len(explicit_phi) + 1, k)
    implicit_phi = collections.deque(maxlen=k)
    implicit_phi.append(f_n)
    for j in range(1, k):
        implicit_phi.append(implicit_phi[j - 1] - explicit_phi[j - 1])
    return implicit_phi


class VariableCoefficientAdamsBashforth(AdaptiveStepsizeODESolver):
    def __init__(self, func, y0, rtol, atol, implicit=True, first_step=None, max_order=_MAX_ORDER, safety=0.9,
                 ifactor=10.0, dfactor=0.2, dtype=torch.float64, **kwargs):
        super(VariableCoefficientAdamsBashforth, self).__init__(dtype=dtype, y0=y0, **kwargs)

        assert _MIN_ORDER <= max_order <= _MAX_ORDER, "max_order must lie between {} and {}".format(_MIN_ORDER,
                                                                                                    _MAX_ORDER)

        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device

        self.func = lambda t, y: func(t.type_as(y), y)
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.implicit = implicit
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=torch.float64, device=y0.device)
        self.max_order = int(max_order)
        self.safety = torch.as_tensor(safety, dtype=torch.float64, device=y0.device)
        self.ifactor = torch.as_tensor(ifactor, dtype=torch.float64, device=y0.device)
        self.dfactor = torch.as_tensor(dfactor, dtype=torch.float64, device=y0.device)

    def _before_integrate(self, t):
        prev_f = collections.deque(maxlen=self.max_order + 1)
        prev_t = collections.deque(maxlen=self.max_order + 1)
        phi = collections.deque(maxlen=self.max_order)

        t0 = t[0]
        f0 = self.func(t0, self.y0)
        prev_t.appendleft(t0)
        prev_f.appendleft(f0)
        phi.appendleft(f0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol, self.atol, f0=f0)
        else:
            first_step = self.first_step

        self.vcabm_state = _VCABMState(self.y0, prev_f, prev_t, next_t=t[0] + first_step, phi=phi, order=1)

    def _advance(self, final_t):
        while final_t > self.vcabm_state.prev_t[0]:
            self.vcabm_state = self._adaptive_adams_step(self.vcabm_state, final_t)
        assert final_t == self.vcabm_state.prev_t[0]
        return self.vcabm_state.y_n

    def _adaptive_adams_step(self, vcabm_state, final_t):
        y0, prev_f, prev_t, next_t, prev_phi, order = vcabm_state
        if next_t > final_t:
            next_t = final_t
        dt = (next_t - prev_t[0])
        dt_cast = dt.type_as(y0)

        # Explicit predictor step.
        g, phi = g_and_explicit_phi(prev_t, next_t, prev_phi, order)
        g = g.to(y0)
        p_next = y0 + _scaled_dot_product(dt_cast, g[:max(1, order - 1)], phi[:max(1, order - 1)])

        # Update phi to implicit.
        next_f0 = self.func(next_t, p_next)
        implicit_phi_p = compute_implicit_phi(phi, next_f0, order + 1)

        # Implicit corrector step.
        y_next = p_next + dt_cast * g[order - 1] * implicit_phi_p[order - 1]

        # Error estimation.
        local_error = dt_cast * (g[order] - g[order - 1]) * implicit_phi_p[order]
        tolerance = _error_tol(self.rtol, self.atol, y0, y_next)
        error_k = _compute_error_ratio(local_error, tolerance)
        accept_step = error_k <= 1

        if not accept_step:
            # Retry with adjusted step size if step is rejected.
            dt_next = _optimal_step_size(dt, error_k, self.safety, self.ifactor, self.dfactor, order)
            return _VCABMState(y0, prev_f, prev_t, prev_t[0] + dt_next, prev_phi, order=order)

        # We accept the step. Evaluate f and update phi.
        next_f0 = self.func(next_t, y_next)
        implicit_phi = compute_implicit_phi(phi, next_f0, order + 2)

        next_order = order

        if len(prev_t) <= 4 or order < 3:
            next_order = min(order + 1, 3, self.max_order)
        else:
            error_km1 = _compute_error_ratio(
                dt_cast * (g[order - 1] - g[order - 2]) * implicit_phi_p[order - 1], tolerance
            )
            error_km2 = _compute_error_ratio(
                dt_cast * (g[order - 2] - g[order - 3]) * implicit_phi_p[order - 2], tolerance
            )
            if (error_km1 + error_km2) < error_k:
                next_order = order - 1
            elif order < self.max_order:
                error_kp1 = _compute_error_ratio(dt_cast * _gamma_star[order] * implicit_phi_p[order], tolerance)
                if error_kp1 < error_k:
                    next_order = order + 1

        # Keep step size constant if increasing order. Else use adaptive step size.
        dt_next = dt if next_order > order else _optimal_step_size(
            dt, error_k, self.safety, self.ifactor, self.dfactor, order + 1
        )

        prev_f.appendleft(next_f0)
        prev_t.appendleft(next_t)
        return _VCABMState(p_next, prev_f, prev_t, next_t + dt_next, implicit_phi, order=next_order)
