import collections
import torch
from .solvers import AdaptiveStepsizeODESolver
from .misc import (
    _handle_unused_kwargs, _select_initial_step, _convert_to_tensor, _scaled_dot_product, _is_iterable,
    _optimal_step_size, _compute_error_ratio
)

_MIN_ORDER = 1
_MAX_ORDER = 12

gamma_star = [
    1, -1 / 2, -1 / 12, -1 / 24, -19 / 720, -3 / 160, -863 / 60480, -275 / 24192, -33953 / 3628800, -0.00789255,
    -0.00678585, -0.00592406, -0.00523669, -0.0046775, -0.00421495, -0.0038269
]


class _VCABMState(collections.namedtuple('_VCABMState', 'y_n, prev_f, prev_t, next_t, phi, order')):
    """Saved state of the variable step size Adams-Bashforth-Moulton solver as described in

        Solving Ordinary Differential Equations I - Nonstiff Problems III.5
        by Ernst Hairer, Gerhard Wanner, and Syvert P Norsett.
    """


def g_and_explicit_phi(prev_t, next_t, implicit_phi, k):
    curr_t = prev_t[0]
    dt = next_t - prev_t[0]

    g = torch.empty(k + 1).to(prev_t[0])
    explicit_phi = collections.deque(maxlen=k)
    beta = torch.tensor(1).to(prev_t[0])

    g[0] = 1
    c = 1 / torch.arange(1, k + 2).to(prev_t[0])
    explicit_phi.append(implicit_phi[0])

    for j in range(1, k):
        beta = (next_t - prev_t[j - 1]) / (curr_t - prev_t[j]) * beta
        beat_cast = beta.to(implicit_phi[j][0])
        explicit_phi.append(tuple(iphi_ * beat_cast for iphi_ in implicit_phi[j]))

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
        implicit_phi.append(tuple(iphi_ - ephi_ for iphi_, ephi_ in zip(implicit_phi[j - 1], explicit_phi[j - 1])))
    return implicit_phi


class VariableCoefficientAdamsBashforth(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, implicit=True, first_step=None, max_order=_MAX_ORDER, safety=0.9, ifactor=10.0, dfactor=0.2,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol if _is_iterable(rtol) else [rtol] * len(y0)
        self.atol = atol if _is_iterable(atol) else [atol] * len(y0)
        self.implicit = implicit
        self.first_step = first_step
        self.max_order = int(max(_MIN_ORDER, min(max_order, _MAX_ORDER)))
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)

    def before_integrate(self, t):
        prev_f = collections.deque(maxlen=self.max_order + 1)
        prev_t = collections.deque(maxlen=self.max_order + 1)
        phi = collections.deque(maxlen=self.max_order)

        t0 = t[0]
        f0 = self.func(t0.type_as(self.y0[0]), self.y0)
        prev_t.appendleft(t0)
        prev_f.appendleft(f0)
        phi.appendleft(f0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0).to(t)
        else:
            first_step = _select_initial_step(self.func, t[0], self.y0, 2, self.rtol[0], self.atol[0], f0=f0).to(t)

        self.vcabm_state = _VCABMState(self.y0, prev_f, prev_t, next_t=t[0] + first_step, phi=phi, order=1)

    def advance(self, final_t):
        final_t = _convert_to_tensor(final_t).to(self.vcabm_state.prev_t[0])
        while final_t > self.vcabm_state.prev_t[0]:
            self.vcabm_state = self._adaptive_adams_step(self.vcabm_state, final_t)
        assert final_t == self.vcabm_state.prev_t[0]
        return self.vcabm_state.y_n

    def _adaptive_adams_step(self, vcabm_state, final_t):
        y0, prev_f, prev_t, next_t, prev_phi, order = vcabm_state
        if next_t > final_t:
            next_t = final_t
        dt = (next_t - prev_t[0])
        dt_cast = dt.to(y0[0])

        # Explicit predictor step.
        g, phi = g_and_explicit_phi(prev_t, next_t, prev_phi, order)
        g = g.to(y0[0])
        p_next = tuple(
            y0_ + _scaled_dot_product(dt_cast, g[:max(1, order - 1)], phi_[:max(1, order - 1)])
            for y0_, phi_ in zip(y0, tuple(zip(*phi)))
        )

        # Update phi to implicit.
        next_f0 = self.func(next_t.to(p_next[0]), p_next)
        implicit_phi_p = compute_implicit_phi(phi, next_f0, order + 1)

        # Implicit corrector step.
        y_next = tuple(
            p_next_ + dt_cast * g[order - 1] * iphi_ for p_next_, iphi_ in zip(p_next, implicit_phi_p[order - 1])
        )

        # Error estimation.
        tolerance = tuple(
            atol_ + rtol_ * torch.max(torch.abs(y0_), torch.abs(y1_))
            for atol_, rtol_, y0_, y1_ in zip(self.atol, self.rtol, y0, y_next)
        )
        local_error = tuple(dt_cast * (g[order] - g[order - 1]) * iphi_ for iphi_ in implicit_phi_p[order])
        error_k = _compute_error_ratio(local_error, tolerance)
        accept_step = (torch.tensor(error_k) <= 1).all()

        if not accept_step:
            # Retry with adjusted step size if step is rejected.
            dt_next = _optimal_step_size(dt, error_k, self.safety, self.ifactor, self.dfactor, order=order)
            return _VCABMState(y0, prev_f, prev_t, prev_t[0] + dt_next, prev_phi, order=order)

        # We accept the step. Evaluate f and update phi.
        next_f0 = self.func(next_t.to(p_next[0]), y_next)
        implicit_phi = compute_implicit_phi(phi, next_f0, order + 2)

        next_order = order

        if len(prev_t) <= 4 or order < 3:
            next_order = min(order + 1, 3, self.max_order)
        else:
            error_km1 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 1] - g[order - 2]) * iphi_ for iphi_ in implicit_phi_p[order - 1]), tolerance
            )
            error_km2 = _compute_error_ratio(
                tuple(dt_cast * (g[order - 2] - g[order - 3]) * iphi_ for iphi_ in implicit_phi_p[order - 2]), tolerance
            )
            if min(error_km1 + error_km2) < max(error_k):
                next_order = order - 1
            elif order < self.max_order:
                error_kp1 = _compute_error_ratio(
                    tuple(dt_cast * gamma_star[order] * iphi_ for iphi_ in implicit_phi_p[order]), tolerance
                )
                if max(error_kp1) < max(error_k):
                    next_order = order + 1

        # Keep step size constant if increasing order. Else use adaptive step size.
        dt_next = dt if next_order > order else _optimal_step_size(
            dt, error_k, self.safety, self.ifactor, self.dfactor, order=order + 1
        )

        prev_f.appendleft(next_f0)
        prev_t.appendleft(next_t)
        return _VCABMState(p_next, prev_f, prev_t, next_t + dt_next, implicit_phi, order=next_order)
