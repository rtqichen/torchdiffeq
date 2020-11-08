import bisect
import collections
import torch
from .interp import _interp_evaluate, _interp_fit
from .misc import (_compute_error_ratio,
                   _select_initial_step,
                   _optimal_step_size,
                   _check_timelike,
                   _handle_deprecated_kwargs,
                   _rms_norm,
                   _zero_norm,
                   _mixed_norm)
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

    t0 = t0.type_as(y0)
    dt = dt.type_as(y0)
    t1 = t1.type_as(y0)

    # We use an unchecked assign to put data into k without incrementing its _version counter, so that the backward
    # doesn't throw an (overzealous) error about in-place correctness. We know that it's actually correct.
    k = torch.empty(*f0.shape, len(tableau.alpha) + 1, dtype=y0.dtype, device=y0.device)
    k = _UncheckedAssign.apply(k, f0, (..., 0))
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        if alpha_i == 1.:
            # Always step to perturbing just before the end time, in case of discontinuities.
            ti = t1
            perturb = False
        else:
            ti = t0 + alpha_i * dt
            perturb = None
        yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        f = func(ti, yi, perturb)
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


def rk4_step_func(func, t0, dt, t1, y0, k1=None):
    if k1 is None:
        k1 = func(t0, y0, True)
    half_dt = dt * 0.5
    k2 = func(t0 + half_dt, y0 + half_dt * k1)
    k3 = func(t0 + half_dt, y0 + half_dt * k2)
    k4 = func(t1, y0 + dt * k3, False)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t0, dt, t1, y0, k1=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t0, y0, True)
    k2 = func(t0 + dt * _one_third, y0 + dt * k1 * _one_third)
    k3 = func(t0 + dt * _two_thirds, y0 + dt * (k2 - k1 * _one_third))
    k4 = func(t1, y0 + dt * (k1 - k2 + k3), False)
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeODESolver):
    order: int
    tableau: _ButcherTableau
    mid: torch.Tensor

    def __init__(self, y0, rtol, atol, shapes, is_reversed, norm=None, first_step=None, safety=0.9, ifactor=10.0,
                 dfactor=0.2, max_num_steps=2 ** 31 - 1, step_locations=None, jump_locations=None, dtype=torch.float64,
                 **kwargs):

        if 'grid_points' in kwargs:
            if jump_locations is not None:
                raise ValueError("'grid_points' has been deprecated and is mutually exclusive with 'step_locations'.")
            if jump_locations is not None:
                raise ValueError("'grid_points' has been deprecated and is mutually exclusive with 'jump_locations'.")
            eps = kwargs.get('eps', 0.)
            if eps > 0:
                jump_locations = kwargs['grid_points']
            else:
                step_locations = kwargs['grid_points']
        _handle_deprecated_kwargs(self, kwargs, 'grid_points', "'grid_points' has been deprecated and renamed "
                                                               "'step_locations'.")
        _handle_deprecated_kwargs(self, kwargs, 'eps', "'eps' has been deprecated; use 'jump_locations' instead.")

        super(RKAdaptiveStepsizeODESolver, self).__init__(y0=y0, rtol=rtol, atol=atol, shapes=shapes,
                                                          is_reversed=is_reversed, dtype=dtype, **kwargs)

        # We use mixed precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0.dtype)
        device = y0.device

        if step_locations is None:
            step_locations = torch.tensor([], dtype=dtype, device=device)
        else:
            step_locations, _ = _check_timelike('step_locations', step_locations, False, lambda x: is_reversed)
            step_locations = step_locations.to(dtype)
        if jump_locations is None:
            jump_locations = torch.tensor([], dtype=dtype, device=device)
        else:
            jump_locations, _ = _check_timelike('jump_locations', jump_locations, False, lambda x: is_reversed)
            jump_locations = jump_locations.to(dtype)
        counts = torch.cat([step_locations, jump_locations]).unique(return_counts=True)[1]
        if (counts > 1).any():
            raise ValueError("`step_locations` and `jump_locations` must not have any repeated elements between them.")

        if norm is None:
            if shapes is None:
                # L2 norm over a single input
                norm = _rms_norm
            else:
                # Mixed Linf/L2 norm over tupled input (chosen mostly just for backward compatibility reasons)
                norm = _mixed_norm([_rms_norm for _ in range(len(shapes))], shapes)

        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.norm = norm
        if first_step is None:
            self.first_step = None
        else:
            self.first_step = torch.as_tensor(first_step, dtype=dtype, device=device)
            assert not self.first_step.requires_grad, "first_step cannot require gradient."
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        assert not self.safety.requires_grad, "safety cannot require gradient."
        assert not self.ifactor.requires_grad, "ifactor cannot require gradient."
        assert not self.dfactor.requires_grad, "dfactor cannot require gradient."
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        self.step_locations = step_locations
        self.jump_locations = jump_locations
        self.dtype = dtype

        # Copy from class to instance to set device
        self.tableau = _ButcherTableau(alpha=self.tableau.alpha.to(device=device, dtype=y0.dtype),
                                       beta=[b.to(device=device, dtype=y0.dtype) for b in self.tableau.beta],
                                       c_sol=self.tableau.c_sol.to(device=device, dtype=y0.dtype),
                                       c_error=self.tableau.c_error.to(device=device, dtype=y0.dtype))
        self.mid = self.mid.to(device=device, dtype=y0.dtype)

    @classmethod
    def valid_events(cls):
        return super(RKAdaptiveStepsizeODESolver, cls).valid_events() | {'event_step',
                                                                         'event_accept_step',
                                                                         'event_reject_step'}

    @staticmethod
    def _flip_option(adjoint_options, option_name):
        # We assume that step_locations etc. are given to us ordered in the same direction as for the forward pass (for
        # compatibility with the default adjoint_options=options), so we need to flip them around here.
        try:
            option_value = adjoint_options[option_name]
        except KeyError:
            pass
        else:
            if torch.is_tensor(option_value) and option_value.ndimension() > 0:
                adjoint_options[option_name] = option_value.flip(0)
            # else: an error will be raised when the option is attempted to be used in __init__, but we defer raising
            # the error until then to keep things tidy.

    @staticmethod
    def _adjoint_norm(y0, shapes, adjoint_params, state_norm, param_norm):
        # By default use a mixed L-infinity/RMS norm over the input, where we treat t, each element of y, and each
        # element of adj_y separately over the Linf, but consider all the parameters together. (Done for backward
        # compatibility.)

        if shapes is None:
            shapes = [y0.shape]
        # adj_t, y, adj_y, adj_params, corresponding to the order in aug_state
        norm_fns = [_rms_norm] + [state_norm for _ in range(2 * len(shapes))] + [param_norm]
        adjoint_shapes = [torch.Size(())] + shapes + shapes + [torch.Size([sum(param.numel()
                                                                               for param in adjoint_params)])]
        return _mixed_norm(norm_fns, adjoint_shapes)

    @classmethod
    def adjoint_options_from_options(cls, shapes, y0, options, adjoint_params):
        adjoint_options = super(AdaptiveStepsizeODESolver, cls).adjoint_options_from_options(shapes=shapes,
                                                                                             y0=y0,
                                                                                             options=options,
                                                                                             adjoint_params=adjoint_params)

        cls._flip_option(adjoint_options, 'grid_points')
        cls._flip_option(adjoint_options, 'step_locations')
        cls._flip_option(adjoint_options, 'jump_locations')

        if 'norm' in adjoint_options:
            adjoint_options["norm"] = cls._adjoint_norm(y0, shapes, adjoint_params, adjoint_options['norm'], _rms_norm)
        return adjoint_options

    @classmethod
    def adjoint_options_from_adjoint_options(cls, shapes, y0, options, adjoint_options, adjoint_params):
        adjoint_options = super(AdaptiveStepsizeODESolver, cls).adjoint_options_from_adjoint_options(shapes=shapes,
                                                                                                     y0=y0,
                                                                                                     options=options,
                                                                                                     adjoint_options=adjoint_options,
                                                                                                     adjoint_params=adjoint_params)

        cls._flip_option(adjoint_options, 'grid_points')
        cls._flip_option(adjoint_options, 'step_locations')
        cls._flip_option(adjoint_options, 'jump_locations')

        if 'norm' in options:
            state_norm = _rms_norm
        else:
            state_norm = options['norm']
        if 'norm' not in adjoint_options:
            adjoint_options['norm'] = cls._adjoint_norm(y0, shapes, adjoint_params, state_norm, _rms_norm)
        elif adjoint_options['norm'] == 'seminorm':
            adjoint_options['norm'] = cls._adjoint_norm(y0, shapes, adjoint_params, state_norm, _zero_norm)

        return adjoint_options

    def _before_integrate(self, t):
        f0 = self.func(t[0], self.y0)
        if self.first_step is None:
            with torch.no_grad():
                first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol,
                                                  self.norm, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, [self.y0] * 5)
        self.next_step_index = min(bisect.bisect(self.step_locations.tolist(), t[0]), len(self.step_locations) - 1)
        self.next_jump_index = min(bisect.bisect(self.jump_locations.tolist(), t[0]), len(self.jump_locations) - 1)

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
        self.func.event_step(t0, y0, dt)
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
        assert t1 > t0, 'underflow in dt {}'.format(dt.item())
        assert torch.isfinite(y0).all(), 'non-finite values in state `y`: {}'.format(y0)

        ########################################################
        #     Make step, respecting prescribed grid points     #
        ########################################################

        on_step_loc = False
        if len(self.step_locations):
            next_step_location = self.step_locations[self.next_step_index]
            on_step_loc = t0 < next_step_location < t0 + dt
            if on_step_loc:
                t1 = next_step_location
                dt = t1 - t0

        on_jump_loc = False
        if len(self.jump_locations):
            next_jump_location = self.jump_locations[self.next_jump_index]
            on_jump_loc = t0 < next_jump_location < t0 + dt
            if on_jump_loc:
                on_step_loc = False
                t1 = next_jump_location
                dt = t1 - t0

        # Must be arranged as doing all the step_locations handling, then all the jump_locations handling, in case we
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
        with torch.no_grad():
            error_ratio = _compute_error_ratio(y1_error, self.rtol, self.atol, y0, y1, self.norm)
        accept_step = error_ratio <= 1
        # dtypes:
        # error_ratio.dtype == self.dtype

        ########################################################
        #                   Update RK State                    #
        ########################################################
        if accept_step:
            self.func.event_accept_step(t0, y0, dt)
            t_next = t1
            y_next = y1
            interp_coeff = self._interp_fit(y0, y_next, k, dt)
            if on_step_loc:
                if self.next_step_index != len(self.step_locations) - 1:
                    self.next_step_index += 1
            if on_jump_loc:
                if self.next_jump_index != len(self.jump_locations) - 1:
                    self.next_jump_index += 1
                # We've just passed a discontinuity in f; we should update f to match the side of the discontinuity
                # we're now on.
                f1 = self.func(t_next, y_next, True)
            f_next = f1
        else:
            self.func.event_reject_step(t0, y0, dt)
            t_next = t0
            y_next = y0
            f_next = f0
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
