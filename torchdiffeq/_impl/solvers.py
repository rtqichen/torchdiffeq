import abc
import bisect
import torch
from .interp import _interp_evaluate, _interp_fit
from .rk_common import _ButcherTableau, _RungeKuttaState, _runge_kutta_step
from .misc import (_compute_error_ratio,
                   _error_tol,
                   _expand_as,
                   _handle_unused_kwargs,
                   _scaled_dot_product,
                   _select_initial_step,
                   _optimal_step_size)


# TODO: this is assuming instance attributes y0 and dtype
class AdaptiveStepsizeODESolver(metaclass=abc.ABCMeta):
    def _before_integrate(self, t):
        pass

    @abc.abstractmethod
    def _advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        solution = [self.y0]
        t = t.to(self.dtype)
        self._before_integrate(t)
        for i in range(1, len(t)):
            y = self._advance(t[i])
            solution.append(y)
        return tuple(map(torch.stack, tuple(zip(*solution))))


class FixedGridODESolver(metaclass=abc.ABCMeta):
    order: int

    def __init__(self, func, y0, step_size=None, grid_constructor=None, eps=0, **unused_kwargs):
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.eps = eps

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @staticmethod
    def _grid_constructor_from_step_size(step_size):
        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters, dtype=t.dtype, device=t.device) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    @abc.abstractmethod
    def _step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = [self.y0]

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self._step_func(self.func, t0, t1 - t0, y0)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))

            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1
            y0 = y1

        return tuple(map(torch.stack, tuple(zip(*solution))))

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))


class RKAdaptiveStepsizeODESolver(AdaptiveStepsizeODESolver):
    order: int
    tableau: _ButcherTableau
    mid: list

    def __init__(self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2,
                 max_num_steps=2 ** 31 - 1, grid_points=None, eps=0., dtype=torch.float64, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        # We use mixed-precision. y has its original dtype (probably float32), whilst all 'time'-like objects use
        # `dtype` (defaulting to float64).
        dtype = torch.promote_types(dtype, y0[0].dtype)  # in case of "float128" etc. in the future
        device = y0[0].device

        self.func = lambda t, y: func(t.type_as(y[0]), y)
        self.y0 = y0
        self.rtol = _expand_as(rtol, y0)
        self.atol = _expand_as(atol, y0)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.safety = torch.as_tensor(safety, dtype=dtype, device=device)
        self.ifactor = torch.as_tensor(ifactor, dtype=dtype, device=device)
        self.dfactor = torch.as_tensor(dfactor, dtype=dtype, device=device)
        self.max_num_steps = torch.as_tensor(max_num_steps, dtype=torch.int32, device=device)
        grid_points = torch.tensor([], dtype=dtype, device=device) if grid_points is None else grid_points.to(dtype)
        self.grid_points = grid_points
        self.eps = torch.as_tensor(eps, dtype=dtype, device=device)
        self.dtype = dtype

    def _before_integrate(self, t):
        f0 = self.func(t[0], self.y0)
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, self.order - 1, self.rtol, self.atol, f0=f0)
        else:
            first_step = self.first_step
        self.rk_state = _RungeKuttaState(self.y0, f0, t[0], t[0], first_step, self._init_interp_coeff())
        self.next_grid_index = min(bisect.bisect(self.grid_points.tolist(), t[0]), len(self.grid_points) - 1)

    # TODO: remove?
    def _init_interp_coeff(self):
        return [self.y0] * 5

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

        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert torch.isfinite(y0_).all(), 'non-finite values in state `y`: {}'.format(y0_)

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

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_tol = _error_tol(self.rtol, self.atol, y0, y1)
        mean_sq_error_ratio = _compute_error_ratio(y1_error, error_tol)
        accept_step = max(mean_sq_error_ratio) <= 1

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
        dt_next = _optimal_step_size(dt, mean_sq_error_ratio, self.safety, self.ifactor, self.dfactor, self.order)
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, interp_coeff)
        return rk_state

    # TODO: remove?
    def _interp_fit(self, y0, y1, k, dt):
        """Fit an interpolating polynomial to the results of a Runge-Kutta step."""
        dt = dt.type_as(y0[0])
        y_mid = tuple(y0_ + _scaled_dot_product(dt, self.mid, k_) for y0_, k_ in zip(y0, k))
        f0 = tuple(k_[0] for k_ in k)
        f1 = tuple(k_[-1] for k_ in k)
        return _interp_fit(y0, y1, y_mid, f0, f1, dt)
