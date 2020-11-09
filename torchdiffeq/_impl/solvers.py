import abc
import torch
from .misc import _handle_unused_kwargs


class Solver(metaclass=abc.ABCMeta):
    def __init__(self, **unused_kwargs):
        super(Solver, self).__init__()
        unused_kwargs.pop('rtol')
        unused_kwargs.pop('atol')
        unused_kwargs.pop('state_dtype')
        unused_kwargs.pop('device')
        unused_kwargs.pop('shapes')
        unused_kwargs.pop('is_reversed')
        _handle_unused_kwargs(self, unused_kwargs)

    @classmethod
    def valid_events(cls):
        return set()

    @classmethod
    def adjoint_options_from_options(cls, shapes, y0, options, adjoint_params):
        if options is None:
            return {}
        return options.copy()

    @classmethod
    def adjoint_options_from_adjoint_options(cls, shapes, y0, options, adjoint_options, adjoint_params):
        if adjoint_options is None:
            return {}
        return adjoint_options.copy()

    @abc.abstractmethod
    def integrate(self, func, y0, t):
        raise NotImplementedError


class AdaptiveStepsizeODESolver(Solver):
    def __init__(self, dtype, **kwargs):
        super(AdaptiveStepsizeODESolver, self).__init__(**kwargs)
        self.dtype = dtype

    def _before_integrate(self, func, y0, t):
        pass

    @abc.abstractmethod
    def _advance(self, func, next_t):
        raise NotImplementedError

    def integrate(self, func, y0, t):
        solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
        solution[0] = y0
        t = t.to(self.dtype)
        self._before_integrate(func, y0, t)
        for i in range(1, len(t)):
            solution[i] = self._advance(func, t[i])
        return solution


class FixedGridODESolver(Solver):
    order: int

    def __init__(self, is_reversed, step_size=None, grid_constructor=None, **kwargs):
        super(FixedGridODESolver, self).__init__(is_reversed=is_reversed, **kwargs)

        if step_size is None:
            if grid_constructor is None:
                self.grid_constructor = lambda f, y0, t: t
            else:
                if is_reversed:
                    self.grid_constructor = lambda f, y0, t: -grid_constructor(f, y0, t)
                else:
                    self.grid_constructor = grid_constructor
        else:
            if grid_constructor is None:
                self.grid_constructor = self._grid_constructor_from_step_size(step_size)
            else:
                raise ValueError("step_size and grid_constructor are mutually exclusive arguments.")

    @classmethod
    def valid_events(cls):
        return super(FixedGridODESolver, cls).valid_events() | {'event_step'}

    @classmethod
    def adjoint_options_from_options(cls, shapes, y0, options, adjoint_params):
        adjoint_options = super(FixedGridODESolver, cls).adjoint_options_from_options(shapes=shapes,
                                                                                      y0=y0,
                                                                                      options=options,
                                                                                      adjoint_params=adjoint_params)
        if 'grid_constructor' in adjoint_options:
            _grid_constructor = adjoint_options['grid_constructor']
            adjoint_options['grid_constructor'] = lambda func, y0, t: _grid_constructor(func, y0, t).flip(0)
        return adjoint_options

    @classmethod
    def adjoint_options_from_adjoint_options(cls, shapes, y0, options, adjoint_options, adjoint_params):
        adjoint_options = super(FixedGridODESolver, cls).adjoint_options_from_adjoint_options(shapes=shapes,
                                                                                              y0=y0,
                                                                                              options=options,
                                                                                              adjoint_options=adjoint_options,
                                                                                              adjoint_params=adjoint_params)
        if 'grid_constructor' in adjoint_options:
            _grid_constructor = adjoint_options['grid_constructor']
            adjoint_options['grid_constructor'] = lambda func, y0, t: _grid_constructor(func, y0, t).flip(0)
        return adjoint_options

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
    def _step_func(self, func, t0, dt, t1, y):
        pass

    def integrate(self, func, y0, t):
        time_grid = self.grid_constructor(func, y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *y0.shape, dtype=y0.dtype, device=y0.device)
        solution[0] = y0

        j = 1
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            func.event_step(t0, y0, dt)
            dy = self._step_func(func, t0, dt, t1, y0)
            y1 = y0 + dy

            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, y0, y1, t[j])
                j += 1
            y0 = y1

        return solution

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)
