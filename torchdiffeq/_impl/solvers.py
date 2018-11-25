import abc
import torch
from .misc import _assert_increasing, _handle_unused_kwargs


class AdaptiveStepsizeODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, atol, rtol, **unused_kwargs):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.atol = atol
        self.rtol = rtol

    def before_integrate(self, t):
        pass

    @abc.abstractmethod
    def advance(self, next_t):
        raise NotImplementedError

    def integrate(self, t):
        _assert_increasing(t)
        solution = [self.y0]
        t = t.to(self.y0[0].device, torch.float64)
        self.before_integrate(t)
        for i in range(1, len(t)):
            y = self.advance(t[i])
            solution.append(y)
        return tuple(map(torch.stack, tuple(zip(*solution))))


class FixedGridODESolver(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, func, y0, step_size=None, grid_constructor=None, **unused_kwargs):
        unused_kwargs.pop('rtol', None)
        unused_kwargs.pop('atol', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0

        if step_size is not None and grid_constructor is None:
            self.grid_constructor = self._grid_constructor_from_step_size(step_size)
        elif grid_constructor is None:
            self.grid_constructor = lambda f, y0, t: t
        else:
            raise ValueError("step_size and grid_constructor are exclusive arguments.")

    def _grid_constructor_from_step_size(self, step_size):

        def _grid_constructor(func, y0, t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0, niters).to(t) * step_size + start_time
            if t_infer[-1] > t[-1]:
                t_infer[-1] = t[-1]

            return t_infer

        return _grid_constructor

    @property
    @abc.abstractmethod
    def order(self):
        pass

    @abc.abstractmethod
    def step_func(self, func, t, dt, y):
        pass

    def integrate(self, t):
        _assert_increasing(t)
        t = t.type_as(self.y0[0])
        time_grid = self.grid_constructor(self.func, self.y0, t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]
        time_grid = time_grid.to(self.y0[0])

        solution = [self.y0]

        j = 1
        y0 = self.y0
        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dy = self.step_func(self.func, t0, t1 - t0, y0)
            y1 = tuple(y0_ + dy_ for y0_, dy_ in zip(y0, dy))
            y0 = y1

            while j < len(t) and t1 >= t[j]:
                solution.append(self._linear_interp(t0, t1, y0, y1, t[j]))
                j += 1

        return tuple(map(torch.stack, tuple(zip(*solution))))

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        t0, t1, t = t0.to(y0[0]), t1.to(y0[0]), t.to(y0[0])
        slope = tuple((y1_ - y0_) / (t1 - t0) for y0_, y1_, in zip(y0, y1))
        return tuple(y0_ + slope_ * (t - t0) for y0_, slope_ in zip(y0, slope))
