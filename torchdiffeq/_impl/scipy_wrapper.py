import abc
import torch
from scipy.integrate import solve_ivp
from .misc import _handle_unused_kwargs


class ScipyWrapperODESolver(metaclass=abc.ABCMeta):

    def __init__(self, func, y0, rtol, atol, min_step=0, max_step=float('inf'), solver="LSODA", **unused_kwargs):
        unused_kwargs.pop('norm', None)
        unused_kwargs.pop('grid_points', None)
        unused_kwargs.pop('eps', None)
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.dtype = y0.dtype
        self.device = y0.device
        self.shape = y0.shape
        self.y0 = y0.detach().cpu().numpy().reshape(-1)
        self.rtol = rtol
        self.atol = atol
        self.min_step = min_step
        self.max_step = max_step
        self.solver = solver
        self.func = convert_func_to_numpy(func, self.shape, self.device, self.dtype)

    def integrate(self, t, info=False):
        if t.numel() == 1:
            return torch.tensor(self.y0)[None].to(self.device, self.dtype)
        t = t.detach().cpu().numpy()
        oderesult = solve_ivp(
            self.func,
            t_span=[t.min(), t.max()],
            y0=self.y0,
            t_eval=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
            min_step=self.min_step,
            max_step=self.max_step
        )
        sol = torch.tensor(oderesult.y).T.to(self.device, self.dtype)
        sol = sol.reshape(-1, *self.shape)
        if info:
            return sol, oderesult
        else:
            return sol


def convert_func_to_numpy(func, shape, device, dtype):

    def np_func(t, y):
        t = torch.tensor(t).to(device, dtype)
        y = torch.reshape(torch.tensor(y).to(device, dtype), shape)
        with torch.no_grad():
            f = func(t, y)
        return f.detach().cpu().numpy().reshape(-1)

    return np_func
