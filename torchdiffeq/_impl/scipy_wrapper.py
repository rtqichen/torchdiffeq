import torch
from scipy.integrate import solve_ivp
from .solvers import Solver


_backprop_error = "'{}' required gradient but SciPy solvers do not support backpropagation. " \
                  "Try `odeint_adjoint` instead."


class ScipyWrapperODESolver(Solver):

    def __init__(self, rtol, atol, solver="LSODA", **kwargs):
        super(ScipyWrapperODESolver, self).__init__(rtol=rtol, atol=atol, **kwargs)

        if self.y0.requires_grad:
            raise ValueError(_backprop_error.format('y0'))

        self.dtype = self.y0.dtype
        self.device = self.y0.device
        self.shape = self.y0.shape
        self.y0 = self.y0.cpu().numpy().reshape(-1)
        self.rtol = rtol
        self.atol = atol
        self.solver = solver
        self.func = convert_func_to_numpy(self.func, self.shape, self.device, self.dtype)

    def integrate(self, t):
        if t.requires_grad:
            raise ValueError(_backprop_error.format('t'))
        if t.numel() == 1:
            return torch.as_tensor(self.y0, device=self.device, dtype=self.dtype)[None]
        t = t.cpu().numpy()
        sol = solve_ivp(
            self.func,
            t_span=[t.min(), t.max()],
            y0=self.y0,
            t_eval=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        sol = torch.as_tensor(sol.y, dtype=self.dtype, device=self.device).T
        sol = sol.reshape(-1, *self.shape)
        return sol


def convert_func_to_numpy(func, shape, device, dtype):

    def np_func(t, y):
        t = torch.as_tensor(t, dtype=dtype, device=device)
        y = torch.as_tensor(y, dtype=dtype, device=device).reshape(shape)
        f = func(t, y)
        if f.requires_grad:
            raise ValueError(_backprop_error.format('func(t, y)'))
        return f.detach().cpu().numpy().reshape(-1)

    return np_func
