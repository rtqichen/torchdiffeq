import torch
from scipy.integrate import solve_ivp
from .solvers import Solver


_backprop_error = "'{}' requires gradient but SciPy solvers do not support backpropagation. " \
                  "Try `odeint_adjoint` instead."


class ScipyWrapperODESolver(Solver):

    def __init__(self, rtol, atol, solver="LSODA", **kwargs):
        super(ScipyWrapperODESolver, self).__init__(rtol=rtol, atol=atol, **kwargs)

        self.rtol = rtol
        self.atol = atol
        self.solver = solver

    def integrate(self, func, y0, t):
        if t.requires_grad:
            raise ValueError(_backprop_error.format('t'))
        if y0.requires_grad:
            raise ValueError(_backprop_error.format('y0'))

        shape = y0.shape
        dtype = y0.dtype
        device = y0.device

        func = convert_func_to_numpy(func, shape, device, dtype)

        if t.numel() == 1:
            return torch.as_tensor(y0, device=device, dtype=dtype)[None]

        t = t.cpu().numpy()
        y0 = y0.cpu().numpy().reshape(-1)

        sol = solve_ivp(
            func,
            t_span=[t.min(), t.max()],
            y0=y0,
            t_eval=t,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol,
        )
        sol = torch.as_tensor(sol.y, dtype=dtype, device=device).T
        sol = sol.reshape(-1, *shape)
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
