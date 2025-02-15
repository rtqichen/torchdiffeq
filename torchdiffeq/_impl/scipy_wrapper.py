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

    def integrate(self, t):
        if t.numel() == 1:
            return torch.tensor(self.y0)[None].to(self.device, self.dtype)
        t_np = t.detach().cpu().numpy()
        sol = solve_ivp(
            self.func,
            t_span=[t_np.min(), t_np.max()],
            y0=self.y0,
            t_eval=t_np,
            method=self.solver,
            rtol=self.rtol,
            atol=self.atol
        )
        sol_tensor = torch.tensor(sol.y, requires_grad=True).T.to(self.device, self.dtype)
        sol_tensor = sol_tensor.reshape(-1, *self.shape)
        return sol_tensor
        
    @classmethod
    def valid_callbacks(cls):
        return set()


class RK45Solver(ScipyWrapperODESolver):
    """Explicit Runge-Kutta method of order 5(4)."""
    def __init__(self, func, y0, rtol=1e-7, atol=1e-9, **kwargs):
        super().__init__(func, y0, rtol, atol, solver="RK45", **kwargs)


class DOP853Solver(ScipyWrapperODESolver):
    """Explicit Runge-Kutta method of order 8."""
    def __init__(self, func, y0, rtol=1e-7, atol=1e-9, **kwargs):
        super().__init__(func, y0, rtol, atol, solver="DOP853", **kwargs)


class Radau(ScipyWrapperODESolver):
    """Implicit Runge-Kutta method of the Radau IIA family of order 5."""
    def __init__(self, func, y0, rtol=1e-7, atol=1e-9, **kwargs):
        super().__init__(func, y0, rtol, atol, solver="Radau", **kwargs)


class BDF(ScipyWrapperODESolver):
    """Implicit multi-step variable-order method based on backward differentiation formula."""
    def __init__(self, func, y0, rtol=1e-7, atol=1e-9, **kwargs):
        super().__init__(func, y0, rtol, atol, solver="BDF", **kwargs)


def convert_func_to_numpy(func, shape, device, dtype):
    def np_func(t, y):
        # Convert numpy inputs to torch tensors with requires_grad=True
        t_tensor = torch.tensor(t, dtype=dtype, device=device, requires_grad=True)
        y_tensor = torch.reshape(torch.tensor(y, dtype=dtype, device=device, requires_grad=True), shape)
        
        # Compute function value with gradients
        f = func(t_tensor, y_tensor)
        
        # Create gradient checkpoint to save memory
        def grad_checkpoint(t, y):
            return func(t, y)
            
        # Use gradient checkpointing for better memory efficiency
        f = torch.utils.checkpoint.checkpoint(grad_checkpoint, t_tensor, y_tensor)
        
        # Convert back to numpy while preserving gradient information
        return f.cpu().detach().numpy().reshape(-1)

    return np_func
