import torch
from .dopri5 import Dopri5Solver
from .bosh3 import Bosh3Solver
from .adaptive_heun import AdaptiveHeunSolver
from .fixed_grid import Euler, Midpoint, RK4
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .dopri8 import Dopri8Solver
from .misc import _check_inputs, _flat_to_shape

SOLVERS = {
    'dopri8': Dopri8Solver,
    'dopri5': Dopri5Solver,
    'bosh3': Bosh3Solver,
    'adaptive_heun': AdaptiveHeunSolver,
    'euler': Euler,
    'midpoint': Midpoint,
    'rk4': RK4,
    'explicit_adams': AdamsBashforth,
    'implicit_adams': AdamsBashforthMoulton,
    # Backward compatibility: use the same name as before
    'fixed_adams': AdamsBashforthMoulton,
    # ~Backwards compatibility
}


def odeint(func, y0, t, *, rtol=1e-7, atol=1e-9, method=None, options=None, event_fn=None):
    """Integrate a system of ordinary differential equations.

    Solves the initial value problem for a non-stiff system of first order ODEs:
        ```
        dy/dt = func(t, y), y(t[0]) = y0
        ```
    where y is a Tensor or tuple of Tensors of any shape.

    Output dtypes and numerical precision are based on the dtypes of the inputs `y0`.

    Args:
        func: Function that maps a scalar Tensor `t` and a Tensor holding the state `y`
            into a Tensor of state derivatives with respect to time. Optionally, `y`
            can also be a tuple of Tensors.
        y0: N-D Tensor giving starting value of `y` at time point `t[0]`. Optionally, `y0`
            can also be a tuple of Tensors.
        t: 1-D Tensor holding a sequence of time points for which to solve for
            `y`. The initial time point should be the first element of this sequence,
            and each time must be larger than the previous time.
        rtol: optional float64 Tensor specifying an upper bound on relative error,
            per element of `y`.
        atol: optional float64 Tensor specifying an upper bound on absolute error,
            per element of `y`.
        method: optional string indicating the integration method to use.
        options: optional dict of configuring options for the indicated integration
            method. Can only be provided if a `method` is explicitly set.

    Returns:
        y: Tensor, where the first dimension corresponds to different
            time points. Contains the solved value of y for each desired time point in
            `t`, with the initial value `y0` being the first element along the first
            dimension.

    Raises:
        ValueError: if an invalid `method` is provided.
    """
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS)

    solver = SOLVERS[method](func=func, y0=y0, rtol=rtol, atol=atol, **options)

    if event_fn is None:
        solution = solver.integrate(t)
    else:
        event_t, solution = solver.integrate_until_event(t[0], event_fn)
        event_t = event_t.to(t).detach()
        if decreasing_time:
            event_t = -event_t

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    if event_fn is None:
        return solution
    else:
        return event_t, solution


def odeint_event(func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs):
    assert isinstance(event_fn, torch.nn.Module)

    if reverse_time:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() - 1.0])
    else:
        t = torch.cat([t0.reshape(-1), t0.reshape(-1).detach() + 1.0])

    event_t, solution = odeint_interface(func, y0, t, event_fn=event_fn, **kwargs)

    # Dummy values for rtol, atol, method, and options.
    shapes, func, y0, t, rtol, atol, method, options, event_fn, decreasing_time = _check_inputs(func, y0, t, 1e-7, 1e-9, None, None, event_fn, SOLVERS)

    if shapes is not None:
        state_t = torch.cat([s[-1].reshape(-1) for s in solution])
    else:
        state_t = solution[-1]

    event_t = ImplicitFnGradientRerouting.apply(func, event_fn, event_t, state_t, *event_fn.parameters())
    return event_t, solution


class ImplicitFnGradientRerouting(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, event_fn, event_t, state_t, *params):
        """ event_t is the solution to event_fn """
        ctx.func = func
        ctx.event_fn = event_fn
        ctx.save_for_backward(event_t, state_t, *params)

        return event_t.detach()

    @staticmethod
    def backward(ctx, grad_t):
        func = ctx.func
        event_fn = ctx.event_fn
        event_t, state_t, *params = ctx.saved_tensors

        with torch.enable_grad():

            event_t.requires_grad_(True)
            state_t.requires_grad_(True)
            c = event_fn(event_t, state_t)
            par_dt, dstate = torch.autograd.grad(c, (event_t, state_t), allow_unused=True, retain_graph=True)
            par_dt = torch.zeros_like(event_t) if par_dt is None else par_dt
            dstate = torch.zeros_like(dstate) if dstate is None else dstate
            dcdt = par_dt + torch.sum(dstate * func(event_t, state_t))

            params = tuple(p.requires_grad_(True) for p in params)
            grad_state_t, *grad_params = torch.autograd.grad(c, (state_t,) + params, -1 / (dcdt + 1e-12), allow_unused=True, retain_graph=True)

        return None, None, None, grad_state_t, *grad_params
