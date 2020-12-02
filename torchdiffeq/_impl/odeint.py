import torch
import warnings
from .adaptive_heun import AdaptiveHeunSolver
from .bosh3 import Bosh3Solver
from .dopri5 import Dopri5Solver
from .dopri8 import Dopri8Solver
from .fixed_adams import AdamsBashforth, AdamsBashforthMoulton
from .fixed_grid import Euler, Midpoint, RK4
from .rk_common import RKAdaptiveStepsizeODESolver
from .scipy_wrapper import ScipyWrapperODESolver
from .misc import (_flat_to_shape,
                   _shape_to_flat,
                   _tuple_tol,
                   _assert_floating,
                   _check_timelike,
                   _WrapFunc,
                   _all_callback_names,
                   _all_adjoint_callback_names,
                   _null_callback,
                   _flip_option,
                   _rms_norm,
                   _mixed_norm)

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
    'scipy_solver': ScipyWrapperODESolver,
}


def odeint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None):
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
    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options)
    solver = SOLVERS[method](rtol=rtol, atol=atol, state_dtype=y0.dtype, device=y0.device, **options)

    solution = solver.integrate(func, y0, t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution


def _check_inputs(func, y0, t, rtol, atol, method, options):
    # Normalise to tensor (non-tupled) input
    is_tuple = False
    shapes = None
    if not isinstance(y0, torch.Tensor):
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = _shape_to_flat(y0)
        is_tuple = True
    _assert_floating('y0', y0)

    # Normalise time
    t = _check_timelike('t', t, True)
    is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        is_reversed = True
        t = -t

    # Tol checking
    if isinstance(rtol, torch.Tensor):
        assert not rtol.requires_grad, "rtol cannot require gradient"
    if isinstance(atol, torch.Tensor):
        assert not atol.requires_grad, "atol cannot require gradient"

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

    wrapped_func = _WrapFunc(func, is_tuple, is_reversed, shapes)

    callback_names = set()
    for callback_name in _all_callback_names:
        try:
            callback = getattr(func, callback_name)
        except AttributeError:
            setattr(wrapped_func, callback_name, _null_callback)
        else:
            callback_names.add(callback_name)
            if is_reversed:
                # At the moment all callbacks have the arguments (t0, y0, dt).
                # This will need adjusting on a per-callback basis if that changes in the future.
                def callback(t0, *args, _callback=callback, **kwargs):
                    return _callback(-t0, *args, **kwargs)
            setattr(wrapped_func, callback_name, callback)
    for callback_name in _all_adjoint_callback_names:
        try:
            callback = getattr(func, callback_name)
        except AttributeError:
            pass
        else:
            setattr(wrapped_func, callback_name, callback)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'
                         .format(method, '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    invalid_callbacks = callback_names - SOLVERS[method].valid_callbacks()
    if len(invalid_callbacks) > 0:
        raise ValueError("Solver '{}' does not support callbacks {}.".format(method, invalid_callbacks))

    if is_reversed:
        try:
            _grid_constructor = options['grid_constructor']
        except KeyError:
            pass
        else:
            options['grid_constructor'] = lambda func, y0, t: -_grid_constructor(func, y0, -t)
        _flip_option(options, 'grid_points')
        _flip_option(options, 'step_t')
        _flip_option(options, 'jump_t')

    if issubclass(SOLVERS[method], RKAdaptiveStepsizeODESolver):
        if is_tuple:
            # We accept tupled input. This is an abstraction that is hidden from the rest of odeint (exception when
            # returning values), so here we need to maintain the abstraction by wrapping norm functions.

            try:
                # If the user passed a norm then get that...
                norm = options['norm']
            except KeyError:
                # ...otherwise we default to a mixed Linf/L2 norm over tupled input, for backward compatibility reasons
                norm = _mixed_norm
            finally:
                # In either case, norm(...) is assumed to take a tuple of tensors as input. (As that's what the state looks
                # like from the point of view of the user.)
                # So here we take the tensor that the machinery of odeint has given us, and turn it in the tuple that the
                # norm function is expecting.
                def _norm(tensor):
                    y = _flat_to_shape(tensor, (), shapes)
                    return norm(y)
                options['norm'] = _norm
        else:
            if 'norm' not in options:
                # Else just use the default norm.
                # Technically we don't need to set that here (RKAdaptiveStepsizeODESolver has it as a default), but it
                # makes it easier to reason about, in the adjoint norm logic, if we know that options['norm'] is
                # definitely set to something.
                options['norm'] = _rms_norm

    return shapes, wrapped_func, y0, t, rtol, atol, method, options
