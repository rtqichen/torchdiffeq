from enum import Enum
import math
import numpy as np
import torch
import warnings
from .event_handling import combine_event_functions


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _linf_norm(tensor):
    return tensor.max()


def _rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def _zero_norm(tensor):
    return 0.


def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.
    return max([_rms_norm(tensor) for tensor in tensor_tuple])


def _select_initial_step(func, t0, y0, order, rtol, atol, norm, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """

    dtype = y0.dtype
    device = y0.device
    t_dtype = t0.dtype
    t0 = t0.to(dtype)

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + torch.abs(y0) * rtol

    d0 = norm(y0 / scale)
    d1 = norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=dtype, device=device)
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=dtype, device=device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))

    return torch.min(100 * h0, h1).to(t_dtype)


def _compute_error_ratio(error_estimate, rtol, atol, y0, y1, norm):
    error_tol = atol + rtol * torch.max(y0.abs(), y1.abs())
    return norm(error_estimate / error_tol)


@torch.no_grad()
def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.type_as(last_step)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _assert_one_dimensional(name, t):
    assert t.ndimension() == 1, "{} must be one dimensional".format(name)


def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [torch.as_tensor(tol_).expand(shape.numel()) for tol_, shape in zip(tol, shapes)]
    return torch.cat(tol)


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])


class _TupleInputOnlyFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleInputOnlyFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))


class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)


class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2


class _PerturbFunc(torch.nn.Module):

    def __init__(self, base_func):
        super(_PerturbFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        t = t.to(y.dtype)
        if perturb is Perturb.NEXT:
            # Replace with next smallest representable value.
            t = _nextafter(t, t + 1)
        elif perturb is Perturb.PREV:
            # Replace with prev largest representable value.
            t = _nextafter(t, t - 1)
        else:
            # Do nothing.
            pass
        return self.base_func(t, y)


def _check_inputs(func, y0, t, rtol, atol, method, options, event_fn, SOLVERS):

    if event_fn is not None:
        if len(t) != 2:
            raise ValueError(f"We require len(t) == 2 when in event handling mode, but got len(t)={len(t)}.")

        # Combine event functions if the output is multivariate.
        event_fn = combine_event_functions(event_fn, t[0], y0)

    # Normalise to tensor (non-tupled) input
    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
        if event_fn is not None:
            event_fn = _TupleInputOnlyFunc(event_fn, shapes)
    _assert_floating('y0', y0)

    # Normalise method and options
    if options is None:
        options = {}
    else:
        options = options.copy()
    if method is None:
        method = 'dopri5'
    if method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(method,
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))

    if is_tuple:
        # We accept tupled input. This is an abstraction that is hidden from the rest of odeint (exception when
        # returning values), so here we need to maintain the abstraction by wrapping norm functions.

        if 'norm' in options:
            # If the user passed a norm then get that...
            norm = options['norm']
        else:
            # ...otherwise we default to a mixed Linf/L2 norm over tupled input.
            norm = _mixed_norm

        # In either case, norm(...) is assumed to take a tuple of tensors as input. (As that's what the state looks
        # like from the point of view of the user.)
        # So here we take the tensor that the machinery of odeint has given us, and turn it in the tuple that the
        # norm function is expecting.
        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)
        options['norm'] = _norm

    else:
        if 'norm' in options:
            # No need to change the norm function.
            pass
        else:
            # Else just use the default norm.
            # Technically we don't need to set that here (RKAdaptiveStepsizeODESolver has it as a default), but it
            # makes it easier to reason about, in the adjoint norm logic, if we know that options['norm'] is
            # definitely set to something.
            options['norm'] = _rms_norm

    # Normalise time
    _check_timelike('t', t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        t_is_reversed = True

    if t_is_reversed:
        # Change the integration times to ascending order.
        # We do this by negating the time values and all associated arguments.
        t = -t

        # Ensure time values are un-negated when calling functions.
        func = _ReverseFunc(func, mul=-1.0)
        if event_fn is not None:
            event_fn = _ReverseFunc(event_fn)

        # For fixed step solvers.
        try:
            _grid_constructor = options['grid_constructor']
        except KeyError:
            pass
        else:
            options['grid_constructor'] = lambda func, y0, t: -_grid_constructor(func, y0, -t)

        # For RK solvers.
        _flip_option(options, 'step_t')
        _flip_option(options, 'jump_t')

    # Can only do after having normalised time
    _assert_increasing('t', t)

    # Tol checking
    if torch.is_tensor(rtol):
        assert not rtol.requires_grad, "rtol cannot require gradient"
    if torch.is_tensor(atol):
        assert not atol.requires_grad, "atol cannot require gradient"

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

    # Add perturb argument to func.
    func = _PerturbFunc(func)

    return shapes, func, y0, t, rtol, atol, method, options, event_fn, t_is_reversed


class _StitchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _nextafter(x1, x2):
    with torch.no_grad():
        if hasattr(torch, "nextafter"):
            out = torch.nextafter(x1, x2)
        else:
            out = np_nextafter(x1, x2)
    return _StitchGradient.apply(x1, out)


def np_nextafter(x1, x2):
    warnings.warn("torch.nextafter is only available in PyTorch 1.7 or newer."
                  "Falling back to numpy.nextafter. Upgrade PyTorch to remove this warning.")
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    out = torch.tensor(np.nextafter(x1_np, x2_np)).to(x1)
    return out


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)


def _flip_option(options, option_name):
    try:
        option_value = options[option_name]
    except KeyError:
        pass
    else:
        if isinstance(option_value, torch.Tensor):
            options[option_name] = -option_value
        # else: an error will be raised when the option is attempted to be used in Solver.__init__, but we defer raising
        # the error until then to keep things tidy.
