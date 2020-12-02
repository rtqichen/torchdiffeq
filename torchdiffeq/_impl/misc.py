import math
import numpy as np
import torch
import warnings


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _handle_deprecated_kwargs(solver, kwargs, kwarg, message):
    try:
        kwargs.pop(kwarg)
    except KeyError:
        pass
    else:
        warnings.warn('{}: {}'.format(solver.__class__.__name__, message))


def _linf_norm(tensor):
    return tensor.max()


def _rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()


def _zero_norm(tensor):
    return 0.


def _mixed_norm(tensor_tuple):
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


def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = error_ratio.to(last_step.dtype)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))


class _StitchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


if hasattr(torch, "nextafter"):
    def _nextafter(x1, x2):
        out = torch.nextafter(x1, x2).detach()
        return _StitchGradient.apply(x1, out)
else:
    # TODO: remove once PyTorch 1.7.0 becomes obligatory + remove comment to that effect in FURTHER_DOCUMENTATION.md
    def _nextafter(x1, x2):
        warnings.warn("torch.nextafter is only available in PyTorch 1.7 or newer. Falling back to numpy.nextafter.")
        x1_np = x1.detach().cpu().numpy()
        x2_np = x2.detach().cpu().numpy()
        out = torch.as_tensor(np.nextafter(x1_np, x2_np), device=x1.device)
        return _StitchGradient.apply(x1, out)


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


def _shape_to_flat(tensor):
    return torch.cat([tensor_.reshape(-1) for tensor_ in tensor])


_inf = torch.tensor(math.inf)
_neginf = torch.tensor(-math.inf)


class _WrapFunc(torch.nn.Module):
    def __init__(self, base_func, is_tuple, is_reversed, shapes):
        super(_WrapFunc, self).__init__()
        self.base_func = base_func
        self.is_tuple = is_tuple
        self.is_reversed = is_reversed
        self.shapes = shapes

    def forward(self, t, y, perturb=None):
        t = t.to(y.dtype)
        if perturb is True:
            t = _nextafter(t, _inf)
        elif perturb is False:
            t = _nextafter(t, _neginf)

        if self.is_reversed:
            t = -t
        if self.is_tuple:
            y = _flat_to_shape(y, (), self.shapes)

        f = self.base_func(t, y)

        if self.is_tuple:
            f = _shape_to_flat(f)
        if self.is_reversed:
            f = -f

        return f


_all_callback_names = ['callback_step', 'callback_accept_step', 'callback_reject_step']
_all_adjoint_callback_names = [name + '_adjoint' for name in _all_callback_names]
_null_callback = lambda *args, **kwargs: None


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)
    return timelike


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
