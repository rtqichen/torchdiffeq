import math
import torch
import warnings


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _rms_norm(x):
    return x.norm() / math.sqrt(x.numel())


def _select_initial_step(func, t0, y0, order, rtol, atol, f0=None):
    """Empirically select a good initial step.

    The algorithm is described in [1]_.

    Parameters
    ----------
    func : callable
        Right-hand side of the system.
    t0 : float
        Initial value of the independent variable.
    y0 : ndarray, shape (n,)
        Initial value of the dependent variable.
    direction : float
        Integration direction.
    order : float
        Method order.
    rtol : float
        Desired relative tolerance.
    atol : float
        Desired absolute tolerance.

    Returns
    -------
    h_abs : float
        Absolute value of the suggested initial step.

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4.
    """

    if f0 is None:
        f0 = func(t0, y0)

    scale = atol + y0.abs() * rtol

    d0 = _rms_norm(y0 / scale)
    d1 = _rms_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=y0.dtype, device=y0.device)
    else:
        h0 = 0.01 * d0 / d1

    y1 = y0 + h0 * f0
    f1 = func(t0 + h0, y1)

    d2 = _rms_norm((f1 - f0) / scale) / h0

    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=y0.dtype, device=y0.device), h0 * 1e-3)
    else:
        h1 = (0.01 / (d1 + d2)) ** (1. / float(order + 1))

    return torch.min(100 * h0, h1).type_as(t0)


def _error_tol(rtol, atol, y0, y1):
    return atol + rtol * torch.max(y0.abs(), y1.abs())


def _l2_norm_squared(tensor):
    return [tensor.pow(2).mean()]


def _tuple_l2_norm_squared(shapes):
    def _tupled_norm(tensor):
        total = 0
        out = []
        for shape in shapes:
            next_total = total + shape.numel()
            out.append(tensor[total:next_total].pow(2).mean())
            total = next_total
        return out
    return _tupled_norm


def _compute_error_ratio(error_estimate, error_tol, norm):
    return norm(error_estimate / error_tol)


def _optimal_step_size(last_step, error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    error_ratio = max(error_ratio)
    if error_ratio == 0:
        return last_step * ifactor
    if error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = torch.sqrt(error_ratio).type_as(last_step)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor


def _tuple_tol(name, tol, shapes):
    try:
        iter(tol)
    except TypeError:
        return tol
    tol = tuple(tol)
    assert len(tol) == len(shapes), "If using tupled {} it must have the same length as the tuple y0".format(name)
    tol = [torch.as_tensor(tol_).repeat(shape.numel()) for tol_, shape in zip(tol, shapes)]
    return torch.cat(tol)


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _assert_one_dimensional(name, t):
    assert t.ndimension() == 1, "{} must be one dimensional".format(name)


def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)


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


class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y):
        return -self.base_func(-t, y)


def _check_inputs(func, y0, t, rtol, atol, options):
    # TODO: deprecate tupled input?
    tensor_input = True
    shapes = []
    if not torch.is_tensor(y0):
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        tensor_input = False
        shapes = [y0_.shape for y0_ in y0]
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
    if 'norm' not in options:
        if tensor_input:
            options['norm'] = _l2_norm_squared
        else:
            options['norm'] = _tuple_l2_norm_squared(shapes)
    # ~TODO

    if not torch.is_floating_point(y0):
        raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0.type()))

    assert torch.is_tensor(t), 't must be a torch.Tensor'
    _assert_one_dimensional('t', t)
    if _decreasing(t):
        t = -t
        func = _ReverseFunc(func)
        try:
            grid_points = options['grid_points']
        except (KeyError, TypeError):
            pass
        else:
            options = options.copy()
            options['grid_points'] = -grid_points

    _assert_increasing('t', t)
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    try:
        grid_points = options['grid_points']
    except (KeyError, TypeError):
        pass
    else:
        assert torch.is_tensor(grid_points), 'grid_points must be a torch.Tensor'
        _assert_one_dimensional('grid_points', grid_points)
        _assert_increasing('grid_points', grid_points)

    # TODO: deprecate inputs on different devices?
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~TODO

    return tensor_input, shapes, func, y0, t, rtol, atol, options
