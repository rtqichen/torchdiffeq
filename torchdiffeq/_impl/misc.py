import torch
import warnings

SYMPLECTIC = ['yoshida4th']

def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _rms_norm(inputs):
    return inputs.pow(2).mean().sqrt()


def _mixed_linf_rms_norm(shapes):
    def _norm(inputs):
        total = 0
        out = []
        for shape in shapes:
            next_total = total + shape.numel()
            out.append(_rms_norm(inputs[total:next_total]))
            total = next_total
        assert total == inputs.numel(), "Shapes do not total to the full size of the tensor."
        return max(out)
    return _norm


def _wrap_norm(norm_fns, shapes):
    def _norm(inputs):
        total = 0
        out = []
        for i, shape in enumerate(shapes):
            next_total = total + shape.numel()
            if i < len(norm_fns):
                out.append(norm_fns[i](inputs[total:next_total]))
            else:
                out.append(_rms_norm(inputs[total:next_total]))
            total = next_total
        assert total == inputs.numel(), "Shapes do not total to the full size of the tensor."
        return max(out)
    return _norm


def _linf_norm(inputs):
    return inputs.max()


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


def _flat_to_shape(inputs, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(inputs[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


def _flat_to_shape_symplectic(inputs, length, shapes):
    tensor_list = []
    total = 0
    tensor_l, tensor_r = torch.chunk(inputs,2,dim=-1)
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_l_ = tensor_l[..., total:next_total].view((*length, *shape))
        tensor_r_ = tensor_r[..., total:next_total].view((*length, *shape))
        tensor_ = torch.cat([tensor_l_,tensor_r_],dim=-1)
        tensor_list.append(tensor_)
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


class _TupleFuncSymplectic(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFuncSymplectic, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape_symplectic(y, (), self.shapes))
        shapes, f = _reform_inputs_for_symplectic(f)
        return f


class _ReverseFuncSymplectic(torch.nn.Module):
    def __init__(self, base_func):
        super(_ReverseFuncSymplectic, self).__init__()
        self.base_func = base_func

    def forward(self, t, y):
        return self.base_func(-t, y)


def _reform_inputs_for_symplectic(inputs):
    shapes = []
    counter = 0
    for counter, tensor_ in enumerate(inputs):
        tensor_l_, tensor_r_ = torch.chunk(tensor_,2,dim=-1)
        if counter == 0:
            tensor_l = tensor_l_.reshape(-1)
            tensor_r = tensor_r_.reshape(-1)
        else:
            tensor_l = torch.cat([tensor_l, tensor_l_.reshape(-1)])
            tensor_r = torch.cat([tensor_r, tensor_r_.reshape(-1)])
        shapes.append(tensor_l_.shape)
    return shapes, torch.cat([tensor_l, tensor_r])


def _check_inputs(func, y0, t, rtol, atol, method, options, SOLVERS):
    # Normalise to inputs (non-tupled) input
    shapes = None
    if not torch.is_tensor(y0):
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        if method in SYMPLECTIC:
            shapes, y0 = _reform_inputs_for_symplectic(y0)
            func = _TupleFuncSymplectic(func, shapes)
        else:
            shapes = [y0_.shape for y0_ in y0]
            y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
            func = _TupleFunc(func, shapes)
        rtol = _tuple_tol('rtol', rtol, shapes)
        atol = _tuple_tol('atol', atol, shapes)

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

    try:
        grid_points = options['grid_points']
    except KeyError:
        pass
    else:
        assert torch.is_tensor(grid_points), 'grid_points must be a torch.Tensor'
        _assert_one_dimensional('grid_points', grid_points)
        assert not grid_points.requires_grad, "grid_points cannot require gradient"
        _assert_floating('grid_points', grid_points)

    if 'norm' not in options:
        if shapes is None:
            # L2 norm over a single input
            options['norm'] = _rms_norm
        else:
            # Mixed Linf/L2 norm over tupled input (chosen mostly just for backward compatibility reasons)
            options['norm'] = _mixed_linf_rms_norm(shapes)

    # Normalise time
    assert torch.is_tensor(t), 't must be a torch.Tensor'
    _assert_one_dimensional('t', t)
    _assert_floating('t', t)
    if _decreasing(t):
        t = -t
        if method in SYMPLECTIC:
            func = _ReverseFuncSymplectic(func)
        else:
            func = _ReverseFunc(func)
        try:
            grid_points = options['grid_points']
        except KeyError:
            pass
        else:
            options['grid_points'] = -grid_points

    # Can only do after having normalised time
    _assert_increasing('t', t)
    try:
        grid_points = options['grid_points']
    except KeyError:
        pass
    else:
        _assert_increasing('grid_points', grid_points)

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

    return shapes, func, y0, t, rtol, atol, method, options


