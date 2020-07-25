import math
import torch
import warnings


def _flatten(sequence):
    flat = [p.contiguous().view(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [
        p.contiguous().view(-1) if p is not None else torch.zeros_like(q).view(-1)
        for p, q in zip(sequence, like_sequence)
    ]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _possibly_nonzero(x):
    return isinstance(x, torch.Tensor) or x != 0


def _scaled_dot_product(scale, xs, ys):
    """Calculate a scaled, vector inner product between lists of Tensors."""
    # Using _possibly_nonzero lets us avoid wasted computation.
    return sum([(scale * x) * y for x, y in zip(xs, ys) if _possibly_nonzero(x) or _possibly_nonzero(y)])


def _dot_product(xs, ys):
    """Calculate the vector inner product between two lists of Tensors."""
    return sum([x * y for x, y in zip(xs, ys)])


def _error_tol(rtol, atol, y0, y1):
    return tuple(atol_ + rtol_ * torch.max(y0_.abs(), y1_.abs()) for atol_, rtol_, y0_, y1_ in zip(atol, rtol, y0, y1))


def _expand_as(inputs, target):
    try:
        iter(inputs)
    except TypeError:
        return [inputs] * len(target)
    else:
        return inputs


def _handle_unused_kwargs(solver, unused_kwargs):
    if len(unused_kwargs) > 0:
        warnings.warn('{}: Unexpected arguments {}'.format(solver.__class__.__name__, unused_kwargs))


def _norm(x):
    """Compute RMS norm."""
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

    scale = tuple(atol_ + y0_.abs() * rtol_ for y0_, atol_, rtol_ in zip(y0, atol, rtol))

    d0 = tuple(_norm(y0_ / scale_) for y0_, scale_ in zip(y0, scale))
    d1 = tuple(_norm(f0_ / scale_) for f0_, scale_ in zip(f0, scale))

    if max(d0).item() < 1e-5 or max(d1).item() < 1e-5:
        h0 = torch.tensor(1e-6, dtype=y0.dtype, device=y0.device)
    else:
        h0 = 0.01 * max(d0_ / d1_ for d0_, d1_ in zip(d0, d1))

    y1 = tuple(y0_ + h0 * f0_ for y0_, f0_ in zip(y0, f0))
    f1 = func(t0 + h0, y1)

    d2 = tuple(_norm((f1_ - f0_) / scale_) / h0 for f1_, f0_, scale_ in zip(f1, f0, scale))

    if max(d1).item() <= 1e-15 and max(d2).item() <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=y0.dtype, device=y0.device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1 + d2)) ** (1. / float(order + 1))

    return torch.min(100 * h0, h1).type_as(t0)


def _compute_error_ratio(error_estimate, error_tol):
    error_ratio = tuple(error_estimate_ / error_tol_ for error_estimate_, error_tol_ in zip(error_estimate, error_tol))
    mean_sq_error_ratio = tuple(error_ratio_.pow(2).mean() for error_ratio_ in error_ratio)
    return mean_sq_error_ratio


def _optimal_step_size(last_step, mean_error_ratio, safety, ifactor, dfactor, order):
    """Calculate the optimal size for the next step."""
    mean_error_ratio = max(mean_error_ratio)  # Compute step size based on highest ratio.
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = torch.ones((), dtype=last_step.dtype, device=last_step.device)
    error_ratio = torch.sqrt(mean_error_ratio).type_as(last_step)
    exponent = torch.tensor(order, dtype=last_step.dtype, device=last_step.device).reciprocal()
    factor = torch.min(ifactor, torch.max(safety / error_ratio ** exponent, dfactor))
    return last_step * factor


def _decreasing(t):
    return (t[1:] < t[:-1]).all()


def _assert_one_dimensional(name, t):
    assert t.ndimension() == 1, "{} must be one dimensional".format(name)


def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)


def _check_inputs(func, y0, t, options):
    tensor_input = False
    if torch.is_tensor(y0):
        tensor_input = True
        y0 = (y0,)
        _base_nontuple_func_ = func
        func = lambda t, y: (_base_nontuple_func_(t, y[0]),)
    assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
    for y0_ in y0:
        assert torch.is_tensor(y0_), 'each element must be a torch.Tensor but received {}'.format(type(y0_))

    if _decreasing(t):
        t = -t
        _base_reverse_func = func
        func = lambda t, y: tuple(-f_ for f_ in _base_reverse_func(-t, y))
        try:
            grid_points = options['grid_points']
        except KeyError:
            pass
        else:
            options = options.copy()
            options['grid_points'] = -grid_points

    assert torch.is_tensor(t), 't must be a torch.Tensor'
    _assert_one_dimensional('t', t)
    _assert_increasing('t', t)
    try:
        grid_points = options['grid_points']
    except KeyError:
        pass
    else:
        assert torch.is_tensor(grid_points), 'grid_points must be a torch.Tensor'
        _assert_one_dimensional('grid_points', grid_points)
        _assert_increasing('grid_points', grid_points)

    for y0_ in y0:
        if not torch.is_floating_point(y0_):
            raise TypeError('`y0` must be a floating point Tensor but is a {}'.format(y0_.type()))
    if not torch.is_floating_point(t):
        raise TypeError('`t` must be a floating point Tensor but is a {}'.format(t.type()))

    return tensor_input, func, y0, t, options
