import torch
from .misc import _convert_to_tensor, _dot_product


def _interp_fit(y0, y1, y_mid, f0, f1, dt):
    """Fit coefficients for 4th order polynomial interpolation.

    Args:
        y0: function value at the start of the interval.
        y1: function value at the end of the interval.
        y_mid: function value at the mid-point of the interval.
        f0: derivative value at the start of the interval.
        f1: derivative value at the end of the interval.
        dt: width of the interval.

    Returns:
        List of coefficients `[a, b, c, d, e]` for interpolating with the polynomial
        `p = a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e` for values of `x`
        between 0 (start of interval) and 1 (end of interval).
    """
    a = tuple(
        _dot_product([-2 * dt, 2 * dt, -8, -8, 16], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    b = tuple(
        _dot_product([5 * dt, -3 * dt, 18, 14, -32], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    c = tuple(
        _dot_product([-4 * dt, dt, -11, -5, 16], [f0_, f1_, y0_, y1_, y_mid_])
        for f0_, f1_, y0_, y1_, y_mid_ in zip(f0, f1, y0, y1, y_mid)
    )
    d = tuple(dt * f0_ for f0_ in f0)
    e = y0
    return [a, b, c, d, e]


def _interp_evaluate(coefficients, t0, t1, t):
    """Evaluate polynomial interpolation at the given time point.

    Args:
        coefficients: list of Tensor coefficients as created by `interp_fit`.
        t0: scalar float64 Tensor giving the start of the interval.
        t1: scalar float64 Tensor giving the end of the interval.
        t: scalar float64 Tensor giving the desired interpolation point.

    Returns:
        Polynomial interpolation of the coefficients at time `t`.
    """

    dtype = coefficients[0][0].dtype
    device = coefficients[0][0].device

    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    t1 = _convert_to_tensor(t1, dtype=dtype, device=device)
    t = _convert_to_tensor(t, dtype=dtype, device=device)

    assert (t0 <= t) & (t <= t1), 'invalid interpolation, fails `t0 <= t <= t1`: {}, {}, {}'.format(t0, t, t1)
    x = ((t - t0) / (t1 - t0)).type(dtype).to(device)

    xs = [torch.tensor(1).type(dtype).to(device), x]
    for _ in range(2, len(coefficients)):
        xs.append(xs[-1] * x)

    return tuple(_dot_product(coefficients_, reversed(xs)) for coefficients_ in zip(*coefficients))
