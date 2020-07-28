import collections
import torch


class _ButcherTableau(collections.namedtuple('_ButcherTableau', 'alpha, beta, c_sol, c_error')):
    # TODO: optimise a bit
    @property
    def length(self):
        return len(self.alpha)


_RungeKuttaState = collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')
# Saved state of the Runge Kutta solver.
#
# Attributes:
#     y1: Tensor giving the function value at the end of the last time step.
#     f1: Tensor giving derivative at the end of the last time step.
#     t0: scalar float64 Tensor giving start of the last time step.
#     t1: scalar float64 Tensor giving end of the last time step.
#     dt: scalar float64 Tensor giving the size for the next time step.
#     interp_coeff: list of Tensors giving coefficients for polynomial
#         interpolation between `t0` and `t1`.


class _DistributeGradients(torch.autograd.Function):
    @staticmethod
    def forward(ctx, k, *fs):
        return k

    @staticmethod
    def backward(ctx, grad_k):
        return (grad_k, *grad_k.unbind(dim=-1))


def _runge_kutta_step(func, y0, f0, t0, dt, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.

    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        tableau: _ButcherTableau describing how to take the Runge-Kutta step.

    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """

    # Little bit of black magic coming up.
    # We copy all the func evaluations into this buffer k, as that means we can do the subsequent matrix-vector products
    # faster, as it all happens in PyTorch.
    # However we need to make sure that gradients flow correctly, and by default, if computing gradients wrt t, then it
    # will throw an error about in-place modification of k, for the `matmul`. The tensor `k` has indeed been modified
    # in-place, but only in places which won't affect the computation.
    # So we sneak in the copy by going via torch.no_grad(), and then fix the gradients after the fact (without
    # modifying k, so the version doesn't increment) with _DistributeGradients.
    k = torch.empty(*f0.shape, tableau.length + 1, dtype=f0.dtype, device=f0.device)
    with torch.no_grad():
        k[..., 0] = f0  # sneak past the version checker
    fs = [f0]
    for i, (alpha_i, beta_i) in enumerate(zip(tableau.alpha, tableau.beta)):
        ti = t0 + alpha_i * dt
        yi = y0 + k[..., :i + 1].matmul(beta_i * dt).view_as(f0)
        f = func(ti, yi)
        fs.append(f)
        with torch.no_grad():
            k[..., i + 1] = f  # sneak past the version checker
    k = _DistributeGradients.apply(k, *fs)  # then fix the gradients we broke

    if not (tableau.c_sol[-1] == 0 and (tableau.c_sol[:-1] == tableau.beta[-1]).all()):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = y0 + k.matmul(dt * tableau.c_sol).view_as(f0)

    y1 = yi
    f1 = k[..., -1]
    y1_error = k.matmul(dt * tableau.c_error)
    return y1, f1, y1_error, k


# Precompute divisions
_one_third = 1 / 3
_two_thirds = 2 / 3
_one_sixth = 1 / 6


def rk4_step_func(func, t, dt, y, k1=None):
    if k1 is None:
        k1 = func(t, y)
    half_dt = dt * 0.5
    k2 = func(t + half_dt, y + half_dt * k1)
    k3 = func(t + half_dt, y + half_dt * k2)
    k4 = func(t + dt, y + dt * k3)
    return (k1 + 2 * (k2 + k3) + k4) * dt * _one_sixth


def rk4_alt_step_func(func, t, dt, y, k1=None):
    """Smaller error with slightly more compute."""
    if k1 is None:
        k1 = func(t, y)
    k2 = func(t + dt * _one_third, y + dt * k1 * _one_third)
    k3 = func(t + dt * _two_thirds, y + dt * (k2 - k1 * _one_third))
    k4 = func(t + dt, y + dt * (k1 - k2 + k3))
    return (k1 + 3 * (k2 + k3) + k4) * dt * 0.125
