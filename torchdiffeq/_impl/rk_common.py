# Based on https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/integrate
import collections
from .misc import _scaled_dot_product, _convert_to_tensor

_ButcherTableau = collections.namedtuple('_ButcherTableau', 'alpha beta c_sol c_error')


class _RungeKuttaState(collections.namedtuple('_RungeKuttaState', 'y1, f1, t0, t1, dt, interp_coeff')):
    """Saved state of the Runge Kutta solver.

    Attributes:
        y1: Tensor giving the function value at the end of the last time step.
        f1: Tensor giving derivative at the end of the last time step.
        t0: scalar float64 Tensor giving start of the last time step.
        t1: scalar float64 Tensor giving end of the last time step.
        dt: scalar float64 Tensor giving the size for the next time step.
        interp_coef: list of Tensors giving coefficients for polynomial
            interpolation between `t0` and `t1`.
    """


def _runge_kutta_step(func, y0, f0, t0, dt, tableau):
    """Take an arbitrary Runge-Kutta step and estimate error.

    Args:
        func: Function to evaluate like `func(t, y)` to compute the time derivative
            of `y`.
        y0: Tensor initial value for the state.
        f0: Tensor initial value for the derivative, computed from `func(t0, y0)`.
        t0: float64 scalar Tensor giving the initial time.
        dt: float64 scalar Tensor giving the size of the desired time step.
        tableau: optional _ButcherTableau describing how to take the Runge-Kutta
            step.
        name: optional name for the operation.

    Returns:
        Tuple `(y1, f1, y1_error, k)` giving the estimated function value after
        the Runge-Kutta step at `t1 = t0 + dt`, the derivative of the state at `t1`,
        estimated error at `t1`, and a list of Runge-Kutta coefficients `k` used for
        calculating these terms.
    """
    dtype = y0[0].dtype
    device = y0[0].device

    t0 = _convert_to_tensor(t0, dtype=dtype, device=device)
    dt = _convert_to_tensor(dt, dtype=dtype, device=device)

    k = tuple(map(lambda x: [x], f0))
    for alpha_i, beta_i in zip(tableau.alpha, tableau.beta):
        ti = t0 + alpha_i * dt
        yi = tuple(y0_ + _scaled_dot_product(dt, beta_i, k_) for y0_, k_ in zip(y0, k))
        tuple(k_.append(f_) for k_, f_ in zip(k, func(ti, yi)))

    if not (tableau.c_sol[-1] == 0 and tableau.c_sol[:-1] == tableau.beta[-1]):
        # This property (true for Dormand-Prince) lets us save a few FLOPs.
        yi = tuple(y0_ + _scaled_dot_product(dt, tableau.c_sol, k_) for y0_, k_ in zip(y0, k))

    y1 = yi
    f1 = tuple(k_[-1] for k_ in k)
    y1_error = tuple(_scaled_dot_product(dt, tableau.c_error, k_) for k_ in k)
    return (y1, f1, y1_error, k)


def rk4_step_func(func, t, dt, y, k1=None):
    if k1 is None: k1 = func(t, y)
    k2 = func(t + dt / 2, tuple(y_ + dt * k1_ / 2 for y_, k1_ in zip(y, k1)))
    k3 = func(t + dt / 2, tuple(y_ + dt * k2_ / 2 for y_, k2_ in zip(y, k2)))
    k4 = func(t + dt, tuple(y_ + dt * k3_ for y_, k3_ in zip(y, k3)))
    return tuple((k1_ + 2 * k2_ + 2 * k3_ + k4_) * (dt / 6) for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))


def rk4_alt_step_func(func, t, dt, y, k1=None):
    """Smaller error with slightly more compute."""
    if k1 is None: k1 = func(t, y)
    k2 = func(t + dt / 3, tuple(y_ + dt * k1_ / 3 for y_, k1_ in zip(y, k1)))
    k3 = func(t + dt * 2 / 3, tuple(y_ + dt * (k1_ / -3 + k2_) for y_, k1_, k2_ in zip(y, k1, k2)))
    k4 = func(t + dt, tuple(y_ + dt * (k1_ - k2_ + k3_) for y_, k1_, k2_, k3_ in zip(y, k1, k2, k3)))
    return tuple((k1_ + 3 * k2_ + 3 * k3_ + k4_) * (dt / 8) for k1_, k2_, k3_, k4_ in zip(k1, k2, k3, k4))
