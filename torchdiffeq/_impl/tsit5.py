import torch
from .misc import _scaled_dot_product, _convert_to_tensor, _is_finite, _select_initial_step, _handle_unused_kwargs
from .solvers import AdaptiveStepsizeODESolver
from .rk_common import _RungeKuttaState, _ButcherTableau, _runge_kutta_step

# Parameters from Tsitouras (2011).
_TSITOURAS_TABLEAU = _ButcherTableau(
    alpha=[0.161, 0.327, 0.9, 0.9800255409045097, 1., 1.],
    beta=[
        [0.161],
        [-0.008480655492357, 0.3354806554923570],
        [2.897153057105494, -6.359448489975075, 4.362295432869581],
        [5.32586482843925895, -11.74888356406283, 7.495539342889836, -0.09249506636175525],
        [5.86145544294642038, -12.92096931784711, 8.159367898576159, -0.071584973281401006, -0.02826905039406838],
        [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774],
    ],
    c_sol=[0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774, 0],
    c_error=[
        0.09646076681806523 - 0.001780011052226,
        0.01 - 0.000816434459657,
        0.4798896504144996 - -0.007880878010262,
        1.379008574103742 - 0.144711007173263,
        -3.290069515436081 - -0.582357165452555,
        2.324710524099774 - 0.458082105929187,
        -1 / 66,
    ],
)


def _interp_coeff_tsit5(t0, dt, eval_t):
    t = float((eval_t - t0) / dt)
    b1 = -1.0530884977290216 * t * (t - 1.3299890189751412) * (t**2 - 1.4364028541716351 * t + 0.7139816917074209)
    b2 = 0.1017 * t**2 * (t**2 - 2.1966568338249754 * t + 1.2949852507374631)
    b3 = 2.490627285651252793 * t**2 * (t**2 - 2.38535645472061657 * t + 1.57803468208092486)
    b4 = -16.54810288924490272 * (t - 1.21712927295533244) * (t - 0.61620406037800089) * t**2
    b5 = 47.37952196281928122 * (t - 1.203071208372362603) * (t - 0.658047292653547382) * t**2
    b6 = -34.87065786149660974 * (t - 1.2) * (t - 0.666666666666666667) * t**2
    b7 = 2.5 * (t - 1) * (t - 0.6) * t**2
    return [b1, b2, b3, b4, b5, b6, b7]


def _interp_eval_tsit5(t0, t1, k, eval_t):
    dt = t1 - t0
    y0 = tuple(k_[0] for k_ in k)
    interp_coeff = _interp_coeff_tsit5(t0, dt, eval_t)
    y_t = tuple(y0_ + _scaled_dot_product(dt, interp_coeff, k_) for y0_, k_ in zip(y0, k))
    return y_t


def _optimal_step_size(last_step, mean_error_ratio, safety=0.9, ifactor=10.0, dfactor=0.2, order=5):
    """Calculate the optimal size for the next Runge-Kutta step."""
    if mean_error_ratio == 0:
        return last_step * ifactor
    if mean_error_ratio < 1:
        dfactor = _convert_to_tensor(1, dtype=torch.float64, device=mean_error_ratio.device)
    error_ratio = torch.sqrt(mean_error_ratio).type_as(last_step)
    exponent = torch.tensor(1 / order).type_as(last_step)
    factor = torch.max(1 / ifactor, torch.min(error_ratio**exponent / safety, 1 / dfactor))
    return last_step / factor


def _abs_square(x):
    return torch.mul(x, x)


class Tsit5Solver(AdaptiveStepsizeODESolver):

    def __init__(
        self, func, y0, rtol, atol, first_step=None, safety=0.9, ifactor=10.0, dfactor=0.2, max_num_steps=2**31 - 1,
        **unused_kwargs
    ):
        _handle_unused_kwargs(self, unused_kwargs)
        del unused_kwargs

        self.func = func
        self.y0 = y0
        self.rtol = rtol
        self.atol = atol
        self.first_step = first_step
        self.safety = _convert_to_tensor(safety, dtype=torch.float64, device=y0[0].device)
        self.ifactor = _convert_to_tensor(ifactor, dtype=torch.float64, device=y0[0].device)
        self.dfactor = _convert_to_tensor(dfactor, dtype=torch.float64, device=y0[0].device)
        self.max_num_steps = _convert_to_tensor(max_num_steps, dtype=torch.int32, device=y0[0].device)

    def before_integrate(self, t):
        if self.first_step is None:
            first_step = _select_initial_step(self.func, t[0], self.y0, 4, self.rtol, self.atol).to(t)
        else:
            first_step = _convert_to_tensor(0.01, dtype=t.dtype, device=t.device)
        self.rk_state = _RungeKuttaState(
            self.y0,
            self.func(t[0].type_as(self.y0[0]), self.y0), t[0], t[0], first_step,
            tuple(map(lambda x: [x] * 7, self.y0))
        )

    def advance(self, next_t):
        """Interpolate through the next time point, integrating as necessary."""
        n_steps = 0
        while next_t > self.rk_state.t1:
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self.rk_state = self._adaptive_tsit5_step(self.rk_state)
            n_steps += 1
        return _interp_eval_tsit5(self.rk_state.t0, self.rk_state.t1, self.rk_state.interp_coeff, next_t)

    def _adaptive_tsit5_step(self, rk_state):
        """Take an adaptive Runge-Kutta step to integrate the ODE."""
        y0, f0, _, t0, dt, _ = rk_state
        ########################################################
        #                      Assertions                      #
        ########################################################
        assert t0 + dt > t0, 'underflow in dt {}'.format(dt.item())
        for y0_ in y0:
            assert _is_finite(torch.abs(y0_)), 'non-finite values in state `y`: {}'.format(y0_)
        y1, f1, y1_error, k = _runge_kutta_step(self.func, y0, f0, t0, dt, tableau=_TSITOURAS_TABLEAU)

        ########################################################
        #                     Error Ratio                      #
        ########################################################
        error_tol = tuple(self.atol + self.rtol * torch.max(torch.abs(y0_), torch.abs(y1_)) for y0_, y1_ in zip(y0, y1))
        tensor_error_ratio = tuple(y1_error_ / error_tol_ for y1_error_, error_tol_ in zip(y1_error, error_tol))
        sq_error_ratio = tuple(
            torch.mul(tensor_error_ratio_, tensor_error_ratio_) for tensor_error_ratio_ in tensor_error_ratio
        )
        mean_error_ratio = (
            sum(torch.sum(sq_error_ratio_) for sq_error_ratio_ in sq_error_ratio) /
            sum(sq_error_ratio_.numel() for sq_error_ratio_ in sq_error_ratio)
        )
        accept_step = mean_error_ratio <= 1

        ########################################################
        #                   Update RK State                    #
        ########################################################
        y_next = y1 if accept_step else y0
        f_next = f1 if accept_step else f0
        t_next = t0 + dt if accept_step else t0
        dt_next = _optimal_step_size(dt, mean_error_ratio, self.safety, self.ifactor, self.dfactor)
        k_next = k if accept_step else self.rk_state.interp_coeff
        rk_state = _RungeKuttaState(y_next, f_next, t0, t_next, dt_next, k_next)
        return rk_state
