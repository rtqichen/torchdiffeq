from .rk_common import _ButcherTableau
from .solvers import RKAdaptiveStepsizeODESolver


_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=[1.],
    beta=[
        [1.],
    ],
    c_sol=[0.5, 0.5],
    c_error=[
        0.5,
        -0.5,
    ],
)

AH_C_MID = [
    0.5, 0.
]


class AdaptiveHeunSolver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = AH_C_MID
