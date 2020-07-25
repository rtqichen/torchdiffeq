from .rk_common import _ButcherTableau
from .solvers import RKAdaptiveStepsizeODESolver

_BOGACKI_SHAMPINE_TABLEAU = _ButcherTableau(
    alpha=[1/2, 3/4,  1.],
    beta=[
        [1/2],
        [0., 3/4],
        [2/9, 1/3, 4/9]
    ],
    c_sol=[2/9, 1/3, 4/9, 0.],
    c_error=[2/9-7/24, 1/3-1/4, 4/9-1/3, -1/8],
)

BS_C_MID = [ 0., 0.5,  0., 0.  ]


class Bosh3Solver(RKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _BOGACKI_SHAMPINE_TABLEAU
    mid = BS_C_MID
