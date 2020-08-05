import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1.], dtype=torch.float64),
    beta=[
        torch.tensor([1.], dtype=torch.float64),
    ],
    c_sol=torch.tensor([0.5, 0.5], dtype=torch.float64),
    c_error=torch.tensor([
        0.5,
        -0.5,
    ], dtype=torch.float64),
)

_AH_C_MID = torch.tensor([
    0.5, 0.
], dtype=torch.float64)


class AdaptiveHeunSolver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
