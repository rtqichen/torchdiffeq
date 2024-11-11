import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver


_ADAPTIVE_HEUN_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1.], dtype=torch.get_default_dtype()),
    beta=[
        torch.tensor([1.], dtype=torch.get_default_dtype()),
    ],
    c_sol=torch.tensor([0.5, 0.5], dtype=torch.get_default_dtype()),
    c_error=torch.tensor([
        0.5,
        -0.5,
    ], dtype=torch.get_default_dtype()),
)

_AH_C_MID = torch.tensor([
    0.5, 0.
], dtype=torch.get_default_dtype())


class AdaptiveHeunSolver(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _ADAPTIVE_HEUN_TABLEAU
    mid = _AH_C_MID
