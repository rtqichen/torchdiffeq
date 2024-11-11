import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver

_FEHLBERG2_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2, 1.0], dtype=torch.get_default_dtype()),
    beta=[
        torch.tensor([1 / 2], dtype=torch.get_default_dtype()),
        torch.tensor([1 / 256, 255 / 256], dtype=torch.get_default_dtype()),
    ],
    c_sol=torch.tensor([1 / 512, 255 / 256, 1 / 512], dtype=torch.get_default_dtype()),
    c_error=torch.tensor(
        [-1 / 512, 0, 1 / 512], dtype=torch.get_default_dtype()
    ),
)

_FE_C_MID = torch.tensor([0.0, 0.5, 0.0], dtype=torch.get_default_dtype())


class Fehlberg2(RKAdaptiveStepsizeODESolver):
    order = 2
    tableau = _FEHLBERG2_TABLEAU
    mid = _FE_C_MID
