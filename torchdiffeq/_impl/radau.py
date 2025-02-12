import torch
from .rk_common import _ButcherTableau, RKAdaptiveStepsizeODESolver

# Radau IIA coefficients (order 5, 3 stages)
# Reference: E. Hairer, G. Wanner, "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems"
_RADAU_IIA_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([
        (4 - torch.sqrt(torch.tensor(6.))) / 10,
        (4 + torch.sqrt(torch.tensor(6.))) / 10,
        1.
    ], dtype=torch.float64),
    beta=[
        torch.tensor([(4 - torch.sqrt(torch.tensor(6.))) / 10], dtype=torch.float64),
        torch.tensor([
            (88 - 7 * torch.sqrt(torch.tensor(6.))) / 360,
            (296 + 169 * torch.sqrt(torch.tensor(6.))) / 1800
        ], dtype=torch.float64),
        torch.tensor([
            (296 - 169 * torch.sqrt(torch.tensor(6.))) / 1800,
            (88 + 7 * torch.sqrt(torch.tensor(6.))) / 360,
            (16 - torch.sqrt(torch.tensor(6.))) / 36
        ], dtype=torch.float64),
    ],
    c_sol=torch.tensor([
        (16 - torch.sqrt(torch.tensor(6.))) / 36,
        (16 + torch.sqrt(torch.tensor(6.))) / 36,
        1/9,
        0.
    ], dtype=torch.float64),
    c_error=torch.tensor([
        (16 - torch.sqrt(torch.tensor(6.))) / 36 - (1/9),
        (16 + torch.sqrt(torch.tensor(6.))) / 36 - (1/9),
        0.,
        0.
    ], dtype=torch.float64),
)

# Interpolation coefficients for dense output
RADAU_C_MID = torch.tensor([
    0.5 * ((16 - torch.sqrt(torch.tensor(6.))) / 36),
    0.5 * ((16 + torch.sqrt(torch.tensor(6.))) / 36),
    0.5 * (1/9),
    0.
], dtype=torch.float64)


class RadauSolver(RKAdaptiveStepsizeODESolver):
    order = 5
    tableau = _RADAU_IIA_TABLEAU
    mid = RADAU_C_MID
