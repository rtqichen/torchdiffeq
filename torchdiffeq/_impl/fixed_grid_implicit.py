import torch
from .rk_common import FixedGridFIRKODESolver, FixedGridDIRKODESolver
from .rk_common import _ButcherTableau

_sqrt_2 = torch.sqrt(torch.tensor(2, dtype=torch.float64)).item()
_sqrt_3 = torch.sqrt(torch.tensor(3, dtype=torch.float64)).item()
_sqrt_6 = torch.sqrt(torch.tensor(6, dtype=torch.float64)).item()
_sqrt_15 = torch.sqrt(torch.tensor(15, dtype=torch.float64)).item()

_IMPLICIT_EULER_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1], dtype=torch.float64),
    beta=[
        torch.tensor([1], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64),
)

class ImplicitEuler(FixedGridFIRKODESolver):
    order = 1
    tableau = _IMPLICIT_EULER_TABLEAU

_IMPLICIT_MIDPOINT_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 2], dtype=torch.float64),

    ],
    c_sol=torch.tensor([1], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64),
)

class ImplicitMidpoint(FixedGridFIRKODESolver):
    order = 2
    tableau = _IMPLICIT_MIDPOINT_TABLEAU

_GAUSS_LEGENDRE_4_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2 - _sqrt_3 / 6, 1 / 2 - _sqrt_3 / 6], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 4, 1 / 4 - _sqrt_3 / 6], dtype=torch.float64),
        torch.tensor([1 / 4 + _sqrt_3 / 6, 1 / 4], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 2, 1 / 2], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64),
)

_TRAPEZOID_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0, 1], dtype=torch.float64),
    beta=[
        torch.tensor([0, 0], dtype=torch.float64),
        torch.tensor([1 /2, 1 / 2], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 2, 1 / 2], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64),
)

class Trapezoid(FixedGridFIRKODESolver):
    order = 2
    tableau = _TRAPEZOID_TABLEAU


class GaussLegendre4(FixedGridFIRKODESolver):
    order = 4
    tableau = _GAUSS_LEGENDRE_4_TABLEAU

_GAUSS_LEGENDRE_6_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2 - _sqrt_15 / 10, 1 / 2, 1 / 2 + _sqrt_15 / 10], dtype=torch.float64),
    beta=[
        torch.tensor([5 / 36                , 2 / 9 - _sqrt_15 / 15, 5 / 36 - _sqrt_15 / 30], dtype=torch.float64),
        torch.tensor([5 / 36 + _sqrt_15 / 24, 2 / 9                , 5 / 36 - _sqrt_15 / 24], dtype=torch.float64),
        torch.tensor([5 / 36 + _sqrt_15 / 30, 2 / 9 + _sqrt_15 / 15, 5 / 36                ], dtype=torch.float64),
    ],
    c_sol=torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64),
)

class GaussLegendre6(FixedGridFIRKODESolver):
    order = 6
    tableau = _GAUSS_LEGENDRE_6_TABLEAU

_RADAU_IIA_3_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 3, 1], dtype=torch.float64),
    beta=[
        torch.tensor([5 / 12, -1 / 12], dtype=torch.float64),
        torch.tensor([3 / 4, 1 / 4], dtype=torch.float64)
    ],
    c_sol=torch.tensor([3 / 4, 1 / 4], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64)
)

class RadauIIA3(FixedGridFIRKODESolver):
    order = 3
    tableau = _RADAU_IIA_3_TABLEAU

_RADAU_IIA_5_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([2 / 5 - _sqrt_6 / 10, 2 / 5 + _sqrt_6 / 10, 1], dtype=torch.float64),
    beta=[
        torch.tensor([11 / 45 - 7 * _sqrt_6 / 360    , 37 / 225 - 169 * _sqrt_6 / 1800, -2 / 225 + _sqrt_6 / 75], dtype=torch.float64),
        torch.tensor([37 / 225 + 169 * _sqrt_6 / 1800, 11 / 45 + 7 * _sqrt_6 / 360    , -2 / 225 - _sqrt_6 / 75], dtype=torch.float64),
        torch.tensor([4 / 9 - _sqrt_6 / 36           , 4 / 9 + _sqrt_6 / 36           , 1 / 9], dtype=torch.float64)
    ],
    c_sol=torch.tensor([4 / 9 - _sqrt_6 / 36, 4 / 9 + _sqrt_6 / 36, 1 / 9], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64)
)

class RadauIIA5(FixedGridFIRKODESolver):
    order = 5
    tableau = _RADAU_IIA_5_TABLEAU

gamma = (2. - _sqrt_2) / 2.
_SDIRK_2_TABLEAU = _ButcherTableau(
    alpha = torch.tensor([gamma, 1], dtype=torch.float64),
    beta=[
        torch.tensor([gamma], dtype=torch.float64),
        torch.tensor([1 - gamma, gamma], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 - gamma, gamma], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64)
)

class SDIRK2(FixedGridDIRKODESolver):
    order = 2
    tableau = _SDIRK_2_TABLEAU

gamma = 1. - _sqrt_2 / 2.
beta = _sqrt_2 / 4.
_TRBDF_2_TABLEAU = _ButcherTableau(
    alpha = torch.tensor([0, 2 * gamma, 1], dtype=torch.float64),
    beta=[
        torch.tensor([0], dtype=torch.float64),
        torch.tensor([gamma, gamma], dtype=torch.float64),
        torch.tensor([beta, beta, gamma], dtype=torch.float64),
    ],
    c_sol=torch.tensor([beta, beta, gamma], dtype=torch.float64),
    c_error=torch.tensor([], dtype=torch.float64)
)

class TRBDF2(FixedGridDIRKODESolver):
    order = 2
    tableau = _TRBDF_2_TABLEAU
