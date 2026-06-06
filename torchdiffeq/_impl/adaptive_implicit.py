import torch
from .rk_common import _ButcherTableau
from .rk_common import FIRKAdaptiveStepsizeODESolver, DIRKAdaptiveStepsizeODESolver

_sqrt_3 = torch.sqrt(torch.tensor(3, dtype=torch.float64)).item()
_sqrt_15 = torch.sqrt(torch.tensor(15, dtype=torch.float64)).item()

def polyval(coeffs: list, x: float):
    """
    Evaluate a polynomial for a given x.
    Degree is len(coeffs)-1

    Args:
        coeffs (list): list of coefficients from highest power to constant

        x (float): number to evaluate at
    """
    max_degree = len(coeffs) - 1
    return sum(coeffs[i] * (x**(max_degree-i)) for i in range(len(coeffs)))

_GAUSS_LEGENDRE_4_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2 - _sqrt_3 / 6, 1 / 2 + _sqrt_3 / 6], dtype=torch.float64),
    beta=[
        torch.tensor([1 / 4, 1 / 4 - _sqrt_3 / 6], dtype=torch.float64),
        torch.tensor([1 / 4 + _sqrt_3 / 6, 1 / 4], dtype=torch.float64),
    ],
    c_sol=torch.tensor([1 / 2, 1 / 2], dtype=torch.float64),
    c_error=torch.tensor([1 / 2 + _sqrt_3 / 2, 1 / 2 - _sqrt_3 / 2], dtype=torch.float64),
)

_GL4_C_MID = torch.tensor([1 / 4 + _sqrt_3 / 8, 1 / 4 - _sqrt_3 / 8], dtype=torch.float64)

class AdaptiveGaussLegendre4(FIRKAdaptiveStepsizeODESolver):
    order = 4
    tableau = _GAUSS_LEGENDRE_4_TABLEAU
    mid = _GL4_C_MID

_GAUSS_LEGENDRE_6_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([1 / 2 - _sqrt_15 / 10, 1 / 2, 1 / 2 + _sqrt_15 / 10], dtype=torch.float64),
    beta=[
        torch.tensor([5 / 36                , 2 / 9 - _sqrt_15 / 15, 5 / 36 - _sqrt_15 / 30], dtype=torch.float64),
        torch.tensor([5 / 36 + _sqrt_15 / 24, 2 / 9                , 5 / 36 - _sqrt_15 / 24], dtype=torch.float64),
        torch.tensor([5 / 36 + _sqrt_15 / 30, 2 / 9 + _sqrt_15 / 15, 5 / 36                ], dtype=torch.float64),
    ],
    c_sol=torch.tensor([5 / 18, 4 / 9, 5 / 18], dtype=torch.float64),
    c_error=torch.tensor([-5 / 6, 8 / 3, -5 / 6], dtype=torch.float64),
)

_GL6_C_MID = torch.tensor([-5 / 18, 19 / 18, -5 / 18], dtype=torch.float64)

class AdaptiveGaussLegendre6(FIRKAdaptiveStepsizeODESolver):
    order = 6
    tableau = _GAUSS_LEGENDRE_6_TABLEAU
    mid = _GL6_C_MID

# https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_solver/kvaerno3.py
gamma = 0.43586652150
b31 = (-4 * gamma**2 + 6 * gamma - 1) / (4 * gamma)
b32 = (-2 * gamma + 1) / (4 * gamma)
b41 = (6 * gamma - 1) / (12 * gamma)
b42 = -1 / ((24 * gamma - 12) * gamma)
b43 = (-6 * gamma**2 + 6 * gamma - 1) / (6 * gamma - 3)
_KVAERNO_3_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0, 2 * gamma, 1.0, 1.0], dtype=torch.float64),
    beta = [
        torch.tensor([0], dtype=torch.float64),
        torch.tensor([gamma, gamma], dtype=torch.float64),
        torch.tensor([b31, b32, gamma], dtype=torch.float64),
        torch.tensor([b41, b42, b43, gamma], dtype=torch.float64),
    ],
    c_sol=torch.tensor([b41, b42, b43, gamma], dtype=torch.float64),
    c_error=torch.tensor([b41 - b31, b42 - b32, b43 - gamma, gamma])
)

_KV3_C_MID = torch.tensor([0.35414591, 0.08861962, 0.09340915, -0.03617468], dtype=torch.float64)

class Kvaerno3(DIRKAdaptiveStepsizeODESolver):
    order = 3
    tableau = _KVAERNO_3_TABLEAU
    mid = _KV3_C_MID

gamma = 0.5728160625
b31 = polyval([144,-180,81,-15,1,0], gamma) / polyval([12,-6,1], gamma)
b32 = polyval([-36,39,-15,2,0], gamma) / polyval([12,-6,1], gamma)
b41 = polyval([-144,396,-330,117,18,1], gamma) / 12 / gamma / gamma / polyval([12,-9,0], gamma)
b42 = polyval([72,-126,69,-15,1], gamma) / (12*gamma*gamma) / (3*gamma - 1)
b43 = polyval([-6,6,-1], gamma) * polyval([12,-6,1], gamma) / (12*gamma*gamma) / polyval([12,-9,2], gamma) / (3*gamma -1)
b51 = polyval([288,-312,120,-18,1], gamma) / (48*gamma*gamma) / polyval([12,-9,2], gamma)
b52 = polyval([24,-12,1], gamma) / (48*gamma*gamma) / (3*gamma - 1)
b53 = -(polyval([12,-6,1], gamma)**3) / (48*gamma*gamma) / (3*gamma - 1) / polyval([12,-9,2], gamma) / polyval([6,-6,1], gamma)
b54 = polyval([-24,36,-12,1], gamma) / polyval([24,-24,4], gamma)
alpha3 = gamma + b31 + b32
_KVAERNO_4_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0, 2 * gamma, alpha3, 1.0, 1.0], dtype=torch.float64),
    beta = [
        torch.tensor([0], dtype=torch.float64),
        torch.tensor([gamma, gamma], dtype=torch.float64),
        torch.tensor([b31, b32, gamma], dtype=torch.float64),
        torch.tensor([b41, b42, b43, gamma], dtype=torch.float64),
        torch.tensor([b51, b52, b53, b54, gamma], dtype=torch.float64),
    ],
    c_sol=torch.tensor([b51, b52, b53, b54, gamma], dtype=torch.float64),
    c_error=torch.tensor([b51 - b41, b52 - b42, b53 - b43, b54 - gamma, gamma])
)

# # Solve for C_MID
# import numpy as np
# def polyval(coeffs: list, x: float):
#     """
#     Evaluate a polynomial for a given x.
#     Degree is len(coeffs)-1

#     Args:
#         coeffs (list): list of coefficients from highest power to constant

#         x (float): number to evaluate at
#     """
#     max_degree = len(coeffs) - 1
#     return sum(coeffs[i] * (x**(max_degree-i)) for i in range(len(coeffs)))

# gamma = 0.5728160625
# b31 = polyval([144,-180,81,-15,1,0], gamma) / polyval([12,-6,1], gamma)
# b32 = polyval([-36,39,-15,2,0], gamma) / polyval([12,-6,1], gamma)
# alpha3 = gamma + b31 + b32

# def compute_kvaerno4_midpoint_weights(gamma_val=0.2500000000000000):
#     """
#     Computes the midpoint evaluation weights (alpha) for the 4th-order 
#     Kværnø ESDIRK method at step fraction theta = 0.5.
    
#     Parameters:
#     gamma_val (float): Diagonally implicit parameter matching the solver configuration.
#                        Defaults to Kværnø's canonical 4th-order root (~0.25).
#     """
#     # 1. Define the 5 method nodes (c) based on the implicit parameter gamma
#     c1 = 0.0
#     c2 = 2.0 * gamma_val
#     c3 = alpha3
#     c4 = 1.0 - gamma_val
#     c5 = 1.0
#     nodes = [c1, c2, c3, c4, c5]
    
#     # 2. Construct the Vandermonde matrix for the algebraic order conditions (Orders 1 to 4)
#     # Rows represent: Order 1 (\theta^1), Order 2 (\theta^2), Order 3 (\theta^3), Order 4 (\theta^4)
#     V = np.array([
#         [1.0,   1.0,   1.0,   1.0,   1.0],
#         [c1,    c2,    c3,    c4,    c5],
#         [c1**2, c2**2, c3**2, c4**2, c5**2],
#         [c1**3, c2**3, c3**3, c4**3, c5**3]
#     ])
    
#     # 3. Define the target integrated values evaluated exactly at the midpoint (\theta = 1/2)
#     # The right-hand side corresponds to: \theta^p / p  => [ 1/2, 1/8, 1/24, 1/64 ]
#     b = np.array([1/2, 1/8, 1/24, 1/64])
    
#     # 4. Resolve the system
#     # Since there are 5 stages and 4 constraint equations, we utilize the Moore-Penrose 
#     # pseudo-inverse to isolate the minimum-norm solution for the algebraic extension.
#     alpha_weights = np.linalg.pinv(V) @ b
    
#     # Print results out clearly for the console
#     print(f"--- Kværnø 4th-Order Midpoint Derivation ---")
#     print(f"Implicit Gamma (gamma): {gamma_val:.6f}\n")
#     print(f"Stage Nodes (c_i):")
#     for i, node in enumerate(nodes, 1):
#         print(f"  c_{i} = {node:.6f}")
        
#     print(f"\nCalculated Midpoint Weights (alpha_i):")
#     for i, weight in enumerate(alpha_weights, 1):
#         print(f"  alpha_{i} = {weight:.8f}")
        
#     return nodes, alpha_weights

# # Execute the weight computation
# nodes, weights = compute_kvaerno4_midpoint_weights(gamma)

_KV4_C_MID = torch.tensor([0.16574835, 0.10455361, 0.06864508, 0.34505977, -0.18400681], dtype=torch.float64)

class Kvaerno4(DIRKAdaptiveStepsizeODESolver):
    order = 4
    tableau = _KVAERNO_4_TABLEAU
    mid = _KV4_C_MID

_KVAERNO_5_TABLEAU = _ButcherTableau(
    alpha=torch.tensor([0, 0.52, 1.230333209967908, 0.8957659843500759, 0.43639360985864756, 1.0, 1.0], dtype=torch.float64),
    beta = [
        torch.tensor([0], dtype=torch.float64),
        torch.tensor([0.26, 0.26], dtype=torch.float64),
        torch.tensor([0.13, 0.84033320996790809, 0.26], dtype=torch.float64),
        torch.tensor([0.22371961478320505, 0.47675532319799699, -0.06470895363112615, 0.26], dtype=torch.float64),
        torch.tensor([0.16648564323248321, 0.10450018841591720, 0.03631482272098715, -0.13090704451073998, 0.26], dtype=torch.float64),
        torch.tensor([0.13855640231268224, 0, -0.04245337201752043, 0.02446657898003141, 0.61943039072480676, 0.26], dtype=torch.float64),
        torch.tensor([0.13659751177640291, 0, -0.05496908796538376, -0.04118626728321046, 0.62993304899016403, 0.06962479448202728, 0.26], dtype=torch.float64),
    ],
    c_sol=torch.tensor([0.13659751177640291, 0, -0.05496908796538376, -0.04118626728321046, 0.62993304899016403, 0.06962479448202728, 0.26], dtype=torch.float64),
    c_error=(
        torch.tensor([0.13659751177640291, 0, -0.05496908796538376, -0.04118626728321046, 0.62993304899016403, 0.06962479448202728, 0.26], dtype=torch.float64)-
        torch.tensor([0.13855640231268224, 0, -0.04245337201752043, 0.02446657898003141, 0.61943039072480676, 0.26, 0], dtype=torch.float64)
    )
)

# # Solve for C_MID
# import numpy as np

# def compute_kvaerno5_midpoint_weights():
#     """
#     Computes the midpoint evaluation weights (alpha) for the 5th-order 
#     Kværnø ESDIRK method at step fraction theta = 0.5.
#     """
#     # 1. Define the 5 method nodes (c) based on the implicit parameter gamma
#     c = [0, 0.52, 1.230333209967908, 0.8957659843500759, 0.43639360985864756, 1.0, 1.0]
#     nodes = [c_val for c_val in c]
    
#     # 2. Construct the Vandermonde matrix for the algebraic order conditions (Orders 1 to 4)
#     # Rows represent: Order 1 (\theta^1), Order 2 (\theta^2), Order 3 (\theta^3), Order 4 (\theta^4)
#     V = np.array([
#         [c_val**0 for c_val in c],
#         [c_val**1 for c_val in c],
#         [c_val**2 for c_val in c],
#         [c_val**3 for c_val in c],
#         [c_val**4 for c_val in c],
#     ])
    
#     # 3. Define the target integrated values evaluated exactly at the midpoint (\theta = 1/2)
#     # The right-hand side corresponds to: \theta^p / p  => [ 1/2, 1/8, 1/24, 1/64, 1/160 ]
#     b = np.array([1/2, 1/8, 1/24, 1/64, 1/160])
    
#     # 4. Resolve the system
#     alpha_weights = np.linalg.pinv(V) @ b
    
#     # Print results out clearly for the console
#     print(f"--- Kværnø 5th-Order Midpoint Derivation ---")
#     print(f"Implicit Gamma (gamma): {0.26:.6f}\n")
#     print(f"Stage Nodes (c_i):")
#     for i, node in enumerate(nodes, 1):
#         print(f"  c_{i} = {node:.6f}")
        
#     print(f"\nCalculated Midpoint Weights (alpha_i):")
#     for i, weight in enumerate(alpha_weights, 1):
#         print(f"  alpha_{i} = {weight:.8f}")
        
#     return nodes, alpha_weights

# # Execute the weight computation
# nodes, weights = compute_kvaerno5_midpoint_weights()

_KV5_C_MID = torch.tensor([0.13733277, -0.23363423, -0.07209358, -0.50521749, 0.68463909, 0.24448673, 0.24448673], dtype=torch.float64)

class Kvaerno5(DIRKAdaptiveStepsizeODESolver):
    order = 5
    tableau = _KVAERNO_5_TABLEAU
    mid = _KV5_C_MID
