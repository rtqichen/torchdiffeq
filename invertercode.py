import numpy as np
from scipy.optimize import fsolve, root, least_squares, newton_krylov


import torch
from torch.optim.optimizer import Optimizer

class PCGrad(Optimizer):
    """Projection Conflicting Gradients (Yu et al., NeurIPS 2020).
    Wrap any torch Optimizer   →  same API plus .pc_backward(loss_list)."""

    def __init__(self, optimizer: Optimizer):
        self._optim = optimizer
        self.param_groups = self._optim.param_groups
        self.defaults     = self._optim.defaults
        self.state        = self._optim.state

    # --- thin pass‑throughs --------------------------------------------
    def zero_grad(self, *a, **kw): return self._optim.zero_grad(*a, **kw)
    def step     (self, *a, **kw): return self._optim.step(*a, **kw)

    # --- helpers -------------------------------------------------------
    def _flat_grads(self):
        return [p.grad.detach().clone() if p.grad is not None else None
                for g in self.param_groups for p in g['params']]

    def _write_grads(self, new):
        k = 0
        for g in self.param_groups:
            for p in g['params']:
                p.grad = None if new[k] is None else new[k]
                k += 1

    # --- core ----------------------------------------------------------
    @torch.no_grad()
    def pc_backward(self, losses, retain_graph=False):
        if not isinstance(losses, (list, tuple)):
            losses = [losses]

        # collect gradients for each loss
        task_grads = []
        for i, L in enumerate(losses):
            L.backward(retain_graph=(i < len(losses)-1) or retain_graph)
            task_grads.append(self._flat_grads())
            self.zero_grad(set_to_none=True)

        # project pairwise conflicts
        n_tasks, n_params = len(task_grads), len(task_grads[0])
        merged = [torch.zeros_like(g) if g is not None else None
                  for g in task_grads[0]]

        for t in range(n_tasks):
            g_t = task_grads[t]
            for p in range(n_params):
                if g_t[p] is None: continue
                v = g_t[p].clone()
                for s in range(n_tasks):
                    if s == t or task_grads[s][p] is None: continue
                    g_s = task_grads[s][p]
                    dot = torch.dot(v.view(-1), g_s.view(-1))
                    if dot < 0:
                        v -= dot / (g_s.view(-1).pow(2).sum() + 1e-12) * g_s
                merged[p] = merged[p] + v / n_tasks

        self._write_grads(merged)
"""
Complete Per-Unit Power System Implementation with All Fixes



This implementation includes:
1. Fixed calculate_K method using pre-computed R_kappa
2. Fixed initialize_from_equilibrium with proper integral state conversion
3. Division by zero protection in calculate_Phi
4. Optimized update_network_matrices
5. All optimization functionality from SI version
"""



import math
import torch
import matplotlib.pyplot as plt
from torchdiffeq import odeint_adjoint, odeint
from dataclasses import dataclass
from scipy.optimize import fsolve
import numpy as np
from typing import Tuple, Optional, Union
import time



# CRITICAL: PCGrad import for multi-scenario optimization





@dataclass
class Setpoints:
    """
    System setpoints for power system operation.
    All values in per-unit.
    """
    v_star: torch.Tensor      # Voltage magnitude setpoints [pu]
    p_star: torch.Tensor      # Active power setpoints [pu]
    q_star: torch.Tensor      # Reactive power setpoints [pu]
    theta_star: torch.Tensor  # Relative voltage angles [rad]




import numpy as np
from scipy.optimize import fsolve, root, least_squares, newton_krylov

def super_safe_solve(F, x0,
                     jac=None,
                     scale=None,
                     tol=1e-8,
                     maxfev=40000,
                     methods=("fsolve","hybr","trf","krylov"),
                     n_multistart=3,
                     multistart_radius=1e-1):
    """
    Try many root‐finding strategies in turn, with optional scaling
    and multi‐start.

    Parameters
    ----------
    F      : callable F(x) -> shape (n,)
    x0     : initial guess array, shape (n,)
    jac    : callable J(x)->(n,n) or None
    scale  : array of length n for variable scaling, or None
    tol    : required residual norm
    maxfev : max function evals for fsolve / least_squares
    methods: tuple of methods to try in order:
             'fsolve' -> scipy.optimize.fsolve
             'hybr'   -> scipy.optimize.root(method='hybr')
             'trf'    -> scipy.optimize.least_squares(method='trf')
             'krylov' -> scipy.optimize.newton_krylov
    n_multistart : how many random starting points if all methods fail
    multistart_radius : spread for random restarts around x0
    """
    # 1) handle scaling
    if scale is None:
        scale = np.ones_like(x0)
    invscale = 1.0/scale

    def G(u):
        x = invscale * u
        return scale * F(x)

    def g_jac(u):
        x = invscale * u
        Jx = jac(x)
        # chain‐rule: dG/du = scale * Jx * diag(invscale)
        return (scale[:,None] * Jx) * invscale[None,:]

    # helper to test solution
    def test_solution(x):
        res = np.linalg.norm(F(x))
        return res < tol, res

    # 2) try each method once
    for method in methods:
        try:
            if method == "fsolve":
                sol, info, ier, msg = fsolve(
                    G, scale*x0,
                    fprime=(g_jac if jac else None),
                    full_output=True,
                    maxfev=maxfev
                )
                x_hat = invscale * sol
                ok, res = test_solution(x_hat)

            elif method == "hybr":
                out = root(G, scale*x0, jac=(g_jac if jac else None),
                           method="hybr", options={"maxfev": maxfev})
                x_hat = invscale * out.x
                ok, res = (out.success, np.linalg.norm(F(x_hat)))

            elif method == "trf":
                out = least_squares(
                    G, scale*x0,
                    jac=(g_jac if jac else '2-point'),
                    ftol=tol, xtol=tol,
                    max_nfev=maxfev,
                    method="trf"
                )
                x_hat = invscale * out.x
                ok, res = (out.success, np.linalg.norm(F(x_hat)))

            elif method == "krylov":
                sol = newton_krylov(G, scale*x0, f_tol=tol, maxiter=100)
                x_hat = invscale * sol
                ok, res = test_solution(x_hat)

            else:
                continue

            if ok:
                return x_hat, True, res, f"Converged with {method}"

        except Exception as e:
            # ignore and try next
            pass

    # 3) multi‑start around x0
    for i in range(n_multistart):
        x0p = x0 + multistart_radius * np.random.randn(*x0.shape)
        try:
            sol, info, ier, msg = fsolve(
                G, scale*x0p,
                fprime=(g_jac if jac else None),
                full_output=True,
                maxfev=maxfev
            )
            x_hat = invscale * sol
            ok, res = test_solution(x_hat)
            if ok:
                return x_hat, True, res, f"Converged with fsolve after {i+1} restarts"
        except:
            continue

    # 4) give up
    return None, False, None, "All methods failed to converge"




class PerUnitSystem:
    """
    Per-unit system calculations and conversions.
    """
    def __init__(self, Sb=1e3, Vb=120.0, fb=60.0, device='cpu', dtype=torch.float64):
        """
        Initialize per-unit system with base values.

        Args:
            Sb: Base apparent power (VA)
            Vb: Base RMS phase-to-neutral voltage (V)
            fb: Base frequency (Hz)
        """
        self.Sb = Sb
        self.Vb = Vb
        self.fb = fb
        self.ωb = 2 * math.pi * fb
        self.Ib = Sb / Vb
        self.Zb = Vb**2 / Sb
        self.Lb = self.Zb / self.ωb
        self.Cb = 1.0 / (self.ωb * self.Zb)
        self.Yb = 1.0 / self.Zb
        self.device = device
        self.dtype = dtype

    def to_pu(self, value, quantity):
        """Convert SI value to per-unit."""
        conversions = {
            'voltage': value / self.Vb,
            'current': value / self.Ib,
            'power': value / self.Sb,
            'impedance': value / self.Zb,
            'resistance': value / self.Zb,
            'reactance': value / self.Zb,
            'inductance': value * self.ωb / self.Zb,
            'capacitance': value * self.Zb * self.ωb,
            'admittance': value * self.Zb,
            'conductance': value * self.Zb,
            'frequency': value / self.fb,
            'angular_frequency': value / self.ωb
        }
        return conversions.get(quantity, value)

    def from_pu(self, value, quantity):
        """Convert per-unit value to SI."""
        conversions = {
            'voltage': value * self.Vb,
            'current': value * self.Ib,
            'power': value * self.Sb,
            'impedance': value * self.Zb,
            'resistance': value * self.Zb,
            'reactance': value * self.Zb,
            'inductance': value * self.Zb / self.ωb,
            'capacitance': value / (self.Zb * self.ωb),
            'admittance': value / self.Zb,
            'conductance': value / self.Zb,
            'frequency': value * self.fb,
            'angular_frequency': value * self.ωb
        }
        return conversions.get(quantity, value)





class PowerSystemNetwork:
    """
    Power system network model in per-unit.
    """
    def __init__(self, device='cuda', dtype=torch.float64):
        self.device = device
        self.dtype = dtype

        # Initialize per-unit system
        self.pu = PerUnitSystem(Sb=1000.0, Vb=120.0, fb=60.0, device=device, dtype=dtype)

        # Network size
        self.Nc = 3  # Number of converters
        self.Nt = 3  # Number of transmission lines
        self.n = 2   # αβ dimension

        # Network parameters in SI (from Table I)
        rt_si = 0.05        # Line resistance (50 mΩ)
        lt_si = 0.2e-3      # Line inductance (0.2 mH)
        rL_si = 115.0       # Load resistance (115 Ω)

        # Convert to per-unit
        self.rt = self.pu.to_pu(rt_si, 'resistance')
        self.lt = self.pu.to_pu(lt_si, 'inductance')
        self.rL = self.pu.to_pu(rL_si, 'resistance')
        self.rt = torch.tensor(self.pu.to_pu(rt_si, 'resistance'),
                       dtype=dtype, device=device)
        self.lt = torch.tensor(self.pu.to_pu(lt_si, 'inductance'),
                              dtype=dtype, device=device)
        self.omega0 = torch.tensor(1.0, dtype=dtype, device=device)

        self.omega0 = 1.0  # Nominal frequency in pu

        # Store original values for scenario changes
        self.original_rL = torch.tensor(self.rL, dtype=dtype, device=device)

        # Calculate kappa
        self.kappa = math.atan(self.omega0 * self.lt / self.rt)

        # R(κ) matrix
        R_kappa_base = torch.tensor([
            [math.cos(self.kappa), -math.sin(self.kappa)],
            [math.sin(self.kappa), math.cos(self.kappa)]
        ], dtype=self.dtype, device=self.device)

        self.R_kappa = torch.kron(
            torch.eye(self.Nc, dtype=self.dtype, device=self.device),
            R_kappa_base
        )

        # Breaker status
        self.breaker_status = torch.ones(self.Nc, dtype=torch.bool, device=self.device)

        # Standard matrices
        self.J = torch.tensor([[0.0, -1.0], [1.0, 0.0]], dtype=self.dtype, device=self.device)
        self.In = torch.eye(self.n, dtype=self.dtype, device=self.device)

        self.setup_network()

    def setup_network(self):
        """Setup network matrices in αβ frame."""
        # Network topology - radial connection
        B_lines = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=self.dtype, device=self.device)

        self.B_lines = B_lines
        self.B = torch.kron(B_lines, self.In)

        # Transmission line matrices (per-unit)
        self.Rt = torch.kron(
            torch.eye(self.Nt, dtype=self.dtype, device=self.device),
            self.rt * self.In
        )
        self.Lt = torch.kron(
            torch.eye(self.Nt, dtype=self.dtype, device=self.device),
            self.lt * self.In
        )
        self.Jnt = torch.kron(
            torch.eye(self.Nt, dtype=self.dtype, device=self.device),
            self.J
        )

        # Line impedance
        self.Zt = self.Rt + self.omega0 * (self.Jnt @ self.Lt)

        self.update_network_matrices()

    def update_breaker_status(self, status: torch.Tensor):
        """Update breaker status."""
        self.breaker_status = status.to(device=self.device)
        self.update_network_matrices()

    def update_network_matrices(self):
        """Update active network matrices based on breaker status."""
        if torch.all(self.breaker_status):
            # All breakers closed - no need to modify
            self.B_active = self.B
        else:
            # Apply breaker mask (would implement breaker logic here)
            self.B_active = self.B.clone()
            # For now, assuming all breakers stay closed

    def calculate_total_currents(self, v_nodes: torch.Tensor, i_line: torch.Tensor) -> torch.Tensor:
        """Calculate total current injections."""
        return self.B_active @ i_line

    def compute_algebraic_line_currents(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute steady-state line currents algebraically."""
        Nc = self.Nc
        v_nodes = v.view(Nc, 2)

        S = torch.zeros((2, 2), dtype=self.dtype, device=self.device)
        T = torch.zeros((2,), dtype=self.dtype, device=self.device)
        Z_inv_list = []

        for i in range(Nc):
            idx = slice(2*i, 2*(i+1))
            Z_i = self.Zt[idx, idx]
            Z_i_inv = torch.linalg.inv(Z_i)
            Z_inv_list.append(Z_i_inv)
            S = S + Z_i_inv
            T = T + (Z_i_inv @ v_nodes[i])

        I2 = torch.eye(2, dtype=self.dtype, device=self.device)
        # Ensure rL is a scalar
        rL_scalar = self.rL if isinstance(self.rL, (int, float)) else self.rL.item()
        v_common = torch.linalg.solve(I2 + rL_scalar * S, rL_scalar * T)

        i_line_blocks = []
        for i, Z_i_inv in enumerate(Z_inv_list):
            i_line_i = Z_i_inv @ (v_nodes[i] - v_common)
            i_line_blocks.append(i_line_i)
        i_line = torch.cat(i_line_blocks, dim=0)

        return i_line, v_common

    def line_dynamics(self, v: torch.Tensor, i_line: torch.Tensor) -> torch.Tensor:
        """Line dynamics equation in per-unit."""
        i_sum = torch.sum((self.B_active @ i_line).view(self.Nc, 2), dim=0)
        # Ensure rL is a scalar
        rL_scalar = self.rL if isinstance(self.rL, (int, float)) else self.rL.item()
        v_common = rL_scalar * i_sum
        v_diff = (self.B_active.T @ v) - torch.cat([v_common for _ in range(self.Nt)], dim=0)
        rhs = -self.Zt @ i_line + v_diff
        ωb = self.pu.ωb          # 2π·60  = 377

        di_line =  torch.linalg.solve(self.Lt, rhs)
        return di_line





class ConverterControl:
    """
    Converter control implementation in per-unit.
    """
    def __init__(self, network: PowerSystemNetwork, params: dict):
        self.network = network
        self.device = network.device
        self.dtype = network.dtype
        self.pu = network.pu

        # Filter parameters in SI (from Table I)
        rf_si = 0.124    # Filter resistance (124 mΩ)
        lf_si = 1e-3     # Filter inductance (1 mH)
        cf_si = 24e-6    # Filter capacitance (24 μF)

        # Convert to per-unit
        self.rf = self.pu.to_pu(rf_si, 'resistance')
        self.lf = self.pu.to_pu(lf_si, 'inductance')
        self.cf = self.pu.to_pu(cf_si, 'capacitance')

        # Per-converter filter conductances
        gf_values = [1/rf_si, 1/rf_si, 1/rf_si]  # SI values
        self.gf = torch.tensor([self.pu.to_pu(g, 'conductance') for g in gf_values],
                              dtype=self.dtype, device=self.device)

        # Control gains from params (already in per-unit)
        # eta and eta_a are dimensionless in per-unit after proper conversion
        self.eta = params['eta']
        self.eta_a = params['eta_a']
        self.Kp_v = params['Kp_v']
        self.Ki_v = params['Ki_v']
        self.Kp_f = params['Kp_f']
        self.Ki_f = params['Ki_f']

        # Converter states
        self.converter_states = {
            i: {
                'active': False,
                'voltage_control': False,
                'power_control': False
            }
            for i in range(self.network.Nc)
        }

        # Initialize setpoints (will be set by ConverterState)
        self.setpoints = Setpoints(
            v_star=torch.ones(self.network.Nc, dtype=self.dtype, device=self.device),
            p_star=torch.zeros(self.network.Nc, dtype=self.dtype, device=self.device),
            q_star=torch.zeros(self.network.Nc, dtype=self.dtype, device=self.device),
            theta_star=torch.zeros(self.network.Nc, dtype=self.dtype, device=self.device)
        )

        self.setup_converter_matrices()

    def rebuild_control_matrices(self):
        """Rebuild control matrices with current parameters."""
        Nc = self.network.Nc
        In = self.network.In

        self.Kp_v_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Kp_v * In
        )
        self.Ki_v_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Ki_v * In
        )
        self.Kp_f_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Kp_f * In
        )
        self.Ki_f_mat = torch.kron(
            torch.eye(Nc, dtype=self.dtype, device=self.device),
            self.Ki_f * In
        )

    def setup_converter_matrices(self):
        """Setup converter filter and control matrices."""
        Nc = self.network.Nc
        In = self.network.In
        J = self.network.J
        omega0 = self.network.omega0

        # Filter matrices
        self.Rf = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), self.rf * In)
        self.Lf = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), self.lf * In)
        self.Cf = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), self.cf * In)
        self.Gf = torch.kron(torch.diag(self.gf), In)
        self.Jnc = torch.kron(torch.eye(Nc, dtype=self.dtype, device=self.device), J)

        # Filter impedance/admittance
        self.Zf = self.Rf + omega0 * (self.Jnc @ self.Lf)
        self.Yf = -omega0 * (self.Jnc @ self.Cf)

        # Build control matrices
        self.rebuild_control_matrices()

    def update_converter_state(self, idx, active, voltage_control, power_control):
        """Update converter state."""
        self.converter_states[idx].update({
            'active': active,
            'voltage_control': voltage_control,
            'power_control': power_control
        })

    def calculate_K(self, idx, setpoints):
        """Power-sharing matrix K in per-unit."""
        v_star = setpoints.v_star[idx]
        p_star = setpoints.p_star[idx]
        q_star = setpoints.q_star[idx]

        # Use the pre-computed R_kappa matrix from network
        idx_slice = slice(2*idx, 2*(idx+1))
        R_kappa = self.network.R_kappa[idx_slice, idx_slice]

        P = torch.tensor([
            [p_star, q_star],
            [-q_star, p_star]
        ], dtype=self.dtype, device=self.device)

        K_mat = (1.0 / (v_star**2)) * (R_kappa @ P)
        return K_mat

    def calculate_Phi(self, v_hat_local, idx, setpoints):
        """Calculate Φ = 1 - ||v_hat||^2 / v_star^2."""
        v_star = setpoints.v_star[idx]
        norm_vhat = torch.norm(v_hat_local)
        # Add small epsilon to prevent division by zero
        return 1.0 - (norm_vhat**2) / (v_star**2 + 1e-12)

    def filter_dynamics_active(self, idx, v_local, i_f_local, vm_local, v_full, i_line):
        """Active filter dynamics in per-unit."""
        idx_slice = slice(2*idx, 2*(idx+1))
        ωb = self.network.pu.ωb          # 2π·60  = 377

        Yf_local = self.Yf[idx_slice, idx_slice]
        Zf_local = self.Zf[idx_slice, idx_slice]
        i_total = self.network.calculate_total_currents(v_full, i_line)
        i_o_local = i_total[idx_slice]

        dv = (1/ self.cf) * (-Yf_local @ v_local - i_o_local + i_f_local)
        dif = (1 / self.lf) * (-Zf_local @ i_f_local - v_local + vm_local)
        return dv, dif

    def filter_dynamics_inactive(self, idx, v_local, i_f_local, v_full, i_line):
        """Inactive filter dynamics in per-unit."""
        idx_slice = slice(2*idx, 2*(idx+1))
        ωb = self.network.pu.ωb          # 2π·60  = 377

        Yf_local = self.Yf[idx_slice, idx_slice]
        Zf_local = self.Zf[idx_slice, idx_slice]
        i_total = self.network.calculate_total_currents(v_full, i_line)
        i_o_local = i_total[idx_slice]

        dv = (self.network.pu.ωb / self.cf) * (-Yf_local @ v_local - i_o_local + i_f_local)
        dif = torch.zeros_like(i_f_local)  # No current dynamics when inactive
        return dv, dif

    def voltage_control(self, idx, v_node, vhat_node, i_line, zeta_v_node, v_full, setpoints):
        """Voltage control dynamics."""
        if not (self.converter_states[idx]['voltage_control'] and
                self.converter_states[idx]['active']):
            return (torch.zeros(2, dtype=self.dtype, device=self.device),
                   torch.zeros(2, dtype=self.dtype, device=self.device))

        idx_slice = slice(2*idx, 2*(idx+1))
        i_total = self.network.calculate_total_currents(v_full, i_line)
        i_inj = i_total[idx_slice]

        if self.converter_states[idx]['power_control']:
            # Full dVOC
            K = self.calculate_K(idx, setpoints)
            phi_val = self.calculate_Phi(vhat_node, idx, setpoints)

            # Use pre-computed R_kappa
            R_kappa = self.network.R_kappa[idx_slice, idx_slice]

            dvhat = self.eta * (K @ vhat_node - R_kappa @ i_inj +
                               self.eta_a * phi_val * vhat_node)
        else:
            # Basic voltage regulation
            v_star = setpoints.v_star[idx]
            v_mag = torch.norm(v_node) + 1e-12
            dvhat = -self.Kp_v * ((v_mag - v_star) * (v_node / v_mag))

        dzeta_v =  (v_node - vhat_node)
        return dvhat, dzeta_v

    def calculate_reference_current(self, idx, v_node, vhat_node, i_line, zeta_v_node, v_full):
        """Calculate reference current."""
        idx_slice = slice(2*idx, 2*(idx+1))

        if not self.converter_states[idx]['active']:
            return torch.zeros(2, dtype=self.dtype, device=self.device)

        Yf_local = self.Yf[idx_slice, idx_slice]

        if self.converter_states[idx]['voltage_control']:
            i_total = self.network.calculate_total_currents(v_full, i_line)
            i_inj = i_total[idx_slice]

            if self.converter_states[idx]['power_control']:
                i_ref = (Yf_local @ v_node + i_inj -
                        self.Kp_v_mat[idx_slice, idx_slice] @ (v_node - vhat_node) -
                        self.Ki_v_mat[idx_slice, idx_slice] @ zeta_v_node)
            else:
                i_ref = (Yf_local @ v_node -
                        self.Kp_v_mat[idx_slice, idx_slice] @ (v_node - vhat_node) -
                        self.Ki_v_mat[idx_slice, idx_slice] @ zeta_v_node)
        else:
            i_ref = Yf_local @ v_node

        return i_ref

    def current_control(self, idx, v_node, i_f_node, i_ref_node, zeta_f_node):
        """Current control dynamics."""
        if not self.converter_states[idx]['active']:
            return (torch.zeros(2, dtype=self.dtype, device=self.device),
                   torch.zeros(2, dtype=self.dtype, device=self.device))

        idx_slice = slice(2*idx, 2*(idx+1))
        Zf_local = self.Zf[idx_slice, idx_slice]

        vm_local = (Zf_local @ i_f_node + v_node -
                   self.Kp_f_mat[idx_slice, idx_slice] @ (i_f_node - i_ref_node) -
                   self.Ki_f_mat[idx_slice, idx_slice] @ zeta_f_node)

        dzeta_f =( i_f_node - i_ref_node)
        return vm_local, dzeta_f

    def compute_converter_dynamics(self, idx, full_state, setpoints, i_line):
        """Compute full converter dynamics."""
        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Unpack state
        vhat = full_state[0:n_conv]
        v = full_state[n_conv:2*n_conv]
        zeta_v = full_state[2*n_conv:3*n_conv]
        i_f = full_state[3*n_conv:4*n_conv]
        zeta_f = full_state[4*n_conv:5*n_conv]

        idx_slice = slice(2*idx, 2*(idx+1))
        vhat_local = vhat[idx_slice]
        v_local = v[idx_slice]
        zeta_v_local = zeta_v[idx_slice]
        i_f_local = i_f[idx_slice]
        zeta_f_local = zeta_f[idx_slice]
        v_full = v

        # Default zero derivatives
        dvhat = torch.zeros(2, dtype=self.dtype, device=self.device)
        dv_ = torch.zeros(2, dtype=self.dtype, device=self.device)
        dzeta_v_ = torch.zeros(2, dtype=self.dtype, device=self.device)
        dif_ = torch.zeros(2, dtype=self.dtype, device=self.device)
        dzeta_f_ = torch.zeros(2, dtype=self.dtype, device=self.device)

        if self.network.breaker_status[idx]:
            if self.converter_states[idx]['active']:
                # Active converter
                dvhat, dzeta_v_ = self.voltage_control(
                    idx, v_local, vhat_local, i_line, zeta_v_local, v_full, setpoints
                )
                i_ref_local = self.calculate_reference_current(
                    idx, v_local, vhat_local, i_line, zeta_v_local, v_full
                )
                vm_local, dzeta_f_ = self.current_control(
                    idx, v_local, i_f_local, i_ref_local, zeta_f_local
                )
                dv_, dif_ = self.filter_dynamics_active(
                    idx, v_local, i_f_local, vm_local, v_full, i_line
                )
            else:
                # Inactive but connected
                dv_, dif_ = self.filter_dynamics_inactive(
                    idx, v_local, i_f_local, v_full, i_line
                )
        else:
            # Breaker open
            dif_ = -(self.rf/self.lf) * i_f_local

        return dvhat, dv_, dzeta_v_, dif_, dzeta_f_





class ConverterState:
    """Manage converter states and setpoints."""
    def __init__(self, converter_control: ConverterControl):
        self.converter = converter_control
        self.network = converter_control.network
        device = self.network.device
        dtype = self.network.dtype
        pu = self.network.pu

        # Convert SI setpoints to per-unit
        p_star_si = torch.tensor([43.2, 41.0, 41.0], dtype=dtype, device=device)
        q_star_si = torch.tensor([-0.9, 0.5, -0.5], dtype=dtype, device=device)

        self.initial_setpoints = {
            'v_star': torch.ones(3, dtype=dtype, device=device),  # 1.0 pu
            'p_star': pu.to_pu(p_star_si, 'power'),
            'q_star': pu.to_pu(q_star_si, 'power')
        }

        self.final_setpoints = self.initial_setpoints.copy()

        # Connection times
        self.t_connect = {0: 0.0, 1: 0.0, 2: 0.0}

        # Control activation times
        self.t_sequence = {
            0: {'current': 0.0, 'voltage': 0.0, 'power': 0.0},
            1: {'current': 0.0, 'voltage': 0.0, 'power': 0.0},
            2: {'current': 0.0, 'voltage': 0.0, 'power': 0.0}
        }

        self.t_setpoint_change = 3.5

    def update_states(self, t: float):
        """Update converter states at time t."""
        # Update breaker status
        breaker_status = torch.ones(self.network.Nc, dtype=torch.bool, device=self.network.device)
        self.network.update_breaker_status(breaker_status)

        # Update converter states
        for i in range(self.network.Nc):
            if t >= self.t_connect[i]:
                is_active = (t >= self.t_sequence[i]['current'])
                v_ctl = (t >= self.t_sequence[i]['voltage'])
                p_ctl = (t >= self.t_sequence[i]['power'])
                self.converter.update_converter_state(i, is_active, v_ctl, p_ctl)
            else:
                self.converter.update_converter_state(i, False, False, False)

        # Update setpoints
        sps = self.final_setpoints if t >= self.t_setpoint_change else self.initial_setpoints

        self.converter.setpoints = Setpoints(
            v_star=sps['v_star'],
            p_star=sps['p_star'],
            q_star=sps['q_star'],
            theta_star=torch.zeros(self.network.Nc, dtype=self.network.dtype, device=self.network.device)
        )





class MultiConverterSimulation(torch.nn.Module):
    """
    Complete multi-converter simulation with per-unit implementation.

    Control Parameter Scaling Convention:
    ------------------------------------
    All control parameters are stored in per-unit form. The conversion from SI to p.u. is:

    dVOC parameters:
    - η [(rad/s)·Ω] → η_pu = η_SI / ωb (dimensionless in pu)
    - η_a [S] → η_a_pu = η_a_SI × Zb (dimensionless in pu)

    Control gains:
    - Kp_v [V^-1] → Kp_v_pu = Kp_v_SI × Vb
    - Ki_v [(V·s)^-1] → Ki_v_pu = Ki_v_SI × Vb / ωb
    - Kp_f [A^-1] → Kp_f_pu = Kp_f_SI × Ib
    - Ki_f [(A·s)^-1] → Ki_f_pu = Ki_f_SI × Ib / ωb

    where Vb = 120V, Ib = 8.333A, ωb = 377 rad/s, Zb = 14.4Ω
    """
    def __init__(self, device='cuda', dtype=torch.float64):
        super().__init__()

        self.device = device
        self.dtype = dtype
        self.integrate_line_dynamics = False

        # Initialize network first to get base values
        self.network = PowerSystemNetwork(device=device, dtype=dtype)


        # Convert to per-unit
        Vb = self.network.pu.Vb
        Ib = self.network.pu.Ib
        ωb = self.network.pu.ωb

        # Voltage control gains
        Kp_v_SI = 0.6917
        Ki_v_SI = 0.8552
        self.Kp_v = torch.nn.Parameter(
            torch.tensor(Kp_v_SI * self.network.pu.Zb, dtype=dtype, device=device)
        )
        self.Ki_v = torch.nn.Parameter(
            torch.tensor(Ki_v_SI * self.network.pu.Zb  / ωb, dtype=dtype, device=device)
        )

        # Current control gains
        Kp_f_SI = 7.22
        Ki_f_SI = 14.04
        self.Kp_f = torch.nn.Parameter(
            torch.tensor(Kp_f_SI  / self.network.pu.Zb, dtype=dtype, device=device)
        )
        self.Ki_f = torch.nn.Parameter(
            torch.tensor(Ki_f_SI  / (self.network.pu.Zb * self.network.pu.ωb), dtype=dtype, device=device)
        )

        # dVOC parameters

        eta_SI = 9.66  # (rad/s)·Ω
        eta_a_SI = 4.6256  # S (siemens)

        # Convert to per-unit
        self.eta = torch.nn.Parameter(
            torch.tensor(eta_SI /  ( self.network.pu.Zb*self.network.pu.ωb), dtype=dtype, device=device)
        )
        self.eta_a = torch.nn.Parameter(
            torch.tensor(eta_a_SI * self.network.pu.Zb, dtype=dtype, device=device)
        )



        # Lagrange multipliers
        self.lambda_cond4 = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.lambda_cond5 = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.lambda_cond6 = torch.nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))


        # Initialize converter control
        self.converter = ConverterControl(
            self.network,
            {'eta': self.eta, 'eta_a': self.eta_a,
             'Kp_v': self.Kp_v, 'Ki_v': self.Ki_v,
             'Kp_f': self.Kp_f, 'Ki_f': self.Ki_f}
        )
        self.state_handler = ConverterState(self.converter)

        # Simulation parameters
        self.dt = 0.5e-4*self.network.pu.ωb
        self.T_sim = 0.4*self.network.pu.ωb

        # Scenario handling
        self.scenario = "black_start"
        self.disturbance = None
        self.disturbance_applied = False
        self.disturbance_time = 0.05

        # Equilibrium tracking
        self.default_equilibrium_target = None
        self.scenario_equilibrium_targets = {}

        # Constraint cache
        self._constraint_cache = {}

        # Store original load value
        self.original_rL = self.network.rL
    def _as_finite_tensor(self, x):
        """
        Convert x to a tensor on the model's device/dtype.
        If x is already a tensor, re-use its storage.
        Clamp ±inf/NaN to ±1e12 so ReLU never sees inf.
        """
        t = torch.as_tensor(x, dtype=self.dtype, device=self.device)
        t = torch.where(torch.isfinite(t),
                        t,
                        torch.full_like(t, 1e12).copysign(t))
        return t
    def debug_simulation_values(self, t, state, scenario):
        """Debug helper to print simulation values at key time points."""
        if t < 0.001 or abs(t - 0.1) < 0.001 or abs(t - 0.2) < 0.001:
            print(f"\n=== Debug at t={t:.4f} for {scenario} ===")

            Nc = self.network.Nc
            n_conv = 2 * Nc

            if self.integrate_line_dynamics:
                n_line = 2 * self.network.Nt
                v = state[n_conv + n_line:2*n_conv + n_line]
                i_line = state[n_conv:n_conv + n_line]
                print(f"Line currents (pu): {i_line[:6].detach().cpu().numpy()}")
            else:
                v = state[n_conv:2*n_conv]
                i_line, v_common = self.network.compute_algebraic_line_currents(v)
                print(f"Common voltage (pu): {v_common.detach().cpu().numpy()}")

            # Print voltages
            v_mags_pu = [torch.norm(v[2*i:2*i+2]).item() for i in range(Nc)]
            v_mags_si = [self.network.pu.from_pu(vm, 'voltage') for vm in v_mags_pu]
            print(f"Voltage magnitudes (pu): {v_mags_pu}")
            print(f"Voltage magnitudes (SI): {v_mags_si}")

            # Print load value
            rL_si = self.network.pu.from_pu(self.network.rL, 'resistance')
            print(f"Load resistance: {self.network.rL:.4f} pu ({rL_si:.2f} Ω)")

            # Print line impedance
            print(f"Line impedance rt: {self.network.rt:.4f} pu, lt: {self.network.lt:.4f} pu")

    def forward(self, t: torch.Tensor, state: torch.Tensor):
        """ODE forward pass."""
        # Debug output
        if hasattr(self, 'debug_mode') and self.debug_mode:
            self.debug_simulation_values(float(t), state, self.scenario)

        # Disturbance injection
        if (self.scenario == "disturbance" and
            not self.disturbance_applied and
            t >= self.disturbance_time):
            Nc = self.network.Nc
            n_conv = 2 * Nc
            i_f_off = 3*n_conv if not self.integrate_line_dynamics else 3*n_conv + 2*self.network.Nt
            state = state.clone()
            state[i_f_off:i_f_off+2*Nc] += self.disturbance
            self.disturbance_applied = True

        # Choose dynamics
        if self.integrate_line_dynamics:
            return self.system_equations_differential(t, state)
        else:
            return self.system_equations_algebraic(t, state)

    def system_equations_algebraic(self, t, state):
        """System dynamics with algebraic line currents."""
        with torch.no_grad():
            self.state_handler.update_states(float(t))

        # Update control parameters
        self.converter.eta = self.eta
        self.converter.eta_a = self.eta_a
        self.converter.Kp_v = self.Kp_v
        self.converter.Ki_v = self.Ki_v
        self.converter.Kp_f = self.Kp_f
        self.converter.Ki_f = self.Ki_f
        self.converter.rebuild_control_matrices()

        Nc = self.network.Nc
        n_conv = 2 * Nc

        # Unpack state [vhat, v, zeta_v, i_f, zeta_f]
        vhat = state[0:n_conv]
        v = state[n_conv:2*n_conv]
        zeta_v = state[2*n_conv:3*n_conv]
        i_f = state[3*n_conv:4*n_conv]
        zeta_f = state[4*n_conv:5*n_conv]

        # Compute algebraic line currents
        i_line, v_common = self.network.compute_algebraic_line_currents(v)

        # Converter dynamics
        dvhat_list, dv_list, dzeta_v_list, dif_list, dzeta_f_list = [], [], [], [], []

        for idx in range(Nc):
            dvh, dv, dzv, dif, dzf = self.converter.compute_converter_dynamics(
                idx, torch.cat([vhat, v, zeta_v, i_f, zeta_f], 0),
                self.converter.setpoints, i_line
            )
            dvhat_list.append(dvh)
            dv_list.append(dv)
            dzeta_v_list.append(dzv)
            dif_list.append(dif)
            dzeta_f_list.append(dzf)

        return torch.cat([
            torch.cat(dvhat_list),
            torch.cat(dv_list),
            torch.cat(dzeta_v_list),
            torch.cat(dif_list),
            torch.cat(dzeta_f_list)
        ], 0)

    def system_equations_differential(self, t, state):
        """System dynamics with differential line currents."""
        with torch.no_grad():
            self.state_handler.update_states(float(t))

        # Update control parameters
        self.converter.eta = self.eta
        self.converter.eta_a = self.eta_a
        self.converter.Kp_v = self.Kp_v
        self.converter.Ki_v = self.Ki_v
        self.converter.Kp_f = self.Kp_f
        self.converter.Ki_f = self.Ki_f
        self.converter.rebuild_control_matrices()

        Nc = self.network.Nc
        Nt = self.network.Nt
        n_conv = 2 * Nc
        n_line = 2 * Nt

        # Unpack state [vhat, i_line, v, zeta_v, i_f, zeta_f]
        vhat = state[0:n_conv]
        i_line = state[n_conv:n_conv+n_line]
        v = state[n_conv+n_line:2*n_conv+n_line]
        zeta_v = state[2*n_conv+n_line:3*n_conv+n_line]
        i_f = state[3*n_conv+n_line:4*n_conv+n_line]
        zeta_f = state[4*n_conv+n_line:5*n_conv+n_line]

        # Line dynamics
        di_line = self.network.line_dynamics(v, i_line)

        # Converter dynamics
        dvhat_list, dv_list, dzeta_v_list, dif_list, dzeta_f_list = [], [], [], [], []

        for idx in range(Nc):
            dvh, dv, dzv, dif, dzf = self.converter.compute_converter_dynamics(
                idx, torch.cat([vhat, v, zeta_v, i_f, zeta_f], 0),
                self.converter.setpoints, i_line
            )
            dvhat_list.append(dvh)
            dv_list.append(dv)
            dzeta_v_list.append(dzv)
            dif_list.append(dif)
            dzeta_f_list.append(dzf)

        return torch.cat([
            torch.cat(dvhat_list),
            di_line,
            torch.cat(dv_list),
            torch.cat(dzeta_v_list),
            torch.cat(dif_list),
            torch.cat(dzeta_f_list)
        ], 0)

    def initialize_state(self, scenario: str):
        """Initialize state for scenario."""
        if scenario == "black_start":
            return self.initialize_black_start()
        elif scenario == "load_change":
            # For load change, use the computed equilibrium if available

            return self.initialize_from_equilibrium()

    def initialize_black_start(self):
        """Initialize for black start (all zeros with small perturbation)."""
        Nc = self.network.Nc
        n_conv = 2 * Nc
        state_size = 5 * n_conv

        x0 = torch.zeros(state_size, dtype=self.dtype, device=self.device)

        # Small initial voltage perturbation
        for i in range(2):
            idx_vhat = 2 * i
            x0[idx_vhat:idx_vhat+2] = torch.tensor([0.01/120, 0.0], dtype=self.dtype, device=self.device)

        return x0
    def initialize_from_equilibrium(self):
        """Return an initial state in *pu* that matches the expected layout."""
        Nc = self.network.Nc
        n_conv = 2 * Nc          # 6
        n_line = 2 * self.network.Nt  # 6

        if self.integrate_line_dynamics:
            state_size = 5 * n_conv + n_line     # 36
        else:
            state_size = 5 * n_conv              # 30

        # -------- raw equilibrium in SI (length 36) --------------------------
        equilibrium_si = torch.tensor([
            1.19999892e+02, 2.98296769e-02, 1.19999872e+02, 2.78635891e-02,
            1.19999530e+02, 2.82802388e-02, 3.59346305e-01, 5.99563038e-03,
            3.41118357e-01, -5.83582256e-03, 3.42865166e-01, -1.38492159e-04,
            1.20000446e+02, 2.98451641e-02, 1.20000426e+02, 2.78792361e-02,
            1.20000084e+02, 2.82958090e-02, -4.44584345e-04, -2.32653329e-05,
            -4.44971217e-04, -2.35086964e-05, -4.44918487e-04, -2.34119242e-05,
            3.59616321e-01, -1.07974249e+00, 3.41370586e-01, -1.09157377e+00,
            3.43121164e-01, -1.08587334e+00, -1.49898137e-06, 4.54956374e-06,
            -1.42244797e-06, 4.59992635e-06, -1.43564634e-06, 4.58469070e-06
        ], dtype=self.dtype, device=self.device)
        pu = self.network.pu
        x_pu = torch.zeros(state_size, dtype=self.dtype, device=self.device)

        # -------- common blocks ----------------------------------------------
        # v̂ (0-5)
        x_pu[0:6] = pu.to_pu(equilibrium_si[0:6], "voltage")

        if self.integrate_line_dynamics:
            # i_line (6-11)
            x_pu[6:12] = pu.to_pu(equilibrium_si[6:12], "current")

            # v (12-17)
            x_pu[12:18] = pu.to_pu(equilibrium_si[12:18], "voltage")

            # ζ_v (18-23)   V·s ➜ pu
            x_pu[18:24] = equilibrium_si[18:24] * pu.ωb / pu.Vb

            # i_f (24-29)
            x_pu[24:30] = pu.to_pu(equilibrium_si[24:30], "current")

            # ζ_f (30-35)   A·s ➜ pu
            x_pu[30:36] = equilibrium_si[30:36] * pu.ωb / pu.Ib
        else:
            # algebraic mode: skip i_line
            # v (6-11) comes from 12-17 in SI vector
            x_pu[6:12]  = pu.to_pu(equilibrium_si[12:18], "voltage")

            # ζ_v (12-17) comes from 18-23
            x_pu[12:18] = equilibrium_si[18:24] * pu.ωb / pu.Vb

            # i_f (18-23) comes from 24-29
            x_pu[18:24] = pu.to_pu(equilibrium_si[24:30], "current")

            # ζ_f (24-29) comes from 30-35
            x_pu[24:30] = equilibrium_si[30:36] * pu.ωb / pu.Ib

        return x_pu


    def setup_scenario(self, scenario: str):
        """Setup scenario parameters."""
        self.scenario = scenario
        print(f"\n--- Setting up scenario: {scenario} ---")

        # Restore nominal load
        self.network.rL = self.original_rL  # Ensure we get the scalar value
        self.disturbance_applied = False
        self.disturbance = None

        # Set integration mode
        self.integrate_line_dynamics = (scenario == "load_change")

        # Get default equilibrium if needed
        if self.default_equilibrium_target is None:
            print("Calculating default equilibrium target...")
            guess = self.initialize_from_equilibrium()
            self.default_equilibrium_target = self.compute_equilibrium_point(0.0, guess)
            self.scenario_equilibrium_targets["default"] = self.default_equilibrium_target

        target_equilibrium = self.default_equilibrium_target

        # Scenario-specific setup
        if scenario == "load_change":
            # Change load to 10% of nominal
            new_rL_si = 115.0 * .1  # SI value
            self.network.rL = self.network.pu.to_pu(new_rL_si, 'resistance')
   #         print(f"Load changed from: {115.0:.2f} Ω ({self.original_rL():.4f} pu) to: {new_rL_si:.2f} Ω ({self.network.rL:.4f} pu)")

            # CRITICAL: Update network matrices with new load value
            self.network.setup_network()

            # Compute new equilibrium with the changed load
            print("Computing new equilibrium for changed load...")
         #   prev_eq = self.scenario_equilibrium_targets.get('load_change')
          #  if prev_eq is None:
                # Use default equilibrium as starting guess
           #     prev_eq = self.default_equilibrium_target
           # eq = self.compute_equilibrium_point(0.0, prev_eq)
            #if eq is not None:
             #   self.scenario_equilibrium_targets['load_change'] = eq
              #  target_equilibrium = eq
               # print("New equilibrium found")
            #else:
               # print("Warning: Could not find equilibrium for new load")

        elif scenario == "black_start":
            print("Black start scenario - algebraic line currents")

        elif scenario == "setpoint_change":
            print("Setpoint change scenario")

        elif scenario == "disturbance":
            print("Disturbance scenario")
            # Set up disturbance (in per-unit)
            self.disturbance = torch.randn(2 * self.network.Nc, dtype=self.dtype, device=self.device) * 0.1

        self.scenario_equilibrium_targets[scenario] = target_equilibrium

    def compute_equilibrium_point(self, t_steady_val=0.0, x0_guess=None):
        """Compute equilibrium point."""
        t_steady = torch.tensor(t_steady_val, dtype=self.dtype, device=self.device)

        # Make sure control states are properly set for equilibrium
        self.state_handler.update_states(t_steady_val)

        # Prepare initial guess
        if x0_guess is None:
            x0_guess_tensor = self.initialize_from_equilibrium()
            if self.integrate_line_dynamics:
                x0_guess_tensor = self.map_states_with_differential_line_currents(x0_guess_tensor)
            x0_guess = x0_guess_tensor.cpu().numpy()
        elif isinstance(x0_guess, torch.Tensor):
            if self.integrate_line_dynamics:
                # Check if x0_guess already has line currents
                expected_size = 5 * 2 * self.network.Nc + 2 * self.network.Nt
                if x0_guess.shape[0] < expected_size:
                    x0_guess = self.map_states_with_differential_line_currents(x0_guess).cpu().numpy()
                else:
                    x0_guess = x0_guess.cpu().numpy()
            else:
                x0_guess = x0_guess.cpu().numpy()

        # Residual function
        def F(x_np):
            x_t = torch.as_tensor(x_np, dtype=self.dtype, device=self.device)
            with torch.no_grad():
                if self.integrate_line_dynamics:
                    dx = self.system_equations_differential(t_steady, x_t)
                else:
                    dx = self.system_equations_algebraic(t_steady, x_t)
            return dx.cpu().numpy()

        # Solve for equilibrium
        x_eq_np, ok, residual, msg = super_safe_solve(F, x0_guess)

        if not ok:
            print(f"[equilibrium] Failed to find equilibrium: {msg}, residual={residual:.3e}")
            return None

        print(f"[equilibrium] Found equilibrium with residual={residual:.3e}")
        return torch.as_tensor(x_eq_np, dtype=self.dtype, device=self.device)

    def map_states_with_differential_line_currents(self, state_without_i_line):
        """Add line currents to state vector."""
        Nc = self.network.Nc
        n_conv = 2 * Nc

        vhat = state_without_i_line[0:n_conv]
        v = state_without_i_line[n_conv:2*n_conv]
        zeta_v = state_without_i_line[2*n_conv:3*n_conv]
        i_f = state_without_i_line[3*n_conv:4*n_conv]
        zeta_f = state_without_i_line[4*n_conv:5*n_conv]

        # Compute line currents
        i_line, _ = self.network.compute_algebraic_line_currents(v)

        return torch.cat([vhat, i_line, v, zeta_v, i_f, zeta_f], dim=0)

    def run_simulation_for_scenario(self, scenario):
        """Run simulation for a scenario."""
        self.setup_scenario(scenario)

        # Time span
        steps = int(self.T_sim / self.dt) + 1
        t_span = torch.linspace(0.0, self.T_sim, steps, dtype=self.dtype, device=self.device)

        # Initial state
        x0 = self.initialize_state(scenario)

        # For differential line dynamics, map the state to include line currents
        if self.integrate_line_dynamics:
            x0 = self.map_states_with_differential_line_currents(x0)

        # Solver options for enhanced neural solver
        options = {
            'rtol': 1e-5,
            'atol': 1e-5,
            'use_preconditioning': True,
            'precond_type': 'block_diagonal',
            'precond_block_sizes': [2 * self.network.Nc, 2 * self.network.Nt, 2 * self.network.Nc, 2 * self.network.Nc, 2 * self.network.Nc, 2 * self.network.Nc],
            'gradient_checkpointing': True,
            'monitor_gradient_norm': True,
            'adaptive_regularization': True,
            'gradient_warmup_steps': 100,
            'use_mixed_precision': True,
            'jacobian_autodiff_threshold': 10
        }

        # Run ODE with odeint_adjoint
        sol = odeint_adjoint(
            func=self,
            y0=x0,
            t=t_span,
            method='enhanced_neural',
            options=options
        )

        return t_span, sol

    def compute_loss(self, t_vec, sol):
        """Compute performance loss in per-unit."""
        if sol is None:
            return torch.tensor(1e6, dtype=self.dtype, device=self.device)

        Nc, Nt = self.network.Nc, self.network.Nt
        n_conv, n_line = 2 * Nc, 2 * Nt

        # Extract voltages based on state format
        if self.integrate_line_dynamics:
            vhat_indices = slice(0, n_conv)
            v_indices = slice(n_conv + n_line, 2*n_conv + n_line)
        else:
            vhat_indices = slice(0, n_conv)
            v_indices = slice(n_conv, 2*n_conv)

        vhat_sol = sol[:, vhat_indices].reshape(-1, Nc, 2)
        v_sol = sol[:, v_indices].reshape(-1, Nc, 2)

        # Voltage magnitudes
        vhat_mag = torch.norm(vhat_sol, dim=2)
        v_mag = torch.norm(v_sol, dim=2)

        # Deviations from 1.0 pu
        v_star = torch.ones(Nc, dtype=self.dtype, device=self.device)
        vhat_dev = (vhat_mag - v_star.unsqueeze(0)) / v_star.unsqueeze(0)
        v_dev = (v_mag - v_star.unsqueeze(0)) / v_star.unsqueeze(0)

        # Loss components
        vhat_loss = torch.max(torch.abs(vhat_dev))
        v_loss = torch.max(torch.abs(v_dev))

        # Oscillation penalty
        num_steps = sol.shape[0]
        latter_half_idx = num_steps // 2
        v_mag_latter = v_mag[latter_half_idx:, :]
        peak_to_peak = torch.max(v_mag_latter, dim=0)[0] - torch.min(v_mag_latter, dim=0)[0]
        oscillation_penalty = torch.max(peak_to_peak)

        # Total loss
        total_loss = 3.0 * vhat_loss + 10.0 * v_loss + 0.1 * oscillation_penalty

        # Maintain gradients
        if sol.requires_grad and not total_loss.requires_grad:
            total_loss = total_loss + 0.0 * sol.sum()

        return total_loss

    def check_stability_conditions(self, verbose=False):
        """Check all stability conditions in per-unit."""
        cond4 = self.check_condition4()
        cond5 = self.check_condition5()
        cond6 = self.check_condition6()

        all_satisfied = cond4['satisfied'] and cond5['satisfied'] and cond6['satisfied']
        min_margin = min(cond4['margin'], cond5['margin'], cond6['margin'])

        results = {
            "condition4": cond4,
            "condition5": cond5,
            "condition6": cond6,
            "all_satisfied": all_satisfied,
            "min_margin": min_margin
        }

        if verbose:
            print("Stability Conditions Check:")
            print(f"  Condition 4: {'✓' if cond4['satisfied'] else '✗'} (margin: {cond4['margin']:.6f})")
            print(f"  Condition 5: {'✓' if cond5['satisfied'] else '✗'} (margin: {cond5['margin']:.6f})")
            print(f"  Condition 6: {'✓' if cond6['satisfied'] else '✗'} (margin: {cond6['margin']:.6f})")
            print(f"  All satisfied: {'✓' if all_satisfied else '✗'} (min margin: {min_margin:.6f})")
        return results

    def compare_with_si_results(self, t_vec, sol, si_results_dict):
        """Compare per-unit simulation results with SI results.

        Args:
            t_vec: Time vector
            sol: Solution from per-unit simulation
            si_results_dict: Dictionary with SI results containing:
                - 'time': time vector
                - 'voltage_mag': voltage magnitudes in V
                - 'frequency': frequencies in Hz
                - 'active_power': active powers in W
                - 'reactive_power': reactive powers in VAr
        """
        # Process PU results to SI
        pu_results = self.process_results(t_vec, sol)

        print("\n=== SI vs PU Comparison ===")

        # Find closest time indices for comparison
        compare_times = [0.0, 0.05, 0.1, 0.2, 0.3]

        for t_compare in compare_times:
            # Find closest index in both results
            pu_idx = torch.argmin(torch.abs(t_vec - t_compare)).item()
            si_idx = np.argmin(np.abs(si_results_dict['time'] - t_compare))

            print(f"\nAt t = {t_compare:.2f}s:")

            # Compare voltages
            for i in range(self.network.Nc):
                v_pu = pu_results['voltage_mag'][pu_idx, i]
                v_si = si_results_dict['voltage_mag'][si_idx, i]
                diff = abs(v_pu - v_si)
                rel_diff = diff / v_si * 100 if v_si != 0 else float('inf')
                print(f"  Conv {i+1} Voltage: PU={v_pu:.3f}V, SI={v_si:.3f}V, Diff={diff:.3f}V ({rel_diff:.1f}%)")

            # Compare powers
            for i in range(self.network.Nc):
                p_pu = pu_results['active_power'][pu_idx, i]
                p_si = si_results_dict['active_power'][si_idx, i]
                diff = abs(p_pu - p_si)
                rel_diff = diff / abs(p_si) * 100 if p_si != 0 else float('inf')
                print(f"  Conv {i+1} Power: PU={p_pu:.3f}W, SI={p_si:.3f}W, Diff={diff:.3f}W ({rel_diff:.1f}%)")

        # Plot comparison
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        # Voltage comparison
        for i in range(self.network.Nc):
            axes[0, 0].plot(pu_results['time'], pu_results['voltage_mag'][:, i],
                          label=f'Conv {i+1} (PU)', linestyle='-')
            axes[0, 0].plot(si_results_dict['time'], si_results_dict['voltage_mag'][:, i],
                          label=f'Conv {i+1} (SI)', linestyle='--')
        axes[0, 0].set_ylabel('Voltage (V)')
        axes[0, 0].set_title('Voltage Magnitude Comparison')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Power comparison
        for i in range(self.network.Nc):
            axes[0, 1].plot(pu_results['time'], pu_results['active_power'][:, i],
                          label=f'Conv {i+1} (PU)', linestyle='-')
            axes[0, 1].plot(si_results_dict['time'], si_results_dict['active_power'][:, i],
                          label=f'Conv {i+1} (SI)', linestyle='--')
        axes[0, 1].set_ylabel('Power (W)')
        axes[0, 1].set_title('Active Power Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Voltage error
        for i in range(self.network.Nc):
            # Interpolate to common time points
            v_si_interp = np.interp(pu_results['time'], si_results_dict['time'],
                                  si_results_dict['voltage_mag'][:, i])
            error = np.abs(pu_results['voltage_mag'][:, i] - v_si_interp)
            axes[1, 0].plot(pu_results['time'], error, label=f'Conv {i+1}')
        axes[1, 0].set_ylabel('Absolute Error (V)')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_title('Voltage Magnitude Error')
        axes[1, 0].legend()
        axes[1, 0].grid(True)

        # Power error
        for i in range(self.network.Nc):
            # Interpolate to common time points
            p_si_interp = np.interp(pu_results['time'], si_results_dict['time'],
                                  si_results_dict['active_power'][:, i])
            error = np.abs(pu_results['active_power'][:, i] - p_si_interp)
            axes[1, 1].plot(pu_results['time'], error, label=f'Conv {i+1}')
        axes[1, 1].set_ylabel('Absolute Error (W)')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_title('Active Power Error')
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

        return pu_results

    def check_condition4(self):
        """Check condition 4 in per-unit."""
        c_L, p_star_max, v_star_min, d_max = self.compute_cL()
        rho = self.network.lt / self.network.rt
        constant_term = 5 * (p_star_max / v_star_min) + 10 * d_max
        rhs4 = c_L / (2 * rho * d_max * (c_L + constant_term))
        tol = 1e-6
        margin = rhs4 - self.eta + tol
        satisfied = (self.eta < rhs4 + tol)
        return {'satisfied': satisfied, 'margin': margin, 'rhs': rhs4}

    def check_condition5(self):
        """Check condition 5 in per-unit."""
        ratio1 = self.converter.Ki_v / self.converter.Kp_v
        ratio2 =self.network.pu.ωb* self.converter.Ki_v / self.converter.cf

        if torch.any(ratio2 <= 1):
            return {'satisfied': False, 'margin': -float('inf'), 'lhs': None, 'rhs': None}

        lhs = (1 + torch.max(ratio1)) / (torch.min(ratio2) - 1)
        c2 = self.linearize_voltage_dynamics(idx=0)
        norm_BR = self.compute_norm_BR()
        rhs = 4 * self.eta * c2 / (norm_BR * (1 + 4 * self.eta**2))
        tol = 1e-2
        margin = rhs - lhs + tol
        satisfied = (lhs < rhs + tol)
        return {'satisfied': satisfied, 'margin': margin, 'lhs': lhs, 'rhs': rhs}

    def check_condition6(self):
        """Check condition 6 in per-unit."""
        ratio3 = self.converter.Ki_f / self.converter.Kp_f
        ratio4 =self.network.pu.ωb* self.converter.Ki_f / self.converter.lf

        if torch.any(ratio4 <= 1):
            return {'satisfied': False, 'margin': -float('inf'), 'lhs': None, 'rhs': None}

        lhs = (1 + torch.max(ratio3)) / (torch.min(ratio4) - 1)

        # Beta calculations
        beta34 = (1.0 / self.converter.Ki_v + 1.0 / self.converter.Kp_v).max()
        avg_Kp_v = self.converter.Kp_v.mean()
        beta_tilde42 = self.network.omega0 / math.sin(self.network.kappa) + self.eta * avg_Kp_v
        norm_Yf = torch.norm(self.converter.Yf)
        norm_Kp_v = torch.norm(self.converter.Kp_v)
        norm_BR = self.compute_norm_BR()
        beta_tilde43 = norm_Yf + norm_Kp_v + norm_BR + self.converter.Ki_v.max()
        beta_tilde41 = avg_Kp_v
        beta_tilde_gamma4 = 0.0
        beta_squared = beta_tilde41**2 + beta_tilde42**2 + 4 * (beta_tilde43**2)

        c3 = self.linearize_current_dynamics(idx=0)
        denominator = beta34 * beta_squared + c3 * beta_tilde_gamma4
        rhs = 4 * c3 / (denominator + 1e-6)
        tol = 1e-2
        margin = rhs - lhs + tol
        satisfied = (lhs < rhs + tol)
        return {'satisfied': satisfied, 'margin': margin, 'lhs': lhs, 'rhs': rhs}

    def compute_cL(self):
        """Compute c_L and related quantities in per-unit."""
        v_star_squared = self.converter.setpoints.v_star ** 2
        p_star_squared = self.converter.setpoints.p_star ** 2 + self.converter.setpoints.q_star ** 2
        p_star_max = torch.max(p_star_squared)
        v_star_min = torch.min(v_star_squared)
        d_max = self.compute_network_constants()
        c_L = v_star_min / d_max
        return c_L, p_star_max, v_star_min, d_max

    def compute_network_constants(self):
        """Compute network constants in per-unit."""
        rt = self.network.rt
        lt = self.network.lt
        omega0 = self.network.omega0
        Y_branch = 1.0 / torch.sqrt(rt**2 + (omega0 * lt)**2)
        ones_vec = torch.ones(self.network.Nt * 2, device=self.device, dtype=self.dtype)
        Y_diag = torch.diag(Y_branch * ones_vec)
        norm_val = torch.norm(self.network.B @ Y_diag @ self.network.B.T)
        return norm_val

    def compute_norm_BR(self):
        """Compute ||BR|| in per-unit."""
        if "norm_BR" in self._constraint_cache and self._constraint_cache["norm_BR"] is not None:
            return self._constraint_cache["norm_BR"]

        ones_vec = torch.ones(self.network.Nt * 2, device=self.device, dtype=self.dtype)
        Rt_inv = torch.diag(ones_vec / self.network.rt)
        norm_val = torch.norm(self.network.B @ Rt_inv @ self.network.B.T)

        self._constraint_cache["norm_BR"] = norm_val
        return norm_val

    def linearize_voltage_dynamics(self, idx, eps=1e-5):
        """Linearize voltage dynamics."""
        if "c2" in self._constraint_cache and self._constraint_cache["c2"] is not None:
            return self._constraint_cache["c2"]

        setpoints = self.converter.setpoints
        v_eq = torch.tensor([setpoints.v_star[idx], 0.0], dtype=self.dtype, device=self.device)

        def voltage_error_func(vhat_pert):
            vhat = v_eq + vhat_pert
            i_line = torch.zeros(2 * self.network.Nt, dtype=self.dtype, device=self.device)
            zeta_v = torch.zeros(2, dtype=self.dtype, device=self.device)
            v_full = v_eq.repeat(self.network.Nc)
            dvhat, _ = self.converter.voltage_control(idx, v_eq, vhat, i_line, zeta_v, v_full, setpoints)
            return dvhat

        x0 = torch.zeros(2, dtype=self.dtype, device=self.device)
        J_v = torch.autograd.functional.jacobian(voltage_error_func, x0, vectorize=True, create_graph=True)
        eigenvals = torch.linalg.eigvals(-J_v)
        c2 = eigenvals.real.min()

        self._constraint_cache["c2"] = c2
        return c2

    def linearize_current_dynamics(self, idx, eps=1e-5):
        """Linearize current dynamics."""
        if "c3" in self._constraint_cache and self._constraint_cache["c3"] is not None:
            return self._constraint_cache["c3"]

        setpoints = self.converter.setpoints
        v_eq = torch.tensor([setpoints.v_star[idx], 0.0], dtype=self.dtype, device=self.device)
        i_eq = torch.zeros(2, dtype=self.dtype, device=self.device)

        def current_error_func(i_pert):
            i_f = i_eq + i_pert
            zeta_f = torch.zeros(2, dtype=self.dtype, device=self.device)
            i_ref = torch.zeros(2, dtype=self.dtype, device=self.device)
            vm, _ = self.converter.current_control(idx, v_eq, i_f, i_ref, zeta_f)
            return vm

        x0 = torch.zeros(2, dtype=self.dtype, device=self.device)
        J_c = torch.autograd.functional.jacobian(current_error_func, x0, vectorize=True, create_graph=True)
        eigenvals = torch.linalg.eigvals(-J_c)
        c3 = eigenvals.real.min()

        self._constraint_cache["c3"] = c3
        return c3

    def project_parameters(self):
        """Project parameters to valid bounds."""
        with torch.no_grad():
            # Bounds in SI units, then convert to per-unit
            # eta bounds: [(rad/s)·Ω]
            eta_SI_min, eta_SI_max = 0.1, 100.0
            eta_min = eta_SI_min / self.network.pu.ωb
            eta_max = eta_SI_max / self.network.pu.ωb

            # eta_a bounds: [S]
            eta_a_SI_min, eta_a_SI_max = 0.1, 100.0
            eta_a_min = eta_a_SI_min * self.network.pu.Zb
            eta_a_max = eta_a_SI_max * self.network.pu.Zb

            # Convert bounds to per-unit
            Vb = self.network.pu.Vb
            Ib = self.network.pu.Ib
            ωb = self.network.pu.ωb

            Kp_v_min, Kp_v_max = 0.01 * Vb, 10.0 * Vb
            Ki_v_min, Ki_v_max = 0.01 * Vb / ωb, 10.0 * Vb / ωb
            Kp_f_min, Kp_f_max = 0.1 * Ib, 100.0 * Ib
            Ki_f_min, Ki_f_max = 0.1 * Ib / ωb, 200.0 * Ib / ωb

            # Apply bounds
            self.eta.data.clamp_(eta_min, eta_max)
            self.eta_a.data.clamp_(eta_a_min, eta_a_max)
            self.Kp_v.data.clamp_(Kp_v_min, Kp_v_max)
            self.Ki_v.data.clamp_(Ki_v_min, Ki_v_max)
            self.Kp_f.data.clamp_(Kp_f_min, Kp_f_max)
            self.Ki_f.data.clamp_(Ki_f_min, Ki_f_max)

            # Stability constraints
            self.Ki_v.data.clamp_(min=self.converter.cf * 1.01)
            self.Ki_f.data.clamp_(min=self.converter.lf * 1.01)

    def clear_constraint_cache(self):
        """Clear constraint cache."""
        self._constraint_cache = {}

    def initialize_constraint_cache(self):
        """Initialize constraint cache."""
        self._constraint_cache = {
            "norm_BR": None,
            "c2": None,
            "c3": None
        }

    def update_lagrange_multipliers(self, step_size=0.1):
        """Update Lagrange multipliers."""
        with torch.no_grad():
            stability_results = self.check_stability_conditions()

            # Convert to constraint violations
            g4 = -stability_results["condition4"]["margin"]
            g5 = -stability_results["condition5"]["margin"]
            g6 = -stability_results["condition6"]["margin"]

            g4_tensor =  torch.as_tensor(g4, dtype=self.dtype, device=self.device)
            g5_tensor =  torch.as_tensor(g5, dtype=self.dtype, device=self.device)
            g6_tensor =  torch.as_tensor(g6, dtype=self.dtype, device=self.device)

            # Update multipliers
            self.lambda_cond4.data = torch.clamp(
                self.lambda_cond4.data + step_size * torch.relu(g4_tensor),
                min=0.0
            )
            self.lambda_cond5.data = torch.clamp(
                self.lambda_cond5.data + step_size * torch.relu(g5_tensor),
                min=0.0
            )
            self.lambda_cond6.data = torch.clamp(
                self.lambda_cond6.data + step_size * torch.relu(g6_tensor),
                min=0.0
            )

    def compute_lagrangian_loss(self, t_vec, sol, check_constraints_every=1):
        """Compute Lagrangian loss."""
        # Performance loss
        performance_loss = self.compute_loss(t_vec, sol)

        # Check constraints
        if not hasattr(self, '_opt_step_counter'):
            self._opt_step_counter = 0
            check_now = True
        else:
            check_now = (self._opt_step_counter % check_constraints_every == 0)
            self._opt_step_counter += 1

        if check_now:
            self.clear_constraint_cache()
            stability_results = self.check_stability_conditions()
            self._last_stability_results = stability_results
        else:
            stability_results = self._last_stability_results

        # Constraint values
        margin4 = stability_results["condition4"]["margin"]
        margin5 = stability_results["condition5"]["margin"]
        margin6 = stability_results["condition6"]["margin"]

        g4 = -margin4
        g5 = -margin5
        g6 = -margin6



        # Lagrangian terms
        g4_t = self._as_finite_tensor(g4)
        g5_t = self._as_finite_tensor(g5)
        g6_t = self._as_finite_tensor(g6)

        lagrangian_term4 = self.lambda_cond4 * torch.relu(g4_t)
        lagrangian_term5 = self.lambda_cond5 * torch.relu(g5_t)
        lagrangian_term6 = self.lambda_cond6 * torch.relu(g6_t)

        aug_term4 = 0.5 * torch.relu(g4_t) ** 2
        aug_term5 = 0.5 * torch.relu(g5_t) ** 2
        aug_term6 = 0.5 * torch.relu(g6_t) ** 2
        constraint_terms = (lagrangian_term4 + lagrangian_term5 + lagrangian_term6 +
                           aug_term4 + aug_term5 + aug_term6)

        total_loss = performance_loss + constraint_terms

        constraint_info = {
            "lambda4": self.lambda_cond4.item(),
            "lambda5": self.lambda_cond5.item(),
            "lambda6": self.lambda_cond6.item(),
            "g4": float(g4),
            "g5": float(g5),
            "g6": float(g6),
            "scenario": self.scenario
        }

        return total_loss, performance_loss, constraint_terms, constraint_info

    def run_lagrangian_optimization(self, num_epochs=20, learning_rate=0.01, multiplier_step_size=0.1):
        """Run Lagrangian optimization."""
        # Optimizer for primal variables
        primal_params = [p for name, p in self.named_parameters() if not name.startswith('lambda_')]
        optimizer = torch.optim.Adam(primal_params, lr=learning_rate)

        self.initialize_constraint_cache()

        losses = []
        constraint_satisfaction = []

        for epoch in range(num_epochs):
            optimizer.zero_grad()

            # Run simulation
            t_vec, sol = self.run_simulation_for_scenario(self.scenario)

            # Compute loss
            total_loss, perf_loss, constraint_terms, constraint_info = self.compute_lagrangian_loss(t_vec, sol)
            losses.append(total_loss.item())

            # Backprop
            total_loss.backward()

            # Update parameters
            optimizer.step()
            self.project_parameters()

            # Update multipliers
            self.update_lagrange_multipliers(step_size=multiplier_step_size)

            # Check constraints
            stability_results = self.check_stability_conditions()
            all_satisfied = stability_results["all_satisfied"]
            constraint_satisfaction.append(all_satisfied)

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss={total_loss.item():.6f}, Perf={perf_loss.item():.6f}")
                print(f"  λ4={constraint_info['lambda4']:.4f}, λ5={constraint_info['lambda5']:.4f}, λ6={constraint_info['lambda6']:.4f}")
                print(f"  g4={constraint_info['g4']:.6f}, g5={constraint_info['g5']:.6f}, g6={constraint_info['g6']:.6f}")
                print(f"  Constraints satisfied: {all_satisfied}")
                print(f"  Params (PU): η={self.eta.item():.4f}, ηa={self.eta_a.item():.4f}, "
                      f"Kpv={self.Kp_v.item():.4f}, Kiv={self.Ki_v.item():.4f}, "
                      f"Kpf={self.Kp_f.item():.4f}, Kif={self.Ki_f.item():.4f}")

        final_stability = self.check_stability_conditions(verbose=True)
        print(f"Final constraint satisfaction: {final_stability['all_satisfied']}")

        return losses, constraint_satisfaction

    def run_multi_scenario_optimization(self, num_epochs=20, learning_rate=0.05, multiplier_step_size=0.1):
        """Run multi-scenario optimization with PCGrad."""


        # Scenarios and weights
        scenarios = ["load_change"]
        scenario_weights = {s: 1.0/len(scenarios) for s in scenarios}

        print(f"Running multi-scenario optimization with scenarios: {scenarios}")

        # Create optimizer with PCGrad
        primal_params = [p for n, p in self.named_parameters() if not n.startswith('lambda_')]
        inner_optim = torch.optim.Adam(primal_params, lr=learning_rate)
        optimizer = PCGrad(inner_optim)

        # Tracking
        all_losses = []
        per_scenario_losses = {s: [] for s in scenarios}
        constraint_satisfaction = []

        # Store initial parameters (convert to SI for display)
        initial_params = {
            'eta': self.eta.item() * self.network.pu.ωb,  # Convert to (rad/s)·Ω
            'eta_a': self.eta_a.item() / self.network.pu.Zb,  # Convert to S
            'Kp_v': self.Kp_v.item() / self.network.pu.Vb,
            'Ki_v': self.Ki_v.item() * self.network.pu.ωb / self.network.pu.Vb,  #
            'Kp_f': self.Kp_f.item() / self.network.pu.Ib,  #
            'Ki_f': self.Ki_f.item() * self.network.pu.ωb / self.network.pu.Ib
        }

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            task_losses = {}
            task_scalars = {}

            # Compute losses for each scenario
            for scen in scenarios:
                t_vec, sol = self.run_simulation_for_scenario(scen)
                loss, perf_loss, constraint_terms, constraint_info = self.compute_lagrangian_loss(t_vec, sol)
                task_losses[scen] = loss
                task_scalars[scen] = loss.item()
                per_scenario_losses[scen].append(loss.item())

            # PCGrad backward
            optimizer.pc_backward([task_losses[s] for s in scenarios])
            optimizer.step()

            # Project parameters

            # Update multipliers
            self.update_lagrange_multipliers(step_size=multiplier_step_size)

            # Combined loss for tracking
            combined_loss = sum(scenario_weights[s] * task_losses[s] for s in scenarios)
            all_losses.append(combined_loss.item())

            # Check constraints
            stab = self.check_stability_conditions()
            constraint_satisfaction.append(stab["all_satisfied"])

            if epoch % 1 == 0:
                print(f"Epoch {epoch}: Loss={loss.item():.6f}, Perf={perf_loss.item():.6f}")
                print(f"  λ4={constraint_info['lambda4']:.4f}, λ5={constraint_info['lambda5']:.4f}, λ6={constraint_info['lambda6']:.4f}")
                print(f"  g4={constraint_info['g4']:.6f}, g5={constraint_info['g5']:.6f}, g6={constraint_info['g6']:.6f}")


        # Final report
        final_stability = self.check_stability_conditions(verbose=True)
        print(f"\nFinal constraint satisfaction: {final_stability['all_satisfied']}")

        # Plot progress
        self.plot_optimization_progress(all_losses, per_scenario_losses)

        return all_losses, per_scenario_losses, constraint_satisfaction

    def plot_optimization_progress(self, combined_losses, per_scenario_losses):
        """Plot optimization progress."""
        plt.figure(figsize=(12, 8))

        # Combined loss
        plt.subplot(2, 1, 1)
        plt.plot(combined_losses, 'k-', linewidth=2, label='Combined Loss')
        plt.title('Multi-Scenario Optimization Progress')
        plt.ylabel('Combined Loss')
        plt.grid(True)
        plt.legend()

        # Per-scenario losses
        plt.subplot(2, 1, 2)
        colors = {'black_start': 'b', 'load_change': 'r', 'setpoint_change': 'g', 'disturbance': 'm'}
        for scenario, losses in per_scenario_losses.items():
            color = colors.get(scenario, 'gray')
            plt.plot(losses, color=color, label=scenario.replace('_', ' ').title())

        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def process_results(self, t_vec, sol):
        """Process simulation results to extract quantities in SI units."""
        Nc = self.network.Nc
        Nt = self.network.Nt
        n_conv = 2 * Nc
        n_line = 2 * Nt

        # Determine offsets based on state format
        if self.integrate_line_dynamics:
            off_vhat = 0
            off_iline = n_conv
            off_v = off_iline + n_line
            off_zeta_v = off_v + n_conv
            off_if = off_zeta_v + n_conv
            off_zeta_f = off_if + n_conv
        else:
            off_vhat = 0
            off_v = n_conv
            off_zeta_v = off_v + n_conv
            off_if = off_zeta_v + n_conv
            off_zeta_f = off_if + n_conv
            off_iline = None

        nsamples = sol.shape[0]
        voltage_mag = torch.zeros(nsamples, Nc, dtype=self.dtype)
        freq = torch.zeros(nsamples, Nc, dtype=self.dtype)
        p_out = torch.zeros(nsamples, Nc, dtype=self.dtype)
        q_out = torch.zeros(nsamples, Nc, dtype=self.dtype)

        dt = self.dt
        prev_angle = torch.zeros(Nc, dtype=self.dtype)

        for k in range(nsamples):
            # Update states
            self.state_handler.update_states(float(t_vec[k]))

            statek = sol[k, :]
            v_nodes = statek[off_v:off_v + n_conv]

            # Get line currents
            if self.integrate_line_dynamics:
                i_line = statek[off_iline:off_iline + n_line]
            else:
                i_line, _ = self.network.compute_algebraic_line_currents(v_nodes)

            for i in range(Nc):
                idx_slice = slice(2*i, 2*(i+1))
                v_local = v_nodes[idx_slice]

                # Voltage magnitude (convert to SI)
                v_mag_pu = torch.norm(v_local)
                voltage_mag[k, i] = self.network.pu.from_pu(v_mag_pu, 'voltage')

                # Frequency estimation
                angle = torch.atan2(v_local[1], v_local[0])
                if k > 0:
                    # Handle angle wrapping
                    angle_diff = angle - prev_angle[i]
                    if angle_diff > math.pi:
                        angle_diff -= 2 * math.pi
                    elif angle_diff < -math.pi:
                        angle_diff += 2 * math.pi

                    freq_pu = 1.0 + angle_diff / (2 * math.pi * dt)
                    freq[k, i] = self.network.pu.from_pu(freq_pu, 'frequency')
                else:
                    freq[k, i] = 60.0  # Nominal frequency

                prev_angle[i] = angle

                # Power calculations
                i_total = self.network.calculate_total_currents(v_nodes, i_line)
                i_node = i_total[idx_slice]

                # Power in per unit
                p_pu = v_local[0]*i_node[0] + v_local[1]*i_node[1]
                q_pu = v_local[0]*i_node[1] - v_local[1]*i_node[0]

                # Convert to SI
                p_out[k, i] = self.network.pu.from_pu(p_pu, 'power')
                q_out[k, i] = self.network.pu.from_pu(q_pu, 'power')

        return {
            'time': t_vec.cpu().numpy(),
            'voltage_mag': voltage_mag.cpu().numpy(),
            'frequency': freq.cpu().numpy(),
            'active_power': p_out.cpu().numpy(),
            'reactive_power': q_out.cpu().numpy()
        }

    def plot_results(self, results):
        """Plot simulation results in SI units."""
        t_vec = results['time']
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        # Frequencies
        for i in range(self.network.Nc):
            axes[0].plot(t_vec, results['frequency'][:, i], label=f'Conv {i+1}')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title('System Frequencies')
        axes[0].grid(True)
        axes[0].legend()

        # Voltage magnitudes
        for i in range(self.network.Nc):
            axes[1].plot(t_vec, results['voltage_mag'][:, i], label=f'Conv {i+1}')
        axes[1].set_ylabel('Voltage (V)')
        axes[1].set_title('Voltage Magnitudes')
        axes[1].grid(True)
        axes[1].legend()

        # Active power
        for i in range(self.network.Nc):
            axes[2].plot(t_vec, results['active_power'][:, i], label=f'Conv {i+1}')
        axes[2].set_ylabel('Active Power (W)')
        axes[2].set_title('Active Power')
        axes[2].grid(True)
        axes[2].legend()

        # Reactive power
        for i in range(self.network.Nc):
            axes[3].plot(t_vec, results['reactive_power'][:, i], label=f'Conv {i+1}')
        axes[3].set_ylabel('Reactive Power (VAr)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title('Reactive Power')
        axes[3].grid(True)
        axes[3].legend()

        plt.tight_layout()
        plt.show()

    def process_and_plot_scenario_results(self, t_vec, sol):
        """Process and plot results with scenario information."""
        results = self.process_results(t_vec, sol)

        # Create figure with scenario title
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

        scenario_titles = {
            'black_start': 'Black Start Scenario',
            'load_change': f'Load Change Scenario (rL: {self.network.pu.from_pu(self.network.rL, "resistance"):.2f}Ω)',
            'setpoint_change': 'Setpoint Change Scenario',
            'disturbance': f'Disturbance Scenario (at t={self.disturbance_time}s)'
        }

        fig.suptitle(scenario_titles.get(self.scenario, 'Simulation Results'), fontsize=16)

        # Plot results
        t_vec_np = results['time']

        # Frequencies
        for i in range(self.network.Nc):
            axes[0].plot(t_vec_np, results['frequency'][:, i], label=f'Conv {i+1}')
        axes[0].set_ylabel('Frequency (Hz)')
        axes[0].set_title('System Frequencies')
        axes[0].grid(True)
        axes[0].legend()

        # Voltages
        for i in range(self.network.Nc):
            axes[1].plot(t_vec_np, results['voltage_mag'][:, i], label=f'Conv {i+1}')
        axes[1].set_ylabel('Voltage (V)')
        axes[1].set_title('Voltage Magnitudes')
        axes[1].grid(True)
        axes[1].legend()

        # Active power
        for i in range(self.network.Nc):
            axes[2].plot(t_vec_np, results['active_power'][:, i], label=f'Conv {i+1}')
        axes[2].set_ylabel('Active Power (W)')
        axes[2].set_title('Active Power')
        axes[2].grid(True)
        axes[2].legend()

        # Reactive power
        for i in range(self.network.Nc):
            axes[3].plot(t_vec_np, results['reactive_power'][:, i], label=f'Conv {i+1}')
        axes[3].set_ylabel('Reactive Power (VAr)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_title('Reactive Power')
        axes[3].grid(True)
        axes[3].legend()

        # Mark disturbance time if applicable
        if self.scenario == 'disturbance':
            for ax in axes:
                ax.axvline(x=self.disturbance_time, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        return results





def test_load_change_calculations():
    """Test key calculations for load change scenario to identify mismatches."""
    print("\n=== Testing Load Change Calculations ===")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # Create simulation
    sim = MultiConverterSimulation(device=device, dtype=dtype)

    # Test nominal load
    print(f"\nNominal load:")
    print(f"  SI: 115.0 Ω")
    print(f"  PU: {sim.network.original_rL.item():.4f}")
    print(f"  Back to SI: {sim.network.pu.from_pu(sim.network.original_rL.item(), 'resistance'):.2f} Ω")

    # Test changed load (10% of nominal)
    new_rL_si = 11.5  # 10% of 115 Ω
    new_rL_pu = sim.network.pu.to_pu(new_rL_si, 'resistance')
    print(f"\nChanged load (10% of nominal):")
    print(f"  SI: {new_rL_si:.2f} Ω")
    print(f"  PU: {new_rL_pu:.4f}")
    print(f"  Back to SI: {sim.network.pu.from_pu(new_rL_pu, 'resistance'):.2f} Ω")

    # Test impedance calculations
    print(f"\nLine impedance:")
    print(f"  rt (SI): {sim.network.pu.from_pu(sim.network.rt, 'resistance'):.4f} Ω")
    print(f"  lt (SI): {sim.network.pu.from_pu(sim.network.lt, 'inductance')*1e3:.4f} mH")
    print(f"  Zt magnitude: {torch.norm(sim.network.Zt[0:2, 0:2]).item():.4f} pu")

    # Test with a simple voltage vector
    v_test = torch.ones(6, dtype=dtype, device=device) * sim.network.pu.to_pu(120.0, 'voltage')
    i_line, v_common = sim.network.compute_algebraic_line_currents(v_test)

    print(f"\nTest calculation with 120V on all converters:")
    print(f"  Input voltage (pu): {v_test[0].item():.4f}")
    print(f"  Common voltage (pu): {torch.norm(v_common).item():.4f}")
    print(f"  Common voltage (SI): {sim.network.pu.from_pu(torch.norm(v_common).item(), 'voltage'):.2f} V")
    print(f"  Line current magnitude (pu): {torch.norm(i_line[0:2]).item():.4f}")
    print(f"  Line current magnitude (SI): {sim.network.pu.from_pu(torch.norm(i_line[0:2]).item(), 'current'):.4f} A")

    return sim


def verify_pu_implementation():
    """Verify the per-unit implementation."""
    print("Verifying per-unit implementation...")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    # Create simulation
    sim = MultiConverterSimulation(device=device, dtype=dtype)

    # Test 1: Check base values
    pu = sim.network.pu
    print(f"\nBase values:")
    print(f"  Sb = {pu.Sb} VA")
    print(f"  Vb = {pu.Vb} V")
    print(f"  Ib = {pu.Ib:.3f} A")
    print(f"  Zb = {pu.Zb:.3f} Ω")
    print(f"  fb = {pu.fb} Hz")
    print(f"  ωb = {pu.ωb:.3f} rad/s")

    # Test 2: Verify conversions
    print(f"\nConversion tests:")

    # Voltage
    v_si = 120.0
    v_pu = pu.to_pu(v_si, 'voltage')
    v_si_back = pu.from_pu(v_pu, 'voltage')
    print(f"  Voltage: {v_si} V -> {v_pu:.3f} pu -> {v_si_back:.3f} V")

    # Power
    p_si = 500.0
    p_pu = pu.to_pu(p_si, 'power')
    p_si_back = pu.from_pu(p_pu, 'power')
    print(f"  Power: {p_si} W -> {p_pu:.3f} pu -> {p_si_back:.3f} W")

    # Impedance
    z_si = 14.4
    z_pu = pu.to_pu(z_si, 'impedance')
    z_si_back = pu.from_pu(z_pu, 'impedance')
    print(f"  Impedance: {z_si} Ω -> {z_pu:.3f} pu -> {z_si_back:.3f} Ω")

    # Test 3: Network parameters
    print(f"\nNetwork parameters:")
    print(f"  rt: {sim.network.pu.from_pu(sim.network.rt, 'resistance'):.3f} Ω ({sim.network.rt:.4f} pu)")
    print(f"  lt: {sim.network.pu.from_pu(sim.network.lt, 'inductance')*1e3:.3f} mH ({sim.network.lt:.4f} pu)")
    print(f"  rL: {sim.network.pu.from_pu(sim.network.rL, 'resistance'):.1f} Ω ({sim.network.rL:.4f} pu)")

    # Test 4: Filter parameters
    print(f"\nFilter parameters:")
    print(f"  rf: {sim.network.pu.from_pu(sim.converter.rf, 'resistance'):.3f} Ω ({sim.converter.rf:.4f} pu)")
    print(f"  lf: {sim.network.pu.from_pu(sim.converter.lf, 'inductance')*1e3:.3f} mH ({sim.converter.lf:.4f} pu)")
    print(f"  cf: {sim.network.pu.from_pu(sim.converter.cf, 'capacitance')*1e6:.1f} μF ({sim.converter.cf:.4f} pu)")

    # Test 5: Control parameters
    print(f"\ndVOC parameters (per-unit):")
    print(f"  eta: {sim.eta.item():.4f} pu (dimensionless)")
    print(f"  eta_a: {sim.eta_a.item():.4f} pu (dimensionless)")

    print(f"\ndVOC parameters (SI):")
    print(f"  eta: {sim.eta.item()*pu.ωb:.4f} (rad/s)·Ω")
    print(f"  eta_a: {sim.eta_a.item()/pu.Zb:.4f} S")

    print(f"\nControl gains (per-unit):")
    print(f"  Kp_v: {sim.Kp_v.item():.4f} pu")
    print(f"  Ki_v: {sim.Ki_v.item():.4f} pu")
    print(f"  Kp_f: {sim.Kp_f.item():.4f} pu")
    print(f"  Ki_f: {sim.Ki_f.item():.4f} pu")

    print(f"\nControl gains (SI):")
    print(f"  Kp_v: {sim.Kp_v.item()/pu.Vb:.4f} V^-1")
    print(f"  Ki_v: {sim.Ki_v.item()*pu.ωb/pu.Vb:.4f} (V·s)^-1")
    print(f"  Kp_f: {sim.Kp_f.item()/pu.Ib:.4f} A^-1")
    print(f"  Ki_f: {sim.Ki_f.item()*pu.ωb/pu.Ib:.4f} (A·s)^-1")

    print("\nVerification complete!")
    return True





# Example usage
if __name__ == "__main__":
    # Verify implementation
    verify_pu_implementation()

    # Run simulation
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64

    sim = MultiConverterSimulation(device=device, dtype=dtype)

    # Choose mode
    mode = "optimize_multi"  # Options: "optimize_single", "optimize_multi", "test"

    if mode == "optimize_single":
        sim.scenario = "load_change"
        print(f"\nRunning optimization for {sim.scenario} scenario...")
        losses, constraint_satisfaction = sim.run_lagrangian_optimization(
            num_epochs=15,
            learning_rate=0.01,
            multiplier_step_size=0.2
        )

    elif mode == "optimize_multi":
        print("\nRunning multi-scenario optimization...")
        losses, per_scenario_losses, constraint_satisfaction = sim.run_multi_scenario_optimization(
            num_epochs=15,
            learning_rate=0.005,
            multiplier_step_size=0.2
        )

        print("\nOptimization completed.")
        print("Final parameters (SI units):")
        print(f"  eta = {sim.eta.item()*sim.network.pu.ωb:.6f} (rad/s)·Ω")
        print(f"  eta_a = {sim.eta_a.item()/sim.network.pu.Zb:.6f} S")
        print(f"  Kp_v = {sim.Kp_v.item()/sim.network.pu.Vb:.6f} V^-1")
        print(f"  Ki_v = {sim.Ki_v.item()*sim.network.pu.ωb/sim.network.pu.Vb:.6f} (V·s)^-1")
        print(f"  Kp_f = {sim.Kp_f.item()/sim.network.pu.Ib:.6f} A^-1")
        print(f"  Ki_f = {sim.Ki_f.item()*sim.network.pu.ωb/sim.network.pu.Ib:.6f} (A·s)^-1")

    elif mode == "test":
        scenarios = ["black_start", "load_change"]

        for scenario in scenarios:
            print(f"\nRunning {scenario} scenario...")
            t_vec, sol = sim.run_simulation_for_scenario(scenario)
            results = sim.process_and_plot_scenario_results(t_vec, sol)

            # Display final values
            print("\nFinal values (SI units):")
            for i in range(sim.network.Nc):
                print(f"Converter {i+1}:")
                print(f"  V = {results['voltage_mag'][-1, i]:.2f} V")
                print(f"  f = {results['frequency'][-1, i]:.2f} Hz")
                print(f"  P = {results['active_power'][-1, i]:.2f} W")
                print(f"  Q = {results['reactive_power'][-1, i]:.2f} VAr")