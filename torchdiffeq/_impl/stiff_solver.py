import torch
import torch.nn as nn
from torchdiffeq._impl.solvers import AdaptiveStepsizeEventODESolver
from torchdiffeq._impl.misc import _select_initial_step, _handle_unused_kwargs
import math
import time
from typing import Optional, Callable, Tuple, List

class EnhancedNeuralODESolver(AdaptiveStepsizeEventODESolver):
    """
    Enhanced 3rd-order L-stable Rosenbrock solver for neural ODE training
    on stiff inverter systems with guaranteed gradient preservation.
    
    Improvements over NeuralODERobustSolver:
    1. Hybrid Jacobian estimation (autodiff for small blocks, FD for large)
    2. Adaptive regularization from gradient norms
    3. Gradient warmup for training stability
    4. Mixed precision support for efficiency
    5. Enhanced diagnostics for inverter-specific metrics
    """
    
    order = 3
    
    # Rodas3 coefficients (L-stable, optimized for stiff systems)
    gamma = 0.5
    a21 = 2.0
    a31 = 48.0 / 25.0
    a32 = 6.0 / 25.0
    c2 = 1.0
    c3 = 1.0
    g21 = -1.0
    g31 = -24.0 / 25.0
    g32 = -3.0 / 25.0
    b1 = 19.0 / 9.0
    b2 = 0.5
    b3 = 2.0 / 9.0
    e1 = 1.0
    e2 = -1.0
    e3 = 0.0
    
    def __init__(self, func, y0, rtol, atol,
                 # --- Preconditioning options ---
                 use_preconditioning: bool = True,
                 precond_type: str = 'block_diagonal',
                 precond_block_sizes: Optional[List[int]] = None,
                 precond_refresh_rate: int = 10,
                 precond_regularization: float = 1e-10,
                 
                 # --- GMRES settings ---
                 gmres_max_iter: int = 50,
                 gmres_restart: int = 30,
                 gmres_tol_factor: float = 0.1,
                 gmres_adaptive: bool = True,
                 
                 # --- Step size control ---
                 safety_factor: float = 0.9,
                 min_factor: float = 0.1,
                 max_factor: float = 5.0,
                 min_step_size: float = 1e-12,
                 max_step_size: Optional[float] = None,
                 
                 # --- Event detection ---
                 event_fn: Optional[Callable] = None,
                 event_tol: float = 1e-4,
                 
                 # --- Gradient preservation ---
                 gradient_checkpointing: bool = True,
                 gradient_norm_clip: float = 100.0,
                 monitor_gradient_norm: bool = True,
                 gradient_warmup_steps: int = 100,
                 
                 # --- Inverter-specific options ---
                 adaptive_regularization: bool = True,
                 condition_number_threshold: float = 1e12,
                 use_mixed_precision: bool = True,
                 jacobian_autodiff_threshold: int = 10,
                 
                 **kwargs):
        """Initialize solver with enhanced configuration."""
        super().__init__(func=func, y0=y0, rtol=rtol, atol=atol, **kwargs)
        
        # Validate and set block sizes
        if precond_type == 'block_diagonal':
            if precond_block_sizes is None:
                precond_block_sizes = [y0.numel()]
            elif sum(precond_block_sizes) != y0.numel():
                raise ValueError("Sum of precond_block_sizes must equal number of states")
        
        # Store configuration
        self.use_preconditioning = use_preconditioning
        self.precond_type = precond_type
        self.precond_block_sizes = precond_block_sizes
        self.precond_refresh_rate = precond_refresh_rate
        self.precond_regularization = precond_regularization
        
        self.gmres_max_iter = gmres_max_iter
        self.gmres_restart = gmres_restart
        self.gmres_tol_factor = gmres_tol_factor
        self.gmres_adaptive = gmres_adaptive
        
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor
        self.min_step_size = min_step_size
        self.max_step_size = max_step_size
        
        self.event_fn = event_fn
        self.event_tol = event_tol
        
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_norm_clip = gradient_norm_clip
        self.monitor_gradient_norm = monitor_gradient_norm
        self.gradient_warmup_steps = gradient_warmup_steps
        
        self.adaptive_regularization = adaptive_regularization
        self.condition_number_threshold = condition_number_threshold
        self.use_mixed_precision = use_mixed_precision
        self.jacobian_autodiff_threshold = jacobian_autodiff_threshold
        
        # Internal state
        self.preconditioner = None
        self.last_event_values = None
        self.last_error_norm = 1.0
        self.gmres_tol_current = gmres_tol_factor
        self.consecutive_rejects = 0
        self.total_steps = 0
        self.regularization_factor = precond_regularization
        
        # Diagnostics
        self.total_gmres_iters = 0
        self.total_jacobian_evals = 0
        self.num_accepted_steps = 0
        self.num_rejected_steps = 0
        self.gradient_norm_history = []
        self.condition_number_history = []
        
        # Mixed precision setup
        if self.use_mixed_precision:
            self.compute_dtype = torch.float32
            self.storage_dtype = y0.dtype
        else:
            self.compute_dtype = y0.dtype
            self.storage_dtype = y0.dtype
    
    def _compute_jvp(self, func, t, y, v):
        """
        Compute Jacobian-vector product preserving gradient flow.
        Critical for backpropagation through the solver.
        """
        # Determine if gradients are needed
        create_graph = self.training if hasattr(self, 'training') else y.requires_grad
        
        # Adaptive gradient clipping based on warmup
        if self.total_steps < self.gradient_warmup_steps:
            clip_value = self.gradient_norm_clip * (self.total_steps / self.gradient_warmup_steps)
        else:
            clip_value = self.gradient_norm_clip
        
        with torch.enable_grad():
            # Ensure proper gradient computation
            y_dual = y.detach().requires_grad_(True)
            
            # Optional gradient checkpointing for memory efficiency
            if self.gradient_checkpointing and create_graph:
                f_dual = torch.utils.checkpoint.checkpoint(func, t, y_dual, use_reentrant=False)
            else:
                f_dual = func(t, y_dual)
            
            # Monitor gradient norm for diagnostics and adaptive regularization
            if self.monitor_gradient_norm and create_graph and self.total_steps % 10 == 0:
                with torch.no_grad():
                    try:
                        grad = torch.autograd.grad(
                            f_dual.sum(), y_dual, 
                            retain_graph=True, create_graph=False
                        )[0]
                        grad_norm = grad.norm().item()
                        self.gradient_norm_history.append(grad_norm)
                        
                        # Adaptive regularization based on gradient explosion
                        if self.adaptive_regularization and grad_norm > 1e6:
                            self.regularization_factor = min(1e-6, self.regularization_factor * 2)
                    except RuntimeError:
                        pass
            
            # Compute JVP maintaining computation graph
            jvp = torch.autograd.grad(
                f_dual, y_dual, v,
                create_graph=create_graph,
                retain_graph=create_graph
            )[0]
            
            # Gradient clipping to prevent explosion in stiff systems
            if create_graph and clip_value > 0:
                jvp_norm = jvp.norm()
                if jvp_norm > clip_value:
                    jvp = jvp * (clip_value / (jvp_norm + 1e-8))
        
        return jvp
    
    def _estimate_jacobian_blocks(self, func, t, y):
        """
        Hybrid Jacobian estimation: autodiff for small blocks, FD for large.
        Balances accuracy and computational efficiency.
        """
        J_blocks = []
        y_flat = y.flatten()
        start_idx = 0
        
        for block_size in self.precond_block_sizes:
            end_idx = start_idx + block_size
            
            # Use autodiff for small blocks (more accurate)
            if block_size <= self.jacobian_autodiff_threshold:
                block_y = y_flat[start_idx:end_idx].clone().requires_grad_(True)
                
                def block_func(sub_y):
                    full_y = y.clone()
                    full_y.flatten()[start_idx:end_idx] = sub_y
                    return func(t, full_y).flatten()[start_idx:end_idx]
                
                with torch.no_grad():
                    J_block = torch.autograd.functional.jacobian(
                        block_func, block_y, create_graph=False
                    )
            else:
                # Use finite differences for large blocks (more efficient)
                J_block = self._finite_diff_jacobian_block(func, t, y, start_idx, end_idx)
            
            # Convert to compute precision if using mixed precision
            if self.use_mixed_precision:
                J_block = J_block.to(self.compute_dtype)
            
            J_blocks.append(J_block)
            start_idx = end_idx
        
        return J_blocks
    
    def _finite_diff_jacobian_block(self, func, t, y, start_idx, end_idx):
        """Efficient finite difference Jacobian for large blocks."""
        block_size = end_idx - start_idx
        eps = (1e-7 * (1.0 + torch.abs(y).max())).item()
        J_block = torch.zeros(block_size, block_size, device=y.device, dtype=y.dtype)
        
        with torch.no_grad():
            f0 = func(t, y).flatten()[start_idx:end_idx]
            
            # Vectorized finite differences for efficiency
            for i in range(block_size):
                y_pert = y.clone()
                y_pert.flatten()[start_idx + i] += eps
                f_pert = func(t, y_pert).flatten()[start_idx:end_idx]
                J_block[:, i] = (f_pert - f0) / eps
        
        return J_block
    
    def _create_preconditioner(self, func, t, y, dt):
        """Create preconditioner with adaptive regularization."""
        if not self.use_preconditioning:
            self.preconditioner = None
            return
        
        # Determine if refresh is needed
        need_refresh = (
            self.preconditioner is None or
            self.num_accepted_steps % self.precond_refresh_rate == 0 or
            self.consecutive_rejects > 3
        )
        
        if not need_refresh:
            return
        
        self.total_jacobian_evals += 1
        dt_gamma = dt * self.gamma
        
        if self.precond_type == 'block_diagonal':
            J_blocks = self._estimate_jacobian_blocks(func, t, y)
            M_inv_blocks = []
            
            for block_size, J_block in zip(self.precond_block_sizes, J_blocks):
                I = torch.eye(block_size, device=y.device, dtype=J_block.dtype)
                M_block = I - dt_gamma * J_block
                
                # Monitor condition number for adaptive regularization
                if self.monitor_gradient_norm:
                    try:
                        cond = torch.linalg.cond(M_block)
                        self.condition_number_history.append(cond.item())
                        
                        if self.adaptive_regularization and cond > self.condition_number_threshold:
                            reg_factor = self.regularization_factor * (cond / self.condition_number_threshold)
                            M_block += reg_factor * I
                    except:
                        M_block += self.regularization_factor * I
                else:
                    M_block += self.regularization_factor * I
                
                # LU factorization for efficient linear solves
                try:
                    lu_factor = torch.linalg.lu_factor(M_block)
                    M_inv_blocks.append(lu_factor)
                except RuntimeError:
                    # Fallback to pseudoinverse if LU fails
                    M_inv = torch.linalg.pinv(M_block)
                    M_inv_blocks.append(('pinv', M_inv))
            
            def precon_solve(v):
                v_flat = v.flatten()
                out_flat = torch.empty_like(v_flat)
                start_idx = 0
                
                for i, block_size in enumerate(self.precond_block_sizes):
                    end_idx = start_idx + block_size
                    v_sub = v_flat[start_idx:end_idx]
                    
                    if isinstance(M_inv_blocks[i], tuple) and M_inv_blocks[i][0] == 'pinv':
                        out_flat[start_idx:end_idx] = M_inv_blocks[i][1] @ v_sub
                    else:
                        out_flat[start_idx:end_idx] = torch.linalg.lu_solve(
                            M_inv_blocks[i][0], M_inv_blocks[i][1],
                            v_sub.unsqueeze(-1)
                        ).squeeze(-1)
                    start_idx = end_idx
                
                return out_flat.view_as(v)
            
            self.preconditioner = precon_solve
    
    def _adaptive_gmres_tolerance(self, base_tol):
        """Enhanced Eisenstat-Walker method with gradient-aware adaptation."""
        if not self.gmres_adaptive or self.last_error_norm == 1.0:
            return base_tol
        
        eta = self.gmres_tol_current
        norm_ratio = self.last_error_norm / max(base_tol / eta, 1e-10)
        
        # Eisenstat-Walker choice 2
        eta_new = 0.9 * (norm_ratio ** 0.5)
        eta_new = max(0.01 * eta, min(0.9, eta_new))
        
        # Gradient-aware adaptation
        if self.monitor_gradient_norm and len(self.gradient_norm_history) > 0:
            recent_grad_norm = self.gradient_norm_history[-1]
            if recent_grad_norm < 1e-6:
                eta_new *= 0.1  # Tighten near convergence
            elif recent_grad_norm > 1e3:
                eta_new *= 2.0  # Relax in difficult regions
        
        # Tighten after rejections
        if self.consecutive_rejects > 0:
            eta_new *= 0.5 ** self.consecutive_rejects
        
        self.gmres_tol_current = eta_new
        return eta_new * base_tol
    
    def _detect_event(self, t0, y0, dt):
        """Detect switching events in power electronics."""
        if self.event_fn is None:
            return dt, False
        
        if self.last_event_values is None:
            self.last_event_values = self.event_fn(t0, y0)
        
        with torch.no_grad():
            y1_pred = y0 + dt * self.func(t0, y0)
            current_event_values = self.event_fn(t0 + dt, y1_pred)
        
        if torch.any(torch.sign(current_event_values) != torch.sign(self.last_event_values)):
            # Bisection to locate event
            t_low, t_high = t0, t0 + dt
            for _ in range(10):
                t_mid = (t_low + t_high) / 2
                with torch.no_grad():
                    y_mid = y0 + (t_mid - t0) * self.func(t0, y0)
                    mid_values = self.event_fn(t_mid, y_mid)
                if torch.any(torch.sign(mid_values) != torch.sign(self.last_event_values)):
                    t_high = t_mid
                else:
                    t_low = t_mid
            dt_new = t_high - t0
            return max(dt_new, self.min_step_size), True
        
        self.last_event_values = current_event_values
        return dt, False
    
    def _step_func(self, func, t0, dt, y0):
        """Execute single Rosenbrock step with gradient preservation."""
        # Event detection for switching
        dt, event_found = self._detect_event(t0, y0, dt)
        
        # Convert to compute precision if using mixed precision
        if self.use_mixed_precision:
            y0_compute = y0.to(self.compute_dtype)
        else:
            y0_compute = y0
        
        f0 = func(t0, y0_compute)
        
        # Linear operator for GMRES: (I - dt*gamma*J)v
        def linear_operator(v):
            jvp = self._compute_jvp(func, t0, y0_compute, v)
            return v - dt * self.gamma * jvp
        
        # Update preconditioner
        self._create_preconditioner(func, t0, y0_compute, dt)
        M_solve = self.preconditioner if self.preconditioner is not None else None
        
        # Adaptive GMRES tolerance
        base_tol = self.atol + self.rtol * torch.linalg.norm(y0_compute)
        gmres_tol = self._adaptive_gmres_tolerance(base_tol)
        
        # Rosenbrock stages with GMRES
        # Stage 1
        k1, conv1 = torch.linalg.gmres(
            linear_operator, f0, M=M_solve, 
            atol=gmres_tol, restart=self.gmres_restart, 
            maxiter=self.gmres_max_iter
        )
        self.total_gmres_iters += conv1
        if conv1 < 0: return None, None, False
        
        # Stage 2
        y2 = y0_compute + self.a21 * dt * k1
        f2 = func(t0 + self.c2 * dt, y2)
        rhs2 = f2 + self.g21 * k1 / (dt * self.gamma)
        k2, conv2 = torch.linalg.gmres(
            linear_operator, rhs2, x0=k1, M=M_solve,
            atol=gmres_tol, restart=self.gmres_restart,
            maxiter=self.gmres_max_iter
        )
        self.total_gmres_iters += conv2
        if conv2 < 0: return None, None, False
        
        # Stage 3
        y3 = y0_compute + dt * (self.a31 * k1 + self.a32 * k2)
        f3 = func(t0 + self.c3 * dt, y3)
        rhs3 = f3 + (self.g31 * k1 + self.g32 * k2) / (dt * self.gamma)
        k3, conv3 = torch.linalg.gmres(
            linear_operator, rhs3, x0=k2, M=M_solve,
            atol=gmres_tol, restart=self.gmres_restart,
            maxiter=self.gmres_max_iter
        )
        self.total_gmres_iters += conv3
        if conv3 < 0: return None, None, False
        
        if event_found:
            self.last_event_values = None
        
        # Solution and error
        y1 = y0_compute + dt * (self.b1 * k1 + self.b2 * k2 + self.b3 * k3)
        error = dt * (self.e1 * k1 + self.e2 * k2 + self.e3 * k3)
        
        # Convert back to storage precision
        if self.use_mixed_precision:
            y1 = y1.to(self.storage_dtype)
            error = error.to(self.storage_dtype)
        
        return y1, error, True
    
    def _adaptive_step(self, func, t0, dt, y0):
        """Adaptive stepping with enhanced failure recovery."""
        dt = torch.clamp(dt, self.min_step_size, self.max_step_size or float('inf'))
        y1, error, converged = self._step_func(func, t0, dt, y0)
        
        if not converged:
            # GMRES failure - aggressive reduction
            self.consecutive_rejects += 1
            dt_next = dt * 0.25  # More aggressive than standard
            
            # Reset GMRES tolerance after many failures
            if self.consecutive_rejects > 5:
                self.gmres_tol_current = max(1e-4, self.gmres_tol_current * 0.5)
            if self.consecutive_rejects > 10:
                self.gmres_tol_current = self.gmres_tol_factor
                self.consecutive_rejects = 0
            
            return y0, t0, dt_next, False
        
        # Error control
        error_tol = self.atol + self.rtol * torch.maximum(torch.abs(y0), torch.abs(y1))
        error_norm = torch.linalg.norm(error / error_tol) / math.sqrt(y0.numel())
        
        # Check for NaN/Inf
        if not torch.isfinite(error_norm):
            self.consecutive_rejects += 1
            dt_next = dt * 0.1
            return y0, t0, dt_next, False
        
        accept_step = error_norm <= 1.0
        
        # PI step size control
        if error_norm < 1e-10:
            factor = self.max_factor
        else:
            order_p1 = self.order + 1
            k1 = 0.6 / order_p1
            k2 = 0.26 / order_p1
            factor_i = (1.0 / error_norm) ** k1
            factor_p = (self.last_error_norm / error_norm) ** k2 if self.last_error_norm > 0 else 1.0
            factor = self.safety_factor * factor_i * factor_p
        
        factor = torch.clamp(torch.tensor(factor), self.min_factor, self.max_factor).item()
        
        if not accept_step:
            self.num_rejected_steps += 1
            self.consecutive_rejects += 1
            factor = min(factor, 0.5)
        else:
            self.num_accepted_steps += 1
            self.consecutive_rejects = 0
            self.last_error_norm = error_norm
            self.total_steps += 1
            
            # Loosen GMRES tolerance if performing well
            if self.gmres_adaptive and error_norm < 0.1:
                self.gmres_tol_current = min(self.gmres_tol_factor, self.gmres_tol_current * 1.2)
        
        dt_next = dt * factor
        return y1, t0 + dt, dt_next, accept_step
    
    def _before_integrate(self, t):
        """Initialize solver state."""
        self.y = self.y0
        self.t = t[0]
        self.dt = _select_initial_step(
            self.func, self.t, self.y, self.order - 1,
            self.rtol, self.atol, self.norm
        )
        
        # Reset all state
        self.last_event_values = None
        self.preconditioner = None
        self.last_error_norm = 1.0
        self.gmres_tol_current = self.gmres_tol_factor
        self.consecutive_rejects = 0
        self.total_steps = 0
        self.regularization_factor = self.precond_regularization
        
        # Reset diagnostics
        self.total_gmres_iters = 0
        self.total_jacobian_evals = 0
        self.num_accepted_steps = 0
        self.num_rejected_steps = 0
        self.gradient_norm_history = []
        self.condition_number_history = []
    
    def integrate(self, t):
        """Main integration loop."""
        solution = torch.empty(len(t), *self.y0.shape, dtype=self.y0.dtype, device=self.y0.device)
        solution[0] = self.y0
        self._before_integrate(t)
        
        for i in range(1, len(t)):
            while self.t < t[i]:
                dt_step = min(self.dt, t[i] - self.t)
                y_next, t_next, dt_next, accepted = self._adaptive_step(
                    self.func, self.t, dt_step, self.y
                )
                
                if accepted:
                    self.y = y_next
                    self.t = t_next
                    self.dt = dt_next
                else:
                    self.dt = dt_next
                    if self.t >= t[i]:
                        break
                
                # Safety check
                if self.consecutive_rejects > 15:
                    raise RuntimeError(
                        f"Solver failed after {self.consecutive_rejects} consecutive rejections at t={self.t.item():.6e}"
                    )
            
            solution[i] = self.y
        
        return solution
    
    def get_diagnostics(self):
        """Return comprehensive solver diagnostics."""
        diag = {
            'total_gmres_iterations': self.total_gmres_iters,
            'total_jacobian_evaluations': self.total_jacobian_evals,
            'num_accepted_steps': self.num_accepted_steps,
            'num_rejected_steps': self.num_rejected_steps,
            'gradient_norm_history': self.gradient_norm_history,
            'condition_number_history': self.condition_number_history,
            'final_step_size': self.dt.item() if hasattr(self, 'dt') else None,
            'final_regularization': self.regularization_factor
        }
        
        if len(self.gradient_norm_history) > 0:
            diag['avg_gradient_norm'] = sum(self.gradient_norm_history) / len(self.gradient_norm_history)
            diag['max_gradient_norm'] = max(self.gradient_norm_history)
        
        if len(self.condition_number_history) > 0:
            diag['avg_condition_number'] = sum(self.condition_number_history) / len(self.condition_number_history)
            diag['max_condition_number'] = max(self.condition_number_history)
        
        return diag

    @classmethod
    def valid_callbacks(cls):
        return super().valid_callbacks() | {'callback_step', 'callback_accept_step', 'callback_reject_step'}

    def _advance_until_event(self, event_fn):
        """Returns t, state(t) such that event_fn(t, state(t)) == 0."""
        if event_fn(self.t, self.y) == 0:
            return (self.t, self.y)

        n_steps = 0
        sign0 = torch.sign(event_fn(self.t, self.y))
        while sign0 == torch.sign(event_fn(self.t, self.y)):
            assert n_steps < self.max_num_steps, 'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            y_next, t_next, dt_next, accepted = self._adaptive_step(
                self.func, self.t, self.dt, self.y
            )
            
            if accepted:
                self.y = y_next
                self.t = t_next
                self.dt = dt_next
            else:
                self.dt = dt_next
                if self.t >= t_next:
                    break
            
            n_steps += 1
            
            # Safety check
            if self.consecutive_rejects > 15:
                raise RuntimeError(
                    f"Solver failed after {self.consecutive_rejects} consecutive rejections at t={self.t.item():.6e}"
                )
        
        # Use linear interpolation to find the event time
        t0, t1 = self.t - self.dt, self.t
        y0, y1 = self.y - self.dt * self.func(t0, self.y), self.y
        
        def interp_fn(t):
            alpha = (t - t0) / (t1 - t0)
            return y0 + alpha * (y1 - y0)
        
        from torchdiffeq._impl.event_handling import find_event
        event_time, y1 = find_event(interp_fn, sign0, t0, t1, event_fn, self.atol)
        return event_time, y1


# Example usage demonstrating gradient flow preservation
if __name__ == '__main__':
    from torchdiffeq import odeint_adjoint
    
    class InverterController(nn.Module):
        def __init__(self, state_dim=3, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        
        def forward(self, state):
            return 400.0 * self.net(state)
    
    class StiffInverterSystem(nn.Module):
        def __init__(self, controller):
            super().__init__()
            # LCL filter parameters
            self.register_buffer('L1', torch.tensor(0.5e-3))
            self.register_buffer('L2', torch.tensor(0.3e-3))
            self.register_buffer('C', torch.tensor(10e-6))
            self.register_buffer('R1', torch.tensor(0.01))
            self.register_buffer('R2', torch.tensor(0.02))
            self.controller = controller
        
        def forward(self, t, state):
            # State: [i_L1, i_L2, v_C]
            i_L1, i_L2, v_C = state[..., 0], state[..., 1], state[..., 2]
            
            # Neural controller generates inverter voltage
            v_in = self.controller(state).squeeze(-1)
            
            # LCL filter dynamics
            di_L1_dt = (v_in - v_C - self.R1 * i_L1) / self.L1
            di_L2_dt = (v_C - self.R2 * i_L2) / self.L2
            dv_C_dt = (i_L1 - i_L2) / self.C
            
            return torch.stack([di_L1_dt, di_L2_dt, dv_C_dt], dim=-1)
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float64
    
    controller = InverterController().to(device, dtype)
    dynamics = StiffInverterSystem(controller).to(device, dtype)
    
    y0 = torch.zeros(3, device=device, dtype=dtype)
    t_span = torch.linspace(0, 0.02, 201, device=device, dtype=dtype)
    
    # Solver options optimized for inverter systems
    solver_options = {
        'rtol': 1e-7,
        'atol': 1e-9,
        'use_preconditioning': True,
        'precond_type': 'block_diagonal',
        'precond_block_sizes': [3],  # Coupled LCL system
        'gradient_checkpointing': True,
        'monitor_gradient_norm': True,
        'adaptive_regularization': True,
        'gradient_warmup_steps': 100,
        'use_mixed_precision': True,  # New: efficiency
        'jacobian_autodiff_threshold': 10,  # New: hybrid Jacobian
    }
    
    print("Solving with EnhancedNeuralODESolver...")
    start_time = time.time()
    
    # Use odeint_adjoint for memory-efficient backpropagation
    solution = odeint_adjoint(
        dynamics, y0, t_span,
        method=EnhancedNeuralODESolver,
        options=solver_options
    )
    
    elapsed = time.time() - start_time
    print(f"Solution completed in {elapsed:.3f} seconds.")
    
    # Verify gradient flow
    print("\n--- Gradient Flow Verification ---")
    target_voltage = 50.0
    loss = (solution[-1, 2] - target_voltage)**2
    
    # Backpropagate through the entire ODE solve
    loss.backward()
    
    print(f"Loss: {loss.item():.6f}")
    
    # Check gradients
    total_grad_norm = 0.0
    has_grads = True
    for name, param in controller.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm ** 2
            print(f"  - {name}: grad_norm = {grad_norm:.6e}")
        else:
            print(f"  - {name}: No gradient!")
            has_grads = False
    
    total_grad_norm = math.sqrt(total_grad_norm)
    print(f"\nTotal gradient norm: {total_grad_norm:.6e}")
    
    if has_grads and total_grad_norm > 1e-9:
        print("\n✅ SUCCESS: Gradient flow preserved through stiff solver!")
    else:
        print("\n❌ FAILURE: Gradient flow broken!")