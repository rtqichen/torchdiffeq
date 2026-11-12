"""Forward sensitivities dy/dtheta for `radau`, all parameter directions in one solve.

Staggered direct method, as in CVODES: the state is solved exactly as by
`odeint(..., method='radau')`, on plain values; after every accepted step the
sensitivities are advanced by one linear solve with the step's exact stage
matrix. The step sequence is the one chosen for `y` -- it is not itself
differentiated -- so `S` is the exact derivative of the discrete solution the
solver produced. This is also what reverse mode through `radau` computes.

For one step, with stages `Z` solving G(Z) = Z - h (A kron I) F(Z) = 0 where
F_j = f(t + c_j h, y + Z_j, theta), differentiating G = 0 gives

    M S_Z = h (A kron I) [J_j S_y + P_j]_j,      M = dG/dZ (exact),

with J_j = df/dy and P_j = df/dtheta at stage j, and S_new = S_y + S_Z[last],
since Radau IIA is stiffly accurate. Between steps `S` is interpolated with the
same collocation polynomial as `y`.
"""
import torch

from .jacobian import JacobianEvaluator
from .misc import Perturb, _check_inputs
from .radau import RadauIIA5Adaptive


def odeint_sensitivity(func, y0, t, params, S0=None, *, rtol=1e-7, atol=1e-9, method='radau', options=None):
    """Solve `dy/dt = func(t, y)` and its forward sensitivities with respect to `params`.

    Args:
        func, y0, t, rtol, atol, options: as for `odeint`. `y0` must be a single
            Tensor (not a tuple).
        params: a sequence of Tensors that `func` uses, each with
            `requires_grad=True` (e.g. `list(model.parameters())`). Their elements
            are flattened, in order, into the `p` parameter directions.
        S0: optional Tensor of shape `(*y0.shape, p)`, the sensitivity of `y0`
            with respect to the parameters. Defaults to zeros. To chain solves,
            pass the last `S` of one call as `S0` of the next.
        method: only `'radau'` is supported.

    Returns:
        `(y, S)`: `y` as `odeint` returns it, and `S` of shape `(len(t), *y0.shape, p)`
        with `S[i, ..., k] = d y(t[i]) / d theta_k`. Neither carries an autograd graph.

    Step size control acts on `y` alone; `S` does not influence the steps (CVODES
    calls this `errconS = False`). Its accuracy follows that of `y`.

    The derivatives of `func` are taken by autograd, so `func` must be
    differentiable with respect to `y` and `params`. With the default
    `options={'jacobian': 'auto'}` this is checked against finite differences on
    the 1st, 2nd, 4th, ... evaluation, and a mismatch raises.
    """
    if method != 'radau':
        raise ValueError("odeint_sensitivity supports method='radau' only, got {!r}".format(method))
    if not torch.is_tensor(y0):
        raise NotImplementedError('odeint_sensitivity does not support a tuple `y0` yet; concatenate the state '
                                  'into one Tensor.')
    params = list(params)
    if len(params) == 0:
        raise ValueError('`params` is empty.')
    for i, p in enumerate(params):
        if not torch.is_tensor(p):
            raise TypeError('params[{}] is not a Tensor'.format(i))
        if not p.requires_grad:
            raise ValueError('params[{}] does not require grad. The derivatives of `func` with respect to the '
                             'parameters are taken by autograd, which needs `requires_grad=True`.'.format(i))
    options = dict(options) if options is not None else {}
    if options.get('grad_jacobian', 'exact') != 'exact':
        raise ValueError("odeint_sensitivity always uses the exact stage matrix; `grad_jacobian` must be "
                         "'exact' (the default).")
    if options.get('jacobian', 'auto') == 'fd':
        raise ValueError("odeint_sensitivity takes the derivatives of `func` by autograd, so "
                         "`jacobian='fd'` (no autograd) cannot be used with it.")

    shapes, func, y0, t, rtol, atol, method, options, _, _ = _check_inputs(
        func, y0, t, rtol, atol, method, options, None, {'radau': RadauSensitivity})
    assert shapes is None

    solver = RadauSensitivity(func=func, y0=y0, rtol=rtol, atol=atol, params=params, S0=S0, **options)
    with torch.no_grad():
        return solver.integrate_with_sensitivity(t)


class RadauSensitivity(RadauIIA5Adaptive):
    """`radau` that also carries `S = dy/dparams` along the solution."""

    def __init__(self, func, y0, rtol, atol, params, S0=None, **kwargs):
        super(RadauSensitivity, self).__init__(func, y0, rtol, atol, **kwargs)
        self.params = params
        self.p = sum(q.numel() for q in params)
        n = y0.numel()
        if S0 is None:
            S0 = torch.zeros(n, self.p, dtype=y0.dtype, device=y0.device)
        else:
            if tuple(S0.shape) != tuple(y0.shape) + (self.p,):
                raise ValueError('S0 must have shape (*y0.shape, p) = {}, got {}'.format(
                    tuple(y0.shape) + (self.p,), tuple(S0.shape)))
            S0 = S0.detach().to(dtype=y0.dtype, device=y0.device).reshape(n, self.p)
        self.S0 = S0
        # Checks [df/dy | df/dtheta]; reuses the probe of the state Jacobian check.
        self._check = self._jacobian_evaluator.mode == 'auto'
        self._n_derivs = 0
        self._batched = True

    # ------------------------------------------------------------------ driver

    def integrate_with_sensitivity(self, t):
        self._S_out = []
        y = self.integrate(t)
        S = torch.stack([self.S0] + self._S_out)
        return y, S.reshape(len(t), *self.y0.shape, self.p)

    def _before_integrate(self, t):
        super(RadauSensitivity, self)._before_integrate(t)
        self.S = self.S0
        self.S_old = self.S0
        self.QS = None

    def _advance(self, next_t):
        y = super(RadauSensitivity, self)._advance(next_t)
        self._S_out.append(self._dense_sensitivity(next_t))
        return y

    def _advance_until_event(self, event_fn):
        raise NotImplementedError('odeint_sensitivity does not support events.')

    def _dense_sensitivity(self, t_eval):
        if self.QS is None or t_eval == self.t:
            return self.S
        x = ((t_eval - self.t_old) / self.h_old).to(self.S.dtype)
        powers = torch.stack([x, x * x, x * x * x])
        return self.S_old + torch.matmul(self.QS, powers)

    # ------------------------------------------------------------------ the step

    def _after_accepted_step(self, t, y, h, Z):
        n, p = self.n, self.p
        y_flat = y.reshape(-1)
        Js, Ps = [], []
        for k in range(3):
            J, P = self._derivatives(t + self.C[k] * h, y_flat + Z[k], y.shape,
                                     Perturb.PREV if self.C[k] == 1. else Perturb.NONE)
            Js.append(J)
            Ps.append(P)

        M = self._stage_matrix(Js, h, y)
        Kj = torch.stack([torch.matmul(Js[j], self.S) + Ps[j] for j in range(3)])        # (3, n, p)
        rhs = h.to(y.dtype) * torch.einsum('ij,jnp->inp', self.A, Kj)
        LU, piv = torch.linalg.lu_factor(M)
        SZ = torch.linalg.lu_solve(LU, piv, rhs.reshape(3 * n, p)).reshape(3, n, p)
        if not torch.isfinite(SZ).all():
            raise RuntimeError('radau sensitivities: the sensitivity update at t = {:.6g} is not finite.'
                               .format(float(t)))

        # Dense output, the same polynomial in x as for y: S(t_old + x h) = S_old + QS @ [x, x^2, x^3].
        self.QS = torch.einsum('snp,sk->npk', SZ, self.P)
        self.S_old = self.S
        self.S = self.S + SZ[2]

    def _derivatives(self, t, y_flat, shape, perturb):
        """df/dy (n, n) and df/dparams (n, p) at one point, by one batched backward pass."""
        n = y_flat.numel()
        func, params = self.func, self.params
        with torch.inference_mode(False), torch.enable_grad():
            t_ = t.clone()
            z = y_flat.detach().clone().requires_grad_(True)
            out = func(t_, z.view(shape), perturb=perturb).reshape(-1)
            inputs = [z] + params
            if self._batched:
                try:
                    eye = torch.eye(n, dtype=out.dtype, device=out.device)
                    grads = torch.autograd.grad(out, inputs, grad_outputs=eye, is_grads_batched=True,
                                                allow_unused=True)
                except RuntimeError:
                    # vmap over the backward pass is not supported by every op; one row
                    # at a time gives the same result.
                    self._batched = False
                    out = func(t_, z.view(shape), perturb=perturb).reshape(-1)
            if not self._batched:
                rows = [torch.autograd.grad(out[i], inputs, retain_graph=i < n - 1, allow_unused=True)
                        for i in range(n)]
                grads = [None if rows[0][k] is None else torch.stack([r[k] for r in rows])
                         for k in range(len(inputs))]

        def full(g, numel):
            if g is None:  # the output does not depend on this input
                return torch.zeros(n, numel, dtype=y_flat.dtype, device=y_flat.device)
            return g.detach().reshape(n, numel).to(y_flat.dtype)

        J = full(grads[0], n)
        P = torch.cat([full(g, q.numel()) for g, q in zip(grads[1:], params)], dim=1)
        if not (torch.isfinite(J).all() and torch.isfinite(P).all()):
            raise RuntimeError('radau sensitivities: the autograd derivative of `func` is not finite at '
                               't = {:.6g}.'.format(float(t)))

        if self._check:
            self._n_derivs += 1
            if self._n_derivs & (self._n_derivs - 1) == 0:
                self._verify(t, y_flat, shape, perturb, torch.cat([J, P], dim=1))
        return J, P

    def _verify(self, t, y_flat, shape, perturb, JP):
        """Check [df/dy | df/dparams] against a directional finite difference.

        The parameters are perturbed through `.data`, so that the version counters
        of the user's tensors -- which an autograd graph of theirs may rely on --
        are left alone; their values are restored bit for bit before returning.
        """
        n = y_flat.numel()
        params = self.params
        saved = [q.detach().clone() for q in params]
        theta = torch.cat([q.reshape(-1) for q in saved]).to(y_flat.dtype)
        u0 = torch.cat([y_flat, theta])

        def g(u):
            try:
                offset = n
                for q, q0 in zip(params, saved):
                    q.data.copy_(u[offset:offset + q.numel()].view_as(q0).to(q0.dtype))
                    offset += q.numel()
                return self.func(t, u[:n].view(shape), perturb=perturb).reshape(-1)
            finally:
                for q, q0 in zip(params, saved):
                    q.data.copy_(q0)

        if not JacobianEvaluator._agrees_with_finite_difference(g, u0, JP):
            raise RuntimeError(
                'radau sensitivities: the autograd derivatives of `func` with respect to the state and the '
                'parameters disagree with finite differences (at t = {:.6g}). This happens when part of `func` '
                'is computed outside of autograd (numpy/scipy, `.detach()`, `.item()`, ...), or when `func` '
                'uses the parameters through tensors other than the ones passed in `params`. Make `func` fully '
                "differentiable, or pass `options={{'jacobian': 'autograd'}}` to skip this check."
                .format(float(t)))
