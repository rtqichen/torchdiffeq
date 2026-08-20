"""LSODA: variable-order Adams-Moulton and BDF with automatic stiffness switching.

A native torch implementation of the method behind ODEPACK's LSODA, so it supports
GPU tensors, batched states, and gradients -- by direct backpropagation as well as
through the adjoint -- rather than only being reachable through the SciPy wrapper.

The solver keeps its history in a Nordsieck array

    z_n = [y, h y', h^2 y''/2!, ..., h^q y^(q)/q!]

and runs one of two families over it:

* **Adams-Moulton**, order 1-12, corrected by functional iteration. Cheap per step
  and efficient while the problem is not stiff, but its step is limited by stability
  once `|h * lambda|` grows.
* **BDF**, order 1-5, corrected by Newton iteration with a Jacobian. Expensive per
  step, but stable at step sizes Adams could never take.

The switching rule is LSODA's: while running Adams, the convergence rate of the
functional iteration is itself a measurement of `|h * lambda|` -- the iteration
matrix is `h * l0 * df/dy` -- so the dominant eigenvalue can be estimated without
ever forming a Jacobian. When that estimate says the Adams step is being held back
by stability rather than accuracy, the solver switches to BDF; when the stiffness
recedes, it switches back.

References
----------
L. Petzold, *Automatic selection of methods for solving stiff and nonstiff systems
of ordinary differential equations*, SIAM J. Sci. Stat. Comput. 4(1), 1983.

A. C. Hindmarsh, *ODEPACK, a systematized collection of ODE solvers*, in Scientific
Computing, North-Holland, 1983.
"""
from fractions import Fraction as _Fr

import torch

from .event_handling import find_event
from .misc import Perturb, _select_initial_step
from .solvers import AdaptiveStepsizeEventODESolver


# --------------------------------------------------------------------- tableaux

def _poly_mul(p, r):
    out = [_Fr(0)] * (len(p) + 1)
    for i, c in enumerate(p):
        out[i] += c * r
        out[i + 1] += c
    return out


def _integ(p, a, b):
    def F(x):
        return sum(c * _Fr(x) ** (i + 1) / (i + 1) for i, c in enumerate(p))
    return F(b) - F(a)


def _adams_tableau(qmax=12):
    """Nordsieck l-vectors and error constants for Adams-Moulton.

    With p_q(x) = prod_{i=1}^{q-1} (x + i) and rq1fac = 1/(q-1)!,

        l_0 = rq1fac * int_{-1}^{0} p(x) dx,   l_1 = 1,
        l_j = rq1fac * p_{j-1} / j             (j = 2..q),
        E_q = 1 / |rq1fac * int_{-1}^{0} x p(x) dx|,

    where E_q converts the corrector increment into a local error estimate. These
    reproduce the textbook values: q=2 gives [1/2, 1, 1/2] (the trapezoidal rule)
    with E=6, and q=4 gives [3/8, 1, 11/12, 1/3, 1/24].
    """
    el, err = {}, {}
    for q in range(1, qmax + 1):
        p = [_Fr(1)]
        for i in range(1, q):
            p = _poly_mul(p, _Fr(i))
        fac = _Fr(1)
        for i in range(1, q):
            fac *= i
        rq1fac = _Fr(1) / fac
        l = [_Fr(0)] * (q + 1)
        l[0] = rq1fac * _integ(p, -1, 0)
        l[1] = _Fr(1)
        for j in range(2, q + 1):
            l[j] = rq1fac * p[j - 1] / j
        el[q] = [float(x) for x in l]
        err[q] = float(abs(1 / (rq1fac * _integ([_Fr(0)] + p, -1, 0))))
    return el, err


def _bdf_tableau(qmax=5):
    """Nordsieck l-vectors and error constants for BDF.

    Reproduces the textbook values: q=2 gives [2/3, 1, 1/3], q=5 gives
    [60/137, 1, 225/274, 85/274, 15/274, 1/274].
    """
    el, err = {}, {}
    pc = [_Fr(0)] * (qmax + 2)
    pc[0] = _Fr(1)
    for q in range(1, qmax + 1):
        pc[q] = _Fr(0)
        for ib in range(1, q + 1):
            i = q + 1 - ib
            pc[i] = pc[i - 1] + q * pc[i]
        pc[0] = q * pc[0]
        l = [pc[i] / pc[1] for i in range(q + 1)]
        l[1] = _Fr(1)
        el[q] = [float(x) for x in l]
        err[q] = float(_Fr(q + 1) / l[0])
    return el, err


_ADAMS_L, _ADAMS_E = _adams_tableau()
_BDF_L, _BDF_E = _bdf_tableau()

_QMAX = {'adams': 12, 'bdf': 5}
_L = {'adams': _ADAMS_L, 'bdf': _BDF_L}
_E = {'adams': _ADAMS_E, 'bdf': _BDF_E}

# Stability limits of the Adams-Moulton family on the negative real axis, by order
# (ODEPACK's SM1). The largest step Adams can take before it is stability-limited
# rather than accuracy-limited is roughly SM1[q] / |lambda|.
_SM1 = {1: 0.5, 2: 0.575, 3: 0.55, 4: 0.45, 5: 0.35, 6: 0.25,
        7: 0.20, 8: 0.15, 9: 0.10, 10: 0.075, 11: 0.050, 12: 0.025}


class LSODA(AdaptiveStepsizeEventODESolver):
    order = 1  # nominal; the solver varies its own order

    def __init__(self, func, y0, rtol, atol,
                 first_step=None,
                 min_step=0.,
                 max_step=float('inf'),
                 max_order_adams=12,
                 max_order_bdf=5,
                 max_num_steps=2 ** 31 - 1,
                 corrector_iters=4,
                 switch_patience=3,
                 switch_cooldown=10,
                 max_rejects=40,
                 dtype=torch.float64,
                 **kwargs):
        super(LSODA, self).__init__(dtype=dtype, y0=y0, **kwargs)

        dtype = torch.promote_types(dtype, y0.abs().dtype)
        device = y0.device

        self.func = func
        self.rtol = torch.as_tensor(rtol, dtype=dtype, device=device)
        self.atol = torch.as_tensor(atol, dtype=dtype, device=device)
        self.min_step = torch.as_tensor(min_step, dtype=dtype, device=device)
        self.max_step = torch.as_tensor(max_step, dtype=dtype, device=device)
        self.first_step = None if first_step is None else torch.as_tensor(first_step, dtype=dtype, device=device)
        self.max_num_steps = int(max_num_steps)
        self.corrector_iters = int(corrector_iters)
        self.switch_patience = int(switch_patience)
        self._switch_cooldown = int(switch_cooldown)
        self.max_rejects = int(max_rejects)
        self.qmax = {'adams': min(int(max_order_adams), 12), 'bdf': min(int(max_order_bdf), 5)}
        self.dtype = dtype

    @classmethod
    def valid_callbacks(cls):
        return super(LSODA, cls).valid_callbacks() | {'callback_step',
                                                      'callback_accept_step',
                                                      'callback_reject_step'}

    # ------------------------------------------------------------------ set-up

    def _before_integrate(self, t):
        self.n = self.y0.numel()
        self.t = t[0].to(self.dtype)
        self.y = self.y0
        f0 = self.func(self.t, self.y)

        if self.first_step is None:
            h = _select_initial_step(self.func, t[0], self.y0, 1, self.rtol, self.atol, self.norm, f0=f0)
        else:
            h = self.first_step
        self.h = h.abs().clamp(self.min_step, self.max_step).to(self.dtype)

        self.meth = 'adams'
        self.q = 1
        # Nordsieck array: [y, h*y'], order 1.
        self.zn = torch.stack([self.y.reshape(-1), self.h.to(self.y.dtype) * f0.reshape(-1)])

        self.J = None
        self.J_stale = True
        self.e_saved = None         # corrector increment saved for the order-up test
        self.ialth = 2              # steps to run before reconsidering the order
        self.rho = 0.0              # dominant eigenvalue estimate |lambda|
        self.stiff_votes = 0
        self.nonstiff_votes = 0
        self.steps_since_switch = 0

        self.t_old = self.t
        self.zn_old = self.zn
        self.h_old = self.h
        self.n_steps = 0

    # ------------------------------------------------------------------ pieces

    def _ewt(self, y):
        return 1.0 / (self.rtol.to(y.dtype) * y.abs() + self.atol.to(y.dtype))

    def _wnorm(self, v, ewt):
        return float((v.reshape(-1) * ewt).pow(2).mean().sqrt())

    def _jacobian(self, t, y):
        y_ = y.detach()

        def _g(z):
            return self.func(t, z.view_as(y_)).reshape(-1)

        try:
            with torch.enable_grad():
                J = torch.autograd.functional.jacobian(_g, y_.reshape(-1), vectorize=True)
            J = J.reshape(self.n, self.n).detach()
            if torch.isfinite(J).all():
                return J
        except Exception:
            pass
        with torch.no_grad():
            f0 = _g(y_.reshape(-1))
            eps = torch.finfo(y_.abs().dtype).eps ** 0.5
            J = torch.empty(self.n, self.n, dtype=f0.dtype, device=f0.device)
            flat = y_.reshape(-1)
            for i in range(self.n):
                step = eps * max(float(flat[i].abs()), 1.0)
                pert = flat.clone()
                pert[i] = pert[i] + step
                J[:, i] = (_g(pert) - f0) / step
        return J

    @staticmethod
    def _predict(zn, q):
        """Nordsieck predictor: the Taylor shift by one step, as a Pascal sweep."""
        zp = zn.clone()
        for k in range(q):
            for j in range(q, k, -1):
                zp[j - 1] = zp[j - 1] + zp[j]
        return zp

    def _rescale(self, zn, ratio):
        """Rescale a Nordsieck array for a new step size: z[j] *= ratio**j."""
        if ratio == 1.0:
            return zn
        factors = torch.tensor([ratio ** j for j in range(zn.shape[0])],
                               dtype=zn.dtype, device=zn.device)
        return zn * factors.unsqueeze(-1)

    def _corrector_matrix(self, h, l0):
        eye = torch.eye(self.n, dtype=self.y0.dtype, device=self.y0.device)
        return torch.linalg.lu_factor(eye - (h.to(self.y0.dtype) * l0) * self.J)

    @torch.no_grad()
    def _correct(self, t_new, zp, h, l0, lu, ewt):
        """Solve  e = h*f(t_new, y_pred + l0*e) - zp[1]  for the increment e.

        Functional iteration when `lu` is None (Adams), Newton otherwise (BDF).

        Returns `(e, converged, rate)`. `rate` is the measured contraction ratio of
        the iteration, or None when the corrector converged on its first pass and
        there were simply two iterates to compare. That distinction matters: for the
        functional iteration the ratio *is* the estimate of `|h*l0*lambda|` driving
        the stiffness detection, and inventing one from the initial guess reports
        violent stiffness for problems that are not stiff at all.
        """
        e = torch.zeros_like(zp[0])
        conit = 0.5 / (self.q + 2)
        crate = 1.0
        rate = None
        del_prev = None

        for _ in range(self.corrector_iters):
            y_cur = (zp[0] + l0 * e).view_as(self.y0)
            fv = self.func(t_new, y_cur, perturb=Perturb.PREV).reshape(-1)
            resid = h.to(e.dtype) * fv - zp[1] - e
            if not torch.isfinite(resid).all():
                return e, False, rate
            if lu is None:
                de = resid
            else:
                de = torch.linalg.lu_solve(lu[0], lu[1], resid.reshape(-1, 1)).reshape(-1)
            del_norm = self._wnorm(de, ewt)
            e = e + de

            if del_prev is not None and del_prev > 0:
                rate = del_norm / del_prev
                crate = max(0.2 * crate, rate)
            if del_norm * min(1.0, 1.5 * crate) <= conit:
                return e, True, rate
            if del_prev is not None and del_norm > 2.0 * del_prev:
                return e, False, rate
            del_prev = del_norm

        return e, False, rate

    # ------------------------------------------------------------------ step

    def _step(self):
        y_dtype = self.y0.dtype
        self.func.callback_step(self.t, self.y, self.h)
        h = self.h.clamp(self.min_step, self.max_step)
        zn = self.zn
        if h != self.h:
            zn = self._rescale(zn, float(h / self.h))
        n_reject = 0
        jac_retried = False

        while True:
            assert self.t + h > self.t, 'underflow in dt {}'.format(h.item())
            q = self.q
            l = _L[self.meth][q]
            l0 = l[0]
            t_new = self.t + h
            ewt = self._ewt(self.y.reshape(-1).detach())

            zp = self._predict(zn, q)

            lu = None
            if self.meth == 'bdf':
                if self.J is None or self.J_stale:
                    self.J = self._jacobian(self.t, self.y)
                    self.J_stale = False
                lu = self._corrector_matrix(h, l0)

            e, converged, rate = self._correct(t_new, zp, h, l0, lu, ewt)

            if converged:
                dsm = self._wnorm(e, ewt) / _E[self.meth][q]
            else:
                dsm = float('inf')

            if converged and dsm <= 1.0:
                break

            # Rejected.
            self.func.callback_reject_step(self.t, self.y, h)
            n_reject += 1
            if n_reject > self.max_rejects:
                raise RuntimeError(
                    'LSODA: the step at t={:.6g} was rejected {} times in a row. The nonlinear '
                    'solve is not converging -- loosen the tolerances, or check `func` for a '
                    'discontinuity there.'.format(float(self.t), n_reject))

            if not converged and self.meth == 'adams':
                # A functional iteration that will not contract *is* the stiffness
                # signal: its iteration matrix is h*l0*df/dy, so failure means
                # |h*lambda| is past the point where Adams can work at all. Shrinking
                # the step would let it limp on at a ruinous cost -- switching to BDF
                # is the whole reason this solver carries two methods.
                self._switch_to('bdf')
                if zn.shape[0] > self.q + 1:
                    zn = zn[:self.q + 1]
                continue

            if self.meth == 'bdf' and not converged and not jac_retried:
                # Give a fresh Jacobian one chance -- but only one. Keying this off
                # `J_stale` instead loops forever: the top of this loop clears the
                # flag when it rebuilds the Jacobian, so the condition is restored
                # every pass and the step size is never reduced.
                jac_retried = True
                self.J_stale = True
                continue

            if converged:
                factor = max(0.1, 0.9 / (dsm ** (1.0 / (q + 1)) + 1e-6))
            else:
                factor = 0.25
            h_new = (h * factor).clamp(self.min_step, self.max_step)
            zn = self._rescale(zn, float(h_new / h))
            h = h_new
            self.J_stale = True
            jac_retried = False       # a new step size earns a fresh retry
            self.ialth = max(self.ialth, q + 2)   # settle before reconsidering order

        self.func.callback_accept_step(self.t, self.y, h)

        # Re-attach the autograd graph. The corrector ran under `no_grad`, so one
        # differentiable corrector step from the converged point leaves the value
        # alone and supplies the derivative (implicit function theorem).
        if torch.is_grad_enabled() and (self.y.requires_grad or _params_require_grad(self.func)):
            e = self._reattach(t_new, zp, h, l0, e)

        l_t = torch.tensor(l, dtype=y_dtype, device=self.y0.device)
        zn_new = zp + l_t.unsqueeze(-1) * e.unsqueeze(0)

        self.t_old, self.zn_old, self.h_old = self.t, zn, h
        self.t = t_new
        self.zn = zn_new
        self.y = zn_new[0].view_as(self.y0)
        self.h = h
        self.J_stale = True
        self.n_steps += 1
        self.steps_since_switch += 1

        self._update_stiffness(rate, h, l0)
        self._select_order_and_step(e, dsm, ewt)

    def _reattach(self, t_new, zp, h, l0, e):
        """Differentiable corrector step from the converged increment."""
        y_cur = (zp[0] + l0 * e).view_as(self.y0)
        fv = self.func(t_new, y_cur, perturb=Perturb.PREV).reshape(-1)
        resid = h.to(e.dtype) * fv - zp[1] - e
        if not resid.requires_grad:
            return e
        # dG/de = I - h*l0*J, with the true Jacobian at the converged point.
        try:
            J = self._jacobian(t_new, y_cur)
            eye = torch.eye(self.n, dtype=self.y0.dtype, device=self.y0.device)
            M = eye - (h.to(self.y0.dtype) * l0) * J
            return e + torch.linalg.solve(M, resid)
        except Exception:
            return e

    # ------------------------------------------------------- stiffness & order

    def _update_stiffness(self, rate, h, l0):
        """Estimate |lambda| and decide whether the other method would do better.

        In Adams mode the corrector is a plain functional iteration, whose iteration
        matrix is `h*l0*df/dy`; the observed contraction rate therefore measures
        `|h*l0*lambda|` directly, and no Jacobian is needed to detect stiffness. In
        BDF mode there is a Jacobian to hand, so a couple of power iterations on it
        give the same quantity.
        """
        measured = True
        if self.meth == 'adams':
            denom = abs(float(h)) * l0
            if rate is None or denom <= 0:
                measured = False        # nothing was measured; do not invent a value
            else:
                self.rho = rate / denom
        else:
            if self.J is not None:
                self.rho = self._spectral_radius(self.J)
            else:
                measured = False

        # Never switch on the strength of a single step, and never switch straight
        # back: without both guards the two methods trade the integration back and
        # forth every few steps, and the step size never gets a chance to grow.
        if not measured or self.rho <= 0:
            return
        if self.steps_since_switch < max(self.switch_patience, self._switch_cooldown):
            return

        h_abs = abs(float(h))
        if self.meth == 'adams':
            # Stability-limited? Adams cannot exceed roughly SM1[q]/rho.
            stab_step = _SM1[self.q] / self.rho
            if h_abs > stab_step:
                self.stiff_votes += 1
                self.nonstiff_votes = 0
            else:
                self.stiff_votes = 0
            if self.stiff_votes >= self.switch_patience:
                self._switch_to('bdf')
        else:
            # Would Adams now be stable at this step? If so it is much cheaper.
            q_adams = min(self.q, self.qmax['adams'])
            stab_step = _SM1[q_adams] / self.rho
            if h_abs < 0.2 * stab_step:
                self.nonstiff_votes += 1
                self.stiff_votes = 0
            else:
                self.nonstiff_votes = 0
            if self.nonstiff_votes >= self.switch_patience:
                self._switch_to('adams')

    @staticmethod
    def _spectral_radius(J, iters=8):
        v = torch.ones(J.shape[0], dtype=J.dtype, device=J.device)
        v = v / v.norm()
        rho = 0.0
        for _ in range(iters):
            w = J @ v
            nw = float(w.norm())
            if nw == 0:
                return 0.0
            v = w / nw
            rho = nw
        return rho

    def _switch_to(self, meth):
        if meth == self.meth:
            return
        self.meth = meth
        self.q = min(self.q, self.qmax[meth])
        if self.zn.shape[0] > self.q + 1:
            self.zn = self.zn[:self.q + 1]
        self.ialth = self.q + 1
        self.e_saved = None
        self.stiff_votes = 0
        self.nonstiff_votes = 0
        self.steps_since_switch = 0
        self.J_stale = True

    def _select_order_and_step(self, e, dsm, ewt):
        """Pick the order and step size that allow the largest next step.

        A multistep method must not change its order or step on every step: the
        Nordsieck array is a history, and rescaling or re-ordering it perturbs that
        history. So changes are only considered once every `q+1` steps (ODEPACK's
        IALTH counter), and the step is held fixed in between.
        """
        q = self.q
        qmax = self.qmax[self.meth]

        self.ialth -= 1
        if self.ialth > 0:
            if self.ialth == 1 and q < qmax:
                # One step before the next decision: stash the correction so the
                # order-up test has a difference to work with.
                self.e_saved = e.detach().clone()
            return

        rh_same = 1.0 / (1.2 * dsm ** (1.0 / (q + 1)) + 1e-6)

        rh_down = 0.0
        if q > 1:
            ddn = self._wnorm(self.zn[q], ewt) / _E[self.meth][q - 1]
            rh_down = 1.0 / (1.3 * ddn ** (1.0 / q) + 1e-6)

        rh_up = 0.0
        if q < qmax and self.e_saved is not None and self.e_saved.shape == e.shape:
            dup = self._wnorm(e.detach() - self.e_saved, ewt) / _E[self.meth][q + 1]
            rh_up = 1.0 / (1.4 * dup ** (1.0 / (q + 2)) + 1e-6)

        if rh_up > rh_same and rh_up > rh_down:
            new_q, rh = q + 1, rh_up
        elif rh_down > rh_same:
            new_q, rh = q - 1, rh_down
        else:
            new_q, rh = q, rh_same

        rh = min(max(rh, 0.1), 10.0)
        if rh < 1.1 and new_q == q:
            rh = 1.0                      # not worth disturbing the history

        if new_q > q:
            # Nordsieck order increase: the new top term comes from the corrector
            # increment scaled by l[q]/(q+1) of the *old* order (ODEPACK dstode).
            r = _L[self.meth][q][q] / (q + 1)
            self.zn = torch.cat([self.zn, (r * e).unsqueeze(0)], dim=0)
        elif new_q < q:
            self.zn = self.zn[:new_q + 1]
        self.q = new_q
        self.ialth = new_q + 1
        self.e_saved = None

        if rh != 1.0:
            self.zn = self._rescale(self.zn, rh)
            self.h = (self.h * rh).clamp(self.min_step, self.max_step)

    # ------------------------------------------------------------------ driver

    def _advance(self, next_t):
        n_steps = 0
        while next_t > self.t:
            assert n_steps < self.max_num_steps, \
                'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self._step()
            n_steps += 1
        return self._dense(next_t)

    def _advance_until_event(self, event_fn):
        if event_fn(self.t, self.y) == 0:
            return self.t, self.y
        n_steps = 0
        sign0 = torch.sign(event_fn(self.t, self.y))
        while sign0 == torch.sign(event_fn(self.t, self.y)):
            assert n_steps < self.max_num_steps, \
                'max_num_steps exceeded ({}>={})'.format(n_steps, self.max_num_steps)
            self._step()
            n_steps += 1
        return find_event(self._dense, sign0, self.t_old, self.t, event_fn, self.atol)

    def _dense(self, t_eval):
        """The Nordsieck array is the Taylor polynomial, so evaluation is direct."""
        if t_eval == self.t:
            return self.y
        s = ((t_eval - self.t) / self.h).to(self.zn.dtype)
        out = self.zn[0]
        p = torch.ones((), dtype=self.zn.dtype, device=self.zn.device)
        for j in range(1, self.zn.shape[0]):
            p = p * s
            out = out + p * self.zn[j]
        return out.view_as(self.y0)


def _params_require_grad(func):
    try:
        return any(p.requires_grad for p in func.parameters())
    except AttributeError:
        return False
