"""The `df/dy` Jacobian used by the Newton iterations of the implicit solvers.

Autograd gives the exact Jacobian, but it can only see the parts of `func` that it
recorded. Anything computed outside of it -- numpy/scipy, `.detach()`, `.item()`,
or any evaluation under `torch.inference_mode` -- is silently treated as constant,
and the Jacobian that comes back is finite, plausible-looking and wrong. With the
stiff part of the system missing from it the Newton iteration stalls, and the
solver either crawls or fails.

So rather than trusting autograd blindly, `JacobianEvaluator` checks it now and
then against a directional finite difference, and switches to finite differences
for the rest of the solve if the two disagree.
"""
import warnings

import torch


_MODES = ('auto', 'autograd', 'fd')

# Relative mismatch between `J @ w` and its finite difference estimate above which
# the autograd Jacobian is considered wrong. Loose on purpose: a Jacobian with a
# part missing is wrong by O(1), and this only has to tell those apart.
_PROBE_RTOL = 1e-2

# How many times the probe step may be divided by `_PROBE_SHRINK` while it narrows
# down whether a mismatch is curvature (it shrinks with the step) or a genuinely
# wrong Jacobian (it does not).
_PROBE_TRIES = 5
_PROBE_SHRINK = 16.


def _is_power_of_two(k):
    return k & (k - 1) == 0


class JacobianEvaluator:
    """Evaluates `d func(t, y) / dy` as a dense, detached `(n, n)` matrix.

    `mode` selects the source:

    * `'auto'` (default): autograd, checked against a finite difference probe on
      the 1st, 2nd, 4th, 8th, ... evaluation. The first disagreement switches to
      finite differences for the rest of the solve, with a warning. Costs one or
      two extra evaluations of `func` per check, so a handful per solve.
    * `'autograd'`: autograd, unchecked. For a `func` known to be fully
      differentiable. Errors from autograd are raised rather than worked around.
    * `'fd'`: forward finite differences, `n + 1` evaluations of `func` each time.
      For a `func` known not to be differentiable by autograd.

    Under `torch.inference_mode` no autograd graph can be recorded at all, so
    autograd is run with inference mode briefly switched off. That fails if `func`
    uses a tensor that was itself created under inference mode, in which case
    `'auto'` uses finite differences instead.

    One instance belongs to one solve: once it has fallen back to finite
    differences it stays there.
    """

    def __init__(self, mode='auto', solver_name='solver'):
        if mode not in _MODES:
            raise ValueError("`jacobian` must be one of {}, got {!r}".format(_MODES, mode))
        self.mode = mode
        self.solver_name = solver_name
        self.use_fd = mode == 'fd'
        self.n_autograd = 0

    def __call__(self, func, t, y, **kwargs):
        """Jacobian of `func(t, y, **kwargs)` with respect to `y`, both flattened."""
        y_flat = y.detach().reshape(-1)

        def g(z):
            return func(t, z.view_as(y), **kwargs).reshape(-1)

        if self.use_fd:
            return self._finite_difference(g, y_flat)

        try:
            J = self._autograd(func, t, y_flat, y.shape, kwargs)
        except Exception as e:
            if self.mode == 'autograd':
                raise
            self._fall_back(self._autograd_failed_message(e))
            return self._finite_difference(g, y_flat)

        if not torch.isfinite(J).all():
            if self.mode == 'autograd':
                raise RuntimeError("{}: the autograd Jacobian of `func` is not finite. Use "
                                   "`options={{'jacobian': 'auto'}}` to fall back to finite differences "
                                   "when this happens.".format(self.solver_name))
            # Usually a property of the state rather than of `func` (sqrt at 0, say),
            # so only this one evaluation goes to finite differences.
            return self._finite_difference(g, y_flat)
        if self.mode == 'autograd':
            return J

        self.n_autograd += 1
        if _is_power_of_two(self.n_autograd) and not self._agrees_with_finite_difference(g, y_flat, J):
            self._fall_back(
                "{}: the autograd Jacobian of `func` disagrees with finite differences, so finite "
                "differences are used for the rest of this solve. This happens when part of `func` is "
                "computed outside of autograd (numpy/scipy, `.detach()`, `.item()`, ...), which autograd "
                "silently treats as a constant. Make `func` fully differentiable to get the exact Jacobian "
                "back, or pass `options={{'jacobian': 'fd'}}` to skip the check."
                .format(self.solver_name))
            return self._finite_difference(g, y_flat)
        return J

    # ------------------------------------------------------------------ sources

    @staticmethod
    def _autograd(func, t, y_flat, shape, kwargs):
        # `enable_grad` overrides `no_grad` but not `inference_mode`, which has to be
        # left explicitly. The clones turn inference tensors into normal ones there.
        with torch.inference_mode(False), torch.enable_grad():
            t_ = t.clone() if torch.is_tensor(t) else t
            z = y_flat.clone()
            J = torch.autograd.functional.jacobian(lambda z_: func(t_, z_.view(shape), **kwargs).reshape(-1),
                                                   z, vectorize=True)
        return J.reshape(z.numel(), z.numel()).detach()

    def _finite_difference(self, g, y_flat):
        n = y_flat.numel()
        with torch.no_grad():
            f0 = g(y_flat)
            eps = torch.finfo(y_flat.abs().dtype).eps ** 0.5
            J = torch.empty(n, n, dtype=f0.dtype, device=f0.device)
            for i in range(n):
                h = eps * max(float(y_flat[i].abs()), 1.0)
                y_pert = y_flat.clone()
                y_pert[i] = y_pert[i] + h
                J[:, i] = (g(y_pert) - f0) / h
        if not torch.isfinite(J).all():
            raise RuntimeError(
                '{}: could not form a finite Jacobian of `func` -- it returns non-finite values at or just '
                'next to the current state.'.format(self.solver_name))
        return J

    # ------------------------------------------------------------------ checking

    @staticmethod
    def _agrees_with_finite_difference(g, y_flat, J):
        """Whether `J @ w` matches `(g(y + h w) - g(y)) / h` along a fixed direction `w`.

        A forward difference also differs from `J @ w` by the curvature of `g`, which
        on a stiff problem easily exceeds any sensible tolerance (Robertson's
        `3e7 * y2**2` at `y2 = 0`). That part shrinks in proportion to `h` while
        the error of a Jacobian with a piece missing does not, so when the first
        probe disagrees the step is reduced until the mismatch either drops below
        tolerance or stops shrinking. Only the latter counts as a disagreement; a
        mismatch that is still shrinking when the probes run out gets the benefit
        of the doubt.
        """
        real_dtype = y_flat.abs().dtype
        eps = torch.finfo(real_dtype).eps
        # A fixed direction, positive so that the probe moves away from a lower
        # bound such as a zero concentration rather than across it. Its own
        # generator keeps the user's random state untouched.
        gen = torch.Generator().manual_seed(0)
        w = 0.5 + 0.5 * torch.rand(y_flat.numel(), generator=gen, dtype=real_dtype)
        w = w.to(device=y_flat.device, dtype=y_flat.dtype) * y_flat.abs().clamp(min=1.0)

        with torch.no_grad():
            Jw = torch.mv(J, w)
            f0 = g(y_flat)
            if not torch.isfinite(f0).all():
                return True

            h = eps ** 0.5
            scale = None
            last = None
            for _ in range(_PROBE_TRIES):
                f1 = g(y_flat + h * w)
                if torch.isfinite(f1).all():
                    D = (f1 - f0) / h
                    if scale is None:
                        # Mismatches are measured against the size of `J @ w` itself,
                        # with a floor so that components negligible next to the
                        # largest are not judged. The floor also looks at `D`: were it
                        # set by `J` alone, an all-zero `J` would give a zero floor and
                        # an infinite mismatch, which can never be seen to shrink.
                        size = torch.maximum(Jw.abs().max(), D.abs().max())
                        scale = _PROBE_RTOL * Jw.abs() + (1e-8 * size).clamp(min=torch.finfo(real_dtype).tiny)
                    rounding = 8 * eps * (f0.abs() + f1.abs()) / h
                    mismatch = float((((D - Jw).abs() - rounding).clamp(min=0) / scale).max())
                    if mismatch <= 1:
                        return True
                    if last is not None:
                        if mismatch > last * _PROBE_SHRINK / 4:
                            # Growing as h shrinks: rounding error we underestimated.
                            return True
                        if mismatch > last / 4:
                            # Not shrinking with h: not curvature, so J is wrong.
                            return False
                    last = mismatch
                h = h / _PROBE_SHRINK
        return True

    # ------------------------------------------------------------------ fallback

    def _autograd_failed_message(self, e):
        msg = ("{}: autograd could not differentiate `func` ({}: {}), so finite differences are used for "
               "the rest of this solve. Pass `options={{'jacobian': 'fd'}}` to skip the attempt."
               .format(self.solver_name, type(e).__name__, str(e).split('\n')[0][:120]))
        if torch.is_inference_mode_enabled():
            msg += (" Under `torch.inference_mode` this usually means `func` uses a tensor created inside "
                    "inference mode; creating it outside, or solving outside inference mode (e.g. "
                    "`pl.Trainer(inference_mode=False)`), restores the exact Jacobian.")
        return msg

    def _fall_back(self, message):
        self.use_fd = True
        warnings.warn(message, RuntimeWarning)
