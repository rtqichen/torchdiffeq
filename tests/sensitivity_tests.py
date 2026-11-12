"""Tests for `odeint_sensitivity`: forward sensitivities dy/dtheta through `radau`.

References, from strongest to weakest: the closed-form solution of a stiff linear
system (differentiated by autograd through `matrix_exp`), the reverse-mode
gradient of `radau` itself, forward-mode AD through `radau`, and central finite
differences of two plain solves.
"""
import unittest

import torch
import torch.autograd.forward_ad as fwAD
import torchdiffeq
from torchdiffeq import odeint, odeint_sensitivity

DT = torch.float64


def _rel(a, b):
    return float((a - b).abs().max() / b.abs().max())


class _Linear(torch.nn.Module):
    """y' = A(theta) y, stiff for theta[0] = 1000. Exact solution exp(A t) y0."""

    def __init__(self, theta=(1000.0, 1.0, 1.0)):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor(theta, dtype=DT))

    @staticmethod
    def A(theta):
        zero = torch.zeros((), dtype=DT)
        return torch.stack([torch.stack([-theta[0], theta[1]]), torch.stack([zero, -theta[2]])])

    def forward(self, t, y):
        return y @ self.A(self.theta).T          # works for y of shape (2,) and (..., 2)


def _exact(theta, y0, t):
    return torch.stack([torch.linalg.matrix_exp(_Linear.A(theta) * ti) @ y0 for ti in t])


def _exact_sensitivity(theta, y0_of_theta, t):
    return torch.autograd.functional.jacobian(lambda th: _exact(th, y0_of_theta(th), t), theta.detach())


class _Robertson(torch.nn.Module):
    """Robertson with log rate constants inside `func`."""

    def __init__(self):
        super().__init__()
        self.logk = torch.nn.Parameter(torch.log(torch.tensor([0.04, 3e7, 1e4], dtype=DT)))

    def forward(self, t, y):
        k1, k2, k3 = torch.exp(self.logk).unbind()
        y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2]
        return torch.stack((-k1 * y1 + k3 * y2 * y3,
                            k1 * y1 - k3 * y2 * y3 - k2 * y2 ** 2,
                            k2 * y2 ** 2), dim=-1)


_ROB_Y0 = torch.tensor([1.0, 0.0, 0.0], dtype=DT)
_ROB_T = torch.tensor([0.0, 1e-2, 1.0, 1e2], dtype=DT)
# A fixed first step keeps the step size free of any dependence on the parameters,
# which forward-mode AD through `odeint` would otherwise try (and fail) to follow.
_ROB_KW = dict(rtol=1e-8, atol=1e-8, options={'first_step': 1e-6})


class TestAgainstReferences(unittest.TestCase):

    def test_linear_exact(self):
        f = _Linear()
        y0 = torch.tensor([1.0, 1.0], dtype=DT)
        t = torch.linspace(0, 2, 6, dtype=DT)
        y, S = odeint_sensitivity(f, y0, t, [f.theta], rtol=1e-8, atol=1e-8)
        self.assertEqual(tuple(S.shape), (6, 2, 3))
        self.assertLess(_rel(S, _exact_sensitivity(f.theta, lambda th: y0, t)), 1e-6)

    def test_robertson_against_forward_ad_and_fd(self):
        f = _Robertson()
        y, S = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        for j in range(3):
            with self.subTest(direction=j):
                tangent = torch.zeros(3, dtype=DT)
                tangent[j] = 1.0

                g = _Robertson()
                with torch.no_grad(), fwAD.dual_level():
                    logk = g.logk.detach()
                    del g.logk                    # replace the Parameter with a plain (dual) tensor
                    g.logk = fwAD.make_dual(logk, tangent)
                    d = fwAD.unpack_dual(odeint(g, _ROB_Y0, _ROB_T, method='radau', **_ROB_KW)).tangent.clone()
                self.assertLess(_rel(S[..., j], d), 1e-6)

                def solve(eps):
                    h = _Robertson()
                    logk = h.logk.detach()
                    del h.logk
                    h.logk = logk + eps * tangent
                    with torch.no_grad():
                        return odeint(h, _ROB_Y0, _ROB_T, method='radau', **_ROB_KW)
                fd = (solve(1e-4) - solve(-1e-4)) / 2e-4
                self.assertLess(_rel(S[..., j], fd), 1e-6)

    def test_functional_matches_reverse_mode(self):
        f = _Robertson()
        y, S = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        yr = odeint(f, _ROB_Y0, _ROB_T, method='radau', **_ROB_KW)
        w = torch.linspace(0.5, 1.5, yr.numel(), dtype=DT).view_as(yr)
        grad, = torch.autograd.grad((w * yr).sum(), f.logk)
        self.assertLess(_rel(torch.einsum('tn,tnp->p', w, S), grad), 1e-6)

    def test_state_matches_odeint(self):
        """The state is solved exactly as by `odeint`: same steps, same values."""
        f = _Robertson()
        y, _ = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        with torch.no_grad():
            y_ref = odeint(f, _ROB_Y0, _ROB_T, method='radau', **_ROB_KW)
        self.assertTrue(torch.equal(y, y_ref))


class TestInitialSensitivityAndChaining(unittest.TestCase):

    @staticmethod
    def _y0(theta):
        # An initial state that depends on the parameters.
        return torch.stack([theta[1] / 2, theta[2] ** 2])

    def test_nonzero_S0(self):
        f = _Linear()
        t = torch.linspace(0, 2, 5, dtype=DT)
        theta = f.theta.detach()
        S0 = torch.autograd.functional.jacobian(self._y0, theta)
        y, S = odeint_sensitivity(f, self._y0(theta), t, [f.theta], S0=S0, rtol=1e-8, atol=1e-8)
        self.assertTrue(torch.equal(S[0], S0))
        self.assertLess(_rel(S, _exact_sensitivity(f.theta, self._y0, t)), 1e-6)

    def test_two_chained_calls(self):
        f = _Linear()
        theta = f.theta.detach()
        S0 = torch.autograd.functional.jacobian(self._y0, theta)
        t1 = torch.linspace(0, 1, 4, dtype=DT)
        t2 = torch.linspace(1, 2, 4, dtype=DT)
        ya, Sa = odeint_sensitivity(f, self._y0(theta), t1, [f.theta], S0=S0, rtol=1e-8, atol=1e-8)
        yb, Sb = odeint_sensitivity(f, ya[-1], t2, [f.theta], S0=Sa[-1], rtol=1e-8, atol=1e-8)
        t = torch.cat([t1, t2[1:]])
        S = torch.cat([Sa, Sb[1:]])
        self.assertLess(_rel(S, _exact_sensitivity(f.theta, self._y0, t)), 1e-6)

    def test_chained_robertson_matches_one_call(self):
        f = _Robertson()
        _, S_one = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        ya, Sa = odeint_sensitivity(f, _ROB_Y0, _ROB_T[:3], [f.logk], **_ROB_KW)
        _, Sb = odeint_sensitivity(f, ya[-1], _ROB_T[2:], [f.logk], S0=Sa[-1], **_ROB_KW)
        self.assertLess(_rel(Sb[-1], S_one[-1]), 1e-6)


class TestShapesAndModes(unittest.TestCase):

    def test_decreasing_time(self):
        f = _Linear(theta=(2.0, 1.0, 0.5))
        y0 = torch.tensor([1.0, 1.0], dtype=DT)
        t = torch.linspace(1, 0, 5, dtype=DT)
        y, S = odeint_sensitivity(f, y0, t, [f.theta], rtol=1e-9, atol=1e-9)
        t_from_0 = t - t[0]
        self.assertLess(_rel(y, _exact(f.theta.detach(), y0, t_from_0)), 1e-6)
        self.assertLess(_rel(S, _exact_sensitivity(f.theta, lambda th: y0, t_from_0)), 1e-6)

    def test_batched_state(self):
        f = _Linear()
        y0 = torch.tensor([[1.0, 1.0], [0.5, -2.0]], dtype=DT)
        t = torch.linspace(0, 1, 4, dtype=DT)
        y, S = odeint_sensitivity(f, y0, t, [f.theta], rtol=1e-8, atol=1e-8)
        self.assertEqual(tuple(S.shape), (4, 2, 2, 3))
        for b in range(2):
            ref = _exact_sensitivity(f.theta, lambda th: y0[b], t)
            self.assertLess(_rel(S[:, b], ref), 1e-6)

    def test_plain_tensor_parameters(self):
        """Any tensor that requires grad works, not only `nn.Parameter`s; several at once."""
        theta = torch.tensor([1000.0, 1.0], dtype=DT, requires_grad=True)
        third = torch.tensor(1.0, dtype=DT, requires_grad=True)

        def func(t, y):
            return y @ _Linear.A(torch.cat([theta, third.reshape(1)])).T

        y0 = torch.tensor([1.0, 1.0], dtype=DT)
        t = torch.linspace(0, 2, 5, dtype=DT)
        y, S = odeint_sensitivity(func, y0, t, [theta, third], rtol=1e-8, atol=1e-8)
        ref = _exact_sensitivity(torch.tensor([1000.0, 1.0, 1.0], dtype=DT), lambda th: y0, t)
        self.assertLess(_rel(S, ref), 1e-6)

    def test_unused_parameter_has_zero_sensitivity(self):
        f = _Linear()
        unused = torch.ones(2, dtype=DT, requires_grad=True)
        y, S = odeint_sensitivity(f, torch.tensor([1.0, 1.0], dtype=DT), torch.linspace(0, 1, 3, dtype=DT),
                                  [f.theta, unused], rtol=1e-8, atol=1e-8)
        self.assertEqual(S.shape[-1], 5)
        self.assertTrue(torch.equal(S[..., 3:], torch.zeros_like(S[..., 3:])))

    def test_inference_mode(self):
        f = _Robertson()
        y, S = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        with torch.inference_mode():
            yi, Si = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        self.assertTrue(torch.equal(y, yi))
        self.assertTrue(torch.equal(S, Si))

    def test_parameters_left_untouched(self):
        """The finite difference check perturbs the parameters; they must come back unchanged."""
        f = _Robertson()
        before = f.logk.detach().clone()
        version = f.logk._version
        odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        self.assertTrue(torch.equal(f.logk.detach(), before))
        self.assertEqual(f.logk._version, version)

    def test_no_graph_is_built(self):
        f = _Robertson()
        y, S = odeint_sensitivity(f, _ROB_Y0, _ROB_T, [f.logk], **_ROB_KW)
        self.assertFalse(y.requires_grad)
        self.assertFalse(S.requires_grad)


class TestErrors(unittest.TestCase):
    """No silent fallbacks: what cannot be computed correctly raises."""

    def setUp(self):
        self.f = _Linear()
        self.y0 = torch.tensor([1.0, 1.0], dtype=DT)
        self.t = torch.linspace(0, 1, 3, dtype=DT)

    def _call(self, **kwargs):
        args = dict(rtol=1e-6, atol=1e-8)
        args.update(kwargs)
        params = args.pop('params', [self.f.theta])
        func = args.pop('func', self.f)
        return odeint_sensitivity(func, self.y0, self.t, params, **args)

    def test_parameter_without_grad(self):
        with self.assertRaisesRegex(ValueError, 'does not require grad'):
            self._call(params=[torch.ones(3, dtype=DT)])

    def test_other_methods(self):
        with self.assertRaisesRegex(ValueError, "method='radau' only"):
            self._call(method='lsoda')

    def test_iterative_grad_jacobian(self):
        with self.assertRaisesRegex(ValueError, 'exact stage matrix'):
            self._call(options={'grad_jacobian': 'iterative'})

    def test_fd_jacobian(self):
        with self.assertRaisesRegex(ValueError, "jacobian='fd'"):
            self._call(options={'jacobian': 'fd'})

    def test_tuple_state(self):
        with self.assertRaises(NotImplementedError):
            odeint_sensitivity(lambda t, y: y, (self.y0, self.y0), self.t, [self.f.theta])

    def test_bad_S0_shape(self):
        with self.assertRaisesRegex(ValueError, 'S0 must have shape'):
            self._call(S0=torch.zeros(2, 2, dtype=DT))

    def test_parameter_used_outside_autograd(self):
        """A parameter entering `func` through `.item()` gets no autograd derivative.
        The default check must catch that rather than return a zero sensitivity."""
        theta = torch.tensor([1000.0, 1.0, 1.0], dtype=DT, requires_grad=True)

        def func(t, y):
            A = _Linear.A(torch.stack([theta[0].detach() * 0 + theta[0].item(), theta[1], theta[2]]))
            return y @ A.T

        with self.assertRaisesRegex(RuntimeError, 'disagree with finite differences'):
            self._call(func=func, params=[theta])
        # With the check switched off it runs (and the first column is wrongly zero).
        _, S = self._call(func=func, params=[theta], options={'jacobian': 'autograd'})
        self.assertTrue(torch.equal(S[..., 0], torch.zeros_like(S[..., 0])))


class TestExport(unittest.TestCase):
    def test_exported(self):
        self.assertIs(torchdiffeq.odeint_sensitivity, odeint_sensitivity)


if __name__ == '__main__':
    unittest.main()
