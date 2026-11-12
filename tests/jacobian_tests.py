"""Tests for the Jacobian the implicit solvers hand to their Newton iterations.

Every failure guarded against here used to be silent: autograd returned a finite
Jacobian with part (or all) of `df/dy` missing, the solvers accepted it, and the
only symptom was a solve that crawled, failed or drifted. The answers mostly still
came out right where the solve finished, so most of these tests also assert on
the number of function evaluations -- an accuracy-only test would pass with the
bug present.
"""
import math
import unittest
import warnings

import torch
import torchdiffeq

from problems import ADAPTIVE_IMPLICIT_METHODS

# One of each implicit family: adaptive Radau, LSODA, fixed-grid FIRK and DIRK.
METHODS = ADAPTIVE_IMPLICIT_METHODS + ('radauIIA5', 'sdirk2')

_A = torch.tensor([[-1000.0, 1.0], [0.0, -1.0]], dtype=torch.float64)


def _linear(t, y):
    return _A @ y + 0.001 * y


def _half_detached(t, y):
    # The stiff part computed outside autograd, as a numpy rate law would be: the
    # autograd Jacobian is just 0.001 * I.
    return (_A @ y).detach() + 0.001 * y


def _rober(t, y):
    return torch.stack([-0.04 * y[0] + 1e4 * y[1] * y[2],
                        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
                        3e7 * y[1] ** 2])


class _Counted:
    def __init__(self, f):
        self.f = f
        self.n = 0

    def __call__(self, t, y):
        self.n += 1
        return self.f(t, y)


def _solve(func, y0, t, method, **kwargs):
    """Returns `(solution, number of evaluations, Jacobian warnings)`."""
    counted = _Counted(func)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        y = torchdiffeq.odeint(counted, y0, t, method=method, **kwargs)
    jac_warnings = [str(x.message) for x in w
                    if 'finite differences' in str(x.message) or 'autograd' in str(x.message)]
    return y, counted.n, jac_warnings


class TestDetachedRHS(unittest.TestCase):

    def test_partially_detached_rhs_is_detected(self):
        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
        t = torch.linspace(0, 10, 5, dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                ref, n_ref, w_ref = _solve(_linear, y0, t, method, rtol=1e-6, atol=1e-6)
                y, n, w = _solve(_half_detached, y0, t, method, rtol=1e-6, atol=1e-6)
                self.assertEqual(w_ref, [])
                self.assertEqual(len(w), 1)
                self.assertIn('disagrees with finite differences', w[0])
                self.assertLess((y - ref).abs().max().item(), 1e-6)
                # Finite differences cost n + 1 = 3 evaluations a Jacobian where
                # autograd costs none, so some overhead is expected -- not the
                # 100x or more of a solve running on the wrong Jacobian.
                self.assertLess(n, 3 * n_ref + 20)

    def test_fully_detached_rhs_is_detected(self):
        """Autograd returns zeros, rather than raising, when nothing is recorded."""
        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
        t = torch.linspace(0, 10, 5, dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                ref, n_ref, _ = _solve(_linear, y0, t, method, rtol=1e-6, atol=1e-6)
                y, n, w = _solve(lambda t_, y_: _linear(t_, y_).detach(), y0, t, method, rtol=1e-6, atol=1e-6)
                self.assertEqual(len(w), 1)
                self.assertLess((y - ref).abs().max().item(), 1e-6)
                self.assertLess(n, 3 * n_ref + 20)

    def test_fd_option_skips_autograd(self):
        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
        t = torch.linspace(0, 10, 5, dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                ref, _, _ = _solve(_linear, y0, t, method, rtol=1e-6, atol=1e-6)
                y, _, w = _solve(_half_detached, y0, t, method, rtol=1e-6, atol=1e-6,
                                 options=dict(jacobian='fd'))
                self.assertEqual(w, [])
                self.assertLess((y - ref).abs().max().item(), 1e-6)

    def test_invalid_option(self):
        y0 = torch.tensor([1.0], dtype=torch.float64)
        t = torch.tensor([0.0, 1.0], dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    torchdiffeq.odeint(_linear, y0, t, method=method, options=dict(jacobian='exact'))


class TestNoFalseAlarm(unittest.TestCase):

    def test_robertson(self):
        """A correct Jacobian must pass the check even where the curvature is huge.

        At the initial state `y2 = y3 = 0`, so `3e7 * y2**2` contributes nothing to
        the Jacobian but dominates a forward difference of any practical step.
        """
        y0 = torch.tensor([1., 0., 0.], dtype=torch.float64)
        t = torch.tensor([0., 1e2], dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                options = dict(step_size=0.1) if method in ('radauIIA5', 'sdirk2') else {}
                _, n_auto, w = _solve(_rober, y0, t, method, rtol=1e-6, atol=1e-10, options=options)
                self.assertEqual(w, [])
                _, n_unchecked, _ = _solve(_rober, y0, t, method, rtol=1e-6, atol=1e-10,
                                           options=dict(options, jacobian='autograd'))
                # The check costs a handful of evaluations per solve, no more.
                self.assertLess(n_auto, n_unchecked + 40)


class TestInferenceMode(unittest.TestCase):

    def test_matches_no_grad(self):
        """`inference_mode` (Lightning's test/validate/predict) used to give J = 0."""
        y0 = torch.tensor([1., 0., 0.], dtype=torch.float64)
        t = torch.tensor([0., 1e2], dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                options = dict(step_size=0.1) if method in ('radauIIA5', 'sdirk2') else {}
                with torch.no_grad():
                    ref, n_ref, _ = _solve(_rober, y0, t, method, rtol=1e-6, atol=1e-10, options=options)
                with torch.inference_mode():
                    y, n, w = _solve(_rober, y0, t, method, rtol=1e-6, atol=1e-10, options=options)
                self.assertEqual(w, [])
                self.assertEqual(n, n_ref)
                self.assertTrue(torch.equal(y, ref))

    def test_inference_tensor_in_func_falls_back(self):
        """A constant created under inference mode cannot enter an autograd graph."""
        y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
        t = torch.linspace(0, 10, 5, dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                ref, n_ref, _ = _solve(_linear, y0, t, method, rtol=1e-6, atol=1e-6)
                with torch.inference_mode():
                    A = _A.clone()
                    y, n, w = _solve(lambda t_, y_: A @ y_ + 0.001 * y_, y0, t, method, rtol=1e-6, atol=1e-6)
                self.assertEqual(len(w), 1)
                self.assertIn('inference mode', w[0])
                self.assertLess((y - ref).abs().max().item(), 1e-6)
                self.assertLess(n, 3 * n_ref + 20)


class TestAdjoint(unittest.TestCase):

    def test_adjoint_gradients(self):
        """The adjoint dynamics are opaque to autograd; they used to get J = 0.

        With k = 1000 that made `radau` and `lsoda` fail with a step size underflow
        and the fixed-grid solvers return gradients of order 1e100.
        """
        class F(torch.nn.Module):
            def __init__(self):
                super(F, self).__init__()
                self.k = torch.nn.Parameter(torch.tensor([1000., 1.], dtype=torch.float64))

            def forward(self, t, y):
                return -self.k * y

        t = torch.tensor([0., 1.], dtype=torch.float64)
        expected = torch.tensor([-math.exp(-1000.), -math.exp(-1.)], dtype=torch.float64)
        for method in METHODS:
            with self.subTest(method=method):
                options = dict(step_size=0.01) if method in ('radauIIA5', 'sdirk2') else {}
                func = F()
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    y = torchdiffeq.odeint_adjoint(func, torch.ones(2, dtype=torch.float64), t,
                                                   method=method, rtol=1e-8, atol=1e-10, options=options)
                    y[-1].sum().backward()
                self.assertEqual([x for x in w if 'finite differences' in str(x.message)], [])
                # Fixed grid at h = 0.01 is good to a few 1e-6; the adaptive ones
                # follow the tolerances.
                self.assertLess((func.k.grad - expected).abs().max().item(), 1e-5)


if __name__ == '__main__':
    unittest.main()
