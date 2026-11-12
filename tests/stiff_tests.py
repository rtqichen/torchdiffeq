"""Regression tests for the fixed-grid implicit (stiff) solvers.

These solvers were previously only exercised on `constant` and `exp`, both of which
have a zero Jacobian df/dy along the solution. That is the one regime in which the
nonlinear solve inside an implicit step is trivial, so the tests passed while the
solvers diverged to NaN on genuinely stiff problems. The tests here cover the cases
that matter for an implicit method: a large `dt * df/dy`, and a non-autonomous right
hand side.
"""
import math
import unittest
import warnings

import torch
import torchdiffeq

from problems import FIXED_IMPLICIT_METHODS, DEVICES
from torchdiffeq._impl.odeint import SOLVERS


# Order of accuracy each method advertises.
ORDERS = {'implicit_euler': 1, 'implicit_midpoint': 2, 'trapezoid': 2, 'radauIIA3': 3,
          'gl4': 4, 'radauIIA5': 5, 'gl6': 6, 'sdirk2': 2, 'trbdf2': 2}

# Methods that are L-stable, i.e. whose stability function vanishes at infinity and
# which therefore damp a stiff transient in a single step. The Gauss-Legendre and
# midpoint/trapezoid families are A-stable but not L-stable (|R(inf)| == 1), so they
# legitimately retain an undamped transient at very large `dt * lambda`.
L_STABLE = ('implicit_euler', 'radauIIA3', 'radauIIA5', 'sdirk2', 'trbdf2')


class TestTableaus(unittest.TestCase):

    def test_tableau_consistency(self):
        """Row sums of A must equal the abscissae c, and the weights b must sum to 1.

        This is the cheapest possible check on a Butcher tableau, and it catches
        transcription typos that only show up as an order reduction on a
        non-autonomous problem.
        """
        for name in FIXED_IMPLICIT_METHODS:
            tableau = SOLVERS[name].tableau
            with self.subTest(method=name):
                for i, (row, c_i) in enumerate(zip(tableau.beta, tableau.alpha)):
                    self.assertAlmostEqual(float(row.sum()), float(c_i), places=12,
                                           msg='row {} of A does not sum to c[{}]'.format(i, i))
                self.assertAlmostEqual(float(tableau.c_sol.sum()), 1.0, places=12,
                                       msg='weights b do not sum to 1')


class TestStiff(unittest.TestCase):

    def test_stiff_linear_decay(self):
        """y' = -k y with dt * k >> 1. An implicit solver must stay bounded here."""
        for device in DEVICES:
            for k in (100.0, 1e4, 1e6):
                for method in FIXED_IMPLICIT_METHODS:
                    with self.subTest(device=device, k=k, method=method):
                        y0 = torch.tensor([1.0], dtype=torch.float64, device=device)
                        t = torch.linspace(0, 1, 11, dtype=torch.float64, device=device)
                        with warnings.catch_warnings():
                            warnings.simplefilter('error', RuntimeWarning)
                            y = torchdiffeq.odeint(lambda t_, y_: -k * y_, y0, t,
                                                   method=method, options=dict(step_size=0.1))
                        self.assertTrue(torch.isfinite(y).all(), 'non-finite solution')
                        # The exact solution decays to zero; no A-stable method may grow.
                        self.assertLessEqual(y.abs().max().item(), 1.0 + 1e-9)
                        if method in L_STABLE:
                            self.assertLess(y[-1].abs().item(), 1e-2)

    def test_robertson(self):
        """The textbook stiff problem, at a step size dopri5 has no trouble with."""
        def rober(t, y):
            return torch.stack([-0.04 * y[0] + 1e4 * y[1] * y[2],
                                0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
                                3e7 * y[1] ** 2])

        for device in DEVICES:
            y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)
            t = torch.linspace(0, 0.4, 5, dtype=torch.float64, device=device)
            ref = torchdiffeq.odeint(rober, y0, t, method='dopri5', rtol=1e-10, atol=1e-12)
            for method in FIXED_IMPLICIT_METHODS:
                with self.subTest(device=device, method=method):
                    # At this step size the nonlinear solve is genuinely hard for the
                    # Gauss methods, which are not stiffly accurate; they are allowed
                    # to report non-convergence, but never to return garbage.
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        y = torchdiffeq.odeint(rober, y0, t, method=method,
                                               options=dict(step_size=0.02))
                    self.assertTrue(torch.isfinite(y).all(), 'non-finite solution')
                    self.assertLess((y - ref).abs().max().item(), 1e-2)
                    # Mass is conserved exactly by this system.
                    self.assertLess((y.sum(dim=-1) - 1.0).abs().max().item(), 1e-6)

                    # At a step size that is merely stiff rather than extreme, every
                    # method must converge silently and accurately.
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', RuntimeWarning)
                        y = torchdiffeq.odeint(rober, y0, t, method=method,
                                               options=dict(step_size=1e-3))
                    self.assertTrue(torch.isfinite(y).all(), 'non-finite solution')
                    self.assertLess((y - ref).abs().max().item(), 1e-4)

    def test_van_der_pol(self):
        """Stiff and nonlinear, so the Jacobian genuinely changes during the solve."""
        mu = 100.0

        def vdp(t, y):
            return torch.stack([y[1], mu * ((1 - y[0] ** 2) * y[1] - y[0])])

        for device in DEVICES:
            y0 = torch.tensor([2.0, 0.0], dtype=torch.float64, device=device)
            t = torch.linspace(0, 1, 5, dtype=torch.float64, device=device)
            ref = torchdiffeq.odeint(vdp, y0, t, method='dopri5', rtol=1e-11, atol=1e-13)
            for method in FIXED_IMPLICIT_METHODS:
                with self.subTest(device=device, method=method):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        y = torchdiffeq.odeint(vdp, y0, t, method=method,
                                               options=dict(step_size=1e-3))
                    self.assertTrue(torch.isfinite(y).all(), 'non-finite solution')
                    # A first order method needs a looser bound at this step size.
                    self.assertLess((y - ref).abs().max().item(),
                                    1.5e-1 if ORDERS[method] == 1 else 5e-2)


class TestConvergenceOrder(unittest.TestCase):

    def test_order_non_autonomous(self):
        """Each method must attain its advertised order on y' = cos(5t) - y.

        A non-autonomous problem is essential: an error in a tableau's abscissae `c`
        is invisible on an autonomous one, because there the stage times are unused.
        """
        def f(t, y):
            return torch.cos(5 * t) - y

        def exact(t):
            return ((5 * torch.sin(5 * t) + torch.cos(5 * t) + 25 * torch.exp(-t)) / 26).reshape(1)

        t = torch.tensor([0.0, 1.0], dtype=torch.float64)
        y0, ref = exact(t[0]), exact(t[1])

        for method, order in ORDERS.items():
            with self.subTest(method=method):
                errs = []
                for n in (8, 16, 32):
                    with warnings.catch_warnings():
                        warnings.simplefilter('error', RuntimeWarning)
                        y = torchdiffeq.odeint(f, y0, t, method=method,
                                               options=dict(step_size=1.0 / n))[-1]
                    errs.append((y - ref).abs().max().item())
                # Ignore the comparison if we have already hit round-off.
                if errs[-1] < 1e-12:
                    continue
                observed = math.log2(errs[0] / errs[-1]) / 2
                self.assertGreater(observed, order - 0.5,
                                   '{}: observed order {:.2f}, expected {}'.format(method, observed, order))


class TestImplicitGradients(unittest.TestCase):

    def test_gradients_match_finite_differences(self):
        """The implicit solve must not silently break the autograd path."""
        class F(torch.nn.Module):
            def __init__(self):
                super(F, self).__init__()
                self.a = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))

            def forward(self, t, y):
                return self.a * y + 0.5 * torch.sin(y)

        t = torch.tensor([0.0, 1.0], dtype=torch.float64)
        options = dict(step_size=0.05)

        for method in FIXED_IMPLICIT_METHODS:
            with self.subTest(method=method):
                func = F()
                y0 = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    out = torchdiffeq.odeint(func, y0, t, method=method, options=options)[-1].sum()
                    grad_y0, grad_a = torch.autograd.grad(out, [y0, func.a])

                    def value(dy0=0.0, da=0.0):
                        g = F()
                        with torch.no_grad():
                            g.a += da
                        return torchdiffeq.odeint(g, y0.detach() + dy0, t, method=method,
                                                  options=options)[-1].sum().item()

                    h = 1e-6
                    fd_y0 = torch.tensor(
                        [(value(dy0=torch.tensor([h, 0.0], dtype=torch.float64))
                          - value(dy0=torch.tensor([-h, 0.0], dtype=torch.float64))) / (2 * h),
                         (value(dy0=torch.tensor([0.0, h], dtype=torch.float64))
                          - value(dy0=torch.tensor([0.0, -h], dtype=torch.float64))) / (2 * h)],
                        dtype=torch.float64)
                    fd_a = (value(da=h) - value(da=-h)) / (2 * h)

                self.assertLess((grad_y0 - fd_y0).abs().max().item(), 1e-6)
                self.assertLess(abs(grad_a.item() - fd_a), 1e-6)


class TestDivergenceIsReported(unittest.TestCase):

    def test_divergence_raises_rather_than_returning_nan(self):
        """A diverged step must not be handed back to the caller as silent NaN."""
        def blowup(t, y):
            return -1e6 * y ** 3

        y0 = torch.tensor([1e3], dtype=torch.float64)
        t = torch.tensor([0.0, 10.0], dtype=torch.float64)
        for method in FIXED_IMPLICIT_METHODS:
            with self.subTest(method=method):
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    try:
                        y = torchdiffeq.odeint(blowup, y0, t, method=method,
                                               options=dict(step_size=5.0))
                    except RuntimeError:
                        continue  # diverged, and said so
                self.assertTrue(torch.isfinite(y).all(),
                                '{} returned non-finite values without raising'.format(method))


if __name__ == '__main__':
    unittest.main()
