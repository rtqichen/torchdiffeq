"""Tests for the adaptive-step implicit solvers: `radau` and `lsoda`.

These are the two solvers that choose their own step sizes *and* handle stiffness,
so what has to be checked is different from the fixed-grid implicit family: the
error actually delivered must track the requested tolerance, the step size must
adapt rather than being handed in, and the whole thing must stay differentiable.

`lsoda` additionally has to pick the right method for the problem, which the
switching tests below check directly rather than inferring it from the answer.
"""
import math
import unittest
import warnings

import torch
import torchdiffeq

from problems import ADAPTIVE_IMPLICIT_METHODS, DEVICES


def _rober(t, y):
    return torch.stack([-0.04 * y[0] + 1e4 * y[1] * y[2],
                        0.04 * y[0] - 1e4 * y[1] * y[2] - 3e7 * y[1] ** 2,
                        3e7 * y[1] ** 2])


def _nonautonomous(t, y):
    return torch.cos(5 * t) - y


def _nonautonomous_exact(t):
    return ((5 * torch.sin(5 * t) + torch.cos(5 * t) + 25 * torch.exp(-t)) / 26).reshape(1)


class TestTolerance(unittest.TestCase):

    def test_error_tracks_rtol(self):
        """Tightening rtol must actually tighten the delivered error."""
        t = torch.tensor([0., 3.], dtype=torch.float64)
        y0 = _nonautonomous_exact(t[0])
        ref = _nonautonomous_exact(t[1])
        for method in ADAPTIVE_IMPLICIT_METHODS:
            errs = []
            for rtol in (1e-4, 1e-6, 1e-8, 1e-10):
                with self.subTest(method=method, rtol=rtol):
                    y = torchdiffeq.odeint(_nonautonomous, y0, t, method=method,
                                           rtol=rtol, atol=rtol * 1e-3)
                    err = (y[-1] - ref).abs().max().item()
                    errs.append(err)
                    # Allow a decade of slack: the estimator controls the local
                    # error, not the global one.
                    self.assertLess(err, rtol * 10)
            # And the trend must be monotone-ish, not accidentally flat.
            self.assertLess(errs[-1], errs[0] * 1e-3,
                            '{}: error barely improved with rtol'.format(method))

    def test_no_step_size_needed(self):
        """These solvers must work with only two time points and no options."""
        for method in ADAPTIVE_IMPLICIT_METHODS:
            with self.subTest(method=method):
                y = torchdiffeq.odeint(lambda t, x: -x, torch.tensor([1.0], dtype=torch.float64),
                                       torch.tensor([0., 5.], dtype=torch.float64),
                                       method=method, rtol=1e-9, atol=1e-11)
                self.assertLess(abs(y[-1].item() - math.exp(-5.0)), 1e-8)


class TestStiff(unittest.TestCase):

    def test_robertson(self):
        for device in DEVICES:
            y0 = torch.tensor([1., 0., 0.], dtype=torch.float64, device=device)
            t = torch.linspace(0, 0.4, 5, dtype=torch.float64, device=device)
            ref = torchdiffeq.odeint(_rober, y0, t, method='dopri5', rtol=1e-13, atol=1e-15)
            for method in ADAPTIVE_IMPLICIT_METHODS:
                with self.subTest(device=device, method=method):
                    y = torchdiffeq.odeint(_rober, y0, t, method=method, rtol=1e-8, atol=1e-11)
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertLess((y - ref).abs().max().item(), 1e-6)
                    # This system conserves mass exactly.
                    self.assertLess((y.sum(dim=-1) - 1.0).abs().max().item(), 1e-9)

    def test_extreme_stiffness_stays_bounded(self):
        """y' = -k y with k up to 1e8. Adaptive + L-stable should shrug this off."""
        t = torch.linspace(0, 1, 6, dtype=torch.float64)
        for method in ADAPTIVE_IMPLICIT_METHODS:
            for k in (1e2, 1e4, 1e6, 1e8):
                with self.subTest(method=method, k=k):
                    y = torchdiffeq.odeint(lambda t_, y_: -k * y_,
                                           torch.tensor([1.0], dtype=torch.float64), t,
                                           method=method, rtol=1e-8, atol=1e-10)
                    self.assertTrue(torch.isfinite(y).all())
                    self.assertLessEqual(y.abs().max().item(), 1.0 + 1e-9)
                    self.assertLess(y[-1].abs().item(), 1e-6)

    def test_van_der_pol(self):
        mu = 1000.

        def vdp(t, y):
            return torch.stack([y[1], mu * ((1 - y[0] ** 2) * y[1] - y[0])])

        y0 = torch.tensor([2., 0.], dtype=torch.float64)
        t = torch.linspace(0, 1, 3, dtype=torch.float64)
        ref = torchdiffeq.odeint(vdp, y0, t, method='dopri5', rtol=1e-11, atol=1e-13)
        for method in ADAPTIVE_IMPLICIT_METHODS:
            with self.subTest(method=method):
                y = torchdiffeq.odeint(vdp, y0, t, method=method, rtol=1e-8, atol=1e-10)
                self.assertTrue(torch.isfinite(y).all())
                self.assertLess((y - ref).abs().max().item(), 1e-4)


class TestLSODASwitching(unittest.TestCase):
    """`lsoda` must choose Adams or BDF for the problem in front of it."""

    @staticmethod
    def _trace(f, y0, t, **kwargs):
        from torchdiffeq._impl import lsoda as lsoda_mod
        methods = []
        original = lsoda_mod.LSODA._step

        def traced(self):
            out = original(self)
            methods.append(self.meth)
            return out

        lsoda_mod.LSODA._step = traced
        try:
            y = torchdiffeq.odeint(f, y0, t, method='lsoda', rtol=1e-8, atol=1e-10, **kwargs)
        finally:
            lsoda_mod.LSODA._step = original
        return y, methods

    def test_nonstiff_stays_in_adams(self):
        """Nothing stiff here: paying for a Jacobian would be pure waste."""
        for name, f, y0, t in [
            ('decay', lambda t, y: -y, torch.tensor([1.0], dtype=torch.float64),
             torch.linspace(0, 10, 3, dtype=torch.float64)),
            ('oscillator', lambda t, y: torch.stack([y[1], -y[0]]),
             torch.tensor([1.0, 0.0], dtype=torch.float64),
             torch.linspace(0, 20, 3, dtype=torch.float64)),
        ]:
            with self.subTest(problem=name):
                _, methods = self._trace(f, y0, t)
                bdf_fraction = methods.count('bdf') / len(methods)
                self.assertLess(bdf_fraction, 0.05,
                                '{}: spent {:.0%} of steps in BDF on a non-stiff problem'
                                .format(name, bdf_fraction))

    def test_stiff_uses_bdf(self):
        """Robertson is stiff throughout; Adams alone cannot carry it."""
        y, methods = self._trace(_rober, torch.tensor([1., 0., 0.], dtype=torch.float64),
                                 torch.linspace(0, 0.4, 3, dtype=torch.float64))
        self.assertGreater(methods.count('bdf'), 0, 'never switched to BDF on Robertson')
        self.assertTrue(torch.isfinite(y).all())

    def test_switches_when_the_problem_changes(self):
        """Non-stiff up to t=5, violently stiff after: it has to notice."""
        def mixed(t, y):
            return -(1.0 if t < 5.0 else 1e5) * y

        _, methods = self._trace(mixed, torch.tensor([1.0], dtype=torch.float64),
                                 torch.linspace(0, 10, 3, dtype=torch.float64))
        self.assertGreater(methods.count('adams'), 0)
        self.assertGreater(methods.count('bdf'), 0)
        switches = sum(1 for a, b in zip(methods, methods[1:]) if a != b)
        self.assertGreaterEqual(switches, 1)
        # And it must not oscillate between the two.
        self.assertLess(switches, 20, 'thrashing between methods ({} switches)'.format(switches))


class TestGradients(unittest.TestCase):

    def test_gradients_match_finite_differences(self):
        """Both direct backpropagation and the adjoint must be right."""
        class F(torch.nn.Module):
            def __init__(self):
                super(F, self).__init__()
                self.a = torch.nn.Parameter(torch.tensor(-2.0, dtype=torch.float64))
                self.b = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64))

            def forward(self, t, y):
                return self.a * y + self.b * torch.sin(y)

        t = torch.tensor([0., 1.], dtype=torch.float64)
        kwargs = dict(rtol=1e-10, atol=1e-12)

        for method in ADAPTIVE_IMPLICIT_METHODS:
            for label, integrate in (('direct', torchdiffeq.odeint),
                                     ('adjoint', torchdiffeq.odeint_adjoint)):
                with self.subTest(method=method, mode=label):
                    func = F()
                    y0 = torch.tensor([1.0, -0.5], dtype=torch.float64, requires_grad=True)
                    out = integrate(func, y0, t, method=method, **kwargs)[-1].sum()
                    grads = torch.autograd.grad(out, [y0, func.a, func.b])

                    def value(dy0=0.0, da=0.0, db=0.0):
                        g = F()
                        with torch.no_grad():
                            g.a += da
                            g.b += db
                            return torchdiffeq.odeint(g, y0.detach() + dy0, t,
                                                      method=method, **kwargs)[-1].sum().item()

                    h = 1e-6
                    e0 = torch.tensor([h, 0.], dtype=torch.float64)
                    e1 = torch.tensor([0., h], dtype=torch.float64)
                    fd_y0 = torch.tensor([(value(dy0=e0) - value(dy0=-e0)) / (2 * h),
                                          (value(dy0=e1) - value(dy0=-e1)) / (2 * h)],
                                         dtype=torch.float64)
                    fd_a = (value(da=h) - value(da=-h)) / (2 * h)
                    fd_b = (value(db=h) - value(db=-h)) / (2 * h)

                    self.assertLess((grads[0] - fd_y0).abs().max().item(), 1e-6)
                    self.assertLess(abs(grads[1].item() - fd_a), 1e-6)
                    self.assertLess(abs(grads[2].item() - fd_b), 1e-6)


class TestShapesAndFeatures(unittest.TestCase):

    def test_state_shapes(self):
        for method in ADAPTIVE_IMPLICIT_METHODS:
            for shape in [(3,), (2, 4), (5, 2, 3)]:
                with self.subTest(method=method, shape=shape):
                    n = int(torch.tensor(shape).prod())
                    A = -torch.eye(n, dtype=torch.float64) * 3

                    def f(t, y, _A=A, _s=shape):
                        return (_A @ y.reshape(-1)).reshape(_s)

                    y = torchdiffeq.odeint(f, torch.ones(shape, dtype=torch.float64),
                                           torch.tensor([0., 1.], dtype=torch.float64),
                                           method=method, rtol=1e-9, atol=1e-11)
                    self.assertEqual(y.shape, (2,) + shape)
                    expected = torch.full(shape, math.exp(-3.0), dtype=torch.float64)
                    self.assertLess((y[-1] - expected).abs().max().item(), 1e-7)

    def test_tuple_state(self):
        for method in ADAPTIVE_IMPLICIT_METHODS:
            with self.subTest(method=method):
                y = torchdiffeq.odeint(lambda t, y: (-y[0], -2 * y[1]),
                                       (torch.tensor([1.0], dtype=torch.float64),
                                        torch.tensor([1.0, 2.0], dtype=torch.float64)),
                                       torch.tensor([0., 1.], dtype=torch.float64),
                                       method=method, rtol=1e-9, atol=1e-11)
                self.assertLess(abs(y[0][-1].item() - math.exp(-1.0)), 1e-7)
                self.assertLess((y[1][-1] - torch.tensor([math.exp(-2.0), 2 * math.exp(-2.0)],
                                                         dtype=torch.float64)).abs().max().item(), 1e-7)

    def test_event_handling(self):
        for method in ADAPTIVE_IMPLICIT_METHODS:
            with self.subTest(method=method):
                event_t, _ = torchdiffeq.odeint_event(
                    lambda t, y: -y, torch.tensor([1.0], dtype=torch.float64),
                    torch.tensor(0., dtype=torch.float64),
                    event_fn=lambda t, y: y[0] - 0.5,
                    method=method, rtol=1e-10, atol=1e-12)
                self.assertLess(abs(event_t.item() - math.log(2.0)), 1e-7)

    def test_reverse_time(self):
        for method in ADAPTIVE_IMPLICIT_METHODS:
            with self.subTest(method=method):
                t = torch.tensor([1., 0.], dtype=torch.float64)
                y = torchdiffeq.odeint(lambda t_, y_: -y_,
                                       torch.tensor([math.exp(-1.0)], dtype=torch.float64), t,
                                       method=method, rtol=1e-10, atol=1e-12)
                self.assertLess(abs(y[-1].item() - 1.0), 1e-7)


if __name__ == '__main__':
    unittest.main()
