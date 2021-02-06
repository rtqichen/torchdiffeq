from functools import partial
import unittest
import warnings

import torch
import torchdiffeq

from problems import (construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS, FIXED_METHODS)


def rel_error(true, estimate):
    return ((true - estimate) / true).abs().max()


class TestSolverError(unittest.TestCase):
    def test_odeint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:

                        kwargs = dict()
                        # Have to increase tolerance for dopri8.
                        if method == 'dopri8' and dtype == torch.float64:
                            kwargs = dict(rtol=1e-12, atol=1e-14)
                        if method == 'dopri8' and dtype == torch.float32:
                            kwargs = dict(rtol=1e-7, atol=1e-7)

                        problems = PROBLEMS if method in ADAPTIVE_METHODS else ('constant',)
                        for ode in problems:
                            if method in ['adaptive_heun', 'bosh3']:
                                eps = 4e-3
                            elif ode == 'linear':
                                eps = 2e-3
                            else:
                                eps = 3e-4

                            with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode, method=method):
                                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                         reverse=reverse)
                                y = torchdiffeq.odeint(f, y0, t_points, method=method, **kwargs)
                                self.assertLess(rel_error(sol, y), eps)

    def test_adjoint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for ode in PROBLEMS:
                        if ode == 'linear':
                            eps = 2e-3
                        else:
                            eps = 1e-4

                        with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode):
                            f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                     reverse=reverse)
                            y = torchdiffeq.odeint_adjoint(f, y0, t_points)
                            self.assertLess(rel_error(sol, y), eps)


class TestScipySolvers(unittest.TestCase):

    def test_odeint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for solver in ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']:
                        for ode in PROBLEMS:
                            eps = 1e-3

                            with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode, solver=solver):
                                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                         reverse=reverse)
                                y = torchdiffeq.odeint(f, y0, t_points, method='scipy_solver', options={"solver": solver})
                                self.assertTrue(sol.shape == y.shape)
                                self.assertLess(rel_error(sol, y), eps)


class TestNoIntegration(unittest.TestCase):
    def test_odeint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:
                        for ode in PROBLEMS:

                            with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode, method=method):
                                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                         reverse=reverse)
                                y = torchdiffeq.odeint(f, y0, t_points[0:1], method=method)
                                self.assertLess((sol[0] - y).abs().max(), 1e-12)


class _JumpF():
    def __init__(self):
        self.nfe = 0

    def __call__(self, t, x):
        self.nfe += 1
        if t < 0.5:
            return -0.5 * x
        else:
            return x ** 2


class TestDiscontinuities(unittest.TestCase):
    def test_odeint_jump_t(self):
        for adjoint in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in ADAPTIVE_METHODS:

                        with self.subTest(adjoint=adjoint, dtype=dtype, device=device, method=method):

                            x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype, requires_grad=True)
                            t = torch.tensor([0., 1.0], device=device)

                            simple_f = _JumpF()
                            odeint = partial(torchdiffeq.odeint_adjoint, adjoint_params=()) if adjoint else torchdiffeq.odeint
                            simple_xs = odeint(simple_f, x0, t, atol=1e-6, method=method)

                            better_f = _JumpF()
                            options = dict(jump_t=torch.tensor([0.5], device=device))

                            with warnings.catch_warnings():
                                better_xs = odeint(better_f, x0, t, rtol=1e-6, atol=1e-6, method=method,
                                                   options=options)

                            self.assertLess(better_f.nfe, simple_f.nfe)

                            if adjoint:
                                simple_f.nfe = 0
                                better_f.nfe = 0
                                with warnings.catch_warnings():
                                    simple_xs.sum().backward()
                                    better_xs.sum().backward()
                                self.assertLess(better_f.nfe, simple_f.nfe)

    # An option for fixed step solvers.
    def test_odeint_perturb(self):
        for adjoint in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in FIXED_METHODS:
                        for perturb in (True, False):
                            with self.subTest(adjoint=adjoint, dtype=dtype, device=device, method=method,
                                              perturb=perturb):
                                x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype, requires_grad=True)
                                t = torch.tensor([0., 1.0], device=device)
                                ts = []

                                def f(t, x):
                                    ts.append(t.item())
                                    return -x

                                options = dict(step_size=0.5, perturb=perturb)

                                with warnings.catch_warnings():
                                    odeint = partial(torchdiffeq.odeint_adjoint, adjoint_params=()) if adjoint else torchdiffeq.odeint
                                    xs = odeint(f, x0, t, method=method, options=options)

                                if perturb:
                                    self.assertNotIn(0., ts)
                                    self.assertNotIn(0.5, ts)
                                else:
                                    self.assertIn(0., ts)
                                    self.assertIn(0.5, ts)

                                if adjoint:
                                    ts.clear()
                                    with warnings.catch_warnings():
                                        xs.sum().backward()
                                    if perturb:
                                        self.assertNotIn(1., ts)
                                        self.assertNotIn(0.5, ts)
                                    else:
                                        self.assertIn(1., ts)
                                        self.assertIn(0.5, ts)


class TestGridConstructor(unittest.TestCase):
    def test_grid_constructor(self):
        def f(t, x):
            return x

        for adjoint in (False, True):
            with self.subTest(adjoint=adjoint):
                x0 = torch.tensor(1., requires_grad=True)
                t = torch.tensor([0., 1.])

                first = True

                def grid_constructor(f, y0, t):
                    nonlocal first
                    self.assertEqual(t.shape, (2,))
                    if first:
                        first = False
                        self.assertEqual(t[0], 0.)
                        self.assertEqual(t[1], 1.)
                        return torch.linspace(0, 1, 11)
                    else:
                        # adjoint pass
                        self.assertEqual(t[0], 1.)
                        self.assertEqual(t[1], 0.)
                        return torch.linspace(1, 0, 11)

                odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
                kwargs = {"adjoint_params": ()} if adjoint else {}
                xs = odeint(f, x0, t, method='euler', options=dict(grid_constructor=grid_constructor), **kwargs)
                x1 = xs[1]

                # 'true' wrt the use of the Euler scheme
                true_x1 = x0 * 1.1 ** 10
                self.assertLess((x1 - true_x1).abs().max(), 1e-6)
                if adjoint:
                    x1.backward()
                    # 'true' wrt the use of the Euler scheme
                    true_x0_grad = 1.1 ** 10
                    self.assertLess((x0.grad - true_x0_grad).abs().max(), 1e-6)


if __name__ == '__main__':
    unittest.main()
