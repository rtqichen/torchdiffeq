from functools import partial
import unittest
import warnings

import torch
import torchdiffeq

from problems import (construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS, FIXED_METHODS,
                      ADAMS_METHODS)


def rel_error(true, estimate):
    return ((true - estimate) / true).abs().max()


class TestSolverError(unittest.TestCase):
    def test_odeint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:
                        if dtype == torch.float32 and method == 'dopri8':
                            continue
                        kwargs = dict(rtol=1e-12, atol=1e-14) if method == 'dopri8' else dict()
                        problems = PROBLEMS if method in ADAPTIVE_METHODS else ('constant',)
                        for ode in problems:
                            if method in ['adaptive_heun', 'bosh3']:
                                eps = 4e-3
                            elif ode == 'linear':
                                eps = 2e-3
                            else:
                                eps = 1e-4

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

                        # Why is this case skipped over?
                        if dtype == torch.float32 and method == 'dopri8':
                            continue

                        with self.subTest(adjoint=adjoint, dtype=dtype, device=device, method=method):

                            x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype, requires_grad=True)
                            t = torch.tensor([0., 1.0], device=device)

                            simple_f = _JumpF()
                            odeint = partial(torchdiffeq.odeint_adjoint, adjoint_params=()) if adjoint else torchdiffeq.odeint
                            simple_xs = odeint(simple_f, x0, t, atol=1e-7, method=method)

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

                        # TODO: implement perturb for fixed step adams methods.
                        if method in ADAMS_METHODS:
                            continue
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


if __name__ == '__main__':
    unittest.main()
