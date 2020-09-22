import unittest
import torch
import torchdiffeq

from problems import construct_problem, \
                     PROBLEMS, DTYPES, DEVICES, \
                     METHODS, ADAPTIVE_METHODS, \
                     FIXED_SYMPLECTIC_METHODS


def rel_error(true, estimate):
    return ((true - estimate) / true).abs().max()


class TestSolverError(unittest.TestCase):
    def test_odeint(self):
        torch.manual_seed(42)
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:
                        if dtype == torch.float32 and method == 'dopri8':
                            continue
                        kwargs = dict(rtol=1e-12, atol=1e-14) if method == 'dopri8' else dict()
                        if method in ADAPTIVE_METHODS:
                            problems = PROBLEMS
                        elif method in FIXED_SYMPLECTIC_METHODS:
                            problems = ('harmonic','constant_symplectic',)
                            kwargs = dict(options={'step_size':0.1})
                        else:
                            problems = ('constant',)
                        for ode in problems:
                            if method == 'adaptive_heun':
                                eps = 4e-3
                            elif method == 'bosh3':
                                eps = 3e-3
                                if ode == 'constant_symplectic':
                                    continue
                            elif ode in ['linear','harmonic']:
                                eps = 2e-3
                            else:
                                eps = 1e-4

                            with self.subTest(reverse=reverse, dtype=dtype, 
                                            device=device, ode=ode, method=method):
                                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                         reverse=reverse)
                                y = torchdiffeq.odeint(f, y0, t_points, method=method, **kwargs)
                                self.assertLess(rel_error(sol, y), eps)

    def test_adjoint(self):
        torch.manual_seed(42)
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for ode in PROBLEMS:
                        if ode in ['linear','harmonic']:
                            eps = 2e-3
                        else:
                            eps = 1e-4
                        with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode):
                            f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                     reverse=reverse)
                            if ode in ['harmonic','constant_symplectic']:
                                y = torchdiffeq.odeint_adjoint(f, y0, t_points,
                                                               method='yoshida4th',
                                                               options={'step_size':0.1})
                            else:
                                y = torchdiffeq.odeint_adjoint(f, y0, t_points)
                            self.assertLess(rel_error(sol, y), eps)


class TestNoIntegration(unittest.TestCase):
    def test_odeint(self):
        torch.manual_seed(42)
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:
                        if method in FIXED_SYMPLECTIC_METHODS:
                            problems = ('harmonic',)
                        else:
                            problems = PROBLEMS
                        for ode in problems:
                            with self.subTest(reverse=reverse, dtype=dtype, 
                                        device=device, ode=ode, method=method):
                                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                         reverse=reverse)
                                y = torchdiffeq.odeint(f, y0, t_points[0:1], method=method)
                                self.assertLess((sol[0] - y).abs().max(), 1e-12)


if __name__ == '__main__':
    unittest.main()
