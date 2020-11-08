import unittest
import torch
import torchdiffeq

from problems import construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS


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
                            if method == 'adaptive_heun':
                                eps = 4e-3
                            elif method == 'bosh3':
                                eps = 3e-3
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


class TestGridPoints(unittest.TestCase):
    def test_odeint_grid_points(self):

        for dtype in DTYPES:
            for device in DEVICES:
                for method in ADAPTIVE_METHODS:
                    if dtype == torch.float32 and method == 'dopri8':
                        continue
                    if method == 'scipy_solver':
                        continue
                    with self.subTest(dtype=dtype, device=device, method=method):

                        nfe = [0]

                        def f(t, x):
                            nfe[0] += 1
                            if t < 0.5:
                                return -0.5 * x
                            else:
                                return 1.0 * x**2

                        x0 = torch.tensor([1.0, 2.0]).to(device, dtype)
                        kwargs = dict(rtol=1e-12, atol=1e-14) if method == 'dopri8' else dict()

                        torchdiffeq.odeint(
                            f, x0, torch.tensor([0., 1.0]).to(device), method=method,
                            options={"grid_points": torch.tensor([0.5]).to(device), "eps": 1e-6},
                            **kwargs)
                        fixed_eps_nfe = nfe[0]
                        nfe[0] = 0

                        torchdiffeq.odeint(
                            f, x0, torch.tensor([0., 1.0]).to(device), method=method,
                            options={"grid_points": torch.tensor([0.5]).to(device), "eps": None},
                            **kwargs)
                        nextafter_eps_nfe = nfe[0]
                        nfe[0] = 0

                        torchdiffeq.odeint(
                            f, x0, torch.tensor([0., 1.0]).to(device), method=method,
                            options={"grid_points": torch.tensor([0.5]).to(device), "eps": 0.0},
                            **kwargs)
                        zero_eps_nfe = nfe[0]
                        nfe[0] = 0

                        self.assertLessEqual(nextafter_eps_nfe, fixed_eps_nfe)
                        self.assertLess(nextafter_eps_nfe, zero_eps_nfe)


if __name__ == '__main__':
    unittest.main()
