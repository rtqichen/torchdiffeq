import torch
import torchdiffeq
import unittest
import warnings

from problems import construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS, SCIPY_METHODS


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
                        problems = PROBLEMS if method in ADAPTIVE_METHODS + SCIPY_METHODS else ('constant',)
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
                                if method in SCIPY_METHODS:
                                    y0 = y0.detach()
                                    t_points = t_points.detach()
                                    _f = f
                                    f = lambda t, y: _f(t, y).detach()
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
                                y0 = y0.detach()
                                t_points = t_points.detach()
                                _f = f
                                f = lambda t, y: _f(t, y).detach()

                                y = torchdiffeq.odeint(f, y0, t_points, method='scipy_solver',
                                                       options={"solver": solver})
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
                                y0 = y0.detach()
                                t_points = t_points.detach()
                                _f = f
                                f = lambda t, y: _f(t, y).detach()

                                y = torchdiffeq.odeint(f, y0, t_points[0:1], method=method)
                                self.assertLess((sol[0] - y).abs().max(), 1e-12)


class _JumpF:
    def __init__(self):
        self.nfe = 0

    def __call__(self, t, x):
        self.nfe += 1
        if t < 0.5:
            return -0.5 * x
        else:
            return x ** 2


class TestStepLocations(unittest.TestCase):
    def test_odeint_step_locations(self):
        for dtype in DTYPES:
            for device in DEVICES:
                for method in ADAPTIVE_METHODS:
                    for new_options in (True, False):
                        if dtype == torch.float32 and method == 'dopri8':
                            continue

                        with self.subTest(dtype=dtype, device=device, method=method):

                            x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
                            t = torch.tensor([0., 1.0], device=device)

                            simple_f = _JumpF()
                            torchdiffeq.odeint(simple_f, x0, t, method=method)

                            better_f = _JumpF()
                            if new_options:
                                options = dict(jump_locations=torch.tensor([0.5], device=device))
                            else:
                                options = dict(grid_points=torch.tensor([0.5], device=device), eps=1e-6)
                            with warnings.catch_warnings():
                                if not new_options:
                                    warnings.simplefilter('ignore')
                                torchdiffeq.odeint(better_f, x0, t, method=method, options=options)

                            self.assertLess(better_f.nfe, simple_f.nfe)


class _NeuralF(torch.nn.Module):
    def __init__(self):
        super(_NeuralF, self).__init__()
        self.linear1 = torch.nn.Linear(2, 10)
        self.linear2 = torch.nn.Linear(10, 2)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.linear2(self.linear1(x).tanh()).tanh()


class TestNorms(unittest.TestCase):
    def test_norm(self):
        def norm(tensor):
            return tensor.max()

        def double_norm(tensor):
            return 2 * tensor.max()

        for dtype in DTYPES:
            for device in DEVICES:
                for method in ADAPTIVE_METHODS:
                    if dtype == torch.float32 and method == 'dopri8':
                        continue

                    with self.subTest(dtype=dtype, device=device, method=method):
                        x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
                        t = torch.tensor([0., 1.0], device=device, dtype=dtype)

                        norm_f = _NeuralF().to(dtype, device)
                        torchdiffeq.odeint(norm_f, x0, t, method=method, options=dict(norm=norm))
                        double_norm_f = _NeuralF().to(dtype, device)
                        torchdiffeq.odeint(double_norm_f, x0, t, method=method, options=dict(norm=double_norm))

                        self.assertLess(norm_f.nfe, double_norm_f.nfe)

    def test_seminorm(self):
        for dtype in DTYPES:
            for device in DEVICES:
                for method in ADAPTIVE_METHODS:
                    if dtype == torch.float32 and method == 'dopri8':
                        continue

                    with self.subTest(dtype=dtype, device=device, method=method):

                        x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
                        t = torch.tensor([0., 1.0], device=device, dtype=dtype)

                        norm_f = _NeuralF().to(dtype, device)
                        out = torchdiffeq.odeint_adjoint(norm_f, x0, t, method=method)
                        norm_f.nfe = 0
                        out.sum().backward()

                        seminorm_f = _NeuralF().to(dtype, device)
                        out = torchdiffeq.odeint_adjoint(seminorm_f, x0, t, method=method,
                                                         adjoint_options=dict(norm='seminorm'))
                        seminorm_f.nfe = 0
                        out.sum().backward()

                        self.assertLess(seminorm_f.nfe, norm_f.nfe)


if __name__ == '__main__':
    unittest.main()
