import torch
import torchdiffeq
import unittest
import warnings

from problems import (construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS, FIXED_METHODS,
                      SCIPY_METHODS)


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
    def __init__(self, width, oscillate):
        super(_NeuralF, self).__init__()
        self.linears = torch.nn.Sequential(torch.nn.Linear(2, width),
                                           torch.nn.Tanh(),
                                           torch.nn.Linear(width, 2),
                                           torch.nn.Tanh())
        self.nfe = 0
        self.oscillate = oscillate

    def forward(self, t, x):
        self.nfe += 1
        out = self.linears(x)
        if self.oscillate:
            out = out * t.mul(20).sin()
        return out


class TestNorms(unittest.TestCase):
    def test_norm(self):
        torch.manual_seed(3456789)  # test can be flaky
        def norm(tensor):
            return tensor.abs().max()

        def large_norm(tensor):
            return 10 * tensor.abs().max()

        for dtype in DTYPES:
            for device in DEVICES:
                for method in ADAPTIVE_METHODS:
                    if dtype == torch.float32 and method == 'dopri8':
                        continue

                    with self.subTest(dtype=dtype, device=device, method=method):
                        x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
                        t = torch.tensor([0., 1.0], device=device, dtype=dtype)

                        norm_f = _NeuralF(width=10, oscillate=True).to(device, dtype)
                        torchdiffeq.odeint(norm_f, x0, t, method=method, options=dict(norm=norm))
                        large_norm_f = _NeuralF(width=10, oscillate=True).to(device, dtype)
                        with torch.no_grad():
                            for norm_param, large_norm_param in zip(norm_f.parameters(), large_norm_f.parameters()):
                                large_norm_param.copy_(norm_param)
                        torchdiffeq.odeint(large_norm_f, x0, t, method=method, options=dict(norm=large_norm))

                        self.assertLessEqual(norm_f.nfe, large_norm_f.nfe)

    def test_seminorm(self):
        torch.manual_seed(3456786)  # test can be flaky
        for dtype in DTYPES:
            for device in DEVICES:
                for method in ADAPTIVE_METHODS:
                    if method == 'adaptive_heun':
                        # Adaptive heun is consistently an awful choice with seminorms, it seems. My guess is that it's
                        # consistently overconfident with its step sizes, and that having seminorms turned off means
                        # that it actually gets it right more often.
                        continue
                    if dtype == torch.float32 and method == 'dopri8':
                        continue

                    with self.subTest(dtype=dtype, device=device, method=method):

                        x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype)
                        t = torch.tensor([0., 1.0], device=device, dtype=dtype)

                        norm_f = _NeuralF(width=256, oscillate=True).to(device, dtype)
                        out = torchdiffeq.odeint_adjoint(norm_f, x0, t, atol=3e-7, method=method)
                        norm_f.nfe = 0
                        out.sum().backward()

                        seminorm_f = _NeuralF(width=256, oscillate=True).to(device, dtype)
                        with torch.no_grad():
                            for norm_param, seminorm_param in zip(norm_f.parameters(), seminorm_f.parameters()):
                                seminorm_param.copy_(norm_param)
                        out = torchdiffeq.odeint_adjoint(seminorm_f, x0, t, atol=1e-6, method=method,
                                                         adjoint_options=dict(norm='seminorm'))
                        seminorm_f.nfe = 0
                        out.sum().backward()

                        self.assertLessEqual(seminorm_f.nfe, norm_f.nfe)


class TestEvents(unittest.TestCase):
    def test_wrong_event(self):
        x0 = torch.tensor([1.0, 2.0])
        t = torch.tensor([0., 1.0])

        for method in FIXED_METHODS:
            for event_name in ('event_accept_step', 'event_reject_step'):
                with self.subTest(method=method):
                    f = _NeuralF(width=10, oscillate=False)
                    setattr(f, event_name, lambda t0, y0, dt: None)
                    with self.assertRaises(ValueError):
                        torchdiffeq.odeint(f, x0, t, method=method)

        for method in SCIPY_METHODS:
            for event_name in ('event_step', 'event_accept_step', 'event_reject_step'):
                with self.subTest(method=method):
                    f = _NeuralF(width=10, oscillate=False)
                    setattr(f, event_name, lambda t0, y0, dt: None)
                    with self.assertRaises(ValueError):
                        torchdiffeq.odeint(f, x0, t, method=method)

    def test_steps(self):
        for adjoint in (False, True):
            for method in FIXED_METHODS + ADAPTIVE_METHODS:
                if method == 'dopri8':  # using torch.float32
                    continue
                with self.subTest(adjoint=adjoint, method=method):

                    counter = 0
                    accept_counter = 0
                    reject_counter = 0

                    def event_step(t0, y0, dt):
                        nonlocal counter
                        counter += 1

                    def event_accept_step(t0, y0, dt):
                        nonlocal accept_counter
                        accept_counter += 1

                    def event_reject_step(t0, y0, dt):
                        nonlocal reject_counter
                        reject_counter += 1

                    f = _NeuralF(width=10, oscillate=False).to()
                    if adjoint:
                        f.event_step_adjoint = event_step
                        if method in ADAPTIVE_METHODS:
                            f.event_accept_step_adjoint = event_accept_step
                            f.event_reject_step_adjoint = event_reject_step
                    else:
                        f.event_step = event_step
                        if method in ADAPTIVE_METHODS:
                            f.event_accept_step = event_accept_step
                            f.event_reject_step = event_reject_step
                    x0 = torch.tensor([1.0, 2.0])
                    t = torch.tensor([0., 1.0])

                    if method in FIXED_METHODS:
                        kwargs = dict(options=dict(step_size=0.1))
                    elif method == 'implicit_adams':
                        kwargs = dict(rtol=1e-3, atol=1e-4)
                    else:
                        kwargs = {}
                    torchdiffeq.odeint(f, x0, t, method=method, **kwargs)

                    if method in FIXED_METHODS:
                        self.assertEqual(counter, 10)
                    if method in ADAPTIVE_METHODS:
                        self.assertGreater(counter, 0)
                        self.assertEqual(accept_counter + reject_counter, counter)


if __name__ == '__main__':
    unittest.main()
