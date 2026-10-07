from functools import partial
import unittest
import warnings

import torch
import torchdiffeq

from problems import (construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS, FIXED_METHODS, SCIPY_METHODS, IMPLICIT_METHODS)


def rel_error(true, estimate):
    return ((true - estimate) / true).abs().max()


class TestSolverError(unittest.TestCase):

    def test_odeint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:

                        if method in SCIPY_METHODS and dtype == torch.complex64:
                            # scipy solvers don't support complex types.
                            continue

                        kwargs = dict()
                        # Have to increase tolerance for dopri8.
                        if method == 'dopri8' and dtype == torch.float64:
                            kwargs = dict(rtol=1e-12, atol=1e-14)
                        if method == 'dopri8' and dtype == torch.float32:
                            kwargs = dict(rtol=1e-7, atol=1e-7)

                        if method in ADAPTIVE_METHODS:
                            if method in IMPLICIT_METHODS:
                                problems = PROBLEMS
                            else:
                                problems = tuple(problem for problem in PROBLEMS)
                        elif method in IMPLICIT_METHODS:
                            problems = ('constant', 'exp')
                        else:
                            problems = ('constant',)

                        for ode in problems:
                            if method in ['adaptive_heun', 'bosh3']:
                                eps = 4e-3
                            elif ode == 'linear':
                                eps = 2e-3
                            elif ode == 'exp':
                                eps = 5e-2
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
                                if torch.is_complex(y0) and solver in ["Radau", "LSODA"]:
                                    # scipy solvers don't support complex types.
                                    continue
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

                            if method == "dopri8":
                                # Doesn't seem to work for some reason.
                                continue

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

                        # Singluar matrix error with float32 and implicit_euler
                        if dtype == torch.float32 and method == 'implicit_euler':
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


class TestMinMaxStep(unittest.TestCase):
    def test_min_max_step(self):
        # LSODA will complain about convergence otherwise
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for device in DEVICES:
                for min_step in (0, 2):
                    for max_step in (float('inf'), 5):
                        for method, options in [('dopri5', {}), ('scipy_solver', {"solver": "LSODA"})]:
                            options['min_step'] = min_step
                            options['max_step'] = max_step
                            f, y0, t_points, sol = construct_problem(device=device, ode="linear")
                            torchdiffeq.odeint(f, y0, t_points, method=method, options=options)
                            # Check min step produces far fewer evaluations
                            if min_step > 0:
                                self.assertLess(f.nfe, 50)
                            else:
                                self.assertGreater(f.nfe, 100)


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


class TestCallbacks(unittest.TestCase):
    def test_wrong_callback(self):
        x0 = torch.tensor([1.0, 2.0])
        t = torch.tensor([0., 1.0])

        for method in FIXED_METHODS:
            for callback_name in ('callback_accept_step', 'callback_reject_step'):
                with self.subTest(method=method):
                    f = _NeuralF(width=10, oscillate=False)
                    setattr(f, callback_name, lambda t0, y0, dt: None)
                    with self.assertWarns(Warning):
                        torchdiffeq.odeint(f, x0, t, method=method)

        for method in SCIPY_METHODS:
            for callback_name in ('callback_step', 'callback_accept_step', 'callback_reject_step'):
                with self.subTest(method=method):
                    f = _NeuralF(width=10, oscillate=False)
                    setattr(f, callback_name, lambda t0, y0, dt: None)
                    with self.assertWarns(Warning):
                        torchdiffeq.odeint(f, x0, t, method=method)

    def test_steps(self):
        for forward, adjoint in ((False, True), (True, False), (True, True)):
            for method in FIXED_METHODS + ADAPTIVE_METHODS:
                if method == 'dopri8':  # using torch.float32
                    continue
                with self.subTest(forward=forward, adjoint=adjoint, method=method):

                    f = _NeuralF(width=10, oscillate=False)

                    if forward:
                        forward_counter = 0
                        forward_accept_counter = 0
                        forward_reject_counter = 0

                        def callback_step(t0, y0, dt):
                            nonlocal forward_counter
                            forward_counter += 1

                        def callback_accept_step(t0, y0, dt):
                            nonlocal forward_accept_counter
                            forward_accept_counter += 1

                        def callback_reject_step(t0, y0, dt):
                            nonlocal forward_reject_counter
                            forward_reject_counter += 1

                        f.callback_step = callback_step
                        if method in ADAPTIVE_METHODS:
                            f.callback_accept_step = callback_accept_step
                            f.callback_reject_step = callback_reject_step

                    if adjoint:
                        adjoint_counter = 0
                        adjoint_accept_counter = 0
                        adjoint_reject_counter = 0

                        def callback_step_adjoint(t0, y0, dt):
                            nonlocal adjoint_counter
                            adjoint_counter += 1

                        def callback_accept_step_adjoint(t0, y0, dt):
                            nonlocal adjoint_accept_counter
                            adjoint_accept_counter += 1

                        def callback_reject_step_adjoint(t0, y0, dt):
                            nonlocal adjoint_reject_counter
                            adjoint_reject_counter += 1

                        f.callback_step_adjoint = callback_step_adjoint
                        if method in ADAPTIVE_METHODS:
                            f.callback_accept_step_adjoint = callback_accept_step_adjoint
                            f.callback_reject_step_adjoint = callback_reject_step_adjoint

                    x0 = torch.tensor([1.0, 2.0])
                    t = torch.tensor([0., 1.0])

                    if method in FIXED_METHODS:
                        kwargs = dict(options=dict(step_size=0.1))
                    elif method == 'implicit_adams':
                        kwargs = dict(rtol=1e-3, atol=1e-4)
                    else:
                        kwargs = {}
                    xs = torchdiffeq.odeint_adjoint(f, x0, t, method=method, **kwargs)

                    if forward:
                        if method in FIXED_METHODS:
                            self.assertEqual(forward_counter, 10)
                        if method in ADAPTIVE_METHODS:
                            self.assertGreater(forward_counter, 0)
                            self.assertEqual(forward_accept_counter + forward_reject_counter, forward_counter)
                    if adjoint:
                        xs.sum().backward()
                        if method in FIXED_METHODS:
                            self.assertEqual(adjoint_counter, 10)
                        if method in ADAPTIVE_METHODS:
                            self.assertGreater(adjoint_counter, 0)
                            self.assertEqual(adjoint_accept_counter + adjoint_reject_counter, adjoint_counter)


class TestAdamsGridSpacing(unittest.TestCase):
    def test_shortened_final_step(self):
        for method in ('explicit_adams', 'implicit_adams'):
            for dtype in (torch.float32, torch.float64):
                for reverse in (False, True):
                    with self.subTest(method=method, dtype=dtype, reverse=reverse):
                        t = torch.tensor([0., 1.], dtype=dtype)
                        if reverse:
                            t = t.flip(0)
                        y0 = (t[0] ** 3 / 3).reshape(1)
                        ys = torchdiffeq.odeint(lambda t, y: t ** 2 * torch.ones_like(y),
                                               y0, t, method=method, options={'step_size': 0.3})
                        torch.testing.assert_close(ys[:, 0], t ** 3 / 3, atol=2e-7, rtol=2e-6)

    def test_changed_spacing_in_custom_and_default_grids(self):
        grid = torch.tensor([0., 0.2, 0.4, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.],
                            dtype=torch.float64)
        for method in ('explicit_adams', 'implicit_adams'):
            for reverse in (False, True):
                for custom_grid in (False, True):
                    with self.subTest(method=method, reverse=reverse, custom_grid=custom_grid):
                        integration_grid = grid.flip(0) if reverse else grid
                        if custom_grid:
                            t = integration_grid[[0, -1]]
                            options = {'grid_constructor': lambda f, y0, t: integration_grid.to(t)}
                        else:
                            t = integration_grid
                            options = {}
                        ys = torchdiffeq.odeint(lambda t, y: t ** 2 * torch.ones_like(y),
                                               (t[0] ** 3 / 3).reshape(1), t,
                                               method=method, options=options)
                        torch.testing.assert_close(ys[:, 0], t ** 3 / 3, atol=1e-13, rtol=1e-13)

    def test_uniform_history_survives_roundoff_and_nonuniform_output_times(self):
        for method in ('explicit_adams', 'implicit_adams'):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(method=method, dtype=dtype):
                    calls = []

                    def f(t, y):
                        calls.append(t)
                        return torch.ones_like(y)

                    t = torch.linspace(50., 51.2, 13, dtype=dtype)
                    ys = torchdiffeq.odeint(f, torch.zeros(1, dtype=dtype), t, method=method)
                    uniform_calls = len(calls)
                    self.assertLess(uniform_calls, 3 * (len(t) - 1))
                    torch.testing.assert_close(ys[:, 0], t - t[0], atol=1e-6, rtol=1e-6)

                    calls.clear()
                    # These are output times only; the custom integration grid is unchanged.
                    output_t = torch.tensor([50., 50.03, 50.27, 50.8, 51.11, 51.2], dtype=dtype)
                    ys = torchdiffeq.odeint(f, torch.zeros(1, dtype=dtype), output_t, method=method,
                                           options={'grid_constructor': lambda f, y0, t_out: t})
                    self.assertEqual(len(calls), uniform_calls)
                    torch.testing.assert_close(ys[:, 0], output_t - output_t[0], atol=1e-6, rtol=1e-6)

    def test_shortened_final_step_gradients(self):
        for method in ('explicit_adams', 'implicit_adams'):
            with self.subTest(method=method):
                y0 = torch.tensor([0.4], dtype=torch.float64, requires_grad=True)
                scale = torch.tensor(2., dtype=torch.float64, requires_grad=True)
                t = torch.tensor([0., 1.], dtype=torch.float64, requires_grad=True)
                ys = torchdiffeq.odeint(lambda t, y: scale * t ** 2 * torch.ones_like(y),
                                       y0, t, method=method, options={'step_size': 0.3})
                grads = torch.autograd.grad(ys[-1].sum(), (y0, scale, t))
                torch.testing.assert_close(ys[-1], y0 + scale / 3, atol=1e-13, rtol=1e-13)
                torch.testing.assert_close(grads[0], torch.ones_like(y0), atol=1e-13, rtol=1e-13)
                torch.testing.assert_close(grads[1], torch.tensor(1. / 3, dtype=scale.dtype),
                                           atol=1e-13, rtol=1e-13)
                torch.testing.assert_close(grads[2], torch.tensor([0., 2.], dtype=t.dtype),
                                           atol=1e-13, rtol=1e-13)

    def test_large_time_origin_does_not_hide_changed_spacing(self):
        offset = 2. ** 40
        unit = 2. ** -12
        t = offset + torch.tensor([0., 8., 16., 24., 28.], dtype=torch.float64) * unit
        for method in ('explicit_adams', 'implicit_adams'):
            with self.subTest(method=method):
                ys = torchdiffeq.odeint(lambda time, y: (time - offset) * torch.ones_like(y),
                                       torch.zeros(1, dtype=t.dtype), t, method=method)
                expected = 0.5 * (t - offset) ** 2
                torch.testing.assert_close(ys[:, 0], expected, atol=1e-18, rtol=1e-13)


if __name__ == '__main__':
    unittest.main()
