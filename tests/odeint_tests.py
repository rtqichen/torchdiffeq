import torch
import torchdiffeq
import unittest
import warnings

from problems import (construct_problem, PROBLEMS, DTYPES, DEVICES, METHODS, ADAPTIVE_METHODS, FIXED_METHODS,
                      SCIPY_METHODS, ADAMS_METHODS)


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


class TestDiscontinuities(unittest.TestCase):
    def test_odeint_jump_t(self):
        for adjoint in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in ADAPTIVE_METHODS:
                        for new_options in (True, False):
                            if dtype == torch.float32 and method == 'dopri8':
                                continue

                            with self.subTest(adjoint=adjoint, dtype=dtype, device=device, method=method,
                                              new_options=new_options):

                                x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype, requires_grad=True)
                                t = torch.tensor([0., 1.0], device=device)

                                simple_f = _JumpF()
                                odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
                                simple_xs = odeint(simple_f, x0, t, atol=1e-7, method=method)

                                better_f = _JumpF()
                                if new_options:
                                    options = dict(jump_t=torch.tensor([0.5], device=device))
                                else:
                                    options = dict(grid_points=torch.tensor([0.5], device=device), eps=1e-6)
                                with warnings.catch_warnings():
                                    if not new_options:
                                        warnings.simplefilter('ignore')
                                    better_xs = odeint(better_f, x0, t, rtol=1e-6, atol=1e-6, method=method,
                                                       options=options)

                                self.assertLess(better_f.nfe, simple_f.nfe)

                                if adjoint:
                                    simple_f.nfe = 0
                                    better_f.nfe = 0
                                    with warnings.catch_warnings():
                                        if not new_options:
                                            warnings.simplefilter('ignore')
                                        simple_xs.sum().backward()
                                        better_xs.sum().backward()
                                    self.assertLess(better_f.nfe, simple_f.nfe)

    def test_odeint_perturb(self):
        for adjoint in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in FIXED_METHODS:
                        if method in ADAMS_METHODS:
                            continue
                        for new_options in (True, False):
                            for perturb in (True, False):
                                with self.subTest(adjoint=adjoint, dtype=dtype, device=device, method=method,
                                                  new_options=new_options, perturb=perturb):
                                    x0 = torch.tensor([1.0, 2.0], device=device, dtype=dtype, requires_grad=True)
                                    t = torch.tensor([0., 1.0], device=device)
                                    ts = []

                                    def f(t, x):
                                        ts.append(t.item())
                                        return -x

                                    if new_options:
                                        options = dict(step_size=0.5, perturb=perturb)
                                    else:
                                        options = dict(step_size=0.5, eps=1e-5 if perturb else 0.)

                                    with warnings.catch_warnings():
                                        if not new_options:
                                            warnings.simplefilter('ignore')
                                        odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
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
                                            if not new_options:
                                                warnings.simplefilter('ignore')
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
                    else:
                        # adjoint pass
                        self.assertEqual(t[0], 1.)
                        self.assertEqual(t[1], 0.)
                    return torch.linspace(0, 1, 11)

                odeint = torchdiffeq.odeint_adjoint if adjoint else torchdiffeq.odeint
                xs = odeint(f, x0, t, method='euler', options=dict(grid_constructor=grid_constructor))
                x1 = xs[1]

                # 'true' wrt the use of the Euler scheme
                true_x1 = x0 * 1.1 ** 10
                self.assertLess((x1 - true_x1).abs().max(), 1e-6)
                if adjoint:
                    x1.backward()
                    # 'true' wrt the use of the Euler scheme
                    true_x0_grad = 1.1 ** 10
                    self.assertLess((x0.grad - true_x0_grad).abs().max(), 1e-6)


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
        def f(t, x):
            return x
        t = torch.tensor([0., 1.])

        # First test that tensor input appears in the norm.
        is_called = False
        def norm(tensor):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tensor.shape, ())
            return tensor.pow(2).mean().sqrt()
        x0 = torch.tensor(1.)
        torchdiffeq.odeint(f, x0, t, options=dict(norm=norm))
        self.assertTrue(is_called)

        # Now test that tupled input appears in the norm
        is_called = False
        def norm(tensor):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(tensor, tuple)
            self.assertEqual(len(tensor), 1)
            tensor, = tensor
            self.assertEqual(tensor.shape, ())
            return tensor.pow(2).mean().sqrt()
        x0 = torch.tensor(1.),
        torchdiffeq.odeint(f, x0, t, options=dict(norm=norm))
        self.assertTrue(is_called)

        is_called = False
        def norm(tensor):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(tensor, tuple)
            self.assertEqual(len(tensor), 2)
            tensor1, tensor2 = tensor
            self.assertEqual(tensor1.shape, ())
            self.assertEqual(tensor2.shape, (2, 2))
            return tensor1.pow(2).mean().sqrt()
        x0 = torch.tensor(1.), torch.tensor([[0.5, 0.5], [0.1, 0.1]])
        torchdiffeq.odeint(f, x0, t, options=dict(norm=norm))
        self.assertTrue(is_called)

    def test_adjoint_norm(self):
        def f(t, x):
            return x
        t = torch.tensor([0., 1.])
        adjoint_params = (torch.rand(7, requires_grad=True), torch.rand((), requires_grad=True))

        def make_spy_on_adjoint_norm(adjoint_norm, actual_norm):
            is_spy_called = [False]

            def spy_on_adjoint_norm(tensor):
                nonlocal is_spy_called
                is_spy_called[0] = True
                norm_result = adjoint_norm(tensor)
                true_norm_result = actual_norm(tensor)
                self.assertIsInstance(norm_result, torch.Tensor)
                self.assertEqual(norm_result.shape, true_norm_result.shape)
                self.assertLess((norm_result - true_norm_result).abs().max(), 1e-6)
                return norm_result

            return spy_on_adjoint_norm, is_spy_called

        # Test the various auto-constructed adjoint norms with tensor (not tuple) state
        for shape in ((), (1,), (2, 2)):
            for use_adjoint_options, seminorm in ((False, False), (True, False), (True, True)):
                with self.subTest(shape=shape, use_adjoint_options=use_adjoint_options, seminorm=seminorm):
                    x0 = torch.full(shape, 1.)
                    if use_adjoint_options:
                        if seminorm:
                            # Test passing adjoint_options and wanting the seminorm
                            kwargs = dict(adjoint_options=dict(norm='seminorm'))
                        else:
                            # Test passing adjoint_options but not specify the adjoint norm
                            kwargs = dict(adjoint_options={})
                    else:
                        # Test not passing adjoint_options at all.
                        kwargs = {}
                    xs = torchdiffeq.odeint_adjoint(f, x0, t, adjoint_params=adjoint_params, **kwargs)
                    _adjoint_norm = xs.grad_fn.adjoint_options['norm']

                    is_called = False
                    def actual_norm(tensor_tuple):
                        nonlocal is_called
                        is_called = True
                        self.assertIsInstance(tensor_tuple, tuple)
                        t, y, adj_y, adj_param1, adj_param2 = tensor_tuple
                        self.assertEqual(t.shape, ())
                        self.assertEqual(y.shape, shape)
                        self.assertEqual(adj_y.shape, shape)
                        self.assertEqual(adj_param1.shape, (7,))
                        self.assertEqual(adj_param2.shape, (()))
                        out = max(t.abs(), y.pow(2).mean().sqrt(), adj_y.pow(2).mean().sqrt())
                        if not seminorm:
                            out = max(out, adj_param1.pow(2).mean().sqrt(), adj_param2.abs())
                        return out

                    xs.grad_fn.adjoint_options['norm'], is_spy_called = make_spy_on_adjoint_norm(_adjoint_norm,
                                                                                                 actual_norm)
                    xs.sum().backward()
                    self.assertTrue(is_called)
                    self.assertTrue(is_spy_called[0])

        # Test the various auto-constructed adjoint norms with tuple (not tensor) state
        for use_adjoint_options, seminorm in ((False, False), (True, False), (True, True)):
            with self.subTest(shape=shape, use_adjoint_options=use_adjoint_options, seminorm=seminorm):
                x0 = torch.tensor(1.), torch.tensor([[0.5, 0.5], [0.1, 0.1]])
                if use_adjoint_options:
                    if seminorm:
                        # Test passing adjoint_options and wanting the seminorm
                        kwargs = dict(adjoint_options=dict(norm='seminorm'))
                    else:
                        # Test passing adjoint_options but not specify the adjoint norm
                        kwargs = dict(adjoint_options={})
                else:
                    # Test not passing adjoint_options at all.
                    kwargs = {}
                xs = torchdiffeq.odeint_adjoint(f, x0, t, adjoint_params=adjoint_params, **kwargs)
                adjoint_options_dict = xs[0].grad_fn.next_functions[0][0].next_functions[0][0].adjoint_options
                _adjoint_norm = adjoint_options_dict['norm']

                is_called = False
                def actual_norm(tensor_tuple):
                    nonlocal is_called
                    is_called = True
                    self.assertIsInstance(tensor_tuple, tuple)
                    t, y, adj_y, adj_param1, adj_param2 = tensor_tuple
                    self.assertEqual(t.shape, ())
                    self.assertEqual(y.shape, (5,))
                    self.assertEqual(adj_y.shape, (5,))
                    self.assertEqual(adj_param1.shape, (7,))
                    self.assertEqual(adj_param2.shape, ())
                    ya = y[0]
                    yb = y[1:]
                    adj_ya = adj_y[0]
                    adj_yb = adj_y[1:4]
                    out = max(t.abs(), ya.abs(), yb.pow(2).mean().sqrt(), adj_ya.abs(), adj_yb.pow(2).mean().sqrt())
                    if not seminorm:
                        out = max(out, adj_param1.pow(2).mean().sqrt(), adj_param2.abs())
                    return out

                spy_on_adjoint_norm, is_spy_called = make_spy_on_adjoint_norm(_adjoint_norm, actual_norm)
                adjoint_options_dict['norm'] = spy_on_adjoint_norm
                xs[0].sum().backward()
                self.assertTrue(is_called)
                self.assertTrue(is_spy_called[0])

        # Test user-passed adjoint norms with tensor (not tuple) state
        is_called = False
        def adjoint_norm(tensor_tuple):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(tensor_tuple, tuple)
            t, y, adj_y, adj_param1, adj_param2 = tensor_tuple
            self.assertEqual(t.shape, ())
            self.assertEqual(y.shape, ())
            self.assertEqual(adj_y.shape, ())
            self.assertEqual(adj_param1.shape, (7,))
            self.assertEqual(adj_param2.shape, ())
            return max(t.abs(), y.pow(2).mean().sqrt(), adj_y.pow(2).mean().sqrt(), adj_param1.pow(2).mean().sqrt(),
                       adj_param2.abs())

        x0 = torch.tensor(1.)
        xs = torchdiffeq.odeint_adjoint(f, x0, t, adjoint_params=adjoint_params,
                                        adjoint_options=dict(norm=adjoint_norm))
        xs.sum().backward()
        self.assertTrue(is_called)

        # Test user-passed adjoint norms with tuple (not tensor) state
        is_called = False
        def adjoint_norm(tensor_tuple):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(tensor_tuple, tuple)
            t, ya, yb, adj_ya, adj_yb, adj_param1, adj_param2 = tensor_tuple
            self.assertEqual(t.shape, ())
            self.assertEqual(ya.shape, ())
            self.assertEqual(yb.shape, (2, 2))
            self.assertEqual(adj_ya.shape, ())
            self.assertEqual(adj_yb.shape, (2, 2))
            self.assertEqual(adj_param1.shape, (7,))
            self.assertEqual(adj_param2.shape, ())
            return max(t.abs(), ya.abs(), yb.pow(2).mean().sqrt(), adj_ya.abs(), adj_yb.pow(2).mean().sqrt(),
                       adj_param1.pow(2).mean().sqrt(), adj_param2.abs())

        x0 = torch.tensor(1.), torch.tensor([[0.5, 0.5], [0.1, 0.1]])
        xs = torchdiffeq.odeint_adjoint(f, x0, t, adjoint_params=adjoint_params,
                                        adjoint_options=dict(norm=adjoint_norm))
        xs[0].sum().backward()
        self.assertTrue(is_called)

    def test_large_norm(self):
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


class TestCallbacks(unittest.TestCase):
    def test_wrong_callback(self):
        x0 = torch.tensor([1.0, 2.0])
        t = torch.tensor([0., 1.0])

        for method in FIXED_METHODS:
            for callback_name in ('callback_accept_step', 'callback_reject_step'):
                with self.subTest(method=method):
                    f = _NeuralF(width=10, oscillate=False)
                    setattr(f, callback_name, lambda t0, y0, dt: None)
                    with self.assertRaises(ValueError):
                        torchdiffeq.odeint(f, x0, t, method=method)

        for method in SCIPY_METHODS:
            for callback_name in ('callback_step', 'callback_accept_step', 'callback_reject_step'):
                with self.subTest(method=method):
                    f = _NeuralF(width=10, oscillate=False)
                    setattr(f, callback_name, lambda t0, y0, dt: None)
                    with self.assertRaises(ValueError):
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


if __name__ == '__main__':
    unittest.main()
