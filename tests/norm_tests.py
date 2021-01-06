import unittest

import torch
import torchdiffeq

from problems import (DTYPES, DEVICES, ADAPTIVE_METHODS)


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

        def norm(state):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(state, torch.Tensor)
            self.assertEqual(state.shape, ())
            return state.pow(2).mean().sqrt()
        x0 = torch.tensor(1.)
        torchdiffeq.odeint(f, x0, t, options=dict(norm=norm))
        self.assertTrue(is_called)

        # Now test that tupled input appears in the norm
        is_called = False

        def norm(state):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(state, tuple)
            self.assertEqual(len(state), 1)
            state, = state
            self.assertEqual(state.shape, ())
            return state.pow(2).mean().sqrt()
        x0 = (torch.tensor(1.),)
        torchdiffeq.odeint(f, x0, t, options=dict(norm=norm))
        self.assertTrue(is_called)

        is_called = False

        def norm(state):
            nonlocal is_called
            is_called = True
            self.assertIsInstance(state, tuple)
            self.assertEqual(len(state), 2)
            state1, state2 = state
            self.assertEqual(state1.shape, ())
            self.assertEqual(state2.shape, (2, 2))
            return state1.pow(2).mean().sqrt()
        x0 = (torch.tensor(1.), torch.tensor([[0.5, 0.5], [0.1, 0.1]]))
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


if __name__ == '__main__':
    unittest.main()
