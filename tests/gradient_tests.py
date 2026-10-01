import unittest
import torch
import torchdiffeq

from problems import construct_problem, PROBLEMS, DEVICES, METHODS


def max_abs(tensor):
    return torch.max(torch.abs(tensor))


class TestGradient(unittest.TestCase):
    def test_odeint(self):
        for device in DEVICES:
            for method in METHODS:

                if method == 'scipy_solver':
                    continue

                with self.subTest(device=device, method=method):
                    f, y0, t_points, _ = construct_problem(device=device)
                    func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method=method)
                    self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adjoint(self):
        for device in DEVICES:
            for method in METHODS:

                with self.subTest(device=device, method=method):
                    f, y0, t_points, _ = construct_problem(device=device)
                    func = lambda y0, t_points: torchdiffeq.odeint_adjoint(f, y0, t_points, method=method)
                    self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adjoint_against_odeint(self):
        """
        Test against dopri5
        """
        for device in DEVICES:
            for ode in PROBLEMS:
                for t_grad in (True, False):
                    if ode == 'constant':
                        eps = 1e-12
                    elif ode == 'linear':
                        eps = 1e-5
                    elif ode == 'sine':
                        eps = 5e-3
                    elif ode == 'exp':
                        eps = 1e-2
                    else:
                        raise RuntimeError

                    with self.subTest(device=device, ode=ode, t_grad=t_grad):
                        f, y0, t_points, _ = construct_problem(device=device, ode=ode)
                        t_points.requires_grad_(t_grad)

                        ys = torchdiffeq.odeint(f, y0, t_points, rtol=1e-9, atol=1e-12)
                        torch.manual_seed(0)
                        gradys = torch.rand_like(ys)
                        ys.backward(gradys)

                        reg_y0_grad = y0.grad.clone()
                        reg_t_grad = t_points.grad.clone() if t_grad else None
                        reg_params_grads = []
                        for param in f.parameters():
                            reg_params_grads.append(param.grad.clone())

                        y0.grad.zero_()
                        if t_grad:
                            t_points.grad.zero_()
                        for param in f.parameters():
                            param.grad.zero_()

                        ys = torchdiffeq.odeint_adjoint(f, y0, t_points, rtol=1e-9, atol=1e-12)
                        ys.backward(gradys)

                        adj_y0_grad = y0.grad
                        adj_t_grad = t_points.grad if t_grad else None
                        adj_params_grads = []
                        for param in f.parameters():
                            adj_params_grads.append(param.grad)

                        self.assertLess(max_abs(reg_y0_grad - adj_y0_grad), eps)
                        if t_grad:
                            self.assertLess(max_abs(reg_t_grad - adj_t_grad), eps)
                        for reg_grad, adj_grad in zip(reg_params_grads, adj_params_grads):
                            self.assertLess(max_abs(reg_grad - adj_grad), eps)


class TestComplexAdjointGradient(unittest.TestCase):

    def loss(self, solution):
        if isinstance(solution, tuple):
            return sum(self.loss(part) for part in solution)
        weights = solution.abs().new_tensor([0.2, 0.5, 1.0])
        weights = weights.view((3,) + (1,) * (solution.ndimension() - 1))
        loss = solution.abs().pow(2)
        if torch.is_complex(solution):
            loss = loss + 0.3 * solution.real + 0.2 * solution.imag
        else:
            loss = loss + 0.3 * solution
        return (weights * loss).sum()

    def check_against_exact(self, func, y0, t, exact, method='dopri5', adjoint_options=None, compare_direct=True):
        initial = y0 if isinstance(y0, tuple) else (y0,)
        params = tuple(func.parameters()) if isinstance(func, torch.nn.Module) else ()
        inputs = initial + ((t,) if t.requires_grad else ()) + params
        expected_grads = torch.autograd.grad(self.loss(exact), inputs, allow_unused=True)
        single_precision = initial[0].dtype in (torch.float32, torch.complex64)
        tol = 5e-4 if single_precision else 2e-7
        options = {'step_size': 0.005} if method == 'rk4' else None

        solvers = (torchdiffeq.odeint, torchdiffeq.odeint_adjoint) if compare_direct else (torchdiffeq.odeint_adjoint,)
        for solver in solvers:
            with self.subTest(solver=solver.__name__):
                kwargs = {}
                if solver is torchdiffeq.odeint_adjoint:
                    kwargs['adjoint_params'] = params
                    kwargs['adjoint_options'] = adjoint_options
                solution = solver(func, y0, t, rtol=1e-6 if single_precision else 1e-10,
                                  atol=1e-8 if single_precision else 1e-12,
                                  method=method, options=options, **kwargs)
                parts = solution if isinstance(solution, tuple) else (solution,)
                exact_parts = exact if isinstance(exact, tuple) else (exact,)
                for part, exact_part in zip(parts, exact_parts):
                    self.assertEqual(part.shape, exact_part.shape)
                    self.assertEqual(part.dtype, exact_part.dtype)
                    self.assertLess(max_abs(part - exact_part), tol)
                grads = torch.autograd.grad(self.loss(solution), inputs, allow_unused=True)
                for grad, expected_grad, value in zip(grads, expected_grads, inputs):
                    # Direct autograd leaves unused parameters unset; the adjoint returns zeros.
                    grad = torch.zeros_like(value) if grad is None else grad
                    expected_grad = torch.zeros_like(value) if expected_grad is None else expected_grad
                    self.assertEqual(grad.dtype, value.dtype)
                    self.assertLess(max_abs(grad - expected_grad), tol)

    def test_complex_state_time_gradients(self):
        for device in DEVICES:
            for dtype in (torch.complex64, torch.complex128):
                for reverse in (False, True):
                    for method in ('dopri5', 'rk4'):
                        with self.subTest(device=device, dtype=dtype, reverse=reverse, method=method):
                            y0 = torch.tensor([1 + 0.3j, -0.2 + 0.5j], dtype=dtype,
                                              device=device, requires_grad=True)
                            t = torch.tensor([0.2, 0.35, 0.8], dtype=y0.real.dtype, device=device)
                            t = t.flip(0) if reverse else t
                            t.requires_grad_(True)

                            def func(t, y):
                                return (0.7 + 1.2j + 0.15 * t) * y

                            exponent = (0.7 + 1.2j) * (t - t[0]) + 0.075 * (t.pow(2) - t[0].pow(2))
                            exact = exponent.exp().unsqueeze(-1) * y0
                            # Fixed-grid differentiation can differ from continuous time sensitivities.
                            self.check_against_exact(func, y0, t, exact, method=method,
                                                     compare_direct=method != 'rk4')

    def test_complex_state_parameter_gradients(self):
        class Func(torch.nn.Module):
            def __init__(self, dtype, device):
                super().__init__()
                self.rate = torch.nn.Parameter(torch.tensor(0.7 + 1.2j, dtype=dtype, device=device))
                self.drift = torch.nn.Parameter(self.rate.real.new_tensor(0.15))
                self.unused = torch.nn.Parameter(self.rate.real.new_tensor(0.4))

            def forward(self, t, y):
                return (self.rate + self.drift * t) * y

        for device in DEVICES:
            for dtype in (torch.complex64, torch.complex128):
                for reverse in (False, True):
                    for t_grad in (False, True):
                        with self.subTest(device=device, dtype=dtype, reverse=reverse, t_grad=t_grad):
                            func = Func(dtype, device)
                            y0 = torch.tensor([[1 + 0.3j], [-0.2 + 0.5j]], dtype=dtype,
                                              device=device, requires_grad=True)
                            t = torch.tensor([0.2, 0.35, 0.8], dtype=y0.real.dtype, device=device)
                            t = t.flip(0) if reverse else t
                            t.requires_grad_(t_grad)
                            exponent = func.rate * (t - t[0]) + 0.5 * func.drift * (t.pow(2) - t[0].pow(2))
                            exact = exponent.exp().view(-1, 1, 1) * y0
                            self.check_against_exact(func, y0, t, exact, adjoint_options={'norm': 'seminorm'})

    def test_real_state_complex_parameters(self):
        class Func(torch.nn.Module):
            def __init__(self, dtype, device):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor([[0.7]], dtype=dtype, device=device))
                complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
                self.phase = torch.nn.Parameter(torch.tensor(0.2 + 0.3j, dtype=complex_dtype, device=device))

            def forward(self, t, y):
                return torch.nn.functional.linear(y, self.weight) + self.phase.abs().pow(2) * y

        for device in DEVICES:
            for dtype in (torch.float32, torch.float64):
                for reverse in (False, True):
                    for t_grad in (False, True):
                        with self.subTest(device=device, dtype=dtype, reverse=reverse, t_grad=t_grad):
                            func = Func(dtype, device)
                            y0 = torch.tensor([[1.3], [-0.4]], dtype=dtype, device=device, requires_grad=True)
                            t = torch.tensor([0.2, 0.35, 0.8], dtype=dtype, device=device)
                            t = t.flip(0) if reverse else t
                            t.requires_grad_(t_grad)
                            rate = func.weight.squeeze() + func.phase.abs().pow(2)
                            exact = (rate * (t - t[0])).exp().view(-1, 1, 1) * y0
                            self.check_against_exact(func, y0, t, exact)

    def test_complex_tuple_state(self):
        class Func(torch.nn.Module):
            def __init__(self, device):
                super().__init__()
                self.rate = torch.nn.Parameter(torch.tensor(0.2, dtype=torch.float64, device=device))
                self.phase = torch.nn.Parameter(torch.tensor(0.1 + 0.4j, dtype=torch.complex128, device=device))

            def forward(self, t, y):
                return ((self.rate + self.phase) * y[0], (self.rate - self.phase) * y[1])

        for device in DEVICES:
            for reverse in (False, True):
                with self.subTest(device=device, reverse=reverse):
                    func = Func(device)
                    y0 = (torch.tensor([1 + 0.3j, -0.2 + 0.5j], dtype=torch.complex128,
                                       device=device, requires_grad=True),
                          torch.tensor(0.4 - 0.2j, dtype=torch.complex128, device=device, requires_grad=True))
                    t = torch.tensor([0.2, 0.35, 0.8], dtype=torch.float64, device=device)
                    t = t.flip(0) if reverse else t
                    t.requires_grad_(True)
                    exact = (((func.rate + func.phase) * (t - t[0])).exp().unsqueeze(-1) * y0[0],
                             ((func.rate - func.phase) * (t - t[0])).exp() * y0[1])
                    self.check_against_exact(func, y0, t, exact)

    def test_nonholomorphic_complex_state(self):
        class Func(torch.nn.Module):
            def __init__(self, dtype, device, real_output):
                super().__init__()
                self.rate = torch.nn.Parameter(torch.tensor(0.4, dtype=dtype, device=device))
                self.real_output = real_output

            def forward(self, t, y):
                return self.rate * (y.real if self.real_output else y.conj())

        for device in DEVICES:
            for dtype in (torch.complex64, torch.complex128):
                for reverse in (False, True):
                    for real_output in (False, True):
                        with self.subTest(device=device, dtype=dtype, reverse=reverse, real_output=real_output):
                            y0 = torch.tensor([1 + 0.3j, -0.2 + 0.5j], dtype=dtype,
                                              device=device, requires_grad=True)
                            func = Func(y0.real.dtype, device, real_output)
                            t = torch.tensor([0.2, 0.35, 0.8], dtype=y0.real.dtype, device=device)
                            t = t.flip(0) if reverse else t
                            t.requires_grad_(True)
                            exponent = func.rate * (t - t[0])
                            imaginary = y0.imag if real_output else (-exponent).exp().unsqueeze(-1) * y0.imag
                            exact = exponent.exp().unsqueeze(-1) * y0.real + 1j * imaginary
                            # Direct RK differentiation requires a state-domain-preserving vector field.
                            self.check_against_exact(func, y0, t, exact, compare_direct=not real_output)


class TestCompareAdjointGradient(unittest.TestCase):

    def problem(self, device):
        class Odefunc(torch.nn.Module):
            def __init__(self):
                super(Odefunc, self).__init__()
                self.A = torch.nn.Parameter(torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]))
                self.unused_module = torch.nn.Linear(2, 5)

            def forward(self, t, y):
                return torch.mm(y**3, self.A)

        y0 = torch.tensor([[2., 0.]], device=device, requires_grad=True)
        t_points = torch.linspace(0., 25., 10, device=device, requires_grad=True)
        func = Odefunc().to(device)
        return func, y0, t_points

    def test_against_dopri5(self):
        method_eps = {
            'dopri5': (3e-4, 1e-4, 2e-3),
            'scipy_solver': (3e-4, 1e-4, 2e-3),
        }
        for device in DEVICES:
            for method, eps in method_eps.items():
                for t_grad in (True, False):
                    with self.subTest(device=device, method=method):
                        func, y0, t_points = self.problem(device=device)
                        t_points.requires_grad_(t_grad)

                        ys = torchdiffeq.odeint_adjoint(func, y0, t_points, method=method)
                        gradys = torch.rand_like(ys) * 0.1
                        ys.backward(gradys)

                        adj_y0_grad = y0.grad
                        adj_t_grad = t_points.grad if t_grad else None
                        adj_A_grad = func.A.grad
                        self.assertEqual(max_abs(func.unused_module.weight.grad), 0)
                        self.assertEqual(max_abs(func.unused_module.bias.grad), 0)

                        func, y0, t_points = self.problem(device=device)
                        ys = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
                        ys.backward(gradys)

                        self.assertLess(max_abs(y0.grad - adj_y0_grad), eps[0])
                        if t_grad:
                            self.assertLess(max_abs(t_points.grad - adj_t_grad), eps[1])
                        self.assertLess(max_abs(func.A.grad - adj_A_grad), eps[2])


if __name__ == '__main__':
    unittest.main()
