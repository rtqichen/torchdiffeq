import unittest
import torch
import torchdiffeq

from problems import construct_problem, PROBLEMS, DEVICES, METHODS
from problems import FIXED_EXPLICIT_METHODS
from torchdiffeq._impl.solvers import FixedGridODESolver


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


class TestFixedGridTimeGradient(unittest.TestCase):
    def test_observation_times_on_grid(self):
        for dtype in (torch.float32, torch.float64):
            for method in FIXED_EXPLICIT_METHODS:
                for reverse in (False, True):
                    for interp in ('linear', 'cubic'):
                        for query in (0.5, 0.375):
                            with self.subTest(dtype=dtype, method=method, reverse=reverse, interp=interp, query=query):
                                t = torch.tensor([0., query, 1.], dtype=dtype)
                                if reverse:
                                    t = 1 - t
                                t.requires_grad_()
                                y0 = torch.tensor([0.7, -0.3], dtype=dtype, requires_grad=True)
                                rate = torch.tensor(0.6, dtype=dtype, requires_grad=True)
                                weights = torch.tensor([[1., 2.], [3., -1.], [-2., 4.]], dtype=dtype)
                                solution = torchdiffeq.odeint(
                                    lambda time, value: rate * torch.ones_like(value), y0, t,
                                    method=method, options={'step_size': 0.25, 'interp': interp}
                                )
                                expected = y0.detach()[None] + rate.detach() * (t.detach() - t[0].detach())[:, None]
                                tolerance = 2e-6 if dtype == torch.float32 else 1e-12
                                self.assertTrue(torch.allclose(solution, expected, atol=tolerance, rtol=tolerance))
                                grad_t, grad_y0, grad_rate = torch.autograd.grad(
                                    (solution * weights).sum(), (t, y0, rate)
                                )
                                expected_t = rate.detach() * weights.sum(dim=1)
                                expected_t[0] = -rate.detach() * weights[1:].sum()
                                self.assertTrue(torch.allclose(grad_t, expected_t, atol=tolerance, rtol=tolerance))
                                self.assertTrue(torch.allclose(grad_y0, weights.sum(dim=0), atol=tolerance, rtol=tolerance))
                                expected_rate = ((t.detach() - t[0].detach()) * weights.sum(dim=1)).sum()
                                self.assertTrue(torch.allclose(grad_rate, expected_rate, atol=tolerance, rtol=tolerance))

    def test_linear_interpolation_endpoint_gradcheck(self):
        for query in (0., 1.):
            with self.subTest(query=query):
                arguments = tuple(
                    torch.tensor(value, dtype=torch.float64, requires_grad=True)
                    for value in (0., 1., 2., 5., query)
                )
                interpolate = lambda t0, t1, y0, y1, t: FixedGridODESolver._linear_interp(None, t0, t1, y0, y1, t)
                self.assertTrue(torch.autograd.gradcheck(interpolate, arguments))
                large_y0 = torch.tensor(1e20, dtype=torch.float64)
                small_y1 = torch.tensor(1., dtype=torch.float64)
                result = interpolate(arguments[0], arguments[1], large_y0, small_y1, arguments[-1])
                self.assertEqual(result.item(), large_y0.item() if query == 0 else small_y1.item())

    def test_tuple_observation_times_on_grid(self):
        for dtype in (torch.float32, torch.float64):
            for reverse in (False, True):
                for interp in ('linear', 'cubic'):
                    with self.subTest(dtype=dtype, reverse=reverse, interp=interp):
                        t = torch.tensor([0., 0.5, 1.], dtype=dtype)
                        if reverse:
                            t = 1 - t
                        t.requires_grad_()
                        y0 = (torch.tensor(0.7, dtype=dtype, requires_grad=True),
                              torch.ones(2, 3, dtype=dtype, requires_grad=True))
                        solution = torchdiffeq.odeint(
                            lambda time, state: (torch.ones_like(state[0]), 2 * torch.ones_like(state[1])),
                            y0, t, method='rk4', options={'step_size': 0.25, 'interp': interp}
                        )
                        gradients = torch.autograd.grad(solution[0][1] + solution[1][1].sum(), (t,) + y0)
                        self.assertTrue(torch.allclose(gradients[0], torch.tensor([-13., 13., 0.], dtype=dtype)))
                        self.assertEqual(gradients[1].item(), 1.)
                        self.assertTrue(torch.equal(gradients[2], torch.ones_like(y0[1])))


if __name__ == '__main__':
    unittest.main()
