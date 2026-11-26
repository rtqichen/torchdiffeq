import unittest
import torch
import torchdiffeq

from problems import construct_problem, DTYPES, DEVICES, ADAPTIVE_METHODS


EPS = {torch.float32: 1e-4, torch.float64: 1e-12, torch.complex64: 1e-4}


class TestCollectionState(unittest.TestCase):
    def test_forward(self):
        for dtype in DTYPES:
            eps = EPS[dtype]
            for device in DEVICES:
                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device)
                tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
                tuple_y0 = (y0, y0)
                for method in ADAPTIVE_METHODS:

                    with self.subTest(dtype=dtype, device=device, method=method):
                        tuple_y = torchdiffeq.odeint(tuple_f, tuple_y0, t_points, method=method)
                        max_error0 = (sol - tuple_y[0]).abs().max()
                        max_error1 = (sol - tuple_y[1]).abs().max()
                        self.assertLess(max_error0, eps)
                        self.assertLess(max_error1, eps)

    def test_gradient(self):
        for device in DEVICES:
            f, y0, t_points, sol = construct_problem(device=device)
            tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
            for method in ADAPTIVE_METHODS:
                if method == "scipy_solver":
                    continue

                with self.subTest(device=device, method=method):
                    for i in range(2):
                        func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0), t_points, method=method)[i]
                        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))


class TestScipySingleTimeShape(unittest.TestCase):
    def test_tensor_state_shape(self):
        for dtype in (torch.float32, torch.float64):
            for shape in ((), (3,), (2, 3), (1, 2, 1)):
                for interface in (torchdiffeq.odeint, torchdiffeq.odeint_adjoint):
                    with self.subTest(dtype=dtype, shape=shape, interface=interface.__name__):
                        y0 = torch.arange(1, 1 + torch.Size(shape).numel(), dtype=dtype).reshape(shape)
                        y0.requires_grad_()
                        t = torch.tensor([0.7], dtype=dtype, requires_grad=True)
                        options = {'adjoint_params': ()} if interface is torchdiffeq.odeint_adjoint else {}
                        solution = interface(lambda time, value: value, y0, t, method='scipy_solver', **options)
                        self.assertEqual(solution.shape, (1,) + shape)
                        self.assertEqual(solution.dtype, y0.dtype)
                        self.assertTrue(torch.equal(solution[0], y0))
                        if interface is torchdiffeq.odeint_adjoint:
                            grad_y0, grad_t = torch.autograd.grad(solution.sum(), (y0, t))
                            self.assertTrue(torch.equal(grad_y0, torch.ones_like(y0)))
                            self.assertTrue(torch.equal(grad_t, torch.zeros_like(t)))

    def test_tuple_state_control(self):
        for dtype in (torch.float32, torch.float64):
            for interface in (torchdiffeq.odeint, torchdiffeq.odeint_adjoint):
                with self.subTest(dtype=dtype, interface=interface.__name__):
                    y0 = (torch.tensor(2., dtype=dtype, requires_grad=True),
                          torch.arange(6, dtype=dtype).reshape(2, 3).requires_grad_())
                    t = torch.tensor([0.7], dtype=dtype)
                    options = {'adjoint_params': ()} if interface is torchdiffeq.odeint_adjoint else {}
                    solution = interface(lambda time, value: value, y0, t, method='scipy_solver', **options)
                    for initial, actual in zip(y0, solution):
                        self.assertEqual(actual.shape, (1,) + initial.shape)
                        self.assertTrue(torch.equal(actual[0], initial))
                    if interface is torchdiffeq.odeint_adjoint:
                        gradients = torch.autograd.grad(sum(value.sum() for value in solution), y0)
                        for initial, gradient in zip(y0, gradients):
                            self.assertTrue(torch.equal(gradient, torch.ones_like(initial)))


if __name__ == '__main__':
    unittest.main()
