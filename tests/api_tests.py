import unittest
import torch
import torchdiffeq

from problems import construct_problem, DTYPES, DEVICES, ADAPTIVE_METHODS


EPS = {torch.float32: 1e-4, torch.float64: 1e-12, torch.complex64: 1e-4}


class TestCollectionState(unittest.TestCase):
    def test_dense_output(self):
        for device in DEVICES:
            with self.subTest(device=device):
                # Integrate the state and its running integral together.
                y0 = torch.tensor([1., 2.], dtype=torch.float64, device=device, requires_grad=True)
                integral0 = y0.new_zeros((), requires_grad=True)
                rate = y0.new_tensor(0.3, requires_grad=True)

                def func(t, state):
                    y, integral = state
                    return rate * y, y.sum()

                times = y0.new_tensor([0., 0.23, 0.74, 1.])
                dense = torchdiffeq.odeint_dense(func, (y0, integral0), times[0], times[-1])
                reference = torchdiffeq.odeint(func, (y0, integral0), times)
                for i in (1, 2):
                    result = dense(times[i].cpu())
                    self.assertIsInstance(result, tuple)
                    for part, expected in zip(result, reference):
                        torch.testing.assert_close(part, expected[i])
                inputs = (y0, integral0, rate)
                actual_grad = torch.autograd.grad(result[0].sum() + result[1], inputs)
                expected_grad = torch.autograd.grad(reference[0][i].sum() + reference[1][i], inputs)
                for actual, expected in zip(actual_grad, expected_grad):
                    torch.testing.assert_close(actual, expected)

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


if __name__ == '__main__':
    unittest.main()
