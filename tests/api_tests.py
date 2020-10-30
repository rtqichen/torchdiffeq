import unittest
import torch
import torchdiffeq

from problems import construct_problem, DTYPES, DEVICES, ADAPTIVE_METHODS


EPS = {torch.float32: 1e-5, torch.float64: 1e-12}


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
                        max_error0 = (sol - tuple_y[0]).max()
                        max_error1 = (sol - tuple_y[1]).max()
                        self.assertLess(max_error0, eps)
                        self.assertLess(max_error1, eps)

    def test_gradient(self):
        for device in DEVICES:
            f, y0, t_points, sol = construct_problem(device=device)
            tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
            for method in ADAPTIVE_METHODS:

                with self.subTest(device=device, method=method):
                    for i in range(2):
                        func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0), t_points, method=method)[i]
                        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))


if __name__ == '__main__':
    unittest.main()
