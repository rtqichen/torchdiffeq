import unittest
import torch
import torchdiffeq

from problems import construct_problem

eps = 1e-12

torch.set_default_dtype(torch.float64)
TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def max_abs(tensor):
    return torch.max(torch.abs(tensor))


class TestCollectionState(unittest.TestCase):

    def test_dopri5(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = torchdiffeq.odeint(tuple_f, tuple_y0, t_points, method='dopri5')
        max_error0 = (sol - tuple_y[0]).max()
        max_error1 = (sol - tuple_y[1]).max()
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_dopri5_gradient(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0), t_points, method='dopri5')[i]
            self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adams(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = torchdiffeq.odeint(tuple_f, tuple_y0, t_points, method='adams')
        max_error0 = (sol - tuple_y[0]).max()
        max_error1 = (sol - tuple_y[1]).max()
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_adams_gradient(self):
        f, y0, t_points, sol = construct_problem(TEST_DEVICE)

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0), t_points, method='adams')[i]
            self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))


if __name__ == '__main__':
    unittest.main()
