import unittest
import torch
import torchdiffeq

from problems import construct_problem

eps = 1e-12

torch.set_default_dtype(torch.float64)
TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def max_abs(tensor):
    return torch.max(torch.abs(tensor))


class TestGradient(unittest.TestCase):

    def test_midpoint(self):

        f, y0, t_points, _ = construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='midpoint')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_rk4(self):

        f, y0, t_points, _ = construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='rk4')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_dopri5(self):
        f, y0, t_points, _ = construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='dopri5')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adams(self):
        f, y0, t_points, _ = construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='adams')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adjoint(self):
        """
        Test against dopri5
        """
        f, y0, t_points, _ = construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='dopri5')
        ys = func(y0, t_points)
        torch.manual_seed(0)
        gradys = torch.rand_like(ys)
        ys.backward(gradys)

        # reg_y0_grad = y0.grad
        reg_t_grad = t_points.grad
        reg_a_grad = f.a.grad
        reg_b_grad = f.b.grad

        f, y0, t_points, _ = construct_problem(TEST_DEVICE)

        func = lambda y0, t_points: torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
        ys = func(y0, t_points)
        ys.backward(gradys)

        # adj_y0_grad = y0.grad
        adj_t_grad = t_points.grad
        adj_a_grad = f.a.grad
        adj_b_grad = f.b.grad

        # self.assertLess(max_abs(reg_y0_grad - adj_y0_grad), eps)
        self.assertLess(max_abs(reg_t_grad - adj_t_grad), eps)
        self.assertLess(max_abs(reg_a_grad - adj_a_grad), eps)
        self.assertLess(max_abs(reg_b_grad - adj_b_grad), eps)


class TestCompareAdjointGradient(unittest.TestCase):

    def problem(self):

        class Odefunc(torch.nn.Module):

            def __init__(self):
                super(Odefunc, self).__init__()
                self.A = torch.nn.Parameter(torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]))
                self.unused_module = torch.nn.Linear(2, 5)

            def forward(self, t, y):
                return torch.mm(y**3, self.A)

        y0 = torch.tensor([[2., 0.]]).to(TEST_DEVICE).requires_grad_(True)
        t_points = torch.linspace(0., 25., 10).to(TEST_DEVICE).requires_grad_(True)
        func = Odefunc().to(TEST_DEVICE)
        return func, y0, t_points

    def test_dopri5_adjoint_against_dopri5(self):
        func, y0, t_points = self.problem()
        ys = torchdiffeq.odeint_adjoint(func, y0, t_points, method='dopri5')
        gradys = torch.rand_like(ys) * 0.1
        ys.backward(gradys)

        adj_y0_grad = y0.grad
        adj_t_grad = t_points.grad
        adj_A_grad = func.A.grad
        self.assertEqual(max_abs(func.unused_module.weight.grad), 0)
        self.assertEqual(max_abs(func.unused_module.bias.grad), 0)

        func, y0, t_points = self.problem()
        ys = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
        ys.backward(gradys)

        self.assertLess(max_abs(y0.grad - adj_y0_grad), 3e-4)
        self.assertLess(max_abs(t_points.grad - adj_t_grad), 1e-4)
        self.assertLess(max_abs(func.A.grad - adj_A_grad), 2e-3)

    def test_adams_adjoint_against_dopri5(self):
        func, y0, t_points = self.problem()
        ys_ = torchdiffeq.odeint_adjoint(func, y0, t_points, method='adams')
        gradys = torch.rand_like(ys_) * 0.1
        ys_.backward(gradys)

        adj_y0_grad = y0.grad
        adj_t_grad = t_points.grad
        adj_A_grad = func.A.grad
        self.assertEqual(max_abs(func.unused_module.weight.grad), 0)
        self.assertEqual(max_abs(func.unused_module.bias.grad), 0)

        func, y0, t_points = self.problem()
        ys = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
        ys.backward(gradys)

        self.assertLess(max_abs(y0.grad - adj_y0_grad), 5e-2)
        self.assertLess(max_abs(t_points.grad - adj_t_grad), 5e-4)
        self.assertLess(max_abs(func.A.grad - adj_A_grad), 2e-2)


if __name__ == '__main__':
    unittest.main()
