import unittest
import torch
import torchdiffeq

eps = 1e-12

torch.set_default_dtype(torch.float64)
TEST_DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def max_abs(tensor):
    return torch.max(torch.abs(tensor))


def problem(reverse=False):

    class odefunc(torch.nn.Module):

        def __init__(self):
            super(odefunc, self).__init__()
            self.a = torch.nn.Parameter(torch.tensor(0.2).to(TEST_DEVICE))
            self.b = torch.nn.Parameter(torch.tensor(3.0).to(TEST_DEVICE))

        def forward(self, t, y):
            out = self.a + (y - (self.a * t + self.b))**5
            return out

        def y_exact(self, t):
            return self.a * t + self.b

    f = odefunc()

    t_points = torch.linspace(0, 8, 10).to(TEST_DEVICE).requires_grad_(True)
    sol = f.y_exact(t_points)

    def _flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
        return x[tuple(indices)]

    if reverse:
        t_points = _flip(t_points, 0).clone().detach()
        sol = _flip(sol, 0).clone().detach()

    return f, sol[0].detach(), t_points, sol


class TestLinearSolution(unittest.TestCase):

    def test_euler(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='euler')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_euler(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='euler')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_midpoint(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='midpoint')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_midpoint(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='midpoint')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_rk4(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='rk4')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_rk4_fixed_step_size(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='rk4', options={'step_size': 0.222})
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_rk4(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='rk4')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_explicit_adams(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='explicit_adams')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps * 10)

    def test_odeint_adams(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='adams')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_dopri5(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint(f, y0, t_points, method='dopri5')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_adjoint(self):
        f, y0, t_points, sol = problem()

        y = torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)


class TestLinearSolutionBackwardsInTime(unittest.TestCase):


    def test_odeint_euler(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points, method='euler')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_midpoint(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points, method='midpoint')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_rk4(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points, method='rk4')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_explicit_adams(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points, method='explicit_adams')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_adams(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points, method='adams')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_dopri5(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points, method='dopri5')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)

    def test_adjoint(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint_adjoint(f, y0, t_points, method='dopri5')
        max_error = (sol - y).max()
        self.assertLess(max_error, eps)


class TestNoIntegration(unittest.TestCase):

    def test_odeint_midpoint(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points[0:1], method='midpoint')
        max_error = (sol[0] - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_rk4(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points[0:1], method='rk4')
        max_error = (sol[0] - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_explicit_adams(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points[0:1], method='explicit_adams')
        max_error = (sol[0] - y).max()
        self.assertLess(max_error, eps)

    def test_odeint_adams(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points[0:1], method='adams')
        max_error = (sol[0] - y).max()
        self.assertLess(max_error, eps)

    def test_dopri5(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint(f, y0, t_points[0:1], method='dopri5')
        max_error = (sol[0] - y).max()
        self.assertLess(max_error, eps)

    def test_adjoint(self):
        f, y0, t_points, sol = problem(reverse=True)

        y = torchdiffeq.odeint_adjoint(f, y0, t_points[0:1], method='dopri5')
        max_error = (sol[0] - y).max()
        self.assertLess(max_error, eps)


class TestGradient(unittest.TestCase):

    def test_midpoint(self):

        f, y0, t_points, _ = problem()

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='midpoint')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_rk4(self):

        f, y0, t_points, _ = problem()

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='rk4')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_dopri5(self):
        f, y0, t_points, _ = problem()

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='dopri5')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adams(self):
        f, y0, t_points, _ = problem()

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='adams')
        self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adjoint(self):
        """
        Test against dopri5
        """
        f, y0, t_points, _ = problem()

        func = lambda y0, t_points: torchdiffeq.odeint(f, y0, t_points, method='dopri5')
        ys = func(y0, t_points)
        torch.manual_seed(0)
        gradys = torch.rand_like(ys)
        ys.backward(gradys)

        # reg_y0_grad = y0.grad
        reg_t_grad = t_points.grad
        reg_a_grad = f.a.grad
        reg_b_grad = f.b.grad

        f, y0, t_points, _ = problem()

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


class TestCollectionState(unittest.TestCase):

    def test_dopri5(self):
        f, y0, t_points, sol = problem()

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = torchdiffeq.odeint(tuple_f, tuple_y0, t_points, method='dopri5')
        max_error0 = (sol - tuple_y[0]).max()
        max_error1 = (sol - tuple_y[1]).max()
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_dopri5_gradient(self):
        f, y0, t_points, sol = problem()

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0), t_points, method='dopri5')[i]
            self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))

    def test_adams(self):
        f, y0, t_points, sol = problem()

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))
        tuple_y0 = (y0, y0)

        tuple_y = torchdiffeq.odeint(tuple_f, tuple_y0, t_points, method='adams')
        max_error0 = (sol - tuple_y[0]).max()
        max_error1 = (sol - tuple_y[1]).max()
        self.assertLess(max_error0, eps)
        self.assertLess(max_error1, eps)

    def test_adams_gradient(self):
        f, y0, t_points, sol = problem()

        tuple_f = lambda t, y: (f(t, y[0]), f(t, y[1]))

        for i in range(2):
            func = lambda y0, t_points: torchdiffeq.odeint(tuple_f, (y0, y0), t_points, method='adams')[i]
            self.assertTrue(torch.autograd.gradcheck(func, (y0, t_points)))


class TestCompareSolution(unittest.TestCase):

    def problem(self):

        class Odefunc(torch.nn.Module):

            def __init__(self):
                super(Odefunc, self).__init__()
                self.A = torch.nn.Parameter(torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]))
                self.unused_module = torch.nn.Linear(2, 5)
                self.nfe = 0

            def forward(self, t, y):
                self.nfe += 1
                return torch.mm(y**3, self.A)

        y0 = torch.tensor([[2., 0.]]).to(TEST_DEVICE).requires_grad_(True)
        t_points = torch.linspace(0., 25., 100).to(TEST_DEVICE).requires_grad_(True)
        func = Odefunc().to(TEST_DEVICE)
        return func, y0, t_points

    def test_adams_against_dopri5(self):
        with torch.no_grad():
            func, y0, t_points = self.problem()
            ys_ = torchdiffeq.odeint(func, y0, t_points, method='dopri5')
            nfe_ = func.nfe

            func, y0, t_points = self.problem()
            ys = torchdiffeq.odeint(func, y0, t_points, method='adams')
            nfe = func.nfe

        print('dopri5 NFE: {}'.format(nfe_))
        print('adams NFE: {}'.format(nfe))

        self.assertLess(nfe, nfe_)
        self.assertLess(max_abs(ys_ - ys), 1e-3)


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
