import math
import numpy as np
import scipy.linalg
import torch


class ConstantODE(torch.nn.Module):

    def __init__(self):
        super(ConstantODE, self).__init__()
        self.a = torch.nn.Parameter(torch.tensor(0.2))
        self.b = torch.nn.Parameter(torch.tensor(3.0))

    def forward(self, t, y):
        return self.a + (y - (self.a * t + self.b))**5

    def y_exact(self, t):
        return self.a * t + self.b


class SineODE(torch.nn.Module):
    def forward(self, t, y):
        return 2 * y / t + t**4 * torch.sin(2 * t) - t**2 + 4 * t**3

    def y_exact(self, t):
        return -0.5 * t**4 * torch.cos(2 * t) + 0.5 * t**3 * torch.sin(2 * t) + 0.25 * t**2 * torch.cos(
            2 * t
        ) - t**3 + 2 * t**4 + (math.pi - 0.25) * t**2


class LinearODE(torch.nn.Module):

    def __init__(self, dim=10):
        super(LinearODE, self).__init__()
        torch.manual_seed(0)
        self.dim = dim
        U = torch.randn(dim, dim) * 0.1
        A = 2 * U - (U + U.transpose(0, 1))
        self.A = torch.nn.Parameter(A)
        self.initial_val = np.ones((dim, 1))

    def forward(self, t, y):
        return torch.mm(self.A, y.reshape(self.dim, 1)).reshape(-1)

    def y_exact(self, t):
        t_numpy = t.detach().cpu().numpy()
        A_np = self.A.detach().cpu().numpy()
        ans = []
        for t_i in t_numpy:
            ans.append(np.matmul(scipy.linalg.expm(A_np * t_i), self.initial_val))
        return torch.stack([torch.tensor(ans_) for ans_ in ans]).reshape(len(t_numpy), self.dim).to(t)


PROBLEMS = {'constant': ConstantODE, 'linear': LinearODE, 'sine': SineODE}
DTYPES = (torch.float32, torch.float64)
DEVICES = ['cpu']
if torch.cuda.is_available():
    DEVICES.append('cuda')
FIXED_METHODS = ('euler', 'midpoint', 'rk4', 'explicit_adams', 'implicit_adams')
ADAMS_METHODS = ('explicit_adams', 'implicit_adams')
ADAPTIVE_METHODS = ('adaptive_heun', 'fehlberg2', 'bosh3', 'dopri5', 'dopri8')
SCIPY_METHODS = ('scipy_solver',)
METHODS = FIXED_METHODS + ADAPTIVE_METHODS + SCIPY_METHODS


def construct_problem(device, npts=10, ode='constant', reverse=False, dtype=torch.float64):

    f = PROBLEMS[ode]().to(dtype=dtype, device=device)

    t_points = torch.linspace(1, 8, npts, dtype=dtype, device=device, requires_grad=True)
    sol = f.y_exact(t_points)

    def _flip(x, dim):
        indices = [slice(None)] * x.dim()
        indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=device)
        return x[tuple(indices)]

    if reverse:
        t_points = _flip(t_points, 0).clone().detach()
        sol = _flip(sol, 0).clone().detach()

    return f, sol[0].detach().requires_grad_(True), t_points, sol


if __name__ == '__main__':
    f = SineODE().cpu()
    t_points = torch.linspace(1, 8, 100, device='cpu')
    sol = f.y_exact(t_points)

    import matplotlib.pyplot as plt
    plt.plot(t_points, sol)
    plt.show()
