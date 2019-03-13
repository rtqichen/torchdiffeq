import math
import torch


####################################
# Problem Class A. Single equations.
####################################
def A1():
    diffeq = lambda t, y: -y
    init = lambda: (torch.tensor(0.), torch.tensor(1.))
    solution = lambda t: torch.exp(-t)
    return diffeq, init, solution


def A2():
    diffeq = lambda t, y: -y**3 / 2
    init = lambda: (torch.tensor(0.), torch.tensor(1.))
    solution = lambda t: 1 / torch.sqrt(t + 1)
    return diffeq, init, solution


def A3():
    diffeq = lambda t, y: y * torch.cos(t)
    init = lambda: (torch.tensor(0.), torch.tensor(1.))
    solution = lambda t: torch.exp(torch.sin(t))
    return diffeq, init, solution


def A4():
    diffeq = lambda t, y: y / 4 * (1 - y / 20)
    init = lambda: (torch.tensor(0.), torch.tensor(1.))
    solution = lambda t: 20 / (1 + 19 * torch.exp(-t / 4))
    return diffeq, init, solution


def A5():
    diffeq = lambda t, y: (y - t) / (y + t)
    init = lambda: (torch.tensor(0.), torch.tensor(4.))
    return diffeq, init, None


#################################
# Problem Class B. Small systems.
#################################
def B1():

    def diffeq(t, y):
        dy0 = 2 * (y[0] - y[0] * y[1])
        dy1 = -(y[1] - y[0] * y[1])
        return torch.stack([dy0, dy1])

    def init():
        return torch.tensor(0.), torch.tensor([1., 3.])

    return diffeq, init, None


def B2():

    A = torch.tensor([[-1., 1., 0.], [1., -2., 1.], [0., 1., -1.]])

    def diffeq(t, y):
        dy = torch.mv(A, y)
        return dy

    def init():
        return torch.tensor(0.), torch.tensor([2., 0., 1.])

    return diffeq, init, None


def B3():

    def diffeq(t, y):
        dy0 = -y[0]
        dy1 = y[0] - y[1] * y[1]
        dy2 = y[1] * y[1]
        return torch.stack([dy0, dy1, dy2])

    def init():
        return torch.tensor(0.), torch.tensor([1., 0., 0.])

    return diffeq, init, None


def B4():

    def diffeq(t, y):
        a = torch.sqrt(y[0] * y[0] + y[1] * y[1])
        dy0 = -y[1] - y[0] * y[2] / a
        dy1 = y[0] - y[1] * y[2] / a
        dy2 = y[0] / a
        return torch.stack([dy0, dy1, dy2])

    def init():
        return torch.tensor(0.), torch.tensor([3., 0., 0.])

    return diffeq, init, None


def B5():

    def diffeq(t, y):
        dy0 = y[1] * y[2]
        dy1 = -y[0] * y[2]
        dy2 = -0.51 * y[0] * y[1]
        return torch.stack([dy0, dy1, dy2])

    def init():
        return torch.tensor(0.), torch.tensor([0., 1., 1.])

    return diffeq, init, None


####################################
# Problem Class C. Moderate systems.
####################################
def C1():

    A = torch.zeros(10, 10)
    A.view(-1)[:-1:11] = -1
    A.view(-1)[10::11] = 1

    def diffeq(t, y):
        return torch.mv(A, y)

    def init():
        y0 = torch.zeros(10)
        y0[0] = 1
        return torch.tensor(0.), y0

    return diffeq, init, None


def C2():

    A = torch.zeros(10, 10)
    A.view(-1)[:-1:11] = torch.linspace(-1, -9, 9)
    A.view(-1)[10::11] = torch.linspace(1, 9, 9)

    def diffeq(t, y):
        return torch.mv(A, y)

    def init():
        y0 = torch.zeros(10)
        y0[0] = 1
        return torch.tensor(0.), y0

    return diffeq, init, None


def C3():
    n = 10
    A = torch.zeros(n, n)
    A.view(-1)[::n + 1] = -2
    A.view(-1)[n::n + 1] = 1
    A.view(-1)[1::n + 1] = 1

    def diffeq(t, y):
        return torch.mv(A, y)

    def init():
        y0 = torch.zeros(n)
        y0[0] = 1
        return torch.tensor(0.), y0

    return diffeq, init, None


def C4():
    n = 51
    A = torch.zeros(n, n)
    A.view(-1)[::n + 1] = -2
    A.view(-1)[n::n + 1] = 1
    A.view(-1)[1::n + 1] = 1

    def diffeq(t, y):
        return torch.mv(A, y)

    def init():
        y0 = torch.zeros(n)
        y0[0] = 1
        return torch.tensor(0.), y0

    return diffeq, init, None


def C5():

    k2 = torch.tensor(2.95912208286)
    m0 = torch.tensor(1.00000597682)
    m = torch.tensor([
        0.000954786104043,
        0.000285583733151,
        0.0000437273164546,
        0.0000517759138449,
        0.00000277777777778,
    ]).view(1, 5)

    def diffeq(t, y):
        # y is 2 x 3 x 5
        # y[0] contains y, y[0] contains y'
        # second axis indexes space (x,y,z).
        # third axis indexes 5 bodies.

        dy = y[1, :, :]
        y = y[0]
        r = torch.sqrt(torch.sum(y**2, 0)).view(1, 5)
        d = torch.sqrt(torch.sum((y[:, :, None] - y[:, None, :])**2, 0))
        F = m.view(1, 1, 5) * ((y[:, None, :] - y[:, :, None]) / (d * d * d).view(1, 5, 5) + y.view(3, 1, 5) /
                               (r * r * r).view(1, 1, 5))
        F.view(3, 5 * 5)[:, ::6] = 0
        ddy = k2 * (-(m0 + m) * y / (r * r * r)) + F.sum(2)
        return torch.stack([dy, ddy], 0)

    def init():
        y0 = torch.tensor([
            3.42947415189, 3.35386959711, 1.35494901715, 6.64145542550, 5.97156957878, 2.18231499728, 11.2630437207,
            14.6952576794, 6.27960525067, -30.1552268759, 165699966404, 1.43785752721, -21.1238353380, 28.4465098142,
            15.388265967
        ]).view(5, 3).transpose(0, 1)

        dy0 = torch.tensor([
            -.557160570446, .505696783289, .230578543901, -.415570776342, .365682722812, .169143213293, -.325325669158,
            .189706021964, .0877265322780, -.0240476254170, -.287659532608, -.117219543175, -.176860753121,
            -.216393453025, -.0148647893090
        ]).view(5, 3).transpose(0, 1)

        return torch.tensor(0.), torch.stack([y0, dy0], 0)

    return diffeq, init, None


###################################
# Problem Class D. Orbit equations.
###################################
def _DTemplate(eps):

    def diffeq(t, y):
        r = (y[0]**2 + y[1]**2)**(3 / 2)
        dy0 = y[2]
        dy1 = y[3]
        dy2 = -y[0] / r
        dy3 = -y[1] / r
        return torch.stack([dy0, dy1, dy2, dy3])

    def init():
        return torch.tensor(0.), torch.tensor([1 - eps, 0, 0, math.sqrt((1 + eps) / (1 - eps))])

    return diffeq, init, None


D1 = lambda: _DTemplate(0.1)
D2 = lambda: _DTemplate(0.3)
D3 = lambda: _DTemplate(0.5)
D4 = lambda: _DTemplate(0.7)
D5 = lambda: _DTemplate(0.9)


##########################################
# Problem Class E. Higher order equations.
##########################################
def E1():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = -(y[1] / (t + 1) + (1 - 0.25 / (t + 1)**2) * y[0])
        return torch.stack([dy0, dy1])

    def init():
        return torch.tensor(0.), torch.tensor([.671396707141803, .0954005144474744])

    return diffeq, init, None


def E2():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = (1 - y[0]**2) * y[1] - y[0]
        return torch.stack([dy0, dy1])

    def init():
        return torch.tensor(0.), torch.tensor([2., 0.])

    return diffeq, init, None


def E3():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = y[0]**3 / 6 - y[0] + 2 * torch.sin(2.78535 * t)
        return torch.stack([dy0, dy1])

    def init():
        return torch.tensor(0.), torch.tensor([0., 0.])

    return diffeq, init, None


def E4():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = .32 - .4 * y[1]**2
        return torch.stack([dy0, dy1])

    def init():
        return torch.tensor(0.), torch.tensor([30., 0.])

    return diffeq, init, None


def E5():

    def diffeq(t, y):
        dy0 = y[1]
        dy1 = torch.sqrt(1 + y[1]**2) / (25 - t)
        return torch.stack([dy0, dy1])

    def init():
        return torch.tensor(0.), torch.tensor([0., 0.])

    return diffeq, init, None


###################
# Helper functions.
###################
def _to_tensor(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    return x
