# Implemention of symplectic integrator.
# Whole input as (q,p) and  the shape is (bath_size, 2N)
# where N is the number of paritcles 

import torch
from .solvers import FixedGridODESolver
from .rk_common import rk4_alt_step_func

# symplectic integrators constatn 
_b1 = 1.0/(4.0 - 2.0*pow(2.0,1.0/3.0))
_b2 = (1.0 - pow(2.0,1.0/3.0))/(4.0 - 2.0*pow(2.0,1.0/3.0))
_c1 = 1.0 / (2.0 - pow(2.0,1.0/3.0))
_c2 = 1.0 / (1.0 - pow(2.0,2.0/3.0))


class Yoshida4th(FixedGridODESolver):
    def __init__(self, eps=0., **kwargs):
        super(Yoshida4th, self).__init__(**kwargs)
        self.eps = torch.as_tensor(eps, dtype=self.dtype, device=self.device)

    def _step_symplectic(self, func, y, t, h, h2):
        dy = torch.zeros(y.size())
        n = len(y) // 2
        k_ = func(t + self.eps, y)
        for i in range(n):
            dy[i] = h*_c1*k_[i] + h2*_c1*_b1*k_[n+i]
            dy[n+i] = h*_b1*k_[n+i]

        k_ = func(t + self.eps, y + dy)
        for i in range(n):
            dy[i] = dy[i] \
                         + h*_c2*k_[i] + h2*_c2*_b2*k_[n+i]
            dy[n+i] = dy[n+i] + h*_b2*k_[n+i]

        k_ = func(t + self.eps, y + dy)
        for i in range(n):
            dy[i] = dy[i] \
                         + h*_c1*k_[i] + h2*_c1*_b2*k_[n+i]
            dy[n+i] = dy[n+i] + h*_b2*k_[n+i]

        k_ = func(t + self.eps, y + dy)
        for i in range(n):
            dy[n+i] = dy[n+i] + h*_b1*k_[n+i]

        return dy

    def _step_func(self, func, t, dt, y):
        h2 = dt * dt
        h = dt

        return self._step_symplectic(func, y, t, h, h2)

