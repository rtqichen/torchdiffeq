import unittest

import torch
import torchdiffeq

from problems import ADAPTIVE_METHODS, DTYPES


class PrescribedField(torch.nn.Module):
    def __init__(self, dtype, kind, offset=0.):
        super(PrescribedField, self).__init__()
        self.rate = torch.nn.Parameter(torch.tensor(2., dtype=dtype))
        self.kind = kind
        self.offset = offset

    def forward(self, t, y):
        time = t - self.offset
        if self.kind == 'jump_t':
            forcing = ((time > 0.4) & (time < 0.41)).to(self.rate)
        else:
            forcing = torch.clamp(torch.min(time - 0.375, 0.4375 - time), min=0)
        rate = self.rate * forcing
        if isinstance(y, tuple):
            return tuple(rate * torch.ones_like(part) for part in y)
        return rate * torch.ones_like(y)


class TestAdjointTimeOptions(unittest.TestCase):
    def test_adaptive_prescribed_times(self):
        for option, knots, area in (('jump_t', [0.41, 0.4], 0.01),
                                    ('step_t', [0.4375, 0.375, 0.40625], 0.0625 ** 2 / 4)):
            for dtype in DTYPES:
                for method in ADAPTIVE_METHODS:
                    for reverse in (False, True):
                        for explicit in (False, True):
                            with self.subTest(option=option, dtype=dtype, method=method,
                                              reverse=reverse, explicit=explicit):
                                func = PrescribedField(dtype, option)
                                t = torch.tensor([1., 0.] if reverse else [0., 1.],
                                                 dtype=torch.float64, requires_grad=option == 'jump_t')
                                y0 = torch.tensor([0.3, -0.2], dtype=dtype, requires_grad=True)
                                times = torch.tensor(knots, dtype=torch.float64)
                                options = {option: times, 'first_step': 0.17, 'max_step': 0.4}
                                adjoint_options = options.copy() if explicit else None
                                solution = torchdiffeq.odeint_adjoint(
                                    func, y0, t, method=method, rtol=1e-7, atol=1e-9,
                                    options=options, adjoint_options=adjoint_options)
                                inputs = (y0, func.rate, t) if t.requires_grad else (y0, func.rate)
                                gradients = torch.autograd.grad(solution[-1].sum(), inputs)
                                grad_y0, grad_rate = gradients[:2]
                                signed_area = -area if reverse else area
                                self.assertTrue(torch.allclose(solution[-1], y0 + 2 * signed_area,
                                                               rtol=1e-5, atol=2e-7))
                                self.assertTrue(torch.allclose(grad_rate, func.rate.new_tensor(2 * signed_area),
                                                               rtol=1e-4, atol=2e-7))
                                self.assertTrue(torch.allclose(grad_y0, torch.ones_like(y0)))
                                if t.requires_grad:
                                    self.assertTrue(torch.allclose(gradients[2], torch.zeros_like(t), atol=2e-7))
                                self.assertIs(options[option], times)
                                self.assertTrue(torch.equal(times, torch.tensor(knots, dtype=torch.float64)))
                                if explicit:
                                    self.assertIs(adjoint_options[option], times)

    def test_custom_grid_in_original_time_coordinates(self):
        for dtype in DTYPES:
            for reverse in (False, True):
                for explicit in (False, True):
                    for tuple_state in (False, True):
                        with self.subTest(dtype=dtype, reverse=reverse, explicit=explicit,
                                          tuple_state=tuple_state):
                            offset = -2.
                            func = PrescribedField(dtype, 'jump_t', offset)
                            t = torch.tensor([1., 0.] if reverse else [0., 1.], dtype=torch.float64) + offset
                            y0 = torch.tensor([0.3, -0.2], dtype=dtype, requires_grad=True)
                            if tuple_state:
                                y0 = (y0, torch.tensor(1., dtype=dtype, requires_grad=True))
                            knots = torch.tensor([0.4, 0.41], dtype=torch.float64) + offset

                            def grid_constructor(func, initial, times):
                                low, high = torch.min(times), torch.max(times)
                                interior = knots[(knots > low) & (knots < high)]
                                grid = torch.cat((times[[0, -1]], interior)).sort()[0]
                                return grid.flip(0) if times[0] > times[-1] else grid

                            options = {'grid_constructor': grid_constructor, 'perturb': True}
                            adjoint_options = options.copy() if explicit else None
                            solution = torchdiffeq.odeint_adjoint(
                                func, y0, t, method='rk4', options=options, adjoint_options=adjoint_options)
                            initial_parts = y0 if tuple_state else (y0,)
                            parts = solution if tuple_state else (solution,)
                            gradients = torch.autograd.grad(sum(part[-1].sum() for part in parts),
                                                            initial_parts + (func.rate,))
                            signed_area = -0.01 if reverse else 0.01
                            for initial, part, grad in zip(initial_parts, parts, gradients[:-1]):
                                self.assertTrue(torch.allclose(part[-1], initial + 2 * signed_area,
                                                               rtol=1e-5, atol=2e-7))
                                self.assertTrue(torch.allclose(grad, torch.ones_like(initial)))
                            expected = sum(initial.numel() for initial in initial_parts) * signed_area
                            self.assertTrue(torch.allclose(gradients[-1], func.rate.new_tensor(expected),
                                                           rtol=1e-4, atol=2e-7))
                            self.assertIs(options['grid_constructor'], grid_constructor)
                            if explicit:
                                self.assertIs(adjoint_options['grid_constructor'], grid_constructor)


if __name__ == '__main__':
    unittest.main()
