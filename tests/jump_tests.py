import unittest

import torch
import torchdiffeq

from problems import ADAPTIVE_METHODS, DTYPES


class Pulse(torch.nn.Module):
    def __init__(self, dtype, direction):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.tensor(1., dtype=dtype))
        self.direction = direction

    def forward(self, t, y):
        time = self.direction * t
        rate = self.direction * self.gain * ((time > 0.5) & (time < 0.5625))
        if isinstance(y, tuple):
            return tuple(torch.ones_like(part) * rate for part in y)
        return torch.ones_like(y) * rate


class TestPrescribedEndpoints(unittest.TestCase):
    def assert_close(self, actual, expected):
        # Keep these regressions usable with the project's torch>=1.5 floor.
        self.assertTrue(torch.allclose(actual, expected, rtol=1e-5, atol=1e-7),
                        '{} != {}'.format(actual, expected))

    def test_jump_endpoint_pulse(self):
        # The two jumps bound a pulse of area 1/16. The initial step either
        # lands exactly on the first jump or is shortened to reach it.
        for method in ADAPTIVE_METHODS:
            for dtype in DTYPES:
                for direction in (1, -1):
                    for tuple_state in (False, True):
                        for first_step in (0.5, 0.625):
                            with self.subTest(method=method, dtype=dtype, direction=direction,
                                              tuple_state=tuple_state, first_step=first_step):
                                y0 = torch.tensor([0.25, -0.25], dtype=dtype)
                                if tuple_state:
                                    y0 = (y0, torch.tensor(1., dtype=dtype))
                                solution = torchdiffeq.odeint(
                                    Pulse(dtype, direction), y0,
                                    torch.tensor([0., direction], dtype=torch.float64),
                                    method=method, options={
                                        'first_step': first_step, 'max_num_steps': 1000,
                                        'jump_t': direction * torch.tensor([0.5, 0.5625])})
                                parts = solution if tuple_state else (solution,)
                                initial = y0 if tuple_state else (y0,)
                                for part, start in zip(parts, initial):
                                    self.assertEqual(part.dtype, dtype)
                                    self.assertEqual(part.shape, (2,) + start.shape)
                                    self.assert_close(part[0], start)
                                    self.assert_close(part[-1], start + 0.0625)

    def test_step_endpoint_consumes_next_time(self):
        for method in ADAPTIVE_METHODS:
            for direction in (1, -1):
                for first_step in (0.5, 0.625):
                    with self.subTest(method=method, direction=direction, first_step=first_step):
                        endpoints = []

                        def func(t, y):
                            return torch.ones_like(y)

                        func.callback_accept_step = lambda t, y, dt: endpoints.append(float(t + direction * dt))
                        solution = torchdiffeq.odeint(
                            func, torch.zeros(1, dtype=torch.float64),
                            torch.tensor([0., direction], dtype=torch.float64), method=method,
                            options={'first_step': first_step, 'max_num_steps': 1000,
                                     'step_t': direction * torch.tensor([0.5, 0.5625])})
                        self.assertIn(direction * 0.5, endpoints)
                        self.assertIn(direction * 0.5625, endpoints)
                        self.assert_close(solution[-1], torch.tensor([float(direction)], dtype=torch.float64))

    def test_jump_endpoint_gradients(self):
        for dtype in DTYPES:
            for direction in (1, -1):
                for interface in (torchdiffeq.odeint, torchdiffeq.odeint_adjoint):
                    with self.subTest(dtype=dtype, direction=direction, interface=interface.__name__):
                        func = Pulse(dtype, direction)
                        y0 = torch.zeros(1, dtype=dtype, requires_grad=True)
                        kwargs = {}
                        if interface is torchdiffeq.odeint_adjoint:
                            # Resolve the pulse independently in the adjoint solve.
                            kwargs = {'adjoint_method': 'rk4',
                                      'adjoint_options': {'step_size': 0.015625, 'perturb': True}}
                        solution = interface(
                            func, y0, torch.tensor([0., direction], dtype=torch.float64),
                            options={'first_step': 0.5, 'max_num_steps': 1000,
                                     'jump_t': direction * torch.tensor([0.5, 0.5625])}, **kwargs)
                        solution[-1].sum().backward()
                        self.assert_close(y0.grad, torch.ones_like(y0))
                        self.assert_close(func.gain.grad, torch.tensor(0.0625, dtype=dtype))

    def test_jump_endpoint_event_tuple(self):
        for method in ADAPTIVE_METHODS:
            for direction in (1, -1):
                with self.subTest(method=method, direction=direction):
                    y0 = (torch.zeros(2, dtype=torch.float64), torch.tensor(1., dtype=torch.float64))
                    event_time, solution = torchdiffeq.odeint(
                        Pulse(torch.float64, direction), y0,
                        torch.tensor([0., direction], dtype=torch.float64), method=method,
                        event_fn=lambda t, y: direction * t - 0.75,
                        options={'first_step': 0.5, 'max_num_steps': 1000,
                                 'jump_t': direction * torch.tensor([0.5, 0.5625])})
                    self.assertAlmostEqual(float(event_time), direction * 0.75, places=7)
                    for part, start in zip(solution, y0):
                        self.assert_close(part[-1], start + 0.0625)


if __name__ == '__main__':
    unittest.main()
