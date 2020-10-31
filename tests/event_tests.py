import unittest
import torch
import torchdiffeq

from problems import construct_problem, DTYPES, DEVICES, METHODS, FIXED_METHODS


def rel_error(true, estimate):
    return ((true - estimate) / true).abs().max()


class TestEventHandling(unittest.TestCase):

    def test_odeint(self):
        for reverse in (False, True):
            for dtype in DTYPES:
                for device in DEVICES:
                    for method in METHODS:

                        # TODO: remove after event handling gets enabled.
                        if method == 'scipy_solver':
                            continue

                        for ode in ("constant", "sine"):
                            with self.subTest(reverse=reverse, dtype=dtype, device=device, ode=ode, method=method):
                                if method == "explicit_adams":
                                    tol = 7e-2
                                elif method == "euler":
                                    tol = 5e-3
                                else:
                                    tol = 1e-4

                                f, y0, t_points, sol = construct_problem(dtype=dtype, device=device, ode=ode,
                                                                         reverse=reverse)

                                def event_fn(t, y):
                                    return torch.sum(y - sol[2])

                                if method in FIXED_METHODS:
                                    options = {"step_size": 0.01, "interp": "cubic"}
                                else:
                                    options = {}

                                t, y = torchdiffeq.odeint(f, y0, t_points[0:2], event_fn=event_fn, method=method, options=options)
                                y = y[-1]
                                self.assertLess(rel_error(sol[2], y), tol)
                                self.assertLess(rel_error(t_points[2], t), tol)

    def test_adjoint(self):
        f, y0, t_points, sol = construct_problem(device="cpu", ode="constant")

        def event_fn(t, y):
            return torch.sum(y - sol[-1])

        t, y = torchdiffeq.odeint_adjoint(f, y0, t_points[0:2], event_fn=event_fn, method="dopri5")
        y = y[-1]
        self.assertLess(rel_error(sol[-1], y), 1e-4)
        self.assertLess(rel_error(t_points[-1], t), 1e-4)

        # Make sure adjoint mode backward code can still be run.
        t.backward(retain_graph=True)
        y.sum().backward()


if __name__ == '__main__':
    unittest.main()
