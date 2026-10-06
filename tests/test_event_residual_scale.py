"""Transverse event sensitivities are invariant under residual scaling."""

import pytest
import torch

from torchdiffeq import odeint, odeint_adjoint, odeint_event


class ConstantVelocity(torch.nn.Module):
    def __init__(self, rate):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(rate, dtype=torch.float64))

    def forward(self, time, state):
        return torch.stack([self.rate, torch.zeros_like(self.rate)])


@pytest.mark.parametrize("interface", [odeint, odeint_adjoint])
@pytest.mark.parametrize("reverse_time", [False, True])
@pytest.mark.parametrize("boundary_velocity", [0., .2])
@pytest.mark.parametrize("residual_scale", [1., -3., 1e-9, -2e-14])
def test_event_gradient_matches_closed_form_for_equivalent_residuals(
    interface, reverse_time, boundary_velocity, residual_scale
):
    initial_values = [2.1, .4] if reverse_time else [.1, 1.4]
    initial = torch.tensor(initial_values, dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor(1.4 if reverse_time else .2, dtype=torch.float64, requires_grad=True)
    field = ConstantVelocity(.9 if reverse_time else 1.3)

    def event(time, state):
        return residual_scale * (state[0] - state[1] - boundary_velocity * time)

    # Solve the linear trajectory / moving-boundary intersection analytically.
    expected_time = (initial[1] - initial[0] + field.rate * t0) / (field.rate - boundary_velocity)
    expected_final = torch.stack([initial[0] + field.rate * (expected_time - t0), initial[1]])
    expected_loss = .6 * expected_time + .7 * expected_final[0] - .3 * expected_final[1]
    parameters = (initial, t0, field.rate)
    expected_gradients = torch.autograd.grad(expected_loss, parameters)

    event_time, solution = odeint_event(
        field, initial, t0, event_fn=event, reverse_time=reverse_time,
        odeint_interface=interface, rtol=1e-10, atol=1e-12,
    )
    torch.testing.assert_close(event_time, expected_time, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(solution[-1], expected_final, rtol=1e-9, atol=1e-9)
    actual_loss = .6 * event_time + .7 * solution[-1, 0] - .3 * solution[-1, 1]
    actual_gradients = torch.autograd.grad(actual_loss, parameters)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected, rtol=2e-7, atol=2e-8)
