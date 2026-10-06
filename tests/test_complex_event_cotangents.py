"""Real event times require Hermitian complex-state cotangent contractions."""

import pytest
import torch

from torchdiffeq import odeint, odeint_event


class ExponentialFlow(torch.nn.Module):
    def __init__(self, rate, dtype):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(rate, dtype=dtype))

    def forward(self, time, state):
        return self.rate * state


def closed_form_intersection(initial, t0, rate, level, boundary_growth):
    # |z(t)|^2 = |z0|^2 exp(2 Re(rate) (t-t0)).
    squared_norm = initial.abs().square().sum()
    event_time = ((level / squared_norm).log() + 2. * rate.real * t0) / (2. * rate.real - boundary_growth)
    state = initial * (rate * (event_time - t0)).exp()
    return event_time, state


def real_loss(time, state, loss_kind):
    if loss_kind == "event_time":
        return .6 * time
    squared_weights = state.real.new_tensor([.7, .2])
    real_weights = state.real.new_tensor([.3, -.4])
    imaginary_weights = state.real.new_tensor([-.2, .5])
    return (.6 * time + (squared_weights * state.abs().square()).sum()
            + (real_weights * state.real).sum() + (imaginary_weights * state.imag).sum())


@pytest.mark.parametrize("dtype", [torch.complex64, torch.complex128])
@pytest.mark.parametrize("reverse_time", [False, True])
@pytest.mark.parametrize("boundary_growth", [0., .2])
@pytest.mark.parametrize("loss_kind", ["event_time", "mixed"])
def test_complex_event_sensitivities_match_closed_form(dtype, reverse_time, boundary_growth, loss_kind):
    real_dtype = torch.float32 if dtype == torch.complex64 else torch.float64
    initial = torch.tensor([.7 + .2j, -.3 + .4j], dtype=dtype, requires_grad=True)
    t0 = torch.tensor(1.4 if reverse_time else .2, dtype=real_dtype, requires_grad=True)
    field = ExponentialFlow(.7 + 1.2j, dtype)
    level = initial.real.new_tensor(.2 if reverse_time else 4.)

    def event(time, state):
        return state.abs().square().sum() - level * (boundary_growth * time).exp()

    expected_time, expected_state = closed_form_intersection(initial, t0, field.rate, level, boundary_growth)
    parameters = (initial, t0, field.rate)
    expected_gradients = torch.autograd.grad(real_loss(expected_time, expected_state, loss_kind), parameters)
    event_time, solution = odeint_event(
        field, initial, t0, event_fn=event, reverse_time=reverse_time, odeint_interface=odeint,
        rtol=1e-5 if real_dtype == torch.float32 else 1e-10,
        atol=1e-7 if real_dtype == torch.float32 else 1e-12,
    )
    value_tolerance = 2e-4 if real_dtype == torch.float32 else 2e-8
    torch.testing.assert_close(event_time, expected_time, rtol=value_tolerance, atol=value_tolerance)
    torch.testing.assert_close(solution[-1], expected_state, rtol=value_tolerance, atol=value_tolerance)
    actual_gradients = torch.autograd.grad(real_loss(event_time, solution[-1], loss_kind), parameters)
    for actual, expected in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected,
                                   rtol=5e-4 if real_dtype == torch.float32 else 4e-7,
                                   atol=3e-5 if real_dtype == torch.float32 else 4e-8)


@pytest.mark.parametrize("reverse_time", [False, True])
@pytest.mark.parametrize("boundary_growth", [0., .2])
def test_real_state_event_contract_is_preserved(reverse_time, boundary_growth):
    initial = torch.tensor([.7, -.3], dtype=torch.float64, requires_grad=True)
    t0 = torch.tensor(1.4 if reverse_time else .2, dtype=torch.float64, requires_grad=True)
    field = ExponentialFlow(.7, torch.float64)
    level = initial.new_tensor(.2 if reverse_time else 4.)

    def event(time, state):
        return state.square().sum() - level * (boundary_growth * time).exp()

    expected_time, expected_state = closed_form_intersection(initial, t0, field.rate, level, boundary_growth)
    parameters = (initial, t0, field.rate)
    expected_loss = .6 * expected_time + expected_state.square().sum() + expected_state.sum()
    expected_gradients = torch.autograd.grad(expected_loss, parameters)
    event_time, solution = odeint_event(
        field, initial, t0, event_fn=event, reverse_time=reverse_time,
        odeint_interface=odeint, rtol=1e-10, atol=1e-12,
    )
    actual_loss = .6 * event_time + solution[-1].square().sum() + solution[-1].sum()
    actual_gradients = torch.autograd.grad(actual_loss, parameters)
    torch.testing.assert_close(event_time, expected_time, rtol=2e-8, atol=2e-8)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=4e-7, atol=4e-8)
