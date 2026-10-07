"""Zero-Jacobian vector fields still have initial-state and time sensitivities."""
import pytest
import torch

from torchdiffeq import odeint_adjoint


def objective(solution):
    time_weights = solution.new_tensor([0.4, -0.7, 1.1]).unsqueeze(-1)
    state_weights = solution.new_tensor([0.3, -0.6])
    return (time_weights * (solution * state_weights + 0.2 * solution.square())).sum()


def assert_solution_and_gradients(solution, exact, inputs, dtype):
    tolerance = 4e-5 if dtype == torch.float32 else 4e-8
    torch.testing.assert_close(solution, exact, rtol=tolerance, atol=tolerance)
    expected = torch.autograd.grad(objective(exact), inputs, allow_unused=True)
    actual = torch.autograd.grad(objective(solution), inputs)
    for result, reference, value in zip(actual, expected, inputs):
        if reference is None:
            reference = torch.zeros_like(value)
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result, reference, rtol=tolerance, atol=tolerance)


def initial_conditions(dtype, reverse_time):
    initial = torch.tensor([0.8, -0.3], dtype=dtype, requires_grad=True)
    times = torch.tensor([0.2, 0.55, 0.9], dtype=dtype)
    if reverse_time:
        times = times.flip(0)
    return initial, times.requires_grad_(True)


def solver_options(dtype, method):
    return dict(
        method=method,
        options={'step_size': 0.01} if method == 'rk4' else None,
        rtol=1e-6 if dtype == torch.float32 else 1e-10,
        atol=1e-8 if dtype == torch.float32 else 1e-12,
    )


class ConstantField(torch.nn.Module):
    def __init__(self, dtype, tuple_state):
        super().__init__()
        self.register_buffer('velocity', torch.tensor([0.7, -0.4], dtype=dtype))
        self.unused = torch.nn.Parameter(torch.tensor([0.1, -0.2], dtype=dtype))
        self.tuple_state = tuple_state

    def forward(self, time, state):
        if self.tuple_state:
            return self.velocity[:1], self.velocity[1:]
        return self.velocity


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reverse_time', [False, True])
@pytest.mark.parametrize('method', ['dopri5', 'rk4'])
@pytest.mark.parametrize('tuple_state', [False, True])
def test_constant_field_has_zero_jacobian_and_exact_sensitivities(dtype, reverse_time, method, tuple_state):
    initial, times = initial_conditions(dtype, reverse_time)
    field = ConstantField(dtype, tuple_state)
    state = (initial[:1], initial[1:]) if tuple_state else initial
    solution = odeint_adjoint(field, state, times, **solver_options(dtype, method))
    if tuple_state:
        solution = torch.cat(solution, dim=-1)
    exact = initial.unsqueeze(0) + (times - times[0]).unsqueeze(-1) * field.velocity
    assert_solution_and_gradients(solution, exact, (initial, times, field.unused), dtype)


class ExponentialField(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.rate = torch.nn.Parameter(torch.tensor(0.4, dtype=dtype))
        self.unused = torch.nn.Parameter(torch.tensor(0.1, dtype=dtype))

    def forward(self, time, state):
        return self.rate * state


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reverse_time', [False, True])
@pytest.mark.parametrize('method', ['dopri5', 'rk4'])
def test_parameter_dependent_field_retains_ordinary_adjoint_gradients(dtype, reverse_time, method):
    initial, times = initial_conditions(dtype, reverse_time)
    field = ExponentialField(dtype)
    solution = odeint_adjoint(field, initial, times, **solver_options(dtype, method))
    exact = initial.unsqueeze(0) * (field.rate * (times - times[0])).exp().unsqueeze(-1)
    assert_solution_and_gradients(solution, exact, (initial, times, field.rate, field.unused), dtype)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('reverse_time', [False, True])
def test_constant_callable_without_adjoint_parameters(dtype, reverse_time):
    initial, times = initial_conditions(dtype, reverse_time)
    velocity = initial.new_tensor([0.7, -0.4])

    def field(time, state):
        return velocity

    solution = odeint_adjoint(
        field, initial, times, adjoint_params=(), **solver_options(dtype, 'dopri5'),
    )
    exact = initial.unsqueeze(0) + (times - times[0]).unsqueeze(-1) * velocity
    assert_solution_and_gradients(solution, exact, (initial, times), dtype)
