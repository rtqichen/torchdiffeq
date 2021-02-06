# PyTorch Implementation of Differentiable ODE Solvers

This library provides ordinary differential equation (ODE) solvers implemented in PyTorch. Backpropagation through all solvers is supported using the adjoint method. For usage of ODE solvers in deep learning applications, see [1].

As the solvers are implemented in PyTorch, algorithms in this repository are fully supported to run on the GPU.

## Installation

To install latest stable version:
```
pip install torchdiffeq
```

To install latest on GitHub:
```
pip install git+https://github.com/rtqichen/torchdiffeq
```

## Examples
Examples are placed in the [`examples`](./examples) directory.

We encourage those who are interested in using this library to take a look at [`examples/ode_demo.py`](./examples/ode_demo.py) for understanding how to use `torchdiffeq` to fit a simple spiral ODE.

<p align="center">
<img align="middle" src="./assets/ode_demo.gif" alt="ODE Demo" width="500" height="250" />
</p>

## Basic usage
This library provides one main interface `odeint` which contains general-purpose algorithms for solving initial value problems (IVP), with gradients implemented for all main arguments. An initial value problem consists of an ODE and an initial value,
```
dy/dt = f(t, y)    y(t_0) = y_0.
```
The goal of an ODE solver is to find a continuous trajectory satisfying the ODE that passes through the initial condition.

To solve an IVP using the default solver:
```
from torchdiffeq import odeint

odeint(func, y0, t)
```
where `func` is any callable implementing the ordinary differential equation `f(t, x)`, `y0` is an _any_-D Tensor representing the initial values, and `t` is a 1-D Tensor containing the evaluation points. The initial time is taken to be `t[0]`.

Backpropagation through `odeint` goes through the internals of the solver. Note that this is not numerically stable for all solvers (but should probably be fine with the default `dopri5` method). Instead, we encourage the use of the adjoint method explained in [1], which will allow solving with as many steps as necessary due to O(1) memory usage.

To use the adjoint method:
```
from torchdiffeq import odeint_adjoint as odeint

odeint(func, y0, t)
```
`odeint_adjoint` simply wraps around `odeint`, but will use only O(1) memory in exchange for solving an adjoint ODE in the backward call.

The biggest **gotcha** is that `func` must be a `nn.Module` when using the adjoint method. This is used to collect parameters of the differential equation.

## Differentiable event handling

We allow terminating an ODE solution based on an event function. This can be invoked with `odeint_event`:
```
from torchdiffeq import odeint_event
odeint_event(func, y0, t0, *, event_fn, reverse_time=False, odeint_interface=odeint, **kwargs)
```
 - `func` and `y0` are the same as `odeint`.
 - `t0` is a scalar representing the initial time value.
 - `event_fn(t, y)` returns a tensor, and is a required keyword argument.
 - `reverse_time` is a boolean specifying whether we should solve in reverse time. Default is `False`.
 - `odeint_interface` is one of `odeint` or `odeint_adjoint`, specifying whether adjoint mode should be used for differentiating through the ODE solution. Default is `odeint`.
 - `**kwargs`: any remaining keyword arguments are passed to `odeint_interface`.

The solve is terminated at an event time `t` and state `y` when an element of `event_fn(t, y)` is equal to zero. Multiple outputs from `event_fn` can be used to specify multiple event functions, of which the first to trigger will terminate the solve.

Both the event time and final state are returned from `odeint_event`, and can be differentiated. Gradients will be backpropagated through the event function.

The numerical precision for the event time is determined by the `atol` argument.

See example of simulating and differentiating through a bouncing ball in [`examples/bouncing_ball.py`](./examples/bouncing_ball.py).

<p align="center">
<img align="middle" src="./assets/bouncing_ball.png" alt="Bouncing Ball" width="500" height="250" />
</p>

## Keyword arguments for odeint(_adjoint)

#### Keyword arguments:
 - `rtol` Relative tolerance.
 - `atol` Absolute tolerance.
 - `method` One of the solvers listed below.
 - `options` A dictionary of solver-specific options, see the [further documentation](FURTHER_DOCUMENTATION.md).

#### List of ODE Solvers:

Adaptive-step:
 - `dopri8` Runge-Kutta of order 8 of Dormand-Prince-Shampine.
 - `dopri5` Runge-Kutta of order 5 of Dormand-Prince-Shampine **[default]**.
 - `bosh3` Runge-Kutta of order 3 of Bogacki-Shampine.
 - `fehlberg2` Runge-Kutta-Fehlberg of order 2.
 - `adaptive_heun` Runge-Kutta of order 2.

Fixed-step:
 - `euler` Euler method.
 - `midpoint` Midpoint method.
 - `rk4` Fourth-order Runge-Kutta with 3/8 rule.
 - `explicit_adams` Explicit Adams-Bashforth.
 - `implicit_adams` Implicit Adams-Bashforth-Moulton.

Additionally, all solvers available through SciPy are wrapped for use with `scipy_solver`.

For most problems, good choices are the default `dopri5`, or to use `rk4` with `options=dict(step_size=...)` set appropriately small. Adjusting the tolerances (adaptive solvers) or step size (fixed solvers), will allow for trade-offs between speed and accuracy.

## Frequently Asked Questions
Take a look at our [FAQ](FAQ.md) for frequently asked questions.

## Further documentation
For details of the adjoint-specific and solver-specific options, check out the [further documentation](FURTHER_DOCUMENTATION.md).

## References
[1] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." *Advances in Neural Information Processing Systems.* 2018. [[arxiv]](https://arxiv.org/abs/1806.07366)

---

If you found this library useful in your research, please consider citing
```
@article{chen2018neural,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  journal={Advances in Neural Information Processing Systems},
  year={2018}
}
```
