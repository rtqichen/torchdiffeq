# Frequently Asked Questions (FAQ)

**What are good resources to understand how ODEs can be solved?**<br>
*Solving Ordinary Differential Equations I Nonstiff Problems* by Hairer et al.<br>
[ODE solver selection in MatLab](https://blogs.mathworks.com/loren/2015/09/23/ode-solver-selection-in-matlab/)<br>

**What are the ODE solvers available in this repo?**<br>

- Adaptive-step:
  - `dopri8` Runge-Kutta 7(8) of Dormand-Prince-Shampine
  - `dopri5` Runge-Kutta 4(5) of Dormand-Prince **[default]**.
  - `bosh3` Runge-Kutta 2(3) of Bogacki-Shampine
  - `adaptive_heun` Runge-Kutta 1(2)

- Fixed-step:
  - `euler` Euler method.
  - `midpoint` Midpoint method.
  - `rk4` Fourth-order Runge-Kutta with 3/8 rule.
  - `explicit_adams` Explicit Adams.
  - `implicit_adams` Implicit Adams.


**What are `NFE-F` and `NFE-B`?**<br>
Number of function evaluations for forward and backward pass.

**What are `rtol` and `atol`?**<br>
They refer to relative `rtol` and absolute `atol` error tolerance.

**What is the role of error tolerance in adaptive solvers?**<br>
The basic idea is each adaptive solver can produce an error estimate of the current step, and if the error is greater than some tolerance, then the step is redone with a smaller step size, and this repeats until the error is smaller than the provided tolerance.<br>
[Error Tolerances for Variable-Step Solvers](https://www.mathworks.com/help/simulink/ug/types-of-solvers.html#f11-44943)

**How is the error tolerance calculated?**<br>
The error tolerance is [calculated]((https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/misc.py#L74)) as `atol + rtol * norm of current state`, where the norm being used is a mixed L-infinity/RMS norm. 

**Where is the code that computes the error tolerance?**<br>
It is computed [here.](https://github.com/rtqichen/torchdiffeq/blob/c4c9c61c939c630b9b88267aa56ddaaec319cb16/torchdiffeq/_impl/misc.py#L94)

**How many states must a Neural ODE solver store during a forward pass with the adjoint method?**<br>
The number of states required to be stored in memory during a forward pass is solver dependent. For example, `dopri5` requires 6 intermediate states to be stored.

**How many function evaluations are there per ODE step on adaptive solvers?**<br>

- `dopri5`<br>
	The `dopri5` ODE solver stores at least 6 evaluations of the ODE, then takes a step using a linear combination of them. The diagram below illustrates it: the evaluations marked with `o` are on the estimated path, the others with `x` are not. The first two are for selecting the initial step size.

    ```
	0  1 |  2  3  4  5  6  7 |  8  9  10 12 13 14
	o  x |  x  x  x  x  x  o |  x  x  x  x  x  o
    ```


**How do I obtain evaluations on the estimated path when using an adaptive solver?**<br>
The argument `t` of `odeint` specifies what times should the ODE solver output.<br>
```odeint(func, x0, t=torch.linspace(0, 1, 50))```

Note that the ODE solver will always integrate from `min t(0)` to `max t(1)`, and the intermediate values of `t` have no effect on how the ODE the solved. Intermediate values are computed using polynomial interpolation and have very small cost.

**What non-linearities should I use in my Neural ODE?**<br>
Avoid non-smooth non-linearities such as ReLU and LeakyReLU.<br>
Prefer non-linearities with a theoretically unique adjoint/gradient such as Softplus.

**Where is backpropagation for the Neural ODE defined?**<br>
It's defined [here](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py) if you use the adjoint method `odeint_adjoint`.

**What are Tableaus?**<br>
Tableaus are ways to describe coefficients for [RK methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods). The particular set of coefficients used on this repo was taken from [here](https://www.ams.org/journals/mcom/1986-46-173/S0025-5718-1986-0815836-3/).

**How do I install the repo on Windows?**<br>
Try downloading the code directly and just running python setup.py install.
https://stackoverflow.com/questions/52528955/installing-a-python-module-from-github-in-windows-10

**What is the most memory-expensive operation during training?**<br>
The most memory-expensive operation is the single [backward call](https://github.com/rtqichen/torchdiffeq/blob/master/torchdiffeq/_impl/adjoint.py#L75) made to the network.
    
**My Neural ODE's numerical solution is farther away from the target than the initial value**<br>
Most tricks for initializing residual nets (like zeroing the weights of the last layer) should help for ODEs as well. This will initialize the ODE as an identity.


**My Neural ODE takes too long to train**<br>
This might be because you're running on CPU. Being extremely slow on CPU is expected, as training requires evaluating a neural net multiple times.


**My Neural ODE produces underflow in dt when using adaptive solvers like `dopri5`**<br>
This is a problem of the ODE becoming stiff, essentially acting too erratic in a region and the step size becomes so close to zero that no progress can be made in the solver. We were able to avoid this with regularization such as weight decay and using "nice" activation functions, but YMMV. Other potential options are just to accept a larger error by increasing `atol`, `rtol`, or by switching to a fixed solver.
