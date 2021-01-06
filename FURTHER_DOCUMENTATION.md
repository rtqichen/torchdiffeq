# Further documentation

## Solver options

Adaptive and fixed solvers all support several options. Also shown are their default values.

**Adaptive solvers (dopri8, dopri5, bosh3, adaptive_heun):**<br>
For these solvers, `rtol` and `atol` correspond to the tolerances for accepting/rejecting an adaptive step.

- `first_step=None`: What size the first step of the solver should be; by default this is selected empirically.

- `safety=0.9, ifactor=10.0, dfactor=0.2`: How the next optimal step size is calculated, see E. Hairer, S. P. Norsett G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, Sec. II.4. Roughly speaking, `safety` will try to shrink the step size slightly by this amount, `ifactor` is the most that the step size can grow by, and `dfactor` is the most that it can shrink by.

- `max_num_steps=2 ** 31 - 1`: The maximum number of steps the solver is allowed to take.

- `dtype=torch.float64`: what dtype to use for timelike quantities. Setting this to `torch.float32` will improve speed but may produce underflow errors more easily.

- `step_t=None`: Times that a step must me made to. In particular this is useful when `func` has kinks (derivative discontinuities) at these times, as the solver then does not need to (slowly) discover these for itself. If passed this should be a `torch.Tensor`.

- `jump_t=None`: Times that a step must be made to, and `func` re-evaluated at. In particular this is useful when `func` has discontinuites at these times, as then the solver knows that the final function evaluation of the previous step is not equal to the first function evaluation of this step. (i.e. the FSAL property does not hold at this point.) If passed this should be a `torch.Tensor`. Note that this may not be efficient when using PyTorch 1.6.0 or earlier.

- `norm`: What norm to compute the accept/reject criterion with respect to. Given tensor input, this defaults to an RMS norm. Given tupled input, this defaults to computing an RMS norm over each tensor, and then taking a max over the tuple, producing a mixed L-infinity/RMS norm. If passed this should be a function consuming a tensor/tuple with the same shape as `y0`, and return a scalar corresponding to its norm. When passed as part of `adjoint_options`, then the special value `"seminorm"` may be used to zero out the contribution from the parameters, as per the ["Hey, that's not an ODE"](https://arxiv.org/abs/2009.09457) paper.

**Fixed solvers (euler, midpoint, rk4, explicit_adams, implicit_adams):**<br>

- `step_size=None`: How large each discrete step should be. If not passed then this defaults to stepping between the values of `t`. Note that if using `t` just to specify the start and end of the regions of integration, then it is very important to specify this argument! It is mutually exclusive with the `grid_constructor` argument, below.

- `grid_constructor=None`: A more fine-grained way of setting the steps, by setting these particular locations as the locations of the steps. Should be a callable `func, y0, t -> grid`, transforming the arguments `func, y0, t` of `odeint` into the desired grid (which should be a one dimensional tensor).

- `perturb`: Defaults to False. If True, then automatically add small perturbations to the start and end of each step, so that stepping to discontinuities works. Note that this this may not be efficient when using PyTorch 1.6.0 or earlier.

Individual solvers also offer certain options.

**explicit_adams:**<br>
For this solver, `rtol` and `atol` are ignored. This solver also supports:

- `max_order`: The maximum order of the Adams-Bashforth predictor.

**implicit_adams:**<br>
For this solver, `rtol` and `atol` correspond to the tolerance for convergence of the Adams-Moulton corrector. This solver also supports:

- `max_order`: The maximum order of the Adams-Bashforth-Moulton predictor-corrector.

- `max_iters`: The maximum number of iterations to run the Adams-Moulton corrector for.

**scipy_solver:**<br>
- `solver`: which SciPy solver to use; corresponds to the `'method'` argument of `scipy.integrate.solve_ivp`.

 ## Adjoint options

 The function `odeint_adjoint` offers some adjoint-specific options.

 - `adjoint_rtol`,<br>`adjoint_atol`,<br>`adjoint_method`,<br>`adjoint_options`:<br>The `rtol, atol, method, options` to use for the backward pass. Defaults to the values used for the forward pass.

 - `adjoint_options` has the special key-value pair `{"norm": "seminorm"}` that provides a potentially more efficient adjoint solve when using adaptive step solvers, as described in the ["Hey, that's not an ODE"](https://arxiv.org/abs/2009.09457) paper.

 - `adjoint_params`: The parameters to compute gradients with respect to in the backward pass. Should be a tuple of tensors. Defaults to `tuple(func.parameters())`.
   - If passed then `func` does not have to be a `torch.nn.Module`.
   - If `func` has no parameters, `adjoint_params=()` must be specified.
