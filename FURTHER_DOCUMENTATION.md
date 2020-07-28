# Further documentation

#### Solver options

Individual solvers offer different options. Also shown are their default values.

**dopri8, dopri5, bosh3, adaptive_heun:**<br>
- `first_step=None`: What size the first step of the solver should be; by default this is selected empirically.
- `safety=0.9, ifactor=10.0, dfactor=0.2`: How the next optimal step size is calculated, see E. Hairer, S. P. Norsett G. Wanner, *Solving Ordinary Differential Equations I: Nonstiff Problems*, Sec. II.4. Roughly speaking, `safety` will try to shrink the step size slightly by this amount, `ifactor` is the most that the step size can grow by, and `dfactor` is the most that it can shrink by.
- `max_num_steps=2 ** 31 - 1`: The maximum number of steps the solver is allowed to take.
- `grid_points=None`: The locations of any discontinuities or derivative discontinuities in the vector field. Crossing these will make the solver suddenly take a larger error, so it will reject the step and slow down to resolve the discontinuity. Once it has done that, then it must speed back up again. If the locations of these discontinuities is known in advance, then the solver can be told about their locations and make a step exactly to them. If passed this should be an ordered one dimensional `torch.Tensor`.
- `eps=0.`: A small perturbation either side of each value in `grid_points` to evaluate at. If the vector field is discontinuous then this ensures that evaluations are performed the correct size of the discontinuity. If used it usually best to just set this to a small number like `1e-5`.
- `dtype=torch.float64`: what dtype to use for timelike quantities. Setting this to `torch.float32` will improve speed but may produce underflow errors more easily.

**euler, midpoint, rk4:**<br>
- `step_size=None`: How large each discrete step should be. If not passed then this defaults to stepping between the values of `t`. Note that if using `t` just to specify the start and end of the regions of integration, then it is very important to specify this argument! It is mutually exclusive with the `grid_constructor` argument, below.
- `grid_constructor=None`: A more fine-grained way of setting the steps, by setting these particular locations as the locations of the steps. Should be a callable `func, y0, t -> grid`, transforming the arguments `func, y0, t` of `odeint` into the desired grid (which should be a one dimensional tensor).
- `eps=0.`: Analogous to the `eps` argument of the adaptive solvers, this adds a small perturbation to each end of the every step (not just the grid points), to help stay the right side of discontinuities.

**implicit_adams, explicit_adams, adaptive_adams:**<br>
TODO!


 #### Adjoint options
 
 The function `odeint_adjoint` offers some adjoint-specific options.
 - `adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options`: The `rtol, atol, method, options` to use for the backward pass. Defaults to the values used for the forward pass.
 - `adjoint_params`: The parameters to compute gradients with respect to in the backward pass. Should be a tuple of tensors. Defaults to `tuple(func.parameters())`. If passed manually then `func` does not have to be a `torch.nn.Module`.