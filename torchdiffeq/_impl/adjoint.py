import torch
import torch.nn as nn
import warnings
from .odeint import SOLVERS, odeint, _check_inputs
from .misc import (_flat_to_shape, _all_callback_names, _all_adjoint_callback_names, _rms_norm, _mixed_norm, _zero_norm)
from .rk_common import RKAdaptiveStepsizeODESolver


def _convert_name(name):
    assert name.endswith('_adjoint')
    name = name[:-len('_adjoint')]
    return name


class _AdjointFunc(torch.nn.Module):
    def __init__(self, base_func, t_requires_grad, adjoint_params):
        super(_AdjointFunc, self).__init__()

        self.base_func = base_func
        self.t_requires_grad = t_requires_grad
        self.adjoint_params = adjoint_params

        for callback_name, adjoint_callback_name in zip(_all_callback_names, _all_adjoint_callback_names):
            try:
                callback = getattr(base_func, adjoint_callback_name)
            except AttributeError:
                pass
            else:
                setattr(self, callback_name, callback)

    def forward(self, t, y_aug):
        # Dynamics of the original system augmented with
        # the adjoint wrt y, and an integrator wrt t and args.
        y = y_aug[1]
        adj_y = y_aug[2]
        # ignore gradients wrt time and parameters

        with torch.enable_grad():
            t_ = t.detach()
            t = t_.requires_grad_(True)
            y = y.detach().requires_grad_(True)

            # If using an adaptive solver we don't want to waste time resolving dL/dt unless we need it (which
            # doesn't necessarily even exist if there is piecewise structure in time), so turning off gradients
            # wrt t here means we won't compute that if we don't need it.
            func_eval = self.base_func(t if self.t_requires_grad else t_, y)

            # Workaround for PyTorch bug #39784
            _t = torch.as_strided(t, (), ())
            _y = torch.as_strided(y, (), ())
            _params = tuple(torch.as_strided(param, (), ()) for param in self.adjoint_params)

            vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                func_eval, (t, y) + self.adjoint_params, -adj_y,
                allow_unused=True, retain_graph=True
            )

        # autograd.grad returns None if no gradient, set to zero.
        vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
        vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
        vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                      for param, vjp_param in zip(self.adjoint_params, vjp_params)]

        return (vjp_t, func_eval, vjp_y, *vjp_params)


class _AdjointIntegrate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, y0, t, solver, adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options, t_requires_grad,
                *adjoint_params):

        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        # .detach() so that the tensors y0 and t don't register as requiring gradients when that's checked, e.g. in the
        # SciPy solvers.
        y = solver.integrate(func, y0.detach(), t.detach())
        ctx.save_for_backward(t, y, *adjoint_params)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        with torch.no_grad():
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            t, y, *adjoint_params = ctx.saved_tensors
            adjoint_params = tuple(adjoint_params)

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            aug_state = [torch.zeros((), dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params

            ##################################
            #    Set up backward ODE func    #
            ##################################

            augmented_dynamics = _AdjointFunc(func, t_requires_grad, adjoint_params)

            ##################################
            #       Solve adjoint ODE        #
            ##################################

            if t_requires_grad:
                time_vjps = torch.empty(len(t), dtype=t.dtype, device=t.device)
            else:
                time_vjps = None
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.
                # TODO: Call odeint_adjoint to implement higher order derivatives. (Or better still, this Function.)
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_state[0]

            adj_y = aug_state[2]
            adj_params = aug_state[3:]

        return (None, adj_y, time_vjps, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, rtol=1e-7, atol=1e-9, method=None, options=None, adjoint_rtol=None, adjoint_atol=None,
                   adjoint_method=None, adjoint_options=None, adjoint_params=None):

    shapes, func, y0, t, rtol, atol, method, options = _check_inputs(func, y0, t, rtol, atol, method, options)
    solver = SOLVERS[method](rtol=rtol, atol=atol, state_dtype=y0.dtype, device=y0.device, **options)

    (adjoint_rtol, adjoint_atol,
     adjoint_method, adjoint_options, adjoint_params) = _check_adjoint_inputs(shapes, func, rtol, atol, method,
                                                                              options, adjoint_rtol, adjoint_atol,
                                                                              adjoint_method, adjoint_options,
                                                                              adjoint_params)

    y = _AdjointIntegrate.apply(func, y0, t, solver, adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options,
                                t.requires_grad, *adjoint_params)

    if shapes is not None:
        y = _flat_to_shape(y, (len(t),), shapes)
    return y


def _check_adjoint_inputs(shapes, func, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                          adjoint_options, adjoint_params):
    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of `nn.Module` to specify the adjoint parameters; alternatively '
                         'they can be specified explicitly via the `adjoint_params` argument. If there are no '
                         'parameters then it is allowable to set `adjoint_params=()`.')

    # Set up adjoint defaults based on forward settings.

    if adjoint_rtol is None:
        try:
            iter(rtol)
        except TypeError:
            adjoint_rtol = rtol
        else:
            raise ValueError("`adjoint_rtol` cannot be inferred from `rtol` when `rtol` is an iterable.")
    if adjoint_atol is None:
        try:
            iter(atol)
        except TypeError:
            adjoint_atol = atol
        else:
            raise ValueError("`adjoint_atol` cannot be inferred from `atol` when `atol` is an iterable.")

    if adjoint_method is None:
        adjoint_method = method

    if adjoint_method != method and options is not None and adjoint_options is None:
        raise ValueError("If `adjoint_method != method` then we cannot infer `adjoint_options` from `options`. So as "
                         "`options` has been passed then `adjoint_options` must be passed as well.")

    if adjoint_options is None:
        made_adjoint_options = True
        adjoint_options = options.copy()
    else:
        made_adjoint_options = False
        adjoint_options = adjoint_options.copy()

    if adjoint_params is None:
        adjoint_params = _find_parameters(func)
    else:
        if isinstance(adjoint_params, torch.Tensor):
            raise ValueError("`adjoint_params` should not be a torch.Tensor; it should be a tuple of torch.Tensors.")
    adjoint_params_ = []
    for p in adjoint_params:
        if p.requires_grad:
            adjoint_params_.append(p)
        else:
            if 'norm' in adjoint_options and adjoint_options['norm'] != 'seminorm':
                warnings.warn("An adjoint parameter was passed without requiring gradient. For efficiency this will be "
                              "excluded from the adjoint pass, and will not appear as a tensor in the adjoint norm.")
    adjoint_params = adjoint_params_

    # Flip timelike options
    try:
        _grid_constructor = adjoint_options['grid_constructor']
    except KeyError:
        pass
    else:
        adjoint_options['grid_constructor'] = lambda func, y0, t: _grid_constructor(func, y0, t).flip(0)
    _flip_option(adjoint_options, 'step_t')
    _flip_option(adjoint_options, 'jump_t')
    _flip_option(adjoint_options, 'grid_points')

    # Handle the 'norm' argument. This is a small can of worms, because it can break the abstraction about whether the
    # (forward) state is a tensor or a tuple of tensors.
    if issubclass(SOLVERS[adjoint_method], RKAdaptiveStepsizeODESolver):
        # Get the norm used in the forward pass.
        # (Which will always have been set, by _check_inputs)
        state_norm = options['norm']

        # This is the default adjoint norm on the backward pass: a mixed norm over the tuple of inputs.
        def default_adjoint_norm(tensor_tuple):
            t, y, adj_y, *adj_params = tensor_tuple
            # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
            return max(t.abs(), state_norm(y), state_norm(adj_y), _mixed_norm(adj_params))

        if made_adjoint_options:
            # `adjoint_options` was not explicitly specified by the user. Use the default norm.
            adjoint_options["norm"] = default_adjoint_norm
        else:
            # `adjoint_options` was explicitly specified by the user...
            try:
                adjoint_norm = adjoint_options['norm']
            except KeyError:
                # ...but they did not specify the norm argument. Back to plan A: use the default norm.
                adjoint_options['norm'] = default_adjoint_norm
            else:
                # ...and they did specify the norm argument.
                if adjoint_norm == 'seminorm':
                    # They told us they want to use seminorms. Slight modification to plan A: use the default norm,
                    # but ignore the parameter state
                    def adjoint_seminorm(tensor_tuple):
                        t, y, adj_y, *adj_params = tensor_tuple
                        # (If the state is actually a flattened tuple then this will be unpacked again in state_norm.)
                        return max(t.abs(), state_norm(y), state_norm(adj_y))
                    adjoint_options['norm'] = adjoint_seminorm
                else:
                    # And they're using their own custom norm.
                    if shapes is None:
                        # The state on the forward pass was a tensor, not a tuple. We don't need to do anything, they're
                        # already going to get given the full adjoint state as (t, y, adj_y, adj_params)
                        pass  # this branch included for clarity
                    else:
                        # This is the bit that is tuple/tensor abstraction-breaking, because the odeint machinery
                        # doesn't know about the tupled nature of the forward state. We need to tell the user's adjoint
                        # norm about that ourselves.

                        def _adjoint_norm(tensor_tuple):
                            t, y, adj_y, *adj_params = tensor_tuple
                            y = _flat_to_shape(y, (), shapes)
                            adj_y = _flat_to_shape(adj_y, (), shapes)
                            return adjoint_norm((t, *y, *adj_y, *adj_params))
                        adjoint_options['norm'] = _adjoint_norm

    return adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options, adjoint_params


def _flip_option(adjoint_options, option_name):
    # We assume that step_t etc. are given to us ordered in the same direction as for the forward pass (for
    # compatibility with the default adjoint_options=options), so we need to flip them around here.
    try:
        option_value = adjoint_options[option_name]
    except KeyError:
        pass
    else:
        if torch.is_tensor(option_value) and option_value.ndimension() > 0:
            adjoint_options[option_name] = option_value.flip(0)
        # else: an error will be raised when the option is attempted to be used in Solver.__init__, but we defer raising
        # the error until then to keep things tidy.


def _find_parameters(module):

    assert isinstance(module, nn.Module)

    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return (param for _, param in gen)
    else:
        return module.parameters()
