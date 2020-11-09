import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _check_inputs, _flat_to_shape, _all_callback_names


class _AdjointFunc(torch.nn.Module):
    def __init__(self, base_func, t_requires_grad, adjoint_params):
        super(_AdjointFunc, self).__init__()

        self.base_func = base_func
        self.t_requires_grad = t_requires_grad
        self.adjoint_params = adjoint_params

        for callback_name in _all_callback_names:
            try:
                callback_func = getattr(base_func, callback_name + '_adjoint')
            except AttributeError:
                pass
            else:
                setattr(self, callback_name, callback_func)

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

    shapes, func, y0, t, rtol, atol, method, options, is_reversed = _check_inputs(func, y0, t, rtol, atol, method,
                                                                                  options, SOLVERS)
    solver = SOLVERS[method](rtol=rtol, atol=atol, state_dtype=y0.dtype, device=y0.device, shapes=shapes,
                             is_reversed=is_reversed, **options)

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

    if adjoint_params is None:
        adjoint_params = tuple(find_parameters(func))
    else:
        adjoint_params = tuple(adjoint_params)
    adjoint_params = tuple(p for p in adjoint_params if p.requires_grad)

    if adjoint_options is None:
        adjoint_options = SOLVERS[adjoint_method].adjoint_options_from_options(shapes=shapes,
                                                                               y0=y0,
                                                                               options=options,
                                                                               adjoint_params=adjoint_params)
    else:
        adjoint_options = SOLVERS[adjoint_method].adjoint_options_from_adjoint_options(shapes=shapes,
                                                                                       y0=y0,
                                                                                       options=options,
                                                                                       adjoint_options=adjoint_options,
                                                                                       adjoint_params=adjoint_params)

    y = _AdjointIntegrate.apply(func, y0, t, solver, adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options,
                                t.requires_grad, *adjoint_params)

    if shapes is not None:
        y = _flat_to_shape(y, (len(t),), shapes)
    return y


def find_parameters(module):

    assert isinstance(module, nn.Module)

    if getattr(module, '_is_replica', False):

        def find_tensor_attributes(module):
            tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v) and v.requires_grad]
            return tuples

        gen = module._named_members(get_members_fn=find_tensor_attributes)
        return [param for _, param in gen]
    else:
        return list(module.parameters())
