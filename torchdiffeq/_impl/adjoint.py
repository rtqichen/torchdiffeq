import torch
import torch.nn as nn
from .odeint import odeint
from .misc import _check_inputs, _flat_to_shape


class OdeintAdjointMethod(torch.autograd.Function):

    @staticmethod
    def forward(ctx, func, y0, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        with torch.no_grad():
            y = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, y, *adjoint_params)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        func = ctx.func
        adjoint_rtol = ctx.adjoint_rtol
        adjoint_atol = ctx.adjoint_atol
        adjoint_method = ctx.adjoint_method
        adjoint_options = ctx.adjoint_options
        t_requires_grad = ctx.t_requires_grad

        t, y, *adjoint_params = ctx.saved_tensors
        adjoint_params = tuple(adjoint_params)

        # We assume that any grid points are given to us ordered in the same direction as for the forward pass (for
        # compatibility with setting adjoint_options = options), so we need to flip them around here.
        try:
            grid_points = adjoint_options['grid_points']
        except (KeyError, TypeError):
            pass
        else:
            adjoint_options = adjoint_options.copy()
            adjoint_options['grid_points'] = grid_points.flip(0)

        y_numel = y[0].numel()  # [0] because y is of shape (len(t), *y0.shape)

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y = y_aug[1:y_numel + 1].view_as(grad_y[0])  # [0] because grad_y has shape (len(t), *y0.shape)
            adj_y = y_aug[y_numel + 1:2 * y_numel + 1].view_as(grad_y[0])  #
            # ignored gradients wrt time and parameters

            with torch.enable_grad():
                t = t.detach().requires_grad_(True)
                y = y.detach().requires_grad_(True)
                func_eval = func(t, y)

                # Workaround for PyTorch bug #39784
                _t = torch.as_strided(t, (), ())
                _y = torch.as_strided(y, (), ())
                _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)

                vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                    func_eval, (t, y) + adjoint_params, -adj_y,
                    allow_unused=True, retain_graph=True
                )

            # autograd.grad returns None if no gradient, set to zero.
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
            vjp_params = [torch.zeros_like(param) if vjp_param is None else vjp_param
                          for param, vjp_param in zip(adjoint_params, vjp_params)]

            vjp = [vjp_t, func_eval, vjp_y, *vjp_params]
            vjp = torch.cat([x.reshape(-1) for x in vjp])
            return vjp

        with torch.no_grad():
            # Put everything together into a single flat tensor
            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            aug_state = [torch.zeros(1, dtype=t.dtype, device=t.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            aug_state.extend([torch.zeros_like(param) for param in adjoint_params])  # vjp_params
            aug_state = torch.cat([x.reshape(-1) for x in aug_state])

            time_vjps = []
            for i in range(len(t) - 1, 0, -1):
                if t_requires_grad:
                    # Compute the effect of moving the current time measurement point.
                    # We don't compute this unless we need to, to save some computation.
                    func_eval = func(t[i], y[i])
                    dLd_cur_t = func_eval.reshape(-1).dot(grad_y[i].reshape(-1))
                    aug_state[0] -= dLd_cur_t
                    time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                aug_state = odeint(
                    augmented_dynamics, aug_state,
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = aug_state[1]  # extract just the t[i - 1] value
                aug_state[1:y_numel + 1] = y[i - 1].reshape(-1)  # update to use our forward-pass estimate of the state
                aug_state[y_numel + 1:2 * y_numel + 1] += grad_y[i - 1].reshape(-1)  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps.append(aug_state[0])
                time_vjps = torch.stack(time_vjps[::-1])
            else:
                time_vjps = None

            adj_y = aug_state[y_numel + 1:2 * y_numel + 1].view_as(y[0])  # [0] because y has shape (len(t), *y0.shape)
            flat_adj_params = aug_state[2 * y_numel + 1:]
            adj_params = []
            total_numel = 0
            for param in adjoint_params:
                next_total_numel = total_numel + param.numel()
                adj_param = flat_adj_params[total_numel:next_total_numel].view_as(param)
                adj_params.append(adj_param)
                total_numel = next_total_numel

            return (None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, *adj_params)


def odeint_adjoint(func, y0, t, rtol=1e-6, atol=1e-12, method=None, options=None, adjoint_rtol=None, adjoint_atol=None,
                   adjoint_method=None, adjoint_options=None, adjoint_params=None):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if adjoint_params is not None and not isinstance(func, nn.Module):
        raise ValueError('func must be an instance of nn.Module to specify the adjoint parameters; alternatively they '
                         'can be specified explicitly via the `adjoint_params` argument.')

    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method
    if adjoint_options is None:
        adjoint_options = options
    if adjoint_params is None:
        adjoint_params = tuple(func.parameters())

    tensor_input, shapes, func, y0, t, options = _check_inputs(func, y0, t, options)

    solution = OdeintAdjointMethod.apply(func, y0, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol,
                                         adjoint_method, adjoint_options, t.requires_grad, *adjoint_params)

    if not tensor_input:
        solution = _flat_to_shape(solution, (len(t),), shapes)
    return solution
