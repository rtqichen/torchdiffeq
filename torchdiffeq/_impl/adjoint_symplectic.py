import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _mixed_linf_rms_norm


class OdeintAdjointMethodSymplectic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
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
        with torch.no_grad():
            shapes = ctx.shapes
            func = ctx.func
            adjoint_rtol = ctx.adjoint_rtol
            adjoint_atol = ctx.adjoint_atol
            adjoint_method = ctx.adjoint_method
            adjoint_options = ctx.adjoint_options
            t_requires_grad = ctx.t_requires_grad

            t, y, *adjoint_params = ctx.saved_tensors
            adjoint_params = tuple(adjoint_params)

            ##################################
            #     Set up adjoint_options     #
            ##################################

            if adjoint_options is None:
                adjoint_options = {}
            else:
                adjoint_options = adjoint_options.copy()

            # We assume that any grid points are given to us ordered in the same direction as for the forward pass (for
            # compatibility with setting adjoint_options = options), so we need to flip them around here.
            try:
                grid_points = adjoint_options['grid_points']
            except KeyError:
                pass
            else:
                adjoint_options['grid_points'] = grid_points.flip(0)

            # Backward compatibility: by default use a mixed L-infinity/RMS norm over the input, where we treat t, each
            # element of y, and each element of adj_y separately over the Linf, but consider all the parameters
            # together.
            if 'norm' not in adjoint_options:
                if shapes is None:
                    shapes = [y[-1].shape]  # [-1] because y has shape (len(t), *y0.shape)
                # adj_t, y, adj_y, adj_params, corresponding to the order in aug_state below
                adjoint_shapes = [torch.Size(())] + shapes + shapes + [torch.Size([sum(param.numel()
                                                                                       for param in adjoint_params)])]
                adjoint_options['norm'] = _mixed_linf_rms_norm(adjoint_shapes)
            # ~Backward compatibility

            ##################################
            #      Set up initial state      #
            ##################################

            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            # Symplectic integarator considers the input as (q, p)
            aug_state = [torch.zeros(2, dtype=y.dtype, device=y.device), y[-1], grad_y[-1]]  # vjp_t, y, vjp_y
            aug_state.extend([torch.zeros(2*len(param), dtype=y.dtype, device=y.device) \
                              for param in adjoint_params])  # vjp_params

            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
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
                    func_eval = func(t if t_requires_grad else t_, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())
                    _y = torch.as_strided(y, (), ())
                    _params = tuple(torch.as_strided(param, (), ()) for param in adjoint_params)

                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros(2*len(t.reshape(-1)), dtype=y.dtype, device=y.device) \
                                    if vjp_t is None else \
                                        torch.cat([vjp_t, torch.zeros(len(t))])

                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y

                vjp_params = [torch.zeros(2*len(param), dtype=y.dtype, device=y.device) 
                              if vjp_param is None \
                              else torch.cat([vjp_param,torch.zeros(len(vjp_param))]) \
                              for param, vjp_param in zip(adjoint_params, vjp_params)]

                return (vjp_t, func_eval, vjp_y, *vjp_params)
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
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                aug_state[2] += grad_y[i - 1]  # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0], = torch.chunk(aug_state[0],2)

            adj_y = aug_state[2]
            adj_params = [param[:len(param)//2] for param in aug_state[3:]]

        return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, *adj_params)


