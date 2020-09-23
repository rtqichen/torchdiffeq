import torch
import torch.nn as nn
from .odeint import SOLVERS, odeint
from .misc import _mixed_linf_rms_norm, _flat_to_shape


class OdeintAdjointMethodSymplectic(torch.autograd.Function):

    @staticmethod
    def forward(ctx, shapes, func, y0, t, rtol, atol, method, options, 
                            adjoint_rtol, adjoint_atol, adjoint_method,
                adjoint_options, t_requires_grad, *adjoint_params):

        ctx.shapes = shapes
        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.t_requires_grad = t_requires_grad

        with torch.no_grad():
            y = odeint(func, y0, t, rtol=rtol, atol=atol, 
                                        method=method, options=options)
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
            num_particles = len(y[-1]) // 2

            param_shapes = [param_.shape for param_ in adjoint_params]
            adj_params_len = sum(shape.numel() for shape in param_shapes)
            adjoint_params = tuple(adjoint_params)

            ##################################
            #     Set up adjoint_options     #
            ##################################

            if adjoint_options is None:
                adjoint_options = {}
            else:
                adjoint_options = adjoint_options.copy()

            # We assume that any grid points are given to us ordered 
            # in the same direction as for the forward pass (for
            # compatibility with setting adjoint_options = options), 
            # so we need to flip them around here.
            try:
                grid_points = adjoint_options['grid_points']
            except KeyError:
                pass
            else:
                adjoint_options['grid_points'] = grid_points.flip(0)

            # Backward compatibility: by default use a mixed 
            # L-infinity/RMS norm over the input, where we treat t, each
            # element of y, and each element of adj_y 
            # separately over the Linf, but consider all the parameters
            # together.
            if 'norm' not in adjoint_options:
                if shapes is None:
                    # [-1] because y has shape (len(t), *y0.shape)
                    shapes = [y[-1].shape]  
                # adj_t, y, adj_y, adj_params, 
                # corresponding to the order in aug_state below
                adjoint_shapes = [torch.Size(2)]  \
                                    + shapes + shapes \
                                    + [torch.Size(adj_params_len)] \

                adjoint_options['norm'] = _mixed_linf_rms_norm(adjoint_shapes)
            # ~Backward compatibility

            ##################################
            #      Set up initial state      #
            ##################################

            # Symplectic integarator considers the input as (q, p)
            # In symplectic integrator, 2-dim must be required.
            # Thus, for the 1-dim dynamics, e.g. param derivateive and 
            # t derivative, added auxiliary variable playing a role 
            # of generalized coodinate.
            # In other wards, 
            # 
            # q_theta := added auxiiliary q, p_theta := dL/d_theta
            #       
            #   dq_theta/dt = p_theta,
            #   dp_theta/dt = vjp_params.
            #
            # Furthermore, for the correspondence, the following 
            # coordinates are introduced.
            #
            # q_grad := grad_p, p_grad := grad_q
            #
            # Thus, augmented state should be 
            # ([q_t,p_t],[q,p],[p_grad, q_grad],[q_theta,p_theta])

            adj_y_init = torch.flip(grad_y[-1],dims=[0])
            adj_y_init[num_particles:] = - adj_y_init[num_particles:]

            aug_state = \
                [torch.zeros(2, dtype=y.dtype, device=y.device), y[-1], adj_y_init]  
            # [-1] because y and grad_y are both of shape (len(t), *y0.shape)
            #  for the correspondence, grad_y should be flip.

            aug_state.extend([torch.zeros(2*adj_params_len,
                                          dtype=y.dtype,
                                          device=y.device)])
            ##################################
            #    Set up backward ODE func    #
            ##################################

            # TODO: use a nn.Module and call odeint_adjoint 
            # to implement higher order derivatives.
            def augmented_dynamics(t, y_aug):
                # Dynamics of the original system augmented with
                # the adjoint wrt y, and an integrator wrt t and args.
                y = y_aug[1]
                adj_y = torch.flip(y_aug[2],dims=[0])
                adj_y[:num_particles] = - adj_y[:num_particles]

                with torch.enable_grad():
                    t_ = t.detach()
                    t = t_.requires_grad_(True)
                    y = y.detach().requires_grad_(True)

                    # If using an adaptive solver we don't want to 
                    # waste time resolving dL/dt unless we need it (which
                    # doesn't necessarily even exist if there is 
                    # piecewise structure in time), so turning off gradients
                    # wrt t here means we won't 
                    # compute that if we don't need it.
                    func_eval = func(t if t_requires_grad else t_, y)

                    # Workaround for PyTorch bug #39784
                    _t = torch.as_strided(t, (), ())
                    _y = torch.as_strided(y, (), ())
                    _params = tuple(torch.as_strided(param, (), ()) \
                                            for param in adjoint_params)

                    vjp_t, vjp_y, *vjp_params = torch.autograd.grad(
                        func_eval, (t, y) + adjoint_params, -adj_y,
                        allow_unused=True, retain_graph=True
                    )

                # autograd.grad returns None if no gradient, set to zero.
                vjp_t = torch.zeros_like(t).reshape(-1) if vjp_t is None else vjp_t
                vjp_y = torch.zeros_like(y) if vjp_y is None else vjp_y
                vjp_params = tuple([torch.zeros_like(param) if vjp_param is None else vjp_param
                              for param, vjp_param in zip(adjoint_params, vjp_params)])

                p_t = y_aug[0][1:]
                vjp_t = torch.cat([p_t,vjp_t])

                vjp_y = torch.flip(vjp_y, dims=[0])
                vjp_y[num_particles:] = - vjp_y[num_particles:]

                p_theta = y_aug[3][adj_params_len:]
                vjp_params = torch.cat([vjp_para_.reshape(-1) for vjp_para_ in vjp_params])
                vjp_params = torch.cat([p_theta, vjp_params])

                return (vjp_t, func_eval, vjp_y, vjp_params)
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
                    aug_state[0][1:] -= dLd_cur_t
                    time_vjps[i] = dLd_cur_t

                # Run the augmented system backwards in time.
                aug_state = odeint(
                    augmented_dynamics, tuple(aug_state),
                    t[i - 1:i + 1].flip(0),
                    rtol=adjoint_rtol, atol=adjoint_atol, method=adjoint_method, options=adjoint_options
                )
                aug_state = [a[1] for a in aug_state]  # extract just the t[i - 1] value
                aug_state[1] = y[i - 1]  # update to use our forward-pass estimate of the state
                adj_y_t = torch.flip(grad_y[i - 1],dims=[0])
                adj_y_t[num_particles:] = - adj_y_t[num_particles:]
                aug_state[2] += adj_y_t 
                # update any gradients wrt state at this time point

            if t_requires_grad:
                time_vjps[0] = aug_state[0][1:]

            adj_y = aug_state[2]
            adj_y = torch.flip(adj_y,dims=[0])
            adj_y[:num_particles] = - adj_y[:num_particles]

            adj_params = aug_state[3][adj_params_len:]
            adj_params = _flat_to_shape(adj_params,(),param_shapes)

        return (None, None, adj_y, time_vjps, None, None, None, None, None, None, None, None, None, *adj_params)


