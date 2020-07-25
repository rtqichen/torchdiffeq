import torch
import torch.nn as nn
from .odeint import odeint


class OdeintAdjointMethod(torch.autograd.Function):
    @staticmethod
    def forward(ctx, func, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method, adjoint_options,
                n_tensors, *args):

        params = args[:-n_tensors]
        y0 = args[-n_tensors:]

        ctx.func = func
        ctx.adjoint_rtol = adjoint_rtol
        ctx.adjoint_atol = adjoint_atol
        ctx.adjoint_method = adjoint_method
        ctx.adjoint_options = adjoint_options
        ctx.n_tensors = n_tensors

        with torch.no_grad():
            ans = odeint(func, y0, t, rtol=rtol, atol=atol, method=method, options=options)
        ctx.save_for_backward(t, *params, *ans)
        return ans

    @staticmethod
    def backward(ctx, *grad_output):
        t, *args = ctx.saved_tensors
        params = tuple(args[:-ctx.n_tensors])
        ans = tuple(args[-ctx.n_tensors:])

        try:
            grid_points = ctx.adjoint_options['grid_points']
        except (KeyError, TypeError):
            pass
        else:
            adjoint_options = ctx.adjoint_options.copy()
            adjoint_options['grid_points'] = grid_points.flip(0)

        # TODO: use a nn.Module and call odeint_adjoint to implement higher order derivatives.
        def augmented_dynamics(t, y_aug):
            # Dynamics of the original system augmented with
            # the adjoint wrt y, and an integrator wrt t and args.
            y, adj_y = y_aug[:ctx.n_tensors], y_aug[ctx.n_tensors:2 * ctx.n_tensors]  # Ignore adj_time and adj_params.

            with torch.set_grad_enabled(True):
                t = t.detach().requires_grad_(True)
                y = tuple(y_.detach().requires_grad_(True) for y_ in y)
                func_eval = ctx.func(t, y)

                # Workaround for PyTorch bug #39784
                _t = torch.as_strided(t, (), ())
                _y = tuple(torch.as_strided(y_, (), ()) for y_ in y)
                _f_params = tuple(torch.as_strided(param, (), ()) for param in params)

                vjp_t, *vjp_y_and_params = torch.autograd.grad(
                    func_eval, (t,) + y + params,
                    tuple(-adj_y_ for adj_y_ in adj_y), allow_unused=True, retain_graph=True
                )
            vjp_y = vjp_y_and_params[:ctx.n_tensors]
            vjp_params = vjp_y_and_params[ctx.n_tensors:]

            # autograd.grad returns None if no gradient, set to zero.
            vjp_y = tuple(torch.zeros_like(y_) if vjp_y_ is None else vjp_y_ for vjp_y_, y_ in zip(vjp_y, y))
            vjp_t = torch.zeros_like(t) if vjp_t is None else vjp_t
            vjp_params = tuple(torch.zeros_like(param) if vjp_param is None else vjp_param
                               for vjp_param, param in zip(vjp_params, params))

            return (*func_eval, *vjp_y, vjp_t, *vjp_params)

        with torch.no_grad():
            adj_y = tuple(grad_output_[-1] for grad_output_ in grad_output)
            adj_params = tuple(torch.zeros_like(param) for param in params)
            adj_time = torch.zeros((), dtype=t.dtype, device=t.device)
            time_vjps = []

            for i in range(len(t) - 1, 0, -1):
                ans_i = tuple(ans_[i] for ans_ in ans)
                grad_output_i = tuple(grad_output_[i] for grad_output_ in grad_output)
                func_i = ctx.func(t[i], ans_i)

                # Compute the effect of moving the current time measurement point.
                dLd_cur_t = sum(
                    torch.dot(func_i_.reshape(-1), grad_output_i_.reshape(-1)).reshape(1)
                    for func_i_, grad_output_i_ in zip(func_i, grad_output_i)
                )
                adj_time = adj_time - dLd_cur_t
                time_vjps.append(dLd_cur_t)

                # Run the augmented system backwards in time.
                aug_y0 = (*ans_i, *adj_y, adj_time, *adj_params)
                aug_ans = odeint(augmented_dynamics, aug_y0, t[i - 1:i + 1].flip(0),
                                 rtol=ctx.adjoint_rtol, atol=ctx.adjoint_atol,
                                 method=ctx.adjoint_method, options=ctx.adjoint_options)

                # Unpack aug_ans.
                adj_y = aug_ans[ctx.n_tensors:2 * ctx.n_tensors]
                adj_time = aug_ans[2 * ctx.n_tensors]
                adj_params = aug_ans[2 * ctx.n_tensors + 1:]

                adj_y = tuple(adj_y_[1] for adj_y_ in adj_y)
                adj_time = adj_time[1]
                adj_params = tuple(adj_params_[1] for adj_params_ in adj_params)

                adj_y = tuple(adj_y_ + grad_output_[i - 1] for adj_y_, grad_output_ in zip(adj_y, grad_output))

                del aug_y0, aug_ans

            time_vjps.append(adj_time)
            time_vjps = torch.cat(time_vjps[::-1])

            return (None, time_vjps, None, None, None, None, None, None, None, None, None, *adj_params, *adj_y)


def odeint_adjoint(func, y0, t, rtol=1e-7, atol=1e-12, method=None, options=None, adjoint_rtol=None, adjoint_atol=None,
                   adjoint_method=None, adjoint_options=None, adjoint_buffers=False):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(func, nn.Module):
        raise ValueError('func is required to be an instance of nn.Module.')

    if adjoint_rtol is None:
        adjoint_rtol = rtol
    if adjoint_atol is None:
        adjoint_atol = atol
    if adjoint_method is None:
        adjoint_method = method
    if adjoint_options is None:
        adjoint_options = options

    tensor_input = False
    if torch.is_tensor(y0):

        class TupleFunc(nn.Module):

            def __init__(self, base_func):
                super(TupleFunc, self).__init__()
                self.base_func = base_func

            def forward(self, t, y):
                return (self.base_func(t, y[0]),)

        tensor_input = True
        y0 = (y0,)
        func = TupleFunc(func)

    params = tuple(func.parameters())
    if adjoint_buffers:
        params = params + tuple(buffer for buffer in func.buffers() if buffer.requires_grad)
    n_tensors = len(y0)

    ys = OdeintAdjointMethod.apply(func, t, rtol, atol, method, options, adjoint_rtol, adjoint_atol, adjoint_method,
                                   adjoint_options, n_tensors, *params, *y0)

    if tensor_input:
        ys = ys[0]
    return ys
