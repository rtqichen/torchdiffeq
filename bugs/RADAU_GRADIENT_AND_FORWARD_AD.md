# BUG: `radau` returns a wrong gradient for parameters that are not `nn.Parameter`s; forward-mode AD through `odeint` crashes

**Status:** OPEN
**Branch:** found on `forward-sensitivities` (HEAD `ed8a5f5`); both reproduce identically on `d33a877`
(`fix-stiff-implicit-solvers`), so they predate that branch.
**Environment:** torch 2.13.0+cu132, CPU, float64, Python 3.12.
**Discovered while:** resolving [FORWARD_MODE_SENSITIVITIES.md](FORWARD_MODE_SENSITIVITIES.md).
`odeint_sensitivity` is affected by neither: it takes `params` explicitly and never passes dual tensors to
the solver.

Two independent bugs:

| | bug | severity | affected |
|---|---|---|---|
| A | reverse mode through `radau` silently returns a wrong gradient | **high**: silent, wrong result | `radau`, when the parameter is not an `nn.Parameter` and `y0` does not require grad |
| B | forward-mode AD (`torch.autograd.forward_ad`) through `odeint` raises `NotImplementedError` | low: loud, has a workaround (or use `odeint_sensitivity`) | `radau`, `lsoda` without `first_step`; `dopri5` (and probably every adaptive explicit RK solver) always |

---

## A. `radau`: wrong reverse-mode gradient for a parameter that is not an `nn.Parameter`

### Summary

The Newton iteration runs under `no_grad`; the converged stages get their autograd history back in
`_reattach`. That happens only if (`radau.py:257`)

```python
if torch.is_grad_enabled() and (y.requires_grad or _params_require_grad(self.func)):
```

and `_params_require_grad` looks only at `func.parameters()`. So a tensor with `requires_grad=True` that
`func` uses but that is **not** a registered `nn.Parameter` is invisible to it:

- a closure: `odeint(lambda t, y: -k * y, ...)`,
- a plain tensor attribute of a module: `self.k = torch.tensor(..., requires_grad=True)`,
- a tensor computed outside `func` from parameters and captured (`k = torch.exp(model.lnk)`),
- a dual or batched tensor swapped in for a parameter.

Unless `y0` requires grad, no step is re-attached. The result is **not zero, which would at least be
conspicuous: it is a wrong, plausible-looking number.** Part of a gradient still arrives, most likely through
the step size. The initial step is computed from `func(t0, y0)`, so `h` carries a graph, and the dense
output's `x = (t - t_old) / h_old` passes it on. That this is the path is inferred from the code, not traced.

### Reproduction

```python
import torch
from torchdiffeq import odeint

DT = torch.float64
k = torch.tensor(2.0, dtype=DT, requires_grad=True)          # not an nn.Parameter
y0 = torch.tensor([1.0], dtype=DT)
t = torch.tensor([0.0, 1.0], dtype=DT)

y = odeint(lambda t, y: -k * y, y0, t, method='radau', rtol=1e-8, atol=1e-10)
g, = torch.autograd.grad(y[-1, 0], k)
print(g.item(), 'exact:', -torch.exp(torch.tensor(-2.0)).item())
# -0.054134  exact: -0.135335
```

### Measured

`y' = -k y`, `k = 2`, `dy(1)/dk` (exact `-0.135335`), rtol 1e-8, atol 1e-10:

| method | `nn.Parameter` | plain tensor attribute | closure | closure, `y0.requires_grad=True` |
|---|---|---|---|---|
| `radau` | -0.135335 | **-0.054134** | **-0.054134** | -0.135335 |
| `lsoda` | -0.135335 | -0.135335 | -0.135335 | -0.135335 |
| `dopri5` | -0.135335 | -0.135335 | — | — |
| `radauIIA5`, step 0.01 | -0.135335 | -0.135335 | — | — |
| `implicit_euler`, step 0.01 | -0.135326 | -0.135326 | — | — |

Stiff case, `y' = [[-k, 1], [0, -1]] y`, `k = 1000`, `d(y_0(1) + y_0(10))/dk` (exact `-3.686618e-07`):
`radau` gives `-3.686618e-07` with an `nn.Parameter` and **`-1.475549e-07`** with a plain tensor (60% off);
`lsoda` and `radauIIA5` are right with both.

On the 16-species network from the forward-sensitivities note, `loss = y[-1, 2] + 1e-3 * lnk.sum()` gave
`dloss/dlnk[0] = 1.000000e-03` with a plain tensor against `1.002437e-03` with an `nn.Parameter`, i.e. the
ODE part of that component came out as ~0 there.

### Who is affected

Anyone taking reverse-mode gradients through `method='radau'` of a parameter that is not registered on
`func` as an `nn.Parameter`, with a `y0` that does not require grad. A model whose rate constants are
`nn.Parameter`s of the `func` module is **not** affected. Neither is `odeint_adjoint` (it takes
`adjoint_params`).

### `lsoda`: same condition, apparently not affected

`lsoda.py:366` has the identical condition, yet gave correct gradients in every case above. Probable
reason, from the code and not verified: `lsoda`'s predicted Nordsieck array carries `h * f`, and `f` is
evaluated with grad enabled, so `self.y` requires grad after the first step and every later step is
re-attached. If so, the **first** step's parameter derivative is still missing, which would be a small error
on problems where the first step matters. Worth a targeted test (a long first step on a fast-decaying
component), and worth fixing along with `radau` regardless.

### Suggested fix

Decide from what `func` actually returns, not from `func.parameters()`. `_before_integrate` already evaluates
`self.f = self.func(self.t, self.y)` with grad enabled, so for free:

```python
self._needs_grad = torch.is_grad_enabled() and (self.y0.requires_grad or self.f.requires_grad)
```

and re-attach every step when `_needs_grad`. The fixed-grid implicit solvers already decide this way
(`rk_common.py:555-558`, `f.requires_grad`), which is why they are not affected. Same change in `lsoda`.
`_params_require_grad` can then go from both files.

Expect reverse-mode results to change for **affected** setups only. `radau` with `nn.Parameter`s should stay
bitwise identical; confirm by recording outputs and gradients at the old commit and comparing with
`torch.equal` after the change.

**Related, for evaluation:** the step size `h` carries an autograd graph from the automatic initial step (see
above) in every case, including with `nn.Parameter`s. That makes the gradient include a contribution from
"how the first step size depends on the parameters". That contribution is not a derivative of the ODE
solution, and it is O(tolerance), but it is nonzero. `_step_factor` already detaches the step controller on
purpose; detaching the initial step (`h = h.detach()` in `_before_integrate`, likewise in `lsoda`) would be
consistent with that and would also remove the partial gradient that makes bug A look plausible instead of
zero. It changes reverse-mode gradients by O(tol), so it is a decision, not a free fix.

### Suggested regression test

```python
def test_closure_and_plain_tensor_parameters_get_the_gradient():
    for method in ('radau', 'lsoda'):
        k = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        y = odeint(lambda t, y: -k * y, torch.tensor([1.0], dtype=torch.float64),
                   torch.tensor([0.0, 1.0], dtype=torch.float64), method=method, rtol=1e-8, atol=1e-10)
        g, = torch.autograd.grad(y[-1, 0], k)
        assert abs(g.item() + math.exp(-2.0)) < 1e-6
```

Plus the stiff 2x2 case with a plain tensor, and an `nn.Parameter` case that must stay bitwise unchanged.

---

## B. Forward-mode AD through `odeint` raises `NotImplementedError`

### Summary

Two custom `torch.autograd.Function`s have a `backward` but no `jvp`, so forward-mode AD stops at them:

| function | where | used when |
|---|---|---|
| `_StitchGradient` | `misc.py:347`, applied in `_nextafter` | every `func` call with `perturb=PREV/NEXT` (`radau`'s last stage, `lsoda`, `dopri5`'s last stage) |
| `_UncheckedAssign` | `rk_common.py`, `_runge_kutta_step` l. 70, 81 | every step of the adaptive explicit RK solvers |

Both come from upstream torchdiffeq (`upstream/master` has the same lines).

- `radau` and `lsoda` only reach `_StitchGradient` with a **dual `t`**. That happens when the parameters are
  dual and the initial step is chosen automatically: `_select_initial_step` evaluates `func`, so `h` becomes
  dual, then `t + c h`, then the `nextafter` input. With `first_step` given, `h` and `t` stay plain and forward
  mode works and is accurate (that was V1 in the forward-sensitivities note).
- `dopri5` reaches `_UncheckedAssign` with dual stage values on every step, so it fails regardless.
- The fixed-grid solvers do not perturb by default and work.

### Reproduction

```python
import torch
import torch.autograd.forward_ad as fwAD
from torchdiffeq import odeint

DT = torch.float64
with torch.no_grad(), fwAD.dual_level():
    k = fwAD.make_dual(torch.tensor(2.0, dtype=DT), torch.tensor(1.0, dtype=DT))
    odeint(lambda t, y: -k * y, torch.tensor([1.0], dtype=DT), torch.tensor([0.0, 1.0], dtype=DT),
           method='radau')
# NotImplementedError: You must implement the jvp function for custom autograd.Function to use it
# with forward mode AD.   (from misc._nextafter -> _StitchGradient)
```

### Measured

`y' = -k y`, dual `k = 2` with tangent 1, `dy(1)/dk` (exact `-0.135335`):

| method | automatic first step | `first_step=1e-3` |
|---|---|---|
| `radau` | NotImplementedError | -0.135335 |
| `lsoda` | NotImplementedError | -0.135335 |
| `dopri5` | NotImplementedError | NotImplementedError (`_UncheckedAssign`) |
| `radauIIA5`, step 0.01 | -0.135326 | -0.135326 |
| `implicit_euler`, step 0.01 | -0.135272 | -0.135272 |

### Suggested fix

- Give both Functions a `jvp` (and `setup_context` if needed). Both are identities as far as derivatives go:
  `_StitchGradient.jvp` returns the tangent of `x1`, and `_UncheckedAssign.jvp` writes the tangent of the
  assigned value into the tangent of the target at the same index.
- The initial-step detach suggested under A would also keep `t` plain for `radau`/`lsoda`, so they would no
  longer hit `_StitchGradient` at all.
- Priority is low for this fork's own use: `odeint_sensitivity` covers the forward-sensitivity need for
  `radau` at a fraction of the cost. It matters for anyone using `torch.func.jvp`/`jacfwd` on a model that
  contains an `odeint` call.

### Suggested regression test

Forward-mode `dy/dk` for `y' = -k y` through `radau`, `lsoda` and `dopri5` with an automatic first step,
against `-t exp(-k t)` to 1e-6.
