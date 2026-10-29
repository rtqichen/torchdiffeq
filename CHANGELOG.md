# Changelog

Changes in this fork (`0.2.5+stiff.fix`) relative to upstream torchdiffeq.

## Implicit solvers: checked Jacobian (`0903417`)

**Fixed.** The implicit solvers (`radau`, `lsoda` and the fixed-grid family `implicit_euler`,
`implicit_midpoint`, `trapezoid`, `radauIIA3`, `radauIIA5`, `gl4`, `gl6`, `sdirk2`, `trbdf2`)
accepted any finite Jacobian from autograd. In three situations that Jacobian was silently
wrong:

| situation | what autograd returned | effect |
|---|---|---|
| part of `func` computed outside autograd (numpy/scipy, `.detach()`, `.item()`) | `df/dy` with that part missing | `radau`/`lsoda` up to 200x slower or never finish; fixed-grid solvers return wrong answers |
| solving under `torch.inference_mode` (e.g. Lightning `test`/`validate`/`predict`) | all zeros | `radau`/`lsoda` hit `max_num_steps`; fixed-grid solvers raise "diverged" |
| `odeint_adjoint` backward pass with an implicit solver | all zeros | stiff problems: `radau`/`lsoda` underflow, fixed-grid gradients off by up to ~1e100 |

**Results to re-check** if they were produced before this fix: adjoint gradients from any
implicit solver on a stiff problem, and fixed-grid implicit results whose `func` was partly
detached or that were run under `inference_mode`. `radau`/`lsoda` results that did finish are
correct, just slow.

**New option** `jacobian` (`options={"jacobian": ...}`) on all implicit solvers:

- `'auto'` (default): autograd, checked against finite differences on the 1st, 2nd, 4th, 8th,
  ... Jacobian of each solve. On a mismatch, a one-time `RuntimeWarning` and finite differences
  for the rest of the solve. Costs a few extra `func` evaluations per solve.
- `'autograd'`: autograd without the check, for a `func` known to be fully differentiable.
- `'fd'`: finite differences only, for a `func` known not to be differentiable.

**Other behaviour changes**

- Under `torch.inference_mode`, autograd now runs with inference mode switched off locally, so
  the exact Jacobian is used. If `func` uses a tensor created under inference mode, it warns and
  falls back to finite differences.
- `odeint_adjoint` uses `jacobian='fd'` for the backward solve unless `adjoint_options` sets it.
- A `func` that makes autograd raise now gets a one-time warning; the switch to finite
  differences used to be silent.
- The fixed-grid implicit solvers raise when no finite Jacobian can be formed, instead of
  silently using an all-zero one.

Bug reports with measurements: [bugs/JACOBIAN_VALIDATION.md](bugs/JACOBIAN_VALIDATION.md),
[bugs/INFERENCE_MODE_ZERO_JACOBIAN.md](bugs/INFERENCE_MODE_ZERO_JACOBIAN.md).
