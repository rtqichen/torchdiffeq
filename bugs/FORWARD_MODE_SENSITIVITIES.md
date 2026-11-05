# PERF / FEATURE: forward-mode derivatives through `radau` are correct but cost ~12x a plain solve

**Status:** RESOLVED in `a337502` -- see [Resolution](#resolution) at the end.
**Branch:** `fix-stiff-implicit-solvers` (HEAD `d33a877`, "Close the Jacobian bug reports and add a changelog")
**Severity:** performance; no wrong results found
**Discovered by:** PTK_model_reconciliation (LOG F91, 2026-11-05, script `06_analysis/jvp_probe.py`): a stiff
heterogeneous microkinetics model, 16 species in log/softmax coordinates, 18 rate-constant parameters
(ln k±) living inside `func` (a torch module, not in `y0`), `method="radau"`, rtol = atol = 1e-8,
`options={"jacobian": "autograd"}`, default `grad_jacobian='exact'`.

---

## 1. Goal

The user needs the Jacobian of many outputs (~3000 residuals of observables computed from the solution) with
respect to few parameters (p = 18) for Gauss-Newton / Levenberg-Marquardt, preconditioning and identifiability.
Reverse mode gives one row per backward pass (~3000 passes). Forward mode gives one column per pass (18 passes),
and forward sensitivities in the solver could give all p columns in about the cost of one solve. The model chains
several `odeint` calls (steady state, then 5 cycles), so the sensitivity of the initial state must be carried
from one call into the next.

## 2. Measurement (one direction: one ln k parameter)

| method | d loss / d theta | rel. to reverse mode | wall time |
|---|---|---|---|
| reverse mode (`loss.backward()`), reference | 14.430525 | — | 119 s |
| **V1: `torch.autograd.forward_ad`, under `no_grad`** | 14.430514 | −7.6e-7 | **605 s** |
| V2: forward mode with grad enabled (`_reattach` runs on dual tensors) | 14.430545 | +1.4e-6 | 1400 s |
| central FD, h = 1e-3 (two plain solves) | 14.430627 | +7.1e-6 | 2 x 51 s |
| central FD, h = 1e-4 | 14.427579 | −2.0e-4 (solver noise) | 2 x 51 s |

Plain solve without gradient: 51 s. The derivative curves of the outputs agree between V1 and V2 to <= 1e-7 and
with FD (h = 1e-3) to ~5e-6 where the sensitivity is not tiny. The forward passes ran while another 6-thread job
was running on the machine (24 threads); FD solves in the same conditions took the usual 51 s.

**So forward mode works today and is accurate. It is just ~12x (V1) to ~27x (V2) a plain solve.**

## 3. What happens today (`torchdiffeq/_impl/radau.py`)

- `torch.no_grad` does not affect forward-mode AD (PyTorch semantics). With dual tensors as parameters of `func`,
  every `func` evaluation carries tangents, including:
  - every simplified-Newton iteration in `_solve_stages` (l. 154 ff., 3 stage evaluations per iteration),
  - rejected steps, the error estimate (`_error_estimate`, l. 336) and `self.f` after each step;
  - `_jacobian` (l. 139) via `JacobianEvaluator` with `jacobian='autograd'`: `torch.autograd.functional.jacobian`
    on a function whose parameters are dual tensors, i.e. reverse mode over dual tensors. Its tangent (dJ/dtheta)
    is computed and never needed; the LU of the stage matrix (`_lu_stage_matrix`, l. 185) then also runs on dual
    tensors.
- `_reattach` (l. 270) is skipped in V1: the condition (l. 257) checks `torch.is_grad_enabled()` and
  `requires_grad` only. The tangent of the converged stages comes from the Newton iterations themselves. That it
  is accurate is a consequence of Newton being iterated to a tight tolerance (the tangent of a fixed-point
  iteration converges at the same rate as the value), not a guarantee: with few iterations the tangent error is
  ~(I − M⁻¹ dG/dZ)^k, the same factor the `_reattach` docstring describes.
- V2 additionally builds the reverse graph and three exact stage Jacobians per step (`_exact_stage_jacobian`,
  l. 306), all on dual tensors.

Where exactly the 12x goes has **not been profiled**. Candidates: the dual autograd Jacobians, tangents in all
Newton iterations and rejected steps, dual LU factorizations.

## 4. Requested work

### Step 0: profile V1
Profile one forward-mode solve on a stiff test problem with parameters in `func` (e.g. Robertson from
`tests/problems.py`, rate constants as `nn.Parameter`s, replaced by dual tensors) and report the time split
(Newton iterations, `_jacobian`, LU, error estimate, dense output).

### Step 1 (cheap, if the profile points there): no tangents in the Newton matrix
The Newton/stage matrices only need primal values. Make `_jacobian` / `_lu_stage_matrix` (and
`_exact_stage_jacobian` where it is used for the matrix) work on primal values. Difficulty: tangents enter through
parameters inside `func`, which the solver cannot see or unpack; PyTorch has no public "pause forward AD" context.
Options to evaluate: unpack the result (`fwAD.unpack_dual(J).primal`) right after the Jacobian evaluation (saves
the LU cost, not the Jacobian cost); or evaluate the Jacobian by FD on primal `y` (still dual params ...). Report
what is achievable.

### Step 2 (the real fix): forward sensitivities in `radau`, all directions at once
Staggered direct method as in CVODES:
- New, opt-in API, for example `odeint_sensitivity(func, y0, t, params, S0=None, **odeint_kwargs)` returning
  `(y(t), S(t))` with `S = dy/dparams` of shape `(len(t), *y.shape, p)`. `params` = list of tensors in `func`
  (flattened to p), `S0` = sensitivity of `y0` (default zeros), so chained solves can pass S on.
- Per accepted step, after the converged stages Z (primal, as now): stage sensitivities from one linear solve with
  p right-hand sides,
  `S_Z = −M⁻¹ (dG/dy · S_y + dG/dtheta)`, with `M` = exact `dG/dZ` (already built in `_exact_stage_jacobian`,
  one LU per step), `dG/dtheta = −h (A ⊗ I) dF/dtheta` at the three stage points (n x p per stage, via
  autograd w.r.t. the parameters: min(n, p) passes), then `S_new = S_y + S_Z[2]`, and dense output for S with the
  same collocation polynomial (`Q`, `P`).
- Error control on y only (CVODES "errconS = False"); document it.
- Everything else (step control, Newton, events) unchanged and on primal values.
- Expected cost: one solve plus, per step, 3 parameter Jacobians and one LU solve with p columns. Target: J for
  n ~ 16, p = 18 in <= 3x a plain solve.

## 5. Acceptance

- S against central FD and against forward mode (V1 style) on Robertson and one more stiff problem with
  parameters in `func`: rel. <= 1e-6 at rtol = atol = 1e-8; also with nonzero `S0` and with two chained calls.
- Scalar functional: `sum(w * y(t))` sensitivities equal the reverse-mode gradient to <= 1e-6.
- No behaviour change without the new API: `tests/run_all.py` passes; reverse-mode results of `radau` unchanged
  (bitwise on the existing gradient tests).
- No silent fallbacks: if the exact stage matrix cannot be built, raise (do not fall back to the iterative path,
  cf. the `except Exception: pass` in `_reattach`).
- CHANGELOG entry, separate commit(s), this note updated with a Resolution section.

## 6. Constraints

- The PTK repo uses this fork as an editable install. Develop on a separate branch in a separate git worktree
  (not in the folder the PTK repo imports from) until the change is accepted.
- Keep `radau`'s existing reverse-mode path and defaults untouched.

---

## Resolution

**Resolved in `a337502`** ("Add odeint_sensitivity: forward sensitivities through radau"), on branch
`forward-sensitivities` (worktree `torchdiffeq-dev`, separate from the folder the PTK repo imports from).

### Step 0: profile of V1

Forward-mode AD through `radau` under `no_grad`, wall-clock split per solver method, warm runs, 4 threads,
rtol = atol = 1e-8, `jacobian='autograd'`:

| problem | plain | V1, one direction | Newton iterations share (plain / V1) | Jacobian + LU share in V1 |
|---|---|---|---|---|
| Robertson, 3 ln k | 0.33 s | 1.50 s (4.6x) | 88% / 88% | ~1% |
| synthetic 16 species, 18 ln k | 0.09 s | 0.41 s (4.6x) | 78% / 80% | ~2% |

The time goes where it goes in a plain solve, the Newton iterations; each `func` call is ~4.7x dearer with
dual-tensor parameters. The dual Jacobians and LUs suspected in section 3 are not the cost. The 12x in
section 2 is larger than measured here; a heavier `func` (more ops per call, each paying the dual overhead)
and the concurrent job are the likely reasons, not verified.

### Step 1: not done

Removing tangents from the Newton matrix would save the ~1-2% above. Not worth the code.

### Step 2: `odeint_sensitivity`

As specified in section 4, with these choices:

- `params` is a list of tensors with `requires_grad=True`, not limited to `nn.Parameter`s.
- `df/dy` and `df/dparams` at each stage come from one batched backward pass (`n` rows). The exact stage
  matrix is built from those `df/dy`, factorised once per accepted step, and solved with `p` right-hand sides.
- Both derivatives are checked together against a directional finite difference on the 1st, 2nd, 4th, ...
  evaluation (the probe from the Jacobian fix, over state and parameters). A mismatch **raises**: there is no
  silent fallback. Parameters are perturbed through `.data` and restored bit for bit, so their values and
  version counters are untouched.
- Raises instead of degrading: `method != 'radau'`, tuple `y0`, `grad_jacobian='iterative'`,
  `jacobian='fd'`, a parameter without `requires_grad`, wrong `S0` shape, non-finite derivatives, events.

### Acceptance (section 5)

| criterion | result |
|---|---|
| S vs central FD, Robertson, rtol = atol = 1e-8 | max rel. 3e-7 (FD solves at 1e-8), 8e-9 (FD solves at 1e-12) |
| S vs forward mode (V1), Robertson | max rel. 4e-9 |
| S vs closed form, stiff linear 2x2 (`matrix_exp`) | rel. 1e-10 at tol 1e-10; test bound 1e-6 at tol 1e-8 |
| nonzero `S0`; two chained calls | match closed form <= 1e-6; chained Robertson matches one call <= 1e-6 |
| `sum(w * y(t))` vs reverse-mode gradient | Robertson 9e-11, 16-species network 9e-10 |
| no behaviour change | `radau` outputs and reverse-mode gradients bitwise equal to `d33a877` (15 tensors: exact and iterative `grad_jacobian`, adjoint, reversed time, event); `run_all.py`: 68 of 69 pass |
| cost, 16 species, 18 parameters | **3.0x a plain solve for all 18 directions** (target <= 3x) |

The one failing test in `run_all.py`, `TestMinMaxStep.test_min_max_step`, fails identically on `d33a877`:
the installed scipy's LSODA ignores `min_step`. Unrelated.

Tests: `tests/sensitivity_tests.py` (22 tests). Checked that they can fail: dropping the `J S` term of the
update fails 11 of them; disabling the finite difference check fails the detached-parameter test.

### Found along the way, not fixed here

- **Reverse mode through `radau` drops the gradient of a parameter that is not an `nn.Parameter`.** The
  re-attach after the Newton loop runs only if `y` requires grad or `func.parameters()` does. A plain tensor
  with `requires_grad=True` stored on `func` (or captured in a closure) gets no ODE contribution to its
  gradient, silently: with `loss = y[-1, 2] + 1e-3 * lnk.sum()` on the 16-species network, `dloss/dlnk[0]`
  is 1.000e-3 instead of 1.0024e-3. `odeint_sensitivity` takes `params` explicitly and is not affected.
- **Forward-mode AD through `odeint` with `radau` crashes without `first_step`.** The automatic initial step
  is computed from `func`, so with dual parameters the step size itself becomes a dual tensor, and
  `_nextafter`'s custom autograd function has no forward-mode rule (`NotImplementedError`). Passing
  `first_step` avoids it. `odeint_sensitivity` is not affected.
