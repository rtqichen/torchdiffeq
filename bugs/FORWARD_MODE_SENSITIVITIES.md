# PERF / FEATURE: forward-mode derivatives through `radau` are correct but cost ~12x a plain solve

**Status:** OPEN
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
