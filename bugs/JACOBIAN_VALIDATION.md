# BUG: implicit solvers silently accept a structurally wrong autograd Jacobian

**Status:** FIXED in `0903417` -- see [Resolution](#resolution) at the end.

**Branch:** `fix-stiff-implicit-solvers` (HEAD `f437afe`, "Add native adaptive implicit
solvers: radau and lsoda")
**Severity:** high — silent ~200x slowdown / non-convergence, no warning, wrong-but-finite
intermediate accepted
**Discovered by:** a stiff heterogeneous microkinetics RHS (19 states, 42 reactions) whose
rate law is evaluated in numpy

---

## 1. Summary

`_jacobian` tries `torch.autograd.functional.jacobian` first and **accepts the result
whenever it is finite**. The finite-difference fallback is only reached if autograd
*raises* or returns a non-finite value.

If the right-hand side is only **partially** differentiable — the common case when part of
it calls out to numpy / scipy / C and detaches — autograd returns a Jacobian in which the
non-differentiable contributions are simply **missing**. That result is finite, so it is
accepted. The Newton iteration then runs on a Jacobian that omits the stiff part of the
system, fails to converge, and the step-size controller collapses to its floor.

No warning is emitted. The only symptom is that the integration becomes ~200x slower or
never finishes.

## 2. Affected code — three copies of the same pattern

| file | method | lines |
|---|---|---|
| `torchdiffeq/_impl/radau.py` | `RadauIIA5Adaptive._jacobian` | 136–151 |
| `torchdiffeq/_impl/lsoda.py` | `LSODA._jacobian` | 209–223 |
| `torchdiffeq/_impl/rk_common.py` | `_jacobian` (implicit fixed-grid base) | 446–466 |

Reachable via `method="radau"`, `method="lsoda"`, and the implicit fixed-grid solvers
(`radauIIA5`, `radauIIA3`, …) registered in `odeint.py`.

The offending block (`radau.py:143–151`):

```python
try:
    with torch.enable_grad():
        J = torch.autograd.functional.jacobian(_g, y_.reshape(-1), vectorize=True)
    J = J.reshape(self.n, self.n).detach()
    if torch.isfinite(J).all():      # <-- finiteness is NOT sufficient
        return J
except Exception:
    pass
return self._fd_jacobian(_g, y_.reshape(-1))
```

`rk_common.py` additionally falls through to `torch.zeros(n, n)` as a last resort
(line ~466), which is the same class of problem: a plausible-looking, silently wrong
Jacobian rather than a loud failure.

## 3. Why finiteness is not a sufficient check

A partially-detached RHS produces a Jacobian that is finite, non-zero, and **completely
wrong**. Concretely: an RHS of the form

```python
sigma = expensive_rates(x.detach().cpu().numpy())      # graph severed here
dy    = y_in - eta * y + KAPPA * sigma                 # but `y` is still differentiable
```

leaks a real derivative through the `eta * y` term while contributing *nothing* from
`sigma`. Autograd therefore reports a Jacobian that is neither zero (so a "is it zero?"
check would not catch it) nor correct.

## 4. Measured evidence

RHS: 19-state dimensionless CSTR + catalyst surface, 42-reaction microkinetic rate law
spanning ~29 decades in rate constant, float64, rate law evaluated in numpy and returned
via `torch.as_tensor(...)` after `.detach().cpu().numpy()`.

Jacobian at the initial state, autograd (as accepted by the current code) vs finite
differences on the same `_g`:

| | autograd (accepted) | finite differences (correct) |
|---|---|---|
| `max\|J\|` | 3.8487e+06 | **2.7888e+11** |
| all finite | True | True |
| non-zero entries | **6 / 361** | 266 / 361 |
| all-zero rows | **all 13 surface rows (0–12)** | none |
| relative error vs FD | **1.0000e+00** | — |
| `\|lambda\|` range | 3.849e+06 (one value) | 5.035e-03 … 2.789e+11 |

The accepted Jacobian misses the stiff spectrum by **five orders of magnitude** and has
100% relative error.

End-to-end cost, `method="radau"`, `rtol=atol=1e-6`, `first_step=1e-18`, integrating a
steady state over `t = 0…2000`:

| Jacobian actually used | RHS evals | wall |
|---|---|---|
| accepted partial autograd | 527 215 | **did not finish in 120 s** |
| finite-difference fallback | **2 371** | **0.51 s** |
| exact autograd (RHS made fully differentiable) | 1 971 | 0.36 s |

All three produce the same answer where they finish (agreeing to 6e-14), so this is purely
a performance/robustness failure — but a ~200x silent one.

Note the third row: when the RHS *is* fully differentiable, autograd is genuinely better
than FD (1 971 vs 2 371 evals). **The fix must not be "always use finite differences."**

## 5. Minimal reproduction

```python
import torch
from torchdiffeq import odeint

# stiff 2x2 system where ONE component's contribution is detached
A = torch.tensor([[-1000.0, 1.0], [0.0, -1.0]], dtype=torch.float64)

def rhs(t, y):
    stiff = (A @ y.reshape(-1, 1)).reshape(-1)
    return stiff.detach() + 0.001 * y      # graph only through the weak term

y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
t = torch.linspace(0, 10, 5, dtype=torch.float64)

# Inspect what _jacobian would accept:
def _g(z):
    return rhs(t[0], z)
with torch.enable_grad():
    J = torch.autograd.functional.jacobian(_g, y0, vectorize=True)
print("autograd J:\n", J)          # ~diag(0.001) -- finite, non-zero, and wrong
                                   # true J is dominated by -1000
odeint(rhs, y0, t, method="radau", rtol=1e-6, atol=1e-6)   # crawls
```

## 6. Impact

- Any RHS that mixes torch with a detached non-torch component: numpy/scipy rate laws,
  table lookups, external simulators, compiled kernels, or any `.detach()` used
  defensively inside a user's `forward`.
- **Silent.** No warning; the user sees only an unexplained slowdown or a
  non-convergence message that points at `step_size` / tolerances, which are not the cause.
- Currently the *only* thing that routes such an RHS to the correct FD path is autograd
  **raising**. Downstream we had to add a deliberate `raise` inside our RHS to force it —
  that should not be a user's responsibility, and it is fragile (a user who instead
  detaches cleanly gets silence).

## 7. Suggested fixes

Ordered by recommendation. These are suggestions, not a prescription — the maintainer
should pick based on how much per-step cost is acceptable.

1. **Validate the autograd Jacobian against a directional FD probe (recommended default).**
   One extra RHS evaluation: pick a random unit `v`, compare `J @ v` against
   `(f(y + h v) - f(y)) / h`. If the relative mismatch exceeds a tolerance
   (say 1e-2, loose enough for FD noise), discard and fall back to FD, **and warn once**.
   Cost can be amortised by validating only on the first Jacobian build per `integrate()`
   call, since differentiability is a property of the RHS, not of the state.

2. **Expose the source explicitly**: `options={"jacobian": "auto" | "autograd" | "fd"}`,
   defaulting to a validated `"auto"`. Lets users who know their RHS is opaque skip the
   probe entirely, and makes the behaviour documented rather than emergent.

3. **Cheap structural sanity check** (good complement, not sufficient alone): if a row of
   `J` is identically zero while the corresponding component of `f(y)` is non-zero and
   depends on `y`, the Jacobian is inconsistent. In the case above this catches all 13
   surface rows immediately. Can false-positive on genuinely sparse/decoupled systems, so
   it should warn-and-verify rather than reject outright.

4. **Do not** simply always use FD (loses accuracy — `sqrt(eps)` ≈ 1e-8 truncation — and
   costs `n+1` evals per Jacobian, measurably worse than autograd when the RHS is clean),
   and **do not** rely on `except Exception` as the detection mechanism.

Also worth doing regardless: the three copies should be factored into **one** shared
implementation, so this is fixed once rather than three times, and
`rk_common.py`'s silent `return torch.zeros(n, n)` should at minimum warn.

## 8. Suggested regression test

```python
def test_partially_detached_rhs_does_not_accept_bogus_jacobian():
    """A half-detached stiff RHS must not silently get the autograd Jacobian."""
    A = torch.tensor([[-1000.0, 1.0], [0.0, -1.0]], dtype=torch.float64)

    def rhs(t, y):
        return (A @ y.reshape(-1, 1)).reshape(-1).detach() + 0.001 * y

    y0 = torch.tensor([1.0, 1.0], dtype=torch.float64)
    t = torch.linspace(0, 10, 5, dtype=torch.float64)
    for method in ("radau", "lsoda", "radauIIA5"):
        sol = odeint(rhs, y0, t, method=method, rtol=1e-6, atol=1e-6)
        # reference: the same problem with the detach removed
        assert torch.allclose(sol, reference, rtol=1e-4)
```

Assert on **step/eval count** as well as accuracy — the failure mode here is cost, not a
wrong answer, so an accuracy-only test would pass while the bug is present.

## 9. Notes for whoever picks this up

- `vectorize=True` in `autograd.functional.jacobian` can behave differently from
  `vectorize=False` for some ops; validate the path that is actually used.
- The `except Exception: pass` also swallows genuine bugs in a user's RHS. Consider
  logging at debug level rather than discarding silently.
- Reproducer environment: `hCRNNenv` (Python 3.12, torch 2.13+cu132), this fork installed
  editable. The originating case lives in
  `pinn-neural-kinetics/Experiments/HSA_O_001/HSA_O_001_007.py`
  (`ODE_Wrapper_PTK._np_rates`), which now contains an explicit `raise` as a workaround —
  that workaround should become unnecessary once this is fixed.

---

## Resolution

**Fixed in `0903417`** ("Check the autograd Jacobian before the implicit solvers trust it").
Same root cause as [INFERENCE_MODE_ZERO_JACOBIAN.md](INFERENCE_MODE_ZERO_JACOBIAN.md); both
are closed by the same change.

What was done, against the suggestions in section 7:

- **7.1 (directional FD probe): done.** `torchdiffeq/_impl/jacobian.py` checks autograd
  against a directional forward difference on the 1st, 2nd, 4th, 8th, ... Jacobian of each
  solve -- not only the first, so a detached part whose derivative happens to vanish at the
  initial state is still caught later. A plain one-shot probe with a 1e-2 tolerance raised false
  alarms on Robertson at `t = 0` (curvature of `3e7 * y2**2` dominates any practical step), so
  on a mismatch the probe step is divided by 16 up to four times: curvature shrinks with the
  step, a missing Jacobian part does not. Only a mismatch that stops shrinking counts. The first
  one warns once and switches to finite differences for the rest of the solve.
- **7.2 (explicit option): done.** `options={"jacobian": "auto" | "autograd" | "fd"}` on
  `radau`, `lsoda` and the fixed-grid implicit family; documented in `FURTHER_DOCUMENTATION.md`.
- **7.3 (zero-row heuristic): not needed.** The probe catches all-zero rows and an all-zero
  Jacobian directly.
- **Shared implementation: done.** The three copies are replaced by one `JacobianEvaluator`.
  The silent `torch.zeros(n, n)` fallback in `rk_common.py` is gone; when no finite Jacobian can
  be formed the solver raises.
- **`except Exception: pass`:** an autograd failure now warns once (with the exception type and
  message) instead of being swallowed.

Corrections to the report:

- Section 4 calls this "purely a performance/robustness failure". That holds for `radau` and
  `lsoda`. The fixed-grid solvers (`radauIIA5` etc.) returned **wrong answers**: on the 2x2
  reproduction `radauIIA5` gave ~9.5e10 instead of ~5e-8.
- A fully detached RHS is affected too: autograd returns an all-zero Jacobian rather than
  raising (it only raises if the RHS itself does, e.g. `.numpy()` on a grad-requiring tensor).
- A third instance was found while fixing this: `odeint_adjoint`'s backward dynamics compute
  their vector-Jacobian products without a graph, so every implicit backward solve got `J = 0`.
  With `k = 1000` in `y' = -k y`, `radau`/`lsoda` underflowed and the fixed-grid solvers returned
  gradients off by ~1e100. The adjoint backward solve now uses `jacobian='fd'` by default.

Measured after the fix:

| case | before | after |
|---|---|---|
| 2x2 reproduction, `radau` | 317 614 evals, 15.7 s | 511 evals, 0.19 s |
| 2x2 reproduction, `lsoda` | 61 603 evals, 8.8 s | 711 evals, 0.05 s |
| 2x2 reproduction, `radauIIA5` | wrong answer (~9.5e10) | correct |
| HSA_O_001 steady state, numpy rate law, guard removed | did not finish in 120 s | 2 355 evals, 0.51 s |
| same, torch port of the rate law (fully differentiable) | 1 971 evals | 1 981 evals, no warning |

The last row is the false-alarm check on a badly scaled system (rate constants spanning ~29
decades): the probe accepts the correct Jacobian and costs 10 extra evaluations (0.5%).

Regression tests: `tests/jacobian_tests.py` (partially and fully detached RHS, `jacobian='fd'`,
no false alarm on Robertson, adjoint gradients), across `radau`, `lsoda`, `radauIIA5` and
`sdirk2`. They assert on evaluation counts as well as accuracy, as section 8 asks, and fail on
`eda9a74`.

**Downstream:** the deliberate `raise` in
`pinn-neural-kinetics/Experiments/HSA_O_001/HSA_O_001_007.py` (`ODE_Wrapper_PTK._np_rates`) is
no longer needed. With it in place the solve still works but now warns once that autograd
failed; the clean replacement is removing it and passing `"jacobian": "fd"` in that branch's
solver options.
