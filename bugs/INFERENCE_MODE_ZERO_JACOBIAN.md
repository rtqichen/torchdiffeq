# Implicit solvers use an all-zero Jacobian under torch.inference_mode (radau, lsoda; same pattern in rk_common)

**Status:** FIXED in `0903417` -- see Resolution at the end.

Version: torchdiffeq 0.2.5+stiff.fix, commit eda9a74; torch 2.14.1, CPU, float64.

Summary
radau and lsoda build the Newton Jacobian with torch.autograd.functional.jacobian inside torch.enable_grad(). They fall back to finite differences only if that call raises or returns non-finite values. Under torch.inference_mode(), enable_grad() has no effect, so no graph is recorded. Called without strict=True, jacobian then quietly returns zeros. They are finite, so the solver accepts J = 0. Newton then converges very slowly or not at all, the step size collapses, and the solve runs until max_num_steps.

This hits every PyTorch Lightning trainer.test() / trainer.validate() / trainer.predict() call with default settings, because Lightning runs those under inference_mode.

Reproduction


import time, torch
from torchdiffeq import odeint

def robertson(t, y):
    y1, y2, y3 = y[..., 0], y[..., 1], y[..., 2]
    return torch.stack((-0.04 * y1 + 1e4 * y2 * y3,
                        0.04 * y1 - 1e4 * y2 * y3 - 3e7 * y2**2,
                        3e7 * y2**2), dim=-1)

y0 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
t = torch.tensor([0.0, 1e2], dtype=torch.float64)

for name, ctx in (("no_grad", torch.no_grad), ("inference_mode", torch.inference_mode)):
    with ctx():
        with torch.enable_grad():
            J = torch.autograd.functional.jacobian(lambda z: robertson(t[0], z), y0, vectorize=True)
    print(name, "max|J| =", float(J.abs().max()))

for method in ("radau", "lsoda"):
    for name, ctx in (("no_grad", torch.no_grad), ("inference_mode", torch.inference_mode)):
        t0 = time.perf_counter()
        try:
            with ctx():
                y = odeint(robertson, y0, t, method=method, rtol=1e-6, atol=1e-10,
                           options={"max_num_steps": 20_000})
            res = y[-1].tolist()
        except AssertionError as e:
            res = f"FAILED: {e}"
        print(method, name, f"{time.perf_counter() - t0:.2f} s", res)
Observed


no_grad         jacobian at y0: max|J| = 0.04, finite: True
inference_mode  jacobian at y0: max|J| = 0,    finite: True
radau  no_grad           0.08 s,    1504 RHS calls, y(100) = [0.617235, 6.15359e-06, 0.382759]
radau  inference_mode   41.77 s, 1029102 RHS calls, FAILED: max_num_steps exceeded (20000>=20000)
lsoda  no_grad           0.09 s,     617 RHS calls, y(100) = [0.617239, 6.1537e-06, 0.382754]
lsoda  inference_mode    6.01 s,   80973 RHS calls, FAILED: max_num_steps exceeded (20000>=20000)
On a larger stiff system (16 states, a chemical-kinetics PTK reactor), one Lightning trainer.test() had not finished a single 480 s cycle after 23 min. With inference_mode=False the same cycle takes 13.5 s.

Expected
The same result as under no_grad. Either the finite-difference fallback is used, or there is at least a clear error or warning.

Cause
torchdiffeq/_impl/radau.py:136-151 (same logic in lsoda.py:212-232):


try:
    with torch.enable_grad():          # no effect inside inference_mode
        J = torch.autograd.functional.jacobian(_g, y_.reshape(-1), vectorize=True)
    J = J.reshape(self.n, self.n).detach()
    if torch.isfinite(J).all():        # zeros are finite -> accepted
        return J
except Exception:
    pass
return self._fd_jacobian(...)
rk_common.py:446-466 (the Jacobian for implicit_euler, implicit_midpoint, radauIIA3/5, gl4/6, …) has the same pattern. Those methods are not tested here.

Suggested fix (in all three _jacobians)

Skip autograd and go straight to _fd_jacobian when torch.is_inference_mode_enabled(). Inference mode can never record a graph.
Optionally treat an all-zero autograd J (not J.any()) as a failure and fall back to finite differences, which are cheap compared to a crawling solve. A Jacobian that is truly zero is rare, and the finite-difference check confirms it.
strict=True would also catch it. But it raises whenever any output doesn't depend on the input (e.g. inert species with a constant derivative), so it would force finite differences more often than needed.

Workaround for users
Call the solver outside inference_mode, e.g. pl.Trainer(inference_mode=False).

---

## Resolution

**Fixed in `0903417`** ("Check the autograd Jacobian before the implicit solvers trust it").
Same root cause as [JACOBIAN_VALIDATION.md](JACOBIAN_VALIDATION.md) -- autograd's result was
accepted as soon as it was finite -- and closed by the same change.

- Instead of skipping autograd under inference mode (suggestion 1), the Jacobian is computed with
  inference mode switched off locally (`torch.inference_mode(False)`, with the state and time
  cloned into normal tensors). That gives the **exact** Jacobian rather than finite differences,
  so solves under inference mode now match `no_grad` evaluation for evaluation.
- If `func` uses a tensor that was itself created under inference mode, autograd cannot record it
  and raises; the solver then warns once and uses finite differences.
- An all-zero Jacobian (suggestion 2) is caught in general by the new finite-difference check,
  not by a special case.
- The fixed-grid implicit solvers (`rk_common.py`) were affected as well: under inference mode
  `implicit_euler` and `radauIIA5` raised "the nonlinear solve diverged", wrongly blaming
  `step_size`.

Re-run of the reproduction above after the fix (torch 2.13, CPU):

| method | `no_grad` | `inference_mode` |
|---|---|---|
| radau | 1512 calls, y(100) = [0.617235, 6.15359e-06, 0.382759] | identical |
| lsoda | 633 calls, y(100) = [0.617239, 6.1537e-06, 0.382754] | identical |
| radauIIA5, step 1.0 | 2230 calls | identical |
| implicit_euler, step 1.0 | 728 calls | identical |

(A few more calls than in the `no_grad` rows of the report, 1504 -> 1512 for radau and
617 -> 633 for lsoda: the cost of the new Jacobian check.)

Regression tests: `tests/jacobian_tests.py`, `TestInferenceMode`. The workaround
`pl.Trainer(inference_mode=False)` is no longer needed.
