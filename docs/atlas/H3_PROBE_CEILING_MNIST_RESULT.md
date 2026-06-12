# H3-PC-B Result — the probe-ceiling battery on the real-body MNIST-rotation shadow (ANSWERED: c)

> Run against the FROZEN addendum `H3_PROBE_CEILING_MNIST_ADDENDUM.md` (2026-06-11; second leg of HS4,
> slate 2026-06-10). Frozen test `test_h3_probe_ceiling_mnist.py` ALL PASS before the run; result JSON
> `results/atlas/h3/probe_ceiling_mnist_result.json` (data receipts sha256(X)=d27fb913…,
> arff=fe4410d8…). Published as a verification receipt (owner decision 2026-06-11; un-promoted; not
> peer-reviewed). Language rule: probe-access asymmetries in a ground-truth
> substrate — no claims about introspection as a mental phenomenon.

## Verdict: (c) BOUNDED-PARTIAL — ceiling raised to 0.561; the banked attenuation was MOSTLY probe-relative

**The re-derived TinyCNN's 32-dim GAP shadow carries rotation θ recoverable to R² = 0.5651 (pool-CV)
/ 0.5606 (once-touched 10k split) by Nyström kernel ridge (γ=0.1, α=0.1, floor δ=0.05) — against a
pre-GAP linearly-readable anchor PRE_pool = 0.6239.** The banked Substrate-B reading "GAP partially
attenuates θ: 0.623 → 0.342" survives only as a statement about the LINEAR readout (battery ridge on
the same reps: 0.3567 — stable). The residual attenuation vs the linearly-readable pre-GAP content is
**only ≈ 0.06 R²**, and outcome (a) "attenuation illusory" missed by **0.009** (0.5651 vs the
PRE_pool−0.05 = 0.574 bar). Pre-registered adjudication of the un-receipted "~0.47 strong-probe"
number: **superseded UPWARD** (ceiling 0.5606 > 0.52).

| member | config | pool-CV | frozen split | floor (δ) |
|---|---|---|---|---|
| P1 ridge | α=1.0 | +0.3567 | +0.3446 | 0.05 (construction) |
| P2 MLP | (128,64) | +0.4923 | +0.5282 | 0.05 |
| P3 kNN | k=20 | +0.5113 | +0.5167 | MEMBER-BLIND (label; reading counts — see below) |
| P4 Nyström | γ=0.1, α=0.1 | **+0.5651** | **+0.5606** | 0.05 |
| P5 GBT | lr=0.05, it=500 | +0.5213 | +0.5106 | 0.05 |

- **Counted set (both bars ≥ BASE+0.05 = 0.407):** P2, P3, P4, P5 — every nonlinear member, all
  replicated on the once-touched split. **Ceiling = 0.5606** (P4).
- **Learning curve (best member / ridge):** 0.427/0.369 → 0.525/0.359 → 0.549/0.362 → 0.565/0.357
  (n=2k→20k). Still rising at 20k ⇒ **the ceiling is a LOWER bound**; the linear readout is flat.
- **MI (descriptive):** KSG peaks at PCA-k=8 (0.286 nats vs null-max 0.014) — consistent with a
  low-dimensional nonlinear θ-code in the GAP.
- **BD-1's fix fired in anger:** P3 kNN was labeled MEMBER-BLIND by the δ-liveness rule (R² concavity
  on a signal-bearing baseline — the injection is redundant with θ-signal kNN already extracts) while
  reading 0.51–0.52 on the real reps. Under the frozen rule ("blindness excuses silence, never
  speech") its reading counted toward the ceiling; under the pre-review draft it would have been
  exempted. The label voids only silence-weight.

## Gates / continuity (all PASS)

cnn_acc 0.792 (0.83±0.05) · banked-probe post-GAP θ 0.333 (0.342±0.10) · permutation control −0.023 ·
PRE_pool−BASE = 0.267 (≥0.15 joint-separation) · both δ-calibrations converged (α=0.0474/0.0702) ·
data shape (70000, 784) · fallback never taken.

## What this re-scopes (correction issued to `H3_POOLED_SHADOW_RESULT.md` §Substrate B)

1. The banked P-B1 "resist: PARTIAL — θ 0.62 → 0.34 post-GAP" is a **linear-probe statement**. The
   battery ceiling 0.5606 (replicated) shows the GAP retains θ in a nonlinearly-decodable code; the
   real attenuation vs the linearly-readable pre-GAP anchor is ≈ 0.06, not ≈ 0.28.
2. The "~0.47 under a strong nonlinear probe" doc-level number (never in the banked JSON) is
   adjudicated **superseded upward** by the calibrated battery.
3. The determine half (y-acc 0.87 post-GAP) is untouched by this audit.
4. Combined with leg 1 (synthetic: ceiling 0.126 from a claimed-0.006 rep) the two-leg pattern is
   uniform: **the lab's weak-battery readouts systematically under-read pooled-shadow recoverability;
   silences certified little until calibrated.** On the real body, the under-read was 0.22 R².

## Honest boundaries

- The body is a continuity-gated SIBLING of the banked CNN (acc 0.792 vs 0.83; banked bytes
  unreachable — BLAS-unpinned origin), and all certificates attach to it.
- Floors are linear-direction deltas in the 32-dim GAP; the kNN blind-label artifact above is the
  known concavity failure mode of that calibration, disclosed not patched.
- Ceiling is a lower bound (learning curve unconverged at 20k). One substrate, one body, one nuisance.
- The non-monotone banked sweep stays out of scope (static ceiling only, per the addendum).

## Files

`scripts/h3_probe_ceiling_mnist.py` (+ frozen test) · `results/atlas/h3/probe_ceiling_mnist_result.json`
· prereg `docs/atlas/H3_PROBE_CEILING_MNIST_ADDENDUM.md` · leg 1: `H3_PROBE_CEILING_{PREREG,RESULT}.md`.
