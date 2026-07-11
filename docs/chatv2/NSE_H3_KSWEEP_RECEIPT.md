# NSE-H3 State-Recon K-Sweep at G=675 — Receipt

> 2026-07-10. Read of the frozen `NSE_H3_KSWEEP_SPEC.md`: one K=6-shadow capture
> integration (bit-identical fallback trajectory) + post-sweep. Non-promotional;
> finite, estimator-relative, proxy-relative, no promotion, no-publish.

## Gates (preconditions): PASS

- **G1 label fidelity:** recomputed `e_max` 0.6946284175 vs banked 0.6946284051
  (|d| = 1.2e-8 — float32 export rounding, within the 1e-6 registered tolerance);
  damp 0.30856 vs 0.30674 (adj block = stride-1 prefix of the banked 200k) ✓.
- **G2 K=3 rung reproduces the global-gauge receipt:** FVE 0.6208 vs 0.6322
  (|d| 0.0114), acc 0.8823 vs 0.8866 (|d| 0.0043) — both ≤ 0.05 ✓.

## The measurement (all permutation gates clean at every K)

| K | shadow dim | FVE_vw(Q_K) | enstrophy | eq-wt median R² | control acc (maj 0.6897) | margin |
| --- | --- | --- | --- | --- | --- | --- |
| **1** | **2** (the forced mode (0,3) alone) | **−0.0064** | −0.0074 | −0.0035 | **0.8245** | **0.135 ✓** |
| 2 | 8 | 0.4265 | 0.4212 | 0.0105 | 0.8688 | 0.179 ✓ |
| 3 | 18 | 0.6208 | 0.7074 | 0.0017 | 0.8823 | 0.193 ✓ |
| 4 | 32 | 0.7351 | 0.6216 | 0.0169 | 0.8821 | 0.192 ✓ |
| 5 | 50 | 0.5173 | 0.4303 | 0.0194 | 0.8838 | 0.194 ✓ |
| 6 | 72 | 0.4550 | 0.3770 | 0.0282 | 0.8849 | 0.195 ✓ |

**m_det bracket: > 6** — no K reaches the frozen 0.99 line; even a 72-dim shadow
leaves the remaining 368 dimensions ~55% unexplained (energy-weighted) with the
typical component at R² ≈ 0.03.
**Regime-2 window: K ∈ {1,2,3,4,5,6} — the entire tested range, floor to ceiling.**

## Finding 1 — the control floor IS the forcing coordinate (K=1 addendum, spec §7)

The pre-registered K=1 rung closes the floor: at this cell the force-insert rule
makes the K=1 shadow **exactly the forced mode (0,3) — 2 dimensions** — and it
**decodes the registered action at 0.8245 vs majority 0.6897 (margin 0.135,
powered, permutation-clean)** while determining *nothing* of the state
(FVE −0.006 ≈ 0 on the 438 remaining dims). Control then saturates fast:
0.824 → 0.869 → 0.882, flat thereafter; widening the shadow 36× beyond K=1 buys
+0.06. The static analogue of AT-3's dynamic `K_dec = 1`, at the non-marginal
cell: **the decision is anchored to the forcing; the state is anchored to the
attractor.** Fence: the label's own band contains (0,3), so partial
self-correlation is mechanistically expected — the non-trivial content is that a
*single coordinate* clears the powered margin on a lookahead-max functional of a
9-mode band.

## Finding 2 — the state stays free at every tested width (with the honest subtlety)

FVE(K) is **non-monotone** (0.43 → 0.62 → 0.74 → 0.52 → 0.46), the violation the
spec pre-registered for reporting rather than smoothing. The mechanism is
target composition, not estimator noise: growing K *absorbs* the large-scale,
high-variance, partially-predictable modes into the shadow, so the remaining
Q_K is increasingly the genuinely free small scales — the variance-weighted FVE
of the *remainder* falls. Read correctly, the falling tail **strengthens** the
non-marginality: strip the predictable head into a 72-dim shadow and what's
left is *more* free, not less (eq-wt median ≈ 0.01–0.03 throughout — the typical
unresolved component is unpredictable at every K). Because Q_K is a different
target at each K, the m_det statement is the bracket ("no crossing through
K=6"), never a claim about the curve's shape.

## The regime-2 asymmetry, sharpest form yet

At the matched-Re cell the two sides of the witness are now measured as flat and
far apart across a 9× range of shadow width: **control saturated from 8
dimensions up; state non-marginally under-determined through 72 dimensions.**
The G=200 anchor's K-window sat at the marginal edge (K=3 ≈ determining); at
G=675 the window is the whole tested range with no bracket in sight.

## Caveats (load-bearing)

FVE is estimator-relative (frozen HGB instrument; permutation-gated per K;
cross-K on identical samples). FVE(K) values compare different targets (the
composition effect above) — the bracket, the eq-wt floor, and the control curve
are the invariant statements. One cell, one forcing move, proxy-relative labels,
32² Galerkin, sampled support. No inertial-manifold theorem, no m_det claim
beyond "> 6 at this cell under this estimator," no promotion.

Cross-refs: `NSE_H3_KSWEEP_SPEC.md`, `NSE_H3_GLOBAL_GAUGE_RECEIPT.md` (the K=3
point), `PDE_C1_ROBUSTNESS_WAVE.md` (G=200 K-window comparator),
`AT3_NUDGING_LEDGER_RECEIPT.md` (K_dec=1, the dynamic sibling of Finding 1),
`results/proof/c1-h3-g675-ksweep/{ksweep_manifest.json,at2_export.npz,samples.npz}`.
