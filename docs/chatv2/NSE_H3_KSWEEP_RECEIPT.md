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
| 2 | **8** | 0.4265 | 0.4212 | 0.0105 | **0.8688** | 0.179 ✓ |
| 3 | 18 | 0.6208 | 0.7074 | 0.0017 | 0.8823 | 0.193 ✓ |
| 4 | 32 | 0.7351 | 0.6216 | 0.0169 | 0.8821 | 0.192 ✓ |
| 5 | 50 | 0.5173 | 0.4303 | 0.0194 | 0.8838 | 0.194 ✓ |
| 6 | 72 | 0.4550 | 0.3770 | 0.0282 | 0.8849 | 0.195 ✓ |

**m_det bracket: > 6** — no K reaches the frozen 0.99 line; even a 72-dim shadow
leaves the remaining 368 dimensions ~55% unexplained (energy-weighted) with the
typical component at R² ≈ 0.03.
**Regime-2 window: K ∈ {2,3,4,5,6} — the entire tested range.**

## Finding 1 — control is cheap: the decision floor sits at or below K=2

An **8-dimensional** shadow — 4 modes, containing only 4 of the 9 modes of the
band the label is *defined on* — decodes the registered action at 0.869 vs
majority 0.690. Control then saturates immediately: acc is flat 0.869 → 0.885
across the whole sweep; widening the shadow 9× buys +0.016. The AT-3 `K_dec=1`
relay-precedent's static analogue at the deeper cell: **decoding the proxy needs
almost nothing; the sweep never found the control floor.**

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
