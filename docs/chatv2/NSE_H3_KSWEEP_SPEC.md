# NSE-H3 State-Recon K-Sweep at G=675 — Spec (frozen)

> 2026-07-07. The registered successor to `NSE-H3-GLOBAL-REGIME2-NONMARGINAL`
> (`NSE_H3_GLOBAL_GAUGE_RECEIPT.md`): K=3 gave FVE 0.632 at the matched-Re cell —
> far below the 0.99 determining line. This sweep measures **FVE(K) and
> control(K) for K = 2..6**: the m_det bracket at the deeper cell and the width
> of the non-marginal regime-2 window. Mirrors the G=200 robustness-wave K-window
> (`lock_v5_k2..k6` + state-recon), done as **one capture run + post-processing**
> instead of five integrations. **Frozen before any sweep number exists.**
> Non-promotional; finite, proxy-relative, no promotion, no-publish.

## 1. The capture run (one integration; the only compute-heavy step)

New **non-verdict-bearing** preset `recon_sweep_g675_kf3` (the AT-2 idiom — not
in `VERDICT_BEARING_PRESETS`, so its in-run verdict is `SMOKE_ONLY` by
construction; only the side artifacts are read):

- the `fallback_v7_g675_kf3` trajectory exactly (same seed 20260528, burn-in
  200k, physics, dt, grid) with **`k_signature = 6`** (72-dim emitted shadow);
  cal 50k / gap 5k / **adj 50k** samples at stride 50, lookahead 500.
- run with `--adjudicator twin-state-adaptive` (captures + exports `samples.npz`:
  72-dim signatures, high modes, in-run actions) **and `--at2-export`** (per-step
  **K=3-band** `e_low_k3` + `adj_starts`/`calib_starts` + the `cols_k1..k6`
  column maps into the wide signature).
- the physics step is k_signature-independent, so the trajectory is
  bit-identical to the banked fallback run; the adjudication block is the
  **stride-1 prefix** (first 50k) of the banked 200k adj samples.

## 2. Post-sweep (all post-processing; the frozen reads)

**Labels (fixed across the sweep):** the in-run actions are K=6-band — **not
used**. Labels are recomputed from the exported K=3-band `e_low_k3` by the
registered pi_hat construction (lookahead-max over [s, s+500], threshold = q0.70
of the calibration-block maxes) — the *cell's registered proxy, identical at
every K*. The sweep varies the **shadow only**.

**Per K ∈ {2, 3, 4, 5, 6}:**
- `Φ_K` = the `cols_kK` subvector of the 72-dim signature; `Q_K` = the
  complementary low-6 components ⊕ the captured high modes (same per-mode
  normalization throughout — `mode_vector` is uniform); `comp_k` weights from
  the stepper geometry (k_f = 3, force-insert respected).
- **State half:** the untouched frozen `aggregate_state_recon` (perm gate per K).
- **Control half:** the registered global control read (HGB classifier, 400-gap
  block split, majority + permuted-label controls) of the **fixed K=3 label**
  from `Φ_K`.

## 3. Internal regression gates (preconditions, frozen)

1. **Label fidelity:** recomputed `e_max` must equal the banked fallback
   manifest's 0.6946 (same trajectory, same calibration instants) to 1e-6, and
   adj-block damp must match the banked 0.30674 to ±0.005.
2. **K=3 rung reproduces the global-gauge receipt** on this new sample subset
   (stride-1 prefix vs the receipt's stride-4): `|FVE − 0.6322| ≤ 0.05` and
   `|control acc − 0.8866| ≤ 0.05`. Fail ⇒ `NSE-H3-KSWEEP-APPARATUS-REJECTED`,
   no sweep read.
3. Mode bookkeeping asserted (`cols_kK` ⊂ low-6; dims consistent) in the tool's
   self-test before any real read.

## 4. Outcome (a measurement, not a pass/fail)

Primary verdict **`NSE-H3-KSWEEP-WINDOW-MEASURED`**: the FVE(K) and control(K)
curves, with:
- **m_det bracket:** smallest K with `FVE ≥ 0.99` (frozen line), or
  **"m_det > 6"** if the curve never crosses — i.e. even a 72-dim shadow leaves
  the state non-marginally under-determined;
- **regime-2 window:** `{K : control powered ∧ FVE < 0.99}` — its width at
  G=675 vs the G=200 anchor (where K=3 sat at the marginal edge);
- per-K permutation gates typed (`ESTIMATOR-INVALID` rungs excluded, reported).

Interpretation lines frozen now: control expected powered for K ≥ 3 (Φ_K ⊇
Φ_3's information); **K=2 is the live control question** (does a sub-proxy
shadow still decide? — the AT-3 K_dec=1 precedent at G=200 says small shadows
can); FVE expected monotone-increasing in K (violations reported, not smoothed).

## 5. Claim boundary

A measured window says: at the matched-Re G=675 cell, shadows of dimension
2K² ∈ [8, 72] leave the unresolved state genuinely under-determined (per the
frozen estimator) up to the measured bracket, while the registered proxy stays
decodable from shadows down to the measured floor. One cell, one forcing move,
estimator-relative FVE, proxy-relative labels, 32² Galerkin, sampled support. No
inertial-manifold theorem, no promotion, no infinite-dimensional claim.

## 6. Deliverables + cost

- Harness: the 2-site additive preset above (argparse choices + `build_config`;
  `VERDICT_BEARING_PRESETS` untouched — non-verdict by design). Self-test +
  config echo + smoke before the capture.
- Capture: ~5.2M steps ≈ 45–55 min, agent-run bg (quiet box).
- `scripts/nse_h3_ksweep_g675.py`: `--self-test` (mode bookkeeping: K-subvector
  reconstruction bit-matches a directly-built K-signature; label recomputation
  on synthetic series), `--run` (gates → sweep → receipt table). Post fits
  ≈ 10–25 min per K ⇒ 1–2 h bg total.
- Receipt: `NSE_H3_KSWEEP_RECEIPT.md`.

Cross-refs: `NSE_H3_GLOBAL_GAUGE_SPEC/RECEIPT.md` (the K=3 point + machinery),
`PDE_C1_ROBUSTNESS_WAVE.md` (the G=200 K-window precedent),
`AT2_HARNESS_SIGNOFF_REQUEST.md` (the wider-emitted-shadow capture idiom + the
`cols_k*` export), `results/proof/c1-h3-kf3-g675-fb-*/manifest.json` (e_max /
damp fidelity anchors).

---

> **FINAL (2026-07-10): `NSE-H3-KSWEEP-WINDOW-MEASURED`**
> (`NSE_H3_KSWEEP_RECEIPT.md`). Gates PASS (e_max to 1.2e-8; K=3 rung reproduces
> the global-gauge receipt at |d| 0.011/0.004). **m_det bracket > 6; regime-2
> window = the entire tested range K ∈ {2..6}.** Control saturated from an
> 8-dim shadow up (0.869 → 0.885, all powered); state non-marginal at every K
> (eq-wt median R² ≈ 0.01–0.03 throughout). FVE(K) non-monotone as the frozen
> violation clause anticipated — target-composition effect (growing shadows
> absorb the predictable head, leaving a freer remainder), reported not
> smoothed. The sharpest measured form of the regime-2 asymmetry in the lane.
