# Chat-v2 Phase 2 — Determining-Shadow-Set Spec (Phase 0 freeze)

> 2026-06-03, DRAFT for sign-off — **not run.** This is the runnable Phase-0
> "spec freeze" for the frozen roadmap `docs/SUNDOG_V_ALLELOPATHY.md`. It turns
> that plan-of-record into exact, leakage-clean, reproducible definitions. Per the
> pre-registration discipline, every threshold/rule below may be **tightened or
> replaced before any result is read, never loosened after a sweep is seen.**
> **No verdict-bearing run is admitted until this spec is signed off.**

Reserved implementation: `scripts/chatv2_phase2_shadowset.py`.
Reserved output dir: `results/chatv2/phase2-determining-shadow-set/`.

The headline expectation, restated from the roadmap so the spec cannot drift from
it: the saved pair-XOR latents are **independent by construction**, so Probe 1 is a
**prediction-bank control** whose expected branch is `det_shadow_predicted_null`.
The discovery budget is **Probe 2 (cross-seed transplant)**; a genuine
`k_func << k_state` closure is reserved for a coupled substrate (roadmap Phase 7),
not these bodies.

---

## 1. Inputs (exact)

```
results/chatv2/phase1-seedstab/seed{0,1,2}/bodies/H8_gen.npz    # primary
results/chatv2/phase1-seedstab/seed{0,1,2}/bodies/H8_twin.npz   # twin-floor control
```

Each `.npz` (verified 2026-06-03): keys `bodies`, `z`, `meta`.

| array | shape | meaning |
| --- | --- | --- |
| `bodies` | `(3000, 4, 8, 192)` | `bodies[n, l, i, :]` = fair-readout residual **view** for latent `i` at layer `l`, row `n`. Define `view(l, i) = bodies[:, l, i, :]`, shape `(3000, 192)`. |
| `z` | `(3000, 8)` | `z[:, i]` = binary label of latent `i` (balanced ~0.50). `z_0` is the decision latent; `z_1..z_7` are the non-decision body. |
| `meta` | `()` | 0-d object: the run Cfg. Recorded in the receipt, not used by the probe. |

`H = 8`, `d_model = 192`, `N = 3000`, layers `L = 4`. Untrained bodies are **not
saved**; an untrained-floor control (optional, §10) requires regenerating a
same-init backbone via `chatv2_phase1_contrast.py`'s method and is **deferred**
unless a suspicious positive demands it.

## 2. Determinism, split, resampling (frozen)

- `PROBE_SEED = 0`. All randomness derives from it: `rng = np.random.default_rng(PROBE_SEED)`.
- **Split (per seed-cell, identical index rule across seeds):**
  `perm = np.random.default_rng(PROBE_SEED).permutation(3000)`;
  `TRAIN = perm[:1500]`, `HELD = perm[1500:]`. Disjoint, frozen.
- **Fit on TRAIN, report on HELD.** Standardization stats, readout directions,
  layer selection, subset selection, and every decoder are fit on TRAIN only.
  The only quantities computed on HELD are the final reported metrics.
- **Uncertainty:** `B = 1000` row-bootstraps of the HELD metric for 95% CIs.
  Bootstrap resamples HELD rows; never refits directions/decoders.

## 3. Standardization and atomic shadows (frozen)

- Standardize each `view(l, i)` with **TRAIN** mean/std; apply to HELD (`_std`,
  matching `chatv2_phase0_bodyresist.py`).
- **Readout direction** `w_i` (per latent, at the frozen layer `l*` of §4):
  `LogisticRegression(max_iter=1000).fit(std(view(l*, i))[TRAIN], z[TRAIN, i])`,
  coefficients **unit-normalised** (identical to `fingerprint()`’s `W`).
- **Atomic shadow score** `s_i(n) = std(view(l*, i))[n] · w_i` (scalar). The score
  of latent `i` is the body's projection onto `i`'s readout direction. A shadow set
  `S` contributes the `|S|`-vector `{s_i : i ∈ S}`.

## 4. Layer-selection rule (frozen)

Per seed: `l* = argmax_l mean_i zr_train(l, i)`, where `zr_train(l, i)` is the
4-fold CV accuracy of `LogisticRegression(std(view(l,i))[TRAIN] → z[TRAIN,i])`.
Single `l*` per seed (shared across latents, as in `fingerprint()`), computed on
TRAIN only, then frozen for all reads. Record `l*` per seed in the receipt.

## 5. Shadow sets and subset-selection (frozen)

- A shadow set `S ⊆ {0..7}`, `|S| = k ∈ {1..7}`. Exhaustive over all `C(8,k)`
  subsets (254 total — cheap).
- **Headline selection rule (Attack #2):** at each `k`, pick separate train-only
  headline subsets for the two registered reads:
  - `S_func*_k = argmax_S func_TRAIN(S)`;
  - `S_state*_k = argmax_S state_TRAIN(S)`.
  Report only each selected subset's **HELD** metric as the headline for that
  read at that `k`. The full subset distribution is also recorded so a reader
  can see whether a result is broad or one lucky subset.
- Decoders (latent classifier, score regressor) are fit on **TRAIN** score-vectors
  → TRAIN targets, then applied to HELD score-vectors. No HELD label ever enters a
  fit.

## 6. Targets, metrics, thresholds (frozen)

Two registered targets per subset `S` (predicting the **omitted** set `J = {0..7}\S`):

| target | definition | metric |
| --- | --- | --- |
| **functional** (`k_func`) | per-omitted-latent label recovery: `LR({s_i:i∈S}[TRAIN] → z[TRAIN,j])` applied to HELD, for each `j∈J` | `func(S)` = **mean HELD accuracy** over `j∈J`. Secondary registered variant: predict `XOR_{j∈J} z_j` (base-rate ≈ 0.5). |
| **state** (`k_state`) | omitted score-coordinate reconstruction: ridge `{s_i:i∈S}[TRAIN] → s_j[TRAIN]` applied to HELD, for each `j∈J` | `state(S)` = **mean HELD FVE** over `j∈J` of reconstructing the omitted scores `s_j`. |

Rationale: `s_j` is the body coordinate carrying `z_j`, so reconstructing it is the
saved-body proxy for reconstructing omitted body content; `z_j` recovery is the
omitted-latent functional (roadmap functional #1/#2). Functional #3 (model-output /
next-token) needs a runner extension and is **out of scope for this spec** (a
pointer only; it belongs to roadmap Phase 7's coupled substrate).

**Frozen thresholds** (from the roadmap; may tighten, never loosen):
```
k_func collapse:  func(S_func*_k)   >= 0.70
k_state collapse: state(S_state*_k) >= 0.60   (FVE)
control null:     metric <= chance + 0.12   (acc) / <= 0.12 FVE
```

**Determining numbers:** `k_func` = smallest `k` with
`func(S_func*_k) >= 0.70` (else `none<=7`); `k_state` = smallest `k` with
`state(S_state*_k) >= 0.60` (else `none<=7`); `k_det` = smallest `k` meeting
**either** threshold (else `none<=7`). Report all three per seed + aggregate.

## 7. Probe 1 branch table (frozen — same-seed control)

| branch | condition | reading |
| --- | --- | --- |
| `det_shadow_predicted_null` | no `k≤7` meets either threshold **and** controls pass | **expected**: independent latents do not determine each other |
| `det_shadow_functional_closure` | `k_func < k_state`, or `k_func` exists while `k_state` does not | **suspect on this toy** — must pass the §10 leakage/overfit/permutation attack before any reading; otherwise `det_shadow_void` |
| `det_shadow_state_collapse` | `k_func ≈ k_state < 8` | also unexpected here; determining-set, not closure (still attack-gated) |
| `det_shadow_partial` | one metric passes but a companion read/control fails | constrained in one basis; not a headline |
| `det_shadow_void` | split/leakage/shape/control contract fails | no result; repair |

## 8. Probe 1b — paired-fiber / boundary-layer audit (frozen)

For the headline subsets `S_func*_k` and `S_state*_k` at each `k` (and the
full-`S` reference):

- **Shadow-near pairs:** sample up to `200_000` random HELD row-pairs; `ε_S` =
  the **1st percentile** of their Euclidean distance in `score_S` space (frozen
  quantile `q_eps = 0.01`). A pair is shadow-near if `‖score_S(a)−score_S(b)‖ ≤ ε_S`.
- **Non-injective requirement:** keep only pairs whose omitted body target differs
  by `≥ δ_body`, `δ_body` = **median** omitted-score distance over HELD pairs
  (frozen). These are shadow-near but body-far.
- `D_witness` = fraction of kept pairs whose registered functional **disagrees**
  (predicted omitted label differs on ≥ 1 omitted latent, or `|Δfunc| > 0.5`).
- **Boundary stratification:** bin disagreements by decision/readout margin
  `min_j |s_j|`; report the disagreement rate in the lowest-margin decile vs rest.

| branch | condition | reading |
| --- | --- | --- |
| `paired_fiber_closure_positive` | body differs, functional agrees except a thin margin set | local functional closure (NSE-style) |
| `paired_fiber_boundary_only` | disagreements concentrate in the lowest-margin bin | a boundary layer, not global insufficiency |
| `paired_fiber_conflict` | body and functional differ away from the margin | `S` not functionally sufficient |
| `paired_fiber_deferred_coverage` | `< 200` kept pairs under the frozen radius | no verdict; widen only in a new spec |

## 9. Probe 2 — cross-seed transplant (frozen headline)

Six directed pairs `(a→b)`, `a≠b ∈ {0,1,2}`. Question: does seed-`a`'s shadow
basis read seed-`b`'s body **without target-label refit**? Per-latent transplant
recovery is the unit: apply `a`'s direction `w_i^{(a)}` to `b`'s body and decode
`b`'s `z_i`.

**Tier order (hard — only Tier 1 is allelopathy-positive):**

1. **Direct transplant.** `a`'s layer index `l*_a`, `a`'s standardization stats,
   `a`'s directions `w_i^{(a)}`, and the **`a`-fit** decoder are all applied to
   `b`'s `view(l*_a, i)`[HELD]. No `b` labels, no `b` layer selection, no `b`
   decoder refit. `metric_ab` = mean HELD accuracy over `i` of recovering `b`'s
   `z_i`.
2. **Unlabeled calibration.** As Tier 1 but standardize `b`'s `view(l*_a, i)`
   with `b`'s own TRAIN mean/std (distribution calibration only; still `a`'s
   layer, directions, and decoder). A weaker coordinate-shape pass.
3. **Label-aware / Procrustes ceiling.** Orthogonal Procrustes between the two
   seeds' direction frames using paired latents, choose `b`'s label-selected
   `l*_b`, or refit the decoder on `b`'s labels. **Upper bound only**, always
   labelled ceiling.

**Pass:** a directed pair passes a tier iff `metric ≥ 0.70` (well above the 0.50
chance of `b`'s balanced latents). Aggregate over the 6 pairs.

| branch | condition | reading |
| --- | --- | --- |
| `cross_seed_direct_pass` | Tier-1 passes on `≥ 4/6` directed pairs | independently trained bodies share coordinate structure; the source shadow reads the target |
| `cross_seed_calibrated_only` | Tier-1 fails, Tier-2 passes `≥4/6` | shared only after distribution calibration; weaker |
| `cross_seed_ceiling_only` | only Tier-3 passes | abstract isomorphism, not a transplant |
| `cross_seed_no_transfer` | no non-leaky tier passes | same-seed geometry is not cross-seed readable |
| `cross_seed_void` | pair/shape/contract fails | no result |

## 10. Negative controls (frozen — all must pass before any reading)

1. **Label-leakage guard.** Assert no HELD label enters any direction fit,
   standardization, subset selection, or decoder fit (split-provenance asserts).
2. **Permutation null.** Independently shuffle `z` columns; every metric must fall
   to `≤ chance + 0.12`. Run for both probes.
3. **Random-direction null.** Replace `w_i` with random unit directions; metrics
   must fall to the registered null.
4. **Twin-floor.** Re-run the full Probe-1 sweep on the saved **twin** bodies. A
   positive that appears equally on twin is architectural floor, not gen-specific;
   the receipt reports `gen − twin` for every headline number.
5. **Untrained-floor (deferred).** Only regenerated + run if a suspicious gen
   positive survives 1–4.
6. **Base-rate vacuity.** Every functional target must have a pinned base rate near
   `0.5` (balanced `z`); a degenerate target is `deferred`, not positive.
7. **Bootstrap stability.** Headline `k_*` must be stable across `≥ 90%` of the `B`
   bootstraps, else report as unstable.

## 11. Receipt schema (Phase 3 — `PHASE2_DETERMINING_SHADOW_SET_RECEIPT.md`)

```
- spec file + sha; runner sha; git commit + dirty-state
- PROBE_SEED, split sizes, B, frozen thresholds (echoed)
- per seed: l*, d_dec (from phase1 receipt, cross-check), k_func, k_state, k_det
- aggregate Probe-1 branch (+ per-seed)
- best subsets S_func*_k / S_state*_k at each k AND the subset distribution summary
- twin-floor table: gen vs twin headline metrics; gen−twin
- paired-fiber: D_witness, boundary-layer split, Probe-1b branch
- cross-seed: per-pair tier metrics (6 pairs), aggregate Probe-2 branch + named tier
- controls table (perm null, random-direction null, leakage asserts, base rates)
- measured wall-clock + timing-smoke note
- allowed/forbidden language pulled verbatim from the roadmap
- the exact CSV/JSON data contract for the Phase-4 SVG generator
```

Expected outputs:
```
results/chatv2/phase2-determining-shadow-set/
  same_seed_subset_sweep.csv          # every (seed, k, subset, func, state, controls)
  same_seed_kfunc_kstate_summary.csv  # per-seed k_func/k_state/k_det + branch
  paired_fiber_boundary_audit.csv
  same_seed_controls.csv              # incl. twin-floor + perm + random-dir nulls
  cross_seed_transfer.csv             # per directed pair, per tier
  cross_seed_tiers.csv
  branch_adjudication.md
```

## 12. Timing smoke + inline rule (frozen)

Before the full run: smoke seed 0 only, `k ∈ {1, 7}`, no bootstrap; record
wall-clock and linearly extrapolate to the full `(3 seeds × 254 subsets × targets
× B)` cost. Saved-body LR/ridge fits on `(1500, ≤7)` inputs are tiny; the full run
is expected **well under 10 minutes** and should run inline. If the smoke
extrapolates `> 10 min`, run in the background per the standing inline-compute rule.
No verdict run starts until the smoke's leakage/shape asserts pass.

## 13. Run order and gates

1. **Phase 0 (this spec):** sign-off. Freeze.
2. **Smoke (§12):** asserts + timing.
3. **Phase 1:** same-seed sweep (gen + twin). Adjudicate Probe-1 + Probe-1b.
   **Expected `det_shadow_predicted_null`.**
4. **Phase 2:** cross-seed transplant. Name the tier.
5. **Phase 3:** receipt.
6. **Phase 4 (gated):** SVG only after the receipt, generated from its CSV/JSON;
   a null branch must render as a null.

## 14. Frozen-parameter lock (pre-registration)

| parameter | value |
| --- | --- |
| `PROBE_SEED` | 0 |
| split | 1500 train / 1500 held, `default_rng(0).permutation(3000)` |
| bootstraps `B` | 1000 |
| layer rule | `l* = argmax_l mean_i CV4_train(view(l,i)→z_i)`, per seed |
| readout | unit-normalised LR coef, TRAIN-fit, at `l*` |
| subset scope | exhaustive `|S|=1..7` (254); train-selected `S_func*_k` / `S_state*_k` |
| `k_func` threshold | mean omitted-latent HELD accuracy `≥ 0.70` |
| `k_state` threshold | mean omitted-score HELD FVE `≥ 0.60` |
| control null | `≤ chance + 0.12` (acc) / `≤ 0.12` (FVE) |
| `q_eps` | 0.01 (1st-percentile shadow-distance radius) |
| `δ_body` | median omitted-score distance |
| paired-fiber coverage floor | `≥ 200` kept pairs |
| cross-seed pass | tier metric `≥ 0.70`; branch needs `≥ 4/6` directed pairs |
| bootstrap stability | headline `k_*` stable in `≥ 90%` of bootstraps |

**These may be tightened or replaced before any result is read; never loosened
after a sweep is seen.** Functional #3 (model-output) and the coupled-substrate
closure bracket are explicitly **out of scope** here (roadmap Phases 6/7).

---

## Amendment 1 (2026-06-03) — selection-spectral-gap reliability, Procrustes/subspace ceiling, selection-corrected null

**Why (disclosed, not a silent edit).** A first run surfaced two things the frozen
spec under-specified, and the isotrophy lane's v0.19/v0.20 "seventh projection
lesson" (`CROSS_SUBSTRATE_NOTES.md` §7.2) supplies the principled fix. (i) The
absolute `k_func`/`k_state` thresholds are **trippable by best-of-254 selection
optimism** — random directions reach max-func ~0.60–0.73 — so they are not a valid
null. (ii) `cross_seed_no_transfer` (Tier-3 per-latent refit ≈ 0.51 on all 6 pairs)
cannot, by itself, distinguish *no shared structure* from *shared subspace with a
reparameterization-fragile per-latent frame*. The isotrophy export — *"for an
argmax-selected shadow, compute the selection spectral gap (… **LLM residual-stream
eigengap** …), read its LOW TAIL as the label-blind a-priori reliability predictor;
decide median/tail/threshold before collapsing a region to one number"* — is
transplanted here. This is a **tightening** prompted by cross-substrate transfer;
the headline (cross-seed) is re-read with the stronger ceiling. Tighten-not-loosen
still binds. **Tier-1/2 source-layer rigor (l\*_a) is unchanged.**

### A1. Selection spectral gap + frame-spread (label-blind, per seed × latent)
- **`frame_spread_i`** (direct fragility): refit the readout direction `w_i` on
  `B_g = 30` bootstrap resamples of TRAIN; `frame_spread_i = 1 − mean_{pairs}
  |cos(w^{(b1)}, w^{(b2)})|`. High spread = reparameterization-fragile.
- **`eigengap_i`** (label-blind a-priori): top relative gap of the standardised
  body-view covariance, `(λ_1 − λ_2)/λ_1`, the chatv2 read of "residual-stream
  eigengap." (Honest caveat: a discriminative-conditioning variant is a refinement
  if the covariance gap fails to track `frame_spread`.)
- **Region summary:** the **low tail** `q10` of each over the seed×latent
  population (the fragile sub-population), per the v0.20 tail lesson — not a median.

### A2. Spectral-gap → fragility → transfer test
Report `rho(eigengap, frame_spread)`, `rho(frame_spread, transfer)`,
`rho(eigengap, transfer)` over the seed×latent population, where `transfer_i` =
mean Tier-3b recovery across directed pairs. **Power caveat (pre-registered):** if
cross-seed transfer is null-floored (all ≈ chance, no variance), the correlation is
under-powered; the informative read is then **descriptive** — does the population
sit in the *fragile regime* (large `frame_spread`, small `q10(eigengap)`)? A
uniform fragile regime makes `cross_seed_no_transfer` **consistent with the
fragility principle** (chatv2 as a 4th substrate, descriptive tier); a non-fragile
regime with no transfer is the **stronger** negative.

### A3. Cross-seed ceiling upgrade — separate "no structure" from "fragile frame"
Add, per directed pair `(a→b)`:
- **`subspace_overlap`** = mean `cos²` of the principal angles between `span(W_a)`
  and `span(W_b)` (the two ≤8-dim readout frames). `~1` = same encoding subspace;
  `~0` = orthogonal subspaces.
- **Tier 3b (subspace ceiling):** decode `b`'s `z_i` from **all 8** `a`-transplanted
  scores with a `b`-label-refit decoder (vs Tier-3's single score). If Tier-3b ≫
  Tier-3, `a`'s *subspace* carries `b`'s latents though no single direction aligns.

New cross-seed readings (supersede the bare branch where they apply):
| condition | reading |
| --- | --- |
| `subspace_overlap` high ∧ Tier-3b passes ∧ Tier-1 fails | **shared subspace, fragile per-latent frame** — non-transfer is reparameterization fragility, not absence of structure |
| `subspace_overlap` low ∧ Tier-3b fails | **genuinely different encoding subspaces** — the strong negative |
| any tier ≥0.70 on ≥4/6 (Tier order) | the §9 branch as written |

### A4. Probe-1 selection-corrected null (replaces the absolute threshold for reading)
- **Null floor:** `R = 30` random-direction draws (train-std, randomized `w`);
  each draw's best-of-254 `func`/`state` HELD metric forms the null distribution.
- **Pass rule:** gen (or twin) shows determination only if its best-subset metric
  exceeds the **95th percentile** of the null distribution — not the absolute 0.70.
- The §7 `det_shadow_*` branch is read against this selection-corrected null. The
  `eigengap` q10 (A1) is reported alongside as the label-blind companion predictor.
- The absolute `0.70/0.60` numbers stay as a secondary descriptive column only.

### A5. Amendment frozen parameters
| parameter | value |
| --- | --- |
| `B_g` (frame-spread bootstraps) | 30 |
| `eigengap` | `(λ_1 − λ_2)/λ_1` of body-view covariance |
| region summary | low tail `q10` |
| `subspace_overlap` | mean `cos²` principal angles of `span(W_a)`,`span(W_b)` |
| Tier 3b | 8-score `b`-refit decoder |
| null draws `R` | 30; pass = `> 95th pct` of null |

*Headline read with A3 (cross-seed) + A4 (Probe-1 null); A1/A2 are the cross-substrate
reliability read. Re-run is admitted under this amendment.*

---

*Sundog Research Lab — chatv2 Phase 2 spec, Phase-0 draft. Internal; gated on the
frozen `SUNDOG_V_ALLELOPATHY.md` roadmap. No verdict-bearing run until sign-off.*
