# Sundog Certificate Problem — Syndrome Certificate v5 Distributional-Band Slate (MHK-v5)

Status: **FROZEN — stage-1 slate contract; NOT executed.** No v5 precal or frozen-target run may
execute outside this contract. No v5 frozen-target run may execute until the multi-seed median
pre-calibration has produced the stage-2 locked band, the tightness gate (W≤3.0) holds for every
scored variant, and the v3-rung method validation passes. (Freeze-before-execute, per the lane.)

Date opened: 2026-06-07
Stage-1 frozen: 2026-06-07

This is the **deeper distributional model** v4 named as open work. It is the survival-analysis
successor to [`SUNDOG_CERTIFICATE_SYNDROME_V4_SLATE.md`](SUNDOG_CERTIFICATE_SYNDROME_V4_SLATE.md)
(median point-prediction → in-sample retrofit validated, but missed the fresh R2′ regime's Stern
`l8` by 2.77×; receipt [`receipts/2026-06-07_certificate_syndrome_v4.md`](receipts/2026-06-07_certificate_syndrome_v4.md)).
Design selected by a 4-approach / 3-judge design panel (unanimous): **MHK-v5** = a minimal,
auditable multi-seed Kaplan-Meier-median **band** (spine) + a survival-difference cross-attacker
test (graft). Parametric mixture / EVT machinery was **rejected** for adding misspecification +
fitting-choice p-hack surface the lane penalizes.

## What v5 fixes (the v4 failure, and why a band)

v4 locked a single **point** median per variant and graded `|frozen/point| < 2`. On the fresh
R2′ regime that gave LB 1.43× (pass), Stern `l9` 1.98× (pass), Stern `l8` 2.77× (**fail**). All
three "misses" are the lane's *documented* seed/sample wander (1.43–2.77×), not model error: the
**same** regime's median moves 1.43–2.77× between target samples, and even the cap-immune LB
control diverges 1.43×. A single point prediction **cannot** be right for these regimes. v5
replaces the point with an **empirically-measured band** over `K=8` independent same-size precal
seeds, so the wander the v4 point could not own becomes the band's content — while a hard
tightness ceiling keeps it a real falsifiable test, not an "anything within 10×" non-test.

v5 reuses the **validated** attacker layer unchanged (`v2.make_code`,
`v2.sample_frozen_manifest`, `v2.attacker_run` — rank-valid iteration counting; `per_iter`
base+enum cost model from v4). **Only the statistical / pre-registration layer is new.** It does
**not** re-open v1–v4 science.

## The MHK-v5 estimator and band

**Core statistic** (per regime × variant × seed): convert each throwaway target's `attacker_run`
output to an `(ops, event)` pair — `event=1` iff a valid witness was found (`He*=z ∧ wt(e*)≤τ`,
rank-valid), else `event=0` (right-censored at the common op horizon `B_common`). Build the
**Kaplan-Meier** survival `Ŝ(b) = ∏_{event b_j≤b}(1 − d_j/n_j)` in **op units**. The seed-median
`M_s` uses the lane's midpoint convention: if `Ŝ` crosses below 0.5 at an observed event, use that
event op time; if `Ŝ=0.5` on an observed plateau, use the midpoint between the event that reaches
0.5 and the next observed event. Under zero censoring at `T=64`, this **reduces exactly to v4's
`_c50_ops = 0.5·(order32 + order33)`**, so it is byte-comparable to the existing estimator and to
the v3 ground-truth rungs. If `Ŝ(B_common) > 0.5`, or if the 0.5 plateau's upper endpoint is
censored rather than observed, the seed-median is right-censored →
`precal_insufficient_seed`.

**The locked band** (per variant, over `K=8` seeds): `raw_band = [min_s M_s, max_s M_s]`;
**`locked_band = [C_lo, C_hi] = [min_s M_s / g, max_s M_s · g]`**, guard `g = 1.25`
(symmetric in log-space; just above `√1.43 = 1.20`, the cap-immune LB half-wander, to buy
residual coverage toward a ≤10% out-of-band target at `K=8`). `W = C_hi / C_lo`.

**The single graded object per variant is the BAND.** The mean is logged **diagnostic-only**
(v3 had this backwards). No band wider than `W_max = 3.0` is admitted.

## Locked prediction form (`prediction_lock_v5.json`)

Schema `pvnp-certificate-syndrome-v5-prediction-lock`, written at the **stage-2 freeze** (after
precal, before any frozen target is drawn). Durable path:
`docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json`.

**Global (stage-1 frozen, before any number is computed):** `regime_id`,
`frozen_regime={n,k,w,τ,p,l-set}`, `estimator="km_median_op_units_common_cap"`, `K=8`,
`T_pre=64`, `T_frozen=64`, the explicit `precal_target_seeds` for each regime, the primary
R2′ `frozen_target_seed` (validation regimes have no frozen target), all seeds mutually disjoint
and disjoint from `code_seed`, `code_seed`, `base_fit={α,β}` (v4 mechanism),
`per_iter_ops` per variant, `C_MULT=12`, the `B_common` rule + frozen scalar, guard `g=1.25`,
`W_max=3.0`, the pairwise `B_star_pair` rule + frozen scalars, the cross-attacker verdict rule,
`PYTHONHASHSEED=0`, `code_digest` (asserted identical across all 8 precal seeds **and** the
frozen run).

**Per variant** `v ∈ {lb, stern_l8, stern_l9}` (computed at precal, then frozen):
`M_seed[1..8]`, `raw_band`, `locked_band=[C_lo,C_hi]`, `W`, `per_seed_censored[1..8]`,
`per_seed_target_max_over_min` (heterogeneity — must show the 122×/458×/2750×-class spreads),
`per_iter_ops_v`, `max_B_v = ceil(B_common / per_iter_ops_v)`, `Ŝ_v(B_star_pair)` for each
registered LB-vs-Stern pair, `mean_based_C_DIAGNOSTIC_ONLY`, and **`precal_insufficient`** (true
iff any seed median is censored OR `W > 3.0`). An over-wide or median-censored variant is
**declared `precal_insufficient` and dropped — never widened to pass.**

## Cross-attacker fairness (fixes the v4 asymmetry)

v4 bug: LB capped at `max_B=6888` (0/64 censored) while Stern was capped **lower** at 2975/5001
(10–13/64 censored) — a lower op horizon than LB's worst case — biasing Stern medians low. Two
locked mechanisms:

1. **Common op horizon `B_common`** = `max_v ceil(C_MULT · N_analytic_v) · per_iter_ops_v`
   (`C_MULT=12`) — the **largest** per-variant op budget, so no variant is capped below another's
   worst case. Each variant's iteration cap derives from it: `max_B_v = ceil(B_common /
   per_iter_ops_v)`. Every variant spends the **same ops**; the per-iteration cost difference is
   folded in **before** the cap. At precal, require `Ŝ_v(B_common) ≤ 0.5` (median reached with an
   observed upper endpoint) for every scored variant. If not, the variant is
   `precal_insufficient`. **Do not raise `C_MULT` after seeing precal numbers**; changing it
   requires a new slate id.
2. **Survival-difference + log-rank** (the graft): the winner is **not** decided by separately-
   censored medians nor band overlap, and Stern does **not** get a free per-target min over
   `l∈{8,9}`. Cross-attacker verdicts are pairwise:
   `LB vs stern_l8` and `LB vs stern_l9`, each on the same `B_common`. For each admissible pair,
   lock `B_star_pair = min(1.5 · max(median_s M_s(LB), median_s M_s(Stern_l)), 0.75 · B_common)`.
   This keeps the comparison near the slower median while avoiding the far tail; if the cap binds,
   record `B_star_cap_active=true` and do not retune. Primary
   `ΔS(B_star_pair) = Ŝ_LB(B_star_pair) − Ŝ_Stern_l(B_star_pair)`. Bootstrap CI over shared
   seed/target resample indices (2000 resamples, seed-pinned). Secondary: log-rank on pooled
   `(ops,event)` for `b≤B_common`. Pair verdict `lb_wins`/`stern_wins` only if `ΔS` CI excludes 0
   AND log-rank **`p<0.025`** (Bonferroni for the 2 pairwise comparisons `LB vs stern_l8` and
   `LB vs stern_l9`, holding the family error at α≈0.05); else `indistinguishable_at_op_budget`.
   The dual gate (`ΔS` CI excludes 0 **and** the Bonferroni log-rank) means each pair's joint
   significance is `≤0.025`, so the two-pair family false-positive stays `≤0.05`.

R2′ family verdict derives from the pair verdicts: `stern_wins` iff at least one admissible Stern
variant has pair verdict `stern_wins` and no admissible Stern variant has pair verdict `lb_wins`;
`lb_wins` iff every admissible Stern variant has pair verdict `lb_wins`;
`indistinguishable_at_op_budget` iff every admissible pair is indistinguishable; `mixed_variant`
iff one fixed Stern variant wins and another loses. If LB or both Stern variants are
`precal_insufficient`, R2′ is `precal_insufficient` and frozen scoring is blocked.
Any pair-verdict combination not matched by the four clauses above (e.g. one Stern pair `lb_wins`
and the other `indistinguishable`) conservatively defaults to `indistinguishable_at_op_budget` — it
asserts no Stern win and no clean LB sweep. This default is applied identically to the locked and
the frozen pair verdicts, so GATE-3's locked==frozen match check is unaffected.

**Reported bound (distinct from the verdict).** `C_best := min over admissible variants of the
frozen KM-median `C@50%`` (the best attacker an adversary would pick), with its band carried
alongside; this is the reported capacity upper bound and is computed independently of the pairwise
`ΔS`/log-rank verdict (which only decides *whether* Stern beats LB, not the bound).

## Three falsification gates (all thresholds locked before frozen targets are seen)

- **GATE-1 — per-variant coverage (the real test):** for each *admissible* variant, the frozen
  `C@50%` (KM median over the 64 **frozen** targets, identical estimator, identical `B_common`)
  must satisfy `C_lo ≤ C_frozen ≤ C_hi`. Lane = **`band_validated`** iff all admissible variants
  pass; else **`method_still_off`** with the offender(s) named and `C_frozen/nearest_edge` logged.
- **GATE-2 — tightness / non-vacuity (checked at LOCK, not after frozen data):** every admissible
  R2′ variant has `W ≤ 3.0`. `W>3.0` ⇒ `precal_insufficient`, **removed from scoring** (may NOT
  pass by widening). Validation variants cannot be removed: `W>3.0` there aborts Stage-2b. `3.0`
  is just above the measured 2.77× wander floor — a v3-style 5.32× systematic miss or any `>3×`
  error still falsifies; a 10× band is impossible by construction.
- **GATE-3 — cross-attacker:** the locked verdict (`lb_wins`/`stern_wins`/`indistinguishable…`)
  must match the frozen verdict by the same pairwise `ΔS(B_star_pair)`+log-rank rule and the same
  family-verdict derivation. A flipped, CI-excluding-0, log-rank-significant pairwise disagreement
  is a cross-attacker **falsification**; a locked near-tie staying a near-tie is a pass.

## Precal protocol (staged; `--frozen` operator-gated)

- **Stage 0 — base cost:** reuse the v4 base-fit (refit `α,β` on the two throwaway base sizes,
  lock `per_iter_ops` per variant). No new fitting freedom.
- **Stage 1 — freeze the slate:** freeze the lock schema, all constants, all seeds, and all gate
  rules in this slate. Operator sign-off. This document is now the frozen Stage-1 contract; any
  change to a frozen knob after this point requires a new slate id. The durable prediction lock is
  not written until Stage 2, after throwaway precal.
- **Stage 2 — precal (throwaway targets only; OPERATOR-GATED, the expensive step):** per regime,
  for each of `K=8` `precal_target_seeds`, draw `T_pre=64` throwaway targets via
  `v2.sample_frozen_manifest` (same sampler as the frozen run → exchangeable by construction;
  `T_pre=64` **matches** `T_frozen=64`, removing v4's 48-vs-64 confound). Compute `B_common`,
  derive `max_B_v`. Run `v2.attacker_run` per `(seed,target,variant)` → `(ops,event)`. Build KM,
  read `M_s`, record censoring + heterogeneity. Form `raw_band → locked_band → W`. Assert
  `code_digest` identical across all 8 seeds. Compute each pairwise `B_star_pair`,
  `Ŝ_v(B_star_pair)`, `ΔS` bootstrap CI, log-rank `p`, and the family verdict. Mark
  `precal_insufficient` per the rule. **Write + freeze the durable prediction lock.**
- **Stage 2b — method validation (zero new frozen-target scoring, but not zero compute):** re-run
  the 8-seed band on v3
  **rung-1** `[128,64]w16` and **rung-3** `[192,96]w18`; confirm each v3 **measured** `C@50%`
  (rung1 LB 1.351×10⁸ / Stern `l7` 1.016×10⁸; rung3 LB 1.076×10¹⁰ / Stern `l9` 8.136×10⁹) lands
  **inside** its v5 locked band. A v3-rung containment failure **aborts the lane** (the estimator
  is wrong) before any frozen R2′ target is touched. Validation variants are not droppable:
  `W>3.0` or median censoring on any validation variant is a Stage-2b abort, not a partial pass.
- **Stage 3 — frozen (operator-gated):** only after the lock is written, `W≤3.0` holds for all
  scored variants, and Stage-2b passes — draw the `T=64` frozen targets at `frozen_target_seed`
  and score under the same `B_common`. Apply GATE-1/2/3. **No retuning after the frozen draw.**

### Implementation / operator command contract

The v5 harness is implementation work after this freeze. It must expose these semantics, even if
the script name is refactored before execution. All long commands are operator-gated because the
precal/validation/frozen stages exceed the lane's inline-run budget.

Primary profile (active Stage-1 choice):

```powershell
$env:PYTHONHASHSEED = "0"

python scripts/pvnp-certificate-syndrome-v5.py --smoke `
  --out results/pvnp/certificate-syndrome-v5/_smoke

python scripts/pvnp-certificate-syndrome-v5.py --precal --profile primary `
  --lock-out docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json `
  --out results/pvnp/certificate-syndrome-v5/precal

python scripts/pvnp-certificate-syndrome-v5.py --validate-v3 `
  --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json `
  --out results/pvnp/certificate-syndrome-v5/v3-validation

python scripts/pvnp-certificate-syndrome-v5.py --frozen-r2prime `
  --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json `
  --out results/pvnp/certificate-syndrome-v5/r2prime-frozen

python scripts/pvnp-certificate-syndrome-v5.py --summarize `
  --root results/pvnp/certificate-syndrome-v5 `
  --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json
```

Fallback profile, if and only if selected before the first v5 precal number exists:
replace `--profile primary` with `--profile fallback`, which locks `K=6`, `g=1.35`, and leaves
`W_max=3.0` unchanged.

**Cost (the honest price; owner chose K=8 full).** `K=8 × T_pre=64` run-to-first-success × 3
variants at the largest-budget `B_common` is ≈ 8–11× v4's precal; **rung-3 `[192,96]w18` is the
heaviest line item** (LB `N≈2400`) — full v5 precal is plausibly multi-overnight. **Budget lever
(locked Stage-1, selectable only before any v5 precal number is computed):** if the operator
declares the budget profile before Stage 2, drop to `K=6` and raise `g` to `1.35` (to hold the
≈10% out-of-band target). The chosen profile is written into the lock. After the first precal
number exists, switching profiles requires a new slate id. Never widen the tightness gate to force
a pass.

## Regimes

| Regime | `(n,k,w,τ)` | role | `code_seed` | precal seeds (`K=8`) | frozen seed |
| --- | --- | --- | ---: | --- | ---: |
| **R2′** | `(160,80,16,16)` | **primary** re-resolution (v4 missed Stern `l8` 2.77×); variants `{lb, stern_l8, stern_l9}` | `20263201` (v3 rung-2 code, reused via `code_digest` gate) | `20265201…20265208` | `20265209` |
| v3-R1 | `(128,64,16,16)` | method validation (precal-only); variants `{lb, stern_l7}` | `20263101` | `20265211…20265218` | — |
| v3-R3 | `(192,96,18,18)` | method validation (precal-only); variants `{lb, stern_l9}` | `20263301` | `20265231…20265238` | — |

Out of scope for v5: the v3-R0 backward-compat anchor. KM equivalence to `_c50_ops` is frozen as
a deterministic smoke/unit check under zero censoring, not as a full K-seed regime. A full R0
backward-compat rerun would be a separate slate if needed.

## Verdict branches

- **`band_validated`** — every admissible variant's frozen median lands inside its locked band
  (GATE-1), all admissible `W≤3.0` (GATE-2), and the cross-attacker verdict matches (GATE-3). The
  median-band calibration is a demonstrated **prospective** predictor on the fresh R2′ regime.
- **`method_still_off`** — a frozen median lands outside its band: named, with the offending
  variant + `C_frozen/nearest_edge`. The band model still misses → a deeper model is open work.
- **`precal_insufficient` (per variant; an honest decline, not a failure)** — a variant's band
  exceeds `W=3.0` or is median-censored across seeds: dropped from scoring and reported as
  unmeasurable-at-`T=64` (the likely outcome for `stern_l9`, 2750× spread). R2′ may be only
  partially resolved (e.g. LB + one Stern scored). This is the design working as intended.
- **cross-attacker** — pairwise `lb_wins` / `stern_wins` / `indistinguishable_at_op_budget`,
  plus family verdict `lb_wins` / `stern_wins` / `indistinguishable_at_op_budget` /
  `mixed_variant`, per GATE-3.
- **`void_run`** — code/regime/seed mismatch, `code_digest` mismatch, manifest not before
  attackers, labels leaked, attacker given more than `z`, non-deterministic, or precal not locked
  before frozen scoring.

## Anti-P-Hack rule

- Stage-1 freeze the regimes, all seeds, the estimator, active profile (`K=8`, `g=1.25` unless
  the operator declares the fallback profile before any v5 precal number exists), `T_pre`,
  `W_max`, `C_MULT`, the `B_common`/pairwise `B_star_pair` rules, the cross-attacker verdict rule,
  and the three gates **before** any number is computed. The budget lever (`K=6`, `g=1.35`) is
  itself locked Stage-1; **`W_max=3.0` is never relaxed to force a pass.**
- Precal on **disjoint throwaway** seeds; the lock is frozen before the single held-out
  `frozen_target_seed` is drawn. Method validation (Stage-2b) must pass before R2′ is read.
- The **band**, not the mean, is the graded object; the mean is logged diagnostic-only.
- A variant whose band would exceed `W=3.0` is **declined (`precal_insufficient`)**, never widened.
- Every attacker sees only `z`; labels scoring-only. Deterministic; report it. `code_digest` for
  R2′ asserted identical across precal + frozen (same code as v3 rung-2).
- `C_best` remains an upper bound vs the tested classes (LB, Stern); BJMM/MMT = a separate slate.

## Open risks (from the design panel — carried, not hidden)

1. **Decline-to-predict on the variant we most want.** `stern_l9` (2750× spread, heavy censoring)
   and possibly `stern_l8` may yield `W>3.0` → `precal_insufficient`. The honest non-answer the
   design prizes, but R2′ may be only partially resolved.
2. **Precal cost** ≈ 8–11× v4 (rung-3 heaviest) — may strain the overnight budget; lever = `K=6`,
   `g=1.35`.
3. **Exchangeability is asserted, not tested** — the band's coverage rests on the
   `frozen_target_seed` being exchangeable with the 8 precal seeds. Identical sampler + matched
   `T=64` make this likely; the `K=8` spread + v3-rung validation are the only checks.
4. **Small-`K` support noise** — min/max of 8 seed-medians is itself noisy; `g=1.25` partly papers
   over it. Realized coverage is not a proven frequentist guarantee; 3 frozen variants = 3
   Bernoulli trials, so a single pass is weak calibration evidence (and a single fail could be bad
   luck).
5. **`B_star_pair` placement** couples cross-attacker power to one pre-frozen pairwise rule; too
   close to the tail → genuinely-different curves read `indistinguishable`.
6. **No extrapolation** — the empirical band describes only the spread it sampled; it cannot
   predict an un-precal'd regime nor explain *why* a variant is unstable.
7. **Mild circularity** — `g=1.25` and `W_max=3.0` are calibrated against the same lane-measured
   1.43–2.77× wander they cover; a novel regime with wander `>3×` is flagged `precal_insufficient`
   (safe decline), not falsely passed — so the failure mode is a decline, not a false pass.

## Freeze checklist

Stage-1 slate contract:

- [x] Freeze the regimes (R2′ + v3-R1/R3 validation; R0 out-of-scope), all `code_seed`/precal/
      frozen seeds, and non-enumerability.
- [x] Freeze the estimator + band constants (`K=8`, `T_pre=64`, `g=1.25`, `W_max=3.0`, `C_MULT=12`,
      the `B_common`/pairwise `B_star_pair` rules, the pre-number budget lever `K=6`/`g=1.35`).
- [x] Freeze the three gates + the cross-attacker verdict rule + the `precal_insufficient` decline.
- [x] Confirm the KM median reduces to v4's `_c50_ops` under zero censoring (backward-compat).
- [x] Stage-1 sign-off; frozen scoring blocked until stage-2 lock + Stage-2b validation pass.

Stage-2 prediction lock, after implementation:

- [ ] Add the v5 statistical layer (`scripts/pvnp-certificate-syndrome-v5.py`: KM-median in op
      units, K-seed band, `B_common`/pairwise `B_star_pair`, `ΔS`+log-rank); validate plumbing +
      byte-determinism on a tiny regime.
- [ ] Run the K-seed precal; produce + lock
      `docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V5_PREDICTION_LOCK.json`; confirm `W≤3.0` for every
      scored variant (or declare `precal_insufficient`).
- [ ] Pass Stage-2b: v3 rungs 1/3 measured `C` inside their v5 bands.
- [ ] Re-affirm: attackers see only `z`; labels scoring-only; per-file outputs; censoring +
      heterogeneity disclosed; `code_digest` consistent.

## Boundary

Unchanged from v1–v4: `C_best` is an upper bound vs the tested classes; no cryptographic
one-wayness claim; verification not claimed "in P"; **no progress on P vs NP**; op-count is the
cost, wall-time diagnostic. v5 hardens the prediction *methodology* (point → falsifiable band with
honest decline) and re-resolves one regime fairly; it does **not** enlarge the scientific claim.

## Freeze rule

Edits allowed without a new slate id: typo/path/output-naming corrections preserving the regimes,
seeds, estimator, band constants, gates, and the cross-attacker rule. Any change to a regime, seed,
the estimator, a band constant, a gate, or the cross-attacker rule requires a new slate id.
