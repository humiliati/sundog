# Sundog Certificate Problem — Syndrome Certificate v4 Median-Calibration + Rung-2 Resolution Slate

Status: **FROZEN — stage-1 slate contract; NOT executed.** Stage-1 sign-off 2026-06-06
(consistency pass clean; precal `cap = 12·N_analytic` stress-tested ≈ 6.3× the v3-measured
R2′ median, well clear of the censoring boundary). No v4 frozen-target run may execute until
the median pre-calibration has produced the stage-2 locked prediction and each attacker has
been validated. (Freeze-before-execute, per the lane.)

Date opened: 2026-06-06

This is the **methodology-fix + loose-end-resolution** successor to
[`SUNDOG_CERTIFICATE_SYNDROME_V3_SLATE.md`](SUNDOG_CERTIFICATE_SYNDROME_V3_SLATE.md)
(scaling ladder → crossover located, measured gap scales with `n` while the frozen Claim-A
gate caveat remains; receipt
[`receipts/2026-06-06_certificate_syndrome_v3.md`](receipts/2026-06-06_certificate_syndrome_v3.md)).
v4 does **not** re-open the v3 science (the crossover and measured `n`-scaling, with the
Claim-A gate caveat, stand); it closes the two flaws v3 named in itself.

## What v4 fixes (the two v3 loose ends)

v3's measurements were sound, but two things it flagged need closing before the certificate
ladder's *predictions* are trustworthy and before its one off-pattern point is read:

1. **Prediction method (general fix).** v3's locked `C@50% = (1/p_emp)·ln2·per_iter` is
   **mean-based**, but the frozen run measures the **median** ops-to-first-success. For
   heavy-tailed Stern these diverge — rung-2 Stern missed the lock by **5.32×** — while the
   median-implied diagnostic predicted all three Stern rungs within factor 2 (0.84 / 1.37 /
   0.74). v4 locks a **directly-measured median** prediction and raises the pre-calibration
   target count so the median is stable for heavy-tailed attackers.
2. **Rung-2 resolution (targeted retest).** v3's rung 2 `[160,80] w16` reverted to LB
   (St/LB = 1.96) as a confounded `model_deviation` — extreme pre-cal heterogeneity (6/16
   zeros) **plus** an edge `l=10` selection (vs analytic `l*=8`) that the noisy pre-cal
   picked. v4 re-measures `[160,80] w16` cleanly to settle whether Stern genuinely loses
   there (a real 2-D crossover feature) or the `l=10` handicap caused it (crossover is
   monotone in `w`).

v4 inherits unchanged from v1–v3: the certificate, `Safe(y):=∃e*: He*=Hy ∧ wt(e*)≤τ`, the
flat witness-verifier, the rank-valid convention, the `invert-s`-vacuous / spoof-structural
facts, the two-size base(`m`) cost model (`base(64)` re-validated to 0.1% in v3), and the
attacker classes (LB `p=2`; Stern `p=2`, explicitly fixed `l` values). Only the
**success/median calibration** and the **rung-2 target population / Stern `l` set** change.

## Median calibration (the v4 prediction mechanism)

Replace v3's per-iteration success-rate sampling with a **direct median measurement**, the
exact analog of the frozen `C@50%` (= median ops-to-first-success over the 64 frozen
targets):

Per regime, on throwaway targets (a dedicated `precal_target_seed`, disjoint from the
frozen `target_seed`), after stage-1 freeze and before frozen scoring:

1. **Run-to-first-success.** Run each attacker / fixed Stern-`l` variant to first success
   on `T_pre` throwaway targets (capped at `cap = ceil(C_MULT · N_analytic)` valid iters;
   `cap`-censored targets are recorded, not silently dropped). Record per-target
   ops-to-first-success.
2. **Median lock.** Treat censored targets as `>cap` and sort all `T_pre` outcomes; the
   locked median is the 24th/25th-order statistic average for `T_pre=48` if both median
   positions are observed. If either median position is censored, the attacker/variant is
   `precal_insufficient` and no frozen scoring is allowed for it. Also log the mean-based
   `C` and the heterogeneity (min/max, censored count) as references — the **median is the
   locked comparator**, the mean is the diagnostic (v3 had it backwards).
3. **No Stern `l` re-optimization in v4.** For methodology validation, Stern uses the
   already-measured v3 windows (`l=7` for rung 1, `l=9` for rung 3). For R2′, Stern uses
   the pre-registered fixed set `l∈{8,9}` and both variants are predicted and measured.
   `C_Stern := min(C_Stern(8), C_Stern(9))` is a frozen comparison rule, not a post-hoc
   parameter choice.
4. **Stage-2 lock** `prediction_lock_v4.json`: per regime, per attacker / Stern-`l`
   variant — locked median `C@50%`, per-iter (`base(m)+enum`, two-size calibrated), the
   mean-based reference, censored counts, and the frozen tolerance (factor 2).

**Frozen pre-calibration plan (stage-1 proposed):**

| Param | Value | Rationale |
| --- | --- | --- |
| `T_pre` | **48** (up from v3's 16) | a stable median for heavy-tailed Stern (v3's 16 gave 6/16 zeros) |
| `cap` (`C_MULT`) | `C_MULT = 12` → `cap = ceil(12·N_analytic)` | reach first success for ≥50% of targets even on the heavy tail; censor the rest |
| insufficiency | either median order position censored for the attacker/variant ⇒ `precal_insufficient` | the median cannot be locked from the observed sample |

## The regimes (stage-1 proposed)

`k=n/2`, `τ=w`, GF(2). The frozen `target_seed` for the rung-2 retest is **fresh** (not
v3's), with the **same code** as v3's rung 2 so it is a clean retest of the *same regime*.
The implementation must emit a `code_digest` for the regenerated `(G,H)` and prove the
R2′ pre-calibration and frozen scoring use the same regenerated code. If the code generator
changes before v4 implementation, v4 must either import a durable v3 `(G,H)` code artifact
or move to a new slate id; same `code_seed` alone is not enough to silently change the code.

| Regime | `(n,k,w,τ)` | role | `code_seed` | `target_seed` | `precal_target_seed` | attackers measured |
| --- | --- | --- | ---: | ---: | ---: | --- |
| R2′ (rung-2 retest) | `(160,80,16,16)` | resolve v3's confound | **`20263201`** (v3 rung-2 code, reused) | `20264202` (fresh) | `20264203` (fresh) | LB + **Stern `l=8` and `l=9`** |

**Methodology validation (no new frozen scoring):** re-run the **median** pre-calibration
on v3's rung-1 and rung-3 regimes (fresh throwaway `precal_target_seed`s) and confirm the
median lock predicts v3's *already-measured* frozen `C@50%` within factor 2 for all four
attacker-points. This validates the v4 method against v3's ground truth at zero
frozen-scoring cost.

| Validation regime | `(n,k,w,τ)` | v3 code seed | v3 measured target source | fresh median-precal seed | attacker variants |
| --- | --- | ---: | --- | ---: | --- |
| V3-R1 validation | `(128,64,16,16)` | `20263101` | v3 rung 1 receipt/artifacts | `20265103` | LB + Stern `l=7` |
| V3-R3 validation | `(192,96,18,18)` | `20263301` | v3 rung 3 receipt/artifacts | `20265303` | LB + Stern `l=9` |

Deferred, explicitly **out of v4**: a W14 midpoint rung (`[128,64]w14`) would sharpen the
exact crossover location between `w12` and `w16`, but that is new science beyond the two v3
loose ends. It belongs in a separate slate if desired.

## Rung-2 resolution design

On R2′ (same `[160,80]w16` code as v3, fresh targets): measure **LB**, **Stern `l=8`**,
**Stern `l=9`** on the same 64 frozen targets; `C_Stern := min(C_Stern(8), C_Stern(9))`;
compare to `C_LB`. The median lock is the pre-registered comparator. Outcome:

- **`Stern_wins` (crossover monotone):** `C_Stern < C_LB` — the v3 rung-2 LB-win was the
  `l=10` handicap; Stern beats LB at `[160,80]w16` after all, and the crossover is monotone
  in `w` (Stern wins at every `w≥16` rung tested).
- **`LB_wins` (genuine 2-D feature):** `C_Stern ≥ C_LB` even at the best of `l∈{8,9}` —
  `[160,80]w16` genuinely favors LB, so the LB↔Stern winner is genuinely non-monotone in
  `(n,w)` (a real result about ISD selection on these codes, not an artifact).

Either way the v3 measured ladder is unchanged; v4 only *interprets* its one off-pattern
point with a clean measurement.

## Two-stage freeze (as v3)

1. **Stage 1 — slate contract.** Freeze the regimes, seeds, target separation, the median
   pre-calibration plan (`T_pre=48`, `C_MULT=12`, insufficiency rule), the fixed Stern
   `l` variants (`R2′: {8,9}`, validation: v3-selected `l`s), the rung-2 outcome
   definitions, output schema, and gates.
2. **Stage 2 — prediction lock.** Freeze `prediction_lock_v4.json` after the median
   pre-calibration + harness validation. After this lock the operator may run the frozen
   commands. Any post-lock edit to the locked median `C`, fixed `l` variants, seeds,
   regimes, or tolerances voids v4.

## Implementation and staged commands

The implementation must add `scripts/pvnp-certificate-syndrome-v4.py`, reusing the v3/v2
GF(2) core and emitting the v3-style per-file bundle. Under the repo's ~10-minute rule,
only `--stage1-smoke` may run inline if it measures below the limit; median pre-calibration
and R2′ frozen scoring are staged operator commands.

PowerShell command contract:

```powershell
$env:PYTHONHASHSEED = "0"
python scripts/pvnp-certificate-syndrome-v4.py --stage1-smoke --out results/pvnp/certificate-syndrome-v4/_smoke
python scripts/pvnp-certificate-syndrome-v4.py --median-precal --out results/pvnp/certificate-syndrome-v4/precal --lock-out docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json
python scripts/pvnp-certificate-syndrome-v4.py --validate-v3 --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json --out results/pvnp/certificate-syndrome-v4/method-validation
python scripts/pvnp-certificate-syndrome-v4.py --frozen --regime r2prime --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json --out results/pvnp/certificate-syndrome-v4/r2prime
python scripts/pvnp-certificate-syndrome-v4.py --summarize --root results/pvnp/certificate-syndrome-v4 --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json
```

Wall-time estimates are diagnostic-only and must be measured by `--stage1-smoke` before
any long command. Pre-freeze estimate: R2′ frozen scoring is approximately v3 rung-2 cost
plus a second Stern variant (hours, not minutes); median pre-calibration has `T_pre=48`
and is staged unless the smoke proves otherwise. Completed output roots are read-only; use
a new root for reruns unless a manifest says `complete=false`.

## Gates

| Gate | Required |
| --- | --- |
| Regime non-enumerable | `C(n,w) > 10¹²` (R2′: `C(160,16)=4.06×10²¹`; validation regimes inherit v3 non-enumerability) |
| Code validity | `G Hᵀ = 0` per regime |
| Code identity for R2′ | R2′ pre-calibration and frozen scoring emit the same `code_digest`; if the v3 code generator is changed, v4 requires a new slate or a durable imported v3 `(G,H)` artifact |
| Manifest before attackers | emit + hash `target_manifest.json` (attacker-visible `z` \| labels-only) before any attacker; attackers read only `z` |
| Median lock locked | `prediction_lock_v4.json` (median `C`, fixed `l` variants, constants) frozen before any frozen target scored; disjoint precal targets; `T_pre=48` |
| Method validation | the median lock predicts v3's measured rung-1/rung-3 `C` within factor 2 (zero-frozen-scoring ground-truth check) |
| Verifier cheap | flat `2mn+n+m` reported; `≪ C_best` |
| Witness validity | every success exhibits `He*=z ∧ wt≤τ`, re-checked from public data |
| Privilege / label audit | attackers see only `z`; `s,e,wt(e)` scoring-only |
| Censoring disclosed | per attacker/`l`: cap, censored count (a data-quality flag, as v3's rung-2 17/64) |
| Determinism | seed-pinned (`PYTHONHASHSEED=0`); byte-identical harness-test |

## Verdict branches

- **method_validated** — the median lock predicts every re-measured/validated point within
  factor 2 (incl. v3's rungs 1, 3 and the R2′ retest). The v3 mean-based mis-prediction is
  closed; the median form is the lane's calibration going forward.
- **rung2_resolved → `Stern_wins`** — crossover is monotone in `w`; the v3 rung-2 LB-win was
  the `l=10` handicap. The v3 receipt remains filed as-measured; future summaries may read
  the v3 non-monotone point as resolved by v4's fixed-`l` retest.
- **rung2_resolved → `LB_wins`** — `[160,80]w16` genuinely favors LB; the crossover is a
  real 2-D feature. The v3 "non-monotone" reading stands and is now clean.
- **method_still_off (named)** — the median lock itself mis-predicts a point beyond factor 2:
  report it; a deeper distributional model (not mean or median) is needed — a further slate.
- **6.1/6.4 / void_run** — verifier not below cheapest attacker; or spec/manifest/label/
  determinism integrity failure.

## Anti-P-Hack rule

- Stage-1 freeze the regimes, seeds, the median pre-cal plan, the fixed Stern `l` variants,
  and the rung-2 outcome definitions **before** any run. Do not retune to force a crossover.
- Median pre-calibration on **disjoint throwaway** targets; locked before frozen scoring.
- Rung-2 retest uses the **same code** as v3 rung-2 with **fresh** targets (a clean retest,
  not target-cherry-picking); both `l∈{8,9}` measured (no post-hoc `l` choice).
- Every attacker sees only `z`; labels scoring-only. Deterministic; report it.
- `C_best` is an upper bound vs the tested classes (LB, Stern); BJMM/MMT = a separate slate.

## Freeze checklist

Stage-1 slate contract:

- [x] Freeze R2′ + v3-rung validation regimes, seeds, and non-enumerability. Done 2026-06-06.
- [x] Freeze the median pre-cal plan (`T_pre=48`, `C_MULT=12`, insufficiency, median-locked
      comparator, fixed Stern `l` variants). Done 2026-06-06.
- [x] Freeze the rung-2 outcome definitions (`Stern_wins` vs `LB_wins`) and the
      method-validation gate against v3's measured rungs. Done 2026-06-06.
- [x] Confirm W14 is out-of-scope for v4 and deferred to a future slate if wanted. Done.
- [x] Stage-1 sign-off recorded 2026-06-06; frozen scoring blocked until stage-2 lock exists.

Stage-2 prediction lock, after implementation:

- [ ] Add the median pre-cal mode to the harness (run-to-first-success, `T_pre`, median
      lock); validate plumbing + byte-determinism on a tiny regime.
- [ ] Run median pre-cal; produce + lock `prediction_lock_v4.json`; pass the v3
      ground-truth validation gate.
- [ ] Re-affirm: attackers see only `z`; labels scoring-only; per-file outputs; censoring
      disclosed.

## Required outputs (under `results/pvnp/certificate-syndrome-v4/`)

Per measured regime: the v3 per-file bundle + `precal_report.json` (median `C`, mean
reference, heterogeneity, censored counts, fixed Stern `l` variants). Plus a top-level
`method_validation.json` (median lock vs v3's measured rung-1/3 `C`), `code_identity.json`
(R2′ code digest consistency), a `rung2_resolution.json` (`C_LB`, `C_Stern(8)`,
`C_Stern(9)`, `C_best`, outcome), and `V4_SUMMARY.md`. Plus the durable
`prediction_lock_v4.json`.

## Boundary

Unchanged from v3: `C_best` is an upper bound vs the tested classes; Prange not measured
here (the anchor/formula stand from v3); no cryptographic one-wayness claim; verification is
not claimed "in P"; **no progress on P vs NP**; op-count is the cost, wall-time diagnostic.
v4 hardens the *prediction methodology* and resolves one measured point — it does not
enlarge the scientific claim.

## Freeze rule

Edits allowed without a new slate id: typo/path/output-naming corrections preserving the
regimes, seeds, the median pre-cal plan, and the rung-2 outcome definitions. Any change to a
regime, seed, the calibration method, or an attacker parameter requires a new slate id.
