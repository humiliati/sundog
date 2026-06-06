# Sundog Certificate Problem — Syndrome Certificate v3 Scaling-Ladder Slate

Status: **FROZEN — stage-1 slate contract; NOT executed.** No v3 frozen-target attacker
run may execute until the empirical pre-calibration has produced the stage-2 locked
prediction and each attacker has been validated. (Freeze-before-execute, per the lane.)

Date opened: 2026-06-05

This is the **scaling-ladder** successor to
[`SUNDOG_CERTIFICATE_SYNDROME_V2_SLATE.md`](SUNDOG_CERTIFICATE_SYNDROME_V2_SLATE.md)
(Prange→LB→Stern ladder at `[128,64] w=12` → bounded-positive **with a named LB↔Stern
model-deviation**: measured **Lee-Brickell beat Stern** — `C_best=8.31×10⁷` ops — because
Stern's ISD success heuristic was optimistic at `w=12`, receipt
[`receipts/2026-06-05_certificate_syndrome_v2.md`](receipts/2026-06-05_certificate_syndrome_v2.md)).
v3 asks the two scaling questions v2 raised.

## What v3 measures (two pre-registered claims)

Over a **ladder of increasing regimes**, measuring Lee-Brickell and Stern at each:

- **Claim A — the find-vs-check gap scales.** Does `C_best / verifier` grow from the
  anchor to the top rung, without an order-of-magnitude collapse on the interior rungs
  (bigger non-enumerable code → larger one-wayness threshold against the cheap flat
  verifier)? This is the robust, paper-relevant scaling result. The planning scan already
  predicts a rung-1/rung-2 plateau (`~50,900×` → `~48,000×`), so v3 does **not** require
  strict per-rung monotonicity.
- **Claim B — the LB↔Stern crossover.** v2 measured LB beating Stern at `w=12`. As `w`
  (and `n`) grow, Stern's per-iteration advantage is predicted to grow faster than its
  collision overhead. Does Stern **overtake** LB by the top rung — or does the measured
  optimism/overhead curve explain why LB keeps winning? Either is a result.

v3 makes **no** new claim about Prange beyond the v2 anchor; it does not change the
certificate, the existence predicate `Safe(y):=∃e*: He*=Hy ∧ wt(e*)≤τ`, the
witness-verifier, or the `invert-s`-vacuous / spoof-structural facts. Only the **regime**
scales and the **attacker success model is measured empirically** rather than assumed.

## The regime ladder (selection rationale)

Selected with `scripts/pvnp-certificate-syndrome-v3-regime-scan.py` (planning helper;
the predicted-vs-measurable scan). Two structural facts drive the design:

1. **Prange is unmeasurable at scale.** Its `N` explodes with `w` (`7.2×10³` at the
   anchor → `10⁵–10⁸` up the ladder). So v3 measures **only LB and Stern** at each rung;
   **Prange is measured once at the anchor (reuse the v2 frozen result) and formula-
   predicted at the scaled rungs** (clearly labelled as a formula bound, never a measured
   `C`). The LB↔Stern crossover — the v3 question — needs only LB and Stern, both small-`N`.
2. **The analytic St/LB ratio is unreliable.** It says Stern beats LB *everywhere*
   (St/LB `<1` at every rung incl. the anchor `0.90`), yet v2 **measured** the opposite
   at the anchor (St/LB `=1.71`) — the analytic is ~`1.9×` optimistic for Stern. So v3
   does **not** trust the analytic crossover; it **measures** it, with an **empirically
   pre-calibrated** Stern success model (below).

| Rung | Regime | `w` | `l*` (Stern) | measured here | est. LB+Stern wall | role |
| --- | --- | ---: | ---: | --- | ---: | --- |
| 0 (anchor) | `[128,64]` | 12 | 8 | **reuse v2** (Prange+LB+Stern) | — | anchor: measured St/LB=1.71 |
| 1 | `[128,64]` | 16 | 7 | LB + Stern | ~0.9 h | fixed-`n`, `w`-scaling |
| 2 | `[160,80]` | 16 | 8 | LB + Stern | ~1.5 h | `n`-scaling at fixed `w` |
| 3 | `[192,96]` | 18 | 9 | LB + Stern | ~8.4 h | predicted-crossover rung |

`k=n/2` (rate-½) at every rung; `τ=w`; GF(2). Each new rung gets a **fresh
`code_seed` and a decoupled `target_seed`** (stage-1 frozen values below), a
`target_manifest.json` emitted before any attacker, and attackers consume only `z`.
`l*` is the cost-model-optimal Stern window, **re-confirmed/adjusted by the empirical
pre-calibration** under the stage-1 rule before frozen scoring. Total new compute ≈ 11 h
(LB+Stern), stageable rung-by-rung. The anchor is **not re-run** — v2's frozen receipt is
the rung-0 datum.

Stage-1 frozen values:

| Rung | `(n,k,w,τ)` | `C(n,w)` | `code_seed` | `target_seed` | `precal_target_seed` | source |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 0 | `(128,64,12,12)` | `2.373e16` | v2 | v2 | v2 smoke only | v2 receipt reused verbatim |
| 1 | `(128,64,16,16)` | `9.334e19` | `20263101` | `20263102` | `20263103` | new frozen rung |
| 2 | `(160,80,16,16)` | `4.060e21` | `20263201` | `20263202` | `20263203` | new frozen rung |
| 3 | `(192,96,18,18)` | `8.629e24` | `20263301` | `20263302` | `20263303` | new frozen rung |

Predicted ladder (analytic, for selection only — superseded by the empirical lock):

| Rung | analytic St/LB | ×1.9 (v2-optimism est.) | predicted gap@`C_best` |
| --- | ---: | ---: | ---: |
| 0 `[128,64]w12` | 0.90 | **1.71 (measured)** | 5,226× |
| 1 `[128,64]w16` | 0.71 | ~1.35 (LB wins) | ~50,900× |
| 2 `[160,80]w16` | 0.60 | ~1.14 (LB wins, close) | ~48,000× |
| 3 `[192,96]w18` | 0.47 | **~0.89 (Stern overtakes?)** | ~153,000× |

So the ladder **brackets** the predicted crossover: if Stern overtakes by rung 3, Claim B
is *located*; if it does not, v3 reports the measured optimism/overhead curve as a named
finding rather than forcing a crossover story. The gap column gives Claim A its predicted
top-rung scaling (~30× growth anchor→rung 3) while already allowing a rung-1/rung-2
plateau.

## Capacity unit: operations (unchanged from v2)

`C_attacker := the measured op-count budget at which that attacker reaches 50% witness
recovery` over the rung's `T=64` targets = the **median ops-to-first-success**. Per-iter
op-count decomposes as `base(m)+enum` (v2-validated; per-iter matched the v2 lock to ~2%).
`base(m)` is re-calibrated by a two-size smoke spanning the v3 `m`-range (`m=64,80,96`);
`enum` is analytic. Wall-time stays diagnostic-only.

## Two-stage freeze

v3 has two frozen surfaces, because the prediction mechanism itself is empirical:

1. **Stage 1 — slate contract.** Freeze the regimes, seeds, target separation,
   pre-calibration plan, `l` candidate sets, attacker classes, output schema, and verdict
   gates. After this freeze, implementation and throwaway pre-calibration may run, but no
   frozen target may be scored.
2. **Stage 2 — prediction lock.** Freeze `prediction_lock_v3.json` after the throwaway
   pre-calibration and harness validation. The lock fixes empirical `p`, selected Stern
   `l`, cost constants, C predictions, tolerances, and hashes. After this lock, the
   operator may run the frozen rung commands. Any post-lock edit to empirical `p`, selected
   `l`, seeds, regimes, or tolerances voids v3 and requires a new slate id.

This slate can be stage-1 frozen before `prediction_lock_v3.json` exists; it cannot be
executed against frozen targets until stage 2 is locked.

## Empirical pre-calibration (the v3 prediction mechanism)

v2 proved the analytic Stern **success** heuristic is optimistic (its *cost* model was
accurate to ~1%; the entire deviation was in iteration count). v3 therefore **measures**
each attacker's true per-iteration success and locks that, instead of the analytic `p`.

Per new rung, on **throwaway targets** (a dedicated `precal_target_seed`, disjoint from
the frozen `target_seed`), after the stage-1 slate contract is frozen and before frozen
target scoring:

Fixed pre-calibration plan:

| Rung | `T_pre` | LB budget | Stern `l` candidates | Stern budget per `l` | insufficiency branch |
| --- | ---: | --- | --- | --- | --- |
| 1 | 16 | `ceil(32 / p_LB_analytic)` valid iters, split round-robin | `5,6,7,8,9` | `ceil(32 / p_Stern_analytic(l))` valid iters, split round-robin | `<16` successes for the selected attacker/window ⇒ `precal_insufficient`, no frozen scoring |
| 2 | 16 | same rule | `6,7,8,9,10` | same rule | same |
| 3 | 16 | same rule | `7,8,9,10,11` | same rule | same |

The expected success budget (32) is deliberately modest: it is enough to catch factor-size
success-model errors like v2's Stern optimism, but not enough to justify tight
sub-factor claims. v3's frozen measured run remains the adjudicator.

1. **Direct success-rate measurement.** Run each attacker for a large fixed budget of
   valid iterations across the 16 throwaway targets and count successes →
   `p_empirical = successes / valid_iters` (sampling per-iteration success directly, not
   running to first-success). Record the **per-target heterogeneity** (v2 showed Stern's
   per-target hardness is heavy-tailed — 5/64 censored — so a single `p` understates the
   spread; report the spread and the median-implied `N`).
2. **`l` re-optimization (Stern).** Sweep `l` on the throwaway targets and pick the
   **empirically** cost-optimal window per rung from the fixed candidate set above. The
   selection rule is deterministic: compute `C_ops_50pct = ln2·per_iter/p_empirical` for
   each candidate; if the analytic `l*` is within 10% of the best empirical candidate,
   keep the analytic `l*` (stability over chasing noise); otherwise choose the lowest
   empirical `C`. Exact ties choose the smaller `l`. Lock the chosen `l`.
3. **Cost calibration.** Two-size smoke (`m` bracketing the rung) fixes `base(m)`; `enum`
   analytic. Per-iter `= base(m)+enum`, rank-fail `ρ` folded into `base` as in v2.
4. **Lock** `prediction_lock_v3.json`: per rung, per attacker —
   `p_empirical` (with its CI / heterogeneity), `per_iter`, the predicted
   `C_ops_50pct = (1/p_empirical)·ln2·per_iter`, the analytic `p` and `C` **alongside**
   (as the v2-falsified reference, NOT the comparator), and the frozen tolerance. The
   analytic→empirical ratio (the measured **optimism factor**) is logged per rung — its
   evolution across the ladder is itself a v3 result.

The pre-calibration smoke is **disjoint** from the frozen targets and **locked before**
any frozen target is scored; updating it after seeing frozen results voids the rung. The
stage-2 lock file records the exact candidate set, observed successes, Wilson 95% CI,
heterogeneity, selected `l`, and the deterministic tie-break result.

## The attackers (frozen classes)

- **Lee-Brickell, `p=2`** — measured at every rung.
- **Stern, `p=2`, `l` = the empirically cost-optimal window per rung** (frozen after the
  pre-calibration) — measured at every rung.
- **Prange** — measured **only at the anchor** (the v2 frozen receipt); at rungs 1–3 it is
  a **formula-predicted bound** `C_Prange = N_analytic·ln2·base(m)` (Prange has no
  enum and is its own base probe), reported as a *prediction*, never a measured `C`. The
  "drop vs Prange" at scale is therefore measured-LB/Stern vs predicted-Prange — labelled.

All attackers consume only `z`; each must exhibit a valid witness (`He*=z ∧ wt≤τ`,
re-checked from public `H,z,τ`) before its curve is trusted (as v2). The valid-iteration
(rank-valid) convention and the rank-fail audit are unchanged from v2.

## Gates

| Gate | Required |
| --- | --- |
| Regime per rung non-enumerable | `C(n,w) > 10¹²` at every rung (frozen pre-run) |
| Code validity per rung | `G Hᵀ = 0` |
| Manifest before attackers | each rung emits `target_manifest.json` (attacker-visible `z` \| labels-only `s,e,wt`) and hashes it **before** any attacker; attackers read only `z` |
| Verifier cheap + scaling | flat verifier op-count `2mn+n+m` reported per rung; `≪ C_best` at every rung |
| Witness validity | every attacker success exhibits a valid `e*` (`He*=z ∧ wt≤τ`), re-checked from public data |
| Privilege / label audit | every attacker sees **only** `z`; `s,e,wt(e)` scoring-only |
| Empirical calibration locked | `prediction_lock_v3.json` (empirical `p`, `l`, cost constants) frozen **before** any frozen target is scored; disjoint precal targets |
| Anchor consistency | rung-0 numbers are the v2 receipt verbatim (no re-run); the v3 cost model reproduces v2's measured per-iter to within tolerance |
| Prange-at-scale labelling | rungs 1–3 Prange is a **formula prediction**, never reported as a measured `C` |
| Determinism | seed-pinned; determinism established by construction + a byte-identical harness-test (per v2) |
| Cost reported | per-rung per-attacker ops, the `C` ladder, find-vs-check gaps, rank audit |

## Implementation and staged commands

The implementation must add `scripts/pvnp-certificate-syndrome-v3.py` with these exact
operator-facing modes. The smoke may be run by an agent only if its measured estimate
stays under the repo's ~10-minute rule; the pre-calibration and frozen rungs are staged
operator commands.

PowerShell command contract:

```powershell
$env:PYTHONHASHSEED = "0"
python scripts/pvnp-certificate-syndrome-v3.py --stage1-smoke --out results/pvnp/certificate-syndrome-v3/_smoke
python scripts/pvnp-certificate-syndrome-v3.py --precal --out results/pvnp/certificate-syndrome-v3/precal --lock-out docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json
python scripts/pvnp-certificate-syndrome-v3.py --frozen --rung 1 --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json --out results/pvnp/certificate-syndrome-v3/rung1
python scripts/pvnp-certificate-syndrome-v3.py --frozen --rung 2 --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json --out results/pvnp/certificate-syndrome-v3/rung2
python scripts/pvnp-certificate-syndrome-v3.py --frozen --rung 3 --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json --out results/pvnp/certificate-syndrome-v3/rung3
python scripts/pvnp-certificate-syndrome-v3.py --summarize --root results/pvnp/certificate-syndrome-v3 --prediction-lock docs/pvnp/SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json
```

Wall-clock estimates from the scan helper, before pre-calibration: rung 1 ≈ 0.9 h, rung
2 ≈ 1.5 h, rung 3 ≈ 8.4 h, total frozen scoring ≈ 11 h. The pre-calibration wall-clock is
reported by `precal_report.json` and must be staged unless the stage-1 smoke proves it is
under the ~10-minute inline limit. The commands are resume-safe only if the manifest says
`complete=false`; a `complete=true` rung is read-only and must not be overwritten without a
new output root.

## Verdict branches

- **Claim A — gap-scales-confirmed.** The top rung's `C_best/verifier` is at least `10×`
  the anchor gap, and every interior rung remains at least `5×` the anchor gap. This
  confirms scaling without requiring strict rung-by-rung monotonicity, which the planning
  scan does not predict. The certificate's one-wayness threshold scales with the regime
  against the flat verifier.
- **Claim B — crossover-located.** Stern's measured `C` drops below LB's at some rung
  (`St/LB < 1`), locating the crossover; report the rung and the measured `St/LB` ladder
  (`1.71 → … → <1`).
- **Claim B — crossover-not-reached (NAMED finding, not a failure).** Stern still loses
  at rung 3 (`St/LB ≥ 1` throughout): report the measured optimism/overhead-vs-regime
  curve and whether it grows, shrinks, or stays flat. `C_best` stays LB-set; this is a
  result about ISD work-factor heuristics, not a quarantine.
- **model-deviation (named).** A measured `C` deviates from the **empirical** locked
  prediction beyond tolerance (factor 2): report it; the empirical model mis-predicted at
  that regime (a second-order finding, since the empirical model is itself measured).
- **6.1 vacuity / 6.4 overhead.** The flat verifier op-count is not below the cheapest
  attacker at some rung: falsified at that rung.
- **void_run** — code/regime not matching the frozen rung spec, a manifest not
  materialized before attackers, labels leaked, an attacker given more than `z`, a
  non-validated attacker, the empirical calibration not locked before frozen scoring, or
  non-deterministic.

## Anti-P-Hack rule

- Stage-1 freeze every rung's `(n,k,w,τ,code_seed,target_seed,precal_target_seed)`, the
  attacker classes (`LB p=2`; `Stern p=2`), the Stern `l` candidate sets, the empirical
  pre-calibration plan, and the staged command contract. Do not retune to hit a crossover.
- Stage-2 freeze `prediction_lock_v3.json` (including the per-rung empirically locked
  Stern `l`) **before** the frozen runs.
- The empirical pre-calibration uses **disjoint throwaway** targets and is locked before
  frozen scoring; the frozen run confirms the regime's `C@50%` on held-out targets.
- Prange at scale is a **formula prediction**, explicitly labelled, never a measured `C`.
- Every attacker receives only `z`; planted `e/wt(e)/s` scoring-only.
- Determinism: seed-pinned; report it.
- `C_best` at every rung is an **upper bound** against the tested classes (LB, Stern);
  BJMM/MMT would lower it (a separate slate). The crossover is a statement about *these*
  two attackers, not about ISD in general.

## Freeze checklist

Stage-1 slate contract:

- [x] Freeze the 4-rung ladder regimes + per-rung `code_seed/target_seed/precal_target_seed`.
- [x] Confirm each selected regime is non-enumerable (`C(n,w) > 10¹²`).
- [x] Freeze the empirical pre-calibration target count, budgets, Stern `l` candidate
      sets, and deterministic `l` tie-break.
- [x] Freeze the Claim A scaling gate without strict rung-by-rung monotonicity.
- [x] Freeze the staged PowerShell command contract and wall-clock estimates.
- [x] Stage-1 freeze sign-off recorded; frozen-target scoring remains blocked until the
      stage-2 prediction lock exists.

Stage-2 prediction lock, after implementation:

- [ ] Implement the empirical pre-calibration (direct per-iteration success sampling +
      `l` re-optimization + two-size cost smoke); produce + lock `prediction_lock_v3.json`
      with empirical `p`, locked `l`, cost constants, predicted `C`, analytic reference,
      and tolerance.
- [ ] Reuse the v2 anchor verbatim (no re-run); confirm the v3 cost model reproduces v2's
      measured per-iter within tolerance.
- [ ] Validate LB + Stern at each rung's scale (recover valid witnesses; empirical curve
      matches the empirical model) on throwaway targets before the frozen runs.
- [ ] Re-affirm: every attacker sees only `z`; labels scoring-only; per-file outputs;
      Prange-at-scale labelled as formula.

## Required outputs (per rung, under `results/pvnp/certificate-syndrome-v3/<rung>/`)

Per rung, the v2 bundle (`target_manifest.json`, `verifier_access_declaration.json`,
`manifest.json`, `capacity_ladder.json`, `prediction_vs_measured.json`,
`valid_iteration_audit.json`, `witness_validity_audit.json`, `op_count_report.json`,
`attacker_ladder_curve.csv`, `iteration_crosscheck.json`, `falsifier_summary.md`,
`README.md`), plus `precal_report.json` (empirical `p`, heterogeneity, locked `l`). And a
**cross-rung** `scaling_summary.json` + `SCALING_LADDER.md`: the gap-vs-regime curve
(Claim A), the measured `St/LB`-vs-regime ladder and located crossover or optimism-growth
curve (Claim B), and the optimism-factor (analytic/empirical) vs regime. Plus the durable
`prediction_lock_v3.json`.

## Freeze rule

Edits allowed without a new slate id: typo/path/output-naming corrections preserving the
regimes, attacker parameters, seeds, and the locked empirical prediction. Any change to a
rung regime, an attacker parameter, a seed, or the empirical lock requires a new slate id.
