# Phase 4 Balance Substrate-Leg Receipt

> Phase 4 substrate-empirical leg companion to
> [`COARSE_GRAINING_PROOF_ROADMAP.md`](../COARSE_GRAINING_PROOF_ROADMAP.md) ▸
> Phase 4 and to
> [`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md)
> (the three-body buildout of the same gate). The three-body substrate
> closed Privileged-only on 2026-05-31 ([§IAD Receipt](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md#bf-4b-iad-receipt-2026-05-31-_bf4b-accessibility)).
> This receipt records the **positive** close on Balance.
>
> **Status as of 2026-05-31: CLOSED with Φ-ACCESSIBLE verdict.** The
> Phase-4 substrate-empirical leg succeeds on Balance/cart-pole at the
> `near_fall × light_elevation=8` cell with the `bayes_floor_shadow_particle`
> floor against the `sundog_shadow` signature.

## 1. Why a separate substrate

[`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md) ▸
§IAD Receipt records that the planar restricted three-body near-escape
pocket does NOT admit Φ-sufficient control under
`track_sensor_accel_guarded` signature: the BF-4b
Information-Accessibility Diagnostic closed Privileged-only with mean
regret 0.0213, 95% CI `[-0.00055, 0.052]`, 4/8 negative-regret rate.
That is a substrate-specific negative — it does NOT generalize to
"Phase 4 is dead" without trying a different substrate.

Per the IAD's Privileged-only branch, the pre-registered remediation
is **scope reassessment, not more floor engineering**. Of the
substrates with existing Bayesian-floor infrastructure (Balance, Mines;
Mesa is the Phase-5 substrate without a BF planner), Balance has the
strongest prior Φ-accessibility signal — its Phase 15 full lock
([results/balance/phase15-phase10-full-lock/](../../results/balance/phase15-phase10-full-lock/),
27,200 trials, audits passing, 56/56 hard-gate cells, zero
negative-mean-regret cells) showed `bayes_floor_shadow_particle` beats
`sundog_shadow` consistently. The strongest per-cell signal was
`near_fall × light_elevation=8` (borderline class, claimGatePass=true,
meanRegretVsSundog = 0.267 over 100 seeds — 12× the three-body IAD
signal).

This receipt re-runs the IAD's exact 3-branch decision structure
against Balance at that cell.

## 2. Cell + Signature definitions (pinned)

- **Substrate:** Balance / cart-pole, `near_fall` preset (force=12, rail=2.4,
  push=4.5).
- **Cell:** `light_elevation__8__light_8__delay_0__noise_0__drop_0__force_12__rail_2p4__push_4p5__preset_near_fall`
  — cellClass=borderline, staticBoundaryMechanisms=long_shadow_scaling. Loaded
  from a single-row envelope.csv slate under
  [`results/balance/phase4-iad/_iad-cell/`](../../results/balance/phase4-iad/_iad-cell/)
  via `--cell-slate phase10-output`.
- **Signature `Φ`:** `sundog_shadow` controller's information budget — shadow
  residual, residual velocity, confidence gating, cart proprioception,
  bounded reacquire probe. No belief, no privileged state.
- **Bayesian floor:** `bayes_floor_shadow_particle` — same observation budget
  as `sundog_shadow` plus a particle belief over `(theta, thetaDot)`. No true
  `theta`, no privileged state.
- **Oracle reference:** `oracle` controller (privileged true theta/thetaDot —
  diagnostic ceiling, not admissible).
- **Naive reference:** `naive_shadow` (current residual + cart proprioception,
  no gating, no belief — establishes the floor's lower bound).

The substrate's mode definitions are persisted in each shard's
`manifest.json ▸ modeDefinitions` and lifted verbatim from
[`balance-phase15-bayes-floor.mjs`](../../scripts/balance-phase15-bayes-floor.mjs).

## 3. IAD-equivalent compute settings (vs Balance smoke baseline)

| Knob | Smoke (`balance:phase15:smoke`) | Balance IAD |
| --- | ---: | ---: |
| Particles | 61 | **512** |
| Planning horizon | 0.05 s | **0.5 s** |
| Trial duration | 8 s | 8 s |
| Modes | 4 | 4 |
| Seeds per cell | 8 | 8 |

The horizon increase (10×) is the analog of three-body's 50× horizon boost
during its IAD. Balance lacks a `candidate-hold-steps` analog (planner
re-evaluates each control tick), so per-trial cost scales roughly as
`particles × horizon × duration / dt`. Spec command:

```powershell
node scripts/balance-phase4-iad-shard.mjs --seed <S> --particles 512 --horizon-seconds 0.5
# (or via the orchestrator that fans out across seeds and writes the cell-slate)
node scripts/balance-phase4-iad-concurrent.mjs   # defaults: 8 seeds, 3-wide, 512p, 0.5s horizon
```

## 4. Pre-registered decision (3 branches)

Identical structure to
[`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md)
▸ §IA Diagnostic pre-registered decision, adapted to Balance's
`bayesNormalizedSurvival − sundogNormalizedSurvival` regret and the
Phase-15 reference of 0.267:

- **Φ-accessible** — 95% paired-bootstrap CI lower bound `> 0` AND
  mean ≥ `0.5 × Phase-15 reference (= 0.134)`: off-set arm is
  satisfiable from `Φ` history → **Phase-4 substrate-empirical leg
  closes positive on Balance.**
- **Privileged-only** — CI includes 0: same outcome class as the
  three-body IAD — substrate-specific negative on this cell-`Φ` triple.
- **Partial** — CI lower `> 0` but mean `< 0.134`: quantify the
  accessible fraction; report-only.

## 5. Receipt (2026-05-31, `_iad-merged`)

**Verdict: Φ-ACCESSIBLE.**

Decisive numbers (regret = `bayesNormalizedSurvival − sundogNormalizedSurvival`,
95% paired-bootstrap CI over 8 seeds, 2000 iterations, seed 40604):

| Metric | Value |
| --- | ---: |
| Mean regret | **0.4089** |
| 95% CI | **[0.1040, 0.7132]** |
| Phase-15 reference (100-seed lock) | 0.267 |
| % reference recovered | **153.1%** |
| Negative-regret rate | **0/8** |
| Branch fired | **Φ-ACCESSIBLE** |

The 8-seed Balance IAD recovers 153% of the Phase-15 100-seed reference mean
because the smaller sample happens to skew toward harder trajectories where
the bayes-floor win is biggest; the wide CI (0.10-0.71) reflects this
variance honestly. The CI lower bound at 0.104 is two orders of magnitude
above the three-body IAD's CI lower bound of -0.00055, and the
negative-regret rate is 0/8 vs three-body's 4/8.

**Per-seed regret:**

| Seed | `sundog_shadow` normalized survival | `bayes_floor_shadow_particle` normalized survival | Regret | Direction |
| ---: | ---: | ---: | ---: | --- |
| 0 | 1.000 | 1.000 | +0.000 | tie (easy trajectory) |
| 1 | 1.000 | 1.000 | +0.000 | tie |
| 2 | ~0.18 | ~1.000 | **+0.821** | bayes wins (sundog falls; bayes recovers) |
| 3 | ~0.21 | ~1.000 | **+0.793** | bayes wins |
| 4 | ~0.18 | ~1.000 | **+0.825** | bayes wins |
| 5 | 1.000 | 1.000 | +0.000 | tie |
| 6 | ~0.17 | ~1.000 | **+0.832** | bayes wins |
| 7 | 1.000 | 1.000 | +0.000 | tie |

The pattern is bimodal and clean: 4/8 easy seeds tie at full survival;
4/8 hard seeds show **sundog_shadow falls and `bayes_floor_shadow_particle`
recovers near-full survival with the same `Φ`**. Zero ambiguous middle
ground. (`bayes` survival columns inferred from regret value + sundog
survival; exact numbers in
[`phase4-balance-regret-per-seed.csv`](../../results/balance/phase4-iad/_iad-merged/phase4-balance-regret-per-seed.csv).)

## 6. Doctrinal consequences

Per the pre-registered Φ-accessible branch:

- **Phase-4 substrate-empirical leg CLOSES POSITIVE on Balance.** A
  same-`Φ` Bayesian planner recovers a substantial fraction of the
  oracle-vs-signature headroom; the off-set arm of the gate IS satisfiable
  from `Φ` history on this substrate.
- **Three-body remains the documented negative** for that particular
  substrate-cell-`Φ` triple. Both receipts stand; Phase 4 as a whole closes
  with the substrate split: positive on Balance, negative on three-body.
- **Phase 5 unblocked.** Phase 5 (cross-substrate operator sameness) requires
  ≥ 2 substrates; Balance now provides one (alongside the Mesa Phase-5
  reference cliff from [PHASE6_LAMBDA_CONTROL.md](PHASE6_LAMBDA_CONTROL.md)).
- **No change to Phases 1-3 (LQG / finite-MDP / boundary theorem).**

## 7. Run inventory

| Stage | Rows | Wall | Notes |
| --- | ---: | ---: | --- |
| Rate probe (1 seed × 128 particles × 0.5s horizon) | 1 | **2.3 s** | confirmed balance per-trial cost dominated by short control-loop, not planning depth |
| Full IAD shard fleet (8 seeds × 512 particles, 3-wide fan-out) | 8 | **20.6 s** | each shard 6.3-7.3 s; balanced, no straggler |
| Merge → `_iad-merged` (strict args + slate checks) | 8 trials × 4 modes | < 1 s | `balance-phase4-iad-merge.mjs` |
| Regret reducer (IAD-style, 2000-iter paired-bootstrap CI) | 8 paired rows | < 1 s | `balance-phase4-iad-regret.mjs` |
| **Total end-to-end** | | **~25 s** | |

**5000× faster than the three-body IAD** (which took ~30 h for the same
shape of work). Cart-pole dynamics + 2D particle belief is dramatically
cheaper than orbital mechanics + 6D state belief.

## 8. Sharded infrastructure

Mirrors the three-body IAD's 4-script pattern:

- [`scripts/balance-phase4-iad-shard.mjs`](../../scripts/balance-phase4-iad-shard.mjs)
  — single-seed wrapper, pins all IAD args except seed/particles/horizon, resume-safe.
- [`scripts/balance-phase4-iad-concurrent.mjs`](../../scripts/balance-phase4-iad-concurrent.mjs)
  — worker-pool orchestrator; writes the single-row cell-slate envelope.csv on first invocation; per-shard log capture; signal forwarding via local `trackChild`.
- [`scripts/balance-phase4-iad-merge.mjs`](../../scripts/balance-phase4-iad-merge.mjs)
  — strict-checked merger (verifies pinned harness args match across shards, no duplicate seeds, all completed).
- [`scripts/balance-phase4-iad-regret.mjs`](../../scripts/balance-phase4-iad-regret.mjs)
  — IAD-style mean regret + 95% paired-bootstrap CI from merged `bayes-regret.csv` + 3-branch verdict logic. Phase-15 reference (0.267) and accessible-mean fraction (0.5×) pinned as constants.

npm shortcuts:
`balance:phase4:iad:{shard,concurrent,merge,regret}`.

## 9. Scope guardrails

- **One cell, one signature, one floor.** Verdict is for this triple only.
  Generalizing to "Balance is Φ-accessible everywhere" requires running the
  same probe at other cells; the Phase-15 full lock already shows positive
  signal at 56/56 hard-gate cells, which is supportive but not the same
  IAD-style compute-unconstrained close.
- **Reference frame.** The 153% recovery vs Phase-15 reference is a finite-sample
  artifact (8 seeds skewing toward hard trajectories), not a meaningful
  enhancement of the underlying signal. The CI captures the variance honestly.
- **No Phase-6 lambda-confound control sibling.** Balance's signature path is
  observation-only (no mixed-objective optimizer-tuning knob). The optimizer
  diagnosis flagged in Phase 6 for Mesa does not apply here.

## 10. Cross-references

- [`COARSE_GRAINING_PROOF_ROADMAP.md`](../COARSE_GRAINING_PROOF_ROADMAP.md)
  ▸ Phase 4 — top-of-file status block + §Exit
- [`PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](PHASE4_BAYESIAN_FLOOR_BUILDOUT.md)
  — three-body buildout of the same gate (closed Privileged-only)
- [`PHASE4_THREEBODY.md`](PHASE4_THREEBODY.md) — three-body Phase-4 spec
  (substrate-specific entry condition)
- [`PHASE6_LAMBDA_CONTROL.md`](PHASE6_LAMBDA_CONTROL.md) — Mesa gating control
  (closed Branch 3, operating-envelope reference only)
- [`SUNDOG_V_BALANCE.md`](../SUNDOG_V_BALANCE.md) ▸ Bayesian Floor Profile —
  Balance same-information BF claim-lock receipt the Phase-15 lock backs
