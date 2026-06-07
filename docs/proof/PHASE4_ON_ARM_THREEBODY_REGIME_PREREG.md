# Phase-4 ON-arm — Three-Body Regime Pre-Registration (FROZEN)

> 2026-06-06. **FROZEN before any run** (anti-p-hack: the regime choice + rationale below are fixed
> before a single number is read; the only new decision is the regime, and it is justified by physics,
> not tuned to produce `on`). Gates the staged three-body ON-arm run in
> [`PHASE4_ON_ARM_THREEBODY_STAGING.md`](PHASE4_ON_ARM_THREEBODY_STAGING.md) (Path A Node-shard fleet
> **or** Path B GPU port — this pre-reg is compute-path-independent). Sibling to the executed
> constructed-substrate close [`PHASE4_ON_ARM_FINITE_POMDP_SLATE.md`](PHASE4_ON_ARM_FINITE_POMDP_SLATE.md).

## 1. The one frozen decision: the regime pair

- **ON-candidate regime: `stable`** — the default bound near-circular orbit
  (`seededInitialParticle`: `x≈1, y≈0, vx≈0, vy≈1`, radiusScale=velocityScale=**1.0** nominal).
- **OFF anchor regime: `near_escape`** — the committed BF-4b cell exactly (`x≈1.45, vy≈1.45`,
  radiusScale **1.075**, velocityScale **1.15**), which already produced `on=0` / non-decisive `off`
  (`results/proof/phase4/phase4-regret-summary.csv`).

**Physical rationale (why `stable` is the ON-candidate, stated before the run):**
1. In a bound, regular orbit the test particle sits far from both hazard boundaries (escape `r>4`,
   close-approach). The hazard guard is mostly **off** → the guarded-signature thrust is `[0,0]`, and
   in a bound orbit **doing nothing is the optimal action** (no correction is needed). So the
   guard-off fiber has a **common optimal action `[0,0]`** that the signature determines → the
   optimal-action correspondence is constant on the fiber → `𝓕_σ`-measurable by **Phase-2 Theorem 1**
   → the canonical classifier should label these `on`, with regret → 0 (signature == Bayes floor ==
   do nothing).
2. When the guard *does* fire, the corrective thrust direction is set by the tidal **gradient**
   (`gradX, gradY`) — which is itself part of the fiber partition (`gradient_angle_sector`,
   `log_binned_gradient_magnitude`) — so those fibers are also fiber-constant candidates.
3. Contrast `near_escape`: chaotic sensitivity means two states in the **same** signature fiber can
   have **different** optimal thrusts (one escapes, one survives) → `⋂A* = ∅` → `off`. This is the
   measured `on=0` outcome; it is the principled OFF anchor.

## 2. What is UNCHANGED (not a p-hack surface)

Reused **verbatim** from the committed three-body Phase-4 receipt — *not* retuned:
- **Signature** `track_sensor_accel_guarded`; **Bayes floor** `bayes_floor_particle_mpc` (same
  observation budget).
- **Fiber partition** (`threebody-phase4-regret.mjs` / committed manifest): keys
  `guard_t, log_binned_abs_tidal_magnitude, gradient_angle_sector, log_binned_gradient_magnitude,
  sensor_noise_std`; `fiberMinSamples=20`; `commonActionRule = exact action_key match` (zero action =
  its exact key `0:0`); `undecidableRule = no fiber with ≥ fiberMinSamples Bayes-reached samples`.
- **Bootstrap**: 2000-iter paired, `bootstrap_seed=40604`, `classSeedRule = seed + classIndex*1009`.
- **MPC knobs**: `particleCount=512`, `planningHorizonSteps=800`, `candidateHoldSteps` and guards as
  the committed BF-4b cell.

## 3. Frozen run cell + seeds

- **Envelope (both regimes, except the regime-natural scales in §1):** `massRatio=1`, `dt=0.01`,
  `thrustLimit=0.4`, `sensorNoiseStd=0.01`, guards `gq=0.75, gminr=1.15, gaccel=2.5, gtidal=35`,
  `duration=16` → ~1600 control steps/trial.
- **Seeds:** `seedStart=0`, **`seeds=16`** per regime (2× the committed 8, to give `stable`'s
  fibers a chance to clear `fiberMinSamples=20` despite its lower hazard-event rate). Raising seeds is
  a sample-count lever only (does not bias `on`/`off`); if `stable` returns `undecidable`, raise to 32
  and re-run — this is named here, so it is not a post-hoc move.

## 4. Frozen gate (roadmap verbatim, NOT retuned)

`on` regret → 0 within bootstrap CI **AND** `off` regret CI excludes 0 (bounded away)
(`COARSE_GRAINING_PROOF_ROADMAP.md` §Phase 4).

**Verdict branches:**
- **ON-arm PASS (the target):** `stable` populates `on` (≥20 fiber-constant samples) with regret → 0
  within CI, AND `near_escape` `off` CI excludes 0 → **first real-substrate ON-arm**; the two-sided
  gate fires on a real continuous substrate.
- **Sufficiency-false:** `stable`'s measurable fibers show regret bounded-away > 0 → the Bayes floor
  systematically out-thrusts the do-nothing signature even in bound orbits → halt and falsify the
  real-substrate sufficiency claim (report honestly).
- **Not-measurable null:** `stable` returns no `on` fibers (all `off`/`undecidable`) → **even a bound
  regular orbit is not cleanly `𝓕_σ`-measurable under this signature** → a real-substrate boundary
  (report, do **not** retune the partition or regime). This is the honest prior outcome.
- **Under-sampled:** `stable` `undecidable` (fibers < 20) → raise seeds per §3, re-run once.

## 5. Honest prior + scope

**Prior:** plausible `on` via the guard-off `[0,0]` fibers, but a real chance of `off`/`undecidable`
— the program has never banked a real-substrate `on`, and three-body's chaotic sensitivity may defeat
fiber-constancy even in the stable regime. Either outcome is informative and pre-named (§4).

**Scope (carried into the receipt):** a real-substrate **sufficiency** result (Blackwell
control-sufficiency on the measurable set), still **NOT** body-resistance / regime-2 and **NOT** a
trained body. It tests whether the executed constructed-POMDP `on`
([`PHASE4_ON_ARM_FINITE_POMDP_SLATE.md`](PHASE4_ON_ARM_FINITE_POMDP_SLATE.md)) survives on a real
continuous substrate. Caveat to state plainly if it passes: a `[0,0]`-dominated `on` is a *weak*
sufficiency (the signature is sufficient largely because the optimal action is "do nothing"); the
discrimination against the `near_escape` OFF anchor is the load-bearing content, not the `on` value.

---

*Sundog Research Lab — Phase-4 ON-arm three-body regime pre-registration. FROZEN: ON-candidate
`stable`, OFF anchor `near_escape`; canonical partition/classifier/controllers unchanged. The regime
is the only new choice and it is justified by physics, not tuned. Run is compute-path-independent
(Path A or B). Sufficiency, not resistance; real substrate; honest prior is a possible null.*
