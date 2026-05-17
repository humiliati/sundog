# Near-Anniversary Roadmap Triage

Status: 2026-05-17. This is a short selection note for work that could be
opened or advanced before the implied anniversary target of 2026-05-19. It
does not supersede any existing roadmap, pre-registration, or long-run rule.

Sources:

- [`internal/anniversary/postulations.md`](../internal/anniversary/postulations.md)
- [`internal/anniversary/attack_vectors.md`](../internal/anniversary/attack_vectors.md)
- [`internal/anniversary/fix_roadmap.md`](../internal/anniversary/fix_roadmap.md)
- [`SUNDOG_V_PATH.md`](SUNDOG_V_PATH.md)
- [`SUNDOG_V_BAYES.md`](SUNDOG_V_BAYES.md)
- [`proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md)
- [`proof/PHASE6_LAMBDA_CONTROL.md`](proof/PHASE6_LAMBDA_CONTROL.md)

## Selection Rule

Good near-anniversary work should satisfy at least two of these:

1. Produces a publishable or citeable artifact, not just a promise.
2. Lowers overclaim risk in an existing public claim.
3. Starts a high-value empirical lane without pretending the verdict has landed.
4. Fits the AGENTS.md capped-probe discipline: measure cheap, stage long.
5. Does not create a new public surface that competes with unfinished proof
   roadmaps.

## Shortlist

| Candidate | Near-anniversary deliverable | Value | Risk | Recommendation |
| --- | --- | --- | --- | --- |
| Pushable Occluder Phase 1 | Scene + oracle + non-photometric pusher test, with no flat-controller verdict yet | Very high public value; gives the first visible theorem-predicted boundary a body | Medium/high implementation risk in the older Python/MuJoCo substrate; clip capture may slip | Start only if scoped to Phase 1. Do not target Phase 2 verdict before the anniversary. |
| Bayesian floor rows across workbenches | Add hidden-state / observation / optimal-estimator / metric / falsifier rows to applicable roadmaps | High scientific hygiene; answers the standing Bayes attack without opening a culture-war track | Low for documentation; high for actual baselines | Best low-risk docs target. Actual baseline implementation is a later roadmap lane. |
| Three-body Bayesian floor repair | Run or stage the fresh shaped/guarded capped re-probe and record rate + floor-sanity result | High proof value; unblocks Phase 4 admission only if it passes | Current capped probes exceeded the inline budget and were non-decisive; full lock is multi-day | Do not run full. Treat as a runner/offload handoff unless a smaller sanity fixture is explicitly scoped. |
| Mesa Phase 6 lambda-control probe | Run the two 8-update PPO probe rows, record seconds/update, extrapolate full lock | High anniversary value: protects Postulate 2 cliff language and answers a named attack | Medium; PPO probe is estimated 2-5 minutes per row but may exceed local budget | Best empirical probe if the machine is free. If it exceeds 10 minutes, stop and stage. |
| Postulate 2 framing pass | Move "cliff as predicted boundary, not contradiction" into presentation-safe copy | Free and directly owed by `postulations.md` before 2026-05-19 | Low; wording-only, but must not imply Phase 6 has cleared | Do this if public anniversary text is being edited. Pair with an explicit "control still open" note. |
| Vortex / wishing-well toy | Phase 0 spec only: hidden state, signature, Bayes floor, falsifier | Good narrative fourth substrate from Postulate 5 | Medium narrative dilution; no evidence yet | Wait unless the goal is a speculative roadmap, not a result. |

## Recommended Path

The cleanest near-anniversary lane is a two-step move:

1. **Low-risk claim hygiene:** land the Bayesian-floor baseline rows across the
   applicable roadmaps. This directly addresses the standing defense in
   `attack_vectors.md` and the fix roadmap's Section 9 without creating a
   separate "Bayes vs Sundog" identity track.
2. **One empirical probe, if time is available:** run or stage the Mesa Phase 6
   capped probe. This is the smallest live experiment that can protect the
   anniversary cliff language. It does not close the proof trunk, but it tells
   whether the long lock is worth scheduling.

Pushable Occluder is the best **public artifact** candidate, but the honest
anniversary-scoped version is Phase 1 only: build the block scene and oracle
witness. The Phase 2 flat-controller verdict is the valuable result and should
not be rushed into a pre-anniversary claim.

## Candidate Details

### Pushable Occluder

Current state:

- Phase 0 pre-registration has landed at
  [`prereg/path-hypothesis.md`](prereg/path-hypothesis.md).
- [`SUNDOG_V_PATH.md`](SUNDOG_V_PATH.md) says Phase 1 is unblocked.
- No `occluder_block` code, oracle controller, or harness exists yet.

Near-anniversary scope:

- Add the scene object and 4D action surface.
- Add the oracle reachability witness.
- Add the pusher-is-not-photometric unit test.
- Record the two prereg amendments deferred from Phase 0: pusher embodiment and
  block dynamics parameters.

Do not claim:

- `BOUNDARY FOUND`.
- Flat controller failure.
- Hierarchical control repair.

Decision branch:

- If the oracle does not solve at least 90% of N=64 seeds, fix the apparatus.
- If the oracle passes, Phase 2 becomes the next real empirical target.

### Bayesian Floor Rows

This is the safest "slide in" target because the docs already say it is
soloable as roadmap insertion and non-soloable as implementation.

Applicable roadmaps:

| Surface | Hidden state | Observation channel | Bayesian-floor shape |
| --- | --- | --- | --- |
| Core photometric | Laser/source geometry and mirror state | Detector intensities plus joint/proprioception history | Particle/grid posterior over source/beam geometry; compare terminal intensity and time-to-acquisition |
| Three-body | Raw planar restricted state and envelope cell | Guarded accelerometer-proxy signature history | Particle-belief MPC as staged in `PHASE4_BAYESIAN_FLOOR_BUILDOUT.md` |
| Balance | Pole angle/velocity and cart state | Shadow geometry, cart state, action history | Belief filter over pole state from shadow observations; compare survival/recovery regret |
| Pressure Mines | Mine occupancy grid and pressure field parameters | Pressure ledger, revealed tiles, action history | Belief over mine occupancy / safe-action value; compare budget-adjusted safe progress |
| Future vortex | Euler-fluid microstate / source parameters | Low-dimensional envelope/vortex signature | Exact small-grid or particle posterior before any public claim |

Exit for this docs pass:

- Each applicable roadmap names the floor, the hidden state, the admitted
  observation, the metric, and what would falsify the Sundog advantage.
- Public copy can say "Bayesian-floor work is staged" without implying it has
  already run.

### Mesa Phase 6 Lambda-Control Probe

Why it is attractive:

- It answers a named anniversary attack: if the lambda cliff is optimizer
  path-dependence, the Postulate 2 cliff story should not ship.
- The spec already has capped-probe commands.
- The expected local probe is small enough to attempt under the 10-minute rule,
  but only if stopped promptly if it overruns.

Near-anniversary deliverable:

- Record seconds/update from the two 8-update probe rows.
- Extrapolate the six-row full lock.
- Decide whether to stage the long run on a runner.

Do not claim:

- Phase 6 cleared.
- Mesa cliff is a clean capacity law.

### Vortex Toy

The vortex/wishing-well toy is tempting because it is a visually clean fourth
substrate, but it should not be the next anniversary result. It is a Phase 0
spec candidate:

- define microstate;
- define coarse signature;
- define Bayesian floor;
- define the falsifier;
- define one cheap smoke.

It becomes high value after the Bayesian-floor discipline is installed, not
before.

## Decision

If the goal is the safest useful anniversary work: **Bayesian floor rows first,
Phase 6 capped probe second.**

If the goal is the best public visual artifact: **start Pushable Occluder Phase
1, explicitly labelled as oracle/apparatus work only.**

If the goal is a new speculative roadmap: **Vortex Phase 0**, but keep it out of
public claim copy until it has a baseline and falsifier.
