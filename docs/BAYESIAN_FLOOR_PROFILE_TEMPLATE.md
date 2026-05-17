# Bayesian Floor Profile Template

Status: 2026-05-17. This is a reusable profile contract for installing a
Bayesian floor inside Sundog workbenches without opening a separate
`bayes_v_sundog` track for each surface.

The profile is deliberately modular. Each workbench should fill the same slots:
truth state, admitted observation, objective, baseline policy, belief method,
receipts, and gates. The code may differ by substrate; the audit shape should
not.

Primary references:

- [`proof/PHASE4_THREEBODY.md`](proof/PHASE4_THREEBODY.md)
- [`proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md`](proof/PHASE4_BAYESIAN_FLOOR_BUILDOUT.md)
- [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
- [`SUNDOG_V_BAYES.md`](SUNDOG_V_BAYES.md)
- [`ANNIVERSARY_ROADMAP_TRIAGE.md`](ANNIVERSARY_ROADMAP_TRIAGE.md)

## Three-Body Status

Three-body is the current pattern source, but it is not yet a closed floor.

Implemented or staged:

| Piece | Current state | Reusable lesson |
| --- | --- | --- |
| Admission spec | [`proof/PHASE4_THREEBODY.md`](proof/PHASE4_THREEBODY.md) pins `X`, `Phi`, `J`, `mu`, regret, and the on/off cell gate. | Every floor needs a profile before it gets a runner. |
| Harness safety | `scripts/threebody-mode-validation.test.mjs` passes; unknown controller modes throw loudly. | Unknown modes must fail before rollout. Silent fallback is a floor-killer. |
| Observation contract | `observeGuardedAccelSignature` exists in `public/js/threebody-core.mjs`; `scripts/threebody-signature-observation.test.mjs` passes. | The admitted observation must be a pure function with a parity fixture. |
| Floor evaluator | `scripts/threebody-phase4-bayes-floor.mjs` writes `manifest.json`, `signature-observations.jsonl`, `belief-diagnostics.csv`, `bayes-actions.csv`, and `bayes-trial-outcomes.csv`. | Keep the floor as a separate evaluator until no-leak and runtime gates pass. |
| Belief method | Particle-belief MPC over same-signature history, fixed particles, action lattice, deterministic seeds. | Continuous surfaces should start with particles unless an exact grid is cheap. |
| Same-information guard | The guarded-signature policy is candidate `0`; the floor deviates only above a dt-scaled predicted advantage. | A floor approximation should not be allowed to become worse than the same-information baseline without a sanity failure. |
| Regret reducer | `scripts/threebody-phase4-regret.mjs --self-test` passes; reducer joins Bayes and signature rows, writes regret summaries and cell fibers. | Reducers should be reusable CSV/JSON joiners, not ad hoc notebook math. |
| Capped probes | Two BF-4 probes exceeded the inline budget and were non-decisive; the latest repair needs a fresh re-probe. | Runtime and floor-sanity receipts decide whether the long lock moves to a runner. |

Current blocker:

- No passing capped re-probe after the signature-baseline guard.
- No full-lock handoff in `PHASE4_THREEBODY.md`.
- No Phase 4 empirical admission yet.

## Profile Contract

A Bayesian-floor profile is a compact admission contract for one workbench. It
should be short enough to paste into a roadmap, but precise enough that a future
runner can implement it without moving the goalposts.

### 1. Header

```yaml
profile_id: <surface>-bayesian-floor-v1
surface: <Core Photometric | Three-Body | Balance | Pressure Mines | ...>
owning_roadmap: docs/<roadmap>.md
status: staged | implemented-unprobed | probed-nondecisive | admitted | retired
purpose: <one-sentence reason this floor exists>
not_a_claim: <what this floor does not prove>
```

### 2. Truth State

Pin the hidden state the workbench truth uses.

Required fields:

- `X`: raw decision state and fixed cell/config parameters.
- hidden variables that are forbidden to the signature controller.
- which truth fields may be logged for audit only.
- initialization distribution and seed rule.

Template:

```text
X_t = (<state_t>, t, c)
hidden_for_signature = [...]
truth_logging_allowed_for = [audit, regret_readout, fixture_tests]
truth_forbidden_for = [floor_action_selection, signature_controller]
mu = uniform over <cell slate> x <seed slate>
```

### 3. Admitted Observation

Pin the observation history the floor and the signature controller may use.

Required fields:

- `Phi_t`: admitted signature fields.
- `h_t`: whether the floor receives only current observation or full admitted
  history.
- noise model and seed convention.
- pure observation function or planned function.
- parity test: same fixture, same observation as the deployed signature
  controller.

Template:

```text
Phi_t = (<field_1>, <field_2>, ..., <guard/confidence>, <sensor metadata>)
h_t = [Phi_0..Phi_t, admitted actions, admitted config]
pure_observation_function = <module.function>
parity_fixture = <test or target test>
```

### 4. Objective And Regret

Pin the objective before any floor run.

Required fields:

- `J`: task objective.
- `T_max` or equivalent normalization.
- primary regret formula.
- tie-breakers that do not alter the primary claim.
- what counts as a floor sanity failure.

Template:

```text
J(pi) = E_mu[<normalized outcome>]
regret_i = (J_i(pi_bayes_floor) - J_i(pi_signature)) / <normalizer>
tie_break = <fuel/action count/compute only when primary outcomes tie>
floor_sanity = negative_regret_rate <= 0.05
```

### 5. Floor Policy

Pin the Bayesian approximation as an implementable policy.

Required fields:

- exact grid, Kalman/EKF, particle filter, or other belief method.
- prior / particle initialization.
- likelihood or update rule.
- action candidate set.
- planning objective.
- deterministic tie order.
- same-information baseline guard, if approximation can be weaker than the
  deployed signature policy.
- final approximation settings and how they are locked.

Template:

```text
floor_policy = <exact_grid | ekf | particle_mpc | belief_dp | posterior_table>
belief_state = <posterior over hidden state or hypotheses>
update_rule = p(Phi_t | particle_or_state, c)
action_candidates = [same_information_signature_action, ...lattice_or_actions]
planning_score = E[primary_outcome within horizon + bounded shaping term]
tie_order = [same-information guard, primary score, secondary cost, fixed order]
settings_to_lock_after_probe = [...]
```

### 6. Comparators

Every profile should name the same families even if only some are implemented.

Required rows:

| Family | Role | Information regime |
| --- | --- | --- |
| Signature / Sundog | The controller being evaluated. | Same admitted `Phi` history, no posterior over hidden causes unless the roadmap says otherwise. |
| Bayesian floor | Serious partial-observation baseline. | Same admitted observation plus explicit belief over hidden causes. |
| Oracle | Privileged ceiling / apparatus sanity. | May read truth; never the primary Bayes comparator. |
| Naive / random / passive | Lower bound and leakage check. | Public or minimal observation only. |
| Hybrid | Optional repair lane. | Posterior only at ambiguity gates, if the roadmap includes it. |

### 7. Receipts

Use stable filenames across surfaces unless a surface has a strong reason not
to.

Target output shape:

```text
results/<surface>/<phase-or-floor>/
  manifest.json
  signature-observations.jsonl
  belief-diagnostics.csv
  bayes-actions.csv
  bayes-trial-outcomes.csv
  regret.csv
  regret-summary.csv
  cell-fibers.json
```

Required manifest fields:

- schema;
- `startedAt`, `completedAt`;
- exact args;
- git SHA if available;
- profile id and profile status;
- hidden-state statement;
- observation statement;
- belief method and approximation settings;
- action candidate order;
- tie order;
- floor-sanity threshold;
- runtime summary.

### 8. Gates

Minimum gates:

| Gate | Pass condition |
| --- | --- |
| Unknown-mode / unknown-lane | Bad controller or evaluator names fail loudly before rollout. |
| No-state-leak | Truth state is used only for particle simulation, fixtures, and readout; never direct action selection. |
| Observation parity | The recorded `Phi_t` equals the signature controller's own admitted observation on a fixed fixture. |
| Planner sanity | In a deliberately easy or enriched fixture, the floor chooses an action consistent with the expected posterior optimum. |
| Floor sanity | Negative regret above the pre-registered threshold voids the run as a floor. |
| Runtime | Capped probe records rate; long runs are staged when they exceed the inline budget. |
| Claim gate | Public language is weaker than or equal to the profile's current status. |

### 9. Outcome Branches

Every profile should pre-register branches:

1. **Floor invalid.** Repair the floor; do not interpret the workbench result.
2. **Bayes dominates.** Narrow Sundog claim to heuristic or response-control
   niche.
3. **Signature matches Bayes on admitted set.** Strengthens
   sufficiency-for-control on that substrate.
4. **Signature beats a weak Bayes approximation.** Strengthens nothing until the
   Bayes floor is repaired or replaced.
5. **Ambiguous / underpowered.** Add one pinned midpoint or one stronger fixture
   before changing public status.

## Copy-Paste Profile Skeleton

```markdown
## Bayesian Floor Profile

Profile id: `<surface>-bayesian-floor-v1`
Status: `staged`
Owning roadmap: `<doc path>`

### Purpose

<Why this floor is needed and what it does not prove.>

### Truth State

- `X_t = ...`
- Hidden from signature controller: ...
- Truth logging allowed only for: ...
- Seed / cell slate: ...

### Admitted Observation

- `Phi_t = ...`
- History available to floor: ...
- Pure observation function: ...
- Observation parity fixture: ...

### Objective And Regret

- `J(pi) = ...`
- `regret_i = ...`
- Tie-break: ...
- Floor-sanity threshold: ...

### Floor Policy

- Belief method: ...
- Prior / particles / grid: ...
- Likelihood or update rule: ...
- Action candidates: ...
- Same-information baseline guard: ...
- Tie order: ...
- Settings locked after capped probe: ...

### Comparators

| Mode | Role | Information |
| --- | --- | --- |
| signature | evaluated controller | ... |
| bayes_floor | serious partial-observation baseline | ... |
| oracle | privileged apparatus check | ... |
| naive/random/passive | lower bound | ... |

### Receipts

- `manifest.json`
- `signature-observations.jsonl`
- `belief-diagnostics.csv`
- `bayes-actions.csv`
- `bayes-trial-outcomes.csv`
- `regret.csv`
- `regret-summary.csv`
- `cell-fibers.json`

### Gates

- Unknown mode:
- No-state-leak:
- Observation parity:
- Planner sanity:
- Floor sanity:
- Runtime:
- Claim gate:

### Outcome Branches

1. Floor invalid:
2. Bayes dominates:
3. Signature matches Bayes:
4. Weak-Bayes artifact:
5. Ambiguous:
```

## First-Wave Surface Profiles

These are not implementations. They are the recommended profile shapes to stamp
into each roadmap.

| Surface | Best first floor form | Why | Notes |
| --- | --- | --- | --- |
| Three-body | Particle-belief MPC | Continuous hidden state and existing particle-friendly simulator. | Current pattern source; blocked on passing re-probe after same-information guard. |
| Balance | Low-dimensional particle or EKF floor over pole state | Hidden state is only four-dimensional and `sampleShadowSensor` already isolates the observation. | The staged profile now lives in [`SUNDOG_V_BALANCE.md`](SUNDOG_V_BALANCE.md#bayesian-floor-profile). Candidate actions can be a fixed force lattice plus `sundog_shadow`. |
| Pressure Mines | Frontier-limited occupancy belief / particle board posterior | Discrete hidden mine grid; actions are already enumerated as reveal/flag/scan. | The staged profile now lives in [`sundog_v_minesweeper.md`](sundog_v_minesweeper.md#bayesian-baseline-profile). Exact full-board posterior may be too large; start with frontier particles and explicit approximation limits. |
| Core photometric | Particle or coarse grid posterior over source/beam geometry | The scientific spine needs the comparator, but MuJoCo/Python makes this heavier. | Start after a JS workbench proves the template. Keep oracle DOA separate. |
| Pushable Occluder | Same as core photometric plus occluder pose belief | High-value boundary, but only after the scene/oracle apparatus exists. | The floor should not run before Phase 1 apparatus passes. |
| Geometry / Perception | Estimation-floor profile, not control-floor profile | The action may be route selection, abstain, or request more evidence rather than physical control. | Use the template only if `J`, action set, and admissible observation are pinned. |
| Vortex toy | Exact small grid first, particles second | New substrate can be designed around the template from day one. | Good Phase 0 spec candidate; no public claim before floor and falsifier. |

## Minimal Code Pattern

The reusable code seam is not "one generic Bayesian agent." It is a small set
of adapters per surface plus one shared reducer shape.

Surface adapter functions:

```text
makeProfileConfig(args) -> normalized config
enumerateCells(args) -> cell[]
seedInitialTruth(cell, seed) -> X_0
observeSignature(truth, config, runtime) -> Phi_t
stepTruth(truth, action, config) -> X_{t+1}
enumerateActions(Phi_t, config) -> action[]
signatureBaselineAction(Phi_t, history, config) -> action
scoreOutcome(history, config) -> J_i
terminalOrCap(history, config) -> boolean
```

Generic floor loop:

```text
for cell, seed:
  initialize truth
  initialize belief from profile prior
  while not terminal:
    Phi_t = observeSignature(truth)
    update belief with Phi_t
    candidates = [signatureBaselineAction(Phi_t), ...enumerateActions(Phi_t)]
    action = choose by profile tie order
    log Phi, belief diagnostics, action scores
    truth = stepTruth(truth, action)
  write outcome
```

Shared reducers:

```text
join by profile cell id + seed
regret = (J_bayes_floor - J_signature) / normalizer
classify fibers from logged Phi/action rows where applicable
bootstrap by cell class
apply negative-regret sanity gate
```

This keeps the parts that must be domain-specific domain-specific, while making
the audit trail, receipt names, and claim gates look the same everywhere.
