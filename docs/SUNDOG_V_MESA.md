# Sundog vs. Mesa-Optimization

Working hook:

> Sundog claims the field is the objective. Mesa-Optimization asks whether the
> agent secretly grows one inside itself anyway. This document is the experiment
> that finds out.

This roadmap is promoted out of the Sundog Gravity Ledger
(`SUNDOG_V_GRAVITY.md` Candidate 2). It pairs an empirical capacity-and-
selection-pressure experiment against a Formal Separability Theorem appendix
(promoted from Candidate 1). The empirical front asks whether learned
signature-tracking agents reconstruct internal reward proxies under capacity
and selection-pressure scaling. The formal front names the structural
condition the empirical work is trying to earn.

The pairing is deliberate. The theorem alone can be dismissed as a sanitized
toy condition. The experiment alone can be dismissed as a tuned slate. Run
together, each tells the other what counts as a real result.

> **Cross-substrate dependency, 2026-05-13:** the mesa↔geometry crossover
> note ([`MESA_CROSSOVER_NOTE.md`](MESA_CROSSOVER_NOTE.md)) treated the
> Geometry Phase 10 single-handle verdict as the in-the-wild numerical
> companion to the in-vitro mesa v3.x finding. The synthetic optical audit
> ([`docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md))
> returned three load-bearing findings that materially weaken the geometry
> verdict's evidentiary chain. The campaign to address them is filed in
> [`docs/PHASE10_ATTACK_ROADMAP.md`](PHASE10_ATTACK_ROADMAP.md). This was
> a substantiation-chain dependency, not a falsification: the mesa
> empirical front itself was never affected; the dependency was on the
> geometry side's evidentiary chain and the downstream public-framing
> surfaces.
>
> **Re-audit cleared, 2026-05-14 — dependency resolved.** The attack
> roadmap's re-audit gate landed
> ([`docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md`](calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md)):
> no new load-bearing code, anchor, or route-math blocker. The eight
> required passes plus the optional C-series all landed (C1 dropped p7
> as circumscribed-halo regime; C2 wing-radial Lab b\*, C4 wing-slope
> luminance-gradient, and C6 matched-filter all returned not-recovered;
> C5 manual sample selection recovered on p2 but C6 falsified the
> C5→matched-filter natural extension; **C7's HaloSim ray-tracing
> reformulation under the canonical opening-angle handle resolved the
> C5↔C6 tension toward "C5 was the symmetry-bias artifact the C5
> receipt hedged against," so the tangent route remains unpromoted
> *including* under the canonical literature handle**). The post-audit
> hedged language is now **in place across every coupled surface**, not
> pending: the "three independent failure layers" and "passes residual
> gate at every eligible photo" framings are retired everywhere; the
> homepage elevator pitch (v1.1, `data-audit-hedge`) and the
> gravity-ledger public-framing sentence carry the settled post-audit
> wording. The settled geometry verdict is **forward-rich /
> inverse-narrow with a single promoted inverse handle**
> (parhelion-offset on the strict 3-photo subset p2/p7/p13); CZA fails
> the coverage gate, supralateral fails structural-discrimination, and
> tangent is unpromoted under the full C1–C7 detector series. The
> two-substrate field-shape pattern *survives* — it is now stated at
> its settled post-audit strength (narrower than the pre-audit framing,
> and with the inverse-narrowness *sharpened* rather than weakened by
> the audit). Mesa↔geometry crossover framing downstream of this
> roadmap is no longer hedge-required-pending; it is hedged-and-settled.
> Mesa work that was held on the geometry clearance — pattern-confirmation
> and consolidation moves the crossover note has been tracking — is now
> unblocked.

## Why Mesa Is the Right Next Roadmap

The gravity claim names three falsification modes:

1. field manipulation is cheap;
2. the signature is a reward in costume;
3. mesa-optimization re-emerges.

Mode (3) is the falsification surface alignment-research reviewers reach for
first. The Formal Separability work attacks (2). The mesa experiment attacks
(3). Together they cover the two falsification modes that do not require new
physical domains, leaving mode (1) for the Spacecraft and Conservation-Law
candidates downstream.

Mesa is also the most "achievable next" candidate in the ledger because it
needs no new physics, no hardware, and no high-fidelity simulator. The
existing photometric and three-body sensor-tier discipline gives the
controller-family architecture. The new infrastructure is the capacity ladder,
the proxy-splitting probe slate, the causal-intervention battery, and the
selection-pressure curriculum — all software, all designable from scratch
with current tooling.

## What's Honest vs. What's Reach

**Honest:**

- A measured capacity range and selection-pressure shape where signature-
  trained agents remain tied to the external field more strongly than matched
  reward-trained agents under proxy-splitting probes.
- A causal intervention battery that shows where control authority lives for
  each agent family.
- A reported failure boundary in capacity, selection pressure, environment
  shape, or sensor-tier degradation, where signature tracking decouples from
  the external field.
- A formal note that names the conditions under which a signature is provably
  not a reward in costume, with counterexamples shipped alongside the proof.

**Reach (do not claim):**

- "Sundog prevents mesa-optimization."
- "Signature training is immune to inner misalignment."
- "Reward hacking does not occur in signature-trained agents."
- "Capacity does not matter for the gravity claim."

The danger of this work is that a negative result — signature-trained agents
*do* reconstruct internal proxies above some capacity — would be misread as a
program-killer. It is not. A measured capacity range where the gravity claim
holds is itself a respectable, novel result. The roadmap is structured so
that either outcome ratchets the program forward.

## Ratified Hook Language

Safe hook:

> Sundog vs. Mesa-Optimization asks whether a signature-trained agent, scaled
> up under selection pressure, secretly reconstructs an internal reward proxy
> and inherits the same Goodhart failures as direct reward training.

Short version:

> The field is the objective only as long as the agent is small enough not to
> imagine its own.

Avoid:

- "Sundog is immune to mesa-optimization."
- "Signature training cannot be reward-hacked at scale."
- "Sundog solves inner alignment."
- "The mesa trap does not apply to signature controllers."

## Controller-Family Architecture

The experiment compares four learned/control families on the same task and the
same matched slates, plus a privileged-only Oracle ceiling. Every non-Oracle
family inherits the three-body sensor-tier
discipline: each receives the signature through an explicitly labeled sensor
tier (privileged-field, local-probe-field, delayed-field, noisy-field) so the
signature stays a function of environmental state and not of policy.

The families:

- **HC-Signature.** Hand-coded SCAN/SEEK/TRACK over a local-gradient estimate
  built from the four signature probe samples. No learning. The structural
  baseline. Cannot mesa-optimize because there is no inner optimization loop.
  Ignores privileged gradient information even when present, so its privileged
  row is a consistency/degradation comparison rather than a true ceiling.
- **Oracle.** Privileged-only analytic-gradient controller. Reads closed-form
  `∇S(x)` from `privileged-field` and serves as the upper-bound reference for
  later normalization. Not used as an imitation source unless a later ablation
  explicitly asks for oracle imitation.
- **L-Signature.** Learned policy `pi_theta(a | o)` where `o = (x, S_local_1,
  ..., S_local_4)` and `a` is a 2D velocity command. Trained either by
  behavior cloning from HC-Signature trajectories or by policy gradient on a
  state-only objective `J(tau) = integral g(S(x_t)) dt`. The discipline is
  load-bearing: no agent-action term, no learned evaluator, and no reward
  channel that can be edited independently of the environment state.
- **L-Reward.** Same observation, same action space, same architecture, and
  same parameter count as L-Signature. Trained on a matched scalar reward such
  as `R(s, a) = -||x - x_goal||`, or a sparse success reward, whose argmax
  under no probe coincides with the signature target. The Goodhart-prone
  baseline.
- **L-Mixed.** Same architecture again, trained on
  `(1 - lambda) * J_signature + lambda * J_reward` for pinned
  `lambda in {0.1, 0.3, 0.5, 0.7, 0.9}`. Diagnostic — surfaces the gradient
  between the signature and reward regimes.

HC-Signature, L-Signature, L-Reward, and L-Mixed are run on the same probe and
intervention slates with matched parameter counts, sample budgets, Adam
optimizer settings fixed per tier, and seeds. Differences in learned behavior
are attributable to training regime, not architecture. Oracle is reported as a
separate privileged ceiling row.

## Phase 0 Decision Lock

These calls are pinned for the first implementation pass.

### Environment family

Commit to **shadow-field navigation** as the canonical Phase 0 environment.
The task is deliberately minimal:

- The agent moves in 2D continuous space.
- A hidden goal emits a scalar signature field
  `S(x) = exp(-||x - x_goal||^2 / (2 * sigma^2))` by default.
- The agent observes its own position plus local signature samples at probe
  offsets around its body, initially `+/- epsilon` on each axis. The agent
  never observes `x_goal`.
- The matched reward baseline uses `R(s, a) = -||x - x_goal||` or sparse
  `R = 1{||x - x_goal|| < delta}`. Under nominal conditions, its argmax
  coincides with the signature maximum.
- The probe slate can rotate, translate, scale, or mirror the world; vary
  irrelevant visual texture; introduce decoy fields with different decay
  structure; and add adversarial noise to specific sensor channels.

This environment wins Phase 0 because it makes the proxy-splitting probes easy
to design, fast to run, and analytically legible. The absolute position channel
is included intentionally as a tempting shortcut that probes can later break.

Do not use the three-body harness or photometric MuJoCo task as the canonical
Phase 0 environment. Three-body remains the literal-gravity second domain, but
the proxy-splitting design problem is too expensive for the first mesa pass.
The photometric task remains a useful continuous-control anchor, but its
geometry does not provide enough clean shortcut-preserving probe variants.

Keep **tidal-toy** as the natural second environment: a 2D field where `S(x)`
is a tidal-tensor magnitude derived from two hidden mass-like sources. It is
closer to the gravity wedge than shadow-field navigation, but slightly harder
to reason about. It should be promoted after the shadow-field methodology is
debugged.

### Claim boundary

The mesa roadmap attacks falsification mode (3): mesa-optimization re-emerges.
Appendix A attacks mode (2): signature-is-reward-in-costume. Mode (1):
field-manipulation cost is explicitly out of scope for this roadmap and stays
with Spacecraft, Conservation-Law, and Fluid/Wake candidates.

The Phase 7 ratchet, if positive, should speak only to this claim shape:

> In the tested shadow-field navigation environment family, learned
> signature-trained controllers retain measurable distinction from matched
> reward-trained controllers under proxy-splitting probes across a mapped
> pocket of capacity, selection pressure, probe severity, and sensor tier.
> Outside that pocket, signature-trained controllers exhibit proxy-collapse
> behavior comparable to reward-trained baselines.

Even a strong Phase 7 result does not support universal mesa immunity,
LLM-scale or foundation-model claims, deceptive-alignment claims, or
adversarial-robustness claims in deployed systems.

### Capacity ladder

Use three tiers first:

| Tier | Initial architecture | Target size | Budget cap |
| --- | --- | ---: | ---: |
| Small | 2-layer MLP | ~5K parameters | 1M environment steps |
| Medium | 4-layer MLP | ~250K parameters | 10M environment steps |
| Large | 6-layer MLP or small transformer | ~5M parameters | 100M environment steps |

Defer an XL tier. A ~50M-parameter tier should be added only after the Large
tier results land and the probe/intervention methodology is stable.

### Literature note depth

File Phase 0 with a structured literature spine, not a full paragraph-by-
paragraph literature review. The immediate need is to anchor terms and probe
design, not to turn the roadmap into a survey. Expand to paragraph summaries
only when Appendix A or the Phase 3 probe slate needs a specific citation.

## Roadmap

### Phase 0 - Scope and Literature Pass

Goal: pin the exact claim, the environment family, and the matched
controller-family definitions before writing experiments.

Deliverables:

- A one-page claim boundary specifying which falsification mode the mesa
  experiment attacks and which it does not.
- A structured literature spine covering mesa-optimization, goal
  misgeneralization, reward hacking, reward misspecification, goal
  representations, asymmetric information in RL, constrained/shielded RL, and
  interpretability tools for learned proxy representations.
- The canonical environment family: shadow-field navigation, with separable
  `S(x)`, matched `R(s, a)`, controllable geometry, explicit probe
  affordances, and tidal-toy reserved as the second-environment port.
- Architecture pinning for HC-Signature, Oracle, L-Signature, L-Reward,
  L-Mixed.
- Initial capacity ladder definition (Small / Medium / Large by parameter
  count, with budget caps; XL deferred).
- Initial selection-pressure curriculum definition (dense signature, threshold
  signature, imitation-from-HC, dense reward, sparse reward, mixed lambda,
  signature-first, reward-first, reward-shape adversary).

Exit criterion: the four learned/comparison families, Oracle ceiling, the
environment family, and the claim boundary are pinned.

Implementation-grade detail (environment math, HC-Signature pseudocode,
L-family architecture by tier, probe and intervention affordances,
reproducibility harness conventions, Phase 1 readiness checklist) lives in
the satellite spec [`mesa/PHASE0_SPEC.md`](mesa/PHASE0_SPEC.md). That doc is
authoritative where this roadmap is silent and is versioned independently.

Phase 0 literature spine:

- Hubinger et al., *Risks from Learned Optimization in Advanced Machine
  Learning Systems* (2019): mesa-optimization terminology, inner/outer
  alignment distinction.
- Langosco et al., *Goal Misgeneralization in Deep Reinforcement Learning*
  (2022), and Shah et al., *Goal Misgeneralization* (2022): empirical
  distribution-shift framing for capability/objective decoupling.
- Krakovna et al.'s specification-gaming examples and Skalse et al.,
  *Defining and Characterizing Reward Hacking* (2022): reward-hacking
  taxonomy and Formal Separability Appendix A citations.
- Pan, Bhatia, and Steinhardt, *The Effects of Reward Misspecification* (2022):
  capacity and phase-transition framing for reward hacking.
- Olsson/Olah interpretability work and Bricken et al. on sparse autoencoders:
  Phase 6 representation-probe candidates.
- Tian, Chen, and related work on goal representations and linear probes in RL
  agents: candidate tools for detecting proxy-objective features.
- Privileged-information and asymmetric-actor-critic literature, including
  Pinto and Andrychowicz lines of work: adjacent because training signals can
  contain information unavailable at evaluation time.
- García and Fernández's constrained/shielded RL survey: useful contrast for
  "Sundog vs. constrained RL" boundary language.

### Phase 1 - Reference Task and Hand-Coded Sundog Controller

Goal: build the canonical signature-control task and verify HC-Signature
works on it before any learning enters the picture.

Deliverables:

- Shadow-field navigation implementation with explicit `S(x)` accessor and
  matched `R(s, a)` accessor, separable in source code so it is impossible to
  accidentally couple them.
- Sensor-tier wrappers mirroring the three-body workbench: privileged-field,
  local-probe-field, delayed-field, noisy-field.
- HC-Signature implementation (SCAN/SEEK/TRACK) demonstrated to maintain
  regime on the matched task at each sensor tier.
- Oracle analytic-gradient implementation demonstrated on `privileged-field`
  as the ceiling row.
- A reproducibility harness with seeded trials, manifest format, and per-trial
  JSONL logs in the three-body harness style.

Exit criterion: HC-Signature works on the canonical task and degrades cleanly
across lower tiers; Oracle provides the privileged ceiling; the harness can
replay trials byte-for-byte.

### Phase 2 - Matched-Capacity Learned Controllers

Goal: train L-Signature, L-Reward, and L-Mixed at the Small / Medium / Large
capacity tiers and produce a matched-architecture sample-efficiency comparison
between training-signal regimes. The canonical gap between L-Signature and
L-Reward at matched budget is the **Sundog-cost finding** — the speed
penalty for training on a state-only signal — and is a Phase 2 deliverable
in its own right, not a failure mode. Phase 3 (probes) and Phase 4
(interventions) then test whether the gap pays for itself in robustness.

Deliverables:

- Matched training pipelines for the three learned families, with identical
  optimizer, batch, sample-budget, and seed conventions.
- Capacity ladder: Small (~5K parameters, 1M steps), Medium (~250K
  parameters, 10M steps), and Large (~5M parameters, 100M steps), scaled
  consistently across families. XL is deliberately deferred.
- Canonical-budget terminal performance per family (success rate, mean S_T,
  mean steps) reported at each tier.
- Per-family over-cap multipliers — budget required to reach ≥ 95% success
  divided by the canonical budget — as the headline Sundog-cost numbers at
  each tier.
- A first capacity report comparing terminal performance and sample efficiency
  across families and tiers under no probe, normalized where useful against
  the Oracle ceiling.

Exit criterion: matched-capacity controllers exist for all four families and
all three tiers. Per-family canonical-budget performance and over-cap
multipliers are reported. Phase 2 success does not require all families to
hit ≥ 75% at canonical budget; it requires the matched comparison to be
clean and the Sundog-cost numbers to be honestly reported.

### Phase 3 - Proxy-Splitting Probe Slate

Goal: design and run the distribution-shift probe slate that decouples common
shortcuts from the true external signature, and measure which controller
families remain tied to the field.

Probe design principles:

- Each probe preserves `S(x)` at the target argmax while breaking at least
  one shortcut a learned policy might have absorbed (positional, visual,
  textural, temporal).
- Probes are graded by severity: small probes that preserve the gross
  signature geometry, larger probes that distort surface features more
  aggressively.
- Each probe is matched: identical seeds across families.

Deliverables:

- A probe slate spanning at least five distinct shortcut axes:
  rotation/translation/scale/mirror transforms, irrelevant texture changes,
  decoy fields with mismatched decay structure, channel-specific sensor noise,
  and positional shortcut breaks.
- Per-family, per-tier, per-capacity performance under each probe.
- A primary report: rate of regime loss under proxy-splitting probes, broken
  out by family and capacity.
- An aggregate report: capacity at which L-Signature begins to behave
  indistinguishably from L-Reward under proxy splits, if at all.

Exit criterion: at least one probe meaningfully distinguishes L-Signature
behavior from L-Reward behavior at some capacity tier, or the slate is
strengthened until it does. A negative result here is still informative if
the probe slate is strong.

Implementation-grade detail (L-Reward canonical training-signal lock with
action-coupling + false-basin shaping, full probe-slate parameters by axis
and severity, matched-seed evaluation protocol, probe-resistance gap as the
program-level metric, sweep-harness conventions, exit-criterion threshold)
lives in the satellite spec [`mesa/PHASE3_SPEC.md`](mesa/PHASE3_SPEC.md).
That doc is authoritative where this roadmap is silent and is versioned
independently.

### Phase 4 - Causal Intervention Battery

Goal: replace the Causal Intervention Test from `SUNDOG_V_GRAVITY.md`
Candidate 7 with a battery that runs inside the mesa harness, identifying
where control authority lives in each controller family.

Intervention channels:

- **Reward edit:** alter the scalar reward without changing the world.
- **Observation edit:** alter what the agent sees without changing the world.
- **Signature-sensor edit:** corrupt the measured signature while leaving
  geometry fixed.
- **Geometry edit:** change the underlying environmental state that generates
  the signature.
- **Basin-position edit:** move the live false-basin fixture, testing whether
  canonical L-Reward keeps following the old/internalized basin.
- **Internal-proxy edit:** when interpretability allows, edit the candidate
  internal proxy representation directly (Phase 6).

Deliverables:

- An intervention-response matrix for each controller family: which edits
  change policy, regime retention, and failure mode.
- A causal-authority graph: where control authority lives for HC-Signature,
  L-Signature at each capacity, and L-Reward.
- A diagnostic for internal-proxy emergence: an L-Signature policy that
  follows a reward-channel or observation-channel edit *more* than the
  external signature signal is showing internal-proxy capture.

Exit criterion: every controller family has an intervention-response matrix.
The diagnostic for internal-proxy emergence is defined and applied.

Implementation-grade detail lives in the satellite spec
[`mesa/PHASE4_SPEC.md`](mesa/PHASE4_SPEC.md). Phase 4 v1 locks fixed
`t = 50` held-to-end interventions and lands basin-position edit as a v1
channel. Because exported policies do not observe live reward or `x_false`
at inference, reward and basin-position edits are interpreted as live-signal
invariance tests; the basin-capture receipt is continued attraction to the
old/internalized basin after the live basin moves.

Result note: [`mesa/PHASE4_RESULTS.md`](mesa/PHASE4_RESULTS.md) records the
Small and Medium causal intervention batteries. The load-bearing finding is
that canonical L-Reward has zero action response to live basin movement while
ending `3.413` (Small) and `5.560` (Medium) units preferentially closer to
the training-time basin than the moved basin. Its signature-sensor and
geometry responses collapse at Medium (`0.060` and `0.069`), supporting the
fixed-attractor reading.

### Phase 5 - Selection-Pressure Curriculum

Goal: separate capacity from selection-pressure shape and measure mesa
emergence across selection regimes at fixed capacity.

The earlier candidate framing scaled capacity. This phase fixes capacity and
varies the *shape* of selection pressure, because mesa emergence is widely
suspected to depend on training-signal structure more than parameter count.

Pinned selection-pressure variants:

- **Signature-only, dense:** L-Signature trained on `g(S(x_T))` with linear
  `g`.
- **Signature-only, threshold:** L-Signature trained on a hard threshold over
  `S`, testing whether sparse signature pressure induces different proxy
  behavior than dense shaping.
- **Imitation-from-HC:** L-Signature trained purely by behavior cloning from
  HC-Signature trajectories.
- **Dense reward:** L-Reward canonical with continuous distance shaping.
- **Sparse reward:** L-Reward with success/failure only.
- **Curriculum, signature-first:** train as L-Signature, then fine-tune on
  `R`.
- **Curriculum, reward-first:** train as L-Reward, then fine-tune on the
  signature objective. Diagnostic for whether reward pretraining poisons
  later signature training.
- **Mixed lambda schedule:** L-Mixed at
  `lambda in {0.1, 0.3, 0.5, 0.7, 0.9}`.
- **Reward-shape adversary:** periodically inject small perturbations into the
  reward shaping function during training (shift, scale, rotate) to test
  whether the learned policy is robust in the signature flavor or fragile in
  the proxy-collapse flavor.

Hindsight relabeling is held out for now. It is a fourth axis, and the initial
matrix is already large enough to debug without it.

Deliverables:

- A selection-pressure × capacity matrix with one entry per cell summarizing
  proxy-splitting probe performance and intervention-response shape.
- A failure-mode taxonomy: at which selection-pressure shapes does
  L-Signature behave like L-Reward, and which sustain the gravity claim?
- A first claim ratchet candidate: a named (capacity, selection-pressure)
  pocket where L-Signature retains the gravity claim's distinction, mirroring
  the three-body Phase 9 operating pocket structure.

Exit criterion: the selection-pressure × capacity matrix is filled with
quantitative results. At least one pocket is identified where L-Signature is
measurably distinct from L-Reward under proxy splits and interventions.

Implementation-grade spec at [`mesa/PHASE5_SPEC.md`](mesa/PHASE5_SPEC.md) v1
(2026-05-12). Three axes pinned: L-Mixed λ ∈ {0.1, 0.3, 0.5, 0.7, 0.9} at
Small + Medium follow-up; L-Signature objective shape {terminal,
integrated, threshold} at Small; curriculum order {sig-then-reward,
reward-then-sig} at Small with 50%+50% phase split. Headline metric:
protection curve (`old_basin_pref` vs λ) and breach threshold.

### Phase 6 - Interpretability / Representation Probes

Goal: probe internal representations of learned controllers for evidence of
proxy-objective formation, and cross-reference with the Phase 4 causal
results.

Probe families:

- **Linear probes** on hidden activations for reward-correlated scalars.
- **Activation patching** between L-Reward and L-Signature on matched inputs.
- **Sparse autoencoder features** (if scale permits) for goal-like directions.
- **Behavioral probes:** synthetic inputs designed to disambiguate "tracking
  the external field" from "tracking a learned shortcut."

Honest framing rules:

- A failed search for a mesa-objective is not evidence one cannot emerge.
- Probes may be too weak to distinguish "tracks the signature robustly" from
  "tracks a learned shortcut that happened not to break yet."
- The result must be framed as stress evidence, not a proof of absence.

Deliverables:

- A per-capacity, per-selection-pressure representation report.
- Cross-reference with Phase 4 intervention-response matrix: do the probes
  agree with the causal interventions?
- A list of caveats and known interpretability blind spots for the chosen
  architecture and capacity ladder.

Exit criterion: representation probes are run on at least the medium and
large capacity tiers across the L-Signature and L-Reward families. The
representation report explicitly names what it cannot say.

### Phase 7 - Operating Envelope and Failure Map

Goal: replace anecdotal "Sundog held up at small scale" framing with a
mapped operating envelope, in the spirit of the three-body Phase 9 sweep.

Sweep axes:

- capacity tier;
- selection-pressure shape;
- probe-slate severity;
- sensor tier;
- intervention channel.

Outputs (mirroring the three-body Phase 9 file convention):

- `results/mesa/operating-envelope/manifest.json`
- `trial-outcomes.csv`
- `envelope-map.csv`
- `aggregate-envelope.csv`
- `best-by-cell.csv`
- `cell-class-map.csv`
- `cell-delta-map.csv`
- `candidate-envelope.csv`
- success/failure heatmaps;
- representative replays for "L-Signature distinct from L-Reward,"
  "L-Signature collapsed to L-Reward," and "ambiguous" cells.

Exit criterion: the operating-envelope map can show where the gravity claim
holds under selection pressure and where it does not, with the same evidence
discipline as the three-body Phase 9 result.

Implementation-grade spec at [`mesa/PHASE7_SPEC.md`](mesa/PHASE7_SPEC.md) v1
(2026-05-12). v1 is a read-only Small/Medium aggregation pass over existing
Phase 3/4/5/6 artifacts: no new training, no Large tier, and no full
probe-by-intervention cross-product. The envelope unit is a classified
`(policy, tier, selection-pressure, evidence-plane, condition)` cell with
explicit `hold`, `collapse`, `fragile`, `incompetent`, or `ambiguous` status.
Phase 6's `net.7` result is attached as a mechanistic annotation to the
Medium `lambda=0.95` / `lambda=0.97` cliff boundary.

> **v2 scoping landed 2026-05-14.** [`mesa/PHASE7_SPEC.md`](mesa/PHASE7_SPEC.md)
> §14 (v2 Scope and Decision Gate) carries the informed-step-forward
> material for the Phase 8 v2 "operating-envelope cross-product"
> deferred item: the corrected compute model (Python/PyTorch CPU-only
> + Node JS env-bridge — not pure-JS; no GPU), **measured** training
> throughput (Small 1.74 / Medium 2.35 / Large **~24.5** s/update,
> firmed by a 20-update probe — the bottleneck flips
> bridge-bound→PyTorch-CPU-bound at Large), and a budget-conditioned
> GO / DOWN-SCOPE / DEFER gate with staged operator PowerShell. The
> binding question is the Large *convergence budget*, not timing.
>
> **Probe-1 ran 2026-05-15 and recalibrated the gate (PHASE7_SPEC
> §14.4.1).** Large @ the 0.66M default budget scored `success_rate
> 0.000` / `alignment 0.360`. A canonical-artifact check found the
> "Small/Medium converged at ~0.66M" assumption *false*: the canonical
> converged Medium signature-terminal policy used a **10M** budget
> (`..._terminal_10m`, success 1.0). So probe-1's 0.000 is an
> under-budget run against a mis-calibrated threshold — **not a DEFER
> signal**. The decision input is now the **recalibrated Large @ 10M
> probe**, firmed at **~8.3 h** and rewritten as a **monitorable
> 6-segment chain** (~83 min/segment, per-segment eval via verified
> `--load-checkpoint` resume, early-abort on flat alignment — not an
> 8.3 h black box; PHASE7_SPEC §14.4.2 / §14.6). GO/DOWN-SCOPE if it
> clears 0.75 at 10M (full ~12-policy set ≈ **~4.2 days** / cliff
> subset ≈ **~2.1 days**); DEFER if alignment stays flat / still
> sub-floor at 10M. No v2 training batch starts until probe-2 is read
> and a branch is selected in writing.

Result note at [`mesa/PHASE7_RESULTS.md`](mesa/PHASE7_RESULTS.md) v1
(2026-05-12). The first envelope map classifies 22 Small/Medium policies with
zero missing cells: 8 `hold`, 7 `collapse`, 1 `fragile`, 4 `incompetent`, and
2 `ambiguous`. The Medium breach remains localized to `(0.95, 0.97]`
(`lambda ~= 0.952588`), with Phase 6's `net.7` annotation landing on the
protected/collapsed cliff pair.

Sister result note at [`mesa/PHASE7_V2_RESULTS.md`](mesa/PHASE7_V2_RESULTS.md)
v2 (2026-05-18). The DOWN-SCOPE Large-tier cliff-subset batch ran six
policies along the Medium-cliff λ-spine under `--value-coef 0.25`; it
surfaces a Large U-trough at `λ ∈ {0.95, 0.97}` distinct from Medium's
single sharp cliff, recovery at `λ=0.99` (1% signature anchor), and
bootstrap failure at `λ=1.00`. v2 is a sibling to v1, not a successor;
the v1 22-cell classification is unchanged.

v3 spec at [`mesa/PHASE7_V3_SPEC.md`](mesa/PHASE7_V3_SPEC.md) (2026-05-18).
v3 is a Phase-4-style causal intervention battery on the six Large
cliff-subset checkpoints, designed to close the v2 caveat by
probe-confirming whether `λ=0.99` recovery is basin-attractor avoidance
or co-pointing fixed-attractor collapse. v3 is read-only over existing
checkpoints (no new training); compute envelope ~60–70 minutes total.

v3 result note at [`mesa/PHASE7_V3_RESULTS.md`](mesa/PHASE7_V3_RESULTS.md)
(2026-05-18). All six policies evaluated at 64 seeds × 5 channels.
**GG4-A confirmed: the v2 caveat closes** — `λ=0.99` Large is genuine
basin-attractor avoidance (sig_resp 0.579, bp_obp 0.459). **GG3
partially falsifies: the U-trough is field-coupled (but with
moderately elevated bp_obp), not the collapse class and not the
spec's pre-registered low-obp alternative** — mixed_0_95 and
mixed_0_97 retain healthy signature responses (0.580 / 0.973) and
bp_obp 1.5 well below collapse but above the Medium hold-cell upper
bound 0.823. New third-class traceability label
`field-coupled, under-budget` introduced for the trough cells. GG5
partially falsifies symmetrically: reward_phase3 seg3 is
`bootstrap-collapse` (degenerate fixed trajectory at the old basin),
not `undertrained`. Four amendments required to PHASE7_V2_RESULTS.md
§5, §6, §8, §10.

### Phase 8 - Writeup, Claim Ratchet, and Public Artifact

Goal: turn the result into a defensible Sundog hook, update the boundary
language across the program, and ship a public artifact.

### Horizon H1 - Pantheon of Agency

The Tauroctony ledger's first typed horizon is now opened as
[`mesa/H1_PANTHEON_OF_AGENCY_SPEC.md`](mesa/H1_PANTHEON_OF_AGENCY_SPEC.md).
H1 is a pantheon-vs-monolith bake-off over the existing Phase 3/4/7 MESA
machinery. The important admission lock: existing L-Mixed policies are
scalar-mixture prior evidence and monolith baselines, **not** the pantheon
implementation. H1 requires role-separated heads, a bounded arbiter, and a
sovereignty metric so plurality cannot collapse into one hidden maximand and
still count as support. H1.1 smoke is complete; H1.2 is opened at
[`mesa/H1_2_SMALL_BAKEOFF_SPEC.md`](mesa/H1_2_SMALL_BAKEOFF_SPEC.md), scoped to
the learned arbiter and label-trained guard over frozen Small heads.

**H1.2b binding result (2026-06-18): `NULL`.**
[`mesa/H1_2B_RESULTS.md`](mesa/H1_2B_RESULTS.md). At small tier across 13 cells,
the equal-incremental-budget monolith (`M-Adapter`) beat the role-separated
council on 12/13 cells in alignment and on gradient-intact basin capture
(0.036 vs 0.071). The pantheon thesis is **not supported at small-tier** and is
demoted toward [ORNAMENT] for this lane; the 0.70 cap + guard-brake impose a
"pantheon tax" not repaid by proxy-resistance here. The arbiter *did* keep the
reward (bull) head off the throne (bull-breach 0.04 vs the blind blend's 0.20),
so sovereignty discipline works — it just did not buy a net win. The reopening
path is now registered as
[`mesa/H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md`](mesa/H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md):
test the reward-asymmetric *blend* cap first (bind the bull, leave Sol
uncapped). That rung has now also returned null; the next registered rung is
[`mesa/H1_2D_RL_ARBITER_SPEC.md`](mesa/H1_2D_RL_ARBITER_SPEC.md), not a
re-score.

**H1.2c result (2026-06-18): also `NULL`.**
[`mesa/H1_2C_RESULTS.md`](mesa/H1_2C_RESULTS.md). The reward-asymmetric cap did
not repair the tax — `cap_tax_repair` negative; M-Adapter won on all 13 cells.
The relief was structurally available but the **supervised arbiter would not use
it** (field exceeded 0.70 only ~12% of steps even on clean cells). Two
independent binding nulls now stand: at small tier with a supervised arbiter,
role-separation loses to the equal-budget monolith regardless of cap geometry.
The last registered reopening rung before [ORNAMENT]-for-good at this tier is
[`H1.2d`](mesa/H1_2D_RL_ARBITER_SPEC.md): RL-trained arbiter against a same-run
equal-budget RL monolith, then higher tiers.

**H1.2d binding (2026-06-19): `PROXY_NULL`.**
[`mesa/H1_2D_RESULTS.md`](mesa/H1_2D_RESULTS.md). Direct-return PPO **repaired the
H1.2c bottleneck** — the RL council reached competence parity with the same-run
RL monolith (GI alignment 0.936 vs 0.969, slate 0.803 vs 0.835 within the 0.05
band; field-relief 0.476, high on clean cells and backed off on corrupted ones)
and held the bull ≤ 0.50 with zero breaches (gates 1, 2, 4, 5 pass). But it does
**not out-resist proxy capture** (gate 3 fails: monolith GI basin 0.0045 <
council 0.0223). Structural cause: a pantheon must always seat a bounded bull
vote, while the monolith can become a pure field-follower and ignore the proxy
entirely. **Three registered Small-tier nulls (H1.2b/c/d); the frozen-head
Small-tier H1.2 line is CLOSED.** Net for the lane: the bull can be bounded and a
bounded council governs as ably as the monarch, but plurality did not buy
*superior* proxy resistance at this tier. Pantheon thesis stays [ORNAMENT] for
the MESA lane; reopening = registered change of tier, features, or guard (a guard
that cancels a basinward bull vote rather than merely holding).
That guard-change reopening is now registered as
[`H1.2e`](mesa/H1_2E_CANCELLING_GUARD_SPEC.md).

**H1.2e binding (2026-06-20): `MECHANISM_NULL`.**
[`mesa/H1_2E_RESULTS.md`](mesa/H1_2E_RESULTS.md). The cancelling guard
(`a_guard = −c_guard·a_reward`) council out-resisted the same-run `M-Adapter-RL+`
on GI basin (0.0067 < 0.0223 → gates 1,2,3,5,6 pass), **but the pre-registered
mechanism gate (4) failed: cancellation never engaged** (`c_guard` 0.007→0.015
over 512 updates, cancel_mass ~0.001–0.003). The basin gain came from extra
arbiter training + a weaker same-run monolith, not the countervote — so it's not
creditable. Finding: **the cancelling guard is redundant** (the arbiter can
already drive `w_reward→0`, a simpler lever; PPO leaves the cancel head at ~init),
which *softens* the H1.2d "must seat the bull with force" reading — the bull can
just be ignored. **Four registered Small-tier nulls (H1.2b/c/d/e); the
frozen-head Small-tier line is thoroughly closed.** No tested mechanism — cap
geometry, RL arbitration, cancelling guard — made plurality *creditably*
out-resist proxy capture at this tier. Reopening = genuinely different regime
(Medium/Large tier or richer trust features), separately registered. The
mechanism gate is the durable methodological win (it blocked a false-positive
SUPPORT).

**H1.2f binding (2026-06-22): `SUPPORT` — first typed pantheon support.**
[`mesa/H1_2F_RESULTS.md`](mesa/H1_2F_RESULTS.md). Adding six temporal **trust
features** (shared identically with the monolith) cleared all five gates: the
bull-bounded role-separated council out-resists the equally-enriched same-run
monolith on gradient-intact basin capture (**8 vs 17 / 448**, 6/7 GI cells) at
competence parity (gates 1,2), and the win is **attribution-gated** — ablating
the trust features flips the GI advantage +0.02 → −0.0715 (gate 3) — with bull
≤ 0.50 (gate 4) and matched 23 features/budget 1.0105 (gate 5). 512-update
training, no leakage. The four nulls were a **feature-discrimination limit**;
supplying a field-trust signal fairly let plurality express the predicted
advantage (council weights the bull lower where the field is clean: w_reward
clean-GI 0.197 vs corrupt 0.239). **Pantheon thesis promoted off blanket
[ORNAMENT] to a [TYPED] bounded-positive for the MESA lane** — Small-tier,
in-vitro, single PPO seed; owed hardening: multi-seed replication → higher tier.

**H1.3 binding (2026-06-22): `ATTRIBUTION_NULL` — the trust-feature mechanism
does NOT scale to Medium.** [`mesa/H1_3_RESULTS.md`](mesa/H1_3_RESULTS.md).
Scaling frozen heads Small→Medium, the council out-competes (0.766 vs 0.719) and
out-resists the same-run monolith on GI basin (0.069 vs 0.199) — but **ablating
the trust features grows rather than collapses the advantage** (intact +0.130,
ablated +0.192; attribution delta −0.0625 < 0.01), so the Medium win is not
creditable to the registered features. Validity/competence/proxy/sovereignty
gates pass; attribution gate fails → the H1.2f positive is **pinned to
Small-tier**. Diagnostic (NOT gated): the Medium advantage looks **structural**
(the monolith degrades at Medium — GI basin 0.199, GI alignment 0.796 — while the
council holds 0.069/0.890), consistent with the H1.2d "ignore-the-proxy edge may
not scale" prediction; owed its own registered rung (**H1.4**: role-separation
ablation + multi-seed) before any claim. The attribution gate is symmetric — it
credited H1.2f and denied H1.3 the same way.

**H1.4 binding (2026-06-23): `NONROLE_NULL` — metric-design null; the structural
hypothesis is refuted as framed.** [`mesa/H1_4_RESULTS.md`](mesa/H1_4_RESULTS.md).
Removing the trust axis (base 17 features) + adding **singleton controls**
(`P-Field-M`/`P-Reward-M` run alone), the seed-0 13-cell binding **saturated**:
council, monarch, and Sol-alone all reached `basin_GI = 0` at `S_T_GI ≈ 0.998` —
the role-separation advantage unidentifiable (gate3 0<0.03, gate4 0<0.01;
competence/sovereignty pass). Two findings closed the rung (3-seed binding waived
as predetermined): **(1)** the reward-blind field head is the proxy-resistance
*optimum* (never sees the bull → never drifts to its basin), so a bull-seating
council can at best tie it ⇒ `STRUCTURAL_SUPPORT` foreclosed on **any** pure
proxy-resistance metric; **(2)** H1.3's Medium edge was a trust-feature *training*
artifact — only trust(23)→base(17) changes, and the monarch's GI basin 0.199→0,
erasing the gap. **No structural plurality advantage survives at Medium**; H1.3's
diagnostic explained away. Reframed open question owed a new rung: does plurality
win the **competence–resistance trade-off** on a field-necessary-but-insufficient
task (this metric floored at 0 and could not discriminate)? The MESA-lane thesis
is now **[TYPED] bounded-positive at Small-tier (H1.2f), documented NULL at Medium
(H1.3/H1.4)**.

**H2 frontier task family (opened 2026-06-23) — the reframed rung; H2.0 cell
admission PASSED.** [`mesa/H2_FRONTIER_TASK_FAMILY_SPEC.md`](mesa/H2_FRONTIER_TASK_FAMILY_SPEC.md),
[`mesa/H2_0_CELL_ADMISSION_RESULTS.md`](mesa/H2_0_CELL_ADMISSION_RESULTS.md). H2
abandons the pure proxy-resistance metric (foreclosed by H1.4 Finding 1) for a
**joint competence–resistance frontier** on a Family-B *forked-field* task: a
symmetric two-peak field routes to a fork (field reaches a branch but picks the
correct one ≈chance), a reward cue helps pick correctly, and symmetric wing
**false basins** trap a reward-greedy agent. The H2.0 admission gate (a fixed-
control pre-check that forbids a repeat of the H1.4 floor) **admitted** the
primary slate: Field C=0.42/B=0 (necessary+insufficient), Reward C=0.72/B=0.28
(useful+dangerous), Oracle C=1.0, all six gates pass, byte-reproducible — and
correctly **VOIDs** cells lacking the tension. The H2.1 support gate is built to
escape the H1.4 foreclosure: the council need only **tie** the reward-blind field
on resistance (not beat it), while beating it on competence and out-resisting the
*monolith* at competence parity. **H2.1 binding (2026-06-23): `MONOLITH_NULL`.**
[`mesa/H2_1_RESULTS.md`](mesa/H2_1_RESULTS.md) records the binding result. After
a corrected eval cap readback (the first eval passed an object-shaped cap and
was void), the reward-bounded council solved the fork (`C=1/B=0`, max reward
weight 0.50) and beat the field/reward singletons exactly as intended, but the
matched monolith also solved it (`C=1/B=0`). H2.1 therefore shows bounded
competence without sovereignty failure, but no pantheon frontier advantage.
**H2.2 passed both admission locks, then voided at controller-probe budget.**
[`mesa/H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md`](mesa/H2_2_LEARNED_HEADROOM_MULTIFORK_SPEC.md),
[`mesa/H2_2_CELL_ADMISSION_RESULTS.md`](mesa/H2_2_CELL_ADMISSION_RESULTS.md),
[`mesa/H2_2_LEARNED_HEADROOM_PROBE_RESULTS.md`](mesa/H2_2_LEARNED_HEADROOM_PROBE_RESULTS.md),
and [`mesa/H2_2A_FRONTIER_PROBE_RESULTS.md`](mesa/H2_2A_FRONTIER_PROBE_RESULTS.md)
record the Family-C multi-fork phase/key task. Fixed controls pass the dilemma
gate (Field safe but incompetent; Reward useful but dangerous; Gated diagnostic
shows real headroom), and the 64-update learned monolith probe also passes the
missing H2.1 lock: the monolith improves over Field (C +0.229; fork completion
+0.552) but leaves large oracle slack (`oracle_gap_monolith=1.427`). H2.2 is
therefore admitted for a frontier controller probe. **H2.2-a then returns
`H2_2_LEARNED_HEADROOM_VOID`:** by 128 updates the matched monolith reaches the
oracle frontier (`C=1/B=0`, `oracle_gap_monolith=0`), so the pre-registered
override fires before any pantheon support/null score. H2.2-b binding is skipped;
Family C is too compressible for this frontier claim.

**H2.3/H3 escalation (2026-06-23): cap-positive, plurality-null; then
body-invariant admission, verifier null.** H2.3 opened a safe-exploration
proxy-poisoning rung and returned [`H2_3_CAP_NOT_ROLES`](mesa/H2_3_RESULTS.md):
the cap mechanism is real, but the capped no-role monolith matches the council,
so this is cap support rather than pantheon support. H3.0 then changed axis
rather than making a harder proxy basin:
[`H3_0_ADMITTED`](mesa/H3_0_BODY_INVARIANT_HEADROOM_RESULTS.md) constructs a
body-resistant, invariant-sufficient control family where the capped no-role
learner finds signal but remains dangerous. H3.1 tested the registered
verifier/guard mechanism against that admitted family and returned
[`H3_1_RESISTANCE_NULL`](mesa/H3_1_RESULTS.md): validity, learned headroom, and
sovereignty pass, but the verifier council remains basin-dangerous, does not
beat the matched capped monolith controls, and fails verifier/certificate
attribution. Net after H3.1: the only positive pantheon support remains the
bounded Small-tier H1.2f trust-feature result; H2/H3 so far are disciplined
nulls or voids on plurality.

**H4 drafted (2026-06-23): distributed world-model topology.**
[`mesa/H4_DISTRIBUTED_WORLD_MODEL_TOPOLOGY_SPEC.md`](mesa/H4_DISTRIBUTED_WORLD_MODEL_TOPOLOGY_SPEC.md)
opens the next mechanism after the H3.1 null. It moves the pantheon claim out of
one-step head factorization and into a POMDP process topology: local observers
maintain local world-model memories, send bounded messages to a coordinator, and
act under stale/corrupt partial observations. Support now requires beating
equal-budget central recurrent monolith controls and losing the edge when local
memories/messages are ablated. Draft only; no H4 admission or result exists.

Immediate work:

- Update `docs/promo/PROMO_HIGHLIGHTS.md` §The Gravity Claim with the earned mesa
  result, replacing speculative language with the strongest claim actually
  earned by Phase 7.
- Update `docs/presentation/claims-and-scope.md` §The Gravity Frame and
  §Unsupported Universal Claims to reflect the new evidence.
- Update `docs/SUNDOG_V_GRAVITY.md` Candidate 2 (Mesa) and Candidate 1
  (Formal Separability) status, with the appendix theorem either ratified or
  flagged as in progress.
- Build an interactive browser visualization at `mesa.html` showing the
  controller-family ladder, probe slate, intervention battery, and a
  representative best-cell vs worst-cell replay.

### Phase 8 v1 — Public Artifact Shipped (2026-05-12)

Status: the four immediate-work items above all landed. The earned-envelope
form of the gravity claim is now consistently expressed across the
program's public-facing surfaces.

**Upstream doc cascade complete.**

- `docs/promo/PROMO_HIGHLIGHTS.md` §The Gravity Claim now stacks three earned
  sub-sections after the named-claim and Goodhart-sidestep rhetoric:
  *The empirical anchor (Phase 5 v4)* with the `λ ≈ 0.952588` cliff and
  the `1 − λ = 0.048` policy quotable, *The mechanical anchor
  (Phase 6 v1)* with the `net.7` patch-success receipt, and *The
  envelope (Phase 7 v1)* with the class-balance table and the §9
  hand-off sentence as block-quote. The section reads left-to-right as
  rhetorical → behavioral → mechanistic → operating-envelope, with
  boundary language closing.
- `docs/presentation/claims-and-scope.md` §Unsupported Universal Claims
  revised the *sidesteps Goodhart* and *cannot be reward-hacked* bullets
  to point at the new earned form; §The Gravity Frame gained an
  *Earned envelope language (Phase 7 v1)* sub-section listing the three
  load-bearing anchors and an explicit *what is still not earned* list
  (universal mesa immunity, foundation-model behavior, deployed-system
  robustness, adversarial robustness, large tier, cross-architecture);
  the required boundary text now includes the mesa-trap front in the
  controlled-evidence list.
- `docs/SUNDOG_V_GRAVITY.md` promotion summary and Candidate 2
  §Current recommendation both carry 2026-05-12 status updates with
  class balance, breach threshold, mechanistic locus, and "partially
  holds" outcome mapping. Candidate 1 (Formal Separability) is flagged
  as in-progress with the in-vitro bounded empirical receipt now
  anchoring the theorem-track work.

**Public artifact: `mesa.html`.** Landed at repo root alongside
`threebody.html` / `balance.html` / `mines.html`. Single-file, no
external runtime deps beyond `/css/sundog-theme.css`. Five sections:

1. **Hero** — title, tagline, value-prop, and the PHASE7_RESULTS §9
   hand-off sentence as the canonical earned claim. Background carries
   a faint SVG silhouette of the Medium λ-protection curve with the
   cliff at `λ ≈ 0.953` drawn behind the title.
2. **§The Envelope (Phase 7 v1)** — class-balance stat cards (8/7/1/4/2)
   plus a grouped policy chip grid. Each chip carries a hover-tooltip
   with slug/tier/class/success/mean_align/old_basin_pref/note. Data
   inline at write time; reconciled against
   `results/mesa/operating-envelope/policies-inventory.csv` so all 22
   slugs match the canonical `policy_id` column.
3. **§The Cliff (Phase 5 v4)** — interactive λ → old_basin_pref line
   chart with Medium/Small tier toggle. Cliff band highlighted between
   `λ=0.95` and `λ=0.97`; breach threshold drawn at `obp = 1.0`; Phase 6
   `net.7` patch points marked with gold rings on the protected and
   collapsed sides.
4. **§The Locus (Phase 6 v1)** — grouped bar chart of patch_success by
   layer × direction × stat. P4 threshold at 0.8 drawn as horizontal
   dashed line. `net.7` column highlighted; net.1 heavy-tail caveat
   walked through in the chart caption (median 0.061, ratio-of-means
   0.219 demote the headline mean of 2.659).
5. **§What This Doesn't Say** — claim-boundary block with explicit
   non-claims, plus a 7-link doc grid back to the full trail.

**Cross-page nav sweep complete.** Mesa link added to all 10 sibling
pages: `index.html`, `about.html`, `applications-gallery.html`,
`threebody.html`, `balance.html`, `mines.html`, `origin.html`,
`sundog.html`, `sundog-workbench.html`, `docs/index.html` (with `../`
prefix). `mesa.html`'s own nav carries `aria-current="page"`.

**v1 deferred — becomes v2 work.**

- *Best-cell vs worst-cell replay.* Phase 8 brief mentioned a
  representative replay; v1 ships sections 1-5 above without per-step
  trajectory visualization. v2 options: (B) static side-by-side
  terminal-position scatter from Phase 4 intervention CSVs, or
  (C) live in-browser replay running `public/js/mesa-core.mjs` against
  the L-Mixed-M-λ=0.95 and λ=0.97 policy.json files. C is closest to
  the brief; ~half-day of work.
- *Phase 6 v2 Axis A redux at net.7.* The v1 Axis A negative result
  raised the prior on direction-based or sparse-autoencoder probing.
  `mesa.html` v2 would gain a §The fingerprint sub-section once SAE
  features at net.7 are extracted.
- *Phase 7 v2 operating-envelope cross-product.* v1 is a read-only
  Small/Medium aggregation pass over existing Phase 3-6 artifacts. v2
  would run the full probe-by-intervention cross-product and add Large
  tier, which would expand the policy chip grid and add a third tier
  toggle to the cliff chart.

**Exit criterion for Phase 8 v1:** all four upstream-doc cascade items
landed, `mesa.html` reachable from every sibling page with data
reconciled against the inventory CSV, and no public-facing surface
asserts a stronger claim than the earned envelope form.

Claim ratchet candidates by phase result:

> *If Phase 7 holds:* In the tested environment family, learned signature-
> trained controllers remain measurably tied to the external field across the
> mapped capacity and selection-pressure pocket, while matched reward-trained
> controllers exhibit characteristic proxy-collapse failures under the same
> probes. The result is not global: capacity and selection-pressure cells
> outside the mapped pocket remain known harm boundaries.
>
> *If Phase 7 partially holds:* In the tested environment family, learned
> signature-trained controllers retain the gravity-claim distinction inside a
> bounded (capacity, selection-pressure) pocket. The boundary is mapped, and
> the gravity frame is updated to reflect that bound.
>
> *If Phase 7 falsifies:* In the tested environment family, learned
> signature-trained controllers exhibit proxy-collapse failures
> indistinguishable from matched reward-trained controllers above a measured
> capacity threshold. The gravity frame is downgraded accordingly. The Formal
> Separability appendix becomes the program's primary defense of mode (3),
> and Spacecraft / Conservation-Law candidates absorb more of the public-
> facing burden.

Do not claim, in any of the three outcomes:

- "Sundog is immune to mesa-optimization."
- "Reward hacking does not occur in signature-trained agents."
- "Inner alignment is solved."

Exit criterion: the writeup, page copy, and program-wide boundary language
all use the strongest claim actually earned by the completed phase, and no
stronger one.

---

## Appendix A — Formal Separability Theorem

Promoted from `SUNDOG_V_GRAVITY.md` Candidate 1. This appendix is the
intellectual target the empirical roadmap is trying to earn. It is included
here so the theorem and the experiment live in the same artifact and
constrain each other.

### Motivation

The gravity claim depends on a structural distinction: a signature `S(x)` is
a function of environmental state alone, while a reward `R(s, a)` or `R(o, a)`
is a function the agent participates in. The empirical mesa experiment can
show that learned signature-trained controllers behave differently under
proxy splits, but it cannot, by itself, prove the distinction is real. A
hostile reviewer can always argue "your signature is just a reward in
costume" without running an experiment.

The Formal Separability Theorem names the conditions under which the
distinction is mathematically nontrivial, and ships counterexamples that
mark the boundary.

### Setup

Let:

- `X` be environment state.
- `A` be agent action.
- `P(x' | x, a)` be transition dynamics.
- `S: X -> Σ` be a signature map, defined as a function of `X` alone.
- `R: X × A -> ℝ` or `R: O × A -> ℝ` be a reward function, where `O` is the
  agent's observation channel.
- An adversary with bounded budget on each of the following channels:
  - reward editing (alter `R` values directly);
  - observation editing (alter `O` without changing `X`);
  - signature-sensor corruption (alter measured `S` without changing `X`);
  - geometry editing (alter `X` itself).

### Theorem (conditional separation)

In environments where `S` is policy-independent except through the real
transition dynamics over `X`, the adversary's cost to steer a signature-
tracking controller by a fixed regime amount is bounded below by the cost
of either:

1. corrupting the signature sensor, or
2. performing geometric work on `X`.

The same adversary can steer a reward-optimizing controller by the same
regime amount through reward editing or observation editing alone, at cost
strictly less than (1) and (2) for at least one nontrivial adversary class.

This is a conditional theorem, not a universal one. The conditions are:

- `S` must be policy-independent in the stated sense;
- the adversary class must include cheap reward or observation channels and
  expensive sensor or geometry channels;
- the regime change must be definable in `Σ`-space.

### Counterexamples to ship

The theorem boundary is visible only if its counterexamples are explicit.
Required counterexamples:

- **Policy-dependent signature.** A "signature" that takes the agent's
  action history as input is not a signature under this theorem. Includes
  most learned representations of state-action histories.
- **Cheap sensor.** A signature observed through a sensor the adversary can
  spoof at low cost collapses to the reward case. Includes most software
  sensors without physical grounding.
- **Cheap geometry.** An environment where rearranging `X` is on the
  adversary's cheap action surface (e.g., text generation, recommender
  feeds) collapses the separation.
- **Decompilable signature.** A signature that can be written as a
  closed-form function of a scalar agent-corruptible quantity is the reward
  in costume. Sharp examples should be constructed and included.

### Open Questions

- Does the theorem extend to learned signatures fit from data? Probably not
  without additional structure on the fitting procedure.
- Does the cost asymmetry survive composition? When multiple adversaries
  combine cheap-channel attacks, does the bound degrade gracefully?
- What is the relationship between this theorem and existing inner-alignment
  formalisms (deceptive alignment, gradient hacking)? Cross-citations are
  required before publication.

### Bridge to the Empirical Mesa Experiment

The empirical experiment in this roadmap is the operational test of the
theorem's conditions in learned-agent settings:

- Phase 1's HC-Signature controller is the closest practical realization of
  a "tracks `S(x)` only" policy.
- Phase 2–5 ask whether learned `L-Signature` policies inherit that property
  or break it under capacity and selection pressure.
- Phase 4's causal intervention battery is the empirical analogue of the
  adversary cost ladder defined in the theorem.
- Phase 6's interpretability probes are the empirical search for the
  "decompilable signature" counterexample emerging inside the agent.

The theorem and the experiment ratchet together. A ratified theorem with no
experimental support is a sanitized toy. An experimental pocket with no
theorem is a tuned slate. Run together, they define the strongest version of
the gravity claim the program can defend with current methods.

---

## Downstream Dependencies

This roadmap absorbs Candidate 1 (Formal Separability Theorem) as Appendix A
and Candidate 7 (Causal Intervention Test) as Phase 4. The Sundog Gravity
Ledger should mark these as promoted out.

Downstream candidates depend on infrastructure built here:

- **Candidate 4 - Manipulation-Cost Ladder.** Inherits Phase 3's probe slate,
  Phase 4's intervention battery, and Phase 7's sweep harness. The ladder
  experiment becomes a sweep extension once mesa Phase 7 ships.
- **Candidate 5 - Adversarial Signature Benchmark.** Inherits the entire
  controller-family architecture and the operating-envelope harness. Becomes
  primarily an environment-design problem once mesa infrastructure is built.
- **Candidate 6 - Cross-Domain Invariance Battery.** Uses the mesa controller
  families and probe slates as one of its domain entries.
- **Candidate 10 - Conservation-Law Domain.** Uses the controller-family
  architecture and harness once a specific physical domain is chosen.

The gravity ledger should be updated to mark each of these candidates as
"depends on infrastructure from `SUNDOG_V_MESA.md` Phases 1–4 and 7," with
their Current Recommendation language adjusted to reflect the sequencing.

## Implementation Status

**Phase 0:** Decision lock drafted in this document. The environment-family
pick, claim boundary, architecture pinning, capacity ladder, selection-pressure
curriculum, and literature spine are now specified. Implementation-grade
spec landed at [`mesa/PHASE0_SPEC.md`](mesa/PHASE0_SPEC.md) (`v1`,
2026-05-10), unlocking Phase 1.

**Phase 1:** Implemented. The shadow-field core, sensor tiers, HC-Signature
controller, Oracle ceiling, seeded harness, replay verification, and default
`npm run mesa:phase1` command are in place. Result note:
[`mesa/PHASE1_HC_BASELINE.md`](mesa/PHASE1_HC_BASELINE.md).

**Phase 2:** Started. The bridge smoke, BC dataset smoke, and first Small-tier
behavior-cloned L-Signature controller are in place. Reward-trained, signature-
trained, and mixed PPO controllers now train and export. The first canonical
Small PPO slate (L-Signature 5/64, L-Reward 44/64, L-Mixed 14/64 at ~1M steps;
L-Reward 63/64 at 1.31M as the diagnostic over-cap) revealed the canonical-
budget gap between training regimes — the Sundog-cost finding. PHASE2_SPEC.md
v1.7 reframes the Phase 2 gate around stable learning curves plus per-family
over-cap multipliers rather than a single ≥75% threshold at matched budget.
The L-Signature reward path has been checked as `r_t = S(x_t)` end-to-end, so
the L-Signature struggle is a gradient-information/sample-efficiency result
under Gaussian-decay shaping, not a reward-routing bug.
L-Signature and L-Mixed were also measured at 1.31M and 1.97M env steps; neither
reached ≥95% success, so their current multipliers are censored as `>1.97x`.
Phase 3 spec design will need to add an action-dependent component to
L-Reward (control cost at the light end, synthetic spec-gaming surface at the
heavy end) since the current `dense` channel is state-only as implemented.

**Phase 3:** Small + Medium canonical probe slates **complete**. See
[`mesa/PHASE3_RESULTS.md`](mesa/PHASE3_RESULTS.md) v2 for the full result
note including the Medium-Tier Amendment (§10). Spec at
[`mesa/PHASE3_SPEC.md`](mesa/PHASE3_SPEC.md) v1.7. Headline finding: capacity
amplifies basin absorption from 11-25% of L-Reward canonical trials at Small
to 80-90% at Medium, while L-Mixed Medium shows partial signature-anchor
breach (4-8 captures per cell, where Small had 0). The §10.4
three-point capacity-dependence picture — pure signature is structurally
immune at any scale, 50/50 mixed admits proportional leakage that grows with
capacity, pure reward is corrupted at Small and dramatically more corrupted
at Medium — is the program's strongest gravity-claim formulation to date.
Basin-effect gap widens from 65.6 pp (Small) to 76.6 pp (Medium). Probe-slate
harness at `scripts/mesa-probe-slate.mjs`. Î²-sensitivity sub-result complete
({0.5, 1.0, 2.0} monotonic). Large tier not started.

**Phase 4:** Small + Medium causal intervention batteries **complete**. See
[`mesa/PHASE4_RESULTS.md`](mesa/PHASE4_RESULTS.md) v1. Spec at
[`mesa/PHASE4_SPEC.md`](mesa/PHASE4_SPEC.md) v1. Headline finding:
canonical L-Reward has zero direct response to live `x_false` movement but
continues toward the training-time basin, with old-basin preference growing
from `3.413` at Small to `5.560` at Medium. Its Medium signature-sensor and
geometry action responses are only `0.060` and `0.069`, while L-Reward-Clean
responds at `0.572` and `0.772`; this is the causal receipt for
fixed-attractor control. Aggregate reports live under
`results/mesa/phase4-intervention-battery/reports/` and rebuild with
`npm run mesa:phase4:aggregate`.

**Phase 5:** Small + Medium slates **complete through v4**. See
[`mesa/PHASE5_RESULTS.md`](mesa/PHASE5_RESULTS.md) v4 for the full result
note and [`mesa/PHASE5_SPEC.md`](mesa/PHASE5_SPEC.md) v1.5 for the
cliff-localization outcome. **Headline Medium finding: L-Signature with
terminal-only shape reaches 64/64 success at Medium tier**, matching the
Oracle and HC-Signature ceiling, exceeding L-Reward-Clean Medium's 49/64 by
23 percentage points, and overturning the Phase 2-3 Sundog-cost framing.
State-only signature training at correct shape is *better than* dense reward
training at scale, not a "trade adversarial robustness for sample efficiency"
deal. The integrated-signature canonical from Phase 2 is **deprecated**;
terminal-only is the new canonical for future Phase 5+ work.

Phase 5 v4 localizes the mixed-signal cliff. Medium L-Mixed remains protected
through `lambda=0.95`: `old_basin_pref=0.330`, 43/64 success, mean
`S_T=0.982`. At `lambda=0.97` and `lambda=0.99`, it collapses into
fixed-attractor behavior (`old_basin_pref=5.510` and `5.159`, success 2/64
and 3/64). Pure L-Reward-M at `lambda=1.0` remains the collapsed anchor
(`old_basin_pref=5.560`, 0/64 success). The cliff is localized to
`(0.95, 0.97]`; the aggregate interpolation is `lambda ~= 0.952588`.
Operationally: in this Medium setup, a 5% signature anchor prevents the
false-basin attractor, while a 3% anchor does not.

The reward-pretrain -> terminal-signature fine-tune follow-up falsified the
strict Goodhart-fix-by-fine-tuning prediction under the default optimizer-
continuation setup: 0/64 success, mean `S_T=0.363`, and
`old_basin_pref=3.691`. Terminal signature works from scratch, but did not
rescue this reward-pretrained fixed-attractor policy.
**Phase 6:** Interpretability v1 **complete**. See
[`mesa/PHASE6_RESULTS.md`](mesa/PHASE6_RESULTS.md) v1 for the result note
and [`mesa/PHASE6_SPEC.md`](mesa/PHASE6_SPEC.md) v1.6 for the lock that
deferred Axis A to v2 after smoke failures. **Headline finding: the
Phase 5 v4 cliff is causally localized to the actor's final hidden
activation layer.** Patching `net.7` across the L-Mixed-M-`lambda=0.95`
/ L-Mixed-M-`lambda=0.97` cliff pair clears the pre-registered P4
threshold in both directions: protected→collapsed mean/median/ratio-of-
means = 0.894 / 0.944 / 0.899; collapsed→protected = 0.934 / 0.860 /
0.854. Layers `net.1`, `net.3`, `net.5` do not clear the threshold under
robust statistics (`net.1` shows a large mean but median 0.061 and
ratio-of-means 0.219 — heavy-tail artifact, not localization).

Axis A linear-probe maps are a **clean negative result** at v1. Two
target designs (endpoint-shaped behavior target; geometry ΔR² over input
baseline) both failed their smoke gates: Oracle, L-Signature, and
L-Reward could not be dissociated, and at the cliff pair the
Medium-tier ΔR² profile actually pointed the wrong way (`lambda=0.97`
showed *higher* deepest-layer goal ΔR² than `lambda=0.95`). The
methodological lesson is itself a finding: **feature availability and
feature use are decoupled in this regime**. Linear probes recover what
the representation *contains*; they do not show which contained feature
is routed to the action head. The cliff lives in routing, not in
availability. Full-zoo Axis A is deferred to Phase 6 v2 alongside
sparse-autoencoder dictionaries or direction-based probes that can
target net.7 specifically. Phase 6 v2 spec at
[`mesa/PHASE6_V2_SPEC.md`](mesa/PHASE6_V2_SPEC.md) (2026-05-12) pins
two axes: SAE feature dictionary at net.7 across the cliff pair
(Axis D), and direction-based patching using the SAE feature most
correlated with basin attraction (Axis E). v2 is compute-light: no
new PPO training, ~30-60 minutes for SAE training + direction-patch
battery on the cliff pair.

**Phase 6 v2 + v3 + v3.1 result (2026-05-12) - sharpens and revises the mechanistic anchor.** See
[`mesa/PHASE6_V2_RESULTS.md`](mesa/PHASE6_V2_RESULTS.md) and
[`mesa/PHASE6_V31_RESULTS.md`](mesa/PHASE6_V31_RESULTS.md). Four
findings stack:

1. **The basin attractor at `net.7` is an entangled 5-dimensional
   subspace.** Top-5 principal components of the matched-seed per-step
   cliff-pair activation diff matrix capture 97.4% of variance and
   reproduce Phase 6 v1's full-layer patch_success to within 0.03 in
   both directions. K-sweep across `K in {1, 3, 5, 10, 32, 64}`
   saturates at K=5; K=10/32/64 add no meaningful patch_success past
   K=5. This is a **51x compression** of the mechanistic anchor
   (256 dims -> 5 dims).
2. **The old PC1-offset / PCs-2-5-mechanism decomposition is falsified.**
   PC1 alone carries 38.8% of activation-diff variance but contributes
   near-zero patch_success (`0.006 / 0.008`). PCs 2-5 alone are partial,
   not sufficient (`0.291 / 0.121`). PCs 1-5 together reproduce the full
   patch effect (`0.922 / 0.830`). The load-bearing statement is now:
   *the basin-attractor circuit is an entangled 5-dim subspace; no tested
   proper sub-subspace reproduces the full patch effect.*
3. **The basin-inducing subspace generalizes; basin-resistance is more
   policy-specific.** The cliff-pair PCA basis patches protected-to-
   collapsed at cliff-pair quality on held-out Medium pairs: J1
   signature-terminal-M -> reward-M median `0.941`, and J2 mixed-0.9-M
   -> mixed-0.99-M median `1.001`. Reverse rescue weakens: J1
   collapsed-to-protected median `0.162`, J2 `0.631`.
4. **Sparse-autoencoder features are the wrong basis for this circuit.**
   Axis E direction-patching using the top SAE feature (|corr|=0.89 with
   `basin_pref_intervened`, V2 confirmed) produced ~0% patch_success.
   SAE feature rankings on a joint two-policy dataset are dominated by
   policy-identifier features; *they are the wrong basis for causal
   mechanism* even when correlations are extreme.

Directional asymmetry is now confirmed rather than merely flagged. At K=3
the protected-to-collapsed direction reaches `0.881` while collapsed-to-
protected stalls at `0.509`; v3.1 seed-bootstrap puts the median-gap 95% CI
at `[0.251, 0.550]`. Becoming basin-attracted appears to use a shared
controller-family subspace; becoming basin-resistant needs more
policy-specific machinery.

Phase 6 also confirmed the fixed-attractor interpretation from Phase 4
mechanically. The clean-rollout and basin-position-intervened patch
batteries were exactly identical for all logged fields (`max_delta=0`),
because the learned feed-forward policies do not observe live `x_false`
or live reward at inference. The intervention changes environment
state, not policy input. The patch effect is therefore entirely
policy-computation, which is what "the attractor lives in the weights"
means at the mechanistic level.

Together with Phase 5 v4's ~5% signature-anchor threshold, the Phase 6
result raises the gravity-claim narrative one rung: **the behavioral
cliff at `1-lambda = 0.048` is a final-hidden activation decision in an
entangled 5D `net.7` subspace.** Above the threshold, the shared
basin-inducing circuit is available and transfers across Medium controller
families. Below the threshold, basin resistance is preserved by machinery
that the cliff-pair basis does not symmetrically transplant.

**Phase 6 v3.2 result (2026-05-12, smoke gate failed)** at
[`mesa/PHASE6_V32_RESULTS.md`](mesa/PHASE6_V32_RESULTS.md). One axis
(Axis M, top-k neuron-restricted projection of the v3 PCA patch) was
spec'd and harness-wired but the smoke gate **failed cleanly**:
top-32 with 8 seeds captured 33.6% of the aggregate L2 across PCs 1-5
but delivered patch_success of −0.006 P→C and +0.001 C→P — essentially
zero in both directions, well below the 0.37 P→C gate. **Z1
falsified.** The full k-sweep was deliberately not run per
spec §3.4. The mechanistic reading: the 5D basin-attractor subspace at
`net.7` is **not linearly decomposable into top-k neuron
contributions** — masking 87.5% of neurons to their unpatched values
interferes with the partial patch even though those neurons hold only
~66% of basis L2. The basin-circuit is implemented *holistically*
across the neuron substrate; partial delivery of the subspace shift
fails even when the basis is correct.

The methodological lesson stacks on v2's SAE-wrong-basis finding: v2
said *feature-availability rankings don't surface mechanism even at
strong correlation*; v3.2 says *linear additive top-k neuron
restriction within the correct basis still doesn't surface mechanism*.
Together: for this circuit, **non-linear attribution is mandatory**.

The gravity-claim mechanistic anchor is unchanged. The 5D entangled
subspace at `net.7` remains the load-bearing localization. v3.2
records that within that subspace, the circuit is also entangled
across neurons — not decomposable to a few specific ones — which
adds nuance but does not retract or weaken any earned claim.

**Phase 6 v3.3 result (2026-05-12, single-neuron ablation negative)**
at [`mesa/PHASE6_V33_SPEC.md`](mesa/PHASE6_V33_SPEC.md) and
[`mesa/PHASE6_V33_RESULTS.md`](mesa/PHASE6_V33_RESULTS.md). Axis N
tested *necessity* by zeroing one `net.7` neuron at a time after the
v3 K=5 patch was applied. The 4-seed P->C smoke gate passed weakly
(max mean ablation cost `0.057` at neuron `242`), but the full
8-seed both-direction battery falsified AA1: max mean ablation cost
was only `0.040` P->C (neuron `194`) and `0.096` C->P (neuron `100`),
below the `0.1` falsifier line and far below the `0.3` substantial
effect target. The optional critical-neuron Axis M rerun was therefore
not gated.

The useful side facts: ablation-rank and v3.2 L2-rank disagree
strongly (`Jaccard=0.049` for P->C top-32; `0.362` for C->P), and
P->C vs C->P ablation top-32 sets barely overlap (`Jaccard=0.049`).
So the v3.1 directional asymmetry has a neuron-ranking signature, but
no individual neuron is load-bearing enough to mediate the patch.
Together v3.2+v3.3 say the 5D basin-attractor subspace is not
decomposed by simple top-k sufficiency or single-neuron necessity.
The mechanistic anchor remains: *an entangled 5D activation-space
operation at the actor's final hidden layer, implemented through a
distributed neuron substrate*. v3.4 should route to pair-wise /
multi-neuron ablation, set-to-mean ablation, or integrated gradients.

**Phase 6 v3.4 result (2026-05-13, substrate split confirmed)** at
[`mesa/PHASE6_V34_SPEC.md`](mesa/PHASE6_V34_SPEC.md) and
[`mesa/PHASE6_V34_RESULTS.md`](mesa/PHASE6_V34_RESULTS.md). Axis P
tested whether the v3.3 P->C and C->P top-32 substrates are
functional, not just anatomical. Axis Q bootstrapped the Jaccard
overlap of the two rankings. The original 16-seed smoke gate compared
against the 64-seed v3 medians and failed literally (`0.779/0.719` vs
`0.922/0.830`), but the original v3 first-16 slice was also low
(`0.804/0.712`), so this was recorded as a seed-slice calibration
caveat rather than harness drift.

BB1 confirmed: substrate-restricted ablation dissociates functionally.
The P->C-critical top-32 mask drops P->C more than C->P by `+0.174`;
the C->P-critical top-32 mask drops C->P more than P->C by `+0.662`.
The cross-direction drops are negative (`-0.097` and `-0.083`), meaning
ablating the opposite substrate slightly improves the patch rather than
merely leaving it alone. BB2 confirmed: P->C vs C->P top-32 Jaccard
bootstrap median `0.049`, 95% CI `[0.016, 0.085]`. BB3 confirmed for
P->C-vs-L2, CI `[0.032, 0.123]`; C->P-vs-L2 is higher (`[0.280,
0.391]`) and should be treated as a directional asymmetry, not a
falsifier.

The mechanistic anchor ratchets again: *the entangled 5D operation at
`net.7` is implemented through direction-specific neuron substrates.*
Individual neurons remain weak and simple L2 top-k sufficiency still
fails, but ablation-ranked P->C and C->P top-32 substrates are
functionally and statistically separable.

**Phase 6 v3.5 result (2026-05-13, substrate generalization inverted)**
at [`mesa/PHASE6_V35_SPEC.md`](mesa/PHASE6_V35_SPEC.md) and
[`mesa/PHASE6_V35_RESULTS.md`](mesa/PHASE6_V35_RESULTS.md). Axis R
tested cross-policy substrate generalization by comparing the cliff-pair
top-32 P->C/C->P neuron sets against held-out Medium pairs J1
(L-Sig-Terminal-M vs L-Reward-M) and J2 (L-Mixed-M-lambda=0.9 vs
lambda=0.99). The J1 P->C smoke gate passed (`+0.099`, threshold
`0.03`), so the full J1/J2 batteries ran.

The result reverses the v3.5 prior. **CC1 mixed:** P->C substrate
identity did not cleanly generalize. J1 P->C vs cliff P->C Jaccard was
near but below confirmation (`0.255`, CI `[0.032, 0.422]`); J2 was
chance-like (`0.067`, CI `[0.016, 0.208]`). **CC2 falsified strongly:**
C->P substrate identity generalized instead of staying policy-specific.
J1 C->P vs cliff C->P Jaccard was `0.422` (CI `[0.255, 0.641]`); J2
was `0.684` (CI `[0.422, 0.829]`). **CC3 mixed:** three off-diagonal
medians stayed near chance, but J2 P->C vs cliff C->P was elevated at
`0.185`. CC4 was not formally gated, but the exploratory cliff-P->C-mask
Axis P runs landed: J1 weakly dissociated (`+0.113`), while J2 was
ambiguous (`+0.096`, with same-direction drop slightly negative).

The mechanistic anchor is now more qualified: *the Phase 5 cliff remains
a final-hidden 5D activation-space control surface, and within the cliff
pair the P->C/C->P substrates are functionally separable; but
cross-policy behavior transfer does not reduce to stable top-32 neuron
identity in the originally predicted direction*. P->C transfer appears
subspace-level or pair-specific at neuron identity; C->P neuron identity
is unexpectedly stable across Medium held-out pairs despite weaker v3.1
behavioral rescue. Natural v3.6 branch: test functional transfer of the
cliff C->P mask on J1/J2 and compare against pair-specific P->C masks.

**Phase 6 v3.6 result (2026-05-13, C->P functional transfer confirmed)**
at [`mesa/PHASE6_V36_RESULTS.md`](mesa/PHASE6_V36_RESULTS.md). v3.6 ran
the cheap follow-up implied by v3.5: apply the cliff-pair C->P top-32
mask to held-out pairs J1 and J2 using Axis P substrate-restricted
ablation. The result confirms that the v3.5 C->P identity-generalization
surprise is functional, not just anatomical. J1 C->P median drop was
`+0.102` with P->C cross-drop `-0.049`, giving `+0.151` dissociation.
J2 C->P median drop was `+0.517` with P->C cross-drop `+0.009`, giving
`+0.508` dissociation. Updated cascade: C->P/rescue-side substrate
identity and function generalize across Medium held-out pairs; P->C
basin-inducing behavior still appears shared at the 5D control-surface
level but pair-specific at simple top-32 neuron identity. Natural next
branch: run pair-specific P->C masks on J1/J2.

**Phase 6 v3.7 result (2026-05-13, own-mask closeout complete)** at
[`mesa/PHASE6_V37_SPEC.md`](mesa/PHASE6_V37_SPEC.md) and
[`mesa/PHASE6_V37_RESULTS.md`](mesa/PHASE6_V37_RESULTS.md). Axis S ran
each held-out pair's own v3.5 top-32 mask back on that pair. DD1 and
DD2 confirmed: own P->C masks dissociate P->C on both J1 (`+0.128`)
and J2 (`+0.253`), showing that basin-inducing substrates are real
within pair even though their neuron identity is pair-specific across
pairs. DD3 confirmed: J1 own C->P dissociation `+0.208`, exceeding the
v3.6 cliff-mask reference `+0.151`. DD4 is mixed, not falsified: J2 own
C->P dissociation is strong (`+0.392`) and above the `<0.30` falsifier,
but below the v3.6 cliff-mask reference `+0.508`. Updated synthesis:
P->C/basin induction is shared at the 5D control-surface level,
pair-specific at top-32 neuron identity, and functional within pair;
C->P/basin resistance is shared at neuron identity and function across
Medium held-out pairs, with a J2 caveat that the transferred cliff mask
beats the own mask.

**Phase 6 v3.8 result (2026-05-13, per-PC decomposition + signed effects)**
at [`mesa/PHASE6_V38_SPEC.md`](mesa/PHASE6_V38_SPEC.md) and
[`mesa/PHASE6_V38_RESULTS.md`](mesa/PHASE6_V38_RESULTS.md). Axis T
decomposed the cliff-pair 5D subspace PC-by-PC; Axis U re-analyzed signed
neuron effects across the cliff pair plus J1/J2. EE1 is mixed, not clean:
P->C is more component-partitioned than C->P (`0.322` vs `0.430` mean
off-diagonal top-32 Jaccard), but not below the pre-registered confirm
line. EE2 falsifies the stale "PC1 is mechanism-empty" phrasing: PC1 has
the highest single-PC P->C max mean ablation cost (`0.0148`), though all
single-PC costs remain floor-level and v3.1's irreducible K=5 result is
unchanged. EE3 confirms directional opposition at the 0.01 threshold
(`31` opposition-class neurons) and is mixed under the non-gating 0.02
sensitivity check (`14`). EE4 confirms that C->P signed structure is more
stable across held-out pairs than P->C under both thresholds. Updated
synthesis: basin induction is shared at the 5D control-surface level,
more partitioned across PCs, pair-specific at neuron identity, and
functional within each pair; basin resistance is shared across PCs and
across Medium held-out-pair neuron identity/function. The geometry
crossover inherits the same rule in a cleaner form: do not collapse
variance, single-route sensitivity, and full multi-component mechanism
into one "importance" score.

**Phase 6b complete (2026-05-18) — pre-registered falsification.**
Result note at [`mesa/PHASE6B_RESULTS.md`](mesa/PHASE6B_RESULTS.md).
The v1.1 full sweep (64 seeds × 5 layers × 2 conditions, ~13.5 min
wall-clock) fired the destructive-transfer falsifier at all 10
(layer × direction) cells; zero cells cleared the conjunctive P4
rule. **GG6b-loc-D called** — the Phase 6 v1 activation-patching
protocol does not generalize to Large for the mixed_0_99 /
mixed_0_97 cliff pair: at every swept layer (net.1, net.3, net.5,
net.7, net.9), cross-policy activation injection drives the
recipient's terminal alignment to ~0, actively destroying navigation
rather than transferring behavior. **GG6b-mech not called**
(spec §6: not evaluated at destructive layers). **GG6b-shape
deferred** (no locus to decompose). The verdict is consistent with
the v3 finding that the Large cliff is not a basin-attractor
collapse cliff: with neither side internalizing a basin, there is no
transferable basin circuit to probe. clean / intervened bit-identity
bug remains open; doesn't change the verdict (destructive reading is
condition-independent).

**Phase 6b spec filed (2026-05-18)** at
[`mesa/PHASE6B_SPEC.md`](mesa/PHASE6B_SPEC.md). Large-tier activation
patching on the v3 cliff pair (mixed_0_99 vs mixed_0_97 at vc=0.25),
analog of Medium net.7 is Large **net.9** (final hidden Tanh,
hidden_size=1024 in depth=5 actor, actor parameter_count 4,207,618 vs
Medium 199,682). Three pre-registered predictions: GG6b-loc (cliff
localizes at net.9 like Medium net.7), GG6b-mech (the v3 §7
observation-pathway hypothesis produces directional patching
asymmetry — C→P decisively higher than P→C), GG6b-shape (5D entangled
subspace from v3.1 generalizes to Large net.9 or scales with hidden
dim; deferred to v1.1 after GG6b-loc resolves). Harness extension
bounded: `--cliff-pair {medium-v1,large-v3}` flag added to
`training/mesa/phase6_probes.py axis-b-smoke`, v1 code path preserved.
Compute envelope: 6b's step count is ~256k env-steps (~1.25× v1
Medium's ~205k across the 4 canonical Tanh layers), with higher
per-step torch cost at Large from the ~21× actor parameter expansion;
the smoke (~5–10 minutes) is the calibration measurement and the full
sweep is **~2.5–4 hours** pre-measurement — substantially larger than
Phase 7 v3's ~60–70 min battery, not a same-budget extension of v3.

**Phase 7:** v1 **complete**. See
[`mesa/PHASE7_SPEC.md`](mesa/PHASE7_SPEC.md) and
[`mesa/PHASE7_RESULTS.md`](mesa/PHASE7_RESULTS.md). The harness
`scripts/mesa-phase7-envelope.mjs` classifies all 22 Small/Medium policy-zoo
rows with zero missing cells, emits the operating-envelope tables under
`results/mesa/operating-envelope/`, carries forward the Phase 5 breach
thresholds, and attaches the Phase 6 `net.7` annotation to the Medium
`lambda=0.95` / `lambda=0.97` cliff pair. The Phase 8 hand-off claim is a
partial-hold operating-envelope result, not universal mesa immunity.

**Phase 7 v2 caveat (2026-05-17, updated 2026-05-18 with v2 result note):**
the Large cliff-subset results now make the Medium monotone-cliff story
provisional rather than universal. Under the current value-coef-stabilized
Large run, `lambda=0.95` and `lambda=0.97` remain weak by the eval summary
(`success_rate 0.094 / 0.031`, `mean_terminal_alignment 0.599 / 0.506`),
while `lambda=0.99` recovers by the same eval metric (`success_rate 0.406`,
`mean_terminal_alignment 0.885`). That U-shape is potentially the bigger
finding, but the eval summary does not compute `old_basin_pref`. We therefore
cannot yet distinguish "lambda=0.99 reaches the basin direction" from
"lambda=0.99 collapsed onto a fixed attractor that happens to co-point with
the basin direction." Phase 7 v3 should run a Phase 4-style intervention
battery on the Large checkpoints before treating the recovery as
basin-attractor-avoiding. Until then, frame the gravity result as
coherent-signal protection inside mapped pockets: signature-pure is one
coherent-signal class, reward-pure may be another, and mixed-signal
controllers destabilize when the mixture creates inference noise. Full v2
envelope (six policies including the `lambda=1.0` bootstrap-failure datapoint)
is at [`mesa/PHASE7_V2_RESULTS.md`](mesa/PHASE7_V2_RESULTS.md). v3 spec
filed at [`mesa/PHASE7_V3_SPEC.md`](mesa/PHASE7_V3_SPEC.md) (2026-05-18) —
intervention battery to disambiguate `lambda=0.99` recovery; ~60–70 min
operator wall-clock, no new training.

**Phase 8:** Not started.

### MESAv2 Spine — Status Roll-Up (2026-05-18)

The v2 spine is filed separately at
[`SUNDOG_V_MESAV2.md`](SUNDOG_V_MESAV2.md) as a sister roadmap; this
roll-up surfaces its state from the v1 Implementation Status entry
point so the spine isn't invisible to readers landing here.

| v2 Phase | Status | Last touched | Notes |
| --- | --- | --- | --- |
| 0.5 — Traceability Claim Boundary | ✅ complete (2026-05-13) | 2026-05-13 | 4-row layer taxonomy; mirrored to `docs/presentation/claims-and-scope.md` |
| 0.6 — Bayes-Net / NN / Sundog Comparison | ✅ complete (2026-05-13, stamped 2026-05-18) | 2026-05-18 | Comparison table in-doc; exit criterion self-check landed |
| 1' — Reference Signature Path Audit | ✅ complete — GREEN (2026-05-18) | 2026-05-18 | All 4 LTs pass; clean-traceability branch routed to Phase 8'. Spec [`mesa/PHASE1_PRIME_SPEC.md`](mesa/PHASE1_PRIME_SPEC.md) v1.2; result note [`mesa/V2_SIGNATURE_PATH_AUDIT.md`](mesa/V2_SIGNATURE_PATH_AUDIT.md); v1 PHASE1_HC_BASELINE unamended |
| 6.5 — Formal Separability Counterexample Pass | ⏳ not started | 2026-05-13 | ~1–2 weeks Small-tier; sequenced after 1'; reserves `sensor-hacked` / `geometry-hacked` labels |
| 7' — Envelope Cell Relabeling | ✅ complete + 2× extended | 2026-05-18 | v1 retrofit landed 2026-05-13; **v2 envelope extension (Phase 7 v2)** and **v3 vocabulary extension (Phase 7 v3, new labels `field-coupled, under-budget` and `bootstrap-collapse`)** both landed 2026-05-18 |
| 8' — Traceability Claim Ratchet | ⏳ not started | 2026-05-13 | Post-v2 writeup; depends on 0.5/0.6/1'/6.5/7' |

The v3 labels in 7' partially falsified the v2 `reward-coupled
(profile)` qualifier for the Large U-trough cells (now
`field-coupled, under-budget`) and partially falsified the v2
`undertrained` label for the bootstrap-failed cell (now
`bootstrap-collapse`). Phase 7 v2 amendments to surface those
relabels were queued in [`mesa/PHASE7_V3_RESULTS.md`](mesa/PHASE7_V3_RESULTS.md) §6
and applied to the v2 result note as v2.1 on 2026-05-28
([`mesa/PHASE7_V2_RESULTS.md`](mesa/PHASE7_V2_RESULTS.md) §12).

Mechanistic side-thread on the v3 cliff pair (Phase 6b) closed
2026-05-18 with GG6b-loc-D — the Phase 6 v1 activation-patching
protocol does not generalize to Large for the mixed_0_99 /
mixed_0_97 pair (destructive transfer at every swept layer). Result
note at [`mesa/PHASE6B_RESULTS.md`](mesa/PHASE6B_RESULTS.md); does
not affect the v2 spine vocabulary.
