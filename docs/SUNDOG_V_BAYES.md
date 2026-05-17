Roadmap starts at line ~125


It should look less like “Bayes versus Sundog, one wins” and more like:

Bayes turns evidence into belief. Sundog turns response into control.

That is the cleanest head-to-head.

Bayes’ Theorem asks:

P(world∣signal)∝P(signal∣world)P(world)

Sundog asks:

H(x)≈
dτ
dS
	​


or, in plain terms:

When I act, how does the indirect signal change, and can I use that change to steer?

The fair comparison is not “Bayes gets full state, Sundog gets shadows.” That would be a strawman. A fair Bayesian baseline also gets partial observations. The difference is that Bayes tries to reconstruct a posterior over the hidden world, while Sundog tries to couple action directly to the signal field. The Sundog docs already define the shared pattern this way: the system does not observe the whole truth, receives an indirect signal, transforms it into a control-relevant form, and acts from that transformed signal rather than from full world-state symmetry.

The simplest headline:

Bayes: “Where is the hidden thing, probably?”
Sundog: “Which action makes the trace improve?”

For a visual demo, I would make it a split-screen duel.

Left side: Bayes.
A dark board, hidden target, and a posterior heatmap. Every observation updates the heatmap. The UI shows “prior,” “likelihood,” “posterior,” and “expected utility.” It feels analytical, cool, map-like.

Right side: Sundog.
Same hidden target, but no posterior heatmap. Instead, a live signal field, detector bars, scan arcs, and a trace line showing signal improvement. It feels embodied, experimental, servo-like.

The shared task:

A hidden source emits an indirect signal. The agent must align to it, reach it, or keep a system stable around it. Both agents receive the same sensor readings. Neither receives privileged target coordinates.

The fair Sundog version should use the real project shape: detector intensity plus proprioception, no laser position, no mirror hit point, no target Cartesian position, and a SCAN → SEEK → TRACK loop. That is the current measurable form of the older Sundog idea.

The head-to-head should have three rounds.

Round 1: Clean model. Bayes should win.

The hidden source produces a known Gaussian field. Noise is known. Geometry is known. The Bayesian agent has the correct likelihood model.

Bayes updates its posterior efficiently, localizes the hidden target, and chooses the best action. Sundog still has to scan, seek, and track. It can work, but it pays the acquisition-time cost. This matches the current Sundog evidence posture: the photometric controller can match terminal oracle accuracy in the tested mirror task, but it is much slower, with median time-to-threshold around 188 steps versus 11.5 for the target-aware analytic baseline.

Verdict stamp: BAYES WINS — MODEL KNOWN

Overlay copy:

When the likelihood is right, inference is powerful.
Bayes does not need to wander. It can believe its way toward the target.

Round 2: Unknown response surface. Sundog should look strong.

Now keep the target hidden, but corrupt the model: warped optics, shifted detector calibration, unmodeled surface reflection, mild occlusion, or a field whose shape is not the assumed likelihood.

The Bayesian agent updates confidently into the wrong posterior because its likelihood is wrong. Sundog does not need the full generative model. It perturbs, reads the actual response, and climbs what the world is actually giving back.

Verdict stamp: SUNDOG HOLDS — FIELD READABLE

Overlay copy:

The map lied.
The response still answered.

This is the core Sundog pitch without overclaiming. The project’s promo language already frames the idea as indirect environmental response carrying actionable structure, not perfect world reconstruction.

Round 3: Decoy or local maximum. Sundog should fail.

Add a false signal peak. The visible trace is locally attractive but globally wrong. A flat Sundog controller locks onto the decoy because the local signal improves there. A Bayesian agent with the right structural model can represent two hypotheses, gather disambiguating evidence, and eventually reject the decoy.

Verdict stamp: BOUNDARY FOUND — SIGNAL ALIASED

Overlay copy:

A signal can be useful without being truthful.
Sundog fails when the trace improves for the wrong reason.

This is the most important round. It prevents the comparison from becoming marketing. It says: Bayes is better when the system needs explicit uncertainty over hidden causes; Sundog is better when the goal is direct control from a real response field and a full posterior is unnecessary, unavailable, or too expensive.

The landing-page card could read:

Bayes vs. Sundog
Status: Comparative Benchmark
Bayes: Evidence → posterior → action.
Sundog: Probe → response → control.
Question: Must the agent infer the hidden world, or can it act from the trace?
Verdict: Complementary, not redundant.

For the rail, I would avoid a “Sundog beats Bayes” stamp. Use a sequence of stamps instead:

MODEL KNOWN → Bayes wins.
FIELD READABLE → Sundog holds.
SIGNAL ALIASED → Sundog fails.
HYBRID NEEDED → real systems use both.

The deeper theorem-level distinction:

Bayes is an epistemic theorem. It is about how belief should change when evidence arrives.

Sundog is a control hypothesis. It is about when indirect environmental structure can be transformed into useful action without reconstructing the full hidden state.

So the best head-to-head line is:

Bayes is for knowing under uncertainty. Sundog is for acting under partial sight.

A strong final card:

Title: Posterior vs. Halo
Clip: Left side heatmap sharpens; right side signal trace climbs. Then a decoy appears. The heatmap splits; the Sundog trace locks onto the wrong peak.
Stamp: COMPLEMENTARY
Description: Bayes inferred the hidden cause. Sundog followed the live response. Both were right until the signal stopped meaning what it seemed to mean.
Theorem meaning: Sundog does not replace inference. It identifies the cases where action-coupled signal is enough, and the cases where belief over hidden causes is still necessary.

This would be one of the strongest public-facing comparisons because it respects Bayes instead of caricaturing it, while making Sundog’s distinct claim sharper.



# Sundog vs. Bayes

Working hook:

> Bayes turns evidence into belief. Sundog turns response into control. This
> roadmap tests when the posterior is worth paying for, when the halo is enough,
> and where a hybrid is required.

This roadmap defines a comparative benchmark between Bayesian inference and
Sundog-style response control. It is not a claim that Sundog replaces Bayes.
Bayes is the correct comparator precisely because it is strong: it already
handles uncertainty, hidden causes, partial observations, and principled belief
updates. A weak baseline would make the comparison easier and less useful.

The experiment asks a narrower question:

> Given the same partial observations and the same action budget, should an
> agent maintain an explicit posterior over hidden causes, or can it act directly
> from the action-coupled response field?

The intended result is not one global winner. The intended result is an
envelope:

- known model: Bayes should win;
- readable but mis-modeled field: Sundog should hold;
- aliased or decoy field: flat Sundog should fail;
- mixed regimes: hybrid should earn its keep.

## Why Bayes Is the Right Next Comparator

The current Sundog research claim is narrow: a controller without Cartesian
access to a target can align a mirrored end-effector using sparse photometric
feedback and proprioception, reaching terminal accuracy comparable to a
target-aware analytic baseline in the tested MuJoCo setting. That is a result
against an oracle-style direct-state baseline, not yet against the most serious
partial-observation inference baseline.

Bayes is the serious partial-observation comparator because it does not require
full sight. A Bayesian agent can be denied the true target location, receive the
same indirect signal as Sundog, and still maintain a posterior over hidden
state. If Sundog only works because its baseline is unfairly blind, the Bayes
comparison will expose that. If Sundog is genuinely a different control regime,
the comparison should reveal where direct response control is cheaper, more
robust, or more honest than posterior reconstruction.

The clean distinction:

```text
Bayes:   observation -> likelihood -> posterior -> action
Sundog:  probe       -> response   -> control   -> next probe
```

Bayes asks:

```text
P(hidden_world | signal) ∝ P(signal | hidden_world) P(hidden_world)
```

Sundog asks:

```text
H(x) ≈ dS / dτ
```

or, in operational language:

> When I act, how does the indirect signal change, and can that change steer me?

This comparison is worth running because it will prevent two bad public reads:

1. **Anti-Bayes caricature.** Bayes is not “full-state classic AI.” Bayes is a
   principled way to reason under partial observation.
2. **Sundog inflation.** Sundog is not a universal replacement for inference.
   It is a control posture for cases where the response field is actionable
   before the hidden cause is fully reconstructed.

## What's Honest vs. What's Reach

**Honest:**

- A measured envelope showing when a correct Bayesian model beats direct
  response control on acquisition time, terminal accuracy, calibration, and
  wrong-lock rate.
- A measured envelope showing when a Sundog response controller remains useful
  under model drift, calibration warp, or unknown response geometry where a
  fixed Bayesian likelihood misleads.
- A serious Sundog failure card: signal aliasing / decoy response fields where
  the trace improves for the wrong reason and a flat response controller locks
  onto the wrong basin.
- A hybrid result, if earned, showing when posterior maintenance should be used
  for hypothesis management while Sundog-style response tracking handles local
  control.
- A public visualization that makes the difference legible: posterior heatmap
  versus live response trace, then a decoy case where the heatmap splits and the
  response trace lies.

**Reach (do not claim):**

- "Sundog beats Bayes."
- "Bayesian inference requires full state."
- "Posterior inference is obsolete."
- "Sundog is a faster Bayesian method."
- "Sundog solves partial observability in general."
- "Sundog replaces uncertainty modeling."
- "The failure of one Bayesian implementation refutes Bayes."

The value of this roadmap is complementarity, not conquest. A result where
Bayes wins cleanly under a known model is not a Sundog failure. It is the
calibration point. A result where Sundog fails under aliasing is not a program
failure either. It is the boundary the public rail needs: **the signal can be
useful without being truthful.**

## Ratified Hook Language

Safe hook:

> Sundog vs. Bayes asks whether an agent under partial observation must infer
> the hidden world, or whether it can sometimes act directly from the trace the
> world gives back.

Short version:

> Bayes is for knowing under uncertainty. Sundog is for acting under partial
> sight.

Public one-liner:

> The posterior draws a map. The halo moves the hand.

Avoid:

- "Bayes sees the whole world."
- "Sundog beats probabilistic reasoning."
- "Inference is unnecessary."
- "The posterior is just another proxy."
- "Bayes is slow, Sundog is fast." The current photometric result shows the
  opposite cost shape: Sundog can pay large acquisition time to avoid target
  coordinates.

## Core Experimental Question

The shared task family is hidden-source navigation / alignment.

A hidden target or source emits an indirect signal field. The agent controls a
2D point, mirror pose, probe pose, or simple embodied body. It receives only
sensor observations derived from the field and its own proprioceptive/action
history. It does not receive the hidden target coordinates. The agent must
align to the source, maximize target intensity, maintain a desired response, or
choose when to abstain.

Both Bayes and Sundog receive the same raw observations. The difference is how
they transform them.

- The Bayesian family maintains a belief distribution over hidden causes and
  chooses actions by expected utility, information gain, Thompson sampling,
  model-predictive planning, or a simple value-of-information rule.
- The Sundog family does not maintain a full posterior. It scans, seeks, tracks,
  reacquires, and acts from measured changes in the response field.
- The hybrid family uses posterior structure only for hypothesis management,
  disambiguation, and reset decisions, while using response tracking for local
  control.

The comparison is fair only if Bayes is allowed to be genuinely Bayesian under
partial observation. The comparison is unfair if Bayes is forced to be a direct
state oracle or if Sundog is allowed privileged geometry.

## Applied Workbench Cross-References

This roadmap now has two lanes:

1. the standalone Bayes-vs-Sundog benchmark defined below; and
2. Bayesian-floor profiles stamped into existing Sundog workbenches as
   claim-hygiene comparators.

The applied profiles are not substitutes for the standalone benchmark. They are
the local audit surfaces that keep each workbench from accidentally claiming too
much from weak baselines.

| Workbench | Bayesian comparator status | Cross-reference |
| --- | --- | --- |
| Balance | Executable same-shadow floor scaffold. `bayes_floor_shadow_particle` runs with observation parity, no-state-leak, unknown-mode rejection, Phase 10 slate loading, and regret receipts. Final Phase 15 lock ran 27,200 trials with audits passing, claim gate 56/56 hard-gate cells, 12 reported-only observation-degradation cells, zero negative mean-regret cells, and aggregate regret versus `sundog_shadow` +0.00395. Treat Balance as having earned a same-information Bayesian-floor claim-lock receipt, with reported-only delay/noise/dropout failure-regime lanes preserved as boundary diagnostics. | [`SUNDOG_V_BALANCE.md` - Bayesian Floor Profile](SUNDOG_V_BALANCE.md#bayesian-floor-profile) and Phase 15 status |
| Pressure Mines | Active applied profile. The staged same-field baseline is `mines-bayesian-baseline-v1`: a frontier-limited posterior over hidden mine occupancy, strict budget parity between pressure-only and full Sundog-legal lanes, and a Phase 12 path that starts with legal `Phi_t` serialization, no-leak, observation parity, and tiny posterior smoke before any claim lock. Admission plumbing exists (`npm run mines:phase12:admission`). The repaired executable smoke exists (`npm run mines:phase12:posterior-smoke`) and uses shared cell/seed sensor streams plus a pressure-floor action guard. The 64-seed Phase 10 reducer ran 14,720 trials with no-leak pass: `bayes_frontier_pressure` matches `naive_pressure` / `sundog_minimal` exactly across all 46 cells, while `sundog_lean` remains stronger in the promoted candidate cell. Phase 13 now publishes `public/data/mines-phase13-bayes-floor.json`, a Mines evidence panel, representative posterior-cell hazard/pressure/confidence slices, and replay selectors for confirmed, failure, and Bayes-divergence cells. Treat this as a same-field pressure-floor receipt, not a posterior-dominance claim lock. | [`sundog_v_minesweeper.md` - Bayesian Baseline Profile](sundog_v_minesweeper.md#bayesian-baseline-profile) and Phase 13 status |
| Three-Body / Coarse Graining | Current pattern source for the receipt shape: admission spec, same-information guard, belief diagnostics, regret reducer, and capped-probe discipline. Still not a closed floor. | [`BAYESIAN_FLOOR_PROFILE_TEMPLATE.md`](BAYESIAN_FLOOR_PROFILE_TEMPLATE.md) and [`proof/PHASE4_THREEBODY.md`](proof/PHASE4_THREEBODY.md) |

The interpretation rule is shared across all applied profiles:

- if Bayes is invalid or weak, repair it before interpreting the Sundog result;
- if Bayes dominates, narrow the Sundog claim to a weaker controller or
  heuristic claim;
- if Sundog tracks a repaired same-observation floor, the workbench earns
  stronger evidence that response-control recovered most of the actionable
  information in that envelope;
- if both fail in the same cells, the boundary is more likely to be an
  observability boundary than merely a controller boundary.

Applied queue after the Balance lock:

1. **Pressure Mines Phase 12 admission path.** Declare
   `bayes_frontier_pressure` / `bayes_frontier_full`, serialize the legal
   `Phi_t`, and prove no-leak / observation parity before implementing the
   posterior policy. Landed with `npm run mines:phase12:admission`.
2. **Pressure Mines tiny posterior smoke.** Run only the Phase 10 confirmed
   pocket plus paired failure cell with capped particles and seeds. Landed with
   `npm run mines:phase12:posterior-smoke`; no-leak and easy-cell sanity now
   pass after the same-field sensor repair.
3. **Pressure Mines pocket probe.** Broaden to the `bayes_frontier_full` pocket
   probe and reduce Bayes-vs-`sundog_minimal` / `sundog_lean` regret into the
   Phase 12 summary shape. Landed with `npm run mines:phase12:pocket-probe`.
4. **Pressure Mines Phase 10-slate reducer.** Run the repaired pressure lane
   over the locked Phase 10 envelope cells, then surface Bayes regret fields
   only if the same-field floor holds. Loader smoke landed with
   `npm run mines:phase12:phase10-slate:smoke`; the 64-seed all-cell reducer
   landed as `npm run mines:phase12:phase10-slate`. Public data surfacing now
   lands through `npm run mines:public-data`, `mines.html`, and the
   `alignment.html` side-by-side, with explicit language that this is
   pressure-floor parity, not posterior dominance over `sundog_lean`.
   Phase 13.1 adds representative posterior-cell slices and replay selectors
   for the confirmed pocket, paired failure, and Bayes-divergence cell.
5. **Template backfill.** Any Mines-specific deviations from the generic
   [`BAYESIAN_FLOOR_PROFILE_TEMPLATE.md`](BAYESIAN_FLOOR_PROFILE_TEMPLATE.md)
   have now been backfilled: admission lanes, same-observation seed policy,
   claim-gate fields, reported-only lanes, posterior/evidence slices, and
   public interpretation copy.
6. **Standalone Phase 1 exact reference.** Landed as
   `scripts/bayes-phase1-reference.mjs` with `npm run bayes:phase1:smoke` and
   `npm run bayes:phase1`. The 128-seed lock passes the known-model gate before
   any `bayes.html` work begins.
7. **Standalone Phase 2 mismatch slate.** Landed in the same runner with
   `npm run bayes:phase2:smoke` and `npm run bayes:phase2`. The lock finds a
   narrow anisotropic separation for `sundog_memory` over fixed clean
   `bayes_misspecified`; other variants remain boundary or Bayes-favorable.
8. **Standalone Phase 3 aliasing lock.** Pre-registered dual gate landed and
   `npm run bayes:phase3` reports `dual_gate_pass` in `decoy` and `alias`.
   `symmetric` and `low_probe` remain diagnostic: both show Bayes-adaptive
   recovery, but not the pre-registered HC-Sundog wrong-lock failure arm.

## Controller-Family Architecture

Run all controller families on the same seeded slates, same observation stream,
same action budget, and same termination rules.

### 1. Oracle

Privileged ceiling. Knows the hidden source, true field parameters, and current
geometry. It is not a deployed baseline. It defines the best plausible terminal
and acquisition behavior when the hidden state is granted.

Use it to normalize results and detect impossible task settings. Do not use it
as the primary Bayes comparator.

### 2. Bayes-Correct

Bayesian agent with the correct generative model family and correct noise model.
It does not know the hidden target value, but it knows the likelihood class.
It maintains a posterior over target location, field parameters, or discrete
hypotheses.

Expected behavior: should win Round 1. If it does not, the Bayesian
implementation is probably too weak.

Candidate implementations:

- grid posterior over hidden target location for small 2D fields;
- particle filter for continuous target / source parameters;
- Kalman or extended Kalman filter only where the model is legitimately close
  to linear-Gaussian;
- Gaussian-process surrogate with explicit uncertainty for unknown smooth
  response surfaces;
- active Bayesian optimization for expensive probe settings.

### 3. Bayes-Misspecified

Bayesian agent with a plausible but wrong likelihood. Examples: assumes a
Gaussian field when the actual response is warped, assumes stationary detector
noise when noise is position-dependent, assumes one source when two interacting
sources exist, or assumes the decoy field is weak when it is strong.

Expected behavior: should become confidently wrong in some cells. This is the
main place Sundog should look strong if the response field remains locally
readable.

### 4. Bayes-Adaptive

Bayesian agent that can update some model parameters online or maintain a small
mixture over model families. This is the fairer version of Bayes in the
mis-specified setting. It should recover from some mismatch, but pay an action,
compute, or sample cost.

Expected behavior: may beat both Bayes-Misspecified and Sundog in some cells.
That is acceptable. The comparison should report the cost and failure boundary,
not force Sundog to win.

### 5. HC-Sundog

Hand-coded response controller. Uses SCAN / SEEK / TRACK / REACQUIRE over the
observed signal. Does not represent a posterior over hidden target location.
It may keep memory of best observed response, local slope estimates, phase
state, and confidence gates.

Expected behavior: strong when the action-response field is smooth enough,
readable enough, and not severely aliased. Weak under decoys, disjoint basins,
long-horizon preparatory actions, and cases where the best local response is
not evidence of the true hidden target.

### 6. Sundog-Memory

Response controller with richer empirical memory but still no posterior over
hidden causes. Stores local response samples, basin labels, reacquisition
points, and recent probe outcomes. This separates "Sundog fails because it has
no memory" from "Sundog fails because it lacks causal belief over hidden
hypotheses."

Expected behavior: should improve over HC-Sundog in noisy or drifting fields,
but still fail under structurally aliased signals.

### 7. Hybrid Bayes-Sundog

Maintains a compact posterior or hypothesis set only where ambiguity matters,
then uses response tracking for local control. The design target is not
"Bayes plus everything." The design target is minimal inference added at the
failure boundary.

Expected behavior: should dominate flat Sundog in aliasing cells and avoid the
full overhead of Bayes-Correct in simple readable fields. If it does not, the
hybridization is unjustified.

### 8. Naive Local / Random

Lower-bound baselines.

- Random or passive.
- Greedy local ascent without scan.
- Nearest-highest-sensor heuristic.
- Posterior heatmap without active probing.

These are not the serious comparison. They detect implementation leakage and
provide public legibility.

## Decision Lock for Phase 0

### Environment family

Use **shadow-field navigation** as the first environment because it matches the
Mesa infrastructure, is cheap to simulate, supports exact posteriors in small
versions, and can generate clean probe slates.

Canonical Phase 0 task:

- Agent moves in bounded 2D continuous space.
- Hidden target `x*` emits scalar response field `S(x)`.
- Agent observes local field samples around its body, plus its own position and
  action history. It does not observe `x*`.
- Action is bounded 2D velocity or displacement.
- Objective is to reach and dwell near the true target / source response.
- The slate includes known-model, warped-model, decoy, aliasing, drift, noise,
  and sensor-delay variants.

Start with a grid discretization for exact Bayesian reference. Then port to
continuous particles or GP only after the discrete version is debugged.

Do not start in the MuJoCo mirror task. The photometric task is the scientific
spine, but it is too expensive and geometrically specific for first-pass
posterior diagnostics. After the shadow-field benchmark is stable, port the
best comparison into the photometric mirror setting as Phase 6.5.

### Claim boundary

The roadmap tests a comparative control question, not a theorem of Bayesian
inference.

Possible final claim shape:

> In the tested hidden-source environment family, Bayesian inference dominates
> when the likelihood model is correct and ambiguity must be represented;
> Sundog-style response control remains competitive or more robust when the
> response field is locally readable but the model is mis-specified; flat
> Sundog fails under aliased response fields, motivating a hybrid controller.

If the result is one-sided in Bayes's favor:

> In the tested environment family, posterior-maintaining controllers dominate
> response-only controllers across known, mis-specified, and aliased fields
> under matched action budgets. Sundog should be reframed as a lightweight
> control heuristic that works only when posterior structure is unnecessary.

If the result is one-sided in Sundog's favor:

> In the tested environment family, response-only control outperforms the
> Bayesian implementations under the chosen action/compute budget. This does
> not refute Bayes; it shows that the tested posterior machinery was not worth
> its cost in these readable fields. Stronger Bayesian baselines remain a
> required future pass.

### Evidence tier

Initial status: **candidate comparative benchmark / roadmap**.

A completed Phase 3 result becomes an **Operating-Envelope Study** only if it
has:

- seeded matched trials;
- at least one serious Bayesian baseline;
- at least one Sundog failure boundary;
- action-budget and compute-budget reporting;
- saved run artifacts;
- public summary that avoids "Sundog beats Bayes" language.

## Experimental Rounds

### Round 1 — Known Model

Question:

> What happens when Bayes has the right likelihood?

Setup:

- Hidden source emits known Gaussian or known mixture field.
- Noise distribution is known.
- Sensor geometry is known.
- Agent does not know target coordinates.
- Bayes-Correct maintains posterior over `x*`.
- HC-Sundog scans, seeks, and tracks response.

Expected result:

Bayes-Correct should win on acquisition time, probe efficiency, and possibly
terminal accuracy. Sundog may still succeed, but it should pay a scan cost.

Verdict language:

> MODEL KNOWN — BAYES WINS

This round protects the comparison from becoming anti-inference marketing.

### Round 2 — Warped Field / Wrong Likelihood

Question:

> What happens when the map is wrong but the response is still readable?

Setup:

- Actual field is warped, anisotropic, shifted, clipped, delayed, or mildly
  occluded.
- Bayes-Misspecified assumes the clean field.
- Bayes-Adaptive can fit some parameters but not arbitrary geometry.
- HC-Sundog sees only the live response.

Expected result:

Bayes-Misspecified should sometimes become confidently wrong. HC-Sundog should
track the real response when local gradients remain actionable. Bayes-Adaptive
may recover with a sample cost.

Verdict language:

> FIELD READABLE — SUNDOG HOLDS

This is the strongest pro-Sundog round, but only if the Bayesian baseline is
not artificially weakened.

### Round 3 — Aliased Field / Decoy Basin

Question:

> What happens when the signal improves for the wrong reason?

Setup:

- Add a false response peak or aliased basin.
- Local response improves near the decoy.
- The true target can be disambiguated only by a structured probe, prior, time
  sequence, or multi-sensor pattern.
- Bayes-Correct maintains two hypotheses.
- HC-Sundog follows local improvement.

Expected result:

Flat Sundog should fail. Sundog-Memory may reduce repeated wrong-locks but
should not solve true aliasing without adding belief over hidden causes.
Bayes-Correct or Hybrid should win if the disambiguating evidence is present in
the observation stream.

Verdict language:

> SIGNAL ALIASED — BOUNDARY FOUND

This is the key failure card for the landing-page rail.

### Round 4 — Drift and Reacquisition

Question:

> What happens when the response field changes after the agent has committed?

Setup:

- Target moves slowly, detector calibration drifts, or noise statistics change.
- Some changes preserve local response; others invalidate prior structure.
- Agents must detect failure and reacquire.

Expected result:

Sundog may reacquire quickly when the response itself changes visibly.
Bayes may handle drift better if the motion model is correct. Hybrid should
win when drift alternates between readable and ambiguous regimes.

Verdict language:

> REACQUIRE — MODEL OR RESPONSE?

### Round 5 — Compute/Action Budget Sweep

Question:

> Is the posterior worth its cost?

Setup:

Sweep:

- action budget;
- compute budget;
- sensor count;
- noise;
- field smoothness;
- decoy strength;
- prior quality;
- model mismatch severity.

Expected result:

Produce a Pareto map, not a single winner. Regions should be labeled:

- Bayes-dominant;
- Sundog-dominant;
- Hybrid-dominant;
- all-fail;
- ambiguous / needs stronger baseline.

Verdict language:

> OPERATING ENVELOPE

## Metrics

Primary metrics:

- terminal response / terminal target intensity;
- time-to-threshold;
- cumulative response / regret;
- wrong-lock rate;
- dwell success after threshold;
- reacquisition time after drift;
- action count;
- compute time;
- memory footprint.

Bayes-specific metrics:

- posterior entropy over time;
- posterior calibration;
- negative log likelihood of held-out observations;
- posterior mass on true target / true hypothesis;
- time to disambiguate decoy;
- information-gain per action.

Sundog-specific metrics:

- best-observed response memory;
- scan coverage;
- local slope estimate quality;
- reacquisition trigger correctness;
- false-track duration;
- response-gradient collapse rate.

Shared failure labels:

- **Known-model Bayes win.** Posterior correctly concentrates; Sundog pays scan
  cost.
- **Wrong-model Bayes collapse.** Posterior concentrates on wrong basin because
  likelihood is false.
- **Readable-field Sundog win.** Local response remains coupled to useful
  action despite model mismatch.
- **Signal alias Sundog collapse.** Local response improves toward false basin.
- **Hybrid necessity.** Direct response gets close, posterior disambiguation
  prevents wrong lock.
- **All-fail.** Signal contains insufficient information or action budget is too
  small.

## Roadmap

### Phase 0 — Scope and Literature Pass

Goal: pin the comparison so it is fair to Bayes and useful to Sundog.

Deliverables:

- One-page claim boundary for the Bayes comparison.
- Definitions of observation, hidden state, action, field, and success.
- Baseline list: Oracle, Bayes-Correct, Bayes-Misspecified, Bayes-Adaptive,
  HC-Sundog, Sundog-Memory, Hybrid, Naive Local, Random.
- Applied-profile index pointing to Balance, Pressure Mines, and Three-Body so
  the standalone benchmark inherits the strongest existing receipt conventions
  instead of inventing a parallel shape.
- Literature spine covering Bayesian filtering, Bayesian decision theory,
  active inference / active sensing, Bayesian optimization, POMDPs, extremum
  seeking, visual servoing, and model mismatch.
- Decision about first environment: discrete 2D grid first, continuous field
  second.
- Budget policy: action budget, compute budget, and wall-clock reporting.

Exit criterion: no one can plausibly object that Bayes was straw-manned before
implementation starts.

### Phase 1 — Reference Task and Exact Bayesian Baseline

Goal: build the smallest task where Bayes is exact and should win.

Deliverables:

- Seeded 2D hidden-source grid environment.
- Exact grid posterior over hidden target location.
- Known Gaussian/noisy field observation model.
- Bayes-Correct policy using expected utility and optionally information gain.
- HC-Sundog SCAN / SEEK / TRACK controller over the same observations.
- Oracle and random baselines.
- JSONL trial logs and replay manifest.

Status, 2026-05-17: executable lock landed.

Commands:

```bash
npm run bayes:phase1:smoke
npm run bayes:phase1
```

Receipt paths:

- `results/bayes/phase1-reference-smoke/manifest.json`
- `results/bayes/phase1-reference-lock/manifest.json`
- `results/bayes/phase1-reference-lock/trials.jsonl`
- `results/bayes/phase1-reference-lock/steps.jsonl`
- `results/bayes/phase1-reference-lock/replay-manifest.json`

Phase 1 lock summary (`npm run bayes:phase1`, 128 seeds, 512 trials, 29.529s
after the Phase 2 schema extension):

| Mode | Successes | Success rate | Mean score | Mean hit turn |
| --- | ---: | ---: | ---: | ---: |
| `oracle` | 128/128 | 1.000000 | 1.623372 | 9.039063 |
| `bayes_correct` | 126/128 | 0.984375 | 1.486077 | 11.738095 |
| `hc_sundog` | 27/128 | 0.210938 | 0.062268 | 11.481481 |
| `random` | 10/128 | 0.078125 | -0.257697 | 12.900000 |

Exit gate: **pass**. `bayes_correct` beats `hc_sundog` under the known model:
mean score delta +1.423809, success-rate gate true, score gate true. The hit
turn comparison is only gate-active when success rates are tied, to avoid
rewarding a controller for succeeding only on easy seeds.

Representative replay roles are now selected in
`replay-manifest.json`: `bayes_advantage`, `closest_pair`, and
`bayes_boundary`. The boundary role is useful claim hygiene, not a Phase 1
failure: Bayes still clears the aggregate known-model gate, but the replay keeps
the two missed seeds visible before Phase 2.

Exit criterion: Bayes-Correct beats or matches HC-Sundog under the known model.
If it does not, fix Bayes before proceeding.

### Phase 2 — Model Mismatch Slate

Goal: test whether Sundog has a real advantage when likelihood structure is
wrong but response remains actionable.

Deliverables:

- Warped Gaussian field.
- Anisotropic field.
- Detector calibration shift.
- Mild occlusion / clipped field.
- Delayed sensor response.
- Bayes-Misspecified baseline locked to the clean likelihood.
- Bayes-Adaptive baseline with limited parameter learning.
- Matched trials against HC-Sundog and Sundog-Memory.

Status, 2026-05-17: executable mismatch lock landed.

Commands:

```bash
npm run bayes:phase2:smoke
npm run bayes:phase2
```

Receipt paths:

- `results/bayes/phase2-mismatch-smoke/manifest.json`
- `results/bayes/phase2-mismatch-lock/manifest.json`
- `results/bayes/phase2-mismatch-lock/summary.csv`
- `results/bayes/phase2-mismatch-lock/regret.csv`
- `results/bayes/phase2-mismatch-lock/replay-manifest.json`

Implementation details:

- `bayes_misspecified` keeps an exact grid posterior, but its likelihood is
  locked to the clean radial Gaussian regardless of the true field variant.
- `bayes_adaptive` is a finite-mixture Bayesian baseline over clean, warped,
  anisotropic, calibration-shift, clipped, and delayed model families.
- `sundog_memory` is the response-control repair lane: it scans until the live
  signal rises above its own starting baseline, then probes from remembered
  high-response cells.

Phase 2 lock summary (`npm run bayes:phase2`, 96 seeds, 2,880 trials,
239.345s):

| Scenario | `bayes_misspecified` score / success | best response lane | best response score / success | Result |
| --- | ---: | --- | ---: | --- |
| `anisotropic` | 0.767947 / 0.500000 | `sundog_memory` | 1.359587 / 0.875000 | separated |
| `calibration_shift` | 0.520778 / 0.458333 | `sundog_memory` | 0.549783 / 0.500000 | near miss under +0.1 margin |
| `warped` | 0.978200 / 0.614583 | `sundog_memory` | 0.905387 / 0.677083 | response wins survival, loses score |
| `clipped` | 1.646094 / 1.000000 | `sundog_memory` | 0.785773 / 0.604167 | Bayes-favorable |
| `delayed` | 1.466406 / 1.000000 | `hc_sundog` | 0.108966 / 0.208333 | Bayes-favorable |

Exit gate: **separation_found**. The current Phase 2 receipt supports a narrow
claim: under an anisotropic response surface, fixed clean Bayes is brittle and
the memory response-control lane recovers better. It does **not** support a
blanket mismatch claim; clipped and delayed variants still favor the fixed
posterior baseline, and calibration shift is a close boundary rather than a
locked win.

Exit criterion: at least one mismatch regime cleanly separates fixed posterior
failure from response-control recovery, or the roadmap records that the tested
mismatch was insufficient.

### Phase 3 — Aliasing and Decoy Failure Slate

Goal: force the failure the public rail needs.

Deliverables:

- False peak / decoy field variant.
- Two-source alias variant.
- Symmetric ambiguity variant.
- Low-probe-budget ambiguity variant.
- Disambiguating probe protocol available to Bayes and Hybrid.
- Wrong-lock classifier.
- Representative replay clips.

Pre-registered smoke-harness-first spec, locked before the Phase 3 smoke run:

| Scenario | Field shape | Intended Sundog failure |
| --- | --- | --- |
| `decoy` | True radial Gaussian plus lower-amplitude false peak at a seeded decoy cell. | Flat climb locks onto local decoy. |
| `alias` | True radial Gaussian plus near-equal seeded second source. | Response improves toward either basin. |
| `symmetric` | Target-reflected source makes the field nearly symmetric about an axis. | Identical local response on two sides. |
| `low_probe` | `alias` field with `maxTurns` overridden to 20. | No budget to disambiguate. |

Modes:

```text
oracle,bayes_misspecified,bayes_adaptive,hc_sundog,sundog_memory,random
```

Admission notes:

- `bayes_misspecified` remains locked to the clean single-radial likelihood.
- `bayes_adaptive` extends the finite model mixture with `decoy`, `alias`, and
  `symmetric` families; this is the recovery lane. No new Hybrid mode is
  admitted in Phase 3.
- `low_probe` is a scenario-level budget override; the npm commands stay fixed
  at `--max-turns 40`, and the runner applies 20 only inside `low_probe`.

Wrong-lock classifier:

```text
wrong_lock =
  !success
  AND dist(finalPos, nearestNonTargetBasin) <= 1.5
  AND dwellTurns(within 1.5 of that basin) >= 5
```

Dual gate: Phase 3 is `dual_gate_pass` only if both arms are true in the same
eligible scenario; otherwise it records `aliasing_insufficient` and exits
non-fatally (`pass: true`), matching the Phase 2 pattern.

| Arm | Fixed threshold |
| --- | --- |
| Failure arm | `hc_sundog` wrong-lock rate >= 0.50 in at least one `decoy`, `alias`, or `low_probe` regime. |
| Recovery arm | In that same scenario, `bayes_adaptive` versus the best Sundog lane has `successDelta >= 0` and `scoreDelta >= 0.10`. |

Commands:

```bash
npm run bayes:phase3:smoke
npm run bayes:phase3
```

The lock command was pre-registered before the smoke run and then run unchanged.
Do not edit the thresholds, classifier radius/dwell, or recovery margin after
results land.

Smoke receipt, 2026-05-17:

```text
npm run bayes:phase3:smoke
384 trials in 104.937s
Exit gate: dual_gate_pass (3/4 dual-gate scenarios)
```

Receipt paths:

- `results/bayes/phase3-aliasing-smoke/manifest.json`
- `results/bayes/phase3-aliasing-smoke/summary.csv`
- `results/bayes/phase3-aliasing-smoke/regret.csv`
- `results/bayes/phase3-aliasing-smoke/replay-manifest.json`
- `results/bayes/phase3-aliasing-lock/manifest.json`
- `results/bayes/phase3-aliasing-lock/summary.csv`
- `results/bayes/phase3-aliasing-lock/regret.csv`
- `results/bayes/phase3-aliasing-lock/replay-manifest.json`

Smoke gate rows:

| Scenario | HC wrong-lock rate | Failure arm | Best Sundog lane | Bayes-adaptive score / success | Score delta | Success delta | Dual gate |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `decoy` | 0.5000 | true | `sundog_memory` | 1.676563 / 1.0000 | +1.648635 | +0.8125 | true |
| `alias` | 0.7500 | true | `sundog_memory` | 1.427670 / 0.8750 | +1.441081 | +0.6875 | true |
| `symmetric` | 0.3125 | false | `sundog_memory` | 0.400235 / 0.3750 | +0.107920 | 0.0000 | false |
| `low_probe` | 0.5000 | true | `sundog_memory` | 1.085602 / 0.8125 | +1.170522 | +0.6250 | true |

Smoke interpretation: the smoke harness was strong enough to proceed to the
pre-registered lock unchanged. The lock receipt below is now the claim-bearing
Phase 3 result.

Lock receipt, 2026-05-17:

```text
npm run bayes:phase3
2,304 trials in 512.632s
Exit gate: dual_gate_pass (2/4 dual-gate scenarios)
```

Lock gate rows:

| Scenario | HC wrong-lock rate | Failure arm | Best Sundog lane | Bayes-adaptive score / success | Score delta | Success delta | Dual gate |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| `decoy` | 0.572917 | true | `sundog_memory` | 1.722396 / 1.000000 | +1.329126 | +0.604167 | true |
| `alias` | 0.510417 | true | `sundog_memory` | 1.483721 / 0.906250 | +1.060612 | +0.510417 | true |
| `symmetric` | 0.375000 | false | `sundog_memory` | 0.818609 / 0.562500 | +0.210788 | +0.052083 | false |
| `low_probe` | 0.395833 | false | `sundog_memory` | 1.067770 / 0.781250 | +0.819386 | +0.427083 | false |

Interpretation: Phase 3 now has a lock-backed aliasing boundary. `decoy` and
`alias` satisfy the dual gate: HC-Sundog wrong-locks at or above the
pre-registered 0.50 threshold, and Bayes-adaptive recovers with non-negative
success delta plus score delta above +0.10. `symmetric` and `low_probe` show
Bayes-adaptive recovery but do not satisfy the failure arm, so they stay
diagnostic rather than claim support.

Expected outcome:

HC-Sundog should fail in at least one serious decoy regime. If it does not,
increase alias strength carefully, but do not create an impossible toy just to
force failure. The failure must teach something about response control.

Exit criterion: either a real Sundog failure boundary is mapped, or the result
shows that the selected aliasing slate was not strong enough and Phase 3 must
be redesigned.

### Phase 4 - Hybrid Controller

Goal: test the minimum posterior needed to repair the lock-backed Phase 3
`decoy` and `alias` response-control failure regimes.

Frozen claim surface:

- Claim-bearing scenarios: `decoy` and `alias`.
- Diagnostic-only scenarios: `symmetric` and `low_probe`.
- `oracle` remains a privileged yardstick and is never claim-gated.
- `bestSundog := hc_sundog`, because Phase 4's locked mode set has only one
  no-posterior response lane.

Frozen mode set:

```text
oracle,bayes_adaptive,hc_sundog,hybrid,hybrid_no_posterior,hybrid_posterior_only,hybrid_posterior_reset_only,hybrid_posterior_decoy_disambig
```

The repair predicate is fixed once and reused by the repair and load-bearing
arms:

```text
repair(M) =
  (M.meanScore - hc_sundog.meanScore >= 0.10)
  AND (M.successRate - hc_sundog.successRate >= 0)
```

`0.10` is the existing `phase2SeparationMargin`, reused unchanged from Phases
2 and 3.

Three-arm gate, evaluated per scenario:

| Arm | Predicate |
| --- | --- |
| Repair arm | `repair(hybrid)` |
| Frugality arm | `hybrid.meanLikelihoodEvals <= 0.5 * bayes_adaptive.meanLikelihoodEvals` |
| Load-bearing arm | `NOT repair(hybrid_no_posterior) AND repair(hybrid_posterior_decoy_disambig)` |

New frozen constants:

| Constant | Value | Use |
| --- | ---: | --- |
| `FRUGALITY_FRACTION` | 0.5 | Hybrid must use no more than half the full-posterior ceiling's likelihood-eval count. |
| `AMBIGUITY_MASS` | 0.5 | Posterior ambiguity gate for the Phase 3 ambiguity model families. |

`likelihoodEvals` is the cumulative per-candidate Gaussian log-likelihood eval
count used by posterior updates. It is reported per trial and summarized as
`meanLikelihoodEvals`.

Harness policy note: `hybrid` takes the HC-Sundog response action by default
and allows a frugality/ambiguity-paced posterior refresh to override only when
the Phase 3 ambiguity-family mass is at least `AMBIGUITY_MASS`.

Hard-drop override:

```text
if repair(hybrid_no_posterior) is true in either decoy or alias:
  status = hybrid_unnecessary
```

Otherwise, Phase 4 reports:

- `hybrid_niche_confirmed` only if all three arms are true in both `decoy` and
  `alias`;
- `hybrid_no_niche` if the hard-drop does not fire but either claim scenario
  misses at least one arm.

The Phase 4 gate is non-fatal (`pass: true`), matching the Phase 2/3 pattern.

Commands:

```bash
npm run bayes:phase4:smoke
npm run bayes:phase4
```

The smoke ships first. The lock command is staged unchanged for the operator
after the smoke verifies the harness shape.

Smoke receipt, 2026-05-17:

```text
npm run bayes:phase4:smoke
512 trials in 462.877s
Exit gate: hybrid_niche_confirmed (2/2 claim scenarios all-arms)
```

Receipt paths:

- `results/bayes/phase4-hybrid-smoke/manifest.json`
- `results/bayes/phase4-hybrid-smoke/summary.csv`
- `results/bayes/phase4-hybrid-smoke/regret.csv`
- `results/bayes/phase4-hybrid-smoke/replay-manifest.json`

Smoke gate rows:

| Scenario | Hybrid score / success | HC-Sundog score / success | Hybrid evals | Bayes-adaptive evals | Arms |
| --- | ---: | ---: | ---: | ---: | --- |
| `decoy` | 1.564062 / 1.0000 | -0.061427 / 0.1875 | 1260.25 | 3674.25 | repair true; frugality true; load-bearing true |
| `alias` | 0.810088 / 0.6250 | -0.355776 / 0.0625 | 1881.50 | 4615.00 | repair true; frugality true; load-bearing true |

Smoke interpretation: the harness now confirms the full Phase 4 niche gate in
the cheap run. `hybrid_no_posterior` does not repair either claim scenario, so
the hard-drop override does not fire. `hybrid_posterior_decoy_disambig` repairs
both claim scenarios, making the posterior load-bearing in the ablation arm.
The staged lock should run unchanged unless the smoke artifact itself is being
audited.

Lock staging:

```bash
npm run bayes:phase4
```

Estimate from the smoke rate: 3,072 trials at 1.11 trials/s is about 46
minutes on the current CPU lane, so the lock is operator-run under the repo's
long-run rule. The read-back artifact is
`results/bayes/phase4-hybrid-lock/manifest.json`.

Outcome branches:

- `hybrid_niche_confirmed`: promote Phase 4 from smoke to lock-backed niche
  claim.
- `hybrid_unnecessary`: drop Hybrid; the no-posterior response floor repaired
  a claim scenario.
- `hybrid_no_niche`: keep the smoke receipt but redesign or narrow Phase 4
  before claiming a hybrid niche.

### Phase 5 — Operating Envelope Sweep

Goal: map regions rather than narrate cherry-picked rounds.

Sweep axes:

- field noise;
- sensor count;
- model mismatch severity;
- decoy strength;
- action budget;
- compute budget;
- prior quality;
- field smoothness;
- target drift rate;
- delayed observation severity.

Outputs:

- `results/bayes/phase5-envelope/manifest.json`
- `trial-outcomes.csv`
- `envelope-map.csv`
- `aggregate-envelope.csv`
- `best-by-cell.csv`
- `cell-class-map.csv`
- `cell-delta-map.csv`
- `candidate-envelope.csv`
- heatmaps for Bayes-dominant, Sundog-dominant, Hybrid-dominant, all-fail;
- representative replays for each boundary class.

Exit criterion: the comparison can be summarized as an operating envelope with
failure classes, not as a winner-take-all benchmark.

### Phase 6 — Photometric Port

Goal: port the comparison back to the core mirror-alignment experiment.

Deliverables:

- Bayesian baseline for the photometric task, likely particle or grid posterior
  over laser / target / floor-hit geometry given detector readings.
- Direct comparison against the existing photometric scan/seek/track controller
  and analytic oracle.
- Known-model and wrong-model optical variants.
- At least one occlusion or detector-alias variant if feasible.

Exit criterion: a reviewer can see whether the Bayes-vs-Sundog result is only a
shadow-field toy artifact or relevant to the core photometric task.

### Phase 7 — Public Visualization and Motion Rail Card

Goal: make the comparison legible on `sundog.cc` without flattening it into
marketing.

Deliverables:

- `bayes.html` interactive visualization.
- Split-screen mode:
  - left: posterior heatmap, entropy, most likely hypothesis;
  - right: response trace, scan/seek/track phase, best observed signal.
- Round selector: Known Model, Warped Field, Decoy Field, Hybrid.
- Rail card content for the homepage / applications gallery.
- Short video clips for the highlight rail.

Suggested rail card:

```text
Title: Posterior vs. Halo
Status: Comparative Benchmark
Bayes: evidence -> posterior -> action.
Sundog: probe -> response -> control.
Question: must the agent infer the hidden world, or can it act from the trace?
Stamp sequence: MODEL KNOWN / FIELD READABLE / SIGNAL ALIASED / HYBRID NEEDED
```

Exit criterion: the public artifact says "complementary" more clearly than it
says "competition."

### Phase 8 — Writeup, Claim Ratchet, and Site Integration

Goal: update the research docs with the strongest claim actually earned.

Immediate work after the Pressure Mines applied lane:

- Backfill [`BAYESIAN_FLOOR_PROFILE_TEMPLATE.md`](BAYESIAN_FLOOR_PROFILE_TEMPLATE.md)
  with the fields that proved reusable in Balance and Mines: admission profile,
  same-observation seed policy, claim-gate fields, reported-only lanes,
  posterior-cell slices, and public interpretation copy. *(landed 2026-05-17)*
- Keep `alignment.html` as the public Bayes/Sundog side-by-side. The Mines row
  now has the explicit middle claim: stronger than "Sundog beat naive," weaker
  than "Sundog beat Bayes."
- Phase 1 exact Bayes-Correct reference task has landed: small hidden-source
  grid, correct likelihood, exact posterior, and Sundog response-control
  comparator. The 128-seed lock passes the known-model gate.
- Phase 2 mismatch slate has landed: five field variants, fixed clean
  `bayes_misspecified`, finite-mixture `bayes_adaptive`, `hc_sundog`, and
  `sundog_memory`. The lock finds one real separation (`anisotropic`) plus
  boundary cells, so the claim is complementary and narrow.
- Add `bayes.html` only after the standalone benchmark has real run artifacts.
  Do not publish a decorative Bayes page before Phase 1/2 receipts exist.
- Add a public rail card only if Round 3 or Phase 5 includes a real failure
  boundary.
- Update `docs/PROMO_HIGHLIGHTS.md` with a cautious comparison paragraph.
- Update `docs/SCIENTIFIC_CRITERIA.md` if the comparison introduces a stronger
  target-unaware baseline for the core photometric claim.

Claim ratchet candidates:

> *If Bayes-Correct wins Round 1 and Sundog wins at least one mis-specified
> round:* Bayes and Sundog are complementary. In the tested hidden-source
> family, posterior inference wins when the likelihood is right; response
> control remains useful when the model is wrong but the field is readable.
>
> *If aliased fields break Sundog:* Sundog requires the response trace to be
> meaningfully coupled to the hidden target. When the trace improves for the
> wrong reason, posterior or hybrid hypothesis management is required.
>
> *If Bayes dominates broadly:* Sundog should be framed as a lightweight
> controller for cases where maintaining a posterior is too expensive or
> unnecessary, not as a general partial-observation replacement.
>
> *If Hybrid dominates:* The next program direction should be hybrid: use
> Sundog for local response control and Bayes for ambiguity, reset, and
> hidden-cause disambiguation.

Exit criterion: docs, rail copy, and public language all agree on the same
boundary.

## Implementation Notes

### Suggested files

```text
sundog/
├── docs/
│   └── SUNDOG_V_BAYES.md
├── bayes.html
├── js/
│   ├── bayes-core.mjs
│   ├── bayes-controllers.mjs
│   ├── bayes-visualizer.mjs
│   └── bayes-replay.mjs
├── scripts/
│   └── bayes-phase1-reference.mjs
└── results/
    └── bayes/
        ├── phase1-reference/
        ├── phase2-mismatch/
        ├── phase3-aliasing/
        ├── phase4-hybrid/
        └── phase5-envelope/
```

### Suggested npm commands

```bash
npm run bayes:phase1
npm run bayes:phase2
npm run bayes:phase3
npm run bayes:phase4
npm run bayes:phase5
npm run bayes:replay -- --phase phase3-aliasing --seed 42
```

### Minimal result schema

Each trial row should include:

```text
run_id
seed
phase
scenario
controller_family
controller_variant
hidden_target_id
model_status              # correct / misspecified / adaptive / none
field_variant             # gaussian / warped / clipped / decoy / alias / drift
sensor_variant            # full-local / sparse-local / delayed / noisy
action_budget
compute_budget_ms
terminal_response
time_to_threshold
cumulative_response
wrong_lock
reacquired_after_drift
posterior_entropy_final
posterior_mass_true
sundog_best_observed
failure_label
```

## Public Copy Bank

Use:

- Bayes turns evidence into belief. Sundog turns response into control.
- The posterior draws a map. The halo moves the hand.
- When the model is right, Bayes should win.
- When the map lies but the response still answers, Sundog can hold.
- When the trace improves for the wrong reason, Sundog fails.
- The hybrid is not a compromise. It is the boundary made useful.

Avoid:

- Sundog beats Bayes.
- Bayes needs direct sight.
- Inference is obsolete.
- The halo is the posterior.
- The signal is always truthful.

## Open Questions

1. Phase 1 first substrate is resolved: discrete grid exact posterior first.
   Continuous particles wait for the later continuous-field extension.
2. Should Bayes-Adaptive use Gaussian processes, particles with hyperparameter
   learning, or a small mixture model? Recommendation: mixture first, GP second.
3. Should the public failure card be called **SIGNAL ALIASED**, **BOUNDARY
   FOUND**, or **HYBRID NEEDED**? Recommendation: use SIGNAL ALIASED as the
   Round 3 stamp and HYBRID NEEDED as the concluding stamp.
4. Does the Bayes comparison live as a core research benchmark or as a workbench
   application? Recommendation: core research benchmark until Phase 5, then
   operating-envelope workbench if the browser page ships.
5. Should the photometric paper be updated immediately to mention Bayesian
   baselines? Recommendation: not until Phase 6 produces a real photometric
   Bayes comparator.

## Closing Note

This roadmap should make Sundog smaller and stronger.

If Bayes wins, the program learns when explicit belief is required. If Sundog
wins, the program earns a cleaner statement about response control. If both win
in different cells, the program gets the best possible public posture: not a
universal claim, but a map.

The most important card is the failure card. Sundog should fail when the signal
is aliased, because that failure teaches the theorem's boundary: indirect
signal is only useful when the trace remains coupled to the thing that matters.
