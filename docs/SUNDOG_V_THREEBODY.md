The three-body problem is famously intractable in closed form — no general analytic solution exists. But the Sundog pattern isn't about solving the full dynamics. It's about asking: can an agent act usefully from an indirect, compressed signal instead of requiring the full state?
In three-body systems, the full state is 18-dimensional (three position vectors, three velocity vectors). But many practical questions about three-body dynamics don't require tracking the full state at every instant. They require detecting signatures of dynamically important events: near-collisions, ejections, resonance capture, exchange events, stability transitions. These events cast shadows in lower-dimensional observables long before they resolve in full phase space.
Where H(x) maps onto three-body dynamics
The Sundog move would be: instead of integrating the full 18D system and asking "what happens?", instrument the system with indirect observables and ask "what do those observables tell you about what's about to happen?"
Candidate indirect signals that are cheaper than full state:
The moment of inertia tensor (or its scalar trace) — a 1D or 6D compression of the 9D positional configuration. When a three-body system is about to undergo a close encounter or ejection, the eigenvalue spectrum of the inertia tensor changes character before the event resolves. The smallest eigenvalue dropping signals approach to a collinear configuration; the ratio of eigenvalues signals how "hierarchical" the system is at any moment.
Pairwise energy partition — for each of the three pairs, compute the two-body energy (kinetic + potential of that pair in their center-of-mass frame). In a stable hierarchical triple, one pair's energy is deeply negative (the inner binary) and one is weakly negative (the outer orbit). When the system transitions from hierarchical to democratic (the chaotic scattering regime), these energies approach each other. That crossing is an indirect signal for instability.
Angular momentum exchange rates — the time derivative of the angular momentum of the inner binary. In a stable system this oscillates secularly (Kozai-Lidov). When the oscillation amplitude grows past a threshold, it signals the system is approaching a regime where the hierarchy will break. You don't need the full state to detect this; you need ∂L_inner/∂t, which is a scalar.
The virial ratio (2T/|W|) — kinetic energy over potential energy. For a bound system this oscillates around 1. When it persistently exceeds 1, the system is unbound or about to eject. This is a single scalar that compresses all 18 dimensions into a stability diagnostic.
What a Sundog controller for three-body would look like
Imagine a scenario where you're controlling one body (say, a spacecraft in a three-body gravitational environment) and you can observe some but not all of the system state. The Sundog architecture maps directly:
SCAN: Probe the dynamical landscape by executing small maneuvers and observing how the indirect signals (virial ratio, pairwise energies, angular momentum exchange) respond. This is the three-body analogue of the Lissajous scan in the photometric experiment.
SEEK: Identify the region of control space where the indirect signals indicate favorable dynamics — approaching a stable resonance, avoiding ejection, capturing into a desired orbit family.
TRACK: Use extremum-seeking on the indirect signal to maintain the desired dynamical regime without requiring full state estimation of all three bodies.
The analogue of "tight joint limits" as a failure boundary would be: when the system enters the strongly chaotic scattering regime (all three bodies at comparable mutual distances, no hierarchy), the indirect signals become noisy and the coupling between your control action and the observable weakens. The Sundog controller would lose its gradient signal in the same way the photometric controller loses its signal when the optimum is outside the reachable workspace.
What's honest vs. what's reach
Honest: the indirect-signal-to-action pattern has a structural analogue in three-body dynamics. The observables listed above are well-known in celestial mechanics but typically used for post-hoc analysis, not for online control. Reframing them as control-relevant compressed signals is a genuine contribution if it leads to a controller that works.
Reach: claiming Sundog "solves" three-body dynamics. It doesn't. The chaotic sensitivity is real and no compression scheme eliminates it. What the pattern could do is identify a practical operating envelope where indirect signals are sufficient for useful action — the same trade the photometric experiment documents: comparable terminal performance, slower convergence, known failure boundary.
A concrete first experiment
The minimum viable test: a restricted three-body problem (circular restricted, so you have two primaries and one test particle). The test particle has no access to the primaries' exact masses or positions — only to tidal acceleration (the gradient of the gravitational field at its location) and its own velocity. Can a scan-seek-track controller find and maintain a halo orbit around a Lagrange point using only the tidal signal? The oracle baseline would be a controller with full knowledge of the primaries.
That's small enough to defend and connects directly to the photometric experiment's structure.
Where do you want to push on this?

## Ratification - 2026-05-07

Status: ratified first as a promising research direction, and now supported by
a bounded operating-envelope result in the planar restricted prototype.

The hook is strong because the three-body problem is a familiar symbol of
dynamical complexity, and Sundog has a clean way to enter the conversation:
not by claiming to solve the dynamics, but by showing that lower-dimensional
event signatures can support useful action inside a bounded operating envelope,
while naming where that envelope fails.

The actionable core is not "Sundog solves three-body." The actionable core is:

> In selected three-body regimes, can an agent use compressed, indirect
> dynamical signatures to detect instability, choose interventions, or maintain
> a desired orbit family with less privileged state than an oracle controller?

That is worth pursuing. It maps cleanly onto the existing Sundog pattern:

1. deny full-state access;
2. expose only indirect observables;
3. transform those observables into control-relevant signatures;
4. compare against a privileged baseline;
5. report the failure boundary.

## Actionability Audit

The current proposal has one important caveat: several candidate observables
are compressed, but not automatically indirect. Pairwise energies, inertia
tensors, virial ratio, and angular-momentum exchange can require positions,
velocities, masses, or pair identity. If those are computed from full simulator
state, they are diagnostics, not Sundog-style partial observations.

So the first work item is a sensor-model audit:

- Mark each observable as privileged, partially observable, or locally
  measurable.
- Separate "compressed full-state diagnostic" from "available indirect signal."
- Define exactly what the controlled body can sense: local acceleration, tidal
  tensor estimate, range/range-rate, Doppler shift, bearing-only observations,
  own thrust history, own velocity, or noisy local field samples.
- Preserve an oracle baseline with full CR3BP state for comparison.

This keeps the claim defensible and may make the experiment more interesting.
If a compressed diagnostic works only with privileged state, it can still guide
feature discovery, but it should not be presented as the Sundog controller's
input.

## Ratified Hook Language

Safe hook:

> Sundog does not solve the three-body problem. It asks a smaller, sharper
> question: can an agent act usefully in three-body dynamics by reading the
> signatures around instability instead of reconstructing the full state?

Short version:

> Three-body dynamics are chaotic. Sundog asks whether the shadow of that chaos
> is enough to steer by.

Avoid:

- "Sundog solves the three-body problem."
- "The controller predicts chaotic dynamics."
- "Compressed observables replace integration."
- "This proves general control under chaos."

## Roadmap

### Phase 0 - Scope and Literature Pass

Goal: define the exact claim before writing the simulation.

Deliverables:

- A one-page claim boundary for the three-body study.
- A table of candidate observables with required state access.
- A chosen toy regime: start with planar circular restricted three-body problem
  before attempting full spatial CR3BP or general three-body scattering.
- Baselines: privileged oracle, naive local controller, and passive/no-control
  reference.

Exit criterion: we know which signals are legitimately partial and which are
only full-state diagnostics.

### Phase 1 - Diagnostic Benchmark

Goal: test whether compressed signals forecast useful events at all.

Use full simulator state offline to compute candidate signatures, then measure
whether they predict:

- ejection or escape;
- close approach;
- hierarchy break;
- transition into chaotic scattering;
- departure from a desired orbit family.

Metrics:

- lead time before event;
- precision/recall or AUROC for event detection;
- robustness under noise;
- failure regimes where signals become ambiguous.

Exit criterion: at least one compressed signature gives nontrivial warning or
classification performance against a simple baseline.

### Phase 2 - Sensor-Available Surrogate

Goal: replace privileged diagnostics with locally measurable proxies.

Candidate local signals:

- tidal tensor estimate from small probe maneuvers or local acceleration
  differences;
- own acceleration residual after subtracting commanded thrust;
- short-horizon response curve to small thrust pulses;
- bearing-only or range-rate-only measurements if the scenario allows a weak
  exteroceptive sensor.

Exit criterion: a proxy signal preserves enough of the Phase 1 diagnostic value
to justify control.

### Phase 3 - Scan/Seek/Track Controller

Goal: build the Sundog controller only after the signal earns it.

SCAN: perturb with small maneuvers and estimate local response of the indirect
signal.

SEEK: move toward regions where the signature indicates stability, capture, or
lower escape risk.

TRACK: maintain the desired signature using extremum seeking or receding-horizon
updates without privileged full-state feedback.

Metrics:

- time maintained near target orbit family;
- fuel or delta-v cost;
- recovery after perturbation;
- comparison to oracle and naive baselines;
- known failure boundary.

Exit criterion: the controller succeeds in a bounded operating envelope and
loses cleanly where the signal-action coupling collapses.

### Phase 3.5 - Real-time Web Projection

Goal: port the restricted three-body diagnostic to an interactive browser
visualization before finalizing the public artifact.

Deliverables:

- A standalone HTML page (`threebody.html`) implementing the restricted
  three-body problem in JavaScript with real-time integration.
- Canvas rendering showing:
  - the simulated three-body system (two primaries and one test particle);
  - shadow trails visualizing orbital history;
  - indirect signature overlays (virial ratio, inertia tensor eigenvalues,
    pairwise energies) as time-series graphs;
  - control overlays showing scan/seek/track phases if Phase 3 work is ready.
- Visual design consistent with the site hero graph (parhelion-canvas style):
  gold accents, dark gradient backgrounds, clean typography, responsive layout.
- UI controls for interactive exploration:
  - adjust initial conditions (positions, velocities);
  - modify system parameters (mass ratios, circular restricted vs full 3-body);
  - toggle signature visibility;
  - apply scan/seek/track control moves manually if Phase 3 controller exists,
    or simulate passive observation if not.
- The page serves as the application landing page for the three-body experiment
  and directly supports Phase 4's public demonstration goal.

Exit criterion: a working browser experiment demonstrates that indirect
signatures respond coherently to changes in system state, providing a visual
benchmark for the coupling claim before the full controller is built.

Rationale: A public, visual artifact aligns with the Phase 4 objective and
establishes a concrete demonstration environment early. The browser experiment
can evolve from diagnostic (Phase 1/2 work) to interactive controller
(Phase 3 integration) to final public artifact (Phase 4 refinement) without
requiring a separate build for each stage.

### Phase 4 - Public Artifact

Goal: turn the result into a strong, honest Sundog hook.

Deliverables:

- a small interactive visualization showing full orbit, hidden state, exposed
  indirect signal, and controller action;
- a short writeup titled around "steering by the shadow of chaos";
- explicit claim boundary using the same discipline as the photometric
  experiment;
- a failure-boundary panel, not a footnote.

## Validation Roadmap

The browser prototype is useful, but it is not yet a completed experiment. The
next phases turn the artifact into evidence by adding baseline comparisons,
event labels, quantitative metrics, and a mapped operating envelope.

### Phase 5 - Reproducible Experiment Harness

Goal: move from manual browser exploration to repeatable trial batches.

Deliverables:

- A deterministic simulation harness that can run the same dynamics outside the
  animation loop.
- Seeded initial-condition generation for stable, near-escape, near-collision,
  and chaotic-scattering regimes.
- A trial manifest format recording mass ratio, initial state, controller mode,
  thrust authority, target tidal magnitude, timestep, duration, and seed.
- Per-trial logs for state, exposed signals, controller actions, event labels,
  and terminal outcome.

Exit criterion: a trial can be replayed exactly, and the browser demo can point
to the same equations as the batch harness.

### Phase 6 - Baseline Set

Goal: establish what the Sundog controller is being compared against.

Required baselines:

- **Passive baseline:** no thrust; measures natural survival, escape, and close
  approach rates.
- **Naive local baseline:** uses only simple local acceleration magnitude or
  radial heuristics, without tidal-gradient structure.
- **Privileged oracle:** uses full simulator state, primary positions, and a
  chosen target objective.
- **Ablation controllers:** SCAN-only, SEEK-only, TRACK-only, and tidal-signal
  shuffled/noised variants.

Metrics:

- time before escape or close approach;
- time inside desired tidal band;
- fuel or delta-v cost;
- number of controller saturations;
- terminal distance/energy class;
- recovery after injected perturbations.

Exit criterion: every claimed improvement is stated relative to at least one
baseline, and every baseline uses the same initial-condition slate.

### Phase 7 - Event Labels and Diagnostic Metrics

Goal: test whether compressed signatures actually forecast useful events.

Event labels:

- escape/ejection;
- close approach;
- tidal spike;
- controller saturation;
- loss of target tidal band;
- numerical instability or invalid trajectory.

Diagnostic metrics:

- lead time between warning threshold and event;
- precision, recall, F1, and AUROC for event detection;
- false alarm rate per unit simulated time;
- robustness under sensor noise;
- calibration curves for warning thresholds.

Exit criterion: at least one privileged diagnostic and one sensor-motivated
proxy show quantified predictive value over a naive threshold baseline.

### Phase 8 - Sensor-Model Validation

Goal: decide how much of the current "sensor-limited" mode is genuinely
available to the controlled body.

Sensor variants:

- **Simulated local probe:** current browser implementation; nearby acceleration
  samples are computed directly from the field model.
- **Micro-maneuver estimate:** the controller applies small thrust probes and
  estimates the response from its own acceleration history.
- **Accelerometer-array estimate:** multiple local samples are modeled as if
  measured by a short-baseline physical sensor.
- **Noisy learned surrogate:** a learned or fitted local field model receives
  only prior local observations.

Audit questions:

- Does the controller need hidden primary positions to compute the signal?
- How much delay and noise can the proxy tolerate?
- Does probe thrust contaminate the signal enough to change the control result?
- Which sensor assumptions are plausible for spacecraft, simulation tooling, or
  pure software agents?

Exit criterion: the writeup labels each result by sensor tier, and no result
computed from full state is described as sensor-only.

### Phase 9 - Operating Envelope and Failure Map

Goal: replace anecdotal failure boundaries with a map.

Current command:

```bash
npm run threebody:phase9
npm run threebody:phase9:refine
npm run threebody:phase9:lock
npm run threebody:phase9:axes
npm run threebody:phase9:hazard
npm run threebody:phase9:hazard:refine
```

Sweep axes:

- initial position and velocity of the test particle;
- mass ratio of the primaries;
- thrust authority;
- target tidal magnitude;
- sensor noise and delay;
- integration timestep;
- perturbation impulse size.

Outputs:

- `results/threebody/phase9-operating-envelope/manifest.json`;
- `trial-outcomes.csv`: primary cold-open data artifact. One row per
  non-passive controller trial with seed, regime, initial-condition scales,
  thrust, sensor noise, guard thresholds, terminal outcome, simulated time,
  delta-v, minimum primary distance, matched passive outcome/time, outcome
  effect, and failure mechanism.
- `paired.csv`: legacy/internal per-trial controller rows paired against
  matched passive baselines. Downstream aggregates should be derivable from
  `trial-outcomes.csv`;
- `envelope-map.csv`: one row per regime, initial-condition scale, thrust,
  sensor-noise, and guard setting, including survival deltas, passive survival
  time, controller effort, sensor-error summaries, and dominant failure
  mechanism;
- `aggregate-envelope.csv`: same map aggregated across regimes;
- `best-by-cell.csv`: best setting per regime/radius/velocity cell;
- `cell-class-map.csv`: map-shaped view of best region class by radius and
  velocity;
- `cell-delta-map.csv`: map-shaped view of best survival delta by radius and
  velocity;
- `candidate-envelope.csv`: positive survival-delta candidates that pass the
  current worsened-rate filter;
- success/failure heatmaps;
- escape and close-approach regions;
- controller saturation regions;
- cases where tidal proxies are misleading;
- representative trajectories for wins, losses, and ambiguous regimes.

First Phase 9 smoke map:

- Default run emitted 1,944 trials over stable, near-escape, and chaotic
  regimes, three radius scales, three velocity scales, two thrust limits, three
  accelerometer-noise levels, and two guard-acceleration thresholds.
- Outcome counts: 972 bounded, 879 close approach, 93 escape.
- Positive candidate rows: 29 of 324 `envelope-map.csv` rows after tightening
  the candidate definition to require positive survival delta.
- Best-cell summary: 4 promising cells, 21 neutral cells, and 2 risky cells.
- Interpretation: the current positive pocket is near-escape, especially around
  nominal/slightly-larger radius scales and low-to-moderate velocity scaling.
  Stable cells are mostly neutral because passive already survives; chaotic
  cells are mostly neutral or risky. This is the right shape for an operating
  envelope, but it is not yet a public performance claim.

Refined near-escape map:

- `npm run threebody:phase9:refine` emits a denser 7x7 near-escape radius/velocity
  map over radius scales `0.925..1.075`, velocity scales `0.85..1.15`, thrust
  limits `0.4/0.5`, sensor noise `0/0.01`, guard acceleration `2.5`, and 6
  seeds.
- Default refined run emitted 2,352 trials.
- Positive candidate rows: 116 of 196 `envelope-map.csv` rows.
- Best-cell summary: 32 promising cells, 6 neutral, 4 mixed, 3 risky, and 4
  negative.
- The map is no longer just anecdotal: the strongest connected positive region
  is the upper-right of the near-escape slice, roughly radius scale `>= 0.975`
  and velocity scale `>= 0.95`, with especially strong deltas above velocity
  scale `1.05`. Larger-radius but low-velocity cells are negative/risky, which
  gives the first crisp local failure boundary.
- Mechanistic labels now separate `control_effort_or_saturation` from
  `controller_destabilized_or_shortened_passive`. In the refined map, the
  negative/risky rows are mostly the second mechanism: passive trajectories
  often last longer or survive, while guarded TRACK shortens or destabilizes
  them. A smaller subset shows high controller effort/saturation. No refined
  failure rows currently point primarily to the accelerometer sensor-noise floor.

Locked near-escape map:

- `npm run threebody:phase9:lock` repeats the refined 7x7 near-escape map with
  24 seeds, producing 9,408 trials under
  `results/threebody/phase9-locked-near-escape/`.
- Positive candidate rows: 132 of 196 `envelope-map.csv` rows.
- Best-cell summary: 41 promising cells, 7 mixed cells, and 1 negative cell.
- The connected positive pocket survives the larger seed slate. The strongest
  rows are still at larger radius and higher velocity: radius scale `1.05..1.075`
  and velocity scale `1.1..1.15` often reach survival deltas of `0.79..0.92`
  versus matched passive baselines.
- Failure mechanisms remain honest: harmful trial rows split into 585
  `controller_destabilized_or_shortened_passive` and 359
  `control_effort_or_saturation`. This says the remaining boundary is mostly
  controller interaction with trajectories passive already handles, not a
  detected sensor-noise-floor failure.

Mass-ratio and timestep probe:

- `npm run threebody:phase9:axes` adds the first scoped mass-ratio/timestep
  check inside the locked high-velocity near-escape pocket. It sweeps mass
  ratios `0.01`, `0.3`, and `1`, timesteps `0.008`, `0.01`, and `0.012`, radius
  scales `1.025..1.075`, velocity scales `1.05..1.15`, thrust `0.4`, sensor
  noise `0`, and 8 seeds.
- Default axes run emitted 1,296 trials under
  `results/threebody/phase9-mass-timestep/`.
- Positive candidate rows: 72 of 81 `envelope-map.csv` rows.
- Best-cell summary by mass/timestep: every equal-mass cell is promising; mass
  ratio `0.3` has 8 promising and 1 mixed cell per timestep; mass ratio `0.01`
  has 7 promising and 2 mixed cells per timestep.
- Interpretation: within this already-favorable high-velocity near-escape
  pocket, the effect is not obviously an equal-mass resonance and is not
  sensitive to the small timestep band tested. The lower mass-ratio maps often
  show larger mean survival deltas because passive escapes more often there.
  This does not yet prove robustness outside the locked pocket.

Hazard-derived guard gates:

- `npm run threebody:phase9:hazard` replaces the hand-tuned guard constants in
  the same mass-ratio/timestep slice with thresholds derived from passive
  hazard-score samples. For each case/regime, the passive trials provide
  non-hazard samples; the controller then uses the configured quantile
  (`--track-guard-quantile 0.75`) for local acceleration and tidal magnitude
  guard thresholds, plus the complementary radius quantile for the radius gate.
- Default hazard-gate run emitted 1,296 trials under
  `results/threebody/phase9-hazard-gates/`.
- Positive candidate rows: 81 of 81 `envelope-map.csv` rows.
- Outcome effects: 522 helped, 36 time-helped, 90 tied, and no hurt/time-hurt
  rows in the scoped run.
- Failure mechanisms: no harmful rows were labeled in the hazard-gate run.
- Delta-v check: average controller delta-v fell from about `2.90` in the
  constant-gate mass/timestep probe to about `1.74` in the hazard-gate probe.
- Interpretation: inside the locked high-velocity near-escape pocket, passive
  hazard-score gates improve the controller story: they preserve the pocket,
  remove the harmful rows seen with hand-tuned constants, and reduce control
  effort. This is still scoped to the favorable pocket and should next be tested
  against the full refined near-escape grid.

Full refined hazard-gate check:

- `npm run threebody:phase9:hazard:refine` applies the same passive
  quantile-derived guard gates to the full 7x7 refined near-escape grid.
- Default run emitted 2,352 trials under
  `results/threebody/phase9-hazard-refined-near-escape/`.
- Positive candidate rows: 100 of 196 `envelope-map.csv` rows, versus 116 of
  196 in the constant-gate refined run.
- Best-cell summary: 27 promising, 8 neutral, 7 mixed, 5 risky, and 2 negative.
- Delta-v check: average controller delta-v fell from about `2.07` in the
  constant-gate refined run to about `1.12` in the hazard-gate refined run.
- Interpretation: hazard-derived gates are not simply better everywhere. They
  preserve the high-velocity positive pocket and sharply reduce control effort,
  but they shrink the wider candidate region and still leave low-velocity
  harms. This is a stronger, more honest controller: less tuned and less
  aggressive, but still bounded by the same broad operating-envelope geometry.

Exit criterion: the public page can show where the method works, where it
fails, and where the result is inconclusive.

### Phase 10 - Writeup and Claim Ratchet

Goal: update the public-facing story so it reflects the Phase 9 result without
turning a bounded operating-envelope pocket into a general control claim.

Immediate work:

- Update `docs/threebody-writeup.md` so the headline status no longer says only
  "promising enough to study." The earned claim is now that a guarded
  accelerometer-proxy TRACK controller shows a reproducible positive pocket in
  mapped near-escape regimes, with explicit low-velocity failure boundaries.
- Update `threebody.html` copy and the applications gallery entry to make the
  current result inspectable from the public site.
- Add a short "What changed after Phase 9" panel: initial smoke pocket,
  refined near-escape pocket, locked 24-seed result, mass-ratio/timestep slice,
  and hazard-gate refinement.
- Promote the failure map to first-class narrative material. The low-velocity
  harm boundary and controller-shortened-passive rows are part of the result,
  not a footnote.
- Keep the sensor-tier wording strict: the current best controller is
  accelerometer-proxy guarded TRACK in simulation, not a validated physical
  spacecraft sensor stack.

Current safe claim:

> The three-body page is an interactive prototype showing indirect signatures,
> sensor-model separation, and a scan/seek/track controller scaffold.

Claim after Phase 7:

> In the tested planar restricted setup, selected compressed signatures forecast
> escape or close-approach events with quantified lead time.

Claim after Phase 8:

> In the tested setup, a specified sensor-tier proxy preserves enough diagnostic
> signal to support local control experiments.

Claim after Phase 9:

> In a mapped operating envelope, the Sundog-style controller outperforms
> passive and naive local baselines on specified metrics, while losing to or
> approaching a privileged oracle depending on regime.

Current Phase 9 earned wording:

> In the tested planar restricted setup, a guarded accelerometer-proxy TRACK
> controller improves survival over passive behavior in a connected
> near-escape operating pocket. The effect is not global: low-velocity
> near-escape cells still show harms, and hazard-derived gates trade a smaller
> candidate region for lower control effort.

Current Phase 13 earned wording:

> In the tested planar restricted setup, a guarded accelerometer-proxy TRACK
> controller improves survival over passive and naive local baselines across a
> mapped high-velocity near-escape pocket through a 16-second tested horizon.
> The result is not global: the low-velocity boundary, especially equal-mass
> cells near `velocityScale=0.95`, still exposes controller harms.

Do not claim:

- "Sundog solves three-body dynamics."
- "The controller predicts chaos."
- "Sensor-only control is validated" before Phase 8 passes.
- "Comparable utility from partial information" before baseline metrics show
  the trade.

Exit criterion: the writeup, page copy, and promo language all use the strongest
claim actually earned by the completed validation phase, and no stronger one.

### Phase 11 - Robustness and Outside-Pocket Expansion

Goal: find out whether the Phase 9 pocket is a real operating-envelope feature
or a narrow artifact of one guard quantile, near-escape slice, and controller
parameterization.

Primary checks:

- Guard-quantile sweep: compare passive hazard gates at quantiles `0.5`,
  `0.75`, and `0.9` on the refined near-escape grid.
- Outside-pocket expansion: rerun mass-ratio/timestep probes beyond the locked
  high-velocity pocket, especially lower velocity scales and larger-radius
  low-velocity cells where the current controller harms passive outcomes.
- Diagnostic-to-control chain: summarize Phase 7/8 metrics alongside Phase 9
  outcomes so the public claim reads as a sequence: hazard signatures forecast
  events, sensor proxies preserve usable signal, controller improves outcomes
  only where signal-action coupling remains intact.
- Failure mechanism audit: separate controller-destabilized cases from pure
  control-effort/saturation cases and check whether hazard gates reduce one
  class while worsening another.
- Naive/oracle comparison pass: rerun the best Phase 11 pocket against passive,
  naive local acceleration, guarded accelerometer TRACK, and the privileged
  heuristic oracle on the same slate.

Suggested commands/scripts:

```bash
npm run threebody:phase11:guard-quantiles
npm run threebody:phase11:outside-pocket
npm run threebody:phase11:compare
```

These commands wrap `scripts/threebody-operating-envelope.mjs` with explicit
output directories and manifest labels, rather than replacing the Phase 9
artifacts. The runner supports `--track-guard-quantiles` as a sweep axis for
Phase 11 guard comparisons.

Outputs:

- `results/threebody/phase11-guard-quantiles/`;
- `results/threebody/phase11-outside-pocket/`;
- `results/threebody/phase11-comparison/`;
- `docs/THREEBODY_PHASE11_SUMMARY.md`;
- updated failure-boundary heatmaps for velocity/radius, mass ratio, timestep,
  and guard quantile.

Exit criterion: the project can say whether the Phase 9 positive pocket is
robust to guard quantile and nearby parameter expansion, and can identify the
first outside-pocket region where the Sundog controller should not be used.

Initial Phase 11 results:

- Guard-quantile sweep emitted 7,056 trials under
  `results/threebody/phase11-guard-quantiles/`. The pocket survives quantiles
  `0.5`, `0.75`, and `0.9`; `0.5` is cheapest, `0.9` has the strongest average
  survival delta, and `0.75` remains a viable middle setting.
- Outside-pocket expansion emitted 6,912 trials under
  `results/threebody/phase11-outside-pocket/`. The high-velocity pocket remains
  strong, while lower velocity scales and equal-mass boundary cells carry most
  of the mixed and negative rows.
- Comparison run emitted 2,592 trials under
  `results/threebody/phase11-comparison/`. In the favorable high-velocity
  pocket, guarded accelerometer TRACK produced 81 candidate rows out of 81,
  while the naive local baseline produced none and the privileged heuristic
  oracle produced 34 of 81. This oracle is a heuristic, not an optimal
  controller.
- Summary: `docs/THREEBODY_PHASE11_SUMMARY.md`.

Phase 11 conclusion: the Phase 9 pocket is no longer just a tuned positive
slice. It survives guard-quantile variation and matched baseline comparison in
the favorable high-velocity near-escape region. The same runs also make the
failure boundary sharper: lower velocities and equal-mass boundary cells are the
first places where the controller should not be used.

### Phase 13 - Longer-Horizon Lock

Goal: test whether the Phase 11 operating pocket represents durable control
over a longer rollout, or whether guarded TRACK mainly delays the same failure
within the original 8-second window.

Implementation-grade spec and result note:

- [`docs/threebody/PHASE13_SPEC.md`](threebody/PHASE13_SPEC.md)
- [`docs/threebody/PHASE13_RESULTS.md`](threebody/PHASE13_RESULTS.md)

Commands:

```bash
npm run threebody:phase13:smoke
npm run threebody:phase13
```

Exit criterion: the project can say whether the Phase 11 positive pocket
survives horizon extension, and can distinguish durable survival improvement
from short-window delay.

Phase 13 result:

- `npm run threebody:phase13:smoke` emitted 32 trials under
  `results/threebody/phase13-long-horizon-smoke/` at `duration=16`.
- `npm run threebody:phase13` emitted 3,456 trials under
  `results/threebody/phase13-long-horizon-lock/` at `duration=16`.
- The full lock found 88 candidate envelope rows out of 324: guarded TRACK 77,
  heuristic oracle 11, naive local 0.
- Best-cell class balance: 81 promising, 17 mixed, 5 risky, and 5 negative.
  Guarded TRACK is the best controller in 100 of 108 cells; the heuristic oracle
  is best in 8.
- The high-velocity pocket survives horizon extension. `velocityScale=1.05`
  and `1.15` are 27 / 27 promising; `velocityScale=1.1` is 19 / 27 promising
  and 8 / 27 mixed, with no risky or negative cells.
- The boundary remains visible. All risky and negative best cells sit at
  `velocityScale=0.95`; all 5 negative cells are equal-mass cases.
- Cost rate does not materially explode: guarded TRACK candidate rows use mean
  delta-v `3.665` over 16 seconds (`0.229` per second), compared with Phase 11's
  `1.741` over 8 seconds (`0.218` per second).
- Summary: [`docs/threebody/PHASE13_RESULTS.md`](threebody/PHASE13_RESULTS.md).

### Phase 14 - Mechanism Decomposition and Action Coupling

Goal: decompose the Phase 13 guarded-TRACK win into warning quality, action
coupling, and outcome effect, to test whether the accelerometer/tidal signal is
the operative causal handle or whether the frozen guard mostly suppresses bad
thrust in the tested pocket.

Implementation-grade spec and result note:

- [`docs/threebody/PHASE14_SPEC.md`](threebody/PHASE14_SPEC.md)
- [`docs/threebody/PHASE14_RESULTS.md`](threebody/PHASE14_RESULTS.md)

Commands:

```bash
npm run threebody:phase14:smoke
npm run threebody:phase14
```

Exit criterion: the project can say whether the Phase 13 survival benefit
depends on intact signal-directed action, or whether it survives signal and
action ablation and is therefore a guard-suppression artifact. Pre-registered
negative branch: high warning quality with weak action coupling does not support
a controller claim.

Phase 14 result:

- `npm run threebody:phase14` emitted 6,048 trials under
  `results/threebody/phase14-mechanism-decomposition-lock/`.
- Candidate envelope rows: 130 / 648. Guarded TRACK contributes 77, signal
  delay 48, action shuffle 3, signal shuffle 2, naive 0, and sign flip 0.
- The spec's exact `npm run threebody:phase13` regression gate was rerun this
  session and reproduced Phase 13 bit-for-bit (3,456 trials; 88 / 324 candidate
  envelope rows; 81 promising best cells; outcomes 1,154 bounded / 2,030 escape
  / 272 close approach), so the shared-harness edit is verified non-perturbing.
- Pre-registered branch: provisional partial / mechanism narrowed (the
  "provisional" qualifier now rests only on the science below, not on a gate
  caveat). The guard-only explanation is weakened because
  action shuffle and signal shuffle lose almost all candidate rows and sign
  flip destroys the pocket. The clean causal-handle pass is not earned because
  passive tidal AUROC fails the warning-quality bar and a 0.5-second signal
  delay retains 48 candidate rows.
- Summary: [`docs/threebody/PHASE14_RESULTS.md`](threebody/PHASE14_RESULTS.md).

### Phase 15 - Forward-Oracle / Precision Lock

Goal: test whether the Phase 13/14 high-velocity pocket survives stricter
numerical and privileged forward checks, and resolve the Phase 14 mechanism
question with a timing-sensitive per-step counterfactual and a re-grounded
warning-quality readout.

Implementation-grade spec and result note:

- [`docs/threebody/PHASE15_SPEC.md`](threebody/PHASE15_SPEC.md)
- [`docs/threebody/PHASE15_RESULTS.md`](threebody/PHASE15_RESULTS.md)

Commands:

```bash
npm run threebody:phase15:smoke
npm run threebody:phase15
```

Exit criterion: the project can say whether the pocket survives finer timesteps
and a privileged forward oracle, and whether the per-step counterfactual cleanly
separates intact signal-directed control from shuffled/mistimed control where
the Phase 14 agreement metric failed. Pre-registered negative: a positive
outcome envelope without privileged one-step counterfactual hazard improvement
does not support a causal-control claim.

Status: complete as of 2026-05-27. Formal branch: **Fail-Magnitude**. The
guarded TRACK survival envelope is stable across `dt=0.004-0.012`, the
Richardson precision receipt passes cleanly (`p=4.313`), and ablations collapse
or invert in the favorable pocket. Phase 15 does **not** pass the mechanism
upgrade because the privileged one-step counterfactual is positive but below
the pinned `+0.20` magnitude bar and oracle-hazard AUROC is decidably below
`0.70`.

### Phase 15B - Counterfactual Normalizer Audit

Goal: diagnose whether Phase 15's one-step counterfactual magnitude miss was
partly a measurement artifact from the `1e-9` denominator floor in cells where
the privileged oracle and no-op one-step states nearly coincide.

Implementation-grade spec and result note:

- [`docs/threebody/PHASE15B_SPEC.md`](threebody/PHASE15B_SPEC.md)
- [`docs/threebody/PHASE15B_RESULTS.md`](threebody/PHASE15B_RESULTS.md)

Commands:

```bash
npm run threebody:phase15b:normalizer-smoke
npm run threebody:phase15b:normalizer
```

Exit criterion: the project can say whether denominator-floor collapse is a
material explanation for the Phase 15 Fail-Magnitude read, including in
non-candidate cells. This phase is diagnostic only: it does not revise the
Phase 15 verdict, retune the controller, or upgrade the earned claim. A positive
result points to a separately locked multi-step counterfactual; a negative
result shifts attention to horizon locality or the hazard score itself.

Status: complete as of 2026-05-28. Formal branch: **Mixed / Partial
Diagnostic**. The 1,728-trial lock confirmed that denominator-floor collapse is
near-universal in TRACK rows (`0.971-1.000` floor-hit rate), but floor-hit steps
carry no TRACK-specific hidden positive one-step signal: raw effect is
noise-level, TRACK floor-hit positive rate is near chance, and shuffled arms can
score higher on floor-hit positive rate while failing to produce candidate
cells. Phase 15B therefore does not explain the Phase 15 magnitude miss; it
points to multi-step trajectory steering as the next diagnostic.

### Phase 15C - Multi-Step Counterfactual Horizon Audit

Goal: test whether Phase 15's favorable guarded-TRACK pocket is explained by
cumulative trajectory steering over short horizons rather than by one-step
energy reduction.

Implementation-grade spec:

- [`docs/threebody/PHASE15C_SPEC.md`](threebody/PHASE15C_SPEC.md)

Reserved commands, not runnable until the implementation commit adds the
multi-step audit flag and npm scripts:

```bash
npm run threebody:phase15c:multistep-smoke
npm run threebody:phase15c:multistep
```

Exit criterion: the project can say whether cumulative counterfactual horizon
scores at `N in {4,8,16,32}` explain the survival pocket that one-step energy
reduction did not. This phase is diagnostic only: it does not revise Phase 15,
retune the controller, or upgrade the earned claim.

Status: spec filed 2026-05-28. Implementation is held until an additive runner
commit pins the CSV columns, smoke command, capped rate probe, and lock
readback path.

### Phase 16 - Hazard-Score Channel Audit

Goal: test whether Phase 15's missed oracle-hazard warning-quality bar was a
channel mismatch: the frozen strict-oracle label is geometric
(`r3 > 4` or `minPrimaryDistance < 0.08`), while Phase 15 scored only
instantaneous energy.

Implementation-grade spec and result note:

- [`docs/threebody/PHASE16_SPEC.md`](threebody/PHASE16_SPEC.md)
- [`docs/threebody/PHASE16_RESULTS.md`](threebody/PHASE16_RESULTS.md)

Commands:

```bash
npm run threebody:phase16:hazard-smoke
npm run threebody:phase16:hazard
npm run threebody:phase16:analyze
```

Exit criterion: the project can say whether any locked instantaneous channel
or fixed held-out combination predicts the strict oracle's 32-step
`hazardReached` label with lower-95%-CI AUROC at or above `0.70`. This is a
diagnostic warning-channel audit only; all branches preserve the Phase 15 formal
verdict and do not retune the controller.

Status: completed 2026-05-29 as Branch A, hazard warnable. The 288-passive-trial
lock found that energy missed the lower-CI bar while `radius` passed strongly
(`0.995`, 95% CI `[0.985, 1.000]`). The result repairs the warning-instrument
story but preserves the Phase 15 Fail-Magnitude mechanism verdict.

### Phase 16B - Radius Warning Re-Pose

Goal: test whether Phase 15's warning-quality verdict flips when the warning
score is `radius` instead of energy, using the completed Phase 16 passive lock
receipt and a Phase-15-style per-cell AUROC mean.

Implementation-grade spec and result note:

- [`docs/threebody/PHASE16B_SPEC.md`](threebody/PHASE16B_SPEC.md)
- [`docs/threebody/PHASE16B_RESULTS.md`](threebody/PHASE16B_RESULTS.md)

Command:

```bash
npm run threebody:phase16b:repose
```

Exit criterion: the project can say whether the warning-quality miss was purely
an energy-instrument artifact under the Phase-15-style warning bar: favorable
mean AUROC at least `0.70` with at least `18/27` favorable cells defined. This
phase is offline only and does not retune the controller or revise the Phase 15
mechanism verdict.

Status: completed 2026-05-29 as Branch A. The reducer reproduced Phase 16's
per-cell values exactly to recorded precision: `radius` `0.996624` with `27/27`
cells defined; energy `0.655508` with `27/27` cells defined. This flips the
warning verdict under `radius` while preserving the Phase 15 Fail-Magnitude
mechanism verdict.

### Phase 17 - Hazard-Aligned Counterfactual

Goal: test whether guarded TRACK actions move the state away from the same
frozen hazard boundary used by the Phase 15/16 oracle label, rather than toward
lower energy. The primary score is signed distance to the nearest terminal
hazard boundary: `min(escapeRadius - r3, minPrimaryDistance - closeApproachRadius)`.

Implementation-grade draft and pending result note:

- [`docs/threebody/PHASE17_SPEC.md`](threebody/PHASE17_SPEC.md)
- [`docs/threebody/PHASE17_RESULTS.md`](threebody/PHASE17_RESULTS.md)

Reserved commands:

```bash
npm run threebody:phase17:hazard-cf-smoke
npm run threebody:phase17:hazard-cf
```

Exit criterion: the project can say whether guarded TRACK's survival pocket is
explained by first actions that increase hazard margin at horizons `N in
{8,16,32}`, separated from delay/sign-flip ablations. This is a mechanism audit,
not a controller retune or claim broadening.

Status: spec drafted 2026-05-29; pending lock review. No Phase 17 code has been
written and no Phase 17 command has been run.

### 3D Catalog / Isotrophy Sidecar

Goal: test whether the new Li-Liao 2025 three-dimensional orbit catalog gives
Sundog a cheap first-principles, non-tautological theorem check independent of
the current planar controller envelope.

Sidecar artifacts:

- [`docs/isotrophy/sundog_v_isotrophy.md`](sundog_v_isotrophy.md)
- [`docs/isotrophy/files.math`](isotrophy/files.math)

The sidecar asks whether the residual spacetime `Z2` structure visible through
Li-Liao's fixed ansatz facet predicts the distinct piano-trio family count in
the 273 reported two-equal-mass piano-trio orbits. The raw 21/273 ratio is not
an unconditioned orbit-space sample; the ansatz itself enforces the concrete
beta-class symmetry cut `F_beta = ((12), time reversal, Rpi)`. This is a
theorem-sharpening workbench, not a controller result and not evidence for
Phase 13-15. A parser/gate smoke has run (`npm run isotrophy:parse`,
`npm run isotrophy:precondition-smoke`, `npm run isotrophy:smoke`,
`npm run isotrophy:sigma3-scan:smoke`). The full `m3=1` sigma3 precondition
scan has now run once as an 8.18-hour local receipt. It surfaced two gate issues
rather than confirming the old "21 exactly" contract: the original
`sigma3_inverse` was an opposite-orientation element, not the true inverse, and
the scan aggregator accepted `min(sigma3,sigma3_inverse)` instead of requiring
the full cyclic group to pass. With the true inverse and the closure-relative
full-group `max(sigma3,sigma3_inverse)` gate, the current harness split is 14
canonical candidates, 12 opposite-orientation candidates, and 25
any-orientation IC rows. The old three canonical absolute-pass/closure-fail
near misses are not global rejects; the symmetric opposite-orientation gate
places them at closure scale. The hard-21 readback is now reconciled by the
catalog's strict "single closed trajectory" convention: the SO(3)-gauged gate
admits 25 rows, but the optimized alignment rotation angle separates 21 strict
inertial single-curve choreographies from 4 rotating/relative choreographies
with a nontrivial `120 deg` global rotation. A zero-integration `(E, |L|)`
invariant cluster on the 25 rows produced 25 singleton groups, confirming the
four extra rows are distinct rotating orbits rather than duplicate listings.

G.2 (the σ3 precondition) is now **resolved**: the 21 strict single-curve
choreographies match the catalog's 21 exactly, and the 4 relative/rotating
choreographies are a separate, non-gating category. The `K_facet`
daughter-family experiment was scoped and pre-registered in the sidecar. K1
froze the primary strict prediction from the six-generator classification of
the 21 strict ICs: `K_facet = 0`. All 21 rows pass only the structural
`F_beta` strictly; the SO(3)-gauged diagnostic is 21 because `beta_I` appears
only with `Rpi` absorbed into free alignment, so it is not counted as the
strict prediction. That is now dispositioned as a v0.2 spec failure, not a
theorem result: the static containment operator is the equivariance-only null
(`Z3` choreography symmetry does not generically contain the transposition
`Z2` target). The v0.3a case split now covers both explicit spatial parity
candidates, `tau12_I` and `tau12_Z`, with the SO(3) gauge kept proper. It found
0 endomorphism cases and 21 induced-representation cases among the strict
choreographies; the improper sibling won the residual for 6 rows but moved none
near closure. The follow-up F_beta pair-ID receipt then confirmed 21 singleton
`(E, |L|)` groups, 0 inside-catalog bare-`(12)` partners, and 21/21
F_beta-closure-tight rows; the structural cocycle is manifest-level
`F_beta = ((12), tau-active, Rpi)`, not a per-row tau flag and not a partner
orbit IVP. Any continued v0.3 must therefore derive the induced-representation
functional using this uniform F_beta-conjugation relation before monodromy code
or supplementary-B clustering. A follow-up derivation review accepted the
loop-to-fiber direction but blocked code until a typed transport lemma is
written: the neutral block must be `span{X_H,u_E}`, `G_i` must be constructed
as an explicit fiber map, `G_i^2` must be computed after the cocycle is chosen,
structural subtraction must be a quotient/reduction through `B_i^+`, and
`1/2*dim` is only a candidate count until the semisimple/crossing-form gates
pass. K2-K4 remain paused, and this remains not evidence for the theorem or for
Phase 13-15.

The first typed transport response likely kills the canonical single-fiber
`G_i` and proposes a pair-orbit alpha-fixed kernel instead, but that replacement
is also not locked. The next paper-only gate is now a pair-orbit /
dihedral-representation lemma: fix the shifted-partner loop convention, correct
`A_F` as a map from `t` to `-t`, prove the alpha-fixed graph descends through
`N_i`, derive the `<sigma3,F_beta>` real representation on `K_i^{fib}`, and
only then state a multiplicity rule.

The second pair-orbit draft makes that more concrete: the current candidate is
an anchored real `D3=<sigma3,F_beta>` decomposition
`K_i^{fib} ~= a_i*T + b_i*S + c_i*E`, with `d_i_candidate=c_i`. It is still
paper-only. The blockers are now specific: choose the loop convention, certify
the parent and conjugate partner anchors, make the neutral quotient
`D3`-equivariant, and validate the standard-irrep count through the crossing
form before any monodromy run.

The neutral quotient has now been sharpened: it is `N_C=T*u_E+S*X_H`, not a
wholly trivial block. The post-quotient representation is
`K_i^{fib} ~= (a_i-1)*T + (b_i-1)*S + c_i*E`, so the candidate standard-irrep
count `c_i` is preserved. This is still paper-only; loop convention, typed
half-flow, anchor certification, G1, and G2 crossing-form remain gates.

The v0.3g crossing-form review sharpens G2: the mass perturbation should split
as `Delta H=Delta H_T+Delta H_E`, with no sign-irrep component because it is
`F_beta`-even. The `T` component is expected to collapse out of the standard
sector, leaving the `E` component as the branch-driving part. This is not yet a
locked count. Before any monodromy run, the child spec must define the
neutral-quotiented Floquet crossing form, decide whether `c_i>1` requires a
crossing matrix/rank gate instead of scalar per-block gammas, verify the
reduced-coordinate `Delta H`, prove anchor independence, and define an empirical
closure-relative `gamma_floor`.

The v0.3h refinement makes the matrix formulation primary: `Gamma_i` is the
`c_i x c_i` crossing matrix on the `F_beta`-even standard sector, and the
structural target is `d_i=rank_floor(Gamma_i)`. This exposes a new G2.6 gate:
symplectic block-orthogonality of the `E` copies. If the `E` blocks are
orthogonal in the `omega`-Gram matrix, the scalar rule is the special case; if
not, off-diagonal pairings can carry `T` contributions and the full matrix-rank
gate is required. Receipt schema should include the full matrix, singular
values, rank, `gamma_floor`, and a `symplectic_block_orthogonal_E` flag. This
remains paper-only and is not authorization for monodromy code.

The latest v0.3h review reframes that G2.6 point. The attempted
`M_i+sigma3` Floquet-basis closure is invalid on `K_i^{fib}` because
`K_i^{fib}=ker(M_i-I)/N_i`, so `M_i` is identity on the counted space. The
rank-matrix formula still stands, but the `T` contribution should vanish for a
simpler reason: a `D3`-equivariant `T` component preserves `Fix(F_beta)`, and
`Fix(F_beta)` is isotropic because `F_beta` is anti-symplectic. Thus G2.6 is a
basis-conditioning / scalar-readout diagnostic, not the proof of T-collapse.
Anchor changes should be treated as congruence of the bilinear matrix, and the
`gamma_floor` constants still need a pre-registered sentinel calibration before
any empirical step.

The operational v0.3h stance is now G2.6d: do not canonicalize for the count.
`d_i=rank_floor(Gamma_i)` is basis-invariant; per-block diagonal gammas are
basis-dependent diagnostics only and must record their basis convention. A
half-period `Phi_{T/2}^C` involution on `K_i^{fib}` is retained as a possible
paper follow-up for a physically meaningful scalar readout, not as a blocker.
Remaining before code: write the operator-level `F_beta` preservation proof,
cite the routine real-Schur/no-`S` algebra, replace old Krein-in-`M_i` language
with degeneracy/bimodality of `(partial_epsilon M_i)_E` or `Gamma_i` singular
values, and pre-register the `gamma_floor` calibration receipt.

The v0.3i calibration scope is now one sentinel row only. Primary sentinel is
`O_62`; the backup ladder follows canonical-strict period order if `c_i=0`.
Constants are fixed before the run (`k_gamma=3`, `k_int=10`), and the sentinel
passes only if the `Gamma_i` singular values avoid the marginal band
`[gamma_floor, k_gamma*gamma_floor]`. Receipt schema separates
`dE_perturbation_spectral_degeneracy_E` (basis/scalar diagnostic ambiguity)
from `gamma_singular_bimodality_clean` (rank-gate ambiguity). v0.3i does not
authorize the full 21, supplementary-B, or freezing `K_facet_v0.3`; those would
require a later v0.3j authorization.

The sentinel runner spec is now reviewed for implementation planning. It may
scaffold `npm run isotrophy:kfacet:sentinel`, but should not execute until it
has explicit typed `D3` product construction, a verified reduced symplectic form
for `Gamma_i`, deterministic basis diagnostics, and separate SVD/eigen
degeneracy vocab. The ansatz IC is the `F_beta` fixed anchor in closed form, so
the runner does not search for anchors. Partner-orbit integration is a receipt
sanity check only.

The first 21-row v0.3h sweep calibrated the implementation floors before any
full interpretation: `closure_floor=1e-8` split the kernel noise cluster on
several rows, so the runner now uses `closure_floor=1e-7`; the
joint-baseline gate now uses `1e-8` relative instead of `1e-9`. Only rows with
clean D3/F_beta stabilization at that floor may treat `c_i=0` as structural.
The clean rows so far (`O_62`, `O_64`, `O_231`) have
`ker(M-I)=T(2)+S(5)+E(0)`, hence `d_i=0`. If the calibrated 21-row rerun keeps
that profile, v0.3 Gamma is a catalog-level structural negative rather than a
numerical failure.

The calibrated 21-row rerun now has a sharper split: 21/21 rows pass the
mechanical gates and all read `E=0`, `c_i=0`, `d_i=0`, but five rows
(`O_524`, `O_623`, `O_793`, `O_1488`, `O_1497`) have `F_beta` leakage above the
kernel-stability floor. Those five are conditional, not confirmed structural
zeros, until their F_beta asymmetry is resolved. The runner now promotes D3
kernel stability from diagnostic to Gamma pass condition so leaky rows cannot
silently count as evidence.

A receipt-only leakage triage on those five rows shifts the lead hypothesis
from "third F_beta cocycle" to "adaptive kernel boundary." Each leaky row has
one boundary singular vector just above `closure_floor=1e-7`; including it
stabilizes F_beta (`O_524/O_623/O_1488/O_1497` at `3e-7`, `O_793` at `1e-6`)
without touching the order-scale non-kernel gap. The next registered move is a
no-integration adaptive-floor reprocessor, followed by a single O_1488
confirmation rerun only if the reprocessor keeps `E=0`.

The six-row reprocessor closes the original leakgate split without integration:
the repaired rows read `T(2)+S(6)+E(0)`, so the added boundary vector is
sign-sector, not standard-sector. The full calibrated 21-row reprocessor then
gives the actual load-bearing v0.3h catalog result: **20/21 strict rows are
structural zeros** (`E=0`, `c_i=d_i=0`). The remaining row, `O_617`, is a bridge
case. Its D3 leakage and gap ratio are clean at `k_dim=7`, but its next singular
value is `7.84e-4`, below the registered `1e-3` first-rejected guard; admitting
that vector gives an odd `E(1)` residual rather than a clean real 2D standard
block. O_617 is therefore a sub-investigation, not counted evidence.

The bridge audit now closes that sub-investigation. With the bridge band made
adaptive-floor-aware, the final audit split is `20 no_bridge_present / 1
defective_E_block_confirmed`. `O_617` has negligible neutral overlap, its
Jordan-chain norm amplifies (`drop = 90.04`), and admitting the bridge vector
makes the D3 representation itself defective (`T(2)+S(6)+E(1)`, `P_E` marginal
SV `0.01475`, `sigma3^3-I = 3.96e-2`). The row is outside the v0.3h Gamma
framing at that boundary and remains excluded from evidence, not counted for or
against the prediction.

The deep-dive companion (`docs/isotrophy/kfacet/kfacet_v03h_o617_deep_dive.md`)
corrects the first attribution: `O_617` is a clean opposite-strict catalog row,
with admitting residual `1.01e-8` (`to_closure = 1.105`). The earlier
`1.62e-1` number is the canonical residual and is diagnostic-only for an
opposite-strict row. The bridge vector is exactly tangent to the `(E, |L|)`
level set (cosines `~1e-10`), so the defect lives in the representation rather
than in the orbit. The headline therefore reads: **v0.3h resolves 20 strict
catalog rows as structural zeros; the sole quarantined row, O_617, is a clean
opposite-strict row with a bridge direction outside the valid D3
representation, not a Gamma_i failure and not an admission weakness.**

A signed isotypic pass sharpens the O_617 label: the bridge is near the sign
sector, not the trivial sector (`<v,F_beta v> = -0.9999997`, with
`F_beta^2 v-v = 0`). The small `E(1)` readout is projector contamination from
that near-S bridge under imperfect `sigma3` closure, not evidence for a valid
standard block. A catalog-wide isotypic-edge separator finds this edge only on
O_617; the other 20 rows remain clean `T/S` with no standard `E` direction.

The freeze-and-compare pass closes v0.3 as a structural-null mechanism. The
resolved `Gamma_i` prediction is `K_facet_v0.3h = 0` (20 structural-zero rows,
with O_617 quarantined and not counted). The local supplementary-B mirror
parses as 273 piano-trio rows (38 at `m3=1`, 235 off the equal-mass slice), so
v0.3h does not explain the published piano-trio catalog. This is a negative
about the proposed standard-sector mechanism, not a claim that piano-trios do
not exist.

The II cross-`m_3` sentinel sweep (`docs/isotrophy/kfacet/kfacet_v03_gamma_crossm3_preregistration.md`)
extended the audit chain to seven supplementary-B sentinels (4 at `m_3=0.4`,
3 at `m_3=1.0`). All seven halted at the runner-stage `D3` gate with
kernel-projected residuals `10^3..10^5`, six to eight orders above the
`1e-3` floor: joint verdict **(Q1.D, Q2.D) = gate pathology on both axes**.
A targeted `sigma_3-scan` on the same seven rows resolved the cause:
piano-trio orbits have catalog-style `sigma_3` closure residuals at
`~0.7` (orbit scale), 7-9 orders above the strict G.2 admission criterion,
while 6 of 7 carry `F_beta` closure cleanly at `~1e-8`. Zero of seven rows
satisfy any `sigma_3` admission flavor.

**Supplementary-B piano-trio sentinels sit in `Z_2`-or-smaller symmetry
classes, not `D_3`**. Six of seven tested rows carry the `(12)`-swap symmetry;
outlier `O_434(0.4)` also breaks `F_beta` closure and is flagged as a
sub-investigation. The v0.3 `Gamma_i` mechanism is `D_3`-equivariant by
construction and therefore structurally inapplicable to this catalog. The
(predicted 0, observed 273) mismatch is a **domain-of-applicability finding**:
v0.3 cannot predict piano-trios because they do not carry the assumed symmetry.
The mechanism is intact within its domain (strict G.2); the daughter catalog
lives in a different symmetry class. The v0.3 chapter closes; the v0.4 chapter
opens on `Z_2`-equivariant mechanism candidates, with smaller-symmetry outliers
tracked explicitly and paper-side design before any runner code. Receipts:
`results/isotrophy/k-facet-v03-gamma-crossm3/`,
`results/isotrophy/k-facet-v03-piano-symmetry-probe/`.

Follow-on: v0.4a treats supplementary-B piano-trios as primary `Z_2`
objects and pre-registers a two-pass gauge domain map over all 273 rows;
see [`../docs/isotrophy/kfacet/kfacet_v04a_domain_map_preregistration.md`](../docs/isotrophy/kfacet/kfacet_v04a_domain_map_preregistration.md).
The O_434 anatomy probe (verdict `gauge_artifact`; receipt at
`results/isotrophy/k-facet-v04a0-o434-anatomy/`) found that the default
`sigma_3-scan` tolerances misclassified O_434 by six orders of magnitude;
the two-pass classifier encodes that lesson.

**Update 2026-05-23:** v0.4 has closed as a publishable structural-negative.
v0.4a landed `outcome_A_all_Z2_clean` (273/273 supp-B rows in `Z2_clean`
after the two-pass classifier, 24 rescued by Pass 2). v0.4b registered two
Z_2-shadow stability predictors: `gamma_3` (tangent-isotypic) **retired
pre-sweep** with verdict `form_precondition_failed` after a sanity probe
showed `F_beta` does not preserve `K_fib` on supp-B (leakage 0.10..0.77
across 7 rows; cocycle typed `(I, 0)` so no transport rescue); the
replacement `gamma_3'_orbit_pass2` (predict S iff Pass 2 rescue required)
**falsified** at `chi^2 = 1202.32` vs critical `26.22` (chi-squared(12),
p=0.01), with rule accuracy 63.74% slightly below the always-U 64.47%.
**Stability information on this catalog is not carried by the Z_2 shadow**
at either tangent-isotypic or orbit-gauge-rigidity granularity. v0.5 opens
with a **branch-shadow audit**: catalog-only 4-bit hash on
`(m_3<1, z_0<0.3, |v_z|<1e-6, m_3 z_0^2<2)` with deterministic constant-bit
retirement (`|v_z|<1e-6` and `m_3 z_0^2<2` are constant on supp-B; both
retired). Active signature `(b1, b2)` gives 4 occupied buckets; chi-squared
independence vs S/U at `df = 3`, critical `11.34` (p=0.01). **v0.5a
verdict landed: `branch_hash_passes_audit` at `chi^2 = 34.986`** vs critical
11.34 (p ~= `1.23e-7`, 3.1x threshold). The audit-dominant bucket is
`(m_3<1, z_0<0.3)` with 113 rows / 55.75% S vs the catalog mean 35.53%
(chi^2 contribution 20.17 of 34.99); the other three buckets sit at
20-29% S. v0.5a is an AUDIT, not a predictor. **v0.5b verdict landed
2026-05-23: `branch_predictor_fails_heldout`.** Leave-one-m_3-bin-out
fold-trained branch-majority on `(b1, b2)` over the 263-row gating
subset returned model accuracy `0.6198` vs always-U `0.6388`, accuracy
delta `-0.019` (predictor LOSES), McNemar win=28 / loss=33 / p=1.0.
The load-bearing fold is m_3 = 0.4 (55 rows / 35 S, the catalog's most-stable
bin): held out, the residual five `m_3 < 1` bins flip the
`(m_3 < 1, z_0 < 0.3)` bucket to a U majority, so the predictor cannot
capture the m_3 = 0.4 stable cluster on its own held-out fold. **Joint
v0.5 reading:** the branch shadow is in-sample associated with stability
(v0.5a chi^2 = 34.99) but does not generalize across held-out mass bins
(v0.5b delta = -0.019). Clean projection-limit: the branch hash is a
descriptive catalog partition, not a predictive mechanism. Combined with
v0.4 negatives (Z_2 tangent precondition failed; Z_2 orbit-gauge-rigidity
chi^2 = 1202), stability information on supp-B is NOT carried by either
the Z_2 projection or the catalog-coordinate branch shadow at the tested
granularities. **v0.5 chapter closed 2026-05-23** as the first
**projection-limit** chapter in the isotrophy program; chapter close at
`docs/isotrophy/kfacet/kfacet_v05_writeup.md`. Three sub-results carry
forward to v0.6: bin-locality of the m_3 = 0.4 stable cluster on supp-B;
audit-vs-predictor asymmetry as a methodological finding; the joint
v0.4+v0.5 projection-limit envelope on the supp-B catalog.

**v0.6 parent registered 2026-05-23** with the **conserved-quantity
(E, |L|) stratification** family. Parent registration at
`docs/isotrophy/kfacet/kfacet_v06_mechanism_preregistration.md`. The
chosen projection is strictly richer than v0.5's 2-bit branch hash
(continuous orbit-level invariants vs. catalog-coordinate indicator
bits) while remaining catalog-derivable (per-row E and |L| computed
in seconds from published initial conditions and the three-body
Hamiltonian; no orbit integration). Operational definitions and
non-circularity provenance locked; v0.4/v0.5 disallowed-feature list
inherited; v0.5 audit-then-predictor + asymmetric McNemar + delta
falsifier discipline inherited.

**v0.6a form lock registered 2026-05-23** (audit form A, univariate
energy quartiles). Locked shape: 4-bin quartile contingency on E with
chi-squared(3) at p = 0.01 (critical 11.34); |L| quartile sidecar
report-only; pre-registered alignment-tightness scalar guards
against inheriting v0.5b's bin-locality failure mode (if max Q_E bin
alignment with v0.5a branch_label > 0.8, v0.6b's partition must be
re-registered before any held-out compute). v0.6a form lock at
`docs/isotrophy/kfacet/kfacet_v06a_energy_quartile_audit_form.md`.
Next runner action: per-row E/|L| computation against the supp-B
parser output, sanity-check against the 7 v0.3 cross-m_3 sentinel
rows (per-row residual < 1e-6), bound-orbit check (E < 0 for all
273), then implement and run `scripts/v06a_energy_quartile_audit.py`.
See `kfacet_v04_writeup.md`, `kfacet_v05_writeup.md`,
`kfacet_v05a_branch_map_form.md`,
`kfacet_v05b_branch_predictor_form.md`,
`kfacet_v06_mechanism_preregistration.md`, and
`kfacet_v06a_energy_quartile_audit_form.md`.

**v0.6a verdict landed 2026-05-23:** the energy-quartile audit passes
but with the registered alignment warning. Sanity and bound gates pass
(`max |Delta E| = 0`, `max |Delta |L|| = 0`, all 273 rows have
`E < 0`). Primary result: `chi^2_E = 33.703158` vs critical `11.34`
(df=3, `p = 2.29e-7`), but `alignment_tightness_scalar_E = 0.955882`
exceeds the 0.8 warning threshold. The |L| sidecar is also loud
(`chi^2_|L| = 28.954252`, report-only) and similarly aligned
(`0.956522`). Interpretation: conserved quantities do stratify S/U
in-sample, but this first signal is tightly entangled with the v0.5a
branch shadow. v0.6b may therefore proceed only as an
alignment-breaking re-registration, not as the default leave-one-m_3
predictor. Receipt:
`results/isotrophy/k-facet-v06a-energy-quartile-audit/manifest.json`.

**v0.6b form lock registered 2026-05-23:** alignment-breaking
within-branch audit on the v0.5a `(m_3 < 1, z_0 < 0.3)` stratum
(113 rows / 63 S / 50 U). Locked shape: `Q_E` under v0.6a's GLOBAL
supp-B quartile cutpoints (no re-binning within branch), n_occupied
x 2 contingency, pre-registered sparse-cell fallback (asymptotic
chi-squared if min expected cell >= 5, exact permutation test with
seed `20260523` and `n_permutations = 10000` if any expected cell
< 5, `within_branch_energy_inconclusive_sparse` verdict if any
occupied bin has < 2 rows). Within-branch `Q_|L|` audit emitted as
report-only sidecar. Form lock at
`docs/isotrophy/kfacet/kfacet_v06b_within_branch_energy_audit_form.md`.

**v0.6b verdict landed 2026-05-24:** `within_branch_energy_fails_audit`
at `chi^2 = 6.904`, permutation `p = 0.0292` (sparse-cell fallback
fired because `min_expected = 2.655 < 5`). Within-branch Q_E
contingency: Q1 empty (Q1 entirely outside stratum), Q2 N=6 (33% S),
Q3 N=42 (43% S), Q4 N=65 (66% S). The within-branch direction is
monotone in the same direction as v0.6a's catalog-wide finding, but
permutation p = 0.029 doesn't clear the registered p <= 0.01 floor.
Within-branch Q_E x m_3 joint diagnostic shows Q_E is essentially a
1-to-1 label for m_3 sub-bin within the stratum (Q4 = m_3 in {0.4,
0.5} with 52 of 65 at m_3 = 0.4). |L| sidecar likewise short of
loud-signal threshold (`chi^2 = 4.465`, permutation `p = 0.0741`).
v0.6c (held-out predictor) is NOT licensed. **v0.6 chapter closed
2026-05-24** as the third distinct chapter-close type: a
**conditional-independence close** (distinct from v0.4
structural-negative and v0.5 projection-limit). Joint v0.4 + v0.5 +
v0.6 envelope: three sequential pre-registered low-dimensional
projections of the supp-B body — Z_2 symmetry shadow, 2-bit catalog
branch shadow, and continuous conserved-quantity shadow — none
carries held-out or branch-conditional stability information on this
catalog. Stability on supp-B is bin-local to the m_3 = 0.4 cluster;
no catalog-coordinate or symmetry-shadow projection tested so far
lifts that locality to a generalizable mechanism. Chapter close at
`docs/isotrophy/kfacet/kfacet_v06_writeup.md`. Receipt:
`results/isotrophy/k-facet-v06b-within-branch-energy/manifest.json`.

**v0.7 parent registered 2026-05-24** with the **gamma_1
direction-of-instability** family (codex direction). Parent
registration at `docs/isotrophy/kfacet/kfacet_v07_mechanism_preregistration.md`.
The chosen projection LEAVES catalog-coordinate space entirely:
per-row monodromy M_i is decomposed into its Floquet eigenstructure,
and the eigenvector direction of a pre-registered eigenvalue choice
is projected onto a pre-registered geometric reference frame. The
parent registration explicitly identifies three circularity risks
(eigenvalue-choice, well-definedness, feature-extraction) and locks
the discipline that all three must be addressed in the v0.7a form
lock with an explicit non-circularity argument. v0.4b
disallowed-feature inheritance is re-asserted (no Floquet-magnitude,
no K_fib tangent decomposition); v0.5 audit-then-predictor +
asymmetric McNemar+delta inheritance and v0.6 alignment-tightness
guard + sparse-cell fallback tree inheritance are all locked.

**v0.7a form lock registered 2026-05-24** (D5 + B; velocity-fraction
quartile audit). Codex picked D5 over D1 on the rationale that
eigenvalue ordering is too close to the S/U definition; D5
minimizes the tie-break surface and uses Floquet eigenvectors only
as geometric directions. D1 + A is preserved as a named report-only
sidecar.

**Two amendments locked during the v0.7a run** (each with empirical
evidence): R1 amended the symplecticity sanity gate from 1e-6 to
1e-4 after the 7-row vectorized smoke showed residuals scaled with
period / Floquet amplification rather than implementation breakage;
R2.A added a per-row integration-failure fallback (catch, mark
`integration_blocked`, exclude from chi-squared denominator, with a
5%-of-catalog attrition threshold) after the first full runner
crashed at row 76 (O_194 at m_3=0.5) inside the variational
integrator. R2.C engineering note locked append-per-row + resume
mode. Firewall re-asserted: residuals and blocked-row status are
QC/provenance only.

**v0.7a verdict landed 2026-05-24:**
`velocity_fraction_blocked_integration_attrition` at
`integration_blocked_count = 23` (8.42% of catalog, above the
pre-registered 5%/14-row attrition threshold). 250 analyzable
rows + 11 additional sanity-gate failures = 12.5% data-integrity
issues. Blocked rows cluster at long-period high-m_3, the same
regime v0.4a's two-pass classifier was built to handle. The audit
is integration-attrited at the locked variational precision, NOT
feature-falsified. No catalog-wide chi-squared verdict is licensed.

**v0.7a' restricted-scope confirmation landed 2026-05-24 [PASS]:**
Pre-registered minimum-scoped audit on the 250 analyzable rows with
explicit Pass / Partial / Fail criteria tied to the v0.5a branch
hash. Verdict: `velocity_fraction_restricted_passes_audit` at
`chi^2 = 16.425` (df=3, critical 11.34, `p = 9.3e-4`) with
`alignment_tightness = 0.698 < 0.8`. Non-monotone U-shape signature:
Q1 (gamma_1 mostly positional) 49.2% S, Q2 (mixed) 17.7% S, Q3
29.0% S, Q4 (gamma_1 mostly velocity) 42.9% S. Direction-purity, not
branch label, stratifies stability on the analyzable sub-catalog.
This is the **first** non-branch-aligned positive in the four-chapter
v0.4/v0.5/v0.6/v0.7 envelope. **v0.7 chapter closed 2026-05-24 as
qualified-positive on restricted domain** -- a fourth distinct
chapter-close type. Chapter close at
`docs/isotrophy/kfacet/kfacet_v07_writeup.md`. Receipts:
`results/isotrophy/k-facet-v07a-velocity-fraction-audit/manifest.json`,
`results/isotrophy/k-facet-v07a-prime-restricted-scope/manifest.json`,
`scripts/v07a_velocity_fraction_smoke.py`,
`scripts/v07a_velocity_fraction_audit.py`,
`scripts/v07a_prime_restricted_scope_audit.py`.

**v0.8 parent registered 2026-05-24** on the Floquet direction-purity
mechanism (codex direction following the v0.7a' U-shape). Parent
registration at `docs/isotrophy/kfacet/kfacet_v08_mechanism_preregistration.md`.
Body: the 250 analyzable supp-B rows from v0.7a (attrition carried
as permanent domain restriction). Projection: `purity = abs(vf - 0.5)`,
a monotone deterministic transform of v0.7a's velocity-fraction
inheriting all v0.7a non-circularity provenance.

**v0.8a form lock + verdict landed 2026-05-24 (FAIL):** Codex picked
candidate A (purity-quartile audit). Form lock at
`docs/isotrophy/kfacet/kfacet_v08a_purity_quartile_audit_form.md`;
runner at `scripts/v08a_purity_audit.py`. Verdict:
`purity_quartile_fails_audit` at `chi^2 = 4.94, p = 0.176, alignment
0.587`. The diagnostic purity_signed contingency reproduces v0.7a's
`chi^2 = 16.43` exactly (linear transform; quartile ordering
preserved). Asymmetry diagnostic `|S(Q1_signed) - S(Q4_signed)| =
0.064` BELOW the 0.2 flag threshold -- the S-fractions at the two
pure ends are similar. The asymmetry that breaks the purity
transform is in **catalog row density** (184/250 rows below vf =
0.5; max vf = 0.87), NOT S-fraction. **The v0.7a' chi^2 = 16.43
signal IS real and not branch-aligned, but lives in the SIGNED
direction of vf, not in unsigned distance from 0.5.** v0.8b
(held-out predictor) is NOT licensed under the fails verdict.

**v0.8 chapter closed 2026-05-24** as a structural-negative on
direction-purity; chapter close at
`docs/isotrophy/kfacet/kfacet_v08_writeup.md`. Six chapter-close
types in the envelope so far. v0.7a's chi^2 = 16.43 remains the
load-bearing positive; v0.8a confirmed it is signed not symmetric.

**v0.9 parent registered 2026-05-24** on the signed Floquet
direction-composition mechanism at
`docs/isotrophy/kfacet/kfacet_v09_mechanism_preregistration.md`,
with a new permanent **anti-circular framing discipline**: any
v0.9 child form lock that tests vf-ordering vs S/U on the v0.7a'
analyzable subset via chi^2 of independence reproduces v0.7a' by
linear-transform invariance.

**v0.9a form lock + verdict landed 2026-05-24:** Codex picked
candidate C (three-zone audit with physical cutpoints
{0.25, 0.50}). Verdict:
`signed_vf_three_zone_fails_audit_chi2` at `chi^2 = 7.42` vs
critical 9.21 (df=2, p = 0.0245). Substantive interpretation:
the v0.7a' U-shape was a quartile-boundary artifact at vf ~ 0.297;
under physical cutpoint vf = 0.25, the positional-dominant zone
shrinks from 63 to 19 rows and its S-fraction collapses from 49%
to 10.5%. The actual physical pattern is **monotone-increasing
in vf**, but chi^2 does not reach the p = 0.01 floor.

**v0.9 chapter closed 2026-05-24** as a structural-negative on the
U-shape mechanism hypothesis with the monotone-increasing meta-
finding preserved. Chapter close at
`docs/isotrophy/kfacet/kfacet_v09_writeup.md`.

**Isotrophy program PAUSED 2026-05-24 at end-of-v0.9.** Pause
document at `docs/isotrophy/kfacet/kfacet_isotrophy_program_pause.md`.
Seven sequential pre-registered chapters; seven distinct chapter-
close types; one load-bearing substantive positive (v0.7a' chi^2 =
16.43, branch-independent, signed-direction); one comprehensive
methodology surface (closure-relative discipline, two-pass
classifier, audit-then-predictor, asymmetric falsifier, alignment-
tightness guard, sparse-cell fallback, non-circularity sentence
template for Floquet features, per-row integration-failure
fallback, append-per-row resume, anti-circular framing for
sequential audits). **Eight concrete reopening avenues** recorded
for lab initiates: Jonckheere-Terpstra trend test on v0.9a;
held-out v0.7b on the 250 analyzable rows; v0.7a relaxed-precision
re-run; m_3 = 0.4 sub-catalog targeted investigation; joint
(vf, Q_E) audit; cross-substrate transfer; action-angle / KAM
decomposition; full-catalog audit at rtol = 1e-10. The program
PAUSES, it does not RETIRE -- reopening is invited under the
locked discipline. Bandwidth redirects to three-body Phase 15+
per the original direction call.

### Cross-Substrate Hand-Offs

The Threebody project now has a Mesa/Geometry-style crossover note at
[`docs/threebody/CROSS_SUBSTRATE_NOTES.md`](threebody/CROSS_SUBSTRATE_NOTES.md).
It translates Mesa's entangled-substrate discipline and Geometry's
HaloSim-oracle discipline into candidate Threebody phases: mechanism
decomposition/action coupling, forward-oracle precision lock, and spatial/3D
extension. The note is a design guardrail, not evidence by itself.

## Recommendation

Proceed, but make the first milestone diagnostic rather than controller-first.
The idea is promising because it gives Sundog a high-recognition physics hook
without needing a grandiose claim. The research discipline is to earn three
things in order: first predictive signatures, then sensor-available proxies,
then control.

## Implementation Status (2026-05-07)

**Phase 1-3**: Prototype implemented, validation pending. The interactive
browser visualization at `threebody.html` now scaffolds the diagnostic signals,
sensor-limited proxy mode, and SCAN/SEEK/TRACK controller modes. It does not yet
complete the benchmark requirements for event prediction, oracle comparison, or
failure-boundary mapping.

**Phase 4**: Draft public artifact live, still evidence-bounded. It includes:
- Interactive visualization with real-time RK4 integration
- Canvas rendering of system state with orbital trails
- Indirect signature overlays (virial, inertia, energy, tidal tensor)
- Phase 2 sensor-limited mode toggle
- Phase 3 controller implementation (SCAN/SEEK/TRACK)
- UI controls for exploration and experimentation
- Visual design consistent with site branding

**Public writeup**: See [threebody-writeup.md](threebody-writeup.md) for the
complete "Steering by the Shadow of Chaos" documentation, including:
- Claim boundaries and hypothesized failure modes
- Sensor model audit (privileged vs. sensor-available signals)
- Phase-by-phase implementation details
- Connection to photometric alignment experiment
- Future directions

**Phase 5**: Complete for the prototype tier. A deterministic harness now lives
at:

```bash
npm run threebody:phase5
```

Current Phase 5 implementation:

- `public/js/threebody-core.mjs`: shared planar restricted three-body dynamics,
  signature computation, tidal proxy computation, prototype controller logic,
  event labels, and trial runner.
- `public/js/threebody-browser.mjs`: browser visualization wrapper that imports
  the shared core module instead of carrying mirrored dynamics inline.
- `scripts/threebody-harness.mjs`: CLI batch harness that writes a manifest and
  per-trial JSONL logs.
- Default smoke output: `results/threebody/phase5-smoke/` (ignored by git).

Current status against Phase 5 exit criterion:

- Deterministic replay: smoke trial logs reproduce byte-for-byte across reruns.
- Manifest format: present, including seed, regime, controller mode, initial
  state, mass ratio, timestep, duration, thrust authority, target tidal
  magnitude, log path, and terminal summary.
- Per-trial logs: present, including state, exposed signatures, tidal tensor,
  thrust vector, event labels, and terminal outcome.
- Browser equation sharing: complete. Both the browser and harness now use
  `public/js/threebody-core.mjs` for initialization, integration, signatures,
  tidal proxy computation, and prototype controller thrust.

This does not make baseline or metric claims. It only completes the Phase 5
reproducibility substrate needed for Phase 6 and Phase 7.

**Phase 6**: Started. Baseline modes and a matched-slate runner now exist:

```bash
npm run threebody:phase6
```

Current Phase 6 implementation:

- Passive baseline: `off`, no thrust.
- Naive local baseline: `naive`, pushes against local acceleration plus simple
  velocity damping, without tidal-gradient structure.
- Sundog ablations: `scan`, `seek`, and `track`.
- Tidal-signal ablations: `seek_noisy`, `track_noisy`, `seek_shuffled`, and
  `track_shuffled`. The noisy variants perturb observed tidal magnitude and
  gradient; the shuffled variants preserve controller wiring but replace the
  tidal-gradient direction with a deterministic pseudo-random direction.
- Privileged baseline: `oracle`, a full-state lookahead guard that scores
  candidate thrust directions using simulator state. This is a privileged
  heuristic, not an optimal controller.
- Browser controls expose `naive`, `oracle`, and the tidal-signal ablation modes
  for qualitative inspection.
- Harness output includes:
  - `summary.csv`: grouped by regime and controller mode.
  - `paired.csv`: per-regime/per-seed deltas against matched passive and oracle
    rows.
  - `comparison.csv`: aggregate improvement/tie/worsening counts versus
    passive, plus mean time and fuel deltas.
- The manifest declares baseline mode definitions and primary metrics:
  `terminalOutcome`, `simulatedTime`, `totalDeltaV`, `minPrimaryDistance`,
  `saturationCount`, `targetBandLossCount`, and initial Phase 7 tidal-warning
  diagnostics.
- Default smoke output: `results/threebody/phase6-baselines/` (ignored by git).

Current status against Phase 6 exit criterion:

- Matched slate: present for all current modes across stable, near-escape,
  near-collision, and chaotic seed regimes.
- Comparison artifacts: present via manifest, JSONL logs, summary CSV, paired
  CSV, and aggregate comparison CSV.
- Interpretation: not yet complete. The initial 3-seed smoke slate exposes hard
  regimes and several weak controllers; it should be treated as calibration for
  baseline design, not as evidence for or against the Sundog claim.
- Next Phase 6 work: tune the privileged baseline objective, define a larger
  seed slate, and decide whether the primary metric ordering should privilege
  survival time, bounded outcome, or fuel cost.

**Phase 7**: Started. The harness now computes event-diagnostic metrics from
the per-step event history:

```bash
npm run threebody:phase7
```

Current Phase 7 implementation:

- Hazard label: first escape or close-approach event.
- Warning scores:
  - `tidal_magnitude`: Frobenius norm of the local tidal tensor estimate.
  - `local_acceleration`: magnitude of local gravitational acceleration at the
    test particle; this is the naive local warning baseline.
- Default warning labels: first tidal-spike event under the configured
  `tidalSpikeThreshold`, and first local-acceleration warning under
  `localAccelerationWarningThreshold`.
- Warning window: samples within `eventWarningHorizon` seconds before a hazard.
- Threshold sweeps: configurable `thresholdSweep` values for tidal magnitude and
  `localAccelerationThresholdSweep` values for local acceleration, with
  per-trial and aggregate precision/recall/F1/false-alarm rows.
- Calibration curves: quantile-binned warning scores, with observed event-window
  rate per regime, controller mode, and score type.
- Per-trial diagnostics:
  - first hazard time;
  - first tidal-warning time;
  - warning lead time;
  - true/false positive and true/false negative sample counts;
  - precision, recall, F1, false-alarm rate, and AUROC using tidal magnitude as
    the warning score;
  - the same threshold metrics and AUROC for local acceleration.
- Harness output now includes `event-metrics.csv` alongside `summary.csv`,
  `paired.csv`, and `comparison.csv`.
- Additional Phase 7 outputs:
  - `threshold-sweep.csv`: one row per trial, score type, and threshold.
  - `threshold-summary.csv`: aggregate threshold metrics by regime, controller
    mode, score type, and threshold.
  - `calibration.csv`: binned warning-score calibration rows.
- Default smoke output: `results/threebody/phase7-event-metrics/` (ignored by
  git).

Current status against Phase 7 exit criterion:

- Event plumbing: present for both tidal magnitude and local acceleration.
- Threshold sweeps and calibration rows: present, but only for the current small
  smoke slate.
- Interpretation: not yet complete. Some regimes are label-degenerate under the
  one-second warning horizon, so AUROC or false-alarm rate may be blank when a
  trial has no positive/negative comparison class. Calibration bins may also be
  non-monotone, which means neither raw score is yet a calibrated risk
  probability.
- Next Phase 7 work: add reliability/error summaries for the calibration bins,
  tune threshold grids per regime, and run a larger seed slate before ratcheting
  any public claim.

**Phase 8**: Started. The harness now emits a sensor-model audit for the tidal
tensor proxy:

```bash
npm run threebody:phase8
```

Current Phase 8 implementation:

- Sensor tiers:
  - `simulated_local_probe`: exact virtual local probe samples from the simulator
    field model; this is the current reference tier.
  - `accelerometer_array_noisy`: short-baseline acceleration samples with
    deterministic Gaussian noise.
  - `delayed_local_probe`: exact virtual local probe estimate delayed by the
    configured number of integration samples.
  - `micro_maneuver_noisy`: larger-baseline probe with delay and extra
    maneuver-contamination noise.
- Sensor-tier controller modes:
  - `seek_sensor_accel`, `track_sensor_accel`
  - `track_sensor_accel_guarded`
  - `seek_sensor_delayed`, `track_sensor_delayed`
  - `seek_sensor_micro`, `track_sensor_micro`
- Harness options:
  - `--sensor-variants`
  - `--sensor-audit-every`
  - `--sensor-noise-std`
  - `--sensor-delay-steps`
  - `--micro-maneuver-contamination-std`
- Calibration sweep:
  - `npm run threebody:phase8:calibrate`
  - `npm run threebody:phase8:analyze`
  - `npm run threebody:phase8:focus`
  - Dedicated script: `scripts/threebody-sensor-calibration-sweep.mjs`.
  - Analyzer script: `scripts/threebody-analyze-sensor-calibration.mjs`.
  - Sweeps accelerometer-array noise, delayed-probe latency, and
    micro-maneuver noise/latency/contamination with matched passive baselines
    for each regime and seed.
  - Default output: `results/threebody/phase8-calibration-sweep/` (ignored by
    git).
  - Focused accelerometer output:
    `results/threebody/phase8-focused-accelerometer/` (ignored by git).
- Additional Phase 8 outputs:
  - `sensor-model-samples.csv`: per-trial/per-time sensor estimate rows.
  - `sensor-model-summary.csv`: aggregate tensor magnitude and component errors
    by regime, controller mode, and sensor variant.
- Additional calibration outputs:
  - `paired.csv`: per-trial sensor-controller result paired against the passive
    baseline for the same degradation setting, regime, and seed.
  - `summary.csv`: outcome survival, passive deltas, delta-v, minimum-distance,
    and warning-score aggregates by sensor tier and degradation setting.
  - `sensor-error-summary.csv`: sensor-estimate magnitude/component error by
    tier, degradation setting, regime, and controller mode.
  - `envelope.csv`: analyzer output ranking controller/degradation settings by
    survival delta, worsened rate, and sensor error.
  - `regime-envelope.csv`: analyzer output preserving the per-regime failure
    boundary.
  - `candidate-envelope.csv`: analyzer output for settings that meet the
    current strict candidate filter.
- Default smoke output: `results/threebody/phase8-sensor-model/` (ignored by
  git).

Current status against Phase 8 exit criterion:

- Sensor-tier labels: present in the manifest and summary outputs.
- Reference separation: present. The exact virtual probe is explicitly labeled
  as a simulator-field reference, not as a hardware-valid sensor claim.
- Default calibration sweep: present. The first run emitted 1,044 matched
  calibration trials; after adding guarded accelerometer TRACK, the default run
  emits 1,104 matched calibration trials. It writes paired passive-baseline
  comparisons, aggregate survival rows, and sensor-error rows for accelerometer
  noise, delayed-probe latency, and micro-maneuver degradation.
- Focused accelerometer check: present. The follow-up 12-seed accelerometer run
  emitted 960 trials with SEEK, TRACK, and guarded TRACK. The initial 3-seed
  candidate for ungated `track_sensor_accel` at noise `0.03` did not survive
  the larger slate. The guarded accelerometer TRACK variant is now the only
  focused candidate: noise `0` and `0.01` improve aggregate survival versus the
  matched passive baseline while keeping worsened rate under the strict filter.
  This is a Phase 9 hypothesis, not a validated claim.
- Interpretation: calibration only. The initial smoke run shows noisy
  accelerometer-array estimates preserve the reference tensor far better than
  delayed or contaminated micro-maneuver proxies in fast-changing regimes. When
  those degraded estimates drive SEEK/TRACK, accelerometer-array control mostly
  tracks the ideal behavior, while delayed and micro-maneuver variants break
  more stable cases. The first calibration sweep keeps the same broad shape:
  delayed and micro-maneuver tiers degrade quickly with latency, while
  accelerometer-array error grows with noise. Some noisier controller settings
  also produce better terminal survival on the tiny slate by changing thrust
  behavior, so these rows should be treated as operating-envelope clues, not as
  evidence that noise helps.
**Phase 9**: Started. The first operating-envelope runner maps guarded
accelerometer TRACK across initial-condition, thrust, sensor-noise, and guard
settings:

```bash
npm run threebody:phase9
```

Current Phase 9 implementation:

- Dedicated script: `scripts/threebody-operating-envelope.mjs`.
- Default output: `results/threebody/phase9-operating-envelope/` (ignored by
  git).
- Default smoke map: 1,944 trials across stable, near-escape, and chaotic
  regimes; three radius scales; three velocity scales; two thrust limits; three
  accelerometer-noise levels; and two guard-acceleration thresholds.
- Outputs:
  - `trial-outcomes.csv`
  - `paired.csv`
  - `envelope-map.csv`
  - `aggregate-envelope.csv`
  - `best-by-cell.csv`
  - `cell-class-map.csv`
  - `cell-delta-map.csv`
  - `candidate-envelope.csv`
- First result: 29 positive candidate rows out of 324 envelope rows. The
  best-cell table marks 4 cells promising, 21 neutral, and 2 risky. The positive
  pocket is near-escape; stable cells are mostly neutral and chaotic cells are
  mostly neutral or risky.
- Refined near-escape result: `npm run threebody:phase9:refine` emits a denser
  7x7 radius/velocity map with 2,352 trials. The best-cell matrix shows 32
  promising cells, 6 neutral cells, 4 mixed cells, 3 risky cells, and 4 negative
  cells. The positive pocket is connected at moderate-to-high velocity scales;
  larger-radius low-velocity cells are the local failure boundary.
- Mechanism read: `trial-outcomes.csv` is now the primary data artifact, and
  `envelope-map.csv` carries dominant failure mechanisms. In the refined run,
  harmful trial rows split into 119
  `controller_destabilized_or_shortened_passive` and 65
  `control_effort_or_saturation`; negative/risky map rows are mostly
  controller-shortened passive cases.
- Locked near-escape result: `npm run threebody:phase9:lock` emits 9,408 trials
  over the same refined 7x7 pocket with 24 seeds. The pocket survives: 41
  best-cell rows are promising, 7 are mixed, and 1 is negative. The strongest
  survival deltas remain at larger radius and higher velocity.
- Mass-ratio/timestep result: `npm run threebody:phase9:axes` emits 1,296 trials
  inside the locked high-velocity pocket. The pocket survives mass ratios
  `0.01`, `0.3`, and `1` across timesteps `0.008`, `0.01`, and `0.012`, with
  72 candidate rows out of 81. This supports the pocket as broader than the
  equal-mass/default-timestep geometry, within the scoped slice.
- Hazard-gate result: `npm run threebody:phase9:hazard` replaces the hand-tuned
  guard constants in the mass-ratio/timestep slice with passive quantile-derived
  hazard-score gates. The scoped run keeps 81 candidate rows out of 81, records
  no harmful trial rows, and lowers average delta-v relative to the constant
  gate probe.
- Full refined hazard-gate result: `npm run threebody:phase9:hazard:refine`
  keeps 100 candidate rows out of 196, compared with 116 under constant gates,
  while reducing average delta-v from about `2.07` to `1.12`. It preserves the
  high-velocity pocket but does not remove the low-velocity failure boundary.
- Phase 11 robustness result: `npm run threebody:phase11:guard-quantiles`
  confirms the pocket survives guard quantiles `0.5`, `0.75`, and `0.9`, with
  a clear survival/effort trade. `0.5` is cheapest, `0.9` has the largest
  average survival delta, and `0.75` remains a viable middle setting.
- Phase 11 outside-pocket result: `npm run threebody:phase11:outside-pocket`
  expands beyond the favorable high-velocity slice. It finds 96 promising, 36
  mixed, 9 negative, and 3 neutral best cells, with most harms concentrated at
  lower velocity scales and equal mass ratio.
- Phase 11 comparison result: `npm run threebody:phase11:compare` reruns the
  favorable pocket against passive, naive local acceleration, guarded
  accelerometer TRACK, and a privileged heuristic oracle. Guarded TRACK produces
  81 candidate rows out of 81, the naive local baseline produces none, and the
  heuristic oracle produces 34 of 81.
- Phase 13 long-horizon result: `npm run threebody:phase13` emits 3,456 trials
  at `duration=16`. The lock passes with boundary sharpening: 88 / 324
  candidate envelope rows, 81 / 108 promising best cells, guarded TRACK best in
  100 / 108 cells, and no high-velocity risky/negative best cells. The
  low-velocity `velocityScale=0.95` boundary contains all risky/negative cells,
  including all 5 negative equal-mass cells. Cost rate stays comparable to
  Phase 11 (`0.229` vs `0.218` delta-v per simulated second for guarded TRACK
  candidate rows). See
  [`docs/threebody/PHASE13_RESULTS.md`](threebody/PHASE13_RESULTS.md).
- Next work: move to the cross-substrate follow-ups in
  [`docs/threebody/CROSS_SUBSTRATE_NOTES.md`](threebody/CROSS_SUBSTRATE_NOTES.md).

**Interactive demonstration**: [threebody.html](../threebody.html)
