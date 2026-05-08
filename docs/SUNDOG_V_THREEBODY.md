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

Status: ratified as a promising research direction and public hook, not yet as
evidence for Sundog.

The hook is strong because the three-body problem is a familiar symbol of
dynamical complexity, and Sundog has a clean way to enter the conversation:
not by claiming to solve the dynamics, but by asking whether lower-dimensional
event signatures can be sufficient for useful action inside a bounded operating
envelope.

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

Sweep axes:

- initial position and velocity of the test particle;
- mass ratio of the primaries;
- thrust authority;
- target tidal magnitude;
- sensor noise and delay;
- integration timestep;
- perturbation impulse size.

Outputs:

- success/failure heatmaps;
- escape and close-approach regions;
- controller saturation regions;
- cases where tidal proxies are misleading;
- representative trajectories for wins, losses, and ambiguous regimes.

Exit criterion: the public page can show where the method works, where it
fails, and where the result is inconclusive.

### Phase 10 - Claim Ratchet

Goal: define exactly when the public claim is allowed to strengthen.

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

Do not claim:

- "Sundog solves three-body dynamics."
- "The controller predicts chaos."
- "Sensor-only control is validated" before Phase 8 passes.
- "Comparable utility from partial information" before baseline metrics show
  the trade.

Exit criterion: the writeup, page copy, and promo language all use the strongest
claim actually earned by the completed validation phase, and no stronger one.

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
- Next Phase 8/9 work: move from raw sensor calibration into an operating
  envelope map. The current hypothesis is guarded accelerometer TRACK under
  low-noise sensing. The next useful question is whether that guard remains
  positive across a wider initial-condition map and whether the guard thresholds
  can be calibrated from local hazard scores rather than tuned constants.

**Interactive demonstration**: [threebody.html](../threebody.html)
