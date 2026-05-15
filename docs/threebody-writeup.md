# Steering by the Shadow of Chaos

## The Claim

The three-body problem is famously intractable in closed form — no general analytic solution exists. Sundog does not solve the three-body problem. Instead, it asks a smaller, sharper question:

**Can an agent act usefully in three-body dynamics by reading the signatures around instability instead of reconstructing the full state?**

The current answer is bounded but no longer merely speculative. In the tested
planar restricted setup, a guarded accelerometer-proxy TRACK controller improves
survival over passive and naive local baselines in a robust high-velocity
near-escape operating pocket. The effect is not global: low-velocity and
equal-mass boundary cells still show harms, and guard quantile choice controls
the survival/effort trade.

## The Pattern

Three-body dynamics are chaotic. But the shadow of that chaos — the indirect signatures that emerge before events resolve in full phase space — may be enough to steer by.

The full state of a three-body system is 18-dimensional: three position vectors, three velocity vectors. But many practical questions about three-body dynamics don't require tracking the full state at every instant. They require detecting signatures of dynamically important events:

- Near-collisions
- Ejections
- Resonance capture
- Exchange events
- Stability transitions

These events cast shadows in lower-dimensional observables long before they resolve in full phase space.

## Indirect Signatures

### Phase 1: Diagnostic Benchmark (Privileged Signals)

The following signatures compress the full 18D spatial state into
lower-dimensional diagnostics that may forecast useful events. In the current
browser prototype they are implemented as live diagnostics, not yet as a
statistically evaluated benchmark.

**Virial Ratio (2T/|W|)**
The ratio of kinetic to potential energy. For a bound system this oscillates around 1. When it persistently exceeds 1, the system is unbound or about to eject. This single scalar compresses all 18 dimensions into a stability diagnostic.

**Inertia Tensor Trace**
A compression of the 9D positional configuration. When a three-body system is about to undergo a close encounter or ejection, the eigenvalue spectrum of the inertia tensor changes character before the event resolves. The smallest eigenvalue dropping signals approach to a collinear configuration.

**Pairwise Energy Partition**
For each of the three pairs, compute the two-body energy in their center-of-mass frame. In a stable hierarchical triple, one pair's energy is deeply negative (the inner binary) and one is weakly negative (the outer orbit). When the system transitions from hierarchical to democratic (the chaotic scattering regime), these energies approach each other. That crossing is an indirect signal for instability.

**System Energy (T + W)**
Total mechanical energy. In the absence of external forces, energy is conserved. Tracking energy helps verify integration accuracy and detect numerical drift.

### Phase 2: Sensor-Available Proxies

The Phase 1 diagnostics are **compressed**, but not automatically **indirect**. If computed from full simulator state, they are diagnostics, not Sundog-style partial observations.

Phase 2 replaces privileged diagnostics with locally measurable proxies:

**Tidal Tensor Estimation**
The test particle (body 3) estimates the gravitational field gradient by comparing its acceleration with that of nearby virtual probe points:

```
T_ij = ∂²Φ/∂x_i∂x_j ≈ Δa_i/Δx_j
```

This measurement captures tidal field strength without directly reporting the
positions or masses of the primaries. In the browser implementation, the probe
samples are simulated from the known gravitational field; a physical or stricter
sensor-limited experiment would need an accelerometer array, small real probe
maneuvers, or a learned local field approximation.

**Local Kinetic Energy**
The test particle can measure its own velocity and compute its kinetic energy without privileged access to the other bodies.

**Acceleration Magnitude**
The test particle measures its own acceleration directly, providing a scalar indicator of total gravitational field strength at its location.

### Phase 3: Scan/Seek/Track Controller

With local proxy signals scaffolded, Phase 3 implements a prototype Sundog
control pattern:

**SCAN**: Perturb with small maneuvers and estimate local response of the indirect signal. Apply periodic thrust perturbations and observe how tidal magnitude responds.

**SEEK**: Move toward regions where the signature indicates stability, capture, or lower escape risk. Apply thrust to reduce tidal stress when it exceeds target threshold.

**TRACK**: Maintain the desired signature using extremum seeking or gradient descent without privileged full-state feedback. Continuously adjust thrust to hold tidal magnitude at target value.

The controller acts on tidal magnitude and local tidal-gradient estimates. The
current implementation should be read as a controller prototype, not yet as
evidence that sensor-only control maintains an orbit family or outperforms a
baseline.

## The Boundary

### What This Is

- A demonstration that indirect dynamical signatures respond coherently to changes in system state
- A prototype and batch harness showing how locally motivated proxies (tidal
  tensor, local acceleration) can be wired into control-relevant signals
- A mapped near-escape operating-envelope result for guarded accelerometer-proxy
  TRACK control
- A real-time browser visualization showing the coupling between controller action and signature response

### What This Is Not

- A solution to the three-body problem
- A claim that compressed observables eliminate chaotic sensitivity
- A prediction engine for long-term three-body evolution
- A general stability controller for arbitrary three-body configurations

### Known Failure Boundaries

The Sundog controller loses its gradient signal in the same way the photometric controller loses its signal when the optimum is outside the reachable workspace:

- **Strongly chaotic scattering regime**: When all three bodies are at comparable mutual distances (no hierarchy), indirect signals become noisy and the coupling between control action and observable weakens.
- **Insufficient thrust authority**: If tidal gradients exceed available thrust, the controller cannot maintain target signature.
- **Singular configurations**: Near-collisions or ejection events may evolve faster than controller response time.

## The Honest Hook

Three-body dynamics are chaotic. Sundog asks whether the shadow of that chaos is enough to steer by.

The working hypothesis: sometimes yes, within a bounded operating envelope
where:
1. The system maintains some degree of hierarchy (not fully democratic scattering)
2. Thrust authority is sufficient relative to tidal gradients
3. Events evolve at timescales comparable to or slower than controller response

This is not a universal theorem. The current evidence supports a bounded
near-escape pocket, not general three-body control. Phase 11 shows the pocket is
not an artifact of a single guard quantile, but also confirms that lower
velocity and equal-mass boundary cells remain the first regions where the
controller should not be used.

## Connection to Photometric Alignment

The three-body experiment maps cleanly onto the existing Sundog pattern:

1. **Deny full-state access**: Test particle cannot see primary positions or masses
2. **Expose only indirect observables**: Tidal tensor, local acceleration, own velocity
3. **Transform observables into control-relevant signatures**: Tidal magnitude as stability proxy
4. **Compare against privileged baseline**: Phase 11 now includes a matched
   privileged heuristic oracle comparison, while keeping clear that this oracle
   is not an optimal controller
5. **Report the failure boundary**: current maps expose low-velocity
   near-escape harms and controller-shortened passive survival cases

Where the photometric experiment achieved comparable terminal accuracy with
slower convergence, the three-body project now tests a harder transfer: useful
action from a local proxy in a chaotic dynamical setting, with the operating
envelope and failures made visible.

## Interactive Demonstration

The browser visualization at `threebody.html` implements:

- Real-time RK4 integration of a planar restricted three-body-like setup
- Canvas rendering of two primaries and test particle with orbital trails
- Indirect signature overlays (virial ratio, inertia tensor, system energy) as live metrics
- Phase 2 sensor mode toggle: compute signatures from local measurements only
- Phase 3 controller modes: SCAN/SEEK/TRACK with adjustable thrust authority
- UI controls for initial conditions, system parameters, and signature visibility
- Visual design consistent with site branding (gold accents, dark backgrounds)

The visualization serves as both demonstration and research tool: it allows exploration of which initial conditions and control parameters lead to stable tracking vs. controller saturation vs. escape.

## Status

**Phase 1 (Diagnostic Benchmark)**: Instrumented, not complete. Privileged
signatures are implemented and visualized. Initial event labels and tidal
warning metrics now exist in the harness, but diagnostic comparisons against
naive warning scores are still pending.

**Phase 2 (Sensor-Available Proxies)**: Scaffolded, not validated. Tidal tensor
estimation is implemented through simulated local probe samples; stricter sensor
validation is pending.

**Phase 3 (Scan/Seek/Track Controller)**: Prototype implemented. Controller
modes exist, and matched smoke-slate comparison against passive, naive, oracle,
and tidal-signal ablation baselines has started. This is calibration work, not a
validated performance result.

**Phase 4 (Public Artifact)**: In progress. Browser visualization complete. This writeup documents claim boundaries. Sensor model audit below separates privileged diagnostics from valid indirect signals.

**Phase 5 (Reproducible Harness)**: Complete for the prototype tier.
`npm run threebody:phase5` produces a deterministic smoke batch under
`results/threebody/phase5-smoke/` with a manifest and per-trial JSONL logs. The
harness records state, exposed signatures, tidal proxy values, thrust vectors,
event labels, and terminal outcomes. The browser visualization now imports the
same shared dynamics module as the harness, so initialization, integration,
signature computation, tidal proxy computation, and prototype controller thrust
use one equation source.

**Phase 6 (Baseline Set)**: Started. `npm run threebody:phase6` runs a matched
smoke slate across passive/no-control, naive local acceleration control,
SCAN-only, SEEK, TRACK, noisy and shuffled tidal-signal ablations, and a
privileged full-state lookahead guard. The output
includes per-trial logs, `summary.csv` grouped by regime and controller mode,
`paired.csv` with same-seed deltas against passive and oracle rows, and
`comparison.csv` with aggregate improvement/tie/worsening counts. The current
Phase 6 artifacts are calibration data only: the privileged baseline is
heuristic, the seed count is small, and no public performance claim should be
made from this slate.

**Phase 7 (Event Metrics)**: Started. `npm run threebody:phase7` runs the same
matched controller slate and writes `event-metrics.csv` under
`results/threebody/phase7-event-metrics/`. The current diagnostic defines a
hazard as escape or close approach, a warning as a tidal spike, and a positive
sample as one that falls within the configured one-second warning horizon before
the hazard. The output reports first hazard time, first warning time, lead time,
sample confusion counts, precision, recall, F1, false-alarm rate, and AUROC for
tidal magnitude and local gravitational acceleration magnitude. The latter is
the naive local warning baseline. The harness also writes
`threshold-sweep.csv`, `threshold-summary.csv`, and `calibration.csv` with a
`scoreType` column for threshold tradeoff and binned risk inspection. These
metrics are calibration data only: small-slate and label-degenerate regimes can
produce blank AUROC or false-alarm values, and calibration bins may be
non-monotone.

**Phase 8 (Sensor-Model Validation)**: Started. `npm run threebody:phase8`
runs the matched slate with sensor-tier tidal tensor audits and writes
`sensor-model-samples.csv` plus `sensor-model-summary.csv` under
`results/threebody/phase8-sensor-model/`. The current sensor tiers are exact
simulated local probes, noisy accelerometer-array samples, delayed local probes,
and noisy micro-maneuver proxies. This is not yet a sensor-only controller
claim: it measures proxy error against the simulated reference and keeps the
exact virtual probe labeled as a reference tier.
Phase 8 also includes sensor-tier controller modes for SEEK/TRACK using noisy
accelerometer-array, delayed local-probe, and noisy micro-maneuver tidal
estimates. The first smoke slate suggests the accelerometer-array tier is much
closer to ideal behavior than delayed or contaminated micro-maneuver control,
but this remains calibration work.
`npm run threebody:phase8:calibrate` now runs the next calibration sweep: it
varies accelerometer-array noise, delayed-probe latency, and micro-maneuver
noise/latency/contamination, then writes paired passive-baseline comparisons,
aggregate survival rows, and sensor-error summaries under
`results/threebody/phase8-calibration-sweep/`. The first default sweep emitted
1,044 trials. Its useful result is not a win claim; it is an operating-envelope
map. Latency hurts delayed and micro-maneuver tiers quickly, accelerometer-array
error grows with noise, and any apparent survival gain under noisier settings
needs a larger seed slate before interpretation.
`npm run threebody:phase8:analyze` reduces those rows into
`envelope.csv`, `regime-envelope.csv`, and `candidate-envelope.csv`.
`npm run threebody:phase8:focus` reruns the accelerometer tier with 12 seeds.
That focused run now emits 960 trials across SEEK, TRACK, and guarded TRACK.
The 3-seed hint for ungated `track_sensor_accel` at noise `0.03` did not
survive the larger slate. The guarded accelerometer TRACK variant did: at noise
`0` and `0.01`, it improves aggregate survival against the matched passive
baseline under the strict candidate filter. This is the current Phase 9
hypothesis, not a validated sensor-only result.

## Sensor Model Audit

### Privileged Signals (Full State Required)

These observables require positions, velocities, masses, or pair identity for all bodies:

- **Virial Ratio (2T/|W|)**: Requires all body velocities and pairwise distances to compute total kinetic and potential energy.
- **Inertia Tensor Trace**: Requires all body positions and masses relative to system center of mass.
- **Pairwise Energies**: Requires positions, velocities, and masses of both bodies in each pair.
- **System Energy (T + W)**: Requires all body velocities and pairwise distances.

**Use case**: These are diagnostic tools for analysis and oracle baseline comparison. They demonstrate that compressed signals forecast useful events, but they are not available to a sensor-limited controller.

### Sensor-Available Signals (Local Measurements Only)

These observables can be measured or estimated from the test particle's perspective without privileged state access:

- **Tidal Tensor (∂²Φ/∂x_i∂x_j)**: Estimated by comparing acceleration at the
  test particle position with acceleration at nearby probe points. In this
  implementation the probes are simulated. A stricter sensor model would need
  physical nearby samples, deliberate micro-maneuvers, or a learned field model.
- **Local Acceleration (a)**: Directly measurable from test particle's own motion (accelerometer or inertial measurement unit equivalent).
- **Local Velocity (v)**: Directly measurable from test particle's own state (proprioception or inertial navigation).
- **Local Kinetic Energy (½mv²)**: Computed from local velocity and test particle mass.
- **Thrust History**: Test particle knows its own commanded thrust (actuator feedback).

**Use case**: These are the signals available to the Sundog controller. All Phase 3 control decisions use only these measurements.

### Partially Observable Signals (Requires Sensors Beyond Local Measurements)

These observables could be made available with additional exteroceptive sensors:

- **Range/Range-Rate to Primaries**: Requires active ranging (e.g., laser rangefinder, radar) or passive triangulation.
- **Bearing-Only Observations**: Requires optical or radio sensors to detect primary positions without knowing distance.
- **Doppler Shift Measurements**: Could provide relative velocity information if primaries emit detectable signals.

**Status in current experiment**: Not implemented. The current Phase 2 demonstration uses only local/proprioceptive measurements (accelerometer-equivalent + virtual probe technique).

### Design Implications

The sensor model audit preserves claim defensibility:

1. **Privileged diagnostics** (virial, inertia, pairwise energy) demonstrate that compressed signals forecast useful events. They establish that the Phase 1 benchmark has predictive value.

2. **Sensor-motivated proxies** (tidal tensor, local acceleration) provide the
   current control input. They show a plausible route to reduced-state control,
   but do not yet prove sensor-only control performance.

3. **Comparison between Phase 1 and Phase 2 modes** should quantify the
   performance gap between privileged diagnostics and sensor-limited proxies;
   that quantitative comparison is still future work.

This separation keeps the experiment honest: if a compressed diagnostic works only with privileged state, it guides feature discovery but should not be presented as the Sundog controller's input.

## Future Directions

- **Phase 5, reproducible harness**: Run deterministic trial batches outside the
  browser animation loop, with seeded initial conditions and per-trial logs.
- **Phase 6, baselines**: Expand the current passive/no-control, naive local
  control, privileged lookahead, and SCAN/SEEK/TRACK matched slate into a larger
  baseline comparison. Current primary metrics are terminal outcome, simulated
  time, total delta-v, minimum primary distance, controller saturation count,
  and target-band loss count.
- **Phase 7, event metrics**: Expand the current tidal/local-acceleration
  warning threshold sweeps and calibration curves with reliability/error
  summaries and larger seed slates.
- **Phase 8, sensor-model validation**: Separate simulated local probes from
  micro-maneuver estimates, accelerometer-array estimates, and noisy learned
  local field surrogates. The current calibration sweep and focused
  accelerometer rerun do not justify a sensor-only superiority claim. The next
  step is an operating-envelope map for guarded low-noise accelerometer TRACK
  and a threshold calibration pass that replaces tuned guard constants with
  local hazard-score gates.
- **Phase 9, operating envelope map**: Sweep initial conditions, mass ratio,
  thrust authority, target tidal magnitude, sensor noise, delay, and timestep to
  map success, failure, and ambiguity regions. `npm run threebody:phase9` now
  runs the first map for guarded accelerometer TRACK. The initial smoke map
  emitted 1,944 trials and writes a primary `trial-outcomes.csv` plus
  `envelope-map.csv`, `aggregate-envelope.csv`, `best-by-cell.csv`, and
  `candidate-envelope.csv` under
  `results/threebody/phase9-operating-envelope/`. The first positive pocket is
  near-escape; stable cells are mostly neutral because passive already survives,
  and chaotic cells are mostly neutral or risky.
  `npm run threebody:phase9:refine` now runs a denser 7x7 near-escape
  radius/velocity map under `results/threebody/phase9-refined-near-escape/`.
  That refined map emitted 2,352 trials and adds `cell-class-map.csv` plus
  `cell-delta-map.csv` for reading the operating pocket directly. The strongest
  connected positive region is moderate-to-high near-escape velocity, especially
  above velocity scale `1.05`; larger-radius low-velocity cells form the first
  crisp local failure boundary. The primary CSV now labels each trial as helped,
  hurt, tied, time-helped, or time-hurt versus passive and adds a
  `failure_mechanism` column. In the refined run, harmful rows are mostly
  `controller_destabilized_or_shortened_passive`, with a smaller
  `control_effort_or_saturation` bucket; the current failure boundary is not
  primarily a sensor-noise-floor story.
  `npm run threebody:phase9:lock` repeats the refined near-escape map with 24
  seeds, emitting 9,408 trials under
  `results/threebody/phase9-locked-near-escape/`. The pocket survives the
  larger slate: 41 best cells are promising, 7 are mixed, and 1 is negative.
  The strongest rows remain larger-radius, higher-velocity near-escape cells.
  `npm run threebody:phase9:axes` then probes that locked high-velocity pocket
  across mass ratios `0.01`, `0.3`, and `1` and timesteps `0.008`, `0.01`, and
  `0.012`. The scoped run emitted 1,296 trials under
  `results/threebody/phase9-mass-timestep/` and kept 72 candidate rows out of
  81, suggesting the pocket is not merely an equal-mass/default-timestep
  artifact inside the tested slice.
  `npm run threebody:phase9:hazard` replaces the hand-tuned guard constants in
  that same slice with passive quantile-derived hazard-score gates. The scoped
  hazard run emitted 1,296 trials under
  `results/threebody/phase9-hazard-gates/`, kept 81 candidate rows out of 81,
  produced no harmful trial rows, and reduced average controller delta-v versus
  the constant-gate axes probe. The next check is to run those derived gates
  across the full refined near-escape grid.
  `npm run threebody:phase9:hazard:refine` runs that full-grid check. It emitted
  2,352 trials, kept 100 candidate rows out of 196 versus 116 for the
  constant-gate refined run, and reduced average controller delta-v from about
  `2.07` to about `1.12`. The high-velocity pocket remains, but the low-velocity
  failure boundary remains too.
- **Phase 10, writeup and claim ratchet**: Update the public-facing story to
  reflect the Phase 9 result without inflating it. The earned wording is now
  that a guarded accelerometer-proxy TRACK controller improves survival over
  passive behavior in a connected near-escape operating pocket, while retaining
  explicit low-velocity failure boundaries. This phase should update this
  writeup, `threebody.html`, and the applications gallery with a clear "What
  changed after Phase 9" panel, and should make the failure map part of the
  main story rather than a caveat.
- **Phase 11, robustness and outside-pocket expansion**: Test whether the Phase
  9 pocket survives guard quantile changes and nearby parameter expansion. The
  next scripts should compare passive hazard-gate quantiles `0.5`, `0.75`, and
  `0.9`; expand mass-ratio/timestep probes outside the favorable high-velocity
  near-escape pocket; rerun passive, naive local, guarded accelerometer TRACK,
  and privileged heuristic oracle comparisons on the same slate; and summarize
  the diagnostic-to-control chain from Phase 7/8 through Phase 9. The result
  should identify both the robust positive pocket and the first outside-pocket
  region where the Sundog controller should not be used.
  The initial Phase 11 pass is complete. The guard-quantile sweep emitted 7,056 trials and
  shows the pocket survives quantiles `0.5`, `0.75`, and `0.9`. The
  outside-pocket sweep emitted 6,912 trials and confirms lower velocities plus
  equal-mass boundary cells carry most harms. The comparison slate emitted 2,592
  trials: in the favorable pocket, guarded accelerometer TRACK produced 81
  candidate rows out of 81, naive local control produced none, and the
  privileged heuristic oracle produced 34 of 81. See
  [THREEBODY_PHASE11_SUMMARY.md](THREEBODY_PHASE11_SUMMARY.md).
- **Phase 13, longer-horizon lock**: Test whether the Phase 11 pocket is
  durable over a longer rollout or mainly delays failures that still arrive
  shortly after the original 8-second window. The first command,
  `npm run threebody:phase13:smoke`, runs a tiny 16-second
  pocket-plus-boundary slate. If that smoke passes, `npm run threebody:phase13`
  is the staged evidence run across mass ratio, timestep, radius, velocity,
  passive/naive/guarded/oracle modes, and eight seeds. The claim ratchet is
  pre-registered: pass means "longer tested horizon inside the mapped pocket,"
  partial means "bounded survival/delay benefit," and fail means controller
  redesign before larger sweeps. The smoke has now run: it kept the
  larger-radius high-velocity cell promising at 16 seconds, left one
  high-velocity cell neutral because passive already survived, and confirmed
  the low-velocity boundary as risky. See
  [THREEBODY_PHASE13_SUMMARY.md](THREEBODY_PHASE13_SUMMARY.md).
- **Spatial extension**: Extend from planar to full 3D three-body problem.
- **General three-body**: Move beyond circular restricted problem to include test particle mass and general initial conditions.
- **Orbit family maintenance**: Demonstrate maintenance of specific orbit families (halo orbits around Lagrange points, periodic Lyapunov orbits) using only indirect signatures.

## Conclusion

The three-body experiment is not a solution to celestial mechanics. It is an application of the Sundog pattern to a high-recognition physics problem.

The current result: locally motivated proxies can drive a guarded
scan/seek/track-style controller usefully inside a measured high-velocity
near-escape pocket. That pocket survives a larger seed slate, a scoped
mass-ratio/timestep probe, Phase 11 guard-quantile variation, and matched
comparison against naive local and privileged heuristic baselines, but it does
not erase the low-velocity or equal-mass boundary harms. Phase 13 now asks the
next necessary question: does that pocket persist when the rollout is doubled,
and at what control-effort cost?

The target trade is the same as photometric alignment: useful action from
partial information, with known failure modes and a measurable cost. The honest
claim today is not "Sundog solves three-body," but "in a mapped restricted
three-body pocket, the shadow of chaos is sometimes enough to steer by."

---

**Interactive visualization**: [threebody.html](../threebody.html)
**Full roadmap**: [SUNDOG_V_THREEBODY.md](SUNDOG_V_THREEBODY.md)
**Repository**: [github.com/humiliati/sundog](https://github.com/humiliati/sundog)
