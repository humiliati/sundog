# Three-Body Cross-Substrate Notes

This note translates the Mesa -> Geometry crossover discipline into future
Threebody work. It is not evidence by itself. It is a design guardrail for
turning "Sundog works across substrates" into testable phase specs instead of
theme-matching.

## 1. What Transfers

Mesa and Geometry both converged on field-shaped structure:

- Mesa: a behavioral cliff localized to an entangled 5D `net.7` activation
  subspace, with direction-specific neuron substrates and failed linear
  decompositions.
- Geometry: a small set of shared halo generators whose visible arcs are
  forward-rich but inverse-narrow, with HaloSim acting as an independent
  forward oracle.
- Threebody: a local accelerometer/tidal proxy that improves outcomes only in a
  mapped near-escape pocket, with lower-velocity and equal-mass boundaries where
  control can actively hurt.

The shared theorem posture is:

> Useful action can emerge from indirect signatures when the signature remains
> attached to the field, but the attachment is substrate-specific, asymmetric,
> and bounded by failure maps.

## 2. Mesa Lessons for Threebody

**Do not collapse variance, salience, and mechanism.** In Threebody terms, a
diagnostic can have high AUROC, a large tidal spike, or obvious visual movement
without being the causal control handle. Future specs should separate:

- warning quality: does a signal forecast escape or close approach?
- action coupling: does thrust change that signal in the intended direction?
- outcome effect: does the controller improve survival against passive and
  naive baselines?

**Expect direction-specific substrates.** Mesa's basin-inducing and
basin-resisting directions did not reduce to one symmetric mechanism. Threebody
should treat "move away from hazard" and "maintain bounded near-escape" as
different control problems, not as sign-flipped versions of one policy.

**Honor smoke-gate negatives.** If a boundary cell fails, stop the branch and
record the failure instead of widening the sweep to average it away. Phase 13's
full lock passed the high-velocity horizon test, but all risky/negative best
cells stayed in the `velocityScale=0.95` band, with all 5 negative cells at mass
ratio `1`. That is exactly the kind of boundary Mesa taught us to keep visible.

## 3. HaloSim / Geometry Lessons for Threebody

**Use forward oracles to discipline inverse handles.** HaloSim helped Geometry
reject a tempting manual tangent-arc fit by checking it against the canonical
forward ray-traced curve. Threebody's analogue is to introduce higher-fidelity
or more privileged forward checks for candidate control handles before promoting
them into public claims.

Candidate forward-oracle checks:

- Compare the accelerometer/tidal proxy against a privileged local phase-space
  oracle on the same seeds.
- Run short high-precision integration checks on winning cells to ensure the
  pocket is not a timestep artifact.
- Add a "canonical curve" analogue: for each cell, estimate passive hazard
  quantiles first, then freeze them before controller comparison.
- For future 3D builds, keep a validation render/trajectory oracle separate
  from the public interactive surface, the same way Geometry separates HaloSim
  from the atlas.

**Forward-rich / inverse-narrow is a good warning label.** Geometry can generate
many halo primitives from sun altitude, but only one inverse handle currently
survives the strict photo gate. Threebody may likewise have many signals that
respond to the field, while only one or two are useful for control. Future
phase specs should rank signals by validated control utility, not by how
physically evocative they are.

## 4. Candidate Future Phases

### Phase 14 - Mechanism Decomposition and Action Coupling

Question: is guarded TRACK winning because the accelerometer/tidal signal is
the right causal handle, or because the guard happens to suppress bad thrust in
the tested pocket?

Suggested checks:

- Freeze the Phase 13 winning guard and compare against action-shuffled,
  signal-shuffled, delayed-signal, and sign-flipped controls.
- Report warning quality, action coupling, and outcome effect as separate
  columns.
- Pre-register a negative branch where high warning quality with weak action
  coupling does not support a controller claim.

### Phase 15 - Forward-Oracle / Precision Lock

Question: does the positive pocket survive stricter numerical and privileged
forward checks?

Suggested checks:

- Rerun winning and boundary cells at smaller timesteps.
- Add energy-drift and integration-error receipts.
- Compare accelerometer-proxy TRACK against a privileged local phase-space
  oracle that is explicitly labeled as an oracle, not a deployable controller.
- Preserve the same passive hazard-quantile gates across comparisons.

### Phase 16 - 3D / Spatial Extension

Question: does the signature-action coupling survive the first spatial
extension, or is it a planar artifact?

Suggested checks:

- Start with the Phase 13 winning high-velocity cells only.
- Add one spatial perturbation axis at a time.
- Keep the Phase 13 planar result as the baseline and report degradation, not
  just absolute survival.
- Use a smoke gate before any broad 3D sweep.

### Sidecar - 3D Catalog Isotrophy / Symmetry Descent

Question: can Sundog make a non-tautological first-principles prediction over
an external three-body catalog, before writing another controller sweep?

Suggested checks:

- Use Li-Liao 2025's 21 equal-mass 3D choreographies and 273 piano-trio orbits
  as an external catalog, not as a Phase 13-15 continuation.
- Treat the published fixed ansatz as the crystal cut: it enforces the concrete
  beta-class residual `Z2` called `F_beta`, so the count test is
  facet-conditioned rather than an unconditioned sample over all 3D orbit
  space.
- Classify each choreography by residual spacetime `Z2` generators that survive
  the `S3 -> Z2` mass perturbation.
- Compare the predicted daughter-family count against piano-trio families
  clustered across the `m3 = 0.1*n` grid.
- Treat mismatch as a theorem-sharpening result, not a failed controller
  experiment.

Workbench:

- [`../sundog_v_isotrophy.md`](../sundog_v_isotrophy.md)
- [`../isotrophy/files.math`](../isotrophy/files.math)

## 5. Documentation Shape

Follow the Mesa pattern:

- Keep `../SUNDOG_V_THREEBODY.md` as the narrative spine and claim ledger.
- Put implementation-grade phase locks in `docs/threebody/PHASE*_SPEC.md`.
- Put run receipts and claim impact in `docs/threebody/PHASE*_RESULTS.md`.
- Keep cross-substrate transfer notes here unless they become a concrete phase.

The parent roadmap should summarize decisions and link out. It should not carry
every command, threshold, result table, and readback template once a phase has
graduated to implementation.
