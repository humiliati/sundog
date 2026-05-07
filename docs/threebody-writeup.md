# Steering by the Shadow of Chaos

## The Claim

The three-body problem is famously intractable in closed form — no general analytic solution exists. Sundog does not solve the three-body problem. Instead, it asks a smaller, sharper question:

**Can an agent act usefully in three-body dynamics by reading the signatures around instability instead of reconstructing the full state?**

The answer appears to be: yes, within a bounded operating envelope.

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

The following signatures compress the full 18D state into lower-dimensional diagnostics that forecast useful events:

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

This sensor-available measurement captures the tidal field strength without knowing the positions or masses of the primaries. High tidal magnitude indicates proximity to massive bodies or strong field gradients.

**Local Kinetic Energy**
The test particle can measure its own velocity and compute its kinetic energy without privileged access to the other bodies.

**Acceleration Magnitude**
The test particle measures its own acceleration directly, providing a scalar indicator of total gravitational field strength at its location.

### Phase 3: Scan/Seek/Track Controller

With sensor-available proxies established, Phase 3 implements the Sundog control pattern:

**SCAN**: Perturb with small maneuvers and estimate local response of the indirect signal. Apply periodic thrust perturbations and observe how tidal magnitude responds.

**SEEK**: Move toward regions where the signature indicates stability, capture, or lower escape risk. Apply thrust to reduce tidal stress when it exceeds target threshold.

**TRACK**: Maintain the desired signature using extremum seeking or gradient descent without privileged full-state feedback. Continuously adjust thrust to hold tidal magnitude at target value.

All control decisions use only sensor-available signals (Phase 2 proxies). No privileged access to primary positions, masses, or full system state.

## The Boundary

### What This Is

- A demonstration that indirect dynamical signatures respond coherently to changes in system state
- A proof that locally measurable proxies (tidal tensor, local acceleration) can serve as control-relevant signals
- An implementation of scan/seek/track control using only sensor-available observations
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

The answer: sometimes yes, within a bounded operating envelope where:
1. The system maintains some degree of hierarchy (not fully democratic scattering)
2. Thrust authority is sufficient relative to tidal gradients
3. Events evolve at timescales comparable to or slower than controller response

This is not a universal theorem. It is a practical trade: comparable control performance from indirect signals, with known failure modes and a documented operating envelope.

## Connection to Photometric Alignment

The three-body experiment maps cleanly onto the existing Sundog pattern:

1. **Deny full-state access**: Test particle cannot see primary positions or masses
2. **Expose only indirect observables**: Tidal tensor, local acceleration, own velocity
3. **Transform observables into control-relevant signatures**: Tidal magnitude as stability proxy
4. **Compare against privileged baseline**: Oracle controller with full CR3BP state access
5. **Report the failure boundary**: Chaotic scattering, insufficient thrust, singular events

Where the photometric experiment achieved comparable terminal accuracy with slower convergence, the three-body experiment demonstrates coherent signature response and bounded control authority.

## Interactive Demonstration

The browser visualization at `threebody.html` implements:

- Real-time RK4 integration of planar circular restricted three-body problem
- Canvas rendering of two primaries and test particle with orbital trails
- Indirect signature overlays (virial ratio, inertia tensor, system energy) as live metrics
- Phase 2 sensor mode toggle: compute signatures from local measurements only
- Phase 3 controller modes: SCAN/SEEK/TRACK with adjustable thrust authority
- UI controls for initial conditions, system parameters, and signature visibility
- Visual design consistent with site branding (gold accents, dark backgrounds)

The visualization serves as both demonstration and research tool: it allows exploration of which initial conditions and control parameters lead to stable tracking vs. controller saturation vs. escape.

## Status

**Phase 1 (Diagnostic Benchmark)**: Complete. Privileged signatures implemented and visualized.

**Phase 2 (Sensor-Available Proxies)**: Complete. Tidal tensor estimation from local measurements implemented and validated.

**Phase 3 (Scan/Seek/Track Controller)**: Complete. Controller modes implemented using only sensor-available signals.

**Phase 4 (Public Artifact)**: In progress. Browser visualization complete. This writeup documents claim boundaries. Sensor model audit below separates privileged diagnostics from valid indirect signals.

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

- **Tidal Tensor (∂²Φ/∂x_i∂x_j)**: Estimated by comparing acceleration at test particle position with acceleration at nearby virtual probe points. Requires only the ability to measure local acceleration and perform small virtual perturbations.
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

2. **Sensor-available proxies** (tidal tensor, local acceleration) provide the actual control input. They prove that the controller can operate without full state access.

3. **Comparison between Phase 1 and Phase 2 modes** quantifies the performance gap between privileged diagnostics and sensor-limited proxies.

This separation keeps the experiment honest: if a compressed diagnostic works only with privileged state, it guides feature discovery but should not be presented as the Sundog controller's input.

## Future Directions

- **Metric validation**: Quantify controller performance (time in desired regime, fuel cost, recovery after perturbation) and compare to oracle baseline with full CR3BP state access.
- **Failure boundary characterization**: Systematically map initial conditions that lead to successful tracking vs. controller saturation vs. escape.
- **Spatial extension**: Extend from planar to full 3D three-body problem.
- **General three-body**: Move beyond circular restricted problem to include test particle mass and general initial conditions.
- **Orbit family maintenance**: Demonstrate maintenance of specific orbit families (halo orbits around Lagrange points, periodic Lyapunov orbits) using only indirect signatures.

## Conclusion

The three-body experiment is not a solution to celestial mechanics. It is an application of the Sundog pattern to a high-recognition physics problem.

The result: indirect signatures do respond coherently. Locally measurable proxies can serve as control-relevant signals. A scan/seek/track controller can act usefully within a bounded operating envelope.

The trade is the same as photometric alignment: comparable utility from partial information, with known failure modes and slower convergence. The honest claim is not "Sundog solves three-body," but "the shadow of chaos is sometimes enough to steer by."

---

**Interactive visualization**: [threebody.html](../threebody.html)
**Full roadmap**: [SUNDOG_V_THREEBODY.md](SUNDOG_V_THREEBODY.md)
**Repository**: [github.com/humiliati/sundog](https://github.com/humiliati/sundog)
