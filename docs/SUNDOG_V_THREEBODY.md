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

## Recommendation

Proceed, but make the first milestone diagnostic rather than controller-first.
The idea is promising because it gives Sundog a high-recognition physics hook
without needing a grandiose claim. The research discipline is to earn three
things in order: first predictive signatures, then sensor-available proxies,
then control.
