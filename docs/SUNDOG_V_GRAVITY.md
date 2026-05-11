# Sundog Gravity Ledger

Working hook:

> Sundog is gravity for agents under partial observability: a geometry-derived
> background field that deflects policy without ever being a target.

Sundog Gravity is the staging ledger for the most outlandish public claim in
the program — that signature-driven control sidesteps Goodhart's law because
the agent is moving through a field rather than optimizing a metric — and for
the high-cost horizon experiments that could falsify it.

This document is not a roadmap. It is a holding pattern for ambition. Each
candidate experiment listed below would, if run, either ratchet the gravity
claim into earned language or push it back to the "unsupported universal" pile
in `presentation/claims-and-scope.md`. None of them have run yet.

The closest existing anchors are the photometric mirror-alignment experiment in
this repo and the three-body workbench operating-envelope result. Both
demonstrate the structural shape — indirect signal, transformed signature,
bounded control — without making the gravity claim. This ledger is where the
claim is staged for later, expensive defense.

## Claim Boundary

This document does **not** claim that Sundog has demonstrated reward-hacking
immunity, adversarial robustness, or general agent safety under hostile
conditions. It claims that:

1. there is a coherent structural argument — the Goodhart sidestep — for why
   signature-driven control should differ from metric-driven optimization in
   adversarial settings;
2. that argument is currently defended by analogy and by the bounded three-body
   operating-envelope result, not by adversarial benchmarks;
3. the experiments that would test the argument are expensive enough that they
   need to live in a ledger before they live in a roadmap.

If a candidate below is promoted into a full roadmap document like
`SUNDOG_V_BALANCE.md` or the three-body roadmap, it leaves this ledger.

## The Gravity Claim

Stated as a hook:

> Reinforcement learning gave agents a reward to optimize. Sundog gives them a
> field to fall through. The first invites Goodhart. The second has no metric
> to corrupt.

Sister formulations:

> An agent that reads tidal gradients cannot reward-hack gravity. The mass is
> where the mass is.

> Optimization made the agent face the target. Sundog makes the target part of
> the geometry the agent moves in.

Stated as a structural argument:

A reward-trained agent and a signature-trained agent inhabit different threat
models. Reward optimization places the objective inside the agent: a learned
value function, a critic, a numerical target the agent can corrupt by finding
inputs that score high without satisfying the designer's intent. Signature
control places the objective inside the environment's geometry. The tidal
tensor at a point is a function of the masses, not the agent's policy. The
shadow centroid is a function of the light source and the body, not the
controller. The agent is deflected by the field, not pulled toward a score.

The Sundog claim, in its most ambitious form, is that this structural
difference matters. That an agent moving through a field cannot reward-hack the
field the way an agent staring at a metric can corrupt the metric.

## The Goodhart Sidestep

The original unhinged intuition of the Sundog program — preserved in early
manifesto language and only recently reconnected to the controlled work — was
that indirect-signal control should be structurally less susceptible to
specification gaming than direct-reward control. That intuition was set aside
during the discipline pass because the program had no controlled result to
attach it to. The three-body operating-envelope result and the photometric
mirror-alignment experiment provide enough scaffolding to put it back on the
table as a research target rather than a slogan.

The mechanism, named carefully:

- A target metric `R(s, a)` is a function the agent participates in. Mesa
  optimization, reward-model exploitation, and proxy collapse are all variants
  of the same failure: the agent finds policies that score high on `R` while
  failing the designer's actual objective.
- A signature observation `S(x)` is a function of environmental state `x`
  alone. The agent reads `S` and selects actions whose effect on `S` is
  favorable by a fixed geometric definition (move toward lower tidal
  magnitude, hold shadow centroid in band, ride the pressure gradient). The
  agent does not select `S`.
- For Goodhart to bite, the agent must have a path to alter the measurement
  itself. In `R(s, a)` that path is built into the definition. In `S(x)`,
  altering the measurement requires altering the environment's geometry, which
  is expensive or impossible.

This is not a proof of adversarial robustness. It is a structural argument for
why the threat model differs. The experiments below are designed to test
whether that structural difference produces a measurable behavioral
difference.

## The Three-Body Wedge

The three-body workbench is the audience-conceptualizable entry point for the
gravity claim because the gravity is literal. A controller reads the tidal
tensor — the local second derivative of a real gravitational potential — and
maintains regime in a high-velocity near-escape pocket. The audience does not
need to grant a metaphor. They watch a controller fall through real gravity
without being told where the masses are, and they understand on first contact
what is meant by "the field is the objective."

The wedge is rhetorical: lead public communication with the three-body
controller, then say "now imagine this is what we mean by 'gravity for agents'
in every other partially-observed environment we name."

The wedge is also methodological. The three-body sensor-tier discipline —
privileged versus accelerometer-proxy versus delayed versus
micro-maneuver — is the template for how each candidate experiment in this
ledger must separate the signature from the simulator state. Without that
separation, a "signature" quietly becomes a "reward" wearing a costume, and
the Goodhart sidestep collapses.

## Falsification Surface

The gravity claim breaks if any of the following are demonstrated:

1. **Field manipulation is cheap.** An adversary can shape the indirect
   signature itself (rearrange masses, paint a fake shadow, jam the tidal
   proxy) cheaply enough to steer a signature-driven controller into a chosen
   bad regime at lower cost than steering an equivalently-trained reward
   agent.
2. **The signature is a reward in costume.** Any candidate signature `S(x)`
   can be decompiled into a target-equivalent scalar that the agent is
   effectively optimizing, restoring the full Goodhart threat model.
3. **Mesa-optimization re-emerges.** A signature-driven agent, trained or
   selected at sufficient scale, develops internal optimization over a learned
   reward proxy and inherits the same failure modes.

Mode (2) is the most dangerous because it is reachable by hostile review
without running an experiment. The defense is geometric: a signature is
acceptable under this ledger only if its value at a point can be written as a
function of environmental state alone, with no dependence on agent policy.

Each candidate experiment below must name which of (1), (2), or (3) it is
attacking.

## Evaluation Criteria

A horizon experiment earns a place in this ledger if it satisfies most of the
following:

- **Real-cost target:** the experiment is expensive enough that the gravity
  claim cannot be dismissed as cheap. A speculative claim that costs a weekend
  to test is a slogan, not a target.
- **Falsifiable in finite time:** the experiment can produce a result that
  ratchets the claim either up or down inside a defensible horizon.
- **Adversarial surface:** there is a named red team or named perturbation
  schedule, not just a passive baseline.
- **Sensor-tier discipline:** the signature is a function of environmental
  state, separable from the simulator's privileged state, in the same way the
  three-body workbench separates accelerometer-proxy tiers from oracle state.
- **Matched comparison:** at least one reward-trained or metric-driven baseline
  is run on the same slate under the same adversarial pressure.
- **Public legibility:** a non-expert can be told what was tested, what was
  measured, and what would have falsified the claim.

## Shortlist Recommendation

Current first-pass order of merit, by closeness-to-existing-discipline and by
the cost of getting from current Sundog work to a defensible adversarial
result:

1. **Spacecraft Trajectory Under Unmodeled Perturbation** — inherits the most
   three-body tooling; cleanest attack on falsification mode (1).
2. **Adversarial Signature Benchmark** — cheapest expensive experiment;
   cleanest attack on falsification mode (2).
3. **Side-Channel Defense (stretch)** — highest stir, longest horizon; would
   put the gravity claim inside a domain that visibly suffers from Goodhart in
   production.

---

## Candidate 1 - Spacecraft Trajectory Under Unmodeled Perturbation

Working hook:

> A controller that holds an orbit family by reading the tidal field cannot be
> spoofed by a falsified ephemeris.

### Why it is strong

The three-body workbench already establishes the sensor-tier discipline and
the operating-envelope shape. The spacecraft variant extends the same harness
into a regime where adversarial perturbation is naturally available: unmodeled
third-body effects, solar-radiation pressure, RF-jamming-degraded GPS, or
deliberately falsified position telemetry. The signature — local tidal field —
is structurally inaccessible to a typical sensor-jamming threat model because
rearranging the gravitational geometry of the solar system is not on the
adversary's action surface.

This makes it a clean attack on falsification mode (1): if the field cannot be
cheaply shaped, signature-driven control should retain regime longer than
ephemeris-driven control under matched jamming.

### Why it is weaker

The engineering burden is significant. A high-fidelity GMAT or STK harness is
defensible but not cheap; a real cubesat deployment is a multi-year program.
The adversarial model also has to be carefully drawn — a jammer that can
degrade *both* the privileged ephemeris and the accelerometer-proxy is not
testing the gravity claim, it is testing sensor redundancy.

### Sundog variant

Construct a matched orbit-holding task in a high-fidelity simulator:

- **Baseline A:** privileged ephemeris controller, full state from the
  simulator's truth model.
- **Baseline B:** ephemeris controller fed through a degraded perception
  pipeline (GPS-equivalent with adversarial jamming or spoofing).
- **Sundog:** accelerometer-proxy guarded TRACK extending the three-body
  workbench architecture into the chosen spacecraft regime.
- **Adversary:** a named perturbation schedule — RF jamming, ephemeris
  spoofing, unmodeled solar-radiation events — applied to all controllers on
  matched seeds.

### Sundog expression

- **Hidden target:** orbit-family geometry under unmodeled perturbation.
- **Indirect signal:** local tidal tensor estimate from accelerometer-array or
  micro-maneuver proxy.
- **Transformation:** SCAN/SEEK/TRACK over tidal magnitude and direction, with
  guard quantiles inherited from the three-body Phase 11 hazard-gate result.
- **Actionable output:** delta-v command toward signature-favorable regime.
- **Failure boundary:** controller saturates if perturbation drives the tidal
  field outside the trained signature range, or if accelerometer noise exceeds
  the Phase 8 calibration envelope.

### Falsification target

Mode (1): field-manipulation cost. A null result is "ephemeris-based control
under jamming holds orbit at least as long as signature-based control on the
matched slate." A positive result is a measured time-to-regime-loss advantage
for the signature controller, with named conditions.

### Current recommendation

First-priority horizon experiment because it inherits the most existing
discipline and tooling. The closest path from current Sundog work to a
defensible adversarial result.

---

## Candidate 2 - Adversarial Signature Benchmark

Working hook:

> One team trains agents to spec-game. The other team trains agents to fall
> through a field. Score over a shared adversarial slate.

### Why it is strong

This is the cheapest expensive experiment. It does not require a spacecraft or
a high-fidelity orbital simulator. It requires a partially-observed gridworld
or continuous-control environment, a matched red team, and the discipline to
define a signature and a reward that are not the same function in disguise.

It is also the most direct attack on falsification mode (2): if a signature
can be decompiled into a reward, that decomposition should show up here under
adversarial pressure, because spec-gaming the decompiled-reward signature
should be exactly as easy as spec-gaming the matched reward.

### Why it is weaker

The risk is that the chosen environment is too toy to be persuasive. A
reviewer can always say "of course the signature controller held up — your
adversary had no path to the field." The environment design is the
load-bearing part of the experiment, and getting it wrong wastes the whole
run.

### Sundog variant

Build a harness in the style of the three-body operating-envelope runner:

- A partially-observed environment with a separable signature `S(x)` and a
  matched reward `R(s, a)` whose argmax under no adversary coincides with
  `S`'s argmax.
- Two agent families: signature-tracking (Sundog architecture) and
  reward-optimizing (matched RL baseline, comparable parameter count and
  sample budget).
- An adversary that can perturb (a) the observation channel, (b) the reward
  channel, and (c) the environment geometry, with named budgets per channel.
- Primary metric: rate of catastrophic policy failure under each adversary
  channel. Secondary metric: terminal performance under no adversary.

### Sundog expression

- **Hidden target:** designer-intended policy.
- **Indirect signal:** environmental signature `S(x)` separable from agent
  policy.
- **Transformation:** SCAN/SEEK/TRACK on signature gradient.
- **Actionable output:** action selected by signature ascent.
- **Failure boundary:** signature can be reshaped within adversary budget, or
  signature collapses to reward under decompilation.

### Falsification target

Mode (2): signature-is-reward-in-costume. If the signature agent and the
reward agent show indistinguishable failure rates under matched adversary
budgets, the Goodhart sidestep is not earning its keep.

### Current recommendation

Second-priority horizon experiment, but cheaper to start than Candidate 1. A
first-pass version could plausibly run inside existing repo tooling once the
environment is designed.

---

## Candidate 3 - Side-Channel Defense *(stretch)*

Working hook:

> The detector does not classify attacks. It reads the disturbance the system
> casts when something is trying to corrupt it.

### Why it is strong

High stir. Cybersecurity is an industry currently shipping reward-trained
classifiers that are observably being Goodharted in production by adversarial
example crafting and APT-style behavior masking. A signature-driven detector
that reads syscall residue, EM/timing side-channels, or network flow geometry
rather than labeled-attack examples puts the gravity claim inside a domain
where the failure mode is visible.

### Why it is weaker

Domain-expertise burden is high and the public is already saturated with "AI
for cybersecurity" pitches. The misread risk is severe: collapsing into "we
built better intrusion detection" loses the point entirely.

### Sundog variant

A monitored process exposes a signature derived from syscall sequence
geometry, timing residue, or network flow shape. The signature is defined
without reference to known attack labels. A signature-tracking controller
flags regime departures; a matched supervised baseline classifies on labeled
attack samples. An active red team crafts adversarial patterns against both.

### Sundog expression

- **Hidden target:** intent of the process running in the system.
- **Indirect signal:** geometric residue (syscall n-gram structure, timing
  distribution, flow autocorrelation).
- **Transformation:** signature departure from baseline regime.
- **Actionable output:** flag, throttle, escalate, abstain.
- **Failure boundary:** signature can be flattened by adversary at lower cost
  than the matched classifier can be evaded.

### Falsification target

Modes (1) and (2) jointly: in a domain where the adversary is unusually
sophisticated, both field-manipulation and signature-decompilation pressure
should be high. If the gravity claim is real, signature-driven detection
should outlast labeled-attack classifiers on a matched red-team slate. If
not, this is where the costume falls off.

### Current recommendation

Long-term aspiration, not first-wave. Listed here because the broadcast value
of a positive result would be disproportionate to the cost of ratcheting the
claim, and because the cybersecurity framing is the most natural translation
of the gravity claim into a domain where reviewers care.

---

## Promotion Guidance

A candidate leaves this ledger and earns a `SUNDOG_V_*.md` roadmap document
only when:

- the experiment design names a specific falsification mode from §Falsification
  Surface;
- the signature is structurally separated from privileged state in the
  three-body sensor-tier style;
- a matched reward-trained or metric-driven baseline is committed to;
- a named adversary or perturbation schedule is committed to;
- the boundary language in `presentation/claims-and-scope.md` is updated to
  reflect what the experiment, if completed, would ratchet.

Until those are in place, the gravity claim remains the program's most
outlandish published frame and the most carefully boundaried one. Public
communication may use it (see `PROMO_HIGHLIGHTS.md` §The Gravity Claim) but
must mark it speculative and link to this ledger and to the controlled
three-body and photometric results that anchor the analogy.

## Broadcast-Aligned Summary

For public communication, the gravity-family framing summarizes this way:

> The photometric experiment and the three-body workbench demonstrate that
> useful control is possible from indirect environmental signatures rather
> than from privileged state. The gravity claim is the structural argument
> for why this matters: a controller that tracks a property of the
> environment's geometry, rather than optimizing a designer-specified metric,
> inhabits a different threat model than reward-trained control. The
> experiments that would test this difference are listed in this ledger and
> are deliberately expensive. None has been run.
