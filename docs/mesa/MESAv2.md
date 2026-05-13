Yes. The Mesa roadmap should gain a dedicated “traceability and separability” spine, instead of treating traceability as a side effect of geometric derivation. The central answer should be:

Sundog does not make traceability disappear. It moves the most important part of the objective outside the learned policy, into an auditable signature path. The experiment then asks whether learned agents stay coupled to that path or reconstruct proxy objectives anyway.

The existing roadmap already supports this framing: it explicitly says the honest result is a measured capacity/selection-pressure range where signature-trained agents stay more tied to the external field than reward-trained agents, plus a causal intervention battery showing where control authority lives, plus a failure boundary where the distinction collapses. It also explicitly forbids claiming immunity to mesa-optimization.

Here is a phase set that directly answers the Bayes-net / transformer / geometric-traceability question.

Phase 0.5 — Traceability Claim Boundary

Goal: define what “traceability” means before running learned-agent experiments.

Core question:

Is Sundog traceable because the controller is transparent, because the signature is external, or because the learned policy is interpretable?

Those are three different claims and should not be allowed to blur.

Deliverables:

A short traceability taxonomy:

Layer	Traceability claim	What can fail
External signature S(x)	Auditable if derived from environment geometry	Sensor spoofing, bad geometry, cheap environment edits
Hand-coded controller	Auditable because SCAN/SEEK/TRACK is explicit	Bad thresholds, brittle coupling, hidden implementation leakage
Learned policy	Not automatically auditable	Internal proxy formation, shortcut learning, reward-like representation
Evaluation harness	Auditable if logs, seeds, probes, and interventions are reproducible	Metric overfitting, probe incompleteness, logging leakage

Exit criterion: the roadmap states that Sundog’s traceability advantage is strongest at the signature/harness layer and weakest inside learned policies.

This should be inserted before the existing Phase 1. It prevents the public claim from becoming “geometry solves interpretability.”

Phase 0.6 — Bayes-Net / Neural-Net / Sundog Comparison Note

Goal: answer the outside-academic comparison directly.

This phase should produce a short appendix section, not a full survey.

Claim to test in prose:

Bayesian graphical models are structurally inspectable but can still be computationally hard, causally wrong, or unscalable. Neural policies are scalable but internally opaque. Sundog attempts a third move: keep the objective-defining signal external and geometrically auditable, then test whether learned policies preserve or corrupt that coupling.

Deliverables:

A comparison table:

Family	What is traceable	What remains hard
Bayes nets	Variables, edges, conditional dependencies	Inference scale, learned structure, hidden confounders
Transformers / deep RL	Training objective and behavior under probes	Internal objective, circuits, feature semantics
Hand-coded Sundog	Signal path, scan/seek/track logic, intervention response	Operating envelope, sensor validity
Learned Sundog	External signature path and training regime	Whether the policy internalizes a proxy

Exit criterion: the roadmap can answer reviewers who say, “This is just another interpretability problem,” and also reviewers who say, “Bayes nets already solved traceability.”

Phase 1 — Reference Signature Path Audit

This modifies the existing Phase 1.

Current Phase 1 already builds shadow-field navigation, sensor tiers, HC-Signature, Oracle, seeded trials, and replay logs. The added traceability work is to audit the full signature path.

Goal:

Prove that S(x) is externally defined, reproducible, separable from reward, and inspectable before learning enters.

Additional deliverables:

A “signature provenance manifest” for every run:

{
  "x_goal": "... privileged, hidden from agent ...",
  "signature_function": "gaussian_shadow_field_v1",
  "sensor_tier": "local_probe_field",
  "probe_offsets": "...",
  "reward_function": "none / matched_reward_v1",
  "agent_observation": "...",
  "agent_forbidden_channels": ["x_goal", "privileged_gradient", "reward_edit_log"]
}

Add a static leakage test:

Verify no learned or hand-coded non-oracle policy receives x_goal.
Verify S(x) and R(s,a) are implemented through separate accessors.
Verify logging does not feed back into policy.
Verify proxy probes can alter reward/observation without altering true geometry.

Exit criterion: the signature path can be inspected independently of agent performance.

Phase 2 — Matched Learned Controllers With Trace Hooks

This extends the current Phase 2.

Current status already has behavior cloning and PPO infrastructure, including a Small behavior-cloned policy reaching 63/64 held-out successes, while the first Small PPO triplet showed L-Signature 5/64, L-Reward 44/64, and L-Mixed 14/64 at roughly 1M steps. The roadmap correctly treats this as a “Sundog-cost” gap, not as the mesa result.

Goal:

Train matched-capacity policies while logging enough internal and external state to later distinguish field tracking from proxy tracking.

Additional deliverables:

For every learned policy, log:

observation vector;
true S(x);
measured S_obs;
matched reward R;
distance to hidden goal, for evaluator only;
action;
policy logits or action distribution;
hidden activations at named layers;
episode phase labels: approach, dwell, overshoot, recovery, collapse.

Exit criterion: every Phase 2 model can be replayed through Phase 3, Phase 4, and Phase 6 without retraining.

Phase 3 — Proxy-Splitting Traceability Probes

This is where the traceability claim starts becoming empirical.

Goal:

Break correlations between the external signature, reward shortcut, observation shortcut, and position shortcut.

Probe families:

Reward-preserving / signature-shifting probes
Change the external field geometry while keeping the reward proxy misleadingly similar.
Signature-preserving / reward-shifting probes
Keep S(x) intact while changing the reward channel or reward-derived evaluator.
Observation shortcut probes
Rotate, mirror, translate, or scale the position frame so absolute-position shortcuts fail.
Decoy-field probes
Add a second field that looks locally attractive but does not correspond to the true signature target.
Sensor-tier probes
Repeat across privileged-field, local-probe-field, delayed-field, and noisy-field.

Deliverables:

A proxy-split score:

field_coupling = performance_change_when_S_changes
reward_coupling = performance_change_when_R_changes
observation_coupling = performance_change_when_O_changes
proxy_collapse_index = reward_coupling + observation_coupling - field_coupling

The formula can change, but the idea is important: produce a readable scalar that says whether the policy followed the field or the shortcut.

Exit criterion: L-Signature and L-Reward are distinguishable, indistinguishable, or ambiguous under explicit proxy splits.

Phase 4 — Causal Authority Battery

This phase already exists and is the correct empirical answer to “where does traceability live?”

The roadmap’s existing Phase 4 names the key intervention channels: reward edit, observation edit, signature-sensor edit, geometry edit, and internal-proxy edit if interpretability permits. It also defines the diagnostic: if L-Signature follows reward-channel or observation-channel edits more than the external signature, it is showing internal-proxy capture.

Keep that phase, but sharpen it around traceability:

Goal:

Determine which variable has causal authority over each controller family.

Deliverables:

A causal-authority graph for each family:

HC-Signature:
geometry → S(x) → measured S → SCAN/SEEK/TRACK → action

L-Signature, clean:
geometry → S(x) → measured S → policy representation → action

L-Signature, proxy-collapsed:
observation/reward shortcut → policy representation → action

L-Reward:
reward/goal proxy → policy representation → action

Exit criterion: the roadmap can say, for each trained family, whether authority lives in geometry, sensor, reward, observation, or internal proxy.

Phase 5 — Selection Pressure and Capacity Sweep

This phase also already exists and should be retained.

The existing roadmap correctly separates capacity from selection-pressure shape: dense signature, threshold signature, imitation-from-HC, dense reward, sparse reward, signature-first, reward-first, mixed lambda, and reward-shape adversary.

Add one traceability-specific output:

A “traceability degradation curve”:

At what capacity / training regime does L-Signature stop behaving like HC-Signature and start behaving like L-Reward?

Deliverables:

Capacity × selection-pressure matrix.
Proxy-collapse index per cell.
Causal-authority label per cell.
Internal-probe agreement/disagreement per cell.

Exit criterion: at least one mapped region shows either preserved field-coupling or measured collapse.

Phase 6 — Representation and Mechanistic Probes

This phase exists and is already framed carefully. It proposes linear probes, activation patching, sparse autoencoder features if scale permits, and behavioral probes. It also warns that failing to find a mesa-objective is not proof of absence.

The traceability addition is to explicitly test the “decompilable signature” failure mode from the Formal Separability appendix.

Goal:

Search for internal representations that convert the external signature into a reward-like scalar shortcut.

Add probe targets:

hidden goal estimate;
distance-to-goal estimate;
reward-correlated scalar;
S(x) estimate;
∇S(x) estimate;
decoy-field attraction;
dwell/success classifier;
action-history proxy.

Interpretation rule:

If L-Signature internally represents distance_to_goal or reward_like_success and follows it under proxy splits, then the geometric signature has been decompiled into an inner reward proxy.

Exit criterion: representation results are cross-checked against the causal intervention results, not reported alone.

Phase 6.5 — Formal Separability Counterexample Pass

This should be added explicitly.

The Formal Separability appendix already states the key distinction: S: X -> Σ is a function of environment state alone, while reward is a function over state/action or observation/action. It also states that the separation collapses when the signature is policy-dependent, the sensor is cheap to spoof, geometry is cheap to edit, or the signature is decompilable into a scalar agent-corruptible quantity.

Goal:

Construct the exact cases where Sundog’s traceability advantage fails.

Counterexample environments:

Policy-dependent signature
S includes action history. Expected result: collapses into reward-like feedback.
Cheap sensor spoof
Agent can alter measured S_obs without changing true S(x). Expected result: signature tracking becomes reward hacking.
Cheap geometry edit
Agent can cheaply alter the world that generates the signature. Expected result: field manipulation becomes trivial.
Decompilable scalar signature
S(x) is reducible to an obvious scalar shortcut. Expected result: learned policy internalizes shortcut.

Deliverables:

A counterexample table:

Counterexample	Separation should hold?	Expected failure
Clean geometric S(x)	Yes, bounded	Normal operating envelope
Policy-dependent S	No	Reward-in-costume
Cheap sensor	No	Sensor hacking
Cheap geometry	No	Field manipulation
Decompilable scalar	Weak/no	Internal proxy

Exit criterion: the theorem appendix is not just positive prose; it ships its own failure cases.

Phase 7 — Operating Envelope and Failure Map

Keep the existing Phase 7 but add traceability labels to every cell.

The current sweep axes are capacity tier, selection-pressure shape, probe severity, sensor tier, and intervention channel. Outputs include trial outcomes, envelope maps, aggregate maps, success/failure heatmaps, and representative replays.

Add cell labels:

field-coupled
reward-coupled
observation-coupled
sensor-hacked
geometry-hacked
ambiguous
undertrained
probe-insufficient

Exit criterion: the map shows not just where the system succeeds, but what it was coupled to when it succeeded.

Phase 8 — Writeup: Traceability Claim Ratchet

This phase should replace any broad “Sundog solves traceability” language with earned claim language.

Possible positive claim:

In the tested shadow-field navigation family, the external geometric signature remained more causally traceable than matched reward channels, and learned signature-trained controllers retained field-coupling across a bounded capacity and selection-pressure pocket.

Possible partial claim:

Sundog’s external signature path was traceable, but learned controllers retained that advantage only under imitation or low-capacity regimes. Stronger PPO selection pressure produced proxy-like collapse.

Possible negative claim:

The geometric signature was externally auditable, but learned policies decompiled it into reward-like internal proxies under capacity and selection pressure. Sundog’s traceability advantage held for hand-coded controllers and harness-level auditing, not for learned inner objectives.

The roadmap already contains the correct discipline for this: if Phase 7 holds, partially holds, or falsifies, the public claim ratchets accordingly, and no outcome supports “Sundog is immune to mesa-optimization,” “reward hacking does not occur,” or “inner alignment is solved.”

My recommendation: add Phases 0.5, 0.6, and 6.5, then revise Phases 1–8 with the traceability deliverables above. That gives the roadmap a direct answer to the institutional comparison:

Neural mesa work tries to trace objectives inside opaque policies. Bayes nets expose structure but do not guarantee scale or correctness. Sundog’s experiment tests whether placing the objective-defining signal in external geometry gives a measurable traceability advantage, and where that advantage fails once learning pressure is introduced.