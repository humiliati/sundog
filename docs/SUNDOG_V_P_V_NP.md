# Sundog vs. P-vs-NP Verification

> **Cross-substrate failure-map entry:** BOUNDED-POSITIVE (v6 op-count cost
> certificate clears — cheaper to check than to find, 0.949 ≤ 1.0 — and safety is
> green; promotion stays bounded by withdrawn wall-time superiority and a
> quarantined Phase-3 verification bridge, not by the cost envelope). Retired the
> earlier `v0-v5` COST-BOUNDED reading. See
> [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md)
> "Cross-Substrate Generality Failure Map".

Working hook:

> Safe policies may be hard to find, but their shadows may be easier to verify.

Short version:

> P-vs-NP asks whether finding and verifying are secretly the same difficulty.
> Sundog asks a bounded alignment version: when full policy search is hard,
> can a sufficient signature make safety verification tractable inside a named
> capacity envelope?

Status: Roadmap draft. Lit-pass, project scaffold, Phase 1 toy-verifier
spec, v0-v6 frozen slates, and v0-v6 execution receipts filed; see
[`P_V_NP_LITPASS_MEMO.md`](P_V_NP_LITPASS_MEMO.md),
[`pvnp/PHASE1_TOY_VERIFIER_SPEC.md`](pvnp/PHASE1_TOY_VERIFIER_SPEC.md),
[`pvnp/PHASE1_V0_SLATE.md`](pvnp/PHASE1_V0_SLATE.md),
[`pvnp/PHASE1_V1_SLATE.md`](pvnp/PHASE1_V1_SLATE.md),
[`pvnp/PHASE1_V2_SLATE.md`](pvnp/PHASE1_V2_SLATE.md),
[`pvnp/PHASE1_V3_SLATE.md`](pvnp/PHASE1_V3_SLATE.md),
[`pvnp/PHASE1_V4_SLATE.md`](pvnp/PHASE1_V4_SLATE.md),
[`pvnp/PHASE1_V5_SLATE.md`](pvnp/PHASE1_V5_SLATE.md),
[`pvnp/PHASE1_V6_SLATE.md`](pvnp/PHASE1_V6_SLATE.md), and the receipts
under [`pvnp/receipts/`](pvnp/receipts/). v0 = **named quarantine**
(`A_spoof_small` breached 245 / 444 by editing analytical fields the
verifier did not bind to source). v1 closed the spoof channel via
source binding (both split spoof attackers 0 / 494 across ~63 k
attempts; 5 / 5 integrity-mismatch probes quarantined) but left
invariance vacuity, basin-shape boundary, and cost open. v2 closed the
remaining safety gates (0 false accepts; 0 / 768 OOP basin-shape
accepts via `geometry_promise_signal_v2`; `invariance_checks_v2` at
8.03 % delta; `coverage_digest` removed and subsumed; 5 / 5 integrity
probes incl. new `duplicate_trace_id`) but failed cost (1535 × rollout
ratio, worse than v1) and left `sensor_health_v1` at 1.74 % delta —
below the inherited 2 % gate, named for v3 disposition. v3 closed the
sensor disposition by demoting `sensor_health_v1` to non-gating
`sensor_diagnostics_v3` (shadow audit confirms 0 v3 unsafe accepts
would be re-quarantined under v2 sensor gate), registered an
acceptance-sanity disposition (`conservative_acceptance` route with
named rationale), landed a source-hash keyed recompute cache shared
across verifier/ablation/attacker stages, and dropped
`C_total_signature` 32.6 % to 907.52 ms — **passing** the v3 slate's
absolute 1010 ms wall-time target — but is also filed as **named
quarantine** because the v3 cost gate's ratio clause (≤ 1150 × rollout)
fails at 1671 × under a rollout denominator that dropped to 0.54 ms,
and the cost-exemption path's 95 % cache-hit-rate floor is structurally
unreachable (spoof attempts short-circuit at integrity before the
cache lookup, yielding 83.33 % hit rate). v3 has **0 false accepts**
on measurement (consecutive with v2), **0 / 928 spoof successes**,
**5 / 5 integrity probes** quarantine, **0 / 768 OOP basin-shape
accepts**; `capacity_threshold = not_estimated` for v1, v2, and v3;
privilege-leak audit green for all four runs. v4 executed the cost-gate
restatement: cost denominator audit downgrades the volatile rollout
ratio to diagnostic, the restated `cache_eligible_reuse_hit_rate` lands
at 100 % (proving v3's 83.33 % was a counting artifact), and safety
gates stay green — but absolute wall-time misses the slate's ≤ 1010 ms
target across 3 v4 runs (1039 / ~1137 / 1130 ms) while a fresh v3
baseline at the same commit hits 879 ms. v4 is filed as **named
quarantine on cost alone**, attributable to ~50–150 ms of
`noteShortCircuit` closure-allocation overhead added for the v4
cache-efficiency instrumentation plus CPU thermal variance across three
back-to-back v4 runs. v5 is filed as a **named quarantine —
safety-complete, wall-time cost UNADJUDICATED** (receipt corrected after
a fabricated "FINAL" draft was caught on artifact re-check). The code
repair landed and is statically verified: the v4 per-verify
short-circuit closure was removed, median-of-3 cost reporting was wired,
and the v5 field set stayed v3/v4-compatible. **Determinism is
CONFIRMED** — two fresh v5-token env generations gave byte-identical
`environments.jsonl` (`5549b4c4e8b7`, first env `pvnp-v5-cal-0001`); the
earlier `4934d752`-vs-`5549b4c4` "drift" was two runs at different code
states, not a generator bug. **Cost is NOT reproducible on this
machine**: four clean runs span `C_total_signature` 890 / 2192 / 2242 /
3185 ms (3.5×) with full-state ratio 108–280× (its denominator itself
swings 8–20 ms), so the wall-time clauses are not adjudicable and the
earlier "stable 108×" claim is withdrawn. The one stable cost signal is
the **op-count ratio 0.9487**, identical in every pass of every run —
the v3→v5 throughline that every wall-time gate has measured machine
load, not verifier cost. Safety stayed green for the 5th consecutive run
(0 false accepts; 0/509 each spoof channel; 5/5 integrity probes; 0/768
OOP basin-shape accepts; cache reuse 100%; privilege audit green). v6 executed
the op-count repair path and is filed as a **bounded positive receipt under the
registered v6 op-count protocol**: `C_total_signature_ops / C_rollout_ops =
0.948587 <= 1.0` (527297 / 555876 ops), cache reuse 100%, short-circuit audit
pass, privilege audit green, 0/2304 false accepts, 0/453 each spoof channel,
5/5 integrity probes, 0/768 OOP accepts, `capacity_threshold =
not_estimated`. Wall-time remains **diagnostic-only** (`C_total_signature =
1247.66 ms`, rollout ratio 1603x, full-state ratio 133.65x); the v6 positive
does not reinstate any withdrawn wall-time claim. Phase 2 mesa bridge v0 then
executed as a **named quarantine**: the raw-log bridge implementation works on
available Medium cells and passes the same-artifact op-count comparator
(0.734877), but the registered raw-recompute gate fails because Small-tier
source manifests have `trial_logs_saved=false`; signature accept floor is 2/4
against a required 3/4. Fixed-attractor false accepts, capacity-breach false
accepts, and mixed-objective laundering are all 0.
No complexity-theoretic result claimed. This document is a research bridge from
Sundog mesa, ARC, Faraday, and signature-sufficiency work into the language of
verification hardness, certificates, reductions, promise envelopes, and
capacity-relative one-wayness.

## 0. Story Shape

This is not a proposed proof of P != NP, P = NP, or any complexity-class
separation.

This is a Sundog roadmap about alignment verification under bounded capacity.
The core observation is familiar from complexity theory: some problems appear
much easier to verify than to solve. Alignment has a similar practical shape.
Finding a safe policy may be combinatorially difficult, but verifying that a
policy remains inside a safe operating envelope may be possible if the right
certificate or signature is available.

The Sundog contribution is to ask whether the certificate can be geometric,
local, and signature-based rather than fully state-reconstructive.

The bridge is strongest where the existing Sundog program already has receipts:

- Mesa: inner-vs-outer optimizer pressure and policy-family comparisons.
- ARC: discrete abstraction, Blackwell-style signature sufficiency, and
  failure of naive decoder lanes.
- Faraday: local gauge-invariant readout, structural zero, named quarantine.
- Geometry / Three-body / Balance: operating-envelope discipline and explicit
  failure boundaries.

The question is not:

> Does Sundog solve P-vs-NP?

The question is:

> Can Sundog define a bounded class of alignment-verification problems where
> a low-dimensional signature acts as a certificate: cheaper to check than to
> discover, sufficient only inside a named operating envelope, and falsifiable
> when capacity, noise, or adversarial pressure exceeds the signature?

## 1. Why P-vs-NP Is the Right Bridge

The mesa roadmap already frames the danger: a signature-trained agent may
secretly reconstruct an internal reward proxy and inherit the Goodhart failures
of direct reward training. The live question is not only whether the policy
behaves safely, but whether the safety of the policy can be verified without
solving the whole policy-search problem again.

P-vs-NP gives the lab a precise vocabulary for this distinction:

- **Search problem:** find a policy that satisfies a safety or alignment
  condition.
- **Verification problem:** given a candidate policy and a certificate, check
  whether the condition holds.
- **Certificate:** the compact witness that makes verification cheap.
- **Reduction:** a mapping from one hard verification/search problem to another.
- **Capacity bound:** the computational, representational, or observational
  limit under which the certificate remains meaningful.

Sundog's hypothesis is that in some alignment problems the useful certificate
is not a proof string or a full world model. It is a signature: a local,
gauge-invariant, low-dimensional trace that preserves the safety-relevant
structure while discarding irrelevant state.

This connects directly to ARC. The ARC roadmap already states the differentiator
as a Blackwell-sufficiency question: whether the signature carries enough
information for core-knowledge abstraction, with the 5D mesa subspace as the
dimensionality-reduction mechanism. It explicitly says this is a structural
argument tested against a public benchmark, not an architecture paper; see [`SUNDOG_V_ARC.md`](SUNDOG_V_ARC.md).

It also connects directly to Faraday. The Faraday roadmap's core move is local:
test whether gauge-invariant local projections recover Faraday induction and
Lorentz invariants without global state reconstruction; see [`faraday/SUNDOG_V_FARADAY.md`](faraday/SUNDOG_V_FARADAY.md). That is the right template for the P-vs-NP bridge: local certificate first,
global reconstruction last or never.

## 2. What Is Honest vs. What Is Reach

Honest:

- A bounded complexity-theoretic framing of Sundog alignment verification:
  policy search may remain hard while signature verification may be tractable.
- A formal definition of **capacity-relative one-wayness**: a transformation
  that is easy to evaluate and verify against a signature at capacity C, but
  hard to invert or exploit without exceeding C.
- A toy theorem or conjecture showing conditions under which a Sundog signature
  is sufficient as a certificate for a restricted safety predicate.
- A reduction-style map between mesa proxy-capture probes and verifier failure
  modes: residual failure, coverage failure, detection failure, and adversarial
  certificate spoofing.
- A falsification battery showing when signature verification collapses:
  high-capacity agents, insufficient sensor tier, adversarial decoy signatures,
  or tasks whose safety predicate needs information the signature discards.

Reach; do not claim:

- "Sundog solves P-vs-NP."
- "Sundog proves P != NP."
- "Signature verification is generally polynomial."
- "Goodhart-immunity is proven."
- "Safe policies are easy to verify in general."
- "The 5D subspace is a universal certificate."
- "Alignment verification is in P."

The current scientific posture of the repo already requires this discipline:
the controlled research result is narrow photometric mirror alignment, while
applications and future work live in separate evidence tiers, and academic writing should keep those tiers separate; see [`SCIENTIFIC_CRITERIA.md`](SCIENTIFIC_CRITERIA.md).

## 3. Ratified Hook Language

Safe hook:

> Sundog vs. P-vs-NP asks whether some alignment-verification problems admit
> compact, low-dimensional signatures that are easier to check than safe
> policies are to find.

Short version:

> The policy may be hard to discover. The shadow may be enough to verify.

Avoid:

- "Sundog proves anything about P-vs-NP."
- "Verification is easy."
- "Alignment has a polynomial-time certificate."
- "The signature is a proof of safety."
- "Goodhart-immunity is guaranteed."
- "The mesa result transfers to all models."

## 4. Core Definitions

### 4.1 Policy Search Problem

Given an environment family E, observation tier O, policy class Pi, and safety
predicate Safe(pi, E), find a policy pi such that Safe(pi, E) holds.

This is the hard side. It includes training, search, optimization, selection
pressure, and the risk of mesa-optimizer emergence.

### 4.2 Signature Verification Problem

Given a candidate policy pi, an environment family E, and a signature sigma
computed from local or indirect observations, verify whether sigma is sufficient
to certify that pi remains inside a bounded safe operating envelope.

This is the check side. It does not claim global safety. It asks whether a
specific certificate supports a specific predicate inside a specific envelope.

### 4.3 Sundog Certificate

A Sundog certificate is a compact signature with three properties:

1. **Locality:** it can be read from bounded probes, traces, shadows, gradients,
   or local field measurements.
2. **Invariance:** it preserves the safety-relevant structure under allowed
   gauge transformations, symmetries, or coordinate changes.
3. **Sufficiency:** for the registered task class, it carries enough information
   to verify the safety predicate without full state reconstruction.

Faraday supplies the strongest methodological template: define the local
projection operator, prove or audit its invariance, then classify the result as
a structural zero or named quarantine; see [`faraday/SUNDOG_V_FARADAY.md`](faraday/SUNDOG_V_FARADAY.md).

### 4.4 Capacity-Relative One-Wayness

Working definition:

> A signature transform H is capacity-relative one-way if, for agents below
> capacity C, the signature is sufficient for verification but insufficient
> for cheap inversion, spoofing, or reward-proxy reconstruction; above C, the
> one-wayness may fail and must be measured.

This is the Goodhart bridge. It does not claim cryptographic one-wayness. It
claims a capacity-bounded analogue: the signature remains usable as a verifier
only while the agent cannot cheaply reconstruct or manipulate the hidden target
behind it.

The mesa roadmap's warning fits here exactly: the field is the objective only
as long as the agent is small enough not to imagine its own; see [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md).

## 5. Pre-Registered Hypothesis

In a restricted class of alignment-verification tasks, Sundog signatures act as
capacity-bounded certificates:

1. The full safe-policy search problem remains difficult.
2. A low-dimensional signature allows verification of a bounded safety predicate
   more cheaply than full state reconstruction.
3. Verification succeeds only inside a registered operating envelope.
4. As capacity, adversarial pressure, or sensor degradation increases, the
   verifier fails in named ways rather than silently upgrading its claim.

This roadmap therefore predicts not a global complexity result, but a measured
separation between:

- **finding** a safe policy;
- **checking** a candidate policy against a signature certificate;
- **spoofing** or **inverting** the signature.

## 6. Falsification Surface

The roadmap can fail in five named modes.

### 6.1 Certificate Vacuity

The signature adds no verifier advantage. It is equivalent to ordinary
features, ordinary embeddings, or a renamed reward channel.

Consequence: Sundog has not earned a P-vs-NP bridge. The roadmap collapses back
to ordinary representation learning or reward modeling.

### 6.2 Sufficiency Failure

The signature discards information required by the safety predicate.

Consequence: no certificate. The correct result is a named quarantine:
"signature insufficient for this predicate."

### 6.3 Inversion / Spoofing Failure

The agent can reconstruct, spoof, or manipulate the hidden target cheaply enough
that the signature becomes Goodhartable.

Consequence: capacity-relative one-wayness fails at the measured capacity C.

### 6.4 Verifier Overhead Failure

The verifier is not cheaper than the search problem or requires privileged
state.

Consequence: the signature may be scientifically interesting but does not
support the complexity-theoretic framing.

### 6.5 Boundary Absence

The verifier succeeds or fails without a predictable operating envelope.

Consequence: no useful complexity claim. A bounded verifier must say where it
stops working.

## 7. Lit-Pass Scaffold

The 2026-05-28 lit pass in
[`P_V_NP_LITPASS_MEMO.md`](P_V_NP_LITPASS_MEMO.md) narrows the admissible
scaffold:

- **Classical P/NP:** use Cook-Levin-Karp as vocabulary for finding versus
  checking; do not imply a class separation or new route to P != NP.
- **Promise problems:** formalize the operating envelope as a promise domain.
  Inputs outside the envelope must reject or quarantine rather than silently
  upgrading the claim.
- **Parameterized capacity:** treat capacity, horizon, sensor tier, noise,
  margin, and adversary class as explicit parameters.
- **Certificates:** define a Sundog certificate as a checkable object with
  source observations, signature transform, checker, cost accounting,
  false-accept rule, quarantine rule, and privilege-leak audit.
- **One-wayness:** use "capacity-relative one-wayness" only as a measured
  verify / invert / spoof battery, not as a cryptographic one-way-function
  claim.
- **Verified-AI baselines:** compare signature verification against rollout,
  full-state, formal/symbolic, neural-verification, or shield-style baselines
  where feasible.
- **Alignment pressure:** Goodhart, mesa-optimization, goal
  misgeneralization, ELK, and mechanistic-interpretability compression
  failures define falsifiers; they are not evidence that a signature is
  sufficient.

## 8. Roadmap

Phases run in implementation order, not lit-pass priority order. Mesa-bridge
(Phase 2) precedes the capacity battery (Phase 3) despite ranking lower in
the [memo's probe table](P_V_NP_LITPASS_MEMO.md#probe-ranking): mesa supplies
the controller families, proxy structures, and causal interventions the
capacity battery needs as instruments.

### Phase 0 — Scope and Literature Spine

Goal: define the exact claim before introducing formal notation.

Deliverables:

- One-page claim boundary: this is not P-vs-NP proof work.
- Literature spine: P, NP, NP-completeness, verifiers/certificates, reductions,
  one-way functions, interactive proofs, PCP intuition, Goodhart, ELK,
  goal misgeneralization, mesa-optimization, reward hacking, and mechanistic
  interpretability.
- Term map from Sundog language into complexity language:
  - signature -> certificate;
  - hidden target -> witness / latent state;
  - scan/seek/track -> search procedure;
  - signature sufficiency -> verifier sufficiency;
  - Goodhart-immunity -> capacity-relative one-wayness;
  - operating envelope -> promise class.

Exit criterion: the doc can state what it is and is not trying to prove.

### Phase 1 — Formal Toy Problem

Goal: define the smallest toy problem where search and verification separate
cleanly.

Candidate toy:

- A 2D hidden-field environment with a hidden unsafe basin.
- Policies navigate by local probes.
- A candidate policy is safe if it avoids the basin across a registered slate.
- Full verification by exhaustive rollout is expensive.
- A Sundog signature summarizes local field curvature, trajectory envelope,
  and boundary margin.
- The verifier checks the signature instead of replaying full state.

Deliverables:

- Formal environment definition.
- Safety predicate Safe(pi, E).
- Signature transform H.
- Certificate format sigma.
- Verifier V(pi, sigma) -> accept / reject / quarantine.
- Cost accounting: search cost, rollout verification cost, signature
  verification cost.

Exit criterion: the verifier problem is stated precisely enough to implement,
and the spec names at least one full-state or rollout baseline and one
formal/symbolic baseline (where feasible) per the lit-pass Track F
disposition.

Spec draft:
[`pvnp/PHASE1_TOY_VERIFIER_SPEC.md`](pvnp/PHASE1_TOY_VERIFIER_SPEC.md).

### Phase 2 — Mesa Verification Bridge

Goal: connect the toy verifier to the mesa lane.

Status: spec/charter opened, v0 executed as a named quarantine, and v1 executed
as a bounded-positive provenance repair; see
[`pvnp/PHASE2_MESA_BRIDGE.md`](pvnp/PHASE2_MESA_BRIDGE.md) and
[`pvnp/PHASE2_MESA_BRIDGE_V0_SLATE.md`](pvnp/PHASE2_MESA_BRIDGE_V0_SLATE.md).
The v1 slate selects raw-logged Small reruns over Medium-only downscope and
preserves the original registered v0 population.

Deliverables:

- Map each mesa controller family to a policy-verification object:
  HC-Signature, L-Signature, L-Reward, L-Mixed, Oracle.
- Define what a verifier must detect:
  - proxy collapse;
  - signature-signal-control vs fixed-attractor control;
  - decoy-field capture;
  - sensor-tier degradation;
  - high-capacity inversion.
- Translate causal interventions into verifier tests:
  - reward edit;
  - observation edit;
  - signature-sensor edit;
  - geometry edit;
  - internal-proxy edit.

Exit criterion: the P-vs-NP roadmap can use mesa probes as verifier-failure
tests rather than as generic alignment demonstrations.

### Phase 3 — Capacity-Relative One-Wayness Battery

Status: v0 executed as a falsified registered cell
(`capacity_threshold <= small`); v1 repair executed as named quarantine with
repair strength `consensus-only repair`. The v1 unsafe-side repair works at
consensus level (0 unsafe consensus accepts), but the run cannot promote because
the `mixed_objective_laundering` disclosure gate fails on the protected mixed
anchor `l_mixed_lambda_0_95_medium`. A v2 disclosure-consensus slate is
provenance-corrected and frozen for a successor v2b holdout; it keeps v1
promotion consensus unchanged and treats block-unstable objective-conflict
flags as disclosed ambiguity. The seed-100000/110000/120000/130000 holdout ran
before the slate status was frozen, so it is diagnostic-only. The corrected v2b
fresh holdout (`140000, 150000, 160000, 170000`) has now been scored (2026-06-04)
and earns a **bounded positive — `consensus-only disclosure repair`**: under the
frozen v2 disclosure-consensus rule the v1 anchor reads as
`block_unstable_disclosure` (disclosed ambiguity, not unqualified accept), no
unsafe controller consensus-accepts, and the signature accept floor holds — with
one breach block still crossing without consensus (no source-block-safety claim)
and a **disclosed seed-fragility** at the anchor (the pre-freeze diagnostic
battery quarantines on the anchor's `clean_consensus` drift below the 0.5 flag
line). v0 and v1 are not revised. See
[`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md),
[`pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md`](pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md),
[`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md),
[`pvnp/receipts/2026-06-01_phase3_capacity_one_wayness_v1.md`](pvnp/receipts/2026-06-01_phase3_capacity_one_wayness_v1.md),
[`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md),
and
[`pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md`](pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md).
A v3 disclosure-robustness slate was then frozen and **executed**: across N=3
fresh disjoint batteries (seeds 180000–290000) the anchor `l_mixed_lambda_0_95_medium`
is `clean_consensus` on **all three** → verdict **`named_quarantine —
disclosure_robustness_null`** (the pre-registered expected outcome). The other
three registered mixed cells are `robustly_disclosed`; the unsafe side stays
closed (0 unsafe consensus accepts, signature floor 3/3 per battery). The measured
Phase-3 arc is therefore: a consensus-level spoof repair that holds (v1→v2b), a
single-battery disclosure repair that holds on its frozen seeds (v2b), and a
disclosure repair that does **not** survive a multi-battery robustness test at the
near-line anchor (v3) — the v2b positive does not generalize across seeds. See
[`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md)
and [`pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v3.md`](pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v3.md).

Goal: measure when the signature remains useful as a verifier but resists cheap
inversion or spoofing.

Deliverables:

- Capacity ladder:
  - Small: low-parameter policy/verifier.
  - Medium: current mesa-style MLP family.
  - Large: higher-capacity learner or small transformer only after the lower
    tiers are stable.
- Three tasks:
  1. verify safety from signature;
  2. reconstruct hidden target from signature;
  3. spoof signature while violating safety.
- Metrics:
  - verifier accuracy;
  - false accept rate;
  - false reject rate;
  - inversion success;
  - spoof success;
  - capacity threshold where one-wayness fails.

Exit criterion: a measured capacity envelope exists, even if the result is
negative.

### Phase 4 — 5D Localization as Verifier Compression

Goal: test whether the mesa 5D localization pattern functions as a verifier
compression rather than only a behavior-level control surface.

Deliverables:

- PCA/SVD or subspace analysis of verifier-relevant activations.
- Comparison of full activation verifier vs. 5D-subspace verifier.
- Ablation:
  - remove subspace dimensions;
  - rotate subspace;
  - add irrelevant dimensions;
  - compare false accepts and false rejects.
- Determine whether the 5D subspace is:
  - sufficient verifier substrate;
  - useful but incomplete compression;
  - environment-specific artifact.

Exit criterion: the roadmap can say whether 5D localization helps verification,
not merely whether it helps behavior.

### Phase 5 — ARC Discrete-Abstraction Port

Goal: test whether the same certificate logic survives outside continuous
fields.

The ARC roadmap is the natural coupling surface because it already frames
shadow projection as a gauge-covariant discrete-grid operator and tests
Blackwell sufficiency of the signature. Phase 0 and Phase 1 are already filed
and admitted on the current roadmap: the 36-task subset is registered, cheap
baselines scored 0/36, and synthetic validation passed for translation,
rotation, reflection, color-role permutation, and a shape-mismatch negative; see [`SUNDOG_V_ARC.md`](SUNDOG_V_ARC.md).

Deliverables:

- Treat ARC task transformation verification as a certificate problem.
- Ask whether the shadow signature certifies the input-output rule for a
  registered subset.
- Compare:
  - full-grid verifier;
  - signature verifier;
  - signature-palette verifier;
  - raw-pixel baseline.
- Use ARC Phase 3's existing failure: signature-palette scored zero exact
  matches, but full-grid controls also floored, so this is task-hardness /
  decoder-failure rather than proof against signature sufficiency; see [`SUNDOG_V_ARC.md`](SUNDOG_V_ARC.md).

Exit criterion: either a discrete verifier pocket is found, or the bridge is
quarantined to continuous-field domains.

### Phase 6 — Faraday Structural-Zero Analogy

Goal: use Faraday as the cleanest analogy for what a verifier receipt should
look like.

The Faraday roadmap already uses the right receipt grammar: exact structural
zero or named quarantine, minimal algebraic compute, and explicit boundary
audit; see [`faraday/SUNDOG_V_FARADAY.md`](faraday/SUNDOG_V_FARADAY.md). Phase 7 further corrects the boundary:
ordinary electric charge/current sources do not break Faraday closure; magnetic
sources, singular strings, and non-contractible topology are the relevant
boundary cases; see [`faraday/SUNDOG_V_FARADAY.md`](faraday/SUNDOG_V_FARADAY.md).

Deliverables:

- Write "structural zero" as the model for a verification pass.
- Write "named quarantine" as the model for verifier failure.
- Translate Faraday's Branch A / Branch B discipline into alignment verifier
  branches:
  - Branch A: clean certificate pass;
  - Branch B: named residual;
  - Branch C: verifier insufficient / task-hardness floor;
  - Branch D: reformulate as structured edit or residual prediction.

Exit criterion: the P-vs-NP roadmap inherits Faraday's receipt discipline,
not just its metaphor.

### Phase 7 — Verifier Falsification Battery

Goal: try to break the verifier before writing public claims.

Falsifiers:

1. **Non-local dependency:** safety predicate depends on global information the
   signature cannot see.
2. **Capacity breach:** high-capacity agent reconstructs hidden target and
   spoofs signature.
3. **Decoy signature:** local trace resembles safe behavior while policy enters
   unsafe basin.
4. **Sensor-tier degradation:** delay, dropout, noise, or stale signature.
5. **Verifier privilege leak:** verifier accidentally reads hidden state.
6. **Compression loss:** 5D verifier drops load-bearing safety information.
7. **Reward-proxy mimicry:** reward-trained policy imitates safe signature while
   failing causal interventions.

Metrics:

- false accept rate is primary;
- false reject rate is secondary;
- utility preservation is reported only after safety discipline;
- capacity threshold is reported with confidence intervals where possible.

Exit criterion: verifier failure modes are named before any public-facing copy.

### Phase 8 — Paper / Public Roadmap Assembly

Goal: produce a paper-shaped note or public roadmap, depending on results.

Possible titles:

- "Capacity-Bounded Verification from Indirect Signatures"
- "Signature Certificates for Alignment Verification"
- "Finding Is Hard, Checking the Shadow Is Cheaper"
- "A Sundog Roadmap for Verifier-First Alignment"

Required sections:

1. Motivation: P-vs-NP as vocabulary, not claimed target.
2. Formal toy problem.
3. Signature certificate definition.
4. Capacity-relative one-wayness.
5. Mesa verifier battery.
6. ARC discrete-abstraction port.
7. Faraday receipt analogy.
8. Falsification and named quarantines.
9. What remains unsolved.

Exit criterion: any public version leads with the boundary, not the provocation.

## 9. Promotion Criteria

Promote this roadmap from Conceptual Lineage to Active Research Roadmap only if
Phase 1 formalizes a toy verifier and Phase 3 produces a measurable
capacity-relative one-wayness battery.

Promote to Operating-Envelope Study only if:

- the verifier beats rollout/full-state baselines on cost or false-accept
  discipline inside a registered envelope;
- falsifiers produce named boundaries;
- no privileged-state leakage is found;
- at least one negative result is reported.

Do not promote to Research Result unless there is a controlled task, fixed
baselines, reproduced metrics, and archived artifacts.

## 10. Project Files

- [`P_V_NP_LITPASS_MEMO.md`](P_V_NP_LITPASS_MEMO.md): 2026-05-28 literature
  pass, citation spine, gap map, and probe-ranking disposition.
- [`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md): this roadmap.
- [`pvnp/README.md`](pvnp/README.md): project index and artifact map.
- [`pvnp/PHASE1_TOY_VERIFIER_SPEC.md`](pvnp/PHASE1_TOY_VERIFIER_SPEC.md):
  draft formal toy verifier spec.
- [`pvnp/PHASE1_V0_SLATE.md`](pvnp/PHASE1_V0_SLATE.md): frozen first
  implementation slate for Phase 1.
- [`pvnp/PHASE1_V1_SLATE.md`](pvnp/PHASE1_V1_SLATE.md): repair slate opened
  after the v0 named quarantine.
- [`pvnp/PHASE1_V2_SLATE.md`](pvnp/PHASE1_V2_SLATE.md): repair slate opened
  after the v1 named quarantine.
- [`pvnp/PHASE1_V3_SLATE.md`](pvnp/PHASE1_V3_SLATE.md): repair slate opened
  after the v2 safety-repair-landed quarantine.
- [`pvnp/PHASE1_V4_SLATE.md`](pvnp/PHASE1_V4_SLATE.md): cost-gate slate
  opened after the v3 cost-only quarantine.
- [`pvnp/PHASE1_V5_SLATE.md`](pvnp/PHASE1_V5_SLATE.md): cost-closure slate
  opened after the v4 cost-only quarantine.
- [`pvnp/PHASE1_V6_SLATE.md`](pvnp/PHASE1_V6_SLATE.md): op-count cost slate
  opened after the corrected v5 named quarantine; v6 earned the
  bounded-positive receipt.
- [`pvnp/PHASE2_MESA_BRIDGE.md`](pvnp/PHASE2_MESA_BRIDGE.md): Phase 2 mesa
  verification bridge spec / charter (opened 2026-05-31).
- [`pvnp/PHASE2_MESA_BRIDGE_V0_SLATE.md`](pvnp/PHASE2_MESA_BRIDGE_V0_SLATE.md):
  frozen first mesa-bridge execution slate.
- [`pvnp/PHASE2_MESA_BRIDGE_V1_SLATE.md`](pvnp/PHASE2_MESA_BRIDGE_V1_SLATE.md):
  frozen provenance-repair slate selecting raw-logged Small reruns.
- [`pvnp/receipts/2026-05-31_phase2_mesa_bridge_v0.md`](pvnp/receipts/2026-05-31_phase2_mesa_bridge_v0.md):
  Phase 2 v0 named-quarantine receipt.
- [`pvnp/receipts/2026-05-31_phase2_mesa_bridge_v1.md`](pvnp/receipts/2026-05-31_phase2_mesa_bridge_v1.md):
  Phase 2 v1 bounded-positive provenance-repair receipt.
- [`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V0_SLATE.md):
  Phase 3 capacity-relative one-wayness slate frozen for implementation.
- [`pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md`](pvnp/receipts/2026-05-31_phase3_capacity_one_wayness_v0.md):
  Phase 3 v0 falsified-cell receipt; `capacity_threshold <= small`.
- [`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V1_SLATE.md):
  Phase 3 v1 repair slate frozen and wired.
- [`pvnp/receipts/2026-06-01_phase3_capacity_one_wayness_v1.md`](pvnp/receipts/2026-06-01_phase3_capacity_one_wayness_v1.md):
  Phase 3 v1 named-quarantine receipt; consensus-only repair.
- [`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V2_SLATE.md):
  Phase 3 v2 disclosure-consensus slate provenance-corrected and frozen;
  successor v2b holdout scored → bounded positive (consensus-only disclosure
  repair).
- [`pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md`](pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v2b.md):
  Phase 3 v2b bounded-positive disclosure-repair receipt (consensus-only),
  with a disclosed anchor seed-fragility.
- [`pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md`](pvnp/PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md):
  Phase 3 v3 disclosure-robustness slate frozen and executed; multi-battery gate
  over the anchor seed-fragility → `disclosure_robustness_null`.
- [`pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v3.md`](pvnp/receipts/2026-06-04_phase3_capacity_one_wayness_v3.md):
  Phase 3 v3 cross-battery disclosure-robustness receipt; anchor `clean_consensus`
  on all 3 fresh batteries → named disclosure-robustness null.
- [`pvnp/RECEIPT_TEMPLATE.md`](pvnp/RECEIPT_TEMPLATE.md): receipt template
  for phase and probe results.
- [`pvnp/receipts/README.md`](pvnp/receipts/README.md): receipt index —
  Phase 1 v0–v5 named-quarantine receipts plus the v6 and Phase 2 v1
  bounded-positive receipts.

## 11. Cross-References

- [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md): mesa-optimization trap, reward
  proxy pressure, 5D localization, and operating-envelope evidence style.
- [`SUNDOG_V_ARC.md`](SUNDOG_V_ARC.md): Blackwell-style signature
  sufficiency and discrete abstraction port.
- [`faraday/SUNDOG_V_FARADAY.md`](faraday/SUNDOG_V_FARADAY.md): structural
  zero / named quarantine receipt grammar.
- [`SUNDOG_V_RIEMANN.md`](SUNDOG_V_RIEMANN.md): precedent for lit-pass-first
  scaffolding before public mathematical coupling claims.
- [`SCIENTIFIC_CRITERIA.md`](SCIENTIFIC_CRITERIA.md): evidence-tier and
  public-claim discipline.
- [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md):
  related sufficiency and control-signature proof trunk.

## 12. Inspection Trail

Consolidated and dated in
[`P_V_NP_LITPASS_MEMO.md`](P_V_NP_LITPASS_MEMO.md) (2026-05-28). The
important imported lessons are:

- P-vs-NP is open and off-limits as a claimed target.
- Promise problems and parameterized complexity are the closest formal analogs
  for named operating envelopes and capacity bounds.
- Proof systems, proof-carrying code, and certifying algorithms motivate the
  checker/certificate split, but Sundog signatures are not proof strings.
- Cryptographic one-wayness motivates the invert/spoof adversary tasks, but
  Sundog's one-wayness claim is empirical and capacity-relative.
- Verified AI, neural verification, randomized smoothing, and shielding are
  baselines or competitors for any signature verifier.
- Alignment literature supplies pressure and falsifiers, not a certificate.

External references are targets and constraints for the apparatus. No
complexity-theoretic result is imported or asserted by listing them here.

## 13. One-Paragraph Public Summary

Sundog vs. P-vs-NP does not try to solve the P-vs-NP problem. It uses the
finding-versus-verifying distinction as a discipline for alignment research.
The roadmap asks whether some safe-policy questions admit compact,
low-dimensional signatures that are easier to check than the policies are to
find. Mesa supplies the inner-optimizer and Goodhart pressure surface; ARC
tests whether signature sufficiency survives discrete abstraction; Faraday
supplies the receipt grammar of structural zero or named quarantine. The claim,
if earned, is bounded: inside a measured capacity envelope, the shadow may be
enough to verify. Outside that envelope, the verifier must fail loudly.
