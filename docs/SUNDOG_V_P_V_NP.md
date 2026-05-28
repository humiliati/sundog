# Sundog vs. P-vs-NP Verification

Working hook:

> Safe policies may be hard to find, but their shadows may be easier to verify.

Short version:

> P-vs-NP asks whether finding and verifying are secretly the same difficulty.
> Sundog asks a bounded alignment version: when full policy search is hard,
> can a sufficient signature make safety verification tractable inside a named
> capacity envelope?

Status: Roadmap draft. Lit-pass, project scaffold, and Phase 1 toy-verifier
spec filed 2026-05-28; see
[`P_V_NP_LITPASS_MEMO.md`](P_V_NP_LITPASS_MEMO.md) and
[`pvnp/PHASE1_TOY_VERIFIER_SPEC.md`](pvnp/PHASE1_TOY_VERIFIER_SPEC.md). No
complexity-theoretic result claimed. This document is a research bridge from
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

Deliverables:

- Map each mesa controller family to a policy-verification object:
  HC-Signature, L-Signature, L-Reward, L-Mixed, Oracle.
- Define what a verifier must detect:
  - proxy collapse;
  - reward-channel dependence;
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
- [`pvnp/RECEIPT_TEMPLATE.md`](pvnp/RECEIPT_TEMPLATE.md): receipt template
  for phase and probe results.
- [`pvnp/receipts/README.md`](pvnp/receipts/README.md): receipt index; no
  Phase 1 execution receipt filed yet.

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
