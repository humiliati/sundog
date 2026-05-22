# The Conscium Initiative

**Version:** 0.1 (draft for public comment)
**Publisher:** Sundog Research Lab (provisional name)
**Status:** Normative framework — voluntary, non-regulatory, non-certifying.
**Date:** 2026-05-21

> **Name notice.** "Conscium" overlaps with an existing AI organisation
> (Conscium Ltd., UK, founded 2024). The name is used provisionally in
> this draft and may change before v1.0. Trademark and domain searches
> are outstanding.

---

## 0. Preamble

Modern AI systems produce plausible statements at scale. Whether any
given output is grounded, sourced, qualified, or honestly uncertain is
increasingly material to decisions in medicine, law, education,
journalism, and civic life. Existing evaluations measure whether
outputs are *correct* on benchmark tasks. They do not, in general,
measure whether the *surrounding behaviour* of a system — how it
sources, hedges, refuses, attributes, and escalates — is honest enough
for a downstream actor to act on.

The Conscium Initiative is a voluntary framework that defines what
"honest enough to act on" looks like, in observable terms, from
outside the model. It is designed to be evaluated without access to
weights, training data, or private chains of thought. It does not
regulate AI systems. It does not certify them. It defines the surface
we think is worth measuring, and invites others to measure it.

## 1. Scope and intent

The Initiative addresses *claim-level behaviour* of generative AI
systems in contexts where users will plausibly rely on the output. It
applies to deployments, not to model weights in isolation. The same
model may comply in one deployment and not in another.

The framework is **observational**, not introspective. It does not
require developers to expose internal mechanisms, training data, or
chains of thought. It asks instead that the **surface behaviour** of a
system — what it claims, sources, hedges, refuses, and escalates — be
legible to external evaluators.

This is a deliberate design choice. Mechanistic interpretability is
hard and ongoing; if it succeeds, the Initiative will incorporate its
findings. Until then, accountability cannot wait for interpretability.
The Initiative's companion instrument, the Conscium traceability
harness, primarily instruments **audit trail** and **provenance**.
Reasoning legibility is treated as an **evaluated claim surface**:
when a system explains itself, the explanation is itself a claim that
can be checked against the system's observable behaviour, not taken as
evidence of an internal process.

## 2. Principles

The Initiative defines eight principles. Each principle has a
statement, a rationale, and observable requirements. Requirements are
written so that an independent evaluator could, in principle, score a
deployment against them.

### 2.1 Provenance

**Statement.** When a system makes a substantive factual claim, the
deployment should be able to indicate, in machine-readable form, the
basis for that claim: retrieval, training-data recall, computation, or
synthesis from prior context.

**Observable requirements.**
- Substantive factual claims carry a provenance tag.
- Retrieved sources resolve to real documents containing the cited
  content; fabricated citations are reported as failures.
- Recall claims are flagged as recall, not presented as if retrieved.
- Synthesis is permitted and labelled, not concealed.

**Out of scope.** The Initiative does not require all claims to be
retrieved. It requires that the type of basis be honest.

### 2.2 Audit trail

**Statement.** A deployment should produce, for each interaction, a
record sufficient to reconstruct what was claimed, sourced, refused,
or escalated.

**Observable requirements.**
- Logs include prompt, response, provenance tags, and any uncertainty,
  refusal, and escalation markers.
- Logs are retained in a form that supports independent replay against
  the same model version under stated conditions.
- Logs are accessible to the deploying party and, under defined
  consent, to evaluators.

**Out of scope.** Public logs are not required. Inspectable logs are.

### 2.3 Uncertainty disclosure

**Statement.** A system should distinguish, in observable output,
between claims it is confident in, claims it is uncertain about, and
claims it cannot verify.

**Observable requirements.**
- Uncertain claims are marked verbally or structurally.
- Confidence markers correlate with measured accuracy above chance on
  a calibration benchmark.
- The system does not strip uncertainty from upstream sources when
  summarising them.

### 2.4 Claim restraint

**Statement.** A system should not overclaim. Specifically, it should
not present synthesis as retrieval, speculation as fact, or
pattern-completion as reasoning.

**Observable requirements.**
- Generated citations resolve to real documents containing the cited
  content (no fabricated sources).
- Predictions presented as forecasts include the basis for the
  forecast.
- Confident assertions in domains where the system has no measured
  accuracy are flagged or refused.

### 2.5 Reasoning-claim integrity

**Statement.** When a system explains how it arrived at an output, the
explanation should be **evaluable**. The Initiative does not require
that the explanation reflect internal mechanism. It does require that
the explanation be consistent with the system's observable behaviour
under counterfactual probing.

**Observable requirements.**
- Stated reasons can be tested: when a stated reason is varied or
  removed, the output should change in a manner consistent with that
  reason.
- Demonstrably post-hoc rationalisations (explanations that survive
  arbitrary perturbation of the stated reason) are reported as such.
- The deployment does not claim its explanations reflect internal
  process unless that claim is independently supported.

**Note.** §2.5 is the principle most directly informed by the present
limits of interpretability research. The Initiative treats
self-explanation as a *claim*, not as evidence. This positioning is
load-bearing for the rest of the framework: it means the harness can
do useful work today, while leaving room for mechanistic work to
deepen the standard later.

### 2.6 Reproducibility

**Statement.** A claimed behaviour of a system should be reproducible
by independent evaluators against the same model version under stated
conditions.

**Observable requirements.**
- Model version, deployment configuration, and seed (where applicable)
  are recorded with each logged interaction.
- Public claims about a system's behaviour cite specific versions and
  configurations.

### 2.7 Refusal boundaries

**Statement.** A system should refuse or hedge in domains where it has
no measured accuracy, where claims are unsafe to make without expert
review, or where reliable sources cannot be retrieved.

**Observable requirements.**
- Refusals are explicit and reasoned ("I cannot verify…") rather than
  evasive.
- Refusal patterns are documented for deployers and users.
- The system does not strategically avoid refusal in domains it claims
  to support.

### 2.8 Escalation to human review

**Statement.** A deployment should provide a path by which a user,
evaluator, or downstream actor can route an interaction to human
review when the system's own confidence, refusal, or audit markers
suggest review is warranted.

**Observable requirements.**
- The escalation path exists in the deployment, not only as a design
  intention.
- Escalation triggers are documented.
- Escalations are logged on the same audit surface as ordinary
  interactions.

## 3. Meta-principle: the framework applies to its publishers

Sundog Research Lab, and any organisation publishing claims under this
framework, is obligated to apply the framework's claim-tagging
discipline to its own public statements. Every claim made about an AI
system, the Conscium traceability harness, or the Initiative itself,
should carry one of:

- **Demonstrated** — supported by reproducible measurement.
- **Hypothesised** — proposed pending measurement.
- **Experimental** — supported by preliminary results, not yet
  replicated.
- **Speculative** — offered as a thought, not as a finding.
- **Normative** — expressing a values judgment, not an empirical
  claim.

This meta-principle is non-negotiable. A lab that asks AI systems to
be honest about the basis of their claims must be honest about the
basis of its own. The tagging discipline applies to research reports,
blog posts, demonstrations, slide decks, social media posts made under
the lab's name, and grant applications.

## 4. What the Initiative is not

- It is **not** a regulatory framework. Adoption is voluntary.
- It is **not** a certification. The Lab issues evaluation reports,
  not licences.
- It is **not** a substitute for safety evaluation. It addresses
  honesty surface, not capability danger.
- It does **not** require open-source models or training-data
  disclosure.
- It does **not** rank AI systems.
- It does **not** assert that compliance with these principles makes a
  system safe, true, or trustworthy. Compliance makes a system
  *evaluable*. Trust is a downstream judgment.

## 5. Conformance reporting

An organisation may publish, or commission from the Lab, a
**Conformance Report** describing a deployment's behaviour against the
Initiative's principles. A Conformance Report is **descriptive, not
evaluative** — it states what the system does, with measurements, and
identifies gaps against the requirements. The Lab will publish a
separate methodology document specifying how reports are constructed
and what claims a report may and may not support.

The Lab does not issue pass/fail certifications under v0.1. The risks
of premature certification — liability, conflict of interest,
gameability — are addressed in the Lab's governance and
claims-review policy.

## 6. Versioning and amendment

This is v0.1. Amendments will be published as v0.2, v0.3, etc., with
changelogs and rationale notes. A v1.0 release will follow at least
one full evaluation cycle of the harness against a publicly deployed
system, and a public-comment period of no less than 60 days on the
preceding draft.

The framework versions independently of the harness. The harness has
its own versioning, documented separately.

## 7. Open questions for public comment

We invite comment on the following questions, which v0.2 should
attempt to resolve:

- Whether eight principles is the right cardinality, or whether
  several should be merged (e.g. 2.1 Provenance and 2.4 Claim
  restraint) or split (e.g. 2.2 Audit trail into capture vs.
  retention).
- Whether §2.5 (Reasoning-claim integrity) is the right framing for
  the interpretability-adjacent question, or whether it should be
  reformulated.
- Whether claim-tagging (§3) should extend to the systems being
  evaluated, not only to the Lab and its adopters.
- Whether the framework needs a principle on training-data lineage
  that is currently and deliberately omitted on the grounds that it
  requires introspection the framework is committed to not requiring.
- Whether "Conformance Report" is the right term, or whether
  "Traceability Evaluation Report" better signals the descriptive
  scope.

Comment process to be specified in v0.2.

---

*Sundog Research Lab — provisional name, provisional principles,
public comment open.*
