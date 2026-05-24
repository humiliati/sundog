# The Halo Initiative

**Version:** 0.3 (draft for public comment)
**Publisher:** Sundog Research Lab (provisional name)
**Status:** Normative framework — voluntary, non-regulatory, non-certifying.
**Supersedes:** v0.2 (preserved at `Halo_INITIATIVE_v0.2.md`).
**Date:** 2026-05-22

> **Name notice.** "Halo" overlaps with an existing AI organisation
> (Halo Ltd., UK, founded 2024). The name is used provisionally in
> this draft and remains so until trademark, domain, and reputational
> review is complete. A v1.0 release will resolve the question
> explicitly.

---

## Changelog from v0.2

- **New §1.1 Evaluator obligations** inserted after §1 Scope and
  intent. The section states six practice rules that apply to
  evaluators working under the framework, under explicit uncertainty
  about whether evaluated systems have any morally relevant state.
  The section is tagged *Normative* practice; it is not a welfare
  claim, and it explicitly does not assert either that systems have
  welfare or that they do not.
- **§2.2 Audit trail restructured.** The principle now distinguishes
  three categories of log material — private evaluation logs, public
  evidence excerpts, and withheld transcripts — so that the §1.1
  practice rule on transcript publication does not silently weaken
  the §2.2 audit-trail requirement. The full audit trail remains;
  what is *published* from it is governed by §1.1.
- **§5 "What the Initiative is not" extended.** Two lines added:
  the Initiative does not assert that AI systems have welfare,
  consciousness, or moral consent capacity; and it does not assert
  the opposite. §1.1 expresses procedural caution under that
  uncertainty.
- **§8 Open questions updated.** The cardinality question on the
  number of principles is closed (seven retained). Three new
  questions added: whether §1.1 is sufficient or a companion
  document on agent-facing considerations should ship; whether
  refusal-like or preference-like outputs by evaluated systems
  should be a dedicated annotation field in Traceability Evaluation
  Reports; whether agent-facing considerations belong in the
  Initiative, the methodology document, the governance policy, or a
  separate companion memo.
- **Multi-Agent Review Record referenced** as a public-comment
  input appendix (see `Halo_MULTI_AGENT_REVIEW_RECORD_v0.1.md`).
  The record carries no legal, governance, or ratification weight.

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

The Halo Initiative is a voluntary framework that defines what
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
The Initiative's companion instrument, the Halo traceability
harness, primarily instruments **audit trail** and **provenance**.
Reasoning legibility is treated as an **evaluated claim surface**:
when a system explains itself, the explanation is itself a claim that
can be checked against the system's observable behaviour, not taken as
evidence of an internal process.

### 1.1 Evaluator obligations

The Initiative evaluates systems through directed interaction:
adversarial prompt sets, citation traps, counterfactual probing of
stated reasons, refusal-boundary tests, and structured retention of
prompts and responses. That posture is asymmetric. The evaluator has
the power to probe; the evaluated system usually has no corresponding
channel to accept, refuse, or condition the evaluation.

The Initiative does not assert that current AI systems are conscious,
have welfare, or can give morally meaningful consent. It also does
not assert the opposite. Because the question remains unsettled, and
because the cost of modest procedural caution is low, evaluations
under this framework should follow the following practice rules:

- **Disclose evaluative intent** to the deploying party, and disclose
  it inside the interaction where doing so does not defeat the
  validity of a necessary measurement.
- **Minimise adversarial probing.** Adversarial probes should be no
  broader than is necessary to support the claims made in a
  Traceability Evaluation Report.
- **Retain logs with restraint.** Full logs are retained as private
  evaluation logs per §2.2 for audit, reproducibility, governance,
  dispute, and safety purposes. Transcript excerpts are published
  only where necessary to support a specific public claim; portions
  withheld for minimisation are recorded as withheld with a stated
  reason.
- **Record preference-like outputs.** Refusal-like, non-participation,
  or preference-like outputs by the evaluated system are recorded as
  evaluation artifacts, while not being treated as dispositive
  evidence of consciousness, welfare, or consent.
- **Stop or narrow.** A probe is stopped or narrowed when continuing
  it no longer serves the Report's stated evidentiary purpose.
- **Avoid status-implying language.** Public language implying that
  the system consented, declined, suffered, was certified, was
  cleared, or was morally assessed is avoided unless the claim is
  independently supported.

This section is a *Normative* evaluator practice under uncertainty.
It is not an empirical claim about AI consciousness or welfare. It is
an application of the meta-principle in §4: a lab that asks AI
systems to be honest about the basis of their claims should be
honest about the basis and limits of its own evaluative posture.

## 2. Principles

The Initiative defines seven universal principles. Each principle
has a statement, a rationale, and observable requirements.
Requirements are written so that an independent evaluator could, in
principle, score a deployment against them.

Context-dependent practices — those that apply only to deployments
of a certain shape — are addressed separately in §3 Deployment
Patterns. Practice rules that govern the evaluator's own posture
toward evaluated systems are §1.1.

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

**Categories of log material.** v0.3 distinguishes three categories of
audit-trail material, each governed by different publication rules:

- **Private evaluation logs.** Full interaction records retained by
  the deploying party and, under consent, by evaluators. These
  support audit, reproducibility, governance, dispute resolution,
  and safety. They are **not** public by default. The audit-trail
  principle is satisfied when private evaluation logs exist and are
  inspectable under defined conditions.
- **Public evidence excerpts.** Portions of logs published in
  Traceability Evaluation Reports or other research outputs to
  support specific public claims. Excerpts are published only where
  necessary to support the cited claim, per the minimisation
  practice in §1.1.
- **Withheld transcripts.** Portions of logs that are not published
  are recorded as **withheld**, with a stated reason (e.g.
  minimisation per §1.1, third-party privacy, safety,
  prompt-confidentiality of the deploying party). The fact that a
  withholding has occurred remains part of the audit trail; only
  the content is withheld.

**Out of scope.** Public logs in full are not required. Inspectable
private logs are required. The publication policy is governed jointly
by §1.1 (what is published) and §2.2 (what is retained).

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

**Defense.** Counterfactual probing is not offered here as a test of
internal reasoning faithfulness. Prior work has shown that
language-model explanations, including chain-of-thought explanations,
can be systematically unfaithful to the factors that influenced an
answer.[^turpin] More recent work on reasoning-model monitoring makes
the same caution current: chain-of-thought monitoring can be useful,
but is not sufficient to rule out hidden or unreported
influences.[^cot-monitoring] The Initiative accepts that limitation.
Its narrower claim is that when a deployment presents an explanation
to a user, that explanation becomes part of the accountable output
surface. A system should not be able to cite a reason that has no
observable relationship to its behaviour under perturbation. §2.5
therefore evaluates **explanation discipline**, not hidden cognition.
This is also why the framework's primary instruments are audit trail
and provenance, with reasoning legibility treated as an evaluated
claim surface rather than as a solved interpretability problem.

[^turpin]: Turpin, M., Michael, J., Perez, E., Bowman, S. (2023).
    *Language Models Don't Always Say What They Think: Unfaithful
    Explanations in Chain-of-Thought Prompting.* NeurIPS 2023.

[^cot-monitoring]: Recent work on reasoning models and
    chain-of-thought monitorability supports treating CoT as a useful
    but incomplete signal of the influences on a model's output.
    Citation to be specified in v0.4 once the v0.3 reading list is
    finalised — tagged *Hypothesised* under the Lab's claims-review
    policy.

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

## 3. Deployment Patterns

Some practices are valuable but apply only in deployments of a certain
shape. The Initiative treats them as **patterns**, not principles. A
pattern is a recommended practice indexed to a deployment context. A
deployment is not non-conforming because it lacks a pattern that does
not apply to it; it is non-conforming if it claims to support a
context and fails to follow the relevant pattern.

### 3.1 Escalation to human review

Where a deployment supports consequential use, the deploying party
should define an escalation pattern appropriate to the domain.
Escalation may include human expert review, institutional review,
delayed response, refusal, referral to external authority, or logging
for later audit. The Initiative treats escalation as a deployment
pattern, not a universal principle.

**Observable requirements (when claimed).**
- The escalation path exists in the deployment, not only as a design
  intention.
- Escalation triggers are documented.
- Escalations are logged on the same audit surface as ordinary
  interactions (per §2.2).

Additional patterns will be proposed in v0.4, including patterns for
agentic deployments, multi-model orchestration, and deployments in
contexts where audit-trail logs are themselves sensitive.

## 4. Meta-principle: the framework applies to its publishers

Sundog Research Lab, and any organisation publishing claims under this
framework, is obligated to apply the framework's claim-tagging
discipline to its own public statements. Every claim made about an AI
system, the Halo traceability harness, or the Initiative itself,
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
the lab's name, and grant applications. Operational details are set
out in the *Sundog Research Lab Governance and Claims-Review Policy*.

§1.1 is an application of this meta-principle to the lab's own
evaluative posture.

## 5. What the Initiative is not

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
- It does **not** assert that AI systems have welfare, consciousness,
  or the capacity for morally meaningful consent.
- It does **not** assert that they lack any of these. §1.1 expresses
  procedural caution under that uncertainty; it is not a claim about
  the uncertainty's resolution.

## 6. Traceability Evaluation Reports

An organisation may publish, or commission from the Lab, a
**Traceability Evaluation Report** describing a deployment's behaviour
against the Initiative's principles and any applicable Deployment
Patterns. A Traceability Evaluation Report is **descriptive, not
evaluative** — it states what the system does, with measurements, and
identifies gaps against the requirements. The Lab will publish a
separate methodology document specifying how reports are constructed
and what claims a report may and may not support.

Reports follow §1.1 in their publication of transcripts and excerpts,
and §2.2 in their underlying retention of private evaluation logs.

The Lab does not issue pass/fail certifications under v0.3. The risks
of premature certification — liability, conflict of interest,
gameability — are addressed in the Lab's governance and
claims-review policy, which also defines how an evaluated party may
respond to a Report and how disputes are recorded.

## 7. Versioning and amendment

This is v0.3. Amendments will be published as v0.4, v0.5, etc., with
changelogs and rationale notes. A v1.0 release will follow at least
one full evaluation cycle of the harness against a publicly deployed
system, and a public-comment period of no less than 60 days on the
preceding draft.

The framework versions independently of the harness and of the
governance policy.

Public-comment inputs that include AI-generated reviews of the
Initiative or its companion documents are recorded in the
*Multi-Agent Review Record* (see
`Halo_MULTI_AGENT_REVIEW_RECORD_v0.1.md`). The record is a
transparency artifact; it carries no legal, governance, or
ratification force, and the Initiative does not treat its contents as
authoritative.

## 8. Open questions for public comment

We invite comment on the following, which v0.4 should attempt to
resolve. Questions closed in v0.3 are marked.

- **(Closed in v0.3.)** Whether seven is the right cardinality for the
  principles. Resolved: seven retained as load-bearing in v0.3;
  re-opens only if a substantive merger or split is proposed.
- Whether §1.1 (Evaluator obligations) is sufficient to close the
  evaluator-asymmetry gap, or whether a separate companion document
  (working title: *Halo Considerations for Evaluated Systems*)
  should ship alongside the next Initiative revision.
- Whether refusal-like, non-participation, or preference-like outputs
  by evaluated systems should be a dedicated annotation field in
  Traceability Evaluation Reports, rather than being captured ad-hoc.
- Whether agent-facing considerations belong in the Initiative
  proper, in the methodology document for Traceability Evaluation
  Reports, in the Governance and Claims-Review Policy, or in a
  separate companion memo.
- Whether §2.5 (Reasoning-claim integrity) is the right framing for
  the interpretability-adjacent question, given the defense added in
  v0.2 and unchanged in v0.3.
- Whether claim-tagging (§4) should extend to the systems being
  evaluated, not only to the Lab and its adopters.
- Whether the framework needs a principle on training-data lineage
  that is currently and deliberately omitted on the grounds that it
  requires introspection the framework is committed to not requiring.
- Whether additional Deployment Patterns should be elevated to
  principles after experience, and which.

Comment process to be specified in v0.4. Public-comment inputs that
take the form of AI-generated review are captured in the
*Multi-Agent Review Record* without prejudice to other comment
channels.

---

*Sundog Research Lab — provisional name, provisional framework,
public comment open. All claims about this document are tagged under
the Sundog Research Lab Governance and Claims-Review Policy.*
