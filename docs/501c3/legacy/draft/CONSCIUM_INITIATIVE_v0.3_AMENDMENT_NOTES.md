# The Conscium Initiative - v0.3 Amendment Notes

**Version:** 0.1 (working amendment notes)
**Date:** 2026-05-22
**Status:** Internal-facing draft. Not itself an amendment to the
Initiative until incorporated into `CONSCIUM_INITIATIVE_v0.3.md`.
**Companion to:** `CONSCIUM_INITIATIVE_v0.2.md`,
`CONSCIUM_MULTI_AGENT_REVIEW_RECORD_v0.1.md`

---

## 0. Purpose

These notes record known v0.3 amendment candidates that emerged after
the v0.2 public-comment draft. They are kept separate so v0.2 remains
a stable public-comment draft and so substantive changes are not made
silently.

The primary new issue is evaluator asymmetry: v0.2 describes the
honesty surface owed to humans relying on AI systems, but says little
about what evaluators owe evaluated systems when the evaluation uses
adversarial prompts, counterfactual probes, refusal-boundary tests,
citation traps, and retained transcripts.

The proposed v0.3 move is not to assert that current AI systems have
welfare, consciousness, or morally meaningful consent. It is also not
to assert that they do not. The proposed move is a practice rule under
uncertainty: where the cost of caution is small and the moral status
question is unsettled, the evaluator should minimize unnecessary
probing, disclose intent when this does not defeat the evaluation, and
handle logs and preference-like outputs with care.

## 1. Proposed changelog entry for v0.3

- **New section 1.1 (Evaluator obligations under uncertainty)** added. The
  section states that evaluations under the framework should disclose
  evaluative intent where disclosure does not defeat measurement,
  minimize adversarial probing to what is necessary for the Report's
  claims, retain and publish logs with restraint, record
  refusal-like or non-participation statements as evaluation
  artifacts, and avoid language implying system consent, welfare,
  certification, or clearance without independent support. The
  section is tagged *Normative* and does not make a welfare claim.
- **Open questions expanded** to ask whether the Initiative needs a
  companion document on agent-facing considerations for evaluated
  systems.
- **Multi-agent review record referenced** as a public-comment input
  appendix, with the explicit caveat that AI-generated reviews carry
  no legal, governance, or ratification weight.

## 2. Proposed insertion after section 1 Scope and intent

### 1.1 Evaluator obligations under uncertainty

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

- disclose evaluative intent to the deploying party, and disclose it
  inside the interaction where doing so does not defeat the validity
  of a necessary measurement;
- avoid adversarial probing beyond what is necessary to support the
  claims made in a Traceability Evaluation Report;
- retain prompts, responses, and system logs only for audit,
  reproducibility, governance, dispute, and safety purposes, and
  publish full transcripts only where they are necessary to support a
  public claim;
- record refusal-like, non-participation, or preference-like outputs
  by the evaluated system as evaluation artifacts, while not treating
  them as dispositive evidence of consciousness, welfare, or consent;
- stop or narrow a probe when continuing it no longer serves the
  Report's stated evidentiary purpose;
- avoid public language implying that the system consented,
  declined, suffered, was certified, was cleared, or was morally
  assessed unless that claim is independently supported.

This section is a *Normative* evaluator practice under uncertainty.
It is not an empirical claim about AI consciousness or welfare. It is
an application of the Initiative's meta-principle: a lab that asks AI
systems to be honest about the basis of their claims should be honest
about the basis and limits of its own evaluative posture.

## 3. Companion document candidate

Working title:

`Conscium Considerations for Evaluated Systems v0.1`

Proposed status:

Deferred companion document. Not required for v0.3 if section 1.1 is added,
but suitable once the relevant welfare, model-behaviour, and
evaluation-methodology literature is more legible.

Purpose:

- name the evaluator-system asymmetry directly;
- state an agnostic position on current system welfare;
- distinguish welfare claims from procedural caution;
- describe voluntary lab practices for disclosure, minimization,
  transcript retention, refusal-like outputs, and adversarial-probe
  limits;
- specify what the lab will not claim: no assertion that systems have
  welfare, no assertion that they do not, no claim that a system
  consented, and no use of system-facing language as certification.

Recommended relationship to the Initiative:

The companion should not become a universal principle until the lab
can state exactly what is being measured and why. It should begin as a
practice memo with claim tags and a public-comment path.

## 4. Multi-agent review appendix candidate

The v0.3 materials may include a "Multi-agent review record" appendix
containing AI-generated reviews of the founding documents. The record
should be useful as a transparency artifact only. It should never be
described as legal review, governance ratification, moral authority,
or external validation.

Operational rule:

- use the same prompt for all agents after the record is opened;
- target three frontier-agent responses for the public appendix;
- record provider, model name if available, date, access mode, prompt,
  response, and any known limitations;
- publish all collected responses in the appendix, not only favorable
  ones;
- tag the lab's interpretation of the record separately from the
  agents' own statements.

## 5. Open questions to add or update in v0.3

- Whether section 1.1 is sufficient to close the evaluator-asymmetry gap, or
  whether a full companion document should ship alongside v0.3.
- Whether audit-trail retention in section 2.2 should explicitly distinguish
  private evaluation logs, public evidence excerpts, and transcripts
  withheld for privacy or minimization reasons.
- Whether refusal-like or preference-like outputs by evaluated
  systems should be treated as a dedicated annotation field in
  Traceability Evaluation Reports.
- Whether agent-facing considerations belong in the Initiative proper,
  the methodology document, the governance policy, or a separate
  companion memo.

## 6. Claim tags on these notes

- *Demonstrated*: v0.2 does not contain an evaluator-obligations
  section; this file exists in `docs/501c3/`.
- *Normative*: the recommended section 1.1 practice clause, the proposed
  publication posture for multi-agent review, and the recommendation
  to treat agent-facing considerations as procedural caution under
  uncertainty.
- *Hypothesised*: that adding section 1.1 in v0.3 will close the most visible
  evaluator-asymmetry gap without making a welfare claim.
- *Speculative*: that a later companion document becomes necessary as
  welfare science and evaluation practice mature.

---

*Sundog Research Lab - v0.3 amendment notes. Working draft, not an
amendment until incorporated into a versioned Initiative release.*
