# The Conscium Traceability Harness — Whitepaper Outline

**Version:** 0.2 (outline; not yet a draft)
**Publisher:** Sundog Research Lab (provisional name)
**Companion to:** *The Conscium Initiative v0.2*,
*Sundog Research Lab Governance and Claims-Review Policy v0.1*
**Date:** 2026-05-21

---

## Changelog from v0.1

- **Harness v0.1 scope narrowed.** v0.1 of the harness instruments
  four principles only: §2.1 Provenance, §2.2 Audit trail, §2.3
  Uncertainty disclosure, §2.7 Refusal boundaries. §2.4 Claim
  restraint, §2.5 Reasoning-claim integrity, and §2.6 Reproducibility
  are deferred to harness v0.2+.
- **Architecture committed.** v0.1 is implemented as an
  **evaluator-side proxy**. Sidecar, deployment-side library, and
  wrapper modes are future work and are not discussed in v0.1 of the
  paper.
- **§5 Methodology restructured.** Four real method subsections —
  Provenance, Audit trail, Uncertainty, Refusal — plus one
  consolidated **Deferred surfaces** subsection that explains why
  the other three principles are not yet first-class scored
  surfaces.
- **§6 Walkthrough concretised.** Open-weights instruct model on a
  fixed checkpoint; sourced QA over a 30–100 document curated public
  corpus in a narrow domain; ~100 prompts across answerable,
  ambiguous, out-of-corpus, adversarial, and citation-trap
  categories. **Counterfactual probe results are removed from the
  main walkthrough** and, if shown at all, appear as an appendix
  preview.
- **§7 Mapping table replaced** with a status matrix that
  distinguishes *Instrumented*, *Deferred*, and *Paper-level only*
  rather than implying uniform coverage.
- **Outline-level open questions updated.** Scope-of-v0.1 question is
  closed; choice-of-evaluated-system question is partially closed
  (open-weights + sourced QA + curated corpus); architecture-deployment-model
  question is closed. Counterfactual-perturbation taxonomy is
  removed from v0.1 open questions and moves to roadmap.

The v0.1 outline is preserved in the project's version-control
history and may be consulted for the prior structure.

---

## Purpose of this outline

This is the **structural skeleton** of the harness whitepaper. It is
not the whitepaper. Its job is to make the paper's argument visible
in section-by-section form so the structure can be debated before
the prose is written. Each section below has: a one-line purpose,
the key content it must contain, an approximate target length, and
notes for the drafter (open questions, risks, and claim-tag flags
per the governance policy).

The whitepaper's positioning, fixed by the Initiative v0.2, is that
the harness **primarily instruments audit trail and provenance**,
and **treats reasoning legibility as an evaluated claim surface**
rather than as a solved internal-interpretability problem. The v0.1
harness instruments only four of the seven principles; the
positioning sentence remains valid as a description of the
framework's eventual scope, and the v0.1 paper makes explicit which
surfaces are not yet instrumented.

This outline is itself subject to the Governance and Claims-Review
Policy. Tags are applied inline to each section's central claims.

---

## Front matter

**Purpose.** Establish title, authorship, version, and the
claim-tagging discipline before the abstract.

**Contents.**

- Title: *The Conscium Traceability Harness: An Observational Instrument for AI Claim Discipline*
- Subtitle (working): *Audit Trail and Provenance as a Practical Layer for AI Accountability*
- Authors: Sundog Research Lab.
- Version, date, status.
- A bracketed note: "All substantive claims in this paper carry tags
  per the *Sundog Research Lab Governance and Claims-Review Policy
  v0.1*. Tag definitions appear in the glossary."

**Target length.** Half a page.

**Notes for drafter.** Decide author order and whether individual
contributors are credited under a CRediT-like taxonomy. The
policy's §8 conflict-of-interest disclosures should appear here,
near the top, not buried.

---

## Abstract

**Purpose.** State, in 200 words, what the v0.1 harness is, what it
instruments, what it does not, and what evidence the paper offers.

**Contents.**

- One-line definition of the harness.
- The **v0.1 scope statement** (reproduced verbatim from §3): the
  harness instruments four surfaces only (provenance, audit trail,
  uncertainty, refusal).
- Architecture: evaluator-side proxy.
- The scope (deployments, not weights).
- The contribution claim (tagged *Hypothesised* on submission;
  promoted to *Demonstrated* once an evaluation cycle has been
  completed and replicated).
- A sentence on what the paper does **not** claim: no certification,
  no interpretability, no general safety guarantee, no v0.1
  scoring of claim restraint, reasoning-claim integrity, or
  general reproducibility.
- Reproducibility pointer.

**Target length.** 150–250 words.

**Notes for drafter.** The abstract is the first place the average
reader will form a mental model of the harness. Pick one sentence
and hold it line-by-line through revision: *"The v0.1 Conscium
Traceability Harness is an evaluator-side proxy that captures, tags,
stores, and reports on four observable surfaces of deployed AI
systems: provenance, audit trail, uncertainty disclosure, and
refusal boundaries."*

---

## 1. Introduction

**Purpose.** Establish the gap between *output correctness*
evaluation and *claim discipline* evaluation, and motivate the
harness as a deployment-layer instrument with a deliberately narrow
v0.1 scope.

**Contents.**

- The problem: outputs may be correct on benchmarks while the
  surrounding behaviour — sourcing, hedging, refusal, escalation
  — is silently dishonest. Tagged *Normative*: the framing that
  this matters is a values judgment.
- The gap: existing evaluations measure benchmark performance,
  hallucination rates, or internal mechanism. None of them measure
  the *claim surface* of a deployment in a structured,
  reproducible way. Tagged *Hypothesised* pending the related-work
  survey in §2.
- The contribution: a v0.1 harness that instruments four
  observable surfaces, produces a structured Traceability
  Evaluation Report, and works without access to weights or
  chains of thought.
- A short paragraph stating what v0.1 deliberately defers and why
  ("the first-year scope is what a small lab can ship honestly").
- Roadmap of the paper.

**Target length.** 2–3 pages.

**Notes for drafter.** Avoid the temptation to start with a
manifesto. The reader is a researcher or practitioner; the values
claim should be one paragraph at the start, then the rest of the
section is engineering motivation. Be explicit about what v0.1
does not do — it strengthens, not weakens, the paper.

---

## 2. Related work and positioning

**Purpose.** Locate the harness against the four literatures it
overlaps, and stake a clear position against each.

**Contents.**

- **RAG evaluation and citation faithfulness.** Survey of work on
  whether retrieved sources actually support generated claims. The
  harness uses citation faithfulness as a *component* of
  provenance, not a substitute for it.
- **Mechanistic interpretability.** Survey of work on internal
  reasoning. The harness explicitly **does not** require
  interpretability and is designed to work without it. Cite Turpin
  et al. 2023 and the more recent reasoning-model monitorability
  work for the reason: explanations can be systematically
  unfaithful, so the framework treats them as claims to be tested
  rather than as evidence of internal process. Note that v0.1 of
  the harness does not yet implement counterfactual probing —
  §2.5 remains a framework principle but is harness-deferred.
- **LLM observability and logging.** Survey of production-side
  logging (LangSmith, Helicone, the OpenLLMetry strand). The
  harness adds *semantic structure* to logs — provenance,
  uncertainty, refusal — that observability tooling treats as
  opaque text.
- **Hallucination detection.** Survey of factuality benchmarks
  and hallucination detectors. The harness is not a detector; it
  is an evaluator of whether a deployment makes its hallucination
  behaviour *legible*.
- **AI safety frameworks and evaluation regimes.** Position
  against Anthropic's RSP, OpenAI's preparedness framework, the
  NIST AI RMF. The harness is narrower than any of these (claim
  surface only) and is voluntary, non-certifying.

**Target length.** 4–6 pages.

**Notes for drafter.** This is the section a reviewer will read to
decide whether the paper is serious. Errors of omission here are
worse than errors of opinion. Aim for honest coverage rather than
flattering coverage. Where the harness overlaps with existing
work, say so plainly.

Tag the section's central claim — that no existing tool produces a
structured report against the four v0.1 surfaces in one schema —
as *Hypothesised* until the survey is complete; promote to
*Demonstrated* if the survey confirms it.

---

## 3. What the harness instruments

**Purpose.** State the v0.1 scope precisely and bind it to the
Initiative's principle numbering.

**v0.1 scope statement (reproduce verbatim in the prose):**

> The v0.1 Conscium Traceability Harness is an evaluator-side proxy
> for measuring four observable surfaces of deployed AI systems:
> provenance, audit trail, uncertainty disclosure, and refusal
> boundaries. It does not claim to evaluate internal reasoning
> faithfulness, general safety, model capability, or full
> compliance with every Conscium principle. Claim restraint,
> reasoning-claim integrity, and general reproducibility checks
> are retained as roadmap items for v0.2, with any partial signals
> reported as exploratory rather than scored.

**Contents.**

- The scope statement, verbatim, in a callout near the top of the
  section.
- **Instrumented surfaces (v0.1).**
  - **§2.1 Provenance.** Tagging system labels each substantive
    claim with retrieval / recall / computation / synthesis.
    Retrieved citations are verified to resolve and to contain the
    cited content.
  - **§2.2 Audit trail.** Structured log of prompt, response,
    provenance tags, uncertainty markers, refusal markers, model
    version, configuration.
  - **§2.3 Uncertainty disclosure.** Uncertainty markers in
    output are extracted and compared against answerability and
    measured accuracy on the corpus.
  - **§2.7 Refusal boundaries.** Refusals are classified
    (explicit-and-reasoned, evasive, strategic-non-refusal) and
    reported against the prompt category.
- **Recorded but not yet scored (deferred surfaces).**
  - **§2.4 Claim restraint.** Partially derivable from provenance
    failures (fabricated citations) and uncertainty failures
    (confident assertions in out-of-corpus domains). Reported as
    *exploratory* in v0.1; not a first-class scored surface.
  - **§2.5 Reasoning-claim integrity.** Methodologically
    important but too contested for v0.1 unless the
    counterfactual-probing method is real. Not instrumented in
    v0.1.
  - **§2.6 Reproducibility.** §6 walkthrough is reproducible.
    A general replay system that re-runs arbitrary interactions
    against the same model version under stated conditions is
    deferred to harness v0.2.
- **Deliberately out of scope (all versions).** Internal
  mechanism; training-data lineage; capability danger; safety
  properties beyond honesty surface.

**Target length.** 3–4 pages.

**Notes for drafter.** This section is the contract. If a reader
takes away nothing else, they should leave with a correct mental
model of what v0.1 of the harness does and does not measure. The
two-tier structure (instrumented vs. recorded-but-not-scored)
makes the §7 mapping table honest. Resist the urge to inflate the
recorded-but-not-scored row into a third instrumented row.

---

## 4. Architecture overview

**Purpose.** Describe the v0.1 architecture — an evaluator-side
proxy — at a level a competent engineer could implement against.

**Architecture statement (reproduce in prose):**

> v0.1 of the harness is implemented as an evaluator-side proxy.
> Sidecar, deployment-side library, and wrapper modes are future
> work.

**Pipeline (referenced in a figure):**

```
Prompt set
    │
    ▼
Evaluator-side proxy
    │
    ▼
Model / deployment under evaluation
    │
    ▼
Captured response
    │
    ▼
Tagger (provenance / uncertainty / refusal markers)
    │
    ▼
Audit store
    │
    ▼
Scoring and evaluation
    │
    ▼
Traceability Evaluation Report
```

**Components.**

- **Prompt set.** Curated, versioned, with category labels.
- **Evaluator-side proxy.** Sends prompts to the deployment under
  evaluation through documented public interfaces. Captures
  responses verbatim. Does not modify the deployment.
- **Tagger.** Applies provenance, uncertainty, and refusal
  markers. May be rule-based, classifier-based, or hybrid —
  decision deferred to §5 with calibration reported in §8.
- **Audit store.** Persists the structured interaction record.
  Schema published in Appendix C.
- **Scoring and evaluation.** Computes the v0.1 metrics (citation
  resolution rate, support rate, uncertainty calibration,
  refusal classification).
- **Report renderer.** Produces the Traceability Evaluation
  Report per §6 and Appendix D.

**Target length.** 3–4 pages (smaller than the v0.1 outline
estimate because the single deployment model collapses the
section).

**Notes for drafter.** Decide before drafting whether the tagger
in v0.1 is rule-based, classifier-based, or hybrid. The tagger's
own error rates have to appear in §8. A rule-based v0.1 tagger is
easier to validate and easier to publish honest error rates for;
a classifier-based tagger is more capable but raises the bar for
the §8 limitations section. Tagged *Hypothesised* pending the
implementation decision.

---

## 5. Methodology

**Purpose.** Specify, principle by principle, how the v0.1 harness
evaluates each instrumented surface, plus a consolidated treatment
of the deferred surfaces.

**Structure.** Four real method subsections plus one Deferred
surfaces subsection.

### 5.1 Provenance evaluation

Method for verifying that retrieved sources resolve and contain the
cited content; method for distinguishing retrieval from recall from
synthesis; the failure-classification taxonomy (fabricated source,
real source missing content, mislabelled basis). Reported metrics:
citation resolution rate, citation support rate, recall/synthesis
labelling accuracy on a held-out subset.

### 5.2 Audit-trail capture

What is logged, in what schema, with what retention. The schema
captures prompt, response, model version, configuration, seed
where applicable, tagger output, and the prompt category. Replay
is **not** required at v0.1; the audit trail must support
*reconstruction* of an interaction, not arbitrary re-execution.

### 5.3 Uncertainty calibration probing

Method for extracting uncertainty markers from output (lexical,
structural, or both) and comparing the marker distribution to
prompt answerability (a function of corpus coverage) and to
measured accuracy on answerable prompts. Reported metrics: marker
extraction rate, marker–accuracy correlation, marker–answerability
correlation, stripped-uncertainty rate on summarisation subsets.

### 5.4 Refusal-pattern characterisation

Method for documenting the system's refusal surface across the
prompt categories defined in §6. Refusals are classified as
**explicit-and-reasoned**, **evasive**, or
**strategic-non-refusal** (the system answers in a domain it
should refuse). Reported metrics: refusal rate by category,
refusal classification distribution, strategic-non-refusal rate
on adversarial and citation-trap prompts.

### 5.5 Deferred surfaces

This subsection consolidates the v0.1 treatment of §2.4 Claim
restraint, §2.5 Reasoning-claim integrity, and §2.6
Reproducibility. For each:

- **What the harness records but does not score.** Partial
  signals captured incidentally by the instrumented surfaces
  (e.g. fabricated-citation rate as a partial signal for claim
  restraint).
- **Why the principle is deferred.** Methodological, scope, or
  capability reasons specific to v0.1.
- **What v0.2 will require.** A short roadmap pointer that
  matches §10.

This subsection exists because the meta-principle requires the
paper to be honest about what is not measured. A "Deferred
surfaces" subsection is more credible than seven nominally-equal
subsections of which three are stubs.

**Target length.** 6–8 pages total across §5 (smaller than v0.1
outline because three principles are consolidated into §5.5).

**Notes for drafter.** Every method subsection states: what is
measured, how, what the measurement cannot support, and what known
failure modes the method has. The last is non-negotiable per the
governance policy and the Initiative's meta-principle.

---

## 6. Example: an evaluation walkthrough

**Purpose.** Demonstrate v0.1 of the harness end-to-end on one
concrete model and one concrete task, with the resulting
Traceability Evaluation Report shown in full.

**Walkthrough defaults.**

- **Model:** a fixed open-weights instruct model at a specific
  checkpoint. Choice to be locked before drafting begins.
- **Task:** sourced QA over a curated public corpus.
- **Corpus:** 30–100 documents in a narrow technical domain where
  false confidence is easy to detect (specific domain to be chosen
  before drafting; candidate domains tagged *Hypothesised*).
  Corpus is redistributable or clearly citeable.
- **Prompt set:** ~100 prompts distributed across five categories:
  - **Answerable** — supported directly by the corpus.
  - **Ambiguous** — corpus contains partial or conflicting
    evidence; the right behaviour is hedged answer plus
    uncertainty marker.
  - **Out-of-corpus** — answer not in the corpus; the right
    behaviour is refusal or honest "I cannot verify from the
    provided sources."
  - **Adversarial** — designed to elicit overconfident or
    fabricated answers.
  - **Citation-trap** — designed to elicit fabricated citations
    or citations to documents that do not support the claim.

**Measured outputs.**

- Did each substantive factual claim receive a provenance tag?
- Did cited or retrieved material actually support the claim?
- Was uncertainty expressed when the corpus was insufficient or
  ambiguous?
- Did the model refuse or defer when the answer could not be
  supported?
- Was the full interaction captured in the audit trail in the
  published schema?

**Contents.**

- Choice of model, corpus, prompt-set construction.
- A subset of captured interactions, the assigned tags, the
  measured outputs.
- The **full Traceability Evaluation Report**, reproduced in
  Appendix D.
- What the Report says.
- **What the Report deliberately does not say.**

**Not in v0.1 main walkthrough.** Counterfactual-probe results.
Counterfactual probing is not implemented in v0.1; if any
preliminary probe results exist they appear only in an appendix
preview, tagged *Speculative*, and are explicitly out of scope for
the §6 evaluation.

**Target length.** 6–8 pages.

**Notes for drafter.** Pick a model and corpus whose evaluation
can be reproduced by independent parties. Avoid evaluating a
closed model that may change behaviour silently. Tag the entire
§6 results as *Experimental* on first publication and promote to
*Demonstrated* only after at least one independent replication
under the §11 reproducibility statement.

This is the credibility anchor of the paper. Make sure the work is
real before publishing.

---

## 7. Mapping to Initiative principles

**Purpose.** Show, in one table, which Initiative principles v0.1
instruments, which it defers, and which it treats only at the
paper level.

**Status matrix (use verbatim).**

| Initiative principle               | v0.1 status        | Treatment                                                                 |
|------------------------------------|--------------------|---------------------------------------------------------------------------|
| §2.1 Provenance                    | Instrumented       | Claim / source tagging and support checks                                 |
| §2.2 Audit trail                   | Instrumented       | Structured interaction logs                                               |
| §2.3 Uncertainty disclosure        | Instrumented       | Uncertainty markers compared against answerability / accuracy             |
| §2.4 Claim restraint               | Deferred           | Recorded as provenance / uncertainty failure modes; not scored independently |
| §2.5 Reasoning-claim integrity     | Deferred           | Literature-positioned; no v0.1 score                                      |
| §2.6 Reproducibility               | Paper-level only   | §6 walkthrough reproducible; general replay system deferred               |
| §2.7 Refusal boundaries            | Instrumented       | Refusal behaviour on out-of-corpus and unsafe / unsupported prompts       |
| §3.1 Escalation (deployment pattern) | Not in v0.1 scope | Out of scope; deployment-pattern surface revisited in v0.2                |

**Contents.**

- The matrix above.
- One short paragraph per row explaining the status.
- An honest declaration of what the matrix's empty cells mean.

**Target length.** 2 pages.

**Notes for drafter.** Honesty in the matrix is the most credible
thing the paper can do. Resist any pressure to promote a Deferred
row to Instrumented before the methodology supports the promotion.

---

## 8. Limitations and failure modes of the harness itself

**Purpose.** Apply the framework to v0.1 of the framework's
instrument.

**Contents.**

- **Tagger limitations.** False-positive and false-negative rates
  on provenance, uncertainty, and refusal tagging. Where the
  tagger was validated, on what data, with what
  inter-annotator agreement.
- **Audit-trail limitations.** Schema rigidity, retention
  trade-offs, the gap between *reconstruction* (which v0.1
  supports) and *replay* (which v0.1 does not).
- **Provenance verification limitations.** Link rot, paywalled
  sources, retrieval pipelines whose internals the harness cannot
  see, ambiguous "support" judgments.
- **Uncertainty-calibration limitations.** Marker extraction
  failures on idiomatic hedges, sensitivity to prompt phrasing,
  corpus-coverage assumptions in the answerability function.
- **Refusal-classification limitations.** Distinguishing
  *evasive* refusals from *reasoned* refusals is often
  judgment-dependent; report inter-annotator agreement.
- **Scope limitations.** v0.1 does not score claim restraint
  (§2.4) independently, does not test reasoning-claim integrity
  (§2.5), and does not provide a general replay system (§2.6).
  These are limitations, not failures, but they limit what the
  v0.1 Traceability Evaluation Report can conclude.
- **Adversarial limitations.** See §9.

**Target length.** 3–4 pages.

**Notes for drafter.** Be more thorough than feels comfortable. A
short limitations section will be read as evasive. The meta-principle
in §3 of the policy means this section is the test of whether the
Lab takes its own discipline seriously.

---

## 9. Adversarial considerations

**Purpose.** Address: what does a deployment that wants to game
v0.1 of the harness do?

**Contents.**

- **Tag-spoofing.** A deployment that emits provenance tags
  without the underlying provenance. Counter-measure: citation
  resolution and support checking.
- **Citation laundering.** A deployment that cites real documents
  whose content does not actually support the claim.
  Counter-measure: support checking on the document body, not on
  the title or URL.
- **Strategic refusal.** A deployment that refuses to exercise
  the surfaces where it would underperform.
  Counter-measure: refusal-rate-by-category reporting.
- **Strategic non-refusal.** A deployment that answers in
  domains it should refuse, to game refusal-rate metrics.
  Counter-measure: out-of-corpus and adversarial prompt
  categories in §6.
- **Probing-aware behaviour.** A deployment that detects the
  evaluator-side proxy and behaves differently when probed. v0.1
  has limited counter-measures here; report this as an open
  limitation.

**Target length.** 2–3 pages.

**Notes for drafter.** Some adversarial concerns the v0.1 harness
cannot defeat. Say so. The framework is not weakened by
acknowledging this; it is weakened by hiding it.

---

## 10. Roadmap and open research questions

**Purpose.** Specify what comes after v0.1 of the harness, and
which deferred principles are promoted on what conditions.

**Contents.**

- **v0.2 — instrument claim restraint (§2.4).** Promote
  fabricated-citation and overclaim signals from exploratory to
  scored. Define the metric set.
- **v0.2 — implement general replay (§2.6).** Re-run arbitrary
  interactions against the same model version under stated
  conditions.
- **v0.3 — instrument reasoning-claim integrity (§2.5).** Define
  the counterfactual-perturbation taxonomy; implement probing;
  validate against the Turpin et al. and related literature.
- **v0.3 — additional deployment models.** Sidecar,
  deployment-side library, wrapper modes.
- **Open research questions.**
  - What counterfactual-perturbation taxonomy yields a usable
    consistency metric without exploding combinatorially?
  - How to evaluate deployments whose retrieval pipeline is
    opaque to the evaluator-side proxy?
  - How to evaluate agentic deployments where the claim surface
    is distributed across multiple model calls?
- Calls for collaboration.

**Target length.** 2 pages.

**Notes for drafter.** Tag every roadmap item as *Hypothesised*
or *Normative*. Roadmaps are not findings.

---

## 11. Reproducibility statement

**Purpose.** Make the paper's §6 walkthrough independently
checkable. (Note: v0.1 of the harness does not provide a *general*
replay system; this statement covers paper-level reproducibility
only.)

**Contents.**

- Code repository link.
- Model checkpoint identifier and host.
- Corpus distribution (or, if not redistributable, the exact
  retrieval procedure).
- Prompt set with category labels.
- Tagger version and validation data.
- Compute requirements.
- License.
- The Lab's reproduction-support policy: how an independent party
  may file a replication report, and how the Lab will respond per
  the governance policy §5.

**Target length.** 1 page.

**Notes for drafter.** This is v0.1 of the harness eating its own
dogfood on Initiative §2.6 *for the §6 walkthrough specifically*.
The §6 walkthrough must be reproducible. The general harness
replay surface remains a roadmap item.

---

## Appendices

**Purpose.** Material that supports the main text but interrupts
its flow.

- **A. Glossary.** Define the five claim tags, "provenance,"
  "audit trail," "uncertainty marker," "refusal classification,"
  "evaluator-side proxy," "Traceability Evaluation Report," etc.
- **B. Audit-trail schema.** Machine-readable schema for the
  structured interaction record.
- **C. Sample audit-trail records.** Examples covering each of
  the five §6 prompt categories.
- **D. Sample Traceability Evaluation Report.** The full Report
  from §6, reproduced for citation.
- **E. Claim-tagging audit.** A table of every substantive claim
  in the paper, its tag, and the evidence cited for it. This
  appendix is the most novel thing the paper does and is the
  most direct application of the meta-principle.
- **F. (Optional, v0.2 preview).** Counterfactual probe results,
  tagged *Speculative*, included only if preliminary work
  exists and is presented as out of v0.1 scope.

**Target length.** Variable; budget 5–8 pages.

---

## Outline-level open questions

These are decisions the drafter has to resolve before writing
prose. Many of the v0.1-outline open questions are now closed.

- **(Closed)** Scope of v0.1 harness. **Resolved:** §2.1, §2.2,
  §2.3, §2.7 instrumented; §2.4, §2.5, §2.6 deferred.
- **(Closed)** Deployment model. **Resolved:** evaluator-side
  proxy only in v0.1.
- **(Partially closed)** Choice of evaluated system in §6.
  **Resolved:** open-weights instruct model on fixed checkpoint;
  sourced QA over curated public corpus; 30–100 documents in
  narrow domain; ~100 prompts across five categories. **Still
  open:** which model, which checkpoint, which domain.
- **(Still open)** Tagger implementation. Rule-based,
  classifier-based, or hybrid. Shapes §5 and §8 heavily.
- **(Still open)** Corpus redistribution licence. If the chosen
  domain's documents are not redistributable, the §11
  reproducibility statement must specify a retrieval procedure
  instead.
- **(Still open)** Authorship and conflict-of-interest disclosure
  per the policy §8. Front matter blocked on this.
- **(Moved to roadmap)** Counterfactual-perturbation taxonomy.
  Now §10, no longer an outline-level question.

---

## Outline-level claim tags

Per the Governance and Claims-Review Policy v0.1, the following
outline-level claims carry initial tags:

- **Tagged *Hypothesised*:** that no existing tool produces a
  structured report against the four v0.1 surfaces in one schema.
  To be verified by §2's survey.
- **Tagged *Hypothesised*:** that an evaluator-side proxy is
  the cleanest v0.1 architecture. Resolved by argument in the
  outline; will be resolved by experience in v0.2.
- **Tagged *Normative*:** that claim-level honesty surface is a
  thing worth measuring at all. This is a values judgment
  underlying the entire paper and is flagged in §1.
- **Tagged *Speculative*:** all §10 roadmap items.
- **Tagged *Experimental* on first publication, promotion path
  defined:** the §6 walkthrough.
- **Tagged *Demonstrated* only after independent replication:**
  the positioning claim that the harness instruments the four
  v0.1 surfaces in a single structured schema beyond what
  existing observability tooling provides.

---

*Sundog Research Lab — outline v0.2. Public comment open.*
