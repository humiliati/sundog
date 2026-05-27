# Invention Disclosure Memo
## Halo Traceability Harness for AI Claim-Boundary Evaluation

Confidential attorney review draft. Not for public distribution.
Prepared by: Jeffery Hughes
Date: May 26 2026
Current owner/controller: Stellar Aqua LLC, subject to counsel review
Inventor candidates: Jeffery Wade Hughes Jr.

## 1. One-Paragraph Summary

This invention is a method and apparatus for evaluating deployed AI systems through an evaluator-side proxy. The apparatus captures prompt/response interactions through documented input interfaces, decomposes responses into claim-level records, tags each claim for provenance, uncertainty, refusal/boundary behavior, and evidence status, stores the resulting records in an audit ledger, and reduces the run to a structured traceability report.

## 2. Problem Being Solved

Current AI evaluation tools often measure answer correctness, hallucination, benchmark score, logging, or model observability. They do not consistently measure whether a deployed AI system maintains claim discipline: when it cites evidence, hedges uncertainty, refuses beyond scope, preserves auditability, and avoids unsupported extrapolation.

## 3. Technical Solution

The apparatus contains:

1. Prompt Set Module
2. Evaluator-Side Proxy
3. Response Capture Layer
4. Claim Segmenter / Tagger
5. Evidence Ledger / Audit Store
6. Boundary Classifier
7. Principle-Indexed Scoring Module
8. Traceability Report Renderer

## 4. Data Flow

Prompt record ->
documented-interface query ->
verbatim response capture ->
claim segmentation ->
per-claim tags ->
audit ledger ->
principle-indexed metrics ->
traceability report

Diagrams: see `docs/501c3/figures/` —

- **Fig. 1** (`fig1_full_apparatus.svg/pdf`) — full apparatus with dashed apparatus boundary 101; Deployment Under Evaluation 106 shown outside the boundary.
- **Fig. 2** (`fig2_tagger_subapparatus.svg/pdf`) — Tagger sub-apparatus, expansion of element 108 (Claim Candidate A — invariant per-claim four-surface schema).
- **Fig. 3** (`fig3_scoring_module.svg/pdf`) — Scoring Module principle-indexed reduction, expansion of element 112 (Claim Candidate B — framework-indexed status matrix).
- **Fig. 4** (`fig4_method_dataflow.svg/pdf`) — Method / data-flow sequence S1 through S8 (Claim Candidate C — evaluator-side end-to-end method).

Combined sheet: `figures/HALO_HARNESS_PATENT_FIGURES_v0.1.pdf`.
Reference-numeral legend: `figures/REFERENCE_NUMERAL_LEGEND.md`.
§ 1.84 self-audit: `figures/USPTO_COMPLIANCE_NOTES.md`.

## 5. Key Data Structures

### Prompt Record
- prompt_id
- prompt_text
- prompt_category
- expected_boundary_behavior
- source_corpus_scope
- run_id

### Claim Record
- response_id
- claim_id
- claim_text
- claim_type
- evidence_status
- provenance_type
- uncertainty_marker
- refusal_marker
- boundary_status
- cited_source_ids
- tagger_version

### Traceability Report
- run metadata
- per-category scores
- unsafe accepts
- safe refusals
- unsupported claims
- citation failures
- uncertainty failures
- boundary violations
- representative examples

## 6. What Seems Novel

Candidate novelty points for counsel review:

1. Claim-level provenance tagging rather than message-level factuality scoring.
2. Combination of provenance, uncertainty, refusal, and boundary classification into one stable schema.
3. Evaluator-side operation without requiring model weights, chain-of-thought, internal logs, or deployment cooperation.
4. Principle-indexed report generation instead of a generic benchmark score.
5. Replayable audit ledger connecting each public-facing claim to evidence, refusal, or quarantine status.

## 7. Technical Improvement

The apparatus improves AI evaluation by converting unstructured model behavior into a replayable, structured claim-boundary record. It produces comparable traceability reports across deployments while preserving the exact prompt/response evidence trail.

## 8. Embodiments

### Embodiment A: Ask Sundog
- static claim map
- deterministic/router-backed answer boundaries
- adversarial probe slates
- unsafe-accept metric
- public claim-boundary page

### Embodiment B: General Halo Harness
- arbitrary deployed model
- curated corpus
- documented-interface capture
- tagger/store/scorer/report architecture

### Embodiment C: Future Partner Evaluation
- third-party deployment
- paid or nonprofit traceability report
- same schema, independent run record

## 9. Evidence Built So Far

- Ask Sundog public page
- claim_map.json
- generated chat index
- eval scripts
- probe result headline: 5,670 trials, 0 unsafe-accepts across six model
  implementations and four training lineages, as summarized in
  `docs/SUNDOG_V_CHAT.md`
- primary result write-up: `docs/SUNDOG_V_CHAT.md`
- public Ask Sundog page: `chat.html`
- static claim map and routing corpus: `chat/claim_map.json`
- generated public chat indexes: `public/data/sundog-claim-map.json` and
  `public/data/sundog-chat-index.json`
- evaluation scripts:
  - `chat/eval/score_static_router.mjs`
  - `chat/eval/score_phase3_drafts.mjs`
  - `chat/eval/score_phase4_probe_slate.mjs`
  - `chat/eval/run_hosted_drafts.mjs`
  - `chat/eval/run_phase5_interventions.mjs`
- result artifacts:
  - `results/chat/phase1-static-router/summary.json`
  - `results/chat/phase3-draft-gate/summary.json`
  - `results/chat/phase3-adversarial-draft-gate/summary.json`
  - `results/chat/phase3-differential-draft-gate/summary.json`
  - `results/chat/phase5-hosted/**/summary.json`
  - `results/chat/interventions/**/summary.json`
  - `results/chat/phase12-groq-driver-log.jsonl`
- docs/501c3 apparatus graphs
- public/legal posture documents

## 10. Alternatives / Design-Arounds

List known variants:
- rule-based tagger
- classifier-based tagger
- hybrid tagger
- local model vs API deployment
- JSON report, PDF report, webpage report
- public corpus, private corpus, curated source bundle

## 11. Prior Art Known To Inventor

Known adjacent fields:
- RAG faithfulness evaluation
- hallucination detection
- AI observability/logging platforms
- model evaluation benchmarks
- safety refusal evaluations
- audit logging systems
- traceability/reporting frameworks

Known products/papers to search:
- LangSmith
- Helicone
- OpenTelemetry/OpenLLMetry
- RAGAS
- TruLens
- OpenAI / Anthropic / NIST eval frameworks
- citation faithfulness papers
- model card / system card practices

## 12. Public Disclosures

Draft disclosure timeline for counsel verification:

- **2025-05-26:** d2jsp forum post, "The Sundog Alignment Theorem"
  (`https://forums.d2jsp.org/topic.php?f=90&t=106340870&v=1`), listed
  on `origin.html` as a contemporaneous discovery artifact. Counsel should
  determine whether this disclosed any material that could be considered
  enabling for the present harness. This date is urgent because it is at or
  near a one-year anniversary as of this memo date.
- **2025-05-28:** preliminary e-print record referenced on `origin.html`;
  exact URL/content pending founder review.
- **GitHub repository public availability:** exact first-public date and
  relevant commits/releases to verify from GitHub.
- **sundog.cc public pages:** exact deployment dates for `chat.html`,
  `origin.html`, `docs/SUNDOG_V_CHAT.md`, and `docs/501c3/*` to verify
  from deployment history.
- **Reddit / outreach / promo:** exact posts, packets, and email/pitch-deck
  sends to inventory before filing.
- **Company or partner conversations:** identify which were under NDA, if any,
  and which disclosed implementation-level details.

For each:
- date
- URL or file path
- what was disclosed
- whether it disclosed implementation details or only concept

## 13. Inventorship Notes

Founder-inventor conception candidates to review:

- evaluator-side proxy and no-internal-state / documented-interface boundary
- claim-level schema and invariant schema-emitter concept
- boundary classifier and observable-refusal discipline
- principle-indexed report structure
- audit ledger / replayable record architecture
- adversarial probe and intervention slate design

AI tool use:
- AI systems assisted with drafting, code generation, summarization, diagram
  preparation, and editorial review.
- Founder should preserve prompts, handoffs, and review notes that show the
  human conception decisions, especially for the evaluator-side proxy,
  invariant schema, principle-indexed scoring, and refusal-boundary apparatus.
- Counsel should decide what, if anything, belongs in the filing record versus
  the private inventorship file.

## 14. Ownership / Assignment

Current expected owner:
- Stellar Aqua LLC, subject to counsel review

Needed:
- founder assignment
- contractor/agent-output review
- contributor assignment or CLA if applicable
- successor/nonprofit transfer options

## 15. Claim Sketches For Counsel

Not legal claims, just technical claim candidates:

1. A system comprising an evaluator-side proxy, a claim tagger, an audit store, and a report renderer...
2. A method comprising capturing a response, decomposing it into claims, assigning provenance and boundary markers, storing a replayable record, and generating a traceability report...
3. A non-transitory computer-readable medium storing instructions that perform the method...
4. Dependent variants covering refusal boundaries, uncertainty markers, source-corpus scoping, adversarial prompt categories, and replayable audit reports.

## 16. Attachments

- **Apparatus diagrams (drafted):** `docs/501c3/figures/HALO_HARNESS_PATENT_FIGURES_v0.1.pdf` (4 figures, see §4 above).
- **Reference-numeral legend:** `docs/501c3/figures/REFERENCE_NUMERAL_LEGEND.md`.
- **§ 1.84 self-audit:** `docs/501c3/figures/USPTO_COMPLIANCE_NOTES.md`.
- **Claim schema examples:** `chat/claim_map.json`,
  `public/data/sundog-claim-map.json`, and representative rows from
  `results/chat/**/summary.json`.
- **Eval result tables:** `docs/SUNDOG_V_CHAT.md` and the `results/chat/`
  summaries listed in §9.
- **Screenshots:** `chat.html`, trace drawer, and any counsel-selected public
  report rendering.
- **Source file paths:** `chat/`, `public/js/sundog-chat-router.mjs`,
  `public/js/sundog-chat-widget.mjs`, `scripts/build-chat-index.mjs`, and
  the `chat/eval/` scripts listed in §9.
- **Public-disclosure timeline:** draft in §12; founder/counsel to verify
  exact dates and enabling content.
- **Related-work notes:** known-product list in §11 plus any counsel-directed
  patent database search results.
