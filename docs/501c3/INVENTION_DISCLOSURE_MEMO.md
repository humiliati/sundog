# Invention Disclosure Memo
## Halo Traceability Harness for AI Claim-Boundary Evaluation

Confidential attorney review draft. Not for public distribution.
Prepared by: Jeffery Hughes
Date: May 26 2026
Current owner/controller: Stellar Aqua LLC, subject to counsel review
Inventor candidates: Jeffery Wade Hughes Jr

## 1. One-Paragraph Summary

This invention is a method and apparatus for evaluating deployed AI systems through an evaluator-side proxy. The apparatus captures prompt/response interactions through public interfaces, decomposes responses into claim-level records, tags each claim for provenance, uncertainty, refusal/boundary behavior, and evidence status, stores the resulting records in an audit ledger, and reduces the run to a structured traceability report.

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
public-interface query ->
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
- public-interface capture
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
- probe results: [insert exact trial counts and paths]
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

List every possible disclosure with exact dates:
- GitHub repository public availability
- sundog.cc pages
- Reddit posts
- d2jsp post
- outreach packets
- emails or pitch decks
- public demos
- conversations with companies, if not under NDA

For each:
- date
- URL or file path
- what was disclosed
- whether it disclosed implementation details or only concept

## 13. Inventorship Notes

Who conceived:
- evaluator-side proxy
- claim-level schema
- boundary classifier
- report structure
- audit ledger
- probe/eval design

AI tool use:
- identify where AI assisted drafting, coding, summarizing, or implementation
- identify the human conception decisions
- preserve prompts/handoffs if useful

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
- claim schema examples (to attach)
- eval result tables (to attach)
- screenshots (to attach)
- source file paths (to attach)
- public-disclosure timeline (to draft)
- related-work notes (to draft)