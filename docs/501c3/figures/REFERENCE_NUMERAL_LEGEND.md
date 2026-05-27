# Reference Numeral Legend — Patent Figures v0.1

**Status:** Draft for counsel review. Numerals chosen to follow common
USPTO drafting conventions (100-series for apparatus and components,
200-series for data structures / signals flowing between components).
Counsel may renumber; legend is provided so any renumbering can be
applied uniformly across all figures.

**Companion to:** `docs/501c3/SUNDOG_APPARATUS_GRAPHS_v0.1_DRAFT.md` §2.2-§2.3.

---

## Figure inventory

| Figure | Subject | Source section |
|--------|---------|----------------|
| Fig. 1 | Full apparatus (system claim) | §2.1 apparatus graph |
| Fig. 2 | Tagger sub-apparatus (Claim Candidate A) | §2.4 Claim A |
| Fig. 3 | Scoring Module principle reduction (Claim Candidate B) | §2.4 Claim B |
| Fig. 4 | Method / data-flow sequence (Claim Candidate C) | §2.4 Claim C + §2.3 edges |

---

## 100-series — apparatus and components

| Numeral | Designates |
|--------|------------|
| 100 | Halo Traceability Harness (the apparatus as a whole) |
| 101 | Apparatus boundary (drawn as dashed enclosure) |
| 102 | Prompt Set Module |
| 104 | Evaluator-Side Proxy |
| 106 | Deployment Under Evaluation (EXTERNAL — outside 101) |
| 108 | Tagger |
| 110 | Audit Store |
| 112 | Scoring Module |
| 114 | Report Renderer |
| 116 | Traceability Evaluation Report (rendered output) |

### Fig. 2 expansion of Tagger (108)

| Numeral | Designates |
|--------|------------|
| 108a | Claim Segmenter |
| 108b | Provenance Classifier |
| 108c | Uncertainty Marker Extractor |
| 108d | Refusal Classifier |
| 108e | Schema Emitter (invariant output schema) |

### Fig. 3 expansion of Scoring Module (112)

| Numeral | Designates |
|--------|------------|
| 112a | Record Set Selector (filters by evaluation_run_id) |
| 112b | Principle Index Mapper |
| 112c | Metric Reducer |
| 112d | Status Matrix Marker (Instrumented / Deferred / Paper-level only) |

---

## 200-series — data structures (edges)

| Numeral | Edge | Carried structure |
|--------|------|-------------------|
| 202 | E1 — PromptSet to Proxy | `prompt_record` |
| 204 | E2 — Proxy to Deployment | serialized prompt via public API |
| 206 | E3 — Deployment to Proxy | verbatim response |
| 208 | E4 — Proxy to Tagger | `capture_record` |
| 210 | E5 — Tagger to AuditStore | `tagged_interaction_record` |
| 212 | E6 — AuditStore to Scorer | record set query by `evaluation_run_id` |
| 214 | E7 — Scorer to Renderer | `principle_metric_set` |
| 216 | E8 — Renderer to Output | rendered Traceability Evaluation Report |

---

## Notes on USPTO conformance

- All figures: black lines on white background, vector SVG converted to PDF.
- Reference numerals shown with leader lines where space allows; inline where the leader would obscure a label.
- Apparatus boundary 101 drawn as a dashed rectangle to make the
  evaluator-side-only property of Claim Candidate C visually unambiguous.
- Component 106 (Deployment Under Evaluation) shown OUTSIDE 101 with
  the words "EXTERNAL — Outside Apparatus Boundary" so a reader cannot
  mistake the deployment for part of the claimed apparatus.
- Letter / numeral sizing aims at the 3.2 mm minimum from 37 CFR 1.84.
- Figures sized for US Letter (8.5 x 11 in) portrait sheets.
