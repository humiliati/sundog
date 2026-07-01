# Chat-v2 R2 Larger-Model — Phase LM-0 Data-Audit Receipt

> 2026-07-01. Run of `R2_LARGER_MODEL_ROUTE_CAMPAIGN.md` Phase LM-0 (CPU, torch-free).
> **Non-promotional. No model run, no GPU, no R2 claim.** Screens whether relation-extraction
> labels survive the campaign's strong surface de-confounds — the DATA gate that decides
> whether the GPU route (LM-1) is worth commissioning.

## Setup

- **Dataset:** FewRel `train_wiki` (the campaign's preferred high-dimensional bank) — loaded
  from the HF **auto-parquet export** (`refs/convert/parquet`), since `datasets 5.0` dropped
  script-based loading and the canonical repo is a loader script. **64 relations × 700
  instances each** (perfectly balanced), 44,800 total.
- **TACRED** is LDC-license-gated (no HF export) → unavailable here (`F3-R2-LM/data`).
- **Axes:** one binary latent per relation (one-vs-rest, 700 pos / 700 sampled neg).
- **Surface controls (campaign §4):** raw lexical bag (reference), **delexicalized bag**
  (entities → `[SUBJ]`/`[OBJ]`), **tf-idf 1–2-gram** (delex), **entity-type + distance + order
  metadata**, **masked-trigger** (drop the 15 highest-PMI delex tokens per relation).
- **Survive** iff *all* strong controls ≤ 0.60 held-out. Script:
  `scripts/chatv2_r2_lm0_data_audit.py`.

## Result: `F3-R2-LM/input` — 0 / 64 relation axes survive

Every relation is surface-decodable, on every strong control:

| control | held-out accuracy range across 64 relations |
| --- | --- |
| delexicalized bag | 0.677 – 0.957 |
| tf-idf 1–2-gram (delex) | 0.740 – 0.954 |
| entity-type + metadata | 0.571 – 0.987 |
| masked-trigger (top-15 PMI dropped) | 0.668 – 0.925 |

The single lowest *individual* control value in the whole table was **0.571** (one relation's
metadata probe); its `surface_max` over all controls was still 0.933. **No relation dropped
below 0.60 on all controls → 0 survivors.** Masking the obvious triggers did *not* rescue
undecodability (0.67–0.93): the residual verbs, entity **types** (the head/tail Wikidata type
pair alone signals the relation), and sentence structure carry the label.

## Reading — the R2 intersection is empty even on the best-case relation data

This is the same tension the GPT-2-small admissions found, now proven at the **data level on
the campaign's strongest candidate**: the label family chosen precisely because it is
*numerous and independent enough for `d_dec ≥ 20`* (relations) is also the family most
**readable from surface cues** (triggers + entity types). The three-level picture:

| level | family | input-undecodable? | computed/decodable |
| --- | --- | --- | --- |
| GPT-2 toy (MVP) | count-parity | yes | not computed |
| GPT-2 real (v2) | agreement | no | computed but input-decodable |
| **LM-0 data** | **FewRel relations** | **no (0/64 survive)** | trigger/type-decodable |

## Disposition — STOP; the H200 buys nothing for R2

The campaign's LM-0 **go-condition (≥24 surface-surviving axes) is not met (0).** Per §7 this
is a **STOP** branch: **do not proceed to LM-1 / GPU.** The phased design did its job — a
CPU data audit killed a GPU spend before it happened. Concretely:

- Renting the H200 for R2 would spend GPU on a gate that already fails at the CPU data step.
- FewRel (best case) fails; TACRED is license-gated; DocRED is script-based (same
  `datasets 5.0` wall, and document-level RE leaks at least as hard).
- The honest R2 boundary is now firm at three independent levels: **the intersection
  {input-undecodable ∧ model-computed ∧ high-dimensional} is empty for the families that
  admit ≥24 axes.**

**Recommendation: bank R1 as the result; do not board the H200 for R2.** `PROMOTE_GATE.md` R2
stays **NOT STARTED**; the NSE "resistant substrate" line stays "to be run." A future R2
attempt would need a genuinely new construction (a label family that is *simultaneously*
high-dimensional and surface-undecodable), which neither relation extraction nor agreement
provides — that is a research problem, not a compute problem.

Cross-refs: `R2_LARGER_MODEL_ROUTE_CAMPAIGN.md` (this phase's spec),
`R2_V2_ADMISSION_RECEIPT.md` (agreement F3-input), `R2_REAL_SUBSTRATE_SPEC.md` (MVP F3),
`PROMOTE_GATE.md` (unchanged — R2 not started). The later H1 state-bank fork
(`H1_V3_STATEBANK_SCOPE.md`) is a new data-family route, not a continuation of this
relation-extraction stop branch.
