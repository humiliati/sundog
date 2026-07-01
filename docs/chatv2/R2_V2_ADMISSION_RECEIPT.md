# Chat-v2 R2 v2 — Admission Receipt (agreement/controller family)

> 2026-07-01. Run of the admission screen from `R2_V2_RELATIONAL_SUBSTRATE_SPEC.md` §5.
> **Non-promotional.** No R2 body-resistance claim, no promotion, no public / R3 / world-model
> language. Screens the spec's **primary** family (agreement/controller) against the
> intersection: *input-undecodable* AND *GPT-2-computed* AND *balanced/dimensional*.

## Setup

- Corpus: **WikiText-103** human prose (pyarrow; the corpus-admission prerequisite — met).
- Substrate: GPT-2 small residual stream at the predict-site (token before a masked
  clear-number auxiliary), pretrained + random-init floor, CPU. Script:
  `scripts/chatv2_r2_v2_admission.py`.
- Labels (agreement family): `num` (SG/PL), `tense` (pres/past), `fam_be`, `fam_have`,
  `attractor_mismatch` (nearest-noun number ≠ true number).
- Gates: **balance** ∈ [0.40,0.60]; **input-undecodability** raw-token linear probe ≤ 0.60;
  **GPT-2-carried** ≥ 0.65.

## Result (full, 1000 sites; identical pattern at the 200-site smoke)

| label | balance | raw-token probe | GPT-2 pretrained | random-init floor | in intersection? |
| --- | --- | --- | --- | --- | --- |
| num | 0.353 | **0.700** | **0.963** | 0.637 | no |
| tense | 0.654 | 0.770 | 0.807 | 0.633 | no |
| fam_be | 0.886 | 0.857 | 0.903 | 0.833 | no |
| fam_have | 0.100 | 0.870 | 0.893 | 0.850 | no |
| attractor_mismatch | 0.224 | 0.737 | 0.843 | 0.693 | no |

**Intersection: 0 / 5. Verdict: `F3-R2-v2/input`.**

## Reading — the R2 tension, now empirical on both families

GPT-2 **genuinely computes** number agreement (0.96), exactly as the interpretability
literature says. But a **linear raw-token probe reads it too** (0.70 > 0.60) — because the
**subject that determines the number is in the input**. So the agreement family is
*computed* but **input-decodable**; it fails the de-confound. The two candidate families now
sit on **opposite sides** of the same divide:

| family | input-undecodable? | GPT-2-computed? |
| --- | --- | --- |
| count-parity (MVP) | **yes** (parity is nonlinear) | **no** (GPT-2 doesn't count) |
| agreement (v2) | **no** (subject is in the input) | **yes** |

The only sliver where agreement becomes undecodable is the **attractor** subset (nearest noun
mismatches the controller) — but here `attractor_mismatch` was itself raw-readable (0.737),
imbalanced (0.22), and it is still **one axis** (number). Even a clean attractor-controlled
setup gives a narrow, ~1-dimensional signal, **nowhere near `d_dec ≥ 20`**. The random-init
floor is also high (0.63–0.85), so the objective margin over floor would be small regardless.

## Disposition

**R2 gate not cleared, and the admission screen shows *why* — cheaply, before any verdict
harness.** The intersection {input-undecodable ∧ genuinely-computed ∧ ≥24 independent axes}
is **nearly empty on GPT-2-small**: the undecodable families aren't computed, the computed
families are decodable, and the one undecodable-computed sliver (attractor agreement) is
low-dimensional. Honest onward options (owner's call — none taken here):

1. **Attractor-controlled micro-R2** — filter to hard attractor cases; would at best be a
   *scoped* single-axis body-resistance read, **not** the `d_dec ≥ 20` gate.
2. **Larger model + genuine relational/semantic tasks** — a bigger build (likely GPU /
   annotation-assisted); the intersection may open up on a more capable substrate.
   Scope-and-hold campaign: `R2_LARGER_MODEL_ROUTE_CAMPAIGN.md`.
3. **Bank R1 as the result** — accept that the R2 gate (real-computed ∧ input-undecodable ∧
   high-dim) is very hard on a small pretrained LM; record this admission finding as the
   honest R2 boundary. `PROMOTE_GATE.md` R2 stays **NOT STARTED**; the NSE page's "resistant
   substrate" line stays "to be run."

Cross-refs: `R2_REAL_SUBSTRATE_SPEC.md` (MVP RUN 1, F3-R2), `R2_V2_RELATIONAL_SUBSTRATE_SPEC.md`
(this screen's spec), `PROMOTE_GATE.md` (unchanged).

## Reproduction gotchas

- This box reaches HuggingFace but not the Gutenberg/GitHub corpus routes used in earlier
  sketches.
- On the local Python 3.14 stack, `pyarrow` `iter_batches` segfaulted; use `read_table`
  for WikiText parquet reads.
- Import/read-order matters: importing `torch` before the `pyarrow` corpus read can
  segfault. Read the corpus first, then import/initialize torch and transformers.
