# Chat-v2 R2 v2 - Relational Real-Substrate Scoping

> 2026-07-01, scoping draft. This is not an R2 run and not a
> promotion packet. It is the redesign opened after
> `R2_REAL_SUBSTRATE_SPEC.md` MVP RUN 1 filed `F3-R2`: the count-parity
> bank was too thin on the reachable corpus, and more importantly the
> latent family was probably not computed by GPT-2.
>
> **Admission result:** the primary agreement/controller family was run and filed
> `F3-R2-v2/input` in `R2_V2_ADMISSION_RECEIPT.md`: GPT-2 computes agreement, but
> raw input linearly decodes it too. No final R2 verdict harness was opened.

## 0. Status

R2 remains effectively **not started** on the promotion gate. The MVP produced
no body-resistance read:

- **surface failure:** TinyStories was the only readable corpus on the box, and
  only 18/51 count-parity attributes survived balance plus raw-input
  undecodability gates;
- **structural failure:** count parities are cleanly input-undecodable, but a
  pretrained GPT-2 was never trained to count feature occurrences and take
  parity, so the family likely moves from `F3-R2` to `F2-R2` on a richer corpus.

The v2 task is therefore not "rerun R2 with more text." It is to find a
non-synthetic latent family in the intersection:

1. the label is **not linearly readable from raw input tokens**;
2. the label is **plausibly computed or carried by pretrained GPT-2**;
3. enough independent labels survive to make `d_dec >= 20` meaningful;
4. a random-init same-architecture floor stays materially lower.

Only after that intersection is demonstrated should a final R2 verdict harness
be frozen.

## 1. The design crux

The R1 toy made the hard part easy by construction: the model was trained on
input-undecodable computed latents. In R2, GPT-2 is fixed. We do not get to
choose arbitrary nonlinear labels and then assume the model represents them.

This creates the real v2 constraint:

| desideratum | what count-parity gave | why it failed |
| --- | --- | --- |
| input-undecodable | strong: parity defeats a linear bag-of-token probe | too detached from GPT-2's pretraining objective |
| pretrained-computed | weak: GPT-2 has no reason to store regex-count parity | likely `F2-R2` even if corpus fixes `F3-R2` |
| high-dimensional bank | corpus dependent | TinyStories made many attributes constant |

The replacement family should be relational, positional, or discourse-local:
features whose label depends on how tokens stand in relation to one another,
not on a single lexical cue, and whose computation is useful for next-token
prediction.

## 2. Corpus admission

The corpus is now an admission gate, not a convenience detail.

**Required for verdict-bearing R2 v2:**

- human-written or human-edited prose;
- accessible from this machine through HuggingFace or a local file;
- readable without adding fragile notebook-only dependencies to the run path;
- enough passages at `seq_len >= 128` after filtering;
- enough syntactic and discourse variety that candidate relational labels are
  not near-constant.

**Allowed routes:**

- **Primary:** WikiText or another HF-hosted human-text dataset, if `pyarrow` or
  an equivalent plain-text loader is installed and pinned.
- **Fallback:** a HF-hosted plain-text corpus available through
  `hf_hub_download` without parquet.
- **Operator file:** `--corpus-file <path>` containing one passage per line or
  paragraph-separated human prose.

**Not verdict-bearing:** TinyStories may remain a smoke corpus for loader and
probe plumbing, but it cannot carry an R2 v2 verdict unless the document is
explicitly amended. Its simple vocabulary already caused a bank-thinning F3.

**Corpus admission thresholds:**

- at least 3,000 usable passages for the admission screen;
- at least 1,000 held-out test passages reserved before any final verdict run;
- no single source document contributes more than 5% of the final sample, unless
  the corpus is already shuffled and document boundaries are unavailable;
- the candidate-latent screen must produce at least 32 candidate binary labels
  before the balance/input gates, and at least 24 survivors after them.

If these are not met, file **F3-R2-v2/corpus** and stop before any body-resistance
claim.

## 3. Latent-family candidates

Count-parities are retired from verdict-bearing R2. They may remain harness
smokes because they are cheap and deconfounded, but they are no longer the
primary R2 family.

### A. Agreement / controller features - primary candidate

Labels derived from naturally occurring agreement-like contexts:

- subject/controller number for a later verb or pronoun;
- presence of an intervening attractor with the opposite number;
- whether the next agreement-bearing token matches the nearest noun or a
  non-nearest controller;
- tense or auxiliary agreement when the deciding cue is outside a short local
  window.

Why this is attractive: GPT-2's next-token objective makes agreement useful,
while the label can be made relational by withholding the target token and
using lexical splits.

Main risk: raw tokens may leak number morphology. The deconfound must include
matched or held-out lexical splits, not just a bag-of-token probe.

### B. Cloze-required grammatical state - secondary candidate

Labels attached to a prediction site, where the label is the grammatical or
semantic state needed to predict the next token:

- singular/plural required at the blank;
- determiner class required at the blank;
- tense/aspect class required at the blank;
- quote/dialogue state or sentence-mode state needed at the blank.

The target token and a small local window around it must be removed or masked
from the raw-input probe. Otherwise this becomes surface prediction, not a
computed-state test.

### C. Coreference-ish role relations - exploratory candidate

Labels based on small discourse configurations:

- final pronoun/mention refers to the first or second named entity;
- recipient vs giver role after a transfer verb;
- repeated-name identity relation across a passage.

This family is closer to the intended "computed relation" story, but it is
harder to extract cleanly without a parser or annotation model. If a model is
used to label examples, its role must be frozen and audited; labels cannot be
derived from GPT-2 activations.

### D. Positional/order relations - smoke candidate only unless strengthened

Examples:

- entity A appears before entity B;
- a quoted span opens before a named entity and closes after it;
- a discourse marker occurs before vs after the main entity mention.

These are useful for testing relational plumbing, but many are likely readable
from raw token counts plus position proxies. They should not be the first
verdict-bearing family.

## 4. Strengthened de-confounds

The MVP raw-token probe was necessary but not sufficient for relational labels.
R2 v2 needs three input checks:

1. **raw-token linear probe:** bag/count features over the observed passage;
2. **masked-window raw probe:** same probe after removing the target token and a
   small local window around the decision site;
3. **lexical split or pair split:** train and test must not share the lexical
   cue that makes the label trivial.

A latent survives input-undecodability only if the raw probes stay at or below
the pre-registered chance band on held-out data. The default ceiling remains
`<= 0.60` accuracy unless the final prereg tightens it.

Do not rescue a family by weakening the raw-input gate. If raw input reads it,
the label is input structure, not a computed-state witness.

## 5. Admission screen before final R2

Build an admission harness before rebuilding the verdict harness:

`scripts/chatv2_r2_v2_admission.py`

The admission harness should be cheap, CPU-only, and explicitly non-promotional.
It screens corpus plus latent family and returns one of:

- **ADMIT-R2-v2:** enough relational labels survive, GPT-2 carries them, and a
  compact control decision exists on validation;
- **F3-R2-v2/corpus:** the corpus cannot support the bank;
- **F3-R2-v2/input:** labels are readable from raw input;
- **F2-R2-v2/representation:** GPT-2 does not carry the labels strongly enough;
- **F2-R2-v2/control:** no compact decision reaches the control-sufficiency bar.

Admission metrics:

- candidate labels before gates: `>= 32`;
- survivors after balance plus input gates: `>= 24`;
- per-survivor pretrained GPT-2 linear probe validation accuracy: median
  `>= 0.65`, with at least 24 survivors above `0.60`;
- at least one validation decision reaches `z1_acc >= 0.70` with a compact
  `k_control`;
- random-init floor is measured on the same labels as a sanity read, but the
  final `body_carry - floor >= 0.15` threshold belongs to the verdict harness.

If admission passes, freeze a separate final-v2 prereg before running the full
fingerprint. Do not let the admission data become the final test split.

## 6. Final R2 v2 verdict, if admitted

The final R2 v2 harness should inherit the original gate:

- GPT-2 small, pretrained, CPU inference;
- random-init same-architecture floor;
- information-basis `d_dec`, target `d_dec >= 20`;
- compact decision shadow, `z1_acc >= 0.70`;
- cross-latent leak near chance with H4 liveness and H5 compute-can't-cross;
- `body_carry_pretrained - body_carry_floor >= 0.15`;
- external mech-interp review before any R2 promotion.

The final run must use a frozen held-out split not touched by admission tuning.

## 7. Explicit non-claims

R2 v2, even if internally sharp, would not claim:

- a world model;
- a theory of intelligence;
- that GPT-2 understands the latent;
- that all LLM residual streams resist narrow shadows;
- that synthetic or model-labelled data is equivalent to real text.

It would claim only the R2 form after external review: a real pretrained LLM's
residual stream exhibits the body/shadow resistance fingerprint on a
non-synthetic task under the registered de-confounds.

## 8. Admission result and next fork

The recommended first move was the **agreement/controller** family because it is
both next-token-relevant for GPT-2 and naturally relational. That admission run
is now complete:

- receipt: `R2_V2_ADMISSION_RECEIPT.md`;
- verdict: **`F3-R2-v2/input`**;
- core read: agreement is GPT-2-computed (`num` 0.963) but input-decodable
  (raw-token probe 0.700), so it misses the required intersection.

This leaves no active final R2 v2 harness. Any next move is a new fork, not a
continuation of the filed agreement admission.

Available forks:

1. **Attractor-controlled micro-R2:** a narrow, single-axis read; not the
   `d_dec >= 20` gate.
2. **Larger model plus semantic/relational tasks:** likely GPU and
   annotation-assisted; potentially opens the intersection on a stronger
   substrate. Scope-and-hold campaign:
   `R2_LARGER_MODEL_ROUTE_CAMPAIGN.md`.
3. **Bank R1 as the result:** record R2 as hard on GPT-2 small and stop here
   until a materially different substrate/task is commissioned.

The promotion ledger stays unchanged until a final R2 v2 run clears the gate and
external review signs off.
