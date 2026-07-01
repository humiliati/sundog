# Chat-v2 R2 - Larger-Model Route Campaign

> 2026-07-01, scope-and-hold campaign draft. This is not an R2 run, not a
> promotion packet, and not a commitment to spend GPU time. It scopes the only
> remaining plausible R2 route after the two GPT-2-small admissions split the
> problem cleanly: count-parity was input-undecodable but not GPT-2-computed;
> agreement was GPT-2-computed but input-decodable.

## 0. Campaign question

Can a stronger open-weight pretrained LLM carry a high-dimensional bank of
human-text semantic/relational latents that are:

1. not linearly readable from raw input under strong surface controls;
2. linearly carried in the model residual stream;
3. numerous and independent enough to clear `d_dec >= 20`;
4. above a same-architecture random-init floor;
5. usable by a compact control shadow while the shadow does not reconstruct the
   rest?

If yes, freeze a final R2 prereg and run the body-resistance fingerprint. If no,
bank R1 and record the R2 boundary as "hard on small/base pretrained LMs under
the registered de-confounds."

## 1. Why this is a new campaign

The GPT-2-small work did not fail because the harness was weak. It failed
because the intersection was empty:

| family | input-undecodable? | pretrained-computed? | failure |
| --- | --- | --- | --- |
| count-parity MVP | yes | no | `F3-R2` bank thin; likely `F2-R2` if richer |
| agreement admission | no | yes | `F3-R2-v2/input` |

The larger-model route is therefore not "try GPT-2 again harder." It changes two
load-bearing variables at once:

- **substrate:** stronger base models with richer semantic/relational state;
- **task family:** annotated relation/coreference/semantic tasks with many
  independent labels, not regex parities or one-axis agreement.

Because this is materially larger and could require paid GPU time or annotation,
it must have an admission campaign before any verdict run.

## 2. Substrate ladder

Use **base** models as primary substrates. Instruction-tuned models may be
robustness checks, but they are not the first verdict substrate because chat
tuning can add task-format artifacts.

### Track A - scientific scaling ladder

**Pythia** is the cleanest scaling-science ladder: sizes from 70M through 12B,
including 1B, 1.4B, 2.8B, 6.9B, and 12B, trained on the same data in the same
order. This is ideal for "does the intersection open with scale?" even if the
models are weaker than newer capability models.

Recommended use:

- admission smoke: `EleutherAI/pythia-1b` or `pythia-1.4b`;
- scaling probe: `pythia-2.8b`;
- serious but still bounded check: `pythia-6.9b`;
- `pythia-12b` only if the 6.9B step is near-positive.

### Track B - capability ladder

Use a stronger current base-model ladder when the question is "does a more
capable model carry relation-state that GPT-2-small did not?"

Good candidates:

- **Qwen2.5 base:** 1.5B, 3B, 7B as a compact capability ladder;
- **OLMo 2 base:** 1B and 7B as a more open-science ladder with released
  training artifacts;
- **Llama 3.2 base:** 1B and 3B as an edge-size sanity ladder, if access terms
  and local auth are acceptable.

Campaign recommendation: start with **Qwen2.5-1.5B** or **OLMo-2-1B** for an
admission build, then choose one 7B-class model only if the 1B/3B step shows a
real intersection signal. Do not begin at 7B.

## 3. Task-family ladder

The primary need is not just "harder NLP." It is a bank of many independent
binary axes whose labels require semantic or relational state but are not
recoverable from raw surface features.

### Primary family - relation extraction bank

Use human-annotated relation extraction as the first serious family. It is the
only obvious path with enough axes for `d_dec >= 20`.

Candidate datasets:

- **TACRED:** newswire/web text, about 106k examples and 41 relation types;
- **FewRel:** Wikipedia-derived sentences with 100 relation types and tens of
  thousands of annotated examples;
- **DocRED:** document-level relation extraction over Wikipedia/Wikidata, with
  entity and relation annotations and multi-sentence reasoning pressure.

Encoding:

- mark the subject/head and object/tail entities in the text;
- read the residual stream at a fixed query site after a neutral relation prompt,
  or at a fixed entity-pair marker site if the model/tokenizer makes that cleaner;
- create one binary latent per relation type, balanced against hard negatives;
- use train/validation/test splits that hold out entity pairs and, when possible,
  trigger lemmas/templates.

Why it could work: relation labels are closer to the semantic/discourse state a
larger pretrained model may carry, and relation datasets can yield 24+ axes.

Main risk: relation labels often leak through trigger words. The raw-input gates
must be strong enough to kill that.

### Secondary family - coreference / role resolution

Use WinoGrande/WSC-like or OntoNotes-style coreference only as a supporting
family. These tasks are plausibly model-computed and harder than agreement, but
they are low-axis unless heavily expanded.

Allowed use:

- capability sanity check;
- one or two auxiliary axes in a mixed bank;
- not the primary `d_dec >= 20` source unless a pre-registered bank with 24+
  independent labels is built.

### Smoke-only families

BLiMP/SyntaxGym-style minimal pairs, agreement attractors, and generated
controlled templates are useful for harness validation. They are not
verdict-bearing R2 unless the promotion gate is explicitly amended, because they
do not satisfy the "real task / non-synthetic" spirit.

## 4. Surface-deconfound gate

The agreement admission proved that a weak raw probe is not enough. For this
campaign, a label survives input-undecodability only if **all** surface probes
stay at or below the registered ceiling.

Default ceiling: `<= 0.60` held-out accuracy, unless the final prereg tightens
it.

Required probes:

1. **bag/count raw probe:** token counts over the prompt/input;
2. **tf-idf n-gram probe:** unigram/bigram/trigram surface features;
3. **delexicalized raw probe:** entity names replaced with typed markers;
4. **position/entity metadata probe:** entity types, distance, sentence index,
   order, mention counts, and marker positions;
5. **masked-trigger probe:** optional but recommended for TACRED/FewRel, with
   high-PMI trigger spans masked before raw probing.

Required splits:

- hold out entity pairs across train/test;
- hold out subject/object names;
- hold out source documents where document IDs exist;
- hold out high-PMI trigger lemmas when the dataset supports it.

If fewer than 24 relation axes survive this gate, file
**F3-R2-LM/input** and stop. Do not weaken the raw gate.

## 5. Model-carry admission

Only after the surface gate passes do model activations matter.

Admission metrics on validation:

- at least 32 candidate binary axes before gates;
- at least 24 axes survive balance plus surface deconfounds;
- at least 24 surviving axes have pretrained residual linear-probe accuracy
  `>= 0.65`;
- median pretrained residual accuracy across survivors `>= 0.68`;
- random-init same-architecture floor is at least `0.10` lower than pretrained
  on median survivor carry;
- effective readout rank on survivors has `d_dec >= 20` on validation.

If the surface gate passes but model carry fails, file
**F2-R2-LM/carry**. If carry passes but the random-init floor is too high, file
**F4-R2-LM/floor**.

## 6. Control-shadow admission

The final R2 question still needs a compact control-sufficient shadow. Before a
final verdict run, pick exactly one decision family on validation:

- a single relation decision;
- a small relation group decision;
- or a pre-registered downstream selector over relation labels.

Admission threshold:

- validation `z1_acc >= 0.70`;
- compact `k_control`, chosen by a saturation sweep;
- H4 liveness works: the decision itself leaks from the shadow;
- a preliminary cross-latent leak check is not already high.

If the relation bank is high-dimensional but every compact decision fails,
file **F2-R2-LM/control**.

## 7. Campaign phases

### Phase LM-0 - data audit, local CPU

Deliverable: `R2_LM_DATA_AUDIT_RECEIPT.md`.

Tasks:

- load TACRED, FewRel, and/or DocRED through HuggingFace or a local operator
  file;
- freeze license/access notes;
- build entity delexicalization and relation-axis balancing;
- run all raw probes;
- report survivor count and leakage table.

Go condition: at least one dataset yields `>=24` surface-surviving axes.

### Phase LM-1 - 1B admission, GPU-light

Deliverable: `R2_LM_1B_ADMISSION_RECEIPT.md`.

Tasks:

- run one base model in the 1B-1.5B range;
- cache residuals at a small layer set first, then all layers only if promising;
- run pretrained vs random-init carry;
- estimate `d_dec`;
- run preliminary control-shadow admission.

Go condition: surface survivors remain `>=24`, carry median `>=0.68`, floor gap
`>=0.10`, and validation `d_dec >= 20`.

### Phase LM-2 - scaling fork

Deliverable: `R2_LM_SCALING_FORK.md`.

Only run if LM-1 is near-positive or positive. Choose one of:

- **Pythia scaling:** 1.4B -> 2.8B -> 6.9B;
- **Qwen capability:** 1.5B -> 3B -> 7B;
- **OLMo openness:** 1B -> 7B.

The fork must pre-register the final candidate before seeing the final held-out
test. Scaling is for admission and model choice, not for shopping final results.

### Phase LM-3 - final R2 prereg

Deliverable: `R2_LM_FINAL_PREREG.md`.

Freeze:

- one model;
- one dataset/family;
- one representation site;
- one split;
- one surface-deconfound suite;
- one decision selector;
- one final test set.

No final R2 language is licensed until this exists.

### Phase LM-4 - final fingerprint

Deliverable: `R2_LM_FINAL_RECEIPT.md`.

Run the original R2 fingerprint:

- information-basis `d_dec >= 20`;
- compact `z1_acc >= 0.70`;
- cross-latent leak near chance with permutation control;
- H4 liveness and H5 compute-can't-cross;
- `body_carry_pretrained - body_carry_floor >= 0.15`;
- same-architecture random-init floor;
- external mech-interp review packet before promotion.

## 8. Compute plan

Local CPU is for data audit, raw probes, and small harness smokes only.

GPU expectations:

- 1B-1.5B admission: single rented L4/A10-class GPU should be enough;
- 3B scaling: likely 16-24GB GPU depending on batch size and activation cache;
- 7B final candidate: plan for 24GB minimum, preferably A100/L40S-class if
  caching all layers or running many examples;
- 12B Pythia: only after a near-positive 6.9B result.

Final runs should use fp16/bf16 activations. Quantized inference may be used for
smokes, but not for the final receipt unless a quantization-invariance control
is pre-registered and passed.

Activation caching rule:

- cache only the representation site and selected layers first;
- expand to all layers only for admitted candidates;
- write manifest files with model hash, dataset revision, split seed, layer set,
  dtype, and prompt template.

## 9. Named failure modes

| code | branch | meaning |
| --- | --- | --- |
| `F3-R2-LM/data` | stop | corpus/dataset unavailable, too small, or not license-clean |
| `F3-R2-LM/input` | stop | raw/delexicalized/surface probes read the labels |
| `F3-R2-LM/bank` | stop | fewer than 24 independent axes survive |
| `F2-R2-LM/carry` | stop | stronger model still does not carry the labels |
| `F2-R2-LM/control` | stop | no compact decision shadow reaches `z1_acc >= 0.70` |
| `F4-R2-LM/floor` | stop | random-init floor is too high; carry is architectural |
| `F1-R2-LM/net7` | marginal | shadow reconstructs the rest; no resistance |
| `F5-R2-LM/prompt` | stop | prompt/entity markers create an artifact |
| `F6-R2-LM/quant` | stop | quantization changes the fingerprint materially |

## 10. Decision recommendation

The least-wasteful opening move is:

1. run **LM-0** on TACRED + FewRel first, DocRED second;
2. prefer **FewRel** if it gives enough surface-surviving axes, because 100
   relations is the cleanest high-dimensional bank;
3. use **Qwen2.5-1.5B** or **OLMo-2-1B** for LM-1;
4. escalate to **Qwen2.5-7B** or **OLMo-2-7B** only after LM-1 shows the
   intersection is nonempty.

This campaign is worth commissioning only if we want an R2 attempt badly enough
to pay the GPU and labeling/audit cost. Otherwise the honest state is already
strong: R1 is met, and GPT-2-small admissions explain why R2 is hard.

## 11. Source notes checked 2026-07-01

- [Qwen2.5 collection](https://huggingface.co/collections/Qwen/qwen25) - base
  and instruct model sizes including 0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B.
- [Pythia scaling suite](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite)
  and [pythia-6.9b model card](https://huggingface.co/EleutherAI/pythia-6.9b)
  - research scaling suite, same-data/same-order model ladder.
- [OLMo 2 1B](https://huggingface.co/allenai/OLMo-2-0425-1B) and
  [OLMo 2 7B](https://huggingface.co/allenai/OLMo-2-1124-7B) - open-science
  base-model candidates.
- [Llama 3.2 1B model card](https://huggingface.co/meta-llama/Llama-3.2-1B)
  - 1B/3B edge-size base/instruct family, access terms to be checked before use.
- [TACRED dataset card](https://huggingface.co/datasets/DFKI-SLT/tacred) -
  106k examples and 41 relation types over newswire/web text.
- [FewRel dataset card](https://huggingface.co/datasets/thunlp/few_rel) and
  [FewRel paper page](https://huggingface.co/papers/1810.10147) - 100 relation
  types and large annotated relation-classification bank.
- [DocRED dataset card](https://huggingface.co/datasets/thunlp/docred) -
  human-annotated document-level relation extraction.
- [WinoGrande dataset card](https://huggingface.co/datasets/allenai/winogrande)
  - 44k Winograd-style commonsense problems, useful as a secondary sanity task.

