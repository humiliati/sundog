# Chat-v2 H1 / V3 - State-Bank Delivery Scope

> 2026-07-01, scope-and-hold delivery plan. This is not an R2 run and
> not a promotion packet. It scopes the next constructive route after
> the count-parity, agreement, and relation-extraction admissions failed
> the R2 intersection. H2 showed the intersection exists at low
> dimension; H5 upgraded the data gate. H1 asks whether the high-dimensional
> state-bank version is wide enough to clear `d_dec >= 20`.
>
> **Post V3-0 status:** the initial whole-distribution data admission filed
> `F3-V3/copy` for code and `F3-V3/input` for chess in
> `H1_V3_0_DATA_ADMISSION_RECEIPT.md`. The active amendment is
> `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md`: chess becomes the primary ambiguity-slice
> arm; code is not revived unless an expanded corpus clears its hard floor.

## 0. Original V3-0 fork calls

1. **Primary corpus:** code variable-state.
2. **Parallel secondary:** chess PGN board-state, cheap admission only.
3. **Parked:** narrative entity-state, until a trusted non-synthetic annotation
   path exists.
4. **Include V3-0.5:** GPT-2-small calibration, explicitly not gate-bearing.
5. **V3-1 route:** run local CPU/GTX-1080 existence smokes first; use an H200
   only after the data gate clears and the local route shows a live carry signal.

The H200 is for scale and the scaling fork, not for discovering whether H1 has
a valid data bank.

After V3-0, `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md` amends this ordering: chess is
the primary V3-0b arm, and code is optional behind a hard corpus-expansion floor.

## 1. Delivery gate, with the H5 upgrade

H1 can leave scoping only if it produces a state bank with all of:

- **bank headroom:** at least 24 balanced binary axes, so `d_dec >= 20` is
  possible without shopping;
- **absolute surface-undecodability:** not just linear raw-probe failure. H5
  exposed the count-parity loophole: parity defeats a linear probe while still
  being a deterministic function of the token bag. H1 must test the stronger
  claim that the label is not a function of the registered surface statistic;
- **model carry:** a capable pretrained model carries the bank above a
  same-architecture random-init floor;
- **full fingerprint:** compact control shadow, H4 liveness, H5
  compute-can't-cross, near-chance cross-latent leak, `body_carry - floor`;
- **external review:** no R2 promotion without mech-interp review.

The first two bullets are pure data properties and are CPU-checkable now. They
are V3-0. The model question starts only after they pass.

## 2. Surface criterion

The H5 result separates two notions that the earlier R2 attempts blurred:

- **capacity-relative undecodability:** a weak probe fails, but a stronger
  readout of the same surface statistic may succeed;
- **absolute registered-surface undecodability:** two contexts have the same
  registered surface statistic and different state, so no deterministic readout
  of that statistic can recover the label.

H1 operationalizes the stronger criterion in two ways.

### A. MLP-on-counts surface suite

For each candidate axis, run surface probes over the exact representation the
model sees and over canonicalized variants:

- linear counts;
- tf-idf n-grams;
- MLP on unigram counts;
- MLP on n-gram counts, default `w in {1,2,3}`;
- metadata probes where the domain has structured metadata.

Default survival ceiling: every surface probe must stay `<= 0.60` held-out
accuracy. The MLP probes close the count-parity loophole.

### B. Constructed witness pairs

For each surviving axis, store witness pairs:

- same canonical token bag;
- different ground-truth state for the axis;
- interpreter/replay verified;
- not used as model training/probe examples unless explicitly declared.

These pairs are the empirical `SurfaceBag` certificate: they prove the axis is
not determined by the bag at all. They do not by themselves prove resistance to
all `w > 1` n-gram statistics; that is why the MLP-on-counts suite remains in
the gate.

An axis without at least one witness pair is not H1-admissible, even if the
surface probes happen to fail.

## 3. Primary corpus: code variable-state

**State:** per-variable predicates after straight-line integer arithmetic mined
from real Python.

**Why primary:** the ground truth is mechanical, the state is order-dependent,
and witness pairs are natural: `x = 1; y = 2` versus `y = 2; x = 1` share a
bag but can differ for a variable-slot predicate at the query site. The method
extends H2's hard-slice logic from stack-top to a high-dimensional state bank.

### 3.1 Data source

Use real Python files only:

- local repos first, so V3-0 can run without network;
- HuggingFace code corpora second, if accessible;
- no generated synthetic templates as verdict-bearing data.

Generated witness pairs are allowed as certificates for surface ambiguity, not
as the corpus that supplies the model-carry examples.

### 3.2 Extraction

Mine straight-line blocks with a restricted AST interpreter:

- allowed: integer constants, variable reads, assignments, augmented
  assignments, unary/binary integer arithmetic, comparisons for filtering;
- forbidden: imports, calls, attributes, subscripts with unknown objects, file or
  network access, loops, comprehensions, exceptions, classes, functions with
  side effects;
- execution happens in a sandboxed subprocess or a whitelist interpreter, with a
  short timeout and bounded integer magnitude.

Canonicalize variables by first-use slot: `v0`, `v1`, ..., `v7`. The model sees
the canonicalized prefix and a fixed query marker. The label bank is defined at
that marker.

### 3.3 Candidate axes

Use up to eight live variable slots and 3-4 predicates per slot:

- `defined`: slot is alive/defined at the marker;
- `parity`: value is odd/even;
- `sign`: negative vs nonnegative, or positive vs nonpositive if balance is
  better;
- `zero`: zero vs nonzero;
- `magnitude`: small/large bucket, threshold chosen on validation only.

The target bank is 24-32 axes. The final admitted bank must have at least 24
balanced axes after all filters.

### 3.4 Computed-value filter

Literal copies do not count. For an axis to be H1-admissible, the variable's
current value must depend on at least one nontrivial update:

- expression contains an operator, not just a literal assignment;
- or expression reads a previous variable whose provenance is nontrivial;
- or the value changes across at least one prior update.

Record a provenance bit per variable so copy/literal sites can be excluded from
the model-carry bank. This avoids mistaking surface-visible literals for state
tracking.

### 3.5 Code-specific witness pairs

For every admitted axis, construct at least one pair and preferably a small
panel of pairs:

- same canonical token bag;
- different order of assignments/updates;
- same query marker;
- different predicate truth for the slot;
- both accepted by the restricted interpreter.

Example schema, illustrative only:

```python
v0 = 1
v1 = 2
v0 = v0 + v1
```

versus a bag-matched reordered/update variant whose final `v0` predicate
differs. The final implementation should emit the exact pair text and labels in
the V3-0 receipt.

## 4. Secondary corpus: chess PGN

**State:** board occupancy after replaying legal moves.

**Why secondary:** it offers hundreds of axes and a clean replay oracle. It also
starts with a surface leak: SAN notation includes captures, check markers, and
disambiguation characters that are state-dependent.

V3-0 should still run chess admission because it shares the harness shape:

- parse real PGN games;
- replay legal moves;
- define axes as square occupancy or piece-on-square predicates;
- run the same surface suite;
- include SAN-as-written, sanitized-SAN, and UCI-like controls as separate
  conditions;
- require witness pairs, preferably transposition-style pairs where the same
  move-token bag or piece-move inventory leads to different board state.

Chess can promote to primary only if at least 24 axes survive the upgraded
surface gate. Expect the SAN leak to kill many axes; that is the point of
running admission before model carry.

## 5. Parked corpus: narrative entity-state

Narrative entity-state is closest to the intuitive "world state" language, but
it is not the right first corpus:

- real ground truth is expensive;
- annotator LLMs create a new trust surface;
- synthetic stories would weaken the R2 real-task claim.

Reopen only after code and chess fail, and only with a frozen human or
mechanical annotation path.

## 6. Phased plan

| rung | where | task | gate |
| --- | --- | --- | --- |
| **V3-0 data admission** | CPU now | mine corpus, build bank, run balance plus H5-upgraded surface suite | `>=24` surviving axes, else `F3-V3/input` |
| **V3-0b ambiguity-slice admission** | CPU now | define axes on count-ambiguous slices; bounded witness search; slice-conditioned probes | `>=24` surviving axes, else `F3-V3b/*` |
| **V3-0.5 calibration** | CPU, cheap | H2-style hard-slice probe on GPT-2-small | non-gate expectation anchor |
| **V3-1 1B admission** | CPU-lite / local 1080 / GPU | Qwen2.5-1.5B or OLMo-2-1B, random-init floor, decision selection | carry above floor at `d_dec >= 20`, else `F2-V3/carry` |
| **V3-2 full fingerprint** | GPU | complete R2 battery plus external review packet | only rung that can move `PROMOTE_GATE.md` R2 |

## 7. V3-0 data admission

Deliverable:

`docs/chatv2/H1_V3_0_DATA_ADMISSION_RECEIPT.md`

Script target:

`scripts/chatv2_h1_v3_data_admission.py`

Admission result branches:

| branch | meaning |
| --- | --- |
| `H1-V3-DATA-ADMIT` | at least 24 axes survive balance, MLP/count surface probes, and witness-pair checks |
| `F3-V3/input` | surface suite reads too many axes or witness pairs fail |
| `F3-V3/bank` | fewer than 24 balanced axes exist |
| `F3-V3/corpus` | not enough executable/replayable real corpus |
| `F3-V3/copy` | surviving axes are mostly literal/copy sites, not computed state |

V3-0 is the make-or-break rung. No GPU work before this receipt.

## 8. V3-0.5 GPT-2 calibration

Include this rung.

Purpose:

- anchor expectations against H2;
- verify the extraction site and probe code before larger models;
- estimate whether state carry exists at all on the weakest general LM.

Non-goals:

- no R2 claim;
- no promotion;
- no requirement that GPT-2-small pass;
- no final decision selection.

Deliverable:

`docs/chatv2/H1_V3_0_5_GPT2_CALIBRATION_RECEIPT.md`

Script target:

`scripts/chatv2_h1_v3_gpt2_calibration.py`

If GPT-2 is weak, that is expected. If GPT-2 is strong on a hard slice, V3-1
gets more attractive.

## 9. V3-1 1B admission

Run only after V3-0 admits a bank.

Primary models:

- `Qwen/Qwen2.5-1.5B`;
- OLMo-2 1B base, if the local stack and access path are clean.

Route:

1. local CPU-lite existence run over about 500 short windows, if feasible;
2. local GTX-1080 CUDA smoke for throughput and memory;
3. owner-gated H200 only if the first two show a live carry signal and the bank
   remains worth scaling.

Reuse/piggyback:

- local key handling and no-secret discipline from the Dev keyring notes;
- model-loading, hidden-state harvest, and manifest discipline from
  `sundogcert/AGENTIC_TRACE_H1_OPENWEIGHT_GENERALITY.md`;
- keep claims separate: this is chat-v2 R2/H1 state-bank work, not an agentic
  trace claim.

Admission thresholds:

- at least 24 admitted axes from V3-0;
- validation `d_dec >= 20`;
- pretrained median carry at least `0.10` above same-architecture random-init
  floor;
- preliminary `body_carry - floor` trend plausibly reaches the final `0.15`
  threshold;
- one pre-registered decision selector reaches `z1_acc >= 0.70`.

Branches:

| branch | meaning |
| --- | --- |
| `H1-V3-1-ADMIT` | 1B-scale model carries the bank enough to justify final prereg |
| `F2-V3/carry` | model does not carry the axes |
| `F2-V3/control` | no compact control shadow |
| `F4-V3/floor` | random-init floor explains the carry |
| `F6-V3/compute` | local/H200 route cannot run the registered measurement cleanly |

## 10. V3-2 full fingerprint

V3-2 can be frozen only after V3-1 admits.

It must include:

- one frozen model;
- one frozen corpus and bank;
- frozen train/validation/test split;
- frozen representation site and layer-selection rule;
- final `d_dec >= 20`;
- compact `z1_acc >= 0.70`;
- near-chance cross-latent leak with permutation control;
- H4 liveness;
- H5 compute-can't-cross;
- same-architecture random-init floor;
- `body_carry_pretrained - body_carry_floor >= 0.15`;
- external mech-interp review packet.

Only V3-2 plus external review can alter the R2 row in `PROMOTE_GATE.md`.

## 11. Immediate build order

1. V3-0 is complete: see `H1_V3_0_DATA_ADMISSION_RECEIPT.md`.
2. Run V3-0b per `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md`, chess first.
3. File `H1_V3_0B_SLICE_ADMISSION_RECEIPT.md`.
4. If and only if V3-0b admits at least one bank, run V3-0.5 GPT-2 calibration
   on the admitted slices.
5. Stage V3-1 command lines with local CPU/1080 first, H200 second.

This is not the same trap repeated. The failed R2 families were functions of
surface counts or local triggers. H1 asks for high-dimensional **state**, and
H5 now requires evidence that the state is not determined by the registered
surface statistic at all.
