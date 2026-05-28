# Phase 3A -- Stochastic Per-Task Learner (Branch A)

Parent spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)

Reflection gate: [`PHASE3_5_REFLECTION.md`](PHASE3_5_REFLECTION.md)

Filed: **2026-05-28 (PT)**

Status: **BINDING RECEIPT FILED -- `branch_a_full_grid_floor`**. This file
started Branch A by pre-registering the first stochastic per-task learner
family. The runner/wrapper/freeze-marker amendments and binding receipt are
filed in the parent spec. The raw-grid arena did not open, so no
signature-vs-full-grid sufficiency comparison is licensed from this lane.

## Purpose

V1, V2, and compact-7 all failed before a meaningful signature-vs-full-grid
comparison could begin. The failure was not "signature loses to full grid";
the full-grid controls themselves floored. Compact-7 additionally showed
dominant-color mode collapse: the learner matched output shape and background
mass while omitting the object palette/content.

Branch A changes the learner family rather than the task distribution. The
question is:

> Can a stochastic model trained from scratch per task, using only that task's
> registered demonstrations, open the full-grid control arena and then compare
> `signature_palette` against the matched full-grid arm?

## Scope And Non-Goals

This lane remains inside the registered Phase 0 task class. It does not inspect
ARC evaluation tasks, Kaggle notebooks, private test outputs, or any
non-registered public-training tasks for fitting.

This lane does not use public-training auxiliary meta-training. Every fitted
model is initialized from the frozen seed rule and trained only on the
conditioning demonstrations for the specific held-out instance being predicted.
There is no cross-task learned weight sharing.

This lane does not reopen Branch B. V2 and compact-7 already closed the
narrowed deterministic full-grid-control path. Any quarantine labels in this
spec are diagnostic labels for Branch A receipts, not a new Branch B claim.

## Registered Learner

Learner version: `per_task_coord_mlp_v1`.

The learner is a per-instance, per-arm coordinate-conditioned MLP trained from
scratch. For each LODO or public-training-test instance:

1. Build the conditioning set exactly as Pass B defines it.
2. Initialize fresh shape and color models for the selected arm and seed.
3. Train only on cell examples derived from the conditioning pairs.
4. Predict the query output shape and colors.
5. Discard the fitted weights after scoring the instance.

The fitted weights for one task, lane, arm, or seed must never initialize or
regularize any other task.

## Representation Arms

The runner must reuse the frozen `arc-p3-feature-v1` encoders from the
Blackwell runner unless a later append-only bugfix explicitly changes the
feature schema.

| arm | role | grid exact eligible? |
| --- | --- | --- |
| `raw_grid_per_task` | matched full-grid control using the raw-grid one-hot encoding | yes |
| `signature_palette_per_task` | primary Sundog arm using shape/palette/density plus signature hashes | yes |
| `signature_only_per_task` | strict signature ablation using no palette metadata | quotient metrics only unless a later palette-lift decoder is filed |
| `metadata_only_per_task` | coarse nuisance control | no support claim; discrimination only |

`raw_grid_per_task` and `signature_palette_per_task` are the only arms that can
open or close exact-grid Branch A support in this first stochastic lane.

## Instance Construction

Splits are inherited unchanged from Phase 3 Pass B:

- training tasks remain training tasks;
- validation tasks are used only for seed/model selection;
- test tasks provide the held-out LODO and public-training-test lanes.

For LODO, each training pair of a registered task is held out once and the
learner is trained on the remaining pairs. For public-training test, the
learner is trained on all registered training pairs for that task and predicts
the public-training test output. The runner must preserve the existing
two-runner discipline or an equivalent manifest barrier: public-training test
outputs are not read until the LODO manifest has been written and flushed.

## Feature Rows

For every conditioning pair `(input_grid, output_grid)` and every output cell
coordinate `(oy, ox)`, the color learner receives one supervised row:

- the arm-specific input vector for `input_grid`;
- normalized output coordinates:
  `oy/(output_h - 1)`, `ox/(output_w - 1)`, with denominator `1` for singleton
  axes;
- centered coordinates in `[-1, 1]`;
- boundary flags for top, bottom, left, and right;
- normalized input and output shape features:
  `input_h/30`, `input_w/30`, `output_h/30`, `output_w/30`;
- for `raw_grid_per_task` only, a 3x3 nearest-neighbor source patch sampled
  from the input grid at the output coordinate's normalized position, encoded
  as 11-way one-hot labels (`0..9` plus padding);
- for all non-raw arms, no cell-local raw color feature is allowed.

The label is the output color at `(oy, ox)`.

The shape learner receives one supervised row per conditioning pair:

- the arm-specific input vector for `input_grid`;
- normalized input shape features;
- labels `output_h - 1` and `output_w - 1`.

## Frozen Model Spec

Both shape and color models are MLPs with deterministic initialization from the
derived instance seed.

Shape model:

- architecture: `Linear(input_dim, 128) -> GELU -> LayerNorm(128) ->
  Linear(128, 128) -> GELU -> Linear(128, 30)` for height and a separate
  same-shaped head for width;
- loss: `height_ce + width_ce`;
- optimizer: AdamW;
- learning rate: `1e-3`;
- betas: `(0.9, 0.99)`;
- epsilon: `1e-8`;
- weight decay: `1e-4`;
- max steps: `600`;
- early-stop patience: `80` steps on conditioning loss;
- gradient clip norm: `1.0`.

Color model:

- architecture: `Linear(input_dim, 192) -> GELU -> LayerNorm(192) ->
  Linear(192, 192) -> GELU -> Dropout(0.05) -> Linear(192, 10)`;
- loss: class-balanced cross entropy;
- class weight for observed color `c`:
  `clamp(sqrt(max_observed_count / count_c), 0.25, 5.0)`;
- unobserved colors receive weight `1.0` but have no supervised rows unless
  they occur in the conditioning outputs;
- optimizer: AdamW;
- learning rate: `1e-3`;
- betas: `(0.9, 0.99)`;
- epsilon: `1e-8`;
- weight decay: `1e-4`;
- batch size: `512` cell rows, with full-batch training when fewer rows exist;
- max steps: `900`;
- early-stop patience: `120` steps on conditioning loss;
- gradient clip norm: `1.0`.

Initialization:

- linear weights: Xavier uniform;
- linear biases: zero;
- LayerNorm weights: one;
- LayerNorm biases: zero.

No pretrained weights, public auxiliary weights, or cross-task checkpoints are
admitted.

## Seed Slate And Derived Seeds

Master seed slate:

- `20260528`
- `20260529`
- `20260530`
- `20260531`
- `20260601`

Each model seed is derived by SHA-256 over:

`arc-p3a-per-task-coord-mlp-v1\0<master_seed>\0<lane>\0<task_id>\0<query_index>\0<arm>\0<model_kind>`

where `model_kind` is `shape` or `color`. Interpret the first eight digest
bytes as an unsigned big-endian integer and reduce modulo `2^31 - 1`.

Seed selection is performed independently per arm using validation lanes only:

1. highest validation `grid_exact_any_task_count`, for grid-eligible arms;
2. highest validation `minority_color_recall`;
3. lowest validation `dominant_color_collapse_rate`;
4. lowest validation loss;
5. lowest numeric master seed.

For non-grid arms, step 1 uses `rep_exact_slot1_task_count`.

## Candidate Slots

The runner emits two slots for grid-eligible arms:

- slot 1: top-1 predicted shape and per-cell color argmax;
- slot 2: second-best predicted shape when shape top-2 differs; otherwise
  deterministic color sampling from the slot-1 shape with temperature `0.75`
  and the derived color seed.

Sampling must use the frozen derived seed and must be reproducible byte for
byte on repeated runs on the same platform. Slot 2 is diagnostic; Branch A
support must be visible in slot 1 or any-slot exact under the frozen metrics.

`signature_only_per_task` emits quotient representations and is scored only on
representation/quotient metrics in this first Branch A lane.

## Metrics

The runner must retain all Blackwell receipt metrics and add the following:

- `grid_exact_task_count`
- `minority_color_recall`: exact color recall on target cells whose color is
  not the modal target color for that instance;
- `palette_jaccard_slot1`;
- `predicted_color_count_slot1`;
- `target_color_count`;
- `dominant_color_collapse`: true when the target uses at least 3 colors and
  the prediction uses at most 2 colors;
- `conditioning_train_exact_rate`;
- `seed_instability`: true when the selected arm's exact decision changes
  across seeds for the same instance.

All per-prior tables must report the same metrics by Phase 0
`primary_prior`.

## Arena Gate

Before any signature sufficiency language is allowed, the matched full-grid arm
must open the arena.

`raw_grid_per_task` opens the arena only if the selected-seed receipt satisfies
both:

- at least one exact task on `test_lodo`;
- at least one exact task on `pttest`.

If the raw-grid arm does not open the arena, the receipt verdict is
`branch_a_full_grid_floor`. In that case:

- no `signature_palette` comparison may be interpreted as sufficiency support
  or failure;
- no extra seeds may be run after seeing the failure;
- the next admissible Phase 3 work must be a new append-only learner spec or
  Branch D.

## Branch A Adjudication

If the raw-grid arena gate opens, `signature_palette_per_task` is compared to
`raw_grid_per_task` on held-out lanes.

`branch_a_support` requires:

- `signature_palette_per_task` also opens the arena;
- on both `test_lodo` and `pttest`, its exact task count trails the raw-grid
  arm by no more than one task;
- its `minority_color_recall` trails the raw-grid arm by no more than `0.10`
  on each held-out lane;
- dominant-color collapse rate is not higher than raw grid by more than
  `0.10` on either held-out lane.

`branch_a_bounded_failure` applies when raw grid opens the arena but
`signature_palette_per_task` does not meet the support criteria on the easiest
registered held-out slice (`compact_signal` if the runner reports that slice,
otherwise the all-task held-out lanes).

`branch_a_named_quarantine` may be reported only as a diagnostic sub-verdict
when at least 80% of the exact-task gap is assigned to pre-registered
categories in the quarantine log and the non-quarantined slice satisfies
`branch_a_support`. It does not reopen Phase 3.5 Branch B.

## Quarantine Labels

The quarantine log is pre-registered to use only these labels:

- `insufficient_conditioning_pairs`: fewer than 3 conditioning pairs after LODO;
- `shape_prediction_failure`: output shape wrong before color scoring;
- `dominant_color_mode_collapse`: prediction uses at most two colors while the
  target uses at least three;
- `minority_object_recall_failure`: minority-color recall below `0.25`;
- `palette_lift_failure`: signature-only quotient output matches but exact
  color palette cannot be recovered;
- `color_permutation_quotient`: strict signature representation is invariant
  to a color choice the exact-grid metric requires;
- `stochastic_instability`: seed outcomes disagree on exactness or collapse;
- `conditioning_overfit`: conditioning train exact rate at least `0.95` while
  held-out exact fails;
- `signature_collision`: identical input representation with divergent
  conditioning output representations inside the same task.

Every held-out non-exact row must receive at least one failure label.

## Required Artifacts

Binding output path:

`results/arc/phase3a-per-task-coord-mlp-v1/`

Required files:

- `manifest.json`;
- `split.csv`;
- `phase3a_receipt.json`;
- `scores.csv`;
- `per_task.csv`;
- `per_prior.csv`;
- `per_instance.csv`;
- `learning_curves.csv`;
- `seed_stability.csv`;
- `quarantine_log.csv`;
- `dominant_color_audit.csv`;
- `residuals.jsonl`;
- `branch_adjudication.md`;
- `commands.md`.

The manifest must record:

- this spec path and SHA-256 hash;
- parent spec hash;
- register hash;
- data directory hash policy;
- Python executable and version;
- Torch version;
- device;
- dirty-worktree policy;
- selected seed per arm;
- raw-grid arena-gate result;
- branch decision.

## Runner And Command Hold

The admitted implementation names are reserved but not yet executable:

- Python runner: `docs/prereg/arc/phase3a_per_task_coord_mlp.py`;
- Node wrapper: `scripts/arc-phase3a-per-task-coord-mlp-v1.mjs`;
- npm script: `arc:phase3a:per-task-coord-mlp-v1`.

The first freeze-marker commit for this lane must add the runner, wrapper,
package script, result ignore path, and a matching append-only amendment to
`PHASE3_SUFFICIENCY_SPEC.md`. Until then, this spec is a contract only.

The runner must support:

- `--dry-run`;
- `--probe-only`;
- `--limit-tasks <n>`;
- `--probe-steps <n>`;
- `--device cpu|cuda`;
- `--allow-dirty`;
- `--out <path>`.

Per the repo's ten-minute rule, the first implementation receipt must run only
a capped probe locally unless the full run is measured to complete under about
ten minutes. If the full run is projected longer, `commands.md` must stage the
exact PowerShell command, wall-clock estimate, resume safety notes, and the
decision each outcome selects.

## Public Language

Allowed before a binding receipt:

> "Phase 3A has filed a stochastic per-task learner spec. No Branch A receipt
> exists yet, and no sufficiency claim is admitted."

Allowed if the raw-grid arena floors:

> "The first stochastic per-task learner did not open the full-grid control
> arena; the result calibrates the learner family and does not adjudicate
> signature sufficiency."

Allowed if Branch A support is filed:

> "Within the registered ARC public-training subset and this per-task learner
> family, `signature_palette` was competitive with the matched full-grid control
> on held-out tasks."

Forbidden:

- any claim about ARC public-evaluation or Kaggle performance;
- any claim that `signature_only` is exact-grid sufficient without a filed
  palette-lift decoder;
- any claim that Branch A support, if obtained, solves ARC;
- any extra seed or narrower subset run after an arena-floor receipt without a
  new append-only amendment.

## Binding Receipt Addendum

Filed: **2026-05-28 (PT)**.

The Phase 3A binding receipt is filed in
[`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md) under "Branch A
20-Shard Binding Receipt: `branch_a_full_grid_floor`".

Summary:

- receipt path: `results/arc/phase3a-per-task-coord-mlp-v1/`;
- merge/shard protocol: 20 shards, 4 arms x 5 seeds;
- verdict: `branch_a_full_grid_floor`;
- raw-grid held-out exact tasks: zero on `pttest`, zero on `test_lodo`;
- consequence: no `signature_palette_per_task` vs. `raw_grid_per_task`
  sufficiency comparison is licensed;
- named failure character: **conditioning starvation + shape-generalisation
  failure**.

This addendum does not change the frozen learner contract above. It records that
the admitted lane has now run and closed Branch A in the filed
`per_task_coord_mlp_v1` learner family.
