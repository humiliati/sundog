# Phase 3D -- Different Framing: Structured Edit Residual

Parent spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)

Reflection gate: [`PHASE3_5_REFLECTION.md`](PHASE3_5_REFLECTION.md)

Filed: **2026-05-28 (PT)**

Status: **BINDING RECEIPT FILED -- `branch_d_full_grid_edit_floor`**. This
file started Branch D by pre-registering a different framing of the Phase 3
question. The runner/wrapper/freeze-marker amendments and binding receipt are
filed in the parent spec. The raw-grid edit arena did not open, so no
signature-vs-full-grid sufficiency comparison is licensed from this lane.

## Purpose

V1, V2, compact-7, and Phase 3A all floored before a meaningful
signature-vs-full-grid comparison could be made. The failure modes are now
separated:

- V1/V2: noise-dominated full-grid floor;
- compact-7: dominant-color mode collapse;
- Phase 3A: conditioning starvation + shape-generalisation failure.

Branch D changes the target object being learned. Instead of asking a learner
to emit the whole output grid directly, it asks whether the output can be
modelled as a structured edit of an input-derived baseline:

> input grid -> baseline canvas -> residual edit mask -> edited output grid

This framing is designed to test whether the signature carries enough
information for the transformation once the no-edit/background mass is factored
out explicitly.

## Question

For the registered Phase 0 task class, does `signature_palette_edit` retain
enough information to learn the same structured edit residual as a matched
`raw_grid_edit` control, when both arms use the same baseline family, splits,
seeds, and receipt gates?

## Scope And Non-Goals

This lane uses only ARC public-training tasks in the registered Phase 0 subset.
It does not inspect ARC evaluation tasks, Kaggle notebooks, private test
outputs, or non-registered public-training tasks.

This lane is not a fifth direct-output decoder. It must report direct grid exact
scores only after reconstructing:

`predicted_grid = apply_edit(predicted_baseline, predicted_edit_mask, predicted_edit_colors)`

This lane does not reopen Branch A or Branch B. Its verdict vocabulary is
separate and must not be described as support or failure for the direct-output
learner families already filed.

## Registered Framing

Framing version: `structured_edit_residual_v1`.

For each held-out instance and each representation arm:

1. Select a shape rule and baseline canvas rule using only conditioning pairs.
2. Apply the selected baseline rule to the query input.
3. Train a per-instance edit-mask learner and edit-color learner from scratch on
   residuals between conditioning baselines and conditioning outputs.
4. Predict query edits.
5. Reconstruct the final grid and score exactness, residual quality, and edit
   diagnostics.

The selected baseline and fitted weights for one instance must never initialize
or regularize another instance.

## Representation Arms

The runner must reuse `arc-p3-feature-v1` for arm-level input features unless a
later append-only bugfix explicitly changes the feature schema.

| arm | role | exact-grid eligible? |
| --- | --- | --- |
| `raw_grid_edit` | matched full-grid edit control | yes |
| `signature_palette_edit` | primary Sundog edit arm | yes |
| `signature_only_edit` | strict quotient edit ablation | quotient diagnostics only unless a palette-lift decoder is filed |
| `metadata_only_edit` | coarse nuisance edit control | no support claim; discrimination only |

Only `raw_grid_edit` can open the arena. Only `signature_palette_edit` can
support a Branch D sufficiency claim in this first framing.

## Splits And Lanes

Splits are inherited unchanged from Phase 3 Pass B and Phase 3A:

- validation tasks select seeds and tune edit thresholds;
- test tasks provide held-out LODO and public-training-test lanes;
- public-training test outputs are not read until the LODO manifest barrier has
  been written and flushed, or an equivalent two-stage manifest barrier is
  implemented.

LODO uses each training pair as query once and all remaining pairs as
conditioning. Public-training test uses all registered training pairs as
conditioning.

## Baseline Family

The baseline family is frozen before execution. Each candidate is a tuple:

`(shape_rule, canvas_rule, background_rule)`

Shape rules:

- `same_as_input`: output shape equals query input shape;
- `transpose_input`: output shape is `(input_w, input_h)`;
- `conditioning_unanimous_output`: use the conditioning output shape only if
  all conditioning outputs share one shape;
- `conditioning_median_delta`: add the median conditioning
  `(output_h - input_h, output_w - input_w)` delta to the query input shape,
  clipped to `[1, 30]`;
- `nearest_conditioning_shape`: choose the output shape of the conditioning
  input nearest to the query input under the active arm's input distance.

Canvas rules:

- `constant_background`: fill the canvas with the selected background color;
- `identity_top_left`: copy the query input into the output canvas at top left,
  cropped or padded as needed;
- `rot90_top_left`;
- `rot180_top_left`;
- `rot270_top_left`;
- `reflect_h_top_left`;
- `reflect_v_top_left`;
- `transpose_top_left`;
- `anti_transpose_top_left`;
- `nonzero_bbox_top_left`: crop the query input's nonzero bounding box and
  paste it at top left, preserving colors.

Background rule:

- modal color across conditioning outputs, tie-broken by smallest color id.

Candidate selection:

1. For each conditioning pair, apply every candidate to the conditioning input.
2. Compute residual edit mass against the conditioning output:
   `residual_cells / output_cells`, after shape mismatch is penalized as `1.0`.
3. Select the candidate with lowest mean conditioning residual.
4. Tie-break by lower maximum conditioning residual, then lower canvas-rule
   index, then lower shape-rule index.

The selected candidate is recorded in `baseline_selection.csv`.

## Edit Learner

Each instance trains two per-task models from scratch.

Mask model:

- input: arm-specific query/baseline features plus cell coordinates, baseline
  color one-hot, local 3x3 baseline patch, and normalized shape features;
- output: edit/no-edit logit for each cell;
- architecture: `Linear(input_dim, 192) -> GELU -> LayerNorm(192) ->
  Linear(192, 192) -> GELU -> Linear(192, 1)`;
- loss: positive-class-weighted BCE;
- positive class weight: `clamp(no_edit_count / max(1, edit_count), 1.0, 20.0)`;
- optimizer: AdamW, lr `1e-3`, betas `(0.9, 0.99)`, eps `1e-8`,
  weight_decay `1e-4`;
- max steps: `700`;
- early-stop patience: `120`;
- gradient clip norm: `1.0`.

Color model:

- trained only on edited conditioning cells;
- if no edited conditioning cells exist, the color model is skipped and the
  predicted edit mask must be empty;
- architecture: `Linear(input_dim, 192) -> GELU -> LayerNorm(192) ->
  Linear(192, 192) -> GELU -> Dropout(0.05) -> Linear(192, 10)`;
- loss: class-balanced cross entropy using the Phase 3A class-weight rule;
- optimizer and training budget match the mask model.

Threshold selection:

- candidate mask thresholds: `0.10, 0.20, ..., 0.90`;
- choose the threshold that maximizes conditioning exact reconstruction;
- tie-break by higher edit-mask F1, then lower predicted edit mass, then
  threshold closest to `0.50`.

Initialization and determinism follow Phase 3A: Xavier uniform linear weights,
zero biases, unit LayerNorm weights, zero LayerNorm biases, PyTorch
deterministic algorithms, and SHA-256-derived model seeds.

## Seed Slate

Master seed slate:

- `20260528`
- `20260529`
- `20260530`
- `20260531`
- `20260601`

Derived seed key:

`arc-p3d-structured-edit-residual-v1\0<master_seed>\0<lane>\0<task_id>\0<query_index>\0<arm>\0<model_kind>`

`model_kind` is one of `mask`, `color`, or `threshold_tiebreak`.

Seed selection is validation-only and per arm:

1. highest validation non-baseline exact task count;
2. highest validation edit-mask F1;
3. highest validation minority-edit recall;
4. lowest validation over-edit rate;
5. lowest validation loss;
6. lowest numeric master seed.

## Metrics

The runner must report inherited exact-grid and shape/palette metrics plus:

- `baseline_exact_any_rate`;
- `baseline_residual_mass_mean`;
- `nonbaseline_exact_task_count`;
- `edit_mask_precision`;
- `edit_mask_recall`;
- `edit_mask_f1`;
- `minority_edit_recall`;
- `over_edit_rate`: no-edit target cells predicted as edits;
- `under_edit_rate`: edit target cells predicted as no-edit;
- `edit_color_accuracy`;
- `predicted_edit_mass_mean`;
- `target_edit_mass_mean`;
- `copy_advantage_suppressed`: true when exactness is baseline-only.

Any exact grid that is exact before applying learned edits is counted as
`baseline_exact`, not as non-baseline learned edit success. Branch D support
requires non-baseline exact success.

## Arena Gate

The arena opens only if `raw_grid_edit` produces non-baseline exact success on
both held-out lanes:

- at least one non-baseline exact task on `test_lodo`;
- at least one non-baseline exact task on `pttest`.

If this gate fails, the verdict is `branch_d_full_grid_edit_floor`. No
signature sufficiency comparison is licensed, and no extra seeds or narrower
subsets may be run without a new append-only amendment.

## Branch D Adjudication

If the raw-grid edit arena opens, compare `signature_palette_edit` to
`raw_grid_edit`.

`branch_d_support` requires:

- `signature_palette_edit` also opens the arena;
- on both held-out lanes, its non-baseline exact task count trails raw grid by
  no more than one task;
- its edit-mask F1 trails raw grid by no more than `0.10`;
- its minority-edit recall trails raw grid by no more than `0.10`;
- its over-edit rate exceeds raw grid by no more than `0.10`.

`branch_d_bounded_failure` applies when raw grid opens the arena but
`signature_palette_edit` does not satisfy the support criteria.

`branch_d_named_quarantine` may be used only as a diagnostic sub-verdict when
at least 80% of the non-baseline exact gap is allocated to the pre-registered
quarantine labels below and the non-quarantined slice satisfies
`branch_d_support`.

## Quarantine Labels

Every held-out non-exact row must receive at least one primary label:

- `baseline_shape_failure`: selected shape rule gives the wrong output shape;
- `baseline_canvas_failure`: shape is correct, but the baseline residual mass
  exceeds `0.50`;
- `edit_mask_underdetection`: edit-mask recall below `0.25`;
- `edit_mask_overedit`: over-edit rate above `0.50`;
- `edit_color_failure`: edit-mask F1 at least `0.50`, but edit-color accuracy
  below `0.50`;
- `conditioning_starvation`: fewer than 3 conditioning pairs;
- `copy_prior_absent`: target transformation is not well represented by any
  baseline candidate on conditioning pairs;
- `signature_edit_information_gap`: raw-grid edit succeeds and
  signature-palette edit fails on the same held-out instance;
- `palette_lift_failure`: strict signature quotient predicts edit geometry but
  cannot recover exact colors;
- `stochastic_instability`: seed outcomes disagree on exactness or edit-mask
  collapse.

## Required Artifacts

Binding output path:

`results/arc/phase3d-structured-edit-residual-v1/`

Required files:

- `manifest.json`;
- `split.csv`;
- `phase3d_receipt.json`;
- `scores.csv`;
- `per_task.csv`;
- `per_prior.csv`;
- `per_instance.csv`;
- `baseline_selection.csv`;
- `edit_metrics.csv`;
- `learning_curves.csv`;
- `seed_stability.csv`;
- `quarantine_log.csv`;
- `residuals.jsonl`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

The manifest must record this spec hash, parent spec hash, register hash,
feature schema, framing version, learner version, baseline candidate list,
selected seed by arm, arena gate, and branch adjudication.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- Python runner: `docs/prereg/arc/phase3d_structured_edit_residual.py`;
- Node wrapper: `scripts/arc-phase3d-structured-edit-residual-v1.mjs`;
- npm script: `arc:phase3d:structured-edit-residual-v1`;
- receipt path: `results/arc/phase3d-structured-edit-residual-v1/`.

The freeze-marker amendment must add the runner, wrapper, npm script, result
ignore path, leak-check coverage, and a smoke fingerprint before execution is
admitted.

Per the repo's ten-minute rule, the first implementation receipt must be a
capped probe unless the full run is measured under about ten minutes. Longer
full runs must be staged with exact PowerShell commands, wall-clock estimates,
resume-safety notes, and branch-decision consequences.

## Public Language

Allowed before a binding receipt:

> "Phase 3D has filed a structured edit/residual framing spec. No Branch D
> receipt exists yet, and no sufficiency claim is admitted."

Allowed if the raw-grid edit arena floors:

> "The first Branch D structured-edit framing did not open the full-grid edit
> arena; the result calibrates the framing and does not adjudicate signature
> sufficiency."

Allowed if Branch D support is filed:

> "Within the registered ARC public-training subset and this structured-edit
> framing, `signature_palette_edit` was competitive with the matched
> `raw_grid_edit` control on non-baseline held-out edit tasks."

Forbidden:

- any claim about ARC public-evaluation or Kaggle performance;
- any claim that Branch D support solves ARC;
- any claim that baseline-only exactness supports signature sufficiency;
- any signature-vs-full-grid claim when `raw_grid_edit` does not open the
  non-baseline arena;
- extra seeds or task narrowing after an arena-floor receipt without a new
  append-only amendment.

## Binding Receipt Addendum

Filed: **2026-05-28 (PT)**.

The Phase 3D binding receipt is filed in
[`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md) under "Branch D
20-Shard Binding Receipt: `branch_d_full_grid_edit_floor`".

Summary:

- receipt path: `results/arc/phase3d-structured-edit-residual-v1/`;
- merge/shard protocol: 20 shards, 4 arms x 5 seeds;
- verdict: `branch_d_full_grid_edit_floor`;
- raw-grid edit held-out non-baseline exact tasks: zero on `pttest`, zero on
  `test_lodo`;
- consequence: no `signature_palette_edit` vs. `raw_grid_edit` sufficiency
  comparison is licensed;
- named failure character: **edit-color-rule failure**.

The structured-edit framing decomposed the Phase 3 failure for the first time:
the baseline shape and canvas stages were often usable, the edit-mask learner
found moderate-to-strong WHERE signal, but the edit-color learner failed to
learn the WHAT-color rule well enough for exact reconstruction.

This addendum does not change the frozen framing contract above. It records
that the admitted lane has now run and closed Branch D in the filed
`structured_edit_residual_v1` framing. Future Phase 3 reopens require either a
new pre-registered Branch D variant or a new Branch E spec.

## Edit-Color-Rule Variant Addendum

Filed: **2026-05-28 (PT)**.

The first Branch D bottleneck variant is filed in
[`PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md`](PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md).
It keeps this spec's baseline picker and edit-mask learner fixed while replacing
the learned edit-color MLP with a deterministic conditioning-derived color-rule
bank. Execution remains held pending runner tooling and a freeze-marker
amendment.
