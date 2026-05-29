# Phase 3D Variant -- Edit Color Rule Bottleneck

Parent spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)

Base Branch D spec:
[`PHASE3D_DIFFERENT_FRAMING_SPEC.md`](PHASE3D_DIFFERENT_FRAMING_SPEC.md)

Filed: **2026-05-28 (PT)**

Status: **BINDING RECEIPT FILED --
`branch_d_color_rule_full_grid_floor`**. This file defines the first Branch D
variant after `structured_edit_residual_v1`. It changed only the edit-color
component identified as the Phase 3D bottleneck. The binding receipt did not
open the raw-grid color-rule arena, so no signature-vs-full-grid sufficiency
comparison is licensed in this variant.

## Purpose

The Phase 3D binding receipt decomposed the failure for the first time:

- baseline shape/canvas: often usable;
- edit-mask learner: moderate-to-strong WHERE signal;
- edit-color learner: dominant bottleneck.

The named failure mode was **edit-color-rule failure**: the model often knows
where to edit but not what color to write. This variant tests the narrowest
reasonable repair: keep the baseline picker and edit-mask machinery frozen, but
replace the per-task scratch color MLP with a deterministic color-rule bank
selected from conditioning residuals.

## Question

Does a structured-edit learner with symbolic conditioning-derived edit-color
rules open the full-grid edit arena on the registered Phase 0 task class, and,
if it does, is `signature_palette_edit_color_v2` competitive with the matched
`raw_grid_edit_color_v2` control?

## Scope And Non-Goals

This is a Branch D variant, not Branch E. It does not use a large language
model, external ARC training data, public-evaluation grids, Kaggle notebooks, or
non-registered public-training tasks.

This variant changes exactly one component relative to
`structured_edit_residual_v1`:

- changed: edit-color learner;
- unchanged: registered task class, splits, baseline family, edit-mask learner,
  seed slate, shard discipline, arena-gate discipline, and public-language
  discipline.

No extra seeds or narrower subset from the Phase 3D receipt are admitted here;
this is a new learner/framing variant with its own receipt path.

## Variant Version

Variant version: `structured_edit_color_rule_v2`.

Learner label: `edit_color_rule_bank_v1`.

The final grid is still reconstructed as:

`predicted_grid = apply_edit(predicted_baseline, predicted_edit_mask, predicted_edit_colors)`

Only `predicted_edit_colors` changes. The edit mask is produced by the same
per-instance mask model and threshold-selection rule as
`structured_edit_residual_v1`.

## Representation Arms

| arm | role | exact-grid eligible? |
| --- | --- | --- |
| `raw_grid_edit_color_v2` | matched full-grid control with color-rule bank | yes |
| `signature_palette_edit_color_v2` | primary Sundog edit-color variant | yes |
| `signature_only_edit_color_v2` | strict quotient diagnostic | quotient/color-lift diagnostics only |
| `metadata_only_edit_color_v2` | coarse nuisance control | no support claim; discrimination only |

Only `raw_grid_edit_color_v2` can open the arena. Only
`signature_palette_edit_color_v2` can support a Branch D variant sufficiency
claim.

## Inherited Components

The runner must inherit from `structured_edit_residual_v1` without changing:

- Phase 0 registered task subset and split;
- LODO and public-training-test lane definitions;
- baseline shape rules;
- baseline canvas rules;
- baseline candidate-selection rule;
- mask model architecture and training budget;
- mask threshold-selection rule;
- seed slate: `20260528`, `20260529`, `20260530`, `20260531`, `20260601`;
- two-stage manifest barrier for public-training test outputs;
- shard+merge discipline if the full run exceeds the ten-minute rule.

Any implementation deviation from this list must be filed as a new append-only
amendment before execution.

## Color Rule Bank

For each conditioning pair, compute the selected baseline, gold edit mask, and
gold edit colors. A gold edited cell is any cell where the baseline color differs
from the target output color. The color-rule bank is trained only on those gold
edited cells.

Each rule predicts the color for a candidate edited cell using only the query
input, selected baseline, cell coordinate, conditioning residual table, and the
active arm's representation features.

The frozen rule families are:

1. `constant_edit_color`: always emit one conditioning-observed edit color.
   Candidate colors are all colors observed in conditioning edit targets.
2. `modal_edit_color`: emit the modal conditioning edit target color.
3. `baseline_color_map`: map baseline color -> target edit color by majority
   vote over conditioning edited cells with the same baseline color.
4. `input_nn_color_map`: map nearest-neighbor query/input color -> target edit
   color by majority vote over conditioning edited cells.
5. `input_patch_majority_map`: map the majority color in the 3x3 nearest-input
   patch -> target edit color by majority vote.
6. `baseline_to_input_pair_map`: map `(baseline_color, input_nn_color)` ->
   target edit color by majority vote.
7. `relative_palette_rank_map`: map the rank of a source color within the input
   palette to the same rank, nearest rank, or conditioning-learned rank in the
   target edit palette.
8. `object_role_color_map`: role-normalize nonzero objects in the input and
   baseline, then map object role id -> target edit color by majority vote.
9. `row_col_periodic_color`: infer a period-1, period-2, or period-3 row/column
   color rule from conditioning edited cells.
10. `nearest_edited_neighbor_color`: for each query edited cell, copy the color
    from the nearest conditioning edited cell under normalized coordinate
    distance after applying the selected baseline transform family.

When a rule lacks a key at prediction time, it falls back to `modal_edit_color`.
If no conditioning edited cells exist, all predicted edit masks must be empty
and the receipt labels the instance as `no_conditioning_edits`.

## Rule Selection

For each instance, arm, and seed:

1. Build all concrete candidate rules from the rule families above.
2. Score each rule on conditioning residuals using leave-one-conditioning-pair
   out when at least three conditioning pairs exist; otherwise score on all
   conditioning residual cells and mark `low_k_rule_selection = true`.
3. Primary score: edit-color accuracy on gold edited cells.
4. Tie-breaks:
   - higher rare-color recall, where rare colors are non-modal edit target
     colors;
   - lower color hallucination rate on no-edit cells under the selected mask;
   - lower rule-family index;
   - lower SHA-256 tie-break key:
     `arc-p3d-edit-color-rule-v2\0<master_seed>\0<lane>\0<task_id>\0<query_index>\0<arm>\0<rule_id>`.

The runner must emit `color_rule_selection.csv` with one row per selected rule
and `color_rule_candidates.csv` with all concrete candidate scores.

## Optional Ensemble

The first receipt admits a deterministic top-3 vote ensemble only if at least
three candidate rules tie within `0.05` edit-color accuracy on conditioning.
The ensemble prediction is a plurality vote by color, tie-broken by the highest
ranked individual rule. The manifest must record whether the selected predictor
is a single rule or an ensemble.

No learned color MLP is admitted in this variant.

## Metrics

In addition to all `structured_edit_residual_v1` metrics, the receipt must
report:

- `edit_color_rule_accuracy`;
- `rare_edit_color_recall`;
- `color_rule_family`;
- `color_rule_candidate_count`;
- `low_k_rule_selection_rate`;
- `no_conditioning_edits_rate`;
- `color_oracle_rule_accuracy`: best candidate rule accuracy on the target
  residual, reported only as an oracle diagnostic after scoring;
- `rule_selection_regret`: oracle accuracy minus selected-rule accuracy;
- `mask_conditioned_color_accuracy`: color accuracy only on cells where the
  predicted mask and target edit mask both mark edit.

The branch decision may not use `color_oracle_rule_accuracy`; it exists only to
separate rule-bank coverage from rule-selection failure.

## Arena Gate

The arena opens only if `raw_grid_edit_color_v2` produces non-baseline exact
success on both held-out lanes:

- at least one non-baseline exact task on `test_lodo`;
- at least one non-baseline exact task on `pttest`.

If this gate fails, the verdict is `branch_d_color_rule_full_grid_floor`. No
signature sufficiency comparison is licensed, and no extra seeds or narrower
subsets may be run without a new append-only amendment.

## Branch D Variant Adjudication

If the raw-grid color-rule arena opens, compare
`signature_palette_edit_color_v2` to `raw_grid_edit_color_v2`.

`branch_d_color_rule_support` requires:

- `signature_palette_edit_color_v2` also opens the arena;
- on both held-out lanes, its non-baseline exact task count trails raw grid by
  no more than one task;
- its `edit_color_rule_accuracy` trails raw grid by no more than `0.10`;
- its `rare_edit_color_recall` trails raw grid by no more than `0.10`;
- its `rule_selection_regret` exceeds raw grid by no more than `0.10`.

`branch_d_color_rule_bounded_failure` applies when raw grid opens the arena but
the signature-palette arm does not satisfy support.

`branch_d_color_rule_named_quarantine` may be used only as a diagnostic
sub-verdict when at least 80% of the non-baseline exact gap is allocated to the
pre-registered labels below and the non-quarantined slice satisfies
`branch_d_color_rule_support`.

## Quarantine Labels

Every held-out non-exact row must receive at least one primary label:

- `baseline_shape_failure`: inherited baseline shape wrong;
- `baseline_canvas_failure`: inherited baseline residual mass exceeds `0.50`;
- `edit_mask_failure`: inherited predicted mask F1 below `0.50`;
- `color_rule_bank_coverage_failure`: oracle rule accuracy below `0.50`;
- `color_rule_selection_failure`: oracle rule accuracy at least `0.50`, selected
  rule accuracy below `0.50`;
- `source_binding_failure`: selected rule uses a source-binding family and the
  source key is missing or mismatched on target residual cells;
- `rare_color_failure`: rare edit-color recall below `0.25`;
- `no_conditioning_edits`: no gold edited cells exist in conditioning residuals;
- `conditioning_starvation`: fewer than three conditioning pairs;
- `palette_lift_failure`: strict signature quotient predicts edit geometry but
  cannot recover exact colors;
- `signature_edit_color_gap`: raw-grid color-rule succeeds and
  signature-palette color-rule fails on the same held-out instance;
- `stochastic_instability`: seed outcomes disagree on non-baseline exactness.

## Required Artifacts

Binding output path:

`results/arc/phase3d-edit-color-rule-v2/`

Required files:

- `manifest.json`;
- `split.csv`;
- `phase3d_color_rule_receipt.json`;
- `scores.csv`;
- `per_task.csv`;
- `per_prior.csv`;
- `per_instance.csv`;
- `baseline_selection.csv`;
- `edit_metrics.csv`;
- `color_rule_selection.csv`;
- `color_rule_candidates.csv`;
- `learning_curves.csv` for the inherited mask learner;
- `seed_stability.csv`;
- `quarantine_log.csv`;
- `residuals.jsonl`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

The manifest must record this spec hash, parent spec hash, base Branch D spec
hash, feature schema, variant version, learner label, rule-family list, selected
seed by arm, arena gate, and branch adjudication.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- Python runner: `docs/prereg/arc/phase3d_edit_color_rule_v2.py`;
- Node wrapper: `scripts/arc-phase3d-edit-color-rule-v2.mjs`;
- npm script: `arc:phase3d:edit-color-rule-v2`;
- shard npm script: `arc:phase3d:edit-color-rule-v2:shard`;
- merge npm script: `arc:phase3d:edit-color-rule-v2:merge`;
- receipt path: `results/arc/phase3d-edit-color-rule-v2/`.

The freeze-marker amendment must add the runner, wrapper, npm scripts, result
ignore path, leak-check coverage, and a smoke fingerprint before execution is
admitted.

Per the repo's ten-minute rule, the first implementation receipt must be a
capped probe unless the full run is measured under about ten minutes. Longer
full runs must be staged with exact PowerShell commands, wall-clock estimates,
resume-safety notes, and branch-decision consequences.

## Public Language

Allowed before a binding receipt:

> "Phase 3D has filed an edit-color-rule variant spec targeting the bottleneck
> isolated by the structured-edit receipt. No variant receipt exists yet, and no
> sufficiency claim is admitted."

Allowed if the raw-grid color-rule arena floors:

> "The first Branch D edit-color-rule variant did not open the full-grid edit
> arena; the result calibrates the color-rule repair and does not adjudicate
> signature sufficiency."

Allowed if support is filed:

> "Within the registered ARC public-training subset and this color-rule
> structured-edit variant, `signature_palette_edit_color_v2` was competitive
> with the matched `raw_grid_edit_color_v2` control on non-baseline held-out
> edit tasks."

Forbidden:

- any claim about ARC public-evaluation or Kaggle performance;
- any claim that this variant solves ARC;
- any use of oracle rule accuracy as branch support;
- any signature-vs-full-grid claim when `raw_grid_edit_color_v2` does not open
  the non-baseline arena;
- extra seeds or task narrowing after an arena-floor receipt without a new
  append-only amendment.

## Binding Receipt Addendum

Filed: **2026-05-28 (PT)**.

Binding receipt path:

`results/arc/phase3d-edit-color-rule-v2/`

The binding receipt is summarized in `PHASE3_SUFFICIENCY_SPEC.md` under
"Variant 20-Shard Binding Receipt:
`branch_d_color_rule_full_grid_floor` (bottleneck-shifted)". The verdict is
**`branch_d_color_rule_full_grid_floor`**: `raw_grid_edit_color_v2` scored zero
non-baseline exact tasks on at least one held-out arena lane, so the arena did
not open and no `signature_palette_edit_color_v2` vs.
`raw_grid_edit_color_v2` sufficiency comparison is licensed.

This is the sixth Phase 3 full-grid floor. The variant nevertheless achieved
its diagnostic design goal: it shifted the bottleneck from undifferentiated
edit-color learning to a measured decomposition of the remaining failure.

Primary bottleneck allocation from the binding receipt:

- `edit_mask_failure`: **41%** of failures;
- `color_rule_selection_failure`: **16%** of failures, with approximately
  `0.30` to `0.35` rule-selection regret in the locked-accuracy slice;
- `color_rule_bank_coverage_failure`: **9%** of failures.

The measured order of leverage for later Phase 3 reopens is:

1. mask-targeted Branch D variant;
2. selection-refinement Branch D variant;
3. rule-bank extension Branch D variant;
4. Branch E spec such as generative program search or test-time prompting.

The first item is now filed in `PHASE3D_MASK_TARGET_VARIANT_SPEC.md` as
`structured_edit_mask_target_v3`. That filing admits no execution and no
signature comparison before a binding receipt exists.
