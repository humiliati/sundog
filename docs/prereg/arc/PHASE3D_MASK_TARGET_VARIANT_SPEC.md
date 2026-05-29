# Phase 3D Variant -- Mask-Targeted Structured Edit

Parent spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)

Base Branch D spec:
[`PHASE3D_DIFFERENT_FRAMING_SPEC.md`](PHASE3D_DIFFERENT_FRAMING_SPEC.md)

Prior Branch D variant:
[`PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md`](PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md)

Filed: **2026-05-28 (PT)**

Status: **SPEC FILED; EXECUTION HOLD**. This file starts the second Branch D
variant after `structured_edit_residual_v1`. It targets the dominant bottleneck
identified by the `structured_edit_color_rule_v2` binding receipt:
`edit_mask_failure` at 41% of failures. It admits no run until runner tooling,
npm wiring, result ignore path, leak-check coverage, and a freeze-marker
amendment are committed together.

## Purpose

The `structured_edit_color_rule_v2` receipt eliminated the original
`edit_color_failure` label as the dominant Phase 3D blocker, but the raw-grid
arena still floored with verdict `branch_d_color_rule_full_grid_floor`. The
remaining failure decomposition named the next largest target:

- `edit_mask_failure`: 41% of failures;
- `color_rule_selection_failure`: 16% of failures;
- `color_rule_bank_coverage_failure`: 9% of failures.

This variant tests the narrowest reasonable repair for the leading slice:
replace the inherited scratch mask MLP with a deterministic mask-candidate bank
selected from conditioning residuals, while keeping the baseline picker and the
edit-color rule bank frozen from the prior variant.

## Question

Does a structured-edit learner with a conditioning-derived mask-candidate bank
open the full-grid edit arena on the registered Phase 0 task class, and, if it
does, is `signature_palette_edit_mask_v3` competitive with the matched
`raw_grid_edit_mask_v3` control?

## Scope And Non-Goals

This is a Branch D variant, not Branch E. It does not use a large language
model, external ARC training data, public-evaluation grids, Kaggle notebooks, or
non-registered public-training tasks.

This variant changes exactly one component relative to
`structured_edit_color_rule_v2`:

- changed: edit-mask predictor;
- unchanged: registered task class, splits, baseline family, baseline
  selection, edit-color rule bank, color-rule selection, seed slate, shard
  discipline, arena-gate discipline, and public-language discipline.

No extra seeds or narrower subset from the Phase 3D or edit-color-rule receipts
are admitted here. This is a new mask-targeted variant with its own receipt
path.

## Variant Version

Variant version: `structured_edit_mask_target_v3`.

Learner label: `edit_mask_candidate_bank_v1`.

The final grid is still reconstructed as:

`predicted_grid = apply_edit(predicted_baseline, predicted_edit_mask, predicted_edit_colors)`

Only `predicted_edit_mask` changes. The baseline is produced by the inherited
`structured_edit_residual_v1` baseline picker. Edit colors are produced by the
inherited deterministic `edit_color_rule_bank_v1` from
`structured_edit_color_rule_v2`.

## Representation Arms

| arm | role | exact-grid eligible? |
| --- | --- | --- |
| `raw_grid_edit_mask_v3` | matched full-grid control with mask-candidate bank | yes |
| `signature_palette_edit_mask_v3` | primary Sundog mask-targeted variant | yes |
| `signature_only_edit_mask_v3` | strict quotient diagnostic | quotient/color-lift diagnostics only |
| `metadata_only_edit_mask_v3` | coarse nuisance control | no support claim; discrimination only |

Only `raw_grid_edit_mask_v3` can open the arena. Only
`signature_palette_edit_mask_v3` can support a Branch D mask-targeted
sufficiency claim.

## Inherited Components

The runner must inherit from `structured_edit_color_rule_v2` without changing:

- Phase 0 registered task subset and split;
- LODO and public-training-test lane definitions;
- baseline shape rules;
- baseline canvas rules;
- baseline candidate-selection rule;
- color-rule bank families;
- color-rule candidate construction;
- color-rule selection and tie-breaks;
- optional top-3 color-rule ensemble rule;
- seed slate: `20260528`, `20260529`, `20260530`, `20260531`, `20260601`;
- two-stage manifest barrier for public-training test outputs;
- shard+merge discipline if the full run exceeds the ten-minute rule.

Any implementation deviation from this list must be filed as a new append-only
amendment before execution.

## Mask Candidate Bank

For each conditioning pair, compute the selected baseline and the gold edit
mask. A gold edited cell is any output cell where the selected baseline color
differs from the target output color.

Each mask candidate predicts a Boolean edit mask for the query output canvas
using only the query input, selected baseline, cell coordinate, conditioning
gold masks, conditioning residual metadata, and the active arm's representation
features. Candidate masks must be deterministic functions of those inputs and
the master seed tie-break key; they must not inspect held-out target outputs.

The frozen candidate families are:

1. `empty_mask`: predict no edits.
2. `conditioning_mask_union`: transfer the normalized coordinate union of
   conditioning gold masks to the query output shape.
3. `conditioning_mask_intersection`: transfer the normalized coordinate
   intersection of conditioning gold masks to the query output shape.
4. `conditioning_mask_majority`: transfer cells edited in at least half of
   conditioning gold masks after normalized coordinate alignment.
5. `conditioning_bbox_fill`: transfer each conditioning gold-mask bounding box,
   fill it on the query output shape, and emit one candidate per distinct box
   aggregate: union, intersection, and majority.
6. `conditioning_bbox_outline`: transfer the one-cell outline of the aggregate
   conditioning edit bounding boxes.
7. `row_col_periodic_mask`: infer period-1, period-2, or period-3 row, column,
   checkerboard, or diagonal mask rules from conditioning gold masks.
8. `source_color_mask`: edit query cells whose mapped input or baseline color
   belongs to a source-color set that was edited in conditioning residuals.
9. `source_color_pair_mask`: edit query cells whose
   `(mapped_input_color, baseline_color)` pair was edited in conditioning
   residuals.
10. `object_role_mask`: compute connected components over non-background input
    and baseline canvases, role-normalize by size, centroid, and color rank,
    then edit roles that were edited in conditioning residuals.
11. `nearest_residual_patch_mask`: classify each query output cell by nearest
    conditioning residual cells under normalized coordinate distance plus local
    3x3 input/baseline color patch distance. Candidate thresholds are
    `0.25`, `0.50`, and `0.75` edit-vote probability.
12. `delta_overlay_mask`: transfer the per-pair residual delta overlay already
    registered for Phase 3 candidate-combinator diagnostics into the selected
    baseline's output coordinate frame.
13. `legacy_mlp_threshold_mask`: train the inherited
    `structured_edit_residual_v1` mask MLP and expose thresholded masks at
    `0.10, 0.20, ..., 0.90` as candidates.

Morphological variants are admitted for candidate families 2 through 13:

- identity;
- one-cell dilation;
- one-cell erosion;
- one-cell closing;
- connected-component fill inside each predicted component's bounding box.

Morphological variants must be emitted as distinct candidate IDs and counted in
`mask_candidate_count`.

## Mask Selection

For each instance, arm, and seed:

1. Build all concrete mask candidates from the families above.
2. Score each candidate on conditioning residuals using
   leave-one-conditioning-pair out when at least three conditioning pairs exist;
   otherwise score on all conditioning residual cells and mark
   `low_k_mask_selection = true`.
3. Primary score: edit-mask F1 on gold edited cells.
4. Secondary score: nonmodal-edit recall, where nonmodal edited cells are cells
   whose target edit color is not the modal conditioning edit target color.
5. Tie-breaks:
   - higher precision;
   - lower absolute edit-mass error;
   - lower over-edit rate;
   - lower candidate-family index;
   - lower SHA-256 tie-break key:
     `arc-p3d-mask-target-v3\0<master_seed>\0<lane>\0<task_id>\0<query_index>\0<arm>\0<candidate_id>`.

The runner must emit `mask_candidate_selection.csv` with one row per selected
candidate and `mask_candidates.csv` with all concrete candidate scores.

No learned cross-task mask selector is admitted in this variant. The inherited
mask MLP may appear only as the `legacy_mlp_threshold_mask` candidate family.

## Metrics

In addition to all `structured_edit_color_rule_v2` metrics, the receipt must
report:

- `mask_candidate_family`;
- `mask_candidate_count`;
- `low_k_mask_selection_rate`;
- `mask_candidate_f1`;
- `mask_candidate_precision`;
- `mask_candidate_recall`;
- `mask_nonmodal_edit_recall`;
- `mask_iou`;
- `mask_mass_error`;
- `mask_oracle_candidate_f1`: best candidate F1 on the target residual,
  reported only as an oracle diagnostic after scoring;
- `mask_selection_regret`: oracle candidate F1 minus selected-candidate F1;
- `mask_oracle_exact_nonbaseline`: whether any candidate mask would have
  enabled exact non-baseline reconstruction when paired with the inherited
  color-rule bank, reported only as an oracle diagnostic.

The branch decision may not use `mask_oracle_candidate_f1` or
`mask_oracle_exact_nonbaseline`; they exist only to separate mask-bank coverage
from mask-selection failure.

## Arena Gate

The arena opens only if `raw_grid_edit_mask_v3` produces non-baseline exact
success on both held-out lanes:

- at least one non-baseline exact task on `test_lodo`;
- at least one non-baseline exact task on `pttest`.

If this gate fails, the verdict is `branch_d_mask_target_full_grid_floor`. No
signature sufficiency comparison is licensed, and no extra seeds or narrower
subsets may be run without a new append-only amendment.

## Branch D Mask-Targeted Adjudication

If the raw-grid mask-targeted arena opens, compare
`signature_palette_edit_mask_v3` to `raw_grid_edit_mask_v3`.

`branch_d_mask_target_support` requires:

- `signature_palette_edit_mask_v3` also opens the arena;
- on both held-out lanes, its non-baseline exact task count trails raw grid by
  no more than one task;
- its edit-mask F1 trails raw grid by no more than `0.10`;
- its nonmodal-edit recall trails raw grid by no more than `0.10`;
- its over-edit rate exceeds raw grid by no more than `0.10`;
- its `mask_selection_regret` exceeds raw grid by no more than `0.10`;
- its inherited `edit_color_rule_accuracy` trails raw grid by no more than
  `0.10`.

`branch_d_mask_target_bounded_failure` applies when raw grid opens the arena
but the signature-palette arm does not satisfy support.

`branch_d_mask_target_named_quarantine` may be used only as a diagnostic
sub-verdict when at least 80% of the non-baseline exact gap is allocated to the
pre-registered labels below and the non-quarantined slice satisfies
`branch_d_mask_target_support`.

## Quarantine Labels

Every held-out non-exact row must receive at least one primary label:

- `baseline_shape_failure`: inherited baseline shape wrong;
- `baseline_canvas_failure`: inherited baseline residual mass exceeds `0.50`;
- `mask_candidate_coverage_failure`: oracle candidate F1 below `0.50`;
- `mask_selection_failure`: oracle candidate F1 at least `0.50`, selected
  candidate F1 below `0.50`;
- `mask_overedit_failure`: selected mask precision below `0.50` or over-edit
  rate above `0.50`;
- `mask_underdetection_failure`: selected mask recall below `0.50`;
- `edit_color_rule_failure`: selected mask F1 at least `0.50`, but inherited
  edit-color rule accuracy below `0.50`;
- `color_rule_bank_coverage_failure`: inherited oracle color-rule accuracy
  below `0.50`;
- `color_rule_selection_failure`: inherited oracle color-rule accuracy at
  least `0.50`, selected color-rule accuracy below `0.50`;
- `source_binding_failure`: selected mask or color rule uses a source-binding
  family and the source key is missing or mismatched on target residual cells;
- `rare_color_failure`: rare edit-color recall below `0.25`;
- `no_conditioning_edits`: no gold edited cells exist in conditioning
  residuals;
- `conditioning_starvation`: fewer than three conditioning pairs;
- `palette_lift_failure`: strict signature quotient predicts edit geometry but
  cannot recover exact colors;
- `signature_mask_information_gap`: raw-grid mask-targeted arm succeeds and
  signature-palette mask-targeted arm fails on the same held-out instance;
- `signature_edit_color_gap`: raw-grid color-rule succeeds and
  signature-palette color-rule fails on the same held-out instance;
- `stochastic_instability`: seed outcomes disagree on non-baseline exactness.

## Required Artifacts

Binding output path:

`results/arc/phase3d-mask-target-v3/`

Required files:

- `manifest.json`;
- `split.csv`;
- `phase3d_mask_target_receipt.json`;
- `scores.csv`;
- `per_task.csv`;
- `per_prior.csv`;
- `per_instance.csv`;
- `baseline_selection.csv`;
- `edit_metrics.csv`;
- `mask_candidate_selection.csv`;
- `mask_candidates.csv`;
- `color_rule_selection.csv`;
- `color_rule_candidates.csv`;
- `learning_curves.csv` for the inherited legacy mask MLP candidate family;
- `seed_stability.csv`;
- `quarantine_log.csv`;
- `residuals.jsonl`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

The manifest must record this spec hash, parent spec hash, base Branch D spec
hash, prior color-rule variant spec hash, feature schema, variant version,
learner label, mask candidate-family list, inherited color-rule-family list,
selected seed by arm, arena gate, and branch adjudication.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- Python runner: `docs/prereg/arc/phase3d_mask_target_v3.py`;
- Node wrapper: `scripts/arc-phase3d-mask-target-v3.mjs`;
- npm script: `arc:phase3d:mask-target-v3`;
- shard npm script: `arc:phase3d:mask-target-v3:shard`;
- merge npm script: `arc:phase3d:mask-target-v3:merge`;
- receipt path: `results/arc/phase3d-mask-target-v3/`.

The freeze-marker amendment must add the runner, wrapper, npm scripts, result
ignore path, leak-check coverage, and a smoke fingerprint before execution is
admitted.

Per the repo's ten-minute rule, the first implementation receipt must be a
capped probe unless the full run is measured under about ten minutes. Longer
full runs must be staged with exact PowerShell commands, wall-clock estimates,
resume-safety notes, and branch-decision consequences.

## Public Language

Allowed before a binding receipt:

> "Phase 3D has filed a mask-targeted structured-edit variant spec targeting
> the leading bottleneck isolated by the edit-color-rule receipt. No
> mask-targeted receipt exists yet, and no sufficiency claim is admitted."

Allowed if the raw-grid mask-targeted arena floors:

> "The Branch D mask-targeted variant did not open the full-grid edit arena;
> the result calibrates the mask repair and does not adjudicate signature
> sufficiency."

Allowed if support is filed:

> "Within the registered ARC public-training subset and this mask-targeted
> structured-edit variant, `signature_palette_edit_mask_v3` was competitive
> with the matched `raw_grid_edit_mask_v3` control on non-baseline held-out edit
> tasks."

Forbidden:

- any claim about ARC public-evaluation or Kaggle performance;
- any claim that this variant solves ARC;
- any use of oracle mask diagnostics as branch support;
- any signature-vs-full-grid claim when `raw_grid_edit_mask_v3` does not open
  the non-baseline arena;
- extra seeds or task narrowing after an arena-floor receipt without a new
  append-only amendment.
