# Branch E v2 -- Deterministic Program-Search Solver Expansion

Parent / boundary specs:

- [`PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md`](PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md)
- [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)
- [`PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md`](PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md)
- [`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md)

Filed: **2026-05-29 (PT)**

Status: **SPEC + TOOLING FREEZE-MARKER AMENDMENT FILED; BINDING RECEIPT NOT FILED**.

## Boundary Citation

Branch E v1 returned `branch_e_capability_demonstrated`: a deterministic typed
search over frozen primitives, selected only by train-pair consistency, solved
at least two distinct held-out tasks on both `test_lodo` and `pttest`. That
result is the first non-zero exact-match capability receipt in the Sundog-ARC
program, but it is modest: two distinct gated tasks per lane and about 3% exact
instance rate.

The v1 receipt explicitly deferred three deterministic expansions:

1. intricate deterministic mask families;
2. morphology over deterministic masks;
3. deeper composition beyond depth 2.

Branch E v2 tests exactly those three expansions. It does not use
`signature_palette` geometry, certificate neighborhoods, held-out target outputs,
public-evaluation grids, or learned weights. The Phase 3E certificate program
still stands as a no-collision / no-locality boundary, not as a selector.

## Core Question

Does admitting the deferred deterministic mask families, morphology
cross-product, and depth-3 composition produce a material held-out exact-match
lift above the Branch E v1 capability baseline on the expanded 108-task
registered public-training universe?

## Frozen Inputs

```text
primary_register = docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv
primary_split    = sha256_expansion
U_primary        = test_lodo union pttest
U_all            = validation_lodo union validation_pttest union test_lodo union pttest
```

Training-split only. Public-evaluation grids remain forbidden until Phase 6. The
two-stage no-target fingerprint barrier is written before any `pttest` target
output is read.

## Frozen Primitive Delta from v1

Inherited unchanged from Branch E v1:

- all v1 structural transforms: `identity`, `d4`, `tile`, `translate`, `pad`,
  `crop_bbox`, `scale`, `palette_permute`, `fill_enclosed`,
  `extract_largest_component`;
- all v1 candidate-combinator candidates: `output_copy`, `delta_overlay`;
- the Branch D deterministic color-rule bank:
  `constant_edit_color`, `modal_edit_color`, `baseline_color_map`,
  `input_nn_color_map`, `input_patch_majority_map`,
  `baseline_to_input_pair_map`, `relative_palette_rank_map`,
  `object_role_color_map`, `row_col_periodic_color`,
  `nearest_edited_neighbor_color`;
- top-2 attempts, train-pair consistency selection, deterministic ranking, and
  no use of signature geometry.

Newly admitted in v2:

```text
mask_families_v2 =
  empty_mask
  conditioning_mask_union
  conditioning_mask_intersection
  conditioning_mask_majority
  conditioning_bbox_fill
  conditioning_bbox_outline
  row_col_periodic_mask
  source_color_mask
  source_color_pair_mask
  object_role_mask
  nearest_residual_patch_mask
  delta_overlay_mask
  full_mask

mask_morphology_v2 =
  identity
  dilate1
  erode1
  close1
  bbox_fill
```

`legacy_mlp_threshold_mask` remains excluded because it is learned. The v2 lane
is deterministic-bank only.

## Search Contract

```text
max_composition_depth = 3
candidate_budget_per_instance = 20000
attempts = 2

program =
    base
  | structural >> base
  | structural >> structural >> base

base =
    structural
  | candidate_combinator
  | color_rule_bank(mask_family_v2, morphology_v2)
```

Depth-3 exists to admit transforms such as `crop >> scale >> recolor` or
`d4 >> crop >> mask_recolor`. The first two stages of a depth-3 program must be
structural; color/edit is terminal. Every candidate must reproduce all
conditioning train outputs exactly before it is admitted. Ranking remains:

1. family priority;
2. description length / complexity;
3. lexicographic program id;
4. top-2 distinct output grids.

The candidate budget is a deterministic cap. If exhausted, the instance logs
`budget_exhausted=true`; thresholds and branch gates are not changed.

## Branches

Branch precedence is table order.

| branch | condition on `U_primary` | interpretation |
| --- | --- | --- |
| `branch_e_v2_material_lift` | `test_lodo` solves at least 4 distinct tasks, `pttest` solves at least 4 distinct tasks, and each lane solves at least 2 tasks outside the frozen v1 gated solve set `{be94b721, f25fbde4}`. | v2 materially lifts the v1 capability baseline. |
| `branch_e_v2_capability_replicated` | at least 2 distinct tasks solved on both `test_lodo` and `pttest`, but the material-lift condition is not met. | v2 preserves the v1 floor-clearing capability but does not materially improve it. |
| `branch_e_v2_capability_partial` | at least one exact held-out `U_primary` instance is solved, but the replicated threshold is not met. | non-zero signal below the v1 floor-clearing baseline. |
| `branch_e_v2_capability_floor` | zero exact held-out `U_primary` instances solved. | the expansion floors. |

No threshold, family, budget, depth, or ranking rule may be retuned after seeing
held-out outputs.

## Required Artifacts

Binding output path:

```text
results/arc/phase3-branch-e-v2-program-search/
```

Required files:

- `manifest.json`
- `split.csv`
- `context_fingerprints_no_targets.jsonl`
- `context_fingerprints_no_targets.sha256`
- `programs_by_instance.jsonl`
- `solutions_by_instance.csv`
- `capability_summary.csv`
- `per_prior_capability.csv`
- `family_usage.csv`
- `v1_comparison.csv`
- `phase3_branch_e_v2_program_search_receipt.json`
- `branch_adjudication.md`
- `commands.md`
- `hashes.json`

The manifest must record the v2 spec hash, v1 parent spec/runner hashes, register
hash, split mode, target-output barrier hash, frozen family list, mask morphology
list, search budget/depth/attempts, branch, and source-code hash.

## Reserved Implementation Names

- Python runner: `docs/prereg/arc/phase3_branch_e_v2_program_search.py`
- Node wrapper: `scripts/arc-phase3-branch-e-v2-program-search.mjs`
- npm script: `arc:phase3:branch-e-v2-program-search`
- receipt path: `results/arc/phase3-branch-e-v2-program-search/`

## Public Language

Allowed before a binding receipt:

> "Branch E v2 has filed a deterministic program-search expansion spec. It tests
> the exact deferred v1 expansions: deterministic intricate masks, morphology,
> and depth-3 composition. No binding receipt exists yet."

Allowed if `branch_e_v2_material_lift`:

> "Branch E v2 materially lifted the v1 deterministic program-search capability
> baseline on held-out public-training lanes. This remains a capability result,
> not a Blackwell-sufficiency proof, ARC solve, public-evaluation result, or
> Kaggle claim."

Allowed if `branch_e_v2_capability_replicated`:

> "Branch E v2 replicated the v1 floor-clearing capability but did not materially
> lift the exact-match solve rate under the frozen branch gate."

Forbidden:

- claiming Branch E v2 proves Blackwell sufficiency;
- claiming any public-evaluation or Kaggle result;
- selecting programs by `signature_palette` geometry;
- adding learned mask models;
- retuning the branch gates after seeing outputs.

---

## Amendment A -- Tooling Freeze Marker (2026-05-29 PT)

Append-only. Records the v2 executable lane, smoke receipts, timing estimate, and
binding command. No binding receipt is filed by this amendment.

### Tooling added

- Python runner: `docs/prereg/arc/phase3_branch_e_v2_program_search.py`.
  It imports Branch E v1 as a frozen support library for loaders, IO, structural
  transforms, candidate-combinator candidates, and the deterministic color-rule
  bank. The v2 runner owns only the expanded deterministic mask bank, morphology
  application, depth-3 search driver, v1-comparison rows, and v2 branch
  adjudication.
- Wrapper: `scripts/arc-phase3-branch-e-v2-program-search.mjs`, honoring
  `SUNDOG_PYTHON`.
- npm script: `arc:phase3:branch-e-v2-program-search`.
- gitignore path: `results/arc/phase3-branch-e-v2-program-search/`.

### Frozen v2 delta

The executable delta is exactly the spec delta above:

- `legacy_mlp_threshold_mask` remains excluded;
- deterministic mask families now include `row_col_periodic_mask`,
  `source_color_pair_mask`, `object_role_mask`, and
  `nearest_residual_patch_mask`;
- deterministic morphology now includes `dilate1`, `erode1`, `close1`, and
  `bbox_fill`;
- search permits depth 3 as `structural >> structural >> terminal`.

### Smoke receipts

`py_compile` passed:

```powershell
python -m py_compile docs\prereg\arc\phase3_branch_e_v2_program_search.py
```

Dry-run passed and emitted the full empty artifact set:

```powershell
node scripts/arc-phase3-branch-e-v2-program-search.mjs `
  --out results/arc/_phase3-branch-e-v2-dry `
  --dry-run `
  --allow-dirty
```

Capped one-task smoke:

```powershell
node scripts/arc-phase3-branch-e-v2-program-search.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv `
  --split-mode sha256_expansion `
  --limit-tasks 1 `
  --out results/arc/_phase3-branch-e-v2-smoke1 `
  --allow-dirty
```

Fingerprint: 3 `U_primary` instances (`test_lodo=2`, `pttest=1`), 21.122 s
wall, all required artifacts emitted, branch `branch_e_v2_capability_floor`
inside the deliberately tiny cap. All three instances reported
`budget_exhausted=true`, which is expected under the expanded 20k candidate cap
and is logged rather than hidden.

Known-v1-retention smoke:

```powershell
node scripts/arc-phase3-branch-e-v2-program-search.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv `
  --split-mode sha256_expansion `
  --task-id be94b721,f25fbde4 `
  --out results/arc/_phase3-branch-e-v2-v1-retention-smoke `
  --allow-dirty
```

Fingerprint: 9 gated instances (`test_lodo=7`, `pttest=2`), 29.456 s wall,
branch `branch_e_v2_capability_replicated` inside the task-id smoke filter.
It retained both v1 gated solved tasks on both lanes:

| lane | v1 solved tasks | v2 solved tasks | retained v1 | new v2 |
| --- | ---: | ---: | ---: | ---: |
| `test_lodo` | 2 | 2 | 2 | 0 |
| `pttest` | 2 | 2 | 2 | 0 |

Winning families matched v1: `extract_largest_component` and `crop_bbox>>scale`.
This sanity check shows the v2 runner has not regressed the v1 structural path.

### Ten-minute rule decision

A 12-task smoke was attempted first and exceeded the repository ten-minute rule
under the current CPU-contended workstation state; the stray child process was
stopped and the partial scratch receipt is not a branch artifact. The accepted
smokes above imply a rough current rate between 3.3 and 7.0 s per held-out
instance, depending on task mix and cache behavior. With `U_all=491` for the
expanded register, the full binding run is estimated at roughly 30-60 minutes on
the current machine.

Therefore the binding run is **staged for the operator**, not run inline by this
agent turn.

### Exact staged binding command

Run from the freeze-marker commit in a clean worktree:

```powershell
node scripts/arc-phase3-branch-e-v2-program-search.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv `
  --split-mode sha256_expansion `
  --out results/arc/phase3-branch-e-v2-program-search
```

Read back:

- `results/arc/phase3-branch-e-v2-program-search/manifest.json`
- `results/arc/phase3-branch-e-v2-program-search/capability_summary.csv`
- `results/arc/phase3-branch-e-v2-program-search/v1_comparison.csv`
- `results/arc/phase3-branch-e-v2-program-search/branch_adjudication.md`

Branch mapping is frozen in this spec: material lift, replicated capability,
partial capability, or floor. If the worktree must be dirty because unrelated
long jobs are running, the operator may add `--allow-dirty`; that override must
be called out in the verdict amendment.
