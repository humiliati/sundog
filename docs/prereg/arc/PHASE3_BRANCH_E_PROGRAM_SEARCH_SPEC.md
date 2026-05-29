# Branch E — Deterministic Program-Search Solver

Parent / boundary specs:

- [`PHASE3_5_REFLECTION.md`](PHASE3_5_REFLECTION.md) (Branch E is the live frontier)
- [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md) (arena-gate / floor precedent)
- [`PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md`](PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md)
- [`PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md`](PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md)
- [`PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md`](PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md)
- [`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md)

Filed: **2026-05-29 (PT)**

Status: **SPEC FILED; EXECUTION HOLD**. This file freezes a capability-grounded
deterministic program-search solver for the registered public-training ARC-AGI-2
universe. It admits no execution until runner tooling, npm wiring, result ignore
path, leak-check coverage, a smoke fingerprint, and a freeze-marker amendment are
committed together.

## Boundary Citation (why Branch E, and what it is NOT)

The Phase 3E certificate program closed across four binding receipts —
`phase3e_deferred_label_vacuity`, `phase3e_v2_deferred_sparse_fibers`,
`phase3e_v2_expanded_oracle_regression`, `phase3e_relative_locality_negative` —
on top of the seven full-grid-control floors (`full_grid_control_floor`,
`compact_full_grid_control_floor`, `branch_a_full_grid_floor`,
`branch_d_full_grid_edit_floor`, `branch_d_color_rule_full_grid_floor`,
`branch_d_mask_target_full_grid_floor`, and the V1 Blackwell floor).

Those results are cited here as a **boundary**, not as a license:

- **No-collision**: across the 36- and 108-task registers there were **0** exact
  or representation-level `signature_palette_context` fiber collisions, so no
  collision witness blocks a solver.
- **No-locality**: there is no usable fixed-radius fiber locality (0 near pairs)
  and no usable rank-local sketch coherence (relative-locality negative;
  `signature_palette` tied with metadata, below `raw_grid`, 28.5% hard-
  incompatible). So there is **nothing to motivate a smooth geometric selector**
  over `signature_palette`.

Therefore Branch E is a **deterministic program-search solver that selects
programs by train-pair consistency, never by `signature_palette` geometry**. It
proceeds on **capability grounds**: does composing the registered deterministic
transformation primitives, kept consistent with each task's training pairs,
solve held-out queries exactly? This spec does not retroactively change any
certificate or floor verdict, and a capability result here does not prove
Blackwell sufficiency.

## Core Question

Using only public-training demonstrations, can a deterministic typed search over
a frozen library of grid-transformation primitives — admitting only programs that
reproduce **all** of a task's conditioning train pairs — produce **exact** held-out
outputs on at least the established non-trivial floor (≥2 distinct held-out tasks
on both held-out lanes), where every prior decoder/learner family scored zero?

## Frozen Inputs

```text
primary_register   = docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv
primary_split      = sha256_expansion
diagnostic_register = docs/prereg/arc/P0_TASK_REGISTER.csv
diagnostic_split   = frozen_v2
```

Held-out adjudication universe (inherited from Phase 3E):

```text
U_primary = test_lodo union pttest          # gated
U_all     = validation_lodo union validation_pttest union test_lodo union pttest
```

- `test_lodo`: for each held-out **test** task, hold out one training pair as the
  query and condition on the remaining train pairs.
- `pttest`: for each held-out **test** task, condition on all train pairs and solve
  the public-training test query (output read only after the no-target barrier).
- Validation-lane instances are solved and reported **diagnostically**; the
  capability gate is adjudicated on `U_primary` only.

Training-split only. No public-evaluation grids are read (Phase 6 only). The
two-stage no-target fingerprint barrier is written before any pttest target output
is read, exactly as in the Phase 3E runners.

## Frozen Primitive Library

Each held-out query is solved by searching **programs** built from the families
below. All primitives are pure functions over raw grids (`list[list[int]]`); none
reads `signature_palette`, arm distances, or any certificate output. Parameters
that vary per task are **fit from the conditioning train pairs**, then verified.

**Stage-1 structural transforms** (`grid -> grid`):

1. `identity`
2. `d4` — the eight dihedral variants (identity, rot90, rot180, rot270, reflect_h,
   reflect_v, transpose, anti_transpose)
3. `tile` — repeat the grid `(ky, kx)` times (factors fit from train shape ratios)
4. `translate` — shift by `(dy, dx)` (fit from train)
5. `pad` — pad `(top, left, bottom, right)` with 0 (fit from train)
6. `crop_nonzero_bbox` — crop to the nonzero bounding box
7. `scale` — integer upscale / block-reduce downscale `(sy, sx)` (fit from train)
8. `palette_permute` — apply a color bijection (fit from train pairs)
9. `fill_enclosed` — flood-fill enclosed background regions
10. `extract_largest_component` — the largest connected nonzero component, cropped

**Color / edit families** (reused from Branch D, deterministic only):

11. `color_rule_bank` — the 10 Branch D color-rule families
    (`constant_edit_color`, `modal_edit_color`, `baseline_color_map`,
    `input_nn_color_map`, `input_patch_majority_map`, `baseline_to_input_pair_map`,
    `relative_palette_rank_map`, `object_role_color_map`, `row_col_periodic_color`,
    `nearest_edited_neighbor_color`), each fit on the conditioning recolorings and
    applied over a baseline + mask.
12. `mask_edit` — the **simple deterministic** Branch D mask families
    (`empty_mask`, `conditioning_mask_{union,intersection,majority}`,
    `conditioning_bbox_{fill,outline}`, `source_color_mask`,
    `delta_overlay_mask`), composed with a `color_rule_bank` rule to recolor the
    masked cells over a baseline. **Excluded from v1** (deferred to a noted
    Branch E v2, to keep the binding instrument deterministic and low-risk): the
    learned `legacy_mlp_threshold_mask` family (it is an MLP), and the intricate
    deterministic families `row_col_periodic_mask`, `source_color_pair_mask`,
    `object_role_mask`, `nearest_residual_patch_mask`, plus the
    `{dilate1, erode1, close1, bbox_fill}` morphology cross-product (v1 uses the
    `identity` morphology only). A v2 amendment may admit these once v1 has a
    binding receipt.

**Stage-2 candidate-combinator candidates** (reused, deterministic):

13. `output_copy` — output equals input.
14. `delta_overlay` — same-shape residual overlay fit from a conditioning pair.
15. `bijective_color_map` — same-shape color bijection from a conditioning pair.

This family list is frozen. Adding, removing, or reparameterizing a family after
seeing held-out outputs requires a new append-only amendment.

## Search Contract

```text
max_composition_depth = 2
program = stage1_structural [ then (stage1_structural | color_edit) ]
        | color_edit
        | candidate_combinator_candidate
candidate_budget_per_instance = 5000      # admitted-program evaluations cap
attempts = 2                              # top-2 outputs scored (ARC 2-attempt)
```

- **Fit-then-verify**: each family fits its parameters from the conditioning train
  pairs, then a program is **admitted only if it reproduces EVERY conditioning
  train output exactly** (train-consistency = the sole selection rule). Programs
  are applied to the query input only after admission.
- **Composition** is bounded to depth ≤ 2 with the grammar above; the candidate
  budget caps evaluations with a deterministic family-then-parameter enumeration
  order (if the cap is hit, the truncation is logged, never silently dropped).
- **Ranking** among admitted programs: (1) frozen family priority (structural
  before color/edit before combinator; simpler families first), (2) program
  description length (fewer ops, smaller parameters), (3) lexicographic program id.
  The top-2 **distinct** candidate grids are the attempts.
- **No stochasticity**: the search is fully deterministic; re-runs are byte-
  identical. No seeds, no learned weights, no signature distances.

## Scoring

```text
grid_exact_slot1(query) = (attempt_1 == target)
grid_exact_any(query)   = (attempt_1 == target) or (attempt_2 == target)
```

Per lane, a **task is solved** if at least one of its held-out instances has
`grid_exact_any = true`. Report per-lane: instance count, task count, solved-task
count, exact-instance rate; plus a per-prior breakdown and the family that produced
each winning program.

## Capability Gate (Branches)

Branch precedence is table order. The threshold mirrors the established
non-trivial floor (`PHASE3_SUFFICIENCY_SPEC.md`).

| branch | condition | interpretation |
| --- | --- | --- |
| `branch_e_capability_demonstrated` | ≥ 2 **distinct** held-out tasks solved exactly (top-2) on **both** `test_lodo` **and** `pttest`. | The deterministic program search clears the non-trivial floor every prior learner family floored on — a capability result, not a sufficiency or solve claim. |
| `branch_e_capability_partial` | At least one held-out task solved exactly on at least one lane, but the demonstrated threshold is not met on both lanes. | Non-zero capability signal below the established floor; reported, not a floor-clearing result. |
| `branch_e_capability_floor` | Zero held-out tasks solved exactly on `U_primary`. | The deterministic primitive search floors like the prior learner families; capability is not demonstrated. |

The capability gate is adjudicated on `U_primary`. Validation-lane results are
diagnostic. No threshold, family, budget, depth, or ranking rule may be retuned
after seeing held-out outputs.

## Required Artifacts

Binding output path: `results/arc/phase3-branch-e-program-search/`

- `manifest.json`
- `split.csv`
- `context_fingerprints_no_targets.jsonl` + `context_fingerprints_no_targets.sha256`
- `programs_by_instance.jsonl` — admitted programs + top-2 attempts per instance
- `solutions_by_instance.csv` — lane, task_id, prior, n_admitted, exact_slot1,
  exact_any, winning_family
- `capability_summary.csv` — per lane: n_instances, n_tasks, n_tasks_solved,
  exact_instance_rate
- `per_prior_capability.csv`
- `family_usage.csv` — admitted/winning counts per family
- `phase3_branch_e_program_search_receipt.json`
- `branch_adjudication.md`
- `commands.md`
- `hashes.json`

The manifest must record this spec hash, parent/boundary spec hashes, register
hash, split mode, target-output barrier hash, the frozen family list, the search
budget/depth/attempts, the branch, and source-code hashes.

## Reserved Implementation Names

- Python runner: `docs/prereg/arc/phase3_branch_e_program_search.py`
- Node wrapper: `scripts/arc-phase3-branch-e-program-search.mjs`
- npm script: `arc:phase3:branch-e-program-search`
- receipt path: `results/arc/phase3-branch-e-program-search/`

The freeze-marker amendment must add the runner, wrapper, npm wiring, result
ignore path, leak-check coverage, a smoke fingerprint, and the exact binding
command before execution is admitted. If the binding run exceeds the repo's
ten-minute rule it runs as a background job pinned to the freeze-marker commit.

## Public Language

Allowed before a binding receipt:

> "Branch E has filed a deterministic program-search solver spec. It selects
> programs by train-pair consistency over a frozen primitive library, cites the
> Phase 3E certificates as a no-collision / no-locality boundary, and is gated on
> the established non-trivial exact-match floor. No receipt exists yet."

Allowed if `branch_e_capability_demonstrated`:

> "A deterministic program search over registered transformation primitives
> cleared the established non-trivial exact-match floor on both held-out
> public-training lanes — a capability result. This does not prove Blackwell
> sufficiency, is not an ARC solve, and makes no claim about the public-evaluation
> split or a Kaggle entry."

Allowed if `branch_e_capability_floor`:

> "The deterministic program search floored like the prior learner families: it
> did not clear the non-trivial exact-match floor on the held-out public-training
> lanes."

Forbidden:

- "Sundog solves ARC" or any public-evaluation / Kaggle claim (Phase 6 only);
- claiming a capability result proves Blackwell sufficiency or the 5D subspace;
- claiming the certificate verdicts are changed by this lane;
- selecting programs by `signature_palette` geometry, or reading exact outputs to
  fit/select programs (train-consistency only);
- retuning the family list, budget, depth, ranking, or floor after seeing outputs.
