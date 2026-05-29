# Phase 3E -- Program-Sketch Oracle v2

Parent spec: [`PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md`](PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md)

Filed: **2026-05-29 (PT)**

Status: **SPEC FILED; EXECUTION HOLD**. This file freezes a framing-agnostic
program-sketch oracle for a future Phase 3E certificate rerun. It admits no run
until runner tooling, npm wiring, result ignore path, leak-check coverage, and a
freeze-marker amendment are committed together.

## Purpose

The Phase 3E binding receipt returned `phase3e_deferred_label_vacuity` with a
dual finding:

- no registered exact or near `signature_palette` context-fiber collision was
  found;
- the inherited `program_sketch_v1` oracle was prior-blind on 68% of primary
  contexts, because it was built from the Branch D baseline + mask + color edit
  framing.

`program_sketch_v2` is the narrow repair for the second fact. It labels the
kind of transformation required by a context across all six registered Phase 0
priors without emitting a target grid or choosing a solver.

The goal is to make a future fiber-certificate result adjudicable. It is not to
solve ARC.

## Core Question

Can a deterministic, framing-agnostic oracle label registered ARC contexts with
non-vacuous transformation sketches across all six registered priors while
remaining too coarse to reconstruct outputs?

If yes, rerun the Phase 3E signature-fiber certificate with
`program_sketch_v2` labels and the same frozen context geometry thresholds.

## Scope And Non-Goals

This is a certificate-labeling spec, not a decoder spec and not a Branch E
solver spec.

Allowed:

- computing labels after the target-output barrier;
- using raw registered public-training input/output grids to classify
  input-output relations;
- emitting finite label sets that describe transformation type, relation type,
  and rule scope.

Forbidden:

- emitting candidate output grids;
- choosing between generated outputs;
- using `signature_palette` to construct labels;
- using ARC public-evaluation tasks, private outputs, Kaggle notebooks, or
  non-registered public-training tasks;
- including raw target hashes, exact target pixel coordinates, serialized
  output grids, or per-cell masks in the sketch.

The oracle may read target outputs only after the Phase 3E no-target context
fingerprint barrier is written. Labels are used only for certificate
adjudication, not for training or selecting a solver.

## Context Universe

The context universe is inherited from Phase 3E:

```text
U_primary = test_lodo union pttest
U_all = validation_lodo union validation_pttest union test_lodo union pttest
```

The primary branch decision uses `U_primary`. `U_all` is diagnostic and must be
reported.

## Oracle Contract

`program_sketch_v2` is a deterministic function:

```text
program_sketch_v2(D, X_q, Y_q) -> finite label record
```

where:

- `D` is the conditioning demonstration set;
- `X_q` is the query input;
- `Y_q` is the held-out target output, available only after the target barrier.

The oracle must be representation-neutral:

```text
program_sketch_v2` may inspect raw grids, but not `signature_palette`,
`signature_only`, arm distances, or learned decoder outputs.
```

The oracle must be order-invariant over conditioning demonstrations.

## Sketch Facets

Each context receives a JSON object with these required facets. Each facet is a
set of labels. A facet may contain `none` only when the relation is explicitly
absent, and may contain `unknown` only when the deterministic tests cannot
classify it.

1. `shape_relation`
   - `shape_preserved`
   - `shape_transposed`
   - `shape_cropped`
   - `shape_padded`
   - `shape_scaled_or_tiled`
   - `shape_extracted_from_object`
   - `shape_constructed_from_count`
   - `shape_other`
2. `palette_relation`
   - `palette_preserved`
   - `palette_permuted`
   - `palette_role_recolored`
   - `palette_color_added`
   - `palette_color_removed`
   - `palette_background_changed`
   - `palette_other`
3. `object_relation`
   - `object_identity_preserved`
   - `object_selected`
   - `object_copied`
   - `object_split`
   - `object_merged`
   - `object_completed`
   - `object_removed`
   - `object_created`
   - `object_marker_guided`
   - `object_other`
4. `cardinality_relation`
   - `count_objects`
   - `count_colors`
   - `count_cells`
   - `count_components`
   - `compare_cardinalities`
   - `cardinality_controls_shape`
   - `cardinality_controls_palette`
   - `cardinality_controls_repetition`
   - `cardinality_other`
5. `completion_relation`
   - `fill_holes`
   - `complete_line_or_ray`
   - `complete_rectangle_or_box`
   - `complete_repeating_pattern`
   - `complete_symmetry`
   - `complete_occluded_object`
   - `complete_global_template`
   - `completion_other`
6. `spatial_transform_relation`
   - `translate`
   - `rotate`
   - `reflect`
   - `crop`
   - `pad`
   - `scale`
   - `tile`
   - `rearrange_objects`
   - `coordinate_map`
   - `spatial_other`
7. `symmetry_relation`
   - `reflect_horizontal`
   - `reflect_vertical`
   - `reflect_diagonal`
   - `rotational_symmetry`
   - `periodic_symmetry`
   - `symmetry_completion`
   - `symmetry_breaking`
   - `symmetry_axis_inferred`
   - `symmetry_other`
8. `correspondence_basis`
   - `by_color`
   - `by_shape`
   - `by_size`
   - `by_position`
   - `by_count_order`
   - `by_symmetry_axis`
   - `by_marker`
   - `by_adjacency`
   - `by_containment`
   - `correspondence_other`
9. `rule_scope`
   - `local_pixel`
   - `local_patch`
   - `object_level`
   - `multi_object_relation`
   - `global_count`
   - `global_symmetry`
   - `global_pattern`
   - `shape_level`
   - `scope_other`

Optional diagnostics may include coarse binned quantities such as
`edit_mass_bin`, `object_count_bin`, or `shape_delta_bin`, but exact counts,
coordinates, and target-pixel sets are forbidden.

## Deterministic Label Tests

The runner may use deterministic relation tests, including:

- shape equality and shape delta classification;
- palette set comparison and fitted color-map existence;
- connected-component comparison under color-agnostic and color-aware matching;
- D4 transform checks and translation/crop/pad/tile checks;
- row, column, diagonal, and periodic symmetry checks;
- line, rectangle, hole-fill, and repeating-pattern completion checks;
- cardinality influence checks, where changing a counted quantity across
  demonstrations correlates with output shape, repetition count, palette choice,
  or object count;
- object correspondence checks by color, size, shape, position, containment,
  adjacency, and marker relation.

These tests may classify relations. They may not produce an executable program
or reconstruct `Y_q`.

## Anti-Vacuity Gate

A context is `sketch_v2_vacuous` when fewer than four of the nine required
facets contain a non-`none`, non-`unknown` label.

The oracle clears anti-vacuity only if all are true on `U_primary`:

- overall `sketch_v2_vacuous_fraction <= 0.20`;
- for each primary prior, `sketch_v2_vacuous_fraction <= 0.25`;
- every primary prior has at least one non-vacuous context in both the
  `test_lodo` lane and the `pttest` lane when that lane has a context for that
  prior.

With four primary contexts for most priors, the per-prior threshold means at
most one vacuous primary context per prior.

## Anti-Prior-Laundering Gate

The sketch may not include `primary_prior` or any literal task-prior name as a
facet value.

A non-vacuous context must contain at least two concrete facet labels outside
the facet most directly associated with its registered prior. Examples:

- a counting context cannot be supported only by `cardinality_relation`;
- a symmetry context cannot be supported only by `symmetry_relation`;
- a spatial-transform context cannot be supported only by
  `spatial_transform_relation`;
- a local-completion context cannot be supported only by `completion_relation`.

The runner must emit a `prior_laundering_audit.csv` with:

- per-context count of concrete labels outside the prior-associated facet;
- per-prior sketch diversity;
- cross-prior reuse rate for each high-level facet label.

The oracle fails with `phase3e_oracle_v2_prior_laundering` if more than 10% of
non-vacuous `U_primary` contexts violate the two-extra-facet rule.

## Anti-Solver-Leakage Gate

The sketch must be too coarse to reconstruct an output.

Syntactic leakage failures:

- any target hash, raw output serialization, exact target coordinate set, exact
  edit mask, or exact component coordinate list appears in the sketch;
- any label contains an unbounded integer copied from the target, except for
  coarse bins declared before execution;
- any per-cell assignment is emitted.

Statistical leakage failures on `U_primary`:

- `unique_core_sketch_fraction > 0.60`, where `core_sketch` is the sorted tuple
  of the nine required facets after dropping diagnostics;
- a nearest-neighbor exact-output lookup keyed only by `core_sketch` would solve
  more than 20% of primary contexts exactly.

If any syntactic leakage occurs, or either statistical leakage threshold is
exceeded, the verdict is `phase3e_oracle_v2_solver_leakage`.

## Fiber-Certificate Rerun

If the oracle clears anti-vacuity, anti-prior-laundering, and
anti-solver-leakage gates, the runner reruns the Phase 3E near-fiber
incompatibility adjudication with `program_sketch_v2`.

The context representation and geometry remain frozen:

- `epsilon_primary = 0.05`;
- `epsilon_strict = 0.025`;
- `epsilon_loose = 0.10`;
- `k = 3` cross-task neighbors;
- same `signature_palette_context` identity and distance definitions.

No radius, kNN, or distance retuning is admitted by this oracle spec.

## v2 Incompatibility Rule

A near-fiber pair is `program_sketch_v2_incompatible` when:

1. `d_context_signature_palette <= epsilon_primary`;
2. task IDs differ;
3. both contexts are non-vacuous under `program_sketch_v2`;
4. at least four of the nine required facets are disjoint after dropping
   `none` and `unknown`; or
5. at least one hard-incompatibility pair fires:
   - `rule_scope` contains `global_count` for one context and
     `global_symmetry` for the other, with no shared non-scope facet;
   - one context has `shape_constructed_from_count` and the other has
     `shape_extracted_from_object`, with disjoint object/cardinality facets;
   - one context has `symmetry_completion` and the other has no symmetry facet
     besides `none` or `unknown`;
   - one context has `complete_global_template` and the other has only local or
     object-level scope labels.

Diagnostics must also report Jaccard distance over all required facets.

## Branches

Branch precedence is the table order below.

| branch | condition | interpretation |
| --- | --- | --- |
| `phase3e_oracle_v2_solver_leakage` | Syntactic or statistical anti-solver-leakage gate fails. | The oracle is too specific; it is behaving like a hidden solver or target lookup. |
| `phase3e_oracle_v2_prior_laundering` | Anti-prior-laundering gate fails. | The oracle is mostly restating the registered prior labels rather than classifying transformations. |
| `phase3e_oracle_v2_vacuous` | Anti-vacuity gate fails. | The oracle still cannot label the registered priors well enough for certificate adjudication. |
| `phase3e_v2_exact_fiber_collision` | Phase 3E exact collision exists under unchanged context identity. | Decoder-independent insufficiency witness. This is expected to remain false given the v1 receipt, but must be rechecked. |
| `phase3e_v2_near_fiber_incompatibility` | A cross-task near-fiber pair at `epsilon_primary` is incompatible under `program_sketch_v2`. | Strong locality obstruction for a smooth Branch E selector. |
| `phase3e_v2_fiber_locality_positive` | No exact or near incompatibility, at least 50% of primary contexts have fidelity-passing cross-task neighborhoods, and mean local incompatibility rate is at most 0.10. | Labels are adjudicable and locally consistent; licenses a later Branch E program-selector runner. Does not prove sufficiency. |
| `phase3e_v2_deferred_sparse_fibers` | Oracle gates pass, but fewer than 50% of primary contexts have fidelity-passing cross-task neighborhoods. | Labels are adequate, but the registered context universe is still too sparse at the frozen radius. |

## Required Artifacts

Binding output path:

`results/arc/phase3e-program-sketch-oracle-v2/`

Required files:

- `manifest.json`;
- `split.csv`;
- `context_fingerprints_no_targets.jsonl`;
- `context_fingerprints_no_targets.sha256`;
- `program_sketch_v2.jsonl`;
- `program_sketch_v2_summary.csv`;
- `prior_laundering_audit.csv`;
- `solver_leakage_audit.csv`;
- `oracle_coverage_by_prior.csv`;
- `pairwise_top_neighbors_v2.csv`;
- `near_fiber_incompatibilities_v2.csv`;
- `knn_fiber_locality_v2.csv`;
- `phase3e_program_sketch_oracle_v2_receipt.json`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

The manifest must record this spec hash, parent spec hash, register hash,
feature schema, target-output barrier hash, oracle facet list, all thresholds,
branch, and source code hashes.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- Python runner: `docs/prereg/arc/phase3e_program_sketch_oracle_v2.py`;
- Node wrapper: `scripts/arc-phase3e-program-sketch-oracle-v2.mjs`;
- npm script: `arc:phase3e:program-sketch-oracle-v2`;
- receipt path: `results/arc/phase3e-program-sketch-oracle-v2/`.

The freeze-marker amendment must add the runner, wrapper, npm script, result
ignore path, leak-check coverage, and a smoke fingerprint before execution is
admitted.

Per the repo's ten-minute rule, the first implementation receipt must be a
capped probe unless the full oracle pass is measured under about ten minutes.
If the full run is longer, stage exact PowerShell commands, wall-clock estimate,
resume-safety notes, and branch-decision consequences.

## Public Language

Allowed before a binding receipt:

> "Phase 3E has filed a framing-agnostic program-sketch oracle v2 spec. It is a
> labeler for certificate adjudication, not a solver. No v2 receipt exists yet."

Allowed if `phase3e_oracle_v2_vacuous` is filed:

> "The framing-agnostic program-sketch oracle v2 remained too vacuous to
> adjudicate the registered Phase 3E fiber certificate."

Allowed if `phase3e_oracle_v2_prior_laundering` is filed:

> "The program-sketch oracle v2 failed the anti-prior-laundering gate: it did
> not add enough transformation information beyond the registered prior labels."

Allowed if `phase3e_oracle_v2_solver_leakage` is filed:

> "The program-sketch oracle v2 was too specific to serve as a certificate
> labeler; it risked acting as a hidden solver or target lookup."

Allowed if `phase3e_v2_deferred_sparse_fibers` is filed:

> "The program-sketch oracle v2 cleared coverage and leakage gates, but the
> registered context universe remained too sparse at the frozen signature radius
> to certify fiber locality."

Allowed if `phase3e_v2_fiber_locality_positive` is filed:

> "With a framing-agnostic program-sketch oracle, Phase 3E found no registered
> exact or near signature-fiber incompatibility and enough local neighborhoods
> to license a later Branch E selector test. This does not prove signature
> sufficiency."

Forbidden:

- any claim about ARC public-evaluation or Kaggle performance;
- any claim that a v2 oracle result solves ARC;
- any claim that labels alone prove signature sufficiency;
- using `program_sketch_v2` labels to train or select a solver without a new
  Branch E solver spec;
- retuning `epsilon_primary`, kNN `k`, oracle facets, coverage thresholds,
  laundering thresholds, or leakage thresholds after inspecting v2 output.
