# Phase 3E -- Signature Fiber Certificate

Parent spec: [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md)

Prior Phase 3 structured-edit lane:
[`PHASE3D_MASK_TARGET_VARIANT_SPEC.md`](PHASE3D_MASK_TARGET_VARIANT_SPEC.md)

Filed: **2026-05-28 (PT)**

Status: **SPEC FILED; EXECUTION HOLD**. This file starts Phase 3E as a
certificate lane, not another decoder lane. It admits no run until runner
tooling, npm wiring, result ignore path, leak-check coverage, and a
freeze-marker amendment are committed together.

## Purpose

Seven Phase 3 full-grid controls have floored across direct decoders,
per-task learners, and structured-edit variants. The structured-edit framing
has now been probed at both named bottlenecks: deterministic color-rule bank
and deterministic mask-candidate bank both failed to open the non-baseline
arena.

Phase 3E asks a more surgical question before building a generative program
searcher:

> Do there exist registered ARC contexts whose frozen `signature_palette`
> contexts are identical, or very near, while the required output behavior is
> incompatible?

If yes, the signature has a decoder-independent collision witness in the
registered task class. If no, and nearby signature contexts have locally
consistent action/program sketches, a Branch E program-selector runner becomes
better motivated.

## Core Question

Does the frozen `signature_palette` representation induce fibers over the
registered ARC demonstration contexts that are consistent with the required
held-out output behavior?

Equivalently, for registered contexts `c_i` and `c_j`, does:

```text
sigma_signature_palette(c_i) == sigma_signature_palette(c_j)
```

ever hold while:

```text
target(c_i) != target(c_j)
```

or, for near fibers:

```text
d_signature_palette_context(c_i, c_j) <= epsilon_primary
```

while their registered program-sketch labels are incompatible?

## Scope And Non-Goals

This is Branch E because it changes the framing from "train a decoder and see
if raw grid opens the arena" to "certify the geometry of the representation
fibers before choosing a solver." It does not train a grid decoder, does not
generate ARC submissions, and does not inspect ARC evaluation tasks.

This spec uses only the registered 36 public-training tasks from Phase 0. Public
training test outputs may be read only after the no-target context fingerprint
barrier described below is written.

This spec does not claim that absence of a collision proves sufficiency. It can
only certify a found collision, or certify that this registered finite search
did not find one under the frozen distance and label rules.

## Context Universe

A registered context is one held-out query instance from the Phase 3 split:

- `validation_lodo`: each validation task training pair held out once;
- `validation_pttest`: each validation task public-training test query;
- `test_lodo`: each test task training pair held out once;
- `pttest`: each test task public-training test query.

The primary adjudication universe is:

```text
U_primary = test_lodo union pttest
```

The full diagnostic universe is:

```text
U_all = validation_lodo union validation_pttest union test_lodo union pttest
```

LODO contexts may read the held-out public-training output because it is one of
the registered training pairs. Public-training test contexts must use a
two-stage barrier:

1. write `context_fingerprints_no_targets.jsonl` and its hash before reading
   public-training test outputs;
2. only then read public-training test outputs to compute target labels and
   branch adjudication.

No ARC public-evaluation tasks, private outputs, Kaggle notebooks, or
non-registered public-training tasks are admitted.

## Frozen Representations

Feature schema: `arc-p3-feature-v1`.

The runner must reuse the frozen Pass A grid encoders:

- `signature_only`: canonical object signature plus local signature bag;
- `signature_palette`: `signature_only` plus shape, palette, nonzero cell,
  component, and density metadata;
- `metadata_only`: the Pass A metadata prefix only;
- `raw_grid_context`: raw grid identity, diagnostic only.

The primary arm is `signature_palette_context`. The strict quotient diagnostic
arm is `signature_only_context`. The nuisance control is
`metadata_only_context`.

## Context Identity

For each grid `G`, define:

- `id_signature_only(G) = signatureHash|localBagHash`;
- `id_signature_palette(G) =
  shape|palette|nonzeroCells|nonzeroComponents|density|signatureHash|localBagHash`;
- `id_metadata_only(G) = JSON(metadata_vector(G))`;
- `id_raw_grid(G) = JSON(G)`.

For each conditioning pair `(X_k, Y_k)`, define:

```text
pair_id_a(k) = id_a(X_k) + "=>" + id_a(Y_k)
```

For a context `c = (D, X_q)`, define the exact context identity:

```text
context_id_a(c) =
  id_a(X_q) + "||" + sorted(pair_id_a(1),...,pair_id_a(k))
```

The sort makes conditioning demonstration order irrelevant. Exact-fiber
collisions use byte equality of `context_id_signature_palette`.

## Context Distance

For individual grids, reuse the Phase 3 arm distances:

- `signature_only`: cosine distance over the normalized hashed signature
  suffix;
- `signature_palette`: `0.5 * signature_cosine_distance + 0.5 * metadata_l1`;
- `metadata_only`: metadata L1;
- `raw_grid_context`: normalized Hamming over padded 30 x 30 color labels.

For conditioning pairs:

```text
d_pair_a((X,Y),(X',Y')) = 0.5 * d_a(X,X') + 0.5 * d_a(Y,Y')
```

For two contexts, match conditioning pairs by minimum-cost bipartite matching.
Unmatched pairs receive distance `1.0`. If both contexts have no conditioning
pairs, the conditioning distance is `0.0`; otherwise it is normalized by the
larger conditioning-pair count.

The context distance is:

```text
d_context_a(c,c') =
  0.4 * d_a(X_q, X'_q) + 0.6 * d_conditioning_pairs_a(D,D')
```

The primary near-fiber threshold is:

```text
epsilon_primary = 0.05
```

Diagnostics must also report thresholds:

```text
epsilon_exact = 0.0
epsilon_strict = 0.025
epsilon_loose = 0.10
```

Primary kNN locality excludes same-task neighbors. Same-task neighbors are
reported separately because they can make the fiber look artificially coherent
by reusing the same latent task rule.

## Target Labels

The runner must compute three target labels.

1. `target_exact_hash`: canonical JSON hash of the held-out target output grid.
2. `target_signature_palette_id`: `id_signature_palette(Y_q)`.
3. `program_sketch_v1`: a deterministic finite sketch of the required behavior.

`program_sketch_v1` is not a solver. It is an oracle audit label computed after
the target is available, using only registered candidate families from Branch D:

- `shape_rule_oracle_set`: inherited Branch D shape rules whose output shape
  matches every observed conditioning output and the held-out target;
- `canvas_rule_oracle_set`: inherited Branch D canvas rules with mean residual
  mass at most `0.25` across conditioning pairs plus target;
- `mask_candidate_oracle_family_set`: Phase 3D mask-target candidate families
  that reach mask F1 at least `0.75` on the target residual and mean
  conditioning mask F1 at least `0.50`;
- `color_rule_oracle_family_set`: Phase 3D color-rule families that reach edit
  color accuracy at least `0.75` on the target residual and mean conditioning
  edit-color accuracy at least `0.50`;
- `behavior_invariants`: output shape, shape delta, output palette, palette
  additions/removals, target edit mass relative to the selected Branch D
  baseline, connected-component count delta, and registered Phase 0 prior.

If a candidate family is not applicable because there are no conditioning edits,
the corresponding set contains `no_conditioning_edits`. If no family clears a
threshold, the set contains `none`.

## Incompatibility Rules

An exact-fiber pair is an **exact output collision** when:

```text
context_id_signature_palette(c_i) == context_id_signature_palette(c_j)
and target_exact_hash(c_i) != target_exact_hash(c_j)
```

It is a **representation-level collision** when exact context IDs match and:

```text
target_signature_palette_id(c_i) != target_signature_palette_id(c_j)
```

A near-fiber pair is a **program-sketch incompatibility** when all are true:

1. `d_context_signature_palette(c_i,c_j) <= epsilon_primary`;
2. `task_id(c_i) != task_id(c_j)`;
3. at least two of the following four components are disjoint:
   - `shape_rule_oracle_set`;
   - `canvas_rule_oracle_set`;
   - `mask_candidate_oracle_family_set`;
   - `color_rule_oracle_family_set`;
4. at least one behavior invariant differs outside the allowed identity fields
   (`task_id`, query index, and lane).

For diagnostics, the runner must also compute the Jaccard distance between the
full `program_sketch_v1` sets. A Jaccard distance of at least `0.80` is reported
as `strong_program_sketch_gap`, but the branch decision uses the four-component
disjointness rule above.

## kNN Fiber Locality

For each context in `U_primary`, find the `k = 3` nearest cross-task neighbors
under `d_context_signature_palette`.

A neighbor is fidelity-passing when:

```text
d_context_signature_palette <= epsilon_primary
```

For each fidelity-passing neighborhood, compute:

- `local_incompatibility_rate`: fraction of neighbors incompatible by the
  near-fiber rule;
- `r_k`: distance to the third cross-task neighbor;
- `same_prior_neighbor_rate`;
- `same_program_sketch_component_rate` for each of the four sketch components.

This mirrors the Navier-Stokes kNN fiber-locality discipline: if exact fibers
are sparse, report local fiber fidelity as a radius rather than silently turning
sparse neighborhoods into evidence.

## Branches

| branch | condition | interpretation |
| --- | --- | --- |
| `phase3e_exact_fiber_collision` | At least one `U_primary` exact output collision or representation-level collision under `signature_palette_context`. | Decoder-independent insufficiency witness for exact or representation-level signature sufficiency on the registered context class. |
| `phase3e_near_fiber_incompatibility` | No exact collision, but at least one `U_primary` cross-task near-fiber pair satisfies the program-sketch incompatibility rule at `epsilon_primary`. | Not a formal Blackwell proof, but strong evidence that a smooth/local signature selector will face incompatible nearby actions; Branch E must address this explicitly. |
| `phase3e_fiber_locality_positive` | No exact collision or near-fiber incompatibility in `U_primary`, at least `50%` of `U_primary` contexts have fidelity-passing cross-task `k=3` neighborhoods, and mean local incompatibility rate is at most `0.10`. | No registered collision found; licenses a later Branch E program-selector runner. Does not prove sufficiency. |
| `phase3e_deferred_sparse_fibers` | No collision is found, but fewer than `50%` of `U_primary` contexts have fidelity-passing cross-task neighborhoods. | The registered set is too sparse at `epsilon_primary` to adjudicate locality. |
| `phase3e_deferred_label_vacuity` | More than `30%` of `U_primary` contexts have `none` in at least two program-sketch oracle sets. | The program sketch is too weak or vacuous to adjudicate near-fiber incompatibility. |

Branch precedence is the table order above. Exact collisions dominate all other
branches. `phase3e_deferred_label_vacuity` takes precedence over
`phase3e_fiber_locality_positive`, but not over a found collision.

## Required Artifacts

Binding output path:

`results/arc/phase3e-signature-fiber-certificate/`

Required files:

- `manifest.json`;
- `split.csv`;
- `context_fingerprints_no_targets.jsonl`;
- `context_fingerprints_no_targets.sha256`;
- `target_labels.csv`;
- `program_sketches.jsonl`;
- `exact_fibers.csv`;
- `pairwise_top_neighbors.csv`;
- `near_fiber_incompatibilities.csv`;
- `knn_fiber_locality.csv`;
- `per_prior.csv`;
- `phase3e_signature_fiber_certificate_receipt.json`;
- `branch_adjudication.md`;
- `commands.md`;
- `hashes.json`.

The manifest must record this spec hash, parent spec hash, register hash,
feature schema, context universe counts, target-output barrier hash, distance
thresholds, label thresholds, branch, and any source code hashes.

## Reserved Implementation Names

These names are reserved but not executable by this spec alone:

- Python runner: `docs/prereg/arc/phase3e_signature_fiber_certificate.py`;
- Node wrapper: `scripts/arc-phase3e-signature-fiber-certificate.mjs`;
- npm script: `arc:phase3e:signature-fiber-certificate`;
- receipt path: `results/arc/phase3e-signature-fiber-certificate/`.

The freeze-marker amendment must add the runner, wrapper, npm script, result
ignore path, leak-check coverage, and a smoke fingerprint before execution is
admitted.

Per the repo's ten-minute rule, the first implementation receipt must be a
capped probe unless the full certificate search is measured under about ten
minutes. If the full run is longer, stage exact PowerShell commands, wall-clock
estimate, resume-safety notes, and branch-decision consequences.

## Public Language

Allowed before a binding receipt:

> "Phase 3E has filed a signature-fiber certificate spec. It will test whether
> registered ARC contexts contain exact or near `signature_palette` collisions
> with incompatible required behavior. No Phase 3E receipt exists yet."

Allowed if `phase3e_exact_fiber_collision` is filed:

> "Phase 3E found a registered `signature_palette` fiber collision: two
> registered contexts have the same signature context but incompatible target
> behavior. This is a finite-context insufficiency certificate, not an ARC solve
> claim."

Allowed if `phase3e_near_fiber_incompatibility` is filed:

> "Phase 3E found near-signature registered contexts with incompatible
> program-sketch labels. This does not prove Blackwell insufficiency, but it
> names a locality obstruction that any Branch E program selector must handle."

Allowed if `phase3e_fiber_locality_positive` is filed:

> "Phase 3E found no registered exact or near signature-fiber collision under
> the frozen thresholds. This licenses a later Branch E program-selector test;
> it does not prove signature sufficiency."

Forbidden:

- any claim about ARC public-evaluation or Kaggle performance;
- any claim that absence of a collision proves `signature_palette` is
  sufficient;
- any claim that a near-fiber incompatibility is a formal Blackwell proof;
- any Branch E solver or ARC-AGI submission claim from this certificate alone;
- changing `epsilon_primary`, kNN `k`, or program-sketch thresholds after seeing
  pairwise distances or labels without a new append-only amendment.
