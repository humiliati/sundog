# Phase 3 -- Signature Sufficiency Audit Spec

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Phase 2 receipt: [`PHASE2_PROJECTION_SPEC.md`](PHASE2_PROJECTION_SPEC.md)

Filed: **2026-05-28 (PT)**

Status: **SPEC FILED -- DECODER TRAINING HOLD**.

Baseline-addendum inputs:

- `results/arc/phase2-baselines/manifest.json` SHA-256:
  `4968D6F6958C8D40643D09161407252B664665801FCBA3A399780ED7509623E0`
- `results/arc/phase2-baselines/summary.csv` SHA-256:
  `14977F7F35AB3E5DA40C4603BC0771AB626909C003F89E82F6210AC5B068F165`

Phase 2 projected all 36 registered public-training tasks and then added a
baseline comparison. The result changes the Phase 3 posture: the shadow
signature does **not** shorten the input-output gap. Its mean train-pair
residual (`0.594295`) is higher than raw-pixel Hamming (`0.278825`),
shape/palette/density (`0.102588`), and cell-count (`0.070576`). Phase 3
therefore cannot argue "input signature is close to output signature; learn a
small delta." It must test whether the task-conditioned mapping from input
representation to output representation is learnable from few demonstrations
despite high signature-space distance.

## Claim Under Test

The Phase 3 claim is:

> For the registered ARC subset, a task-conditioned learner using the Phase 2
> shadow representation can infer the input-to-output mapping from
> demonstrations at least as well as matched low-capacity full-grid and
> metadata controls.

The working representation is **not signature alone**. Phase 1 and Phase 2
carried forward a color-identity caveat: the canonical signature quotients
bijective color roles, so any sufficiency claim that depends on absolute color
must use `(signature, palette)` or explicitly quarantine color-dependent tasks.

## Representation Arms

Phase 3 must compare all arms below before claiming support:

| arm | contents | role |
| --- | --- | --- |
| `signature_only` | canonical object signature + local signature bag | strictest sufficiency test; expected to fail on color-identity tasks |
| `signature_palette` | `signature_only` plus raw palette and shape metadata | primary Sundog representation after Phase 1 caveat |
| `metadata_only` | shape, palette size/list, density, component count | detects whether coarse metadata explains the result |
| `raw_grid_lowcap` | raw grids under the same low-capacity learner family | matched full-state control |
| `oracle_copy_floor` | nearest/copy/DSL floors already filed in Phase 0 | floor only; no learned claim |

No arm may read public-evaluation grids or Kaggle private/semi-private data.

## Evaluation Shape

Phase 3 has two lanes:

1. **LODO representation audit** -- leave-one-demonstration-out within each
   registered task. For a task with `k` train pairs, hold out one train pair;
   condition on the other `k-1` pairs; predict the held output representation
   from the held input representation.
2. **Public-training test audit** -- after LODO is frozen, evaluate on the
   public-training test input/output pairs already allowed by Phase 0. This is
   confirmatory inside the public training split, not a public-evaluation or
   Kaggle claim.

Primary metric: exact output-grid match where a grid decoder is used.

Secondary metrics:

- exact output-representation match;
- palette exact match;
- shape exact match;
- pixel accuracy;
- per-prior breakdown using `P0_TASK_REGISTER.csv`;
- residual/coverage/detection failure-mode label.

Representation-level success does not by itself support an ARC solve claim.
Grid exact match remains the primary external metric.

## Learner Constraint

The first learner family must be deliberately small:

- no pretrained model;
- no internet;
- no public-evaluation data;
- no task ID memorization beyond the in-context demonstrations for that task;
- no learned parameters fit across the 36 tasks unless the same training budget
  is given to every representation arm;
- no post-hoc task-specific operator edits after seeing Phase 3 outcomes.

Allowed first families:

- linear or nearest-neighbor representation mapper;
- shallow MLP with fixed architecture and seed slate;
- finite candidate selector over outputs generated from the demonstration
  output representations.

If a learner is expected to exceed the repo's ten-minute rule, stage the exact
PowerShell command with a wall-clock estimate instead of running it inline.

## Gates

Phase 3 can produce only the following verdicts:

| result | verdict |
| --- | --- |
| `signature_palette` beats `metadata_only` and is competitive with `raw_grid_lowcap` on LODO and public-training test audit | **support on registered subset** |
| `signature_palette` beats `metadata_only` but trails `raw_grid_lowcap` materially | **partial support / lossy representation** |
| `signature_only` fails but `signature_palette` works | **palette-dependent support**; update public language to include palette metadata |
| `signature_palette` does not beat `metadata_only` | **sufficiency failure**; quarantine signature sufficiency for discrete abstraction |
| every arm scores near zero exact | **task hardness / decoder failure**; no sufficiency conclusion |

No Phase 3 result may be described as "Sundog solves ARC" or as evidence on
the public evaluation split.

## Admission Rule

Decoder implementation is **held** until a follow-up amendment specifies:

1. exact feature serialization for every representation arm;
2. learner architecture and seed slate;
3. LODO and public-training test split construction;
4. success thresholds for "beats", "competitive", and "materially trails";
5. artifact paths and hashes to be filed after the run.

## Amendments

Append-only. Each amendment must carry a timestamp, author, justification, and
verdict impact.

### 2026-05-28 -- Decoder Admission Roadmap

Author: Codex, with operator-provided gap list.

Justification: the initial Phase 3 spec correctly holds decoder execution, but
the held items are broader than the first five admission bullets. This
amendment expands the admission checklist into a few-pass roadmap so Phase 3
cannot quietly choose representation encodings, split handling, thresholds, or
public wording after observing results.

Verdict impact: no Phase 3 execution is admitted by this amendment. Decoder
implementation remains held until all passes below are filed append-only, with
exact commands and artifact paths frozen before the first run.

#### Pass A -- Representation And Decoder Contract

File the exact feature serialization for every arm:

- `signature_only`: canonical signature and local signature bag serialization,
  including token vocabulary version, position/index rules, padding/truncation,
  sparse/dense format, unknown-token behavior, and hash/version fields.
- `signature_palette`: `signature_only` plus raw palette and shape metadata,
  with color ordering, absent-color slots, and normalization rules fixed.
- `metadata_only`: shape, palette, density, component count, and any other
  coarse metadata fields as an explicit vector schema.
- `raw_grid_lowcap`: grid tensor serialization, padding value, maximum grid
  envelope, mask handling, and whether the low-capacity raw-grid learner is a
  small CNN. If a CNN is chosen, this pass closes the missing Phase 2.5
  simple-convolution-feature baseline gap called out after Phase 2.

Also file decoder semantics before any exact-grid comparison:

- `signature_only` is structurally lossy because multiple grids can share one
  gauge-quotiented signature. It may be scored directly on output-representation
  exact match, but exact-grid scoring requires a deterministic orbit-selection
  rule frozen before the run. If no such rule is filed, `signature_only` is
  marked "representation-only" for the primary exact-grid metric rather than
  compared as though it emitted a unique grid.
- `signature_palette` must define whether palette metadata is enough to select
  a concrete grid from the predicted signature orbit, and must use the same
  deterministic orbit-selection rule whenever ambiguity remains.
- The pre-registered color-role quarantine set is, by default, the six
  `primary_prior=color_role` tasks in `P0_TASK_REGISTER.csv`: `08ed6ac7`,
  `0a2355a6`, `2601afb7`, `292dd178`, `37d3e8b2`, and `3ad05f52`. Any narrower
  quarantine set requires an append-only rule before execution.

#### Pass B -- Split, Floor, And Discrimination Contract

File deterministic instance construction:

- LODO instances are ordered by registered task order, then original train-pair
  index. For each task with `k` train pairs, hold out train pair `i` and
  condition on all other `k-1` train pairs.
- The five `k=2` tasks stay in the audit but are reported as a separate
  small-k stratum. Phase 3 verdicts must report both all-task and `k>=3`
  results, and no support verdict may depend only on the `k=2` stratum.
- Public-training test instances are run only after the LODO protocol is
  frozen. They use the public-training test inputs and outputs admitted by
  Phase 0 and remain confirmatory inside the public-training split.
- Every arm emits no more than two ordered candidate grids per instance, matching
  the ARC-AGI two-prediction discipline. Ranking and tie-breaking must be
  deterministic.

File floor handling:

- `oracle_copy_floor` must have two readings: the Phase 0 all-train-pair numbers
  as historical reference, and an apples-to-apples LODO rerun using only the
  `k-1` conditioning pairs. Only the LODO rerun participates in Phase 3 arm
  comparisons.
- A discrimination floor must be reported before arm scores are interpreted:
  unique held-out output representations, majority-output-representation rate,
  and per-task collapse counts. If held-out outputs collapse to one or two
  representations, the receipt must say that the learner task was weak for that
  slice.

#### Pass C -- Learner, Seeds, Metrics, And Failure Labels

File the first learner family and all hyperparameters:

- Choose exactly one first learner family before execution: nearest-neighbor or
  linear representation mapper, shallow MLP, finite candidate selector, or a
  pinned low-capacity raw-grid CNN if Pass A selects one for `raw_grid_lowcap`.
- Pin architecture, hidden sizes if any, optimizer if any, learning rate, batch
  shape, epoch/step cap, early-stop rule, loss, regularization, and tie-break
  order.
- Pin a master seed list. Any stochastic learner or tie-breaker must derive a
  distinct deterministic seed from `(master_seed, task_id, lane, heldout_index,
  arm, candidate_rank)` so cross-task seed reuse cannot bias a style globally.

File the metric hierarchy and thresholds:

- Exact-grid success is primary where an arm has a defined grid decoder.
- Exact output-representation match, palette exact match, shape exact match,
  pixel accuracy, and normalized representation distance are required secondary
  metrics.
- "Beats", "competitive", and "materially trails" must be numeric thresholds
  over the metric hierarchy before execution. Exact-rate `0` vs `0` ties cannot
  support any sufficiency claim; ties must fall through to representation exact,
  then pixel accuracy, then normalized representation distance, then declared
  inconclusive if still tied.

File per-instance failure labels:

- `coverage`: the target output grid or representation is outside the arm's
  declared decoder/candidate support.
- `detection`: the target is inside support, but the task-conditioned learner
  selects or ranks the wrong representation/candidate from the demonstrations.
- `residual`: the selected representation is correct or nearest under the
  declared metric, but deterministic decoding leaves grid, geometry, or color
  residual error.

#### Pass D -- Receipt, Command, Freeze, And Public Language

File receipt paths and hashes before execution. The default receipt directory is
`results/arc/phase3-sufficiency/`, with at least:

- `manifest.json`: command, git commit, data inputs, protocol version, seed
  slate, started/completed timestamps, platform, and dependency versions.
- `scores.csv`: one row per lane/arm/stratum with primary and secondary scores.
- `per_instance.csv`: one row per task/lane/heldout-or-test instance/arm with
  candidate count, selected rank, metrics, and failure label.
- `discrimination.csv`: held-out output-representation uniqueness and majority
  floors by task and stratum.
- `hashes.json`: SHA-256 hashes for all emitted receipt files, using uppercase
  hexadecimal strings.

File the exact reproducible command shape before execution. The expected command
slot is `npm run arc:phase3:sufficiency`, but the runner script and any flags
remain unadmitted until this pass pins them in `package.json` and in this spec.

Unify terminology:

- `learner`: the task-conditioned mapper or candidate ranker.
- `decoder`: deterministic conversion from a predicted representation or
  candidate to a concrete grid.
- `runner`: the complete Phase 3 protocol implementation that constructs
  instances, invokes arms, scores candidates, and writes receipts.

Freeze marker: the commit that files Passes A-D, adds the exact command, and
adds the runner is the decoder-freeze commit. After that commit, no post-hoc
operator edits to arm schemas, learner choices, thresholds, split handling, or
failure labels are allowed before the first receipt. Later bug fixes require a
new append-only amendment naming the defect, expected metric impact, and whether
the first receipt is void or superseded.

Public-language drafts are pre-registered as follows:

- Support on registered subset: "On a pre-registered public-training subset,
  the Sundog signature-plus-palette representation supported a low-capacity
  input-to-output audit at or near the matched raw-grid control. This is not an
  ARC solve claim and does not use the public evaluation split."
- Partial support / lossy representation: "The signature-plus-palette
  representation retained measurable task information beyond coarse metadata,
  but trailed the matched raw-grid control; the representation is useful but not
  sufficient for the registered subset."
- Palette-dependent support: "The ARC representation that works is
  signature-plus-palette. Canonical signature alone is insufficient on
  color-role tasks because it quotients absolute color identity."
- Sufficiency failure: "The registered Phase 3 audit did not show that the
  shadow signature representation is sufficient for ARC-style
  input-to-output transformations beyond coarse metadata controls."
- Task hardness / decoder failure: "The first Phase 3 runner did not produce an
  interpretable sufficiency verdict because all arms were near the exact-match
  floor; the result is a decoder or task-hardness finding, not evidence for
  signature sufficiency."

### 2026-05-28 -- Pass A Representation And Decoder Contract

Author: Codex.

Justification: Pass A must freeze the concrete representation objects and
learner-facing feature serialization before any Phase 3 learner, split runner,
or decoder implementation is admitted. The Phase 2 projection cache is useful
as a receipt, but it does not contain the full `localSignatureBag`; therefore
Phase 3 must recompute `P_shadow_grid_v0` from the allowed public-training grids
instead of treating `results/arc/phase2-projections/grid-projections.json` as
the sole feature source.

Verdict impact: Pass A is filed. Decoder/learner execution remains held pending
Passes B-D.

#### Schema Version And Scope

All Phase 3 representation artifacts filed under this contract use:

- `featureSchemaVersion`: `arc-p3-feature-v1`
- `operatorVersion`: `P_shadow_grid_v0`
- `localRadius`: `1`
- `maxGridHeight`: `30`
- `maxGridWidth`: `30`
- `arcColors`: integer colors `0..9`, with `0` as background

The registered-subset envelope measured before filing this amendment is:

| quantity | value |
| --- | ---: |
| registered tasks | `36` |
| train pairs | `115` |
| public-training test inputs | `36` |
| max grid height | `30` |
| max grid width | `30` |
| max grid cells | `900` |
| max non-zero cells | `559` |
| max distinct colors in a grid | `9` |

Feature serialization must be computed from the raw ARC training task JSON
allowed by Phase 0. It may read the Phase 2 projection receipt only for
cross-checking hashes and counts, not as the complete source of representation
fields.

#### Shared Shadow Projection Fields

For each grid, `P_shadow_grid_v0` emits the following structured fields before
arm-specific serialization:

- `shape`: `[height, width]`
- `palette`: sorted unique ARC colors present in the grid, including `0` when
  present
- `nonzeroPalette`: sorted unique non-zero ARC colors present in the grid
- `nonzeroCells`: count of cells whose color is not `0`
- `nonzeroComponents`: 4-connected non-zero component count
- `density`: `nonzeroCells / (height * width)`, rounded to 6 decimal places
- `canonicalObjectSignature`: either `empty` or
  `bboxWidthxbboxHeight|x:y:role;...`
- `localSignatureBag`: sorted list of role-normalized radius-1 stencil strings,
  one string per non-zero cell, each string read row-major after D4
  canonicalization of the stencil
- `signatureHash`: uppercase SHA-256 of `canonicalObjectSignature`
- `localBagHash`: uppercase SHA-256 of `JSON.stringify(localSignatureBag)`

The canonical object signature is the lexicographically smallest signature over
the eight D4 transforms after translating non-zero cells to the local
top-left corner and remapping non-zero colors to roles by first occurrence in
row-major order. This is the same quotient used in the Phase 1 and Phase 2
scripts: translation, D4 orientation, and bijective non-zero color relabeling
are removed from the canonical signature.

#### Learner-Feature Hashing

Signature-bearing arms use a deterministic sparse hashing suffix so the runner
does not learn a token vocabulary from held-out outputs.

Constants:

- `metadataDim`: `28`
- `signatureHashDim`: `4096`
- `signatureVectorDim`: `4124`

Hash bucket rule:

1. Build the UTF-8 string
   `arc-p3-feature-v1\0<namespace>\0<token>`.
2. Compute SHA-256.
3. Interpret the first four digest bytes as an unsigned big-endian 32-bit
   integer.
4. Bucket is `metadataDim + (uint32 % signatureHashDim)`.

Object-signature tokenization:

- `obj:empty` for an empty signature; otherwise:
- `obj:bbox_w=<bboxWidth>`
- `obj:bbox_h=<bboxHeight>`
- `obj:role_count=<roleCount>`
- `obj:cell_count=<nonZeroTokenCount>`
- one `obj:cell=<x>:<y>:<role>` token for each cell token in
  `canonicalObjectSignature`

Local-bag tokenization:

- one `bag:stencil=<stencil>` token per unique stencil value in
  `localSignatureBag`, with multiplicity preserved as its count

Token values:

- object-token contributions are `count / objectTokenCount`
- local-bag contributions are `count / max(1, localSignatureBag.length)`
- contributions landing in the same bucket are summed
- after accumulation, the hashed suffix is L2-normalized if its L2 norm is
  non-zero

Sparse vectors are serialized as sorted `[index, value]` pairs with `index`
ascending and `value` rounded to 9 decimal places.

#### Metadata Vector

The metadata prefix is a dense 28-dimensional vector, serialized in this exact
order with every value rounded to 9 decimal places:

| index | value |
| ---: | --- |
| `0` | `height / 30` |
| `1` | `width / 30` |
| `2` | `(height * width) / 900` |
| `3` | `palette.length / 10` |
| `4` | `nonzeroPalette.length / 9` |
| `5` | `nonzeroCells / 900` |
| `6` | `density` |
| `7` | `nonzeroComponents / 900` |
| `8..17` | palette mask for colors `0..9` |
| `18..27` | color histogram for colors `0..9`, each `count(color) / (height * width)` |

#### Arm Serialization

`signature_only`:

- Structured representation:
  - `canonicalObjectSignature`
  - `localSignatureBag`
  - `signatureHash`
  - `localBagHash`
- Learner vector:
  - length `4124`
  - indices `0..27` are all `0`
  - indices `28..4123` contain the sparse hashed signature suffix above
- It does not include raw palette, full grid shape, absolute translation, or
  original D4 orientation.

`signature_palette`:

- Structured representation:
  - all `signature_only` structured fields
  - `shape`
  - `palette`
  - `nonzeroPalette`
  - `nonzeroCells`
  - `nonzeroComponents`
  - `density`
- Learner vector:
  - length `4124`
  - indices `0..27` contain the metadata vector
  - indices `28..4123` contain the sparse hashed signature suffix

`metadata_only`:

- Structured representation:
  - `shape`
  - `palette`
  - `nonzeroPalette`
  - `nonzeroCells`
  - `nonzeroComponents`
  - `density`
  - color histogram for colors `0..9`
- Learner vector:
  - length `4124`
  - indices `0..27` contain the metadata vector
  - indices `28..4123` are all `0`

`raw_grid_lowcap`:

- Structured representation:
  - original integer grid
  - `shape`
- Learner tensor:
  - shape `[30, 30, 11]`
  - channels `0..9` are ARC colors
  - channel `10` is padding
  - for in-grid cells `(y < height, x < width)`, exactly one color channel
    matching `grid[y][x]` is `1` and padding channel is `0`
  - for out-of-grid padding cells, channels `0..9` are `0` and channel `10` is
    `1`
- If Pass C chooses a non-convolutional learner, the tensor is flattened
  row-major by `y`, then `x`, then channel, producing a length-`9900` dense
  vector. If Pass C chooses a small CNN for this arm, this tensor is its only
  input and that choice closes the Phase 2.5 simple-convolution-feature
  baseline gap.

#### Decoder Semantics

The representation contract distinguishes representation prediction from grid
decoding:

- `signature_only` has no admitted exact-grid decoder. It is scored on exact
  output-representation match and representation-distance metrics only. Any
  exact-grid column for direct `signature_only` decoding is `NA`, not `0`.
- `signature_palette` has a deterministic diagnostic decoder named
  `top_left_palette_orbit_v1`; this decoder is intentionally conservative and
  exists only to make representation loss visible.
- `metadata_only` has no admitted exact-grid decoder unless a later Pass C
  candidate selector carries concrete candidate grids. Direct metadata-only
  exact-grid decoding is `NA`.
- `raw_grid_lowcap` uses identity decoding when a learner or candidate selector
  emits a concrete grid. If a later learner emits a per-cell channel tensor, the
  decoder must take the argmax over channels `0..9` inside the emitted shape;
  the emitted-shape rule must be filed in Pass C before use.

`top_left_palette_orbit_v1`:

1. Parse `canonicalObjectSignature`.
2. If signature is `empty`, return the all-zero grid with the supplied `shape`.
3. Require `shape` to be present and require the signature bounding box to fit
   inside that shape; otherwise mark `coverage` failure.
4. Require the number of non-zero roles in the signature to equal
   `nonzeroPalette.length`; otherwise mark `coverage` failure.
5. Map role `1` to the smallest non-zero palette color, role `2` to the next
   smallest, and so on.
6. Create an all-zero grid of `shape`.
7. Place the canonical object at `(x=0, y=0)` in its canonical D4 orientation.
8. Ignore `localSignatureBag` during rehydration, then re-project the decoded
   grid and report whether the decoded `canonicalObjectSignature`,
   `localBagHash`, `shape`, and `palette` match the predicted representation.

If the diagnostic decoder's grid differs from the target because the target
uses another translation, orientation, or color-role-to-palette assignment, the
instance is a `residual` error, not a learner `detection` error.

#### Color-Role Quarantine

The pre-registered color-role quarantine set for Pass A is exactly the six
registered tasks whose `primary_prior` is `color_role`:

- `08ed6ac7`
- `0a2355a6`
- `2601afb7`
- `292dd178`
- `37d3e8b2`
- `3ad05f52`

These tasks are not removed from Phase 3. They are reported as an explicit
stratum, and any claim that depends on `signature_only` must name this stratum
as structurally color-identity lossy.

### 2026-05-28 -- Pass A Clarifications

Author: Claude (Opus 4.7).

Justification: post-audit of the Pass A Representation And Decoder Contract
surfaced four implicit definitions that the runner author should not have to
infer. This amendment pins them without altering Pass A's normative contract
or any envelope numbers.

Verdict impact: no execution admission. Pass A's freeze stands; this
amendment supplements it.

#### objectTokenCount Definition

For the token-contribution formula `count / objectTokenCount`,
`objectTokenCount` is the total number of object tokens emitted for the
grid's canonical signature, computed before contribution accumulation:

- if `canonicalObjectSignature` is `empty`, `objectTokenCount = 1` (only the
  `obj:empty` token is emitted)
- otherwise, `objectTokenCount = 4 + N`, where the constant `4` covers the
  fixed `obj:bbox_w`, `obj:bbox_h`, `obj:role_count`, and `obj:cell_count`
  tokens, and `N` is the number of `obj:cell=x:y:role` cell tokens (one per
  non-zero cell after canonicalization)

`objectTokenCount` divides every object-token contribution including the
fixed-meta tokens, so the sum of object-token contributions equals `1` before
the hashed-suffix L2 normalization step.

#### roleCount Derivation

`roleCount` in the `obj:role_count=<roleCount>` token is derived from the
canonical signature's cell tokens, not stored in the signature string. It is
the count of distinct roles in the cell-token set, which equals
`nonzeroPalette.length` by construction (first-occurrence row-major
remapping assigns exactly one role per distinct non-zero color). For an empty
signature, `roleCount = 0` and the `obj:role_count` token is not emitted at
all (only `obj:empty` is emitted).

#### Hash Case Convention

All Phase 3 hash fields are uppercase hexadecimal: `signatureHash`,
`localBagHash`, every entry in the receipt's `hashes.json`, and any hash
literal referenced inside this spec. Phase 2's `signature_hash` and
`local_bag_hash` were emitted with Node's default lowercase encoding; any
cross-check against Phase 2 artifacts must normalize case (uppercase the
Phase 2 values or use case-insensitive comparison) before asserting
equality.

The runner is free to compute hashes in either case internally as long as
every emitted artifact field and cross-check report uses the uppercase
convention.

#### L2 Normalization Scope

L2 normalization applies only to the hashed signature suffix (indices
`28..4123`), not to the dense 28-dimensional metadata prefix (indices
`0..27`). This is intentional:

- the signature suffix is normalized so its contribution magnitude is
  invariant to token count (a 4-cell signature and a 40-cell signature both
  have unit L2 norm in the suffix block);
- the metadata prefix sits on its pre-registered per-feature scale
  (`height/30`, `width/30`, etc.) so per-feature comparisons remain
  interpretable.

Pass C learner selection must account for the resulting scale difference.
Any regularizer or distance metric whose behavior depends on global feature
scale must be filed in Pass C with a justification of how it handles the
metadata-vs-signature split.

### 2026-05-28 -- Pass B Split, Floor, And Discrimination Contract

Author: Claude (Opus 4.7).

Justification: Pass B closes the second roadmap slot. It pins how Phase 3
constructs LODO and public-training test instances, handles the five `k=2`
tasks, enforces the ARC-AGI two-prediction discipline, defines the
`oracle_copy_floor` double reading, and specifies the discrimination floor
that must be reported alongside arm scores. Decoder/learner execution
remains held pending Pass C.

Verdict impact: no execution admission. Pass B is filed.

#### LODO Instance Construction

LODO instances are constructed deterministically from
`P0_TASK_REGISTER.csv`:

1. Iterate registered task rows in the order they appear in the binding
   register CSV (the order is the `task_id`-ascending output of
   `arc-phase0-draft-register.mjs`; verify the binding CSV matches that
   order at runtime).
2. For each task with `k` train pairs (`k ≥ 2` in the registered subset),
   generate `k` LODO instances: hold out train pair index `i` (0-based) and
   condition on the other `k-1` pairs.
3. Instance identifier: `lodo:<task_id>:<heldout_index>`. The cross-task
   ordering of LODO instances is the concatenation in the iteration order
   above.

Total LODO instances: `115` (sum of `k` over the 36 tasks; matches the
Phase 2 train-pair count `trainPairCount=115`).

#### Small-k Stratum

The five `k=2` tasks (`00576224`, `025d127b`, `08ed6ac7`, `0b17323b`,
`11e1fe23`) generate LODO instances whose conditioning set has size 1.
These instances enter the audit but are reported as a separate stratum.
Three strata appear in every receipt:

| stratum | tasks | LODO instances |
| --- | ---: | ---: |
| `all_tasks` | 36 | 115 |
| `k_ge_3` | 31 | 105 |
| `k_eq_2` | 5 | 10 |

Phase 3 verdict tables must report both `all_tasks` and `k_ge_3` rows. No
support verdict may depend solely on the `k_eq_2` stratum: the `k_ge_3`
result must independently meet the success threshold defined in Pass C. If
the `k_eq_2` and `k_ge_3` strata diverge sharply, the receipt must name the
divergence in the verdict text.

#### Public-Training Test Instances

The public-training test audit runs after the LODO scoring is complete and
the LODO receipt is serialized to disk. It produces one instance per
registered task per test input:

1. Identifier: `pttest:<task_id>:<test_index>`.
2. Conditioning set: all `k` train pairs for that task.
3. Target: the held public-training test pair's output. Phase 0 admitted
   manual inspection of public-training rows; the public-training test
   outputs live in the same task JSON files as the train pairs and are
   allowed at scoring time, not at predict time.
4. Total instances: `36` (one per registered task; matches the
   `test_inputs` envelope).

Discipline guard: the runner must close and re-open the LODO manifest from
disk before reading any public-training test pair's `output` field. The
preferred Pass D architecture is a two-runner split (one runner per lane)
so the order is enforced by file dependency; if Pass D chooses a single
runner, the in-memory test outputs must remain unread until after
`scores.csv` and `per_instance.csv` rows for the LODO lane are written and
flushed.

#### Two-Prediction Discipline

Every arm emits at most two ordered candidate grids per instance, matching
the ARC-AGI-2 evaluation protocol's two-prediction allowance. Ordering
rules:

- if the learner emits multiple candidates with confidence scores, rank by
  descending confidence;
- on confidence ties, rank by ascending deterministic tie-break seed derived
  from `(master_seed, task_id, lane, heldout_index, arm, candidate_rank)`
  (the seed-derivation rule is the same as Pass C will pin for learners);
- if the learner emits exactly one candidate, slot `1` is filled and slot
  `2` is empty (not duplicated);
- if the learner emits zero candidates, both slots are empty and the
  instance is recorded with failure label `coverage`.

Exact-grid scoring credits the instance if either slot matches the target.
Representation-level scoring uses slot `1` only; slot `2` is reported in
`per_instance.csv` but does not enter the primary representation aggregate.

#### oracle_copy_floor Double Reading

`oracle_copy_floor` is reported with two readings, each emitted as a
distinct row in `scores.csv`:

1. `phase0_reference` -- the Phase 0 cheap-baseline numbers on the
   registered subset (all five non-random baselines hit `0/36` exact). Sourced
   from `results/arc/phase0-baselines/summary.csv`; included for traceability
   only and not eligible to participate in Phase 3 arm comparisons.
2. `lodo_rerun` -- the same `random_valid`, `identity_copy`, `dsl_lite_v0`,
   `dsl_lite_v1`, `dsl_lite_v2`, and `tiny_learned_v0` baseline logic
   re-applied per LODO instance using only the `k-1` conditioning pairs (and
   per public-training test instance using all `k` pairs, matching the
   conditioning set sizes of the learner arms). This is the apples-to-apples
   floor and is the value that participates in Phase 3 verdict gates.

The Pass D runner must source `lodo_rerun` from the same baseline
implementation file (`scripts/arc-phase0-baselines.mjs`) to keep the
oracle floor reproducible against the frozen Phase 0 implementations.

#### Discrimination Floor

Before arm-comparison scores are interpreted, the runner emits a
`discrimination.csv` report covering:

- `unique_heldout_signatures` -- per task, the count of distinct
  canonical-output-signatures across that task's LODO held-outs;
- `majority_signature_rate` -- per task, the fraction of LODO instances
  whose held output's canonical signature equals the most common held-output
  signature in that task;
- `collapse_count` -- per task, the number of LODO instances whose held
  output's canonical signature was already seen in an earlier LODO instance
  of the same task;
- `learner_task_trivial` -- per task, `true` if all LODO held-outputs in
  that task collapse to a single canonical signature, else `false`.

Aggregation rules:

- if `learner_task_trivial = true` for a task, the arm-comparison verdict
  for that task is suppressed (its per-task arm scores still appear in
  `per_instance.csv` but the arm aggregate excludes it);
- if more than `30%` of registered tasks (i.e. more than 10 of 36) are
  `learner_task_trivial`, the overall run verdict is automatically
  `task hardness / decoder failure` regardless of arm scores; the receipt
  text must name the collapse rate and which tasks triggered it.

The 30% threshold reflects that with `k=2..5` demonstrations per task,
some genuine signature collapse is expected; pervasive collapse means the
LODO protocol itself is not generating a meaningful prediction task and the
verdict is uninterpretable.

### 2026-05-28 -- Pass C Learner, Seeds, Metrics, And Failure Labels

Author: Codex.

Justification: Pass C closes the third roadmap slot by choosing the first
learner family, pinning all distance and ranking rules, freezing the seed slate,
and defining the metric hierarchy and failure labels used to interpret Phase 3.
The first learner is intentionally low-capacity and deterministic so the run
tests representation neighborhoods before adding any fitted neural capacity.

Verdict impact: no execution admission. Pass C is filed. Runner implementation,
artifact hash schema, command wiring, and decoder execution remain held pending
Pass D.

#### First Learner Family

The first Phase 3 learner is `nn_output_transfer_v1`, a deterministic
nearest-input output-transfer selector.

For each lane instance and arm:

1. Build the conditioning set from Pass B.
2. Represent the query input and every conditioning input under the arm.
3. Compute arm-specific input distance from the query input to each
   conditioning input.
4. Rank conditioning pairs by ascending input distance, then by deterministic
   tie-break key, then by original pair index.
5. Transfer the ranked conditioning output representation as the candidate
   output. Emit at most the first two distinct candidate outputs.

This learner has:

- no hidden layers;
- no optimizer;
- no learned parameters;
- no epochs, gradient steps, or early stopping;
- no cross-task fitted state;
- `candidateLimit = 2`;
- one run seed, used only for deterministic tie-breaking.

No MLP, linear regressor, CNN, or stochastic learner is admitted for the first
Phase 3 receipt. A later learner family requires a new append-only amendment and
must produce a separate receipt; it may not overwrite the `nn_output_transfer_v1`
receipt.

#### Candidate Identity And De-Duplication

Candidate identity is arm-specific:

- `signature_only`: `signatureHash|localBagHash`
- `signature_palette`:
  `shape|palette|nonzeroCells|nonzeroComponents|density|signatureHash|localBagHash`
- `metadata_only`: JSON serialization of the Pass A metadata vector
- `raw_grid_lowcap`: `JSON.stringify(grid)`

If two conditioning pairs produce the same candidate identity, keep only the
first under the ranked order and record all duplicate source pair indices in
`per_instance.csv`. A candidate copied from a conditioning output is still a
representation candidate for representation arms; `signature_only` does not gain
a direct exact-grid decoder from this copy.

#### Arm Distances

All distances are in `[0, 1]` except where noted; lower is better.

`signature_only`:

- distance is cosine distance over the Pass A hashed signature suffix only:
  `1 - dot(a_suffix, b_suffix)`;
- the suffixes are L2-normalized by Pass A, so no metadata scale enters.

`signature_palette`:

- distance is `0.5 * signature_cosine_distance + 0.5 * metadata_l1`;
- `metadata_l1` is the mean absolute difference over the 28 metadata-prefix
  coordinates;
- this fixed 50/50 split is the Pass C scale rule required by the Pass A L2
  clarification.

`metadata_only`:

- distance is `metadata_l1`.

`raw_grid_lowcap`:

- convert each grid to a `[30, 30]` label field where in-grid cells are colors
  `0..9` and out-of-grid padding cells are label `10`;
- distance is normalized Hamming over all `900` label positions.

Output-representation distance uses the same arm distance between the selected
candidate output representation and the target output representation.

#### Seed Slate And Tie-Breaks

The first receipt uses exactly one master seed:

- `masterSeed = 20260528`

Tie-break key:

1. Build the UTF-8 string
   `arc-p3-c0-v1\0<masterSeed>\0<task_id>\0<lane>\0<query_index>\0<arm>\0<source_pair_index>`.
2. Compute SHA-256.
3. Interpret the first eight digest bytes as an unsigned big-endian integer.
4. Sort ascending.

For LODO, `query_index` is the held-out train pair index. For public-training
test instances, `query_index` is the public-training test index. This rule
clarifies Pass B's tie-break phrase: the pre-rank candidate key is the source
pair index; the emitted `candidate_rank` is assigned only after sorting.

#### Metric Columns

Each `per_instance.csv` row must include these fields where applicable:

- `grid_exact_slot1`
- `grid_exact_any_slot`
- `rep_exact_slot1`
- `rep_exact_any_slot`
- `shape_exact_slot1`
- `palette_exact_slot1`
- `pixel_accuracy_slot1`
- `pixel_accuracy_best`
- `output_rep_distance_slot1`
- `output_rep_distance_best`
- `candidate_pool_contains_target_rep`
- `candidate_pool_contains_target_grid`
- `top2_contains_target_rep`
- `top2_contains_target_grid`
- `failure_label`

`grid_exact_*` and `pixel_accuracy_*` are `NA` for direct `signature_only` and
direct `metadata_only` representation scoring. `signature_palette` grid columns
use `top_left_palette_orbit_v1`; `raw_grid_lowcap` grid columns use identity
decoding. Pixel accuracy follows Phase 0: if shapes differ, pixel accuracy is
`0`.

Aggregate `scores.csv` rows must report, at minimum:

- `grid_exact_rate_any_slot`
- `rep_exact_rate_slot1`
- `rep_exact_rate_any_slot`
- `shape_exact_rate_slot1`
- `palette_exact_rate_slot1`
- `mean_pixel_accuracy_best`
- `mean_output_rep_distance_slot1`
- `mean_output_rep_distance_best`
- `mean_output_rep_similarity_best = 1 - mean_output_rep_distance_best`
- `coverage_failure_rate`
- `detection_failure_rate`
- `residual_failure_rate`

Rates are computed after applying Pass B's `learner_task_trivial` suppression
rule. Suppressed instances still appear in `per_instance.csv` with
`suppressed_by_discrimination = true`.

#### Comparison Hierarchy

Arm comparisons use this ordered metric ladder:

1. `grid_exact_rate_any_slot`, only when both arms have non-`NA` grid exact;
2. `rep_exact_rate_slot1`;
3. `shape_exact_rate_slot1`, only when both arms expose shape;
4. `palette_exact_rate_slot1`, only when both arms expose palette;
5. `mean_pixel_accuracy_best`, only when both arms have non-`NA` pixel accuracy;
6. `mean_output_rep_similarity_best`.

Exact-rate comparisons are decisive only when the absolute rate difference is
at least `0.05` and the corresponding instance-count difference is at least
`max(2, ceil(0.05 * comparable_instance_count))`.

Continuous comparisons are decisive only when the absolute difference is at
least `0.02`.

If a metric is tied or non-comparable, continue to the next metric in the
ladder. If every comparable metric is non-decisive, the arms are
`competitive_tie`.

Definitions:

- `A beats B`: the first decisive comparable metric favors `A`.
- `A competitive_with B`: no comparable metric shows `A` materially trailing
  `B`; this includes `competitive_tie`.
- `A materially_trails B`: the first decisive comparable metric favors `B`.

Exact-rate `0` vs `0` can never support a sufficiency claim by itself; it must
fall through to the lower ladder metrics or end as `competitive_tie`.

#### Verdict Thresholds

The verdict table in the original spec is interpreted as follows:

- **support on registered subset** requires:
  - `signature_palette beats metadata_only` on LODO `k_ge_3`;
  - `signature_palette beats metadata_only` on public-training test
    `all_tasks`;
  - `signature_palette competitive_with raw_grid_lowcap` on LODO `k_ge_3`;
  - `signature_palette competitive_with raw_grid_lowcap` on public-training
    test `all_tasks`;
  - LODO `all_tasks` does not contradict the `k_ge_3` result.
- **partial support / lossy representation** applies when
  `signature_palette beats metadata_only` on LODO `k_ge_3` but
  `signature_palette materially_trails raw_grid_lowcap` on either LODO
  `k_ge_3` or public-training test `all_tasks`.
- **palette-dependent support** applies when `signature_only materially_trails
  signature_palette` on the color-role stratum and `signature_palette` otherwise
  meets the support or partial-support criteria.
- **sufficiency failure** applies when `signature_palette` does not beat
  `metadata_only` on LODO `k_ge_3`.
- **task hardness / decoder failure** applies when Pass B's discrimination
  collapse threshold fires, or when every grid-bearing arm has
  `grid_exact_rate_any_slot < 0.03` and every representation arm has
  `rep_exact_rate_slot1 < 0.05` on LODO `k_ge_3`.

Public-training test results can downgrade a support verdict to partial,
sufficiency failure, or task-hardness/decoder failure. They cannot upgrade a
LODO failure to support.

#### Failure Labels

Per-instance failure labels are assigned after candidate ranking and decoding:

- `none`: at least one emitted candidate receives exact credit on the primary
  available metric for that arm.
- `coverage`: the target output representation or grid is absent from the full
  pre-top-2 candidate pool, or the decoder cannot produce a candidate because
  required structural fields are missing or invalid.
- `detection`: the target is present in the full candidate pool but absent from
  the emitted top two candidates, or the target representation is present in
  top two but not in slot 1 for a representation-level slot-1 metric.
- `residual`: the selected representation is exact or nearest under the arm
  distance, but deterministic decoding leaves grid, geometry, translation,
  orientation, or color-assignment error.

When multiple failure labels could apply, precedence is:
`coverage` before `detection` before `residual`. A row with all primary and
secondary metrics `NA` must be labeled `coverage`.

### 2026-05-28 -- Pass C Clarifications

Author: Claude (Opus 4.7).

Justification: post-audit of Pass C surfaced four operationalization notes
that the runner author should not have to infer. This amendment pins them
without altering Pass C's normative contract.

Verdict impact: no execution admission. Pass C's freeze stands; this
amendment supplements it.

#### Lane String Values

The `<lane>` token in the Pass C tie-break key takes exactly two literal
values:

- `lodo` for the LODO representation audit lane
- `pttest` for the public-training test audit lane

The instance identifier prefixes from Pass B (`lodo:<task_id>:<heldout_index>`
and `pttest:<task_id>:<test_index>`) use the same literal strings.

#### Contradict Semantics

"LODO `all_tasks` does not contradict the `k_ge_3` result" in the verdict
threshold is operationalized as: for every pairwise comparison that the
support verdict requires on `k_ge_3`, the same comparison on `all_tasks`
must not reverse direction. Concretely:

- if `k_ge_3` says `signature_palette beats metadata_only`, `all_tasks` must
  not say `signature_palette materially_trails metadata_only`;
  `all_tasks` may be `beats`, `competitive_with`, or `competitive_tie`;
- if `k_ge_3` says `signature_palette competitive_with raw_grid_lowcap`,
  `all_tasks` must not say `signature_palette materially_trails
  raw_grid_lowcap`.

The contradiction flag must be computed deterministically by re-running the
comparison hierarchy on the `all_tasks` stratum and recording the result
alongside the `k_ge_3` result in `scores.csv`. The verdict text must name
any contradiction explicitly when it fires.

#### Failure-Label Distribution Disclosure

`nn_output_transfer_v1` emits only de-duplicated conditioning-output
candidates: the candidate pool for an instance is the set of unique
conditioning outputs under the arm's candidate identity. For LODO `k_ge_3`,
that is at most 4 candidates; for `k_eq_2`, at most 1. Most ARC tasks
present a different output per train pair, so for the majority of LODO
instances the held-out output's representation is **not** in the
conditioning pool, and the failure label is `coverage`.

This is a learner-class limitation, not a representation defect. The Pass D
public-language drafts must frame a high `coverage_failure_rate` as a
property of `nn_output_transfer_v1` rather than as evidence against any
representation arm. Later learner families (separate amendments) may emit
synthesized candidates and produce meaningfully different failure-label
distributions; comparing `coverage_failure_rate` across learner-family
amendments is not informative unless the candidate-pool construction is
also held constant.

#### Failure-Label Sum Invariant

For every `per_instance.csv` row, exactly one of the following holds:

- `failure_label = none` (at least one slot received exact credit on the
  primary available metric for that arm);
- `failure_label = coverage`;
- `failure_label = detection`;
- `failure_label = residual`.

For each `(lane, arm, stratum)` aggregate row in `scores.csv`, the
invariant

```
exact_credit_rate + coverage_failure_rate + detection_failure_rate + residual_failure_rate = 1
```

must hold within rounding tolerance `1e-9`, where `exact_credit_rate` is
the rate of `failure_label == none` over the same denominator as the three
failure-rate columns. The runner must compute this sum and abort the run
(non-zero exit) if it fails the tolerance.

Instances suppressed by Pass B's `learner_task_trivial` rule are excluded
from both the numerator and the denominator of all four rates, matching
the Pass C "rates are computed after applying Pass B's
`learner_task_trivial` suppression rule" requirement.

### 2026-05-28 -- Pass D Receipt, Command, Freeze, And Public Language

Author: Claude (Opus 4.7).

Justification: Pass D closes the final roadmap slot before runner
implementation. It pins runner architecture, exact command shape, output
directory layout, receipt artifact schema, freeze marker semantics,
terminology, and per-verdict public-language drafts. After this amendment
is committed alongside the runner script and `npm` wiring, decoder/learner
execution against the registered Phase 0 subset is admitted.

Verdict impact: no execution admission by this amendment alone. Execution
becomes admissible once the freeze-marker commit (defined below) lands.

#### Terminology

Three roles are distinct and used consistently across Passes A-D and the
runner:

- `learner` -- the task-conditioned mapper or candidate ranker. For the
  first receipt this is `nn_output_transfer_v1` per Pass C.
- `decoder` -- deterministic conversion from a predicted representation or
  candidate to a concrete grid. For `signature_palette` this is Pass A's
  `top_left_palette_orbit_v1`; for `raw_grid_lowcap` it is identity; for
  `signature_only` and `metadata_only` it is `NA`.
- `runner` -- the complete Phase 3 protocol implementation that constructs
  instances per Pass B, invokes the learner per arm per Pass C, scores
  candidates, applies failure labels, and writes the receipt per the schema
  in this amendment.

The Phase 3 spec status header `DECODER TRAINING HOLD` covers all three.

#### Runner Architecture

Phase 3 uses the two-runner architecture preferred by Pass B's discipline
guard:

- `scripts/arc-phase3-lodo.mjs` -- runs the LODO lane only, emits the LODO
  portion of every receipt artifact, exits non-zero on any contract
  violation.
- `scripts/arc-phase3-pttest.mjs` -- runs the public-training test lane,
  requires the LODO lane's `manifest.json` as a `--lodo-manifest`
  argument, refuses to start if the LODO manifest is missing or its
  protocol/feature/receipt versions do not match this amendment.

This split enforces the Pass B order-of-operations constraint by file
dependency: the public-training test runner cannot read any test pair's
`output` field until the LODO runner has written and flushed its manifest.
A single-runner architecture is forbidden for the first receipt and would
require a new append-only amendment.

The runner scripts must be added in the same commit as this amendment to
constitute the freeze-marker commit (see below). Until that commit lands,
neither runner is admitted.

#### npm Commands and Reproducibility

The frozen command surface is three `npm` scripts:

```
"arc:phase3:lodo": "node scripts/arc-phase3-lodo.mjs"
"arc:phase3:pttest": "node scripts/arc-phase3-pttest.mjs"
"arc:phase3:sufficiency": "npm run arc:phase3:lodo && npm run arc:phase3:pttest"
```

The frozen reproducible invocation for the full Phase 3 first receipt is:

```powershell
npm run arc:phase3:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-sufficiency
```

(`npm run` forwards `--` only to the first script in a chain; the
`arc:phase3:sufficiency` umbrella must thread arguments to both inner
runners. The runner author may either re-implement the chain inside a
single wrapper script or require the user to invoke `arc:phase3:lodo` and
`arc:phase3:pttest` separately with explicit arguments. Pin the choice in
the freeze-marker commit.)

Each runner accepts:

- `--data-dir <path>` -- path to the public ARC-AGI-2 `data/` directory;
  the runner must only read from `<data-dir>/training/` and must refuse to
  read from `<data-dir>/evaluation/`;
- `--register <path>` -- path to `docs/prereg/arc/P0_TASK_REGISTER.csv`;
- `--out <path>` -- output directory for receipt artifacts (default
  `results/arc/phase3-sufficiency`);
- `--master-seed <int>` -- override Pass C's `masterSeed=20260528`; the
  default is the Pass C value and any override is recorded in `manifest.json`.

The `arc-phase3-pttest.mjs` runner additionally accepts:

- `--lodo-manifest <path>` -- required; path to the LODO runner's
  `manifest.json` for protocol-version cross-check.

#### Output Directory Layout

The default receipt directory is `results/arc/phase3-sufficiency/`. After
a successful `npm run arc:phase3:sufficiency` invocation, the directory
contains:

```
results/arc/phase3-sufficiency/
  manifest.json          -- protocol metadata, command, git commit, seed, timings, deps, version strings
  scores.csv             -- one row per (lane, arm, stratum); columns per Pass C aggregate list
  per_instance.csv       -- one row per (instance, arm); columns per Pass C per-instance list
  discrimination.csv     -- one row per task and per stratum; columns per Pass B discrimination floor
  hashes.json            -- uppercase SHA-256 of every other artifact in this directory
  lodo/                  -- lane-local raw artifacts (per-instance candidate dumps, debug)
  pttest/                -- lane-local raw artifacts
```

The five top-level files (`manifest.json`, `scores.csv`, `per_instance.csv`,
`discrimination.csv`, `hashes.json`) are the canonical receipt; the
`lodo/` and `pttest/` subdirectories may include lane-local debug
artifacts but no field referenced by the verdict text may live there
exclusively.

`results/arc/phase3-sufficiency/` is added to `.gitignore` in the same
freeze-marker commit. Receipt content is reproducible from `npm run
arc:phase3:sufficiency` plus the frozen register and dataset.

#### Receipt Artifact Schema

Schema versions emitted in `manifest.json`:

- `featureSchemaVersion`: `arc-p3-feature-v1` (matches Pass A)
- `protocolVersion`: `arc-p3-protocol-v1`
- `receiptSchemaVersion`: `arc-p3-receipt-v1`
- `learnerVersion`: `nn_output_transfer_v1` (matches Pass C)

`manifest.json` minimum fields (additional fields allowed):

- `generatedAt` -- ISO-8601 timestamp at runner start
- `completedAt` -- ISO-8601 timestamp after the final artifact flush
- `tool` -- runner script path
- `command` -- the exact argv after node arg parsing
- `gitCommit` -- repo HEAD commit hash at runner start (uppercase hex);
  the runner must refuse to run on a dirty worktree unless invoked with
  `--allow-dirty`, which is recorded in `manifest.json`
- `dataDir`, `registerPath`, `outDir`
- `masterSeed`, `seedOverridden` (boolean)
- `featureSchemaVersion`, `protocolVersion`, `receiptSchemaVersion`,
  `learnerVersion`
- `taskCount`, `lodoInstanceCount`, `pttestInstanceCount`
- `dataDirHash` -- uppercase SHA-256 of a manifest of training-task
  filenames and per-file SHA-256 hashes for the 36 registered tasks
- `registerHash` -- uppercase SHA-256 of `P0_TASK_REGISTER.csv`
- `phase2BaselinesManifestHash` -- uppercase SHA-256 of
  `results/arc/phase2-baselines/manifest.json` for traceability; the
  runner does not need that file to exist, but if it exists the hash is
  recorded and any mismatch with the value pinned at the top of this spec
  produces a warning in `manifest.json` rather than aborting
- `platform`, `nodeVersion`, `dependencies` (only the runner's direct
  dependencies; no `node_modules` walk)

`scores.csv` columns (one row per `(lane, arm, stratum)`):

- `lane`, `arm`, `stratum`, `instance_count`, `suppressed_count`
- the Pass C aggregate metric list:
  `grid_exact_rate_any_slot`, `rep_exact_rate_slot1`,
  `rep_exact_rate_any_slot`, `shape_exact_rate_slot1`,
  `palette_exact_rate_slot1`, `mean_pixel_accuracy_best`,
  `mean_output_rep_distance_slot1`, `mean_output_rep_distance_best`,
  `mean_output_rep_similarity_best`,
  `coverage_failure_rate`, `detection_failure_rate`,
  `residual_failure_rate`, `exact_credit_rate`

`per_instance.csv` columns (one row per `(instance_id, arm)`):

- `instance_id`, `lane`, `task_id`, `primary_prior`, `arm`, `stratum`,
  `query_index`, `candidate_pool_size`, `candidate_pool_size_unique`,
  `suppressed_by_discrimination`
- the Pass C per-instance metric list (omitted here for brevity but
  enumerated verbatim from Pass C)
- `duplicate_source_pair_indices` -- semicolon-separated source pair
  indices collapsed under Pass C de-duplication
- `slot1_candidate_identity`, `slot2_candidate_identity`

`discrimination.csv` columns (per Pass B):

- `task_id`, `stratum`, `instance_count`, `unique_heldout_signatures`,
  `majority_signature_rate`, `collapse_count`, `learner_task_trivial`

`hashes.json` schema: object mapping artifact filename (relative to the
receipt directory) to uppercase SHA-256 hex string. Includes every file
under the receipt directory at write time except `hashes.json` itself.

#### Freeze Marker

The freeze-marker commit is the single Git commit that:

1. lands this Pass D amendment;
2. adds `scripts/arc-phase3-lodo.mjs` and `scripts/arc-phase3-pttest.mjs`;
3. adds the three Pass D npm script entries to `package.json`;
4. adds `results/arc/phase3-sufficiency/` to `.gitignore`;
5. adds `arc-phase3-lodo.mjs` and `arc-phase3-pttest.mjs` to the
   leak-check's allowlist if and only if they need `evaluation` literals
   for the refuse-to-read guard (they do; the guard rejects
   `<data-dir>/evaluation/` access).

After the freeze-marker commit:

- no post-hoc edits to arm schemas (Pass A), split or floor handling
  (Pass B), learner choices or thresholds (Pass C), receipt schema or
  command shape (Pass D) are allowed before the first receipt;
- bug fixes that change runner behavior require a new append-only
  amendment that names the defect, expected metric impact, and whether the
  first receipt is void or superseded;
- changes that are pure documentation (typo fix in a comment, README
  update) do not require an amendment but must be in a commit that does
  not touch any runner or schema file.

#### Public-Language Drafts

Adopted verbatim from the Decoder Admission Roadmap, with one addition
covering the failure-label disclosure from the Pass C Clarifications:

- **Support on registered subset**: "On a pre-registered public-training
  subset, the Sundog signature-plus-palette representation supported a
  low-capacity input-to-output audit at or near the matched raw-grid
  control. This is not an ARC solve claim and does not use the public
  evaluation split."
- **Partial support / lossy representation**: "The signature-plus-palette
  representation retained measurable task information beyond coarse
  metadata, but trailed the matched raw-grid control; the representation
  is useful but not sufficient for the registered subset."
- **Palette-dependent support**: "The ARC representation that works is
  signature-plus-palette. Canonical signature alone is insufficient on
  color-role tasks because it quotients absolute color identity."
- **Sufficiency failure**: "The registered Phase 3 audit did not show that
  the shadow signature representation is sufficient for ARC-style
  input-to-output transformations beyond coarse metadata controls."
- **Task hardness / decoder failure**: "The first Phase 3 runner did not
  produce an interpretable sufficiency verdict because all arms were near
  the exact-match floor or because the LODO held-out outputs collapsed
  beyond Pass B's discrimination threshold; the result is a decoder or
  task-hardness finding, not evidence for or against signature
  sufficiency."

Additional disclosure required in any public copy that cites a
`coverage_failure_rate` from the first Phase 3 receipt:

> "The first Phase 3 learner (`nn_output_transfer_v1`) is a deterministic
> nearest-input output-transfer selector. Its candidate pool is the
> unique conditioning outputs only, so for ARC tasks where each
> demonstration shows a different output, a high coverage failure rate
> is a property of the learner class, not of the representation."

No public copy may describe a Phase 3 result as evidence on the public
evaluation split or as an ARC solve claim. The Phase 0-2 public-language
constraints (no "Sundog solves ARC", no "human-level abstraction", no
"the 5D subspace is universal", no claim that a Kaggle entry validates
the theory without the Phase 3 sufficiency audit) remain in force.

### 2026-05-28 -- Freeze-Marker Implementation Note

Author: Codex.

Justification: the Pass D amendment landed in commit `c81c594` before the
runner files were added. This implementation commit completes the operational
freeze marker by adding the two required lane runners, a thin argument-forwarding
umbrella runner, npm wiring, the explicit Phase 3 receipt ignore path, and the
leak-check allowlist entries for the lane runners' refuse-to-read guard.

Verdict impact: after this implementation commit lands, the first Phase 3
receipt is admitted under the frozen Pass A-D contracts. The umbrella command
choice is pinned as:

```json
"arc:phase3:sufficiency": "node scripts/arc-phase3-sufficiency.mjs"
```

The wrapper invokes `scripts/arc-phase3-lodo.mjs` first and then
`scripts/arc-phase3-pttest.mjs`, forwarding the same user arguments to both
lanes and supplying `--lodo-manifest <out>/manifest.json` to the second lane
when the caller did not provide it. This is the Pass D "re-implement the chain
inside a single wrapper script" option.

The Phase 0 floor is sourced by exporting the frozen predictors from
`scripts/arc-phase0-baselines.mjs`; its CLI behavior is preserved.

### 2026-05-28 -- First-Receipt Verdict

Author: Claude (Opus 4.7).

Justification: this amendment files the first Phase 3 receipt verdict per
the Pass D admission rule. The freeze-marker commit
(`7B664BBA3D398BC3D7FCED4BA74AE446D2E684BD`) plus the
freeze-marker-implementation amendment together admitted the run; this
amendment closes the loop by recording outcomes, applying the Pass C
verdict gates, picking between concurrent verdict triggers, and adopting
the Pass D public-language draft.

Verdict impact: **task hardness / decoder failure**. No support, partial
support, palette-dependent support, or sufficiency-failure conclusion is
admitted by this receipt. Public-evaluation grid inspection, Kaggle
notebook work, and any "Sundog solves ARC" or sufficiency claim remain
forbidden.

#### Run Provenance

Command:

```powershell
npm run arc:phase3:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-sufficiency
```

| field | value |
| --- | --- |
| `gitCommit` | `7B664BBA3D398BC3D7FCED4BA74AE446D2E684BD` |
| `generatedAt` (LODO start) | `2026-05-28T05:10:22.089Z` |
| `completedAt` (pttest end) | `2026-05-28T05:10:37.200Z` |
| wall clock | ~16 s |
| `featureSchemaVersion` | `arc-p3-feature-v1` |
| `protocolVersion` | `arc-p3-protocol-v1` |
| `receiptSchemaVersion` | `arc-p3-receipt-v1` |
| `learnerVersion` | `nn_output_transfer_v1` |
| `masterSeed` | `20260528` (default; not overridden) |
| `allowDirty` | `false` (clean worktree) |
| `taskCount` | `36` |
| `lodoInstanceCount` | `115` |
| `pttestInstanceCount` | `36` |
| `phase2BaselinesManifestHash` | matches frozen value; no warning |

Receipt artifacts (uppercase SHA-256):

- `results/arc/phase3-sufficiency/manifest.json`:
  `F501B85FCA0F0257CCCCE387E82613755BFDB6EF2DD75C343A6614F9A39C6383`
- `results/arc/phase3-sufficiency/scores.csv` (66 rows):
  `32EB1A39C33BD451DC2A44C321A0D1347035CBDE5F0A4559ED59191671A56497`
- `results/arc/phase3-sufficiency/per_instance.csv` (1510 rows):
  `EE4F8DD72E52864DFFA9A17C4DB0CB40E8D72C970473F2E98FE3678F1AD91F33`
- `results/arc/phase3-sufficiency/discrimination.csv`:
  `ECF4593FA857BBF89179BB1B12C0E43C6877A2AC2E616A68047F18D8A9E25D02`

Pass C failure-label sum invariant: held within the runner's 1e-9
tolerance for every `(lane, arm, stratum)` row (runner exited 0).
Discrimination summary: `trivialTaskCount=2` (`05269061`, `11e1fe23`),
`learnerTaskTrivialThresholdFired=false` (2/36 = 5.6% < 30%).

#### Outcome (LODO `k_ge_3`, 105 instances, 102 active)

| arm | grid_exact | rep_exact_slot1 | shape_exact | palette_exact | pixel_acc_best | rep_sim_best | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `signature_only` | NA | `0.000` | NA | NA | NA | `0.639` | `1.000` |
| `signature_palette` | `0.000` | `0.000` | `0.588` | `0.480` | `0.351` | `0.784` | `1.000` |
| `metadata_only` | NA | `0.039` | `0.608` | `0.480` | NA | `0.939` | `0.951` |
| `raw_grid_lowcap` | `0.000` | `0.000` | `0.598` | `0.451` | `0.402` | `0.912` | `1.000` |

Oracle floors (LODO rerun, all `0/115` exact across all six Phase 0
baselines) confirm the Phase 0 zero-floor character survives the
`k-1` conditioning constraint.

#### Verdict Analysis

Two Pass C verdict triggers fire concurrently on this receipt:

**Trigger 1 -- task hardness / decoder failure** (Pass C):

- every grid-bearing arm has `grid_exact_rate_any_slot < 0.03`:
  `signature_palette = 0.000`, `raw_grid_lowcap = 0.000` -- pass;
- every representation arm has `rep_exact_rate_slot1 < 0.05`:
  `signature_only = 0.000`, `signature_palette = 0.000`,
  `metadata_only = 0.039`, `raw_grid_lowcap = 0.000` -- pass.

**Trigger 2 -- sufficiency failure** (Pass C):

The "does `signature_palette` beat `metadata_only` on LODO `k_ge_3`?"
comparison resolves through the Pass C hierarchy as follows:

1. `grid_exact_rate_any_slot`: `metadata_only` is NA, skip.
2. `rep_exact_rate_slot1`: `0.000` vs `0.039`; absolute rate difference
   `0.039` < `0.05` and instance-count difference `4` < `6` (= `max(2,
   ceil(0.05 * 102))`) -- not decisive.
3. `shape_exact_rate_slot1`: `0.588` vs `0.608`; absolute difference
   `0.0196` < `0.02` -- not decisive.
4. `palette_exact_rate_slot1`: `0.480` vs `0.480` -- not decisive.
5. `mean_pixel_accuracy_best`: `metadata_only` is NA, skip.
6. `mean_output_rep_similarity_best`: `0.784` vs `0.939`; absolute
   difference `0.155` >= `0.02` -- **decisive, `metadata_only` ahead**.

→ `signature_palette materially_trails metadata_only`, so the
sufficiency-failure trigger fires.

**Trigger precedence: task hardness preempts.** The decisive metric for
trigger 2 (`mean_output_rep_similarity_best`) measures a learner-class
artifact rather than representation sufficiency. Per the Pass C
Clarifications, `nn_output_transfer_v1`'s candidate pool is the unique
conditioning outputs only after de-duplication, and the candidate
identity rule for `metadata_only` is a 28-dim metadata vector while the
identity rule for `signature_palette` adds the canonical-signature hash
and local-bag hash. The coarser metadata identity therefore gets more
pool hits by chance for any reason a conditioning output happens to
share shape, palette, density, and color histogram with the held-out
output -- which is independent of whether `signature_palette` is the
better representation. With every grid-bearing exact metric at the
floor, attributing the `rep_sim_best` gap to "the signature
representation is insufficient" would over-claim from a learner artifact.

Task hardness / decoder failure is therefore the honest first-receipt
verdict. The receipt is evidence about `nn_output_transfer_v1`'s
candidate-pool structure, not about signature sufficiency.

The four `metadata_only` exact matches on `rep_exact_slot1` (4 of 102
active LODO `k_ge_3` instances) are recorded for traceability; they
represent grids in the same task whose 28-dim metadata vectors collide
even though their full grids differ, and they are not interpretable as
evidence that `metadata_only` is sufficient for ARC sufficiency.

#### Adopted Public Language

Per Pass D's pre-registered draft for this verdict:

> "The first Phase 3 runner did not produce an interpretable sufficiency
> verdict because all arms were near the exact-match floor or because
> the LODO held-out outputs collapsed beyond Pass B's discrimination
> threshold; the result is a decoder or task-hardness finding, not
> evidence for or against signature sufficiency."

Per Pass C Clarifications, any public copy citing this receipt's
`coverage_failure_rate` (`1.000` for `signature_only` /
`signature_palette` / `raw_grid_lowcap`; `0.951` for `metadata_only` on
LODO `k_ge_3`) must include:

> "The first Phase 3 learner (`nn_output_transfer_v1`) is a deterministic
> nearest-input output-transfer selector. Its candidate pool is the
> unique conditioning outputs only, so for ARC tasks where each
> demonstration shows a different output, a high coverage failure rate
> is a property of the learner class, not of the representation."

#### Carry-Forward

Admitted by this amendment:

- the first Phase 3 receipt is filed;
- the task-hardness verdict is the receipt's primary verdict;
- the sufficiency-failure trigger is named-but-preempted on the
  learner-class grounds above;
- the Pass D adopted public language is the only language allowed for
  public copy that cites this receipt.

Still forbidden, unchanged:

- editing arm schemas (Pass A), split or floor handling (Pass B),
  learner choices or thresholds (Pass C), or receipt schema or command
  shape (Pass D) without a new append-only amendment;
- public-evaluation grid inspection (Phase 6 only);
- Kaggle notebook prep, Kaggle private/semi-private splits;
- any sufficiency, dimensionality, palette-dependent, or partial-support
  claim from this receipt;
- describing this receipt as evidence on the public-evaluation split or
  as an ARC solve.

The next admitted Phase 3 work is the next-learner amendment that
follows below, which queues a synthesis-capable learner for a separate
receipt without overwriting `nn_output_transfer_v1`'s.

### 2026-05-28 -- Next-Learner Slot Reservation

Author: Claude (Opus 4.7).

Justification: the first receipt confirmed Pass C's a-priori reservation
language: `nn_output_transfer_v1`'s candidate pool is structurally the
unique conditioning outputs, so for ARC tasks whose demonstrations each
emit a different output, exact-match metrics are floored near zero by
the learner family rather than by representation choice. To produce a
receipt that can adjudicate signature sufficiency on something other
than learner-class artifacts, a synthesis-capable learner is required.
This amendment reserves the slot and pins the constraints any
next-learner amendment must satisfy; it does not pick the learner
itself.

Verdict impact: no execution admission. No second receipt may be
generated until a follow-up amendment lands that fills in the learner
family, hyperparameters, distance / identity rules (if changed), and
receipt path. The first receipt under `nn_output_transfer_v1` remains
the binding Phase 3 receipt until then.

#### Why a Synthesis-Capable Learner Is Needed

The first-receipt verdict above traces the exact-rate floor to
`nn_output_transfer_v1`'s candidate-pool construction: candidates are
the unique outputs from the conditioning train pairs only, and ARC
tasks generally emit a different output per train pair. The verdict
amendment recorded that all grid-bearing arms had
`grid_exact_rate_any_slot < 0.03` and all representation arms had
`rep_exact_rate_slot1 < 0.05` on LODO `k_ge_3` -- exactly the
Pass C task-hardness criterion. With this learner family, that result
is structural and would replicate on a second `nn_output_transfer_v1`
receipt with no change in parameters.

A learner that can produce output candidates outside the conditioning
pool is required to test whether `signature_palette` (or any arm) can
support a non-trivial exact rate. Without one, future Phase 3 receipts
would only re-record the same learner-class floor, no matter how many
amendments are filed.

#### Candidate Learner Families

Three families are named here as candidates for the next-learner
amendment. The amendment must pick exactly one and pin every parameter
per the Pass C template (architecture, hyperparams, seed slate, tie-
break, candidate limit, distance / identity rules if changed).

1. **`nn_delta_transfer_v1`** -- nearest-input transfer plus an additive
   cell-delta synthesis step. For the nearest conditioning pair
   `(input_i, output_i)` under the arm distance, emit a candidate
   computed cell-wise as `output_i` modified by the cells where
   `query_input` differs from `input_i` (specific delta rule to be
   pinned). Deterministic, parameter-free, candidate pool grows beyond
   the conditioning outputs. Trade-off: still purely local, will not
   handle shape-change tasks where output shape differs from input
   shape.

2. **`candidate_combinator_v1`** -- enumerate a bounded set of
   combinations over the conditioning outputs (cell-wise union with the
   most-frequent palette mask, D4 transforms of the nearest
   conditioning output, a small set of recolored variants) plus the
   conditioning outputs themselves. Pick the top-two by arm-distance to
   the predicted output representation (which is itself the nearest-
   input output transfer from `nn_output_transfer_v1`, used as a target
   ranking signal rather than the emitted candidate). Deterministic,
   bounded combinatorial. Trade-off: the synthesis rules are a small
   ad-hoc DSL; the amendment must pin them or the receipt isn't
   reproducible.

3. **`finite_program_selector_v1`** -- per task, enumerate a frozen
   small DSL of grid-to-grid transforms (taking the Phase 0
   `dsl_lite_v2` primitive set as the seed list), select the program
   consistent with the conditioning pairs under the arm-specific
   identity rule, apply it to the query input. Deterministic, bounded
   by the DSL size. Trade-off: this collapses much of the distinction
   between `signature_palette` and `raw_grid_lowcap` for the program-
   selection step (program consistency is mostly a grid-level check),
   so the representation arms degrade to "which arm helped rank
   candidate programs," not "which arm encoded the rule." If picked,
   the amendment must define how the arm representation enters program
   selection beyond grid equality.

These three are starting points, not an exhaustive list. The
next-learner amendment may name a different family entirely, subject to
the constraints below.

#### Constraints on Any Next-Learner Amendment

The next-learner amendment must:

1. specify exactly one `learnerVersion` string with a v1 or higher
   suffix (e.g. `nn_delta_transfer_v1`,
   `candidate_combinator_v1`,
   `finite_program_selector_v1`, or a new name);
2. keep Pass A's representation arms, feature serialization, and decoder
   semantics unchanged unless it adds a new arm or modifies arm
   serialization, in which case the amendment must include a Pass-A-style
   contract for the change with its own envelope and hash bucket rules;
3. keep Pass B's instance construction, strata, two-prediction
   discipline, and discrimination-floor reporting unchanged;
4. keep Pass C's metric columns, comparison hierarchy, verdict
   thresholds, and failure-label definitions and precedence unchanged
   unless the learner's candidate-pool construction makes a label
   inapplicable, in which case the amendment must name the substitute
   label and its trigger;
5. keep Pass D's receipt schema, hash convention, and dirty-worktree
   guard unchanged;
6. point the receipt at a learner-version-suffixed subdirectory of
   `results/arc/phase3-sufficiency/` (e.g.
   `results/arc/phase3-sufficiency-<learner_version>/`) so the first
   receipt remains the binding `nn_output_transfer_v1` receipt and
   cannot be overwritten;
7. add the new learner's runner script(s) under `scripts/` (and
   `scripts/lib/` for shared code) in the same commit as the amendment,
   following the freeze-marker-commit pattern in the freeze-marker
   implementation note above;
8. pin a `--lodo-manifest` cross-check that asserts the new learner
   receipt's `featureSchemaVersion`, `protocolVersion`, and
   `receiptSchemaVersion` match the values frozen in Pass A and Pass D,
   so a later arm or schema bump cannot quietly drift the comparison;
9. record any change to seed-slate convention, including new tie-break
   namespace strings, before the first run of the new learner.

The next-learner amendment may not:

- relax the public-evaluation forbid (Phase 6 only) or admit any Kaggle
  work;
- claim sufficiency or non-sufficiency from this first receipt's
  numbers;
- modify the binding `P0_TASK_REGISTER.csv` or the color-role
  quarantine list filed in Pass A;
- overwrite or amend the first receipt's artifact files.

#### Public-Language Implications

The next-learner amendment must pre-register public-language drafts for
every verdict it can produce, in the same format as Pass D. If those
drafts overlap with Pass D's drafts (because the same five verdict
buckets apply), the next-learner amendment may adopt them verbatim and
mark which buckets it newly covers vs reuses.

Any public copy that cites the first receipt remains bound by the Pass D
adopted language and the Pass C coverage-failure disclosure. The
next-learner receipt's public language applies only to that receipt's
verdict, not retroactively to the first.

#### Reserved Naming and Receipt Path

The receipt directory pattern reserved for next-learner work is:

```
results/arc/phase3-sufficiency-<learner_version>/
```

The next-learner amendment must add the chosen subdirectory to
`.gitignore` in the freeze-marker commit for that learner, alongside the
already-ignored `results/arc/phase3-sufficiency/`.

The next-learner runner scripts should follow the existing pattern:

```
scripts/arc-phase3-<learner_version>-lodo.mjs
scripts/arc-phase3-<learner_version>-pttest.mjs
scripts/arc-phase3-<learner_version>-sufficiency.mjs
scripts/lib/arc-phase3-<learner_version>-core.mjs
```

npm script wiring should mirror the existing surface:

```
"arc:phase3:<learner_version>:lodo"
"arc:phase3:<learner_version>:pttest"
"arc:phase3:<learner_version>:sufficiency"
```

This keeps the umbrella `npm run arc:phase3:sufficiency` reserved for the
first receipt (binding) and admits new learners only under their own
namespaces.

### 2026-05-28 -- `nn_delta_transfer_v1` Next-Learner Admission

Author: Codex.

Justification: the next-learner slot reservation requires exactly one
synthesis-capable learner before a second receipt can be generated.
`nn_delta_transfer_v1` is the narrowest admitted follow-up: it keeps the
first learner's nearest-input selection discipline, preserves all Phase A-D
arms, metrics, split handling, thresholds, and receipt schema, and adds only
a deterministic same-shape cell overlay candidate so the candidate pool can
extend beyond the unique conditioning outputs.

Verdict impact: execution is admitted only for a separate
`nn_delta_transfer_v1` receipt after the amendment and runner are committed
together as the new learner freeze marker. The binding
`nn_output_transfer_v1` receipt remains unchanged and may not be overwritten.
This amendment admits no public-evaluation grid inspection, Kaggle work, or
public sufficiency claim before the second receipt is filed and interpreted.

#### Frozen Learner Family

The selected learner family is exactly:

```
nn_delta_transfer_v1
```

The learner has no fitted parameters, optimizer, epoch loop, minibatches,
regularization, learned cross-task weights, or early stopping. It is fully
deterministic except for the existing hash tie-breaker, which remains
deterministic from the master seed. The default master seed remains
`20260528`, and the seed namespace for this learner is
`arc-p3-nn-delta-v1`.

Distinct tie-break hashes are derived from:

```
(master_seed, task_id, lane, query_index, arm:candidate_kind, source_pair_index)
```

This replaces the first learner's tie-break namespace only for the
`nn_delta_transfer_v1` runner. No learned parameters or random samples are
shared across tasks.

#### Candidate Construction

For each LODO or public-training test instance and each representation arm:

1. Represent the query input and every conditioning input under the arm's
   frozen Pass A serialization.
2. Score every conditioning pair by the existing arm-specific input
   distance from Pass C.
3. For every conditioning pair `(input_i, output_i)`, emit candidates in
   this order:
   - `delta_overlay` with rank `0`, only when `query_input`, `input_i`, and
     `output_i` all have the same height and width. The candidate starts as a
     clone of `output_i`. For every cell `(y, x)` where
     `query_input[y][x] != input_i[y][x]`, replace the candidate cell with
     `query_input[y][x]`. If the three grids are not same-shape, this
     candidate is not emitted.
   - `output_copy` with rank `1`, always emitted as a clone of `output_i`.
     This preserves the first learner's support as a fallback while allowing
     the delta candidate to be ranked ahead of it for the same source pair.
4. Represent each emitted candidate grid under the same arm. For
   `signature_palette`, the candidate grid is still decoded through the
   already frozen top-left palette-orbit decoder; no orbit-selection rule is
   changed here.
5. Sort candidates by:
   - ascending input distance;
   - ascending candidate rank (`delta_overlay` before `output_copy`);
   - deterministic tie-break hash;
   - ascending source pair index;
   - lexical candidate kind, as a final unreachable guard.
6. Deduplicate candidates using the existing arm-specific candidate identity
   rule, then emit no more than the top two ordered candidates.

The reported `candidate_pool_size` is the pre-dedup generated candidate count.
`candidate_pool_size_unique` remains the post-dedup support size used by the
existing receipt schema.

#### Metrics, Floors, And Failure Labels

All Pass C metric columns, comparison hierarchy, verdict thresholds, failure
label definitions, and failure-label precedence are unchanged. The existing
LODO-rerun Phase 0 oracle floors remain in the score table. Because
`output_copy` is always present and `delta_overlay` is only an additional
candidate, coverage can only expand relative to the first learner under the
same candidate identity rule; if the target is still outside the generated
candidate pool, the failure label remains `coverage`.

The `k=2` handling, color-role quarantine list, discrimination-floor report,
two-prediction discipline, and public-training test lane are unchanged.

#### Receipt Path And Commands

The new receipt directory is:

```
results/arc/phase3-sufficiency-nn_delta_transfer_v1/
```

The admitted runner files are:

```
scripts/arc-phase3-nn_delta_transfer_v1-lodo.mjs
scripts/arc-phase3-nn_delta_transfer_v1-pttest.mjs
scripts/arc-phase3-nn_delta_transfer_v1-sufficiency.mjs
scripts/lib/arc-phase3-nn_delta_transfer_v1-core.mjs
```

The admitted npm scripts are:

```
npm run arc:phase3:nn_delta_transfer_v1:lodo
npm run arc:phase3:nn_delta_transfer_v1:pttest
npm run arc:phase3:nn_delta_transfer_v1:sufficiency
```

The exact second-receipt command is:

```powershell
npm run arc:phase3:nn_delta_transfer_v1:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-sufficiency-nn_delta_transfer_v1
```

The pttest runner requires `--lodo-manifest` when run standalone and asserts
that `featureSchemaVersion`, `protocolVersion`, `receiptSchemaVersion`, and
`learnerVersion` match the frozen LODO manifest for this learner.

#### Public-Language Drafts

The five Pass D public-language buckets are adopted unchanged for this
learner's future verdict, with one additional required disclosure for any
public copy citing `nn_delta_transfer_v1`:

> "The second Phase 3 learner (`nn_delta_transfer_v1`) is still a
> deliberately low-capacity nearest-neighbor learner. It can synthesize only a
> same-shape cell-overlay candidate plus the original conditioning-output
> fallback, so failures on shape-changing or relational ARC tasks remain
> learner-class limits unless the receipt shows otherwise."

No public copy may combine the first and second receipts into a sufficiency
claim unless the second receipt satisfies the already frozen Pass C support
gate.

### 2026-05-28 -- `nn_delta_transfer_v1` Second-Receipt Verdict

Author: Codex.

Justification: this amendment files the second Phase 3 receipt generated by
the admitted `nn_delta_transfer_v1` follow-up learner. The previous amendment
froze the learner and runner at commit
`2BF53FBDFEBBFF96EC0E80027755F91A276A518D`; this amendment records the clean
run, applies the existing Pass C verdict gates, and binds public language for
this receipt.

Verdict impact: **task hardness / decoder failure** again. The delta learner
expanded the candidate pool enough to produce one active LODO `k>=3` raw-grid
exact hit, but all grid-bearing and representation arms remain below the Pass
C exact-floor thresholds. No support, partial support, palette-dependent
support, or sufficiency-failure conclusion is admitted by this receipt.

#### Run Provenance

Command:

```powershell
npm run arc:phase3:nn_delta_transfer_v1:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-sufficiency-nn_delta_transfer_v1
```

| field | value |
| --- | --- |
| `gitCommit` | `2BF53FBDFEBBFF96EC0E80027755F91A276A518D` |
| `generatedAt` (LODO start) | `2026-05-28T05:34:51.061Z` |
| `completedAt` (pttest end) | `2026-05-28T05:35:05.658Z` |
| wall clock | ~15 s |
| `featureSchemaVersion` | `arc-p3-feature-v1` |
| `protocolVersion` | `arc-p3-protocol-v1` |
| `receiptSchemaVersion` | `arc-p3-receipt-v1` |
| `learnerVersion` | `nn_delta_transfer_v1` |
| `masterSeed` | `20260528` (default; not overridden) |
| `allowDirty` | `false` (clean worktree) |
| `taskCount` | `36` |
| `lodoInstanceCount` | `115` |
| `pttestInstanceCount` | `36` |
| `phase2BaselinesManifestHash` | matches frozen value; no warning |

Receipt artifacts (uppercase SHA-256):

- `results/arc/phase3-sufficiency-nn_delta_transfer_v1/manifest.json`:
  `40CBE4B0600F46BDBD5652BEE49DC2CF71D2F5E7D92539BCFDC161567CBCC5A7`
- `results/arc/phase3-sufficiency-nn_delta_transfer_v1/scores.csv` (66 rows):
  `5ED74A4E060DFDD6D78361FAC8241728B2AE549F47D3E6CCB01D84469B2D6790`
- `results/arc/phase3-sufficiency-nn_delta_transfer_v1/per_instance.csv`
  (1510 rows):
  `57C0FFA0EBF612A2D13B8C349803F9FD0C23D3B8D2583824D42A6859166621D8`
- `results/arc/phase3-sufficiency-nn_delta_transfer_v1/discrimination.csv`:
  `ECF4593FA857BBF89179BB1B12C0E43C6877A2AC2E616A68047F18D8A9E25D02`

Discrimination summary is unchanged from the first receipt:
`trivialTaskCount=2` (`05269061`, `11e1fe23`),
`learnerTaskTrivialThresholdFired=false` (2/36 = 5.6% < 30%).

#### Outcome (LODO `k_ge_3`, 105 instances, 102 active)

| arm | grid_exact | rep_exact_slot1 | shape_exact | palette_exact | pixel_acc_best | rep_sim_best | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `signature_only` | NA | `0.010` | NA | NA | NA | `0.674` | `0.990` |
| `signature_palette` | `0.000` | `0.010` | `0.588` | `0.441` | `0.348` | `0.803` | `0.990` |
| `metadata_only` | NA | `0.010` | `0.608` | `0.431` | NA | `0.942` | `0.941` |
| `raw_grid_lowcap` | `0.010` | `0.010` | `0.598` | `0.431` | `0.439` | `0.920` | `0.990` |

The one active LODO `k>=3` raw-grid exact hit is task `1acc24af`, query index
`3`, under `raw_grid_lowcap`. The same instance reaches representation exact
under `signature_palette`, but the frozen top-left palette-orbit decoder does
not reconstruct the exact grid, so that arm records a residual rather than a
grid exact hit.

Public-training test remains at `0.000` grid exact for both grid-bearing arms
on `k_ge_3`; representation exact stays at or below `0.033`.

#### Verdict Analysis

The Pass C task-hardness / decoder-failure trigger fires:

- every grid-bearing arm has `grid_exact_rate_any_slot < 0.03` on LODO
  `k_ge_3`: `signature_palette = 0.000`,
  `raw_grid_lowcap = 0.010`;
- every representation arm has `rep_exact_rate_slot1 < 0.05` on LODO
  `k_ge_3`: `signature_only = 0.010`,
  `signature_palette = 0.010`, `metadata_only = 0.010`,
  `raw_grid_lowcap = 0.010`.

The usual signature-vs-metadata comparison again reaches a later decisive
metric only after exact and shape/palette metrics fail to decide. On LODO
`k_ge_3`, `signature_palette` and `metadata_only` tie on
`rep_exact_rate_slot1`; shape differs by `0.0196`, below the `0.02`
decisive threshold; palette differs by `0.0098`, below threshold. The first
decisive comparison is `mean_output_rep_similarity_best`, where
`metadata_only = 0.942` and `signature_palette = 0.803`.

As in the first receipt, task hardness preempts any sufficiency-failure claim.
The decisive similarity metric is still operating inside a learner class whose
candidate support is almost entirely missing the held-out outputs. The delta
overlay slightly widens support, but coverage remains `0.990` for
`signature_palette` and `raw_grid_lowcap` on LODO `k_ge_3`, and the exact
rates remain below the Pass C floor. The receipt is therefore another
learner-class / decoder-hardness result, not evidence for or against
signature sufficiency.

#### Adopted Public Language

Per Pass D's pre-registered draft for this verdict:

> "The first Phase 3 runner did not produce an interpretable sufficiency
> verdict because all arms were near the exact-match floor or because
> the LODO held-out outputs collapsed beyond Pass B's discrimination
> threshold; the result is a decoder or task-hardness finding, not
> evidence for or against signature sufficiency."

For this receipt, substitute "second Phase 3 runner" for "first Phase 3
runner" when citing `nn_delta_transfer_v1`.

Any public copy citing this receipt must also include the disclosure filed in
the admission amendment:

> "The second Phase 3 learner (`nn_delta_transfer_v1`) is still a
> deliberately low-capacity nearest-neighbor learner. It can synthesize only a
> same-shape cell-overlay candidate plus the original conditioning-output
> fallback, so failures on shape-changing or relational ARC tasks remain
> learner-class limits unless the receipt shows otherwise."

#### Carry-Forward

Admitted by this amendment:

- the second Phase 3 receipt is filed under the learner-specific path;
- the task-hardness / decoder-failure verdict is the receipt's primary
  verdict;
- the first receipt remains binding for `nn_output_transfer_v1`;
- the `nn_delta_transfer_v1` public language above is the only public copy
  allowed for this receipt.

Still forbidden, unchanged:

- editing arm schemas, split handling, thresholds, or receipt schema without a
  new append-only amendment;
- public-evaluation grid inspection (Phase 6 only);
- Kaggle notebook prep, Kaggle private/semi-private splits;
- any sufficiency, dimensionality, palette-dependent, or partial-support
  claim from this receipt;
- describing either Phase 3 receipt as evidence on the public-evaluation split
  or as an ARC solve.

### 2026-05-28 -- `candidate_combinator_v1` Next-Learner Admission

Author: Claude (Opus 4.7).

Justification: the first two Phase 3 receipts both verdicted task hardness /
decoder failure. The first learner (`nn_output_transfer_v1`) was bounded to
emitting conditioning outputs verbatim; the second (`nn_delta_transfer_v1`)
added a same-shape cell-overlay primitive and produced one active LODO
`k>=3` raw-grid exact hit (`1acc24af` query 3) but coverage stayed at
`0.99`. `candidate_combinator_v1` is the third admitted learner: it stays
deterministic and bounded but materially widens the candidate pool through
D4 variants, per-pair fitted color maps, and cross-pair cell unions of
conditioning outputs. This tests whether enriched deterministic synthesis
crosses the coverage floor without touching the Pass A representation arms
or any Pass B-D contract.

Verdict impact: execution is admitted only for a separate
`candidate_combinator_v1` receipt after this amendment and its runner are
committed together as the learner freeze marker. The binding
`nn_output_transfer_v1` and `nn_delta_transfer_v1` receipts remain
unchanged and may not be overwritten. This amendment admits no
public-evaluation grid inspection, Kaggle work, or public sufficiency claim
before the third receipt is filed and interpreted.

#### Frozen Learner Family

The selected learner family is exactly:

```
candidate_combinator_v1
```

The learner has no fitted parameters, optimizer, epoch loop, minibatches,
regularization, learned cross-task weights, or early stopping. It is
deterministic except for the existing hash tie-breaker, which remains
deterministic from the master seed. The default master seed remains
`20260528`. The seed namespace for this learner is
`arc-p3-candidate-combinator-v1`.

The tie-break hash is derived from:

```
(master_seed, task_id, lane, query_index, arm:candidate_kind, source_pair_index)
```

The same SHA-256-of-namespace + first-eight-bytes-as-big-endian-uint64
construction as Pass C and the second learner; only the namespace string
differs. No learned parameters or random samples are shared across tasks.

#### Candidate Construction

For each LODO or public-training test instance and each representation arm:

1. Represent the query input and every conditioning input under the arm's
   frozen Pass A serialization.
2. Score every conditioning pair by the existing arm-specific input
   distance from Pass C.
3. For every conditioning pair `(input_i, output_i)`, emit the following
   candidates in this order:

   | candidate_kind | rank | emit condition | construction |
   | --- | ---: | --- | --- |
   | `colormap_fit` | `0` | `input_i` and `output_i` share shape AND a bijective color map `M : color -> color` exists with `M(input_i[y][x]) = output_i[y][x]` for every cell | apply `M` to `query_input` (colors not in `M`'s domain pass through unchanged); emit the resulting grid |
   | `d4_rot90` | `1` | always | `rot90(output_i)` |
   | `d4_rot180` | `2` | always | `rot180(output_i)` |
   | `d4_rot270` | `3` | always | `rot270(output_i)` |
   | `d4_reflect_h` | `4` | always | `reflectHorizontal(output_i)` |
   | `d4_reflect_v` | `5` | always | `reflectVertical(output_i)` |
   | `d4_transpose` | `6` | always | `transpose(output_i)` |
   | `d4_anti_transpose` | `7` | always | `antiTranspose(output_i)` |
   | `cell_union_<j>` | `8 + j` | `j != i`; `output_i` and `output_j` share shape | for every cell `(y, x)`: emit `output_i[y][x]` if non-zero, else `output_j[y][x]` |
   | `output_copy` | `100` | always | clone of `output_i` (preserves the `nn_output_transfer_v1` fallback so coverage cannot regress) |

   `j` indexes other conditioning pairs by their original train-pair index.
   `d4_id` is not emitted as a separate candidate because it equals
   `output_copy`. The D4 transforms reuse the same orientation set named in
   Pass A and the existing core.

4. Represent each emitted candidate grid under the same arm. For
   `signature_palette`, the candidate grid is still decoded through the
   already frozen `top_left_palette_orbit_v1` decoder; no orbit-selection
   rule is changed here.

5. Sort candidates by, in order:

   - ascending input distance from query input to the candidate's source
     conditioning pair input;
   - ascending `candidate_kind` rank from the table above;
   - deterministic tie-break hash with the new namespace;
   - ascending `source_pair_index` (`i`);
   - lexical `candidate_kind` string, as a final unreachable guard.

6. Deduplicate candidates using the existing arm-specific candidate
   identity rule from Pass C; keep the first occurrence under the sort.
   Then emit no more than the top two ordered candidates per Pass B.

`candidate_pool_size` reports the pre-dedup generated candidate count;
`candidate_pool_size_unique` remains the post-dedup support size.

For `k = 2` tasks the cross-pair `cell_union` slot enumerates at most one
partner pair; for `k = 5` it enumerates four. With `k = 3`, the maximum
generated pool per arm per instance is `3 * (1 + 7 + 2 + 1) = 33`
candidates before dedup.

#### Metrics, Floors, And Failure Labels

All Pass C metric columns, comparison hierarchy, verdict thresholds,
failure label definitions, and failure-label precedence are unchanged. The
existing LODO-rerun Phase 0 oracle floors remain in the score table.
Because `output_copy` is always present, coverage can only expand relative
to the first learner under the same candidate identity rule; coverage may
also expand relative to the second learner because the cross-pair
`cell_union` primitive can match held-out outputs that neither
`output_copy` alone nor `delta_overlay` alone produces. If the target is
still outside the generated candidate pool after all six primitive
families fire, the failure label remains `coverage`.

The `k=2` handling, color-role quarantine list, discrimination-floor
report, two-prediction discipline, and public-training test lane are
unchanged. The discrimination CSV is again expected to be byte-identical
to the first two receipts because discrimination is purely a function of
the data and the register.

#### Receipt Path And Commands

The new receipt directory is:

```
results/arc/phase3-sufficiency-candidate_combinator_v1/
```

This directory must be added to `.gitignore` in the freeze-marker commit
for this learner, alongside the already-ignored
`results/arc/phase3-sufficiency/` and
`results/arc/phase3-sufficiency-nn_delta_transfer_v1/`.

The admitted runner files are:

```
scripts/arc-phase3-candidate_combinator_v1-lodo.mjs
scripts/arc-phase3-candidate_combinator_v1-pttest.mjs
scripts/arc-phase3-candidate_combinator_v1-sufficiency.mjs
scripts/lib/arc-phase3-candidate_combinator_v1-core.mjs
```

The admitted npm scripts are:

```
"arc:phase3:candidate_combinator_v1:lodo"
"arc:phase3:candidate_combinator_v1:pttest"
"arc:phase3:candidate_combinator_v1:sufficiency"
```

The exact third-receipt command is:

```powershell
npm run arc:phase3:candidate_combinator_v1:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-sufficiency-candidate_combinator_v1
```

The pttest runner requires `--lodo-manifest` when run standalone and
asserts that `featureSchemaVersion`, `protocolVersion`,
`receiptSchemaVersion`, and `learnerVersion` match the frozen LODO
manifest for this learner. The leak-check allowlist must be extended to
include the lane runners' refuse-to-read guard literals, mirroring the
nn_delta_transfer_v1 freeze commit.

#### Public-Language Drafts

The five Pass D public-language buckets are adopted unchanged for this
learner's future verdict, with one additional required disclosure for any
public copy citing `candidate_combinator_v1`:

> "The third Phase 3 learner (`candidate_combinator_v1`) is still a
> deliberately low-capacity nearest-neighbor learner. It generates
> candidates only from frozen deterministic primitives applied to
> conditioning outputs (D4 variants, per-pair fitted color maps,
> cross-pair cell unions, and the original output-copy fallback) and
> cannot synthesize candidates that require relational reasoning across
> multiple input-output pairs, programmatic composition, or learned
> representations. Failures on those task classes remain learner-class
> limits unless the receipt shows otherwise."

No public copy may combine the first, second, and third receipts into a
sufficiency claim unless one of the receipts satisfies the already frozen
Pass C support gate. Receipts may be cited together to characterize the
deterministic-learner family's joint coverage of the registered subset;
the framing must remain "learner-class characterization", not a
sufficiency claim about the representation arms.

### 2026-05-28 -- `candidate_combinator_v1` Clarification: output_copy Rank

Author: Claude (Opus 4.7).

Justification: a pre-freeze-marker smoke run of the admitted
`candidate_combinator_v1` runner exposed a structural regression caused
by the candidate ranking. The original amendment placed `output_copy` at
rank `100` as a fallback, intending to "preserve the
nn_output_transfer_v1 fallback so coverage cannot regress." This
preserves COVERAGE (output_copy stays in the unique pool) but does
**not** preserve slot-1 / slot-2 SELECTION: for source pairs where
output_copy is the correct answer, synthesis candidates at ranks 0-8+
displace output_copy out of the top two, causing a paradoxical
exact-rate regression vs nn_output_transfer_v1 on arms (notably
`raw_grid_lowcap`) that previously credited output_copy.

This clarification revises `output_copy`'s rank to `-1`, placing it
ahead of every synthesis primitive within the same source pair. Slot 1
for the nearest pair is now always output_copy; slot 2 is the
highest-ranked synthesis primitive from the same pair. This preserves
both the nn_output_transfer_v1 baseline AND tests synthesis as a
secondary candidate, matching the original amendment's stated intent.

Verdict impact: no execution admission. The candidate_combinator_v1
freeze marker has not yet been committed (runner files are pre-commit).
This clarification revises the contract before the binding receipt is
generated. The runner must be updated so that the freeze-marker commit
lands with the corrected ranking.

#### Revised Candidate Ranking

The complete candidate-rank table for `candidate_combinator_v1` is now:

| candidate_kind | rank | emit condition | construction |
| --- | ---: | --- | --- |
| `output_copy` | `-1` | always | clone of `output_i` (slot-1 guard preserving the nn_output_transfer_v1 baseline) |
| `colormap_fit` | `0` | (unchanged) | (unchanged) |
| `d4_rot90` | `1` | always | `rot90(output_i)` |
| `d4_rot180` | `2` | always | `rot180(output_i)` |
| `d4_rot270` | `3` | always | `rot270(output_i)` |
| `d4_reflect_h` | `4` | always | `reflectHorizontal(output_i)` |
| `d4_reflect_v` | `5` | always | `reflectVertical(output_i)` |
| `d4_transpose` | `6` | always | `transpose(output_i)` |
| `d4_anti_transpose` | `7` | always | `antiTranspose(output_i)` |
| `cell_union_<j>` | `8 + j` | (unchanged) | (unchanged) |

All other contract elements remain unchanged: Pass A representation
arms, Pass B instance handling and strata, Pass C metric hierarchy and
verdict thresholds, Pass D receipt schema, the seed namespace
`arc-p3-candidate-combinator-v1`, all runner script names and npm script
names, and the receipt directory path
`results/arc/phase3-sufficiency-candidate_combinator_v1/`.

#### Expected Effect

With output_copy at rank `-1`:

- slot 1 for every `(instance, arm)` is now the conditioning output from
  the nearest pair under arm distance -- identical to
  `nn_output_transfer_v1`'s slot 1 emission for the same instance;
- slot 2 is the highest-ranked synthesis candidate from the same
  nearest pair (`colormap_fit` if applicable, else `d4_rot90`, then
  `d4_rot180`, and so on);
- coverage is unchanged (full pool is the same set);
- exact rates can only equal or exceed `nn_output_transfer_v1`'s, never
  regress: any slot-1 exact hit `nn_output_transfer_v1` had is
  preserved, and slot 2 adds additional opportunities to credit
  synthesis candidates that `nn_output_transfer_v1` lacked.

This restores the "candidate_combinator_v1 cannot regress vs
nn_output_transfer_v1" guarantee that the original amendment intended
but did not enforce through the rank-100 ordering.

#### Receipt Continuity

Any pre-clarification smoke artifacts under
`results/arc/phase3-sufficiency-candidate_combinator_v1/` from the
original rank-100 ordering are not the binding receipt. The binding
receipt requires:

1. this clarification amendment committed;
2. the runner core updated so that `output_copy` is emitted with
   `rank: -1` in `candidateCombinatorCandidates`;
3. both committed in the freeze-marker commit alongside `package.json`,
   `.gitignore`, and the three thin entry scripts;
4. a fresh `npm run arc:phase3:candidate_combinator_v1:sufficiency`
   invocation without `--allow-dirty` to generate the binding receipt.

### 2026-05-28 -- `candidate_combinator_v1` Clarification: delta_overlay Primitive Addition

Author: Claude (Opus 4.7).

Justification: a second pre-freeze-marker smoke run, after the previous
output_copy clarification, confirmed that
`candidate_combinator_v1`'s slot-1 receipt now exactly equals
`nn_output_transfer_v1`'s slot-1 receipt on every learner arm
(`rep_exact_slot1` matches `0.000 / 0.000 / 0.039 / 0.000` for
`signature_only / signature_palette / metadata_only / raw_grid_lowcap`
respectively). However, the smoke trace showed
`candidate_combinator_v1` cannot reproduce
`nn_delta_transfer_v1`'s one `1acc24af` query-3 `raw_grid_lowcap` exact
hit because the slot-1 candidate there is the conditioning-output (a
non-zero grid) while the actual held-out target is the all-zero grid
that `nn_delta_transfer_v1`'s `delta_overlay` primitive produced by
overlaying query-input zeros onto the conditioning output. No
`candidate_combinator_v1` primitive (`output_copy`, `colormap_fit`, D4
variants, `cell_union`) can synthesize the all-zero grid from this
conditioning data; the hit is structurally outside the original
primitive set.

This clarification adds `delta_overlay` to
`candidate_combinator_v1`'s primitive set as a fifth family, with rank
`-1` between `output_copy` (rank `-2`) and `colormap_fit` (rank `0`).
The construction is identical to `nn_delta_transfer_v1`'s
`delta_overlay`: same-shape required across `(query_input, input_i,
output_i)`; start from a clone of `output_i`; for every cell `(y, x)`
where `query_input[y][x] != input_i[y][x]`, replace the candidate cell
with `query_input[y][x]`. This makes `candidate_combinator_v1` a
strict superset of both `nn_output_transfer_v1` (slot-1 = output_copy)
and `nn_delta_transfer_v1` (slot-2 = delta_overlay when emitted).

Scope acknowledgement: this is an honest expansion of the original
`candidate_combinator_v1` primitive set rather than a typo or bug fix.
The original admission amendment specified four primitive families
(`output_copy`, `colormap_fit`, D4, `cell_union`); this clarification
makes it five. Filing as a clarification amendment rather than a new
learner version (`candidate_combinator_v2`) is permitted under the
Pass-A-clarification convention used elsewhere because the contract is
still pre-freeze-marker: no committed runner, no binding receipt, no
verdict yet. After this clarification commits with the runner, no
further additions to the primitive set are allowed without a new
learner version.

Verdict impact: no execution admission. The freeze marker has not yet
been committed.

#### Revised Candidate Ranking (Full Table)

| candidate_kind | rank | emit condition | construction |
| --- | ---: | --- | --- |
| `output_copy` | `-2` | always | clone of `output_i` |
| `delta_overlay` | `-1` | `query_input`, `input_i`, `output_i` all share shape | clone of `output_i`; for each `(y, x)` with `query_input[y][x] != input_i[y][x]`, replace candidate cell with `query_input[y][x]` |
| `colormap_fit` | `0` | `input_i` and `output_i` share shape AND a bijective `M : color -> color` exists with `M(input_i[y][x]) = output_i[y][x]` for every cell | apply `M` to `query_input` (out-of-domain colors pass through) |
| `d4_rot90` | `1` | always | `rot90(output_i)` |
| `d4_rot180` | `2` | always | `rot180(output_i)` |
| `d4_rot270` | `3` | always | `rot270(output_i)` |
| `d4_reflect_h` | `4` | always | `reflectHorizontal(output_i)` |
| `d4_reflect_v` | `5` | always | `reflectVertical(output_i)` |
| `d4_transpose` | `6` | always | `transpose(output_i)` |
| `d4_anti_transpose` | `7` | always | `antiTranspose(output_i)` |
| `cell_union_<j>` | `8 + j` | `j != i`; `output_i` and `output_j` share shape | per cell: `output_i[y][x]` if non-zero, else `output_j[y][x]` |

All other contract elements remain unchanged: arms, distances, seed
namespace, runner script names, npm script names, receipt directory
path.

#### Expected Effect

With `output_copy` at rank `-2` and `delta_overlay` at rank `-1`:

- slot 1 for every `(instance, arm)` is always `output_copy` of the
  nearest pair under arm distance -- still identical to
  `nn_output_transfer_v1`'s slot 1;
- slot 2 is the highest-ranked further candidate from the same nearest
  pair: `delta_overlay` when it can be emitted, else `colormap_fit` if
  it fits, else `d4_rot90`, and so on;
- coverage and full-pool size both expand by up to one candidate per
  same-shape source pair (delta_overlay only emits when shapes match);
- exact rates can only equal or exceed both `nn_output_transfer_v1`'s
  and `nn_delta_transfer_v1`'s slot-1 + slot-2 numbers on the same
  registered subset, never regress against either.

The 1acc24af query-3 `raw_grid_lowcap` hit specifically is expected to
return as a `grid_exact_any_slot = true` result via slot 2; slot 1
will remain `output_copy` (non-zero, wrong), and `grid_exact_slot1`
will stay `false` for that instance.

#### Public-Language Update

The candidate_combinator_v1 admission amendment's public-language
disclosure listed the primitive families as "D4 variants, per-pair
fitted color maps, cross-pair cell unions, and the original
output-copy fallback." This clarification updates the disclosure to:

> "The third Phase 3 learner (`candidate_combinator_v1`) is still a
> deliberately low-capacity nearest-neighbor learner. It generates
> candidates only from frozen deterministic primitives applied to
> conditioning outputs and the query input: the output-copy fallback,
> a same-shape delta-overlay synthesis, per-pair fitted bijective color
> maps, the eight D4 orientation variants, and cross-pair cell unions.
> It cannot synthesize candidates that require relational reasoning
> across multiple input-output pairs beyond the pairwise cell union,
> programmatic composition, or learned representations. Failures on
> those task classes remain learner-class limits unless the receipt
> shows otherwise."

This adoption supersedes the prior disclosure for any public copy
citing `candidate_combinator_v1`.

### 2026-05-28 -- `candidate_combinator_v1` Third-Receipt Verdict

Author: Codex.

Justification: this amendment files the third Phase 3 receipt generated by
the admitted `candidate_combinator_v1` learner after the output-copy-rank and
delta-overlay clarifications were committed with the runner. The freeze-marker
commit is `F19B2E3BA45551DEC231BA4D47D786389F0FD58B`. This amendment records
the clean run, corrects one non-metric wording error from the previous
clarification, applies the existing Pass C verdict gates, and binds public
language for this receipt.

Verdict impact: **task hardness / decoder failure** again. The combinator is
a strict-superset learner over the two earlier finite-candidate lanes for the
registered task subset, but exact rates remain below the Pass C floor. No
support, partial support, palette-dependent support, or sufficiency-failure
conclusion is admitted by this receipt.

#### Correction To Prior Clarification

The `delta_overlay` clarification described the `1acc24af` query-3 held-out
target as "the all-zero grid." That wording is incorrect. The held-out target
is a `12x12` grid with palette `0,1,2,5` and 23 nonzero cells. The operational
claim remains correct: under `raw_grid_lowcap`, slot 1 is the nearest-pair
`output_copy` and is not exact; slot 2 is the `delta_overlay` candidate and is
exact, so `grid_exact_slot1=false`, `grid_exact_any_slot=true`, and
`failure_label=none` for that instance.

This correction has no metric impact and does not change the frozen candidate
construction.

#### Run Provenance

Command:

```powershell
npm run arc:phase3:candidate_combinator_v1:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-sufficiency-candidate_combinator_v1
```

| field | value |
| --- | --- |
| `gitCommit` | `F19B2E3BA45551DEC231BA4D47D786389F0FD58B` |
| `generatedAt` (LODO start) | `2026-05-28T09:24:46.405Z` |
| `completedAt` (pttest end) | `2026-05-28T09:25:16.382Z` |
| wall clock | ~30 s |
| `featureSchemaVersion` | `arc-p3-feature-v1` |
| `protocolVersion` | `arc-p3-protocol-v1` |
| `receiptSchemaVersion` | `arc-p3-receipt-v1` |
| `learnerVersion` | `candidate_combinator_v1` |
| `masterSeed` | `20260528` (default; not overridden) |
| `allowDirty` | `false` (clean worktree) |
| `taskCount` | `36` |
| `lodoInstanceCount` | `115` |
| `pttestInstanceCount` | `36` |
| `phase2BaselinesManifestHash` | matches frozen value; no warning |

Receipt artifacts (uppercase SHA-256):

- `results/arc/phase3-sufficiency-candidate_combinator_v1/manifest.json`:
  `FCFF1144886436C43A5B0FB6F878D590E0F1DAC03C1717E1F44299C451D0F7A1`
- `results/arc/phase3-sufficiency-candidate_combinator_v1/scores.csv`
  (66 rows):
  `98D52FBAFB7EAAB96CCD672579D4529516E11C94388E67994587E59A15E2E9F6`
- `results/arc/phase3-sufficiency-candidate_combinator_v1/per_instance.csv`
  (1510 rows):
  `358BB9FB27BD1DC9ABA09E3C30C43288AE520683292EB9ADED0B78737E3CB514`
- `results/arc/phase3-sufficiency-candidate_combinator_v1/discrimination.csv`:
  `ECF4593FA857BBF89179BB1B12C0E43C6877A2AC2E616A68047F18D8A9E25D02`

Discrimination summary is unchanged from the previous receipts:
`trivialTaskCount=2` (`05269061`, `11e1fe23`),
`learnerTaskTrivialThresholdFired=false` (2/36 = 5.6% < 30%).

#### Outcome (LODO `k_ge_3`, 105 instances, 102 active)

| arm | grid_exact | rep_exact_slot1 | rep_exact_any | shape_exact | palette_exact | pixel_acc_best | rep_sim_best | coverage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `signature_only` | NA | `0.000` | `0.010` | NA | NA | NA | `0.652` | `0.990` |
| `signature_palette` | `0.000` | `0.000` | `0.010` | `0.588` | `0.480` | `0.334` | `0.789` | `0.990` |
| `metadata_only` | NA | `0.039` | `0.049` | `0.608` | `0.480` | NA | `0.942` | `0.941` |
| `raw_grid_lowcap` | `0.010` | `0.000` | `0.010` | `0.598` | `0.461` | `0.433` | `0.922` | `0.990` |

This receipt preserves the first learner's slot-1 behavior while adding the
delta learner's slot-2 opportunity. The observed LODO `k_ge_3` convergence is:

| metric | `nn_output_transfer_v1` | `nn_delta_transfer_v1` | `candidate_combinator_v1` |
| --- | ---: | ---: | ---: |
| `raw_grid_lowcap.grid_exact_any_slot` | `0.000` | `0.010` | `0.010` |
| `metadata_only.rep_exact_slot1` | `0.039` | `0.010` | `0.039` |
| `metadata_only.rep_exact_any_slot` | `0.039` | `0.020` | `0.049` |

Public-training test remains at `0.000` grid exact for both grid-bearing arms
on `k_ge_3`; representation exact stays at or below `0.033` for slot 1.

#### Verdict Analysis

The Pass C task-hardness / decoder-failure trigger fires:

- every grid-bearing arm has `grid_exact_rate_any_slot < 0.03` on LODO
  `k_ge_3`: `signature_palette = 0.000`,
  `raw_grid_lowcap = 0.010`;
- every representation arm has `rep_exact_rate_slot1 < 0.05` on LODO
  `k_ge_3`: `signature_only = 0.000`,
  `signature_palette = 0.000`, `metadata_only = 0.039`,
  `raw_grid_lowcap = 0.000`.

The signature-vs-metadata comparison again reaches a later decisive metric
only after exact and shape/palette metrics fail to decide. On LODO `k_ge_3`,
`signature_palette` trails `metadata_only` on `rep_exact_slot1` by `0.039`,
below the Pass C decisive threshold; shape differs by `0.0196`, below the
`0.02` decisive threshold; palette ties at `0.480`. The first decisive
comparison is `mean_output_rep_similarity_best`, where
`metadata_only = 0.942` and `signature_palette = 0.789`.

As in the first two receipts, task hardness preempts any sufficiency-failure
claim. The decisive similarity gap is still measured inside a learner class
whose candidate support misses nearly all held-out outputs. The expanded
candidate pool restores the expected slot-2 exact hit and improves
`metadata_only.rep_exact_any_slot`, but coverage remains `0.990` for
`signature_palette` and `raw_grid_lowcap` on LODO `k_ge_3`, and exact rates
remain below the Pass C floor. The receipt is therefore another
learner-class / decoder-hardness result, not evidence for or against
signature sufficiency.

#### Adopted Public Language

Per Pass D's pre-registered draft for this verdict:

> "The first Phase 3 runner did not produce an interpretable sufficiency
> verdict because all arms were near the exact-match floor or because
> the LODO held-out outputs collapsed beyond Pass B's discrimination
> threshold; the result is a decoder or task-hardness finding, not
> evidence for or against signature sufficiency."

For this receipt, substitute "third Phase 3 runner" for "first Phase 3
runner" when citing `candidate_combinator_v1`.

Any public copy citing this receipt must also use the corrected form of the
disclosure filed in the delta-overlay clarification:

> "The third Phase 3 learner (`candidate_combinator_v1`) is still a
> deliberately low-capacity nearest-neighbor learner. It generates
> candidates only from frozen deterministic primitives applied to
> conditioning outputs and the query input: the output-copy fallback,
> a same-shape delta-overlay synthesis, per-pair fitted bijective color
> maps, the seven non-identity D4 orientation variants, and cross-pair cell
> unions. The D4 identity case is the output-copy fallback.
> It cannot synthesize candidates that require relational reasoning
> across multiple input-output pairs beyond the pairwise cell union,
> programmatic composition, or learned representations. Failures on
> those task classes remain learner-class limits unless the receipt
> shows otherwise."

#### Carry-Forward

Admitted by this amendment:

- the third Phase 3 receipt is filed under the learner-specific path;
- the task-hardness / decoder-failure verdict is the receipt's primary
  verdict;
- the first two receipts remain binding for their own learner versions;
- the `candidate_combinator_v1` public language above is the only public copy
  allowed for this receipt.

Still forbidden, unchanged:

- editing arm schemas, split handling, thresholds, or receipt schema without a
  new append-only amendment;
- public-evaluation grid inspection (Phase 6 only);
- Kaggle notebook prep, Kaggle private/semi-private splits;
- any sufficiency, dimensionality, palette-dependent, or partial-support
  claim from this receipt;
- describing any Phase 3 receipt as evidence on the public-evaluation split or
  as an ARC solve.

### 2026-05-28 -- Blackwell Sufficiency Unpause Spec

Author: Codex, from operator-provided Blackwell sufficiency work order.

Justification: Phase 3.5 paused the deterministic-low-capacity learner branch
after three task-hardness / decoder-failure receipts. The operator has selected
the next clean unpause path: re-state the core claim as a finite Blackwell
sufficiency test over the registered demonstrations, then admit a held-out-task
decoder design only after the algebraic conditions, splits, metrics, and
quarantine taxonomy are frozen. This amendment is the algebraic receipt and
experiment-design receipt. It does not run a decoder and it does not change any
prior receipt verdict.

Verdict impact: Phase 3 is unpaused for **Blackwell-spec implementation only**.
Decoder execution remains held until a runner, npm command wiring, and
freeze-marker commit implement this amendment. The three existing Phase 3
receipts remain task-hardness / decoder-failure receipts and may not be
reinterpreted as support or failure.

#### Core Question

The Phase 3 Blackwell question is:

> Does `signature_palette` -- or, on the stricter quotient, `signature_only` --
> contain enough information to learn the registered demonstration-conditioned
> mapping `input_rep -> output_rep` for the registered core-knowledge task class,
> with held-out-task generalization indistinguishable from the full-grid control?

The full-grid control is the already frozen `raw_grid_lowcap` serialization from
Pass A. In this amendment it is the "full projected grid" arm: it receives the
complete registered grid tensor under the same learner family and budget as the
signature arms. The arm remains named `raw_grid_lowcap` in receipt tables for
schema continuity.

#### Algebraic Formulation

Let `G` be the finite set of ARC grids admitted by the registered public-training
tasks, and let a task be a deterministic latent rule `theta` drawn from the
registered class `Theta_P` for prior stratum `P`.

For a query instance:

- `D = ((X_1,Y_1),...,(X_k,Y_k))` is the demonstration set;
- `X_q` is the query input;
- `Y_q = f_theta(X_q; D)` is the target output;
- `O_full = (D, X_q)` is the full-grid experiment;
- `sigma_a` is the frozen representation map for arm `a`;
- `O_a = sigma_a(O_full)` is the arm experiment, applying the same arm map to
  every demonstration input, demonstration output, and query input.

Because `O_a` is a deterministic garbling of `O_full`, the full-grid experiment
is always weakly more informative. Arm `a` is Blackwell sufficient for the
mapping on the registered task class only if the garbling loses no information
needed by the output decision problem:

```text
Pr(Y_q | O_full, P) = Pr(Y_q | O_a, P)
```

for every registered prior stratum `P`, up to the declared output quotient for
the arm. Equivalently:

```text
I(Y_q ; O_full | O_a, P) = 0
```

inside the registered finite experiment. In deterministic ARC language this
reduces to a fiber condition:

```text
sigma_a(O_full) = sigma_a(O'_full)  =>  Y_q == Y'_q
```

for exact-grid sufficiency, or:

```text
sigma_a(O_full) = sigma_a(O'_full)  =>  tau_a(Y_q) == tau_a(Y'_q)
```

for representation-level sufficiency, where `tau_a` is the output quotient
declared for that arm. A held-out-task decoder is an empirical attempt to learn
the measurable decision rule:

```text
d_a: sigma_a(D), sigma_a(X_q) -> Y_q
```

or, when exact grid decoding is not well-defined:

```text
h_a: sigma_a(D), sigma_a(X_q) -> tau_a(Y_q)
```

The information that must be preserved by a signature is therefore not "the
whole grid"; it is the equivalence class of full-grid observations that leaves
the conditional output rule invariant. A sufficient signature must preserve:

- all latent rule parameters identifiable from demonstrations and needed at
  query time;
- object topology, local relations, shape envelope, and palette/color-role
  information whenever those variables change the output;
- enough orientation and symmetry phase to choose the correct representative
  inside a gauge orbit;
- enough global context for completion tasks whose local bags alone do not
  identify the fill rule;
- a deterministic decoder from predicted output representation to exact grid,
  or an explicitly declared output quotient that makes exact-grid scoring
  inapplicable.

Immediate structural zeros and quarantines:

- `signature_only` quotients absolute color identity. Exact-grid Blackwell
  sufficiency for primary color-role tasks is structurally zero unless the
  receipt uses a declared color-quotient metric. The pre-registered color-role
  quarantine is unchanged: `08ed6ac7`, `0a2355a6`, `2601afb7`, `292dd178`,
  `37d3e8b2`, `3ad05f52`.
- `signature_only` still has no admitted deterministic exact-grid decoder under
  Pass A. It can support only representation-level or quotient-level claims
  until a later append-only amendment admits a decoder.
- The three deterministic-low-capacity receipts are not Blackwell receipts
  because their candidate pools miss almost every held-out output. They remain
  learner-family receipts and cannot close Branch A, B, or C below.
- Phase 2's high shadow train-pair residual (`0.594295`) forbids a small-delta
  assumption. The next learner must synthesize outputs from the represented
  demonstration-conditioned mapping; copy/nearest-output transfer is already
  exhausted.

#### Branch Criteria

These branches supersede no older receipt; they govern only the future
Blackwell runner admitted by this amendment.

| branch | condition | interpretation |
| --- | --- | --- |
| Branch A -- clean structural zero / strong sufficiency | `signature_palette` clears the non-trivial floor and is statistically indistinguishable from, or better than, `raw_grid_lowcap` on held-out-task LODO and held-out-task public-training test. `signature_only` may independently satisfy this branch only on representation-level or declared quotient metrics. | Signature is Blackwell sufficient for the registered task class under the tested learner family and quotient. |
| Branch B -- named quarantine | A gap exists, but at least 80% of the gap mass is assigned before adjudication to one or more pre-registered quarantine categories below, and the non-quarantined slice satisfies Branch A. | The sufficiency claim is narrowed to the named non-quarantined class. |
| Branch C -- bounded failure | `signature_palette` fails to clear the non-trivial floor on the easiest registered subset, or materially trails `raw_grid_lowcap` on the non-quarantined slice with an explicit reconstruction residual. | Signature is not sufficient for this registered Blackwell test. |

The easiest registered subset is the union of tasks not marked
`full-state-only dependency (residual category)` and not primary
`color_role`, evaluated on the held-out-task test split. If that subset has
fewer than four active tasks after runtime failures, no Branch C verdict is
admitted; the run is an execution failure.

Statistical comparison is paired by task ID. The runner must report:

- task-level bootstrap confidence intervals with `10000` resamples over task
  IDs and master seed recorded in the manifest;
- a practical equivalence margin of `0.05` absolute exact-rate difference for
  Branch A;
- a material-trailing threshold of `0.05` absolute exact-rate difference or a
  non-overlapping paired bootstrap interval, whichever is stricter for the
  observed sample size.

The non-trivial floor is:

- exact-grid: greater than the Phase 0 cheap-baseline exact floor (`0/36`) and
  greater than the best filed deterministic Phase 3 exact floor on comparable
  LODO `k_ge_3` instances (`0.010`), with at least two exact successes from
  distinct held-out tasks;
- representation-level: greater than the best deterministic Phase 3
  `rep_exact_any` rate on comparable LODO `k_ge_3` instances (`0.049`), with
  at least two exact representation successes from distinct held-out tasks.

#### Pass A -- Representation Freeze

This Blackwell runner uses the existing frozen `arc-p3-feature-v1`
serialization. No arm schema change is admitted here.

| arm | Blackwell role | scoring note |
| --- | --- | --- |
| `raw_grid_lowcap` | full projected grid control | exact-grid and representation metrics |
| `signature_palette` | primary Sundog statistic | exact-grid metrics through the admitted decoder plus representation metrics |
| `signature_only` | strict quotient statistic | representation/quotient metrics primary; exact-grid color claims quarantined |
| `metadata_only` | coarse nuisance control | reported for discrimination only; cannot support Branch A |

Any later runner implementation must fail fast if the manifest does not report
`featureSchemaVersion=arc-p3-feature-v1`.

#### Pass B -- Held-Out-Task Splits, Floors, And Metrics

The 36 registered tasks are split by primary prior, sorted by task ID within
each prior: first four train, fifth validation, sixth test. This yields 24
train tasks, 6 validation tasks, and 6 held-out test tasks.

| prior | train tasks | validation task | test task |
| --- | --- | --- | --- |
| `color_role` | `08ed6ac7`, `0a2355a6`, `2601afb7`, `292dd178` | `37d3e8b2` | `3ad05f52` |
| `counting` | `009d5c81`, `00dbd492`, `025d127b`, `045e512c` | `05269061` | `05a7bcf2` |
| `local_completion` | `03560426`, `05f2a901`, `0b17323b`, `0e671a1a` | `11e1fe23` | `13713586` |
| `objectness` | `11dc524f`, `150deff5`, `1acc24af`, `1b60fb0c` | `2bee17df` | `3906de3d` |
| `spatial_transform` | `00576224`, `0a1d4ef5`, `0b148d64`, `0bb8deee` | `0c9aba6e` | `137eaa0f` |
| `symmetry` | `007bbfb7`, `00d62c1b`, `017c7c7b`, `0520fde7` | `0692e18c` | `0a938d79` |

Training instances are LODO instances from train tasks only. Validation
instances are LODO instances from validation tasks only and are used for early
stopping/model selection. Test instances are:

1. held-out-task LODO instances from the six test tasks;
2. public-training test inputs for the six test tasks, run only after the LODO
   test protocol and model selection are frozen.

No gradient update, hyperparameter selection, or architecture change may use
validation or test task targets except through the declared validation early-stop
rule. No public-evaluation grid is admitted.

Primary metrics:

- `grid_exact_top1`;
- `grid_exact_any_slot` with the existing two-prediction discipline;
- `rep_exact_top1`;
- `rep_exact_any_slot`.

Secondary metrics:

- pixel accuracy;
- shape exact;
- palette exact;
- output-representation cosine similarity;
- per-prior and per-`predicted_boundary` breakdowns;
- residual class: `coverage`, `detection`, `residual`, `structural_zero`,
  `execution_failure`, or `none`.

Floor rows must include Phase 0 cheap baselines, the three filed deterministic
Phase 3 receipts, and a majority-output-representation control computed on the
new train split.

#### Pass C -- Learner Contract

The first Blackwell learner family is `blackwell_task_decoder_v1`.

It is a small set-conditioned decoder trained from scratch across train tasks,
separately for each arm and seed:

- input example: padded sequence of up to five demonstration pairs plus one
  query input;
- per-grid feature: the frozen Pass A arm vector for representation arms, or
  the frozen raw grid tensor for `raw_grid_lowcap`;
- token types: `demo_input`, `demo_output`, `query_input`, and `pad`;
- architecture: 2 transformer encoder layers, `d_model=128`, 4 attention
  heads, feed-forward width `256`, dropout `0.10`;
- output heads: `height` class (`1..30`), `width` class (`1..30`), and
  `30x30x10` color logits;
- loss: `1.0 * masked_cell_cross_entropy + 0.25 * height_cross_entropy +
  0.25 * width_cross_entropy`; representation heads may be added only if they
  are reported as auxiliary and use the same weight for every arm;
- optimizer: AdamW, learning rate `0.0003`, weight decay `0.0001`;
- batch size: `16`;
- maximum epochs: `400`;
- early stop: patience `40` epochs on validation `grid_exact_any_slot`, with
  validation loss as deterministic tie-break;
- seed slate: `20260528`, `20260529`, `20260530`, `20260531`, `20260601`;
- top-2 candidates: top cell-logit grid after predicted shape, plus the best
  alternative from a single deterministic cellwise margin flip pass; tie-break
  by lower validation loss, then seed, then arm name.

The same architecture, optimizer, epoch cap, seed slate, and early-stop rule
apply to every arm. If a representation arm cannot emit an exact grid under
Pass A, the runner must still emit representation/quotient metrics and mark the
grid metric `NA` rather than zero.

The full five-seed run is expected to be cheap, but the repository's
ten-minute rule still applies. If a one-seed probe projects the full run above
ten minutes, the runner implementation receipt must stage the exact PowerShell
command instead of running the full slate inline, and it must record the
measured seconds/epoch and extrapolated wall clock.

#### Pass D -- Receipts, Commands, And Quarantine Log

Reserved output path:

```text
results/arc/phase3-blackwell-sufficiency-v1/
```

Reserved command:

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1
```

The command is not admitted until the implementation lands with package wiring
and a freeze-marker amendment.

Required artifacts:

- `manifest.json` -- git commit, dirty flag, schema versions, seed slate,
  task split, runtime, data/register hashes, and command argv;
- `split.csv` -- one row per task with `task_id`, `primary_prior`,
  `predicted_boundary`, and split assignment;
- `learning_curves.csv` -- one row per epoch, seed, and arm;
- `per_instance.csv` -- one row per instance, arm, seed, and slot;
- `per_task.csv` -- task-level aggregates used for paired bootstrap;
- `per_prior.csv` -- primary-prior and predicted-boundary breakdowns;
- `scores.csv` -- aggregate metrics, floor rows, and bootstrap intervals;
- `residuals.jsonl` -- compact per-instance residual payloads;
- `quarantine_log.csv` -- required for every Branch B or Branch C claim;
- `branch_adjudication.md` -- final Branch A/B/C application with public
  language draft.

The quarantine taxonomy is frozen as:

| quarantine key | registered source | attribution requirement |
| --- | --- | --- |
| `color_permutation_quotient` | primary `color_role`; six tasks listed above | `signature_only` loses absolute color or `signature_palette` fails only where palette role binding is ambiguous |
| `count_capacity_cliff` | `capacity pressure (count is high-entropy)` | failures concentrate on counting tasks and residuals show missing cardinality or multiplicity |
| `shape_capacity_cliff` | `capacity pressure (shape change carries structural info)` | failures concentrate on spatial-transform tasks and residuals show wrong output envelope |
| `symmetry_phase_loss` | `gauge-breaking ambiguity (symmetry as gauge)` | failures require orientation/reflection representative not preserved by the signature |
| `nonlocal_completion_context` | `non-local information (global context required to fill)` | failures require global context absent from local signature bags |
| `full_state_residual` | `full-state-only dependency (residual category)` | full-grid control succeeds where signature arms miss cell-level residual information |
| `learner_capacity` | any stratum | all arms, including full-grid control, remain near the floor |

For Branch B, `quarantine_log.csv` must allocate each failed held-out task to
exactly one primary quarantine key, include the evidence column(s) used for the
allocation, and show that the non-quarantined slice meets Branch A. A
post-hoc key may be added only as `unregistered_other`, and any use of that key
blocks Branch B.

#### Public Language Until Receipt

Allowed after this amendment and before a Blackwell receipt:

> "Phase 3's deterministic low-capacity branch produced three task-hardness
> receipts and was paused. A new Blackwell sufficiency unpause spec is now
> filed, with held-out-task splits, algebraic sufficiency conditions, and a
> frozen decoder design. No new sufficiency verdict is admitted until the
> Blackwell runner produces a receipt."

Still forbidden:

- "Sundog solves ARC";
- any public-evaluation or Kaggle private/semi-private claim;
- any statement that the three existing Phase 3 receipts prove or disprove
  signature sufficiency;
- any Branch A/B/C claim before `branch_adjudication.md` is filed and hashed.

### 2026-05-28 -- Blackwell Runner Backend Freeze

Author: Codex, from operator-provided minimal-friction setup.

Justification: the Blackwell unpause spec reserved the learner family and
command shape but did not choose the concrete training backend. This amendment
freezes the hybrid implementation path: Python/PyTorch owns the actual model
and receipt generation; Node remains a thin npm-compatible orchestration shell.

Verdict impact: runner scaffolding is admitted for smoke/probe work only. This
amendment does not admit a full training receipt, does not change the three
filed deterministic-low-capacity verdicts, and does not permit Branch A/B/C
language before the Python runner emits and hashes `branch_adjudication.md`.

#### Files And Commands

Primary implementation:

```text
docs/prereg/arc/phase3_decoder.py
```

Thin npm wrapper:

```text
scripts/arc-phase3-blackwell-sufficiency-v1.mjs
```

Reserved command:

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1
```

The wrapper invokes `python` by default. Operators may set `SUNDOG_PYTHON` to a
specific interpreter path without changing the receipt command semantics; the
Python runner records `sys.executable`, Python version, Torch version, and
platform in `manifest.json`.

#### Exact Backend And Model Spec

The admitted backend is local CPU PyTorch. The first supported local runtime
observed before this amendment was:

```text
Python 3.14.4
torch 2.11.0+cpu
```

The runner must record the actual runtime in every manifest; matching these
versions is not required for a smoke/probe receipt, but any full receipt must
name version differences in its freeze-marker note.

Frozen model and training details:

| field | value |
| --- | --- |
| learner version | `blackwell_task_decoder_v1` |
| backend | Python/PyTorch, CPU |
| layers | `2` transformer encoder layers |
| `d_model` | `128` |
| attention heads | `4` |
| feed-forward width | `256` |
| dropout | `0.10` |
| token order | repeated `demo_input`, `demo_output`, then `query_input`; pad to `11` tokens |
| token types | `demo_input`, `demo_output`, `query_input`, `pad` |
| output heads | height class `1..30`, width class `1..30`, `30x30x10` color logits |
| loss | cell cross entropy inside target shape + `0.25` height CE + `0.25` width CE |
| optimizer | AdamW |
| AdamW betas | `(0.9, 0.999)` |
| AdamW eps | `1e-8` |
| weight decay | `0.0001` |
| learning rate | `0.0003` |
| learning-rate schedule | constant |
| batch size | `16` |
| max epochs | `400` for full run |
| early stop | patience `40` epochs |
| seed slate | `20260528`, `20260529`, `20260530`, `20260531`, `20260601` |
| weight init | Xavier-uniform linear weights, zero linear biases, normal embedding weights `mean=0,std=0.02`, unit LayerNorm weights, zero LayerNorm biases |
| determinism | Python RNG and Torch RNG seeded; Torch deterministic algorithms requested with warn-only behavior; Torch threads set to `1` |
| top-2 rule | top cell-logit grid, plus a deterministic single-cell margin-flip candidate |

For grid-scorable arms, validation early stopping ranks by validation
`grid_exact_any_slot`, with validation loss as tie-break. For
`signature_only` and `metadata_only`, where Pass A does not admit direct
exact-grid scoring, validation ranks by `rep_exact_any_slot`, with validation
loss as tie-break. This is a backend operationalization of the earlier
Pass C rule, not a new verdict metric.

#### Receipt Surface

The Python runner must emit the artifacts already required by the Blackwell
unpause spec and additionally emits the operator-requested umbrella receipt:

```text
phase3_receipt.json
```

`phase3_receipt.json` contains the manifest, selected-seed ranking, score
rows, per-task rows, residual payloads, and the branch decision. The markdown
summary remains:

```text
branch_adjudication.md
```

The first admitted uses are:

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1 --dry-run --allow-dirty
```

and a capped timing probe:

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1-probe --probe-only --probe-epochs 5 --allow-dirty
```

If the probe extrapolates the full five-seed run above the repo's ten-minute
rule, the full command must be staged for the operator instead of run inline,
with measured seconds/epoch and projected wall clock recorded in the next
receipt note.

### 2026-05-28 -- Blackwell Probe Timing Receipt And Full-Run Staging

Author: Codex, using operator-run timing probe output.

Justification: the Backend Freeze amendment admitted a capped timing probe
before any full Blackwell run. The operator ran the 5-epoch, one-seed probe.
This amendment records the measured runtime, hashes the probe artifacts, and
applies the repo's ten-minute rule to stage the full command rather than run it
inline.

Verdict impact: **no Branch A/B/C adjudication**. The probe used one seed,
five epochs, and `--allow-dirty`; it is a timing and harness receipt only. The
`Branch C` string emitted in the first probe's `branch_adjudication.md` is
non-adjudicative and must not be cited as a verdict. The runner was patched
after this probe so all future `mode=probe` receipts emit
`branch=not_adjudicated`.

#### Probe Command

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1-probe --probe-only --probe-epochs 5 --allow-dirty
```

#### Probe Provenance

| field | value |
| --- | --- |
| `generatedAt` | `2026-05-28T11:26:34Z` |
| `completedAt` | `2026-05-28T11:27:08Z` |
| wall clock from manifest | `34.0 s` |
| `gitCommit` | `C8E27B36FBA25B992DE85B41276E70631B1D4825` |
| `gitDirty` | `false` |
| `mode` | `probe` |
| `seedSlateEffective` | `20260528` |
| `maxEpochsEffective` | `5` |
| `torchVersion` | `2.11.0+cpu` |
| `pythonVersion` | `3.14.4` |
| train / validation / test LODO / pttest instances | `78 / 18 / 19 / 6` |

Console per-arm timing:

| arm | selected seed | elapsed |
| --- | ---: | ---: |
| `raw_grid_lowcap` | `20260528` | `8.6 s` |
| `signature_palette` | `20260528` | `7.2 s` |
| `signature_only` | `20260528` | `7.1 s` |
| `metadata_only` | `20260528` | `7.1 s` |

Learning-curve readback: each arm ran exactly 5 epochs and selected epoch 5.
Validation metric remained `0.0` throughout the capped probe while validation
loss decreased monotonically. This supports the interpretation that the probe
is too short to adjudicate and that early-stop timing cannot be assumed to fire
near epoch 5 in the full run.

#### Probe Artifact Hashes

Path: `results/arc/phase3-blackwell-sufficiency-v1-probe/`

| artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `09508F086813A11FFAF80F2183A22BD4626B716A6713902534ACE748F9DFE380` |
| `split.csv` | `ED0341B981ED6B069FA2C9BB70C80334B70CC9B947719C5DF6CC0E25123C0A95` |
| `learning_curves.csv` | `EBAE3E7C12408B17398B72DDEFCFFC10DFB8EBC52379E5D7930DFF25E264CF2F` |
| `scores.csv` | `CE2E97493F648FA472E3ABA960917B03D95D2C335EAFDA96FE29415D18A056EE` |
| `per_instance.csv` | `6396C5E80BBE63FA3CC91FA1B75FCF016C8598DEC9ACB6F83452CCEE804C934C` |
| `per_task.csv` | `7383E11491D62EC609667BB5C409E48550EC8C234B2184E23781AD228CB08C70` |
| `per_prior.csv` | `6010BF217A59AFD2AEB8C7DBC13C56FA5DF4583E2482F87462F26D33D81D6637` |
| `quarantine_log.csv` | `96F704FBFE91E25F5F662D96E461C6821FD0961E0AF4E997E9020ED4E3792CEA` |
| `residuals.jsonl` | `2BFD980B3C07AA50794F9C1EADC2ABB4B15FBD033040B873512447BC56600A3A` |
| `phase3_receipt.json` | `DEC30F301EB0DE3120E6BE30E125360E22F9CEA0CCF1264EDD16410B7654D62F` |
| `branch_adjudication.md` | `1DABB9A02F271BA826EC01FA4015514A2EBB099746AD73C6DC13CE1EFB5FCB8B` |

#### Extrapolated Full Cost

The probe measured one seed across all four arms for 5 epochs in 34 seconds.
The registered full run is five seeds, all four arms, up to 400 epochs:

```text
34.0 s * (400 / 5) * 5 = 13600 s = 3.78 h
```

Even the optimistic earliest practical early-stop envelope exceeds the inline
budget. If a full run stopped after 45 epochs for each of five seeds:

```text
34.0 s * (45 / 5) * 5 = 1530 s = 25.5 min
```

Therefore the full run is **not** an inline agent command under the repo's
ten-minute rule.

#### Staged Full Command

After committing the runner/freeze marker and starting from a clean worktree,
the operator can run:

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1
```

Estimated wall clock: **~3.8 hours worst-case from the 5-epoch probe**, with a
plausible lower bound still above the inline limit. The run is resume-unsafe in
this first implementation; if interrupted, delete or move the partial output
directory and restart the same command. The read-back paths are:

- `results/arc/phase3-blackwell-sufficiency-v1/manifest.json`;
- `results/arc/phase3-blackwell-sufficiency-v1/phase3_receipt.json`;
- `results/arc/phase3-blackwell-sufficiency-v1/branch_adjudication.md`;
- `results/arc/phase3-blackwell-sufficiency-v1/hashes.json`.

Only `branch_adjudication.md` from this full clean run can select Branch A,
Branch B, or Branch C.

#### Post-Probe Runner Patch

Two harness-only corrections were made after reading the probe:

- probe and dry-run modes now emit `branch=not_adjudicated` instead of applying
  Branch A/B/C logic;
- `enable_nested_tensor=False` is passed to PyTorch's `TransformerEncoder` to
  avoid the prototype nested-tensor warning in future receipts.

A one-epoch smoke after this patch completed under the same command family and
reported `Branch decision: not_adjudicated`.

### 2026-05-28 -- Blackwell Full-Run Receipt

Author: Codex, using operator-run full Blackwell receipt.

Justification: the operator ran the staged full `blackwell_task_decoder_v1`
command on a clean worktree. This amendment records the manifest, artifact
hashes, selected seeds, score table, and Branch A/B/C adjudication.

Verdict impact: **Branch C -- bounded failure** for the registered
`blackwell_task_decoder_v1` Blackwell runner. `signature_palette` fails to
clear the non-trivial exact-grid floor on the easiest held-out subset and on
both held-out lanes. This is not Branch A support and not Branch B named
quarantine. Because `raw_grid_lowcap` also remains at exact-grid zero, this
receipt should be described as a bounded failure of the registered Blackwell
decoder to demonstrate sufficiency, not as evidence that the full grid control
learned a rule that the signature lost.

#### Command

```powershell
npm run arc:phase3:blackwell:sufficiency -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-blackwell-sufficiency-v1
```

#### Provenance

| field | value |
| --- | --- |
| `generatedAt` | `2026-05-28T11:40:25Z` |
| `completedAt` | `2026-05-28T11:52:59Z` |
| wall clock from manifest | `754 s` / `12m34s` |
| `gitCommit` | `79469BC8AA36F575175FB0D00F3EF71C1EE2A71B` |
| `gitDirty` | `false` |
| `allowDirty` | `false` |
| `mode` | `full` |
| `featureSchemaVersion` | `arc-p3-feature-v1` |
| `protocolVersion` | `arc-p3-blackwell-protocol-v1` |
| `receiptSchemaVersion` | `arc-p3-blackwell-receipt-v1` |
| `learnerVersion` | `blackwell_task_decoder_v1` |
| `torchVersion` | `2.11.0+cpu` |
| `pythonVersion` | `3.14.4` |
| train / validation / test LODO / pttest instances | `78 / 18 / 19 / 6` |

Selected seeds:

| arm | selected seed |
| --- | ---: |
| `raw_grid_lowcap` | `20260528` |
| `signature_palette` | `20260529` |
| `signature_only` | `20260531` |
| `metadata_only` | `20260528` |

Console per-arm runtime:

| arm | elapsed |
| --- | ---: |
| `raw_grid_lowcap` | `289.4 s` |
| `signature_palette` | `144.8 s` |
| `signature_only` | `161.3 s` |
| `metadata_only` | `151.7 s` |

Validation exact metric stayed `0.0` for all arms and seeds. Seed selection was
therefore determined by validation loss tie-break. Selected epochs:

| arm | selected seed | selected epoch | validation loss |
| --- | ---: | ---: | ---: |
| `raw_grid_lowcap` | `20260528` | `8` | `3.679690123` |
| `signature_palette` | `20260529` | `10` | `3.989161015` |
| `signature_only` | `20260531` | `11` | `3.839058638` |
| `metadata_only` | `20260528` | `7` | `4.029307127` |

#### Outcome

| lane | arm | grid exact any | rep exact any | shape exact slot1 | palette exact slot1 | pixel best |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `pttest` | `metadata_only` | `NA` | `0.000` | `0.000` | `0.000` | `0.000` |
| `pttest` | `raw_grid_lowcap` | `0.000` | `0.000` | `0.167` | `0.000` | `0.095` |
| `pttest` | `signature_only` | `NA` | `0.000` | `0.333` | `0.000` | `0.095` |
| `pttest` | `signature_palette` | `0.000` | `0.000` | `0.167` | `0.000` | `0.095` |
| `test_lodo` | `metadata_only` | `NA` | `0.000` | `0.000` | `0.000` | `0.000` |
| `test_lodo` | `raw_grid_lowcap` | `0.000` | `0.000` | `0.158` | `0.000` | `0.110` |
| `test_lodo` | `signature_only` | `NA` | `0.000` | `0.316` | `0.000` | `0.128` |
| `test_lodo` | `signature_palette` | `0.000` | `0.000` | `0.158` | `0.000` | `0.111` |
| `validation_lodo` | `metadata_only` | `NA` | `0.000` | `0.167` | `0.000` | `0.127` |
| `validation_lodo` | `raw_grid_lowcap` | `0.000` | `0.000` | `0.167` | `0.000` | `0.127` |
| `validation_lodo` | `signature_only` | `NA` | `0.000` | `0.167` | `0.000` | `0.127` |
| `validation_lodo` | `signature_palette` | `0.000` | `0.000` | `0.167` | `0.000` | `0.127` |

The receipt-selected `signature_palette` exact rates are:

- held-out-task LODO: `0/19`;
- held-out public-training test: `0/6`.

The receipt-selected `raw_grid_lowcap` exact rates are also:

- held-out-task LODO: `0/19`;
- held-out public-training test: `0/6`.

Bootstrap task metric means and confidence intervals are all `0.0` in
`scores.csv`.

#### Failure And Quarantine Readback

The runner's automatic branch decision:

```text
Branch C
signature_palette fails to clear the exact-grid floor on the easiest registered held-out subset
```

Failure labels in `per_instance.csv` are dominated by `detection` for every
grid-scorable arm. `signature_only` additionally records `structural_zero` on
the primary color-role rows, as pre-registered. `quarantine_log.csv` allocates
signature-arm failures evenly across the six registered boundary categories,
three rows per `(arm, quarantine_key)` because it covers validation LODO,
test LODO, and pttest for each held-out task.

Branch interpretation:

- Branch A fails because `signature_palette` does not clear the non-trivial
  exact-grid floor on either held-out lane.
- Branch B fails because the non-quarantined slice does not satisfy Branch A;
  failures are not confined to a pre-registered category.
- Branch C fires under the pre-registered easiest-subset rule.

The raw-grid exact floor is a required caveat. The full-grid control did not
demonstrate held-out exact learning either, so this receipt closes the
registered `blackwell_task_decoder_v1` lane as a bounded decoder/representation
failure, not as a demonstration of a learned full-grid rule that proves a
specific signature information loss.

#### Artifact Hashes

Path: `results/arc/phase3-blackwell-sufficiency-v1/`

| artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `9216312914AC53680A4A798579EB31B6B5DFBB09138A7FB20E23355E54723741` |
| `split.csv` | `ED0341B981ED6B069FA2C9BB70C80334B70CC9B947719C5DF6CC0E25123C0A95` |
| `learning_curves.csv` | `8FC1D83E2C2A4559A04BE4E88A3D1D608F61E8CC5D351E2C9DFE05A523DE804C` |
| `scores.csv` | `CD03C9EA28990E0E336F1FBF2AB70FE677590FF5C5C4ECD511168729A3899287` |
| `per_instance.csv` | `E4357681A157C054D25206F5F7EF4A5FDB798551CE15106C81BA92EE8408D804` |
| `per_task.csv` | `8F1B69E630BFAB9BD8D65FBBCFE2B178936989A6F4717A0D812A46E4CB4D66CB` |
| `per_prior.csv` | `D59C884C03B696C57B984BEEB892825157CA8F35BC316B794AACFA1ED719D7D3` |
| `quarantine_log.csv` | `96F704FBFE91E25F5F662D96E461C6821FD0961E0AF4E997E9020ED4E3792CEA` |
| `residuals.jsonl` | `33F31457946F92BC7DC4B640633AEB254C960F7346D4C5C290B0AB6C255B09AD` |
| `phase3_receipt.json` | `7E149E941442EB1782080B56A5C1180703FEBCEBB0862B9703DCE5E175D0678F` |
| `branch_adjudication.md` | `E52940D89CF1AE9F4338054C63FEBD309283E30B2B631E7ACF15577FF6057CD1` |

#### Public Language

Allowed:

> "The registered Phase 3 Blackwell decoder did not support signature
> sufficiency. In the full clean receipt, `signature_palette` scored zero
> exact matches on both held-out-task LODO and held-out public-training test
> lanes, so the pre-registered Branch C bounded-failure rule fired. The matched
> full-grid control also scored zero exact matches, so this is a bounded failure
> of the registered decoder lane rather than evidence that the full grid learned
> a rule the signature lost."

Still forbidden:

- "Sundog solves ARC";
- any public-evaluation or Kaggle private/semi-private claim;
- any Branch A support or Branch B narrowed-support claim from this receipt;
- describing this receipt as evidence on the public-evaluation split.

### 2026-05-28 -- Post-Blackwell Comparison Gate And Next Paths

Author: Codex, from operator-provided strategic interpretation.

Justification: the full Blackwell receipt fired Branch C, but the matched
`raw_grid_lowcap` control also stayed at exact zero. This amendment freezes the
interpretation of that fact: the current receipt calibrates the decoder lane,
not the signature-vs-full-grid information comparison. It also names the
minimal next paths before any renewed sufficiency comparison is meaningful.

Verdict impact: no change to the Blackwell full-run verdict. Branch C bounded
failure remains filed for `blackwell_task_decoder_v1`. This amendment narrows
how the verdict may be interpreted and gates any future Phase 3 comparison.

#### Comparison Gate

A meaningful `signature_palette` vs `raw_grid_lowcap` Blackwell comparison
requires a passing full-grid control. The minimum gate for a future lane is:

- `raw_grid_lowcap` must clear the registered non-trivial exact-grid floor on
  held-out-task LODO and held-out public-training test lanes;
- the run must be clean, hashed, and use a pre-registered learner/subset;
- only after that gate passes may the receipt ask whether `signature_palette`
  is statistically indistinguishable from, better than, or materially behind
  the full-grid control.

Until that happens, a zero-exact `signature_palette` result is a decoder-lane
calibration point rather than a clean information-loss result.

#### Admitted Next Paths

Future Phase 3 work must choose one of the following before execution:

1. **Strengthen the decoder lane.** Admit a stronger learner by append-only
   amendment: deeper/wider model, better optimization, improved curriculum,
   more demonstrations or augmentation from public-training tasks, or a
   different architecture. The amendment must freeze capacity, data policy,
   seeds, wall-clock expectation, and a full-grid-control pass gate before
   looking at results.
2. **Narrow the registered task subset.** File a new subset amendment or
   sidecar register for a class where the current lane can plausibly learn from
   the full grid. The subset must be chosen by pre-registered criteria, not by
   post-hoc success inspection, and must preserve leak discipline.
3. **Do both.** Admit a stronger learner and a narrower, explicitly scoped
   registered class. In that case the claim boundary is the narrowed class,
   not the original 36-task Phase 0 register.

No renewed sufficiency claim may be made from any of these paths unless
`raw_grid_lowcap` passes first.

#### Strategic Interpretation

This result is not a setback for the Sundog-vs-ARC direction. It is a
calibration point:

- the frozen small transformer lane is too weak for the breadth of the
  36-task Phase 0 register;
- the next honest test needs either a stronger decoder, an easier/narrower
  registered task distribution, or both;
- Phase 0-2 remain useful receipts: they define the registered subset, leak
  discipline, shadow-domain operator, projection behavior, color caveat, and
  baseline floor;
- Phase 3 now contributes an honest negative boundary: under the registered
  small-transformer lane, neither signature nor full grid learned held-out
  exact outputs.

Theory-paper posture:

> "We present the shadow/signature formulation and Phase 0-2 receipts, then
> report a Phase 3 bounded-failure calibration: the registered
> `blackwell_task_decoder_v1` did not produce a passing full-grid control and
> therefore could not adjudicate signature sufficiency. A valid sufficiency
> test requires either a stronger decoder, a narrower registered task class, or
> both, with the full-grid control passing before the signature comparison is
> interpreted."

Still forbidden:

- describing the Blackwell full receipt as a signature-specific falsification
  independent of decoder capacity;
- describing `raw_grid_lowcap` as having learned the registered task class;
- using a narrowed future subset to make claims about the original 36-task
  register without filing a separate cross-subset receipt.

### 2026-05-28 -- Strengthened Decoder Lane Admission: Raw-Grid Gate V2

Author: Codex, following the post-Blackwell comparison gate.

Justification: the Blackwell full receipt could not adjudicate signature
sufficiency because the full-grid control stayed at exact zero. This amendment
starts with the first admitted next path: strengthen the decoder lane and test
whether `raw_grid_lowcap` can pass before any renewed signature comparison is
attempted.

Verdict impact: no signature sufficiency claim is admitted. This lane is a
**full-grid control gate only**. If it passes, a later append-only amendment may
admit matched signature arms under the same or explicitly comparable training
budget. If it fails, the comparison gate remains closed.

#### Learner And Scope

Learner version:

```text
blackwell_publictrain_rawgrid_gate_v2
```

Primary implementation:

```text
docs/prereg/arc/phase3_decoder_v2.py
```

Npm wrapper:

```text
scripts/arc-phase3-rawgrid-gate-v2.mjs
```

Command:

```powershell
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2
```

This lane scores only `raw_grid_lowcap`. `signature_palette`,
`signature_only`, and `metadata_only` are intentionally not run here; the point
is to establish whether a strengthened full-grid control can clear the gate.

#### Strengthening Moves

Relative to `blackwell_task_decoder_v1`, V2 changes two things:

1. **More public-training demonstrations.** Meta-training may use all ARC
   public-training tasks except the frozen validation and test tasks from the
   36-task register. For each meta-train task, LODO train-pair instances are
   admitted; public-training test instances with available outputs are also
   admitted as train instances. No public-evaluation grids, Kaggle private
   data, or manual inspection outside the existing register is admitted.
2. **More capacity.** The set-conditioned transformer is widened/deepened while
   keeping the same raw-grid serialization and deterministic receipt surface.

Frozen model details:

| field | V2 value |
| --- | --- |
| arm | `raw_grid_lowcap` only |
| layers | `4` transformer encoder layers |
| `d_model` | `192` |
| attention heads | `6` |
| feed-forward width | `768` |
| dropout | `0.10` |
| optimizer | AdamW |
| AdamW betas | `(0.9, 0.95)` |
| AdamW eps | `1e-8` |
| weight decay | `0.01` |
| learning rate | `0.0002` |
| learning-rate schedule | constant |
| batch size | `24` |
| max epochs | `120` |
| early stop | patience `20` |
| seed slate | `20260528`, `20260529`, `20260530` |
| loss | same as V1: cell CE + `0.25` height CE + `0.25` width CE |
| top-2 rule | same as V1: top cell-logit grid plus deterministic single-cell margin flip |

The V2 receipt must report auxiliary public-training task count, train-instance
count, validation-instance count, selected seed, selected epoch, validation
loss, and held-out exact metrics.

#### Gate

The V2 full-grid control passes only if the selected `raw_grid_lowcap` seed
records at least two exact successes from distinct held-out tasks on each lane:

- held-out-task LODO over the six frozen test tasks;
- held-out public-training test over the same six frozen test tasks.

If the gate passes, Phase 3 may admit a matched V2 signature comparison. If the
gate fails, the correct public language is:

> "Strengthening the decoder with public-training meta-training and higher
> capacity still did not produce a passing full-grid control, so the
> signature-vs-full-grid sufficiency comparison remains gated."

#### Probe Discipline

The runner supports `--dry-run`, `--probe-only`, `--probe-epochs`, and
`--limit-aux-tasks` for capped probes. `--limit-aux-tasks` is a smoke/probe
control and must not be used for a full receipt unless a later amendment
changes the data policy.

First admitted smoke:

```powershell
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2-dryrun --dry-run --allow-dirty
```

First admitted timing probe:

```powershell
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2-probe --probe-only --probe-epochs 2 --limit-aux-tasks 24 --allow-dirty
```

If the uncapped full run projects above the repo's ten-minute rule, stage the
full command for the operator with measured probe rates and expected wall
clock. Do not run a long full V2 receipt inline.

### 2026-05-28 -- Raw-Grid Gate V2 Dry-Run And Timing Probe

Author: Codex.

Justification: the V2 strengthened decoder lane admission required a dry-run
and capped timing probe before any full receipt. This amendment records both,
applies the ten-minute rule, and stages the uncapped full command.

Verdict impact: no full-grid gate verdict is admitted. The dry-run and capped
probe are harness/timing receipts only. The comparison gate remains closed
until an uncapped clean V2 full receipt is filed.

#### Dry-Run Command

```powershell
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2-dryrun --dry-run --allow-dirty
```

Dry-run result:

| field | value |
| --- | --- |
| `taskCount` | `1000` |
| `trainInstanceCount` | `4259` |
| `validationInstanceCount` | `24` |
| `testLodoInstanceCount` | `19` |
| `pttestInstanceCount` | `6` |
| `learnerVersion` | `blackwell_publictrain_rawgrid_gate_v2` |
| `mode` | `dry_run` |

This validates that the V2 data policy loads all public-training tasks while
preserving the frozen six validation and six test tasks from the registered
36-task split.

#### Capped Probe Command

```powershell
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2-probe --probe-only --probe-epochs 2 --limit-aux-tasks 24 --allow-dirty
```

Probe result:

| field | value |
| --- | --- |
| `generatedAt` | `2026-05-28T12:25:29Z` |
| `completedAt` | `2026-05-28T12:25:40Z` |
| wall clock from manifest | `11.0 s` |
| selected seed | `20260528` |
| best epoch | `2` |
| elapsed training time | `9.699 s` |
| `taskCount` | `60` |
| `trainInstanceCount` | `204` |
| `validationInstanceCount` | `24` |
| `gateDecision` | `not_adjudicated` |

Probe scores are zero exact on both held-out lanes, as expected for a two-epoch
probe. They do not adjudicate the gate.

Probe artifact hashes:

| artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `7F31096FDF4AAB5DD72797BFBA49CF96FDCF2270F2D8EC47FCD0D83202308046` |
| `split.csv` | `C921EC8E49F7B1E86443606AA559D5F28621954EAC2C70BC13BBB3107F7AC8A5` |
| `learning_curves.csv` | `B3F30F20C85E40DA044AB13CF945B8798BA967E6B8A851F14EB99E0B5522E69E` |
| `scores.csv` | `A3B8E570A8562EFA917933AA4392EA100BB7F42BDEB1024E6004F1639CC2DB37` |
| `per_instance.csv` | `CCF9EE0780AF03D349848E4C03CCBD3D7E9D45BEDDCB52C136CF0EFD16CD3BDB` |
| `per_task.csv` | `2528EB74FD6B9D51D9EFE6414BE20AC05DCCB9BB645D90F497B5D0427068EB73` |
| `per_prior.csv` | `32581FAB4628C8C0F210D44F3B4404D965B738158E308CDDFD9B90EFCF03079A` |
| `quarantine_log.csv` | `E32EE735A3BF6C158D607DE5DC0354CE2C7B23D6714C3500496911398E7F586B` |
| `residuals.jsonl` | `51FCF88DAB7F0A9422A3FFC5014766F5287ABBDAF106C7C24E316B2344007A84` |
| `phase3_receipt.json` | `0EB3A8FD226DB013B9CAA2A00936310662BE4896AF6555EB5A719113DDF5DF55` |
| `branch_adjudication.md` | `B2F0CAC39363EE04471F1F38739A0D298718A1B4A71E31145DFFB32357BAE8EB` |

#### Full-Run Estimate

Scaling from the capped probe:

```text
probe wall: 11.0 s
probe train instances: 204
uncapped train instances: 4259
instance scale: 20.877
epoch scale: 120 / 2
seed scale: 3

11.0 * 20.877 * 60 * 3 = 41337 s = 11.48 h
```

This exceeds the inline agent budget by a wide margin.

#### Staged Full Command

Run only after committing the V2 admission/scaffold and starting from a clean
worktree:

```powershell
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2
```

Estimated wall clock: **~11.5 hours** from the capped probe. The first V2
implementation is resume-unsafe; if interrupted, move or delete the partial
output directory and rerun the same command. Read back:

- `results/arc/phase3-rawgrid-gate-v2/manifest.json`;
- `results/arc/phase3-rawgrid-gate-v2/phase3_receipt.json`;
- `results/arc/phase3-rawgrid-gate-v2/branch_adjudication.md`;
- `results/arc/phase3-rawgrid-gate-v2/hashes.json`.

### 2026-05-28 -- Raw-Grid Gate V2 Shard+Merge Protocol Admission

Author: Claude (Opus 4.7).

Justification: the staged uncapped V2 full command has a ~11.5 h wall-clock
estimate per the timing probe above. The three seeds in `SEED_SLATE` are
independent (each re-seeds RNG, instantiates a fresh `TaskDecoder`, trains
and evaluates without cross-seed state), and seed selection runs only after
all three complete. The natural shard axis is therefore seed: three
parallel ~3.8 h shards collapse the wall clock by ~3×, modulo
shard-equivalence preservation. This amendment admits a shard+merge
protocol that produces a binding receipt byte-equivalent to a serial V2
run with the same seeds (verified by a probe-grade smoke), without
altering the V2 frozen contract.

Verdict impact: no full-grid-control gate verdict is admitted. The
shard+merge protocol is an orchestration option for the same V2 run; the
gate criterion, learner contract, model spec, data policy, and seed slate
are unchanged. A binding sharded receipt's gate decision adjudicates
identically to a serial receipt's.

#### Frozen Shard Surface

Added to `docs/prereg/arc/phase3_decoder_v2.py`:

- `--shard-seed <int>` — must be a value in `SEED_SLATE`
  (`{20260528, 20260529, 20260530}`). Trains and evaluates that single
  seed using the same code path as the serial run, then writes a per-shard
  receipt at `--out`. Sets `manifest.mode = "shard"`,
  `manifest.shardSeed = <int>`, `manifest.gateDecision = {"gate":
  "not_adjudicated", "reason": "shard run only"}`. Compatible with
  `--probe-only`, `--probe-epochs`, `--max-epochs`, `--limit-aux-tasks`.
- `--merge` plus `--shard-dirs <dir1,dir2,...>` — reads each shard's
  intermediate, asserts frozen-config consistency across shards, sorts
  by `shardSeed`, concatenates per-instance / per-slot / learning /
  residual rows in seed order, runs the V2 selection rule on the merged
  candidate list, runs `aggregate_per_task` / `aggregate_per_prior` /
  `aggregate_scores` from `phase3_decoder.py` on the merged
  `per_instance_any`, adjudicates the gate in `"full"` mode, and writes
  the binding receipt at `--out`. Sets `manifest.mode = "full"`,
  `manifest.shardedRun = true`, `manifest.shardSources = [...]` listing
  each shard.

Each shard writes the existing V2 receipt schema plus two intermediates
required by the merge:

- `per_instance_any.jsonl` — the raw `per_instance_any` rows the
  aggregation functions consume (not a CSV roundtrip; preserves native
  types);
- `validation_candidates.json` — the per-seed validation candidate dict
  (`arm`, `seed`, `best_epoch`, `validation_metric`, `validation_loss`,
  `elapsed_seconds`).

Serial runs also emit both files (cheap; lets a serial run be
post-validated against a sharded run if needed).

Npm wrappers (added to `package.json`):

```json
"arc:phase3:rawgrid-gate-v2:shard": "node scripts/arc-phase3-rawgrid-gate-v2.mjs --shard-seed",
"arc:phase3:rawgrid-gate-v2:merge": "node scripts/arc-phase3-rawgrid-gate-v2.mjs --merge"
```

The shard command takes the seed as the next argument after `--`:

```powershell
npm run arc:phase3:rawgrid-gate-v2:shard -- 20260528 --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase3-rawgrid-gate-v2-shard-20260528
```

The merge command takes shard dirs comma-separated:

```powershell
npm run arc:phase3:rawgrid-gate-v2:merge -- --shard-dirs results/arc/phase3-rawgrid-gate-v2-shard-20260528,results/arc/phase3-rawgrid-gate-v2-shard-20260529,results/arc/phase3-rawgrid-gate-v2-shard-20260530 --out results/arc/phase3-rawgrid-gate-v2
```

#### Shard-Equivalence Guarantee

For a sharded run to substitute for a serial V2 run, the merge must
produce a binding receipt whose adjudication artifacts (`scores.csv`,
`per_task.csv`, `per_instance.csv`, `per_prior.csv`,
`learning_curves.csv`, `residuals.jsonl`, `quarantine_log.csv`,
`branch_adjudication.md`) are byte-identical to the serial run's with
the same seed slate, master seed, `max_epochs`, and `limit_aux_tasks`.

This is the binding shard-equivalence criterion. It is enforced by:

1. **Identical training per seed.** Each shard runs the same
   `train_model` / `evaluate_model` calls with the same seed and the
   same model spec. PyTorch CPU determinism + `set_global_determinism`
   per-seed makes per-seed outputs deterministic.
2. **Deterministic concatenation order.** The merge sorts shards by
   `shardSeed` ascending, then concatenates per-instance / per-slot /
   learning / residual rows in that order. This matches the order a
   serial run produces (the serial run iterates `for seed in
   SEED_SLATE`, which is also seed-ascending).
3. **Deterministic selection.** The merge's selection rule is the
   exact V2 rule from `phase3_decoder_v2.py:153`:
   `sort by (-validation_metric, validation_loss, seed); pick [0]`.
4. **Identical aggregation.** The merge calls
   `phase3_decoder.aggregate_per_task` / `aggregate_per_prior` /
   `aggregate_scores` on the merged `per_instance_any` with the
   selected seed — the same functions a serial run calls.
5. **Identical gate adjudication.** The merge calls
   `adjudicate_raw_grid_gate(per_task_rows, "full")` — the same
   function a serial run calls.

The fields that differ between a serial and a sharded receipt are
documented and confined to metadata (not adjudication):

- `generatedAt`, `completedAt`, `mergeStartedAt`, per-shard timings;
- `command`, `tool`, `outDir`;
- `shardedRun`, `shardSources`, `mergeGitCommit`, `mergeGitDirty`,
  `mergeAllowDirty` (new fields, only present in sharded receipts);
- `hashes.json` (different because manifest differs);
- `phase3_receipt.json` (embeds the manifest);
- `validation_candidates.json` (functionally identical content but
  ordering may differ — sharded merge sorts the combined list with the
  same comparator as serial, so the contents and order are identical).

Smoke verification command shape:

```powershell
# Serial baseline
npm run arc:phase3:rawgrid-gate-v2 -- --data-dir ... --register ... --out results/arc/phase3-rawgrid-gate-v2-serial-smoke --max-epochs 2 --limit-aux-tasks 24 --allow-dirty

# Three shards (cheap probe-grade)
npm run arc:phase3:rawgrid-gate-v2:shard -- 20260528 --out results/arc/phase3-rawgrid-gate-v2-shard-20260528 --max-epochs 2 --limit-aux-tasks 24 --allow-dirty ...
npm run arc:phase3:rawgrid-gate-v2:shard -- 20260529 --out results/arc/phase3-rawgrid-gate-v2-shard-20260529 --max-epochs 2 --limit-aux-tasks 24 --allow-dirty ...
npm run arc:phase3:rawgrid-gate-v2:shard -- 20260530 --out results/arc/phase3-rawgrid-gate-v2-shard-20260530 --max-epochs 2 --limit-aux-tasks 24 --allow-dirty ...

# Merge
npm run arc:phase3:rawgrid-gate-v2:merge -- --shard-dirs <three-dirs> --out results/arc/phase3-rawgrid-gate-v2-merged-smoke --allow-dirty
```

Smoke result (2026-05-28, pre-amendment commit, `--max-epochs 2
--limit-aux-tasks 24`): `scores.csv`, `per_task.csv`, and
`per_instance.csv` are byte-identical between serial and merged
receipts; gate decisions match (`full_grid_control_floor`).

#### Frozen-Config Consistency Assertion

The merge refuses to combine shards that disagree on any of:

- `learnerVersion`, `protocolVersion`, `receiptSchemaVersion`,
  `featureSchemaVersion`;
- `arm`;
- `registerHash`, `dataDirHash`;
- `gitCommit`;
- `modelSpec` (full dict equality);
- `registerPath`, `dataDir`;
- `limitAuxTasks`, `maxEpochsEffective`.

The merge also refuses duplicate `shardSeed` values across shards. If
any of these checks fail the merge aborts and the partial receipt is
not written.

#### Resume And Failure Recovery

Each shard is independently resume-unsafe: if a shard process is
interrupted mid-training, its output directory must be deleted and the
shard re-run. The merge is fast (~2 minutes) and re-runnable; the
merge itself does not need to be sharded.

For the 11.5 h staged full run, sharded into three ~3.8 h shards:

- if any one shard fails, restart that shard only; do not re-run the
  others;
- if more than one shard fails simultaneously, treat as a hardware
  issue and audit before re-running.

The shard-equivalence guarantee does not require the three shards to
run on the same wall clock; they may overlap, be staggered, or use
different cores/devices, as long as the merge consistency assertions
pass.

#### Public Language

The shard+merge protocol does not change any public language. The
adopted public-language drafts from prior V2 amendments apply
unchanged to a sharded binding receipt. Any external citation of a
sharded V2 receipt should note `shardedRun = true` and the
`shardSources` list from the merged manifest if reproducibility
discussion is part of the citation.

### 2026-05-28 -- Raw-Grid Gate V2 Latent Bug Fix + CUDA Determinism Env

Author: Claude (Opus 4.7).

Justification: a pre-binding probe with no `--limit-aux-tasks` cap (the
shape the staged full command at the prior probe-and-staging amendment
would have used) crashed in `phase3_decoder.make_record` with
`ValueError: train_lodo:253bf280:0 exceeds max token count`. The frozen
model spec sets `MAX_DEMOS = 5`, which gives `MAX_TOKENS = MAX_DEMOS * 2
+ 1 = 11`. Any aux task with more than 5 train pairs generates either a
LODO instance (k-1 conditioning + query = 2k-1 tokens; fails for k > 6)
or a `train_pttest` instance (k conditioning + query = 2k+1 tokens;
fails for k > 5) that exceeds `MAX_TOKENS`. The capped 24-aux timing
probe missed this by alphabetical luck; the staged ~11.5 h serial full
command would have crashed mid-run.

Separately, the V2 runner invokes
`phase3_decoder.set_global_determinism(...)` which calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. On CUDA >=
10.2 this requires `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`) to be
set as an environment variable before cuBLAS handles are created; the
first GPU full-aux probe attempt crashed with
`STATUS_STACK_BUFFER_OVERRUN` until the env var was set explicitly.

This amendment lands two coupled changes in
`docs/prereg/arc/phase3_decoder_v2.py`:

1. `load_public_training_tasks` now skips aux tasks with
   `len(parsed["train"]) > p3.MAX_DEMOS` and records the exclusion in
   the manifest as `auxExclusions = {aux_excluded_token_cap,
   aux_excluded_token_cap_ids, max_demos}`.
2. `os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")` runs
   before `import torch` so the V2 runner is CUDA-determinism compliant
   without operator action. CPU-only invocations are unaffected.

The 36 registered tasks all have `k <= 5` per `P0_TASK_REGISTER.csv` so
no registered task is excluded. The exclusion is aux-only.

Verdict impact: no binding V2 receipt has been issued; nothing is voided
or superseded by the fix. The prior probe-and-staging amendment's
staged 11.5 h serial command would have crashed; this fix supersedes
that staging in favor of the launch posture below.

#### Excluded Aux Tasks (29 of 1000)

The first dry-run with the fix in place reported
`aux_excluded_token_cap = 29` and listed all 29 IDs in
`manifest.auxExclusions.aux_excluded_token_cap_ids`. Sample (first 5,
alphabetical): `239be575`, `253bf280`, `2685904e`, `27a28665`,
`281123b4`. The full list is reproducible by re-running any V2
dry-run after this fix is committed.

The effective V2 data policy is therefore:

> All ARC public-training tasks **except**
> (a) the 36 registered Phase 0 tasks' frozen validation/test split
> (already excluded) **and**
> (b) the 29 aux tasks with more than `MAX_DEMOS = 5` train pairs.

This narrows the original V2 admission's "all ARC public-training
tasks except the frozen validation/test register tasks" by 29 tasks
(2.9% of the corpus). The narrowing is structural for the frozen
`MAX_DEMOS = 5` model spec; a future V3 with a wider encoder could
admit those 29 tasks.

#### Revised Wall-Clock Estimate

Probe-grade timings post-fix (probe-only, 2 epochs, all 971 admitted
tasks):

| device | per-shard 2-epoch wall | per-epoch | full-run per-shard (120 epochs) |
| --- | ---: | ---: | ---: |
| CPU (i7-7820HK, 4 cores) | ~3.8 min (extrapolated) | ~115 s | ~3.83 h |
| CUDA (GTX 1080, py3.12, torch 2.5.1+cu121) | 107.2 s | ~53.6 s | ~1.79 h |

Estimated full-run wall clocks (3 seeds):

| protocol | wall |
| --- | ---: |
| CPU serial (staged in prior amendment) | ~11.5 h |
| CPU 3-parallel naive (default threads, 4 cores) | ~10-11 h (no win) |
| CPU 3-parallel with `OMP_NUM_THREADS=2` | ~7.6 h |
| GPU serial (`--device cuda`) | ~5.4 h |
| **GPU 3-parallel sharded (`--device cuda`)** | **~2-2.5 h** |

The GPU 3-parallel sharded path is the admitted launch posture given
the GTX 1080 + py3.12 + torch 2.5.1+cu121 setup is in place. The
shard+merge protocol amendment above pins the per-shard / merge CLI
surface and the shard-equivalence guarantee; both apply unchanged on
CUDA. cuDNN determinism is preserved by the auto-set
`CUBLAS_WORKSPACE_CONFIG` and the existing
`torch.use_deterministic_algorithms(True, warn_only=True)` call.

#### Launch Posture For The First V2 Binding Receipt

Pre-launch checklist:

1. This amendment + the corresponding `phase3_decoder_v2.py` edits are
   committed (freeze-marker commit). The shard+merge protocol
   amendment above is already committed.
2. `SUNDOG_PYTHON` points at the Python 3.12 interpreter with
   `torch==2.5.1+cu121` installed (verified by
   `torch.cuda.is_available() == True` and
   `torch.cuda.get_device_name(0) == "NVIDIA GeForce GTX 1080"`).
3. Worktree is clean (no `--allow-dirty`).

Launch (three parallel background shards, each ~1.9 h on GPU):

```powershell
$env:SUNDOG_PYTHON = "C:\Users\hughe\AppData\Local\Programs\Python\Python312\python.exe"
foreach ($seed in 20260528, 20260529, 20260530) {
    Start-Process -NoNewWindow `
        -RedirectStandardOutput "results/arc/phase3-rawgrid-gate-v2-shard-$seed.log" `
        -RedirectStandardError "results/arc/phase3-rawgrid-gate-v2-shard-$seed.err" `
        -FilePath "npm" -ArgumentList @(
            "run", "arc:phase3:rawgrid-gate-v2:shard", "--",
            "$seed",
            "--data-dir", "$env:USERPROFILE\Datasets\ARC-AGI-2\data",
            "--register", "docs/prereg/arc/P0_TASK_REGISTER.csv",
            "--out", "results/arc/phase3-rawgrid-gate-v2-shard-$seed",
            "--device", "cuda"
        )
}
```

Merge (after all three shards complete):

```powershell
npm run arc:phase3:rawgrid-gate-v2:merge -- `
    --shard-dirs results/arc/phase3-rawgrid-gate-v2-shard-20260528,results/arc/phase3-rawgrid-gate-v2-shard-20260529,results/arc/phase3-rawgrid-gate-v2-shard-20260530 `
    --out results/arc/phase3-rawgrid-gate-v2
```

Resume rules from the shard+merge protocol amendment apply unchanged:
each shard is independently resume-unsafe; if one fails, delete its
output directory and re-run that shard alone, then merge when all
three are present.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Raw-Grid Gate V2 Binding Receipt: `full_grid_control_floor`

The 3-seed GPU-sharded V2 binding run completed under freeze-marker
commit `79C5B060AC732402CE57FDD44892FC600D63854E`. All three shards ran
clean on the same commit with `gitDirty=false`, and the merge step
likewise ran clean (`mergeGitDirty=false`, `mergeAllowDirty=false`).
**No prior verdict changes**: this is the first binding V2 receipt and
it supersedes nothing.

Binding receipt path: `results/arc/phase3-rawgrid-gate-v2/`.
Shards: `results/arc/phase3-rawgrid-gate-v2-shard-{20260528,20260529,20260530}/`.

#### Validation Candidate Rank And Seed Selection

The pre-registered selection rule
(`-validation_metric, +validation_loss, +seed` lexicographic) ranks
the three candidates as follows:

| rank | seed | best_epoch | val_loss | val_metric | shard wall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `20260529` | 12 | `1.62495446` | `0.0` | 1373.8 s (22.9 min) |
| 2 | `20260528` | 14 | `1.62936294` | `0.0` | 1446.7 s (24.1 min) |
| 3 | `20260530` | 15 | `1.64143372` | `0.0` | 1501.8 s (25.0 min) |

All three seeds tied at `validation_metric = 0.0` (no exact grid match
on the 24-instance validation pool), so the tiebreaker fell to
`validation_loss`; seed `20260529` was selected. All three shards
early-stopped under the registered `early_stop_patience = 20` floor,
consistent with the loss plateauing without exact-match signal.

Selection record written to
`results/arc/phase3-rawgrid-gate-v2/validation_candidates.json`.

#### Gate Adjudication

Pre-registered floor (from §8): `lodo_success >= 2 AND pttest_success
>= 2` where a "success" is a task with
`grid_exact_any_rate > 0.010` on the named held-out lane for arm
`raw_grid_lowcap`.

Observed (selected seed `20260529`, from
`results/arc/phase3-rawgrid-gate-v2/scores.csv`):

| lane | instance_count | grid_exact_any | shape_exact_slot1 | palette_exact_slot1 | pixel_best_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `pttest` | 6 | `0.000` | `0.667` | `0.000` | `0.235` |
| `test_lodo` | 19 | `0.000` | `0.526` | `0.000` | `0.251` |
| `validation_lodo` | 18 | `0.000` | `0.722` | `0.000` | `0.336` |
| `validation_pttest` | 6 | `0.000` | `0.500` | `0.000` | `0.292` |

`pttest_exact_tasks = 0`, `test_lodo_exact_tasks = 0`. Both held-out
lanes fall the pre-registered floor.

**Gate decision: `full_grid_control_floor`**.

Manifest: `results/arc/phase3-rawgrid-gate-v2/manifest.json` records
`gateDecision = {"gate": "full_grid_control_floor",
"pttest_exact_tasks": 0, "test_lodo_exact_tasks": 0, "reason":
"raw_grid_lowcap did not clear the pre-registered exact-grid floor on
both held-out lanes"}`. Adjudication narrative:
`results/arc/phase3-rawgrid-gate-v2/branch_adjudication.md`.

#### Shard-Equivalence Verification

The pre-registered shard-equivalence guarantee (the merged receipt's
gate-relevant artifacts must be byte-identical to a serial run that
used the same seed selection) was verified by `cmp` against the
selected-seed shard's outputs:

| artifact | merge vs shard-20260529 |
| --- | --- |
| `scores.csv` | byte-identical (cmp exit 0) |
| `per_task.csv` | byte-identical (cmp exit 0) |
| `branch_adjudication.md` | differs only on the gate-decision line: shard reports `not_adjudicated` ("shard run only"); merged reports `full_grid_control_floor`. Expected — only the merge step adjudicates. |
| `per_instance.csv` | merged contains all three seeds' rows for audit; the selected-seed slice equals the shard's content. |

Hash list: `results/arc/phase3-rawgrid-gate-v2/hashes.json`.

#### Failure Character

`shape_exact_slot1` is moderately high on every lane (0.50–0.72), i.e.
the V2 decoder predicts the correct output **shape** for about half to
three-quarters of held-out instances. But `palette_exact_slot1 = 0.0`
on every lane: the decoder never produced the correct palette set,
even when the shape matched. `pixel_best_mean` is in the 0.23–0.34
band — well below an "almost-right" regime. This is the same
shape-matches-but-content-fails character as the Blackwell V1 receipt
(see the V1 receipt amendment above), now reproduced under the
strengthened raw-grid-only V2 lane with all admitted public-training
auxiliary tasks. The 29 token-cap exclusions documented in the latent
bug-fix amendment above did not change the floor.

#### What This Verdict Does And Does Not Entail

**It does**:

- Close the V2 raw-grid-only control gate as a **bounded failure** of
  the pre-registered decoder lane on the registered held-out task
  splits, under the strengthened V2 data policy (all 971 admitted aux
  tasks vs. the V1 24-aux capped probe).
- Confirm that the strengthening from V1 to V2 (24 aux → 971 aux,
  matched architecture, matched freeze-marker discipline, matched
  shard-equivalence merge) did **not** lift the held-out exact-match
  floor off zero.
- Preserve the Phase 3 PHASE3_5_REFLECTION characterisation: this is
  another receipt within the deterministic-low-capacity-learner family;
  the verdict speaks to that family, not to the shadow-projection
  representation in the abstract.

**It does not**:

- Support Branch A (signature-shadow projection sufficient) of
  PHASE3_5_REFLECTION. The full-grid control floored, so a Branch A
  claim still has no full-grid baseline to outperform.
- Support Branch B (narrowed task class) without an additional
  narrowed-support receipt; this verdict only narrows the V1 receipt's
  decoder-capacity caveat, not the task class.
- Adjudicate the shadow-projection sufficiency hypothesis directly.
  Per the spec body's framing and PHASE3_5_REFLECTION, a sufficiency
  adjudication requires a passing full-grid control first. This
  receipt is a second consecutive failure to produce that baseline.
- Unblock any Kaggle, public-evaluation, or
  signature-vs-full-grid copy. The README public-language constraint
  remains as filed.

#### Public-Language Update

Permitted (additive to the body's filed constraint):

- "Raw-grid gate V2, the strengthened full-grid control, also
  returned a bounded-failure receipt at the registered held-out
  exact-grid floor (`full_grid_control_floor`); both V1 and V2
  full-grid controls now agree, so the held-out exact-grid floor
  remains unobserved for any decoder in the registered deterministic
  family."

Forbidden (additions to the existing list):

- "V2 failed → signature representation is favoured" — the verdict is
  the opposite: V2 floors the same way V1 did, so no comparison is
  licensed.
- "V2 receipt closes Phase 3" — it does not. PHASE3_5_REFLECTION
  Branch C remains the active framing; a Branch A (stochastic
  per-task learner), Branch B (narrowed task class), or Branch D
  (different framing) reopen is the only path back into Phase 3.

#### Frozen By This Verdict

- The V2 binding receipt body, manifest, and hashes are frozen at the
  paths above; any subsequent V3/V4 must allocate a new `learnerVersion`
  rather than overwriting V2 artifacts.
- The 29-task aux exclusion list (`auxExclusions.aux_excluded_token_cap_ids`
  in the manifest) is frozen as the V2 exclusion set under the
  `MAX_DEMOS = 5` model spec. A future V3 with a wider encoder may
  re-admit those tasks, but only under a new `learnerVersion`.
- The `validation_candidates.json` ordering is frozen as the V2
  selection trace.

No body section above the amendments line changes; only the V2
binding-receipt status moves from "staged" (prior amendment) to
"filed, Branch C bounded failure" (this amendment).

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch B Compact Subset Path Admitted

Following the V2 binding-receipt verdict above, a Branch B diagnostic
lane is admitted under PHASE3_5_REFLECTION Branch B. The lane is
pre-registered in its own artifact:

- Spec: [`PHASE3B_COMPACT_SUBSET.md`](PHASE3B_COMPACT_SUBSET.md).
- Frozen subset split file: `docs/prereg/arc/PHASE3B_COMPACT_SPLIT.csv`.

**Why this is a Branch B move, not a V3 raw-grid bump**: the V1 and
V2 receipts (both filed above) agree on the full-grid exact-grid
floor across two structurally distinct strengthening axes (24 aux
tasks → 971 aux tasks, plus shard+merge equivalence and token-cap
exclusions). A third raw-grid V-bump in the deterministic-low-capacity
family would not be informative. The Branch B lane instead narrows
the *task distribution* — keeping the decoder, the hyperparameters,
the selection rule, the seed slate, and the metrics frozen — to ask
the narrower diagnostic question: **does the current decoder lane
produce any exact-match signal at all on the compact-signal slice of
the registered tasks**, where Phase 2 already showed the smallest
signature-collision-residual and the smallest train-pair residuals.

**Scope of admission**:

- This amendment admits the diagnostic lane and the supporting
  tooling (the `--subset-spec` plumbing in `phase3_decoder_v2.py`
  and the `subsetSpecHash` / `subsetSplit` manifest fields, including
  their assert_shard_consistency checks). It does **not** admit any
  binding compact receipt. Receipt admission requires its own
  amendment with a binding `compact_full_grid_control_*` gate
  decision and a verdict block in the style of the V1 / V2
  receipt-filed amendments above.
- The minimal floor (≥1 exact task on `pttest` AND ≥1 on `test_lodo`)
  is pre-registered in `PHASE3B_COMPACT_SUBSET.md` §3 and is the
  smallest non-trivial integer floor compatible with the compact-7
  held-out lanes (2 `pttest` instances, ≈4–6 `test_lodo` instances).
- The Branch A / B / C criteria scoped to the compact subset are
  pre-registered in `PHASE3B_COMPACT_SUBSET.md` §4. Compact-Branch-A
  / B can only be filed if and only if the compact-Branch-C floor
  has first been cleared by the full-grid control.

**What stays frozen**: per `PHASE3B_COMPACT_SUBSET.md` §5 — decoder
architecture, hyperparameters, seed slate, selection rule, signature
extraction, metrics, evaluation code, leak-check, shard-equivalence,
aux pool (same 971 admitted tasks; same 29 token-cap exclusions; the
36 registered tasks not in the compact subset are dropped entirely,
not promoted to aux).

**What changes**: held-out task distribution (36 → 7); internal split
(pre-registered 4 / 1 / 2 in `PHASE3B_COMPACT_SPLIT.csv`); gate floor
(≥2 → ≥1); gate name (`full_grid_control_*` →
`compact_full_grid_control_*`); receipt output directory namespace
(`results/arc/phase3-rawgrid-compact-7-*`). Possibly the seed slate
narrows to a single seed (`20260529`) for the first tranche; the
other two seeds are admitted as optional follow-on verification.

**Tooling deltas** (committed alongside this amendment):

- `phase3_decoder_v2.py`: new `--subset-spec PATH` arg;
  `read_subset_spec()`, `adjudicate_compact_subset_gate()` helpers;
  `load_public_training_tasks()` accepts an optional `subset_split`
  mapping and uses `dataclasses.replace` to override task splits;
  `assert_shard_consistency` now requires `subsetSpecHash`,
  `subsetSplit`, and `subsetTaskCount` to match across shards;
  `run_merge` selects the compact gate adjudicator when shards
  declare a `subsetSpecHash`; `write_gate_summary` titles the
  compact lane separately.
- No new npm scripts. The existing
  `arc:phase3:rawgrid-gate-v2:{shard,merge}` wrappers pass through
  the new flag.

**Smoke-test fingerprint** (CPU, `--limit-aux-tasks 4`,
`--probe-epochs 1`, single seed): shard run produced 7 subset tasks
+ 4 aux tasks, `pttestInstanceCount=2`, `testLodoInstanceCount=6`,
`validationInstanceCount=5`, `subsetSpecHash`
`5759038A94DBCAF1A192C59C27CA201E13D447E7463172A43ACEE606345A836B`.
Merge of the single shard correctly produced
`compact_full_grid_control_floor` as expected (probe epochs = 1, no
learning), exercising the full compact-gate adjudication path. The
smoke output directories were deleted before commit.

**Launch posture for the first compact-7 tranche** (per
`PHASE3B_COMPACT_SUBSET.md` §7 and §8): single seed `20260529`,
`--device cuda`, full 120-epoch budget, with the pre-commit
freeze-marker discipline applied. The shard receipt is then merged
(N=1) to produce the binding compact receipt with full gate
adjudication. If the floor clears, run the `signature_palette` arm
under the same subset and compare. If the floor does not clear, file
a compact-Branch-C bounded-failure amendment and close Branch B in
the deterministic-low-capacity family.

**Verdict impact**: no prior verdict changes. The V1 and V2 binding
receipts above remain Branch C bounded failures unchanged; the
Phase 3 lane status moves from "two full-grid controls floored" to
"two full-grid controls floored; one Branch B diagnostic admitted,
no binding compact receipt yet."

**Public-language constraint**: see `PHASE3B_COMPACT_SUBSET.md` §10.
Additions to `docs/prereg/arc/README.md` will land in the
compact-receipt verdict amendment, not in this admission amendment.
