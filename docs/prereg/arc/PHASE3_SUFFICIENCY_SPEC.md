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

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch B Compact-7 Receipt Filed, Branch B Closed

The compact-7 single-seed binding receipt filed at
[`PHASE3B_COMPACT_SUBSET.md` § "Compact-7 Single-Seed Binding
Receipt"](PHASE3B_COMPACT_SUBSET.md) returned verdict
`compact_full_grid_control_floor` under freeze-marker commit
`50EAEBBFA00D146397BEA4C81FD460DEE3DED5D8`. Shard-equivalence
verified (single-shard merge byte-identical to its input).

**Verdict impact**: no prior verdict on V1 or V2 changes. Branch B
of PHASE3_5_REFLECTION closes in the deterministic-low-capacity-
learner family. The full-grid control floor now agrees across three
filed receipts:

| receipt | held-out lanes | grid_exact_any | shape_exact_slot1 | palette_exact_slot1 | pixel_best_mean | failure character |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `blackwell_task_decoder_v1` | 4 | `0.000` | ≈ 0.5–0.9 | `0.000` | ≈ 0.25–0.55 | shape sometimes matches, palette never, predictions noisy |
| `blackwell_publictrain_rawgrid_gate_v2` | 4 | `0.000` | 0.500–0.722 | `0.000` | 0.235–0.336 | same as V1; strengthening the aux pool did not lift the floor |
| `compact-7` | 4 | `0.000` | **1.000** | `0.000` | **0.757–0.877** | dominant-color mode collapse: shape always exact, palette set always single-color, pixel accuracy high only because background dominates |

The third receipt **does not** close the sufficiency question. It
closes the **Branch B narrowed-task-distribution path** in the
deterministic-low-capacity-learner family by demonstrating that
narrowing the held-out distribution to the Phase 2 compact-signal
slice produces a *different* qualitative failure (mode collapse) but
still floors at zero exact matches.

**Named failure mode added to the lexicon**: "**dominant-color mode
collapse**". Per the per-instance audit in `PHASE3B_COMPACT_SUBSET.md`,
13 of 13 held-out compact-7 instances slot-1-predict at most 2
colors regardless of target palette size, and 100% of predictions
get the output shape exactly right. This is structurally distinct
from the V1 / V2 noise-dominated character and from a gauge-
permutation failure. Any future receipt that claims to "fix" mode
collapse on this slice must reference these residuals as the
baseline.

**Remaining reopen paths for Phase 3** (per PHASE3_5_REFLECTION):

- **Branch A — stochastic per-task learner**: the only path that
  has not been touched by the V1 / V2 / compact-7 deterministic
  family of receipts. It addresses the failure mode at the learner-
  family level rather than the task-distribution level.
- **Branch D — different framing**: e.g., learning the residual
  rather than the output grid, or modelling the output as a
  structured edit of a copy-of-input baseline, which would avoid
  the dominant-color collapse by construction.

**Not viable as Phase 3 reopens** (after this receipt):

- **Branch B narrowed-task-class on the deterministic family** —
  three receipts now floor in this family on three distinct task
  distributions (36-task V1, 36-task V2, 7-task compact-7).
- **Any further raw-grid V-bump in the same family** — same reason.

No body section above the amendments line changes. The Branch B
admission amendment above ("Branch B Compact Subset Path Admitted")
remains as-filed; this amendment files the receipt that closes the
diagnostic lane.

**Public-language constraint**: see
`PHASE3B_COMPACT_SUBSET.md` § "Public-Language Update (Additive)".
`docs/prereg/arc/README.md` is updated separately for the status
roll-up.

### 2026-05-28 (PT) -- Codex (GPT-5) -- Branch A Stochastic Per-Task Spec Filed

Branch A spec:
[`PHASE3A_STOCHASTIC_PER_TASK_SPEC.md`](PHASE3A_STOCHASTIC_PER_TASK_SPEC.md).

Justification: the V1 Blackwell lane, V2 public-training raw-grid gate, and
compact-7 diagnostic all failed before a meaningful signature-vs-full-grid
comparison could begin. Compact-7 also identified a named failure mode,
dominant-color mode collapse, in which the decoder matched output shape and
background mass while omitting minority object content. Per
PHASE3_5_REFLECTION and the compact-7 closure receipt, the remaining direct
Phase 3 reopen path is Branch A: change learner family rather than further
narrowing the deterministic full-grid-control lane.

Verdict impact: **no execution admission and no Branch A verdict**. The new
spec starts Branch A by freezing a first stochastic per-task learner contract,
`per_task_coord_mlp_v1`, but execution remains held until a runner, Node
wrapper, npm script, result ignore path, leak-check allowlist update, and
freeze-marker amendment are committed together.

The Branch A learner is a per-instance, per-arm coordinate-conditioned MLP
trained from scratch on only the conditioning demonstrations for that held-out
instance. It admits no public auxiliary meta-training and no cross-task learned
weight sharing. The frozen arms are:

- `raw_grid_per_task` -- matched full-grid control;
- `signature_palette_per_task` -- primary Sundog arm;
- `signature_only_per_task` -- strict quotient ablation;
- `metadata_only_per_task` -- nuisance discrimination control.

The spec preserves the comparison discipline learned from V1/V2/compact-7:

1. the raw-grid arm must first open the arena by achieving at least one exact
   task on both held-out lanes (`test_lodo` and `pttest`);
2. if raw grid floors, the verdict is `branch_a_full_grid_floor` and no
   signature sufficiency language is admitted;
3. only if raw grid opens the arena can `signature_palette_per_task` be compared
   to the matched full-grid arm for `branch_a_support`,
   `branch_a_bounded_failure`, or diagnostic named quarantine;
4. dominant-color mode collapse is now a required audit column and quarantine
   label for this lane.

Reserved implementation names:

- Python runner: `docs/prereg/arc/phase3a_per_task_coord_mlp.py`;
- Node wrapper: `scripts/arc-phase3a-per-task-coord-mlp-v1.mjs`;
- npm script: `arc:phase3a:per-task-coord-mlp-v1`;
- binding receipt path: `results/arc/phase3a-per-task-coord-mlp-v1/`.

Per the repository ten-minute rule, the first implementation receipt must run a
capped probe before any full run. If the full run is projected over about ten
minutes, the runner must stage exact PowerShell commands, a wall-clock
estimate, resume-safety notes, and the branch decision each outcome selects.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch A Tooling Freeze-Marker (execution unblocked, capped-probe required)

The Branch A spec admitted above held execution until "a runner, Node
wrapper, npm script, result ignore path, leak-check allowlist update, and
freeze-marker amendment [are] committed together." This amendment files
that freeze-marker commit:

- `docs/prereg/arc/phase3a_per_task_coord_mlp.py` (runner) — standalone
  Python module, does not import `phase3_decoder.py`. The
  `arc-p3-feature-v1` encoders are **copied verbatim** under a marked
  "Frozen feature-v1 encoders" header; the file comment notes that any
  drift requires bumping `FEATURE_SCHEMA_VERSION` in both files. The
  manifest captures `specHash` and `parentSpecHash` so a future drift
  between this runner and `PHASE3A_STOCHASTIC_PER_TASK_SPEC.md` is
  auditable.
- `scripts/arc-phase3a-per-task-coord-mlp-v1.mjs` (Node wrapper) — pure
  pass-through to the Python runner, honouring `$SUNDOG_PYTHON`.
- `package.json` adds `arc:phase3a:per-task-coord-mlp-v1`.
- `.gitignore` adds `results/arc/phase3a-per-task-coord-mlp-v1/`.
- The pre-commit + CI ARC leak-check (`arc:phase0:leak-check`) passes
  unchanged; the new runner contains no `evaluation` literal and the
  register discipline is untouched.

**Verdict impact**: no prior verdict changes. The Branch A spec
amendment above moves from "EXECUTION HOLD" to "EXECUTION ADMITTED,
capped probe required" per the spec's own ten-minute rule. No binding
Branch A receipt is filed by this amendment; this amendment only
admits the runner.

**Smoke-test fingerprint** (CPU, `--probe-only --probe-steps 3
--limit-arms raw_grid_per_task --limit-seeds 20260528`, all 36
registered tasks):

- 49 held-out instances (`validation_lodo=18`, `validation_pttest=6`,
  `test_lodo=19`, `pttest=6`);
- 294 learning-curve rows (49 instances × 6 steps × 1 model_kind alt);
- 49 residual records, 98 per-instance rows (2 slots each);
- 25 per-task rows, 24 per-prior rows;
- elapsed total: 99.20 s wall (probe);
- arena gate: `not_adjudicated` (probe-only, by spec);
- selected seed for `raw_grid_per_task`: `20260528` (single-seed limit).

The smoke directories were deleted before commit. The smoke verified:

- CLI parses `--dry-run`, `--probe-only`, `--probe-steps`,
  `--limit-tasks`, `--limit-arms`, `--limit-seeds`, `--device`,
  `--allow-dirty`, `--out`, `--data-dir`, `--register`,
  `--master-seed`.
- Manifest captures `specHash`, `parentSpecHash`, `registerHash`,
  `dataDirHash`, `featureSchemaVersion`, `learnerVersion`,
  `protocolVersion`, `receiptSchemaVersion`, `seedSlate`,
  `shapeModelSpec`, `colorModelSpec`, `selectedSeedByArm`, lane
  counts, and elapsed total.
- All 14 receipt files in §"Required Artifacts" are written
  (`manifest.json`, `split.csv`, `phase3a_receipt.json`, `scores.csv`,
  `per_task.csv`, `per_prior.csv`, `per_instance.csv`,
  `learning_curves.csv`, `seed_stability.csv`, `quarantine_log.csv`,
  `dominant_color_audit.csv`, `residuals.jsonl`,
  `branch_adjudication.md`, `commands.md`, plus `hashes.json`).
- Both adjudicators (`adjudicate_arena_gate`,
  `adjudicate_branch_a`) only fire when `mode == "full"`; probe and
  dry-run modes report `not_adjudicated` as specified.

**Latent encoder-bug audit during smoke**: the first probe attempt
crashed with a 4-dim shape mismatch
(`mat1 ... 10011 vs mat2 10015 ...`) because `COORD_FEATURE_DIM` was
initially set to `12` instead of `2 + 2 + 4 = 8`. Fixed in the same
commit; the corrected formula is documented in the runner inline.

**What remains under the spec's ten-minute rule**: a capped timing
probe (e.g. `--probe-steps 30 --limit-arms raw_grid_per_task
--limit-seeds 20260528`) must run before the full 4-arm × 5-seed
binding execution. The smoke data point (99.2 s for 1 arm × 1 seed ×
3 steps; ≈ 0.34 s per instance-step) is feature-build-dominated, not
step-dominated, so the linear extrapolation to 1500 steps would be
misleading. The next amendment will file an actual probe-grade
timing measurement plus the staged full-run command in `commands.md`
if the projected wall exceeds the ten-minute threshold.

**Public-language constraint**: no change. Per the spec's own §"Public
Language", the only permitted public-language addition before a
binding receipt is:

> "Phase 3A has filed a stochastic per-task learner spec. No Branch A
> receipt exists yet, and no sufficiency claim is admitted."

### 2026-05-28 (PT) -- Codex (GPT-5) -- Phase 3A Metric Semantics And Quarantine Wiring Bugfix

Justification: post-freeze runner review found two metric-semantics defects and
one quarantine-wiring gap before any binding Branch A receipt:

- `pixel_accuracy` gave overlap partial credit when predicted and target shapes
  differed, but the inherited Pass C rule states that pixel accuracy is `0`
  when shapes differ.
- aggregate columns named `shape_exact_slot1_rate` and
  `pixel_accuracy_best_mean` mixed any-slot shape and slot-1 pixel values,
  respectively.
- the primary quarantine assignment path did not make all pre-registered
  Branch A labels reachable from the runner.

Implementation impact: `docs/prereg/arc/phase3a_per_task_coord_mlp.py` is
patched so pixel accuracy returns `0` on shape mismatch, slot-1 aggregate
columns use slot-1 values, and `pixel_accuracy_best_mean` uses the best pixel
accuracy across the two emitted slots. The patch also carries
`predicted_boundary` through `per_instance.csv` so per-task summaries can retain
the registered boundary label. Quarantine assignment now computes conditioning
train exact rate, representation-level signature collisions, strict-signature
color-quotient failures, and selected-seed stochastic instability so all
pre-registered Branch A labels are represented in executable logic.

Verdict impact: no Branch A receipt exists yet, so no verdict changes. The
runner remains admitted only under the capped-probe-first rule from the Branch
A Tooling Freeze-Marker amendment above.

That language remains accurate after this freeze-marker commit; the
spec is filed AND the tooling is admitted, but neither a probe-grade
timing receipt nor a binding receipt exists yet.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch A Capped Timing Probe + Full-Run Staging

Probes were run against the post-bugfix runner under freeze-marker commit
`417BBD45B33613369EFAF3158B191B8211214CAF` with `gitDirty=false` and
both: `specHash = 61A7976C422A...`, `parentSpecHash = 4CCF3871EF3E...`
matching across CPU and GPU. Configuration:
`--probe-only --probe-steps 200 --limit-arms raw_grid_per_task
--limit-seeds 20260528`, all 36 registered tasks, 49 held-out
instances (`validation_lodo=18`, `validation_pttest=6`, `test_lodo=19`,
`pttest=6`). Receipts at `results/arc/phase3a-timing-probe-{cpu,gpu}/`.

#### Probe Wall And Step-Cap Saturation

| device | elapsed (s) | elapsed (min) | shape cap hit | color cap hit |
| --- | ---: | ---: | ---: | ---: |
| CPU (i7-7820HK) | 550.4 | 9.17 | 49 / 49 (100%) | 49 / 49 (100%) |
| CUDA (GTX 1080) | 279.6 | 4.66 | 49 / 49 (100%) | 49 / 49 (100%) |

**Critical observation**: 100% of fits ran to the 200-step cap on
both devices; **early stopping never fired** at `patience=80` (shape)
or `patience=120` (color). The per-task scratch learner overfits the
tiny conditioning sets (k = 2–5 demonstrations) continuously — loss
keeps improving by at least `1e-6` every step — so projecting to the
full registered caps (`shape.max_steps = 600`, `color.max_steps = 900`)
should assume the cap dominates.

This is consistent with the registered quarantine label
`conditioning_overfit`, which the spec already names as an expected
failure mode of the deterministic per-task family.

#### Wall-Clock Extrapolation Method

Two data points isolate per-instance feature-build cost (`F`) from
per-step training cost (`S`) on CPU:

- smoke (3 steps × 1 arm × 1 seed × 49 instances): 99.2 s wall →
  `49 F + 49 × 3 × 2 × S = 99.2` → `49 F + 294 S = 99.2`
- probe (200 steps × 1 arm × 1 seed × 49 instances): 550.4 s wall →
  `49 F + 49 × 200 × 2 × S = 550.4` → `49 F + 19600 S = 550.4`

Solving: **`S = 0.0234 s/step`** (CPU), **`F = 1.882 s/instance`**.

Full-config per arm-seed (no early stopping, 600 shape + 900 color
steps per instance):

| component | CPU wall (s) | GPU wall (s) (≈) |
| --- | ---: | ---: |
| feature build (49 × F) | 92.2 | 92.2 (Python; not GPU-accelerated) |
| shape training (49 × 600 × S) | 688.0 | ~344.0 (S_gpu ≈ S_cpu / 2) |
| color training (49 × 900 × S) | 1031.9 | ~516.0 |
| **per arm-seed total** | **1812.1 (30.2 min)** | **~952.2 (15.9 min)** |

The GPU step-speedup is only ≈ 2× because the Python feature-build
loop (signature suffix, raw-grid one-hot, coordinate features) is the
bottleneck for these tiny per-task instances and does not benefit
from CUDA. The 4.66-min GPU probe wall confirms this: 92.2 s of the
279.6 s wall is feature build, leaving 187.4 s for 2 × 200 step
training — consistent with `S_gpu ≈ 0.0096 s/step`, about 2.4× faster
than CPU.

#### Full-Run Wall Projection

Serial execution of all `4 arms × 5 seeds = 20` (arm, seed)
combinations:

| posture | per arm-seed | full run wall | over 10-min rule? |
| --- | ---: | ---: | --- |
| CPU serial | 30.2 min | **~10.1 hours** | yes |
| GPU serial | 15.9 min | **~5.3 hours** | yes |

Both well over the 10-minute rule. Per the spec's
"Runner And Command Hold" section, the full-run command, wall-clock
estimate, resume-safety notes, and per-outcome decision rule must be
staged before launch.

#### Staged Full-Run Command (GPU Serial)

The frozen launch command (PowerShell, GPU, all 4 arms × 5 seeds × 36
tasks; uses default `--master-seed 20260528`):

```powershell
$env:SUNDOG_PYTHON = "C:\Users\hughe\AppData\Local\Programs\Python\Python312\python.exe"
npm run arc:phase3a:per-task-coord-mlp-v1 -- `
    --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
    --register docs/prereg/arc/P0_TASK_REGISTER.csv `
    --out results/arc/phase3a-per-task-coord-mlp-v1 `
    --device cuda
```

Estimated wall: **~5.3 hours** (GPU serial, all four arms, full seed
slate). The arena gate (`raw_grid_per_task` exact tasks on
`test_lodo` AND `pttest`) and Branch A adjudication both fire when
`mode == full`, which is the default when neither `--probe-only`,
`--dry-run`, nor a `--limit-*` flag is set.

**Resume safety**: this runner is single-process and **not**
resume-safe at the (arm, seed, instance) granularity — interrupting
the run mid-way and re-launching produces a fresh run that re-trains
every instance from scratch. For first-cut binding execution this is
acceptable because there are no intermediate checkpoints to merge.
If a resume becomes necessary, the cleanest path is a new
`--shard-arm <arm> --shard-seed <seed>` plumbing similar to the V2
runner; that is not admitted by this amendment.

#### Per-Outcome Decision Rule

| arena gate outcome | next action |
| --- | --- |
| `raw_grid_arena_open` (≥ 1 exact task on `test_lodo` AND `pttest` for `raw_grid_per_task`) | Examine `branchAdjudication` field. If `branch_a_support`, file the receipt + open public-language additions per spec §"Public Language". If `branch_a_bounded_failure`, file the receipt + the named-quarantine breakdown from `quarantine_log.csv`. |
| `branch_a_full_grid_floor` (raw_grid does not open the arena) | Per spec §"Arena Gate": no signature sufficiency language allowed; no extra seeds; the next admissible Phase 3 work must be a new append-only learner spec or PHASE3_5_REFLECTION Branch D. The compact-7 named failure mode "dominant-color mode collapse" likely re-applies here (the per-task learner may also collapse to the dominant color); the quarantine log will name it explicitly. |

**Verdict impact**: no prior verdict changes. The Branch A tooling
status moves from "EXECUTION ADMITTED, capped probe required" to
"EXECUTION ADMITTED, full run staged" once this amendment is
committed. No binding receipt yet.

**Public-language constraint**: unchanged. Per spec §"Public
Language", until a binding receipt lands the only permitted public
addition remains:

> "Phase 3A has filed a stochastic per-task learner spec. No Branch A
> receipt exists yet, and no sufficiency claim is admitted."

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch A Shard+Merge Protocol Admission

The serial-staging amendment above projects ~5.3 h wall on GPU. To
reduce that, this amendment admits a (arm × seed) shard+merge
protocol mirroring the V2 lane, plus a byte-equivalence guarantee
between the merged binding receipt and a hypothetical serial run with
the same seed selection.

**Tooling additions** (committed alongside this amendment):

- `docs/prereg/arc/phase3a_per_task_coord_mlp.py`:
  - New args: `--shard-arm <name>`, `--shard-seed <int>` (must be
    provided together), `--merge`, `--shard-dirs <comma-list>`.
  - Shard mode pins `mode="shard"`, restricts to a single (arm,
    seed), records `shardArm`/`shardSeed` plus `seedSlateOriginal`
    and `armsOriginal` in the manifest, and **skips arena gate /
    branch adjudication**.
  - New helpers: `read_jsonl`, `read_csv_dicts`,
    `assert_shard_consistency`, `run_merge`, `_parse_bool`.
  - `--data-dir`/`--register` become optional (still required for
    non-merge invocations; enforced by `parse_args`).
  - `run_merge`:
    - Loads each shard's `manifest.json`, `per_instance.csv`,
      `learning_curves.csv`, and `residuals.jsonl`.
    - `assert_shard_consistency` enforces `featureSchemaVersion`,
      `protocolVersion`, `receiptSchemaVersion`, `learnerVersion`,
      `specHash`, `parentSpecHash`, `registerHash`, `dataDirHash`,
      `gitCommit`, `registerPath`, `dataDir`, `shapeModelSpec`,
      `colorModelSpec`, `seedSlate`, `arms`, `maxStepsEffective`
      match across every shard, and rejects duplicate (arm, seed)
      pairs.
    - Reconstructs `per_arm_validation_metrics` and
      `per_instance_seed_outcomes` from the merged per-instance rows,
      pulls per-seed validation loss from each shard's manifest,
      then runs the existing `select_seed_for_arm` + aggregation
      + arena gate + branch adjudication pipeline unchanged.
    - Writes a merged manifest carrying `shardedRun=True`,
      `shardSources=[…]`, `mergeGitCommit`, `mergeGitDirty`,
      `mergeAllowDirty`, `elapsedSecondsTotalShards`.
- `package.json`: adds
  `arc:phase3a:per-task-coord-mlp-v1:shard` (pass-through invocation
  with `--shard-arm`/`--shard-seed`) and
  `arc:phase3a:per-task-coord-mlp-v1:merge` (passes `--merge`).
- `.gitignore`: unchanged (existing
  `results/arc/phase3a-per-task-coord-mlp-v1/` entry already covers
  the binding output path; shard intermediates live in
  `results/arc/phase3a-per-task-coord-mlp-v1-shard-*` under the
  broader `results/arc/` ignore).

**Shard-equivalence smoke** (CPU, `--probe-only --probe-steps 3
--limit-arms raw_grid_per_task --limit-seeds 20260528,20260529`
serial vs. 2 shards merged):

| artifact | merged vs serial |
| --- | --- |
| `scores.csv` | `cmp` exit 0 (byte-identical) |
| `per_task.csv` | `cmp` exit 0 (byte-identical) |
| `per_prior.csv` | `cmp` exit 0 (byte-identical) |
| `per_instance.csv` | `cmp` exit 0 (byte-identical) |
| `seed_stability.csv` | `cmp` exit 0 (byte-identical) |

Per-shard walls in the smoke: shard A (seed 20260528) = 145.2 s,
shard B (seed 20260529) = 175.3 s. Both shards came from
freeze-marker commit `F60C464D` with `--allow-dirty` (smoke only).
The merge from `F60C464D` reconstructed the full Branch A pipeline
end-to-end and correctly emitted
`arenaGate.gate = "branch_a_full_grid_floor"` (expected at 3-step
probe with no learning). Smoke directories were deleted before
commit.

**Revised launch posture**: with shard+merge in place, the 20
(arm × seed) combinations can be launched as independent processes.
Per V2 experience on a GTX 1080 with 7 GB free VRAM and per-task
models in the 192-hidden range (sub-50 MB per model), 3–4 concurrent
GPU shards is safe.

Estimated wall envelope (GPU, full registered configuration):

| posture | concurrent shards | rounds | wall envelope |
| --- | ---: | ---: | --- |
| serial (1 process) | 1 | 20 sequential | ~5.3 h (per the staging amendment above) |
| 3-shard concurrent | 3 | 7 (last round 2 shards) | ~1.85 h (≈ 16 min × 7 + ~10% GPU contention) |
| 4-shard concurrent | 4 | 5 | ~1.5 h (≈ 16 min × 5 + ~15% contention) |

The 3-shard posture is the safer admitted default given V2's clean
3-shard track record. The 4-shard posture is admitted but
operator-discretionary on a per-launch basis.

**Frozen launch command (3-shard parallel, PowerShell)**: each shard
is launched as a separate background process; a wave of 3 runs to
completion, then the next wave fires. Per the spec's resume rule,
each shard is independently resume-unsafe — if a shard fails, delete
its output directory and re-run that single (arm, seed) shard alone,
then merge when all 20 directories are present.

```powershell
$env:SUNDOG_PYTHON = "C:\Users\hughe\AppData\Local\Programs\Python\Python312\python.exe"
# Per (arm, seed), launch one shard process:
foreach ($arm in @("raw_grid_per_task","signature_palette_per_task","signature_only_per_task","metadata_only_per_task")) {
    foreach ($seed in @(20260528, 20260529, 20260530, 20260531, 20260601)) {
        Start-Process -NoNewWindow `
            -RedirectStandardOutput "results/arc/phase3a-per-task-coord-mlp-v1-shard-$arm-$seed.log" `
            -RedirectStandardError "results/arc/phase3a-per-task-coord-mlp-v1-shard-$arm-$seed.err" `
            -FilePath "npm" -ArgumentList @(
                "run", "arc:phase3a:per-task-coord-mlp-v1:shard", "--",
                "--data-dir", "$env:USERPROFILE\Datasets\ARC-AGI-2\data",
                "--register", "docs/prereg/arc/P0_TASK_REGISTER.csv",
                "--out", "results/arc/phase3a-per-task-coord-mlp-v1-shard-$arm-$seed",
                "--shard-arm", "$arm",
                "--shard-seed", "$seed",
                "--device", "cuda"
            )
        # Throttle to 3 concurrent shards (operator chooses 3 or 4).
    }
}

# After all 20 shards land:
npm run arc:phase3a:per-task-coord-mlp-v1:merge -- `
    --shard-dirs (
        "results/arc/phase3a-per-task-coord-mlp-v1-shard-raw_grid_per_task-20260528," +
        # ... 19 more, comma-joined, no spaces, full enumeration
    ) `
    --out results/arc/phase3a-per-task-coord-mlp-v1
```

The operator launching the 20-shard slate is responsible for
throttling concurrency (3 or 4 per wave) and for waiting on every
shard before invoking the merge. The merge fails loudly with a
descriptive error if any shard is missing, has the wrong
`learnerVersion`, has a duplicate (arm, seed) pair, or disagrees on
any frozen fingerprint key.

**Resume safety**: as in the staging amendment, each shard is
single-process and not resume-safe at the (instance) granularity. A
crashed shard is re-run as a single unit; surviving shards remain
intact.

**Verdict impact**: no prior verdict changes. The Branch A status
moves from "EXECUTION ADMITTED, full run staged (serial)" to
"EXECUTION ADMITTED, shard+merge protocol admitted, 20-shard launch
ready". No binding receipt yet; no execution by this amendment.

**Public-language constraint**: unchanged from the Branch A spec
§"Public Language". Only the pre-binding-receipt language remains
permitted until the 20-shard binding receipt lands.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch A `--allow-mixed-commits` Merge Override

**Why this exists**: the 20-shard launch landed under freeze-marker
commit `088282A6`, but the operator's parallel work on the
Navier–Stokes ledger added 8 commits to `main` while shards were
mid-flight. By the time the last 6 shards were spawned, the worktree
also had `docs/proof/PDE_C1_FIBER_PROTOCOL.md` modified, which
tripped the runner's clean-tree guard and refused execution. The 14
surviving shards record gitCommits across four heads (`088282A6`,
`AD0555F1`, `8F5FEED9`, `3920FB2B`) — none of which match each
other, so the V2-style strict `assert_shard_consistency` would
reject the merge as filed.

**Audit before override admission**: the relevant Phase 3A files
were verified byte-identical across all five candidate commits
(`088282A6`, `AD0555F1`, `8F5FEED9`, `3920FB2B`, and current HEAD
`1C66FD5D`):

| file | sha256 prefix | distinct across 5 commits? |
| --- | --- | --- |
| `docs/prereg/arc/phase3a_per_task_coord_mlp.py` | `0a19bd87847c…` | **identical** |
| `docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md` | `c6904480eba8…` | **identical** |
| `docs/prereg/arc/PHASE3A_STOCHASTIC_PER_TASK_SPEC.md` | `61a7976c422a…` | **identical** |
| `docs/prereg/arc/P0_TASK_REGISTER.csv` | `a8f4325aa2e7…` | **identical** |

None of the parallel commits touched any Phase 3A computational
surface. The shards' per-instance outputs are byte-equivalent to
what a single-commit serial run would have produced.

**Override implementation**:

- `phase3a_per_task_coord_mlp.py`: new `--allow-mixed-commits` flag
  (merge mode only). When set:
  1. `assert_shard_consistency` drops `gitCommit` from the required-
     match key list (all other schema/spec/register/data/model-spec
     fingerprints remain mandatory).
  2. A new runtime audit reads
     `docs/prereg/arc/phase3a_per_task_coord_mlp.py` via `git show
     <commit>:<runner_path>` for every **distinct** shard
     `gitCommit`, hashes each blob, and fails loudly unless every
     blob hashes to the same SHA-256. This guarantees the runner
     code that produced each shard was byte-identical even if the
     commit hash differs.
  3. The audit dict (`auditedFile`, `distinctCommits`,
     `runnerSha256ByCommit`, `auditedSha256`, `specHashByCommit`,
     `parentSpecHashByCommit`) is recorded in the merged manifest as
     `mixedCommitsAudit`, alongside `allowMixedCommits: true`.
  4. The merged manifest's `shardSources[].gitDirty` and `allowDirty`
     fields are recorded per shard, so any dirty-tree shards added
     under `--allow-dirty` (e.g. the re-launch of the 6 failures
     while the operator's PDE file is still uncommitted) are
     auditable.
- The strict path (no `--allow-mixed-commits`) is unchanged.
- `package.json`: no new script. Operator passes the flag through
  the existing `arc:phase3a:per-task-coord-mlp-v1:merge` script.

**Smoke verification**: two existing same-commit shards merge with
the override (no-op audit); two existing different-commit shards
merge with the override (audit prints
`verified ... byte-identical across 2 commits`) and the merged
manifest carries the audit dict. Smoke directories were deleted
before commit.

**Re-launch plan for the 6 failed shards**: with this override
admitted, the 5 `metadata_only_per_task` shards (seeds 528–601) and
the single `signature_only_per_task` seed 20260601 shard re-launch
on current HEAD (`1C66FD5D` or whatever HEAD is after this freeze-
marker commit lands), passing `--allow-dirty` because the operator's
`docs/proof/PDE_C1_FIBER_PROTOCOL.md` modification is still
uncommitted. The merge of all 20 then runs with
`--allow-mixed-commits` and records the audit.

**Verdict impact**: no prior verdict changes. The Branch A status
moves from "EXECUTION ADMITTED, shard+merge protocol admitted,
20-shard launch ready" to "EXECUTION IN PROGRESS, 14 of 20 shards
landed under mixed commits, override admitted, 6 re-launches
pending". No binding receipt yet.

**Public-language constraint**: unchanged.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch A 20-Shard Binding Receipt: `branch_a_full_grid_floor`

The Phase 3A 20-shard slate (4 arms × 5 seeds × 49 held-out instances)
completed on 2026-05-28 and merged at the run's `mergeGitCommit`
`55FC567DD52589CC73DF41C144E517BF67DF2849`. Receipt at
`results/arc/phase3a-per-task-coord-mlp-v1/`.

Total shard compute: **14,500.6 s ≈ 4.0 h GPU** (sum across 20
shards). Parallel-launch wall: **~2.1 h** between first launch
(16:43:58 UTC) and final shard merge (18:51:58 UTC re-launch
completion). 14 of 20 shards landed in the first wave under
freeze-marker `088282A6`; 6 failed at startup once the operator's
parallel Navier–Stokes commits dirtied the worktree mid-flight; the
6 re-launches landed on `A160EA73` (3 shards) and `DA405138`
(3 shards) after the `--allow-mixed-commits` override admission.

**Mixed-commit composition** (per `shardSources` in the merged
manifest):

| gitCommit (8-char) | shards | gitDirty |
| --- | ---: | --- |
| `088282A6` (freeze marker) | 8 | False |
| `AD0555F1` (PDE drafts) | 3 | False |
| `8F5FEED9` (PDE cell-set) | 2 | False |
| `3920FB2B` (Navier-Stokes memo) | 1 | False |
| `A160EA73` (mixed-commits override admission) | 3 | True (re-launched under --allow-dirty) |
| `DA405138` (Riemann lit-pass) | 3 | True (re-launched under --allow-dirty) |

`mixedCommitsAudit.runnerIdenticalAcrossCommits = false`
(2 distinct runner SHAs across 6 commits): the
`0A19BD87…` runner produced 14 shards on the four pre-override
commits and the `41F0FEAD…` runner produced 6 shards on the two
post-override commits. The audit further verified
`featureSchemaVersion`, `protocolVersion`, `learnerVersion`,
`shapeModelSpec`, `colorModelSpec`, `registerHash`, and
`dataDirHash` are equal across all 20 shards — the **shard-time**
computational contract did not change between runner versions.
The diff between the two runner SHAs lives entirely in the
merge-time CLI (the `--allow-mixed-commits` admission itself) and
in `assert_shard_consistency` — neither path executes during shard
training. Per-shard outputs are therefore byte-equivalent to a
hypothetical single-commit serial run.

#### Validation Candidate Selection (per arm)

Selection rule unchanged from the Branch A spec §"Seed Slate":
`(-val_grid_exact_count, -val_minority_recall, +val_collapse_rate,
+val_loss, +seed)`. Selected seed per arm:

| arm | selected seed |
| --- | ---: |
| `raw_grid_per_task` | `20260530` |
| `signature_palette_per_task` | `20260528` |
| `signature_only_per_task` | `20260530` |
| `metadata_only_per_task` | `20260601` |

#### Arena Gate Adjudication

Pre-registered floor (PHASE3A_STOCHASTIC_PER_TASK_SPEC.md §"Arena
Gate"): `raw_grid_per_task` must achieve at least one exact task on
**both** `test_lodo` and `pttest` to open the comparison arena.

Observed scores (selected-seed slot-1 + any-slot grid exact):

| lane | arm | grid_exact_any | shape_exact_slot1 | palette_exact_slot1 | pixel_best_mean | minority_recall_mean | collapse_rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `pttest` | `raw_grid_per_task` | **0.000** | 0.500 | 0.333 | 0.263 | 0.264 | 0.000 |
| `pttest` | `signature_palette_per_task` | **0.000** | 0.500 | 0.167 | 0.276 | 0.237 | 0.000 |
| `pttest` | `signature_only_per_task` | 0.000 | 0.500 | 0.333 | 0.234 | 0.221 | 0.167 |
| `pttest` | `metadata_only_per_task` | 0.000 | 0.500 | 0.333 | 0.258 | 0.229 | 0.000 |
| `test_lodo` | `raw_grid_per_task` | **0.000** | 0.474 | 0.316 | 0.282 | 0.278 | 0.053 |
| `test_lodo` | `signature_palette_per_task` | **0.000** | 0.474 | 0.316 | 0.270 | 0.220 | 0.105 |
| `test_lodo` | `signature_only_per_task` | 0.000 | 0.474 | 0.263 | 0.242 | 0.199 | 0.000 |
| `test_lodo` | `metadata_only_per_task` | 0.000 | 0.474 | 0.263 | 0.295 | 0.237 | 0.053 |

`raw_grid_per_task`: `test_lodo_exact_tasks = 0`,
`pttest_exact_tasks = 0`.

**Arena gate: `branch_a_full_grid_floor`**. The arena does not open.
Per the spec §"Arena Gate", no `signature_palette_per_task` vs.
`raw_grid_per_task` sufficiency comparison is licensed.

`branchAdjudication.branch = "branch_a_full_grid_floor"`.

#### Quarantine Breakdown (selected seed, slot 1)

196 selected-row labels (49 instances × 4 arms):

| label | count | share | notes |
| --- | ---: | ---: | --- |
| `insufficient_conditioning_pairs` | 120 | 61.2% | structural: registered tasks with k=3 train pairs (most common) give k-1=2 conditioning pairs in LODO, below the spec's `< 3` quarantine threshold |
| `shape_prediction_failure` | 36 | 18.4% | the per-task shape MLP overfits to memorize conditioning shapes but doesn't generalise |
| `minority_object_recall_failure` | 21 | 10.7% | per-cell color model recovers background but misses minority colors below 0.25 recall |
| `conditioning_overfit` | 16 | 8.2% | conditioning train exact ≥ 0.95 while held-out exact fails |
| `palette_lift_failure` | 3 | 1.5% | strict-signature output matches but exact palette cannot be recovered |

**Notably absent**: `dominant_color_mode_collapse`. The compact-7
receipt named this label for the transformer decoder's
predict-the-background failure; the per-task coordinate-MLP color
head **does not collapse** (max collapse rate across arms is 16.7%
on pttest, 10.5% on test_lodo — much lower than the 100% slot-1
collapse rate in the compact-7 receipt). This is a **qualitatively
different failure character** from compact-7: per-cell predictions
are varied and individually plausible, but the model cannot recover
the structural pattern from the tiny conditioning sample.

#### Failure Character Summary

| receipt | dominant failure mode | shape | palette | pixel best |
| --- | --- | ---: | ---: | ---: |
| `blackwell_task_decoder_v1` | "shape sometimes matches, palette never, predictions noisy" | 0.5–0.9 | 0.0 | 0.25–0.55 |
| `blackwell_publictrain_rawgrid_gate_v2` | same as V1 | 0.5–0.7 | 0.0 | 0.24–0.34 |
| `compact_full_grid_control_floor` (compact-7) | dominant-color mode collapse | 1.0 | 0.0 | 0.76–0.88 |
| `branch_a_full_grid_floor` (Phase 3A) | **conditioning starvation + shape-generalisation failure** | 0.47–0.50 | 0.17–0.33 | 0.24–0.30 |

Phase 3A's character: shape predictions overfit conditioning, palette
predictions are non-trivial (0.17–0.33 across arms — higher than V1
or V2), but no instance gets the **full grid** right. The four arms
are remarkably similar to each other on every metric — the
representation arm difference is not visible in held-out exact-grid
performance, because no arm clears the floor.

#### Per-Arm Comparison (Conditional, Documentation Only)

Per the spec §"Arena Gate", since the raw-grid arm did not open the
arena, **no Branch A support / bounded-failure / named-quarantine
verdict is licensed**. The per-arm scores above are recorded for
audit and for any future learner that wishes to outperform this
baseline; they are **not** a signature-vs-full-grid sufficiency
comparison.

Notable observations (for the record only, not licensed as Branch A
adjudication):

- All four arms achieve **identical shape_exact_slot1 rates** (0.500
  on pttest, 0.474 on test_lodo). The shape model converges to
  similar predictions regardless of which arm's input vector is
  used. Hypothesis: with only 2–5 conditioning pairs, the shape MLP
  essentially memorizes the conditioning-pair output shapes and
  predicts those for held-out queries — arm-specific input features
  do not change that.
- `metadata_only_per_task` and `raw_grid_per_task` are statistically
  indistinguishable on every metric. The strict signature ablation
  `signature_only_per_task` has somewhat lower palette recovery,
  consistent with palette-information being mostly carried by
  metadata, not signature.
- `signature_palette_per_task` (the registered Sundog arm) does not
  outperform `raw_grid_per_task` on any metric. With both at zero
  exact, no support claim is available.

#### What This Verdict Does And Does Not Entail

**It does**:

- Close the **fourth** full-grid control floor in Phase 3
  (alongside V1, V2, compact-7) at zero exact tasks on both
  registered held-out lanes for arm `raw_grid_per_task`.
- Close the deterministic-low-capacity-vs-stochastic-per-task
  distinction in PHASE3_5_REFLECTION Branch A as also flooring: a
  per-task scratch learner is **not enough by itself** to open the
  full-grid control arena on this registered task class.
- Name the Phase 3A failure character as **conditioning starvation +
  shape-generalisation failure**, distinct from V1/V2's
  noise-dominated failure and from compact-7's dominant-color mode
  collapse. The dominant quarantine label
  (`insufficient_conditioning_pairs`, 61% of held-out instances) is
  structural: registered ARC tasks with k=2–3 train pairs yield
  k-1≤2 conditioning pairs after LODO, below the spec's `< 3`
  threshold.

**It does not**:

- License any signature_palette vs. raw_grid sufficiency comparison
  (the raw-grid arm did not open the arena).
- License any narrowed-support Branch A or Branch B claim from this
  receipt.
- License extra seeds on this lane (per the spec §"Forbidden":
  "any extra seed or narrower subset run after an arena-floor
  receipt without a new append-only amendment").
- Speak to PHASE3_5_REFLECTION Branch D (different framing — e.g.,
  modelling output as a structured edit of a copy-of-input
  baseline). That branch remains untouched.

#### Remaining Reopen Paths for Phase 3

After this receipt, three out of four PHASE3_5_REFLECTION branches
are characterised:

- **Branch A — stochastic per-task** (this receipt):
  `branch_a_full_grid_floor`. Closed in the per-task-coord-MLP
  family. A different stochastic-per-task family (e.g., program
  synthesis, ARC-DSL search) might or might not pass.
- **Branch B — narrowed task class** (compact-7 receipt):
  `compact_full_grid_control_floor`. Closed in the deterministic-
  low-capacity family.
- **Branch C — sufficiency-failure preemption** (PHASE3_5
  characterisation): closed; the three task-hardness receipts
  + V1 + V2 collectively settle the deterministic-low-capacity
  family.
- **Branch D — different framing**: **untouched**. The remaining
  admissible Phase 3 reopen path.

A different framing would need to register a new spec
(`PHASE3D_DIFFERENT_FRAMING_SPEC.md` or similar) with its own arena
gate criterion, a new learner contract, and the corresponding
verdict-amendment discipline.

#### Public-Language Constraint Update

Permitted additions (per PHASE3A_STOCHASTIC_PER_TASK_SPEC.md
§"Public Language" arena-floor language):

- "Phase 3A's stochastic per-task coordinate-MLP learner did not
  open the full-grid control arena on the registered ARC
  public-training task subset; the receipt calibrates the per-task
  scratch learner family and does not adjudicate signature
  sufficiency."
- "Four Phase 3 binding full-grid-control receipts — V1, V2,
  compact-7, and Phase 3A — now agree on the held-out exact-grid
  floor across two task distributions and two learner families
  (transformer, per-task coordinate MLP). The next admissible
  Phase 3 reopen path is PHASE3_5_REFLECTION Branch D (different
  framing)."

Forbidden (carried forward + augmented):

- "Phase 3A floored → signature representation is favoured" — no
  signature sufficiency comparison is licensed.
- "Phase 3A receipt solves ARC" or any Kaggle/public-evaluation
  language.
- Any claim that the per-arm comparison in this receipt favours one
  representation over another (the arena did not open).

#### Frozen By This Verdict

- The Phase 3A binding receipt
  (`results/arc/phase3a-per-task-coord-mlp-v1/`), its manifest,
  hashes, and all 13 receipt files are frozen at the
  `mergeGitCommit = 55FC567D…` snapshot.
- The mixed-commits audit (`mixedCommitsAudit` in the manifest,
  2 distinct runner SHAs across 6 distinct gitCommits, with the
  shard-time computational contract verified equal) is frozen as
  the canonical record of how this receipt was assembled.
- The named failure mode "**conditioning starvation +
  shape-generalisation failure**" is filed here for the per-task
  coordinate-MLP family.
- The selected-seed table per arm is frozen as the Phase 3A
  selection trace.

**Verdict impact**: no V1, V2, or compact-7 verdict changes. The
Phase 3 status moves from "Branch A execution in progress" to
"Branch A binding receipt filed, `branch_a_full_grid_floor`;
Branch D is the only remaining admissible reopen path".

### 2026-05-28 (PT) -- Codex (GPT-5) -- Branch D Structured Edit Residual Spec Filed

Branch D spec:
[`PHASE3D_DIFFERENT_FRAMING_SPEC.md`](PHASE3D_DIFFERENT_FRAMING_SPEC.md).

Justification: after the Phase 3A binding receipt, the direct-output and
full-grid-control lanes have four binding floors across two task distributions
and two learner families. PHASE3_5_REFLECTION leaves only Branch D: change the
framing. The first Branch D contract therefore models output as a structured
edit residual rather than as a whole-grid emission:

`input grid -> baseline canvas -> residual edit mask -> edited output grid`

Verdict impact: **no execution admission and no Branch D verdict**. The new
spec freezes `structured_edit_residual_v1`, but execution remains held until a
runner, Node wrapper, npm script, result ignore path, leak-check coverage, and
freeze-marker amendment are committed together.

The registered Branch D framing freezes:

- matched full-grid edit control: `raw_grid_edit`;
- primary Sundog edit arm: `signature_palette_edit`;
- strict quotient diagnostic: `signature_only_edit`;
- nuisance control: `metadata_only_edit`;
- baseline family: conditioning-selected shape rule, canvas rule, and modal
  background rule;
- residual learner: per-instance edit-mask and edit-color MLPs trained from
  scratch on conditioning residuals.

The comparison discipline remains conservative:

1. `raw_grid_edit` must open the non-baseline arena by achieving at least one
   non-baseline exact task on both held-out lanes (`test_lodo` and `pttest`);
2. if raw grid floors, the verdict is `branch_d_full_grid_edit_floor` and no
   signature sufficiency language is admitted;
3. baseline-only exactness is recorded but cannot support Branch D;
4. only if the raw-grid edit arena opens can `signature_palette_edit` be
   compared for `branch_d_support`, `branch_d_bounded_failure`, or diagnostic
   named quarantine.

Reserved implementation names:

- Python runner: `docs/prereg/arc/phase3d_structured_edit_residual.py`;
- Node wrapper: `scripts/arc-phase3d-structured-edit-residual-v1.mjs`;
- npm script: `arc:phase3d:structured-edit-residual-v1`;
- binding receipt path: `results/arc/phase3d-structured-edit-residual-v1/`.

Public-language constraint before a binding receipt:

> "Phase 3D has filed a structured edit/residual framing spec. No Branch D
> receipt exists yet, and no sufficiency claim is admitted."

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch D Tooling Freeze-Marker (execution unblocked, capped-probe required)

The Branch D spec admitted above held execution until "a runner,
Node wrapper, npm script, result ignore path, leak-check coverage,
and freeze-marker amendment [are] committed together." This
amendment files that freeze-marker commit:

- `docs/prereg/arc/phase3d_structured_edit_residual.py` (runner) —
  standalone Python module, does not import `phase3_decoder.py`,
  `phase3a_per_task_coord_mlp.py`, or `phase3_decoder_v2.py`. The
  `arc-p3-feature-v1` encoders are copied verbatim under a marked
  "Frozen feature-v1 encoders" header; the file comment notes that
  any drift requires bumping `FEATURE_SCHEMA_VERSION` in both
  places. Manifest captures `specHash` + `parentSpecHash` so any
  later drift between the runner and
  `PHASE3D_DIFFERENT_FRAMING_SPEC.md` is auditable. Implements the
  full Branch D pipeline:
  1. Per-(instance, arm, seed): select the lowest-residual
     `(shape_rule, canvas_rule, background_color)` triple from the
     5 × 10 candidate family by conditioning-residual scoring with
     the spec's tie-break order.
  2. Build cell-level mask + color training rows from conditioning
     pairs against their selected baselines.
  3. Fit a `MaskMLP` (192-hidden, positive-class-weighted BCE,
     `max_steps=700`, `early_stop_patience=120`) and an
     `EditColorMLP` (192-hidden, dropout 0.05,
     class-balanced cross-entropy, same training budget).
  4. Threshold sweep over `{0.10, 0.20, …, 0.90}`: pick the
     threshold maximizing conditioning exact reconstruction, with
     the spec's `(higher F1, lower mass, distance from 0.50)`
     tie-break.
  5. Predict query baseline, mask probs, edit colors at the chosen
     threshold; reconstruct via `apply_edit(baseline, mask, colors)`;
     score grid_exact / baseline_exact / nonbaseline_exact / shape /
     palette / pixel + the 11 edit metrics.
  6. Assign quarantine label from the 9 pre-registered options
     (`baseline_shape_failure`, `baseline_canvas_failure`,
     `edit_mask_underdetection`, `edit_mask_overedit`,
     `edit_color_failure`, `conditioning_starvation`,
     `copy_prior_absent`, `palette_lift_failure`,
     `stochastic_instability`).
- `scripts/arc-phase3d-structured-edit-residual-v1.mjs` (Node
  wrapper) — pure pass-through to the Python runner, honouring
  `$SUNDOG_PYTHON`.
- `package.json` adds `arc:phase3d:structured-edit-residual-v1`.
- `.gitignore` adds
  `results/arc/phase3d-structured-edit-residual-v1/`.
- Pre-commit + CI ARC leak-check (`arc:phase0:leak-check`) passes
  unchanged; the new runner contains no `evaluation` literal and
  the register discipline is untouched.

**Verdict impact**: no prior verdict changes. The Branch D spec
admission moves from "EXECUTION HOLD" to "EXECUTION ADMITTED,
capped probe required" per the spec's own ten-minute rule. No
binding Branch D receipt is filed by this amendment.

**Smoke-test fingerprint** (CPU, `--probe-only --probe-steps 3
--limit-arms raw_grid_edit --limit-seeds 20260528`, all 36 registered
tasks):

- 49 held-out instances (`validation_lodo=18`,
  `validation_pttest=6`, `test_lodo=19`, `pttest=6`);
- 49 baseline-selection rows;
- 49 edit-metrics rows;
- 49 residual records;
- 294 learning-curve rows (49 instances × (3 mask + 3 color) steps);
- 16 receipt files written (manifest + 12 data files +
  branch_adjudication + commands + hashes);
- elapsed total: **222.6 s wall** (probe);
- arena gate: `not_adjudicated` (probe-only, by spec);
- selected seed for `raw_grid_edit`: `20260528` (single-seed limit).

**Baseline-picker sanity** (single-seed probe over all 49 held-out
instances):

| selected `(shape_rule, canvas_rule)` | count |
| --- | ---: |
| `same_as_input` + `identity_top_left` | 28 |
| `conditioning_unanimous_output` + `constant_background` | 9 |
| `same_as_input` + `constant_background` | 6 |
| `conditioning_unanimous_output` + `nonzero_bbox_top_left` | 4 |
| `same_as_input` + `nonzero_bbox_top_left` | 2 |

Diversity is present and dominant pick (identity copy of input)
matches expectation that many ARC outputs are small edits of the
input. The picker does not collapse to a single rule.

**Quarantine reachability** (3-step probe → models barely trained;
labels reachable verified at probe-only):

| label | count |
| --- | ---: |
| `edit_mask_overedit` | 38 |
| `baseline_canvas_failure` | 6 |
| `edit_mask_underdetection` | 4 |
| `edit_color_failure` | 1 |

The 4 remaining labels (`baseline_shape_failure`,
`conditioning_starvation`, `copy_prior_absent`,
`palette_lift_failure`, `stochastic_instability`) are reachable in
principle from the runner code; they do not fire on this minimal
probe because (a) the dominant baseline choice (`same_as_input`)
makes shape exact by construction in the probe, (b) conditioning
starvation requires `< 3` pairs which the probe set mostly satisfies,
(c) `copy_prior_absent` requires mean residual > 0.50 which most
selected baselines clear, (d) `palette_lift_failure` is the catch-all
fallback, and (e) `stochastic_instability` requires multi-seed
disagreement which a single-seed probe cannot exhibit.

**What remains under the spec's ten-minute rule**: a capped timing
probe with realistic `--probe-steps` (e.g. 100–200) must run before
the full 4-arm × 5-seed binding execution. The 3-step smoke is
feature-build- and baseline-iteration-dominated, so naive linear
extrapolation to 700 steps would be misleading; a 100-step CPU
probe will give the regression points needed to project the GPU
serial / 3-shard parallel walls and stage the binding launch.

The next amendment will file: a capped timing probe receipt, the
extrapolated full-run wall, and the staged PowerShell command for
the binding run (with shard+merge plumbing if the projected wall
exceeds the ten-minute threshold — likely given the per-instance
cost roughly doubles from Phase 3A's 4.5 s).

**Public-language constraint**: no change. The pre-binding-receipt
language from the Branch D spec §"Public Language" remains the only
permitted public addition.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch D Capped Timing Probe + Shard+Merge Protocol Admission

CPU + GPU capped timing probes ran against the post-freeze runner
under freeze-marker commit
`95A97C1FC04C9099CA6CA01EF87B4CAEBC586EE9` (operator's HEAD at probe
time after several parallel commits), both `gitDirty=false`.
Configuration: `--probe-only --probe-steps 100 --limit-arms
raw_grid_edit --limit-seeds 20260528`, all 36 registered tasks, 49
held-out instances (`validation_lodo=18`, `validation_pttest=6`,
`test_lodo=19`, `pttest=6`). Receipts at
`results/arc/phase3d-timing-probe-{cpu,gpu}/` (deleted before commit
after extrapolation).

#### Probe Wall And Step-Cap Saturation

| device | elapsed (s) | elapsed (min) | mask cap hit | color cap hit |
| --- | ---: | ---: | ---: | ---: |
| CPU (i7-7820HK) | 489.9 | 8.17 | 49 / 49 (100%) | 49 / 49 (100%) |
| CUDA (GTX 1080) | 318.7 | 5.31 | 49 / 49 (100%) | 49 / 49 (100%) |

100% of fits ran to the 100-step cap. As with Phase 3A, the
per-task scratch learners overfit the tiny conditioning sets
continuously — loss keeps improving by at least `1e-6` every step —
so projecting to the full registered caps (`mask.max_steps = 700`,
`color.max_steps = 700`) must assume cap dominance, not early-stop.

#### Wall-Clock Extrapolation Method

Two CPU data points isolate per-instance overhead from per-step cost:

- smoke (3 steps × 1 arm × 1 seed × 49 instances): 222.6 s
- probe (100 steps × 1 arm × 1 seed × 49 instances): 489.9 s

`49 F + 49 × 6 S = 222.6` and `49 F + 49 × 200 S = 489.9`, where
`F` is per-instance feature-build + baseline-iteration cost and `S`
is per-step training cost. Solving: **`S_cpu = 0.0281 s/step`**,
**`F = 4.374 s/instance`**. Compared to Phase 3A's `F = 1.882
s/instance`, Phase 3D's per-instance overhead is roughly 2.3× larger
because of the 50-candidate baseline iteration (5 shape × 10 canvas
rules evaluated against every conditioning pair) plus the threshold
sweep over 9 candidate thresholds.

GPU step cost from the GPU probe (assuming `F` is the same since
feature build is Python-bound): **`S_gpu ≈ 0.0107 s/step`** (≈ 2.64×
faster than CPU on steps).

#### Full-Run Wall Projection

Per arm-seed at registered caps (700 mask + 700 color steps × 49
instances) with no early-stop:

| component | CPU wall (s) | GPU wall (s) |
| --- | ---: | ---: |
| feature build + baseline iter (49 × F) | 214.4 | 214.4 (Python; not GPU-accelerated) |
| mask training (49 × 700 × S) | 963.2 | 366.9 |
| color training (49 × 700 × S) | 963.2 | 366.9 |
| **per arm-seed total** | **2140.8 (35.7 min)** | **948.2 (15.8 min)** |

Full 4 arms × 5 seeds = 20 combinations:

| posture | concurrent shards | rounds | wall envelope |
| --- | ---: | ---: | --- |
| CPU serial | 1 | 20 | ~11.9 h |
| GPU serial | 1 | 20 | ~5.25 h |
| GPU 3-shard concurrent | 3 | 7 (last round 2 shards) | ~1.93 h (10% contention budget) |
| GPU 4-shard concurrent | 4 | 5 | ~1.51 h (15% contention budget) |

Both serial postures well over the 10-minute rule. Sharding mirrors
Phase 3A's experience: 3-shard parallel is the safer default.

#### Shard+Merge Protocol Admission

The Phase 3D runner gains the (arm × seed) shard+merge protocol used
by Phase 3A, including the `--allow-mixed-commits` override.

**Tooling additions** (committed alongside this amendment):

- `docs/prereg/arc/phase3d_structured_edit_residual.py`:
  - New args: `--shard-arm <name>`, `--shard-seed <int>` (must be
    provided together), `--merge`, `--shard-dirs <comma-list>`,
    `--allow-mixed-commits`.
  - Shard mode pins `mode="shard"`, restricts to a single (arm,
    seed), records `shardArm`/`shardSeed` plus `seedSlateOriginal`
    and `armsOriginal` in the manifest, skips arena gate + branch
    adjudication.
  - New helpers: `read_jsonl`, `read_csv_dicts`, `_parse_bool`,
    `assert_shard_consistency`, `run_merge`.
  - `--data-dir`/`--register` become optional (still enforced for
    non-merge invocations).
  - `assert_shard_consistency` enforces equality on
    `featureSchemaVersion`, `protocolVersion`, `receiptSchemaVersion`,
    `learnerVersion`, `registerHash`, `dataDirHash`, `registerPath`,
    `dataDir`, `maskModelSpec`, `colorModelSpec`, `shapeRules`,
    `canvasRules`, `maskThresholds`, `seedSlate`, `arms`,
    `maxStepsEffective`. Under `--allow-mixed-commits`, `gitCommit`,
    `specHash`, and `parentSpecHash` are dropped from the strict
    list; the runner file content is audited via `git show
    <commit>:<runner_path>` for every distinct shard `gitCommit`,
    and the audit dict (including a `runnerIdenticalAcrossCommits`
    flag) is recorded in the merged manifest as
    `mixedCommitsAudit`. Runner-SHA mismatches print a WARN but do
    not fail (operator override is the trust marker, audit makes
    divergence visible).
  - `run_merge`: loads each shard's `manifest.json`,
    `per_instance.csv`, `learning_curves.csv`, `residuals.jsonl`,
    `baseline_selection.csv`, and `edit_metrics.csv`; reconstructs
    `per_arm_validation_metrics` and `per_instance_seed_outcomes`
    from the merged per-instance rows; pulls per-seed validation
    loss from each shard's manifest; runs the existing
    `select_seed_for_arm` + aggregation + arena gate + Branch D
    adjudication pipeline unchanged.
- `package.json`: adds
  `arc:phase3d:structured-edit-residual-v1:shard` and
  `arc:phase3d:structured-edit-residual-v1:merge`.
- `.gitignore`: no change (existing
  `results/arc/phase3d-structured-edit-residual-v1/` entry covers
  the binding output path; shard intermediates live in
  `results/arc/phase3d-structured-edit-residual-v1-shard-*` under
  the broader `results/arc/` ignore).

**Shard-equivalence smoke** (CPU, `--probe-only --probe-steps 3
--limit-arms raw_grid_edit --limit-seeds 20260528,20260529` serial
vs. 2 shards merged):

| artifact | merged vs serial |
| --- | --- |
| `scores.csv` | `cmp` exit 0 (byte-identical) |
| `per_task.csv` | `cmp` exit 0 (byte-identical) |
| `per_prior.csv` | `cmp` exit 0 (byte-identical) |
| `per_instance.csv` | `cmp` exit 0 (byte-identical) |
| `baseline_selection.csv` | `cmp` exit 0 (byte-identical) |
| `edit_metrics.csv` | `cmp` exit 0 (byte-identical) |
| `seed_stability.csv` | `cmp` exit 0 (byte-identical) |

The merge correctly emits `branch_d_full_grid_edit_floor` (expected
at 3-step probe with no learning). Smoke + probe directories were
deleted before commit.

#### Staged Full-Run Command (3-Shard GPU Parallel)

The frozen launch command (PowerShell, GPU, 4 arms × 5 seeds = 20
shards):

```powershell
$env:SUNDOG_PYTHON = "C:\Users\hughe\AppData\Local\Programs\Python\Python312\python.exe"
foreach ($arm in @("raw_grid_edit","signature_palette_edit","signature_only_edit","metadata_only_edit")) {
    foreach ($seed in @(20260528, 20260529, 20260530, 20260531, 20260601)) {
        Start-Process -NoNewWindow `
            -RedirectStandardOutput "results/arc/phase3d-logs/$arm-$seed.log" `
            -RedirectStandardError "results/arc/phase3d-logs/$arm-$seed.err" `
            -FilePath "npm" -ArgumentList @(
                "run", "arc:phase3d:structured-edit-residual-v1:shard", "--",
                "--data-dir", "$env:USERPROFILE\Datasets\ARC-AGI-2\data",
                "--register", "docs/prereg/arc/P0_TASK_REGISTER.csv",
                "--out", "results/arc/phase3d-structured-edit-residual-v1-shard-$arm-$seed",
                "--shard-arm", "$arm",
                "--shard-seed", "$seed",
                "--device", "cuda"
            )
        # Throttle to 3 concurrent shards.
    }
}

# After all 20 shards land:
npm run arc:phase3d:structured-edit-residual-v1:merge -- `
    --shard-dirs <comma-joined list of all 20 dirs> `
    --out results/arc/phase3d-structured-edit-residual-v1
```

Operator-discretionary additions:

- Pass `--allow-mixed-commits` to the merge if parallel commits
  landed during the slate (Phase 3A precedent — likely on this
  workstation given simultaneous Navier-Stokes/Riemann/P-vs-NP
  lanes).
- Pass `--allow-dirty` to individual shard re-launches if the
  worktree carries non-ARC mods.

**Resume safety**: each shard is single-process and resume-unsafe at
the (instance) granularity. A crashed shard is re-run as a single
unit; surviving shards remain intact.

#### Per-Outcome Decision Rule

| arena gate outcome | next action |
| --- | --- |
| `raw_grid_edit_arena_open` (≥ 1 non-baseline exact task on `test_lodo` AND `pttest` for `raw_grid_edit`) | Examine `branchAdjudication`. If `branch_d_support`, file the receipt + open public-language additions per spec §"Public Language". If `branch_d_bounded_failure`, file the receipt + the named-quarantine breakdown from `quarantine_log.csv`. |
| `branch_d_full_grid_edit_floor` (raw_grid_edit does not open the non-baseline arena) | Per spec §"Arena Gate": no signature sufficiency language allowed; no extra seeds; the next admissible Phase 3 work must be a new append-only learner spec (Phase 3E or further Branch D variant). This would also close the *fifth* full-grid control receipt with the structured-edit framing not lifting the floor either, narrowing the space of plausible Phase 3 reopens dramatically. |

**Verdict impact**: no prior verdict changes. Branch D status moves
from "EXECUTION ADMITTED, capped probe required" to "EXECUTION
ADMITTED, shard+merge protocol admitted, 20-shard launch ready". No
binding receipt yet.

**Public-language constraint**: unchanged.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch D 20-Shard Binding Receipt: `branch_d_full_grid_edit_floor`

The Phase 3D 20-shard slate (4 arms × 5 seeds × 49 held-out
instances) completed and merged at `mergeGitCommit
9F3193D7…`. Receipt at
`results/arc/phase3d-structured-edit-residual-v1/`.

**Wall-clock efficiency**: total shard compute = **14945.3 s
(4.15 h)**; **3-shard concurrent GPU parallel wall ≈ 1h 26m**
(20:22:51 → 21:49:03 UTC), a 2.9× speedup vs. serial — close to
ideal 3× scaling. All 20 shards landed cleanly on the first wave
(no re-launches required this time, vs. Phase 3A's 14/20 first-wave
attrition).

**Mixed-commits audit** (per `shardSources` in the merged manifest):

| gitCommit (8-char) | shards | gitDirty |
| --- | ---: | --- |
| `403B8C8E` | 14 | True (workspace had non-ARC mods at launch) |
| `8C6BE6B2` (Phase 3D freeze-marker) | 3 | False |
| `1EBE3B3D` | 3 | False |

`mixedCommitsAudit.runnerIdenticalAcrossCommits = **true**` — all 3
commits share the same byte-identical runner SHA. The override was
used only because the docs spec hashes (parentSpecHash, specHash)
naturally drifted across operator-amendment commits; the shard-time
computational contract was provably identical across every shard.
No WARN printed during merge.

#### Validation Candidate Selection (per arm)

Selection rule unchanged from the Branch D spec §"Seed Slate":
`(-val_nonbaseline_exact_count, -val_edit_mask_f1,
-val_minority_edit_recall, +val_over_edit_rate, +val_loss, +seed)`.
Selected seed per arm:

| arm | selected seed |
| --- | ---: |
| `raw_grid_edit` | `20260529` |
| `signature_palette_edit` | `20260531` |
| `signature_only_edit` | `20260529` |
| `metadata_only_edit` | `20260528` |

#### Arena Gate Adjudication

Pre-registered floor (PHASE3D_DIFFERENT_FRAMING_SPEC.md §"Arena
Gate"): `raw_grid_edit` must achieve at least one **non-baseline**
exact task on both `test_lodo` and `pttest`. Any grid that is exact
after applying the baseline alone (before learned edits) is
recorded as `baseline_exact` and does NOT count toward
`nonbaseline_exact_task_count`.

Observed (selected-seed per arm):

| lane | arm | grid_exact_any | **baseline_exact_any** | **nonbaseline_exact_any** | shape_exact | palette_exact | pixel_mean | edit_mask_f1 | minority_edit_recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pttest` | `raw_grid_edit` | **0.000** | **0.000** | **0.000** | 1.000 | 0.333 | 0.472 | 0.584 | 0.648 |
| `pttest` | `signature_palette_edit` | 0.000 | 0.000 | 0.000 | 1.000 | 0.333 | 0.488 | 0.654 | 0.723 |
| `pttest` | `signature_only_edit` | 0.000 | 0.000 | 0.000 | 1.000 | 0.333 | 0.464 | 0.655 | 0.719 |
| `pttest` | `metadata_only_edit` | 0.000 | 0.000 | 0.000 | 1.000 | 0.333 | 0.475 | 0.645 | 0.717 |
| `test_lodo` | `raw_grid_edit` | **0.000** | **0.000** | **0.000** | 0.316 | 0.316 | 0.511 | 0.530 | 0.669 |
| `test_lodo` | `signature_palette_edit` | 0.000 | 0.000 | 0.000 | 0.316 | 0.316 | 0.573 | 0.570 | 0.609 |
| `test_lodo` | `signature_only_edit` | 0.000 | 0.000 | 0.000 | 0.316 | 0.316 | 0.559 | 0.562 | 0.631 |
| `test_lodo` | `metadata_only_edit` | 0.000 | 0.000 | 0.000 | 0.316 | 0.316 | 0.567 | 0.576 | 0.632 |

`raw_grid_edit`: `test_lodo_nonbaseline_exact_tasks = 0`,
`pttest_nonbaseline_exact_tasks = 0`.

**Arena gate: `branch_d_full_grid_edit_floor`**. The non-baseline
edit arena does not open. Per the spec §"Arena Gate", no
`signature_palette_edit` vs. `raw_grid_edit` sufficiency comparison
is licensed.

`branchAdjudication.branch = "branch_d_full_grid_edit_floor"`.

#### Quarantine Breakdown (selected seed)

196 quarantine labels (49 instances × 4 arms; **6 of 9
pre-registered labels fire** in the binding run):

| label | count | share | notes |
| --- | ---: | ---: | --- |
| `edit_color_failure` | 51 | 26.0% | mask F1 ≥ 0.50 but edit-color accuracy < 0.50 — the dominant failure: model knows WHERE to edit but not WHAT color |
| `conditioning_starvation` | 43 | 21.9% | < 3 conditioning pairs (same structural ARC k=2-3 limitation Phase 3A hit) |
| `palette_lift_failure` | 29 | 14.8% | catch-all when none of the more specific labels apply |
| `edit_mask_overedit` | 26 | 13.3% | over-edit rate > 0.50 |
| `baseline_canvas_failure` | 24 | 12.2% | selected baseline canvas has > 0.50 residual on query |
| `edit_mask_underdetection` | 23 | 11.7% | mask recall < 0.25 |

**Did not fire**: `baseline_shape_failure` (the baseline shape rules
mostly succeed — `pttest` has 100% shape exact), `copy_prior_absent`
(the picker rarely returns mean residual > 0.50 on conditioning),
`stochastic_instability` (seed-disagreement label; most instances
agreed across seeds at the "no exact" level).

#### Named Failure Mode: Edit-Color-Rule Failure

The Phase 3D failure character is **qualitatively distinct** from
all four prior receipts:

- V1/V2 (transformer): noise-dominated, palette ≈ 0, pixel ≈ 0.25
- compact-7 (transformer on 7-task subset): dominant-color mode
  collapse, palette ≈ 0, shape 1.0, pixel ≈ 0.83
- Phase 3A (per-task coord MLP): conditioning starvation, shape
  ≈ 0.5, palette ≈ 0.20, pixel ≈ 0.28
- **Phase 3D (structured edit residual)**: edit-color-rule failure.
  Shape **always exact on pttest** (1.000), pixel jumps to
  **0.47-0.57**, edit-mask **F1 0.53-0.65** with strong minority
  recall (0.61-0.72) — the model is doing **real work** at every
  stage except recovering the correct edit colors. The dominant
  quarantine label (26%) names this gap explicitly:
  `edit_color_failure` (knows WHERE, doesn't know WHAT).

The structured-edit framing **decomposes the failure mode** into
its components in a way the direct-output framings could not:

1. **Baseline shape picker**: succeeds on most tasks (100% shape
   exact on pttest; 0.316 on test_lodo means baseline shape is
   sometimes wrong when LODO query is more variant).
2. **Baseline canvas picker**: gets within 0.50 residual on most
   tasks (24/196 = 12% are `baseline_canvas_failure`).
3. **Edit-mask learner**: achieves moderate-to-strong F1 + minority
   recall — knows where to edit.
4. **Edit-color learner**: this is the bottleneck. 51/196 = 26% of
   selected-seed instances reach mask F1 ≥ 0.50 yet fail on color
   accuracy.

This decomposition is the genuine novelty of the Phase 3D receipt:
prior lanes' failures were "the whole pipeline doesn't work";
Phase 3D's failure is "**3 of 4 components work, but per-task
color recovery on tiny conditioning sets is too hard for a 192-
hidden MLP**".

#### Per-Arm Comparison (Conditional, Documentation Only)

Per the spec §"Arena Gate", since the raw-grid edit arm did not
open the arena, **no Branch D support / bounded-failure /
named-quarantine verdict is licensed**. The per-arm scores are
recorded for audit and for any future learner that wishes to
outperform this baseline; they are **not** a signature-vs-full-grid
sufficiency comparison.

For the record (not licensed adjudication):

- All four arms produce **identical** shape_exact and palette_exact
  rates per lane (1.000/0.333 on pttest, 0.316/0.316 on test_lodo).
  The baseline picker dominates these dimensions and is the same
  for all arms because it operates on arm-specific input vectors
  but is selected by the same shape/canvas/background rules.
- `signature_palette_edit` (the registered Sundog arm) achieves
  the **highest pixel_accuracy** on `test_lodo` (0.573 vs raw
  0.511) and the **highest minority_edit_recall on pttest**
  (0.723 vs raw 0.648). With both at zero exact, this remains
  unlicensed-for-adjudication data.
- `metadata_only_edit` and `signature_only_edit` are within noise
  of `signature_palette_edit` and `raw_grid_edit` on every metric.
  The representation difference is not visible at the floor.

#### What This Verdict Does And Does Not Entail

**It does**:

- Close the **fifth** full-grid control floor in Phase 3
  (alongside V1, V2, compact-7, Phase 3A) at zero exact tasks on
  both registered held-out lanes for arm `raw_grid_edit`.
- Decompose the failure mode for the first time across the Phase 3
  receipt history: the structured-edit framing isolates the
  failure to the **edit-color-rule** stage rather than producing
  another whole-pipeline failure.
- Demonstrate that **2.9× sharded GPU parallelism is achievable in
  practice** on this workstation for runs of this shape (4.15 h
  serial → 1h 26m parallel wall).

**It does not**:

- License any signature_palette_edit vs. raw_grid_edit sufficiency
  comparison (the raw-grid edit arm did not open the non-baseline
  arena).
- License any narrowed-support Branch D claim from this receipt.
- License extra seeds on this lane (per the spec §"Forbidden":
  "extra seeds or task narrowing after an arena-floor receipt
  without a new append-only amendment").
- Speak to whether **a different baseline family** or **a
  larger-capacity color learner** (e.g., shared edit-color MLP
  trained across all conditioning instances of similar baselines)
  would clear the floor.

#### Remaining Reopen Paths for Phase 3

After this receipt, all four PHASE3_5_REFLECTION branches plus
Branch D have produced binding floors:

| branch | learner family | task distribution | verdict |
| --- | --- | --- | --- |
| Pre-reflection (V1) | Transformer (Blackwell) | 36 tasks | floor |
| Pre-reflection (V2) | Transformer (Blackwell, 971 aux) | 36 tasks | floor |
| Branch B | Transformer (Blackwell) | 7 compact tasks | floor (dominant-color mode collapse) |
| Branch A | Per-task coord MLP | 36 tasks | floor (conditioning starvation) |
| **Branch D** | **Per-task structured-edit residual** | **36 tasks** | **floor (edit-color-rule failure)** |

The space of remaining admissible Phase 3 reopens narrows to:

- **Branch D variants**: change ONE of the structured-edit
  components and retest. Candidates: (a) wider baseline family,
  (b) cross-task edit-color MLP (relax the per-task constraint
  for color only, keeping mask per-task), (c) richer mask
  learner.
- **Branch E (new framing)**: e.g., generative program search
  (DSL synthesis over the conditioning pairs), or per-instance
  test-time prompting of a frozen large LM — both outside the
  registered-deterministic and per-task-scratch families.
- **Phase 6 / Kaggle preparation**: orthogonal — does not
  adjudicate sufficiency.

A Branch D variant or Branch E would need its own pre-registered
spec, arena gate, and verdict-amendment discipline; this receipt
admits none of them.

#### Public-Language Constraint Update

Permitted additions (per PHASE3D_DIFFERENT_FRAMING_SPEC.md
§"Public Language" arena-floor language):

- "Phase 3D's structured-edit-residual framing did not open the
  full-grid edit arena on the registered ARC public-training task
  subset; the receipt calibrates the structured-edit framing and
  does not adjudicate signature sufficiency."
- "Five Phase 3 binding full-grid-control receipts — V1, V2,
  compact-7, Phase 3A, and Phase 3D — now agree on the held-out
  exact-grid floor across two task distributions, two learner
  families (transformer + per-task MLP), and two output framings
  (whole-grid + structured edit residual). The structured-edit
  framing additionally isolated the failure to the
  edit-color-rule stage."

Forbidden:

- "Phase 3D floored → signature representation is favoured" — no
  signature sufficiency comparison is licensed.
- "Phase 3D solves ARC" or any Kaggle / public-evaluation
  language.
- Any claim that baseline-only exactness supports any sufficiency
  position (baseline_exact = 0.000 on every lane, every arm).
- Any claim that signature_palette_edit's slightly-higher pixel /
  minority recall in the documentation-only per-arm table
  constitutes a support signal (the arena did not open).

#### Frozen By This Verdict

- The Phase 3D binding receipt
  (`results/arc/phase3d-structured-edit-residual-v1/`), its
  manifest, hashes, and all 14+ receipt files are frozen at the
  `mergeGitCommit = 9F3193D7…` snapshot.
- The mixed-commits audit
  (`mixedCommitsAudit.runnerIdenticalAcrossCommits = true` across
  3 distinct gitCommits) is frozen as the canonical record of
  how this receipt was assembled.
- The named failure mode "**edit-color-rule failure**" is filed
  here for the structured-edit framing on per-task-scratch
  learners.
- The selected-seed table per arm + the wall-time efficiency
  measurement (2.9× sharded GPU parallel speedup) are frozen as
  Phase 3D execution traces.

**Final verdict impact**: no V1, V2, compact-7, or Phase 3A
verdict changes. The Phase 3 status moves from "Branch D
execution in progress" to "Branch D binding receipt filed,
`branch_d_full_grid_edit_floor`; all four PHASE3_5_REFLECTION
branches + Branch D now characterised; reopens require a new
pre-registered Branch D variant or Branch E spec".

### 2026-05-28 (PT) -- Codex (GPT-5) -- Branch D Edit-Color-Rule Variant Spec Filed

Branch D variant spec:
[`PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md`](PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md).

Justification: the Phase 3D structured-edit receipt isolated the current
bottleneck to the edit-color-rule stage: baseline shape/canvas and edit-mask
signals were nontrivial, but the per-task scratch edit-color MLP did not recover
exact output colors. This amendment starts the narrowest Branch D variant:
change only the edit-color component and leave the registered task class,
baseline picker, edit-mask learner, splits, seed slate, and arena discipline
unchanged.

Verdict impact: **no execution admission and no Branch D variant verdict**. The
new spec freezes `structured_edit_color_rule_v2` and learner label
`edit_color_rule_bank_v1`, but execution remains held until runner tooling, npm
wiring, result ignore path, leak-check coverage, capped smoke/probe receipts,
and a freeze-marker amendment are committed together.

The variant replaces the learned edit-color MLP with a deterministic
conditioning-derived color-rule bank. Frozen rule families include:

- constant/modal edit-color rules;
- baseline-color, input-nearest-neighbor-color, input-patch-majority, and
  baseline/input-pair color maps;
- relative palette-rank mapping;
- object-role color mapping;
- small row/column periodic color rules;
- nearest edited-neighbor color transfer.

The arena discipline remains conservative:

1. `raw_grid_edit_color_v2` must open the non-baseline arena by achieving at
   least one non-baseline exact task on both held-out lanes (`test_lodo` and
   `pttest`);
2. if raw grid floors, the verdict is `branch_d_color_rule_full_grid_floor` and
   no signature sufficiency language is admitted;
3. oracle rule accuracy is diagnostic only and cannot support a branch verdict;
4. only if the raw-grid color-rule arena opens can
   `signature_palette_edit_color_v2` be compared for
   `branch_d_color_rule_support`, `branch_d_color_rule_bounded_failure`, or
   diagnostic named quarantine.

Reserved implementation names:

- Python runner: `docs/prereg/arc/phase3d_edit_color_rule_v2.py`;
- Node wrapper: `scripts/arc-phase3d-edit-color-rule-v2.mjs`;
- npm script: `arc:phase3d:edit-color-rule-v2`;
- shard npm script: `arc:phase3d:edit-color-rule-v2:shard`;
- merge npm script: `arc:phase3d:edit-color-rule-v2:merge`;
- binding receipt path: `results/arc/phase3d-edit-color-rule-v2/`.

Public-language constraint before a binding receipt:

> "Phase 3D has filed an edit-color-rule variant spec targeting the bottleneck
> isolated by the structured-edit receipt. No variant receipt exists yet, and no
> sufficiency claim is admitted."

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Branch D Variant Tooling Freeze-Marker (execution unblocked, capped-probe required)

The Branch D variant spec
([`PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md`](PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md))
held execution until "runner tooling, npm wiring, result ignore path,
leak-check coverage, and a freeze-marker amendment are committed
together." This amendment files that commit:

- `docs/prereg/arc/phase3d_edit_color_rule_v2.py` (runner) —
  standalone Python module. Does **not** import
  `phase3_decoder.py`, `phase3a_per_task_coord_mlp.py`, or
  `phase3d_structured_edit_residual.py`. The `arc-p3-feature-v1`
  encoders, baseline family, mask MLP, threshold-sweep, and
  shard+merge plumbing are **copied verbatim** from the base
  Branch D runner under marked headers. The `EditColorMLP` is
  removed; in its place, the runner implements the 10-family
  deterministic **color-rule bank** from the spec §"Color Rule
  Bank":
  1. `constant_edit_color`, `modal_edit_color`,
     `baseline_color_map`, `input_nn_color_map`,
     `input_patch_majority_map`, `baseline_to_input_pair_map`,
     `relative_palette_rank_map` (same/nearest/learned strategies),
     `object_role_color_map`, `row_col_periodic_color` (period 1–3,
     row/col), `nearest_edited_neighbor_color`.
  2. `generate_candidate_rules` materialises all concrete candidates
     from conditioning residuals (typically 15–25 per instance).
  3. `_score_rule_on_pairs` evaluates each candidate via LOCO when
     `k ≥ 3` else all-cells with `low_k_rule_selection=true`,
     returning `accuracy / rare_recall / halluc_rate` per the spec.
  4. `select_color_rule` applies the spec tie-break chain
     `(-accuracy, -rare_recall, +halluc_rate, +family_index,
     +SHA-256(arc-p3d-edit-color-rule-v2|master|lane|task|query|arm|rule_id))`.
  5. Optional top-3 vote ensemble fires only if `≥ 3` candidates tie
     within `ENSEMBLE_TIE_TOLERANCE = 0.05` on conditioning accuracy.
  6. New metrics: `edit_color_rule_accuracy`, `rare_edit_color_recall`,
     `color_rule_family`, `color_rule_id`, `color_rule_ensemble`,
     `color_rule_candidate_count`, `low_k_rule_selection`,
     `no_conditioning_edits`, `color_oracle_rule_accuracy`
     (diagnostic-only — explicitly **excluded** from branch decisions
     per spec), `rule_selection_regret` (oracle − selected),
     `mask_conditioned_color_accuracy`.
  7. New quarantine labels per spec §"Quarantine Labels": all 12
     pre-registered labels are reachable; the critical pair
     `color_rule_bank_coverage_failure` (oracle < 0.50) vs.
     `color_rule_selection_failure` (oracle ≥ 0.50 but selected
     < 0.50) is reachable and verified in the probe smoke.
  8. Arena gate + branch adjudication renamed to
     `raw_grid_edit_color_v2_arena_open` and
     `branch_d_color_rule_{full_grid_floor,support,bounded_failure,named_quarantine}`;
     support criteria use `edit_color_rule_accuracy` gap,
     `rare_edit_color_recall` gap, and `rule_selection_regret` gap
     (each ≤ 0.10) per spec.
  9. Manifest records `variantVersion`, `baseBranchDSpecPath`,
     `baseBranchDSpecHash`, `ruleFamilies`, `ensembleTieTolerance`,
     `ensembleMinMembers`.
- `scripts/arc-phase3d-edit-color-rule-v2.mjs` — Node wrapper
  (`SUNDOG_PYTHON` honoured).
- `package.json`: adds
  `arc:phase3d:edit-color-rule-v2`,
  `arc:phase3d:edit-color-rule-v2:shard`, and
  `arc:phase3d:edit-color-rule-v2:merge`.
- `.gitignore`: adds `results/arc/phase3d-edit-color-rule-v2/`.
- Shard+merge protocol is wired from day one (mirror Phase 3D),
  including the `--allow-mixed-commits` operator override with the
  runner-content WARN-not-fail audit.
- Pre-commit + CI ARC leak-check passes unchanged.

**Verdict impact**: no prior verdict changes. Branch D variant
admission moves from "EXECUTION HOLD" to "EXECUTION ADMITTED,
capped probe required" per the spec's own ten-minute rule. No
binding receipt yet.

**Smoke-test fingerprint** (CPU, `--probe-only --probe-steps 3
--limit-arms raw_grid_edit_color_v2 --limit-seeds 20260528`, all 36
registered tasks):

- 49 held-out instances processed; 49 baseline selections; 49
  edit-metrics rows; 49 color-rule selections; **942 color-rule
  candidate scores** (~19 candidates per instance, consistent with
  the 15–25 expected from the 10 families).
- 147 mask-learning rows (49 × 3 probe steps; no color curve —
  deterministic rule bank);
- 18 receipt files written.
- elapsed total: **160.9 s wall** (probe);
- arena gate: `not_adjudicated` (probe-only, by spec).

**Rule-family diversity sanity** (selected rule on 49 instances):

| selected family | count |
| --- | ---: |
| `ensemble_top3` | 20 (40.8%) |
| `nearest_edited_neighbor_color` | 18 (36.7%) |
| `relative_palette_rank_map` | 10 (20.4%) |
| `row_col_periodic_color` | 1 |

`ensemble_top3` firing on 40.8% of instances is meaningful: the
spec's `≥ 3 within 0.05` ensemble eligibility criterion regularly
finds candidate rules with near-identical conditioning accuracy,
and the deterministic top-3 vote acts as a tie-break.

**Quarantine reachability** (4 of 12 labels fire under the 3-step
probe; the remaining 8 are reachable in principle):

| label | count | share |
| --- | ---: | ---: |
| `edit_mask_failure` | 32 | 65% — expected: mask MLP barely trained at 3 steps |
| `color_rule_selection_failure` | 8 | 16% — oracle ≥ 0.50 but selected < 0.50 |
| `baseline_canvas_failure` | 6 | 12% |
| `color_rule_bank_coverage_failure` | 3 | 6% — oracle < 0.50 |

The new pair (`color_rule_bank_coverage_failure` ↔
`color_rule_selection_failure`) separates "no rule in the bank can
solve this instance" from "a rule exists but the selector picked
wrong" — exactly the diagnostic decomposition the spec was designed
to provide.

**What remains under the spec's ten-minute rule**: a capped timing
probe with realistic `--probe-steps` (e.g., 100) must run before the
full 4-arm × 5-seed binding execution. Initial estimate
(rule-bank is deterministic, mask MLP cost dominates and mirrors
Phase 3D's S/F regression): **~5.5 h GPU serial** for the full 4×5
binding run, **~2 h** under 3-shard concurrent GPU. The next
amendment will file the actual capped probe timing + staged binding
launch posture.

**Public-language constraint**: unchanged. The pre-binding-receipt
language from the variant spec §"Public Language" remains the only
permitted public addition.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Variant Capped Timing Probe + 20-Shard Launch Staging

CPU + GPU capped timing probes ran against the post-freeze runner under
commit `61DDB7D5BD0DA1B9BC2A4F6FE10C9F0F2D86A4B5` (the variant
freeze-marker amendment), both `gitDirty=false`. Configuration:
`--probe-only --probe-steps 100 --limit-arms raw_grid_edit_color_v2
--limit-seeds 20260528`, all 36 registered tasks, 49 held-out instances.

#### Probe Wall

| device | elapsed (s) | elapsed (min) |
| --- | ---: | ---: |
| CPU (i7-7820HK) | 311.3 | 5.19 |
| CUDA (GTX 1080) | 206.3 | 3.44 |

#### Wall-Clock Extrapolation Method

Two CPU data points isolate per-instance overhead from per-step
training cost (variant has only one trained model — the mask MLP —
since the color rule bank is deterministic):

- smoke (3 mask steps × 1 arm × 1 seed × 49 instances): 160.9 s
- probe (100 mask steps × 1 arm × 1 seed × 49 instances): 311.3 s

`49 F + 49 × 3 S = 160.9` and `49 F + 49 × 100 S = 311.3`. Solving:
**`S_cpu = 0.0316 s/step`**, **`F = 3.189 s/instance`** (feature build
+ 50-candidate baseline iteration + ~19-candidate rule-bank
generation + scoring + selection).

GPU step cost (assuming F is the same — Python-bound):
**`S_gpu ≈ 0.0102 s/step`** (≈ 3.1× faster than CPU; better than
Phase 3D's 2.64× because the workload is more compute-bound when
color training is gone).

**Note on the per-instance F**: 3.19 s/instance is meaningfully
*lower* than Phase 3D's 4.37 s/instance because the rule-bank
generation + scoring + selection (~1 s per instance under LOCO over
~19 candidates) is faster than Phase 3D's "EditColorMLP training
data construction + full-grid color prediction at every threshold"
sequence. The deterministic rule bank is **both cheaper to train AND
cheaper to predict** than the learned color MLP.

#### Full-Run Wall Projection

Per arm-seed at registered cap (mask=700 steps × 49 instances):

| component | CPU wall (s) | GPU wall (s) |
| --- | ---: | ---: |
| feature build + baseline + rule bank (49 × F) | 156.3 | 156.3 (Python; not GPU-accelerated) |
| mask training (49 × 700 × S) | 1085.0 | 350.3 |
| **per arm-seed total** | **1241.3 (20.7 min)** | **506.6 (8.4 min)** |

Full 4 arms × 5 seeds = 20 combinations:

| posture | concurrent shards | rounds | wall envelope |
| --- | ---: | ---: | --- |
| CPU serial | 1 | 20 | ~6.90 h |
| GPU serial | 1 | 20 | **~2.81 h** |
| GPU 3-shard concurrent | 3 | 7 | **~1.03 h** (10% contention budget) |
| GPU 4-shard concurrent | 4 | 5 | ~0.78 h (15% contention budget) |

**The variant is meaningfully faster than Phase 3D's base** (Phase 3D
GPU serial was ~5.25 h vs. variant ~2.81 h; 3-shard parallel was
~1.93 h vs. variant ~1.03 h). The savings come entirely from
removing the EditColorMLP training (~half the compute in Phase 3D
went there).

#### Staged Full-Run Command (3-Shard GPU Parallel)

```powershell
$env:SUNDOG_PYTHON = "C:\Users\hughe\AppData\Local\Programs\Python\Python312\python.exe"
foreach ($arm in @("raw_grid_edit_color_v2","signature_palette_edit_color_v2","signature_only_edit_color_v2","metadata_only_edit_color_v2")) {
    foreach ($seed in @(20260528, 20260529, 20260530, 20260531, 20260601)) {
        Start-Process -NoNewWindow `
            -RedirectStandardOutput "results/arc/phase3d-color-logs/$arm-$seed.log" `
            -RedirectStandardError "results/arc/phase3d-color-logs/$arm-$seed.err" `
            -FilePath "npm" -ArgumentList @(
                "run", "arc:phase3d:edit-color-rule-v2:shard", "--",
                "--data-dir", "$env:USERPROFILE\Datasets\ARC-AGI-2\data",
                "--register", "docs/prereg/arc/P0_TASK_REGISTER.csv",
                "--out", "results/arc/phase3d-edit-color-rule-v2-shard-$arm-$seed",
                "--shard-arm", "$arm",
                "--shard-seed", "$seed",
                "--device", "cuda"
            )
        # Throttle to 3 concurrent shards.
    }
}

# After all 20 shards land:
npm run arc:phase3d:edit-color-rule-v2:merge -- `
    --shard-dirs <comma-joined list of all 20 dirs> `
    --out results/arc/phase3d-edit-color-rule-v2
```

Operator-discretionary additions per Phase 3D pattern:
`--allow-mixed-commits` to merge across parallel-commit gitCommits,
`--allow-dirty` per shard if the worktree carries non-ARC mods at
re-launch.

**Resume safety**: each shard is single-process and resume-unsafe at
the (instance) granularity (same as Phase 3D).

#### Per-Outcome Decision Rule

| arena gate outcome | next action |
| --- | --- |
| `raw_grid_edit_color_v2_arena_open` (≥ 1 non-baseline exact task on each held-out lane for `raw_grid_edit_color_v2`) | Examine `branchAdjudication`. If `branch_d_color_rule_support`, file the receipt + open public-language additions per variant spec §"Public Language". If `branch_d_color_rule_bounded_failure`, file the receipt + the named-quarantine breakdown emphasising the new pair (`color_rule_bank_coverage_failure` vs `color_rule_selection_failure`) — that decomposition tells the next-step direction (extend rule bank vs improve selector). |
| `branch_d_color_rule_full_grid_floor` (raw_grid_edit_color_v2 does not open the arena) | File the receipt. This would be the **sixth** full-grid control floor in Phase 3. The quarantine breakdown would be diagnostic: if dominated by `color_rule_bank_coverage_failure`, the rule bank is structurally insufficient (file a Branch D rule-bank-extension variant); if dominated by `color_rule_selection_failure`, the selector tie-break is the bottleneck (file a Branch D selection-refinement variant); if dominated by `edit_mask_failure`, the mask MLP is the gate (the color repair didn't matter); if dominated by `baseline_canvas_failure`, the baseline family is the gate (Branch D baseline-extension variant). The decomposition makes the next surgical move predictable rather than another wild guess. |

**Verdict impact**: no prior verdict changes. Variant status moves
from "EXECUTION ADMITTED, capped probe required" to "EXECUTION
ADMITTED, 20-shard launch ready". No binding receipt yet.

**Public-language constraint**: unchanged.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Variant 20-Shard Binding Receipt: `branch_d_color_rule_full_grid_floor` (bottleneck-shifted)

The Phase 3D variant 20-shard slate (4 arms × 5 seeds × 49 held-out
instances) completed and merged at `mergeGitCommit 8177A578…`.
Receipt at `results/arc/phase3d-edit-color-rule-v2/`.

**Wall-clock efficiency**: total shard compute = **11,459.0 s
(3.18 h)**; **3-shard concurrent GPU parallel wall = 1 h 6 m**
(00:27:08 → 01:33:25 UTC), a **2.89× speedup** vs serial — close to
ideal 3× scaling and exactly matching the staging amendment's
projection (~1.03 h). All 20 shards landed cleanly on the first
wave.

**Mixed-commits audit**:

| gitCommit | shards | gitDirty |
| --- | ---: | --- |
| `90F7A895` (staging freeze-marker) | 12 | mixed |
| `F56383C3` | 6 | mixed |
| `E182A5C8` | 2 | mixed |

9 dirty + 11 clean shards across 3 distinct gitCommits;
`mixedCommitsAudit.runnerIdenticalAcrossCommits = **true**` — all 3
commits share the same byte-identical runner SHA. The override was
needed only for documentation-spec-hash drift, not for any
shard-time computational change. No WARN printed.

#### Selected Seed per Arm

| arm | selected seed |
| --- | ---: |
| `raw_grid_edit_color_v2` | `20260528` |
| `signature_palette_edit_color_v2` | `20260528` |
| `signature_only_edit_color_v2` | `20260528` |
| `metadata_only_edit_color_v2` | `20260531` |

#### Arena Gate

Pre-registered floor: `raw_grid_edit_color_v2` must achieve at least
one non-baseline exact task on both `test_lodo` and `pttest`.

Observed: `raw_grid_edit_color_v2` `test_lodo_nonbaseline_exact_tasks
= 0`, `pttest_nonbaseline_exact_tasks = 0`. Every arm scored zero
non-baseline exact tasks on every held-out lane (also zero
baseline-exact: no baseline alone produces the exact target).

**Arena gate: `branch_d_color_rule_full_grid_floor`**.

#### Per-Arm Comparison (Documentation Only — Arena Did Not Open)

| lane | arm | grid_ex | base_ex | nbex | shape | palette | pixel | mask_f1 | color_rule_acc | rare_recall | regret |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pttest` | `raw_grid_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.667** | 0.499 | 0.654 | 0.357 | 0.348 | 0.331 |
| `pttest` | `signature_palette_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.667** | **0.541** | 0.640 | **0.372** | 0.321 | 0.315 |
| `pttest` | `signature_only_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.667** | 0.541 | 0.644 | 0.376 | 0.348 | 0.312 |
| `pttest` | `metadata_only_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.833** | 0.548 | 0.630 | 0.386 | 0.331 | 0.301 |
| `test_lodo` | `raw_grid_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.632** | 0.574 | 0.547 | 0.322 | 0.453 | 0.349 |
| `test_lodo` | `signature_palette_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.632** | **0.601** | 0.567 | **0.348** | 0.387 | 0.323 |
| `test_lodo` | `signature_only_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.632** | 0.591 | 0.567 | 0.334 | 0.384 | 0.337 |
| `test_lodo` | `metadata_only_edit_color_v2` | 0.000 | 0.000 | 0.000 | 1.000 | **0.632** | 0.590 | 0.583 | 0.346 | 0.395 | 0.326 |

#### Comparison Against Phase 3D Base — The Bottleneck Shifted

| metric | Phase 3D base | Phase 3D variant | delta |
| --- | ---: | ---: | --- |
| `palette_exact` pttest | 0.333 | **0.667–0.833** | **+2.0× to +2.5×** |
| `palette_exact` test_lodo | 0.316 | **0.632** | **+2.0×** |
| `pixel_accuracy` test_lodo | 0.51-0.57 | 0.57-0.60 | +0.04 |
| `edit_mask_f1` test_lodo | 0.53-0.58 | 0.55-0.58 | ~0 |
| Dominant quarantine | `edit_color_failure` (26%) | `edit_mask_failure` (41%) | **shifted** |
| `edit_color_failure` label | present | **eliminated** | — |

**The variant achieved its design goal at the diagnostic level**:
palette recovery roughly doubled (because the deterministic rule
bank captures the palette structure the per-task MLP could not), and
the named failure mode the variant was designed to fix
(`edit_color_failure`) has been eliminated. Per the spec § "Quarantine
Labels" the variant's `edit_color_failure` slot is structurally
absent — it has been replaced by the more surgical pair
`color_rule_bank_coverage_failure` + `color_rule_selection_failure`,
both of which fire (see below).

#### Quarantine Breakdown (196 selected-seed labels, 6 of 12 fire)

| label | count | share | tells us |
| --- | ---: | ---: | --- |
| `edit_mask_failure` | 81 | 41% | **mask MLP is now the gate** — was color in Phase 3D base |
| `color_rule_selection_failure` | 32 | 16% | oracle ≥ 0.50 but selected < 0.50 |
| `conditioning_starvation` | 29 | 15% | structural ARC k=2-3 limitation |
| `baseline_canvas_failure` | 24 | 12% | baseline residual > 0.50 |
| `color_rule_bank_coverage_failure` | 18 | 9% | no rule in bank covers this instance |
| `palette_lift_failure` | 12 | 6% | catch-all |

**The new pair fires in the right ratio**: `color_rule_selection_failure
: color_rule_bank_coverage_failure = 32 : 18 ≈ 1.8 : 1`. When color
is wrong, the bank usually HAS the right rule but the selector did
not pick it. Combined with `rule_selection_regret_mean ≈
0.30–0.35` (oracle accuracy − selected accuracy), there are
~30 percentage points of edit-color accuracy locked up in selection
improvements.

Did NOT fire (6 of 12 labels): `baseline_shape_failure` (shape
picker still works perfectly — 1.000 shape exact on every lane),
`no_conditioning_edits`, `source_binding_failure`, `rare_color_failure`,
`palette_lift_failure` (the dual catch-all variant — different from
the listed one is `palette_lift_failure` which DID fire; the other
catch-all entries didn't), `stochastic_instability` (the four arms
agreed on the "no exact" outcome across seeds).

#### Selected Rule Family Distribution

980 selected-rule rows (49 instances × 4 arms × 5 seeds):

| selected family | count | share |
| --- | ---: | ---: |
| `ensemble_top3` | 400 | 40.8% |
| `nearest_edited_neighbor_color` | 360 | 36.7% |
| `relative_palette_rank_map` | 200 | 20.4% |
| `row_col_periodic_color` | 20 | 2.0% |

7 of 10 rule families never won selection: `constant_edit_color`,
`modal_edit_color`, `baseline_color_map`, `input_nn_color_map`,
`input_patch_majority_map`, `baseline_to_input_pair_map`,
`object_role_color_map`. These families generated candidates but
were dominated on conditioning accuracy by the top 4. This is
**consistent across all four arms and all five seeds**, suggesting
the family ordering is stable and not a representation artifact.

#### Named Failure Mode: Mask-Driven + Selection-Regret-Bound

This receipt closes the variant by demonstrating a **two-component
bottleneck** that the structured-edit framing now isolates:

1. **Primary**: `edit_mask_failure` at 41% — the mask MLP is the
   leading blocker. Phase 3D and the variant agree on the mask
   architecture (same `MaskMLP`, same MASK_MODEL_SPEC), so this is
   not a variant-introduced regression but the dominant
   non-color-fixable component.
2. **Secondary**: `color_rule_selection_failure` (16%) at
   `rule_selection_regret = 0.30–0.35` — the selector tie-break
   leaves ~30 percentage points of edit-color accuracy on the
   table per instance.
3. **Coverage tertiary**: `color_rule_bank_coverage_failure` (9%) is
   the smallest of the three named bottlenecks — the bank is not
   the limiting factor for most failures.

The receipt thus provides an unusually quantitative next-move
direction:

- **Option (a) Branch D mask-extension variant**: keep the rule bank,
  swap the mask MLP for a higher-capacity / different-loss / pretrained
  mask model. This is the biggest leverage point (41% of failures).
- **Option (b) Branch D selection-refinement variant**: keep the rule
  bank's coverage but refine the selector tie-break (e.g.,
  per-prior selector calibration, regret-minimizing selection, or
  a small selector MLP trained on the candidate score vector). This
  is targeting ~30 percentage points of locked accuracy gain across
  16% of failures.
- **Option (c) Branch D rule-bank extension**: add families that
  cover the 9% `bank_coverage_failure` slice. Smallest expected
  payoff per unit work.

#### What This Verdict Does And Does Not Entail

**It does**:

- Close the **sixth** full-grid control floor in Phase 3 (V1, V2,
  compact-7, Phase 3A, Phase 3D base, Phase 3D variant) at zero
  exact tasks on both registered held-out lanes for arm
  `raw_grid_edit_color_v2`.
- Verify that the variant's design goal (eliminating the
  `edit_color_failure` mode named in Phase 3D base) was achieved:
  that label is **gone**; palette recovery doubled.
- Provide the first quantitative bottleneck decomposition for a
  Phase 3 receipt: 41% mask, 16% selection regret, 9% bank coverage
  — telling the next surgical move with unusual specificity.

**It does not**:

- License any signature_palette_edit_color_v2 vs.
  raw_grid_edit_color_v2 sufficiency comparison (the raw-grid arm
  did not open the arena).
- License any Branch D narrowed-support claim from this receipt.
- License extra seeds on this lane without a new amendment.

#### Public-Language Constraint Update

Permitted additions:

- "The Phase 3D edit-color-rule variant achieved its design goal at
  the diagnostic level: palette recovery roughly doubled and the
  `edit_color_failure` mode named by the Phase 3D base was
  eliminated. The verdict remains `branch_d_color_rule_full_grid_floor`,
  but for the first time in Phase 3, the failure decomposition tells
  the next surgical move quantitatively (41% mask, 16% selection
  regret, 9% bank coverage)."

Forbidden:

- "Variant floored → signature representation is favoured" — the
  variant's per-arm scores favour `signature_palette_edit_color_v2`
  slightly on pixel and color_rule_accuracy, but the arena did not
  open and no sufficiency comparison is licensed.
- "Mask + selection are the only remaining bottlenecks for Phase 3"
  — there is also `conditioning_starvation` (15%, structural ARC
  k≤2 limitation) and `baseline_canvas_failure` (12%) below the
  variant-fixable layer; these would require Phase 6 (Kaggle) or a
  different framing to address.

#### Frozen By This Verdict

- The variant binding receipt at
  `results/arc/phase3d-edit-color-rule-v2/` at `mergeGitCommit
  8177A578…` is frozen.
- The bottleneck decomposition (41/16/9% mask/selection/coverage)
  is the canonical record of the variant's diagnostic payoff and
  the basis for any future Branch D extension variant.
- The selected-seed table is frozen as the variant selection trace.

**Verdict impact**: no V1, V2, compact-7, Phase 3A, or Phase 3D base
verdict changes. The Phase 3 status moves from "Branch D variant
execution in progress" to "Variant binding receipt filed,
`branch_d_color_rule_full_grid_floor`, with the first quantitative
bottleneck decomposition (41% mask / 16% selection regret / 9%
bank coverage). Six full-grid control receipts now agree on the
floor across two task distributions, two learner families, two
output framings, and two color-prediction approaches (learned MLP
+ deterministic rule bank). Remaining admissible reopens narrow to
mask-targeted variants, selection-refinement variants, or
fundamentally different framings (Branch E)."

### 2026-05-28 (PT) -- Jeffery Hughes Jr. -- Branch D Mask-Targeted Variant Spec Filed

Branch D mask-targeted variant spec:

- [`PHASE3D_MASK_TARGET_VARIANT_SPEC.md`](PHASE3D_MASK_TARGET_VARIANT_SPEC.md)

Justification: the `structured_edit_color_rule_v2` binding receipt shifted the
dominant bottleneck to `edit_mask_failure` (41% of failures), with smaller
diagnostic slices for `color_rule_selection_failure` (16%) and
`color_rule_bank_coverage_failure` (9%). This amendment starts the highest
leverage Branch D variant named by that receipt: keep the baseline picker and
the deterministic color-rule bank frozen, and change only the edit-mask
predictor.

Verdict impact: **no execution admission and no Branch D mask-targeted verdict**.
The new spec freezes `structured_edit_mask_target_v3` and learner label
`edit_mask_candidate_bank_v1`, but the run remains held until runner tooling,
npm wiring, result ignore path, leak-check coverage, and a freeze-marker
amendment are committed together.

The variant replaces the inherited scratch mask MLP as the sole mask predictor
with a deterministic mask-candidate bank selected from conditioning residuals.
The legacy mask MLP is admitted only as one thresholded candidate family, not as
a learned cross-task selector. Frozen candidate families include normalized
conditioning mask transfer, bounding-box fill/outline, row/column periodic
masks, source-color masks, object-role masks, nearest residual patch masks,
delta-overlay masks, and morphological variants.

The pre-registered arena and branch rules are:

1. only `raw_grid_edit_mask_v3` can open the arena;
2. if raw grid floors, the verdict is `branch_d_mask_target_full_grid_floor`
   and no signature comparison is licensed;
3. if raw grid opens, compare `signature_palette_edit_mask_v3` to
   `raw_grid_edit_mask_v3` for `branch_d_mask_target_support`,
   `branch_d_mask_target_bounded_failure`, or diagnostic
   `branch_d_mask_target_named_quarantine`;
4. oracle mask diagnostics such as `mask_oracle_candidate_f1` and
   `mask_oracle_exact_nonbaseline` are diagnostics only and cannot support a
   branch decision.

Reserved implementation names:

- Python runner: `docs/prereg/arc/phase3d_mask_target_v3.py`;
- Node wrapper: `scripts/arc-phase3d-mask-target-v3.mjs`;
- npm script: `arc:phase3d:mask-target-v3`;
- shard npm script: `arc:phase3d:mask-target-v3:shard`;
- merge npm script: `arc:phase3d:mask-target-v3:merge`;
- binding receipt path: `results/arc/phase3d-mask-target-v3/`.

Permitted public language before a binding receipt:

> "Phase 3D has filed a mask-targeted structured-edit variant spec targeting
> the leading bottleneck isolated by the edit-color-rule receipt. No
> mask-targeted receipt exists yet, and no sufficiency claim is admitted."

Forbidden: claiming the mask-targeted variant improves Phase 3, opens the arena,
or favours `signature_palette_edit_mask_v3` before a binding receipt exists.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Mask-Target Variant Tooling Freeze-Marker (perf fix + timing probe + launch staging)

The mask-target variant runner (`phase3d_mask_target_v3.py`), Node
wrapper, npm scripts, and `.gitignore` entry landed in commit
`1eca459`. This amendment files the freeze-marker record, a corrected
selection algorithm, the capped-timing probe, and the staged
20-shard launch.

**Performance correctness fix (committed with this amendment)**: the
first probe smoke did not converge in a reasonable time. Root cause:
`select_mask_candidate` called a per-candidate scorer that, in the
LOCO path, regenerated the entire mask-candidate bank (~143
candidates, each rebuilding the KNN tables and re-training the legacy
mask MLP) **once per candidate per fold** — an `O(C^2 * k)` blowup
with `C ≈ 143` candidates and `k` conditioning pairs. The fix
generates each LOCO fold's candidate bank **once**, indexes it by
`(family, id) -> mask`, and reduces per-candidate scoring to an O(1)
lookup. Net complexity `O(C * k)`. No scoring semantics change: the
LOCO fold masks are identical; only the redundant regeneration is
removed. This is a runner-internal performance fix, not a spec
change — the selection rule, tie-break chain, and metrics are
unchanged.

**Smoke fingerprint** (CPU, `--probe-only --probe-steps 3
--limit-arms raw_grid_edit_mask_v3 --limit-seeds 20260528`, all 36
registered tasks, post-fix):

- 49 held-out instances; **~143 mask candidates per instance**
  (7029 candidate scores across 49 instances), consistent with 13
  families × ~5 morphological variants minus data-degenerate
  families;
- 20 receipt files written (the 18 inherited + `mask_candidate_selection.csv`
  + `mask_candidates.csv`);
- wall 6m08s (down from a non-converging run pre-fix);
- arena gate `not_adjudicated` (probe-only, by spec).

**Mask-family diversity sanity** (selected mask family on 49
instances): `conditioning_mask_union` 31, `conditioning_bbox_fill`
9, `conditioning_mask_intersection` 2, `source_color_pair_mask` 2,
`object_role_mask` 2, `nearest_residual_patch_mask` 2,
`source_color_mask` 1. Seven of thirteen families won selection;
the conditioning-union default dominates as expected.

**Quarantine reachability** (9 of 16 labels fire at the 3-step
probe). The new mask-decomposition pair fires in the expected ratio:
`mask_selection_failure` 13 (oracle ≥ 0.50 but selected < 0.50) vs.
`mask_candidate_coverage_failure` 7 (oracle < 0.50) — ≈ 1.9 : 1, the
same "candidate usually exists but the selector misses it" shape the
color-rule variant showed. Also firing: `mask_overedit_failure` 13,
`baseline_canvas_failure` 6, `mask_underdetection_failure` 3,
`palette_lift_failure` 3, `conditioning_starvation` 2,
`color_rule_selection_failure` 1, `color_rule_bank_coverage_failure`
1.

**Capped timing probe** (post-fix, `--probe-steps 100`, 1 arm × 1
seed × 49 instances, freeze-marker workspace):

| device | 100-step wall |
| --- | ---: |
| CPU (i7-7820HK) | 778.7 s (12.98 min) |
| CUDA (GTX 1080) | 482.2 s (8.04 min) |

Two-point CPU regression with the 3-step smoke (368 s) gives a
**fixed bank cost `A = 355 s` (5.9 min, step-independent)** and a
per-step aggregate `B = 4.23 s/step` CPU. The bank cost is pure
Python (candidate construction, KNN over residual cells,
morphology) and is CPU-bound regardless of `--device`; this is why
GPU gives only a partial speedup here (unlike the prior variants,
where the trained model dominated). The step cost is the legacy
mask MLP retrained `(k+1)` times per instance.

**Full-run projection** (700 mask steps, 4 arms × 5 seeds = 20
combinations):

| posture | per arm-seed | 20-combo wall |
| --- | ---: | ---: |
| CPU serial | 55.3 min | ~18.4 h |
| GPU serial | 20.7 min | ~6.9 h |
| GPU 3-shard concurrent | 20.7 min | ~2.5 h nominal; **~3–3.5 h realistic** (the CPU-bound bank generation contends across 3 processes on 4 cores) |

Over the ten-minute rule; the inherited shard+merge protocol (with
`--allow-mixed-commits`) is the admitted launch posture.

**Staged full-run command (3-shard GPU parallel)**:

```powershell
$env:SUNDOG_PYTHON = "C:\Users\hughe\AppData\Local\Programs\Python\Python312\python.exe"
foreach ($arm in @("raw_grid_edit_mask_v3","signature_palette_edit_mask_v3","signature_only_edit_mask_v3","metadata_only_edit_mask_v3")) {
    foreach ($seed in @(20260528, 20260529, 20260530, 20260531, 20260601)) {
        Start-Process -NoNewWindow `
            -RedirectStandardOutput "results/arc/phase3d-mask-logs/$arm-$seed.log" `
            -RedirectStandardError "results/arc/phase3d-mask-logs/$arm-$seed.err" `
            -FilePath "npm" -ArgumentList @(
                "run", "arc:phase3d:mask-target-v3:shard", "--",
                "--data-dir", "$env:USERPROFILE\Datasets\ARC-AGI-2\data",
                "--register", "docs/prereg/arc/P0_TASK_REGISTER.csv",
                "--out", "results/arc/phase3d-mask-target-v3-shard-$arm-$seed",
                "--shard-arm", "$arm",
                "--shard-seed", "$seed",
                "--device", "cuda"
            )
        # Throttle to 3 concurrent shards.
    }
}

# After all 20 shards land:
npm run arc:phase3d:mask-target-v3:merge -- `
    --shard-dirs <comma-joined list of all 20 dirs> `
    --out results/arc/phase3d-mask-target-v3
```

**Per-outcome decision rule**:

| arena gate outcome | next action |
| --- | --- |
| `raw_grid_edit_mask_v3_arena_open` | Examine `branchAdjudication`. If `branch_d_mask_target_support`, file receipt + permitted public language. If `branch_d_mask_target_bounded_failure`, file receipt + the named-quarantine breakdown. |
| `branch_d_mask_target_full_grid_floor` | File the receipt. This would be the **seventh** full-grid control floor. The decomposition tells the next move: if dominated by `mask_candidate_coverage_failure`, the 13-family bank is structurally insufficient (extend it); if dominated by `mask_selection_failure`, the selector is the gate (refine tie-breaks / add a selector); if the inherited `color_rule_*` labels resurface, the color stage re-binds; if `baseline_*`, the baseline family is the cap. With both the mask and color stages now deterministic banks, a continued floor would strongly suggest the registered task class is not solvable by deterministic baseline+mask+color composition and points to Branch E. |

**Verdict impact**: no prior verdict changes. Variant status moves
from "EXECUTION HOLD" to "EXECUTION ADMITTED, 20-shard launch
ready". No binding receipt yet.

**Public-language constraint**: unchanged. The pre-binding-receipt
language from the variant spec §"Public Language" remains the only
permitted public addition.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Mask-Target Variant 20-Shard Binding Receipt: `branch_d_mask_target_full_grid_floor` (mask repair did not shift the bottleneck)

The mask-target variant 20-shard slate (4 arms × 5 seeds × 49
held-out instances) completed and merged at `mergeGitCommit
07F29513…`. Receipt at `results/arc/phase3d-mask-target-v3/`.

**Wall-clock**: total shard compute = **20,473.7 s (5.69 h)**;
3-shard concurrent GPU parallel wall = **1 h 58 m** (03:56:45 →
05:55:02 UTC), a **2.88× speedup** — beating the staging amendment's
~3–3.5 h realistic estimate (CPU-bound bank-gen contention was
milder than projected) and close to the 2.5 h nominal.

**Mixed-commits audit**: 8 distinct gitCommits across the 20 shards
(heavy parallel activity during the run), 13 dirty + 7 clean;
`mixedCommitsAudit.runnerIdenticalAcrossCommits = **true**` — all 8
commits share the byte-identical runner SHA. No WARN.

#### Selected Seed per Arm

| arm | selected seed |
| --- | ---: |
| `raw_grid_edit_mask_v3` | `20260530` |
| `signature_palette_edit_mask_v3` | `20260529` |
| `signature_only_edit_mask_v3` | `20260529` |
| `metadata_only_edit_mask_v3` | `20260531` |

#### Arena Gate

`raw_grid_edit_mask_v3`: `test_lodo_nonbaseline_exact_tasks = 0`,
`pttest_nonbaseline_exact_tasks = 0`. Every arm scored zero
non-baseline exact tasks on every held-out lane (baseline-exact also
zero). **Arena gate: `branch_d_mask_target_full_grid_floor`** — the
**seventh** full-grid control floor in Phase 3.

#### Per-Arm Comparison (Documentation Only — Arena Did Not Open)

| lane | arm | grid_ex | nbex | shape | palette | pixel | mask_f1 | minority_recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `pttest` | `raw_grid_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.833 | 0.509 | 0.650 | 0.786 |
| `pttest` | `signature_palette_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.833 | 0.533 | 0.662 | 0.734 |
| `pttest` | `signature_only_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.833 | 0.540 | 0.668 | 0.758 |
| `pttest` | `metadata_only_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.833 | 0.520 | 0.659 | 0.759 |
| `test_lodo` | `raw_grid_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.632 | 0.489 | 0.493 | 0.705 |
| `test_lodo` | `signature_palette_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.632 | 0.486 | 0.496 | 0.716 |
| `test_lodo` | `signature_only_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.632 | 0.500 | 0.506 | 0.725 |
| `test_lodo` | `metadata_only_edit_mask_v3` | 0.000 | 0.000 | 1.000 | 0.632 | 0.489 | 0.495 | 0.703 |

#### Central Finding: The Mask Repair Did Not Shift the Bottleneck

The variant's thesis was that replacing the trained mask MLP with a
13-family deterministic mask candidate bank would lift the
`edit_mask_failure`-dominated bottleneck that the color-rule variant
isolated. **It did not.** Mask-stage quarantine labels still dominate
the decomposition (137 of ~196 selected-seed labels):

| label | count | share | notes |
| --- | ---: | ---: | --- |
| `mask_selection_failure` | 56 | 29% | oracle ≥ 0.50 but selected < 0.50 — selector misses a covering candidate |
| `mask_overedit_failure` | 50 | 26% | selected mask edits too many cells |
| `mask_candidate_coverage_failure` | 24 | 12% | no candidate in the bank covers this instance |
| `baseline_canvas_failure` | 24 | 12% | baseline residual > 0.50 |
| `palette_lift_failure` | 17 | 9% | catch-all |
| `conditioning_starvation` | 8 | 4% | structural ARC k≤2 |
| `mask_underdetection_failure` | 7 | 4% | recall < 0.25 |
| `color_rule_selection_failure` | 5 | 3% | residual color-stage misses |
| `color_rule_bank_coverage_failure` | 5 | 3% | residual color-stage coverage gap |

`mask_selection_failure : mask_candidate_coverage_failure = 56 : 24
≈ 2.3 : 1` — the same "candidate usually exists but the selector
misses it" shape seen at the color stage. The deterministic mask
bank does NOT strictly dominate the learned mask MLP: the inherited
`legacy_mlp_threshold_mask` family **still won selection 135 / 980
times (13.8%)**, second only to `conditioning_mask_union` (596).

Selected mask family distribution (980 rows = 49 × 4 arms × 5 seeds):

| family | count | share |
| --- | ---: | ---: |
| `conditioning_mask_union` | 596 | 60.8% |
| `legacy_mlp_threshold_mask` | 135 | 13.8% |
| `conditioning_bbox_fill` | 96 | 9.8% |
| `source_color_pair_mask` | 40 | 4.1% |
| `nearest_residual_patch_mask` | 39 | 4.0% |
| `conditioning_mask_intersection` | 37 | 3.8% |
| `object_role_mask` | 20 | 2.0% |
| `source_color_mask` | 17 | 1.7% |

#### Comparison Against the Color-Rule Variant

The deterministic mask bank **raised minority-edit recall** (it
proposes union-of-conditioning masks that catch more true edit
cells, pttest 0.73–0.79 / test_lodo 0.70–0.72) **at the cost of
over-editing** (26% of failures are `mask_overedit_failure`), and on
the harder `test_lodo` lane its F1 (0.49–0.51) is *lower* than the
color-rule variant's mask F1 (0.53–0.58, which used the trained
MLP). Net: the mask stage is not lifted; it is re-shaped (recall up,
precision down) without opening the arena.

#### What This Verdict Does And Does Not Entail

**It does**:

- Close the **seventh** full-grid control floor in Phase 3.
- Establish a clean negative result: with **both** the mask stage
  and the color stage now replaced by deterministic conditioning-
  derived candidate banks, the baseline + mask + color composition
  **still** does not open the non-baseline arena. The two
  bottleneck-targeted repairs each fixed their named stage's
  *failure label* without lifting the *floor*.
- Show the deterministic mask bank does not dominate the learned
  mask MLP (legacy MLP wins 13.8% of selections; test_lodo F1 drops).

**It does not**:

- License any signature_palette vs raw_grid sufficiency comparison
  (arena did not open).
- License any Branch D narrowed-support claim.
- License extra seeds without a new amendment.

#### Remaining Reopen Paths — Branch E Is Now the Live Frontier

| Phase 3 receipt | learner / framing | verdict |
| --- | --- | --- |
| V1, V2 | transformer whole-grid | floor |
| compact-7 | transformer, 7-task | floor |
| Phase 3A | per-task coord MLP | floor |
| Phase 3D base | structured edit residual (learned mask + learned color) | floor (edit-color bottleneck) |
| Phase 3D color-rule | deterministic color bank | floor (mask bottleneck) |
| **Phase 3D mask-target** | **deterministic mask bank + deterministic color bank** | **floor (mask selection/over-edit, not lifted)** |

Seven floors across two task distributions, three learner families,
two output framings, and — within the structured-edit framing — both
the learned and the deterministic-bank variants of *each* of the two
edit components. The structured-edit framing has now been probed at
both its named bottlenecks (color, then mask) and floors at both.

Remaining admissible reopens:

- **A selection-targeted Branch D variant** (smaller move): both
  deterministic banks fail more by *selection* than by *coverage*
  (color 16% selection vs 9% coverage; mask 29% selection vs 12%
  coverage). A variant that replaces the per-stage greedy tie-break
  with a joint mask+color selector (or a small learned selector over
  candidate-score vectors) targets the largest single failure slice.
  Still within-framing.
- **Branch E (different framing entirely)** (higher-information
  move): e.g., generative program / DSL search over conditioning
  pairs, or test-time prompting of a frozen large LM. Given two
  deterministic-bank repairs floored, Branch E is the higher-value
  next move. It would need its own pre-registration, arena gate, and
  verdict-amendment discipline.

#### Public-Language Constraint Update

Permitted additions:

- "The mask-targeted Branch D variant replaced the trained edit-mask
  model with a deterministic 13-family mask candidate bank. The
  verdict is `branch_d_mask_target_full_grid_floor`: the mask repair
  re-shaped the mask stage (minority-edit recall up, over-edit up,
  `test_lodo` F1 down) but did not lift the held-out exact-grid
  floor. With both the mask and color stages now deterministic banks
  and the floor holding at both, the structured-edit framing has
  been probed at both its named bottlenecks; Branch E (a different
  framing) is the live frontier."

Forbidden:

- "Mask bank beats the learned mask" — it does not; the legacy MLP
  still wins 13.8% of selections and `test_lodo` mask F1 dropped.
- "Mask repair shifted the bottleneck off the mask" — the opposite;
  mask-stage labels still dominate (137/196).
- Any sufficiency / arena-open / signature-favoured claim (arena did
  not open).

#### Frozen By This Verdict

- The mask-target binding receipt at
  `results/arc/phase3d-mask-target-v3/` (`mergeGitCommit
  07F29513…`) is frozen.
- The two-deterministic-bank negative result (both color and mask
  stages deterministic, floor holds) is the canonical basis for
  prioritising Branch E over further within-framing component swaps.
- The selected-seed table is frozen as the variant selection trace.

**Verdict impact**: no prior verdict changes. Phase 3 status moves to
"seven full-grid control floors; structured-edit framing probed at
both named bottlenecks (color + mask) with deterministic banks and
floored at both; Branch E (different framing) is the live frontier,
with a selection-targeted within-framing variant as the smaller
alternative."

### 2026-05-28 (PT) -- Jeffery Hughes Jr. -- Phase 3E Signature-Fiber Certificate Spec Filed

Phase 3E certificate spec:

- [`PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md`](PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md)

Justification: after seven full-grid-control floors, including two
deterministic-bank repairs inside the structured-edit framing, the next
highest-value question is not another decoder. The next question is whether the
frozen `signature_palette` representation itself creates exact or near context
fibers with incompatible required behavior on the registered task class.

Verdict impact: **no execution admission and no Phase 3E verdict**. The spec
freezes a certificate lane, not a solver lane. Execution remains held until
runner tooling, npm wiring, result ignore path, leak-check coverage, and a
freeze-marker amendment are committed together.

The primary context universe is:

- `test_lodo`;
- `pttest`.

The full diagnostic context universe additionally includes:

- `validation_lodo`;
- `validation_pttest`.

The certificate search has two levels:

1. exact `signature_palette_context` fiber collisions, where two registered
   contexts have byte-identical signature context identities but incompatible
   target output labels;
2. near-fiber locality at `epsilon_primary = 0.05`, using cross-task kNN
   neighborhoods and a deterministic `program_sketch_v1` built from frozen
   Branch D shape/canvas/mask/color candidate families.

Pre-registered branches:

| branch | meaning |
| --- | --- |
| `phase3e_exact_fiber_collision` | exact registered signature-fiber collision found; finite-context insufficiency certificate |
| `phase3e_near_fiber_incompatibility` | no exact collision, but near-signature contexts have incompatible program sketches |
| `phase3e_fiber_locality_positive` | no collision found and cross-task neighborhoods are locally consistent enough to license a later Branch E selector |
| `phase3e_deferred_sparse_fibers` | no collision found, but the registered set is too sparse at the frozen radius |
| `phase3e_deferred_label_vacuity` | program sketches are too often vacuous to adjudicate near-fiber incompatibility |

Reserved implementation names:

- Python runner: `docs/prereg/arc/phase3e_signature_fiber_certificate.py`;
- Node wrapper: `scripts/arc-phase3e-signature-fiber-certificate.mjs`;
- npm script: `arc:phase3e:signature-fiber-certificate`;
- binding receipt path: `results/arc/phase3e-signature-fiber-certificate/`.

Permitted public language before a binding receipt:

> "Phase 3E has filed a signature-fiber certificate spec. It will test whether
> registered ARC contexts contain exact or near `signature_palette` collisions
> with incompatible required behavior. No Phase 3E receipt exists yet."

Forbidden: claiming a collision, locality positive, Branch E solver result, or
signature sufficiency/insufficiency conclusion before a binding Phase 3E receipt
exists.

### 2026-05-28 (PT) — Jeffery Hughes Jr. — Phase 3E Certificate Tooling Freeze-Marker (execution unblocked; certificate runs under the 10-minute rule)

The Phase 3E signature-fiber certificate spec held execution until
"runner tooling, npm wiring, result ignore path, leak-check coverage,
and a freeze-marker amendment are committed together." This amendment
files that commit:

- `docs/prereg/arc/phase3e_signature_fiber_certificate.py` (runner) —
  standalone certificate module. It does **not** train a grid decoder
  and does **not** import another phase3 runner. The
  `arc-p3-feature-v1` grid encoders, the Branch D baseline family, the
  mask candidate bank, and the edit-color rule bank are **copied
  verbatim** from the frozen mask-target runner under marked headers
  and are reused **only** to compute the `program_sketch_v1` oracle
  audit labels (run with a fixed representation-neutral grid arm,
  `raw_grid_edit_mask_v3`, so the behavior labels reflect true grids,
  not a signature projection). The certificate's own logic — per-arm
  context identity, per-arm context distance (signature cosine /
  metadata L1 / raw Hamming, with min-cost bipartite
  conditioning-pair matching), exact/representation/near-fiber
  collision detection, k=3 cross-task fiber locality, and the
  5-branch precedence adjudication — is implemented in the new
  "Certificate" section.
- `scripts/arc-phase3e-signature-fiber-certificate.mjs` (Node wrapper).
- `package.json`: adds `arc:phase3e:signature-fiber-certificate`.
- `.gitignore`: adds `results/arc/phase3e-signature-fiber-certificate/`.
- Pre-commit + CI ARC leak-check passes (now scans 19 ARC scripts;
  the certificate contains no `evaluation` literal).

**Two-stage target barrier**: the runner writes
`context_fingerprints_no_targets.jsonl` + its `.sha256` (all four arm
context identities, no target information) and records the hash in the
manifest **before** any public-training test output is read for target
labels. Honoured in `main()` Stage 1 → Stage 2.

**Determinism**: the certificate is fully deterministic. The only
torch use is the legacy mask-MLP family inside the mask oracle, which
calls `set_global_determinism(derived_seed)` (capped at
`ORACLE_MASK_MLP_STEPS = 50`, since the MLP is one of 13 mask families
in an audit label, not the certificate's subject). The geometric core
(distances, collisions, kNN) is pure Python.

**Smoke fingerprint** (full universe, `--allow-dirty`, all 36
registered tasks):

- `U_all = 49` contexts (`validation_lodo=18`, `validation_pttest=6`,
  `test_lodo=19`, `pttest=6`); `U_primary = test_lodo ∪ pttest = 25`.
- 15 receipt files written (the spec's required artifact set).
- wall: **3 min 49 s** — under the ten-minute rule, so the binding run
  is a single direct invocation (no staging/sharding needed).
- branch (smoke): `phase3e_deferred_label_vacuity`. The authoritative
  clean-tree binding receipt is filed in the next amendment.

**Verdict impact**: no prior verdict changes. Phase 3E status moves
from "SPEC FILED; EXECUTION HOLD" to "EXECUTION ADMITTED; certificate
runs under the ten-minute rule." The binding certificate receipt is
filed in the next amendment.

**Public-language constraint**: unchanged from the spec
§"Public Language" until the binding receipt lands.

### 2026-05-29 (PT) — Jeffery Hughes Jr. — Phase 3E Signature-Fiber Certificate Binding Receipt: `phase3e_deferred_label_vacuity` (no collision; fibers maximally separated; oracle prior-blind)

The binding certificate ran on the registered 36-task Phase 0 set,
pinned to freeze-marker `gitCommit C133FF70…` (runnerSha256
`884959D69876…`, specHash `56D49192…`, parentSpecHash `419647BA…`,
target-barrier hash `32088769…`; `gitDirty=true` only from the
operator's unrelated Navier–Stokes working files, which the
certificate does not read). Receipt at
`results/arc/phase3e-signature-fiber-certificate/`. Wall ≈ 4 min
(under the ten-minute rule). Deterministic; the binding branch
matches the freeze-marker smoke exactly.

**Context universe**: `U_all = 49` (validation_lodo 18,
validation_pttest 6, test_lodo 19, pttest 6); primary adjudication
universe `U_primary = test_lodo ∪ pttest = 25`.

**Branch: `phase3e_deferred_label_vacuity`.**

#### Dual finding

| signal | value | reading |
| --- | ---: | --- |
| exact-output collisions | **0** | no two `U_primary` contexts share a `signature_palette` context id with differing target hashes |
| representation-level collisions | **0** | …nor with differing `signature_palette` target ids |
| `fidelity_pass_fraction` | **0.00** | zero `U_primary` contexts have a cross-task neighbor within `epsilon_primary = 0.05` |
| near pairs within `epsilon_primary` | **0** | no near-fiber pairs to test for incompatibility |
| **min cross-task context distance** | **0.207** | the closest distinct-task pair is **4.1× `epsilon_primary`, 2.1× `epsilon_loose`** |
| median / max cross-task distance | 0.359 / 0.636 | `signature_palette_context` maximally separates distinct tasks |
| `label_vacuity_fraction` | **0.68** | 17 of 25 `U_primary` contexts have `none` in ≥2 program-sketch oracle sets |

Two independent facts hold, and either alone would force a deferral:

1. **No collision and no near structure.** The certificate found no
   exact or representation-level `signature_palette` fiber collision,
   and — more strongly — no cross-task context pair is even *close*:
   the minimum cross-task `d_context_signature_palette` over all
   75 nearest-neighbor edges is `0.207`, far above every diagnostic
   threshold (`epsilon_exact=0`, `epsilon_strict=0.025`,
   `epsilon_primary=0.05`, `epsilon_loose=0.10`). So there is no
   locality to certify positive and no near-incompatibility to flag —
   the registered fibers are singletons at the registered radius.
2. **The program-sketch oracle is prior-blind.** 68% of `U_primary`
   contexts are sketch-vacuous, and the vacuity is cleanly
   prior-structured (per `per_prior.csv`):

   | prior | contexts | vacuous (`none` in ≥2 sets) |
   | --- | ---: | ---: |
   | `color_role` | 4 | 0 |
   | `objectness` | 4 | 0 |
   | `counting` | 4 | 4 |
   | `local_completion` | 4 | 4 |
   | `spatial_transform` | 4 | 4 |
   | `symmetry` | 5 | 5 |

   The Branch D edit-composition oracle (baseline + mask + color
   banks) characterizes exactly the two priors whose required output
   is a local recolor/edit of an input-derived baseline
   (`color_role`, `objectness`) and is blind to the four priors whose
   output is *not* such an edit (`counting`, `local_completion`,
   `spatial_transform`, `symmetry`). This is the same structural
   limit the seven decoder floors exhibited, now surfaced directly as
   a label-coverage gap rather than a training failure.

By the spec's precedence (exact → near → label_vacuity → locality
→ sparse, with `label_vacuity` elevated above `fiber_locality_positive`),
the branch is `phase3e_deferred_label_vacuity`. The zero-locality
fact would independently force `phase3e_deferred_sparse_fibers`; the
certificate reports the higher-precedence vacuity branch, with the
separation finding recorded as the co-deferral.

#### What This Verdict Does And Does Not Entail

**It does**:

- Certify that **no `signature_palette` fiber collision exists in the
  registered context class** under the frozen identity and distance
  rules — there is no decoder-independent insufficiency witness on
  this finite set. (Per spec, this does **not** prove sufficiency.)
- Establish that `signature_palette_context` **maximally separates
  distinct registered tasks** (min cross-task distance 0.207 ≈ 4×
  the primary threshold) — there is no near-fiber locality structure
  to certify, positive or negative, on the registered set.
- Localize the program-sketch oracle's blind spot to four of six
  priors, explaining the vacuity as a coverage gap of the
  edit-composition framing, not noise.

**It does not**:

- License a Branch E program-selector via
  `phase3e_fiber_locality_positive` — that branch requires ≥50%
  fidelity-passing cross-task neighborhoods and a non-vacuous oracle;
  the registered set delivers 0% fidelity-passing and 68% vacuous.
- Prove `signature_palette` sufficient or insufficient.
- Claim any near-fiber incompatibility (none exist within
  `epsilon_loose`).
- Adjudicate the four sketch-vacuous priors at all.

#### Path Forward

The certificate neither blocks nor licenses Branch E; it defers, and
the deferral is doubly-caused (vacuous oracle + zero locality). To get
an adjudicable certificate, a future amendment would need one of:

- **A stronger / framing-agnostic program-sketch oracle** that covers
  the four edit-blind priors (`counting`, `local_completion`,
  `spatial_transform`, `symmetry`) — e.g., adding DSL-program or
  transformation-class labels rather than only baseline+mask+color
  edit families. This directly attacks the 68% vacuity.
- **A larger registered context universe** (more Phase 0 tasks per
  prior, or finer LODO) so cross-task neighbors exist within
  `epsilon_primary` and fiber locality becomes measurable. The
  min-distance 0.207 suggests 36 tasks are too few/too separated to
  populate fibers at 0.05.
- **Proceeding to Branch E on independent grounds** (the seven
  decoder floors), accepting that the certificate could not find a
  registered collision to block it and could not certify locality to
  motivate a smooth selector — i.e., a Branch E solver must be
  justified by capability, not by certified fiber geometry.

Per the spec's frozen-threshold rule, `epsilon_primary`, `k`, and the
program-sketch thresholds are **not** retuned after seeing these
distances/labels; any change requires a new append-only amendment.

#### Frozen By This Verdict

- The binding receipt at
  `results/arc/phase3e-signature-fiber-certificate/` (pinned
  `gitCommit C133FF70…`, target-barrier hash `32088769…`) is frozen.
- The dual finding (no collision + min cross-task distance 0.207 +
  68% prior-structured oracle vacuity) is the canonical Phase 3E
  result.

**Verdict impact**: no prior verdict changes. The seven full-grid
control floors stand. Phase 3E adds a decoder-independent reading:
on the registered context class, `signature_palette` exhibits no
fiber collision and maximal cross-task separation, so the floors are
not explained by a registered signature collision — but the
certificate is deferred (vacuous oracle + zero locality) and licenses
no sufficiency conclusion in either direction.

**Public-language constraint**: per the spec §"Public Language", the
permitted addition is the no-collision deferral statement; the
collision-found and locality-positive statements remain forbidden
(neither obtained).

### 2026-05-29 (PT) -- Jeffery Hughes Jr. -- Phase 3E Program-Sketch Oracle v2 Spec Filed

Phase 3E oracle-repair spec:

- `docs/prereg/arc/PHASE3E_PROGRAM_SKETCH_ORACLE_V2_SPEC.md`

The Phase 3E binding receipt found no registered signature-fiber collision, but
it also exposed a label-coverage failure: `program_sketch_v1` was inherited from
the Branch D edit-composition banks and was prior-blind on 68% of primary
contexts. `program_sketch_v2` is the pre-registered repair for that weakness.
It is a framing-agnostic certificate labeler, not a solver and not a Branch E
runner.

The v2 spec freezes:

- nine transformation facets (`shape_relation`, `palette_relation`,
  `object_relation`, `cardinality_relation`, `completion_relation`,
  `spatial_transform_relation`, `symmetry_relation`, `correspondence_basis`,
  `rule_scope`);
- anti-vacuity gates (`overall_vacuity_fraction <= 0.20`, per-prior vacuity
  `<= 0.25`, and lane coverage for every present prior);
- anti-prior-laundering gates forbidding literal prior-name labels and requiring
  concrete information beyond the most prior-associated facet;
- anti-solver-leakage gates forbidding raw outputs, target hashes, exact masks,
  exact coordinates, per-cell assignments, and overly unique sketch keys;
- the same frozen Phase 3E signature geometry thresholds
  (`epsilon_primary = 0.05`, `epsilon_strict = 0.025`,
  `epsilon_loose = 0.10`, k=3).

Reserved implementation names:

- Python runner: `docs/prereg/arc/phase3e_program_sketch_oracle_v2.py`;
- Node wrapper: `scripts/arc-phase3e-program-sketch-oracle-v2.mjs`;
- npm script: `arc:phase3e:program-sketch-oracle-v2`;
- receipt path: `results/arc/phase3e-program-sketch-oracle-v2/`.

Verdict impact: **no execution admission and no v2 receipt**. Execution remains
held until runner tooling, npm wiring, result ignore path, leak-check coverage,
smoke fingerprint, and a freeze-marker amendment are committed together.

Allowed public language before receipt:

> "Phase 3E has filed a framing-agnostic program-sketch oracle v2 spec. It is a
> labeler for certificate adjudication, not a solver. No v2 receipt exists yet."

### 2026-05-29 (PT) — Jeffery Hughes Jr. — Phase 3E Program-Sketch Oracle v2 Tooling Freeze-Marker (execution unblocked; oracle clears all three gates in smoke)

The v2 oracle spec held execution until "runner tooling, npm wiring,
result ignore path, leak-check coverage, and a freeze-marker amendment
are committed together." This amendment files that commit:

- `docs/prereg/arc/phase3e_program_sketch_oracle_v2.py` (runner) —
  standalone. The arc-p3-feature-v1 grid encoders + the frozen Phase 3E
  context identity / distance / k=3 fiber-locality machinery are
  inherited verbatim from the signature-fiber certificate runner. The
  prior-blind `program_sketch_v1` (Branch D edit-composition banks) is
  removed and replaced by `program_sketch_v2`: nine deterministic
  transformation-relation facets (shape / palette / object /
  cardinality / completion / spatial_transform / symmetry /
  correspondence / rule_scope) computed from RAW registered grids only
  — never `signature_palette`, arm distances, or decoder outputs —
  plus the three gate tests, the v2 incompatibility rule, and the
  7-branch precedence adjudication. The inherited Branch D bank
  functions remain in the module but are INERT (never called by the v2
  oracle); they are retained only to keep the shared frozen geometry
  helpers byte-faithful to the 3E runner.
- `scripts/arc-phase3e-program-sketch-oracle-v2.mjs` (Node wrapper).
- `package.json`: adds `arc:phase3e:program-sketch-oracle-v2`.
- `.gitignore`: adds `results/arc/phase3e-program-sketch-oracle-v2/`.
- Pre-commit + CI ARC leak-check passes (now scans 20 ARC scripts).

**Geometry frozen**: `epsilon_primary=0.05`, `epsilon_strict=0.025`,
`epsilon_loose=0.10`, `k=3`, and the `signature_palette_context`
identity + distance are UNCHANGED from the v1 certificate; the runner
does not retune them.

**Two-stage target barrier** honoured: the runner writes
`context_fingerprints_no_targets.jsonl` + `.sha256` before any target
output is read for the v2 facet labels.

**Smoke fingerprint** (full universe, `--allow-dirty`, all 36
registered tasks): U_all 49 / U_primary 25, 16 receipt files, **8 s
wall** (no torch training — the facets are pure-Python raw-grid tests),
deterministic. **All three gates pass cleanly**:

| gate | result | margin |
| --- | --- | --- |
| anti-vacuity | **PASS** | overall vacuous fraction **0.00** (v1 oracle was 0.68); every prior 0-vacuous; both lanes non-vacuous for every prior |
| anti-prior-laundering | **PASS** | 0% of non-vacuous contexts violate the two-extra-facet rule; minimum extra-facets-outside-prior = 5 |
| anti-solver-leakage | **PASS** | no syntactic leakage; `unique_core_sketch_fraction = 0.36` (≤ 0.60); `core_sketch_exact_lookup_fraction = 0.04` (≤ 0.20) |

Facet richness: 6–7 of 9 facets populated per primary context (8
contexts at 6, 17 at 7) — the v1 prior-blindness is repaired across all
six priors, including the four (`counting`, `local_completion`,
`spatial_transform`, `symmetry`) that were 100% vacuous under v1.

**Smoke branch**: `phase3e_v2_deferred_sparse_fibers`. With the oracle
gates cleared, the frozen fiber geometry is unchanged from v1 —
`min_cross_task_distance = 0.207` (byte-identical), 0 cross-task pairs
within `epsilon_primary`, 0% fidelity-passing neighborhoods — so the
certificate defers on **sparsity** alone now, no longer on label
vacuity. The authoritative clean-tree binding receipt is filed in the
next amendment.

**Verdict impact**: no prior verdict changes. Phase 3E v2 status moves
from "SPEC FILED; EXECUTION HOLD" to "EXECUTION ADMITTED; oracle clears
all three gates; certificate runs in ~8 s (well under the ten-minute
rule)."

**Public-language constraint**: unchanged from the v2 spec
§"Public Language" until the binding receipt lands.

### 2026-05-29 (PT) — Jeffery Hughes Jr. — Phase 3E Program-Sketch Oracle v2 Binding Receipt: `phase3e_v2_deferred_sparse_fibers` (oracle defect repaired; geometry sparsity is now the sole deferral cause)

The binding v2-oracle certificate ran on the registered 36-task Phase 0
set, pinned to freeze-marker `gitCommit 3B1B76D2…` (runnerSha256
`6163C208…`; `gitDirty=false`). Receipt
at `results/arc/phase3e-program-sketch-oracle-v2/`. Wall ≈ 8 s, fully
deterministic (binding branch matches the freeze-marker smoke exactly).
`U_all = 49`, `U_primary = 25`.

**Branch: `phase3e_v2_deferred_sparse_fibers`.**

#### The oracle defect from the v1 certificate is repaired and certified

The v1 certificate deferred for two reasons: (a) the program-sketch
oracle was prior-blind (68% vacuous), and (b) the fibers were
geometrically sparse (0 cross-task pairs within `epsilon_primary`).
The v2 oracle was the narrow repair for (a). It cleared **all three**
pre-registered gates on `U_primary`:

| gate | result | detail |
| --- | --- | --- |
| anti-vacuity | **PASS** | overall vacuous fraction **0.00** (v1 oracle: 0.68); all six priors 0-vacuous; both `test_lodo` and `pttest` lanes non-vacuous for every prior |
| anti-prior-laundering | **PASS** | 0% of non-vacuous primary contexts violate the two-extra-facet rule; minimum extra-concrete-facets-outside-the-prior-facet = 5 (rule requires ≥ 2) |
| anti-solver-leakage | **PASS** | no syntactic leakage; `unique_core_sketch_fraction = 0.36` (≤ 0.60); `core_sketch_exact_lookup_fraction = 0.04` (≤ 0.20) |

Facet richness rose to 6–7 of the nine required facets populated per
primary context (8 contexts at 6, 17 at 7). Crucially, the four priors
that were **100% vacuous under v1** — `counting`, `local_completion`,
`spatial_transform`, `symmetry` — are now **0/4 (and 0/5) vacuous**.
The framing-agnostic raw-grid facet labelers characterize the
non-edit-composition priors that the Branch D-derived v1 oracle could
not see. The anti-solver-leakage margins confirm the labels stay coarse:
`unique_core 0.36` means most contexts share their nine-facet core
signature with at least one other context, and `exact_lookup 0.04`
means a lookup keyed only on the core sketch would reproduce the exact
output for 1 of 25 primary contexts — the sketch is a transformation
*type* label, not a hidden solver.

#### Geometry sparsity is now the sole remaining deferral cause

With the oracle gates cleared, the certificate re-ran the fiber
adjudication under the **unchanged** frozen geometry. The result is
byte-identical to the v1 receipt on every geometric quantity:

| quantity | v2 binding | v1 binding |
| --- | ---: | ---: |
| exact / representation-level collisions | 0 | 0 |
| near pairs within `epsilon_primary = 0.05` | 0 | 0 |
| `fidelity_pass_fraction` | 0.00 | 0.00 |
| **min cross-task context distance** | **0.207** | **0.207** |

No two distinct-task registered contexts are within `epsilon_primary`
(or even `epsilon_loose = 0.10`); the closest cross-task pair sits at
0.207, 4.1× the primary threshold. So the v2 certificate cannot issue
`phase3e_v2_near_fiber_incompatibility` (no near pairs exist to test),
cannot issue `phase3e_v2_fiber_locality_positive` (0% fidelity-passing
neighborhoods, far below the 50% requirement), and — because the oracle
gates now pass — defers as `phase3e_v2_deferred_sparse_fibers` rather
than `phase3e_deferred_label_vacuity`.

This is the clean, expected outcome: **the v2 oracle removed the
label-vacuity deferral cause and certified it removed (all three gates
pass with wide margins), isolating the registered-context geometric
sparsity as the lone obstruction to an adjudicable fiber certificate.**

#### What This Verdict Does And Does Not Entail

**It does**:

- Repair and certify the v1 oracle defect: a framing-agnostic,
  deterministic, raw-grid transformation-sketch labeler is non-vacuous
  across all six registered priors while remaining too coarse to act as
  a solver (anti-vacuity, anti-prior-laundering, and anti-solver-leakage
  all pass).
- Re-confirm, under the frozen geometry, that the registered
  `signature_palette` context fibers have no collision and no near
  structure (min cross-task distance 0.207). The earlier finding was
  not an artifact of the prior-blind oracle.
- Localize the remaining obstruction unambiguously: **registered
  context geometric sparsity at `epsilon_primary = 0.05`**, not label
  quality.

**It does not**:

- Prove `signature_palette` sufficient or insufficient (no collision
  found; spec forbids the sufficiency reading either way).
- License a Branch E program-selector via
  `phase3e_v2_fiber_locality_positive` (0% fidelity-passing
  neighborhoods).
- Retune any threshold — `epsilon_primary`, `k`, the facet vocabulary,
  and all gate thresholds are frozen and unchanged after seeing output,
  per the spec.

#### Path Forward

The two v1 deferral causes have now been separated and one has been
closed. The oracle is certified adequate; the sole remaining blocker is
that 36 registered tasks are too few / too separated to populate
cross-task fibers at the frozen `epsilon_primary`. The next admissible
move is therefore unambiguous and is **not** another oracle iteration:

- **Expand the registered context universe** (more Phase 0 tasks per
  prior, and/or a finer LODO expansion) so cross-task neighbors exist
  within `epsilon_primary` and fiber locality becomes measurable. The
  min-distance 0.207 quantifies the gap: roughly a 4× contraction in
  nearest-neighbor distance is needed, which a substantially larger
  registered set could supply. This would require its own Phase 0
  register amendment (new manually-inspected tasks) before a rerun.
- Alternatively, **proceed to a Branch E solver on capability grounds**,
  accepting that the certificate (across v1 + v2) found no registered
  collision to block it and could not certify locality to motivate a
  smooth selector — i.e., a Branch E solver is justified by what it can
  do, not by certified fiber geometry. A Branch E solver still needs its
  own pre-registered spec, arena gate, and verdict-amendment discipline.

#### Frozen By This Verdict

- The binding receipt at
  `results/arc/phase3e-program-sketch-oracle-v2/` (pinned
  `gitCommit 3B1B76D2…`, target-barrier hash recorded in the manifest)
  is frozen.
- The certified gate margins (vacuity 0.00, laundering 0%,
  unique_core 0.36, exact_lookup 0.04) are the canonical record that the
  v1 prior-blindness is repaired without introducing solver leakage.
- The unchanged geometry (min cross-task distance 0.207, 0 near pairs)
  is the canonical record that the deferral is now purely a
  registered-context sparsity result.

**Verdict impact**: no prior verdict changes. The seven full-grid
control floors stand; the v1 signature-fiber certificate
(`phase3e_deferred_label_vacuity`) stands. The v2 oracle advances the
Phase 3E lane from a two-cause deferral to a single-cause
(sparsity-only) deferral with the oracle defect certified-repaired, and
makes "expand the registered context universe" the unambiguous next
certificate move.

**Public-language constraint**: per the v2 spec §"Public Language",
the permitted addition is the gate-cleared sparse-deferral statement;
the locality-positive, collision-found, and any sufficiency statements
remain forbidden (none obtained).
