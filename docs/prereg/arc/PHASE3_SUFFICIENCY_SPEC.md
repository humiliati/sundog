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
