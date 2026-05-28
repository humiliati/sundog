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
