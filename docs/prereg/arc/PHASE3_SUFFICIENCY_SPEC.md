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
