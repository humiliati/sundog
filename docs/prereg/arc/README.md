# ARC-AGI Abstraction Pre-Registration

Roadmaps:
[`SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md),
[`SUNDOG_V_GRAVITY.md`](../../SUNDOG_V_GRAVITY.md) Candidate 14

Filed: **2026-05-28 (PT)**

Status: **Phase 3D BINDING RECEIPT FILED -- `branch_d_full_grid_edit_floor`; ALL 4 REFLECTION BRANCHES + BRANCH D NOW CLOSED**.
Phase 0 admitted; Phase 1 synthetic gate strengthened and passed; Phase 2
projection-measurement plus baseline-comparison passed; Phase 3 filed
three deterministic-low-capacity binding receipts (`nn_output_transfer_v1`,
`nn_delta_transfer_v1`, `candidate_combinator_v1`), all verdict
**task hardness / decoder failure**, and one Blackwell sufficiency lane.
The three-receipt convergence is characterised at
[`PHASE3_5_REFLECTION.md`](PHASE3_5_REFLECTION.md) as a finding about
the deterministic-low-capacity-learner family rather than a
sufficiency-failure conclusion about the shadow-projection representation.
The Blackwell sufficiency amendment in
[`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md) filed algebraic
conditions, held-out-task splits, Branch A/B/C criteria, a frozen
Python/PyTorch decoder design, a timing probe, and a full clean receipt for
`blackwell_task_decoder_v1` -- **Branch C bounded failure**:
`signature_palette` scored zero exact matches on both held-out-task LODO
and held-out public-training test lanes, and the matched `raw_grid_lowcap`
control also scored zero exact matches. The strengthened follow-up
`blackwell_publictrain_rawgrid_gate_v2` was launched as a raw-grid-only
control gate with all 971 admitted public-training auxiliary tasks (29
excluded by the registered `MAX_DEMOS = 5` token cap), 3 seeds, 120-epoch
budget, on GPU under a shard+merge protocol with byte-equivalence to a
serial run; freeze-marker commit `79C5B060`. The V2 binding receipt is
**`full_grid_control_floor`**: every held-out lane (`pttest`, `test_lodo`,
`validation_lodo`, `validation_pttest`) scored `grid_exact_any_rate = 0.0`
at selected seed `20260529`. The shape-exact-slot1 rate sits at 0.50-0.72
across lanes while palette-exact-slot1 holds at 0.0 -- the same
shape-matches-content-fails character as V1, now reproduced under the
strengthened V2 lane.

A Branch B compact-subset diagnostic lane was then admitted with a
narrower question: does the registered decoder produce any exact-match
signal on the 7 Phase 2 compact-signal tasks? The compact-7 single-seed
binding receipt (
[`PHASE3B_COMPACT_SUBSET.md`](PHASE3B_COMPACT_SUBSET.md), seed `20260529`,
freeze-marker `50EAEBBF`) returned verdict
**`compact_full_grid_control_floor`** with a qualitatively distinct
failure character: every held-out instance gets the output shape exactly
right (`shape_exact_slot1 = 1.000` on all four lanes) and pixel accuracy
jumps to 0.757-0.877, but every prediction collapses to the dominant
background color (palette_exact_slot1 = 0.000 everywhere; 13 of 13
held-out predictions slot-1-use at most 2 colors regardless of target
palette size). This is filed as a named failure mode -- **"dominant-color
mode collapse"** -- distinct from V1/V2's noise-dominated character.
Per the pre-registered §8 stop rule, this verdict closes Branch B in the
deterministic-low-capacity-learner family: `signature_palette` does not
run on this subset, additional seeds do not run, and the path forward
is PHASE3_5_REFLECTION Branch A (stochastic per-task learner) or
Branch D (different framing) rather than further narrowing within the
deterministic family. Kaggle notebook work and public-evaluation grid
inspection remain blocked until Phase 6.

Branch A is filed at
[`PHASE3A_STOCHASTIC_PER_TASK_SPEC.md`](PHASE3A_STOCHASTIC_PER_TASK_SPEC.md).
`per_task_coord_mlp_v1` is a stochastic per-instance coordinate MLP trained
from scratch on only that instance's conditioning demonstrations. The
20-shard binding receipt (4 arms x 5 seeds, ~2.1 h GPU parallel wall on a
GTX 1080, sharded with the spec's (arm, seed) shard+merge protocol and the
`--allow-mixed-commits` operator override after parallel Navier-Stokes
commits landed mid-flight) returned verdict **`branch_a_full_grid_floor`**:
zero exact tasks on both held-out lanes for arm `raw_grid_per_task`, so no
signature_palette_per_task vs. raw_grid_per_task sufficiency comparison is
licensed. Failure character is qualitatively distinct from V1, V2, and
compact-7: **conditioning starvation + shape-generalisation failure**
(61% of held-out instances quarantined as `insufficient_conditioning_pairs`
because registered ARC tasks have k=2-3 train pairs, k-1<=2 conditioning
after LODO). All four arms achieve identical shape_exact_slot1 rates
(0.500 pttest, 0.474 test_lodo) -- the per-task scratch shape MLP memorizes
conditioning shapes regardless of arm. The per-task coordinate-MLP color
head does NOT collapse to background like compact-7 did (max collapse rate
16.7% on pttest, 10.5% on test_lodo).

Four Phase 3 binding full-grid-control receipts -- V1, V2, compact-7, and
Phase 3A -- now agree on the held-out exact-grid floor across two task
distributions and two learner families (transformer, per-task coordinate
MLP). The remaining admissible Phase 3 reopen path is
PHASE3_5_REFLECTION Branch D (different framing). A future Branch D would
need its own pre-registered spec, arena gate, and learner contract.

Branch D is filed at
[`PHASE3D_DIFFERENT_FRAMING_SPEC.md`](PHASE3D_DIFFERENT_FRAMING_SPEC.md).
`structured_edit_residual_v1` models each output as an input-derived baseline
canvas plus a residual edit mask and edit colors. The 20-shard binding
receipt (4 arms x 5 seeds, ~1h 26m GPU parallel wall via the (arm, seed)
shard+merge protocol with `--allow-mixed-commits` operator override --
3 distinct gitCommits, but `runnerIdenticalAcrossCommits=true` so no WARN
fired) returned verdict **`branch_d_full_grid_edit_floor`**: zero
non-baseline exact tasks on both held-out lanes for arm `raw_grid_edit`,
so no `signature_palette_edit` vs. `raw_grid_edit` sufficiency comparison
is licensed. Failure character is qualitatively distinct from V1, V2,
compact-7, and Phase 3A: **edit-color-rule failure**. The structured-edit
framing recovers shape (1.000 on pttest), partial palette (0.333), pixel
accuracy 0.47-0.57, and edit-mask F1 0.53-0.65 with strong minority recall
(0.61-0.72) -- but the per-instance scratch color learner cannot recover
the exact edit colors. The dominant quarantine label (26% of held-out
instances) is `edit_color_failure` (mask F1 >= 0.50 but color accuracy
< 0.50): the model knows WHERE to edit but not WHAT color.

Five Phase 3 binding full-grid-control receipts -- V1, V2, compact-7,
Phase 3A, and Phase 3D -- now agree on the held-out exact-grid floor
across two task distributions, two learner families (transformer +
per-task MLP), and two output framings (whole-grid + structured edit
residual). The structured-edit framing additionally decomposed the
failure mode into its components (shape picker, canvas picker, edit-mask
learner, edit-color learner) for the first time, isolating the bottleneck
to the edit-color-rule stage. All four PHASE3_5_REFLECTION branches plus
Branch D are now characterised; remaining admissible Phase 3 reopens
require either a new pre-registered Branch D variant (wider baseline
family, cross-task edit-color MLP, richer mask learner) or a new
Branch E spec (e.g., generative program search, test-time LM prompting)
with its own arena gate and verdict-amendment discipline.

## Official Anchors

Checked **2026-05-28** against:

- [ARC Prize 2026 overview](https://arcprize.org/competitions/2026): competition
  start March 25, 2026; submissions due November 2, 2026; papers due
  November 8, 2026; results announced December 4, 2026.
- [Paper Prize](https://arcprize.org/competitions/2026/paper): paper
  submissions must link to a Kaggle code submission for ARC-AGI-2 or ARC-AGI-3;
  a high score is not required for eligibility, but the score feeds the
  Accuracy rubric.
- [ARC-AGI-2 track](https://arcprize.org/competitions/2026/arc-agi-2): static
  grid reasoning track; Kaggle notebook submission; no internet during
  evaluation; two predicted outputs per test input; exact-match task score.
- [Official ARC-AGI-2 repo](https://github.com/arcprize/ARC-AGI-2): 1,000
  public training tasks and 120 public evaluation tasks, with private sets held
  out for the competition.

## Current Phase Artifacts

- [`PHASE0_TASK_SUBSET_SPEC.md`](PHASE0_TASK_SUBSET_SPEC.md) -- frozen Phase 0
  work order for task inventory, subset registration, baselines, and
  evaluation leak control. Two amendments on file (PARTIAL ADMIT then ADMIT).
- [`P0_BASELINES.md`](P0_BASELINES.md) -- Phase 0 inventory/register/baseline
  receipt with both verdicts (frozen 3-baseline PARTIAL ADMIT, then 6-baseline
  ADMIT after baseline expansion).
- [`P0_TASK_REGISTER.csv`](P0_TASK_REGISTER.csv) -- 36 public-training tasks,
  6 per registered prior, all manually inspected.
- [`EVAL_BLIND_SELECTION.md`](EVAL_BLIND_SELECTION.md) -- stub pattern for
  future Phase 1+ evaluation-blind register rows (no manual grid inspection,
  selection by preregistered metadata/hash rule).
- [`PHASE1_SHADOW_DOMAIN_SPEC.md`](PHASE1_SHADOW_DOMAIN_SPEC.md) -- Phase 1
  discrete grid shadow-domain spec. Two amendments: original 5-fixture
  synthetic gate (later audited as construction-only) and a strengthened
  9-fixture + 50-grid discrimination gate that adds falsifiable cases
  (1-cell flip, color collision, stencil-bag invariance, signature
  discrimination). Phase 2 projection scaffold admitted with falsifiable
  support.
- [`PHASE2_PROJECTION_SPEC.md`](PHASE2_PROJECTION_SPEC.md) -- Phase 2
  registered-subset projection measurement receipt with baseline-comparison
  addendum. `36` tasks / `266` grids projected; shadow operator collapses 6
  byte-distinct grids correctly as gauge equivalents, but its train-pair
  residual (`0.594`) is the highest of all four representations tested
  (raw-pixel Hamming `0.279`, shape/palette/density `0.103`, cell-count
  `0.071`). Phase 3 sufficiency-spec writing admitted with the harder
  framing: signature space does not shorten the input-output gap, so
  Phase 3 must explicitly model the transformation. No decoder or
  public-evaluation work admitted by this receipt.
- [`PHASE3_SUFFICIENCY_SPEC.md`](PHASE3_SUFFICIENCY_SPEC.md) -- Phase 3
  sufficiency audit spec. It freezes the harder post-Phase-2 posture:
  `input_sig -> output_sig` is not a small-delta problem, and the primary
  Sundog representation is `signature_palette`, not signature alone.
  Three binding receipts filed (`nn_output_transfer_v1`,
  `nn_delta_transfer_v1`, `candidate_combinator_v1`), all verdict
  task hardness / decoder failure. A Blackwell sufficiency lane is also
  filed: the full clean `blackwell_task_decoder_v1` receipt is Branch C
  bounded failure with the raw-grid exact-floor caveat. The strengthened
  follow-up `blackwell_publictrain_rawgrid_gate_v2` is filed as a
  raw-grid-only control gate with all 971 admitted public-training aux
  tasks, 3 seeds (`20260528/20260529/20260530`), 120 epochs, GPU
  shard+merge protocol with shard-equivalence verification, freeze-marker
  commit `79C5B060`. Selected seed `20260529` (val_loss 1.6249,
  validation_metric ties at 0.0). V2 binding-receipt verdict:
  **`full_grid_control_floor`** -- `grid_exact_any_rate = 0.0` on every
  held-out lane (`pttest`, `test_lodo`, `validation_lodo`,
  `validation_pttest`); `shape_exact_slot1` 0.50-0.72 while
  `palette_exact_slot1 = 0.0`, the same shape-matches-content-fails
  character as V1. A Branch B compact-subset diagnostic lane is then
  admitted and the compact-7 single-seed receipt is filed (see next
  bullet); Phase 3 then pauses with Branch B closed.
- [`PHASE3B_COMPACT_SUBSET.md`](PHASE3B_COMPACT_SUBSET.md) +
  [`PHASE3B_COMPACT_SPLIT.csv`](PHASE3B_COMPACT_SPLIT.csv) -- Phase 3
  Branch B compact-subset diagnostic lane. 7 Phase 2 compact-signal
  tasks, pre-registered 4/1/2 internal split, minimal floor (≥1 exact
  task on `pttest` AND `test_lodo`), decoder/hyperparams/seed slate/aux
  pool frozen relative to V2. Compact-7 single-seed binding receipt
  filed under freeze-marker `50EAEBBF` with seed `20260529` (best_epoch
  55, val_loss 0.7628). Verdict: **`compact_full_grid_control_floor`**.
  Failure character is qualitatively distinct from V1/V2: every
  held-out instance has `shape_exact_slot1 = 1.000` and pixel accuracy
  0.757-0.877, but every prediction collapses to the dominant
  background color (13 of 13 held-out predictions slot-1-use at most 2
  colors regardless of target palette size). Named failure mode filed:
  **"dominant-color mode collapse"**. Per the pre-registered §8 stop
  rule, Branch B closes in the deterministic-low-capacity-learner
  family; no additional seeds, no `signature_palette` arm on the
  compact subset, no further raw-grid V-bumps within the family.
- [`PHASE3A_STOCHASTIC_PER_TASK_SPEC.md`](PHASE3A_STOCHASTIC_PER_TASK_SPEC.md)
  -- Branch A stochastic per-task learner spec. It reserves
  `per_task_coord_mlp_v1`, freezes the `raw_grid_per_task`,
  `signature_palette_per_task`, `signature_only_per_task`, and
  `metadata_only_per_task` arms, requires the raw-grid arm to open the arena
  before any signature sufficiency comparison, and adds dominant-color collapse
  audits. **Binding 20-shard receipt filed**: `branch_a_full_grid_floor` at
  selected seeds (raw=`20260530`, sig_palette=`20260528`, sig_only=`20260530`,
  metadata=`20260601`). Zero exact tasks on both held-out lanes for any arm;
  all four arms achieve identical shape_exact_slot1 rates (0.500/0.474).
  Failure mode: **conditioning starvation + shape-generalisation failure**
  (61% of held-out instances quarantined as `insufficient_conditioning_pairs`).
  Run executed via the (arm, seed) shard+merge protocol with the operator
  `--allow-mixed-commits` override (parallel Navier-Stokes commits landed
  mid-launch; runner-content audit verified shard-time computational contract
  unchanged across 6 distinct gitCommits / 2 distinct runner SHAs;
  `mixedCommitsAudit` recorded in the merged manifest). No
  signature_palette vs. raw_grid sufficiency comparison licensed.
- [`PHASE3D_DIFFERENT_FRAMING_SPEC.md`](PHASE3D_DIFFERENT_FRAMING_SPEC.md)
  -- Branch D structured edit/residual framing spec. It freezes
  `structured_edit_residual_v1`, with `raw_grid_edit`,
  `signature_palette_edit`, `signature_only_edit`, and `metadata_only_edit`
  arms, a conditioning-selected baseline family, edit-mask/edit-color residual
  learners, and a non-baseline arena gate. **20-shard binding receipt
  filed** under `mergeGitCommit 9F3193D7` (3 distinct gitCommits via
  `--allow-mixed-commits` override; `runnerIdenticalAcrossCommits=true` so
  no WARN; 14 dirty + 6 clean shards). Selected seeds: raw=`20260529`,
  sig_palette=`20260531`, sig_only=`20260529`, metadata=`20260528`.
  Verdict: **`branch_d_full_grid_edit_floor`**. Zero non-baseline
  exact tasks on both held-out lanes for every arm; baseline_exact also
  zero (the baselines never produce the exact target). Shape exact 1.000
  on pttest, 0.316 on test_lodo; palette exact 0.333; pixel mean
  0.47-0.57; edit-mask F1 0.53-0.65; minority edit recall 0.61-0.72.
  6 of 9 quarantine labels fire; dominant label is **`edit_color_failure`**
  (51/196 = 26%), naming the new failure mode: the per-instance scratch
  color learner knows WHERE to edit but not WHAT color. Wall: ~1h 26m
  parallel (2.9x sharded GPU speedup vs ~5.3 h serial projection).
- [`PHASE3_5_REFLECTION.md`](PHASE3_5_REFLECTION.md) -- reflection doc
  naming the three-receipt convergence under the
  deterministic-low-capacity-learner family as a methodological finding,
  scoping what kind of learner would produce a non-trivial sufficiency
  verdict, and surveying four Phase 4 branch options. Admits no
  execution and changes no verdict. The
  Pass A representation/decoder contract, Pass B split/floor/discrimination
  contract, Pass C learner/metric contract, and Pass D receipt/command contract
  are filed. The freeze-marker runner wiring is implemented; the first receipt
  is admitted only under the frozen Pass A-D contracts.

## Discipline Tooling

The following enforces public-evaluation and Kaggle discipline rather than
just documenting it:

- `npm run arc:phase0:leak-check` -- audits inventory manifest, register
  splits, predictions ⊆ register, no Kaggle scaffolding, no non-inventory ARC
  script with `evaluation` literals. Exits nonzero on any FAIL.
- `.githooks/pre-commit` -- runs the leak check before every commit. Installed
  automatically by the `prepare` npm script (sets `core.hooksPath=.githooks`).
- `.github/workflows/arc-discipline.yml` -- runs the leak check on every push
  or PR that touches `docs/prereg/arc/**`, `scripts/arc-*`,
  `tests/arc-baselines/**`, or related infrastructure.
- `scripts/arc-phase0-inventory.mjs` double-flag override -- emitting
  evaluation test outputs requires `--include-evaluation-test-output` *and*
  `--authorize-evaluation-leak`, and `--out` must end in `_PRIVILEGED_AUDIT`.
  Single-flag-by-mistake leaks are blocked at runtime.

## Append-Only Rule

Once a phase spec is used to admit or block a run, the body above its
amendments line is frozen. Corrections or refinements must be appended with:

- date and timezone;
- author;
- one-line justification;
- explicit statement of whether the prior verdict changes.

## Public-Language Constraint

Phase 3 has **five** full-grid-control binding receipts on file (V1, V2,
compact-7, Phase 3A `per_task_coord_mlp_v1`, and Phase 3D
`structured_edit_residual_v1`), all floored at zero exact matches; no
sufficiency support adjudication is on file. Public copy may say:

- ARC-AGI abstraction coupling roadmap;
- registered task-subset audit;
- shadow-projection hypothesis for static grid reasoning;
- falsifiable sufficiency test, filed; the deterministic-low-capacity
  learner branch produced three task-hardness verdicts, the
  `blackwell_task_decoder_v1` lane produced a Branch C bounded-failure
  receipt with the raw-grid exact-floor caveat, the strengthened
  `blackwell_publictrain_rawgrid_gate_v2` lane produced a
  `full_grid_control_floor` receipt on all four held-out lanes, the
  Branch B compact-subset diagnostic lane (compact-7) produced a
  `compact_full_grid_control_floor` receipt with a qualitatively
  distinct failure -- dominant-color mode collapse -- on every
  held-out instance, the Branch A stochastic per-task lane
  (`per_task_coord_mlp_v1`) produced a `branch_a_full_grid_floor`
  receipt with a third qualitatively distinct failure --
  conditioning starvation + shape-generalisation failure -- and the
  Branch D structured-edit-residual lane
  (`structured_edit_residual_v1`) produced a
  `branch_d_full_grid_edit_floor` receipt with a fourth qualitatively
  distinct failure -- edit-color-rule failure. All five full-grid
  controls now agree on the floor across two task distributions, two
  learner families (transformer + per-task MLP), and two output framings
  (whole-grid + structured edit residual). All four
  PHASE3_5_REFLECTION branches plus Branch D are now characterised;
  remaining admissible Phase 3 reopens require either a new
  pre-registered Branch D variant (wider baseline family, cross-task
  edit-color MLP, richer mask learner) or a new Branch E spec (e.g.,
  generative program search, test-time LM prompting) with its own
  arena gate and verdict-amendment discipline.

Avoid:

- "Sundog solves ARC";
- "human-level abstraction";
- "the 5D subspace is universal";
- any claim that a Kaggle entry validates the theory without a
  non-trivial Phase 3 sufficiency receipt;
- any Branch A, Branch B, or Branch D support claim from any of
  the five filed binding receipts (V1, V2, compact-7, Phase 3A,
  Phase 3D);
- describing any Phase 3 binding receipt as a signature-specific
  falsification independent of decoder capacity;
- describing the five filed Phase 3 receipts as a sufficiency-failure
  conclusion -- per the reflection, they characterise the
  deterministic-low-capacity, per-task-scratch, and structured-edit
  framing families on the registered task class, not the
  shadow-projection representation;
- claiming "V2 failed -> signature representation is favoured" -- V2
  floored the same way V1 did; no signature-vs-full-grid comparison
  is licensed by either receipt;
- claiming "compact-7 failed -> compact tasks are unsolvable" -- the
  failure is of this learner family on this slice; a different
  framing (Branch D) might or might not pass on the same slice, and
  that is the next question, not a settled one;
- claiming "Phase 3A floored -> signature representation is favoured"
  -- the Phase 3A raw-grid arm did not open the arena, so no per-arm
  comparison is licensed; the all-arms-equal pattern in the receipt
  is a consequence of the floor, not evidence of per-arm equivalence;
- claiming "Phase 3D floored -> signature representation is favoured"
  -- same caveat: the raw_grid_edit arm did not open the
  non-baseline arena; `signature_palette_edit`'s slightly higher
  pixel / minority recall in the documentation-only per-arm table
  is unlicensed-for-adjudication data, not a support signal;
- claiming any single Phase 3 receipt closes the sufficiency
  question -- the five floors together close PHASE3_5_REFLECTION
  Branches A, B, C, and D in the filed learner families and
  framings, but the sufficiency question remains open under a
  Branch D variant (wider baseline / cross-task color / richer
  mask) or a Branch E (different framing entirely);
- any spatial_transform or local_completion claim from the compact-7
  receipt -- those priors are not represented in the compact-signal
  slice;
- claiming "Phase 3D's baseline-exact = 0 means the input copy
  baseline doesn't work" -- the baseline picker correctly identifies
  same_as_input + identity_top_left as the dominant pick (28/49 in
  smoke) but the target outputs of registered tasks genuinely differ
  from the baselines after the editing operation; `baseline_exact = 0`
  means "no task in the registered subset is solved by baseline
  alone", which is the expected design property of ARC tasks.
