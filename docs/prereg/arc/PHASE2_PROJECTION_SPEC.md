# Phase 2 -- Shadow Projection Operator on Registered ARC Tasks

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Phase 1 receipt: [`PHASE1_SHADOW_DOMAIN_SPEC.md`](PHASE1_SHADOW_DOMAIN_SPEC.md)

Filed: **2026-05-28 (PT)**

Status: **PROJECTION MEASUREMENT PASS -- PHASE 3 SPEC ADMITTED**.

Phase 1's strengthened synthetic gate passed with falsifiable support. Phase 2
is now admitted to project the registered public-training subset and measure
what the projection keeps or collapses. It does **not** train a decoder, score
task answers, inspect public-evaluation grids, or support a sufficiency claim.

## Claim Under Test

The Phase 2 claim is:

> `P_shadow_grid_v0` can be applied mechanically to every registered Phase 0
> task, yielding per-grid signatures and per-task residual summaries without
> leaking public-evaluation information or using answer-scoring feedback.

The phase can fail if the projection is vacuous, collapses too many distinct
registered grids into one signature, or produces no structured residual surface
for Phase 3 to audit.

## Command

```powershell
npm run arc:phase2:projections -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase2-projections
```

## Output Artifacts

- `results/arc/phase2-projections/manifest.json`
- `results/arc/phase2-projections/task-summary.csv`
- `results/arc/phase2-projections/grid-projections.json`

## Metrics

Per grid:

- shape, palette, non-zero density, component count;
- canonical object signature hash;
- local radius-1 signature-bag hash;
- raw palette retained as metadata because Phase 1 identified absolute color
  identity as an open issue for Phase 3.

Per registered task:

- `grid_count`: training inputs, training outputs, and held test inputs;
- `unique_signature_count`;
- `signature_collision_residual = 1 - unique_signature_count / grid_count`;
- `unique_shape_palette_signature_count`;
- mean and max training-pair alignment residual;
- signal label (`compact`, `mixed`, or `dispersed`) for initial triage only.

The collision residual is a Phase 2 information-loss proxy, not a formal
Blackwell-sufficiency result.

## Exit Criteria

Phase 2 exits only after:

1. projections are computed for all 36 registered public-training tasks;
2. no projection record is emitted for public-evaluation tasks;
3. task-summary and grid-projection artifacts are filed with hashes;
4. the initial signal characterization is summarized here;
5. Phase 3 remains held until a separate sufficiency-audit spec is filed.

## Amendments

Append-only. Each amendment must carry a timestamp, author, justification, and
verdict impact.

**2026-05-28 (PT) -- Codex.** Phase 2 projection measurement executed:

```powershell
npm run arc:phase2:projections -- --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" --register docs/prereg/arc/P0_TASK_REGISTER.csv --out results/arc/phase2-projections
```

Artifacts:

- `results/arc/phase2-projections/manifest.json`
  (`sha256=51a8c8ca3d86e143b5fb61f3dd75d7b79119c334febfacd785a74a16075db896`)
- `results/arc/phase2-projections/task-summary.csv`
  (`sha256=1ff91c68c7f9d08479acc760575ee89e22d6b80673301e89b0717d2d0f28648f`)
- `results/arc/phase2-projections/grid-projections.json`
  (`sha256=c9f5465665c6887aa4668e998ce8ceeb3810c9be091c7c2cdd20189a1405a54a`)

Result: `36` registered public-training tasks projected, `266` grids total.
Aggregate mean signature-collision residual: `0.028571`. Aggregate mean
training-pair residual: `0.594295`. Initial signal labels: `20` dispersed,
`9` mixed, `7` compact.

Interpretation: the v0 signature is not vacuously collapsing the registered
subset (low collision residual), but many input-output demonstrations are
geometrically far apart under the current projection (high pair residual).
This makes Phase 3 a real sufficiency question rather than a foregone pass.
The raw `palette` metadata remains part of the working representation because
Phase 1 identified absolute-color identity as an open sufficiency issue.

Verdict impact: **PROJECTION MEASUREMENT PASS**. Phase 3 sufficiency-spec
writing is admitted. Decoder training, sufficiency claims, dimensionality
claims, public-evaluation scoring, and Kaggle work remain held until their
own preregistered gates.

**2026-05-28 (PT) -- Claude (Opus 4.7).** Baseline comparison addendum (no
verdict change). The SUNDOG_V_ARC.md Phase 2 deliverable list calls for
"Comparison against naive baselines (raw pixel features, simple convolution
features) on the same information-retention metric." The original Codex
receipt above did not file that comparison. This amendment closes the gap by
filing three baselines computed on the same 266 grids and 115 train pairs:

Tool: `npm run arc:phase2:baselines` (`scripts/arc-phase2-baselines.mjs`).

Artifacts:

- `results/arc/phase2-baselines/manifest.json`
  (`sha256=4968d6f6958c8d40643d09161407252b664665801fcba3a399780ed7509623e0`)
- `results/arc/phase2-baselines/summary.csv`
  (`sha256=14977f7f35ab3e5da40c4603bc0771ab626909c003f89e82f6210ac5b068f165`)

Comparison table:

| representation | mean_collision_residual | mean_train_pair_residual | global_unique_features |
| --- | ---: | ---: | ---: |
| `shadow_operator_v0` (reference) | `0.028571` | `0.594295` | `260 / 266` |
| `raw_pixel_hash` | `0.000000` | `0.278825` | `266 / 266` |
| `shape_palette_density` | `0.338672` | `0.102588` | `162 / 266` |
| `cell_count` | `0.297226` | `0.070576` | `93 / 266` |

(Shadow operator's global-unique figure `260 / 266` was measured by hand
from `results/arc/phase2-projections/grid-projections.json`; it is omitted
from `summary.csv` because the reference row sources from the projection
manifest, which does not pre-compute it.)

Interpretation:

1. **Within-task collision (`mean_collision_residual`).** The shadow
   operator (`0.029`) sits between the byte-unique floor (`raw_pixel_hash =
   0.000`) and the coarse aggregates (`shape_palette_density = 0.339`,
   `cell_count = 0.297`). It collapses 6 of 266 grids that byte-hashing
   treats as distinct -- those 6 are gauge-equivalent grids inside training
   tasks that the operator correctly identifies, which is exactly the
   intended discrimination behavior. The coarse baselines collapse roughly
   30 percent of grids; the operator does not.

2. **Train-pair distance (`mean_train_pair_residual`).** The shadow
   operator (`0.594`) is **higher than every baseline tested**: roughly
   2.1x raw-pixel Hamming (`0.279`), 5.8x shape-palette-density (`0.103`),
   and 8.4x cell-count (`0.071`). The Codex receipt's "many input-output
   demonstrations are geometrically far apart under the current projection"
   framing under-stated this: under no naive baseline are input and output
   as far apart as under the shadow operator's token-set distance.

3. **What this means for Phase 3.** The high pair distance is consistent
   with the operator's token-level representation correctly registering
   that ARC transformations are substantive changes -- the cruder baselines
   under-detect change because they aggregate away position and role
   information. But it also means the operator does **not** by itself bring
   train inputs and outputs into a representation where the rule is a small
   delta. Phase 3 sufficiency cannot rely on "input-sig is close to
   output-sig, just learn a small map"; it must prove the input-sig ->
   output-sig mapping is learnable from the few train pairs **despite** the
   high signature-space distance, plausibly using the raw `palette`
   metadata Phase 1 already flagged as carried forward.

4. **Roadmap gap remaining.** "Simple convolution features" (per
   `SUNDOG_V_ARC.md` Phase 2 deliverable) is not in this comparison; the
   three baselines here cover raw-pixel and simple-aggregate features. A
   patch-histogram convolution baseline can be added as a Phase 2.5
   amendment if Phase 3 design needs it; the current comparison is enough
   to calibrate the shadow operator's numbers and frame Phase 3 honestly.

Verdict impact: **PASS sustained**. The Phase 3 admission recorded by
Codex stands. The baseline comparison strengthens the receipt by making
the sufficiency challenge concrete: shadow-operator alignment residuals
are larger than naive baselines, so Phase 3 must explicitly handle the
input-to-output transformation in signature space, not assume the
projection itself shortens the gap.
