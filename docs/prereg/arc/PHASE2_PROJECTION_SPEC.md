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
