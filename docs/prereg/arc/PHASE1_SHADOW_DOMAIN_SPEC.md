# Phase 1 -- ARC Grid Shadow Domain Specification

Roadmap: [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md)

Phase 0 receipt: [`P0_BASELINES.md`](P0_BASELINES.md)

Filed: **2026-05-28 (PT)**

Status: **SYNTHETIC VALIDATION PASS -- PHASE 2 PROJECTION SCAFFOLD ADMITTED**.

Phase 0 admitted Phase 1 after the baseline expansion established a hard
`0/36` exact floor on the registered ARC training subset. Phase 1 now defines
the discrete grid shadow domain and validates its gauge behavior on synthetic
fixtures before any registered ARC task performance is scored.

## Claim Under Test

The Phase 1 claim is not yet that the Sundog signature solves ARC tasks. The
claim is narrower:

> ARC grids can be represented as finite structured fields, and a local
> shadow projection can produce signatures whose comparison behavior is stable
> under the grid gauges Sundog intends to quotient: translation, rotation,
> reflection, and color-role permutation.

If this synthetic gauge behavior fails, the ARC lane stops before Phase 2.

## Formal Correspondence

| ARC object | Shadow-domain correspondence |
| --- | --- |
| grid cell `(x, y)` | point on a finite rectangular lattice |
| integer color `0..9` | fiber value / channel label at that lattice point |
| background color `0` | ordinary empty field value, not discarded by default |
| non-zero connected component | object candidate under 4-connectivity |
| local neighborhood | radius-1 stencil around a lattice point |
| task demonstration pair | two fields plus an input-output relation to be aligned later |

The first implementation may use a radius-1 stencil only. Larger radii are a
Phase 2+ amendment unless synthetic Phase 1 fixtures show radius-1 is
degenerate.

## Projection Operator v0

`P_shadow^grid(v0)` computes, for each grid:

1. full-grid metadata: shape, palette, non-zero component count, density;
2. a bag of local cell signatures derived from radius-1 stencils;
3. a canonical object signature over non-zero cells, normalized across the
   dihedral grid symmetries and color-role relabelings;
4. an alignment residual between two grids under discrete dihedral transforms
   plus translation.

Color labels are treated as role labels when testing color gauge behavior.
The operator may preserve the raw palette in metadata, but the canonical
signature must be invariant to bijective color permutation of non-zero colors.

## Synthetic Validation Battery

The first validator is:

```powershell
npm run arc:phase1:synthetic
```

It writes:

- `results/arc/phase1-shadow-domain-synthetic/manifest.json`
- `results/arc/phase1-shadow-domain-synthetic/summary.csv`

Required fixtures:

| fixture | expected result |
| --- | --- |
| translated object | canonical signature unchanged; alignment residual `0` |
| rotated object | canonical signature unchanged; alignment residual `0` |
| reflected object | canonical signature unchanged; alignment residual `0` |
| color-role permutation | canonical signature unchanged |
| shape mismatch negative | canonical signature differs or residual is non-zero |

## Exit Criteria

Phase 1 exits only after:

1. the synthetic validator passes all required fixtures;
2. the operator definition above is updated if implementation details differ;
3. the validation receipt is appended below with artifact hashes;
4. no registered ARC task exact-match score has been produced by the Phase 1
   operator before the synthetic receipt.

Until then, Phase 2 projection over the registered ARC subset is held.

## Amendments

Append-only. Each amendment must carry a timestamp, author, justification, and
verdict impact.

**2026-05-28 (PT) -- Codex.** Phase 1 synthetic validator implemented and run:
`npm run arc:phase1:synthetic` (`scripts/arc-phase1-shadow-domain.mjs`).
Output artifacts:
`results/arc/phase1-shadow-domain-synthetic/manifest.json`
(`sha256=3c57d37b756770a63a8a90474dcf72d817acd0be5a2271408f482c87f73dccca`)
and `results/arc/phase1-shadow-domain-synthetic/summary.csv`
(`sha256=17bf944b1140747fbb7b4db7ba6447c5e4e1aa20bf1abb8a4917185b0648170b`).
Fixture results: translated object PASS (`residual=0`), rotated object PASS
(`residual=0`), reflected object PASS (`residual=0`), color-role permutation
PASS (`residual=0`), shape-mismatch negative PASS (`residual=0.428571` and
canonical signature differs). Verdict impact: **synthetic validation PASS**.
Phase 2 projection scaffold is admitted for the registered training subset;
no sufficiency, dimensionality-collapse, public-evaluation, or Kaggle claim is
admitted by this receipt.
