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

**2026-05-28 (PT) -- Claude (Opus 4.7).** Synthetic-gate strengthening. A
post-pass audit of the original 5-fixture battery found that all four
positive cases passed *tautologically*: the operator's canonical signature
is invariant by construction (sort+normalize for translation,
D4-min-search for rotation/reflection, role-mapping for color permutation),
and each positive test grid is an exact gauge-orbit-mate of the base grid.
The original receipt confirmed that the gauge-quotienting code *runs*; it
did not confirm any non-trivial gauge-discrimination behavior. The
`localSignatureBag` field was also computed but never asserted on, and the
suite had no discrimination check.

Four fixtures and one aggregate check were added (v0 fixtures unchanged):

- `single_cell_flip_negative` -- baseGrid + one new cell at (x=4, y=1)
  color 2. Expect signature change and non-zero residual.
- `color_collision_negative` -- baseGrid with both non-zero colors mapped
  to a single new color (non-bijective recoloring). Expect signature change
  (role count drops 2 -> 1) and non-zero residual.
- `stencil_bag_translation_positive` -- assert
  `localSignatureBag` is byte-identical between baseGrid and its
  translated-into-canvas copy.
- `stencil_bag_rotation_positive` -- assert `localSignatureBag` is
  byte-identical between baseGrid and its 90-degree rotation.
- `discrimination_50_random_5x5` -- 50 deterministically-seeded random
  5x5 grids (seed `20260528`, palette `1..5`, density `0.5`) must produce
  50 distinct canonical signatures. Catches over-aggressive signature
  collapse.

Re-run results: all 9 fixtures PASS, discrimination 50/50 distinct,
residuals scale with edit magnitude (1-cell flip `0.166667` <
shape-mismatch `0.428571` < color-collision `0.571429`). Artifacts:
`results/arc/phase1-shadow-domain-synthetic/manifest.json`
(`sha256=85d63d8a6839475e63244436bfd1c39b1ccc457d498bc43ca8e7ac9a4b226557`)
and `results/arc/phase1-shadow-domain-synthetic/summary.csv`
(`sha256=990bf245567cdfac5ea093b4cf1c4f1310dc67b2c6f128403266a1c590fd15d5`).

Open issues named but not resolved by this amendment (carried to Phase 3):
the canonical signature quotients out *all* color identity, so any ARC
sufficiency claim that depends on absolute colors must read the raw
`palette` metadata alongside the signature. The Phase 3 sufficiency audit
must explicitly treat (signature, palette) as the working representation,
not signature alone.

Verdict impact: **synthetic validation PASS sustained** with falsifiable
support. Phase 2 admission previously recorded by Codex stands; this
amendment substantiates rather than overturns it.
