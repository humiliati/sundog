# Ghost Phase 3 - Aperiodic Reader Spec

- Artifact id: `GHOST-PHASE3-APERIODIC-READER-SPEC`
- Date: 2026-06-27
- Status: **Built + QA passed 2026-06-27** (acceptance `pass=13 fail=0`; browser
  QA desktop + mobile). Exit gate met, pending owner sign-off to flip COMPLETE.
  Internal-only artifact; no public route.
- Ledger: [`../SUNDOG_V_GHOST.md`](../SUNDOG_V_GHOST.md)
- Lit-pass: [`../GHOST_LITPASS_MEMO.md`](../GHOST_LITPASS_MEMO.md)
- Phase 2: [`PHASE2_TOY_WORKBENCH_SPEC.md`](PHASE2_TOY_WORKBENCH_SPEC.md)

> Ordered does not mean repeating. Local does not mean closed.

## 1. Purpose

Phase 3 adds a reader mode for ONE real, planar aperiodic substrate, stepping up
from the 1-D Fibonacci stripe of Phase 2. The substrate is the **rhombic Penrose
tiling (P3)**. The reader must make the **inflation hierarchy visible as
hierarchy** and let a viewer see the Phase-2 lesson in two dimensions:

> A finite patch is lawful and locally concrete, yet its full explanation lives
> in a larger supertile hierarchy the patch does not contain. For Penrose that
> hierarchy is **recoverable at a finite radius** (the tiling is recognizable),
> so the ghost here is bounded, not unerasable.

This is a reader/exhibit surface, not a theorem probe. It does not introduce a
"Ghost invariant" and does not claim new tiling mathematics.

## 2. Substrate Lock - Penrose P3 (rhombic)

Two prototiles, unit edge length:

- **thick (fat) rhombus** - angles 72 deg / 108 deg;
- **thin (skinny) rhombus** - angles 36 deg / 144 deg.

Generation is by **Robinson-triangle deflation**, the standard route. Each
rhombus is two mirror Robinson triangles; deflation subdivides triangles by the
golden ratio phi = (1 + sqrt 5) / 2.

**Verified subdivision rule** (cross-checked against a canonical reference
implementation; combinatorics re-derived from the rule itself, not taken from a
secondary source):

- type RED = half of the THIN rhombus (acute, 36 deg apex);
- type BLUE = half of the THICK rhombus (obtuse, 108 deg).

For a RED triangle with ordered vertices (A, B, C):

```text
P = A + (B - A) / phi
children: RED (C, P, B), BLUE (P, C, A)
```

For a BLUE triangle with ordered vertices (A, B, C):

```text
Q = B + (A - B) / phi
R = B + (C - B) / phi
children: BLUE (R, C, A), BLUE (Q, R, B), RED (R, Q, A)
```

Seed: a decagon "sun" of 10 RED triangles around the origin, alternating
mirror orientation.

**Re-derived combinatorics (used as ground truth for tests):**

- RED -> {1 RED, 1 BLUE}; BLUE -> {1 RED, 2 BLUE}.
- count matrix `[[1, 1], [1, 2]]`, Perron-Frobenius eigenvalue
  `phi^2 = (3 + sqrt 5)/2 ~ 2.618` => total tile count grows by `phi^2` per
  inflation step (area scaling);
- the linear inflation factor is `phi`;
- RED:BLUE -> phi, hence thick:thin rhombus ratio -> phi.

Note the distinction the lane cares about: the **linear** inflation factor is
phi, while the **count/area** growth eigenvalue is phi^2. Do not state
"PF eigenvalue = phi" for the triangle count substitution; that is the linear
factor, not the count eigenvalue.

Citations: Penrose (1974) / Gardner (1977) for P3; substitution / recognizability
backbone via Baake-Grimm, *Aperiodic Order* Vol. 1, and the Bielefeld Tilings
Encyclopedia; recognizability framing per this lane's Q2 Resolution
(`GHOST_LITPASS_MEMO.md`). The reference implementation cross-check (Preshing,
"Penrose Tiling Explained", 2011) is an implementation aid, not a math source.

## 3. Recognizability Framing (the Ghost hook)

Penrose tilings have the **unique composition property** (they are
recognizable): the inflation/supertile structure is uniquely and locally
recoverable. The reader makes this concrete:

- choose a finite circular window (the "circle");
- group the finest tiles by their **supertile at ancestry level k** (k inflation
  steps up);
- a supertile fully inside the window is **recoverable from inside** - the ghost
  has collapsed for it;
- a supertile crossing the window edge is **ghost at the boundary** - its
  ancestry is not yet determined from inside the window;
- raising k moves the explanation to a larger supertile (the ghost recedes
  upward); shrinking k or growing the window resolves more ancestry.

This is the recognizability-radius idea (Q2 Resolution) made visual: the outside
debt is a finite radius, not an unerasable outside.

## 4. Domain Lock / Out of Scope

- No Hat/Spectre geometry or assets (still deferred).
- No reuse of any published Penrose figure or asset; geometry is generated from
  the rule in section 2.
- No undecidability / SFT simulation; the computability cliff stays a named
  boundary, as in Phase 2.
- No theorem-shaped "Ghost Boundary" claim; no claim of new tiling math.

## 5. Artifacts

- Pure core: `ghost/aperiodic-core.js` (no DOM) - generation, ancestry,
  window/recognizability analysis, reader export.
- Reader UI: `ghost/aperiodic.html` - internal, noindex, served via
  `npm run ghost:serve` at `/ghost/aperiodic.html`. No public route, no
  `site-pages.json` entry, no SEO/social metadata.
- Acceptance tests: `scripts/ghost-aperiodic-tests.mjs`, wired as
  `npm run ghost:aperiodic:test`.

## 6. Interaction Contract

- inflation-depth control (how many deflation steps to generate);
- ancestry-level control k (supertile size to highlight);
- circular window center + radius controls;
- SVG render: tiles colored by rhombus type, supertile membership shown by
  grouping/outline at level k, window drawn, boundary-crossing supertiles
  emphasized as the ghost boundary;
- metrics panel: tile counts, thick:thin ratio, tiles in window, supertiles
  in window, supertiles fully contained vs crossing;
- regime panel: bounded recognition (recoverable ancestry) vs the named
  computability cliff (not measured here).

## 7. Acceptance Tests

The pure core must pass:

1. seed wheel has 10 RED triangles, apex at origin, far vertices on the unit
   circle;
2. RED subdivides to {1 RED, 1 BLUE}; BLUE to {1 RED, 2 BLUE};
3. total tile count grows by `phi^2` per inflation step (within tolerance) for
   depths up to 6;
4. RED:BLUE ratio -> phi (within tolerance) at depth >= 6;
5. thick:thin rhombus ratio -> phi (within tolerance);
6. every finest tile has ancestry-path length == depth + 1 (one seed index plus
   one entry per inflation step), and supertile grouping at any level k
   partitions all finest tiles (disjoint, exhaustive);
7. window analysis reports tiles-in-window and contained-vs-crossing supertiles
   consistently (contained + crossing == supertiles touching the window);
8. reader export contains no theorem/proof/invariant/conjecture/claim field.

Run:

```text
npm run ghost:aperiodic:test
```

## 8. Exit Gate

Phase 3 is complete when:

- the acceptance suite passes;
- the reader opens at `/ghost/aperiodic.html` via `npm run ghost:serve`;
- the inflation hierarchy is visible as an interaction (supertiles nest as k
  rises), and the window shows bounded recoverable ancestry vs boundary ghost;
- a specialist would not object to the phrasing "this visualizes one aspect of
  the known proof structure (the inflation hierarchy and its recognizability)";
- `SUNDOG_V_GHOST.md` Phase 3 points to this spec and the artifact.

## 9. Public-Page Caveat (Penrose name / assets)

Penrose tilings carry a trademark/patent history (patents now expired). The
generated geometry from published substitution rules is clean for an internal
reader. Any future PUBLIC page must (a) add a name/attribution note, (b) avoid
reproducing copyrighted figures, and (c) get an asset/license check before
launch, mirroring the Hat/Spectre asset discipline.
