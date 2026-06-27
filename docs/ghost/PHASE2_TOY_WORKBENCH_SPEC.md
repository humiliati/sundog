# Ghost Phase 2 - Toy Closure Workbench Spec

- Artifact id: `GHOST-PHASE2-TOY-CLOSURE-WORKBENCH-SPEC`
- Date: 2026-06-22
- Status: **Phase 2 COMPLETE (2026-06-27).** Acceptance suite passes
  (`pass=13 fail=0`) and browser QA confirmed render on desktop + mobile.
  Internal-only artifact; no public route.
- Ledger: [`../SUNDOG_V_GHOST.md`](../SUNDOG_V_GHOST.md)
- Lit-pass: [`../GHOST_LITPASS_MEMO.md`](../GHOST_LITPASS_MEMO.md)

> Ordered does not mean repeating. Local does not mean closed.

## 1. Purpose

This spec starts the Phase 2 toy workbench for Ghost. The workbench is a
reader/exhibit surface, not a theorem probe. It should make three distinctions
visible:

1. **Periodic closure:** a finite repeat cell can explain the whole pattern
   inside a declared periodic family.
2. **Substitution ancestry:** a non-periodic substitution stripe has lawful
   local structure and recognizable ancestry, but no translational repeat cell.
3. **Computability cliff:** general Wang/SFT extension is not the next rung of
   the same ladder; it is a separate regime where extension/global-existence
   can become undecidable.

The workbench must not imply that "outside debt" is a new invariant or a smooth
scalar metric.

## 2. Domain Lock

Primary substrate:

- One-dimensional symbolic stripes rendered as finite windows.
- Periodic control: motif `A B C D` repeated.
- Substitution control: Fibonacci substitution `A -> AB`, `B -> A`.
- Displayed sample: finite generated words only.

Out of scope:

- Hat/Spectre geometry or assets;
- Penrose, Wang, or SFT generation;
- undecidability demonstrations;
- any theorem-shaped "Ghost Boundary" claim;
- any claim that the Fibonacci stripe proves a statement about planar
  aperiodic tilings.

Why one-dimensional first:

- the periodic/substitution contrast is visible without asset licensing risk;
- the pure logic is small enough to test;
- the computability cliff can be named as a boundary without pretending to
  simulate it.

## 3. Registered Systems

### 3.1 Periodic Control

System id: `periodic4`

Definition:

```text
P[i] = motif[i mod 4], motif = A B C D
```

Reader claim:

> Once the repeat cell is captured inside the selected window, the outside
> collapses to a bounded repeat rule.

Displayed observables:

- selected window length;
- symbol counts;
- local period candidates up to the configured cap;
- repeat-cell verdict.

### 3.2 Fibonacci Substitution Control

System id: `fibonacci`

Definition:

```text
A -> AB
B -> A
```

Reader claim:

> The selected window has lawful substitution ancestry, but the stripe is not
> explained by a finite translational repeat cell.

Displayed observables:

- selected window length;
- symbol counts;
- local period candidates, marked as finite-window artifacts;
- selected ancestry level;
- number of supertile blocks intersecting the window at that level;
- number of internal block boundaries at that level.

The word "ancestry" is a reader label for substitution block structure. It is
not a new formal object.

### 3.3 Computability Cliff

System id: none in Phase 2.

Displayed only as a boundary note:

> General Wang/SFT extension can encode computation and become undecidable. This
> workbench does not measure that regime.

The cliff must not appear as another row in a smooth metric ladder.

## 4. Interaction Contract

The internal workbench should provide:

- a system selector;
- a center control for the selected finite window;
- a radius control for the selected finite window;
- an ancestry-level selector for the Fibonacci system;
- a stripe display with the selected window and boundary edges visible;
- a compact metrics panel;
- a regime panel that separates bounded recognition from the computability
  cliff.

No public navigation, route registration, SEO metadata, or site launch wiring.

## 5. Visual Contract

The workbench should show:

- non-selected context cells muted;
- selected window cells emphasized;
- left/right boundary edges of the selected window as the "ghost boundary";
- substitution block boundaries for the chosen ancestry level when the
  Fibonacci system is active;
- no Hat/Spectre silhouettes or aperiodic monotile imagery.

The exhibit should still make sense with all conjecture/heuristic language
removed.

## 6. Acceptance Tests

The pure core must pass:

1. periodic `ABCD` sample has global period 4;
2. a sufficiently large periodic window admits period 4 and no smaller period;
3. Fibonacci generation obeys the length recurrence;
4. Fibonacci generation contains both symbols for level >= 2;
5. a large Fibonacci prefix has no small translational period under the test cap;
6. Fibonacci supertile intervals partition the generated word at each level;
7. a selected Fibonacci window reports intersecting ancestry blocks;
8. exported analysis contains no theorem/proof/verdict field that claims a new
   invariant.

These 8 criteria are implemented as 13 granular checks (T1-T13) in
`scripts/ghost-workbench-tests.mjs`; the suite reports `pass=13 fail=0` as of
2026-06-27.

Run:

```text
npm run ghost:test
```

Serve the workbench for browser QA (no Vite needed):

```text
npm run ghost:serve   # http://127.0.0.1:5188/ghost/workbench.html
```

## 7. Exit Gate

Phase 2 is started when:

- the internal workbench opens at `/ghost/workbench.html` under a local static
  server;
- the test script passes;
- `SUNDOG_V_GHOST.md` points to this spec and the internal artifact;
- the computability cliff is visible as a separate regime, not a smooth
  continuation of the bounded examples.

Phase 2 is complete later only if browser QA confirms the selected window,
boundary edges, periodic closure state, substitution ancestry state, and cliff
note all render clearly on desktop and mobile.

**Confirmed complete 2026-06-27.** Browser QA via `npm run ghost:serve` verified:
window + L/R boundary edges render; periodic system reports "repeat cell
captured" (closed/green); Fibonacci system reports "bounded substitution
recognition" with level-N ancestry blocks/boundaries and no global period
(open/blue); cliff note renders as a separate regime; controls and dashboard
collapse to a single column at 375px with the stripe scrolling horizontally; no
console errors. Verified via accessibility snapshot + computed styles + DOM
reads (the environment's screenshot pipeline was unavailable).
