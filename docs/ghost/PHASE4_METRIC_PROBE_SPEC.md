# Ghost Phase 4 - Metric Probe Spec (Rigorous Falsification Battery)

- Artifact id: `GHOST-PHASE4-METRIC-PROBE-SPEC`
- Date: 2026-06-27
- Status: Phase 4 start (optional, non-gating). Pre-registration of definitions,
  method, and falsification criteria BEFORE any results are computed.
- Ledger: [`../SUNDOG_V_GHOST.md`](../SUNDOG_V_GHOST.md)
- Lit-pass: [`../GHOST_LITPASS_MEMO.md`](../GHOST_LITPASS_MEMO.md) (Q2 Resolution)
- Phase 3: [`PHASE3_APERIODIC_READER_SPEC.md`](PHASE3_APERIODIC_READER_SPEC.md)

> Ordered does not mean repeating. Local does not mean closed.

## 0. Pre-registration note

This spec fixes the observable, the substrates, the method, and the pass/fail
criteria up front. Results go in a separate memo
(`GHOST_PHASE4_METRIC_MEMO.md`). Criteria here are not to be edited after seeing
results; if the method proves infeasible, that is recorded as an outcome, not
patched away. Default verdict is "not a new invariant" unless the data forces
otherwise.

## 1. Purpose

Phase 4 is the optional metric probe. It prototypes a finite-patch observable,
maps it to existing vocabulary, and tries to falsify the **Ghost Boundary
Heuristic**. Per the ledger, three outcomes are all acceptable: the metric is
identified as known vocabulary, it dies cleanly, or it earns a narrow technical
memo with an explicit "not a new invariant" default. Phase 4 must not gate the
reader (Phase 3 is complete on its own).

## 2. Observable (mapped to known vocabulary in Q2)

Primary observable: the **recognizability radius**, i.e. Mosse's **constant of
recognizability** for a primitive substitution. Bilateral form (cited):

> there is L such that if the (2L+1)-window around position i equals the
> (2L+1)-window around position j, and i is a level-1 cut point (supertile
> boundary), then j is a level-1 cut point of the same role.

The recognizability radius is the least such L. It is a finite, computable,
fixed constant of the morphism (Mosse; Durand & Leroy, "The constant of
recognizability is computable for primitive morphisms," arXiv:1610.05577).

Secondary / control observable: **repeat-cell capture radius** for a periodic
control (the least window that forces the true period and no smaller period).

These are known objects. The probe does NOT define a new invariant.

## 3. Substrates

1D primitive substitutions (cut points and roles taken from the generation
ancestry):

- **Fibonacci**: a -> ab, b -> a.
- **Period-doubling**: a -> ab, b -> aa.
- **Thue-Morse**: a -> ab, b -> ba.

Periodic 1D control:

- **periodic4**: motif `A B C D` (reuse of the Phase 2 notion).

2D substrate (the showcase, harder; numerical):

- **Penrose P3** (reuse `ghost/aperiodic-core.js`): recognizability radius over
  interior tiles, patches compared up to isometry.

## 4. Method

### 4.1 1D recognizability radius

- Generate the substitution word to depth d with per-letter ancestry; mark
  **cut points** (positions where a level-1 supertile begins) and each letter's
  **role** (its index within its level-1 super-letter).
- For radius L, a position i is **evaluable** if [i-L, i+L] lies inside the word.
- L is sufficient iff for every pair of evaluable positions i, j with equal
  (2L+1)-window, cut-point status and role agree. Comparison is
  **translation-only** (windows compared directly, not reversed), matching the
  cited bilateral constant `u[i-L,i+L) = u[j-L,j+L)`; reflection/rotation is used
  only for the 2D geometric variant in 4.3, where the tiling genuinely carries an
  isometry group. Role = (source letter, offset in block), so a sufficient L
  also recovers the desubstitution, not only the cut points.
- Recognizability radius = least sufficient L. Report it.

### 4.2 Periodic repeat-cell capture radius

- Least window length w such that the window admits the true period p and no
  divisor period < p. Exactly computable; finite.

### 4.3 2D Penrose recognizability radius (numerical)

- Generate Penrose to depth d (`makePenrose`). Restrict to **interior tiles**:
  those whose centroid lies within `(1 - margin)` of the origin, evaluated only
  for radii `r <= margin`, so no patch is truncated by the finite boundary.
- For a tile t and radius r, the **patch** is the set of tiles whose centroid is
  within r of t's centroid, each expressed in t's local frame (translate to t's
  centroid; rotate by minus t's orientation, where orientation is the angle of
  (vertexA - centroid)). The patch **signature** is the sorted multiset of
  `(type, round(x, k), round(y, k))` at decimal precision `k = 4`.
- Two tiles have the **same r-patch up to isometry** iff their signatures are
  equal, or one equals the other's **mirrored** signature (negate y).
- t's **role** = (parent color, child index) from its ancestry path.
- r is sufficient iff equal-signature interior tiles always share a role.
  Recognizability radius = least sufficient r (reported in tile-edge units),
  with the tolerance `k` and `margin` recorded.

## 5. Falsification target and pre-registered criteria

**Ghost Boundary Heuristic (universal/unbounded reading under test):**

> Any local rule system that forces non-periodic global order must leave, on
> every sufficiently large finite circle, a detectable dependency on context
> outside that circle.

Read as the strong claim that the outside debt is **unbounded** (does not
stabilize). Pre-registered tests:

- **P1 (finiteness):** for each aperiodic substrate, the recognizability radius
  is finite.
- **P2 (depth stability):** the 1D recognizability radius is identical at two
  sufficiently large depths (Mosse's fixed-constant property). This is the
  privileged-truth check: a depth-dependent or diverging value would indicate a
  bug, not unboundedness.
- **P3 (control collapse):** the periodic control's repeat-cell capture radius
  is finite and its non-periodic "ghost" is absent (it has a repeat cell).
- **P4 (2D consistency):** the Penrose recognizability radius is finite on the
  interior core and does not grow with depth beyond numerical noise.

Decision:

- If P1+P2 hold (expected, by Mosse/Durand-Leroy), the **unbounded** form of the
  Ghost Boundary Heuristic is **FALSIFIED**; the surviving statement is the
  bounded form: outside debt = a finite recognizability radius.
- The genuinely unbounded/undecidable regime is Wang/SFT extension, which is NOT
  simulable here; it is recorded as the boundary where the heuristic's
  "detectable" premise itself fails, not as a measured data point.
- Verdict default: the observable is the **known constant of recognizability**
  (Mosse; Durand-Leroy), bounded; **not a new invariant**. Outcome class:
  "metric identified as known vocabulary."

## 6. Artifacts

- Pure core: `ghost/metric-probe-core.js` (1D substitutions + recognizability,
  periodic capture radius, 2D Penrose recognizability, falsification harness).
- Acceptance tests: `scripts/ghost-metric-tests.mjs`, wired as
  `npm run ghost:metric:test`.
- Memo (results, written after the run): `docs/ghost/GHOST_PHASE4_METRIC_MEMO.md`.

Staged implementation (each stage validated before the next): (S1) 1D family +
periodic control + tests; (S2) 2D Penrose recognizability + tests; (S3) memo +
ledger/memory update.

## 7. Acceptance Tests (pre-registered)

1. each substitution generator obeys its length recurrence / letter counts;
2. cut points partition the generated word; roles are well defined;
3. recognizability radius is finite for Fibonacci, period-doubling, Thue-Morse;
4. recognizability radius is depth-stable (equal at depth d and d+1 for d large
   enough) for each 1D substitution (privileged-truth check);
5. periodic control repeat-cell capture radius equals the period-based value and
   is finite;
6. Penrose recognizability radius is finite on the interior core and stable
   across available depths within tolerance;
7. the falsification harness reports the unbounded heuristic FALSIFIED and the
   bounded recognizability radius for every aperiodic substrate;
8. exported probe data contains no theorem/proof/invariant/conjecture/claim
   field asserting a new invariant.

Run:

```text
npm run ghost:metric:test
```

## 8. Exit Gate

Phase 4 is complete when:

- the acceptance suite passes;
- the memo records, per substrate, the recognizability radius (or repeat-cell
  capture radius) and the falsification result, with the "not a new invariant"
  default explicit;
- the observable is stated as known vocabulary (constant of recognizability /
  repeat-cell capture radius), satisfying the ledger's Phase 4 exit gate;
- `SUNDOG_V_GHOST.md` Phase 4 points to this spec and the memo.

Phase 4 remains explicitly optional: failure or clean death here does not affect
the Phase 3 reader's completion.
