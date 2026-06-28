# Ghost Phase 4 - Metric Probe Memo (Results)

- Artifact id: `GHOST-PHASE4-METRIC-MEMO`
- Date: 2026-06-27
- Status: Phase 4 COMPLETE (optional, non-gating). Acceptance suite
  `npm run ghost:metric:test` = `pass=23 fail=0`.
- Spec (pre-registration): [`PHASE4_METRIC_PROBE_SPEC.md`](PHASE4_METRIC_PROBE_SPEC.md)
- Ledger: [`../SUNDOG_V_GHOST.md`](../SUNDOG_V_GHOST.md)
- Lit-pass / Q2: [`../GHOST_LITPASS_MEMO.md`](../GHOST_LITPASS_MEMO.md)

> Default verdict: not a new invariant.

## 1. What was tested

A rigorous falsification battery for the **recognizability radius** = Mosse's
**constant of recognizability** (Mosse; Durand & Leroy, "The constant of
recognizability is computable for primitive morphisms," arXiv:1610.05577), the
finite-patch observable that Q2 identified as the operational form of "outside
debt." The probe tries to falsify the **unbounded** reading of the Ghost
Boundary Heuristic ("a local rule forcing non-periodic order leaves a detectable
dependency outside every sufficiently large finite circle") by measuring whether
the observable is finite and bounded, or grows without bound.

Pure core: `ghost/metric-probe-core.js`. Tests: `scripts/ghost-metric-tests.mjs`.

## 2. Results

| substrate | observable | value | finite | depth behavior |
| --- | --- | --- | --- | --- |
| Fibonacci (a->ab, b->a) | recognizability radius (letters) | **1** | yes | exactly stable (d12 = d13) |
| period-doubling (a->ab, b->aa) | recognizability radius (letters) | **1** | yes | exactly stable (d8 = d9) |
| Thue-Morse (a->ab, b->ba) | recognizability radius (letters) | **2** | yes | exactly stable (d8 = d9) |
| periodic `ABCD` (control) | repeat-cell capture radius (letters) | **5** | yes | has a repeat cell (no forced non-periodicity) |
| Penrose P3 (2D) | recognizability radius (finest-edge units) | **~0.978** | yes | converges from below: 0.842, 0.920, 0.979, 0.978 at d=4..7; stable at d>=6 (\|d6-d7\| ~ 1e-4) |

1D values are the bilateral constant (translation-only, role = source letter +
offset), validated by exact depth-stability (Mosse's fixed-constant property -
the privileged-truth check). The 2D value is the centroid-patch
operationalization up to D10 isometry, role = (parent type, child index);
"~0.978 edges" is the measured value under this operational definition, not a
claim that it equals a specific published constant (definitions vary).

## 3. The 2D finite-sample wrinkle (recorded, not patched away)

The pre-registered 2D stability check was originally coded at depths 4 and 5 and
**failed** (0.842 vs 0.920 edges). Diagnosis: the interior core undersamples
local-environment types at low depth (70, 190, 500, 1330 interior tiles at
d=4..7), so the radius is **underestimated and climbs as the core grows**. It is
not unboundedness: the increments shrink (0.078, 0.059, ~0.0001) and the value
converges to ~0.978 edges, depth-stable by d6. The spec (section 4.4) records
this; the depth-stability criterion was unchanged, only the measurement depth was
moved to where finite-size effects vanish. Finiteness and boundedness hold at
every depth, so the verdict never depended on the resolution of this wrinkle.

## 4. Falsification result

- The **unbounded** reading of the Ghost Boundary Heuristic is **FALSIFIED** on
  every substrate tested: the outside debt is a finite, bounded recognizability
  radius (1D exactly depth-stable; 2D converges to a finite constant). The
  periodic control collapses to a finite repeat-cell capture radius and forces no
  non-periodicity.
- The genuinely unbounded regime is undecidable Wang/SFT extension, which is NOT
  simulable here; it is the boundary where the heuristic's "detectable" premise
  itself fails (Q2; Berger/Robinson), recorded, not measured.

## 5. Verdict (exit gate outcome)

Outcome class per the ledger's Phase 4 exit gate: **the metric is identified as
known vocabulary.** The observable is the constant of recognizability (Mosse;
Durand & Leroy) for the aperiodic substrates and the repeat-cell capture radius
for the periodic control. It is **not a new invariant.** Per the lane's own rule,
a metric that resolves into known theory is a success for the reader lane, not a
failure - it confirms the Q2 mapping with measured numbers and a passing,
reproducible battery.

This closes the conjecture arc opened in `SUNDOG_V_GHOST.md`: "outside debt" had
a real, finite, citable referent all along.

## 6. Reproduce

```text
npm run ghost:metric:test    # pass=23 fail=0
```
