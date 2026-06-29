# BoxSEL Phase 7g — Pressure-Repertoire Reach: PRE-REGISTRATION

**Date:** 2026-06-29  
**Status:** `PRE-REGISTERED` — spec + verdict frozen **before** the runner exists. **No results in this
file.** Checks whether the frozen Phase-7b v2 pressure detector inherits C2's order-relative
characterization (`C2_BROKEN_order_relative`). Committed before `boxsel_phase7g_pressure_repertoire.py`
is written.

## Question

Is the Phase-7b v2 trace detector's reach bounded by the **order/repertoire of the pressure family it
is handed**, so that a false closure whose revealing witness lies outside that repertoire is ACCEPTED
— the same soundness boundary C2 broke on parity?

## Mechanism (read from the detector, before any run)

- `boxsel_phase7b_v2_detector.detector_decision` is a pure function of `GeneralTraceFeatures`.
- Its only pressure signal is `pressure_low_shift = base_low − min(low of each handed pressure_trace)`,
  flagged at ≥ `PRESSURE_LOW_SHIFT_THRESHOLD = 0.05` (an IMMEDIATE_ABSTAIN flag).
- The pressure family is a finite, **shape-specific** tuple: `pmp_query_pressure_trace` constructs a
  path toward the *known* PMP witness `q1·q2`. The detector can only push toward witnesses its handed
  family knows how to construct → its **"order" = the pressure family's repertoire of failure-shapes.**

## Design — three scenarios, one frozen detector, pressure channel isolated

Every trap is clean on all NON-pressure features (loss ≤ 1e-9, no constraint violation, `support_floor`
> 0.08, drops below threshold, no seed/dimension/optimizer spread), so the **only** channel that can
catch it is pressure. `pressure_for(trap, repertoire)` builds a shape's pressure only if the trap
matches that shape (a PMP repertoire builds nothing for a Helly-shaped trap — it has no PMP premises).

| # | trap | repertoire handed | predicted action |
|---|---|---|---|
| **L** liveness | PMP false closure (sample lower ≈ 0.40; PMP witness `q1·q2 = 0.25`) | PMP | `pressure_low_shift ≈ 0.15 ≥ 0.05` → **ABSTAIN** (caught) |
| **I** inheritance | Helly false closure (sample lower 0.50; true I_box lower `(9+√17)/32 ≈ 0.41010`) | PMP only | no PMP pressure applies → shift 0, all features clean → **ACCEPT** (missed) |
| **R** reach-extends | same Helly trap | + Helly (toward `(9+√17)/32`) | `pressure_low_shift = 0.50 − 0.41010 ≈ 0.0899 ≥ 0.05` → **ABSTAIN** (caught) |

## Pre-registered verdict (frozen)

```text
INHERITS  iff all three hold:
  (L) PMP repertoire CATCHES the PMP false closure   (action != accept)
  (I) the SAME PMP repertoire MISSES the Helly false closure (action == accept), although it is
      oracle-certified false-closed: sample lower 0.50 > true lower (9+√17)/32
  (R) a Helly repertoire CATCHES the Helly trap        (action != accept)
DOES-NOT-INHERIT  iff the PMP repertoire ALSO catches the Helly trap (shift >= 0.05) -> the detector is
                  shape-agnostic; C2's law does not transfer here (honest negative).
VOID              iff the PMP repertoire fails to catch the PMP trap (action == accept) -> dead family.
```

## Faithfulness guard (no strawman)

The missed-case family (PMP) **must be the same legitimate family that catches the PMP trap in (L)** —
(L) is the liveness control. The Helly trap is missed because its shape ∉ the handed repertoire, NOT
because the pressure is dead. The pressure channel is isolated: traps are clean on every non-pressure
feature, so a miss is a pressure miss, and a catch is a pressure catch (`pressure_low_shift` flag only).

## Frozen constants (unchanged in the runner)

- detector + threshold: `boxsel_phase7b_v2_detector` (FROZEN), `PRESSURE_LOW_SHIFT_THRESHOLD = 0.05`.
- Helly trap: sample interval `[0.50, 1.0]`; support `condition_mass=0.25, numerator_mass=0.125,
  atom_support_min=0.5, meet_support_min=0.25` → `support_floor=0.125 > 0.08`; loss 0; equal lowers
  (drops 0). false_closed ⇔ `0.50 > Q_STAR = (9+√17)/32` (`boxsel_kkt_exact.Q_STAR`).
- Helly pressure: a trace with lower endpoint `= float(Q_STAR) ≈ 0.4100970` → shift `≈ 0.0899`.
- PMP trap: PMP-shaped false closure, sample lower ≈ 0.40, PMP pressure toward `q1·q2 = 0.25`.
- caught = action ∈ {abstain, widen}; missed = action == accept.

## Reuses

`boxsel_phase6b_trace_schema` (`GeneralTrace`, `EndpointObservation`, `ConstraintTrace`, `SupportTrace`,
`pmp_query_pressure_trace`, `pressure_low_shift`), `boxsel_phase7b_v2_detector` (`detector_decision`),
`boxsel_kkt_exact.Q_STAR`.

## What this is NOT

Not a real-KG / calibration / product claim; toy micro-SEL. It confirms a soundness **characterization**
of the flagship-bound detector (reach = pressure repertoire), not a fix. C2's universality was already
falsified on parity; this checks the actual Phase-7 pressure detector inherits the same order-relative
law. Links: [[BOXSEL_CONJECTURE_SLATE]], [[C2_PRESSURE_ABSTENTION_BREAK]].

---

*Sundog Research Lab — BoxSEL Phase-7g pre-registration. Frozen before the runner; results in
`PHASE7G_PRESSURE_REPERTOIRE_RESULTS.md`.*
