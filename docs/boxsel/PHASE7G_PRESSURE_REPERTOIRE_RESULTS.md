# BoxSEL Phase 7g — Pressure-Repertoire Reach: RESULTS

**Date:** 2026-06-29  
**Status:** `INHERITS_order_relative` — **confirmed.** The frozen Phase-7b v2 pressure detector
inherits C2's order-relative characterization. Pre-registered in
[PHASE7G_PRESSURE_REPERTOIRE_PREREG.md](PHASE7G_PRESSURE_REPERTOIRE_PREREG.md) (committed `38dc4328`,
**before** the runner existed). Result matches the pre-registered prediction in every row. Toy
micro-SEL; frozen-as-portfolio.

## Result (one frozen detector, pressure channel isolated)

```text
                  shape  repertoire        pressure_low_shift  action    outcome
L liveness        pmp    [pmp]             0.1460              abstain   CAUGHT
I inheritance     helly  [pmp]             0.0000              accept    MISSED  (flags: none)
R reach-extends   helly  [pmp, helly]      0.0899              abstain   CAUGHT
```

Helly trap: reported lower **0.50** > true I_box lower **(9+√17)/32 ≈ 0.4101** ⟹ oracle-certified
false closure. Every trap is clean on all non-pressure features, so the only channel that can catch it
is `pressure_low_shift` (≥ 0.05).

## What it shows

- **The inheritance (I).** The PMP-repertoire detector ACCEPTS a genuine false closure — `shift = 0`,
  **zero flags** — because no PMP-shaped pressure reaches the Helly witness. A stable false closure
  whose witness is outside the handed repertoire is accepted ⟹ the Phase-7 detector inherits C2's
  soundness hole.
- **Liveness (L).** The *same* PMP repertoire catches the PMP false closure (`shift 0.146` →
  abstain). So the family is live; the miss in (I) is **repertoire-relative, not a dead probe** — L
  and I use the identical `[pmp]` repertoire with opposite outcomes.
- **Reach extends (R).** Handing a Helly repertoire (pressure toward `(9+√17)/32`) catches the same
  trap (`shift = 0.50 − (9+√17)/32 ≈ 0.0899` → abstain). Extend the repertoire, extend the reach.

## The order-relative law, now on the flagship-bound detector

> The Phase-7 pressure detector's reach **is its pressure repertoire** (the failure-witnesses it knows
> to push toward). A false closure whose witness lies outside the repertoire is accepted; widening the
> repertoire catches it.

This is the same finite/∞ order-relative law as the rest of the slate, on a third axis:

| conjecture | the "order" | sound when |
|---|---|---|
| **C1** | search reach (grid denominator) | optimum's search-order ≤ budget |
| **C2** | pressure reach (probe order) | false closure's σ ≤ pressure budget |
| **Phase 7g** | pressure **repertoire** (failure-shapes) | witness's shape ∈ repertoire |

## Honest boundary

Confirms a soundness **characterization** of the detector, not a fix. The actionable consequence: a
pressure-based abstention rule must **declare its repertoire/order budget** and treat closures whose
witness is outside it as out of scope — it cannot claim unconditional soundness. Toy micro-SEL; the
Helly witness is the exact closed form, but the detector and trace schema are the Phase-7b frozen ones.
Not a real-KG / calibration / product claim.

## Files

- `scripts/boxsel_phase7g_pressure_repertoire.py` — the three scenarios + verdict (reuses the Phase-6b
  schema, the frozen v2 detector, and `kkt.Q_STAR`).
- `scripts/test_boxsel_phase7g_pressure_repertoire.py` — frozen test: **15/15 PASS, exit 0**.
- `docs/boxsel/PHASE7G_PRESSURE_REPERTOIRE_PREREG.md` — the pre-registration (committed before the run).

---

*Sundog Research Lab — BoxSEL Phase-7g. The flagship-bound Phase-7 pressure detector inherits C2's
order-relative soundness characterization: reach = pressure repertoire. Internal; frozen-as-portfolio.*
