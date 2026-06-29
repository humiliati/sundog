# C2 — Orthogonal-Pressure Abstention: the Parity-Substrate Break

**Date:** 2026-06-29  
**Status:** `C2_BROKEN_order_relative` — the conjecture's **soundness is FALSIFIED as stated**, and the
break hands back a clean bounded-positive **repair** (order-relative soundness). C2 of the cross-lane
conjecture slate ([BOXSEL_CONJECTURE_SLATE.md](BOXSEL_CONJECTURE_SLATE.md)). First move was to break
it, not confirm it — per the slate discipline, *before* it could become a flagship abstention rule.

## What C2 claimed (sharp enough to attack)

> Audit a reasoner's narrow answer WITHOUT the oracle by applying **pressures it did not itself apply**;
> if the answer is **stable under every orthogonal pressure**, ACCEPT it; if it MOVES, flag false
> closure. **Soundness (load-bearing):** stable-under-all-pressure ⟹ genuinely closed. The dangerous
> error is *accepting a false closure*.
> **Falsifier:** a stable false closure invisible to every orthogonal pressure.

## The break construction

Any oracle-free pressure family is itself **finite-order** — you cannot apply an ∞-order perturbation
without the oracle. Model a reasoner that **closes** its answer at order `K` (commits from order-≤K
features) and a detector that **escalates pressure** up to budget `M > K`. A pressure of order `m`
*moves the answer* iff order-≤`m` features expose target structure the reasoner missed (detection
`own_r2 > 0.10` at `m` while the close-order saw nothing). Reuses `scripts/order_meter.py` verbatim
(`own_r2`, `history_xy`, `lfsr`, `liouville`). Reasoner closes at order 2; budget = orders {2,4,6,8}.

## Result

```text
                            close   m=2    m=4    m=6    m=8    exposed   detector
RESIST     Liouville (σ=∞)  0.000   0.000  0.000  0.000  0.000  never     STABLE → ACCEPT   ← the break
GUARD      LFSR(5)  (σ=5)   0.026   0.029  0.418  0.999  1.000  @order 4  MOVED  → flag     ← live
ABOVE-BUDG LFSR(12) (σ=12)  0.000   0.000  0.000  0.000  0.000  never     STABLE → ACCEPT   ← σ>M
```

All three are **genuine false closures** (undetermined at the reasoner's close-order). Yet:

- **The break.** The σ=∞ Liouville false closure is invisible to *every* orthogonal pressure → the
  detector ACCEPTS it. A stable false closure exists ⟹ **C2-as-stated is unsound.**
- **Live (non-vacuity guard).** The in-budget LFSR(5) false closure IS caught (moves at order 4 ≤ M),
  so the failures are a property of the *target*, not a dead probe.
- **It's the budget, not ∞-mystique.** The *finite* LFSR(12) (σ=12 > M=8) is **also** missed. The law
  is σ vs pressure-order, with σ=∞ merely the limit.

## The repair (the bounded-positive the break hands back)

> Orthogonal-pressure abstention is **sound only for false closures of order ≤ the pressure budget M**;
> it accepts every false closure of order > M, σ=∞ included. **The detector's reach IS the pressure
> family's order.**

This ties C2 straight to **C1** (search-reachability: the detector's reach = the search/pressure
order) and to the **sufficient-statistic-order schema** ([[project_sundog_suffstat_order_slate]]):
abstention is yet another order-filtered determine/resist instance. A flagship abstention rule built
on pressure must therefore *declare its order budget* and treat σ>M latents as out of scope — it
cannot silently claim soundness on resist-side closures.

## Honest boundary

- Falsifying a **universal** soundness claim needs only one substrate; parity is the canonical σ=∞
  one. The BoxSEL Phase-7 pressure detector inherits the same order-relative characterization
  (follow-up confirmation, not re-derivation).
- `lfsr(5)`'s recurrence (`x⁵+x⁴+1`, non-primitive) leaks partial structure at order 4, so the guard
  is caught at order 4 rather than 5 — *within* budget either way; the liveness conclusion is
  unaffected. `lfsr(12)` shows no low-order leak (0.000 throughout ≤8) → a clean above-budget miss.
- Not a claim that *no* detector catches σ=∞ closures — one with oracle access or an ∞-order family
  could; the point is the oracle-FREE pressure detector cannot. Toy; portfolio.

## Files

- `scripts/c2_pressure_abstention.py` — the detector, the three scenarios, the pre-registered verdict
  (reuses `order_meter.py`).
- `scripts/test_c2_pressure_abstention.py` — frozen test (threshold-crossings + verdict, robust to MLP
  drift): **18/18 PASS, exit 0**.

---

*Sundog Research Lab — C2 of the cross-lane conjecture slate. Orthogonal-pressure abstention is
broken-as-stated (a σ>budget stable false closure, σ=∞ included, is invisible and accepted) and
repaired to order-relative soundness: the detector's reach is the pressure family's order. Internal;
frozen-as-portfolio.*
