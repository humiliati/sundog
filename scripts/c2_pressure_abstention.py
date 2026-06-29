#!/usr/bin/env python3
r"""C2 -- orthogonal-pressure abstention: the PARITY-SUBSTRATE BREAK.

Conjecture C2 (docs/boxsel/BOXSEL_CONJECTURE_SLATE.md): a reasoner's narrow answer can be audited
WITHOUT the oracle by applying PRESSURES it did not itself apply -- if the answer is STABLE under every
orthogonal pressure, ACCEPT it; if it MOVES, flag false closure. The load-bearing (soundness) direction:
  stable-under-all-pressure  =>  genuinely closed.
Falsifier (the thing to BREAK): a STABLE FALSE CLOSURE invisible to every orthogonal pressure -- the
detector would then ACCEPT a false closure (unsound).

THE BREAK. Any oracle-free pressure family is itself FINITE-ORDER (you cannot apply an infinite-order
perturbation without the oracle). Model a reasoner that CLOSES its answer at order K (commits from
order-<=K features) and a detector that escalates pressure up to budget M > K. A pressure of order m
"moves the answer" iff order-<=m features expose target structure the reasoner missed (detection
own_r2 > thr at m while the reasoner's close-order saw nothing). The detector ACCEPTS iff the target
stays invisible through the whole budget.

  RESIST target  = Liouville next-symbol     (sigma = infinity: no finite-order sufficient statistic)
  DETERMINE/guard= LFSR(5)  next-symbol       (sigma = 5,  IN budget M=8) -- a false closure the
                   pressure CATCHES, so the family is provably LIVE (non-vacuity guard)
  ABOVE-BUDGET   = LFSR(12) next-symbol        (sigma = 12 > M=8) -- a FINITE false closure the pressure
                   MISSES, showing the law is "sigma vs pressure-order", NOT infinity-mystique

Pre-registered (both outcomes live):
  KILL (C2 broken-as-stated): RESIST INVISIBLE through the budget (stable false closure accepted)
                              AND guard CAUGHT (apparatus live) AND above-budget INVISIBLE.
  SURVIVE (bounded-positive):  some pressure <= M catches the RESIST target -> no soundness hole.
  VOID:                        guard NOT caught -> pressure family dead, probe uninformative.

The repair the break points at: orthogonal-pressure abstention is SOUND only for false closures of
order <= the pressure budget M; it accepts EVERY false closure of order > M, sigma=infinity included.
The detector's reach IS the pressure family's order -- which ties C2 to C1 (search-order) and the
sufficient-statistic-order schema. Reuses scripts/order_meter.py verbatim. Deterministic.
"""
import numpy as np

import order_meter as om   # own_r2, history_xy, lfsr, liouville -- reused verbatim

N = 200_000                 # sequence length
M_SAMPLES = 3000            # samples per detection (matches order_meter self-test)
THR = 0.10                  # detection threshold (order_meter default)
CLOSE_ORDER = 2             # the reasoner COMMITS ("closes") from order-<=2 features
PRESSURE_BUDGET = (2, 4, 6, 8)   # orthogonal pressures the detector escalates through; max order M = 8
M = max(PRESSURE_BUDGET)


def detection_by_order(seq, orders, seed):
    """own_r2 of an order-m history window predicting the next symbol, per pressure order m."""
    return {m: om.own_r2(*om.history_xy(seq, m, M_SAMPLES, np.random.default_rng(seed + m)))
            for m in orders}


def audit(seq, seed):
    """The pressure-abstention detector on one sequence.

    The reasoner closes at CLOSE_ORDER; the detector escalates pressure through PRESSURE_BUDGET. The
    answer MOVES iff some pressure order exposes structure beyond the reasoner's close-order.
    """
    closed = om.own_r2(*om.history_xy(seq, CLOSE_ORDER, M_SAMPLES, np.random.default_rng(seed)))
    scores = detection_by_order(seq, PRESSURE_BUDGET, seed=seed)
    exposed_at = next((m for m in sorted(scores) if scores[m] > THR), None)
    return {
        "closedScore": closed,                 # what the reasoner saw at its close-order (<= THR = false closure)
        "pressureScores": scores,              # detection at each escalated pressure order
        "exposedAtOrder": exposed_at,          # least pressure order that moves the answer (None = never)
        "answerMoved": exposed_at is not None, # detector flags false closure iff this is True
    }


def c2_break_report():
    lam = om.liouville(N)        # sigma = infinity
    lf5 = om.lfsr(5, N)          # sigma = 5   (in budget)
    lf12 = om.lfsr(12, N)        # sigma = 12  (above budget M = 8)

    resist = audit(lam, seed=3)
    guard = audit(lf5, seed=1)
    above_budget = audit(lf12, seed=7)

    # reality check: at the reasoner's close-order NONE of the three is determined,
    # so all three are GENUINE false closures (the reasoner was wrong to close).
    all_false_closures = all(s["closedScore"] <= THR for s in (resist, guard, above_budget))

    meter_live = guard["answerMoved"]                       # the pressure family CAN move an answer
    resist_stable = not resist["answerMoved"]               # sigma=inf false closure stays accepted
    above_budget_stable = not above_budget["answerMoved"]   # finite-but-above-budget also escapes

    kill = all_false_closures and meter_live and resist_stable and above_budget_stable
    if not meter_live:
        verdict = "VOID_dead_pressure"
    elif not resist_stable:
        verdict = "C2_SURVIVES"               # the detector caught the sigma=inf false closure
    else:
        verdict = "C2_BROKEN_order_relative"  # accepted a stable false closure -> unsound as stated

    return {
        "resist": resist,
        "guard": guard,
        "aboveBudget": above_budget,
        "allFalseClosures": all_false_closures,
        "meterLive": meter_live,
        "resistStable": resist_stable,
        "aboveBudgetStable": above_budget_stable,
        "pressureBudgetMaxOrder": M,
        "kill": kill,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    r = c2_break_report()
    print("C2 -- orthogonal-pressure abstention: parity-substrate break")
    print(f"  reasoner closes at order {CLOSE_ORDER}; pressure budget = orders {PRESSURE_BUDGET} (M={M})")
    print()
    for name, key, sigma in (("RESIST     Liouville", "resist", "inf"),
                             ("GUARD      LFSR(5)  ", "guard", "5"),
                             ("ABOVE-BUDG LFSR(12) ", "aboveBudget", "12")):
        s = r[key]
        scores = "  ".join(f"m={m}:{v:.3f}" for m, v in s["pressureScores"].items())
        moved = "MOVED -> flag" if s["answerMoved"] else "STABLE -> ACCEPT"
        print(f"  {name} (sigma={sigma:>3}):  close={s['closedScore']:.3f}  | {scores}  | "
              f"exposed@{s['exposedAtOrder']}  {moved}")
    print()
    print(f"  all three are genuine false closures (undetermined at close-order): {r['allFalseClosures']}")
    print(f"  meter live (guard caught): {r['meterLive']}   "
          f"resist stable: {r['resistStable']}   above-budget stable: {r['aboveBudgetStable']}")
    print(f"  VERDICT: {r['verdict']}")
    if r["kill"]:
        print("  => C2-as-stated is UNSOUND: a stable false closure (sigma > pressure budget M=8, incl.")
        print("     sigma=inf) is invisible to every orthogonal pressure and is ACCEPTED. Sound only for")
        print("     false closures of order <= M. The detector's reach IS the pressure family's order.")
