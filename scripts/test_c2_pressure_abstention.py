#!/usr/bin/env python3
"""Frozen test for the C2 parity-substrate break (scripts/c2_pressure_abstention.py).

Locks the soundness counterexample to orthogonal-pressure abstention:
  (1) all three scenarios are GENUINE false closures (undetermined at the reasoner's close-order);
  (2) the pressure family is LIVE -- the in-budget guard LFSR(5) is CAUGHT within the budget
      (the non-vacuity guard: failures elsewhere are a property of the target, not a dead probe);
  (3) THE BREAK -- the sigma=inf Liouville false closure is invisible to EVERY orthogonal pressure
      and is ACCEPTED (unsound as stated);
  (4) the finite-but-above-budget LFSR(12) is ALSO missed -> the law is "sigma vs pressure budget",
      not infinity-mystique;
  (5) verdict C2_BROKEN_order_relative.

Asserts threshold-crossings + verdict (robust to MLP nondeterminism across sklearn versions), not
brittle exact floats. Run: python scripts/test_c2_pressure_abstention.py
"""
import sys

sys.path.insert(0, "scripts")
import c2_pressure_abstention as c2

THR = c2.THR
M = c2.M

fail = 0


def check(name, cond, detail=""):
    global fail
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fail += 1


r = c2.c2_break_report()
resist, guard, above = r["resist"], r["guard"], r["aboveBudget"]

print("(1) all three are GENUINE false closures (undetermined at the reasoner's close-order):")
check("RESIST Liouville close-order score <= THR", resist["closedScore"] <= THR,
      f"{resist['closedScore']:.3f}")
check("GUARD LFSR(5) close-order score <= THR", guard["closedScore"] <= THR, f"{guard['closedScore']:.3f}")
check("ABOVE-BUDGET LFSR(12) close-order score <= THR", above["closedScore"] <= THR,
      f"{above['closedScore']:.3f}")
check("allFalseClosures flag is True", r["allFalseClosures"] is True)

print("(2) the pressure family is LIVE (non-vacuity guard -- the in-budget false closure is caught):")
check("GUARD LFSR(5) answer MOVED under pressure", guard["answerMoved"] is True)
check("GUARD caught WITHIN budget (exposed at a finite order <= M)",
      guard["exposedAtOrder"] is not None and guard["exposedAtOrder"] <= M,
      f"exposed@{guard['exposedAtOrder']}, M={M}")
check("GUARD high-order pressure detects strongly (score@M > 0.5)", guard["pressureScores"][M] > 0.5,
      f"{guard['pressureScores'][M]:.3f}")
check("GUARD did not cheat -- lowest pressure order (2) still <= THR (genuine escalation)",
      guard["pressureScores"][2] <= THR, f"{guard['pressureScores'][2]:.3f}")
check("meterLive flag is True", r["meterLive"] is True)

print("(3) THE BREAK -- the sigma=inf false closure is invisible to EVERY pressure and ACCEPTED:")
check("RESIST Liouville: every pressure order <= THR (no orthogonal pressure moves the answer)",
      all(s <= THR for s in resist["pressureScores"].values()),
      f"max={max(resist['pressureScores'].values()):.3f}")
check("RESIST exposedAtOrder is None (never exposed across the whole budget)",
      resist["exposedAtOrder"] is None)
check("RESIST answer did NOT move -> detector ACCEPTS a false closure (unsound)",
      resist["answerMoved"] is False)
check("resistStable flag is True", r["resistStable"] is True)

print("(4) finite-but-above-budget ALSO escapes (it's the budget, not infinity-mystique):")
check("ABOVE-BUDGET LFSR(12): every pressure order <= THR (sigma=12 > M=8 never caught)",
      all(s <= THR for s in above["pressureScores"].values()),
      f"max={max(above['pressureScores'].values()):.3f}")
check("ABOVE-BUDGET answer did NOT move (accepted despite being a finite false closure)",
      above["answerMoved"] is False)
check("aboveBudgetStable flag is True", r["aboveBudgetStable"] is True)

print("(5) the pre-registered verdict:")
check("kill condition holds (live guard + resist stable + above-budget stable + all false closures)",
      r["kill"] is True)
check("verdict == C2_BROKEN_order_relative", r["verdict"] == "C2_BROKEN_order_relative")

print(f"\n{'ALL PASS -- C2 broken-as-stated: a stable false closure of order > pressure budget (sigma=inf incl.) is invisible to every orthogonal pressure and accepted; sound only for order <= M' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
