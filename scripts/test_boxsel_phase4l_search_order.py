#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-4l search-order filtration (scripts/boxsel_phase4l_search_order.py).

Locks the measured determine/resist split on REACHABILITY:
  (1) the nested chain 6|12|24 gives a MONOTONE grid_min (1/2 >= 4/9 >= 4/9) -> proper filtration;
  (2) the order-meter is LIVE -- a determine-side target (1/2) is reached at finite order 6 (the
      non-vacuity guard: the optimum's unreachability is a property of the target, not a dead probe);
  (3) the n=2 optimum (9+sqrt 17)/32 is RESIST -- infinite search order over the chain (None), not
      even approached within +0.02, reachable only down to the floor;
  (4) the measured RESIST MARGIN = floor - optimum = 4/9 - (9+sqrt 17)/32 = (47 - 9 sqrt 17)/288
      ~= 0.0343 > 0 (a measured plateau, not "irrational ergo absent");
  (5) verdict search_resist_confirmed.
Exact/pure (Fraction + Q(sqrt 17) Surd). Run: python scripts/test_boxsel_phase4l_search_order.py
"""
import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_phase4l_search_order as sl
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S

fail = 0


def check(name, cond, detail=""):
    global fail
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fail += 1


r = sl.determine_resist_report()
floors = r["floors"]
MARGIN = S(F(47, 288), F(-1, 32))   # 4/9 - (9 + sqrt 17)/32, exact

print("(1) the nested chain is a PROPER filtration (monotone grid_min):")
check("chain is the divisibility chain 6 | 12 | 24", r["chain"] == (6, 12, 24))
check("grid_min(6) = 1/2, grid_min(12) = 4/9, grid_min(24) = 4/9 (exact)",
      floors[6] == F(1, 2) and floors[12] == F(4, 9) and floors[24] == F(4, 9))
check("grid_min is monotone non-increasing along the nested chain (1/2 >= 4/9 >= 4/9)", r["monotone"])
check("reachable floor (best bounded exact search) = 4/9", r["reachableFloor"] == F(4, 9))

print("(2) the order-meter is LIVE (the non-vacuity guard):")
check("determine-side target 1/2 is reached at FINITE order 6", r["determineTargetOrder"] == 6)
check("meterLive flag is True (the meter can reach a target, so it is not a dead probe)", r["meterLive"] is True)
check("the reachable floor 4/9 is itself finite-order (reached at g=12)",
      sl.search_order(S(F(4, 9)), F(0), floors) == 12)

print("(3) the n=2 optimum is RESIST -- infinite search order over the chain:")
check("sigma_search(optimum, 0) is None (irrational -> no finite rung reaches it exactly)",
      r["optimumOrderExact"] is None)
check("sigma_search(optimum, +0.02) is None (bounded search does not even APPROACH the optimum)",
      r["optimumOrderWithinMargin"] is None)
check("optimum is reachable ONLY down to the floor (sigma_search(optimum, margin) is finite = 12)",
      r["optimumOrderAtFloor"] == 12)

print("(4) the RESIST MARGIN is measured, exact, and positive:")
check("resist margin = 4/9 - (9+sqrt 17)/32 = (47 - 9 sqrt 17)/288 exactly",
      r["resistMarginExact"] == MARGIN)
check("resist margin ~= 0.0343, strictly inside (0.03, 0.04) and > 0 (plateau, not vanishing)",
      0.03 < float(r["resistMarginExact"]) < 0.04 and r["resistMarginExact"].sign() > 0,
      f"{float(r['resistMarginExact']):.6f}")
check("plateau holds: every rung stays >= optimum + 0.02 (search never approaches truth)",
      r["plateauHolds"] is True)

print("(5) the determine/resist split is genuine and the verdict stands:")
check("DETERMINE finite-order (6) vs RESIST infinite-order (None) -- the split is real, not vacuous",
      r["determineTargetOrder"] is not None and r["optimumOrderExact"] is None
      and r["determineTargetOrder"] != r["optimumOrderExact"])
check("C1 search-reachability filtration SUPPORTED", r["c1Supported"] is True)
check("verdict == search_resist_confirmed", r["verdict"] == "search_resist_confirmed")

print(f"\n{'ALL PASS -- search-order filtration: determine (1/2 @ order 6) / resist ((9+sqrt17)/32, order inf over chain); measured resist margin 4/9 - opt ~= 0.0343' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
