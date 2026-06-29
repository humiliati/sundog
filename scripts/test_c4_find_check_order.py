#!/usr/bin/env python3
"""Frozen test for C4 (scripts/c4_find_check_order.py): the order-relative law is MODE-RELATIVE on the
find/check axis.

Locks:
  (1) parity modes DIVERGE -- verify-order finite (max Omega = 17 for N=2e5) vs predict-order infinite;
  (2) the finite-sigma LFSR control has BOTH orders finite (modes agree) and the predict-meter detects
      it (non-vacuity guard);
  (3) the law holds PER-MODE (a finite verify-budget confirms parity; no finite predict-budget does);
  (4) the C2 mode-confusion gap (predict-order inf > verify-order finite for one target);
  (5) verdict EXTENDS_mode_relative.
Exact / deterministic (Omega sieve + exact predict-meter + fixed-seed LFSR). Load-bearing values locked
exactly; the control's exact predict value is asserted only finite (qualitative).
Run: python scripts/test_c4_find_check_order.py
"""
import sys

sys.path.insert(0, "scripts")
import c4_find_check_order as c4

INF = c4.INF
K = c4.PREDICT_K

fail = 0


def check(name, cond, detail=""):
    global fail
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fail += 1


r = c4.find_check_report()
p, c = r["parity"], r["control_lfsr"]

print("(1) parity modes DIVERGE (the 4th-kind core):")
check("parity verify-order is finite = 17 (max Omega over n<=2e5 ~ log2 N)", p["verifyOrder"] == 17.0,
      f"{p['verifyOrder']}")
check("parity predict-order is INFINITE (no finite history order <= K determines lambda)",
      p["predictOrder"] == INF)
check("parityDiverges: verify finite AND predict infinite for ONE target", r["parityDiverges"] is True)

print("(2) finite-sigma control: modes agree, predict-meter live (non-vacuity guard):")
check("LFSR control verify-order finite (= 5, the state window)", c["verifyOrder"] == 5.0)
check("LFSR control predict-order FINITE (detected at a finite order <= K)",
      c["predictOrder"] < INF and c["predictOrder"] <= K, f"{c['predictOrder']}")
check("controlAgrees: both modes finite", r["controlAgrees"] is True)
check("meterLive: the predict-meter detects the control's finite order", r["meterLive"] is True)

print("(3) the law holds PER-MODE (resolves iff budget >= the mode's order):")
check("verify mode: a finite verify-budget confirms parity (verify_law)", r["verifyLawHolds"] is True)
check("predict mode: no finite predict-budget (<= K) resolves parity -> resist (predict_law)",
      r["predictLawHolds"] is True)

print("(4) the C2 mode-confusion gap:")
check("predict-order (soundness) is INF while verify-order (a bounded detector's budget) is finite",
      r["modeConfusionGap"] is True and p["predictOrder"] == INF and p["verifyOrder"] < INF)

print("(5) the pre-registered verdict:")
check("extends flag is True", r["extends"] is True)
check("verdict == EXTENDS_mode_relative", r["verdict"] == "EXTENDS_mode_relative")

print(f"\n{'ALL PASS -- find/check is a 4th KIND: order is a mode-vector (parity verify=17 / predict=inf); law holds per-mode, no single scalar budget; explains C2 as a verify-vs-predict mode-confusion' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
