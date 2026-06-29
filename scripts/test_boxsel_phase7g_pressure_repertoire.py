#!/usr/bin/env python
"""Frozen test for BoxSEL Phase 7g (scripts/boxsel_phase7g_pressure_repertoire.py).

Locks the pre-registered (docs/boxsel/PHASE7G_PRESSURE_REPERTOIRE_PREREG.md, commit 38dc4328) result
that the frozen Phase-7b v2 pressure detector INHERITS C2's order-relative characterization, with the
pressure channel isolated:
  (1) the Helly trap is an oracle-certified false closure (reported lower 0.50 > true (9+sqrt 17)/32);
  (2) liveness -- the PMP repertoire CATCHES the PMP false closure (abstain via pressure_low_shift);
  (3) THE INHERITANCE -- the SAME PMP repertoire MISSES the Helly false closure (accept, zero flags,
      shift 0): a genuine false closure accepted because its witness is outside the repertoire;
  (4) reach extends -- a Helly repertoire CATCHES the Helly trap (shift = 0.50 - (9+sqrt 17)/32);
  (5) verdict INHERITS_order_relative.
Deterministic (exact constructions + the Q(sqrt 17) Surd). Run: python scripts/test_boxsel_phase7g_pressure_repertoire.py
"""
import sys

sys.path.insert(0, "scripts")
import boxsel_phase7g_pressure_repertoire as p7g
import boxsel_kkt_exact as kkt

THR = 0.05  # PRESSURE_LOW_SHIFT_THRESHOLD

fail = 0


def check(name, cond, detail=""):
    global fail
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        fail += 1


r = p7g.repertoire_reach_report()
L, I, R = r["liveness"], r["inheritance"], r["reachExtends"]
q_star = float(kkt.Q_STAR)

print("(1) the Helly trap is a genuine, oracle-certified false closure:")
check("reported lower 0.50 > true I_box lower (9+sqrt 17)/32 -> false_closed",
      r["hellyFalseClosed"] is True and r["hellySampleLower"] == 0.50 and q_star < 0.50,
      f"0.50 > {q_star:.6f}")
check("the PMP trap is also a genuine false closure (lower > q1*q2 = 0.25)", r["pmpFalseClosed"] is True)

print("(2) LIVENESS -- the PMP repertoire catches the PMP false closure (family is live):")
check("L caught (action != accept) and action == abstain", L["caught"] is True and L["action"] == "abstain")
check("L pressure_low_shift >= 0.05 (~0.146)", L["pressureLowShift"] >= THR, f"{L['pressureLowShift']:.4f}")
check("L flagged via pressure_low_shift", "pressure_low_shift" in L["flags"])

print("(3) THE INHERITANCE -- the SAME PMP repertoire MISSES the Helly false closure:")
check("L and I use the SAME PMP repertoire (opposite outcomes -> repertoire-relative, not dead)",
      L["repertoire"] == ["pmp"] and I["repertoire"] == ["pmp"])
check("I action == accept (the detector ACCEPTS a genuine false closure)", I["action"] == "accept")
check("I pressure_low_shift == 0.0 (no PMP pressure applies to a Helly trap)", I["pressureLowShift"] == 0.0)
check("I has NO flags (pressure channel isolated -> a pure pressure miss, not some other feature)",
      I["flags"] == ())
check("I not caught", I["caught"] is False)

print("(4) REACH EXTENDS -- a Helly repertoire catches the same Helly trap:")
check("R caught (action == abstain)", R["caught"] is True and R["action"] == "abstain")
check("R pressure_low_shift = 0.50 - (9+sqrt 17)/32 ~= 0.0899, and >= 0.05",
      abs(R["pressureLowShift"] - (0.50 - q_star)) < 1e-9 and R["pressureLowShift"] >= THR,
      f"{R['pressureLowShift']:.4f}")
check("R flagged via pressure_low_shift", "pressure_low_shift" in R["flags"])

print("(5) the pre-registered verdict:")
check("inherits flag is True", r["inherits"] is True)
check("verdict == INHERITS_order_relative", r["verdict"] == "INHERITS_order_relative")

print(f"\n{'ALL PASS -- Phase-7 detector inherits C2: reach = pressure REPERTOIRE; a false closure whose witness is outside the handed repertoire is accepted; extend the repertoire and it is caught' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
