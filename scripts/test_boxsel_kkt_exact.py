#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-4d exact KKT candidate (scripts/boxsel_kkt_exact.py).

Verifies, in exact Q(sqrt 17) arithmetic, that the constructed n=2 config is feasible and achieves
q* = (9 + sqrt 17)/32 -- strictly below the old rational witness 513/1250 and the Nelder-Mead
value, and inside the certified bracket [1/4, (9+sqrt17)/32].
Run: python scripts/test_boxsel_kkt_exact.py
"""
import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_kkt_exact as kkt
from boxsel_kkt_exact import Surd as S

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


e = kkt.optimal_config()
A, B, C = e["A"], e["B"], e["C"]
half, quarter, eighth = S(F(1, 2)), S(F(1, 4)), S(F(1, 8))

print("(1) the optimizing x is the root of 4x^2 - 9x + 4 = 0 in (0,1):")
check("x = (9 - sqrt 17)/8 satisfies 4x^2 - 9x + 4 = 0",
      (S(4) * kkt.X_OPT * kkt.X_OPT - S(9) * kkt.X_OPT + S(4)) == 0)
check("0 < x < 1", kkt.X_OPT > 0 and kkt.X_OPT < S(1), f"x ~= {float(kkt.X_OPT):.6f}")

print("(2) the config is exactly feasible (marginals + overlaps, in Q(sqrt 17)):")
check("|A| = |B| = |C| = 1/2 exactly",
      kkt.box_volume(A) == half and kkt.box_volume(B) == half and kkt.box_volume(C) == half)
check("|A&C| = 1/4 and |B&C| = 1/4 (both ACTIVE)",
      kkt.meet_volume([A, C]) == quarter and kkt.meet_volume([B, C]) == quarter)
check("|A&B| = (9 - sqrt 17)/16 and is SLACK (> 1/4)",
      kkt.meet_volume([A, B]) == S(F(9, 16), F(-1, 16)) and kkt.meet_volume([A, B]) > quarter,
      f"|A&B| ~= {float(kkt.meet_volume([A,B])):.6f}")
check("|A&B&C| = 1/8 exactly", kkt.meet_volume([A, B, C]) == eighth)

print("(3) the exact KKT-candidate value:")
q = kkt.optimal_query_value()
check("q = |A&B&C|/|A&B| = (9 + sqrt 17)/32 exactly",
      q == kkt.Q_STAR and q == S(F(9, 32), F(1, 32)), f"q ~= {float(q):.7f}")
check("q*  >  1/4 (the proven lower bound is not met -- strict)", q > quarter)
check("q*  <  513/1250 (strictly better than the Phase-4 rational witness)", q < S(F(513, 1250)),
      f"{float(q):.7f} < {513/1250:.7f}")
check("q*  <  the Nelder-Mead numerical value 0.4100984 (search under-converged above the candidate)",
      float(q) < 0.4100980)

print("(4) the certified sandwich tightens:")
lo, upper = kkt.certified_sandwich()
check("1/4 <= inf I_box^n <= (9 + sqrt 17)/32, with lower < candidate upper (gap persists, > 0)",
      lo == quarter and upper == kkt.Q_STAR and lo < upper and lo.sign() > 0)
check("candidate float value ~= 0.4100970 (matches the Phase-4c numerical infimum)",
      abs(float(upper) - 0.4100970) < 1e-6, f"{float(upper):.7f}")

print(f"\n{'ALL PASS -- exact KKT candidate: q = (9 + sqrt 17)/32 ~= 0.4100970 is certified achievable in Q(sqrt 17); both C-overlaps active; < 513/1250' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
