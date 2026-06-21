#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-4c optimizer evidence (scripts/boxsel_exact_inf_optimizer.py).

Locks: (1) the certified bracket [1/4, 513/1250]; (2) the deterministic exact-grid search-gap probe
(coarse AND fine grids stall >= 4/9, badly missing the ~0.41 candidate); (3) the witness is a valid
certified rational upper bound but NOT the exact KKT candidate; (4) every recorded from-scratch
optimizer misses the analytic candidate, and the certified lower bound q >= 1/4 holds. Exact/pure
(no scipy).
Run: python scripts/test_boxsel_exact_inf_optimizer.py
"""
import sys
from fractions import Fraction as F

sys.path.insert(0, "scripts")
import boxsel_exact_inf_optimizer as opt
import boxsel_phase4_interval_gap as gap

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


print("(1) certified bracket for inf I_box^n (n>=2):")
lo, hi = opt.certified_bracket()
check("bracket is [1/4, 513/1250] with 1/4 < 513/1250 and lower > 0 (no vanishing)",
      (lo, hi) == (F(1, 4), F(513, 1250)) and lo < hi and lo > 0)
check("best numerical candidate is at n=2 (finite-dimensional candidate, not a vanishing-limit pattern)",
      opt.INF_ATTAINED_DIM == 2)

print("(2) deterministic exact-grid search-gap probe (grids miss the ~0.41 candidate):")
for g in (4, 6, 8, 12, 16, 24):
    m = opt.exact_grid_min(g)
    check(f"exact grid g={g}: min q = {m} ({float(m):.4f}) > witness 513/1250 (search misses)",
          m is not None and m > hi, f"grid={float(m):.4f} vs witness={float(hi):.4f}")
check("systematic-grid floor stalls at >= 4/9 (well above the candidate ~0.41)",
      opt.search_gap_min() >= F(4, 9))

print("(3) the witness is a valid certified upper bound, but NOT the exact KKT candidate:")
w2 = gap.rational_box2_shrink_witness()
check("witness feasible and q = 513/1250 (= certified upper)",
      gap.satisfies_seed_ontology(w2) and gap.query_value(w2) == hi)
check("witness is NOT flagged as the exact KKT candidate (candidate ~0.41010 has both |A&C|,|B&C| active)",
      opt.WITNESS_IS_EXACT_OPTIMUM is False and opt.INF_NUMERICAL < float(hi))

print("(4) from-scratch optimizers all MISS the analytic candidate (search gap); lower bound holds:")
seeded = opt.OPTIMIZER_RESULTS["nelder_mead_seeded_from_witness"][2]
for name, res in opt.OPTIMIZER_RESULTS.items():
    if name == "nelder_mead_seeded_from_witness":
        continue
    worst = res if isinstance(res, F) else min(res.values())
    val = float(worst)
    check(f"{name}: best reached {val:.4f} > witness 0.4104 (fails to reach the analytic candidate)",
          val > float(hi))
check("only the witness-seeded local search reaches the candidate (~0.41010 < 0.4104)",
      seeded < float(hi))
check("certified lower bound q >= 1/4 respects the bracket (and witness >= 1/4)",
      gap.query_value(w2) >= F(1, 4) and opt.INF_NUMERICAL >= 0.25)

print(f"\n{'ALL PASS -- optimizer evidence: best candidate at n=2 ~=0.41010 in [1/4, 513/1250]; every from-scratch search misses (severe search gap); matching lower bound open' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
