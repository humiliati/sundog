#!/usr/bin/env python
"""Frozen test for H8 (scripts/double_descent_cusp.py) — is the regularization-induced disappearance of
double descent a Whitney A3 cusp? Locks the validated closed form + the pre-registered NULL: the bump is a
SINGLE max (no max+min fold-pair) that slides to γ→∞ and flattens, and the jet classifier finds the
critical-point caustic but NO cusp. Run: python scripts/test_double_descent_cusp.py
"""
import sys
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import double_descent_cusp as dd   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H8 — double-descent removal as a Whitney A3 cusp?  (the pre-registered NULL):\n")

# (1) the closed-form risk is VALIDATED against Monte-Carlo ridge
rc, rm = dd.risk(0.8, 0.2), dd.risk_mc(0.8, 0.2)
check("closed-form ridge risk matches Monte-Carlo (rel err < 12%)", abs(rc - rm) / rm < 0.12,
      f"closed={rc:.3f} MC={rm:.3f} ({abs(rc-rm)/rm:.1%})")

# (2) double descent is reproduced: bump present at small λ, gone at large λ
check("double descent: a γ≈1 bump present at small λ", dd.n_critical(0.1) >= 1)
check("double descent: bump GONE at large λ (regularization removal)", dd.n_critical(0.6) == 0)

# (3) NULL: it is a SINGLE max (no max+min fold-pair) — never 2 critical points
maxcrit = max(dd.n_critical(lam) for lam in (0.1, 0.2, 0.3, 0.35, 0.4))
check("NULL: never a max+min PAIR (≤1 critical point at every λ → no fold-pair to annihilate)",
      maxcrit <= 1, f"max #critical-points over λ = {maxcrit}")

# (4) NULL mechanism: the peak SLIDES to γ→∞ as λ→λ* (escape, not finite-point annihilation)
def peakg(lam):
    g = np.linspace(1.02, 80.0, 30000)
    R = dd.risk(g, lam); dR = np.gradient(R, g)
    idx = [i for i in range(2, len(g) - 2) if dR[i - 1] > 0 and dR[i + 1] < 0]
    return g[max(idx, key=lambda j: R[j])] if idx else float("inf")
pg_lo, pg_hi = peakg(0.10), peakg(0.36)
check("NULL mechanism: the peak SLIDES to large γ as λ grows (peak-γ diverges, ≥2× over the sweep)",
      pg_hi > 2.0 * pg_lo, f"peak γ: {pg_lo:.2f} (λ=0.1) → {pg_hi:.2f} (λ=0.36)")

# (5) the jet classifier finds the caustic but NO cusp (corank-1) — not an A3/A4 catastrophe
r = dd.cusp_chart((1.05, 4.0), (0.05, 0.6))
check("jet classifier: caustic present (the critical-point locus) but NO cusp (the null)",
      r["caustic"] and r["n_cusps"] == 0, f"caustic={r['caustic']} #cusps={r['n_cusps']}")
check("jet classifier: corank-1 (not a D4 umbilic)", r["corank"] == 1)

print(f"\n{'ALL PASS -- the regularization-induced disappearance of double descent is NOT a Whitney cusp: a single peak slides to γ→∞ and flattens (no max+min fold-pair, no cusp). A clean, pre-registered NULL — the recast forbade the pole-as-fold error, and the real test came out null.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
