#!/usr/bin/env python
"""Frozen test for H4 Stage 1 (scripts/grokking_catastrophe.py) — the tool-validation win: the Atlas's
calibrated jet classifier reads the canonical Whitney fold+cusp as an A3 cusp, distinct from the A4
swallowtail and the D4 umbilic. Deterministic (no training). The Stage-2 grokking NULL is documented in
docs/atlas/H4_GROKKING_CATASTROPHE_RESULT.md + the reproducible result JSONs (results/atlas/h4/).
Run: python scripts/test_grokking_catastrophe.py
"""
import sys
import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import grokking_catastrophe as gc   # noqa: E402
import atlas_jet_classify as jc     # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H4 Stage 1 — the calibrated jet classifier on the Whitney fold+cusp:\n")

# the Whitney cusp F(x,y)=(x, y^3+xy): exactly one A3 cusp, |c3| analytically = 6, corank-1
phi, c2, c3 = gc.synthetic_cusp(0.0)
cusps = jc.cusp_c3(phi, c2, c3)
X, Y, d = gc.synthetic_cusp_chart(0.0)
rank = jc.corank_from_chart(X, Y, d, d)["corank"]

check("Whitney cusp has exactly ONE cusp", len(cusps) == 1, f"#cusps={len(cusps)}")
check("|c3| at the cusp = 6.0 (A3, bounded; analytic value)", abs(cusps[0] - 6.0) < 0.2,
      f"|c3|={cusps[0]:.3f}")
check("the cusp is corank-1 (a cuspoid, NOT a D4 umbilic)", rank == 1, f"corank={rank}")
check("|c3|=6.0 is STABLE across translations (genuine A3, not a coincidence)",
      max(abs(jc.cusp_c3(*gc.synthetic_cusp(s))[0] - 6.0) for s in (0.2, -0.2)) < 0.3)

# A4 separation: swallowtail |c3| collapses toward 0 as h->0 while the cusp stays at 6.0
a4_gen = jc.cusp_c3(*jc.synthetic_swallowtail(-0.40))[0]
a4_near = jc.cusp_c3(*jc.synthetic_swallowtail(-0.02))[0]
check("A4 swallowtail |c3| COLLAPSES as h->0 (ratio < 0.30), separating A4 from the stable A3 cusp",
      (a4_near / a4_gen) < 0.30, f"ratio={a4_near/a4_gen:.2f} (gen={a4_gen:.2f}, near-merge={a4_near:.2f})")

# D4 separation: the umbilic fires corank-2, the cusp does not
Xu, Yu, du = jc.synthetic_umbilic(0.0)
check("D4 hyperbolic umbilic fires corank-2 (the cusp stays corank-1) -> A3 vs D4 separated",
      jc.corank_from_chart(Xu, Yu, du, du)["corank"] == 2)

print(f"\n{'ALL PASS -- the calibrated jet classifier reads a Whitney fold+cusp as A3 (|c3|=6.0, corank-1), distinct from A4 (collapse) and D4 (corank-2). The A2/A3/A4/D4 synthetic-control ladder is closed.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
