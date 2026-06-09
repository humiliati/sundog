#!/usr/bin/env python
"""H4 (slate `ww6koomb1`) — grokking as a Whitney fold+cusp: point the calibrated jet classifier at it.

The catastrophe-theory reading of grokking: the generalization order parameter, as a function of two
control knobs (weight decay x train fraction), is the equilibrium set of an effective potential, and the
memorize<->generalize transition is a FOLD (A2) that, where its discontinuity first appears, terminates in
a CUSP (A3) — the canonical 2-control model of a transition that is continuous on one side of a point and
discontinuous (a jump, with a bistable/hysteretic wedge) on the other.

STAGE 1 (this file, first): validate that the Atlas's CALIBRATED jet classifier (scripts/atlas_jet_classify.py,
already calibrated on the A4 swallowtail and the D4 umbilic) reads a KNOWN Whitney cusp correctly — the
classifier's calibration suite had no clean synthetic A3 cusp normal form (its A3 control was real column-
apex halo cusps). The pure Whitney cusp map F(x,y) = (x, y^3 + x*y) has a single A3 cusp at the origin
(fold curve x=-3y^2, c3 bounded ~6), corank-1, NOT an A4 (c3->0) and NOT a D4 (corank-2). Confirming this
establishes the cusp TEMPLATE the grokking chart (Stage 2) is matched against, and closes the A2/A3/A4/D4
synthetic-control ladder.

STAGE 2 (grokking_catastrophe_sweep.py): a real grokking sweep -> the equilibrium chart -> classify.

NOT public-eligible. Attribution: Whitney 1955 (stable maps of the plane); Thom/Zeeman (cusp catastrophe);
Power et al. 2022 (grokking); the Atlas jet classifier (Berry/Nye/Thom-Boardman lineage).
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_jet_classify as jc


def synthetic_cusp_chart(s=0.0, ng=420, L=1.0):
    """The canonical Whitney CUSP map F(x,y) = (x, y^3 + (x + s) y) on a grid around the origin.
    det DF = 3y^2 + x + s; fold = {x = -3y^2 - s}; a single A3 cusp at (x,y)=(-s,0). The cusp is a STABLE
    plane-map singularity (no unfolding parameter destroys it, unlike the swallowtail's h->0); `s` only
    translates it. Returns (X, Y, spacing) in the jet_from_chart convention."""
    g = np.linspace(-L, L, ng)
    a = np.linspace(-L, L, ng)
    G, A = np.meshgrid(g, a, indexing="ij")          # axis0 = x (=X), axis1 = y
    X = G
    Y = A ** 3 + (G + s) * A
    return X, Y, 2 * L / (ng - 1)


def synthetic_cusp(s=0.0, ng=420, L=1.0):
    X, Y, d = synthetic_cusp_chart(s, ng, L)
    return jc.jet_from_chart(X, Y, d, d)


def main():
    print("=" * 86)
    print("H4 Stage 1 — does the CALIBRATED jet classifier read a Whitney fold+cusp as an A3 cusp?")
    print("=" * 86)

    print("\n(1) THE WHITNEY CUSP  F(x,y)=(x, y^3 + x*y):  fold = {x=-3y^2}, one A3 cusp at the origin.")
    print("    expect: exactly 1 cusp, |c3| BOUNDED away from 0 (A3), and corank-1 (NOT a D4 umbilic).")
    for s in (0.0, 0.2, -0.2):
        phi, c2, c3 = synthetic_cusp(s)
        vals = jc.cusp_c3(phi, c2, c3)
        top = vals[0] if vals else float("nan")
        X, Y, d = synthetic_cusp_chart(s)
        r = jc.corank_from_chart(X, Y, d, d)
        tag = "A3 (c3 bounded)" if (vals and top >= 1.0) else "??"
        print(f"    s={s:+.1f}: #cusps={len(vals)}  max|c3|={top:6.3f}  [{tag}]   "
              f"corank={r['corank']} (s1_min_rel={r['s1_min_rel']:.3f}, want corank-1)")

    print("\n(2) CONTRAST — the same classifier on the A4 swallowtail (c3->0 at h=0) and the D4 umbilic:")
    print("    A4 swallowtail F_h=(x, y^4+h y^2+x y): |c3| at the cusp(s) -> 0 as h -> 0:")
    for h in (-0.40, -0.10, -0.02, 0.0):
        v = jc.cusp_c3(*jc.synthetic_swallowtail(h))
        t = v[0] if v else float("nan")
        print(f"      h={h:+.2f}: max|c3|={t:6.3f}   {'<- A4 (c3->0)' if t < 0.2 else '<- A3-like (bounded)'}")
    print("    D4 hyperbolic umbilic (corank-2 fires at w=0):")
    for w in (0.0, 0.2):
        X, Y, d = jc.synthetic_umbilic(w)
        r = jc.corank_from_chart(X, Y, d, d)
        print(f"      w={w:.1f}: corank={r['corank']}  s1_min_rel={r['s1_min_rel']:.4f}   "
              f"{'<- D4 (corank-2)' if r['corank'] == 2 else '<- corank-1'}")

    # ---- verdict ---- #
    phi, c2, c3 = synthetic_cusp(0.0)
    cusp_vals = jc.cusp_c3(phi, c2, c3)
    Xc, Yc, dc = synthetic_cusp_chart(0.0)
    cusp_rank = jc.corank_from_chart(Xc, Yc, dc, dc)["corank"]
    # A4 separation = the swallowtail's |c3| COLLAPSES toward 0 as h->0 (vs the cusp's STABLE bounded value);
    # the exact merge h=0 returns nan (no distinct cusp), so use the collapsing ratio near the merge.
    a4_gen = jc.cusp_c3(*jc.synthetic_swallowtail(-0.40))[0]
    a4_near = jc.cusp_c3(*jc.synthetic_swallowtail(-0.02))[0]
    a4_ratio = a4_near / a4_gen
    cusp_stable = max(abs(jc.cusp_c3(*synthetic_cusp(s))[0] - 6.0) for s in (0.0, 0.2, -0.2))  # A3 |c3| stays ~6

    one_cusp = len(cusp_vals) == 1
    bounded = bool(cusp_vals) and cusp_vals[0] >= 1.0
    corank1 = cusp_rank == 1
    a4_separates = a4_ratio < 0.30 and cusp_stable < 0.5   # swallowtail collapses, Whitney cusp stable
    ok = one_cusp and bounded and corank1 and a4_separates

    print("\n" + "=" * 86)
    print("GATES")
    print("=" * 86)
    print(f"  [{'PASS' if one_cusp else 'FAIL'}] the Whitney cusp chart has exactly ONE cusp (#={len(cusp_vals)})")
    print(f"  [{'PASS' if bounded else 'FAIL'}] |c3| at the cusp is BOUNDED away from 0 (A3): {cusp_vals[0] if cusp_vals else float('nan'):.3f} >= 1.0")
    print(f"  [{'PASS' if corank1 else 'FAIL'}] the cusp is corank-1 (a cuspoid, NOT a D4 umbilic): corank={cusp_rank}")
    print(f"  [{'PASS' if a4_separates else 'FAIL'}] A4 separation: swallowtail |c3| collapses (ratio {a4_ratio:.2f} < 0.30) "
          f"while the Whitney cusp stays at 6.0 (max drift {cusp_stable:.3f} < 0.5)")
    print("\n" + "=" * 86)
    if ok:
        print("VERDICT: the calibrated jet classifier correctly reads the Whitney fold+cusp as an A3 cusp")
        print("  (c3 bounded, corank-1), distinct from the A4 swallowtail (c3->0) and the D4 umbilic")
        print("  (corank-2). The A2/A3/A4/D4 synthetic-control ladder is closed; the cusp TEMPLATE for the")
        print("  grokking chart (Stage 2) is established.")
    else:
        print("VERDICT: FAIL — the classifier does not cleanly read the canonical cusp; fix before Stage 2.")
    print("=" * 86)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
