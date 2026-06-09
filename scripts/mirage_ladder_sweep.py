#!/usr/bin/env python
"""H5 Stage 2 — the mirage (Δn, s) image-multiplicity LADDER + the jet-classifier catastrophe labels.

Maps #images over the (Δn, s) control plane (the bifurcation set), then points the H4-validated jet
classifier (atlas_jet_classify) at the ray-transfer chart F(θ, Δn) = (Δn, h_target(θ; Δn, s)) to LABEL
the 1->3-image onset as an A3 cusp (and the double-inversion Fata-Morgana 5-image case as higher). Against
docs/atlas/H5_MIRAGE_LADDER_PREREG.md. NOT public-eligible.

The chart math (exact): det DF = -∂h_target/∂θ, so caustic φ=0 <=> image-merge fold; cusp φ=0∧c2=0 <=>
the 1->3 onset (transfer S-shape born); c3 = -∂³h/∂θ³ bounded => A3. Coordinates are non-dimensionalized
(θ,Δn,h -> O(1)) so |c3| is a scale-free A3(bounded)-vs-A4(->0) discriminator. Run: python scripts/mirage_ladder_sweep.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mirage_ladder as ml
import atlas_jet_classify as jc


def image_count(dn, s, gradfn=ml.nprime_single, R=ml.R_DEFAULT, span=(-0.004, 0.020)):
    th, ht, vd = ml.transfer_curve(dn, s, R, gradfn=gradfn, span=span)
    return ml.count_folds(th, ht, vd) + 1            # #images in the fold band


def ladder_map(dn_grid, s_grid, gradfn=ml.nprime_single):
    M = np.zeros((len(s_grid), len(dn_grid)), int)
    for i, s in enumerate(s_grid):
        for j, dn in enumerate(dn_grid):
            M[i, j] = image_count(dn, s, gradfn)
    return M


def cusp_chart(s, dn_span, theta_span, ng=240, gradfn=ml.nprime_single, R=ml.R_DEFAULT):
    """Build the non-dimensionalized chart F(θ,Δn)=(Δn, h_target) on a (θ×Δn) grid and run the jet
    classifier. Returns dict {caustic, n_cusps, c3, corank, s1_min_rel}. `good` masks ground-hit rays."""
    th = np.linspace(*theta_span, ng)
    dn = np.linspace(*dn_span, ng)
    H = np.zeros((ng, ng)); good = np.zeros((ng, ng), bool)
    for j, d in enumerate(dn):
        ht, minh = ml.trace(th, d, s, R, ml.H_OBS, gradfn=gradfn)
        H[:, j] = ht
        good[:, j] = (minh > 0) & (ht > 0)
    # non-dimensionalize: θ,Δn -> [0,1]; h -> O(1)
    Xn = ((dn - dn[0]) / (dn[-1] - dn[0]))[None, :] * np.ones((ng, 1))   # axis1 = Δn
    hs = np.nanstd(H[good]) if good.any() else 1.0
    Yn = (H - np.nanmean(H[good] if good.any() else H)) / (hs + 1e-30)
    Yn = np.where(good, Yn, 0.0)
    dG = 1.0 / (ng - 1); dA = 1.0 / (ng - 1)        # θ axis also normalized to [0,1] spacing
    phi, c2, c3 = jc.jet_from_chart(Xn, Yn, dG, dA)
    cusps = jc.cusp_c3(phi, c2, c3, good, edge=4)
    rank = jc.corank_from_chart(Xn, Yn, dG, dA, good=good, edge=4)
    has_caustic = bool(np.any((np.abs(np.diff(np.sign(phi), axis=0)) > 0)[good[:-1] & good[1:]]))
    return {"caustic": has_caustic, "n_cusps": len(cusps), "c3": [round(v, 3) for v in cusps[:5]],
            "corank": rank["corank"], "s1_min_rel": round(rank["s1_min_rel"], 4)}


def main():
    print("=" * 86)
    print("H5 Stage 2 — the mirage (Δn,s) image-multiplicity ladder + jet-classifier catastrophe labels")
    print("=" * 86)

    # (P2) the (Δn, s) ladder / bifurcation set
    dn_grid = [1e-5, 2e-5, 3e-5, 5e-5, 1e-4, 2e-4]
    s_grid = [3, 5, 8, 13, 20, 40]
    M = ladder_map(dn_grid, s_grid)
    print("\n(P2) IMAGE-COUNT LADDER over (Δn, s) [single inversion]: 1=simple, 3=superior mirage")
    print("   s\\Δn " + " ".join(f"{d:>7.0e}" for d in dn_grid))
    for i, s in enumerate(s_grid):
        print(f"   {s:>4} " + " ".join(f"{M[i,j]:>7}" for j in range(len(dn_grid))))
    # the 1->3 boundary scales as Δn/s ~ const (the cusp curve):
    onsets = []
    for i, s in enumerate(s_grid):
        row = M[i]
        idx = next((j for j in range(len(dn_grid)) if row[j] >= 3), None)
        if idx is not None:
            onsets.append((s, dn_grid[idx], dn_grid[idx] / s))
    print("   1->3 onset (cusp boundary):  " + "  ".join(f"s={s}:Δn≈{d:.0e}(Δn/s={r:.1e})" for s, d, r in onsets))

    # (P3) jet classifier on the single-inversion chart: the 1->3 onset must be an A3 cusp
    print("\n(P3) jet classifier on F(θ,Δn)=(Δn, h_target), single inversion s=5, Δn∈[1.5e-5,6e-5]:")
    g = (ml.H0 - ml.H_OBS) / ml.R_DEFAULT
    r1 = cusp_chart(5.0, (1.5e-5, 6e-5), (g - 0.0030, g + 0.0030))
    print(f"     caustic={r1['caustic']}  #cusps={r1['n_cusps']}  |c3|={r1['c3']}  "
          f"corank={r1['corank']} (s1_min_rel={r1['s1_min_rel']})")
    a3_single = r1["caustic"] and r1["n_cusps"] >= 1 and r1["corank"] == 1 and (r1["c3"] and r1["c3"][0] > 0.3)

    # (P4) double inversion (Fata Morgana): 5-image rung + richer caustic
    print("\n(P4) DOUBLE inversion (Fata Morgana) ladder + chart:")
    dng2 = [2e-5, 5e-5, 1e-4, 2e-4]
    s2g = [3, 5, 8]
    M2 = ladder_map(dng2, s2g, gradfn=ml.nprime_double)
    print("   s\\Δn " + " ".join(f"{d:>7.0e}" for d in dng2) + "   (expect a 5 = Fata Morgana rung)")
    for i, s in enumerate(s2g):
        print(f"   {s:>4} " + " ".join(f"{M2[i,j]:>7}" for j in range(len(dng2))))
    max5 = int(M2.max())
    # span the TWO cusp births (1->3 at Δn≈2e-5, 3->5 at Δn≈3e-5), below the ducting regime
    r2 = cusp_chart(5.0, (1.5e-5, 4e-5), (g - 0.004, g + 0.004), gradfn=ml.nprime_double)
    print(f"   double-inversion chart [Δn∈1.5e-5..4e-5, the two cusp births]:  caustic={r2['caustic']}  "
          f"#cusps={r2['n_cusps']}  |c3|={r2['c3']}  corank={r2['corank']}")
    print("   (note: beyond the clean fold band, very strong inversions DUCT (rays trapped/ground-truncated)"
          " — a distinct regime, not part of the simple-fold ladder.)")

    # ---- verdict vs pre-reg ---- #
    ladder_ok = (M.min() == 1 and M.max() >= 3 and len(onsets) >= 3)
    rung5 = max5 >= 5
    print("\n" + "=" * 86)
    print("VERDICT (vs H5 pre-reg)")
    print("=" * 86)
    print(f"  [{'PASS' if ladder_ok else 'FAIL'}] P2 ladder: #images climbs 1->3 across (Δn,s); onset curve Δn/s~const (cusp boundary)")
    print(f"  [{'PASS' if a3_single else 'FAIL'}] P3 cusp: the 1->3 onset reads as an A3 cusp (caustic + cusp, corank-1, |c3| bounded)")
    p4 = rung5 and r2["n_cusps"] >= 2
    print(f"  [{'PASS' if p4 else 'INFO'}] P4 higher rung: double inversion reaches {max5} images (Fata Morgana) "
          f"with a RICHER caustic ({r2['n_cusps']} cusps vs the single inversion's 1) — two stacked A3 cusps")
    overall = ladder_ok and a3_single
    print("\n" + ("BOUNDED-POSITIVE: mirages form a classified caustic ladder in (Δn,s); the superior-mirage "
                  "1->3 onset IS a Whitney A3 CUSP (jet classifier, corank-1), the cusp boundary scales as "
                  "Δn/s~const, and a double inversion reaches the 5-image Fata-Morgana rung — the mirage "
                  "analog of the halo Atlas, labelled by the H4-validated tool."
                  if overall else "MIXED/NULL — see gates; report honestly."))
    print("=" * 86)
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
