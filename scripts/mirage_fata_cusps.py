#!/usr/bin/env python
"""H9 (research leg) — resolving + classifying the multi-cusp Fata-Morgana caustic. Closes the one PARTIAL
gate in the H5 mirage-ladder slate (P4): the double-inversion (Fata-Morgana) 5-image case is born from TWO
A3 cusp births (1->3 at Δn≈2e-5, 3->5 at Δn≈3e-5), but the H5 jet classifier's isolated-cusp detector only
RESOLVED ONE (they were close in (θ,Δn) and ndimage.label merged them).

This file (a) RESOLVES both cusps (finer grid + per-Δn-band localization), classifies each via the
calibrated iterated-kernel discriminant (A3 iff |c3| bounded away from 0; A4 iff c3->0; corank for D4);
then (b) sweeps the inversion-layer GAP and tests whether the two A3 cusps COALESCE into an A4 swallowtail
(c3->0 at a merge gap) — the Fata-Morgana metamorphosis. Honest report: two-A3 vs an A4 merge.

FROZEN-INTERNAL research (catastrophe vocabulary) — NOT public-eligible (the public Fata-Morgana explainer
is textbook-only). Reuses the FLAT H5 model (scripts/mirage_ladder.py) + the calibrated jet classifier.
Run: python scripts/mirage_fata_cusps.py
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

H0 = ml.H0                       # inversion centre [m]
R = ml.R_DEFAULT                 # object range [m]
H_OBS = ml.H_OBS


def double_grad(gap):
    """gradient closure for two inversion layers at h0±gap/2, callable as gradfn(h, dn, ss, h0)."""
    def g(h, dn, ss, h0):
        return (-(dn / (2 * ss)) / np.cosh((h - (h0 - gap / 2)) / ss) ** 2
                - (dn / (2 * ss)) / np.cosh((h - (h0 + gap / 2)) / ss) ** 2)
    return g


def fata_chart(gradfn, dn_lo, dn_hi, s=12.0, ng=300):
    """Non-dimensionalized (θ,Δn) chart F=(Δn, h_target) for a double-inversion gradient, fed to the jet
    classifier. Returns (n_cusps, c3_list, corank, s1_min_rel)."""
    g0 = (H0 - H_OBS) / R
    th = np.linspace(g0 - 0.0045, g0 + 0.0045, ng)
    dn = np.linspace(dn_lo, dn_hi, ng)
    Hgrid = np.zeros((ng, ng)); good = np.zeros((ng, ng), bool)
    for j, d in enumerate(dn):
        ht, minh = ml.trace(th, d, s, R, H_OBS, h0=H0, gradfn=gradfn)
        Hgrid[:, j] = ht
        good[:, j] = (minh > 0) & (ht > 0)
    Xn = ((dn - dn[0]) / (dn[-1] - dn[0]))[None, :] * np.ones((ng, 1))
    hs = np.nanstd(Hgrid[good]) if good.any() else 1.0
    Yn = np.where(good, (Hgrid - np.nanmean(Hgrid[good] if good.any() else Hgrid)) / (hs + 1e-30), 0.0)
    d = 1.0 / (ng - 1)
    phi, c2, c3 = jc.jet_from_chart(Xn, Yn, d, d)
    cusps = jc.cusp_c3(phi, c2, c3, good, edge=3)
    rank = jc.corank_from_chart(Xn, Yn, d, d, good=good, edge=3)
    corank = rank["corank"] if rank else 1
    s1m = round(rank["s1_min_rel"], 4) if rank else float("nan")
    return len(cusps), [round(v, 2) for v in cusps], corank, s1m


def resolve_two_cusps(gradfn, s=12.0):
    """Resolve the two cusps by localizing one per Δn-band (lower birth ~2e-5, upper ~3e-5)."""
    lo = fata_chart(gradfn, 1.6e-5, 2.6e-5, s)        # band around the 1->3 birth
    hi = fata_chart(gradfn, 2.6e-5, 3.8e-5, s)        # band around the 3->5 birth
    return lo, hi


def main():
    print("=" * 88)
    print("H9 — resolving + classifying the multi-cusp Fata-Morgana caustic (closes H5 P4)")
    print("=" * 88)

    # (1) the H5 single-pass detector (reference): how many cusps over the full two-birth range?
    g60 = double_grad(60.0)
    nfull, c3full, ck, s1 = fata_chart(g60, 1.6e-5, 3.8e-5)
    print(f"\n(1) full-range chart (the H5 setup): #cusps={nfull}  |c3|={c3full}  corank={ck}"
          f"  (H5 found only 1 — the two births merged in the detector)")

    # (2) RESOLVE both cusps by per-Δn-band localization
    lo, hi = resolve_two_cusps(g60)
    print("\n(2) per-Δn-band localization (gap=60 m) — resolve EACH cusp:")
    print(f"    lower band Δn∈[1.6,2.6]e-5 (1->3 birth): #cusps={lo[0]}  |c3|={lo[1]}  corank={lo[2]}")
    print(f"    upper band Δn∈[2.6,3.8]e-5 (3->5 birth): #cusps={hi[0]}  |c3|={hi[1]}  corank={hi[2]}")
    both_a3 = (lo[0] >= 1 and hi[0] >= 1 and lo[1] and hi[1]
               and lo[1][0] > 0.3 and hi[1][0] > 0.3 and lo[2] == 1 and hi[2] == 1)
    print(f"    => both births resolved as A3 cusps (|c3| bounded, corank-1): {both_a3}")

    # (3) GAP sweep — do the two A3 cusps COALESCE into an A4 swallowtail (c3->0) as the layers merge?
    print("\n(3) inversion-GAP sweep — does the Fata-Morgana caustic merge to an A4 swallowtail (c3->0)?")
    print(f"    {'gap [m]':>8} {'lower |c3|':>11} {'upper |c3|':>11}   note")
    refc3 = max(lo[1][0] if lo[1] else 1, hi[1][0] if hi[1] else 1)
    gap_curve = []
    for gap in (80, 60, 45, 32, 22, 14, 8):
        gg = double_grad(float(gap))
        lo_g = fata_chart(gg, 1.6e-5, 2.8e-5)
        hi_g = fata_chart(gg, 2.6e-5, 4.2e-5)
        lc = lo_g[1][0] if lo_g[1] else float("nan")
        hc = hi_g[1][0] if hi_g[1] else float("nan")
        mn = np.nanmin([lc, hc])
        gap_curve.append((gap, lc, hc, mn))
        tag = ("<- A4 merge (c3->0)" if mn < 0.25 * refc3 else
               ("single cusp / merged" if (lo_g[0] == 0 or hi_g[0] == 0) else "two A3 cusps"))
        print(f"    {gap:>8} {lc:>11.2f} {hc:>11.2f}   {tag}")
    merged = [g for g in gap_curve if np.isfinite(g[3]) and g[3] < 0.25 * refc3]
    a4_found = len(merged) > 0

    print("\n" + "=" * 88)
    print("VERDICT")
    print("=" * 88)
    if both_a3:
        print("RESOLVED: the 5-image Fata-Morgana caustic is TWO STACKED A3 CUSPS (each |c3| bounded away")
        print("  from 0, corank-1) — one per inversion layer (the 1->3 and 3->5 image births). The H5 P4")
        print("  PARTIAL is closed: it was a detector-resolution limit (two close cusps merged into one),")
        print("  not a missing catastrophe.")
    else:
        print("PARTIAL still — band localization did not cleanly resolve both A3 cusps; inspect above.")
    if a4_found:
        print(f"  GAP metamorphosis: as the two inversion layers approach (gap≈{merged[-1][0]} m), the two A3")
        print("  cusps COALESCE into an A4 SWALLOWTAIL (|c3|->0) — a genuine higher-catastrophe merge.")
    else:
        print("  GAP sweep: the two A3 cusps do NOT coalesce into an A4 within the swept gaps (they stay")
        print("  distinct A3, or annihilate/duct out) — no swallowtail; the Fata-Morgana stays two-A3.")
    print("=" * 88)
    return 0 if both_a3 else 1


if __name__ == "__main__":
    sys.exit(main())
