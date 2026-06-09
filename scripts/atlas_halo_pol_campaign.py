#!/usr/bin/env python
"""Measured-sky LINEAR-pol halo campaign — feasibility + falsification-design scorecard.

Promotes the Atlas DoP(R) observable (`atlas_halo_polarization.py`) toward a measured-sky campaign by
computing, per halo: predicted DoP, detectability vs real instrument floors, angular ISOLATION (is the
ring resolvable / non-degenerate?), and the confound-robust DIFFERENTIAL test the campaign should run.

Grounded by a 2026-06-09 lit/instrument workflow (citations in ATLAS_HALO_POL_CAMPAIGN.md). Key facts:
  - Measured today: ONLY the 22-deg halo/parhelion linear pol (Können 1991/2003; Pust & Shaw 2008).
    The 46-deg halo + the entire pyramidal odd-radius family are UNMEASURED -> the campaign's novelty.
  - DoP(R) is the geometric-optics FLOOR/ORDERING (birefringence-diffraction rides ABOVE it: Können's
    intrinsic 22-deg peak ~8.7% vs the 3.7% Fresnel floor). The robust, falsifiable claim is the
    MONOTONE ORDERING vs radius, NOT an absolute peak value.
  - Linear-pol DoFP polarimeters (specMACS / Sony IMX250MZR) measure linear Stokes natively -> NO
    linear<->circular V crosstalk: the linear campaign is FAR more feasible than the circular-V one.
  - Background: halo (radial, +Q) and sky (tangential, -Q) are BOTH U=0 -> orthogonality buys nothing;
    the ring's spatial NARROWNESS + Können's annular background fit is what isolates it. U=0 is a NULL
    veto channel (radial sign of Q), not background rejection.

In-house campaign PREP only. The actual measurement (collaboration + instrument + cirrus season) and any
outreach stay OWNER-GATED / external (Stage C scope-and-hold). NOT public-eligible.
Run:  python scripts/atlas_halo_pol_campaign.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so  # noqa: E402

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# The Atlas refraction halos (name, radius_deg, kind). Radii from halo_min_deviation / the Atlas.
HALOS = [
    ("9-deg pyramidal", 8.96, "pyramidal"),
    ("18-deg pyramidal", 18.0, "pyramidal"),
    ("20-deg pyramidal", 20.0, "pyramidal"),
    ("22-deg (prism)", float(so.halo_min_deviation(60)), "regular"),
    ("23-deg pyramidal", 23.0, "pyramidal"),
    ("24-deg pyramidal", 23.82, "pyramidal"),
    ("35-deg pyramidal", 35.0, "pyramidal"),
    ("46-deg (prism+basal)", float(so.halo_min_deviation(90)), "regular"),
]

# Instrument linear-pol DoP floors (grounded; see ATLAS_HALO_POL_CAMPAIGN.md).
INSTRUMENTS = {
    "DoFP single-shot (IMX250MZR)": 0.008,     # 0.8% photon-noise floor, one frame
    "DoFP 100-frame stack": 0.0008,            # 0.8%/sqrt(100)
    "specMACS swath (LUCID DoFP)": 0.05,       # ~5% relative DoLP budget (aircraft)
    "Shaw all-sky (LCVR, full Stokes)": 0.005,  # ~0.5% absolute DoLP floor (best; rooftop archive)
}

# angular separation below which two rings are not cleanly resolvable / are pol-degenerate
ISO_SEP_DEG = 3.0
# the chosen reference floor for the angular-degeneracy DoP-gap test (commodity DoFP single-shot)
REF_FLOOR = 0.008


def _radii():
    return np.array([R for _, R, _ in HALOS])


def nearest_neighbor_sep(R, radii):
    others = radii[np.abs(radii - R) > 1e-9]
    return float(np.min(np.abs(others - R))) if others.size else float("inf")


def scorecard():
    """Per-halo: predicted DoP, nearest-neighbor separation, isolation flag, detectability tier."""
    radii = _radii()
    rows = []
    for nm, R, kind in HALOS:
        dop = float(so.halo_pol_dop(R))
        sep = nearest_neighbor_sep(R, radii)
        # angularly isolated AND its DoP gap to the nearest neighbor exceeds the DoFP floor
        nn_R = radii[np.argmin(np.where(np.abs(radii - R) > 1e-9, np.abs(radii - R), np.inf))]
        dop_gap = abs(dop - float(so.halo_pol_dop(nn_R)))
        isolated = sep >= ISO_SEP_DEG and dop_gap >= REF_FLOOR
        if dop < REF_FLOOR:
            tier = "archival/stack"             # below single-shot floor
        elif not isolated:
            tier = "degenerate"                 # SNR fine but blended with a neighbor
        elif dop >= 5 * REF_FLOOR:
            tier = "easy"
        else:
            tier = "moderate"
        rows.append(dict(name=nm, R=R, kind=kind, dop=dop, sep=sep, isolated=isolated, tier=tier))
    return rows


# The clean confound-robust differential test = the 3 angularly-isolated radii spanning the range.
CLEAN_TEST_RADII = [8.96, 35.0, float(so.halo_min_deviation(90))]


def differential_test():
    """The pre-registered falsification: monotone DoP across the isolated radii + the predicted ratios
    (calibration-immune; a non-monotone result kills the radius-only law)."""
    dops = [float(so.halo_pol_dop(R)) for R in CLEAN_TEST_RADII]
    ratios = {
        "DoP(46)/DoP(35)": dops[2] / dops[1],
        "DoP(35)/DoP(9)": dops[1] / dops[0],
        "DoP(46)/DoP(9)": dops[2] / dops[0],
    }
    monotone = all(np.diff(dops) > 0)
    return dict(radii=CLEAN_TEST_RADII, dops=dops, ratios=ratios, monotone=monotone)


def _report():
    print("=" * 78)
    print("Measured-sky LINEAR-pol halo campaign — feasibility + falsification scorecard")
    print("=" * 78)
    print("  predicted observable: DoP(R)=(1-cos^4(R/2))/(1+cos^4(R/2)), radial, U=0, no V")
    print("  (the FLOOR/ORDERING law; real peaks ride above via birefringence-diffraction)\n")

    print(f"  {'halo':22s} {'R':>6s} {'DoP':>7s} {'nn-sep':>7s} {'isolated':>9s} {'tier':>14s}")
    for r in scorecard():
        print(f"  {r['name']:22s} {r['R']:6.2f} {r['dop']*100:6.2f}% {r['sep']:6.1f}° "
              f"{('yes' if r['isolated'] else 'no'):>9s} {r['tier']:>14s}")

    print("\n  detectability vs instrument floors (predicted DoP / floor = SNR):")
    for inst, fl in INSTRUMENTS.items():
        easy = sum(1 for _, R, _ in HALOS if so.halo_pol_dop(R) >= fl)
        print(f"    {inst:34s} floor={fl*100:5.2f}%  -> {easy}/8 halos above floor")

    dt = differential_test()
    print("\n  THE CLEAN FALSIFICATION TEST — the 3 angularly-isolated radii (the 22/23/24° cluster is a")
    print("  degenerate trap; discard it as a discriminator):")
    print(f"    radii {[round(R,2) for R in dt['radii']]} deg -> DoP "
          f"{[f'{d*100:.2f}%' for d in dt['dops']]}")
    print("    pre-registered CONFOUND-ROBUST RATIOS (common-mode calibration cancels):")
    for k, v in dt["ratios"].items():
        print(f"       {k} = {v:.2f}")
    print(f"    monotone increase with radius: {dt['monotone']}  -> KILL if a measured ring breaks it")
    print("    + U=0 / radial-sign-of-Q veto on every frame (separates halo from sky-background residual)")
    print("\n  (see docs/atlas/ATLAS_HALO_POL_CAMPAIGN.md for the full campaign plan + citations.")
    print("   In-house prep; the measurement + outreach stay owner-gated/external.)")


if __name__ == "__main__":
    _report()
