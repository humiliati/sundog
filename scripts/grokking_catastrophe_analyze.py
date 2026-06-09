#!/usr/bin/env python
"""H4 Stage 2 analysis — build the grokking equilibrium chart and point the calibrated jet classifier at
it, against the LOCKED pre-registration (docs/atlas/H4_GROKKING_CATASTROPHE_PREREG.md).

Reads results/atlas/h4/grok_sweep.json (the frac x wd x seed sweep). Produces:
  1. the mean order-parameter surface z*(frac,wd) and the GROK-FRACTION (basin-occupancy) map
     g(frac,wd) = fraction of seeds that grok (test>0.5) -- the BISTABILITY map;
  2. the BIMODAL cell list (a seed >0.7 AND a seed <0.3 in the same cell) -- the cusp's fingerprint;
  3. a control->observable CHART F(frac,wd) = (test_acc, -log_wnorm) interpolated to a fine grid, fed to
     the calibrated jet classifier (atlas_jet_classify) -- caustic (det DF=0)? cusp (A3)? corank-2 (D4)?
  4. the pre-registered verdict: CUSP (A3, bistable wedge + classifier reads a cusp) / FOLD (A2, strip,
     no cusp) / NULL (no bimodality, smooth crossover).

NOT public-eligible. Run: python scripts/grokking_catastrophe_analyze.py
"""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import atlas_jet_classify as jc

SWEEP = Path("results/atlas/h4/grok_sweep.json")


def load_surface():
    d = json.load(open(SWEEP))
    fracs = d["frac_grid"]; wds = d["wd_grid"]; seeds = d["seeds"]
    nf, nw = len(fracs), len(wds)
    test = np.full((nf, nw, len(seeds)), np.nan)
    wnorm = np.full((nf, nw, len(seeds)), np.nan)
    fi = {f: i for i, f in enumerate(fracs)}; wi = {w: j for j, w in enumerate(wds)}
    si = {s: k for k, s in enumerate(seeds)}
    for r in d["results"]:
        i, j, k = fi[r["frac"]], wi[r["wd"]], si[r["seed"]]
        test[i, j, k] = r["test"]; wnorm[i, j, k] = r["wnorm"]
    return d, np.array(fracs), np.array(wds), np.array(seeds), test, wnorm


def main():
    if not SWEEP.exists():
        print("no sweep JSON yet"); return 1
    d, fracs, wds, seeds, test, wnorm = load_surface()
    ncomplete = int(np.sum(~np.isnan(test).any(axis=2)))
    nf, nw = len(fracs), len(wds)
    print(f"H4 grokking-catastrophe analysis: {ncomplete}/{nf*nw} cells complete "
          f"({np.sum(~np.isnan(test))} of {nf*nw*len(seeds)} runs)\n")

    zmean = np.nanmean(test, axis=2)                          # order parameter surface
    grokfrac = np.nanmean((test > 0.5).astype(float), axis=2)  # basin-occupancy / bistability map
    logw = np.log10(np.nanmean(wnorm, axis=2))

    print("z* = mean test-acc (rows=frac, cols=wd):")
    print("  frac\\wd " + " ".join(f"{w:5.2f}" for w in wds))
    for i, f in enumerate(fracs):
        print(f"  {f:5.3f}  " + " ".join(f"{zmean[i,j]:5.2f}" for j in range(nw)))
    print("\ngrok-fraction g (seeds that grok; 0<g<1 = BISTABLE):")
    print("  frac\\wd " + " ".join(f"{w:5.2f}" for w in wds))
    for i, f in enumerate(fracs):
        print(f"  {f:5.3f}  " + " ".join(f"{grokfrac[i,j]:5.2f}" for j in range(nw)))

    # ---- bimodality (the cusp fingerprint) ---- #
    bimodal = []
    for i in range(nf):
        for j in range(nw):
            v = test[i, j][~np.isnan(test[i, j])]
            if len(v) >= 3 and v.max() > 0.7 and v.min() < 0.3:
                bimodal.append((fracs[i], wds[j], sorted(round(float(x), 2) for x in v)))
    print(f"\nBIMODAL cells (a seed>0.7 AND a seed<0.3 -> bistability): {len(bimodal)}")
    for f, w, v in bimodal:
        print(f"  frac={f:.3f} wd={w:.2f}: {v}")
    bistable_region = int(np.sum((grokfrac > 0.05) & (grokfrac < 0.95)))
    print(f"intermediate-g cells (0.05<g<0.95): {bistable_region}")

    # ---- the chart for the classifier (only meaningful if the surface is complete enough) ---- #
    chart_done = ncomplete >= nf * nw
    if chart_done:
        from scipy.interpolate import RegularGridInterpolator
        # coords: frac (linear) x log10(wd) (the geometric axis -> even)
        lw = np.log10(wds)
        # observables: test-acc and -log_wnorm (memorization axis), both normalized to [0,1]
        z1 = zmean
        z2 = (logw - np.nanmin(logw)) / (np.nanmax(logw) - np.nanmin(logw) + 1e-9)
        ng = 160
        fg = np.linspace(fracs[0], fracs[-1], ng); wg = np.linspace(lw[0], lw[-1], ng)
        FG, WG = np.meshgrid(fg, wg, indexing="ij")
        pts = np.stack([FG.ravel(), WG.ravel()], -1)
        X = RegularGridInterpolator((fracs, lw), z1)(pts).reshape(ng, ng)
        Y = RegularGridInterpolator((fracs, lw), z2)(pts).reshape(ng, ng)
        dF = (fracs[-1] - fracs[0]) / (ng - 1); dW = (lw[-1] - lw[0]) / (ng - 1)
        phi, c2, c3 = jc.jet_from_chart(X, Y, dF, dW)
        cusps = jc.cusp_c3(phi, c2, c3)
        rank = jc.corank_from_chart(X, Y, dF, dW)
        has_caustic = bool(np.any(np.abs(np.diff(np.sign(phi), axis=0))) or
                           np.any(np.abs(np.diff(np.sign(phi), axis=1))))
        print("\n--- jet classifier on the control->(test, -log_wnorm) chart (interpolated 160x160) ---")
        print(f"  caustic (det DF sign-change) present: {has_caustic}")
        print(f"  #cusps={len(cusps)}  |c3| at cusps = {[round(v,3) for v in cusps[:6]]}")
        print(f"  corank detector: corank={rank['corank']}  s1_min_rel={rank['s1_min_rel']:.4f}")
    else:
        print(f"\n[chart deferred: need all {nf*nw} cells, have {ncomplete}]")

    # ---- preliminary pre-registered verdict ---- #
    print("\n" + "=" * 80)
    if ncomplete < nf * nw:
        print(f"PRELIMINARY ({ncomplete}/{nf*nw} cells). Re-run when the sweep completes.")
    if len(bimodal) == 0 and bistable_region <= 1:
        print("LEANING: NULL — no bistability (no bimodal cells, g essentially in {0,1}); grokking here")
        print("  looks like a SINGLE-VALUED transition (sharp or smooth), not a bistable cusp.")
    elif len(bimodal) >= 2:
        print(f"LEANING: BISTABLE — {len(bimodal)} bimodal cells; check whether the bistable region is")
        print("  WEDGE-shaped (cusp) vs a uniform strip (fold), and whether the chart reads a cusp.")
    else:
        print("LEANING: borderline — few bimodal cells; needs the full surface + chart.")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
