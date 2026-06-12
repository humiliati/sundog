#!/usr/bin/env python
"""H8-RF POST-VERDICT DESCRIPTIVE DIAGNOSTICS — informs the NAMED FOLLOW-UP only.

The registered verdict of scripts/double_descent_cusp_rf.py is K2 (germ-indeterminate: |c3| locus ratio
0.475 in the pre-registered [0.25,0.5) band at the tau2=0.4 locus end, escalation-stable). NOTHING in this
file amends that verdict — the prereg (docs/atlas/H8_RF_CUSP_PREREG.md s5) forbids softer verdicts from
the band. This script measures, descriptively:
  (D1) the K3 fold-pair scaling battery on the primary slice (never reached — adjudication stopped at K2);
  (D2) per-slice SELF-normalized |c3| ratios (each chart normalized by its OWN window's R mean/std) — the
       test of the named follow-up hypothesis that the monotone |c3|(tau2) trend under the SHARED
       normalization is a risk-scale artifact, not germ structure.
Run: python scripts/h8_rf_diag_postverdict.py
"""
import sys
from pathlib import Path
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
sys.path.insert(0, str(Path(__file__).resolve().parent))
import double_descent_cusp_rf as dd


def main():
    print("H8-RF post-verdict diagnostics (DESCRIPTIVE ONLY — registered verdict: K2, unchanged)\n")
    slices = [w for t2 in dd.SLICES if (w := dd.slice_windows(t2)) is not None
              and w["lbar_c"] >= dd.LBAR_C_MIN]
    prim = next(w for w in slices if w["tau2"] == dd.PRIMARY)
    lc = prim["lbar_c"]
    print(f"(D1) K3 scaling battery, primary slice tau2={dd.PRIMARY} (descriptive):")

    def crit_at(lb):
        return dd.census(dd.PSI2, dd.PRIMARY, lb)[0]
    slope, r2, npts = dd.scaling_fit(crit_at, lc, 2)
    print(f"     lbar_c={lc:.6f}  slope={slope:.4f}  R2={r2:.5f}  pts={npts}"
          f"   (A3 normal form predicts 1/2; descriptive — K3 was never adjudicated)\n")

    print("(D2) per-slice SELF-normalized |c3| (each chart normalized by its own window):")
    vals = {}
    for w in slices:
        pre = dd.rf_chart(dd.PSI2, w["tau2"], w["p_span"], w["l_span"],
                          (0.0, 1.0, w["l_span"][0], w["l_span"][1]))
        r = dd.rf_chart(dd.PSI2, w["tau2"], w["p_span"], w["l_span"],
                        (pre["R_mean"], pre["R_std"], w["l_span"][0], w["l_span"][1]))
        vals[w["tau2"]] = r["c3"][0] if r["c3"] else float("nan")
        print(f"     tau2={w['tau2']}: #cusps={r['n_cusps']} self-norm |c3|={r['c3']} corank={r['corank']}")
    med = float(np.median(list(vals.values())))
    ratios = {k: round(v / med, 3) for k, v in vals.items()}
    print(f"     self-normalized median={med:.4f}; ratios={ratios}")
    print(f"     min self-normalized ratio = {min(ratios.values()):.3f} "
          f"(shared-normalization min was 0.475; if this is >= 0.5 and the spread tightens, the band hit")
    print("      was a normalization-scale artifact -> the named follow-up's hypothesis gains support).")
    print("\nRegistered verdict REMAINS: K2 — finite merge of unresolved germ class (band rule, prereg s5).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
