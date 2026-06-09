#!/usr/bin/env python
"""Frozen test for H7 (scripts/riemann_gappair.py) — the Riemann gap-pair caustic probe. Locks the
height-resolved NULL-A: the zeros unfold to unit mean spacing; the consecutive-gap anti-correlation is
HEIGHT-DEPENDENT and EXTRAPOLATES (corr = C∞ + A/logγ) to the GUE value; a fast GUE control confirms the
limit; Poisson is uncorrelated; and the spectral form factor shows the GUE ramp toward the τ=1 caustic.
Run: python scripts/test_riemann_gappair.py
"""
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import riemann_gappair as r   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H7 — the Riemann gap-pair caustic (height-resolved NULL-A):\n")

w = r.load_unfolded()
g = np.diff(w)
gamma = np.array([float(x) for x in r.ZEROS.read_text().split()])

check("unfolding: 100k zeros → unit mean spacing", len(w) == 100000 and abs(g.mean() - 1.0) < 1e-3,
      f"N={len(w)}, mean gap={g.mean():.4f}")

cz = float(np.corrcoef(g[:-1], g[1:])[0, 1])
check("static consecutive-gap anti-correlation ≈ −0.357 (finite-height-averaged)", abs(cz + 0.357) < 0.01,
      f"corr={cz:+.3f}")

# height-resolution: corr(γ) = C∞ + A/logγ; A<0 (weakens with height), C∞ ≈ GUE value
xs, ys = [], []
nbk = 20
for k in range(nbk):
    lo, hi = k * len(g) // nbk, (k + 1) * len(g) // nbk
    gm = np.exp(np.mean(np.log(gamma[lo:hi])))
    seg = g[lo:hi]
    ys.append(float(np.corrcoef(seg[:-1], seg[1:])[0, 1])); xs.append(1.0 / np.log(gm))
A, Cinf = np.polyfit(xs, ys, 1)
check("anti-correlation is HEIGHT-DEPENDENT, weakening with height (A<0)", A < -0.2, f"A={A:+.3f}")
check("extrapolated limit C∞ ≈ −0.30 (the GUE value) — the excess is the ~1/logγ finite-height correction",
      abs(Cinf + 0.30) < 0.03, f"C∞={Cinf:+.4f}")

# fast GUE control (a few eigvalsh blocks) confirms the GUE consecutive-gap correlation ≈ −0.31
gue = r.cue_blocks(6)
gc = np.mean([np.corrcoef(np.diff(b)[:-1], np.diff(b)[1:])[0, 1] for b in gue])
check("GUE control consecutive-gap correlation ≈ −0.31 (matches the zeros' extrapolated limit)",
      abs(gc + 0.31) < 0.03, f"GUE={gc:+.3f}  (zeros→{Cinf:+.3f})")

# Poisson control: independent gaps → ~0 correlation
poi = r.poisson_blocks(6)
pc = np.mean([np.corrcoef(np.diff(b)[:-1], np.diff(b)[1:])[0, 1] for b in poi])
check("Poisson control ≈ 0 (independent gaps)", abs(pc) < 0.03, f"Poisson={pc:+.3f}")

# the τ=1 caustic: the form factor ramps up toward τ=1 (GUE ramp K≈τ)
K = r.form_factor([w[i * r.L:(i + 1) * r.L] for i in range(20)])
ramp = float(np.polyfit(r.TAU[(r.TAU >= 0.3) & (r.TAU <= 0.9)], K[(r.TAU >= 0.3) & (r.TAU <= 0.9)], 1)[0])
check("the τ=1 CAUSTIC: the spectral form factor shows the GUE ramp K≈τ (positive ramp slope)",
      ramp > 0.4, f"ramp slope={ramp:+.2f}")

print(f"\n{'ALL PASS — the zeros are GUE-universal: the gap-pair caustic is the universal τ=1 form-factor kink, and the low-zero anti-correlation excess is the known ~1/logγ finite-height arithmetic correction, extrapolating home to GUE. NULL-A, height-resolved.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
