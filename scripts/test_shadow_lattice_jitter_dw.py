#!/usr/bin/env python
"""Frozen test for HS10 LATTICE-FRAG (scripts/shadow_lattice_jitter_dw.py).

Runs BEFORE the verdict run (prereg section 2c): pins the GATES and the apparatus contract --
G2 regression (existing pops byte-unchanged), the paired-draw design, the lam=0 estimator anchor
(machine precision), the analytic charFun product law, mask logic, crossing interpolation.
It does NOT pin verdict outcomes; headline numbers are byte-pinned in the banking commit.
Run: python scripts/test_shadow_lattice_jitter_dw.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import shadow_charfun_populations as scp        # noqa: E402
import shadow_lattice_jitter_dw as dw           # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("HS10 LATTICE-FRAG frozen test (gates + apparatus contract):\n")

# ---- G2: regression -- the eps extension leaves existing pops byte-unchanged ---- #
LAMS = [0.0, 0.5, 1.0, 2.0]
lat_c, _ = scp.sweep_pop("lattice", n=400, lams=LAMS)
gau_c, _ = scp.sweep_pop("gaussian", n=400, lams=LAMS)
check("G2 lattice cont byte-equal to banked baseline", lat_c == [0.7289, 0.4391, 0.5373, 0.566],
      f"{lat_c}")
check("G2 gaussian cont byte-equal to banked baseline", gau_c == [0.6939, 0.5059, 0.0, 0.0],
      f"{gau_c}")

# ---- the paired-draw contract: eps is ONLY a multiplier on a shared (lat, gz) stream ---- #
rng1 = np.random.default_rng(123)
lat = rng1.choice([-1.0, 1.0], (5, 7))
gz = rng1.standard_normal((5, 7))
xi0 = scp.draw_pop(np.random.default_rng(123), (5, 7), "lattice_jitter", 0.0)
xi3 = scp.draw_pop(np.random.default_rng(123), (5, 7), "lattice_jitter", 0.3)
check("paired draws: eps=0 equals the lattice component", np.array_equal(xi0, lat))
check("paired draws: eps=0.3 equals lat + 0.3*gz", np.allclose(xi3, lat + 0.3 * gz, atol=0, rtol=0))

# ---- analytic charFun: product law ---- #
s = np.array([0.0, 1.0, np.pi, 2 * np.pi])
check("charfun lattice_jitter(eps=0) == cos(s)",
      np.allclose(scp.charfun_re("lattice_jitter", s, 0.0), np.cos(s)))
check("charfun product law at s=2pi, eps=0.3: cos(2pi)*exp(-(0.6pi)^2/2)",
      abs(scp.charfun_re("lattice_jitter", 2 * np.pi, 0.3)
          - np.cos(2 * np.pi) * np.exp(-(0.3 * 2 * np.pi) ** 2 / 2)) < 1e-12)

# ---- lam=0 estimator anchor: no jitter effect, no wash -> a_emp == a_pred + noise-mean only;
#      with the SAME rng the signal pipelines must agree at machine precision when noise has no
#      systematic projection. Test the exact contract: at lam=0 the fringe IS its expectation. ---- #
a_emp, a_pred = dw.cell(0.0, 0.3, n=200)
check("lam=0 anchor: Re phi(0)=1, fringe exact -> mean residual is pure obs-noise projection (~0)",
      abs(float(a_emp.mean() - a_pred.mean())) < 0.02,
      f"|mean(a_emp - a_pred)| = {abs(float(a_emp.mean() - a_pred.mean())):.5f}")
# noiseless variant: byte-exact equality of the two pipelines at lam=0
_noise = dw.h.NOISE
try:
    dw.h.NOISE = 0.0
    a_emp0, a_pred0 = dw.cell(0.0, 0.3, n=50)
    check("lam=0 noiseless: estimator == analytic pipeline at machine precision",
          float(np.max(np.abs(a_emp0 - a_pred0))) < 1e-10,
          f"max|diff| = {float(np.max(np.abs(a_emp0 - a_pred0))):.2e}")
finally:
    dw.h.NOISE = _noise

# ---- G3 determinism ---- #
b1, _ = dw.cell(1.0, 0.3, n=100)
b2, _ = dw.cell(1.0, 0.3, n=100)
check("G3 cell determinism (byte-identical rerun)", np.array_equal(b1, b2))

# ---- mask logic ---- #
check("Nyquist window: lam_max(0.5) = 8.75/3", abs(dw.lam_max(0.5) - 8.75 / 3.0) < 1e-12)
check("Nyquist window: lam_max(0.01) capped at 5.5", dw.lam_max(0.01) == 5.5)
_, p05 = dw.cell(0.5, 0.0, n=200)
_, p10 = dw.cell(1.0, 0.0, n=200)
check("denominator floor excludes the lam=0.5 recurrence null",
      abs(float(p05.mean())) < dw.DENOM_FLOOR, f"|pred(0.5,0)| = {abs(float(p05.mean())):.4f}")
check("denominator floor keeps the lam=1.0 recurrence point",
      abs(float(p10.mean())) >= dw.DENOM_FLOOR, f"|pred(1.0,0)| = {abs(float(p10.mean())):.4f}")

# ---- crossing interpolation ---- #
rows = [(1.0, 0.8, 0, 0), (2.0, 0.6, 0, 0), (3.0, 0.4, 0, 0)]
check("first_crossing interpolates 0.5 between 2.0 and 3.0", abs(dw.first_crossing(rows) - 2.5) < 1e-12)
check("first_crossing returns None when no crossing",
      dw.first_crossing([(1.0, 0.9, 0, 0), (2.0, 0.8, 0, 0)]) is None)

# ---- v2 instrument (deviation D1): per-t paired ratio, pred-weighted aggregation ---- #
_noise = dw.h.NOISE
try:
    dw.h.NOISE = 0.0
    fe0, fp0, _, _ = dw.cell_pert(0.0, 0.3, n=50)
    check("v2 lam=0 noiseless: per-t estimator == analytic pipeline at machine precision",
          float(np.max(np.abs(fe0 - fp0))) < 1e-10,
          f"max|diff| = {float(np.max(np.abs(fe0 - fp0))):.2e}")
finally:
    dw.h.NOISE = _noise

# the structural fix: near the lam=1.5 recurrence null, the v2 R_pred stays on the DW plateau
# (run 1's band-aggregate R_pred excursed below theta there -- the G4-abort diagnosis)
mini = {0.0: {}, 0.15: {}}
for lam in (1.40, 1.45, 1.55, 1.60):
    for eps in (0.0, 0.15):
        f_emp, f_pred, f_h1, f_h2 = dw.cell_pert(lam, eps, n=200)
        mini[eps][float(lam)] = dict(f_emp=f_emp, f_pred=f_pred, f_h1=f_h1, f_h2=f_h2)
_saved = dw.UNION_GRID
try:
    dw.UNION_GRID = np.array([1.40, 1.45, 1.55, 1.60])
    rows15 = dw.masked_grid(mini, 0.15)
finally:
    dw.UNION_GRID = _saved
plateau = [r_pred for (_, _, r_pred, _) in rows15]
check("v2 null-robustness: R_pred near the lam=1.5 null stays on the DW plateau (no sub-theta dip)",
      min(plateau) > dw.THETA, f"R_pred @ lam{[r[0] for r in rows15]} = {[f'{v:.3f}' for v in plateau]}")

# ---- BANKED PINS (post-verdict, run 2 of 2026-06-11; CONFIRMED) ----
# headline: pooled masked RMS = 0.0048 (K1 <= 0.10); lam*(eps) = 2.516/1.818/1.367/0.654 at
# eps = 0.15/0.2/0.3/0.5; log-log slope = -1.0880 (R^2 = 0.9797), C = lam*eps = 0.327 (K2 pass).
# Three deterministic single-cell pins re-verify the v2 observable byte-stably (full-n cells):
PINS = [(1.0, 0.3, 0.6396254755296876), (2.5, 0.15, 0.5051400142820378),
        (0.65, 0.5, 0.5038552656892787)]
for lam, eps, want in PINS:
    mini = {0.0: {}, eps: {}}
    for e in (0.0, eps):
        f_emp, f_pred, f_h1, f_h2 = dw.cell_pert(lam, e)
        mini[e][float(lam)] = dict(f_emp=f_emp, f_pred=f_pred, f_h1=f_h1, f_h2=f_h2)
    _saved2 = dw.UNION_GRID
    try:
        dw.UNION_GRID = np.array([lam])
        got = dw.masked_grid(mini, eps)[0][1]
    finally:
        dw.UNION_GRID = _saved2
    check(f"banked pin R_emp(lam={lam}, eps={eps}) = {want:.6f}", abs(got - want) < 1e-9,
          f"got {got!r}")

print(f"\n{'ALL PASS -- gates pinned; ready for the verdict run.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
