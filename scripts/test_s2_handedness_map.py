#!/usr/bin/env python
"""Frozen test for the S2 per-feature +/-V(phi) handedness map (scripts/s2_handedness_map.py) and the
TIR-phase mechanism it rests on (s2_optics.tir_retardance / mueller_tir).

Locks the pre-registered result (docs/atlas/S2_HANDEDNESS_MAP_PREREG.md):
  Stage 1 — the Fresnel total-internal-reflection s-p retardance reproduces the analytic anchors
            (ice delta_max=30.56 deg @ 59.1 deg; glass 45-deg Fresnel-rhomb pair; 0 below crit / at
            grazing; pure retarder = no diattenuation).
  Stage 2 — Claim A (per-feature +/-V is REAL: %-level circular pol from TIR phase + birefringence) and
            Claim B (the achiral-ice display nets to ~0: V(phi) is azimuthally ANTISYMMETRIC and
            integral V -> 0 — the V-analog of Koennen & Tinbergen 1991's measured U=0), with the
            anti-tautology check that a single CHIRAL sub-path does NOT self-cancel (net ~100%), so the
            cancellation is the mirror-partner physics, not per-ray averaging.
Deterministic (no RNG). Run: python scripts/test_s2_handedness_map.py
"""
import sys
import os
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import s2_optics as so          # noqa: E402
import s2_handedness_map as hm  # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("S2 deepening — TIR-phase retarder + per-feature +/-V(phi) map:\n")

# ===== Stage 1 — the TIR-phase mechanism vs analytic Fresnel-rhomb anchors =================== #
print("Stage 1 — TIR-phase retarder (the missing primary linear->circular mechanism):")

# (1) ice peak retardance: delta_max = 30.56 deg at theta = 59.1 deg
n = 1.31
th_max = np.degrees(np.arcsin(np.sqrt(2 * (1 / n) ** 2 / (1 + (1 / n) ** 2))))
d_max = np.degrees(so.tir_retardance(th_max, n, 1.0))
check("ice TIR retardance peak delta_max ~ 30.56 deg", abs(d_max - 30.56) < 0.2,
      f"delta_max={d_max:.2f} deg @ theta={th_max:.2f} deg")
check("ice peak is at theta ~ 59.1 deg", abs(th_max - 59.1) < 0.3, f"theta_max={th_max:.2f} deg")

# (2) below the critical angle and at grazing the pure-TIR retardance is 0
thc = np.degrees(np.arcsin(1.0 / n))
check("delta = 0 below the critical angle (no TIR)", so.tir_retardance(thc - 5.0, n, 1.0) == 0.0,
      f"theta_c={thc:.2f} deg")
check("delta -> 0 at grazing (theta -> 90)", np.degrees(so.tir_retardance(89.99, n, 1.0)) < 0.1)

# (3) the glass Fresnel rhomb: delta crosses 45 deg at TWO angles bracketing the peak (the textbook
#     linear->circular demonstration)
ng = 1.51
ths = np.linspace(np.degrees(np.arcsin(1 / ng)) + 0.01, 89.99, 4000)
dg = np.array([np.degrees(so.tir_retardance(t, ng, 1.0)) for t in ths])
cross45 = ths[:-1][(dg[:-1] - 45.0) * (dg[1:] - 45.0) < 0]
check("glass (n=1.51) delta peak ~ 45.9 deg", abs(dg.max() - 45.9) < 0.3, f"delta_max={dg.max():.2f} deg")
check("glass delta crosses 45 deg at TWO rhomb angles bracketing the peak",
      len(cross45) == 2 and cross45[0] < ths[np.argmax(dg)] < cross45[1],
      f"crossings @ {np.round(cross45, 1)} deg")

# (4) the TIR Mueller matrix is a PURE retarder (energy-conserving, no diattenuation)
M = so.mueller_tir(70.0, n, 1.0, 0.3)
check("mueller_tir is a pure retarder (M00=1, no intensity<->pol coupling in row/col 0)",
      abs(M[0, 0] - 1.0) < 1e-12 and np.allclose(M[0, 1:], 0) and np.allclose(M[1:, 0], 0))

# ===== Stage 2 — the per-feature +/-V(phi) map ============================================== #
print("\nStage 2 — per-feature +/-V(phi) handedness map (Claims A and B):")
r0 = hm.sweep(dn=0.0)        # pure-TIR mechanism
rb = hm.sweep(dn=so.DN_ICE)  # full chain (TIR phase + ice birefringence)

# (5) Claim A — per-feature +/-V is REAL: %-level circular polarization is produced
check("Claim A: TIR phase ALONE makes %-level circular pol (per-ray |V/I| > 1%)",
      r0["peak_ray_VoverI"] > 0.01, f"per-ray peak |V/I| = {r0['peak_ray_VoverI']*100:.2f}%")
check("Claim A: full chain (TIR+birefringence) is larger still",
      rb["peak_ray_VoverI"] > r0["peak_ray_VoverI"],
      f"TIR-only {r0['peak_ray_VoverI']*100:.2f}% -> full {rb['peak_ray_VoverI']*100:.2f}%")
check("Claim A: the feature carries a nonzero observable (flux-avg |V/I| > 0.5%)",
      rb["peak_VoverI_total"] > 0.005, f"flux-avg feature |V/I| = {rb['peak_VoverI_total']*100:.2f}%")

# (6) Claim B — the achiral display nets to ~0: V(phi) antisymmetric, integral V -> 0
for tag, r in (("TIR-only", r0), ("full chain", rb)):
    check(f"Claim B [{tag}]: net |∮V|/∮|V| -> 0 (population handedness cancels)",
          r["net_ratio_total"] < 0.01, f"net = {r['net_ratio_total']*100:.2f}%")
    check(f"Claim B [{tag}]: V(phi) azimuthally ANTISYMMETRIC (residual -> 0)",
          r["antisym_residual_total"] < 0.01, f"antisym residual = {r['antisym_residual_total']*100:.2f}%")

# (7) anti-tautology — a single CHIRAL sub-path does NOT self-cancel (net ~ 100%); so Claim B's
#     cancellation is the mirror-partner physics of achiral ice, not per-ray averaging
check("anti-tautology: a one-handed sub-path keeps net handedness (chiral net > 50%)",
      r0["net_ratio_chiral"] > 0.5, f"chiral sub-path net = {r0['net_ratio_chiral']*100:.1f}%")

# (8) the self-mirror exit face (face 3) straddles the principal plane in +/- pairs: each ray carries
#     V, but its NET contribution is ~0 (emergent symmetry signature, not imposed)
rays3 = [hm.trace_ray(p, 20.0, exit_face=3, dn=0.0)
         for p in np.linspace(0, 2 * np.pi, 720, endpoint=False)]
V3 = np.array([r["V"] for r in rays3 if r["valid"]])
I3 = np.array([r["I"] for r in rays3 if r["valid"]])
net3 = abs(V3.sum()) / I3.sum() if I3.sum() > 0 else 0.0
permv = np.mean(np.abs(V3 / np.where(I3 > 0, I3, 1)))      # per-ray |V/I| is genuinely nonzero
check("self-mirror exit face -> NET V ~ 0 (per-ray V nonzero but +/- pairs cancel)",
      len(V3) > 0 and net3 < 1e-3 and permv > 1e-3,
      f"net|V|/I = {net3:.1e}, per-ray mean|V/I| = {permv*100:.2f}%")

print(f"\n{'ALL PASS -- the TIR-phase mechanism reproduces the Fresnel-rhomb anchors, per-feature +/-V is REAL (~%-level from TIR phase + birefringence), and the achiral-ice display is azimuthally ANTISYMMETRIC with net V -> 0 (the V-analog of Koennen U=0) -- while a chiral sub-path retains full net handedness, so the cancellation is mirror-partner physics, not averaging.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
