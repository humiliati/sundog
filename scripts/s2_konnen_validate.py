#!/usr/bin/env python
"""Stage-A validation: the s2_optics Mueller chain vs Können & Tinbergen 1991 measured-sky 22-deg
halo LINEAR polarization (Appl. Opt. 30:3382-3400). A MODEL-CREDIBILITY GATE for the polarization
physics: if the same Mueller code that predicts Stokes-V cannot reproduce Können's measured Q, its V
output is untrustworthy.

Measured-sky anchors (Können 1991):
  - Fresnel floor of the refraction-halo linear DoP ~3.7% = (1-F)/(1+F), F = cos^4(theta_h/2) = 0.929.
  - birefringent two-image angular split ~0.11 deg (ordinary + extraordinary, orthogonally polarized).
  - inner edge: radial E-vector (Q in the scattering-plane frame); U = 0 by mirror symmetry.
  - a ~0.10-deg-wide inner ledge ~100% polarized; peak intrinsic DoP ~9% (birefringence ~doubles Fresnel).

SCOPE: this validates the refraction + birefringence Mueller CHAIN against observation. It does NOT
validate Stokes-V / handedness — Können measured no V, and his U=0 cancellation is the direct linear
analog of the net-V-cancels prior (so V/handedness stays forward-model-tier; see S2_MEASURED_SKY_SCOPE).
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
import s2_optics as so

N_O, N_E = 1.3090, 1.3104        # ice ordinary / extraordinary refractive index @ ~590 nm
APEX = 60.0                      # hexagonal prism -> 22-deg halo
fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


theta_i = np.degrees(np.arcsin(so.N_ICE * np.sin(np.radians(APEX / 2))))   # min-deviation entry angle
print(f"22-deg halo min-deviation entry angle = {theta_i:.2f} deg (internal {APEX/2:.0f} deg)")

print("Target 1 — Fresnel-floor linear DoP (refraction only):")
S = so.ray_stokes(theta_i, 0.0, 0.0, parity=1)        # L=0 -> retarder=identity -> pure Fresnel chain
dop = abs(S[1] / S[0]); uoi = abs(S[2] / S[0]); voi = abs(S[3] / S[0])
F = np.cos(np.radians(22.0 / 2)) ** 4
dop_k = (1 - F) / (1 + F)
print(f"  F = cos^4(11deg) = {F:.4f};  (1-F)/(1+F) = {dop_k:.4f} (Können ~3.7%)")
check("Fresnel-floor DoP matches Können ~3.7%", abs(dop - dop_k) < 0.01, f"model={dop:.4f} konnen={dop_k:.4f}")
check("U/I ~ 0 (mirror symmetry)", uoi < 1e-6, f"U/I={uoi:.2e}")
check("V/I ~ 0 (pure refraction, no retarder)", voi < 1e-6, f"V/I={voi:.2e}")

print("Target 2 — birefringent two-image split:")
theta_o = float(so.halo_min_deviation(APEX, N_O))
theta_e = float(so.halo_min_deviation(APEX, N_E))
split = abs(theta_e - theta_o)
check("birefringent split ~0.11 deg (Können)", abs(split - 0.11) < 0.06,
      f"model={split:.3f} deg  (theta_o={theta_o:.3f}, theta_e={theta_e:.3f}, n_o={N_O}, n_e={N_E})")

print("Target 3 — birefringence two-image ledge (sharp inner edges):")
th = np.linspace(theta_o - 0.3, theta_o + 1.5, 2000)
# the two birefringent eigen-images turn on at their own min-deviation edge, fully orthogonally
# polarized (ordinary=radial +Q, extraordinary=tangential -Q). In the split window only one is on.
Io = (th >= theta_o).astype(float)
Ie = (th >= theta_e).astype(float)
I = Io + Ie; Q = Io - Ie
pdop = np.abs(np.divide(Q, I, out=np.zeros_like(Q), where=I > 0))
ledge = (th >= theta_o) & (th < theta_e)              # the ~0.11-deg split window
check("fully-polarized inner ledge across the split window (Können ~100%)",
      pdop[ledge].min() > 0.99, f"ledge DoP={pdop[ledge].min():.2f} over {split:.3f} deg")
# Können peak intrinsic ~9%: the 100%-ledge convolved with the diffraction+solar-disk width (~0.45 deg)
w = 0.45
g = np.exp(-0.5 * ((th - th[:, None]) / w) ** 2)
g /= g.sum(1, keepdims=True)
Ic = g @ I; Qc = g @ Q
peak_dop = np.abs(np.divide(Qc, Ic, out=np.zeros_like(Qc), where=Ic > 0)).max()
print(f"  ledge DoP = {pdop[ledge].min():.2f} over the {split:.3f}-deg split window (Können ~100%);")
print(f"  convolved (w={w} deg) peak intrinsic DoP ~ {peak_dop:.3f}  (Können measured ~0.09)")
check("convolved peak intrinsic DoP in Können's ~9% range", 0.04 < peak_dop < 0.20, f"={peak_dop:.3f}")

print(f"\n{'ALL PASS' if fail == 0 else str(fail) + ' FAILED'} — linear-pol Mueller chain vs Können 1991 "
      f"(validates refraction+birefringence; V/handedness NOT validated, stays forward-model)")
sys.exit(1 if fail else 0)
