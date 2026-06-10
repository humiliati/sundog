#!/usr/bin/env python
"""H8 v2 / substrate S3phi — reaction-diffusion PHASE-resist + DEFECT-CHARGE-determine shadow.

Successor to the H8 v1 NULL (`H8_REACTION_DIFFUSION_RESULT.md`): v1's continuous latent (wavelength via a
diffusion-scale knob) had a FINITE MEAN, which the charFun law correctly predicts DETERMINES (concentrates
by LLN) -- confirmed by a half-life that GREW with K. v2 fixes the one structural error: the continuous
latent now enters through a PHASE (the charFun-decaying channel), and the discrete latent is a TOPOLOGICAL
CHARGE (chirality) that is first-order-statistically invisible (mirror images), fixing v1's cc-threshold
tautology. Mirrors S1 (phase resists, winding determines) and the S2 +/-V handedness leg, on a real
reaction-diffusion spiral medium (complex Ginzburg-Landau).

Substrate: a CGL spiral A=u+iw (the canonical oscillatory-RD amplitude equation) seeded from a charge-q
vortex; chirality q in {+1,-1} (q=-1 = the exact mirror of a q=+1 field -> identical first-order stats);
the rotational PHASE phi is imposed by spatially rotating the field (rotation = m=1 azimuthal phase shift).
Shadow = mean over K subunits whose phase is jittered by lambda. feature() = a PHASE block (m=1,2 azimuthal
coeffs -> washes by charFun) + a CHIRALITY block (boundary winding charge -> phase-invariant, survives) + a
mirror-invariant BLIND block (radial power + intensity histogram + component count -> for the anti-tautology
control). THE BINDING NEW GATE (G-KINV): the phase half-life must be K-INVARIANT (charFun), the exact test
v1 failed.

Modes: --calibrate (throwaway seed, small grid; tune power knobs) ; --frozen (data seed, primary).
NOT public-eligible. Attribution: Aranson & Kramer (RMP 74:99, 2002, CGL spirals); the Shadow-Invertibility
/ charFun laws; Debye/Waller; S1/S2 (the phase-resist + handedness-determine precedents).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from scipy import ndimage
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---- FROZEN constants (gates + lambda-grid + seeds NOT calibratable) --------------------------- #
LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
CV = 4
PROBE_SEED = 0
DATA_SEED, CALIB_SEED = 20260609, 999
CONT0_MIN, DISC0_MIN = 0.70, 0.95
CONT_MAX_MAX, DISC_MIN_MIN = 0.10, 0.95
NONTRIVIAL_MAX = 0.60          # handedness-BLIND probe must NOT recover chirality above this

CFG = dict(
    grid=64, steps=1500, K=8, n=120, noise=0.15,
    b1=0.5, c1=-0.8, dt=0.05, r0=0.15,                # CGL (stable-spiral regime) + vortex-core scale
    phase_lo=0.0, phase_hi=1.5, jit_rad=2.5,          # rotational-phase range (rad) + jitter scale (rad/lambda)
    m_angles=120, r_bases=6,                          # library: pre-rotated angles x base spirals
    radii=(0.30, 0.45, 0.60), m_modes=(1, 2),         # azimuthal sampling (ref-normalization + winding)
    pr=6, pt=24,                                       # phase block = polar resampling (N_r x N_theta)
    r_bins=16, h_bins=10, thr=0.0, n_theta=180,
)

# globals filled by build_library()
GRID = None
POLAR = None          # (radii x n_theta) sampling indices + e^{i m theta} kernels
RAD_BINS = RAD_COUNT = None
BLOCKS = {}           # feature block index ranges
NFEAT = None
LIB = {}              # chirality -> array [r_bases, m_angles, NFEAT]
ANGLES = None         # the m_angles grid (radians)


# ============================ complex Ginzburg-Landau spiral ==================================== #
def _lap(a):
    return (np.roll(a, 1, 1) + np.roll(a, -1, 1) + np.roll(a, 1, 2) + np.roll(a, -1, 2) - 4.0 * a)


def cgl_spiral_batch(B, grid, steps, chirality, rng):
    """dA/dt = A + (1+i b1) lap(A) - (1+i c1)|A|^2 A, seeded from a charge-`chirality` vortex."""
    b1, c1, dt, r0 = CFG["b1"], CFG["c1"], CFG["dt"], CFG["r0"]
    g = np.linspace(-1, 1, grid)
    X, Y = np.meshgrid(g, g)
    r = np.sqrt(X ** 2 + Y ** 2); th = np.arctan2(Y, X)
    A = np.empty((B, grid, grid), np.complex64)
    for bi in range(B):
        A[bi] = (np.tanh(r / r0) * np.exp(1j * (chirality * th + 0.4 * rng.standard_normal()))).astype(np.complex64)
    for _ in range(steps):
        A += dt * (A + (1 + 1j * b1) * _lap(A) - (1 + 1j * c1) * (np.abs(A) ** 2) * A)
    return np.real(A).astype(np.float32), np.imag(A).astype(np.float32)


# ============================ feature blocks ==================================================== #
def _polar_setup(grid):
    c = grid // 2
    th = np.linspace(0, 2 * np.pi, CFG["n_theta"], endpoint=False)
    idx, kern = [], []
    for rfrac in CFG["radii"]:                                  # rings for ref-normalization + winding
        rr = rfrac * c
        xs = np.clip((c + rr * np.cos(th)).astype(int), 0, grid - 1)
        ys = np.clip((c + rr * np.sin(th)).astype(int), 0, grid - 1)
        idx.append((ys, xs))
        kern.append({m: np.exp(1j * m * th) for m in CFG["m_modes"]})
    # full polar grid (N_r x N_theta) for the rotation-sensitive PHASE block
    nr, nt = CFG["pr"], CFG["pt"]
    rfr = np.linspace(0.18, 0.72, nr); ang = np.linspace(0, 2 * np.pi, nt, endpoint=False)
    pys = np.clip((c + (rfr[:, None] * c) * np.sin(ang[None, :])).astype(int), 0, grid - 1)
    pxs = np.clip((c + (rfr[:, None] * c) * np.cos(ang[None, :])).astype(int), 0, grid - 1)
    return th, idx, kern, (pys, pxs)


def phase_block(u):
    """Polar resampling (N_r x N_theta), per-ring DC-removed: the angular pattern at each radius. Rotation
    of the spiral = circular shift in theta (rotation-SENSITIVE), and many features (noise-ROBUST). Averaging
    over phase-jittered rotations smears the angular structure -> the phase decoheres (charFun)."""
    pys, pxs = POLAR[3]
    pol = u[pys, pxs]                                            # (N_r, N_theta)
    pol = pol - pol.mean(1, keepdims=True)
    return pol.ravel()


def _radial_setup(grid):
    ky = np.fft.fftfreq(grid)[:, None]; kx = np.fft.fftfreq(grid)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    rb = CFG["r_bins"]
    b = np.clip((kr / (kr.max() + 1e-12) * rb).astype(int), 0, rb - 1)
    return b, np.bincount(b.ravel(), minlength=rb).astype(float)


def winding(u, w):
    """Boundary winding number of arg(u+iw) at EACH ring (CHIRALITY; rotation/phase-invariant; survives).
    Returned at all 3 radii (redundant clean +/-1 features) so the determine probe reliably finds the
    handedness even when the washed phase block adds noise dimensions."""
    out = []
    for ys, xs in POLAR[1]:
        phi = np.arctan2(w[ys, xs], u[ys, xs])
        out.append(np.sum(np.diff(np.unwrap(np.concatenate([phi, phi[:1]])))) / (2 * np.pi))
    return np.array(out)


def blind_block(u):
    """Radial power spectrum + intensity histogram + component count (MIRROR-INVARIANT; the blind probe)."""
    P = np.abs(np.fft.fft2(u - u.mean())) ** 2
    rb = CFG["r_bins"]
    s = np.bincount(RAD_BINS.ravel(), P.ravel(), minlength=rb) / (RAD_COUNT + 1e-12)
    s[0] = 0.0; s = s / (s.sum() + 1e-12)
    un = (u - u.min()) / (u.max() - u.min() + 1e-12)
    hist = np.histogram(un, bins=CFG["h_bins"], range=(0, 1), density=True)[0]
    hist = hist / (hist.sum() + 1e-12)
    cc = float(ndimage.label(un > 0.6)[1])
    return np.concatenate([s, hist, [cc]])


def feature(u, w):
    return np.concatenate([phase_block(u), winding(u, w), blind_block(u)])


def _set_blocks():
    global BLOCKS, NFEAT
    npb = CFG["pr"] * CFG["pt"]; nch = len(CFG["radii"])
    nbl = CFG["r_bins"] + CFG["h_bins"] + 1
    BLOCKS = {"phase": (0, npb), "chir": (npb, npb + nch), "blind": (npb + nch, npb + nch + nbl)}
    NFEAT = npb + nch + nbl


# ============================ library (pre-rotated features) ==================================== #
def _ref_normalize(u, w):
    """Rotate a base field so its m=1 phase (at the middle radius) is ~0 -> phi is an absolute angle."""
    idx, kern = POLAR[1], POLAR[2]
    ys, xs = idx[1]
    z = np.sum(u[ys, xs] * kern[1][1])
    a0 = np.degrees(np.angle(z))            # rotate by +phase to zero the m=1 reference (rot +a decreases phase)
    mag = abs(z) / len(ys) + 1e-6           # m=1 amplitude -> normalize so every base has unit m=1 magnitude
    return (ndimage.rotate(u / mag, a0, reshape=False, mode="grid-wrap", order=1),
            ndimage.rotate(w / mag, a0, reshape=False, mode="grid-wrap", order=1))


def build_library(grid, seed):
    global GRID, POLAR, RAD_BINS, RAD_COUNT, ANGLES, LIB
    GRID = grid
    POLAR = _polar_setup(grid)
    RAD_BINS, RAD_COUNT = _radial_setup(grid)
    _set_blocks()
    rng = default_rng(seed)
    up, wp = cgl_spiral_batch(CFG["r_bases"], grid, CFG["steps"], +1, rng)
    ANGLES = np.linspace(0, 360, CFG["m_angles"], endpoint=False)
    LIB = {1: np.empty((CFG["r_bases"], CFG["m_angles"], NFEAT)),
           -1: np.empty((CFG["r_bases"], CFG["m_angles"], NFEAT))}
    chir = {1: 0.0, -1: 0.0}
    for bi in range(CFG["r_bases"]):
        u0, w0 = _ref_normalize(up[bi], wp[bi])               # +1 base, phase-normalized
        um, wm = np.fliplr(u0).copy(), np.fliplr(w0).copy()   # mirror (reflect x) flips the winding to -q
        for q, (ub, wb) in [(1, (u0, w0)), (-1, (um, wm))]:
            for ai, ang in enumerate(ANGLES):
                ur = ndimage.rotate(ub, ang, reshape=False, mode="grid-wrap", order=1)
                wr = ndimage.rotate(wb, ang, reshape=False, mode="grid-wrap", order=1)
                LIB[q][bi, ai] = feature(ur, wr)
            chir[q] += LIB[q][bi, 0, BLOCKS["chir"][0]]
    print(f"    [lib] {CFG['r_bases']} CGL bases x {CFG['m_angles']} angles x2 chir; "
          f"mean winding +q={chir[1]/CFG['r_bases']:+.2f} -q={chir[-1]/CFG['r_bases']:+.2f}", flush=True)


def _pick(q, base_idx, deg):
    ai = int(round((deg % 360) / 360.0 * CFG["m_angles"])) % CFG["m_angles"]
    return LIB[q][base_idx, ai]


# ============================ generators (shadow = mean over K phase-jittered subunits) ========== #
def gen_phase_c(n, lam, rng, noise):
    """Continuous-RESISTS leg: xc = rotational phase phi (chirality fixed +1, dummy xd)."""
    K = CFG["K"]; scale = np.degrees(CFG["jit_rad"])               # jitter in degrees (xi in rad -> deg)
    xc = rng.uniform(CFG["phase_lo"], CFG["phase_hi"], n)
    xd = rng.choice([-1.0, 1.0], n)                      # dummy
    feats = np.empty((n, NFEAT))
    for i in range(n):
        base = int(rng.integers(0, CFG["r_bases"]))
        degs = np.degrees(xc[i]) + lam * scale * rng.standard_normal(K)
        feats[i] = np.mean([_pick(1, base, d) for d in degs], axis=0)
    return feats + rng.normal(0, noise, feats.shape), xc, xd


def gen_phase_d(n, lam, rng, noise):
    """Discrete-DETERMINES leg: xd = chirality {+1,-1}; xc = dummy phase (common range)."""
    K = CFG["K"]; scale = np.degrees(CFG["jit_rad"])
    xd = np.where(np.arange(n) < n // 2, 1.0, -1.0); rng.shuffle(xd)
    xc = rng.uniform(CFG["phase_lo"], CFG["phase_hi"], n)   # dummy
    feats = np.empty((n, NFEAT))
    for i in range(n):
        base = int(rng.integers(0, CFG["r_bases"]))
        q = int(xd[i])
        degs = np.degrees(xc[i]) + lam * scale * rng.standard_normal(K)
        feats[i] = np.mean([_pick(q, base, d) for d in degs], axis=0)
    return feats + rng.normal(0, noise, feats.shape), xc, xd


# ============================ probe (copied from the frozen apparatus) =========================== #
def _std(X):
    return StandardScaler().fit_transform(X)


def cont_recovery(X, y):
    Xs = _std(X); kf = KFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LinearRegression(), Xs, y, cv=kf, scoring="r2").mean())
    mlp = float(cross_val_score(MLPRegressor(hidden_layer_sizes=(64,), max_iter=600, random_state=0),
                                Xs, y, cv=kf, scoring="r2").mean())
    return {"lin": max(0.0, lin), "mlp": max(0.0, mlp), "best": max(0.0, lin, mlp)}


def disc_recovery(X, y):
    Xs = _std(X); yb = (y > 0).astype(int); maj = max(yb.mean(), 1 - yb.mean())
    skf = StratifiedKFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=skf, scoring="accuracy").mean())
    mlp = float(cross_val_score(MLPClassifier(hidden_layer_sizes=(64,), max_iter=600, random_state=0),
                                Xs, yb, cv=skf, scoring="accuracy").mean())
    det = lambda a: (a - maj) / max(1 - maj, 1e-9)
    return {"lin": det(lin), "mlp": det(mlp), "best": det(max(lin, mlp)), "maj": maj}


def half_life(curve, lams):
    base = curve[0]
    if base <= 0:
        return None
    for lam, v in zip(lams, curve):
        if v <= 0.5 * base:
            return lam
    return None


def sweep(gen, n, seed, noise, tag):
    cont, disc, imb = [], [], []
    for lam in LAMBDAS:
        rng = default_rng(seed + int(round(lam * 1000)) + 7)
        X, yc, yd = gen(n, lam, rng, noise)
        c = cont_recovery(X, yc); d = disc_recovery(X, yd)
        cont.append(round(c["best"], 4)); disc.append(round(d["best"], 4)); imb.append(round(d["maj"], 3))
        print(f"  [{tag} lam={lam:<4}] cont={c['best']:.3f}  disc={d['best']:.3f}  maj={d['maj']:.2f}", flush=True)
    return {"cont": cont, "disc": disc, "maj": imb,
            "lambda_star_c": half_life(cont, LAMBDAS), "lambda_star_d": half_life(disc, LAMBDAS)}


def k_dependence(seed, noise, ks=(8, 64, 512), lam_test=None):
    """THE BINDING GATE (G-KINV), the clean (redundancy-proof) form. At a lambda PAST the charFun crossover
    (default lambda_max), a genuine charFun resist has DESTROYED the phase information -- so the recovery
    stays ~0 NO MATTER how large the ensemble K. A finite-mean LLN latent (v1) RECOVERS the latent as K
    grows (cont RISES with K). So: is_charfun_resist iff cont(lam_test) stays <= 0.15 across K (does NOT
    rise with K). v1 FAILED exactly this -- its cont ROSE with K at fixed lambda (0.15 -> 0.68)."""
    lam_test = max(LAMBDAS) if lam_test is None else lam_test
    base_K = CFG["K"]; cont_vs_k = {}
    for kk in ks:
        CFG["K"] = kk
        cont_vs_k[str(kk)] = round(cont_recovery(
            *gen_phase_c(CFG["n"], lam_test, default_rng(seed + kk + 31), noise)[:2])["best"], 4)
    CFG["K"] = base_K
    vals = [cont_vs_k[str(k)] for k in ks]
    charfun = (max(vals) <= 0.15) and (vals[-1] - vals[0] <= 0.10)   # stays low AND does not rise with K
    return {"lam_test": lam_test, "cont_vs_K": cont_vs_k, "is_charfun_resist": bool(charfun),
            "rises_with_K": bool(vals[-1] - vals[0] > 0.10)}


def nontrivial_control(seed, noise, lam=0.0):
    """C-NONTRIVIAL: a handedness-BLIND probe (mirror-invariant blind block ONLY) must FAIL to recover
    chirality (<= NONTRIVIAL_MAX). Only the winding/phase handedness features separate +/-q."""
    X, _, yd = gen_phase_d(CFG["n"], lam, default_rng(seed + 91), noise)
    b0, b1 = BLOCKS["blind"]; c0, c1 = BLOCKS["chir"]
    blind = disc_recovery(X[:, b0:b1], yd)["best"]
    chir = disc_recovery(X[:, c0:c1], yd)["best"]
    return {"blind_block_disc": round(blind, 4), "chir_block_disc": round(chir, 4),
            "nontrivial_ok": blind <= NONTRIVIAL_MAX}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--out", default="results/atlas/h8v2")
    args = ap.parse_args()
    if args.frozen:
        CFG.update(grid=88, steps=2000, K=8, n=200, noise=0.15, m_angles=180, r_bases=10)
        seed, mode = DATA_SEED, "frozen"
    else:
        seed, mode = CALIB_SEED, "calibrate"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[cfg-S3phi] mode={mode} seed={seed} grid={CFG['grid']} steps={CFG['steps']} K={CFG['K']} "
          f"n={CFG['n']} noise={CFG['noise']} lambdas={LAMBDAS}", flush=True)
    print("Building CGL spiral library...", flush=True)
    build_library(CFG["grid"], seed)
    print(f"  library built ({round(time.time()-t0,1)}s)", flush=True)

    print("S3phi-c (continuous-RESISTS: rotational phase):", flush=True)
    s3c = sweep(gen_phase_c, CFG["n"], seed, CFG["noise"], "phase")
    print("S3phi-d (discrete-DETERMINES: spiral chirality):", flush=True)
    s3d = sweep(gen_phase_d, CFG["n"], seed, CFG["noise"], "chir")

    print(f"G-KINV: cont(lam={max(LAMBDAS)}) vs K -- charFun DESTROYS the phase (stays ~0 for all K) vs "
          f"LLN RECOVERS it (rises with K):", flush=True)
    kdep = k_dependence(seed, CFG["noise"])
    print(f"    cont vs K = {kdep['cont_vs_K']}  -> charFun-resist={kdep['is_charfun_resist']} "
          f"(rises_with_K={kdep['rises_with_K']})", flush=True)
    ctrl = nontrivial_control(seed, CFG["noise"])
    print(f"C-NONTRIVIAL: handedness-blind disc={ctrl['blind_block_disc']} (<= {NONTRIVIAL_MAX}?), "
          f"chir-block disc={ctrl['chir_block_disc']}  -> ok={ctrl['nontrivial_ok']}", flush=True)

    g = {
        "G1_preflight_c": s3c["cont"][0] >= CONT0_MIN,
        "G2_preflight_d": s3d["disc"][0] >= DISC0_MIN,
        "G3_continuous_resists": (s3c["cont"][-1] <= CONT_MAX_MAX) and (s3c["lambda_star_c"] is not None),
        "G4_discrete_determines": (min(s3d["disc"]) >= DISC_MIN_MIN) and (s3d["lambda_star_d"] is None),
        "G_KINV_charfun_resist": kdep["is_charfun_resist"],
        "C_NONTRIVIAL": ctrl["nontrivial_ok"],
        "C5_class_balanced": all(0.45 <= m <= 0.55 for m in s3d["maj"]),
    }
    g["G5_CROSSOVER"] = (g["G1_preflight_c"] and g["G2_preflight_d"] and g["G3_continuous_resists"]
                        and g["G4_discrete_determines"] and g["G_KINV_charfun_resist"]
                        and g["C_NONTRIVIAL"] and g["C5_class_balanced"])
    print(f"\n== {mode.upper()} S3phi ==")
    print(f"  S3phi-c: cont0={s3c['cont'][0]} contMax={s3c['cont'][-1]} lam*_c={s3c['lambda_star_c']}")
    print(f"  S3phi-d: disc0={s3d['disc'][0]} minDisc={min(s3d['disc'])} lam*_d={s3d['lambda_star_d']}")
    print(f"  G-KINV(charFun resist)={g['G_KINV_charfun_resist']}  C-NONTRIVIAL={g['C_NONTRIVIAL']}")
    print(f"  GENUINE CROSSOVER = {g['G5_CROSSOVER']}")
    if mode == "calibrate":
        print(f"  CALIBRATION {'PASSES -> freeze + run --frozen' if g['G5_CROSSOVER'] else 'NEEDS TUNING / may NULL'}")

    (out / f"{mode}.json").write_text(json.dumps(
        {"mode": mode, "seed": seed, "cfg": CFG, "lambdas": LAMBDAS, "blocks": BLOCKS,
         "S3phi_c": s3c, "S3phi_d": s3d, "k_dependence": kdep, "nontrivial": ctrl, "gates": g,
         "wall_s": round(time.time() - t0, 1)}, indent=2,
        default=lambda o: bool(o) if isinstance(o, np.bool_) else o))
    print(f"  wrote {out/(mode+'.json')}  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
