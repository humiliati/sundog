#!/usr/bin/env python
"""H8 / substrate S3 — reaction-diffusion as a determine/resist shadow.

Extends the Shadow-Invertibility Law (S0 1-D toy, S1 2-D field, S2 halo optics) to a 2D Gray-Scott
reaction-diffusion BODY. Tests, against docs/atlas/H8_REACTION_DIFFUSION_PREREG.md, whether a lossy
ENSEMBLE-jitter shadow (Debye-Waller averaging over K kinetics-jittered subunits, NOT single-frame blur):
  * S3c  RESISTS a continuous latent (the pattern wavelength, set by a diffusion-scale knob), and
  * S3d  DETERMINES a discrete latent (the morphology class: spots vs stripes),
with the split sharpening as the ensemble lossiness lambda grows. Plus a B-panel "is-it-RD" 4-way
mechanism confusion matrix {RD, Cahn-Hilliard, matched-spectrum GRF, FitzHugh-Nagumo wave} and the
matched-GRF non-vacuity null.

Cost trick: real Gray-Scott fields are precomputed ONCE into a batched kinetics->field LIBRARY; shadows
are then formed by averaging the precomputed translation-invariant features (radial S(k) + intensity
histogram + connected-component count) over K subunits. Faithful to the decoherence mechanism, affordable.

Modes (the apparatus discipline, cf. pvnp_phase5_lossiness_crossover.py):
  --calibrate : throwaway CALIB_SEED, 64^2 smoke -- the pre-freeze power check (tune power knobs ONLY).
  --frozen    : frozen DATA_SEED, 128^2 primary -- the real run (only after calibration sign-off + freeze).

NOT public-eligible. Attribution: Turing 1952; Gray & Scott 1983/84; Pearson 1993 (Science 261:189);
Shadow-Invertibility Law (ATLAS_PHASE5_CROSS_SUBSTRATE.md); Debye 1913/Waller 1923; Liu 2013 (PNAS
110:11905, mussel beds = Cahn-Hilliard != Turing). Probe fns copied from pvnp_phase5_lossiness_crossover.
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
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# ---- FROZEN constants (gates + lambda-grid + seeds are NOT calibratable) ----------------------- #
LAMBDAS = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]
CV = 4
PROBE_SEED = 0
DATA_SEED, CALIB_SEED = 20260609, 999
CONT0_MIN, DISC0_MIN = 0.70, 0.95
CONT_MAX_MAX, DISC_MIN_MIN = 0.10, 0.95

# Gray-Scott diffusion + reaction anchors (Pearson basins; Du,Dv fixed, diffusion-SCALE s is the knob)
DU, DV = 0.16, 0.08
BASINS = {            # (F, k) per morphology class
    "spots":     (0.030, 0.062),
    "stripes":   (0.030, 0.055),
    "labyrinth": (0.046, 0.063),
    "gaps":      (0.026, 0.055),
}

# ---- POWER KNOBS (calibratable on CALIB_SEED; frozen before --frozen) -------------------------- #
CFG = dict(
    grid=64, steps=3500, K=6, n=64, noise=0.04,
    r_bins=24, h_bins=12, thr=0.20,
    s_lo=0.70, s_hi=1.30, s_floor=0.45, s_ceil=1.50, sig_s=0.35,   # diffusion-scale (wavelength) knob
    m_kin=40, r_ic=8,                                              # library resolution per class
    panel_n=48, panel_k=6,                                        # B-panel (secondary) sample/subunit count
)

# globals filled by build_libraries()
RAD_BINS = None          # (grid,grid) radial bin index
RAD_COUNT = None         # (r_bins,) bin populations
NFEAT = None
LIB = {}                 # class -> dict(feats=(M*R,NFEAT), s=(M*R,), by_bin=list of idx arrays)
S_GRID = None


# ============================ Gray-Scott (batched) ============================================== #
def _laplacian(a):
    return (np.roll(a, 1, 1) + np.roll(a, -1, 1) + np.roll(a, 1, 2) + np.roll(a, -1, 2) - 4.0 * a)


def gray_scott_batch(F, k, s, grid, steps, rng):
    """Integrate B Gray-Scott fields in parallel. F,k,s are (B,) per-field arrays; s scales diffusion
    (wavelength ~ sqrt(s)). Returns the v-field (B,grid,grid)."""
    B = len(F)
    Fb = np.asarray(F, np.float32)[:, None, None]; kb = np.asarray(k, np.float32)[:, None, None]
    Du = (DU * np.asarray(s, np.float32))[:, None, None]; Dv = (DV * np.asarray(s, np.float32))[:, None, None]
    u = np.ones((B, grid, grid), np.float32); v = np.zeros((B, grid, grid), np.float32)
    # seeded IC: random square patches of (u=0.5, v=0.25) + small noise
    for b in range(B):
        nb = int(rng.integers(8, 16))
        for _ in range(nb):
            cy, cx = int(rng.integers(0, grid)), int(rng.integers(0, grid))
            r = int(rng.integers(2, 5))
            ys = slice(max(0, cy - r), min(grid, cy + r)); xs = slice(max(0, cx - r), min(grid, cx + r))
            u[b, ys, xs] = 0.50; v[b, ys, xs] = 0.25
    u += 0.02 * rng.standard_normal((B, grid, grid)).astype(np.float32)
    v += 0.02 * rng.standard_normal((B, grid, grid)).astype(np.float32)
    for _ in range(steps):
        uvv = u * v * v
        u += Du * _laplacian(u) - uvv + Fb * (1.0 - u)
        v += Dv * _laplacian(v) + uvv - (Fb + kb) * v
        np.clip(u, 0.0, 1.0, out=u); np.clip(v, 0.0, 1.0, out=v)
    return v


# ============================ translation-invariant features ==================================== #
def _radial_setup(grid, r_bins):
    ky = np.fft.fftfreq(grid)[:, None]; kx = np.fft.fftfreq(grid)[None, :]
    kr = np.sqrt(kx ** 2 + ky ** 2)
    idx = np.clip((kr / (kr.max() + 1e-12) * r_bins).astype(int), 0, r_bins - 1)
    count = np.bincount(idx.ravel(), minlength=r_bins).astype(float)
    return idx, count


def feature(v):
    """(B,grid,grid) -> (B,NFEAT): [radial S(k) shape (r_bins) | intensity hist (h_bins) | comp-count]."""
    B = v.shape[0]; rb, hb, thr = CFG["r_bins"], CFG["h_bins"], CFG["thr"]
    vm = v - v.mean((1, 2), keepdims=True)
    P = np.abs(np.fft.fft2(vm)) ** 2
    radial = np.empty((B, rb))
    flat_idx = RAD_BINS.ravel()
    for b in range(B):
        s = np.bincount(flat_idx, P[b].ravel(), minlength=rb) / (RAD_COUNT + 1e-12)
        s[0] = 0.0                                        # drop DC
        radial[b] = s / (s.sum() + 1e-12)                 # spectral SHAPE (peak location = wavelength)
    # per-FIELD-normalized histogram (fixed [0,1] range after dividing by the field's own max) so the
    # range is per-field-independent and CONTRAST-invariant — removes the per-batch v.max() s-correlated
    # leak the adversarial review flagged (different chunk composition was changing the same field's hist).
    hist = np.stack([np.histogram(v[b] / (v[b].max() + 1e-9), bins=hb, range=(0.0, 1.0), density=True)[0]
                     for b in range(B)])
    hist = hist / (hist.sum(1, keepdims=True) + 1e-12)
    cc = np.array([ndimage.label(v[b] > thr)[1] for b in range(B)], float)[:, None]
    return np.concatenate([radial, hist, cc], axis=1)


# ============================ libraries ========================================================= #
def _build_one(F, k, grid, steps, seed):
    """Library for one (F,k) class across the diffusion-scale grid x r_ic ICs (chunked, batched)."""
    rng = default_rng(seed)
    global S_GRID
    S_GRID = np.linspace(CFG["s_floor"], CFG["s_ceil"], CFG["m_kin"])
    s_all = np.repeat(S_GRID, CFG["r_ic"])
    Fv = np.full(len(s_all), F); kv = np.full(len(s_all), k)
    feats = np.empty((len(s_all), NFEAT))
    chunk = 200
    for a in range(0, len(s_all), chunk):
        b = min(a + chunk, len(s_all))
        v = gray_scott_batch(Fv[a:b], kv[a:b], s_all[a:b], grid, steps, rng)
        feats[a:b] = feature(v)
    by_bin = [np.where(np.isclose(s_all, sv))[0] for sv in S_GRID]   # indices per s-bin (for IC variety)
    return dict(feats=feats, s=s_all, by_bin=by_bin)


def build_libraries(grid, steps, seed, classes):
    global RAD_BINS, RAD_COUNT, NFEAT, LIB
    RAD_BINS, RAD_COUNT = _radial_setup(grid, CFG["r_bins"])
    NFEAT = CFG["r_bins"] + CFG["h_bins"] + 1
    LIB = {}
    for j, cls in enumerate(classes):
        F, k = BASINS[cls]
        LIB[cls] = _build_one(F, k, grid, steps, seed + 101 * j)
        print(f"    [lib] {cls:<9} (F={F},k={k}) {len(LIB[cls]['s'])} fields  "
              f"<cc>={LIB[cls]['feats'][:, -1].mean():.1f}", flush=True)


def _pick(lib, s_vals, rng):
    """Nearest s-bin per requested s, random IC variant within the bin -> library feature rows."""
    bins = np.clip(np.searchsorted(S_GRID, s_vals), 0, len(S_GRID) - 1)
    # snap to nearer of the two neighbours
    lo = np.clip(bins - 1, 0, len(S_GRID) - 1)
    take_lo = np.abs(s_vals - S_GRID[lo]) < np.abs(s_vals - S_GRID[bins])
    bins = np.where(take_lo, lo, bins)
    rows = [lib["by_bin"][bb][rng.integers(0, len(lib["by_bin"][bb]))] for bb in bins]
    return lib["feats"][rows]


# ============================ generators (shadow = mean over K subunits) ========================= #
def gen_s3c(n, lam, rng, noise):
    """Continuous-resists leg: xc = diffusion-scale (sets wavelength), class FIXED = stripes; xd dummy."""
    K = CFG["K"]
    xc = rng.uniform(CFG["s_lo"], CFG["s_hi"], n)
    xd = rng.choice([-1.0, 1.0], n)                         # dummy discrete
    feats = np.empty((n, NFEAT))
    for i in range(n):
        s_i = np.clip(xc[i] + lam * CFG["sig_s"] * rng.standard_normal(K), CFG["s_floor"], CFG["s_ceil"])
        feats[i] = _pick(LIB["stripes"], s_i, rng).mean(0)
    return feats + rng.normal(0, noise, feats.shape), xc, xd


def gen_s3d(n, lam, rng, noise):
    """Discrete-determines leg: xd = class {spots +1, stripes -1}; xc = dummy diffusion-scale drawn
    from the COMMON range (class _|_ wavelength). Same K-subunit ensemble shadow as S3c."""
    K = CFG["K"]
    xd = np.where(np.arange(n) < n // 2, 1.0, -1.0); rng.shuffle(xd)   # balanced classes (clean C5)
    xc = rng.uniform(CFG["s_lo"], CFG["s_hi"], n)           # dummy continuous (decorrelated from class)
    feats = np.empty((n, NFEAT))
    for i in range(n):
        s_i = np.clip(xc[i] + lam * CFG["sig_s"] * rng.standard_normal(K), CFG["s_floor"], CFG["s_ceil"])
        lib = LIB["spots"] if xd[i] > 0 else LIB["stripes"]
        feats[i] = _pick(lib, s_i, rng).mean(0)
    return feats + rng.normal(0, noise, feats.shape), xc, xd


# ---- B-panel mechanisms (CH, GRF, FHN) for the is-it-RD confusion matrix ----------------------- #
def cahn_hilliard_batch(grid, steps, rng, B):
    """Spectral semi-implicit Cahn-Hilliard (phase separation, coarsening). c in [-1,1]."""
    c = 0.1 * rng.standard_normal((B, grid, grid))
    kx = np.fft.fftfreq(grid)[None, None, :] * 2 * np.pi
    ky = np.fft.fftfreq(grid)[None, :, None] * 2 * np.pi
    k2 = kx ** 2 + ky ** 2; k4 = k2 ** 2
    M, kappa, dt = 1.0, 1.0, 0.05
    denom = 1.0 + dt * M * kappa * k4
    for _ in range(steps):
        mu = c ** 3 - c
        c = np.real(np.fft.ifft2((np.fft.fft2(c) - dt * M * k2 * np.fft.fft2(mu)) / denom))
    return 0.5 * (c + 1.0)                                  # map to [0,1] like a concentration


def grf_matched(target_radial, grid, rng, B):
    """Phase-randomized Gaussian field with a prescribed radial power spectrum (the matched-power null)."""
    kr = np.sqrt(np.fft.fftfreq(grid)[None, None, :] ** 2 + np.fft.fftfreq(grid)[None, :, None] ** 2)
    rb = CFG["r_bins"]; idx = np.clip((kr / (kr.max() + 1e-12) * rb).astype(int), 0, rb - 1)
    amp = np.sqrt(np.maximum(target_radial, 0))[:, idx[0]]   # (B,grid,grid) sqrt-power per k-bin
    phase = np.exp(2j * np.pi * rng.random((B, grid, grid)))
    f = np.real(np.fft.ifft2(amp * phase))
    f = (f - f.min((1, 2), keepdims=True))
    return f / (f.max((1, 2), keepdims=True) + 1e-12)


def fitzhugh_nagumo_batch(grid, steps, rng, B):
    """FitzHugh-Nagumo excitable medium seeded with a broken wavefront (spiral waves)."""
    u = np.zeros((B, grid, grid)); w = np.zeros((B, grid, grid))
    u[:, :, :grid // 2] = 1.0; w[:, :grid // 2, :] = 0.5    # cross IC -> spiral
    a, b, eps, Du2, dt = 0.7, 0.8, 0.08, 1.0, 0.1
    for _ in range(steps):
        u += dt * (Du2 * _laplacian(u) + u - u ** 3 / 3.0 - w)
        w += dt * eps * (u + a - b * w)
        np.clip(u, -2, 2, out=u)
    umin = u.min((1, 2), keepdims=True)
    return (u - umin) / (u.max((1, 2), keepdims=True) - umin + 1e-12)


def panel_features(n, lam, rng, noise, grid, steps):
    """Shadow features for the 4-way is-it-RD panel. Each mechanism -> n samples, each = mean of K
    subunit fields' features (same ensemble shadow). RD draws across all four basins."""
    K = CFG["panel_k"]
    X, y = [], []
    # RD (label 0): mix basins
    rd_classes = list(BASINS)
    for i in range(n):
        cls = rd_classes[i % len(rd_classes)]
        s_i = np.clip(rng.uniform(CFG["s_lo"], CFG["s_hi"]) + lam * CFG["sig_s"] * rng.standard_normal(K),
                      CFG["s_floor"], CFG["s_ceil"])
        if cls in LIB:
            X.append(_pick(LIB[cls], s_i, rng).mean(0)); y.append(0)
    # CH (1), GRF (2), FHN (3): build K subunit fields each, average features
    for lab, fn in [(1, "ch"), (2, "grf"), (3, "fhn")]:
        for i in range(n):
            if fn == "ch":
                fields = cahn_hilliard_batch(grid, steps // 3 + int(lam * 200), rng, K)
            elif fn == "fhn":
                fields = fitzhugh_nagumo_batch(grid, steps // 4, rng, K)
            else:
                tgt = LIB["stripes"]["feats"][rng.integers(0, len(LIB["stripes"]["s"]), K), :CFG["r_bins"]]
                fields = grf_matched(tgt, grid, rng, K)
            X.append(feature(fields).mean(0)); y.append(lab)
    X = np.array(X) + rng.normal(0, noise, (len(X), NFEAT))
    return X, np.array(y)


# ============================ probe (copied from pvnp_phase5_lossiness_crossover) ================ #
def _std(X):
    return StandardScaler().fit_transform(X)


def cont_recovery(X, y):
    Xs = _std(X); kf = KFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LinearRegression(), Xs, y, cv=kf, scoring="r2").mean())
    mlp = float(cross_val_score(MLPRegressor(hidden_layer_sizes=(64,), max_iter=500, random_state=0),
                                Xs, y, cv=kf, scoring="r2").mean())
    return {"lin": max(0.0, lin), "mlp": max(0.0, mlp), "best": max(0.0, lin, mlp)}


def disc_recovery(X, y):
    Xs = _std(X); yb = (y > 0).astype(int); maj = max(yb.mean(), 1 - yb.mean())
    skf = StratifiedKFold(CV, shuffle=True, random_state=PROBE_SEED)
    lin = float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=skf, scoring="accuracy").mean())
    mlp = float(cross_val_score(MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=0),
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
        print(f"  [{tag} lam={lam:<4}] cont={c['best']:.3f} (lin {c['lin']:.2f}/mlp {c['mlp']:.2f})  "
              f"disc={d['best']:.3f} (lin {d['lin']:.2f}/mlp {d['mlp']:.2f})  maj={d['maj']:.2f}", flush=True)
    return {"cont": cont, "disc": disc, "maj": imb,
            "lambda_star_c": half_life(cont, LAMBDAS), "lambda_star_d": half_life(disc, LAMBDAS)}


def panel_run(seed, n, noise, grid, steps, lam):
    rng = default_rng(seed + 555)
    X, y = panel_features(n, lam, rng, noise, grid, steps)
    Xs = _std(X)
    skf = StratifiedKFold(CV, shuffle=True, random_state=PROBE_SEED)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=700, random_state=0)
    pred = cross_val_predict(clf, Xs, y, cv=skf)
    bal = float(balanced_accuracy_score(y, pred))
    cm = confusion_matrix(y, pred).tolist()
    # RD-vs-matched-GRF margin (C2 null): binary balanced acc on labels {0 RD, 2 GRF}
    m = np.isin(y, [0, 2]); yb = (y[m] == 0).astype(int)
    rdgrf = float(cross_val_score(LogisticRegression(max_iter=2000), Xs[m], yb,
                                  cv=StratifiedKFold(CV, shuffle=True, random_state=PROBE_SEED),
                                  scoring="balanced_accuracy").mean())
    return {"panel_balanced_acc": round(bal, 4), "confusion": cm, "labels": ["RD", "CH", "GRF", "FHN"],
            "rd_vs_grf_balacc": round(rdgrf, 4), "lam": lam}


# ============================ controls (C1 non-triviality, C4 permutation) ====================== #
def k_dependence(seed, noise, ks=(1, 2, 8, 32, 128)):
    """C1 / THE DECISIVE MECHANISM TEST. The charFun law distinguishes a genuine RESIST from a finite-K
    artifact by the K-dependence of the wavelength-recovery half-life:
      * GENUINE charFun/Debye-Waller resist (S0/S1/S2): the population's charFun decays, so the wash is
        K-INVARIANT — half-life saturates regardless of how many subunits you average.
      * FINITE-MEAN (DETERMINE-type) latent washed only by finite-K LLN slack: half-life moves OUTWARD as
        K grows — a bigger ensemble RECOVERS the latent (it concentrates by the law of large numbers).
    H8's diffusion-scale s has a finite centered mean, so the charFun law PREDICTS it determines; this
    test reads the half-life vs K to confirm which regime S3c is in. Returns {K: (half_life, cont@lam1)}."""
    base_K = CFG["K"]; out = {}
    for kk in ks:
        CFG["K"] = kk
        cont = []
        for lam in LAMBDAS:
            X, yc, _ = gen_s3c(CFG["n"], lam, default_rng(seed + int(round(lam * 1000)) + kk + 31), noise)
            cont.append(round(cont_recovery(X, yc)["best"], 4))
        lc = half_life(cont, LAMBDAS)
        i1 = LAMBDAS.index(1.0) if 1.0 in LAMBDAS else len(LAMBDAS) - 1
        out[str(kk)] = {"half_life": lc, "cont_lam1": cont[i1]}
    CFG["K"] = base_K
    # verdict: is the half-life K-INVARIANT (charFun resist) or GROWING with K (LLN / determine-type)?
    hl = [out[str(k)]["half_life"] for k in ks]
    hl_num = [h if h is not None else (max(LAMBDAS) + 1) for h in hl]
    grows = all(hl_num[i] <= hl_num[i + 1] for i in range(len(hl_num) - 1)) and hl_num[-1] > hl_num[0]
    out["is_charfun_resist"] = not grows          # genuine resist iff NOT growing with K
    out["half_life_grows_with_K"] = bool(grows)
    return out


def label_permutation(seed, noise, lam=0.1):
    """C4 — shuffle the labels: both recoveries must collapse to chance (no leakage / no overfit)."""
    Xc, yc, _ = gen_s3c(CFG["n"], lam, default_rng(seed + 13), noise)
    Xd, _, yd = gen_s3d(CFG["n"], lam, default_rng(seed + 17), noise)
    p = default_rng(seed + 23)
    return {"cont_perm": round(cont_recovery(Xc, p.permutation(yc))["best"], 4),
            "disc_perm": round(disc_recovery(Xd, p.permutation(yd))["best"], 4)}


# ============================ gates + verdict =================================================== #
def gates(s3c, s3d):
    g = {
        "G1_preflight_c": s3c["cont"][0] >= CONT0_MIN,
        "G2_preflight_d": s3d["disc"][0] >= DISC0_MIN,
        "G3_continuous_resists": (s3c["cont"][-1] <= CONT_MAX_MAX) and (s3c["lambda_star_c"] is not None),
        "G4_discrete_determines": (min(s3d["disc"]) >= DISC_MIN_MIN) and (s3d["lambda_star_d"] is None),
        "C5_class_balanced": all(0.45 <= m <= 0.55 for m in s3d["maj"]),
    }
    # C5 folded in (the review's consistency point): an imbalanced determine cannot report CROSSOVER.
    g["G5_CROSSOVER"] = g["G3_continuous_resists"] and g["G4_discrete_determines"] and g["C5_class_balanced"]
    g["preflight_ok"] = g["G1_preflight_c"] and g["G2_preflight_d"]
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--frozen", action="store_true")
    ap.add_argument("--no-panel", action="store_true", help="skip the B-panel (faster A-crossover check)")
    ap.add_argument("--out", default="results/atlas/h8")
    args = ap.parse_args()

    if args.frozen:
        CFG.update(grid=128, steps=4000, K=8, n=160, noise=0.04, m_kin=40, r_ic=8,
                   panel_n=60, panel_k=6)
        seed, mode = DATA_SEED, "frozen"
    else:
        seed, mode = CALIB_SEED, "calibrate"     # CFG defaults already the 64^2 smoke

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[cfg-S3] mode={mode} seed={seed} grid={CFG['grid']} steps={CFG['steps']} K={CFG['K']} "
          f"n={CFG['n']} noise={CFG['noise']} lambdas={LAMBDAS}", flush=True)

    classes = ["spots", "stripes"] if args.no_panel else ["spots", "stripes", "labyrinth", "gaps"]
    print(f"Building Gray-Scott libraries {classes}...", flush=True)
    build_libraries(CFG["grid"], CFG["steps"], seed, classes)
    print(f"  libraries built ({round(time.time()-t0,1)}s)", flush=True)

    print("S3c (continuous-resists: wavelength):", flush=True)
    s3c = sweep(gen_s3c, CFG["n"], seed, CFG["noise"], "S3c")
    print("S3d (discrete-determines: spots vs stripes):", flush=True)
    s3d = sweep(gen_s3d, CFG["n"], seed, CFG["noise"], "S3d")
    g = gates(s3c, s3d)

    print("DECISIVE mechanism test — K-dependence of the wavelength 'resist' (charFun-invariant vs LLN-growing):",
          flush=True)
    kdep = k_dependence(seed, CFG["noise"])
    for k in ["1", "2", "8", "32", "128"]:
        if k in kdep:
            print(f"    K={k:>3}: half-life lam*_c={kdep[k]['half_life']}  cont@lam1={kdep[k]['cont_lam1']}", flush=True)
    print(f"    half-life grows with K = {kdep['half_life_grows_with_K']}  => "
          f"{'GENUINE charFun resist' if kdep['is_charfun_resist'] else 'finite-mean LLN slack (DETERMINE-type) — NOT a Shadow-law resist'}",
          flush=True)

    panel = ctrl = None
    if not args.no_panel:
        print("Controls (C4 label-permutation):", flush=True)
        ctrl = {"permutation": label_permutation(seed, CFG["noise"])}
        print(f"  perm {ctrl['permutation']}", flush=True)
        print("B-panel (is-it-RD: RD/CH/GRF/FHN) @ lam=0 and lam=0.5:", flush=True)
        panel = {"lam0": panel_run(seed, CFG["panel_n"], CFG["noise"], CFG["grid"], CFG["steps"], 0.0),
                 "lam_mid": panel_run(seed, CFG["panel_n"], CFG["noise"], CFG["grid"], CFG["steps"], 0.5)}
        print(f"  panel balacc lam0={panel['lam0']['panel_balanced_acc']} "
              f"lam_mid={panel['lam_mid']['panel_balanced_acc']}  "
              f"RD-vs-GRF(lam0)={panel['lam0']['rd_vs_grf_balacc']}", flush=True)

    print(f"\n== {mode.upper()} S3 ==")
    print(f"  S3c: cont0={s3c['cont'][0]} contMax={s3c['cont'][-1]} lam*_c={s3c['lambda_star_c']} "
          f"-> resists={g['G3_continuous_resists']}")
    print(f"  S3d: disc0={s3d['disc'][0]} minDisc={min(s3d['disc'])} lam*_d={s3d['lambda_star_d']} "
          f"-> determines={g['G4_discrete_determines']}")
    # HONEST verdict: a genuine Shadow-law crossover requires (i) preflight power (both latents recoverable
    # at lambda=0 -- else the charFun check reads noise), (ii) the gate-level crossover, AND (iii) the resist
    # is a charFun resist (K-invariant half-life), not finite-K LLN slack. The diffusion-scale wavelength
    # fails (iii) -> NULL.
    genuine = g["preflight_ok"] and g["G5_CROSSOVER"] and kdep["is_charfun_resist"]
    print(f"  gate-CROSSOVER={g['G5_CROSSOVER']}  charFun-resist={kdep['is_charfun_resist']}  "
          f"=> GENUINE crossover={genuine}")
    print(f"  VERDICT: {'GENUINE determine/resist crossover on RD' if genuine else 'NULL — the wavelength is a finite-mean DETERMINE-type latent (charFun law predicts it concentrates by LLN); the gate-level wash is a finite-K artifact, NOT a Shadow-law resist. A genuine RD resist needs a phase (charFun-decaying) latent.'}")

    (out / f"{mode}.json").write_text(json.dumps(
        {"mode": mode, "seed": seed, "cfg": CFG, "lambdas": LAMBDAS,
         "S3c": s3c, "S3d": s3d, "gates": g, "k_dependence": kdep, "genuine_crossover": bool(genuine),
         "controls": ctrl, "panel": panel, "wall_s": round(time.time() - t0, 1)}, indent=2,
        default=lambda o: bool(o) if isinstance(o, np.bool_) else o))
    print(f"  wrote {out/(mode+'.json')}  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
