#!/usr/bin/env python
"""H3 v2 — the CORRECTED imported-wall test: can a TRAINED body defeat the Shadow-Invertibility
continuous-resist by learning to DEMODULATE before pooling, and is it OBJECTIVE-DEPENDENT?

v1 (scripts/shadow_pooled_synthetic.py) was CONFOUNDED: it made the continuous latent c a SHARED MEAN
across units, so raw mean-pooling trivially CONCENTRATED onto c (a sufficient statistic) — raw averaging
with NO encoder recovered c at R2=0.94. That tested nothing about the trained body. (See the adversarial
verification in results/atlas/h3/ and docs/atlas/H3_POOLED_SHADOW_PREREG.md honest-boundaries.)

v2 FIX — make raw averaging genuinely WASH c (a HARD anti-confound gate), so any post-pool c-recovery is
attributable to the TRAINED encoder, not sufficient-statistic concentration:
  * c lives in a per-unit RANDOM-FOURIER FRINGE: fringe_i = cos(w_m * c_i + psi_m), c_i = c + lam*xi_i,
    with HIGH frequencies w_m so each component Debye-Waller WASHES under the ensemble spread:
    mean_i cos(w_m c_i + psi_m) = cos(w_m c + psi_m) * exp(-w_m^2 lam^2 / 2) -> 0.
  * c is still IDENTIFIABLE from a SINGLE un-pooled unit (the RFF embedding is injective) -> C1.
  * KEY: a LINEAR readout of the RFF still washes under pooling (mean commutes with linear, and the raw
    RFF mean is washed). Only a NONLINEAR demodulator phi makes mean_i phi(u_i) ~ mean_i c_i = c. So the
    trained NONLINEAR encoder is genuinely load-bearing; demodulate-THEN-pool is what defeats the wash.
  * the DISCRETE latent d is carried THROUGH lossy averaging (per-unit noisy channel d*a + eta_i, NOT a
    noiseless broadcast constant) and survives by structural stability -> a real determine half.

THE QUESTION (objective-dependence): a body trained to REGRESS c (reg_c) is incentivized to learn the
nonlinear demodulator -> should RECOVER c post-pool (DEFEAT the resist). A body trained only to CLASSIFY d
(clf_d), a random untrained phi, and raw averaging have no such incentive -> c should WASH (resist HOLDS).

Pre-registered gates:
  C0 (anti-confound, HARD): raw mean_i u_i c-R2 -> low at high lam (raw averaging WASHES c). If this fails
     the construction is still confounded and the result is void.
  C1: single un-pooled unit recovers c (the latent IS present per-unit before pooling).
  DETERMINE: d-acc stays high post-pool AND from raw mean (structural stability through lossy averaging).
  DEFEAT (headline): reg_c post-pool c-R2 >> clf_d / random-phi / raw at high lam -> a trained, incentivized
     body DEFEATS the continuous-resist via learned demodulation. OBJECTIVE-DEPENDENT.
  If reg_c ALSO washes c -> the resist is robust even to an incentivized trained body (also a clean answer).

Reproducible: torch.set_num_threads(1) + fixed integer seeds (v1's hash()-salted seed + unpinned BLAS made
it non-deterministic). NOT public-eligible. Attribution: Shadow-Invertibility Law; Debye-Waller; DeepSets
(Zaheer 2017); random Fourier features (Rahimi & Recht 2007).
"""
import json
import sys
import warnings
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

torch.set_num_threads(1)                       # reproducibility (v1 defect: unpinned BLAS)
SEED = 1234
K = 64                                         # units per sample
M = 64                                         # RFF dims (continuous channel)
D = 8                                          # discrete-channel dims
H = 32                                         # rep dim
C_LO, C_HI = 1.0, 2.0
TRAIN_LAM = 1.0                                # envelope washed at train -> body MUST demodulate
LAMBDAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
N_TRAIN, N_PROBE = 8000, 2000
OBJ_SEED = {"clf_d": 11, "reg_c": 22, "recon": 33}   # FIXED per-objective offsets (no hash())

# ---- fixed code: RFF frequencies/phases + discrete direction (seeded, frozen) ---- #
_rng = np.random.default_rng(SEED)
W_RFF = _rng.uniform(2.5, 6.0, size=M)          # high freqs -> each component washes under spread
PSI = _rng.uniform(0, 2 * np.pi, size=M)
A_DISC = _rng.standard_normal(D); A_DISC /= np.linalg.norm(A_DISC)   # discrete direction (unit)
SIGMA_D = 1.5                                   # per-unit discrete noise (genuine lossiness on d)
OBS_NOISE = 0.05


def gen(n, lam, seed):
    """n samples, each K units. Continuous c in the washing RFF fringe (c_i=c+lam*xi_i); discrete d in a
    per-unit noisy channel d*a+eta. Returns units (n,K,F), c (n,), d (n,)."""
    rng = np.random.default_rng(seed)
    c = rng.uniform(C_LO, C_HI, n)
    d = rng.choice([-1.0, 1.0], n)
    xi = rng.standard_normal((n, K))
    c_i = c[:, None] + lam * xi                                  # (n,K) per-unit continuous
    fringe = np.cos(W_RFF[None, None, :] * c_i[:, :, None] + PSI[None, None, :])   # (n,K,M)
    eta = rng.standard_normal((n, K, D)) * SIGMA_D
    disc = d[:, None, None] * A_DISC[None, None, :] + eta        # (n,K,D) discrete + per-unit noise
    units = np.concatenate([fringe, disc], axis=2)              # (n,K,F)
    units = units + rng.standard_normal(units.shape) * OBS_NOISE
    return units.astype(np.float32), c.astype(np.float32), d.astype(np.float32)


F = M + D


class Phi(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, H))

    def forward(self, u):                       # u: (B,K,F) -> per-unit phi then mean-pool -> (B,H)
        return self.net(u).mean(dim=1)


def train_body(objective, units, c, d):
    torch.manual_seed(SEED + OBJ_SEED[objective])
    np.random.seed(SEED + OBJ_SEED[objective])
    phi = Phi()
    if objective == "clf_d":
        head = nn.Linear(H, 2); crit = nn.CrossEntropyLoss(); target = torch.tensor((d > 0).astype(np.int64))
    elif objective == "reg_c":
        head = nn.Linear(H, 1); crit = nn.MSELoss(); target = torch.tensor(c).float()[:, None]
    else:  # recon: predict the pooled raw input mean_i u_i
        head = nn.Linear(H, F); crit = nn.MSELoss(); target = torch.tensor(units).mean(dim=1)
    U = torch.tensor(units)
    opt = torch.optim.Adam(list(phi.parameters()) + list(head.parameters()), lr=1e-3)
    n = U.shape[0]; bs = 256
    for epoch in range(120):
        perm = torch.randperm(n)
        for b in range(0, n, bs):
            idx = perm[b:b + bs]
            opt.zero_grad()
            out = head(phi(U[idx]))
            loss = crit(out, target[idx])
            loss.backward(); opt.step()
    # training fit (did reg_c actually learn to demodulate?)
    with torch.no_grad():
        out = head(phi(U))
        if objective == "reg_c":
            pred = out[:, 0].numpy(); fit = 1 - np.var(c - pred) / np.var(c)   # train R2
        elif objective == "clf_d":
            fit = float(((out.argmax(1).numpy() > 0.5) == (d > 0)).mean())
        else:
            fit = float(1 - (crit(out, target)).item() / target.var().item())
    phi.eval()
    return phi, float(fit)


def phi_pool(phi, units):
    with torch.no_grad():
        return phi(torch.tensor(units)).numpy()


def c_r2(X, y):
    Xs = StandardScaler().fit_transform(X)
    kf = KFold(5, shuffle=True, random_state=0)
    return float(max(0.0, cross_val_score(Ridge(alpha=1.0), Xs, y, cv=kf, scoring="r2").mean()))


def d_acc(X, y):
    Xs = StandardScaler().fit_transform(X)
    yb = (y > 0).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    return float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=skf,
                                 scoring="balanced_accuracy").mean())


def main():
    print("=" * 88)
    print("H3 v2 — does a TRAINED body defeat the continuous-resist by demodulate-then-pool? (objective-dep)")
    print("=" * 88)
    # train all bodies ONCE at TRAIN_LAM (envelope washed -> must demodulate), plus a random untrained phi
    units_tr, c_tr, d_tr = gen(N_TRAIN, TRAIN_LAM, SEED + 1)
    bodies, fits = {}, {}
    for obj in ["clf_d", "reg_c", "recon"]:
        bodies[obj], fits[obj] = train_body(obj, units_tr, c_tr, d_tr)
        print(f"  trained {obj:6s}  train-fit={fits[obj]:.3f}")
    torch.manual_seed(SEED + 999); rand_phi = Phi(); rand_phi.eval()
    print(f"  reg_c train-fit is the demodulation check: high => it learned to demodulate c per-unit.\n")

    grid = {k: {"raw_c": [], "raw_d": [], "unit_c": [],
                "clf_d_c": [], "reg_c_c": [], "recon_c": [], "rand_c": [],
                "clf_d_d": [], "reg_c_d": [], "recon_d": []} for k in ["v"]}
    rows = []
    for lam in LAMBDAS:
        u, c, d = gen(N_PROBE, lam, SEED + 7 + int(lam * 1000))
        raw_mean = u.mean(axis=1)                                  # (n,F) raw pooled input (NO phi)
        unit0 = u[:, 0, :]                                         # single un-pooled unit
        feats = {
            "raw": raw_mean, "unit": unit0,
            "clf_d": phi_pool(bodies["clf_d"], u), "reg_c": phi_pool(bodies["reg_c"], u),
            "recon": phi_pool(bodies["recon"], u), "rand": phi_pool(rand_phi, u),
        }
        r = {"lam": lam,
             "raw_c": c_r2(feats["raw"], c), "raw_d": d_acc(feats["raw"], d),
             "unit_c": c_r2(feats["unit"], c),
             "clf_d_c": c_r2(feats["clf_d"], c), "reg_c_c": c_r2(feats["reg_c"], c),
             "recon_c": c_r2(feats["recon"], c), "rand_c": c_r2(feats["rand"], c),
             "clf_d_d": d_acc(feats["clf_d"], d), "reg_c_d": d_acc(feats["reg_c"], d),
             "recon_d": d_acc(feats["recon"], d)}
        rows.append(r)

    # ---- tables ---- #
    def line(key, label):
        return f"  {label:<22}" + " ".join(f"{r[key]:5.2f}" for r in rows)
    print("  c-RECOVERY (R2) vs lambda " + " ".join(f"{l:>5}" for l in LAMBDAS))
    print(line("raw_c",   "raw mean (NO phi)"))
    print(line("unit_c",  "single unit (C1)"))
    print(line("rand_c",  "random-phi pool"))
    print(line("clf_d_c", "clf_d pool"))
    print(line("recon_c", "recon pool"))
    print(line("reg_c_c", "reg_c pool  <== "))
    print("\n  d-RECOVERY (balanced-acc) vs lambda")
    print(line("raw_d",   "raw mean (NO phi)"))
    print(line("clf_d_d", "clf_d pool"))
    print(line("reg_c_d", "reg_c pool"))
    print(line("recon_d", "recon pool"))

    # ---- gates / verdict ---- #
    hi = [r for r in rows if r["lam"] >= 1.0]                      # the washed regime
    raw_hi = max(r["raw_c"] for r in hi)
    rand_hi = max(r["rand_c"] for r in hi)
    clf_hi = max(r["clf_d_c"] for r in hi)
    regc_hi = min(r["reg_c_c"] for r in hi)
    unit0lam = rows[0]["unit_c"]
    det_min = min(min(r["clf_d_d"], r["reg_c_d"], r["recon_d"], r["raw_d"]) for r in rows)

    C0 = raw_hi <= 0.15                       # anti-confound: raw averaging washes c at high lam
    C1 = unit0lam >= 0.5                       # c present in a single un-pooled unit (lam=0)
    DET = det_min >= 0.85                      # d determined (>=0.85 balanced-acc) through lossy avg
    DEFEAT = (regc_hi >= 0.5) and (regc_hi - clf_hi >= 0.25) and (regc_hi - raw_hi >= 0.25)
    RAND_WASH = rand_hi <= 0.2                 # untrained phi does NOT demodulate (control)

    print("\n" + "=" * 88)
    print("GATES")
    print("=" * 88)
    print(f"  [{'PASS' if C0 else 'FAIL'}] C0 anti-confound: raw mean WASHES c at high lam (max raw_c[lam>=1]={raw_hi:.3f} <= 0.15)")
    print(f"  [{'PASS' if C1 else 'FAIL'}] C1: c present in a single un-pooled unit (unit_c[lam=0]={unit0lam:.3f} >= 0.5)")
    print(f"  [{'PASS' if RAND_WASH else 'FAIL'}] control: random untrained phi pool WASHES c (max rand_c[lam>=1]={rand_hi:.3f} <= 0.2)")
    print(f"  [{'PASS' if DET else 'FAIL'}] DETERMINE: d-acc >= 0.85 through lossy averaging (min over all={det_min:.3f})")
    print(f"  [{'PASS' if DEFEAT else 'FAIL'}] DEFEAT (headline): reg_c RECOVERS c post-pool at high lam "
          f"(min reg_c_c[lam>=1]={regc_hi:.3f}) and beats clf_d ({clf_hi:.3f}) & raw ({raw_hi:.3f}) by >=0.25")
    print(f"        reg_c train-fit={fits['reg_c']:.3f} (the demodulation check)")

    print("\n" + "=" * 88)
    if C0 and C1 and DEFEAT and DET:
        verdict = ("BOUNDED-POSITIVE: a TRAINED, incentivized body (reg_c) DEFEATS the Shadow-Invertibility "
                   "continuous-resist by learning a nonlinear demodulate-then-pool code, while raw averaging, "
                   "a random encoder, and the un-incentivized clf_d all let the continuous WASH. The resist is "
                   "OBJECTIVE-DEPENDENT and FRAGILE to a trained per-unit encoder. Determine holds throughout.")
    elif C0 and C1 and DET and not DEFEAT:
        verdict = ("BOUNDED-NULL: even an incentivized trained body (reg_c) does NOT recover the continuous "
                   "post-pool -> the Shadow-Invertibility resist is ROBUST to a trained per-unit encoder "
                   "(the demodulate-then-pool defeat does not materialize). Determine holds; a clean answer.")
    elif not C0:
        verdict = ("VOID: anti-confound C0 FAILED — raw averaging still recovers c, so the construction is "
                   "confounded (like v1). Re-tune the RFF frequencies / spread before interpreting.")
    else:
        verdict = "MIXED — see gates above; report honestly."
    print("VERDICT:", verdict)
    print("=" * 88)

    out = {"params": {"K": K, "M": M, "D": D, "H": H, "train_lam": TRAIN_LAM, "lambdas": LAMBDAS,
                      "c_range": [C_LO, C_HI], "w_rff_range": [2.5, 6.0], "sigma_d": SIGMA_D, "seed": SEED},
           "train_fit": fits, "rows": rows,
           "gates": {"C0_raw_washes": C0, "C1_unit_has_c": C1, "rand_phi_washes": RAND_WASH,
                     "determine": DET, "DEFEAT_objective_dependent": DEFEAT},
           "key": {"raw_c_hi": raw_hi, "rand_c_hi": rand_hi, "clf_d_c_hi": clf_hi, "reg_c_c_hi": regc_hi,
                   "unit_c_lam0": unit0lam, "determine_min": det_min, "reg_c_train_fit": fits["reg_c"]},
           "verdict": verdict}
    p = Path("results/atlas/h3/synthetic_v2_result.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o)
                            if isinstance(o, (np.floating,)) else o))
    print(f"\nwrote {p}")
    return 0 if (C0 and C1 and DET) else 1


if __name__ == "__main__":
    sys.exit(main())
