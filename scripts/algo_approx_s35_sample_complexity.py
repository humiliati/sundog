"""
S3-5 — the eps-essential region count as a SAMPLE-complexity predictor (algorithmic-approximation
slate 3).

N-4 showed the eps-essential region count predicts a ReLU net's generalization-WIDTH threshold
(capacity axis), modulo SGD trainability. S3-5 tests a SECOND axis: at FIXED, generous width
(capacity removed as the bottleneck), does the eps-essential count predict the SAMPLE complexity
— the training-set size needed to generalize within the bar? If region geometry is the learnable
invariant, it should govern DATA as well as capacity.

Design (decouples from N-4):
  * width fixed at 2k + 8 (>> any eps-essential here), so the net can always represent the target;
    the only thing varied is how much data it sees.
  * sweep n_train over a geometric grid; best-of-restarts oracle (so the threshold reflects
    learnability from that much data, not one SGD run's luck); a FIXED large held-out test set
    per target (stable frac-variance measurement).
  * sample_threshold(bar) = smallest n_train with held-out frac-variance-unexplained < bar.

Prediction: sample_threshold tracks eps_essential(bar) (more operative pieces -> more data),
across families and bars, and better than exact k or a constant. Reported via Spearman rank
correlation + a within-target monotonicity check (tighter bar -> more data).

Falsifier EPS_NOT_SAMPLE_PREDICTOR: sample_threshold tracks eps_essential no better than exact k
(or a constant) — region geometry governs capacity but not data.

Deterministic (seeded). CPU. Reuses the N-4 harness. Prints tables + a verdict line.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from algo_approx_n4_eps_regions import target_fn, eps_essential_pieces, MLP, LO, HI  # noqa: E402

torch.set_num_threads(4)


# ----- sample-complexity fitting oracle (variable n_train, fixed large held-out) --------------
def _fit_fracvar(f, width, n, seed, xte_t, yte_t, yte_var, sigma, epochs=1000, lr=0.01) -> float:
    """Train on n seed-drawn samples (labels + N(0,sigma^2)); return frac-variance-unexplained on
    the fixed CLEAN test (measures recovery of the true function)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(LO, HI, (n, 1)).astype(np.float32)
    ytr_np = f(xtr[:, 0]).astype(np.float32)
    if sigma > 0:
        ytr_np = ytr_np + rng.normal(0.0, sigma, size=n).astype(np.float32)
    ytr = torch.from_numpy(ytr_np).unsqueeze(1)
    xtr_t = torch.from_numpy(xtr)
    model = MLP(width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); lossf(model(xtr_t), ytr).backward(); opt.step()
    with torch.no_grad():
        return lossf(model(xte_t), yte_t).item() / yte_var


def oracle_fracvar_by_n(f, width, ns, restarts, xte_t, yte_t, yte_var, sigma):
    """best-of-restarts held-out frac-variance at each training size n in ns."""
    return {n: min(_fit_fracvar(f, width, n, 1000 * n + s, xte_t, yte_t, yte_var, sigma)
                   for s in range(restarts)) for n in ns}


def sample_threshold(fv_by_n, bar):
    for n in sorted(fv_by_n):
        if fv_by_n[n] < bar:
            return n
    return None


# ----- rank correlation (no scipy) -----------------------------------------------------------
def _rankavg(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    if len(a) < 2:
        return float("nan")
    ra, rb = _rankavg(a), _rankavg(b)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


# ----- experiment ---------------------------------------------------------------------------
FAMILIES = ["smooth", "essential", "jagged"]
KS = [3, 5, 8]
BARS = [0.10, 0.05, 0.02, 0.01]
NS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
RESTARTS = 4


def run_regime(noise_frac: float):
    tag = "noiseless" if noise_frac == 0.0 else f"noise={noise_frac}*std"
    print(f"###### S3-5 regime: {tag}   (width=2k+8, train-label noise; CLEAN held-out test) ######")
    cells = []  # (family, k, bar, sample_threshold, eps_essential)
    for kind in FAMILIES:
        print(f"=== family: {kind} ===")
        for k in KS:
            f = target_fn(kind, k)
            width = 2 * k + 8
            fstd = float(f(np.linspace(LO, HI, 201)).std())
            sigma = noise_frac * fstd
            rng_te = np.random.default_rng(424242 + k)
            xte = rng_te.uniform(LO, HI, (2000, 1)).astype(np.float32)
            xte_t = torch.from_numpy(xte)
            yte_t = torch.from_numpy(f(xte[:, 0]).astype(np.float32)).unsqueeze(1)
            yte_var = max(float(yte_t.var(unbiased=False).item()), 1e-12)
            fv = oracle_fracvar_by_n(f, width, NS, RESTARTS, xte_t, yte_t, yte_var, sigma)
            parts = []
            for bar in BARS:
                st = sample_threshold(fv, bar)
                ee = eps_essential_pieces(f, bar)
                cells.append((kind, k, bar, st, ee))
                parts.append(f"bar={bar:>4}: st={str(st):>5} ee={ee:>2}")
            print(f"  k={k}  width={width}  | " + "  | ".join(parts))
        print()

    ok = [(kind, k, bar, st, ee) for (kind, k, bar, st, ee) in cells if st is not None]
    n_ok = len(ok)
    rho_ee = rho_k = float("nan")
    if n_ok >= 2:
        st_arr = np.array([st for (_, _, _, st, _) in ok], dtype=float)
        ee_arr = np.array([ee for (_, _, _, _, ee) in ok], dtype=float)
        k_arr = np.array([k for (_, k, _, _, _) in ok], dtype=float)
        rho_ee = spearman(st_arr, ee_arr)
        rho_k = spearman(st_arr, k_arr)
    mono_ok, mono_tot = 0, 0
    floored = 0
    for kind in FAMILIES:
        for k in KS:
            sub = [st for (fk, kk, _, st, _) in cells if fk == kind and kk == k]
            present = [s for s in sub if s is not None]
            if len(present) == len(sub) and len(sub) >= 2:
                mono_tot += 1
                if all(sub[i] <= sub[i + 1] for i in range(len(sub) - 1)):
                    mono_ok += 1
            if present and all(s == NS[0] for s in present):
                floored += 1

    print(f"--- VERDICT (S3-5, {tag}) ---")
    print(f"  cells with a sample threshold: {n_ok}/{len(cells)}   "
          f"(targets floored at n_min={NS[0]}: {floored}/{len(FAMILIES) * len(KS)})")
    print(f"  Spearman(sample_threshold, eps_essential) = {rho_ee:.3f}")
    print(f"  Spearman(sample_threshold, exact k)        = {rho_k:.3f}")
    print(f"  within-target monotone (tighter bar -> more data): {mono_ok}/{mono_tot}")
    if floored >= 0.6 * (len(FAMILIES) * len(KS)):
        print("  -> FLOORED: sample complexity is below the grid's minimum for most targets; the")
        print("     data axis is trivial in this regime, so the hypothesis is untestable here.")
    elif (not np.isnan(rho_ee)) and rho_ee >= 0.5 and rho_ee > rho_k + 0.05 \
            and mono_tot > 0 and mono_ok >= 0.7 * mono_tot:
        print("  -> CONFIRMS: the eps-essential region count predicts SAMPLE complexity, not just")
        print("     capacity -- region geometry governs the data needed to generalize, tracking the")
        print("     operative scale (the bar) where exact k does not.")
    elif (not np.isnan(rho_ee)) and rho_ee > rho_k:
        print("  -> PARTIAL: eps-essential predicts sample complexity better than exact k, but not")
        print("     cleanly; a directional, not sharp, result.")
    else:
        print("  -> EPS_NOT_SAMPLE_PREDICTOR: eps-essential tracks sample complexity no better than")
        print("     exact k; region geometry governs capacity (N-4) but not data (this axis).")
    print()
    return rho_ee, rho_k, floored


def main():
    print("###### S3-5: eps-essential region count as a SAMPLE-complexity predictor ######")
    print("legend: st=sample_threshold(n_train clearing the bar); ee=eps-essential pieces\n")
    run_regime(0.0)
    run_regime(0.15)


if __name__ == "__main__":
    main()
