"""PDE C2 — Sabra shell-state dimensionality probe (v2).

Fixes v1 artifacts: (a) restrict the body to dynamically-real inertial shells
(<|u_n|^2> above a noise floor; excludes the pinned forcing shell's dominance
and the numerical-underflow tail), (b) z-score only those (no divide-by-~0),
(c) add a PERM CONTROL as the arbiter — real vs permuted-shadow R^2
distinguishes genuine under-determination from a block-split artifact. Longer
(near-blow-up) window; saves raw samples. Honest caveat: the stable window is
~1 burst time, so the slow/intermittent modes are under-sampled — a definitive
measure needs the adaptive integrator. Spec:
docs/proof/PDE_C2_SHELL_DIMENSIONALITY_PROBE.md.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from pde_c2_sabra_cell import ShellModel, build_config  # noqa: E402

BURN = 300_000
NSAMP = 20_000
STRIDE = 100          # ~1e-2 time units; total ~2.3M steps (< ~3.5M blow-up)
ENERGY_FLOOR = 1e-11  # keep inertial shells above numerical noise
SPLIT = 0.7
OUT = ROOT / "results" / "proof" / "c2-shell-dim-v2"


def finite(u: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(u))) and float(np.max(np.abs(u))) < 1e6


def main() -> None:
    cfg = build_config(
        SimpleNamespace(preset="smoke", model="sabra", forcing="fixed-amplitude",
                        diagnostic=False, self_test=False, allow_unregistered_overrides=False, out=OUT)
    )
    model = ShellModel(cfg)
    n = cfg.n_shells
    k = model.k
    t0 = time.perf_counter()

    u = model.initial_state()
    blew = False
    for i in range(BURN):
        u = model.step(u)
        if not finite(u):
            blew = True
            print(f"[c2-dim-v2] BLOWUP in burn-in at {i}", flush=True)
            break
    samples = np.empty((NSAMP, n), dtype=np.complex128)
    got = 0
    if not blew:
        for s in range(NSAMP):
            for _ in range(STRIDE):
                u = model.step(u)
            if not finite(u):
                blew = True
                print(f"[c2-dim-v2] BLOWUP in sampling at {s}", flush=True)
                break
            samples[s] = u
            got = s + 1
            if s % 4000 == 0:
                print(f"[c2-dim-v2] sample {s}/{NSAMP}, elapsed {time.perf_counter()-t0:.0f}s", flush=True)
    samples = samples[:got]
    nsamp = samples.shape[0]
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "samples.npz", u=samples, k=k, nu_k2=model.nu_k2)
    print(f"[c2-dim-v2] collected {nsamp}; blew_up={blew}; window ~{nsamp*STRIDE*cfg.dt:.0f} time units", flush=True)

    e_mean = np.mean(np.abs(samples) ** 2, axis=0)
    eps_contrib = model.nu_k2 * e_mean
    real_shells = [j for j in range(n) if e_mean[j] >= ENERGY_FLOOR]
    print(f"[c2-dim-v2] dynamically-real shells (|u|^2>{ENERGY_FLOOR}): {[j+1 for j in real_shells]}", flush=True)

    # Body: z-scored Re/Im of the real shells only (real std -> no blow-up).
    cols = []
    for j in real_shells:
        cols.append(samples[:, j].real)
        cols.append(samples[:, j].imag)
    xb = np.stack(cols, axis=1)
    zb = (xb - xb.mean(0)) / (xb.std(0) + 1e-12)
    cb = np.corrcoef(zb, rowvar=False)
    evb = np.clip(np.linalg.eigvalsh(cb), 0, None)
    pr_real = float(evb.sum() ** 2 / ((evb ** 2).sum() + 1e-300))

    from sklearn.ensemble import HistGradientBoostingRegressor
    ntr = int(SPLIT * nsamp)
    rng = np.random.default_rng(cfg.random_seed + 1)
    perm = rng.permutation(nsamp)

    def r2(y, xin):
        ytr, yte = y[:ntr], y[ntr:]
        if yte.size < 2 or float(np.var(yte)) <= 0.0:
            return float("nan")
        m = HistGradientBoostingRegressor(max_iter=150, random_state=0).fit(xin[:ntr], ytr)
        p = m.predict(xin[ntr:])
        return float(1.0 - np.sum((yte - p) ** 2) / (np.sum((yte - np.mean(yte)) ** 2) + 1e-300))

    # Shadow = z-scored Re/Im of the low-K real shells; predict each real shell.
    K = 3
    shadow_cols = [c for j in real_shells[:K] for c in (samples[:, j].real, samples[:, j].imag)]
    shadow = np.stack(shadow_cols, axis=1)
    shadow = (shadow - shadow.mean(0)) / (shadow.std(0) + 1e-12)
    shadow_perm = shadow[perm]

    rows = []
    for idx, j in enumerate(real_shells):
        yj = zb[:, 2 * idx]  # real part (representative); could avg with imag
        rr = r2(yj, shadow)
        rp = r2(yj, shadow_perm)
        rows.append({"shell": j + 1, "k": float(k[j]), "E": float(e_mean[j]),
                     "eps": float(eps_contrib[j]), "r2_real": rr, "r2_perm": rp})

    # Decision over the ε-carrying real shells beyond the shadow band.
    eps_thr = float(np.quantile([row["eps"] for row in rows], 0.5))
    burst = [row for row in rows if row["eps"] >= eps_thr and row["shell"] > K]
    mean_real = float(np.nanmean([b["r2_real"] for b in burst])) if burst else float("nan")
    mean_perm = float(np.nanmean([b["r2_perm"] for b in burst])) if burst else float("nan")

    result = {
        "blew_up": blew, "n_samples": nsamp, "window_time_units": nsamp * STRIDE * cfg.dt,
        "real_shells_1indexed": [j + 1 for j in real_shells],
        "effective_rank_real_shells": pr_real, "of_dim": 2 * len(real_shells),
        "per_shell": rows, "shadow_K": K,
        "burst_shells_mean_r2_real": mean_real, "burst_shells_mean_r2_perm": mean_perm,
        "elapsed_seconds": time.perf_counter() - t0,
    }
    (OUT / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print("\n=== SABRA SHELL DIMENSIONALITY v2 (real shells, perm arbiter) ===")
    print(f"window: {nsamp} samples ~ {result['window_time_units']:.0f} time units (burst recurrence ~294 -> ~1 burst; UNDER-sampled)")
    print(f"effective rank of real-shell body: {pr_real:.1f} of {2*len(real_shells)}")
    print("shell  k_n      <|u|^2>    eps        R2_real  R2_perm  (real>>perm=slaved; real~perm=under-determined/artifact)")
    for row in rows:
        print(f"  {row['shell']:2d}  {row['k']:.3g}  {row['E']:.2e}  {row['eps']:.2e}   {row['r2_real']:+.3f}  {row['r2_perm']:+.3f}")
    print(f"\nburst-relevant real shells (>K, top-half eps): mean R2_real={mean_real:+.3f}  R2_perm={mean_perm:+.3f}")
    print("READ: real>>perm and high -> SLAVED (marginal). real~perm~0 -> genuinely under-determined (resists). real~perm<<0 -> split artifact (inconclusive).")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
