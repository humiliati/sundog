#!/usr/bin/env python3
"""AT-3 -- the maintained ledger: AOT nudging at sub-determining budget (frozen spec:
docs/chatv2/AT3_NUDGING_LEDGER_SPEC.md). New script; READ-ONLY import of the frozen C1
harness (no harness change). Non-promotional; 32x32 truncation; no NSE sync theorem.

Per (K_obs, mu) config: integrate truth u (seed 0) + nudged ledger v (fresh field, seed 1)
for a 500k-step window post-burnin (spec v1.1); record state-sync error and the ledger's decision
readout vs the true frozen J_q(500) actions. Controls: scrambled-observation ledger
(mu=10), 9-mode Galerkin decision-only twin. Resumable: one JSON per config.
Run: python scripts/pde_c1_nudging_ledger.py --grashof 300 --out results/proof/at3-g300
Smoke: add --smoke (20k window, K in {2,6}, mu=10; fallback threshold; non-verdict).
"""
import argparse, json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pde_c1_kolmogorov_cell import KolmogorovStepper, select_low_modes  # read-only
from at2b_growth_law import lookmean  # noqa: F401 (parity of imports; not used)

BURNIN, WINDOW, INTERVAL, LOOK, Q = 100_000, 500_000, 50, 500, 0.70  # v1.1: window 200k->500k (envelope wander)
K_GRID, MU_GRID, MU_SCR = [1, 2, 3, 4, 5, 6], [1.0, 10.0, 50.0], 10.0
SYNC_LINE, FLOOR_LINE, TRANSIENT_SKIP = 0.01, 0.10, 0.25
CAL_COUNT, CAL_GAP = 50_000, 5_000   # calibration block PRECEDES the window (C1 convention)


def make_cfg(grashof):
    from at2_growth_law import CFG as _  # noqa: F401
    import pde_c1_kolmogorov_cell as c1
    return c1.RunConfig(
        preset=f"at3_ledger_g{int(grashof)}", grid_size=32, n_modes=16, k_signature=3,
        forcing_wavenumber=2, grashof=grashof, forcing_amplitude=1.0,
        viscosity=float(np.sqrt(1.0 / grashof)), dt=0.01, burnin_steps=BURNIN,
        sample_count=0, sample_interval_steps=INTERVAL, lookahead_steps=LOOK,
        n_min=30, delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01,
        e_max_burnin_fraction=1.0, random_seed=0, integrator="semi-implicit",
        signature_dimension=18, action_tiebreak="damp", adjudicator="knn",
        k_neighbors=30, delta_incompat=0.01, twin_k_neighbors=50,
        twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=Q,
        calibration_sample_count=0, calibration_gap_steps=0,
    )


def obs_mask(stepper, k_obs):
    m = stepper.cfg.grid_size
    wave = np.fft.fftfreq(m, d=1.0 / m)
    mask = np.zeros((m, m), dtype=bool)
    for ix, iy in select_low_modes(wave, k_obs, stepper.cfg.forcing_wavenumber):
        mask[ix, iy] = True
        mask[(m - ix) % m, (m - iy) % m] = True
    return mask


def k3_mask(stepper):
    return obs_mask(stepper, 3)


def nudged_step(stepper, v, u_prev, mask, mu):
    """Semi-implicit step with the AOT term -mu*P_K(v-u) added at the explicit stage."""
    dt = stepper.cfg.dt
    explicit = v + dt * (-stepper.nonlinear_hat(v) + stepper.forcing_hat)
    explicit[mask] -= dt * mu * (v[mask] - u_prev[mask])
    nxt = explicit / (1.0 + dt * stepper.cfg.viscosity * stepper.k2)
    nxt[~stepper.dealias_mask] = 0.0
    nxt[stepper.zero_mean] = 0.0
    return nxt


def run_config(cfg, u0, e_series_true, e_max_true, k_obs, mu, mode, window, rng_scr=None):
    """One ledger run. mode: full | twin | scrambled. Returns summary dict."""
    import pde_c1_kolmogorov_cell as c1
    stepper = KolmogorovStepper(cfg)
    mask = obs_mask(stepper, k_obs)
    m3 = k3_mask(stepper)
    rng_v = np.random.default_rng(1)
    # fresh random field, built like initial_state but with seed 1
    m = cfg.grid_size
    noise = rng_v.standard_normal((m, m))
    v = np.fft.fft2(noise)
    v[~stepper.dealias_mask] = 0.0
    v[stepper.zero_mean] = 0.0
    u = u0.copy()
    obs_buf = []  # for scrambled: past observations
    errs, sigs, e_led = [], [], []
    t0 = time.time()
    for t in range(window):
        u_obs = u
        if mode == "scrambled":
            obs_buf.append(u[mask].copy())
            j = int(rng_scr.integers(0, len(obs_buf)))
            u_obs = u.copy()
            u_obs[mask] = obs_buf[j]
        v = nudged_step(stepper, v, u_obs, mask, mu)
        if mode == "twin":
            v[~m3] = 0.0  # 9-mode Galerkin truncation (closure-free, registered crude)
        u = stepper.step(u)
        if t % INTERVAL == 0:
            errs.append(float(np.linalg.norm(v - u) / max(np.linalg.norm(u), 1e-30)))
            sigs.append(stepper.signature(v))
            e_led.append(t)
    errs = np.array(errs)
    sigs = np.array(sigs)
    # sync classification (frozen)
    tail = errs[int(len(errs) * (1 - TRANSIENT_SKIP)):]
    med_tail = float(np.median(tail))
    state = ("synchronized" if med_tail < SYNC_LINE
             else "state_insufficient" if med_tail >= FLOOR_LINE else "transient")
    # decision readout: logistic on Phi_K3(v) vs true actions, post-transient
    from sklearn.linear_model import LogisticRegression
    start = int(len(sigs) * TRANSIENT_SKIP)
    steps_idx = np.array(e_led[start:])
    y = (np.array([float(np.max(e_series_true[s:s + LOOK + 1])) for s in steps_idx])
         > e_max_true).astype(int)
    X = sigs[start:]
    split = int(0.7 * len(y))
    gap = max(1, 2500 // INTERVAL)
    tr = slice(0, split)
    te = slice(split + gap, len(y))
    acc = maj = float("nan")
    if len(set(y[tr].tolist())) > 1 and len(y[te.start:]) > 20:
        mu_x, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
        clf = LogisticRegression(max_iter=500).fit((X[tr] - mu_x) / sd, y[tr])
        acc = float(clf.score((X[te] - mu_x) / sd, y[te]))
        maj = float(max(np.mean(y[te]), 1 - np.mean(y[te])))
    return {"k_obs": k_obs, "mu": mu, "mode": mode, "median_tail_err": med_tail,
            "state": state, "decision_acc": acc, "majority": maj,
            "n_eval": int(len(y)), "elapsed_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grashof", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    window = 20_000 if a.smoke else WINDOW  # smoke sized so the gapped split leaves a test set
    k_grid = [2, 6] if a.smoke else K_GRID
    mu_grid = [MU_SCR] if a.smoke else MU_GRID
    os.makedirs(a.out, exist_ok=True)
    print(f"AT3_NUDGING_LEDGER  G={a.grashof:.0f}  window={window}  "
          f"{'SMOKE (non-verdict)' if a.smoke else 'LOCK'}  [NON-PROMOTIONAL]", flush=True)

    cfg = make_cfg(a.grashof)
    stepper = KolmogorovStepper(cfg)
    # Truth: burn-in, then CALIBRATION block, gap, THEN the ledger window (the registered
    # C1 ordering — calibration precedes adjudication). u0 = state at window start.
    u = stepper.initial_state()
    for _ in range(BURNIN):
        u = stepper.step(u)
    total = CAL_COUNT + CAL_GAP + window + LOOK + 1
    e_series = np.empty(total, dtype=np.float64)
    uu = u.copy()
    u0 = None
    off = CAL_COUNT + CAL_GAP
    for t in range(total):
        sig = stepper.signature(uu)
        e_series[t] = float(np.dot(sig, sig))
        if t == off:
            u0 = uu.copy()
        uu = stepper.step(uu)
    cal_starts = np.arange(0, CAL_COUNT - LOOK - 1, INTERVAL)
    cal_m = np.array([float(np.max(e_series[s:s + LOOK + 1])) for s in cal_starts])
    e_max_true = float(np.quantile(cal_m, Q))
    e_win = e_series[off:]
    win_starts = np.arange(0, window, INTERVAL)
    damp_win = float(np.mean(np.array(
        [float(np.max(e_win[s:s + LOOK + 1])) for s in win_starts]) > e_max_true))
    print(f"(truth) burn-in done; e_max(500) = {e_max_true:.4f}; calib n = {len(cal_starts)}; "
          f"realized in-window damp = {damp_win:.3f}", flush=True)
    if a.smoke and (damp_win == 0.0 or damp_win == 1.0):
        e_max_true = float(np.median(np.array(
            [float(np.max(e_win[s:s + LOOK + 1])) for s in win_starts])))
        print(f"(SMOKE FALLBACK) degenerate window damp -> threshold reset to window "
              f"median {e_max_true:.4f} (readout-path validation ONLY; non-verdict)", flush=True)

    configs = []
    for k in k_grid:
        for mu in mu_grid:
            configs.append((k, mu, "full"))
            configs.append((k, mu, "twin"))
        configs.append((k, MU_SCR, "scrambled"))
    results = []
    for k, mu, mode in configs:
        tag = f"k{k}_mu{mu:g}_{mode}" + ("_smoke" if a.smoke else "")
        path = os.path.join(a.out, f"at3_{tag}.json")
        if os.path.exists(path):
            results.append(json.load(open(path)))
            print(f"  [skip, done] {tag}", flush=True)
            continue
        rng_scr = np.random.default_rng(2) if mode == "scrambled" else None
        r = run_config(cfg, u0, e_win, e_max_true, k, mu, mode, window, rng_scr)
        r["smoke"] = a.smoke
        with open(path, "w") as f:
            json.dump(r, f, indent=1)
        results.append(r)
        print(f"  {tag}: err_tail={r['median_tail_err']:.4f} ({r['state']}) "
              f"acc={r['decision_acc']:.3f} maj={r['majority']:.3f} "
              f"[{r['elapsed_s']}s]", flush=True)
    with open(os.path.join(a.out, "at3_summary.json"), "w") as f:
        json.dump({"spec": "AT3_NUDGING_LEDGER_SPEC.md", "grashof": a.grashof,
                   "smoke": a.smoke, "e_max_true": e_max_true, "results": results},
                  f, indent=1)
    print(f"(wrote) {a.out}/at3_summary.json")
    print("  (Non-promotional. Verdicts come from the receipt evaluation, not this log.)")


if __name__ == "__main__":
    main()
