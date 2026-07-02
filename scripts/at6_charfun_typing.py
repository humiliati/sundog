#!/usr/bin/env python3
"""AT-6 -- charFun typing of compact shadows (frozen spec: AT6_CHARFUN_TYPING_SPEC.md).

Does time-averaging (the trajectory-bag) keep regime/component-type decision labels and
wash phase/timing-type ones, per the in-tree ShadowDecay dichotomy? CPU post-processing on
self-generated streams; read-only import of the frozen C1 harness (file untouched).
Non-promotional; no control claim; no PDE theorem. Classes/gates frozen in the spec.
Run: python scripts/at6_charfun_typing.py
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1  # read-only import; harness file untouched

T_GRID = [1, 10, 50, 250, 1000, 5000]
STREAM, BURNIN, EVAL_STRIDE, GAP = 100_000, 100_000, 50, 5_000
FLOOR, SURVIVE = 0.55, 0.65
ROWS = {  # frozen classes (spec section 2)
    "R1_regime_Elow": "component", "R2_regime_Zlow": "component",
    "R3_mode_phase": "phase", "R4_rising": "phase", "R5_imminence": "phase",
}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "chatv2", "at6")


def make_cfg(grashof: float) -> "c1.RunConfig":
    """lock_v7-pinned cell, stream-emission posture (no lookahead machinery used)."""
    return c1.RunConfig(
        preset=f"at6_stream_g{int(grashof)}", grid_size=32, n_modes=16, k_signature=3,
        forcing_wavenumber=2, grashof=grashof, forcing_amplitude=1.0,
        viscosity=float(np.sqrt(1.0 / grashof)), dt=0.01, burnin_steps=BURNIN,
        sample_count=0, sample_interval_steps=EVAL_STRIDE, lookahead_steps=500,
        n_min=30, delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01,
        e_max_burnin_fraction=1.0, random_seed=0, integrator="semi-implicit",
        signature_dimension=18, action_tiebreak="damp", adjudicator="knn",
        k_neighbors=30, delta_incompat=0.01, twin_k_neighbors=50,
        twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=0.70,
        calibration_sample_count=0, calibration_gap_steps=0,
    )


def generate_stream(grashof: float):
    cfg = make_cfg(grashof)
    st = c1.KolmogorovStepper(cfg)
    w = st.initial_state()
    t0 = time.time()
    for _ in range(BURNIN):
        w = st.step(w)
    obs = np.empty((STREAM, 24))
    phase = np.empty(STREAM)  # arg of mode (1,0) for R3 + the mechanism check
    for t in range(STREAM):
        w = st.step(w)
        obs[t, :18] = st.signature(w)
        obs[t, 18] = st.low_energy(w)
        obs[t, 19] = st.low_enstrophy(w)
        obs[t, 20] = st.high_energy(w)
        obs[t, 21] = st.high_enstrophy(w)
        obs[t, 22] = st.palinstrophy(w)
        obs[t, 23] = st.top_shell_energy(w)
        phase[t] = np.angle(w[1, 0])
    print(f"  (stream) G={grashof:.0f}: {BURNIN}+{STREAM} steps in {time.time()-t0:.0f}s", flush=True)
    return obs, phase


def labels_from(obs, phase):
    e_low = obs[:, 18]
    z_low = obs[:, 19]
    look = np.array([e_low[t + 1:t + 501].max() if t + 501 <= len(e_low) else np.nan
                     for t in range(len(e_low))])
    y = {
        "R1_regime_Elow": e_low > np.median(e_low),
        "R2_regime_Zlow": z_low > np.median(z_low),
        "R3_mode_phase": np.cos(phase) > 0,
        "R4_rising": np.r_[np.zeros(50, bool), (e_low[50:] - e_low[:-50]) > 0],
        "R5_imminence": look > np.nanmedian(look),
    }
    valid = ~np.isnan(look)
    return y, valid


def running_mean(x, T):
    if T == 1:
        return x.copy()
    c = np.cumsum(x, axis=0, dtype=np.float64)
    out = np.full_like(x, np.nan, dtype=np.float64)
    out[T - 1:] = (c[T - 1:] - np.r_[np.zeros((1, x.shape[1])), c[:-T]][: len(c) - T + 1]) / T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", type=int, default=STREAM)
    args = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    os.makedirs(OUT_DIR, exist_ok=True)
    print("AT6_CHARFUN_TYPING  [NON-PROMOTIONAL, CPU]  spec: AT6_CHARFUN_TYPING_SPEC.md\n", flush=True)
    summary = {}
    for G in (200.0, 300.0):
        obs, phase = generate_stream(G)
        y_all, valid = labels_from(obs, phase)
        # eval points: every EVAL_STRIDE, past max window, label-valid
        pts = np.arange(max(T_GRID), len(obs) - 501, EVAL_STRIDE)
        pts = pts[valid[pts]]
        split = int(0.7 * len(pts))
        tr_pts, te_pts = pts[:split], pts[split:]
        te_pts = te_pts[te_pts > tr_pts[-1] + GAP]  # contiguous split + gap
        # mechanism check: |<e^{i theta}>_T|
        mech = {}
        eic = np.stack([np.cos(phase), np.sin(phase)], axis=1)
        for T in T_GRID:
            m = running_mean(eic, T)
            mech[T] = float(np.nanmean(np.linalg.norm(m[pts], axis=1)))
        print(f"\n[G={G:.0f}]  n_eval train/test = {len(tr_pts)}/{len(te_pts)}"
              f"  | mechanism |<e^i.theta>_T|: " +
              " ".join(f"T{T}:{mech[T]:.2f}" for T in T_GRID), flush=True)
        print(f"  {'row':>16} {'class':>10} " + " ".join(f"T={T:>5}" for T in T_GRID))
        acc = {}
        for row, cls in ROWS.items():
            accs = []
            yv = y_all[row]
            for T in T_GRID:
                X = running_mean(obs, T)
                mu, sd = X[tr_pts].mean(0), X[tr_pts].std(0) + 1e-9
                clf = LogisticRegression(max_iter=500)
                clf.fit((X[tr_pts] - mu) / sd, yv[tr_pts])
                accs.append(float(clf.score((X[te_pts] - mu) / sd, yv[te_pts])))
            acc[row] = accs
            print(f"  {row:>16} {cls:>10} " + " ".join(f"{a:>7.3f}" for a in accs), flush=True)
        # gates (frozen)
        if any(acc[r][0] < FLOOR for r in ROWS):
            verdict = "AT6_DEAD_APPARATUS"
        else:
            ok_T = [T_GRID[i] for i in range(len(T_GRID))
                    if all(acc[r][i] <= FLOOR for r, c in ROWS.items() if c == "phase")
                    and all(acc[r][i] >= SURVIVE for r, c in ROWS.items() if c == "component")]
            if ok_T:
                verdict = f"AT6_TYPING_CONFIRMED (matched T = {ok_T})"
            else:
                broken = []
                for i, T in enumerate(T_GRID):
                    ph_ok = [r for r, c in ROWS.items() if c == "phase" and acc[r][i] > FLOOR]
                    co_ok = [r for r, c in ROWS.items() if c == "component" and acc[r][i] < SURVIVE]
                    broken.append((T, ph_ok, co_ok))
                worst = min(broken, key=lambda b: len(b[1]) + len(b[2]))
                verdict = f"AT6_TYPING_BROKEN(rows={worst[1] + worst[2]} at best T={worst[0]})"
        summary[int(G)] = {"verdict": verdict, "acc": acc, "mechanism": mech,
                           "n_eval": [len(tr_pts), len(te_pts)]}
        print(f"  VERDICT[G={G:.0f}]: {verdict}", flush=True)
    with open(os.path.join(OUT_DIR, "at6_typing.json"), "w") as f:
        json.dump({"spec": "AT6_CHARFUN_TYPING_SPEC.md", "date": "2026-07-02",
                   "T_grid": T_GRID, "floor": FLOOR, "survive": SURVIVE,
                   "rows": ROWS, "regimes": summary}, f, indent=1)
    print(f"\n(wrote) results/chatv2/at6/at6_typing.json")
    print("  (Non-promotional. Types shadows, not resistance; no PDE theorem; no control claim.)")


if __name__ == "__main__":
    main()
