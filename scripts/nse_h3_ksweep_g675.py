#!/usr/bin/env python3
"""NSE-H3 state-recon K-sweep at G=675 (frozen spec:
docs/chatv2/NSE_H3_KSWEEP_SPEC.md). Post-processing on the K=6-shadow capture
(samples.npz + at2_export.npz): for K in 2..6, Phi_K/Q_K are rebuilt from the
72-dim emitted signature + captured high modes, labels are the recomputed
REGISTERED K=3-band pi_hat (fixed across the sweep), and the frozen
aggregate_state_recon + the registered control read run per K. Non-promotional.

  --self-test    mode bookkeeping (per-mode bit-match vs directly-built K-signatures)
                 + label recomputation micro-test
  --run OUT_DIR  gates -> sweep -> manifest + table
"""
import argparse, json, math, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1
from pde_c1_kolmogorov_cell import KolmogorovStepper, select_low_modes
from nse_h3_global_gauge import control_read, _cfg, _f

K_GRID = [2, 3, 4, 5, 6]
FVE_MARGINAL = 0.99
# Internal regression anchors (banked; spec section 3):
BANKED_E_MAX = 0.6946284050907446     # fallback_v7_g675_kf3 manifest
BANKED_DAMP = 0.30674
K3_FVE, K3_ACC, K3_TOL = 0.6322, 0.8866, 0.05  # global-gauge receipt, stride-4 subset


def recompute_labels(e_low_k3, calib_starts, adj_starts, look=500, q=0.70):
    """The registered pi_hat construction on the K=3 band (calibration-first)."""
    e = np.asarray(e_low_k3, dtype=np.float64)
    cal_m = np.array([float(np.max(e[s:s + look + 1])) for s in calib_starts])
    e_max = float(np.quantile(cal_m, q))
    adj_m = np.array([float(np.max(e[s:s + look + 1])) for s in adj_starts])
    return (adj_m > e_max).astype(np.int8), e_max


def geometry(kf=3, grid=32, n_modes=16):
    """Column |k| weights for the 72-dim low-6 signature and the high block."""
    st = KolmogorovStepper(_cfg(0, "smoke", kf))
    # st was built with k_signature=3; rebuild at 6 for the capture geometry:
    cfg6 = c1.RunConfig(**{**st.cfg.__dict__, "k_signature": 6})
    st6 = KolmogorovStepper(cfg6)
    k_low6 = np.array([math.sqrt(float(st6.k2[ix, iy])) for ix, iy in st6.low_indices
                       for _ in range(2)])
    k_high = np.array([math.sqrt(float(st6.k2[ix, iy])) for ix, iy in st6.high_indices
                       for _ in range(2)])
    return st6, k_low6, k_high


def self_test():
    print("SELF-TEST (K-sweep bookkeeping; synthetic)", flush=True)
    st6, k_low6, k_high = geometry()
    wave = np.fft.fftfreq(32, d=1.0 / 32)
    pos6 = {m: i for i, m in enumerate(st6.low_indices)}
    rng = np.random.default_rng(0)
    omega = np.fft.fft2(rng.standard_normal((32, 32)))
    omega[~st6.dealias_mask] = 0.0
    omega[0, 0] = 0.0
    sig72 = st6.signature(omega)
    for K in K_GRID:
        lowK = select_low_modes(wave, K, 3)
        assert all(m in pos6 for m in lowK), f"T1 K={K}: subset violated"
        # per-mode bit-match: the K-signature's components equal the extracted columns
        cfgK = c1.RunConfig(**{**st6.cfg.__dict__, "k_signature": K})
        stK = KolmogorovStepper(cfgK)
        sigK = stK.signature(omega)
        for j, m in enumerate(lowK):
            p = pos6[m]
            assert sigK[2 * j] == sig72[2 * p] and sigK[2 * j + 1] == sig72[2 * p + 1], \
                f"T1 K={K} mode {m}: extraction != direct"
    print(f"  T1 PASS  K=2..6 low sets are subsets of low-6; per-mode extraction "
          f"bit-matches directly-built signatures", flush=True)
    # T2: label recomputation micro-test
    e = np.abs(rng.standard_normal(5000)).astype(np.float32)
    cal = np.arange(0, 2000, 50)
    adj = np.arange(2500, 4400, 50)
    y, e_max = recompute_labels(e, cal, adj)
    cal_m = [max(e[s:s + 501]) for s in cal]
    assert abs(e_max - np.quantile(np.array(cal_m, dtype=np.float64), 0.70)) < 1e-12, "T2 e_max"
    assert 0.0 < y.mean() < 1.0, "T2 labels non-constant"
    print(f"  T2 PASS  label recomputation matches direct construction "
          f"(e_max {e_max:.4f}, damp {y.mean():.3f})", flush=True)
    print("SELF-TEST 2/2 PASS", flush=True)


def run(out_dir):
    z = np.load(os.path.join(out_dir, "samples.npz"))
    a2 = np.load(os.path.join(out_dir, "at2_export.npz"))
    sig72 = z["signatures"].astype(np.float64)
    high = z["high_modes"].astype(np.float64)
    n = len(sig72)
    st6, k_low6, k_high = geometry()
    assert sig72.shape[1] == len(k_low6) == 72, sig72.shape
    assert high.shape[1] == len(k_high), (high.shape, len(k_high))

    # Labels: the registered K=3-band pi_hat, recomputed (in-run K=6 actions NOT used).
    act, e_max = recompute_labels(a2["e_low_k3"], a2["calib_starts"], a2["adj_starts"])
    damp = float(act.mean())
    g1a = abs(e_max - BANKED_E_MAX) <= 1e-6
    g1b = abs(damp - BANKED_DAMP) <= 0.005
    print(f"[gate1] e_max {e_max:.10f} vs banked {BANKED_E_MAX:.10f} "
          f"[{'OK' if g1a else 'X'}]  damp {damp:.5f} vs {BANKED_DAMP} "
          f"[{'OK' if g1b else 'X'}]", flush=True)

    rows = []
    reject = not (g1a and g1b)
    if not reject:
        pos6 = {m: i for i, m in enumerate(st6.low_indices)}
        wave = np.fft.fftfreq(32, d=1.0 / 32)
        for K in K_GRID:
            t0 = time.time()
            cols = np.asarray(a2[f"cols_k{K}"], dtype=np.intp)
            comp = np.setdiff1d(np.arange(72), cols)
            phi = sig72[:, cols]
            qk = np.concatenate([sig72[:, comp], high], axis=1)
            comp_k = np.concatenate([k_low6[comp], k_high])
            # verdict-bearing cfg: the post reads are the registered reads on the
            # registered cell's trajectory (spec section 2); k_signature recorded.
            cfg = c1.RunConfig(**{**_cfg(n, "fallback_v7_g675_kf3", 3).__dict__,
                                  "k_signature": K})
            state, _, _, _ = c1.aggregate_state_recon(phi, qk, comp_k, cfg)
            ctrl, _, _ = control_read(phi, act, cfg.random_seed)
            row = {"K": K, "dim": int(2 * K * K), "phi_dim": int(phi.shape[1]),
                   "q_dim": int(qk.shape[1]),
                   "fve_vw": state.get("fve_Q_K_varweighted"),
                   "fve_enstrophy": state.get("fve_uniform_enstrophy_norm"),
                   "eqwt_median": state.get("r2_median_uniform_components"),
                   "perm": state.get("r2_E_high_perm_control"),
                   "state_verdict": state.get("verdict"),
                   "acc": ctrl["acc"], "majority": ctrl["majority"],
                   "margin": ctrl["acc_minus_majority"], "powered": ctrl["powered"],
                   "ctrl_perm_ok": ctrl["perm_ok"],
                   "elapsed_s": round(time.time() - t0, 1)}
            rows.append(row)
            print(f"[K={K}] dim {row['dim']}  FVE_vw={_f(row['fve_vw'])} "
                  f"(enst {_f(row['fve_enstrophy'])}, eqwt {_f(row['eqwt_median'])}, "
                  f"perm {_f(row['perm'])})  acc={_f(row['acc'])} vs maj "
                  f"{_f(row['majority'])} powered={row['powered']} "
                  f"[{row['elapsed_s']}s]", flush=True)
            if K == 3:
                g2a = abs(row["fve_vw"] - K3_FVE) <= K3_TOL
                g2b = abs(row["acc"] - K3_ACC) <= K3_TOL
                print(f"[gate2] K=3 rung vs global-gauge receipt: "
                      f"|FVE-{K3_FVE}|={abs(row['fve_vw']-K3_FVE):.4f} "
                      f"[{'OK' if g2a else 'X'}]  |acc-{K3_ACC}|="
                      f"{abs(row['acc']-K3_ACC):.4f} [{'OK' if g2b else 'X'}]",
                      flush=True)
                if not (g2a and g2b):
                    reject = True
                    break

    if reject:
        branch = "NSE-H3-KSWEEP-APPARATUS-REJECTED"
    else:
        valid = [r for r in rows if r["state_verdict"] == "STATE_RECON_MEASURED"
                 and r["ctrl_perm_ok"]]
        crossing = [r["K"] for r in valid if r["fve_vw"] >= FVE_MARGINAL]
        m_det_bracket = f"K={min(crossing)}" if crossing else ">6"
        window = [r["K"] for r in valid if r["powered"] and r["fve_vw"] < FVE_MARGINAL]
        branch = "NSE-H3-KSWEEP-WINDOW-MEASURED"
        print(f"[sweep] m_det bracket (first FVE >= {FVE_MARGINAL}): {m_det_bracket}",
              flush=True)
        print(f"[sweep] regime-2 window (powered AND FVE < {FVE_MARGINAL}): "
              f"K in {window}", flush=True)
    out = {"spec": "NSE_H3_KSWEEP_SPEC.md", "e_max": e_max, "damp": damp,
           "rows": rows, "branch": branch}
    if branch == "NSE-H3-KSWEEP-WINDOW-MEASURED":
        out["m_det_bracket"] = m_det_bracket
        out["regime2_window"] = window
    json.dump(out, open(os.path.join(out_dir, "ksweep_manifest.json"), "w"),
              indent=1, default=str)
    print(f"BRANCH: {branch}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--run", metavar="OUT_DIR")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.run:
        run(a.run)
    else:
        ap.error("use --self-test or --run")


if __name__ == "__main__":
    main()
