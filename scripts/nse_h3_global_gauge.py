#!/usr/bin/env python3
"""NSE-H3 global-gauge (non-fiber) probe (frozen spec:
docs/chatv2/NSE_H3_GLOBAL_GAUGE_SPEC.md). Coverage-free regime-2 read on banked
samples.npz: state half = the UNTOUCHED frozen aggregate_state_recon (FVE(Q_K|Phi_K),
permutation-gated); control half = registered HGB classifier of action from Phi_K
(block split + 400-sample guard gap, majority + permuted-label controls).
Post-processing only; non-promotional.

  --self-test          synthetic (determined vs independent state; control + controls)
  --regress G200 G300  section-3 anchor precondition
  --run CELL_DIR       the G=675 read (registered stride-4 subsample to 50k) + branch
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1
from pde_c1_kolmogorov_cell import KolmogorovStepper

GAP = 400              # control-half guard gap (samples) = 20k steps
ACC_MARGIN = 0.10      # powered: acc - majority >= this (delta_action idiom)
PERM_TOL = 0.05        # permuted-label acc - majority < this
FVE_MARGINAL = 0.99    # the frozen receipt threshold (write_receipt_state_recon)


def _cfg(n, preset, kf):
    return c1.RunConfig(
        preset=preset, grid_size=32, n_modes=16, k_signature=3, forcing_wavenumber=kf,
        grashof=200.0, forcing_amplitude=1.0, viscosity=0.0707, dt=0.01, burnin_steps=0,
        sample_count=n, sample_interval_steps=50, lookahead_steps=500, n_min=30,
        delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01, e_max_burnin_fraction=1.0,
        random_seed=20260528, integrator="semi-implicit", signature_dimension=18,
        action_tiebreak="damp", adjudicator="state-recon", k_neighbors=30,
        delta_incompat=0.01, twin_k_neighbors=50, twin_delta_high_fraction=0.05,
        twin_high_norm_floor=1e-6, twin_min_witness_fraction=0.01,
        twin_min_unique_pairs=100, objective="portable-quantile",
        objective_quantile=0.70, calibration_sample_count=50000,
        calibration_gap_steps=5000,
    )


def comp_k_for(kf):
    st = KolmogorovStepper(_cfg(0, "smoke", kf))
    import math
    return np.array([math.sqrt(float(st.k2[ix, iy])) for ix, iy in st.high_indices
                     for _ in range(2)], dtype=np.float64)


def control_read(sig, act, seed):
    """Global control-sufficiency: action from Phi_K, block split + guard gap,
    majority floor + permuted-label control. Registered in spec section 1."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    n = len(sig)
    ntr = int(0.7 * n)
    tr = slice(0, ntr)
    te = slice(ntr + GAP, n)
    y_te = act[te]
    majority = float(max(y_te.mean(), 1 - y_te.mean()))
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(sig[tr], act[tr])
    acc = float(clf.score(sig[te], y_te))
    rng = np.random.default_rng(seed + 11)
    act_p = act[rng.permutation(n)]
    clf_p = HistGradientBoostingClassifier(max_iter=200, random_state=0).fit(sig[tr], act_p[tr])
    acc_p = float(clf_p.score(sig[te], act_p[te]))
    maj_p = float(max(act_p[te].mean(), 1 - act_p[te].mean()))
    powered = acc - majority >= ACC_MARGIN
    perm_ok = acc_p - maj_p < PERM_TOL
    return {"acc": acc, "majority": majority, "acc_minus_majority": acc - majority,
            "acc_perm": acc_p, "majority_perm": maj_p, "powered": bool(powered),
            "perm_ok": bool(perm_ok), "n_test": int(n - ntr - GAP)}, clf, te


def tercile_tables(sig, high, act, clf, te):
    """Reported stratification: one model per half, test metrics by energy tercile."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    E = np.sum(sig.astype(np.float64) ** 2, axis=1)
    q1, q2 = np.quantile(E, [1 / 3, 2 / 3])
    n = len(sig)
    ntr = int(0.7 * n)
    e_high = np.sum(high * high, axis=1)
    reg = HistGradientBoostingRegressor(max_iter=200, random_state=0).fit(sig[:ntr], e_high[:ntr])
    idx_te = np.arange(ntr + GAP, n)
    pred = reg.predict(sig[idx_te])
    rows = []
    for lab, m in (("lowE", E[idx_te] <= q1), ("midE", (E[idx_te] > q1) & (E[idx_te] <= q2)),
                   ("highE", E[idx_te] > q2)):
        sub = idx_te[m]
        if len(sub) < 50:
            rows.append({"band": lab, "n": int(len(sub))})
            continue
        y = e_high[sub]
        p = pred[m]
        r2 = float(1 - np.sum((y - p) ** 2) / (np.sum((y - y.mean()) ** 2) + 1e-300))
        acc = float(clf.score(sig[sub], act[sub]))
        maj = float(max(act[sub].mean(), 1 - act[sub].mean()))
        rows.append({"band": lab, "n": int(len(sub)), "r2_E_high": round(r2, 4),
                     "control_acc": round(acc, 4), "control_majority": round(maj, 4)})
    return rows


def read_cell(cell_dir, stride):
    z = np.load(os.path.join(cell_dir, "samples.npz"))
    preset = str(z["preset"])
    kf = 3 if "kf3" in preset else 2
    sig = z["signatures"][::stride].astype(np.float64)
    high = z["high_modes"][::stride].astype(np.float64)
    act = np.asarray(z["actions"][::stride]).astype(np.int8)
    ck = comp_k_for(kf)
    assert len(ck) == high.shape[1], f"comp_k dim {len(ck)} != high dim {high.shape[1]}"
    cfg = _cfg(len(sig), preset, kf)
    t0 = time.time()
    state, _, _, _ = c1.aggregate_state_recon(sig, high, ck, cfg)
    ctrl, clf, te = control_read(sig, act, cfg.random_seed)
    terc = tercile_tables(sig, high, act, clf, te)
    out = {"preset": preset, "n_used": int(len(sig)), "stride": stride,
           "state": {k: state.get(k) for k in (
               "verdict", "interpretable", "fve_Q_K_varweighted",
               "r2_E_high_from_signature", "r2_E_high_perm_control",
               "fve_components_covered", "r2_median_uniform_components",
               "fve_uniform_energy_norm", "fve_uniform_enstrophy_norm",
               "state_residual_varweighted", "state_residual_enstrophy_norm")},
           "control": ctrl, "terciles": terc,
           "elapsed_s": round(time.time() - t0, 1)}
    return out


def _f(x):
    """None/nan-tolerant number formatting -- a report must never kill a run."""
    try:
        return f"{float(x):.4f}"
    except (TypeError, ValueError):
        return str(x)


def branch_of(r):
    st, ct = r["state"], r["control"]
    if st["verdict"] == "STATE_RECON_ESTIMATOR_INVALID" or not ct["perm_ok"]:
        return "NSE-H3-GLOBAL-ESTIMATOR-INVALID"
    if not ct["powered"]:
        return "NSE-H3-GLOBAL-CONTROL-FAILS"
    fve = st["fve_Q_K_varweighted"]
    return ("NSE-H3-GLOBAL-REGIME2-NONMARGINAL" if fve < FVE_MARGINAL
            else "NSE-H3-GLOBAL-REGIME2-MARGINAL")


def pr(r):
    st, ct = r["state"], r["control"]
    print(f"  [{r['preset']}] n={r['n_used']}  state={st['verdict']}  "
          f"FVE_vw={_f(st['fve_Q_K_varweighted'])}  "
          f"(enstrophy {_f(st['fve_uniform_enstrophy_norm'])}, "
          f"eq-wt median {_f(st['r2_median_uniform_components'])})  "
          f"perm={_f(st['r2_E_high_perm_control'])}", flush=True)
    print(f"      control acc={_f(ct['acc'])} maj={_f(ct['majority'])} "
          f"(margin {_f(ct['acc_minus_majority'])}, powered={ct['powered']}, "
          f"perm {_f(ct['acc_perm'])} vs {_f(ct['majority_perm'])} ok={ct['perm_ok']})", flush=True)
    for row in r["terciles"]:
        print(f"      {row}", flush=True)


def self_test():
    print("SELF-TEST (global gauge; synthetic)", flush=True)
    rng = np.random.default_rng(0)
    # n >= 10k so HGB's auto early-stopping is active -- below that it overfits a
    # permuted target to R2 ~ -0.1 and trips the frozen validity gate (which is
    # calibrated for the 50k banked runs).
    n, hd = 12000, 24
    sig = rng.normal(0, 1, (n, 18))
    # T1 determined: high = linear map of sig + tiny noise -> FVE ~ 1
    A = rng.normal(0, 1, (18, hd))
    high1 = sig @ A + rng.normal(0, 0.05, (n, hd))
    cfg = _cfg(n, "lock_v7_g200", 2)
    st1, _, _, _ = c1.aggregate_state_recon(sig, high1, None, cfg)
    assert st1["verdict"] == "STATE_RECON_MEASURED", st1["verdict"]
    assert st1["fve_Q_K_varweighted"] > 0.9, f"T1 FVE {st1['fve_Q_K_varweighted']}"
    print(f"  T1 PASS  determined state: FVE {st1['fve_Q_K_varweighted']:.3f} > 0.9", flush=True)
    # T2 independent: high independent of sig -> FVE ~ 0
    high2 = rng.normal(0, 1, (n, hd))
    st2, _, _, _ = c1.aggregate_state_recon(sig, high2, None, cfg)
    assert st2["fve_Q_K_varweighted"] < 0.1, f"T2 FVE {st2['fve_Q_K_varweighted']}"
    print(f"  T2 PASS  independent state: FVE {st2['fve_Q_K_varweighted']:.3f} < 0.1", flush=True)
    # T3 control: action determined by sig -> powered; permuted -> at majority
    act = (sig[:, 0] > 0.5).astype(np.int8)
    ct, _, _ = control_read(sig, act, 0)
    assert ct["powered"] and ct["perm_ok"], ct
    print(f"  T3 PASS  control: acc {ct['acc']:.3f} vs maj {ct['majority']:.3f}, "
          f"perm ok ({ct['acc_perm']:.3f})", flush=True)
    print("SELF-TEST 3/3 PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--regress", nargs=2, metavar=("G200_DIR", "G300_DIR"))
    ap.add_argument("--run", metavar="CELL_DIR")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.regress:
        print("REGRESSION (section 3): anchors must be estimator-valid + control-powered", flush=True)
        ok = True
        for d in a.regress:
            r = read_cell(d, stride=1)
            json.dump(r, open(os.path.join(d, "global_gauge_manifest.json"), "w"), indent=1)
            pr(r)
            cell_ok = (r["state"]["verdict"] == "STATE_RECON_MEASURED"
                       and r["control"]["powered"] and r["control"]["perm_ok"])
            ok = ok and cell_ok
        print(f"REGRESSION: {'PASS -> G=675 global read unblocked' if ok else 'FAIL -> NSE-H3-GLOBAL-APPARATUS-REJECTED'}",
              flush=True)
        sys.exit(0 if ok else 1)
    if a.run:
        r = read_cell(a.run, stride=4)   # registered subsample: 200k -> 50k
        branch = branch_of(r)
        json.dump({**r, "branch": branch},
                  open(os.path.join(a.run, "global_gauge_manifest.json"), "w"), indent=1)
        pr(r)
        print(f"BRANCH: {branch}", flush=True)
        return
    ap.error("use --self-test, --regress, or --run")


if __name__ == "__main__":
    main()
