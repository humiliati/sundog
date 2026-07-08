#!/usr/bin/env python3
"""NSE-H3 rung 0 -- forcing-axis (k_f=3, G=200) formation admission (frozen:
docs/chatv2/NSE_H3_KF3_SCOPE.md section 3). READ-ONLY import of the frozen C1
harness. Truth-only; no kNN, twin, or fiber number is computed. Non-promotional.

Run:  python scripts/nse_h3_kf3_admission.py --out results/proof/nse-h3-kf3-g200-adm
Self-test: --self-test.
"""
import argparse, json, os, sys, time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1
from pde_c1_kolmogorov_cell import KolmogorovStepper, select_low_modes

# Frozen constants (scope section 3).
BURNIN, WINDOW, TAIL = 100_000, 500_000, 501
INTERVAL, LOOK, QCAL, CAL_STEPS = 50, 500, 0.70, 100_000
DAMP_LO, DAMP_HI, BLK_LO, BLK_HI, N_BLOCKS = 0.20, 0.80, 0.10, 0.90, 8
ATOM_EPS, ATOM_MASS, LIVE_IQR = 1e-6, 0.05, 1e-9


def make_cfg(grashof=200.0, burnin=BURNIN):
    return c1.RunConfig(
        preset=f"nse_h3_kf3_adm_g{int(grashof)}", grid_size=32, n_modes=16, k_signature=3,
        forcing_wavenumber=3, grashof=grashof, forcing_amplitude=1.0,
        viscosity=float(np.sqrt(1.0 / grashof)), dt=0.01, burnin_steps=burnin,
        sample_count=0, sample_interval_steps=INTERVAL, lookahead_steps=LOOK,
        n_min=30, delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01,
        e_max_burnin_fraction=1.0, random_seed=20260528, integrator="semi-implicit",
        signature_dimension=18, action_tiebreak="damp", adjudicator="knn",
        k_neighbors=30, delta_incompat=0.01, twin_k_neighbors=50,
        twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=QCAL,
        calibration_sample_count=0, calibration_gap_steps=0,
    )


def self_test():
    print("SELF-TEST (apparatus only; non-verdict)", flush=True)
    # T1: forced-mode force-insert -- (0,3) lands in the K=3 signature set, count 9
    wave = np.fft.fftfreq(32, d=1.0 / 32)
    low = select_low_modes(wave, 3, 3)
    assert len(low) == 9, "T1 count"
    forced = [(ix, iy) for ix, iy in low if wave[ix] == 0 and wave[iy] == 3]
    assert len(forced) == 1, "T1 forced-mode membership"
    print("  T1 PASS  (0,3) force-inserted; signature set count 9 (d=18)", flush=True)
    # T2: stepper builds and steps at k_f=3 (forcing_hat nonzero at the forced mode)
    stepper = KolmogorovStepper(make_cfg())
    assert np.abs(stepper.forcing_hat).max() > 0, "T2 forcing"
    u = stepper.initial_state()
    for _ in range(100):
        u = stepper.step(u)
    assert np.isfinite(stepper.low_energy(u)), "T2 step"
    print("  T2 PASS  k_f=3 stepper integrates finitely (100 steps)", flush=True)
    # T3: M window convention on a synthetic series (inclusive [s, s+LOOK])
    e = np.arange(1000, dtype=np.float64)
    m = sliding_window_view(e, LOOK + 1).max(axis=1)
    assert m[0] == e[LOOK] and m[10] == e[10 + LOOK], "T3"
    print("  T3 PASS  M(s) = max over [s, s+500] (inclusive)", flush=True)
    # T4: atom detector fires on constant, clean on continuum
    const = np.full(1000, 2.0)
    assert float(np.mean(np.abs(const - 2.0) <= ATOM_EPS)) > ATOM_MASS, "T4a"
    cont = np.linspace(0.0, 1.0, 1000)
    assert float(np.mean(np.abs(cont - 0.5) <= ATOM_EPS)) <= ATOM_MASS, "T4b"
    print("  T4 PASS  atom detector: fires on constant, clean on continuum", flush=True)
    print("SELF-TEST 4/4 PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--grashof", type=float, default=200.0)
    ap.add_argument("--burnin", type=int, default=BURNIN)
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.out is None:
        ap.error("--out required unless --self-test")
    os.makedirs(a.out, exist_ok=True)
    summary_path = os.path.join(a.out, "h3_admission.json")
    if os.path.exists(summary_path):
        print(f"[skip, done] {summary_path}", flush=True)
        return
    print(f"NSE_H3_KF3 rung 0  k_f=3 G={a.grashof:.0f}  burnin={a.burnin}  "
          f"LOCK-SEED  [NON-PROMOTIONAL]", flush=True)
    cfg = make_cfg(a.grashof, a.burnin)
    stepper = KolmogorovStepper(cfg)
    u = stepper.initial_state()
    t0 = time.time()
    for _ in range(a.burnin):
        u = stepper.step(u)
    total = WINDOW + TAIL
    e = np.empty(total, dtype=np.float64)
    for t in range(total):
        e[t] = stepper.low_energy(u)
        u = stepper.step(u)
        if (t + 1) % 100_000 == 0:
            print(f"(truth) {t + 1}/{total} [{time.time() - t0:.0f}s]", flush=True)
    if not np.all(np.isfinite(e)):
        result = {"branch": "NSE-H3-INPUT-UNPOWERED", "stage": "diagnostics_nonfinite"}
        json.dump(result, open(summary_path, "w"), indent=1)
        print(f"BRANCH: {result['branch']} ({result['stage']})", flush=True)
        return

    m = sliding_window_view(e, LOOK + 1).max(axis=1)  # M(s) for s in [0, WINDOW)
    starts = np.arange(0, WINDOW, INTERVAL)
    m_s = m[starts]
    cal_mask = starts < CAL_STEPS
    thr = float(np.quantile(m_s[cal_mask], QCAL))
    ev_starts, ev_m = starts[~cal_mask], m_s[~cal_mask]
    y = ev_m > thr
    damp = float(np.mean(y))
    blk = (ev_starts - CAL_STEPS) // ((WINDOW - CAL_STEPS) // N_BLOCKS)
    blk_damp = [float(np.mean(y[blk == b])) for b in range(N_BLOCKS)]
    atom = float(np.mean(np.abs(ev_m - thr) <= ATOM_EPS))
    iqr = float(np.subtract(*np.percentile(ev_m, [75, 25])))
    # regime character (reported, not gated)
    med = np.median(e[:WINDOW])
    detr = e[:WINDOW] - med
    ac = np.correlate(detr - detr.mean(), detr - detr.mean(), "full")[len(detr) - 1:]
    ac = ac / ac[0]
    first_zero = int(np.argmax(ac <= 0)) if np.any(ac <= 0) else -1

    print(f"(gates) blockwise damp table (FIRST read): "
          f"{['%.3f' % d for d in blk_damp]}", flush=True)
    fails = []
    if not (DAMP_LO <= damp <= DAMP_HI):
        fails.append(f"G1_damp={damp:.3f}")
    if not all(BLK_LO <= d <= BLK_HI for d in blk_damp):
        fails.append("G2_blockwise")
    if atom > ATOM_MASS:
        fails.append(f"G3_atom={atom:.3f}")
    if iqr < LIVE_IQR:
        fails.append(f"G4_liveness_iqr={iqr:.3e}")
    branch = "H3_CELL_ADMITTED_PROBE_TIER" if not fails else "NSE-H3-INPUT-UNPOWERED"
    result = {
        "scope": "NSE_H3_KF3_SCOPE.md", "cell": f"kf3_g{int(a.grashof)}_grid32",
        "seed": 20260528, "burnin": a.burnin,
        "threshold_q70": thr, "damp_heldout": damp, "blockwise_damp": blk_damp,
        "atom_mass": atom, "iqr_m": iqr, "n_cal": int(cal_mask.sum()),
        "n_eval": int(len(ev_m)),
        "regime_character_REPORTED": {
            "e_low_mean": float(np.mean(e[:WINDOW])), "e_low_std": float(np.std(e[:WINDOW])),
            "e_low_min": float(np.min(e[:WINDOW])), "e_low_max": float(np.max(e[:WINDOW])),
            "detrended_autocorr_first_zero_steps": first_zero,
        },
        "gate_failures": fails, "branch": branch,
    }
    json.dump(result, open(summary_path, "w"), indent=1)
    print(f"(gates) damp={damp:.3f} atom={atom:.3f} iqr={iqr:.3e} "
          f"env=[{result['regime_character_REPORTED']['e_low_min']:.3f},"
          f"{result['regime_character_REPORTED']['e_low_max']:.3f}] "
          f"ac0={first_zero}", flush=True)
    print(f"BRANCH: {branch}" + (f"  fails={fails}" if fails else ""), flush=True)
    print(f"(wrote) {summary_path}", flush=True)
    print("  (Truth-only formation admission; adjudication lives in the rung-1 locks.)",
          flush=True)


if __name__ == "__main__":
    main()
