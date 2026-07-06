#!/usr/bin/env python3
"""NSE-H1 rung 0 -- paired-action J-selector admission (frozen spec:
docs/chatv2/NSE_H1_JSELECTOR_SPEC.md). READ-ONLY import of the frozen C1 harness
(no harness change). Non-promotional; 32x32 truncation; admission only -- no
fiber-adjudication number is computed or read here.

Run:   python scripts/nse_h1_jselector_admission.py --grashof 200 --out results/proof/nse-h1-g200
Smoke: add --smoke (cal 20k / eval 40k, non-verdict). Self-test: --self-test.
Resumable: truth + rollout checkpoints under --out.
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1
from pde_c1_kolmogorov_cell import KolmogorovStepper, select_low_modes  # read-only

# Frozen constants (spec section 1-3). Do not retune after any read.
BURNIN, CAL_STEPS, GAP, EVAL_STEPS = 100_000, 100_000, 5_000, 400_000
STRIDE, TAU, TAU_ACT, MU_ACT, QCAL = 250, 500, 100, 10.0, 0.70
MU_ACT_V11 = 1.0          # spec section 8 (v1.1 formation amendment)
BENEFIT_MASS_V11 = 0.05   # G6 (v1.1 only)
DAMP_LO, DAMP_HI, BLK_LO, BLK_HI, N_BLOCKS = 0.20, 0.40, 0.10, 0.60, 8
ATOM_EPS, ATOM_MASS = 1e-6, 0.05
LIVE_ZERO, LIVE_ZFRAC, LIVE_IQR = 1e-12, 0.10, 1e-9
MIN_EVAL, MIN_BLK = 800, 100


def make_cfg(grashof):
    return c1.RunConfig(
        preset=f"nse_h1_g{int(grashof)}", grid_size=32, n_modes=16, k_signature=3,
        forcing_wavenumber=2, grashof=grashof, forcing_amplitude=1.0,
        viscosity=float(np.sqrt(1.0 / grashof)), dt=0.01, burnin_steps=BURNIN,
        sample_count=0, sample_interval_steps=STRIDE, lookahead_steps=TAU,
        n_min=30, delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01,
        e_max_burnin_fraction=1.0, random_seed=0, integrator="semi-implicit",
        signature_dimension=18, action_tiebreak="damp", adjudicator="knn",
        k_neighbors=30, delta_incompat=0.01, twin_k_neighbors=50,
        twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=QCAL,
        calibration_sample_count=0, calibration_gap_steps=0,
    )


def k3_mask(stepper):
    m = stepper.cfg.grid_size
    wave = np.fft.fftfreq(m, d=1.0 / m)
    mask = np.zeros((m, m), dtype=bool)
    for ix, iy in select_low_modes(wave, 3, stepper.cfg.forcing_wavenumber):
        mask[ix, iy] = True
        mask[(m - ix) % m, (m - iy) % m] = True
    return mask


def damped_step(stepper, v, mask, mu):
    """Semi-implicit step with the registered actuator -mu*P_K3(v) at the explicit
    stage (AT-3's nudging form with target 0)."""
    dt = stepper.cfg.dt
    explicit = v + dt * (-stepper.nonlinear_hat(v) + stepper.forcing_hat)
    explicit[mask] -= dt * mu * v[mask]
    nxt = explicit / (1.0 + dt * stepper.cfg.viscosity * stepper.k2)
    nxt[~stepper.dealias_mask] = 0.0
    nxt[stepper.zero_mean] = 0.0
    return nxt


def j1_rollout(stepper, state, mask, mu):
    """J_1(s) = max E_low over (s, s+TAU] under damp_low_band (exclusive window)."""
    v = state.copy()
    mx = -np.inf
    for t in range(1, TAU + 1):
        v = damped_step(stepper, v, mask, mu) if t <= TAU_ACT else stepper.step(v)
        e = stepper.low_energy(v)
        if e > mx:
            mx = e
    return mx


def atom_check(delta, m):
    near = float(np.mean(np.abs(delta - m) <= ATOM_EPS))
    below = float(np.mean((delta < m) & (delta >= m - ATOM_EPS)))
    above = float(np.mean((delta >= m) & (delta <= m + ATOM_EPS)))
    return {"mass_near_m": near, "straddle_below": below, "straddle_above": above}


def self_test():
    print("SELF-TEST (apparatus only; non-verdict)", flush=True)
    cfg = make_cfg(200.0)
    stepper = KolmogorovStepper(cfg)
    mask = k3_mask(stepper)
    u = stepper.initial_state()
    for _ in range(2000):
        u = stepper.step(u)
    # T1: mu=0 damped step bitwise-equals the plain step
    assert np.array_equal(damped_step(stepper, u, mask, 0.0), stepper.step(u)), "T1"
    print("  T1 PASS  damped_step(mu=0) == step (bitwise)", flush=True)
    # T2: damped_step equals AT-3 nudged_step with zero target
    from pde_c1_nudging_ledger import nudged_step
    zero = np.zeros_like(u)
    assert np.array_equal(damped_step(stepper, u, mask, MU_ACT),
                          nudged_step(stepper, u, zero, mask, MU_ACT)), "T2"
    print("  T2 PASS  damped_step == nudged_step(target=0) (bitwise)", flush=True)
    # T3: sustained damping reduces E_low vs free run at the same offset
    vd, vf = u.copy(), u.copy()
    for _ in range(TAU_ACT):
        vd = damped_step(stepper, vd, mask, MU_ACT)
        vf = stepper.step(vf)
    assert stepper.low_energy(vd) < stepper.low_energy(vf), "T3"
    print(f"  T3 PASS  E_low damped {stepper.low_energy(vd):.4f} < free "
          f"{stepper.low_energy(vf):.4f} after {TAU_ACT} steps", flush=True)
    # T4: series-derived J_0 matches an explicit no_op rollout from the captured state
    e_mini = np.empty(TAU + 51)
    uu = u.copy()
    cap = None
    for t in range(TAU + 51):
        if t == 50:
            cap = uu.copy()
        e_mini[t] = stepper.low_energy(uu)
        uu = stepper.step(uu)
    j0_series = float(np.max(e_mini[51:51 + TAU]))  # (s, s+TAU], s=50 -> indices 51..550
    v = cap.copy()
    mx = -np.inf
    for t in range(1, TAU + 1):
        v = stepper.step(v)
        mx = max(mx, stepper.low_energy(v))
    assert mx == j0_series, "T4"
    print("  T4 PASS  series J_0 == explicit no_op rollout (bitwise)", flush=True)
    # T5: quantile targeting hits damp 0.30 on synthetic data
    x = np.random.default_rng(0).standard_normal(10_000)
    m = float(np.quantile(x, QCAL))
    assert abs(float(np.mean(x >= m)) - 0.30) < 0.001, "T5"
    print("  T5 PASS  q0.70 margin targets damp 0.300 on synthetic data", flush=True)
    # T6: atom detector fires on a constant, stays quiet on a continuum
    const = np.full(1000, 1.234)
    assert atom_check(const, 1.234)["mass_near_m"] > ATOM_MASS, "T6a"
    cont = np.linspace(0.0, 1.0, 1000)
    assert atom_check(cont, 0.5)["mass_near_m"] <= ATOM_MASS, "T6b"
    print("  T6 PASS  atom detector: fires on constant, clean on continuum", flush=True)
    print("SELF-TEST 6/6 PASS", flush=True)


def run_cell(grashof, out, smoke, fver):
    burnin = 20_000 if smoke else BURNIN
    cal_steps = 20_000 if smoke else CAL_STEPS
    eval_steps = 40_000 if smoke else EVAL_STEPS
    n_blocks = 4 if smoke else N_BLOCKS
    mu_act = MU_ACT_V11 if fver == "v1.1" else MU_ACT
    os.makedirs(out, exist_ok=True)
    summary_path = os.path.join(out, "h1_summary.json")
    if os.path.exists(summary_path):
        print(f"[skip, done] {summary_path}", flush=True)
        return
    print(f"NSE_H1_JSELECTOR rung 0 {fver}  G={grashof:.0f}  mu_act={mu_act:g}  "
          f"{'SMOKE (non-verdict)' if smoke else 'LOCK'}  [NON-PROMOTIONAL]", flush=True)
    cfg = make_cfg(grashof)
    stepper = KolmogorovStepper(cfg)
    mask = k3_mask(stepper)
    off = cal_steps + GAP
    total = off + eval_steps + TAU + 1
    cal_grid = np.arange(0, cal_steps - TAU, STRIDE)
    eval_grid = np.arange(off, off + eval_steps, STRIDE)
    instants = np.concatenate([cal_grid, eval_grid])

    truth_ckpt = os.path.join(out, "truth_ckpt.npz")
    if os.path.exists(truth_ckpt):
        z = np.load(truth_ckpt)
        e_series, states = z["e_series"], z["states"]
        print(f"(truth) resumed from checkpoint ({len(e_series)} steps)", flush=True)
    else:
        t0 = time.time()
        u = stepper.initial_state()
        for _ in range(burnin):
            u = stepper.step(u)
        e_series = np.empty(total, dtype=np.float64)
        states = np.empty((len(instants), cfg.grid_size, cfg.grid_size), dtype=np.complex128)
        pos = {int(s): i for i, s in enumerate(instants)}
        uu = u
        for t in range(total):
            e_series[t] = stepper.low_energy(uu)
            i = pos.get(t)
            if i is not None:
                states[i] = uu
            uu = stepper.step(uu)
            if (t + 1) % 100_000 == 0:
                print(f"(truth) {t + 1}/{total} steps [{time.time() - t0:.0f}s]", flush=True)
        np.savez_compressed(truth_ckpt, e_series=e_series, states=states, instants=instants)
        print(f"(truth) done: {total} steps + burn-in {burnin} [{time.time() - t0:.0f}s]", flush=True)

    # J_0 off the truth series -- exclusive window (s, s+TAU]
    j0 = np.array([float(np.max(e_series[s + 1:s + TAU + 1])) for s in instants])
    # pi_hat anchor -- house INCLUSIVE window [s, s+TAU] (matches the banked object)
    j0_incl = np.array([float(np.max(e_series[s:s + TAU + 1])) for s in instants])

    # J_1 rollouts (the compute body), resumable
    roll_ckpt = os.path.join(out, "rollout_ckpt.npz")
    j1 = np.full(len(instants), np.nan)
    start = 0
    if os.path.exists(roll_ckpt):
        z = np.load(roll_ckpt)
        j1, start = z["j1"], int(z["done"])
        print(f"(rollout) resumed at {start}/{len(instants)}", flush=True)
    t0 = time.time()
    for i in range(start, len(instants)):
        j1[i] = j1_rollout(stepper, states[i], mask, mu_act)
        if (i + 1) % 200 == 0 or i + 1 == len(instants):
            np.savez(roll_ckpt, j1=j1, done=i + 1)
            rate = (i + 1 - start) / max(time.time() - t0, 1e-9)
            print(f"(rollout) {i + 1}/{len(instants)}  "
                  f"[{time.time() - t0:.0f}s, {rate:.2f} inst/s]", flush=True)

    n_cal = len(cal_grid)
    delta = j0 - j1
    d_cal, d_eval = delta[:n_cal], delta[n_cal:]
    m = float(np.quantile(d_cal, QCAL))
    e_max_cal = float(np.quantile(j0_incl[:n_cal], QCAL))
    a_eval = d_eval >= m
    y_pi = j0_incl[n_cal:] > e_max_cal

    damp = float(np.mean(a_eval))
    blk_size = eval_steps // n_blocks
    blk_idx = (eval_grid - off) // blk_size
    blk_damp = [float(np.mean(a_eval[blk_idx == b])) for b in range(n_blocks)]
    blk_n = [int(np.sum(blk_idx == b)) for b in range(n_blocks)]
    atom = atom_check(d_eval, m)
    zfrac = float(np.mean(np.abs(d_eval) <= LIVE_ZERO))
    iqr = float(np.subtract(*np.percentile(d_eval, [75, 25])))
    overlap = float(np.mean(a_eval == y_pi))
    ac1 = float(np.corrcoef(a_eval[:-1].astype(float), a_eval[1:].astype(float))[0, 1])

    print(f"(gates) blockwise damp table (FIRST read): "
          f"{['%.3f' % d for d in blk_damp]} n={blk_n}", flush=True)
    fails = []
    if not (DAMP_LO <= damp <= DAMP_HI):
        fails.append(f"G1_damp={damp:.3f}")
    if not all(BLK_LO <= d <= BLK_HI for d in blk_damp):
        fails.append("G2_blockwise")
    if atom["mass_near_m"] > ATOM_MASS:
        fails.append(f"G3_atom={atom['mass_near_m']:.3f}")
    if zfrac > LIVE_ZFRAC or iqr < LIVE_IQR:
        fails.append(f"G4_liveness zfrac={zfrac:.3f} iqr={iqr:.3e}")
    if len(a_eval) < MIN_EVAL or any(n < MIN_BLK for n in blk_n):
        fails.append("G5_power")
    pos_frac = float(np.mean(d_eval > 0))
    if fver == "v1.1" and pos_frac < BENEFIT_MASS_V11:
        fails.append(f"G6_benefit_mass={pos_frac:.4f}")
    if smoke:
        branch = "SMOKE_NON_VERDICT"
    elif not fails:
        branch = "H1_CELL_ADMITTED"
    else:
        branch = "NSE-H1-UNPOWERED"

    sigs = np.array([stepper.signature(states[i]) for i in range(len(instants))])
    np.savez_compressed(
        os.path.join(out, "h1_export.npz"),
        s=instants, n_cal=n_cal, sig=sigs, j0=j0, j0_incl=j0_incl, j1=j1,
        delta=delta, a_j=np.concatenate([d_cal >= m, a_eval]),
        y_pi=np.concatenate([j0_incl[:n_cal] > e_max_cal, y_pi]),
        m=m, e_max_cal=e_max_cal)
    summary = {
        "spec": "NSE_H1_JSELECTOR_SPEC.md", "preset": cfg.preset, "grashof": grashof,
        "formation_version": fver, "mu_act": mu_act,
        "smoke": smoke, "n_cal": n_cal, "n_eval": int(len(a_eval)),
        "m_margin": m, "e_max_cal": e_max_cal,
        "benefit_mass_eval": pos_frac,
        "benefit_mass_cal": float(np.mean(d_cal > 0)),
        "m_sign_fence": ("benefit-positive-at-margin" if m > 0
                         else "mixed-least-harm-ranking"),
        "damp_cal": float(np.mean(d_cal >= m)), "damp_eval": damp,
        "blockwise_damp": blk_damp, "blockwise_n": blk_n, "atom": atom,
        "liveness": {"zero_frac": zfrac, "iqr": iqr,
                     "median_j0": float(np.median(j0[n_cal:])),
                     "median_j1": float(np.median(j1[n_cal:])),
                     "median_delta": float(np.median(d_eval))},
        "overlap_pi_hat_REPORTED_NOT_GATED": overlap, "a_j_lag1_autocorr": ac1,
        "gate_failures": fails, "branch": branch,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"(gates) damp_eval={damp:.3f} atom={atom['mass_near_m']:.3f} "
          f"zfrac={zfrac:.3f} iqr={iqr:.3e} overlap={overlap:.3f} "
          f"benefit_mass={pos_frac:.4f} m={m:+.5f}", flush=True)
    print(f"BRANCH: {branch}" + (f"  fails={fails}" if fails else ""), flush=True)
    print(f"(wrote) {summary_path}", flush=True)
    print("  (Non-promotional. The verdict comes from the receipt evaluation, not this log.)",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grashof", type=float)
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--formation-version", choices=["v1", "v1.1"], default="v1")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.grashof is None or a.out is None:
        ap.error("--grashof and --out are required unless --self-test")
    run_cell(a.grashof, a.out, a.smoke, a.formation_version)


if __name__ == "__main__":
    main()
