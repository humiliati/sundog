#!/usr/bin/env python3
"""NSE-H2 apparatus probe (NON-VERDICT): is the G=300 / grid-48 blow-up a dt/CFL
stability boundary? READ-ONLY import of the frozen C1 harness; no harness change.
Integrates the lock_v7_g300_n48 cell at dt in {0.01, 0.005, 0.0025} for up to
600k steps each, early-exit on the first non-finite E_low. A stable arm here is a
DIAGNOSIS AID for a possible v1.1 amendment, not a stability certificate for the
full 5.1M-step lock. Prints only apparatus facts; reads no verdict field.
"""
import os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1
from pde_c1_kolmogorov_cell import KolmogorovStepper

MAX_STEPS, CHECK_EVERY = 600_000, 100


def make_cfg(dt):
    return c1.RunConfig(
        preset="h2_dt_probe_nonverdict", grid_size=48, n_modes=24, k_signature=3,
        forcing_wavenumber=2, grashof=300.0, forcing_amplitude=1.0,
        viscosity=float(np.sqrt(1.0 / 300.0)), dt=dt, burnin_steps=0,
        sample_count=0, sample_interval_steps=50, lookahead_steps=500,
        n_min=30, delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01,
        e_max_burnin_fraction=1.0, random_seed=20260528, integrator="semi-implicit",
        signature_dimension=18, action_tiebreak="damp", adjudicator="knn",
        k_neighbors=30, delta_incompat=0.01, twin_k_neighbors=50,
        twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=0.70,
        calibration_sample_count=0, calibration_gap_steps=0,
    )


def probe(dt):
    stepper = KolmogorovStepper(make_cfg(dt))
    u = stepper.initial_state()
    t0 = time.time()
    e_max_seen = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for t in range(1, MAX_STEPS + 1):
            u = stepper.step(u)
            if t % CHECK_EVERY == 0:
                e = stepper.low_energy(u)
                if not np.isfinite(e):
                    return {"dt": dt, "stable": False, "first_nonfinite_step": t,
                            "e_max_seen": e_max_seen, "elapsed_s": round(time.time() - t0, 1)}
                e_max_seen = max(e_max_seen, e)
    return {"dt": dt, "stable": True, "steps": MAX_STEPS,
            "e_max_seen": e_max_seen, "elapsed_s": round(time.time() - t0, 1)}


def main():
    print("NSE_H2 dt/CFL PROBE  G=300 grid=48  [NON-VERDICT APPARATUS CHECK]", flush=True)
    for dt in (0.01, 0.005, 0.0025):
        r = probe(dt)
        if r["stable"]:
            print(f"  dt={dt:g}: STABLE through {r['steps']} steps "
                  f"(max E_low {r['e_max_seen']:.3f}) [{r['elapsed_s']}s]", flush=True)
            break
        print(f"  dt={dt:g}: NON-FINITE at step {r['first_nonfinite_step']} "
              f"(max E_low before {r['e_max_seen']:.3f}) [{r['elapsed_s']}s]", flush=True)
    print("  (Diagnosis aid only; not a certificate for the 5.1M-step lock.)", flush=True)


if __name__ == "__main__":
    main()
