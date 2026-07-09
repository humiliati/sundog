"""H3-PL (HS1) pre-freeze tuning pass -- CONTROL-ONLY (touches no c-probe, fixes no threshold).

Finds the per-unit discrete-noise level SIGMA_D at which the TRAINED clf_d body's POOLED d-acc
lands in [0.75, 0.85] at lambda=2.0 (the de-saturated body all H3-PL arms will use), then
re-verifies the two hard gates C0 (raw mean-of-u washes c at high lambda) and C1 (a single raw
unit recovers c at lambda=0) on that body -- both are c-channel properties, independent of the
d-noise knob, so they must survive.

Reuses scripts/shadow_pooled_synthetic_v2.py verbatim (gen, Phi, train_body, phi_pool, c_r2,
d_acc); only monkeypatches the module global SIGMA_D across the sweep. Emits
results/atlas/hs1/desaturate_tuning.json with the pinnable value.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import shadow_pooled_synthetic_v2 as sub  # noqa: E402

SWEEP = [6.0, 7.0, 8.0, 8.5, 9.0, 10.0]
TARGET_LO, TARGET_HI = 0.75, 0.85
TUNE_SEED = 86235
N_TRAIN, N_PROBE = sub.N_TRAIN, sub.N_PROBE
OUT = os.path.join("results", "atlas", "hs1", "desaturate_tuning.json")


def measure(sigma_d):
    """Train clf_d at this SIGMA_D, return pooled + per-unit-all d-acc at lambda=2.0."""
    sub.SIGMA_D = sigma_d
    units_tr, c_tr, d_tr = sub.gen(N_TRAIN, sub.TRAIN_LAM, TUNE_SEED + 1)
    body, dfit = sub.train_body("clf_d", units_tr, c_tr, d_tr)
    units_ev, c_ev, d_ev = sub.gen(N_PROBE, 2.0, TUNE_SEED + 2)
    pooled = sub.phi_pool(body, units_ev)                     # (n, H)
    perunit = units_ev.reshape(units_ev.shape[0], -1)         # (n, K*F) raw per-unit d readout
    return dict(sigma_d=sigma_d, train_fit=round(dfit, 4),
                pooled_d_acc=round(sub.d_acc(pooled, d_ev), 4),
                perunit_d_acc=round(sub.d_acc(perunit, d_ev), 4)), body


def main():
    sweep_rows, bodies = [], {}
    for s in SWEEP:
        row, body = measure(s)
        bodies[s] = body
        sweep_rows.append(row)
        print(f"  SIGMA_D={s:.2f}: train_fit={row['train_fit']:.3f} "
              f"pooled_d_acc={row['pooled_d_acc']:.3f} perunit_d_acc={row['perunit_d_acc']:.3f}")

    in_band = [r for r in sweep_rows if TARGET_LO <= r["pooled_d_acc"] <= TARGET_HI]
    if in_band:
        chosen = min(in_band, key=lambda r: abs(r["pooled_d_acc"] - 0.80))
        status = "LANDED"
    else:
        chosen = min(sweep_rows, key=lambda r: abs(r["pooled_d_acc"] - 0.80))
        status = "REFINE"  # nearest miss; caller narrows the sweep around chosen.sigma_d
    sd = chosen["sigma_d"]

    # ---- re-verify c-channel gates on the chosen de-saturated body ---- #
    sub.SIGMA_D = sd
    u0, c0, _ = sub.gen(N_PROBE, 0.0, TUNE_SEED + 3)          # C1: single raw unit at lambda=0
    c1_val = sub.c_r2(u0[:, 0, :], c0)
    u2, c2, _ = sub.gen(N_PROBE, 2.0, TUNE_SEED + 4)          # C0: raw mean-of-u at lambda=2
    c0_val = sub.c_r2(u2.mean(axis=1), c2)
    gates = dict(C0_raw_mean_c_r2=round(c0_val, 4), C0_pass=bool(c0_val <= 0.10),
                 C1_single_unit_c_r2=round(c1_val, 4), C1_pass=bool(c1_val >= 0.50))

    out = dict(pass_name="hs1_desaturate", knob="SIGMA_D", sweep=sweep_rows,
               chosen_sigma_d=sd, chosen_pooled_d_acc=chosen["pooled_d_acc"],
               chosen_perunit_d_acc=chosen["perunit_d_acc"], status=status, gates=gates,
               target_band=[TARGET_LO, TARGET_HI], tune_seed=TUNE_SEED,
               note="control-only; touches no c-probe threshold; SIGMA_D to pin into prereg section 7")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSTATUS: {status}  chosen SIGMA_D={sd}  pooled_d_acc={chosen['pooled_d_acc']}")
    print(f"GATES: C0 raw-mean c-R2={gates['C0_raw_mean_c_r2']} (pass={gates['C0_pass']})  "
          f"C1 single-unit c-R2={gates['C1_single_unit_c_r2']} (pass={gates['C1_pass']})")


if __name__ == "__main__":
    main()
