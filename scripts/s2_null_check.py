#!/usr/bin/env python
"""S2 shuffled-parity null (PHASE5 §3.12 verification).

Guards the discrete-determines legs (ice-phase, handedness) against a label-leak `disc=1.0`: if the
features genuinely encode x_d, then scoring against a SHUFFLED x_d must collapse to chance (disc ~ 0).
A high disc on shuffled labels would mean the harness/feature can predict ANY label (a leak).

Run: python scripts/s2_null_check.py
"""
import sys
import numpy as np
sys.path.insert(0, "scripts")
import pvnp_phase5_lossiness_crossover as H

SEED = 20260605
N = 800
fail = 0


def one(name, gen, noise, lam):
    rng = H.default_rng(SEED + int(round(lam * 1000)) + 7)
    X, xc, xd = gen(N, lam, rng, noise)
    true = H.disc_recovery(X, xd)["best"]
    rng2 = H.default_rng(SEED + 12345)
    xd_s = rng2.permutation(xd)
    shuf = H.disc_recovery(X, xd_s)["best"]
    ok = (true > 0.90) and (shuf < 0.10)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} lam={lam}: disc(true)={true:.3f}  disc(shuffled)={shuf:.3f}")
    return ok


print("shuffled-parity null (disc must be ~0 on scrambled labels):")
legs = [("ice-phase", H.gen_s2_phase, H.S2["hp_noise"]),
        ("handedness", H.gen_s2_hand, H.S2["hh_noise"])]
for nm, gen, noise in legs:
    for lam in (0.0, 2.0):
        if not one(nm, gen, noise, lam):
            fail += 1

print(f"\n{'ALL PASS — disc is real, no label leak' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
