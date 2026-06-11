#!/usr/bin/env python
"""Frozen test for H3-PC (scripts/h3_probe_ceiling.py) — pins the APPARATUS at reduced size.

Per the frozen prereg (docs/atlas/H3_PROBE_CEILING_PREREG.md section 6): the reduced pool/test are the
FIRST 4000/2000 rows of the FULL-SIZE draws gen(20000,2.0,51235) / gen(10000,2.0,61235) — a strict
subset, NOT a fresh smaller gen() call (which would be a different realization). Pins:
  * C0 continuity (raw mean washes c on the reduced pool),
  * bisection convergence (both injection levels calibrate within tolerance),
  * per-member liveness machinery runs and yields booleans,
  * byte-identical battery readouts + (battery-only) verdict letter across an IN-PROCESS rerun.
Equality with the FULL run's verdict letter is NOT asserted (prereg: pinning it would force apparatus
tuning — the regen-drift failure mode). The MI leg is full-size-only (subsample 5000 > reduced pool)
and is NOT exercised here; the test verdict letter is the battery-only letter by construction.
Run: python scripts/test_h3_probe_ceiling.py     (~10-20 min CPU, deterministic)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import sys
import warnings

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")

import numpy as np

sys.path.insert(0, "scripts")
import shadow_pooled_synthetic_v2 as v2            # noqa: E402
import h3_probe_ceiling as pc                      # noqa: E402
from sklearn.linear_model import Ridge             # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H3-PC frozen test — apparatus pins at reduced size (pool 4000 / test 2000, subset rows):\n")

# bodies (deterministic retrain; determinism itself is pinned by test_shadow_pooled_synthetic_v2)
u_tr, c_tr, d_tr = v2.gen(v2.N_TRAIN, v2.TRAIN_LAM, v2.SEED + 1)
clf, _ = v2.train_body("clf_d", u_tr, c_tr, d_tr)
reg, _ = v2.train_body("reg_c", u_tr, c_tr, d_tr)

# reduced draws = FIRST rows of the FULL-SIZE draws (binding subset rule)
u_pool_f, c_pool_f, _ = v2.gen(pc.N_POOL, pc.LAM, pc.POOL_SEED)
u_test_f, c_test_f, _ = v2.gen(pc.N_TEST, pc.LAM, pc.TEST_SEED)
u_pool, c_pool = u_pool_f[:4000], c_pool_f[:4000]
u_test, c_test = u_test_f[:2000], c_test_f[:2000]
z_pool, z_test = v2.phi_pool(clf, u_pool), v2.phi_pool(clf, u_test)
zr_pool = v2.phi_pool(reg, u_pool)

# C0 continuity + positive control at reduced size
c0 = pc.cv_r2(Ridge(alpha=1.0), u_pool.mean(axis=1), c_pool)
check("C0 continuity: raw mean washes c on the reduced pool (<= 0.05)", c0 <= 0.05, f"raw_c={c0:+.4f}")
pos = pc.cv_r2(Ridge(alpha=1.0), zr_pool, c_pool)
check("positive control: reg_c reps carry c at reduced size (ridge >= 0.40)", pos >= 0.40,
      f"reg_c ridge={pos:+.4f}")

# injection calibration converges at both levels (reduced size)
vdir = np.random.default_rng(pc.VDIR_SEED).standard_normal(v2.H)
vdir /= np.linalg.norm(vdir)
g_pool = (c_pool - c_pool.mean()) / c_pool.std()


def inject(alpha):
    return z_pool + alpha * g_pool[:, None] * vdir[None, :]


def bisect(target):
    lo, hi = 0.0, 3.0
    for _ in range(28):
        mid = 0.5 * (lo + hi)
        if pc.cv_r2(Ridge(alpha=1.0), inject(mid), c_pool) < target:
            lo = mid
        else:
            hi = mid
    a = 0.5 * (lo + hi)
    return a, pc.cv_r2(Ridge(alpha=1.0), inject(a), c_pool)


alphas = {}
for lvl in pc.INJ_LEVELS:
    a, r = bisect(lvl)
    alphas[lvl] = a
    check(f"calibration {lvl} converges (|achieved-target| <= 0.01)", abs(r - lvl) <= 0.01,
          f"alpha={a:.5f} ridge-CV={r:+.4f}")

# per-member liveness machinery runs and yields booleans (values not pinned — size-dependent)
z_inj = {lvl: inject(alphas[lvl]) for lvl in pc.INJ_LEVELS}
fams = pc.battery()
live = {}
for fam_name, fam in fams.items():
    if fam_name == "P1_ridge":
        continue
    live[fam_name] = {}
    for lvl in pc.INJ_LEVELS:
        hit = False
        for cfg_name, mk in fam:
            if pc.cv_r2(mk(), z_inj[lvl], c_pool) >= 0.05:
                hit = True
                break
        live[fam_name][lvl] = hit
check("liveness machinery yields a boolean per member per level",
      all(isinstance(b, bool) for d in live.values() for b in d.values()),
      str({f: d for f, d in live.items()}))

# battery readouts + battery-only verdict letter, byte-identical across an in-process rerun
def run_battery():
    rd = {}
    for fam_name, fam in fams.items():
        cfg, mk, cv = pc.select(fam, z_pool, c_pool)
        sp = pc.split_r2(mk(), z_pool, c_pool, z_test, c_test)
        rd[fam_name] = (cfg, cv, sp)
    a_hit = any(cv >= 0.30 and sp >= 0.24 for _, cv, sp in rd.values())
    blind = {f for f, d in live.items() if not any(d.values())}
    b_hit = all(cv <= 0.05 and sp <= 0.05 for f, (_, cv, sp) in rd.items() if f not in blind)
    letter = "a" if a_hit else ("b" if b_hit else "c")
    return rd, letter


rd1, letter1 = run_battery()
rd2, letter2 = run_battery()
for f in rd1:
    print(f"    {f:13s} [{rd1[f][0]:12s}]  pool-CV={rd1[f][1]:+.4f}  split={rd1[f][2]:+.4f}")
check("battery readouts byte-identical across in-process rerun",
      all(rd1[f][1] == rd2[f][1] and rd1[f][2] == rd2[f][2] for f in rd1))
check("battery-only verdict letter stable across rerun and in {a,b,c}",
      letter1 == letter2 and letter1 in ("a", "b", "c"), f"letter={letter1}")

print(f"\n{'ALL PASS — H3-PC apparatus pinned at reduced size (full-run verdict NOT asserted here).' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
