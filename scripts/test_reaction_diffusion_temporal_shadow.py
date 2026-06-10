#!/usr/bin/env python
"""Frozen test for H8 v3 / substrate S3tau (scripts/reaction_diffusion_temporal_shadow.py) — locks the NULL.

v3 does NOT achieve a load-bearing charFun resist. This test locks the two killers the adversarial review
+ verification established, plus the determine-type G-KINV failure:
  1. The cross-test (the load-bearing discriminator) is VACUOUS: a REAL-RD field with byte-identical
     dynamics but COLUMN-SHUFFLED features 'fails' it (cross R2 ~ 0) while tau is fully recoverable within
     that shuffled distribution (own R2 high) -- so the cross-test measures feature layout, not mechanism.
  2. The v2 SO(2) trap is NOT escaped: tau is recoverable from rotation-only frames and rotation->real
     transfers (rotation encodes tau). tau is read off spiral orientation.
  3. The resist is determine-type: G-KINV cont(lam=2) RISES with K (LLN), not a charFun resist.
Run: python scripts/test_reaction_diffusion_temporal_shadow.py  (~1-2 min)
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "scripts")
import reaction_diffusion_temporal_shadow as r   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H8 v3 / S3tau — load-bearing temporal-phase resist: the NULL (frozen test):\n")

r.CFG.update(grid=56, settle=900, nframes=50, stride=5, n=100, r_bases=8, down=8, r_amp=12,
             t_lo=0.0, t_hi=12.0, jit=5.0, noise=0.10)
SEED = 999
r.build_library(56, SEED)

# the surface phenomenon: tau recoverable at K=8 and washes (this part is real, but it is the SO(2) channel)
cont0 = r.cont_recovery(*r.gen_c(100, 0.0, np.random.default_rng(7), 0.10))
contmax = r.cont_recovery(*r.gen_c(100, 2.0, np.random.default_rng(2007), 0.10))
check("surface: tau recoverable at K=8 and washes at fixed K (a symmetry-coordinate decoherence)",
      cont0 >= 0.70 and contmax <= 0.15, f"cont0={cont0:.3f} cont(2.0,K=8)={contmax:.3f}")

# 1. THE CROSS-TEST IS VACUOUS (the load-bearing discriminator is invalid)
Xr, yr = r.gen_c(100, 0.0, np.random.default_rng(1), 0.10)
Xr2, yr2 = r.gen_c(100, 0.0, np.random.default_rng(99), 0.10)
perm = np.random.default_rng(5).permutation(r.NFEAT)
Xshuf = Xr2[:, perm]                                          # REAL dynamics, columns permuted
cross_real = r.cross_test(Xr, yr, Xr2, yr2)
cross_shuf = r.cross_test(Xr, yr, Xshuf, yr2)
own_shuf = r.cont_recovery(Xshuf, yr2)
check("cross-test is VACUOUS: real-RD with SHUFFLED columns 'fails' it (cross~0) yet tau IS there (own high)",
      cross_shuf <= 0.20 and own_shuf >= 0.60 and cross_real >= 0.60,
      f"real->real={cross_real:.3f}  real->shuffled={cross_shuf:.3f}  own(shuffled)={own_shuf:.3f}")

# 2. THE v2 SO(2) TRAP IS NOT ESCAPED: rotation encodes tau
Xrot, yrot = r.gen_c(100, 0.0, np.random.default_rng(3), 0.10, mode="rotation")
rot_own = r.cont_recovery(Xrot, yrot)
rot_to_real = r.cross_test(Xrot, yrot, Xr, yr)
check("v2 SO(2) trap NOT escaped: tau recoverable from rotation-only frames AND rotation->real transfers",
      rot_own >= 0.60 and rot_to_real >= 0.50,
      f"rotation own cont0={rot_own:.3f}  rotation->real cross={rot_to_real:.3f}")

# 3. DETERMINE-TYPE: G-KINV cont(lam=2) RISES with K (LLN), not a charFun resist
kc = {}
for K in [8, 512]:
    r.CFG["K"] = K
    kc[K] = r.cont_recovery(*r.gen_c(100, 2.0, np.random.default_rng(900 + K), 0.10))
r.CFG["K"] = 8
check("resist is DETERMINE-type: cont(lam=2) RISES with K (LLN concentration), G-KINV fails",
      kc[512] - kc[8] >= 0.10, f"cont(lam=2): K=8 -> {kc[8]:.3f}, K=512 -> {kc[512]:.3f}")

print(f"\n{'ALL PASS — v3 NULL locked: the load-bearing cross-test is vacuous (real-RD shuffle fails it), the temporal phase is read off spiral orientation = the v2 SO(2) symmetry coordinate (rotation encodes tau), and the resist is determine-type (G-KINV fails). v3 CONFIRMS the obstacle (no load-bearing charFun-resist on the CGL-spiral family), it does not escape it.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
