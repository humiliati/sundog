#!/usr/bin/env python
"""Frozen test for H10 ADV-HIDE-D (scripts/shadow_adv_hide_d.py).

Runs BEFORE the verdict run: pins the apparatus contract and the prereg ABORT gate --
gen_hide structure/determinism (v2 module untouched, import-only), the antisymmetric
exact-cancellation identity behind A3, GradReverse semantics, GRL micro-arm byte-reproducibility,
probe-protocol sanity and determinism. Does NOT pin verdict outcomes (banked in the banking
commit after the run). Run: python scripts/test_shadow_adv_hide_d.py
"""
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import torch                                    # noqa: E402
import shadow_adv_hide_d as ah                  # noqa: E402
import shadow_pooled_synthetic_v2 as v2         # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H10 ADV-HIDE-D frozen test (apparatus contract + ABORT gate):\n")

# ---- gen_hide: structure, determinism, v2 passthrough ---- #
u1, c1, d1 = ah.gen_hide(40, 1.0, 999)
u2, c2, d2 = ah.gen_hide(40, 1.0, 999)
uv, cv, dv = v2.gen(40, 1.0, 999)
check("gen_hide shape = (n, K, F+P)", u1.shape == (40, ah.K, ah.F_HIDE))
check("gen_hide deterministic (byte-identical rerun)",
      np.array_equal(u1, u2) and np.array_equal(c1, c2) and np.array_equal(d1, d2))
check("gen_hide first F dims byte-equal v2.gen (same seed; v2 module untouched)",
      np.array_equal(u1[:, :, :v2.F], uv) and np.array_equal(c1, cv) and np.array_equal(d1, dv))
check("positional block = POS slot constants (sample-independent)",
      np.array_equal(u1[:, :, v2.F:], np.broadcast_to(ah.POS[None], (40, ah.K, ah.P))))

# ---- the A3 antisymmetric identity: pooled s*zhat is EXACTLY d-free ---- #
check("alternating signs sum to zero (K even)", float(ah.S_SIGNS.sum()) == 0.0)
rng = np.random.default_rng(0)
eta = rng.standard_normal((ah.K, ah.D)).astype(np.float32)
g = eta @ v2.A_DISC                                                  # (K,) noise projection
pool_plus = float((ah.S_SIGNS * (+1.0 + g)).mean())
pool_minus = float((ah.S_SIGNS * (-1.0 + g)).mean())
check("antisymmetric cancellation: pooled s*zhat identical for d=+1 vs d=-1 (same noise)",
      abs(pool_plus - pool_minus) < 1e-7, f"|diff| = {abs(pool_plus - pool_minus):.2e}")

# ---- GradReverse semantics ---- #
x = torch.ones(4, requires_grad=True)
ah.GradReverse.apply(x, 2.5).sum().backward()
check("GradReverse: identity forward, -lam-scaled backward",
      bool(torch.allclose(x.grad, torch.full((4,), -2.5))))

# ---- ABORT gate: GRL micro-arm byte-reproducibility ---- #
ut, ct, dt = ah.gen_hide(400, ah.TRAIN_LAM, 31337)
phi_a, fit_a = ah.train_arm("A1_1.0", "hide", ut, ct, dt, ladv=1.0, epochs=3)
phi_b, fit_b = ah.train_arm("A1_1.0", "hide", ut, ct, dt, ladv=1.0, epochs=3)
ra, rb = ah.pool_feats(phi_a, ut), ah.pool_feats(phi_b, ut)
check("ABORT gate: GRL training byte-reproducible under fixed seeds (pooled reps identical)",
      np.array_equal(ra, rb), f"max|diff| = {float(np.max(np.abs(ra - rb))):.2e}")
check("retention-aux arm also byte-reproducible",
      np.array_equal(
          ah.pool_feats(ah.train_arm("A2_1.0", "hide", ut, ct, dt, ladv=1.0, retention=True,
                                     epochs=2)[0], ut),
          ah.pool_feats(ah.train_arm("A2_1.0", "hide", ut, ct, dt, ladv=1.0, retention=True,
                                     epochs=2)[0], ut)))

# ---- probe-protocol sanity (tiny synthetic) ---- #
rng = np.random.default_rng(1)
n = 300
ys = rng.choice([-1.0, 1.0], n)
X_sep = np.concatenate([ys[:, None] * 2.0 + rng.standard_normal((n, 1)) * 0.1,
                        rng.standard_normal((n, 3))], axis=1)
X_noise = rng.standard_normal((n, 4))
sep = ah.d_probe_best(X_sep, ys)["best"]
nse = ah.d_probe_best(X_noise, ys)["best"]
check("d_probe_best finds a separable signal (>= 0.95)", sep >= 0.95, f"best={sep:.3f}")
check("d_probe_best reads ~chance on pure noise (in [0.38, 0.62])", 0.38 <= nse <= 0.62,
      f"best={nse:.3f}")
uh = rng.standard_normal((50, ah.K, 4)).astype(np.float32)
r1 = ah.retention_acc(uh, ys[:50])
r2 = ah.retention_acc(uh, ys[:50])
check("retention probe deterministic across calls", r1 == r2, f"{r1:.4f} vs {r2:.4f}")

print(f"\n{'ALL PASS -- apparatus pinned; ready for smoke, then the verdict run.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
