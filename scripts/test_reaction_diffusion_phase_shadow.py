#!/usr/bin/env python
"""Frozen test for H8 v2 / substrate S3phi (scripts/reaction_diffusion_phase_shadow.py) — locks the
HONEST SCOPED picture (a bounded-positive, NOT a clean RD crossover):
  1. The crossover gates pass AS GATED: the phase resists (charFun), the chirality determines, the fixed-
     lambda G-KINV passes, C-NONTRIVIAL holds.
  2. The phase resist is GENUINE charFun, not v1's LLN: at lambda=2.0 v2's phase stays ~0 across K while a
     finite-MEAN control RECOVERS with K (the decisive discriminator).
  3. The reaction-diffusion dynamics are NOT load-bearing: a bare analytic vortex (zero PDE) reproduces the
     gates identically (=> this is ~S1, not an RD substrate extension -- the honest scope-down).
  4. C-CHANNEL FAILS (the honest caveat): the phase block leaks chirality (channels are NOT orthogonal).
Run: python scripts/test_reaction_diffusion_phase_shadow.py  (~1-2 min)
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "scripts")
import reaction_diffusion_phase_shadow as r   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H8 v2 / S3phi — phase-resist + chirality-determine: the SCOPED bounded-positive (frozen test):\n")

CAL = dict(grid=56, steps=1000, K=8, n=96, m_angles=90, r_bases=5, noise=0.15, jit_rad=2.5)
SEED = 999
_real_cgl = r.cgl_spiral_batch
r.CFG.update(CAL); r.build_library(56, SEED)

# 1. crossover gates (as gated)
s3c = r.sweep(r.gen_phase_c, r.CFG["n"], SEED, r.CFG["noise"], "phase")
s3d = r.sweep(r.gen_phase_d, r.CFG["n"], SEED, r.CFG["noise"], "chir")
kd = r.k_dependence(SEED, r.CFG["noise"])
ctrl = r.nontrivial_control(SEED, r.CFG["noise"])
check("phase RESISTS (cont0 high, washes to ~0)", s3c["cont"][0] >= 0.70 and s3c["cont"][-1] <= 0.15,
      f"cont0={s3c['cont'][0]:.3f} cont(2.0)={s3c['cont'][-1]:.3f}")
check("chirality DETERMINES (disc high, never washes)", min(s3d["disc"]) >= 0.90 and s3d["lambda_star_d"] is None,
      f"minDisc={min(s3d['disc']):.3f}")
check("G-KINV (fixed-lambda): phase destroyed for all K", kd["is_charfun_resist"], f"cont_vs_K={kd['cont_vs_K']}")
check("C-NONTRIVIAL: handedness-blind probe at chance", ctrl["nontrivial_ok"],
      f"blind={ctrl['blind_block_disc']:.3f} chir={ctrl['chir_block_disc']:.3f}")

# 2. GENUINE charFun (vs LLN): v2 phase dead at lambda=2 across K; finite-mean control recovers
def finite_mean(n, lam, rng, noise, K):
    xc = rng.uniform(0, 1.5, n); f = np.empty((n, 8))
    for i in range(n):
        m = (xc[i] + lam * rng.standard_normal(K)).mean()
        f[i] = [m, m ** 2, np.sin(m), np.cos(m), np.tanh(m), 0.5 * m, abs(m), m + 0.1]
    return f + rng.normal(0, noise, f.shape), xc

v2_hi = []; lln_hi = []
for K in [8, 512]:
    r.CFG["K"] = K
    v2_hi.append(round(r.cont_recovery(*r.gen_phase_c(96, 2.0, np.random.default_rng(900 + K), 0.15)[:2])["best"], 3))
    X, yc = finite_mean(96, 2.0, np.random.default_rng(900 + K), 0.15, K)
    lln_hi.append(round(r.cont_recovery(X, yc)["best"], 3))
r.CFG["K"] = CAL["K"]
check("phase is GENUINE charFun: v2 stays ~0 at lambda=2 across K, finite-mean control RECOVERS",
      max(v2_hi) <= 0.15 and lln_hi[-1] - lln_hi[0] >= 0.3,
      f"v2 K8/512={v2_hi}  finite-mean K8/512={lln_hi}")

# 3. RD dynamics NOT load-bearing: bare analytic vortex (no PDE) reproduces the gates
def bare_vortex(B, grid, steps, chirality, rng):
    g = np.linspace(-1, 1, grid); X, Y = np.meshgrid(g, g)
    rr = np.sqrt(X ** 2 + Y ** 2); th = np.arctan2(Y, X)
    u = np.empty((B, grid, grid), np.float32); w = np.empty((B, grid, grid), np.float32)
    for bi in range(B):
        A = np.tanh(rr / r.CFG["r0"]) * np.exp(1j * (chirality * th + 0.4 * rng.standard_normal()))
        u[bi] = A.real; w[bi] = A.imag
    return u, w

r.cgl_spiral_batch = bare_vortex; r.CFG.update(CAL); r.build_library(56, SEED)
bv_c0 = r.cont_recovery(*r.gen_phase_c(96, 0.0, np.random.default_rng(7), 0.15)[:2])["best"]
bv_kd = r.k_dependence(SEED, 0.15)
r.cgl_spiral_batch = _real_cgl
check("RD NOT load-bearing: bare vortex (zero PDE) reproduces the gates (=> ~S1, not an RD extension)",
      bv_c0 >= 0.70 and bv_kd["is_charfun_resist"], f"bare-vortex cont0={bv_c0:.3f} G-KINV={bv_kd['cont_vs_K']}")

# 4. C-CHANNEL FAILS (honest caveat): the phase block leaks chirality (channels NOT orthogonal)
r.CFG.update(CAL); r.build_library(56, SEED)
Xd, _, yd = r.gen_phase_d(96, 0.0, np.random.default_rng(91), 0.15)
p0, p1 = r.BLOCKS["phase"]
phase_leak = r.disc_recovery(Xd[:, p0:p1], yd)["best"]
check("C-CHANNEL FAILS (disclosed): phase block LEAKS chirality (NOT orthogonal channels)", phase_leak > 0.60,
      f"phase-block->chirality disc={phase_leak:.3f} (> 0.60 = leak)")

# 5. determinism
Xa = r.gen_phase_c(12, 0.3, np.random.default_rng(7), 0.15)[0]
Xb = r.gen_phase_c(12, 0.3, np.random.default_rng(7), 0.15)[0]
check("deterministic (same seed -> identical features)", np.allclose(Xa, Xb, atol=1e-6),
      f"max|diff|={np.abs(Xa - Xb).max():.1e}")

print(f"\n{'ALL PASS — v2 SCOPED bounded-positive locked: the charFun mechanism is genuine (phase charFun-resists vs a finite-mean LLN control; chirality first-order-invisibly determines), fixing v1 errors -- BUT the RD dynamics are NOT load-bearing (bare vortex identical => ~S1, not an RD substrate extension), and C-CHANNEL fails (phase leaks chirality). Honest scope, not a clean RD crossover.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
