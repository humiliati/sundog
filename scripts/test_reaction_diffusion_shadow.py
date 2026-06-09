#!/usr/bin/env python
"""Frozen test for H8 / substrate S3 (scripts/reaction_diffusion_shadow.py). Locks the determine/resist
CROSSOVER on a fast config (small grid/steps/library so it runs in ~1-2 min): the Gray-Scott morphology
classes separate topologically (spots >> stripes components); the ensemble shadow DETERMINES the class
(disc high across lambda) and RESISTS the wavelength (cont recoverable at lambda=0, washes with an
in-grid half-life); the shadow is deterministic. Run: python scripts/test_reaction_diffusion_shadow.py
"""
import sys
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, "scripts")
import reaction_diffusion_shadow as r   # noqa: E402

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


print("H8 / S3 — reaction-diffusion determine/resist shadow (fast frozen test):\n")

# fast config (NOT the frozen primary; just enough to exhibit the crossover qualitatively)
r.CFG.update(grid=48, steps=1500, K=4, n=56, noise=0.04, m_kin=12, r_ic=4)
LAMS = [0.0, 0.05, 0.2, 0.5, 1.0, 2.0]
r.LAMBDAS = LAMS
SEED = 999
r.build_libraries(48, 1500, SEED, ["spots", "stripes"])

cc_spots = r.LIB["spots"]["feats"][:, -1].mean()
cc_stripes = r.LIB["stripes"]["feats"][:, -1].mean()
check("morphology classes separate topologically (spots components >> stripes)", cc_spots > 4 * cc_stripes,
      f"<cc> spots={cc_spots:.1f} stripes={cc_stripes:.1f}")

s3c = r.sweep(r.gen_s3c, r.CFG["n"], SEED, r.CFG["noise"], "S3c")
s3d = r.sweep(r.gen_s3d, r.CFG["n"], SEED, r.CFG["noise"], "S3d")

check("S3c preflight: wavelength recoverable at lambda=0 (injectivity)", s3c["cont"][0] >= 0.45,
      f"cont0={s3c['cont'][0]:.3f}")
check("S3c RESISTS: wavelength washes at high lambda (continuous-resist)", s3c["cont"][-1] <= 0.15,
      f"cont(lam=2)={s3c['cont'][-1]:.3f}")
check("S3c resist has an in-grid half-life (ensemble decoherence, not flat)", s3c["lambda_star_c"] is not None,
      f"lambda*_c={s3c['lambda_star_c']}")
check("S3d DETERMINES: class recovered at lambda=0", s3d["disc"][0] >= 0.90, f"disc0={s3d['disc'][0]:.3f}")
check("S3d DETERMINES: class survives all lambda (min disc high, never washes)",
      min(s3d["disc"]) >= 0.90 and s3d["lambda_star_d"] is None,
      f"minDisc={min(s3d['disc']):.3f} lambda*_d={s3d['lambda_star_d']}")
check("CROSSOVER: continuous washes while discrete survives (the headline)",
      s3c["cont"][-1] <= 0.15 and min(s3d["disc"]) >= 0.90, "")

# determinism: same seed -> byte-identical shadow features
Xa = r.gen_s3c(20, 0.3, np.random.default_rng(7), 0.04)[0]
Xb = r.gen_s3c(20, 0.3, np.random.default_rng(7), 0.04)[0]
check("shadow is deterministic (same seed -> identical features)", np.allclose(Xa, Xb, atol=1e-6),
      f"max|diff|={np.abs(Xa - Xb).max():.2e}")

print(f"\n{'ALL PASS — the ensemble shadow of a Gray-Scott body DETERMINES the morphology class and RESISTS the wavelength: the Shadow-Invertibility split reaches a nonlinear reaction-diffusion substrate (S3).' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
