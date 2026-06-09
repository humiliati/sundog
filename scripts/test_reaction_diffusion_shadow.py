#!/usr/bin/env python
"""Frozen test for H8 / substrate S3 (scripts/reaction_diffusion_shadow.py) — locks the NULL.

The v1 RD determine/resist crossover does NOT hold, and this test locks WHY (the falsifiable, decisive
signatures the adversarial review surfaced and the lab verified):
  1. The discrete 'determine' (spots vs stripes) is a TRIVIAL single-threshold separator (component count
     ranges are non-overlapping), so the lossiness axis never actually tests it.
  2. The continuous 'resist' (wavelength via diffusion-scale s) is NOT a charFun/Debye-Waller resist: its
     recovery half-life GROWS with the ensemble size K (the law-of-large-numbers signature of a
     finite-MEAN, DETERMINE-type latent that CONCENTRATES under averaging) instead of being K-invariant
     (the genuine charFun-resist signature, which S0/S1/S2 satisfy). The charFun law PREDICTS this: a
     finite centered mean => determines, not resists.
  3. Therefore there is no genuine crossover (NULL). A real RD resist needs a PHASE (charFun-decaying)
     latent. Run: python scripts/test_reaction_diffusion_shadow.py  (~1-2 min)
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


print("H8 / S3 — reaction-diffusion determine/resist: the NULL (frozen test):\n")

r.CFG.update(grid=48, steps=1200, K=8, n=72, noise=0.04, m_kin=16, r_ic=4)
r.LAMBDAS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
SEED = 999
r.build_libraries(48, 1200, SEED, ["spots", "stripes"])

# 1. the 'determine' pair is a TRIVIAL single-threshold separator (non-overlapping component counts) —
#    a real result would need overlapping classes; here averaging trivially preserves the gap.
cc_spots = r.LIB["spots"]["feats"][:, -1]
cc_stripes = r.LIB["stripes"]["feats"][:, -1]
check("determine is TRIVIAL: component-count ranges are non-overlapping (single threshold separates)",
      cc_spots.min() > cc_stripes.max(),
      f"spots cc in [{cc_spots.min():.0f},{cc_spots.max():.0f}] vs stripes [{cc_stripes.min():.0f},{cc_stripes.max():.0f}]")

# 2. THE DECISIVE TEST: the wavelength 'resist' half-life GROWS with K (LLN / determine-type),
#    it is NOT K-invariant (charFun). This is the falsification of the v1 crossover.
kdep = r.k_dependence(SEED, r.CFG["noise"], ks=(1, 8, 64))
hl = {k: kdep[k]["half_life"] for k in ["1", "8", "64"]}
print(f"    half-life vs K: {hl}")
check("RESIST is finite-K LLN slack: half-life GROWS with K (NOT a charFun resist)",
      kdep["half_life_grows_with_K"] is True and kdep["is_charfun_resist"] is False,
      f"grows_with_K={kdep['half_life_grows_with_K']} charFun_resist={kdep['is_charfun_resist']}")
check("ensemble RECOVERS the wavelength as K grows (bigger ensemble -> larger half-life: concentration)",
      (kdep['64']['half_life'] or 99) > (kdep['1']['half_life'] or 0),
      f"lam*_c: K=1 -> {kdep['1']['half_life']}, K=64 -> {kdep['64']['half_life']}")

# 3. no genuine crossover (NULL), and the charFun law's prediction (finite-mean -> determine) is borne out.
check("NULL: no genuine Shadow-law determine/resist crossover on RD (wavelength is determine-type)",
      kdep["is_charfun_resist"] is False, "the charFun law predicted the finite-mean wavelength concentrates")

# determinism: same seed -> identical shadow features
Xa = r.gen_s3c(16, 0.3, np.random.default_rng(7), 0.04)[0]
Xb = r.gen_s3c(16, 0.3, np.random.default_rng(7), 0.04)[0]
check("shadow is deterministic (same seed -> identical features)", np.allclose(Xa, Xb, atol=1e-6),
      f"max|diff|={np.abs(Xa - Xb).max():.2e}")

print(f"\n{'ALL PASS — the NULL is locked: the v1 RD crossover does not hold. The wavelength is a finite-mean DETERMINE-type latent (half-life grows with K = LLN concentration), exactly as the charFun law predicts; the discrete determine is a trivial single-threshold separator. A genuine RD resist needs a PHASE (charFun-decaying) latent.' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
