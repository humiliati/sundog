#!/usr/bin/env python
"""Hypothesis 2 (the charFun-spectrum determine/resist law) — the EMPIRICAL falsifiable leg.

Swap the AVERAGING POPULATION in the S0 lossiness-crossover (gen_s0's per-subunit spread xi) from
Gaussian to {uniform, Cauchy, lattice +-1}, rerun the FROZEN cont/disc sweep through the unchanged
recovery apparatus, and test the pre-registered claim that resistance is governed by the population's
CHARACTERISTIC FUNCTION phi_mu(s)=E[e^{i s xi}], NOT its variance. The exact mechanism:

    mean_i cos(2*pi*(xc + lam*xi)*t) = cos(2*pi*xc*t) * Re[phi_mu(2*pi*lam*t)]     (Debye-Waller)

  * AC (absolutely-continuous) populations  -- gaussian exp(-s^2/2), uniform sinc, CAUCHY exp(-|s|)
    -- have phi_mu(s) -> 0 as |s| -> inf (Riemann-Lebesgue), so the size-bearing fringe WASHES
    (cont -> 0). Cauchy resists DESPITE infinite variance: the surprise that isolates charFun-decay
    from variance as the governing quantity.
  * The LATTICE (xi = +-1) has phi_mu(s) = cos(s), which does NOT decay -- it recurs to +-1 at
    resonant s -- so the fringe is only MODULATED, never washed: cont stays elevated (the separating
    case). At any fixed probe t != 0 the lattice envelope cos(2*pi*lam*t) stays O(1) as lam -> inf,
    whereas every AC envelope -> 0.

DETERMINE half (separate condition): a shared label survives averaging iff the population has a FINITE
CENTERED MEAN. gaussian/uniform/lattice (E[xi]=0, finite) determine; Cauchy's sample mean is itself
Cauchy (averaging does NOT concentrate it -- no mean) so it CANNOT determine. Resist and determine are
governed by two DIFFERENT spectral conditions (charFun decay vs finite mean); Cauchy separates them.

PRE-REGISTERED kill criteria:
  KILLED if the lattice cont collapses to ~0 at all lam like Gaussian (=> charFun does NOT govern; the
          mechanism would be raw variance or something else), OR if an AC finite-variance population
          (uniform) fails to resist (=> contradicts Riemann-Lebesgue).

NOT public-eligible. Attribution: Debye 1913 / Waller 1923 (thermal damping factor); Riemann-Lebesgue
lemma; Lukacs, "Characteristic Functions". Reuses the frozen pvnp_phase5_lossiness_crossover apparatus.
"""
import math
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pvnp_phase5_lossiness_crossover as h            # noqa: E402  (frozen S0 apparatus)

POPS = ["gaussian", "uniform", "cauchy", "lattice"]
KIND = {"gaussian": "AC", "uniform": "AC", "cauchy": "AC", "lattice": "lattice"}
T_REF = h.S0["t0_f"]                                   # the off-centre fringe peak (dominant size probe)


def draw_pop(rng, shape, pop):
    """Unit-scale per-subunit spread draws. gaussian/uniform/lattice have unit VARIANCE; cauchy has
    unit SCALE (gamma=1) -- its variance is infinite, which is the entire point of including it."""
    if pop == "gaussian":
        return rng.standard_normal(shape)
    if pop == "uniform":
        return rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), shape)   # Var = 1
    if pop == "cauchy":
        return rng.standard_cauchy(shape)                            # scale gamma = 1, no mean/var
    if pop == "lattice":
        return rng.choice([-1.0, 1.0], shape)                        # Var = 1, phi = cos(s)
    raise ValueError(pop)


def charfun_re(pop, s):
    """Analytic Re[phi_mu(s)] = E[cos(s*xi)] for the unit-scale populations -- the theoretical
    Debye-Waller envelope the empirical cont curve must track."""
    s = np.asarray(s, float)
    if pop == "gaussian":
        return np.exp(-s ** 2 / 2.0)
    if pop == "uniform":
        a = math.sqrt(3.0)
        return np.where(np.abs(s) < 1e-12, 1.0, np.sin(a * s) / (a * s))   # sinc
    if pop == "cauchy":
        return np.exp(-np.abs(s))
    if pop == "lattice":
        return np.cos(s)
    raise ValueError(pop)


def gen_s0_pop(n, lam, rng, noise, pop):
    """gen_s0 (frozen) with ONLY the averaging population swapped. pop='gaussian' reproduces the frozen
    gen_s0 code path exactly (the control)."""
    c = h.S0
    t = np.linspace(-1, 1, c["T"])
    env_g = np.exp(-t ** 2 / (2 * c["w"] ** 2))
    env_f = np.exp(-(np.abs(t) - c["t0_f"]) ** 2 / (2 * c["w_f"] ** 2))
    bump = np.exp(-(t - c["t0"]) ** 2 / (2 * c["w_b"] ** 2))
    xc = rng.uniform(c["xc_lo"], c["xc_hi"], n)            # continuous label (fringe freq / size)
    xd = rng.choice([-1.0, 1.0], n)                       # discrete label (parity sign)
    xi = draw_pop(rng, (n, h.K), pop)                    # <-- the only change vs gen_s0
    xci = xc[:, None] + lam * xi
    fringe = np.cos(2 * np.pi * xci[:, :, None] * t[None, None, :]).mean(1)
    parity = (xd[:, None] * np.sin(2 * np.pi * c["f_p"] * t)[None, :])
    sig = (c["D"] * bump[None, :]
           + c["A"] * fringe * env_f[None, :]
           + c["C"] * parity * env_g[None, :])
    sig = sig + rng.normal(0, noise, sig.shape)
    return sig, xc, xd


def sweep_pop(pop, n=600, seed=20260608, noise=None, lams=None):
    """Frozen cont/disc sweep over `lams` (default h.LAMBDAS) for one population (per-lambda
    deterministic draw, matching the harness convention)."""
    noise = h.NOISE if noise is None else noise
    lams = h.LAMBDAS if lams is None else lams
    cont, disc = [], []
    for lam in lams:
        rng = np.random.default_rng(seed + int(round(lam * 1000)) + 7)
        X, yc, yd = gen_s0_pop(n, lam, rng, noise, pop)
        cont.append(round(h.cont_recovery(X, yc)["best"], 4))
        disc.append(round(h.disc_recovery(X, yd)["best"], 4))
    return cont, disc


DET_KS = [16, 64, 256, 1024]


def determination_probe(pop, ks=None, reps=8000, lam=1.0, seed=4242):
    """The DETERMINE half = does the shared-label average CONCENTRATE (law of large numbers)? Average
    a shared label d=1 plus per-unit noise lam*xi over K units, over `reps` ensembles, for growing K.
    Returns the list of median|avg-d| at each K. A finite centered mean => spread ~ lam*sigma/sqrt(K)
    SHRINKS with K (determined); Cauchy's K-average is ITSELF Cauchy (a stable law, scale unchanged by
    averaging) => the spread stays FLAT no matter how many units you average (never determined)."""
    ks = DET_KS if ks is None else ks
    rng = np.random.default_rng(seed)
    out = []
    for K_ in ks:
        xi = draw_pop(rng, (reps, K_), pop)
        out.append(float(np.median(np.abs(lam * xi.mean(1)))))   # |avg - d|, d cancels
    return out


def main():
    print("=" * 78)
    print("Hypothesis 2 -- charFun-spectrum determine/resist law (EMPIRICAL leg)")
    print("  swap the S0 averaging population xi; resistance must track phi_mu(2*pi*lam*t), not Var.")
    print("=" * 78)
    print("\nPRE-REGISTERED:  AC {gaussian,uniform,cauchy}: phi->0 => cont WASHES (resist, lam*_c in grid)")
    print("                 lattice (phi=cos, recurs to 1): cont SURVIVES (no half-life in grid)")
    print(f"                 KILL if lattice washes like gaussian, or uniform fails to resist.\n")

    s_ref = 2 * np.pi * np.array(h.LAMBDAS) * T_REF       # charFun argument at the fringe-peak probe
    rows = {}
    for pop in POPS:
        cont, disc = sweep_pop(pop)
        lc = h.half_life(cont, h.LAMBDAS)
        env = np.abs(charfun_re(pop, s_ref))             # analytic |Re phi| envelope (for corroboration)
        rows[pop] = dict(cont=cont, disc=disc, lc=lc, env=env)
        print(f"-- {pop:<8} ({KIND[pop]})  phi_mu(s) = "
              f"{ {'gaussian':'exp(-s^2/2)','uniform':'sinc(sqrt3 s)','cauchy':'exp(-|s|)','lattice':'cos(s)'}[pop] }")
        print(f"   {'lam':>6} " + " ".join(f"{l:>5}" for l in h.LAMBDAS))
        print(f"   {'cont':>6} " + " ".join(f"{v:5.2f}" for v in cont))
        print(f"   {'|phi|':>6} " + " ".join(f"{v:5.2f}" for v in env) + "   (analytic envelope @ t=t0_f)")
        print(f"   {'disc':>6} " + " ".join(f"{v:5.2f}" for v in disc))
        cmax = max(cont[1:])                              # ignore lam=0 base
        print(f"   half-life lam*_c = {lc}   cont(lam=2) = {cont[-1]:.3f}   max cont(lam>0) = {cmax:.3f}\n")

    # ---- verdicts (pre-registered gates) ---- #
    print("=" * 78)
    print("VERDICTS")
    print("=" * 78)
    ac_resist = []
    for pop in ["gaussian", "uniform", "cauchy"]:
        c = rows[pop]["cont"]
        resisted = (c[-1] <= h.CONT_MAX_MAX) and (rows[pop]["lc"] is not None)
        ac_resist.append(resisted)
        print(f"  [{'PASS' if resisted else 'FAIL'}] {pop:<8} (AC) RESISTS: cont(lam=2)={c[-1]:.3f} "
              f"<= {h.CONT_MAX_MAX}, half-life={rows[pop]['lc']}")
    lc_lat = rows["lattice"]["cont"]
    lat_survives = (lc_lat[-1] > 0.30) and (rows["lattice"]["lc"] is None)
    print(f"  [{'PASS' if lat_survives else 'FAIL'}] lattice    SURVIVES: cont(lam=2)={lc_lat[-1]:.3f} "
          f"> 0.30, half-life={rows['lattice']['lc']} (censored = never washes)")

    print("\n  DETERMINE half (shared label survives iff the average CONCENTRATES -- finite centered mean):")
    print(f"    {'pop':<8} median|avg-d| at K = " + " ".join(f"{k:>7}" for k in DET_KS) + "   ratio K0/K3")
    for pop in POPS:
        curve = determination_probe(pop)
        ratio = curve[0] / max(curve[-1], 1e-9)          # concentration factor over the K-sweep
        ok = ratio > 3.0                                 # ~sqrt(1024/16)=8 if LLN holds; ~1 for Cauchy
        note = ("determined (LLN: average concentrates)" if ok
                else "BREAK (no finite mean -- average never concentrates)")
        print(f"    [{'det ' if ok else 'BREAK'}] {pop:<8} " + " ".join(f"{v:7.3f}" for v in curve)
              + f"   {ratio:6.1f}x  {note}")

    killed = (not all(ac_resist)) or (not lat_survives)
    print("\n" + "=" * 78)
    if killed:
        print("RESULT: HYPOTHESIS KILLED (an AC pop failed to resist, or lattice washed like gaussian).")
    else:
        print("RESULT: BOUNDED-POSITIVE. Resistance tracks charFun decay, NOT variance: the three AC")
        print("  populations (incl. infinite-variance CAUCHY) wash; the lattice (phi=cos) survives.")
        print("  Determine is the separate finite-mean condition; Cauchy resists but breaks it.")
    print("=" * 78)
    return 1 if killed else 0


if __name__ == "__main__":
    sys.exit(main())
