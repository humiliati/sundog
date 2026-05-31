#!/usr/bin/env python3
"""
Riemann Q1 control — does a genuine sine-kernel (GUE/CUE) spectrum reproduce the
Probe 05 reversibility statistic D within the registered floor?

Answers External Review Packet Q1 in-house: "does the GUE / sine-kernel baseline
already predict that D = -0.0064 inside floor 0.0424 is just a standard null
rather than any structural-zero edge?"

Method: run the IDENTICAL S2 reversibility statistic used by
`scripts/riemann-probe05-reversibility.mjs` (D = mean over consecutive pairs of
sign(s_i - s_{i+1}), tie tol 1e-8; floor tau_ind = 3/sqrt(m); same circular
moving-block bootstrap) on synthetic spectra and compare the observed zeta D to
the resulting D-distributions:

  (1) CUE eigenphases — the exact bulk sine-kernel process (the Montgomery-Odlyzko
      law: zeta zero local statistics == CUE/GUE bulk). Haar unitary via Mezzadri
      (2007) QR construction; eigenphases unfolded to unit mean spacing by the
      THEORETICAL density n/(2*pi), mirroring the probe's theoretical-density
      unfolding. Independent blocks concatenated; cross-block pairs excluded.
  (2) i.i.d. GUE Wigner-surmise spacings — a correlation-free contrast (exact 2x2
      GUE gaps, mean-normalized). Shows the reversibility null is predicted even
      by the model with NO level-repulsion correlations, i.e. it carries zero
      pair-correlation information let alone structural-zero information.

This is NOT a structural-zero probe and NOT evidence for or against RH.
"""
import argparse
import json
import os
import time
import numpy as np

# --- Observed Probe 05 result (registered) ----------------------------------
# results/riemann/probe05-nonlinear-zero-statistics/reversibility_summary.json
D_ZETA = -0.006402561024409764
M_ZETA = 4998
TAU_IND_ZETA = 0.04243489469900033      # = 3 / sqrt(4998)
TAU_BOOT_ZETA = 0.020409163665466606

# --- Statistic params copied verbatim from the probe ------------------------
TIE_TOL = 1e-8
BLOCK_LENGTH = 64
B_BOOT = 10000
BOOT_QUANTILE = 0.9975


def signs_within(spacings):
    """Within-sequence consecutive-pair signs, identical tie rule to the probe."""
    delta = spacings[:-1] - spacings[1:]
    return np.where(delta > TIE_TOL, 1.0, np.where(delta < -TIE_TOL, -1.0, 0.0))


def cue_block_unfolded(n, rng):
    """One CUE block: eigenphases of a Haar-random unitary (Mezzadri 2007),
    unfolded to unit mean spacing by the theoretical density n/(2*pi)."""
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    ph = np.diagonal(r)
    ph = ph / np.abs(ph)
    q = q * ph                                  # Haar-distributed unitary
    ang = np.sort(np.angle(np.linalg.eigvals(q)))
    gaps = np.diff(ang)                          # n-1 within-block spacings (no wrap)
    return gaps * n / (2.0 * np.pi)              # unfold by theoretical density


def cue_realization_D(n, blocks, rng):
    """One realization: `blocks` independent CUE blocks; D over within-block pairs
    only (cross-block pairs excluded)."""
    sgn = [signs_within(cue_block_unfolded(n, rng)) for _ in range(blocks)]
    signs = np.concatenate(sgn)
    return signs, float(signs.mean()), int(signs.size)


def wigner_gue_spacings(count, rng):
    """`count` i.i.d. spacings ~ GUE Wigner surmise, via exact 2x2 GUE gaps."""
    a = rng.standard_normal(count)
    d = rng.standard_normal(count)
    b = rng.standard_normal(count) / np.sqrt(2.0)
    c = rng.standard_normal(count) / np.sqrt(2.0)
    gap = np.sqrt((a - d) ** 2 + 4.0 * (b ** 2 + c ** 2))
    return gap / gap.mean()


def tau_boot(signs, rng):
    """Circular moving-block bootstrap floor; same block / B / quantile as the probe."""
    m = signs.size
    centered = signs - signs.mean()
    nblocks = int(np.ceil(m / BLOCK_LENGTH))
    offs = np.arange(BLOCK_LENGTH)
    means = np.empty(B_BOOT)
    for bi in range(B_BOOT):
        starts = rng.integers(0, m, size=nblocks)
        idx = ((starts[:, None] + offs[None, :]).ravel() % m)[:m]
        means[bi] = centered[idx].mean()
    return float(np.quantile(np.abs(means), BOOT_QUANTILE))


def dist_stats(D, label):
    D = np.asarray(D)
    mean, std = float(D.mean()), float(D.std(ddof=1))
    z = (D_ZETA - mean) / std if std > 0 else float("nan")
    pct = float((D <= D_ZETA).mean())                 # percentile of observed zeta D
    return {
        "label": label,
        "realizations": int(D.size),
        "mean_D": mean,
        "std_D": std,
        "min_D": float(D.min()),
        "max_D": float(D.max()),
        "q025": float(np.quantile(D, 0.025)),
        "q500": float(np.quantile(D, 0.5)),
        "q975": float(np.quantile(D, 0.975)),
        "frac_within_tau_ind_zeta": float((np.abs(D) <= TAU_IND_ZETA).mean()),
        "zeta_z_score": float(z),
        "zeta_percentile": pct,
        "zeta_within_central_95pct": bool(np.quantile(D, 0.025) <= D_ZETA <= np.quantile(D, 0.975)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500, help="CUE block dimension")
    ap.add_argument("--blocks", type=int, default=10, help="CUE blocks per realization")
    ap.add_argument("--realizations", type=int, default=300, help="CUE realizations")
    ap.add_argument("--iid-realizations", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260530)
    ap.add_argument("--out", default="results/riemann/probe05-q1-sinekernel-control")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    t0 = time.time()

    # (1) CUE / sine-kernel
    cue_D, cue_m = [], None
    first_signs = None
    for r in range(args.realizations):
        signs, D, m = cue_realization_D(args.n, args.blocks, rng)
        cue_D.append(D)
        cue_m = m
        if first_signs is None:
            first_signs = signs
        if (r + 1) % 50 == 0:
            print(f"  CUE {r+1}/{args.realizations}  ({time.time()-t0:.0f}s)", flush=True)
    cue_D = np.array(cue_D)

    # floor spot-check: probe's own bootstrap machinery on a genuine sine-kernel run
    tau_boot_cue = tau_boot(first_signs, np.random.default_rng(args.seed + 1))

    # (2) i.i.d. Wigner-GUE surmise contrast (m matched to the zeta run)
    iid_D = np.empty(args.iid_realizations)
    for r in range(args.iid_realizations):
        sp = wigner_gue_spacings(M_ZETA + 1, rng)
        iid_D[r] = signs_within(sp).mean()

    cue_stats = dist_stats(cue_D, "CUE_sine_kernel")
    cue_stats["m_per_realization"] = cue_m
    cue_stats["tau_boot_spotcheck"] = tau_boot_cue
    iid_stats = dist_stats(iid_D, "iid_wigner_gue")
    iid_stats["m_per_realization"] = M_ZETA

    # Verdict: the observed zeta D is a typical sine-kernel draw, and the genuine
    # sine-kernel spectra sit within the registered floor.
    cue_ok = (
        abs(cue_stats["mean_D"]) < 3 * cue_stats["std_D"] / np.sqrt(cue_D.size)  # mean ~ 0
        and abs(cue_stats["zeta_z_score"]) < 2.0                                  # zeta typical
        and cue_stats["frac_within_tau_ind_zeta"] > 0.95                          # floor bounds sine-kernel
    )
    verdict = (
        "CONTROL CONFIRMS STANDARD NULL (R-NL-NEG-A): the observed zeta D is a "
        "typical draw from the sine-kernel D-distribution and genuine sine-kernel "
        "spectra reproduce the bounded reversibility null within the registered floor."
        if cue_ok else
        "CONTROL DOES NOT CONFIRM: the observed zeta D is atypical of the sine-kernel "
        "baseline; investigate before declaring a standard null."
    )

    out = {
        "probe": "riemann_q1_sinekernel_control",
        "purpose": "External Review Packet Q1 — is the Probe 05 reversibility null a standard GUE/sine-kernel result?",
        "observed_zeta": {
            "D": D_ZETA, "m": M_ZETA, "tau_ind": TAU_IND_ZETA, "tau_boot": TAU_BOOT_ZETA,
            "abs_D_le_tau_ind": abs(D_ZETA) <= TAU_IND_ZETA,
        },
        "params": vars(args),
        "cue_sine_kernel": cue_stats,
        "iid_wigner_gue": iid_stats,
        "floor_consistency": {
            "tau_boot_zeta": TAU_BOOT_ZETA,
            "tau_boot_cue_spotcheck": tau_boot_cue,
            "note": "Probe's own block-bootstrap floor, recomputed on a genuine sine-kernel run.",
        },
        "verdict_pass": bool(cue_ok),
        "verdict": verdict,
        "runtime_sec": round(time.time() - t0, 1),
    }

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "q1_sinekernel_summary.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({
        "verdict_pass": out["verdict_pass"],
        "zeta_D": D_ZETA,
        "cue_mean_D": cue_stats["mean_D"], "cue_std_D": cue_stats["std_D"],
        "zeta_z_score": cue_stats["zeta_z_score"], "zeta_percentile": cue_stats["zeta_percentile"],
        "cue_frac_within_tau_ind": cue_stats["frac_within_tau_ind_zeta"],
        "tau_boot_cue_vs_zeta": [tau_boot_cue, TAU_BOOT_ZETA],
        "iid_mean_D": iid_stats["mean_D"], "iid_std_D": iid_stats["std_D"],
        "runtime_sec": out["runtime_sec"],
    }, indent=2))


if __name__ == "__main__":
    main()
