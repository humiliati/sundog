#!/usr/bin/env python3
"""Isotrophy v0.10a: one-sided Jonckheere-Terpstra trend test.

Locked by docs/isotrophy/kfacet/kfacet_v10a_jt_trend_form.md. Consumes the FROZEN
v0.9a receipt (no new variational compute, no vf re-derivation). Primary p-value is
a deterministic EXACT fixed-margin enumeration of the multivariate-hypergeometric
null; the 10,000-shuffle permutation is a secondary sanity check; normal-approx
(from the exact null moments) is context only.

Hypothesis (one-sided, increasing): S-fraction increases across the ordered zones
positional-dominant < mixed < velocity-heavy.
"""
import argparse
import csv
import json
import math
import os

ZONE_ORDER = ["positional-dominant", "mixed", "velocity-heavy"]
CUTPOINTS = (0.25, 0.50)
EXPECTED = {  # frozen v0.9a manifest: zone -> (N, S, U)
    "positional-dominant": (19, 2, 17),
    "mixed": (165, 56, 109),
    "velocity-heavy": (66, 29, 37),
}
EXPECTED_TOTAL = (250, 87, 163)
ALPHA = 0.01
PERM_SEED = 20260523
PERM_N = 10000


def zone_of(vf):
    if vf < CUTPOINTS[0]:
        return "positional-dominant"
    if vf < CUTPOINTS[1]:
        return "mixed"
    return "velocity-heavy"


def jt_statistic(s, u):
    """Jonckheere-Terpstra J from per-zone (S,U) counts (binary response S=1>U=0).

    U_{ab} = u_a*s_b + 0.5*(s_a*s_b + u_a*u_b) for zones a<b; J = sum over a<b.
    """
    k = len(s)
    j = 0.0
    for a in range(k):
        for b in range(a + 1, k):
            j += u[a] * s[b] + 0.5 * (s[a] * s[b] + u[a] * u[b])
    return j


def load_rows(in_dir):
    path = os.path.join(in_dir, "per_row_table.csv")
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return rows, path


def frozen_input_crosscheck(rows):
    """Recompute zones from velocity_fraction + assert exact counts vs v0.9a."""
    per_zone = {z: {"N": 0, "S": 0, "U": 0} for z in ZONE_ORDER}
    zone_recompute_mismatch = 0
    for r in rows:
        vf = float(r["velocity_fraction"])
        recomputed = zone_of(vf)
        stored = r["zone"].strip()
        if recomputed != stored:
            zone_recompute_mismatch += 1
        z = recomputed
        per_zone[z]["N"] += 1
        if r["stability"].strip() == "S":
            per_zone[z]["S"] += 1
        else:
            per_zone[z]["U"] += 1
    counts_match = all(
        (per_zone[z]["N"], per_zone[z]["S"], per_zone[z]["U"]) == EXPECTED[z]
        for z in ZONE_ORDER
    )
    tot = (
        sum(per_zone[z]["N"] for z in ZONE_ORDER),
        sum(per_zone[z]["S"] for z in ZONE_ORDER),
        sum(per_zone[z]["U"] for z in ZONE_ORDER),
    )
    return per_zone, counts_match and tot == EXPECTED_TOTAL, zone_recompute_mismatch


def exact_enumeration(j_obs):
    n = [EXPECTED[z][0] for z in ZONE_ORDER]   # 19, 165, 66
    total_s = EXPECTED_TOTAL[1]                 # 87
    denom = math.comb(EXPECTED_TOTAL[0], total_s)  # C(250,87)
    p_mass = 0.0
    p_tail = 0.0
    mean = 0.0
    second = 0.0
    count = 0
    for s0 in range(0, n[0] + 1):
        for s2 in range(0, n[2] + 1):
            s1 = total_s - s0 - s2
            if s1 < 0 or s1 > n[1]:
                continue
            prob = (
                math.comb(n[0], s0) * math.comb(n[1], s1) * math.comb(n[2], s2)
            ) / denom
            s = (s0, s1, s2)
            u = (n[0] - s0, n[1] - s1, n[2] - s2)
            j = jt_statistic(s, u)
            p_mass += prob
            mean += j * prob
            second += j * j * prob
            if j >= j_obs:
                p_tail += prob
            count += 1
    var = max(second - mean * mean, 0.0)
    return {
        "feasible_table_count": count,
        "probability_mass_total": p_mass,
        "J_null_mean": mean,
        "J_null_sd": math.sqrt(var),
        "p_value_one_sided": p_tail,
    }


def permutation_sanity(rows, j_obs):
    try:
        import numpy as np
    except ImportError:
        return {"seed": PERM_SEED, "n": PERM_N, "status": "numpy_unavailable",
                "p_value_one_sided": None}
    zones = np.array([ZONE_ORDER.index(zone_of(float(r["velocity_fraction"]))) for r in rows])
    is_s = np.array([1 if r["stability"].strip() == "S" else 0 for r in rows])
    rng = np.random.default_rng(PERM_SEED)
    ge = 0
    j_sum = 0.0
    for _ in range(PERM_N):
        perm = rng.permutation(is_s)
        s = [int(perm[zones == z].sum()) for z in range(3)]
        n = [int((zones == z).sum()) for z in range(3)]
        u = [n[i] - s[i] for i in range(3)]
        j = jt_statistic(s, u)
        j_sum += j
        if j >= j_obs:
            ge += 1
    return {
        "seed": PERM_SEED,
        "n": PERM_N,
        "J_null_mean": j_sum / PERM_N,
        "p_value_one_sided": (1 + ge) / (1 + PERM_N),
    }


def normal_approx(j_obs, mean, sd):
    if sd <= 0:
        return {"E_J": mean, "sd_J": sd, "z": None, "p_value_one_sided": None}
    z = (j_obs - mean) / sd
    p = 0.5 * math.erfc(z / math.sqrt(2.0))  # one-sided upper tail
    return {"E_J": mean, "sd_J": sd, "z": z, "p_value_one_sided": p,
            "note": "moments from the exact fixed-margin enumeration; context only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default="results/isotrophy/k-facet-v09a-signed-vf-three-zone")
    ap.add_argument("--out", dest="out_dir",
                    default="results/isotrophy/k-facet-v10a-jt-trend")
    args = ap.parse_args()

    rows, in_path = load_rows(args.in_dir)
    per_zone, matches, zone_mismatch = frozen_input_crosscheck(rows)

    s = [per_zone[z]["S"] for z in ZONE_ORDER]
    u = [per_zone[z]["U"] for z in ZONE_ORDER]
    n = [per_zone[z]["N"] for z in ZONE_ORDER]
    j_obs = jt_statistic(s, u)

    crosscheck = {
        "per_zone_N": {z: per_zone[z]["N"] for z in ZONE_ORDER},
        "per_zone_S": {z: per_zone[z]["S"] for z in ZONE_ORDER},
        "per_zone_U": {z: per_zone[z]["U"] for z in ZONE_ORDER},
        "per_zone_S_fraction": {z: (per_zone[z]["S"] / per_zone[z]["N"] if per_zone[z]["N"] else None)
                                for z in ZONE_ORDER},
        "zones_recomputed_from_velocity_fraction": zone_mismatch == 0,
        "zone_recompute_mismatch_count": zone_mismatch,
        "matches_v09a_manifest": matches,
    }

    if not matches or zone_mismatch != 0:
        verdict = "ABORT_frozen_input_crosscheck_failed"
        result = {"mode": "v0.10a-jt-trend", "verdict": verdict,
                  "frozen_input_crosscheck": crosscheck}
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
        print(f"[isotrophy v0.10a] ABORT — frozen-input cross-check failed "
              f"(counts_match={matches}, zone_mismatch={zone_mismatch})")
        raise SystemExit(1)

    exact = exact_enumeration(j_obs)
    perm = permutation_sanity(rows, j_obs)
    napprox = normal_approx(j_obs, exact["J_null_mean"], exact["J_null_sd"])

    verdict = ("jt_trend_monotone_registered"
               if exact["p_value_one_sided"] < ALPHA
               else "jt_trend_not_significant")

    result = {
        "schema": "sundog.isotrophy.v0.10a-jt-trend.v1",
        "mode": "v0.10a-jt-trend",
        "form_lock": "docs/isotrophy/kfacet/kfacet_v10a_jt_trend_form.md",
        "input_v09a_per_row_table": in_path.replace("\\", "/"),
        "frozen_input_crosscheck": crosscheck,
        "zone_order": ZONE_ORDER,
        "response_coding": {"S": 1, "U": 0},
        "J_observed": j_obs,
        "exact_enumeration": exact,
        "permutation_sanity": perm,
        "normal_approx": napprox,
        "alpha": ALPHA,
        "verdict": verdict,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    with open(os.path.join(args.out_dir, "jt_trend_receipt.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print("[isotrophy v0.10a] frozen-input cross-check PASSED "
          f"(zones {n}, S {s}, U {u})")
    print(f"[isotrophy v0.10a] J_observed = {j_obs}")
    print(f"[isotrophy v0.10a] exact enumeration: {exact['feasible_table_count']} tables, "
          f"mass={exact['probability_mass_total']:.10f}, "
          f"p_exact(one-sided) = {exact['p_value_one_sided']:.3e}")
    if perm.get("p_value_one_sided") is not None:
        print(f"[isotrophy v0.10a] permutation sanity (n={perm['n']}): "
              f"p_mc = {perm['p_value_one_sided']:.3e}")
    if napprox.get("p_value_one_sided") is not None:
        print(f"[isotrophy v0.10a] normal-approx (context): z = {napprox['z']:.3f}, "
              f"p = {napprox['p_value_one_sided']:.3e}")
    print(f"[isotrophy v0.10a] VERDICT: {verdict}  (alpha={ALPHA}, one-sided)")


if __name__ == "__main__":
    main()
