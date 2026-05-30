#!/usr/bin/env python3
"""Isotrophy v0.11: m3-conditional vf rank test.

Locked by docs/isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md.
Consumes the frozen v0.9a per-row table plus v0.10a/v0.10b manifests. No new
orbit integration, variational integration, or feature derivation is performed.

Primary gate: within fixed m3 strata, does the frozen three-zone vf score rank
stable rows above unstable rows? Binding p-value is an exact stratified null
from per-stratum multivariate-hypergeometric enumeration and convolution. A
within-stratum permutation sidecar guards the hand-rolled enumeration kernel.
"""
import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict

ZONE_ORDER = ["positional-dominant", "mixed", "velocity-heavy"]
CUTPOINTS = (0.25, 0.50)
EXPECTED_ZONE = {
    "positional-dominant": (19, 2, 17),
    "mixed": (165, 56, 109),
    "velocity-heavy": (66, 29, 37),
}
EXPECTED_TOTAL = (250, 87, 163)
EXPECTED_PRIMARY = {
    "0.4": (55, 35, 20),
    "0.5": (30, 15, 15),
    "0.6": (22, 6, 16),
    "0.7": (18, 4, 14),
    "0.8": (18, 6, 12),
    "0.9": (21, 6, 15),
    "1.0": (35, 7, 28),
    "1.1": (23, 1, 22),
    "1.2": (7, 2, 5),
}
EXPECTED_REPORT_ONE_CLASS = {"1.7": (5, 0, 5)}
EXPECTED_REPORT_TINY = {
    "1.3": (4, 2, 2),
    "1.4": (4, 0, 4),
    "1.5": (4, 1, 3),
    "1.6": (2, 2, 0),
    "1.9": (2, 0, 2),
}
EXPECTED_PRIMARY_TOTAL = (229, 82, 147)
V10B_POOLED_AUC = 0.4125
ALPHA = 0.01
PERM_SEED = 20260523
PERM_N = 10000


def zone_of(vf):
    if vf < CUTPOINTS[0]:
        return 0
    if vf < CUTPOINTS[1]:
        return 1
    return 2


def m3_key(row):
    return row.get("m3_key", "").strip() or f"{float(row['m3']):.1f}"


def label_s(row):
    return row["stability"].strip() == "S"


def j2_from_counts(s_by_zone, u_by_zone):
    """Return doubled J so tied half-pairs remain integer-valued."""
    j2 = 0
    lower_u = 0
    for z in range(len(ZONE_ORDER)):
        j2 += int(s_by_zone[z]) * (2 * lower_u + int(u_by_zone[z]))
        lower_u += int(u_by_zone[z])
    return j2


def auc_from_j2(j2, d_pairs):
    return j2 / (2.0 * d_pairs) if d_pairs else None


def stratum_counts(rows):
    n_by_zone = [0, 0, 0]
    s_by_zone = [0, 0, 0]
    u_by_zone = [0, 0, 0]
    for row in rows:
        z = zone_of(float(row["velocity_fraction"]))
        n_by_zone[z] += 1
        if label_s(row):
            s_by_zone[z] += 1
        else:
            u_by_zone[z] += 1
    return n_by_zone, s_by_zone, u_by_zone


def exact_stratum_distribution(n_by_zone, s_total):
    """Exact null for one stratum as {J2: probability} plus receipt metadata."""
    n_total = sum(n_by_zone)
    denom = math.comb(n_total, s_total)
    dist = defaultdict(float)
    allocation_count = 0

    for s0 in range(0, n_by_zone[0] + 1):
        for s1 in range(0, n_by_zone[1] + 1):
            s2 = s_total - s0 - s1
            if s2 < 0 or s2 > n_by_zone[2]:
                continue
            s_by_zone = [s0, s1, s2]
            u_by_zone = [n_by_zone[z] - s_by_zone[z] for z in range(3)]
            prob = 1.0
            for z in range(3):
                prob *= math.comb(n_by_zone[z], s_by_zone[z])
            prob /= denom
            dist[j2_from_counts(s_by_zone, u_by_zone)] += prob
            allocation_count += 1

    mass = sum(dist.values())
    return {
        "distribution": dict(dist),
        "feasible_allocation_count": allocation_count,
        "support_size": len(dist),
        "probability_mass": mass,
    }


def convolve_distributions(distributions):
    combined = {0: 1.0}
    for dist in distributions:
        next_dist = defaultdict(float)
        for a, pa in combined.items():
            for b, pb in dist.items():
                next_dist[a + b] += pa * pb
        combined = dict(next_dist)
    return combined


def permutation_sanity(primary_by_m3, observed_j2):
    rng = random.Random(PERM_SEED)
    strata = []
    for key in sorted(primary_by_m3, key=float):
        rows = primary_by_m3[key]
        zones = [zone_of(float(row["velocity_fraction"])) for row in rows]
        s_total = sum(1 for row in rows if label_s(row))
        strata.append((zones, s_total))

    ge = 0
    j2_sum = 0.0
    for _ in range(PERM_N):
        total_j2 = 0
        for zones, s_total in strata:
            labels = [1] * s_total + [0] * (len(zones) - s_total)
            rng.shuffle(labels)
            s_by_zone = [0, 0, 0]
            u_by_zone = [0, 0, 0]
            for z, is_s in zip(zones, labels):
                if is_s:
                    s_by_zone[z] += 1
                else:
                    u_by_zone[z] += 1
            total_j2 += j2_from_counts(s_by_zone, u_by_zone)
        j2_sum += total_j2
        if total_j2 >= observed_j2:
            ge += 1

    p = (1 + ge) / (1 + PERM_N)
    se = math.sqrt(max(p * (1.0 - p), 0.0) / PERM_N)
    return {
        "seed": PERM_SEED,
        "n": PERM_N,
        "p_perm_sanity": p,
        "J2_null_mean": j2_sum / PERM_N,
        "standard_error": se,
    }


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_json(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def abort(out_dir, verdict, manifest):
    manifest["verdict"] = verdict
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"[v0.11] {verdict}")
    raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir",
                    default="results/isotrophy/k-facet-v09a-signed-vf-three-zone")
    ap.add_argument("--v10a", dest="v10a_dir",
                    default="results/isotrophy/k-facet-v10a-jt-trend")
    ap.add_argument("--v10b", dest="v10b_dir",
                    default="results/isotrophy/k-facet-v10b-monotone-vf-heldout")
    ap.add_argument("--out", dest="out_dir",
                    default="results/isotrophy/k-facet-v11-m3-conditional-vf-rank")
    args = ap.parse_args()

    per_row_path = os.path.join(args.in_dir, "per_row_table.csv")
    v10a_manifest_path = os.path.join(args.v10a_dir, "manifest.json")
    v10b_manifest_path = os.path.join(args.v10b_dir, "manifest.json")

    with open(per_row_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    v10a = load_json(v10a_manifest_path)
    v10b = load_json(v10b_manifest_path)

    zone_counts = {z: [0, 0, 0] for z in ZONE_ORDER}
    zone_mismatch = 0
    by_m3 = defaultdict(list)
    for row in rows:
        zi = zone_of(float(row["velocity_fraction"]))
        if ZONE_ORDER[zi] != row["zone"].strip():
            zone_mismatch += 1
        zone_name = ZONE_ORDER[zi]
        zone_counts[zone_name][0] += 1
        if label_s(row):
            zone_counts[zone_name][1] += 1
        else:
            zone_counts[zone_name][2] += 1
        by_m3[m3_key(row)].append(row)

    total = (
        len(rows),
        sum(zone_counts[z][1] for z in ZONE_ORDER),
        sum(zone_counts[z][2] for z in ZONE_ORDER),
    )
    counts_ok = all(tuple(zone_counts[z]) == EXPECTED_ZONE[z] for z in ZONE_ORDER)
    v10a_ok = (
        v10a.get("verdict") == "jt_trend_monotone_registered"
        and v10a.get("exact_enumeration", {}).get("p_value_one_sided", 1.0) < ALPHA
    )
    v10b_ok = (
        v10b.get("verdict") == "monotone_vf_predictor_fails_heldout"
        and v10b.get("primary_metric", {}).get("auc_observed", 1.0) <= 0.5
    )

    crosscheck = {
        "row_count": len(rows),
        "stability_totals": {"S": total[1], "U": total[2]},
        "zone_counts": {z: zone_counts[z] for z in ZONE_ORDER},
        "matches_v09a_manifest": counts_ok and total == EXPECTED_TOTAL,
        "zone_recompute_mismatch_count": zone_mismatch,
        "v10a_verdict": v10a.get("verdict"),
        "v10a_p_under_0p01": v10a_ok,
        "v10b_verdict": v10b.get("verdict"),
        "v10b_auc_observed": v10b.get("primary_metric", {}).get("auc_observed"),
        "v10b_auc_condition": v10b_ok,
    }

    manifest = {
        "schema": "sundog.isotrophy.v0.11-m3-conditional-vf-rank.v1",
        "mode": "v0.11-m3-conditional-vf-rank",
        "form_lock": "docs/isotrophy/kfacet/kfacet_v11_m3_conditional_vf_rank_form.md",
        "input_v09a_per_row_table": per_row_path.replace("\\", "/"),
        "input_v10a_manifest": v10a_manifest_path.replace("\\", "/"),
        "input_v10b_manifest": v10b_manifest_path.replace("\\", "/"),
        "frozen_input_crosscheck": crosscheck,
    }

    if not (crosscheck["matches_v09a_manifest"] and zone_mismatch == 0 and v10a_ok and v10b_ok):
        abort(args.out_dir, "ABORT_frozen_input_crosscheck_failed", manifest)

    primary_by_m3 = {}
    report_one_class = {}
    report_tiny = {}
    for key in sorted(by_m3, key=float):
        group = by_m3[key]
        n = len(group)
        s = sum(1 for row in group if label_s(row))
        u = n - s
        if n >= 5 and s >= 1 and u >= 1:
            primary_by_m3[key] = group
        elif n >= 5:
            report_one_class[key] = group
        else:
            report_tiny[key] = group

    primary_summary = {
        key: (len(group), sum(1 for row in group if label_s(row)),
              sum(1 for row in group if not label_s(row)))
        for key, group in primary_by_m3.items()
    }
    one_class_summary = {
        key: (len(group), sum(1 for row in group if label_s(row)),
              sum(1 for row in group if not label_s(row)))
        for key, group in report_one_class.items()
    }
    tiny_summary = {
        key: (len(group), sum(1 for row in group if label_s(row)),
              sum(1 for row in group if not label_s(row)))
        for key, group in report_tiny.items()
    }
    primary_total = (
        sum(v[0] for v in primary_summary.values()),
        sum(v[1] for v in primary_summary.values()),
        sum(v[2] for v in primary_summary.values()),
    )
    if (
        primary_summary != EXPECTED_PRIMARY
        or one_class_summary != EXPECTED_REPORT_ONE_CLASS
        or tiny_summary != EXPECTED_REPORT_TINY
        or primary_total != EXPECTED_PRIMARY_TOTAL
    ):
        manifest.update({
            "primary_strata_found": primary_summary,
            "report_one_class_found": one_class_summary,
            "report_tiny_found": tiny_summary,
            "primary_total_found": primary_total,
        })
        abort(args.out_dir, "ABORT_domain_mismatch", manifest)

    per_stratum = []
    exact_summaries = []
    distributions = []
    observed_j2 = 0
    observed_d = 0
    probability_mass_ok = True
    for key in sorted(primary_by_m3, key=float):
        group = primary_by_m3[key]
        n_by_zone, s_by_zone, u_by_zone = stratum_counts(group)
        s_total = sum(s_by_zone)
        u_total = sum(u_by_zone)
        d_pairs = s_total * u_total
        j2 = j2_from_counts(s_by_zone, u_by_zone)
        exact = exact_stratum_distribution(n_by_zone, s_total)
        probability_mass_ok = probability_mass_ok and abs(exact["probability_mass"] - 1.0) <= 1e-10
        distributions.append(exact["distribution"])
        observed_j2 += j2
        observed_d += d_pairs
        per_stratum.append({
            "m3": key,
            "N": len(group),
            "S": s_total,
            "U": u_total,
            "zone0_N": n_by_zone[0],
            "zone1_N": n_by_zone[1],
            "zone2_N": n_by_zone[2],
            "zone0_S": s_by_zone[0],
            "zone1_S": s_by_zone[1],
            "zone2_S": s_by_zone[2],
            "zone0_U": u_by_zone[0],
            "zone1_U": u_by_zone[1],
            "zone2_U": u_by_zone[2],
            "J2_m": j2,
            "J_m": j2 / 2.0,
            "D_m": d_pairs,
            "AUC_m": auc_from_j2(j2, d_pairs),
        })
        exact_summaries.append({
            "m3": key,
            "feasible_allocation_count": exact["feasible_allocation_count"],
            "support_size": exact["support_size"],
            "probability_mass": exact["probability_mass"],
        })

    combined = convolve_distributions(distributions)
    combined_mass = sum(combined.values())
    exact_p = sum(prob for j2, prob in combined.items() if j2 >= observed_j2)
    probability_mass_ok = probability_mass_ok and abs(combined_mass - 1.0) <= 1e-9
    if not probability_mass_ok:
        manifest.update({
            "observed": {"J2_cond": observed_j2, "D_cond": observed_d,
                         "AUC_cond": auc_from_j2(observed_j2, observed_d)},
            "exact_null": {
                "per_stratum_support": {row["m3"]: row["support_size"] for row in exact_summaries},
                "combined_support": len(combined),
                "per_stratum_probability_mass": {
                    row["m3"]: row["probability_mass"] for row in exact_summaries
                },
                "combined_probability_mass": combined_mass,
                "p_value_one_sided": exact_p,
            },
        })
        abort(args.out_dir, "ABORT_exact_null_probability_mass_failed", manifest)

    perm = permutation_sanity(primary_by_m3, observed_j2)
    tolerance = 5.0 * perm["standard_error"] + (1.0 / PERM_N)
    perm["consistent_with_exact"] = abs(perm["p_perm_sanity"] - exact_p) <= tolerance
    perm["consistency_tolerance"] = tolerance
    if not perm["consistent_with_exact"]:
        manifest.update({
            "observed": {"J2_cond": observed_j2, "D_cond": observed_d,
                         "AUC_cond": auc_from_j2(observed_j2, observed_d)},
            "exact_null": {"p_value_one_sided": exact_p},
            "permutation_sanity": perm,
        })
        abort(args.out_dir, "ABORT_exact_vs_permutation_sanity_divergence", manifest)

    auc_cond = auc_from_j2(observed_j2, observed_d)
    if auc_cond <= 0.5:
        verdict = "m3_conditional_vf_rank_fails"
    elif exact_p <= ALPHA:
        verdict = "m3_conditional_vf_rank_passes"
    else:
        verdict = "m3_conditional_vf_rank_fails"

    for row in per_stratum:
        row["pair_weight"] = row["D_m"] / observed_d

    report_rows = []
    for bucket, groups in (("one-class", report_one_class), ("tiny", report_tiny)):
        for key in sorted(groups, key=float):
            group = groups[key]
            n_by_zone, s_by_zone, u_by_zone = stratum_counts(group)
            s_total = sum(s_by_zone)
            u_total = sum(u_by_zone)
            d_pairs = s_total * u_total
            j2 = j2_from_counts(s_by_zone, u_by_zone) if d_pairs else 0
            report_rows.append({
                "bucket": bucket,
                "m3": key,
                "N": len(group),
                "S": s_total,
                "U": u_total,
                "zone0_N": n_by_zone[0],
                "zone1_N": n_by_zone[1],
                "zone2_N": n_by_zone[2],
                "J2_m": j2,
                "D_m": d_pairs,
                "AUC_m": auc_from_j2(j2, d_pairs),
            })

    branch_alignment = defaultdict(lambda: {"N": 0, "S": 0, "U": 0})
    for row in rows:
        key = row.get("branch_label", "").strip() or "(blank)"
        branch_alignment[key]["N"] += 1
        if label_s(row):
            branch_alignment[key]["S"] += 1
        else:
            branch_alignment[key]["U"] += 1

    exact_null = {
        "per_stratum_support": {row["m3"]: row["support_size"] for row in exact_summaries},
        "combined_support": len(combined),
        "per_stratum_probability_mass": {
            row["m3"]: row["probability_mass"] for row in exact_summaries
        },
        "combined_probability_mass": combined_mass,
        "p_value_one_sided": exact_p,
        "alpha": ALPHA,
    }
    manifest.update({
        "primary_strata": [
            {"m3": key, "N": vals[0], "S": vals[1], "U": vals[2]}
            for key, vals in primary_summary.items()
        ],
        "report_only_strata": {
            "one_class_N_ge_5": [
                {"m3": key, "N": vals[0], "S": vals[1], "U": vals[2]}
                for key, vals in one_class_summary.items()
            ],
            "tiny_N_lt_5": [
                {"m3": key, "N": vals[0], "S": vals[1], "U": vals[2]}
                for key, vals in tiny_summary.items()
            ],
        },
        "score": {
            "feature": "zone_index",
            "values": {
                "positional": 0,
                "mixed": 1,
                "velocity_heavy": 2,
            },
            "training": False,
        },
        "observed": {
            "J2_cond": observed_j2,
            "J_cond": observed_j2 / 2.0,
            "D_cond": observed_d,
            "AUC_cond": auc_cond,
            "AUC_delta_vs_constant": auc_cond - 0.5,
        },
        "exact_null": exact_null,
        "permutation_sanity": perm,
        "v10b_comparison": {
            "global_pooled_heldout_auc": V10B_POOLED_AUC,
            "v11_conditional_auc": auc_cond,
            "note": "v0.10b global AUC remains the locked held-out null; v0.11 is conditional and in-sample.",
        },
        "m3_0p4_diagnostic": next(row for row in per_stratum if row["m3"] == "0.4"),
        "branch_label_alignment_report_only": dict(sorted(branch_alignment.items())),
        "verdict": verdict,
    })

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    write_csv(
        os.path.join(args.out_dir, "per_stratum_table.csv"),
        per_stratum,
        [
            "m3", "N", "S", "U",
            "zone0_N", "zone1_N", "zone2_N",
            "zone0_S", "zone1_S", "zone2_S",
            "zone0_U", "zone1_U", "zone2_U",
            "J2_m", "J_m", "D_m", "AUC_m", "pair_weight",
        ],
    )
    write_csv(
        os.path.join(args.out_dir, "report_only_strata.csv"),
        report_rows,
        ["bucket", "m3", "N", "S", "U", "zone0_N", "zone1_N", "zone2_N",
         "J2_m", "D_m", "AUC_m"],
    )
    write_csv(
        os.path.join(args.out_dir, "exact_null_summary.csv"),
        exact_summaries + [{
            "m3": "combined",
            "feasible_allocation_count": "",
            "support_size": len(combined),
            "probability_mass": combined_mass,
        }],
        ["m3", "feasible_allocation_count", "support_size", "probability_mass"],
    )

    print("[v0.11] frozen-input cross-check PASSED")
    print(f"[v0.11] primary domain: {EXPECTED_PRIMARY_TOTAL[0]} rows "
          f"({EXPECTED_PRIMARY_TOTAL[1]} S / {EXPECTED_PRIMARY_TOTAL[2]} U), "
          f"D_cond={observed_d}")
    print(f"[v0.11] observed AUC_cond={auc_cond:.4f} "
          f"(J={observed_j2 / 2.0:.1f}, J2={observed_j2})")
    print(f"[v0.11] exact null: support={len(combined)}, mass={combined_mass:.10f}, "
          f"p_exact={exact_p:.4e}")
    print(f"[v0.11] permutation sanity: p={perm['p_perm_sanity']:.4e}, "
          f"se={perm['standard_error']:.4e}, consistent={perm['consistent_with_exact']}")
    print(f"[v0.11] v0.10b global held-out AUC remains {V10B_POOLED_AUC:.4f}; "
          f"v0.11 conditional AUC is {auc_cond:.4f}")
    print(f"[v0.11] VERDICT: {verdict}")


if __name__ == "__main__":
    main()
