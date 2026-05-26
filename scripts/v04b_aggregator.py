"""v0.4b aggregator: read kfacet-row-z2-sweep receipts, apply the locked
threshold-rule baseline, and emit the verdict against the pre-registered
chi-squared(12) falsifier.

Form-lock reference: docs/isotrophy/kfacet/kfacet_v04b_gamma3_form.md.
Pre-registration: docs/isotrophy/kfacet/kfacet_v04b_mechanism_preregistration.md.

Verdict outcomes:
  pass                  -- chi-squared test stat <= 26.22 at chi-squared(12).
                           Threshold rule consistent with the observed table.
  fail                  -- chi-squared test stat > 26.22. This BASELINE rule
                           is falsified; gamma_3 family still open via
                           gamma_3' (e.g., ratio rule); a separate
                           registration would be required.
  retired_no_variation  -- >= 99% of rows have the same threshold-rule
                           prediction. Predictor has nothing to discriminate
                           on; baseline retires as non-informative.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

V04B_VERSION = "v0.4b-aggregator"
V04B_CHI2_CRITICAL = 26.22  # chi-squared(12), p=0.01
V04B_CHI2_DF = 12
V04B_GATING_MIN_N = 5
V04B_NO_VARIATION_FRACTION = 0.99

ROOT = Path(__file__).resolve().parent.parent
SWEEP_ROOT = ROOT / "results/isotrophy/k-facet-v04b-z2-sweep"
OUT_DIR = ROOT / "results/isotrophy/k-facet-v04b-z2-sweep"

# Same m_3 strata as v0.4a, computed empirically.
M3_VALUES = ("0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1",
             "1.2", "1.3", "1.4", "1.5", "1.6", "1.7", "1.9")


def load_rows() -> list[dict]:
    rows: list[dict] = []
    for m3 in M3_VALUES:
        m3_dir = SWEEP_ROOT / f"m3eq{m3}"
        if not m3_dir.is_dir():
            continue
        for row_dir in sorted(m3_dir.iterdir()):
            if not row_dir.is_dir() or not row_dir.name.startswith("O"):
                continue
            receipt_path = row_dir / "row_z2_receipt.json"
            if not receipt_path.is_file():
                continue
            r = json.loads(receipt_path.read_text(encoding="utf-8"))
            r["m3_input"] = m3
            rows.append(r)
    return rows


def aggregate(rows: list[dict]) -> dict:
    # Build per-m_3 contingency:
    per_m3 = {m3: {"S": 0, "U": 0, "predicted_S": 0, "predicted_U": 0,
                   "n": 0, "even_dim": [], "odd_dim": []}
              for m3 in M3_VALUES}
    for r in rows:
        m3 = r["m3_input"]
        if m3 not in per_m3:
            continue
        per_m3[m3]["n"] += 1
        if r["observed_stability"] == "S":
            per_m3[m3]["S"] += 1
        else:
            per_m3[m3]["U"] += 1
        if r["gamma3_prediction"] == "S":
            per_m3[m3]["predicted_S"] += 1
        else:
            per_m3[m3]["predicted_U"] += 1
        per_m3[m3]["even_dim"].append(r["F_beta_even_dim"])
        per_m3[m3]["odd_dim"].append(r["F_beta_odd_dim"])

    # No-variation retirement check
    total = len(rows)
    predicted_S_total = sum(1 for r in rows if r["gamma3_prediction"] == "S")
    predicted_U_total = total - predicted_S_total
    no_variation = (
        total == 0
        or (predicted_S_total / total) >= V04B_NO_VARIATION_FRACTION
        or (predicted_U_total / total) >= V04B_NO_VARIATION_FRACTION
    )

    # Chi-squared over gating bins (N >= V04B_GATING_MIN_N)
    chi2 = 0.0
    gating_bins = []
    diagnostic_bins = []
    for m3, c in per_m3.items():
        if c["n"] == 0:
            continue
        p_predicted = c["predicted_S"] / c["n"]
        observed_S = c["S"]
        expected_S = p_predicted * c["n"]
        expected_U = (1.0 - p_predicted) * c["n"]
        # Per-bin chi-squared contribution
        contribution = 0.0
        if expected_S > 0:
            contribution += (observed_S - expected_S) ** 2 / expected_S
        if expected_U > 0:
            contribution += ((c["U"]) - expected_U) ** 2 / expected_U
        bin_summary = {
            "m3": m3,
            "n": c["n"],
            "S_observed": c["S"],
            "U_observed": c["U"],
            "S_predicted_fraction": p_predicted,
            "S_expected": expected_S,
            "U_expected": expected_U,
            "chi2_contribution": contribution,
            "binomial_residual_sigma": (
                (observed_S - expected_S) / math.sqrt(max(expected_S * (1 - p_predicted), 1e-12))
                if p_predicted not in (0.0, 1.0) else 0.0
            ),
        }
        if c["n"] >= V04B_GATING_MIN_N:
            chi2 += contribution
            bin_summary["gating"] = True
            gating_bins.append(bin_summary)
        else:
            bin_summary["gating"] = False
            diagnostic_bins.append(bin_summary)

    # Verdict
    if no_variation:
        verdict = "retired_no_variation"
    elif chi2 > V04B_CHI2_CRITICAL:
        verdict = "fail"
    else:
        verdict = "pass"

    correct = sum(1 for r in rows if r["gamma3_prediction_correct"])
    return {
        "mode": "kfacet_v04b_aggregator",
        "version": V04B_VERSION,
        "form_lock": "docs/isotrophy/kfacet/kfacet_v04b_gamma3_form.md",
        "thresholds": {
            "chi2_critical": V04B_CHI2_CRITICAL,
            "chi2_df": V04B_CHI2_DF,
            "gating_min_N": V04B_GATING_MIN_N,
            "no_variation_fraction": V04B_NO_VARIATION_FRACTION,
        },
        "summary": {
            "total_rows": total,
            "predicted_S": predicted_S_total,
            "predicted_U": predicted_U_total,
            "predicted_S_fraction": predicted_S_total / total if total else 0.0,
            "correct_predictions": correct,
            "accuracy": correct / total if total else 0.0,
            "chi2_test_statistic": chi2,
            "no_variation_retired": no_variation,
        },
        "gating_bins": gating_bins,
        "diagnostic_bins": diagnostic_bins,
        "per_row_table": rows,
        "verdict": verdict,
    }


def main():
    rows = load_rows()
    if not rows:
        print("[v04b-aggregator] no rows loaded from sweep root", SWEEP_ROOT)
        return 1
    result = aggregate(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    # Per-row CSV
    per_row_csv = OUT_DIR / "per_row_table.csv"
    with per_row_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row_index", "label", "m3", "z0", "period", "stability",
            "kernel_dim", "neutral_dim", "K_fib_dim",
            "F_beta_even_dim", "F_beta_odd_dim",
            "bridge_band_count", "F_beta_leakage_inf",
            "neutral_sector_conditioning",
            "gamma3_prediction", "observed_stability",
            "gamma3_prediction_correct",
        ])
        for r in rows:
            writer.writerow([
                r["row_index"], r["label"], r["m3"], r["z0"], r["period"], r["stability"],
                r["kernel_dim"], r["neutral_dim"], r["K_fib_dim"],
                r["F_beta_even_dim"], r["F_beta_odd_dim"],
                r["bridge_band_count"], r["F_beta_leakage_inf"],
                r["neutral_sector_conditioning"],
                r["gamma3_prediction"], r["observed_stability"],
                r["gamma3_prediction_correct"],
            ])
    # Per-m_3 CSV
    per_m3_csv = OUT_DIR / "per_m3_table.csv"
    with per_m3_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "m3", "n", "S_observed", "U_observed", "S_predicted_fraction",
            "S_expected", "U_expected", "chi2_contribution",
            "binomial_residual_sigma", "gating",
        ])
        for b in result["gating_bins"] + result["diagnostic_bins"]:
            writer.writerow([
                b["m3"], b["n"], b["S_observed"], b["U_observed"],
                b["S_predicted_fraction"], b["S_expected"], b["U_expected"],
                b["chi2_contribution"], b["binomial_residual_sigma"], b["gating"],
            ])

    print(f"[v04b-aggregator] verdict: {result['verdict']}")
    print(f"  rows: {result['summary']['total_rows']}")
    print(f"  predicted_S: {result['summary']['predicted_S']}  "
          f"({result['summary']['predicted_S_fraction']:.4f})")
    print(f"  accuracy:    {result['summary']['accuracy']:.4f}  "
          f"({result['summary']['correct_predictions']}/{result['summary']['total_rows']})")
    print(f"  chi^2:       {result['summary']['chi2_test_statistic']:.4f}  "
          f"(critical {V04B_CHI2_CRITICAL} at p=0.01, df={V04B_CHI2_DF})")
    print(f"  manifest:    {OUT_DIR / 'manifest.json'}")
    print(f"  per_row:     {per_row_csv}")
    print(f"  per_m3:      {per_m3_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
