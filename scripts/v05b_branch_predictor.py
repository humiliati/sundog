"""v0.5b held-out branch predictor.

Implements the signed-off registration in
`docs/isotrophy/kfacet/kfacet_v05b_branch_predictor_form.md`.

The runner uses only the v0.5a per-row branch table. It performs no
integration, no tangent-space computation, and no feature search.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "results/isotrophy/k-facet-v05a-branch-map/per_row_table.csv"
DEFAULT_MANIFEST = ROOT / "results/isotrophy/k-facet-v05a-branch-map/manifest.json"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v05b-branch-predictor"

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v05b_branch_predictor_form.md"
VERSION = "v0.5b-branch-predictor-heldout"

GATING_M3 = ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.5", "1.6", "1.7"]
REPORT_ONLY_M3 = ["1.3", "1.4", "1.9"]
ACTIVE_BITS = ["b1_m3_lt_1", "b2_z0_lt_0p3"]
RANDOM_HALF_SEED = 20260523
MCNEMAR_ALPHA = 0.01
LOW_DISCORDANCE_FLOOR = 10


def format_m3(value: str | float) -> str:
    return f"{float(value):.1f}"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise ValueError(f"expected boolean text, got {value!r}")


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {
                "label": raw["label"],
                "index": int(raw["index"]),
                "m3": float(raw["m3"]),
                "m3_key": format_m3(raw["m3"]),
                "z0": float(raw["z0"]),
                "period": float(raw["period"]),
                "stability": raw["stability"],
                "vz": float(raw["vz"]),
                "abs_vz": float(raw["abs_vz"]),
                "m3_z0_squared": float(raw["m3_z0_squared"]),
                "branch_label": raw["branch_label"],
                "b1_m3_lt_1": parse_bool(raw["b1_m3_lt_1"]),
                "b2_z0_lt_0p3": parse_bool(raw["b2_z0_lt_0p3"]),
                "b3_abs_vz_lt_1e_minus_6": parse_bool(raw["b3_abs_vz_lt_1e_minus_6"]),
                "b4_m3_z0_sq_lt_2": parse_bool(raw["b4_m3_z0_sq_lt_2"]),
            }
            rows.append(row)
    return rows


def verify_v05a_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    active_bits = manifest.get("summary", {}).get("active_bits")
    if active_bits != ACTIVE_BITS:
        raise ValueError(f"v0.5a active bits mismatch: expected {ACTIVE_BITS}, found {active_bits}")
    return manifest


def stability_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["stability"] for row in rows)
    return {"S": counts.get("S", 0), "U": counts.get("U", 0)}


def branch_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: {"S": 0, "U": 0})
    for row in rows:
        counts[row["branch_label"]][row["stability"]] += 1
    return dict(counts)


def train_branch_predictor(train_rows: list[dict]) -> tuple[dict[str, dict], str]:
    counts = branch_counts(train_rows)
    global_counts = stability_counts(train_rows)
    global_prediction = "S" if global_counts["S"] > global_counts["U"] else "U"
    model: dict[str, dict] = {}
    for branch, c in counts.items():
        prediction = "S" if c["S"] > c["U"] else "U"
        p_hat_s = (c["S"] + 0.5) / (c["S"] + c["U"] + 1.0)
        model[branch] = {
            "training_S": c["S"],
            "training_U": c["U"],
            "predicted_class": prediction,
            "p_hat_S": p_hat_s,
            "fallback_used": False,
        }
    return model, global_prediction


def predict_branch(model: dict[str, dict], global_prediction: str, branch: str) -> dict:
    if branch in model:
        return model[branch]
    return {
        "training_S": 0,
        "training_U": 0,
        "predicted_class": global_prediction,
        "p_hat_S": 0.5,
        "fallback_used": True,
    }


def score_predictions(rows: list[dict], prediction_key: str = "predicted_class") -> dict:
    model_correct = 0
    baseline_correct = 0
    win = 0
    loss = 0
    both_correct = 0
    both_wrong = 0
    brier_total = 0.0
    log_loss_total = 0.0
    class_totals = {"S": 0, "U": 0}
    class_correct_model = {"S": 0, "U": 0}
    class_correct_baseline = {"S": 0, "U": 0}

    for row in rows:
        truth = row["stability"]
        pred = row[prediction_key]
        baseline = row.get("baseline_class", "U")
        p_hat_s = float(row.get("p_hat_S", 1.0 if pred == "S" else 0.0))
        p_hat_s = min(max(p_hat_s, 1e-12), 1.0 - 1e-12)
        target_s = 1.0 if truth == "S" else 0.0
        brier_total += (p_hat_s - target_s) ** 2
        log_loss_total += -(math.log(p_hat_s) if truth == "S" else math.log(1.0 - p_hat_s))

        is_model_correct = pred == truth
        is_baseline_correct = baseline == truth
        model_correct += int(is_model_correct)
        baseline_correct += int(is_baseline_correct)
        class_totals[truth] += 1
        class_correct_model[truth] += int(is_model_correct)
        class_correct_baseline[truth] += int(is_baseline_correct)

        if is_model_correct and not is_baseline_correct:
            win += 1
            row["discordance"] = "win"
        elif (not is_model_correct) and is_baseline_correct:
            loss += 1
            row["discordance"] = "loss"
        elif is_model_correct and is_baseline_correct:
            both_correct += 1
            row["discordance"] = "both_correct"
        else:
            both_wrong += 1
            row["discordance"] = "both_wrong"

        row["model_correct"] = is_model_correct
        row["baseline_correct"] = is_baseline_correct

    n = len(rows)
    n_disc = win + loss
    p_value = exact_one_sided_binomial_tail(win, n_disc) if win > loss else 1.0
    accuracy_model = model_correct / n if n else 0.0
    accuracy_baseline = baseline_correct / n if n else 0.0
    bal_model = balanced_accuracy(class_correct_model, class_totals)
    bal_baseline = balanced_accuracy(class_correct_baseline, class_totals)

    return {
        "N": n,
        "S": class_totals["S"],
        "U": class_totals["U"],
        "model_correct": model_correct,
        "baseline_correct": baseline_correct,
        "accuracy_model": accuracy_model,
        "accuracy_always_U": accuracy_baseline,
        "accuracy_delta": accuracy_model - accuracy_baseline,
        "balanced_accuracy_model": bal_model,
        "balanced_accuracy_always_U": bal_baseline,
        "win": win,
        "loss": loss,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "n_discordant": n_disc,
        "mcnemar_exact_one_sided_p": p_value,
        "brier_score": brier_total / n if n else None,
        "log_loss": log_loss_total / n if n else None,
    }


def balanced_accuracy(correct: dict[str, int], totals: dict[str, int]) -> float | None:
    recalls = []
    for cls in ("S", "U"):
        if totals[cls] > 0:
            recalls.append(correct[cls] / totals[cls])
    return sum(recalls) / len(recalls) if recalls else None


def exact_one_sided_binomial_tail(win: int, n: int) -> float:
    if n <= 0:
        return 1.0
    # n <= 263 here; integer summation is exact and then converted once.
    tail = sum(math.comb(n, k) for k in range(win, n + 1))
    return tail / (2 ** n)


def verdict_from_summary(summary: dict) -> str:
    if summary["n_discordant"] < LOW_DISCORDANCE_FLOOR:
        return "branch_predictor_inconclusive_low_discordance"
    if (
        summary["accuracy_model"] > summary["accuracy_always_U"]
        and summary["win"] > summary["loss"]
        and summary["mcnemar_exact_one_sided_p"] <= MCNEMAR_ALPHA
    ):
        return "branch_predictor_passes_heldout"
    return "branch_predictor_fails_heldout"


def run_primary(rows: list[dict]) -> dict:
    gate_rows = [row for row in rows if row["m3_key"] in GATING_M3]
    report_rows = [row for row in rows if row["m3_key"] in REPORT_ONLY_M3]
    unexpected = sorted({row["m3_key"] for row in rows} - set(GATING_M3) - set(REPORT_ONLY_M3))
    if unexpected:
        raise ValueError(f"unexpected m3 bins in input: {unexpected}")

    per_row_predictions: list[dict] = []
    per_fold_table: list[dict] = []
    branch_training_table: list[dict] = []
    absent_branch_fallback_used = False

    for held_out in GATING_M3:
        train_rows = [row for row in gate_rows if row["m3_key"] != held_out]
        test_rows = [row for row in gate_rows if row["m3_key"] == held_out]
        if not test_rows:
            raise ValueError(f"locked fold m3={held_out} has no rows")

        model, global_prediction = train_branch_predictor(train_rows)
        fold_branches = sorted({row["branch_label"] for row in test_rows} | set(model))

        fold_predictions: list[dict] = []
        for row in test_rows:
            pred_info = predict_branch(model, global_prediction, row["branch_label"])
            absent_branch_fallback_used = absent_branch_fallback_used or pred_info["fallback_used"]
            pred_row = {
                **row,
                "held_out_m3": held_out,
                "predicted_class": pred_info["predicted_class"],
                "baseline_class": "U",
                "p_hat_S": pred_info["p_hat_S"],
                "training_S_for_branch": pred_info["training_S"],
                "training_U_for_branch": pred_info["training_U"],
                "absent_branch_fallback_used": pred_info["fallback_used"],
            }
            fold_predictions.append(pred_row)

        score_predictions(fold_predictions)
        per_row_predictions.extend(fold_predictions)

        for branch in fold_branches:
            pred_info = predict_branch(model, global_prediction, branch)
            branch_test_rows = [row for row in fold_predictions if row["branch_label"] == branch]
            branch_summary = score_predictions([dict(row) for row in branch_test_rows]) if branch_test_rows else {
                "N": 0, "S": 0, "U": 0, "model_correct": 0, "baseline_correct": 0,
                "accuracy_model": None, "accuracy_always_U": None, "accuracy_delta": None,
                "win": 0, "loss": 0, "both_correct": 0, "both_wrong": 0,
            }
            per_fold_table.append({
                "held_out_m3": held_out,
                "branch_label": branch,
                "training_S": pred_info["training_S"],
                "training_U": pred_info["training_U"],
                "predicted_class": pred_info["predicted_class"],
                "p_hat_S": pred_info["p_hat_S"],
                "test_N": branch_summary["N"],
                "test_S": branch_summary["S"],
                "test_U": branch_summary["U"],
                "fold_model_correct": branch_summary["model_correct"],
                "fold_baseline_correct": branch_summary["baseline_correct"],
                "fold_win": branch_summary["win"],
                "fold_loss": branch_summary["loss"],
                "fold_both_correct": branch_summary["both_correct"],
                "fold_both_wrong": branch_summary["both_wrong"],
                "absent_branch_fallback_used": pred_info["fallback_used"],
            })
            branch_training_table.append({
                "held_out_m3": held_out,
                "branch_label": branch,
                "training_S": pred_info["training_S"],
                "training_U": pred_info["training_U"],
                "predicted_class": pred_info["predicted_class"],
                "p_hat_S": pred_info["p_hat_S"],
                "absent_branch_fallback_used": pred_info["fallback_used"],
            })

    primary_summary = score_predictions(per_row_predictions)
    primary_summary["verdict"] = verdict_from_summary(primary_summary)
    primary_summary["absent_branch_fallback_used"] = absent_branch_fallback_used

    report_only_predictions = score_report_only_rows(gate_rows, report_rows)

    return {
        "gate_rows": gate_rows,
        "report_only_rows": report_rows,
        "per_row_predictions": per_row_predictions,
        "per_fold_table": per_fold_table,
        "branch_training_table": branch_training_table,
        "primary_summary": primary_summary,
        "report_only_predictions": report_only_predictions,
    }


def score_report_only_rows(gate_rows: list[dict], report_rows: list[dict]) -> list[dict]:
    if not report_rows:
        return []
    model, global_prediction = train_branch_predictor(gate_rows)
    scored = []
    for row in report_rows:
        pred_info = predict_branch(model, global_prediction, row["branch_label"])
        scored.append({
            **row,
            "held_out_m3": "report_only",
            "predicted_class": pred_info["predicted_class"],
            "baseline_class": "U",
            "p_hat_S": pred_info["p_hat_S"],
            "training_S_for_branch": pred_info["training_S"],
            "training_U_for_branch": pred_info["training_U"],
            "absent_branch_fallback_used": pred_info["fallback_used"],
        })
    score_predictions(scored)
    return scored


def run_single_rule_sidecar(gate_rows: list[dict]) -> tuple[dict, list[dict]]:
    rows = []
    for row in gate_rows:
        pred = "S" if (row["b1_m3_lt_1"] and row["b2_z0_lt_0p3"]) else "U"
        rows.append({
            **row,
            "predicted_class": pred,
            "baseline_class": "U",
            "p_hat_S": 1.0 if pred == "S" else 0.0,
        })
    summary = score_predictions(rows)
    summary["verdict_like"] = verdict_from_summary(summary)
    return summary, rows


def run_random_half_sidecar(gate_rows: list[dict]) -> tuple[dict, list[dict]]:
    rows = [dict(row) for row in gate_rows]
    rng = random.Random(RANDOM_HALF_SEED)
    rows_sorted = sorted(rows, key=lambda row: (row["m3"], row["index"], row["label"]))
    rng.shuffle(rows_sorted)
    split_at = len(rows_sorted) // 2
    halves = {"A": rows_sorted[:split_at], "B": rows_sorted[split_at:]}

    predictions: list[dict] = []
    for train_name, test_name in (("A", "B"), ("B", "A")):
        train_rows = halves[train_name]
        test_rows = halves[test_name]
        model, global_prediction = train_branch_predictor(train_rows)
        for row in test_rows:
            pred_info = predict_branch(model, global_prediction, row["branch_label"])
            predictions.append({
                **row,
                "train_half": train_name,
                "test_half": test_name,
                "predicted_class": pred_info["predicted_class"],
                "baseline_class": "U",
                "p_hat_S": pred_info["p_hat_S"],
                "training_S_for_branch": pred_info["training_S"],
                "training_U_for_branch": pred_info["training_U"],
                "absent_branch_fallback_used": pred_info["fallback_used"],
            })
    summary = score_predictions(predictions)
    summary["seed"] = RANDOM_HALF_SEED
    summary["verdict_like"] = verdict_from_summary(summary)
    return summary, predictions


def build_result(rows: list[dict], input_path: Path, manifest_path: Path) -> dict:
    primary = run_primary(rows)
    single_rule_summary, single_rule_rows = run_single_rule_sidecar(primary["gate_rows"])
    random_half_summary, random_half_rows = run_random_half_sidecar(primary["gate_rows"])

    gate_counts = stability_counts(primary["gate_rows"])
    report_counts = stability_counts(primary["report_only_rows"])
    branch_counts_gate = branch_counts(primary["gate_rows"])

    return {
        "mode": "kfacet_v05b_branch_predictor",
        "version": VERSION,
        "form_lock": FORM_LOCK,
        "input_per_row_table": str(input_path.relative_to(ROOT)) if input_path.is_relative_to(ROOT) else str(input_path),
        "input_v05a_manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "active_bits": ACTIVE_BITS,
        "primary_partition": "leave_one_m3_bin_out",
        "predictor_form": "fold_trained_branch_majority_on_active_b1_b2",
        "tie_rule": "U",
        "baseline": "always_U",
        "thresholds": {
            "mcnemar_alpha": MCNEMAR_ALPHA,
            "low_discordance_floor": LOW_DISCORDANCE_FLOOR,
            "random_half_seed": RANDOM_HALF_SEED,
        },
        "gating_m3_bins": GATING_M3,
        "excluded_report_only_m3_bins": REPORT_ONLY_M3,
        "gate_counts": {
            "N_gate": len(primary["gate_rows"]),
            "S_gate": gate_counts["S"],
            "U_gate": gate_counts["U"],
            "branch_counts": branch_counts_gate,
        },
        "report_only_counts": {
            "N_report_only": len(primary["report_only_rows"]),
            "S_report_only": report_counts["S"],
            "U_report_only": report_counts["U"],
        },
        "primary_summary": primary["primary_summary"],
        "sidecar_single_rule_summary": single_rule_summary,
        "sidecar_random_half_summary": random_half_summary,
        "report_only_summary": score_predictions(primary["report_only_predictions"]) if primary["report_only_predictions"] else None,
        "diagnostics_summary": {
            "continuous_features_used_for_prediction": [],
            "continuous_features_report_only": [
                "period", "abs_vz", "vz", "m3_z0_squared",
            ],
        },
        "_tables": {
            "per_row_predictions": primary["per_row_predictions"],
            "per_fold_table": primary["per_fold_table"],
            "branch_training_table": primary["branch_training_table"],
            "sidecar_single_rule_rows": single_rule_rows,
            "sidecar_random_half_rows": random_half_rows,
            "report_only_predictions": primary["report_only_predictions"],
        },
    }


def write_receipts(result: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tables = result.pop("_tables")
    try:
        (out_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

        write_csv(out_dir / "per_row_predictions.csv", tables["per_row_predictions"], [
            "label", "index", "m3", "m3_key", "z0", "period", "stability",
            "branch_label", "held_out_m3", "training_S_for_branch",
            "training_U_for_branch", "predicted_class", "baseline_class",
            "p_hat_S", "model_correct", "baseline_correct", "discordance",
            "absent_branch_fallback_used",
        ])
        write_csv(out_dir / "per_fold_table.csv", tables["per_fold_table"], [
            "held_out_m3", "branch_label", "training_S", "training_U",
            "predicted_class", "p_hat_S", "test_N", "test_S", "test_U",
            "fold_model_correct", "fold_baseline_correct", "fold_win",
            "fold_loss", "fold_both_correct", "fold_both_wrong",
            "absent_branch_fallback_used",
        ])
        write_csv(out_dir / "branch_training_table.csv", tables["branch_training_table"], [
            "held_out_m3", "branch_label", "training_S", "training_U",
            "predicted_class", "p_hat_S", "absent_branch_fallback_used",
        ])
        write_csv(out_dir / "sidecar_single_rule.csv", tables["sidecar_single_rule_rows"], [
            "label", "index", "m3", "m3_key", "z0", "period", "stability",
            "branch_label", "predicted_class", "baseline_class",
            "model_correct", "baseline_correct", "discordance",
        ])
        write_csv(out_dir / "sidecar_random_half.csv", tables["sidecar_random_half_rows"], [
            "label", "index", "m3", "m3_key", "z0", "period", "stability",
            "branch_label", "train_half", "test_half", "training_S_for_branch",
            "training_U_for_branch", "predicted_class", "baseline_class",
            "p_hat_S", "model_correct", "baseline_correct", "discordance",
            "absent_branch_fallback_used",
        ])
        if tables["report_only_predictions"]:
            write_csv(out_dir / "report_only_predictions.csv", tables["report_only_predictions"], [
                "label", "index", "m3", "m3_key", "z0", "period", "stability",
                "branch_label", "predicted_class", "baseline_class",
                "p_hat_S", "model_correct", "baseline_correct", "discordance",
            ])
    finally:
        result["_tables"] = tables


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows = load_rows(args.input)
    verify_v05a_manifest(args.manifest)
    result = build_result(rows, args.input.resolve(), args.manifest.resolve())
    write_receipts(result, args.out)

    summary = result["primary_summary"]
    print("[v05b-branch-predictor] verdict:", summary["verdict"])
    print(f"  gate rows:       {summary['N']}  S={summary['S']} U={summary['U']}")
    print(f"  accuracy:        model={summary['accuracy_model']:.6f} always_U={summary['accuracy_always_U']:.6f} delta={summary['accuracy_delta']:.6f}")
    print(f"  McNemar:         win={summary['win']} loss={summary['loss']} n_disc={summary['n_discordant']} p={summary['mcnemar_exact_one_sided_p']:.6g}")
    print(f"  manifest:        {args.out / 'manifest.json'}")
    print(f"  per-fold table:  {args.out / 'per_fold_table.csv'}")


if __name__ == "__main__":
    main()
