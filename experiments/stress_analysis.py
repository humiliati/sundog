"""Plot stress curves from sweep_summary.json files.

For each stressor, plots mean +/- 95% bootstrap CI of terminal target
intensity vs stressor level, one line per condition. Saved as PNG in
results/stress_tests/<stressor>/stress_curve.png.

Also emits a stress_summary.csv with one row per (stressor, level, condition)
for inclusion in the paper's Table 4.x.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    n = values.shape[0]
    if n == 0:
        return (0.0, 0.0)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = values[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


COLORS = {
    "photometric": "tab:blue",
    "doa_direct":  "tab:green",
    "doa_noisy":   "tab:orange",
    "random":      "tab:red",
}


def plot_one_sweep(sweep_path: str, out_dir: str | None = None) -> None:
    if not _HAS_MPL:
        print("[stress_analysis] matplotlib not available; skipping plot")
        return
    with open(sweep_path) as f:
        summary = json.load(f)
    stressor = summary["stressor"]
    levels = summary["levels"]
    out_dir = out_dir or os.path.dirname(sweep_path)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    conditions = sorted({cond for d in summary["results"].values() for cond in d.keys()})
    for cond in conditions:
        means, ci_lo, ci_hi = [], [], []
        xs = []
        for level in levels:
            entry = summary["results"].get(str(level), {}).get(cond)
            if entry is None:
                continue
            xs.append(level)
            vals = np.asarray(entry["values"])
            means.append(float(vals.mean()))
            lo, hi = bootstrap_ci(vals)
            ci_lo.append(lo)
            ci_hi.append(hi)
        if not xs:
            continue
        c = COLORS.get(cond, None)
        ax.plot(xs, means, "-o", label=cond, color=c)
        ax.fill_between(xs, ci_lo, ci_hi, alpha=0.2, color=c)

    ax.set_xlabel(stressor)
    ax.set_ylabel("terminal target intensity (mean +/- 95% CI)")
    ax.set_title(f"Stress sweep: {stressor}")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(out_dir, "stress_curve.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[stress_analysis] {stressor} -> {out_path}")


def emit_csv(stress_root: str, out_csv: str) -> None:
    sweeps = glob(os.path.join(stress_root, "*", "sweep_summary.json"))
    rows = []
    for s in sweeps:
        with open(s) as f:
            summary = json.load(f)
        stressor = summary["stressor"]
        levels = summary["levels"]
        for level in levels:
            for cond, entry in summary["results"].get(str(level), {}).items():
                rows.append({
                    "stressor": stressor,
                    "level": level,
                    "condition": cond,
                    "mean_terminal": entry["mean_terminal"],
                    "std_terminal": entry["std_terminal"],
                    "n": len(entry["values"]),
                })
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["stressor", "level", "condition", "mean_terminal", "std_terminal", "n"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[stress_analysis] csv -> {out_csv}  ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stress-root", type=str,
                        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "stress_tests"))
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    sweeps = sorted(glob(os.path.join(args.stress_root, "*", "sweep_summary.json")))
    if not sweeps:
        print(f"[stress_analysis] no sweeps found under {args.stress_root}")
        return
    for s in sweeps:
        plot_one_sweep(s)
    csv_out = args.csv_out or os.path.join(args.stress_root, "stress_summary.csv")
    emit_csv(args.stress_root, csv_out)


if __name__ == "__main__":
    main()
