"""Analysis of the baseline comparison.

Reads results/{condition}/seed_*.npz produced by run_baseline_comparison.py.
Produces:

- analysis_summary.json    : per-condition statistics + pairwise tests
- convergence_curves.png   : mean +/- 95% bootstrap CI of target intensity
                             vs step, per condition
- terminal_intensity_box.png : box plot of terminal intensity per condition
- convergence_time_box.png   : box plot of time-to-90% per condition

Metrics
-------
- terminal_intensity : mean of target_intensity over the last 50 steps
- time_to_90         : first step where target_intensity > 0.9; if never
                       reached, set to N_STEPS (right-censored)
- terminal_joint_std : std of joint_angles[:,0] and [:,1] over last 100
                       steps, then mean of the two stds (a single
                       scalar per episode)

Statistical tests
-----------------
Mann-Whitney U two-sided, photometric vs each other condition, on
terminal_intensity. Reports U statistic and p-value. Also reports
bootstrap 95% CI on the difference of means.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np

# Matplotlib is optional — if not installed, we still emit the JSON summary.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False

# scipy.stats is optional too; we fall back to a numpy Mann-Whitney
# implementation if it isn't installed.
try:
    from scipy import stats as scipy_stats
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


RESULTS_DIR_DEFAULT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "..",
    "results",
)
CONDITIONS_DEFAULT = ["photometric", "doa_direct", "doa_noisy", "random"]
CONVERGENCE_THRESHOLD = 0.9
TERMINAL_WINDOW = 50
STABILITY_WINDOW = 100


@dataclass
class ConditionResults:
    name: str
    target_intensity: np.ndarray  # shape (n_seeds, n_steps)
    terminal_intensity: np.ndarray  # shape (n_seeds,)
    time_to_threshold: np.ndarray  # shape (n_seeds,)
    terminal_joint_std: np.ndarray  # shape (n_seeds,)


def load_condition(results_dir: str, condition: str) -> ConditionResults:
    cond_dir = os.path.join(results_dir, condition)
    if not os.path.isdir(cond_dir):
        raise FileNotFoundError(f"missing condition directory: {cond_dir}")
    files = sorted(f for f in os.listdir(cond_dir) if f.endswith(".npz"))
    if not files:
        raise RuntimeError(f"no npz files in {cond_dir}")

    target_intensity_runs = []
    joint_angles_runs = []
    for f in files:
        d = np.load(os.path.join(cond_dir, f))
        target_intensity_runs.append(d["target_intensity"])
        joint_angles_runs.append(d["joint_angles"])

    target_intensity = np.stack(target_intensity_runs)
    joint_angles = np.stack(joint_angles_runs)  # (n_seeds, n_steps, 2)

    n_steps = target_intensity.shape[1]
    terminal = target_intensity[:, -TERMINAL_WINDOW:].mean(axis=1)

    # First step where intensity exceeds threshold; n_steps if never.
    above = target_intensity > CONVERGENCE_THRESHOLD
    time_to = np.full(target_intensity.shape[0], n_steps, dtype=np.float64)
    for i in range(target_intensity.shape[0]):
        idx = np.argmax(above[i])
        if above[i, idx]:
            time_to[i] = float(idx)

    # Stability: std over last STABILITY_WINDOW steps, mean of the two joints.
    last = joint_angles[:, -STABILITY_WINDOW:, :]
    joint_std = last.std(axis=1).mean(axis=1)

    return ConditionResults(
        name=condition,
        target_intensity=target_intensity,
        terminal_intensity=terminal,
        time_to_threshold=time_to,
        terminal_joint_std=joint_std,
    )


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05) -> tuple[float, float]:
    """Bootstrap percentile CI of the mean of `values`."""
    rng = np.random.default_rng(0)
    n = values.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = values[idx]
    means = samples.mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_curve_ci(curves: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    """Per-step bootstrap CI of mean across seeds.

    curves : (n_seeds, n_steps)
    Returns (mean, lo, hi) each shape (n_steps,).
    """
    rng = np.random.default_rng(0)
    n_seeds, n_steps = curves.shape
    idx = rng.integers(0, n_seeds, size=(n_boot, n_seeds))
    boot_means = curves[idx].mean(axis=1)  # (n_boot, n_steps)
    mean = curves.mean(axis=0)
    lo = np.quantile(boot_means, alpha / 2, axis=0)
    hi = np.quantile(boot_means, 1 - alpha / 2, axis=0)
    return mean, lo, hi


def mann_whitney_u(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    if _HAS_SCIPY:
        result = scipy_stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    # Fallback: rank-based exact U.
    combined = np.concatenate([a, b])
    ranks = combined.argsort().argsort().astype(float) + 1.0
    # Average ranks for ties.
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            mask = inverse == i
            ranks[mask] = ranks[mask].mean()
    n_a = len(a)
    n_b = len(b)
    u_a = ranks[:n_a].sum() - n_a * (n_a + 1) / 2
    u = min(u_a, n_a * n_b - u_a)
    # Approximate p-value via normal approximation (large-n).
    mu = n_a * n_b / 2
    sd = np.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    z = (u - mu) / sd if sd > 0 else 0.0
    # Two-sided p from standard normal CDF approximation.
    p = float(2 * 0.5 * (1 - _erf(abs(z) / np.sqrt(2))))
    return float(u), p


def _erf(x: float) -> float:
    # Abramowitz & Stegun 7.1.26.
    a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
    p = 0.3275911
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    return float(sign * y)


def make_plots(conditions: dict[str, ConditionResults], out_dir: str) -> None:
    if not _HAS_MPL:
        print("[analysis] matplotlib not available, skipping plots")
        return

    os.makedirs(out_dir, exist_ok=True)

    # 1) convergence curves
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {
        "photometric": "tab:blue",
        "doa_direct":  "tab:green",
        "doa_noisy":   "tab:orange",
        "random":      "tab:red",
    }
    for name, res in conditions.items():
        mean, lo, hi = bootstrap_curve_ci(res.target_intensity)
        steps = np.arange(mean.shape[0])
        c = colors.get(name, None)
        ax.plot(steps, mean, label=name, color=c)
        ax.fill_between(steps, lo, hi, alpha=0.25, color=c)
    ax.set_xlabel("step")
    ax.set_ylabel("target detector intensity")
    ax.set_title("Convergence: mean +/- 95% bootstrap CI")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence_curves.png"), dpi=150)
    plt.close(fig)

    # 2) terminal intensity box
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = list(conditions.keys())
    data = [conditions[n].terminal_intensity for n in labels]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("terminal intensity (last 50 steps)")
    ax.set_title("Terminal target-detector intensity per condition")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "terminal_intensity_box.png"), dpi=150)
    plt.close(fig)

    # 3) time-to-threshold box
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [conditions[n].time_to_threshold for n in labels]
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel(f"steps until intensity > {CONVERGENCE_THRESHOLD}")
    ax.set_title("Convergence time per condition (right-censored at run length)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "convergence_time_box.png"), dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze sundog baseline comparison.")
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR_DEFAULT)
    parser.add_argument("--conditions", type=str, nargs="+", default=CONDITIONS_DEFAULT)
    parser.add_argument("--out-dir", type=str, default=None,
                        help="Where to write plots and summary JSON. "
                             "Defaults to <results-dir>/analysis.")
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    conditions: dict[str, ConditionResults] = {}
    for name in args.conditions:
        try:
            conditions[name] = load_condition(args.results_dir, name)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"[analysis] skipping {name}: {e}")

    if not conditions:
        print("[analysis] no conditions loaded; nothing to do")
        return

    summary: dict = {"conditions": {}, "tests": {}}
    for name, res in conditions.items():
        ti = res.terminal_intensity
        ci_lo, ci_hi = bootstrap_ci(ti)
        ttt_ci_lo, ttt_ci_hi = bootstrap_ci(res.time_to_threshold)
        summary["conditions"][name] = {
            "n_seeds": int(ti.shape[0]),
            "terminal_intensity": {
                "mean": float(ti.mean()),
                "std": float(ti.std(ddof=1)),
                "median": float(np.median(ti)),
                "ci95": [ci_lo, ci_hi],
            },
            "time_to_threshold": {
                "mean": float(res.time_to_threshold.mean()),
                "median": float(np.median(res.time_to_threshold)),
                "ci95": [ttt_ci_lo, ttt_ci_hi],
                "n_failed": int((res.time_to_threshold >= res.target_intensity.shape[1]).sum()),
            },
            "terminal_joint_std": {
                "mean": float(res.terminal_joint_std.mean()),
                "median": float(np.median(res.terminal_joint_std)),
            },
        }

    if "photometric" in conditions:
        a = conditions["photometric"].terminal_intensity
        for other in conditions:
            if other == "photometric":
                continue
            b = conditions[other].terminal_intensity
            u, p = mann_whitney_u(a, b)
            summary["tests"][f"photometric_vs_{other}_terminal_intensity"] = {
                "U": u, "p": p,
                "mean_diff_photometric_minus_other": float(a.mean() - b.mean()),
            }

    summary_path = os.path.join(out_dir, "analysis_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[analysis] summary -> {summary_path}")

    make_plots(conditions, out_dir)
    if _HAS_MPL:
        print(f"[analysis] plots -> {out_dir}")


if __name__ == "__main__":
    main()
