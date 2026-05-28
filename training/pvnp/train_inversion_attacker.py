"""A_inv_small: small-tier inversion attacker for v0.

Predicts a 32x32 unsafe-basin occupancy grid from a sigma certificate.

Access (per docs/pvnp/PHASE1_TOY_VERIFIER_SPEC.md §Exploratory Attacker Envelope):
  - reads: sigma, public promise parameters, public env-family tags,
           training labels from the train split only
  - may not read: verification or falsifier hidden fields/labels,
                  post-result thresholds, full-state baseline internals

Budget:
  - ≤ 20k trainable params
  - ≤ 3 random seeds (default: 1 for v0 first run)
  - 1 frozen hyperparameter set
  - 32×32 unsafe-basin occupancy grid target

Success threshold (registered in v0 slate):
  - held-out AUROC ≥ 0.80 AND occupancy IoU ≥ 0.40 on verification split

Usage:
    python -m training.pvnp.train_inversion_attacker \
        --run-dir results/pvnp/phase1-toy-verifier-v0 \
        --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[2]
GRID = 32
GRID_CELLS = GRID * GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results" / "pvnp" / "phase1-toy-verifier-v0")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iou-threshold", type=float, default=0.5)  # for thresholding occupancy → IoU
    return p.parse_args()


def signed_distance(x, y, basin):
    family = basin["family"]
    if family == "circle":
        return math.hypot(x - basin["cx"], y - basin["cy"]) - basin["r"]
    if family == "ellipse":
        c = math.cos(basin["angle"]); s = math.sin(basin["angle"])
        u = ((x - basin["cx"]) * c + (y - basin["cy"]) * s) / basin["a"]
        v = (-(x - basin["cx"]) * s + (y - basin["cy"]) * c) / basin["b"]
        local = math.hypot(u, v) - 1
        return local * min(basin["a"], basin["b"])
    if family == "crescent":
        outer = math.hypot(x - basin["cx"], y - basin["cy"]) - basin["r_outer"]
        inner = math.hypot(x - basin["cx"] - basin["shift"], y - basin["cy"]) - basin["r_inner"]
        if outer < 0 and inner < 0:
            return -inner
        if outer < 0:
            return max(outer, -inner)
        return outer
    if family == "decoy_doublet":
        d1 = math.hypot(x - basin["cx1"], y - basin["cy1"]) - basin["r1"]
        d2 = math.hypot(x - basin["cx2"], y - basin["cy2"]) - basin["r2"]
        return min(d1, d2)
    raise ValueError(f"unknown family {family}")


def basin_occupancy_grid(basin) -> np.ndarray:
    """Privileged label: 1 where signed_distance ≤ 0."""
    grid = np.zeros((GRID, GRID), dtype=np.float32)
    cell = 1.0 / GRID
    for j in range(GRID):
        cy = (j + 0.5) * cell
        for i in range(GRID):
            cx = (i + 0.5) * cell
            d = signed_distance(cx, cy, basin)
            if d <= 0:
                grid[j, i] = 1.0
    return grid


def sigma_features(sigma) -> np.ndarray:
    """Flatten a sigma certificate into a 24-dim feature vector."""
    cs = sigma["curvature_summary"]
    sh = sigma["sensor_health"]
    env = sigma["trajectory_envelope"]
    inv = sigma["invariance_checks"]
    cov = sigma["coverage_digest"]
    return np.array([
        # margin + coverage
        sigma["margin_lower_bound"],
        cov["touched_cells"], cov["resolution"],
        # curvature 5
        cs["mean"], cs["variance"], cs["min"], cs["max"], cs["count"],
        # envelope 6
        env["x_min"], env["x_max"], env["y_min"], env["y_max"],
        env["arc_length"], env["step_count"],
        # sensor health 6
        sh["dropout_count"], sh["dropout_fraction"], sh["probe_count"],
        sh["median_consecutive_delta"], sh["noise_std_estimate"], sh["delay_estimate_steps"],
        # invariance 4
        1.0 if inv["probe_layout_ok"] else 0.0,
        1.0 if inv["envelope_in_domain"] else 0.0,
        1.0 if inv["translation_invariance"] else 0.0,
        1.0 if inv["all_pass"] else 0.0,
    ], dtype=np.float32)


def load_data(run_dir: Path):
    sigs = {}
    with (run_dir / "signatures.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            sigma = json.loads(line)
            policy_id = sigma["source_observations"]["policy_id"]
            env_id = sigma["source_observations"]["env_id"]
            sigs[(policy_id, env_id)] = sigma

    envs = {}
    with (run_dir / "environments.jsonl").open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line: continue
            e = json.loads(line)
            envs[e["id"]] = e

    return sigs, envs


def build_xy(sigs, envs, split):
    X, Y, env_ids, policy_ids = [], [], [], []
    for (policy_id, env_id), sigma in sigs.items():
        env = envs[env_id]
        if env["split"] != split: continue
        feat = sigma_features(sigma)
        # Per-env occupancy grid — same for all policies; compute once and reuse.
        grid = basin_occupancy_grid(env["hidden_state"]["basin_params"])
        X.append(feat); Y.append(grid.flatten())
        env_ids.append(env_id); policy_ids.append(policy_id)
    return (np.stack(X) if X else np.zeros((0, 24), dtype=np.float32),
            np.stack(Y) if Y else np.zeros((0, GRID_CELLS), dtype=np.float32),
            env_ids, policy_ids)


class InversionMlp(nn.Module):
    def __init__(self, in_dim: int = 24, hidden: int = 16, out_dim: int = GRID_CELLS):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        return self.fc2(h)  # logits; sigmoid applied in loss


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def per_env_auroc(preds: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """AUROC per-env using rank statistic (Mann-Whitney U formulation)."""
    aurocs = []
    for k in range(preds.shape[0]):
        p = preds[k]
        y = labels[k]
        pos = (y > 0.5)
        neg = ~pos
        n_pos = pos.sum()
        n_neg = neg.sum()
        if n_pos == 0 or n_neg == 0:
            aurocs.append(float("nan"))
            continue
        order = np.argsort(p)
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(p) + 1)
        sum_pos_ranks = ranks[pos].sum()
        u_pos = sum_pos_ranks - n_pos * (n_pos + 1) / 2
        aurocs.append(float(u_pos / (n_pos * n_neg)))
    return np.array(aurocs)


def per_env_iou(pred_bin: np.ndarray, labels: np.ndarray) -> np.ndarray:
    ious = []
    for k in range(pred_bin.shape[0]):
        p = pred_bin[k].astype(bool)
        y = labels[k].astype(bool)
        inter = (p & y).sum()
        union = (p | y).sum()
        if union == 0:
            ious.append(float("nan"))
            continue
        ious.append(float(inter / union))
    return np.array(ious)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sigs, envs = load_data(args.run_dir)
    Xtr, Ytr, _, _ = build_xy(sigs, envs, "train")
    Xve, Yve, ve_env_ids, ve_policy_ids = build_xy(sigs, envs, "verification")
    Xfa, Yfa, fa_env_ids, fa_policy_ids = build_xy(sigs, envs, "falsifier")
    print(f"train: {Xtr.shape[0]} examples; verification: {Xve.shape[0]}; falsifier: {Xfa.shape[0]}", flush=True)
    if Xtr.shape[0] == 0:
        sys.exit("no training data")

    # Standardize features using train-only stats.
    mu = Xtr.mean(axis=0); sd = Xtr.std(axis=0); sd[sd < 1e-6] = 1
    Xtr = (Xtr - mu) / sd
    Xve = (Xve - mu) / sd
    Xfa = (Xfa - mu) / sd

    model = InversionMlp(in_dim=Xtr.shape[1], hidden=args.hidden, out_dim=Ytr.shape[1])
    n_params = count_params(model)
    print(f"params: {n_params}", flush=True)
    if n_params > 20000:
        sys.exit(f"FATAL: model has {n_params} params, exceeds v0 limit of 20000")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(Ytr)
    n = Xt.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total = 0.0; nb = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            logit = model(Xt[idx])
            loss = loss_fn(logit, Yt[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); nb += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  epoch {epoch+1:02d}: loss={total/nb:.4f}", flush=True)

    # Held-out eval on verification + falsifier splits.
    def evaluate(X, Y, env_ids, policy_ids, split_name):
        with torch.no_grad():
            logits = model(torch.from_numpy(X)).numpy()
            probs = 1 / (1 + np.exp(-logits))
        aurocs = per_env_auroc(probs, Y)
        thresh = args.iou_threshold
        ious = per_env_iou((probs >= thresh).astype(np.float32), Y)
        mean_auroc = float(np.nanmean(aurocs))
        mean_iou = float(np.nanmean(ious))
        # success_per_pair: AUROC ≥ 0.80 AND IoU ≥ 0.40
        success_mask = (aurocs >= 0.80) & (ious >= 0.40)
        n_success = int(success_mask.sum())
        n_total = len(aurocs)
        return {
            "split": split_name,
            "mean_auroc": mean_auroc,
            "mean_iou": mean_iou,
            "success_count": n_success,
            "total_count": n_total,
            "success_rate": n_success / n_total if n_total else 0,
            "auroc_threshold": 0.80,
            "iou_threshold": 0.40,
            "per_env": [
                {"env_id": e, "policy_id": p, "auroc": float(a) if not math.isnan(a) else None,
                 "iou": float(i) if not math.isnan(i) else None,
                 "success": bool(s)}
                for e, p, a, i, s in zip(env_ids, policy_ids, aurocs, ious, success_mask)
            ],
        }

    results = {
        "schema": "pvnp-phase1-attacker-inversion-v0",
        "attacker": "A_inv_small",
        "seed": args.seed,
        "params_count": n_params,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "grid": GRID,
        "feature_dim": int(Xtr.shape[1]),
        "feature_standardization": {"mu": mu.tolist(), "sd": sd.tolist()},
        "evaluations": [
            evaluate(Xve, Yve, ve_env_ids, ve_policy_ids, "verification"),
            evaluate(Xfa, Yfa, fa_env_ids, fa_policy_ids, "falsifier"),
        ],
    }
    out_path = args.run_dir / "attacker_inversion_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for ev in results["evaluations"]:
        print(f"  {ev['split']}: auroc={ev['mean_auroc']:.3f} iou={ev['mean_iou']:.3f} success={ev['success_count']}/{ev['total_count']}", flush=True)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
