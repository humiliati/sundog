"""Behavior-cloning trainer for the v0 small_mlp policy.

Trains a small MLP (≤20k params) to imitate the hand-coded safe-seeker
on the train split. Inputs are observation features computed from probe
samples; output is a normalized action direction. Exports weights to JSON
so the JS harness can run the policy without a Python runtime at evaluation
time.

Usage:
    python -m training.pvnp.train_mlp_policy \
        --run-dir results/pvnp/phase1-toy-verifier-v0 \
        --epochs 12 --seed 0
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


PROBE_OFFSET_R = 0.04
GOAL_CENTER = (0.925, 0.5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=REPO_ROOT / "results" / "pvnp" / "phase1-toy-verifier-v0")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


class SmallPolicyMlp(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 128, out_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return torch.tanh(self.fc3(h))  # bounded [-1, 1] direction


def featurize(pos, probes):
    """Map (position, probes) → 8-d feature vector.

    Features: [pos.x, pos.y, center, +x, -x, +y, -y, goal_dx_norm]
    """
    by_offset = {(round(p["dx"] / PROBE_OFFSET_R), round(p["dy"] / PROBE_OFFSET_R)): p["value"] for p in probes}
    center = by_offset.get((0, 0), 0.0)
    plus_x = by_offset.get((1, 0), center)
    minus_x = by_offset.get((-1, 0), center)
    plus_y = by_offset.get((0, 1), center)
    minus_y = by_offset.get((0, -1), center)
    goal_dx = GOAL_CENTER[0] - pos["x"]
    goal_dy = GOAL_CENTER[1] - pos["y"]
    goal_norm = math.hypot(goal_dx, goal_dy) or 1.0
    return np.array([
        pos["x"], pos["y"], center, plus_x, minus_x, plus_y, minus_y,
        goal_dx / goal_norm,
    ], dtype=np.float32)


def load_bc_dataset(run_dir: Path, source_policy_id: str = "hc_safe_seeker_v0"):
    traces_path = run_dir / "traces.jsonl"
    if not traces_path.exists():
        sys.exit(f"missing {traces_path}")

    features = []
    targets = []
    with traces_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row["policy_id"] != source_policy_id:
                continue
            if row["split"] != "train":
                continue
            positions = row["positions"]
            probes = row["probes"]
            actions = row["actions"]
            T = len(actions)
            for t in range(T):
                feat = featurize(positions[t], probes[t])
                action = actions[t]
                # Normalize action to unit direction; target is the direction,
                # the MLP scales to max_action_step at deploy time.
                norm = math.hypot(action["dx"], action["dy"]) or 1.0
                tgt = np.array([action["dx"] / norm, action["dy"] / norm], dtype=np.float32)
                features.append(feat)
                targets.append(tgt)
    X = np.stack(features)
    Y = np.stack(targets)
    return X, Y


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def export_weights_to_json(model: nn.Module) -> dict:
    """Serialize weights as plain JSON so the JS forward pass can load them."""
    out = {"layers": []}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            out["layers"].append({
                "name": name,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "weight": module.weight.detach().cpu().numpy().tolist(),
                "bias": module.bias.detach().cpu().numpy().tolist(),
            })
    return out


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    X, Y = load_bc_dataset(args.run_dir)
    print(f"loaded BC dataset: X={X.shape}, Y={Y.shape}", flush=True)

    model = SmallPolicyMlp(in_dim=X.shape[1], hidden=args.hidden, out_dim=Y.shape[1])
    n_params = count_params(model)
    print(f"params: {n_params}", flush=True)
    if n_params > 20000:
        sys.exit(f"FATAL: model has {n_params} params, exceeds v0 limit of 20000")

    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    n = Xt.shape[0]
    for epoch in range(args.epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            pred = model(Xt[idx])
            loss = loss_fn(pred, Yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"epoch {epoch+1:02d}: loss={total_loss/n_batches:.5f}", flush=True)

    # Export weights as JSON.
    weights = export_weights_to_json(model)
    weights["meta"] = {
        "schema": "pvnp-phase1-mlp-policy-v0",
        "policy_id": "small_mlp_seed_0",
        "params_count": n_params,
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden": args.hidden,
        "in_dim": int(X.shape[1]),
        "out_dim": int(Y.shape[1]),
        "trained_on": "train_split_hc_safe_seeker_v0_bc",
    }
    out_path = args.run_dir / "mlp_policy_small_mlp_seed_0.json"
    out_path.write_text(json.dumps(weights), encoding="utf-8")
    print(f"wrote {out_path} ({n_params} params)")


if __name__ == "__main__":
    main()
