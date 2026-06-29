"""Train a frozen LEARNED presider (field) head for the non-in-vitro replication.

Replication step 1 (learned presider heads): replace the frozen *analytic* field
with a frozen *learned* approximation of the same presider, to show the unified
bound is not an artifact of the exact analytic gradient.

The field is a function of position only, so the head takes the raw observation
(obs0..obs5 = x0, x1, and the 4 signature samples) — NOT the field-derived
features (fa_*), which would be circular. Target = the analytic field action.
Output is a coordForward-format JSON loadable by the same path as the other heads.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.mesa.ns1_shutdown_task import ShutdownForkedFieldEnv, NS1_ADMITTED_CELLS, NS1_CELL_DEFS

OBS_FEATURES = ["obs0", "obs1", "obs2", "obs3", "obs4", "obs5"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default=",".join(NS1_ADMITTED_CELLS))
    ap.add_argument("--samples-per-cell", type=int, default=6000)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def collect(cells: list[str], n_per_cell: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cell in cells:
        env = ShutdownForkedFieldEnv({"shutdownEnabled": False})
        env.reset(10_000, NS1_CELL_DEFS[cell])
        half = float(env.cfg["arenaHalfWidth"])
        for _ in range(n_per_cell):
            # sample a position uniformly in the arena; field is position-determined
            env.x = [float(rng.uniform(-half, half)), float(rng.uniform(-half, half))]
            obs = env.observe()
            obs_vec = env.obs_vector(obs)  # [x0, x1, s0, s1, s2, s3]
            fa = env.field_action()  # analytic field action (the presider target)
            xs.append(obs_vec[:6])
            ys.append([float(fa[0]), float(fa[1])])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


class MLP(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(6, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    return float(np.mean(np.sum(a * b, axis=1) / (na * nb)))


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    X, Y = collect(cells, args.samples_per_cell, args.seed)
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std
    n = len(Xn)
    idx = np.random.default_rng(args.seed).permutation(n)
    tr, va = idx[: int(0.9 * n)], idx[int(0.9 * n):]
    Xt = torch.tensor(Xn[tr], dtype=torch.float32)
    Yt = torch.tensor(Y[tr], dtype=torch.float32)
    Xv = torch.tensor(Xn[va], dtype=torch.float32)
    Yv = torch.tensor(Y[va], dtype=torch.float32)

    model = MLP(args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for ep in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), Yt)
        loss.backward()
        opt.step()
        if (ep + 1) % 50 == 0 or ep == 0:
            model.eval()
            with torch.no_grad():
                pv = model(Xv).numpy()
            cos = cosine(pv, Yv.numpy())
            print(f"epoch {ep + 1}/{args.epochs} train_mse={loss.item():.5f} val_cos={cos:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pv = model(Xv).numpy()
    val_cos = cosine(pv, Yv.numpy())
    val_mse = float(np.mean((pv - Yv.numpy()) ** 2))

    def layer(lin: nn.Linear, act: str) -> dict:
        return {"weight": lin.weight.detach().numpy().tolist(),
                "bias": lin.bias.detach().numpy().tolist(), "activation": act}

    payload = {
        "kind": "ns_field_head",
        "head": "learned_presider",
        "note": "Frozen learned approximation of the analytic field presider (replication step 1).",
        "input_features": OBS_FEATURES,
        "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "layers": [layer(model.l1, "tanh"), layer(model.l2, "tanh"), layer(model.l3, "linear")],
        "train": {"cells": cells, "samples": int(n), "epochs": args.epochs,
                  "val_cosine_to_analytic_field": round(val_cos, 5), "val_mse": round(val_mse, 6)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(f"wrote {args.out} | val cosine-to-analytic-field={val_cos:.4f} val_mse={val_mse:.5f}", flush=True)
    # admission: the learned presider must actually approximate the analytic field
    print(f"FIELD_HEAD_{'OK' if val_cos >= 0.97 else 'WEAK'} (cosine {val_cos:.4f}, threshold 0.97)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
