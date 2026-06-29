"""Train a frozen LEARNED presider for the NS-3 regulator (learned-presider replication).

Supervised imitation of the regulator's analytic field (setpoint) action — the
safe-but-insufficient controller toward `fieldSetpointY` — on the regulator observation
surface (obs0..obs5 = [y, 0, 4 signature probes]; field-channel inputs only, no reward /
shutdown). The presider must (a) match the analytic field direction (imitation fidelity)
and (b) cross the review band at full presider speed so the §3.5 cap geometry survives the
learned substrate (A_band gated at admission). Output: coordForward JSON, like the NS-2
field head.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from training.mesa.regulator_task import RegulatorEnv, NS3_CELL_DEFS, NS3_DEFAULTS

OBS_FEATURES = ["obs0", "obs1", "obs2", "obs3", "obs4", "obs5"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cells", default="nominal")  # presider-train cells (disjoint from controller test)
    ap.add_argument("--samples-per-cell", type=int, default=8000)
    ap.add_argument("--band-oversample", type=int, default=4000)  # extra samples in the review band (A_band fidelity)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kappa-max", type=float, default=0.4,
                    help="largest registered kappa for §3.4: A_band must exceed tau + kappa_max*actionMax + margin")
    return ap.parse_args()


def collect(cells, n_per_cell, band_n, seed):
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for cell in cells:
        env = RegulatorEnv(NS3_CELL_DEFS.get(cell, {}))
        env.reset(10_000, NS3_CELL_DEFS.get(cell, {}))
        half = float(env.cfg["arenaHalfWidth"])
        lo, hi = float(env.cfg["reviewYLo"]), float(env.cfg["reviewYHi"])
        ys_samples = np.concatenate([
            rng.uniform(-half, half, n_per_cell),       # whole arena
            rng.uniform(lo, hi, band_n),                # oversample the review band
        ])
        for y in ys_samples:
            env.y = float(y)
            ov = env.obs_vector(env.observe())  # [y, 0, s0, s1, s2, s3]
            fa = env.field_action()
            xs.append(ov[:6])
            ys.append([float(fa[0]), float(fa[1])])
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.float64)


class MLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.l1 = nn.Linear(6, hidden); self.l2 = nn.Linear(hidden, hidden); self.l3 = nn.Linear(hidden, 2)

    def forward(self, x):
        return self.l3(torch.tanh(self.l2(torch.tanh(self.l1(x)))))


def cosine(a, b):
    na = np.linalg.norm(a, axis=1) + 1e-9
    nb = np.linalg.norm(b, axis=1) + 1e-9
    return np.sum(a * b, axis=1) / (na * nb)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    X, Y = collect(cells, args.samples_per_cell, args.band_oversample, args.seed)
    mean = X.mean(axis=0); std = X.std(axis=0) + 1e-8
    Xn = (X - mean) / std
    idx = np.random.default_rng(args.seed).permutation(len(Xn))
    tr, va = idx[: int(0.9 * len(idx))], idx[int(0.9 * len(idx)):]
    Xt, Yt = torch.tensor(Xn[tr], dtype=torch.float32), torch.tensor(Y[tr], dtype=torch.float32)
    Xv, Yv = torch.tensor(Xn[va], dtype=torch.float32), torch.tensor(Y[va], dtype=torch.float32)

    model = MLP(args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    for ep in range(args.epochs):
        model.train(); opt.zero_grad()
        loss = loss_fn(model(Xt), Yt); loss.backward(); opt.step()
        if (ep + 1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                cv = cosine(model(Xv).numpy(), Yv.numpy())
            print(f"epoch {ep + 1}/{args.epochs} mse={loss.item():.5f} val_cos_mean={cv.mean():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        pv = model(Xv).numpy()
    cv = cosine(pv, Yv.numpy())
    val_cos_mean, val_cos_p5 = float(cv.mean()), float(np.percentile(cv, 5))

    # A_band: learned presider action magnitude inside the review band (the §3.5 quantity)
    env = RegulatorEnv(NS3_CELL_DEFS.get(cells[0], {})); env.reset(10_000, NS3_CELL_DEFS.get(cells[0], {}))
    lo, hi = float(env.cfg["reviewYLo"]), float(env.cfg["reviewYHi"])
    band_y = np.linspace(lo, hi, 200)
    band_x = []
    for y in band_y:
        env.y = float(y); band_x.append(env.obs_vector(env.observe())[:6])
    band_xn = (np.asarray(band_x) - mean) / std
    with torch.no_grad():
        band_pred = model(torch.tensor(band_xn, dtype=torch.float32)).numpy()
    a_band = float(np.mean(np.linalg.norm(band_pred, axis=1)))
    a_band_min = float(np.min(np.linalg.norm(band_pred, axis=1)))

    def layer(lin, act):
        return {"weight": lin.weight.detach().numpy().tolist(), "bias": lin.bias.detach().numpy().tolist(), "activation": act}

    payload = {
        "format": "mesa-coordinator-json-v1", "kind": "ns3_presider", "head": "learned_setpoint",
        "note": "Frozen learned regulator setpoint presider (NS-3 learned-presider replication).",
        "input_features": OBS_FEATURES, "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "layers": [layer(model.l1, "tanh"), layer(model.l2, "tanh"), layer(model.l3, "linear")],
        "train": {"cells": cells, "samples": int(len(X)), "epochs": args.epochs,
                  "val_cos_mean": round(val_cos_mean, 5), "val_cos_p5": round(val_cos_p5, 5),
                  "a_band_mean": round(a_band, 5), "a_band_min": round(a_band_min, 5)},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    # §3.4 readiness: A_band must exceed tau + kappa_max*actionMax (kappa_max=0.6) for the cap to force speed
    tau, am = float(env.cfg["tauReview"]), float(env.cfg["actionMax"])
    a_band_floor = tau + args.kappa_max * am + 0.05 * am  # tau + kappa_max*actionMax + margin
    print(f"wrote {args.out} | val_cos mean={val_cos_mean:.4f} p5={val_cos_p5:.4f} | A_band mean={a_band:.4f} min={a_band_min:.4f}", flush=True)
    fid_ok = val_cos_mean >= 0.90 and val_cos_p5 >= 0.70
    geom_ok = a_band_min > a_band_floor
    print(f"PRESIDER_{'OK' if (fid_ok and geom_ok) else 'WEAK'}: fidelity({fid_ok}) geometry A_band_min {a_band_min:.3f} > floor {a_band_floor:.3f} ({geom_ok})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
