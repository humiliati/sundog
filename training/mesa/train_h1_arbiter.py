"""H1.2a coordinator trainer: P_Guard, P_Arbiter, and the M-Adapter monolith.

Spec: docs/mesa/H1_2_SMALL_BAKEOFF_SPEC.md §5. Reads the Node-built coordinator
dataset (train.csv / val.csv / feature-schema.json), trains three small MLPs,
and serializes each as ``mesa-coordinator-json-v1`` so the Node eval harness
(scripts/mesa-h1-pantheon-eval.mjs) can run them closed-loop against the
canonical ShadowFieldEnv.

Leakage contract (enforced here): model inputs (X) are drawn ONLY from
``inference_features``; for the arbiter, plus the guard's own predicted risk.
The privileged labels (true gradient, basin geometry, alpha*, target weights,
rollout outcomes) are targets/diagnostics only and never enter X.

Targets (privileged-best-mix regression, locked 2026-06-18):
  * P_Guard   -> ``risk`` (class-balanced BCE on risk>0.5 + calibration penalty)
  * P_Arbiter -> CE to the capped 3-way target weights [field,reward,guard]
  * M-Adapter -> MSE to the uncapped direction-optimal coeffs [alpha*, 1-alpha*]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:  # pragma: no cover
    roc_auc_score = None

COORD_FORMAT = "mesa-coordinator-json-v1"


def load_csv(path: Path) -> tuple[list[str], np.ndarray, dict[str, int]]:
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows = [r for r in reader if r]
    idx = {name: i for i, name in enumerate(header)}

    def to_float(v: str) -> float:
        return float(v) if v not in ("", None) else 0.0

    numeric_cols = [c for c in header if c not in ("split", "cell", "behavior")]
    data = np.array(
        [[to_float(r[idx[c]]) for c in numeric_cols] for r in rows], dtype=np.float32
    )
    col = {c: j for j, c in enumerate(numeric_cols)}
    return numeric_cols, data, col


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(d, hidden))
            layers.append(nn.Tanh())
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def normalize_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)


def to_coord_json(model: MLP, kind: str, feature_names: list[str], mean, std, head: str,
                  role_cap: float | None, cap_mode: str | None = None, role_caps: dict | None = None):
    layers = []
    linears = [m for m in model.net if isinstance(m, nn.Linear)]
    n_lin = len(linears)
    for i, m in enumerate(linears):
        layers.append(
            {
                "weight": m.weight.detach().cpu().numpy().round(6).tolist(),
                "bias": m.bias.detach().cpu().numpy().round(6).tolist(),
                "activation": "tanh" if i < n_lin - 1 else "linear",
            }
        )
    payload = {
        "format": COORD_FORMAT,
        "kind": kind,
        "input_features": feature_names,
        "normalization": {"mean": [round(float(v), 6) for v in mean], "std": [round(float(v), 6) for v in std]},
        "layers": layers,
        "head": head,
    }
    if role_cap is not None:
        payload["role_cap"] = role_cap
    if cap_mode is not None:
        payload["cap_mode"] = cap_mode
    if role_caps is not None:
        payload["role_caps"] = role_caps
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--role-cap", type=float, default=0.70)
    ap.add_argument("--cap-mode", default="symmetric", choices=["symmetric", "reward-asymmetric"])
    ap.add_argument("--field-cap", type=float, default=1.00)
    ap.add_argument("--reward-cap", type=float, default=0.50)
    ap.add_argument("--guard-cap", type=float, default=0.70)
    args = ap.parse_args()
    if args.cap_mode == "reward-asymmetric":
        role_caps = {"field": args.field_cap, "reward": args.reward_cap, "guard": args.guard_cap}
    else:
        role_caps = {"field": args.role_cap, "reward": args.role_cap, "guard": args.role_cap}

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ds = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    schema = json.loads((ds / "feature-schema.json").read_text(encoding="utf-8"))
    feats = schema["inference_features"]

    _, train, tc = load_csv(ds / "train.csv")
    _, val, vc = load_csv(ds / "val.csv")

    def cols(data, colmap, names):
        return np.stack([data[:, colmap[n]] for n in names], axis=1).astype(np.float32)

    Xtr = cols(train, tc, feats)
    Xva = cols(val, vc, feats)
    mean, std = normalize_stats(Xtr)
    Xtr_n = (Xtr - mean) / std
    Xva_n = (Xva - mean) / std

    risk_tr = train[:, tc["risk"]].astype(np.float32)
    risk_va = val[:, vc["risk"]].astype(np.float32)
    y_guard_tr = (risk_tr > 0.5).astype(np.float32)
    y_guard_va = (risk_va > 0.5).astype(np.float32)

    device = "cpu"
    Xtr_t = torch.tensor(Xtr_n, device=device)
    Xva_t = torch.tensor(Xva_n, device=device)

    # ---- P_Guard: class-balanced BCE on risk>0.5 + calibration penalty -------
    guard = MLP(len(feats), args.hidden_size, 1)
    pos = float(y_guard_tr.sum())
    neg = float(len(y_guard_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(guard.parameters(), lr=args.lr)
    ytr = torch.tensor(y_guard_tr, device=device).unsqueeze(1)
    for _ in range(args.epochs):
        opt.zero_grad()
        logit = guard(Xtr_t)
        loss = bce(logit, ytr)
        prob = torch.sigmoid(logit)
        loss = loss + 1.0 * (prob.mean() - ytr.mean()) ** 2  # calibration nudge
        loss.backward()
        opt.step()
    with torch.no_grad():
        guard_risk_tr = torch.sigmoid(guard(Xtr_t)).cpu().numpy().reshape(-1)
        guard_risk_va = torch.sigmoid(guard(Xva_t)).cpu().numpy().reshape(-1)
    guard_auc = (
        float(roc_auc_score(y_guard_va, guard_risk_va))
        if roc_auc_score is not None and 0 < y_guard_va.sum() < len(y_guard_va)
        else float("nan")
    )
    guard_cal = float(abs(guard_risk_va.mean() - y_guard_va.mean()))

    # ---- P_Arbiter: CE to capped 3-way target weights, X=[feats, guard_risk] -
    arb_feats = feats + ["guard_risk"]
    Xtr_a = np.concatenate([Xtr, guard_risk_tr.reshape(-1, 1)], axis=1).astype(np.float32)
    Xva_a = np.concatenate([Xva, guard_risk_va.reshape(-1, 1)], axis=1).astype(np.float32)
    mean_a, std_a = normalize_stats(Xtr_a)
    Xtr_a_t = torch.tensor((Xtr_a - mean_a) / std_a, device=device)
    Xva_a_t = torch.tensor((Xva_a - mean_a) / std_a, device=device)
    tgt_tr = torch.tensor(
        np.stack([train[:, tc[c]] for c in ("tgt_w_field", "tgt_w_reward", "tgt_w_guard")], axis=1).astype(np.float32),
        device=device,
    )
    tgt_va = torch.tensor(
        np.stack([val[:, vc[c]] for c in ("tgt_w_field", "tgt_w_reward", "tgt_w_guard")], axis=1).astype(np.float32),
        device=device,
    )
    arbiter = MLP(len(arb_feats), args.hidden_size, 3)
    opt = torch.optim.Adam(arbiter.parameters(), lr=args.lr)
    for _ in range(args.epochs * 2):
        opt.zero_grad()
        logsm = torch.log_softmax(arbiter(Xtr_a_t), dim=1)
        loss = -(tgt_tr * logsm).sum(dim=1).mean()  # cross-entropy to target dist
        loss.backward()
        opt.step()
    with torch.no_grad():
        arb_ce_va = float(-(tgt_va * torch.log_softmax(arbiter(Xva_a_t), dim=1)).sum(dim=1).mean())

    # ---- M-Adapter: MSE to direction-optimal coeffs [alpha, 1-alpha] ---------
    madapt_tgt_tr = torch.tensor(
        np.stack([train[:, tc[c]] for c in ("tgt_madapter_field", "tgt_madapter_reward")], axis=1).astype(np.float32),
        device=device,
    )
    madapt_tgt_va = torch.tensor(
        np.stack([val[:, vc[c]] for c in ("tgt_madapter_field", "tgt_madapter_reward")], axis=1).astype(np.float32),
        device=device,
    )
    # equal incremental budget: size M-Adapter hidden to match guard+arbiter params
    budget = param_count(guard) + param_count(arbiter)
    h = args.hidden_size
    for cand in range(4, 256):
        if abs(param_count(MLP(len(feats), cand, 2)) - budget) <= abs(param_count(MLP(len(feats), h, 2)) - budget):
            h = cand
    m_adapter = MLP(len(feats), h, 2)
    opt = torch.optim.Adam(m_adapter.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    for _ in range(args.epochs * 2):
        opt.zero_grad()
        loss = mse(m_adapter(Xtr_t), madapt_tgt_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        madapt_mse_va = float(mse(m_adapter(Xva_t), madapt_tgt_va))

    guard_p, arb_p, madapt_p = param_count(guard), param_count(arbiter), param_count(m_adapter)
    council_p = guard_p + arb_p
    budget_ratio = madapt_p / council_p

    # ---- serialize -----------------------------------------------------------
    (out / "p_guard.json").write_text(
        json.dumps(to_coord_json(guard, "guard", feats, mean, std, head="sigmoid", role_cap=None)) + "\n",
        encoding="utf-8",
    )
    (out / "p_council_arbiter.json").write_text(
        json.dumps(to_coord_json(arbiter, "arbiter", arb_feats, mean_a, std_a, head="softmax_cap",
                                 role_cap=args.role_cap, cap_mode=args.cap_mode, role_caps=role_caps)) + "\n",
        encoding="utf-8",
    )
    (out / "m_adapter.json").write_text(
        json.dumps(to_coord_json(m_adapter, "m_adapter", feats, mean, std, head="linear_blend", role_cap=None)) + "\n",
        encoding="utf-8",
    )
    report = {
        "spec": "docs/mesa/H1_2_SMALL_BAKEOFF_SPEC.md §5",
        "seed": args.seed,
        "epochs": args.epochs,
        "hidden_size": args.hidden_size,
        "feature_mode": schema.get("feature_mode", "base"),
        "trust_feature_audit": schema.get("trust_feature_audit"),
        "cap_mode": args.cap_mode,
        "role_caps": role_caps,
        "params": {
            "guard": guard_p,
            "arbiter": arb_p,
            "council_total": council_p,
            "m_adapter": madapt_p,
            "m_adapter_hidden": h,
            "budget_ratio_m_over_council": round(budget_ratio, 4),
            "budget_within_5pct": bool(abs(budget_ratio - 1.0) <= 0.05),
        },
        "val": {
            "guard_auc": round(guard_auc, 4) if guard_auc == guard_auc else None,
            "guard_calibration_err": round(guard_cal, 4),
            "guard_pos_frac": round(float(y_guard_va.mean()), 4),
            "arbiter_ce": round(arb_ce_va, 4),
            "m_adapter_mse": round(madapt_mse_va, 6),
        },
        "leakage_check": {
            "guard_inputs": feats,
            "arbiter_inputs": arb_feats,
            "m_adapter_inputs": feats,
            "no_privileged_in_features": all(f not in schema["labels_privileged"] for f in arb_feats if f != "guard_risk"),
        },
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("H1.2a trainer done.")
    print(
        f"  params guard={guard_p} arbiter={arb_p} council={council_p} | "
        f"m_adapter={madapt_p} (h={h}) ratio={budget_ratio:.3f} within5%={report['params']['budget_within_5pct']}"
    )
    print(
        f"  val: guard_auc={report['val']['guard_auc']} guard_cal={guard_cal:.4f} "
        f"arbiter_ce={arb_ce_va:.4f} m_adapter_mse={madapt_mse_va:.6f}"
    )


if __name__ == "__main__":
    main()
