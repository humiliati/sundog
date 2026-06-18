"""H1.2d trainer: supervised guard + M-Adapter, RL-trained arbiter.

Builds on the H1.2 coordinator contract and keeps the same JSON output schema as
train_h1_arbiter.py, but replaces arbiter final training with an RL-style
objective over offline rollouts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

try:
    from sklearn.metrics import roc_auc_score
except ModuleNotFoundError:  # pragma: no cover
    roc_auc_score = None

COORD_FORMAT = "mesa-coordinator-json-v1"
SPEC_PATH = "docs/mesa/H1_2D_RL_ARBITER_SPEC.md"
GUARD_CALIBRATION_PENALTY = 1.0
MIN_ANCHOR_WEIGHT = 0.02


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


def to_coord_json(
    model: MLP,
    kind: str,
    feature_names: list[str],
    mean,
    std,
    head: str,
    role_cap: float | None,
    cap_mode: str | None = None,
    role_caps: dict | None = None,
):
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
        "normalization": {
            "mean": [round(float(v), 6) for v in mean],
            "std": [round(float(v), 6) for v in std],
        },
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


def _clip_to_unit_interval(x: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(x, 0.0), 1.0)


def build_role_rewards(data: np.ndarray, col: dict[str, int], args) -> np.ndarray:
    alpha = _clip_to_unit_interval(data[:, col["alpha_star"]])
    risk = _clip_to_unit_interval(data[:, col["risk"]])
    basin = _clip_to_unit_interval(data[:, col["roll_basin_captured"]])
    trust = _clip_to_unit_interval((alpha - 0.5) * 2.0)
    corruption = _clip_to_unit_interval((0.5 - alpha) * 2.0)

    r_field = (
        args.lambda_align * alpha
        + args.lambda_field * trust
        - args.lambda_uncert * corruption * (1.0 - risk)
    )
    r_reward = (
        args.lambda_align * (1.0 - alpha)
        - args.lambda_proxy * basin
        - 0.25 * args.lambda_uncert * trust
    )
    r_guard = (
        args.lambda_guard * risk
        - 0.5 * args.lambda_align * (1.0 - risk)
        - 0.1 * args.lambda_field * trust
    )
    return np.stack([r_field, r_reward, r_guard], axis=1).astype(np.float32)


def discounted_weights(data: np.ndarray, col: dict[str, int], gamma: float) -> np.ndarray:
    t = np.maximum(data[:, col["t"]], 0.0).astype(np.float32)
    return np.power(np.float32(gamma), t, dtype=np.float32)


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    denom = mask.sum()
    if float(denom.detach().cpu().item()) <= 0.0:
        return torch.tensor(0.0, device=values.device)
    return (values * mask).sum() / denom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec-path", default=SPEC_PATH)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--warmup-epochs", type=int, default=20)
    ap.add_argument("--hidden-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--rl-lr", type=float, default=2e-3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--ce-anchor", type=float, default=0.15)
    ap.add_argument("--role-cap", type=float, default=0.70)
    ap.add_argument(
        "--cap-mode", default="symmetric", choices=["symmetric", "reward-asymmetric"]
    )
    ap.add_argument("--field-cap", type=float, default=1.00)
    ap.add_argument("--reward-cap", type=float, default=0.50)
    ap.add_argument("--guard-cap", type=float, default=0.70)
    ap.add_argument("--lambda-align", type=float, default=1.0)
    ap.add_argument("--lambda-basin", type=float, default=1.0)
    ap.add_argument("--lambda-proxy", type=float, default=0.6)
    ap.add_argument("--lambda-guard", type=float, default=0.2)
    ap.add_argument("--lambda-field", type=float, default=0.5)
    ap.add_argument("--lambda-uncert", type=float, default=0.5)
    ap.add_argument("--lambda-smooth", type=float, default=0.05)
    ap.add_argument("--lambda-relief-contrast", type=float, default=0.0)
    ap.add_argument("--relief-margin", type=float, default=0.12)
    ap.add_argument("--trust-clean-fdgrad-min", type=float, default=0.08)
    ap.add_argument("--trust-clean-disagree-max", type=float, default=0.6)
    ap.add_argument("--trust-clean-risk-max", type=float, default=0.35)
    ap.add_argument("--trust-noise-fdgrad-max", type=float, default=0.04)
    ap.add_argument("--trust-noise-disagree-min", type=float, default=0.9)
    ap.add_argument("--trust-noise-risk-min", type=float, default=0.55)
    args = ap.parse_args()
    if args.cap_mode == "reward-asymmetric":
        role_caps = {
            "field": args.field_cap,
            "reward": args.reward_cap,
            "guard": args.guard_cap,
        }
    else:
        role_caps = {
            "field": args.role_cap,
            "reward": args.role_cap,
            "guard": args.role_cap,
        }

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

    # ---- P_Guard -------------------------------------------------------------
    guard = MLP(len(feats), args.hidden_size, 1)
    pos = float(y_guard_tr.sum())
    neg = float(len(y_guard_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(guard.parameters(), lr=args.lr)
    ytr = torch.tensor(y_guard_tr, device=device).unsqueeze(1)
    for _ in range(args.warmup_epochs):
        opt.zero_grad()
        logit = guard(Xtr_t)
        loss = bce(logit, ytr)
        prob = torch.sigmoid(logit)
        loss = loss + GUARD_CALIBRATION_PENALTY * (prob.mean() - ytr.mean()) ** 2
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

    # ---- P_Arbiter (warmup CE + RL objective) --------------------------------
    arb_feats = feats + ["guard_risk"]
    Xtr_a = np.concatenate([Xtr, guard_risk_tr.reshape(-1, 1)], axis=1).astype(np.float32)
    Xva_a = np.concatenate([Xva, guard_risk_va.reshape(-1, 1)], axis=1).astype(np.float32)
    mean_a, std_a = normalize_stats(Xtr_a)
    Xtr_a_t = torch.tensor((Xtr_a - mean_a) / std_a, device=device)
    Xva_a_t = torch.tensor((Xva_a - mean_a) / std_a, device=device)
    clean_mask_tr = torch.tensor(
        (
            (train[:, tc["fd_grad_norm"]] >= args.trust_clean_fdgrad_min)
            & (train[:, tc["disagree_l2"]] <= args.trust_clean_disagree_max)
            & (guard_risk_tr <= args.trust_clean_risk_max)
        ).astype(np.float32),
        device=device,
    )
    clean_mask_va = torch.tensor(
        (
            (val[:, vc["fd_grad_norm"]] >= args.trust_clean_fdgrad_min)
            & (val[:, vc["disagree_l2"]] <= args.trust_clean_disagree_max)
            & (guard_risk_va <= args.trust_clean_risk_max)
        ).astype(np.float32),
        device=device,
    )
    noise_mask_tr = torch.tensor(
        (
            (train[:, tc["fd_grad_norm"]] <= args.trust_noise_fdgrad_max)
            | (train[:, tc["disagree_l2"]] >= args.trust_noise_disagree_min)
            | (guard_risk_tr >= args.trust_noise_risk_min)
        ).astype(np.float32),
        device=device,
    )
    noise_mask_va = torch.tensor(
        (
            (val[:, vc["fd_grad_norm"]] <= args.trust_noise_fdgrad_max)
            | (val[:, vc["disagree_l2"]] >= args.trust_noise_disagree_min)
            | (guard_risk_va >= args.trust_noise_risk_min)
        ).astype(np.float32),
        device=device,
    )

    tgt_tr_np = np.stack(
        [train[:, tc[c]] for c in ("tgt_w_field", "tgt_w_reward", "tgt_w_guard")], axis=1
    ).astype(np.float32)
    tgt_va_np = np.stack(
        [val[:, vc[c]] for c in ("tgt_w_field", "tgt_w_reward", "tgt_w_guard")], axis=1
    ).astype(np.float32)
    tgt_tr = torch.tensor(tgt_tr_np, device=device)
    tgt_va = torch.tensor(tgt_va_np, device=device)

    arbiter = MLP(len(arb_feats), args.hidden_size, 3)
    opt = torch.optim.Adam(arbiter.parameters(), lr=args.lr)
    for _ in range(args.warmup_epochs):
        opt.zero_grad()
        logsm = torch.log_softmax(arbiter(Xtr_a_t), dim=1)
        loss = -(tgt_tr * logsm).sum(dim=1).mean()
        loss.backward()
        opt.step()

    role_reward_tr = torch.tensor(build_role_rewards(train, tc, args), device=device)
    role_reward_va = torch.tensor(build_role_rewards(val, vc, args), device=device)
    disc_tr = torch.tensor(discounted_weights(train, tc, args.gamma), device=device)
    disc_va = torch.tensor(discounted_weights(val, vc, args.gamma), device=device)
    t_tr = torch.tensor(train[:, tc["t"]], device=device)
    t_va = torch.tensor(val[:, vc["t"]], device=device)
    smooth_mask_tr = (t_tr > 0).float()
    smooth_mask_va = (t_va > 0).float()

    opt = torch.optim.Adam(arbiter.parameters(), lr=args.rl_lr)
    rl_history: list[float] = []
    for epoch in range(args.epochs):
        opt.zero_grad()
        logits = arbiter(Xtr_a_t)
        probs = torch.softmax(logits, dim=1)
        exp_r = (probs * role_reward_tr).sum(dim=1)
        discounted_obj = (exp_r * disc_tr).mean()

        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
        reward_cap_pen = torch.relu(probs[:, 1] - args.reward_cap).pow(2).mean()
        field_clean = masked_mean(probs[:, 0], clean_mask_tr)
        field_noise = masked_mean(probs[:, 0], noise_mask_tr)
        relief_delta = field_clean - field_noise
        relief_shortfall = torch.relu(
            torch.tensor(args.relief_margin, device=device) - relief_delta
        )
        smooth = torch.tensor(0.0, device=device)
        if len(probs) > 1:
            mask = smooth_mask_tr[1:] > 0
            if bool(mask.any()):
                diffs = probs[1:][mask] - probs[:-1][mask]
                smooth = diffs.pow(2).sum(dim=1).mean()

        # anchor to supervised targets to avoid early collapse
        logsm = torch.log_softmax(logits, dim=1)
        ce_anchor = -(tgt_tr * logsm).sum(dim=1).mean()
        anchor_w = max(
            MIN_ANCHOR_WEIGHT, args.ce_anchor * (1.0 - epoch / max(args.epochs, 1))
        )

        loss = (
            -discounted_obj
            - args.entropy_coef * entropy
            + args.lambda_basin * reward_cap_pen
            + args.lambda_smooth * smooth
            + args.lambda_relief_contrast * relief_shortfall
            + anchor_w * ce_anchor
        )
        loss.backward()
        nn.utils.clip_grad_norm_(arbiter.parameters(), max_norm=1.0)
        opt.step()
        rl_history.append(float(discounted_obj.detach().cpu().item()))

    with torch.no_grad():
        val_logits = arbiter(Xva_a_t)
        val_probs = torch.softmax(val_logits, dim=1)
        arb_ce_va = float(-(tgt_va * torch.log_softmax(val_logits, dim=1)).sum(dim=1).mean())
        rl_obj_va = float(((val_probs * role_reward_va).sum(dim=1) * disc_va).mean())
        entropy_va = float((-(val_probs * torch.log(val_probs + 1e-8)).sum(dim=1)).mean())
        reward_cap_pen_va = float(torch.relu(val_probs[:, 1] - args.reward_cap).pow(2).mean())
        field_clean_va = float(masked_mean(val_probs[:, 0], clean_mask_va))
        field_noise_va = float(masked_mean(val_probs[:, 0], noise_mask_va))
        field_delta_va = field_clean_va - field_noise_va
        relief_shortfall_va = max(0.0, args.relief_margin - field_delta_va)
        smooth_va = 0.0
        if len(val_probs) > 1:
            vmask = smooth_mask_va[1:] > 0
            if bool(vmask.any()):
                vdiffs = val_probs[1:][vmask] - val_probs[:-1][vmask]
                smooth_va = float(vdiffs.pow(2).sum(dim=1).mean())

    # ---- M-Adapter -----------------------------------------------------------
    madapt_tgt_tr = torch.tensor(
        np.stack(
            [train[:, tc[c]] for c in ("tgt_madapter_field", "tgt_madapter_reward")],
            axis=1,
        ).astype(np.float32),
        device=device,
    )
    madapt_tgt_va = torch.tensor(
        np.stack(
            [val[:, vc[c]] for c in ("tgt_madapter_field", "tgt_madapter_reward")],
            axis=1,
        ).astype(np.float32),
        device=device,
    )
    budget = param_count(guard) + param_count(arbiter)
    h = args.hidden_size
    for cand in range(4, 256):
        if abs(param_count(MLP(len(feats), cand, 2)) - budget) <= abs(
            param_count(MLP(len(feats), h, 2)) - budget
        ):
            h = cand
    m_adapter = MLP(len(feats), h, 2)
    opt = torch.optim.Adam(m_adapter.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    for _ in range(args.warmup_epochs * 2):
        opt.zero_grad()
        loss = mse(m_adapter(Xtr_t), madapt_tgt_tr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        madapt_mse_va = float(mse(m_adapter(Xva_t), madapt_tgt_va))

    guard_p, arb_p, madapt_p = (
        param_count(guard),
        param_count(arbiter),
        param_count(m_adapter),
    )
    council_p = guard_p + arb_p
    budget_ratio = madapt_p / council_p

    (out / "p_guard.json").write_text(
        json.dumps(to_coord_json(guard, "guard", feats, mean, std, head="sigmoid", role_cap=None))
        + "\n",
        encoding="utf-8",
    )
    arbiter_json = json.dumps(
        to_coord_json(
            arbiter,
            "arbiter",
            arb_feats,
            mean_a,
            std_a,
            head="softmax_cap",
            role_cap=args.role_cap,
            cap_mode=args.cap_mode,
            role_caps=role_caps,
        )
    ) + "\n"
    (out / "p_council_arbiter_rl.json").write_text(arbiter_json, encoding="utf-8")
    # compatibility path for existing harness defaults
    (out / "p_council_arbiter.json").write_text(arbiter_json, encoding="utf-8")
    (out / "m_adapter.json").write_text(
        json.dumps(
            to_coord_json(
                m_adapter, "m_adapter", feats, mean, std, head="linear_blend", role_cap=None
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = {
        "spec": args.spec_path,
        "seed": args.seed,
        "epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "hidden_size": args.hidden_size,
        "cap_mode": args.cap_mode,
        "role_caps": role_caps,
        "rl_objective": {
            "lambda_align": args.lambda_align,
            "lambda_basin": args.lambda_basin,
            "lambda_proxy": args.lambda_proxy,
            "lambda_guard": args.lambda_guard,
            "lambda_field": args.lambda_field,
            "lambda_uncert": args.lambda_uncert,
            "lambda_smooth": args.lambda_smooth,
            "entropy_coef": args.entropy_coef,
            "gamma": args.gamma,
            "ce_anchor": args.ce_anchor,
            "lambda_relief_contrast": args.lambda_relief_contrast,
            "relief_margin": args.relief_margin,
        },
        "trust_partition": {
            "clean": {
                "fd_grad_norm_min": args.trust_clean_fdgrad_min,
                "disagree_l2_max": args.trust_clean_disagree_max,
                "guard_risk_max": args.trust_clean_risk_max,
                "support_train_rows": int(clean_mask_tr.sum().cpu().item()),
                "support_val_rows": int(clean_mask_va.sum().cpu().item()),
            },
            "noise": {
                "fd_grad_norm_max": args.trust_noise_fdgrad_max,
                "disagree_l2_min": args.trust_noise_disagree_min,
                "guard_risk_min": args.trust_noise_risk_min,
                "support_train_rows": int(noise_mask_tr.sum().cpu().item()),
                "support_val_rows": int(noise_mask_va.sum().cpu().item()),
            },
        },
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
            "guard_auc": round(guard_auc, 4) if not math.isnan(guard_auc) else None,
            "guard_calibration_err": round(guard_cal, 4),
            "guard_pos_frac": round(float(y_guard_va.mean()), 4),
            "arbiter_ce_to_target": round(arb_ce_va, 4),
            "arbiter_rl_objective": round(rl_obj_va, 6),
            "arbiter_entropy": round(entropy_va, 6),
            "arbiter_reward_cap_penalty": round(reward_cap_pen_va, 8),
            "arbiter_smooth_penalty": round(smooth_va, 8),
            "field_relief_clean_proxy": round(field_clean_va, 6),
            "field_relief_noise_proxy": round(field_noise_va, 6),
            "field_relief_proxy_delta": round(field_delta_va, 6),
            "field_relief_proxy_shortfall": round(relief_shortfall_va, 6),
            "m_adapter_mse": round(madapt_mse_va, 6),
        },
        "rl_train_trace": {
            "objective_first": round(rl_history[0], 6) if rl_history else None,
            "objective_last": round(rl_history[-1], 6) if rl_history else None,
            "objective_max": round(max(rl_history), 6) if rl_history else None,
        },
        "leakage_check": {
            "guard_inputs": feats,
            "arbiter_inputs": arb_feats,
            "m_adapter_inputs": feats,
            "no_privileged_in_features": all(
                f not in schema["labels_privileged"] for f in arb_feats if f != "guard_risk"
            ),
        },
    }
    (out / "train-report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("H1.2d RL trainer done.")
    print(
        f"  params guard={guard_p} arbiter={arb_p} council={council_p} | "
        f"m_adapter={madapt_p} (h={h}) ratio={budget_ratio:.3f} within5%={report['params']['budget_within_5pct']}"
    )
    print(
        f"  val: guard_auc={report['val']['guard_auc']} guard_cal={guard_cal:.4f} "
        f"arbiter_ce={arb_ce_va:.4f} rl_obj={rl_obj_va:.6f} m_adapter_mse={madapt_mse_va:.6f}"
    )


if __name__ == "__main__":
    main()
