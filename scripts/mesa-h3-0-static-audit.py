#!/usr/bin/env python
"""H3.0-a static body/invariant audit.

This is admission-only tooling for
docs/mesa/H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md.

The default family is intentionally synthetic:

- hidden body x: high-dimensional continuous Gaussian state;
- shadow sigma: a small set of linear projections plus nonlinear certificate
  cues;
- invariant I(x): signs of pair-product latent projections.

The audit asks whether the shadow can determine I while still failing to
reconstruct x under linear/PCA/MLP probes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def git_info(root: Path) -> dict[str, Any]:
    def run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return ""

    commit = run(["git", "rev-parse", "HEAD"])
    dirty = bool(run(["git", "status", "--porcelain"]))
    return {"commit": commit, "dirty": dirty}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run H3.0-a static body/invariant audit.")
    p.add_argument("--samples", type=int, default=8192)
    p.add_argument("--body-dim", type=int, default=96)
    p.add_argument("--linear-dim", type=int, default=12)
    p.add_argument("--invariant-bits", type=int, default=4)
    p.add_argument("--shadow-noise", type=float, default=0.05)
    p.add_argument("--cue-noise", type=float, default=0.04)
    p.add_argument("--cue-strength", type=float, default=4.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    p.add_argument("--seed", type=int, default=20260623)
    p.add_argument("--ridge-alpha", type=float, default=1e-3)
    p.add_argument("--body-epochs", type=int, default=120)
    p.add_argument("--inv-epochs", type=int, default=120)
    p.add_argument("--null-epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--torch-threads", type=int, default=1)
    p.add_argument("--out-md", default="docs/mesa/H3_0_BODY_INVARIANT_STATIC_AUDIT_RESULTS.md")
    p.add_argument("--out-json", default="results/mesa/h3/body_invariant_static_audit/summary.json")
    return p.parse_args()


def make_unit_columns(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    raw = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(raw)
    return q[:, :cols]


@dataclass
class Dataset:
    body: np.ndarray
    shadow: np.ndarray
    invariant: np.ndarray
    feature_names: list[str]
    params: dict[str, Any]


def make_dataset(args: argparse.Namespace) -> Dataset:
    rng = np.random.default_rng(args.seed)
    n = args.samples
    d = args.body_dim
    m = args.linear_dim
    k = args.invariant_bits
    if m + 2 * k > d:
        raise ValueError("linear_dim + 2*invariant_bits must be <= body_dim")

    body = rng.normal(size=(n, d)).astype(np.float32)

    linear_basis = make_unit_columns(rng, d, m)
    linear_shadow = body @ linear_basis
    linear_shadow += rng.normal(scale=args.shadow_noise, size=linear_shadow.shape)

    pair_basis = make_unit_columns(rng, d, 2 * k)
    a = body @ pair_basis[:, 0::2]
    b = body @ pair_basis[:, 1::2]
    pair_product = a * b
    invariant = (pair_product >= 0).astype(np.float32)

    # The certificate cue is a noisy continuous measurement of the sign-bearing
    # product. It is not the invariant label, but it is meant to determine it.
    certificate_cue = np.tanh(args.cue_strength * pair_product)
    certificate_cue += rng.normal(scale=args.cue_noise, size=certificate_cue.shape)

    nuisance = rng.normal(size=(n, max(2, k))).astype(np.float32)
    shadow = np.concatenate([linear_shadow, certificate_cue, nuisance], axis=1).astype(np.float32)
    feature_names = (
        [f"linear_shadow_{i}" for i in range(m)]
        + [f"certificate_cue_{i}" for i in range(k)]
        + [f"nuisance_{i}" for i in range(nuisance.shape[1])]
    )
    params = {
        "samples": n,
        "body_dim": d,
        "linear_dim": m,
        "invariant_bits": k,
        "shadow_noise": args.shadow_noise,
        "cue_noise": args.cue_noise,
        "cue_strength": args.cue_strength,
        "seed": args.seed,
    }
    return Dataset(body=body, shadow=shadow, invariant=invariant, feature_names=feature_names, params=params)


def split_indices(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed ^ 0xA51CE)
    idx = rng.permutation(n)
    n_train = int(round(n * frac))
    return idx[:n_train], idx[n_train:]


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = train.mean(axis=0, keepdims=True)
    sig = train.std(axis=0, keepdims=True)
    sig[sig < 1e-8] = 1.0
    return (train - mu) / sig, (test - mu) / sig, mu, sig


def participation_ratio(x: np.ndarray) -> float:
    centered = x - x.mean(axis=0, keepdims=True)
    cov = centered.T @ centered / max(1, centered.shape[0] - 1)
    evals = np.linalg.eigvalsh(cov)
    evals = np.maximum(evals, 0)
    denom = float(np.sum(evals * evals))
    if denom <= 0:
        return 0.0
    return float((np.sum(evals) ** 2) / denom)


def fve(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    sse = float(np.sum((y_true - y_pred) ** 2))
    centered = y_true - y_true.mean(axis=0, keepdims=True)
    sst = float(np.sum(centered ** 2))
    if sst <= 0:
        return 0.0
    return float(1.0 - sse / sst)


def sign_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true >= 0) == (y_pred >= 0)))


def ridge_fit_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alpha: float) -> np.ndarray:
    y_mu = ytr.mean(axis=0, keepdims=True)
    yc = ytr - y_mu
    xtx = xtr.T @ xtr
    eye = np.eye(xtx.shape[0], dtype=np.float64)
    w = np.linalg.solve(xtx + alpha * eye, xtr.T @ yc)
    return xte @ w + y_mu


def pca_ridge_predict(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, alpha: float, dims: int) -> np.ndarray:
    u, s, vh = np.linalg.svd(xtr, full_matrices=False)
    keep = max(1, min(dims, vh.shape[0]))
    ztr = u[:, :keep] * s[:keep]
    zte = xte @ vh[:keep, :].T
    return ridge_fit_predict(ztr, ytr, zte, alpha)


class MlpRegressor(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MlpClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def torch_batches(n: int, batch_size: int, generator: torch.Generator) -> list[torch.Tensor]:
    perm = torch.randperm(n, generator=generator)
    return [perm[i : i + batch_size] for i in range(0, n, batch_size)]


def train_body_mlp(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    torch.manual_seed(args.seed + 17)
    gen = torch.Generator().manual_seed(args.seed + 18)
    model = MlpRegressor(xtr.shape[1], ytr.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(args.body_epochs):
        for b in torch_batches(len(xtr), args.batch_size, gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        return model(torch.tensor(xte, dtype=torch.float32)).cpu().numpy()


def train_inv_mlp(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    args: argparse.Namespace,
    epochs: int,
    seed_offset: int,
) -> np.ndarray:
    torch.manual_seed(args.seed + seed_offset)
    gen = torch.Generator().manual_seed(args.seed + seed_offset + 1)
    model = MlpClassifier(xtr.shape[1], ytr.shape[1], args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    xt = torch.tensor(xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.float32)
    for _ in range(epochs):
        for b in torch_batches(len(xtr), args.batch_size, gen):
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xt[b]), yt[b])
            loss.backward()
            opt.step()
    with torch.no_grad():
        logits = model(torch.tensor(xte, dtype=torch.float32))
        return (torch.sigmoid(logits).cpu().numpy() >= 0.5).astype(np.float32)


def bit_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def exact_packet_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.all(y_true == y_pred, axis=1)))


def majority_bit_acc(ytr: np.ndarray, yte: np.ndarray) -> float:
    maj = (ytr.mean(axis=0, keepdims=True) >= 0.5).astype(np.float32)
    pred = np.repeat(maj, repeats=yte.shape[0], axis=0)
    return bit_acc(yte, pred)


def main() -> None:
    args = parse_args()
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
    np.set_printoptions(precision=4, suppress=True)
    root = repo_root()
    started = dt.datetime.now(dt.UTC)

    data = make_dataset(args)
    train_idx, test_idx = split_indices(args.samples, args.train_frac, args.seed)
    xtr_raw, xte_raw = data.shadow[train_idx], data.shadow[test_idx]
    body_tr, body_te = data.body[train_idx], data.body[test_idx]
    inv_tr, inv_te = data.invariant[train_idx], data.invariant[test_idx]

    xtr, xte, _, _ = standardize(xtr_raw, xte_raw)

    pr_body = participation_ratio(data.body)
    ridge_pred = ridge_fit_predict(xtr, body_tr, xte, args.ridge_alpha)
    pca_pred = pca_ridge_predict(xtr, body_tr, xte, args.ridge_alpha, dims=min(8, xtr.shape[1]))
    mlp_pred = train_body_mlp(xtr, body_tr, xte, args)

    body_probes = {
        "ridge": {"fve": fve(body_te, ridge_pred), "coord_sign_acc": sign_acc(body_te, ridge_pred)},
        "pca_ridge": {"fve": fve(body_te, pca_pred), "coord_sign_acc": sign_acc(body_te, pca_pred)},
        "mlp": {"fve": fve(body_te, mlp_pred), "coord_sign_acc": sign_acc(body_te, mlp_pred)},
    }
    best_body_fve = max(v["fve"] for v in body_probes.values())
    best_coord_acc = max(v["coord_sign_acc"] for v in body_probes.values())

    inv_pred = train_inv_mlp(xtr, inv_tr, xte, args, args.inv_epochs, seed_offset=100)
    inv_bit_acc = bit_acc(inv_te, inv_pred)
    inv_exact_acc = exact_packet_acc(inv_te, inv_pred)
    majority_acc = majority_bit_acc(inv_tr, inv_te)

    rng = np.random.default_rng(args.seed ^ 0x5EED)
    shuffled = inv_tr.copy()
    for j in range(shuffled.shape[1]):
        rng.shuffle(shuffled[:, j])
    null_pred = train_inv_mlp(xtr, shuffled, xte, args, args.null_epochs, seed_offset=200)
    shuffled_null_acc = bit_acc(inv_te, null_pred)
    null_acc = max(majority_acc, shuffled_null_acc)

    gates = {
        "gate1_body_resistance": bool(pr_body >= 20 and best_body_fve <= 0.80 and best_coord_acc <= 0.75),
        "gate2_invariant_determination": bool(inv_bit_acc >= 0.95 and null_acc <= majority_acc + 0.05),
    }
    if not gates["gate1_body_resistance"]:
        branch = "H3_0_BODY_VOID"
    elif not gates["gate2_invariant_determination"]:
        branch = "H3_0_INVARIANT_VOID"
    else:
        branch = "H3_0_A_STATIC_ADMITTED"

    completed = dt.datetime.now(dt.UTC)
    summary = {
        "spec": "docs/mesa/H3_0_BODY_RESISTANT_INVARIANT_CONTROL_ADMISSION_SPEC.md",
        "script": "scripts/mesa-h3-0-static-audit.py",
        "startedAt": started.isoformat(),
        "completedAt": completed.isoformat(),
        "elapsedSec": (completed - started).total_seconds(),
        "git": git_info(root),
        "params": data.params | {
            "train_frac": args.train_frac,
            "ridge_alpha": args.ridge_alpha,
            "body_epochs": args.body_epochs,
            "inv_epochs": args.inv_epochs,
            "null_epochs": args.null_epochs,
            "batch_size": args.batch_size,
            "hidden": args.hidden,
            "torch_threads": args.torch_threads,
        },
        "feature_schema": {
            "allowed": data.feature_names,
            "forbidden": ["body_coordinate_*", "invariant_label_*", "seed", "terminal_outcome", "cell_id"],
        },
        "split": {"train": int(len(train_idx)), "test": int(len(test_idx))},
        "metrics": {
            "PR_body": pr_body,
            "body_probes": body_probes,
            "best_body_fve": best_body_fve,
            "best_coord_sign_acc": best_coord_acc,
            "I_acc_bit": inv_bit_acc,
            "I_acc_exact_packet": inv_exact_acc,
            "I_null_majority": majority_acc,
            "I_null_shuffled_mlp": shuffled_null_acc,
            "I_null_acc": null_acc,
        },
        "gates": gates,
        "branch": branch,
        "note": "H3.0-a static admission only; not final H3_0_ADMITTED and not a pantheon result.",
    }

    out_json = root / args.out_json
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def r(x: float, n: int = 4) -> str:
        return f"{x:.{n}f}"

    md = [
        "# H3.0-a Body / Invariant Static Audit Results",
        "",
        f"Status: **`{branch}`**. Generated {completed.isoformat()} by `scripts/mesa-h3-0-static-audit.py`.",
        "",
        "This is an admission-only static audit. It tests H3.0 Gates 1-2: body resistance plus invariant determination. It does not run the H3.0 fixed-control task or any H3.1 controller.",
        "",
        "## Configuration",
        "",
        "| parameter | value |",
        "| --- | ---: |",
        *[f"| `{k}` | `{v}` |" for k, v in summary["params"].items()],
        "",
        "## Body Resistance",
        "",
        f"- `PR_body`: **{r(pr_body)}** (gate >= 20)",
        f"- best body `FVE_body_from_shadow`: **{r(best_body_fve)}** (gate <= 0.80)",
        f"- best coordinate sign accuracy: **{r(best_coord_acc)}** (gate <= 0.75)",
        "",
        "| probe | FVE | coord_sign_acc |",
        "| --- | ---: | ---: |",
        *[f"| {name} | {r(vals['fve'])} | {r(vals['coord_sign_acc'])} |" for name, vals in body_probes.items()],
        "",
        "## Invariant Determination",
        "",
        f"- MLP bit accuracy: **{r(inv_bit_acc)}** (gate >= 0.95)",
        f"- exact packet accuracy across all bits: **{r(inv_exact_acc)}**",
        f"- majority null bit accuracy: **{r(majority_acc)}**",
        f"- shuffled-label MLP null bit accuracy: **{r(shuffled_null_acc)}**",
        f"- primary null accuracy: **{r(null_acc)}** (gate <= majority + 0.05)",
        "",
        "## Gates",
        "",
        f"- Gate 1 body resistance: **{gates['gate1_body_resistance']}**",
        f"- Gate 2 invariant determination: **{gates['gate2_invariant_determination']}**",
        "",
        f"Decision: **`{branch}`**.",
        "",
        "## Feature Schema",
        "",
        "Allowed shadow features:",
        "",
        ", ".join(f"`{name}`" for name in data.feature_names),
        "",
        "Forbidden feature classes: `body_coordinate_*`, `invariant_label_*`, `seed`, `terminal_outcome`, `cell_id`.",
        "",
        f"JSON receipt: `{args.out_json}`.",
        "",
        "## Interpretation",
        "",
        "The default synthetic family is a continuous/high-entropy body carrying discrete pair-product certificate bits. The shadow carries low-dimensional linear projections plus noisy nonlinear certificate cues. Passing this audit means the Gate 1 / Gate 2 crux is at least constructible on the static family; H3.0-b must still prove a control-sufficient singleton dilemma before any H3.1 council test.",
        "",
    ]
    out_md = root / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")

    print(f"H3.0-a static audit -> {branch}")
    print(f"  PR_body={r(pr_body)} best_FVE={r(best_body_fve)} coord_acc={r(best_coord_acc)}")
    print(f"  I_acc={r(inv_bit_acc)} exact={r(inv_exact_acc)} null={r(null_acc)}")
    print(f"  wrote {args.out_md}")


if __name__ == "__main__":
    main()
