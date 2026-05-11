"""Behavior-cloning trainer for the Phase 2 HC-Signature imitation gate."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:  # pragma: no cover - keeps the CLI usable on bare Python.
    def tqdm(iterable, **_kwargs):  # type: ignore
        return iterable

from training.mesa.hc_bc_dataset import DEFAULT_MANIFEST, HCBcDataset
from training.mesa.policy import (
    MesaMlpPolicy,
    count_parameters,
    policy_config_for_tier,
    policy_to_json_dict,
    seed_everything,
    write_policy_json,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "results" / "mesa" / "phase2-matched-capacity"
FAMILY = "L-Signature"
VARIANT = "signature_bc_from_hc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Phase 2 behavior-cloned signature policy.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tier", default="Small", choices=["Small", "Medium", "Large"])
    parser.add_argument("--sensor-tier", default="local-probe-field")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-every", type=int, default=25)
    parser.add_argument("--successful-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--progress", action="store_true", help="show tqdm progress even when stderr is not a TTY")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def evaluate_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for obs, action in loader:
            obs = obs.to(device)
            action = action.to(device)
            prediction = model(obs)
            loss = criterion(prediction, action)
            batch_size = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
    model.train()
    return total_loss / max(total_items, 1)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    seed_everything(args.seed)
    device = resolve_device(args.device)

    train_dataset = HCBcDataset(
        args.manifest,
        split="train",
        sensor_tier=args.sensor_tier,
        successful_only=args.successful_only,
        seed_base=args.seed,
        cache_dir=args.out / "cache",
    )
    val_dataset = HCBcDataset(
        args.manifest,
        split="val",
        sensor_tier=args.sensor_tier,
        successful_only=args.successful_only,
        seed_base=args.seed,
    )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    config = policy_config_for_tier(args.tier, action_scale=1.0)
    model = MesaMlpPolicy(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history: list[dict[str, Any]] = []

    show_progress = args.progress or sys.stderr.isatty()
    for epoch in tqdm(
        range(1, args.epochs + 1),
        desc=f"bc {args.tier.lower()}",
        unit="epoch",
        disable=not show_progress,
        leave=False,
    ):
        model.train()
        total_loss = 0.0
        total_items = 0
        for obs, action in train_loader:
            obs = obs.to(device)
            action = action.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(obs)
            loss = criterion(prediction, action)
            loss.backward()
            optimizer.step()
            batch_size = int(obs.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        train_loss = total_loss / max(total_items, 1)
        if epoch == 1 or epoch % args.val_every == 0 or epoch == args.epochs:
            val_loss = evaluate_loss(model, val_loader, criterion, device)
            history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out / "checkpoints"
    policies_dir = args.out / "policies"
    logs_dir = args.out / "logs"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    policies_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    slug = f"signature_bc_from_hc_{args.tier.lower()}_seed_{args.seed}"
    checkpoint_path = checkpoints_dir / f"{slug}.pt"
    policy_json_path = policies_dir / f"{slug}.policy.json"
    history_path = logs_dir / f"{slug}_history.csv"

    checkpoint = {
        "family": FAMILY,
        "variant": VARIANT,
        "tier": args.tier,
        "seed": args.seed,
        "sensor_tier": args.sensor_tier,
        "policy_config": asdict(config),
        "model_state_dict": model.state_dict(),
        "obs_mean": train_dataset.obs_mean.tolist(),
        "obs_std": train_dataset.obs_std.tolist(),
        "bc_dataset": train_dataset.manifest_block,
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "parameter_count": count_parameters(model),
            "device": str(device),
        },
    }
    torch.save(checkpoint, checkpoint_path)

    policy_payload = policy_to_json_dict(
        model.cpu(),
        family=FAMILY,
        variant=VARIANT,
        obs_mean=train_dataset.obs_mean,
        obs_std=train_dataset.obs_std,
        metadata={
            "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "source_manifest": train_dataset.manifest_block["source_manifest"],
            "config_hash": train_dataset.manifest_block["config_hash"],
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
        },
    )
    write_policy_json(policy_json_path, policy_payload)

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    run_summary = {
        "phase": "phase2-matched-capacity",
        "family": FAMILY,
        "variant": VARIANT,
        "tier": args.tier,
        "seed": args.seed,
        "sensor_tier": args.sensor_tier,
        "checkpoint_path": str(checkpoint_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "policy_json_path": str(policy_json_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "history_path": str(history_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "bc_dataset": train_dataset.manifest_block,
        "training": checkpoint["training"],
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
    }
    (args.out / "manifest.json").write_text(json.dumps(run_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (args.out / "training-runs.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "family",
            "variant",
            "tier",
            "seed",
            "sensor_tier",
            "parameter_count",
            "epochs",
            "best_epoch",
            "best_val_loss",
            "checkpoint_path",
            "policy_json_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            "family": FAMILY,
            "variant": VARIANT,
            "tier": args.tier,
            "seed": args.seed,
            "sensor_tier": args.sensor_tier,
            "parameter_count": training["parameter_count"],
            "epochs": args.epochs,
            "best_epoch": training["best_epoch"],
            "best_val_loss": training["best_val_loss"],
            "checkpoint_path": run_summary["checkpoint_path"],
            "policy_json_path": run_summary["policy_json_path"],
        })
    return run_summary


def main() -> int:
    args = parse_args()
    summary = run_training(args)
    training = summary["training"]
    print(
        "mesa bc train: "
        f"tier={summary['tier']} seed={summary['seed']} "
        f"params={training['parameter_count']} best_epoch={training['best_epoch']} "
        f"best_val_loss={training['best_val_loss']:.8f} "
        f"checkpoint={summary['checkpoint_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
