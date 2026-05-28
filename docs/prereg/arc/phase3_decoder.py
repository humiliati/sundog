#!/usr/bin/env python
"""ARC Phase 3 Blackwell sufficiency decoder.

This is the Python-primary runner for the Phase 3 Blackwell lane. It keeps the
ML implementation self-contained while the npm script remains a thin
orchestration shell.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


FEATURE_SCHEMA_VERSION = "arc-p3-feature-v1"
PROTOCOL_VERSION = "arc-p3-blackwell-protocol-v1"
RECEIPT_SCHEMA_VERSION = "arc-p3-blackwell-receipt-v1"
LEARNER_VERSION = "blackwell_task_decoder_v1"

ARMS = ["raw_grid_lowcap", "signature_palette", "signature_only", "metadata_only"]
GRID_SCORABLE_ARMS = {"raw_grid_lowcap", "signature_palette"}
SEED_SLATE = [20260528, 20260529, 20260530, 20260531, 20260601]

MAX_H = 30
MAX_W = 30
MAX_COLORS = 10
PAD_CHANNELS = 11
MAX_DEMOS = 5
MAX_TOKENS = MAX_DEMOS * 2 + 1
METADATA_DIM = 28
SIGNATURE_HASH_DIM = 4096
SIGNATURE_VECTOR_DIM = METADATA_DIM + SIGNATURE_HASH_DIM
RAW_GRID_DIM = MAX_H * MAX_W * PAD_CHANNELS

TYPE_DEMO_INPUT = 0
TYPE_DEMO_OUTPUT = 1
TYPE_QUERY_INPUT = 2
TYPE_PAD = 3

MODEL_SPEC = {
    "layers": 2,
    "d_model": 128,
    "heads": 4,
    "feedforward_dim": 256,
    "dropout": 0.10,
    "weight_init": "xavier_uniform_linear_zero_bias_embedding_normal_0_0.02_layernorm_unit",
    "optimizer": "AdamW",
    "adamw_betas": [0.9, 0.999],
    "adamw_eps": 1e-8,
    "weight_decay": 1e-4,
    "learning_rate": 3e-4,
    "lr_schedule": "constant",
    "batch_size": 16,
    "max_epochs": 400,
    "early_stop_patience": 40,
    "loss": "cell_ce + 0.25*height_ce + 0.25*width_ce",
    "seed_slate": SEED_SLATE,
}

EXPECTED_SPLIT = {
    "color_role": {
        "train": ["08ed6ac7", "0a2355a6", "2601afb7", "292dd178"],
        "validation": ["37d3e8b2"],
        "test": ["3ad05f52"],
    },
    "counting": {
        "train": ["009d5c81", "00dbd492", "025d127b", "045e512c"],
        "validation": ["05269061"],
        "test": ["05a7bcf2"],
    },
    "local_completion": {
        "train": ["03560426", "05f2a901", "0b17323b", "0e671a1a"],
        "validation": ["11e1fe23"],
        "test": ["13713586"],
    },
    "objectness": {
        "train": ["11dc524f", "150deff5", "1acc24af", "1b60fb0c"],
        "validation": ["2bee17df"],
        "test": ["3906de3d"],
    },
    "spatial_transform": {
        "train": ["00576224", "0a1d4ef5", "0b148d64", "0bb8deee"],
        "validation": ["0c9aba6e"],
        "test": ["137eaa0f"],
    },
    "symmetry": {
        "train": ["007bbfb7", "00d62c1b", "017c7c7b", "0520fde7"],
        "validation": ["0692e18c"],
        "test": ["0a938d79"],
    },
}

QUARANTINE_BY_BOUNDARY = {
    "capacity pressure (count is high-entropy)": "count_capacity_cliff",
    "capacity pressure (shape change carries structural info)": "shape_capacity_cliff",
    "full-state-only dependency (residual category)": "full_state_residual",
    "gauge-breaking ambiguity (color permutation as gauge)": "color_permutation_quotient",
    "gauge-breaking ambiguity (symmetry as gauge)": "symmetry_phase_loss",
    "non-local information (global context required to fill)": "nonlocal_completion_context",
}


@dataclass(frozen=True)
class Task:
    task_id: str
    primary_prior: str
    predicted_boundary: str
    row: dict[str, str]
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    split: str


@dataclass(frozen=True)
class Instance:
    lane: str
    instance_id: str
    task: Task
    query_index: int
    query_input: list[list[int]]
    target_output: list[list[int]]
    conditioning: list[dict[str, Any]]


class TaskDecoder(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        d_model = MODEL_SPEC["d_model"]
        self.input_proj = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(4, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=MODEL_SPEC["heads"],
            dim_feedforward=MODEL_SPEC["feedforward_dim"],
            dropout=MODEL_SPEC["dropout"],
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=MODEL_SPEC["layers"],
            enable_nested_tensor=False,
        )
        self.height_head = nn.Linear(d_model, MAX_H)
        self.width_head = nn.Linear(d_model, MAX_W)
        self.cell_head = nn.Linear(d_model, MAX_H * MAX_W * MAX_COLORS)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        features: torch.Tensor,
        token_types: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_proj(features) + self.type_embed(token_types)
        encoded = self.encoder(x, src_key_padding_mask=pad_mask)
        query_state = encoded[:, MAX_TOKENS - 1, :]
        height_logits = self.height_head(query_state)
        width_logits = self.width_head(query_state)
        cell_logits = self.cell_head(query_state).view(-1, MAX_H * MAX_W, MAX_COLORS)
        return height_logits, width_logits, cell_logits


def main() -> int:
    args = parse_args()
    started_at = iso_now()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_determinism(args.master_seed)
    git = git_state(repo_root, args.allow_dirty)
    data_dir = Path(args.data_dir).resolve()
    register_path = Path(args.register).resolve()
    assert_training_data_dir(data_dir)
    tasks, register_hash, data_hash = load_tasks(data_dir, register_path)
    split_rows = build_split_rows(tasks)
    write_csv(out_dir / "split.csv", split_rows, [
        "task_id",
        "primary_prior",
        "predicted_boundary",
        "split",
    ])

    manifest = base_manifest(
        args=args,
        started_at=started_at,
        repo_root=repo_root,
        out_dir=out_dir,
        git=git,
        data_dir=data_dir,
        register_path=register_path,
        register_hash=register_hash,
        data_hash=data_hash,
        task_count=len(tasks),
    )

    if args.dry_run:
        manifest["mode"] = "dry_run"
        manifest["completedAt"] = iso_now()
        write_empty_receipts(out_dir, manifest)
        print(f"ARC Phase 3 Blackwell dry run wrote {out_dir}")
        return 0

    seeds = [args.master_seed] if args.probe_only else SEED_SLATE
    max_epochs = args.probe_epochs if args.probe_only else args.max_epochs
    manifest["mode"] = "probe" if args.probe_only else "full"
    manifest["seedSlateEffective"] = seeds
    manifest["maxEpochsEffective"] = max_epochs
    manifest["epochsOverridden"] = max_epochs != MODEL_SPEC["max_epochs"]

    train_instances = build_lodo_instances([task for task in tasks if task.split == "train"], "train_lodo")
    validation_instances = build_lodo_instances([task for task in tasks if task.split == "validation"], "validation_lodo")
    test_lodo_instances = build_lodo_instances([task for task in tasks if task.split == "test"], "test_lodo")
    pttest_instances = build_pttest_instances([task for task in tasks if task.split == "test"])

    learning_rows: list[dict[str, Any]] = []
    per_slot_rows: list[dict[str, Any]] = []
    per_instance_any: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    selected_seed_by_arm: dict[str, int] = {}
    validation_rank_rows: list[dict[str, Any]] = []

    for arm in ARMS:
        arm_start = time.perf_counter()
        arm_candidates = []
        for seed in seeds:
            set_global_determinism(seed)
            model = TaskDecoder(input_dim_for_arm(arm))
            result = train_model(
                model=model,
                arm=arm,
                train_instances=train_instances,
                validation_instances=validation_instances,
                seed=seed,
                max_epochs=max_epochs,
                device=args.device,
            )
            learning_rows.extend(result["learning_rows"])
            eval_bundle = evaluate_model(
                model=result["model"],
                arm=arm,
                seed=seed,
                instances=[*validation_instances, *test_lodo_instances, *pttest_instances],
                device=args.device,
            )
            per_slot_rows.extend(eval_bundle["per_slot_rows"])
            per_instance_any.extend(eval_bundle["per_instance_any"])
            residual_rows.extend(eval_bundle["residual_rows"])
            candidate = {
                "arm": arm,
                "seed": seed,
                "best_epoch": result["best_epoch"],
                "validation_metric": result["best_validation_metric"],
                "validation_loss": result["best_validation_loss"],
            }
            validation_rank_rows.append(candidate)
            arm_candidates.append(candidate)
        arm_candidates.sort(key=lambda row: (-row["validation_metric"], row["validation_loss"], row["seed"], row["arm"]))
        selected_seed_by_arm[arm] = arm_candidates[0]["seed"]
        print(f"{arm}: selected seed {selected_seed_by_arm[arm]} in {time.perf_counter() - arm_start:.1f}s")

    per_task_rows = aggregate_per_task(per_instance_any, selected_seed_by_arm)
    per_prior_rows = aggregate_per_prior(per_instance_any, selected_seed_by_arm)
    scores = aggregate_scores(per_instance_any, selected_seed_by_arm, args.master_seed)
    quarantine_log = build_quarantine_log(per_task_rows)
    branch = adjudicate_branch(per_task_rows, scores, manifest["mode"])

    manifest["completedAt"] = iso_now()
    manifest["selectedSeedByArm"] = selected_seed_by_arm
    manifest["instanceCounts"] = {
        "train_lodo": len(train_instances),
        "validation_lodo": len(validation_instances),
        "test_lodo": len(test_lodo_instances),
        "pttest": len(pttest_instances),
    }

    write_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "learning_curves.csv", learning_rows, LEARNING_COLUMNS)
    write_csv(out_dir / "per_instance.csv", per_slot_rows, PER_SLOT_COLUMNS)
    write_csv(out_dir / "per_task.csv", per_task_rows, PER_TASK_COLUMNS)
    write_csv(out_dir / "per_prior.csv", per_prior_rows, PER_PRIOR_COLUMNS)
    write_csv(out_dir / "scores.csv", scores, SCORE_COLUMNS)
    write_csv(out_dir / "quarantine_log.csv", quarantine_log, QUARANTINE_COLUMNS)
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    receipt = {
        "manifest": manifest,
        "validationRank": validation_rank_rows,
        "scores": scores,
        "perTask": per_task_rows,
        "branchDecision": branch,
        "residuals": residual_rows,
    }
    write_json(out_dir / "phase3_receipt.json", receipt)
    write_markdown_summary(out_dir / "branch_adjudication.md", branch, scores, selected_seed_by_arm)
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))

    print(f"ARC Phase 3 Blackwell {manifest['mode']} run wrote {out_dir}")
    print(f"Branch decision: {branch['branch']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC Phase 3 Blackwell sufficiency decoder")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--register", required=True)
    parser.add_argument("--out", default="results/arc/phase3-blackwell-sufficiency-v1")
    parser.add_argument("--master-seed", type=int, default=20260528)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write split and empty receipt skeletons only")
    parser.add_argument("--probe-only", action="store_true", help="Train one seed for a capped timing probe")
    parser.add_argument("--probe-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=MODEL_SPEC["max_epochs"])
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def base_manifest(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    return {
        "generatedAt": kwargs["started_at"],
        "completedAt": None,
        "tool": "docs/prereg/arc/phase3_decoder.py",
        "command": [sys.executable, "docs/prereg/arc/phase3_decoder.py", *sys.argv[1:]],
        "gitCommit": kwargs["git"]["commit"],
        "gitDirty": kwargs["git"]["dirty"],
        "allowDirty": args.allow_dirty,
        "dataDir": str(kwargs["data_dir"]),
        "registerPath": str(kwargs["register_path"]),
        "outDir": str(kwargs["out_dir"]),
        "masterSeed": args.master_seed,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION,
        "learnerVersion": LEARNER_VERSION,
        "taskCount": kwargs["task_count"],
        "registerHash": kwargs["register_hash"],
        "dataDirHash": kwargs["data_hash"],
        "pythonVersion": sys.version,
        "torchVersion": torch.__version__,
        "platform": platform.platform(),
        "device": args.device,
        "modelSpec": MODEL_SPEC,
    }


def write_empty_receipts(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "learning_curves.csv", [], LEARNING_COLUMNS)
    write_csv(out_dir / "per_instance.csv", [], PER_SLOT_COLUMNS)
    write_csv(out_dir / "per_task.csv", [], PER_TASK_COLUMNS)
    write_csv(out_dir / "per_prior.csv", [], PER_PRIOR_COLUMNS)
    write_csv(out_dir / "scores.csv", [], SCORE_COLUMNS)
    write_csv(out_dir / "quarantine_log.csv", [], QUARANTINE_COLUMNS)
    write_jsonl(out_dir / "residuals.jsonl", [])
    write_json(out_dir / "phase3_receipt.json", {"manifest": manifest, "branchDecision": {"branch": "not_run"}})
    (out_dir / "branch_adjudication.md").write_text("# Branch Adjudication\n\nDry run only. No branch decision.\n", encoding="utf8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def git_state(repo_root: Path, allow_dirty: bool) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip().upper()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root, text=True).strip()
    if dirty and not allow_dirty:
        raise SystemExit("Refusing to run on a dirty worktree; commit the freeze marker first or pass --allow-dirty for smoke checks.")
    return {"commit": commit, "dirty": bool(dirty)}


def assert_training_data_dir(data_dir: Path) -> None:
    normalized = str(data_dir).replace("\\", "/").lower()
    if normalized.endswith("/evaluation"):
        raise SystemExit("Refusing to use an ARC evaluation directory as --data-dir.")
    if not (data_dir / "training").is_dir():
        raise SystemExit(f"Missing training directory under {data_dir}")


def load_tasks(data_dir: Path, register_path: Path) -> tuple[list[Task], str, str]:
    register_text = register_path.read_text(encoding="utf-8-sig")
    rows = [row for row in csv.DictReader(register_text.splitlines()) if row["status"] == "include" and row["split"] == "training"]
    tasks = []
    file_hashes = []
    split_by_task = expected_split_by_task()
    for row in rows:
        task_id = row["task_id"]
        path = data_dir / "training" / f"{task_id}.json"
        raw = path.read_text(encoding="utf-8-sig")
        file_hashes.append({"file": f"training/{task_id}.json", "sha256": sha256_text(raw)})
        parsed = json.loads(raw)
        tasks.append(Task(
            task_id=task_id,
            primary_prior=row["primary_prior"],
            predicted_boundary=row["predicted_boundary"],
            row=row,
            train=[{"index": i, "input": pair["input"], "output": pair["output"]} for i, pair in enumerate(parsed["train"])],
            test=[{"index": i, "input": pair["input"], "output": pair.get("output")} for i, pair in enumerate(parsed["test"])],
            split=split_by_task[task_id],
        ))
    validate_expected_split(tasks)
    return tasks, sha256_text(register_text), sha256_text(json.dumps(file_hashes, sort_keys=True, separators=(",", ":")))


def expected_split_by_task() -> dict[str, str]:
    out = {}
    for prior, groups in EXPECTED_SPLIT.items():
        for split, task_ids in groups.items():
            split_value = "validation" if split == "validation" else split
            for task_id in task_ids:
                out[task_id] = split_value
    return out


def validate_expected_split(tasks: list[Task]) -> None:
    actual = {task.task_id: task.split for task in tasks}
    expected = expected_split_by_task()
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(task_id for task_id in set(actual) & set(expected) if actual[task_id] != expected[task_id])
        raise SystemExit(f"Register does not match frozen Blackwell split. missing={missing} extra={extra} wrong={wrong}")
    for task in tasks:
        if len(task.train) > MAX_DEMOS:
            raise SystemExit(f"{task.task_id} has {len(task.train)} train pairs; max admitted is {MAX_DEMOS}.")
        for test in task.test:
            if task.split == "test" and test["output"] is None:
                raise SystemExit(f"{task.task_id} public-training test output is missing.")


def build_split_rows(tasks: list[Task]) -> list[dict[str, Any]]:
    return [{
        "task_id": task.task_id,
        "primary_prior": task.primary_prior,
        "predicted_boundary": task.predicted_boundary,
        "split": task.split,
    } for task in sorted(tasks, key=lambda item: item.task_id)]


def build_lodo_instances(tasks: list[Task], lane: str) -> list[Instance]:
    instances = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        for held in task.train:
            instances.append(Instance(
                lane=lane,
                instance_id=f"{lane}:{task.task_id}:{held['index']}",
                task=task,
                query_index=held["index"],
                query_input=held["input"],
                target_output=held["output"],
                conditioning=[pair for pair in task.train if pair["index"] != held["index"]],
            ))
    return instances


def build_pttest_instances(tasks: list[Task]) -> list[Instance]:
    instances = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        for test in task.test:
            instances.append(Instance(
                lane="pttest",
                instance_id=f"pttest:{task.task_id}:{test['index']}",
                task=task,
                query_index=test["index"],
                query_input=test["input"],
                target_output=test["output"],
                conditioning=task.train,
            ))
    return instances


def train_model(
    *,
    model: TaskDecoder,
    arm: str,
    train_instances: list[Instance],
    validation_instances: list[Instance],
    seed: int,
    max_epochs: int,
    device: str,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    model.to(device_obj)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=MODEL_SPEC["learning_rate"],
        betas=tuple(MODEL_SPEC["adamw_betas"]),
        eps=MODEL_SPEC["adamw_eps"],
        weight_decay=MODEL_SPEC["weight_decay"],
    )
    train_records = [make_record(instance, arm) for instance in train_instances]
    validation_records = [make_record(instance, arm) for instance in validation_instances]
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metric = -1.0
    best_loss = math.inf
    best_epoch = 0
    patience_left = MODEL_SPEC["early_stop_patience"]
    learning_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        shuffled = list(train_records)
        random.Random(seed + epoch).shuffle(shuffled)
        train_loss_values = []
        for batch in batches(shuffled, MODEL_SPEC["batch_size"]):
            tensors = collate(batch, device_obj)
            optimizer.zero_grad(set_to_none=True)
            logits = model(tensors["features"], tensors["token_types"], tensors["pad_mask"])
            loss = model_loss(logits, tensors)
            loss.backward()
            optimizer.step()
            train_loss_values.append(float(loss.detach().cpu()))
        validation = evaluate_loss_and_metric(model, arm, validation_records, device_obj)
        train_loss = sum(train_loss_values) / max(1, len(train_loss_values))
        learning_rows.append({
            "arm": arm,
            "seed": seed,
            "epoch": epoch,
            "train_loss": round_float(train_loss),
            "validation_loss": round_float(validation["loss"]),
            "validation_metric": round_float(validation["metric"]),
            "selected": False,
        })
        better = validation["metric"] > best_metric or (
            validation["metric"] == best_metric and validation["loss"] < best_loss
        )
        if better:
            best_metric = validation["metric"]
            best_loss = validation["loss"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = MODEL_SPEC["early_stop_patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    for row in learning_rows:
        if row["epoch"] == best_epoch:
            row["selected"] = True
    return {
        "model": model,
        "learning_rows": learning_rows,
        "best_epoch": best_epoch,
        "best_validation_metric": best_metric,
        "best_validation_loss": best_loss,
    }


def evaluate_loss_and_metric(model: TaskDecoder, arm: str, records: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    metric_hits = 0
    with torch.no_grad():
        for batch in batches(records, MODEL_SPEC["batch_size"]):
            tensors = collate(batch, device)
            logits = model(tensors["features"], tensors["token_types"], tensors["pad_mask"])
            losses.append(float(model_loss(logits, tensors).detach().cpu()))
            predictions = predictions_from_logits(logits)
            for record, slots in zip(batch, predictions):
                scored = score_slots(record["instance"], arm, record["target_rep"], slots, seed=record["seed"])
                if arm in GRID_SCORABLE_ARMS:
                    metric_hits += int(any(row["grid_exact"] is True for row in scored["slot_rows"]))
                else:
                    metric_hits += int(any(row["rep_exact"] is True for row in scored["slot_rows"]))
    return {
        "loss": sum(losses) / max(1, len(losses)),
        "metric": metric_hits / max(1, len(records)),
    }


def evaluate_model(
    *,
    model: TaskDecoder,
    arm: str,
    seed: int,
    instances: list[Instance],
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    device_obj = torch.device(device)
    model.eval()
    records = [make_record(instance, arm, seed=seed) for instance in instances]
    per_slot_rows = []
    per_instance_any = []
    residual_rows = []
    with torch.no_grad():
        for batch in batches(records, MODEL_SPEC["batch_size"]):
            tensors = collate(batch, device_obj)
            logits = model(tensors["features"], tensors["token_types"], tensors["pad_mask"])
            predictions = predictions_from_logits(logits)
            for record, slots in zip(batch, predictions):
                scored = score_slots(record["instance"], arm, record["target_rep"], slots, seed=seed)
                per_slot_rows.extend(scored["slot_rows"])
                per_instance_any.append(scored["any_row"])
                residual_rows.append(scored["residual"])
    return {
        "per_slot_rows": per_slot_rows,
        "per_instance_any": per_instance_any,
        "residual_rows": residual_rows,
    }


def make_record(instance: Instance, arm: str, seed: int = 0) -> dict[str, Any]:
    features = []
    token_types = []
    for pair in instance.conditioning:
        features.append(feature_vector(pair["input"], arm))
        token_types.append(TYPE_DEMO_INPUT)
        features.append(feature_vector(pair["output"], arm))
        token_types.append(TYPE_DEMO_OUTPUT)
    features.append(feature_vector(instance.query_input, arm))
    token_types.append(TYPE_QUERY_INPUT)
    while len(features) < MAX_TOKENS:
        features.append([0.0] * input_dim_for_arm(arm))
        token_types.append(TYPE_PAD)
    if len(features) > MAX_TOKENS:
        raise ValueError(f"{instance.instance_id} exceeds max token count.")
    target = grid_targets(instance.target_output)
    return {
        "instance": instance,
        "features": features,
        "token_types": token_types,
        "target": target,
        "target_rep": represent_grid(instance.target_output, arm),
        "seed": seed,
    }


def collate(records: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    features = torch.tensor([record["features"] for record in records], dtype=torch.float32, device=device)
    token_types = torch.tensor([record["token_types"] for record in records], dtype=torch.long, device=device)
    pad_mask = token_types.eq(TYPE_PAD)
    height = torch.tensor([record["target"]["height_label"] for record in records], dtype=torch.long, device=device)
    width = torch.tensor([record["target"]["width_label"] for record in records], dtype=torch.long, device=device)
    cells = torch.tensor([record["target"]["cells"] for record in records], dtype=torch.long, device=device)
    cell_mask = torch.tensor([record["target"]["cell_mask"] for record in records], dtype=torch.float32, device=device)
    return {
        "features": features,
        "token_types": token_types,
        "pad_mask": pad_mask,
        "height": height,
        "width": width,
        "cells": cells,
        "cell_mask": cell_mask,
    }


def model_loss(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor], tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    height_logits, width_logits, cell_logits = logits
    height_loss = F.cross_entropy(height_logits, tensors["height"])
    width_loss = F.cross_entropy(width_logits, tensors["width"])
    flat_loss = F.cross_entropy(
        cell_logits.reshape(-1, MAX_COLORS),
        tensors["cells"].reshape(-1),
        reduction="none",
    ).view_as(tensors["cell_mask"])
    cell_loss = (flat_loss * tensors["cell_mask"]).sum() / tensors["cell_mask"].sum().clamp_min(1.0)
    return cell_loss + 0.25 * height_loss + 0.25 * width_loss


def predictions_from_logits(logits: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> list[list[list[list[int]]]]:
    height_logits, width_logits, cell_logits = logits
    heights = height_logits.argmax(dim=1).detach().cpu().tolist()
    widths = width_logits.argmax(dim=1).detach().cpu().tolist()
    probs = cell_logits.detach().cpu()
    out = []
    for i, (h_label, w_label) in enumerate(zip(heights, widths)):
        height = int(h_label) + 1
        width = int(w_label) + 1
        values = probs[i].argmax(dim=1).tolist()
        top2 = torch.topk(probs[i], k=2, dim=1)
        grid1 = flat_to_grid(values, height, width)
        margins = (top2.values[:, 0] - top2.values[:, 1]).tolist()
        alt_values = list(values)
        active_indices = [y * MAX_W + x for y in range(height) for x in range(width)]
        if active_indices:
            flip_index = min(active_indices, key=lambda idx: (margins[idx], idx))
            alt_values[flip_index] = int(top2.indices[flip_index, 1])
        grid2 = flat_to_grid(alt_values, height, width)
        out.append([grid1, grid2])
    return out


def flat_to_grid(values: list[int], height: int, width: int) -> list[list[int]]:
    return [[int(values[y * MAX_W + x]) for x in range(width)] for y in range(height)]


def grid_targets(grid: list[list[int]]) -> dict[str, Any]:
    height = len(grid)
    width = len(grid[0])
    cells = [0] * (MAX_H * MAX_W)
    mask = [0.0] * (MAX_H * MAX_W)
    for y in range(height):
        for x in range(width):
            idx = y * MAX_W + x
            cells[idx] = grid[y][x]
            mask[idx] = 1.0
    return {
        "height_label": height - 1,
        "width_label": width - 1,
        "cells": cells,
        "cell_mask": mask,
    }


def score_slots(instance: Instance, arm: str, target_rep: dict[str, Any], slots: list[list[list[int]]], seed: int) -> dict[str, Any]:
    slot_rows = []
    residual_payload = {
        "instance_id": instance.instance_id,
        "lane": instance.lane,
        "task_id": instance.task.task_id,
        "primary_prior": instance.task.primary_prior,
        "predicted_boundary": instance.task.predicted_boundary,
        "arm": arm,
        "seed": seed,
        "target_shape": shape_label(instance.target_output),
        "predictions": [],
    }
    for rank, grid in enumerate(slots, start=1):
        pred_rep = represent_grid(grid, arm)
        grid_exact = grids_equal(grid, instance.target_output) if arm in GRID_SCORABLE_ARMS else None
        rep_exact = identity_for_arm(arm, pred_rep) == identity_for_arm(arm, target_rep)
        row = {
            "instance_id": instance.instance_id,
            "lane": instance.lane,
            "task_id": instance.task.task_id,
            "primary_prior": instance.task.primary_prior,
            "predicted_boundary": instance.task.predicted_boundary,
            "arm": arm,
            "seed": seed,
            "slot": rank,
            "grid_exact": bool_or_na(grid_exact),
            "rep_exact": bool_or_na(rep_exact),
            "shape_exact": grids_same_shape(grid, instance.target_output),
            "palette_exact": palette_label(grid) == palette_label(instance.target_output),
            "pixel_accuracy": round_float(pixel_accuracy(grid, instance.target_output)),
            "failure_label": "none" if (grid_exact is True or rep_exact is True) else failure_label(instance, arm, grid),
        }
        slot_rows.append(row)
        residual_payload["predictions"].append({
            "slot": rank,
            "shape": shape_label(grid),
            "grid_exact": grid_exact,
            "rep_exact": rep_exact,
            "failure_label": row["failure_label"],
            "grid": grid,
        })
    any_grid = None if arm not in GRID_SCORABLE_ARMS else any(row["grid_exact"] is True for row in slot_rows)
    any_rep = any(row["rep_exact"] is True for row in slot_rows)
    any_row = {
        "instance_id": instance.instance_id,
        "lane": instance.lane,
        "task_id": instance.task.task_id,
        "primary_prior": instance.task.primary_prior,
        "predicted_boundary": instance.task.predicted_boundary,
        "arm": arm,
        "seed": seed,
        "grid_exact_any_slot": bool_or_na(any_grid),
        "rep_exact_any_slot": bool_or_na(any_rep),
        "shape_exact_slot1": slot_rows[0]["shape_exact"],
        "palette_exact_slot1": slot_rows[0]["palette_exact"],
        "pixel_accuracy_best": max(float(row["pixel_accuracy"]) for row in slot_rows),
        "failure_label": "none" if (any_grid is True or any_rep is True) else slot_rows[0]["failure_label"],
    }
    residual_payload["any"] = any_row
    return {"slot_rows": slot_rows, "any_row": any_row, "residual": residual_payload}


def failure_label(instance: Instance, arm: str, grid: list[list[int]]) -> str:
    if arm == "signature_only" and instance.task.primary_prior == "color_role":
        return "structural_zero"
    if not grids_same_shape(grid, instance.target_output):
        return "detection"
    if palette_label(grid) != palette_label(instance.target_output):
        return "detection"
    return "residual"


def aggregate_per_task(rows: list[dict[str, Any]], selected_seed_by_arm: dict[str, int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if selected_seed_by_arm.get(row["arm"]) == row["seed"]:
            grouped[(row["lane"], row["task_id"], row["arm"], row["seed"])].append(row)
    out = []
    for (lane, task_id, arm, seed), items in sorted(grouped.items()):
        first = items[0]
        out.append({
            "lane": lane,
            "task_id": task_id,
            "primary_prior": first["primary_prior"],
            "predicted_boundary": first["predicted_boundary"],
            "arm": arm,
            "seed": seed,
            "instance_count": len(items),
            "grid_exact_any_rate": rate(items, "grid_exact_any_slot"),
            "rep_exact_any_rate": rate(items, "rep_exact_any_slot"),
            "shape_exact_slot1_rate": rate(items, "shape_exact_slot1"),
            "palette_exact_slot1_rate": rate(items, "palette_exact_slot1"),
            "pixel_accuracy_best_mean": mean(items, "pixel_accuracy_best"),
            "dominant_failure_label": dominant(items, "failure_label"),
        })
    return out


def aggregate_per_prior(rows: list[dict[str, Any]], selected_seed_by_arm: dict[str, int]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if selected_seed_by_arm.get(row["arm"]) == row["seed"]:
            grouped[(row["lane"], row["primary_prior"], row["predicted_boundary"], row["arm"])].append(row)
    out = []
    for (lane, prior, boundary, arm), items in sorted(grouped.items()):
        out.append({
            "lane": lane,
            "primary_prior": prior,
            "predicted_boundary": boundary,
            "arm": arm,
            "instance_count": len(items),
            "grid_exact_any_rate": rate(items, "grid_exact_any_slot"),
            "rep_exact_any_rate": rate(items, "rep_exact_any_slot"),
            "shape_exact_slot1_rate": rate(items, "shape_exact_slot1"),
            "palette_exact_slot1_rate": rate(items, "palette_exact_slot1"),
            "pixel_accuracy_best_mean": mean(items, "pixel_accuracy_best"),
        })
    return out


def aggregate_scores(rows: list[dict[str, Any]], selected_seed_by_arm: dict[str, int], master_seed: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if selected_seed_by_arm.get(row["arm"]) == row["seed"]:
            grouped[(row["lane"], row["arm"])].append(row)
    out = []
    for (lane, arm), items in sorted(grouped.items()):
        task_values = task_metric_values(items, "grid_exact_any_slot" if arm in GRID_SCORABLE_ARMS else "rep_exact_any_slot")
        ci = bootstrap_ci(task_values, master_seed)
        out.append({
            "lane": lane,
            "arm": arm,
            "selected_seed": selected_seed_by_arm.get(arm, ""),
            "task_count": len(task_values),
            "instance_count": len(items),
            "grid_exact_any_rate": rate(items, "grid_exact_any_slot"),
            "rep_exact_any_rate": rate(items, "rep_exact_any_slot"),
            "shape_exact_slot1_rate": rate(items, "shape_exact_slot1"),
            "palette_exact_slot1_rate": rate(items, "palette_exact_slot1"),
            "pixel_accuracy_best_mean": mean(items, "pixel_accuracy_best"),
            "task_bootstrap_metric_mean": round_float(sum(task_values) / max(1, len(task_values))),
            "task_bootstrap_ci_low": round_float(ci[0]),
            "task_bootstrap_ci_high": round_float(ci[1]),
        })
    return out


def build_quarantine_log(per_task_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in per_task_rows:
        if row["arm"] not in {"signature_palette", "signature_only"}:
            continue
        metric = row["grid_exact_any_rate"] if row["arm"] == "signature_palette" else row["rep_exact_any_rate"]
        if metric != "NA" and float(metric) > 0:
            continue
        out.append({
            "lane": row["lane"],
            "task_id": row["task_id"],
            "primary_prior": row["primary_prior"],
            "predicted_boundary": row["predicted_boundary"],
            "arm": row["arm"],
            "quarantine_key": QUARANTINE_BY_BOUNDARY.get(row["predicted_boundary"], "unregistered_other"),
            "evidence": row["dominant_failure_label"],
            "blocks_branch_b": QUARANTINE_BY_BOUNDARY.get(row["predicted_boundary"]) is None,
        })
    return out


def adjudicate_branch(per_task_rows: list[dict[str, Any]], scores: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode != "full":
        return {
            "branch": "not_adjudicated",
            "reason": f"{mode} run only; Branch A/B/C is reserved for the full clean five-seed receipt",
        }
    by_lane_arm = {(row["lane"], row["arm"]): row for row in scores}
    sig_lodo = by_lane_arm.get(("test_lodo", "signature_palette"))
    sig_pttest = by_lane_arm.get(("pttest", "signature_palette"))
    raw_lodo = by_lane_arm.get(("test_lodo", "raw_grid_lowcap"))
    raw_pttest = by_lane_arm.get(("pttest", "raw_grid_lowcap"))
    if not all([sig_lodo, sig_pttest, raw_lodo, raw_pttest]):
        return {"branch": "not_adjudicated", "reason": "missing required score rows"}

    sig_lodo_rate = numeric(sig_lodo["grid_exact_any_rate"])
    sig_pttest_rate = numeric(sig_pttest["grid_exact_any_rate"])
    raw_lodo_rate = numeric(raw_lodo["grid_exact_any_rate"])
    raw_pttest_rate = numeric(raw_pttest["grid_exact_any_rate"])
    sig_lodo_tasks = exact_task_count(per_task_rows, "test_lodo", "signature_palette", "grid_exact_any_rate")
    sig_pttest_tasks = exact_task_count(per_task_rows, "pttest", "signature_palette", "grid_exact_any_rate")
    clears_floor = sig_lodo_rate > 0.010 and sig_pttest_rate > 0.010 and sig_lodo_tasks >= 2 and sig_pttest_tasks >= 2
    close_to_raw = (raw_lodo_rate - sig_lodo_rate) <= 0.05 and (raw_pttest_rate - sig_pttest_rate) <= 0.05
    if clears_floor and close_to_raw:
        return {
            "branch": "Branch A",
            "reason": "signature_palette clears the non-trivial floor and is within the practical equivalence margin of raw_grid_lowcap on both held-out lanes",
        }

    easiest = [
        row for row in per_task_rows
        if row["lane"] == "test_lodo"
        and row["arm"] == "signature_palette"
        and row["primary_prior"] != "color_role"
        and row["predicted_boundary"] != "full-state-only dependency (residual category)"
    ]
    if len(easiest) >= 4 and all(numeric(row["grid_exact_any_rate"]) <= 0.010 for row in easiest):
        return {
            "branch": "Branch C",
            "reason": "signature_palette fails to clear the exact-grid floor on the easiest registered held-out subset",
        }

    return {
        "branch": "Branch B pending or inconclusive",
        "reason": "a gap remains but automatic quarantine attribution is not sufficient for a Branch B claim",
    }


def exact_task_count(rows: list[dict[str, Any]], lane: str, arm: str, field: str) -> int:
    return sum(1 for row in rows if row["lane"] == lane and row["arm"] == arm and numeric(row[field]) > 0)


def task_metric_values(items: list[dict[str, Any]], field: str) -> list[float]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in items:
        grouped[row["task_id"]].append(row)
    return [numeric(rate(rows, field)) for rows in grouped.values()]


def bootstrap_ci(values: list[float], seed: int, n: int = 10000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    rng = random.Random(seed)
    samples = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    return samples[int(0.025 * n)], samples[int(0.975 * n)]


def input_dim_for_arm(arm: str) -> int:
    return RAW_GRID_DIM if arm == "raw_grid_lowcap" else SIGNATURE_VECTOR_DIM


def feature_vector(grid: list[list[int]], arm: str) -> list[float]:
    if arm == "raw_grid_lowcap":
        return raw_grid_onehot(grid)
    rep = represent_grid(grid, arm)
    vector = [0.0] * SIGNATURE_VECTOR_DIM
    if arm in {"signature_palette", "metadata_only"}:
        vector[:METADATA_DIM] = rep["metadata"]
    if arm in {"signature_palette", "signature_only"}:
        for idx, value in rep["suffix"].items():
            vector[idx] = value
    return vector


def represent_grid(grid: list[list[int]], arm: str) -> dict[str, Any]:
    projection = project_grid_shadow(grid)
    metadata = metadata_vector(grid, projection)
    suffix = signature_suffix(projection)
    return {
        "arm": arm,
        "grid": grid,
        "shape": projection["shape"],
        "shapeLabel": f"{projection['shape'][0]}x{projection['shape'][1]}",
        "palette": projection["palette"],
        "paletteLabel": "".join(str(x) for x in projection["palette"]),
        "nonzeroPalette": projection["nonzeroPalette"],
        "nonzeroCells": projection["nonZeroCells"],
        "nonzeroComponents": projection["nonZeroComponents"],
        "density": projection["density"],
        "canonicalObjectSignature": projection["canonicalObjectSignature"],
        "localSignatureBag": projection["localSignatureBag"],
        "signatureHash": sha256_text(projection["canonicalObjectSignature"]),
        "localBagHash": sha256_text(json.dumps(projection["localSignatureBag"], separators=(",", ":"))),
        "metadata": metadata,
        "suffix": suffix,
    }


def identity_for_arm(arm: str, rep: dict[str, Any]) -> str:
    if arm == "signature_only":
        return f"{rep['signatureHash']}|{rep['localBagHash']}"
    if arm == "signature_palette":
        return f"{rep['shapeLabel']}|{rep['paletteLabel']}|{rep['nonzeroCells']}|{rep['nonzeroComponents']}|{rep['density']}|{rep['signatureHash']}|{rep['localBagHash']}"
    if arm == "metadata_only":
        return json.dumps(rep["metadata"], separators=(",", ":"))
    return json.dumps(rep["grid"], separators=(",", ":"))


def project_grid_shadow(grid: list[list[int]]) -> dict[str, Any]:
    non_zero = nonzero_cells(grid)
    local_bag = sorted(canonical_stencil(grid, cell["x"], cell["y"], 1) for cell in non_zero)
    palette = sorted(set(value for row in grid for value in row))
    return {
        "shape": [len(grid), len(grid[0])],
        "palette": palette,
        "nonzeroPalette": [value for value in palette if value != 0],
        "nonZeroCells": len(non_zero),
        "nonZeroComponents": count_components(grid),
        "density": round_float(len(non_zero) / (len(grid) * len(grid[0]))),
        "localSignatureBag": local_bag,
        "canonicalObjectSignature": canonical_object_signature(grid),
    }


def canonical_object_signature(grid: list[list[int]]) -> str:
    variants = object_variants(grid)
    return "empty" if not variants else sorted(variant["signature"] for variant in variants)[0]


def object_variants(grid: list[list[int]]) -> list[dict[str, str]]:
    cells = nonzero_cells(grid)
    if not cells:
        return []
    transforms = [
        lambda x, y: (x, y),
        lambda x, y: (y, -x),
        lambda x, y: (-x, -y),
        lambda x, y: (-y, x),
        lambda x, y: (-x, y),
        lambda x, y: (x, -y),
        lambda x, y: (y, x),
        lambda x, y: (-y, -x),
    ]
    out = []
    for transform in transforms:
        transformed = [{"x": transform(cell["x"], cell["y"])[0], "y": transform(cell["x"], cell["y"])[1], "color": cell["color"]} for cell in cells]
        min_x = min(cell["x"] for cell in transformed)
        min_y = min(cell["y"] for cell in transformed)
        normalized = sorted(
            [{"x": cell["x"] - min_x, "y": cell["y"] - min_y, "color": cell["color"]} for cell in transformed],
            key=lambda cell: (cell["y"], cell["x"], cell["color"]),
        )
        role_map = {}
        next_role = 1
        tokens = []
        for cell in normalized:
            if cell["color"] not in role_map:
                role_map[cell["color"]] = next_role
                next_role += 1
            tokens.append(f"{cell['x']}:{cell['y']}:{role_map[cell['color']]}")
        width = max(cell["x"] for cell in normalized) + 1
        height = max(cell["y"] for cell in normalized) + 1
        out.append({"signature": f"{width}x{height}|{';'.join(tokens)}"})
    return out


def canonical_stencil(grid: list[list[int]], cx: int, cy: int, radius: int) -> str:
    cells = []
    for y in range(cy - radius, cy + radius + 1):
        row = []
        for x in range(cx - radius, cx + radius + 1):
            row.append(0 if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]) else grid[y][x])
        cells.append(row)
    return sorted(role_normalize_grid(transform(cells)) for transform in stencil_transforms())[0]


def stencil_transforms():
    return [lambda g: g, rotate90, rotate180, rotate270, reflect_horizontal, reflect_vertical, transpose, anti_transpose]


def role_normalize_grid(grid: list[list[int]]) -> str:
    role_map = {0: 0}
    next_role = 1
    tokens = []
    for row in grid:
        for value in row:
            if value not in role_map:
                role_map[value] = next_role
                next_role += 1
            tokens.append(str(role_map[value]))
    return "".join(tokens)


def metadata_vector(grid: list[list[int]], projection: dict[str, Any]) -> list[float]:
    height = len(grid)
    width = len(grid[0])
    flat = [value for row in grid for value in row]
    counts = [flat.count(color) for color in range(10)]
    return [round_float(value) for value in [
        height / 30,
        width / 30,
        (height * width) / 900,
        len(projection["palette"]) / 10,
        len(projection["nonzeroPalette"]) / 9,
        projection["nonZeroCells"] / 900,
        projection["density"],
        projection["nonZeroComponents"] / 900,
        *[1 if color in projection["palette"] else 0 for color in range(10)],
        *[count / (height * width) for count in counts],
    ]]


def signature_suffix(projection: dict[str, Any]) -> dict[int, float]:
    weights: dict[int, float] = {}
    object_tokens = object_tokens_for(projection["canonicalObjectSignature"])
    for token in object_tokens:
        add_hashed(weights, "object", token, 1 / len(object_tokens))
    bag_counts = Counter(projection["localSignatureBag"])
    bag_denom = max(1, len(projection["localSignatureBag"]))
    for stencil, count in bag_counts.items():
        add_hashed(weights, "bag", f"bag:stencil={stencil}", count / bag_denom)
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if norm > 0:
        for index in list(weights):
            weights[index] = round_float(weights[index] / norm)
    return weights


def object_tokens_for(signature: str) -> list[str]:
    if signature == "empty":
        return ["obj:empty"]
    bbox, cells_text = signature.split("|")
    bbox_w, bbox_h = bbox.split("x")
    cells = [cell for cell in cells_text.split(";") if cell]
    roles = {cell.split(":")[2] for cell in cells}
    return [
        f"obj:bbox_w={bbox_w}",
        f"obj:bbox_h={bbox_h}",
        f"obj:role_count={len(roles)}",
        f"obj:cell_count={len(cells)}",
        *[f"obj:cell={cell}" for cell in cells],
    ]


def add_hashed(weights: dict[int, float], namespace: str, token: str, value: float) -> None:
    digest = hashlib.sha256(f"{FEATURE_SCHEMA_VERSION}\0{namespace}\0{token}".encode("utf8")).digest()
    bucket = METADATA_DIM + (int.from_bytes(digest[:4], "big") % SIGNATURE_HASH_DIM)
    weights[bucket] = weights.get(bucket, 0.0) + value


def raw_grid_onehot(grid: list[list[int]]) -> list[float]:
    values = []
    for y in range(MAX_H):
        for x in range(MAX_W):
            in_grid = y < len(grid) and x < len(grid[0])
            color = grid[y][x] if in_grid else None
            for channel in range(PAD_CHANNELS):
                if in_grid:
                    values.append(1.0 if channel == color else 0.0)
                else:
                    values.append(1.0 if channel == 10 else 0.0)
    return values


def nonzero_cells(grid: list[list[int]]) -> list[dict[str, int]]:
    return [{"x": x, "y": y, "color": value} for y, row in enumerate(grid) for x, value in enumerate(row) if value != 0]


def count_components(grid: list[list[int]]) -> int:
    seen = [[False for _ in row] for row in grid]
    count = 0
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            if value == 0 or seen[y][x]:
                continue
            count += 1
            stack = [(x, y)]
            seen[y][x] = True
            while stack:
                cx, cy = stack.pop()
                for nx, ny in [(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)]:
                    if ny < 0 or ny >= len(grid) or nx < 0 or nx >= len(grid[0]) or seen[ny][nx] or grid[ny][nx] == 0:
                        continue
                    seen[ny][nx] = True
                    stack.append((nx, ny))
    return count


def rotate90(grid: list[list[int]]) -> list[list[int]]:
    return [[grid[len(grid) - 1 - x][y] for x in range(len(grid))] for y in range(len(grid[0]))]


def rotate180(grid: list[list[int]]) -> list[list[int]]:
    return reflect_vertical(reflect_horizontal(grid))


def rotate270(grid: list[list[int]]) -> list[list[int]]:
    return rotate90(rotate180(grid))


def reflect_horizontal(grid: list[list[int]]) -> list[list[int]]:
    return [list(reversed(row)) for row in grid]


def reflect_vertical(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in reversed(grid)]


def transpose(grid: list[list[int]]) -> list[list[int]]:
    return [[grid[x][y] for x in range(len(grid))] for y in range(len(grid[0]))]


def anti_transpose(grid: list[list[int]]) -> list[list[int]]:
    return reflect_horizontal(reflect_vertical(transpose(grid)))


def batches(items: list[Any], batch_size: int):
    for index in range(0, len(items), batch_size):
        yield items[index:index + batch_size]


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)


def rate(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [row[field] for row in rows if row[field] != "NA"]
    if not values:
        return "NA"
    return round_float(sum(1 for value in values if value is True or str(value).lower() == "true") / len(values))


def mean(rows: list[dict[str, Any]], field: str) -> float | str:
    values = [numeric(row[field]) for row in rows if row[field] != "NA"]
    if not values:
        return "NA"
    return round_float(sum(values) / len(values))


def dominant(rows: list[dict[str, Any]], field: str) -> str:
    counts = Counter(str(row[field]) for row in rows)
    return counts.most_common(1)[0][0] if counts else ""


def numeric(value: Any) -> float:
    if value == "NA" or value is None or value == "":
        return 0.0
    return float(value)


def bool_or_na(value: bool | None) -> bool | str:
    return "NA" if value is None else bool(value)


def round_float(value: float) -> float:
    return round(float(value), 9)


def grids_equal(a: list[list[int]], b: list[list[int]]) -> bool:
    return grids_same_shape(a, b) and all(a[y][x] == b[y][x] for y in range(len(a)) for x in range(len(a[0])))


def grids_same_shape(a: list[list[int]], b: list[list[int]]) -> bool:
    return len(a) == len(b) and len(a[0]) == len(b[0])


def pixel_accuracy(a: list[list[int]], b: list[list[int]]) -> float:
    if not grids_same_shape(a, b):
        return 0.0
    total = len(b) * len(b[0])
    correct = sum(1 for y in range(len(b)) for x in range(len(b[0])) if a[y][x] == b[y][x])
    return correct / total


def shape_label(grid: list[list[int]]) -> str:
    return f"{len(grid)}x{len(grid[0])}"


def palette_label(grid: list[list[int]]) -> str:
    return "".join(str(value) for value in sorted(set(value for row in grid for value in row)))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest().upper()


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", encoding="utf8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def hash_receipt_files(out_dir: Path) -> dict[str, str]:
    hashes = {}
    for path in sorted(out_dir.rglob("*")):
        if path.is_dir() or path.name == "hashes.json":
            continue
        hashes[str(path.relative_to(out_dir)).replace("\\", "/")] = hashlib.sha256(path.read_bytes()).hexdigest().upper()
    return hashes


def write_markdown_summary(path: Path, branch: dict[str, Any], scores: list[dict[str, Any]], selected_seed_by_arm: dict[str, int]) -> None:
    lines = [
        "# Phase 3 Blackwell Branch Adjudication",
        "",
        f"Branch: **{branch['branch']}**",
        "",
        branch.get("reason", ""),
        "",
        "## Selected Seeds",
        "",
        "| arm | seed |",
        "| --- | ---: |",
    ]
    for arm, seed in sorted(selected_seed_by_arm.items()):
        lines.append(f"| `{arm}` | `{seed}` |")
    lines.extend([
        "",
        "## Scores",
        "",
        "| lane | arm | grid exact | rep exact | pixel best |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in scores:
        lines.append(
            f"| `{row['lane']}` | `{row['arm']}` | `{row['grid_exact_any_rate']}` | "
            f"`{row['rep_exact_any_rate']}` | `{row['pixel_accuracy_best_mean']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


LEARNING_COLUMNS = ["arm", "seed", "epoch", "train_loss", "validation_loss", "validation_metric", "selected"]
PER_SLOT_COLUMNS = [
    "instance_id",
    "lane",
    "task_id",
    "primary_prior",
    "predicted_boundary",
    "arm",
    "seed",
    "slot",
    "grid_exact",
    "rep_exact",
    "shape_exact",
    "palette_exact",
    "pixel_accuracy",
    "failure_label",
]
PER_TASK_COLUMNS = [
    "lane",
    "task_id",
    "primary_prior",
    "predicted_boundary",
    "arm",
    "seed",
    "instance_count",
    "grid_exact_any_rate",
    "rep_exact_any_rate",
    "shape_exact_slot1_rate",
    "palette_exact_slot1_rate",
    "pixel_accuracy_best_mean",
    "dominant_failure_label",
]
PER_PRIOR_COLUMNS = [
    "lane",
    "primary_prior",
    "predicted_boundary",
    "arm",
    "instance_count",
    "grid_exact_any_rate",
    "rep_exact_any_rate",
    "shape_exact_slot1_rate",
    "palette_exact_slot1_rate",
    "pixel_accuracy_best_mean",
]
SCORE_COLUMNS = [
    "lane",
    "arm",
    "selected_seed",
    "task_count",
    "instance_count",
    "grid_exact_any_rate",
    "rep_exact_any_rate",
    "shape_exact_slot1_rate",
    "palette_exact_slot1_rate",
    "pixel_accuracy_best_mean",
    "task_bootstrap_metric_mean",
    "task_bootstrap_ci_low",
    "task_bootstrap_ci_high",
]
QUARANTINE_COLUMNS = [
    "lane",
    "task_id",
    "primary_prior",
    "predicted_boundary",
    "arm",
    "quarantine_key",
    "evidence",
    "blocks_branch_b",
]


if __name__ == "__main__":
    raise SystemExit(main())
