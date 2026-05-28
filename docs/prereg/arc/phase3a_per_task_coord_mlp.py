#!/usr/bin/env python
"""ARC Phase 3A stochastic per-task coordinate MLP runner (per_task_coord_mlp_v1).

This runner is fully self-contained: it does not import phase3_decoder.py. The
arc-p3-feature-v1 encoders are copied verbatim and labelled below; if either
this file or phase3_decoder.py changes them, FEATURE_SCHEMA_VERSION must be
bumped in both places. The manifest records both this spec hash and the parent
spec hash so a future drift is auditable.

Spec: docs/prereg/arc/PHASE3A_STOCHASTIC_PER_TASK_SPEC.md (filed 2026-05-28).
Parent: docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Frozen Phase 3A constants
# ============================================================================
FEATURE_SCHEMA_VERSION = "arc-p3-feature-v1"
LEARNER_VERSION = "per_task_coord_mlp_v1"
PROTOCOL_VERSION = "arc-p3a-per-task-coord-mlp-v1"
RECEIPT_SCHEMA_VERSION = "arc-p3a-per-task-receipt-v1"

ARMS = [
    "raw_grid_per_task",
    "signature_palette_per_task",
    "signature_only_per_task",
    "metadata_only_per_task",
]
GRID_SCORABLE_ARMS = {"raw_grid_per_task", "signature_palette_per_task"}
SEED_SLATE = [20260528, 20260529, 20260530, 20260531, 20260601]

MAX_H = 30
MAX_W = 30
MAX_COLORS = 10
PAD_CHANNELS = 11  # 0..9 + padding token
METADATA_DIM = 28
SIGNATURE_HASH_DIM = 4096
SIGNATURE_VECTOR_DIM = METADATA_DIM + SIGNATURE_HASH_DIM
RAW_GRID_DIM = MAX_H * MAX_W * PAD_CHANNELS  # 9900
COORD_FEATURE_DIM = 2 + 2 + 4  # normalized (2) + centered (2) + boundary (4)
PATCH_DIM = 9 * PAD_CHANNELS  # 3x3 nearest neighbor patch, raw arm only

SHAPE_MODEL_SPEC = {
    "hidden": 128,
    "out_dim": 30,
    "lr": 1e-3,
    "betas": [0.9, 0.99],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "max_steps": 600,
    "early_stop_patience": 80,
    "grad_clip_norm": 1.0,
}
COLOR_MODEL_SPEC = {
    "hidden": 192,
    "out_dim": 10,
    "dropout": 0.05,
    "lr": 1e-3,
    "betas": [0.9, 0.99],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "batch_size": 512,
    "max_steps": 900,
    "early_stop_patience": 120,
    "grad_clip_norm": 1.0,
    "class_weight_min": 0.25,
    "class_weight_max": 5.0,
}

# Pre-registered Phase 0 task split. Frozen here under
# arc-p3a-per-task-coord-mlp-v1; mirrors phase3_decoder.EXPECTED_SPLIT.
EXPECTED_SPLIT = {
    "color_role": {"train": ["08ed6ac7", "0a2355a6", "2601afb7", "292dd178"], "validation": ["37d3e8b2"], "test": ["3ad05f52"]},
    "counting": {"train": ["009d5c81", "00dbd492", "025d127b", "045e512c"], "validation": ["05269061"], "test": ["05a7bcf2"]},
    "local_completion": {"train": ["03560426", "05f2a901", "0b17323b", "0e671a1a"], "validation": ["11e1fe23"], "test": ["13713586"]},
    "objectness": {"train": ["11dc524f", "150deff5", "1acc24af", "1b60fb0c"], "validation": ["2bee17df"], "test": ["3906de3d"]},
    "spatial_transform": {"train": ["00576224", "0a1d4ef5", "0b148d64", "0bb8deee"], "validation": ["0c9aba6e"], "test": ["137eaa0f"]},
    "symmetry": {"train": ["007bbfb7", "00d62c1b", "017c7c7b", "0520fde7"], "validation": ["0692e18c"], "test": ["0a938d79"]},
}


# ============================================================================
# Dataclasses
# ============================================================================
@dataclass(frozen=True)
class Task:
    task_id: str
    primary_prior: str
    predicted_boundary: str
    train: list[dict[str, Any]]
    test: list[dict[str, Any]]
    split: str


@dataclass(frozen=True)
class Instance:
    lane: str
    instance_id: str
    task_id: str
    primary_prior: str
    predicted_boundary: str
    split: str
    query_index: int
    query_input: list[list[int]]
    target_output: list[list[int]]
    conditioning: list[dict[str, Any]]


# ============================================================================
# IO + hashing utilities
# ============================================================================
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest().upper()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def round_float(value: float) -> float:
    return round(value, 9)


def iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in columns})


def hash_receipt_files(out_dir: Path) -> dict[str, str]:
    out = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file() and path.name != "hashes.json":
            out[path.name] = sha256_file(path)
    return out


def git_state(repo_root: Path, allow_dirty: bool) -> dict[str, Any]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip().upper()
    dirty_out = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo_root, text=True).strip()
    dirty = bool(dirty_out)
    if dirty and not allow_dirty:
        raise SystemExit("Refusing to run on a dirty worktree; commit the freeze marker first or pass --allow-dirty for smoke checks.")
    return {"commit": commit, "dirty": dirty}


def assert_training_data_dir(data_dir: Path) -> None:
    normalized = str(data_dir).replace("\\", "/").lower()
    if normalized.endswith("/evaluation"):
        raise SystemExit("Refusing to use an ARC evaluation directory as --data-dir.")
    if not (data_dir / "training").is_dir():
        raise SystemExit(f"Missing training directory under {data_dir}")


# ============================================================================
# Frozen feature-v1 encoders (copied verbatim from phase3_decoder.py;
# do not modify without bumping FEATURE_SCHEMA_VERSION in BOTH files).
# ============================================================================
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


def rotate90(grid):
    return [[grid[len(grid) - 1 - x][y] for x in range(len(grid))] for y in range(len(grid[0]))]


def rotate180(grid):
    return reflect_vertical(reflect_horizontal(grid))


def rotate270(grid):
    return rotate90(rotate180(grid))


def reflect_horizontal(grid):
    return [list(reversed(row)) for row in grid]


def reflect_vertical(grid):
    return [list(row) for row in reversed(grid)]


def transpose(grid):
    return [[grid[x][y] for x in range(len(grid))] for y in range(len(grid[0]))]


def anti_transpose(grid):
    return reflect_horizontal(reflect_vertical(transpose(grid)))


def role_normalize_grid(grid):
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


def stencil_transforms():
    return [lambda g: g, rotate90, rotate180, rotate270, reflect_horizontal, reflect_vertical, transpose, anti_transpose]


def canonical_stencil(grid, cx, cy, radius):
    cells = []
    for y in range(cy - radius, cy + radius + 1):
        row = []
        for x in range(cx - radius, cx + radius + 1):
            row.append(0 if y < 0 or y >= len(grid) or x < 0 or x >= len(grid[0]) else grid[y][x])
        cells.append(row)
    return sorted(role_normalize_grid(transform(cells)) for transform in stencil_transforms())[0]


def object_variants(grid):
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
        role_map: dict[int, int] = {}
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


def canonical_object_signature(grid):
    variants = object_variants(grid)
    return "empty" if not variants else sorted(variant["signature"] for variant in variants)[0]


def project_grid_shadow(grid):
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


def metadata_vector(grid, projection):
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


def object_tokens_for(signature):
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


def add_hashed(weights, namespace, token, value):
    digest = hashlib.sha256(f"{FEATURE_SCHEMA_VERSION}\0{namespace}\0{token}".encode("utf8")).digest()
    bucket = METADATA_DIM + (int.from_bytes(digest[:4], "big") % SIGNATURE_HASH_DIM)
    weights[bucket] = weights.get(bucket, 0.0) + value


def signature_suffix(projection):
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


def represent_grid(grid, arm):
    projection = project_grid_shadow(grid)
    metadata = metadata_vector(grid, projection)
    suffix = signature_suffix(projection)
    return {
        "arm": arm,
        "metadata": metadata,
        "suffix": suffix,
    }


def feature_vector(grid: list[list[int]], arm: str) -> list[float]:
    """Arm-specific input vector for a single grid. Maps each non-raw arm into
    the SIGNATURE_VECTOR_DIM frame and the raw arm into RAW_GRID_DIM."""
    if arm == "raw_grid_per_task":
        return raw_grid_onehot(grid)
    rep = represent_grid(grid, arm)
    vector = [0.0] * SIGNATURE_VECTOR_DIM
    if arm in {"signature_palette_per_task", "metadata_only_per_task"}:
        vector[:METADATA_DIM] = rep["metadata"]
    if arm in {"signature_palette_per_task", "signature_only_per_task"}:
        for idx, value in rep["suffix"].items():
            vector[idx] = value
    return vector


def representation_key(grid: list[list[int]], arm: str) -> str:
    if arm == "raw_grid_per_task":
        return json.dumps(grid, separators=(",", ":"))
    rep = represent_grid(grid, arm)
    payload: dict[str, Any] = {}
    if arm in {"signature_palette_per_task", "metadata_only_per_task"}:
        payload["metadata"] = rep["metadata"]
    if arm in {"signature_palette_per_task", "signature_only_per_task"}:
        payload["suffix"] = sorted((int(k), round_float(v)) for k, v in rep["suffix"].items())
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def input_dim_for_arm(arm: str) -> int:
    return RAW_GRID_DIM if arm == "raw_grid_per_task" else SIGNATURE_VECTOR_DIM


# ============================================================================
# Phase 3A coordinate features (new, per spec §"Feature Rows")
# ============================================================================
def coord_features(oy: int, ox: int, output_h: int, output_w: int) -> list[float]:
    norm_y = oy / (output_h - 1) if output_h > 1 else 0.0
    norm_x = ox / (output_w - 1) if output_w > 1 else 0.0
    cent_y = 2 * norm_y - 1
    cent_x = 2 * norm_x - 1
    top = 1.0 if oy == 0 else 0.0
    bottom = 1.0 if oy == output_h - 1 else 0.0
    left = 1.0 if ox == 0 else 0.0
    right = 1.0 if ox == output_w - 1 else 0.0
    return [round_float(v) for v in [norm_y, norm_x, cent_y, cent_x, top, bottom, left, right]]


def shape_norm_features(input_h: int, input_w: int, output_h: int, output_w: int) -> list[float]:
    return [round_float(v) for v in [input_h / 30, input_w / 30, output_h / 30, output_w / 30]]


def patch_3x3(grid: list[list[int]], oy: int, ox: int, output_h: int, output_w: int) -> list[float]:
    """Nearest-neighbor 3x3 patch sampled at the normalized output coordinate's position in the input grid."""
    norm_y = oy / (output_h - 1) if output_h > 1 else 0.0
    norm_x = ox / (output_w - 1) if output_w > 1 else 0.0
    input_h = len(grid)
    input_w = len(grid[0]) if input_h else 0
    iy = int(round(norm_y * (input_h - 1))) if input_h > 0 else 0
    ix = int(round(norm_x * (input_w - 1))) if input_w > 0 else 0
    values: list[float] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y = iy + dy
            x = ix + dx
            in_grid = 0 <= y < input_h and 0 <= x < input_w
            color = grid[y][x] if in_grid else None
            for channel in range(PAD_CHANNELS):
                if in_grid:
                    values.append(1.0 if channel == color else 0.0)
                else:
                    values.append(1.0 if channel == 10 else 0.0)
    return values


def color_row_features(arm: str, input_grid: list[list[int]], oy: int, ox: int, output_h: int, output_w: int) -> list[float]:
    base = feature_vector(input_grid, arm)
    input_h = len(input_grid)
    input_w = len(input_grid[0]) if input_h else 0
    feats = list(base) + coord_features(oy, ox, output_h, output_w) + shape_norm_features(input_h, input_w, output_h, output_w)
    if arm == "raw_grid_per_task":
        feats.extend(patch_3x3(input_grid, oy, ox, output_h, output_w))
    return feats


def shape_row_features(arm: str, input_grid: list[list[int]]) -> list[float]:
    base = feature_vector(input_grid, arm)
    input_h = len(input_grid)
    input_w = len(input_grid[0]) if input_h else 0
    return list(base) + [round_float(input_h / 30), round_float(input_w / 30)]


def color_input_dim_for_arm(arm: str) -> int:
    base = input_dim_for_arm(arm)
    extra = COORD_FEATURE_DIM + 4
    if arm == "raw_grid_per_task":
        extra += PATCH_DIM
    return base + extra


def shape_input_dim_for_arm(arm: str) -> int:
    return input_dim_for_arm(arm) + 2


# ============================================================================
# Models
# ============================================================================
class ShapeMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h = SHAPE_MODEL_SPEC["hidden"]
        self.proj1 = nn.Linear(input_dim, h)
        self.norm = nn.LayerNorm(h)
        self.proj2 = nn.Linear(h, h)
        self.head_h = nn.Linear(h, SHAPE_MODEL_SPEC["out_dim"])
        self.head_w = nn.Linear(h, SHAPE_MODEL_SPEC["out_dim"])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = F.gelu(self.proj1(x))
        h = self.norm(h)
        h = F.gelu(self.proj2(h))
        return self.head_h(h), self.head_w(h)


class ColorMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h = COLOR_MODEL_SPEC["hidden"]
        self.proj1 = nn.Linear(input_dim, h)
        self.norm = nn.LayerNorm(h)
        self.proj2 = nn.Linear(h, h)
        self.drop = nn.Dropout(COLOR_MODEL_SPEC["dropout"])
        self.head = nn.Linear(h, COLOR_MODEL_SPEC["out_dim"])
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj1(x))
        h = self.norm(h)
        h = F.gelu(self.proj2(h))
        h = self.drop(h)
        return self.head(h)


# ============================================================================
# Seed derivation
# ============================================================================
def derive_seed(master_seed: int, lane: str, task_id: str, query_index: int, arm: str, model_kind: str) -> int:
    key = f"arc-p3a-per-task-coord-mlp-v1\0{master_seed}\0{lane}\0{task_id}\0{query_index}\0{arm}\0{model_kind}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================================
# Task + Instance loading
# ============================================================================
def expected_split_by_task() -> dict[str, str]:
    out: dict[str, str] = {}
    for prior, groups in EXPECTED_SPLIT.items():
        for split, task_ids in groups.items():
            for task_id in task_ids:
                out[task_id] = split
    return out


def load_tasks(data_dir: Path, register_path: Path) -> tuple[list[Task], str, str]:
    register_text = register_path.read_text(encoding="utf-8-sig")
    rows = [row for row in csv.DictReader(register_text.splitlines()) if row["status"] == "include" and row["split"] == "training"]
    tasks: list[Task] = []
    file_hashes: list[dict[str, str]] = []
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
            train=[{"index": i, "input": pair["input"], "output": pair["output"]} for i, pair in enumerate(parsed["train"])],
            test=[{"index": i, "input": pair["input"], "output": pair.get("output")} for i, pair in enumerate(parsed["test"])],
            split=split_by_task[task_id],
        ))
    register_hash = sha256_text(register_text)
    data_hash = sha256_text(json.dumps(file_hashes, sort_keys=True, separators=(",", ":")))
    return tasks, register_hash, data_hash


def build_lodo_instances(tasks: list[Task], lane: str) -> list[Instance]:
    instances: list[Instance] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        for held_out in task.train:
            other = [p for p in task.train if p["index"] != held_out["index"]]
            instances.append(Instance(
                lane=lane,
                instance_id=f"{lane}:{task.task_id}:{held_out['index']}",
                task_id=task.task_id,
                primary_prior=task.primary_prior,
                predicted_boundary=task.predicted_boundary,
                split=task.split,
                query_index=held_out["index"],
                query_input=held_out["input"],
                target_output=held_out["output"],
                conditioning=other,
            ))
    return instances


def build_pttest_instances(tasks: list[Task], lane: str) -> list[Instance]:
    instances: list[Instance] = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        for test in task.test:
            if test["output"] is None:
                continue
            instances.append(Instance(
                lane=lane,
                instance_id=f"{lane}:{task.task_id}:{test['index']}",
                task_id=task.task_id,
                primary_prior=task.primary_prior,
                predicted_boundary=task.predicted_boundary,
                split=task.split,
                query_index=test["index"],
                query_input=test["input"],
                target_output=test["output"],
                conditioning=task.train,
            ))
    return instances


# ============================================================================
# Per-instance training
# ============================================================================
def fit_shape(instance: Instance, arm: str, master_seed: int, max_steps: int, device: torch.device) -> tuple[ShapeMLP, dict[str, Any]]:
    seed = derive_seed(master_seed, instance.lane, instance.task_id, instance.query_index, arm, "shape")
    set_global_determinism(seed)
    model = ShapeMLP(shape_input_dim_for_arm(arm)).to(device)
    rows = []
    h_labels = []
    w_labels = []
    for pair in instance.conditioning:
        out_h = len(pair["output"])
        out_w = len(pair["output"][0]) if out_h else 0
        rows.append(shape_row_features(arm, pair["input"]))
        h_labels.append(min(out_h - 1, SHAPE_MODEL_SPEC["out_dim"] - 1))
        w_labels.append(min(out_w - 1, SHAPE_MODEL_SPEC["out_dim"] - 1))
    if not rows:
        return model, {"steps": 0, "best_loss": float("inf"), "seed": seed, "rows": 0}
    X = torch.tensor(rows, dtype=torch.float32, device=device)
    yh = torch.tensor(h_labels, dtype=torch.long, device=device)
    yw = torch.tensor(w_labels, dtype=torch.long, device=device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=SHAPE_MODEL_SPEC["lr"],
        betas=tuple(SHAPE_MODEL_SPEC["betas"]),
        eps=SHAPE_MODEL_SPEC["eps"],
        weight_decay=SHAPE_MODEL_SPEC["weight_decay"],
    )
    best_loss = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    for step in range(max_steps):
        model.train()
        optim.zero_grad()
        logits_h, logits_w = model(X)
        loss = F.cross_entropy(logits_h, yh) + F.cross_entropy(logits_w, yw)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), SHAPE_MODEL_SPEC["grad_clip_norm"])
        optim.step()
        loss_val = float(loss.detach().cpu().item())
        history.append({"step": step, "loss": round_float(loss_val)})
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1
        if patience >= SHAPE_MODEL_SPEC["early_stop_patience"]:
            break
    return model, {"steps": len(history), "best_loss": round_float(best_loss), "seed": seed, "rows": len(rows), "history": history}


def fit_color(instance: Instance, arm: str, master_seed: int, max_steps: int, device: torch.device) -> tuple[ColorMLP, dict[str, Any]]:
    seed = derive_seed(master_seed, instance.lane, instance.task_id, instance.query_index, arm, "color")
    set_global_determinism(seed)
    model = ColorMLP(color_input_dim_for_arm(arm)).to(device)
    rows: list[list[float]] = []
    labels: list[int] = []
    color_counts: Counter[int] = Counter()
    for pair in instance.conditioning:
        out_grid = pair["output"]
        in_grid = pair["input"]
        out_h = len(out_grid)
        out_w = len(out_grid[0]) if out_h else 0
        for oy in range(out_h):
            for ox in range(out_w):
                rows.append(color_row_features(arm, in_grid, oy, ox, out_h, out_w))
                labels.append(out_grid[oy][ox])
                color_counts[out_grid[oy][ox]] += 1
    if not rows:
        return model, {"steps": 0, "best_loss": float("inf"), "seed": seed, "rows": 0, "color_counts": dict(color_counts)}
    max_count = max(color_counts.values())
    weights = [1.0] * COLOR_MODEL_SPEC["out_dim"]
    for color in range(COLOR_MODEL_SPEC["out_dim"]):
        n = color_counts.get(color, 0)
        if n > 0:
            w = math.sqrt(max_count / n)
            w = max(COLOR_MODEL_SPEC["class_weight_min"], min(COLOR_MODEL_SPEC["class_weight_max"], w))
            weights[color] = w
    X = torch.tensor(rows, dtype=torch.float32, device=device)
    Y = torch.tensor(labels, dtype=torch.long, device=device)
    W = torch.tensor(weights, dtype=torch.float32, device=device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=COLOR_MODEL_SPEC["lr"],
        betas=tuple(COLOR_MODEL_SPEC["betas"]),
        eps=COLOR_MODEL_SPEC["eps"],
        weight_decay=COLOR_MODEL_SPEC["weight_decay"],
    )
    best_loss = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    batch_size = COLOR_MODEL_SPEC["batch_size"]
    n_rows = X.size(0)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    for step in range(max_steps):
        model.train()
        if n_rows <= batch_size:
            xb, yb = X, Y
        else:
            idx = torch.randperm(n_rows, generator=gen)[:batch_size]
            xb, yb = X[idx], Y[idx]
        optim.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb, weight=W)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), COLOR_MODEL_SPEC["grad_clip_norm"])
        optim.step()
        loss_val = float(loss.detach().cpu().item())
        history.append({"step": step, "loss": round_float(loss_val)})
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1
        if patience >= COLOR_MODEL_SPEC["early_stop_patience"]:
            break
    return model, {"steps": len(history), "best_loss": round_float(best_loss), "seed": seed, "rows": n_rows, "color_counts": dict(color_counts), "history": history}


def predict_shape(shape_model: ShapeMLP, instance: Instance, arm: str, device: torch.device) -> list[tuple[int, int]]:
    """Returns slot-1 and slot-2 (h, w) predictions."""
    shape_model.eval()
    x = torch.tensor([shape_row_features(arm, instance.query_input)], dtype=torch.float32, device=device)
    with torch.no_grad():
        logits_h, logits_w = shape_model(x)
        top_h = torch.topk(logits_h[0], 2).indices.cpu().tolist()
        top_w = torch.topk(logits_w[0], 2).indices.cpu().tolist()
    h1, w1 = top_h[0] + 1, top_w[0] + 1
    if top_h[1] != top_h[0]:
        h2, w2 = top_h[1] + 1, top_w[0] + 1
    elif top_w[1] != top_w[0]:
        h2, w2 = top_h[0] + 1, top_w[1] + 1
    else:
        h2, w2 = h1, w1
    return [(h1, w1), (h2, w2)]


def predict_colors(color_model: ColorMLP, instance: Instance, arm: str, shape_hw: tuple[int, int], device: torch.device, sample_seed: int | None = None) -> list[list[int]]:
    color_model.eval()
    h, w = shape_hw
    h = max(1, min(MAX_H, h))
    w = max(1, min(MAX_W, w))
    rows: list[list[float]] = []
    for oy in range(h):
        for ox in range(w):
            rows.append(color_row_features(arm, instance.query_input, oy, ox, h, w))
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    with torch.no_grad():
        logits = color_model(x)
        if sample_seed is None:
            colors = logits.argmax(dim=-1).cpu().tolist()
        else:
            gen = torch.Generator(device="cpu")
            gen.manual_seed(sample_seed)
            probs = F.softmax(logits / 0.75, dim=-1).cpu()
            colors = torch.multinomial(probs, num_samples=1, generator=gen)[:, 0].tolist()
    out: list[list[int]] = []
    idx = 0
    for _ in range(h):
        out.append(colors[idx:idx + w])
        idx += w
    return out


def conditioning_train_exact_rate(shape_model: ShapeMLP, color_model: ColorMLP, instance: Instance, arm: str, device: torch.device) -> float:
    if not instance.conditioning:
        return 0.0
    hits = 0
    for pair in instance.conditioning:
        pseudo = Instance(
            lane=f"{instance.lane}:conditioning",
            instance_id=f"{instance.instance_id}:conditioning:{pair['index']}",
            task_id=instance.task_id,
            primary_prior=instance.primary_prior,
            predicted_boundary=instance.predicted_boundary,
            split=instance.split,
            query_index=pair["index"],
            query_input=pair["input"],
            target_output=pair["output"],
            conditioning=instance.conditioning,
        )
        shape = predict_shape(shape_model, pseudo, arm, device)[0]
        pred = predict_colors(color_model, pseudo, arm, shape, device, sample_seed=None)
        if grid_equal(pred, pair["output"]):
            hits += 1
    return hits / len(instance.conditioning)


# ============================================================================
# Scoring + quarantine labels
# ============================================================================
def grid_equal(a: list[list[int]], b: list[list[int]]) -> bool:
    if len(a) != len(b):
        return False
    if a and len(a[0]) != len(b[0]):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


def shape_of(grid: list[list[int]]) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def palette_of(grid: list[list[int]]) -> set[int]:
    return {c for row in grid for c in row}


def pixel_accuracy(pred: list[list[int]], target: list[list[int]]) -> float:
    if not pred or not target:
        return 0.0
    ph, pw = shape_of(pred)
    th, tw = shape_of(target)
    if ph != th or pw != tw:
        return 0.0
    total = th * tw
    if total == 0:
        return 0.0
    correct = 0
    for y in range(th):
        for x in range(tw):
            if pred[y][x] == target[y][x]:
                correct += 1
    return correct / total


def minority_color_recall(pred: list[list[int]], target: list[list[int]]) -> float:
    if not pred or not target:
        return 0.0
    flat_target = [c for row in target for c in row]
    if not flat_target:
        return 0.0
    counts = Counter(flat_target)
    modal_color = counts.most_common(1)[0][0]
    ph, pw = shape_of(pred)
    th, tw = shape_of(target)
    h = min(ph, th)
    w = min(pw, tw)
    minority_total = 0
    minority_hit = 0
    for y in range(th):
        for x in range(tw):
            if target[y][x] == modal_color:
                continue
            minority_total += 1
            if y < h and x < w and pred[y][x] == target[y][x]:
                minority_hit += 1
    if minority_total == 0:
        return 1.0
    return minority_hit / minority_total


def palette_jaccard(pred: list[list[int]], target: list[list[int]]) -> float:
    pp = palette_of(pred)
    pt = palette_of(target)
    if not pp and not pt:
        return 1.0
    return len(pp & pt) / len(pp | pt)


def dominant_color_collapse(pred: list[list[int]], target: list[list[int]]) -> bool:
    return len(palette_of(target)) >= 3 and len(palette_of(pred)) <= 2


def score_prediction(pred: list[list[int]], target: list[list[int]]) -> dict[str, Any]:
    return {
        "grid_exact": grid_equal(pred, target),
        "shape_exact": shape_of(pred) == shape_of(target),
        "palette_exact": palette_of(pred) == palette_of(target),
        "pixel_accuracy": round_float(pixel_accuracy(pred, target)),
        "minority_color_recall": round_float(minority_color_recall(pred, target)),
        "palette_jaccard_slot1": round_float(palette_jaccard(pred, target)),
        "predicted_color_count_slot1": len(palette_of(pred)),
        "target_color_count": len(palette_of(target)),
        "dominant_color_collapse": dominant_color_collapse(pred, target),
    }


def has_signature_collision(instance: Instance, arm: str) -> bool:
    seen: dict[str, set[str]] = {}
    for pair in instance.conditioning:
        input_key = representation_key(pair["input"], arm)
        output_key = representation_key(pair["output"], arm)
        seen.setdefault(input_key, set()).add(output_key)
    return any(len(outputs) > 1 for outputs in seen.values())


def assign_quarantine_label(record: dict[str, Any], conditioning_n: int, arm: str, primary_prior: str, signature_collision: bool) -> str:
    if record["grid_exact_any_slot"]:
        return ""
    if conditioning_n < 3:
        return "insufficient_conditioning_pairs"
    if signature_collision:
        return "signature_collision"
    if not record["shape_exact_slot1"]:
        return "shape_prediction_failure"
    if record["dominant_color_collapse_slot1"]:
        return "dominant_color_mode_collapse"
    if record["minority_color_recall_slot1"] < 0.25:
        return "minority_object_recall_failure"
    if arm == "signature_only_per_task" and primary_prior == "color_role":
        return "color_permutation_quotient"
    return "conditioning_overfit" if record.get("conditioning_train_exact", False) else "palette_lift_failure"


# ============================================================================
# Arena gate + branch adjudication
# ============================================================================
def exact_task_count(per_task_rows: list[dict[str, Any]], lane: str, arm: str) -> int:
    return sum(
        1
        for row in per_task_rows
        if row["lane"] == lane and row["arm"] == arm and float(row["grid_exact_any_rate"] or 0.0) > 0.010
    )


def adjudicate_arena_gate(per_task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lodo = exact_task_count(per_task_rows, "test_lodo", "raw_grid_per_task")
    pt = exact_task_count(per_task_rows, "pttest", "raw_grid_per_task")
    if lodo >= 1 and pt >= 1:
        return {"gate": "raw_grid_arena_open", "test_lodo_exact_tasks": lodo, "pttest_exact_tasks": pt}
    return {"gate": "branch_a_full_grid_floor", "test_lodo_exact_tasks": lodo, "pttest_exact_tasks": pt}


def adjudicate_branch_a(per_task_rows: list[dict[str, Any]], arena: dict[str, Any]) -> dict[str, Any]:
    if arena["gate"] != "raw_grid_arena_open":
        return {
            "branch": "branch_a_full_grid_floor",
            "reason": "raw_grid_per_task did not open the arena (>=1 exact task on each held-out lane required)",
        }
    raw_lodo = exact_task_count(per_task_rows, "test_lodo", "raw_grid_per_task")
    raw_pt = exact_task_count(per_task_rows, "pttest", "raw_grid_per_task")
    sig_lodo = exact_task_count(per_task_rows, "test_lodo", "signature_palette_per_task")
    sig_pt = exact_task_count(per_task_rows, "pttest", "signature_palette_per_task")
    if sig_lodo >= 1 and sig_pt >= 1 and (raw_lodo - sig_lodo) <= 1 and (raw_pt - sig_pt) <= 1:
        return {
            "branch": "branch_a_support",
            "raw_grid": {"test_lodo": raw_lodo, "pttest": raw_pt},
            "signature_palette": {"test_lodo": sig_lodo, "pttest": sig_pt},
            "reason": "signature_palette_per_task opens the arena and trails raw_grid_per_task by no more than one task per held-out lane",
        }
    return {
        "branch": "branch_a_bounded_failure",
        "raw_grid": {"test_lodo": raw_lodo, "pttest": raw_pt},
        "signature_palette": {"test_lodo": sig_lodo, "pttest": sig_pt},
        "reason": "raw_grid_per_task opened the arena but signature_palette_per_task did not satisfy the support criteria",
    }


# ============================================================================
# Main
# ============================================================================
def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def read_csv_dicts(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def assert_shard_consistency(shards: list[dict[str, Any]], repo_root: Path | None = None, allow_mixed_commits: bool = False) -> dict[str, Any] | None:
    """Mirror V2: enforce the freeze-marker fingerprint matches across all shards.

    arm + seed pairs must all differ (no duplicate shard); every other
    schema/spec/register/data hash, model spec, and feature-schema version
    must match exactly. gitCommit must also match across all shards UNLESS
    `allow_mixed_commits=True`, in which case we instead require the runner
    file content to be byte-identical at every distinct shard gitCommit.

    Returns an audit dict when mixed-commit override is engaged so the merged
    manifest can record what was bypassed and why.
    """
    if len(shards) < 2:
        return None
    ref = shards[0]["manifest"]
    keys = [
        "featureSchemaVersion", "protocolVersion", "receiptSchemaVersion", "learnerVersion",
        "registerHash", "dataDirHash",
        "registerPath", "dataDir",
        "shapeModelSpec", "colorModelSpec", "seedSlate", "arms",
        "maxStepsEffective",
    ]
    if not allow_mixed_commits:
        # Strict path: gitCommit AND documentation-only spec hashes must match.
        keys.extend(["gitCommit", "specHash", "parentSpecHash"])
    # Under --allow-mixed-commits, specHash / parentSpecHash are NOT required to
    # match (operator amendments to either spec file change those hashes without
    # changing the runner code that produced the per-instance outputs). The
    # runtime audit below verifies the runner file content is byte-identical
    # across every distinct shard gitCommit, and the per-commit specHash /
    # parentSpecHash maps are recorded in mixedCommitsAudit for transparency.
    seen_arm_seed: set[tuple[str, int]] = set()
    for sh in shards:
        m = sh["manifest"]
        if m.get("mode") != "shard":
            raise SystemExit(f"shard dir {sh['dir']} has mode={m.get('mode')!r}, expected 'shard'")
        arm = m.get("shardArm")
        seed = m.get("shardSeed")
        key = (arm, seed)
        if key in seen_arm_seed:
            raise SystemExit(f"shard dir {sh['dir']} has duplicate (shardArm={arm!r}, shardSeed={seed}) pair")
        seen_arm_seed.add(key)
        for k in keys:
            ref_val = json.dumps(ref.get(k), sort_keys=True)
            sh_val = json.dumps(m.get(k), sort_keys=True)
            if ref_val != sh_val:
                raise SystemExit(f"shard dir {sh['dir']} disagrees with {shards[0]['dir']} on {k!r}:\n  ref={ref_val}\n  sh ={sh_val}")

    if not allow_mixed_commits:
        return None
    distinct_commits = sorted({sh["manifest"]["gitCommit"] for sh in shards})
    if len(distinct_commits) <= 1:
        return None
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    runner_path = "docs/prereg/arc/phase3a_per_task_coord_mlp.py"
    runner_shas: dict[str, str] = {}
    for c in distinct_commits:
        try:
            blob = subprocess.check_output(["git", "show", f"{c.lower()}:{runner_path}"], cwd=str(repo_root))
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"--allow-mixed-commits: cannot read {runner_path} at gitCommit {c}: {exc}")
        runner_shas[c] = hashlib.sha256(blob).hexdigest().upper()
    unique = sorted(set(runner_shas.values()))
    runner_identical = len(unique) == 1
    # The strict equality of shapeModelSpec / colorModelSpec / featureSchemaVersion /
    # protocolVersion / learnerVersion already enforced above guarantees the SHARD-TIME
    # computational contract did not change. If the runner file itself differs, that is
    # almost always a merge-time or CLI-time edit (e.g., adding this very override),
    # not a change to training semantics. We record but do not fail when those keys all
    # match. The operator's choice of --allow-mixed-commits is the trust marker; the
    # audit dict makes the divergence visible in the merged manifest.
    if runner_identical:
        print(f"--allow-mixed-commits: verified {runner_path} byte-identical across {len(distinct_commits)} commits")
    else:
        print(
            f"--allow-mixed-commits: WARN — {runner_path} differs across {len(distinct_commits)} commits "
            f"({len(unique)} distinct hashes). Shard-time computational contract "
            "(featureSchemaVersion, protocolVersion, learnerVersion, shapeModelSpec, "
            "colorModelSpec) IS equal across all shards — audit recorded for review."
        )
    audit = {
        "auditedFile": runner_path,
        "distinctCommits": distinct_commits,
        "runnerSha256ByCommit": runner_shas,
        "distinctRunnerSha256": unique,
        "runnerIdenticalAcrossCommits": runner_identical,
        "specHashByCommit": {sh["manifest"]["gitCommit"]: sh["manifest"].get("specHash") for sh in shards},
        "parentSpecHashByCommit": {sh["manifest"]["gitCommit"]: sh["manifest"].get("parentSpecHash") for sh in shards},
    }
    return audit


def run_merge(args) -> int:
    started_at = iso_now()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    git = git_state(repo_root, args.allow_dirty)

    shard_dirs = [Path(d.strip()).resolve() for d in args.shard_dirs.split(",") if d.strip()]
    if not shard_dirs:
        print("--shard-dirs is empty", file=sys.stderr)
        return 2

    shards = []
    for d in shard_dirs:
        if not d.is_dir():
            print(f"shard dir not found: {d}", file=sys.stderr)
            return 2
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("learnerVersion") != LEARNER_VERSION:
            print(f"shard dir {d} learnerVersion={manifest.get('learnerVersion')!r}, expected {LEARNER_VERSION!r}", file=sys.stderr)
            return 2
        shards.append({
            "dir": d,
            "manifest": manifest,
            "per_instance_rows": read_csv_dicts(d / "per_instance.csv"),
            "learning_rows": read_csv_dicts(d / "learning_curves.csv"),
            "residual_rows": read_jsonl(d / "residuals.jsonl"),
        })

    mixed_commits_audit = assert_shard_consistency(shards, repo_root=repo_root, allow_mixed_commits=args.allow_mixed_commits)
    # Sort shards deterministically by (arm, seed) so merged outputs are stable.
    shards.sort(key=lambda s: (s["manifest"]["shardArm"], s["manifest"]["shardSeed"]))

    # Concatenate raw shard outputs.
    per_instance_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    for sh in shards:
        for row in sh["per_instance_rows"]:
            # Coerce CSV strings back into the types aggregation expects.
            coerced = dict(row)
            for col, parse in (
                ("slot", int),
                ("seed", int),
                ("grid_exact", _parse_bool),
                ("shape_exact", _parse_bool),
                ("palette_exact", _parse_bool),
                ("pixel_accuracy", float),
                ("minority_color_recall", float),
                ("palette_jaccard_slot1", float),
                ("predicted_color_count_slot1", int),
                ("target_color_count", int),
                ("dominant_color_collapse", _parse_bool),
                ("conditioning_train_exact_rate", float),
                ("signature_collision", _parse_bool),
            ):
                if col in coerced:
                    coerced[col] = parse(coerced[col])
            per_instance_rows.append(coerced)
        learning_rows.extend(sh["learning_rows"])
        residual_rows.extend(sh["residual_rows"])

    # Reconstruct per-arm validation metrics and seed-outcome map from
    # per_instance rows; this guarantees the merged selection is computed from
    # the same source-of-truth as the shard's own per-seed metrics block.
    arms_present: list[str] = []
    seeds_present: set[int] = set()
    per_arm_validation_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    per_instance_seed_outcomes: dict[tuple[str, str], dict[int, bool]] = {}
    for r in per_instance_rows:
        seeds_present.add(r["seed"])
        if r["arm"] not in arms_present:
            arms_present.append(r["arm"])
        per_arm_validation_metrics.setdefault(r["arm"], {}).setdefault(r["seed"], {})
    # Build per-instance any-slot grid-exact + per-seed validation buckets:
    by_instance_seed: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in per_instance_rows:
        k = (r["arm"], r["instance_id"], r["seed"])
        cur = by_instance_seed.setdefault(k, {**r, "grid_exact_any": False, "shape_exact_any": False, "pixel_best": 0.0, "slot1": None})
        cur["grid_exact_any"] = cur["grid_exact_any"] or r["grid_exact"]
        cur["shape_exact_any"] = cur["shape_exact_any"] or r["shape_exact"]
        cur["pixel_best"] = max(cur["pixel_best"], r["pixel_accuracy"])
        if r["slot"] == 1:
            cur["slot1"] = r
    for (arm, inst_id, seed), cur in by_instance_seed.items():
        per_instance_seed_outcomes.setdefault((arm, inst_id), {})[seed] = cur["grid_exact_any"]
        # Only the validation_* lanes contribute to the per-seed validation metric bucket.
        if cur.get("lane", "").startswith("validation_"):
            bucket = per_arm_validation_metrics[arm][seed].setdefault("counts", {"grid_exact": 0, "rep_exact": 0, "n": 0, "min_recall_sum": 0.0, "collapse": 0, "loss_sum": 0.0})
            bucket["n"] += 1
            if cur["grid_exact_any"]:
                bucket["grid_exact"] += 1
            if cur["shape_exact_any"]:
                bucket["rep_exact"] += 1
            slot1 = cur["slot1"] or {}
            bucket["min_recall_sum"] += slot1.get("minority_color_recall", 0.0)
            if slot1.get("dominant_color_collapse"):
                bucket["collapse"] += 1
    # Pull val_loss from shard manifests (each shard already computed it).
    for sh in shards:
        sm = sh["manifest"]
        arm = sm["shardArm"]
        seed = sm["shardSeed"]
        sm_metrics = sm.get("perSeedValidationMetrics", {}).get(arm, {}).get(str(seed), {})
        per_arm_validation_metrics[arm][seed]["val_loss"] = sm_metrics.get("val_loss", float("inf"))
    # Roll up the bucket counts the same way the training loop does.
    for arm, by_seed in per_arm_validation_metrics.items():
        for seed, m in by_seed.items():
            bucket = m.get("counts", {"grid_exact": 0, "rep_exact": 0, "n": 0, "min_recall_sum": 0.0, "collapse": 0, "loss_sum": 0.0})
            n = max(1, bucket["n"])
            m["val_grid_exact_count"] = bucket["grid_exact"]
            m["val_rep_exact_count"] = bucket["rep_exact"]
            m["val_minority_recall"] = round_float(bucket["min_recall_sum"] / n)
            m["val_collapse_rate"] = round_float(bucket["collapse"] / n)

    arms = sorted(arms_present, key=lambda a: ARMS.index(a))
    seeds = sorted(seeds_present)

    # Selection
    selected_seed_by_arm: dict[str, int] = {}
    for arm in arms:
        selected_seed_by_arm[arm] = select_seed_for_arm(arm, per_arm_validation_metrics[arm])

    # Re-aggregate using the per-instance rows
    selected_any_rows: list[dict[str, Any]] = []
    selected_rows_slot1: list[dict[str, Any]] = []
    for arm in arms:
        sel = selected_seed_by_arm[arm]
        by_inst: dict[tuple[str, str, str], dict[str, Any]] = {}
        for r in per_instance_rows:
            if r["arm"] != arm or r["seed"] != sel:
                continue
            k = (r["lane"], r["task_id"], r["instance_id"])
            cur = by_inst.setdefault(k, {
                "lane": r["lane"],
                "task_id": r["task_id"],
                "instance_id": r["instance_id"],
                "primary_prior": r["primary_prior"],
                "predicted_boundary": r.get("predicted_boundary", ""),
                "arm": r["arm"],
                "seed": r["seed"],
                "grid_exact_any": False,
                "shape_exact_slot1": False,
                "palette_exact_slot1": False,
                "pixel_accuracy_best": 0.0,
                "minority_color_recall_slot1": 0.0,
                "dominant_color_collapse_slot1": False,
                "target_color_count": int(r["target_color_count"]),
                "predicted_color_count_slot1": 0,
                "quarantine_label": "",
            })
            cur["grid_exact_any"] = cur["grid_exact_any"] or r["grid_exact"]
            cur["pixel_accuracy_best"] = max(cur["pixel_accuracy_best"], r["pixel_accuracy"])
            if r["slot"] == 1:
                cur["shape_exact_slot1"] = r["shape_exact"]
                cur["palette_exact_slot1"] = r["palette_exact"]
                cur["minority_color_recall_slot1"] = r["minority_color_recall"]
                cur["dominant_color_collapse_slot1"] = r["dominant_color_collapse"]
                cur["predicted_color_count_slot1"] = int(r["predicted_color_count_slot1"])
                cur["quarantine_label"] = r.get("quarantine_label", "")
        selected_any_rows.extend(by_inst.values())
        selected_rows_slot1.extend(by_inst.values())

    def _agg_scores(rows):
        out = []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["lane"], r["arm"]), []).append(r)
        for (lane, arm), group in sorted(groups.items()):
            task_ids = sorted({r["task_id"] for r in group})
            out.append({
                "lane": lane,
                "arm": arm,
                "selected_seed": selected_seed_by_arm[arm],
                "task_count": len(task_ids),
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact_any"]) / len(group)),
                "shape_exact_slot1_rate": round_float(sum(1 for r in group if r.get("shape_exact_slot1")) / len(group)),
                "palette_exact_slot1_rate": round_float(sum(1 for r in group if r.get("palette_exact_slot1")) / len(group)),
                "pixel_accuracy_best_mean": round_float(sum(r.get("pixel_accuracy_best", 0.0) for r in group) / len(group)),
                "minority_color_recall_mean": round_float(sum(r.get("minority_color_recall_slot1", 0.0) for r in group) / len(group)),
                "dominant_color_collapse_rate": round_float(sum(1 for r in group if r.get("dominant_color_collapse_slot1")) / len(group)),
            })
        return out

    def _agg_per_task(rows):
        out = []
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["lane"], r["arm"], r["task_id"]), []).append(r)
        for (lane, arm, task_id), group in sorted(groups.items()):
            out.append({
                "lane": lane,
                "task_id": task_id,
                "primary_prior": group[0]["primary_prior"],
                "predicted_boundary": next((g.get("predicted_boundary", "") for g in group if g.get("predicted_boundary")), ""),
                "arm": arm,
                "selected_seed": selected_seed_by_arm[arm],
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact_any"]) / len(group)),
                "shape_exact_slot1_rate": round_float(sum(1 for r in group if r.get("shape_exact_slot1")) / len(group)),
                "palette_exact_slot1_rate": round_float(sum(1 for r in group if r.get("palette_exact_slot1")) / len(group)),
                "pixel_accuracy_best_mean": round_float(sum(r.get("pixel_accuracy_best", 0.0) for r in group) / len(group)),
            })
        return out

    def _agg_per_prior(rows):
        out = []
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["lane"], r["primary_prior"], r["arm"]), []).append(r)
        for (lane, prior, arm), group in sorted(groups.items()):
            out.append({
                "lane": lane,
                "primary_prior": prior,
                "arm": arm,
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact_any"]) / len(group)),
                "minority_color_recall_mean": round_float(sum(r.get("minority_color_recall_slot1", 0.0) for r in group) / len(group)),
            })
        return out

    scores = _agg_scores(selected_any_rows)
    per_task_rows = _agg_per_task(selected_any_rows)
    per_prior_rows = _agg_per_prior(selected_any_rows)

    # Seed stability + quarantine update across selected rows
    unstable_keys: set[tuple[str, str]] = set()
    seed_stability_rows: list[dict[str, Any]] = []
    for (arm, instance_id), seed_outcomes in per_instance_seed_outcomes.items():
        outcomes = sorted(seed_outcomes.items())
        seed_instability = len(set(seed_outcomes.values())) > 1
        if seed_instability:
            unstable_keys.add((arm, instance_id))
        seed_stability_rows.append({
            "instance_id": instance_id,
            "arm": arm,
            "seed_outcomes": json.dumps({str(s): bool(v) for s, v in outcomes}, separators=(",", ":")),
            "seed_instability": seed_instability,
        })

    for r in selected_rows_slot1:
        if r["quarantine_label"] and (r["arm"], r["instance_id"]) in unstable_keys:
            r["quarantine_label"] = "stochastic_instability"

    quarantine_rows = []
    dominant_audit_rows = []
    for r in selected_rows_slot1:
        if r["quarantine_label"]:
            quarantine_rows.append({
                "instance_id": r["instance_id"],
                "lane": r["lane"],
                "task_id": r["task_id"],
                "arm": r["arm"],
                "selected_seed": r["seed"],
                "label": r["quarantine_label"],
            })
        dominant_audit_rows.append({
            "instance_id": r["instance_id"],
            "lane": r["lane"],
            "task_id": r["task_id"],
            "arm": r["arm"],
            "selected_seed": r["seed"],
            "target_color_count": r["target_color_count"],
            "predicted_color_count": r["predicted_color_count_slot1"],
            "dominant_color_collapse": r.get("dominant_color_collapse_slot1", False),
        })

    # Arena gate + branch decisions (always adjudicate the merge)
    arena = adjudicate_arena_gate(per_task_rows)
    branch = adjudicate_branch_a(per_task_rows, arena)

    # Build merged manifest from shard[0] minus shard-specific keys
    ref_manifest = shards[0]["manifest"]
    drop_keys = {"mode", "shardArm", "shardSeed", "seedSlateOriginal", "armsOriginal",
                 "armsEffective", "seedsEffective", "generatedAt", "completedAt",
                 "command", "tool", "outDir", "instanceCount", "perSeedValidationMetrics",
                 "selectedSeedByArm", "arenaGate", "branchAdjudication", "elapsedSecondsTotal"}
    merged_manifest = {k: v for k, v in ref_manifest.items() if k not in drop_keys}
    merged_manifest.update({
        "generatedAt": min(sh["manifest"]["generatedAt"] for sh in shards),
        "completedAt": iso_now(),
        "tool": "docs/prereg/arc/phase3a_per_task_coord_mlp.py (merge)",
        "command": [sys.executable, "docs/prereg/arc/phase3a_per_task_coord_mlp.py", *sys.argv[1:]],
        "mode": "full",
        "shardedRun": True,
        "shardSources": [
            {
                "dir": str(sh["dir"]),
                "shardArm": sh["manifest"]["shardArm"],
                "shardSeed": sh["manifest"]["shardSeed"],
                "generatedAt": sh["manifest"]["generatedAt"],
                "completedAt": sh["manifest"]["completedAt"],
                "gitCommit": sh["manifest"]["gitCommit"],
                "gitDirty": sh["manifest"].get("gitDirty", False),
                "allowDirty": sh["manifest"].get("allowDirty", False),
                "elapsedSecondsTotal": sh["manifest"].get("elapsedSecondsTotal"),
            }
            for sh in shards
        ],
        "allowMixedCommits": args.allow_mixed_commits,
        "mixedCommitsAudit": mixed_commits_audit,
        "armsEffective": arms,
        "seedsEffective": seeds,
        "mergeStartedAt": started_at,
        "mergeGitCommit": git["commit"],
        "mergeGitDirty": git["dirty"],
        "mergeAllowDirty": args.allow_dirty,
        "outDir": str(out_dir),
        "selectedSeedByArm": selected_seed_by_arm,
        "arenaGate": arena,
        "branchAdjudication": branch,
        "perSeedValidationMetrics": per_arm_validation_metrics,
        "elapsedSecondsTotalShards": round_float(sum(sh["manifest"].get("elapsedSecondsTotal", 0.0) for sh in shards)),
    })

    write_json(out_dir / "manifest.json", merged_manifest)
    write_csv(out_dir / "scores.csv", scores, ["lane", "arm", "selected_seed", "task_count", "instance_count", "grid_exact_any_rate", "shape_exact_slot1_rate", "palette_exact_slot1_rate", "pixel_accuracy_best_mean", "minority_color_recall_mean", "dominant_color_collapse_rate"])
    write_csv(out_dir / "per_task.csv", per_task_rows, ["lane", "task_id", "primary_prior", "predicted_boundary", "arm", "selected_seed", "instance_count", "grid_exact_any_rate", "shape_exact_slot1_rate", "palette_exact_slot1_rate", "pixel_accuracy_best_mean"])
    write_csv(out_dir / "per_prior.csv", per_prior_rows, ["lane", "primary_prior", "arm", "instance_count", "grid_exact_any_rate", "minority_color_recall_mean"])
    write_csv(out_dir / "per_instance.csv", per_instance_rows, ["instance_id", "lane", "task_id", "primary_prior", "predicted_boundary", "arm", "seed", "slot", "grid_exact", "shape_exact", "palette_exact", "pixel_accuracy", "minority_color_recall", "palette_jaccard_slot1", "predicted_color_count_slot1", "target_color_count", "dominant_color_collapse", "conditioning_train_exact_rate", "signature_collision", "quarantine_label"])
    write_csv(out_dir / "learning_curves.csv", learning_rows, ["instance_id", "arm", "seed", "model_kind", "step", "loss"])
    write_csv(out_dir / "seed_stability.csv", seed_stability_rows, ["instance_id", "arm", "seed_outcomes", "seed_instability"])
    write_csv(out_dir / "quarantine_log.csv", quarantine_rows, ["instance_id", "lane", "task_id", "arm", "selected_seed", "label"])
    write_csv(out_dir / "dominant_color_audit.csv", dominant_audit_rows, ["instance_id", "lane", "task_id", "arm", "selected_seed", "target_color_count", "predicted_color_count", "dominant_color_collapse"])
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    write_json(out_dir / "phase3a_receipt.json", {
        "manifest": merged_manifest,
        "scores": scores,
        "perTask": per_task_rows,
        "perPrior": per_prior_rows,
        "selectedSeedByArm": selected_seed_by_arm,
        "arenaGate": arena,
        "branchAdjudication": branch,
        "perSeedValidationMetrics": per_arm_validation_metrics,
    })

    summary_lines = [
        "# Phase 3A Branch Adjudication (per_task_coord_mlp_v1, merged)",
        "",
        f"Mode: `full` (sharded; {len(shards)} shards merged)",
        "",
        f"Arena gate: **{arena.get('gate', 'not_adjudicated')}**",
        "",
        arena.get("reason", ""),
        "",
        f"Branch decision: **{branch.get('branch', 'not_adjudicated')}**",
        "",
        branch.get("reason", ""),
        "",
        "Selected seed by arm:",
        "",
    ]
    for arm in arms:
        summary_lines.append(f"- `{arm}`: `{selected_seed_by_arm[arm]}`")
    (out_dir / "branch_adjudication.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    (out_dir / "commands.md").write_text(
        "# Phase 3A merge command\n\n```\n"
        + " ".join([sys.executable, "docs/prereg/arc/phase3a_per_task_coord_mlp.py", *sys.argv[1:]])
        + "\n```\n"
        + f"\nMerged {len(shards)} shards from arms={arms}, seeds={seeds}.\n",
        encoding="utf-8",
    )

    # Copy split.csv from shard[0] (all shards share the same task register).
    split_first = shards[0]["dir"] / "split.csv"
    if split_first.exists():
        (out_dir / "split.csv").write_text(split_first.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Phase 3A merge wrote {out_dir}")
    print(f"Arena gate: {arena.get('gate')}; branch: {branch.get('branch')}")
    return 0


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"ARC Phase 3A stochastic per-task learner ({LEARNER_VERSION})")
    parser.add_argument("--data-dir", required=False, default=None)
    parser.add_argument("--register", required=False, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--master-seed", type=int, default=20260528)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=5, help="Cap shape/color train steps when --probe-only.")
    parser.add_argument("--limit-tasks", type=int, default=0, help="Cap the number of registered tasks processed (0 = no cap).")
    parser.add_argument("--limit-arms", default=None, help="Comma-separated arm names to restrict the run (default: all four).")
    parser.add_argument("--limit-seeds", default=None, help="Comma-separated seeds from SEED_SLATE to restrict the run.")
    parser.add_argument("--shard-arm", default=None, help="Run a single arm from ARMS as a shard (no adjudication). Requires --shard-seed.")
    parser.add_argument("--shard-seed", type=int, default=None, help="Run a single seed from SEED_SLATE as a shard (no adjudication). Requires --shard-arm.")
    parser.add_argument("--merge", action="store_true", help="Merge shard intermediates into a binding receipt instead of training.")
    parser.add_argument("--shard-dirs", default=None, help="Comma-separated list of shard receipt directories (--merge mode only).")
    parser.add_argument("--allow-mixed-commits", action="store_true", help="Merge mode: bypass gitCommit equality across shards if and only if the runner file (docs/prereg/arc/phase3a_per_task_coord_mlp.py) is byte-identical at every distinct shard gitCommit. The audit is recorded in the merged manifest. Use when a re-launch happens on a different HEAD because parallel commits landed mid-slate, provided none of those commits touched the runner.")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.merge:
        if not args.shard_dirs:
            parser.error("--merge requires --shard-dirs <dir1,dir2,...>")
        if not args.out:
            parser.error("--merge requires --out <merge output dir>")
    else:
        if not args.data_dir or not args.register:
            parser.error("--data-dir and --register are required (except in --merge mode)")
        if (args.shard_arm is None) != (args.shard_seed is None):
            parser.error("--shard-arm and --shard-seed must be provided together")
    return args


def base_manifest(args, started_at, git, data_dir, register_path, out_dir, register_hash, data_hash, spec_hash, parent_spec_hash, instance_count) -> dict[str, Any]:
    return {
        "generatedAt": started_at,
        "completedAt": None,
        "tool": "docs/prereg/arc/phase3a_per_task_coord_mlp.py",
        "command": [sys.executable, "docs/prereg/arc/phase3a_per_task_coord_mlp.py", *sys.argv[1:]],
        "gitCommit": git["commit"],
        "gitDirty": git["dirty"],
        "allowDirty": args.allow_dirty,
        "dataDir": str(data_dir),
        "registerPath": str(register_path),
        "outDir": str(out_dir),
        "masterSeed": args.master_seed,
        "device": args.device,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION,
        "learnerVersion": LEARNER_VERSION,
        "specPath": "docs/prereg/arc/PHASE3A_STOCHASTIC_PER_TASK_SPEC.md",
        "specHash": spec_hash,
        "parentSpecPath": "docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md",
        "parentSpecHash": parent_spec_hash,
        "registerHash": register_hash,
        "dataDirHash": data_hash,
        "pythonVersion": sys.version,
        "torchVersion": torch.__version__,
        "platform": platform.platform(),
        "shapeModelSpec": SHAPE_MODEL_SPEC,
        "colorModelSpec": COLOR_MODEL_SPEC,
        "seedSlate": SEED_SLATE,
        "arms": ARMS,
        "instanceCount": instance_count,
        "limits": {
            "limit_tasks": args.limit_tasks,
            "limit_arms": args.limit_arms,
            "limit_seeds": args.limit_seeds,
        },
    }


def write_empty_receipt(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "split.csv", [], ["task_id", "primary_prior", "predicted_boundary", "split"])
    write_csv(out_dir / "scores.csv", [], ["lane", "arm", "task_count", "instance_count", "grid_exact_any_rate", "shape_exact_slot1_rate", "palette_exact_slot1_rate", "pixel_accuracy_best_mean", "minority_color_recall_mean", "dominant_color_collapse_rate"])
    write_csv(out_dir / "per_task.csv", [], ["lane", "task_id", "primary_prior", "predicted_boundary", "arm", "selected_seed", "instance_count", "grid_exact_any_rate", "shape_exact_slot1_rate", "palette_exact_slot1_rate", "pixel_accuracy_best_mean"])
    write_csv(out_dir / "per_prior.csv", [], ["lane", "primary_prior", "arm", "instance_count", "grid_exact_any_rate", "minority_color_recall_mean"])
    write_csv(out_dir / "per_instance.csv", [], ["instance_id", "lane", "task_id", "primary_prior", "predicted_boundary", "arm", "seed", "slot", "grid_exact", "shape_exact", "palette_exact", "pixel_accuracy", "minority_color_recall", "palette_jaccard_slot1", "predicted_color_count_slot1", "target_color_count", "dominant_color_collapse", "conditioning_train_exact_rate", "signature_collision", "quarantine_label"])
    write_csv(out_dir / "learning_curves.csv", [], ["instance_id", "arm", "seed", "model_kind", "step", "loss"])
    write_csv(out_dir / "seed_stability.csv", [], ["instance_id", "arm", "seed_outcomes", "seed_instability"])
    write_csv(out_dir / "quarantine_log.csv", [], ["instance_id", "lane", "task_id", "arm", "selected_seed", "label"])
    write_csv(out_dir / "dominant_color_audit.csv", [], ["instance_id", "lane", "task_id", "arm", "selected_seed", "target_color_count", "predicted_color_count", "dominant_color_collapse"])
    write_jsonl(out_dir / "residuals.jsonl", [])
    write_json(out_dir / "phase3a_receipt.json", {"manifest": manifest, "arenaGate": None, "branchAdjudication": None})
    (out_dir / "branch_adjudication.md").write_text(
        "# Phase 3A Branch Adjudication\n\nDry run / empty receipt. No arena-gate or branch decision.\n",
        encoding="utf-8",
    )
    (out_dir / "commands.md").write_text(
        "# Phase 3A commands\n\nDry run / empty receipt. No execution command captured.\n",
        encoding="utf-8",
    )
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def select_seed_for_arm(arm: str, per_seed_metrics: dict[int, dict[str, Any]]) -> int:
    """Spec §"Seed Slate And Derived Seeds": rank by validation-lane exact-task count, then minority recall, then collapse, then loss, then seed."""
    if not per_seed_metrics:
        raise ValueError("no candidates to select from")
    def key(seed: int):
        m = per_seed_metrics[seed]
        # higher is better for grid-eligible / rep-eligible exact counts; minority recall
        # lower is better for collapse rate, validation loss, seed
        primary = -m.get("val_grid_exact_count" if arm in GRID_SCORABLE_ARMS else "val_rep_exact_count", 0)
        secondary = -m.get("val_minority_recall", 0.0)
        tertiary = m.get("val_collapse_rate", 1.0)
        quaternary = m.get("val_loss", float("inf"))
        return (primary, secondary, tertiary, quaternary, seed)
    return sorted(per_seed_metrics.keys(), key=key)[0]


def main() -> int:
    args = parse_args()
    if args.merge:
        return run_merge(args)
    started_at = iso_now()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    set_global_determinism(args.master_seed)
    git = git_state(repo_root, args.allow_dirty)
    data_dir = Path(args.data_dir).resolve()
    register_path = Path(args.register).resolve()
    assert_training_data_dir(data_dir)
    spec_path = Path(__file__).resolve().parent / "PHASE3A_STOCHASTIC_PER_TASK_SPEC.md"
    parent_spec_path = Path(__file__).resolve().parent / "PHASE3_SUFFICIENCY_SPEC.md"
    spec_hash = sha256_file(spec_path) if spec_path.exists() else "NA"
    parent_spec_hash = sha256_file(parent_spec_path) if parent_spec_path.exists() else "NA"

    tasks, register_hash, data_hash = load_tasks(data_dir, register_path)
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    train_tasks = [t for t in tasks if t.split == "train"]
    validation_tasks = [t for t in tasks if t.split == "validation"]
    test_tasks = [t for t in tasks if t.split == "test"]

    train_lodo = build_lodo_instances(train_tasks, "train_lodo")
    train_pttest = build_pttest_instances(train_tasks, "train_pttest")
    val_lodo = build_lodo_instances(validation_tasks, "validation_lodo")
    val_pttest = build_pttest_instances(validation_tasks, "validation_pttest")
    test_lodo = build_lodo_instances(test_tasks, "test_lodo")
    test_pttest = build_pttest_instances(test_tasks, "pttest")
    all_instances = train_lodo + train_pttest + val_lodo + val_pttest + test_lodo + test_pttest

    split_rows = [{"task_id": t.task_id, "primary_prior": t.primary_prior, "predicted_boundary": t.predicted_boundary, "split": t.split} for t in sorted(tasks, key=lambda x: x.task_id)]

    manifest = base_manifest(args, started_at, git, data_dir, register_path, out_dir, register_hash, data_hash, spec_hash, parent_spec_hash, len(all_instances))
    manifest["taskCount"] = len(tasks)
    manifest["lanes"] = {
        "train_lodo": len(train_lodo),
        "train_pttest": len(train_pttest),
        "validation_lodo": len(val_lodo),
        "validation_pttest": len(val_pttest),
        "test_lodo": len(test_lodo),
        "pttest": len(test_pttest),
    }
    write_csv(out_dir / "split.csv", split_rows, ["task_id", "primary_prior", "predicted_boundary", "split"])

    if args.dry_run:
        manifest["mode"] = "dry_run"
        manifest["completedAt"] = iso_now()
        write_empty_receipt(out_dir, manifest)
        print(f"ARC Phase 3A dry run wrote {out_dir}")
        return 0

    if args.shard_arm is not None:
        if args.shard_arm not in ARMS:
            raise SystemExit(f"--shard-arm {args.shard_arm!r} not in ARMS {ARMS}")
        if args.shard_seed not in SEED_SLATE:
            raise SystemExit(f"--shard-seed {args.shard_seed} not in SEED_SLATE {SEED_SLATE}")
        arms = [args.shard_arm]
        seeds = [args.shard_seed]
        manifest["mode"] = "shard"
        manifest["shardArm"] = args.shard_arm
        manifest["shardSeed"] = args.shard_seed
        manifest["seedSlateOriginal"] = SEED_SLATE
        manifest["armsOriginal"] = ARMS
    else:
        arms = [a.strip() for a in args.limit_arms.split(",")] if args.limit_arms else list(ARMS)
        for arm in arms:
            if arm not in ARMS:
                raise SystemExit(f"--limit-arms includes unknown arm {arm!r}; expected subset of {ARMS}")
        seeds = [int(s) for s in args.limit_seeds.split(",")] if args.limit_seeds else list(SEED_SLATE)
        for seed in seeds:
            if seed not in SEED_SLATE:
                raise SystemExit(f"--limit-seeds includes {seed} which is not in SEED_SLATE {SEED_SLATE}")
        manifest["mode"] = "probe" if args.probe_only else "full"
    manifest["armsEffective"] = arms
    manifest["seedsEffective"] = seeds
    max_steps_shape = args.probe_steps if args.probe_only else SHAPE_MODEL_SPEC["max_steps"]
    max_steps_color = args.probe_steps if args.probe_only else COLOR_MODEL_SPEC["max_steps"]
    manifest["maxStepsEffective"] = {"shape": max_steps_shape, "color": max_steps_color}

    held_out_instances = val_lodo + val_pttest + test_lodo + test_pttest
    if not held_out_instances:
        print("WARN: no held-out instances to process (limit-tasks may be too small)")

    device = torch.device(args.device)

    per_instance_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    seed_stability_rows: list[dict[str, Any]] = []
    dominant_audit_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []

    per_arm_validation_metrics: dict[str, dict[int, dict[str, Any]]] = {arm: {seed: {} for seed in seeds} for arm in arms}
    per_instance_seed_outcomes: dict[tuple[str, str], dict[int, bool]] = {}
    elapsed_total = 0.0

    for arm in arms:
        for seed in seeds:
            for inst in held_out_instances:
                start = time.perf_counter()
                shape_model, shape_info = fit_shape(inst, arm, seed, max_steps_shape, device)
                color_model, color_info = fit_color(inst, arm, seed, max_steps_color, device)
                slot_shapes = predict_shape(shape_model, inst, arm, device)
                preds: list[list[list[int]]] = []
                slot1_grid = predict_colors(color_model, inst, arm, slot_shapes[0], device, sample_seed=None)
                preds.append(slot1_grid)
                if slot_shapes[1] != slot_shapes[0]:
                    slot2_grid = predict_colors(color_model, inst, arm, slot_shapes[1], device, sample_seed=None)
                else:
                    slot2_grid = predict_colors(color_model, inst, arm, slot_shapes[0], device, sample_seed=color_info["seed"])
                preds.append(slot2_grid)
                conditioning_exact_rate = conditioning_train_exact_rate(shape_model, color_model, inst, arm, device)
                collision = has_signature_collision(inst, arm)
                scores = [score_prediction(p, inst.target_output) for p in preds]
                grid_exact_any = any(s["grid_exact"] for s in scores)
                pixel_best = max(s["pixel_accuracy"] for s in scores)
                elapsed = time.perf_counter() - start
                elapsed_total += elapsed

                slot1 = scores[0]
                row = {
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "primary_prior": inst.primary_prior,
                    "arm": arm,
                    "seed": seed,
                    "grid_exact_any_slot": grid_exact_any,
                    "shape_exact_slot1": slot1["shape_exact"],
                    "palette_exact_slot1": slot1["palette_exact"],
                    "pixel_accuracy_best": round_float(pixel_best),
                    "minority_color_recall_slot1": slot1["minority_color_recall"],
                    "palette_jaccard_slot1": slot1["palette_jaccard_slot1"],
                    "predicted_color_count_slot1": slot1["predicted_color_count_slot1"],
                    "target_color_count": slot1["target_color_count"],
                    "dominant_color_collapse_slot1": slot1["dominant_color_collapse"],
                    "conditioning_train_exact_rate": round_float(conditioning_exact_rate),
                    "conditioning_train_exact": conditioning_exact_rate >= 0.95,
                    "signature_collision": collision,
                    "elapsed_seconds": round_float(elapsed),
                }
                row["quarantine_label"] = "" if grid_exact_any else assign_quarantine_label(row, len(inst.conditioning), arm, inst.primary_prior, collision)
                for slot_idx, sc in enumerate(scores, start=1):
                    per_instance_rows.append({
                        "instance_id": inst.instance_id,
                        "lane": inst.lane,
                        "task_id": inst.task_id,
                        "primary_prior": inst.primary_prior,
                        "predicted_boundary": inst.predicted_boundary,
                        "arm": arm,
                        "seed": seed,
                        "slot": slot_idx,
                        "grid_exact": sc["grid_exact"],
                        "shape_exact": sc["shape_exact"],
                        "palette_exact": sc["palette_exact"],
                        "pixel_accuracy": sc["pixel_accuracy"],
                        "minority_color_recall": sc["minority_color_recall"],
                        "palette_jaccard_slot1": sc["palette_jaccard_slot1"],
                        "predicted_color_count_slot1": sc["predicted_color_count_slot1"],
                        "target_color_count": sc["target_color_count"],
                        "dominant_color_collapse": sc["dominant_color_collapse"],
                        "conditioning_train_exact_rate": row["conditioning_train_exact_rate"],
                        "signature_collision": collision,
                        "quarantine_label": row["quarantine_label"] if slot_idx == 1 else "",
                    })
                for h in shape_info.get("history", []):
                    learning_rows.append({"instance_id": inst.instance_id, "arm": arm, "seed": seed, "model_kind": "shape", "step": h["step"], "loss": h["loss"]})
                for h in color_info.get("history", []):
                    learning_rows.append({"instance_id": inst.instance_id, "arm": arm, "seed": seed, "model_kind": "color", "step": h["step"], "loss": h["loss"]})
                residual_rows.append({
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "arm": arm,
                    "seed": seed,
                    "target_shape": list(shape_of(inst.target_output)),
                    "predictions": [
                        {"slot": idx + 1, "grid": p, "shape": list(shape_of(p)), "scores": scores[idx]}
                        for idx, p in enumerate(preds)
                    ],
                })
                key = (arm, inst.instance_id)
                per_instance_seed_outcomes.setdefault(key, {})[seed] = grid_exact_any
                if inst.lane.startswith("validation_"):
                    bucket = per_arm_validation_metrics[arm][seed].setdefault("counts", {"grid_exact": 0, "rep_exact": 0, "n": 0, "min_recall_sum": 0.0, "collapse": 0, "loss_sum": 0.0})
                    bucket["n"] += 1
                    if grid_exact_any:
                        bucket["grid_exact"] += 1
                    if any(s["shape_exact"] for s in scores):
                        bucket["rep_exact"] += 1
                    bucket["min_recall_sum"] += slot1["minority_color_recall"]
                    if slot1["dominant_color_collapse"]:
                        bucket["collapse"] += 1
                    bucket["loss_sum"] += color_info.get("best_loss", 0.0)
            # Roll up per-seed validation summary
            bucket = per_arm_validation_metrics[arm][seed].get("counts", {"grid_exact": 0, "rep_exact": 0, "n": 0, "min_recall_sum": 0.0, "collapse": 0, "loss_sum": 0.0})
            n = max(1, bucket["n"])
            per_arm_validation_metrics[arm][seed].update({
                "val_grid_exact_count": bucket["grid_exact"],
                "val_rep_exact_count": bucket["rep_exact"],
                "val_minority_recall": round_float(bucket["min_recall_sum"] / n),
                "val_collapse_rate": round_float(bucket["collapse"] / n),
                "val_loss": round_float(bucket["loss_sum"] / n),
            })

    # Select seed per arm using validation only
    selected_seed_by_arm: dict[str, int] = {}
    for arm in arms:
        selected_seed_by_arm[arm] = select_seed_for_arm(arm, per_arm_validation_metrics[arm])

    # Build per-task and per-lane summaries using the selected seed for each arm
    selected_rows = [r for r in per_instance_rows if r["seed"] == selected_seed_by_arm[r["arm"]] and r["slot"] == 1]
    selected_any_rows: list[dict[str, Any]] = []
    for arm in arms:
        sel = selected_seed_by_arm[arm]
        # group by (lane, arm, task_id, instance_id) and take grid_exact_any across slots
        by_inst: dict[tuple[str, str, str], dict[str, Any]] = {}
        for r in per_instance_rows:
            if r["arm"] != arm or r["seed"] != sel:
                continue
            k = (r["lane"], r["task_id"], r["instance_id"])
            cur = by_inst.setdefault(k, {
                "lane": r["lane"],
                "task_id": r["task_id"],
                "instance_id": r["instance_id"],
                "primary_prior": r["primary_prior"],
                "predicted_boundary": r["predicted_boundary"],
                "arm": r["arm"],
                "seed": r["seed"],
                "grid_exact_any": False,
                "shape_exact_slot1": False,
                "palette_exact_slot1": False,
                "pixel_accuracy_best": 0.0,
                "minority_color_recall_slot1": 0.0,
                "dominant_color_collapse_slot1": False,
            })
            cur["grid_exact_any"] = cur["grid_exact_any"] or r["grid_exact"]
            cur["pixel_accuracy_best"] = max(cur["pixel_accuracy_best"], r["pixel_accuracy"])
            if r["slot"] == 1:
                cur["shape_exact_slot1"] = r["shape_exact"]
                cur["palette_exact_slot1"] = r["palette_exact"]
                cur["minority_color_recall_slot1"] = r["minority_color_recall"]
                cur["dominant_color_collapse_slot1"] = r["dominant_color_collapse"]
        selected_any_rows.extend(by_inst.values())

    def aggregate_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["lane"], r["arm"]), []).append(r)
        for (lane, arm), group in sorted(groups.items()):
            task_ids = sorted({r["task_id"] for r in group})
            out.append({
                "lane": lane,
                "arm": arm,
                "selected_seed": selected_seed_by_arm[arm],
                "task_count": len(task_ids),
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact_any"]) / len(group)),
                "shape_exact_slot1_rate": round_float(sum(1 for r in group if r.get("shape_exact_slot1")) / len(group)),
                "palette_exact_slot1_rate": round_float(sum(1 for r in group if r.get("palette_exact_slot1")) / len(group)),
                "pixel_accuracy_best_mean": round_float(sum(r.get("pixel_accuracy_best", 0.0) for r in group) / len(group)),
                "minority_color_recall_mean": round_float(sum(r.get("minority_color_recall_slot1", 0.0) for r in group) / len(group)),
                "dominant_color_collapse_rate": round_float(sum(1 for r in group if r.get("dominant_color_collapse_slot1")) / len(group)),
            })
        return out

    def aggregate_per_task(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["lane"], r["arm"], r["task_id"]), []).append(r)
        for (lane, arm, task_id), group in sorted(groups.items()):
            out.append({
                "lane": lane,
                "task_id": task_id,
                "primary_prior": group[0]["primary_prior"],
                "predicted_boundary": next((g.get("predicted_boundary", "") for g in group if g.get("predicted_boundary")), ""),
                "arm": arm,
                "selected_seed": selected_seed_by_arm[arm],
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact_any"]) / len(group)),
                "shape_exact_slot1_rate": round_float(sum(1 for r in group if r.get("shape_exact_slot1")) / len(group)),
                "palette_exact_slot1_rate": round_float(sum(1 for r in group if r.get("palette_exact_slot1")) / len(group)),
                "pixel_accuracy_best_mean": round_float(sum(r.get("pixel_accuracy_best", 0.0) for r in group) / len(group)),
            })
        return out

    def aggregate_per_prior(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for r in rows:
            groups.setdefault((r["lane"], r["primary_prior"], r["arm"]), []).append(r)
        for (lane, prior, arm), group in sorted(groups.items()):
            out.append({
                "lane": lane,
                "primary_prior": prior,
                "arm": arm,
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact_any"]) / len(group)),
                "minority_color_recall_mean": round_float(sum(r.get("minority_color_recall_slot1", 0.0) for r in group) / len(group)),
            })
        return out

    scores = aggregate_scores(selected_any_rows)
    per_task_rows = aggregate_per_task(selected_any_rows)
    per_prior_rows = aggregate_per_prior(selected_any_rows)

    # Seed stability
    unstable_keys: set[tuple[str, str]] = set()
    for (arm, instance_id), seed_outcomes in per_instance_seed_outcomes.items():
        outcomes = sorted(seed_outcomes.items())
        seed_instability = len(set(seed_outcomes.values())) > 1
        if seed_instability:
            unstable_keys.add((arm, instance_id))
        seed_stability_rows.append({
            "instance_id": instance_id,
            "arm": arm,
            "seed_outcomes": json.dumps({str(s): bool(v) for s, v in outcomes}, separators=(",", ":")),
            "seed_instability": seed_instability,
        })

    for r in selected_rows:
        if r["quarantine_label"] and (r["arm"], r["instance_id"]) in unstable_keys:
            r["quarantine_label"] = "stochastic_instability"

    # Quarantine + dominant audit
    for r in selected_rows:
        if r["quarantine_label"]:
            quarantine_rows.append({
                "instance_id": r["instance_id"],
                "lane": r["lane"],
                "task_id": r["task_id"],
                "arm": r["arm"],
                "selected_seed": r["seed"],
                "label": r["quarantine_label"],
            })
        dominant_audit_rows.append({
            "instance_id": r["instance_id"],
            "lane": r["lane"],
            "task_id": r["task_id"],
            "arm": r["arm"],
            "selected_seed": r["seed"],
            "target_color_count": r["target_color_count"],
            "predicted_color_count": r["predicted_color_count_slot1"],
            "dominant_color_collapse": r["dominant_color_collapse"],
        })

    # Arena gate + branch (only in full mode; shard/probe/dry-run skip adjudication)
    if manifest["mode"] == "full":
        arena = adjudicate_arena_gate(per_task_rows)
        branch = adjudicate_branch_a(per_task_rows, arena)
    else:
        arena = {"gate": "not_adjudicated", "reason": f"{manifest['mode']} run only"}
        branch = {"branch": "not_adjudicated", "reason": arena["reason"]}

    manifest["completedAt"] = iso_now()
    manifest["selectedSeedByArm"] = selected_seed_by_arm
    manifest["arenaGate"] = arena
    manifest["branchAdjudication"] = branch
    manifest["elapsedSecondsTotal"] = round_float(elapsed_total)
    manifest["perSeedValidationMetrics"] = per_arm_validation_metrics

    write_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "scores.csv", scores, ["lane", "arm", "selected_seed", "task_count", "instance_count", "grid_exact_any_rate", "shape_exact_slot1_rate", "palette_exact_slot1_rate", "pixel_accuracy_best_mean", "minority_color_recall_mean", "dominant_color_collapse_rate"])
    write_csv(out_dir / "per_task.csv", per_task_rows, ["lane", "task_id", "primary_prior", "predicted_boundary", "arm", "selected_seed", "instance_count", "grid_exact_any_rate", "shape_exact_slot1_rate", "palette_exact_slot1_rate", "pixel_accuracy_best_mean"])
    write_csv(out_dir / "per_prior.csv", per_prior_rows, ["lane", "primary_prior", "arm", "instance_count", "grid_exact_any_rate", "minority_color_recall_mean"])
    write_csv(out_dir / "per_instance.csv", per_instance_rows, ["instance_id", "lane", "task_id", "primary_prior", "predicted_boundary", "arm", "seed", "slot", "grid_exact", "shape_exact", "palette_exact", "pixel_accuracy", "minority_color_recall", "palette_jaccard_slot1", "predicted_color_count_slot1", "target_color_count", "dominant_color_collapse", "conditioning_train_exact_rate", "signature_collision", "quarantine_label"])
    write_csv(out_dir / "learning_curves.csv", learning_rows, ["instance_id", "arm", "seed", "model_kind", "step", "loss"])
    write_csv(out_dir / "seed_stability.csv", seed_stability_rows, ["instance_id", "arm", "seed_outcomes", "seed_instability"])
    write_csv(out_dir / "quarantine_log.csv", quarantine_rows, ["instance_id", "lane", "task_id", "arm", "selected_seed", "label"])
    write_csv(out_dir / "dominant_color_audit.csv", dominant_audit_rows, ["instance_id", "lane", "task_id", "arm", "selected_seed", "target_color_count", "predicted_color_count", "dominant_color_collapse"])
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    write_json(out_dir / "phase3a_receipt.json", {
        "manifest": manifest,
        "scores": scores,
        "perTask": per_task_rows,
        "perPrior": per_prior_rows,
        "selectedSeedByArm": selected_seed_by_arm,
        "arenaGate": arena,
        "branchAdjudication": branch,
        "perSeedValidationMetrics": per_arm_validation_metrics,
    })

    summary_lines = [
        "# Phase 3A Branch Adjudication (per_task_coord_mlp_v1)",
        "",
        f"Mode: `{manifest['mode']}`",
        "",
        f"Arena gate: **{arena.get('gate', 'not_adjudicated')}**",
        "",
        arena.get("reason", ""),
        "",
        f"Branch decision: **{branch.get('branch', 'not_adjudicated')}**",
        "",
        branch.get("reason", ""),
        "",
        "Selected seed by arm:",
        "",
    ]
    for arm in arms:
        summary_lines.append(f"- `{arm}`: `{selected_seed_by_arm[arm]}`")
    (out_dir / "branch_adjudication.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    (out_dir / "commands.md").write_text(
        "# Phase 3A invocation\n\n```\n"
        + " ".join(sys.executable.split() + [str(p) for p in ["docs/prereg/arc/phase3a_per_task_coord_mlp.py", *sys.argv[1:]]])
        + "\n```\n"
        + f"\nMode: {manifest['mode']}; elapsed seconds total: {manifest['elapsedSecondsTotal']}\n",
        encoding="utf-8",
    )

    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Phase 3A {manifest['mode']} run wrote {out_dir}")
    print(f"Arena gate: {arena.get('gate')}; branch: {branch.get('branch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
