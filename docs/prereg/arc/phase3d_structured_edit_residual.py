#!/usr/bin/env python
"""ARC Phase 3D structured-edit-residual runner (structured_edit_residual_v1).

Standalone Python runner. Does not import phase3_decoder.py or
phase3a_per_task_coord_mlp.py. The arc-p3-feature-v1 encoders are copied
verbatim under a marked header; the feature schema is shared logically with
the V2 and Phase 3A lanes but its bytes are pinned here for independence.

Spec: docs/prereg/arc/PHASE3D_DIFFERENT_FRAMING_SPEC.md (filed 2026-05-28).
Parent: docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Frozen Phase 3D constants
# ============================================================================
FEATURE_SCHEMA_VERSION = "arc-p3-feature-v1"
LEARNER_VERSION = "structured_edit_residual_v1"
PROTOCOL_VERSION = "arc-p3d-structured-edit-residual-v1"
RECEIPT_SCHEMA_VERSION = "arc-p3d-structured-edit-receipt-v1"

ARMS = [
    "raw_grid_edit",
    "signature_palette_edit",
    "signature_only_edit",
    "metadata_only_edit",
]
GRID_SCORABLE_ARMS = {"raw_grid_edit", "signature_palette_edit"}
SEED_SLATE = [20260528, 20260529, 20260530, 20260531, 20260601]

MAX_H = 30
MAX_W = 30
MAX_COLORS = 10
PAD_CHANNELS = 11
METADATA_DIM = 28
SIGNATURE_HASH_DIM = 4096
SIGNATURE_VECTOR_DIM = METADATA_DIM + SIGNATURE_HASH_DIM
RAW_GRID_DIM = MAX_H * MAX_W * PAD_CHANNELS  # 9900
COORD_FEATURE_DIM = 2 + 2 + 4  # normalized + centered + boundary (8)
PATCH_DIM = 9 * PAD_CHANNELS  # 3x3 patch (99)

# Frozen per-spec §"Edit Learner"
MASK_MODEL_SPEC = {
    "hidden": 192,
    "out_dim": 1,
    "lr": 1e-3,
    "betas": [0.9, 0.99],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "max_steps": 700,
    "early_stop_patience": 120,
    "grad_clip_norm": 1.0,
    "pos_weight_min": 1.0,
    "pos_weight_max": 20.0,
    "batch_size": 512,
}
COLOR_MODEL_SPEC = {
    "hidden": 192,
    "out_dim": 10,
    "dropout": 0.05,
    "lr": 1e-3,
    "betas": [0.9, 0.99],
    "eps": 1e-8,
    "weight_decay": 1e-4,
    "max_steps": 700,
    "early_stop_patience": 120,
    "grad_clip_norm": 1.0,
    "class_weight_min": 0.25,
    "class_weight_max": 5.0,
    "batch_size": 512,
}

# Frozen per-spec §"Baseline Family"
SHAPE_RULES = [
    "same_as_input",
    "transpose_input",
    "conditioning_unanimous_output",
    "conditioning_median_delta",
    "nearest_conditioning_shape",
]
CANVAS_RULES = [
    "constant_background",
    "identity_top_left",
    "rot90_top_left",
    "rot180_top_left",
    "rot270_top_left",
    "reflect_h_top_left",
    "reflect_v_top_left",
    "transpose_top_left",
    "anti_transpose_top_left",
    "nonzero_bbox_top_left",
]

MASK_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1..0.9

# Pre-registered Phase 0 task split (mirrored from Phase 3A; frozen here)
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
# IO + hashing utilities (copied from Phase 3A)
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
# Frozen feature-v1 encoders (copied verbatim from phase3_decoder.py via
# phase3a_per_task_coord_mlp.py; do not modify without bumping
# FEATURE_SCHEMA_VERSION in BOTH places).
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
    return {"arm": arm, "metadata": metadata, "suffix": suffix}


def feature_vector(grid: list[list[int]], arm: str) -> list[float]:
    """Arm-specific input vector. Maps each non-raw arm into
    SIGNATURE_VECTOR_DIM and the raw arm into RAW_GRID_DIM."""
    if arm == "raw_grid_edit":
        return raw_grid_onehot(grid)
    rep = represent_grid(grid, arm)
    vector = [0.0] * SIGNATURE_VECTOR_DIM
    if arm in {"signature_palette_edit", "metadata_only_edit"}:
        vector[:METADATA_DIM] = rep["metadata"]
    if arm in {"signature_palette_edit", "signature_only_edit"}:
        for idx, value in rep["suffix"].items():
            vector[idx] = value
    return vector


def input_dim_for_arm(arm: str) -> int:
    return RAW_GRID_DIM if arm == "raw_grid_edit" else SIGNATURE_VECTOR_DIM


def arm_distance(arm: str, grid_a: list[list[int]], grid_b: list[list[int]]) -> float:
    """L2 distance between arm-specific input vectors (used for
    nearest_conditioning_shape candidate selection)."""
    a = feature_vector(grid_a, arm)
    b = feature_vector(grid_b, arm)
    return math.sqrt(sum((x - y) * (x - y) for x, y in zip(a, b)))


# ============================================================================
# Coordinate features (mirrored from Phase 3A)
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


def color_onehot(color: int) -> list[float]:
    out = [0.0] * PAD_CHANNELS
    if 0 <= color <= 9:
        out[color] = 1.0
    else:
        out[10] = 1.0
    return out


def baseline_patch_3x3(baseline: list[list[int]], oy: int, ox: int) -> list[float]:
    """3x3 patch of the BASELINE grid at the same (oy, ox)."""
    bh = len(baseline)
    bw = len(baseline[0]) if bh else 0
    values: list[float] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y = oy + dy
            x = ox + dx
            in_grid = 0 <= y < bh and 0 <= x < bw
            color = baseline[y][x] if in_grid else None
            for channel in range(PAD_CHANNELS):
                if in_grid:
                    values.append(1.0 if channel == color else 0.0)
                else:
                    values.append(1.0 if channel == 10 else 0.0)
    return values


def cell_features(arm: str, query_input: list[list[int]], baseline: list[list[int]], oy: int, ox: int) -> list[float]:
    """Per-cell feature row for the mask/color models.

    Includes: arm-specific input vector for the query INPUT, coord features in the
    BASELINE/output frame, baseline color one-hot at (oy,ox), 3x3 baseline patch,
    normalized input+output shape features.
    """
    base = feature_vector(query_input, arm)
    out_h = len(baseline)
    out_w = len(baseline[0]) if out_h else 0
    input_h = len(query_input)
    input_w = len(query_input[0]) if input_h else 0
    baseline_color = baseline[oy][ox] if 0 <= oy < out_h and 0 <= ox < out_w else 10
    return (
        list(base)
        + coord_features(oy, ox, out_h, out_w)
        + color_onehot(baseline_color)
        + baseline_patch_3x3(baseline, oy, ox)
        + shape_norm_features(input_h, input_w, out_h, out_w)
    )


def cell_input_dim_for_arm(arm: str) -> int:
    return input_dim_for_arm(arm) + COORD_FEATURE_DIM + PAD_CHANNELS + PATCH_DIM + 4


# ============================================================================
# Seed derivation + determinism (mirrored from Phase 3A)
# ============================================================================
def derive_seed(master_seed: int, lane: str, task_id: str, query_index: int, arm: str, model_kind: str) -> int:
    key = f"arc-p3d-structured-edit-residual-v1\0{master_seed}\0{lane}\0{task_id}\0{query_index}\0{arm}\0{model_kind}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 31 - 1)


def set_global_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ============================================================================
# Task + Instance loading (mirrored from Phase 3A)
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
# Baseline family
# ============================================================================
def modal_background(conditioning: list[dict[str, Any]]) -> int:
    """Modal color across all conditioning output cells; tie-broken by smallest color id."""
    counts: Counter[int] = Counter()
    for pair in conditioning:
        for row in pair["output"]:
            for c in row:
                counts[c] += 1
    if not counts:
        return 0
    # max by (count, -color) → highest count, smallest color on ties
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0]


def shape_for_rule(rule: str, query_input: list[list[int]], conditioning: list[dict[str, Any]], arm: str) -> tuple[int, int]:
    in_h = len(query_input)
    in_w = len(query_input[0]) if in_h else 0
    if rule == "same_as_input":
        return (in_h, in_w)
    if rule == "transpose_input":
        return (in_w, in_h)
    if rule == "conditioning_unanimous_output":
        shapes = {(len(p["output"]), len(p["output"][0])) for p in conditioning}
        if len(shapes) == 1:
            return next(iter(shapes))
        return (in_h, in_w)  # fall back to input shape
    if rule == "conditioning_median_delta":
        if not conditioning:
            return (in_h, in_w)
        dh = sorted(len(p["output"]) - len(p["input"]) for p in conditioning)
        dw = sorted(len(p["output"][0]) - len(p["input"][0]) for p in conditioning)
        m = len(dh) // 2
        median_dh = dh[m]
        median_dw = dw[m]
        return (max(1, min(MAX_H, in_h + median_dh)), max(1, min(MAX_W, in_w + median_dw)))
    if rule == "nearest_conditioning_shape":
        if not conditioning:
            return (in_h, in_w)
        best = min(conditioning, key=lambda p: arm_distance(arm, query_input, p["input"]))
        return (len(best["output"]), len(best["output"][0]))
    raise ValueError(f"unknown shape rule {rule!r}")


def canvas_for_rule(rule: str, query_input: list[list[int]], shape_hw: tuple[int, int], background: int) -> list[list[int]]:
    out_h, out_w = shape_hw
    if rule == "constant_background":
        return [[background] * out_w for _ in range(out_h)]
    source: list[list[int]]
    if rule == "identity_top_left":
        source = query_input
    elif rule == "rot90_top_left":
        source = rotate90(query_input) if query_input and query_input[0] else query_input
    elif rule == "rot180_top_left":
        source = rotate180(query_input) if query_input and query_input[0] else query_input
    elif rule == "rot270_top_left":
        source = rotate270(query_input) if query_input and query_input[0] else query_input
    elif rule == "reflect_h_top_left":
        source = reflect_horizontal(query_input)
    elif rule == "reflect_v_top_left":
        source = reflect_vertical(query_input)
    elif rule == "transpose_top_left":
        source = transpose(query_input) if query_input and query_input[0] else query_input
    elif rule == "anti_transpose_top_left":
        source = anti_transpose(query_input) if query_input and query_input[0] else query_input
    elif rule == "nonzero_bbox_top_left":
        cells = nonzero_cells(query_input)
        if not cells:
            source = query_input
        else:
            min_x = min(c["x"] for c in cells)
            max_x = max(c["x"] for c in cells)
            min_y = min(c["y"] for c in cells)
            max_y = max(c["y"] for c in cells)
            source = [
                [query_input[y][x] for x in range(min_x, max_x + 1)]
                for y in range(min_y, max_y + 1)
            ]
    else:
        raise ValueError(f"unknown canvas rule {rule!r}")
    # Paste source at top-left, padding the rest with background
    canvas = [[background] * out_w for _ in range(out_h)]
    sh = len(source)
    sw = len(source[0]) if sh else 0
    for y in range(min(sh, out_h)):
        for x in range(min(sw, out_w)):
            canvas[y][x] = source[y][x]
    return canvas


def residual_mass(baseline: list[list[int]], target: list[list[int]]) -> float:
    """Per-cell mismatch rate vs target output. Shape mismatch → 1.0."""
    if not target:
        return 1.0
    th = len(target)
    tw = len(target[0])
    if len(baseline) != th or (baseline and len(baseline[0]) != tw):
        return 1.0
    mismatches = 0
    for y in range(th):
        for x in range(tw):
            if baseline[y][x] != target[y][x]:
                mismatches += 1
    return mismatches / (th * tw)


def select_baseline_candidate(query_input: list[list[int]], conditioning: list[dict[str, Any]], arm: str) -> dict[str, Any]:
    """Per-spec §"Baseline Family": iterate every (shape_rule, canvas_rule) pair, score by mean conditioning residual, tie-break by max residual then rule index."""
    background = modal_background(conditioning)
    best: dict[str, Any] | None = None
    for s_idx, s_rule in enumerate(SHAPE_RULES):
        for c_idx, c_rule in enumerate(CANVAS_RULES):
            cond_residuals: list[float] = []
            for pair in conditioning:
                cond_shape = shape_for_rule(s_rule, pair["input"], [p for p in conditioning if p is not pair], arm)
                cond_canvas = canvas_for_rule(c_rule, pair["input"], cond_shape, background)
                cond_residuals.append(residual_mass(cond_canvas, pair["output"]))
            if not cond_residuals:
                continue
            mean_res = sum(cond_residuals) / len(cond_residuals)
            max_res = max(cond_residuals)
            key = (mean_res, max_res, c_idx, s_idx)
            if best is None or key < best["sort_key"]:
                best = {
                    "shape_rule": s_rule,
                    "canvas_rule": c_rule,
                    "background_color": background,
                    "shape_rule_index": s_idx,
                    "canvas_rule_index": c_idx,
                    "mean_conditioning_residual": round_float(mean_res),
                    "max_conditioning_residual": round_float(max_res),
                    "sort_key": key,
                }
    if best is None:
        # Fallback: same_as_input + constant_background
        return {
            "shape_rule": "same_as_input",
            "canvas_rule": "constant_background",
            "background_color": background,
            "shape_rule_index": 0,
            "canvas_rule_index": 0,
            "mean_conditioning_residual": 1.0,
            "max_conditioning_residual": 1.0,
            "sort_key": (1.0, 1.0, 0, 0),
        }
    best.pop("sort_key")
    return best


def apply_baseline(query_input: list[list[int]], candidate: dict[str, Any], conditioning: list[dict[str, Any]], arm: str) -> list[list[int]]:
    shape = shape_for_rule(candidate["shape_rule"], query_input, conditioning, arm)
    return canvas_for_rule(candidate["canvas_rule"], query_input, shape, candidate["background_color"])


def apply_edit(baseline: list[list[int]], edit_mask: list[list[bool]], edit_colors: list[list[int]]) -> list[list[int]]:
    """Reconstruct: where mask is True, overwrite baseline with edit_color."""
    out = [list(row) for row in baseline]
    for y in range(len(out)):
        for x in range(len(out[0])):
            if 0 <= y < len(edit_mask) and 0 <= x < len(edit_mask[0]) and edit_mask[y][x]:
                out[y][x] = edit_colors[y][x]
    return out


# ============================================================================
# Models
# ============================================================================
class MaskMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        h = MASK_MODEL_SPEC["hidden"]
        self.proj1 = nn.Linear(input_dim, h)
        self.norm = nn.LayerNorm(h)
        self.proj2 = nn.Linear(h, h)
        self.head = nn.Linear(h, 1)
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
        return self.head(h).squeeze(-1)


class EditColorMLP(nn.Module):
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
# Per-instance training
# ============================================================================
def build_conditioning_examples(arm: str, conditioning: list[dict[str, Any]], baselines: list[list[list[int]]]) -> tuple[list[list[float]], list[float], list[list[float]], list[int]]:
    """For each conditioning pair, generate per-cell mask rows (all cells) and
    per-cell color rows (only on edited cells)."""
    mask_X: list[list[float]] = []
    mask_y: list[float] = []
    color_X: list[list[float]] = []
    color_y: list[int] = []
    for pair, baseline in zip(conditioning, baselines):
        target = pair["output"]
        oh = len(target)
        ow = len(target[0]) if oh else 0
        bh = len(baseline)
        bw = len(baseline[0]) if bh else 0
        if oh != bh or ow != bw:
            # Shape mismatch: every cell counts as an edit in the overlapping
            # region; cells outside the baseline are skipped (mask model can't
            # be trained on them in this framing).
            for oy in range(min(oh, bh)):
                for ox in range(min(ow, bw)):
                    feats = cell_features(arm, pair["input"], baseline, oy, ox)
                    is_edit = baseline[oy][ox] != target[oy][ox]
                    mask_X.append(feats)
                    mask_y.append(1.0 if is_edit else 0.0)
                    if is_edit:
                        color_X.append(feats)
                        color_y.append(target[oy][ox])
            continue
        for oy in range(oh):
            for ox in range(ow):
                feats = cell_features(arm, pair["input"], baseline, oy, ox)
                is_edit = baseline[oy][ox] != target[oy][ox]
                mask_X.append(feats)
                mask_y.append(1.0 if is_edit else 0.0)
                if is_edit:
                    color_X.append(feats)
                    color_y.append(target[oy][ox])
    return mask_X, mask_y, color_X, color_y


def fit_mask(arm: str, mask_X: list[list[float]], mask_y: list[float], seed: int, max_steps: int, device: torch.device) -> tuple[MaskMLP | None, dict[str, Any]]:
    set_global_determinism(seed)
    if not mask_X:
        return None, {"steps": 0, "best_loss": float("inf"), "seed": seed, "rows": 0, "edit_count": 0}
    edit_count = int(sum(mask_y))
    no_edit_count = len(mask_y) - edit_count
    pos_weight = max(
        MASK_MODEL_SPEC["pos_weight_min"],
        min(MASK_MODEL_SPEC["pos_weight_max"], no_edit_count / max(1, edit_count)),
    )
    model = MaskMLP(cell_input_dim_for_arm(arm)).to(device)
    X = torch.tensor(mask_X, dtype=torch.float32, device=device)
    Y = torch.tensor(mask_y, dtype=torch.float32, device=device)
    W = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=MASK_MODEL_SPEC["lr"],
        betas=tuple(MASK_MODEL_SPEC["betas"]),
        eps=MASK_MODEL_SPEC["eps"],
        weight_decay=MASK_MODEL_SPEC["weight_decay"],
    )
    best_loss = float("inf")
    patience = 0
    history: list[dict[str, Any]] = []
    batch_size = MASK_MODEL_SPEC["batch_size"]
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
        loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=W)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), MASK_MODEL_SPEC["grad_clip_norm"])
        optim.step()
        loss_val = float(loss.detach().cpu().item())
        history.append({"step": step, "loss": round_float(loss_val)})
        if loss_val < best_loss - 1e-6:
            best_loss = loss_val
            patience = 0
        else:
            patience += 1
        if patience >= MASK_MODEL_SPEC["early_stop_patience"]:
            break
    return model, {"steps": len(history), "best_loss": round_float(best_loss), "seed": seed, "rows": n_rows, "edit_count": edit_count, "pos_weight": pos_weight, "history": history}


def fit_color(arm: str, color_X: list[list[float]], color_y: list[int], seed: int, max_steps: int, device: torch.device) -> tuple[EditColorMLP | None, dict[str, Any]]:
    set_global_determinism(seed)
    if not color_X:
        # Per-spec: "if no edited conditioning cells exist, the color model is skipped"
        return None, {"steps": 0, "best_loss": float("inf"), "seed": seed, "rows": 0, "skipped": True}
    color_counts: Counter[int] = Counter(color_y)
    max_count = max(color_counts.values())
    weights = [1.0] * COLOR_MODEL_SPEC["out_dim"]
    for c in range(COLOR_MODEL_SPEC["out_dim"]):
        n = color_counts.get(c, 0)
        if n > 0:
            w = math.sqrt(max_count / n)
            w = max(COLOR_MODEL_SPEC["class_weight_min"], min(COLOR_MODEL_SPEC["class_weight_max"], w))
            weights[c] = w
    model = EditColorMLP(cell_input_dim_for_arm(arm)).to(device)
    X = torch.tensor(color_X, dtype=torch.float32, device=device)
    Y = torch.tensor(color_y, dtype=torch.long, device=device)
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
    return model, {"steps": len(history), "best_loss": round_float(best_loss), "seed": seed, "rows": n_rows, "skipped": False, "history": history}


def predict_mask_probs(model: MaskMLP, arm: str, query_input: list[list[int]], baseline: list[list[int]], device: torch.device) -> list[list[float]]:
    h = len(baseline)
    w = len(baseline[0]) if h else 0
    if model is None or h == 0 or w == 0:
        return [[0.0] * w for _ in range(h)]
    rows: list[list[float]] = []
    for oy in range(h):
        for ox in range(w):
            rows.append(cell_features(arm, query_input, baseline, oy, ox))
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().tolist()
    out: list[list[float]] = []
    idx = 0
    for _ in range(h):
        out.append([round_float(p) for p in probs[idx:idx + w]])
        idx += w
    return out


def predict_edit_colors(model: EditColorMLP | None, arm: str, query_input: list[list[int]], baseline: list[list[int]], mask: list[list[bool]], device: torch.device) -> list[list[int]]:
    h = len(baseline)
    w = len(baseline[0]) if h else 0
    out_colors = [list(row) for row in baseline]  # default to baseline color (no edit)
    if model is None:
        return out_colors
    rows: list[list[float]] = []
    coords: list[tuple[int, int]] = []
    for oy in range(h):
        for ox in range(w):
            if mask[oy][ox]:
                rows.append(cell_features(arm, query_input, baseline, oy, ox))
                coords.append((oy, ox))
    if not rows:
        return out_colors
    x = torch.tensor(rows, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        argmax = logits.argmax(dim=-1).cpu().tolist()
    for (oy, ox), c in zip(coords, argmax):
        out_colors[oy][ox] = c
    return out_colors


def select_mask_threshold(probs_per_pair: list[list[list[float]]], baselines: list[list[list[int]]], targets: list[list[list[int]]], colors_per_pair: list[list[list[int]]]) -> tuple[float, dict[str, Any]]:
    """Sweep MASK_THRESHOLDS, pick the one that maximizes conditioning exact reconstruction.

    Tie-break order (per spec §"Threshold selection"):
      1. higher edit-mask F1
      2. lower predicted edit mass
      3. closer to 0.50
    """
    if not probs_per_pair:
        return 0.5, {"tested": [], "selected_metric": None}
    best: tuple[Any, float] | None = None
    audit: list[dict[str, Any]] = []
    for thr in MASK_THRESHOLDS:
        exact = 0
        f1_total = 0.0
        mass_total = 0.0
        n = 0
        for probs, baseline, target, colors in zip(probs_per_pair, baselines, targets, colors_per_pair):
            h = len(baseline)
            w = len(baseline[0]) if h else 0
            mask = [[probs[y][x] >= thr for x in range(w)] for y in range(h)]
            pred = apply_edit(baseline, mask, colors)
            if grid_equal(pred, target):
                exact += 1
            tp, fp, fn = 0, 0, 0
            th, tw = len(target), len(target[0]) if target else 0
            for y in range(min(h, th)):
                for x in range(min(w, tw)):
                    target_edit = baseline[y][x] != target[y][x]
                    pred_edit = mask[y][x]
                    if pred_edit and target_edit:
                        tp += 1
                    elif pred_edit and not target_edit:
                        fp += 1
                    elif not pred_edit and target_edit:
                        fn += 1
            denom = 2 * tp + fp + fn
            f1 = (2 * tp / denom) if denom > 0 else 0.0
            f1_total += f1
            mass_total += sum(1 for row in mask for v in row if v) / max(1, h * w)
            n += 1
        avg_f1 = f1_total / max(1, n)
        avg_mass = mass_total / max(1, n)
        audit.append({"threshold": thr, "conditioning_exact": exact, "avg_f1": round_float(avg_f1), "avg_mass": round_float(avg_mass)})
        # Sort key: (-exact, -avg_f1, +avg_mass, |thr - 0.5|)
        key = (-exact, -avg_f1, avg_mass, abs(thr - 0.5))
        if best is None or key < best[0]:
            best = (key, thr)
    return best[1], {"tested": audit, "selected_threshold": best[1]}


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
    correct = sum(1 for y in range(th) for x in range(tw) if pred[y][x] == target[y][x])
    return correct / total


def edit_metrics(mask: list[list[bool]], baseline: list[list[int]], target: list[list[int]]) -> dict[str, Any]:
    th, tw = shape_of(target)
    bh, bw = shape_of(baseline)
    tp = fp = fn = tn = 0
    minority_total = 0
    minority_hit = 0
    target_modal = Counter(c for row in target for c in row).most_common(1)[0][0] if target else 0
    for y in range(min(th, bh, len(mask))):
        for x in range(min(tw, bw, len(mask[0]) if mask else 0)):
            target_edit = baseline[y][x] != target[y][x]
            pred_edit = mask[y][x]
            if target_edit and target[y][x] != target_modal:
                minority_total += 1
                if pred_edit:
                    minority_hit += 1
            if pred_edit and target_edit:
                tp += 1
            elif pred_edit and not target_edit:
                fp += 1
            elif not pred_edit and target_edit:
                fn += 1
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    over_edit = fp / (fp + tn) if (fp + tn) else 0.0
    under_edit = fn / (fn + tp) if (fn + tp) else 0.0
    target_edit_mass = sum(1 for y in range(th) for x in range(tw) if y < bh and x < bw and baseline[y][x] != target[y][x]) / max(1, th * tw)
    predicted_edit_mass = sum(1 for row in mask for v in row if v) / max(1, len(mask) * (len(mask[0]) if mask else 0))
    minority_recall = (minority_hit / minority_total) if minority_total else 1.0
    return {
        "edit_mask_precision": round_float(precision),
        "edit_mask_recall": round_float(recall),
        "edit_mask_f1": round_float(f1),
        "over_edit_rate": round_float(over_edit),
        "under_edit_rate": round_float(under_edit),
        "target_edit_mass": round_float(target_edit_mass),
        "predicted_edit_mass": round_float(predicted_edit_mass),
        "minority_edit_recall": round_float(minority_recall),
    }


def edit_color_accuracy(mask: list[list[bool]], pred_colors: list[list[int]], target: list[list[int]]) -> float:
    th, tw = shape_of(target)
    cells = 0
    correct = 0
    for y in range(min(th, len(mask))):
        for x in range(min(tw, len(mask[0]) if mask else 0)):
            if mask[y][x]:
                cells += 1
                if y < len(pred_colors) and x < len(pred_colors[0]) and pred_colors[y][x] == target[y][x]:
                    correct += 1
    return (correct / cells) if cells else 1.0


def assign_quarantine_label(record: dict[str, Any], conditioning_n: int) -> str:
    if record["grid_exact"]:
        return ""
    if not record["shape_exact"]:
        return "baseline_shape_failure"
    if record["baseline_residual_mass"] > 0.50:
        return "baseline_canvas_failure"
    if record["edit_mask_recall"] < 0.25:
        return "edit_mask_underdetection"
    if record["over_edit_rate"] > 0.50:
        return "edit_mask_overedit"
    if record["edit_mask_f1"] >= 0.50 and record["edit_color_accuracy"] < 0.50:
        return "edit_color_failure"
    if conditioning_n < 3:
        return "conditioning_starvation"
    if record["copy_prior_absent"]:
        return "copy_prior_absent"
    return "palette_lift_failure"


# ============================================================================
# Arena gate + Branch D adjudication
# ============================================================================
def nonbaseline_exact_task_count(per_task_rows: list[dict[str, Any]], lane: str, arm: str) -> int:
    return sum(
        1
        for row in per_task_rows
        if row["lane"] == lane and row["arm"] == arm and float(row.get("nonbaseline_exact_any_rate") or 0.0) > 0.010
    )


def adjudicate_arena_gate(per_task_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lodo = nonbaseline_exact_task_count(per_task_rows, "test_lodo", "raw_grid_edit")
    pt = nonbaseline_exact_task_count(per_task_rows, "pttest", "raw_grid_edit")
    if lodo >= 1 and pt >= 1:
        return {"gate": "raw_grid_edit_arena_open", "test_lodo_nonbaseline_exact_tasks": lodo, "pttest_nonbaseline_exact_tasks": pt}
    return {"gate": "branch_d_full_grid_edit_floor", "test_lodo_nonbaseline_exact_tasks": lodo, "pttest_nonbaseline_exact_tasks": pt}


def adjudicate_branch_d(per_task_rows: list[dict[str, Any]], per_lane_rows: list[dict[str, Any]], arena: dict[str, Any]) -> dict[str, Any]:
    if arena["gate"] != "raw_grid_edit_arena_open":
        return {
            "branch": "branch_d_full_grid_edit_floor",
            "reason": "raw_grid_edit did not open the non-baseline arena (>=1 non-baseline exact task on each held-out lane required)",
        }
    raw_lodo = nonbaseline_exact_task_count(per_task_rows, "test_lodo", "raw_grid_edit")
    raw_pt = nonbaseline_exact_task_count(per_task_rows, "pttest", "raw_grid_edit")
    sig_lodo = nonbaseline_exact_task_count(per_task_rows, "test_lodo", "signature_palette_edit")
    sig_pt = nonbaseline_exact_task_count(per_task_rows, "pttest", "signature_palette_edit")
    # Pull mask F1 + minority + over_edit by lane/arm
    by_la = {(r["lane"], r["arm"]): r for r in per_lane_rows}
    raw_lodo_r = by_la.get(("test_lodo", "raw_grid_edit"), {})
    sig_lodo_r = by_la.get(("test_lodo", "signature_palette_edit"), {})
    raw_pt_r = by_la.get(("pttest", "raw_grid_edit"), {})
    sig_pt_r = by_la.get(("pttest", "signature_palette_edit"), {})
    def fnum(d, k):
        return float(d.get(k) or 0.0)
    f1_gap_lodo = fnum(raw_lodo_r, "edit_mask_f1_mean") - fnum(sig_lodo_r, "edit_mask_f1_mean")
    f1_gap_pt = fnum(raw_pt_r, "edit_mask_f1_mean") - fnum(sig_pt_r, "edit_mask_f1_mean")
    min_gap_lodo = fnum(raw_lodo_r, "minority_edit_recall_mean") - fnum(sig_lodo_r, "minority_edit_recall_mean")
    min_gap_pt = fnum(raw_pt_r, "minority_edit_recall_mean") - fnum(sig_pt_r, "minority_edit_recall_mean")
    over_gap_lodo = fnum(sig_lodo_r, "over_edit_rate_mean") - fnum(raw_lodo_r, "over_edit_rate_mean")
    over_gap_pt = fnum(sig_pt_r, "over_edit_rate_mean") - fnum(raw_pt_r, "over_edit_rate_mean")
    if (
        sig_lodo >= 1 and sig_pt >= 1
        and (raw_lodo - sig_lodo) <= 1 and (raw_pt - sig_pt) <= 1
        and f1_gap_lodo <= 0.10 and f1_gap_pt <= 0.10
        and min_gap_lodo <= 0.10 and min_gap_pt <= 0.10
        and over_gap_lodo <= 0.10 and over_gap_pt <= 0.10
    ):
        return {
            "branch": "branch_d_support",
            "reason": "signature_palette_edit opens the arena and meets all support thresholds (exact task delta <= 1, F1 gap <= 0.10, minority recall gap <= 0.10, over-edit gap <= 0.10)",
            "raw_grid_edit": {"test_lodo": raw_lodo, "pttest": raw_pt},
            "signature_palette_edit": {"test_lodo": sig_lodo, "pttest": sig_pt},
        }
    return {
        "branch": "branch_d_bounded_failure",
        "reason": "raw_grid_edit opened the non-baseline arena but signature_palette_edit did not satisfy the support thresholds",
        "raw_grid_edit": {"test_lodo": raw_lodo, "pttest": raw_pt},
        "signature_palette_edit": {"test_lodo": sig_lodo, "pttest": sig_pt},
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


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def assert_shard_consistency(shards: list[dict[str, Any]], repo_root: Path | None = None, allow_mixed_commits: bool = False) -> dict[str, Any] | None:
    """Mirror Phase 3A: every shard must share schema/spec/register/data/model fingerprints.

    Under `allow_mixed_commits=True`, `gitCommit` / `specHash` / `parentSpecHash`
    may differ; the runner file content is audited across distinct gitCommits
    and the audit dict is returned for the merged manifest. Runner SHA
    differences print a WARN but do not fail (the operator's override is the
    trust marker; the audit makes the divergence visible).
    """
    if len(shards) < 2:
        return None
    ref = shards[0]["manifest"]
    keys = [
        "featureSchemaVersion", "protocolVersion", "receiptSchemaVersion", "learnerVersion",
        "registerHash", "dataDirHash",
        "registerPath", "dataDir",
        "maskModelSpec", "colorModelSpec", "shapeRules", "canvasRules", "maskThresholds",
        "seedSlate", "arms",
        "maxStepsEffective",
    ]
    if not allow_mixed_commits:
        keys.extend(["gitCommit", "specHash", "parentSpecHash"])
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
    runner_path = "docs/prereg/arc/phase3d_structured_edit_residual.py"
    runner_shas: dict[str, str] = {}
    for c in distinct_commits:
        try:
            blob = subprocess.check_output(["git", "show", f"{c.lower()}:{runner_path}"], cwd=str(repo_root))
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"--allow-mixed-commits: cannot read {runner_path} at gitCommit {c}: {exc}")
        runner_shas[c] = hashlib.sha256(blob).hexdigest().upper()
    unique = sorted(set(runner_shas.values()))
    runner_identical = len(unique) == 1
    if runner_identical:
        print(f"--allow-mixed-commits: verified {runner_path} byte-identical across {len(distinct_commits)} commits")
    else:
        print(
            f"--allow-mixed-commits: WARN — {runner_path} differs across {len(distinct_commits)} commits "
            f"({len(unique)} distinct hashes). Shard-time computational contract "
            "(featureSchemaVersion, protocolVersion, learnerVersion, maskModelSpec, "
            "colorModelSpec, shapeRules, canvasRules, maskThresholds) IS equal across "
            "all shards — audit recorded for review."
        )
    return {
        "auditedFile": runner_path,
        "distinctCommits": distinct_commits,
        "runnerSha256ByCommit": runner_shas,
        "distinctRunnerSha256": unique,
        "runnerIdenticalAcrossCommits": runner_identical,
        "specHashByCommit": {sh["manifest"]["gitCommit"]: sh["manifest"].get("specHash") for sh in shards},
        "parentSpecHashByCommit": {sh["manifest"]["gitCommit"]: sh["manifest"].get("parentSpecHash") for sh in shards},
    }


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
            "baseline_sel_rows": read_csv_dicts(d / "baseline_selection.csv"),
            "edit_metrics_rows": read_csv_dicts(d / "edit_metrics.csv"),
        })

    mixed_audit = assert_shard_consistency(shards, repo_root=repo_root, allow_mixed_commits=args.allow_mixed_commits)
    shards.sort(key=lambda s: (s["manifest"]["shardArm"], s["manifest"]["shardSeed"]))

    # Concatenate raw shard outputs.
    per_instance_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    baseline_sel_rows: list[dict[str, Any]] = []
    edit_metrics_rows: list[dict[str, Any]] = []
    for sh in shards:
        for row in sh["per_instance_rows"]:
            coerced = dict(row)
            for col, parse in (
                ("seed", int),
                ("background_color", int),
                ("grid_exact", _parse_bool),
                ("baseline_exact", _parse_bool),
                ("nonbaseline_exact", _parse_bool),
                ("shape_exact", _parse_bool),
                ("palette_exact", _parse_bool),
                ("pixel_accuracy", float),
                ("baseline_residual_mass", float),
                ("edit_mask_precision", float),
                ("edit_mask_recall", float),
                ("edit_mask_f1", float),
                ("over_edit_rate", float),
                ("under_edit_rate", float),
                ("target_edit_mass", float),
                ("predicted_edit_mass", float),
                ("edit_color_accuracy", float),
                ("minority_edit_recall", float),
                ("selected_threshold", float),
                ("copy_prior_absent", _parse_bool),
            ):
                if col in coerced:
                    try:
                        coerced[col] = parse(coerced[col])
                    except (ValueError, TypeError):
                        pass
            per_instance_rows.append(coerced)
        learning_rows.extend(sh["learning_rows"])
        residual_rows.extend(sh["residual_rows"])
        baseline_sel_rows.extend(sh["baseline_sel_rows"])
        edit_metrics_rows.extend(sh["edit_metrics_rows"])

    # Reconstruct per-arm validation metrics + seed outcomes from per-instance rows + manifests.
    arms_present: list[str] = []
    seeds_present: set[int] = set()
    per_arm_validation_metrics: dict[str, dict[int, dict[str, Any]]] = {}
    per_instance_seed_outcomes: dict[tuple[str, str], dict[int, bool]] = {}
    for r in per_instance_rows:
        seeds_present.add(r["seed"])
        if r["arm"] not in arms_present:
            arms_present.append(r["arm"])
        per_arm_validation_metrics.setdefault(r["arm"], {}).setdefault(r["seed"], {})
        per_instance_seed_outcomes.setdefault((r["arm"], r["instance_id"]), {})[r["seed"]] = r["nonbaseline_exact"]
        if r.get("lane", "").startswith("validation_"):
            bucket = per_arm_validation_metrics[r["arm"]][r["seed"]].setdefault("counts", {
                "nonbaseline_exact": 0, "n": 0,
                "f1_sum": 0.0, "min_recall_sum": 0.0, "over_edit_sum": 0.0, "loss_sum": 0.0,
            })
            bucket["n"] += 1
            if r["nonbaseline_exact"]:
                bucket["nonbaseline_exact"] += 1
            bucket["f1_sum"] += r["edit_mask_f1"]
            bucket["min_recall_sum"] += r["minority_edit_recall"]
            bucket["over_edit_sum"] += r["over_edit_rate"]
    # Pull val_loss from each shard manifest.
    for sh in shards:
        sm = sh["manifest"]
        arm = sm["shardArm"]
        seed = sm["shardSeed"]
        sm_metrics = sm.get("perSeedValidationMetrics", {}).get(arm, {}).get(str(seed), {})
        per_arm_validation_metrics[arm][seed]["val_loss"] = sm_metrics.get("val_loss", float("inf"))
    for arm, by_seed in per_arm_validation_metrics.items():
        for seed, m in by_seed.items():
            bucket = m.get("counts", {"nonbaseline_exact": 0, "n": 0, "f1_sum": 0.0, "min_recall_sum": 0.0, "over_edit_sum": 0.0, "loss_sum": 0.0})
            n = max(1, bucket["n"])
            m["val_nonbaseline_exact_count"] = bucket["nonbaseline_exact"]
            m["val_edit_mask_f1"] = round_float(bucket["f1_sum"] / n)
            m["val_minority_edit_recall"] = round_float(bucket["min_recall_sum"] / n)
            m["val_over_edit_rate"] = round_float(bucket["over_edit_sum"] / n)

    arms = sorted(arms_present, key=lambda a: ARMS.index(a))
    seeds = sorted(seeds_present)

    selected_seed_by_arm = {arm: select_seed_for_arm(arm, per_arm_validation_metrics[arm]) for arm in arms}
    selected_rows = [r for r in per_instance_rows if r["seed"] == selected_seed_by_arm[r["arm"]]]

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
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact"]) / len(group)),
                "baseline_exact_any_rate": round_float(sum(1 for r in group if r["baseline_exact"]) / len(group)),
                "nonbaseline_exact_any_rate": round_float(sum(1 for r in group if r["nonbaseline_exact"]) / len(group)),
                "shape_exact_rate": round_float(sum(1 for r in group if r["shape_exact"]) / len(group)),
                "palette_exact_rate": round_float(sum(1 for r in group if r["palette_exact"]) / len(group)),
                "pixel_accuracy_mean": round_float(sum(r["pixel_accuracy"] for r in group) / len(group)),
                "edit_mask_f1_mean": round_float(sum(r["edit_mask_f1"] for r in group) / len(group)),
                "minority_edit_recall_mean": round_float(sum(r["minority_edit_recall"] for r in group) / len(group)),
                "over_edit_rate_mean": round_float(sum(r["over_edit_rate"] for r in group) / len(group)),
                "predicted_edit_mass_mean": round_float(sum(r["predicted_edit_mass"] for r in group) / len(group)),
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
                "predicted_boundary": group[0].get("predicted_boundary", ""),
                "arm": arm,
                "selected_seed": selected_seed_by_arm[arm],
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact"]) / len(group)),
                "nonbaseline_exact_any_rate": round_float(sum(1 for r in group if r["nonbaseline_exact"]) / len(group)),
                "baseline_exact_any_rate": round_float(sum(1 for r in group if r["baseline_exact"]) / len(group)),
                "shape_exact_rate": round_float(sum(1 for r in group if r["shape_exact"]) / len(group)),
                "palette_exact_rate": round_float(sum(1 for r in group if r["palette_exact"]) / len(group)),
                "pixel_accuracy_mean": round_float(sum(r["pixel_accuracy"] for r in group) / len(group)),
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
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact"]) / len(group)),
                "nonbaseline_exact_any_rate": round_float(sum(1 for r in group if r["nonbaseline_exact"]) / len(group)),
                "edit_mask_f1_mean": round_float(sum(r["edit_mask_f1"] for r in group) / len(group)),
            })
        return out

    scores = _agg_scores(selected_rows)
    per_task_rows_agg = _agg_per_task(selected_rows)
    per_prior_rows = _agg_per_prior(selected_rows)

    # Seed stability
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

    for r in selected_rows:
        if r["quarantine_label"] and (r["arm"], r["instance_id"]) in unstable_keys:
            r["quarantine_label"] = "stochastic_instability"

    quarantine_rows: list[dict[str, Any]] = []
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

    arena = adjudicate_arena_gate(per_task_rows_agg)
    branch = adjudicate_branch_d(per_task_rows_agg, scores, arena)

    ref_manifest = shards[0]["manifest"]
    drop_keys = {"mode", "shardArm", "shardSeed", "seedSlateOriginal", "armsOriginal",
                 "armsEffective", "seedsEffective", "generatedAt", "completedAt",
                 "command", "tool", "outDir", "instanceCount", "perSeedValidationMetrics",
                 "selectedSeedByArm", "arenaGate", "branchAdjudication", "elapsedSecondsTotal"}
    merged_manifest = {k: v for k, v in ref_manifest.items() if k not in drop_keys}
    merged_manifest.update({
        "generatedAt": min(sh["manifest"]["generatedAt"] for sh in shards),
        "completedAt": iso_now(),
        "tool": "docs/prereg/arc/phase3d_structured_edit_residual.py (merge)",
        "command": [sys.executable, "docs/prereg/arc/phase3d_structured_edit_residual.py", *sys.argv[1:]],
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
        "allowMixedCommits": args.allow_mixed_commits,
        "mixedCommitsAudit": mixed_audit,
    })

    write_json(out_dir / "manifest.json", merged_manifest)
    write_csv(out_dir / "scores.csv", scores, SCORE_COLS)
    write_csv(out_dir / "per_task.csv", per_task_rows_agg, PER_TASK_COLS)
    write_csv(out_dir / "per_prior.csv", per_prior_rows, PER_PRIOR_COLS)
    write_csv(out_dir / "per_instance.csv", per_instance_rows, PER_INSTANCE_COLS)
    write_csv(out_dir / "baseline_selection.csv", baseline_sel_rows, BASELINE_SEL_COLS)
    write_csv(out_dir / "edit_metrics.csv", edit_metrics_rows, EDIT_METRICS_COLS)
    write_csv(out_dir / "learning_curves.csv", learning_rows, LEARNING_COLS)
    write_csv(out_dir / "seed_stability.csv", seed_stability_rows, SEED_STABILITY_COLS)
    write_csv(out_dir / "quarantine_log.csv", quarantine_rows, QUARANTINE_COLS)
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    write_json(out_dir / "phase3d_receipt.json", {
        "manifest": merged_manifest,
        "scores": scores,
        "perTask": per_task_rows_agg,
        "perPrior": per_prior_rows,
        "selectedSeedByArm": selected_seed_by_arm,
        "arenaGate": arena,
        "branchAdjudication": branch,
        "perSeedValidationMetrics": per_arm_validation_metrics,
    })

    summary_lines = [
        "# Phase 3D Branch Adjudication (structured_edit_residual_v1, merged)",
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
        "# Phase 3D merge command\n\n```\n"
        + " ".join([sys.executable, "docs/prereg/arc/phase3d_structured_edit_residual.py", *sys.argv[1:]])
        + "\n```\n"
        + f"\nMerged {len(shards)} shards from arms={arms}, seeds={seeds}.\n",
        encoding="utf-8",
    )

    split_first = shards[0]["dir"] / "split.csv"
    if split_first.exists():
        (out_dir / "split.csv").write_text(split_first.read_text(encoding="utf-8"), encoding="utf-8")

    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Phase 3D merge wrote {out_dir}")
    print(f"Arena gate: {arena.get('gate')}; branch: {branch.get('branch')}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"ARC Phase 3D structured-edit-residual ({LEARNER_VERSION})")
    parser.add_argument("--data-dir", required=False, default=None)
    parser.add_argument("--register", required=False, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--master-seed", type=int, default=20260528)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-steps", type=int, default=5)
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--limit-arms", default=None)
    parser.add_argument("--limit-seeds", default=None)
    parser.add_argument("--shard-arm", default=None, help="Run a single arm from ARMS as a shard (no adjudication). Requires --shard-seed.")
    parser.add_argument("--shard-seed", type=int, default=None, help="Run a single seed from SEED_SLATE as a shard (no adjudication). Requires --shard-arm.")
    parser.add_argument("--merge", action="store_true", help="Merge shard intermediates into a binding receipt instead of training.")
    parser.add_argument("--shard-dirs", default=None, help="Comma-separated list of shard receipt directories (--merge mode only).")
    parser.add_argument("--allow-mixed-commits", action="store_true", help="Merge mode: bypass gitCommit / specHash / parentSpecHash equality across shards if the runner file content is verified across all distinct shard gitCommits. The audit is recorded in the merged manifest.")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    if args.merge:
        if not args.shard_dirs:
            parser.error("--merge requires --shard-dirs <dir1,dir2,...>")
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
        "tool": "docs/prereg/arc/phase3d_structured_edit_residual.py",
        "command": [sys.executable, "docs/prereg/arc/phase3d_structured_edit_residual.py", *sys.argv[1:]],
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
        "specPath": "docs/prereg/arc/PHASE3D_DIFFERENT_FRAMING_SPEC.md",
        "specHash": spec_hash,
        "parentSpecPath": "docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md",
        "parentSpecHash": parent_spec_hash,
        "registerHash": register_hash,
        "dataDirHash": data_hash,
        "pythonVersion": sys.version,
        "torchVersion": torch.__version__,
        "platform": platform.platform(),
        "maskModelSpec": MASK_MODEL_SPEC,
        "colorModelSpec": COLOR_MODEL_SPEC,
        "shapeRules": SHAPE_RULES,
        "canvasRules": CANVAS_RULES,
        "maskThresholds": MASK_THRESHOLDS,
        "seedSlate": SEED_SLATE,
        "arms": ARMS,
        "instanceCount": instance_count,
        "limits": {
            "limit_tasks": args.limit_tasks,
            "limit_arms": args.limit_arms,
            "limit_seeds": args.limit_seeds,
        },
    }


PER_INSTANCE_COLS = [
    "instance_id", "lane", "task_id", "primary_prior", "predicted_boundary", "arm", "seed",
    "shape_rule", "canvas_rule", "background_color",
    "grid_exact", "baseline_exact", "nonbaseline_exact",
    "shape_exact", "palette_exact", "pixel_accuracy", "baseline_residual_mass",
    "edit_mask_precision", "edit_mask_recall", "edit_mask_f1",
    "over_edit_rate", "under_edit_rate",
    "target_edit_mass", "predicted_edit_mass", "edit_color_accuracy",
    "minority_edit_recall", "selected_threshold", "copy_prior_absent",
    "quarantine_label",
]
PER_TASK_COLS = ["lane", "task_id", "primary_prior", "predicted_boundary", "arm", "selected_seed", "instance_count", "grid_exact_any_rate", "nonbaseline_exact_any_rate", "baseline_exact_any_rate", "shape_exact_rate", "palette_exact_rate", "pixel_accuracy_mean"]
PER_PRIOR_COLS = ["lane", "primary_prior", "arm", "instance_count", "grid_exact_any_rate", "nonbaseline_exact_any_rate", "edit_mask_f1_mean"]
SCORE_COLS = ["lane", "arm", "selected_seed", "task_count", "instance_count", "grid_exact_any_rate", "baseline_exact_any_rate", "nonbaseline_exact_any_rate", "shape_exact_rate", "palette_exact_rate", "pixel_accuracy_mean", "edit_mask_f1_mean", "minority_edit_recall_mean", "over_edit_rate_mean", "predicted_edit_mass_mean"]
BASELINE_SEL_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "shape_rule", "canvas_rule", "background_color", "mean_conditioning_residual", "max_conditioning_residual"]
EDIT_METRICS_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "selected_threshold", "edit_mask_f1", "edit_mask_precision", "edit_mask_recall", "minority_edit_recall", "over_edit_rate", "under_edit_rate", "edit_color_accuracy", "predicted_edit_mass", "target_edit_mass"]
LEARNING_COLS = ["instance_id", "arm", "seed", "model_kind", "step", "loss"]
SEED_STABILITY_COLS = ["instance_id", "arm", "seed_outcomes", "seed_instability"]
QUARANTINE_COLS = ["instance_id", "lane", "task_id", "arm", "selected_seed", "label"]
SPLIT_COLS = ["task_id", "primary_prior", "predicted_boundary", "split"]


def write_empty_receipt(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(out_dir / "manifest.json", manifest)
    write_csv(out_dir / "split.csv", [], SPLIT_COLS)
    write_csv(out_dir / "scores.csv", [], SCORE_COLS)
    write_csv(out_dir / "per_task.csv", [], PER_TASK_COLS)
    write_csv(out_dir / "per_prior.csv", [], PER_PRIOR_COLS)
    write_csv(out_dir / "per_instance.csv", [], PER_INSTANCE_COLS)
    write_csv(out_dir / "baseline_selection.csv", [], BASELINE_SEL_COLS)
    write_csv(out_dir / "edit_metrics.csv", [], EDIT_METRICS_COLS)
    write_csv(out_dir / "learning_curves.csv", [], LEARNING_COLS)
    write_csv(out_dir / "seed_stability.csv", [], SEED_STABILITY_COLS)
    write_csv(out_dir / "quarantine_log.csv", [], QUARANTINE_COLS)
    write_jsonl(out_dir / "residuals.jsonl", [])
    write_json(out_dir / "phase3d_receipt.json", {"manifest": manifest, "arenaGate": None, "branchAdjudication": None})
    (out_dir / "branch_adjudication.md").write_text(
        "# Phase 3D Branch Adjudication\n\nDry run / empty receipt. No arena-gate or branch decision.\n",
        encoding="utf-8",
    )
    (out_dir / "commands.md").write_text(
        "# Phase 3D commands\n\nDry run / empty receipt. No execution command captured.\n",
        encoding="utf-8",
    )
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def select_seed_for_arm(arm: str, per_seed_metrics: dict[int, dict[str, Any]]) -> int:
    if not per_seed_metrics:
        raise ValueError("no candidates to select from")
    def key(seed: int):
        m = per_seed_metrics[seed]
        # higher = better for non-baseline exact count, mask F1, minority recall
        # lower = better for over-edit, loss, seed
        return (
            -m.get("val_nonbaseline_exact_count", 0),
            -m.get("val_edit_mask_f1", 0.0),
            -m.get("val_minority_edit_recall", 0.0),
            m.get("val_over_edit_rate", 1.0),
            m.get("val_loss", float("inf")),
            seed,
        )
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
    spec_path = Path(__file__).resolve().parent / "PHASE3D_DIFFERENT_FRAMING_SPEC.md"
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
    write_csv(out_dir / "split.csv", split_rows, SPLIT_COLS)

    if args.dry_run:
        manifest["mode"] = "dry_run"
        manifest["completedAt"] = iso_now()
        write_empty_receipt(out_dir, manifest)
        print(f"ARC Phase 3D dry run wrote {out_dir}")
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
    max_steps_mask = args.probe_steps if args.probe_only else MASK_MODEL_SPEC["max_steps"]
    max_steps_color = args.probe_steps if args.probe_only else COLOR_MODEL_SPEC["max_steps"]
    manifest["maxStepsEffective"] = {"mask": max_steps_mask, "color": max_steps_color}
    manifest["armsEffective"] = arms
    manifest["seedsEffective"] = seeds

    held_out_instances = val_lodo + val_pttest + test_lodo + test_pttest
    if not held_out_instances:
        print("WARN: no held-out instances to process (limit-tasks may be too small)")

    device = torch.device(args.device)

    per_instance_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    seed_stability_rows: list[dict[str, Any]] = []
    quarantine_rows: list[dict[str, Any]] = []
    baseline_sel_rows: list[dict[str, Any]] = []
    edit_metrics_rows: list[dict[str, Any]] = []

    per_arm_validation_metrics: dict[str, dict[int, dict[str, Any]]] = {arm: {seed: {} for seed in seeds} for arm in arms}
    per_instance_seed_outcomes: dict[tuple[str, str], dict[int, bool]] = {}
    elapsed_total = 0.0

    for arm in arms:
        for seed in seeds:
            for inst in held_out_instances:
                start = time.perf_counter()
                # 1. Baseline candidate selection on conditioning pairs
                candidate = select_baseline_candidate(inst.query_input, inst.conditioning, arm)
                # 2. Apply baseline to each conditioning input + query input
                cond_baselines = [apply_baseline(p["input"], candidate, [q for q in inst.conditioning if q is not p], arm) for p in inst.conditioning]
                query_baseline = apply_baseline(inst.query_input, candidate, inst.conditioning, arm)
                # 3. Build cell-level training rows
                mask_X, mask_y, color_X, color_y = build_conditioning_examples(arm, inst.conditioning, cond_baselines)
                mask_seed = derive_seed(seed, inst.lane, inst.task_id, inst.query_index, arm, "mask")
                color_seed = derive_seed(seed, inst.lane, inst.task_id, inst.query_index, arm, "color")
                threshold_seed = derive_seed(seed, inst.lane, inst.task_id, inst.query_index, arm, "threshold_tiebreak")
                _ = threshold_seed  # reserved for future stochastic tie-break; not used by deterministic threshold sweep
                # 4. Fit mask + color models
                mask_model, mask_info = fit_mask(arm, mask_X, mask_y, mask_seed, max_steps_mask, device)
                color_model, color_info = fit_color(arm, color_X, color_y, color_seed, max_steps_color, device)
                # 5. Predict conditioning masks + colors for threshold selection
                cond_probs = [predict_mask_probs(mask_model, arm, p["input"], b, device) for p, b in zip(inst.conditioning, cond_baselines)]
                cond_pred_colors_per_thr: dict[float, list[list[list[int]]]] = {}
                # For threshold selection we need predicted colors at every threshold; predict
                # colors once with the union-of-edits mask (high recall) and reuse argmax per cell.
                # Cheaper approximation: use full-grid color prediction once, then mask at each threshold.
                cond_full_colors = [predict_edit_colors(color_model, arm, p["input"], b, [[True] * len(b[0]) for _ in range(len(b))], device) for p, b in zip(inst.conditioning, cond_baselines)]
                threshold, threshold_audit = select_mask_threshold(
                    cond_probs,
                    cond_baselines,
                    [p["output"] for p in inst.conditioning],
                    cond_full_colors,
                )
                # 6. Predict query mask + colors at chosen threshold
                query_probs = predict_mask_probs(mask_model, arm, inst.query_input, query_baseline, device)
                query_mask = [[query_probs[y][x] >= threshold for x in range(len(query_baseline[0]))] for y in range(len(query_baseline))] if query_baseline else []
                query_colors = predict_edit_colors(color_model, arm, inst.query_input, query_baseline, query_mask, device)
                predicted_grid = apply_edit(query_baseline, query_mask, query_colors)
                # 7. Score
                grid_exact = grid_equal(predicted_grid, inst.target_output)
                baseline_exact = grid_equal(query_baseline, inst.target_output)
                nonbaseline_exact = grid_exact and not baseline_exact
                shape_exact = shape_of(predicted_grid) == shape_of(inst.target_output)
                pal_exact = palette_of(predicted_grid) == palette_of(inst.target_output)
                px_acc = pixel_accuracy(predicted_grid, inst.target_output)
                base_residual = residual_mass(query_baseline, inst.target_output)
                em = edit_metrics(query_mask, query_baseline, inst.target_output)
                ec_acc = edit_color_accuracy(query_mask, query_colors, inst.target_output)
                copy_prior_absent = candidate["mean_conditioning_residual"] > 0.50
                elapsed = time.perf_counter() - start
                elapsed_total += elapsed

                row = {
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "primary_prior": inst.primary_prior,
                    "predicted_boundary": inst.predicted_boundary,
                    "arm": arm,
                    "seed": seed,
                    "shape_rule": candidate["shape_rule"],
                    "canvas_rule": candidate["canvas_rule"],
                    "background_color": candidate["background_color"],
                    "grid_exact": grid_exact,
                    "baseline_exact": baseline_exact,
                    "nonbaseline_exact": nonbaseline_exact,
                    "shape_exact": shape_exact,
                    "palette_exact": pal_exact,
                    "pixel_accuracy": round_float(px_acc),
                    "baseline_residual_mass": round_float(base_residual),
                    **em,
                    "edit_color_accuracy": round_float(ec_acc),
                    "selected_threshold": threshold,
                    "copy_prior_absent": copy_prior_absent,
                    "elapsed_seconds": round_float(elapsed),
                }
                row["quarantine_label"] = "" if grid_exact else assign_quarantine_label(row, len(inst.conditioning))
                per_instance_rows.append(row)
                baseline_sel_rows.append({
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "arm": arm,
                    "seed": seed,
                    "shape_rule": candidate["shape_rule"],
                    "canvas_rule": candidate["canvas_rule"],
                    "background_color": candidate["background_color"],
                    "mean_conditioning_residual": candidate["mean_conditioning_residual"],
                    "max_conditioning_residual": candidate["max_conditioning_residual"],
                })
                edit_metrics_rows.append({
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "arm": arm,
                    "seed": seed,
                    "selected_threshold": threshold,
                    **em,
                    "edit_color_accuracy": round_float(ec_acc),
                })
                for h in mask_info.get("history", []):
                    learning_rows.append({"instance_id": inst.instance_id, "arm": arm, "seed": seed, "model_kind": "mask", "step": h["step"], "loss": h["loss"]})
                for h in color_info.get("history", []):
                    learning_rows.append({"instance_id": inst.instance_id, "arm": arm, "seed": seed, "model_kind": "color", "step": h["step"], "loss": h["loss"]})
                residual_rows.append({
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "arm": arm,
                    "seed": seed,
                    "target_shape": list(shape_of(inst.target_output)),
                    "baseline_shape": list(shape_of(query_baseline)),
                    "predicted_shape": list(shape_of(predicted_grid)),
                    "selected_threshold": threshold,
                    "threshold_audit": threshold_audit,
                    "predicted_grid": predicted_grid,
                    "baseline_grid": query_baseline,
                    "baseline_candidate": {k: v for k, v in candidate.items() if k != "sort_key"},
                })
                per_instance_seed_outcomes.setdefault((arm, inst.instance_id), {})[seed] = grid_exact and not baseline_exact

                if inst.lane.startswith("validation_"):
                    bucket = per_arm_validation_metrics[arm][seed].setdefault("counts", {
                        "nonbaseline_exact": 0, "n": 0,
                        "f1_sum": 0.0, "min_recall_sum": 0.0, "over_edit_sum": 0.0, "loss_sum": 0.0,
                    })
                    bucket["n"] += 1
                    if nonbaseline_exact:
                        bucket["nonbaseline_exact"] += 1
                    bucket["f1_sum"] += em["edit_mask_f1"]
                    bucket["min_recall_sum"] += em["minority_edit_recall"]
                    bucket["over_edit_sum"] += em["over_edit_rate"]
                    bucket["loss_sum"] += mask_info.get("best_loss", 0.0)
            # Roll up per-seed validation summary
            bucket = per_arm_validation_metrics[arm][seed].get("counts", {
                "nonbaseline_exact": 0, "n": 0,
                "f1_sum": 0.0, "min_recall_sum": 0.0, "over_edit_sum": 0.0, "loss_sum": 0.0,
            })
            n = max(1, bucket["n"])
            per_arm_validation_metrics[arm][seed].update({
                "val_nonbaseline_exact_count": bucket["nonbaseline_exact"],
                "val_edit_mask_f1": round_float(bucket["f1_sum"] / n),
                "val_minority_edit_recall": round_float(bucket["min_recall_sum"] / n),
                "val_over_edit_rate": round_float(bucket["over_edit_sum"] / n),
                "val_loss": round_float(bucket["loss_sum"] / n),
            })

    # Select seed per arm
    selected_seed_by_arm: dict[str, int] = {arm: select_seed_for_arm(arm, per_arm_validation_metrics[arm]) for arm in arms}

    # Aggregations using selected seed
    selected_rows = [r for r in per_instance_rows if r["seed"] == selected_seed_by_arm[r["arm"]]]

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
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact"]) / len(group)),
                "baseline_exact_any_rate": round_float(sum(1 for r in group if r["baseline_exact"]) / len(group)),
                "nonbaseline_exact_any_rate": round_float(sum(1 for r in group if r["nonbaseline_exact"]) / len(group)),
                "shape_exact_rate": round_float(sum(1 for r in group if r["shape_exact"]) / len(group)),
                "palette_exact_rate": round_float(sum(1 for r in group if r["palette_exact"]) / len(group)),
                "pixel_accuracy_mean": round_float(sum(r["pixel_accuracy"] for r in group) / len(group)),
                "edit_mask_f1_mean": round_float(sum(r["edit_mask_f1"] for r in group) / len(group)),
                "minority_edit_recall_mean": round_float(sum(r["minority_edit_recall"] for r in group) / len(group)),
                "over_edit_rate_mean": round_float(sum(r["over_edit_rate"] for r in group) / len(group)),
                "predicted_edit_mass_mean": round_float(sum(r["predicted_edit_mass"] for r in group) / len(group)),
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
                "predicted_boundary": group[0].get("predicted_boundary", ""),
                "arm": arm,
                "selected_seed": selected_seed_by_arm[arm],
                "instance_count": len(group),
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact"]) / len(group)),
                "nonbaseline_exact_any_rate": round_float(sum(1 for r in group if r["nonbaseline_exact"]) / len(group)),
                "baseline_exact_any_rate": round_float(sum(1 for r in group if r["baseline_exact"]) / len(group)),
                "shape_exact_rate": round_float(sum(1 for r in group if r["shape_exact"]) / len(group)),
                "palette_exact_rate": round_float(sum(1 for r in group if r["palette_exact"]) / len(group)),
                "pixel_accuracy_mean": round_float(sum(r["pixel_accuracy"] for r in group) / len(group)),
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
                "grid_exact_any_rate": round_float(sum(1 for r in group if r["grid_exact"]) / len(group)),
                "nonbaseline_exact_any_rate": round_float(sum(1 for r in group if r["nonbaseline_exact"]) / len(group)),
                "edit_mask_f1_mean": round_float(sum(r["edit_mask_f1"] for r in group) / len(group)),
            })
        return out

    scores = _agg_scores(selected_rows)
    per_task_rows = _agg_per_task(selected_rows)
    per_prior_rows = _agg_per_prior(selected_rows)

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

    # Arena gate + branch
    if manifest["mode"] == "full":
        arena = adjudicate_arena_gate(per_task_rows)
        branch = adjudicate_branch_d(per_task_rows, scores, arena)
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
    write_csv(out_dir / "scores.csv", scores, SCORE_COLS)
    write_csv(out_dir / "per_task.csv", per_task_rows, PER_TASK_COLS)
    write_csv(out_dir / "per_prior.csv", per_prior_rows, PER_PRIOR_COLS)
    write_csv(out_dir / "per_instance.csv", per_instance_rows, PER_INSTANCE_COLS)
    write_csv(out_dir / "baseline_selection.csv", baseline_sel_rows, BASELINE_SEL_COLS)
    write_csv(out_dir / "edit_metrics.csv", edit_metrics_rows, EDIT_METRICS_COLS)
    write_csv(out_dir / "learning_curves.csv", learning_rows, LEARNING_COLS)
    write_csv(out_dir / "seed_stability.csv", seed_stability_rows, SEED_STABILITY_COLS)
    write_csv(out_dir / "quarantine_log.csv", quarantine_rows, QUARANTINE_COLS)
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    write_json(out_dir / "phase3d_receipt.json", {
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
        "# Phase 3D Branch Adjudication (structured_edit_residual_v1)",
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
        "# Phase 3D invocation\n\n```\n"
        + " ".join([sys.executable, "docs/prereg/arc/phase3d_structured_edit_residual.py", *sys.argv[1:]])
        + "\n```\n"
        + f"\nMode: {manifest['mode']}; elapsed seconds total: {manifest['elapsedSecondsTotal']}\n",
        encoding="utf-8",
    )

    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Phase 3D {manifest['mode']} run wrote {out_dir}")
    print(f"Arena gate: {arena.get('gate')}; branch: {branch.get('branch')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
