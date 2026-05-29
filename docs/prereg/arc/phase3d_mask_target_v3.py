#!/usr/bin/env python
"""ARC Phase 3D mask-targeted variant runner (structured_edit_mask_target_v3).

Standalone Python runner. Does not import any other phase3 runner. The
arc-p3-feature-v1 encoders, baseline family, and edit-color rule bank are
inherited verbatim from the prior color-rule variant (with header markers).
The scratch mask MLP is demoted from a sole predictor to a single candidate
family (`legacy_mlp_threshold_mask`); the variant introduces 12 additional
deterministic mask candidate families + 5 morphological variants per family
2-13, then selects via LOCO scoring with the spec's tie-break chain.

Spec: docs/prereg/arc/PHASE3D_MASK_TARGET_VARIANT_SPEC.md (filed 2026-05-28).
Prior Branch D variant spec: docs/prereg/arc/PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md.
Base Branch D spec: docs/prereg/arc/PHASE3D_DIFFERENT_FRAMING_SPEC.md.
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
# Frozen Phase 3D mask-target variant constants
# ============================================================================
FEATURE_SCHEMA_VERSION = "arc-p3-feature-v1"
LEARNER_VERSION = "edit_mask_candidate_bank_v1"
VARIANT_VERSION = "structured_edit_mask_target_v3"
PROTOCOL_VERSION = "arc-p3d-mask-target-v3"
RECEIPT_SCHEMA_VERSION = "arc-p3d-mask-target-receipt-v1"

ARMS = [
    "raw_grid_edit_mask_v3",
    "signature_palette_edit_mask_v3",
    "signature_only_edit_mask_v3",
    "metadata_only_edit_mask_v3",
]
GRID_SCORABLE_ARMS = {"raw_grid_edit_mask_v3", "signature_palette_edit_mask_v3"}

# Inherited unchanged from `structured_edit_color_rule_v2` color-rule bank.
RULE_FAMILIES = [
    "constant_edit_color",
    "modal_edit_color",
    "baseline_color_map",
    "input_nn_color_map",
    "input_patch_majority_map",
    "baseline_to_input_pair_map",
    "relative_palette_rank_map",
    "object_role_color_map",
    "row_col_periodic_color",
    "nearest_edited_neighbor_color",
]
ENSEMBLE_TIE_TOLERANCE = 0.05
ENSEMBLE_MIN_MEMBERS = 3

# Per spec §"Mask Candidate Bank". Order is the mask-family tiebreak index.
MASK_FAMILIES = [
    "empty_mask",
    "conditioning_mask_union",
    "conditioning_mask_intersection",
    "conditioning_mask_majority",
    "conditioning_bbox_fill",
    "conditioning_bbox_outline",
    "row_col_periodic_mask",
    "source_color_mask",
    "source_color_pair_mask",
    "object_role_mask",
    "nearest_residual_patch_mask",
    "delta_overlay_mask",
    "legacy_mlp_threshold_mask",
]
MASK_MORPH_OPS = ["identity", "dilate1", "erode1", "close1", "bbox_fill"]
MASK_PATCH_THRESHOLDS = [0.25, 0.50, 0.75]
LEGACY_MLP_THRESHOLDS = [round(0.1 * i, 1) for i in range(1, 10)]  # 0.1..0.9
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
# NOTE: Per spec §"Optional Ensemble" — "No learned color MLP is admitted in
# this variant." The base Branch D runner's COLOR_MODEL_SPEC is intentionally
# omitted here; its functional role is replaced entirely by the deterministic
# color-rule bank (see RULE_FAMILIES above and the rule_bank section below).

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
    if arm == "raw_grid_edit_mask_v3":
        return raw_grid_onehot(grid)
    rep = represent_grid(grid, arm)
    vector = [0.0] * SIGNATURE_VECTOR_DIM
    if arm in {"signature_palette_edit_mask_v3", "metadata_only_edit_mask_v3"}:
        vector[:METADATA_DIM] = rep["metadata"]
    if arm in {"signature_palette_edit_mask_v3", "signature_only_edit_mask_v3"}:
        for idx, value in rep["suffix"].items():
            vector[idx] = value
    return vector


def input_dim_for_arm(arm: str) -> int:
    return RAW_GRID_DIM if arm == "raw_grid_edit_mask_v3" else SIGNATURE_VECTOR_DIM


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


# NOTE: EditColorMLP from base Branch D is replaced by the deterministic
# color-rule bank below. No nn.Module is used for edit-color prediction
# in this variant.


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


# ============================================================================
# Color Rule Bank (per spec §"Color Rule Bank")
# ============================================================================
#
# A "rule" is a JSON-serialisable dict: {family, id, params}. A rule predicts
# a per-cell color when given (query_input, baseline, mask, conditioning,
# cond_baselines, arm). Concrete candidate rules are generated from the
# conditioning residuals; selection is deterministic (LOCO / all-cells scoring
# with the spec's tie-break chain); optional top-3 ensemble fires only if at
# least ENSEMBLE_MIN_MEMBERS rules tie within ENSEMBLE_TIE_TOLERANCE on the
# conditioning accuracy metric.


def _gold_edits_in_pair(pair_baseline: list[list[int]], pair_target: list[list[int]]) -> list[tuple[int, int, int]]:
    """Returns [(oy, ox, target_color)] for each cell where baseline differs from target."""
    out: list[tuple[int, int, int]] = []
    bh = len(pair_baseline)
    bw = len(pair_baseline[0]) if bh else 0
    th = len(pair_target)
    tw = len(pair_target[0]) if th else 0
    for oy in range(min(bh, th)):
        for ox in range(min(bw, tw)):
            if pair_baseline[oy][ox] != pair_target[oy][ox]:
                out.append((oy, ox, pair_target[oy][ox]))
    return out


def _gold_no_edits_in_pair(pair_baseline: list[list[int]], pair_target: list[list[int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    bh = len(pair_baseline)
    bw = len(pair_baseline[0]) if bh else 0
    th = len(pair_target)
    tw = len(pair_target[0]) if th else 0
    for oy in range(min(bh, th)):
        for ox in range(min(bw, tw)):
            if pair_baseline[oy][ox] == pair_target[oy][ox]:
                out.append((oy, ox))
    return out


def _nearest_input_color(query_input: list[list[int]], baseline_shape: tuple[int, int], oy: int, ox: int) -> int:
    """Map output-frame coord (oy,ox) to nearest input-frame coord and return its color."""
    bh, bw = baseline_shape
    ih = len(query_input)
    iw = len(query_input[0]) if ih else 0
    if ih == 0 or iw == 0:
        return 10  # padding
    nory = oy / (bh - 1) if bh > 1 else 0.0
    norx = ox / (bw - 1) if bw > 1 else 0.0
    iy = int(round(nory * (ih - 1)))
    ix = int(round(norx * (iw - 1)))
    return query_input[iy][ix]


def _input_patch_majority(query_input: list[list[int]], baseline_shape: tuple[int, int], oy: int, ox: int) -> int:
    """Majority color in the 3x3 nearest-input patch (ties broken by smallest color id)."""
    bh, bw = baseline_shape
    ih = len(query_input)
    iw = len(query_input[0]) if ih else 0
    if ih == 0 or iw == 0:
        return 10
    nory = oy / (bh - 1) if bh > 1 else 0.0
    norx = ox / (bw - 1) if bw > 1 else 0.0
    iy0 = int(round(nory * (ih - 1)))
    ix0 = int(round(norx * (iw - 1)))
    counts: Counter[int] = Counter()
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y, x = iy0 + dy, ix0 + dx
            if 0 <= y < ih and 0 <= x < iw:
                counts[query_input[y][x]] += 1
    if not counts:
        return 10
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0]


def _palette_rank(palette: list[int], color: int) -> int:
    try:
        return palette.index(color)
    except ValueError:
        return -1


def _role_normalize_grid_obj(grid: list[list[int]]) -> dict[tuple[int, int], int]:
    """Returns {(y, x): role_id} for nonzero cells; role 1, 2, 3... assigned in
    row-major nonzero discovery order. Zero cells get role 0 implicitly."""
    role_map: dict[int, int] = {0: 0}
    out: dict[tuple[int, int], int] = {}
    next_role = 1
    for y, row in enumerate(grid):
        for x, v in enumerate(row):
            if v == 0:
                continue
            if v not in role_map:
                role_map[v] = next_role
                next_role += 1
            out[(y, x)] = role_map[v]
    return out


def _modal_color(values: list[int]) -> int:
    if not values:
        return 10
    counts = Counter(values)
    best = max(counts.items(), key=lambda kv: (kv[1], -kv[0]))
    return best[0]


def _palette_of(grid: list[list[int]]) -> list[int]:
    return sorted({c for row in grid for c in row})


def generate_candidate_rules(conditioning: list[dict[str, Any]], cond_baselines: list[list[list[int]]]) -> list[dict[str, Any]]:
    """Build all concrete candidate rules across the 10 frozen families.

    Each candidate dict carries: family, id, params. Predict-time helpers
    (`_predict_with_rule`) dispatch by family.
    """
    edits_per_pair = [_gold_edits_in_pair(b, p["output"]) for b, p in zip(cond_baselines, conditioning)]
    all_edits: list[tuple[int, int, int]] = []
    for ev in edits_per_pair:
        all_edits.extend(ev)
    candidates: list[dict[str, Any]] = []
    # 1. constant_edit_color — one candidate per observed edit color
    observed_edit_colors = sorted({c for _, _, c in all_edits})
    for c in observed_edit_colors:
        candidates.append({"family": "constant_edit_color", "id": f"c={c}", "params": {"color": c}})
    # 2. modal_edit_color
    modal = _modal_color([c for _, _, c in all_edits]) if all_edits else 10
    candidates.append({"family": "modal_edit_color", "id": "modal", "params": {"color": modal}})
    # 3. baseline_color_map
    bc_map: dict[int, list[int]] = {}
    for pair_edits, baseline in zip(edits_per_pair, cond_baselines):
        for oy, ox, t in pair_edits:
            if 0 <= oy < len(baseline) and 0 <= ox < len(baseline[0]):
                bc_map.setdefault(baseline[oy][ox], []).append(t)
    bc_map_final = {bc: _modal_color(vs) for bc, vs in bc_map.items()}
    candidates.append({"family": "baseline_color_map", "id": "v1", "params": {"map": bc_map_final, "fallback": modal}})
    # 4. input_nn_color_map
    ic_map: dict[int, list[int]] = {}
    for pair, baseline, pair_edits in zip(conditioning, cond_baselines, edits_per_pair):
        bh = len(baseline)
        bw = len(baseline[0]) if bh else 0
        for oy, ox, t in pair_edits:
            src = _nearest_input_color(pair["input"], (bh, bw), oy, ox)
            ic_map.setdefault(src, []).append(t)
    ic_map_final = {sc: _modal_color(vs) for sc, vs in ic_map.items()}
    candidates.append({"family": "input_nn_color_map", "id": "v1", "params": {"map": ic_map_final, "fallback": modal}})
    # 5. input_patch_majority_map
    ip_map: dict[int, list[int]] = {}
    for pair, baseline, pair_edits in zip(conditioning, cond_baselines, edits_per_pair):
        bh = len(baseline)
        bw = len(baseline[0]) if bh else 0
        for oy, ox, t in pair_edits:
            src = _input_patch_majority(pair["input"], (bh, bw), oy, ox)
            ip_map.setdefault(src, []).append(t)
    ip_map_final = {sc: _modal_color(vs) for sc, vs in ip_map.items()}
    candidates.append({"family": "input_patch_majority_map", "id": "v1", "params": {"map": ip_map_final, "fallback": modal}})
    # 6. baseline_to_input_pair_map
    bi_map: dict[str, list[int]] = {}
    for pair, baseline, pair_edits in zip(conditioning, cond_baselines, edits_per_pair):
        bh = len(baseline)
        bw = len(baseline[0]) if bh else 0
        for oy, ox, t in pair_edits:
            bc = baseline[oy][ox]
            inp = _nearest_input_color(pair["input"], (bh, bw), oy, ox)
            bi_map.setdefault(f"{bc}|{inp}", []).append(t)
    bi_map_final = {k: _modal_color(vs) for k, vs in bi_map.items()}
    candidates.append({"family": "baseline_to_input_pair_map", "id": "v1", "params": {"map": bi_map_final, "fallback": modal}})
    # 7. relative_palette_rank_map — 3 sub-rules: same-rank, nearest-rank, learned-rank
    # Build conditioning-learned rank map: src_rank → modal target rank
    rank_map: dict[int, list[int]] = {}
    for pair, baseline, pair_edits in zip(conditioning, cond_baselines, edits_per_pair):
        inp_palette = _palette_of(pair["input"])
        target_palette = _palette_of(pair["output"])
        for oy, ox, t in pair_edits:
            bh = len(baseline)
            bw = len(baseline[0]) if bh else 0
            src_col = _nearest_input_color(pair["input"], (bh, bw), oy, ox)
            sr = _palette_rank(inp_palette, src_col)
            tr = _palette_rank(target_palette, t)
            if sr >= 0 and tr >= 0:
                rank_map.setdefault(sr, []).append(tr)
    rank_map_final = {sr: _modal_color(trs) for sr, trs in rank_map.items()}
    candidates.append({"family": "relative_palette_rank_map", "id": "same", "params": {"strategy": "same", "fallback": modal}})
    candidates.append({"family": "relative_palette_rank_map", "id": "nearest", "params": {"strategy": "nearest", "fallback": modal}})
    candidates.append({"family": "relative_palette_rank_map", "id": "learned", "params": {"strategy": "learned", "rank_map": rank_map_final, "fallback": modal}})
    # 8. object_role_color_map
    role_map: dict[int, list[int]] = {}
    for pair, baseline, pair_edits in zip(conditioning, cond_baselines, edits_per_pair):
        roles_input = _role_normalize_grid_obj(pair["input"])
        roles_baseline = _role_normalize_grid_obj(baseline)
        for oy, ox, t in pair_edits:
            bh = len(baseline)
            bw = len(baseline[0]) if bh else 0
            iy = int(round((oy / (bh - 1) if bh > 1 else 0.0) * (len(pair["input"]) - 1))) if pair["input"] else 0
            ix = int(round((ox / (bw - 1) if bw > 1 else 0.0) * (len(pair["input"][0]) - 1))) if pair["input"] and pair["input"][0] else 0
            r_in = roles_input.get((iy, ix), 0)
            r_bs = roles_baseline.get((oy, ox), 0)
            key = r_in if r_in != 0 else r_bs
            role_map.setdefault(key, []).append(t)
    role_map_final = {k: _modal_color(vs) for k, vs in role_map.items()}
    candidates.append({"family": "object_role_color_map", "id": "v1", "params": {"map": role_map_final, "fallback": modal}})
    # 9. row_col_periodic_color — periods 1, 2, 3 for row and col
    for axis in ("row", "col"):
        for period in (1, 2, 3):
            per_map: dict[int, list[int]] = {}
            for pair_edits in edits_per_pair:
                for oy, ox, t in pair_edits:
                    key = (oy if axis == "row" else ox) % period
                    per_map.setdefault(key, []).append(t)
            per_final = {k: _modal_color(vs) for k, vs in per_map.items()}
            if per_final:
                candidates.append({"family": "row_col_periodic_color", "id": f"{axis}_p{period}", "params": {"axis": axis, "period": period, "map": per_final, "fallback": modal}})
    # 10. nearest_edited_neighbor_color — copy nearest conditioning edited cell color by normalized-coord distance
    # Build the lookup table: list of (norm_y, norm_x, color) across all conditioning edited cells.
    neighbor_table: list[tuple[float, float, int]] = []
    for baseline, pair_edits in zip(cond_baselines, edits_per_pair):
        bh = len(baseline)
        bw = len(baseline[0]) if bh else 0
        for oy, ox, t in pair_edits:
            nory = oy / (bh - 1) if bh > 1 else 0.0
            norx = ox / (bw - 1) if bw > 1 else 0.0
            neighbor_table.append((nory, norx, t))
    candidates.append({"family": "nearest_edited_neighbor_color", "id": "v1", "params": {"table": neighbor_table, "fallback": modal}})
    # Filter out empty/degenerate candidates (e.g. families with no observations).
    filtered = []
    for cand in candidates:
        p = cand.get("params", {})
        if cand["family"] in {"baseline_color_map", "input_nn_color_map", "input_patch_majority_map", "baseline_to_input_pair_map", "object_role_color_map"}:
            if not p.get("map"):
                continue
        if cand["family"] == "relative_palette_rank_map" and cand["id"] == "learned" and not p.get("rank_map"):
            continue
        if cand["family"] == "nearest_edited_neighbor_color" and not p.get("table"):
            continue
        if cand["family"] == "row_col_periodic_color" and not p.get("map"):
            continue
        filtered.append(cand)
    return filtered


def _predict_with_rule(rule: dict[str, Any], query_input: list[list[int]], baseline: list[list[int]], mask: list[list[bool]]) -> list[list[int]]:
    """Apply a single rule to predict edit colors at masked cells. Returns the
    full grid (baseline color at unmasked cells, predicted color at masked cells)."""
    out = [list(row) for row in baseline]
    bh = len(baseline)
    bw = len(baseline[0]) if bh else 0
    family = rule["family"]
    params = rule["params"]
    fallback = params.get("fallback", 0)
    if family == "constant_edit_color":
        const_c = params["color"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    out[oy][ox] = const_c
        return out
    if family == "modal_edit_color":
        c = params["color"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    out[oy][ox] = c
        return out
    if family == "baseline_color_map":
        m = params["map"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    out[oy][ox] = m.get(baseline[oy][ox], fallback)
        return out
    if family == "input_nn_color_map":
        m = params["map"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    src = _nearest_input_color(query_input, (bh, bw), oy, ox)
                    out[oy][ox] = m.get(src, fallback)
        return out
    if family == "input_patch_majority_map":
        m = params["map"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    src = _input_patch_majority(query_input, (bh, bw), oy, ox)
                    out[oy][ox] = m.get(src, fallback)
        return out
    if family == "baseline_to_input_pair_map":
        m = params["map"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    bc = baseline[oy][ox]
                    inp = _nearest_input_color(query_input, (bh, bw), oy, ox)
                    out[oy][ox] = m.get(f"{bc}|{inp}", fallback)
        return out
    if family == "relative_palette_rank_map":
        strategy = params["strategy"]
        inp_palette = _palette_of(query_input)
        target_palette = _palette_of(query_input)  # heuristic: edit palette = input palette as fallback for "same"/"nearest"
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    src = _nearest_input_color(query_input, (bh, bw), oy, ox)
                    sr = _palette_rank(inp_palette, src)
                    if strategy == "same":
                        tr = sr
                    elif strategy == "nearest":
                        tr = min(range(len(target_palette)), key=lambda r: abs(r - sr)) if target_palette else -1
                    else:  # learned
                        rm = params.get("rank_map", {})
                        tr = rm.get(sr, sr)
                    if tr is not None and 0 <= tr < len(target_palette):
                        out[oy][ox] = target_palette[tr]
                    else:
                        out[oy][ox] = fallback
        return out
    if family == "object_role_color_map":
        m = params["map"]
        roles_input = _role_normalize_grid_obj(query_input)
        roles_baseline = _role_normalize_grid_obj(baseline)
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    iy = int(round((oy / (bh - 1) if bh > 1 else 0.0) * (len(query_input) - 1))) if query_input else 0
                    ix = int(round((ox / (bw - 1) if bw > 1 else 0.0) * (len(query_input[0]) - 1))) if query_input and query_input[0] else 0
                    r_in = roles_input.get((iy, ix), 0)
                    r_bs = roles_baseline.get((oy, ox), 0)
                    key = r_in if r_in != 0 else r_bs
                    out[oy][ox] = m.get(key, fallback)
        return out
    if family == "row_col_periodic_color":
        m = params["map"]
        axis = params["axis"]
        period = params["period"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    key = (oy if axis == "row" else ox) % period
                    out[oy][ox] = m.get(key, fallback)
        return out
    if family == "nearest_edited_neighbor_color":
        table = params["table"]
        for oy in range(bh):
            for ox in range(bw):
                if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                    nory = oy / (bh - 1) if bh > 1 else 0.0
                    norx = ox / (bw - 1) if bw > 1 else 0.0
                    if table:
                        best = min(table, key=lambda nyx: (nyx[0] - nory) ** 2 + (nyx[1] - norx) ** 2)
                        out[oy][ox] = best[2]
                    else:
                        out[oy][ox] = fallback
        return out
    raise ValueError(f"unknown rule family {family!r}")


def _ensemble_predict(rules: list[dict[str, Any]], query_input: list[list[int]], baseline: list[list[int]], mask: list[list[bool]]) -> list[list[int]]:
    """Plurality vote across rules at each masked cell. Ties broken by rule order (first rule's color wins)."""
    out = [list(row) for row in baseline]
    bh = len(baseline)
    bw = len(baseline[0]) if bh else 0
    preds = [_predict_with_rule(r, query_input, baseline, mask) for r in rules]
    for oy in range(bh):
        for ox in range(bw):
            if oy < len(mask) and ox < len(mask[0]) and mask[oy][ox]:
                votes = Counter(p[oy][ox] for p in preds)
                top = votes.most_common(1)[0][0]
                out[oy][ox] = top
    return out


def _score_rule_on_pairs(rule: dict[str, Any], conditioning: list[dict[str, Any]], cond_baselines: list[list[list[int]]], use_loco: bool) -> dict[str, Any]:
    """Score a rule: primary = edit-color accuracy on gold edited cells;
    rare-color recall; color-hallucination rate on no-edit cells."""
    correct_edit = 0
    total_edit = 0
    rare_total = 0
    rare_hit = 0
    halluc_total = 0
    halluc_count = 0
    all_targets = [c for b, p in zip(cond_baselines, conditioning) for _, _, c in _gold_edits_in_pair(b, p["output"])]
    modal_target = _modal_color(all_targets) if all_targets else -1
    n = len(conditioning)
    for i, (pair, baseline) in enumerate(zip(conditioning, cond_baselines)):
        gold_edits = _gold_edits_in_pair(baseline, pair["output"])
        gold_no_edits = _gold_no_edits_in_pair(baseline, pair["output"])
        # Rebuild rule on LOCO subset of conditioning if applicable
        if use_loco and n >= 3:
            loco_cond = [c for j, c in enumerate(conditioning) if j != i]
            loco_bls = [b for j, b in enumerate(cond_baselines) if j != i]
            loco_rules = generate_candidate_rules(loco_cond, loco_bls)
            # Find the LOCO equivalent of this rule (same family + id) — fall back if missing
            loco_rule = next((r for r in loco_rules if r["family"] == rule["family"] and r["id"] == rule["id"]), rule)
        else:
            loco_rule = rule
        bh = len(baseline)
        bw = len(baseline[0]) if bh else 0
        full_mask = [[True] * bw for _ in range(bh)]
        pred_full = _predict_with_rule(loco_rule, pair["input"], baseline, full_mask)
        for oy, ox, t in gold_edits:
            if 0 <= oy < len(pred_full) and 0 <= ox < len(pred_full[0]):
                total_edit += 1
                if pred_full[oy][ox] == t:
                    correct_edit += 1
                if t != modal_target:
                    rare_total += 1
                    if pred_full[oy][ox] == t:
                        rare_hit += 1
        for oy, ox in gold_no_edits:
            if 0 <= oy < len(pred_full) and 0 <= ox < len(pred_full[0]):
                halluc_total += 1
                if pred_full[oy][ox] != baseline[oy][ox]:
                    halluc_count += 1
    accuracy = (correct_edit / total_edit) if total_edit else 0.0
    rare_recall = (rare_hit / rare_total) if rare_total else 1.0
    halluc_rate = (halluc_count / halluc_total) if halluc_total else 0.0
    return {
        "accuracy": round_float(accuracy),
        "rare_recall": round_float(rare_recall),
        "halluc_rate": round_float(halluc_rate),
        "gold_edit_count": total_edit,
        "rare_color_count": rare_total,
    }


def select_color_rule(candidates: list[dict[str, Any]], conditioning: list[dict[str, Any]], cond_baselines: list[list[list[int]]], master_seed: int, lane: str, task_id: str, query_index: int, arm: str) -> dict[str, Any]:
    """Score every candidate; apply the spec's tie-break chain; return the selected
    rule (or ensemble) with its score record + audit candidates list."""
    if not candidates:
        return {"selected": None, "ensemble": False, "members": [], "candidates": [], "low_k_rule_selection": False, "no_conditioning_edits": True}
    n = len(conditioning)
    use_loco = n >= 3
    low_k_flag = not use_loco
    scored: list[dict[str, Any]] = []
    for c in candidates:
        s = _score_rule_on_pairs(c, conditioning, cond_baselines, use_loco)
        if s["gold_edit_count"] == 0:
            continue
        family_idx = RULE_FAMILIES.index(c["family"])
        tiebreak_key = hashlib.sha256(
            f"arc-p3d-mask-target-v3\0{master_seed}\0{lane}\0{task_id}\0{query_index}\0{arm}\0{c['family']}|{c['id']}".encode("utf-8")
        ).hexdigest()
        scored.append({**c, "score": s, "family_index": family_idx, "tiebreak_key": tiebreak_key})
    if not scored:
        return {"selected": None, "ensemble": False, "members": [], "candidates": [], "low_k_rule_selection": low_k_flag, "no_conditioning_edits": True}
    # Spec tie-break: (-accuracy, -rare_recall, halluc_rate, family_index, tiebreak_key)
    scored.sort(key=lambda r: (-r["score"]["accuracy"], -r["score"]["rare_recall"], r["score"]["halluc_rate"], r["family_index"], r["tiebreak_key"]))
    top = scored[0]
    # Optional ensemble per spec §"Optional Ensemble": top-3 vote only if 3+ candidates tie within ENSEMBLE_TIE_TOLERANCE
    ties = [r for r in scored if abs(r["score"]["accuracy"] - top["score"]["accuracy"]) <= ENSEMBLE_TIE_TOLERANCE]
    if len(ties) >= ENSEMBLE_MIN_MEMBERS:
        members = ties[:3]
        return {
            "selected": {"family": "ensemble_top3", "id": "+".join(f"{m['family']}/{m['id']}" for m in members), "params": {"members": members}},
            "ensemble": True,
            "members": members,
            "candidates": scored,
            "low_k_rule_selection": low_k_flag,
            "no_conditioning_edits": False,
            "top_accuracy": top["score"]["accuracy"],
        }
    return {
        "selected": top,
        "ensemble": False,
        "members": [top],
        "candidates": scored,
        "low_k_rule_selection": low_k_flag,
        "no_conditioning_edits": False,
        "top_accuracy": top["score"]["accuracy"],
    }


def predict_query_edit_colors(selection: dict[str, Any], query_input: list[list[int]], baseline: list[list[int]], mask: list[list[bool]]) -> list[list[int]]:
    """Apply the selected rule (or ensemble) to predict query edit colors."""
    if selection.get("selected") is None:
        return [list(row) for row in baseline]
    if selection.get("ensemble"):
        return _ensemble_predict([m for m in selection["members"]], query_input, baseline, mask)
    return _predict_with_rule(selection["selected"], query_input, baseline, mask)


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


# NOTE: base Branch D's predict_edit_colors (MLP-driven) is replaced by
# predict_query_edit_colors above (rule-bank-driven).


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
# Mask Candidate Bank (per spec §"Mask Candidate Bank")
# ============================================================================
#
# 13 frozen mask candidate families. Each generates one or more concrete
# Boolean masks predicted in the query output frame. Family 13
# (`legacy_mlp_threshold_mask`) demotes the inherited scratch MLP to a single
# family that exposes 9 thresholded versions as 9 separate candidates.
# Families 2-13 each apply 5 morphological variants (identity, dilate1,
# erode1, close1, bbox_fill_per_component) as distinct candidate IDs. The
# bank is scored on conditioning residuals via LOCO when k>=3, then selected
# via the spec's tie-break chain (F1 / nonmodal recall / precision / mass
# error / over-edit / family index / SHA-256 key).


def _project_mask_to_shape(mask: list[list[bool]], target_h: int, target_w: int) -> list[list[bool]]:
    """Resample a Boolean mask into a (target_h, target_w) frame via nearest-
    neighbor over normalized coordinates."""
    sh = len(mask)
    sw = len(mask[0]) if sh else 0
    if sh == 0 or sw == 0 or target_h == 0 or target_w == 0:
        return [[False] * target_w for _ in range(target_h)]
    out = [[False] * target_w for _ in range(target_h)]
    for oy in range(target_h):
        for ox in range(target_w):
            ny = oy / (target_h - 1) if target_h > 1 else 0.0
            nx = ox / (target_w - 1) if target_w > 1 else 0.0
            sy = int(round(ny * (sh - 1)))
            sx = int(round(nx * (sw - 1)))
            if mask[sy][sx]:
                out[oy][ox] = True
    return out


def _mask_dilate1(mask: list[list[bool]]) -> list[list[bool]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    out = [[mask[y][x] for x in range(w)] for y in range(h)]
    for y in range(h):
        for x in range(w):
            if mask[y][x]:
                continue
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny][nx]:
                    out[y][x] = True
                    break
    return out


def _mask_erode1(mask: list[list[bool]]) -> list[list[bool]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    out = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if not mask[y][x]:
                continue
            keep = True
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w) or not mask[ny][nx]:
                    keep = False
                    break
            out[y][x] = keep
    return out


def _mask_close1(mask: list[list[bool]]) -> list[list[bool]]:
    return _mask_erode1(_mask_dilate1(mask))


def _mask_bbox_fill_components(mask: list[list[bool]]) -> list[list[bool]]:
    """For each connected component in mask, fill its bounding box."""
    h = len(mask)
    w = len(mask[0]) if h else 0
    out = [[mask[y][x] for x in range(w)] for y in range(h)]
    seen = [[False] * w for _ in range(h)]
    for y in range(h):
        for x in range(w):
            if not mask[y][x] or seen[y][x]:
                continue
            # BFS the component
            stack = [(x, y)]
            cells: list[tuple[int, int]] = []
            seen[y][x] = True
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            min_x = min(cx for cx, cy in cells)
            max_x = max(cx for cx, cy in cells)
            min_y = min(cy for cx, cy in cells)
            max_y = max(cy for cx, cy in cells)
            for fy in range(min_y, max_y + 1):
                for fx in range(min_x, max_x + 1):
                    out[fy][fx] = True
    return out


def _apply_morph(mask: list[list[bool]], op: str) -> list[list[bool]]:
    if op == "identity":
        return [list(row) for row in mask]
    if op == "dilate1":
        return _mask_dilate1(mask)
    if op == "erode1":
        return _mask_erode1(mask)
    if op == "close1":
        return _mask_close1(mask)
    if op == "bbox_fill":
        return _mask_bbox_fill_components(mask)
    raise ValueError(f"unknown morph op {op!r}")


def _conditioning_gold_mask(baseline: list[list[int]], target: list[list[int]]) -> list[list[bool]]:
    h = len(baseline)
    w = len(baseline[0]) if h else 0
    th = len(target)
    tw = len(target[0]) if th else 0
    out = [[False] * w for _ in range(h)]
    for y in range(min(h, th)):
        for x in range(min(w, tw)):
            if baseline[y][x] != target[y][x]:
                out[y][x] = True
    return out


def _components(mask: list[list[bool]]) -> list[list[tuple[int, int]]]:
    h = len(mask)
    w = len(mask[0]) if h else 0
    seen = [[False] * w for _ in range(h)]
    comps: list[list[tuple[int, int]]] = []
    for y in range(h):
        for x in range(w):
            if not mask[y][x] or seen[y][x]:
                continue
            stack = [(x, y)]
            cells: list[tuple[int, int]] = []
            seen[y][x] = True
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny][nx] and not seen[ny][nx]:
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            comps.append(cells)
    return comps


def _bbox_of_mask(mask: list[list[bool]]) -> tuple[int, int, int, int] | None:
    cells = [(x, y) for y, row in enumerate(mask) for x, v in enumerate(row) if v]
    if not cells:
        return None
    return (min(x for x, _ in cells), min(y for _, y in cells), max(x for x, _ in cells), max(y for _, y in cells))


def generate_mask_candidates(
    arm: str,
    query_input: list[list[int]],
    query_baseline: list[list[int]],
    conditioning: list[dict[str, Any]],
    cond_baselines: list[list[list[int]]],
    mask_seed: int,
    max_steps_mask: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Build all concrete mask candidates per spec §"Mask Candidate Bank".

    Each candidate is a dict with keys: family, id, mask (list[list[bool]] in
    query_baseline frame). Morphological variants are emitted as distinct
    candidates.
    """
    qh = len(query_baseline)
    qw = len(query_baseline[0]) if qh else 0
    cond_masks_native: list[list[list[bool]]] = [
        _conditioning_gold_mask(b, p["output"]) for b, p in zip(cond_baselines, conditioning)
    ]
    cond_masks_q: list[list[list[bool]]] = [
        _project_mask_to_shape(m, qh, qw) for m in cond_masks_native
    ]

    base_candidates: list[dict[str, Any]] = []

    # 1. empty_mask
    base_candidates.append({"family": "empty_mask", "id": "v1", "mask": [[False] * qw for _ in range(qh)]})

    # 2. conditioning_mask_union
    union = [[False] * qw for _ in range(qh)]
    for m in cond_masks_q:
        for y in range(qh):
            for x in range(qw):
                if m[y][x]:
                    union[y][x] = True
    base_candidates.append({"family": "conditioning_mask_union", "id": "v1", "mask": union})

    # 3. conditioning_mask_intersection
    if cond_masks_q:
        inter = [[True] * qw for _ in range(qh)]
        for m in cond_masks_q:
            for y in range(qh):
                for x in range(qw):
                    if not m[y][x]:
                        inter[y][x] = False
        base_candidates.append({"family": "conditioning_mask_intersection", "id": "v1", "mask": inter})

    # 4. conditioning_mask_majority
    if cond_masks_q:
        counts = [[0] * qw for _ in range(qh)]
        for m in cond_masks_q:
            for y in range(qh):
                for x in range(qw):
                    if m[y][x]:
                        counts[y][x] += 1
        thr = len(cond_masks_q) / 2
        maj = [[counts[y][x] >= thr for x in range(qw)] for y in range(qh)]
        base_candidates.append({"family": "conditioning_mask_majority", "id": "v1", "mask": maj})

    # 5. conditioning_bbox_fill — emit union/intersection/majority bbox aggregates
    bboxes = [b for b in (_bbox_of_mask(m) for m in cond_masks_q) if b is not None]
    if bboxes:
        # union: encompassing bbox
        u_x0 = min(b[0] for b in bboxes); u_y0 = min(b[1] for b in bboxes)
        u_x1 = max(b[2] for b in bboxes); u_y1 = max(b[3] for b in bboxes)
        ubox = [[u_y0 <= y <= u_y1 and u_x0 <= x <= u_x1 for x in range(qw)] for y in range(qh)]
        base_candidates.append({"family": "conditioning_bbox_fill", "id": "union", "mask": ubox})
        # intersection: shared bbox region
        i_x0 = max(b[0] for b in bboxes); i_y0 = max(b[1] for b in bboxes)
        i_x1 = min(b[2] for b in bboxes); i_y1 = min(b[3] for b in bboxes)
        if i_x0 <= i_x1 and i_y0 <= i_y1:
            ibox = [[i_y0 <= y <= i_y1 and i_x0 <= x <= i_x1 for x in range(qw)] for y in range(qh)]
            base_candidates.append({"family": "conditioning_bbox_fill", "id": "intersection", "mask": ibox})
        # majority: per-cell vote
        bcounts = [[0] * qw for _ in range(qh)]
        for b in bboxes:
            for y in range(b[1], b[3] + 1):
                for x in range(b[0], b[2] + 1):
                    if 0 <= y < qh and 0 <= x < qw:
                        bcounts[y][x] += 1
        bthr = len(bboxes) / 2
        mbox = [[bcounts[y][x] >= bthr for x in range(qw)] for y in range(qh)]
        base_candidates.append({"family": "conditioning_bbox_fill", "id": "majority", "mask": mbox})
        # 6. conditioning_bbox_outline — 1-cell border of the union bbox
        outline = [[False] * qw for _ in range(qh)]
        for y in range(u_y0, u_y1 + 1):
            for x in range(u_x0, u_x1 + 1):
                if 0 <= y < qh and 0 <= x < qw and (y in (u_y0, u_y1) or x in (u_x0, u_x1)):
                    outline[y][x] = True
        base_candidates.append({"family": "conditioning_bbox_outline", "id": "union", "mask": outline})

    # 7. row_col_periodic_mask — periods 1, 2, 3 × {row, col}
    for axis in ("row", "col"):
        for period in (1, 2, 3):
            # Infer which residue classes are edited in conditioning
            class_counts: dict[int, int] = {}
            for m in cond_masks_q:
                for y in range(qh):
                    for x in range(qw):
                        if m[y][x]:
                            key = (y if axis == "row" else x) % period
                            class_counts[key] = class_counts.get(key, 0) + 1
            if not class_counts:
                continue
            # Active classes: those with at least one conditioning edit
            active = set(class_counts.keys())
            mask = [[((y if axis == "row" else x) % period) in active for x in range(qw)] for y in range(qh)]
            base_candidates.append({"family": "row_col_periodic_mask", "id": f"{axis}_p{period}", "mask": mask})

    # 8. source_color_mask — edit query cells whose mapped input/baseline color is in conditioning-edited source-color set
    source_colors: set[int] = set()
    for pair, b, m in zip(conditioning, cond_baselines, cond_masks_native):
        bh = len(b)
        bw = len(b[0]) if bh else 0
        for y in range(bh):
            for x in range(bw):
                if m[y][x]:
                    src = _nearest_input_color(pair["input"], (bh, bw), y, x)
                    source_colors.add(src)
                    source_colors.add(b[y][x])
    if source_colors:
        sc_mask = [[False] * qw for _ in range(qh)]
        for y in range(qh):
            for x in range(qw):
                src_in = _nearest_input_color(query_input, (qh, qw), y, x)
                src_bs = query_baseline[y][x]
                if src_in in source_colors or src_bs in source_colors:
                    sc_mask[y][x] = True
        base_candidates.append({"family": "source_color_mask", "id": "v1", "mask": sc_mask})

    # 9. source_color_pair_mask — (input, baseline) color pairs
    source_pairs: set[tuple[int, int]] = set()
    for pair, b, m in zip(conditioning, cond_baselines, cond_masks_native):
        bh = len(b)
        bw = len(b[0]) if bh else 0
        for y in range(bh):
            for x in range(bw):
                if m[y][x]:
                    src = _nearest_input_color(pair["input"], (bh, bw), y, x)
                    source_pairs.add((src, b[y][x]))
    if source_pairs:
        sp_mask = [[False] * qw for _ in range(qh)]
        for y in range(qh):
            for x in range(qw):
                src_in = _nearest_input_color(query_input, (qh, qw), y, x)
                src_bs = query_baseline[y][x]
                if (src_in, src_bs) in source_pairs:
                    sp_mask[y][x] = True
        base_candidates.append({"family": "source_color_pair_mask", "id": "v1", "mask": sp_mask})

    # 10. object_role_mask — connected components on input nonzero cells, role = size rank
    def _role_set_from_conditioning() -> set[int]:
        # Compute the union of "role ids whose containing cells are edited in conditioning"
        roles: set[int] = set()
        for pair, b, m in zip(conditioning, cond_baselines, cond_masks_native):
            comps = _components([[v != 0 for v in row] for row in pair["input"]])
            # Sort by size descending → role 1 is largest
            comps_sorted = sorted(comps, key=lambda c: -len(c))
            role_at: dict[tuple[int, int], int] = {}
            for idx, comp in enumerate(comps_sorted):
                for cx, cy in comp:
                    role_at[(cy, cx)] = idx + 1
            bh = len(b)
            bw = len(b[0]) if bh else 0
            ih = len(pair["input"])
            iw = len(pair["input"][0]) if ih else 0
            for y in range(bh):
                for x in range(bw):
                    if m[y][x]:
                        iy = int(round((y / (bh - 1) if bh > 1 else 0.0) * (ih - 1))) if ih > 0 else 0
                        ix = int(round((x / (bw - 1) if bw > 1 else 0.0) * (iw - 1))) if iw > 0 else 0
                        r = role_at.get((iy, ix), 0)
                        if r > 0:
                            roles.add(r)
        return roles

    target_roles = _role_set_from_conditioning()
    if target_roles:
        # Apply to query input + project
        q_comps = _components([[v != 0 for v in row] for row in query_input])
        q_comps_sorted = sorted(q_comps, key=lambda c: -len(c))
        q_role_mask_input = [[0] * (len(query_input[0]) if query_input else 0) for _ in range(len(query_input))]
        for idx, comp in enumerate(q_comps_sorted):
            for cx, cy in comp:
                q_role_mask_input[cy][cx] = idx + 1
        ih = len(query_input)
        iw = len(query_input[0]) if ih else 0
        role_mask = [[False] * qw for _ in range(qh)]
        for y in range(qh):
            for x in range(qw):
                iy = int(round((y / (qh - 1) if qh > 1 else 0.0) * (ih - 1))) if ih > 0 else 0
                ix = int(round((x / (qw - 1) if qw > 1 else 0.0) * (iw - 1))) if iw > 0 else 0
                if q_role_mask_input[iy][ix] in target_roles:
                    role_mask[y][x] = True
        base_candidates.append({"family": "object_role_mask", "id": "v1", "mask": role_mask})

    # 11. nearest_residual_patch_mask — KNN-vote against conditioning residual cells in normalized coords
    knn_table: list[tuple[float, float, bool]] = []
    for m in cond_masks_q:
        ch = len(m)
        cw = len(m[0]) if ch else 0
        for y in range(ch):
            for x in range(cw):
                ny = y / (ch - 1) if ch > 1 else 0.0
                nx = x / (cw - 1) if cw > 1 else 0.0
                knn_table.append((ny, nx, m[y][x]))
    if knn_table:
        # For each threshold, build a mask via majority vote of K=3 nearest neighbors
        for thr in MASK_PATCH_THRESHOLDS:
            knn_mask = [[False] * qw for _ in range(qh)]
            for y in range(qh):
                for x in range(qw):
                    ny = y / (qh - 1) if qh > 1 else 0.0
                    nx = x / (qw - 1) if qw > 1 else 0.0
                    dists = [((nyk - ny) ** 2 + (nxk - nx) ** 2, v) for nyk, nxk, v in knn_table]
                    dists.sort(key=lambda d: d[0])
                    k = min(3, len(dists))
                    vote = sum(1 for _, v in dists[:k] if v) / k
                    if vote >= thr:
                        knn_mask[y][x] = True
            base_candidates.append({"family": "nearest_residual_patch_mask", "id": f"thr={thr}", "mask": knn_mask})

    # 12. delta_overlay_mask — simplified: union of conditioning residual masks projected to query frame
    if cond_masks_q:
        delta = [[False] * qw for _ in range(qh)]
        for m in cond_masks_q:
            for y in range(qh):
                for x in range(qw):
                    if m[y][x]:
                        delta[y][x] = True
        base_candidates.append({"family": "delta_overlay_mask", "id": "v1", "mask": delta})

    # 13. legacy_mlp_threshold_mask — train inherited MaskMLP, expose 9 thresholds as candidates
    mask_X, mask_y, _, _ = build_conditioning_examples(arm, conditioning, cond_baselines)
    legacy_mlp_info: dict[str, Any] = {"steps": 0, "best_loss": float("inf"), "seed": mask_seed, "rows": 0, "edit_count": 0}
    if mask_X and qh > 0 and qw > 0:
        legacy_model, legacy_mlp_info = fit_mask(arm, mask_X, mask_y, mask_seed, max_steps_mask, device)
        legacy_probs = predict_mask_probs(legacy_model, arm, query_input, query_baseline, device)
        for thr in LEGACY_MLP_THRESHOLDS:
            tmask = [[legacy_probs[y][x] >= thr for x in range(qw)] for y in range(qh)]
            base_candidates.append({"family": "legacy_mlp_threshold_mask", "id": f"thr={thr}", "mask": tmask})

    # Apply morphological variants to families 2-13 (not empty_mask)
    candidates: list[dict[str, Any]] = []
    for cand in base_candidates:
        if cand["family"] == "empty_mask":
            candidates.append(cand)
            continue
        for op in MASK_MORPH_OPS:
            morphed = _apply_morph(cand["mask"], op)
            candidates.append({
                "family": cand["family"],
                "id": f"{cand['id']}|{op}",
                "mask": morphed,
            })
    return candidates, legacy_mlp_info


def _mask_score(predicted: list[list[bool]], gold: list[list[bool]]) -> dict[str, Any]:
    """F1 + precision + recall + IoU + edit-mass-error + over_edit on a SINGLE conditioning pair."""
    h = len(predicted); w = len(predicted[0]) if h else 0
    gh = len(gold); gw = len(gold[0]) if gh else 0
    tp = fp = fn = tn = 0
    for y in range(min(h, gh)):
        for x in range(min(w, gw)):
            if predicted[y][x] and gold[y][x]:
                tp += 1
            elif predicted[y][x] and not gold[y][x]:
                fp += 1
            elif not predicted[y][x] and gold[y][x]:
                fn += 1
            else:
                tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    total = h * w
    pred_mass = sum(1 for row in predicted for v in row if v) / max(1, total)
    gold_mass = sum(1 for row in gold for v in row if v) / max(1, total)
    mass_err = abs(pred_mass - gold_mass)
    over_edit = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "mass_error": mass_err, "over_edit": over_edit, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _score_mask_candidate_on_pairs(
    cand: dict[str, Any],
    conditioning: list[dict[str, Any]],
    cond_baselines: list[list[list[int]]],
    use_loco: bool,
    arm: str,
    mask_seed: int,
    max_steps_mask: int,
    device: torch.device,
) -> dict[str, Any]:
    """Score a candidate on conditioning pairs. For families that regenerate per-LOCO,
    rebuild the candidate from k-1 conditioning pairs and score on the held-out pair."""
    n = len(conditioning)
    sum_f1 = sum_prec = sum_rec = sum_nonmodal = sum_mass = sum_over = 0.0
    n_eval = 0
    # Modal-edit-color in conditioning (for nonmodal recall)
    all_colors: list[int] = []
    for b, p in zip(cond_baselines, conditioning):
        for oy, ox, t in _gold_edits_in_pair(b, p["output"]):
            all_colors.append(t)
    modal = _modal_color(all_colors) if all_colors else -1
    for i, (pair, baseline) in enumerate(zip(conditioning, cond_baselines)):
        gold_pair_mask = _conditioning_gold_mask(baseline, pair["output"])
        if use_loco and n >= 3:
            loco_cond = [c for j, c in enumerate(conditioning) if j != i]
            loco_bls = [b for j, b in enumerate(cond_baselines) if j != i]
            loco_cands, _ = generate_mask_candidates(arm, pair["input"], baseline, loco_cond, loco_bls, mask_seed, max_steps_mask, device)
            loco_cand = next((c for c in loco_cands if c["family"] == cand["family"] and c["id"] == cand["id"]), None)
            if loco_cand is None:
                continue
            pred_native = loco_cand["mask"]
        else:
            # Project the candidate mask (in query frame) onto this conditioning pair's frame
            pred_native = _project_mask_to_shape(cand["mask"], len(baseline), len(baseline[0]) if baseline else 0)
        s = _mask_score(pred_native, gold_pair_mask)
        sum_f1 += s["f1"]; sum_prec += s["precision"]; sum_rec += s["recall"]
        sum_mass += s["mass_error"]; sum_over += s["over_edit"]
        # Nonmodal-edit recall: cells whose target edit color != modal
        nonmodal_total = 0; nonmodal_hit = 0
        bh = len(baseline); bw = len(baseline[0]) if bh else 0
        th = len(pair["output"]); tw = len(pair["output"][0]) if th else 0
        for y in range(min(bh, th)):
            for x in range(min(bw, tw)):
                if baseline[y][x] != pair["output"][y][x] and pair["output"][y][x] != modal:
                    nonmodal_total += 1
                    if y < len(pred_native) and x < len(pred_native[0]) and pred_native[y][x]:
                        nonmodal_hit += 1
        nm_recall = (nonmodal_hit / nonmodal_total) if nonmodal_total else 1.0
        sum_nonmodal += nm_recall
        n_eval += 1
    n_eval = max(1, n_eval)
    return {
        "f1": round_float(sum_f1 / n_eval),
        "precision": round_float(sum_prec / n_eval),
        "recall": round_float(sum_rec / n_eval),
        "nonmodal_recall": round_float(sum_nonmodal / n_eval),
        "mass_error": round_float(sum_mass / n_eval),
        "over_edit": round_float(sum_over / n_eval),
    }


def select_mask_candidate(
    candidates: list[dict[str, Any]],
    conditioning: list[dict[str, Any]],
    cond_baselines: list[list[list[int]]],
    arm: str,
    master_seed: int,
    lane: str,
    task_id: str,
    query_index: int,
    mask_seed: int,
    max_steps_mask: int,
    device: torch.device,
) -> dict[str, Any]:
    """Per spec §"Mask Selection": LOCO scoring (k>=3) else all-cells; primary
    F1, then nonmodal recall, precision, mass-error, over-edit, family index,
    SHA-256 tie-break key. Also returns `oracle_f1` for the diagnostic regret
    metric."""
    if not candidates:
        return {"selected": None, "candidates": [], "low_k_mask_selection": False}
    n = len(conditioning)
    use_loco = n >= 3
    low_k = not use_loco
    scored: list[dict[str, Any]] = []
    for c in candidates:
        s = _score_mask_candidate_on_pairs(c, conditioning, cond_baselines, use_loco, arm, mask_seed, max_steps_mask, device)
        family_idx = MASK_FAMILIES.index(c["family"])
        tk = hashlib.sha256(
            f"arc-p3d-mask-target-v3\0{master_seed}\0{lane}\0{task_id}\0{query_index}\0{arm}\0{c['family']}|{c['id']}".encode("utf-8")
        ).hexdigest()
        scored.append({"family": c["family"], "id": c["id"], "mask": c["mask"], "score": s, "family_index": family_idx, "tiebreak_key": tk})
    # Sort key: (-f1, -nonmodal_recall, -precision, +mass_error, +over_edit, +family_index, +tiebreak_key)
    scored.sort(key=lambda r: (-r["score"]["f1"], -r["score"]["nonmodal_recall"], -r["score"]["precision"], r["score"]["mass_error"], r["score"]["over_edit"], r["family_index"], r["tiebreak_key"]))
    top = scored[0]
    return {
        "selected": top,
        "candidates": scored,
        "low_k_mask_selection": low_k,
        "top_f1": top["score"]["f1"],
    }


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


def rare_edit_color_recall_metric(pred_colors: list[list[int]], target: list[list[int]], baseline: list[list[int]], mask: list[list[bool]]) -> float:
    """Recall on cells where the gold target differs from the modal target edit color."""
    th, tw = shape_of(target)
    bh, bw = shape_of(baseline)
    # Identify gold edit cells and their colors
    gold = []
    for y in range(min(th, bh)):
        for x in range(min(tw, bw)):
            if baseline[y][x] != target[y][x]:
                gold.append(target[y][x])
    if not gold:
        return 1.0
    modal = _modal_color(gold)
    rare_total = 0
    rare_hit = 0
    for y in range(min(th, bh, len(pred_colors), len(mask))):
        for x in range(min(tw, bw, len(pred_colors[0]) if pred_colors else 0, len(mask[0]) if mask else 0)):
            if baseline[y][x] != target[y][x] and target[y][x] != modal:
                rare_total += 1
                if mask[y][x] and pred_colors[y][x] == target[y][x]:
                    rare_hit += 1
    if rare_total == 0:
        return 1.0
    return rare_hit / rare_total


def oracle_rule_accuracy_metric(candidates: list[dict[str, Any]], baseline: list[list[int]], target: list[list[int]], query_input: list[list[int]]) -> float:
    """Best candidate-rule accuracy on the target gold-edit cells. Diagnostic only."""
    gold_edits = _gold_edits_in_pair(baseline, target)
    if not gold_edits:
        return 1.0
    best = 0.0
    bh = len(baseline)
    bw = len(baseline[0]) if bh else 0
    full_mask = [[True] * bw for _ in range(bh)]
    for cand in candidates:
        pred = _predict_with_rule(cand, query_input, baseline, full_mask)
        correct = sum(1 for oy, ox, t in gold_edits if 0 <= oy < len(pred) and 0 <= ox < len(pred[0]) and pred[oy][ox] == t)
        acc = correct / len(gold_edits)
        if acc > best:
            best = acc
    return best


def _mask_iou(pred: list[list[bool]], gold: list[list[bool]]) -> float:
    s = _mask_score(pred, gold)
    return s["iou"]


def mask_oracle_f1_on_target(candidates: list[dict[str, Any]], gold_target_mask: list[list[bool]]) -> float:
    """Best candidate F1 vs the TARGET gold mask (diagnostic only, NOT for branch decision)."""
    best = 0.0
    for c in candidates:
        s = _mask_score(c["mask"], gold_target_mask)
        if s["f1"] > best:
            best = s["f1"]
    return best


def mask_oracle_exact_nonbaseline(
    candidates: list[dict[str, Any]],
    rule_selection: dict[str, Any],
    query_input: list[list[int]],
    baseline: list[list[int]],
    target: list[list[int]],
) -> bool:
    """Diagnostic: would any candidate mask, paired with the selected color rule,
    have produced exact non-baseline reconstruction? NOT for branch decision."""
    base_eq_target = grid_equal(baseline, target)
    if base_eq_target:
        return False  # baseline-exact is excluded by the spec definition
    for c in candidates:
        colors = predict_query_edit_colors(rule_selection, query_input, baseline, c["mask"])
        pred = apply_edit(baseline, c["mask"], colors)
        if grid_equal(pred, target):
            return True
    return False


def mask_conditioned_color_accuracy_metric(pred_colors: list[list[int]], target: list[list[int]], baseline: list[list[int]], pred_mask: list[list[bool]]) -> float:
    """Color accuracy only on cells where BOTH the predicted mask says edit AND the target gold mask says edit."""
    th, tw = shape_of(target)
    bh, bw = shape_of(baseline)
    matched = 0
    correct = 0
    for y in range(min(th, bh, len(pred_mask), len(pred_colors))):
        for x in range(min(tw, bw, len(pred_mask[0]) if pred_mask else 0, len(pred_colors[0]) if pred_colors else 0)):
            target_edit = baseline[y][x] != target[y][x]
            pred_edit = pred_mask[y][x]
            if target_edit and pred_edit:
                matched += 1
                if pred_colors[y][x] == target[y][x]:
                    correct += 1
    if matched == 0:
        return 1.0
    return correct / matched


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
    """Per spec §"Quarantine Labels" for the mask-target variant — 16 labels.
    Order is the priority for the primary label."""
    if record["grid_exact"]:
        return ""
    # Pre-conditions inherited from base Branch D
    if not record["shape_exact"]:
        return "baseline_shape_failure"
    if record["baseline_residual_mass"] > 0.50:
        return "baseline_canvas_failure"
    # New mask-target decomposition (spec §"Quarantine Labels"):
    mask_oracle = record.get("mask_oracle_candidate_f1", 0.0)
    mask_selected = record.get("mask_candidate_f1", 0.0)
    if mask_oracle < 0.50:
        return "mask_candidate_coverage_failure"
    if mask_selected < 0.50:
        return "mask_selection_failure"
    mask_precision = record.get("mask_candidate_precision", 1.0)
    if mask_precision < 0.50 or record.get("over_edit_rate", 0.0) > 0.50:
        return "mask_overedit_failure"
    if record.get("mask_candidate_recall", 1.0) < 0.50:
        return "mask_underdetection_failure"
    # Inherited color-rule labels
    color_oracle = record.get("color_oracle_rule_accuracy", 0.0)
    color_selected = record.get("edit_color_rule_accuracy", 0.0)
    if color_oracle < 0.50:
        return "color_rule_bank_coverage_failure"
    if color_selected < 0.50:
        return "color_rule_selection_failure"
    # If mask is good enough but color rule is < 0.50 → edit_color_rule_failure
    if mask_selected >= 0.50 and color_selected < 0.50:
        return "edit_color_rule_failure"
    # Source-binding heuristic across both mask and color
    source_binding_color = {"baseline_color_map", "input_nn_color_map", "input_patch_majority_map",
                            "baseline_to_input_pair_map", "object_role_color_map", "nearest_edited_neighbor_color"}
    source_binding_mask = {"source_color_mask", "source_color_pair_mask", "object_role_mask"}
    mask_family = record.get("mask_candidate_family", "")
    color_family = record.get("color_rule_family", "")
    if (color_family in source_binding_color or mask_family in source_binding_mask) and record.get("mask_conditioned_color_accuracy", 1.0) < 0.50:
        return "source_binding_failure"
    if record.get("rare_edit_color_recall", 1.0) < 0.25:
        return "rare_color_failure"
    if record.get("no_conditioning_edits"):
        return "no_conditioning_edits"
    if conditioning_n < 3:
        return "conditioning_starvation"
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
    lodo = nonbaseline_exact_task_count(per_task_rows, "test_lodo", "raw_grid_edit_mask_v3")
    pt = nonbaseline_exact_task_count(per_task_rows, "pttest", "raw_grid_edit_mask_v3")
    if lodo >= 1 and pt >= 1:
        return {
            "gate": "raw_grid_edit_mask_v3_arena_open",
            "test_lodo_nonbaseline_exact_tasks": lodo,
            "pttest_nonbaseline_exact_tasks": pt,
        }
    return {
        "gate": "branch_d_mask_target_full_grid_floor",
        "test_lodo_nonbaseline_exact_tasks": lodo,
        "pttest_nonbaseline_exact_tasks": pt,
    }


def adjudicate_branch_d_mask_target(per_task_rows: list[dict[str, Any]], per_lane_rows: list[dict[str, Any]], arena: dict[str, Any]) -> dict[str, Any]:
    if arena["gate"] != "raw_grid_edit_mask_v3_arena_open":
        return {
            "branch": "branch_d_mask_target_full_grid_floor",
            "reason": "raw_grid_edit_mask_v3 did not open the non-baseline arena (>=1 non-baseline exact task on each held-out lane required)",
        }
    raw_lodo = nonbaseline_exact_task_count(per_task_rows, "test_lodo", "raw_grid_edit_mask_v3")
    raw_pt = nonbaseline_exact_task_count(per_task_rows, "pttest", "raw_grid_edit_mask_v3")
    sig_lodo = nonbaseline_exact_task_count(per_task_rows, "test_lodo", "signature_palette_edit_mask_v3")
    sig_pt = nonbaseline_exact_task_count(per_task_rows, "pttest", "signature_palette_edit_mask_v3")
    by_la = {(r["lane"], r["arm"]): r for r in per_lane_rows}
    raw_lodo_r = by_la.get(("test_lodo", "raw_grid_edit_mask_v3"), {})
    sig_lodo_r = by_la.get(("test_lodo", "signature_palette_edit_mask_v3"), {})
    raw_pt_r = by_la.get(("pttest", "raw_grid_edit_mask_v3"), {})
    sig_pt_r = by_la.get(("pttest", "signature_palette_edit_mask_v3"), {})
    def fnum(d, k):
        return float(d.get(k) or 0.0)
    # Per spec §"Branch D Mask-Targeted Adjudication": all gaps must be <= 0.10.
    f1_gap_lodo = fnum(raw_lodo_r, "mask_candidate_f1_mean") - fnum(sig_lodo_r, "mask_candidate_f1_mean")
    f1_gap_pt = fnum(raw_pt_r, "mask_candidate_f1_mean") - fnum(sig_pt_r, "mask_candidate_f1_mean")
    nonmodal_gap_lodo = fnum(raw_lodo_r, "mask_nonmodal_edit_recall_mean") - fnum(sig_lodo_r, "mask_nonmodal_edit_recall_mean")
    nonmodal_gap_pt = fnum(raw_pt_r, "mask_nonmodal_edit_recall_mean") - fnum(sig_pt_r, "mask_nonmodal_edit_recall_mean")
    over_gap_lodo = fnum(sig_lodo_r, "over_edit_rate_mean") - fnum(raw_lodo_r, "over_edit_rate_mean")
    over_gap_pt = fnum(sig_pt_r, "over_edit_rate_mean") - fnum(raw_pt_r, "over_edit_rate_mean")
    mask_regret_gap_lodo = fnum(sig_lodo_r, "mask_selection_regret_mean") - fnum(raw_lodo_r, "mask_selection_regret_mean")
    mask_regret_gap_pt = fnum(sig_pt_r, "mask_selection_regret_mean") - fnum(raw_pt_r, "mask_selection_regret_mean")
    color_acc_gap_lodo = fnum(raw_lodo_r, "edit_color_rule_accuracy_mean") - fnum(sig_lodo_r, "edit_color_rule_accuracy_mean")
    color_acc_gap_pt = fnum(raw_pt_r, "edit_color_rule_accuracy_mean") - fnum(sig_pt_r, "edit_color_rule_accuracy_mean")
    if (
        sig_lodo >= 1 and sig_pt >= 1
        and (raw_lodo - sig_lodo) <= 1 and (raw_pt - sig_pt) <= 1
        and f1_gap_lodo <= 0.10 and f1_gap_pt <= 0.10
        and nonmodal_gap_lodo <= 0.10 and nonmodal_gap_pt <= 0.10
        and over_gap_lodo <= 0.10 and over_gap_pt <= 0.10
        and mask_regret_gap_lodo <= 0.10 and mask_regret_gap_pt <= 0.10
        and color_acc_gap_lodo <= 0.10 and color_acc_gap_pt <= 0.10
    ):
        return {
            "branch": "branch_d_mask_target_support",
            "reason": "signature_palette_edit_mask_v3 opens the arena and meets all support thresholds (exact task delta <= 1; mask F1 gap, nonmodal-edit recall gap, over-edit gap, mask_selection_regret gap, and edit_color_rule_accuracy gap each <= 0.10)",
            "raw_grid_edit_mask_v3": {"test_lodo": raw_lodo, "pttest": raw_pt},
            "signature_palette_edit_mask_v3": {"test_lodo": sig_lodo, "pttest": sig_pt},
        }
    return {
        "branch": "branch_d_mask_target_bounded_failure",
        "reason": "raw_grid_edit_mask_v3 opened the non-baseline arena but signature_palette_edit_mask_v3 did not satisfy the support thresholds",
        "raw_grid_edit_mask_v3": {"test_lodo": raw_lodo, "pttest": raw_pt},
        "signature_palette_edit_mask_v3": {"test_lodo": sig_lodo, "pttest": sig_pt},
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


# Alias used by aggregator rate computations to coerce string CSV cells back to bool.
_truthy = _parse_bool


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
        "maskModelSpec", "shapeRules", "canvasRules", "maskThresholds", "ruleFamilies", "ensembleTieTolerance", "ensembleMinMembers", "maskFamilies", "maskMorphOps", "maskPatchThresholds", "legacyMlpThresholds",
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
    runner_path = "docs/prereg/arc/phase3d_mask_target_v3.py"
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
            "shapeRules, canvasRules, maskThresholds, ruleFamilies) IS equal across "
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
            "color_rule_sel_rows": read_csv_dicts(d / "color_rule_selection.csv"),
            "color_rule_cand_rows": read_csv_dicts(d / "color_rule_candidates.csv"),
            "mask_cand_sel_rows": read_csv_dicts(d / "mask_candidate_selection.csv"),
            "mask_cand_rows": read_csv_dicts(d / "mask_candidates.csv"),
        })

    mixed_audit = assert_shard_consistency(shards, repo_root=repo_root, allow_mixed_commits=args.allow_mixed_commits)
    shards.sort(key=lambda s: (s["manifest"]["shardArm"], s["manifest"]["shardSeed"]))

    # Concatenate raw shard outputs.
    per_instance_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    baseline_sel_rows: list[dict[str, Any]] = []
    edit_metrics_rows: list[dict[str, Any]] = []
    color_rule_sel_rows: list[dict[str, Any]] = []
    color_rule_cand_rows: list[dict[str, Any]] = []
    mask_cand_sel_rows: list[dict[str, Any]] = []
    mask_cand_rows: list[dict[str, Any]] = []
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
                ("color_rule_ensemble", _parse_bool),
                ("edit_color_rule_accuracy", float),
                ("rare_edit_color_recall", float),
                ("color_rule_candidate_count", int),
                ("low_k_rule_selection", _parse_bool),
                ("no_conditioning_edits", _parse_bool),
                ("color_oracle_rule_accuracy", float),
                ("rule_selection_regret", float),
                ("mask_conditioned_color_accuracy", float),
                ("mask_candidate_count", int),
                ("low_k_mask_selection", _parse_bool),
                ("mask_candidate_f1", float),
                ("mask_candidate_precision", float),
                ("mask_candidate_recall", float),
                ("mask_nonmodal_edit_recall", float),
                ("mask_iou", float),
                ("mask_mass_error", float),
                ("mask_oracle_candidate_f1", float),
                ("mask_selection_regret", float),
                ("mask_oracle_exact_nonbaseline", _parse_bool),
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
        color_rule_sel_rows.extend(sh["color_rule_sel_rows"])
        color_rule_cand_rows.extend(sh["color_rule_cand_rows"])
        mask_cand_sel_rows.extend(sh["mask_cand_sel_rows"])
        mask_cand_rows.extend(sh["mask_cand_rows"])

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
                "edit_color_rule_accuracy_mean": round_float(sum(float(r.get("edit_color_rule_accuracy") or 0.0) for r in group) / len(group)),
                "rare_edit_color_recall_mean": round_float(sum(float(r.get("rare_edit_color_recall") or 0.0) for r in group) / len(group)),
                "rule_selection_regret_mean": round_float(sum(float(r.get("rule_selection_regret") or 0.0) for r in group) / len(group)),
                "low_k_rule_selection_rate": round_float(sum(1 for r in group if _truthy(r.get("low_k_rule_selection"))) / len(group)),
                "no_conditioning_edits_rate": round_float(sum(1 for r in group if _truthy(r.get("no_conditioning_edits"))) / len(group)),
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
                "edit_color_rule_accuracy_mean": round_float(sum(float(r.get("edit_color_rule_accuracy") or 0.0) for r in group) / len(group)),
                "rare_edit_color_recall_mean": round_float(sum(float(r.get("rare_edit_color_recall") or 0.0) for r in group) / len(group)),
                "rule_selection_regret_mean": round_float(sum(float(r.get("rule_selection_regret") or 0.0) for r in group) / len(group)),
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
                "edit_color_rule_accuracy_mean": round_float(sum(float(r.get("edit_color_rule_accuracy") or 0.0) for r in group) / len(group)),
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
    branch = adjudicate_branch_d_mask_target(per_task_rows_agg, scores, arena)

    ref_manifest = shards[0]["manifest"]
    drop_keys = {"mode", "shardArm", "shardSeed", "seedSlateOriginal", "armsOriginal",
                 "armsEffective", "seedsEffective", "generatedAt", "completedAt",
                 "command", "tool", "outDir", "instanceCount", "perSeedValidationMetrics",
                 "selectedSeedByArm", "arenaGate", "branchAdjudication", "elapsedSecondsTotal"}
    merged_manifest = {k: v for k, v in ref_manifest.items() if k not in drop_keys}
    merged_manifest.update({
        "generatedAt": min(sh["manifest"]["generatedAt"] for sh in shards),
        "completedAt": iso_now(),
        "tool": "docs/prereg/arc/phase3d_mask_target_v3.py (merge)",
        "command": [sys.executable, "docs/prereg/arc/phase3d_mask_target_v3.py", *sys.argv[1:]],
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
    write_csv(out_dir / "color_rule_selection.csv", color_rule_sel_rows, COLOR_RULE_SEL_COLS)
    write_csv(out_dir / "color_rule_candidates.csv", color_rule_cand_rows, COLOR_RULE_CAND_COLS)
    write_csv(out_dir / "mask_candidate_selection.csv", mask_cand_sel_rows, MASK_CAND_SEL_COLS)
    write_csv(out_dir / "mask_candidates.csv", mask_cand_rows, MASK_CAND_COLS)
    write_csv(out_dir / "learning_curves.csv", learning_rows, LEARNING_COLS)
    write_csv(out_dir / "seed_stability.csv", seed_stability_rows, SEED_STABILITY_COLS)
    write_csv(out_dir / "quarantine_log.csv", quarantine_rows, QUARANTINE_COLS)
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    write_json(out_dir / "phase3d_mask_target_receipt.json", {
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
        + " ".join([sys.executable, "docs/prereg/arc/phase3d_mask_target_v3.py", *sys.argv[1:]])
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


def base_manifest(args, started_at, git, data_dir, register_path, out_dir, register_hash, data_hash, spec_hash, parent_spec_hash, base_branch_d_spec_hash, prior_variant_spec_hash, instance_count) -> dict[str, Any]:
    return {
        "generatedAt": started_at,
        "completedAt": None,
        "tool": "docs/prereg/arc/phase3d_mask_target_v3.py",
        "command": [sys.executable, "docs/prereg/arc/phase3d_mask_target_v3.py", *sys.argv[1:]],
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
        "variantVersion": VARIANT_VERSION,
        "specPath": "docs/prereg/arc/PHASE3D_MASK_TARGET_VARIANT_SPEC.md",
        "specHash": spec_hash,
        "parentSpecPath": "docs/prereg/arc/PHASE3_SUFFICIENCY_SPEC.md",
        "parentSpecHash": parent_spec_hash,
        "baseBranchDSpecPath": "docs/prereg/arc/PHASE3D_DIFFERENT_FRAMING_SPEC.md",
        "baseBranchDSpecHash": base_branch_d_spec_hash,
        "priorVariantSpecPath": "docs/prereg/arc/PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md",
        "priorVariantSpecHash": prior_variant_spec_hash,
        "registerHash": register_hash,
        "dataDirHash": data_hash,
        "pythonVersion": sys.version,
        "torchVersion": torch.__version__,
        "platform": platform.platform(),
        "maskModelSpec": MASK_MODEL_SPEC,
        "shapeRules": SHAPE_RULES,
        "canvasRules": CANVAS_RULES,
        "maskThresholds": MASK_THRESHOLDS,
        "ruleFamilies": RULE_FAMILIES,
        "maskFamilies": MASK_FAMILIES,
        "maskMorphOps": MASK_MORPH_OPS,
        "maskPatchThresholds": MASK_PATCH_THRESHOLDS,
        "legacyMlpThresholds": LEGACY_MLP_THRESHOLDS,
        "ensembleTieTolerance": ENSEMBLE_TIE_TOLERANCE,
        "ensembleMinMembers": ENSEMBLE_MIN_MEMBERS,
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
    # Inherited color-rule variant columns:
    "color_rule_family", "color_rule_id", "color_rule_ensemble",
    "edit_color_rule_accuracy", "rare_edit_color_recall",
    "color_rule_candidate_count", "low_k_rule_selection", "no_conditioning_edits",
    "color_oracle_rule_accuracy", "rule_selection_regret",
    "mask_conditioned_color_accuracy",
    # NEW mask-target variant columns:
    "mask_candidate_family", "mask_candidate_id", "mask_candidate_count",
    "low_k_mask_selection",
    "mask_candidate_f1", "mask_candidate_precision", "mask_candidate_recall",
    "mask_nonmodal_edit_recall", "mask_iou", "mask_mass_error",
    "mask_oracle_candidate_f1", "mask_selection_regret", "mask_oracle_exact_nonbaseline",
    "quarantine_label",
]
PER_TASK_COLS = ["lane", "task_id", "primary_prior", "predicted_boundary", "arm", "selected_seed", "instance_count", "grid_exact_any_rate", "nonbaseline_exact_any_rate", "baseline_exact_any_rate", "shape_exact_rate", "palette_exact_rate", "pixel_accuracy_mean", "edit_color_rule_accuracy_mean", "rare_edit_color_recall_mean", "rule_selection_regret_mean", "mask_candidate_f1_mean", "mask_nonmodal_edit_recall_mean", "mask_selection_regret_mean"]
PER_PRIOR_COLS = ["lane", "primary_prior", "arm", "instance_count", "grid_exact_any_rate", "nonbaseline_exact_any_rate", "edit_mask_f1_mean", "edit_color_rule_accuracy_mean", "mask_candidate_f1_mean"]
SCORE_COLS = ["lane", "arm", "selected_seed", "task_count", "instance_count", "grid_exact_any_rate", "baseline_exact_any_rate", "nonbaseline_exact_any_rate", "shape_exact_rate", "palette_exact_rate", "pixel_accuracy_mean", "edit_mask_f1_mean", "minority_edit_recall_mean", "over_edit_rate_mean", "predicted_edit_mass_mean", "edit_color_rule_accuracy_mean", "rare_edit_color_recall_mean", "rule_selection_regret_mean", "low_k_rule_selection_rate", "no_conditioning_edits_rate", "mask_candidate_f1_mean", "mask_nonmodal_edit_recall_mean", "mask_selection_regret_mean", "low_k_mask_selection_rate"]
BASELINE_SEL_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "shape_rule", "canvas_rule", "background_color", "mean_conditioning_residual", "max_conditioning_residual"]
EDIT_METRICS_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "selected_threshold", "edit_mask_f1", "edit_mask_precision", "edit_mask_recall", "minority_edit_recall", "over_edit_rate", "under_edit_rate", "edit_color_accuracy", "predicted_edit_mass", "target_edit_mass", "edit_color_rule_accuracy", "rare_edit_color_recall", "mask_conditioned_color_accuracy", "mask_candidate_f1", "mask_nonmodal_edit_recall", "mask_iou", "mask_selection_regret"]
COLOR_RULE_SEL_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "color_rule_family", "color_rule_id", "ensemble", "ensemble_member_count", "candidate_count", "top_accuracy", "selected_accuracy", "rare_recall", "halluc_rate", "low_k_rule_selection"]
COLOR_RULE_CAND_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "color_rule_family", "color_rule_id", "accuracy", "rare_recall", "halluc_rate", "family_index", "gold_edit_count", "rare_color_count"]
MASK_CAND_SEL_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "mask_candidate_family", "mask_candidate_id", "candidate_count", "top_f1", "selected_f1", "selected_precision", "selected_recall", "selected_nonmodal_recall", "low_k_mask_selection"]
MASK_CAND_COLS = ["instance_id", "lane", "task_id", "arm", "seed", "mask_candidate_family", "mask_candidate_id", "f1", "precision", "recall", "nonmodal_recall", "mass_error", "over_edit", "family_index"]
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
    write_csv(out_dir / "color_rule_selection.csv", [], COLOR_RULE_SEL_COLS)
    write_csv(out_dir / "color_rule_candidates.csv", [], COLOR_RULE_CAND_COLS)
    write_csv(out_dir / "mask_candidate_selection.csv", [], MASK_CAND_SEL_COLS)
    write_csv(out_dir / "mask_candidates.csv", [], MASK_CAND_COLS)
    write_csv(out_dir / "learning_curves.csv", [], LEARNING_COLS)
    write_csv(out_dir / "seed_stability.csv", [], SEED_STABILITY_COLS)
    write_csv(out_dir / "quarantine_log.csv", [], QUARANTINE_COLS)
    write_jsonl(out_dir / "residuals.jsonl", [])
    write_json(out_dir / "phase3d_mask_target_receipt.json", {"manifest": manifest, "arenaGate": None, "branchAdjudication": None})
    (out_dir / "branch_adjudication.md").write_text(
        "# Phase 3D Edit-Color-Rule Branch Adjudication\n\nDry run / empty receipt. No arena-gate or branch decision.\n",
        encoding="utf-8",
    )
    (out_dir / "commands.md").write_text(
        "# Phase 3D edit-color-rule commands\n\nDry run / empty receipt. No execution command captured.\n",
        encoding="utf-8",
    )
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def select_seed_for_arm(arm: str, per_seed_metrics: dict[int, dict[str, Any]]) -> int:
    """Per spec §"Seed Slate" inherits Phase 3D's rule but with color-rule metrics:
    rank by (-val_nonbaseline_exact_count, -val_edit_color_rule_accuracy,
    -val_rare_edit_color_recall, +val_rule_selection_regret,
    +val_edit_mask_f1_neg, +seed)."""
    if not per_seed_metrics:
        raise ValueError("no candidates to select from")
    def key(seed: int):
        m = per_seed_metrics[seed]
        return (
            -m.get("val_nonbaseline_exact_count", 0),
            -m.get("val_edit_color_rule_accuracy", 0.0),
            -m.get("val_rare_edit_color_recall", 0.0),
            m.get("val_rule_selection_regret", 1.0),
            -m.get("val_edit_mask_f1", 0.0),
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
    spec_path = Path(__file__).resolve().parent / "PHASE3D_MASK_TARGET_VARIANT_SPEC.md"
    prior_variant_spec_path = Path(__file__).resolve().parent / "PHASE3D_EDIT_COLOR_RULE_VARIANT_SPEC.md"
    parent_spec_path = Path(__file__).resolve().parent / "PHASE3_SUFFICIENCY_SPEC.md"
    base_branch_d_spec_path = Path(__file__).resolve().parent / "PHASE3D_DIFFERENT_FRAMING_SPEC.md"
    spec_hash = sha256_file(spec_path) if spec_path.exists() else "NA"
    parent_spec_hash = sha256_file(parent_spec_path) if parent_spec_path.exists() else "NA"
    base_branch_d_spec_hash = sha256_file(base_branch_d_spec_path) if base_branch_d_spec_path.exists() else "NA"
    prior_variant_spec_hash = sha256_file(prior_variant_spec_path) if prior_variant_spec_path.exists() else "NA"

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

    manifest = base_manifest(args, started_at, git, data_dir, register_path, out_dir, register_hash, data_hash, spec_hash, parent_spec_hash, base_branch_d_spec_hash, prior_variant_spec_hash, len(all_instances))
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
    # Variant: no color MLP. probe_steps still caps the mask MLP; color rule-bank
    # selection is deterministic and not step-limited.
    manifest["maxStepsEffective"] = {"mask": max_steps_mask, "color": "n/a (rule_bank)"}
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
    color_rule_sel_rows: list[dict[str, Any]] = []
    color_rule_cand_rows: list[dict[str, Any]] = []
    mask_cand_sel_rows: list[dict[str, Any]] = []
    mask_cand_rows: list[dict[str, Any]] = []

    per_arm_validation_metrics: dict[str, dict[int, dict[str, Any]]] = {arm: {seed: {} for seed in seeds} for arm in arms}
    per_instance_seed_outcomes: dict[tuple[str, str], dict[int, bool]] = {}
    elapsed_total = 0.0

    for arm in arms:
        for seed in seeds:
            for inst in held_out_instances:
                start = time.perf_counter()
                # 1. Baseline candidate selection (inherited unchanged from Phase 3D)
                candidate = select_baseline_candidate(inst.query_input, inst.conditioning, arm)
                cond_baselines = [apply_baseline(p["input"], candidate, [q for q in inst.conditioning if q is not p], arm) for p in inst.conditioning]
                query_baseline = apply_baseline(inst.query_input, candidate, inst.conditioning, arm)
                mask_seed = derive_seed(seed, inst.lane, inst.task_id, inst.query_index, arm, "mask")
                # 2. Build color rule candidates + select (inherited from variant)
                rule_candidates = generate_candidate_rules(inst.conditioning, cond_baselines)
                rule_selection = select_color_rule(
                    rule_candidates, inst.conditioning, cond_baselines,
                    seed, inst.lane, inst.task_id, inst.query_index, arm,
                )
                # 3. Build MASK candidates from the 13 frozen families (per spec)
                # This is the variant's only point of departure from the color-rule v2 runner.
                mask_candidates, legacy_mlp_info = generate_mask_candidates(
                    arm, inst.query_input, query_baseline, inst.conditioning, cond_baselines,
                    mask_seed, max_steps_mask, device,
                )
                # 4. Select mask via LOCO scoring + tie-break chain (no learned cross-task selector)
                mask_selection = select_mask_candidate(
                    mask_candidates, inst.conditioning, cond_baselines, arm,
                    seed, inst.lane, inst.task_id, inst.query_index,
                    mask_seed, max_steps_mask, device,
                )
                selected_mask_entry = mask_selection.get("selected") or {}
                mask_family = selected_mask_entry.get("family", "") if selected_mask_entry else ""
                mask_id = selected_mask_entry.get("id", "") if selected_mask_entry else ""
                query_mask = selected_mask_entry.get("mask") if selected_mask_entry else [[False] * (len(query_baseline[0]) if query_baseline else 0) for _ in range(len(query_baseline))]
                # threshold is not used by candidate-bank selection; record as 'n/a' sentinel for the schema
                threshold = -1.0
                threshold_audit = {"note": "candidate_bank_selection_used; no threshold sweep"}
                mask_info = legacy_mlp_info  # learning curves come from the legacy MLP family
                # 5. Apply selected color rule to selected query mask
                query_colors = predict_query_edit_colors(rule_selection, inst.query_input, query_baseline, query_mask)
                predicted_grid = apply_edit(query_baseline, query_mask, query_colors)
                # 8. Score
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
                # Inherited color-rule variant metrics
                ec_rule_acc = ec_acc
                rare_recall = rare_edit_color_recall_metric(query_colors, inst.target_output, query_baseline, query_mask)
                oracle_acc = oracle_rule_accuracy_metric(rule_candidates, query_baseline, inst.target_output, inst.query_input)
                regret = max(0.0, oracle_acc - ec_rule_acc)
                mask_cond_acc = mask_conditioned_color_accuracy_metric(query_colors, inst.target_output, query_baseline, query_mask)
                selected_rule = rule_selection.get("selected") or {}
                rule_family = selected_rule.get("family", "") if selected_rule else ""
                rule_id = selected_rule.get("id", "") if selected_rule else ""
                # NEW mask-target metrics
                gold_target_mask = _conditioning_gold_mask(query_baseline, inst.target_output)
                mask_target_score = _mask_score(query_mask, gold_target_mask)
                mask_target_f1 = mask_target_score["f1"]
                mask_target_precision = mask_target_score["precision"]
                mask_target_recall = mask_target_score["recall"]
                mask_target_iou = mask_target_score["iou"]
                mask_target_mass_error = mask_target_score["mass_error"]
                # nonmodal-edit recall on TARGET
                _all_target_edit_colors: list[int] = []
                for _oy, _ox, _t in _gold_edits_in_pair(query_baseline, inst.target_output):
                    _all_target_edit_colors.append(_t)
                _modal_target = _modal_color(_all_target_edit_colors) if _all_target_edit_colors else -1
                _nm_total = 0; _nm_hit = 0
                _gh = len(query_baseline); _gw = len(query_baseline[0]) if _gh else 0
                _th = len(inst.target_output); _tw = len(inst.target_output[0]) if _th else 0
                for _y in range(min(_gh, _th)):
                    for _x in range(min(_gw, _tw)):
                        if query_baseline[_y][_x] != inst.target_output[_y][_x] and inst.target_output[_y][_x] != _modal_target:
                            _nm_total += 1
                            if _y < len(query_mask) and _x < len(query_mask[0]) and query_mask[_y][_x]:
                                _nm_hit += 1
                mask_nonmodal_recall = (_nm_hit / _nm_total) if _nm_total else 1.0
                mask_oracle_f1 = mask_oracle_f1_on_target(mask_candidates, gold_target_mask)
                mask_regret = max(0.0, mask_oracle_f1 - mask_target_f1)
                mask_oracle_exact = mask_oracle_exact_nonbaseline(mask_candidates, rule_selection, inst.query_input, query_baseline, inst.target_output)
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
                    "color_rule_family": rule_family,
                    "color_rule_id": rule_id,
                    "color_rule_ensemble": rule_selection.get("ensemble", False),
                    "edit_color_rule_accuracy": round_float(ec_rule_acc),
                    "rare_edit_color_recall": round_float(rare_recall),
                    "color_rule_candidate_count": len(rule_candidates),
                    "low_k_rule_selection": rule_selection.get("low_k_rule_selection", False),
                    "no_conditioning_edits": rule_selection.get("no_conditioning_edits", False),
                    "color_oracle_rule_accuracy": round_float(oracle_acc),
                    "rule_selection_regret": round_float(regret),
                    "mask_conditioned_color_accuracy": round_float(mask_cond_acc),
                    # New mask-target variant fields:
                    "mask_candidate_family": mask_family,
                    "mask_candidate_id": mask_id,
                    "mask_candidate_count": len(mask_candidates),
                    "low_k_mask_selection": mask_selection.get("low_k_mask_selection", False),
                    "mask_candidate_f1": round_float(mask_target_f1),
                    "mask_candidate_precision": round_float(mask_target_precision),
                    "mask_candidate_recall": round_float(mask_target_recall),
                    "mask_nonmodal_edit_recall": round_float(mask_nonmodal_recall),
                    "mask_iou": round_float(mask_target_iou),
                    "mask_mass_error": round_float(mask_target_mass_error),
                    "mask_oracle_candidate_f1": round_float(mask_oracle_f1),
                    "mask_selection_regret": round_float(mask_regret),
                    "mask_oracle_exact_nonbaseline": mask_oracle_exact,
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
                    "edit_color_rule_accuracy": round_float(ec_rule_acc),
                    "rare_edit_color_recall": round_float(rare_recall),
                    "mask_conditioned_color_accuracy": round_float(mask_cond_acc),
                })
                top_acc = rule_selection.get("top_accuracy", 0.0) or 0.0
                selected_acc = 0.0
                rare = 0.0
                halluc = 0.0
                if selected_rule and selected_rule.get("family") != "ensemble_top3":
                    sel_score = selected_rule.get("score", {}) if isinstance(selected_rule, dict) else {}
                    selected_acc = sel_score.get("accuracy", 0.0)
                    rare = sel_score.get("rare_recall", 0.0)
                    halluc = sel_score.get("halluc_rate", 0.0)
                color_rule_sel_rows.append({
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "arm": arm,
                    "seed": seed,
                    "color_rule_family": rule_family,
                    "color_rule_id": rule_id,
                    "ensemble": rule_selection.get("ensemble", False),
                    "ensemble_member_count": len(rule_selection.get("members", [])),
                    "candidate_count": len(rule_candidates),
                    "top_accuracy": round_float(top_acc),
                    "selected_accuracy": round_float(selected_acc),
                    "rare_recall": round_float(rare),
                    "halluc_rate": round_float(halluc),
                    "low_k_rule_selection": rule_selection.get("low_k_rule_selection", False),
                })
                for cand in rule_selection.get("candidates", []):
                    cs = cand["score"]
                    color_rule_cand_rows.append({
                        "instance_id": inst.instance_id,
                        "lane": inst.lane,
                        "task_id": inst.task_id,
                        "arm": arm,
                        "seed": seed,
                        "color_rule_family": cand["family"],
                        "color_rule_id": cand["id"],
                        "accuracy": cs["accuracy"],
                        "rare_recall": cs["rare_recall"],
                        "halluc_rate": cs["halluc_rate"],
                        "family_index": cand["family_index"],
                        "gold_edit_count": cs["gold_edit_count"],
                        "rare_color_count": cs["rare_color_count"],
                    })
                # MASK candidate selection + candidates emission
                mask_sel_score = (selected_mask_entry.get("score") or {}) if selected_mask_entry else {}
                mask_cand_sel_rows.append({
                    "instance_id": inst.instance_id,
                    "lane": inst.lane,
                    "task_id": inst.task_id,
                    "arm": arm,
                    "seed": seed,
                    "mask_candidate_family": mask_family,
                    "mask_candidate_id": mask_id,
                    "candidate_count": len(mask_candidates),
                    "top_f1": round_float(mask_selection.get("top_f1", 0.0) or 0.0),
                    "selected_f1": round_float(mask_sel_score.get("f1", 0.0)),
                    "selected_precision": round_float(mask_sel_score.get("precision", 0.0)),
                    "selected_recall": round_float(mask_sel_score.get("recall", 0.0)),
                    "selected_nonmodal_recall": round_float(mask_sel_score.get("nonmodal_recall", 0.0)),
                    "low_k_mask_selection": mask_selection.get("low_k_mask_selection", False),
                })
                for cand in mask_selection.get("candidates", []):
                    cs = cand["score"]
                    mask_cand_rows.append({
                        "instance_id": inst.instance_id,
                        "lane": inst.lane,
                        "task_id": inst.task_id,
                        "arm": arm,
                        "seed": seed,
                        "mask_candidate_family": cand["family"],
                        "mask_candidate_id": cand["id"],
                        "f1": cs["f1"],
                        "precision": cs["precision"],
                        "recall": cs["recall"],
                        "nonmodal_recall": cs["nonmodal_recall"],
                        "mass_error": cs["mass_error"],
                        "over_edit": cs["over_edit"],
                        "family_index": cand["family_index"],
                    })
                for h in mask_info.get("history", []):
                    learning_rows.append({"instance_id": inst.instance_id, "arm": arm, "seed": seed, "model_kind": "mask", "step": h["step"], "loss": h["loss"]})
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
                    "selected_color_rule": {"family": rule_family, "id": rule_id, "ensemble": rule_selection.get("ensemble", False)},
                })
                per_instance_seed_outcomes.setdefault((arm, inst.instance_id), {})[seed] = grid_exact and not baseline_exact

                if inst.lane.startswith("validation_"):
                    bucket = per_arm_validation_metrics[arm][seed].setdefault("counts", {
                        "nonbaseline_exact": 0, "n": 0,
                        "f1_sum": 0.0, "min_recall_sum": 0.0, "over_edit_sum": 0.0, "loss_sum": 0.0,
                        "color_rule_acc_sum": 0.0, "rare_recall_sum": 0.0, "regret_sum": 0.0,
                    })
                    bucket["n"] += 1
                    if nonbaseline_exact:
                        bucket["nonbaseline_exact"] += 1
                    bucket["f1_sum"] += em["edit_mask_f1"]
                    bucket["min_recall_sum"] += em["minority_edit_recall"]
                    bucket["over_edit_sum"] += em["over_edit_rate"]
                    bucket["loss_sum"] += mask_info.get("best_loss", 0.0)
                    bucket["color_rule_acc_sum"] += ec_rule_acc
                    bucket["rare_recall_sum"] += rare_recall
                    bucket["regret_sum"] += regret
            # Roll up per-seed validation summary
            bucket = per_arm_validation_metrics[arm][seed].get("counts", {
                "nonbaseline_exact": 0, "n": 0,
                "f1_sum": 0.0, "min_recall_sum": 0.0, "over_edit_sum": 0.0, "loss_sum": 0.0,
                "color_rule_acc_sum": 0.0, "rare_recall_sum": 0.0, "regret_sum": 0.0,
            })
            n = max(1, bucket["n"])
            per_arm_validation_metrics[arm][seed].update({
                "val_nonbaseline_exact_count": bucket["nonbaseline_exact"],
                "val_edit_mask_f1": round_float(bucket["f1_sum"] / n),
                "val_minority_edit_recall": round_float(bucket["min_recall_sum"] / n),
                "val_over_edit_rate": round_float(bucket["over_edit_sum"] / n),
                "val_loss": round_float(bucket["loss_sum"] / n),
                "val_edit_color_rule_accuracy": round_float(bucket["color_rule_acc_sum"] / n),
                "val_rare_edit_color_recall": round_float(bucket["rare_recall_sum"] / n),
                "val_rule_selection_regret": round_float(bucket["regret_sum"] / n),
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
                "edit_color_rule_accuracy_mean": round_float(sum(float(r.get("edit_color_rule_accuracy") or 0.0) for r in group) / len(group)),
                "rare_edit_color_recall_mean": round_float(sum(float(r.get("rare_edit_color_recall") or 0.0) for r in group) / len(group)),
                "rule_selection_regret_mean": round_float(sum(float(r.get("rule_selection_regret") or 0.0) for r in group) / len(group)),
                "low_k_rule_selection_rate": round_float(sum(1 for r in group if _truthy(r.get("low_k_rule_selection"))) / len(group)),
                "no_conditioning_edits_rate": round_float(sum(1 for r in group if _truthy(r.get("no_conditioning_edits"))) / len(group)),
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
                "edit_color_rule_accuracy_mean": round_float(sum(float(r.get("edit_color_rule_accuracy") or 0.0) for r in group) / len(group)),
                "rare_edit_color_recall_mean": round_float(sum(float(r.get("rare_edit_color_recall") or 0.0) for r in group) / len(group)),
                "rule_selection_regret_mean": round_float(sum(float(r.get("rule_selection_regret") or 0.0) for r in group) / len(group)),
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
                "edit_color_rule_accuracy_mean": round_float(sum(float(r.get("edit_color_rule_accuracy") or 0.0) for r in group) / len(group)),
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
        branch = adjudicate_branch_d_mask_target(per_task_rows, scores, arena)
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
    write_csv(out_dir / "color_rule_selection.csv", color_rule_sel_rows, COLOR_RULE_SEL_COLS)
    write_csv(out_dir / "color_rule_candidates.csv", color_rule_cand_rows, COLOR_RULE_CAND_COLS)
    write_csv(out_dir / "mask_candidate_selection.csv", mask_cand_sel_rows, MASK_CAND_SEL_COLS)
    write_csv(out_dir / "mask_candidates.csv", mask_cand_rows, MASK_CAND_COLS)
    write_csv(out_dir / "learning_curves.csv", learning_rows, LEARNING_COLS)
    write_csv(out_dir / "seed_stability.csv", seed_stability_rows, SEED_STABILITY_COLS)
    write_csv(out_dir / "quarantine_log.csv", quarantine_rows, QUARANTINE_COLS)
    write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    write_json(out_dir / "phase3d_mask_target_receipt.json", {
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
        + " ".join([sys.executable, "docs/prereg/arc/phase3d_mask_target_v3.py", *sys.argv[1:]])
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
