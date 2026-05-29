#!/usr/bin/env python
"""ARC Phase 3E relative-locality certificate runner.

Standalone rank-locality runner. It is NOT a decoder and NOT a Branch E solver.
It inherits the Phase 3E `program_sketch_v2` oracle and asks whether
`signature_palette_context` nearest neighbors are more program-sketch coherent
than metadata, raw-grid, random, stratified-random, and permutation controls.

Inherited verbatim (with header markers) from the frozen Phase 3E signature-fiber
certificate runner: the arc-p3-feature-v1 grid encoders, the context identity, the
context distance, and the k=3 cross-task fiber-locality machinery. Geometry
thresholds (epsilon_primary=0.05, k=3, signature_palette_context identity +
distance) are UNCHANGED and are NOT retuned.

New and replacing v1: the nine-facet `program_sketch_v2` labelers (raw grids
only; NEVER signature_palette / arm distances / decoder outputs), the three gate
tests, the v2 incompatibility rule, and the 7-branch precedence adjudication.

The Branch D baseline/mask/color bank functions remain in this file (inherited
with the seed) but are INERT: `program_sketch_v2` never calls them. They are
retained only so the frozen geometry helpers they share a module with stay
byte-faithful to the 3E runner.

Spec: docs/prereg/arc/PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md (filed 2026-05-29).
Parent certificate spec: docs/prereg/arc/PHASE3E_SIGNATURE_FIBER_CERTIFICATE_SPEC.md.
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
# Phase 3E signature-fiber certificate constants
# ============================================================================
FEATURE_SCHEMA_VERSION = "arc-p3-feature-v1"
LEARNER_VERSION = "relative_locality_certificate"
PROTOCOL_VERSION = "arc-p3e-relative-locality-v1"
RECEIPT_SCHEMA_VERSION = "arc-p3e-relative-locality-receipt-v1"

# Nine required program_sketch_v2 facets (spec §"Sketch Facets"). Frozen order.
SKETCH_FACETS = [
    "shape_relation",
    "palette_relation",
    "object_relation",
    "cardinality_relation",
    "completion_relation",
    "spatial_transform_relation",
    "symmetry_relation",
    "correspondence_basis",
    "rule_scope",
]
# Facet most directly associated with each registered Phase 0 prior, for the
# anti-prior-laundering "two extra concrete facets" rule.
PRIOR_ASSOCIATED_FACET = {
    "counting": "cardinality_relation",
    "symmetry": "symmetry_relation",
    "spatial_transform": "spatial_transform_relation",
    "local_completion": "completion_relation",
    "color_role": "palette_relation",
    "objectness": "object_relation",
}
# Anti-vacuity (spec §"Anti-Vacuity Gate").
VACUITY_MIN_NONEMPTY_FACETS = 4
VACUITY_OVERALL_MAX = 0.20
VACUITY_PER_PRIOR_MAX = 0.25
# Anti-prior-laundering (spec §"Anti-Prior-Laundering Gate").
LAUNDERING_MIN_EXTRA_FACETS = 2
LAUNDERING_MAX_VIOLATION_FRACTION = 0.10
# Anti-solver-leakage (spec §"Anti-Solver-Leakage Gate").
LEAKAGE_UNIQUE_CORE_MAX = 0.60
LEAKAGE_EXACT_LOOKUP_MAX = 0.20
# v2 incompatibility (spec §"v2 Incompatibility Rule").
V2_MIN_DISJOINT_FACETS = 4
# Non-informative facet tokens.
NONINFORMATIVE = {"none", "unknown"}

# Certificate arms (context identity + distance). The PRIMARY adjudication arm
# is signature_palette_context.
CERT_ARMS = [
    "signature_palette_context",
    "signature_only_context",
    "metadata_only_context",
    "raw_grid_context",
]
PRIMARY_CERT_ARM = "signature_palette_context"

# Frozen thresholds (spec §"Context Distance" / §"Branches"). Changing any of
# these after seeing pairwise distances requires a new append-only amendment.
EPSILON_PRIMARY = 0.05
EPSILON_EXACT = 0.0
EPSILON_STRICT = 0.025
EPSILON_LOOSE = 0.10
KNN_K = 3
LOCAL_INCOMPAT_MAX = 0.10           # fiber_locality_positive ceiling
FIDELITY_PASS_FRACTION = 0.50       # >=50% U_primary contexts must have fidelity-passing neighborhoods
LABEL_VACUITY_FRACTION = 0.30       # >30% with `none` in >=2 sketch sets -> deferred_label_vacuity
JACCARD_STRONG_GAP = 0.80           # diagnostic strong_program_sketch_gap

# program_sketch_v1 oracle thresholds (spec §"Target Labels").
PROGRAM_SKETCH_ORACLE_ARM = "raw_grid_edit_mask_v3"   # representation-neutral grid arm for Branch D banks
CANVAS_ORACLE_RESIDUAL_MAX = 0.25
MASK_ORACLE_F1_TARGET = 0.75
MASK_ORACLE_F1_COND = 0.50
COLOR_ORACLE_ACC_TARGET = 0.75
COLOR_ORACLE_ACC_COND = 0.50
# The legacy mask MLP family is one of the Branch D mask families; for the
# finite oracle audit it is capped low (it is not the certificate's subject).
ORACLE_MASK_MLP_STEPS = 50

# Branch D arm names retained for the inherited bank functions only.
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


# Fixed expansion batch tag (PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md).
EXPANSION_BATCH = "fiber_context_expansion_v1"


def sha256_expansion_split_by_task(rows: list[dict[str, str]]) -> dict[str, str]:
    """Spec PHASE0_CONTEXT_EXPANSION_FOR_FIBERS §"Context Universe" partition.

    For each primary_prior group of included tasks, sort task IDs by
    SHA-256("fiber_context_expansion_v1|" + task_id) and assign the first
    max(3, floor(n/3)) to validation, the remainder to test. There is no train
    split under this mode: every included task becomes validation or test. This
    only assigns the val/test partition for the expanded certificate; all
    Phase 3E geometry (identity, distance, k, epsilons, oracle, gates, barrier)
    is untouched.
    """
    by_prior: dict[str, list[str]] = {}
    for row in rows:
        by_prior.setdefault(row["primary_prior"], []).append(row["task_id"])
    out: dict[str, str] = {}
    for prior, task_ids in by_prior.items():
        ordered = sorted(task_ids, key=lambda tid: hashlib.sha256(f"{EXPANSION_BATCH}|{tid}".encode("utf-8")).hexdigest())
        n = len(ordered)
        n_val = min(n, max(3, n // 3))
        for idx, tid in enumerate(ordered):
            out[tid] = "validation" if idx < n_val else "test"
    return out


def load_tasks(data_dir: Path, register_path: Path, split_mode: str = "frozen_v2") -> tuple[list[Task], str, str]:
    register_text = register_path.read_text(encoding="utf-8-sig")
    rows = [row for row in csv.DictReader(register_text.splitlines()) if row["status"] == "include" and row["split"] == "training"]
    tasks: list[Task] = []
    file_hashes: list[dict[str, str]] = []
    if split_mode == "sha256_expansion":
        split_by_task = sha256_expansion_split_by_task(rows)
    else:
        split_by_task = expected_split_by_task()
    for row in rows:
        task_id = row["task_id"]
        if task_id not in split_by_task:
            raise SystemExit(
                f"task_id {task_id!r} has no split assignment under split_mode={split_mode!r}. "
                f"The expanded register requires --split-mode sha256_expansion."
            )
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
    SHA-256 tie-break key.

    PERFORMANCE: the LOCO fold candidate banks are generated ONCE per fold
    (not once per candidate per fold). Each fold bank is indexed by
    (family, id) -> mask so per-candidate scoring is an O(1) lookup. This
    avoids the O(C^2 * k) blowup of regenerating the bank for every candidate.
    """
    if not candidates:
        return {"selected": None, "candidates": [], "low_k_mask_selection": False}
    n = len(conditioning)
    use_loco = n >= 3
    low_k = not use_loco

    # Modal-edit-color across all conditioning (for nonmodal recall)
    all_colors: list[int] = []
    for b, p in zip(cond_baselines, conditioning):
        for _oy, _ox, t in _gold_edits_in_pair(b, p["output"]):
            all_colors.append(t)
    modal = _modal_color(all_colors) if all_colors else -1

    # Precompute, per conditioning pair i: the gold mask, and (LOCO) the fold
    # candidate bank indexed by (family,id). Generated ONCE per fold.
    fold_data: list[dict[str, Any]] = []
    for i, (pair, baseline) in enumerate(zip(conditioning, cond_baselines)):
        gold_pair_mask = _conditioning_gold_mask(baseline, pair["output"])
        # Nonmodal target cells for this pair
        bh = len(baseline); bw = len(baseline[0]) if bh else 0
        th = len(pair["output"]); tw = len(pair["output"][0]) if th else 0
        nonmodal_cells: list[tuple[int, int]] = []
        for y in range(min(bh, th)):
            for x in range(min(bw, tw)):
                if baseline[y][x] != pair["output"][y][x] and pair["output"][y][x] != modal:
                    nonmodal_cells.append((y, x))
        fold_bank: dict[tuple[str, str], list[list[bool]]] = {}
        if use_loco:
            loco_cond = [c for j, c in enumerate(conditioning) if j != i]
            loco_bls = [b for j, b in enumerate(cond_baselines) if j != i]
            loco_cands, _ = generate_mask_candidates(arm, pair["input"], baseline, loco_cond, loco_bls, mask_seed, max_steps_mask, device)
            for lc in loco_cands:
                fold_bank[(lc["family"], lc["id"])] = lc["mask"]
        fold_data.append({
            "gold": gold_pair_mask,
            "nonmodal_cells": nonmodal_cells,
            "bank": fold_bank,
            "baseline_shape": (bh, bw),
        })

    def _score_candidate(cand: dict[str, Any]) -> dict[str, Any]:
        sum_f1 = sum_prec = sum_rec = sum_nonmodal = sum_mass = sum_over = 0.0
        n_eval = 0
        for fd in fold_data:
            bh, bw = fd["baseline_shape"]
            if use_loco:
                pred_native = fd["bank"].get((cand["family"], cand["id"]))
                if pred_native is None:
                    # Candidate family/id absent from this fold's bank (data-dependent);
                    # fall back to projecting the query-frame mask.
                    pred_native = _project_mask_to_shape(cand["mask"], bh, bw)
            else:
                pred_native = _project_mask_to_shape(cand["mask"], bh, bw)
            s = _mask_score(pred_native, fd["gold"])
            sum_f1 += s["f1"]; sum_prec += s["precision"]; sum_rec += s["recall"]
            sum_mass += s["mass_error"]; sum_over += s["over_edit"]
            nm_total = len(fd["nonmodal_cells"]); nm_hit = 0
            for (y, x) in fd["nonmodal_cells"]:
                if y < len(pred_native) and x < len(pred_native[0]) and pred_native[y][x]:
                    nm_hit += 1
            sum_nonmodal += (nm_hit / nm_total) if nm_total else 1.0
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

    scored: list[dict[str, Any]] = []
    for c in candidates:
        s = _score_candidate(c)
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


# ============================================================================
# Certificate: context identity + distance
# ============================================================================
import itertools  # noqa: E402

_PROJ_CACHE: dict[str, dict[str, Any]] = {}


def _grid_key(grid: list[list[int]]) -> str:
    return json.dumps(grid, separators=(",", ":"))


def grid_proj(grid: list[list[int]]) -> dict[str, Any]:
    k = _grid_key(grid)
    p = _PROJ_CACHE.get(k)
    if p is None:
        p = project_grid_shadow(grid)
        p["_signatureHash"] = sha256_text(p["canonicalObjectSignature"])
        p["_localBagHash"] = sha256_text(json.dumps(p["localSignatureBag"], separators=(",", ":")))
        p["_metadata"] = metadata_vector(grid, p)
        p["_suffix"] = signature_suffix(p)
        _PROJ_CACHE[k] = p
    return p


def grid_id_cert(arm: str, grid: list[list[int]]) -> str:
    """Per spec §"Context Identity"."""
    if arm == "raw_grid_context":
        return _grid_key(grid)
    p = grid_proj(grid)
    sh = p["_signatureHash"]
    bh = p["_localBagHash"]
    if arm == "signature_only_context":
        return f"{sh}|{bh}"
    if arm == "metadata_only_context":
        return json.dumps(p["_metadata"], separators=(",", ":"))
    if arm == "signature_palette_context":
        shape = p["shape"]
        palette = "".join(str(x) for x in p["palette"])
        return f"{shape[0]}x{shape[1]}|{palette}|{p['nonZeroCells']}|{p['nonZeroComponents']}|{p['density']}|{sh}|{bh}"
    raise ValueError(f"unknown cert arm {arm!r}")


def _cosine_distance_suffix(a: dict[int, float], b: dict[int, float]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # signature_suffix vectors are already L2-normalized to unit norm.
    keys = set(a) | set(b)
    dot = sum(a.get(k, 0.0) * b.get(k, 0.0) for k in keys)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0.0 or nb == 0.0:
        return 1.0
    cos = dot / (na * nb)
    return max(0.0, min(1.0, 1.0 - cos))


def _metadata_l1(a: list[float], b: list[float]) -> float:
    n = max(1, len(a))
    return sum(abs(x - y) for x, y in zip(a, b)) / n


def _raw_hamming(ga: list[list[int]], gb: list[list[int]]) -> float:
    diff = 0
    for y in range(MAX_H):
        for x in range(MAX_W):
            va = ga[y][x] if (y < len(ga) and x < len(ga[0])) else 10
            vb = gb[y][x] if (y < len(gb) and x < len(gb[0])) else 10
            if va != vb:
                diff += 1
    return diff / (MAX_H * MAX_W)


def grid_dist_cert(arm: str, ga: list[list[int]], gb: list[list[int]]) -> float:
    """Per spec §"Context Distance" — individual grid distances."""
    if arm == "raw_grid_context":
        return _raw_hamming(ga, gb)
    pa, pb = grid_proj(ga), grid_proj(gb)
    if arm == "metadata_only_context":
        return _metadata_l1(pa["_metadata"], pb["_metadata"])
    sig_cos = _cosine_distance_suffix(pa["_suffix"], pb["_suffix"])
    if arm == "signature_only_context":
        return sig_cos
    if arm == "signature_palette_context":
        return 0.5 * sig_cos + 0.5 * _metadata_l1(pa["_metadata"], pb["_metadata"])
    raise ValueError(f"unknown cert arm {arm!r}")


def context_identity(arm: str, ctx: dict[str, Any]) -> str:
    qid = grid_id_cert(arm, ctx["query_input"])
    pair_ids = sorted(
        grid_id_cert(arm, p["input"]) + "=>" + grid_id_cert(arm, p["output"])
        for p in ctx["conditioning"]
    )
    return qid + "||" + "|".join(pair_ids)


def _pair_dist(arm: str, pa: dict[str, Any], pb: dict[str, Any]) -> float:
    return 0.5 * grid_dist_cert(arm, pa["input"], pb["input"]) + 0.5 * grid_dist_cert(arm, pa["output"], pb["output"])


def conditioning_distance(arm: str, da: list[dict[str, Any]], db: list[dict[str, Any]]) -> float:
    """Min-cost bipartite matching of conditioning pairs; unmatched -> 1.0;
    normalized by larger conditioning-pair count (spec §"Context Distance")."""
    m, n = len(da), len(db)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0
    cost = [[_pair_dist(arm, pa, pb) for pb in db] for pa in da]
    k = max(m, n)
    # Square the cost matrix with dummy rows/cols costing 1.0 (unmatched).
    sq = [[1.0] * k for _ in range(k)]
    for i in range(m):
        for j in range(n):
            sq[i][j] = cost[i][j]
    if k <= 7:
        best = None
        for perm in itertools.permutations(range(k)):
            total = sum(sq[i][perm[i]] for i in range(k))
            if best is None or total < best:
                best = total
        return best / k
    # Greedy fallback for unexpectedly large k (registered set has k<=5).
    used_cols: set[int] = set()
    total = 0.0
    for i in range(k):
        bestj, bestc = None, None
        for j in range(k):
            if j in used_cols:
                continue
            if bestc is None or sq[i][j] < bestc:
                bestc, bestj = sq[i][j], j
        used_cols.add(bestj)
        total += bestc
    return total / k


def context_distance(arm: str, ca: dict[str, Any], cb: dict[str, Any]) -> float:
    dq = grid_dist_cert(arm, ca["query_input"], cb["query_input"])
    dc = conditioning_distance(arm, ca["conditioning"], cb["conditioning"])
    return round_float(0.4 * dq + 0.6 * dc)


# ============================================================================
# Certificate: target labels + program_sketch_v1 oracle
# ============================================================================
def target_exact_hash(target: list[list[int]]) -> str:
    return sha256_text(json.dumps(target, separators=(",", ":")))


def target_signature_palette_id(target: list[list[int]]) -> str:
    return grid_id_cert("signature_palette_context", target)


# ============================================================================
# program_sketch_v2: framing-agnostic raw-grid relation labelers (spec §"Sketch Facets")
# These NEVER read signature_palette / arm distances / decoder outputs.
# ============================================================================
def _d4_variants(grid: list[list[int]]) -> dict[str, list[list[int]]]:
    out = {"identity": grid}
    if grid and grid[0]:
        out["rot90"] = rotate90(grid)
        out["rot180"] = rotate180(grid)
        out["rot270"] = rotate270(grid)
        out["reflect_h"] = reflect_horizontal(grid)
        out["reflect_v"] = reflect_vertical(grid)
        out["transpose"] = transpose(grid)
        out["anti_transpose"] = anti_transpose(grid)
    return out


def _nonzero_bbox_crop(grid: list[list[int]]) -> list[list[int]] | None:
    cells = nonzero_cells(grid)
    if not cells:
        return None
    minx = min(c["x"] for c in cells); maxx = max(c["x"] for c in cells)
    miny = min(c["y"] for c in cells); maxy = max(c["y"] for c in cells)
    return [[grid[y][x] for x in range(minx, maxx + 1)] for y in range(miny, maxy + 1)]


def _is_subgrid(small: list[list[int]], big: list[list[int]]) -> bool:
    sh, sw = shape_of(small)
    bh, bw = shape_of(big)
    if sh == 0 or sw == 0 or sh > bh or sw > bw:
        return False
    for oy in range(bh - sh + 1):
        for ox in range(bw - sw + 1):
            if all(big[oy + y][ox + x] == small[y][x] for y in range(sh) for x in range(sw)):
                return True
    return False


def _tile_relation(x: list[list[int]], y: list[list[int]]) -> bool:
    xh, xw = shape_of(x); yh, yw = shape_of(y)
    if xh == 0 or xw == 0 or yh == 0 or yw == 0:
        return False
    if yh % xh != 0 or yw % xw != 0:
        return False
    if (yh // xh) * (yw // xw) <= 1:
        return False
    variants = _d4_variants(x)
    for by in range(yh // xh):
        for bx in range(yw // xw):
            block = [[y[by * xh + r][bx * xw + c] for c in range(xw)] for r in range(xh)]
            if not any(block == v for v in variants.values() if shape_of(v) == (xh, xw)):
                return False
    return True


def _color_map_relation(x: list[list[int]], y: list[list[int]]) -> str | None:
    """For shape-equal pairs: classify the cellwise color relation."""
    if shape_of(x) != shape_of(y):
        return None
    mapping: dict[int, int] = {}
    consistent = True
    changed = False
    for r in range(len(x)):
        for c in range(len(x[0])):
            a, b = x[r][c], y[r][c]
            if a != b:
                changed = True
            if a in mapping and mapping[a] != b:
                consistent = False
            mapping[a] = b
    if not changed:
        return "palette_preserved"
    if consistent:
        # consistent global recolor map
        if set(mapping.keys()) == set(mapping.values()):
            return "palette_permuted"
        return "palette_role_recolored"
    return None


def _symmetries(grid: list[list[int]]) -> set[str]:
    s: set[str] = set()
    if not grid or not grid[0]:
        return s
    if grid == reflect_horizontal(grid):
        s.add("reflect_vertical")  # mirror across vertical axis (left-right)
    if grid == reflect_vertical(grid):
        s.add("reflect_horizontal")  # mirror across horizontal axis (top-bottom)
    if shape_of(grid) == shape_of(transpose(grid)) and grid == transpose(grid):
        s.add("reflect_diagonal")
    if grid and grid[0] and len(grid) == len(grid[0]) and grid == rotate180(grid):
        s.add("rotational_symmetry")
    elif grid == rotate180(grid):
        s.add("rotational_symmetry")
    return s


def _has_periodic(grid: list[list[int]]) -> bool:
    h, w = shape_of(grid)
    for p in (1, 2, 3):
        if w > p and all(grid[r][c] == grid[r][c % p] for r in range(h) for c in range(w)):
            return True
        if h > p and all(grid[r][c] == grid[r % p][c] for r in range(h) for c in range(w)):
            return True
    return False


def _count_metrics(grid: list[list[int]]) -> dict[str, int]:
    return {
        "objects": count_components(grid),
        "colors": len(palette_of(grid)),
        "cells": sum(1 for row in grid for v in row if v != 0),
        "components": count_components(grid),
    }


def _facet_shape(x, y, conditioning) -> list[str]:
    out: list[str] = []
    xh, xw = shape_of(x); yh, yw = shape_of(y)
    if (xh, xw) == (yh, yw):
        out.append("shape_preserved")
    if (yh, yw) == (xw, xh) and xh != xw:
        out.append("shape_transposed")
    if yh <= xh and yw <= xw and (yh, yw) != (xh, xw) and _is_subgrid(y, x):
        out.append("shape_cropped")
    if yh >= xh and yw >= xw and (yh, yw) != (xh, xw) and _is_subgrid(x, y):
        out.append("shape_padded")
    if _tile_relation(x, y):
        out.append("shape_scaled_or_tiled")
    crop = _nonzero_bbox_crop(x)
    if crop is not None and shape_of(crop) == (yh, yw) and crop != x:
        out.append("shape_extracted_from_object")
    return out if out else ["shape_other"]


def _facet_palette(x, y, conditioning) -> list[str]:
    out: list[str] = []
    px, py = set(palette_of(x)), set(palette_of(y))
    if px == py:
        out.append("palette_preserved")
    if py - px:
        out.append("palette_color_added")
    if px - py:
        out.append("palette_color_removed")
    cm = _color_map_relation(x, y)
    if cm == "palette_permuted":
        out.append("palette_permuted")
    elif cm == "palette_role_recolored":
        out.append("palette_role_recolored")
    # background change: modal color differs
    if x and y:
        mx = Counter(v for row in x for v in row).most_common(1)[0][0]
        my = Counter(v for row in y for v in row).most_common(1)[0][0]
        if mx != my:
            out.append("palette_background_changed")
    return out if out else ["palette_other"]


def _facet_object(x, y, conditioning) -> list[str]:
    out: list[str] = []
    cx, cy = count_components(x), count_components(y)
    if cx == cy and shape_of(x) == shape_of(y):
        out.append("object_identity_preserved")
    if cy < cx:
        out.append("object_removed")
    if cy > cx:
        out.append("object_created")
    crop = _nonzero_bbox_crop(x)
    if crop is not None and crop == y:
        out.append("object_selected")
    # marker-guided: a singleton-cell color present in input
    counts = Counter(v for row in x for v in row if v != 0)
    if any(n == 1 for n in counts.values()):
        out.append("object_marker_guided")
    return out if out else ["object_other"]


def _facet_cardinality(x, y, conditioning) -> list[str]:
    out: list[str] = []
    # cross-demo "controls" tests over D + query
    pairs = list(conditioning) + [{"input": x, "output": y}]
    metrics = [(_count_metrics(p["input"]), p["output"]) for p in pairs]
    if len({m["objects"] for m, _ in metrics}) > 1:
        # input object count varies across demos
        out.append("count_objects")
        out_shapes = {shape_of(o) for _, o in metrics}
        if len(out_shapes) > 1:
            out.append("cardinality_controls_shape")
        out_pal = {len(palette_of(o)) for _, o in metrics}
        if len(out_pal) > 1:
            out.append("cardinality_controls_palette")
    if len({m["colors"] for m, _ in metrics}) > 1:
        out.append("count_colors")
    if len({m["cells"] for m, _ in metrics}) > 1:
        out.append("count_cells")
    return out if out else ["none"]


def _facet_completion(x, y, conditioning) -> list[str]:
    out: list[str] = []
    if shape_of(x) == shape_of(y):
        sx, sy = _symmetries(x), _symmetries(y)
        if sy - sx:
            out.append("complete_symmetry")
        # fill: y has more nonzero cells than x in same positions kept
        nx = sum(1 for row in x for v in row if v != 0)
        ny = sum(1 for row in y for v in row if v != 0)
        if ny > nx:
            # x cells preserved + new cells added (a completion/fill)
            preserved = all(y[r][c] == x[r][c] for r in range(len(x)) for c in range(len(x[0])) if x[r][c] != 0)
            if preserved:
                out.append("fill_holes")
        if _has_periodic(y) and not _has_periodic(x):
            out.append("complete_repeating_pattern")
    return out if out else ["none"]


def _facet_spatial(x, y, conditioning) -> list[str]:
    out: list[str] = []
    variants = _d4_variants(x)
    if shape_of(x) == shape_of(y):
        for name, mapping in (("rot90", "rotate"), ("rot180", "rotate"), ("rot270", "rotate"),
                              ("reflect_h", "reflect"), ("reflect_v", "reflect"),
                              ("transpose", "reflect"), ("anti_transpose", "reflect")):
            if variants.get(name) == y:
                out.append(mapping)
    if _tile_relation(x, y):
        out.append("tile")
    yh, yw = shape_of(y); xh, xw = shape_of(x)
    if yh < xh or yw < xw:
        out.append("crop")
    if yh > xh or yw > xw:
        out.append("pad")
    return sorted(set(out)) if out else ["none"]


def _facet_symmetry(x, y, conditioning) -> list[str]:
    out: list[str] = []
    sy = _symmetries(y)
    for s in sy:
        out.append(s)
    if _has_periodic(y):
        out.append("periodic_symmetry")
    sx = _symmetries(x)
    if sy - sx:
        out.append("symmetry_completion")
    if sx - sy:
        out.append("symmetry_breaking")
    return sorted(set(out)) if out else ["none"]


def _facet_correspondence(x, y, conditioning) -> list[str]:
    out: list[str] = []
    if set(palette_of(x)) & set(palette_of(y)):
        out.append("by_color")
    if shape_of(x) == shape_of(y):
        out.append("by_position")
    crop = _nonzero_bbox_crop(x)
    if crop is not None and crop == y:
        out.append("by_shape")
    counts = Counter(v for row in x for v in row if v != 0)
    if any(n == 1 for n in counts.values()):
        out.append("by_marker")
    if count_components(x) > 1:
        out.append("by_adjacency")
    return sorted(set(out)) if out else ["correspondence_other"]


def _facet_rule_scope(x, y, conditioning, facets_so_far: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    if shape_of(x) == shape_of(y):
        out.append("local_pixel")
    if "cardinality_controls_shape" in facets_so_far.get("cardinality_relation", []) or "count_objects" in facets_so_far.get("cardinality_relation", []):
        out.append("global_count")
    if any(s not in NONINFORMATIVE for s in facets_so_far.get("symmetry_relation", [])):
        out.append("global_symmetry")
    if "shape_scaled_or_tiled" in facets_so_far.get("shape_relation", []) or "tile" in facets_so_far.get("spatial_transform_relation", []):
        out.append("global_pattern")
    if shape_of(x) != shape_of(y):
        out.append("shape_level")
    if count_components(x) > 1:
        out.append("multi_object_relation")
    return sorted(set(out)) if out else ["scope_other"]


def program_sketch_v2(ctx: dict[str, Any]) -> dict[str, Any]:
    """Framing-agnostic nine-facet transformation sketch over RAW grids.
    Order-invariant over conditioning; never reads signature_palette."""
    x = ctx["query_input"]
    y = ctx["target_output"]
    cond = sorted(ctx["conditioning"], key=lambda p: (json.dumps(p["input"], separators=(",", ":")), json.dumps(p["output"], separators=(",", ":"))))
    facets: dict[str, list[str]] = {}
    facets["shape_relation"] = _facet_shape(x, y, cond)
    facets["palette_relation"] = _facet_palette(x, y, cond)
    facets["object_relation"] = _facet_object(x, y, cond)
    facets["cardinality_relation"] = _facet_cardinality(x, y, cond)
    facets["completion_relation"] = _facet_completion(x, y, cond)
    facets["spatial_transform_relation"] = _facet_spatial(x, y, cond)
    facets["symmetry_relation"] = _facet_symmetry(x, y, cond)
    facets["correspondence_basis"] = _facet_correspondence(x, y, cond)
    facets["rule_scope"] = _facet_rule_scope(x, y, cond, facets)
    # coarse diagnostic bins (allowed; no exact counts)
    em = 0
    if shape_of(x) == shape_of(y):
        em = sum(1 for r in range(len(x)) for c in range(len(x[0])) if x[r][c] != y[r][c])
        em = em / max(1, len(x) * len(x[0]))
    diagnostics = {
        "edit_mass_bin": round(min(1.0, em) * 4) / 4 if shape_of(x) == shape_of(y) else "shape_change",
        "object_count_bin": min(count_components(y), 5),
        "shape_delta_bin": [max(-1, min(1, shape_of(y)[0] - shape_of(x)[0])), max(-1, min(1, shape_of(y)[1] - shape_of(x)[1]))],
    }
    return {"facets": facets, "diagnostics": diagnostics}


# ============================================================================
# program_sketch_v2: gates
# ============================================================================
def _concrete_labels(facet_values: list[str]) -> list[str]:
    return [v for v in facet_values if v not in NONINFORMATIVE]


def nonempty_facet_count(sketch: dict[str, Any]) -> int:
    return sum(1 for f in SKETCH_FACETS if _concrete_labels(sketch["facets"][f]))


def is_vacuous(sketch: dict[str, Any]) -> bool:
    return nonempty_facet_count(sketch) < VACUITY_MIN_NONEMPTY_FACETS


def laundering_extra_facet_count(sketch: dict[str, Any], prior: str) -> int:
    assoc = PRIOR_ASSOCIATED_FACET.get(prior)
    return sum(1 for f in SKETCH_FACETS if f != assoc and _concrete_labels(sketch["facets"][f]))


def core_sketch_tuple(sketch: dict[str, Any]) -> tuple:
    return tuple(tuple(sorted(sketch["facets"][f])) for f in SKETCH_FACETS)


def _syntactic_leakage(sketch: dict[str, Any]) -> list[str]:
    """No raw target hashes/serializations/coords/masks/unbounded ints in the sketch."""
    fails: list[str] = []
    blob = json.dumps(sketch, separators=(",", ":"))
    # All facet labels are from the frozen vocabulary; diagnostics are coarse bins.
    # Guard: no value may be a long digit run (would indicate a copied count/coord).
    import re as _re
    if _re.search(r"\d{3,}", blob):
        fails.append("unbounded_integer_in_sketch")
    # core facet labels must all be strings from known facet vocabularies (no grids)
    for f in SKETCH_FACETS:
        for v in sketch["facets"][f]:
            if not isinstance(v, str):
                fails.append(f"non_string_label_in_{f}")
    return fails


# ============================================================================
# program_sketch_v2: incompatibility rule (spec §"v2 Incompatibility Rule")
# ============================================================================
def _facet_disjoint(a: list[str], b: list[str]) -> bool:
    sa = set(_concrete_labels(a))
    sb = set(_concrete_labels(b))
    if not sa or not sb:
        return False
    return sa.isdisjoint(sb)


def _hard_incompat(sa: dict[str, Any], sb: dict[str, Any]) -> bool:
    fa, fb = sa["facets"], sb["facets"]
    scope_a, scope_b = set(fa["rule_scope"]), set(fb["rule_scope"])
    shared_nonscope = any(
        set(_concrete_labels(fa[f])) & set(_concrete_labels(fb[f]))
        for f in SKETCH_FACETS if f != "rule_scope"
    )
    # global_count vs global_symmetry with no shared non-scope facet
    if (("global_count" in scope_a and "global_symmetry" in scope_b) or
        ("global_symmetry" in scope_a and "global_count" in scope_b)) and not shared_nonscope:
        return True
    # shape_constructed_from_count vs shape_extracted_from_object w/ disjoint obj+card facets
    def _has(s, facet, lab): return lab in s["facets"][facet]
    obj_card_disjoint = _facet_disjoint(fa["object_relation"], fb["object_relation"]) and _facet_disjoint(fa["cardinality_relation"], fb["cardinality_relation"])
    if ((_has(sa, "shape_relation", "shape_constructed_from_count") and _has(sb, "shape_relation", "shape_extracted_from_object")) or
        (_has(sb, "shape_relation", "shape_constructed_from_count") and _has(sa, "shape_relation", "shape_extracted_from_object"))) and obj_card_disjoint:
        return True
    # one has symmetry_completion, other has no symmetry facet beyond none/unknown
    def _sym_only_noninf(s): return not _concrete_labels(s["facets"]["symmetry_relation"])
    if (_has(sa, "symmetry_relation", "symmetry_completion") and _sym_only_noninf(sb)) or (_has(sb, "symmetry_relation", "symmetry_completion") and _sym_only_noninf(sa)):
        return True
    # one has complete_global_template, other only local/object-level scope
    local_scopes = {"local_pixel", "local_patch", "object_level"}
    def _only_local(s): return set(_concrete_labels(s["facets"]["rule_scope"])) and set(_concrete_labels(s["facets"]["rule_scope"])).issubset(local_scopes)
    if (_has(sa, "completion_relation", "complete_global_template") and _only_local(sb)) or (_has(sb, "completion_relation", "complete_global_template") and _only_local(sa)):
        return True
    return False


def v2_incompatible(sa: dict[str, Any], sb: dict[str, Any]) -> tuple[bool, int]:
    if is_vacuous(sa) or is_vacuous(sb):
        return False, 0
    n_disjoint = sum(1 for f in SKETCH_FACETS if _facet_disjoint(sa["facets"][f], sb["facets"][f]))
    incompat = (n_disjoint >= V2_MIN_DISJOINT_FACETS) or _hard_incompat(sa, sb)
    return incompat, n_disjoint


def _v2_facet_jaccard(sa: dict[str, Any], sb: dict[str, Any]) -> float:
    a, b = set(), set()
    for f in SKETCH_FACETS:
        a |= {f"{f}:{x}" for x in _concrete_labels(sa["facets"][f])}
        b |= {f"{f}:{x}" for x in _concrete_labels(sb["facets"][f])}
    if not a and not b:
        return 0.0
    return 1.0 - (len(a & b) / len(a | b) if (a | b) else 0.0)




