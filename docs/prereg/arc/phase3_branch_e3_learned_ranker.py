#!/usr/bin/env python3
"""Branch E3 -- learned ranker over the frozen Branch E v2 candidate generator.

Spec: PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md. The candidate generator is the
frozen Branch E v2 generator, reused byte-for-byte (this module replicates only
the v2 enumeration DRIVER, calling v2's frozen family functions, and returns the
FULL admitted candidate list instead of the top-2). E3 replaces only the SELECTOR
with a learned MLP ranker trained on non-gated public-training candidate examples.

Leak discipline: target-FREE features; a no-target candidate fingerprint barrier
per split (aux/validation/U_primary); U_primary targets are read only for final
scoring after its barrier; no signature_palette geometry; no task-id feature; no
public-evaluation data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import phase3_branch_e_v2_program_search as v2  # noqa: E402  (frozen candidate generator)

v1 = v2.v1

# Shared helpers (inherited from the v1/v2 frozen runners).
load_tasks = v1.load_tasks
build_lodo_instances = v1.build_lodo_instances
build_pttest_instances = v1.build_pttest_instances
write_json = v1.write_json
write_csv = v1.write_csv
write_jsonl = v1.write_jsonl
sha256_text = v1.sha256_text
sha256_file = v1.sha256_file
round_float = v1.round_float
iso_now = v1.iso_now
git_state = v1.git_state
assert_training_data_dir = v1.assert_training_data_dir
hash_receipt_files = v1.hash_receipt_files

# ============================================================================
# Frozen constants (spec)
# ============================================================================
FEATURE_SCHEMA_VERSION = "arc-p3-branch-e3-ranker-feature-v1"
PROTOCOL_VERSION = "arc-p3-branch-e3-v1"
RECEIPT_SCHEMA_VERSION = "arc-p3-branch-e3-receipt-v1"
LEARNER_VERSION = "branch_e3_learned_ranker"

SEED_SLATE = [20260529, 20260530, 20260531, 20260601, 20260602]
PROGRAM_HASH_DIM = 512
S_V1V2 = {"be94b721", "f25fbde4"}
ATTEMPTS = 2

# Frozen ranker MLP hyperparameters (spec §"Ranker Model").
RANKER = {
    "h1": 192, "h2": 96, "dropout": 0.05, "lr": 1.0e-3, "betas": (0.9, 0.99),
    "eps": 1.0e-8, "weight_decay": 1.0e-4, "batch_size": 2048, "max_epochs": 40,
    "early_stop_patience": 6, "grad_clip_norm": 1.0, "pos_weight_min": 1.0,
    "pos_weight_max": 100.0,
}

# Frozen branch gates (spec §"Branches").
MATERIAL_LIFT_MIN_TASKS = 4
MATERIAL_LIFT_MIN_NEW = 2
REPLICATED_MIN_TASKS = 2

PRIORS = ["objectness", "counting", "symmetry", "spatial_transform", "local_completion", "color_role"]

# Fixed family vocabularies for one-hot syntax features (deterministic, frozen).
STRUCT_FAMILIES = ["identity", "d4", "palette_permute", "translate", "crop_bbox",
                   "fill_enclosed", "extract_largest_component", "tile", "scale", "pad",
                   "output_copy", "delta_overlay", "coloredit"]
COLOR_FAMILIES = ["constant_edit_color", "modal_edit_color", "baseline_color_map",
                  "input_nn_color_map", "input_patch_majority_map", "baseline_to_input_pair_map",
                  "relative_palette_rank_map", "object_role_color_map", "row_col_periodic_color",
                  "nearest_edited_neighbor_color"]
MASK_FAMILIES = list(v2.V2_MASK_FAMILIES)
MORPH_OPS = list(v2.V2_MASK_MORPH_OPS)


# ============================================================================
# Candidate enumeration -- replicates the v2 driver, returns the FULL admitted set
# (byte-for-byte: calls v2's frozen family functions + consistency + composition).
# ============================================================================
def enumerate_admitted(inst: Any) -> tuple[list[tuple[int, int, str, list[list[int]]]], bool]:
    pairs = inst.conditioning
    query = inst.query_input
    admitted: list[tuple[int, int, str, list[list[int]]]] = []
    budget = v2.CANDIDATE_BUDGET

    def _try(name: str, priority: int, complexity: int, fn: Any) -> None:
        nonlocal budget
        if budget <= 0:
            return
        budget -= 1
        if v2._consistent(fn, pairs):
            cand = v1._safe(lambda: fn(query))
            if cand is not None and len(cand) > 0 and len(cand[0]) > 0:
                admitted.append((priority, complexity, name, cand))

    base_progs = v1.structural_programs(pairs) + v1.combinator_programs(pairs) + v2.color_edit_programs_v2(pairs)
    for name, priority, complexity, fn in base_progs:
        _try(name, priority, complexity, fn)

    stage1 = v1.structural_programs(pairs)
    for n1, p1, c1, f1 in stage1:
        if n1 == "identity" or budget <= 0:
            continue
        t1_pairs = v2._transform_pairs(pairs, f1)
        if t1_pairs is None:
            continue
        for n2, p2, c2, f2 in v1.structural_programs(t1_pairs) + v2.color_edit_programs_v2(t1_pairs):
            if n2 == "identity" or budget <= 0:
                continue
            _try(f"{n1}>>{n2}", max(p1, p2) + 15, c1 + c2 + 1, v2._compose(f1, f2))
        for n2, p2, c2, f2 in v1.structural_programs(t1_pairs):
            if n2 == "identity" or budget <= 0:
                continue
            t2_pairs = v2._transform_pairs(t1_pairs, f2)
            if t2_pairs is None:
                continue
            for n3, p3, c3, f3 in v1.structural_programs(t2_pairs) + v2.color_edit_programs_v2(t2_pairs):
                if n3 == "identity" or budget <= 0:
                    continue
                _try(f"{n1}>>{n2}>>{n3}", max(p1, p2, p3) + 30, c1 + c2 + c3 + 2, v2._compose(f1, f2, f3))

    budget_exhausted = budget <= 0
    admitted.sort(key=lambda t: (t[0], t[1], t[2]))  # v2 deterministic order
    return admitted, budget_exhausted


# ============================================================================
# Target-free feature extraction (spec §"Ranker Feature Schema")
# ============================================================================
def _dims(g: list[list[int]]) -> tuple[int, int]:
    return (len(g), len(g[0]) if g else 0)


def _palette(g: list[list[int]]) -> set:
    return {v for row in g for v in row}


def _nonzero(g: list[list[int]]) -> int:
    return sum(1 for row in g for v in row if v != 0)


def _components(g: list[list[int]]) -> int:
    h, w = _dims(g)
    seen = [[False] * w for _ in range(h)]
    n = 0
    for y in range(h):
        for x in range(w):
            if g[y][x] != 0 and not seen[y][x]:
                n += 1
                stack = [(y, x)]
                while stack:
                    cy, cx = stack.pop()
                    if 0 <= cy < h and 0 <= cx < w and not seen[cy][cx] and g[cy][cx] != 0:
                        seen[cy][cx] = True
                        stack.extend([(cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)])
    return n


def _sym_flags(g: list[list[int]]) -> list[float]:
    h, w = _dims(g)
    if h == 0 or w == 0:
        return [0.0, 0.0, 0.0, 0.0]
    hf = 1.0 if all(g[y] == g[y][::-1] for y in range(h)) else 0.0
    vf = 1.0 if g == g[::-1] else 0.0
    r180 = 1.0 if g == [row[::-1] for row in g[::-1]] else 0.0
    tf = 1.0 if (h == w and all(g[y][x] == g[x][y] for y in range(h) for x in range(w))) else 0.0
    return [hf, vf, r180, tf]


def _periodic_flags(g: list[list[int]]) -> list[float]:
    h, w = _dims(g)
    rowp = 1.0 if (h >= 2 and any(all(g[y] == g[y % p] for y in range(h)) for p in range(1, h))) else 0.0
    colp = 0.0
    if w >= 2:
        for p in range(1, w):
            if all(g[y][x] == g[y][x % p] for y in range(h) for x in range(w)):
                colp = 1.0
                break
    return [rowp, colp]


def _onehot(value: str, vocab: list[str]) -> list[float]:
    out = [0.0] * len(vocab)
    if value in vocab:
        out[vocab.index(value)] = 1.0
    return out


def _terminal_family(name: str) -> str:
    last = name.split(">>")[-1]
    if last.startswith("d4:"):
        return "d4"
    if last.startswith("coloredit:"):
        return "coloredit"
    return last


def _color_family(name: str) -> str:
    for part in name.split(">>"):
        if part.startswith("coloredit:"):
            toks = part.split(":")
            if len(toks) >= 2:
                return toks[1]
    return ""


def _mask_family(name: str) -> str:
    for part in name.split(">>"):
        if part.startswith("coloredit:"):
            toks = part.split(":")
            if len(toks) >= 4:
                return toks[3].split("|")[0]
    return ""


def _morph_op(name: str) -> str:
    for op in MORPH_OPS:
        if f"|{op}" in name:
            return op
    return "identity" if "coloredit:" in name else ""


def _feature_hash(name: str) -> list[float]:
    vec = [0.0] * PROGRAM_HASH_DIM
    parts = name.split(">>")
    toks: list[str] = []
    for part in parts:
        toks.append(f"op:{part.split(':')[0]}")
        for sub in part.split(":"):
            toks.append(f"tok:{sub}")
    for a, b in zip(parts, parts[1:]):
        toks.append(f"bg:{a.split(':')[0]}>{b.split(':')[0]}")
    for t in toks:
        hh = int.from_bytes(hashlib.sha256(t.encode("utf-8")).digest()[:8], "big")
        idx = hh % PROGRAM_HASH_DIM
        sign = 1.0 if (hh >> 63) & 1 else -1.0
        vec[idx] += sign
    return vec


def _conditioning_summary(inst: Any) -> list[float]:
    pairs = inst.conditioning
    n = len(pairs)
    hr, wr, pal_change, edit_density, same_shape = [], [], [], [], 0
    for p in pairs:
        ih, iw = _dims(p["input"]); oh, ow = _dims(p["output"])
        hr.append((oh / ih) if ih else 0.0)
        wr.append((ow / iw) if iw else 0.0)
        pal_change.append(len(_palette(p["output"]) ^ _palette(p["input"])) / 10.0)
        if (ih, iw) == (oh, ow):
            same_shape += 1
            diff = sum(1 for y in range(ih) for x in range(iw) if p["input"][y][x] != p["output"][y][x])
            edit_density.append(diff / max(1, ih * iw))
    def _m(xs):
        return (sum(xs) / len(xs)) if xs else 0.0
    return [min(n, 9) / 9.0, _m(hr), _m(wr), _m(pal_change), _m(edit_density), (same_shape / n) if n else 0.0]


def candidate_features(inst: Any, name: str, priority: int, complexity: int, grid: list[list[int]], v2_rank: int, n_admitted: int) -> list[float]:
    q = inst.query_input
    gh, gw = _dims(grid)
    qh, qw = _dims(q)
    depth = v2._program_depth(name)
    op_count = name.count(">>") + 1
    struct_prefix = sum(1 for part in name.split(">>") if not part.startswith("coloredit:"))
    combinator = 1.0 if (name in ("output_copy", "delta_overlay")) else 0.0
    syntax = [
        depth / 3.0, op_count / 3.0, struct_prefix / 3.0, complexity / 20.0,
        (v2_rank / n_admitted) if n_admitted else 0.0, priority / 60.0, combinator,
    ]
    syntax += _onehot(_terminal_family(name), STRUCT_FAMILIES)
    syntax += _onehot(_mask_family(name), MASK_FAMILIES)
    syntax += _onehot(_morph_op(name), MORPH_OPS)
    syntax += _onehot(_color_family(name), COLOR_FAMILIES)

    gpal = _palette(grid)
    gnz = _nonzero(grid)
    grid_feats = [
        gh / 30.0, gw / 30.0, (gh * gw) / 900.0, len(gpal) / 10.0,
        gnz / max(1, gh * gw), _components(grid) / 20.0,
    ]
    grid_feats += _sym_flags(grid) + _periodic_flags(grid)

    qpal = _palette(q)
    union = gpal | qpal
    rel = [
        (gh / qh) if qh else 0.0, (gw / qw) if qw else 0.0,
        1.0 if (gh, gw) == (qh, qw) else 0.0,
        (len(gpal & qpal) / len(union)) if union else 0.0,
        (len(gpal) - len(qpal)) / 10.0,
    ]
    if (gh, gw) == (qh, qw) and gh > 0:
        ham = sum(1 for y in range(gh) for x in range(gw) if grid[y][x] != q[y][x]) / (gh * gw)
        nz_overlap = sum(1 for y in range(gh) for x in range(gw) if grid[y][x] != 0 and q[y][x] != 0) / max(1, gh * gw)
    else:
        ham, nz_overlap = 1.0, 0.0
    rel += [ham, nz_overlap]

    cond = _conditioning_summary(inst)
    cond += _onehot(inst.primary_prior, PRIORS)

    return [round_float(x) for x in (syntax + _feature_hash(name) + grid_feats + rel + cond)]


# Compute INPUT_DIM once from a probe vector.
def _input_dim() -> int:
    dummy = type("D", (), {"query_input": [[0]], "conditioning": [{"input": [[0]], "output": [[0]]}], "primary_prior": "objectness"})()
    return len(candidate_features(dummy, "identity", 0, 1, [[0]], 0, 1))


INPUT_DIM = _input_dim()


# ============================================================================
# Ranker MLP (input_dim -> 192 -> 96 -> 1, weighted BCE)
# ============================================================================
class RankerMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.proj1 = nn.Linear(input_dim, RANKER["h1"])
        self.norm = nn.LayerNorm(RANKER["h1"])
        self.proj2 = nn.Linear(RANKER["h1"], RANKER["h2"])
        self.drop = nn.Dropout(RANKER["dropout"])
        self.head = nn.Linear(RANKER["h2"], 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.proj1(x))
        h = self.norm(h)
        h = F.gelu(self.proj2(h))
        h = self.drop(h)
        return self.head(h).squeeze(-1)


def set_determinism(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def fit_ranker(X: list[list[float]], y: list[float], seed: int, shuffle_labels: bool, device: torch.device) -> tuple[RankerMLP, dict[str, Any]]:
    set_determinism(seed)
    model = RankerMLP(INPUT_DIM).to(device)
    if not X:
        return model, {"epochs": 0, "rows": 0, "pos": 0, "pos_weight": 1.0, "history": []}
    labels = list(y)
    if shuffle_labels:
        rng = random.Random(seed)
        rng.shuffle(labels)
    pos = int(sum(labels))
    neg = len(labels) - pos
    pos_weight = max(RANKER["pos_weight_min"], min(RANKER["pos_weight_max"], neg / max(1, pos)))
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    Yt = torch.tensor(labels, dtype=torch.float32, device=device)
    W = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    optim = torch.optim.AdamW(model.parameters(), lr=RANKER["lr"], betas=RANKER["betas"], eps=RANKER["eps"], weight_decay=RANKER["weight_decay"])
    n = Xt.size(0)
    bs = RANKER["batch_size"]
    gen = torch.Generator(device="cpu"); gen.manual_seed(seed)
    best = float("inf"); patience = 0; history = []
    for epoch in range(RANKER["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=gen)
        ep_loss = 0.0; nb = 0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = Xt[idx], Yt[idx]
            optim.zero_grad()
            loss = F.binary_cross_entropy_with_logits(model(xb), yb, pos_weight=W)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), RANKER["grad_clip_norm"])
            optim.step()
            ep_loss += float(loss.detach().cpu().item()); nb += 1
        avg = ep_loss / max(1, nb)
        history.append({"epoch": epoch, "loss": round_float(avg)})
        if avg < best - 1e-6:
            best = avg; patience = 0
        else:
            patience += 1
        if patience >= RANKER["early_stop_patience"]:
            break
    return model, {"epochs": len(history), "rows": n, "pos": pos, "pos_weight": round_float(pos_weight), "best_loss": round_float(best), "history": history}


def score_candidates(model: RankerMLP, feats: list[list[float]], device: torch.device) -> list[float]:
    if not feats:
        return []
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(feats, dtype=torch.float32, device=device)).cpu().tolist()


# ============================================================================
# Per-instance candidate record (target-free fingerprint + label after barrier)
# ============================================================================
def build_instance_candidates(inst: Any) -> dict[str, Any]:
    admitted, budget_exhausted = enumerate_admitted(inst)
    n = len(admitted)
    cands = []
    for rank, (priority, complexity, name, grid) in enumerate(admitted):
        feats = candidate_features(inst, name, priority, complexity, grid, rank, n)
        cands.append({
            "program_id": name, "v2_rank": rank, "priority": priority, "complexity": complexity,
            "grid": grid, "grid_hash": sha256_text(json.dumps(grid, separators=(",", ":"))),
            "feat": feats, "feat_hash": sha256_text(json.dumps(feats, separators=(",", ":"))),
        })
    return {"instance_id": inst.instance_id, "lane": inst.lane, "task_id": inst.task_id,
            "primary_prior": inst.primary_prior, "query_index": inst.query_index,
            "n_conditioning": len(inst.conditioning), "n_candidates": n,
            "budget_exhausted": budget_exhausted, "candidates": cands}


def fingerprint_row(rec: dict[str, Any]) -> dict[str, Any]:
    """No-target fingerprint: identity + per-candidate hashes + v2 rank + count. NO target."""
    return {
        "instance_id": rec["instance_id"], "lane": rec["lane"], "task_id": rec["task_id"],
        "query_index": rec["query_index"], "n_conditioning": rec["n_conditioning"],
        "n_candidates": rec["n_candidates"], "budget_exhausted": rec["budget_exhausted"],
        "candidates": [{"program_id": c["program_id"], "v2_rank": c["v2_rank"],
                        "grid_hash": c["grid_hash"], "feat_hash": c["feat_hash"]} for c in rec["candidates"]],
    }


# ============================================================================
# Split construction
# ============================================================================
def all_instances(tasks: list[Any]) -> dict[str, list[Any]]:
    validation = [t for t in tasks if t.split == "validation"]
    test = [t for t in tasks if t.split == "test"]
    return {
        "validation_lodo": build_lodo_instances(validation, "validation_lodo"),
        "validation_pttest": build_pttest_instances(validation, "validation_pttest"),
        "test_lodo": build_lodo_instances(test, "test_lodo"),
        "pttest": build_pttest_instances(test, "pttest"),
    }


def load_aux_task_ids(data_dir: Path, register_path: Path) -> list[str]:
    """Public-training task ids minus the registered ids (sorted, deterministic)."""
    reg_text = register_path.read_text(encoding="utf-8-sig")
    import csv as _csv
    reg_ids = {r["task_id"] for r in _csv.DictReader(reg_text.splitlines())}
    training_dir = data_dir / "training"
    ids = sorted(p.stem for p in training_dir.glob("*.json") if p.stem not in reg_ids)
    return ids


def aux_instances_for_ids(data_dir: Path, task_ids: list[str]) -> list[Any]:
    insts: list[Any] = []
    for tid in task_ids:
        raw = (data_dir / "training" / f"{tid}.json").read_text(encoding="utf-8-sig")
        parsed = json.loads(raw)
        task = v1.Task(task_id=tid, primary_prior="aux", predicted_boundary="aux",
                       train=[{"index": i, "input": p["input"], "output": p["output"]} for i, p in enumerate(parsed["train"])],
                       test=[{"index": i, "input": p["input"], "output": p.get("output")} for i, p in enumerate(parsed["test"])],
                       split="aux")
        insts += build_lodo_instances([task], "aux_lodo")
        insts += build_pttest_instances([task], "aux_pttest")
    return insts


def write_barrier(out_dir: Path, name: str, fingerprints: list[dict[str, Any]]) -> str:
    path = out_dir / f"candidate_fingerprints_no_targets_{name}.jsonl"
    write_jsonl(path, fingerprints)
    h = sha256_file(path)
    (out_dir / f"candidate_fingerprints_no_targets_{name}.sha256").write_text(h + "\n", encoding="utf-8")
    return h


# ============================================================================
# Selectors (top-2 distinct grids per a scoring rule)
# ============================================================================
def top2_by(records: list[dict[str, Any]], key) -> list[dict[str, Any]]:
    ordered = sorted(records, key=key)
    out, seen = [], set()
    for c in ordered:
        if c["grid_hash"] in seen:
            continue
        seen.add(c["grid_hash"])
        out.append(c)
        if len(out) >= ATTEMPTS:
            break
    return out


def v2_selector(cands: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return top2_by(cands, key=lambda c: (c["priority"], c["complexity"], c["program_id"]))


def learned_selector(cands: list[dict[str, Any]], scores: dict[str, float]) -> list[dict[str, Any]]:
    return top2_by(cands, key=lambda c: (-scores.get(id(c), 0.0), c["v2_rank"], c["complexity"], c["program_id"]))


def main() -> int:
    return run(parse_args())


# ============================================================================
# CLI
# ============================================================================
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC Branch E3 learned ranker over frozen v2 candidates")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--register", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--split-mode", choices=["frozen_v2", "sha256_expansion"], default="sha256_expansion")
    p.add_argument("--shard-aux", action="store_true", help="generate aux candidate shard")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument("--merge", action="store_true", help="merge aux shards + train + score + adjudicate")
    p.add_argument("--limit-aux", type=int, default=0, help="smoke: cap aux task count")
    p.add_argument("--limit-tasks", type=int, default=0, help="smoke: cap register task count")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    if not args.dry_run and (not args.data_dir or not args.register):
        p.error("--data-dir and --register are required (except --dry-run)")
    return args


RL_FILES = {
    "split.csv": ["task_id", "primary_prior", "predicted_boundary", "split"],
    "aux_task_list.csv": ["shard_index", "shard_count", "task_id"],
    "training_summary.csv": ["arm", "rows", "pos", "pos_weight", "epochs", "best_loss"],
    "learning_curves.csv": ["arm", "seed", "epoch", "loss"],
    "seed_selection.csv": ["seed", "val_distinct_tasks", "val_exact_instance_rate", "val_bce", "selected"],
    "solutions_by_instance.csv": ["lane", "instance_id", "task_id", "primary_prior", "n_candidates", "v2_exact_any", "learned_exact_any", "metadata_exact_any", "oracle_ceiling", "winning_family"],
    "capability_summary.csv": ["selector", "lane", "n_instances", "n_tasks", "n_tasks_solved", "exact_instance_rate"],
    "selector_comparison.csv": ["lane", "v2_tasks", "learned_tasks", "metadata_tasks", "oracle_ceiling_tasks", "learned_new_vs_v1v2"],
    "candidate_ceiling.csv": ["lane", "n_instances", "ceiling_solvable_instances", "ceiling_solvable_tasks"],
    "per_prior_capability.csv": ["selector", "lane", "primary_prior", "n_tasks", "n_tasks_solved"],
    "quarantine_by_instance.csv": ["lane", "instance_id", "task_id", "quarantine_label"],
    "family_usage.csv": ["selector", "winning_family", "n_solved_instances"],
}


def write_empty_receipt(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(out_dir / "manifest.json", manifest)
    for fname, cols in RL_FILES.items():
        write_csv(out_dir / fname, [], cols)
    for nm in ("aux", "validation", "u_primary"):
        write_jsonl(out_dir / f"candidate_fingerprints_no_targets_{nm}.jsonl", [])
        (out_dir / f"candidate_fingerprints_no_targets_{nm}.sha256").write_text("", encoding="utf-8")
    write_json(out_dir / "ranker_feature_schema.json", {"schema": FEATURE_SCHEMA_VERSION, "input_dim": INPUT_DIM, "program_hash_dim": PROGRAM_HASH_DIM})
    write_jsonl(out_dir / "scores_by_instance.jsonl", [])
    write_json(out_dir / "phase3_branch_e3_learned_ranker_receipt.json", {"manifest": manifest, "branch": None})
    (out_dir / "branch_adjudication.md").write_text("# Branch E3 learned ranker\n\nDry run / empty receipt.\n", encoding="utf-8")
    (out_dir / "commands.md").write_text("# Branch E3 commands\n\nDry run / empty receipt.\n", encoding="utf-8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def base_manifest(args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    git = git_state(repo_root, args.allow_dirty)
    spec_path = Path(__file__).resolve().parent / "PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md"
    return {
        "generatedAt": iso_now(), "tool": "docs/prereg/arc/phase3_branch_e3_learned_ranker.py",
        "command": [sys.executable, "docs/prereg/arc/phase3_branch_e3_learned_ranker.py", *sys.argv[1:]],
        "gitCommit": git["commit"], "gitDirty": git["dirty"], "allowDirty": args.allow_dirty,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION, "protocolVersion": PROTOCOL_VERSION,
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION, "learnerVersion": LEARNER_VERSION,
        "specPath": "docs/prereg/arc/PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md",
        "specHash": (sha256_file(spec_path) if spec_path.exists() else "NA"),
        "runnerSha256": sha256_file(Path(__file__).resolve()),
        "v2GeneratorSha256": sha256_file(Path(__file__).resolve().parent / "phase3_branch_e_v2_program_search.py"),
        "v1GeneratorSha256": sha256_file(Path(__file__).resolve().parent / "phase3_branch_e_program_search.py"),
        "pythonVersion": sys.version, "platform": platform.platform(),
        "inputDim": INPUT_DIM, "programHashDim": PROGRAM_HASH_DIM, "seedSlate": SEED_SLATE,
        "ranker": {k: (list(v) if isinstance(v, tuple) else v) for k, v in RANKER.items()},
        "sV1V2": sorted(S_V1V2),
    }


def shard_path(out_dir: Path, idx: int) -> Path:
    return out_dir / f"aux_candidates_shard_{idx}.jsonl"


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        manifest = base_manifest(args, out_dir); manifest["mode"] = "dry_run"; manifest["completedAt"] = iso_now()
        write_empty_receipt(out_dir, manifest)
        print(f"ARC Branch E3 dry run wrote {out_dir}")
        return 0

    data_dir = Path(args.data_dir).resolve()
    register_path = Path(args.register).resolve()
    assert_training_data_dir(data_dir)
    device = torch.device(args.device)

    if args.shard_aux:
        return run_shard_aux(args, out_dir, data_dir, register_path)
    if args.merge:
        return run_merge(args, out_dir, data_dir, register_path, device)
    print("Specify --shard-aux, --merge, or --dry-run.")
    return 2


def run_shard_aux(args: argparse.Namespace, out_dir: Path, data_dir: Path, register_path: Path) -> int:
    aux_ids = load_aux_task_ids(data_dir, register_path)
    if args.limit_aux > 0:
        aux_ids = aux_ids[: args.limit_aux]
    chunk = [tid for i, tid in enumerate(aux_ids) if i % args.shard_count == args.shard_index]
    write_csv(out_dir / f"aux_task_list_shard_{args.shard_index}.csv",
              [{"shard_index": args.shard_index, "shard_count": args.shard_count, "task_id": t} for t in chunk],
              RL_FILES["aux_task_list.csv"])
    insts = aux_instances_for_ids(data_dir, chunk)
    rows = []
    for inst in insts:
        rec = build_instance_candidates(inst)
        # label AFTER fingerprint (barrier discipline is enforced at merge for the combined file;
        # per-shard we still record the fingerprint hash separately from labels)
        target = inst.target_output
        for c in rec["candidates"]:
            c["label"] = 1 if c["grid"] == target else 0
        rows.append(rec)
    write_jsonl(shard_path(out_dir, args.shard_index), rows)
    fp_rows = [fingerprint_row(r) for r in rows]
    fp_path = out_dir / f"aux_fingerprints_shard_{args.shard_index}.jsonl"
    write_jsonl(fp_path, fp_rows)
    print(f"ARC Branch E3 aux shard {args.shard_index}/{args.shard_count} wrote {len(rows)} instances "
          f"({sum(r['n_candidates'] for r in rows)} candidates) to {shard_path(out_dir, args.shard_index)}")
    return 0


def _solved_tasks(rows: list[dict[str, Any]], lane: str, key: str) -> set:
    return {r["task_id"] for r in rows if r["lane"] == lane and r[key]}


def run_merge(args: argparse.Namespace, out_dir: Path, data_dir: Path, register_path: Path, device: torch.device) -> int:
    manifest = base_manifest(args, out_dir)
    manifest["splitMode"] = args.split_mode

    # ---- merge aux shards (byte-equivalent to serial: concatenate by shard index) ----
    shard_files = sorted(out_dir.glob("aux_candidates_shard_*.jsonl"), key=lambda p: int(p.stem.split("_")[-1]))
    aux_records: list[dict[str, Any]] = []
    for sf in shard_files:
        for line in sf.read_text(encoding="utf-8").splitlines():
            if line.strip():
                aux_records.append(json.loads(line))
    fp_files = sorted(out_dir.glob("aux_fingerprints_shard_*.jsonl"), key=lambda p: int(p.stem.split("_")[-1]))
    aux_fp: list[dict[str, Any]] = []
    for ff in fp_files:
        for line in ff.read_text(encoding="utf-8").splitlines():
            if line.strip():
                aux_fp.append(json.loads(line))
    # Canonical sort makes the merge byte-equivalent regardless of shard layout/count.
    aux_records.sort(key=lambda r: (r["task_id"], r["instance_id"]))
    aux_fp.sort(key=lambda r: (r["task_id"], r["instance_id"]))
    aux_barrier = write_barrier(out_dir, "aux", aux_fp)
    manifest["nAuxShards"] = len(shard_files)
    manifest["nAuxInstances"] = len(aux_records)

    # ---- build training matrix from aux candidates ----
    train_X: list[list[float]] = []
    train_y: list[float] = []
    for rec in aux_records:
        for c in rec["candidates"]:
            train_X.append(c["feat"]); train_y.append(float(c["label"]))
    # metadata-only training matrix: zero out the syntax-onehot + program-hash dims, keep grid + cond + rel.
    # (We zero the first (syntax+hash) block; INPUT_DIM layout is syntax|hash|grid|rel|cond.)
    syntax_len = 7 + len(STRUCT_FAMILIES) + len(MASK_FAMILIES) + len(MORPH_OPS) + len(COLOR_FAMILIES)
    meta_mask_lo = 0
    meta_mask_hi = syntax_len + PROGRAM_HASH_DIM

    def _meta(vec: list[float]) -> list[float]:
        out = list(vec)
        for i in range(meta_mask_lo, meta_mask_hi):
            out[i] = 0.0
        return out

    # ---- register splits + candidate gen for validation + u_primary ----
    tasks, register_hash, data_hash = load_tasks(data_dir, register_path, args.split_mode)
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    manifest["registerHash"] = register_hash; manifest["dataDirHash"] = data_hash
    lanes = all_instances(tasks)
    write_csv(out_dir / "split.csv", [{"task_id": t.task_id, "primary_prior": t.primary_prior, "predicted_boundary": t.predicted_boundary, "split": t.split} for t in sorted(tasks, key=lambda x: x.task_id)], RL_FILES["split.csv"])

    val_lanes = ["validation_lodo", "validation_pttest"]
    prim_lanes = ["test_lodo", "pttest"]
    val_recs = [build_instance_candidates(i) for ln in val_lanes for i in lanes[ln]]
    val_barrier = write_barrier(out_dir, "validation", [fingerprint_row(r) for r in val_recs])
    prim_recs = [build_instance_candidates(i) for ln in prim_lanes for i in lanes[ln]]
    prim_barrier = write_barrier(out_dir, "u_primary", [fingerprint_row(r) for r in prim_recs])
    manifest["barrierHashes"] = {"aux": aux_barrier, "validation": val_barrier, "u_primary": prim_barrier}

    # label validation + u_primary (after barriers) using lane instance targets
    target_by_inst = {}
    for ln in val_lanes + prim_lanes:
        for i in lanes[ln]:
            target_by_inst[i.instance_id] = i.target_output
    for rec in val_recs + prim_recs:
        tgt = target_by_inst[rec["instance_id"]]
        for c in rec["candidates"]:
            c["label"] = 1 if c["grid"] == tgt else 0

    # ---- train + seed-select using validation distinct-task exact ----
    training_rows = []
    curve_rows = []
    seed_rows = []

    def val_metrics(model):
        solved_tasks = set(); exact_inst = 0; bce_num = 0.0; bce_den = 0
        for rec in val_recs:
            cands = rec["candidates"]
            if not cands:
                continue
            scores = score_candidates(model, [c["feat"] for c in cands], device)
            smap = {id(c): s for c, s in zip(cands, scores)}
            top = learned_selector(cands, smap)
            if any(c["label"] == 1 for c in top):
                exact_inst += 1; solved_tasks.add(rec["task_id"])
            for c, s in zip(cands, scores):
                # numerically-stable BCE-with-logits: max(s,0) - s*label + log1p(exp(-|s|))
                bce_num += max(s, 0.0) - s * c["label"] + math.log1p(math.exp(-abs(s))); bce_den += 1
        return len(solved_tasks), (exact_inst / len(val_recs) if val_recs else 0.0), (bce_num / bce_den if bce_den else 0.0)

    best_key = None; best_model = None; best_seed = None
    for seed in SEED_SLATE:
        model, info = fit_ranker(train_X, train_y, seed, shuffle_labels=False, device=device)
        training_rows.append({"arm": f"learned_ranker:seed={seed}", "rows": info["rows"], "pos": info["pos"], "pos_weight": info.get("pos_weight"), "epochs": info["epochs"], "best_loss": info.get("best_loss")})
        for h in info["history"]:
            curve_rows.append({"arm": "learned_ranker", "seed": seed, "epoch": h["epoch"], "loss": h["loss"]})
        vd, vr, vb = val_metrics(model)
        key = (-vd, -vr, vb, seed)
        seed_rows.append({"seed": seed, "val_distinct_tasks": vd, "val_exact_instance_rate": round_float(vr), "val_bce": round_float(vb), "selected": False})
        if best_key is None or key < best_key:
            best_key = key; best_model = model; best_seed = seed
    for r in seed_rows:
        r["selected"] = (r["seed"] == best_seed)
    manifest["selectedSeed"] = best_seed

    # metadata-only + label-shuffle controls (single seed = first slate seed)
    meta_model, meta_info = fit_ranker([_meta(x) for x in train_X], train_y, SEED_SLATE[0], shuffle_labels=False, device=device)
    training_rows.append({"arm": "metadata_only_ranker", "rows": meta_info["rows"], "pos": meta_info["pos"], "pos_weight": meta_info.get("pos_weight"), "epochs": meta_info["epochs"], "best_loss": meta_info.get("best_loss")})
    shuf_model, shuf_info = fit_ranker(train_X, train_y, SEED_SLATE[0], shuffle_labels=True, device=device)
    training_rows.append({"arm": "label_shuffle_ranker", "rows": shuf_info["rows"], "pos": shuf_info["pos"], "pos_weight": shuf_info.get("pos_weight"), "epochs": shuf_info["epochs"], "best_loss": shuf_info.get("best_loss")})

    write_csv(out_dir / "training_summary.csv", training_rows, RL_FILES["training_summary.csv"])
    write_csv(out_dir / "learning_curves.csv", curve_rows, RL_FILES["learning_curves.csv"])
    write_csv(out_dir / "seed_selection.csv", seed_rows, RL_FILES["seed_selection.csv"])

    # ---- score U_primary under all selectors ----
    sol_rows = []; quar_rows = []; score_rows = []
    fam_usage = {"v2_deterministic_selector": Counter(), "learned_ranker": Counter()}
    for rec in prim_recs:
        cands = rec["candidates"]
        if cands:
            lscores = score_candidates(best_model, [c["feat"] for c in cands], device)
            mscores = score_candidates(meta_model, [_meta(c["feat"]) for c in cands], device)
        else:
            lscores = []; mscores = []
        lmap = {id(c): s for c, s in zip(cands, lscores)}
        mmap = {id(c): s for c, s in zip(cands, mscores)}
        v2_top = v2_selector(cands)
        learned_top = learned_selector(cands, lmap)
        meta_top = learned_selector(cands, mmap)
        v2_hit = any(c["label"] == 1 for c in v2_top)
        learned_hit = any(c["label"] == 1 for c in learned_top)
        meta_hit = any(c["label"] == 1 for c in meta_top)
        ceiling = any(c["label"] == 1 for c in cands)
        if v2_hit:
            fam_usage["v2_deterministic_selector"][next(c["program_id"] for c in v2_top if c["label"] == 1)] += 1
        if learned_hit:
            fam_usage["learned_ranker"][next(c["program_id"] for c in learned_top if c["label"] == 1)] += 1
        # quarantine label
        if not cands:
            q = "no_admitted_programs"
        elif not ceiling:
            q = "budget_exhausted_candidate_unknown" if rec["budget_exhausted"] else "candidate_coverage_failure"
        elif v2_hit and learned_hit:
            q = "v2_ranker_already_solved"
        elif learned_hit and not v2_hit:
            q = "metadata_only_matches" if meta_hit else "v2_crowding_repaired"
        elif v2_hit and not learned_hit:
            q = "learned_ranker_regression"
        else:
            q = "learned_ranker_miss"
        sol_rows.append({"lane": rec["lane"], "instance_id": rec["instance_id"], "task_id": rec["task_id"],
                         "primary_prior": rec["primary_prior"], "n_candidates": rec["n_candidates"],
                         "v2_exact_any": v2_hit, "learned_exact_any": learned_hit, "metadata_exact_any": meta_hit,
                         "oracle_ceiling": ceiling,
                         "winning_family": (next((c["program_id"] for c in learned_top if c["label"] == 1), "") if learned_hit else "")})
        quar_rows.append({"lane": rec["lane"], "instance_id": rec["instance_id"], "task_id": rec["task_id"], "quarantine_label": q})
        score_rows.append({"instance_id": rec["instance_id"], "lane": rec["lane"], "task_id": rec["task_id"],
                           "learned_top2": [c["program_id"] for c in learned_top], "v2_top2": [c["program_id"] for c in v2_top],
                           "n_candidates": rec["n_candidates"], "oracle_ceiling": ceiling})
    write_jsonl(out_dir / "scores_by_instance.jsonl", score_rows)
    write_csv(out_dir / "solutions_by_instance.csv", sol_rows, RL_FILES["solutions_by_instance.csv"])
    write_csv(out_dir / "quarantine_by_instance.csv", quar_rows, RL_FILES["quarantine_by_instance.csv"])

    # ---- capability summaries + selector comparison + ceiling ----
    cap_rows = []; comp_rows = []; ceil_rows = []; per_prior_rows = []; fam_rows = []
    selectors = {"v2_deterministic_selector": "v2_exact_any", "learned_ranker": "learned_exact_any", "metadata_only_ranker": "metadata_exact_any"}
    for sel, key in selectors.items():
        for ln in prim_lanes:
            lr = [r for r in sol_rows if r["lane"] == ln]
            solved = {r["task_id"] for r in lr if r[key]}
            cap_rows.append({"selector": sel, "lane": ln, "n_instances": len(lr), "n_tasks": len({r["task_id"] for r in lr}),
                             "n_tasks_solved": len(solved), "exact_instance_rate": round_float(sum(1 for r in lr if r[key]) / len(lr) if lr else 0.0)})
            by_prior = defaultdict(lambda: [set(), set()])
            for r in lr:
                by_prior[r["primary_prior"]][0].add(r["task_id"])
                if r[key]:
                    by_prior[r["primary_prior"]][1].add(r["task_id"])
            for pr, (ts, sv) in sorted(by_prior.items()):
                per_prior_rows.append({"selector": sel, "lane": ln, "primary_prior": pr, "n_tasks": len(ts), "n_tasks_solved": len(sv)})
    for ln in prim_lanes:
        lr = [r for r in sol_rows if r["lane"] == ln]
        v2t = {r["task_id"] for r in lr if r["v2_exact_any"]}
        lt = {r["task_id"] for r in lr if r["learned_exact_any"]}
        mt = {r["task_id"] for r in lr if r["metadata_exact_any"]}
        ct = {r["task_id"] for r in lr if r["oracle_ceiling"]}
        comp_rows.append({"lane": ln, "v2_tasks": len(v2t), "learned_tasks": len(lt), "metadata_tasks": len(mt),
                          "oracle_ceiling_tasks": len(ct), "learned_new_vs_v1v2": len(lt - S_V1V2)})
        ceil_rows.append({"lane": ln, "n_instances": len(lr), "ceiling_solvable_instances": sum(1 for r in lr if r["oracle_ceiling"]), "ceiling_solvable_tasks": len(ct)})
    for sel in ("v2_deterministic_selector", "learned_ranker"):
        for fam, n in fam_usage[sel].most_common():
            fam_rows.append({"selector": sel, "winning_family": fam, "n_solved_instances": n})
    write_csv(out_dir / "capability_summary.csv", cap_rows, RL_FILES["capability_summary.csv"])
    write_csv(out_dir / "selector_comparison.csv", comp_rows, RL_FILES["selector_comparison.csv"])
    write_csv(out_dir / "candidate_ceiling.csv", ceil_rows, RL_FILES["candidate_ceiling.csv"])
    write_csv(out_dir / "per_prior_capability.csv", per_prior_rows, RL_FILES["per_prior_capability.csv"])
    write_csv(out_dir / "family_usage.csv", fam_rows, RL_FILES["family_usage.csv"])

    # ---- branch adjudication (learned_ranker over U_primary vs S_v1v2) ----
    learned_tl = _solved_tasks(sol_rows, "test_lodo", "learned_exact_any")
    learned_pt = _solved_tasks(sol_rows, "pttest", "learned_exact_any")
    v2_tl = _solved_tasks(sol_rows, "test_lodo", "v2_exact_any")
    v2_pt = _solved_tasks(sol_rows, "pttest", "v2_exact_any")
    tl_new = len(learned_tl - S_V1V2); pt_new = len(learned_pt - S_V1V2)
    u_primary_exact = sum(1 for r in sol_rows if r["learned_exact_any"])
    if len(learned_tl) >= MATERIAL_LIFT_MIN_TASKS and len(learned_pt) >= MATERIAL_LIFT_MIN_TASKS and tl_new >= MATERIAL_LIFT_MIN_NEW and pt_new >= MATERIAL_LIFT_MIN_NEW:
        branch = "branch_e3_ranker_material_lift"
    elif (len(learned_tl - S_V1V2) >= 1) or (len(learned_pt - S_V1V2) >= 1):
        branch = "branch_e3_ranker_selector_lift_below_material"
    elif len(learned_tl) >= REPLICATED_MIN_TASKS and len(learned_pt) >= REPLICATED_MIN_TASKS and tl_new == 0 and pt_new == 0:
        branch = "branch_e3_ranker_replicated"
    elif (len(learned_tl) < REPLICATED_MIN_TASKS or len(learned_pt) < REPLICATED_MIN_TASKS) and len(v2_tl) >= REPLICATED_MIN_TASKS and len(v2_pt) >= REPLICATED_MIN_TASKS:
        branch = "branch_e3_ranker_regression"
    elif u_primary_exact > 0:
        branch = "branch_e3_ranker_partial"
    else:
        branch = "branch_e3_ranker_floor"

    manifest["completedAt"] = iso_now()
    manifest["capability"] = {
        "learned": {"test_lodo": sorted(learned_tl), "pttest": sorted(learned_pt), "test_lodo_new": tl_new, "pttest_new": pt_new},
        "v2": {"test_lodo": sorted(v2_tl), "pttest": sorted(v2_pt)},
        "u_primary_exact_instances": u_primary_exact,
    }
    manifest["branch"] = branch
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "ranker_feature_schema.json", {"schema": FEATURE_SCHEMA_VERSION, "input_dim": INPUT_DIM, "program_hash_dim": PROGRAM_HASH_DIM,
                                                         "groups": ["program_syntax", "program_id_hash", "candidate_grid", "candidate_vs_input", "conditioning_summary"]})
    write_json(out_dir / "phase3_branch_e3_learned_ranker_receipt.json", {"manifest": manifest, "branch": branch})
    (out_dir / "branch_adjudication.md").write_text(
        f"# Branch E3 -- Learned Ranker -- Branch Adjudication\n\n**Branch: `{branch}`**\n\n"
        f"- learned_ranker U_primary: test_lodo solved {len(learned_tl)} (new vs v1/v2 {tl_new}), pttest {len(learned_pt)} (new {pt_new})\n"
        f"- v2_deterministic control: test_lodo {len(v2_tl)}, pttest {len(v2_pt)}\n"
        f"- S_v1v2 = {sorted(S_V1V2)}\n\n"
        f"Selection is learned over the FROZEN v2 candidate set (train-pair-consistent programs only); "
        f"features are target-free; U_primary targets read only for final scoring after the barrier.\n",
        encoding="utf-8")
    (out_dir / "commands.md").write_text("# Branch E3 commands\n\nSee the spec freeze-marker amendment for the staged shard + merge commands.\n", encoding="utf-8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Branch E3 merge wrote {out_dir}")
    print(f"Branch: {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
