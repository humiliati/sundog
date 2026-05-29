#!/usr/bin/env python
"""ARC Branch E v2 deterministic program-search solver.

This runner is an append-only expansion of Branch E v1. It reuses the v1
loaders, IO, structural programs, candidate combinators, and Branch D
deterministic color-rule bank. It changes exactly three frozen search knobs:

1. admits the deferred deterministic mask families;
2. applies the deterministic morphology cross-product;
3. allows structural >> structural >> terminal depth-3 programs.

It never reads signature geometry or public-evaluation grids, and it admits a
program only when it reproduces every conditioning train pair exactly.

Spec: docs/prereg/arc/PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md
Parent: docs/prereg/arc/PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import phase3_branch_e_program_search as v1  # noqa: E402


LEARNER_VERSION = "branch_e_v2_program_search"
PROTOCOL_VERSION = "arc-p3-branch-e-v2"
RECEIPT_SCHEMA_VERSION = "arc-p3-branch-e-v2-receipt-v1"

ATTEMPTS = 2
MAX_DEPTH = 3
CANDIDATE_BUDGET = 20000
CAP_MIN_TASKS = 2
MATERIAL_LIFT_MIN_TASKS = 4
MATERIAL_LIFT_MIN_NEW_TASKS = 2

V1_GATED_SOLVED_TASKS = {
    "test_lodo": {"be94b721", "f25fbde4"},
    "pttest": {"be94b721", "f25fbde4"},
}

V2_MASK_FAMILIES = [
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
    "full_mask",
]
V2_MASK_MORPH_OPS = ["identity", "dilate1", "erode1", "close1", "bbox_fill"]
V2_ONLY_MASK_FAMILIES = {
    "row_col_periodic_mask",
    "source_color_pair_mask",
    "object_role_mask",
    "nearest_residual_patch_mask",
}
V2_ONLY_MORPH_OPS = {"dilate1", "erode1", "close1", "bbox_fill"}


BE_V2_FILES_EMPTY = {
    "split.csv": ["task_id", "primary_prior", "predicted_boundary", "split"],
    "solutions_by_instance.csv": [
        "lane",
        "instance_id",
        "task_id",
        "primary_prior",
        "n_admitted",
        "exact_slot1",
        "exact_any",
        "winning_family",
        "winning_depth",
        "winning_v2_feature",
        "budget_exhausted",
    ],
    "capability_summary.csv": ["lane", "n_instances", "n_tasks", "n_tasks_solved", "exact_instance_rate"],
    "per_prior_capability.csv": ["lane", "primary_prior", "n_tasks", "n_tasks_solved"],
    "family_usage.csv": ["winning_family", "n_solved_instances"],
    "v1_comparison.csv": ["lane", "v1_solved_tasks", "v2_solved_tasks", "retained_v1_tasks", "new_v2_tasks"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC Branch E v2 deterministic program-search solver")
    parser.add_argument("--data-dir", required=False, default=None)
    parser.add_argument("--register", required=False, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit-tasks", type=int, default=0)
    parser.add_argument("--task-id", action="append", default=[], help="Optional smoke/debug task id filter. Not used for binding.")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--split-mode", choices=["frozen_v2", "sha256_expansion"], default="frozen_v2")
    args = parser.parse_args()
    if not args.dry_run and (not args.data_dir or not args.register):
        parser.error("--data-dir and --register are required (except in --dry-run mode)")
    return args


def _mask_descriptor_sort_key(cand: dict[str, Any]) -> tuple[int, int, str]:
    family = cand["family"]
    family_idx = V2_MASK_FAMILIES.index(family) if family in V2_MASK_FAMILIES else 999
    morph = cand["id"].rsplit("|", 1)[-1] if "|" in cand["id"] else "identity"
    morph_idx = V2_MASK_MORPH_OPS.index(morph) if morph in V2_MASK_MORPH_OPS else 999
    return (family_idx, morph_idx, cand["id"])


def generate_deterministic_mask_candidates_v2(
    query_input: list[list[int]],
    query_baseline: list[list[int]],
    conditioning: list[dict[str, Any]],
    cond_baselines: list[list[list[int]]],
) -> list[dict[str, Any]]:
    """Deterministic Branch D mask bank without the learned legacy MLP family."""

    qh = len(query_baseline)
    qw = len(query_baseline[0]) if qh else 0
    cond_masks_native = [
        v1._conditioning_gold_mask(b, p["output"]) for b, p in zip(cond_baselines, conditioning)
    ]
    cond_masks_q = [v1._project_mask_to_shape(m, qh, qw) for m in cond_masks_native]

    base_candidates: list[dict[str, Any]] = []
    base_candidates.append({"family": "empty_mask", "id": "v1", "mask": [[False] * qw for _ in range(qh)]})

    union = [[False] * qw for _ in range(qh)]
    for m in cond_masks_q:
        for y in range(qh):
            for x in range(qw):
                if m[y][x]:
                    union[y][x] = True
    base_candidates.append({"family": "conditioning_mask_union", "id": "v1", "mask": union})

    if cond_masks_q:
        inter = [[True] * qw for _ in range(qh)]
        counts = [[0] * qw for _ in range(qh)]
        for m in cond_masks_q:
            for y in range(qh):
                for x in range(qw):
                    if not m[y][x]:
                        inter[y][x] = False
                    if m[y][x]:
                        counts[y][x] += 1
        base_candidates.append({"family": "conditioning_mask_intersection", "id": "v1", "mask": inter})
        thr = len(cond_masks_q) / 2
        majority = [[counts[y][x] >= thr for x in range(qw)] for y in range(qh)]
        base_candidates.append({"family": "conditioning_mask_majority", "id": "v1", "mask": majority})

    bboxes = [b for b in (v1._bbox_of_mask(m) for m in cond_masks_q) if b is not None]
    if bboxes:
        u_x0 = min(b[0] for b in bboxes)
        u_y0 = min(b[1] for b in bboxes)
        u_x1 = max(b[2] for b in bboxes)
        u_y1 = max(b[3] for b in bboxes)
        ubox = [[u_y0 <= y <= u_y1 and u_x0 <= x <= u_x1 for x in range(qw)] for y in range(qh)]
        base_candidates.append({"family": "conditioning_bbox_fill", "id": "union", "mask": ubox})
        i_x0 = max(b[0] for b in bboxes)
        i_y0 = max(b[1] for b in bboxes)
        i_x1 = min(b[2] for b in bboxes)
        i_y1 = min(b[3] for b in bboxes)
        if i_x0 <= i_x1 and i_y0 <= i_y1:
            ibox = [[i_y0 <= y <= i_y1 and i_x0 <= x <= i_x1 for x in range(qw)] for y in range(qh)]
            base_candidates.append({"family": "conditioning_bbox_fill", "id": "intersection", "mask": ibox})
        bcounts = [[0] * qw for _ in range(qh)]
        for b in bboxes:
            for y in range(b[1], b[3] + 1):
                for x in range(b[0], b[2] + 1):
                    if 0 <= y < qh and 0 <= x < qw:
                        bcounts[y][x] += 1
        bthr = len(bboxes) / 2
        mbox = [[bcounts[y][x] >= bthr for x in range(qw)] for y in range(qh)]
        base_candidates.append({"family": "conditioning_bbox_fill", "id": "majority", "mask": mbox})
        outline = [[False] * qw for _ in range(qh)]
        for y in range(u_y0, u_y1 + 1):
            for x in range(u_x0, u_x1 + 1):
                if 0 <= y < qh and 0 <= x < qw and (y in (u_y0, u_y1) or x in (u_x0, u_x1)):
                    outline[y][x] = True
        base_candidates.append({"family": "conditioning_bbox_outline", "id": "union", "mask": outline})

    for axis in ("row", "col"):
        for period in (1, 2, 3):
            active: set[int] = set()
            for m in cond_masks_q:
                for y in range(qh):
                    for x in range(qw):
                        if m[y][x]:
                            active.add((y if axis == "row" else x) % period)
            if active:
                mask = [[((y if axis == "row" else x) % period) in active for x in range(qw)] for y in range(qh)]
                base_candidates.append({"family": "row_col_periodic_mask", "id": f"{axis}_p{period}", "mask": mask})

    source_colors: set[int] = set()
    for pair, b, m in zip(conditioning, cond_baselines, cond_masks_native):
        bh = len(b)
        bw = len(b[0]) if bh else 0
        for y in range(bh):
            for x in range(bw):
                if m[y][x]:
                    source_colors.add(v1._nearest_input_color(pair["input"], (bh, bw), y, x))
                    source_colors.add(b[y][x])
    if source_colors:
        sc_mask = [[False] * qw for _ in range(qh)]
        for y in range(qh):
            for x in range(qw):
                src_in = v1._nearest_input_color(query_input, (qh, qw), y, x)
                src_bs = query_baseline[y][x]
                if src_in in source_colors or src_bs in source_colors:
                    sc_mask[y][x] = True
        base_candidates.append({"family": "source_color_mask", "id": "v1", "mask": sc_mask})

    source_pairs: set[tuple[int, int]] = set()
    for pair, b, m in zip(conditioning, cond_baselines, cond_masks_native):
        bh = len(b)
        bw = len(b[0]) if bh else 0
        for y in range(bh):
            for x in range(bw):
                if m[y][x]:
                    source_pairs.add((v1._nearest_input_color(pair["input"], (bh, bw), y, x), b[y][x]))
    if source_pairs:
        sp_mask = [[False] * qw for _ in range(qh)]
        for y in range(qh):
            for x in range(qw):
                src_in = v1._nearest_input_color(query_input, (qh, qw), y, x)
                src_bs = query_baseline[y][x]
                if (src_in, src_bs) in source_pairs:
                    sp_mask[y][x] = True
        base_candidates.append({"family": "source_color_pair_mask", "id": "v1", "mask": sp_mask})

    target_roles: set[int] = set()
    for pair, b, m in zip(conditioning, cond_baselines, cond_masks_native):
        comps = v1._components([[cell != 0 for cell in row] for row in pair["input"]])
        role_at: dict[tuple[int, int], int] = {}
        for idx, comp in enumerate(sorted(comps, key=lambda c: -len(c))):
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
                    role = role_at.get((iy, ix), 0)
                    if role > 0:
                        target_roles.add(role)
    if target_roles:
        q_comps = v1._components([[cell != 0 for cell in row] for row in query_input])
        q_role_mask_input = [[0] * (len(query_input[0]) if query_input else 0) for _ in range(len(query_input))]
        for idx, comp in enumerate(sorted(q_comps, key=lambda c: -len(c))):
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
        for thr in v1.MASK_PATCH_THRESHOLDS:
            knn_mask = [[False] * qw for _ in range(qh)]
            for y in range(qh):
                for x in range(qw):
                    ny = y / (qh - 1) if qh > 1 else 0.0
                    nx = x / (qw - 1) if qw > 1 else 0.0
                    dists = [((nyk - ny) ** 2 + (nxk - nx) ** 2, edited) for nyk, nxk, edited in knn_table]
                    dists.sort(key=lambda d: d[0])
                    k = min(3, len(dists))
                    vote = sum(1 for _, edited in dists[:k] if edited) / k
                    if vote >= thr:
                        knn_mask[y][x] = True
            base_candidates.append({"family": "nearest_residual_patch_mask", "id": f"thr={thr}", "mask": knn_mask})

    if cond_masks_q:
        delta = [[False] * qw for _ in range(qh)]
        for m in cond_masks_q:
            for y in range(qh):
                for x in range(qw):
                    if m[y][x]:
                        delta[y][x] = True
        base_candidates.append({"family": "delta_overlay_mask", "id": "v1", "mask": delta})

    base_candidates.append({"family": "full_mask", "id": "identity", "mask": [[True] * qw for _ in range(qh)]})

    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for cand in base_candidates:
        if cand["family"] in {"empty_mask", "full_mask"}:
            key = (cand["family"], cand["id"])
            if key not in seen:
                seen.add(key)
                candidates.append(cand)
            continue
        for op in V2_MASK_MORPH_OPS:
            key = (cand["family"], f"{cand['id']}|{op}")
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "family": cand["family"],
                "id": f"{cand['id']}|{op}",
                "mask": v1._apply_morph(cand["mask"], op),
            })
    candidates.sort(key=_mask_descriptor_sort_key)
    return candidates


def mask_family_fns_v2(pairs: list[dict[str, Any]]) -> list[tuple[str, Any]]:
    if not pairs:
        return []
    cond_baselines = [p["input"] for p in pairs]
    seed_cands = generate_deterministic_mask_candidates_v2(
        pairs[0]["input"], pairs[0]["input"], pairs, cond_baselines
    )
    descriptors = [(c["family"], c["id"]) for c in seed_cands]
    candidate_cache: dict[str, dict[tuple[str, str], list[list[bool]]]] = {}

    def _bank_for(gg: list[list[int]]) -> dict[tuple[str, str], list[list[bool]]]:
        key = json.dumps(gg, separators=(",", ":"))
        if key not in candidate_cache:
            cands = generate_deterministic_mask_candidates_v2(gg, gg, pairs, cond_baselines)
            candidate_cache[key] = {(c["family"], c["id"]): c["mask"] for c in cands}
        return candidate_cache[key]

    out = []
    for family, mid in descriptors:
        def _mk(family: str, mid: str):
            def f(gg: list[list[int]]):
                return _bank_for(gg).get((family, mid))
            return f
        out.append((f"{family}:{mid}", _mk(family, mid)))
    return out


def color_edit_programs_v2(pairs: list[dict[str, Any]]) -> list[tuple[str, int, int, Any]]:
    if not all(v1._dims(p["input"]) == v1._dims(p["output"]) for p in pairs):
        return []
    cond_baselines = [p["input"] for p in pairs]
    rules = v1._safe(lambda: v1.generate_candidate_rules(pairs, cond_baselines)) or []
    masks = mask_family_fns_v2(pairs)
    progs = []
    for rule in rules:
        for mname, mfn in masks:
            priority = 13 if mname == "full_mask:identity" else 14

            def _mk(rule: dict[str, Any], mfn: Any):
                def f(gg: list[list[int]]):
                    mask = mfn(gg)
                    if mask is None:
                        return None
                    return v1._predict_with_rule(rule, gg, gg, mask)
                return f
            progs.append((f"coloredit:{rule['family']}:{rule['id']}:{mname}", priority, 4, _mk(rule, mfn)))
    return progs


def _consistent(fn: Any, pairs: list[dict[str, Any]]) -> bool:
    for p in pairs:
        out = v1._safe(lambda: fn(p["input"]))
        if out is None or out != p["output"]:
            return False
    return True


def _transform_pairs(pairs: list[dict[str, Any]], fn: Any) -> list[dict[str, Any]] | None:
    transformed = []
    for p in pairs:
        ti = v1._safe(lambda: fn(p["input"]))
        if ti is None:
            return None
        transformed.append({"input": ti, "output": p["output"]})
    return transformed


def _compose(*fns: Any):
    def composed(gg: list[list[int]]):
        cur = gg
        for fn in fns:
            cur = fn(cur)
            if cur is None:
                return None
        return cur
    return composed


def _program_depth(name: str) -> int:
    return name.count(">>") + 1 if name else 0


def _v2_feature(name: str) -> str:
    if not name:
        return ""
    features = []
    if _program_depth(name) >= 3:
        features.append("depth3")
    for fam in sorted(V2_ONLY_MASK_FAMILIES):
        if fam in name:
            features.append(fam)
    for op in sorted(V2_ONLY_MORPH_OPS):
        if f"|{op}" in name:
            features.append(f"morph_{op}")
    return "+".join(features)


def solve_instance(inst: Any) -> dict[str, Any]:
    pairs = inst.conditioning
    query = inst.query_input
    admitted: list[tuple[int, int, str, list[list[int]]]] = []
    budget = CANDIDATE_BUDGET

    def _try(name: str, priority: int, complexity: int, fn: Any) -> None:
        nonlocal budget
        if budget <= 0:
            return
        budget -= 1
        if _consistent(fn, pairs):
            cand = v1._safe(lambda: fn(query))
            if cand is not None and len(cand) > 0 and len(cand[0]) > 0:
                admitted.append((priority, complexity, name, cand))

    base_progs = v1.structural_programs(pairs) + v1.combinator_programs(pairs) + color_edit_programs_v2(pairs)
    for name, priority, complexity, fn in base_progs:
        _try(name, priority, complexity, fn)

    stage1 = v1.structural_programs(pairs)
    for n1, p1, c1, f1 in stage1:
        if n1 == "identity" or budget <= 0:
            continue
        t1_pairs = _transform_pairs(pairs, f1)
        if t1_pairs is None:
            continue
        stage2 = v1.structural_programs(t1_pairs) + color_edit_programs_v2(t1_pairs)
        for n2, p2, c2, f2 in stage2:
            if n2 == "identity" or budget <= 0:
                continue
            _try(f"{n1}>>{n2}", max(p1, p2) + 15, c1 + c2 + 1, _compose(f1, f2))

        stage2_structural = v1.structural_programs(t1_pairs)
        for n2, p2, c2, f2 in stage2_structural:
            if n2 == "identity" or budget <= 0:
                continue
            t2_pairs = _transform_pairs(t1_pairs, f2)
            if t2_pairs is None:
                continue
            stage3 = v1.structural_programs(t2_pairs) + color_edit_programs_v2(t2_pairs)
            for n3, p3, c3, f3 in stage3:
                if n3 == "identity" or budget <= 0:
                    continue
                _try(f"{n1}>>{n2}>>{n3}", max(p1, p2, p3) + 30, c1 + c2 + c3 + 2, _compose(f1, f2, f3))

    admitted.sort(key=lambda t: (t[0], t[1], t[2]))
    attempts: list[tuple[str, list[list[int]]]] = []
    seen = set()
    for _priority, _complexity, name, grid in admitted:
        key = json.dumps(grid, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        attempts.append((name, grid))
        if len(attempts) >= ATTEMPTS:
            break
    target = inst.target_output
    exact1 = bool(attempts) and attempts[0][1] == target
    exact_any = any(g == target for _, g in attempts)
    winning = next((n for n, g in attempts if g == target), "")
    return {
        "lane": inst.lane,
        "instance_id": inst.instance_id,
        "task_id": inst.task_id,
        "primary_prior": inst.primary_prior,
        "n_admitted": len(admitted),
        "exact_slot1": exact1,
        "exact_any": exact_any,
        "winning_family": winning,
        "winning_depth": _program_depth(winning),
        "winning_v2_feature": _v2_feature(winning),
        "budget_exhausted": budget <= 0,
    }


def write_empty_receipt(out_dir: Path, manifest: dict[str, Any]) -> None:
    v1.write_json(out_dir / "manifest.json", manifest)
    for fname, cols in BE_V2_FILES_EMPTY.items():
        v1.write_csv(out_dir / fname, [], cols)
    v1.write_jsonl(out_dir / "context_fingerprints_no_targets.jsonl", [])
    (out_dir / "context_fingerprints_no_targets.sha256").write_text("", encoding="utf-8")
    v1.write_jsonl(out_dir / "programs_by_instance.jsonl", [])
    v1.write_json(out_dir / "phase3_branch_e_v2_program_search_receipt.json", {"manifest": manifest, "branch": None})
    (out_dir / "branch_adjudication.md").write_text("# Branch E v2 program search\n\nDry run / empty receipt.\n", encoding="utf-8")
    (out_dir / "commands.md").write_text("# Branch E v2 commands\n\nDry run / empty receipt.\n", encoding="utf-8")
    v1.write_json(out_dir / "hashes.json", v1.hash_receipt_files(out_dir))


def main() -> int:
    args = parse_args()
    started_at = v1.iso_now()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    git = v1.git_state(repo_root, args.allow_dirty)

    runner_path = Path(__file__).resolve()
    spec_path = runner_path.parent / "PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md"
    parent_spec_path = runner_path.parent / "PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md"
    parent_runner_path = runner_path.parent / "phase3_branch_e_program_search.py"

    manifest: dict[str, Any] = {
        "generatedAt": started_at,
        "completedAt": None,
        "tool": "docs/prereg/arc/phase3_branch_e_v2_program_search.py",
        "command": [sys.executable, "docs/prereg/arc/phase3_branch_e_v2_program_search.py", *sys.argv[1:]],
        "gitCommit": git["commit"],
        "gitDirty": git["dirty"],
        "allowDirty": args.allow_dirty,
        "outDir": str(out_dir),
        "featureSchemaVersion": v1.FEATURE_SCHEMA_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION,
        "learnerVersion": LEARNER_VERSION,
        "specPath": "docs/prereg/arc/PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md",
        "specHash": v1.sha256_file(spec_path) if spec_path.exists() else "NA",
        "parentBranchESpecHash": v1.sha256_file(parent_spec_path) if parent_spec_path.exists() else "NA",
        "parentBranchERunnerHash": v1.sha256_file(parent_runner_path) if parent_runner_path.exists() else "NA",
        "runnerSha256": v1.sha256_file(runner_path),
        "pythonVersion": sys.version,
        "platform": platform.platform(),
        "attempts": ATTEMPTS,
        "maxDepth": MAX_DEPTH,
        "candidateBudget": CANDIDATE_BUDGET,
        "capabilityFloorMinTasks": CAP_MIN_TASKS,
        "materialLiftMinTasks": MATERIAL_LIFT_MIN_TASKS,
        "materialLiftMinNewTasks": MATERIAL_LIFT_MIN_NEW_TASKS,
        "v1GatedSolvedTasks": {lane: sorted(tasks) for lane, tasks in V1_GATED_SOLVED_TASKS.items()},
        "maskFamilies": V2_MASK_FAMILIES,
        "maskMorphology": V2_MASK_MORPH_OPS,
        "taskIdFilter": args.task_id,
    }

    if args.dry_run:
        manifest["mode"] = "dry_run"
        manifest["completedAt"] = v1.iso_now()
        write_empty_receipt(out_dir, manifest)
        print(f"ARC Branch E v2 dry run wrote {out_dir}")
        return 0

    data_dir = Path(args.data_dir).resolve()
    register_path = Path(args.register).resolve()
    v1.assert_training_data_dir(data_dir)
    tasks, register_hash, data_hash = v1.load_tasks(data_dir, register_path, args.split_mode)
    if args.task_id:
        wanted = set()
        for raw in args.task_id:
            wanted.update(part.strip() for part in raw.split(",") if part.strip())
        tasks = [t for t in tasks if t.task_id in wanted]
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    manifest["dataDir"] = str(data_dir)
    manifest["registerPath"] = str(register_path)
    manifest["registerHash"] = register_hash
    manifest["dataDirHash"] = data_hash
    manifest["splitMode"] = args.split_mode

    validation_tasks = [t for t in tasks if t.split == "validation"]
    test_tasks = [t for t in tasks if t.split == "test"]
    v1.write_csv(
        out_dir / "split.csv",
        [{"task_id": t.task_id, "primary_prior": t.primary_prior, "predicted_boundary": t.predicted_boundary, "split": t.split}
         for t in sorted(tasks, key=lambda x: x.task_id)],
        BE_V2_FILES_EMPTY["split.csv"],
    )

    instances = (
        v1.build_lodo_instances(validation_tasks, "validation_lodo")
        + v1.build_pttest_instances(validation_tasks, "validation_pttest")
        + v1.build_lodo_instances(test_tasks, "test_lodo")
        + v1.build_pttest_instances(test_tasks, "pttest")
    )

    fingerprints = [
        {
            "instance_id": i.instance_id,
            "lane": i.lane,
            "task_id": i.task_id,
            "query_index": i.query_index,
            "n_conditioning": len(i.conditioning),
        }
        for i in instances
    ]
    v1.write_jsonl(out_dir / "context_fingerprints_no_targets.jsonl", fingerprints)
    barrier_hash = v1.sha256_file(out_dir / "context_fingerprints_no_targets.jsonl")
    (out_dir / "context_fingerprints_no_targets.sha256").write_text(barrier_hash + "\n", encoding="utf-8")
    manifest["targetBarrierHash"] = barrier_hash

    rows = [solve_instance(i) for i in instances]
    v1.write_jsonl(out_dir / "programs_by_instance.jsonl", rows)
    v1.write_csv(out_dir / "solutions_by_instance.csv", rows, BE_V2_FILES_EMPTY["solutions_by_instance.csv"])

    lanes = ["validation_lodo", "validation_pttest", "test_lodo", "pttest"]
    summary_rows = []
    solved_tasks_by_lane: dict[str, set[str]] = {ln: set() for ln in lanes}
    per_prior: dict[tuple[str, str], dict[str, set[str]]] = {}
    family_solved: dict[str, int] = {}
    for lane in lanes:
        lane_rows = [r for r in rows if r["lane"] == lane]
        tasks_in = {r["task_id"] for r in lane_rows}
        solved = {r["task_id"] for r in lane_rows if r["exact_any"]}
        solved_tasks_by_lane[lane] = solved
        n_inst = len(lane_rows)
        n_exact_inst = sum(1 for r in lane_rows if r["exact_any"])
        summary_rows.append({
            "lane": lane,
            "n_instances": n_inst,
            "n_tasks": len(tasks_in),
            "n_tasks_solved": len(solved),
            "exact_instance_rate": v1.round_float(n_exact_inst / n_inst if n_inst else 0.0),
        })
        for r in lane_rows:
            key = (lane, r["primary_prior"])
            d = per_prior.setdefault(key, {"tasks": set(), "solved": set()})
            d["tasks"].add(r["task_id"])
            if r["exact_any"]:
                d["solved"].add(r["task_id"])
                family_solved[r["winning_family"]] = family_solved.get(r["winning_family"], 0) + 1

    v1.write_csv(out_dir / "capability_summary.csv", summary_rows, BE_V2_FILES_EMPTY["capability_summary.csv"])
    v1.write_csv(
        out_dir / "per_prior_capability.csv",
        [{"lane": lane, "primary_prior": prior, "n_tasks": len(d["tasks"]), "n_tasks_solved": len(d["solved"])}
         for (lane, prior), d in sorted(per_prior.items())],
        BE_V2_FILES_EMPTY["per_prior_capability.csv"],
    )
    v1.write_csv(
        out_dir / "family_usage.csv",
        [{"winning_family": family, "n_solved_instances": n}
         for family, n in sorted(family_solved.items(), key=lambda t: (-t[1], t[0]))],
        BE_V2_FILES_EMPTY["family_usage.csv"],
    )

    comparison_rows = []
    for lane in ("test_lodo", "pttest"):
        v1_tasks = V1_GATED_SOLVED_TASKS[lane]
        v2_tasks = solved_tasks_by_lane[lane]
        comparison_rows.append({
            "lane": lane,
            "v1_solved_tasks": len(v1_tasks),
            "v2_solved_tasks": len(v2_tasks),
            "retained_v1_tasks": len(v1_tasks & v2_tasks),
            "new_v2_tasks": len(v2_tasks - v1_tasks),
        })
    v1.write_csv(out_dir / "v1_comparison.csv", comparison_rows, BE_V2_FILES_EMPTY["v1_comparison.csv"])

    test_lodo_solved = len(solved_tasks_by_lane["test_lodo"])
    pttest_solved = len(solved_tasks_by_lane["pttest"])
    test_lodo_new = len(solved_tasks_by_lane["test_lodo"] - V1_GATED_SOLVED_TASKS["test_lodo"])
    pttest_new = len(solved_tasks_by_lane["pttest"] - V1_GATED_SOLVED_TASKS["pttest"])
    u_primary_exact = sum(1 for r in rows if r["lane"] in ("test_lodo", "pttest") and r["exact_any"])

    if (
        test_lodo_solved >= MATERIAL_LIFT_MIN_TASKS
        and pttest_solved >= MATERIAL_LIFT_MIN_TASKS
        and test_lodo_new >= MATERIAL_LIFT_MIN_NEW_TASKS
        and pttest_new >= MATERIAL_LIFT_MIN_NEW_TASKS
    ):
        branch = "branch_e_v2_material_lift"
        reason = (
            f"Material lift: test_lodo solved {test_lodo_solved} tasks ({test_lodo_new} new vs v1), "
            f"pttest solved {pttest_solved} tasks ({pttest_new} new vs v1)."
        )
    elif test_lodo_solved >= CAP_MIN_TASKS and pttest_solved >= CAP_MIN_TASKS:
        branch = "branch_e_v2_capability_replicated"
        reason = (
            f"Replicated Branch E floor-clearing capability: test_lodo solved {test_lodo_solved}, "
            f"pttest solved {pttest_solved}, but material lift gate was not met."
        )
    elif u_primary_exact > 0:
        branch = "branch_e_v2_capability_partial"
        reason = (
            f"Non-zero capability below replicated threshold: test_lodo solved {test_lodo_solved}, "
            f"pttest solved {pttest_solved}."
        )
    else:
        branch = "branch_e_v2_capability_floor"
        reason = "Zero held-out U_primary instances solved exactly."

    manifest["completedAt"] = v1.iso_now()
    manifest["contextUniverse"] = {lane: len([r for r in rows if r["lane"] == lane]) for lane in lanes}
    manifest["capability"] = {
        "test_lodo_solved_tasks": test_lodo_solved,
        "pttest_solved_tasks": pttest_solved,
        "test_lodo_new_vs_v1": test_lodo_new,
        "pttest_new_vs_v1": pttest_new,
        "u_primary_exact_instances": u_primary_exact,
        "validation_lodo_solved_tasks": len(solved_tasks_by_lane["validation_lodo"]),
        "validation_pttest_solved_tasks": len(solved_tasks_by_lane["validation_pttest"]),
    }
    manifest["branch"] = branch

    v1.write_json(out_dir / "manifest.json", manifest)
    v1.write_json(
        out_dir / "phase3_branch_e_v2_program_search_receipt.json",
        {"manifest": manifest, "branch": branch, "branchReason": reason},
    )
    (out_dir / "branch_adjudication.md").write_text(
        f"# Branch E v2 -- Deterministic Program Search -- Branch Adjudication\n\n"
        f"**Branch: `{branch}`**\n\n{reason}\n\n"
        f"- split_mode {args.split_mode}; attempts {ATTEMPTS}; max_depth {MAX_DEPTH}; budget {CANDIDATE_BUDGET}\n"
        f"- U_primary solved tasks: test_lodo {test_lodo_solved} ({test_lodo_new} new vs v1), "
        f"pttest {pttest_solved} ({pttest_new} new vs v1)\n"
        f"- diagnostic: validation_lodo {len(solved_tasks_by_lane['validation_lodo'])}, "
        f"validation_pttest {len(solved_tasks_by_lane['validation_pttest'])}\n\n"
        f"Selection is train-pair consistency only; signature_palette geometry is never used. "
        f"The v2 delta is deterministic intricate masks, morphology, and depth-3 composition.\n",
        encoding="utf-8",
    )
    (out_dir / "commands.md").write_text(
        "# Branch E v2 commands\n\n```\nnode scripts/arc-phase3-branch-e-v2-program-search.mjs \\\n"
        "  --data-dir \"$env:USERPROFILE/Datasets/ARC-AGI-2/data\" \\\n"
        "  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv \\\n"
        "  --split-mode sha256_expansion \\\n"
        "  --out results/arc/phase3-branch-e-v2-program-search\n```\n",
        encoding="utf-8",
    )
    v1.write_json(out_dir / "hashes.json", v1.hash_receipt_files(out_dir))
    print(f"ARC Branch E v2 program search wrote {out_dir}")
    print(f"Branch: {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
