"""GEN-1 Object-DSL v0 ceiling probe (gate-2) — perception parser + CEGIS hole-solvers + probe harness.

Implements EXACTLY the frozen vocabulary/grammar of GEN1_OBJECT_DSL_V0_PROBE_SPEC.md (SS1-SS2) and the
validation-only ceiling probe (SS3). Adjudicates NO capability branch: it measures whether the v0
object-DSL candidate bank contains the correct output for materially more validation tasks than the
frozen Branch-E v2 bank (baseline computed OFFLINE from the E3 receipt's validation fingerprints,
per-candidate grid_hash, identical hash function). Barrier discipline: all candidates are generated
from conditioning pairs only and fingerprinted+hashed BEFORE any target is read.

Deterministic implementation choices (frozen with the tooling amendment):
- object canonical order = (r0, c0, shape); color ties -> smallest color id;
- modal(shape) tie -> smallest shape key; unique(attr) = all objects whose attr-value freq == 1;
- gravity processes objects far-side-first (down: descending bottom edge; up: ascending r0; etc.);
- paints clip silently at grid bounds; later paints overwrite (canonical order);
- budget counts FULL verifications (pair-0 hole pre-filtering is cheap and uncounted), cap 20000;
- hole domains: recolor kappa = colors in conditioning OUTPUTS; where kappa = colors in conditioning
  INPUTS; move/copy delta = same-shape same-color component matches in output0 (cap 64); dims =
  constant conditioning output dims, or constant integer ratio (a,b)!=(1,1).
Run: python docs/prereg/arc/gen1_object_dsl_probe.py --self-test   (solver-correctness synthetics)
     ... --data-dir <ARC> --register <csv> --split-mode sha256_expansion --out <dir>   (the probe)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_branch_e_program_search as v1  # noqa: E402  (frozen loaders / IO / hashing)

VIEWS = ("cc4", "cc8", "blob4")
GRAVITY_DIRS = ("up", "down", "left", "right")
D4 = ("id", "r90", "r180", "r270", "fh", "fv", "tp", "atp")
SCALES = (2, 3)
BUDGET = 20000
DELTA_CAP = 64
E3_VAL_FP_DEFAULT = "results/arc/phase3-branch-e3-learned-ranker/candidate_fingerprints_no_targets_validation.jsonl"


def grid_hash(grid: list[list[int]]) -> str:
    return v1.sha256_text(json.dumps(grid, separators=(",", ":")))


# ============================================================================
# SS1 — perception: views, components, attributes
# ============================================================================
def parse_objects(grid: list[list[int]], view: str) -> list[dict[str, Any]]:
    h, w = len(grid), len(grid[0])
    if view == "cc8":
        nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        nbrs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    same_color = view in ("cc4", "cc8")
    seen = [[False] * w for _ in range(h)]
    objs = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or seen[r][c]:
                continue
            col0 = grid[r][c]
            stack, cells = [(r, c)], {}
            seen[r][c] = True
            while stack:
                y, x = stack.pop()
                cells[(y, x)] = grid[y][x]
                for dy, dx in nbrs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] != 0 \
                            and (not same_color or grid[ny][nx] == col0):
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            objs.append(_attrs(cells, h, w))
    objs.sort(key=lambda o: (o["r0"], o["c0"], o["shape"]))
    return objs


def _attrs(cells: dict[tuple[int, int], int], H: int, W: int) -> dict[str, Any]:
    rs = [r for r, _ in cells]; cs = [c for _, c in cells]
    r0, c0 = min(rs), min(cs)
    h, w = max(rs) - r0 + 1, max(cs) - c0 + 1
    counts = Counter(cells.values())
    color = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    shape = tuple(sorted((r - r0, c - c0) for r, c in cells))
    return {
        "cells": cells, "area": len(cells), "color": color, "n_colors": len(counts),
        "r0": r0, "c0": c0, "h": h, "w": w,
        "centroid": (sum(rs) / len(rs), sum(cs) / len(cs)),
        "shape": shape, "shapeD4": _shape_d4(shape, h, w),
        "touches_border": r0 == 0 or c0 == 0 or r0 + h == H or c0 + w == W,
        "n_holes": _n_holes(shape, h, w),
    }


def _d4_offsets(shape: tuple, h: int, w: int, d: str) -> tuple[tuple, int, int]:
    """Transform offset set; returns (normalized offsets, new h, new w)."""
    if d == "id":
        pts, nh, nw = shape, h, w
    elif d == "r90":
        pts, nh, nw = [(c, h - 1 - r) for r, c in shape], w, h
    elif d == "r180":
        pts, nh, nw = [(h - 1 - r, w - 1 - c) for r, c in shape], h, w
    elif d == "r270":
        pts, nh, nw = [(w - 1 - c, r) for r, c in shape], w, h
    elif d == "fh":
        pts, nh, nw = [(r, w - 1 - c) for r, c in shape], h, w
    elif d == "fv":
        pts, nh, nw = [(h - 1 - r, c) for r, c in shape], h, w
    elif d == "tp":
        pts, nh, nw = [(c, r) for r, c in shape], w, h
    else:                                                  # atp
        pts, nh, nw = [(w - 1 - c, h - 1 - r) for r, c in shape], w, h
    return tuple(sorted(pts)), nh, nw


def _shape_d4(shape: tuple, h: int, w: int) -> tuple:
    return min(_d4_offsets(shape, h, w, d)[0] for d in D4)


def _n_holes(shape: tuple, h: int, w: int) -> int:
    filled = set(shape)
    bg = {(r, c) for r in range(h) for c in range(w) if (r, c) not in filled}
    border = {p for p in bg if p[0] in (0, h - 1) or p[1] in (0, w - 1)}
    reach = set(border); stack = list(border)
    while stack:
        y, x = stack.pop()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (y + dy, x + dx)
            if n in bg and n not in reach:
                reach.add(n); stack.append(n)
    enclosed = bg - reach
    holes = 0; seen: set = set()
    for p in enclosed:
        if p in seen:
            continue
        holes += 1; stack = [p]; seen.add(p)
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (y + dy, x + dx)
                if n in enclosed and n not in seen:
                    seen.add(n); stack.append(n)
    return holes


# ============================================================================
# SS2 — selectors (12, frozen)
# ============================================================================
def select_objects(objs: list[dict], sel: str, kappa: int | None) -> list[dict]:
    if sel == "all":
        return list(objs)
    if sel in ("argmax_area", "argmin_area"):
        if not objs:
            return []
        ext = max(o["area"] for o in objs) if sel == "argmax_area" else min(o["area"] for o in objs)
        return [o for o in objs if o["area"] == ext]
    if sel.startswith("unique_"):
        attr = sel[7:]
        freq = Counter(o[attr] for o in objs)
        return [o for o in objs if freq[o[attr]] == 1]
    if sel in ("modal_shape", "nonmodal_shape"):
        if not objs:
            return []
        freq = Counter(o["shape"] for o in objs)
        best = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        return [o for o in objs if (o["shape"] == best) == (sel == "modal_shape")]
    if sel == "touches_border":
        return [o for o in objs if o["touches_border"]]
    if sel == "not_touches_border":
        return [o for o in objs if not o["touches_border"]]
    if sel == "where_color":
        return [o for o in objs if o["color"] == kappa]
    raise ValueError(sel)


SELECTORS = ("all", "argmax_area", "argmin_area", "unique_color", "unique_shape", "unique_shapeD4",
             "unique_area", "modal_shape", "nonmodal_shape", "touches_border", "not_touches_border",
             "where_color")


# ============================================================================
# SS2 — transforms + rendering
# ============================================================================
def _paint(out: list[list[int]], cells: dict[tuple[int, int], int]) -> None:
    H, W = len(out), len(out[0])
    for (r, c), col in cells.items():
        if 0 <= r < H and 0 <= c < W:
            out[r][c] = col


def _erase(out: list[list[int]], cells: dict[tuple[int, int], int]) -> None:
    H, W = len(out), len(out[0])
    for (r, c) in cells:
        if 0 <= r < H and 0 <= c < W:
            out[r][c] = 0


def _shift(cells: dict, dy: int, dx: int) -> dict:
    return {(r + dy, c + dx): col for (r, c), col in cells.items()}


def _transform_cells(o: dict, t: tuple, out: list[list[int]], objs: list[dict]) -> dict | None:
    """Cells to paint for object o under transform t (gravity reads current canvas occupancy)."""
    kind = t[0]
    if kind in ("identity", "delete"):
        return {} if kind == "delete" else dict(o["cells"])
    if kind == "recolor":
        return {p: t[1] for p in o["cells"]}
    if kind == "recolor_to_largest":
        if not objs:
            return None
        big = sorted(objs, key=lambda x: (-x["area"], x["color"]))[0]
        return {p: big["color"] for p in o["cells"]}
    if kind in ("move", "copy_move"):
        return _shift(o["cells"], t[1][0], t[1][1])
    if kind == "gravity":
        dy, dx = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[t[1]]
        H, W = len(out), len(out[0])
        cells = dict(o["cells"])
        while True:
            nxt = _shift(cells, dy, dx)
            if any(not (0 <= r < H and 0 <= c < W) for (r, c) in nxt):
                break
            if any(out[r][c] != 0 for (r, c) in nxt):
                break
            cells = nxt
        return cells
    if kind == "reflect":
        pts, nh, nw = _d4_offsets(o["shape"], o["h"], o["w"], t[1])
        colmap = {(r - o["r0"], c - o["c0"]): col for (r, c), col in o["cells"].items()}
        src = list(sorted(colmap))
        mapping = dict(zip(sorted(_d4_offsets(tuple(src), o["h"], o["w"], t[1])[0]), [0] * len(src)))
        # rebuild per-cell colors through the same point transform
        outcells = {}
        for (r, c), col in colmap.items():
            tr = _d4_offsets(((r, c),), o["h"], o["w"], t[1])[0][0]
            outcells[(o["r0"] + tr[0], o["c0"] + tr[1])] = col
        _ = pts, nh, nw, mapping
        return outcells
    if kind == "scale":
        k = t[1]
        outcells = {}
        for (r, c), col in o["cells"].items():
            br, bc = o["r0"] + k * (r - o["r0"]), o["c0"] + k * (c - o["c0"])
            for i in range(k):
                for j in range(k):
                    outcells[(br + i, bc + j)] = col
        return outcells
    raise ValueError(kind)


def render(program: dict, grid: list[list[int]], parse_cache: dict) -> list[list[int]] | None:
    view, canvas, sel, t, others, post = (program["view"], program["canvas"], program["select"],
                                          program["transform"], program["others"], program["post"])
    key = (id(grid), view)
    objs = parse_cache.get(key)
    if objs is None:
        objs = parse_objects(grid, view)
        parse_cache[key] = objs
    selected = select_objects(objs, sel, program.get("kappa_sel"))
    sel_ids = {id(o) for o in selected}
    nonsel = [o for o in objs if id(o) not in sel_ids]
    H, W = len(grid), len(grid[0])

    if canvas == "input_copy":
        out = [row[:] for row in grid]
        if t[0] != "copy_move":
            for o in selected:
                _erase(out, o["cells"])
        if others == "delete":
            for o in nonsel:
                _erase(out, o["cells"])
    elif canvas == "blank_like_input":
        out = [[0] * W for _ in range(H)]
        if others == "keep":
            for o in nonsel:
                _paint(out, o["cells"])
    elif canvas == "blank_solved_dims":
        dh, dw = program["dims_fn"](H, W)
        out = [[0] * dw for _ in range(dh)]
        if others == "keep":
            for o in nonsel:
                _paint(out, o["cells"])
    elif canvas == "selection_bbox_crop":
        out = [row[:] for row in grid]
        if t[0] != "copy_move":
            for o in selected:
                _erase(out, o["cells"])
        if others == "delete":
            for o in nonsel:
                _erase(out, o["cells"])
    else:
        raise ValueError(canvas)

    if t[0] == "gravity":
        order_key = {"down": lambda o: (-(o["r0"] + o["h"]), o["c0"]), "up": lambda o: (o["r0"], o["c0"]),
                     "left": lambda o: (o["c0"], o["r0"]), "right": lambda o: (-(o["c0"] + o["w"]), o["r0"])}[t[1]]
        ordered = sorted(selected, key=order_key)
    else:
        ordered = selected
    painted_all: list[dict] = []
    for o in ordered:
        cells = _transform_cells(o, t, out, objs)
        if cells is None:
            return None
        _paint(out, cells)
        painted_all.append(cells)

    if canvas == "selection_bbox_crop":
        pts = [p for cells in painted_all for p in cells]
        if not pts:
            return None
        r0 = min(r for r, _ in pts); r1 = max(r for r, _ in pts)
        c0 = min(c for _, c in pts); c1 = max(c for _, c in pts)
        r0 = max(0, r0); c0 = max(0, c0); r1 = min(len(out) - 1, r1); c1 = min(len(out[0]) - 1, c1)
        out = [row[c0:c1 + 1] for row in out[r0:r1 + 1]]
    if program["post"] == "crop_nonzero_bbox":
        pts = [(r, c) for r, row in enumerate(out) for c, v in enumerate(row) if v != 0]
        if not pts:
            return None
        r0 = min(r for r, _ in pts); r1 = max(r for r, _ in pts)
        c0 = min(c for _, c in pts); c1 = max(c for _, c in pts)
        out = [row[c0:c1 + 1] for row in out[r0:r1 + 1]]
    _ = post
    return out if out and out[0] else None


# ============================================================================
# CEGIS hole domains (conditioning only)
# ============================================================================
def dims_candidates(cond: list[dict]) -> list:
    """Callable dims solvers: constant conditioning-output dims, or constant integer ratio != (1,1)."""
    outs = [(len(p["output"]), len(p["output"][0])) for p in cond]
    ins = [(len(p["input"]), len(p["input"][0])) for p in cond]
    cands = []
    if len(set(outs)) == 1:
        H, W = outs[0]
        cands.append(("const", lambda h, w, H=H, W=W: (H, W)))
    ratios = set()
    for (ih, iw), (oh, ow) in zip(ins, outs):
        if ih and iw and oh % ih == 0 and ow % iw == 0:
            ratios.add((oh // ih, ow // iw))
        else:
            ratios.add(None)
    if len(ratios) == 1 and None not in ratios:
        a, b = next(iter(ratios))
        if (a, b) != (1, 1):
            cands.append((f"ratio{a}x{b}", lambda h, w, a=a, b=b: (a * h, b * w)))
    return cands


def delta_candidates(cond: list[dict], view: str, sel: str, kappa_sel: int | None,
                     parse_cache: dict) -> list[tuple[int, int]]:
    """Move-vector candidates: same-shape same-cell-color component matches of the selected
    objects of input0 inside output0 (frozen SS2 CEGIS rule; cap DELTA_CAP)."""
    if not cond:
        return []
    g_in, g_out = cond[0]["input"], cond[0]["output"]
    key = (id(g_in), view)
    objs = parse_cache.get(key) or parse_objects(g_in, view)
    parse_cache[key] = objs
    selected = select_objects(objs, sel, kappa_sel)[:4]
    out_objs = parse_objects(g_out, view)
    cands = []
    for o in selected:
        pattern = {(r - o["r0"], c - o["c0"]): col for (r, c), col in o["cells"].items()}
        for m in out_objs:
            if m["shape"] != o["shape"]:
                continue
            mpat = {(r - m["r0"], c - m["c0"]): col for (r, c), col in m["cells"].items()}
            if mpat == pattern:
                cands.append((m["r0"] - o["r0"], m["c0"] - o["c0"]))
    uniq = sorted(set(cands))
    return uniq[:DELTA_CAP]


def color_domain(cond: list[dict], which: str) -> list[int]:
    cols: set[int] = set()
    for p in cond:
        for row in p[which]:
            cols.update(row)
    cols.discard(0)
    return sorted(cols)


# ============================================================================
# Enumeration + admission (budget counts FULL verifications)
# ============================================================================
def program_id(p: dict) -> str:
    bits = [p["view"], p["canvas"], p["select"]]
    if p.get("kappa_sel") is not None:
        bits[-1] += f"={p['kappa_sel']}"
    t = p["transform"]
    bits.append(t[0] + ("" if len(t) == 1 else f"({t[1]})"))
    bits += [p["others"], p["post"]]
    if p.get("dims_name"):
        bits.append(p["dims_name"])
    return "|".join(str(b) for b in bits)


def verify(p: dict, cond: list[dict], parse_cache: dict) -> bool:
    for pair in cond:
        if render(p, pair["input"], parse_cache) != pair["output"]:
            return False
    return True


def enumerate_admitted(cond: list[dict], query_input: list[list[int]]) -> tuple[list[dict], int, bool]:
    """Returns (unique candidate outputs [{grid, grid_hash, program_id, n_programs}], n_verifs, exhausted)."""
    parse_cache: dict = {}
    out_colors = None
    in_colors = None
    verifs = 0
    exhausted = False
    by_hash: dict[str, dict] = {}

    def admit_check(p: dict) -> None:
        nonlocal verifs, exhausted
        if exhausted:
            return
        if verifs >= BUDGET:
            exhausted = True
            return
        verifs += 1
        if verify(p, cond, parse_cache):
            g = render(p, query_input, parse_cache)
            if g is None:
                return
            hsh = grid_hash(g)
            rec = by_hash.get(hsh)
            if rec is None:
                by_hash[hsh] = {"grid": g, "grid_hash": hsh, "program_id": program_id(p), "n_programs": 1}
            else:
                rec["n_programs"] += 1

    dimcands = dims_candidates(cond)
    for view in VIEWS:
        for sel in SELECTORS:
            kappas_sel = [None]
            if sel == "where_color":
                if in_colors is None:
                    in_colors = color_domain(cond, "input")
                kappas_sel = in_colors or []
            for kappa_sel in kappas_sel:
                transforms: list[tuple] = [("identity",), ("delete",), ("recolor_to_largest",)]
                transforms += [("gravity", d) for d in GRAVITY_DIRS]
                transforms += [("reflect", d) for d in D4 if d != "id"]
                transforms += [("scale", k) for k in SCALES]
                if out_colors is None:
                    out_colors = color_domain(cond, "output")
                transforms += [("recolor", k) for k in out_colors]
                deltas = delta_candidates(cond, view, sel, kappa_sel, parse_cache)
                transforms += [("move", d) for d in deltas] + [("copy_move", d) for d in deltas]
                for t in transforms:
                    for canvas in ("input_copy", "blank_like_input", "selection_bbox_crop"):
                        if t[0] == "delete" and canvas == "selection_bbox_crop":
                            continue
                        for others in ("keep", "delete"):
                            for post in ("identity", "crop_nonzero_bbox"):
                                admit_check({"view": view, "canvas": canvas, "select": sel,
                                             "kappa_sel": kappa_sel, "transform": t, "others": others,
                                             "post": post})
                    for dname, dfn in dimcands:
                        for others in ("keep", "delete"):
                            for post in ("identity", "crop_nonzero_bbox"):
                                admit_check({"view": view, "canvas": "blank_solved_dims", "select": sel,
                                             "kappa_sel": kappa_sel, "transform": t, "others": others,
                                             "post": post, "dims_fn": dfn, "dims_name": dname})
    cands = sorted(by_hash.values(), key=lambda r: r["grid_hash"])
    return cands, verifs, exhausted


# ============================================================================
# The probe (SS3): barrier -> ceilings -> gates
# ============================================================================
def run_probe(args: argparse.Namespace) -> int:
    out_dir = Path(args.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve(); register = Path(args.register).resolve()
    v1.assert_training_data_dir(data_dir)
    t0 = time.time()
    tasks, register_hash, data_hash = v1.load_tasks(data_dir, register, args.split_mode)
    validation = [t for t in tasks if t.split == "validation"]
    if args.limit_tasks > 0:
        validation = validation[: args.limit_tasks]
    lanes = {"validation_lodo": v1.build_lodo_instances(validation, "validation_lodo"),
             "validation_pttest": v1.build_pttest_instances(validation, "validation_pttest")}
    insts = lanes["validation_lodo"] + lanes["validation_pttest"]
    v1.write_csv(out_dir / "split.csv",
                 [{"task_id": t.task_id, "primary_prior": t.primary_prior, "split": t.split}
                  for t in sorted(validation, key=lambda x: x.task_id)],
                 ["task_id", "primary_prior", "split"])

    # ---- generate candidates (conditioning only; targets untouched) ----
    rows, fp_rows = [], []
    for i, inst in enumerate(insts):
        cands, verifs, exhausted = enumerate_admitted(inst.conditioning, inst.query_input)
        rows.append({"instance_id": inst.instance_id, "task_id": inst.task_id, "lane": inst.lane,
                     "primary_prior": inst.primary_prior, "n_candidates": len(cands),
                     "n_verifications": verifs, "budget_exhausted": exhausted,
                     "candidates": cands})
        fp_rows.append({"instance_id": inst.instance_id, "task_id": inst.task_id, "lane": inst.lane,
                        "n_conditioning": len(inst.conditioning), "n_candidates": len(cands),
                        "budget_exhausted": exhausted,
                        "candidates": [{"program_id": c["program_id"], "grid_hash": c["grid_hash"]}
                                       for c in cands]})
        if args.progress and (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(insts)} instances ({time.time() - t0:.0f}s)", flush=True)
    v1.write_jsonl(out_dir / "candidates_by_instance.jsonl", rows)
    barrier_path = out_dir / "candidate_fingerprints_no_targets.jsonl"
    v1.write_jsonl(barrier_path, fp_rows)
    barrier_hash = v1.sha256_file(barrier_path)
    (out_dir / "candidate_fingerprints_no_targets.sha256").write_text(barrier_hash + "\n", encoding="utf-8")

    # ---- barrier written: NOW read targets ----
    target_hash = {inst.instance_id: grid_hash(inst.target_output) for inst in insts}
    gen1_solvable = {r["instance_id"]: any(c["grid_hash"] == target_hash[r["instance_id"]]
                                           for c in r["candidates"]) for r in rows}

    # ---- v2 baseline: OFFLINE from the E3 validation fingerprint file ----
    e3_fp = Path(args.e3_fingerprints).resolve()
    e3_rows = [json.loads(l) for l in e3_fp.read_text(encoding="utf-8").splitlines() if l.strip()]
    e3_hash = v1.sha256_file(e3_fp)
    e3_by_id = {r["instance_id"]: r for r in e3_rows}
    v2_solvable = {}
    for inst in insts:
        r = e3_by_id.get(inst.instance_id)
        v2_solvable[inst.instance_id] = bool(r) and any(
            c["grid_hash"] == target_hash[inst.instance_id] for c in r["candidates"])
    missing = [i.instance_id for i in insts if i.instance_id not in e3_by_id]

    # ---- ceilings ----
    def ceiling(solv: dict) -> dict:
        per_lane = {}
        for lane, li in lanes.items():
            ids = [i.instance_id for i in li]
            tasks_solved = {i.task_id for i in li if solv[i.instance_id]}
            per_lane[lane] = {"instances": len(ids), "solvable_instances": sum(solv[x] for x in ids),
                              "solvable_tasks": len(tasks_solved), "task_ids": sorted(tasks_solved)}
        pooled = {i.task_id for i in insts if solv[i.instance_id]}
        per_lane["pooled"] = {"instances": len(insts),
                              "solvable_instances": sum(solv[i.instance_id] for i in insts),
                              "solvable_tasks": len(pooled), "task_ids": sorted(pooled)}
        return per_lane

    gen1_c, v2_c = ceiling(gen1_solvable), ceiling(v2_solvable)
    g1, g2 = gen1_c["pooled"]["solvable_tasks"], v2_c["pooled"]["solvable_tasks"]
    if g1 >= max(2 * g2, g2 + 3):
        gate = "GEN1_CEILING_LIFT"
    elif g1 <= g2 + 1:
        gate = "GEN1_CEILING_EMPTY"
    else:
        gate = "GEN1_CEILING_MARGINAL"

    no_admit = sum(1 for r in rows if r["n_candidates"] == 0)
    summary_rows = []
    for lane in ("validation_lodo", "validation_pttest", "pooled"):
        summary_rows.append({"lane": lane, "arm": "gen1_v0", **{k: v for k, v in gen1_c[lane].items() if k != "task_ids"}})
        summary_rows.append({"lane": lane, "arm": "v2_baseline", **{k: v for k, v in v2_c[lane].items() if k != "task_ids"}})
    v1.write_csv(out_dir / "ceiling_summary.csv", summary_rows,
                 ["lane", "arm", "instances", "solvable_instances", "solvable_tasks"])
    v1.write_csv(out_dir / "v2_baseline_ceiling.csv",
                 [{"lane": ln, **{k: v for k, v in v2_c[ln].items() if k != "task_ids"}} for ln in v2_c],
                 ["lane", "instances", "solvable_instances", "solvable_tasks"])
    prior_rows = []
    for prior in sorted({i.primary_prior for i in insts}):
        li = [i for i in insts if i.primary_prior == prior]
        prior_rows.append({"primary_prior": prior, "instances": len(li),
                           "gen1_solvable": sum(gen1_solvable[i.instance_id] for i in li),
                           "v2_solvable": sum(v2_solvable[i.instance_id] for i in li)})
    v1.write_csv(out_dir / "per_prior_ceiling.csv", prior_rows,
                 ["primary_prior", "instances", "gen1_solvable", "v2_solvable"])

    spec_path = Path(__file__).resolve().parent / "GEN1_OBJECT_DSL_V0_PROBE_SPEC.md"
    manifest = {
        "generatedAt": v1.iso_now(), "specSha256": v1.sha256_file(spec_path),
        "runnerSha256": v1.sha256_file(Path(__file__).resolve()),
        "registerHash": register_hash, "dataDirHash": data_hash, "splitMode": args.split_mode,
        "budget": BUDGET, "views": list(VIEWS), "nInstances": len(insts),
        "barrierSha256": barrier_hash, "e3FingerprintSha256": e3_hash,
        "e3MissingInstances": missing, "noAdmittedRate": round(no_admit / max(1, len(rows)), 4),
        "gate": gate, "gen1PooledTasks": g1, "v2PooledTasks": g2,
        "limitTasks": args.limit_tasks, "wallSeconds": round(time.time() - t0, 1),
        "git": v1.git_state(Path(__file__).resolve().parents[3], args.allow_dirty),
    }
    v1.write_json(out_dir / "manifest.json", manifest)
    v1.write_json(out_dir / "gen1_probe_receipt.json",
                  {"manifest": manifest, "gate": gate, "gen1": gen1_c, "v2": v2_c})
    (out_dir / "probe_adjudication.md").write_text(
        f"# GEN-1 v0 ceiling probe\n\nGate: **{gate}**\n\n"
        f"- gen1 pooled solvable tasks: {g1} ({sorted(gen1_c['pooled']['task_ids'])})\n"
        f"- v2 baseline pooled solvable tasks: {g2} ({sorted(v2_c['pooled']['task_ids'])})\n"
        f"- no-admitted rate (gen1): {no_admit}/{len(rows)}\n"
        f"- limit_tasks: {args.limit_tasks} (0 = full validation)\n", encoding="utf-8")
    v1.write_json(out_dir / "hashes.json", {"barrier": barrier_hash, "e3_fingerprints": e3_hash})
    (out_dir / "commands.md").write_text("```\n" + " ".join(sys.argv) + "\n```\n", encoding="utf-8")
    print(f"GEN-1 probe wrote {out_dir}\nGate: {gate}  (gen1={g1} tasks vs v2={g2} tasks pooled; "
          f"no-admitted {no_admit}/{len(rows)}; {time.time() - t0:.0f}s)")
    return 0


# ============================================================================
# Self-test: solver-correctness synthetics (known v0-solvable tasks)
# ============================================================================
def _mk(h, w, rects):
    g = [[0] * w for _ in range(h)]
    for (r, c, hh, ww, col) in rects:
        for i in range(hh):
            for j in range(ww):
                g[r + i][c + j] = col
    return g


def self_test() -> int:
    ok = True

    def check(name, cond, query, target):
        nonlocal ok
        cands, verifs, exhausted = enumerate_admitted(cond, query)
        th = grid_hash(target)
        hit = any(c["grid_hash"] == th for c in cands)
        winner = next((c["program_id"] for c in cands if c["grid_hash"] == th), "-")
        print(f"  [{'PASS' if hit else 'FAIL'}] {name}: target-in-bank={hit} "
              f"(cands={len(cands)}, verifs={verifs}, exhausted={exhausted}) {winner}")
        ok = ok and hit
        return hit

    def check_neg(name, cond, query, target):
        nonlocal ok
        cands, _, _ = enumerate_admitted(cond, query)
        th = grid_hash(target)
        hit = any(c["grid_hash"] == th for c in cands)
        print(f"  [{'PASS' if not hit else 'FAIL'}] {name}: target-in-bank={hit} (want False)")
        ok = ok and not hit
        return not hit

    # 1. extract largest component (be94b721 analog)
    def crop(g, r, c, h, w):
        return [row[c:c + w] for row in g[r:r + h]]
    cond, q = [], None
    layouts = [((1, 1, 2, 2, 3), (5, 5, 3, 4, 2)), ((0, 6, 2, 3, 4), (4, 1, 4, 4, 1)),
               ((2, 2, 1, 2, 5), (6, 3, 3, 5, 3))]
    for a, b in layouts:
        g = _mk(10, 10, [a, b])
        big = b if b[2] * b[3] > a[2] * a[3] else a
        cond.append({"input": g, "output": crop(g, big[0], big[1], big[2], big[3])})
    qg = _mk(10, 10, [(0, 0, 2, 2, 6), (5, 2, 4, 5, 7)])
    check("extract-largest", cond, qg, crop(qg, 5, 2, 4, 5))

    # 2. recolor the unique-shape object to color 4 (CEGIS kappa)
    cond = []
    for base in [(0, 0), (1, 1), (2, 0)]:
        g = _mk(9, 9, [(base[0], base[1], 2, 2, 3), (base[0] + 4, base[1], 2, 2, 3),
                       (base[0] + 1, base[1] + 5, 1, 3, 3)])
        out = _mk(9, 9, [(base[0], base[1], 2, 2, 3), (base[0] + 4, base[1], 2, 2, 3),
                         (base[0] + 1, base[1] + 5, 1, 3, 4)])
        cond.append({"input": g, "output": out})
    qg = _mk(9, 9, [(3, 0, 2, 2, 3), (6, 6, 2, 2, 3), (0, 4, 1, 3, 3)])
    qt = _mk(9, 9, [(3, 0, 2, 2, 3), (6, 6, 2, 2, 3), (0, 4, 1, 3, 4)])
    check("recolor-unique-shape(kappa)", cond, qg, qt)

    # 3. gravity down (two objects, separate columns)
    cond = []
    for (r1, c1, r2, c2) in [(1, 1, 2, 5), (0, 2, 3, 6), (2, 0, 1, 4)]:
        g = _mk(8, 8, [(r1, c1, 2, 2, 2), (r2, c2, 1, 2, 3)])
        out = _mk(8, 8, [(6, c1, 2, 2, 2), (7, c2, 1, 2, 3)])
        cond.append({"input": g, "output": out})
    qg = _mk(8, 8, [(0, 0, 2, 2, 2), (1, 5, 1, 2, 3)])
    qt = _mk(8, 8, [(6, 0, 2, 2, 2), (7, 5, 1, 2, 3)])
    check("gravity-down", cond, qg, qt)

    # 4. move by constant delta (2,1) (CEGIS delta)
    cond = []
    for (r, c) in [(1, 1), (2, 3), (0, 2)]:
        g = _mk(9, 9, [(r, c, 2, 2, 5)])
        out = _mk(9, 9, [(r + 2, c + 1, 2, 2, 5)])
        cond.append({"input": g, "output": out})
    qg = _mk(9, 9, [(3, 4, 2, 2, 5)])
    qt = _mk(9, 9, [(5, 5, 2, 2, 5)])
    check("move-delta(2,1)", cond, qg, qt)

    # 5. delete the smallest object
    cond = []
    for (a, b) in [((1, 1, 3, 3, 2), (6, 6, 1, 1, 4)), ((0, 4, 2, 4, 1), (5, 0, 1, 2, 7)),
                   ((4, 4, 3, 2, 6), (0, 0, 1, 1, 8))]:
        g = _mk(9, 9, [a, b])
        out = _mk(9, 9, [a])
        cond.append({"input": g, "output": out})
    qg = _mk(9, 9, [(2, 2, 3, 4, 9), (7, 1, 1, 1, 3)])
    qt = _mk(9, 9, [(2, 2, 3, 4, 9)])
    check("delete-smallest", cond, qg, qt)

    # 6. negative control: latent rule outside the grammar (no false universal)
    import random as _rnd
    _rnd.seed(7)
    cond = []
    for _ in range(3):
        g = [[_rnd.randint(0, 4) for _ in range(6)] for _ in range(6)]
        out = [[_rnd.randint(0, 4) for _ in range(6)] for _ in range(6)]
        cond.append({"input": g, "output": out})
    qg = [[_rnd.randint(0, 4) for _ in range(6)] for _ in range(6)]
    qt = [[_rnd.randint(0, 4) for _ in range(6)] for _ in range(6)]
    check_neg("random-noise (negative)", cond, qg, qt)

    print(f"self-test: {'ALL PASS' if ok else 'FAILURES PRESENT'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--register", default=None)
    ap.add_argument("--split-mode", choices=["frozen_v2", "sha256_expansion"], default="sha256_expansion")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--e3-fingerprints", default=E3_VAL_FP_DEFAULT)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if not (args.data_dir and args.register and args.out):
        print("Need --data-dir, --register, --out (or --self-test).")
        return 2
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
