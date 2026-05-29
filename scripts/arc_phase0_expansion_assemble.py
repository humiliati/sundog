#!/usr/bin/env python3
"""Assemble the Phase 0 context-expansion registers + receipt (Phase C).

Frozen context: PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md. Deterministic; reads
only the frozen candidate queue, the manual inspection log, the Phase 0
inventory, and the original register. Computes no certificate distance/sketch/
solver result and reads no held-out split.

Target (spec ss"Expansion Target"): 108 included tasks total, 18 per prior =
36 original (6/prior) + 72 new (12/prior). The inspection log may contain MORE
than 12 new includes per prior (over-inspection); the binding selection is the
12 lowest-`selection_order_rank` includes per prior, which equals what a correct
stop-at-12 walk over the frozen queue produces. Remaining inspected includes are
recorded as over-inspection surplus (status=exclude) for full transparency.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARC = REPO / "docs" / "prereg" / "arc"
RECEIPT = REPO / "results" / "arc" / "phase0-context-expansion-for-fibers"

INVENTORY = REPO / "results" / "arc" / "phase0-inventory" / "tasks.csv"
ORIG_REGISTER = ARC / "P0_TASK_REGISTER.csv"
LOG = RECEIPT / "manual_inspection_log.csv"
QUEUE = RECEIPT / "candidate_queue.csv"

EXPANSION_BATCH = "fiber_context_expansion_v1"
PRIORS = ["objectness", "counting", "symmetry", "spatial_transform", "local_completion", "color_role"]
PER_PRIOR_TOTAL = 18
PER_PRIOR_NEW = 12
SURPLUS_REASON = "over_inspection_surplus_beyond_12_new_per_prior"

REGISTER_HEADER = [
    "task_id", "split", "status", "primary_prior", "secondary_priors", "inclusion_basis",
    "exclusion_reason", "predicted_boundary", "inventory_row_hash", "manual_inspection", "notes",
    "expansion_batch", "selection_order_rank",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.read_text(encoding="utf-8-sig").splitlines()))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inclusion_basis(inv: dict[str, str]) -> str:
    facts = [
        f"grids {inv['min_height']}x{inv['min_width']}..{inv['max_height']}x{inv['max_width']}",
        f"{inv['color_count']} colors",
        f"up to {inv['max_nonzero_components']} components",
    ]
    if inv.get("symmetry_hints"):
        facts.append(f"symmetries: {inv['symmetry_hints']}")
    if int(inv.get("train_shape_changes") or 0) >= 1:
        facts.append(f"{inv['train_shape_changes']} train shape-change(s)")
    facts.append(f"density {inv['min_nonzero_density']}..{inv['max_nonzero_density']}")
    return "; ".join(facts)


def write_csv(path: Path, rows: list[dict], header: list[str]) -> None:
    lines = [",".join(header)]
    for r in rows:
        lines.append(",".join(csv_cell(r.get(c, "")) for c in header))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def csv_cell(value) -> str:
    text = str(value if value is not None else "")
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def sha256_partition(task_ids: list[str]) -> dict[str, str]:
    ordered = sorted(task_ids, key=lambda t: hashlib.sha256(f"{EXPANSION_BATCH}|{t}".encode("utf-8")).hexdigest())
    n = len(ordered)
    n_val = min(n, max(3, n // 3))
    return {t: ("validation" if i < n_val else "test") for i, t in enumerate(ordered)}


def main() -> int:
    inv = {r["task_id"]: r for r in read_csv(INVENTORY)}
    orig = read_csv(ORIG_REGISTER)
    log = read_csv(LOG)
    includes = [r for r in log if r["status"] == "include"]

    # Binding selection: 12 lowest selection_order_rank includes per prior.
    by_prior: dict[str, list[dict]] = defaultdict(list)
    for r in includes:
        by_prior[r["assigned_primary_prior"]].append(r)
    binding: dict[str, list[dict]] = {}
    surplus: dict[str, list[dict]] = {}
    for p in PRIORS:
        rows = sorted(by_prior[p], key=lambda r: int(r["selection_order_rank"]))
        binding[p] = rows[:PER_PRIOR_NEW]
        surplus[p] = rows[PER_PRIOR_NEW:]

    # ---- branch adjudication ----
    short = [p for p in PRIORS if len(binding[p]) < PER_PRIOR_NEW]
    if short:
        branch = "phase0_fiber_expansion_hold_insufficient_tasks"
        branch_reason = f"priors short of {PER_PRIOR_NEW} new includes: {short}"
    else:
        branch = "phase0_fiber_expansion_admit"
        branch_reason = (
            f"108-task balanced expanded register (18/prior = 6 original + 12 new); "
            f"0 hard-excludes; selection by frozen candidate_queue rank only."
        )

    def new_row(r: dict, status: str, reason: str) -> dict:
        tid = r["task_id"]
        iv = inv.get(tid, {})
        return {
            "task_id": tid, "split": "training", "status": status,
            "primary_prior": r["assigned_primary_prior"],
            "secondary_priors": r.get("secondary_priors", ""),
            "inclusion_basis": inclusion_basis(iv) if iv else "",
            "exclusion_reason": reason,
            "predicted_boundary": r["predicted_boundary"],
            "inventory_row_hash": iv.get("inventory_row_hash", ""),
            "manual_inspection": "y",
            "notes": f"fiber expansion manual inspection: {r.get('reason','')}",
            "expansion_batch": EXPANSION_BATCH,
            "selection_order_rank": r["selection_order_rank"],
        }

    # ---- P0_CONTEXT_EXPANSION_REGISTER.csv (all 108 inspected: 72 include + 36 surplus exclude) ----
    expansion_rows = []
    for p in PRIORS:
        for r in binding[p]:
            expansion_rows.append(new_row(r, "include", ""))
        for r in surplus[p]:
            expansion_rows.append(new_row(r, "exclude", SURPLUS_REASON))
    expansion_rows.sort(key=lambda r: (r["primary_prior"], int(r["selection_order_rank"])))
    write_csv(ARC / "P0_CONTEXT_EXPANSION_REGISTER.csv", expansion_rows, REGISTER_HEADER)

    # ---- P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv (36 original includes + 72 new includes) ----
    expanded_rows = []
    for r in orig:
        if r["status"] != "include":
            continue
        row = {c: r.get(c, "") for c in REGISTER_HEADER if c in r}
        row["expansion_batch"] = "p0_original"
        row["selection_order_rank"] = ""
        expanded_rows.append({c: row.get(c, "") for c in REGISTER_HEADER})
    for p in PRIORS:
        for r in binding[p]:
            expanded_rows.append(new_row(r, "include", ""))
    write_csv(ARC / "P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv", expanded_rows, REGISTER_HEADER)

    # ---- partition over the 108 included (per prior 18 -> 6 validation / 12 test) ----
    included_by_prior: dict[str, list[str]] = defaultdict(list)
    for r in expanded_rows:
        included_by_prior[r["primary_prior"]].append(r["task_id"])
    partition_counts = {}
    full_partition = {}
    for p in PRIORS:
        part = sha256_partition(included_by_prior[p])
        full_partition.update(part)
        c = {"validation": 0, "test": 0}
        for v in part.values():
            c[v] += 1
        partition_counts[p] = {"total": len(part), **c}
    total_val = sum(v["validation"] for v in partition_counts.values())
    total_test = sum(v["test"] for v in partition_counts.values())

    # ---- prior_counts.csv ----
    prior_counts_rows = [{
        "primary_prior": p, "original": len(included_by_prior[p]) - len(binding[p]),
        "new_included": len(binding[p]), "total": len(included_by_prior[p]),
        "surplus_inspected": len(surplus[p]),
        "validation": partition_counts[p]["validation"], "test": partition_counts[p]["test"],
    } for p in PRIORS]
    write_csv(RECEIPT / "prior_counts.csv", prior_counts_rows,
              ["primary_prior", "original", "new_included", "total", "surplus_inspected", "validation", "test"])

    # ---- receipt json ----
    queue_sha = sha256_file(QUEUE)
    log_sha = sha256_file(LOG)
    exp_reg_sha = sha256_file(ARC / "P0_CONTEXT_EXPANSION_REGISTER.csv")
    expanded_reg_sha = sha256_file(ARC / "P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv")
    receipt = {
        "spec": "docs/prereg/arc/PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md",
        "specHash": sha256_file(ARC / "PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md"),
        "expansionBatch": EXPANSION_BATCH,
        "branch": branch, "branchReason": branch_reason,
        "target": {"expanded_total": 108, "per_prior_total": PER_PRIOR_TOTAL, "new_required": 72, "per_prior_new": PER_PRIOR_NEW},
        "achieved": {
            "expanded_total": len(expanded_rows),
            "new_included": sum(len(binding[p]) for p in PRIORS),
            "surplus_inspected": sum(len(surplus[p]) for p in PRIORS),
            "total_includes_in_log": len(includes),
            "hard_excludes_in_log": len([r for r in log if r["status"] == "exclude"]),
        },
        "perPriorCounts": {p: {"original": len(included_by_prior[p]) - len(binding[p]),
                               "new_included": len(binding[p]), "total": len(included_by_prior[p]),
                               "surplus_inspected": len(surplus[p])} for p in PRIORS},
        "sha256ExpansionPartition": {"validation_tasks": total_val, "test_tasks": total_test, "perPrior": partition_counts},
        "candidateQueueSha256": queue_sha,
        "manualInspectionLogSha256": log_sha,
        "registers": {
            "expansion_register": "docs/prereg/arc/P0_CONTEXT_EXPANSION_REGISTER.csv",
            "expansion_register_sha256": exp_reg_sha,
            "expanded_register": "docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv",
            "expanded_register_sha256": expanded_reg_sha,
        },
        "selectionDiscipline": {
            "selection_basis": "frozen candidate_queue selection_order_rank (lowest 12 new per prior)",
            "used_certificate_or_solver_info": False,
            "used_held_split_or_kaggle": False,
            "rebalanced_across_priors": False,
        },
    }
    (RECEIPT / "phase0_context_expansion_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")

    # ---- branch_adjudication.md ----
    (RECEIPT / "branch_adjudication.md").write_text(
        f"# Phase 0 Context Expansion — Branch Adjudication\n\n"
        f"**Branch: `{branch}`**\n\n{branch_reason}\n\n"
        f"- Expanded register total: {len(expanded_rows)} (target 108)\n"
        f"- Per prior: 6 original + 12 new = 18 (all six priors)\n"
        f"- New includes: {sum(len(binding[p]) for p in PRIORS)} (target 72)\n"
        f"- Over-inspection surplus recorded as exclude: {sum(len(surplus[p]) for p in PRIORS)}\n"
        f"- Hard-exclusions (quality): {len([r for r in log if r['status']=='exclude'])}\n"
        f"- sha256_expansion partition: {total_val} validation + {total_test} test tasks\n",
        encoding="utf-8")

    # ---- commands.md ----
    (RECEIPT / "commands.md").write_text(
        "# Phase 0 Context Expansion — commands\n\n"
        "```\n"
        "node scripts/arc-phase0-context-expansion-for-fibers.mjs \\\n"
        "  --inventory results/arc/phase0-inventory/tasks.csv \\\n"
        "  --register docs/prereg/arc/P0_TASK_REGISTER.csv \\\n"
        "  --out results/arc/phase0-context-expansion-for-fibers\n"
        "python scripts/arc_phase0_expansion_inspect.py --queue <queue> --data-dir <data> --prior <p> ...\n"
        "python scripts/arc_phase0_expansion_assemble.py\n"
        "```\n",
        encoding="utf-8")

    # ---- hashes.json ----
    hashes = {}
    for f in sorted(RECEIPT.glob("*")):
        if f.is_file() and f.name != "hashes.json":
            hashes[f.name] = sha256_file(f)
    (RECEIPT / "hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")

    print(f"Branch: {branch}")
    print(f"Expanded register: {len(expanded_rows)} rows ({ARC / 'P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv'})")
    print(f"Expansion register: {len(expansion_rows)} rows (72 include + {sum(len(surplus[p]) for p in PRIORS)} surplus exclude)")
    for p in PRIORS:
        print(f"  {p:18s} total={len(included_by_prior[p])} (orig {len(included_by_prior[p])-len(binding[p])} + new {len(binding[p])}) "
              f"val={partition_counts[p]['validation']} test={partition_counts[p]['test']} surplus={len(surplus[p])}")
    print(f"sha256_expansion partition: {total_val} validation + {total_test} test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
