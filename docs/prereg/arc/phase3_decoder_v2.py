#!/usr/bin/env python
"""ARC Phase 3 strengthened full-grid control gate.

This runner is intentionally narrower than phase3_decoder.py: it tests whether
a stronger/data-richer full-grid lane can clear the control gate before any
renewed signature-vs-full-grid comparison is attempted.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

import phase3_decoder as p3


LEARNER_VERSION = "blackwell_publictrain_rawgrid_gate_v2"
PROTOCOL_VERSION = "arc-p3-rawgrid-gate-v2"
RECEIPT_SCHEMA_VERSION = "arc-p3-rawgrid-gate-receipt-v1"
ARM = "raw_grid_lowcap"
SEED_SLATE = [20260528, 20260529, 20260530]

MODEL_SPEC_V2 = {
    **p3.MODEL_SPEC,
    "layers": 4,
    "d_model": 192,
    "heads": 6,
    "feedforward_dim": 768,
    "dropout": 0.10,
    "optimizer": "AdamW",
    "adamw_betas": [0.9, 0.95],
    "adamw_eps": 1e-8,
    "weight_decay": 0.01,
    "learning_rate": 2e-4,
    "lr_schedule": "constant",
    "batch_size": 24,
    "max_epochs": 120,
    "early_stop_patience": 20,
    "seed_slate": SEED_SLATE,
    "data_policy": "all ARC public-training tasks except frozen validation/test register tasks",
}


def main() -> int:
    args = parse_args()
    patch_v1_globals()
    if args.merge:
        return run_merge(args)
    started_at = p3.iso_now()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    p3.set_global_determinism(args.master_seed)
    git = p3.git_state(repo_root, args.allow_dirty)
    data_dir = Path(args.data_dir).resolve()
    register_path = Path(args.register).resolve()
    p3.assert_training_data_dir(data_dir)
    tasks, register_hash, data_hash = load_public_training_tasks(data_dir, register_path, args.limit_aux_tasks)
    split_rows = p3.build_split_rows([task for task in tasks if task.task_id in p3.expected_split_by_task()])
    write_split_with_aux(out_dir / "split.csv", tasks, split_rows)

    train_tasks = [task for task in tasks if task.split == "train"]
    validation_tasks = [task for task in tasks if task.split == "validation"]
    test_tasks = [task for task in tasks if task.split == "test"]
    train_instances = [
        *p3.build_lodo_instances(train_tasks, "train_lodo"),
        *build_pttest_if_available(train_tasks, "train_pttest"),
    ]
    validation_instances = [
        *p3.build_lodo_instances(validation_tasks, "validation_lodo"),
        *build_pttest_if_available(validation_tasks, "validation_pttest"),
    ]
    test_lodo_instances = p3.build_lodo_instances(test_tasks, "test_lodo")
    pttest_instances = p3.build_pttest_instances(test_tasks)

    manifest = base_manifest(
        args=args,
        started_at=started_at,
        git=git,
        data_dir=data_dir,
        register_path=register_path,
        out_dir=out_dir,
        register_hash=register_hash,
        data_hash=data_hash,
        task_count=len(tasks),
        train_instance_count=len(train_instances),
        validation_instance_count=len(validation_instances),
        test_lodo_count=len(test_lodo_instances),
        pttest_count=len(pttest_instances),
    )

    if args.dry_run:
        manifest["mode"] = "dry_run"
        manifest["completedAt"] = p3.iso_now()
        write_empty(out_dir, manifest)
        print(f"ARC Phase 3 raw-grid gate v2 dry run wrote {out_dir}")
        return 0

    if args.shard_seed is not None:
        if args.shard_seed not in SEED_SLATE:
            print(f"--shard-seed {args.shard_seed} is not in SEED_SLATE {SEED_SLATE}", file=sys.stderr)
            return 2
        seeds = [args.shard_seed]
        max_epochs = args.probe_epochs if args.probe_only else args.max_epochs
        manifest["mode"] = "shard"
        manifest["shardSeed"] = args.shard_seed
        manifest["seedSlateOriginal"] = SEED_SLATE
    else:
        seeds = [args.master_seed] if args.probe_only else SEED_SLATE
        max_epochs = args.probe_epochs if args.probe_only else args.max_epochs
        manifest["mode"] = "probe" if args.probe_only else "full"
    manifest["seedSlateEffective"] = seeds
    manifest["maxEpochsEffective"] = max_epochs
    manifest["epochsOverridden"] = max_epochs != MODEL_SPEC_V2["max_epochs"]

    learning_rows: list[dict[str, Any]] = []
    per_slot_rows: list[dict[str, Any]] = []
    per_instance_any: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []

    for seed in seeds:
        p3.set_global_determinism(seed)
        start = time.perf_counter()
        model = p3.TaskDecoder(p3.input_dim_for_arm(ARM))
        result = p3.train_model(
            model=model,
            arm=ARM,
            train_instances=train_instances,
            validation_instances=validation_instances,
            seed=seed,
            max_epochs=max_epochs,
            device=args.device,
        )
        learning_rows.extend(result["learning_rows"])
        eval_bundle = p3.evaluate_model(
            model=result["model"],
            arm=ARM,
            seed=seed,
            instances=[*validation_instances, *test_lodo_instances, *pttest_instances],
            device=args.device,
        )
        per_slot_rows.extend(eval_bundle["per_slot_rows"])
        per_instance_any.extend(eval_bundle["per_instance_any"])
        residual_rows.extend(eval_bundle["residual_rows"])
        elapsed = time.perf_counter() - start
        candidates.append({
            "arm": ARM,
            "seed": seed,
            "best_epoch": result["best_epoch"],
            "validation_metric": result["best_validation_metric"],
            "validation_loss": result["best_validation_loss"],
            "elapsed_seconds": round(elapsed, 3),
        })
        print(f"{ARM}: seed {seed} best_epoch={result['best_epoch']} elapsed={elapsed:.1f}s")

    candidates.sort(key=lambda row: (-row["validation_metric"], row["validation_loss"], row["seed"]))
    selected_seed = candidates[0]["seed"]
    selected_seed_by_arm = {ARM: selected_seed}
    per_task_rows = p3.aggregate_per_task(per_instance_any, selected_seed_by_arm)
    per_prior_rows = p3.aggregate_per_prior(per_instance_any, selected_seed_by_arm)
    scores = p3.aggregate_scores(per_instance_any, selected_seed_by_arm, args.master_seed)
    gate = adjudicate_raw_grid_gate(per_task_rows, manifest["mode"])

    manifest["completedAt"] = p3.iso_now()
    manifest["selectedSeedByArm"] = selected_seed_by_arm
    manifest["validationRank"] = candidates
    manifest["gateDecision"] = gate

    p3.write_json(out_dir / "manifest.json", manifest)
    p3.write_csv(out_dir / "learning_curves.csv", learning_rows, p3.LEARNING_COLUMNS)
    p3.write_csv(out_dir / "per_instance.csv", per_slot_rows, p3.PER_SLOT_COLUMNS)
    p3.write_csv(out_dir / "per_task.csv", per_task_rows, p3.PER_TASK_COLUMNS)
    p3.write_csv(out_dir / "per_prior.csv", per_prior_rows, p3.PER_PRIOR_COLUMNS)
    p3.write_csv(out_dir / "scores.csv", scores, p3.SCORE_COLUMNS)
    p3.write_csv(out_dir / "quarantine_log.csv", [], p3.QUARANTINE_COLUMNS)
    p3.write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    p3.write_jsonl(out_dir / "per_instance_any.jsonl", per_instance_any)
    p3.write_json(out_dir / "validation_candidates.json", candidates)
    p3.write_json(out_dir / "phase3_receipt.json", {
        "manifest": manifest,
        "validationRank": candidates,
        "scores": scores,
        "perTask": per_task_rows,
        "gateDecision": gate,
        "residuals": residual_rows,
    })
    write_gate_summary(out_dir / "branch_adjudication.md", gate, scores, selected_seed)
    p3.write_json(out_dir / "hashes.json", p3.hash_receipt_files(out_dir))
    print(f"ARC Phase 3 raw-grid gate v2 {manifest['mode']} run wrote {out_dir}")
    print(f"Gate decision: {gate['gate']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ARC Phase 3 strengthened raw-grid control gate")
    parser.add_argument("--data-dir", required=False)
    parser.add_argument("--register", required=False)
    parser.add_argument("--out", default="results/arc/phase3-rawgrid-gate-v2")
    parser.add_argument("--master-seed", type=int, default=20260528)
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--probe-epochs", type=int, default=3)
    parser.add_argument("--max-epochs", type=int, default=MODEL_SPEC_V2["max_epochs"])
    parser.add_argument("--limit-aux-tasks", type=int, default=0, help="Probe-only cap for auxiliary public-training tasks")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard-seed", type=int, default=None, help="Run a single seed from SEED_SLATE and emit a shard intermediate (no gate adjudication).")
    parser.add_argument("--merge", action="store_true", help="Merge shard intermediates into a binding receipt instead of training.")
    parser.add_argument("--shard-dirs", default=None, help="Comma-separated list of shard receipt directories (--merge mode only).")
    args = parser.parse_args()
    if not args.merge:
        if not args.data_dir or not args.register:
            parser.error("--data-dir and --register are required (except in --merge mode)")
    else:
        if not args.shard_dirs:
            parser.error("--merge requires --shard-dirs <dir1,dir2,...>")
    return args


def run_merge(args) -> int:
    started_at = p3.iso_now()
    repo_root = Path(__file__).resolve().parents[3]
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    git = p3.git_state(repo_root, args.allow_dirty)

    shard_dirs = [Path(d.strip()).resolve() for d in args.shard_dirs.split(",") if d.strip()]
    if len(shard_dirs) == 0:
        print("--shard-dirs is empty", file=sys.stderr)
        return 2

    shards = []
    for d in shard_dirs:
        if not d.is_dir():
            print(f"shard dir not found: {d}", file=sys.stderr)
            return 2
        manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("mode") != "shard":
            print(f"shard dir {d} has mode={manifest.get('mode')!r}, expected 'shard'", file=sys.stderr)
            return 2
        if manifest.get("learnerVersion") != LEARNER_VERSION:
            print(f"shard dir {d} learnerVersion={manifest.get('learnerVersion')!r}, expected {LEARNER_VERSION!r}", file=sys.stderr)
            return 2
        shards.append({
            "dir": d,
            "manifest": manifest,
            "per_instance_any": read_jsonl(d / "per_instance_any.jsonl"),
            "per_slot_rows": read_csv_dicts(d / "per_instance.csv"),
            "learning_rows": read_csv_dicts(d / "learning_curves.csv"),
            "residual_rows": read_jsonl(d / "residuals.jsonl"),
            "candidates": json.loads((d / "validation_candidates.json").read_text(encoding="utf-8")),
        })

    assert_shard_consistency(shards)
    shards.sort(key=lambda s: s["manifest"]["shardSeed"])

    per_instance_any: list[dict[str, Any]] = []
    per_slot_rows: list[dict[str, Any]] = []
    learning_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for sh in shards:
        per_instance_any.extend(sh["per_instance_any"])
        per_slot_rows.extend(sh["per_slot_rows"])
        learning_rows.extend(sh["learning_rows"])
        residual_rows.extend(sh["residual_rows"])
        all_candidates.extend(sh["candidates"])

    all_candidates.sort(key=lambda row: (-row["validation_metric"], row["validation_loss"], row["seed"]))
    selected_seed = all_candidates[0]["seed"]
    selected_seed_by_arm = {ARM: selected_seed}
    per_task_rows = p3.aggregate_per_task(per_instance_any, selected_seed_by_arm)
    per_prior_rows = p3.aggregate_per_prior(per_instance_any, selected_seed_by_arm)
    scores = p3.aggregate_scores(per_instance_any, selected_seed_by_arm, shards[0]["manifest"]["masterSeed"])
    gate = adjudicate_raw_grid_gate(per_task_rows, "full")

    first_manifest = shards[0]["manifest"]
    drop_keys = {"mode", "shardSeed", "seedSlateEffective", "seedSlateOriginal", "generatedAt", "completedAt", "command", "tool", "outDir"}
    merged_manifest = {k: v for k, v in first_manifest.items() if k not in drop_keys}
    merged_manifest.update({
        "generatedAt": min(sh["manifest"]["generatedAt"] for sh in shards),
        "completedAt": p3.iso_now(),
        "tool": "docs/prereg/arc/phase3_decoder_v2.py (merge)",
        "command": [sys.executable, "docs/prereg/arc/phase3_decoder_v2.py", *sys.argv[1:]],
        "mode": "full",
        "shardedRun": True,
        "shardSources": [
            {
                "dir": str(sh["dir"]),
                "shardSeed": sh["manifest"]["shardSeed"],
                "generatedAt": sh["manifest"]["generatedAt"],
                "completedAt": sh["manifest"]["completedAt"],
                "gitCommit": sh["manifest"]["gitCommit"],
            }
            for sh in shards
        ],
        "seedSlateEffective": [sh["manifest"]["shardSeed"] for sh in shards],
        "mergeStartedAt": started_at,
        "mergeGitCommit": git["commit"],
        "mergeGitDirty": git["dirty"],
        "mergeAllowDirty": args.allow_dirty,
        "outDir": str(out_dir),
        "selectedSeedByArm": selected_seed_by_arm,
        "validationRank": all_candidates,
        "gateDecision": gate,
    })

    p3.write_json(out_dir / "manifest.json", merged_manifest)
    p3.write_csv(out_dir / "learning_curves.csv", learning_rows, p3.LEARNING_COLUMNS)
    p3.write_csv(out_dir / "per_instance.csv", per_slot_rows, p3.PER_SLOT_COLUMNS)
    p3.write_csv(out_dir / "per_task.csv", per_task_rows, p3.PER_TASK_COLUMNS)
    p3.write_csv(out_dir / "per_prior.csv", per_prior_rows, p3.PER_PRIOR_COLUMNS)
    p3.write_csv(out_dir / "scores.csv", scores, p3.SCORE_COLUMNS)
    p3.write_csv(out_dir / "quarantine_log.csv", [], p3.QUARANTINE_COLUMNS)
    p3.write_jsonl(out_dir / "residuals.jsonl", residual_rows)
    p3.write_jsonl(out_dir / "per_instance_any.jsonl", per_instance_any)
    p3.write_json(out_dir / "validation_candidates.json", all_candidates)
    p3.write_json(out_dir / "phase3_receipt.json", {
        "manifest": merged_manifest,
        "validationRank": all_candidates,
        "scores": scores,
        "perTask": per_task_rows,
        "gateDecision": gate,
        "residuals": residual_rows,
    })
    write_gate_summary(out_dir / "branch_adjudication.md", gate, scores, selected_seed)
    split_first = shards[0]["dir"] / "split.csv"
    if split_first.exists():
        (out_dir / "split.csv").write_text(split_first.read_text(encoding="utf-8"), encoding="utf-8")
    p3.write_json(out_dir / "hashes.json", p3.hash_receipt_files(out_dir))
    print(f"ARC Phase 3 raw-grid gate v2 merge wrote {out_dir}")
    print(f"Gate decision: {gate['gate']}")
    return 0


def assert_shard_consistency(shards: list[dict[str, Any]]) -> None:
    if len(shards) < 2:
        return
    ref = shards[0]["manifest"]
    keys = [
        "learnerVersion", "protocolVersion", "receiptSchemaVersion", "featureSchemaVersion",
        "arm", "registerHash", "dataDirHash", "gitCommit", "modelSpec",
        "registerPath", "dataDir", "limitAuxTasks", "maxEpochsEffective",
    ]
    seeds: set[int] = set()
    for sh in shards:
        m = sh["manifest"]
        seed = m.get("shardSeed")
        if seed in seeds:
            raise SystemExit(f"shard {sh['dir']} has duplicate shardSeed={seed}")
        seeds.add(seed)
        for key in keys:
            ref_val = json.dumps(ref.get(key), sort_keys=True)
            sh_val = json.dumps(m.get(key), sort_keys=True)
            if ref_val != sh_val:
                raise SystemExit(f"shard {sh['dir']} disagrees with {shards[0]['dir']} on '{key}':\n  ref={ref_val}\n  sh ={sh_val}")


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


def patch_v1_globals() -> None:
    p3.LEARNER_VERSION = LEARNER_VERSION
    p3.PROTOCOL_VERSION = PROTOCOL_VERSION
    p3.RECEIPT_SCHEMA_VERSION = RECEIPT_SCHEMA_VERSION
    p3.MODEL_SPEC.clear()
    p3.MODEL_SPEC.update(MODEL_SPEC_V2)


def load_public_training_tasks(data_dir: Path, register_path: Path, limit_aux_tasks: int) -> tuple[list[p3.Task], str, str]:
    registered, register_hash, _ = p3.load_tasks(data_dir, register_path)
    registered_by_id = {task.task_id: task for task in registered}
    split_by_id = p3.expected_split_by_task()
    heldout = {task_id for task_id, split in split_by_id.items() if split in {"validation", "test"}}
    tasks: list[p3.Task] = []
    file_hashes = []
    aux_count = 0
    for path in sorted((data_dir / "training").glob("*.json")):
        task_id = path.stem
        raw = path.read_text(encoding="utf-8-sig")
        file_hashes.append({"file": f"training/{task_id}.json", "sha256": p3.sha256_text(raw)})
        parsed = json.loads(raw)
        if task_id in registered_by_id:
            task = registered_by_id[task_id]
            tasks.append(task)
            continue
        if task_id in heldout:
            continue
        if limit_aux_tasks and aux_count >= limit_aux_tasks:
            continue
        aux_count += 1
        tasks.append(p3.Task(
            task_id=task_id,
            primary_prior="public_training_aux",
            predicted_boundary="auxiliary_public_training",
            row={"task_id": task_id},
            train=[{"index": i, "input": pair["input"], "output": pair["output"]} for i, pair in enumerate(parsed["train"])],
            test=[{"index": i, "input": pair["input"], "output": pair.get("output")} for i, pair in enumerate(parsed["test"])],
            split="train",
        ))
    return tasks, register_hash, p3.sha256_text(json.dumps(file_hashes, sort_keys=True, separators=(",", ":")))


def build_pttest_if_available(tasks: list[p3.Task], lane: str) -> list[p3.Instance]:
    instances = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        for test in task.test:
            if test["output"] is None:
                continue
            instances.append(p3.Instance(
                lane=lane,
                instance_id=f"{lane}:{task.task_id}:{test['index']}",
                task=task,
                query_index=test["index"],
                query_input=test["input"],
                target_output=test["output"],
                conditioning=task.train,
            ))
    return instances


def base_manifest(**kwargs: Any) -> dict[str, Any]:
    args = kwargs["args"]
    return {
        "generatedAt": kwargs["started_at"],
        "completedAt": None,
        "tool": "docs/prereg/arc/phase3_decoder_v2.py",
        "command": [sys.executable, "docs/prereg/arc/phase3_decoder_v2.py", *sys.argv[1:]],
        "gitCommit": kwargs["git"]["commit"],
        "gitDirty": kwargs["git"]["dirty"],
        "allowDirty": args.allow_dirty,
        "dataDir": str(kwargs["data_dir"]),
        "registerPath": str(kwargs["register_path"]),
        "outDir": str(kwargs["out_dir"]),
        "masterSeed": args.master_seed,
        "featureSchemaVersion": p3.FEATURE_SCHEMA_VERSION,
        "protocolVersion": PROTOCOL_VERSION,
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION,
        "learnerVersion": LEARNER_VERSION,
        "arm": ARM,
        "taskCount": kwargs["task_count"],
        "trainInstanceCount": kwargs["train_instance_count"],
        "validationInstanceCount": kwargs["validation_instance_count"],
        "testLodoInstanceCount": kwargs["test_lodo_count"],
        "pttestInstanceCount": kwargs["pttest_count"],
        "registerHash": kwargs["register_hash"],
        "dataDirHash": kwargs["data_hash"],
        "pythonVersion": sys.version,
        "torchVersion": torch.__version__,
        "platform": p3.platform.platform(),
        "device": args.device,
        "modelSpec": MODEL_SPEC_V2,
        "limitAuxTasks": args.limit_aux_tasks,
    }


def write_split_with_aux(path: Path, tasks: list[p3.Task], registered_rows: list[dict[str, Any]]) -> None:
    rows = list(registered_rows)
    aux_count = sum(1 for task in tasks if task.primary_prior == "public_training_aux")
    rows.append({
        "task_id": "__auxiliary_public_training_count__",
        "primary_prior": "public_training_aux",
        "predicted_boundary": "auxiliary_public_training",
        "split": f"train_count={aux_count}",
    })
    p3.write_csv(path, rows, ["task_id", "primary_prior", "predicted_boundary", "split"])


def write_empty(out_dir: Path, manifest: dict[str, Any]) -> None:
    p3.write_json(out_dir / "manifest.json", manifest)
    p3.write_csv(out_dir / "learning_curves.csv", [], p3.LEARNING_COLUMNS)
    p3.write_csv(out_dir / "per_instance.csv", [], p3.PER_SLOT_COLUMNS)
    p3.write_csv(out_dir / "per_task.csv", [], p3.PER_TASK_COLUMNS)
    p3.write_csv(out_dir / "per_prior.csv", [], p3.PER_PRIOR_COLUMNS)
    p3.write_csv(out_dir / "scores.csv", [], p3.SCORE_COLUMNS)
    p3.write_csv(out_dir / "quarantine_log.csv", [], p3.QUARANTINE_COLUMNS)
    p3.write_jsonl(out_dir / "residuals.jsonl", [])
    p3.write_json(out_dir / "phase3_receipt.json", {"manifest": manifest, "gateDecision": {"gate": "not_run"}})
    (out_dir / "branch_adjudication.md").write_text("# Raw-Grid Gate V2\n\nDry run only. No gate decision.\n", encoding="utf8")
    p3.write_json(out_dir / "hashes.json", p3.hash_receipt_files(out_dir))


def adjudicate_raw_grid_gate(per_task_rows: list[dict[str, Any]], mode: str) -> dict[str, Any]:
    if mode != "full":
        return {"gate": "not_adjudicated", "reason": f"{mode} run only"}
    lodo_success = exact_task_count(per_task_rows, "test_lodo")
    pttest_success = exact_task_count(per_task_rows, "pttest")
    if lodo_success >= 2 and pttest_success >= 2:
        return {
            "gate": "full_grid_control_pass",
            "reason": "raw_grid_lowcap cleared the non-trivial exact-grid floor on both held-out lanes",
            "test_lodo_exact_tasks": lodo_success,
            "pttest_exact_tasks": pttest_success,
        }
    return {
        "gate": "full_grid_control_floor",
        "reason": "raw_grid_lowcap did not clear the pre-registered exact-grid floor on both held-out lanes",
        "test_lodo_exact_tasks": lodo_success,
        "pttest_exact_tasks": pttest_success,
    }


def exact_task_count(rows: list[dict[str, Any]], lane: str) -> int:
    return sum(
        1
        for row in rows
        if row["lane"] == lane and row["arm"] == ARM and p3.numeric(row["grid_exact_any_rate"]) > 0.010
    )


def write_gate_summary(path: Path, gate: dict[str, Any], scores: list[dict[str, Any]], selected_seed: int) -> None:
    lines = [
        "# Phase 3 Raw-Grid Gate V2",
        "",
        f"Gate: **{gate['gate']}**",
        "",
        gate.get("reason", ""),
        "",
        f"Selected seed: `{selected_seed}`",
        "",
        "| lane | grid exact | pixel best |",
        "| --- | ---: | ---: |",
    ]
    for row in scores:
        lines.append(f"| `{row['lane']}` | `{row['grid_exact_any_rate']}` | `{row['pixel_accuracy_best_mean']}` |")
    path.write_text("\n".join(lines) + "\n", encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())
