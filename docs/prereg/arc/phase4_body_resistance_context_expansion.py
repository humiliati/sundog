#!/usr/bin/env python3
"""ARC Phase 4 v2 -- body-resistance context expansion (spectrum-blind sample-size test).

Spec: PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md. The disciplined reopen of
v1's `arc_body_inconclusive`: does the raw-grid body participation ratio clear the
UNCHANGED v1 bar (`PR_HIGH_MIN=20`) when the context universe is expanded by a
deterministic, spectrum-blind rule over ALL public-training tasks? Thresholds and
estimators are carried verbatim from v1 by importing `phase4_body_resistance`;
only the all-training loader and the 4-branch adjudication are new. Read-off
dimensionality only. Training-split only. torch.linalg (numpy unavailable).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import phase4_body_resistance as p4          # noqa: E402  (v1 estimators + frozen constants)
import phase3d_mask_target_v3 as rep         # noqa: E402  (frozen ARC representations)
import phase3_branch_e_program_search as io  # noqa: E402  (IO helpers)

write_json = io.write_json
write_csv = io.write_csv
sha256_file = io.sha256_file
sha256_text = io.sha256_text
round_float = io.round_float
iso_now = io.iso_now
git_state = io.git_state
assert_training_data_dir = io.assert_training_data_dir
hash_receipt_files = io.hash_receipt_files

# Frozen constants carried verbatim from v1 (imported, never redefined here).
PR_HIGH_MIN = p4.PR_HIGH_MIN
PR_MARGINAL_MAX = p4.PR_MARGINAL_MAX
FVE_RECON_CEILING = p4.FVE_RECON_CEILING
FVE_MARGINAL_MIN = p4.FVE_MARGINAL_MIN
PR_BOUND_SATURATION_MAX = p4.PR_BOUND_SATURATION_MAX
RIDGE_LAMBDA = p4.RIDGE_LAMBDA
SHADOW_DIM_K = p4.SHADOW_DIM_K
ENERGY_LEVELS = p4.ENERGY_LEVELS
PCA_K_GRID = p4.PCA_K_GRID

PROTOCOL_VERSION = "arc-p4v2-context-expansion-v1"
RECEIPT_SCHEMA_VERSION = "arc-p4v2-context-expansion-receipt-v1"
LANES = ["pttest", "train_lodo"]            # lexicographic lane order (spec prefix_order)
PREFIX_SIZES = [491, 750, 1000, 1500, 2000, 3000, "all"]

P4V2_FILES = {
    "task_inventory.csv": ["task_id", "n_train", "n_test", "valid", "n_contexts", "exclude_reason"],
    "split.csv": ["task_id", "lane", "instance_id", "heldout"],
    "body_spectrum.csv": ["scope", "n_contexts", "participation_ratio", "rank_90", "rank_95", "rank_99", "top1_energy_frac"],
    "shadow_fve.csv": ["shadow", "shadow_dim", "fve_heldout"],
    "matched_dim_fve_curve.csv": ["k", "fve_heldout"],
    "prefix_dimensionality.csv": ["prefix_size", "n_contexts", "participation_ratio", "rank_95", "pr_bound_ratio"],
    "per_lane_dimensionality.csv": ["lane", "n_contexts", "participation_ratio", "rank_95"],
}


class Ctx:
    __slots__ = ("task_id", "lane", "instance_id", "query_input")

    def __init__(self, task_id, lane, instance_id, query_input):
        self.task_id = task_id
        self.lane = lane
        self.instance_id = instance_id
        self.query_input = query_input

    @property
    def sort_key(self):
        return (self.task_id, self.lane, self.instance_id)


def _valid_grid(g: Any) -> bool:
    if not isinstance(g, list) or not g or not all(isinstance(r, list) and r for r in g):
        return False
    w = len(g[0])
    if not (1 <= len(g) <= 30 and 1 <= w <= 30):
        return False
    for r in g:
        if len(r) != w:
            return False
        for v in r:
            if not isinstance(v, int) or not (0 <= v <= 9):
                return False
    return True


def _valid_task(t: Any) -> bool:
    if not isinstance(t, dict) or "train" not in t or "test" not in t:
        return False
    if not isinstance(t["train"], list) or not isinstance(t["test"], list) or not t["train"] or not t["test"]:
        return False
    for sec in ("train", "test"):
        for pair in t[sec]:
            if not isinstance(pair, dict) or "input" not in pair or not _valid_grid(pair["input"]):
                return False
    return True


def load_all_training_tasks(data_dir: Path):
    """Spectrum-blind: every public-training task id under data_dir/training, minus
    JSON/grid-invalid files. No selection by spectrum/solver/complexity."""
    train_dir = data_dir / "training"
    inventory, contexts, file_hashes, excluded = [], [], [], 0
    for path in sorted(train_dir.glob("*.json")):
        task_id = path.stem
        raw = path.read_bytes()
        file_hashes.append(f"{task_id}:{hashlib.sha256(raw).hexdigest()}")
        try:
            t = json.loads(raw)
        except json.JSONDecodeError:
            t = None
        if t is None or not _valid_task(t):
            inventory.append({"task_id": task_id, "n_train": 0, "n_test": 0, "valid": 0, "n_contexts": 0, "exclude_reason": "json_or_grid_invalid"})
            excluded += 1
            continue
        n_train, n_test = len(t["train"]), len(t["test"])
        task_ctx = []
        for j, pair in enumerate(t["train"]):
            task_ctx.append(Ctx(task_id, "train_lodo", f"{task_id}:train_lodo:{j}", pair["input"]))
        for k, pair in enumerate(t["test"]):
            task_ctx.append(Ctx(task_id, "pttest", f"{task_id}:pttest:{k}", pair["input"]))
        contexts.extend(task_ctx)
        inventory.append({"task_id": task_id, "n_train": n_train, "n_test": n_test, "valid": 1, "n_contexts": len(task_ctx), "exclude_reason": ""})
    contexts.sort(key=lambda c: c.sort_key)
    data_hash = sha256_text("\n".join(sorted(file_hashes)))
    return contexts, inventory, data_hash, excluded


def matched_dim_curve(B_tr: torch.Tensor, B_he: torch.Tensor, k_grid):
    """Held-out FVE(body | top-k PCA) for every k via ONE train-body SVD.
    Mathematically identical to calling p4.fve_pca per k (same Bc, same Vh slice);
    folded into one SVD because the v2 train set is ~9x larger."""
    b_mean = B_tr.mean(dim=0, keepdim=True)
    Bc = (B_tr - b_mean).double()
    _, _, Vh = torch.linalg.svd(Bc, full_matrices=False)
    B_he_c = (B_he - b_mean).double()
    ss_tot = float((B_he_c ** 2).sum())
    out = {}
    for k in k_grid:
        kk = min(k, Vh.shape[0])
        Vk = Vh[:kk].t()
        proj = B_he_c @ Vk @ Vk.t()
        ss_res = float(((B_he_c - proj) ** 2).sum())
        out[k] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return out


def spectrum_stats(B_sub: torch.Tensor):
    Bc = (B_sub - B_sub.mean(dim=0, keepdim=True)).double()
    sv = torch.linalg.svdvals(Bc)
    pr = p4.participation_ratio(sv)
    eranks = p4.energy_ranks(sv, ENERGY_LEVELS)
    top1 = float((sv.double()[0] ** 2) / (sv.double() ** 2).sum()) if B_sub.shape[0] else 0.0
    return pr, eranks, top1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC Phase 4 v2 body-resistance context expansion")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--inventory-only", action="store_true", help="smoke: count tasks/contexts, no spectral math")
    p.add_argument("--limit-tasks", type=int, default=0, help="smoke: cap number of training tasks loaded (canonical order)")
    p.add_argument("--allow-dirty", action="store_true")
    args = p.parse_args()
    if not args.dry_run and not args.data_dir:
        p.error("--data-dir is required (except --dry-run)")
    return args


def base_manifest(args) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    git = git_state(repo_root, args.allow_dirty)
    here = Path(__file__).resolve().parent
    spec_v2 = here / "PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md"
    spec_v1 = here / "PHASE4_BODY_RESISTANCE_SPEC.md"
    return {
        "generatedAt": iso_now(), "tool": "docs/prereg/arc/phase4_body_resistance_context_expansion.py",
        "command": [sys.executable, "docs/prereg/arc/phase4_body_resistance_context_expansion.py", *sys.argv[1:]],
        "gitCommit": git["commit"], "gitDirty": git["dirty"], "allowDirty": args.allow_dirty,
        "protocolVersion": PROTOCOL_VERSION, "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION,
        "specPathV2": "docs/prereg/arc/PHASE4_BODY_RESISTANCE_CONTEXT_EXPANSION_SPEC.md",
        "specHashV2": (sha256_file(spec_v2) if spec_v2.exists() else "NA"),
        "specPathV1": "docs/prereg/arc/PHASE4_BODY_RESISTANCE_SPEC.md",
        "specHashV1": (sha256_file(spec_v1) if spec_v1.exists() else "NA"),
        "runnerSha256": sha256_file(Path(__file__).resolve()),
        "v1RunnerSha256": sha256_file(here / "phase4_body_resistance.py"),
        "repSha256": sha256_file(here / "phase3d_mask_target_v3.py"),
        "pythonVersion": sys.version, "platform": platform.platform(),
        "bodyDim": rep.RAW_GRID_DIM, "metadataShadowDim": rep.METADATA_DIM, "signatureShadowDim": rep.SIGNATURE_VECTOR_DIM,
        "thresholds": {"pr_high_min": PR_HIGH_MIN, "pr_marginal_max": PR_MARGINAL_MAX,
                       "fve_recon_ceiling": FVE_RECON_CEILING, "fve_marginal_min": FVE_MARGINAL_MIN,
                       "pr_bound_saturation_max": PR_BOUND_SATURATION_MAX, "ridge_lambda": RIDGE_LAMBDA,
                       "shadow_dim_k": SHADOW_DIM_K, "heldout_rule": "sha256(instance_id)%10<3",
                       "energy_levels": ENERGY_LEVELS, "pca_k_grid": PCA_K_GRID, "carried_from": "phase4_body_resistance v1 Amendment 1"},
        "prefixSizes": [str(s) for s in PREFIX_SIZES],
    }


def write_empty(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(out_dir / "manifest.json", manifest)
    for fname, cols in P4V2_FILES.items():
        write_csv(out_dir / fname, [], cols)
    write_json(out_dir / "phase4_body_resistance_context_expansion_receipt.json", {"manifest": manifest, "branch": None})
    (out_dir / "branch_adjudication.md").write_text("# ARC Phase 4 v2 context expansion\n\nDry run / empty receipt.\n", encoding="utf-8")
    (out_dir / "commands.md").write_text("# ARC Phase 4 v2 commands\n\nDry run / empty receipt.\n", encoding="utf-8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(args)

    if args.dry_run:
        manifest["mode"] = "dry_run"; manifest["completedAt"] = iso_now()
        write_empty(out_dir, manifest)
        print(f"ARC Phase 4 v2 dry run wrote {out_dir}")
        return 0

    data_dir = Path(args.data_dir).resolve()
    assert_training_data_dir(data_dir)
    contexts, inventory, data_hash, excluded = load_all_training_tasks(data_dir)
    if args.limit_tasks > 0:
        keep = {row["task_id"] for row in inventory if row["valid"]}
        keep = set(sorted(keep)[: args.limit_tasks])
        contexts = [c for c in contexts if c.task_id in keep]
    write_csv(out_dir / "task_inventory.csv", inventory, P4V2_FILES["task_inventory.csv"])
    manifest["dataDirHash"] = data_hash
    manifest["taskInventoryHash"] = sha256_text(json.dumps(inventory, sort_keys=True))
    manifest["nTasksTotal"] = len(inventory)
    manifest["nTasksExcluded"] = excluded
    manifest["nTasksValid"] = sum(r["valid"] for r in inventory)
    n = len(contexts)
    manifest["nContexts"] = n
    manifest["contextUniverse"] = {ln: sum(1 for c in contexts if c.lane == ln) for ln in LANES}

    split_rows = [{"task_id": c.task_id, "lane": c.lane, "instance_id": c.instance_id, "heldout": int(p4.is_heldout(c.instance_id))} for c in contexts]
    write_csv(out_dir / "split.csv", split_rows, P4V2_FILES["split.csv"])

    if args.inventory_only:
        manifest["mode"] = "inventory_only"; manifest["completedAt"] = iso_now()
        for fname in ("body_spectrum.csv", "shadow_fve.csv", "matched_dim_fve_curve.csv", "prefix_dimensionality.csv", "per_lane_dimensionality.csv"):
            write_csv(out_dir / fname, [], P4V2_FILES[fname])
        write_json(out_dir / "manifest.json", manifest)
        write_json(out_dir / "phase4_body_resistance_context_expansion_receipt.json", {"manifest": manifest, "branch": None})
        (out_dir / "branch_adjudication.md").write_text(f"# ARC Phase 4 v2 -- inventory smoke\n\n{manifest['nTasksValid']} valid tasks, {n} contexts ({manifest['contextUniverse']}).\n", encoding="utf-8")
        (out_dir / "commands.md").write_text("# inventory smoke\n", encoding="utf-8")
        write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
        print(f"ARC Phase 4 v2 inventory: {manifest['nTasksValid']} tasks, {n} contexts {manifest['contextUniverse']}")
        return 0

    # ---- body + shadow matrices (canonical context order) ----
    B = torch.tensor([p4.body_vec(c.query_input) for c in contexts], dtype=torch.float32)
    S_meta = torch.tensor([p4.metadata_shadow_vec(c.query_input) for c in contexts], dtype=torch.float32)
    S_sig = torch.tensor([p4.signature_shadow_vec(c.query_input) for c in contexts], dtype=torch.float32)
    held = torch.tensor([p4.is_heldout(c.instance_id) for c in contexts], dtype=torch.bool)
    tr = ~held
    manifest["nHeldout"] = int(held.sum())

    # ---- full-set spectrum / PR ----
    pr, eranks, top1 = spectrum_stats(B)
    write_csv(out_dir / "body_spectrum.csv", [{"scope": "U_all_expanded", "n_contexts": n, "participation_ratio": round_float(pr),
                                               "rank_90": eranks["rank_90"], "rank_95": eranks["rank_95"], "rank_99": eranks["rank_99"],
                                               "top1_energy_frac": round_float(top1)}], P4V2_FILES["body_spectrum.csv"])

    # ---- per-lane PR ----
    per_lane = []
    for ln in LANES:
        idx = [j for j, c in enumerate(contexts) if c.lane == ln]
        if len(idx) < 2:
            per_lane.append({"lane": ln, "n_contexts": len(idx), "participation_ratio": 0.0, "rank_95": 0}); continue
        plpr, pleranks, _ = spectrum_stats(B[idx])
        per_lane.append({"lane": ln, "n_contexts": len(idx), "participation_ratio": round_float(plpr), "rank_95": pleranks["rank_95"]})
    write_csv(out_dir / "per_lane_dimensionality.csv", per_lane, P4V2_FILES["per_lane_dimensionality.csv"])

    # ---- prefix downsample stability (descriptive only; canonical prefix order) ----
    prefix_rows = []
    for size in PREFIX_SIZES:
        m = n if size == "all" else int(size)
        if m > n:
            prefix_rows.append({"prefix_size": str(size), "n_contexts": "skipped", "participation_ratio": "", "rank_95": "", "pr_bound_ratio": ""})
            continue
        ppr, peranks, _ = spectrum_stats(B[:m])
        pbound = float(min(m - 1, B.shape[1]))
        prefix_rows.append({"prefix_size": str(size), "n_contexts": m, "participation_ratio": round_float(ppr),
                            "rank_95": peranks["rank_95"], "pr_bound_ratio": round_float(ppr / pbound if pbound > 0 else 0.0)})
    write_csv(out_dir / "prefix_dimensionality.csv", prefix_rows, P4V2_FILES["prefix_dimensionality.csv"])

    # ---- FVE(body | shadow) held-out ----
    shadow_rows = []
    for name, S, dim in [("metadata_only", S_meta, rep.METADATA_DIM), ("signature_palette", S_sig, rep.SIGNATURE_VECTOR_DIM)]:
        fve = p4.fve_ridge(B[tr], S[tr], B[held], S[held], RIDGE_LAMBDA)
        shadow_rows.append({"shadow": name, "shadow_dim": dim, "fve_heldout": round_float(fve)})
    write_csv(out_dir / "shadow_fve.csv", shadow_rows, P4V2_FILES["shadow_fve.csv"])

    # ---- matched-dim PCA FVE curve (held-out, one train SVD) ----
    curve = matched_dim_curve(B[tr], B[held], PCA_K_GRID)
    write_csv(out_dir / "matched_dim_fve_curve.csv", [{"k": k, "fve_heldout": round_float(curve[k])} for k in PCA_K_GRID], P4V2_FILES["matched_dim_fve_curve.csv"])
    fve_matched = curve[SHADOW_DIM_K]

    # ---- branch adjudication (table-order precedence; spec section "Branches") ----
    pr_bound = float(min(n - 1, B.shape[1]))
    pr_bound_ratio = (pr / pr_bound) if pr_bound > 0 else 0.0
    if pr_bound_ratio <= PR_BOUND_SATURATION_MAX and pr >= PR_HIGH_MIN and fve_matched <= FVE_RECON_CEILING:
        branch = "arc_body_high_dim_expanded"
        reason = f"PR/bound {pr_bound_ratio:.3f} <= {PR_BOUND_SATURATION_MAX}, body PR {pr:.1f} >= {PR_HIGH_MIN}, and matched-dim FVE(top-{SHADOW_DIM_K} PCA) {fve_matched:.3f} <= {FVE_RECON_CEILING}: the expanded spectrum-blind universe clears the unchanged v1 high-dim body gate."
    elif pr <= PR_MARGINAL_MAX or fve_matched >= FVE_MARGINAL_MIN:
        branch = "arc_body_marginal_expanded"
        reason = f"body PR {pr:.1f} <= {PR_MARGINAL_MAX} or matched-dim FVE {fve_matched:.3f} >= {FVE_MARGINAL_MIN}: the larger universe shows the body in the marginal band."
    elif pr_bound_ratio > PR_BOUND_SATURATION_MAX:
        branch = "arc_body_sample_saturated_expanded"
        reason = f"PR/bound {pr_bound_ratio:.3f} > {PR_BOUND_SATURATION_MAX}: the run is sample-bound and cannot adjudicate the high-dim claim."
    else:
        branch = "arc_body_inconclusive_expanded"
        reason = f"PR {pr:.1f} (PR/bound {pr_bound_ratio:.3f}) and matched-dim FVE {fve_matched:.3f}: the larger universe still does not adjudicate the body-resistance threshold."

    manifest["completedAt"] = iso_now()
    manifest["bodyResistance"] = {
        "participation_ratio": round_float(pr), "energy_ranks": eranks, "top1_energy_frac": round_float(top1),
        "pr_sample_bound": pr_bound, "pr_bound_ratio": round_float(pr_bound_ratio), "sample_saturated": pr_bound_ratio > PR_BOUND_SATURATION_MAX,
        "fve_metadata_shadow": shadow_rows[0]["fve_heldout"], "fve_signature_shadow": shadow_rows[1]["fve_heldout"],
        "fve_matched_dim_pca": round_float(fve_matched), "matched_dim_k": SHADOW_DIM_K,
        "fve_curve": {str(k): round_float(curve[k]) for k in PCA_K_GRID},
    }
    manifest["branch"] = branch
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "phase4_body_resistance_context_expansion_receipt.json", {"manifest": manifest, "branch": branch, "branchReason": reason})

    v1_pr, v1_fve = 9.146503441, 0.658560494
    (out_dir / "branch_adjudication.md").write_text(
        f"# ARC Phase 4 v2 -- Body-Resistance Context Expansion -- Branch Adjudication\n\n**Branch: `{branch}`**\n\n{reason}\n\n"
        f"- {manifest['nTasksValid']} valid public-training tasks ({excluded} excluded); n_contexts **{n}** "
        f"(train_lodo {manifest['contextUniverse']['train_lodo']} / pttest {manifest['contextUniverse']['pttest']}; held-out {int(held.sum())}); body dim {rep.RAW_GRID_DIM}\n"
        f"- body participation ratio: **{pr:.2f}** (v1 491-context PR was {v1_pr:.2f}; marginal substrates ~ 2); "
        f"{pr_bound_ratio:.3f} of sample bound {pr_bound:.0f} (saturation guard {PR_BOUND_SATURATION_MAX})\n"
        f"- top-1 energy {top1:.3f}; energy ranks 90/95/99 = {eranks['rank_90']}/{eranks['rank_95']}/{eranks['rank_99']}\n"
        f"- matched-dim FVE(body | top-{SHADOW_DIM_K} PCA) held-out: **{fve_matched:.3f}** (v1 was {v1_fve:.3f}; high_dim if <= {FVE_RECON_CEILING})\n"
        f"- coarse shadow FVE held-out: metadata {shadow_rows[0]['fve_heldout']:.3f}, signature {shadow_rows[1]['fve_heldout']:.3f} (baselines only)\n"
        f"- prefix PR sweep (sample-size diagnostic): "
        + ", ".join(f"{r['prefix_size']}->{r['participation_ratio']}" for r in prefix_rows if r["n_contexts"] != "skipped") + "\n\n"
        f"Thresholds carried UNCHANGED from v1 (PR_HIGH_MIN={PR_HIGH_MIN}, FVE ceiling {FVE_RECON_CEILING}); not retuned. "
        f"Read-off dimensionality only -- NOT a control witness, sufficiency proof, ARC solve, or eval/Kaggle claim.\n",
        encoding="utf-8")
    (out_dir / "commands.md").write_text(
        "# ARC Phase 4 v2 commands\n\n```\nnode scripts/arc-phase4-body-resistance-context-expansion.mjs \\\n"
        "  --data-dir \"$env:USERPROFILE/Datasets/ARC-AGI-2/data\" \\\n"
        "  --out results/arc/phase4-body-resistance-context-expanded\n```\n", encoding="utf-8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Phase 4 v2 wrote {out_dir}")
    print(f"Branch: {branch}  PR={pr:.2f}  FVE28={fve_matched:.3f}  n={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
