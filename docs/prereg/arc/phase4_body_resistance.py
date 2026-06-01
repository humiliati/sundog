#!/usr/bin/env python3
"""ARC Phase 4 -- body-resistance / low-dimensional-collapse probe.

Spec: PHASE4_BODY_RESISTANCE_SPEC.md. Ports the C1/Mesa participation-ratio +
FVE(body|shadow) estimators to ARC: is the raw-grid BODY genuinely
high-dimensional (effective rank >> the marginal substrates' ~2), or does it
collapse to a low-dim manifold like Mesa net.7? Read-off dimensionality only --
NOT a control witness, NOT sufficiency, NOT an ARC solve (see spec §"Honest
Scope"). Training-split only. All linear algebra is torch (numpy is unavailable).
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
import phase3_branch_e_program_search as v1  # noqa: E402  (loaders + IO helpers)
import phase3d_mask_target_v3 as rep         # noqa: E402  (frozen ARC representations)

load_tasks = v1.load_tasks
build_lodo_instances = v1.build_lodo_instances
build_pttest_instances = v1.build_pttest_instances
write_json = v1.write_json
write_csv = v1.write_csv
sha256_text = v1.sha256_text
sha256_file = v1.sha256_file
round_float = v1.round_float
iso_now = v1.iso_now
git_state = v1.git_state
assert_training_data_dir = v1.assert_training_data_dir
hash_receipt_files = v1.hash_receipt_files

# ============================================================================
# Frozen constants (spec). Thresholds calibrated ONLY to the marginal
# substrates' published numbers (PR ~ 2; FVE ~ 0.99), never to ARC output.
# ============================================================================
FEATURE_SCHEMA_VERSION = "arc-p4-body-resistance-v1"
PROTOCOL_VERSION = "arc-p4-body-resistance-v1"
RECEIPT_SCHEMA_VERSION = "arc-p4-body-resistance-receipt-v1"
LEARNER_VERSION = "body_resistance_probe"

RIDGE_LAMBDA = 1.0
HELDOUT_FRACTION_MOD = 10          # held-out if sha256(instance_id) % 10 < 3 -> ~30%
HELDOUT_CUT = 3
ENERGY_LEVELS = [0.90, 0.95, 0.99]
PCA_K_GRID = [1, 2, 5, 10, 28, 50, 100, 200]
SHADOW_DIM_K = rep.METADATA_DIM    # 28 -- the matched-dim PCA cut

# Branch thresholds (frozen; calibrated to the marginal band PR~2 / FVE~0.99).
PR_HIGH_MIN = 20.0                 # >= 10x the marginal PR ~ 2
PR_MARGINAL_MAX = 5.0              # near the marginal band
FVE_RECON_CEILING = 0.90           # high_dim needs matched-dim FVE <= this
FVE_MARGINAL_MIN = 0.95            # marginal if matched-dim FVE >= this
# Sample-bound guard (spec branch table "or sample-bound dominates" + caveat 3):
# PR is bounded by min(n_features, n_contexts-1). If PR sits near that bound the
# spectrum is near-isotropic at the sample scale and the high-dim reading is
# unreliable -> inconclusive. Strict guard: makes high_dim HARDER, never easier.
PR_BOUND_SATURATION_MAX = 0.90

LANES = ["validation_lodo", "validation_pttest", "test_lodo", "pttest"]


def body_vec(grid: list[list[int]]) -> list[float]:
    return rep.raw_grid_onehot(grid)


def metadata_shadow_vec(grid: list[list[int]]) -> list[float]:
    return rep.metadata_vector(grid, rep.project_grid_shadow(grid))


def signature_shadow_vec(grid: list[list[int]]) -> list[float]:
    # Provably identical to the frozen Phase 3E signature_palette arm vector.
    return rep.feature_vector(grid, "signature_palette_edit_mask_v3")


def participation_ratio(singular_values: torch.Tensor) -> float:
    lam = singular_values.double() ** 2
    s1 = float(lam.sum())
    s2 = float((lam ** 2).sum())
    return (s1 * s1 / s2) if s2 > 0 else 0.0


def energy_ranks(singular_values: torch.Tensor, levels: list[float]) -> dict[str, int]:
    lam = singular_values.double() ** 2
    total = float(lam.sum())
    if total <= 0:
        return {f"rank_{int(l*100)}": 0 for l in levels}
    cum = torch.cumsum(lam, dim=0) / total
    out = {}
    for l in levels:
        idx = int((cum >= l).nonzero()[0].item()) + 1 if bool((cum >= l).any()) else len(lam)
        out[f"rank_{int(l*100)}"] = idx
    return out


def fve_ridge(B_tr, S_tr, B_he, S_he, lam: float) -> float:
    """Held-out FVE of predicting body B from shadow S via ridge (torch only)."""
    b_mean = B_tr.mean(dim=0, keepdim=True)
    s_mean = S_tr.mean(dim=0, keepdim=True)
    Bc = (B_tr - b_mean).double()
    Sc = (S_tr - s_mean).double()
    d = Sc.shape[1]
    A = Sc.t() @ Sc + lam * torch.eye(d, dtype=torch.float64)
    W = torch.linalg.solve(A, Sc.t() @ Bc)            # d x body_dim
    B_he_c = (B_he - b_mean).double()
    pred = (S_he - s_mean).double() @ W
    ss_res = float(((B_he_c - pred) ** 2).sum())
    ss_tot = float((B_he_c ** 2).sum())
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def fve_pca(B_tr, B_he, k: int) -> float:
    """Held-out FVE of reconstructing body from its own top-k principal components."""
    b_mean = B_tr.mean(dim=0, keepdim=True)
    Bc = (B_tr - b_mean).double()
    _, _, Vh = torch.linalg.svd(Bc, full_matrices=False)
    kk = min(k, Vh.shape[0])
    Vk = Vh[:kk].t()                                  # body_dim x k
    B_he_c = (B_he - b_mean).double()
    proj = B_he_c @ Vk @ Vk.t()
    ss_res = float(((B_he_c - proj) ** 2).sum())
    ss_tot = float((B_he_c ** 2).sum())
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def is_heldout(instance_id: str) -> bool:
    h = int.from_bytes(hashlib.sha256(instance_id.encode()).digest()[:8], "big")
    return (h % HELDOUT_FRACTION_MOD) < HELDOUT_CUT


P4_FILES = {
    "split.csv": ["task_id", "primary_prior", "predicted_boundary", "split"],
    "body_spectrum.csv": ["scope", "n_contexts", "participation_ratio", "rank_90", "rank_95", "rank_99", "top1_energy_frac"],
    "shadow_fve.csv": ["shadow", "shadow_dim", "fve_heldout"],
    "matched_dim_fve_curve.csv": ["k", "fve_heldout"],
    "per_lane_dimensionality.csv": ["lane", "n_contexts", "participation_ratio", "rank_95"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ARC Phase 4 body-resistance probe")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--register", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--split-mode", choices=["frozen_v2", "sha256_expansion"], default="sha256_expansion")
    p.add_argument("--limit-tasks", type=int, default=0)
    args = p.parse_args()
    if not args.dry_run and (not args.data_dir or not args.register):
        p.error("--data-dir and --register are required (except --dry-run)")
    return args


def base_manifest(args, out_dir: Path) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[3]
    git = git_state(repo_root, args.allow_dirty)
    spec_path = Path(__file__).resolve().parent / "PHASE4_BODY_RESISTANCE_SPEC.md"
    return {
        "generatedAt": iso_now(), "tool": "docs/prereg/arc/phase4_body_resistance.py",
        "command": [sys.executable, "docs/prereg/arc/phase4_body_resistance.py", *sys.argv[1:]],
        "gitCommit": git["commit"], "gitDirty": git["dirty"], "allowDirty": args.allow_dirty,
        "featureSchemaVersion": FEATURE_SCHEMA_VERSION, "protocolVersion": PROTOCOL_VERSION,
        "receiptSchemaVersion": RECEIPT_SCHEMA_VERSION, "learnerVersion": LEARNER_VERSION,
        "specPath": "docs/prereg/arc/PHASE4_BODY_RESISTANCE_SPEC.md",
        "specHash": (sha256_file(spec_path) if spec_path.exists() else "NA"),
        "runnerSha256": sha256_file(Path(__file__).resolve()),
        "repSha256": sha256_file(Path(__file__).resolve().parent / "phase3d_mask_target_v3.py"),
        "pythonVersion": sys.version, "platform": platform.platform(),
        "bodyDim": rep.RAW_GRID_DIM, "metadataShadowDim": rep.METADATA_DIM, "signatureShadowDim": rep.SIGNATURE_VECTOR_DIM,
        "thresholds": {"pr_high_min": PR_HIGH_MIN, "pr_marginal_max": PR_MARGINAL_MAX,
                       "fve_recon_ceiling": FVE_RECON_CEILING, "fve_marginal_min": FVE_MARGINAL_MIN,
                       "pr_bound_saturation_max": PR_BOUND_SATURATION_MAX,
                       "ridge_lambda": RIDGE_LAMBDA, "shadow_dim_k": SHADOW_DIM_K,
                       "heldout_rule": "sha256(instance_id)%10<3", "energy_levels": ENERGY_LEVELS, "pca_k_grid": PCA_K_GRID},
    }


def write_empty(out_dir: Path, manifest: dict[str, Any]) -> None:
    write_json(out_dir / "manifest.json", manifest)
    for fname, cols in P4_FILES.items():
        write_csv(out_dir / fname, [], cols)
    write_json(out_dir / "phase4_body_resistance_receipt.json", {"manifest": manifest, "branch": None})
    (out_dir / "branch_adjudication.md").write_text("# ARC Phase 4 body-resistance\n\nDry run / empty receipt.\n", encoding="utf-8")
    (out_dir / "commands.md").write_text("# ARC Phase 4 commands\n\nDry run / empty receipt.\n", encoding="utf-8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = base_manifest(args, out_dir)

    if args.dry_run:
        manifest["mode"] = "dry_run"; manifest["completedAt"] = iso_now()
        write_empty(out_dir, manifest)
        print(f"ARC Phase 4 dry run wrote {out_dir}")
        return 0

    data_dir = Path(args.data_dir).resolve()
    register_path = Path(args.register).resolve()
    assert_training_data_dir(data_dir)
    tasks, register_hash, data_hash = load_tasks(data_dir, register_path, args.split_mode)
    if args.limit_tasks > 0:
        tasks = tasks[: args.limit_tasks]
    manifest["registerHash"] = register_hash; manifest["dataDirHash"] = data_hash; manifest["splitMode"] = args.split_mode

    validation = [t for t in tasks if t.split == "validation"]
    test = [t for t in tasks if t.split == "test"]
    lane_instances = {
        "validation_lodo": build_lodo_instances(validation, "validation_lodo"),
        "validation_pttest": build_pttest_instances(validation, "validation_pttest"),
        "test_lodo": build_lodo_instances(test, "test_lodo"),
        "pttest": build_pttest_instances(test, "pttest"),
    }
    contexts = [i for ln in LANES for i in lane_instances[ln]]
    write_csv(out_dir / "split.csv", [{"task_id": t.task_id, "primary_prior": t.primary_prior, "predicted_boundary": t.predicted_boundary, "split": t.split} for t in sorted(tasks, key=lambda x: x.task_id)], P4_FILES["split.csv"])

    # ---- build body + shadow matrices (one row per context, query grid) ----
    B = torch.tensor([body_vec(c.query_input) for c in contexts], dtype=torch.float32)
    S_meta = torch.tensor([metadata_shadow_vec(c.query_input) for c in contexts], dtype=torch.float32)
    S_sig = torch.tensor([signature_shadow_vec(c.query_input) for c in contexts], dtype=torch.float32)
    held = torch.tensor([is_heldout(c.instance_id) for c in contexts], dtype=torch.bool)
    tr = ~held
    n = len(contexts)
    manifest["contextUniverse"] = {ln: len(lane_instances[ln]) for ln in LANES}
    manifest["nContexts"] = n
    manifest["nHeldout"] = int(held.sum())

    # ---- body spectrum / PR (full set) ----
    Bc = (B - B.mean(dim=0, keepdim=True)).double()
    sv = torch.linalg.svdvals(Bc)
    pr = participation_ratio(sv)
    eranks = energy_ranks(sv, ENERGY_LEVELS)
    top1 = float((sv.double()[0] ** 2) / (sv.double() ** 2).sum()) if n else 0.0
    write_csv(out_dir / "body_spectrum.csv", [{"scope": "U_all", "n_contexts": n, "participation_ratio": round_float(pr),
                                               "rank_90": eranks["rank_90"], "rank_95": eranks["rank_95"], "rank_99": eranks["rank_99"],
                                               "top1_energy_frac": round_float(top1)}], P4_FILES["body_spectrum.csv"])

    # ---- per-lane PR ----
    per_lane = []
    for ln in LANES:
        idx = [j for j, c in enumerate(contexts) if c.lane == ln]
        if len(idx) < 2:
            per_lane.append({"lane": ln, "n_contexts": len(idx), "participation_ratio": 0.0, "rank_95": 0}); continue
        Bl = B[idx]; Blc = (Bl - Bl.mean(dim=0, keepdim=True)).double()
        svl = torch.linalg.svdvals(Blc)
        per_lane.append({"lane": ln, "n_contexts": len(idx), "participation_ratio": round_float(participation_ratio(svl)),
                         "rank_95": energy_ranks(svl, [0.95])["rank_95"]})
    write_csv(out_dir / "per_lane_dimensionality.csv", per_lane, P4_FILES["per_lane_dimensionality.csv"])

    # ---- FVE(body | shadow) held-out ----
    shadow_rows = []
    for name, S, dim in [("metadata_only", S_meta, rep.METADATA_DIM), ("signature_palette", S_sig, rep.SIGNATURE_VECTOR_DIM)]:
        fve = fve_ridge(B[tr], S[tr], B[held], S[held], RIDGE_LAMBDA)
        shadow_rows.append({"shadow": name, "shadow_dim": dim, "fve_heldout": round_float(fve)})
    write_csv(out_dir / "shadow_fve.csv", shadow_rows, P4_FILES["shadow_fve.csv"])

    # ---- matched-dim PCA FVE curve (held-out) ----
    curve = []
    fve_matched = 0.0
    for k in PCA_K_GRID:
        f = fve_pca(B[tr], B[held], k)
        curve.append({"k": k, "fve_heldout": round_float(f)})
        if k == SHADOW_DIM_K:
            fve_matched = f
    write_csv(out_dir / "matched_dim_fve_curve.csv", curve, P4_FILES["matched_dim_fve_curve.csv"])

    # ---- branch adjudication ----
    pr_bound = float(min(B.shape[1], n - 1))           # PR sample/feature bound
    pr_bound_ratio = (pr / pr_bound) if pr_bound > 0 else 0.0
    saturated = pr_bound_ratio > PR_BOUND_SATURATION_MAX
    if saturated:
        branch = "arc_body_inconclusive"
        reason = f"Body PR {pr:.1f} is {pr_bound_ratio:.2f} of the sample bound {pr_bound:.0f} (> {PR_BOUND_SATURATION_MAX}): the spectrum is near-isotropic at this register size, so the high-dim reading is sample-saturated (caveat 3)."
    elif pr >= PR_HIGH_MIN and fve_matched <= FVE_RECON_CEILING:
        branch = "arc_body_high_dim"
        reason = f"Body PR {pr:.1f} >= {PR_HIGH_MIN} ({pr_bound_ratio:.2f} of sample bound {pr_bound:.0f}) and matched-dim FVE(top-{SHADOW_DIM_K} PCA) {fve_matched:.3f} <= {FVE_RECON_CEILING}: the ARC grid body is genuinely high-dimensional (read-off; control-shadow question open)."
    elif pr <= PR_MARGINAL_MAX or fve_matched >= FVE_MARGINAL_MIN:
        branch = "arc_body_marginal"
        reason = f"Body PR {pr:.1f} <= {PR_MARGINAL_MAX} or matched-dim FVE {fve_matched:.3f} >= {FVE_MARGINAL_MIN}: the body collapses to low dimension like the marginal control substrates."
    else:
        branch = "arc_body_inconclusive"
        reason = f"Body PR {pr:.1f} and matched-dim FVE {fve_matched:.3f} fall between the frozen thresholds; the dimensionality reading is ambiguous at this register size (n={n})."

    manifest["completedAt"] = iso_now()
    manifest["bodyResistance"] = {
        "participation_ratio": round_float(pr), "energy_ranks": eranks, "top1_energy_frac": round_float(top1),
        "pr_sample_bound": pr_bound, "pr_bound_ratio": round_float(pr_bound_ratio), "sample_saturated": saturated,
        "fve_metadata_shadow": shadow_rows[0]["fve_heldout"], "fve_signature_shadow": shadow_rows[1]["fve_heldout"],
        "fve_matched_dim_pca": round_float(fve_matched), "matched_dim_k": SHADOW_DIM_K,
    }
    manifest["branch"] = branch
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "phase4_body_resistance_receipt.json", {"manifest": manifest, "branch": branch, "branchReason": reason})
    (out_dir / "branch_adjudication.md").write_text(
        f"# ARC Phase 4 -- Body-Resistance -- Branch Adjudication\n\n**Branch: `{branch}`**\n\n{reason}\n\n"
        f"- n_contexts {n} (held-out {int(held.sum())}); body dim {rep.RAW_GRID_DIM}\n"
        f"- body participation ratio: **{pr:.2f}** (marginal substrates ~ 2); {pr_bound_ratio:.2f} of sample bound {pr_bound:.0f} "
        f"(saturation guard {PR_BOUND_SATURATION_MAX}); top-1 energy {top1:.3f}; "
        f"energy ranks 90/95/99 = {eranks['rank_90']}/{eranks['rank_95']}/{eranks['rank_99']}\n"
        f"- matched-dim FVE(body | top-{SHADOW_DIM_K} PCA) held-out: **{fve_matched:.3f}** (high_dim if <= {FVE_RECON_CEILING})\n"
        f"- coarse shadow FVE held-out: metadata {shadow_rows[0]['fve_heldout']:.3f}, signature {shadow_rows[1]['fve_heldout']:.3f} (baselines per caveat 2)\n\n"
        f"Read-off dimensionality only -- NOT a control witness, sufficiency proof, ARC solve, or eval/Kaggle claim.\n",
        encoding="utf-8")
    (out_dir / "commands.md").write_text(
        "# ARC Phase 4 commands\n\n```\nnode scripts/arc-phase4-body-resistance.mjs \\\n"
        "  --data-dir \"$env:USERPROFILE/Datasets/ARC-AGI-2/data\" \\\n"
        "  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv \\\n"
        "  --split-mode sha256_expansion \\\n"
        "  --out results/arc/phase4-body-resistance\n```\n", encoding="utf-8")
    write_json(out_dir / "hashes.json", hash_receipt_files(out_dir))
    print(f"ARC Phase 4 body-resistance wrote {out_dir}")
    print(f"Branch: {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
