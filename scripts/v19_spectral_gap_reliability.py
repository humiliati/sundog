#!/usr/bin/env python3
"""v0.19 liao2021 Floquet spectral-gap reliability mechanism runner.

Locked by docs/isotrophy/kfacet/kfacet_v19_spectral_gap_reliability_form.md.

First-principles MECHANISM chapter (not new evidence). Re-measures the exact v0.18
2880-row sample, reproducing v0.18's vf/score/frame_spread bit-for-bit (gate), and adds
the identity-frame Floquet spectrum + re_gap. Tests H1 (per-orbit re_gap -> frame_spread,
label-blind), H2 (per-cell gap_reliability -> AUC_cell, confirmatory), with a pinned
falsification guard. H3 (spectral type vs vf/stability) is report-only.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v07a_velocity_fraction_audit as v07a  # type: ignore
import scripts.v13a_liao2021_preflight as v13a  # type: ignore
import scripts.v14_liao2021_sampled_transfer as v14  # type: ignore
import scripts.v16_liao2021_tail_resolved_transfer as v16  # type: ignore

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v19_spectral_gap_reliability_form.md"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v19-spectral-gap-reliability"
DEFAULT_V18 = ROOT / "results/isotrophy/k-facet-v18-liao2021-reliability-auc"

SEED = 20260523
SHARD_COUNT = 16
PERMUTATIONS = 100000
EXPECTED_TARGET_SHA = "9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4"
FRAME_DEGREES = v16.FRAME_DEGREES
DEGEN_THRESH = v07a.DEGENERACY_THRESHOLD_REAL_PART

H1_RHO_BAR = -0.45
H1_P_THRESHOLD = 0.01
H2_RHO_BAR = 0.45
H2_P_THRESHOLD = 0.05
SPREAD_HIGH = 0.10
GAP_TERCILE = 2.0 / 3.0
FALSIFIER_MAX_FRACTION = 0.05
REPRODUCE_TOL = 1e-9

VERDICT_PREPARED = "spectral_gap_prepared_not_measured"
VERDICT_MERGED = "spectral_gap_merged_not_analyzed"

SPECTRUM_FIELDS = [
    "re_gap", "gap_reliability", "max_abs_lambda", "selected_re", "selected_im",
    "degenerate_count", "eigenspace_dim", "selected_group_re_width", "selection_sv_gap",
    "eigs_real", "eigs_imag",
]
PER_ROW_FIELDS = list(v16.PER_ROW_FIELDS) + SPECTRUM_FIELDS


def operator_commands(out: Path) -> str:
    return "\n".join([
        "# v0.19 Spectral-Gap Reliability Mechanism Commands",
        "",
        "Estimated wall-clock: about 5 hours for the 16 shards on the project machine.",
        "Decision readback: `results/isotrophy/k-facet-v19-spectral-gap-reliability/manifest.json`",
        "Primary receipts: `reproduce_v18_check.json`, `gap_frame_spread_test.json`,",
        "`per_cell_gap_reliability.csv`, `h2_gap_auc_test.json`, `spectral_type_panel.csv`.",
        "",
        "```powershell",
        "npm run isotrophy:v19:prepare",
        "foreach ($i in 0..15) {",
        "  npm run isotrophy:v19:shard -- --shard-index $i --shard-count 16",
        "}",
        "npm run isotrophy:v19:merge",
        "npm run isotrophy:v19:analyze",
        "```",
        "",
        "The shard loop is resume-safe at the file level: completed",
        "`sample_shard_XX_of_16.csv` files can be left in place and missing shard indexes",
        "rerun individually before merge.",
        "",
        f"Output directory: `{v14.relpath(out)}`",
        "",
    ]) + "\n"


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return None


def resolve_dir(text: str) -> Path:
    p = Path(text)
    return p if p.is_absolute() else ROOT / p


def check_prior_receipts(target_path: Path, target_sha: str, v18_dir: Path) -> dict:
    if target_sha != EXPECTED_TARGET_SHA:
        raise SystemExit(f"target SHA mismatch: {target_sha} != locked {EXPECTED_TARGET_SHA}")
    v13 = v14.check_prior_receipts(target_path, target_sha)
    man = v14.read_json(v18_dir / "manifest.json")
    if man.get("verdict") != "reliability_drives_per_cell_auc_supported":
        raise SystemExit(f"v0.18 verdict must be reliability_drives_per_cell_auc_supported, got {man.get('verdict')}")
    if man.get("target_sha256") != target_sha:
        raise SystemExit(f"v0.18 target SHA mismatch: {man.get('target_sha256')} != {target_sha}")
    for fn in ("sample_frame.csv", "per_row_sample.csv", "per_cell_auc.csv"):
        if not (v18_dir / fn).exists():
            raise SystemExit(f"missing v0.18 {fn}")
    checks = dict(v13.get("checks", {}))
    checks.update({
        "v18_verdict_reliability_supported": man.get("verdict") == "reliability_drives_per_cell_auc_supported",
        "v18_target_sha_matches": man.get("target_sha256") == target_sha,
        "v18_per_row_available": (v18_dir / "per_row_sample.csv").exists(),
        "v18_per_cell_auc_available": (v18_dir / "per_cell_auc.csv").exists(),
    })
    if not all(checks.values()):
        failed = [k for k, ok in checks.items() if not ok]
        raise SystemExit("prior receipt check(s) failed: " + ", ".join(failed))
    return {
        "target_slug": v14.TARGET_SLUG, "target_tier": v14.TARGET_TIER,
        "target_path": v14.relpath(target_path), "target_sha256": target_sha,
        "checks": checks, "v13_context": v13,
        "v18_context": {"path": v14.relpath(v18_dir), "verdict": man.get("verdict"),
                        "AUC_cond": man.get("AUC_cond"), "reliability_rho": man.get("reliability_rho")},
    }


def identity_spectrum(masses: np.ndarray, x0: np.ndarray, v0: np.ndarray, period: float) -> dict:
    """Identity-frame monodromy spectrum + re_gap, replicating select_gamma_1 grouping."""
    integrated = v13a.integrate_liao2021_state(masses, x0, v0, period)
    M = v07a.compute_monodromy_vectorized(integrated, v13a.RTOL, v13a.ATOL, v13a.MAX_STEP_FRACTION)
    eigvals, eigvecs = np.linalg.eig(M)
    real_parts = eigvals.real
    max_real = float(real_parts.max())
    sel = v07a.select_gamma_1(M, masses)

    group_mask = real_parts >= max_real - DEGEN_THRESH
    group_re = real_parts[group_mask]
    non_group = real_parts[~group_mask]
    re_gap = float(max_real - non_group.max()) if non_group.size else 0.0
    gap_reliability = math.log10(max(re_gap, 1e-9))
    selected_group_re_width = float(group_re.max() - group_re.min())

    sv_gap = None
    if int(sel["eigenspace_dim_used"]) > 1:
        cols: list[np.ndarray] = []
        for idx in np.where(group_mask)[0]:
            vec = eigvecs[:, idx]
            cols.append(vec.real)
            if not np.allclose(vec.imag, 0.0, atol=1e-10):
                cols.append(vec.imag)
        raw = np.column_stack(cols)
        u_full, s_vals, _ = np.linalg.svd(raw, full_matrices=False)
        rank = int(np.sum(s_vals > 1e-10))
        basis = u_full[:, :rank]
        vpm = basis[9:, :] * np.sqrt(np.repeat(masses, 3))[:, None]
        _, s_proj, _ = np.linalg.svd(vpm, full_matrices=False)
        if s_proj.size >= 2:
            sv_gap = float(s_proj[0] - s_proj[1])
        elif s_proj.size == 1:
            sv_gap = float(s_proj[0])

    return {
        "re_gap": re_gap, "gap_reliability": gap_reliability,
        "max_abs_lambda": float(np.max(np.abs(eigvals))),
        "selected_re": sel["eigenvalue_real"], "selected_im": sel["eigenvalue_imag"],
        "degenerate_count": int(sel["degenerate_eigenvalue_count"]),
        "eigenspace_dim": int(sel["eigenspace_dim_used"]),
        "selected_group_re_width": selected_group_re_width,
        "selection_sv_gap": sv_gap,
        "eigs_real": ";".join(f"{r:.10g}" for r in real_parts),
        "eigs_imag": ";".join(f"{im:.10g}" for im in eigvals.imag),
    }


def command_prepare(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.19 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    started = v14.utc_now()
    target = v14.resolve_target(args.target)
    v18_dir = resolve_dir(args.v18)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target_sha = v14.sha256_file(target)
    receipts = check_prior_receipts(target, target_sha, v18_dir)

    v18_sample = v14.read_csv(v18_dir / "sample_frame.csv")
    v14.write_csv(out / "sample_frame.csv", v18_sample, list(v18_sample[0].keys()))
    manifest = {
        "schema": "sundog.isotrophy.v0.19-spectral-gap-reliability.v1",
        "form_lock": FORM_LOCK, "stage": "prepared", "verdict": VERDICT_PREPARED,
        "startedAt": started, "completedAt": v14.utc_now(), "gitCommit": git_commit(),
        "target_slug": v14.TARGET_SLUG, "target_tier": v14.TARGET_TIER,
        "target_path": v14.relpath(target), "target_sha256": target_sha,
        "prior_receipts": receipts, "v18_path": v14.relpath(v18_dir),
        "sample_provenance": "v0.18_remeasure", "sample_rows_requested": len(v18_sample),
        "sample_rows_success": None, "shard_count": args.shard_count, "sample_seed": SEED,
        "frame_degrees": FRAME_DEGREES, "score": "median(vf_0,vf_37,vf_90,vf_211)",
    }
    v14.write_json(out / "manifest.json", manifest)
    (out / "operator_commands.md").write_text(operator_commands(out), encoding="utf-8")
    print(f"[v19:prepare] re-measure target = {len(v18_sample)} v0.18 rows; provenance v0.18_remeasure")
    print(f"[v19:prepare] wrote {v14.relpath(out / 'sample_frame.csv')}")


def command_shard(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.19 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise SystemExit("--shard-index must be in [0, --shard-count)")
    out = Path(args.out)
    manifest = v14.read_json(out / "manifest.json")
    target = ROOT / manifest["target_path"]
    rows = v14.parse_liao2021(target)
    row_by_index = {row.index: row for row in rows}
    sample = v14.read_csv(out / "sample_frame.csv")
    shard_sample = [r for r in sample if int(r["sample_ordinal"]) % args.shard_count == args.shard_index]

    records = []
    started = time.perf_counter()
    for pos, sample_rec in enumerate(shard_sample, start=1):
        idx = int(sample_rec["orbit_index"])
        row = row_by_index[idx]
        masses, x0, v0 = v14.expand_liao2021_state(row, center_com=True)
        meas = v16.measure_ensemble(masses, x0, v0, row.period)  # frozen; reproduces v0.18
        if meas["status"] == "success":
            try:
                spec = identity_spectrum(masses, x0, v0, row.period)
            except Exception as exc:  # noqa: BLE001
                meas = dict(meas)
                meas["status"] = "integration_blocked"
                meas["integration_error_stage"] = "identity_spectrum"
                meas["integration_error_message"] = f"{type(exc).__name__}: {exc}"
                spec = {k: None for k in SPECTRUM_FIELDS}
        else:
            spec = {k: None for k in SPECTRUM_FIELDS}
        records.append({**sample_rec, **meas, **spec})
        print(f"[v19:shard {args.shard_index}/{args.shard_count}] {pos}/{len(shard_sample)} "
              f"idx={idx} status={meas['status']} re_gap={spec.get('re_gap')} "
              f"t={float(meas['total_seconds']):.1f}s", flush=True)

    sample_path = out / f"sample_shard_{args.shard_index:02d}_of_{args.shard_count}.csv"
    v14.write_csv(sample_path, records, PER_ROW_FIELDS)
    v14.write_json(out / f"shard_{args.shard_index:02d}_of_{args.shard_count}.json", {
        "schema": "sundog.isotrophy.v0.19-shard.v1", "shard_index": args.shard_index,
        "shard_count": args.shard_count, "rows": len(records),
        "success": sum(1 for r in records if r["status"] == "success"),
        "wall_seconds": round(time.perf_counter() - started, 3), "completedAt": v14.utc_now(),
    })
    print(f"[v19:shard] wrote {v14.relpath(sample_path)}")


def command_merge(args: argparse.Namespace) -> None:
    if args.shard_count != SHARD_COUNT:
        raise SystemExit(f"v0.19 shard count is locked at {SHARD_COUNT}; got {args.shard_count}")
    out = Path(args.out)
    sample = v14.read_csv(out / "sample_frame.csv")
    shard_rows: list[dict] = []
    missing = []
    for i in range(args.shard_count):
        p = out / f"sample_shard_{i:02d}_of_{args.shard_count}.csv"
        if not p.exists():
            missing.append(v14.relpath(p))
            continue
        shard_rows.extend(v14.read_csv(p))
    if missing:
        raise SystemExit("missing shard output(s): " + ", ".join(missing))
    expected = {int(r["sample_ordinal"]) for r in sample}
    got = {int(r["sample_ordinal"]) for r in shard_rows}
    if expected != got:
        raise SystemExit(f"merged ordinals mismatch; missing={sorted(expected-got)[:10]} extra={sorted(got-expected)[:10]}")
    shard_rows.sort(key=lambda r: int(r["sample_ordinal"]))
    v14.write_csv(out / "per_row_spectrum.csv", shard_rows, PER_ROW_FIELDS)

    n = len(shard_rows)
    success = sum(1 for r in shard_rows if r["status"] == "success")
    blocked = sum(1 for r in shard_rows if r["status"] == "integration_blocked")
    sanity = sum(1 for r in shard_rows if r["status"] == "sanity_failed")
    lo, hi = v14.wilson_ci(blocked + sanity, n)
    manifest = v14.read_json(out / "manifest.json")
    manifest.update({
        "stage": "merged", "verdict": VERDICT_MERGED, "completedAt": v14.utc_now(),
        "sample_rows_requested": n, "sample_rows_success": success,
        "integration_blocked": blocked, "sanity_failed": sanity,
        "attrition_fraction": (blocked + sanity) / n if n else 0.0,
        "attrition_wilson95_low": lo, "attrition_wilson95_high": hi,
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v19:merge] rows={n} success={success} blocked={blocked} sanity={sanity}")


def reproduce_v18_check(rows: list[dict], v18_dir: Path) -> dict:
    v18 = {int(r["sample_ordinal"]): r for r in v14.read_csv(v18_dir / "per_row_sample.csv")}
    cols = ["vf_0", "vf_37", "vf_90", "vf_211", "score", "frame_spread"]
    max_diff = 0.0
    compared = 0
    missing_reference_count = 0
    identity_mismatch_count = 0
    status_mismatch_count = 0
    value_missing_count = 0
    for r in rows:
        o = int(r["sample_ordinal"])
        ref = v18.get(o)
        if ref is None:
            missing_reference_count += 1
            continue
        if (
            str(r.get("orbit_index")) != str(ref.get("orbit_index"))
            or str(r.get("mass_cell")) != str(ref.get("mass_cell"))
            or str(r.get("stability")) != str(ref.get("stability"))
        ):
            identity_mismatch_count += 1
        if str(r.get("status")) != str(ref.get("status")):
            status_mismatch_count += 1
            continue
        if r.get("status") != "success":
            continue
        compared += 1
        for c in cols:
            if r.get(c, "") == "" or ref.get(c, "") == "":
                value_missing_count += 1
                continue
            max_diff = max(max_diff, abs(float(r[c]) - float(ref[c])))
    expected_success = sum(1 for r in v18.values() if r.get("status") == "success")
    unexpected_reference_count = len(set(v18) - {int(r["sample_ordinal"]) for r in rows})
    passes = (
        len(rows) == len(v18)
        and compared == expected_success
        and missing_reference_count == 0
        and unexpected_reference_count == 0
        and identity_mismatch_count == 0
        and status_mismatch_count == 0
        and value_missing_count == 0
        and max_diff <= REPRODUCE_TOL
    )
    return {
        "rows": len(rows),
        "expected_rows": len(v18),
        "compared": compared,
        "expected_success": expected_success,
        "max_abs_diff": max_diff,
        "missing_reference_count": missing_reference_count,
        "unexpected_reference_count": unexpected_reference_count,
        "identity_mismatch_count": identity_mismatch_count,
        "status_mismatch_count": status_mismatch_count,
        "value_missing_count": value_missing_count,
        "passes": passes,
    }


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = v16.midranks(np.asarray(x, dtype=float))
    ry = v16.midranks(np.asarray(y, dtype=float))
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def spearman_perm(x: list[float], y: list[float], perms: int, seed: int, tail: str) -> dict:
    rx = v16.midranks(np.asarray(x, dtype=float))
    ry = v16.midranks(np.asarray(y, dtype=float))
    rho_obs = spearman(x, y)
    if rho_obs is None:
        return {"rho": None, "p": None, "ge_or_le": None, "permutations": perms, "n": len(x)}
    rxc = rx - rx.mean()
    dx = math.sqrt(float((rxc ** 2).sum()))
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(perms):
        perm = ry[rng.permutation(len(ry))]
        pc = perm - perm.mean()
        denom = dx * math.sqrt(float((pc ** 2).sum()))
        rho = float(rxc @ pc) / denom if denom else 0.0
        if tail == "le":
            cnt += int(rho <= rho_obs + 1e-12)
        else:
            cnt += int(rho >= rho_obs - 1e-12)
    return {"rho": rho_obs, "p": (cnt + 1) / (perms + 1), "ge_or_le": cnt,
            "permutations": perms, "n": len(x)}


def command_analyze(args: argparse.Namespace) -> None:
    if args.permutations != PERMUTATIONS:
        raise SystemExit(f"v0.19 permutation count is locked at {PERMUTATIONS}; got {args.permutations}")
    if args.seed != SEED:
        raise SystemExit(f"v0.19 permutation seed is locked at {SEED}; got {args.seed}")
    out = Path(args.out)
    manifest = v14.read_json(out / "manifest.json")
    v18_dir = resolve_dir(manifest.get("v18_path", str(DEFAULT_V18)))
    rows = v14.read_csv(out / "per_row_spectrum.csv")

    reproduce = reproduce_v18_check(rows, v18_dir)

    success = [r for r in rows if r["status"] == "success" and r.get("re_gap", "") not in ("", "None")
               and r.get("frame_spread", "") != "" and r["stability"] in ("S", "U")]
    re_gap = [float(r["re_gap"]) for r in success]
    frame_spread = [float(r["frame_spread"]) for r in success]
    gap_reliability = [float(r["gap_reliability"]) for r in success]

    # H1: per-orbit re_gap -> frame_spread (negative)
    h1 = spearman_perm(re_gap, frame_spread, args.permutations, args.seed, tail="le")
    h1_pass = bool(h1["rho"] is not None and h1["rho"] <= H1_RHO_BAR and h1["p"] is not None and h1["p"] <= H1_P_THRESHOLD)

    # Falsifier: top-tercile re_gap AND frame_spread >= SPREAD_HIGH
    gap_clear = float(np.quantile(np.asarray(re_gap), GAP_TERCILE))
    box = [r for r in success if float(r["re_gap"]) >= gap_clear and float(r["frame_spread"]) >= SPREAD_HIGH]
    falsifier_fraction = len(box) / len(success) if success else 0.0

    # H2: per-cell median gap_reliability -> AUC_cell (positive); AUC_cell from v0.18 per_cell_auc
    v18_auc = {r["mass_cell"]: float(r["AUC_cell"]) for r in v14.read_csv(v18_dir / "per_cell_auc.csv")
               if v14.parse_bool(r["primary"]) and r.get("AUC_cell", "") != ""}
    by_cell: dict[str, list[float]] = defaultdict(list)
    for r in success:
        by_cell[r["mass_cell"]].append(float(r["gap_reliability"]))
    cells = sorted(set(by_cell) & set(v18_auc))
    cell_gap_rel = [float(np.median(by_cell[c])) for c in cells]
    cell_auc = [v18_auc[c] for c in cells]
    h2 = spearman_perm(cell_gap_rel, cell_auc, args.permutations, args.seed, tail="ge")
    h2_confirm = bool(h2["rho"] is not None and h2["rho"] >= H2_RHO_BAR and h2["p"] is not None and h2["p"] <= H2_P_THRESHOLD)

    # H3 report-only: spectral type vs vf/stability
    type_panel = []
    for r in success:
        mal = float(r["max_abs_lambda"])
        type_panel.append({"orbit_index": r["orbit_index"], "mass_cell": r["mass_cell"],
                           "stability": r["stability"], "spectral_type": "hyperbolic" if mal > 1.0 + 1e-6 else "elliptic",
                           "max_abs_lambda": mal, "re_gap": float(r["re_gap"]),
                           "vf_0": (float(r["vf_0"]) if r.get("vf_0", "") != "" else None),
                           "score": (float(r["score"]) if r.get("score", "") != "" else None)})

    if not reproduce["passes"]:
        verdict = "blocked_by_nonreproduction"
    elif not h1_pass:
        verdict = "spectral_gap_mechanism_not_supported"
    elif falsifier_fraction > FALSIFIER_MAX_FRACTION or not h2_confirm:
        verdict = "spectral_gap_mechanism_partial"
    else:
        verdict = "spectral_gap_explains_reliability"

    v14.write_json(out / "reproduce_v18_check.json", reproduce)
    v14.write_json(out / "gap_frame_spread_test.json", {
        **h1, "rho_bar": H1_RHO_BAR, "p_threshold": H1_P_THRESHOLD, "pass": h1_pass,
        "GAP_CLEAR": gap_clear, "SPREAD_HIGH": SPREAD_HIGH,
        "large_gap_high_spread_count": len(box), "large_gap_high_spread_fraction": falsifier_fraction,
        "falsifier_max_fraction": FALSIFIER_MAX_FRACTION,
    })
    v14.write_csv(out / "per_cell_gap_reliability.csv",
                  [{"mass_cell": c, "median_gap_reliability": float(np.median(by_cell[c])),
                    "AUC_cell": v18_auc[c], "n_orbits": len(by_cell[c])} for c in cells],
                  ["mass_cell", "median_gap_reliability", "AUC_cell", "n_orbits"])
    v14.write_json(out / "h2_gap_auc_test.json", {**h2, "rho_bar": H2_RHO_BAR,
                   "p_threshold": H2_P_THRESHOLD, "confirmatory_pass": h2_confirm, "k_cells": len(cells)})
    v14.write_csv(out / "spectral_type_panel.csv", type_panel,
                  ["orbit_index", "mass_cell", "stability", "spectral_type", "max_abs_lambda",
                   "re_gap", "vf_0", "score"])
    manifest.update({
        "stage": "analyzed", "verdict": verdict, "completedAt": v14.utc_now(),
        "reproduce_v18_max_abs_diff": reproduce["max_abs_diff"], "reproduce_v18_passes": reproduce["passes"],
        "H1_rho": h1["rho"], "H1_p": h1["p"], "H1_n": h1["n"], "H1_pass": h1_pass,
        "GAP_CLEAR": gap_clear, "SPREAD_HIGH": SPREAD_HIGH,
        "large_gap_high_spread_fraction": falsifier_fraction,
        "H2_rho": h2["rho"], "H2_p": h2["p"], "H2_confirmatory_pass": h2_confirm, "H2_k_cells": len(cells),
        "target_tier": v14.TARGET_TIER,
        "claim_boundary": ("Tier-2 / Li-Liao lineage / stable-support / within-cell / tail-resolved "
                           "score / not coarse-zone / not full-catalog / not Tier-3 / not theorem-facing"),
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v19:analyze] reproduce_v18 max_abs_diff={reproduce['max_abs_diff']:.2e} passes={reproduce['passes']}")
    print(f"[v19:analyze] H1 rho={h1['rho']} p={h1['p']} pass={h1_pass}")
    print(f"[v19:analyze] falsifier fraction={falsifier_fraction:.4f} (GAP_CLEAR={gap_clear:.4g})")
    print(f"[v19:analyze] H2 rho={h2['rho']} p={h2['p']} confirm={h2_confirm}")
    print(f"[v19:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="reference the exact v0.18 sample for re-measurement")
    p.add_argument("--target", default=str(v14.DEFAULT_TARGET))
    p.add_argument("--v18", default=str(DEFAULT_V18))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    p.set_defaults(func=command_prepare)
    s = sub.add_parser("shard", help="re-measure ensemble + dump identity-frame spectrum")
    s.add_argument("--out", default=str(DEFAULT_OUT))
    s.add_argument("--shard-index", type=int, required=True)
    s.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    s.set_defaults(func=command_shard)
    m = sub.add_parser("merge", help="merge shards + attrition")
    m.add_argument("--out", default=str(DEFAULT_OUT))
    m.add_argument("--shard-count", type=int, default=SHARD_COUNT)
    m.set_defaults(func=command_merge)
    a = sub.add_parser("analyze", help="reproduce gate + H1/H2 + falsifier + H3 panel")
    a.add_argument("--out", default=str(DEFAULT_OUT))
    a.add_argument("--permutations", type=int, default=PERMUTATIONS)
    a.add_argument("--seed", type=int, default=SEED)
    a.set_defaults(func=command_analyze)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
