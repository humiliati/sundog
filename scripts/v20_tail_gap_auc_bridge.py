#!/usr/bin/env python3
"""v0.20 liao2021 tail-gap AUC bridge runner (confirmatory re-analysis).

Locked by docs/isotrophy/kfacet/kfacet_v20_tail_gap_auc_bridge_form.md.

Confirmatory AGGREGATION chapter (NOT new evidence). Pure bookkeeping over the
existing v0.18 per_cell_auc.csv + v0.19 per_row_spectrum.csv receipts: no D5
integration, no monodromy computation, no sampling. It asks whether the direct
cell-level spectral-gap -> AUC bridge that v0.19's H2 missed with a MEDIAN gap
appears when the gap is read as a pre-registered low q10 TAIL. Same 18 cells,
same AUC: a pass can only say "the median H2 missed because the bridge is
tail-aggregated"; it cannot upgrade the Tier-2 evidence base, promote, or claim
a new vf->stability mechanism.

H1  (primary):  Spearman(tail_gap_reliability, AUC_cell), one-sided perm.
H1b (non-binding): rho_tail - rho_median_v19 >= 0.30 (auto-satisfied by H1).
H2  (coherence): Spearman(tail_gap_reliability, frame_reliability), one-sided perm.
H3  (report-only): min gap, near-gap fraction, q05/q20 sensitivity, leave-one-out.
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.v14_liao2021_sampled_transfer as v14  # type: ignore
import scripts.v16_liao2021_tail_resolved_transfer as v16  # type: ignore

FORM_LOCK = "docs/isotrophy/kfacet/kfacet_v20_tail_gap_auc_bridge_form.md"
DEFAULT_OUT = ROOT / "results/isotrophy/k-facet-v20-tail-gap-auc-bridge"
DEFAULT_V18 = ROOT / "results/isotrophy/k-facet-v18-liao2021-reliability-auc"
DEFAULT_V19 = ROOT / "results/isotrophy/k-facet-v19-spectral-gap-reliability"

SEED = 20260523
PERMUTATIONS = 100000
EXPECTED_TARGET_SHA = "9c06eedc41b537b2ed926217cdf149d9f808a21866148ea3d89b9eb3c6069fd4"
Q_PRIMARY = 0.10
Q05 = 0.05
Q20 = 0.20
EXPECTED_CELLS = 18
EXPECTED_SUCCESS_ROWS = 2880

H1_RHO_BAR = 0.45
H1_P_THRESHOLD = 0.05
H1B_DELTA_BAR = 0.30
RHO_MEDIAN_V19 = 0.0629514963880289  # locked v0.19 H2 median-gap rho
H2_RHO_BAR = 0.45
H2_P_THRESHOLD = 0.05

VERDICT_PREPARED = "tail_gap_bridge_prepared_not_analyzed"

CLAIM_BOUNDARY = (
    "confirmatory re-analysis only / same 18 cells + same AUC as v0.18-v0.19 / "
    "Tier-2 Li-Liao lineage / stable-support / within-cell / tail-resolved score / "
    "not fresh external evidence / not full-catalog prevalence / not Tier-3 / "
    "not theorem-facing / not an altitude-1 explanation of why vf -> stability"
)


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


def operator_commands(out: Path) -> str:
    return "\n".join([
        "# v0.20 Tail-Gap AUC Bridge Commands",
        "",
        "Confirmatory re-analysis of the existing v0.18 + v0.19 receipts.",
        "No D5 integration, no monodromy, no shards -- runs in seconds.",
        "",
        "```powershell",
        "npm run isotrophy:v20:prepare",
        "npm run isotrophy:v20:analyze",
        "```",
        "",
        "Decision readback: `results/isotrophy/k-facet-v20-tail-gap-auc-bridge/manifest.json`",
        "Primary receipts: `tail_gap_auc_test.json` (H1 + H1b + LOO + q05/q20),",
        "`tail_gap_frame_reliability_test.json` (H2), `per_cell_tail_gap_bridge.csv`,",
        "`sidecar_tail_sensitivity.csv`.",
        "",
        f"Output directory: `{v14.relpath(out)}`",
        "",
    ]) + "\n"


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


def check_receipts(target_sha: str, v18_dir: Path, v19_dir: Path) -> tuple[bool, dict, dict, dict]:
    v18 = v14.read_json(v18_dir / "manifest.json") if (v18_dir / "manifest.json").exists() else {}
    v19 = v14.read_json(v19_dir / "manifest.json") if (v19_dir / "manifest.json").exists() else {}
    checks = {
        "target_sha_matches_lock": target_sha == EXPECTED_TARGET_SHA,
        "v18_verdict_supported": v18.get("verdict") == "reliability_drives_per_cell_auc_supported",
        "v18_per_cell_auc_available": (v18_dir / "per_cell_auc.csv").exists(),
        "v19_verdict_partial": v19.get("verdict") == "spectral_gap_mechanism_partial",
        "v19_reproduce_passes": v19.get("reproduce_v18_passes") is True,
        "v19_h1_pass": v19.get("H1_pass") is True,
        "v19_h2_confirmatory_false": v19.get("H2_confirmatory_pass") is False,
        "v19_success_rows_2880": v19.get("sample_rows_success") == EXPECTED_SUCCESS_ROWS,
        "v19_k_cells_18": v19.get("H2_k_cells") == EXPECTED_CELLS,
        "v19_per_row_spectrum_available": (v19_dir / "per_row_spectrum.csv").exists(),
    }
    return all(checks.values()), checks, v18, v19


def command_prepare(args: argparse.Namespace) -> None:
    started = v14.utc_now()
    target = v14.resolve_target(args.target)
    v18_dir = resolve_dir(args.v18)
    v19_dir = resolve_dir(args.v19)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target_sha = v14.sha256_file(target)
    ok, checks, v18man, v19man = check_receipts(target_sha, v18_dir, v19_dir)
    if not ok:
        failed = [k for k, v in checks.items() if not v]
        raise SystemExit("v0.20 receipt gate(s) failed (blocked_by_receipt): " + ", ".join(failed))

    manifest = {
        "schema": "sundog.isotrophy.v0.20-tail-gap-auc-bridge.v1",
        "form_lock": FORM_LOCK, "stage": "prepared", "verdict": VERDICT_PREPARED,
        "startedAt": started, "completedAt": v14.utc_now(), "gitCommit": git_commit(),
        "target_slug": v14.TARGET_SLUG, "target_tier": v14.TARGET_TIER,
        "target_path": v14.relpath(target), "target_sha256": target_sha,
        "v18_path": v14.relpath(v18_dir), "v19_path": v14.relpath(v19_dir),
        "v18_verdict": v18man.get("verdict"), "v19_verdict": v19man.get("verdict"),
        "receipt_checks": checks, "receipts_ok": ok,
        "q_primary": Q_PRIMARY, "seed": SEED, "permutations": PERMUTATIONS,
        "rho_median_v19": RHO_MEDIAN_V19,
    }
    v14.write_json(out / "manifest.json", manifest)
    (out / "operator_commands.md").write_text(operator_commands(out), encoding="utf-8")
    print(f"[v20:prepare] receipts OK ({len(checks)}/{len(checks)} gate checks); "
          f"q={Q_PRIMARY} seed={SEED} perms={PERMUTATIONS}")
    print(f"[v20:prepare] wrote {v14.relpath(out / 'manifest.json')}")


def command_analyze(args: argparse.Namespace) -> None:
    if args.permutations != PERMUTATIONS:
        raise SystemExit(f"v0.20 permutation count is locked at {PERMUTATIONS}; got {args.permutations}")
    if args.seed != SEED:
        raise SystemExit(f"v0.20 permutation seed is locked at {SEED}; got {args.seed}")
    out = Path(args.out)
    manifest = v14.read_json(out / "manifest.json")
    v18_dir = resolve_dir(manifest.get("v18_path", str(DEFAULT_V18)))
    v19_dir = resolve_dir(manifest.get("v19_path", str(DEFAULT_V19)))
    target = ROOT / manifest["target_path"]
    target_sha = v14.sha256_file(target)
    ok, checks, _v18man, _v19man = check_receipts(target_sha, v18_dir, v19_dir)

    # v0.18 per-cell AUC + frame_p90 for primary cells
    cell_meta: dict[str, dict] = {}
    for r in v14.read_csv(v18_dir / "per_cell_auc.csv"):
        if not v14.parse_bool(r["primary"]):
            continue
        if r.get("AUC_cell", "") == "" or r.get("frame_p90", "") == "":
            continue
        cell_meta[r["mass_cell"]] = {"AUC_cell": float(r["AUC_cell"]),
                                     "frame_p90": float(r["frame_p90"])}

    # v0.19 per-row re_gap grouped by cell (success rows only)
    by_cell: dict[str, list[float]] = defaultdict(list)
    all_re_gap: list[float] = []
    for r in v14.read_csv(v19_dir / "per_row_spectrum.csv"):
        if r.get("status") != "success":
            continue
        rg = r.get("re_gap", "")
        if rg in ("", "None"):
            continue
        val = float(rg)
        by_cell[r["mass_cell"]].append(val)
        all_re_gap.append(val)

    cells = sorted(set(by_cell) & set(cell_meta))
    cells_ok = len(cells) == EXPECTED_CELLS
    checks["analysis_18_cells"] = cells_ok
    receipts_ok = ok and cells_ok

    global_q10 = float(np.quantile(np.asarray(all_re_gap), Q_PRIMARY)) if all_re_gap else float("nan")

    per_cell_rows: list[dict] = []
    sidecar_rows: list[dict] = []
    for c in cells:
        gaps = np.asarray(by_cell[c], dtype=float)
        q10 = float(np.quantile(gaps, Q_PRIMARY))
        tail_gap_reliability = math.log10(max(q10, 1e-9))
        frame_p90 = cell_meta[c]["frame_p90"]
        frame_reliability = -math.log10(frame_p90 + 1e-9)
        auc_c = cell_meta[c]["AUC_cell"]
        per_cell_rows.append({
            "mass_cell": c, "n_orbits": int(gaps.size),
            "cell_gap_q10": q10, "tail_gap_reliability": tail_gap_reliability,
            "AUC_cell": auc_c, "frame_p90": frame_p90, "frame_reliability": frame_reliability,
        })
        q05v = float(np.quantile(gaps, Q05))
        q20v = float(np.quantile(gaps, Q20))
        sidecar_rows.append({
            "mass_cell": c, "n_orbits": int(gaps.size), "AUC_cell": auc_c,
            "frame_p90": frame_p90, "frame_reliability": frame_reliability,
            "tail_gap_reliability_q10": tail_gap_reliability,
            "tail_gap_reliability_q05": math.log10(max(q05v, 1e-9)),
            "tail_gap_reliability_q20": math.log10(max(q20v, 1e-9)),
            "min_gap_reliability": math.log10(max(float(gaps.min()), 1e-9)),
            "near_gap_fraction": float(np.mean(gaps <= global_q10)),
        })

    tail = [r["tail_gap_reliability"] for r in per_cell_rows]
    auc = [r["AUC_cell"] for r in per_cell_rows]
    frame_rel = [r["frame_reliability"] for r in per_cell_rows]

    # H1 (primary): tail_gap_reliability -> AUC_cell, positive
    h1 = spearman_perm(tail, auc, args.permutations, args.seed, tail="ge")
    h1_pass = bool(h1["rho"] is not None and h1["rho"] >= H1_RHO_BAR
                   and h1["p"] is not None and h1["p"] <= H1_P_THRESHOLD)

    # H1b (non-binding improvement guard)
    h1b_delta = (h1["rho"] - RHO_MEDIAN_V19) if h1["rho"] is not None else None
    h1b_pass = bool(h1b_delta is not None and h1b_delta >= H1B_DELTA_BAR)

    # H2 (coherence): tail_gap_reliability -> frame_reliability, positive
    h2 = spearman_perm(tail, frame_rel, args.permutations, args.seed, tail="ge")
    h2_pass = bool(h2["rho"] is not None and h2["rho"] >= H2_RHO_BAR
                   and h2["p"] is not None and h2["p"] <= H2_P_THRESHOLD)

    # H3 report-only: q05/q20 sensitivity of the H1 correlation
    tail_q05 = [r["tail_gap_reliability_q05"] for r in sidecar_rows]
    tail_q20 = [r["tail_gap_reliability_q20"] for r in sidecar_rows]
    rho_q05 = spearman(tail_q05, auc)
    rho_q20 = spearman(tail_q20, auc)

    # H3 report-only: leave-one-cell-out robustness of H1
    loo: list[float] = []
    for i in range(len(cells)):
        xt = [tail[j] for j in range(len(cells)) if j != i]
        ya = [auc[j] for j in range(len(cells)) if j != i]
        rv = spearman(xt, ya)
        if rv is not None:
            loo.append(rv)
    loo_min = min(loo) if loo else None
    loo_med = float(np.median(loo)) if loo else None
    loo_max = max(loo) if loo else None

    # verdict tree (locked)
    if not receipts_ok:
        verdict = "blocked_by_receipt"
    elif h1_pass and h1b_pass and h2_pass:
        verdict = "tail_gap_bridge_supported_confirmatory"
    elif (h1_pass and not (h1b_pass and h2_pass)) or (h2_pass and not h1_pass):
        verdict = "tail_gap_bridge_partial"
    elif (not h1_pass) and (not h2_pass):
        verdict = "tail_gap_bridge_not_supported"
    else:
        verdict = "tail_gap_bridge_partial"

    v14.write_csv(out / "per_cell_tail_gap_bridge.csv", per_cell_rows,
                  ["mass_cell", "n_orbits", "cell_gap_q10", "tail_gap_reliability",
                   "AUC_cell", "frame_p90", "frame_reliability"])
    v14.write_json(out / "tail_gap_auc_test.json", {
        **h1, "rho_bar": H1_RHO_BAR, "p_threshold": H1_P_THRESHOLD, "pass": h1_pass,
        "h1b_rho_median_v19": RHO_MEDIAN_V19, "h1b_delta": h1b_delta,
        "h1b_delta_bar": H1B_DELTA_BAR, "h1b_pass": h1b_pass, "h1b_binding": False,
        "loo_rho_min": loo_min, "loo_rho_median": loo_med, "loo_rho_max": loo_max,
        "sidecar_q05_rho": rho_q05, "sidecar_q20_rho": rho_q20, "q_primary": Q_PRIMARY,
    })
    v14.write_json(out / "tail_gap_frame_reliability_test.json", {
        **h2, "rho_bar": H2_RHO_BAR, "p_threshold": H2_P_THRESHOLD, "pass": h2_pass,
        "note": ("coherence guard; largely implied by v0.19 per-orbit H1 "
                 "(small re_gap -> high frame_spread); a FAILURE is the informative event"),
    })
    v14.write_csv(out / "sidecar_tail_sensitivity.csv", sidecar_rows,
                  ["mass_cell", "n_orbits", "AUC_cell", "frame_p90", "frame_reliability",
                   "tail_gap_reliability_q10", "tail_gap_reliability_q05",
                   "tail_gap_reliability_q20", "min_gap_reliability", "near_gap_fraction"])
    manifest.update({
        "stage": "analyzed", "verdict": verdict, "completedAt": v14.utc_now(),
        "gitCommit": git_commit(), "receipt_checks": checks, "receipts_ok": receipts_ok,
        "target_sha256": target_sha,
        "v18_per_cell_auc_sha256": v14.sha256_file(v18_dir / "per_cell_auc.csv"),
        "v19_per_row_spectrum_sha256": v14.sha256_file(v19_dir / "per_row_spectrum.csv"),
        "n_cells": len(cells), "q_primary": Q_PRIMARY, "global_re_gap_q10": global_q10,
        "seed": args.seed, "permutations": args.permutations,
        "H1_rho": h1["rho"], "H1_p": h1["p"], "H1_pass": h1_pass,
        "H1b_delta": h1b_delta, "H1b_pass": h1b_pass, "H1b_binding": False,
        "H1b_rho_median_v19": RHO_MEDIAN_V19,
        "H2_rho": h2["rho"], "H2_p": h2["p"], "H2_pass": h2_pass,
        "loo_rho_min": loo_min, "loo_rho_median": loo_med, "loo_rho_max": loo_max,
        "sidecar_q05_rho": rho_q05, "sidecar_q20_rho": rho_q20,
        "claim_boundary": CLAIM_BOUNDARY,
    })
    v14.write_json(out / "manifest.json", manifest)
    print(f"[v20:analyze] receipts_ok={receipts_ok} cells={len(cells)}")
    print(f"[v20:analyze] H1 tail->AUC rho={h1['rho']} p={h1['p']} pass={h1_pass}")
    print(f"[v20:analyze] H1b delta={h1b_delta} (bar {H1B_DELTA_BAR}, non-binding) pass={h1b_pass}")
    print(f"[v20:analyze] H2 tail->frame_rel rho={h2['rho']} p={h2['p']} pass={h2_pass}")
    print(f"[v20:analyze] LOO rho min/med/max={loo_min}/{loo_med}/{loo_max}")
    print(f"[v20:analyze] q05_rho={rho_q05} q20_rho={rho_q20} (report-only)")
    print(f"[v20:analyze] verdict={verdict}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="verify v0.18/v0.19 receipt gates + write manifest")
    p.add_argument("--target", default=str(v14.DEFAULT_TARGET))
    p.add_argument("--v18", default=str(DEFAULT_V18))
    p.add_argument("--v19", default=str(DEFAULT_V19))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.set_defaults(func=command_prepare)
    a = sub.add_parser("analyze", help="compute tail-gap statistic + H1/H1b/H2 + sidecars + verdict")
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
