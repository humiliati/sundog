"""Percival Track-C B1/B2 bridge: statistics + adjudication (instant, from score caches).

Pre-registration: docs/percival/PERCIVAL_TRACKC_B1B2_PREREG.md (frozen 2026-07-03; gates verbatim).

Reads the per-item caches written by percival_b1b2_cache.py, computes the paired/unpaired bootstrap
variances, the classical-formula ratio R, sign stability, the floor audit, the T-arm contrast, and
adjudicates B2-a, B1-a, B1-b, B1-c, B2-b.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SEED = 20260703
B = 2000
N_SUB = 500
REPO_ROOT = Path(__file__).resolve().parents[1]
SCORES = REPO_ROOT / "results/percival/b1b2/scores"

LADDER = ["main-142000", "main-140000", "main-130000", "main-110000", "main-70000", "main-30000"]
MAIN = "main-143000-passA"
SELF_B = "main-143000-passB"
WIDE = "dedup-143000"


def load_scores(tag, task):
    p = SCORES / tag / f"{task}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    if task == "hellaswag":
        return {"a": np.array([int(r["pred"] == r["gold"]) for r in d["rows"]]),
                "pred": np.array([r["pred"] for r in d["rows"]])}
    if task == "lambada":
        return {"a": np.array([r["correct"] for r in d["rows"]]), "pred": None}
    if task == "tarm":
        return {"a": np.array([r["mean"] for r in d["rows"]]), "pred": None}
    raise ValueError(task)


def pair_stats(a, b, pred_a=None, pred_b=None, rng=None):
    """d_b, d_s, deltas, bootstrap paired/unpaired variances, R, sign stability."""
    n = len(a)
    d_s = float(np.mean(a != b))
    d_b = float(np.mean(pred_a != pred_b)) if pred_a is not None else None
    pA, pB = float(np.mean(a)), float(np.mean(b))
    delta = pA - pB
    r_formula_num = d_s - delta ** 2
    r_formula_den = pA * (1 - pA) + pB * (1 - pB)
    r_formula = r_formula_num / r_formula_den if r_formula_den > 0 else None
    rng = rng or np.random.default_rng(SEED)
    pm = np.empty(B); um = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, N_SUB)
        pm[i] = float(np.mean(a[idx] - b[idx]))
        ia = rng.integers(0, n, N_SUB); ib = rng.integers(0, n, N_SUB)
        um[i] = float(np.mean(a[ia]) - np.mean(b[ib]))
    var_p, var_u = float(np.var(pm)), float(np.var(um))
    r_meas = var_p / var_u if var_u > 0 else None
    full_sign = np.sign(delta)
    stab_p = float(np.mean(np.sign(pm) == full_sign)) if full_sign != 0 else float(np.mean(pm == 0))
    stab_u = float(np.mean(np.sign(um) == full_sign)) if full_sign != 0 else float(np.mean(um == 0))
    return {"n": n, "d_b": d_b, "d_s": d_s, "p_A": round(pA, 4), "p_B": round(pB, 4),
            "delta": round(delta, 4), "var_paired": var_p, "var_unpaired": var_u,
            "R_meas": None if r_meas is None else round(r_meas, 4),
            "R_formula": None if r_formula is None else round(r_formula, 4),
            "rel_err": None if (r_meas is None or not r_formula) else round(abs(r_meas - r_formula) / r_formula, 4),
            "sign_stab_paired": round(stab_p, 4), "sign_stab_unpaired": round(stab_u, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor-only", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    out = {"prereg": "docs/percival/PERCIVAL_TRACKC_B1B2_PREREG.md", "seed": SEED, "B": B, "n_sub": N_SUB}

    # ---- B2-a: self-pair floor (runs FIRST; gates everything) ----
    sa, sb = load_scores(MAIN, "hellaswag"), load_scores(SELF_B, "hellaswag")
    if sa is None or sb is None:
        print("[stats] self-pair caches missing; run passA/passB first"); return
    floor_d_s = float(np.mean(sa["a"] != sb["a"]))
    floor_d_b = float(np.mean(sa["pred"] != sb["pred"]))
    out["floor"] = {"d_s_self": floor_d_s, "d_b_self": floor_d_b}
    print(f"[stats] FLOOR (self-pair): d_s={floor_d_s:.6f} d_b={floor_d_b:.6f}")
    if args.floor_only:
        gate = "CLEAN (exact zero)" if floor_d_s == 0 else f"nonzero -- rungs need d_s > {10*floor_d_s:.6f}"
        print(f"[stats] B2-a precondition: {gate}")
        (REPO_ROOT / "results/percival/b1b2").mkdir(parents=True, exist_ok=True)
        (REPO_ROOT / "results/percival/b1b2/floor.json").write_text(json.dumps(out["floor"]) + "\n", encoding="utf-8")
        return

    # ---- ladder + wide pairs, both tasks ----
    pairs = {}
    for task in ["hellaswag", "lambada"]:
        base = load_scores(MAIN, task)
        for tag in LADDER + [WIDE]:
            other = load_scores(tag, task)
            if base is None or other is None:
                continue
            pairs[f"{task}:{tag}"] = pair_stats(base["a"], other["a"], base["pred"], other["pred"], rng)
    out["pairs"] = pairs

    hs = [pairs.get(f"hellaswag:{t}") for t in LADDER]
    wide = pairs.get(f"hellaswag:{WIDE}")
    have_all = all(x is not None for x in hs) and wide is not None

    # B2-a
    min_rung_ds = min(x["d_s"] for x in hs) if have_all else None
    b2a = floor_d_s == 0 or (min_rung_ds is not None and floor_d_s < min_rung_ds / 10)
    resolvable = [t for t, x in zip(LADDER, hs) if x and (floor_d_s == 0 or x["d_s"] > 10 * floor_d_s)]

    # B1-a: rel_err <= 0.15 on >=5/6 rungs AND wide
    b1a_hits = sum(1 for x in hs if x and x["rel_err"] is not None and x["rel_err"] <= 0.15)
    b1a = have_all and b1a_hits >= 5 and wide["rel_err"] is not None and wide["rel_err"] <= 0.15

    # B1-b: d_s and var_paired nondecreasing with >=4 strict increases of 5 steps
    def monotone(vals):
        diffs = np.diff(vals)
        return bool(np.all(diffs >= -1e-12) and np.sum(diffs > 0) >= 4)
    b1b = have_all and monotone([x["d_s"] for x in hs]) and monotone([x["var_paired"] for x in hs])

    # B1-c: T-arm (main-143000-passA vs main-130000)
    b1c = None
    ta, tb = load_scores(MAIN, "tarm"), load_scores("main-130000", "tarm")
    la, lb_ = load_scores(MAIN, "lambada"), load_scores("main-130000", "lambada")
    if ta is not None and tb is not None and la is not None and lb_ is not None:
        nt = min(len(ta["a"]), len(tb["a"]))
        a7, b7 = ta["a"][:nt], tb["a"][:nt]
        a0, b0 = la["a"][:nt], lb_["a"][:nt]
        st0 = pair_stats(a0, b0, rng=rng)
        st7 = pair_stats(a7, b7, rng=rng)
        agree0 = a0 == b0                       # T=0 score-agreement items
        m0 = (a0 - b0)[agree0]; m7 = (a7 - b7)[agree0]
        agree_var_t0, agree_var_t7 = float(np.var(m0)), float(np.var(m7))
        r0, r7 = st0["R_meas"], st7["R_meas"]
        b1c = (r0 is not None and r7 is not None and r7 >= 3 * r0
               and agree_var_t0 == 0 and agree_var_t7 > 0)
        out["t_arm"] = {"R_T0": r0, "R_T07": r7, "agree_var_T0": agree_var_t0,
                        "agree_var_T07": agree_var_t7, "n": nt,
                        "stats_T0": st0, "stats_T07": st7}

    # B2-b: exists rung with paired stab >= .99 and unpaired < .90
    b2b = have_all and any(x["sign_stab_paired"] >= 0.99 and x["sign_stab_unpaired"] < 0.90 for x in hs + [wide])

    gates = {"B2a_floor": bool(b2a), "B1a_calibration": bool(b1a), "B1b_monotone": bool(b1b),
             "B1c_t_arm": b1c, "B2b_crispness": bool(b2b)}
    out["gates"] = gates
    out["resolvable_rungs"] = resolvable

    if not b2a and not resolvable:
        verdict = "B1B2_FLOOR_BLOCKED"
    elif all(v for v in gates.values()):
        verdict = "B1B2_BRIDGE_CONFIRMED"
    elif gates["B1b_monotone"] and gates["B1c_t_arm"] and gates["B2a_floor"] and gates["B2b_crispness"] and not gates["B1a_calibration"]:
        verdict = "B1B2_QUALITATIVE_ONLY"
    elif not gates["B1b_monotone"]:
        verdict = "B1B2_REFUTED"
    else:
        verdict = "B1B2_GAP"
    out["verdict"] = verdict

    res_dir = REPO_ROOT / "results/percival/b1b2"
    res_dir.mkdir(parents=True, exist_ok=True)
    (res_dir / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    # receipt
    lines = ["# Percival Track-C B1/B2 -- real-system bridge (results)", "",
             f"Pre-reg: [`PERCIVAL_TRACKC_B1B2_PREREG.md`](PERCIVAL_TRACKC_B1B2_PREREG.md) (frozen; claim register: CALIBRATION of the classical law -- McNemar/Connor; Miller 2024; Kotawala 2026 -- along a training trajectory + the dial/floor/T-contrast design contributions).", "",
             f"## Verdict: **{verdict}**", "",
             f"Floor (self-pair, fp32 fixed-batch): d_s = {floor_d_s:.6f}, d_b = {floor_d_b:.6f}.",
             f"Gates: {json.dumps(gates)}", "",
             "| pair | d_b | d_s | delta | R_meas | R_formula | rel_err | stab_p | stab_u |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for key, x in pairs.items():
        lines.append(f"| {key} | {x['d_b'] if x['d_b'] is not None else '--'} | {x['d_s']:.4f} | {x['delta']} | "
                     f"{x['R_meas']} | {x['R_formula']} | {x['rel_err']} | {x['sign_stab_paired']} | {x['sign_stab_unpaired']} |")
    if "t_arm" in out:
        t = out["t_arm"]
        lines += ["", f"T-arm (n={t['n']}): R(T=0)={t['R_T0']} vs R(T=0.7)={t['R_T07']}; "
                      f"agreement-item margin variance {t['agree_var_T0']} (T=0) vs {t['agree_var_T07']:.6f} (T=0.7)."]
    lines += ["", "## Honest boundary", "",
              "Calibration of a classical law on one substrate family (Pythia-160M) and one primary task; "
              "replication-grade statistics, design-grade novelty (checkpoint dial, floor audit, T-contrast). "
              "No deception-detection claims; the Track-C link is interpretive. Misses are findings; no post-hoc gate edits.", ""]
    (REPO_ROOT / "docs/percival/PERCIVAL_TRACKC_B1B2_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[stats] {verdict}  gates={json.dumps(gates)}")
    print(f"[stats] wrote results/percival/b1b2/summary.json + docs/percival/PERCIVAL_TRACKC_B1B2_RESULTS.md")


if __name__ == "__main__":
    main()
