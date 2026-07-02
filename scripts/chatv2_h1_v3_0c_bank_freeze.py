#!/usr/bin/env python3
"""Chat-v2 H1/V3-0c — bank freeze: slices + REPAIRED witness search + frozen surface
baselines (chess arm; CPU, non-promotional).
Pre-reg: docs/chatv2/H1_V3_0C_CROSSOVER_SPEC.md (crossover gate; supersedes V3-0b's
absolute ceiling by owner decision). Receipt: docs/chatv2/H1_V3_0C_BANK_RECEIPT.md.

Data gate: >=24 axes with slice floor 120 + balance [0.40,0.60] + >=1 witness pair
(target 3; 900 s budget-exhausting, INTERACTION-DIRECTED search: rotate the same-color
move at one of the axis square's change plies by <=3 positions, optional extra adjacent
swap — V3-0b showed undirected legal swaps mostly commute). Surface baselines (full probe
suite, max) frozen per axis into the manifest BEFORE any model run. Liveness inherited.
Branches: H1-V3-0C-DATA-ADMIT / F3-V3c/{witness,bank}.
Run: python scripts/chatv2_h1_v3_0c_bank_freeze.py
"""
import collections, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatv2_h1_v3_data_admission import (BAL_LO, BAL_HI, MIN_N, OUT_DIR,
                                         PREDS, SAN_STRIP, load_games, probe_suite, sq_code)
from chatv2_h1_v3_0b_slice_admission import legal_final

MARKER, AXIS_CAP, WIT_TARGET, WIT_BUDGET_S, LIVENESS_MIN = 40, 48, 3, 900, 0.95


def build(games, marker):
    """Replay to the marker; track per-square change counts AND change plies."""
    import chess
    out = []
    for i, g in enumerate(games):
        if len(g["san"]) < marker + 4:
            continue
        b = chess.Board()
        prev = [sq_code(b, s) for s in chess.SQUARES]
        plies = [[] for _ in range(64)]
        ucis, ok = [], True
        for t, mv in enumerate(g["san"][:marker]):
            try:
                m = b.parse_san(mv)
                ucis.append(m.uci())
                b.push(m)
            except Exception:
                ok = False
                break
            cur = [sq_code(b, s) for s in chess.SQUARES]
            for s in range(64):
                if cur[s] != prev[s]:
                    plies[s].append(t)
            prev = cur
        if not ok:
            continue
        sanM = g["san"][:marker]
        out.append({"gid": i,
                    "sq": {chess.SQUARE_NAMES[s]: prev[s] for s in chess.SQUARES},
                    "chg": {chess.SQUARE_NAMES[s]: len(plies[s]) for s in range(64)},
                    "plies": {chess.SQUARE_NAMES[s]: plies[s] for s in range(64) if len(plies[s]) >= 2},
                    "eco": g["eco"], "uci": " ".join(ucis),
                    "sanit": " ".join(SAN_STRIP.sub("", m) for m in sanM)})
    return out


def rotations(ucis, chg_plies):
    """SYSTEMATIC candidate reorderings: for each change ply t of the square and each
    source ply in {t-1, t, t+1}, rotate that move within its color sequence by +/-1..3
    positions (multiset-preserving). Deterministic sweep — no random draws."""
    W, B = ucis[0::2], ucis[1::2]
    seen = set()
    for t in chg_plies[:6]:
        for src in (t - 1, t, t + 1):
            if not (0 <= src < len(ucis)):
                continue
            base = W if src % 2 == 0 else B
            i = src // 2
            if i >= len(base) or len(base) < 2:
                continue
            for off in (-3, -2, -1, 1, 2, 3):
                j = i + off
                if not (0 <= j < len(base)):
                    continue
                key = (src % 2, i, j)
                if key in seen:
                    continue
                seen.add(key)
                seq = base[:]
                mv = seq.pop(i)
                seq.insert(j, mv)
                pw = seq if src % 2 == 0 else W
                pb = seq if src % 2 == 1 else B
                yield [m for pair in itertools.zip_longest(pw, pb) for m in pair if m is not None]


def witness_search(inst, axes_want, budget_s=WIT_BUDGET_S, target=WIT_TARGET):
    """Budget-exhausting, per-axis round-robin, SYSTEMATIC interaction enumeration."""
    import chess
    rng = np.random.default_rng(3)
    cand = {ax: list(rng.permutation([k for k, r in enumerate(inst)
                                      if ax.split(".")[1] in r["plies"]]))
            for ax in axes_want}
    ptr = {ax: 0 for ax in axes_want}
    found = collections.defaultdict(list)
    t0 = time.time()
    while time.time() - t0 < budget_s:
        need = [a for a in axes_want if len(found[a]) < target and ptr[a] < len(cand[a])]
        if not need:
            break
        for ax in need:
            if time.time() - t0 > budget_s:
                break
            pname, sq = ax.split(".")
            fn = PREDS[pname]
            r = inst[int(cand[ax][ptr[ax]])]
            ptr[ax] += 1
            ucis = r["uci"].split()
            for alt in rotations(ucis, r["plies"][sq]):
                if alt == ucis:
                    continue
                b = legal_final(alt)
                if b is None:
                    continue
                c1 = sq_code(b, chess.parse_square(sq))
                if fn(r["sq"][sq]) != fn(c1):
                    found[ax].append({"a": r["uci"], "b": " ".join(alt), "axis": ax,
                                      "label_a": bool(fn(r["sq"][sq])), "label_b": bool(fn(c1)),
                                      "gid": r["gid"]})
                    break
    return found, time.time() - t0


def main():
    import chess
    print("H1_V3_0C_BANK_FREEZE  chess arm  [NON-PROMOTIONAL, CPU]\n", flush=True)
    games = load_games(2600, 28)
    inst = build(games, MARKER)
    co = [sum(1 for sq in chess.SQUARE_NAMES if r["chg"][sq] >= 2) for r in inst]
    print(f"(bank) marker ply {MARKER}: {len(inst)} instances; "
          f"co-ambiguity median {int(np.median(co))} p75 {int(np.percentile(co, 75))}", flush=True)

    # floor + balance on slices (inherited)
    scored = []
    for sq in chess.SQUARE_NAMES:
        sub = [r for r in inst if r["chg"][sq] >= 2]
        if len(sub) < MIN_N:
            continue
        for pname, fn in PREDS.items():
            bal = float(np.mean([fn(r["sq"][sq]) for r in sub]))
            if BAL_LO <= bal <= BAL_HI:
                scored.append((f"{pname}.{sq}", len(sub), bal))
    print(f"(balance) {len(scored)} axes balanced on their slices")
    if len(scored) < 24:
        print(f"\nVERDICT: F3-V3c/bank - only {len(scored)} balanced sliced axes (<24)")
        return
    bank = sorted(scored, key=lambda t: abs(t[2] - 0.5))[:AXIS_CAP]

    # liveness (inherited apparatus gate)
    live_sub = [r for r in inst if any(r["chg"][ax.split(".")[1]] >= 2 for ax, _, _ in bank[:5])]
    y_live = ["e2e4" in r["uci"].split() for r in live_sub]
    res_live = probe_suite([r["uci"] for r in live_sub], y_live, [r["gid"] for r in live_sub])
    lv = res_live["LR-counts"] if res_live else 0.0
    print(f"(liveness) 'e2e4-present' LR-counts: {lv:.3f} (>= {LIVENESS_MIN} required)")
    if lv < LIVENESS_MIN:
        print("\nAPPARATUS-ABORT: liveness axis unreadable; no verdict issued.")
        return

    # repaired witness search
    wit, used = witness_search(inst, [ax for ax, _, _ in bank])
    ncov = sum(1 for ax, _, _ in bank if wit[ax])
    print(f"(witness) {ncov}/{len(bank)} axes covered, {sum(len(v) for v in wit.values())} pairs, "
          f"{used:.0f}s of {WIT_BUDGET_S}s budget", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "witness_pairs_v3c_chess.jsonl"), "w") as f:
        for v in wit.values():
            for w in v:
                f.write(json.dumps(w) + "\n")

    # frozen surface baselines (the matched baseline for all model rungs)
    ecos = sorted({r["eco"] for r in inst})
    eco_map = {e: k for k, e in enumerate(ecos)}
    manifest_axes, admitted = [], []
    print(f"\n  {'axis':>8} {'n_slice':>7} {'bal':>5} {'surface_max':>11} {'wit':>4}  in-bank?", flush=True)
    for ax, n, bal in bank:
        pname, sq = ax.split(".")
        fn = PREDS[pname]
        sub = [r for r in inst if r["chg"][sq] >= 2]
        y = [fn(r["sq"][sq]) for r in sub]
        meta = np.zeros((len(sub), len(ecos)))
        for k, r in enumerate(sub):
            meta[k, eco_map[r["eco"]]] = 1.0
        res = probe_suite([r["uci"] for r in sub], y, [r["gid"] for r in sub], extra=meta)
        if res is None:
            continue
        mx = max(res.values())
        ok = len(wit[ax]) >= 1
        if ok:
            admitted.append(ax)
        manifest_axes.append({"axis": ax, "n_slice": len(sub), "balance": round(bal, 4),
                              "baselines": {k: round(v, 4) for k, v in res.items()},
                              "surface_max": round(mx, 4), "n_witness": len(wit[ax]),
                              "gids": [r["gid"] for r in sub], "in_bank": ok})
        print(f"  {ax:>8} {len(sub):>7} {bal:>5.2f} {mx:>11.3f} {len(wit[ax]):>4}  "
              f"{'YES' if ok else 'no (witness)'}", flush=True)

    manifest = {"spec": "H1_V3_0C_CROSSOVER_SPEC.md", "date": "2026-07-01", "marker_ply": MARKER,
                "corpus": "Lichess/standard-chess-games 2013-01", "n_instances": len(inst),
                "co_ambiguity": {"median": int(np.median(co)), "p75": int(np.percentile(co, 75))},
                "liveness_LR_counts": round(lv, 4), "split": "GroupShuffleSplit(1,0.3,seed0) by gid",
                "crossover_margins": {"vs_surface": 0.15, "vs_randinit_floor": 0.15},
                "model_bank_gate": 20, "axes": manifest_axes}
    mpath = os.path.join(OUT_DIR, "v3_0c_bank_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"\n  manifest frozen -> results/chatv2/h1_v3/v3_0c_bank_manifest.json "
          f"({len(manifest_axes)} axes, {len(admitted)} in-bank)")
    print(f"\n  BANK: {len(admitted)} / 24 needed (witnessed, balanced, floored, baselines frozen)")
    if len(admitted) >= 24:
        v = (f"H1-V3-0C-DATA-ADMIT - {len(admitted)}-axis bank frozen with matched surface "
             "baselines; V3-0.5 GPT-2 calibration (crossover form, non-gate) may run")
    else:
        v = f"F3-V3c/witness - repaired search certifies only {len(admitted)} axes (<24)"
    print(f"\nVERDICT: {v}")
    print("  (Non-promotional bank freeze. No model run, no GPU, no R2 claim.)")


if __name__ == "__main__":
    main()
