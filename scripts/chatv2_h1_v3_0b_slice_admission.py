#!/usr/bin/env python3
"""Chat-v2 H1/V3-0b — AMBIGUITY-SLICE bank admission (chess arm; CPU, non-promotional).
Pre-reg: docs/chatv2/H1_V3_0B_AMBIGUITY_SLICE_SPEC.md (amends H1_V3_STATEBANK_SCOPE.md).
Receipt: docs/chatv2/H1_V3_0B_SLICE_ADMISSION_RECEIPT.md.

Slice = candidate criterion (square piece-code changes >=2 during the prefix; board-diff, so
captures / en passant / castling covered) + per-axis witness pairs via bounded LOCAL search
(adjacent same-color swaps, 1-3 per try; random full permutations rarely stay legal at depth).
Gates INHERITED UNCHANGED but evaluated ON THE SLICE: floor 120, balance [0.40,0.60], ALL
probes <=0.60 held-out (UCI = registered condition; SAN/sanitized = leak controls on the
first 10 axes, informational), >=1 witness pair/axis (target 3, 600 s budget), >=24 axes,
cap 48 by balance rank. NEW liveness control: bag-determined axis 'e2e4-token-present' must
probe >=0.95 on slice instances, else apparatus-abort (no verdict). Marker scan {24,32,40}
by slice-floor axis count only (mechanical, no probe peeking).
Branches: H1-V3-0B-ADMIT / F3-V3b/{slice,input,witness,bank}. No model, no GPU, no R2 claim.
Run: python scripts/chatv2_h1_v3_0b_slice_admission.py
"""
import collections, itertools, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatv2_h1_v3_data_admission import (CEIL, BAL_LO, BAL_HI, MIN_N, OUT_DIR,
                                         PREDS, SAN_STRIP, load_games, probe_suite, sq_code)

MARKERS, AXIS_CAP, WIT_TARGET, WIT_BUDGET_S, LIVENESS_MIN = (24, 32, 40), 48, 3, 600, 0.95


def build(games, marker):
    """Replay to the marker, tracking per-square piece-code change counts (board-diff)."""
    import chess
    out = []
    for i, g in enumerate(games):
        if len(g["san"]) < marker + 4:
            continue
        b = chess.Board()
        prev = [sq_code(b, s) for s in chess.SQUARES]
        changes = [0] * 64
        ucis, ok = [], True
        for mv in g["san"][:marker]:
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
                    changes[s] += 1
            prev = cur
        if not ok:
            continue
        sanM = g["san"][:marker]
        out.append({"gid": i,
                    "sq": {chess.SQUARE_NAMES[s]: prev[s] for s in chess.SQUARES},
                    "chg": {chess.SQUARE_NAMES[s]: changes[s] for s in range(64)},
                    "eco": g["eco"], "san": " ".join(sanM),
                    "sanit": " ".join(SAN_STRIP.sub("", m) for m in sanM),
                    "uci": " ".join(ucis)})
    return out


def floor_axes(inst):
    """(pred, sq) axes whose candidate slice (chg >= 2) meets the 120 floor."""
    import chess
    out = []
    for sq in chess.SQUARE_NAMES:
        n = sum(1 for r in inst if r["chg"][sq] >= 2)
        if n >= MIN_N:
            out += [(f"{p}.{sq}", n) for p in PREDS]
    return out


def perturb(ucis, rng):
    """1-3 adjacent same-color swaps; returns the re-interleaved sequence."""
    W, B = ucis[0::2], ucis[1::2]
    for _ in range(int(rng.integers(1, 4))):
        seq = W if rng.random() < 0.5 else B
        if len(seq) < 2:
            continue
        i = int(rng.integers(0, len(seq) - 1))
        seq[i], seq[i + 1] = seq[i + 1], seq[i]
    return [m for pair in itertools.zip_longest(W, B) for m in pair if m is not None]


def legal_final(alt):
    import chess
    b = chess.Board()
    try:
        for u in alt:
            b.push_uci(u)
    except Exception:
        return None
    return b


def witness_search(inst, axes_want, budget_s=WIT_BUDGET_S, target=WIT_TARGET, tries=30):
    """Per-axis witness panels via bounded local search over real-instance reorderings."""
    import chess
    rng = np.random.default_rng(3)
    found = collections.defaultdict(list)
    t0 = time.time()
    for k in rng.permutation(len(inst)):
        if time.time() - t0 > budget_s or all(len(found[a]) >= target for a in axes_want):
            break
        r = inst[int(k)]
        ucis = r["uci"].split()
        need = [a for a in axes_want if len(found[a]) < target and r["chg"][a.split(".")[1]] >= 2]
        if not need:
            continue
        for _ in range(tries):
            alt = perturb(ucis, rng)
            if alt == ucis:
                continue
            b = legal_final(alt)
            if b is None:
                continue
            for ax in need:
                if len(found[ax]) >= target:
                    continue
                pname, sq = ax.split(".")
                fn = PREDS[pname]
                c1 = sq_code(b, chess.parse_square(sq))
                if fn(r["sq"][sq]) != fn(c1):
                    found[ax].append({"a": r["uci"], "b": " ".join(alt), "axis": ax,
                                      "label_a": bool(fn(r["sq"][sq])), "label_b": bool(fn(c1)),
                                      "gid": r["gid"]})
    return found, time.time() - t0


def coverage_metric(inst, axes, rng, n=200, tries=40):
    """Honesty metric: fraction of sampled (instance, axis) slice-candidates that get an
    instance-level witness within a small per-instance budget. Reported, not gated."""
    import chess
    pool = [(i, ax) for ax in axes for i, r in enumerate(inst) if r["chg"][ax.split(".")[1]] >= 2]
    samp = [pool[int(j)] for j in rng.choice(len(pool), min(n, len(pool)), replace=False)]
    hit = 0
    for i, ax in samp:
        r = inst[i]
        ucis = r["uci"].split()
        pname, sq = ax.split(".")
        fn = PREDS[pname]
        for _ in range(tries):
            alt = perturb(ucis, rng)
            if alt == ucis:
                continue
            b = legal_final(alt)
            if b is None:
                continue
            if fn(r["sq"][sq]) != fn(sq_code(b, chess.parse_square(sq))):
                hit += 1
                break
    return hit, len(samp)


def main():
    import chess
    print("H1_V3_0B_SLICE_ADMISSION  chess arm  [NON-PROMOTIONAL, CPU]\n", flush=True)
    games = load_games(2600, 28)

    # marker scan: slice-floor axis count ONLY (declared; no probe peeking)
    best, best_inst, best_ax = None, None, []
    for marker in MARKERS:
        cand = build(games, marker)
        ax = floor_axes(cand)
        co = [sum(1 for sq in chess.SQUARE_NAMES if r["chg"][sq] >= 2) for r in cand]
        print(f"(marker scan) ply {marker}: {len(cand)} instances, {len(ax)} axes meet slice floor "
              f"{MIN_N}; co-ambiguity median {int(np.median(co))} p75 {int(np.percentile(co, 75))}")
        if len(ax) > len(best_ax):
            best, best_inst, best_ax = marker, cand, ax
    inst = best_inst
    print(f"(chosen) marker ply {best}: {len(best_ax)} floor axes, {len(inst)} instances", flush=True)
    if len(best_ax) < 24:
        print(f"\nVERDICT: F3-V3b/slice - only {len(best_ax)} axes meet the slice floor (<24)")
        return

    # balance on the slice, then cap 48 by balance rank
    scored = []
    for ax, n in best_ax:
        pname, sq = ax.split(".")
        fn = PREDS[pname]
        sub = [r for r in inst if r["chg"][sq] >= 2]
        bal = float(np.mean([fn(r["sq"][sq]) for r in sub]))
        if BAL_LO <= bal <= BAL_HI:
            scored.append((ax, len(sub), bal))
    print(f"(balance) {len(scored)} axes balanced on their slices")
    if len(scored) < 24:
        print(f"\nVERDICT: F3-V3b/bank - only {len(scored)} balanced sliced axes (<24)")
        return
    bank = sorted(scored, key=lambda t: abs(t[2] - 0.5))[:AXIS_CAP]

    # liveness (apparatus gate, not a verdict branch): bag-determined axis on slice instances
    live_sub = [r for r in inst if any(r["chg"][ax.split(".")[1]] >= 2 for ax, _, _ in bank[:5])]
    y_live = ["e2e4" in r["uci"].split() for r in live_sub]
    res_live = probe_suite([r["uci"] for r in live_sub], y_live, [r["gid"] for r in live_sub])
    lv = res_live["LR-counts"] if res_live else 0.0
    print(f"(liveness) 'e2e4-present' LR-counts on slice instances: {lv:.3f} (must be >= {LIVENESS_MIN})")
    if lv < LIVENESS_MIN:
        print("\nAPPARATUS-ABORT: liveness axis unreadable; no verdict issued.")
        return

    # witness panels + coverage honesty metric
    wit, used = witness_search(inst, [ax for ax, _, _ in bank])
    n_pairs = sum(len(v) for v in wit.values())
    print(f"(witness) {sum(1 for a, _, _ in bank if wit[a])}/{len(bank)} axes covered, "
          f"{n_pairs} pairs, {used:.0f}s of {WIT_BUDGET_S}s budget", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "witness_pairs_v3b_chess.jsonl"), "w") as f:
        for v in wit.values():
            for w in v:
                f.write(json.dumps(w) + "\n")
    rng = np.random.default_rng(4)
    hit, tot = coverage_metric(inst, [ax for ax, _, _ in bank], rng)
    print(f"(coverage) instance-level witness coverage on {tot}-sample: {hit}/{tot} = {hit/max(tot,1):.2f}")

    # probes on the slice (UCI = registered; SAN/sanit leak controls on first 10 axes)
    ecos = sorted({r["eco"] for r in inst})
    eco_map = {e: k for k, e in enumerate(ecos)}
    admitted, kills = [], collections.Counter()
    print(f"\n  {'axis':>8} {'n_slice':>7} {'bal':>5} {'LRcnt':>6} {'tfidf':>6} {'MLPw1':>6} "
          f"{'MLPw2':>6} {'MLPw3':>6} {'meta':>6} {'max':>6} {'wit':>4}  admit?", flush=True)
    ctrl_rows = []
    for j, (ax, n, bal) in enumerate(bank):
        pname, sq = ax.split(".")
        fn = PREDS[pname]
        sub = [r for r in inst if r["chg"][sq] >= 2]
        y = [fn(r["sq"][sq]) for r in sub]
        meta = np.zeros((len(sub), len(ecos)))
        for k, r in enumerate(sub):
            meta[k, eco_map[r["eco"]]] = 1.0
        res = probe_suite([r["uci"] for r in sub], y, [r["gid"] for r in sub], extra=meta)
        if res is None:
            kills["split"] += 1
            continue
        mx = max(res.values())
        has_wit = len(wit[ax]) >= 1
        ok = mx <= CEIL and has_wit
        if ok:
            admitted.append(ax)
        else:
            kills["probe" if mx > CEIL else "witness"] += 1
        print(f"  {ax:>8} {len(sub):>7} {bal:>5.2f} {res['LR-counts']:>6.3f} {res['LR-tfidf12']:>6.3f} "
              f"{res['MLP-w1']:>6.3f} {res['MLP-w2']:>6.3f} {res['MLP-w3']:>6.3f} "
              f"{res['LR-meta']:>6.3f} {mx:>6.3f} {len(wit[ax]):>4}  {'YES' if ok else 'no'}", flush=True)
        if j < 10:  # leak controls (informational, declared cost cap)
            for cond in ("san", "sanit"):
                rc = probe_suite([r[cond] for r in sub], y, [r["gid"] for r in sub], extra=meta)
                if rc:
                    ctrl_rows.append((ax, cond, max(rc.values())))
    print("\n  (leak controls, first 10 axes, informational)")
    for ax, cond, mx in ctrl_rows:
        print(f"    {ax:>8} [{cond:>5}] probe-max {mx:.3f}")

    print(f"\n  ADMITTED sliced axes: {len(admitted)} / 24 needed")
    print(f"  kill causes: {dict(kills)}")
    if len(admitted) >= 24:
        v = (f"H1-V3-0B-ADMIT - {len(admitted)} axes survive the slice floor + balance + "
             "surface suite + witness panel: an ambiguity-slice state bank EXISTS on real games")
    elif kills["probe"] >= max(kills["witness"], 1):
        v = (f"F3-V3b/input - the surface suite still reads {kills['probe']} axes ON their ambiguity "
             "slices: natural move statistics pin state even on formally-undetermined fibers. "
             "STRONG negative (ambiguous fibers are not uniform enough), not a bookkeeping failure.")
    elif kills["witness"] > 0:
        v = f"F3-V3b/witness - witness search under-covers ({kills['witness']} axes without pairs in budget)"
    else:
        v = f"F3-V3b/bank - only {len(admitted)} axes clear (<24)"
    print(f"\nVERDICT: {v}")
    print("  (Non-promotional data admission. No model run, no GPU, no R2 claim.)")


if __name__ == "__main__":
    main()
