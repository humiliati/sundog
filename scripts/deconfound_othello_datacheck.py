#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND §6 — model-free Othello determining-set data-check.

Question: on REAL Othello positions (no model), does a SUBSET of board squares determine
the candidate functional (func) before it determines the FULL board (state)?
  k_func << k_state(board)  =>  the closure functional is WELL-POSED
  at the data level -> green-light SUNDOG_V_DECONFOUND 0B.

This is the Othello analog of the coupled-toy data-check that grounded Phase 7 before any
training. Pure numpy + sklearn; no model.
"""
import argparse

import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression

DIRS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def init_board():
    b = np.zeros((8, 8), dtype=np.int8)
    b[3, 3] = 2; b[3, 4] = 1; b[4, 3] = 1; b[4, 4] = 2   # 1=black, 2=white
    return b


def legal_moves(b, player):
    opp = 3 - player
    mv = []
    for r in range(8):
        for c in range(8):
            if b[r, c] != 0:
                continue
            for dr, dc in DIRS:
                rr, cc, seen = r + dr, c + dc, False
                while 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == opp:
                    rr += dr; cc += dc; seen = True
                if seen and 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == player:
                    mv.append((r, c)); break
    return mv


def apply_move(b, player, mv):
    r, c = mv; opp = 3 - player; b[r, c] = player
    for dr, dc in DIRS:
        rr, cc, line = r + dr, c + dc, []
        while 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == opp:
            line.append((rr, cc)); rr += dr; cc += dc
        if line and 0 <= rr < 8 and 0 <= cc < 8 and b[rr, cc] == player:
            for (lr, lc) in line:
                b[lr, lc] = player
    return b


def gen_positions(n, rng):
    boards, legals = [], []
    while len(boards) < n:
        b = init_board(); player = 1
        for _ in range(int(rng.integers(4, 50))):
            mv = legal_moves(b, player)
            if not mv:
                player = 3 - player
                mv = legal_moves(b, player)
                if not mv:
                    break
            apply_move(b, player, mv[int(rng.integers(len(mv)))])
            player = 3 - player
        lm = legal_moves(b, player)
        if not lm:
            continue
        mine = (b == player).astype(np.int8); theirs = (b == 3 - player).astype(np.int8)
        boards.append((mine - theirs).flatten())              # 64: +1 mine, -1 theirs, 0 empty
        lv = np.zeros(64, dtype=np.int8)
        for (r, c) in lm:
            lv[r * 8 + c] = 1
        legals.append(lv)
    return np.array(boards), np.array(legals)


def frontier_labels(board):
    labels = np.zeros_like(board, dtype=np.int8)
    for i, row in enumerate(board.reshape((-1, 8, 8))):
        out = np.zeros((8, 8), dtype=np.int8)
        for r in range(8):
            for c in range(8):
                if row[r, c] == 0:
                    continue
                for dr, dc in DIRS:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < 8 and 0 <= cc < 8 and row[rr, cc] == 0:
                        out[r, c] = 1
                        break
        labels[i] = out.flatten()
    return labels


def make_functional(name, board, legal):
    if name == "legal":
        return legal, "64 binary labels: legal next-move squares"
    if name == "frontier":
        return frontier_labels(board), "64 binary labels: occupied squares adjacent to empties"
    if name == "material_parity":
        # Parity of material advantage is intentionally nonlinear in square indicators.
        # It is a useful escape-hatch screen precisely because it should fail if the
        # linear closure instrument cannot see XOR-like real-board aggregates.
        y = (np.abs(board.sum(axis=1).astype(np.int64)) % 2).reshape((-1, 1))
        return y.astype(np.int8), "1 binary label: parity of absolute material advantage"
    if name == "mobility":
        counts = legal.sum(axis=1)
        threshold = int(np.median(counts))
        y = (counts >= threshold).reshape((-1, 1))
        return y.astype(np.int8), f"1 binary label: legal-move count >= sample median ({threshold})"
    raise ValueError(f"unknown functional {name!r}")


def _acc(Xtr, ytr, Xhe, yhe):
    if len(np.unique(ytr)) < 2:
        return None, None
    base = np.bincount(yhe).max() / len(yhe)
    acc = LogisticRegression(max_iter=200).fit(Xtr, ytr).score(Xhe, yhe)
    return acc, base


def sweep(board, functional, rng, ks, R=15, ntgt=12):
    N = len(board); tr = slice(0, N // 2); he = slice(N // 2, N)
    if functional.ndim == 1:
        functional = functional.reshape((-1, 1))
    out = {}
    for k in ks:
        fa, fb, sa, sb = [], [], [], []
        for _ in range(R):
            S = rng.choice(64, k, replace=False)
            Xtr, Xhe = board[tr][:, S], board[he][:, S]
            functional_targets = np.arange(functional.shape[1])
            for j in rng.choice(functional_targets, min(ntgt, len(functional_targets)), replace=False):
                a, bse = _acc(Xtr, functional[tr][:, j], Xhe, functional[he][:, j])
                if a is not None:
                    fa.append(a); fb.append(bse)
            J = np.array([j for j in range(64) if j not in set(S.tolist())])
            for j in rng.choice(J, min(ntgt, len(J)), replace=False):       # state: board squares (3-class)
                a, bse = _acc(Xtr, (board[tr][:, j] + 1), Xhe, (board[he][:, j] + 1))
                if a is not None:
                    sa.append(a); sb.append(bse)
        fa, fb, sa, sb = map(lambda x: float(np.mean(x)), (fa, fb, sa, sb))
        det_f = (fa - fb) / (1 - fb + 1e-9)
        det_s = (sa - sb) / (1 - sb + 1e-9)
        out[k] = dict(func_acc=fa, func_base=fb, det_func=det_f,
                      state_acc=sa, state_base=sb, det_state=det_s)
    return out


def summarize_verdict(res, ks):
    max_k = max(ks)
    max_func = res[max_k]["det_func"]
    max_state = res[max_k]["det_state"]
    lead = [res[k]["det_func"] - res[k]["det_state"] for k in ks]
    best_lead = max(lead)
    kf = next((k for k in ks if res[k]["det_func"] >= 0.80), None)
    ks_ = next((k for k in ks if res[k]["det_state"] >= 0.80), None)
    if max_func < 0.50:
        verdict = "BLOCKED: functional not linearly readable enough even at high k"
    elif kf and (ks_ is None or kf < ks_) and best_lead >= 0.20:
        verdict = "CLOSURE WELL-POSED: functional leads board state"
    else:
        verdict = "BLOCKED: no non-vacuous k_func << k_state bracket"
    return verdict, lead, kf, ks_, best_lead, max_func, max_state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--functionals", default="legal",
                        help="comma-separated: legal,material_parity,frontier,mobility,all")
    parser.add_argument("--ks", default="1,2,4,8,16,24,32,48,60")
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--targets", type=int, default=12)
    args = parser.parse_args()

    rng = default_rng(args.seed)
    assert len(legal_moves(init_board(), 1)) == 4, "Othello rules sanity (opening = 4 legal)"
    print("[gen] generating real Othello positions...", flush=True)
    board, legal = gen_positions(args.n, rng)
    print(f"[gen] {len(board)} positions | mean legal moves/pos = {legal.sum(1).mean():.1f} | "
          f"board occupancy = {(board != 0).mean():.2f}", flush=True)
    ks = [int(k) for k in args.ks.split(",") if k.strip()]
    names = [name.strip() for name in args.functionals.split(",") if name.strip()]
    if "all" in names:
        names = ["legal", "material_parity", "frontier", "mobility"]

    for name in names:
        functional, description = make_functional(name, board, legal)
        print(f"\n=== functional: {name} ===")
        print(f"[functional] {description} | positive/base mean = {functional.mean():.3f}", flush=True)
        res = sweep(board, functional, rng, ks, R=args.repeats, ntgt=args.targets)
        print("\n k  | func acc/base det | state(board) acc/base det")
        for k in ks:
            r = res[k]
            print(f" {k:>2} |  {r['func_acc']:.3f} / {r['func_base']:.3f}  {r['det_func']:.3f}"
                  f"   |  {r['state_acc']:.3f} / {r['state_base']:.3f}  {r['det_state']:.3f}", flush=True)
        verdict, lead, kf, ks_, best_lead, max_func, max_state = summarize_verdict(res, ks)
        print(f"\n det_func - det_state by k: {[round(x, 3) for x in lead]}")
        print(f" k_func(det>=.80)={kf}  k_state(det>=.80)={ks_}  "
              f"best_lead={best_lead:.3f}  high_k_func={max_func:.3f}  high_k_state={max_state:.3f}")
        print(f" verdict: {verdict}", flush=True)
