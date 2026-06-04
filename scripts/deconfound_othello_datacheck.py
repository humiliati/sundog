#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND §6 — model-free legal-move determining-set data-check.

Question: on REAL Othello positions (no model), does a SUBSET of board squares determine
the LEGAL-MOVE set (func) before it determines the FULL board (state)?
  k_func(legal) << k_state(board)  =>  the closure functional (legal moves) is WELL-POSED
  at the data level -> green-light SUNDOG_V_DECONFOUND 0B.

This is the Othello analog of the coupled-toy data-check that grounded Phase 7 before any
training. Pure numpy + sklearn; no model.
"""
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


def _acc(Xtr, ytr, Xhe, yhe):
    if len(np.unique(ytr)) < 2:
        return None, None
    base = max(np.bincount(yhe).max() / len(yhe), 1 - yhe.mean() if yhe.mean() < .5 else yhe.mean())
    base = np.bincount(yhe).max() / len(yhe)
    acc = LogisticRegression(max_iter=200).fit(Xtr, ytr).score(Xhe, yhe)
    return acc, base


def sweep(board, legal, rng, ks, R=15, ntgt=12):
    N = len(board); tr = slice(0, N // 2); he = slice(N // 2, N)
    out = {}
    for k in ks:
        fa, fb, sa, sb = [], [], [], []
        for _ in range(R):
            S = rng.choice(64, k, replace=False)
            Xtr, Xhe = board[tr][:, S], board[he][:, S]
            for j in rng.choice(64, min(ntgt, 64), replace=False):          # func: legal-move squares
                a, bse = _acc(Xtr, legal[tr][:, j], Xhe, legal[he][:, j])
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


if __name__ == "__main__":
    rng = default_rng(0)
    assert len(legal_moves(init_board(), 1)) == 4, "Othello rules sanity (opening = 4 legal)"
    print("[gen] generating real Othello positions...", flush=True)
    board, legal = gen_positions(2500, rng)
    print(f"[gen] {len(board)} positions | mean legal moves/pos = {legal.sum(1).mean():.1f} | "
          f"board occupancy = {(board != 0).mean():.2f}", flush=True)
    ks = [1, 2, 4, 8, 16, 24, 32, 48, 60]
    res = sweep(board, legal, rng, ks)
    print("\n k  | func(legal) acc/base det | state(board) acc/base det")
    for k in ks:
        r = res[k]
        print(f" {k:>2} |  {r['func_acc']:.3f} / {r['func_base']:.3f}  {r['det_func']:.3f}"
              f"   |  {r['state_acc']:.3f} / {r['state_base']:.3f}  {r['det_state']:.3f}", flush=True)
    # crude verdict: does det_func clearly lead det_state at small k?
    lead = [res[k]['det_func'] - res[k]['det_state'] for k in ks]
    print(f"\n det_func - det_state by k: {[round(x,3) for x in lead]}")
    kf = next((k for k in ks if res[k]['det_func'] >= 0.80), None)
    ks_ = next((k for k in ks if res[k]['det_state'] >= 0.80), None)
    print(f" k_func(det>=.80)={kf}  k_state(det>=.80)={ks_}  -> "
          f"{'CLOSURE WELL-POSED (legal leads board)' if kf and (ks_ is None or kf < ks_) else 'no clear lead — re-examine functional'}")
