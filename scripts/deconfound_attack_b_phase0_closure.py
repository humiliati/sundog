#!/usr/bin/env python
"""SUNDOG_V_DECONFOUND Attack-B Phase 0B — real-feature closure cell.

Implements docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_SPEC.md (locked 2026-06-04).

Three bodies on de-confound-clean real digit features b in {0,1}^8, u = XOR(b_0,b_1,b_2):
  * state-keeper        (read-dim 8) — reconstruct all b   -> must keep state
  * functional-keeper   (read-dim 8) — predict u           -> keep functional, drop b_j notin S
  * functional-keeper-compressed (read-dim 3) — predict u  -> capacity-disambiguation diagnostic

Read each body with a determining-shadow-set probe (k_func on u, k_state on the outside bits
b_j notin S, k_null on an independent u_null), best-size-k subset selected on a probe-TRAIN
split and scored on a held-out probe split, with a selection-corrected label-permutation null
(p<=0.01). Headline = paired keeper_gap (functional - state). Branch per spec §6 (precedence-
ordered). The compressed body is consulted ONLY to adjudicate a stateful functional-keeper.

Linear probe = closed-form ridge LSQ classifier (a standard, fast linear probe; makes the
1000x permutation null tractable). No claim beyond R1.5; functional is constructed, not
model-discovered.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
from sklearn.datasets import load_digits

import torch
import torch.nn as nn

# --------------------------------------------------------------------------- #
# substrate  (matches the passed 0-pre digit_bits(8): 4x2 pooled, median bits)
# --------------------------------------------------------------------------- #
D = 8
S_IDX = [0, 1, 2]
OUT_IDX = [3, 4, 5, 6, 7]
SPLIT_SEED = 20260604
LAM = 1.0                      # ridge probe regularizer
DET_THR = 0.70
ALPHA = 0.01


def digit_features():
    X = load_digits().images.astype(np.float64)            # (1797, 8, 8)
    n = X.shape[0]
    feats = np.zeros((n, D))
    f = 0
    for br in range(4):                                    # 4x2 grid of 2-row x 4-col blocks
        for bc in range(2):
            feats[:, f] = X[:, br * 2:(br + 1) * 2, bc * 4:(bc + 1) * 4].mean((1, 2))
            f += 1
    b = (feats > np.median(feats, 0)).astype(np.float32)   # (n, 8) balanced real bits
    u = (b[:, 0].astype(int) ^ b[:, 1].astype(int) ^ b[:, 2].astype(int)).astype(np.float32)
    return b, u


def strat_split(y, fracs, seed):
    """Stratified index split by binary y into len(fracs) parts."""
    rng = default_rng(seed)
    parts = [[] for _ in fracs]
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        cuts = (np.cumsum(fracs) * len(idx)).astype(int)[:-1]
        for p, chunk in enumerate(np.split(idx, cuts)):
            parts[p].extend(chunk.tolist())
    return [np.array(sorted(p)) for p in parts]


# --------------------------------------------------------------------------- #
# bodies
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(self, read_dim, head_out):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, 32), nn.GELU(),
                                  nn.Linear(32, read_dim), nn.GELU())
        self.head = nn.Linear(read_dim, head_out)

    def forward(self, x):
        h = self.body(x)
        return self.head(h), h


def train_model(kind, read_dim, b, targ, tr, va, seed, init_state=None,
                max_epochs=500, patience=40):
    """kind: 'state' (reconstruct b, head_out=8) | 'func' (predict u, head_out=1)."""
    torch.manual_seed(seed)
    head_out = 8 if kind == "state" else 1
    model = Model(read_dim, head_out)
    if init_state is not None:                              # matched body init for the rd8 pair
        model.body.load_state_dict(init_state)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xtr = torch.tensor(b[tr]); Xva = torch.tensor(b[va])
    if kind == "state":
        Ytr = torch.tensor(b[tr]); Yva = torch.tensor(b[va])
    else:
        Ytr = torch.tensor(targ[tr]).unsqueeze(1); Yva = torch.tensor(targ[va]).unsqueeze(1)
    best_va, best_state, bad = float("inf"), None, 0
    for _ in range(max_epochs):
        model.train(); opt.zero_grad()
        logits, _ = model(Xtr)
        loss = lossf(logits, Ytr)
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = lossf(model(Xva)[0], Yva).item()
        if vl < best_va - 1e-5:
            best_va, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def body_acts(model, b):
    with torch.no_grad():
        _, h = model(torch.tensor(b))
    return h.numpy()


# --------------------------------------------------------------------------- #
# determining-shadow-set read (closed-form ridge probe + selection-corrected null)
# --------------------------------------------------------------------------- #
def _design(H, S):
    return np.concatenate([H[:, list(S)], np.ones((len(H), 1))], axis=1)


def _det(acc, base):
    return (acc - base) / max(1.0 - base, 1e-9)


def determine_k(H, Y, tr, he, ks, rng, n_perm):
    """Smallest k with held-out mean det(Y) >= DET_THR and selection-corrected p <= ALPHA.
    Y: (n, m) binary block (m=1 for u/u_null, m=5 for the outside-state block).
    Subset selected by mean train-acc; scored on held; null permutes the label block rows."""
    n, d = H.shape
    m = Y.shape[1]
    base_he = np.array([max(Y[he, j].mean(), 1 - Y[he, j].mean()) for j in range(m)])
    # pre-generate permutations of the label block (rows), shared across k
    P = np.stack([rng.permutation(n) for _ in range(n_perm)])          # (n_perm, n)
    for k in ks:
        subsets = list(itertools.combinations(range(d), k))
        # ---- observed: best subset by mean train-acc ----
        best_det, best_acc = -1.0, -1.0
        for S in subsets:
            Xtr, Xhe = _design(H[tr], S), _design(H[he], S)
            M = np.linalg.solve(Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1]), Xtr.T)
            W = M @ (2 * Y[tr] - 1)
            acc_tr = ((Xtr @ W > 0) == Y[tr]).mean(0).mean()
            if acc_tr > best_acc:
                acc_he = ((Xhe @ W > 0) == Y[he]).mean(0)
                best_acc = acc_tr
                best_det = float(np.mean([_det(acc_he[j], base_he[j]) for j in range(m)]))
        if best_det < DET_THR:
            continue
        # ---- selection-corrected null at this k (vectorized over perms) ----
        acc_tr_all = np.empty((len(subsets), n_perm))
        det_he_all = np.empty((len(subsets), n_perm))
        for si, S in enumerate(subsets):
            Xtr, Xhe = _design(H[tr], S), _design(H[he], S)
            M = np.linalg.solve(Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1]), Xtr.T)
            acc_tr_p = np.zeros(n_perm); det_he_p = np.zeros(n_perm)
            for j in range(m):                                         # per state-bit column
                Yj = Y[:, j]
                Yp = Yj[P]                                             # (n_perm, n)
                Tp_tr = (2 * Yp[:, tr] - 1).T                          # (n_tr, n_perm)
                W = M @ Tp_tr                                          # (k+1, n_perm)
                ptr = (Xtr @ W > 0)                                    # (n_tr, n_perm)
                acc_tr_p += (ptr == Yp[:, tr].T).mean(0)
                phe = (Xhe @ W > 0)
                acc_he_p = (phe == Yp[:, he].T).mean(0)
                bp = np.maximum(Yp[:, he].mean(1), 1 - Yp[:, he].mean(1))
                det_he_p += (acc_he_p - bp) / np.maximum(1 - bp, 1e-9)
            acc_tr_all[si] = acc_tr_p / m
            det_he_all[si] = det_he_p / m
        pick = acc_tr_all.argmax(0)                                    # best subset per perm
        null_det = det_he_all[pick, np.arange(n_perm)]
        p = (1 + int((null_det >= best_det).sum())) / (n_perm + 1)
        if p <= ALPHA:
            return {"k": int(k), "det": round(best_det, 4), "p": round(p, 4)}
    return {"k": None, "det": None, "p": None}


def read_body(H, b, u, seed, read_dim, n_perm):
    """Run k_func / k_state / k_null on one body's activations."""
    n = len(H)
    pr = strat_split(u, [0.6, 0.4], SPLIT_SEED + 13 + seed)
    tr, he = pr[0], pr[1]
    ks_primary = list(range(1, read_dim + 1))                          # rd8 -> 1..8, rd3 -> 1..3
    rng = default_rng(SPLIT_SEED + 7 * seed)
    kf = determine_k(H, u[:, None], tr, he, ks_primary, rng, n_perm)
    ks_state = determine_k(H, b[:, OUT_IDX], tr, he, ks_primary, rng, n_perm)
    base = u[he].mean()
    unull = (default_rng(SPLIT_SEED + 1000 * seed).random(n) < base).astype(np.float32)
    kn = determine_k(H, unull[:, None], tr, he, ks_primary, rng, n_perm)
    return {"k_func": kf, "k_state": ks_state, "k_null": kn}


def bracket(read, kmax):
    kf, ks_, kn = read["k_func"]["k"], read["k_state"]["k"], read["k_null"]["k"]
    has = kf is not None and (ks_ is None or ks_ >= kf + 2) and kn is None
    mf = (kmax + 1) if kf is None else kf
    msx = (kmax + 1) if ks_ is None else ks_
    margin = max(0, msx - mf)
    return has, margin, (kn is not None)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def _native(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(repr(o))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="attack-b-phase0-closure")
    ap.add_argument("--out", default="results/deconfound/attack-b-phase0-closure")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--smoke", action="store_true", help="1 seed, 200 perms (timing only)")
    args = ap.parse_args()
    if args.smoke:
        args.seeds, args.n_perm = [0], 200
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    b, u = digit_features()
    n = len(b)
    tr, va, te = strat_split(u, [0.6, 0.2, 0.2], SPLIT_SEED)
    # split sanity (gate): u base rates within 0.08
    brates = [float(u[s].mean()) for s in (tr, va, te)]
    split_ok = (max(brates) - min(brates)) <= 0.08
    # de-confound replay (gate): linear input-probe on b -> u within det<=0.20 (the 0-pre row)
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    dc_acc = float(cross_val_score(LogisticRegression(max_iter=300), b, u, cv=5).mean())
    dc_base = float(max(u.mean(), 1 - u.mean()))
    dc_det = (dc_acc - dc_base) / max(1 - dc_base, 1e-9)
    deconfound_ok = dc_det <= 0.20

    per_seed = []
    t0 = time.time()
    for seed in args.seeds:
        torch.manual_seed(seed)
        init_body = Model(8, 1).body.state_dict()                      # matched init for rd8 pair
        sk = train_model("state", 8, b, u, tr, va, seed, init_state=init_body)
        fk = train_model("func", 8, b, u, tr, va, seed, init_state=init_body)
        fkc = train_model("func", 3, b, u, tr, va, seed)               # rd3 diagnostic (own init)

        # learned-body gates on the held-out test split
        def acc_det_u(model):
            with torch.no_grad():
                p = (torch.sigmoid(model(torch.tensor(b[te]))[0]).numpy().ravel() > 0.5)
            a = float((p == u[te]).mean()); base = max(u[te].mean(), 1 - u[te].mean())
            return a, _det(a, base)

        def acc_det_state(model):
            with torch.no_grad():
                p = (torch.sigmoid(model(torch.tensor(b[te]))[0]).numpy() > 0.5)
            accs = [(p[:, j] == b[te, j]).mean() for j in range(D)]
            bases = [max(b[te, j].mean(), 1 - b[te, j].mean()) for j in range(D)]
            return float(np.mean(accs)), float(np.mean([_det(accs[j], bases[j]) for j in range(D)]))

        fk_a, fk_d = acc_det_u(fk)
        sk_a, sk_d = acc_det_state(sk)
        fkc_a, fkc_d = acc_det_u(fkc)
        gates = {
            "func_learned": fk_d >= 0.70 and fk_a >= 0.80,
            "state_learned": sk_d >= 0.70 and sk_a >= 0.80,
            "compressed_learned": fkc_d >= 0.70 and fkc_a >= 0.80,
            "func_acc": round(fk_a, 3), "func_det": round(fk_d, 3),
            "state_acc": round(sk_a, 3), "state_det": round(sk_d, 3),
            "compressed_acc": round(fkc_a, 3), "compressed_det": round(fkc_d, 3),
        }

        reads, brk = {}, {}
        for name, model, rd in (("state", sk, 8), ("func", fk, 8), ("compressed", fkc, 3)):
            H = body_acts(model, b)
            r = read_body(H, b, u, seed, rd, args.n_perm)
            has, margin, null_hit = bracket(r, rd)
            reads[name] = r
            brk[name] = {"bracket": has, "margin": margin, "u_null_hit": null_hit}
        per_seed.append({"seed": seed, "gates": gates, "reads": reads, "brackets": brk})
        print(f"[seed {seed}] func_bracket={brk['func']['bracket']} "
              f"state_bracket={brk['state']['bracket']} comp_bracket={brk['compressed']['bracket']} "
              f"| gap={brk['func']['margin'] - brk['state']['margin']} "
              f"({round(time.time() - t0, 1)}s)", flush=True)

    # ---- aggregate + branch (precedence-ordered, §6) ----
    ns = len(per_seed)
    interpreted = [s for s in per_seed
                   if s["gates"]["func_learned"] and s["gates"]["state_learned"] and split_ok]
    func_brk = sum(s["brackets"]["func"]["bracket"] for s in interpreted)
    state_brk = sum(s["brackets"]["state"]["bracket"] for s in interpreted)
    func_stateful = sum(not s["brackets"]["func"]["bracket"] for s in interpreted)
    comp_brk = sum(s["brackets"]["compressed"]["bracket"] for s in interpreted)
    comp_stateful = sum(not s["brackets"]["compressed"]["bracket"] for s in interpreted)
    comp_learned = all(s["gates"]["compressed_learned"] for s in interpreted) and len(interpreted) > 0
    gaps = sorted(s["brackets"]["func"]["margin"] - s["brackets"]["state"]["margin"] for s in interpreted)
    median_gap = float(np.median(gaps)) if gaps else 0.0
    unull_hit = any(s["brackets"][bod]["u_null_hit"] for s in interpreted for bod in ("state", "func"))

    need = max(1, int(np.ceil(0.8 * ns)))                              # ">=4/5" generalized
    most = max(1, int(np.ceil(0.4 * ns)))                             # ">=2/5"

    if not interpreted or any(not (s["gates"]["func_learned"] and s["gates"]["state_learned"])
                              for s in per_seed) or not split_ok:
        verdict = "closure_void_unlearned"
    elif not deconfound_ok:
        verdict = "closure_void_deconfound_replay"
    elif unull_hit:
        verdict = "closure_void_control"
    elif state_brk >= most:
        verdict = "attack_b_state_keeper_closure_like"
    elif func_stateful >= need:
        if not comp_learned:
            verdict = "attack_b_functional_keeper_stateful_capacity_unresolved"
        elif comp_brk >= need:
            verdict = "attack_b_functional_keeper_stateful_capacity"
        elif comp_stateful >= need:
            verdict = "attack_b_functional_keeper_stateful_objective"
        else:
            verdict = "attack_b_functional_keeper_stateful_capacity_unresolved"
    elif func_brk >= need and state_brk <= (ns - need) and median_gap >= 2 and not unull_hit:
        verdict = "attack_b_closure_confirmed"
    else:
        verdict = "attack_b_objective_gap_absent"

    summary = {
        "phase": args.phase, "n_seeds": ns, "n_perm": args.n_perm,
        "split_base_rates": [round(x, 3) for x in brates], "split_ok": split_ok,
        "deconfound_replay": {"det": round(dc_det, 4), "ok": deconfound_ok},
        "counts": {"func_bracket": func_brk, "state_bracket": state_brk,
                   "func_stateful": func_stateful, "compressed_bracket": comp_brk,
                   "compressed_learned": comp_learned, "median_keeper_gap": median_gap},
        "verdict": verdict, "per_seed": per_seed,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_native))
    print("\n==== VERDICT:", verdict, "====")
    print(f"  func_bracket {func_brk}/{ns}  state_bracket {state_brk}/{ns}  "
          f"median_keeper_gap {median_gap}  deconfound_replay_det {round(dc_det,4)}")
    print(f"  wrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
