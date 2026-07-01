#!/usr/bin/env python3
"""Chat-v2 R2 intersection slate — H2 hardening: the bracket STACK-TOP test.
Pre-reg: docs/chatv2/R2_INTERSECTION_HYPOTHESES.md H2. NON-promotional, CPU.

Conjecture test: over real code, is the bracket STACK-TOP (which delimiter must close
next -- an order-dependent, long-range state) carried linearly in GPT-2's residual stream
while an order-blind surface probe cannot read it?  DEPTH (= net bracket count, linear in
counts) is the decodable CONTROL: the surface probe MUST read depth (apparatus liveness).

Reaches the intersection iff: residual(stack-top) >> surface(stack-top), AND surface(depth)
is high (control works). Run: python scripts/chatv2_h2_stacktop_probe.py [--n 400]
"""
import os, glob, argparse
import numpy as np

BR = {")": "(", "]": "[", "}": "{"}


def load_code(cap=700_000):
    files = []
    for root in ["scripts", "../scripts", "../sundogcert/scripts", "../eyesonly"]:
        files += glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), root, "**", "*.py"),
                           recursive=True)
    txt, total = [], 0
    for f in files:
        try:
            s = open(f, encoding="utf-8", errors="ignore").read()
        except Exception:
            continue
        txt.append(s); total += len(s)
        if total >= cap:
            break
    return "\n".join(txt)


def token_stack(tok_texts):
    """per-token (stack_top, depth, valid) with a light string/comment skip; window-local."""
    stack, in_str, in_com, valid = [], None, False, True
    out = []
    for t in tok_texts:
        for ch in t:
            if in_com:
                if ch == "\n":
                    in_com = False
                continue
            if in_str:
                if ch == in_str:
                    in_str = None
                continue
            if ch == "#":
                in_com = True; continue
            if ch in ("'", '"'):
                in_str = ch; continue
            if ch in "([{":
                stack.append(ch)
            elif ch in ")]}":
                if stack and stack[-1] == BR[ch]:
                    stack.pop()
                else:
                    valid = False
        out.append((stack[-1] if stack else None, len(stack), valid, len(set(stack))))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=400); a = ap.parse_args()
    import torch
    from transformers import GPT2TokenizerFast, GPT2Model
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    print(f"H2_STACKTOP_PROBE  GPT-2 small (CPU)  real Python code  [NON-PROMOTIONAL]\n", flush=True)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    code = load_code()
    ids = tok(code)["input_ids"]
    W, N = 128, a.n
    starts = np.random.default_rng(0).integers(0, max(1, len(ids) - W), size=N)
    windows = [ids[s:s + W] for s in starts]
    tok_str = {i: tok.decode([i]) for i in set(x for w in windows for x in w)}

    torch.set_grad_enabled(False)
    model = GPT2Model.from_pretrained("gpt2").eval()
    print("(extract) GPT-2 residual over code windows ...", flush=True)
    query = []  # (layer_hidden dict, prefix_text, stacktop, depthbucket)
    Hs = []
    for i in range(0, len(windows), 48):
        batch = windows[i:i + 48]
        arr = torch.tensor([w + [0] * (W - len(w)) for w in batch])
        hs = model(arr, output_hidden_states=True).hidden_states  # tuple L+1 [b,W,d]
        Hs.append([h.numpy() for h in hs])
    LAYERS = [4, 8, 11]
    C2I = {"(": 0, "[": 1, "{": 2}
    rows = []
    bi = 0
    for i in range(0, len(windows), 48):
        batch = windows[i:i + 48]; hb = Hs[bi]; bi += 1
        for r, w in enumerate(batch):
            states = token_stack([tok_str[t] for t in w])
            for j in range(6, len(w)):
                top, depth, valid, ndist = states[j]
                if not valid or depth < 1:
                    continue
                if np.random.default_rng(i + r + j).random() > 0.5:   # subsample positions
                    continue
                prefix = "".join(tok_str[t] for t in w[:j + 1])
                hard = ndist >= 2   # >=2 distinct types UNCLOSED: counts ambiguous, top is order-determined
                feat = {L: hb[L][r, j] for L in LAYERS}
                rows.append((feat, prefix, top, min(depth, 3), hard))
    n_hard = sum(r[4] for r in rows)
    print(f"(data) {len(rows)} query positions (depth>=1, valid); {n_hard} HARD (>=2 distinct types unclosed)", flush=True)
    if len(rows) < 300:
        print("  too few positions; increase --n."); return

    def balance(rs):
        buckets = {0: [], 1: [], 2: []}
        for r in rs:
            buckets[C2I[r[2]]].append(r)
        m = min(len(v) for v in buckets.values())
        rng = np.random.default_rng(2); out = []
        for v in buckets.values():
            out += [v[k] for k in rng.choice(len(v), m, replace=False)]
        return out

    def count_feats(prefix):   # pure ORDER-BLIND sufficient statistic: bracket counts
        o = [prefix.count(c) for c in "([{"]; c = [prefix.count(c) for c in ")]}"]
        unclosed = [o[k] - c[k] for k in range(3)]
        return o + c + unclosed + [sum(unclosed)]

    def evaluate(rs, tag):
        rs = balance(rs); n = len(rs)
        if n < 180:
            print(f"\n  [{tag}] too few after class-balancing ({n}); skip."); return None
        idx = np.arange(n); np.random.default_rng(1).shuffle(idx)
        tr, te = idx[:int(.7 * n)], idx[int(.7 * n):]
        ys_t = np.array([C2I[r[2]] for r in rs]); yd = np.array([r[3] for r in rs])
        maj_t = np.bincount(ys_t).max() / n; maj_d = np.bincount(yd).max() / n

        def acc(X, y):
            return float(LogisticRegression(max_iter=400).fit(X[tr], y[tr]).score(X[te], y[te]))
        Xbag = CountVectorizer(max_features=3000).fit_transform([r[1] for r in rs]).toarray()
        Xcnt = np.array([count_feats(r[1]) for r in rs], float)
        Xcnt = (Xcnt - Xcnt[tr].mean(0)) / (Xcnt[tr].std(0) + 1e-6)
        surf_bag = {"stack-top": acc(Xbag, ys_t), "depth": acc(Xbag, yd)}
        surf_cnt = {"stack-top": acc(Xcnt, ys_t), "depth": acc(Xcnt, yd)}
        print(f"\n  [{tag}]  n(balanced)={n}")
        print(f"  {'quantity':>10} {'chance':>7} {'surf-counts*':>12} {'surf-bag':>9} {'residual(best)':>15}")
        out = {}
        for name, y, maj in [("stack-top", ys_t, maj_t), ("depth", yd, maj_d)]:
            res = {}
            for L in LAYERS:
                X = np.array([r[0][L] for r in rs]); X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
                res[L] = acc(X, y)
            bestL = max(res, key=res.get)
            print(f"  {name:>10} {maj:>7.3f} {surf_cnt[name]:>12.3f} {surf_bag[name]:>9.3f} {res[bestL]:>11.3f} (L{bestL})")
            out[name] = (surf_cnt[name], surf_bag[name], res[bestL])
        return out  # (*surf-counts = pure order-blind statistic; the decisive baseline)

    all_r = evaluate(rows, "ALL positions (class-balanced)")
    hard_r = evaluate([r for r in rows if r[4]], "HARD positions (>=2 distinct types unclosed = count-ambiguous, balanced)")

    print()
    if hard_r is None:
        print("  VERDICT: inconclusive — too few hard positions after balancing; raise --n."); return
    sc_top, sb_top, r_top = hard_r["stack-top"]; sc_dep, _, _ = hard_r["depth"]
    s_top = sc_top   # decisive baseline = the PURE order-blind statistic (counts)
    if r_top >= s_top + 0.15 and sc_dep >= 0.70:
        v = (f"H2 CONFIRMED — at count-ambiguous positions the residual reads the ORDER-DEPENDENT stack-top "
             f"({r_top:.3f}) far above the pure order-blind statistic (counts, {s_top:.3f}), while counts DO read "
             f"depth ({sc_dep:.3f}, control live): stack-top is IN the intersection. State-tracking answer holds.")
    elif s_top >= r_top - 0.05:
        v = (f"H2 FALSIFIED (GPT-2/code) — even at count-ambiguous positions the order-blind count statistic "
             f"({s_top:.3f}) ~ residual ({r_top:.3f}): stack-top stays surface-decodable. Needs a code/bigger model.")
    else:
        v = (f"H2 PARTIAL — residual ({r_top:.3f}) > order-blind counts ({s_top:.3f}) at count-ambiguous "
             f"positions but margin <0.15; leans state-tracking, not decisive on GPT-2-small.")
    print(f"  VERDICT: {v}")
    print("  (Non-promotional. No R2 promotion / public / R3 / world-model claim.)")


if __name__ == "__main__":
    main()
