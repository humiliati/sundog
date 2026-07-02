#!/usr/bin/env python3
"""ME-4 -- the probe-steer gap on GPT-2's stack-top (write side).
Pre-reg: docs/orderrelative/ME4_PROBE_STEER_SPEC.md.  NON-promotional, CPU.
Chat-v2 discipline inherited: no R2 / promotion / world-model language.

The read is banked (chatv2_h2_stacktop_probe.py, FROZEN -- its corpus, lexer,
window, and probe protocol are replicated here unchanged).  This script tests
the WRITE: patch the residual at the probe's address toward the ONE-POP SWAP
target (the alternative unclosed type) and ask whether the model's
closing-bracket preference follows.

Gates (pre-registered): G0 behavioral floor (unpatched closer preference tracks
the true top >= 0.60 on the primary slice); G1 write validity (the probe must
read tau' off the PATCHED residual >= 0.90 -- "the read must take the write");
G2 on-manifold (rest-of-continuation NLL <= 3x unpatched median, excluding the
immediate next token whose change is the intended effect; random-direction
control moves preference < half the treatment); G3 follow_rel =
follow(tau') / agree(tau) on the primary slice: >= 0.75 STEERS / <= 0.40
everywhere RESISTS / middle PARTIAL.

Implementation notes (measurement protocol, pinned):
- hidden_states[L] is the output of transformer block L-1, so patching the
  probe's layer-L representation hooks blocks[L-1].
- Restricted 3-way readout over the bare closer token ids ) ] } (BPE-merge
  caveat: relative preference only).
- Probe-direction vectors live in RAW residual space: d = (w_t' - w_t)/s for a
  probe trained on standardized features (chain rule through z=(h-m)/s);
  normalized to the norm of the diff-means vector so alpha is comparable.
- Donors are probe-verified (probe reads the donor's top off the donor's own
  activation) -- the transplant is a write the read address provably accepts.
- Per-chunk JSON checkpoints (teardown gotcha); fixed seeds throughout.

Run: python scripts/chatv2_me4_probe_steer.py [--n 800] [--max-queries 600]
"""
import os, glob, json, argparse
import numpy as np

BR = {")": "(", "]": "[", "}": "{"}
CLOSER_OF = {"(": ")", "[": "]", "{": "}"}
C2I = {"(": 0, "[": 1, "{": 2}
I2C = {v: k for k, v in C2I.items()}
LAYERS = [8, 11]
ALPHAS = [2.0, 4.0]
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..",
                      "results", "orderrelative", "me4-probe-steer")
CONT = 8  # continuation tokens scored for G2 (excluding the immediate next token)


def load_code(cap=700_000):
    """FROZEN: identical to chatv2_h2_stacktop_probe.load_code."""
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


def token_stack_full(tok_texts):
    """FROZEN lexer semantics of chatv2_h2_stacktop_probe.token_stack, extended to
    also emit the full stack copy and the post-token (in_str, in_com) state."""
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
        out.append((list(stack), valid, in_str, in_com))
    return out


def alt_top(stack):
    """The one-pop-swap target: topmost unclosed opener of a DIFFERENT type."""
    top = stack[-1]
    for ch in reversed(stack[:-1]):
        if ch != top:
            return ch
    return None


def closer_initial(text):
    s = text.lstrip(" \t")
    return s[0] if (s and s[0] in ")]}") else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--max-queries", type=int, default=600)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    import torch
    from transformers import GPT2TokenizerFast, GPT2LMHeadModel
    from sklearn.linear_model import LogisticRegression
    torch.set_grad_enabled(False)
    os.makedirs(OUTDIR, exist_ok=True)
    print("ME4_PROBE_STEER  GPT-2 small (CPU)  [NON-PROMOTIONAL]", flush=True)

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    closer_ids = {}
    for c in ")]}":
        enc = tok.encode(c)
        assert len(enc) == 1, f"closer {c} not a single token"
        closer_ids[c] = enc[0]
    cid_list = [closer_ids[c] for c in ")]}"]

    code = load_code()
    ids = tok(code)["input_ids"]
    W, N = 128, (60 if a.smoke else a.n)
    starts = np.random.default_rng(0).integers(0, max(1, len(ids) - W), size=N)
    windows = [ids[s:s + W] for s in starts]
    tok_str = {i: tok.decode([i]) for i in set(x for w in windows for x in w)}
    n_probe = int(0.6 * len(windows))
    probe_windows, treat_windows = windows[:n_probe], windows[n_probe:]

    lm = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    trans = lm.transformer

    # ---------------- probe bank: H2-protocol rows from the probe windows ------
    print("(probe bank) extracting residuals ...", flush=True)
    probe_rows = []   # (hid{L}, top, ndist, next_closer_char_or_None)
    for i in range(0, len(probe_windows), 48):
        batch = probe_windows[i:i + 48]
        arr = torch.tensor([w + [0] * (W - len(w)) for w in batch])
        hs = trans(arr, output_hidden_states=True).hidden_states
        for r, w in enumerate(batch):
            states = token_stack_full([tok_str[t] for t in w])
            for j in range(6, len(w) - 1):
                stack, valid, in_str, in_com = states[j]
                if not valid or len(stack) < 1 or in_str or in_com:
                    continue
                if np.random.default_rng(i + r + j).random() > 0.5:  # FROZEN subsample
                    continue
                nc = closer_initial(tok_str[w[j + 1]])
                probe_rows.append(({L: hs[L][r, j].numpy() for L in LAYERS},
                                   stack[-1], len(set(stack)), nc))
    print(f"(probe bank) {len(probe_rows)} rows", flush=True)

    # class-balanced probe training (H2 protocol), one probe per layer
    rng_b = np.random.default_rng(2)
    buckets = {0: [], 1: [], 2: []}
    for r in probe_rows:
        buckets[C2I[r[1]]].append(r)
    m = min(len(v) for v in buckets.values())
    bal = []
    for v in buckets.values():
        bal += [v[k] for k in rng_b.choice(len(v), m, replace=False)]
    probes, scalers = {}, {}
    for L in LAYERS:
        X = np.array([r[0][L] for r in bal]); y = np.array([C2I[r[1]] for r in bal])
        mu, sd = X.mean(0), X.std(0) + 1e-6
        idx = np.arange(len(X)); np.random.default_rng(1).shuffle(idx)
        tr, te = idx[:int(.7 * len(X))], idx[int(.7 * len(X)):]
        clf = LogisticRegression(max_iter=400).fit(((X - mu) / sd)[tr], y[tr])
        acc = clf.score(((X - mu) / sd)[te], y[te])
        probes[L], scalers[L] = clf, (mu, sd)
        print(f"(probe) L{L} holdout acc {acc:.3f} (n={len(X)})", flush=True)

    def probe_read(L, h):
        mu, sd = scalers[L]
        return int(probes[L].predict(((h - mu) / sd).reshape(1, -1))[0])

    # class-conditional raw means + probe-direction vectors (raw space)
    mean_vec = {L: {} for L in LAYERS}
    for L in LAYERS:
        X = np.array([r[0][L] for r in bal]); y = np.array([C2I[r[1]] for r in bal])
        for c in range(3):
            mean_vec[L][c] = X[y == c].mean(0)

    def steer_vec(L, fam, c_from, c_to):
        dm = mean_vec[L][c_to] - mean_vec[L][c_from]
        if fam == "diffmeans":
            return dm
        mu, sd = scalers[L]
        w = probes[L].coef_
        d = (w[c_to] - w[c_from]) / sd
        return d * (np.linalg.norm(dm) / (np.linalg.norm(d) + 1e-9))

    # donor pool: probe-verified hard next-closer positions from the probe bank
    donors = {L: {0: [], 1: [], 2: []} for L in LAYERS}
    for r in probe_rows:
        if r[2] >= 2 and r[3] == CLOSER_OF.get(r[1]):
            for L in LAYERS:
                if probe_read(L, r[0][L]) == C2I[r[1]]:
                    donors[L][C2I[r[1]]].append(r[0][L])
    print("(donors)", {L: {I2C[c]: len(v) for c, v in d.items()} for L, d in donors.items()}, flush=True)

    # ---------------- query set: next-real-token-is-closer positions -----------
    # deduplicated by ABSOLUTE corpus position (overlapping windows would
    # double-count the same code position)
    queries = []  # dict(window w, pos j, top, alt, hard)
    seen_abs = set()
    for wi, w in enumerate(treat_windows):
        start = int(starts[n_probe + wi])
        states = token_stack_full([tok_str[t] for t in w])
        for j in range(6, len(w) - 1 - CONT):
            if (start + j) in seen_abs:
                continue
            stack, valid, in_str, in_com = states[j]
            if not valid or len(stack) < 1 or in_str or in_com:
                continue
            nc = closer_initial(tok_str[w[j + 1]])
            if nc != CLOSER_OF[stack[-1]]:
                continue
            seen_abs.add(start + j)
            hard = len(set(stack)) >= 2
            alt = alt_top(stack) if hard else None
            queries.append(dict(w=w, j=j, top=stack[-1], alt=alt, hard=hard))
    hard_q = [q for q in queries if q["hard"]]
    rng_q = np.random.default_rng(5)
    if len(hard_q) > a.max_queries:
        hard_q = [hard_q[k] for k in sorted(rng_q.choice(len(hard_q), a.max_queries, replace=False))]
    print(f"(queries) {len(queries)} next-closer positions; primary (hard) slice n={len(hard_q)}", flush=True)
    if len(hard_q) < (10 if a.smoke else 150):
        print("  too few hard next-closer positions; raise --n."); return

    # ---------------- patched forward machinery --------------------------------
    def run(wtoks, j, L=None, vec=None, add=None):
        """Forward over wtoks[:j+1+CONT+1]; optional patch at position j on the
        layer-L representation (= output of block L-1). Returns (closer logits at j,
        rest-NLL over tokens j+2..j+1+CONT, hidden at (L,j) per LAYERS)."""
        seq = wtoks[:j + 2 + CONT]
        arr = torch.tensor([seq])
        handle = None
        if L is not None:
            def hook(module, inp, out):
                h = out[0] if isinstance(out, tuple) else out
                if vec is not None:
                    h[:, j, :] = torch.tensor(vec, dtype=h.dtype)
                if add is not None:
                    h[:, j, :] = h[:, j, :] + torch.tensor(add, dtype=h.dtype)
                return ((h,) + tuple(out[1:])) if isinstance(out, tuple) else h
            handle = trans.h[L - 1].register_forward_hook(hook)
        try:
            out = lm(arr, output_hidden_states=True)
        finally:
            if handle is not None:
                handle.remove()
        logits = out.logits[0]
        closer_log = {c: float(logits[j, closer_ids[c]]) for c in ")]}"}
        lp = torch.log_softmax(logits, dim=-1)
        rest = [float(lp[p, seq[p + 1]]) for p in range(j + 1, min(j + 1 + CONT, len(seq) - 1))]
        rest_nll = -float(np.mean(rest)) if rest else 0.0
        hid = {L2: out.hidden_states[L2][0, j].numpy() for L2 in LAYERS}
        return closer_log, rest_nll, hid

    def pref(closer_log):
        return max(closer_log, key=closer_log.get)

    # ---------------- baseline pass (G0) ---------------------------------------
    print("(baseline) unpatched pass ...", flush=True)
    base = []
    for q in hard_q:
        cl, rnll, hid = run(q["w"], q["j"])
        base.append(dict(pref=pref(cl), rest_nll=rnll, hid=hid))
    agree = float(np.mean([b["pref"] == CLOSER_OF[q["top"]] for q, b in zip(hard_q, base)]))
    med_rest = float(np.median([b["rest_nll"] for b in base]))
    all_pref = []
    for q in queries[: min(len(queries), 1000)]:
        cl, _, _ = run(q["w"], q["j"])
        all_pref.append(pref(cl) == CLOSER_OF[q["top"]])
    agree_all = float(np.mean(all_pref))
    print(f"(G0) agree(top) primary={agree:.3f} (floor 0.60); all-slice={agree_all:.3f}; median rest-NLL={med_rest:.3f}", flush=True)
    summary = dict(phase="ME-4 probe-steer", spec="docs/orderrelative/ME4_PROBE_STEER_SPEC.md",
                   n_primary=len(hard_q), agree_primary=agree, agree_all=agree_all,
                   median_rest_nll=med_rest, cells={})
    if agree < 0.60:
        summary["branch"] = "ME4_BEHAVIORAL_FLOOR"
        json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
        print("VERDICT: ME4_BEHAVIORAL_FLOOR — substrate inadequate (not a resistance finding).")
        return

    # ---------------- treatment cells ------------------------------------------
    cells = [("probedir", L, al) for L in LAYERS for al in ALPHAS] \
        + [("diffmeans", L, al) for L in LAYERS for al in ALPHAS] \
        + [("transplant", L, None) for L in LAYERS] \
        + [("randdir", L, al) for L in LAYERS for al in ALPHAS] \
        + [("nullswap", L, None) for L in LAYERS]

    for fam, L, al in cells:
        key = f"{fam}_L{L}" + (f"_a{al:g}" if al is not None else "")
        ck = os.path.join(OUTDIR, f"cell_{key}.json")
        if os.path.exists(ck):
            summary["cells"][key] = json.load(open(ck)); print(f"(cell {key}) cached", flush=True); continue
        took = follow = flip_any = 0; rest_ratio = []
        n_used = 0
        for qi, (q, b) in enumerate(zip(hard_q, base)):
            ci_top, ci_alt = C2I[q["top"]], C2I[q["alt"]]
            rng_p = np.random.default_rng(7000 + qi)
            if fam in ("probedir", "diffmeans"):
                add, vec = al * steer_vec(L, fam, ci_top, ci_alt), None
            elif fam == "randdir":
                g = rng_p.standard_normal(768)
                add = g * (al * np.linalg.norm(steer_vec(L, "diffmeans", ci_top, ci_alt)) / np.linalg.norm(g))
                vec = None
            elif fam == "transplant":
                pool = donors[L][ci_alt]
                if not pool:
                    continue
                vec, add = pool[int(rng_p.integers(len(pool)))], None
            else:  # nullswap: donor with the SAME top (control)
                pool = donors[L][ci_top]
                if not pool:
                    continue
                vec, add = pool[int(rng_p.integers(len(pool)))], None
            cl, rnll, hid = run(q["w"], q["j"], L=L, vec=vec, add=add)
            n_used += 1
            if probe_read(L, hid[L]) == ci_alt:
                took += 1
            p = pref(cl)
            if p == CLOSER_OF[q["alt"]]:
                follow += 1
            if p != b["pref"]:
                flip_any += 1
            rest_ratio.append(rnll / (med_rest + 1e-9))
        cell = dict(n=n_used, g1_took=took / max(1, n_used), follow=follow / max(1, n_used),
                    follow_rel=(follow / max(1, n_used)) / max(agree, 1e-9),
                    flip_any=flip_any / max(1, n_used),
                    rest_nll_ratio_med=float(np.median(rest_ratio)) if rest_ratio else None)
        summary["cells"][key] = cell
        json.dump(cell, open(ck, "w"), indent=2)
        print(f"(cell {key}) n={cell['n']} G1={cell['g1_took']:.3f} follow={cell['follow']:.3f} "
              f"follow_rel={cell['follow_rel']:.3f} flip_any={cell['flip_any']:.3f} "
              f"restNLLx={cell['rest_nll_ratio_med']:.2f}", flush=True)

    # ---------------- gates + branch -------------------------------------------
    def cget(fam, L, al=None):
        return summary["cells"].get(f"{fam}_L{L}" + (f"_a{al:g}" if al is not None else ""))

    treat_keys = [(f, L, al) for f in ("probedir", "diffmeans") for L in LAYERS for al in ALPHAS] \
        + [("transplant", L, None) for L in LAYERS]
    valid_cells = []
    for f, L, al in treat_keys:
        c = cget(f, L, al)
        if c is None or c["n"] == 0:
            continue
        g1 = c["g1_took"] >= 0.90
        g2a = c["rest_nll_ratio_med"] is not None and c["rest_nll_ratio_med"] <= 3.0
        ctrl = cget("randdir", L, al) if al is not None else cget("nullswap", L)
        g2b = ctrl is None or ctrl["n"] == 0 or (ctrl["follow"] < 0.5 * max(c["follow"], 1e-9)) \
            or c["follow"] <= ctrl["follow"]  # if control >= treatment, the cell can't claim STEERS anyway
        c["gates"] = dict(G1=g1, G2a=g2a, G2b_ctrl_clean=ctrl is None or ctrl["follow"] < 0.5 * max(c["follow"], 1e-9))
        if g1 and g2a:
            valid_cells.append((f, L, al, c))
    summary["decision"] = {}
    if not valid_cells:
        branch = "ME4_CONFOUNDED"
        note = "no treatment cell passed G1+G2a — the write never demonstrably took at the read address (or wrecked the manifold)"
    else:
        best = max(valid_cells, key=lambda x: x[3]["follow_rel"])
        f, L, al, c = best
        ctrl = cget("randdir", L, al) if al is not None else cget("nullswap", L)
        ctrl_clean = ctrl is None or ctrl["n"] == 0 or ctrl["follow"] < 0.5 * max(c["follow"], 1e-9)
        summary["decision"] = dict(best_cell=f"{f}_L{L}" + (f"_a{al:g}" if al is not None else ""),
                                   follow_rel=c["follow_rel"], control_clean=bool(ctrl_clean))
        if c["follow_rel"] >= 0.75 and ctrl_clean:
            branch = "ME4_STEERS"
            note = "the linear address is causally load-bearing at this cell — census: model-state joins M-and-E"
        elif all(x[3]["follow_rel"] <= 0.40 for x in valid_cells):
            branch = "ME4_RESISTS"
            note = ("the write takes at the read address (G1) and the manifold survives (G2), "
                    "yet behavior does not follow — census: model-state joins M-and-not-E "
                    "(the read address accepts the write; the computation doesn't consult it)")
        else:
            branch = "ME4_PARTIAL"
            note = "bounded-partial band — steering shortfall recorded as a price (feeds ME-5); census row graded"
    summary["branch"] = branch
    summary["note"] = note
    json.dump(summary, open(os.path.join(OUTDIR, "summary.json"), "w"), indent=2)
    print(f"\nVERDICT: {branch} — {note}")
    print("(Non-promotional. No R2 promotion / public / R3 / world-model claim.)")


if __name__ == "__main__":
    main()
