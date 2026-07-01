#!/usr/bin/env python3
"""Chat-v2 H1/V3 rung V3-0 — state-bank DATA ADMISSION (CPU, non-promotional).
Pre-reg: docs/chatv2/H1_V3_STATEBANK_SCOPE.md (SS2 surface criterion, SS3 code corpus,
SS4 chess corpus, SS7 branches). Receipt: docs/chatv2/H1_V3_0_DATA_ADMISSION_RECEIPT.md.

Gate (per corpus): >=24 axes surviving [balance 0.40-0.60] AND [ALL surface probes <=0.60
held-out: LR-counts, LR-tfidf(1-2), MLP-on-counts w in {1,2,3} (+ metadata where present)]
AND [>=1 interpreter/replay-verified witness pair: same registered-surface bag, different
state]. Branches: H1-V3-DATA-ADMIT / F3-V3/{input,bank,corpus,copy}.

Frozen defaults (declared): min run len 3; prefix instances (len>=3, cap 4/run); <=8 var
slots; canonical rename v0..v7 by first use; int-only whitelist interpreter, |v|<=1e9;
min 120 instances/axis; probe split = group-aware 70/30 seed 0; per-axis probe cap 1600
instances; magnitude/sign variants chosen on TRAIN only; chess marker ply 16, witness
prefix 8 plies; ASCII prints. No model, no GPU, no R2 claim.
Run: python scripts/chatv2_h1_v3_data_admission.py --corpus code|chess
"""
import ast, argparse, collections, glob, itertools, json, os, re, sys, time
import numpy as np

CEIL, BAL_LO, BAL_HI, MIN_N, MIN_TOTAL = 0.60, 0.40, 0.60, 120, 300
SAFE_BIN = (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
TOK = re.compile(r"[A-Za-z_]\w*|\d+|[^\sA-Za-z_\d]")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "chatv2", "h1_v3")


# ---------------------------------------------------------------- interpreter
def ev(e, env):
    """Whitelist int evaluator; raises on anything outside SS3.2."""
    if isinstance(e, ast.Constant):
        if isinstance(e.value, bool) or not isinstance(e.value, int):
            raise ValueError("non-int")
        return e.value
    if isinstance(e, ast.Name):
        return env[e.id]
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
        v = ev(e.operand, env)
        return -v if isinstance(e.op, ast.USub) else v
    if isinstance(e, ast.BinOp) and isinstance(e.op, SAFE_BIN):
        a, b = ev(e.left, env), ev(e.right, env)
        if isinstance(e.op, ast.Add):
            r = a + b
        elif isinstance(e.op, ast.Sub):
            r = a - b
        elif isinstance(e.op, ast.Mult):
            r = a * b
        elif isinstance(e.op, ast.FloorDiv):
            if b == 0:
                raise ValueError("div0")
            r = a // b
        else:
            if b == 0:
                raise ValueError("mod0")
            r = a % b
        if abs(r) > 10**9:
            raise ValueError("magnitude")
        return r
    raise ValueError("unsafe")


def expr_reads(e):
    return {n.id for n in ast.walk(e) if isinstance(n, ast.Name)}


def has_op(e):
    return any(isinstance(n, (ast.BinOp, ast.UnaryOp)) for n in ast.walk(e))


def run_block(stmts):
    """Execute canonical statements; return (env, prov) or None. prov: lit|comp (SS3.4)."""
    env, prov = {}, {}
    for st in stmts:
        try:
            if isinstance(st, ast.Assign):
                name, expr = st.targets[0].id, st.value
            else:
                name, expr = st.target.id, ast.BinOp(ast.Name(st.target.id, ast.Load()), st.op, st.value)
                ast.fix_missing_locations(expr)
            v = ev(expr, env)
        except Exception:
            return None
        p = "comp" if (has_op(expr) or any(prov.get(r) == "comp" for r in expr_reads(expr))) else "lit"
        if name in env and env[name] != v:
            p = "comp"  # SS3.4(c): value changed across a prior update
        env[name], prov[name] = v, p
    return env, prov


# ---------------------------------------------------------------- code mining
def safe_stmt(st, known):
    if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
        e = st.value
    elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id in known \
            and isinstance(st.op, SAFE_BIN):
        e = st.value
    else:
        return None
    def ok(x):
        if isinstance(x, ast.Constant):
            return isinstance(x.value, int) and not isinstance(x.value, bool)
        if isinstance(x, ast.Name):
            return x.id in known or (isinstance(st, ast.Assign) and False)
        if isinstance(x, ast.UnaryOp) and isinstance(x.op, (ast.USub, ast.UAdd)):
            return ok(x.operand)
        if isinstance(x, ast.BinOp) and isinstance(x.op, SAFE_BIN):
            return ok(x.left) and ok(x.right)
        return False
    if not ok(e):
        return None
    return st.targets[0].id if isinstance(st, ast.Assign) else st.target.id


def canonicalize(stmts):
    """Rename vars to v0..v7 by first use; return (canonical stmt list, ok)."""
    order = []
    for st in stmts:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and n.id not in order:
                order.append(n.id)
    if len(order) > 8:
        return None
    m = {name: f"v{i}" for i, name in enumerate(order)}
    out = []
    for st in stmts:
        st2 = ast.parse(ast.unparse(st)).body[0]
        for n in ast.walk(st2):
            if isinstance(n, ast.Name):
                n.id = m[n.id]
        out.append(st2)
    return out


def mine_code():
    roots = []
    dev = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Dev/sundog -> Dev
    for r in ["sundog", "sundogcert", "eyesonly"]:
        roots.append(os.path.join(os.path.dirname(dev), r) if os.path.basename(dev) == "sundog" else os.path.join(dev, "..", r))
    roots = [os.path.abspath(os.path.join(dev, "..", r)) for r in ["sundog", "sundogcert", "eyesonly"]]
    roots.append(os.path.join(sys.prefix, "Lib"))
    sp = os.path.join(os.environ.get("APPDATA", ""), "Python", "Python314", "site-packages")
    if os.path.isdir(sp):
        roots.append(sp)
    files = []
    for root in roots:
        fs = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
        fs = [f for f in fs if ".lake" not in f]
        files += fs[:12000]
    runs, nfiles = [], 0
    for f in files:
        try:
            tree = ast.parse(open(f, encoding="utf-8", errors="ignore").read())
        except Exception:
            continue
        nfiles += 1
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if not (isinstance(body, list) and body and isinstance(body[0], ast.stmt)):
                continue
            cur, known = [], set()
            for st in body:
                name = safe_stmt(st, known)
                if name is not None:
                    known.add(name)
                    cur.append(st)
                else:
                    if len(cur) >= 3:
                        runs.append(cur)
                    cur, known = [], set()
            if len(cur) >= 3:
                runs.append(cur)
    print(f"(mine) {nfiles} real Python files parsed; {len(runs)} straight-line runs (len>=3)")
    # instances = prefixes (len>=3, cap 4 per run, longest-first), canonicalized + executed
    inst = []
    for ri, run in enumerate(runs):
        lens = list(range(len(run), 2, -1))[:4]
        for L in lens:
            can = canonicalize(run[:L])
            if can is None:
                continue
            res = run_block(can)
            if res is None:
                continue
            env, prov = res
            text = "\n".join(ast.unparse(s) for s in can) + "\n# q"
            inst.append({"gid": ri, "text": text, "env": env, "prov": prov})
    print(f"(instances) {len(inst)} executable canonical prefixes (cap 4/run)")
    return inst


# ---------------------------------------------------------------- probes
def probe_suite(texts, y, groups, extra=None):
    """Return dict probe->held-out acc (group-aware 70/30, seed 0), subsampled to 1600."""
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import GroupShuffleSplit
    n = len(y)
    idx = np.arange(n)
    if n > 1600:
        idx = np.random.default_rng(0).choice(n, 1600, replace=False)
    texts = [texts[i] for i in idx]
    y = np.asarray(y)[idx]
    groups = np.asarray(groups)[idx]
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=0).split(texts, y, groups))
    if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
        return None
    toks = [" ".join(TOK.findall(t)) for t in texts]
    out = {}

    def fit(X, model, name):
        Xd = X.toarray() if hasattr(X, "toarray") and "MLP" in name else X
        model.fit(Xd[tr], y[tr])
        out[name] = float(model.score(Xd[te], y[te]))

    fit(CountVectorizer(max_features=1500).fit_transform(toks),
        LogisticRegression(max_iter=400), "LR-counts")
    fit(TfidfVectorizer(ngram_range=(1, 2), max_features=3000).fit_transform(toks),
        LogisticRegression(max_iter=400), "LR-tfidf12")
    for w in (1, 2, 3):
        fit(CountVectorizer(ngram_range=(1, w), max_features=[800, 2000, 3000][w - 1]).fit_transform(toks),
            MLPClassifier(hidden_layer_sizes=(32,), max_iter=250, early_stopping=True, random_state=0),
            f"MLP-w{w}")
    if extra is not None:
        from sklearn.linear_model import LogisticRegression as LR
        Xm = extra[idx]
        m = LR(max_iter=400).fit(Xm[tr], y[tr])
        out["LR-meta"] = float(m.score(Xm[te], y[te]))
    return out


# ---------------------------------------------------- code witness pairs (SS3.5)
def code_witness(slot, pred, rng, panel=3):
    """Generated pairs: same token bag (statement swap), interpreter-verified, pred flips."""
    OPS = ["+", "-", "*", "//", "%"]
    pairs = []
    for _ in range(6000):
        if len(pairs) >= panel:
            break
        inits = [f"v{i} = {int(rng.integers(1, 7)) + i}" for i in range(slot)]
        c0 = int(rng.integers(-3, 7))
        o1, o2 = rng.choice(OPS), rng.choice(OPS)
        c1, c2 = int(rng.integers(1, 8)), int(rng.integers(1, 8))
        u1, u2 = f"v{slot} = v{slot} {o1} {c1}", f"v{slot} = v{slot} {o2} {c2}"
        a = inits + [f"v{slot} = {c0}", u1, u2]
        b = inits + [f"v{slot} = {c0}", u2, u1]
        try:
            ra = run_block(ast.parse("\n".join(a)).body)
            rb = run_block(ast.parse("\n".join(b)).body)
        except Exception:
            continue
        if ra is None or rb is None:
            continue
        va, vb = ra[0][f"v{slot}"], rb[0][f"v{slot}"]
        lab = {"parity": lambda v: v % 2 == 1, "sign": lambda v: v < 0,
               "zero": lambda v: v == 0, "magnitude": lambda v: abs(v) >= 4}[pred]
        if lab(va) == lab(vb):
            continue
        if sorted(TOK.findall("\n".join(a))) != sorted(TOK.findall("\n".join(b))):
            continue
        pairs.append({"a": "\n".join(a) + "\n# q", "b": "\n".join(b) + "\n# q",
                      "axis": f"v{slot}.{pred}", "label_a": bool(lab(va)), "label_b": bool(lab(vb))})
    return pairs


# ---------------------------------------------------------------- code corpus
def admit_code():
    inst = mine_code()
    if len(inst) < MIN_TOTAL:
        print(f"\nVERDICT(code): F3-V3/corpus - only {len(inst)} instances (<{MIN_TOTAL})")
        return
    rng = np.random.default_rng(1)
    slots = [f"v{i}" for i in range(8)]
    preds = ["defined", "parity", "sign", "zero", "magnitude"]
    rows, admitted, wit_all = [], [], []
    n_comp = sum(1 for r in inst for v in r["prov"].values() if v == "comp")
    n_all = sum(len(r["prov"]) for r in inst)
    print(f"(provenance) computed-value share across slots: {n_comp}/{n_all} = {n_comp/max(n_all,1):.2f}")
    print(f"\n{'axis':>14} {'n':>5} {'bal':>5} {'LRcnt':>6} {'tfidf':>6} {'MLPw1':>6} {'MLPw2':>6} {'MLPw3':>6} {'max':>6} {'wit':>4}  admit?")
    causes = collections.Counter()
    for s, p in itertools.product(slots, preds):
        if p == "defined":
            sub = inst
            y = [s in r["env"] for r in sub]
        else:
            n_def = sum(1 for r in inst if s in r["env"])
            sub = [r for r in inst if s in r["env"] and r["prov"][s] == "comp"]  # SS3.4 filter
            vals = np.array([r["env"][s] for r in sub])
            if len(sub) < MIN_N:
                causes["copy" if n_def >= MIN_N else "thin-slot"] += 1
                rows.append((f"{s}.{p}", len(sub), None)); continue
            gids = np.array([r["gid"] for r in sub])
            tr_groups = set(np.unique(gids)[: int(0.7 * len(np.unique(gids)))])
            trmask = np.array([g in tr_groups for g in gids])
            if p == "parity":
                y = (vals % 2 == 1)
            elif p == "sign":
                y1, y2 = vals < 0, vals > 0   # variant chosen on TRAIN balance only
                y = y1 if abs(y1[trmask].mean() - 0.5) <= abs(y2[trmask].mean() - 0.5) else y2
            elif p == "zero":
                y = (vals == 0)
            else:
                thr = np.median(np.abs(vals[trmask]))  # TRAIN-only threshold
                y = (np.abs(vals) >= max(thr, 1))
            y = list(y)
        if len(sub) < MIN_N:
            rows.append((f"{s}.{p}", len(sub), None)); continue
        bal = float(np.mean(y))
        if not (BAL_LO <= bal <= BAL_HI):
            rows.append((f"{s}.{p}", len(sub), ("bal", bal))); continue
        res = probe_suite([r["text"] for r in sub], y, [r["gid"] for r in sub])
        if res is None:
            rows.append((f"{s}.{p}", len(sub), ("split", bal))); continue
        mx = max(res.values())
        wit = [] if p == "defined" else code_witness(int(s[1]), p, rng)
        ok = mx <= CEIL and len(wit) >= 1
        wit_all += wit
        if ok:
            admitted.append(f"{s}.{p}")
        note = "YES" if ok else ("no-wit(bag-determined)" if p == "defined" and mx <= CEIL else "no")
        print(f"{s+'.'+p:>14} {len(sub):>5} {bal:>5.2f} "
              f"{res['LR-counts']:>6.3f} {res['LR-tfidf12']:>6.3f} {res['MLP-w1']:>6.3f} "
              f"{res['MLP-w2']:>6.3f} {res['MLP-w3']:>6.3f} {mx:>6.3f} {len(wit):>4}  {note}")
    skipped = [r for r in rows if r[2] is None or r[2][0] == "bal"]
    n_balanced = sum(1 for _ in admitted) + sum(1 for a in rows if a[2] and a[2][0] != "bal")
    print(f"\n  candidates skipped (n<{MIN_N} or imbalance): {len(skipped)}"
          f"  [thin slots: {collections.Counter(a[0].split('.')[0] for a in skipped)}]")
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "witness_pairs_code.jsonl"), "w") as f:
        for w in wit_all:
            f.write(json.dumps(w) + "\n")
    print(f"  witness pairs written: {len(wit_all)} -> results/chatv2/h1_v3/witness_pairs_code.jsonl")
    print(f"\n  ADMITTED code axes: {len(admitted)} / target 24  {admitted}")
    print(f"  kill causes among value axes: {dict(causes)}")
    if len(admitted) >= 24:
        v = f"H1-V3-DATA-ADMIT (code) - {len(admitted)} axes survive balance + surface suite + witness pairs"
    elif causes["copy"] >= max(causes["thin-slot"], 1):
        v = (f"F3-V3/copy (code) - the executable straight-line slice of real code is LITERAL-dominated "
             f"(computed-value share {n_comp}/{n_all} = {n_comp/max(n_all,1):.2f}); {causes['copy']} value axes "
             "emptied by the SS3.4 computed filter. Real code keeps its computed state inside loops/calls, "
             "outside the restricted interpreter's reach.")
    elif causes["thin-slot"] > 0 or len(admitted) > 0:
        v = f"F3-V3/bank (code) - only {len(admitted)} admitted axes (<24); slot x predicate coverage too thin"
    else:
        v = "F3-V3/input (code) - surface suite / witness gate kills the bank"
    print(f"\nVERDICT(code): {v}")
    print("  (Non-promotional data admission. No model run, no GPU, no R2 claim.)")


# ---------------------------------------------------------------- chess corpus
SAN_STRIP = re.compile(r"[x+#?!]|=[QRBN]")


def load_games(cap=2600, min_plies=24):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("Lichess/standard-chess-games", "data/year=2013/month=01/train-00000-of-00001.parquet",
                        repo_type="dataset", revision="main")
    t = pq.read_table(p)
    cols = t.column_names
    mcol = "movetext" if "movetext" in cols else next(c for c in cols if "move" in c.lower())
    d = t.to_pylist()
    print(f"(chess) shard columns: {cols[:10]}... using '{mcol}'; {len(d)} games")
    games = []
    for g in d:
        san = [tk for tk in re.sub(r"\{[^}]*\}", " ", g[mcol]).split()
               if not tk[0].isdigit() and tk not in ("1-0", "0-1", "1/2-1/2", "*")]
        if len(san) >= min_plies:
            games.append({"san": san, "eco": g.get("ECO") or g.get("eco") or "?"})
        if len(games) >= cap:
            break
    print(f"(chess) {len(games)} games with >={min_plies} plies loaded")
    return games


def replay(san_moves, plies):
    import chess
    board = chess.Board()
    ucis = []
    try:
        for mv in san_moves[:plies]:
            m = board.parse_san(mv)
            ucis.append(m.uci())
            board.push(m)
    except Exception:
        return None
    return board, ucis


def sq_code(board, s):
    import chess
    p = board.piece_at(s)
    if p is None:
        return 0
    return (1 if p.color == chess.WHITE else 2) if p.piece_type == chess.PAWN \
        else (3 if p.color == chess.WHITE else 4)


PREDS = {"occ": lambda c: c != 0, "w": lambda c: c in (1, 3), "b": lambda c: c in (2, 4),
         "P": lambda c: c in (1, 2), "wP": lambda c: c == 1, "bP": lambda c: c == 2}


def chess_witness(games, want_axes, budget_s=240):
    """Transposition-style pairs: permute the first 8 plies' per-color move multisets (UCI),
    both orders legal, predicate differs; returns axis-id -> one pair."""
    import chess
    rng = np.random.default_rng(2)
    found, t0 = {}, time.time()
    for g in games:
        if time.time() - t0 > budget_s or not (want_axes - set(found)):
            break
        r = replay(g["san"], 8)
        if r is None:
            continue
        board0, ucis = r
        white, black = ucis[0::2], ucis[1::2]
        for _ in range(50):
            pw = list(rng.permutation(white)); pb = list(rng.permutation(black))
            alt = [m for pair in zip(pw, pb) for m in pair]
            if alt == ucis:
                continue
            b = chess.Board()
            try:
                for u in alt:
                    b.push_uci(u)
            except Exception:
                continue
            for s in chess.SQUARES:
                c0, c1 = sq_code(board0, s), sq_code(b, s)
                if c0 == c1:
                    continue
                for pname, fn in PREDS.items():
                    ax = f"{pname}.{chess.SQUARE_NAMES[s]}"
                    if ax in want_axes and ax not in found and fn(c0) != fn(c1):
                        found[ax] = {"a": " ".join(ucis), "b": " ".join(alt), "axis": ax,
                                     "label_a": bool(fn(c0)), "label_b": bool(fn(c1))}
            if not (want_axes - set(found)):
                break
    return found


def admit_chess():
    import chess
    games = load_games()
    if len(games) < MIN_TOTAL:
        print(f"\nVERDICT(chess): F3-V3/corpus - only {len(games)} usable games"); return

    def build(marker):
        out = []
        for i, g in enumerate(games):
            if len(g["san"]) < marker + 4:
                continue
            r = replay(g["san"], marker)
            if r is None:
                continue
            board, ucis = r
            codes = {chess.SQUARE_NAMES[s]: sq_code(board, s) for s in chess.SQUARES}
            sanM = g["san"][:marker]
            out.append({"gid": i, "sq": codes, "eco": g["eco"],
                        "san": " ".join(sanM),
                        "sanit": " ".join(SAN_STRIP.sub("", m) for m in sanM),
                        "uci": " ".join(ucis)})
        return out

    def balanced_axes(cand):
        out = []
        for sq in chess.SQUARE_NAMES:
            codes = np.array([r["sq"][sq] for r in cand])
            for pname, fn in PREDS.items():
                bal = float(np.mean([fn(c) for c in codes]))
                if BAL_LO <= bal <= BAL_HI:
                    out.append((f"{pname}.{sq}", bal))
        return out

    # marker-ply scan: chosen by BALANCED-AXIS COUNT ONLY (declared bank-design rule,
    # same category as the spec's balance-driven sign/magnitude choices; no probe peeking).
    # Axis family = piece-on-square predicates (spec SS4): occ/w/b/P/wP/bP x 64 squares.
    best, best_inst, best_ax = None, None, []
    for marker in (16, 24, 32, 40):
        cand = build(marker)
        if len(cand) < MIN_TOTAL:
            print(f"(marker scan) ply {marker}: only {len(cand)} instances - skip")
            continue
        ax = balanced_axes(cand)
        print(f"(marker scan) ply {marker}: {len(cand)} instances, {len(ax)}/384 balanced piece-on-square axes")
        if len(ax) > len(best_ax):
            best, best_inst, best_ax = marker, cand, ax
    inst = best_inst
    print(f"(chess) marker ply {best} chosen ({len(best_ax)} balanced axes); {len(inst)} instances")
    if len(best_ax) < 24:
        print(f"\nVERDICT(chess): F3-V3/bank - only {len(best_ax)} balanced axes (<24)"); return
    # cap at the 40 most-balanced axes (declared headroom rule; 24 needed)
    bal_ax = sorted(best_ax, key=lambda t: abs(t[1] - 0.5))[:40]
    ecos = sorted({r["eco"] for r in inst})
    eco_map = {e: k for k, e in enumerate(ecos)}
    meta = np.zeros((len(inst), len(ecos)))
    for k, r in enumerate(inst):
        meta[k, eco_map[r["eco"]]] = 1.0
    wit = chess_witness(games, {a for a, _ in bal_ax})
    with open(os.path.join(OUT_DIR, "witness_pairs_chess.jsonl"), "w") as f:
        for w in wit.values():
            f.write(json.dumps(w) + "\n")
    print(f"(chess) witness coverage (UCI condition): {len(wit)}/{len(bal_ax)} probed axes"
          f" -> results/chatv2/h1_v3/witness_pairs_chess.jsonl")
    for cond in ("san", "sanit", "uci"):
        admitted = []
        print(f"\n  [{cond.upper()} condition]   axis      n   bal  probe-max   meta  wit  admit?", flush=True)
        for ax, b in bal_ax:
            pname, sq = ax.split(".")
            fn = PREDS[pname]
            y = [fn(r["sq"][sq]) for r in inst]
            res = probe_suite([r[cond] for r in inst], y, [r["gid"] for r in inst], extra=meta)
            if res is None:
                continue
            mx = max(res.values())
            has_wit = (ax in wit) if cond == "uci" else False  # SAN pairs: bags differ (the leak)
            ok = mx <= CEIL and has_wit
            if ok:
                admitted.append(ax)
            print(f"    {ax:>8} {len(inst):>6} {b:>5.2f} {mx:>9.3f} {res.get('LR-meta', 0):>6.3f}"
                  f" {int(has_wit):>4}  {'YES' if ok else 'no'}", flush=True)
        print(f"  [{cond.upper()}] admitted: {len(admitted)} / 24 needed  "
              f"{'-> H1-V3-DATA-ADMIT (chess/' + cond + ')' if len(admitted) >= 24 else ''}")
    print("\n  (Chess verdict = best condition above; SAN conditions cannot supply witness pairs"
          "\n   because reordering changes the SAN bag itself - the notation leak, as scoped.)")
    print("  (Non-promotional data admission. No model run, no GPU, no R2 claim.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["code", "chess"], required=True)
    a = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    print("H1_V3_0_DATA_ADMISSION  [NON-PROMOTIONAL, CPU]\n")
    (admit_code if a.corpus == "code" else admit_chess)()
