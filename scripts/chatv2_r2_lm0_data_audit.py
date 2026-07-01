#!/usr/bin/env python3
"""Chat-v2 R2 Larger-Model campaign — Phase LM-0 data audit (CPU, torch-free).
Pre-reg: docs/chatv2/R2_LARGER_MODEL_ROUTE_CAMPAIGN.md §4,§7. NON-promotional.
Question: how many FewRel relation axes survive STRONG surface de-confounds
(delexicalized bag, tf-idf n-gram, entity-type/metadata, masked-trigger)?
Go: >= 24 surface-surviving axes -> LM-0 PASS (GPU LM-1 worth it). Else F3-R2-LM/input.
Run: python scripts/chatv2_r2_lm0_data_audit.py
"""
import re, collections, argparse
import numpy as np

CEIL = 0.60  # input-undecodability ceiling (campaign §4 default)


def load_fewrel():
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("thunlp/few_rel", "default/train_wiki/0000.parquet",
                        repo_type="dataset", revision="refs/convert/parquet")
    return pq.read_table(p).to_pylist()


def delex(tokens, head, tail):
    hs = {i for span in head["indices"] for i in span}
    ts = {i for span in tail["indices"] for i in span}
    out = []
    for i, tk in enumerate(tokens):
        if i in hs:
            if not (out and out[-1] == "[SUBJ]"):
                out.append("[SUBJ]")
        elif i in ts:
            if not (out and out[-1] == "[OBJ]"):
                out.append("[OBJ]")
        else:
            out.append(tk.lower())
    return " ".join(out)


def meta_string(head, tail):
    hi = [i for s in head["indices"] for i in s]
    ti = [i for s in tail["indices"] for i in s]
    dist = min(abs(a - b) for a in hi for b in ti)
    order = "HTfirst" if min(hi) < min(ti) else "THfirst"
    return f"HT_{head['type']} TT_{tail['type']} DIST_{min(dist,20)} {order}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--per", type=int, default=700); a = ap.parse_args()
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
    from sklearn.model_selection import cross_val_score
    print("R2_LM0_DATA_AUDIT  FewRel train_wiki  [NON-PROMOTIONAL, CPU]\n")
    d = load_fewrel()
    by = collections.defaultdict(list)
    for r in d:
        by[r["relation"]].append(r)
    rels = sorted(by)
    print(f"(corpus) FewRel: {len(rels)} relations, {len(d)} instances; per-relation cap = {a.per}")

    # per-instance surface views (frozen)
    delx = {id(r): delex(r["tokens"], r["head"], r["tail"]) for r in d}
    lex = {id(r): " ".join(t.lower() for t in r["tokens"]) for r in d}
    meta = {id(r): meta_string(r["head"], r["tail"]) for r in d}
    rng = np.random.default_rng(0)

    def probe(pos, neg, texts_of, vec):
        rows = pos + neg
        y = np.array([1] * len(pos) + [0] * len(neg))
        X = vec.fit_transform([texts_of[id(r)] for r in rows])
        return float(cross_val_score(LogisticRegression(max_iter=300), X, y, cv=4, scoring="accuracy").mean())

    print(f"\n{'relation':>8} {'lex':>6} {'delex':>6} {'tfidf':>6} {'meta':>6} {'mask-trig':>9}  surface_max  survive?")
    survivors, rows_out = [], []
    others_pool = d
    for R in rels:
        pos = by[R][:a.per]
        neg = list(rng.choice(others_pool, size=len(pos) * 2, replace=False))
        neg = [r for r in neg if r["relation"] != R][:len(pos)]
        lex_a = probe(pos, neg, lex, CountVectorizer(max_features=5000))
        delex_a = probe(pos, neg, delx, CountVectorizer(max_features=5000))
        tfidf_a = probe(pos, neg, delx, TfidfVectorizer(ngram_range=(1, 2), max_features=8000))
        meta_a = probe(pos, neg, meta, CountVectorizer())
        # masked-trigger: drop the 15 highest-PMI delex tokens for R, re-probe delex bag
        cnt_pos = collections.Counter(w for r in pos for w in set(delx[id(r)].split()))
        cnt_all = collections.Counter(w for r in (pos + neg) for w in set(delx[id(r)].split()))
        pmi = {w: cnt_pos[w] / (cnt_all[w] + 1) for w in cnt_pos}
        trig = {w for w, _ in sorted(pmi.items(), key=lambda x: -x[1])[:15]}
        masked = {id(r): " ".join(w for w in delx[id(r)].split() if w not in trig) for r in (pos + neg)}
        mask_a = probe(pos, neg, masked, CountVectorizer(max_features=5000))
        surf_max = max(delex_a, tfidf_a, meta_a, mask_a)  # strong controls (lex reported for reference)
        surv = surf_max <= CEIL
        survivors.append(surv)
        rows_out.append((R, lex_a, delex_a, tfidf_a, meta_a, mask_a, surf_max, surv))
        print(f"{R:>8} {lex_a:>6.3f} {delex_a:>6.3f} {tfidf_a:>6.3f} {meta_a:>6.3f} {mask_a:>9.3f}  {surf_max:>10.3f}  {'YES' if surv else 'no'}")

    n_surv = sum(survivors)
    print(f"\n  surface-surviving relation axes (all strong controls <= {CEIL}): {n_surv} / {len(rels)}")
    if n_surv >= 24:
        v = f"LM-0 PASS — {n_surv} axes survive; a >=24-axis d_dec>=20 bank is possible -> LM-1 (1B GPU admission) is worth it"
    else:
        v = (f"F3-R2-LM/input — only {n_surv} relation axes survive strong surface controls (< 24). "
             "Relation labels leak through triggers / entity types even delexicalized: the larger-model "
             "route does NOT clear the DATA gate on FewRel. GPU (H200) buys nothing for R2 here.")
    print(f"\n  VERDICT: {v}")
    print("  (Non-promotional data audit. No R2 claim, no model run, no promotion, no GPU implied by this file.)")


if __name__ == "__main__":
    main()
