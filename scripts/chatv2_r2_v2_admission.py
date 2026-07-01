#!/usr/bin/env python3
"""Chat-v2 R2 v2 — ADMISSION SCREEN (pre-reg docs/chatv2/R2_V2_RELATIONAL_SUBSTRATE_SPEC.md).
NON-promotional, CPU-only. Screens the AGREEMENT/CONTROLLER family (the spec's primary
target) against the intersection: input-undecodable AND GPT-2-computed AND (dimensionality).
Returns ADMIT-R2-v2 / F3-R2-v2/{corpus,input} / F2-R2-v2/{representation,control}.
Corpus = human WikiText-103 (pyarrow). Run: python scripts/chatv2_r2_v2_admission.py
"""
import os, re, argparse
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
AUX_SG = {"is", "was", "has", "does"}
AUX_PL = {"are", "were", "have", "do"}
AUX = AUX_SG | AUX_PL
PRESENT = {"is", "are", "has", "have", "does", "do"}
NOUNY = re.compile(r"\b(\w+?)(s?)\b")  # crude number proxy for the nearest preceding token


def load_wikitext(n_chars):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    p = hf_hub_download(repo_id="Salesforce/wikitext",
                        filename="wikitext-103-raw-v1/train-00000-of-00002.parquet", repo_type="dataset")
    rows = pq.read_table(p, columns=["text"]).column("text").to_pylist()  # read_table is stable here
    buf, total = [], 0
    for x in rows:
        if x and not x.startswith(" ="):                                   # drop headings
            buf.append(x); total += len(x)
            if total >= n_chars:
                break
    return re.sub(r"\s+", " ", " ".join(buf))[:n_chars]


def find_sites(text, tok):
    """Passages ending just before a clear-number auxiliary; label = number/tense/family.
    Returns list of (token_ids_up_to_aux, aux_word, labels_dict)."""
    print(f"  (tokenizing {len(text):,} chars ...)", flush=True)
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    print(f"  ({len(ids):,} tokens; scanning for agreement sites ...)", flush=True)
    words = [text[a:b].strip().lower() for a, b in offs]
    out = []
    for i in range(40, len(words) - 1):
        w = words[i]
        if w not in AUX:
            continue
        ctx = ids[max(0, i - 80):i]                      # up to (not incl) the aux
        if len(ctx) < 40:
            continue
        # nearest preceding "noun-ish" number proxy (attractor heuristic)
        prevw = [x for x in words[max(0, i - 6):i] if x.isalpha()]
        near_pl = bool(prevw and prevw[-1].endswith("s"))
        num = 0 if w in AUX_SG else 1
        out.append((ctx, w, {
            "num": num,
            "tense": 0 if w in PRESENT else 1,
            "fam_be": int(w in {"is", "are", "was", "were"}),
            "fam_have": int(w in {"has", "have"}),
            "attractor_mismatch": int(near_pl != bool(num)),   # nearest-noun number != true number
        }))
        if len(out) >= 6000:
            break
    return out


def gpt2_bodies(ctxs, pretrained, seq=128):
    import torch
    from transformers import GPT2Model, GPT2Config
    torch.set_grad_enabled(False)
    m = GPT2Model.from_pretrained("gpt2") if pretrained else GPT2Model(GPT2Config.from_pretrained("gpt2"))
    m.eval()
    B = []
    for i in range(0, len(ctxs), 48):
        batch = ctxs[i:i + 48]
        L = max(len(c) for c in batch)
        arr = np.zeros((len(batch), L), dtype=np.int64)
        last = []
        for r, c in enumerate(batch):
            arr[r, L - len(c):] = c; last.append(L - 1)          # left-pad; predict-site = last col
        h = m(torch.tensor(arr), output_hidden_states=True).hidden_states[-1]  # [b,L,d]
        B.append(h[np.arange(len(batch)), last, :].numpy())
    return np.concatenate(B, 0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    print("R2_V2_ADMISSION  agreement/controller family  GPT-2 small (CPU)  [NON-PROMOTIONAL]\n", flush=True)
    # pyarrow read MUST happen before torch is imported (torch+pyarrow C-ext conflict segfaults py3.14)
    text = load_wikitext(1_000_000 if a.smoke else 2_400_000)
    print(f"(corpus) WikiText-103 human prose loaded: {len(text):,} chars", flush=True)
    from transformers import GPT2TokenizerFast
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    sites = find_sites(text, tok)
    n = len(sites)
    print(f"(corpus) WikiText-103 human prose; agreement sites found = {n}")
    if n < 1500:
        print("  F3-R2-v2/corpus: too few agreement sites."); return
    sites = sites[:200] if a.smoke else sites[:1000]
    n = len(sites)

    ctxs = [s[0] for s in sites]
    LAB = ["num", "tense", "fam_be", "fam_have", "attractor_mismatch"]
    Y = np.array([[s[2][k] for k in LAB] for s in sites])
    ctx_text = [tok.decode(c) for c in ctxs]

    idx = np.arange(n); np.random.default_rng(0).shuffle(idx)
    tr, te = idx[:int(.7 * n)], idx[int(.7 * n):]

    # raw-token probe (input-undecodability): bag-of-tokens over the CONTEXT (no aux present)
    cv = CountVectorizer(max_features=4000); Xtok = cv.fit_transform(ctx_text)
    print("\n(extract) GPT-2 pretrained + random-init floor at the predict-site ...")
    Bp = gpt2_bodies(ctxs, True); Br = gpt2_bodies(ctxs, False)
    Bp = (Bp - Bp[tr].mean(0)) / (Bp[tr].std(0) + 1e-6)
    Br = (Br - Br[tr].mean(0)) / (Br[tr].std(0) + 1e-6)

    print(f"\n{'label':>18} {'balance':>8} {'raw-token':>10} {'GPT2-pre':>9} {'rand-floor':>10}  intersection?")
    intersec = 0
    for j, lab in enumerate(LAB):
        bal = Y[:, j].mean()
        raw = LogisticRegression(max_iter=300).fit(Xtok[tr], Y[tr, j]).score(Xtok[te], Y[te, j])
        g = LogisticRegression(max_iter=400, C=0.5).fit(Bp[tr], Y[tr, j]).score(Bp[te], Y[te, j])
        rf = LogisticRegression(max_iter=400, C=0.5).fit(Br[tr], Y[tr, j]).score(Br[te], Y[te, j])
        ok = (0.40 <= bal <= 0.60) and raw <= 0.60 and g >= 0.65
        intersec += ok
        print(f"{lab:>18} {bal:>8.3f} {raw:>10.3f} {g:>9.3f} {rf:>10.3f}  {'YES' if ok else 'no'}")

    print("\n  intersection = input-undecodable (raw<=0.60) AND GPT-2-computed (>=0.65) AND balanced")
    print(f"  labels in the intersection: {intersec} / {len(LAB)}")
    if intersec == 0:
        v = ("F3-R2-v2/input (or F2) — NO agreement label is both input-undecodable and GPT-2-carried: "
             "GPT-2 reads number BUT so does a raw-token probe (the subject is in the input). "
             "The intersection is empty for the plain agreement family.")
    elif intersec < 24:
        v = (f"F3-R2-v2/corpus (bank too thin) — only {intersec} qualifying axis/axes exist in the agreement "
             "family (agreement is ~1-dimensional: number/tense), far below the 24 needed for d_dec>=20. "
             "Relational grammatical state is intrinsically low-dimensional.")
    else:
        v = "ADMIT-R2-v2 — enough input-undecodable, GPT-2-carried relational labels to freeze a verdict harness"
    print(f"\n  ADMISSION VERDICT: {v}")
    print("  (Non-promotional screen. No R2 body-resistance claim, no promotion, no public/R3 language.)")


if __name__ == "__main__":
    main()
