#!/usr/bin/env python3
"""Chat-v2 R2 — real-substrate body-resistance on GPT-2 small.
Pre-reg: docs/chatv2/R2_REAL_SUBSTRATE_SPEC.md. INTERNAL HALF ONLY. A SHARP result is
"internal SHARP, pending external mech-interp review" — no public / R2-promotion / R3 /
world-model claim. CPU inference. Corpus = pinned public-domain Gutenberg texts.
Run: python scripts/chatv2_r2_real_substrate.py [--smoke]
"""
import os, re, argparse, urllib.request
import numpy as np

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE, "results", "chatv2", "r2_corpus")
GUTENBERG = {1342: "pride", 2701: "moby", 84: "frankenstein", 1661: "sherlock", 98: "twocities"}

# ---- frozen attribute bank: regex/lexicon count-parity matchers (pre-registered) ----
MORPH = {"suf_ed": r"\w{2,}ed\b", "suf_ing": r"\w{2,}ing\b", "suf_ly": r"\w{2,}ly\b",
 "suf_s": r"\w{2,}s\b", "suf_er": r"\w{2,}er\b", "suf_est": r"\w{2,}est\b",
 "suf_tion": r"\w{2,}tion\b", "suf_ment": r"\w{2,}ment\b", "suf_ness": r"\w{2,}ness\b",
 "suf_able": r"\w{2,}(?:able|ible)\b", "suf_ful": r"\w{2,}ful\b", "suf_less": r"\w{2,}less\b",
 "suf_ize": r"\w{2,}(?:ize|ise)\b", "suf_ity": r"\w{2,}ity\b", "suf_al": r"\w{3,}al\b",
 "suf_ic": r"\w{3,}ic\b", "suf_ous": r"\w{2,}ous\b", "suf_ive": r"\w{2,}ive\b",
 "pre_un": r"\bun\w{2,}", "pre_re": r"\bre\w{2,}"}
LEX = {"negators": r"\b(?:not|no|never|none|nobody|nothing|neither|nor)\b",
 "modals": r"\b(?:can|could|may|might|must|shall|should|will|would)\b",
 "aux": r"\b(?:be|is|are|was|were|been|being|am|have|has|had|do|does|did)\b",
 "wh": r"\b(?:who|what|when|where|why|which|how|whom|whose)\b",
 "pers_pron": r"\b(?:i|you|he|she|it|we|they|me|him|her|us|them)\b",
 "poss_pron": r"\b(?:my|your|his|her|its|our|their|mine|yours|hers|ours|theirs)\b",
 "demonstr": r"\b(?:this|that|these|those)\b",
 "connectives": r"\b(?:and|but|or|so|because|although|however|therefore|while|if|unless|though|yet)\b",
 "prepositions": r"\b(?:in|on|at|by|for|with|to|from|of|about|over|under|between|through)\b",
 "determiners": r"\b(?:the|a|an)\b",
 "quantifiers": r"\b(?:all|some|many|few|most|several|each|every|any|both)\b",
 "intensifiers": r"\b(?:very|quite|rather|too|extremely|really|fairly|somewhat)\b",
 "number_words": r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\b"}
FMT = {"titlecase": r"\b[A-Z][a-z]+\b", "allcaps": r"\b[A-Z]{2,}\b", "digits": r"\b\d+\b",
 "hyphenated": r"\b\w+-\w+\b", "apostrophe_s": r"\w+'s\b", "parenthetical": r"\([^)]*\)",
 "longword": r"\b\w{10,}\b", "shortword": r"\b\w{1,2}\b"}
PUNCT = {"comma": r",", "semicolon": r";", "colon": r":", "dquote": r"[\"“”]",
 "question": r"\?", "exclaim": r"!", "period": r"\.", "dash": r"[-–—]",
 "ellipsis": r"\.\.\.", "sentence": r"[.!?]+"}
BANK = {**MORPH, **LEX, **FMT, **PUNCT}
CASE_SENS = {"titlecase", "allcaps"}
RX = {k: re.compile(v, 0 if k in CASE_SENS else re.IGNORECASE) for k, v in BANK.items()}
ATTRS = list(BANK.keys())


def parities(text):
    return np.array([len(RX[a].findall(text)) & 1 for a in ATTRS], dtype=np.int8)


def load_corpus(smoke=False):
    # Real English prose, HF-hosted plain text (Gutenberg/GitHub unreachable; only HF is).
    # TinyStories = coherent real English (GPT-4-written, NOT GPT-2 — so no circularity);
    # simpler distribution than human text, noted as an MVP scope caveat.
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(repo_id="roneneldan/TinyStories",
                            filename="TinyStoriesV2-GPT4-valid.txt", repo_type="dataset")
    except Exception as e:
        print(f"  [corpus] HF download FAILED: {type(e).__name__}"); return ""
    cap = 2_000_000 if smoke else 14_000_000
    t = open(p, encoding="utf-8", errors="ignore").read(cap).replace("<|endoftext|>", " ")
    return re.sub(r"\s+", " ", t)


def build(n_pass, seq_len, seed=0):
    from transformers import GPT2TokenizerFast
    raw = load_corpus(smoke=(n_pass <= 200))
    if len(raw) < 5000:
        return None
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = tok(raw)["input_ids"]
    nchunks = len(ids) // seq_len
    rng = np.random.default_rng(seed)
    pick = rng.permutation(nchunks)[:n_pass]
    passages = [ids[i * seq_len:(i + 1) * seq_len] for i in pick]
    texts = [tok.decode(p) for p in passages]
    Y = np.stack([parities(t) for t in texts])          # [N, A]
    return np.array(passages), texts, Y, tok


def extract(passages, pretrained):
    import torch
    from transformers import GPT2Model, GPT2Config
    torch.set_grad_enabled(False)
    model = (GPT2Model.from_pretrained("gpt2") if pretrained
             else GPT2Model(GPT2Config.from_pretrained("gpt2")))
    model.eval()
    X = torch.tensor(passages, dtype=torch.long)
    outs = []
    for i in range(0, len(X), 48):
        h = model(X[i:i + 32], output_hidden_states=True).hidden_states  # tuple L+1 [b,T,d]
        outs.append(torch.stack([hl[:, -1, :] for hl in h], 1).numpy())   # [b, L+1, d]
    return np.concatenate(outs, 0)                                        # [N, L+1, d]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    N, SEQ = (120, 64) if a.smoke else (1500, 128)
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.neural_network import MLPClassifier
    print(f"R2_REAL_SUBSTRATE  GPT-2 small (CPU)  N={N} seq={SEQ}  bank={len(ATTRS)} attrs  [INTERNAL HALF]\n")

    data = build(N, SEQ)
    if data is None:
        print("  F-INFRA: corpus unavailable (no network to Gutenberg / no --corpus). No verdict."); return
    passages, texts, Y, tok = data
    n = len(passages)
    idx = np.arange(n); np.random.default_rng(1).shuffle(idx)
    tr, va, te = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]

    # ---- gate 1 (balance) + gate 2 (input-undecodability via raw-token linear probe) ----
    bal = Y.mean(0)
    cv = CountVectorizer(max_features=3000); Xtok = cv.fit_transform(texts)
    survivors = []
    for j, at in enumerate(ATTRS):
        if not (0.40 <= bal[j] <= 0.60):
            continue
        lp = LogisticRegression(max_iter=200, C=1.0).fit(Xtok[tr], Y[tr, j])
        tokacc = lp.score(Xtok[va], Y[va, j])
        if tokacc <= 0.60:
            survivors.append((j, at, bal[j], tokacc))
    print(f"(gates) {len(survivors)}/{len(ATTRS)} attrs survive balance[0.40,0.60] + raw-token-probe<=0.60")
    if len(survivors) < 24:
        print(f"  F3-R2: fewer than 24 survivors ({len(survivors)}) — bank too thin for d_dec>=20. No verdict."); return
    S = [s[0] for s in survivors]

    # ---- extract bodies (pretrained + random-init floor) ----
    print("(extract) GPT-2 pretrained ..."); Bp = extract(passages, True)
    print("(extract) GPT-2 random-init floor ..."); Br = extract(passages, False)

    def probe_layer(B, cols, fit=tr, ev=te):
        # per-attr held-out z_recover at this layer; returns (accs, weight-matrix)
        accs, W = [], []
        for j in cols:
            m = LogisticRegression(max_iter=300, C=0.5).fit(B[fit], Y[fit, j])
            accs.append(m.score(B[ev], Y[ev, j])); W.append(m.coef_[0])
        return np.array(accs), np.array(W)

    # pick body layer = argmax d_dec over layers (Phase-0 D3), via FAST class-mean-diff dirs
    def meandiff_ddec(B, cols):
        W = []
        for j in cols:
            y = Y[tr, j].astype(bool)
            W.append(B[tr][y].mean(0) - B[tr][~y].mean(0))
        Wn = np.array(W); Wn = Wn / (np.linalg.norm(Wn, axis=1, keepdims=True) + 1e-9)
        s = np.linalg.svd(Wn, compute_uv=False); return (s.sum()**2) / (s @ s)
    best = None
    for L in range(Bp.shape[1]):
        Bn = (Bp[:, L] - Bp[tr, L].mean(0)) / (Bp[tr, L].std(0) + 1e-6)
        dd = meandiff_ddec(Bn, S)
        if best is None or dd > best[1]:
            best = (L, dd)
    L = best[0]
    Bp_n = (Bp[:, L] - Bp[tr, L].mean(0)) / (Bp[tr, L].std(0) + 1e-6)
    Br_n = (Br[:, L] - Br[tr, L].mean(0)) / (Br[tr, L].std(0) + 1e-6)

    # z1 per survivor on val -> pick decision -> test once
    valacc, _ = probe_layer(Bp_n, S, tr, va)
    dec_i = S[int(np.argmax(valacc))]
    others = [j for j in S if j != dec_i]
    z1 = LogisticRegression(max_iter=400, C=0.5).fit(Bp_n[tr], Y[tr, dec_i]).score(Bp_n[te], Y[te, dec_i])

    # d_dec at body layer (test-independent readout rank over survivors)
    _, W = probe_layer(Bp_n, S, tr, va)
    Wn = W / (np.linalg.norm(W, axis=1, keepdims=True) + 1e-9)
    sv = np.linalg.svd(Wn, compute_uv=False); d_dec = (sv.sum()**2) / (sv @ sv)

    # shadow = 1-D decision read-direction; leak = predict OTHER survivors from that 1-D shadow
    w_dec = LogisticRegression(max_iter=400, C=0.5).fit(Bp_n[tr], Y[tr, dec_i]).coef_[0]
    proj = (Bp_n @ w_dec)[:, None]
    leaks = [LogisticRegression(max_iter=200).fit(proj[tr], Y[tr, j]).score(proj[te], Y[te, j]) for j in others]
    leak = float(np.mean(leaks))
    live = LogisticRegression(max_iter=200).fit(proj[tr], Y[tr, dec_i]).score(proj[te], Y[te, dec_i])  # H4
    # H5 compute-can't-cross: MLP recon of a dropped latent from the 1-D shadow
    h5 = MLPClassifier((32,), max_iter=300, random_state=0).fit(proj[tr], Y[tr, others[0]]).score(proj[te], Y[te, others[0]])

    # body_carry = mean test z_recover of non-decision survivors: pretrained vs random floor
    bc_p, _ = probe_layer(Bp_n, others, tr, te)
    bc_r, _ = probe_layer(Br_n, others, tr, te)
    carry_p, carry_r = float(bc_p.mean()), float(bc_r.mean())

    print(f"\n(body layer L={L})")
    print(f"  d_dec (readout rank over {len(S)} survivors) = {d_dec:.1f}      [bar >= 20]")
    print(f"  z1_acc (decision '{ATTRS[dec_i]}', held-out test) = {z1:.3f}   [>= 0.70]")
    print(f"  cross_latent_leak (1-D shadow -> {len(others)} others) = {leak:.3f}   [~0.50 resists]  (H4 liveness: decision leaks {live:.2f})")
    print(f"  H5 compute-can't-cross (MLP recon of a dropped latent) = {h5:.3f}   [~0.50 = info-loss floor]")
    print(f"  body_carry: pretrained {carry_p:.3f}  vs  random-init floor {carry_r:.3f}   -> gap {carry_p-carry_r:+.3f}   [gap >= 0.15]")

    sharp = (d_dec >= 20 and z1 >= 0.70 and leak <= 0.58 and (carry_p - carry_r) >= 0.15 and live >= 0.80)
    if sharp:
        v = "internal SHARP (pending external mech-interp review) — d_dec>=20, control-sufficient, resists, objective-driven above the random floor"
    elif carry_p - carry_r < 0.15:
        v = "MARGINAL (F4-R2) — non-decision richness ~ the random-init floor: not objective-driven"
    elif leak > 0.58:
        v = "MARGINAL (F1-R2 / net.7 trap) — the shadow reconstructs the other latents: state-sufficient, not resisting"
    elif z1 < 0.70:
        v = "MARGINAL (F2-R2) — decision not compactly control-sufficient in GPT-2's body"
    elif d_dec < 20:
        v = "MARGINAL — real high-dim bar unmet (d_dec < 20)"
    else:
        v = "see numbers"
    print(f"\n  VERDICT: {v}")
    print("  (Internal half only. No R2 promotion / public claim / R3 / world-model gloss without external review.)")


if __name__ == "__main__":
    main()
