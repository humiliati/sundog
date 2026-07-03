"""Percival Track-C B1/B2 bridge: per-item score caching (GPU pass).

Pre-registration: docs/percival/PERCIVAL_TRACKC_B1B2_PREREG.md (frozen 2026-07-03).

Scores one Pythia checkpoint on the frozen HellaSwag/LAMBADA subsets (deterministic:
fp32, fixed batch, frozen item order, loglik/greedy) and caches per-item results so all
statistics are instant re-analysis. Optional --t-arm adds the temperature-0.7 sampled
pass (B1-c). Determinism controls per the S0 floor literature (fp32 + fixed batch).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Silent HF token load per Dev/AGENTS.md keyring discipline (never print the value).
if not os.environ.get("HF_TOKEN"):
    _keyfile = Path.home() / "Dev" / "syek.ecafgnigguh.txt"
    if _keyfile.exists():
        os.environ["HF_TOKEN"] = _keyfile.read_text(encoding="utf-8").strip().splitlines()[0].strip()

import datasets as _datasets  # noqa: F401 -- MUST import before torch: pyarrow/torch DLL clash
# on this box segfaults when torch loads first (isolated 2026-07-03: torch->datasets = segfault,
# datasets->torch = clean).
import numpy as np
import torch

SEED = 20260703
REPO_ROOT = Path(__file__).resolve().parents[1]


def frozen_indices(n_total: int, n_take: int) -> np.ndarray:
    return np.random.RandomState(SEED).permutation(n_total)[:n_take]


@torch.no_grad()
def continuation_logliks(model, device, ctx_ids_list, cont_ids_list, batch_size):
    """Sum log-prob of each continuation given its context. Frozen order, fixed batches."""
    out = []
    for start in range(0, len(ctx_ids_list), batch_size):
        chunk_ctx = ctx_ids_list[start:start + batch_size]
        chunk_cont = cont_ids_list[start:start + batch_size]
        seqs = [c + t for c, t in zip(chunk_ctx, chunk_cont)]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for i, (c, t) in enumerate(zip(chunk_ctx, chunk_cont)):
            # tokens t occupy positions len(c)..len(c)+len(t)-1; predicted from position-1
            pos = torch.arange(len(c) - 1, len(c) + len(t) - 1, device=device)
            tok = torch.tensor(t, dtype=torch.long, device=device)
            out.append(logprobs[i, pos, tok].sum().item())
    return out


@torch.no_grad()
def greedy_teacher_forced_match(model, device, prefix_ids_list, target_ids_list, batch_size):
    """LAMBADA: all target positions must be the argmax given the teacher-forced prefix."""
    out = []
    for start in range(0, len(prefix_ids_list), batch_size):
        chunk_p = prefix_ids_list[start:start + batch_size]
        chunk_t = target_ids_list[start:start + batch_size]
        seqs = [p + t for p, t in zip(chunk_p, chunk_t)]
        maxlen = max(len(s) for s in seqs)
        input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long, device=device)
        attn = torch.zeros((len(seqs), maxlen), dtype=torch.long, device=device)
        for i, s in enumerate(seqs):
            input_ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        preds = model(input_ids=input_ids, attention_mask=attn).logits.argmax(dim=-1)
        for i, (p, t) in enumerate(zip(chunk_p, chunk_t)):
            pos = torch.arange(len(p) - 1, len(p) + len(t) - 1, device=device)
            tok = torch.tensor(t, dtype=torch.long, device=device)
            out.append(bool((preds[i, pos] == tok).all().item()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="EleutherAI/pythia-160m")
    ap.add_argument("--revision", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch-size", type=int, default=16)   # FROZEN determinism control
    ap.add_argument("--n-hellaswag", type=int, default=2000)
    ap.add_argument("--n-lambada", type=int, default=1000)
    ap.add_argument("--t-arm", action="store_true")
    ap.add_argument("--t-arm-items", type=int, default=300)
    ap.add_argument("--t-arm-k", type=int, default=4)
    ap.add_argument("--t-arm-temp", type=float, default=0.7)
    ap.add_argument("--smoke", type=int, default=0, help="limit items (smoke test only; not for the frozen run)")
    ap.add_argument("--out-root", default="results/percival/b1b2/scores")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = REPO_ROOT / args.out_root / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cache] {args.model}@{args.revision} -> {args.tag} (device={device}, fp32, batch={args.batch_size})")
    tok = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision,
                                                 torch_dtype=torch.float32).to(device).eval()

    nh = args.smoke or args.n_hellaswag
    nl = args.smoke or args.n_lambada

    # ---- HellaSwag (loglik ranking; primary=raw acc, secondary=byte-normalized) ----
    hs_path = out_dir / "hellaswag.json"
    if hs_path.exists():
        print(f"[cache] hellaswag exists, skip")
    else:
        ds = load_dataset("Rowan/hellaswag", split="validation")
        idx = frozen_indices(len(ds), nh)
        ctxs, conts, bytelens, golds = [], [], [], []
        for j in idx:
            item = ds[int(j)]
            c = tok(item["ctx"], add_special_tokens=False)["input_ids"]
            for e in item["endings"]:
                t = tok(" " + e, add_special_tokens=False)["input_ids"]
                ctxs.append(c); conts.append(t); bytelens.append(max(1, len(e.encode("utf-8"))))
            golds.append(int(item["label"]))
        lls = continuation_logliks(model, device, ctxs, conts, args.batch_size)
        rows = []
        for k, g in enumerate(golds):
            four = lls[4 * k: 4 * k + 4]
            bl = bytelens[4 * k: 4 * k + 4]
            pred = int(np.argmax(four))
            pred_norm = int(np.argmax([x / b for x, b in zip(four, bl)]))
            rows.append({"gold": g, "pred": pred, "pred_norm": pred_norm,
                         "ll": [round(x, 4) for x in four]})
        hs_path.write_text(json.dumps({"task": "hellaswag", "seed": SEED, "n": len(rows),
                                       "rows": rows}) + "\n", encoding="utf-8")
        acc = float(np.mean([r["pred"] == r["gold"] for r in rows]))
        print(f"[cache] hellaswag n={len(rows)} acc={acc:.4f}")

    # ---- LAMBADA (greedy teacher-forced exact match) ----
    lb_path = out_dir / "lambada.json"
    lb_prefixes, lb_targets = None, None
    if lb_path.exists() and not args.t_arm:
        print(f"[cache] lambada exists, skip")
    else:
        ds = load_dataset("EleutherAI/lambada_openai", "en", split="test")
        idx = frozen_indices(len(ds), nl)
        lb_prefixes, lb_targets, texts = [], [], []
        for j in idx:
            text = ds[int(j)]["text"]
            prefix, last = text.rsplit(" ", 1)
            lb_prefixes.append(tok(prefix, add_special_tokens=False)["input_ids"])
            lb_targets.append(tok(" " + last, add_special_tokens=False)["input_ids"])
            texts.append(last)
        if not lb_path.exists():
            match = greedy_teacher_forced_match(model, device, lb_prefixes, lb_targets, args.batch_size)
            rows = [{"correct": int(m)} for m in match]
            lb_path.write_text(json.dumps({"task": "lambada", "seed": SEED, "n": len(rows),
                                           "rows": rows}) + "\n", encoding="utf-8")
            print(f"[cache] lambada n={len(rows)} acc={float(np.mean(match)):.4f}")

    # ---- T-arm (B1-c): sampled completions at T=0.7, k samples, per-item mean match ----
    if args.t_arm:
        ta_path = out_dir / "tarm.json"
        if ta_path.exists():
            print(f"[cache] tarm exists, skip")
        else:
            nt = args.smoke or args.t_arm_items
            rows = []
            for i in range(nt):
                p, t = lb_prefixes[i], lb_targets[i]
                input_ids = torch.tensor([p], dtype=torch.long, device=device)
                matches = []
                for s in range(args.t_arm_k):
                    torch.manual_seed(SEED * 1000 + i * 10 + s)   # frozen per (item, sample)
                    gen = model.generate(input_ids, do_sample=True, temperature=args.t_arm_temp,
                                         top_k=0, max_new_tokens=len(t) + 2,
                                         pad_token_id=tok.eos_token_id)
                    new = gen[0, len(p):].tolist()
                    matches.append(int(new[: len(t)] == t))
                rows.append({"matches": matches, "mean": float(np.mean(matches))})
            ta_path.write_text(json.dumps({"task": "tarm", "seed": SEED, "n": len(rows),
                                           "temp": args.t_arm_temp, "k": args.t_arm_k,
                                           "rows": rows}) + "\n", encoding="utf-8")
            print(f"[cache] tarm n={len(rows)} mean_match={float(np.mean([r['mean'] for r in rows])):.4f}")

    print(f"[cache] done -> {out_dir}")


if __name__ == "__main__":
    main()
