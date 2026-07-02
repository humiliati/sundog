#!/usr/bin/env python3
"""Chat-v2 H1/V3-1 -- 1B-class model admission under A1 (crossover + order-shuffle control).

Spec: docs/chatv2/H1_V3_0C_CROSSOVER_SPEC.md gates + H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md.
Input bank: results/chatv2/h1_v3/v3_0c_bank_manifest.json (frozen surface baselines).
Battery reused verbatim from the owner's V3-0.5 script (same splits, readout, margins).

Per axis: acc_model >= surface_max + 0.15 AND >= random-init floor + 0.15 AND
shuffle_drop >= 0.10. Bank gate: >= 20 crossing axes -> H1-V3-1-CROSS-ADMIT;
else F2-V3c/carry (dominant blocker = surface) / F4-V3c/floor (dominant = floor).
Non-promotional. GATE RUNG (unlike V3-0.5): a result here still licenses NO R2 claim
without V3-2 + external review.

Feature extraction is RESUME-SAFE (npz cache per model x variant, checkpoint every 20
batches) -- safe to Ctrl-C and re-run; it continues where it stopped.

Run (CPU-lite, owner terminal -- multi-hour):
  python scripts/chatv2_h1_v3_1_admission.py --model Qwen/Qwen2.5-1.5B --max-inst 900
GTX-1080:  add  --device cuda  (fp16)
Full bank: --max-inst 0
"""
from __future__ import annotations

import argparse, hashlib, json, os, re, sys, time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatv2_h1_v3_0c_bank_freeze import MARKER, build                      # noqa: E402
from chatv2_h1_v3_data_admission import OUT_DIR, PREDS, load_games         # noqa: E402
from chatv2_h1_v3_gpt2_calibration import (MAX_LEN, color_stratified_shuffle,  # noqa: E402
                                           fit_score_layer, load_manifest, split_indices)

THRESH = {"vs_surface": 0.15, "vs_randinit": 0.15, "shuffle_drop": 0.10}
CKPT_EVERY = 20


def pick_layers(n_layers: int) -> List[int]:
    """{~L/3, ~2L/3, L} on hidden_states indices 1..L (0 = embeddings)."""
    return sorted({max(1, round(n_layers / 3)), max(2, round(2 * n_layers / 3)), n_layers})


def model_tag(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name.split("/")[-1])


def extract_features(texts: List[str], model_name: str, pretrained: bool, layers: List[int],
                     batch_size: int, device: str, cache_path: str, sig: str) -> Dict[int, np.ndarray]:
    """Final-position hidden states at `layers`, resume-safe via npz checkpointing."""
    done, parts = 0, {l: [] for l in layers}
    if os.path.exists(cache_path):
        z = np.load(cache_path, allow_pickle=False)
        if str(z["sig"]) == sig:
            done = int(z["done"])
            for l in layers:
                parts[l] = [z[f"L{l}"]]
            if done >= len(texts):
                print(f"    (cache) {os.path.basename(cache_path)} complete ({done})", flush=True)
                return {l: parts[l][0] for l in layers}
            print(f"    (cache) resuming at {done}/{len(texts)}", flush=True)
        else:
            print("    (cache) signature mismatch - re-extracting", flush=True)

    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device == "cuda" else torch.float32
    if pretrained:
        model = AutoModel.from_pretrained(model_name, dtype=dtype).eval().to(device)
    else:
        torch.manual_seed(0)
        model = AutoModel.from_config(AutoConfig.from_pretrained(model_name)).to(dtype).eval().to(device)
    torch.set_grad_enabled(False)

    def save(n_done: int) -> None:
        arrs = {f"L{l}": np.concatenate(parts[l], axis=0) if parts[l] else np.zeros((0, 1)) for l in layers}
        np.savez(cache_path, done=n_done, sig=sig, **arrs)

    t0 = time.time()
    for bi, start in enumerate(range(done, len(texts), batch_size)):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"]
        last = mask.sum(dim=1) - 1
        rows = torch.arange(mask.shape[0], device=device)
        for l in layers:
            parts[l].append(out.hidden_states[l][rows, last].float().cpu().numpy())
        if (bi + 1) % CKPT_EVERY == 0:
            n_done = start + len(batch)
            save(n_done)
            rate = (n_done - done) / max(time.time() - t0, 1)
            eta = (len(texts) - n_done) / max(rate, 1e-9)
            print(f"    ... {n_done}/{len(texts)} ({rate:.1f} inst/s, ~{eta/60:.0f} min left)", flush=True)
    save(len(texts))
    return {l: np.concatenate(parts[l], axis=0) for l in layers}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--manifest", default=os.path.join(OUT_DIR, "v3_0c_bank_manifest.json"))
    ap.add_argument("--max-inst", type=int, default=900,
                    help="CPU-lite union-instance cap (deterministic seed-0 subsample); 0 = full")
    ap.add_argument("--max-axes", type=int, default=0, help="debug cap; 0 = all")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = ap.parse_args()

    tag = model_tag(args.model)
    print(f"H1_V3_1_ADMISSION  {args.model}  A1 battery  [NON-PROMOTIONAL, GATE RUNG]\n", flush=True)
    manifest = load_manifest(args.manifest)
    axes = manifest["axes"][: args.max_axes] if args.max_axes else manifest["axes"]
    print(f"(manifest) {len(axes)} axes; marker ply {manifest['marker_ply']}")

    # corpus/parquet BEFORE torch import (R2 reproduction gotchas)
    games = load_games(2600, 28)
    inst = build(games, MARKER)
    by_gid = {int(r["gid"]): r for r in inst}
    gids = sorted({int(g) for ax in axes for g in ax["gids"] if int(g) in by_gid})
    if args.max_inst and len(gids) > args.max_inst:
        rng = np.random.default_rng(0)
        gids = sorted(rng.choice(gids, args.max_inst, replace=False).tolist())
        print(f"(cpu-lite) union subsampled to {len(gids)} instances (seed 0)")
    rows = [by_gid[g] for g in gids]
    gid_to_row = {g: i for i, g in enumerate(gids)}
    orig_texts = [r["uci"] for r in rows]
    shuf_texts = [color_stratified_shuffle(r["uci"], int(r["gid"])) for r in rows]
    sig = hashlib.sha1((",".join(map(str, gids)) + f"|{MAX_LEN}|{args.model}").encode()).hexdigest()[:16]
    print(f"(data) {len(rows)} unique slice instances; cache sig {sig}")

    from transformers import AutoConfig
    layers = pick_layers(AutoConfig.from_pretrained(args.model).num_hidden_layers)
    print(f"(layers) hidden-state indices {layers} (validation-chosen per axis)")

    feats = {}
    for variant, texts, pre in (("orig", orig_texts, True), ("shuf", shuf_texts, True),
                                ("rand", orig_texts, False)):
        print(f"(extract) {tag} / {variant} ...", flush=True)
        cache = os.path.join(OUT_DIR, f"v3_1_feats_{tag}_{variant}.npz")
        feats[variant] = extract_features(texts, args.model, pre, layers,
                                          args.batch_size, args.device, cache, sig)

    results, blockers = [], {"surface": 0, "floor": 0, "shuffle": 0}
    for ax in axes:
        name = ax["axis"]
        pname, sq = name.split(".")
        fn = PREDS[pname]
        sample_gids = [int(g) for g in ax["gids"] if int(g) in gid_to_row]
        if len(sample_gids) < 40:
            results.append({"axis": name, "status": "skip_thin", "n_slice": len(sample_gids)})
            continue
        row_idx = np.array([gid_to_row[g] for g in sample_gids], dtype=int)
        groups = np.array(sample_gids)
        y = np.array([bool(fn(by_gid[g]["sq"][sq])) for g in sample_gids], dtype=int)
        train, val, test = split_indices(groups, y)
        sc = {v: fit_score_layer({l: X[row_idx] for l, X in feats[v].items()}, y, train, val, test)
              for v in ("orig", "shuf", "rand")}
        surface = float(ax["surface_max"])
        if sc["orig"]["test_acc"] is None:
            results.append({"axis": name, "status": "skip_split", "n_slice": len(sample_gids)})
            continue
        m = {"vs_surface": round(sc["orig"]["test_acc"] - surface, 4),
             "vs_randinit": None if sc["rand"]["test_acc"] is None
             else round(sc["orig"]["test_acc"] - sc["rand"]["test_acc"], 4),
             "shuffle_drop": None if sc["shuf"]["test_acc"] is None
             else round(sc["orig"]["test_acc"] - sc["shuf"]["test_acc"], 4)}
        ok = all(m[k] is not None and m[k] >= THRESH[k] for k in THRESH)
        if not ok:
            if m["vs_surface"] is None or m["vs_surface"] < THRESH["vs_surface"]:
                blockers["surface"] += 1
            elif m["vs_randinit"] is None or m["vs_randinit"] < THRESH["vs_randinit"]:
                blockers["floor"] += 1
            else:
                blockers["shuffle"] += 1
        results.append({"axis": name, "n_slice": len(sample_gids), "balance": ax["balance"],
                        "surface_max": surface, "orig": sc["orig"], "randinit": sc["rand"],
                        "shuffle": sc["shuf"], "margins": m,
                        "status": "cross" if ok else "no_cross"})
        print(f"  {name:>8} n={len(sample_gids):>4} surf={surface:.3f} orig={sc['orig']['test_acc']} "
              f"rand={sc['rand']['test_acc']} shuf={sc['shuf']['test_acc']} m={m} "
              f"{'CROSS' if ok else 'no_cross'}", flush=True)

    n_cross = sum(1 for r in results if r.get("status") == "cross")
    n_measured = len([r for r in results if "margins" in r])
    print(f"\n(result) {n_cross}/{n_measured} axes cross; blockers {blockers}")
    if n_measured < 20:
        verdict = (f"INCONCLUSIVE - only {n_measured} axes measured (<20; thin subsample or "
                   "smoke caps); no gate branch issued")
    elif n_cross >= 20:
        verdict = (f"H1-V3-1-CROSS-ADMIT - {n_cross} axes cross all margins on {args.model}; "
                   "V3-2 prereg may be drafted (still NO R2 claim without external review)")
    elif blockers["floor"] > blockers["surface"]:
        verdict = f"F4-V3c/floor - random-init floor explains the carry ({blockers['floor']} axes floor-blocked)"
    else:
        verdict = (f"F2-V3c/carry - {args.model} does not cross the frozen surface baseline "
                   f"({blockers['surface']} axes surface-blocked)")
    out = {"spec": "H1_V3_0C_CROSSOVER_SPEC.md + A1", "date": "2026-07-01",
           "rung": "V3-1 model admission", "gate_rung": True, "non_promotional": True,
           "model": args.model, "device": args.device, "max_inst": args.max_inst,
           "manifest": args.manifest, "layers": layers, "thresholds": THRESH,
           "n_axes": len(results), "n_crossing_axes": n_cross, "blockers": blockers,
           "verdict": verdict, "results": results}
    suffix = "_smoke" if (args.max_axes or (args.max_inst and args.max_inst < 300)) else ""
    out_path = os.path.join(OUT_DIR, f"v3_1_admission_{tag}{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"(wrote) {out_path}")
    print(f"VERDICT: {verdict}")
    print("  (Non-promotional. No R2 claim; V3-2 + external review remain gated.)")


if __name__ == "__main__":
    main()
