#!/usr/bin/env python3
"""Chat-v2 H1/V3-0.5 -- GPT-2 calibration under A1 order-shuffle control.

Spec: docs/chatv2/H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md.
Input bank: results/chatv2/h1_v3/v3_0c_bank_manifest.json.

Non-promotional CPU rung. Reports, per axis, whether GPT-2 small's residual reads
the chess state above (a) the frozen surface baseline, (b) a random-init floor,
and (c) same-bag order-shuffled features.

Run:
  python scripts/chatv2_h1_v3_gpt2_calibration.py [--max-axes 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatv2_h1_v3_0c_bank_freeze import MARKER, build  # noqa: E402
from chatv2_h1_v3_data_admission import OUT_DIR, PREDS, load_games  # noqa: E402


LAYERS_GPT2 = [4, 8, 12]
MAX_LEN = 192
SEED = 0


def load_manifest(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def color_stratified_shuffle(uci_text: str, gid: int, seed: int = SEED) -> str:
    moves = uci_text.split()
    white = moves[0::2]
    black = moves[1::2]
    rng = np.random.default_rng(seed * 1_000_003 + int(gid))
    white = list(rng.permutation(white))
    black = list(rng.permutation(black))
    out: List[str] = []
    for i in range(max(len(white), len(black))):
        if i < len(white):
            out.append(str(white[i]))
        if i < len(black):
            out.append(str(black[i]))
    return " ".join(out)


def split_indices(groups: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit

    idx = np.arange(len(y))
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=0)
    trainval, test = next(gss.split(idx, y, groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1)
    tr_rel, val_rel = next(gss2.split(trainval, y[trainval], groups[trainval]))
    train = trainval[tr_rel]
    val = trainval[val_rel]
    return train, val, test


def zfit(X: np.ndarray, rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = X[rows].mean(axis=0)
    sd = X[rows].std(axis=0) + 1e-6
    return mu, sd


def has_two_classes(y: np.ndarray, rows: Iterable[int]) -> bool:
    return len(set(y[list(rows)].tolist())) >= 2


def fit_score_layer(
    feats: Dict[int, np.ndarray],
    y: np.ndarray,
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
) -> dict:
    from sklearn.linear_model import LogisticRegression

    if not (has_two_classes(y, train) and has_two_classes(y, val) and has_two_classes(y, test)):
        return {"layer": None, "val_acc": None, "test_acc": None}

    val_scores = {}
    for layer, X in feats.items():
        mu, sd = zfit(X, train)
        Xtr = (X[train] - mu) / sd
        Xval = (X[val] - mu) / sd
        clf = LogisticRegression(max_iter=700, class_weight="balanced")
        clf.fit(Xtr, y[train])
        val_scores[layer] = float(clf.score(Xval, y[val]))

    layer = max(val_scores, key=val_scores.get)
    trainval = np.concatenate([train, val])
    X = feats[layer]
    mu, sd = zfit(X, trainval)
    clf = LogisticRegression(max_iter=700, class_weight="balanced")
    clf.fit((X[trainval] - mu) / sd, y[trainval])
    test_acc = float(clf.score((X[test] - mu) / sd, y[test]))
    return {"layer": int(layer), "val_acc": round(val_scores[layer], 4), "test_acc": round(test_acc, 4)}


def extract_features(texts: List[str], pretrained: bool, batch_size: int) -> Dict[int, np.ndarray]:
    import torch
    from transformers import GPT2Config, GPT2Model, GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    if pretrained:
        model = GPT2Model.from_pretrained("gpt2").eval()
    else:
        torch.manual_seed(0)
        model = GPT2Model(GPT2Config.from_pretrained("gpt2")).eval()
    torch.set_grad_enabled(False)

    chunks = {layer: [] for layer in LAYERS_GPT2}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN)
        out = model(**enc, output_hidden_states=True)
        mask = enc["attention_mask"]
        last = mask.sum(dim=1) - 1
        rows = torch.arange(mask.shape[0])
        for layer in LAYERS_GPT2:
            chunks[layer].append(out.hidden_states[layer][rows, last].detach().cpu().numpy())
    return {layer: np.concatenate(parts, axis=0) for layer, parts in chunks.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=os.path.join(OUT_DIR, "v3_0c_bank_manifest.json"))
    ap.add_argument("--max-axes", type=int, default=0, help="debug cap; 0 means all axes")
    ap.add_argument("--batch-size", type=int, default=24)
    args = ap.parse_args()

    print("H1_V3_0_5_GPT2_CALIBRATION  A1 order-shuffle control  [NON-PROMOTIONAL]\n", flush=True)
    manifest = load_manifest(args.manifest)
    axes = manifest["axes"]
    if args.max_axes:
        axes = axes[:args.max_axes]
    print(f"(manifest) {len(axes)} axes from {args.manifest}; marker ply {manifest['marker_ply']}")

    # Important: load/read parquet corpus before importing torch. See R2 reproduction gotchas.
    games = load_games(2600, 28)
    inst = build(games, MARKER)
    by_gid = {int(r["gid"]): r for r in inst}
    gids = sorted({int(g) for ax in axes for g in ax["gids"] if int(g) in by_gid})
    rows = [by_gid[g] for g in gids]
    gid_to_row = {g: i for i, g in enumerate(gids)}
    orig_texts = [r["uci"] for r in rows]
    shuf_texts = [color_stratified_shuffle(r["uci"], int(r["gid"])) for r in rows]
    print(f"(data) {len(rows)} unique slice instances across axes")

    print("(extract) GPT-2 pretrained / original order ...", flush=True)
    feats_orig = extract_features(orig_texts, pretrained=True, batch_size=args.batch_size)
    print("(extract) GPT-2 pretrained / same-bag shuffled order ...", flush=True)
    feats_shuf = extract_features(shuf_texts, pretrained=True, batch_size=args.batch_size)
    print("(extract) GPT-2 random-init floor / original order ...", flush=True)
    feats_rand = extract_features(orig_texts, pretrained=False, batch_size=args.batch_size)

    results = []
    for ax in axes:
        name = ax["axis"]
        pname, sq = name.split(".")
        fn = PREDS[pname]
        sample_gids = [int(g) for g in ax["gids"] if int(g) in by_gid]
        row_idx = np.array([gid_to_row[g] for g in sample_gids], dtype=int)
        groups = np.array(sample_gids)
        y = np.array([bool(fn(by_gid[g]["sq"][sq])) for g in sample_gids], dtype=int)
        train, val, test = split_indices(groups, y)
        fo = {layer: X[row_idx] for layer, X in feats_orig.items()}
        fs = {layer: X[row_idx] for layer, X in feats_shuf.items()}
        fr = {layer: X[row_idx] for layer, X in feats_rand.items()}
        orig = fit_score_layer(fo, y, train, val, test)
        shuf = fit_score_layer(fs, y, train, val, test)
        rand = fit_score_layer(fr, y, train, val, test)
        surface = float(ax["surface_max"])
        if orig["test_acc"] is None:
            status = "skip_split"
            margins = {"vs_surface": None, "vs_randinit": None, "shuffle_drop": None}
        else:
            margins = {
                "vs_surface": round(orig["test_acc"] - surface, 4),
                "vs_randinit": None if rand["test_acc"] is None else round(orig["test_acc"] - rand["test_acc"], 4),
                "shuffle_drop": None if shuf["test_acc"] is None else round(orig["test_acc"] - shuf["test_acc"], 4),
            }
            status = "cross" if (
                margins["vs_surface"] is not None and margins["vs_surface"] >= 0.15 and
                margins["vs_randinit"] is not None and margins["vs_randinit"] >= 0.15 and
                margins["shuffle_drop"] is not None and margins["shuffle_drop"] >= 0.10
            ) else "no_cross"
        rec = {
            "axis": name,
            "n_slice": len(sample_gids),
            "balance": ax["balance"],
            "surface_max": surface,
            "orig": orig,
            "randinit": rand,
            "shuffle": shuf,
            "margins": margins,
            "status": status,
        }
        results.append(rec)
        print(
            f"  {name:>8} n={len(sample_gids):>3} surf={surface:.3f} "
            f"orig={orig['test_acc']} rand={rand['test_acc']} shuf={shuf['test_acc']} "
            f"m={margins} {status}",
            flush=True,
        )

    n_cross = sum(1 for r in results if r["status"] == "cross")
    out = {
        "spec": "H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md",
        "date": "2026-07-01",
        "rung": "V3-0.5 GPT-2 calibration",
        "non_promotional": True,
        "manifest": args.manifest,
        "layers": LAYERS_GPT2,
        "thresholds": {"vs_surface": 0.15, "vs_randinit": 0.15, "shuffle_drop": 0.10},
        "n_axes": len(results),
        "n_crossing_axes": n_cross,
        "results": results,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "_smoke" if args.max_axes else ""
    out_path = os.path.join(OUT_DIR, f"v3_0_5_gpt2_calibration{suffix}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"\n(result) {n_cross}/{len(results)} axes cross all A1 calibration margins")
    print(f"(wrote) {out_path}")
    print("VERDICT: H1-V3-A1-GPT2-CALIBRATED (non-gate; no R2 claim)")


if __name__ == "__main__":
    main()
