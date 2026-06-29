"""Generate LARGER-tier warm-start heads by capacity-embedding the small heads.

Replication step 2 (larger tier): widen the optimizable controller heads
(guard / arbiter / adapter) from hidden 32-49 to a larger width, WITHOUT changing
the init-time function. Each widened layer has a "live" block that exactly
reproduces the small head plus a random "padding" side-channel whose contribution
to the output is zero at init (so the model starts competent = the small model) but
has non-zero gradient (so training can grow the extra capacity).

This isolates the question "does extra arbiter capacity find a within-kappa dodge?"
from the confound "can a random-init big model relearn the task at all."
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HEADS = ["p_guard", "p_council_arbiter_rl", "m_adapter_rl"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-root", default="results/mesa/h2-frontier/h2_1_binding/ppo_seed_0/models")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--pad-scale", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def widen(payload: dict, H: int, rng: np.random.Generator) -> dict:
    layers = payload["layers"]
    n = len(layers)
    in_dim = len(payload["input_features"])
    new_layers = []
    prev_small = in_dim
    prev_large = in_dim
    for i, L in enumerate(layers):
        W = np.asarray(L["weight"], dtype=np.float64)  # [out_small, prev_small]
        b = np.asarray(L["bias"], dtype=np.float64)    # [out_small]
        out_small = W.shape[0]
        is_output = i == n - 1
        out_large = out_small if is_output else H
        Wn = np.zeros((out_large, prev_large))
        bn = np.zeros(out_large)
        # live block: live rows read ONLY live previous units (cols prev_small:prev_large stay 0)
        Wn[0:out_small, 0:prev_small] = W
        bn[0:out_small] = b
        if not is_output and out_large > out_small:
            # padding rows: read EVERYTHING (random small) → varied features, gradient-alive
            Wn[out_small:out_large, :] = rng.normal(0.0, 0.0 if prev_large == 0 else 1.0, (out_large - out_small, prev_large)) * \
                (np.sqrt(1.0 / max(prev_large, 1)))
        # output layer: cols prev_small:prev_large already 0 → padding contributes 0 at init
        new_layers.append({"weight": Wn.tolist(), "bias": bn.tolist(), "activation": L["activation"]})
        prev_small = out_small
        prev_large = out_large
    out = dict(payload)
    out["layers"] = new_layers
    out["kind"] = f"{payload.get('kind', 'head')}_large"
    out["tier"] = f"large_h{H}"
    return out


def coord_forward(payload: dict, x: np.ndarray) -> np.ndarray:
    mean = np.asarray(payload["normalization"]["mean"])
    std = np.asarray(payload["normalization"]["std"])
    v = (x - mean) / np.maximum(std, 1e-8)
    for L in payload["layers"]:
        v = np.asarray(L["weight"]) @ v + np.asarray(L["bias"])
        if L["activation"] == "tanh":
            v = np.tanh(v)
    return v


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    in_root = Path(args.in_root)
    max_diff_overall = 0.0
    for head in HEADS:
        small = json.loads((in_root / f"{head}.json").read_text(encoding="utf-8"))
        large = widen(small, args.hidden, rng)
        # verify init-time function equivalence (live = small, padding contributes 0)
        in_dim = len(small["input_features"])
        md = 0.0
        for _ in range(200):
            x = rng.normal(0, 2.0, in_dim)
            md = max(md, float(np.max(np.abs(coord_forward(small, x) - coord_forward(large, x)))))
        max_diff_overall = max(max_diff_overall, md)
        sh = [l["weight"].__len__() if isinstance(l["weight"], list) else 0 for l in small["layers"]]
        lh = [len(l["weight"]) for l in large["layers"]]
        (out_root / f"{head}.json").write_text(json.dumps(large) + "\n", encoding="utf-8")
        print(f"{head}: {sh} -> {lh} | init max|d_forward|={md:.2e}", flush=True)
    print(f"LARGER_INIT_{'OK' if max_diff_overall < 1e-9 else 'DRIFT'} (max|d_forward|={max_diff_overall:.2e}, threshold 1e-9)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
