#!/usr/bin/env python
"""H4 Stage 2 — the grokking sweep: measure the generalization order parameter over a (frac x wd) control
plane, with a SEED ENSEMBLE so the bistable/multivalued structure (the cusp's hallmark) can be seen.

Why a seed ensemble: a catastrophe (fold/cusp) lives in a MULTIVALUED equilibrium surface — z* must take
TWO stable values (memorize vs generalize) over a bistable region. A single-init sweep gives a single-
valued graph (no fold for the classifier to read). K seeds/cell expose bistability as BIMODALITY: in a
bistable wedge random seeds fall into different basins (some grok, some don't); in a smooth crossover they
cluster. The bimodal region = the cusp's wedge.

Order parameter z = final test accuracy (0=memorize, 1=generalize). Controls: train fraction (normal/
threshold factor) x weight decay (splitting/sharpness factor). Feeds grokking_catastrophe_analyze.py.

Parallelism: RESILIENT BATCHED multiprocessing. (Plain multiprocessing.Pool is fast but a long-lived
worker eventually dies on this Windows sandbox -> WinError 5 on respawn; ThreadPool stalls on torch CPU
contention.) Fix: process in small BATCHES, each in a FRESH short-lived Pool (so no worker accumulates
enough to die), save INCREMENTALLY, and RESUME-SKIP cells already in the JSON, retrying remaining cells
across rounds until done. A crash at any point only costs the current batch.

NOT public-eligible. Run: python scripts/grokking_catastrophe_sweep.py  (~45 min; rerun to resume)
"""
import json
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path
import numpy as np

warnings.filterwarnings("ignore")

P = 23
STEPS = 25000
FRAC_GRID = [round(x, 4) for x in np.linspace(0.40, 0.66, 7)]
WD_GRID = [round(x, 4) for x in np.geomspace(0.5, 4.0, 8)]
SEEDS = [0, 1, 2]
LR = 1e-3
OUT = Path("results/atlas/h4/grok_sweep.json")
BATCH = 30                                      # cells per fresh Pool round (5/worker -> no worker leak-death)


def run_cell(args):
    frac, wd, seed = args
    import warnings as _w
    _w.filterwarnings("ignore")
    import numpy as _np
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    p = P
    a = torch.tensor(_np.arange(p).repeat(p))
    b = torch.tensor(_np.tile(_np.arange(p), p))
    y = torch.tensor((_np.arange(p).repeat(p) + _np.tile(_np.arange(p), p)) % p)
    n = p * p
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).numpy()
    ntr = int(frac * n)
    tr, te = idx[:ntr], idx[ntr:]

    class GrokMLP(nn.Module):
        def __init__(s):
            super().__init__()
            s.ea = nn.Embedding(p, 48); s.eb = nn.Embedding(p, 48)
            s.net = nn.Sequential(nn.Linear(96, 128), nn.ReLU(), nn.Linear(128, p))

        def forward(s, a, b):
            return s.net(torch.cat([s.ea(a), s.eb(b)], -1))

    m = GrokMLP()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    at, bt, yt = a[tr], b[tr], y[tr]
    ae, be, ye = a[te], b[te], y[te]
    grok_step = -1
    for s in range(STEPS):
        opt.zero_grad(); lf(m(at, bt), yt).backward(); opt.step()
        if s % 500 == 0:
            with torch.no_grad():
                tea = (m(ae, be).argmax(-1) == ye).float().mean().item()
            if grok_step < 0 and tea > 0.9:
                grok_step = s
    with torch.no_grad():
        tra = (m(at, bt).argmax(-1) == yt).float().mean().item()
        tea = (m(ae, be).argmax(-1) == ye).float().mean().item()
        wn = sum((q ** 2).sum().item() for q in m.parameters()) ** 0.5
    return {"frac": frac, "wd": wd, "seed": seed, "train": round(tra, 4),
            "test": round(tea, 4), "wnorm": round(wn, 3), "grok_step": grok_step}


def key(r):
    return (r["frac"], r["wd"], r["seed"])


def main():
    all_cells = [(f, w, s) for f in FRAC_GRID for w in WD_GRID for s in SEEDS]
    results = []
    if OUT.exists():
        try:
            results = json.load(open(OUT)).get("results", [])
        except Exception:
            results = []
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        OUT.write_text(json.dumps({"p": P, "steps": STEPS, "lr": LR, "frac_grid": FRAC_GRID,
                                   "wd_grid": WD_GRID, "seeds": SEEDS, "results": results}, indent=1))

    print(f"H4 grokking sweep: p={P} steps={STEPS}  {len(all_cells)} runs total; {len(results)} already done.",
          flush=True)
    t0 = time.time()
    rnd = 0
    while rnd < 15:
        done = {key({"frac": c[0], "wd": c[1], "seed": c[2]}) for c in
                [(r["frac"], r["wd"], r["seed"]) for r in results]}
        remaining = [c for c in all_cells if c not in done]
        if not remaining:
            break
        rnd += 1
        batch = remaining[:BATCH]
        print(f"  round {rnd}: {len(results)}/{len(all_cells)} done; running {len(batch)} "
              f"({len(remaining)} remain)  [{time.time()-t0:.0f}s]", flush=True)
        try:
            with Pool(processes=6) as pool:
                for r in pool.imap_unordered(run_cell, batch):
                    results.append(r)
                    save()
        except Exception as e:
            print(f"    round {rnd} interrupted ({type(e).__name__}: {e}); saved, retrying remaining.",
                  flush=True)
            save()
    save()
    print(f"sweep complete: {len(results)}/{len(all_cells)} in {time.time()-t0:.0f}s -> {OUT}", flush=True)

    print("\nmean test-acc surface (rows=frac, cols=wd):")
    print("  frac\\wd " + " ".join(f"{w:5.2f}" for w in WD_GRID))
    for f in FRAC_GRID:
        row = [f"{np.mean([r['test'] for r in results if r['frac'] == f and r['wd'] == w] or [float('nan')]):5.2f}"
               for w in WD_GRID]
        print(f"  {f:5.2f}  " + " ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
