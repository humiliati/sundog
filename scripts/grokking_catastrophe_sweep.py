#!/usr/bin/env python
"""H4 Stage 2 — the grokking sweep: measure the generalization order parameter over a (frac x wd) control
plane, with a SEED ENSEMBLE so the bistable/multivalued structure (the cusp's hallmark) can be seen.

Why a seed ensemble: a catastrophe (fold/cusp) lives in a MULTIVALUED equilibrium surface — the order
parameter z* must take TWO stable values (memorize vs generalize) over a bistable region. A single-init
sweep gives a single-valued graph (no fold for the classifier to read). Running K seeds per cell exposes
the bistability as BIMODALITY: in a truly bistable wedge, random seeds fall into different basins (some
grok, some don't); in a smooth crossover they cluster. The bimodal region = the cusp's wedge.

Order parameter z = final test accuracy (0 = memorized, 1 = generalized). Controls: train fraction (the
'normal'/threshold factor) x weight decay (the 'splitting'/sharpness factor — the scan showed a wd WINDOW
where grokking occurs, opening as frac rises). Output JSON feeds grokking_catastrophe_analyze.py, which
builds the equilibrium chart and points the calibrated jet classifier at it.

NOT public-eligible. Run: python scripts/grokking_catastrophe_sweep.py  (parallel over cores; ~25-35 min)
"""
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np

P = 23
STEPS = 25000
FRAC_GRID = [round(x, 4) for x in np.linspace(0.40, 0.66, 7)]
WD_GRID = [round(x, 4) for x in np.geomspace(0.5, 4.0, 8)]
SEEDS = [0, 1, 2]
LR = 1e-3


def _run_cell(args):
    frac, wd, seed = args
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as _np
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    _np.random.seed(seed)
    p = P
    a = _np.arange(p).repeat(p)
    b = _np.tile(_np.arange(p), p)
    y = (a + b) % p
    a, b, y = torch.tensor(a), torch.tensor(b), torch.tensor(y)
    n = p * p
    idx = _np.random.permutation(n)
    ntr = int(frac * n)
    tr, te = idx[:ntr], idx[ntr:]

    class GrokMLP(nn.Module):
        def __init__(self, p, d=48, h=128):
            super().__init__()
            self.ea = nn.Embedding(p, d)
            self.eb = nn.Embedding(p, d)
            self.net = nn.Sequential(nn.Linear(2 * d, h), nn.ReLU(), nn.Linear(h, p))

        def forward(self, a, b):
            return self.net(torch.cat([self.ea(a), self.eb(b)], -1))

    m = GrokMLP(p)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=wd)
    lf = nn.CrossEntropyLoss()
    at, bt, yt = a[tr], b[tr], y[tr]
    ae, be, ye = a[te], b[te], y[te]
    grok_step = -1
    for s in range(STEPS):
        opt.zero_grad()
        lf(m(at, bt), yt).backward()
        opt.step()
        if s % 500 == 0:
            with torch.no_grad():
                tea = (m(ae, be).argmax(-1) == ye).float().mean().item()
            if grok_step < 0 and tea > 0.9:
                grok_step = s
    with torch.no_grad():
        tra = (m(at, bt).argmax(-1) == yt).float().mean().item()
        tea = (m(ae, be).argmax(-1) == ye).float().mean().item()
        wn = sum((q ** 2).sum().item() for q in m.parameters()) ** 0.5
    return {"frac": frac, "wd": wd, "seed": seed, "train": tra, "test": tea,
            "wnorm": wn, "grok_step": grok_step}


def main():
    cells = [(f, w, s) for f in FRAC_GRID for w in WD_GRID for s in SEEDS]
    print(f"H4 grokking sweep: p={P} steps={STEPS}  {len(FRAC_GRID)}x{len(WD_GRID)} grid x {len(SEEDS)} seeds "
          f"= {len(cells)} runs", flush=True)
    t0 = time.time()
    with Pool(processes=6) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_run_cell, cells)):
            results.append(r)
            if (i + 1) % 12 == 0:
                print(f"  {i+1}/{len(cells)} done ({time.time()-t0:.0f}s)  last: "
                      f"frac={r['frac']} wd={r['wd']} seed={r['seed']} test={r['test']:.2f}", flush=True)
    dt = time.time() - t0
    print(f"sweep done in {dt:.0f}s", flush=True)

    out = {"p": P, "steps": STEPS, "lr": LR, "frac_grid": FRAC_GRID, "wd_grid": WD_GRID,
           "seeds": SEEDS, "results": results}
    pth = Path("results/atlas/h4/grok_sweep.json")
    pth.parent.mkdir(parents=True, exist_ok=True)
    pth.write_text(json.dumps(out, indent=1))
    print(f"wrote {pth}", flush=True)

    # quick textual surface: mean test acc per (frac,wd)
    print("\nmean test-acc surface (rows=frac, cols=wd):")
    print("  frac\\wd " + " ".join(f"{w:5.2f}" for w in WD_GRID))
    for f in FRAC_GRID:
        row = []
        for w in WD_GRID:
            ts = [r["test"] for r in results if r["frac"] == f and r["wd"] == w]
            row.append(f"{np.mean(ts):5.2f}")
        print(f"  {f:5.2f}  " + " ".join(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
