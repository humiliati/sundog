#!/usr/bin/env python
"""H4 Stage 2b — the DECOUPLED bistability probe: the clean cusp-vs-null test.

The main sweep (grokking_catastrophe_sweep.py) confounded the data split with the init (one `seed` set
both), so its apparent "bistability" (partial-grok cells) turned out to be ONE seed grokking everywhere =
a split-QUALITY effect, not init-dependent two-basin bistability. This probe DECOUPLES them: at a few
marginal transition cells, FIX the data split and vary ONLY the init. Then:
  * if outcome is SPLIT-DETERMINED (a split either always groks or never groks, tight over inits) -> NO
    bistability -> grokking is a (seed/split-sensitive) single-valued crossover -> NULL (no cusp).
  * if outcome is INIT-DEPENDENT (the SAME split groks for some inits, not others -> bimodal over inits)
    -> genuine two-basin BISTABILITY -> cusp-consistent.

Probes frac in {0.487,0.53,0.573} x wd in {0.91,2.21} (the marginal wd's where only one seed groked) x
split_seed in {0,1,2} x init_seed in {0..7}. Resilient batched Pool. NOT public-eligible.
Run: python scripts/grokking_bistability_probe.py
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
LR = 1e-3
FRACS = [0.487, 0.53, 0.573]
WDS = [0.91, 2.21]
SPLIT_SEEDS = [0, 1, 2]
INIT_SEEDS = [0, 1, 2, 3, 4, 5, 6, 7]
OUT = Path("results/atlas/h4/grok_bistability.json")
BATCH = 30


def run_cell(args):
    frac, wd, split_seed, init_seed = args
    import warnings as _w
    _w.filterwarnings("ignore")
    import numpy as _np
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    p = P
    a = torch.tensor(_np.arange(p).repeat(p))
    b = torch.tensor(_np.tile(_np.arange(p), p))
    y = torch.tensor((_np.arange(p).repeat(p) + _np.tile(_np.arange(p), p)) % p)
    n = p * p
    gsplit = torch.Generator().manual_seed(1000 + split_seed)     # data split: from split_seed ONLY
    idx = torch.randperm(n, generator=gsplit).numpy()
    ntr = int(frac * n)
    tr, te = idx[:ntr], idx[ntr:]
    torch.manual_seed(7000 + init_seed)                           # model init: from init_seed ONLY

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
    for s in range(STEPS):
        opt.zero_grad(); lf(m(at, bt), yt).backward(); opt.step()
    with torch.no_grad():
        tea = (m(ae, be).argmax(-1) == ye).float().mean().item()
    return {"frac": frac, "wd": wd, "split_seed": split_seed, "init_seed": init_seed,
            "test": round(tea, 4)}


def main():
    cells = [(f, w, sp, ii) for f in FRACS for w in WDS for sp in SPLIT_SEEDS for ii in INIT_SEEDS]
    results = []
    if OUT.exists():
        try:
            results = json.load(open(OUT)).get("results", [])
        except Exception:
            results = []
    OUT.parent.mkdir(parents=True, exist_ok=True)

    def save():
        OUT.write_text(json.dumps({"p": P, "steps": STEPS, "fracs": FRACS, "wds": WDS,
                                   "split_seeds": SPLIT_SEEDS, "init_seeds": INIT_SEEDS,
                                   "results": results}, indent=1))

    print(f"H4 decoupled bistability probe: {len(cells)} runs (fixed split x varied init)", flush=True)
    t0 = time.time(); rnd = 0
    while rnd < 15:
        done = {(r["frac"], r["wd"], r["split_seed"], r["init_seed"]) for r in results}
        remaining = [c for c in cells if c not in done]
        if not remaining:
            break
        rnd += 1
        batch = remaining[:BATCH]
        print(f"  round {rnd}: {len(results)}/{len(cells)} done; {len(batch)} now ({time.time()-t0:.0f}s)",
              flush=True)
        try:
            with Pool(processes=6) as pool:
                for r in pool.imap_unordered(run_cell, batch):
                    results.append(r); save()
        except Exception as e:
            print(f"    round {rnd} interrupted ({type(e).__name__}); saved, retrying.", flush=True)
            save()
    save()
    print(f"probe complete: {len(results)}/{len(cells)} in {time.time()-t0:.0f}s -> {OUT}\n", flush=True)

    # verdict: for each (frac, wd, split), the spread over inits -> bimodal (bistable) or tight?
    print("Decoupled spread over INITS at fixed split (BIMODAL over inits = genuine bistability):")
    from collections import defaultdict
    grp = defaultdict(list)
    for r in results:
        grp[(r["frac"], r["wd"], r["split_seed"])].append(r["test"])
    n_bistable = 0
    for (f, w, sp), ts in sorted(grp.items()):
        ts = sorted(round(t, 2) for t in ts)
        bistable = (max(ts) > 0.7 and min(ts) < 0.3)
        n_bistable += bistable
        print(f"  frac={f} wd={w:4.2f} split={sp}: inits={ts}  {'<<BISTABLE (init-dependent!)' if bistable else 'tight (split-determined)'}")
    print(f"\n{n_bistable} of {len(grp)} (frac,wd,split) groups are INIT-BISTABLE.")
    print("VERDICT:", "BISTABILITY CONFIRMED (cusp-consistent)" if n_bistable >= 2
          else "NO genuine init-bistability -> grokking here is a split-determined single-valued transition"
               " (NULL: not a Whitney bistable cusp).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
