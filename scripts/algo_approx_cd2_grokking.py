"""
C-D2 — the grokking / generalization-threshold experiment (algorithmic-approximation slate).

Tests the slate prediction: for a *tropical* target, a ReLU MLP's generalization onset
tracks the target's tropical (compiled) complexity — `compileToDag` size as the upper-bound
target. Falsifier `GROKS_FAR_BELOW_COMPILED_SIZE`: clean generalization far below the
compiled size, unrelated to tropical complexity.

Controlled target with KNOWN tropical complexity:
    f_k(x) = max_{j=1..k} (s_j * x + b_j),  the max of k tangent lines to g(x)=0.5 x^2.
This is a convex piecewise-linear function with EXACTLY k active pieces over the sampled
range (a tropical / max-plus polynomial of complexity k). A 1-hidden-layer ReLU net
represents a 1-D convex PL function with m pieces using ~ m-1 ReLU units (#breakpoints =
#active units), so the predicted generalization-width threshold is ~ k-1, i.e. linear in
the tropical complexity k.

Two parts:
  (A) CAPACITY THRESHOLD (primary, falsifiable): for each k, sweep hidden width w and find
      the smallest w whose held-out error clears a variance-explained bar. Prediction:
      threshold(k) grows ~linearly with k.
  (B) GROKKING DYNAMICS (secondary): small train set + weight decay, long training; look
      for delayed generalization (test error dropping well after train error).

Deterministic (seeded). CPU. Prints a table + a verdict line.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)
DEVICE = "cpu"


def target_pieces(k: int, lo: float = -2.0, hi: float = 2.0, kind: str = "smooth"):
    """k-piece convex tropical polynomial f = max_j (s_j x + b_j).

    kind="smooth":    tangent lines to g(x)=0.5 x^2. The per-breakpoint slope jump is
                      (hi-lo)/(k-1) -> SHRINKS with k, so the target approaches a smooth
                      parabola and its *approximation* complexity is far below the exact k
                      pieces (a confound: exact size != effective complexity).
    kind="essential": a convex PL with FIXED slope jumps (=1) so every breakpoint is sharp
                      and essential regardless of k -> effective complexity tracks k."""
    if kind == "smooth":
        ts = np.linspace(lo, hi, k)
        return ts, -0.5 * ts ** 2
    elif kind == "essential":
        zs = np.linspace(lo, hi, k + 1)[1:-1]          # k-1 interior breakpoints
        slopes = (np.arange(k) - (k - 1) / 2.0)        # k increasing slopes, fixed jump 1
        intercepts = np.zeros(k)
        for i in range(1, k):                          # continuity at each breakpoint
            intercepts[i] = intercepts[i - 1] + (slopes[i - 1] - slopes[i]) * zs[i - 1]
        return slopes, intercepts
    else:
        raise ValueError(kind)


def f_k(x: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray) -> np.ndarray:
    # x: (N,1) -> (N,)  max_j (s_j x + b_j)
    vals = x[:, :1] * slopes[None, :] + intercepts[None, :]   # (N,k)
    return vals.max(axis=1)


class MLP(nn.Module):
    def __init__(self, width: int, depth: int = 1):
        super().__init__()
        layers = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def make_data(k, n_train, n_test, seed, lo=-2.5, hi=2.5, kind="smooth"):
    rng = np.random.default_rng(seed)
    slopes, intercepts = target_pieces(k, kind=kind)
    xtr = rng.uniform(lo, hi, size=(n_train, 1)).astype(np.float32)
    xte = rng.uniform(lo, hi, size=(n_test, 1)).astype(np.float32)
    ytr = f_k(xtr, slopes, intercepts).astype(np.float32)
    yte = f_k(xte, slopes, intercepts).astype(np.float32)
    return (torch.from_numpy(xtr), torch.from_numpy(ytr).unsqueeze(1),
            torch.from_numpy(xte), torch.from_numpy(yte).unsqueeze(1))


def train_eval(k, width, seed, n_train=400, n_test=400, epochs=3000, wd=0.0, lr=0.01,
               kind="smooth"):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = make_data(k, n_train, n_test, seed, kind=kind)
    model = MLP(width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = lossf(model(xtr), ytr)
        loss.backward()
        opt.step()
    with torch.no_grad():
        te = lossf(model(xte), yte).item()
    var = yte.var(unbiased=False).item()
    return te / max(var, 1e-12)   # fraction of variance UNexplained on held-out data


def capacity_threshold(ks=(2, 3, 4, 6, 8), seeds=(0, 1, 2), wmax_factor=2, bar=0.01,
                       kind="smooth"):
    """For each k, sweep w; threshold = smallest w with median rel-test-MSE < bar."""
    print(f"=== CAPACITY THRESHOLD [{kind} target]: median held-out frac-variance-unexplained ===")
    print("    (prediction: threshold width grows ~linearly with k; ~ k-1)")
    results = {}
    for k in ks:
        wmax = wmax_factor * k + 2
        row = []
        threshold = None
        for w in range(1, wmax + 1):
            rels = [train_eval(k, w, s, kind=kind) for s in seeds]
            med = float(np.median(rels))
            row.append((w, med))
            if threshold is None and med < bar:
                threshold = w
        results[k] = (row, threshold)
        sweep = "  ".join(f"w{w}:{med:.3f}" for w, med in row)
        print(f"k={k:2d}  threshold(rel<{bar})= {threshold}   | {sweep}")
    return results


def verdict(res, label):
    ks = sorted(res.keys())
    ths = [res[k][1] for k in ks]
    print(f"\n--- VERDICT [{label}] ---")
    print(f"  k        = {ks}")
    print(f"  threshold= {ths}   (predicted ~ k-1)")
    if not all(t is not None for t in ths):
        print("  -> INCONCLUSIVE: some k never cleared the bar.")
        return
    far_below = ths[-1] <= max(2, ks[-1] // 3)            # large k generalizes at tiny width
    tracks = all(ths[i] <= ths[i + 1] for i in range(len(ths) - 1)) and ths[-1] >= ks[-1] - 2
    if tracks:
        print("  -> SUPPORT: threshold tracks tropical complexity k (~ k-1).")
    elif far_below:
        print("  -> GROKS_FAR_BELOW_COMPILED_SIZE: generalizes far below exact piece count k.")
    else:
        print("  -> INCONCLUSIVE: threshold neither clearly tracks k nor sits far below.")


def grokking_dynamics(k=6, width=10, seed=0, n_train=40, epochs=8000, wd=1e-2, lr=0.01):
    """Small data + weight decay: log train/test over time; detect delayed generalization."""
    print("\n=== (B) GROKKING DYNAMICS: small data + weight decay ===")
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = make_data(k, n_train, 400, seed)
    model = MLP(width).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.MSELoss()
    var_te = yte.var(unbiased=False).item()
    var_tr = ytr.var(unbiased=False).item()
    log = []
    for ep in range(epochs + 1):
        opt.zero_grad()
        tr = lossf(model(xtr), ytr)
        tr.backward()
        opt.step()
        if ep % 200 == 0:
            with torch.no_grad():
                te = lossf(model(xte), yte).item()
            log.append((ep, tr.item() / max(var_tr, 1e-12), te / max(var_te, 1e-12)))
    # detect: epoch where train first generalizes (<0.05) vs test first generalizes (<0.05)
    tr_gen = next((ep for ep, t, _ in log if t < 0.05), None)
    te_gen = next((ep for ep, _, v in log if v < 0.05), None)
    for ep, t, v in log:
        if ep % 1000 == 0:
            print(f"  ep {ep:5d}  train_rel={t:.3f}  test_rel={v:.3f}")
    print(f"  train clears 0.05 @ ep {tr_gen};  test clears 0.05 @ ep {te_gen}")
    delayed = (tr_gen is not None and te_gen is not None and te_gen >= tr_gen + 1000)
    print(f"  delayed generalization (grokking-like, test lags train by >=1000 ep): {delayed}")
    return tr_gen, te_gen, delayed


def main():
    np.random.seed(0)
    print("############ C-D2: generalization-width threshold vs tropical complexity ############\n")
    res_smooth = capacity_threshold(kind="smooth")
    verdict(res_smooth, "smooth target (tangent-to-parabola; slope jumps shrink with k)")
    print()
    res_ess = capacity_threshold(kind="essential")
    verdict(res_ess, "essential target (fixed slope jumps; every piece matters)")
    print("\n=== CONTRAST ===")
    print("  If 'smooth' generalizes far below k but 'essential' tracks k, the driver is the")
    print("  EFFECTIVE (approximation) complexity, not the exact tropical piece count —")
    print("  exact compiled size is an upper bound; the threshold tracks the eps-essential pieces.")
    grokking_dynamics()


if __name__ == "__main__":
    main()
