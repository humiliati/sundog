"""
U-5 — the representation-vs-optimization gap: constructed net vs SGD-trained net
(algorithmic-approximation slate 4).

Every approximant in the lane is *constructed* (explicit weights with a proved L-infinity bound),
never *trained*. The sharpest construction is the Telgarsky sawtooth for x^2 on [0,1]:

    R_m(x) = x - sum_{k=1}^m T^[k](x) / 4^k ,   T(x) = 1 - |2x - 1| ,

machine-checked in `SawtoothShared`/`SawtoothDag` to have L-infinity error <= 1/(4*4^m) at depth ~m
and O(m) gates -- accuracy EXPONENTIAL in depth. This script asks the optimization question: at the
*same depth*, does plain SGD find that accuracy, or does it plateau because it cannot exploit depth?

Design (deterministic, CPU):
  * target f(x) = x^2 on [0,1] (the lane's canonical analytic gate).
  * CONSTRUCTED: R_m for m = 1..M, L-infinity error on a dense grid (should track 1/(4*4^m)).
  * SGD-DEEP: a depth-m ReLU MLP (modest width) trained on x^2 by Adam, best-of-restarts, L-infinity.
  * SGD-SHALLOW: a 1-hidden-layer ReLU MLP whose parameter count is matched to the depth-m net
    (SGD's best-conditioned architecture), best-of-restarts -- the fair "what SGD can actually train".
  * gap(m) = (best SGD L-infinity) / (constructed L-infinity).

Prediction (REPRESENTATION_GAP): the constructed error decays exponentially in depth while SGD's best
plateaus (it does not access the depth-efficiency), so gap(m) GROWS with m -- representability is not
trainability for the depth-efficient construction.

Falsifier NO_REPRESENTATION_GAP: SGD matches the constructed error at the same depth/budget across m
(gap ~ O(1)) -- the construction's depth-efficiency is reachable by plain SGD, so there is no gap.

CPU, seeded. Prints a table + a verdict line.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn

torch.set_num_threads(4)

LO, HI = 0.0, 1.0
GRID = np.linspace(LO, HI, 2001).astype(np.float32)        # dense L-infinity grid
GRID_T = torch.from_numpy(GRID).unsqueeze(1)
YGRID = (GRID ** 2).astype(np.float32)
YGRID_T = torch.from_numpy(YGRID).unsqueeze(1)


# ----- the constructed approximant (the proved closed form) -----------------------------------
def tent(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(2.0 * x - 1.0)


def R_m(m: int, x: np.ndarray) -> np.ndarray:
    """R_m = x - sum_{k=1}^m T^[k](x)/4^k -- the Yarotsky/Telgarsky sawtooth (closed form)."""
    s = x.copy()
    tk = x.copy()
    for k in range(1, m + 1):
        tk = tent(tk)
        s = s - tk / (4.0 ** k)
    return s


def constructed_linf(m: int) -> float:
    return float(np.max(np.abs(YGRID - R_m(m, GRID))))


# ----- SGD-trained ReLU nets ------------------------------------------------------------------
class ReLUNet(nn.Module):
    def __init__(self, depth: int, width: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(1, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers += [nn.Linear(width, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def param_count(depth: int, width: int) -> int:
    return sum(p.numel() for p in ReLUNet(depth, width).parameters())


def train_linf(depth: int, width: int, seed: int, n_train: int = 1024,
               epochs: int = 3000, lr: float = 0.01) -> float:
    """Train a depth x width ReLU net on x^2 over [0,1]; return L-infinity error on the dense grid."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    xtr = rng.uniform(LO, HI, (n_train, 1)).astype(np.float32)
    ytr = torch.from_numpy((xtr[:, 0] ** 2).astype(np.float32)).unsqueeze(1)
    xtr_t = torch.from_numpy(xtr)
    model = ReLUNet(depth, width)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    for ep in range(epochs):
        opt.zero_grad()
        lossf(model(xtr_t), ytr).backward()
        opt.step()
    with torch.no_grad():
        return float((model(GRID_T) - YGRID_T).abs().max().item())


def sgd_best_linf(depth: int, width: int, restarts: int = 5) -> float:
    return min(train_linf(depth, width, seed=7919 * depth + 31 * width + s) for s in range(restarts))


# ----- matched-budget shallow width -----------------------------------------------------------
def matched_shallow_width(depth: int, deep_width: int) -> int:
    """Width of a 1-hidden-layer net whose param count first meets/exceeds the depth-m net's."""
    target = param_count(depth, deep_width)
    w = 1
    while param_count(1, w) < target:
        w += 1
    return w


# ----- experiment -----------------------------------------------------------------------------
MS = [1, 2, 3, 4, 5, 6, 7, 8]
DEEP_WIDTH = 8
RESTARTS = 5


def main():
    print("###### U-5: representation-vs-optimization gap (constructed sawtooth vs SGD) ######")
    print("target f(x) = x^2 on [0,1];  L-infinity error on a 2001-pt grid;  best-of-"
          f"{RESTARTS} restarts\n")
    print(f"{'m':>2} {'constr.':>10} {'1/(4*4^m)':>10} | {'SGD-deep':>10} (d={'m'},w={DEEP_WIDTH}) "
          f"| {'SGD-shal':>10} (d=1) | {'gap=SGD/constr':>14}")
    rows = []
    for m in MS:
        c = constructed_linf(m)
        bound = 1.0 / (4.0 * 4.0 ** m)
        deep = sgd_best_linf(m, DEEP_WIDTH, RESTARTS)
        wsh = matched_shallow_width(m, DEEP_WIDTH)
        shal = sgd_best_linf(1, wsh, RESTARTS)
        best_sgd = min(deep, shal)
        gap = best_sgd / c if c > 0 else float("inf")
        rows.append((m, c, bound, deep, shal, best_sgd, gap))
        print(f"{m:>2} {c:>10.2e} {bound:>10.2e} | {deep:>10.2e}          "
              f"| {shal:>10.2e} (w={wsh:>3}) | {gap:>14.1f}")

    # verdict: does the gap grow? (best SGD error vs constructed across m)
    gaps = [r[6] for r in rows]
    constr = [r[1] for r in rows]
    sgd = [r[5] for r in rows]
    # construction keeps improving exponentially; does SGD track it, or plateau?
    constr_drop = constr[0] / constr[-1]          # how much the construction improved m=1 -> M
    sgd_drop = sgd[0] / max(sgd[-1], 1e-12)        # how much SGD's best improved
    gap_growth = gaps[-1] / max(gaps[0], 1e-12)

    print(f"\n--- VERDICT (U-5) ---")
    print(f"  constructed L-inf  m=1 -> m={MS[-1]}: {constr[0]:.2e} -> {constr[-1]:.2e} "
          f"(improved {constr_drop:.0f}x)")
    print(f"  best SGD   L-inf   m=1 -> m={MS[-1]}: {sgd[0]:.2e} -> {sgd[-1]:.2e} "
          f"(improved {sgd_drop:.1f}x)")
    print(f"  gap (SGD/constructed): {gaps[0]:.1f}x  ->  {gaps[-1]:.1f}x   (grew {gap_growth:.0f}x)")
    if gap_growth >= 5.0 and constr_drop >= 50.0 * sgd_drop:
        print("  -> REPRESENTATION_GAP: the constructed net's accuracy decays exponentially in depth")
        print("     while SGD's best plateaus -- depth-efficiency is representable but not trainable;")
        print("     the gap grows with depth. (Construction is in Lean; SGD is the optimization wall.)")
    elif gap_growth >= 2.0:
        print("  -> PARTIAL: a growing but not sharp gap; SGD lags the construction with depth but not")
        print("     dramatically in this budget.")
    else:
        print("  -> NO_REPRESENTATION_GAP: SGD tracks the constructed error across depth; the")
        print("     depth-efficient accuracy is reachable by plain SGD in this budget.")


if __name__ == "__main__":
    main()
