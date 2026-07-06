#!/usr/bin/env python3
"""FC-4: the sigma (order-meter) split -- pixel vs object filtration on abstraction-style tasks.

The GEN-1 pre-test (GENERATOR_CLASS_SLATE.md): the object-centric-DSL bet is precisely the bet that
ARC-style rules are LOW-ORDER IN THE OBJECT FILTRATION and high-order in the pixel filtration. This
probe measures that sigma split on a CONTROLLED toy family we own (per the FC-4 hook in
FIND_CHECK_SUFFICIENCY_SLATE.md: toy tasks, no ARC dataset dependency), with the cheap confounds
designed out and two controls that keep the apparatus honest.

THE FAMILY (L1, primary): 16x16 grids, 3-7 solid non-adjacent rectangles (1-cell gaps => every rect
is its own connected component), colors 1..4 WITH REUSE, and TOTAL COLORED AREA FIXED at 48 -- so the
order-1 pixel statistic (total count) is constant by construction. Latent = AREA OF THE LARGEST
CONNECTED COMPONENT: the canonical ARC-style object rule ("the largest object"), an order-1 statistic
of the object filtration BY CONSTRUCTION (that is what "abstraction-style" means here), while in the
pixel filtration it is a global connectivity-class property. Named imported anchor (measured, NOT
proved here): Minsky-Papert perceptron order -- connectivity-class predicates are not computable by
bounded-order / diameter-limited feature families.

PRE-REGISTERED GATES:
  SPLIT     := obj_order1_R2 >= 0.80  AND  obj_order1_R2 - best_bounded_pixel_R2 >= 0.30
               (bounded pixel ladder = hist k1 / adjacent-pairs k2 / 2x2 / 3x3 patch histograms;
                the raw-grid MLP arm is a trained-encoder DIAGNOSTIC, outside the bounded ladder)
  APPARATUS := on the SAME grids, latent = count of color-1 pixels -> hist R2 >= 0.90
               (the pixel ladder is live, not rigged to fail)
  REVERSE   := on dot-sprinkled grids, latent = color-5 dot count ->
               hist R2 >= 0.90 AND hist R2 - obj R2 >= 0.20
               (the pixel filtration wins on its home turf => sigma is PER-FILTRATION -- the schema
                caveat honored -- and the object filtration is NOT universally richer; kills the
                "feature-map artifact" reading)
               v2 DESIGN (v1 flaw self-caught + reported): v1 (variable m, one dot color) leaked --
               n_comps minus the count of area>=4 rows reconstructs the dot count EXACTLY (obj
               R2=0.994). v2 isolates the latent from the frozen object vocabulary: m in {6,7} rects
               (top-6 rows = rects only, no dot rows visible) + 0-6 NUISANCE color-6 dots, so n_comps
               = m + nd5 + nd6 confounds the species; the histogram still reads color-5 exactly.
  trivial-FAIL: shuffled latent -> ~0.
HONEST LEAKS NAMED IN ADVANCE: bounded patches see corners/edges => estimate the COMPONENT COUNT =>
partial R2 on L1 via fragmentation (max-area anticorrelates with n at fixed total); the object
vocabulary's n_comps entry partially leaks the REVERSE dot count. Both reported in the dissection.
FROZEN object vocabulary (order-1): top-6 components by area as (area,color,h,w) rows + n_comps.
Run: python scripts/findcheck_fc4_sigma_split.py    NOT public-eligible.
"""
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold, cross_val_score

G = 16          # grid side
TOTAL = 48      # fixed total colored area (kills the order-1 total-count statistic by construction)
N = 400         # samples per dataset
SEED = 20260701
AREAS = {4: (2, 2), 6: (2, 3), 8: (2, 4), 9: (3, 3), 12: (3, 4), 16: (4, 4), 20: (4, 5)}


def _completable():
    ok = {0}
    for _ in range(12):
        ok |= {s + a for s in ok for a in AREAS if s + a <= TOTAL}
    return ok


COMPLETABLE = _completable()


def sample_partition(rng, lo=3, hi=7):
    """Random multiset of areas from AREAS summing to TOTAL with lo..hi parts."""
    while True:
        parts, rem = [], TOTAL
        while rem > 0:
            opts = [a for a in AREAS if (rem - a) in COMPLETABLE]
            if not opts:
                break
            a = int(rng.choice(opts)); parts.append(a); rem -= a
        if rem == 0 and lo <= len(parts) <= hi:
            return parts


def place_rects(parts, rng):
    """Place rectangles with a 1-cell margin (so every rect is its own component). Colors 1..4."""
    for _ in range(60):                                   # retry whole layouts
        g = np.zeros((G, G), dtype=np.int8); ok = True
        for a in parts:
            h, w = AREAS[a]
            if rng.random() < 0.5:
                h, w = w, h
            placed = False
            for _ in range(80):
                r, c = rng.integers(0, G - h + 1), rng.integers(0, G - w + 1)
                r0, c0 = max(0, r - 1), max(0, c - 1)
                if g[r0:r + h + 1, c0:c + w + 1].max() == 0:
                    g[r:r + h, c:c + w] = rng.integers(1, 5); placed = True; break
            if not placed:
                ok = False; break
        if ok:
            return g


def components(g):
    """4-connected same-color components of nonzero cells -> list of (area, color, h, w)."""
    seen = np.zeros_like(g, dtype=bool); out = []
    for r in range(G):
        for c in range(G):
            if g[r, c] != 0 and not seen[r, c]:
                col, stack, cells = g[r, c], [(r, c)], []
                seen[r, c] = True
                while stack:
                    y, x = stack.pop(); cells.append((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < G and 0 <= nx < G and not seen[ny, nx] and g[ny, nx] == col:
                            seen[ny, nx] = True; stack.append((ny, nx))
                ys = [y for y, _ in cells]; xs = [x for _, x in cells]
                out.append((len(cells), int(col), max(ys) - min(ys) + 1, max(xs) - min(xs) + 1))
    return out


# ---------------- frozen feature maps ---------------- #
def f_hist(g):                                            # pixel order-1 (colors 0..6)
    return np.bincount(g.ravel(), minlength=7).astype(float)


def f_pairs(g):                                           # pixel order-2 (adjacent pairs, h+v)
    out = np.zeros((7, 7))
    for a, b in zip(g[:, :-1].ravel(), g[:, 1:].ravel()):
        out[min(a, b), max(a, b)] += 1
    for a, b in zip(g[:-1, :].ravel(), g[1:, :].ravel()):
        out[min(a, b), max(a, b)] += 1
    return out[np.triu_indices(7)]


def f_patch(g, k, buckets):                               # kxk patch-pattern histogram (hashed)
    out = np.zeros(buckets)
    for r in range(G - k + 1):
        for c in range(G - k + 1):
            out[hash(tuple(g[r:r + k, c:c + k].ravel().tolist())) % buckets] += 1
    return out


def f_obj(g):                                             # OBJECT order-1: top-6 comps by area + count
    comps = sorted(components(g), reverse=True)[:6]
    rows = [list(t) for t in comps] + [[0, 0, 0, 0]] * (6 - len(comps))
    return np.array([v for row in rows for v in row] + [len(components(g))], dtype=float)


def own_r2(X, y):
    X = np.asarray(X, float); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    kf = KFold(4, shuffle=True, random_state=0)
    lin = cross_val_score(LinearRegression(), X, y, cv=kf, scoring="r2").mean()
    mlp = cross_val_score(MLPRegressor(hidden_layer_sizes=(32,), max_iter=500, random_state=0),
                          X, y, cv=kf, scoring="r2").mean()
    return max(0.0, lin, mlp)


def ladder(grids, y, label):
    feats = {
        "pixel hist (k=1)":  [f_hist(g) for g in grids],
        "pixel pairs (k=2)": [f_pairs(g) for g in grids],
        "pixel 2x2 patch":   [f_patch(g, 2, 64) for g in grids],
        "pixel 3x3 patch":   [f_patch(g, 3, 128) for g in grids],
        "OBJECT order-1":    [f_obj(g) for g in grids],
    }
    r = {name: own_r2(X, y) for name, X in feats.items()}
    print(f"  {label}:")
    for name, v in r.items():
        print(f"    {name:18s} own-R2 = {v:.3f}")
    return r


def main():
    rng = np.random.default_rng(SEED)
    print(f"[FC-4 sigma-split] G={G} TOTAL={TOTAL} N={N} seed={SEED}", flush=True)

    # ---- L1 primary: largest-component area (abstraction-style object rule) ----
    grids = [place_rects(sample_partition(rng), rng) for _ in range(N)]
    y1 = np.array([max(a for a, *_ in components(g)) for g in grids], dtype=float)
    print(f"L1 latent = largest-component area (range {y1.min():.0f}-{y1.max():.0f}, "
          f"std {y1.std():.2f}); total colored area fixed = {TOTAL}")
    r1 = ladder(grids, y1, "L1 dissection")
    raw = own_r2([g.ravel() for g in grids], y1)
    print(f"    raw-grid MLP (diag) own-R2 = {raw:.3f}   (trained encoder, outside the bounded ladder)")
    perm = own_r2([f_obj(g) for g in grids], np.random.default_rng(7).permutation(y1))
    print(f"    trivial-FAIL (shuffled y)  = {perm:.3f}")
    best_px = max(v for k, v in r1.items() if k.startswith("pixel"))
    split = r1["OBJECT order-1"] >= 0.80 and (r1["OBJECT order-1"] - best_px) >= 0.30
    print(f"  ** SPLIT gate: obj={r1['OBJECT order-1']:.3f}  best-bounded-pixel={best_px:.3f}  "
          f"gap={(r1['OBJECT order-1'] - best_px):.3f}  -> {split} **")

    # ---- APPARATUS control: pixel-easy latent on the SAME grids ----
    y2 = np.array([f_hist(g)[1] for g in grids], dtype=float)      # count of color-1 pixels
    rh = own_r2([f_hist(g) for g in grids], y2)
    print(f"APPARATUS control (latent = color-1 pixel count): hist own-R2 = {rh:.3f}  "
          f"(>=0.90 -> pixel ladder is live) -> {rh >= 0.90}")

    # ---- REVERSE control v2: color-5 dot count, isolated from the object vocabulary ----
    # m in {6,7} rects (top-6 rows = rects only) + 0-6 NUISANCE color-6 dots (confound n_comps).
    def sprinkle(g, nd, color):
        placed = 0
        for _ in range(300):
            if placed == nd:
                break
            r, c = rng.integers(1, G - 1), rng.integers(1, G - 1)
            if g[r - 1:r + 2, c - 1:c + 2].max() == 0:
                g[r, c] = color; placed += 1
        return placed

    dg, yd = [], []
    for _ in range(N):
        g = place_rects(sample_partition(rng, lo=6, hi=7), rng).copy()
        sprinkle(g, int(rng.integers(0, 7)), 6)           # nuisance species first
        yd.append(sprinkle(g, int(rng.integers(0, 7)), 5))
        dg.append(g)
    yd = np.array(yd, dtype=float)
    rev_h = own_r2([f_hist(g) for g in dg], yd)
    rev_o = own_r2([f_obj(g) for g in dg], yd)
    rev = rev_h >= 0.90 and (rev_h - rev_o) >= 0.20
    print(f"REVERSE control v2 (latent = color-5 dot count; m=6-7 rects + color-6 nuisance dots): "
          f"hist={rev_h:.3f}  obj={rev_o:.3f}  gap={(rev_h - rev_o):.3f}  -> {rev}")

    verdict = ("SPLIT CONFIRMED: the abstraction-style latent is order-1 in the OBJECT filtration and "
               "high-order in the bounded PIXEL filtration, the apparatus is live, and the reverse "
               "control shows the split is PER-FILTRATION (sigma is a schema, not 'objects always win') "
               "-- the GEN-1 class bet passes its cheapest falsification."
               if split and rh >= 0.90 and rev else "CHECK (a gate failed)")
    print(f"\nFC-4: SPLIT={split} APPARATUS={rh >= 0.90} REVERSE={rev} => {verdict}")


if __name__ == "__main__":
    main()
