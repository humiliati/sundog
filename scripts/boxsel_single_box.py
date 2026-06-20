#!/usr/bin/env python
"""BoxSEL Phase 2 - single-box realizability probes (representation-gap seeds).

Separated from the exact-oracle alignment (`boxsel_exact_oracle.py`). This asks a narrower,
purely geometric question: which tiny type-volume models are realizable by a SINGLE
axis-parallel box per atom -- BoxSEL's actual body -- as opposed to arbitrary nonnegative type
weights, which is what the exact oracle ranges over?

KEY FACT (axis-parallel Helly, Helly number 2, dimension-independent). If three axis-parallel
boxes have all three PAIRWISE intersections of positive volume, then their TRIPLE intersection
also has positive volume. Per axis this is 1-D Helly for positive-length overlaps: a box
overlaps another with positive volume iff their intervals overlap with positive length on every
axis, and for three intervals the triple overlap length is `min(highs) - max(lows)` = the
overlap of the largest-low interval with the smallest-high interval, itself one of the pairwise
overlaps, hence positive. Taking the product over axes gives positive triple volume.

CONSEQUENCE (the representation-gap seed). The type-volume model with positive mass on the
pairwise cells {A,B}, {B,C}, {A,C} and ZERO mass on the triple cell {A,B,C} is a perfectly
valid SEL type-volume model (the exact oracle admits it as a finite/volume model), yet NO
single-box embedding can realize it. So zero-loss box embeddings (I_box) are a STRICT subset of
all SEL models (I*) in general. This is realizability only; the query-interval manifestation
(I_box ( I* on a specific P(D|C)) is the Phase-4 target, seeded here.

This module is exact (Fractions) and deterministic. It is NOT a single-box optimizer.
"""
from fractions import Fraction
from itertools import combinations, product
from random import Random
from typing import Mapping, Sequence


def _frac(value) -> Fraction:
    if isinstance(value, Fraction):
        return value
    if isinstance(value, float):
        return Fraction(str(value))
    return Fraction(value)


def make_box(*intervals) -> tuple:
    """An axis-parallel box as a tuple of (lo, hi) intervals, one per axis (Fractions)."""
    box = tuple((_frac(lo), _frac(hi)) for lo, hi in intervals)
    for lo, hi in box:
        if lo > hi:
            raise ValueError(f"interval lo > hi: ({lo}, {hi})")
    return box


def interval_meet(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo <= hi else None


def box_meet(boxes: Sequence[tuple]):
    """Intersection box (per axis), or None if empty on any axis."""
    if not boxes:
        return None
    dim = len(boxes[0])
    out = []
    for k in range(dim):
        cur = boxes[0][k]
        for bx in boxes[1:]:
            cur = interval_meet(cur, bx[k])
            if cur is None:
                return None
        out.append(cur)
    return tuple(out)


def box_volume(box) -> Fraction:
    vol = Fraction(1)
    for lo, hi in box:
        vol *= (hi - lo)
    return vol


def meet_volume(boxes: Sequence[tuple]) -> Fraction:
    m = box_meet(boxes)
    return Fraction(0) if m is None else box_volume(m)


def type_volumes(embedding: Mapping[str, tuple], atoms: Sequence[str], ambient: tuple) -> dict:
    """Exact volume of every Boolean type (Venn cell) induced by the box embedding inside `ambient`.

    Each atom maps to an axis-parallel box of the same dimension as `ambient`. Returns a dict
    frozenset(atoms-present) -> Fraction volume. The empty type is the ambient region outside all
    boxes. Computed by partitioning each axis at the (clipped) box endpoints into elementary
    intervals; every elementary cell lies inside a definite set of boxes.
    """
    atoms = tuple(atoms)
    dim = len(ambient)
    elem_axes = []
    for k in range(dim):
        alo, ahi = ambient[k]
        pts = {alo, ahi}
        for a in atoms:
            for p in embedding[a][k]:
                if alo <= p <= ahi:
                    pts.add(p)
        bp = sorted(pts)
        elem_axes.append([(bp[i], bp[i + 1]) for i in range(len(bp) - 1) if bp[i] < bp[i + 1]])

    vols = {frozenset(combo): Fraction(0)
            for r in range(len(atoms) + 1) for combo in combinations(atoms, r)}
    for cell in product(*elem_axes):
        mids = tuple((lo + hi) / 2 for lo, hi in cell)
        vol = Fraction(1)
        for lo, hi in cell:
            vol *= (hi - lo)
        present = frozenset(
            a for a in atoms
            if all(embedding[a][k][0] <= mids[k] <= embedding[a][k][1] for k in range(dim))
        )
        vols[present] += vol
    return vols


def triple_overlap_forced(boxes: Sequence[tuple]):
    """Axis-parallel Helly-2 witness on a concrete triple.

    Returns True if all three pairwise meets have positive volume AND the triple meet does too
    (the law holds), False if the law is VIOLATED (would refute Helly-2), or None if the
    hypothesis (all pairwise positive) is not met.
    """
    a, b, c = boxes
    if meet_volume([a, b]) > 0 and meet_volume([b, c]) > 0 and meet_volume([a, c]) > 0:
        return meet_volume([a, b, c]) > 0
    return None


FORBIDDEN_ATOMS = ("A", "B", "C")


def forbidden_pattern_weights() -> dict:
    """The unrealizable target: positive on every pairwise cell, ZERO on the triple cell."""
    return {
        frozenset({"A", "B"}): Fraction(1),
        frozenset({"B", "C"}): Fraction(1),
        frozenset({"A", "C"}): Fraction(1),
        frozenset({"A", "B", "C"}): Fraction(0),
    }


def random_box(rng: Random, dim: int, span: int = 4) -> tuple:
    axes = []
    for _ in range(dim):
        lo = rng.randint(0, span - 1)
        hi = rng.randint(lo + 1, span)
        axes.append((Fraction(lo), Fraction(hi)))
    return tuple(axes)


def helly_battery(samples: int = 4000, dim: int = 2, seed: int = 240711821) -> tuple:
    """Sample random axis-parallel box triples; return (tested, violations).

    `tested` counts triples with all pairwise overlaps positive (the non-vacuous hypothesis);
    `violations` counts any with empty triple overlap (would refute Helly-2 -- must stay 0).
    """
    rng = Random(seed)
    tested = violations = 0
    for _ in range(samples):
        boxes = [random_box(rng, dim) for _ in range(3)]
        verdict = triple_overlap_forced(boxes)
        if verdict is None:
            continue
        tested += 1
        if verdict is False:
            violations += 1
    return tested, violations


def forbidden_pattern_battery(samples: int = 3000, dim: int = 2, span: int = 4, seed: int = 240711821) -> tuple:
    """Sample random 3-atom box embeddings; confirm the forbidden Venn never occurs.

    `tested` counts embeddings whose three exclusive pairwise cells {A,B}, {B,C}, {A,C} are all
    positive (the non-vacuous hypothesis); `violations` counts those that ALSO have a zero triple
    cell {A,B,C} -- the forbidden, unrealizable pattern, which must stay 0. (Exclusive cell ( meet,
    so all pairwise cells positive => all pairwise meets positive => Helly-2 => triple cell > 0.)
    """
    rng = Random(seed)
    atoms = ("A", "B", "C")
    ambient = tuple((Fraction(0), Fraction(span)) for _ in range(dim))
    ab, bc, ac, abc = (frozenset({"A", "B"}), frozenset({"B", "C"}),
                       frozenset({"A", "C"}), frozenset({"A", "B", "C"}))
    tested = violations = 0
    for _ in range(samples):
        emb = {a: random_box(rng, dim, span) for a in atoms}
        v = type_volumes(emb, atoms, ambient)
        if v[ab] > 0 and v[bc] > 0 and v[ac] > 0:
            tested += 1
            if v[abc] == 0:
                violations += 1
    return tested, violations


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    for d in (1, 2):
        t1, v1 = helly_battery(dim=d)
        t2, v2 = forbidden_pattern_battery(dim=d)
        print(f"dim={d}: Helly-2 meets tested={t1} viol={v1}; forbidden-cell tested={t2} viol={v2}")
    print("forbidden target (pairwise cells positive, triple cell zero) is unrealizable by single boxes.")
