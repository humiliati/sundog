#!/usr/bin/env python
"""BoxSEL Phase 4 seed - turn the Helly realizability gap into query interval gaps.

This is the first measured I_box^1 vs I* witness, not the full extremal optimizer.
The ontology fixes three atomic concepts at length/probability 1/2 and requires
each pairwise co-occurrence to be at least 1/4. Query:

    q = P(C | A and B)

The exact type-volume oracle admits q=0. One-dimensional single-box embeddings
cannot: three intervals of length 1/2 whose pairwise overlaps are all at least
1/4 have centers within a radius band of width 1/4, so their triple overlap has
length at least 1/4; since |A and B| <= 1/2, q >= 1/2.

For n >= 2, this module records a certified rational construction with
q = 513/1250 < 1/2, embedded into higher dimensions by adding full axes. That is
a dimension-sweep upper bound on the box lower endpoint, not a proof of the
higher-dimensional infimum.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Mapping

import boxsel_exact_oracle as oracle
import boxsel_single_box as sb


F = Fraction
A = oracle.atom("A")
B = oracle.atom("B")
C = oracle.atom("C")
ATOMS = ("A", "B", "C")
AMBIENT_1D = sb.make_box((0, 1))


@dataclass(frozen=True)
class BoxIntervalGap:
    exact_lower: Fraction
    exact_upper: Fraction
    box1_lower: Fraction
    box1_upper: Fraction

    @property
    def lower_gap(self) -> Fraction:
        return self.box1_lower - self.exact_lower


@dataclass(frozen=True)
class DimensionSweepRow:
    dim: int
    exact_interval: tuple[Fraction, Fraction]
    certified_box_q: Fraction
    status: str

    @property
    def gap_to_exact_lower(self) -> Fraction:
        return self.certified_box_q - self.exact_interval[0]


def helly_interval_gap_ontology():
    """Ontology forcing atom sizes 1/2 and pairwise co-occurrences >= 1/4."""
    ontology = [
        oracle.conditional(A, oracle.TOP, F(1, 2)),
        oracle.conditional(B, oracle.TOP, F(1, 2)),
        oracle.conditional(C, oracle.TOP, F(1, 2)),
        oracle.conditional(A & B, oracle.TOP, F(1, 4), F(1)),
        oracle.conditional(A & C, oracle.TOP, F(1, 4), F(1)),
        oracle.conditional(B & C, oracle.TOP, F(1, 4), F(1)),
    ]
    return tuple(ontology), C, A & B


def exact_interval_gap() -> tuple[Fraction, Fraction]:
    ontology, consequent, condition = helly_interval_gap_ontology()
    return oracle.exact_interval(ontology, consequent, condition, atoms=ATOMS).interval_exact()


def box1_interval_gap() -> tuple[Fraction, Fraction]:
    """Analytic I_box^1 interval for the seed ontology and query."""
    return F(1, 2), F(1)


def ambient_box(dim: int) -> tuple:
    return sb.make_box(*([(0, 1)] * dim))


def measured_gap() -> BoxIntervalGap:
    exact_lower, exact_upper = exact_interval_gap()
    box1_lower, box1_upper = box1_interval_gap()
    return BoxIntervalGap(exact_lower, exact_upper, box1_lower, box1_upper)


def query_value(embedding: Mapping[str, tuple]) -> Fraction:
    denominator = sb.meet_volume([embedding["A"], embedding["B"]])
    if denominator == 0:
        raise ValueError("query denominator A and B has zero volume")
    numerator = sb.meet_volume([embedding["A"], embedding["B"], embedding["C"]])
    return numerator / denominator


def query_value_1d(embedding: Mapping[str, tuple]) -> Fraction:
    return query_value(embedding)


def satisfies_seed_ontology(embedding: Mapping[str, tuple]) -> bool:
    dim = len(next(iter(embedding.values())))
    ambient_vol = sb.box_volume(ambient_box(dim))
    atoms_ok = all(sb.meet_volume([embedding[name]]) / ambient_vol == F(1, 2) for name in ATOMS)
    pairwise_ok = (
        sb.meet_volume([embedding["A"], embedding["B"]]) / ambient_vol >= F(1, 4)
        and sb.meet_volume([embedding["A"], embedding["C"]]) / ambient_vol >= F(1, 4)
        and sb.meet_volume([embedding["B"], embedding["C"]]) / ambient_vol >= F(1, 4)
    )
    return atoms_ok and pairwise_ok


def satisfies_seed_ontology_1d(embedding: Mapping[str, tuple]) -> bool:
    return satisfies_seed_ontology(embedding)


def lower_box1_witness() -> dict[str, tuple]:
    """Achieves q=1/2, so the analytic lower bound is tight."""
    return {
        "A": sb.make_box((0, F(1, 2))),
        "B": sb.make_box((0, F(1, 2))),
        "C": sb.make_box((F(1, 4), F(3, 4))),
    }


def upper_box1_witness() -> dict[str, tuple]:
    """Achieves q=1."""
    return {
        "A": sb.make_box((0, F(1, 2))),
        "B": sb.make_box((0, F(1, 2))),
        "C": sb.make_box((0, F(1, 2))),
    }


def rational_box2_shrink_witness() -> dict[str, tuple]:
    """A certified 2-D satisfying witness with q = 513/1250 < 1/2.

    Family:
        A = [1-x, 1] x [1 - 1/(2x), 1]
        B = [0, 1]   x [1/2, 1]
        C = [0, z]   x [1 - 1/(2x), 1 - 1/(2x) + 1/(2z)]

    With x=25/41 and z=32/41, all atom areas are 1/2, AC=1/4,
    BC=513/2050 >= 1/4, AB=25/82, and q=ABC/AB=513/1250.
    """

    x = F(25, 41)
    z = F(32, 41)
    y0 = F(1) - F(1, 2) / x
    return {
        "A": sb.make_box((F(1) - x, F(1)), (y0, F(1))),
        "B": sb.make_box((F(0), F(1)), (F(1, 2), F(1))),
        "C": sb.make_box((F(0), z), (y0, y0 + F(1, 2) / z)),
    }


def extend_embedding_full_axes(embedding: Mapping[str, tuple], dim: int) -> dict[str, tuple]:
    """Embed a lower-dimensional witness into `dim` by appending full [0,1] axes."""
    current_dim = len(next(iter(embedding.values())))
    if dim < current_dim:
        raise ValueError("target dimension must be at least the embedding dimension")
    extra = tuple((F(0), F(1)) for _ in range(dim - current_dim))
    return {name: tuple(box) + extra for name, box in embedding.items()}


def rational_boxn_shrink_witness(dim: int) -> dict[str, tuple]:
    if dim == 1:
        return lower_box1_witness()
    return extend_embedding_full_axes(rational_box2_shrink_witness(), dim)


def dimension_sweep_candidates(max_dim: int = 4) -> tuple[DimensionSweepRow, ...]:
    exact = exact_interval_gap()
    rows = []
    for dim in range(1, max_dim + 1):
        emb = rational_boxn_shrink_witness(dim)
        q = query_value(emb)
        status = "tight_analytic" if dim == 1 else "certified_upper_bound"
        rows.append(DimensionSweepRow(dim=dim, exact_interval=exact, certified_box_q=q, status=status))
    return tuple(rows)


def exact_lower_type_model() -> oracle.WeightedTypeModel:
    """A finite/type-volume witness with q=0.

    Positive exclusive pairwise cells plus outside mass:
        AB = AC = BC = outside = 1/4, ABC = 0.
    """
    weights = {
        frozenset(): F(1, 4),
        frozenset({"A", "B"}): F(1, 4),
        frozenset({"A", "C"}): F(1, 4),
        frozenset({"B", "C"}): F(1, 4),
    }
    return oracle.WeightedTypeModel(ATOMS, tuple(weights.get(t, F(0)) for t in oracle.enumerate_types(ATOMS)))


def exact_upper_type_model() -> oracle.WeightedTypeModel:
    """A finite/type-volume witness with q=1."""
    weights = {
        frozenset(): F(1, 2),
        frozenset({"A", "B", "C"}): F(1, 2),
    }
    return oracle.WeightedTypeModel(ATOMS, tuple(weights.get(t, F(0)) for t in oracle.enumerate_types(ATOMS)))


def interval_center(box: tuple) -> Fraction:
    lo, hi = box[0]
    return (lo + hi) / 2


def analytic_box1_lower_bound(embedding: Mapping[str, tuple]) -> Fraction:
    """Return the proven lower bound for q for a satisfying 1-D seed embedding."""
    if not satisfies_seed_ontology_1d(embedding):
        raise ValueError("embedding does not satisfy the seed ontology")
    centers = [interval_center(embedding[name]) for name in ATOMS]
    diameter = max(centers) - min(centers)
    triple_lower = F(1, 2) - diameter
    denominator_upper = F(1, 2)
    return triple_lower / denominator_upper


if __name__ == "__main__":
    gap = measured_gap()
    print("exact I*       =", (gap.exact_lower, gap.exact_upper))
    print("single-box I1  =", (gap.box1_lower, gap.box1_upper))
    print("lower gap      =", gap.lower_gap)
    for row in dimension_sweep_candidates(4):
        print(f"dim={row.dim}: certified q={row.certified_box_q} ({row.status})")
