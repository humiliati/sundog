#!/usr/bin/env python
"""BoxSEL Phase 3 - ordinary restart sampler for the Helly-seed query.

This is the observed nesting layer:

    I_sample^{n,N} subset I_box^n subset I*

Phase 4 closed the exact box-attainable lower endpoint for the Helly seed:

    inf I_box^1 = 1/2
    inf I_box^n = (9 + sqrt(17))/32  for n >= 2.

This module deliberately does NOT use query pressure. It samples zero-loss feasible box embeddings
from ordinary random restarts for the seed ontology:

    |A|=|B|=|C|=1/2,
    |A&B|, |A&C|, |B&C| >= 1/4,
    query q = P(C | A&B).

The resulting `I_sample` is a baseline for the search gap: even when every sampled run has zero
ontology loss, the sampled lower endpoint can sit well above the exact I_box lower endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, Mapping

import boxsel_phase4_interval_gap as gap
import boxsel_phase4k_dimension_compression as closed


ATOM_TARGET = 0.5
PAIR_TARGET = 0.25
DEFAULT_TOLERANCE = 1e-12
EXACT_I_STAR = (0.0, 1.0)


FloatInterval = tuple[float, float]
FloatBox = tuple[FloatInterval, ...]
FloatEmbedding = Mapping[str, FloatBox]


@dataclass(frozen=True)
class RestartTrace:
    restart: int
    attempts: int
    loss: float
    q: float
    atom_volumes: tuple[float, float, float]
    pair_overlaps: tuple[float, float, float]

    @property
    def min_pairwise_slack(self) -> float:
        return min(p - PAIR_TARGET for p in self.pair_overlaps)


@dataclass(frozen=True)
class SampleReport:
    dim: int
    restarts: int
    seed: int
    tolerance: float
    traces: tuple[RestartTrace, ...]
    exact_interval: tuple[float, float] = EXACT_I_STAR

    @property
    def accepted(self) -> int:
        return len(self.traces)

    @property
    def sample_interval(self) -> tuple[float, float]:
        if not self.traces:
            raise ValueError("no accepted traces")
        qs = [t.q for t in self.traces]
        return min(qs), max(qs)

    @property
    def exact_box_interval(self) -> tuple[float, float]:
        lower = 0.5 if self.dim == 1 else float(closed.exact_global_infimum())
        return lower, 1.0

    @property
    def lower_search_gap(self) -> float:
        return self.sample_interval[0] - self.exact_box_interval[0]

    @property
    def exact_lower_gap(self) -> float:
        return self.sample_interval[0] - self.exact_interval[0]

    @property
    def max_loss(self) -> float:
        return max(t.loss for t in self.traces)

    @property
    def min_slack(self) -> float:
        return min(t.min_pairwise_slack for t in self.traces)


def _volume(box: FloatBox) -> float:
    out = 1.0
    for lo, hi in box:
        out *= hi - lo
    return out


def _meet_volume(boxes: Iterable[FloatBox]) -> float:
    boxes = tuple(boxes)
    out = 1.0
    for axis in range(len(boxes[0])):
        lo = max(box[axis][0] for box in boxes)
        hi = min(box[axis][1] for box in boxes)
        if hi <= lo:
            return 0.0
        out *= hi - lo
    return out


def _random_lengths_product_half(dim: int, rng: random.Random, sigma: float = 0.35) -> tuple[float, ...]:
    """Draw side lengths in (0,1] whose product is exactly 1/2 up to float rounding."""
    target_log = math.log(ATOM_TARGET)
    for _ in range(1000):
        logs = [rng.gauss(target_log / dim, sigma) for _ in range(dim)]
        shift = (target_log - sum(logs)) / dim
        lengths = tuple(math.exp(x + shift) for x in logs)
        if all(0.0 < length <= 1.0 for length in lengths):
            return lengths
    raise RuntimeError(f"failed to draw valid side lengths for dim={dim}")


def random_box_volume_half(dim: int, rng: random.Random) -> FloatBox:
    lengths = _random_lengths_product_half(dim, rng)
    intervals = []
    for length in lengths:
        lo = rng.random() * (1.0 - length)
        intervals.append((lo, lo + length))
    return tuple(intervals)


def random_embedding(dim: int, rng: random.Random) -> dict[str, FloatBox]:
    return {name: random_box_volume_half(dim, rng) for name in ("A", "B", "C")}


def query_value(embedding: FloatEmbedding) -> float:
    denominator = _meet_volume((embedding["A"], embedding["B"]))
    if denominator <= 0.0:
        return math.inf
    return _meet_volume((embedding["A"], embedding["B"], embedding["C"])) / denominator


def seed_loss(embedding: FloatEmbedding) -> tuple[float, tuple[float, float, float], tuple[float, float, float]]:
    atoms = (
        _volume(embedding["A"]),
        _volume(embedding["B"]),
        _volume(embedding["C"]),
    )
    pairs = (
        _meet_volume((embedding["A"], embedding["B"])),
        _meet_volume((embedding["A"], embedding["C"])),
        _meet_volume((embedding["B"], embedding["C"])),
    )
    atom_loss = sum((vol - ATOM_TARGET) ** 2 for vol in atoms)
    pair_loss = sum(max(0.0, PAIR_TARGET - overlap) ** 2 for overlap in pairs)
    return atom_loss + pair_loss, atoms, pairs


def zero_loss_restart(
    dim: int,
    restart: int,
    rng: random.Random,
    tolerance: float = DEFAULT_TOLERANCE,
    max_attempts: int = 10000,
) -> RestartTrace:
    """Run one ordinary loss-only restart by rejection sampling feasible zero-loss embeddings."""
    for attempts in range(1, max_attempts + 1):
        embedding = random_embedding(dim, rng)
        loss, atoms, pairs = seed_loss(embedding)
        if loss <= tolerance:
            return RestartTrace(
                restart=restart,
                attempts=attempts,
                loss=loss,
                q=query_value(embedding),
                atom_volumes=atoms,
                pair_overlaps=pairs,
            )
    raise RuntimeError(f"restart {restart} failed to find a zero-loss embedding in {max_attempts} attempts")


def ordinary_restart_report(
    dim: int = 2,
    restarts: int = 128,
    seed: int = 314159,
    tolerance: float = DEFAULT_TOLERANCE,
) -> SampleReport:
    rng = random.Random(seed)
    traces = tuple(zero_loss_restart(dim, i, rng, tolerance=tolerance) for i in range(restarts))
    return SampleReport(dim=dim, restarts=restarts, seed=seed, tolerance=tolerance, traces=traces)


def cumulative_endpoint_trace(report: SampleReport) -> tuple[tuple[int, float, float], ...]:
    lo = math.inf
    hi = -math.inf
    out = []
    for trace in report.traces:
        lo = min(lo, trace.q)
        hi = max(hi, trace.q)
        out.append((trace.restart + 1, lo, hi))
    return tuple(out)


def seed_variance_report(
    dim: int = 2,
    restarts: int = 64,
    seeds: tuple[int, ...] = (11, 23, 37, 53),
) -> tuple[SampleReport, ...]:
    return tuple(ordinary_restart_report(dim=dim, restarts=restarts, seed=seed) for seed in seeds)


def sampler_summary(report: SampleReport) -> dict[str, float | int | tuple[float, float]]:
    return {
        "dim": report.dim,
        "restarts": report.restarts,
        "accepted": report.accepted,
        "sample_interval": report.sample_interval,
        "exact_box_interval": report.exact_box_interval,
        "exact_interval": report.exact_interval,
        "lower_search_gap": report.lower_search_gap,
        "max_loss": report.max_loss,
        "min_slack": report.min_slack,
    }


def rational_witness_summary() -> dict[str, float]:
    """Keep the sampler anchored to the exact Phase-4 endpoint witness."""
    witness = gap.rational_boxn_shrink_witness(2)
    return {
        "rational_witness_q": float(gap.query_value(witness)),
        "exact_box_lower": float(closed.exact_global_infimum()),
        "rational_witness_gap_above_exact": float(gap.query_value(witness)) - float(closed.exact_global_infimum()),
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    for dim in (2, 3):
        report = ordinary_restart_report(dim=dim, restarts=128, seed=314159 + dim)
        print(f"dim={dim}:", sampler_summary(report))
    print("seed variance lows:", [r.sample_interval[0] for r in seed_variance_report()])
    print("exact anchor:", rational_witness_summary())
