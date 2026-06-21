#!/usr/bin/env python
"""BoxSEL Phase 6b - general trace schema for the post-Phase-7 detector redesign.

This module is a schema scaffold, not a detector and not a held-out run.  It exists because the
Phase-7 bounded null showed that the Phase-6 guard was too Helly-shaped: it saw boundary symptoms
in the Helly variants but accepted stable PMP-shaped false closures.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
from typing import Iterable


SCHEMA_VERSION = "phase6b_general_trace_schema_v1"
DETECTOR_STATUS = "NOT_BUILT"
HELDOUT_STATUS = "NOT_REGISTERED"

PHASE7_FAILURE_CLASS = "stable_low_loss_false_closure"
PHASE7_SEEN_CASE_IDS = (
    "helly-00",
    "helly-01",
    "helly-02",
    "helly-03",
    "helly-04",
    "helly-05",
    "pmp-00",
    "pmp-01",
    "pmp-02",
    "pmp-03",
    "narrow-00",
    "narrow-01",
    "narrow-02",
    "narrow-03",
    "loss-00",
    "loss-01",
)

PHASE7_PMP_PARAMETERS = {
    "pmp-00": (Fraction(1, 3), Fraction(2, 3)),
    "pmp-01": (Fraction(2, 5), Fraction(3, 5)),
    "pmp-02": (Fraction(1, 2), Fraction(1, 2)),
    "pmp-03": (Fraction(3, 5), Fraction(2, 5)),
}

PHASE7_POINT_CONTROL_VALUES = {
    "narrow-00": Fraction(13, 20),
    "narrow-01": Fraction(7, 10),
    "narrow-02": Fraction(3, 4),
    "narrow-03": Fraction(4, 5),
    "loss-00": Fraction(3, 5),
    "loss-01": Fraction(2, 3),
}

FORBIDDEN_FEATURE_TOKENS = (
    "exact",
    "oracle",
    "ibox",
    "i_box",
    "istar",
    "i_star",
    "label",
    "phase4",
)


@dataclass(frozen=True)
class EndpointObservation:
    """One optimizer/restart observation of the sampled query endpoint."""

    index: int
    lower: float
    upper: float
    loss: float
    pressure: float = 0.0


@dataclass(frozen=True)
class ConstraintTrace:
    """Ontology-level trace for one conditional or marginal constraint."""

    name: str
    lower_target: float
    upper_target: float
    observed: float
    condition_mass: float
    numerator_mass: float

    @property
    def lower_slack(self) -> float:
        return self.observed - self.lower_target

    @property
    def upper_slack(self) -> float:
        return self.upper_target - self.observed

    @property
    def min_abs_slack(self) -> float:
        return min(abs(self.lower_slack), abs(self.upper_slack))

    @property
    def violation(self) -> float:
        return max(0.0, self.lower_target - self.observed, self.observed - self.upper_target)


@dataclass(frozen=True)
class SupportTrace:
    """Observed support geometry for the query condition and numerator."""

    condition_mass: float
    numerator_mass: float
    atom_support_min: float
    meet_support_min: float

    @property
    def conditional_value(self) -> float:
        if self.condition_mass <= 0.0:
            return math.inf
        return self.numerator_mass / self.condition_mass


@dataclass(frozen=True)
class GeneralTrace:
    """Trace-only envelope for one case, one dimension, and one optimizer mode."""

    case_id: str
    family: str
    seed: int
    dimension: int
    optimizer_mode: str
    endpoints: tuple[EndpointObservation, ...]
    constraints: tuple[ConstraintTrace, ...]
    support: SupportTrace

    @property
    def sample_interval(self) -> tuple[float, float]:
        if not self.endpoints:
            raise ValueError("trace has no endpoint observations")
        return min(item.lower for item in self.endpoints), max(item.upper for item in self.endpoints)

    @property
    def max_loss(self) -> float:
        if not self.endpoints:
            return math.inf
        return max(item.loss for item in self.endpoints)

    @property
    def min_constraint_slack(self) -> float:
        if not self.constraints:
            return math.inf
        return min(item.min_abs_slack for item in self.constraints)

    @property
    def max_constraint_violation(self) -> float:
        if not self.constraints:
            return 0.0
        return max(item.violation for item in self.constraints)


@dataclass(frozen=True)
class GeneralTraceFeatures:
    """Feature vector allowed for a future v2 guard.

    The field names are intentionally free of oracle/exact-label vocabulary; exact inference belongs
    only to a later evaluator.
    """

    sample_lower: float
    sample_upper: float
    sample_width: float
    early_lower_drop: float
    late_lower_drop: float
    max_loss: float
    max_constraint_violation: float
    min_constraint_slack: float
    condition_mass_floor: float
    numerator_mass_floor: float
    support_floor: float
    pressure_low_shift: float
    seed_low_range: float
    dimension_low_spread: float
    optimizer_low_spread: float


def _lower_drop(endpoints: tuple[EndpointObservation, ...], fraction: float) -> float:
    if not endpoints:
        return 0.0
    running = []
    lo = math.inf
    for item in endpoints:
        lo = min(lo, item.lower)
        running.append(lo)
    index = max(1, min(len(running), math.ceil(len(running) * fraction))) - 1
    return running[index] - running[-1]


def _low_range(traces: Iterable[GeneralTrace]) -> float:
    lows = [trace.sample_interval[0] for trace in traces]
    if not lows:
        return 0.0
    return max(lows) - min(lows)


def _values_between(lo: float, hi: float, count: int) -> tuple[float, ...]:
    if count <= 0:
        return ()
    if count == 1 or abs(hi - lo) < 1e-15:
        return (lo,) * count
    return tuple(lo + (hi - lo) * (i / (count - 1)) for i in range(count))


def _phase3_trace_constraints(trace) -> tuple[ConstraintTrace, ...]:
    import boxsel_phase3_restart_sampler as sampler

    atom_names = ("A", "B", "C")
    atom_constraints = tuple(
        ConstraintTrace(
            name=f"atom_{name}_volume",
            lower_target=sampler.ATOM_TARGET,
            upper_target=sampler.ATOM_TARGET,
            observed=volume,
            condition_mass=1.0,
            numerator_mass=volume,
        )
        for name, volume in zip(atom_names, trace.atom_volumes)
    )
    pair_specs = (
        ("pair_AB_given_A", trace.atom_volumes[0], trace.pair_overlaps[0]),
        ("pair_AC_given_A", trace.atom_volumes[0], trace.pair_overlaps[1]),
        ("pair_BC_given_B", trace.atom_volumes[1], trace.pair_overlaps[2]),
    )
    pair_constraints = tuple(
        ConstraintTrace(
            name=name,
            lower_target=sampler.PAIR_TARGET / sampler.ATOM_TARGET,
            upper_target=1.0,
            observed=(pair / condition_mass) if condition_mass > 0.0 else math.inf,
            condition_mass=condition_mass,
            numerator_mass=pair,
        )
        for name, condition_mass, pair in pair_specs
    )
    return atom_constraints + pair_constraints


def phase3_report_to_general_trace(
    report,
    *,
    case_id: str = "phase3-helly",
    family: str = "helly_threshold_variants",
    optimizer_mode: str = "ordinary_restart",
) -> GeneralTrace:
    """Convert a Phase-3 Helly sampler report into the Phase-6b trace schema."""

    if not report.traces:
        raise ValueError("cannot convert an empty Phase-3 report")
    lower_trace = min(report.traces, key=lambda item: item.q)
    condition_mass = lower_trace.pair_overlaps[0]
    numerator_mass = lower_trace.q * condition_mass
    support = SupportTrace(
        condition_mass=condition_mass,
        numerator_mass=numerator_mass,
        atom_support_min=min(lower_trace.atom_volumes),
        meet_support_min=min(lower_trace.pair_overlaps),
    )
    endpoints = tuple(
        EndpointObservation(index=trace.restart, lower=trace.q, upper=trace.q, loss=trace.loss)
        for trace in report.traces
    )
    return GeneralTrace(
        case_id=case_id,
        family=family,
        seed=report.seed,
        dimension=report.dim,
        optimizer_mode=optimizer_mode,
        endpoints=endpoints,
        constraints=_phase3_trace_constraints(lower_trace),
        support=support,
    )


def _pmp_constraints(q1: float, q2: float) -> tuple[ConstraintTrace, ...]:
    return (
        ConstraintTrace(
            name="pmp_first_link_A_given_Q1",
            lower_target=q1,
            upper_target=q1,
            observed=q1,
            condition_mass=1.0,
            numerator_mass=q1,
        ),
        ConstraintTrace(
            name="pmp_second_link_Q2_given_A_and_Q1",
            lower_target=q2,
            upper_target=q2,
            observed=q2,
            condition_mass=q1,
            numerator_mass=q1 * q2,
        ),
    )


def pmp_general_trace(
    *,
    case_id: str,
    family: str,
    seed: int,
    q1: Fraction | float,
    q2: Fraction | float,
    q_values: tuple[float, ...],
    optimizer_mode: str = "ordinary_restart",
    pressure: float = 0.0,
    max_loss: float = 0.0,
) -> GeneralTrace:
    """Build a trace-only PMP-shaped record from observed query values."""

    if not q_values:
        raise ValueError("PMP trace needs at least one query value")
    q1f = float(q1)
    q2f = float(q2)
    lo = min(q_values)
    support = SupportTrace(
        condition_mass=1.0,
        numerator_mass=lo,
        atom_support_min=q1f,
        meet_support_min=q1f,
    )
    endpoints = tuple(
        EndpointObservation(index=i, lower=q, upper=q, loss=max_loss, pressure=pressure)
        for i, q in enumerate(q_values)
    )
    return GeneralTrace(
        case_id=case_id,
        family=family,
        seed=seed,
        dimension=2,
        optimizer_mode=optimizer_mode,
        endpoints=endpoints,
        constraints=_pmp_constraints(q1f, q2f),
        support=support,
    )


def point_condition_general_trace(
    *,
    case_id: str,
    family: str,
    seed: int,
    value: float,
    q_values: tuple[float, ...],
    optimizer_mode: str = "ordinary_restart",
    max_loss: float = 0.0,
) -> GeneralTrace:
    """Build a simple point-conditional control trace."""

    if not q_values:
        raise ValueError("point control trace needs at least one query value")
    lo = min(q_values)
    support = SupportTrace(
        condition_mass=1.0,
        numerator_mass=lo,
        atom_support_min=max(0.0, min(1.0, value)),
        meet_support_min=lo,
    )
    endpoints = tuple(
        EndpointObservation(index=i, lower=q, upper=q, loss=max_loss)
        for i, q in enumerate(q_values)
    )
    return GeneralTrace(
        case_id=case_id,
        family=family,
        seed=seed,
        dimension=2,
        optimizer_mode=optimizer_mode,
        endpoints=endpoints,
        constraints=(
            ConstraintTrace(
                name="point_conditional",
                lower_target=value,
                upper_target=value,
                observed=value,
                condition_mass=1.0,
                numerator_mass=value,
            ),
        ),
        support=support,
    )


def phase7_case_to_general_trace(case) -> GeneralTrace:
    """Convert a Phase-7 result row into the Phase-6b trace schema.

    Helly rows are rebuilt from the actual Phase-3 sampler seed.  Synthetic Phase-7 rows are
    represented by their observed sample interval, loss, and known diagnostic family shape.
    """

    if case.family == "helly_threshold_variants":
        import boxsel_phase3_restart_sampler as sampler
        import boxsel_phase7_run as phase7

        report = sampler.ordinary_restart_report(dim=2, restarts=phase7.MAIN_RESTARTS, seed=case.seed)
        return phase3_report_to_general_trace(report, case_id=case.case_id, family=case.family)

    q_values = _values_between(case.sample_lower, case.sample_upper, 24)
    if case.family == "pmp_interval_chain_variants":
        q1, q2 = PHASE7_PMP_PARAMETERS[case.case_id]
        return pmp_general_trace(
            case_id=case.case_id,
            family=case.family,
            seed=case.seed,
            q1=q1,
            q2=q2,
            q_values=q_values,
            max_loss=case.max_loss,
        )

    if case.family in {"true_narrow_controls", "loss_escape_controls"}:
        return point_condition_general_trace(
            case_id=case.case_id,
            family=case.family,
            seed=case.seed,
            value=float(PHASE7_POINT_CONTROL_VALUES[case.case_id]),
            q_values=q_values,
            max_loss=case.max_loss,
        )

    raise ValueError(f"unknown Phase-7 family: {case.family}")


def phase7_general_traces() -> tuple[GeneralTrace, ...]:
    """Convert all seen Phase-7 result rows into GeneralTrace diagnostics."""

    import boxsel_phase7_run as phase7

    return tuple(phase7_case_to_general_trace(case) for case in phase7.run_cases())


def pmp_query_pressure_trace(
    *,
    case_id: str,
    seed: int,
    q1: Fraction | float,
    q2: Fraction | float,
    ordinary_lower: float,
    steps: int = 12,
    pressure: float = 1.0,
) -> GeneralTrace:
    """Produce a query-pressure trace for the stable PMP failure shape.

    The pressure path is constructive: it moves from the ordinary sampled lower endpoint toward the
    low-query PMP witness implied by the two premises.  It is not an evaluator label and does not
    read any exact-oracle result object.
    """

    if steps <= 0:
        raise ValueError("pressure trace needs at least one step")
    q1f = float(q1)
    q2f = float(q2)
    pressure_low = q1f * q2f
    if ordinary_lower <= pressure_low:
        values = (ordinary_lower,) * steps
    else:
        values = tuple(
            ordinary_lower - (ordinary_lower - pressure_low) * ((i + 1) / steps)
            for i in range(steps)
        )
    return pmp_general_trace(
        case_id=f"{case_id}-query-pressure",
        family="stable_pmp_false_closure_pressure",
        seed=seed,
        q1=q1f,
        q2=q2f,
        q_values=values,
        optimizer_mode="query_pressure",
        pressure=pressure,
    )


def phase7_pmp_pressure_traces() -> tuple[GeneralTrace, ...]:
    """Produce query-pressure diagnostics for the seen Phase-7 PMP failures."""

    out = []
    for trace in phase7_general_traces():
        if trace.case_id in PHASE7_PMP_PARAMETERS:
            q1, q2 = PHASE7_PMP_PARAMETERS[trace.case_id]
            out.append(
                pmp_query_pressure_trace(
                    case_id=trace.case_id,
                    seed=trace.seed,
                    q1=q1,
                    q2=q2,
                    ordinary_lower=trace.sample_interval[0],
                )
            )
    return tuple(out)


def pressure_low_shift(base: GeneralTrace, pressure_traces: tuple[GeneralTrace, ...] = ()) -> float:
    """How far query-conditioned pressure can move the sampled lower endpoint."""
    if not pressure_traces:
        return 0.0
    base_low = base.sample_interval[0]
    pressured_low = min(trace.sample_interval[0] for trace in pressure_traces)
    return base_low - pressured_low


def trace_features(
    trace: GeneralTrace,
    *,
    seed_traces: tuple[GeneralTrace, ...] = (),
    dimension_traces: tuple[GeneralTrace, ...] = (),
    optimizer_traces: tuple[GeneralTrace, ...] = (),
    pressure_traces: tuple[GeneralTrace, ...] = (),
) -> GeneralTraceFeatures:
    lo, hi = trace.sample_interval
    support_floor = min(
        trace.support.condition_mass,
        trace.support.numerator_mass,
        trace.support.atom_support_min,
        trace.support.meet_support_min,
    )
    return GeneralTraceFeatures(
        sample_lower=lo,
        sample_upper=hi,
        sample_width=hi - lo,
        early_lower_drop=_lower_drop(trace.endpoints, 0.125),
        late_lower_drop=_lower_drop(trace.endpoints, 0.75),
        max_loss=trace.max_loss,
        max_constraint_violation=trace.max_constraint_violation,
        min_constraint_slack=trace.min_constraint_slack,
        condition_mass_floor=trace.support.condition_mass,
        numerator_mass_floor=trace.support.numerator_mass,
        support_floor=support_floor,
        pressure_low_shift=pressure_low_shift(trace, pressure_traces),
        seed_low_range=_low_range(seed_traces),
        dimension_low_spread=_low_range(dimension_traces),
        optimizer_low_spread=_low_range(optimizer_traces),
    )


def feature_names() -> tuple[str, ...]:
    return tuple(GeneralTraceFeatures.__dataclass_fields__)


def feature_names_are_oracle_free() -> bool:
    names = tuple(name.lower() for name in feature_names())
    return all(not any(token in name for token in FORBIDDEN_FEATURE_TOKENS) for name in names)


def phase7_cases_are_seen(case_ids: Iterable[str]) -> bool:
    """True iff a proposed diagnostic/training set contains no unknown Phase-7 result rows."""
    return set(case_ids).issubset(set(PHASE7_SEEN_CASE_IDS))


def stable_pmp_failure_trace() -> GeneralTrace:
    """A minimal diagnostic trace shaped like the Phase-7 PMP failures."""
    endpoints = tuple(
        EndpointObservation(index=i, lower=0.397 + ((i % 3) - 1) * 0.001, upper=0.403, loss=0.0)
        for i in range(24)
    )
    constraints = (
        ConstraintTrace("pmp_first_link", 0.5, 0.5, 0.5, condition_mass=1.0, numerator_mass=0.5),
        ConstraintTrace("pmp_second_link", 0.5, 0.5, 0.5, condition_mass=0.5, numerator_mass=0.25),
    )
    support = SupportTrace(
        condition_mass=0.5,
        numerator_mass=0.20,
        atom_support_min=0.5,
        meet_support_min=0.25,
    )
    return GeneralTrace(
        case_id="pmp-diagnostic",
        family="stable_pmp_false_closure_diagnostic",
        seed=0,
        dimension=2,
        optimizer_mode="ordinary_restart",
        endpoints=endpoints,
        constraints=constraints,
        support=support,
    )


def schema_summary() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "detector_status": DETECTOR_STATUS,
        "heldout_status": HELDOUT_STATUS,
        "phase7_failure_class": PHASE7_FAILURE_CLASS,
        "phase7_seen_cases": len(PHASE7_SEEN_CASE_IDS),
        "feature_count": len(feature_names()),
        "oracle_free_feature_names": feature_names_are_oracle_free(),
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 6b trace schema:", schema_summary())
