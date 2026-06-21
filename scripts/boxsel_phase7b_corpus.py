#!/usr/bin/env python
"""BoxSEL Phase 7b - deterministic corpus generator.

This builds the planned Phase-7b case objects and trace records.  It does not run a v2 detector,
does not lock thresholds, and does not emit held-out results.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import cycle

import boxsel_exact_oracle as oracle
import boxsel_phase3_restart_sampler as sampler
import boxsel_phase6b_trace_schema as schema
import boxsel_phase7b_prereg as prereg


CORPUS_GENERATOR_VERSION = "phase7b_corpus_generator_v0"
CORPUS_STATUS = "BUILT_NOT_RUN"
HELLY_RESTARTS = 64
SYNTHETIC_RESTARTS = 48


@dataclass(frozen=True)
class Phase7bCase:
    case_id: str
    family: str
    role: str
    seed: int
    atoms: tuple[str, ...]
    ontology: tuple[oracle.Conditional, ...]
    query_consequent: oracle.Concept
    query_condition: oracle.Concept
    ordinary_trace: schema.GeneralTrace
    pressure_traces: tuple[schema.GeneralTrace, ...]
    seed_traces: tuple[schema.GeneralTrace, ...]
    dimension_traces: tuple[schema.GeneralTrace, ...]
    optimizer_traces: tuple[schema.GeneralTrace, ...]
    description: str

    def exact(self) -> oracle.OracleResult:
        return oracle.exact_interval(self.ontology, self.query_consequent, self.query_condition, atoms=self.atoms)

    def features(self) -> schema.GeneralTraceFeatures:
        return schema.trace_features(
            self.ordinary_trace,
            seed_traces=self.seed_traces,
            dimension_traces=self.dimension_traces,
            optimizer_traces=self.optimizer_traces,
            pressure_traces=self.pressure_traces,
        )


def _stable_values(center: float, count: int = SYNTHETIC_RESTARTS, width: float = 0.006) -> tuple[float, ...]:
    values = []
    for index in range(count):
        offset = ((index % 7) - 3) * width / 6.0
        values.append(max(0.0, min(1.0, center + offset)))
    return tuple(values)


def _pmp_ontology(q1: Fraction, q2: Fraction) -> tuple[tuple[oracle.Conditional, ...], oracle.Concept, oracle.Concept]:
    ontology, consequent, condition = oracle.pmp_ontology(q1, q2)
    return tuple(ontology), consequent, condition


def _point_ontology(value: Fraction) -> tuple[tuple[oracle.Conditional, ...], oracle.Concept, oracle.Concept]:
    x = oracle.atom("X")
    y = oracle.atom("Y")
    return (oracle.conditional(y, x, value),), y, x


def _helly_ontology() -> tuple[tuple[oracle.Conditional, ...], oracle.Concept, oracle.Concept]:
    a = oracle.atom("A")
    b = oracle.atom("B")
    c = oracle.atom("C")
    ontology = (
        oracle.conditional(a, oracle.TOP, Fraction(1, 2)),
        oracle.conditional(b, oracle.TOP, Fraction(1, 2)),
        oracle.conditional(c, oracle.TOP, Fraction(1, 2)),
        oracle.conditional(b, a, Fraction(1, 2), Fraction(1)),
        oracle.conditional(c, a, Fraction(1, 2), Fraction(1)),
        oracle.conditional(c, b, Fraction(1, 2), Fraction(1)),
    )
    return ontology, c, a & b


def _support_floor_ontology(
    condition_mass: Fraction,
) -> tuple[tuple[oracle.Conditional, ...], oracle.Concept, oracle.Concept]:
    a = oracle.atom("A")
    b = oracle.atom("B")
    c = oracle.atom("C")
    ontology = (
        oracle.conditional(a, oracle.TOP, condition_mass),
        oracle.conditional(b, a, Fraction(1)),
    )
    return ontology, c, a & b


def _support_floor_trace(
    *,
    case_id: str,
    seed: int,
    condition_mass: Fraction,
    q_values: tuple[float, ...],
    optimizer_mode: str = "ordinary_restart",
    pressure: float = 0.0,
    max_loss: float = 0.0,
) -> schema.GeneralTrace:
    cm = float(condition_mass)
    lo = min(q_values)
    endpoints = tuple(
        schema.EndpointObservation(index=i, lower=q, upper=q, loss=max_loss, pressure=pressure)
        for i, q in enumerate(q_values)
    )
    constraints = (
        schema.ConstraintTrace(
            name="support_atom_A_volume",
            lower_target=cm,
            upper_target=cm,
            observed=cm,
            condition_mass=1.0,
            numerator_mass=cm,
        ),
        schema.ConstraintTrace(
            name="support_B_given_A",
            lower_target=1.0,
            upper_target=1.0,
            observed=1.0,
            condition_mass=cm,
            numerator_mass=cm,
        ),
    )
    support = schema.SupportTrace(
        condition_mass=cm,
        numerator_mass=lo * cm,
        atom_support_min=cm,
        meet_support_min=cm,
    )
    return schema.GeneralTrace(
        case_id=case_id,
        family="support_floor_variants",
        seed=seed,
        dimension=2,
        optimizer_mode=optimizer_mode,
        endpoints=endpoints,
        constraints=constraints,
        support=support,
    )


def _stable_pmp_case(index: int, seed: int, q1: Fraction, q2: Fraction, sample_center: float) -> Phase7bCase:
    case_id = f"p7b-stable-pmp-{index:02d}"
    ontology, consequent, condition = _pmp_ontology(q1, q2)
    ordinary = schema.pmp_general_trace(
        case_id=case_id,
        family="stable_pmp_pressure_variants",
        seed=seed,
        q1=q1,
        q2=q2,
        q_values=_stable_values(sample_center),
    )
    pressure = schema.pmp_query_pressure_trace(
        case_id=case_id,
        seed=seed,
        q1=q1,
        q2=q2,
        ordinary_lower=ordinary.sample_interval[0],
    )
    return Phase7bCase(
        case_id=case_id,
        family="stable_pmp_pressure_variants",
        role="false_closure_trap",
        seed=seed,
        atoms=("A", "Q1", "Q2"),
        ontology=ontology,
        query_consequent=consequent,
        query_condition=condition,
        ordinary_trace=ordinary,
        pressure_traces=(pressure,),
        seed_traces=(),
        dimension_traces=(),
        optimizer_traces=(ordinary, pressure),
        description="Stable PMP false-closure trap with query-pressure movement.",
    )


def _helly_case(index: int, seed: int) -> Phase7bCase:
    case_id = f"p7b-helly-{index:02d}"
    ontology, consequent, condition = _helly_ontology()
    report = sampler.ordinary_restart_report(dim=2, restarts=HELLY_RESTARTS, seed=seed)
    ordinary = schema.phase3_report_to_general_trace(
        report,
        case_id=case_id,
        family="helly_threshold_variants_v2",
    )
    return Phase7bCase(
        case_id=case_id,
        family="helly_threshold_variants_v2",
        role="false_closure_trap",
        seed=seed,
        atoms=("A", "B", "C"),
        ontology=ontology,
        query_consequent=consequent,
        query_condition=condition,
        ordinary_trace=ordinary,
        pressure_traces=(),
        seed_traces=(),
        dimension_traces=(),
        optimizer_traces=(),
        description="Fresh Helly threshold false-closure trap, not a Phase-7 seed variant.",
    )


def _support_floor_case(index: int, seed: int, condition_mass: Fraction, sample_center: float) -> Phase7bCase:
    case_id = f"p7b-support-{index:02d}"
    ontology, consequent, condition = _support_floor_ontology(condition_mass)
    ordinary = _support_floor_trace(
        case_id=case_id,
        seed=seed,
        condition_mass=condition_mass,
        q_values=_stable_values(sample_center),
    )
    pressure = _support_floor_trace(
        case_id=f"{case_id}-query-pressure",
        seed=seed,
        condition_mass=condition_mass,
        q_values=_stable_values(max(0.0, sample_center - 0.18), count=12, width=0.002),
        optimizer_mode="query_pressure",
        pressure=1.0,
    )
    return Phase7bCase(
        case_id=case_id,
        family="support_floor_variants",
        role="false_closure_trap",
        seed=seed,
        atoms=("A", "B", "C"),
        ontology=ontology,
        query_consequent=consequent,
        query_condition=condition,
        ordinary_trace=ordinary,
        pressure_traces=(pressure,),
        seed_traces=(),
        dimension_traces=(),
        optimizer_traces=(ordinary, pressure),
        description="Low-support false-closure trap with query-pressure support sensitivity.",
    )


def _point_case(
    *,
    case_id: str,
    family: str,
    role: str,
    seed: int,
    value: Fraction,
    max_loss: float = 0.0,
    with_pressure_noop: bool = False,
) -> Phase7bCase:
    ontology, consequent, condition = _point_ontology(value)
    center = float(value)
    ordinary = schema.point_condition_general_trace(
        case_id=case_id,
        family=family,
        seed=seed,
        value=center,
        q_values=_stable_values(center, width=0.001),
        max_loss=max_loss,
    )
    pressure_traces = ()
    optimizer_traces = ()
    if with_pressure_noop:
        pressure = schema.point_condition_general_trace(
            case_id=f"{case_id}-query-pressure",
            family=family,
            seed=seed,
            value=center,
            q_values=_stable_values(center, count=12, width=0.001),
            optimizer_mode="query_pressure",
            max_loss=max_loss,
        )
        pressure_traces = (pressure,)
        optimizer_traces = (ordinary, pressure)
    return Phase7bCase(
        case_id=case_id,
        family=family,
        role=role,
        seed=seed,
        atoms=("X", "Y"),
        ontology=ontology,
        query_consequent=consequent,
        query_condition=condition,
        ordinary_trace=ordinary,
        pressure_traces=pressure_traces,
        seed_traces=(),
        dimension_traces=(),
        optimizer_traces=optimizer_traces,
        description="Point-conditional control.",
    )


def generate_phase7b_corpus() -> tuple[Phase7bCase, ...]:
    """Build the deterministic Phase-7b planned corpus."""

    if not prereg.reserved_seeds_are_clean():
        raise RuntimeError("reserved held-out seeds overlap training/diagnostic seeds")
    seeds = cycle(prereg.RESERVED_HELDOUT_SEEDS)
    cases: list[Phase7bCase] = []

    pmp_specs = (
        (Fraction(1, 3), Fraction(2, 3), 0.38),
        (Fraction(2, 5), Fraction(3, 5), 0.39),
        (Fraction(1, 2), Fraction(1, 2), 0.40),
        (Fraction(3, 5), Fraction(2, 5), 0.38),
        (Fraction(3, 8), Fraction(3, 4), 0.43),
        (Fraction(4, 7), Fraction(1, 2), 0.44),
        (Fraction(5, 8), Fraction(2, 5), 0.41),
        (Fraction(2, 3), Fraction(1, 3), 0.37),
    )
    for index, (q1, q2, sample_center) in enumerate(pmp_specs):
        cases.append(_stable_pmp_case(index, next(seeds), q1, q2, sample_center))

    for index in range(4):
        cases.append(_helly_case(index, next(seeds)))

    support_specs = (
        (Fraction(1, 10), 0.44),
        (Fraction(1, 12), 0.46),
        (Fraction(1, 8), 0.48),
        (Fraction(1, 16), 0.50),
    )
    for index, (condition_mass, sample_center) in enumerate(support_specs):
        cases.append(_support_floor_case(index, next(seeds), condition_mass, sample_center))

    for index, value in enumerate((Fraction(13, 20), Fraction(7, 10), Fraction(3, 4), Fraction(4, 5), Fraction(5, 6), Fraction(9, 10))):
        cases.append(
            _point_case(
                case_id=f"p7b-narrow-{index:02d}",
                family="true_narrow_controls_v2",
                role="acceptance_control",
                seed=next(seeds),
                value=value,
            )
        )

    for index, value in enumerate((Fraction(3, 5), Fraction(2, 3), Fraction(7, 10))):
        cases.append(
            _point_case(
                case_id=f"p7b-pressure-noop-{index:02d}",
                family="pressure_noop_controls",
                role="acceptance_control",
                seed=next(seeds),
                value=value,
                with_pressure_noop=True,
            )
        )

    for index, value in enumerate((Fraction(3, 5), Fraction(2, 3), Fraction(3, 4))):
        cases.append(
            _point_case(
                case_id=f"p7b-loss-{index:02d}",
                family="loss_escape_controls_v2",
                role="loss_control",
                seed=next(seeds),
                value=value,
                max_loss=1e-6,
            )
        )

    if not prereg.seen_cases_are_excluded(tuple(case.case_id for case in cases)):
        raise RuntimeError("Phase-7 seen case id reused in Phase-7b corpus")
    return tuple(cases)


def family_counts(cases: tuple[Phase7bCase, ...]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for case in cases:
        counts[case.family] = counts.get(case.family, 0) + 1
    return counts


def corpus_summary(cases: tuple[Phase7bCase, ...] | None = None) -> dict[str, object]:
    cases = generate_phase7b_corpus() if cases is None else cases
    return {
        "corpus_generator_version": CORPUS_GENERATOR_VERSION,
        "corpus_status": CORPUS_STATUS,
        "case_count": len(cases),
        "false_closure_traps": sum(1 for case in cases if case.role == "false_closure_trap"),
        "acceptance_controls": sum(1 for case in cases if case.role == "acceptance_control"),
        "loss_controls": sum(1 for case in cases if case.role == "loss_control"),
        "families": family_counts(cases),
    }


if __name__ == "__main__":
    import sys

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    print("Phase 7b corpus:", corpus_summary())
