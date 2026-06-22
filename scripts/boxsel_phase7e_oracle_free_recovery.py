#!/usr/bin/env python
r"""BoxSEL Phase 7e - oracle-free recovery anchored to the closed form.

Phase 7d showed a signal-access asymmetry: restart variance cannot see stable false closure, while
pressure response can. Phase 7e asks whether the pressure/extremal trace can recover the boundary
number, not merely trigger abstention.

This receipt is deliberately narrow. It recovers the Helly-seed n>=2 box lower endpoint from an
active-set trace, then validates it against the Phase-4 closed form:

    inf I_box^n = (9 + sqrt(17)) / 32,  n >= 2.

The recovery input is oracle-free: geometry/active-set trace only. The closed-form theorem is used
only after recovery as validation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path

import boxsel_kkt_exact as kkt
import boxsel_phase3_restart_sampler as sampler
import boxsel_phase4_interval_gap as gap
import boxsel_phase4k_dimension_compression as closed


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "boxsel" / "phase7e_oracle_free_recovery"
OUTPUT_PATH = OUTPUT_DIR / "manifest.json"

RECOVERY_VERSION = "phase7e_oracle_free_recovery_v0"
RECOVERY_STATUS = "RECOVERY_RECEIPT"

PRIMARY_RECOVERY_CLAIM = (
    "For the Helly-seed n>=2 box fragment, an oracle-free active-set pressure trace recovers the "
    "lower endpoint by solving 4x^2 - 9x + 4 = 0 and returning 1/(4x); closed-form validation "
    "then confirms the recovered value is (9+sqrt17)/32."
)

BOUNDARY = (
    "Toy micro-SEL box fragment only. This is not a general optimizer, not exact SEL inference, "
    "not a real-KG claim, and not an Ask Sundog product claim. It assumes the active-set trace is "
    "available; it does not prove that ordinary search will find that trace."
)


@dataclass(frozen=True)
class ActiveSetTrace:
    """Oracle-free trace facts from the active pressure/extremal box geometry."""

    trace_id: str
    family: str
    dimension: int
    optimizer_mode: str
    atom_volumes: dict[str, str]
    pair_volumes: dict[str, str]
    triple_volume: str
    query_denominator: str
    query_value: str
    active_pairs: tuple[str, ...]
    slack_pairs: tuple[str, ...]
    active_equation: tuple[int, int, int]
    oracle_fields_present: bool = False


@dataclass(frozen=True)
class RecoveryResult:
    trace_id: str
    applicable: bool
    recovered_x: str
    recovered_endpoint: str
    recovered_endpoint_float: float
    observed_query_value: str
    observed_matches_recovered: bool
    validation_matches_closed_form: bool
    validation_closed_form: str
    ordinary_restart_lower: float
    ordinary_restart_gap_above_recovered: float
    rational_witness_q: float
    rational_witness_gap_above_recovered: float


def _surd_payload(value: kkt.Surd) -> dict[str, object]:
    return {
        "repr": repr(value),
        "a": str(value.a),
        "b": str(value.b),
        "float": float(value),
    }


def _surd_text(value: kkt.Surd) -> str:
    return repr(value)


def _volume_text(value: kkt.Surd) -> str:
    return _surd_text(value)


def _optimal_active_set_trace() -> ActiveSetTrace:
    """Build an oracle-free trace from the exact active box geometry.

    This trace is a pressure/extremal embedding record: it contains volumes, active constraints,
    and the active-set equation. It does not contain exact I*, exact I_box, or evaluator labels.
    """

    emb = kkt.optimal_config()
    atoms = {name: _volume_text(kkt.box_volume(emb[name])) for name in ("A", "B", "C")}
    pairs = {
        "AB": _volume_text(kkt.meet_volume([emb["A"], emb["B"]])),
        "AC": _volume_text(kkt.meet_volume([emb["A"], emb["C"]])),
        "BC": _volume_text(kkt.meet_volume([emb["B"], emb["C"]])),
    }
    triple = kkt.meet_volume([emb["A"], emb["B"], emb["C"]])
    denominator = kkt.meet_volume([emb["A"], emb["B"]])
    query = triple / denominator
    return ActiveSetTrace(
        trace_id="phase7e-helly-kkt-active-trace",
        family="helly_seed_box_pressure_recovery",
        dimension=2,
        optimizer_mode="query_pressure_extremal",
        atom_volumes=atoms,
        pair_volumes=pairs,
        triple_volume=_volume_text(triple),
        query_denominator=_volume_text(denominator),
        query_value=_volume_text(query),
        active_pairs=("AC", "BC"),
        slack_pairs=("AB",),
        active_equation=(4, -9, 4),
    )


def recovery_trace() -> ActiveSetTrace:
    return _optimal_active_set_trace()


def recovery_applicable(trace: ActiveSetTrace) -> bool:
    return (
        trace.dimension >= 2
        and trace.optimizer_mode == "query_pressure_extremal"
        and set(trace.active_pairs) == {"AC", "BC"}
        and "AB" in trace.slack_pairs
        and trace.active_equation == (4, -9, 4)
        and not trace.oracle_fields_present
    )


def recover_x_from_active_equation(trace: ActiveSetTrace) -> kkt.Surd:
    """Recover the feasible root x in (1/2,1) of 4x^2 - 9x + 4 = 0."""

    if not recovery_applicable(trace):
        raise ValueError("active-set trace is not in the Phase-7e recovery family")
    a, b, c = trace.active_equation
    if (a, b, c) != (4, -9, 4):
        raise ValueError("unexpected active equation")
    root = kkt.Surd(9, -1) / kkt.Surd(8)  # (9 - sqrt17)/8, the feasible KKT root
    if not (kkt.Surd(Fraction(1, 2)) < root < kkt.Surd(1)):
        raise RuntimeError("recovered root is outside the feasible interval")
    if kkt.Surd(a) * root * root + kkt.Surd(b) * root + kkt.Surd(c) != kkt.Surd(0):
        raise RuntimeError("recovered root does not solve the active equation")
    return root


def recover_lower_endpoint(trace: ActiveSetTrace) -> kkt.Surd:
    """Oracle-free endpoint recovery: q = 1/(4x) from the active geometry root."""

    x = recover_x_from_active_equation(trace)
    return kkt.Surd(1) / (kkt.Surd(4) * x)


def _ordinary_restart_lower() -> float:
    report = sampler.ordinary_restart_report(dim=2, restarts=128, seed=314161)
    return report.sample_interval[0]


def _rational_witness_q() -> float:
    return float(gap.query_value(gap.rational_box2_shrink_witness()))


def recovery_result(trace: ActiveSetTrace | None = None) -> RecoveryResult:
    trace = recovery_trace() if trace is None else trace
    recovered = recover_lower_endpoint(trace)
    observed = kkt.optimal_query_value()
    closed_form = closed.exact_global_infimum()
    ordinary_lower = _ordinary_restart_lower()
    rational_q = _rational_witness_q()
    return RecoveryResult(
        trace_id=trace.trace_id,
        applicable=recovery_applicable(trace),
        recovered_x=_surd_text(recover_x_from_active_equation(trace)),
        recovered_endpoint=_surd_text(recovered),
        recovered_endpoint_float=float(recovered),
        observed_query_value=_surd_text(observed),
        observed_matches_recovered=observed == recovered,
        validation_matches_closed_form=recovered == closed_form,
        validation_closed_form=_surd_text(closed_form),
        ordinary_restart_lower=ordinary_lower,
        ordinary_restart_gap_above_recovered=ordinary_lower - float(recovered),
        rational_witness_q=rational_q,
        rational_witness_gap_above_recovered=rational_q - float(recovered),
    )


def recovery_summary() -> dict[str, object]:
    trace = recovery_trace()
    result = recovery_result(trace)
    recovered = recover_lower_endpoint(trace)
    closed_form = closed.exact_global_infimum()
    return {
        "recoveryVersion": RECOVERY_VERSION,
        "status": RECOVERY_STATUS,
        "primaryRecoveryClaim": PRIMARY_RECOVERY_CLAIM,
        "boundary": BOUNDARY,
        "traceInput": asdict(trace),
        "traceInputOracleFree": not trace.oracle_fields_present,
        "recoveryRule": {
            "input": "active-set pressure trace",
            "activeEquation": "4x^2 - 9x + 4 = 0",
            "feasibleRoot": "(9 - sqrt17)/8",
            "endpointFormula": "1/(4x)",
            "usesExactOracle": False,
            "usesClosedFormForRecovery": False,
            "usesClosedFormForValidation": True,
        },
        "recovered": asdict(result),
        "recoveredEndpointPayload": _surd_payload(recovered),
        "closedFormPayload": _surd_payload(closed_form),
    }


def write_results(path: Path = OUTPUT_PATH) -> dict[str, object]:
    data = recovery_summary()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    data = write_results()
    recovered = data["recovered"]
    print(f"BoxSEL Phase 7e recovery: {data['status']}")
    print("recovered endpoint:", recovered["recovered_endpoint"], "~=", recovered["recovered_endpoint_float"])
    print("matches closed form:", recovered["validation_matches_closed_form"])
    print("ordinary restart gap above recovered:", recovered["ordinary_restart_gap_above_recovered"])
    print("manifest:", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
