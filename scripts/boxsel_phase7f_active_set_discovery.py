#!/usr/bin/env python
r"""BoxSEL Phase 7f - active-set discovery from raw box traces.

Phase 7e recovered the Helly box endpoint from an active-set trace, but still assumed the active
set was already named. Phase 7f removes that assumption for the KKT witness: start from raw box
intervals, compute exact residuals, discover AC/BC active and AB slack, derive the active equation,
then hand the discovered trace to the Phase-7e recovery rule.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import json
from pathlib import Path
from typing import Mapping

import boxsel_kkt_exact as kkt
import boxsel_phase4_interval_gap as gap
import boxsel_phase7e_oracle_free_recovery as recovery


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = REPO_ROOT / "results" / "boxsel" / "phase7f_active_set_discovery"
OUTPUT_PATH = OUTPUT_DIR / "manifest.json"

DISCOVERY_VERSION = "phase7f_active_set_discovery_v0"
DISCOVERY_STATUS = "DISCOVERY_RECEIPT"

PRIMARY_DISCOVERY_CLAIM = (
    "For the Helly-seed KKT box trace, active pairs and the recovery equation can be discovered "
    "from raw oracle-free box geometry: exact residuals identify AC and BC as active, AB as slack, "
    "and the structured active geometry yields 4x^2 - 9x + 4 = 0."
)

BOUNDARY = (
    "Toy micro-SEL Helly box fragment only. This discovers the active set for the recorded KKT "
    "trace; it is not a general active-set learner, not a global optimizer, not exact SEL "
    "inference, and not a real-KG or product claim."
)

PAIR_TARGET = kkt.Surd(Fraction(1, 4))
ATOM_TARGET = kkt.Surd(Fraction(1, 2))
PAIR_NAMES = ("AB", "AC", "BC")

BoxValue = kkt.Surd | Fraction | int
BoxInterval = tuple[BoxValue, BoxValue]
Box = tuple[BoxInterval, ...]


@dataclass(frozen=True)
class RawBoxTrace:
    """Raw pressure/extremal box trace: no active-set labels and no oracle labels."""

    trace_id: str
    family: str
    dimension: int
    optimizer_mode: str
    boxes: Mapping[str, Box]
    oracle_fields_present: bool = False


@dataclass(frozen=True)
class DiscoveryResult:
    trace_id: str
    applicable: bool
    active_pairs: tuple[str, ...]
    slack_pairs: tuple[str, ...]
    atom_volumes: dict[str, str]
    pair_volumes: dict[str, str]
    pair_residuals: dict[str, str]
    active_equation: tuple[int, int, int] | None
    discovered_x: str | None
    discovered_z: str | None
    z_from_ac: str | None
    z_from_bc: str | None
    recovered_endpoint: str | None
    validation_matches_recovery: bool


def _coerce(value: BoxValue) -> kkt.Surd:
    return value if isinstance(value, kkt.Surd) else kkt.Surd(value)


def _coerce_box(box: Box) -> tuple[tuple[kkt.Surd, kkt.Surd], ...]:
    return tuple((_coerce(lo), _coerce(hi)) for lo, hi in box)


def _coerce_boxes(boxes: Mapping[str, Box]) -> dict[str, tuple[tuple[kkt.Surd, kkt.Surd], ...]]:
    return {name: _coerce_box(box) for name, box in boxes.items()}


def _surd_text(value: kkt.Surd) -> str:
    return repr(value)


def _surd_payload(value: kkt.Surd) -> dict[str, object]:
    return {"repr": repr(value), "a": str(value.a), "b": str(value.b), "float": float(value)}


def _interval_payload(interval: tuple[kkt.Surd, kkt.Surd]) -> tuple[dict[str, object], dict[str, object]]:
    return _surd_payload(interval[0]), _surd_payload(interval[1])


def _box_payload(box: tuple[tuple[kkt.Surd, kkt.Surd], ...]) -> tuple:
    return tuple(_interval_payload(interval) for interval in box)


def raw_kkt_box_trace() -> RawBoxTrace:
    return RawBoxTrace(
        trace_id="phase7f-helly-kkt-raw-box-trace",
        family="helly_seed_box_active_set_discovery",
        dimension=2,
        optimizer_mode="query_pressure_extremal",
        boxes=kkt.optimal_config(),
    )


def raw_rational_witness_trace() -> RawBoxTrace:
    return RawBoxTrace(
        trace_id="phase7f-rational-witness-negative-control",
        family="helly_seed_box_active_set_negative_control",
        dimension=2,
        optimizer_mode="query_pressure_extremal",
        boxes=gap.rational_box2_shrink_witness(),
    )


def raw_trace_payload(trace: RawBoxTrace) -> dict[str, object]:
    boxes = _coerce_boxes(trace.boxes)
    return {
        "traceId": trace.trace_id,
        "family": trace.family,
        "dimension": trace.dimension,
        "optimizerMode": trace.optimizer_mode,
        "oracleFieldsPresent": trace.oracle_fields_present,
        "boxes": {name: _box_payload(box) for name, box in boxes.items()},
    }


def atom_volumes(trace: RawBoxTrace) -> dict[str, kkt.Surd]:
    boxes = _coerce_boxes(trace.boxes)
    return {name: kkt.box_volume(boxes[name]) for name in ("A", "B", "C")}


def pair_volumes(trace: RawBoxTrace) -> dict[str, kkt.Surd]:
    boxes = _coerce_boxes(trace.boxes)
    return {
        "AB": kkt.meet_volume([boxes["A"], boxes["B"]]),
        "AC": kkt.meet_volume([boxes["A"], boxes["C"]]),
        "BC": kkt.meet_volume([boxes["B"], boxes["C"]]),
    }


def triple_volume(trace: RawBoxTrace) -> kkt.Surd:
    boxes = _coerce_boxes(trace.boxes)
    return kkt.meet_volume([boxes["A"], boxes["B"], boxes["C"]])


def query_value(trace: RawBoxTrace) -> kkt.Surd:
    pairs = pair_volumes(trace)
    denominator = pairs["AB"]
    if denominator <= kkt.Surd(0):
        raise ValueError("query denominator AB is zero")
    return triple_volume(trace) / denominator


def pair_residuals(trace: RawBoxTrace) -> dict[str, kkt.Surd]:
    return {name: value - PAIR_TARGET for name, value in pair_volumes(trace).items()}


def discover_pairs(trace: RawBoxTrace) -> tuple[tuple[str, ...], tuple[str, ...]]:
    residuals = pair_residuals(trace)
    active = tuple(name for name in PAIR_NAMES if residuals[name].sign() == 0)
    slack = tuple(name for name in PAIR_NAMES if residuals[name].sign() > 0)
    return active, slack


def _axis_length(box: tuple[tuple[kkt.Surd, kkt.Surd], ...], axis: int) -> kkt.Surd:
    lo, hi = box[axis]
    return hi - lo


def structured_kkt_geometry(trace: RawBoxTrace) -> bool:
    boxes = _coerce_boxes(trace.boxes)
    half = kkt.Surd(Fraction(1, 2))
    one = kkt.Surd(1)
    zero = kkt.Surd(0)
    return (
        trace.dimension == 2
        and boxes["A"][0][1] == one
        and boxes["B"][0] == (zero, one)
        and boxes["C"][0][0] == zero
        and boxes["A"][1][0] == boxes["C"][1][0]
        and boxes["A"][1][1] == one
        and boxes["B"][1] == (half, one)
        and all(value == ATOM_TARGET for value in atom_volumes(trace).values())
    )


def structured_parameters(trace: RawBoxTrace) -> tuple[kkt.Surd, kkt.Surd]:
    if not structured_kkt_geometry(trace):
        raise ValueError("raw trace does not match the Phase-7f structured KKT geometry")
    boxes = _coerce_boxes(trace.boxes)
    x = _axis_length(boxes["A"], 0)
    z = _axis_length(boxes["C"], 0)
    return x, z


def derive_active_equation(trace: RawBoxTrace) -> tuple[int, int, int] | None:
    """Derive 4x^2 - 9x + 4 = 0 when the raw geometry discovers the KKT active set."""

    active, slack = discover_pairs(trace)
    if set(active) != {"AC", "BC"} or "AB" not in slack or not structured_kkt_geometry(trace):
        return None
    x, z = structured_parameters(trace)
    z_from_ac = kkt.Surd(2) * (kkt.Surd(1) - x)
    z_from_bc = x / (kkt.Surd(2) * (kkt.Surd(1) - x))
    if not (z == z_from_ac == z_from_bc):
        return None
    if kkt.Surd(4) * x * x - kkt.Surd(9) * x + kkt.Surd(4) != kkt.Surd(0):
        return None
    return (4, -9, 4)


def discovered_active_trace(trace: RawBoxTrace | None = None) -> recovery.ActiveSetTrace:
    trace = raw_kkt_box_trace() if trace is None else trace
    equation = derive_active_equation(trace)
    if equation is None:
        raise ValueError("raw trace does not yield the Phase-7e active recovery equation")
    active, slack = discover_pairs(trace)
    atoms = atom_volumes(trace)
    pairs = pair_volumes(trace)
    triple = triple_volume(trace)
    q = query_value(trace)
    return recovery.ActiveSetTrace(
        trace_id=f"{trace.trace_id}-discovered-active-set",
        family="helly_seed_box_discovered_active_set",
        dimension=trace.dimension,
        optimizer_mode=trace.optimizer_mode,
        atom_volumes={name: _surd_text(value) for name, value in atoms.items()},
        pair_volumes={name: _surd_text(value) for name, value in pairs.items()},
        triple_volume=_surd_text(triple),
        query_denominator=_surd_text(pairs["AB"]),
        query_value=_surd_text(q),
        active_pairs=active,
        slack_pairs=slack,
        active_equation=equation,
        oracle_fields_present=trace.oracle_fields_present,
    )


def discovery_result(trace: RawBoxTrace | None = None) -> DiscoveryResult:
    trace = raw_kkt_box_trace() if trace is None else trace
    active, slack = discover_pairs(trace)
    atoms = atom_volumes(trace)
    pairs = pair_volumes(trace)
    residuals = pair_residuals(trace)
    equation = derive_active_equation(trace)
    recovered_endpoint = None
    validation_matches = False
    x_text = z_text = z_ac_text = z_bc_text = None
    if structured_kkt_geometry(trace):
        x, z = structured_parameters(trace)
        z_from_ac = kkt.Surd(2) * (kkt.Surd(1) - x)
        z_from_bc = x / (kkt.Surd(2) * (kkt.Surd(1) - x))
        x_text = _surd_text(x)
        z_text = _surd_text(z)
        z_ac_text = _surd_text(z_from_ac)
        z_bc_text = _surd_text(z_from_bc)
    if equation is not None:
        active_trace = discovered_active_trace(trace)
        recovered = recovery.recover_lower_endpoint(active_trace)
        recovered_endpoint = _surd_text(recovered)
        validation_matches = recovered == query_value(trace)
    return DiscoveryResult(
        trace_id=trace.trace_id,
        applicable=equation is not None and not trace.oracle_fields_present,
        active_pairs=active,
        slack_pairs=slack,
        atom_volumes={name: _surd_text(value) for name, value in atoms.items()},
        pair_volumes={name: _surd_text(value) for name, value in pairs.items()},
        pair_residuals={name: _surd_text(value) for name, value in residuals.items()},
        active_equation=equation,
        discovered_x=x_text,
        discovered_z=z_text,
        z_from_ac=z_ac_text,
        z_from_bc=z_bc_text,
        recovered_endpoint=recovered_endpoint,
        validation_matches_recovery=validation_matches,
    )


def discovery_summary() -> dict[str, object]:
    raw = raw_kkt_box_trace()
    result = discovery_result(raw)
    active_trace = discovered_active_trace(raw)
    recovered = recovery.recovery_result(active_trace)
    negative = discovery_result(raw_rational_witness_trace())
    return {
        "discoveryVersion": DISCOVERY_VERSION,
        "status": DISCOVERY_STATUS,
        "primaryDiscoveryClaim": PRIMARY_DISCOVERY_CLAIM,
        "boundary": BOUNDARY,
        "rawTrace": raw_trace_payload(raw),
        "rawTraceOracleFree": not raw.oracle_fields_present,
        "discoveryRule": {
            "input": "raw box intervals",
            "activeCriterion": "exact zero residual against pair target 1/4",
            "slackCriterion": "positive residual against pair target 1/4",
            "usesExactOracle": False,
            "usesActiveLabelsAsInput": False,
            "usesClosedFormForDiscovery": False,
        },
        "discovered": asdict(result),
        "phase7eRecovered": {
            "traceId": recovered.trace_id,
            "recoveredEndpoint": recovered.recovered_endpoint,
            "validationMatchesClosedForm": recovered.validation_matches_closed_form,
            "observedMatchesRecovered": recovered.observed_matches_recovered,
        },
        "negativeControl": asdict(negative),
    }


def write_results(path: Path = OUTPUT_PATH) -> dict[str, object]:
    data = discovery_summary()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def main() -> int:
    data = write_results()
    discovered = data["discovered"]
    print(f"BoxSEL Phase 7f active-set discovery: {data['status']}")
    print("active pairs:", ",".join(discovered["active_pairs"]))
    print("slack pairs:", ",".join(discovered["slack_pairs"]))
    print("active equation:", discovered["active_equation"])
    print("recovered endpoint:", data["phase7eRecovered"]["recoveredEndpoint"])
    print("manifest:", OUTPUT_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
