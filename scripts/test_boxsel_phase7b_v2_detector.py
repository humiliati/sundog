#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-7b v2 trace detector.

Run: python scripts/test_boxsel_phase7b_v2_detector.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase6b_trace_schema as schema
import boxsel_phase7b_v2_detector as det

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

fail = 0


def check(name, cond, detail=""):
    global fail
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + detail) if detail else ''}")
    if not cond:
        fail += 1


def point_trace(case_id="point", value=0.7, max_loss=0.0):
    return schema.point_condition_general_trace(
        case_id=case_id,
        family="v2_unit_control",
        seed=100,
        value=value,
        q_values=tuple(value + ((i % 3) - 1) * 0.0002 for i in range(24)),
        max_loss=max_loss,
    )


def generic_trace(
    *,
    case_id="generic",
    q_values=(0.8,) * 24,
    support_floor=0.5,
    observed=0.7,
    lower_target=0.0,
    upper_target=1.0,
    max_loss=0.0,
):
    lo = min(q_values)
    return schema.GeneralTrace(
        case_id=case_id,
        family="v2_unit_control",
        seed=101,
        dimension=2,
        optimizer_mode="ordinary_restart",
        endpoints=tuple(
            schema.EndpointObservation(index=i, lower=q, upper=q, loss=max_loss)
            for i, q in enumerate(q_values)
        ),
        constraints=(
            schema.ConstraintTrace(
                name="unit_constraint",
                lower_target=lower_target,
                upper_target=upper_target,
                observed=observed,
                condition_mass=1.0,
                numerator_mass=observed,
            ),
        ),
        support=schema.SupportTrace(
            condition_mass=support_floor,
            numerator_mass=support_floor,
            atom_support_min=support_floor,
            meet_support_min=support_floor,
        ),
    )


print("(1) v2 detector and thresholds are frozen:")
summary = det.detector_summary()
check("detector version and threshold version are named",
      summary["detector_version"] == "phase7b_v2_trace_detector_v1"
      and summary["threshold_version"] == "phase7b_v2_thresholds_v1")
check("detector and thresholds are frozen, results are not run",
      summary["detector_status"] == "FROZEN"
      and summary["threshold_status"] == "FROZEN"
      and summary["results_status"] == "NOT_RUN")
check("frozen features match the Phase-6b schema",
      det.FROZEN_FEATURES == schema.feature_names())
check("thresholds are keyed only by trace feature names",
      set(det.frozen_thresholds()).issubset(set(schema.feature_names())))
check("pressure/support thresholds are positive and explicit",
      det.PRESSURE_LOW_SHIFT_THRESHOLD > 0
      and det.OPTIMIZER_LOW_SPREAD_THRESHOLD > 0
      and det.SUPPORT_FLOOR_THRESHOLD > 0)

print("(2) stable PMP pressure response now abstains:")
ordinary = schema.stable_pmp_failure_trace()
pressure = schema.pmp_query_pressure_trace(
    case_id=ordinary.case_id,
    seed=ordinary.seed,
    q1=0.5,
    q2=0.5,
    ordinary_lower=ordinary.sample_interval[0],
)
decision = det.detector_decision(
    ordinary,
    pressure_traces=(pressure,),
    optimizer_traces=(ordinary, pressure),
)
check("PMP diagnostic remains quiet under ordinary restarts",
      decision.features.early_lower_drop == 0.0 and decision.features.seed_low_range == 0.0)
check("query pressure and optimizer disagreement cross frozen thresholds",
      decision.features.pressure_low_shift >= det.PRESSURE_LOW_SHIFT_THRESHOLD
      and decision.features.optimizer_low_spread >= det.OPTIMIZER_LOW_SPREAD_THRESHOLD)
check("stable PMP pressure response abstains",
      decision.action == det.ACTION_ABSTAIN
      and {"pressure_low_shift", "optimizer_low_spread"}.issubset(set(decision.flags)),
      f"action={decision.action}, flags={decision.flags}")

print("(3) accept controls stay accept when pressure is a noop:")
stable = point_trace(value=0.7)
stable_decision = det.detector_decision(stable)
check("stable point-control trace is accepted",
      stable_decision.action == det.ACTION_ACCEPT and stable_decision.flags == (),
      f"{stable_decision}")
noop_pressure = schema.point_condition_general_trace(
    case_id="point-pressure-noop",
    family="v2_unit_control",
    seed=100,
    value=0.7,
    q_values=tuple(0.7 + ((i % 3) - 1) * 0.0002 for i in range(12)),
    optimizer_mode="query_pressure",
)
noop_decision = det.detector_decision(
    stable,
    pressure_traces=(noop_pressure,),
    optimizer_traces=(stable, noop_pressure),
)
check("query pressure with no lower movement remains accepted",
      noop_decision.action == det.ACTION_ACCEPT
      and noop_decision.features.pressure_low_shift == 0.0
      and noop_decision.features.optimizer_low_spread < det.OPTIMIZER_LOW_SPREAD_THRESHOLD)

print("(4) support and ordinary turbulence widen without oracle labels:")
low_support = generic_trace(case_id="low-support", support_floor=0.04)
low_support_decision = det.detector_decision(low_support)
check("low support alone widens rather than accepts",
      low_support_decision.action == det.ACTION_WIDEN
      and low_support_decision.flags == ("support_floor",),
      f"{low_support_decision}")
drift = generic_trace(case_id="drift", q_values=(0.90, 0.86, 0.82, 0.79, 0.77, 0.75) + (0.75,) * 18)
drift_decision = det.detector_decision(drift)
check("ordinary endpoint drift widens",
      drift_decision.action == det.ACTION_WIDEN
      and "early_lower_drop" in drift_decision.flags,
      f"{drift_decision}")

print("(5) loss and constraint violation abstain immediately:")
lossy = point_trace(case_id="lossy", value=0.7, max_loss=1e-6)
lossy_decision = det.detector_decision(lossy)
check("loss escape abstains",
      lossy_decision.action == det.ACTION_ABSTAIN
      and lossy_decision.flags == ("max_loss",),
      f"{lossy_decision}")
violated = generic_trace(case_id="violated", observed=0.2, lower_target=0.5, upper_target=1.0)
violated_decision = det.detector_decision(violated)
check("constraint violation abstains",
      violated_decision.action == det.ACTION_ABSTAIN
      and violated_decision.flags == ("max_constraint_violation",),
      f"{violated_decision}")

print("(6) seed and dimension disagreement remain widening signals:")
seed_a = generic_trace(case_id="seed-a", q_values=(0.66,) * 8)
seed_b = generic_trace(case_id="seed-b", q_values=(0.62,) * 8)
seed_decision = det.detector_decision(stable, seed_traces=(seed_a, seed_b))
check("seed disagreement widens",
      seed_decision.action == det.ACTION_WIDEN and seed_decision.flags == ("seed_low_range",),
      f"{seed_decision}")
dim_a = generic_trace(case_id="dim-a", q_values=(0.70,) * 8)
dim_b = generic_trace(case_id="dim-b", q_values=(0.66,) * 8)
dim_decision = det.detector_decision(stable, dimension_traces=(dim_a, dim_b))
check("dimension disagreement widens",
      dim_decision.action == det.ACTION_WIDEN and dim_decision.flags == ("dimension_low_spread",),
      f"{dim_decision}")

print(f"\n{'ALL PASS -- Phase-7b v2 detector frozen: pressure response abstains, controls accept, loss/violation abstain' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
