#!/usr/bin/env python
"""Frozen test for the BoxSEL Phase-6b general trace schema scaffold.

Run: python scripts/test_boxsel_phase6b_trace_schema.py
"""

import sys

sys.path.insert(0, "scripts")
import boxsel_phase3_restart_sampler as sampler
import boxsel_phase7_run as phase7
import boxsel_phase6b_trace_schema as schema

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


print("(1) schema status is scaffold-only:")
summary = schema.schema_summary()
check("schema version is named",
      summary["schema_version"] == "phase6b_general_trace_schema_v1")
check("no v2 detector is claimed",
      summary["detector_status"] == "NOT_BUILT")
check("no Phase-7b held-out protocol is registered",
      summary["heldout_status"] == "NOT_REGISTERED")
check("Phase-7 failure class is stable low-loss false closure",
      summary["phase7_failure_class"] == "stable_low_loss_false_closure")
check("Phase-7 seen constants cover PMP and point controls",
      len(schema.PHASE7_PMP_PARAMETERS) == 4 and len(schema.PHASE7_POINT_CONTROL_VALUES) == 6)

print("(2) Phase-7 cases are seen diagnostics only:")
check("all 16 Phase-7 cases are recorded as seen",
      len(schema.PHASE7_SEEN_CASE_IDS) == 16)
check("known Phase-7 cases are recognized as seen",
      schema.phase7_cases_are_seen(("helly-00", "pmp-03", "loss-01")))
check("unknown case is not treated as already seen",
      not schema.phase7_cases_are_seen(("helly-00", "phase7b-new-00")))

print("(3) feature names are oracle-free and include the missing signal class:")
names = schema.feature_names()
check("feature list is nonempty and oracle-free",
      names and schema.feature_names_are_oracle_free(), f"{names}")
check("feature list includes pressure response",
      "pressure_low_shift" in names)
check("feature list includes optimizer disagreement",
      "optimizer_low_spread" in names)
check("feature list includes support floors",
      {"condition_mass_floor", "numerator_mass_floor", "support_floor"}.issubset(set(names)))
check("feature list has no exact/oracle/label fields",
      all("exact" not in n and "oracle" not in n and "label" not in n for n in names))

print("(4) constraint and support traces compute local quantities only:")
constraint = schema.ConstraintTrace(
    "c1",
    lower_target=0.25,
    upper_target=0.75,
    observed=0.30,
    condition_mass=0.80,
    numerator_mass=0.24,
)
check("constraint slack is local to the observed constraint",
      abs(constraint.lower_slack - 0.05) < 1e-15
      and abs(constraint.upper_slack - 0.45) < 1e-15)
check("satisfied constraint has zero violation",
      constraint.violation == 0.0)
bad_constraint = schema.ConstraintTrace("c2", 0.25, 0.75, 0.10, 0.80, 0.08)
check("violated constraint reports positive violation",
      abs(bad_constraint.violation - 0.15) < 1e-15)
support = schema.SupportTrace(0.5, 0.2, 0.4, 0.25)
check("support trace computes conditional value without oracle labels",
      abs(support.conditional_value - 0.4) < 1e-15)

print("(5) diagnostic stable-PMP trace has the intended quiet shape:")
trace = schema.stable_pmp_failure_trace()
features = schema.trace_features(trace)
check("diagnostic trace is stable and low-loss",
      features.max_loss == 0.0
      and features.early_lower_drop <= 0.002
      and features.late_lower_drop <= 0.002,
      f"early={features.early_lower_drop}, late={features.late_lower_drop}")
check("diagnostic trace has high constraint slack",
      features.min_constraint_slack >= 0.0 and features.max_constraint_violation == 0.0)
check("diagnostic trace exposes support masses",
      features.condition_mass_floor == 0.5
      and features.numerator_mass_floor == 0.2
      and features.support_floor == 0.2)
check("ordinary diagnostic trace has no pressure shift by itself",
      features.pressure_low_shift == 0.0)

print("(6) Phase-3 reports convert to GeneralTrace:")
phase3_report = sampler.ordinary_restart_report(dim=2, restarts=16, seed=101)
phase3_trace = schema.phase3_report_to_general_trace(phase3_report, case_id="phase3-smoke")
check("Phase-3 adapter preserves sample interval",
      phase3_trace.sample_interval == phase3_report.sample_interval)
check("Phase-3 adapter preserves seed, dimension, and endpoint count",
      phase3_trace.seed == 101 and phase3_trace.dimension == 2 and len(phase3_trace.endpoints) == 16)
check("Phase-3 adapter emits atom and pair constraints",
      len(phase3_trace.constraints) == 6
      and {item.name for item in phase3_trace.constraints}.issuperset({"atom_A_volume", "pair_AB_given_A"}))
check("Phase-3 support conditional equals the sampled lower endpoint",
      abs(phase3_trace.support.conditional_value - phase3_report.sample_interval[0]) < 1e-15)
check("Phase-3 features remain oracle-free and trace-derived",
      schema.trace_features(phase3_trace).max_loss == phase3_report.max_loss)

print("(7) Phase-7 result rows convert to GeneralTrace diagnostics:")
case_rows = phase7.run_cases()
phase7_traces = schema.phase7_general_traces()
trace_by_id = {trace.case_id: trace for trace in phase7_traces}
case_by_id = {case.case_id: case for case in case_rows}
check("all Phase-7 result rows convert",
      len(phase7_traces) == 16 and set(trace_by_id) == set(schema.PHASE7_SEEN_CASE_IDS))
check("converted rows are marked as seen diagnostics",
      schema.phase7_cases_are_seen(trace_by_id))
check("Helly Phase-7 conversion rebuilds real sampler trace",
      len(trace_by_id["helly-00"].endpoints) == phase7.MAIN_RESTARTS
      and abs(trace_by_id["helly-00"].sample_interval[0] - case_by_id["helly-00"].sample_lower) < 1e-15
      and abs(trace_by_id["helly-00"].sample_interval[1] - case_by_id["helly-00"].sample_upper) < 1e-15)
check("PMP Phase-7 conversion preserves observed interval",
      abs(trace_by_id["pmp-02"].sample_interval[0] - case_by_id["pmp-02"].sample_lower) < 1e-15
      and abs(trace_by_id["pmp-02"].sample_interval[1] - case_by_id["pmp-02"].sample_upper) < 1e-15)
check("loss-control conversion preserves nonzero loss",
      schema.trace_features(trace_by_id["loss-00"]).max_loss == case_by_id["loss-00"].max_loss)

print("(8) pressure and comparison traces move only trace-derived spreads:")
pressure_trace = schema.GeneralTrace(
    case_id="pmp-diagnostic-pressure",
    family=trace.family,
    seed=1,
    dimension=2,
    optimizer_mode="query_pressure",
    endpoints=(
        schema.EndpointObservation(0, 0.31, 0.41, 0.0, pressure=1.0),
        schema.EndpointObservation(1, 0.30, 0.42, 0.0, pressure=1.0),
    ),
    constraints=trace.constraints,
    support=trace.support,
)
dim_trace = schema.GeneralTrace(
    case_id="pmp-diagnostic-dim3",
    family=trace.family,
    seed=2,
    dimension=3,
    optimizer_mode="ordinary_restart",
    endpoints=(schema.EndpointObservation(0, 0.36, 0.41, 0.0),),
    constraints=trace.constraints,
    support=trace.support,
)
with_pressure = schema.trace_features(
    trace,
    pressure_traces=(pressure_trace,),
    dimension_traces=(trace, dim_trace),
    optimizer_traces=(trace, pressure_trace),
)
check("pressure_low_shift records trace-only lower movement",
      abs(with_pressure.pressure_low_shift - 0.096) < 1e-15,
      f"shift={with_pressure.pressure_low_shift}")
check("dimension spread records low-endpoint disagreement",
      abs(with_pressure.dimension_low_spread - 0.036) < 1e-15,
      f"spread={with_pressure.dimension_low_spread}")
check("optimizer spread records ordinary-vs-pressure disagreement",
      abs(with_pressure.optimizer_low_spread - 0.096) < 1e-15,
      f"spread={with_pressure.optimizer_low_spread}")

print("(9) PMP query-pressure producer exposes the stable failure direction:")
pmp_trace = trace_by_id["pmp-02"]
q1, q2 = schema.PHASE7_PMP_PARAMETERS["pmp-02"]
pmp_pressure = schema.pmp_query_pressure_trace(
    case_id=pmp_trace.case_id,
    seed=pmp_trace.seed,
    q1=q1,
    q2=q2,
    ordinary_lower=pmp_trace.sample_interval[0],
)
pressure_features = schema.trace_features(
    pmp_trace,
    pressure_traces=(pmp_pressure,),
    optimizer_traces=(pmp_trace, pmp_pressure),
)
check("PMP pressure trace is query-pressure mode and low-loss",
      pmp_pressure.optimizer_mode == "query_pressure" and pmp_pressure.max_loss == 0.0)
check("PMP pressure trace moves the lower endpoint below ordinary sampling",
      pmp_pressure.sample_interval[0] < pmp_trace.sample_interval[0])
check("PMP pressure shift matches ordinary-minus-pressure lower endpoints",
      abs(pressure_features.pressure_low_shift - (pmp_trace.sample_interval[0] - pmp_pressure.sample_interval[0])) < 1e-15)
check("PMP optimizer spread sees ordinary-vs-pressure disagreement",
      pressure_features.optimizer_low_spread == pressure_features.pressure_low_shift)
all_pmp_pressure = schema.phase7_pmp_pressure_traces()
check("pressure producer returns one pressure trace per seen PMP failure",
      len(all_pmp_pressure) == 4
      and {trace.case_id for trace in all_pmp_pressure}
      == {f"{case_id}-query-pressure" for case_id in schema.PHASE7_PMP_PARAMETERS})

print(f"\n{'ALL PASS -- Phase-6b trace schema adapters: Phase-3/Phase-7 traces convert; PMP pressure traces expose the stable failure direction without held-out claims' if fail == 0 else str(fail) + ' FAILED'}")
sys.exit(1 if fail else 0)
