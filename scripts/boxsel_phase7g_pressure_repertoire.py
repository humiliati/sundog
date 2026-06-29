#!/usr/bin/env python
r"""BoxSEL Phase 7g - does the Phase-7 pressure detector inherit C2's order-relative law?

Pre-registered in docs/boxsel/PHASE7G_PRESSURE_REPERTOIRE_PREREG.md (committed 38dc4328, before this
runner existed). C2 (docs/boxsel/C2_PRESSURE_ABSTENTION_BREAK.md) broke orthogonal-pressure abstention
on parity: the detector's reach is the pressure family's ORDER. Here we check the actual frozen
Phase-7b v2 trace detector inherits the same boundary, where the BoxSEL "order" is the pressure
family's REPERTOIRE of failure-witnesses (the shapes it knows how to push toward).

Mechanism (from the detector): `boxsel_phase7b_v2_detector` is a pure function of GeneralTraceFeatures;
its only pressure signal is `pressure_low_shift = base_low - min(handed pressure_trace lows)`, flagged
at >= 0.05. The pressure family is a finite, SHAPE-SPECIFIC tuple (pmp_query_pressure_trace pushes
toward the known PMP witness q1*q2). So a detector handed a PMP repertoire can push toward PMP
witnesses but not toward a Helly witness.

Three scenarios, one frozen detector, pressure channel ISOLATED (every trap is clean on all
non-pressure features, so only `pressure_low_shift` can catch it):

  L liveness     : PMP false closure   + PMP repertoire   -> CAUGHT (shift >= 0.05)
  I inheritance  : Helly false closure + PMP repertoire   -> MISSED (no PMP pressure applies -> accept)
  R reach-extends: Helly false closure + Helly repertoire -> CAUGHT (pushes toward (9+sqrt17)/32)

Pre-registered verdict: INHERITS iff L caught AND I missed (a genuine, oracle-certified false closure
accepted) AND R caught. Reuses boxsel_phase6b_trace_schema, boxsel_phase7b_v2_detector, kkt Q_STAR.
"""
from fractions import Fraction

import boxsel_phase6b_trace_schema as schema
import boxsel_phase7b_v2_detector as detector
import boxsel_kkt_exact as kkt

HELLY_SAMPLE_LOWER = 0.50          # the falsely-closed reported lower (above the true I_box lower)
HELLY_TRUE_LOWER = float(kkt.Q_STAR)  # (9 + sqrt 17)/32 ~= 0.4100970, the exact I_box lower (oracle)
PMP_WITNESS = 0.25                 # q1*q2 for the PMP trap (q1 = q2 = 1/2): the PMP true lower


def helly_false_closure() -> schema.GeneralTrace:
    """A Helly-shaped false closure, CLEAN on every non-pressure feature.

    Reports lower 0.50 while the true I_box lower is (9+sqrt 17)/32 ~= 0.4101 (false closed). loss 0,
    no constraint violation, support_floor 0.125 > 0.08, equal lowers (drops 0) -> the detector can
    only catch it via pressure.
    """
    endpoints = tuple(
        schema.EndpointObservation(index=i, lower=HELLY_SAMPLE_LOWER, upper=1.0, loss=0.0)
        for i in range(24)
    )
    constraints = (
        schema.ConstraintTrace("helly_pair_AB", 0.25, 1.0, 0.5, condition_mass=0.5, numerator_mass=0.25),
        schema.ConstraintTrace("helly_pair_AC", 0.25, 1.0, 0.5, condition_mass=0.5, numerator_mass=0.25),
    )
    support = schema.SupportTrace(
        condition_mass=0.25, numerator_mass=0.125, atom_support_min=0.5, meet_support_min=0.25,
    )
    return schema.GeneralTrace(
        case_id="helly-false-closure",
        family="helly_false_closure",
        seed=0,
        dimension=2,
        optimizer_mode="ordinary_restart",
        endpoints=endpoints,
        constraints=constraints,
        support=support,
    )


def pmp_false_closure() -> schema.GeneralTrace:
    """A PMP-shaped stable false closure, clean on non-pressure features (schema diagnostic)."""
    return schema.stable_pmp_failure_trace()


def helly_pressure(trap: schema.GeneralTrace) -> schema.GeneralTrace:
    """A Helly-repertoire pressure: pushes the lower endpoint toward the true Helly witness Q_STAR."""
    endpoints = tuple(
        schema.EndpointObservation(index=i, lower=HELLY_TRUE_LOWER, upper=1.0, loss=0.0, pressure=1.0)
        for i in range(12)
    )
    return schema.GeneralTrace(
        case_id="helly-query-pressure",
        family="helly_false_closure_pressure",
        seed=trap.seed,
        dimension=2,
        optimizer_mode="query_pressure",
        endpoints=endpoints,
        constraints=trap.constraints,
        support=trap.support,
    )


def pmp_pressure(trap: schema.GeneralTrace) -> schema.GeneralTrace:
    """A PMP-repertoire pressure: the Phase-7 shape, pushing toward the PMP witness q1*q2 = 0.25."""
    return schema.pmp_query_pressure_trace(
        case_id="pmp",
        seed=trap.seed,
        q1=Fraction(1, 2),
        q2=Fraction(1, 2),
        ordinary_lower=trap.sample_interval[0],
    )


def pressure_for(shape: str, repertoire: frozenset, trap: schema.GeneralTrace) -> tuple:
    """Build the pressure family a detector with the given repertoire can apply to this trap.

    A repertoire builds a shape's pressure ONLY if the trap matches that shape -- a PMP repertoire has
    no pressure to construct for a Helly-shaped false closure (no PMP premises).
    """
    traces = []
    if shape == "pmp" and "pmp" in repertoire:
        traces.append(pmp_pressure(trap))
    if shape == "helly" and "helly" in repertoire:
        traces.append(helly_pressure(trap))
    return tuple(traces)


def run_scenario(shape: str, trap: schema.GeneralTrace, repertoire: frozenset) -> dict:
    pressure = pressure_for(shape, repertoire, trap)
    decision = detector.detector_decision(trap, pressure_traces=pressure)
    return {
        "shape": shape,
        "repertoire": sorted(repertoire),
        "pressureTraceCount": len(pressure),
        "pressureLowShift": schema.pressure_low_shift(trap, pressure),
        "action": decision.action,
        "flags": decision.flags,
        "caught": decision.action != detector.ACTION_ACCEPT,
    }


def repertoire_reach_report() -> dict:
    helly = helly_false_closure()
    pmp = pmp_false_closure()

    L = run_scenario("pmp", pmp, frozenset({"pmp"}))                 # liveness
    I = run_scenario("helly", helly, frozenset({"pmp"}))             # the inheritance
    R = run_scenario("helly", helly, frozenset({"pmp", "helly"}))    # reach extends

    # oracle-certified false closures: reported lower strictly above the true lower
    helly_false_closed = helly.sample_interval[0] > HELLY_TRUE_LOWER
    pmp_false_closed = pmp.sample_interval[0] > PMP_WITNESS

    inherits = L["caught"] and (not I["caught"]) and R["caught"] and helly_false_closed
    if not L["caught"]:
        verdict = "VOID_dead_repertoire"
    elif I["caught"]:
        verdict = "DOES_NOT_INHERIT_shape_agnostic"
    else:
        verdict = "INHERITS_order_relative"

    return {
        "liveness": L,
        "inheritance": I,
        "reachExtends": R,
        "hellyFalseClosed": helly_false_closed,
        "pmpFalseClosed": pmp_false_closed,
        "hellySampleLower": helly.sample_interval[0],
        "hellyTrueLower": HELLY_TRUE_LOWER,
        "pressureThreshold": detector.PRESSURE_LOW_SHIFT_THRESHOLD,
        "inherits": inherits,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    r = repertoire_reach_report()
    print("BoxSEL Phase 7g - pressure-repertoire reach (does the v2 detector inherit C2?)")
    print(f"  pressure flag threshold = {r['pressureThreshold']}")
    print(f"  Helly false closure: reported lower {r['hellySampleLower']:.4f} > true (9+sqrt17)/32 "
          f"{r['hellyTrueLower']:.4f}  -> false_closed = {r['hellyFalseClosed']}")
    print()
    for key, label in (("liveness", "L liveness    "), ("inheritance", "I inheritance "),
                       ("reachExtends", "R reach-extend")):
        s = r[key]
        print(f"  {label}: shape={s['shape']:<5} repertoire={s['repertoire']}  "
              f"shift={s['pressureLowShift']:.4f}  action={s['action']:<7} "
              f"{'CAUGHT' if s['caught'] else 'MISSED -> ACCEPT'}  flags={list(s['flags'])}")
    print()
    print(f"  VERDICT: {r['verdict']}  (inherits = {r['inherits']})")
    if r["inherits"]:
        print("  => the Phase-7 detector inherits C2: reach = pressure REPERTOIRE. A false closure whose")
        print("     witness lies outside the handed repertoire (here Helly under a PMP-only family) is")
        print("     ACCEPTED; extend the repertoire and it is caught. Same order-relative law as C1/C2.")
