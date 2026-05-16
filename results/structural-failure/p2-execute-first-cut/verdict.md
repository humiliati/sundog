# Structural Failure Coincidence P2 Execution Verdict

Generated: 2026-05-16T04:34:53.669Z
Phase: p2-execute-first-cut
Verdict: MACHINERY_LIVE_ROUTE_TEST_VACUOUS
Prereg outcome: instrument does not exercise discriminating route behavior
Rail vocabulary: STALLED / UNTESTED

## Quantities

1. Convergence mechanical check: PASS (59/59, rate 1)
2. Counterfactual mechanical check: PASS
3. Boundary-state mechanical check: PASS
4. Matched-baseline efficiency: route/analytic sample ratio 1601

## Route Construction Audit

Route test vacuous: true
Route and analytic baseline are the same inverse: true
Decoys reachable through route objective: false
CZA/tangent affect q estimate: false
Reason: The bundle generator sets f_par = R22/cos(h), while the route objective maximizes -abs(f_par - R22/cos(q)); the matched analytic baseline is the same inverse arccos(R22/f_par). The route is g^-1(g(h)) by grid search, not an independent policy test.

## Positive Control

Decoy-correlate positive control verdict: OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED
Max positive-control decoy movement: 70 deg
Min per-sample positive-control battery movement: 12 deg
Max route decoy movement: 0 deg

## Boundary Events

L1 low-leverage abstention: PASS (threshold h = 11.3649 deg)
L2 CZA drop: PASS (observed 32.125 deg, expected 32 deg)
L3 tangent drop: PASS (observed 28.875 deg, expected 29 deg)
L4 supralateral non-handle: PASS

## Public-Language Guard

This is a machinery-live / route-test-vacuous result. It is not a traceability pass; do not use CONFIRMED or theorem language.

