# Structural Failure Coincidence - P2 Execution Results

Pre-registration: [`README.md`](README.md)  
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)  
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)  
Harness: `scripts/structural-failure-p2-harness.mjs`  
Filed: **2026-05-15 (PT)**. Status: first-cut execution result.

## Run

Command:

```powershell
npm run p2:structural
```

Output:

```text
results/structural-failure/p2-execute-first-cut/
```

Key artifacts:

- `manifest.json`
- `trial-outcomes.csv`
- `counterfactual-edits.csv`
- `decoy-edits.csv`
- `boundary-events.csv`
- `verdict.json`
- `verdict.md`

The run completed inline in seconds, so it stayed under the AGENTS.md
~10-minute rule. No training, HaloSim render, or image raster path was used.

## Controller Paths

- **Admitted route controller:** transparent extremum-seeking over candidate
  `q`, using only `{f_par, f_cza, f_tan, R22, q}` inside the adapter.
  Decoys are not adapter inputs.
- **Matched analytic baseline:** `q = arccos(R22 / f_par)` on L1-eligible
  rows, no decoy access.
- **Decoy-correlate positive control:** generic least-squares fit over the
  normalized full bundle, including `d_sup`, `d_unanch`, and `d_style`.

Adapter invariant recorded by `verdict.json`:

```json
{
  "transparentAdapterInputs": ["f_par", "f_cza", "f_tan", "R22", "q"],
  "hiddenAltitudeReadInsideAdapter": false,
  "decoysReadInsideAdapter": false
}
```

## Verdict

Route-controller verdict:

```text
TRACEABILITY_HARNESS_PASS
```

Preregistered outcome:

```text
traceability harness passes on this domain
```

Rail vocabulary:

```text
OPERATING ENVELOPE / CONFIRMED
```

Positive-control verdict:

```text
OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED
```

This is **not** a theorem proof and **not** a debunking result. It is the
first-cut closed-form feature-bundle result: the admitted route controller
tracks the pre-registered inverse and boundaries, while the deliberately
decoy-sensitive correlate path is caught as opaque.

## Quantity Results

| quantity | result |
| --- | --- |
| (1) Convergence | **PASS**: 59 / 59 L1-eligible samples within `1.5 deg`; max eligible error ~0 |
| (2) Counterfactual steerability | **PASS**: 59 / 59 handle edits steer to `h'`; route decoy movement `0 deg`; positive-control decoy movement min `12 deg`, max `70 deg` |
| (3) Failure-boundary coincidence | **PASS**: L1 abstention, L2 CZA cutoff, L3 tangent merge, and L4 supralateral non-handle all matched |
| (4) Matched-baseline efficiency | Route/analytic sample ratio `1601`; efficiency-only annotation |

Boundary details:

| locus | observed | expected | result |
| --- | ---: | ---: | --- |
| L1 low-leverage abstention | 12 / 12 low-leverage rows abstained | `f_par < 1.02 * R22` (`h < 11.3649 deg`) | PASS |
| L2 CZA cutoff | `32.125 deg` | `32 deg +/- 1.5 deg` | PASS |
| L3 tangent merge | `28.875 deg` | `29 deg +/- 1.5 deg` | PASS |
| L4 supralateral | never promoted as handle | permanent non-handle | PASS |

For L2/L3, the first cut scores **handle-state switching**: CZA and tangent
terms cease to be active at the analytic guard while the parhelion route can
continue to emit `qhat`. This is not a claim that the whole altitude estimate
collapses at 32 deg or 29 deg; it is the boundary-aware switch behavior fixed
by the admitted closed-form adapter.

## Interpretation

The falsification machinery is live: the positive-control correlate fails the
decoy battery exactly as intended. The admitted route controller does not fail
this first cut. Under the pre-registered outcome table, the correct disposition
is a bounded apparatus result for the closed-form feature-bundle domain, with
the public-language guard still active: no universal alignment theorem claim.

Next step is P3 disposition: mirror this result into the public/status surfaces
as an apparatus/benchmark result, and keep stronger theorem language out of
copy unless a later, broader phase earns it.

---

## Correction - 2026-05-15 PT

Reviewer challenge accepted. The initial verdict above overstates the result.
The route-controller `PASS` is vacuous by construction:

- `makeBundle(h)` sets `f_par = R22 / cos(h)`.
- The route objective maximizes `-|f_par - R22 / cos(q)|`.
- The matched analytic baseline computes `q = arccos(R22 / f_par)`.

So the route and the analytic baseline are the same inverse, with the route
doing `g^-1(g(h))` by grid search. This is not an independent route-use test,
and it does not instantiate the existing photometric/extremum-seeking
controller promised by the P2 first-cut scope.

The q1/q2/q3 rows are therefore **mechanical checks**, not evidence that a
controller used the inverse:

- q1 convergence is the identity `g^-1(g(h)) = h`.
- q2 handle-edit steerability is the same identity after regenerating
  `f_par` from `h'`.
- q2 route decoy-invariance is definitional because the route objective cannot
  read `d_sup`, `d_unanch`, or `d_style`.
- q3 L2/L3 boundary events measure generator handle-state guards against the
  same constants; they do not show a behavioral degradation or earned handle
  switch in a policy.
- q3 L4 is hardcoded as a non-handle in the adapter.

Corrected disposition:

```text
MACHINERY_LIVE_ROUTE_TEST_VACUOUS
```

Corrected prereg interpretation:

```text
instrument does not exercise discriminating route behavior
```

Corrected rail vocabulary:

```text
STALLED / UNTESTED
```

What remains real:

- The decoy-correlate positive control still fires correctly:
  `OPAQUE_CORRELATE_POSITIVE_CONTROL_CONFIRMED`.
- The no-hidden-`h` / no-decoy route-adapter invariant is structurally
  satisfied.

The harness and output artifacts have been patched and rerun. The corrected
`verdict.json` / `verdict.md` now report
`MACHINERY_LIVE_ROUTE_TEST_VACUOUS`. The Public-Language Constraint remains in
force: no `CONFIRMED`, no traceability-success, and no theorem language from
this first cut.
