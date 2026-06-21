# BoxSEL Phase 6 - Trace Detector Start

**Date:** 2026-06-21  
**Status:** First accept/widen/abstain guard built on observable restart traces.

## Purpose

Phase 6 is the actual Sundog contribution for this lane:

```text
Can observable embedding traces flag false closure without the oracle?
```

This start note implements the first detector slice on the Helly-seed trace from Phase 3. The guard
uses only sampled traces:

- sampled endpoint movement,
- zero-loss status,
- constraint slack,
- seed variance,
- dimension sensitivity.

It does **not** use `I*`, the exact `I_box` lower endpoint, or the Phase-4 closed form as decision
features. Those exact quantities are used only after the decision, to score whether the guard
accepted a known false-closure case.

## Guard

The rule returns:

```text
accept / widen / abstain
```

Current thresholds:

```text
loss_escape              : max_loss > 1e-9
endpoint_drift           : early lower-endpoint drop > 0.05
late_endpoint_drift      : late lower-endpoint drop > 0.01
active_constraint_slack  : min pairwise slack < 0.005
seed_variance            : sampled lower range across seeds > 0.02
dimension_sensitivity    : sampled lower spread across dimensions > 0.02
```

Decision logic:

```text
loss_escape or >=3 flags  -> abstain
1-2 flags                 -> widen
0 flags                   -> accept
```

## Helly-Seed Receipt

The canonical Phase-3 trace is:

```text
dim=2, N=128, seed=314159
I_sample = [0.5336525204919725, 1.0]
```

The detector sees four oracle-free warning flags:

```text
endpoint_drift
active_constraint_slack
seed_variance
dimension_sensitivity
```

and returns:

```text
abstain
```

The oracle evaluator then checks the already-known exact endpoint:

```text
I_box lower = (9 + sqrt 17)/32 ~= 0.4100970508005519
lower search gap = 0.1235554696914206
```

So the case is false-closed relative to `I_box`, but the detector does **not** accept it.

## Controls

The frozen test also locks simple controls:

- a stable high-slack synthetic trace is accepted,
- a one-flag active-slack trace widens,
- a loss-escape trace abstains immediately.

These controls prevent the start rule from degenerating into "always abstain."

## Claim Boundary

This is not a Phase-7 result. It is one seed-trap receipt and a rule skeleton:

```text
Helly false closure abstains under trace-only flags.
```

The detector is not yet validated on held-out ontologies and does not yet beat a restart-variance
baseline. Phase 7 remains the kill-gate.

## Artifacts

- `scripts/boxsel_phase6_trace_detector.py`
- `scripts/test_boxsel_phase6_trace_detector.py`

Verification:

```text
python scripts/test_boxsel_phase6_trace_detector.py
```

Result:

```text
15/15 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-6 trace detector start. Internal; one seed-trap abstention
receipt, not a held-out detector claim.*
