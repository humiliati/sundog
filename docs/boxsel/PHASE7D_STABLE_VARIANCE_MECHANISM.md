# BoxSEL Phase 7d - Stable Variance Mechanism

**Date:** 2026-06-21  
**Status:** `MECHANISM_RECEIPT`

Phase 7d reframes the Phase-7b result away from a thin benchmark win and toward the mechanism:

```text
stable false closure is exactly the case restart variance is structurally unable to see
```

The point is not that `0/16` beats `16/16` on a small corpus. The point is that the missed class
has the negation of the baseline's only signal.

## Mechanism Claim

The locked restart-variance baseline is:

```text
restart_variance_only_v0:
  widen iff seed_low_range > 0.02
  otherwise accept
```

Therefore, for any false-closure case satisfying:

```text
seed_low_range <= 0.02
```

the baseline accepts by construction. It has no observable that can distinguish:

```text
stable because the model space is genuinely concentrated
stable because ordinary restarts are missing the endpoint-sensitive direction
```

The Phase-7b stable PMP pressure family instantiates exactly that blind spot:

```text
8 / 8 stable PMP traps:
  false_closed              = true
  seed_low_range            = 0
  restart-variance action   = accept
  pressure_low_shift        >= 0.05
  optimizer_low_spread      >= 0.05
  v2 action                 = abstain
```

The pressure-noop controls show that pressure response is not merely "be nervous whenever stable":

```text
3 / 3 pressure-noop controls:
  false_closed              = false
  seed_low_range            = 0
  restart-variance action   = accept
  pressure_low_shift        = 0
  optimizer_low_spread      = 0
  v2 action                 = accept
```

## Equivalence-Pair Certificate

The receipt pairs every stable PMP trap with every pressure-noop control:

```text
8 traps x 3 controls = 24 pairs
```

For every pair:

```text
shared variance observable : seed_low_range = 0
shared baseline action     : accept
opposite false-closure label
different pressure signal
different v2 action
```

That is the clarity result. The variance baseline cannot separate the pair because, to that
baseline, the pair is identical. The pressure trace can separate it because it observes a different
quantity.

## What This Does Not Claim

This receipt does not claim:

```text
all stable false closures are detectable by pressure
pressure response is exact inference
the detector works on real KGs
the Phase-7b toy corpus is enough for product claims
restart variance is a bad baseline in every setting
```

It claims only a signal-access asymmetry on the Phase-7b fragment:

```text
stable low-variance false closure is invisible to a variance-only rule,
and visible to the pressure-response signal in the stable PMP family.
```

## Artifacts

```text
scripts/boxsel_phase7d_stable_variance_mechanism.py
scripts/test_boxsel_phase7d_stable_variance_mechanism.py
results/boxsel/phase7d_stable_variance_mechanism/manifest.json
```

## Verification

```text
python scripts/boxsel_phase7d_stable_variance_mechanism.py
python scripts/test_boxsel_phase7d_stable_variance_mechanism.py
```

Result at start:

```text
24/24 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7d mechanism receipt. Stable false closure versus variance-only detection.*
