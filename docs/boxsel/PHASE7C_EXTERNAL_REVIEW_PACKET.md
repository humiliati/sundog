# BoxSEL Phase 7c - External Review Packet

**Date:** 2026-06-21  
**Status:** `READY_FOR_EXTERNAL_REVIEW`

Phase 7c is the review ask. It does not add a new detector run and it does not promote the
Phase-7b result beyond the toy micro-SEL boundary.

## Review Claim

```text
On the locked tiny role-free micro-SEL fragment, restart-variance-only detection is
structurally blind to stable false closure because it observes only seed_low_range;
the stable PMP traps and pressure-noop controls share seed_low_range=0 while
pressure-response traces separate the labels.
```

That is the whole claim being sent for review. It is not a real-KG claim, not a calibration
guarantee, not an Ask Sundog product claim, and not a claim that query pressure is exact
inference.

## Why It Is Worth Review

Phase 7 failed in the right way:

```text
Phase 7 status                         : FAIL_PREREG_GATE
detector accepted false closures        : 4 / 10
restart-variance baseline accepted      : 4 / 10
triggered kills                         : KILL7-1, KILL7-2
missed family                           : stable low-loss PMP-shaped false closure
```

Phase 7b then froze a new schema and detector before the held-out run:

```text
schema          : phase6b_general_trace_schema_v1
detector        : phase7b_v2_trace_detector_v1
thresholds      : phase7b_v2_thresholds_v1
baseline        : restart_variance_only_v0
prereg status   : LOCKED_NOT_RUN
result rows     : none in prereg
```

The run passed:

```text
Phase 7b status                         : PASS_PREREG_GATE
detector accepted false closures        : 0 / 16
restart-variance baseline accepted      : 16 / 16
true-narrow controls accepted           : 9 / 9
loss-escape controls accepted           : 0 / 3
stable-PMP pressure warnings            : 8 / 8
baseline stable-PMP pressure warnings   : 0 / 8
```

Phase 7d extracts the mechanism from those numbers:

```text
restart_variance_only_v0 observes only seed_low_range
stable PMP traps       : seed_low_range = 0, false_closed = true,  baseline accepts, v2 abstains
pressure-noop controls : seed_low_range = 0, false_closed = false, baseline accepts, v2 accepts
equivalence pairs      : 24 trap/control pairs with identical variance observable and opposite labels
```

The review is therefore not "is this benchmark impressive?" The review is:

```text
Is this stable/variance mechanism receipt clean, leakage-free, and semantically aligned,
and what exact blocker remains before any wider claim?
```

## Review Questions

```text
P7C-Q1 leakage
Does any detector feature, threshold, or action read exact endpoints, oracle labels,
Phase-4 closed forms, or evaluator-only fields?

P7C-Q2 heldout
Are Phase-7 diagnostic rows and seeds excluded from Phase-7b held-out validation?

P7C-Q3 baseline
Is the Phase-7d stable/variance dichotomy correct: does restart_variance_only_v0 truly
observe only seed_low_range, making stable false closure a blind spot by construction?

P7C-Q4 pressure
Is query pressure a legitimate observable trace, or is it too close to extremal inference
for the intended abstention claim?

P7C-Q5 semantic_alignment
Do the exact finite-counting labels, PMP cases, Helly cases, and support-floor cases align
with the BoxSEL volume semantics they are meant to stress?

P7C-Q6 controls
Are the true-narrow, pressure-noop, and loss controls enough to rule out an always-abstain
or pressure-hypersensitive detector?

P7C-Q7 scope
Does every outward sentence preserve the toy micro-SEL boundary and the failed Phase-7 history?
```

## Break Conditions

The Phase-7b toy claim should be withdrawn or redesigned if review finds any of:

```text
detector decisions depend on evaluator-only truth
Phase-7 seen rows are reused as held-out evidence
query pressure smuggles in the exact endpoint
finite-counting labels are mismatched with the intended geometric-volume semantics
a trivial abstention or pressure-only rule matches the result while preserving controls
the packet implies real-KG transfer, calibration, product behavior, or a retroactive Phase-7 pass
```

## Possible Review Outcomes

```text
P7C-O1 toy_claim_review_pass
Reviewer accepts only the bounded toy micro-SEL claim and finds no leakage or semantic-label break.

P7C-O2 followup_gate_required
Reviewer accepts the packet shape but requires a stricter baseline, recovery test, more controls,
or a larger registered corpus before any stronger claim.

P7C-O3 claim_withdrawn_or_redesigned
Reviewer finds leakage, semantic mismatch, or overclaim sufficient to withdraw the Phase-7b
detector claim.
```

## Packet Manifest

Machine-readable packet:

```text
results/boxsel/phase7c_external_review_packet/manifest.json
```

Generator and guard:

```text
scripts/boxsel_phase7c_review_packet.py
scripts/test_boxsel_phase7c_review_packet.py
scripts/boxsel_phase7d_stable_variance_mechanism.py
scripts/test_boxsel_phase7d_stable_variance_mechanism.py
```

The manifest records:

```text
primary review claim
non-claims
Phase-7 failure metrics
Phase-7b pass metrics
Phase-7d stable/variance mechanism summary
frozen detector metadata
leakage/boundary audit booleans
review questions and possible outcomes
SHA-256 hashes for review artifacts
```

## Verification

```text
python scripts/boxsel_phase7c_review_packet.py
python scripts/test_boxsel_phase7c_review_packet.py
```

Result at start:

```text
25/25 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7c external-review packet. Review-ready toy claim only.*
