# BoxSEL Phase 7c - External Review Packet

**Date:** 2026-06-21  
**Status:** `READY_FOR_EXTERNAL_REVIEW`

Phase 7c is the review ask. It does not add a new detector run and it does not promote the
Phase-7b result beyond the toy micro-SEL boundary.

## Review Claim

```text
On a locked tiny role-free micro-SEL corpus, the frozen oracle-free Phase-7b v2
GeneralTrace detector accepted 0/16 false-closure traps, while the locked
restart-variance baseline accepted 16/16.
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

The review is therefore not "is this impressive?" The review is:

```text
Is this a clean, leakage-free, reproducible toy demonstration of trace-based false-closure gating,
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
Is restart_variance_only_v0 an adequate first comparator, and what stricter oracle-free
baseline should be required before promotion?

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

P7C-O2 phase7d_required
Reviewer accepts the packet shape but requires a stricter baseline, more controls, or a larger
registered corpus before any stronger claim.

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
```

The manifest records:

```text
primary review claim
non-claims
Phase-7 failure metrics
Phase-7b pass metrics
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
21/21 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7c external-review packet. Review-ready toy claim only.*
