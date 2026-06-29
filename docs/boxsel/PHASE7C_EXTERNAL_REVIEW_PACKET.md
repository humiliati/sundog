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

Phase 7e is CONDITIONAL recovery, not blind recovery. GIVEN a config whose active set is already
exposed, the lower endpoint is reconstructible from oracle-free geometry alone:

```text
input (assumed held)   : active-set / pressure trace -- AC, BC active, AB slack
reconstruction         : 4x^2 - 9x + 4 = 0  ->  x = (9 - sqrt17)/8  ->  q = (9 + sqrt17)/32
reads no oracle        : not I*, not exact labels, not the Phase-4 theorem
                         (the theorem is used only to VALIDATE, after recovery)
validation             : matches the Phase-4 closed form exactly
NOT claimed            : that ordinary restarts FIND this config -- that is the open
                         search gap (Phase 4c / 7c), not something 7e closes
```

So 7e says the endpoint LIVES IN observable geometry; it does not say search can REACH it.

Phase 7f removes 7e's NAMED-active-set assumption (it discovers the active set from raw residuals)
but KEEPS the held-config assumption:

```text
input                  : raw box intervals of a GIVEN config (optimal or near-miss)
discovered active set   : AC = 0, BC = 0 residual -> active;  AB > 0 -> slack  (no labels supplied)
derived equation        : 4x^2 - 9x + 4 = 0  ->  handed to the 7e recovery rule
negative control (real) : the rational witness discovers only AC active -> cannot derive the
                          equation -> correctly NOT recovered (rules out "any nice witness recovers")
still NOT claimed       : finding the optimal config from random restarts (the open search gap)
```

7f's load-bearing piece is the negative control: observable residuals alone separate the
KKT-optimal config from a near-miss. It is not hard-coded to the answer.

The headline under review is the Phase-7d signal-access asymmetry -- a mechanism, not a benchmark:
24 trap/control pairs identical to the variance baseline yet opposite in truth. Phase 7e/7f are
SECONDARY and CONDITIONAL: they show the endpoint is reconstructible from oracle-free geometry
GIVEN the optimal config -- they do not claim search can find that config.

The review is therefore:

```text
Is the 7d asymmetry clean and leakage-free; and do 7e/7f stay honestly conditional
(no blind-recovery overclaim); and what exact blocker remains before any wider claim?
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

P7C-Q8 recovery_conditionality
Do Phases 7e/7f anywhere imply BLIND recovery? Confirm both presuppose a held config
(7e: a named active set; 7f: a given config's raw intervals), and that neither claims ordinary
restarts find it -- i.e. the recovery must not silently close the search gap it depends on.
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
Phase 7e/7f are presented as recovering the endpoint without already holding the optimal config
  -- i.e. the open search gap is smuggled shut
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
scripts/boxsel_phase7e_oracle_free_recovery.py
scripts/test_boxsel_phase7e_oracle_free_recovery.py
scripts/boxsel_phase7f_active_set_discovery.py
scripts/test_boxsel_phase7f_active_set_discovery.py
```

The manifest records:

```text
primary review claim
non-claims
Phase-7 failure metrics
Phase-7b pass metrics
Phase-7d stable/variance mechanism summary
Phase-7e oracle-free recovery summary
Phase-7f active-set discovery summary
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
32/32 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7c external-review packet. Review-ready toy claim only.*
