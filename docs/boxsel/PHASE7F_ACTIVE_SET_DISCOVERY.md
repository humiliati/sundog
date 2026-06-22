# BoxSEL Phase 7f - Active-Set Discovery

**Date:** 2026-06-21  
**Status:** `DISCOVERY_RECEIPT`

Phase 7f is an active-set discovery receipt. Phase 7e showed that an
oracle-free active-set pressure trace can recover the
Helly-seed lower endpoint:

```text
q = (9+sqrt17)/32
```

But Phase 7e still assumed the active set was already named. Phase 7f removes
that assumption for the recorded KKT trace.

## Discovery Claim

Starting only from raw box intervals, the Phase-7f receipt computes exact box
volumes and residuals against the ontology pair target:

```text
target pair lower bound = 1/4
```

The discovered residual pattern is:

```text
AC residual = 0       active
BC residual = 0       active
AB residual > 0       slack
```

From that discovered active set and the raw structured geometry, the trace derives:

```text
z = 2(1-x)
z = x / (2(1-x))
```

and therefore:

```text
4x^2 - 9x + 4 = 0
```

The discovered trace is then handed to the Phase-7e recovery rule, which returns:

```text
(9+sqrt17)/32
```

The discovery input does not include:

```text
active-pair labels
the KKT equation
I*
the exact I_box endpoint
Phase-4 closed-form theorem
oracle labels
```

## Negative Control

The earlier rational 2-D witness is deliberately rejected by the same discovery
path. Its raw residuals discover only:

```text
AC active
AB slack
BC slack
```

so it does not derive the KKT active equation and cannot be converted into the
Phase-7e recovery trace.

That matters because Phase 7f is not simply hard-coding "any good-looking
2-D witness recovers the endpoint." It distinguishes the closed-form KKT trace
from a nearby suboptimal witness using observable geometry.

## Boundary

This is still a toy micro-SEL Helly-fragment receipt. It claims:

```text
raw box intervals -> residual active set -> KKT equation -> Phase-7e recovery
```

for the recorded KKT trace.

It does not claim:

```text
general active-set learning
global optimization
ordinary random restarts find the trace
exact SEL inference
real-KG transfer
Ask Sundog product behavior
```

## Artifacts

```text
scripts/boxsel_phase7f_active_set_discovery.py
scripts/test_boxsel_phase7f_active_set_discovery.py
results/boxsel/phase7f_active_set_discovery/manifest.json
```

## Verification

```text
python scripts/boxsel_phase7f_active_set_discovery.py
python scripts/test_boxsel_phase7f_active_set_discovery.py
```

Result at start:

```text
22/22 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7f discovery receipt. Raw box intervals first;
active-set recovery second; closed-form validation only after recovery.*
