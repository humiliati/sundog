# BoxSEL Phase 7e - Oracle-Free Recovery

**Date:** 2026-06-21  
**Status:** `RECOVERY_RECEIPT`

Phase 7e asks whether the trace can recover a boundary number, not merely trigger abstention. This
is the oracle-free recovery slice of the BoxSEL lane.

The target is the Phase-4 closed form for the Helly-seed box fragment:

```text
inf I_box^n = (9+sqrt17)/32   for every n >= 2
```

## Recovery Claim

For the Helly-seed n>=2 box fragment, an oracle-free active-set pressure trace recovers the lower
endpoint by solving:

```text
4x^2 - 9x + 4 = 0
```

choosing the feasible root:

```text
x = (9 - sqrt17)/8
```

and returning:

```text
q = 1/(4x) = (9+sqrt17)/32
```

The recovery input is a geometry/active-set trace:

```text
optimizer mode : query_pressure_extremal
dimension      : 2
active pairs   : AC, BC
slack pair     : AB
equation       : 4x^2 - 9x + 4 = 0
```

This receipt assumes that active-set trace is available. It does not prove that an ordinary
optimizer, from random initialization, will find the trace.

It does not read:

```text
I*
exact oracle labels
exact I_box endpoint
Phase-4 closed-form theorem
```

The Phase-4 theorem is used only after recovery to validate that the recovered endpoint equals the
proved closed form.

## What Was Recovered

```text
recovered x        = (9/8 + -1/8*sqrt17)
recovered endpoint = (9/32 + 1/32*sqrt17)
```

The recovered endpoint matches the observed active trace query value, and then matches the
Phase-4k closed-form validation target exactly.

## Search Contrast

Ordinary restart sampling and the earlier rational witness sit above the recovered endpoint:

```text
ordinary restart lower > recovered endpoint
rational witness q     > recovered endpoint
```

That matters because Phase 7e is not another accept/reject result. It is a recovery receipt:

```text
the pressure/extremal trace exposes enough structure to reconstruct the endpoint
that ordinary restart sampling misses
```

## Boundary

This receipt does not claim:

```text
general endpoint recovery
real-KG transfer
calibration
Ask Sundog product behavior
pressure response as exact inference
```

It claims only:

```text
on the Helly-seed box fragment, this active-set trace recovers the same lower endpoint
that Phase 4 later validates as the exact box endpoint.
```

## Artifacts

```text
scripts/boxsel_phase7e_oracle_free_recovery.py
scripts/test_boxsel_phase7e_oracle_free_recovery.py
results/boxsel/phase7e_oracle_free_recovery/manifest.json
```

## Verification

```text
python scripts/boxsel_phase7e_oracle_free_recovery.py
python scripts/test_boxsel_phase7e_oracle_free_recovery.py
```

Result at start:

```text
19/19 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-7e recovery receipt. Oracle-free trace input, closed-form validation after recovery.*
