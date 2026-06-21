# BoxSEL Phase 3 - Restart Sampler

**Date:** 2026-06-21  
**Status:** Ordinary zero-loss restart sampler built for the Helly-seed query.

## Purpose

Phase 4 closed the exact box-attainable endpoint:

```text
inf I_box^1 = 1/2
inf I_box^n = (9 + sqrt 17)/32      for every n >= 2.
```

Phase 3 adds the observed nesting layer:

```text
I_sample^{n,N} subset I_box^n subset I*.
```

The sampler deliberately uses **ordinary loss-only restarts**, not query-conditioned extremal
optimization. It samples zero-loss feasible boxes for the Helly-seed ontology:

```text
|A| = |B| = |C| = 1/2
|A and B|, |A and C|, |B and C| >= 1/4
q = P(C | A and B).
```

## What It Measures

For deterministic `dim=2`, `N=128`, `seed=314159`:

```text
I_sample = [0.5336525204919725, 1.0]
I_box    = [(9 + sqrt 17)/32, 1.0] ~= [0.4100970508005519, 1.0]
I*       = [0.0, 1.0]
```

So the sampled lower endpoint misses the exact box lower endpoint by:

```text
0.1235554696914206.
```

This is a clean search-gap receipt:

- every sampled run has zero ontology loss,
- every sampled query lies inside `I*` and inside the exact `I_box`,
- but ordinary restarts do not discover the extremal lower endpoint.

## Trace Fields

Each restart logs:

- ontology loss,
- query value,
- atom volumes,
- pairwise overlaps,
- minimum pairwise slack.

The report also exposes:

- cumulative endpoint movement as restarts accumulate,
- sampled interval,
- exact `I_box` interval,
- exact `I*` interval,
- lower search gap,
- seed-variance reports.

These are the fields Phase 6 can promote into detector features.

## Scope

This is a baseline sampler for the Helly-seed query, not a general BoxSEL optimizer. It is useful
because Phase 4 already supplies the exact target. The receipt is:

```text
zero-loss ordinary restarts can be nested and still falsely closed relative to I_box.
```

That is the empirical object the later trace detector has to learn to flag without using the oracle.

## Artifacts

- `scripts/boxsel_phase3_restart_sampler.py`
- `scripts/test_boxsel_phase3_restart_sampler.py`

Verification:

```text
python scripts/test_boxsel_phase3_restart_sampler.py
```

Result:

```text
20/20 checks pass, exit 0.
```

---

*Sundog Research Lab - BoxSEL Phase-3 restart sampler. Internal; observed I_sample layer built for
the Helly-seed query and anchored to the exact Phase-4 endpoint.*
