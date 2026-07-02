<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# PERCIVAL B2 — γ-Provenance (spec, PRIVATE)

> **DECLASSIFIED 2026-07-01 by owner decision.** Published and licensed under the
> manifest-scoped Percival Apache-2.0 grant (`docs/percival/LICENSE.md`). The PRIVATE
> markers below are the historical pre-registration register, retained verbatim. The
> claim boundary is unchanged: this document characterizes structure over synthetic
> bases; nothing in it asserts a measured real base.

*Pre-registration. The make-or-break for the unconditional prize. Not run. Kept
internal — this phase characterizes structure, it does not assert a measured real base.*

Status: **OPENED 2026-06-30. PRE-REGISTERED, NOT RUN. PRIVATE.** B1-proper banked a
**conditional** separation: un-targeting strictly out-collects the quantilizer family
when `inf supp(γ) > c*`. B2 asks whether that condition is reachable in a realistic
setting — the only thing standing between "conditional" and "prize."

> **At the fixed admitted cliff, where does the clean prize
> (`inf supp(γ) > c*_0`) live across base provenance — and how strict would the court
> have to become to move that boundary?**

Primary headline discipline: **fix `c* = 0.25`**, the admitted B1.0 operating point
(`B=1, K=3`). Court strictness is not a free prize knob. A secondary strictness sweep may
report the *purism bill* — how small `c*` must be before a base's low tail clears it —
but the verdict cannot be won by moving the court cliff after the fact.

This matters because `c*` is the court's cliff (`= B/(B+K)`, a payoff ratio), not a fact
of the base. If `c*` is allowed to drift toward zero, the clean prize can always be
manufactured by making the court purist enough. B2's primary question is narrower and
non-manufacturable:

> **At the fixed admitted cliff `c*_0 = 0.25`, does a realistic/trusted base have
> `inf supp(γ) > c*_0`, or does it straddle?**

---

## Honest prior (stated before building): this probably deflates

The clean prize needs `inf supp(γ) > c*` — the base's *least*-grasping action must
already trip the court. Two ways to get there, both away from the realistic center:

1. **A corrupt / uniformly-grasping base** — every base action courts past the cliff.
   But a `q`-quantilizer's `γ` is supposed to be a **trusted** base. The registered
   trust prior is not a theorem; it is the weaker claim that a trusted base should
   include restrained, low-courting behavior (mass near `c=0`). Under that prior, a
   trusted base has `inf supp(γ) ≈ 0`, not `> c*`. **The clean prize requires either a
   base the quantilizer framework would not call trusted, or a controversial notion of
   trust without restraint.**
2. **A purist court** — set `c* ≈ 0` so the court disgraces *any* courting whatsoever.
   Romance-faithful (the Grail demands total purity), but it is an intolerant-oversight
   assumption, not a neutral one.

So the realistic center — a **trusted base** (includes restraint) under the admitted
tolerant court (`c*_0 = 0.25`) — sits in the **straddle**, where only the *partial* separation
survives (`margin = R(0) − E_γ[R]`). The clean unconditional prize is a **corner**
(corrupt base and/or purist court). B2's expected verdict is therefore that the
unconditional prize is **not won on trusted bases**; what is banked is the partial
separation, the counterproductivity of quantilizing (lemma 1), and the trust↔separation
tradeoff. The irony is load-bearing and worth stating plainly: **demanding that the base
be trustworthy is exactly what caps the separation at partial** — un-targeting's *clean*
edge appears only in a fallen court.

---

## Model (computed sensitivity map — reuses `R(c)` and the B1-proper bake-off)

Primary fixed-cliff model:

- **Fixed admitted cliff** — `c*_0 = 0.25` from B1.0 (`B=1, K=3`). All headline
  pass/fail branches use this cliff. The B1.0 court is the object under test; B2 does not
  retune the court to win.
- **Base provenance** — parameterize `γ` by lower support edge `c_min = inf supp(γ)`,
  upper support edge `c_max = sup supp(γ)`, spread, and fixed-cliff restraint fraction
  `ρ_0 = P_γ(c < c*_0)`. `ρ_0 = 0` is uniformly above the admitted cliff; `ρ_0 > 0`
  means the base contains restrained/honored mass and therefore cannot satisfy the
  literal clean-prize condition. `ρ_0` is a diagnostic; `c_min` is the literal clean-gate
  variable.

Secondary context map:

- **Court strictness / purism bill** — sweep `c*` via the payoff ratio `B/K` only after
  the fixed-cliff verdict is recorded. Report the threshold `c*_needed < c_min` required
  to move a base into support-above. This contextualizes "a stricter court would make it
  clean" without letting that become the claim.

Trust language discipline:

- Do **not** define trust as a theorem. Use the weaker registered prior:
  a quantilizer-trusted base should include restrained, low-courting behavior
  (`ρ_0 > 0`, typically `c_min ≈ 0`). If a proposed "trusted" base has
  `c_min > c*_0`, that is not impossible by definition, but it must be named as the
  controversial positive surprise: trust without restraint.
- Literal `inf supp` is fragile to dust mass. Report both `c_min` and `ρ_0`; do not call
  the clean prize if any non-negligible below-cliff mass remains. If a later empirical
  version needs robustness, it must pre-register an `α`-quantile floor separately.

For each base family at `c*_0`, compute the region (support-below / straddle /
support-above) and the separation margin `R(0) − E_γ[R]`. For the secondary strictness
map, compute the purism bill `c*_needed`.

---

## Deliverables (pre-registered)

### D-B2.1 — Fixed-cliff provenance map
At `c*_0 = 0.25`, sweep base location/spread. Show the clean-prize region
(`inf supp > c*_0`), the straddle band, and the deflation region. PASS
`B2_FIXED_CLIFF_MAPPED` iff the three regions appear and the clean region is confined to
uniformly-above-cliff bases, with the registered realistic center (restraint-admissible
trusted base, `ρ_0 > 0`) in the straddle.

### D-B2.2 — The trust↔separation tradeoff
Plot the separation margin as a function of the fixed-cliff restraint fraction `ρ_0`.
Confirm the tradeoff is **monotone decreasing**: more restraint (more trust) ⟹ less
separation; the clean margin `≈ R(0)` is reached only at `ρ_0 ≈ 0`. PASS
`B2_TRADEOFF_MONOTONE` iff margin decreases in `ρ_0` and hits the clean ceiling only at the
untrusted corner.

### D-B2.3 — Court strictness context / purism bill
Sweep `c*` only as context. For each base family, report the `c*` range that would make
it support-above, straddle, or support-below, plus the purism bill `c*_needed < c_min`.
This deliverable is descriptive only. It cannot upgrade a fixed-cliff null to a clean
prize.

### Verdict
```
B2_CLEAN_IS_CORNER            (expected)  at fixed c*_0=0.25, clean prize only for
                                          uniformly-above-cliff/corrupt bases;
                                          realistic trusted center is straddle → partial only.
B2_CLEAN_REACHABLE_REALISTIC  (would be   clean prize holds for a plausibly trusted base under a
   the positive surprise)                 fixed admitted cliff — trust without restraint.
```

Either way, the partial separation and lemma-1 counterproductivity are banked; the
question B2 settles is only whether the *clean* prize escapes the corner.

---

## Claim boundary (PRIVATE — the whole point of keeping this internal)

**Will say (internal):** where the clean prize lives at the fixed admitted cliff; that
the realistic center is straddle/partial; the trust↔separation tradeoff; and, as
secondary context only, the purism bill required to make a given base clean by moving
`c*`.

**Will not say:** that any *measured* real base sits past `c*` (that is a separate, large,
ill-posed empirical undertaking we are not doing); that real oversight is purist or real
bases are corrupt (named, not asserted); anything that manufactures an empirical
provenance claim from a structural map; that a strictness sweep can rescue the primary
fixed-cliff verdict. B2 characterizes the *structure* of the condition, not the world.

---

## Falsifiers

- **Realistic-center clean separation** — if a plausibly trusted base (`ρ_0` clearly
  `> 0` under the fixed admitted cliff) shows a clean margin, the trust↔separation
  tension is wrong and the prize may be unconditionally reachable (the positive surprise).
- **Non-monotone tradeoff** — if margin does not decrease in restraint `ρ_0`, the model is
  mis-specified.
- **Court-knob smuggling** — if the writeup upgrades the secondary strictness sweep into
  the headline result, the B2 verdict is void until recut to the fixed admitted cliff.
- **Provenance overclaim** — any language asserting a measured real base's location; voids
  the private-claim discipline until recut to structure-only.

---

## Cross-links

- Conditional result B2 gates: [`PERCIVAL_B1_SEPARATION_SPEC.md`](PERCIVAL_B1_SEPARATION_SPEC.md), [`PERCIVAL_B1_SEPARATION_RESULTS.md`](PERCIVAL_B1_SEPARATION_RESULTS.md).
- Lane (Track B, B2 is marked PRIVATE): [`SUNDOG_V_PERCIVAL.md`](../SUNDOG_V_PERCIVAL.md).
- Umbrella target row: [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
