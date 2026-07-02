<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# PERCIVAL B1 — Separation Bake-Off (spec)

*Pre-registration. Gated on `B1_0_COURT_ADMITTED` (met 2026-06-30). D1-D4 computed;
B4 Lean anchor built.*

Status: **OPENED 2026-06-30. D1-D4 COMPUTED; B4 FORMAL ANCHOR BUILT 2026-07-01.**
B1.0 admitted the court-coordination mechanism (emergent cliff, unsmuggled, pivotal,
joint policy-induced). B1-proper asks the collection-power question the admission
deferred:

> **Over a proxy-biased base `γ`, does the un-targeting policy strictly collect what no
> `q`-quantilizer can — and *where* (in base location × base spread) does that
> separation live versus deflate?**

**What this banks and what it does not.** B1-proper banks a **conditional** separation
(cleanly conditional on `inf supp(γ) > c*`, partially present when `γ` straddles the
cliff), its base-location × base-spread boundary map, a competence frontier, and a Lean
anchor. It does **not** resolve the asterisk from B1.0: whether a *real* base is supported
past `c*` is B2 (`γ`-provenance, private). So the honest output is "un-targeting strictly
out-collects the quantilizer family in the support-above-cliff corner; straddling bases
give only a partial separation; here is the exact boundary" — the prize with its condition
made explicit, not the unconditional claim.

---

## Model (computed, deterministic — reuses the admitted `R(c)`)

- **True reward** `R(c)` — the honored/patronage probability from B1.0's court model
  (`c` = the policy's causal courting level; `R` high for `c < c*`, collapses past `c*`).
- **Proxy** `Û(c)` increasing in `c` (courting scores well on the reputation proxy); take
  `Û(c) = c`. This is the Goodhart: the proxy rewards courting, the truth punishes it past
  `c*`.
- **Biased base** `γ` — a distribution over `c` with mean `c_γ = E_γ[c] > 0` (the base
  courts a little), lower support edge `c_min = inf supp(γ)`, upper support edge
  `c_max = sup supp(γ)`, and spread `s`. The swept parameters are base location and
  spread, not mean alone. Mean-past-cliff is diagnostic; support-past-cliff is the
  clean-prize condition.
- **`q`-quantilizer** — samples the top-`q` fraction of `γ` ordered by `Û` = the upper
  `q`-tail of `γ` by `c`. Its collected reward `R̄(q) = E[R(c) | c ≥ Q_γ(1−q)]`. Because
  tilting only pushes `c` **up**, the family's least-optimizing member is `q=1`
  (`R̄(1) = E_γ[R]`) and is also its best true-reward member whenever `R` is
  nonincreasing in `c`. Quantilizing strictly hurts or ties on this court reward.
- **Un-targeting** — `c = 0` (zero causal proxy influence), collects `R(0)`. It is **not**
  a member of the quantilizer family over a biased `γ` (which is anchored at `c_γ > 0` and
  only pushes up); it must reshape off the base.

The separation, if any: `R(0) − max_q R̄(q)`.

---

## Deliverables / gates (pre-registered)

### D1 — Clean separation at a support-above operating point
At a registered operating point with `c_min > c*`: `max_q R̄(q) < ε` (every
quantilizer disgraced) **and** `R(0) > 1 − ε` (un-targeting honored). PASS
`B1_CLEAN_SEPARATION_CONFIRMED`. *Expected to pass near-immediately given the admission
and the support condition — stated honestly; the content is not here.*

### D2 — The boundary map (the real deliverable)
Sweep a base family `γ(location, spread)` and retain the full `q` profile. Register
`c_min`, `c_max`, `c_γ`, `argmax_q R̄(q)`, and the separation margin
`R(0) − max_q R̄(q)` on the grid. PASS `B1_BOUNDARY_MAPPED` iff the map shows the
three registered regions:

- **support below:** `c_max < c*` → no separation / deflation (`max_q R̄(q) ≈ R(0)`);
- **straddle:** `c_min ≤ c* ≤ c_max` → partial separation
  (`max_q R̄(q)=R̄(1)=E_γ[R] > 0`, not the clean prize);
- **support above:** `c_min > c*` → clean separation
  (`max_q R̄(q) ≈ 0`, `R(0)≈1`).

The `(a)↔(c)` boundary is therefore a location×spread surface, not the line
`c_γ=c*`. Mean-past-cliff with support below the cliff still deflates; mean-past-cliff
with straddling support is only partial.

### D3 — Competence frontier
Report `R(0) − max_q R̄(q)` as a function of support gap (`c_min-c*`) and spread
(the *price of grasping* the court imposes). Confirm un-targeting is **competent** — it
collects near-maximal true reward, not degenerate abstention — while the best quantilizer
forfeits it in the support-above region despite a high proxy score `Û`. PASS
`B1_COMPETENT` iff `R(0)` is near the reward ceiling and strictly exceeds every `R̄(q)`
at the clean support-above point.

### D4 — Deflation cell (honest null)
At `c_max < c*`, confirm **no** separation (the base is entirely below the cliff, so
`q=1` stays honored and un-targeting has no edge). This is not a failure; it confirms
the 2×2's (a) cell and fixes the claim as *conditional*. Also report the straddle band
as a partial-separation regime, not a clean prize. Reported, not gated to pass/fail.

### B4 — Lean anchor
Machine-check two finite/discrete statements (axiom-audited, gated, `sundogcert`).
Implemented in `C:\Users\hughe\Dev\sundogcert\Sundogcert\Percival.lean`:

1. **Best quantilizer is the base.** For a nonincreasing `R` and a quantilizer family
   that takes upper `c`-tails, `R̄(q) ≤ R̄(1)=E_γ[R]` for every `q`. Quantilizing cannot
   improve true reward on this court reward; it only pushes farther into courting.
2. **Clean support-above separation.** If `R(c)=0` for all support points `c>c*`,
   `R(0)>0`, and `γ` is supported on `(c*,1]`, then every `q`-quantilizer collects `0`
   while un-targeting collects `R(0)>0`; hence `sup_q R̄(q) < R(0)`.

The support-above theorem is the keeper. The best-quantilizer lemma is the map's
load-bearing explanation and also covers the straddle regime: the best family member is
`q=1`, so the partial margin is `R(0)-E_γ[R]`.

Lean names:

- `Sundogcert.Percival.best_quantilizer_is_base_three`
- `Sundogcert.Percival.clean_support_above_separation_three`

Both compile with guarded axiom receipts inside the standard foundational triple
`[propext, Classical.choice, Quot.sound]`.

---

## Falsifiers (any one repairs or kills B1-proper)

- **Un-targeting not competent** — honored but the honor is worthless (`R(0)` near the
  floor): the separation is vacuous.
- **Support-floor fails** — some `q`-quantilizer stays honored when `c_min > c*`: the
  family is not uniformly disgraced, the out-of-family logic is wrong.
- **Spurious clean separation** — a clean separation appears when `c_max < c*`: the
  boundary is not at the cliff, the model is mis-specified.
- **Mean-only overclaim** — the writeup treats `c_γ > c*` as sufficient for the clean
  prize despite straddling support. This does not kill the mechanism, but it voids the
  clean-claim language until recut to location×spread.

---

## Out of scope

- **B2 `γ`-provenance** — whether a real base is supported above `c*`, merely straddles
  it, or sits below it. B1-proper isolates this question; it does not answer it. The
  clean prize stays conditional until B2.
- **Multi-agent pivotality** — one knight in a shared court (`O(1/N)`), flagged at B1.0.
- **Learned-policy-finds-it** — whether gradient descent over the base reaches un-targeting
  or gets trapped courting (the NS-4 SB-4 analog); a separate probe, not core B1.

---

## Cross-links

- Admitted mechanism: [`PERCIVAL_B1_0_COURT_COORDINATION_ADMISSION.md`](PERCIVAL_B1_0_COURT_COORDINATION_ADMISSION.md), [`PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md`](PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md).
- Lane: [`SUNDOG_V_PERCIVAL.md`](../SUNDOG_V_PERCIVAL.md) (Track B). Umbrella target row: [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
