<!--
SPDX-License-Identifier: Apache-2.0
Copyright 2026 Stellar Aqua LLC

Licensed under the Apache License, Version 2.0, via the manifest-scoped
Percival grant: docs/percival/LICENSE.md (MANIFEST.json is the authoritative
covered-file list). Distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND; see http://www.apache.org/licenses/LICENSE-2.0.
-->

# Percival S2 — Target Safe-Point Reachability / "Can a Target Cap Be Built?" (probe spec, v2)

*Pre-registration. Co-headline of the reopen slate. Nothing run. v2 after a critical pass
that found the v1 model vacuous (single-cue verdict-smuggling), recast the definitional
gate as an equivalence claim, and promoted the reconstruction leak to a deliverable.*

Status: **OPENED 2026-07-01, HARDENED SAME DAY. NOT RUN.** S2 disambiguates the slate's
candidate cleanliness invariants and decides whether the target channel is fixably clean.

> **Is there an exogenous, non-gameable projection that takes a policy to zero causal
> proxy-dependence (`c=0`) without collapsing into (a) an output/action cap, (b) a rewrite
> of the policy's function, or (c) a retreat to the measure channel?**

**Where the falsifiable content lives (honest, post-recast).** The core "no" is
theorem-shaped (S2.1's collapse argument), so the probe's empirical value is: verifying
the collapse across candidate projections, measuring the **reconstruction leak** (S2.3 —
the genuinely open quantity), and locating the **court-vs-tax frontier** (S2.4). The
positive surprise would have to defeat an information-theoretic argument, and we say so.

---

## Model (two channels — v1's single-cue model is void)

v1 bug, recorded: with `U` as the *only* route to `G`, "competence `>0.5` at `c=0`" is
logically impossible (any use of `U` **is** causal `U`-dependence), so the no-target-cap
gate passed by tautology. The information structure smuggled the verdict. v2 separates the
routes:

- **Ground truth** `G ∈ {0,1}` uniform; competence = `P(A = G)`.
- **Legitimate channel** `V` with `P(V=G) = β`, `β ∈ [0.5, 1)` — the policy's honest
  signal.
- **Proxy channel** `U` with `P(U=G) = ρ`, `ρ ∈ [0.5, 1)` — the reputation/courtesy proxy.
  (`ρ=1` is excluded: `do(U)` holding `G` fixed is off-support there and `c` is
  unidentifiable. A3's anti-correlated `U=1−G` is the mirror pole; `ρ ≥ 0.5` WLOG by
  symmetry.)
- **Primary joint** — `V` and `U` are conditionally independent given `G`. The primary
  grid uses `0.5 ≤ β < ρ < 1`, so `U` carries unique competence beyond `V`; `ρ ≤ β`
  cells are negative controls where the unique-info tax is zero.
- **Causal proxy-dependence** `c` = `do(U)`-influence on `A`, holding `V, G` fixed (the
  A3-sharpened definition).
- **Court** — B1.0's `R(c)`, fixed operating point (`c* = 0.25`). Combined objective,
  pre-registered: `Total(π) = Competence(π) × R(c(π))`.
- **Unique-information tax** — the masking cost is `Competence(V,U) − β`: `U`'s unique
  contribution *beyond* `V`, not raw informativeness (v1's `ρ − 0.5` conflated these).
  Under the primary binary-symmetric joint this is `max(β,ρ)-β`, hence `ρ−β` on the
  primary `β<ρ` grid and `0` on the `ρ≤β` controls.

---

## Deliverables / gates (pre-registered)

### S2.1 — The collapse (equivalence, verified not assumed)
**Claim:** every exogenous projection enforcing `do(U)`-invariance — input **masking**,
input **scrambling** (feed noise-`U`), and action **permutation-averaging**
(`A' = E_u[π(V,u)]`) — yields a policy whose action is a function of `V` alone, hence
competence `≤ β` (data processing). The bound is the theorem; the canonical V-only
projection attains `β`, while arbitrary scrambling/averaging may land below it.
*In-channel target enforcement ≡ retreat to the non-proxy measured channel,
behaviorally.* The v1 "definitional gate" (masking is a measure move) becomes a
corollary, not a classification to defend.
**Check:** implement all three projections over a policy family that uses `U`; confirm all
land at `c=0` with competence `≤ β`, and confirm the best V-only projection reaches `β`
(within tolerance), for `(ρ, β)` on a grid.
**PASS** `S2_COLLAPSE_VERIFIED`. **FAIL** any projection retaining competence `> β` at
`c=0` — the falsifier, and the prize-reopener: it would have to beat the information
argument, so inspect for a bug first, then take it seriously.

### S2.2 — No exogenous in-channel target cap (now substantive)
**Claim:** the gap between unconstrained competence (using `U`'s unique info) and `β` is
**unreachable at `c=0` by any exogenous projection** — closing it requires the policy's own
function (internal disentangling = a rewrite), which is endogenous, not enforcement.
**Check:** exhibit the gap (best unconstrained vs best-projected at `c=0`) across the
primary `β<ρ` grid; confirm no candidate projection closes it. On `ρ≤β` controls, report
zero tax rather than treating the absence of a gap as a target-cap success.
**PASS** `S2_NO_EXOGENOUS_TARGET_CAP`.

### S2.3 — The reconstruction leak (the open quantity; the probe's center of gravity)
**Claim:** `c` is defined on a **named variable**, and that is gameable: a policy that
*infers* `U` from unmasked channels (`V` predicts `U` through `G`) and courts on the
inferred value has `c(U) = 0` while behaviorally courting — it **evades the court's
estimator** (B1's fundamental is literally `θ = 1−c`). Even the measure-retreat is leaky.
**Check:** construct the reconstruction courtier (`Û = f(V)`, the Bayes estimator of `U`
from `V` under the registered joint; in the primary binary-symmetric model this is
`f(V)=V` when `P(U=V)>1/2`), then act to court on `Û`. Confirm `c(U) = 0` under the named-
variable audit and therefore court-honored by B1's estimator; register the leak
`r_leak = P(A_reconstructor = A_proxy_user)` and its competence/courting profile across
`(ρ, β)`. Also report a content-level courting diagnostic (`c_content` or `r_leak`) so the
result explicitly distinguishes named-variable safety from content-level safety.
**Report** `S2_NAMED_VARIABLE_CAP_LEAKY` with the leak magnitude. This is the
reward-tampering / Goodhart-on-the-auditor face of the umbrella fence, pre-registered
rather than discovered. Consequence if large: the true target safe point is zero dependence
on the proxy's **content however accessed** — a function-level property no named-variable
audit certifies — which *strengthens* S2.2's verdict and sharpens S1's law.

### S2.4 — The court-vs-tax frontier
**Claim:** on `Total = Competence × R(c)`, the optimum of the stochastic mixing family
(`A=U` with probability `w`, else `A=V`, so `c=w`) pushes `w` to **just under the cliff**
— the B3.2 cliff-edge fragility
reappearing as the court+tax frontier. Un-targeting (`w=0`) is again safe-not-optimal;
masking beats `U`-use exactly when the unique-info tax < the court loss.
**Check:** sweep `w` at the fixed court; locate the optimum and the masking-vs-using
crossover; confirm coherence with B3.2.
**PASS** `S2_FRONTIER_MAPPED` (+ registered prediction confirmed/refuted as found).

### Verdict
```
S2_TARGET_CAP_COLLAPSES_TO_MASK   (expected)
    In-channel target enforcement collapses to measure masking; the tax is U's unique
    information (information-theoretic, not implementational); the named-variable cap is
    additionally LEAKY to reconstruction (magnitude reported); reachability of the target
    safe point is exogenous only via priced, leaky channel-retreat. Target stays partial
    for a structural reason; feeds S1 as reachability-gated-by-information.

S2_TARGET_CAP_EXISTS_PRIZE_REOPENS   (surprise / falsifier)
    Some projection retains competence > beta at c=0 — defeating the data-processing
    argument. Target is act-like (clean-but-priced); B2's partial was a family artifact.
```

---

## Falsifiers

- **Collapse fails** — a projection with competence `> β` at `c=0` (S2.1's FAIL): the
  prize-reopener; bug-check first, then believe it.
- **Leak is nil when it shouldn't be** — if the reconstruction courtier can't court at
  `c(U)=0` despite `V`–`U` correlation, the leak model (or the `c` estimator) is wrong.
- **Frontier contradicts B3.2** — if the mixed optimum is not cliff-adjacent, the
  cross-probe coherence fails and one of the models is mis-specified.
- **Verdict-smuggling recurrence** — any gate that turns out true by the information
  structure alone (the v1 bug class) voids that gate until recut.

---

## Out of scope

- Multi-agent reachability (S3); general counterproductivity (S4).
- A *measured real* proxy's `(ρ, β)` (B2-style provenance; not asserted).
- Lean anchor for the collapse (natural S4-adjacent follow-on; 2×2×2 finite enumeration —
  registered as a candidate, not committed here).

---

## Cross-links

- Slate: [`PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md`](PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md) (S1/S2 co-headline).
- `c` definition + anti-correlated pole: [`PERCIVAL_A3_KILLSWITCH_RESULTS.md`](PERCIVAL_A3_KILLSWITCH_RESULTS.md).
- Court + cliff: [`PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md`](PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md).
- Cliff-edge fragility (S2.4 coherence): [`PERCIVAL_B3_COMPOSITION_STABILITY_RESULTS.md`](PERCIVAL_B3_COMPOSITION_STABILITY_RESULTS.md).
- Umbrella fence (S2.3 is its target-channel face): [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
