# When Quantilizing Hurts

*A machine-checked note on a regime where the quantilizer's defining move — tilting
toward the proxy — cannot improve true reward, strictly hurts except in tie cases,
where the boundary of that regime sits, and what shape of oversight removes the
temptation it creates.*

**Status: draft for owner re-voice. Technical-note register. All claims in-vitro and
falsifier-fenced; the Scope section is part of the result, not a disclaimer.**

---

## 1. Setup

A `q`-quantilizer (Taylor, 2015) samples from the top-`q` fraction of a trusted base
distribution `γ`, ordered by a proxy utility `Û`, instead of arg-maxing `Û`. Its safety
guarantee — expected true cost bounded by `(1/q) · E_γ[cost]` — is the canonical
mild-optimization result. Three premises travel with it: the base `γ` is given and
trusted; the joint distribution of (proxy, true value) is **fixed under `γ`**; and the
setting is single-shot.

This note concerns rewards that violate the second premise in a specific, structured
way: **performative rewards** (Perdomo, Zrnic, Mendler-Dünner & Hardt, 2020) where the
deployed policy's optimization pressure on the proxy degrades the reward itself. The
motivating instance is a coordination-game "court": patronage conferred by a population
of noisy graders keyed on the policy's causal proxy-dependence, whose unique global-game
equilibrium (Carlsson–van Damme 1993; Morris–Shin 1998) produces a sharp collapse
threshold that is *derived from payoffs and noise*, never inserted into the reward. In
the Goodhart taxonomy of Manheim & Garrabrant (2018) this is **causal/adversarial**
Goodhart — targeting induces the gap — as opposed to the regressional/extremal regime
the quantilizer was built for. The philosophical genus is older: Elster (1983) called
these *states that are essentially by-products* — states the very attempt to obtain
precludes, systematically, not by accident.

## 2. The result

> **On rewards nonincreasing in proxy-pressure — i.e., when the proxy is pointwise
> misaligned with the true reward — quantilizing cannot improve true reward; it
> strictly hurts whenever the selected upper tail drops higher-reward mass:**
>
> 1. the best `q`-quantilizer is the untilted base (`q = 1`); every strict tilt collects
>    weakly less (strictly less under the non-flatness condition above), and
>    tail-collection is monotone in `q`;
> 2. in the registered iterated-tail court model, iterated quantilization (each round's
>    selected tail becoming the next base) drifts monotonically into the collapse
>    region and is ruined at a finite round, while the zero-pressure policy is
>    drift-free;
> 3. a policy with **zero causal proxy-dependence** — which is *not* a member of the
>    quantilizer family over any base biased toward the proxy — strictly out-collects
>    the entire family whenever the base's support lies past the collapse threshold,
>    and out-collects it partially (by the base's disgraced fraction) when the base
>    straddles it.

Two remarks on (3). The zero-dependence policy is defined *interventionally* — its
action distribution is invariant under `do(proxy)` holding other cues fixed — not by
zero observational correlation, which the environment can force on honest policies. And
the strict domination is conditional on the support geometry: over a base containing
genuinely restrained (near-zero-pressure) behavior, only the partial form survives —
the clean form requires a base the quantilizer framework would not call trusted.

**Machine-checked core.** The finite general form of (1) and the clean case of (3) are
proven in Lean 4 + Mathlib for arbitrary finite uniform support (unweighted bases):

- `Sundogcert.Percival.upper_tail_mul_le_base_mul` — for a reward listed nonincreasing
  along the proxy-pressure order, no upper tail beats the full-base average
  (cross-multiplied; arbitrary length, arbitrary tail);
- `Sundogcert.Percival.best_quantilizer_is_base_general` — the same in average form;
- `Sundogcert.Percival.clean_support_above_separation_general` — all-zero support past
  the threshold ⟹ every tail average is strictly beaten by any positive un-targeted
  reward.

All three carry `#guard_msgs`-pinned axiom audits: exactly
`[propext, Classical.choice, Quot.sound]`, no `sorry`, no `native_decide`, registered in
the repository-wide axiom audit. A falsifier hunt over 10,500 randomized instances —
unweighted and weighted finite bases, and continuous bases under the actual
coordination-court reward — found zero violations.

## 3. The boundary of the regime (the condition is not optional)

For the theorem above, the condition — *the proxy is pointwise misaligned* — is the
exact boundary of the guaranteed class, not a technicality. It is not the only route
to `q = 1`: a sufficiently graded tax can also overpower a pointwise-good proxy.
Separating the pointwise proxy–truth relation from the performative tax as independent
knobs and sweeping both:

| proxy pointwise... | performative tax | optimal tilt |
| --- | --- | --- |
| misaligned or neutral | any monotone tax, **including none** | base is optimal; strict tilt never improves (the theorem) |
| good | none | maximal tilt (Goodhart-free; quantilize away) |
| good | threshold (cliff) | tilt to *just under* the threshold — the temptation regime |
| good | graded and steep | `q* = 1` restored — don't tilt at all |

Drop the condition and the theorem is false: for a pointwise-good proxy with pressure
headroom, tilting strictly helps. Performativity alone does not make tilting bad; a
threshold creates cliff-edge temptation, while a graded and steep tax can restore
`q = 1` by charging marginal pressure. The zero-dependence policy is strictly dominated
on the pointwise-good/no-tax half — so "never optimize the proxy" is not free wisdom
either; it is the right move exactly where the proxy is misaligned or the oversight is
graded-and-steep.

## 4. A corollary about oversight shape

The table's third and fourth rows are, we think, the most practically suggestive part.
The **cliff-edge temptation** — the reward-optimal agent courting right up to the review
boundary, fragile to any perturbation — is a **threshold-oversight artifact**: it exists
because a threshold tax has a flat, zero-marginal region below the cliff, where tilting
is free. A **graded** tax charges every unit of proxy-pressure at the margin, and when
its marginal rate dominates the pointwise gain it makes *not tilting at all* optimal
even for a pointwise-good proxy. Genuinely interior optima exist only on the knife-edge
between regimes. In short: threshold oversight *creates* the incentive to operate at the
edge of what review permits; graded oversight *removes* it. (The same conclusion was
reached independently, from the empirical side, by a graded-review knob in a related
reinforcement-learning testbed — the theory and the instrument agree.)

## 5. Scope — what this does and does not claim

**It is:** a controlled, in-vitro result about a mechanism. The environments are toys
(an analytic coordination court, binary cues, uniform bases); the Lean theorems cover
unweighted finite bases (uniform weights), with weighted/continuous forms verified
computationally only; the multi-agent case is *worse* for the zero-dependence policy
(under a shared-regime court, one agent's purity is non-pivotal and the safe point is
reachable only by a coalition), so single-agent numbers must not be extrapolated
upward. The unconditional "un-targeting beats quantilizers" claim is **not** made — it
deflates structurally on any base containing restraint, and we report that as a central
finding rather than a caveat.

**It is not:** a foundation-model or deployed-agent claim; a measured claim about any
real base distribution or any real oversight regime; or a replacement for
quantilization where its premises hold. Where the fixed-joint premise holds and the
proxy is pointwise informative, Taylor's bound is the right tool; this note maps where
that stops.

**Falsifiers, standing:** a nonincreasing reward and base geometry where some tilt
strictly beats the base (would refute the theorem's computed extensions); a natural
performative map, monotone in pressure, where tilting helps despite pointwise
misalignment (would break the class boundary); an exogenous projection achieving zero
causal proxy-dependence while retaining above-baseline competence on an entangled proxy
(would make the "target channel" cleanly enforceable and materially soften this note).

## 6. Reproduce

- Lean: `lake build Sundogcert.AxiomAudit` in the `sundogcert` repository (green build
  = all anchors compile with the pinned axiom sets).
- Computed receipts: `scripts/percival-s4-counterproductivity-general.mjs` (falsifier
  hunt), `scripts/percival-b1-0-court-coordination.mjs` (the emergent-threshold court),
  `scripts/percival-b1-separation-bakeoff.mjs` (separation boundary map),
  `scripts/percival-b3-composition-stability.mjs` (iteration/drift),
  `scripts/percival-s6-class-boundary.mjs` (the class-boundary table above), each
  emitting a markdown receipt and JSON artifact.

## References

- J. Taylor. *Quantilizers: A Safer Alternative to Maximizers for Limited Optimization.*
  MIRI technical report / AAAI-16 workshop, 2015.
- J. Perdomo, T. Zrnic, C. Mendler-Dünner, M. Hardt. *Performative Prediction.* ICML 2020.
- J. Elster. *Sour Grapes: Studies in the Subversion of Rationality.* Cambridge, 1983.
- D. Manheim, S. Garrabrant. *Categorizing Variants of Goodhart's Law.* 2018.
- H. Carlsson, E. van Damme. *Global Games and Equilibrium Selection.* Econometrica 1993.
- S. Morris, H. Shin. *Unique Equilibrium in a Model of Self-Fulfilling Currency
  Attacks.* AER 1998.
