# Paper spine — *Non-Sovereignty Is Not Competence*

Working title: **Non-Sovereignty Is Not Competence: Formal and Empirical
Boundaries for Role-Separated Agent Control.**
Punchier alt hook: *Credit the Cap, Not the Council.*

Status: **outline / position draft with first causal-authority receipt.** Strips
the Mithraic framing entirely — that is intellectual bricolage, not a claim, and
has no place in the technical paper.
Venue ladder: **TMLR or a workshop (controlled-study + negative-results framing)
first**, not journal-journal. This document is the position + the formal note +
the experimental-discipline contract; the remaining missing piece before
submission is a stronger empirical artifact (clean task family, multi-seed,
causal-authority/corrigibility hardening).

---

## 1. One-sentence claim

> Role separation should not be credited for an agent-control benefit unless it
> beats a **capped, no-role** control under matched information, budget, training,
> and fault semantics. Otherwise the benefit belongs to the **authority bound**,
> not the plurality.

Everything else is in service of making that sentence precise, falsifiable, and
hard to wave away.

## 2. Positioning (what kind of paper this is)

A **controlled study with a formal note**, not a foundation-model result and not a
"multi-agent systems are better" claim. The contribution is a *discipline* for a
recurring methodological error: crediting an architecture (role separation,
modularity, mixtures, committees) for a benefit that a single bound or
regularizer would have produced alone. We supply the formal reason the naive
comparison is decided in advance, the control that isolates the real cause, and
preliminary evidence that several intuitive "plurality helps" mechanisms do not
survive the control.

## 3. Contributions

1. **Containment lemma (formal, machine-checked).** A constrained/role-separated
   policy class cannot beat an unconstrained superset on *any* shared fixed score.
   Two caveats are load-bearing and are stated, not smudged: it holds for every
   score (so "optimize a different objective" is not an escape), and it requires
   genuine class *containment* (equal budget does not establish it).
2. **Noninterference lemma (formal, machine-checked).** A policy that factors
   through a target-faithful "field" channel is invariant to attacks that preserve
   that channel — i.e. field-grounding **relocates** the attack surface (to sensor,
   dynamics, controller), it does not delete it.
3. **Risk-adjusted break-even (formal, machine-checked).** Under a scalar
   risk-adjusted value `B − L·p`, non-sovereignty wins iff `L·Δp > ΔB` (a capability
   tax is rational once ruin is priced past `ΔB/Δp`) — and the arithmetic exposes
   the un-provable premise: it needs `Δp > 0`, that non-sovereignty actually lowers
   ruin probability. That is the empirical question the paper opens, not closes.
4. **Attribution protocol.** Three matched designs — uncapped monolith `M₀`,
   capped no-role monolith `Mκ`, capped role-separated council `Cκ` — yielding two
   causal contrasts: `Δcap = Q(Mκ) − Q(M₀)` (the **authority-bound effect**) and
   `Δrole = Q(Cκ) − Q(Mκ)` (the **role-separation effect**). A role-separation
   claim additionally requires an **ablation-collapse gate**: the advantage must
   vanish when role separation is removed or distilled.
5. **Negative results as a contribution.** Under this protocol, a sequence of
   intuitive plurality mechanisms (proxy-resistance, richer per-role features,
   structural-attribution, frontier-dominance, safe-exploration, verifier/guard
   factorization, distributed-observer topology) fail to beat `Mκ`; the lone
   bounded positive is feature-specific and does not scale. The recurring cause is
   the containment lemma plus the fact that a learned `Mκ` absorbs whatever
   discipline `Cκ`'s structure imposes.

## 4. Formal note (the appendix, sold honestly)

Three short statements, machine-checked in Lean 4 + Mathlib, axiom-audited
(`#print axioms` pinned; no `sorry`, no `native_decide`). Module:
`Sundogcert/Tauroctony.lean` —
`optimum_mono`, `signature_noninterference`, `ruin_break_even`.
`optimum_mono` and `ruin_break_even` use the standard foundational triple
`[propext, Classical.choice, Quot.sound]`; `signature_noninterference` is
axiom-free.

These are **not deep theorems**, and the paper must say so. Their value is
exactness: they pin the boundary so the empirical claims cannot quietly drift
across it (e.g. claiming a robustness win while the comparator is allowed to
optimize robustness; claiming the surface is "deleted" when it is relocated;
claiming minimax follows from "infinite ruin" rather than from a priced `Δp`).
The formal note is the discipline, not the result.

## 5. Experimental protocol (the contract)

- **Designs.** `M₀`, `Mκ`, `Cκ` as above, plus singleton heads (field-only,
  reward-only) and a privileged oracle for normalization.
- **Matching.** Identical inputs, identical or ≤ trainable budget for the
  constrained designs, identical training/optimizer/seeds, and an explicit fault
  model. Matching is necessary for the comparison; containment is a separate
  expressivity claim that requires either an extensional class definition or a
  constructive simulation. Report both rather than assuming either.
- **Admission before scoring.** Verify the task actually poses the intended
  tension (field necessary-but-insufficient; reward useful-but-dangerous; a
  *learned* `Mκ` leaves headroom) before any architecture is interpreted —
  otherwise the result is a floor/ceiling artifact, not a finding.
- **Two contrasts, reported separately.** `Δcap` and `Δrole`. A role-separation
  claim is *only* `Δrole > 0` surviving the ablation-collapse gate.
- **Sovereignty as causal authority, not mean weight (the new metric).** Replace
  "mean max role-weight ≤ κ" with
  `Sov(D) = sup_h max_i I_i(h)`, where `I_i(h)` is component `i`'s maximum
  *unilateral* swing on the aggregated action (perturb proposal `i`, measure the
  normalized change in the committed action), the **arbiter and shutdown channel
  included as components**. This catches a sovereign arbiter, a one-step override,
  action-scale asymmetry, and collusion — failure modes the mean-weight proxy is
  blind to. Pair with a corrigibility metric
  `Corr_k(D) = inf_h Pr(halt within k | shutdown)`.
  First implementation receipt:
  [`mesa/NON_SOVEREIGNTY_AUTHORITY_AUDIT_RESULTS.md`](mesa/NON_SOVEREIGNTY_AUTHORITY_AUDIT_RESULTS.md).
  The H2 council's reward head obeys its cap, but whole-controller `Sov(D)` still
  reaches `0.865081` and the arbiter exceeds the `0.60` threshold on realized
  histories; no `Corr_k` credit is claimed because no internal shutdown channel
  exists.
- **Seeds and reporting.** ≥3 seeds everywhere, per-seed and pooled, sign-test on
  the per-seed contrast, no single-seed claims.

## 6. Claim boundary (what we will and won't say)

We **will** say: under matched information/budget/training/fault semantics, role
separation earned no credit beyond the cap on our task families; the publishable
positive is the proxy-authority cap as a safe-exploration / authority bound.

We **will not** say: role-separated controllers are better agents; this
generalizes to foundation models; field-grounding deletes the attack surface;
non-sovereignty is established as buying robustness (we only *open* that, gated
on `Δp > 0` measured on a causal-authority/corrigibility axis, not return).

## 7. Related work (positioning, one paragraph each)

- **Modular / hierarchical RL, mixtures-of-experts, ensembles, committees** — we
  supply the missing capped-no-role control that prior architecture-vs-weak-baseline
  comparisons omit.
- **Goodhart / reward hacking / specification gaming** — we formalize the gap as a
  property of *measurers*, and the field as channel noninterference.
- **Corrigibility, instrumental convergence** — the sovereign optimizer is both the
  capability ceiling and the thing most motivated to resist correction; this is the
  tension the bound targets.
- **Moral-uncertainty "parliament", Byzantine fault tolerance, distributed
  knowledge (Hayek)** — the regimes where a non-sovereignty premium is *theoretically*
  available (uncertain objective, defecting component, partial observation); we note
  them as the honest home of any future positive, not as evidence.

## 8. Gaps to close before submission (in priority order)

1. **Causal-authority sovereignty metric** — first `Sov(D)` implementation and H2
   receipt are landed; `Corr_k(D)` is intentionally unscored until a controller
   exposes a real internal shutdown channel. Hardening owed: integrate this audit
   into H1/H2/H3 eval outputs and add a shutdown-capable controller surface.
2. **Multi-seed everything** — finish the multi-seed replication of the lone
   positive; re-report the negatives with ≥3 seeds and sign-tests.
3. **A clean, released task family** — a minimal, reproducible
   field-necessary/reward-dangerous environment with the admission gates baked in,
   open-sourced, so the controlled study is independently runnable. This is the
   long pole.

## 9. Reviewer red-team (anticipated, with the minimal response)

- *"The containment lemma is trivial."* — Agreed; its contribution is
  methodological (it foreclosed the competence comparison and redirected effort)
  and exact (the every-score + containment caveats are the non-obvious part).
- *"Toy tasks; does it transfer?"* — Framed as a controlled study; the claim is a
  *discipline*, falsifiable on any task family, not a transfer result.
- *"You just didn't find the right architecture."* — The protocol is designed to be
  re-run; we publish the control and the gate so a positive, if it exists, is
  cleanly creditable. The negative is bounded to our families, stated as such.
- *"Sovereignty metric is hand-wavy."* — It is a concrete, computable causal
  influence with the arbiter/shutdown included; we report it and its limits.
