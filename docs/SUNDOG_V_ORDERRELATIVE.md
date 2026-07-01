# SUNDOG_V_ORDERRELATIVE — Keyed Composition / the Cleanliness-σ Bridge

*Lane header + slate for the order-relative program: the machinery already banked in
`sundogcert` (`OrderRelative*.lean`) and the schema slate
([`SUFFICIENT_STAT_ORDER_SLATE.md`](SUFFICIENT_STAT_ORDER_SLATE.md)), promoted to a named
roadmap and pointed at the two laws the Percival reopen handed it on 2026-07-01.*

Status: **OPENED 2026-07-01. ROADMAP + SLATE SPEC FOR RED-TEAM, NOTHING RUN.**
Dependency note, stated up front: several load-bearing inputs (the S1 cleanliness law and
the S2/S3/S4 receipts under `docs/percival/`) are working-tree artifacts of the live
reopen slate, owner-held and uncommitted at open time. This lane consumes them read-only;
nothing here promotes or publicizes them, and no entry may report a RESULT that cites a
receipt which has not itself landed.

---

## Why a lane, why now

Two independent seams converged on the same machinery inside one day:

1. **Percival S3's rider** (the multi-agent result's sharpest sentence): *composition of a
   guarantee is decided by what it is keyed on* — behavior-keyed guarantees (caps) compose
   by construction; outcome-keyed guarantees (the court's collection) compose only as far
   as the reward decomposes. Measured contrast: per-agent caps aggregate **linearly**
   (0/2583 violations under adversarial co-agents at every `M`), court purity aggregates by
   **threshold** (unilateral un-targeting non-pivotal for `M ≥ 4`; safe point reachable
   only by a coalition `f* ≈ 0.286`). [`percival/PERCIVAL_S3_MULTIAGENT_RESULTS.md`](percival/PERCIVAL_S3_MULTIAGENT_RESULTS.md).
2. **The S1/S2 cleanliness law's information core**: the target channel is partial because
   "you cannot use information you do not causally depend on" — a data-processing bound —
   while the umbrella has said from birth that channel access is "the determine/resist
   invariant pointed at alignment" ([`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md)).
   That sentence is currently a slogan. This lane's job is to make it a theorem-shaped
   claim with a falsifier, or kill it.

Both seams are order-relative questions: *does a guarantee/condition factor through the
structure you compose or project along?* The lane already owns the relevant banked lemmas;
the composition-law expedition was queued in the Lean lane before Percival made it urgent.

## Banked machinery this lane stands on (nothing new claimed here)

| piece | statement | where |
| --- | --- | --- |
| Order-Relative Resolution Law | `Resolves k t ↔ ord t ≤ k`; determine/resist = finite/∞, per filtration | `Sundogcert/OrderRelative.lean` (gated, axiom-clean) |
| Grading law | `ord(product) = lcm` of orders — divisibility-lattice join; any monoids; n-ary form | `OrderRelativeCompose`, `OrderRelativeGrading`, `OrderRelativeStructure` |
| **Composition boundary** | join-homomorphism **iff cancellation-free** (+ the general coproduct law); converse fails (idempotency obstruction: moment is join-homo, not a group order); search-reach inflates | `OrderRelativeComposeLaw`, `OrderRelativeConverse`, `OrderRelativeSearchNeg` |
| σ schema | σ = least-order sufficient statistic; **schema not scalar** — ≥8 incomparable filtrations; strong single-scalar form falsified and *unrescuable* (Kolmogorov axis breaks the dichotomy) | [`SUFFICIENT_STAT_ORDER_SLATE.md`](SUFFICIENT_STAT_ORDER_SLATE.md) H1 + frontier + 8th-filtration addendum |
| Cleanliness law | clean ⟺ agent cannot reach the bound ∧ bound can reach the safe point; three tiers (no-primitive / clean / partial-retreat-priced) | [`percival/PERCIVAL_S1_CLEANLINESS_LAW.md`](percival/PERCIVAL_S1_CLEANLINESS_LAW.md) |
| Target-channel one-law | collapse, leak-ceiling, cliff-temptation = three faces of one data-processing bound | [`percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md`](percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md) |

---

## Spine 1 — the Keyed-Composition Law

> **A guarantee composes across components exactly as far as the variable it is keyed on
> aggregates homomorphically.** Behavior-keyed guarantees — per-component constraints on
> an owned outgoing variable — compose by construction; outcome-keyed guarantees —
> conditions on a shared aggregate — compose only as far as the aggregate decomposes, and
> fail at threshold aggregation.

Conjectured formal home: the banked composition boundary. Behavior-keyed margins live on a
cancellation-free (join-homomorphic) axis, where the composed order is the lattice join of
the parts; the court's purity is keyed on a thresholded aggregate — a non-cancellative
composition — where join-homomorphism is exactly what fails. **The fence (the gleaner
post-mortem lesson, binding):** the mapping earns the claim only if the *same lemma* does
the work — `OrderRelativeComposeLaw`'s boundary must classify the S3 contrast as a literal
instance, not decorate it. An analogy with new bespoke structure built so the answer comes
out is a kill, not a confirmation.

## Spine 2 — the Cleanliness-σ Bridge

> **A channel screens cleanly iff its safe-point condition factors through an
> overseer-projectable statistic. Clean/partial is determine/resist — finite/∞ σ —
> computed on the overseer's filtration.**

The honest gap, named before anything runs: the cleanliness law is a *reachability*
condition (the primitive projects the system **onto** the safe point — an arrow the
overseer travels down), while σ is a *factorization* condition (the condition factors
**through** a statistic — a different arrow entirely). These may be identical, dual, or
merely correlated on the four channels so far. The house pattern (H1, H2, H3 of the σ
slate; every strong form so far) predicts the landing: **CONFIRMED-but-SHARPENED** — the
bridge holding per-filtration, with "overseer-projectable statistics" entering as a new
(9th) filtration rather than a universal scalar. A clean disagreement between the two
classifications on any channel is the falsifier, and finding one would be the more
valuable outcome: it would pin exactly where reachability and factorization diverge, which
neither the umbrella nor the schema can currently express.

Already in hand for the bridge: the S2 one-law is *shaped* like the resist pole (the
target safe-point condition — zero causal proxy-dependence — is a property of the policy's
function that no exogenous statistic expresses; every projection collapses to masking),
and the S1 tier table is shaped like the determine pole (act/aggregate safe points are
functions of variables the overseer owns). OR-3/OR-4 test whether that shape survives
being written down.

---

## The slate

### OR-5 — the `Sov_opt` classification check *(hygiene; S1's own registered kill; run FIRST)*
**Claim:** in a typed CID-style finite model, `Sov_opt` is formally an **outgoing /
downstream influence cap** — the primitive constrains the realized action-swing variable
through the actuator (control-incentive side), not a response/incoming-dependence property
of the aggregator's function.
**Falsifier:** the formalization classifies it response-incentive-side. This is S1's
registered void condition: the cleanliness law is recut, the clean tier loses aggregate,
and every downstream entry's mapping table changes shape.
**Prior:** confirms (S1 argued it from the constrained variable), but it is registered
discipline debt and everything below leans on the tier table — so it runs first.
**Cost:** low (small Lean or typed check). **Shoppable:** no — hygiene. Its value is that
OR-1..OR-4 inherit a checked table instead of an argued one.

### OR-1 — the keyed-composition law, formal core *(drill, co-headline)*
**Claim:** in a finite multi-agent model, a behavior-keyed guarantee (per-agent bound on an
owned outgoing variable) composes by construction — the composed system satisfies every
per-agent bound with margins aggregating linearly — while an outcome-keyed guarantee
(condition on a shared aggregate) composes **iff** the aggregate is a homomorphic image of
per-agent contributions, with threshold aggregation as the machine-checked counterexample.
**Falsifier (the live one):** the **recoding degeneracy** — every outcome-keyed guarantee
is re-expressible as behavior-keyed on a derived per-agent variable without loss, making
the distinction definitional rather than structural. Secondary: a behavior-keyed cap that
fails to compose in the finite setting (S3.4 makes this unlikely; 0/2583).
**Prior:** confirms in the toy — S3 already exhibits both poles computationally; the open
question is whether the law survives the recoding attack when stated formally.
**Cost:** low-medium (Lean over finite lists/Finset, house idiom; the computed side is
banked). **Shoppable:** yes — "which safety guarantees compose across agents, and why" is
a quotable, non-literary alignment claim in the same register as S4's counterproductivity.

### OR-2 — keying = the composition boundary *(the queued expedition, executed)*
**Claim:** OR-1's law is a **literal instance** of the banked boundary: behavior-keyed
margins compose on a cancellation-free axis where `OrderRelativeComposeLaw`'s
join-homomorphism holds; the court's thresholded aggregate is a non-cancellative
composition where it fails — the S3 contrast is the join-homo boundary wearing an
incentives costume, one lemma, two costumes.
**Falsifier (metaphor fence, binding):** no natural monoid/order structure on the S3 model
makes the contrast an instance of the *existing* lemma — if the instantiation requires
bespoke structure engineered for the verdict, the entry dies as decoration and says so.
**Prior:** open-leaning-confirms; the gleaner post-mortem ("exotic names overselling
elementary math") is the named warning and the reason the fence is the entry.
**Cost:** low (reading + one Lean instance file in the `OrderRelative*` idiom).
**Shoppable:** only if literal — in which case it upgrades OR-1 from "a law about this
toy" to "an instance of a proved algebraic boundary," which is the lane's whole bet.

### OR-3 — the cleanliness-σ bridge *(co-headline, definitional drill)*
**Claim:** define the **overseer filtration** per channel — the statistics of the system
the overseer can exogenously compute and act on — and let σ_ov be the least order at which
the channel's safe-point condition factors through it. Then: clean channels (act,
aggregate) have finite σ_ov (the safe-point condition is a function of an owned variable);
the partial channel (target) has σ_ov = ∞ (zero causal proxy-dependence factors through no
overseer statistic — S2's collapse); measure is trivially determined (nothing to enforce).
Clean/partial **is** determine/resist on this filtration.
**Falsifier:** a channel where the classifications disagree — (a) clean-without-
factorization: a primitive that projects to the safe point while the condition provably
fails to factor through any overseer statistic; or (b) factorization-without-reach: the
condition factors, yet no projection reaches the safe point. Either fires the bridge as
stated; (b) is genuinely live because *onto* and *through* are different arrows.
**Prior:** CONFIRMED-but-SHARPENED expected (per the H1 pattern): the bridge as a
per-filtration schema instance, not a new universal scalar; a clean disagreement would be
the higher-information outcome and gets banked as the deliverable if found.
**Cost:** medium (definitional work + the four-channel mapping table with receipts; no new
experiments — this is synthesis over S1/S2/NS receipts in the σ idiom).
**Shoppable:** high if it lands — one formal home for the portfolio's two flagship
invariants (determine/resist and clean/partial), and the umbrella's "access is the
sufficient statistic" line finally earns its noun.

### OR-4 — S2's one law as the bridge's machine-checked cell *(drill, rigor)*
**Claim:** the 2×2×2 collapse anchor S2 registered ("any do(U)-invariance projection ⟹
action = f(V) alone ⟹ competence ≤ β") is, stated in the σ idiom, the resist half of
OR-3 for the target channel — Lean-ify it in the `SurfaceBag`/`ParityNoSufficientStat`
`IsSufficient` idiom and the bridge owns its first axiom-clean cell.
**Falsifier:** the Lean-ification needs strictly more than factorization failure — e.g.
the reliability-edge price (`max(β,ρ)−β`) does independent work the sufficiency language
cannot express — in which case the bridge under-describes the target tier and OR-3's claim
is at best a coarsening (recorded as such).
**Prior:** confirms as a one-cell instance; the price-term risk is real and is exactly
what distinguishes "the bridge classifies" from "the bridge explains."
**Cost:** low-medium (the anchor was already registered as a Lean candidate in S2).
**Shoppable:** medium — it is the standalone machine-checked statement "in-channel target
enforcement collapses to masking," independent of the Grail frame.

### OR-6 — σ_surface grading, w ≥ 2 *(fill; imported from the R2 arc; the 8th filtration hardened)*
**Claim:** σ_surface(stack-top) = ∞ — for **every** window `w`, there is a witness pair of
valid bracket prefixes with identical `w`-gram count vectors and different stack-tops
(generalizing the `([` / `[(` bag witness by padding the ambiguity beyond any window).
**Falsifier:** some finite `w` determines the stack-top (it factors through `w`-gram
counts).
**Prior:** confirms — the witness construction looks mechanical; the value is closing the
conjectured-∞ hook the `SurfaceBag` module and the schema addendum both flag as open.
**Cost:** low (a `SurfaceBag` extension in the same idiom). **Shoppable:** medium —
hardens the 8th filtration and the R2 state-tracking story ("stack-top is undecodable at
any window, not just for bags").

---

## Vetting / priority

- **First cut: OR-5 → OR-1 → OR-3.** OR-5 is the cheapest and can void the table everything
  else reads from; OR-1 banks the lane's shoppable law on machinery that already exists;
  OR-3 is the definitional headline and needs no computation, only discipline.
- **Riders:** OR-2 rides OR-1 (same model, one more file); OR-4 rides OR-3 (its first cell).
- **Independent fill:** OR-6 touches neither spine's risk and can run any time.
- Suggested landing shape: OR-1 + OR-2 give spine 1 a verdict; OR-3 + OR-4 give spine 2
  one grounded cell plus a mapping table with a live falsifier; OR-5 and OR-6 retire two
  registered debts (S1's kill-check, SurfaceBag's conjectured ∞).

## Standing discipline (binds every entry)

Pre-registered kill criterion per entry — **a clean null/kill is a success and gets banked
as the deliverable**; forward-generate only; deterministic seeded runs where anything is
computed; cheap headless first leg (no GPU, no spend — OR-1..OR-6 are all CPU/Lean);
name the nearest prior and state the delta; the **same-lemma metaphor fence** (OR-2's
falsifier) applies lane-wide; no RESULT block, no claim; nothing promotes past the
in-vitro tier; do not touch owner WIP modules in `sundogcert` (`AnalyticGate`,
`AbstractionCert`).

## The genus (cite, don't reinvent)

- **Composing specifications / assume-guarantee**: Abadi–Lamport ("Composing
  Specifications," TOPLAS 1993), Misra–Chandy. The delta claimed: they compose
  *specifications given interface disciplines*; OR-1 keys composability to **the variable
  the guarantee constrains** and welds the boundary to a proved order/lattice law (OR-2)
  rather than a proof rule.
- **Sufficiency / comparison of experiments**: Fisher–Blackwell sufficiency and Blackwell
  garbling — "overseer-projectable statistic" is a garbling-flavored notion; OR-3 must say
  precisely where it differs (projection is an *intervention*, not a stochastic map on
  observations).
- **Causal incentives (CID)**: Everitt et al. (AAAI 2021) — already the umbrella's genus;
  OR-5 is a CID-typing exercise.
- **Threshold public goods / pivotality**: Palfrey–Rosenthal (1984) — the court's coalition
  threshold is a threshold public good; S3 cites it implicitly, OR-1 should cite it
  explicitly.
- **Data processing**: the S2 one-law is a data-processing-inequality instance; the lane
  adds no new information theory, only the factorization framing.

## Claim boundary

**Will say, if earned:** that behavior-keyed and outcome-keyed guarantees compose
differently, with the boundary machine-checked as an instance of a proved algebraic law
(OR-1/OR-2); that the clean/partial cleanliness tiers coincide with determine/resist on a
named overseer filtration, with one machine-checked cell (OR-3/OR-4); that the stack-top
resists every finite surface window (OR-6).

**Will not say:** anything beyond the in-vitro/toy tier; that the bridge is a universal
scalar (the σ slate already proved that door shut); that Percival results are public or
final while they remain owner-held working-tree artifacts; that a mapping is an instance
when it is an analogy — the same-lemma fence decides, and a fence-kill is reported as the
result.

**Honest prior:** OR-5 and OR-6 confirm; OR-1 confirms in the toy with the recoding
degeneracy as the real risk; OR-2 is open (metaphor risk is the house's own named failure
mode); OR-3 lands sharpened rather than clean; OR-4 confirms as one cell. The lane's
highest-information outcomes are OR-2 failing its fence and OR-3 finding a
reachability/factorization disagreement — both are banked, not buried, if they happen.

## Cross-links

- Umbrella (spine 2 refines its fence): [`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md)
  — a "what this headers" back-link is owed there but that file is mid-edit in the live
  reopen session at open time; **TODO: add the back-link when the working tree settles.**
- Cleanliness law + receipts: [`percival/PERCIVAL_S1_CLEANLINESS_LAW.md`](percival/PERCIVAL_S1_CLEANLINESS_LAW.md),
  [`percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md`](percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md),
  [`percival/PERCIVAL_S3_MULTIAGENT_RESULTS.md`](percival/PERCIVAL_S3_MULTIAGENT_RESULTS.md),
  slate [`percival/PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md`](percival/PERCIVAL_REOPEN_HYPOTHESIS_SLATE.md).
- σ schema (spine 2's other half): [`SUFFICIENT_STAT_ORDER_SLATE.md`](SUFFICIENT_STAT_ORDER_SLATE.md)
  + `internal/slates/SUFF_STAT_ORDER_INTERNAL_2026-06-29.md`.
- Lean machinery: `sundogcert` `Sundogcert/OrderRelative*.lean`, `SurfaceBag.lean`,
  `ParityNoSufficientStat.lean` (the `IsSufficient` idiom), `Percival*.lean`.
- R2 arc (OR-6's home context): [`chatv2/R2_INTERSECTION_HYPOTHESES.md`](chatv2/R2_INTERSECTION_HYPOTHESES.md).
- Memory: [[project_sundog_suffstat_order_slate]], [[project_sundog_causal_access_umbrella]],
  [[project_sundog_percival_lane]], [[project_sundog_lean_formalization]].
