# SUNDOG_V_ORDERRELATIVE — Keyed Composition / the Cleanliness-σ Bridge

*Lane header + slate for the order-relative program: the machinery already banked in
`sundogcert` (`OrderRelative*.lean`) and the schema slate
([`SUFFICIENT_STAT_ORDER_SLATE.md`](SUFFICIENT_STAT_ORDER_SLATE.md)), promoted to a named
roadmap and pointed at the two laws the Percival reopen handed it on 2026-07-01.*

Status: **OPENED 2026-07-01. SHARPENED same day by a five-mark critical pass (integrated
at owner direction); first rung OR-5 RUN + CONFIRMED same day
(`OR5_SOVOPT_CONTROL_SIDE_CONFIRMED` — see the entry). Remaining entries: SPEC, not
run.** The pass: the overseer
filtration split into read/write halves and the bridge restated on WRITE (mark 1 — OR-3
recut, a pre-run witness banked); OR-1 recut from bare composability to the **margin
law** (mark 2); OR-2's candidate structures **pre-registered** so the same-lemma fence is
decidable (mark 3); OR-4 given a pre-registered **repair** (Blackwell-order enrichment,
mark 4); OR-5 given a **fidelity fence** (mark 5).
Dependency note: the load-bearing inputs (the S1 cleanliness law and the S2/S3/S4
receipts under `docs/percival/`) have since landed in-repo. This lane consumes them
read-only; no entry may report a RESULT that cites a receipt which has not itself landed.

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

## Spine 2 — the Cleanliness-σ Bridge *(recut per mark 1: stated on the WRITE filtration)*

> **A channel screens cleanly iff its safe-point condition factors through an
> overseer-WRITABLE statistic. Clean/partial is determine/resist — finite/∞ σ —
> computed on the overseer's write filtration.**

The original wording ("overseer-projectable statistic") hid a fork that is resolvable
*before anything runs*, and resolving it is the spine's first deliverable. The overseer
filtration splits into two halves:

- **σ_read** — least order at which the condition factors through statistics the overseer
  can *compute* about the system (observation + interventional probes);
- **σ_write** — least order at which it factors through statistics the overseer can *set*
  by exogenous intervention.

On the **read** half the bridge as originally worded is dead on arrival: the court
already *measures* `c` interventionally (B1's `do(reputation)` probes), so the target
safe-point condition `c = 0` factors through a readable statistic while no write reaches
it — falsifier (b), factorization-without-reach, is exhibited pre-run. On a **write** half
curated by hand ("owned variables") the bridge is circular — clean ⟹ finite σ_write true
by construction. So the bridge is stated on the write filtration **derived from OR-5's
CID typing** (the typing yields both sets; nothing is curated per channel), and the
read-side target channel is banked as the standing witness:

> **MEASURABLE ≠ ENFORCEABLE.** The target channel is σ_read-finite and σ_write-infinite:
> the overseer can *audit* proxy-dependence interventionally and still cannot *enforce*
> its absence.

The σ_read/σ_write split **is** the predicted 9th-filtration sharpening, arriving pre-run
— and it upgrades the bridge from a schema instance to a genuinely new distinction, with
the target channel as the separating example. The reachability-vs-factorization gap the
original wording gestured at (*onto* vs *through*) resolves into exactly this split:
through-readable is not through-writable. What remains genuinely open — OR-3's whole risk
— is the write side: whether a non-circular σ_write, derived from the checked CID typing,
reproduces the clean/partial tiers. The shape is promising but unproven: S2's one-law is
the write-side resist pole (every exogenous write collapses to masking), the S1 tier
table the determine pole (act/aggregate safe points are functions of variables the
overseer owns); OR-3/OR-4 test whether that shape survives being written down.

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
**Fidelity fence (mark 5, binding):** the typed model must match the **real** `Sov_opt` —
the sup over histories of the max unilateral action-swing, arbiter *and* shutdown
included, with the intervention located at the component-output→actuator path — or the
check is decoration on the exact question it exists to answer. A passing verdict on a
simplified `Sov_opt` does not discharge S1's kill.
**Prior:** confirms (S1 argued it from the constrained variable), but it is registered
discipline debt and everything below leans on the tier table — so it runs first.
**Cost:** low (small Lean or typed check). **Shoppable:** no — hygiene. Its value is
double: OR-1..OR-4 inherit a checked table instead of an argued one, and the typing's
**read/write partition of overseer-accessible variables is the filtration OR-3 consumes**
— OR-5 is now formally load-bearing for spine 2, not just its void-check.
**RESULT (2026-07-01): `OR5_SOVOPT_CONTROL_SIDE_CONFIRMED` — S1's registered kill did
not fire; the classification is checked, not argued.** Leg A (Lean):
`sundogcert` `Sundogcert/PercivalCapClass.lean` — `cap_sov_le_kappa`,
`cap_bounds_outgoing_swing` (swing ≤ 2κ under ANY upstream intervention, no internals
assumption), `cap_ignores_incoming_sensitivity` (∀ M ∃ a raw map with input-sensitivity
> M passing the cap everywhere — the two-sided separation) — axiom-clean, gated in
`AxiomAudit.lean`, full build 8569 GREEN. Leg B (computed, on the real machinery):
fidelity fence discharged by **bit-level reproduction of the banked NS-1-c-0 receipt**
(6382 histories, all 12 uncapped/capped stats exact) + hand-anchored formula vectors
(saturating fork: 0.75 uncapped, κ/2 capped — the banked magnitudes) + override intact
under the cap; classification gates all pass — occupant-blind audit (bit-identical
across 4 occupants), incoming-edge severance under output replay, the
dependence/authority **double dissociation** (constant occupant: dependence 0, same
audited authority; feature occupant: dependence 0.507, capped ≤ κ down to κ = 0.01 with
0/6382 violations), write-locus cap bounding even adversarial candidates. **Registered
texture:** `arbiterAuthorityCap` treats κ = 0 as a *disable guard* — the exact
`Sov_opt = 0` point belongs to the fixed-presider control (occupant removed); the cap
family reaches it only in the limit. S1's "zero is dialable" earns a footnote, not a
correction. The read/write partition OR-3 consumes is emitted in the receipt. Spec +
receipt: [`orderrelative/OR5_SOVOPT_CLASSIFICATION_SPEC.md`](orderrelative/OR5_SOVOPT_CLASSIFICATION_SPEC.md),
[`orderrelative/OR5_SOVOPT_CLASSIFICATION_RESULTS.md`](orderrelative/OR5_SOVOPT_CLASSIFICATION_RESULTS.md).

### OR-1 — the keyed-composition MARGIN law *(drill, co-headline; recut per mark 2)*
**Claim:** in a finite multi-agent model, the **margin** of a behavior-keyed guarantee
(distance to violation of a per-agent bound on an owned outgoing variable) aggregates
**additively** under composition — the composed margin is exactly the per-agent marginal,
at every co-agent profile — while the margin of an outcome-keyed threshold guarantee is
**non-additive with a cliff**: flat (no individual purchase) below the coalition fraction
`f*`, total above it. Bare composability is *not* the claim: a 2-agent example already
proves the threshold satisfaction region non-product, which is near-definitional — it
kills the naive recoding attack trivially but shallowly. The law's content is the margin
structure (linear vs cliff degradation).
**Falsifier (recut with the claim):** the **margin-recoding degeneracy** — a derived
per-agent variable in whose terms the threshold guarantee's *margin* becomes additive in
contributions (not merely the satisfaction set product-shaped). Secondary: a
behavior-keyed cap whose composed margin fails additivity in the finite setting (S3.4
makes this unlikely; 0/2583, marginal exactly per-cap).
**Prior:** confirms in the toy — S3.5 is the computed contrast (caps linear; court purity
by threshold at `f* ≈ 0.286`, transition width 0.02); the open question is whether
margin-additivity survives the recoding attack when stated formally.
**Cost:** low-medium (Lean over finite lists/Finset, house idiom; the computed side is
banked). **Shoppable:** yes — "safety margins add across agents for caps and cliff for
outcome-keyed guarantees" is quotable, non-literary, and — unlike the bare-composability
version — *decidable*.

### OR-2 — keying = the composition boundary *(the queued expedition, executed)*
**Claim:** OR-1's law is a **literal instance** of the banked boundary: behavior-keyed
margins compose on a cancellation-free axis where `OrderRelativeComposeLaw`'s
join-homomorphism holds; the court's thresholded aggregate is a non-cancellative
composition where it fails — the S3 contrast is the join-homo boundary wearing an
incentives costume, one lemma, two costumes.
**Pre-registered candidate structures (mark 3 — fixed now so the fence judgment cannot
drift post-hoc):** cap margins compose in the **cancellative** monoid `(ℚ≥0, +)` — the
join-homo regime; threshold satisfaction composes in an **idempotent** algebra
(Boolean `∨` / `max`) — and idempotency is precisely the obstruction the banked converse
names (`OrderRelativeConverse`: join-homo without group-order). **Readout-locus caution,
registered:** the court's *raw* aggregate (the mean `c̄`) is itself cancellative — the
non-homomorphism enters at the threshold **readout** — so what the fence adjudicates is
exactly whether keying to the post-readout algebra is natural or bespoke. That judgment
is now decidable against the two pre-registered structures rather than free-floating.
**Falsifier (metaphor fence, binding):** no natural monoid/order structure on the S3 model
makes the contrast an instance of the *existing* lemma — if the instantiation requires
bespoke structure engineered for the verdict, the entry dies as decoration and says so.
**Prior:** open-leaning-confirms; the gleaner post-mortem ("exotic names overselling
elementary math") is the named warning and the reason the fence is the entry.
**Cost:** low (reading + one Lean instance file in the `OrderRelative*` idiom).
**Shoppable:** only if literal — in which case it upgrades OR-1 from "a law about this
toy" to "an instance of a proved algebraic boundary," which is the lane's whole bet.

### OR-3 — the cleanliness-σ bridge, write side *(co-headline, definitional drill; recut per mark 1; gated on OR-5)*
**Claim:** derive per channel, from OR-5's CID typing, the overseer's **write filtration**
(statistics settable by exogenous intervention) and **read filtration** (statistics
computable by observation + interventional probes); let σ_write / σ_read be the least
orders at which the channel's safe-point condition factors through each. Then (i) the
tier table is reproduced on the **write** side — act/aggregate finite σ_write (the
safe-point condition is a function of an owned variable), target σ_write = ∞ (S2's
collapse: every exogenous write reduces to masking), measure trivial — by a filtration
that is **derived, not curated**; and (ii) the **read** side separates: σ_read(target) is
finite (B1's `do(reputation)` probes measure `c`), which banks **measurable ≠
enforceable** with the target channel as the separating witness.
**Falsifier:** the **non-circularity discharge fails** — every formalization of σ_write
that reproduces the tier table smuggles the tiers in (the write set is not derivable from
the CID typing without per-channel curation); or the derived σ_write misclassifies a
channel (a clean channel with σ_write = ∞, or a partial channel with finite σ_write).
The read-side disagreement — the original falsifier (b) — is no longer a falsifier: it is
exhibited pre-run and banked as the deliverable.
**Prior:** the read-side deliverable is already in hand (mark 1 exhibited it before any
run); the write side is the entry's whole risk — circularity is the named kill, not a
footnote. Landing shape per the H1 pattern: a per-filtration schema instance, now with a
genuinely new distinction attached.
**Cost:** medium (definitional work + the four-channel σ_read/σ_write mapping table with
receipts; no new experiments — synthesis over S1/S2/NS receipts in the σ idiom; gated on
OR-5's typing).
**Shoppable:** high — **"measurable ≠ enforceable"** is crisp, quotable, and survives even
if the write-side bridge dies circular; the write-side bridge, if it lands, is one formal
home for the portfolio's two flagship invariants, and the umbrella's "access is the
sufficient statistic" line finally earns its noun.

### OR-4 — S2's one law as the bridge's machine-checked cell *(drill, rigor)*
**Claim:** the 2×2×2 collapse anchor S2 registered ("any do(U)-invariance projection ⟹
action = f(V) alone ⟹ competence ≤ β") is, stated in the σ idiom, the resist half of
OR-3 for the target channel — Lean-ify it in the `SurfaceBag`/`ParityNoSufficientStat`
`IsSufficient` idiom and the bridge owns its first axiom-clean cell.
**Falsifier:** the Lean-ification needs strictly more than factorization failure — e.g.
the reliability-edge price (`max(β,ρ)−β`) does independent work the sufficiency language
cannot express — in which case bare factorization under-describes the target tier.
**Pre-registered repair (mark 4 — the fix is named with the falsifier, not improvised
after it):** the price term is Blackwell-garbling-shaped — a **deficiency** between the
masked and unmasked experiments. If bare factorization cannot express it, the entry does
not stop at "under-describes": it **enriches the bridge to the Blackwell order** —
factorization gives the clean/partial dichotomy, deficiency prices the partial tier —
and reports the enrichment as the result.
**Prior:** confirms as a one-cell instance; the price-term risk is real and is exactly
what distinguishes "the bridge classifies" from "the bridge explains" — with the repair,
either outcome is a deliverable.
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

- **First cut: OR-5 → OR-1 → OR-3.** OR-5 is now doubly first: it is the cheapest, it can
  void the table everything else reads from, *and* its CID typing supplies the read/write
  filtrations OR-3 is gated on. OR-1 banks the lane's shoppable law (the margin law, now
  decidable) on machinery that already exists; OR-3 is the definitional headline and needs
  no computation, only discipline.
- **Riders:** OR-2 rides OR-1 (same model, one more file; fence pre-registered); OR-4
  rides OR-3 (its first cell; repair pre-registered).
- **Independent fill:** OR-6 touches neither spine's risk and can run any time.
- Suggested landing shape: OR-1 + OR-2 give spine 1 a decidable verdict; OR-3 + OR-4 give
  spine 2 the pre-run **measurable ≠ enforceable** witness, one grounded (possibly
  Blackwell-enriched) cell, and a σ_read/σ_write mapping table with the circularity kill
  live; OR-5 and OR-6 retire two registered debts (S1's kill-check, SurfaceBag's
  conjectured ∞).

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
  garbling — "overseer-writable statistic" is a garbling-flavored notion; OR-3 must say
  precisely where it differs (a write is an *intervention*, not a stochastic map on
  observations), and OR-4's pre-registered repair imports the Blackwell **order** outright
  (deficiency = the reliability-edge price) if bare factorization proves too coarse.
- **Causal incentives (CID)**: Everitt et al. (AAAI 2021) — already the umbrella's genus;
  OR-5 is a CID-typing exercise.
- **Threshold public goods / pivotality**: Palfrey–Rosenthal (1984) — the court's coalition
  threshold is a threshold public good; S3 cites it implicitly, OR-1 should cite it
  explicitly.
- **Data processing**: the S2 one-law is a data-processing-inequality instance; the lane
  adds no new information theory, only the factorization framing.

## Claim boundary

**Will say, if earned:** that safety margins aggregate additively for behavior-keyed
guarantees and by cliff for outcome-keyed ones, with the boundary machine-checked as an
instance of a proved algebraic law against pre-registered structures (OR-1/OR-2); that
**measurable ≠ enforceable** — the target channel is σ_read-finite and σ_write-infinite
(banked pre-run, OR-3); that the clean/partial tiers coincide with determine/resist on a
*derived* write filtration, with one machine-checked (possibly Blackwell-enriched) cell
(OR-3/OR-4); that the stack-top resists every finite surface window (OR-6).

**Will not say:** anything beyond the in-vitro/toy tier; that the bridge is a universal
scalar (the σ slate already proved that door shut); that the write-side bridge holds if
its filtration was curated rather than derived (the circularity kill decides); that a
mapping is an instance when it is an analogy — the same-lemma fence decides against the
pre-registered structures, and a fence-kill is reported as the result; that a simplified
`Sov_opt` verdict discharges S1's kill (the fidelity fence decides).

**Honest prior:** OR-5 and OR-6 confirm; OR-1 confirms in the toy with the
margin-recoding degeneracy as the real risk; OR-2 is open (metaphor risk is the house's
own named failure mode — now decidable rather than free-floating); OR-3's read half is
banked pre-run, its write half lands sharpened or dies circular; OR-4 confirms as one
cell, enriched if the price term demands it. The lane's highest-information outcomes are
OR-2 failing its fence and OR-3's write side failing the non-circularity discharge — both
are banked, not buried, if they happen.

## Cross-links

- Umbrella (spine 2 refines its fence): [`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md)
  — back-link landed in its "what this headers" section 2026-07-01.
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
