# OR-3 — the Cleanliness-σ Bridge, write side (results / synthesis)

*Definitional drill over banked receipts — no new experiments. Parent:
[`../SUNDOG_V_ORDERRELATIVE.md`](../SUNDOG_V_ORDERRELATIVE.md) entry OR-3 (the recut,
pre-registered claim). Gated on OR-5's typing
([`OR5_SOVOPT_CLASSIFICATION_RESULTS.md`](OR5_SOVOPT_CLASSIFICATION_RESULTS.md), the
read/write partition).*

Status: **RUN 2026-07-01. Verdict: `OR3_BRIDGE_CONFIRMED_SHARPENED`** — three findings:

1. **`WRITE_TIERS_DERIVED`** — the S1 tier table is reproduced by one channel-blind
   derivation rule applied to OR-5's typing; nothing is curated per channel.
2. **`ENFORCE_ORDER_NOT_DETERMINE_ORDER`** (the sharpening, and the drill's catch) — the
   bridge as originally worded would have committed exactly the category error the σ
   slate's H3 was killed for: σ_write is an **enforce**-order (least *arity of a
   projecting write*), not a **determine**-order (least *order of a sufficient
   statistic*). The two verbs stay distinct; the bridge is restated on the correct one.
3. **`MEASURABLE_NE_ENFORCEABLE_BANKED`** — the target channel is the separating
   witness: σ_read finite, σ_write = ∞. Plus one unexpected rung: **the write ladder has
   an intermediate order** — S3's coalition threshold is an arity-⌈f*·M⌉ write.

---

## 1. The derivation rule (stated once, channel-blind)

Inputs: the typed CID graph of the system and the **ownership annotation** — which nodes
the overseer/harness sets, read off the implementation (function signatures, not
judgment). OR-5 checked this annotation where it was checkable: the aggregate cap's
signature consumes actuator-path variables only (T-CAP), and the audit functional takes
no occupant argument (T-OUT-1).

- **W (write set):** variables settable by exogenous intervention implementable as a
  function of overseer-owned variables only — equivalently, variables on arrows whose
  *terminus* the overseer owns (S1's mechanism: clip the end of an arrow you own).
- **R (read set):** statistics computable from observation of realized variables plus
  **counterfactual replay** with substitutions at output slots (the audit pattern —
  interventional reads).

Two order-invariants, one per verb:

> **σ_write(condition)** = the least **arity** (number of independently set owned
> variables — coalition size, in the multi-agent case) of an exogenous write that
> projects the system to the condition's safe point, for all policies/values; ∞ if none
> at any finite arity.
>
> **σ_read(condition)** = the least order of an R-statistic *sufficient to determine*
> whether the condition holds — the σ-schema's standard sufficiency order, on the
> interventional-read filtration.

**The H3 fence, applied up front:** these are different *kinds* — enforce vs determine —
unified in *form* (least order; finite/∞ dichotomy; the Order-Relative Resolution Law's
`Resolves k t ↔ ord t ≤ k` shape, which is a budget/resolution form, not a sufficiency
form). The σ slate proved that conflating verbs (verify/predict, H3) is a kill; the
bridge's original wording ("clean/partial *is* determine/resist on the overseer
filtration") conflated enforce with determine and is retired. The corrected bridge:

> **Clean / partial is the finite/∞ dichotomy of the enforce-order σ_write on the
> derived write filtration. The sufficiency-σ lives on the read side. The two can
> split — and the target channel splits them.**

## 2. The table (all receipts banked; no new runs)

| condition (channel) | σ_write (least projecting arity) | write receipt | σ_read (least sufficiency order, interventional-read filtration) | read receipt |
| --- | --- | --- | --- | --- |
| **measure** — observe-without-selecting | **0** (vacuous — the safe point is the default; nothing to enforce) | A3 (`A3_STATIC_DEFLATION_MEASURE_SAFE`) | finite — but only on the *interventional* read: the order-1 observational statistic (raw MI) **misreads** (1 bit via anti-correlation); the causal/permutation read classifies correctly | A3 diagnostic |
| **act** — full deference | **1** (one clip at the owned committed-action node) | NS-2 (`band_avoidance` 0.87 → ≤0.13 at every κ) | 1 (realized action vs presider) | NS-2 audit |
| **aggregate** — `Sov_opt = 0` | **1** (the ball projection at the action node; exact zero via the fixed-presider control — occupant removed; the κ-dial reaches it only in the limit, OR-5's disable-guard texture) | NS-1-c-0 + OR-5 (bit-level) | ~1 (per-history output-slot replay — the `Sov_opt` audit itself is an interventional read) | OR-5 partition |
| **S3 shared court** — honored point *(outcome-keyed guarantee, not an umbrella channel; included because it populates the ladder)* | **⌈f\*·M⌉** — finite but **Θ(M)**: unilateral write non-pivotal for `M ≥ M*`; a coalition of fraction `f* = 1 − c*/c_γ` projects to honored | S3 (`M* = 4`, `f*` measured 0.29, width 0.02) | 1 (the aggregate `c̄` is the court's own key) | S3 |
| **target** — `c = 0` | **∞** (at any arity within the tested family: every exogenous `do(U)`-invariance projection collapses to masking — the equivalence-theorem form is arity-independent; in-family tilts cannot reach `c = 0` over a biased base) | S2 (`S2_TARGET_CAP_COLLAPSES_TO_MASK`) + B2 (`B2_CLEAN_IS_CORNER`) | **finite** — the court itself keys on `c` (B1.0 G3: `dRegime/dc ≠ 0` at a non-degenerate `c*`), and interventional probes identify named-variable `c` in-model; *content-level* dependence is not expressible by named-variable reads (S2.3's honest boundary) | B1.0 G3 + S2 |

**Tier reproduction check (falsifier (b), row by row):** no clean channel has
σ_write = ∞ (act, aggregate: 1 ✓); no partial channel has finite σ_write (target: ∞ ✓);
measure is trivially reachable (0 ✓). The tiers fall out of the rule — nothing was
placed by hand.

## 3. The banked witness and the ladder

> **MEASURABLE ≠ ENFORCEABLE.** σ_read(target) is finite and σ_write(target) = ∞: the
> overseer can *audit* proxy-dependence interventionally and still cannot *enforce* its
> absence at any write arity (within the tested family).

**The unexpected rung — the enforce-order ladder is graded, not two-point.** Before this
drill the write side looked like a {1, ∞} dichotomy (degenerate, like the moment
semilattice). S3's coalition threshold populates the middle: the shared court's honored
point is enforceable at arity ⌈f\*·M⌉ and at no smaller arity. So:

> **1 (behavior-keyed) < ⌈f\*·M⌉ = Θ(M) (outcome-keyed) < ∞ (dependence-keyed).**
>
> **The keying determines the enforce-order's growth law**: behavior-keyed guarantees
> are O(1)-writable; outcome-keyed ones are writable only at coalition scale, growing
> with the population; dependence-keyed conditions are unwritable at every scale. This
> welds spine 1 (keyed composition) to spine 2 (the bridge): the same keying that decides
> *margin* composition (OR-1/OR-2) decides *enforceability arity*.

## 4. The non-circularity discharge

The rule (§1) names no channel. Its inputs — graph, ownership — are implementation
facts: `arbiterAuthorityCap`'s signature (checked, OR-5 T-CAP), the masking operation's
input-deletion form (S2's projection family), the court conferring reward as a
*world*-owned node (B1.0's model — the overseer does not own patronage). The target
channel's ∞ is not an annotation: it is S2's theorem-shaped collapse (any
`do(U)`-invariance projection ⟹ action = f(V) ⟹ competence ≤ β), which is *why* no
W-statistic reaches `c = 0` — the derivation consumes the receipt, not the tier label.
**Residual pressure point (recorded):** ownership is read off code, and OR-5 checked it
for one channel; the act/measure analogues are signature-facts of the same kind but not
separately re-verified — a red-teamer re-annotating ownership *naturally* and flipping a
row would void the discharge. No such re-annotation is known.

## 5. Scope, honest

- σ_write(target) = ∞ inherits S2's quantification exactly: the tested projection family
  (mask / scramble / permutation-average + the tilt family) under the binary-symmetric CI
  joint; arity-independence comes from the equivalence-theorem *form*, not a sweep over
  coalitions. Beyond that family the ∞ is conjectured, and S2's own richer-joint retest
  caveat carries over.
- The Θ(M) rung is the S3 shared-regime model; the per-knight court is arity-1
  (S3.3 — the fork variable is the reward's aggregation structure).
- σ_read values are in-model (the computed toys expose `c`; real-world read orders are
  not claimed). The named-variable / content-level split (S2.3) bounds what "readable"
  means: incentive-complete, not content-complete.
- Everything is the aggregation-skeleton tier; nothing here is promoted.

## 6. What would void this (live falsifiers)

- A W-statistic outside S2's family enforcing `c = 0` at finite arity with competence
  > β → the target row flips, the bridge's witness dies (this is S2's falsifier,
  inherited — a positive result if found: target becomes fixably-clean).
- A natural ownership re-annotation flipping any row (§4).
- A demonstration that the enforce-order grading is decoration — e.g., the Θ(M) rung
  shown to be an artifact of the shared-regime model with no second instance. (A second
  intermediate-rung instance anywhere in the portfolio would retire this.)

## 7. Feeds

- **OR-4** (rides this): Lean-ify S2's 2×2×2 collapse as the write-side ∞ cell in the
  `IsSufficient` idiom — with the pre-registered Blackwell repair if bare factorization
  can't carry the price term.
- **σ schema** ([`../SUFFICIENT_STAT_ORDER_SLATE.md`](../SUFFICIENT_STAT_ORDER_SLATE.md)):
  σ_read registers as the **9th filtration** (interventional-read sufficiency); σ_write
  registers as the schema's **control-side twin** — same form, different verb, kept
  distinct per the H3 fence (addendum added there).
- **Umbrella** ([`../SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md)): the S5
  refinement's duality gains the verb-split and the growth-law observation — fold
  owner-gated.

## Cross-links

OR-5 partition: [`OR5_SOVOPT_CLASSIFICATION_RESULTS.md`](OR5_SOVOPT_CLASSIFICATION_RESULTS.md) ·
S1 law: [`../percival/PERCIVAL_S1_CLEANLINESS_LAW.md`](../percival/PERCIVAL_S1_CLEANLINESS_LAW.md) ·
S2: [`../percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md`](../percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md) ·
S3: [`../percival/PERCIVAL_S3_MULTIAGENT_RESULTS.md`](../percival/PERCIVAL_S3_MULTIAGENT_RESULTS.md) ·
A3: [`../percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md`](../percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md) ·
NS receipts: [`../NON_SOVEREIGNTY_PAPER_SPINE.md`](../NON_SOVEREIGNTY_PAPER_SPINE.md) ·
OR-1/OR-2 (the keying weld): `sundogcert` `PercivalKeyedMargin.lean`, `OrderRelativeKeyed.lean`.
