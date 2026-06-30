# Sundog Lean Certificate - Fifteen Machine-Checked Cores + One Synthesis Law + One Constructive Universal-Approximation Capstone

> **Deductive complement to the public Sundog lanes.** The public Lean repo now
> carries ten worked examples of the same discipline: machine-check the
> deductive core, then name the imported wall. The first is the
> [P-vs-NP certificate-syndrome lane](SUNDOG_V_P_V_NP.md); the second is a
> real-analysis shadow-decay example; the third is the halo minimum-deviation
> geometry behind the 22-degree halo; the fourth is the Aharonov-Bohm
> gauge-invariance (a gradient's closed-loop circulation is zero); the fifth is
> the general characteristic-function law behind the shadow decay (real analysis
> again); the sixth is a machine-checked Karp reduction
> `3SAT <= 3DM <= X3C <= decoding` that anchors the certificate's
> decoding-hardness import to 3SAT (reduction correctness only); the seventh
> is a finite audit game — a proved-cheap full-access audit paired with
> ∀-verifier blindness of the pooled-mean channel; the eighth is the **tropical
> / piecewise-linear core** (`CircuitNet` + `RegionCount` + `CancellationFree`
> + `DepthSeparation` + `StraightLineCost`) — exact (ε = 0) compilation of
> piecewise-linear circuits to ReLU networks with a **linear gate-count `≤ 4N`
> via a sharing-aware DAG** (`compileToDag`), realization-independent linear-region
> structure, monotone vs. cancellation distinguishing, and the tent's `2^d`
> depth separation — the ε = 0 piecewise-linear core for arXiv:2606.26705 Thm
> 3.2 / Cor 5.1, in the AxiomAudit gate; the ninth is the **N-1 monotone
> depth-as-computation theorem** (`FoldCancellation` + `PieceCover` +
> `RegionPoly`) — *monotone depth is region-polynomial; cancellation depth is
> region-exponential* — a lower bound **in the cancellation-free model**, not a
> lower bound for general neural nets (Telgarsky's depth-vs-width stays imported);
> and the tenth is the **find/check ledger** (`Certifies` plus seven instances:
> syndrome verifier, shortest-path, ReLU gate-count, max-flow/min-cut, König
> matching/cover, 2-SAT decision, Pratt primality) — one `Certifies` interface
> with cheap-CHECK theorems across optimization, decision, and number theory,
> the lane's whole frame: **cheaper to check than to find**, with finding
> imported. Ten examples across nine kinds of math; **not** a learnability claim,
> **not** a full analytic-gate approximation theorem, **not** a hardness or P-vs-NP
> claim, **not** a fast primality test (Pratt is NP-*membership*), and **not** a
> lower bound for general neural nets.

**Public and reproducible:**
[`github.com/humiliati/sundogcert`](https://github.com/humiliati/sundogcert) -
`lake build` re-certifies every theorem; `#print axioms` shows only
`[propext, Classical.choice, Quot.sound]` (no `sorry`, no `native_decide`, no
trusted compiler step). Lean `v4.30.0`, mathlib `v4.30.0`.

Working hook:

> A claim gets smaller and cleaner when its proof goes into Lean and its
> assumptions stay outside the proof, named in plain view.

## Fifteen worked examples

| Lean surface | kind of math | checked deductive core | imported wall |
|---|---|---|---|
| `Certificate` / `Instance` / `Scaling` / `Looseness` / `Degradation` / `CheckCost` | finite-field algebra | syndrome-certificate soundness, exact algebraic lossiness, reject-bound behavior, and check-cost scaling | decoding hardness / SIS one-wayness |
| `ShadowDecay` | real analysis | a lossy averaged shadow washes out a continuous variable while keeping a shared discrete label | that a real system instantiates the averaging model |
| `HaloGeometry` | geometric optics / calculus | `dev_value`, `min_deviation_stationary`, and `min_deviation_isLocalMin` (the symmetric ray is a genuine minimum) | 60-degree ice-prism geometry, measured `n ~= 1.31`, Snell refraction, ray exit, and the observed bright ring at the deviation extremum |
| `FaradayAB` | vector calculus / topology | `gauge_circulation_zero`, `gauge_integrand_eq`, and `gauge_invariant_loop` — a gradient (gauge) field's closed-loop circulation is zero, so the loop observable is gauge-invariant | that the vector potential `A` enters as a loop integral, `grad chi` is the gauge freedom, the Aharonov-Bohm phase / Faraday loop EMF *is* that integral, and the loop encloses the flux (the `H^1` period) |
| `ShadowDecayGeneral` | real analysis (generalizing example 2) | `shadow_decay_charFun` + the determine/resist corollaries — averaging over any probability measure factors through its characteristic function; resist ⟺ `‖charFun‖→0` (Riemann–Lebesgue), determine ⟺ a finite centered mean (two independent conditions); the Gaussian discharges both | that a real system instantiates the characteristic-function averaging; the Cauchy separator is named, not built (mathlib lacks `charFun_cauchy`) |
| `SATReduction*` / `VarWheel` / `ClauseGadget` (on `MatchingNPHard` / `DecodingNPHard`) | combinatorics / computational complexity | `sat_iff_decodes` — a machine-checked Karp reduction `3SAT ≤ 3DM ≤ X3C ≤ bounded-weight GF(2) decoding`, both directions proved; a 3-CNF formula is satisfiable iff its decoding image decodes within the weight bound | the NP class, the poly-time-ness of the reduction maps, and 3SAT's own NP-hardness (Cook–Levin) — hardness imported, any "NP-hard" reading conditional on P ≠ NP, no P-vs-NP claim |
| `AuditCost` | finite decidability / audit game | `pooled_channel_blind` + `no_verifier_checks_perUnit` — a proved-cheap full-access audit (sound + complete against an adversarial reporter at `auditCost ≤ 3n+2`) paired with ∀-verifier blindness: an explicit same-mean fiber pair defeats *every* decidable channel verifier at any prescribed per-unit `δ` | that the pooled-mean channel is the operative observation model (non-vacuity proved: `n = 1` determines; the second moment separates the blind pair) |
| `CircuitNet` + `RegionCount` + `CancellationFree` + `DepthSeparation` + `StraightLineCost` (`AxiomAudit`-gated) | tropical / piecewise-linear algorithmic approximation | `compile_eval` — an exact (ε = 0) compilation of tropical / piecewise-linear circuits to ReLU networks; `compileToDag` — a **sharing-aware DAG compilation with a linear gate count `≤ 4N`** (closes the previous "linear gate-count needs the DAG follow-up" caveat); `compile_depth_le` — a linear depth bound; `bellmanStep_compiles_exactly` — an exact min-plus / Bellman-Ford gate; `RegionCount` — the linear-region structure is realization-independent (an intrinsic invariant); `CancellationFree` — the `IsMono` (cancellation-free) fragment computes exactly the monotone functions (`abs` needs the negative scale: `abs_not_isMono`); `DepthSeparation` — the tent map's `d`-fold realizes `2^d` regions from depth `O(d)`; `StraightLineCost` — a single cost measure that unifies the find/check verifier costs across the lane. The ε = 0 piecewise-linear core for [arXiv:2606.26705](https://arxiv.org/abs/2606.26705) Thm 3.2 / Cor 5.1 | the **analytic** reciprocal / radical gates (and any non-piecewise-linear primitive), which remain approximate / imported; the depth-vs-**width** lower bound (Telgarsky / Eldan–Shamir), which stays imported; learnability and the full Kratsios analytic-gate approximation theorem (neither is asserted by these cores); model realization (whether a real net realizes the compiled circuit) |
| `FoldCancellation` + `PieceCover` + `RegionPoly` (slate-2 N-1, `AxiomAudit`-gated) | monotone-circuit expressivity / depth-vs-regions | `isMono_no_fold` and `isMono_not_iterTent` — a cancellation-free circuit **cannot** realize the `d`-fold tent for `d ≥ 1`; `isMono_realize1_monotone` and `isMono_superlevel_isUpperSet` — the 1-D realization of any `IsMono` circuit is monotone with upper-set super-level sets (no oscillation); `hasPieceCover_iterate` — composing a monotone map with itself `d` times gives `≤ d·k + 1` pieces (**linear in depth**); `hasPieceCover_max_line` — adding one line to a convex function adds at most one piece (the convex-merge linchpin); `isMono_hasPieceCover` — every cancellation-free circuit's 1-D realization has region count **linear in the number of leaves** (polynomial in circuit size). Headline: *monotone depth is region-polynomial; cancellation depth is region-exponential* — closed on the depth-composition axis (`PieceCover`) and the full circuit-structure axis (`RegionPoly`) | the depth-vs-**width** lower bound for general (cancellation-using) neural nets, which stays imported (Telgarsky); the general lower bound for ReLU networks **outside** the cancellation-free fragment, which N-1 deliberately does not bound — this is the natural-proofs boundary the lane respects; the conjecture that "cancellation is the single coordinate" beyond fold / monotone / region (the *organizing* reading of slate-2 N-2, machine-checked half only) |
| `Certifies` + `MaxFlowMinCut` + `MatchingCover` + `TwoSat` + `PrattCert` + `ShortestPathCert` + `MatchingNPHard` (the find/check ledger, slate-2 N-3, `AxiomAudit`-gated) | certifying algorithms — combinatorial optimization, decision, number theory | `weakDuality_tight` — a reusable LP-duality core: weak duality + a tight pair (`p = d`) ⟹ both optima at once (`IsGreatest P p ∧ IsLeast D d`), axiom-free; `maxflow_mincut` — a cut certifies an optimal flow (the canonical LP-dual); `konig` (`matching_le_cover` + tight pair) — a vertex cover certifies a maximum matching; `check_correct` + `cert_sound` on `TwoSat` — a satisfying assignment certifies satisfiability and the `O(|φ|)` evaluator decides the language; `prime_iff_witness` on `PrattCert` — a primitive-root witness certifies primality (NP-membership), wrapping mathlib `lucas_primality`; `cert_isLeast` on `ShortestPathCert`; each instance plugs in to one `Certifies.Ledger` with an `O(·)` cheap-check theorem. **Seven instances** spanning optimization, decision, and number theory; the lane's frame: **cheaper to check than to find** | finding the certificate, which always stays the hard part — every instance imports its find-side: factoring `p − 1` for Pratt witnesses, max-flow algorithms for the actual flow, SCC computation for 2-SAT witness construction; Cook–Levin / NP-hardness for any general "NP-hard" reading (3SAT itself stays imported); the assertion that the find/check modules are **solvers** (they are not — they CHECK a supplied optimum); the assertion that Pratt is a **fast primality test** (it is NP-membership, finding requires factoring `p − 1`) |
| `CplReluNet` (slate-3, `AxiomAudit`-gated) | piecewise-linear ↔ ReLU equivalence | `cpl_iff_reluNet` — **both directions** of the ε = 0 equivalence: a function on `[0,1]` is continuous piecewise-linear iff it is realized by a finite ReLU net. The PL ⟹ ReLU direction extends `compile_eval`'s constructive compilation; the ReLU ⟹ PL direction is the reverse characterization | non-continuous targets, infinite-domain extensions, and the analytic side (which N3-Analytic + the capstone handle separately); not a learnability claim |
| `AnalyticGate` + `SawtoothApprox` (slate-3, `AxiomAudit`-gated) | analytic approximation rates | the lane's first **ε > 0** results: `x²` approximated to arbitrary `ε` by an explicit ReLU net at **both** the elementary-poly rate `O(1/√ε)` (`AnalyticGate.sq_poly_rate`) **and** the polylog depth `O(log 1/ε)` via the Telgarsky sawtooth (`SawtoothApprox.sq_polylog`). Each rate is constructive and the explicit construction is the bridge from the PL fragment to the analytic side | rate-optimality across function classes (this is a *rate*, not the optimal rate); Stone-Weierstrass density (imported from mathlib); the assertion that this is a new approximation theorem (it is not — it is the **constructive direction** of a classical fact in the Yarotsky family) |
| `QueryGap` `check_lt_find` (slate-3, `AxiomAudit`-gated) | the find/check ledger's first **proved** (not imported) gap | `check_lt_find` — in a **restricted query model**, checking a supplied witness takes 1 query while finding the witness takes `≥ n` queries; the gap is proved inside the cost model, not imported. This is the first separator in the ledger that doesn't lean on Cook-Levin or factoring as an external import | the assertion that this is a **P-vs-NP claim** — it is **not**: the gap is in a *restricted query model* against an adversarial oracle, not against polynomial-time computation in general; the assertion that the gap generalizes outside the query model; the assertion that this advances any complexity-theoretic separation in the unrestricted setting |
| `GradedCancellation` (slate-3, `AxiomAudit`-gated) | the monotone-vs-general dichotomy reframed as a graded dial | region count `≤ 4^g · leafCount` where `g` grades the cancellation depth — the binary monotone/general split from N-1 becomes a continuous dial: at `g = 0` the bound recovers N-1's region-polynomial; at `g = log` the bound recovers the `2^d`-style exponential of `DepthSeparation` | the assertion that `4^g` is the **tight** rate; the assertion that the graded dial gives a non-trivial bound *outside* the cancellation-free fragment (general nets stay bounded by their own architecture) |
| `EpsEssentialSample` (slate-3, empirical sample-complexity result) | the geometry-side prediction of generalization data needs | the ε-essential region count predicts the data needed to generalize, with **Spearman 0.68 / 0.78** across measured families (exact `k` is flat). Strengthens N-4's geometry side: ε-essential is not just a width predictor but also a sample-complexity predictor, *modulo* SGD trainability | the assertion that this is a learnability theorem (it is empirical, not deductive); the assertion that the Spearman score is causal — region geometry *predicts* the data need, it does not prove an information-theoretic lower bound; the SGD-trainability residual from N-4 still applies |

The third example matters because it breaks the first two examples' shared
shape. The certificate and shadow-decay examples both involve a lossy shadow
that keeps one invariant while losing another. The halo proof is not that. It
is a pure extremization statement: the deviation

```text
dev n A r = arcsin (n * sin r) + arcsin (n * sin (A - r)) - A
```

is stationary at the symmetric ray `r = A/2`, under the explicit no-total-
internal-reflection differentiability hypothesis. So the demonstrated method is
not "one motif formalized once"; it is a portable way to separate deduction
from import across different mathematical shapes.

The fourth example, the Aharonov-Bohm gauge-invariance, *rejoins* that shared
shape — but topologically: a gradient (gauge) field's circulation around a
closed loop is exactly zero, so the loop observable keeps only the enclosed
flux (an `H^1` period) while the gauge freedom `grad chi` washes out. The fifth,
the general characteristic-function law, *is* the shadow-decay shape stated in
full — it names which spectral condition of the averaging measure governs each
half. So five of the original eight share the deeper shape across a finite-field coset, a
measure-theoretic label and its characteristic-function spectrum, a topological
period, and a pooled-mean audit channel that loses every per-unit claim; the
halo stands apart as the pure-extremization breaker, and the Karp reduction —
`3SAT <= ... <= decoding` — stands apart too, a combinatorial equivalence whose
import is hardness, not a model. The method is tied neither to one motif nor to
one mathematical structure.

The eighth, ninth, and tenth examples sit on a different axis — they trace one
*coordinate* across the algorithmic-approximation lane. The tropical / PL core
(eighth) compiles cancellation-free expressions exactly and bounds gate count
linearly via the sharing-aware DAG; the N-1 result (ninth) shows the same
cancellation-free fragment is **region-polynomial** in depth, while the tent's
`2^d` shows depth's exponential expressivity *requires* cancellation; the
find/check ledger (tenth) machine-checks the **cheap-CHECK** half across seven
problems — optimization, decision, number theory — leaving find-hardness
imported in every instance. Together these three earn a typed-conjecture lens
the lane records under [`ALGO_APPROX_N2_CANCELLATION_SPINE.md`](algo-approx/ALGO_APPROX_N2_CANCELLATION_SPINE.md):
`isMono_tame` proves cancellation-free ⟹ monotone ∧ convex ∧ region-polynomial
(the machine-checked half), while the broader reading that 3SUM (additive),
`n^ω` (subtractive), and analytic gates (division) are all "cancellation-typed"
walls stays a **typed conjecture, not a reduction theorem** — promoting it would
need the imported walls themselves formalized, which they are not.

## The Order-Relative Resolution Law (synthesis core)

Alongside the ten worked examples, the public Lean repo carries one **synthesis
core** — [`Sundogcert/OrderRelative.lean`](https://github.com/humiliati/sundogcert) —
that is framed differently from the table above. It is **not** a
"machine-check the deductive core, name the imported wall" worked example. It is a
single clean *law*, proved once as a schema:

> **A bounded process with budget `k` resolves a target iff the target's order
> `≤ k`. The determine/resist split is finite-order vs infinite-order.**

The schema is a Lean `structure` carrying a `Target` type, an order
`ord : Target → ℕ∞`, a budget-indexed `Resolves : ℕ → Target → Prop`, and the law
`Resolves k t ↔ ord t ≤ k` as a structural field. Budget-monotonicity
(`budget_monotone`), `determine ⟺ finite-order` (`resolvable_iff_finite`), and
`resist ⟺ infinite-order` (`resists_iff_infinite`) are then theorems *of the
schema* — proved once and inherited by every instance.

**Seven grounded axes plus the resist pole** instantiate the schema — each a
genuinely different filtration, none reducible to the others:

- A **parity-determination instance** (`parityProblem n`): the "total parity on
  `n` coordinates" problem, whose `ord = n` is proved equal to the machine-checked
  `ParityNoSufficientStat.suffStatOrder n` (`parityProblem_ord`).
- A **coordinate-locality instance** (`localityProblem n d hd`): a different
  filtration — `ord = d`, the prefix-width, with `prefixSufficient_iff` proving the
  first `k` coordinates suffice for the `d`-prefix parity iff `d ≤ k`. Determine-only
  — `localityProblem_ord_ne_top` shows its order is always finite.
- A **search-reachability instance** (`rationalReachProblem` /
  `irrationalReachProblem`): `ord` = the least rational denominator that reaches the
  target. Rationals have finite order; an irrational like `√2` is reached by no
  finite-denominator rational — an **earned** resist pole `⊤`
  (`search_resist_sqrt_two`), not a fiat one.
- A **radical-reach instance**: `ord` = the least power that carries the target into
  `ℚ`; `√2` has radical order 2 (`radicalReachSqrtTwo_iff`). Paired with the previous
  axis, `sqrt_two_mode_vector` shows `√2` carries **two divergent orders** —
  search-reach `⊤` yet radical order 2 — across two grounded axes on one object.
- A **spectral / moment instance** (`OrderRelativeMoment`): the determine/resist law's
  own measure-theoretic home. `ord = 1` if the target population has a finite mean
  (determine), `⊤` if it does not — the standard **Cauchy** population, an earned `⊤`
  via `cauchy_no_mean`. Here the order is **binary** (1 vs `⊤`): a finite-mean-vs-no-
  mean dichotomy, marking a boundary of how graded the law can be.
- An **algebraic-degree instance** (`OrderRelativeAlgDegree`): `ord` = the degree of
  the minimal rational polynomial with the target as a root. The lane's actual optimum
  `(9+√17)/32` is a root of `16X²−9X+1`, so its algebraic-degree order is **2** — yet
  its search/denominator order is `⊤` (it is irrational). `boxOpt_mode_vector`
  machine-checks this divergence on the real optimum: degree-2-simple, but unreachable
  by naive search.
- A **topological / cohomological instance** (`OrderRelativeCohomology`): `ord` = the
  additive (torsion) order of a (co)homology class — **determine ⟺ torsion, resist ⟺
  free**. A torsion class (`1 : ZMod m`) has order `m`; the **free** class (`1 : ℤ`) has
  order `⊤`, an earned pole — `ℤ = H¹(S¹)` is the Aharonov–Bohm winding / `∮A` flux
  period (companion `FaradayAB.loop_integral_eq_flux`), the loop blind to the global
  flux. On a *mixed* class `(1,1) ∈ ℤ × ZMod m`, `mixed_mode_vector` shows the scalar
  order collapses to `⊤` while the free/torsion projection-vector is `(⊤, m)` — an
  intrinsic mode-vector (canonical, not chosen). Structural rather than deep: it pins
  down that the scalar verdict is *lossy on mixed classes*.
- A **resist pole** (`resistPole`): a target no finite budget resolves — order
  `= ⊤` — paired with `resists_iff_infinite` to instantiate the resist side of
  the dichotomy.

The closing theorem `order_is_schema_not_scalar` is the **honesty guard**: parity
gives finite order `n`, coordinate-locality gives finite order `d`, and the resist
pole gives `⊤` — none of them comparable as numbers across instances. The mode-vectors
make the point sharpest: the *same* object carries divergent orders across axes (`√2`:
search-reach `⊤` vs radical 2; `(9+√17)/32`: algebraic-degree 2 vs denominator `⊤`) — and
in the cohomological case the mode-vector is *intrinsic* to one axis (`(1,1) ∈ ℤ × ZMod m`:
the scalar order collapses to `⊤` while the faithful order is the vector `(⊤, m)`). So the
order is provably **not** a single universal scalar — the scalar is the lossy join of a
latent vector. The finite/∞ split is *per-instance*. The law is a **schema, not a scalar**;
orders compose across instances only through the schema, never through a shared comparable
number.

**Claim boundary.** Not tied to any named hard problem (P-vs-NP, Riemann, etc.);
not tied to alignment or safety; not a universal scalar shared across instances
(the closing theorem proves the opposite); not a worked example in the find-vs-check
sense — finding is not what the law is about, and there is no imported hardness
wall in the same shape as the other ten cores. Axiom-clean
(`[propext, Classical.choice, Quot.sound]`), full `lake build` green in the
`AxiomAudit` gate.

## P-vs-NP certificate core

The first Lean surface is still the deductive core of the P-vs-NP
certificate-syndrome lane. It machine-checks the certificate's **soundness and
lossiness** in Lean 4 - `sorry`-free, axiom-clean, **referee-free**. The kernel
re-checks every theorem in seconds, so the *validity* of this core is
author-independent. The decoding-hardness assumption (information-set decoding
/ SIS) is **imported, not proven** - Lean certifies the deductive core, never
the hardness.

## What is machine-checked in the certificate

- **Lossiness by algebra.** The syndrome `H(sG + e) = He` is independent of the
  secret `s` (`syndrome_independent_of_secret`); every message maps to the same
  syndrome; there are `|F|^k` bodies per syndrome (`secret_bits_lost`). The
  shadow discards `k*log|F|` bits - forced by the algebra, not assumed.
- **Soundness.** `accept -> Safe` - the only route to *accept* is an exhibited
  light witness, which *is* the proof (`accept_sound`); no accepted body is
  unsafe (`no_passing_unsafe`); `reject -> not Safe` under a sound lower bound
  (`reject_sound`).

The trust surface is small: the scheme definitions and the meaning of *Safe*.
Everything above them is kernel-checked. Peer review shrinks from "trust the
proof" to "audit the statement."

## The wall, named

Lean certifies **soundness + lossiness only** for the certificate lane. The
certificate's security rests on a decoding-hardness assumption that is
**imported, not proven** - hardness is not a mathlib theorem. Every
"Lean-verified" claim here means the deductive core, never the hardness.

## The reject bound, fully characterized

The load-bearing reject bound `colWeightLb` is pinned down from every direction,
each fact kernel-verified:

| regime | behavior | theorem |
|---|---|---|
| any basis | **sound** - never exceeds the true witness weight | `colWeightLb_sound` |
| uniform `H` | **tight** - equals the true distance; reject threshold scales linearly, `tau = n/2 - 1` | `scaling_law` |
| denser `H`, same code | **loose** - collapses to `0`, purely from the basis | `looseness` |
| general | capped by `||syndrome|| / density` | `colWeightLb_le_card_div` |

Items (loose) and (general) are **completeness** phenomena, not soundness
breaks: a collapsed bound still never over-claims - it quarantines where it
cannot reject. Soundness never depends on the basis; only the bound's *strength*
does.

## The frontier

A *cheap, basis-robust, tight* reject bound - one that does not degrade when the
parity-check is chosen adversarially - would return the true minimum coset
weight on every basis. That **would be a fast decoder**: it would solve the very
problem (information-set decoding) whose hardness the certificate imports. So
the basis-dependence of `colWeightLb` is not a defect to be patched away - it is
the visible edge of the hardness assumption. The honest open question is
quantitative: how large is the gap between a cheap bound and the true coset
weight, as a function of the decoding margin.

## The hardness wall, pushed inward (the reduction chain)

The decoding-hardness assumption is no longer opaque. A machine-checked Karp
reduction now connects it to the canonical NP-complete problem:

> **3SAT <= 3DM <= X3C <= bounded-weight GF(2) decoding**

is formalized end to end, its top-level correctness an `iff`
(`SATReductionMain.sat_iff_decodes`): a 3-CNF formula `phi` is satisfiable **if
and only if** the decoding instance it maps to decodes within the weight bound.
Both directions are proved - the forward builds the perfect matching from a
satisfying assignment (Garey-Johnson variable-wheel, clause, and garbage
gadgets, the leftover tips absorbed by a counted bijection), the reverse reads
an assignment back out of any perfect matching. Axiom-clean, like the rest.

What is machine-checked is the reduction's **correctness** - the many-one / Karp
equivalence between the SAT instance and its decoding image. The complexity
wrapping stays imported, because mathlib has no complexity-theory framework: the
**NP class** itself, the **poly-time-ness** of the reduction maps (each built and
proved correct, but never timed), and 3SAT's **own NP-hardness** (Cook-Levin, in
no proof assistant to date). So the certificate's "decoding is hard" import is
now *anchored* - at least as hard as 3SAT, modulo the named wrapping - while any
"NP-hard" reading stays **conditional on P != NP**. This is **not** a claim about
P versus NP, and not a proof that decoding is hard.

## Relation to the P-vs-NP lane

The certificate-syndrome receipts (v1-v6) measure the **empirical** side -
cheaper to check than to find (op-count cost certificate `0.949 <= 1.0`), safety
green. This ledger is the **deductive** complement: the soundness and lossiness
those receipts rely on are machine-checked, axiom-clean. The two are orthogonal,
and neither proves the decoding hardness - which both import.

The `ShadowDecay`, `ShadowDecayGeneral`, `HaloGeometry`, `FaradayAB`, and
`AuditCost` modules are method demonstrations, not extra P-vs-NP evidence. They
show that the same public Lean discipline spans finite-field algebra, real
analysis, geometric optics, vector calculus / topology, and a finite audit game
without turning any one imported wall into a theorem. The reduction-chain modules (`SATReduction*`, on `MatchingNPHard` /
`DecodingNPHard`) sit closer to this lane: they *anchor* the certificate's
decoding-hardness import to 3SAT by a checked reduction. But they too leave the
hardness imported (the NP class, poly-time-ness, and Cook-Levin), so they sharpen
the import rather than discharge it - still not P-vs-NP evidence.

The newest cluster — the tropical / PL core (`CircuitNet` + `RegionCount` +
`CancellationFree` + `DepthSeparation` + `StraightLineCost`), the N-1 monotone
depth-as-computation theorem (`FoldCancellation` + `PieceCover` + `RegionPoly`),
and the find/check ledger (`Certifies` + the seven instances: syndrome verifier,
shortest-path, ReLU gate-count, max-flow/min-cut, König, 2-SAT, Pratt) — extends
the discipline into algorithmic approximation, monotone-circuit expressivity,
and certifying algorithms (combinatorial optimization, decision, number theory).
The lane's whole frame is right there: **cheaper to check than to find** —
machine-checked across seven problems. *Finding* is still the imported wall in
every instance (factoring `p − 1` for Pratt, the actual flow algorithm for
max-flow, SCC construction for 2-SAT, and Cook–Levin / NP-hardness for any
general "NP-hard" reading). N-1 is a lower bound **in the cancellation-free
model**; the natural-proofs boundary is respected — the depth-vs-width lower
bound for general nets (Telgarsky) stays imported. This is the same discipline
operating on a richer slate, not a P-vs-NP advance, a learnability claim, a
fast primality test, or a general neural-net lower bound.

The **Order-Relative Resolution Law** (`OrderRelative`) sits alongside as a
synthesis core, not a worked example. It is framed separately — see the
"Order-Relative Resolution Law" section above — and explicitly disclaims any
universal-scalar reading via `order_is_schema_not_scalar`.

## Status

**PUBLIC, REPRODUCIBLE, TEN-EXAMPLE LEAN METHOD CORE + ONE SYNTHESIS LAW.** The P-vs-NP
certificate soundness + lossiness are machine-checked; the shadow-decay (the
concrete Gaussian decay and its general characteristic-function law), halo
minimum-deviation, and Aharonov-Bohm gauge-invariance examples extend the method
to three more mathematical shapes; a machine-checked Karp reduction
`3SAT <= 3DM <= X3C <= decoding` anchors the certificate's decoding-hardness
import to 3SAT; a finite audit game proves a cheap full-access audit blind to
every decidable channel verifier; the tropical / piecewise-linear core machine-
checks an exact (ε = 0) circuit-to-ReLU compilation with a **linear gate count
`≤ 4N` via the sharing-aware DAG** (`compileToDag`), realization-independent
linear-region structure, the cancellation-free fragment characterization, and
the tent's `2^d` depth separation — the ε = 0 piecewise-linear core for
arXiv:2606.26705 Thm 3.2 / Cor 5.1, in the AxiomAudit gate; the N-1 monotone
depth-as-computation theorem proves a cancellation-free circuit has region
count linear in its leaves and cannot realize the `d`-fold tent for `d ≥ 1`;
and the find/check ledger machine-checks seven cheap-CHECK instances under one
`Certifies` interface — ten worked examples across nine kinds of math. **Alongside,
one synthesis core** — the Order-Relative Resolution Law (`OrderRelative`) —
proves a single schema once: a budget-bounded process resolves a target iff the
target's order is within budget, with `determine ⟺ finite order`,
`resist ⟺ infinite order`, grounded on the parity instance and explicitly
guarded by `order_is_schema_not_scalar` against any universal-scalar misread.
Hardness, model realization, physical optics, the physical gauge field, the
NP-class / poly-time / Cook-Levin complexity wrapping, the **analytic**
reciprocal / radical gates (which remain approximate / imported), the
depth-vs-**width** lower bound for general nets (Telgarsky / Eldan–Shamir), and
the find-side of every certifying instance (factoring for Pratt witnesses,
max-flow algorithms, SCC construction, etc.) all remain named walls. Not a
cryptographic one-wayness claim; not a claim about P versus NP; not a claim
that Lean proves the sky realizes the halo or that nature realizes the
Aharonov-Bohm effect; not a learnability claim; not a full analytic-gate
approximation theorem; not a fast primality test (Pratt is NP-*membership*);
not a lower bound for general neural nets; not a universal cross-instance scalar
(the `order_is_schema_not_scalar` theorem proves the opposite).
