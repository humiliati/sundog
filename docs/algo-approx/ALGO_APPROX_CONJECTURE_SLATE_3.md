# Algorithmic Approximation — Fresh Conjecture Slate 3 (opened 2026-06-29)

> **What this is.** Slates 1 and 2 are closed:
> [`ALGO_APPROX_CONJECTURE_SLATE.md`](ALGO_APPROX_CONJECTURE_SLATE.md) (the exact-ε=0 PL core,
> regions-intrinsic, the monotone wall, depth=computation, the find/check seed) and
> [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](ALGO_APPROX_CONJECTURE_SLATE_2.md) (the cancellation
> spine: N-1 region-polynomiality both axes, N-3 the 7-instance `Certifies` ledger, N-2 the
> spine retrospective, N-4 the ε-essential empirical redemption). This slate mines the seams
> those closures left **open** — the places the lane kept saying "imported," "ε-free only," or
> "binary." Five hooks plus one daydream.
>
> **Discipline (inherited, unchanged).** Nothing is promoted without a receipt. Each entry is
> stated so it can come back **NULL**: a claim, the traction the existing machine-checked cores
> give, the classical question it touches, a **named falsifier**, a **tier**, and a **first
> move**. Status is earned only by a Lean core or an experiment. The standing boundary holds:
> **not a P-vs-NP / Millennium claim; find-hardness stays imported except where a hook explicitly
> proves it; "Lean-verified" means the deductive core.**
>
> **Provenance.** Authored inline from the slate-1/-2 closures, not a cold fan-out.

## Tier legend

- **[FORMALIZABLE]** — a Lean target on the existing `CircuitNet` / `CancellationFree` /
  `RegionPoly` / `Certifies` surface; could become an axiom-clean, gated core.
- **[FORMALIZABLE-HARD]** — formalizable, but needs a genuinely new construction or analysis
  (not just assembling existing lemmas).
- **[EMPIRICAL]** — a runnable CPU experiment with a falsifiable prediction.
- **[DAYDREAM]** — mapped to mark the frontier, not yet actionable.

## The seams slate 2 left open

1. The PL core is **exact (ε = 0) only** — the paper's actual headline is *polylog-1/ε rates*
   for analytic gates, which the lane has not touched in Lean (S3-4).
2. Region-polynomiality is **binary** (`IsMono` vs not) — cancellation is graded, but the bound
   is not (S3-2).
3. The find/check ledger's gap is **imported in every one of its 7 instances** — not one is a
   machine-checked `check ≪ find` (S3-3).
4. "Exactly ReLU-representable" is **used but never characterized** — the paper's title is a
   *characterization of universal approximation* (S3-1).
5. N-4's ε-essential invariant is shown for one axis (generalization-width threshold) on one
   harness — its reach (sample complexity, transfer) is untested (S3-5).

---

## S3-1 — Exact representability: continuous-PL ⟺ finite-width ReLU. [FORMALIZABLE]

> **Status (2026-06-29): CLOSED — the full `⟺` is machine-checked.** `Sundogcert/ExactRepr.lean`:
> - `cpl_iff_reluNet (f)`: `(∃ k, HasPieceCover f k) ↔ (∃ g : Net 1, realize1N g = f)` — **the
>   characterization at ε = 0, both directions.** Since `HasPieceCover` *is* the exact continuous-PL
>   predicate, this reads: a 1-D function is exactly a finite ReLU net **iff** it is continuous
>   piecewise-linear with finitely many pieces.
> - Converse (CPL ⟹ ReLU): `cpl_realizable` — peeling induction on the cut set
>   (`Finset.strongInduction`): at the largest breakpoint `b`, subtract one ReLU correction
>   `Δ·relu(· − b)` (`Δ` = slope jump, from `f`'s values, no `Classical.choose`); the remainder is
>   affine away from `S.erase b`, recurse. `convexCPL_realizable` is the convex special case via the
>   upper-envelope construction (`convex_eq_sup_lines` → `linesTrop` → `compile`).
> - Forward (ReLU ⟹ CPL): `net_hasPieceCover` — induction on the net; the only real content is
>   `hasPieceCover_relu`, the general non-convex **doubling** bound `HasPieceCover f k →
>   HasPieceCover (max f 0) (2k)` (the cut set is `f`'s breakpoints plus one zero-crossing per piece
>   — the crossing-enumeration scaffold deferred in N-1, now discharged).
>
> All four theorems axiom-clean, in the `AxiomAudit` gate, full build green (8531 jobs);
> `CPL_NOT_REPRESENTABLE` did not fire in either direction. Only the *minimal-width* sharpening
> (width = breakpoint count) is left, and it is a quantitative refinement, not a gap in the `⟺`.
> *(Local only; not committed — owner-gated.)*

*Claim.* A 1-D function is **exactly** representable by a finite ReLU network **iff** it is
continuous piecewise-linear with finitely many pieces — and the minimal width is the breakpoint
count. The ⟸ direction (ReLU ⟹ CPL) is essentially `decompile`/`compile_eval`; the ⟹ direction
(CPL with `p` breakpoints ⟹ a width-`p` net realizing it exactly) is the new content, and the
`RegionPoly` piece-line machinery (`convex_eq_sup_lines`, `HasPieceCover`) is most of it.

*Traction.* This is the **exact (ε = 0) case of the paper's title** — "a characterization of
universal approximation" — and the lane already has both halves' machinery. It also *upgrades*
N-1: `isMono_hasPieceCover` bounds region count; this would pin it to an exact representability
equivalence (the bound is tight and achieved).

*Classical hook.* Arora et al. (ReLU nets compute exactly the CPL functions); the 1-D minimal-width
question.

*Falsifier* (`CPL_NOT_REPRESENTABLE`): a finite-piece CPL function provably **not** realizable at
finite width, or a width strictly below the breakpoint count realizing a `p`-piece target exactly.

*First move.* Define `IsCPL (f) (p)` (a finite breakpoint set off which `f` is affine, the
`HasPieceCover` shape without the ε), then prove `IsCPL f p ↔ ∃ net, width ≤ p ∧ realize1N net = f`
— ⟸ via `decompile`, ⟹ by a per-breakpoint ReLU construction (signed, for non-convex; the convex
case is `convex_eq_sup_lines` composed with `compile`).

---

## S3-2 — Cancellation is a *graded* resource; region growth is graded with it. [FORMALIZABLE]

> **Status (2026-06-29): LANDED.** `Sundogcert/GradedCancellation.lean` —
> `hasPieceCover_graded (e : Trop 1) : HasPieceCover (realize1 e) (4 ^ cancelMax e * leafCount e)`.
> The dichotomy becomes a **dial**. Plus `cancelMax_eq_zero_of_isMono` (budget `0` ⇒ recovers N-1's
> linear `leafCount` bound). Both axiom-clean, in the `AxiomAudit` gate, full build green (8533 jobs).
> `GRADE_DOESNT_BOUND` did not fire.
>
> **Refinement of the budget (a finding):** the operative budget is **`cancelMax e`** = the number
> of `max` gates whose subtree contains a negative scale — *not* the raw negative-scale count.
> Cancellation only blows up regions *through a fold*: a `max` of convex pieces stays convex and
> does not double (`hasPieceCover_max`); only a `max` of a non-monotone (hence non-convex) argument
> can. A circuit with negative scales but no folding `max` is still affine-piece-cheap, and
> `cancelMax` correctly reads `0` there. So the honest cancellation budget localizes to the
> cancellation-*exposed folds*. Base `4` (not `2`) because the non-convex `max` bound routes through
> the ReLU doubling `max(f,g) = g + relu(f−g)` (`hasPieceCover_relu`), a constant-factor loose; a
> direct symmetric `2·(n+m)` `max` bound would give base `2`. The grading is unchanged.
> *(Local only; not committed — owner-gated.)*

*Claim.* Define a **cancellation budget** `g(e)` = the number of negative-scale gates in a `Trop`
circuit. Then the region count is bounded by an **interpolation** `≤ 2^{g} · poly(leafCount)`:
at `g = 0` it is `isMono_hasPieceCover` (polynomial), and each unit of cancellation at most
doubles the piece budget. Cancellation is not a switch but a dial, and the dial reading bounds
the geometry.

*Traction.* Sharpens N-1 (binary `IsMono`/not) and N-2 (the spine reading) into one graded
theorem: it would make "cancellation is the coordinate" a *quantitative* statement, not a
dichotomy. The `RegionPoly` gates already give the `g = 0` floor and the per-gate piece bounds;
the new work is a `negCount` recursion and a "one negation at most doubles" lemma.

*Classical hook.* Monotone-vs-general circuit complexity as a *graded* (not binary) separation;
the "amount of cancellation" as a complexity resource.

*Falsifier* (`GRADE_DOESNT_BOUND`): a circuit with small `g` (say `g = 1`) whose region count is
super-polynomial in size — cancellation budget fails to bound the geometry.

*First move.* `def negCount : Trop n → ℕ`; prove `hasPieceCover (realize1 e) (2^(negCount e) *
leafCount e)` by induction, reusing the `RegionPoly` cut/scale/max gates and handling a negative
`scale` as the one step that can double (it flips a convex piece to concave, doubling the merge).

---

## S3-3 — A non-imported find/check gap (the ledger's first proven separation). [FORMALIZABLE]

> **Status (2026-06-29): LANDED.** `Sundogcert/QueryGap.lean` — a self-contained decision-tree
> (query) model `DTree` (`eval`/`depth`/`queriedOn`) over `x : Fin n → Bool`, unstructured search:
> - **CHECK** `checkTree i` reads `x i` and reports it: `checkTree_eval` (correct), `checkTree_depth
>   = 1` (one query). Both `[propext]`-only — axiom-lighter than the standard triple.
> - **FIND** `search_needs_n_queries`: any tree with `(∀ x, eval t x = true ↔ ∃ i, x i = true)`
>   has `depth ≥ n`. Proof = the **adversary** lemma `queried_all_of_decides` (on the all-false
>   input the tree must query *every* position — else flip an unqueried one without changing the
>   path: `eval_eq_of_agree`) + `card_queriedOn_le_depth`.
> - **The gap** `check_lt_find`: for `n ≥ 2`, `(checkTree i).depth < t.depth` for any correct
>   finder `t`. **Both sides machine-checked; nothing imported** — the ledger's first proved
>   `check ≪ find`. Axiom-clean, in the `AxiomAudit` gate, full build green (8532 jobs).
>   `GAP_COLLAPSES_IN_MODEL` did not fire. Honest scope: an *unconditional lower bound in a
>   restricted (query) model*, **not** a P-vs-NP claim — which is exactly what lets it be proved
>   rather than imported. *(Local only; not committed — owner-gated.)*

*Claim.* Every one of the 7 `Certifies` instances imports its find-hardness. Add **one instance
where the gap is proved in Lean**: in a query / decision-tree model, *checking* a supplied witness
costs `O(1)` while *finding* it provably costs `Ω(n)` queries (an adversary argument). This is the
ledger's first **machine-checked `check ≪ find`** — both sides deductive, nothing imported.

*Traction.* It converts the ledger's standing honesty caveat ("find imported") into a *partial
theorem*: in a restricted but real model, the asymmetry the whole lane is about is itself proven.
The cheap-check half is exactly the `Certifies` shape; the new content is a small query-complexity
formalization (a decision-tree / adversary lower bound for unstructured search — `OR`/needle).

*Classical hook.* Query (decision-tree) complexity; the deterministic adversary lower bound for
search; the NP "verify ≤ find" gap made concrete in a model where the lower bound is unconditional.

*Falsifier* (`GAP_COLLAPSES_IN_MODEL`): the chosen model admits a checker-cheap **and** finder-cheap
algorithm — no provable separation survives — so the gap stays genuinely imported everywhere.

*First move.* A Lean `DecisionTree`/query model over `Fin n → Bool`; prove "any tree deciding
`∃ i, x i` correctly has depth ≥ n on the all-false-vs-one-true adversary family" (find ≥ n) and
"a supplied index `i` with `x i = true` is checked in 1 query" (check = 1); register as a
`Certifies.Ledger` instance carrying *both* bounds.

---

## S3-4 — An analytic-gate ε-rate (the paper's actual headline, in Lean). [FORMALIZABLE-HARD]

> **Status (2026-06-29): FIRST STRIKE LANDED (ε > 0 milestone at the polynomial rate).**
> `Sundogcert/AnalyticGate.lean` — the canonical analytic gate `x²` (the Yarotsky/Telgarsky
> building block, *not* piecewise-linear) is approximated on `[0,1]` by an explicit finite ReLU net
> with a **machine-checked L∞ bound**:
> - `sqNet_approx (n) (hn : 0 < n) : ∀ x ∈ [0,1], |x² − realize1N (sqNet n) x| ≤ 1/(4n²)`.
> - `sq_eps_approx (ε) (hε : 0 < ε) : ∃ g : Net 1, ∀ x ∈ [0,1], |x² − realize1N g x| ≤ ε` — the
>   lane's **first ε > 0 result**: an analytic gate IS a finite ReLU net to any ε, proved.
>
> Architecture reuses S1: the chordal interpolant is the upper envelope of its secants
> (`linesTrop` → `compile`), so "PL ⟹ ReLU" is free; the new content is the error bound, which is
> *elementary* for `x²` because the secant error has a **closed form** `secant(x) − x² = (x−a)(b−x)`
> — one `sq_nonneg` for the upper bound, `mul_nonneg` for the lower. Axiom-clean, in the
> `AxiomAudit` gate, full build green (8534 jobs).
>
> **Status (2026-06-29): HARD STEP ALSO LANDED — the polylog rate is machine-checked.**
> `Sundogcert/SawtoothApprox.lean` — `x²` on `[0,1]` via the Yarotsky/Telgarsky **sawtooth**, at
> **logarithmic depth**. Reuses the lane's depth-separation tent `T(x) = 1 − |2x − 1|`. Define the
> self-similar approximant `R 0 x = x`, `R (m+1) x = x − T x/2 + R m (T x)/4`. The entire error
> analysis collapses to one **self-similar recursion** `e_{m+1}(x) = e_m(T x)/4` (from the tent
> identity `(T x − 1)² = (2x − 1)²`, here trivial since `T x − 1 = −|2x − 1|`), giving by induction
> - `sq_sub_R_le : |x² − R m x| ≤ 1/(4·4^m)` on `[0,1]` — **geometric** error decay in `m`;
> - `Rcirc_depth : (Rcirc m).depth ≤ m·(2·tent.depth + 4)` — **linear** depth (built by `m`-fold
>   `subst0` composition with the tent);
> - `sqNet_approx` / `sqNet_depth` — same after `compile` (depth `O(m)` via `compile_depth_le`);
> - `sq_polylog_approx (ε) : ∃ m g, (∀x∈[0,1], |x²−realize1N g x| ≤ ε) ∧ depth g ≤ 4·m·(…) ∧
>   1/(4·4^m) ≤ ε` — **the polylog rate**: error `ε` at depth `O(m) = O(log 1/ε)`, exponentially
>   fewer gates than the first strike's `O(1/√ε)` pieces.
>
> **`RATE_NOT_POLYLOG` is now CLEARED** — the paper's headline rate (Cor 5.1 spirit) is
> machine-checked for `x²`. The sawtooth = the depth-separation tent (`DepthSeparation`,
> exponential pieces from linear depth) put to *constructive* use. Axiom-clean, in the `AxiomAudit`
> gate, full build green (8537 jobs). *(Local only; not committed — owner-gated.)*

> **First strike (still standing, the elementary poly rate).**
> `Sundogcert/AnalyticGate.lean` — `x²` on `[0,1]` by an explicit `n`-piece interpolant with a
> machine-checked L∞ bound: `sqNet_approx : |x² − net| ≤ 1/(4n²)`; `sq_eps_approx : ∃ g, error ≤ ε`.
> The lane's first ε > 0 result, at the `O(1/√ε)`-piece (polynomial) rate; the secant error has the
> closed form `secant(x) − x² = (x−a)(b−x)` (one `sq_nonneg`). Subsumed in *rate* by the sawtooth
> above, but kept as the clean elementary construction.

*Claim.* The lane's Lean core is **exact (ε = 0)** and stops at the PL fragment; the paper's
result is *polylog-1/ε* size for **analytic** gates. Machine-check **one** such rate: `√x` (or
`1/x`) on `[a,b]` is approximable to `L∞` error `ε` by a ReLU net of size `O(polylog 1/ε)`, with a
**proven** error bound — the first ε > 0 result in the lane.

*Traction.* This is the lane's most direct engagement with arXiv:2606.26705's Cor 5.1 spirit (the
APSP/shortest-path rate the lane has only matched at ε = 0). The construction is classical
(geometric-breakpoint PL interpolation of a convex analytic function — Telgarsky's sawtooth /
the standard `√x` net); the work is the **L∞ error bound** and compiling it through `CircuitNet`.

*Classical hook.* Yarotsky / Telgarsky ReLU approximation rates; the convex-function PL-interpolation
error bound; arXiv:2606.26705 Cor 5.1.

*Falsifier* (`RATE_NOT_POLYLOG`): the explicit construction's proven error needs `poly(1/ε)` (not
`polylog`) width — the analytic-gate rate does not transfer to the Lean construction at the claimed
size.

*First move.* Pick `√x` on `[1,2]`; define the geometric-breakpoint interpolant `pl_sqrt ε`; prove
`‖√· − pl_sqrt ε‖∞ ≤ ε` with `pieces = O(log 1/ε)` (convexity bounds the per-interval error by the
chord gap, which the geometric spacing equalizes); compile via `CircuitNet`. Hard part: the clean
`L∞` bound, not the compilation.

---

## S3-5 — The ε-essential count predicts *sample complexity*, not just width. [EMPIRICAL]

> **Status (2026-06-29): CONFIRMS.**
> [`ALGO_APPROX_S35_SAMPLE_COMPLEXITY_RESULT.md`](ALGO_APPROX_S35_SAMPLE_COMPLEXITY_RESULT.md) ·
> `scripts/algo_approx_s35_sample_complexity.py`. At **fixed generous width** (capacity removed),
> sweeping `n_train` and reading `sample_threshold(bar)`: the ε-essential region count tracks the
> sample threshold (Spearman **0.675** noiseless / **0.784** under label noise; monotone 9/9),
> while exact `k` does **not** (−0.149 / 0.028 — flat-to-uninformative). `EPS_NOT_SAMPLE_PREDICTOR`
> did not fire. So region geometry governs the **data** axis as well as N-4's **capacity** axis;
> exact `k` predicts neither. Honest notes: a flooring artifact (grid starting at `n=8`, already
> sufficient) was caught and fixed (floor `n=4` + noisy regime); empirical, 1-D PL families,
> trainability bounded-not-removed by the best-of-restarts oracle. *(Local only; not committed.)*

*Claim.* N-4 showed the ε-essential region count tracks the generalization-**width** threshold
(modulo trainability). Extend the axis: at **fixed** width, the ε-essential count should also
predict the **sample complexity** — the training-set size at which held-out error clears the bar —
and/or the transfer gap to a second target family. If region geometry is the learnable invariant,
it should govern *data* as well as *capacity*.

*Traction.* Turns N-4 from a one-axis result into a two-axis one on the same harness
(`scripts/algo_approx_n4_eps_regions.py`), and re-tests the trainability residual: at fixed
(sufficient) width the SGD confound that dragged the jagged family should shrink, so the
sample-complexity prediction may be *cleaner* than the width one.

*Classical hook.* Sample-complexity / capacity bounds keyed to *effective* (ε) rather than exact
complexity; the bias-variance reading of region count.

*Falsifier* (`EPS_NOT_SAMPLE_PREDICTOR`): the sample-complexity threshold tracks the ε-essential
count no better than a trivial baseline (e.g. exact `k`, or a constant) — region geometry governs
capacity but not data.

*First move.* Reuse the N-4 harness; fix width at `≈ 2·(max ε-essential)` (representational
headroom), sweep `n_train`, define `sample_threshold(bar)` = smallest `n_train` clearing the bar,
and fit it against `ε-essential(bar)` vs exact `k`.

---

## S3-6 — Cancellation is the natural-proofs coordinate. [DAYDREAM]

*Claim.* N-2 read cancellation as the single imported-wall coordinate. The deepest version:
cancellation is *exactly where monotone (provable) lower bounds stop being "natural."* The lane
proves region-count lower bounds on the cancellation-free fragment (`isMono_hasPieceCover`,
`isMono_not_iterTent`, `monotone_transfer`) and never extends them to general nets — that
non-extension *is* the natural-proofs barrier, located at the negative scale.

*Traction.* Would tie the lane's cancellation spine to the Razborov–Rudich barrier as a *typed
location*, not a metaphor — predicting that any attempt to push the lane's lower bounds past the
cancellation-free fragment meets the same wall complexity theory already named.

*Classical hook.* Razborov–Rudich natural proofs; monotone vs general circuit lower bounds.

### Frontier marker — where Razborov–Rudich might (and might not) bite

This is the slate's outer edge: a *mapped daydream*, fenced on every side. The point is to mark the
frontier precisely — to say what would have to be true for the barrier to apply literally, and to
name where the lane's own machinery already lives in barrier territory.

**The structural rhyme (the honest part).** There is a clean correspondence between the lane's
spine and the classical monotone-vs-general story:

| lane (tropical / ReLU)                         | classical (Boolean circuits)                  |
|------------------------------------------------|-----------------------------------------------|
| negative scale = **cancellation**              | **negation** gate                             |
| `IsMono` fragment, region-polynomial           | monotone circuits (Razborov's clique bound)   |
| region/piece-count lower bound (provable)      | monotone-circuit size lower bound (provable)  |
| non-extension to general (`cancelMax > 0`)     | non-extension to general circuits             |

In both columns a lower bound is *provable on the monotone (cancellation/negation-free) side* and
*stalls at the boundary set by the sign-flip*. Razborov–Rudich names why the Boolean column stalls:
a **natural property** — one that is (i) *constructive* (decidable in time `2^{O(n)}` from a truth
table), (ii) *large* (a random function has it w.h.p.), (iii) *useful* (every function with small
general circuits lacks it) — that is useful against `P/poly` would break sub-exponential pseudo-
random generators. So no *natural* technique separates the monotone-provable bound from the general
one; **negation is the coordinate the barrier protects.** The daydream: *cancellation is the
tropical shadow of that coordinate.*

**`D-RR` — the speculative probe (might utilize R–R).** Two halves, with opposite honesty status:

- *Literal half (the lane's Boolean sub-artifacts).* The lane is **not** purely real-valued: it
  contains a machine-checked Boolean core — the `DecodingNPHard` chain (3SAT ≤ 3DM ≤ X3C ≤ syndrome
  decoding over GF(2)), the syndrome `Certificate`, and the find/check ledger's NP-membership side
  (`QueryGap`, `Certifies`). *There* `P/poly`, natural properties, and R–R are literally defined.
  A genuine probe: take the lane's `monotone_transfer`-style "provable on the easy fragment, not on
  the hard one" pattern and ask whether the obstruction it names instantiates a *natural property*
  in the R–R sense against the GF(2)-circuit class. If it does and is useful-against-general, R–R
  predicts a crypto consequence — i.e. it *can't*, which is exactly the barrier marking the wall.

- *Metaphor half (the real PL fragment) — fenced.* On the tropical/ReLU side there is **no native
  pseudorandom-generator / one-way-function notion**: a region-count "lower bound" is geometric, not
  a Boolean-circuit-size bound, and "large/useful natural property" has no crypto counterpart over
  ℝ. So a *literal* Razborov–Rudich statement on `Trop`/`Net` would be a **category error**. The
  daydream's claim here is only the analogy + a question: *is there a PRG-analog for the tropical
  class* (a pseudorandom family of PL targets indistinguishable from random by small ReLU nets)?
  Only if such an object existed would R–R transport literally. Marking that missing object *is* the
  frontier.

**Typed shape, if ever pursued (no theorem claimed).** A Lean-statable `NaturalProperty` predicate
bundling `Constructive ∧ Large ∧ Useful` over a function class, with the lane's `cancelMax` /
region-count offered as a *candidate* constructive-and-large property at the boundary — and the
honest output a **typed statement of non-transfer**, never a lower bound for general nets.

*Falsifiers.*
- (`MONOTONE_BOUND_TRANSFERS`) a monotone lower bound in the lane that *does* transfer to the
  general (cancellation-using) fragment — which would breach the barrier framing outright.
- (`RR_CATEGORY_ERROR`) treating the real-valued PL fragment as if it literally hosts R–R; the
  marker explicitly forbids this — the literal hook is the Boolean sub-fragment only.

*First move.* None actionable; mapped only. The whole value of this entry is the **frontier mark**:
the lane already touches barrier territory on its Boolean core, the cancellation spine names the
right coordinate, and the missing piece for a literal transport (a tropical PRG-analog) is now
written down as the thing that would have to exist. Daydream stays a daydream until that object does.

---

## What this slate is NOT

- **Not a Millennium / P-vs-NP attack.** S3-3's proven gap is in a *restricted query model*; it is
  not an unconditional separation and makes no P-vs-NP claim.
- **Not a learnability theory.** S3-5 is empirical and keeps the trainability factor explicit.
- **Not promoted.** Every entry is PROPOSED until a Lean core or experiment receipt lands.

## Recommended first attacks (ranked)

> **Slate status (2026-06-29): COMPLETE.** All five actionable hooks landed — S3-1/S3-2/S3-3/S3-4
> as axiom-clean, gated Lean cores (`ExactRepr`, `GradedCancellation`, `QueryGap`, `AnalyticGate` +
> `SawtoothApprox`), S3-4 at *both* the poly and polylog rates; S3-5 as an empirical CONFIRMS. S3-6
> is the closing daydream (frontier-marked, no build). Local/uncommitted; site deploy owner-gated.

1. **S3-1 — exact representability.** ✅ **CLOSED** — the full `⟺` is machine-checked
   (`ExactRepr.cpl_iff_reluNet`): converse `cpl_realizable` (peeling induction) + forward
   `net_hasPieceCover` (via the non-convex doubling bound `hasPieceCover_relu`). Continuous-PL ⟺
   exact finite ReLU net, at ε = 0. Only the optional minimal-width sharpening remains.
2. **S3-3 — the non-imported gap.** ✅ **LANDED** (`QueryGap.check_lt_find`): in the decision-tree
   model, CHECK = 1 query, FIND ≥ n (adversary), both machine-checked — the ledger's first proved
   `check ≪ find`, cracking its only standing "find imported" caveat (for this query-model toy).
3. **S3-2 — graded cancellation.** ✅ **LANDED** (`GradedCancellation.hasPieceCover_graded`):
   region count `≤ 4 ^ cancelMax e · leafCount e`, budget `0` ⇒ N-1 linear. Sharpened N-1/N-2 from
   a dichotomy to a quantitative dial, with the budget localized to cancellation-exposed folds.
4. **S3-4 — the analytic ε-rate.** ✅ **CLOSED, both rates.** First strike
   (`AnalyticGate.sq_eps_approx`, poly `O(1/√ε)`) *and* the hard step
   (`SawtoothApprox.sq_polylog_approx`, **polylog `O(log 1/ε)` depth** via the Telgarsky sawtooth).
   `RATE_NOT_POLYLOG` cleared — the paper's headline rate is machine-checked for `x²`.
5. **S3-5 — sample-complexity empirical.** ✅ **CONFIRMS** (`algo_approx_s35_sample_complexity.py`):
   ε-essential predicts sample complexity (Spearman 0.68 / 0.78) where exact `k` is flat (≈0) —
   region geometry governs the *data* axis as well as N-4's *capacity* axis.
6. **S3-6 — the natural-proofs daydream.** Frontier marked: a Razborov–Rudich frontier marker
   (`D-RR`) — literal on the lane's Boolean sub-core (`DecodingNPHard`/syndrome/`QueryGap`), analogy
   on the real PL fragment (a *tropical PRG-analog* is the named missing object), fenced by
   `RR_CATEGORY_ERROR`. Daydream-tier, no build — the closing edge of the slate.

> Cross-links: [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](ALGO_APPROX_CONJECTURE_SLATE_2.md) ·
> [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md) ·
> [`ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`](ALGO_APPROX_N4_EPS_REGIONS_RESULT.md) ·
> `Sundogcert/RegionPoly.lean` · `Sundogcert/Certifies.lean` · `Sundogcert/CircuitNet.lean` ·
> `Sundogcert/CancellationFree.lean`.
