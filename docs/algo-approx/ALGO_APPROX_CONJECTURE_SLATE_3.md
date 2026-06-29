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

> **Status (2026-06-29): the convex case of the converse is LANDED.**
> `Sundogcert/ExactRepr.lean` — `convexCPL_realizable`: `ConvexOn ℝ Set.univ f → HasPieceCover f n →
> ∃ g : Net 1, realize1N g = f`. A convex continuous-PL function is the *exact* realization of a
> finite ReLU net, built canonically as the upper envelope of its piece-lines
> (`convex_eq_sup_lines`) realized by a tropical `max`-circuit (`linesTrop`) and compiled through
> `compile`/`realize1_compile`. Axiom-clean, in the `AxiomAudit` gate, full build green (8531 jobs).
> The general (non-convex) converse — `f` as a difference of two convex envelopes (DC
> decomposition) — and the forward direction (ReLU ⟹ CPL, via a general non-convex `max`/`relu`
> piece bound) remain the staged extensions; `CPL_NOT_REPRESENTABLE` did not fire on the convex
> case. *(Local only; not committed — owner-gated.)*

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
already imports the monotone-circuit bound (`monotone_transfer`) and never extends it to general
nets — that non-extension *is* the natural-proofs barrier, located at the negative scale.

*Traction.* Would tie the lane's cancellation spine to the Razborov–Rudich barrier as a *typed
location*, not a metaphor — predicting that any attempt to push the lane's lower bounds past the
cancellation-free fragment meets the same wall complexity theory already named.

*Classical hook.* Razborov–Rudich natural proofs; monotone vs general circuit lower bounds.

*Falsifier* (`MONOTONE_BOUND_TRANSFERS`): a monotone lower bound in the lane that *does* transfer
to the general (cancellation-using) fragment — which would breach the barrier framing.

*First move.* None actionable; mapped only. If pursued, the honest target is a *typed statement of
non-transfer* (the barrier's location), never a lower bound for general nets.

---

## What this slate is NOT

- **Not a Millennium / P-vs-NP attack.** S3-3's proven gap is in a *restricted query model*; it is
  not an unconditional separation and makes no P-vs-NP claim.
- **Not a learnability theory.** S3-5 is empirical and keeps the trainability factor explicit.
- **Not promoted.** Every entry is PROPOSED until a Lean core or experiment receipt lands.

## Recommended first attacks (ranked)

1. **S3-1 — exact representability.** ✅ **convex case LANDED** (`ExactRepr.convexCPL_realizable`);
   next increment is the general non-convex converse (DC decomposition) and/or the forward
   direction (ReLU ⟹ CPL). Cleanest: most of the machinery was in `RegionPoly`; closes the paper's
   title at ε = 0 (convex) and tightens N-1.
2. **S3-3 — the non-imported gap.** Highest value: the ledger's only honesty caveat is "find
   imported"; one proven `check ≪ find` (even in a query model) cracks it.
3. **S3-2 — graded cancellation.** Sharpens N-1/N-2 from a dichotomy to a quantitative dial.
4. **S3-5 — sample-complexity empirical.** Cheapest to run; extends N-4 to a second axis.
5. **S3-4 — the analytic ε-rate.** The paper's headline, but a genuinely new construction +
   error analysis; do it when there's appetite for the harder build.
6. **S3-6 — the natural-proofs daydream.** Framing only; revisit after S3-2/S3-3 land.

> Cross-links: [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](ALGO_APPROX_CONJECTURE_SLATE_2.md) ·
> [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md) ·
> [`ALGO_APPROX_N4_EPS_REGIONS_RESULT.md`](ALGO_APPROX_N4_EPS_REGIONS_RESULT.md) ·
> `Sundogcert/RegionPoly.lean` · `Sundogcert/Certifies.lean` · `Sundogcert/CircuitNet.lean` ·
> `Sundogcert/CancellationFree.lean`.
