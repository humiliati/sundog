# Algorithmic Approximation — Fresh Conjecture Slate 4 (opened 2026-06-30)

> **What this is.** Slate 3 and the general-analytic-gate continuation are closed: the constructive
> Cor-5.1 ladder now runs end to end —
> [`ALGO_APPROX_CONJECTURE_SLATE_3.md`](ALGO_APPROX_CONJECTURE_SLATE_3.md) (exact representability,
> the analytic ε-rate, the graded dial, the proved query gap, the sample-complexity empirical) plus
> `AnalyticGate`/`SawtoothApprox` (x²) → `MultiplyGate` (ε-multiply) → `MonomialEval` (xᵈ) →
> `PolyEval` (any polynomial) → `UniversalApprox` (**every continuous f on [0,1] is ReLU-approximable**,
> Stone–Weierstrass imported, axiom-clean). This slate takes that capstone — *universal
> approximation* — and asks what it costs to make the approximation **certified, multivariate,
> cost-accounted, definable, and physically grounded.** The through-line: **from universal
> approximation to certified approximation.** Six hooks.
>
> **Discipline (inherited, unchanged).** Nothing is promoted without a receipt. Each entry is stated
> so it can come back **NULL**: a claim, the traction the existing machine-checked cores give, the
> classical question it touches, a **named falsifier**, a **tier**, and a **first move**. Status is
> earned only by a Lean core or an experiment. The standing boundary holds: **not a P-vs-NP /
> Millennium claim; the universal-approximation results are the *constructive* direction of classical
> facts (density imported from Weierstrass); "Lean-verified" means the deductive core.**
>
> **Frozen-lane note.** Two hooks (U-4 → the o-minimal H-A5 entry; U-6 → the Atlas halo lane) would
> **reopen lanes currently frozen-as-portfolio.** They are recorded here as candidates with honest
> tiers; *actually reopening either is an owner decision* — this slate states their status, it does
> not argue for the reopen.
>
> **Provenance.** Hooks U-1…U-6 proposed by the owner; vetting (tier / falsifier / traction / first
> move / honest scope) authored inline against the just-landed capstone modules.

## Tier legend

- **[FORMALIZABLE]** — a Lean target on the existing `UniversalApprox` / `PolyEval` / `MultiplyGate`
  / `Certifies` / `StraightLineCost` surface; could become an axiom-clean, gated core.
- **[FORMALIZABLE-HARD]** — formalizable, but needs a genuinely new construction or a non-trivial
  mathlib extraction (not just assembling existing lemmas).
- **[EMPIRICAL]** — a runnable CPU experiment with a falsifiable prediction.
- **[DAYDREAM]** — mapped to mark the frontier (or gated on a frozen-lane reopen / a missing mathlib
  substrate), not yet actionable.

## The seams the capstone left open

1. Universal approximation is **1-D only** (`f : C([0,1], ℝ)`) — the paper and the application both
   want **[0,1]ⁿ** (U-1).
2. The approximant is **produced** (an explicit net with a proved bound) but the lane's **find/check
   ledger never ingests it** — there is no `ApproxCert` instance saying "this net ε-approximates that
   target, cheaply checkable" (U-2).
3. The rate bounds are stated in **depth** (and a loose width), not in the lane's own **op-count cost
   measure** (`StraightLineCost`) — the construction and the cost ledger are not yet wired together
   (U-3).
4. ReLU approximants are **semialgebraic / definable**, but the lane never connects that to the
   o-minimal **H-A5** definability hook from Slate-1 (U-4).
5. Every approximant here is **constructed**, never **trained** — the representation-vs-optimization
   gap (the residual N-4/S3-5 keep naming) is untested head-to-head (U-5).
6. The capstone is proved for **abstract** continuous f; it has **no physical witness** — one real
   continuous curve from the portfolio would make it concrete (U-6).

---

## U-1 — Multivariate cube lift: `[0,1]ⁿ` universal approximation. [FORMALIZABLE-HARD]

*Claim.* Extend `continuous_relu_approximable` from `[0,1]` to `[0,1]ⁿ`: every
`f ∈ C([0,1]ⁿ, ℝ)` is uniformly ε-approximable by an explicit ReLU `Net n`, via (a) an explicit
**multivariate** polynomial evaluator built from the `MultiplyGate` (monomials `x₁^{a₁}…xₙ^{aₙ}` as
products of the per-coordinate `MonomialEval` powers, summed in the monomial basis exactly), and (b)
**multivariate Stone–Weierstrass** (mathlib) for the density input. The 1-D capstone is the `n = 1`
instance.

*Traction.* The hard half is *already done*: `MultiplyGate.mult_polylog` gives products and
`MonomialEval`/`PolyEval` give the per-axis powers and the exact linear combination. The new content
is (i) an `n`-input monomial evaluator (`graft`/`subst` generalized to `Net n`, a product of
single-axis power-nets via the multiply gate) and (ii) extracting a concrete approximating
polynomial from mathlib's multivariate density (`ContinuousMap.subalgebra_topologicalClosure_eq_top_of_separatesPoints`
applied to multivariate `polynomialFunctions`, plus an `MvPolynomial`→coeff-table bridge mirroring
`UniversalApprox.polyVal_coeffList`).

*Classical hook.* Multivariate universal approximation (Cybenko / Hornik in the density form;
Yarotsky in the rate form); Stone–Weierstrass on a compact product.

*Falsifier* (`MULTIVARIATE_BRIDGE_FAILS`): either the `MvPolynomial`→ReLU-`Net n` evaluator's depth
or op-count is provably **super-polynomial in n at fixed degree** (the cube lift does not stay an
explicit construction), or mathlib's multivariate density cannot be extracted into a concrete
polynomial the evaluator can ingest (the bridge `MvPolynomial.eval = netVal` does not close).

*First move.* `monoTropN : (Fin n → ℕ) → Trop n` (a monomial `∏ xᵢ^{aᵢ}` as a multiply-tree over
the `MonomialEval` axis powers); `polyValN`/`polyTropN` over a finite coefficient table; prove
`polyTropN_approx` (errors add) and the `MvPolynomial` bridge; then `continuous_relu_approximable_cube`
via mathlib multivariate Stone–Weierstrass. Hard part: the multivariate density extraction and
keeping the monomial count / depth honest in n.

*Honest scope.* Still the **constructive direction of a classical fact** (multivariate universal
approximation); not a new approximation theorem, not a curse-of-dimensionality result (the monomial
count is exponential in n at fixed degree — state that bound, do not hide it).

---

## U-2 — Approximation certificates: extend `Certifies` to `ApproxCert`. [FORMALIZABLE]

*Claim.* The find/check ledger (`Certifies`, 7 exact instances) only certifies **exact** optima. Add
an **approximation** certificate: an `ApproxCert` packaging *(target f, net g, ε, a proof-carrying
L∞ bound)* whose **CHECK** verifies `∀ x ∈ K, |f x − g x| ≤ ε` cheaply, while a **FIND** produces the
net. For the lane's *constructed* approximants (`sq_eps_approx`, `poly_polylog`,
`continuous_relu_approximable`) the FIND side is **also in Lean** — so `ApproxCert` becomes the
ledger's second instance (after `QueryGap`) where *neither* side is imported.

*Traction.* Reframes the whole capstone as a ledger entry: the proved bounds (`*_approx`) *are* the
certificate payloads. It also forces the honest question the exact ledger never had to face — **what
makes an approximation check "cheap"** when the target is a continuum (see falsifier).

*Classical hook.* Certifying algorithms (Mehlhorn–Näher) generalized from exact to ε-optimal
outputs; a-posteriori error certificates in numerical analysis.

*Falsifier* (`APPROX_CHECK_NOT_CHEAP`): verifying `|f − g| ≤ ε` over a continuum reduces to
re-deriving the original bound (no separable cheap check exists) **unless** a structural hypothesis is
smuggled in (a Lipschitz/modulus bound on `f − g` + a finite sample net) — i.e. `ApproxCert`
collapses into "re-prove the theorem," with no genuine check/find factorization.

*First move.* Define `ApproxCert (K) (f) := { g : Net _, ε : ℝ, bound : ∀ x ∈ K, |f x − g x| ≤ ε }`
as a `Certifies`-shaped structure; register `sq_eps_approx` / `poly_polylog` as instances; then probe
the cheap-check question with the Lipschitz+grid route (CHECK = verify the bound on a δ-net + a
modulus bound ⇒ the continuum bound) and see whether the check is *strictly* cheaper than the find.

*Honest scope.* The interesting deliverable is the **honest verdict on the cheap-check question**,
not a foregone "yes"; if `APPROX_CHECK_NOT_CHEAP` fires, that null is the result.

---

## U-3 — Unified cost: wire `UniversalApprox` depth/size into `StraightLineProgram.cost`. [FORMALIZABLE]

*Claim.* The capstone bounds are in **depth** (and a loose width); the lane already has an op-count
cost measure (`StraightLineCost`: `compileToDag_cost_le`, the shared construction/verification
ledger). Connect them: bound the **actual gate count** of `sqNet` / `powNet` / `polyNet` via
`compileToDag`, giving each approximant a **certified op-count** `cost ≤ O(deg · log 1/ε)` (or the
honest multivariate analogue), not just a depth bound.

*Traction.* Mostly **assembly** of existing lemmas: the approximant nets already `compile`, and
`StraightLineCost.compileToDag_cost_le` already bounds the DAG op-count of a compiled circuit. The
new content is the leaf/gate count of the specific approximant circuits (a `size` recursion on
`Rcirc` / `powTrop` / `polyTropFrom`) fed into the existing cost bound — turning the lane's two
halves (approximation rate + cost ledger) into one statement.

*Classical hook.* Circuit size vs depth; the size–depth trade-off for ReLU approximants (the paper's
Cor 5.1 is a *size* statement).

*Falsifier* (`COST_BOUND_VACUOUS`): the compiled approximant's op-count, via `compileToDag`, is not
sub-quadratic in the depth bound (sharing fails to control size), so the unified cost statement is
strictly weaker than the standalone depth bound — no real unification.

*First move.* `size : Trop n → ℕ` (gate count); prove `size (Rcirc m) ≤ O(m)`,
`size (polyTropFrom …) ≤ …`; compose with `StraightLineCost.compileToDag_cost_le` to land
`polyNet_cost_le` / `sqNet_cost_le`; add to the gate.

*Honest scope.* A consolidation result, not a new rate; its value is that the approximant's cost is
now in the **same** certified ledger as the find/check checkers.

---

## U-4 — Definable approximant floor: ReLU/semialgebraic as the o-minimal H-A5 bridge. [DAYDREAM — frozen-lane reopen]

*Claim.* Every ReLU net realizes a **semialgebraic** (indeed piecewise-linear) function, so the whole
approximant family lives inside an **o-minimal** structure. Slate-1's **H-A5** raised definability as
a frontier; this hook would make it concrete: *the universal-approximation construction is a
definability statement* — continuous f is approximated by uniformly-definable (semialgebraic)
functions, and the approximation rate is a tameness/complexity parameter inside the o-minimal
structure.

*Traction.* Conceptually clean (PL = semialgebraic is immediate), and it would connect the lane to
the geometry/definability vocabulary. **But the substrate is missing:** mathlib has no usable
o-minimality / semialgebraic-tameness API to formalize against, so a *machine-checked* statement
beyond "PL is semialgebraic" (which is trivial and adds nothing) is not currently reachable.

*Classical hook.* o-minimal structures; semialgebraic geometry; tame topology (Wilkie / van den
Dries); definable approximation.

*Falsifier* (`DEFINABILITY_ADDS_NOTHING`): the only Lean-statable content is the trivial "PL ⊆
semialgebraic," and no checkable consequence (a tameness bound, a uniform-definability rate) follows
without an o-minimal substrate that mathlib does not provide — the bridge is a relabeling, not a
result.

*First move.* None actionable in Lean today; **gated on (a) the owner reopening H-A5 and (b) an
o-minimal/semialgebraic substrate existing in mathlib.** Until then: mark the frontier (record that
the approximant family is uniformly semialgebraic, name the missing substrate), exactly as S3-6 marks
the Razborov–Rudich frontier.

*Honest scope.* Daydream-tier and **reopen-gated**; recorded to keep the H-A5 thread visible, not to
argue for the reopen.

---

## U-5 — Representation-vs-training gap: constructed net vs SGD-trained net. [EMPIRICAL]

*Claim.* Every approximant in the lane is **constructed** (explicit weights with a proved bound),
never **trained**. On the *same* target and *same* architecture/budget, compare the constructed net
(achieving the proved `O(log 1/ε)` / `O(deg·log 1/ε)` rate) against an SGD-trained net: measure the
**representation-vs-optimization gap** — does SGD reach the rate the construction proves is
attainable, and where does it fall short?

*Traction.* This is exactly the trainability residual N-4 and S3-5 kept naming, now isolated as the
*primary* measurement instead of a confound. Targets are ready (`x²`, a fixed polynomial, a jagged
PL family); the constructed nets are the lane's own; the SGD side reuses the N-4/S3-5 harness
(`scripts/algo_approx_*`). The interesting outcome is the **gap**: construction hits the rate, SGD
may plateau above it.

*Classical hook.* Representation vs optimization in deep learning; the gap between approximation-
theoretic existence and gradient-trainability; lottery-ticket / spectral-bias readings.

*Falsifier* (`NO_REPRESENTATION_GAP`): SGD matches the constructed net's error at the same size on
all tested targets (no measurable gap — approximation-theoretic existence and trainability coincide
here), **or** the comparison is dominated by tuning artifacts (the "gap" is an optimizer-hyperparameter
effect, not a structural one) — in which case the null is reported.

*First move.* For each target, instantiate the constructed net (read its proved error), then train a
same-width net by SGD (best-of-restarts, matched budget); report `error_SGD / error_constructed` and
the size at which SGD first matches the construction. Reuse the S3-5 oracle-restart discipline to keep
trainability bounded-not-confounded.

*Honest scope.* Empirical, small 1-D families; measures a *gap*, claims no learnability theorem.

---

## U-6 — Atlas surrogate: one compact continuous halo branch as a physical short-program test case. [EMPIRICAL — frozen-lane reopen] ✅ DONE 2026-06-30

> **Status: LANDED (physical witness + honest size verdict).** Owner reopened the Atlas lane for this
> single worked witness. Result:
> [`ALGO_APPROX_U6_ATLAS_SURROGATE_RESULT.md`](ALGO_APPROX_U6_ATLAS_SURROGATE_RESULT.md) ·
> [`scripts/algo_approx_u6_atlas_surrogate.py`](../../scripts/algo_approx_u6_atlas_surrogate.py).
> Curve = the repo's own `h-of-x` relation `offset = R₂₂/cos(h)`, `h ∈ [0°,60°]`, normalized
> `g(x)=sec(πx/3)−1` on `[0,1]` (C∞, single-valued, pole at `x=1.5` outside domain → `ATLAS_BRANCH_NOT_CLEAN`
> did **not** fire). The **constructed** ReLU net (sawtooth-square + polarization + `clamp01`, the
> lane's Lean cores) tracks the near-minimax polynomial to `~1e-9` on the real curve (the witness);
> polynomial convergence is geometric at observed **3.726×/degree** vs predicted `ρ=2+√3=3.732`
> (free analytic cross-check). **Honest size verdict:** for this smooth 1-D curve a naive shallow
> linear spline is the *smaller* net until an extreme crossover `ε*~1e-12` — the construction's
> `O(deg·log 1/ε)` advantage is real but constant-heavy (`54m` gates/square, `O(d²)` squares); it does
> not contradict the `x²` depth-separation, it locates where the whole-curve pipeline pays its
> constants. A single physical witness, no halo / approximation-theory claim.

*Claim.* The capstone proves universal approximation for *abstract* continuous f. Ground it once:
take **one** compact, continuous, 1-parameter branch from the Atlas (a halo-geometry curve — e.g. a
single classified bifurcation branch as a function of the one driving parameter) as a concrete target
`f`, run `PolyEval` / `continuous_relu_approximable` on it, and exhibit the **explicit short-program
(ReLU net) approximant** with its measured error — a physical witness that the constructive ladder
approximates a real curve from the portfolio.

*Traction.* Turns the capstone from a theorem into a worked example on real data, and tests the
"short program for a physical curve" reading directly. The approximation machinery is done; the work
is (a) extracting one clean continuous branch from the Atlas as sampled `(param, value)` data and (b)
running the constructed approximant + reporting error/size.

*Classical hook.* Minimum-description-length / short-program readings of physical curves; surrogate
modeling of a simulator output by a small net.

*Falsifier* (`ATLAS_BRANCH_NOT_CLEAN`): the chosen halo branch is not cleanly continuous / single-
valued in one parameter (folds, multivaluedness, discontinuities at catastrophe points), so it is not
a fair `C([0,1])` target without surgery — the "physical test case" needs so much preprocessing that
it stops witnessing the capstone.

*First move.* None until the Atlas lane is reopened. **Gated on the owner reopening Atlas;** then:
sample one bifurcation branch over its parameter range, normalize to `[0,1]`, fit with `PolyEval`'s
constructed net, report L∞/L² error and net size against a baseline.

*Honest scope.* Empirical application, **reopen-gated** (touches the frozen Atlas lane); a single
worked witness, not a claim about halos or about approximation theory.

---

## What this slate is NOT

- **Not a P-vs-NP / Millennium attack.** The certificate and cost hooks live in the deductive
  CHECK/cost ledger; no separation is claimed.
- **Not a new approximation theorem.** U-1's multivariate lift is the *constructive* direction of a
  classical fact (density imported from Stone–Weierstrass), as the 1-D capstone was.
- **Not a learnability theory.** U-5 is empirical and keeps the optimizer factor explicit.
- **Not a frozen-lane reopen by default.** U-4 (H-A5) and U-6 (Atlas) are recorded as candidates;
  reopening either is an owner decision, stated here, not argued for.
- **Not promoted.** Every entry is PROPOSED until a Lean core or experiment receipt lands.

## Buildability order (cleanest first strike on the current surface — *not* a priority ranking)

1. **U-3 — unified cost.** [FORMALIZABLE] Mostly assembly: a `size` recursion fed into the existing
   `StraightLineCost.compileToDag_cost_le`. Lowest new-construction cost; lands the approximants in
   the lane's own op-count ledger.
2. **U-1 — multivariate cube lift.** [FORMALIZABLE-HARD] The direct generalization of the capstone;
   reuses `MultiplyGate`/`MonomialEval`/`PolyEval`. Hard part is the `MvPolynomial`→net bridge and the
   multivariate density extraction.
3. **U-2 — `ApproxCert`.** [FORMALIZABLE] Reframes the capstone as a ledger entry; the honest payoff
   is the cheap-check verdict (which may come back `APPROX_CHECK_NOT_CHEAP`).
4. **U-5 — representation-vs-training gap.** [EMPIRICAL] Reuses the N-4/S3-5 harness; isolates the
   trainability residual as the primary measurement.
5. **U-4 — definability / H-A5 bridge.** [DAYDREAM] Frontier-marked; gated on an o-minimal substrate
   in mathlib **and** an owner H-A5 reopen.
6. **U-6 — Atlas surrogate.** [EMPIRICAL] Gated on an owner Atlas reopen; a single physical witness
   once a clean branch is available.

> Cross-links: [`ALGO_APPROX_CONJECTURE_SLATE_3.md`](ALGO_APPROX_CONJECTURE_SLATE_3.md) ·
> [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md) ·
> `Sundogcert/UniversalApprox.lean` · `Sundogcert/PolyEval.lean` · `Sundogcert/MultiplyGate.lean` ·
> `Sundogcert/MonomialEval.lean` · `Sundogcert/StraightLineCost.lean` · `Sundogcert/Certifies.lean`.
