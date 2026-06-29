# Algorithmic Approximation — Fresh Conjecture Slate (opened 2026-06-27)

> **What this is.** A working slate of conjectures that aim the lane's *machine-checked
> arithmetic* — exact tropical→ReLU compilation (`Sundogcert/CircuitNet.lean`) and the
> shared op-count ledger (`Sundogcert/StraightLineCost.lean`) — at well-known classical
> questions. The parent lane ([`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md)) closed
> its 6-hook slate; this is the next-horizon scratchpad to work off.
>
> **Discipline (inherited).** Nothing here is promoted. Each entry is stated so it can
> come back **NULL**: a claim, the traction the new arithmetic gives, the classical
> question it touches, a **named falsifier**, a **tier**, and a **first move**. A
> conjecture earns status only by a receipt (experiment) or a Lean core.
>
> **Provenance.** Authored inline from live session context, not a cold subagent
> fan-out; any one theme can be spun into a focused deep-dive agent on request.

## Tier legend

- **[FORMALIZABLE]** — a Lean target on the existing `CircuitNet`/`StraightLineCost`
  surface; could become an axiom-clean core in the public repo.
- **[IMPORT-AND-CHECK]** — the classical theorem already exists; the work is verifying
  the bridge is faithful *and naming exactly where it breaks*.
- **[EMPIRICAL]** — a runnable frontier-ML experiment with a falsifiable prediction.
- **[DAYDREAM]** — general/ambitious; mapped to mark the frontier, not yet actionable.

## The shared object (what "this fresh arithmetic" is)

A **tropical circuit** over `(min,+)` / `(max,+)` is exactly the ReLU-representable
piecewise-linear fragment: `max p q = q + relu(p−q)`, compiled exactly (ε = 0) by
`CircuitNet`, with linear depth and — via the sharing DAG `RProg` — linear **gate count**
(`compileToDag_gate_count`/`_eval`). `StraightLineCost` puts both the *constructive* gate
count and the *certificate verifier* op-count into one `costOf` ledger.

Two levers fall out:

1. **Exactness transfers classical tropical/monotone/min-plus results into *exact*
   statements about ReLU nets** (equalities, not bounds; lower bounds, not just upper).
2. **The find/check ledger gives a model in which construction-vs-verification gaps can
   sometimes be *proved*** — the lane's bounded-verification thesis, now with an exact
   arithmetic under it.

One load-bearing fact to keep in view throughout: **a ReLU net is a tropical *rational*
map** (relu + affine gives max-plus *with* subtraction), while the cleanly-transferable
classical bounds live over tropical *polynomials* (monotone, no cancellation). That gap
*is* the monotone-vs-general circuit wall — and it is where the most interesting entries
below earn or break.

---

## Theme A — Tropical geometry ⇄ exact ReLU expressivity

**C-A1 — Linear regions are an exact tropical invariant. [FORMALIZABLE] — LANDED 2026-06-27
(intrinsic-ness + max-gate anchor axiom-clean + audited). See `Sundogcert/RegionCount.lean`.**
*Claim.* The linear-region structure of the ReLU DAG `compile e` is an intrinsic invariant
of the source tropical function — equality, not bounds, because compilation is exact.
*Result.* **`realize1_compile`/`affine_compile_iff`** — the compiled ReLU net realizes the
*identical* 1-D function as the source tropical circuit (`compile_eval`), so *any*
region/affine property is shared: the region count is **realization-independent**. The
falsifier `REGIONS_NOT_INTRINSIC` dies *from exactness* — no geometry needed. **Region
anchor `max_two_not_affine`:** a tropical `max` of two distinct-slope affines is **not
globally affine** (≥ 2 regions), proved via "an affine function `≥ 0` everywhere has zero
slope" (`max ≥ gᵢ` forces the affine candidate's slope to equal both `a₁` and `a₂`).
Capstone **`compiled_maxGate_not_affine`**: the compiled ReLU net of that gate has ≥ 2
regions, intrinsically. All axiom-clean, in the AxiomAudit gate.
*Named wall (imported):* the exact region-count *formula* for a general tropical circuit
(tropical-hypersurface chambers / Newton-polytope mixed volume) is the geometry the
literature bounds rather than computes — proved here: intrinsic-ness + the atomic count.

**C-A2 — APSP regions index shortest-path trees. [IMPORT-AND-CHECK] — RAN 2026-06-27,
verdict SUPPORT (analytical; `NO_TREE_BIJECTION` does not fire).**
*Claim tested.* The compiled min-plus circuit's linear regions (over edge-weight parameter
space) biject with the distinct shortest-path trees; boundaries = parametric-SP breakpoints.
*Result.* The SSSP/APSP distance is `min over paths` of the path weight — a tropical
polynomial in the edge weights whose **argmin chambers** are exactly the regions of
parameter space on which a fixed optimal tree wins. That is precisely the classical
**parametric shortest path** structure (Carstensen 1983; Gusfield 1980): the breakpoints
where the optimal tree changes are the region boundaries, and the full-dimensional chambers
are in bijection with the distinct optimal **shortest-path trees** (ties = lower-dimensional
shared boundaries, the genericity caveat). By C-A1's `realize1_compile`, the *compiled ReLU
net* inherits this region structure exactly (same function), so its regions index
shortest-path trees too. The bijection holds up to degeneracy, so the falsifier does not
fire.
*Boundary (imported).* The *count* of regions = the number of distinct optimal trees is the
parametric-SP combinatorics (Carstensen/Gusfield), imported; a full Lean bijection would
need that machinery (the same bounds-only region-count wall as C-A1). Analytical close.

---

## Theme B — Unconditional ReLU lower bounds, and the cancellation barrier *(the heart)*

**C-B1 — Monotone `(min,+)` lower bounds transfer to the cancellation-free ReLU
fragment. [IMPORT-AND-CHECK] — PHASES 1+2 LANDED 2026-06-27 (fragment proper + reverse
compilation + imported-bound transfer; the Jerrum–Snir bound itself remains the named
wall, axiom-clean, audited).**
*Claim.* `CircuitNet` compiles tropical *polynomials* into ReLU exactly; if the *reverse*
holds on a "cancellation-free" ReLU fragment (no effective use of subtraction to cancel),
then the classical **unconditional** monotone `(min,+)`-circuit lower bounds
(Jerrum–Snir 1982 and successors — explicit `n^Ω(log n)` / exponential bounds for
shortest-path / spanning-tree / permanent-style functions) become **unconditional
ReLU-DAG gate-count lower bounds** for the same explicit functions — a rarity for neural
nets.
*Phase 1 (DONE) — `Sundogcert/CancellationFree.lean`:* the fragment is defined and proven
proper. `IsMono` = the monotone (max-plus polynomial) sub-grammar of `Trop` (no negative
scaling). **`monotone_of_isMono`** — a cancellation-free circuit computes a **monotone**
function (and `monotone_compile_of_isMono` — so does its compiled ReLU net). **The
barrier, witnessed:** `abs` is in the general fragment (`abs_in_general`) but not monotone
(`abs_not_monotone`), so **no cancellation-free circuit computes it** (`abs_not_isMono`).
Cancellation buys exactly the non-monotone functions — *that* is why the monotone bound is
not automatically a ReLU bound. Axiom-clean, wired into root + `AxiomAudit`.
*The honest barrier (this is the contribution).* A general ReLU net is tropical
*rational* and can use cancellation to beat any monotone bound (exactly as general
circuits beat monotone ones). The Phase-1 result *precisely locates* the monotone-vs-
general wall — and the Razborov–Rudich natural-proofs barrier — against a machine-checked
compiler.
*Classical hook.* Monotone circuit lower bounds; the monotone/general gap; natural proofs.
*Falsifier* (`CANCELLATION_DOMINATES`): for the candidate functions, small ReLU nets with
cancellation exist, and the cancellation-free fragment is too weak to be interesting → the
transfer is vacuous. (Held in spirit: the fragment is non-trivial — it is exactly the
monotone functions — and properly excludes `abs`.)
*Phase 2 (DONE 2026-06-27) — reverse compilation + imported-bound transfer:*
**`decompile`** (`Net → Trop`, `relu a ↦ max a 0`) + **`decompile_eval`** prove every ReLU
net is a tropical(-rational) circuit exactly — the converse of `compile`, completing
`Net ≅ Trop`. **`monotone_transfer`** is the imported-bound framing: given the Jerrum–Snir
monotone max-gate lower bound as an explicit hypothesis `hLB` (every `IsMono` circuit for
`f` has `≥ B` max-gates — the imported content), *every* cancellation-free ReLU net for `f`
has `≥ B` gates; the bridge is **`compileToDag_maxCount_ge`** (`maxCount e ≤` compiled gate
count, a lower bound mirroring `compileToDag_gate_count`). All axiom-clean, audited. **The
wall, named:** `decompile` sends a *general* net to a *general* (non-`IsMono`) `Trop`, so
the transfer covers the compile-image fragment but not all monotone-computing nets — the
monotone-vs-general / natural-proofs gap, sandwiched precisely between the two provable
halves, imported not crossed. The Jerrum–Snir bound itself stays the named import.

**C-B2 — A provable find-vs-check gap inside the tropical/monotone model. [FORMALIZABLE] —
RAN 2026-06-27, verdict BOUNDED: `CHEAP_MONOTONE_CONSTRUCTOR` *fires* for the shortest-path
instance. Analytical close. See [`ALGO_APPROX_CB2_FINDCHECK.md`](ALGO_APPROX_CB2_FINDCHECK.md).**
*Claim tested.* Exhibit a family where the monotone *constructor* cost is superpolynomially
above the *verifier* cost, both as `StraightLineCost` instances, unconditional in-model.
*Result.* With **`ShortestPathCert` as the verifier** (cheap side fully PROVED, `O(E)`), the
instance is shortest paths — which is in **P**, so it has a *polynomial* constructor
(Bellman–Ford `O(VE)`). The gap is therefore polynomial, an *exhibit* not a lower-bound
separation, and `CHEAP_MONOTONE_CONSTRUCTOR` fires. A **superpolynomial** in-model gap
requires a **monotone-hard** `f` (Jerrum–Snir) with a cheap verifier — i.e. exactly C-B1's
`monotone_transfer` with the bound **imported**; an *unconditional* superpolynomial
separation is not Lean-reachable (a complexity breakthrough). Net: the cost ledger proves
the **CHECK is cheap** (3 instances) and the **FIND-hardness is the imported wall**
everywhere — the same decomposition the P-vs-NP sibling lane rests on.
*Classical hook.* P vs NP, restricted to a model where one side is provable; the lane's
"safe to verify, hard to find" thesis.

---

## Theme C — The find/check ledger → certificates & fine-grained complexity

**C-C1 — Shortest-path tree as a cheap verification certificate (a tropical sibling of
the syndrome cert). [FORMALIZABLE] — CLOSED 2026-06-27, exact optimality certificate
(both phases axiom-clean, audited).**
*Claim.* A shortest-path feasible potential is a cheap-to-verify witness:
`costOf(verifier) = O(E + V)` (check source distance, all-edge relaxation *inequalities*,
and tightness on *tree* edges only), while *finding* it is the full APSP construction.
*Phase 1 — `Sundogcert/ShortestPathCert.lean`:* the `Reaches` walk relation, a `Feasible`
potential, and **`feasible_le_walk`** — a feasible potential **lower-bounds the weight of
every walk** (the LP-dual "nothing is shorter" half), by telescoping induction;
`verifyCost = 2·|E|+1` with a `HasStraightLineCost (SPInstance n)` instance — the **second
concrete instance of the `StraightLineCost` find/check ledger** (after the syndrome
verifier), alongside the ReLU-DAG gate count.
*Phase 2 — the exact certificate:* `TightTree` (each non-source vertex tight on an incoming
edge, with a `rank` decreasing to the source) + **`tree_achieves`** (the source reaches `v`
by a walk of weight exactly `dist v`, by induction on the rank bound) ⟹ **`cert_isLeast`**:
`dist v` is the `IsLeast` element of the walk-weight set — i.e. `dist` *is* the true
shortest-path distance, **exactly**, not merely a lower bound. The cheap `O(E + V)` check
certifies global optimality.
*Classical hook.* Certifying algorithms; the NP-style "verification is easy" half, machine-
checked on an optimization problem; LP duality / Bellman–Ford optimality for shortest paths.
*Falsifier* (`VERIFY_NOT_CHEAPER`): the soundest verifier is not asymptotically below the
constructor → no certificate-grade gap. (Held: `O(E+V)` check ≪ any search.) Also gives
C-B2 its verifier side.

**C-C2 — Fine-grained cornerstones as one cost-ledger family. [DAYDREAM→IMPORT] — RAN
2026-06-27, verdict `UNIFIES_AS_TROPICAL_COST_FAMILY` (typed-positive, bounded). Analytical
tabulation, no Lean. See [`ALGO_APPROX_CC2_FINEGRAINED.md`](ALGO_APPROX_CC2_FINEGRAINED.md).**
*Claim tested.* APSP, edit distance, OV and other SETH/fine-grained anchors share
tropical structure, compile to ReLU DAGs whose gate counts track their DP/brute-force size,
and the reductions preserve the cost measure.
*Result.* Two semirings, **both in the `CircuitNet` fragment**: `(min,+)` (APSP / min-plus
matmul / negative-triangle `Θ(n³)`; edit distance / LCS / Fréchet `Θ(n²)`) and Boolean
`(OR,AND)` on `{0,1}` = `(max,min)` (BMM; **OV = `max_{(i,j)} min_k(1−min(a_k,b_k))`**,
`Θ(n²d)` — worked out exactly). Gate counts = DP/brute-force sizes; the canonical reductions
(APSP≡min-plus-matmul≡neg-triangle; OV→edit-distance/LCS/Fréchet; SETH→OV) are
gate-count-preserving up to the fine-grained factor, so the falsifier does **not** fire.
*Boundary (named).* **3SUM is out of fragment** (needs exact additive cancellation, not
min/max/Boolean), as is the `n^ω` algebraic-matmul route (cancellation side — the same C-B1
monotone-vs-general wall). The fine-grained *hardness conjectures* (the lower bounds) stay
imported; the cost ledger gives construction *upper* bounds only.

---

## Theme D — Frontier ML theory

**C-D1 — Depth separations are tropical-depth separations. [IMPORT-AND-CHECK] — LANDED
2026-06-27 (constructive half axiom-clean + audited; the depth-vs-width lower bound is the
named import). See `Sundogcert/DepthSeparation.lean`.**
*Claim.* The classical PL depth-separation results (Telgarsky's sawtooth) are tropical-depth
separations: the witness is a tropical circuit of linear depth producing exponentially many
linear regions.
*Result.* The **tent map** `T(x)=1−|2x−1|` is in the tropical fragment (`tent_eval`); its
`d`-fold composition `iterTent d` (via `subst0`) is a tropical circuit computing `T^[d]`
(`iterTent_eval`) of depth **linear in d** (`iterTent_depth_le ≤ d·depth(tent)`; ReLU depth
`O(d)` via `compile_depth_le`). The punchline **`tent_iterate_dyadic`** proves
`T^[d](j/2^d) = parity(j)` — the `2^d+1` dyadic samples alternate `0,1,0,1,…`, so the
depth-`O(d)` circuit has **≥ 2^d linear pieces**: exponential expressivity from linear
depth, the formal core of "depth = computation". All axiom-clean, in the AxiomAudit gate.
*Falsifier* (`SEPARATION_NOT_TROPICAL`): a PL depth separation with no tropical-depth
explanation — did not fire; the canonical witness (sawtooth/tent) *is* a tropical circuit.
*Named wall (imported):* the matching depth-vs-**width** lower bound — that no shallow
sub-exponential-width net matches `2^d` regions — is Telgarsky 2016, imported (this module
proves the upper/achievability side).

**C-D2 — Grokking = SGD discovering `compileToDag`; the compiled-size threshold is the
first test. [EMPIRICAL] — RAN 2026-06-27, verdict INCONCLUSIVE (naive prediction NOT
supported; `GROKS_FAR_BELOW_COMPILED_SIZE` fires but confounded). See
[`ALGO_APPROX_CD2_GROKKING_RESULT.md`](ALGO_APPROX_CD2_GROKKING_RESULT.md).**
*Claim tested.* The trained ReLU MLP's generalization-width threshold grows `~ k-1` in the
tropical piece-count `k` (`compileToDag` size as the upper-bound target).
*Result.* threshold(k) for k=[2,3,4,6,8]: smooth `[3,3,5,4,4]`, essential `[3,3,3,4,5]`,
jagged (non-convex) `[3,3,10,3,5]` — none track `k`; far below `k` and/or erratic. No
delayed generalization (no "grokking") on PL regression. **Two confounds, themselves the
finding:** (i) **exact size ≠ ε-complexity** — convex/smooth targets are ε-approximable far
below `k` (the lane's exact-vs-ε seam); (ii) **existence ≠ trainability** — for jagged
targets SGD *fails to find* the representable solution at the capacity threshold (k=4 stuck
at rel≈0.68 until w=10), the lane's standing imported wall, hit empirically. So the exact
compiled size is an upper bound that does **not** predict the trained-generalization
threshold; the operative quantities are ε-approximation complexity and SGD trainability.
*Honest next step (not run):* decouple the confounds — measure the *representational*
threshold via a fitting oracle (exhaustive small-width fit), not SGD.

**C-D3 — Mechanistic interpretability has a canonical form on tropical tasks. [EMPIRICAL]**
*Claim.* A tropical task admits a canonical minimal ReLU circuit (`compileToDag` up to
wire/gauge symmetry), so mech-interp of a net trained on it reduces to **matching weights
to the canonical DAG** — interpretability as circuit-recognition.
*Falsifier* (`NO_GAUGE_MATCH`): a fully-generalizing trained net is provably *not* gauge-
equivalent to `compileToDag` (it found a genuinely different exact circuit) → no canonical
form.
*First move.* Take the C-D2 grokked sorting net and attempt an explicit gauge alignment to
`compileToDag`.

**C-D4 — The `4N` gate bound is a lottery-ticket / pruning target, not yet a floor.
[FORMALIZABLE + EMPIRICAL]**
*Claim.* DAG sharing (`compileToDag`) is what pruning seeks: `4N` is a proven exact
upper bound on the prunable size for an `N`-node tropical tree. The conjectural "floor"
is task-specific: for chosen families, prove a matching or near-matching
exact-computation lower bound, then test whether pruning stalls there.
*Falsifier* (`PRUNES_BELOW_PROVED_MIN`): after a real lower bound is proved for a target
family, an exact pruned subnetwork beats it → the lower-bound model is wrong.
*First move.* Prove a first exact-computation gate-count lower bound for a tiny nested
`min`/`max` chain, compare it with the `compileToDag` upper bound, and only then run the
pruning experiment against the proved band.

---

## What this slate is NOT

- **Not a Millennium attack.** Scope = tropical / piecewise-linear / monotone computation
  and ML theory. Number-theoretic questions (Riemann, etc.) are *off-substrate* — the
  arithmetic is about computation and optimization, not arithmetic geometry. Do not force
  them.
- **Not an unconditional P-vs-NP claim.** Every "lower bound" here lives in a **named
  restricted model** (tropical / monotone / cancellation-free). The honest P-vs-NP touch
  is the **monotone barrier** (C-B1/B2): unconditional in-model, provably non-extending to
  general circuits (natural proofs). That boundary is the feature, not a bug to paper over.
- **Not promoted.** Tiers are aspirations; status is earned only by a Lean core or an
  experiment receipt.

## Recommended first attacks (ranked)

0. **C-C1 — `ShortestPathCert`. ✅ CLOSED 2026-06-27** (both phases axiom-clean, audited):
   `feasible_le_walk` (dual-certificate lower bound) + `cert_isLeast` (the exact optimality
   certificate via `tree_achieves`) + the `O(E+V)` cost-ledger instance. The find/check
   ledger now has three instances (construct / check-syndrome / check-shortest-path); it
   also supplies C-B2's verifier side.
0. **C-B1 — cancellation-free fragment. ✅ CLOSED 2026-06-27** (both phases axiom-clean,
   audited): `IsMono` fragment + `monotone_of_isMono` + `abs` separation (Phase 1);
   `decompile`/`decompile_eval` (reverse compilation) + `monotone_transfer`
   (imported-bound framing) + `compileToDag_maxCount_ge` (Phase 2). The monotone-vs-general
   / natural-proofs wall is machine-located, sandwiched between the two provable halves;
   the Jerrum–Snir bound stays the named import.
0. **C-D2 — grokking-threshold experiment. ▢ RAN 2026-06-27, INCONCLUSIVE** (naive
   prediction unsupported; `GROKS_FAR_BELOW_COMPILED_SIZE` fires but confounded by
   exact-vs-ε and existence-vs-trainability). Documented negative —
   [`ALGO_APPROX_CD2_GROKKING_RESULT.md`](ALGO_APPROX_CD2_GROKKING_RESULT.md).

0. **C-D1 — depth separations = tropical-depth. ✅ LANDED 2026-06-27** (`DepthSeparation.lean`,
   axiom-clean + audited): the tent map is a tropical circuit, its `d`-fold composition is a
   depth-`O(d)` circuit (`iterTent_depth_le`) with `≥ 2^d` linear pieces
   (`tent_iterate_dyadic`) — exponential expressivity from linear depth ("depth =
   computation"); the depth-vs-width lower bound is the named import.

0. **C-C2 — fine-grained tropical cost-family. ✅ LANDED 2026-06-27** (analytical tabulation,
   `ALGO_APPROX_CC2_FINEGRAINED.md`): APSP/edit-distance `(min,+)` + OV/BMM Boolean-on-`{0,1}`
   all in the `CircuitNet` fragment, gate counts = DP sizes, reductions cost-preserving; 3SUM
   + `n^ω` algebraic route named out-of-fragment; hardness conjectures imported.

**Status:** eight hooks resolved — **five Lean closures** (C-C1 ✅, C-B1 ✅, C-D1 ✅, C-A1 ✅,
+ the foundational compiler), **three analytical closes** (C-C2 ✅ fine-grained family;
C-A2 ✅ APSP regions ↔ shortest-path trees; C-B2 ◐ bounded find/check), and **one documented
empirical negative** (C-D2 ▢). A recurring spine: every result proves a cheap CHECK /
construction *upper* bound (or an intrinsic-via-exactness fact) and names the *lower* bound
(decoding / monotone / fine-grained / depth-vs-width / exact-region-formula) as the imported
wall. Remaining open: only the empirical D-hooks (**C-D3** mech-interp canonical form,
**C-D4** pruning floor).

> **Follow-on:** [`ALGO_APPROX_CONJECTURE_SLATE_2.md`](ALGO_APPROX_CONJECTURE_SLATE_2.md)
> (opened 2026-06-28) mines these closures for the **cancellation spine** — the one
> wall (C-B1 / C-C2 / H-A1 / C-D1) — into four hooks N-1..N-4.
>
> Cross-links: [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md) (parent lane) ·
> [`SUNDOG_V_P_V_NP.md`](../SUNDOG_V_P_V_NP.md) (the find/check sibling) ·
> [`SUNDOG_V_CERTIFICATE_LEAN.md`](../SUNDOG_V_CERTIFICATE_LEAN.md) (the Lean method) ·
> `Sundogcert/CircuitNet.lean` · `Sundogcert/StraightLineCost.lean`.
