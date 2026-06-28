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

**C-A1 — Linear regions are an exact tropical invariant. [FORMALIZABLE]**
*Claim.* The number of linear regions of the ReLU DAG `compileToDag e` equals a purely
combinatorial invariant of the source tropical circuit `e` (chambers of the associated
tropical-hypersurface arrangement / vertices of a Newton polytope) — an **equality**,
because compilation is exact, where the region-count literature (Montúfar–Pascanu–Cho–
Bengio; Serra–Tjandraatmadja–Ramalingam) gives only bounds.
*Classical hook.* Counting linear regions / the complexity of PL functions; tropical
Bézout and mixed volumes.
*Falsifier* (`REGIONS_NOT_INTRINSIC`): the region count of `compileToDag e` depends on the
realization, not on `e`'s tropical data alone → no clean invariant.
*First move.* Prove it for one `max` gate (2 regions) and for `bellmanStep` over `k`
incoming edges (`≤ k+1` candidate regions, including the incumbent distance, with
degeneracies allowed), defining a `linearRegions` count on `RProg` in Lean.

**C-A2 — APSP regions index shortest-path trees. [IMPORT-AND-CHECK]**
*Claim.* The exact ReLU compilation of the min-plus APSP circuit has linear regions in
bijection with the distinct **shortest-path trees** of the input graph; the region
boundaries are exactly the parametric-shortest-path breakpoints.
*Classical hook.* Parametric shortest paths and the number of distinct optima
(Carstensen, Gusfield) — a classical combinatorial-optimization quantity.
*Falsifier* (`NO_TREE_BIJECTION`): regions and shortest-path trees fail to biject on a
small explicit graph.
*First move.* Enumerate regions of `compileToDag (bellmanStep …)` for `k = 3,4` and
compare to the shortest-path-tree count.

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

**C-B2 — A provable find-vs-check gap inside the tropical/monotone model. [FORMALIZABLE]**
*Claim.* Exhibit an explicit family `f_n` where, in the `costOf` ledger,
`costOf(any tropically-monotone net computing f_n)` is superpolynomially larger than
`costOf(a verifier checking a supplied witness for f_n)` — a **provable, unconditional**
find ≫ check separation **in the named restricted model**, sidestepping P≠NP exactly as
the P-vs-NP lane's bounded version does, but now with both sides as `StraightLineCost`
instances.
*Classical hook.* P vs NP, restricted to a model where one side is provable; the lane's
"safe to verify, hard to find" thesis.
*Falsifier* (`CHEAP_MONOTONE_CONSTRUCTOR`): every candidate `f_n` also has a small
monotone constructor → no gap in-model.
*First move.* Reuse a Jerrum–Snir hard function as the constructor side; build its
witness-verifier as a tropical `RProg` and bound its `costOf`.

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

**C-C2 — Fine-grained cornerstones as one cost-ledger family. [DAYDREAM→IMPORT]**
*Claim.* APSP, edit distance, orthogonal-vectors and other SETH/fine-grained anchors
share min-plus/tropical structure; they compile to ReLU DAGs whose gate counts track their
fine-grained hardness, and fine-grained reductions are (approximately) `costOf`-preserving
maps.
*Classical hook.* Fine-grained complexity — the APSP conjecture, the OV conjecture, SETH.
*Falsifier* (`REDUCTIONS_NOT_COST_PRESERVING`): the standard reductions blow up gate count
beyond the conjectured factor.
*First move.* Tabulate the known min-plus formulations and their gate counts under
`CircuitNet`; check one reduction (OV → …) for cost preservation by hand.

---

## Theme D — Frontier ML theory

**C-D1 — Depth separations are tropical-depth separations. [IMPORT-AND-CHECK]**
*Claim.* The classical PL depth-separation results (Telgarsky's sawtooth; Eldan–Shamir)
are, for piecewise-linear targets, exactly **tropical-circuit depth lower bounds**, made
tight by exact compilation: every such separation has a tropical-depth witness and
`CircuitNet` realizes the optimal-depth net.
*Classical hook.* Why depth helps; depth-vs-width separations.
*Falsifier* (`SEPARATION_NOT_TROPICAL`): a PL depth separation with no tropical-depth
explanation.
*First move.* Re-derive Telgarsky's sawtooth as a tropical circuit and read its depth off
`compile_depth_le`.

**C-D2 — Grokking = SGD discovering `compileToDag`; the compiled-size threshold is the
first test. [EMPIRICAL]**
*Claim.* On algorithmic tasks with exact tropical targets (sorting, min/max-plus matrix
product, single-source shortest path), delayed generalization ("grokking") is SGD
converging to a symmetry-class of the exact compiled DAG. The `≤ 4N` gate count is a
certified exact **upper** bound and the first numeric threshold to test; it becomes a
floor only for tasks where an independent lower bound is proved. Prediction: generalization
onset should cluster near the minimal exact-circuit size, with `compileToDag` as the
initial upper-bound target.
*Classical/frontier hook.* Grokking, algorithmic generalization, phase transitions in
training.
*Falsifier* (`GROKS_FAR_BELOW_COMPILED_SIZE`): clean generalization occurs far below the
compiled exact DAG size, without a smaller exact circuit being identified → the compiled
size is not the threshold driver.
*First move.* Train a small MLP on min-plus 4×4 matrix product; sweep width across the
compiled-size band; check generalization onset and whether learned weights match
`compileToDag` up to permutation. Separately, prove or enumerate minimal exact size for
the tiny task before calling any threshold a floor.

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
1. **C-D2 — the grokking-threshold experiment.** Runnable frontier-ML with a sharp,
   falsifiable numeric target (the compiled-size band first; a true floor only after a
   lower bound); fast feedback, high signal. **The top open first-strike.**

> Cross-links: [`SUNDOG_V_ALGO_APPROX.md`](SUNDOG_V_ALGO_APPROX.md) (parent lane) ·
> [`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md) (the find/check sibling) ·
> [`SUNDOG_V_CERTIFICATE_LEAN.md`](SUNDOG_V_CERTIFICATE_LEAN.md) (the Lean method) ·
> `Sundogcert/CircuitNet.lean` · `Sundogcert/StraightLineCost.lean`.
