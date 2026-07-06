# Find/Check Sufficiency for Abstraction — conjecture slate (2026-06-29)

> Generate→adversarially-vet→rank, in the house format (cf. `docs/atlas/FRESH_HYPOTHESES_SLATE_2.md`,
> `docs/algo-approx/ALGO_APPROX_CONJECTURE_SLATE_2.md`). **The seed:** the algo-approx headway machine-checked a
> NEW lens the ARC lane never used. ARC was driven through every *learner-sufficiency* framing (Branch A
> stochastic per-task, B threshold revision, C pause, D structured-edit/residual + variants, Phase 3E) — all
> hit full-grid-control floors; the only admissible reopen paths left are "a new Branch D variant or a Branch E
> spec." Meanwhile the algo-approx lane proved, in Lean, the **find/check (search-vs-verify)** axis:
> `QueryGap.lean` (the ledger's first *non-imported* find/check gap, CHECK O(1) vs FIND Ω(n) in a restricted
> query model), the `Ledger`/`Certifies` class (7 cheap-verifier instances), and the suffstat-order **σ
> order-meter**. ARC is *fundamentally* a find/check problem (verify a candidate program cheaply; find it is the
> hard search). This slate develops that lens on **toy task families we own** and stages an ARC **Branch E**
> reopen *gated* on it landing — a FRESH SIBLING ledger, NOT a reopen (the algo-approx pattern).
> **Discipline:** REFRAME + toy-first; the unconditional find-hardness (P≠NP) and ARC's own A–D/Phase-3E floors
> are NAMED IMPORTED WALLS; nothing here claims to crack ARC or separate complexity classes. NOT public-eligible.

## The bridge (definitions-level; the lens, not a theorem about ARC)

| algo-approx asset (machine-checked) | abstraction-task reading |
|---|---|
| `Ledger`/`Certifies` cheap verifier (`checkCost` straight-line) | CHECK: apply a candidate DSL program to train inputs, exact-match outputs — O(grid·\|p\|) |
| `QueryGap.check_lt_find` (CHECK O(1), FIND Ω(n), restricted model) | FIND: search the DSL for a program reproducing all train pairs — the hard side |
| suffstat-order σ (least-order sufficient statistic; SCHEMA not scalar) | task hardness coordinate: least order of feature that suffices to solve the task |
| "FIND-hardness imported" caveat (P≠NP wall) | why no ARC learner family cracked it — the search is the wall, not the representation |

**The reframing claim (the lens):** the ARC Branch-A–D floors are plausibly **find/check gaps** — cheap-to-verify,
hard-to-search task families — *not* representation-insufficiency verdicts. The deterministic-low-capacity and
edit/residual learners failed the FIND, while CHECK was always cheap. This is a coordinate the learner-sufficiency
branches never instrumented.

**Named imported walls (stated, NOT breached):** unconditional super-polynomial find/check separation (P≠NP) is
out of reach (every `Ledger` instance imports the find-hardness; `QueryGap` is only a *restricted-model*
separation — that is *why* it is provable). ARC's A–D + Phase-3E floors are imported as the empirical record this
lens must explain, not re-litigate.

## Survivors (ranked)

### Rank 1 — FC-2: a QueryGap-style UNCONDITIONAL find/check separation for a toy abstraction family (strength 6)
**The deepest, most novel hook: a synthetic abstraction task class where verifying a candidate transform is cheap
but FINDING it is provably hard in a bounded oracle model** — the search-vs-verify gap the ARC floors instantiate,
proved (not imported) on a toy we own.
- **Core (machine-checkable, the QueryGap pattern):** a toy family of grid transforms parameterized by a hidden
  rule `r` from a finite DSL `D`; CHECK(`p`, train) = O(\|train\|·\|p\|) exact-match (a `Ledger` instance);
  FIND = identify `r` from train pairs via DSL queries. Target theorem `abstraction_find_ge` — any decider that
  outputs the correct `r` must make ≥\|D\| (or Ω(log\|D\|), per the oracle) queries, via the `QueryGap` adversary
  (an all-but-one-consistent input forces querying every rule). GAP = `check_lt_find` for the abstraction family.
- **Imported wall (named, NOT proved):** the unconditional P≠NP find/check separation; this is a RESTRICTED
  query-model statement (honest scope verbatim from `QueryGap`: a bounded-model lower bound, NOT P≠NP).
- **Attack:** `sundogcert/Sundogcert/AbstractionQueryGap.lean` reusing `QueryGap`'s `DTree`/adversary +
  `Certifies`/`Ledger` for the verifier; axiom audit + `#guard_msgs` gate. **Kill if** the verifier isn't
  provably cheap, the lower bound leaks the imported P≠NP wall, the toy DSL is a strawman (trivially un-findable
  for non-abstraction reasons), or `GAP_COLLAPSES_IN_MODEL` fires (a finite-order feature finds `r` cheaply ⇒ no
  gap, i.e. the family isn't genuinely abstraction-hard).
- **Why it survived / caps:** highest novelty — it would be the first *machine-checked find/check model of an
  abstraction task*, the genuine bridge object. Capped at 6 (not higher) because it is FORMALIZABLE-HARD (an
  honest toy DSL + a non-strawman lower bound is real work) and the scope is a restricted model, never P≠NP.
  Highest strength but NOT the first-strike (see recommendation).

#### FC-2 RESULT (run 2026-06-29): LANDED — machine-checked, axiom-clean, build-gated

Lean core: `sundogcert/Sundogcert/AbstractionQueryGap.lean` (namespace `Sundog.AbstractionQueryGap`),
wired into root + the `AxiomAudit.lean` `#guard_msgs` gate. Full `lake build` GREEN (8542 jobs), module
warning-clean. Independent sanity mirror: `sundog/scripts/findcheck_fc2_sanity.py` (all 5 facts reproduce).

- **The needle family (a genuine DSL family, built on FC-1).** Candidate `j : Fin D` is the *real*
  `AbstractionCert` program `recolor j (j+1)`; its private probe is `probe j = [[j]]`. PROVED:
  `eval_ruleOf_probe_self` (rule `j` changes its own probe `[[j]]↦[[j+1]]`) and `eval_ruleOf_probe_other`
  (every other rule fixes `probe j`) — each probe reveals exactly one candidate's consistency.
- **CHECK = the FC-1 verifier (the bridge).** `cbit_eq_verify` — the consistency bit of a candidate at its
  probe is *exactly* `AbstractionCert.Verify` on the single distinguishing training pair. FC-1's CHECK *is*
  the query.
- **The hard instances are real DSL behaviors (non-vacuity).** `cvec_id` — the identity behavior gives the
  all-false oracle; `cvec_rule` — rule `m` gives the one-hot at `m`. So the adversary's instances are
  genuinely realizable on this family: `GAP_COLLAPSES_IN_MODEL` does not fire.
- **The separation (both sides machine-checked, nothing imported).** `find_ge` — any prober deciding "some
  candidate is consistent" needs `≥ D` probes (`QueryGap.search_needs_n_queries`, the adversary). Headline
  `abstraction_check_lt_find` — for `D ≥ 2`, checking one candidate costs a single probe (depth 1) while any
  correct prober needs `≥ D`: **`check ≪ find` for a real abstraction family.**
- **Axiom audit (build-gated).** `cbit_eq_verify` = `[propext]`; `cvec_rule`/`find_ge`/`abstraction_check_lt_find`
  = `[propext, Classical.choice, Quot.sound]`. Pinned in `AxiomAudit.lean`; a regression fails `lake build`.
- **Kill conditions cleared.** (1) Verifier provably cheap (`cbit`/`checkTree` = one probe). (2) Lower bound
  does NOT leak P≠NP — it is `QueryGap`'s *restricted query-model* bound, reused honestly. (3) Not a strawman —
  the rules are real DSL programs and the needle structure is PROVED (`eval_ruleOf_probe_self/_other`); the
  hard instances are realized by genuine behaviors (`cvec_id`/`cvec_rule`). (4) `GAP_COLLAPSES_IN_MODEL` did
  not fire.
- **Wall named, not breached.** Unconditional query-model lower bound, NOT P≠NP; full-model program-search
  hardness stays imported. The honest content: a genuine DSL rule-family instantiates the find/check gap with
  no shortcut.

**Verdict: FC-2 LANDED (strength 6).** The deepest hook is real Lean — the first machine-checked find/check
*separation* for an abstraction family, bridged to FC-1's verifier. Remaining: FC-3 (reframe the ARC
receipts) → the ARC Branch-E gate (fires only after FC-2 + FC-3).

### Rank 2 — FC-1: the abstraction-task verifier as a new `Ledger` instance (strength 5.5)
**The cleanest near-term builder and the foundational object FC-2/FC-3 reference: formalize "∃ DSL program `p`
reproducing all train pairs" with a machine-checked cheap CHECK.** An 8th `Ledger`/`Certifies` instance.
- **Core (machine-checkable):** a `Ledger` instance `AbstractionCert` — `Verify(p, task) = ∀ (in,out)∈train,
  eval(p, in) = out`, with `checkCost ≤ |train|·evalCost(p)` (a straight-line `HasStraightLineCost`), on a small
  fixed DSL (identity/recolor/translate/tile over finite grids). Proves the CHECK-cheap half; FIND-hardness named.
- **Imported wall (named, NOT proved):** program-search hardness over the DSL (the FIND side) — imported, as in
  every `Ledger` instance.
- **Attack:** `sundogcert/Sundogcert/AbstractionCert.lean` on the `ShortestPathCert`/`MaxFlowMinCut` pattern;
  axiom audit + gate; a ~20-line numpy sanity check that the verifier accepts the planted program and rejects
  off-by-one. **Kill if** the check isn't O(\|train\|·\|p\|) forward, the eval needs an unbounded fixpoint (then
  it's not a straight-line cost), or it collapses to a trivial restatement of `HasStraightLineCost` with no
  abstraction content (the DSL-eval structure must be load-bearing).
- **Why it survived:** most buildable, de-risks the whole bridge (the verifier object FC-2 and FC-3 both need),
  reuses a proven pattern. Caps at 5.5 — it is a genuine new instance but incremental over the existing ledger;
  the find/check *content* lives in FC-2.

#### FC-1 RESULT (run 2026-06-29): LANDED — machine-checked, axiom-clean, build-gated

Lean core: `sundogcert/Sundogcert/AbstractionCert.lean` (namespace `Sundog.AbstractionCert`), wired into
root `Sundogcert.lean` + the build-enforced `AxiomAudit.lean` `#guard_msgs` gate. Full `lake build` GREEN
(8539 jobs). Independent sanity mirror: `sundog/scripts/findcheck_fc1_sanity.py` (all 3 facts reproduce).

- **What is PROVED (about the TOY, nothing about ARC).** A tiny total grid DSL `Prog`
  (`id`/`recolor`/`flipH`/`flipV`/`comp`) with a structural `eval` (no unbounded fixpoint), tasks =
  input→output grid pairs, and the CHECK `Verify p task = task.all (eval p · = ·)`:
  - `verify_iff` — the CHECK is exactly "consistent with every training pair," decidable.
  - `verify_planted` — the CHECK ACCEPTS the program that generated a task (completeness).
  - `train_underdetermines` (**headline**) — a task + two DISTINCT programs (`id`, `recolor 1 2`) both
    passing `Verify` yet disagreeing on a held-out grid: the training evidence does NOT pin the program.
    The find/check analog of `ParityNoSufficientStat.partial_not_sufficient` — cheap CHECK ≠ pinned answer.
  - `cost_le` — CHECK cost `|task|·(evalCost p + 1) + 1`, routed through `Certifies.Ledger`: the find/check
    ledger's program-synthesis instance (the 8th instance — syndrome/shortest-path/ReLU/max-flow/König/
    2-SAT/Pratt/**abstraction**).
- **Axiom audit (build-gated).** `verify_iff`/`verify_planted` = `[propext, Quot.sound]`;
  `train_underdetermines`/`cost_le` = *no axioms at all* (pure `decide`/`rfl`). Pinned in `AxiomAudit.lean`;
  a regression fails `lake build`.
- **Kill conditions cleared.** (1) Check is O(|task|·evalCost) forward (`cost_le`). (2) `eval` is structural,
  not an unbounded fixpoint. (3) NOT a trivial restatement of `HasStraightLineCost`: the DSL-eval structure
  is load-bearing (`eval_comp` composition; `train_underdetermines` is a real theorem about the toy, not the
  cost interface).
- **Walls named, not breached.** FIND (program search over the DSL) is imported; the GENERALIZATION wall
  (consistency-on-train ≠ correctness-on-test) is *witnessed* by `train_underdetermines`, named as why ARC is
  hard — no claim about cracking ARC. This is the FC-2 seed (the underdetermination a search must contend with).

**Verdict: FC-1 LANDED (strength 5.5).** The bridge object is real Lean. First-strike done; next is FC-2 (the
unconditional toy find/check separation) then FC-3 (the ARC reframe); the Branch-E gate fires only after both.

### Rank 3 — FC-3: reframe the ARC task-hardness receipts as find/check / order-meter measurements (strength 5)
**The ARC-facing payoff (no reopen): re-read the three filed ARC task-hardness receipts + the Branch-A–D floors
as find/check-gap and σ-order measurements, not representation-insufficiency verdicts.**
- **Core (empirical/analytical):** take the on-disk ARC receipts (`nn_output_transfer_v1`, `nn_delta_transfer_v1`,
  `candidate_combinator_v1`) and the Phase-3E binding receipts; show CHECK was always cheap (coverage ≥0.94 on
  grid-bearing arms) while the floors trace to FIND failure (learner-class candidate-pool structure, per the
  receipts' own `rep_sim_best` gap analysis). Map the failure to a search-cost / σ-order proxy on the Phase-0
  subset. Deliverable: a receipt `docs/findcheck/ARC_FLOORS_AS_FINDCHECK.md`.
- **Imported wall (named, NOT proved):** ARC's A–D + Phase-3E floors (the empirical record; this REREADS them,
  does not overturn them) and public-language constraints (the receipts are NOT a sufficiency-failure conclusion).
- **Attack:** read-only over the filed receipts + a small search-cost/order-meter instrument on the Phase-0
  register. **Kill if** the reframe is not falsifiable (any floor can be relabeled "find/check" post hoc), or the
  receipts' own analysis already attributes the floor to representation (not search) — then the reframe is wrong.
- **Why it survived:** uses assets already on disk, is the concrete ARC-facing output, and is the evidence the
  Branch-E gate needs. Caps at 5 because it is interpretive (re-reading existing receipts), not a new theorem.

#### FC-3 RESULT (run 2026-06-30): LANDED — reframe confirmed, Branch-E gate SUPERSEDED

Receipt: `docs/findcheck/ARC_FLOORS_AS_FINDCHECK.md` (read-only; changes no ARC verdict; public-language
preserved). Grounded entirely in filed ARC receipts (`PHASE3_5_REFLECTION.md`, the Branch-E program-search
spec + Amendment B).

- **The reframe holds, and is the receipts' OWN reading.** Learner floors: `grid_exact_any ≤ 0.010`,
  `rep_exact_slot1 ≤ 0.039` across all learner families. The receipts attribute this to the **candidate pool**:
  *"held-out outputs are nearly always outside the learner's candidate pool"* (coverage failure ≥0.99), and
  explicitly disclaim the representation reading (the `rep_sim_best` gap `0.139–0.155` reflects candidate-identity
  vs chance-collision, *"not how well each representation captures the underlying rule"*). So the floors are FIND
  gaps, not CHECK/representation gaps — the **kill does not fire** (receipts attribute to FIND, not representation).
- **Falsifiable prediction CONFIRMED (not post-hoc).** The find/check reading predicts search+verify should beat
  learning on covered tasks. **Branch E tested it and it held**: deterministic program search by train-pair
  consistency cleared the floor every learner floored at zero (`branch_e_capability_demonstrated`, 2/72 each gated
  lane). A Branch-E floor would have refuted the reframe.
- **Maps to the machine-checked ledger.** CHECK = `AbstractionCert.Verify` (FC-1) = Branch E's selection rule;
  `train_underdetermines` (FC-1) = the per-task-novel-rule under-determination; `abstraction_check_lt_find` (FC-2)
  = the CHECK ≪ FIND separation. FC-1/FC-2 are the Lean backbone for ARC's own search-beats-learning fact.
- **Honest residual (CORRECTED 2026-06-30):** an earlier draft guessed **library coverage**; **Branch E v2
  refuted it** (`branch_e_v2_capability_replicated` — grew the library, 0 new solves, *hurt* validation via
  top-2 crowding). The real residual is **selection under under-determination** (FC-1 `train_underdetermines`):
  the cheap CHECK admits many train-consistent-but-wrong programs, and growing the library only enlarges that
  ambiguous set. Toy confirmation `scripts/findcheck_underdetermination_crowding.py` (hit-rate collapses as the
  train-consistent count explodes; oracle ceiling stays 1.0 = selection, not coverage). Live frontier = a
  stronger-than-verifier ranker (Branch E3, built but compute-blocked/paused ~48h).
- **Branch-E gate: SUPERSEDED.** The gate pre-committed to *filing* a Branch-E spec after FC-2+FC-3. Inspection
  shows the ARC lane **already filed `PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md` with a binding receipt** that
  *confirms* the reframe. So FC-3 files **no** new ARC spec (it would change no verdict and duplicate filed work);
  the find/check contribution is the **unifying lens**, not a reopen. The genuinely-open ARC direction is
  **selection under under-determination** (Branch E3 learned ranker, built but compute-blocked/paused) — NOT
  library coverage (Branch E v2 refuted that). The ledger frames it but does not close it.

**Verdict: FC-3 LANDED (strength 5).** The find/check reframe of the ARC floors is confirmed by the ARC lane's
own Branch-E result and backed by the machine-checked FC-1/FC-2. The slate is complete; the Branch-E reopen
question is resolved (superseded), not pending.

### Rank 4 — FC-4: σ (order-meter) as the second hardness coordinate of abstraction tasks (strength 4.5)
**Synthesis tying the suffstat-order slate to abstraction: conjecture abstraction tasks are high/∞-σ in
pixel features but finite-σ in object features** — a quantitative read on why object-level priors help.
- **Core:** on a toy task family, measure σ (least-order sufficient statistic, the order-meter) under a pixel
  feature map vs an object/component feature map; predict σ_pixel ≫ σ_object for "abstraction" tasks.
- **Imported wall (named, NOT proved):** σ is a SCHEMA not a single scalar (the suffstat-order finding) — so the
  claim is per-filtration, not a universal ordering.
- **Attack:** reuse the order-meter harness; toy tasks only. **Kill if** σ_pixel ≈ σ_object (no separation), or
  the result is an artifact of the feature map rather than the task (the suffstat "σ is a schema" caveat fires).
- **Why it survived (barely):** a clean conjecture that fuses two lanes; lowest strength because it depends on
  FC-1/FC-2 for a task family and the σ-schema caveat blunts a clean headline.

#### FC-4 RESULT (run 2026-07-01): LANDED — the σ split CONFIRMED; the GEN-1 pre-test PASSES

Probe: `scripts/findcheck_fc4_sigma_split.py` (controlled toy family we own — no ARC dataset; pre-registered
gates in the header). Elevated from "optional synthesis" to **GEN-1's gate-0 pre-test** by
`GENERATOR_CLASS_SLATE.md`. Named imported anchor (measured, NOT proved): **Minsky–Papert perceptron order**
(connectivity-class predicates are not computable by bounded-order/diameter-limited feature families).

- **The family (confounds designed out):** 16×16 grids, 3–7 non-adjacent rectangles (1-cell gaps ⇒ every rect
  its own component), colors reused, **total colored area FIXED at 48** (kills the order-1 total-count statistic
  by construction). Latent = **area of the largest connected component** — the canonical ARC-style object rule,
  order-1 in the object filtration *by construction*; a global connectivity-class property in pixels.
- **SPLIT — PASSED (the headline):** OBJECT order-1 own-R² = **1.000** vs bounded-pixel ladder: hist **0.000**
  (the fixed-total design working), pairs 0.525, 2×2 patch 0.535, 3×3 patch 0.143 → best bounded pixel
  **0.535**, gap **0.465** (gate ≥0.30). The partial pixel R² is the pre-named honest leak (patches count
  corners ⇒ estimate fragmentation ⇒ max-area anticorrelates with n at fixed total). **Raw-grid MLP diagnostic
  = 0.000** — even the trained encoder fails the global property at this scale (the Minsky–Papert anchor,
  empirically). Trivial-FAIL = 0.000.
- **APPARATUS — PASSED:** pixel-easy latent (color-1 count) on the same grids: hist own-R² = 1.000 — the pixel
  ladder is live, not rigged to fail.
- **REVERSE — v1 flaw SELF-CAUGHT, v2 PASSED (the σ-schema/artifact kill):** v1 (variable m, one dot species)
  leaked — the frozen object vocabulary was *more informative than predicted*: `n_comps` minus the count of
  area≥4 rows reconstructs the dot count exactly (obj R²=0.994) → control redesigned, both runs reported. **v2**
  isolates the latent from the vocabulary (m∈{6,7} rects so no dot rows are visible in top-6; 0–6 nuisance
  color-6 dots confound `n_comps`): hist = **1.000**, obj = **0.509**, gap **0.491** (gate ≥0.20) — and the
  residual obj R² lands on the *predicted* confound value (~0.48 from the variance analysis), confirming the
  mechanism is understood. The pixel filtration wins on its home turf ⇒ **σ is PER-FILTRATION (a schema), the
  object map is not universally richer** — the kill condition ("artifact of the feature map") does not fire.
- **Honest scope:** a designed toy family (like every Shadow-lane probe); the claim is the split's *existence
  and direction* for abstraction-style rules, per-filtration, not a universal ordering and not an ARC result.

**Verdict: FC-4 LANDED (strength 4.5 → delivered as GEN-1 gate-0).** The GEN-1 class bet — ARC-style rules are
low-σ in the object filtration, high-σ in bounded pixels — passes its cheapest falsification. Next GEN-1 gate
per the generator-class slate: freeze the perception vocabulary + rule grammar v0, then the ceiling probe.

## The ARC Branch-E gate (the reopen criterion — RESOLVED / SUPERSEDED 2026-06-30)
Reopening ARC was **NOT** assumed by this slate. It was gated:
> **IF** FC-2 lands a clean toy find/check separation (axiom-clean, non-strawman) **AND** FC-3 shows the ARC
> floors re-read consistently as find/check gaps (falsifiably, against the receipts' own analysis), **THEN** file
> an ARC **Branch E** pre-reg spec (the find/check-sufficiency lens, with its own arena gate, verdict discipline,
> and public-language constraints, per `PHASE3_5_REFLECTION.md`'s Branch-E requirement). Otherwise the lane stays
> a frozen sibling ledger and ARC stays paused. **No ARC spec is filed before the gate fires.**

**Resolution (FC-3 finding):** the gate's premise is **already met by the ARC lane itself**. Both conditions
hold (FC-2 separation landed; FC-3 reframe confirmed), but on inspection
`PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md` is **already filed with a binding receipt**
(`branch_e_capability_demonstrated`, 2026-05-29) whose result *confirms* the find/check reframe. So the honest
move is **NOT** to file a new ARC Branch-E spec (it would change no verdict and duplicate filed work); the gate
is **superseded**. The find/check contribution is the **unifying lens + Lean backbone** (FC-1 CHECK, FC-2
CHECK ≪ FIND) for an ARC result that already exists. The one genuinely-open ARC direction the lens surfaces is
**selection under under-determination** — a stronger-than-verifier ranker over the train-consistent set
(Branch E3, tooling built but the binding run compute-blocked/paused ~48h, operator decision, `docs/TODO.md`).
Branch E v2 already **refuted** the library-coverage reading (grew the library → 0 new solves, top-2 crowding).
ARC stays paused; no new ARC spec filed. The find/check ledger frames the open direction but does not close it.

## The kill record (the discipline is part of the deliverable)
- **FC-5 — "does Shadow determine/resist predict ARC solvability?" — KILLED (strength 2, daydream).** ARC is a
  SEARCH problem (find a program), not a lossy-shadow recovery problem; the determine/resist law is about
  recoverability of a latent from an ensemble shadow. No honest map (the surface tie is the word "determine").
  **KILLED, and the kill is a deliverable** — it pre-empts the over-reach of dragging the Shadow law into ARC.
- **"Reopen ARC now on the find/check lens" — KILLED (premature).** ARC was ground through A–D + Phase-3E; a
  reopen needs the toy result + the receipt reframe FIRST (the Branch-E gate). Reopening blind repeats the
  learner-sufficiency churn under a new label.
- **"Claim P≠NP / unconditional search-hardness via abstraction" — KILLED (the wall).** Every `Ledger` instance
  imports find-hardness; `QueryGap` is restricted-model by construction. Any candidate implying an unconditional
  super-poly separation is a complexity breakthrough, not Lean-reachable, dead on arrival.
- **"ARC verifier needs the full ARC-AGI-2 dataset" — KILLED (scope).** FC-1/FC-2 are toy/synthetic (families we
  own); FC-3 is read-only over already-filed receipts. No dataset access, no public-eval inspection (the ARC
  Phase-≤5 constraints stay intact).

## Recommendation
**First-strike FC-1** (the verifier `Ledger` instance — cleanest, de-risks the bridge, the object FC-2/FC-3
need), **then FC-2** (the toy find/check separation — the deepest, the real new content), **then FC-3** (the ARC
reframe — the Branch-E evidence). FC-4 is optional synthesis. The Branch-E gate fires only after FC-2 + FC-3.
**Under no circumstance present this as cracking ARC or separating complexity classes** — the walls are imported,
the kill record (FC-5, the blind-reopen kill, the P≠NP kill) is the proof of discipline.

## Honest scope & boundaries
- REFRAME + toy-first, frozen-as-portfolio, NOT public-eligible. No ARC reopen without the Branch-E gate; no
  dataset access; ARC public-language constraints preserved (the receipts characterise learner families, they are
  not a sufficiency-failure conclusion).
- All find-hardness is IMPORTED (P≠NP / ARC's own floors); `QueryGap`-style results are restricted-model lower
  bounds, never complexity-class separations. The lane contributes the cheap verifier, the toy model, and the
  reframing lens — not an unconditional separation.
- Attribution: the algo-approx find/check ledger (`QueryGap`, `Certifies`/`Ledger`, `ShortestPathCert`); the
  suffstat-order σ order-meter; the ARC lane receipts + `PHASE3_5_REFLECTION.md` (Branch A–E discipline); the
  P-vs-NP certificate cores (find/check sibling); arXiv:2606.26705 (the algorithmic-complexity rates).
