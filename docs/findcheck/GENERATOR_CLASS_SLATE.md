# Generator-class slate — what replaces the frozen deterministic bank (2026-07-01)

> Generate→adversarially-vet→rank, in the house format (cf. `FIND_CHECK_SUFFICIENCY_SLATE.md`,
> `docs/algo-approx/ALGO_APPROX_CONJECTURE_SLATE_2.md`). **The seed:** the deterministic program-search arc is
> CLOSED at its generator ceiling (Branch E v1 capability → v2 more-primitives-adds-0 → E3 Amendment D:
> oracle ceiling = exactly the solved set, so **no selector can lift**; quarantine **89% `no_admitted_programs`**
> — for nine in ten gated instances the frozen bank contains not one train-consistent program). The binding wall
> is **generator-class expressivity**: the per-task-novel rule is not *expressible* in the fixed menu of global
> grid transforms at any budget. This slate vets what a genuinely different candidate-generator CLASS looks like.
> **Discipline (load-bearing): SPEC-ONLY — nothing here admits a run.** Executing any candidate requires its own
> pre-registered ARC binding spec (arena gate, verdict table, public-language constraints) under
> `docs/prereg/arc/`; ARC stays paused until then. ARC public language preserved throughout (capability
> characterization only — NOT solve/sufficiency/eval/Kaggle claims; Phase 6 gates public-eval). NOT
> public-eligible. Calendar fact, not a recommendation: ARC Prize 2026 Paper Track dates remain Nov 2/8, 2026.

## The three imported design lessons (from the E1→E3 receipts; every candidate below inherits them)

1. **CEILING-FIRST GATING (the E3 lesson, now mandatory).** The oracle candidate ceiling is computable cheaply,
   on non-gated lanes, *before* any selector/harness investment: generate candidates for validation-lane
   instances, measure the target-in-bank rate and the `no_admitted_programs` rate. E3 spent a full harness +
   ranker build to learn its ceiling was empty. **Every candidate class below carries a pre-registered CEILING
   PROBE as its FIRST gate** — kill or proceed on that number. A class whose probe ceiling does not materially
   exceed the frozen v2 bank's on the same lanes is dead before any solver is built.
2. **SELECTION RETURNS WITH EXPRESSIVITY.** FC-1 `train_underdetermines` (machine-checked) + the crowding toy:
   the richer the class, the more consistent-but-wrong candidates. A class that lifts the ceiling re-opens the
   selection problem E3 found empty — **the E3 ranker machinery is downstream infrastructure, not waste**; its
   spec re-attaches to whichever class first produces a nonzero new-task ceiling.
3. **THE CHECK IS INVARIANT.** The verifier (train-pair consistency; FC-1 `AbstractionCert.Verify`, cost
   `O(|task|·evalCost)`) is generator-independent. Only FIND's *generate* term changes class. The find/check
   ledger frame carries over untouched.

## Survivors (ranked)

### Rank 1 — GEN-1: object-centric DSL with constraint-solved parameters (strength 6)
**Replace the global-transform menu with a perception layer + typed rules over OBJECTS — the class the 89%
failure diagnoses, and the only candidate preserving every house discipline.**
- **The class:** parse grid → object set (connected components with frozen attribute vocabulary: size, color,
  bbox, shape-hash, symmetry flags, adjacency/containment relations) → programs = typed **select → transform →
  place** rules (`filter/sort/map` over attributes; move/recolor/copy/delete/draw per object; anchors and
  relations as targets) → render. Parameters with holes are **CEGIS-solved** against the train pairs
  (counterexample-guided), not enumerated — the v1 "fit-then-verify" mechanism generalized to structured
  templates. Deterministic, byte-reproducible, contamination-free, cheap-CHECK.
- **Why this class (evidence-anchored):** the two tasks the old bank solves are exactly the whole-grid-rule
  tasks; typical register tasks (move-object-to-anchor, recolor-the-odd-one-out, symmetry-completion, counting)
  are one short object-rule but an inexpressible pixel-composition. The FC-4 σ-conjecture states this
  precisely: abstraction tasks are high-σ in the pixel filtration, low-σ in the object filtration.
- **Cheap pre-test (before ANY DSL build):** run FC-4 — measure σ (order-meter) under pixel vs object feature
  maps on register tasks; predict σ_pixel ≫ σ_object. If no separation, the class bet is wrong at the cheapest
  possible price.
- **Gate 1 (the ceiling probe, pre-registered):** freeze the perception vocabulary + rule grammar v0; enumerate/
  solve candidates on **validation lanes only**; measure oracle target-in-bank rate + no-admitted rate vs the
  frozen v2 bank's on the same lanes. Proceed only on a material ceiling lift.
- **Imported walls (named):** ARC's per-task-novelty may defeat *any* fixed object vocabulary too (that is what
  the probe tests, honestly); full-model program-search hardness stays imported; selection re-opens on success
  (lesson 2).
- **Kill if:** the perception vocabulary cannot be frozen cleanly (object segmentation itself is underdetermined
  — connectivity/multicolor/background choices leak per-task tuning); `GEN1_CEILING_EMPTY` (probe ceiling ≈ v2
  bank's); or the CEGIS/enumeration budget explodes past a pre-registered cap without ceiling gain.
- **Why it survived / caps:** direct answer to the diagnosed wall; falsifiable at two cheap gates (FC-4, probe)
  before the expensive build; every receipt discipline intact. Capped at 6: the build is substantial (a real
  perception layer + grammar + solver), and the honest possibility that ARC defeats fixed object vocabularies
  too is live — the probe exists precisely because of it.

### Rank 2 — GEN-4: test-time LM proposer with a contamination fence (strength 5)
**The highest-expressivity class — an LLM proposes programs in a fixed executable DSL from the train pairs;
the CHECK stays ours — and the class where the receipt discipline must work hardest.**
- **The class:** proposal distribution = a language model conditioned on the train pairs (and nothing else —
  the no-target barrier applies verbatim); candidates = emitted programs in a frozen executable target language;
  admission = the same train-pair-consistency CHECK; ranking = consistency + pre-registered tie-breaks. This is
  the only class with world-prior expressivity plausibly matching ARC's per-task novelty (the direction the
  public ARC frontier took).
- **The TWO fences (pre-registered, or the class is inadmissible):**
  - **Contamination:** public ARC training tasks are in frontier pretraining corpora, so "capability on held-out
    public-training lanes" is confounded by memorization. The spec must (a) use local open-weight models with
    documented cutoffs, (b) name the memorization confound in every receipt, and (c) treat public-training
    results as upper bounds, with any clean claim deferred to genuinely post-cutoff or private data. No API
    frontier models for binding runs.
  - **Determinism:** greedy/temperature-0 decoding, pinned weights + seeds, logged token streams; if bitwise
    reproducibility still fails across runs, the spec must downgrade its claims to distributional with n-run
    receipts.
- **Gate 1 (ceiling probe):** same as GEN-1 — proposal ceiling on validation lanes vs the v2 bank, before any
  harness investment.
- **Kill if:** the contamination fence cannot be written tightly enough to make any claim meaningful;
  reproducibility collapses; or the probe ceiling is not material.
- **Why it survived / caps:** the only plausibly ARC-competitive class; capped at 5 because the two fences are
  each heavier than the solver itself, and a memorization-confounded positive is worth far less to this lab
  than a clean deterministic one.

### Rank 3 — GEN-3: learned library (abstraction discovery), PHASE-GATED behind GEN-1 (strength 4.5)
**Grow the language by DISCOVERED abstractions (DreamCoder-style compress-what-you-solve), not hand-added
menu items — different in kind from what v2 refuted.**
- **The class:** solve what the seed language can on the aux pool; compress recurring sub-programs into new
  named primitives; re-search with the enriched library; iterate. The library grows at the *class* level.
- **The named bootstrap wall (from our own receipts):** the aux pool's consistent-program rate under the frozen
  bank was tiny (most aux instances admit zero) — a library learner starves without a seed language that
  already solves something. **Therefore phase-gated: admissible only after GEN-1's probe shows a nonzero
  object-rule ceiling to compress.**
- **Kill if:** GEN-1 fails its gates (nothing to bootstrap from); or discovered abstractions are all
  register-specific (compression without generalization — measured by held-out ceiling, not compression ratio).

## The kill record (the discipline is part of the deliverable)

- **GEN-2 — "CEGIS templates over the EXISTING global-transform vocabulary" — KILLED AT VETTING.** The receipts
  already refute the premise that *search/parameters* are the binding constraint: only 8% of gated instances
  were `budget_exhausted` while **89% admitted no program at any enumerated parameterization**, and v1's
  fit-then-verify already solves per-family parameters. Smarter parameter-solving over the same vocabulary
  attacks the non-binding term. CEGIS survives only as the parameter mechanism *inside* GEN-1's new vocabulary.
- **GEN-5 — "per-task learned generator (test-time training)" — KILLED (for this program).** Crosses every
  audit line at once (learned primitives, per-task weights, seed-sensitive receipts), is the most
  compute-hungry, and its expressivity argument is dominated by GEN-4's with strictly worse auditability.
  Reopen only if GEN-1 and GEN-4 both die at their probes.
- **"More deterministic primitives / deeper composition" — KILLED (standing, by receipt).** Branch E v2:
  0 new solves, validation harmed via crowding.
- **"A better selector over the frozen bank" — KILLED (standing, by receipt).** E3 Amendment D: the oracle
  ceiling equals the solved set; no selector can lift. (Selectors return *after* a class lifts the ceiling —
  lesson 2 — but never before.)

## Recommendation
**GEN-1, staged as: FC-4 σ pre-test → perception/grammar v0 freeze → ceiling probe → (only on a material lift)
a full ARC Branch-F-style binding spec** with the E3 ranker spec re-attached downstream. GEN-3 phase-gated
behind it. GEN-4 only if its two fences are written first and the owner accepts the confound-laden claim class.
Nothing in this slate admits a run; each gate is its own owner decision. If GEN-1 dies at its probes, the
honest terminal state is: *the ARC program stands closed at the deterministic baseline, with the generator-class
wall characterized* — itself a complete, well-receipted story.

## Honest scope & boundaries
- SPEC-ONLY; frozen-as-portfolio; NOT public-eligible. No dataset access is needed until a probe spec is filed;
  probes run on validation lanes only (Phase-≤5 constraints intact; no public-eval grids before Phase 6).
- All hardness walls stay imported (program-search hardness, ARC per-task-novelty design); the slate contributes
  class definitions, gates, and kill conditions — not a solver claim.
- Attribution: the Branch E1/E2/E3 receipts (`PHASE3_BRANCH_E*_SPEC.md`, E3 Amendment D); the find/check ledger
  (FC-1/FC-2, `train_underdetermines`, the crowding toy); the suffstat-order σ meter (FC-4); DreamCoder
  (Ellis et al.) and the ARC DSL/synthesis lineage (Hodel `arc-dsl`, ARGA, program-sampling / test-time-compute
  approaches) as the named prior art for GEN-1/3/4.
