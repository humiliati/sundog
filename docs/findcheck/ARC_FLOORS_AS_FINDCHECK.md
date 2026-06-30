# FC-3 — the ARC floors re-read as find/check gaps (a reframe, not a new result)

> **2026-06-29. Read-only reframe; NOT public-eligible; frozen-as-portfolio.** This document re-reads the
> *already-filed* ARC receipts through the find/check ledger lens (FC-1 `AbstractionCert`, FC-2
> `AbstractionQueryGap`). It introduces **no new ARC run, changes no certificate or floor verdict**, and
> preserves the ARC lane's public-language constraints (the receipts characterize *learner/search families*;
> they are NOT an ARC solve, a Blackwell-sufficiency proof, or any public-evaluation/Kaggle claim). Every number
> below is quoted from a filed receipt; the contribution is the *vocabulary*, not the data.
> Slate hook **FC-3** (`FIND_CHECK_SUFFICIENCY_SLATE.md`). Sources: `docs/prereg/arc/PHASE3_5_REFLECTION.md`
> (the canonical characterization of the three task-hardness receipts) and
> `docs/prereg/arc/PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md` (Amendment B, `branch_e_capability_demonstrated`).

## The lens (definitions, from the find/check ledger)
For an abstraction task = input→output training pairs:
- **CHECK** a candidate program = does it reproduce *every* training pair? Cheap, decidable — exactly
  `AbstractionCert.Verify` (FC-1) and exactly Branch E's *sole selection rule* ("admit only programs that
  reproduce ALL conditioning train pairs").
- **FIND** a generalizing train-consistent program. This splits into **(a) library coverage** — is the right
  program even *in* the candidate pool? — and **(b) search** — given it is, locate it. FC-2
  (`abstraction_check_lt_find`) is the machine-checked `check ≪ find` for a needle family; the imported wall is
  full-model program-search hardness.

The claim of this note: **on the registered ARC subset the floors are FIND gaps, not CHECK / representation
gaps** — and this is the ARC receipts' *own* finding, here named in ledger vocabulary and tied to the
machine-checked FC-1/FC-2 backbone.

## The evidence (all quoted from filed ARC receipts)

**1. FIND-by-learning floors at zero.** Across the three deterministic-low-capacity learner receipts
(`nn_output_transfer_v1`, `nn_delta_transfer_v1`, `candidate_combinator_v1`) and Branches A/D + variants:
`grid_exact_rate_any_slot` **never exceeds 0.010**, `rep_exact_rate_slot1` **never exceeds 0.039**
(`PHASE3_5_REFLECTION.md` §"Exact-rate floor"). Every learner family scored at the floor.

**2. The floor is a candidate-pool (FIND) failure, in the receipts' own words.** Grid-bearing arms report
**coverage failure on ≥0.99** of LODO `k_ge_3` instances: *"The held-out outputs are nearly always outside the
learner's candidate pool, regardless of which deterministic primitive set we extend the pool with"* (§"Coverage
floor"). That is the FIND side by definition — the answer is not in the searched space.

**3. The receipts explicitly DISCLAIM the representation (CHECK) reading.** The decisive Pass-C metric
(`rep_sim_best`, with the `metadata_only − signature_palette` gap `0.139–0.155`) is, per the receipts,
*"computed inside a learner class whose candidate pool overwhelmingly does not contain the held-out output. The
metric gap therefore reflects how the candidate-identity rules … trade off against chance-collision …, **not**
how well each representation captures the underlying rule"* (§"What The Receipts Did Not Show"). So the kill
condition for this reframe — *"the receipts' own analysis already attributes the floor to representation"* —
**does not fire**; the receipts attribute it to the candidate pool (FIND), not representation (CHECK).

**4. The falsifiable prediction — CONFIRMED.** If the floors are FIND gaps and CHECK is cheap, then explicit
*search* over a fixed library, kept train-pair-consistent, should crack tasks that *learning* floored on. Branch
E **tested exactly this** and the prediction **held**: a deterministic typed program search over the registered
primitives, selecting by train-pair consistency only, cleared the established non-trivial floor —
`branch_e_capability_demonstrated`, **2 distinct held-out tasks on each gated lane** (`test_lodo` 2/72 ≈ 0.027,
`pttest` 2/72 ≈ 0.026), three of four winners depth-2 compositions (`extract_largest_component`,
`crop_bbox >> scale`) — *"searching a fixed library succeeds (modestly) where learning the transforms from
scratch did not"* (Amendment B). Had Branch E *also* floored at zero, this reframe would be false; it did not.

## The map to the machine-checked ledger (FC-1 / FC-2)
The ARC arc instantiates the find/check ledger, now with a Lean backbone:
- **CHECK is cheap and works** — `AbstractionCert.Verify` (FC-1, `cost_le` O(|task|·evalCost)) IS Branch E's
  train-pair-consistency selection rule and the receipts' coverage notion.
- **Train evidence does not pin the program** — `AbstractionCert.train_underdetermines` (FC-1) is the toy form of
  the receipts' *"ARC tasks are explicitly designed so that the registered rule is novel per task"* (the
  generalization / under-determination wall).
- **CHECK ≪ FIND** — `AbstractionQueryGap.abstraction_check_lt_find` (FC-2) is the machine-checked separation:
  verifying one candidate is one probe, while finding one is search-hard; Branch E's "search beats learning, but
  only modestly" is this gap, with the residual located below.

## The honest residual (CORRECTED 2026-06-30 — it is SELECTION, not coverage)
An earlier draft of this note guessed the dominant residual was **library coverage** (grow the generator and the
rate rises). **Branch E v2 tested exactly that and refuted it.** Admitting the deferred intricate mask families +
morphology cross-product + depth-3 composition (`PHASE3_BRANCH_E_V2_PROGRAM_SEARCH_SPEC.md`, Amendment B,
`branch_e_v2_capability_replicated`) solved **0 new held-out tasks** and *slightly hurt* validation (2→1 per
lane). Its own diagnosis: a larger train-consistent candidate set **crowds the correct program out of the top-2**
("top-2 crowding"). So the wall is **selection under under-determination**, not coverage: the cheap CHECK
(train-pair consistency) is satisfiable by *many* programs that disagree off-train (FC-1
`train_underdetermines`), and **growing the library only enlarges that ambiguous set** — strictly worsening the
crowding. This is the find/check ledger's own prediction; the coverage reading is dead.

**Toy confirmation** (`scripts/findcheck_underdetermination_crowding.py`): recolor programs over `C` colors; a
program is train-consistent iff it matches the truth on the colors *seen* in train (unseen colors are free). As
coverage falls, the train-consistent count explodes (`Cᶠʳᵉᵉ`: 1→6→36→216→1296→7776 at C=6) and a deterministic
top-2 selector's held-out hit-rate collapses (1.00→0.46→0.14→0.03→0.00) **while the oracle ceiling stays 1.000**
— the truth is always in the pool. Selection, not coverage, exactly as v2 found. (The under-determination itself
is machine-checked in FC-1 `train_underdetermines`.)

**The live frontier is therefore a stronger-than-verifier SELECTOR.** Branch E3
(`PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md`) targets precisely this — a learned ranker over the *frozen* v2
candidate set to beat top-2 crowding (with an `oracle_candidate_ceiling` control). Its tooling is **built** but
the binding run is **PAUSED / compute-blocked (~48 h candidate generation; operator decision; `docs/TODO.md`)**.
That is the genuinely-open ARC direction, owner-gated on compute — not a coverage question.

## Status of the ARC Branch-E gate (updated — partly superseded)
The slate pre-committed: *file an ARC Branch-E spec only after FC-2 lands a separation AND FC-3 reframes the
floors falsifiably.* On inspection the premise is **already met by the ARC lane itself**: a
`PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md` is filed **with a binding receipt** (`branch_e_capability_demonstrated`,
2026-05-29), and its result *confirms* the find/check reframe. So FC-3 does **not** file a new ARC spec (that
would change no verdict and duplicate filed work). Instead the find/check contribution is the **unifying lens**:
the machine-checked FC-1 (CHECK) + FC-2 (CHECK ≪ FIND) supply the theoretical backbone for the *already-observed*
ARC fact that search+verify beats learning. The remaining genuinely-open ARC direction is **selection under
under-determination** — a stronger-than-verifier ranker over the train-consistent set (Branch E3, built but
compute-blocked/paused), NOT library coverage (Branch E v2 refuted that). The find/check ledger frames it
(CHECK admits many; the gap to correctness is the selector's job) but does not close it.

## Falsifiability, kill record, walls
- **Falsifiable, not post-hoc:** the reframe made a prediction (search+verify > learning on covered tasks) that
  Branch E independently confirmed; a Branch-E floor would have refuted it.
- **Kill did not fire:** the receipts attribute the floor to the candidate pool (FIND), explicitly *not* to
  representation (CHECK) — so this is the receipts' own reading, sharpened, not an imposed relabel.
- **Imported walls (named, NOT touched):** ARC's seven full-grid-control floors + four Phase-3E certificate
  verdicts (this note changes none of them); full-model program-search hardness; the per-task-novel-rule design
  of ARC. Public-language constraints from the ARC specs are preserved in full: no solve, no sufficiency proof,
  no public-evaluation/Kaggle claim.

## Attribution
The ARC lane receipts (`PHASE3_5_REFLECTION.md`, `PHASE3_SUFFICIENCY_SPEC.md`, the Branch-E program-search spec +
Amendment B); the find/check ledger (`AbstractionCert`/`AbstractionQueryGap`/`QueryGap`/`Certifies`); the
suffstat-order σ meter (the separate FC-4 axis, not used here).
