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
  only modestly" is this gap with the library-coverage limit on top.

## The honest residual (the deepest point)
Branch E cleared the floor only **modestly** (~2.6–3.4%). The dominant remaining floor is **library coverage**,
not search difficulty and not representation: ARC tasks are *designed* so the per-task rule is "novel per task,
not a member of any frozen pre-listed DSL" (§"Why A Fourth … Would Likely Converge"); the Phase-0
`oracle_copy_floor` already shows the natural seed DSLs exhausted (`0/115`). So the FIND gap here is dominated by
the **expressivity of the candidate generator** (is the rule in the pool at all), with the FC-2 search-hardness
as a second-order term. This is still a FIND-side characterization — the CHECK / representation reading stays
disclaimed by the receipts — but it locates the live difficulty precisely: **grow a generator that covers more
per-task-novel rules**, the exact direction a Branch E v2 (richer primitives) would test.

## Status of the ARC Branch-E gate (updated — partly superseded)
The slate pre-committed: *file an ARC Branch-E spec only after FC-2 lands a separation AND FC-3 reframes the
floors falsifiably.* On inspection the premise is **already met by the ARC lane itself**: a
`PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md` is filed **with a binding receipt** (`branch_e_capability_demonstrated`,
2026-05-29), and its result *confirms* the find/check reframe. So FC-3 does **not** file a new ARC spec (that
would change no verdict and duplicate filed work). Instead the find/check contribution is the **unifying lens**:
the machine-checked FC-1 (CHECK) + FC-2 (CHECK ≪ FIND) supply the theoretical backbone for the *already-observed*
ARC fact that search+verify beats learning. The remaining genuinely-open ARC direction is the **library-coverage
/ generator-expressivity** limit (a Branch E v2 question), which the find/check ledger frames but does not close.

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
