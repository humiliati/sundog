# Sundog vs. BoxSEL (the false-closure lane)

Working hook:

> A box embedding can report a razor-thin probability and be completely wrong about
> *why* it is thin. Is the interval narrow because the ontology **entails** it — or
> because the optimizer, the box, and the loss quietly hid the models that would have
> widened it?

Short version:

> A BoxSEL-style reasoner answers a query `q_w(D|C)` from a learned geometric embedding.
> When that answer is **narrow**, the narrowness can come from four very different places:
> the ontology logically concentrates it, the restart sampler only found a sliver of the
> feasible set, the box family cannot represent the wider models, or nonzero loss distorts
> the read. The lane builds an oracle that can tell these apart on small fragments, then
> asks the real question: **can a geometric reasoner detect when its apparent certainty is
> only missing-model certainty — without the oracle?** That detector is an **abstention
> rule** (accept / widen / abstain), which is why this is **flagship-tied R&D**, not
> generality R&D — it feeds Ask Sundog's claim-boundary machinery.

**Status:** Scaffold opened 2026-06-19; lit-pass Phase 0.5 **filled 2026-06-20**
([`docs/boxsel/BOXSEL_LITPASS_MEMO.md`](boxsel/BOXSEL_LITPASS_MEMO.md); anchor = Zhu, Potyka,
Xiong, Tran, Nayyeri, Kharlamov, Staab, accepted UAI 2026). Phase 1 produced a PMP
calculator + frozen test; Phase 2 has started with a role-free exact-oracle smoke slice
(`scripts/boxsel_exact_oracle.py` + `scripts/test_boxsel_exact_oracle.py`) but is **not
cleared**. No sampler, extremal optimizer, prereg, public page, `site-pages.json` entry, or
external packet exists. Two
lit-pass findings now bind this ledger: **(1) the PMP discrepancy is confirmed in the arXiv
v1/v2 TeX source** — two checks, not one (§7 Phase 1); **(2) the coherence gap is demoted to a
watch item** because the anchor paper's Theorem 3 is zero-loss *inference* soundness, which
points the other way (§4). **Phase 1 (PMP replication gate) cleared 2026-06-20** —
`scripts/boxsel_pmp.py` (+ `test_boxsel_pmp.py`, 22/22); verdict `PMP_BODY_CONFIRMED` +
`PMP_PREMISE_SHAPE_UNRESOLVED`, blast radius held pending the authors' eval code
([`boxsel/PHASE1_PMP_REPLICATION_NOTE.md`](boxsel/PHASE1_PMP_REPLICATION_NOTE.md)).

Attribution: Statistical EL (SEL) and the entailment-interval framing, box embeddings
(BoxSEL), and probabilistic EL / probabilistic modus-ponens bounds are the literature's
(see the lit-pass citation spine). The Sundog contribution is the **synthesis**: the
search/representation/loss **gap decomposition** (coherence held as a watch item, §4) and the
**trace-based false-closure detector**. Bounded-novelty and no-priority rules (below) bind
every claim.

This is not a claim to have improved KG embeddings, to have solved uncertainty
quantification, or to have found a bug that invalidates Evgeny's paper. It is a
measurement question about when geometric certainty is logically warranted.

## 1. Thesis — separate logical concentration from projection-induced concentration

A narrow interval *looks* like knowledge. But a box embedding minimizes a training loss
over a geometric family; "narrow" is the joint output of the ontology **and** the
approximation pipeline. The lane's claim is that these are separable and that the
separation is measurable:

- **Logical concentration** — the SEL ontology genuinely entails a narrow `[l*, u*]`.
  Real knowledge; the reasoner *should* be confident.
- **Projection-induced concentration** — the interval is narrow because optimizer
  sampling, box representational limits, or loss distortion **dropped the wider models**.
  False closure; the reasoner is confident about its own blind spot.

The same vocabulary the rest of the portfolio uses for shadows applies: a box embedding is
a **lossy projection** of the SEL model class, and false closure is the embedding reporting
certainty about a hidden variable (the missing models) it has actually lost. **But see §5 —
this is a *different* decomposition from the Shadow-Invertibility tower; do not conflate them.**

## 2. Claim Boundary

This roadmap does **not** claim:

- that BoxSEL / Evgeny's intervals are wrong (the replication gate is a *sanity check on
  ground truth*, not an attack);
- that box embeddings are a bad method, or that any other embedding family is better;
- that the toy SEL fragments transfer to web-scale ontologies or real KGs;
- that the detector is a calibration guarantee — it is a guarded heuristic until the
  preregistered falsifier (§7, Phase 7) clears on held-out ontologies;
- anything about Ask Sundog's behavior until the flagship bridge (§6) is built and tested,
  not just asserted.

## 3. Why this fits Sundog

**It has an exact oracle.** `I* = [l*, u*]` is computable by type-enumeration + LP on small
SEL fragments. That is rare in the portfolio — Riemann, Yang-Mills, and Navier-Stokes are
bounded-null *because* no ground truth exists to check the body against. Here the nesting is
provable **and** the target is computable, so the lane can mint **structural-receipt-class**
artifacts (the bounded-operating-envelope sibling in `SCIENTIFIC_CRITERIA.md`), not vibes.

**The open contribution is the detector, not the nesting.** The inclusions in §4 are mostly
provable bookkeeping (cheap to bank). The genuinely new, genuinely Sundog move is Phase 6:
**can embedding traces flag false closure when the oracle is unavailable?** That is a
structural-zero receipt for *certainty itself*, and it ports to the product.

**It is a measurement question on a designed substrate**, asked first on tiny ontologies
where the answer is checkable — the house pattern (cf. `SUNDOG_V_JEPA.md`).

## 4. The formal core

Assume the zero-loss sets are nonempty. For a query `q_w(D|C)`:

```
I*            = [ l*, u* ]                          exact SEL entailment interval,
                                                    inf/sup over ALL finite SEL models
I_box^n       = [ inf_w q_w(D|C), sup_w q_w(D|C) ]  over zero-loss n-dim box embeddings
I_sample^{n,N}= [ min_i q_{w_i}(D|C), max_i ... ]   over N ordinary optimizer runs
```

The nesting (load-bearing, **of unequal strength** — the lit-pass, Track D, settled which is
which):

```
I_sample^{n,N} ⊆ I_box^n ⊆ I*
```

- **`I_sample ⊆ I_box` — definitional.** Each zero-loss run is a feasible box embedding, so
  its query value lies between `inf_w` and `sup_w`. Nothing to prove.
- **`I_box ⊆ I*` — the anchor paper's Theorem 3 (zero-loss *inference* soundness), under its
  geometric-volume semantics.** A *banked result, not a gap to attack*: if `L_T(w)=0` and the
  ontology entails `(D|C)[l,u]`, then `q_w(D|C) ∈ [l,u]`, so every zero-loss embedding's value
  sits inside `I*`. The one genuine open question is **semantic alignment** — our exact
  micro-oracle uses *finite-counting* models while Theorem 3 is stated over *geometric-volume*
  semantics; Phase 2 must verify the two `I*`'s coincide on the micro-fragments before relying
  on the inclusion. That check, not a coherence violation, is what is open.

The banked gaps are **three** (the lit-pass demoted the fourth):

```
search gap        :  I_box^n \ I_sample^{n,N}     optimizer found a sliver of the feasible set
representation gap:  I* \ I_box^n                 the box family cannot reach the wider models
loss gap          :  estimates outside I* when L_T(w) > ε   nonzero-loss distortion
```

**Coherence gap — WATCH ITEM, not a banked gap.** The earlier draft promoted "`I_box ⊄ I*` at
zero loss" to the headline finding; the lit-pass corrects that. Theorem 3 points the other way,
so a zero-loss escape is reachable **only** through one of three *audits* producing a receipt:
(1) a finite-vs-volume semantics mismatch (Phase 2), (2) an implementation/training mismatch
(near-zero numerical loss that satisfies sampled but not intended constraints), or (3) a
PMP/evaluation mismatch (§7 Phase 1). Until then the lane must **not** say "BoxSEL is
incoherent" or "zero-loss can escape the true interval" (falsifier `BOXSEL-ZEROLOSS-OVERCLAIM`).

## 5. Relationship to the Shadow-Invertibility tower (naming caution)

"Shadow Gap" deliberately echoes the Shadow-Invertibility Law
(`docs/atlas/ATLAS_PHASE5_CROSS_SUBSTRATE.md`, the charFun determine/resist law). **They are
not the same decomposition and a reviewer must not be led to think so:**

- The tower asks whether a *lossy shadow* **determines** a discrete hidden variable and
  **resists** a continuous one (a statement about hidden-variable *kind*).
- BoxSEL asks whether an *approximation pipeline* drops models that would widen an interval
  (a statement about approximation *quality* against an exact oracle).

The shared intuition (a lossy projection can manufacture false certainty) is real and worth
naming; the formal objects are different. The ledger and any public copy must state this
relationship explicitly. The honest framing: BoxSEL is a **new substrate** for the
false-certainty intuition, with the unusual luxury of an exact ground truth.

## 6. Flagship tie — false closure → Ask Sundog abstention

The reason this is flagship-tied R&D and not frozen-as-portfolio generality:

- Phase 6's guarded decision rule — **accept / widen / abstain** — is an **abstention rule**.
- That is the same shape as the `/chat` lane's claim-boundary preservation under prompt
  pressure (`SUNDOG_V_CHAT.md`) and the **same-observation Bayesian floor**
  (`BAYESIAN_FLOOR_PROFILE_TEMPLATE.md`). A *falsely-closed narrow interval* is a floor
  violation expressed in interval form.
- So the bridge is concrete: BoxSEL's false-closure detector is a candidate trigger for Ask
  Sundog to **widen or abstain** instead of reporting unwarranted confidence. The lane earns
  its keep iff that trigger beats restart variance alone (the §7 success gate) **and** the
  product can consume it.

**Bridge is a hypothesis, not a deliverable, until built and tested.** Until then the lane
runs as kill-gated R&D against the build-gate in §7; the flagship tie is the *reason to run
it*, not a claim that it already helps the product.

## 7. Roadmap (prereg-disciplined)

Each empirical phase produces a **prereg/result pair** under `docs/boxsel/`, LOCKED before
running, with `P-*` predictions and `KILL-*` criteria (house model:
`docs/atlas/H3_POOLED_SHADOW_PREREG.md`). Falsifiers are locked at the *front* of each
phase, never appended after.

- **Phase 0.5 — Lit-pass (HARD PRECONDITION).** `docs/boxsel/BOXSEL_LITPASS_MEMO.md`. Settle
  SEL semantics, box-embedding mechanics, the PMP bound, closest neighbors (selective
  prediction / KG calibration), and the no-priority map. Gates every outward claim.

- **Phase 1 — Replication gate (PMP oracle).** **The discrepancy is confirmed in the arXiv
  v1/v2 TeX source** (`BOXSEL_LITPASS_MEMO.md` Track C); resolve its *blast radius* before any
  published interval is treated as ground truth. Two checks, not one:
  - **(a) Upper-slack mismatch.** Body Prop 2: `min(1, q1·q2 + 1 − q1)`; Appendix Algorithm 2:
    `min(1, q1·q2 + 1 − q2)` — at `q1=0.2, q2=0.8`, `0.96` vs `0.36`. The `1 − q1` (= `1 − l1`)
    form is correct (escape mass = the first link's slack); the `1 − q2` form fails the
    `q1=1 → q2` sanity check (vacuous `1`). Independently confirmed via Wagner's conditional-PMP
    form `ab ≤ P(H) ≤ ab + 1 − b`.
  - **(b) Premise-shape drift (v2).** Prop 2 needs the second premise over `C∧D` —
    `(Q2 | A∧Q1)[q2]`; v2 Algorithm 2 prints `(Q2 | A)[q2]`. Harmless **only** if construction
    forces `A ⊆ Q1` (or an equivalent deterministic relation); otherwise the chain requires
    `Q2 ⊥ Q1 | A` and the bound can be invalid.
  - **Deliverables:** minimal PMP calculator + unit tests (incl. `q1=1 → q2` and
    `q1=0.2,q2=0.8 → [0.16, 0.96]` — note this *is* the paper's exact toy interval, so the
    calculator and the toy example validate each other); a per-query premise-shape audit; gate
    note `docs/boxsel/PHASE1_PMP_REPLICATION_NOTE.md` carrying one verdict token
    (`PMP_BODY_CONFIRMED` / `…_NO_METRIC_BLAST_RADIUS` / `…_AFFECTS_METRICS` /
    `PMP_PREMISE_SHAPE_UNRESOLVED`).
  - **Gate:** reproduce the paper's illustrative `[0.43, 0.83]` (sampled) vs `[0.16, 0.96]`
    (exact) example; **do not frame the typo as an attack** before the blast radius is known
    (falsifier `BOXSEL-PMP-TYPO-AS-ATTACK`).
  - **DONE 2026-06-20 — gate cleared.** `scripts/boxsel_pmp.py` + `scripts/test_boxsel_pmp.py`
    (22/22). Verdict `PMP_BODY_CONFIRMED` (body form + toy `[0.16,0.96]` locked) +
    `PMP_PREMISE_SHAPE_UNRESOLVED` (a finite-model counterexample shows the `(Q2|A)` drift breaks
    soundness when `A ⊄ Q1`: computed upper `0.52` < true `0.80`; harmlessness needs the authors'
    construction). Blast-radius codes held — no public eval repo surfaced. Full note:
    [`boxsel/PHASE1_PMP_REPLICATION_NOTE.md`](boxsel/PHASE1_PMP_REPLICATION_NOTE.md).

- **Phase 2 — Exact micro-SEL oracle (`I*`).** Tiny role-free SEL fragments; exact bounds by
  type enumeration + LP — a *local testbed*, not a new reasoning method (Lutz/Schröder lineage;
  no scalability claim). Deliverables: ontology generator, exact interval solver, held-out
  corpus labeled by exact width. **Gate:** solver agrees with the Phase-1 hand-derived PMP
  cases and the anchor toy example. **Semantic-alignment check (load-bearing; replaces the old
  "coherence check"):** confirm the finite-counting oracle's `I*` coincides with the
  geometric-volume semantics Theorem 3 is stated over, before relying on `I_box ⊆ I*` (§4).
  - **STARTED 2026-06-20 (not cleared).** `scripts/boxsel_exact_oracle.py` (+test, 41/41) — an
    **exact rational** type-enumeration oracle (in-house two-phase simplex; numpy/scipy removed;
    `interval_exact()` returns `Fraction`s), a deterministic tiny corpus with source-containment
    checks, and a count↔volume scale-invariance alignment smoke. Banked: the as-printed Algorithm 2
    is **unsound on a finite model** (§7 Phase 1 update). A separated **single-box realizability
    probe** (`scripts/boxsel_single_box.py` +test 11/11) seeds the representation gap: axis-parallel
    **Helly-2** ⟹ the `{A,B}/{B,C}/{A,C}`-positive, `{A,B,C}`-zero Venn is an oracle-valid model no
    single box can realize. Notes: [`boxsel/PHASE2_EXACT_MICRO_SEL_ORACLE_START.md`](boxsel/PHASE2_EXACT_MICRO_SEL_ORACLE_START.md),
    [`boxsel/PHASE2_SINGLE_BOX_REALIZABILITY_PROBE.md`](boxsel/PHASE2_SINGLE_BOX_REALIZABILITY_PROBE.md).

  **STARTED 2026-06-20:** `scripts/boxsel_exact_oracle.py` implements the first role-free
  type-enumeration + LP oracle; `scripts/test_boxsel_exact_oracle.py` passes the Phase-1 PMP
  hand cases, the toy `[0.16, 0.96]`, the sharp interval-premise check, and an as-printed
  Algorithm-2 shipped counterexample. The same smoke now includes deterministic tiny-corpus
  generation plus finite-counting/type-volume semantic alignment over disjoint Boolean type
  cells; this is **not** yet a single-box realizability claim. Start note:
  [`boxsel/PHASE2_EXACT_MICRO_SEL_ORACLE_START.md`](boxsel/PHASE2_EXACT_MICRO_SEL_ORACLE_START.md).

- **Phase 3 — BoxSEL baseline sampler (`I_sample`).** Wrap box embeddings + random-restart
  training (dimension `n`, restarts `N`, loss tolerance `ε`); estimator
  `I_sample = [min_i q, max_i q]`; logs for loss, endpoint movement, constraint slack, seed
  variance. **Gate (prereg):** under low loss, sampled estimates lie inside `I*`; mark every
  escape as loss-induced, and route any *zero-loss* escape to the coherence watch-item audits
  (§4) rather than reporting it as a banked gap.
  - **RESTART SAMPLER BUILT 2026-06-21** (`scripts/boxsel_phase3_restart_sampler.py` +test 19/19):
    ordinary **loss-only** zero-loss restarts for the Helly-seed ontology, deliberately without
    query pressure. For `dim=2`, `N=128`, `seed=314159`, the observed interval is
    `I_sample=[0.5336525204919725,1.0]`, nested inside the exact Phase-4 box interval
    `I_box=[(9+√17)/32,1.0]≈[0.4100970508005519,1.0]` and the oracle interval `I*=[0,1]`.
    Lower search gap: `0.1235554696914206`. Logs include loss, query, atom volumes, pairwise
    overlaps, constraint slack, cumulative endpoint movement, and seed variance. This is a
    baseline sampler, not a general BoxSEL optimizer; its value is quantifying the observed
    `I_box \ I_sample` gap against the now-exact Phase-4 endpoint. Note:
    [`boxsel/PHASE3_RESTART_SAMPLER.md`](boxsel/PHASE3_RESTART_SAMPLER.md).

- **Phase 4 — Extremal query optimization (`I_box`).** Replace passive restarts with
  query-conditioned training: `min/max q_w(D|C) s.t. L_T(w) ≤ ε`. Approximates
  `I_box = [inf_w q, sup_w q]`. **Gate:** distinguish ordinary sampling failure (search gap)
  from box-representation failure (representation gap). **SEED STARTED 2026-06-20:**
  `scripts/boxsel_phase4_interval_gap.py` converts the Helly realizability obstruction into a
  measured one-dimensional query gap: for a tiny ontology with atom sizes `1/2` and pairwise
  co-occurrences `≥1/4`, the exact oracle gives `I*=[0,1]` for `P(C|A∧B)`, while
  one-dimensional single boxes give `I_box^1=[1/2,1]`. The same seed now has a certified 2-D
  rational witness with `q=513/1250 < 1/2`, inherited by every `n≥2` via full-axis extension;
  this proves the 1-D gap shrinks in higher dimensions but does **not** settle the exact
  higher-dimensional infimum. This is not yet the full extremal optimizer. Seed note:
  [`boxsel/PHASE4_HELLY_INTERVAL_GAP_SEED.md`](boxsel/PHASE4_HELLY_INTERVAL_GAP_SEED.md).
  - **TREND RESOLVED 2026-06-20** (`scripts/boxsel_inf_trend.py` +test 17/17): the box query
    **factorizes over axes**, `q = ∏_k q_k` (each `q_k = P(C_k|A_k∩B_k) ∈ (0,1]`), and a proven,
    exhaustively-grid-verified per-axis lemma `q_k ≥ P(C|A_k)P(C|B_k)` multiplies up (via the exact
    `∏_k P(C|A_k) = |A∩C|/|A| ≥ 1/2`) to **`q ≥ 1/4` for every `n`**. With the `513/1250` witness:
    **`1/4 ≤ inf I_box^n ≤ 513/1250` for `n ≥ 2`** — the representation gap **persists (≥ 1/4), it
    does NOT vanish** with dimension. (A naive random search over-estimates the infimum in higher
    dims — a live search-gap demo.) The *exact* infimum in `[1/4, 513/1250]` stays open. Note:
    [`boxsel/PHASE4B_INF_TREND_RESOLVED.md`](boxsel/PHASE4B_INF_TREND_RESOLVED.md).
  - **EXACT-INF OPTIMIZER 2026-06-20** (`scripts/boxsel_exact_inf_optimizer.py` +test 17/17): the
    infimum is **attained at `n=2`** (`≈0.41010`, with `|A∩C|=|B∩C|=1/4` both active) — `n≥3` do
    **not** improve, so it is a finite-dimensional minimum, not a vanishing limit; the `513/1250`
    witness is near-optimal but **not** exact (only `|A∩C|` active). **Headline = a severe search
    gap:** every from-scratch optimizer misses the optimum — random `0.437`, SLSQP `0.50`, DE
    `0.43`, exact grid `≥4/9` — only Nelder-Mead **seeded from the analytic witness** reaches
    `≈0.41010`. The exact algebraic value + a proof `n≥3 ⊁ n=2` stay open. Note:
    [`boxsel/PHASE4C_EXACT_INF_OPTIMIZER.md`](boxsel/PHASE4C_EXACT_INF_OPTIMIZER.md).
  - **EXACT VALUE — KKT solve 2026-06-20** (`scripts/boxsel_kkt_exact.py` +test 12/12): the n=2
    KKT candidate has a **closed form**, `q_KKT = (9+√17)/32 ≈ 0.4100971`. KKT with both
    `|A∩C|=|B∩C|=1/4` active forces `4x²−9x+4=0`, `x=(9−√17)/8`; then `|A∩B∩C|=1/8`, `|A∩B|=x/2`,
    so `q*=1/(4x)=(9+√17)/32` — **verified exactly in `ℚ(√17)`** (a tiny exact `Surd` field).
    Strictly below the old rational witness `513/1250`; certified achievable + numerically the
    best global candidate seen. Tightened sandwich: **`1/4 ≤ inf I_box^n ≤ (9+√17)/32`**; the matching
    *lower* bound is the one open thread. Note:
    [`boxsel/PHASE4D_KKT_EXACT_OPTIMUM.md`](boxsel/PHASE4D_KKT_EXACT_OPTIMUM.md).
  - **LOWER-BOUND CLOSURE START 2026-06-20** (`scripts/boxsel_phase4e_lower_bound.py` +test 17/17):
    the KKT value is now proved to be the exact minimum **inside the structured 2-D normal form**
    used by Phase 4D. The proof reduces feasibility to
    `2(1-x) ≤ z ≤ x/(2(1-x))`, giving the lower root `x*=(9−√17)/8`; for fixed `x`, `q(x,z)` is
    concave in `z`, so the minimum is at an active-constraint endpoint, where
    `q0(x)=(-2x²+5x−2)/(2x²)` and `q0(x*)=(9+√17)/32`. This closes the local perturbation loophole
    but **does not** close the global theorem; remaining obligations are a 2-D endpoint-order atlas
    and an `n≥3` compression/no-improvement proof. Note:
    [`boxsel/PHASE4E_LOWER_BOUND_CLOSURE_START.md`](boxsel/PHASE4E_LOWER_BOUND_CLOSURE_START.md).
  - **ENDPOINT-ORDER ATLAS START 2026-06-20** (`scripts/boxsel_phase4f_cell_atlas.py` +test 13/13):
    the first of those two obligations, attacked. **Certified enumeration:** a per-axis cell is an
    `(σ,τ)` left/right-endpoint ordering ⟹ `36` per-axis types → `12` orbits; `1296` n=2 pairs →
    **`123` orbits** under {A↔B, reflection, axis-swap}. **Exact elimination:** C "helps" (`q_k<1`)
    iff it sticks out of `A∩B` (largest-left or smallest-right); the **31** zero-help orbits force
    `q=1 > q_KKT`. That leaves **92 live** orbits (56 one-help + 36 two-help; `q_KKT` sits on a
    two-help boundary). A **symmetry-orbit-seeded** search over the live cells finds nothing below
    `q_KKT` (numerical backstop — the blind searches all missed it, so seeding is what makes this
    trustworthy). Global lower bound still the proven `1/4`; sandwich `1/4 ≤ inf I_box^n ≤ (9+√17)/32`
    **unchanged**. OPEN: exact per-cell minima for the 92 live orbits + the `n≥3` compression. Note:
    [`boxsel/PHASE4F_ENDPOINT_ORDER_ATLAS.md`](boxsel/PHASE4F_ENDPOINT_ORDER_ATLAS.md).
  - **MIN-PAIR REDUCTION 2026-06-20** (`scripts/boxsel_phase4g_minpair_reduction.py` +test 13/13):
    the 92-live frontier shrinks exactly. For three pairwise-overlapping intervals on one axis,
    `|A∩B∩C| = min(|A∩B|, |A∩C|, |B∩C|)`. Thus if every helping axis can be assigned to the same
    side `S∈{AC,BC}`, then `q = |S|/|A∩B| ≥ (1/4)/(1/2) = 1/2 > q_KKT`. This exact-closes **47**
    formerly-live same-side orbits (21 one-help + 26 two-help). With the 31 zero-help closures from
    4f: **78/123 n=2 orbits exact-closed**, **45 mixed-side n=2 orbits remain open** (35 one-help
    mixed + 10 two-help AC-vs-BC mixed). Global sandwich unchanged; `n≥3` compression still open.
    Note: [`boxsel/PHASE4G_MINPAIR_REDUCTION.md`](boxsel/PHASE4G_MINPAIR_REDUCTION.md).
  - **ONE-HELP CLOSURE 2026-06-20** (`scripts/boxsel_phase4h_one_help_closure.py` +test 12/12):
    the 35 one-help mixed-side orbits close exactly. If the only helping axis is `AC` and
    `q=|A_1∩C_1|/|A_1∩B_1|<1/2`, then the global `|A∩C|≥1/4` constraint would force
    `|A_2∩C_2|>1/(2|A_1∩B_1|)`, but `|A_2∩C_2|≤|A_2|=1/(2|A_1|)≤1/(2|A_1∩B_1|)`, contradiction
    (and symmetrically for `BC`). So every one-help orbit has `q≥1/2>q_KKT`. Now
    **113/123 n=2 orbits exact-closed**; only the **10 two-help AC-vs-BC mixed** n=2 orbits remain,
    plus `n≥3` compression. Note: [`boxsel/PHASE4H_ONE_HELP_CLOSURE.md`](boxsel/PHASE4H_ONE_HELP_CLOSURE.md).
  - **TWO-HELP MIXED CORE 2026-06-20** (`scripts/boxsel_phase4i_two_help_mixed_core.py` +test 13/13):
    the final 10 n=2 orbits reduce to one explicit envelope. WLOG axis 1 is `AC`-min and axis 2 is
    `BC`-min. With `r=|BC_1|/|AC_1|` and `t=4|BC_1||BC_2|`, the mixed geometry gives
    `1≤t≤r≤2` and a relaxation envelope
    `P(r,t)=max_y r|AB_1||AB_2|/t`; closing n=2 now means proving
    `P(r,t)≤P(2,1)=(9−√17)/8`. Endpoint identity is exact:
    `1/(4P(2,1))=(9+√17)/32=q_KKT`; the `t=1` edge is monotone by a `16r^3>0` certificate, and a
    rational-grid guard finds no envelope violation. Phase 4j closes the full two-parameter envelope;
    Phase 4k compresses `n≥3` to it. Note:
    [`boxsel/PHASE4I_TWO_HELP_MIXED_CORE.md`](boxsel/PHASE4I_TWO_HELP_MIXED_CORE.md).
  - **MIXED ENVELOPE CLOSURE 2026-06-20** (`scripts/boxsel_phase4j_mixed_envelope_closure.py`
    +test 22/22): the Phase-4i envelope maximum is proved exactly. In shifted variables
    `a=r−1`, `b=t−1`, either the comparison is automatic (`M≤0`) or the legal squared residual
    factors as `D−L²=(a+1)²F/(8(b+1)²)`. On the `M>0` branch, `M` is decreasing in `b`, `F` is
    concave, `F(a,0)≥0` whenever `M(a,0)>0`, and `Res_b(F,M)` has fixed nonzero sign on the live
    `M=0` endpoint. Therefore `P(r,t)≤P(2,1)` on `1≤t≤r≤2`, closing the final 10 mixed n=2
    orbits. **n=2 exact result:** `inf I_box^2=(9+√17)/32`. Phase 4k lifts this to all `n≥2`. Note:
    [`boxsel/PHASE4J_MIXED_ENVELOPE_CLOSURE.md`](boxsel/PHASE4J_MIXED_ENVELOPE_CLOSURE.md).
  - **DIMENSION COMPRESSION 2026-06-21** (`scripts/boxsel_phase4k_dimension_compression.py`
    +test 17/17): arbitrary `n≥3` embeddings compress to the Phase-4i/4j envelope. Partition axes
    into `AC`-help, `BC`-help, and neutral `AB`-min groups. Same-side cases give
    `q≥1/2>q_KKT`; mixed cases use the `A/B` symmetry to orient neutral slack with `rho≤1`, giving
    the same domain `s≥1/2`, `1≤t≤r≤2` and neutral-adjusted bounds no larger than the pure mixed
    envelope. Therefore `rP/t≤P_phase4i(r,t)≤(9−√17)/8`, so
    `q≥(9+√17)/32`. Full-axis padding attains the value in every `n≥2`. **Global exact result:**
    `inf I_box^n=(9+√17)/32` for every `n≥2` (with `n=1` still `1/2`). Note:
    [`boxsel/PHASE4K_DIMENSION_COMPRESSION.md`](boxsel/PHASE4K_DIMENSION_COMPRESSION.md).

- **Phase 5 — Shadow-gap taxonomy.** Classify every case into search / representation / loss
  (+ a **coherence** watch-item bucket, populated only by an audit receipt per §4); gap-size
  metrics; dimension/restart/loss sensitivity curves.

- **Phase 6 — Delineator signals → abstention rule (THE OPEN CONTRIBUTION; flagship).** Test
  whether observable traces predict false closure *without* the oracle: endpoint drift vs
  restarts, endpoint movement vs dimension, ordinary-vs-extremal disagreement, constraint-slack
  concentration, regularization sensitivity, loss/query-gradient conflict, low-loss-basin
  instability. **Deliverable:** a guarded rule — accept / widen / abstain.
  - **TRACE DETECTOR START 2026-06-21** (`scripts/boxsel_phase6_trace_detector.py` +test 15/15):
    first trace-only guard over the Phase-3 Helly sampler. Features are sampled endpoint movement,
    zero-loss status, constraint slack, seed variance, and dimension sensitivity; the decision does
    **not** read `I*` or the exact `I_box` endpoint. On the canonical `dim=2,N=128,seed=314159`
    trace, the guard flags `endpoint_drift`, `active_constraint_slack`, `seed_variance`, and
    `dimension_sensitivity`, then returns `abstain`. Oracle evaluation is separate and confirms the
    case was false-closed relative to `I_box` (lower search gap `0.1235554696914206`) but not
    accepted. Controls lock `accept` for a stable high-slack trace, `widen` for a one-flag active
    slack trace, and `abstain` for loss escape. This is a seed-trap receipt, not a held-out
    Phase-7 success claim. Note:
    [`boxsel/PHASE6_TRACE_DETECTOR_START.md`](boxsel/PHASE6_TRACE_DETECTOR_START.md).
  - **PHASE-6B SCHEMA/ADAPTER START 2026-06-21** (`scripts/boxsel_phase6b_trace_schema.py` +test 39/39):
    post-Phase-7 redesign scaffold. The failure class is now named
    `stable_low_loss_false_closure`: the v1 guard caught turbulent Helly-style false closure but
    accepted stable PMP-shaped false closure. The new schema separates endpoint observations,
    per-constraint observations, support masses, seed/dimension/optimizer comparisons, and
    query-pressure observations; adapters convert real Phase-3 Helly reports and seen Phase-7
    result rows into `GeneralTrace`, and a PMP query-pressure producer exposes ordinary-vs-pressure
    lower-endpoint movement for the stable failure class. Feature names remain oracle-free and no
    v2 detector or Phase-7b held-out protocol is claimed. Phase-7 rows are now seen diagnostics only. Note:
    [`boxsel/PHASE7_FAILURE_ANALYSIS_AND_V2_SPEC.md`](boxsel/PHASE7_FAILURE_ANALYSIS_AND_V2_SPEC.md).

- **Phase 7 — Preregistered falsifier (LOCK before running).** Held-out ontologies with low
  loss, stable endpoints, narrow restart disagreement, **but a substantially wider exact
  interval.** **Build-gate / kill-gate:** if the guard repeatedly *accepts* these false-closure
  traps, AND does not beat a restart-variance baseline by the pre-registered margin, the lane
  is shelved. This is the make-or-break, not a victory lap.
  - **PREREG LOCKED 2026-06-21** (`scripts/boxsel_phase7_prereg.py` +test 18/18):
    detector thresholds, trace-only feature list, held-out seeds, corpus families, baseline,
    predictions, and kill criteria are frozen before held-out runs. Planned corpus: 16 held-out
    cases, including 10 false-closure traps, 4 true-narrow controls, and 2 loss-escape controls.
    Primary kill metric is accepted false-closure rate; guard must also beat
    `restart_variance_only_v0` by 20 percentage points, accept at least 50% of true-narrow controls,
    and accept zero loss escapes. Results status is explicitly `NOT_RUN`. Note:
    [`boxsel/PHASE7_FALSE_CLOSURE_PREREG.md`](boxsel/PHASE7_FALSE_CLOSURE_PREREG.md).
  - **RUN FAILED 2026-06-21** (`scripts/boxsel_phase7_run.py` +test 24/24):
    the first preregistered detector run trips `KILL7-1` and `KILL7-2`. The guard catches the
    six Helly seed-variant false closures (`abstain`), accepts all four stable PMP-shaped false
    closures, accepts all four true-narrow controls, and abstains on both loss-escape controls.
    Metrics: accepted false-closure rate `4/10 = 0.40`; restart-variance baseline also `4/10`;
    baseline improvement `0.00`; true-narrow accept rate `1.00`; loss-escape accepts `0`.
    This is a bounded null for the Phase-6 start rule, not a Phase-7 pass. Phase 8 remains gated.
    Note: [`boxsel/PHASE7_FALSE_CLOSURE_RUN.md`](boxsel/PHASE7_FALSE_CLOSURE_RUN.md).
  - **PHASE-7B LOCKED, NOT RUN 2026-06-21** (`scripts/boxsel_phase7b_prereg.py` +test 30/30):
    preregistration is now `LOCKED_NOT_RUN`. It freezes the Phase-6b schema feature boundary,
    excludes all Phase-7 rows as seen diagnostics, reserves a fresh seed pool, names the
    stable-PMP pressure family, keeps `restart_variance_only_v0` as primary baseline, and freezes
    the v2 detector/thresholds (`scripts/boxsel_phase7b_v2_detector.py` +test 16/16). Rule:
    loss/constraint violation or pressure/optimizer movement -> abstain; low support or ordinary
    endpoint/seed/dimension movement -> widen; no warnings -> accept. Corpus generator and
    evaluator are built (`scripts/boxsel_phase7b_corpus.py`, `scripts/boxsel_phase7b_evaluator.py`;
    `test_boxsel_phase7b_corpus_evaluator.py`, 24/24). Results remain `NOT_RUN`; held-out run is
    `READY_NOT_RUN`. Notes:
    [`boxsel/PHASE7B_FALSE_CLOSURE_PREREG_START.md`](boxsel/PHASE7B_FALSE_CLOSURE_PREREG_START.md),
    [`boxsel/PHASE7B_CORPUS_EVALUATOR_START.md`](boxsel/PHASE7B_CORPUS_EVALUATOR_START.md),
    [`boxsel/PHASE7B_V2_FREEZE_LOCK.md`](boxsel/PHASE7B_V2_FREEZE_LOCK.md).
  - **PHASE-7B RUN PASSED 2026-06-21** (`scripts/boxsel_phase7b_run.py` +test 24/24):
    frozen v2 passed the locked false-closure gate. Detector accepted false closures:
    `0/16 = 0.00`; restart-variance baseline accepted `16/16 = 1.00`; baseline improvement `1.00`;
    true-narrow accept rate `9/9 = 1.00`; loss-escape accepts `0/3`; stable-PMP pressure warning
    rate `8/8 = 1.00` vs baseline `0/8 = 0.00`. No kill criteria triggered and all five
    preregistered predictions were supported. This unblocks Phase 8 as a toy micro-SEL workbench
    and detector-demonstration lane, **not** a real-KG/product claim. Result note:
    [`boxsel/PHASE7B_FALSE_CLOSURE_RUN.md`](boxsel/PHASE7B_FALSE_CLOSURE_RUN.md).
  - **PHASE-7C REVIEW PACKET READY 2026-06-21** (`scripts/boxsel_phase7c_review_packet.py`
    +test 25/25): the review-worthy ask is now mechanism-centered, not a broad benchmark claim.
    Primary claim: on the locked tiny role-free micro-SEL fragment, restart-variance-only
    detection is structurally blind to stable false closure because it observes only
    `seed_low_range`; the stable PMP traps and pressure-noop controls share `seed_low_range = 0`
    while pressure-response traces separate the labels. The packet still carries the Phase-7b
    `0/16` vs `16/16` result as evidence, preserves the failed Phase-7 history, hashes review
    artifacts, audits detector/oracle leakage, records held-out hygiene, and names seven reviewer
    breakpoints. Outcome remains `NOT_REVIEWED`; even a clean review would certify only the toy
    micro-SEL claim. Note:
    [`boxsel/PHASE7C_EXTERNAL_REVIEW_PACKET.md`](boxsel/PHASE7C_EXTERNAL_REVIEW_PACKET.md).
  - **PHASE-7D STABLE/VARIANCE MECHANISM 2026-06-21**
    (`scripts/boxsel_phase7d_stable_variance_mechanism.py` +test 24/24): the clarity axis is
    now explicit. `restart_variance_only_v0` observes only `seed_low_range`, so any false closure
    with `seed_low_range <= 0.02` is accepted by construction. The stable PMP pressure family
    instantiates that blind spot: `8/8` false-closed rows have `seed_low_range = 0`, are accepted
    by restart variance, and are separated by pressure/optimizer movement. The pressure-noop
    controls share the same variance observable (`seed_low_range = 0`) but have no pressure signal
    and are accepted by v2. The receipt records `24` trap/control equivalence pairs with identical
    variance-observable/baseline action and opposite labels. This is the mechanism claim; still
    toy micro-SEL only. Note:
    [`boxsel/PHASE7D_STABLE_VARIANCE_MECHANISM.md`](boxsel/PHASE7D_STABLE_VARIANCE_MECHANISM.md).
  - **PHASE-7E ORACLE-FREE RECOVERY 2026-06-21**
    (`scripts/boxsel_phase7e_oracle_free_recovery.py` +test 19/19): the recovery slice now emits
    a boundary number, not just an abstention. For the Helly-seed `n>=2` box fragment, an
    oracle-free active-set pressure trace with active `AC`/`BC` constraints recovers
    `4x^2 - 9x + 4 = 0`, chooses `x=(9-sqrt17)/8`, and returns
    `q=1/(4x)=(9+sqrt17)/32`. The trace contains geometry/active-set facts, not `I*`, oracle
    labels, exact `I_box`, or the Phase-4 theorem. Phase 4k is used only after recovery to
    validate the recovered endpoint exactly. Ordinary restart sampling and the earlier rational
    witness both sit above the recovered value. This is still a Helly-fragment recovery receipt,
    not a general optimizer or real-KG claim. Note:
    [`boxsel/PHASE7E_ORACLE_FREE_RECOVERY.md`](boxsel/PHASE7E_ORACLE_FREE_RECOVERY.md).

- **Phase 8 — Workbench (gated on a Phase-7 pass).** Three nested interval bars (sampled ⊆
  box-attainable ⊆ exact); controls for dimension, restarts, loss tolerance, query pressure,
  regularization. Purpose: make false closure a visible boundary event. No public surface
  until the promote-gate clears.
  - **STARTED 2026-06-21** (`boxsel.html`, `scripts/boxsel_phase8_workbench_data.py`):
    review-only, noindex workbench over the passed Phase-7b micro-corpus. Data is generated from
    `results/boxsel/phase7b_false_closure_run/manifest.json` into
    `public/data/boxsel-phase8-workbench.json`. The page shows exact/sample/box layers where the
    Phase-4 Helly closed form is available, and labels PMP/support middle layers as
    `Pressure trace` rather than `I_box`. Controls cover query pressure, Helly dimension, restart
    reveal, and loss tolerance. Boundary remains toy micro-SEL only; no public inbound path. Note:
    [`boxsel/PHASE8_WORKBENCH_START.md`](boxsel/PHASE8_WORKBENCH_START.md).

## 8. Promote-gate / tier

- **Phases 1–5:** toy tier — exact-oracle bookkeeping on designed fragments. No real-KG word.
- **Phase 6–7:** the R1 result — a measured detector that beats restart variance on held-out
  ontologies, or a clean bounded null.
- **Phase 8 / product integration / any "Ask Sundog abstains correctly" claim:** requires the
  flagship bridge built **and** tested, plus external review. R-vocabulary about real KGs is
  **forbidden** until a registered theory gate is defined.

## 9. Allowed / Forbidden language

**Allowed (on a Phase-6/7 pass):**
> On small SEL fragments with an exact entailment oracle, observable embedding traces flagged
> falsely-closed intervals substantially better than restart variance alone, supporting an
> accept/widen/abstain rule. The effect held on held-out ontologies designed as false-closure
> traps and passed the loss and semantic-alignment controls.

**Forbidden:**
- "We found a bug that breaks Evgeny's paper" (the gate is a sanity check on ground truth).
- "Zero-loss BoxSEL can escape the true interval" without a specific semantic/implementation
  mismatch receipt (Theorem 3 points the other way — §4 watch item).
- "BoxSEL is overconfident / unsafe" as a general claim.
- "This calibrates KG reasoning" / any guarantee language for the heuristic detector.
- any statement about Ask Sundog's real behavior drawn from the toy oracle before the bridge
  is built and tested.

## 10. Open question (aspiration, not claim)

If a reasoner can read its own false closure off its traces — without the oracle that defines
truth — then "I don't have enough models to be this sure" becomes a *measurable* internal
state, not a slogan. The exact-oracle fragments are the one setting where that capability can
be earned against ground truth before it is asked to guard a product. Tagged *Normative*; it
motivates the lane, it is not a result.

---

*Sundog Research Lab — SUNDOG_V_BOXSEL scaffold. Internal; flagship-tied kill-gated R&D.
Lit-pass is Phase 0.5 and gates every outward claim. No run or public surface until operator
lock.*
