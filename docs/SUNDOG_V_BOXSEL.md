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
Xiong, Tran, Nayyeri, Kharlamov, Staab, accepted UAI 2026). No oracle, sampler, extremal
optimizer, prereg, run, public page, `site-pages.json` entry, or external packet exists. Two
lit-pass findings now bind this ledger: **(1) the PMP discrepancy is confirmed in the arXiv
v1/v2 TeX source** — two checks, not one (§7 Phase 1); **(2) the coherence gap is demoted to a
watch item** because the anchor paper's Theorem 3 is zero-loss *inference* soundness, which
points the other way (§4).

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

- **Phase 2 — Exact micro-SEL oracle (`I*`).** Tiny role-free SEL fragments; exact bounds by
  type enumeration + LP — a *local testbed*, not a new reasoning method (Lutz/Schröder lineage;
  no scalability claim). Deliverables: ontology generator, exact interval solver, held-out
  corpus labeled by exact width. **Gate:** solver agrees with the Phase-1 hand-derived PMP
  cases and the anchor toy example. **Semantic-alignment check (load-bearing; replaces the old
  "coherence check"):** confirm the finite-counting oracle's `I*` coincides with the
  geometric-volume semantics Theorem 3 is stated over, before relying on `I_box ⊆ I*` (§4).

- **Phase 3 — BoxSEL baseline sampler (`I_sample`).** Wrap box embeddings + random-restart
  training (dimension `n`, restarts `N`, loss tolerance `ε`); estimator
  `I_sample = [min_i q, max_i q]`; logs for loss, endpoint movement, constraint slack, seed
  variance. **Gate (prereg):** under low loss, sampled estimates lie inside `I*`; mark every
  escape as loss-induced, and route any *zero-loss* escape to the coherence watch-item audits
  (§4) rather than reporting it as a banked gap.

- **Phase 4 — Extremal query optimization (`I_box`).** Replace passive restarts with
  query-conditioned training: `min/max q_w(D|C) s.t. L_T(w) ≤ ε`. Approximates
  `I_box = [inf_w q, sup_w q]`. **Gate:** distinguish ordinary sampling failure (search gap)
  from box-representation failure (representation gap).

- **Phase 5 — Shadow-gap taxonomy.** Classify every case into search / representation / loss
  (+ a **coherence** watch-item bucket, populated only by an audit receipt per §4); gap-size
  metrics; dimension/restart/loss sensitivity curves.

- **Phase 6 — Delineator signals → abstention rule (THE OPEN CONTRIBUTION; flagship).** Test
  whether observable traces predict false closure *without* the oracle: endpoint drift vs
  restarts, endpoint movement vs dimension, ordinary-vs-extremal disagreement, constraint-slack
  concentration, regularization sensitivity, loss/query-gradient conflict, low-loss-basin
  instability. **Deliverable:** a guarded rule — accept / widen / abstain.

- **Phase 7 — Preregistered falsifier (LOCK before running).** Held-out ontologies with low
  loss, stable endpoints, narrow restart disagreement, **but a substantially wider exact
  interval.** **Build-gate / kill-gate:** if the guard repeatedly *accepts* these false-closure
  traps, AND does not beat a restart-variance baseline by the pre-registered margin, the lane
  is shelved. This is the make-or-break, not a victory lap.

- **Phase 8 — Workbench (gated on a Phase-7 pass).** Three nested interval bars (sampled ⊆
  box-attainable ⊆ exact); controls for dimension, restarts, loss tolerance, query pressure,
  regularization. Purpose: make false closure a visible boundary event. No public surface
  until the promote-gate clears.

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
