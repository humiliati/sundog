# Sundog vs. Algorithmic Approximation Theory

> **Lane status: ONE MACHINE-CHECKED CORE LANDED + four analytical hooks closed
> (2026-06-26); one hook remains.** Five of six hooks closed: **H-A0** (PDF-verify);
> **H-A2** → a real axiom-clean Lean core `Sundogcert/CircuitNet.lean` (`CORE_EARNED`:
> exact tropical→ReLU compilation, the ε = 0 piecewise-linear case of arXiv:2606.26705
> Thm 3.2 / Cor 5.1; full `lake build` green, in the AxiomAudit gate); **H-A1**
> cost-certificate isomorphism (deductive → `UNIFIES_ON_EXACT_FRAGMENT`: one gate-count
> measure, coinciding exactly on the linear/PL fragment with the P-vs-NP `verifyCost`,
> bounded by direction/ε/lossiness); **H-A3** competence-dominance reread (deductive →
> `SUPPORT_NO_SIZE_SEPARATION`, no revival of pantheon-for-competence); **H-A4** Atlas
> short-program (deductive → `ATLAS_IS_A_SHORT_PROGRAM`: the halo atlas is bounded-depth,
> polylog-in-1/ε — the √x side of the paper's divide, with the Atlas's own
> multi-scattering failure boundary the Brownian side). Remaining: H-A5 (o-minimality as a
> shadow-tower floor). Spun off
> [arXiv:2606.26705](https://arxiv.org/abs/2606.26705) — *"Algorithmic Foundations of
> Deep Learning: Complexity-Theoretic Rates and a Characterization of Universal
> Approximation"* (Kratsios, Brugiapaglia, Kim, Cousins, Sáez de Ocáriz Borde). It
> registers the lane, carries the H-A2 deductive receipt, and names what the remaining
> hooks (H-A1/H-A4/H-A5) would have to show to earn a place on the
> [Cross-Substrate Generality Failure Map](CROSS_SUBSTRATE_NOTES.md).

> **PDF-verification note (H-A0 closed 2026-06-26):** theorem numbers below were
> checked against the local PDF `docs/2606.26705v1.pdf`. The original HTML extraction
> was broadly right on the labels, but the formal Corollary 5.1 / Proposition 5.2 /
> Proposition 5.4 resource statements needed correction; the table below is the
> corrected source of record for this lane.

Working hook:

> Network size should price the *computation*, not just the *roughness*. A function
> with a short program (√x, shortest paths) is cheap to a network even when a smooth
> sibling of equal regularity (a Brownian path) is not — because the network can
> emulate the program, and classical worst-case-over-a-smoothness-class rates can't
> see the program.

Short version:

> Classical approximation theory rates a target by its **regularity** and pays a
> curse-of-dimensionality tax set by the *worst* member of the smoothness class. This
> paper swaps the unit of account to **algorithmic complexity**: rate a target by the
> circuit that computes it over a fixed gate language, then compile that circuit into
> a network whose depth/width/parameters track the circuit's resources. Sundog's
> interest is *not* the approximation theorem itself — it's that this is the
> constructive **upper-bound** mirror of two lanes built on **lower-bound** /
> resistance objects ([P-vs-NP](SUNDOG_V_P_V_NP.md), the
> [Lean certificate cores](SUNDOG_V_CERTIFICATE_LEAN.md)), and that its **o-minimal
> definability** backbone is a candidate next floor of the
> [characteristic-function / shadow tower](CROSS_SUBSTRATE_NOTES.md#63-bridging-vocabulary-table-across-substrates).

---

## 1. What the paper actually proves (PDF-verified H-A0 pass)

| Tag | Statement (as extracted) | Sundog-relevant content |
|---|---|---|
| **Def 2.6** | Complexity classes by **gate language** `𝔾`, lifting dimension `d0`, **depth ≤ Δ**, and **width ≤ Υ**. Table 2 languages extend from `𝔾_alg` (const/add/mult) to `𝔾_t-alg` (+ abs, max), `𝔾_rt-alg` (+ powers/radicals), and `𝔾_Rat` (+ reciprocal). | An explicit, finite **op-count cost measure** on a function — the same shape as the P-vs-NP lane's op-count cost certificate (0.949 ≤ 1.0). |
| **Assumption 2.4** | A class is admissible if it is (i) **definable** in an o-minimal expansion of the real field (e.g. the **Pfaffian closure of `ℝ_{an,exp}`**), (ii) **parallelizable** (closed under concatenation), and — for the main quantitative result — (iii) **non-piecewise-linear**. The qualitative Thm 3.1 uses only (i) and (ii). | The o-minimal floor: ONE structure covers MLPs, ResNets, transformers (companion catalogue). The Lean-formalizable object. |
| **Thm 3.1** | Under Assumption 2.4(i)-(ii), a definable feedforward class is universal for uniformly continuous `f:[0,1]^d→ℝ` (equivalently continuous targets on the compact cube) **iff** its dictionary contains a **non-affine nonlinearity**. | A near-tight **iff** universality characterization — the cleanest result; sharper than the usual one-way sufficiency. |
| **Thm 3.2** | **Circuit-to-network compilation**: an ε-computing `𝔾`-circuit of depth Δ, width Υ, gate count `N`, and lifting dimension `d0` compiles to a network computing within `2ε`, with explicit gate-class-dependent depth/width/nonzero-parameter bounds. | The engine that turns every circuit upper bound into a network upper bound, with overheads determined by the gate language. |
| **Cor 5.1** | Source-target **shortest path** on a `k`-vertex graph: the formal binary min-plus Bellman-Ford-Moore circuit has `Δ_k = O(k log k)`, `N_k = O(k³)`, `Υ_k = O(k²)`; the compiled network has depth/nonzero parameters `O(k log k log(k log k/ε) log^{∘2}(k log k/ε))`, hence fixed-`k` size `Õ_k(log(1/ε))` — vs the generic Lipschitz scale `O(ε^{−Θ(k²)})`. The surrounding all-pairs DP discussion also cites tropical depth `O(k)` and gate count `O(k³)`. | Depth-as-computation: log-rate where black-box bounds are exponential in ambient dimension. The headline example. |
| **Prop 5.2** | **Power iteration** `A ↦ λ_max(A)` on positive-definite matrices with spectral-gap and initialization-overlap conditions: depth and nonzero parameters `O(log(1/ε)^2(1+log^{∘2}(1/ε)))`, width `O(1)`, constants depending on `d,δ,γ,Δ,ρ0`; implemented through reciprocal and square-root/radical subblocks. | Algorithm emulation with an explicit formal rate; not the coarser `O(ε^{-1})` intro-table reading. |
| **Prop 5.4** | **Newton-Raphson / radical** `z ↦ z^{1/ℓ}` on an annulus `[m,M]`: explicit `K,K_inv` iteration counts; for fixed `ℓ,m,M`, depth and nonzero parameters scale as `O(log(1/ε) log log(1/ε))` with ε-independent width. | Algorithm emulation with an explicit formal rate; not the coarser `O(ε^{-1})` intro-table reading. |

---

## 2. Why this is a fresh vector, not a reopen

The paper is the **constructive / upper-bound** twin of objects two existing lanes
already own on the **resistance / lower-bound** side. They share a complexity-measure
shape but point opposite directions, so this gets its own ledger:

- **[P-vs-NP lane](SUNDOG_V_P_V_NP.md)** — measures a *find-vs-check* gap and an
  **op-count cost certificate** (cheaper to verify than to find). This paper measures
  a *compute-vs-approximate* gap and an **op-count construction cost** (cheaper to
  emulate the program than to approximate the worst-case smoothness sibling). Same
  ledger arithmetic (gate count vs a budget), inverted question.
- **[Lean certificate cores](SUNDOG_V_CERTIFICATE_LEAN.md)** — machine-check a
  deductive core, name the imported wall. The `𝔾`-gate-language / o-minimal
  definability layer is a candidate **eighth core**: a checkable definability /
  compilation lemma whose imported wall is "a trained net realizes the compiled
  circuit."

---

## 3. Hypothesis slate (H-A0 + H-A1 + H-A2 + H-A3 + H-A4 closed; H-A5 NOT-YET-RUN)

Each hook is stated so it can come back **NULL**. None is promoted.

- **H-A0 — PDF verification gate [CLOSED 2026-06-26].** The actual PDF was read
  against §1. Labels were confirmed; the resource summaries for Cor 5.1, Prop 5.2,
  and Prop 5.4 were corrected to match the formal statements. Falsifier result:
  `HTML_EXTRACTION_TOO_COARSE_ON_FORMAL_BOUNDS`; §1 has been fixed before any
  downstream hook is allowed to run.

- **H-A1 — Cost-certificate isomorphism (→ [P-vs-NP](SUNDOG_V_P_V_NP.md)) — RAN 2026-06-26, verdict `UNIFIES_ON_EXACT_FRAGMENT` (typed-positive, bounded; the falsifier `COST_MEASURE_NONUNIFIABLE` did NOT fire). Deductive/reading hook, no new Lean.**
  *Claim tested:* the paper's gate-count *construction* cost (Def 2.6: a `𝔾`-circuit's
  depth/width/gate-count `N`) and the P-vs-NP lane's op-count *check* cost
  (`Sundogcert/CheckCost.lean` `verifyCost ≤ 2(m·n)+n+m+2`) are instances of one measure.

  **Result — a genuine common measure exists, so the falsifier fails; but the
  unification is bounded.**

  1. **The shared measure is real, not verbal.** Both are *the faithful structural
     op-count of a straight-line program over a fixed algebraic gate dictionary.* The
     certificate's `verifyCost` is dominated by `syndromeMulCost = ∑_{i,j} 1 = m·n`, the
     gate count of the **matrix–vector product** `y ↦ H·y` over GF(2). The paper's `N`
     counts gates of an ℝ-circuit over `𝔾_alg ⊂ 𝔾_t-alg ⊂ …`. The *affine map* `y ↦ H·y`
     is literally an affine gate in both dictionaries — its cost is the nonzero-entry
     count in either, GF(2) or ℝ. So the measures coincide **exactly on the linear /
     piecewise-linear (exact, ε = 0) fragment**, and **H-A2's `compile_eval` (ε = 0
     exactness of the tropical fragment) is the formal witness that this fragment needs
     no error term** — the two cost notions are the *same number* there.

  2. **The bound (why it is `_ON_EXACT_FRAGMENT`, not full).** Three honest limits keep
     it a shared *yardstick*, not a merged theorem:
     - **direction** — the certificate counts the *verifier* circuit, the paper counts the
       *constructor* circuit. That is not a contradiction; it is the **find-vs-check axis
       itself** — the P-vs-NP gap *is* `gateCount(verifier) ≪ gateCount(any solver)`, and
       the paper supplies *constructor* upper bounds. Same measure, two circuits.
     - **ε-dependence** — the paper's *analytic* gates (reciprocal/radical, Prop 5.2/5.4)
       carry a polylog-in-`1/ε` cost the exact GF(2) certificate has no analog for. The
       unification *locus is exactly the exact fragment*; it dissolves where ε enters.
     - **lossiness** — the certificate's load-bearing property (`H·e` drops the secret) is
       an *information* property, not a cost one; the paper has no lossiness analog. The
       lanes share the cost axis and nothing else.

  **Net + a named follow-up.** The earlier "shared toy = the radical √·" framing was
  wrong (the certificate is GF(2) linear algebra and never touches radicals); the *true*
  shared toy is the **affine / matrix–vector map**. Both Lean witnesses already live in
  the *same* repo — `CircuitNet` (gate-count-able `Trop`/`Net`) and `CheckCost`
  (`verifyCost`) — so a generic `StraightLineProgram.cost` of which both are instances is
  a concrete future Lean hook (gated by the same DAG/gate-count wall named in H-A2; not
  built now). No claim promoted; this records a real but bounded yardstick-sharing.

- **H-A2 — Definability core in Lean (→ [certificate cores](SUNDOG_V_CERTIFICATE_LEAN.md)) — RAN 2026-06-26, verdict `CORE_EARNED` (the falsifier `CORE_IMPORTS_THE_THEOREM` did NOT fire). Real Lean, axiom-clean, full build green.**
  *Claim tested:* a circuit→net compilation step is machine-checkable axiom-clean,
  paired with a named imported wall, *without* all the content sitting in imported
  o-minimality.

  **Result.** Landed `Sundogcert/CircuitNet.lean` in the public Lean repo
  ([github.com/humiliati/sundogcert](https://github.com/humiliati/sundogcert)): the
  **exact (ε = 0)** special case of Kratsios et al. Thm 3.2 for the **tropical /
  piecewise-linear** gate fragment — the gate set the APSP headline (Cor 5.1) runs on.
  Three headline theorems, all **axiom-clean** (`[propext, Classical.choice, Quot.sound]`,
  no `sorryAx`/`native_decide`; wired into the build-enforced `AxiomAudit` gate; full
  `lake build` = 3534 jobs green):
  - `compile_eval` — every tropical circuit (`var/const/+/scale/max`) compiles to a ReLU
    net computing the *same* function exactly, by structural induction; the only nonlinear
    case is the identity `max p q = q + relu(p−q)`.
  - `compile_depth_le` — the compiled net's **depth is linear** in circuit depth
    (`≤ 4·depth`), the resource Cor 5.1 bounds.
  - derived **min-plus / Bellman–Ford** gates (`min`, `neg`, `abs`, `relaxEdge`,
    `bellmanStep`) each proved exact, and `bellmanStep_compiles_exactly` (the APSP inner
    loop compiles exactly).

  **Why the falsifier did NOT fire:** o-minimality does **zero** work — every step is
  finite real algebra re-checked by the kernel. The core is genuinely the *compilation*,
  not a wrapper around the definability import. **Named wall (honest boundary):** (i) the
  *analytic* gates (real ×, reciprocal, radical) are not piecewise-linear → only
  *approximable* (ε > 0), which is where o-minimality/Newton error analysis lives —
  outside this exact core; (ii) **linear gate-COUNT needs a DAG** — the `max` identity
  shares an operand, so a *tree* blows up under nested min/max; only wire fan-out keeps it
  linear (depth is tree-stable, hence proved; gate-count is named as the next increment, a
  compiler-correctness proof with wire-index refinement); (iii) **trainability** (SGD
  finds the weights) is imported, as everywhere in this development.

  *Registration:* wired into `Sundogcert.lean` root + `AxiomAudit.lean`; README file-map
  row added; public-safety grep clean (no frozen-lane terms). The sundog-site
  `SUNDOG_V_CERTIFICATE_LEAN` "Nth pillar" bump + deploy is **owner-gated** (left for
  review, as prior pillars). NOTE: the sundogcert README/METHOD "seven demonstrations"
  headline is already stale (SortingCert, RSCertificate, Tauroctony, AgenticTrace,
  DiscreteHolonomy + now CircuitNet are all in-build but unlisted) — a unified recount is
  an owner editorial pass, deliberately not guessed here.

- **H-A3 — Competence-dominance reread (→ [Mesa / non-sovereignty](SUNDOG_V_MESA.md), [reframe](mesa/PANTHEON_DOMINANCE_LEMMA_AND_NONSOVEREIGNTY_REFRAME.md)) — RAN 2026-06-26, verdict `SUPPORT_NO_SIZE_SEPARATION` (confirms the retirement; no revival). Deductive/reading hook, no experiment, no Lean.**
  *Claim tested:* Thm 3.1's universality + the compilation bound (Thm 3.2) is a
  constructive restatement of the **Competence-Dominance Lemma**
  (`Π_council ⊆ Π_monolith` ⟹ `max_{Π_M} R ≥ max_{Π_C} R`), with no size gap a plural
  topology could exploit on the return axis.

  **Result — the falsifier `NO_SIZE_SEPARATION` fires as pre-registered, doubly:**

  1. **Translation re-derives the lemma, adds nothing on `R`.** Read the paper's
     vocabulary onto the lemma: a *realizable policy class* `Π` is a *representable
     function class*; *size* is *circuit complexity*; *universality* (Thm 3.1) is
     *representability is free given a non-affine gate*. The lemma's hypotheses (same
     inputs `X`, budget `≤ β`, cap/role-factorization as pure constraints) say the
     council's representable class is a **subset** — constraints only *remove*
     representable functions, so at matched budget the unconstrained monolith realizes
     a superset. The dominance conclusion survives translation **unchanged** (it is
     still the one-line superset argument). Thm 3.1 *adds* only that both classes share
     the same **dense closure** asymptotically, so the subset is strict only at finite
     `β`, via the cap. → re-derivation in a second vocabulary, **says nothing the lemma
     didn't on `R`.**

  2. **The one genuine tension — steelmanned and resolved.** The paper's headline is
     that *structured/compositional* targets have **small** circuits (Cor 5.1, APSP).
     Could role-factorization therefore be a size-win for a compositional `R`? **No, not
     at the optimum:** structured-circuit cheapness is a property of the **target**,
     available to *both* classes — the unconstrained monolith can adopt the very same
     factored circuit (it lies in `Π_M`). The constraint forbids the monolith from
     *leaving* the factored family; it cannot make the council strictly cheaper. This is
     exactly the reframe's **§1.2 learnability corollary** in circuit vocabulary: a
     structural prior can help a *learner find* the cheap circuit, never lower the
     *optimum*.

  3. **Second confirmation — the theory is blind to the lone loophole.** Per §4 of
     this ledger, the paper is an **existence/optimum** theory (it does not address SGD
     trainability). The *only* place plurality earns a competence positive is
     **learnability** (the H2.3 cap / §1.2). An existence theory is structurally
     incapable of seeing learnability → it cannot even reach, let alone revive, the one
     loophole. `NO_SIZE_SEPARATION` thus holds on the optimum **and** the lens can't
     touch the axis where the lone positive lives.

  **The sharpening (the useful output, not a revival):** by making representability
  *free* (universality) and pricing only *computation*, the approximation lens
  **independently relocates** the live question off the architecture/size axis and onto
  **objective selection** — the monolith *can* represent the council's
  hedged / fault-tolerant / corrigible policy, but `R`-optimization *won't select* it.
  That is precisely the axis the [§4 Non-Sovereignty Premium](mesa/PANTHEON_DOMINANCE_LEMMA_AND_NONSOVEREIGNTY_REFRAME.md)
  already names. A second discipline (circuit-complexity approximation theory) thus
  **agrees** the action is on the objective, not the topology — strengthening the
  reframe without adding return-axis evidence for plurality.

  **Honest boundary:** this is a deductive re-reading *in the paper's vocabulary*, not
  an experiment and not a Lean core. It does **not** independently re-prove
  `Π_C ⊆ Π_M` at fixed budget — that stays the Mesa lane's own fairness construction;
  it re-derives the *dominance* in circuit terms and contributes the universality-closure
  and existence-theory-blindness observations. No claim promoted; the Mesa retirement of
  "pantheon for competence" stands, now witnessed from a second formalism.

- **H-A4 — Short-description ↔ small-parameter forward model (→ [Atlas](SUNDOG_V_ATLAS.md)) — RAN 2026-06-26, verdict `ATLAS_IS_A_SHORT_PROGRAM` (typed-positive, bounded; the falsifier `ATLAS_NOT_A_CIRCUIT` did NOT fire). Deductive/reading hook, no new Lean.**
  *Claim tested:* the Atlas's "whole classified halo atlas from ~1 continuous parameter +
  fixed ice lattice" is a **short-program / bounded-depth** target in the paper's sense —
  the √x-vs-Brownian distinction made physical.

  **Result — grounded in the actual generator** (`scripts/atlas_model.py`,
  `atlas_bifurcation_set.py`, `atlas_caustic_map.py`). The forward generator is a
  **fixed composition of definable/analytic gates** — `sin/cos/arccos/arctan2`, `sqrt`,
  `abs` — over `~1` continuous parameter (`n ≈ 1.31`) + the *fixed* ice lattice
  (`c/a = 1.628`, zero free geometry) + a *finite* habit branch (~7 cases). The **only**
  iterative element in the whole generator is a single **bisection**
  (`while hi − lo > tol` in `atlas_caustic_map.py`, locating the A₃-cusp merge elevation),
  whose inner `is_merged` test is a *fixed finite-grid* closed-form evaluation. Bisection
  converges in `O(log(1/ε))` steps ⟹ the generator is **bounded depth**, gate count
  **polylog in 1/ε** — squarely the paper's regime. So the falsifier (which needs
  *unbounded* depth) fails.

  **The √x-vs-Brownian dichotomy maps cleanly, two ways:**
  - **√x side (short program):** the classified atlas itself — closed-form analytic
    forward model + one log-depth bisection. Confirmed.
  - **Brownian side (no short description):** the Atlas's *own named failure boundaries* —
    multi-scattering, turbulent/random caustic fields, the §0.2 angular smear — explicitly
    out-of-scope in the Phase-11 capstone. The paper's dichotomy *is* the Atlas's
    in-scope/out-of-scope boundary.
  - **Cost-profile resonance:** the atlas's single hardest feature (the caustic merge,
    needing the bisection) has cost `O(log 1/ε)` — the *same* log-in-1/ε profile as the
    paper's APSP headline (Cor 5.1). The cheap-physics target and the cheap-algorithm
    target share a rate.

  **Honest bounds.** (i) The generator uses **transcendental** gates (trig), *outside* the
  exact algebraic fragment H-A2 nailed — they live in the paper's broader **definable
  (`ℝ_an,exp`) + ε-approximate** regime, the same ε-boundary as H-A1/H-A2's analytic wall;
  so "short program" holds in the paper's *general* definable-architecture sense, not the
  exact tropical fragment. (ii) This is a **qualitative** correspondence (a "short
  description" match + a rate resonance), not a quantitative gate-count theorem — no Lean
  (like H-A1/H-A3). No claim promoted; it records that the Atlas sits firmly on the
  cheap-to-compute side of the paper's own divide.

- **H-A5 — Definability as the next shadow-tower floor (→ [charFun law](SUNDOG_V_CERTIFICATE_LEAN.md), shadow-invertibility).**
  *Claim to test:* "definable in an o-minimal structure" is the structural sibling of
  the charFun-spectrum determine/resist separator — both are *one structural condition
  on a function class* that the whole tower hangs from. *Falsifier*
  (`NO_SHARED_SEPARATOR`): o-minimality and the Riemann-Lebesgue charFun tail govern
  unrelated phenomena and don't compose into one tower.

---

## 4. What this lane is NOT

- **Not a learnability result.** Every theorem here is **existence / emulation** —
  it bounds the size of a net that *can* represent the target via an explicit
  construction. It says nothing about whether SGD *finds* those weights. Any Sundog
  hook that drifts into "so training is cheap" is out of bounds.
- **Not new cross-substrate generality evidence.** The landed H-A2 Lean core is a
  **method / deductive example** (like the non-certificate Lean cores — Halo, Faraday,
  ShadowDecay), not a body-resistance / generality result. It earns the lane a
  machine-checked artifact, but the Cross-Substrate Failure-Map entry stays LIT-NOTE: no
  hook has produced return-axis or generality evidence for the Sundog thesis.
- **Not a generality claim.** "Definable architectures including transformers" is the
  *paper's* breadth, imported; Sundog imports it, does not re-derive it.

---

## 5. Next admissible action

H-A0 (PDF verify), **H-A1** (cost-certificate isomorphism → `UNIFIES_ON_EXACT_FRAGMENT`),
**H-A2** (the Lean `CircuitNet` core, `CORE_EARNED`), **H-A3** (competence-dominance
reread → `SUPPORT_NO_SIZE_SEPARATION`), and **H-A4** (Atlas short-program →
`ATLAS_IS_A_SHORT_PROGRAM`) are closed. Remaining hook: **H-A5** (o-minimality as a
shadow-tower floor). Owner-gated follow-ups
now owed off H-A1 + H-A2: (a) the sundog-site `SUNDOG_V_CERTIFICATE_LEAN` "Nth pillar"
bump + deploy; (b) the **DAG/sharing** extension upgrading linear-*depth* to
linear-*gate-count* (a compiler-correctness proof with wire-index refinement — the named
wall of `CircuitNet`); (c) the generic `StraightLineProgram.cost` of which both
`CircuitNet` circuit size and `CheckCost.verifyCost` are instances (the H-A1 unification,
formalized — gated by (b)). No hook is scheduled; this file is the registration, not a
commitment.

> Cross-links: [`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md) ·
> [`SUNDOG_V_CERTIFICATE_LEAN.md`](SUNDOG_V_CERTIFICATE_LEAN.md) ·
> [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) ·
> [`mesa/PANTHEON_DOMINANCE_LEMMA_AND_NONSOVEREIGNTY_REFRAME.md`](mesa/PANTHEON_DOMINANCE_LEMMA_AND_NONSOVEREIGNTY_REFRAME.md) ·
> [`SUNDOG_V_ATLAS.md`](SUNDOG_V_ATLAS.md) ·
> [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md)
