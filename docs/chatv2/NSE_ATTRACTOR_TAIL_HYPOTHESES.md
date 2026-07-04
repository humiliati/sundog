# NSE Attractor-Tail Hypotheses Slate (∞-dim body-resistance analogue)

> 2026-07-02. Drafted by the sibling agent commissioned in
> [`SIBLING_HANDOFF_ATTRACTOR_TAIL_SLATE.md`](SIBLING_HANDOFF_ATTRACTOR_TAIL_SLATE.md).
> **Status: live slate with receipts.** AT-6 has run and its Lean shim is clear;
> AT-1 decision #5 is resolved by `AT1_HARNESS_SIGNOFF.md`; all other entries
> remain slate/spec candidates unless a receipt says otherwise. Nothing here is
> public. No-publish (docs/chatv2/ is in `DOCS_NO_PUBLISH`). Before any
> verdict-bearing run, the entry must be lifted into its own frozen spec (exact
> commands, frozen thresholds) per house rule — the gates below are the binding
> *shapes*, not yet the frozen cells.
>
> Inherits, in full: the determining-modes claim boundary
> ([`../proof/PDE_DETERMINING_MODES_POSTULATE1.md`](../proof/PDE_DETERMINING_MODES_POSTULATE1.md)
> — no Millennium claim, no new determining-modes theorem, no smaller mode
> count), the state-crossover keep-true box
> ([`PROMO_WEBDEV_HANDOFF_STATE_CROSSOVER.md`](PROMO_WEBDEV_HANDOFF_STATE_CROSSOVER.md)),
> and the six binding lessons of the handoff §3.

---

## 0. Verification trail (receipts read before drafting; prose corrections found)

Receipts checked directly: `results/proof/c1-paired-fiber-g200/`,
`.../c1-paired-fiber-g300/`, `.../c1-disc-g200/`, `.../c1-disc-g300/` manifests;
`results/chatv2/` mtimes; the C1 doc chain (separation statement, twin-state
certificate, fiber discipline docs); `SUNDOG_V_NAVIERSTOKES.md`;
`RESIST_SIDE_CONSTRUCTION_ROADMAP.md`; `SUFFICIENT_STAT_ORDER_SLATE.md` +
addenda; sundogcert `OrderRelative.lean` / `ShadowDecay*.lean` /
`SurfaceBag*.lean` theorem inventories.

Three receipts-over-prose findings, recorded so the next reader doesn't re-derive:

1. **Handoff §2 dimension garble.** The handoff says "G=200 d=32, G=300 d=18."
   Receipts: `signature_dimension = 18` (K=3) at **both** regimes; `32` is
   `grid_size`. Per-regime: G=200 (`lock_v5`) ε_K=0.0606, 693,795 unique witness
   pairs, D_witness=0.0367; G=300 (`lock_v7_g300`) ε_K=0.0664, 942,834 pairs,
   D_witness=0.0382. Both `TWIN_STATE_CERTIFIED` + `PAIRED_FIBER_CONSTANCY_POSITIVE`.
2. **An unpursued measured anomaly sits in the C1 receipts.** The 2026-05-31
   objective-overlap discriminator
   ([`../proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`](../proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md)
   §12) filed `PDE-C1-DISC-INCONCLUSIVE` (G=200) / `PDE-C1-DISC-REFUTE` (G=300,
   label `control_sufficient_even_where_unpredictable`) — and inside it:
   **palinstrophy at G=200 is perfectly Φ_K-predictable (R²=1.0000) yet
   control-INsufficient (a_mm=0.195, NEG-A)**, flipping to POSITIVE at G=300.
   The doc's own closing line calls it a "separate probe, not pursued here."
   This is the cheapest live foothold in the whole lane (AT-1).
3. **Candidate 6 (resist-side construction) RAN.** The handoff doesn't mention
   it: Passes 0–4 all ran 2026-06-28, win condition met
   ([`../RESIST_SIDE_CONSTRUCTION_ROADMAP.md`](../RESIST_SIDE_CONSTRUCTION_ROADMAP.md)).
   Two load-bearing imports for this slate: (a) Pass 4 refuted "high-Re ⇒
   natural resistance by energy" (2D energy-FVE stays ~0.99 across Re) and
   flagged the real natural candidate — **low-energy small-scale intermittency,
   control-relevance unknown, compute-gated** (AT-7's target); (b) the internal
   σ-slate H5 fence binds: the resist-construction's ρ-dial is **info-loss-typed
   resistance (fiber>1), not computational (fiber=1, σ=∞)** — no slate entry may
   present constructed or info-loss resistance as natural computational
   resistance on NSE.

## 1. The target, and three organizing findings

The target (handoff §1): does a dissipative PDE system — the NSE global
attractor or an honest finite proxy — carry a **decision-relevant functional
that no compact shadow can determine**, the way GPT-2's stack-top defeats every
count statistic? Two forms in scope: the **crossover / input-undecodability
form** (instantiated once, LLM side) and the **maintained-ledger / shadow form**
(never built).

Analysis of the inroads produced three structural findings the slate is built
around. These are the "fresh eyes" content; each is falsifiable and each has an
entry.

**(F1) The absolute resist pole is CLOSED on the attractor's asymptotic mode
filtration — by the very theorems the lane imports.** Define σ_modes(F) = least
K such that the full asymptotic P_K-observation history determines F on the
admitted attractor. Determining-modes theorems (Foias–Temam 1984, CFMT 1985,
the 2408.01064 synchronization framing) say a finite K\* determines the *entire*
asymptotic state — so σ_modes(F) ≤ K\* for **every** functional. There is no
σ_modes=∞ functional on the attractor at infinite observation window. This is
the PDE twin of the LLM-side lesson learned at six families' expense
("absolute input-undecodability at bank scale on natural data looks
unattainable") — here it is not an empirical trend but an imported theorem.
Consequences: (i) every entry below uses **relative/crossover-form or
growth-form gates**, never an absolute-undecodability gate; (ii) the honest
analogue of σ=∞ is **unbounded growth** of the decision functional's least
sufficient budget in a resource parameter (horizon τ, Grashof G, window T) —
exactly the machine-checked idiom of `suffStatOrder_eq` (σ=n grows unboundedly
with n) and OR-3's graded enforce-ladder (1 < Θ(M) < ∞). AT-2 is this finding
as a measurement; AT-5 is its formal shadow.

**(F2) The C1 cell already contains its own "ambiguity slice," measured and
unpursued.** The LLM-side positives all lived on slices where the surface
statistic goes ambiguous (H2's count-ambiguous positions). The PDE-side twin is
**the decision boundary layer**: palinstrophy at G=200 is value-determined
(R²=1.0) yet decision-insufficient (a_mm=0.195) — predictability ≠ decidability,
the inverse of "ambiguity ≠ unpredictability," and the same lesson: sufficiency
is decided at the *decision margin*, not on the bulk distribution. If the
anomaly is real (not a lookahead-max burst artifact), the same natural cell
carries a regime-2 objective (E_low) and a regime-3 objective (palinstrophy) at
the same shadow — a two-pole structure on one substrate, which is the H2
crossover's objective-level analogue. AT-1 types this.

**(F3) The two target forms unify: a time-average is the bag of a trajectory.**
The LLM crossover's surface statistic (w-gram counts) is the order-blind
functional of a token stream; the PDE side's canonical compact shadows —
time-averaged observables, spectra, structure functions, invariant-measure
statistics — are the order-blind functionals of a trajectory. The maintained
ledger (a nudging/data-assimilation estimator) is precisely the
**order-dependent carrier** built from the *same* observation stream. So the
"maintained-ledger form" and the "crossover form" are one experiment: registered
surface = order-blind window statistics of the observation stream; carrier = the
maintained ledger; slice = where the window statistic is regime/decision-
ambiguous. `SurfaceBag`/`SurfaceBagGraded` is already the formal core of this
shape; `ShadowDecay*` already proves which functional *types* survive averaging
(discrete/component survives, continuous/phase washes) — the underused Lean is
underused no longer: it supplies AT-4's surface taxonomy (AT-6) and AT-5's
pattern.

## 2. The slate

Seven entries, four ranks, cheapest-decisive-first within rank. Verdict-token
namespaces `AT1_…`–`AT7_…`. Compute reality (binds all): CPU-tier preferred;
GTX-1080 fp16 available (no entry below needs it except optionally AT-4's
learned-surrogate arm); **the H200 is not to be boarded on this lane's
account**; agent-launched background jobs die at session teardown — anything
multi-hour is owner-run. C1 harness runs are ~20–50 min per preset (owner-run
under the 10-minute rule).

---

### AT-1 — Type the palinstrophy anomaly: boundary-layer artifact or the cell's second pole (Rank A, cheapest)

**Claim.** On the frozen C1 cell at G=200, the measured
predictable-but-not-control-sufficient palinstrophy read (R²=1.0000,
a_mm=0.195) is EITHER a thin decision-boundary layer (action disagreement
concentrates within a registered margin band around E_max and decays
monotonically as the band is excised — the same benign boundary-layer class as
D_witness≈0.037) OR a genuine decision-level failure that persists at every
margin band — in which case the natural cell certifiably carries both a
regime-2 and a regime-3 objective at the same Φ_K shadow.

**Mechanism / bridge.** Crossover-form at the *objective* level; the
ambiguity-slice diagnosis transplanted (slice = the label-margin band); the
verb-split vocabulary (value MEASURABLE ≠ decision readable-off-the-shadow) —
note it in OR-3's registered terms only as *analogy*, no formal import claimed.

**Cheapest decisive probe.** Rung 1: re-run `lock_disc_g200` with per-sample
(M, Φ_K, action) emission (harness extension, additive; ~45 min, owner-run).
Rung 2 (CPU minutes): margin-band excision curve a_mm(band); burst-artifact
control = a registered *new* sibling objective (fixed-horizon palinstrophy
value or lookahead-mean instead of lookahead-max — a new registration, not a
rescue of the frozen one); regime replication read against the banked G=300
flip (NEG-A→POSITIVE). Rung 3 (only if rung 2 says persistent): twin-pair
composition — do the certified witness pairs disagree on the palinstrophy
action specifically (paired exhibit, mirrors D_witness machinery).

**Gates (shapes; freeze at pre-reg).**
- `AT1_BOUNDARY_LAYER_ARTIFACT`: a_mm(band) → below the POSITIVE line (≤0.005)
  once a band of ≤ registered width is excised, and the sibling objective is
  POSITIVE unbanded ⇒ the anomaly is the known measure-δ boundary layer; C1
  framing unchanged; record and stop.
- `AT1_TWO_POLE_CONFIRMED`: a_mm stays ≥ the NEG-A line (0.015) at every
  registered band AND the sibling objective replicates ⇒ the cell's second pole
  is real; this is the slate's first positive and AT-4's natural label family.
- `AT1_UNDERPOWERED`: damp_fraction drifts out of [0.20, 0.40] on any rung ⇒
  file underpowered, do not interpret (the YM lesson, inherited from the
  discriminator's own power gate).
- **Kill:** any post-hoc change to band grid, sibling objective, or gates after
  a read = `AT1_NEG_B`, voids the entry.

**Standing falsifiers answered.** The §3.6 vacuity negative does not bite here
(the anomaly is *already* a measured non-collapse: value-sufficiency and
control-sufficiency separate in the direction the vacuity gauge doesn't cover);
the falsifier this entry answers is its own artifact branch — the burst-unstable
lookahead-max flagged in the discriminator's caveats. If artifact: record
plainly, no rescue.

**Lean anchor.** None at PDE tier (DAYDREAM). The two-pole *statement* on a
finite proxy folds into AT-5's module if confirmed.

**Does not claim.** No ∞-dim NSE content; no new PDE result; a
`AT1_TWO_POLE_CONFIRMED` does not demote the C1 separation (E_low's regime-2
read is untouched — the discriminator already established control ≠
predictability without downgrade); palinstrophy physics (dissipation-rate
proxy) is not asserted to be a *safety-relevant* objective, only a registered
one.

> **Harness decision (2026-07-02): `AT1_HARNESS_SIGNED_OFF`.** Owner decision #5
> is resolved in `AT1_HARNESS_SIGNOFF.md`: the frozen C1 harness may be touched
> only for additive, schema-versioned AT-1 sample emission and post-processing;
> existing no-export presets must keep their old semantics. This authorizes the
> harness work needed to write the frozen AT-1 spec. It is not a verdict-bearing
> run.

> **Post-run status (2026-07-02): RUN — `AT1_INCONCLUSIVE_MIXED`, and BOTH named
> hypotheses refuted** (spec `AT1_PALINSTROPHY_BOUNDARY_LAYER_SPEC.md` frozen
> first; regression gate passed; owner-run lock + rung 2; receipt
> `AT1_BOUNDARY_LAYER_RECEIPT.md`). Reproduction exact (a_mm 0.1949 vs banked
> 0.195, R²=1.0). Not a thin boundary layer (a_mm ≥ 0.141 at every powered
> band); not a second pole (burst-robust sibling a_mm = 0.0002, POSITIVE).
> **Mechanism found: threshold-on-near-atom degeneracy** — the near-periodic
> G=200 cell puts ≈0.50 mass within ±10⁻⁶ of one recurring burst peak and the
> q=0.70 calibration lands e_max inside it; the action label splits the atom at
> the 6th decimal (ball-straddle = 1.000). Explains the banked G=300 flip
> (aperiodic → no atom). **F2's two-pole candidate is typed and closed as an
> instrument degeneracy; the cell is decision-clean for burst-robust
> palinstrophy. New registration check for AT-2/AT-4: threshold-atom clearance**
> (the power gate does not catch it — damp was 0.300 while the label was
> degenerate). Rung 3 does not trigger; AT-1 complete. Same physical root as
> AT-6's finding: G=200 near-periodicity.

---

### AT-2 — The growth law: the decision functional's determining window grows; the energy functional's doesn't (Rank A)

**Claim.** On the C1 truncation, the least control-sufficient mode count
K_min(J; τ, G) for the registered decision objective grows strictly with
decision horizon τ (and from G=200→300) while K_min for the matched
energy-value objective stays flat within ±1 — the honest finite analogue of
σ=∞ as **unbounded growth**, given F1's closed absolute pole.

**Mechanism / bridge.** σ-order schema, as a **new per-filtration order**
(σ_modes: budget = Galerkin observation window; `Resolves k t ↔ ord t ≤ k`
shape). Registration of σ_modes as a schema filtration is an owner decision
(§4); as of 2026-07-02 the registered count is 9 filtrations + σ_write as the
control-side twin — this would be a candidate 10th. The growth form is the
`suffStatOrder_eq` idiom (σ grows with the resource), which is the only σ=∞
statement the portfolio has ever machine-checked.

**Cheapest decisive probe.** K-sweep (K ∈ {1..6}) × τ-grid (τ ∈ {250, 500,
1000, 2000} steps) at G ∈ {200, 300} on the existing presets — the harness
already parameterizes K and τ; ~8–16 runs × ~30–45 min = an owner-run batch
(bg-death gotcha applies). Side deliverable owed anyway: the K\* upper bracket
(smallest K where the twin-state witness vanishes) that
`PDE_C1_SEPARATION_STATEMENT.md` §5 pre-registered as "the K-window companion,
registered next run" and never ran — the same sweep delivers it.

**Gates (shapes).**
- `AT2_GROWTH_CONFIRMED`: K_min(J_q; τ) increases by ≥2 across the τ-grid at
  fixed G (or by ≥1 at matched τ from G=200→300) while K_min(E_low-value)
  stays flat within ±1, with the power gate honored at every cell.
- `AT2_FLAT_NULL`: both flat ⇒ the attractor analogue is NOT growth-typed at
  this cell; record — this bounds the whole slate's ambition honestly and is
  itself informative (the "cheap check found the graded structure absent"
  negative).
- `AT2_COLLAPSE_VACUOUS` (**the §3.6 vacuity negative, verbatim**): if at every
  (K, τ) cell control sufficiency flips exactly where state
  reconstruction/synchronization flips under the standard data-assimilation
  gauge, the Postulate-1 reading is vacuous on this cell — record, do not
  rescue.
- **Kill:** post-hoc τ-grid or objective change = `AT2_NEG_B`.

**Standing falsifiers answered.** §3.6 vacuity is a first-class branch (above).
Binding lesson 1 (recoverability-not-dimension): the gate is on
control-sufficiency thresholds, never on rank/d of anything.

**Lean anchor.** The growth form on a finite symbolic shadow is FORMALIZABLE
(the `parityProblem_ord` / `suffStatOrder_eq` pattern, already in-tree —
a new `OrderRelative` instance whose ord grows with a horizon parameter).
The PDE statement itself: DAYDREAM (no mathlib attractor substrate — honest
tier per the U-4 lesson, and unlike U-4 there is no hidden discharge route
here; the dynamics are the wall).

**Does not claim.** No new determining-mode bound (K\* is *measured on the
truncation*, never asserted for NSE); K_min values are cell-specific; growth on
a 32×32 Galerkin cell licenses zero ∞-dim language.

---


> **Post-run status (2026-07-03): RUN — `NO_GATE_READ` at both regimes** (spec
> v1.1 frozen; harness sign-off stamped + regression gate passed; owner batch;
> receipt `AT2_GROWTH_LAW_RECEIPT.md`). Only 2/4 τ-cells included per regime —
> **the finding: lookahead-max is a short-horizon instrument on this cell**
> (G=200: AT-1's threshold-atom is horizon-dependent, mass 0→0.507→1.000 across
> τ; G=300: damp saturates 0.53→1.0 — the AT-1-registered atom-clearance check's
> first live catch). Reported non-gate reads: cross-regime K_min +1 at both
> matched τ (M2's direction); K_min(decision)=K_min(value)=K_state at every
> included cell (the §3.6 vacuity PATTERN, visible but not gate-fired — honest
> prior for any AT-2b is COLLAPSE_VACUOUS, not growth); §2.5 event sub-read
> `AT2_EVENT_FLAT` both regimes (no motion for the 3D parked lead). **K\*
> bracket (§5 debt): closed with a typed wall — K\* > 4, twin instrument
> DEFERRED_COVERAGE (zero evaluable neighborhoods) at K ≥ 5.** σ_modes
> registration: receipts now weigh against; deferred.

> **AT-2b (2026-07-03): RUN — `AT2B_COLLAPSE_VACUOUS` at BOTH regimes, full
> strength** (new registration `AT2B_GROWTH_LAW_SPEC.md`, mean-form primary,
> in-band τ-grid {100..750}; zero new simulation — post-processing on the banked
> exports; receipt `AT2B_GROWTH_LAW_RECEIPT.md`). 10/10 cells included;
> K_min(decision)=K_min(value)=K_state at EVERY cell (1's at G=200, 2's at
> G=300); Δ = 0 everywhere; cross-regime +1 rides the state budget (not
> decision-specific — adjudicated under the registered precedence); event
> sub-read FLAT again. **The anchor doc's §3.6 vacuity negative is completed as
> a first-class measured result: on this cell, deciding and reconstructing cost
> the same budget. AT-3 is now the slate's sharpest question (can a maintained
> ledger decouple what the static read cannot); σ_modes registration: defer,
> definitively.**

### AT-3 — The maintained ledger: nudging at sub-determining budget carries the decision without carrying the state (Rank B; the never-built form)

**Claim.** There is an observation budget K_obs strictly below the measured
synchronization threshold K_sync of an AOT/nudging estimator on the C1 cell at
which the nudged ledger's decision readout matches the frozen J_q selector to
within a registered δ while its state-synchronization error stays above a
registered floor for the full run — and a decision-only twin ledger (same
budget, tuned only for J_q) matches the full ledger's decision read, certifying
that the carry is decision-typed, not residual state reconstruction.

**Mechanism / bridge.** The maintained-ledger form of the target, built
directly: nudging **is** an actively maintained shadow, and its literature
(SIAM 20M1323229, 20M136058X, 2408.01064) is the determine side. New content vs
C1: C1's read was static (signature balls at sample instants); this tests
whether the split **survives the trip to a dynamically maintained estimator** —
the exact object `navierstokes.html` calls "the shadow (a maintained ledger),"
designed and never built. Also inherits the chatv2 objective-contrast
discipline: the decision-only twin is the "control-trained" arm; leak/
recoverability gates, not rank gates (binding lesson 1).

**Cheapest decisive probe.** New script (`pde_c1_nudging_ledger.py`) wrapping
the existing 32×32 integrator with the standard AOT nudging term −μ(I_K(v) −
I_K(u)); sweep K_obs ∈ {1..6} + the K\* bracket from AT-2; measure per K:
state-sync error curve, decision accuracy vs the frozen selector, twin.
Controls with known answers: above K_sync both state and decision must succeed
(apparatus liveness); a scrambled-observation ledger must fail both (floor).
CPU; each K point ~20–40 min ⇒ owner-run sweep. No training, no GPU.

**Gates (shapes).**
- `AT3_LEDGER_SPLIT_CONFIRMED`: ∃ registered K_obs with sync error ≥ floor
  (state-insufficient, non-transient) AND decision acc ≥ scrambled-floor + δ
  AND ≥ chance + δ AND twin matches full ledger within δ_twin.
- `AT3_SHARP` / `AT3_GRADED`: the failure *shape* in K_obs (cliff vs smooth) —
  recorded either way; the handoff's "whether the failure is sharp (regime-2)
  or graded" question, answered as a measurement, not a preference.
- `AT3_VACUOUS_GAUGE_COLLAPSE` (**§3.6 verbatim**): decision readout and state
  sync cross at the same K_obs within sweep resolution, at every tested μ and
  horizon ⇒ control sufficiency collapses to state sufficiency under the
  standard data-assimilation gauge — record, do not rescue.
- `AT3_JOINT_INSUFFICIENT`: below K_sync the decision fails too ⇒ regime-3 at
  ledger level; record plainly (a legitimate "the ledger form doesn't separate
  here" negative).
- **Kill:** tuning μ per-verdict after a read (μ must be frozen or swept on a
  pre-registered grid) = `AT3_NEG_B`.

**Standing falsifiers answered.** §3.6 is the entry's own central branch — this
is the entry that finally *runs* the anchor doc's pre-registered negative
instead of citing it. Binding lesson 4: a nudging loop that fails to
synchronize even at K > K_sync files `AT3_DEAD_APPARATUS` (walled ≠ negative).

**Lean anchor.** DAYDREAM at PDE tier. The abstract split (an update-map
estimator that enforces a decision predicate without enforcing state equality)
might admit a finite `IsSufficient`-idiom cell, but no honest tier above
DAYDREAM until someone writes the toy — do not promise it.

**Does not claim.** No new synchronization threshold for NSE (K_sync is
measured on the truncation); no claim the ledger "understands" or "models" the
flow; μ-tuned nudging performance claims only relative to the registered
controls.

---

### AT-4 — The crossover transplant: order-blind window statistics collapse on the decision-ambiguous slice; the maintained ledger holds (Rank B; rides AT-3)

**Claim.** On slices of the C1 observation stream where the registered
order-blind surface statistic is decision-ambiguous (located by label-margin
bands and/or MZ-closure residual spikes — the closure R²(R|Φ_K)≈0.998/0.990 is
a *bulk* number; the slice is where its residual concentrates), the maintained
ledger (AT-3's estimator; full-state read as ceiling) reads the registered
decision functional above the best registered surface statistic of the same
stream by ≥ δ, while (liveness) a bag-determined control axis reads ≈1.0
on-slice and (order control) an order-shuffled window drops the ledger-side
read by ≥ δ_shuffle but not the surface read.

**Mechanism / bridge.** The crossover form, transplanted whole: surface = w-gram
/ windowed moments of the observation stream (order-blind by construction, the
F3 "bag of a trajectory"); carrier = maintained ledger; slice discipline +
liveness + order-shuffle = the V3-0b/A1 controls, inherited as design (binding
lessons 2, 3). Relative margins ONLY — F1 forbids absolute gates. AT-6's typing
supplies the pre-registered *prediction* of which labels are surface-readable
(component-type) vs surface-blocked (phase/order-type): the crossover gate runs
on a label the typing predicts surface-blocked, and the typing's
component-type label doubles as the liveness axis.

**Cheapest decisive probe.** Post-processing + one harness extension: emit the
K-band observation stream + per-sample labels from an AT-1/AT-3 run; fit the
registered surface family (frozen list: windowed moments, spectra, w-gram
count-vectors over discretized observables, w ∈ {1,2,4,8}) and the ledger
readout on identical splits. CPU, minutes-to-an-hour per config once the
streams exist; optional GTX-1080 arm only if a small learned surrogate is
registered as an additional carrier (owner decision; not needed for the gate).

**Gates (shapes).**
- `AT4_CROSSOVER_CONFIRMED`: per-slice ledger ≥ surface_max + δ AND ≥
  scrambled-ledger floor + δ, liveness ≈1.0, shuffle drop on carrier only, at
  ≥ the registered slice mass N_min.
- `AT4_SURFACE_SUFFICIENT`: the label is window-statistic-determined on-slice
  (predicted for component-type labels) — record; feeds AT-6's table, not a
  failure.
- `AT4_SLICE_THIN`: slice mass < N_min after balance — the V3-0b outcome,
  pre-planned: report slice mass + skew as the finding (natural-measure
  thinness is itself the phenomenon), do not force the gate.
- `AT4_DEAD_APPARATUS`: liveness axis fails on-slice ⇒ void, fix, re-run
  (order-meter discipline).
- **Kill:** more than one same-day re-registration of slice or surface family
  = `AT4_NEG_B` (no verdict-shopping).

**Standing falsifiers answered.** §3.6: if the ledger's on-slice advantage
exists only where state sync also succeeds (checked against AT-3's K_sync
table), the crossover is state-reconstruction in disguise — file under
`AT3_VACUOUS_GAUGE_COLLAPSE`'s shadow, record. Binding lesson 3 in full:
ambiguity ≠ unpredictability — residual distributional pinning on-slice is
expected; that is why the gate is relative.

**Lean anchor.** The surface half is AT-5 (the statement "no window order
determines the itinerary functional" is exactly `stackTop_resists_every_window`
transplanted). The carrier half stays empirical forever (keep-true box: the
Lean pair is about labels and statistics, never about the model/ledger).

**Does not claim.** Nothing about the ∞-dim attractor; no "the flow has a world
model"; a confirmed crossover says "the maintained ledger reads the decision
better than the registered surface statistic allows" — the locked claim
grammar from the state-crossover handoff, nothing more.

---

### AT-6 — charFun typing of compact shadows: time-averaging keeps regime-type functionals, washes phase/timing-type (Rank C, cheapest in rank; feeds AT-4)

*(Ordered before AT-5 within rank: it is an afternoon of post-processing.)*

**Claim.** Decision-relevant observables of the C1 cell split into the two
in-tree ShadowDecay classes under increasing time-averaging window T: phase/
timing-type functionals (burst timing, orbit phase, transition imminence) damp
toward chance with T following the charFun→0 pattern (`resistance_general` /
`cauchy_resists` class), while regime/component-type functionals (which
metastable branch, band-occupancy) survive (`twoPoint_shadow_survives` /
`determination_general` class) — so *which* decisions a time-averaged compact
shadow can carry is predicted by an existing machine-checked dichotomy, not
measured ad hoc.

**Mechanism / bridge.** The charFun axis (handoff §5 hook 4), made concrete:
the measure being averaged is the trajectory's empirical measure over the
window; discrete/lattice-valued functionals survive averaging, absolutely
continuous phase washes. This is the *only* entry whose deductive core is
already fully in-tree.

**Cheapest decisive probe.** Pure post-processing on existing/regenerated C1
sample streams (50k samples, both regimes): pick the registered observable
pairs, sweep T, fit decay curves, classify. CPU minutes. No new simulation if
AT-1's per-sample emission lands first.

**Gates (shapes).**
- `AT6_TYPING_CONFIRMED`: every registered phase-type row decays to floor,
  every component-type row survives, at matched T.
- `AT6_TYPING_BROKEN(row)`: any row crosses class — record *which*; a
  surviving phase-type row is the interesting outcome (a discrete invariant
  hiding in a nominally continuous observable — the AB/topological pattern),
  not a failure to rescue.
- **Kill:** class assignments are frozen before the first decay curve is read;
  re-typing after a read = `AT6_NEG_B`.

**Standing falsifiers answered.** §3.6 does not apply (no control claim); the
entry's own falsifier is the broken-typing branch. Feeds AT-4's registered
surface taxonomy; a broken typing *re-registers* AT-4's label family before
AT-4 runs (sequencing: AT-6 before AT-4).

**Lean anchor.** FORMALIZABLE, cheapest in the slate: a rotation-vs-two-point
toy where window-averaging is literally the in-tree averaging map —
`ShadowDecayLattice` already holds `twoPoint`, `absContMeasure`, and
`resist_separates_ac_from_lattice`; the new lemma is an interpretation shim,
not new mathematics. The bridge "physical time-average = this averaging map"
is a **named import**, stated in the docstring per METHOD.md discipline.

**Does not claim.** That time-averages are the only compact shadows; that the
typing is a resistance result (it types *shadows*, the resistance question
stays with AT-3/AT-4); no ergodicity theorem (mixing rates are imported
empirics on this cell).

> **Post-run status (2026-07-02): RUN — `AT6_TYPING_BROKEN(R3,R4,R5)` at both
> regimes** (`AT6_CHARFUN_TYPING_SPEC.md` frozen first; receipt
> `AT6_CHARFUN_TYPING_RECEIPT.md`). The falsifier branch fired informatively:
> R3 was mis-registered in scale (phase period ≈3×10⁴ steps ≫ T_max); the real
> finding is that **amplitude-washing ≠ decodability-washing on a noiseless
> deterministic cell** (G=200 cycle-timing stays 0.99-decodable under 5-cycle
> averaging; the predicted decay appears only at aperiodic G=300). AT-4's
> surface family is hereby re-registered as **SNR-aware** (declared observation-
> noise model or quantized readouts) with label frequencies inside the window
> grid, per the receipt's consequence block — the sequencing this entry existed
> to provide. **Lean status: `AT6_LEAN_CLEAR`; shim LANDED 2026-07-02:**
> `sundogcert/Sundogcert/AveragingDecodability.lean` (axiom-clean, gated, build
> 8577 GREEN) pins both halves — the amplitude typing packaged from in-tree
> (`averaging_types_shadows`) and the receipt's finding as a theorem
> (`amplitude_washes_readout_does_not`: the Debye–Waller factor is never zero at
> finite spread yet → 0, so nonzero attenuation preserves exact decodability and
> wash-out is a limit/noise-floor phenomenon).

---

### AT-5 — The symbolic attractor stack-top, machine-checked: an itinerary functional that resists every trajectory-window order (Rank C)

**Claim.** Over a finite symbolic proxy of regime dynamics (a two-plus-symbol
subshift with transition constraints, Kolmogorov-cell-inspired), band-occupancy
/ frequency functionals factor through window counts at order 1, while the
itinerary/branch functional (which regime path led here — the attractor's
"stack-top") is not a function of w-gram counts at ANY window order —
σ_traj(itinerary) = ∞ on the trajectory-window filtration, axiom-clean, gated
in `AxiomAudit`.

**Mechanism / bridge.** σ_surface transplanted from token streams to symbolic
trajectories; the `SurfaceBagGraded` construction pattern (context-swap witness
family + position involution). This is the handoff's "determining-modes
filtration Lean instance" hook, corrected by F1: the honest ∞-pole lives on
the **finite-window trajectory filtration** (where the LLM-side ∞ lives), not
on the asymptotic mode filtration (closed by K\*).

**Cheapest decisive probe.** Zero compute; Lean effort (days-tier). Module
sketch: alphabet = regime symbols; validity machine = allowed transitions;
`WindowSufficient w f` verbatim from `SurfaceBagGraded`; witness = two valid
trajectories with equal w-gram counts and different itinerary-functional
values, for every w — the OR-6 lesson pre-empted: naive padding fails
({1,2}-gram counts determine a string's last letter — degree argument), so the
construction must be the context-swap family with the swap separated by more
than any window, exactly the banked `P(P[P` vs `P[P(P` move. **The intended
empirical mirror is already certified:** C1's 693,795/942,834 twin pairs are
the "same shadow, different hidden state" witnesses; the module is their
symbolic-dynamics idealization.

**Gates.** `AT5_LEAN_LANDED` = full build green, axiom-clean
(`[propext, Quot.sound]` target, no `Classical.choice` if the SurfaceBag route
holds), `#guard_msgs` gated, fence docstring naming the imported wall. **Kill:**
if the chosen itinerary functional turns out count-determined at some w (the
degeneracy trap), the functional is re-chosen ONCE at spec time; failing twice
files `AT5_CONSTRUCTION_DEAD` and the entry dies as decoration (the same-lemma
fence discipline).

**Standing falsifiers answered.** §3.6 not applicable (no control claim). The
binding fence: the module proves a statement about **labels and statistics on
symbolic trajectories**; that Kolmogorov-cell regime dynamics realize the
subshift is empirical and stays with AT-1/AT-4 (keep-true box, clause 2).

**Lean tier.** FORMALIZABLE — honestly: finite lists/Finsets, an existing
in-tree pattern to mirror, no new mathlib substrate needed (the U-4 mis-tier
lesson cuts the other way here: this is the U-4 situation, not the H-A5 one).

**Does not claim.** No PDE theorem; no claim about NSE trajectories; not a new
filtration *registration* (that is an owner decision, §4) — the module stands
alone as another `SurfaceBag`-family instance regardless.

---

### AT-7 — The natural-resistance candidate, named and parked: control-relevant low-energy intermittency (Rank D; compute-gated, NOT runnable)

**Claim (conditional, not runnable at current posture).** The ε-carrying
dissipation-range degrees of freedom that outgrow any fixed shadow in count
while staying energy-recoverable (the Pass-4 divergence) are control-relevant
for a dissipation-range decision objective under stationarity-gated sampling —
i.e., the natural (non-constructed, fiber=1) resistance of NSE-class systems
lives exactly where the C2 numerical wall stands.

**Mechanism / bridge.** The recoverability axis (binding lesson 1) + the
resist-side Pass-4 receipt + the σ-slate H5 fiber×σ typing: this is the
portfolio's designated **computational-resistance corner** on a physics
substrate, distinct from the ρ-dial's info-loss resistance — the fence §0.3(b)
binds.

**Cheapest decisive probe (when unlocked).** The Sabra stationarity-gated
re-pose (C2 v1) with AT-4's crossover-form gates replacing the original
absolute gates, plus the AT-6 typing run first. Blocked by the adaptive
integrator (the stable window caps at ~1 burst time; eff-rank 1.7 was measured
on ~0.7 burst times — a *directional* marginal, honestly typed as
window-limited). **No compute is boarded by this slate.** Verdict typing
pre-committed per binding lesson 4: integrator-walled runs file
`AT7_NUMERICALLY_WALLED`, never a resistance negative.

**Gates (shapes, for the future spec).** `AT7_INTERMITTENCY_CONTROL_RELEVANT` /
`AT7_SLAVED_CONFIRMED` (burst shells determined by the low shadow at
stationarity ⇒ the third-time marginal becomes definitive — a clean, valuable
negative) / `AT7_NUMERICALLY_WALLED`.

**Standing falsifiers answered.** §3.6: at stationarity, if dissipation-range
decisions are readable exactly iff the state is reconstructable under the
nudging gauge, file the collapse. The entry exists so the slate says plainly
where the real thing lives and what it costs — not to smuggle a compute ask.

**Lean anchor.** None (DAYDREAM). **Does not claim.** That the wall has been
crossed, that Sabra resistance exists (three marginals say the prior is
against), or that any GPU/H200 spend is justified by this slate.

---

## 3. Do-not-say list (binding on every entry, receipt, and future promo)

Inherited and extended; violations void the offending receipt.

1. **No Millennium claim; no new determining-modes theorem; no smaller mode
   count; determining modes are not a Sundog invention.** (Anchor doc claim
   boundary, verbatim scope.)
2. **Keep-true box, transplanted:** (a) any positive is a *finite-proxy /
   cell-level* existence result — no ∞-dim NSE language beyond "hypothesized";
   (b) Lean modules are about labels and statistics on symbolic objects, never
   about the flow or the ledger — the empirical half is claims-relative-to-
   registered-matched-baselines only; (c) negatives are said plainly
   (`AT2_FLAT_NULL`, `AT3_JOINT_INSUFFICIENT`, `AT7_SLAVED_CONFIRMED` are
   publishable-quality outcomes, not embarrassments).
3. **No "the ledger/model understands / world-models the flow."** The licensed
   grammar: "reads the decision better than the registered surface statistic
   allows."
4. **Type every resistance:** info-loss (fiber>1) vs computational (fiber=1,
   order-∞) — and never present constructed or info-loss resistance as natural
   computational resistance on NSE (the internal H5 fence; binds AT-7
   especially).
5. **Schema, not scalar:** σ_modes / σ_traj are per-filtration orders; no
   cross-filtration comparison of σ values, no "the" σ of NSE.
6. **UNLEARNED / underpowered / numerically-walled ≠ resistance negative** —
   typed verdicts throughout; degenerate objectives file underpowered
   (discriminator power-gate discipline).
7. **This slate reopens nothing:** PROMOTE_GATE stays as-is (R2 NOT STARTED),
   `navierstokes.html` is not edited from here, chatv2 stays no-publish, and
   no entry inherits promotion from another lane's positive.

## 4. Owner decisions (open unless resolved below)

1. **Board anything at all, and in what order.** AT-6 has already run and is
   Lean-clear. Recommended next cut if boarded: AT-1 → AT-2 → AT-3 (AT-4 rides
   AT-3; AT-5 parallel, compute-free). Long runs stay owner-run.
2. **σ_modes / σ_traj registration** as schema filtrations in
   `SUFFICIENT_STAT_ORDER_SLATE.md` (owner-committed doc; candidate 10th axis).
3. **AT-5's module destination** — public sundogcert (needs the standard
   sensitivity scan; "attractor/itinerary" vocabulary check against the
   frozen-lane term list) vs held.
4. **The registered decision functional for AT-3/AT-4** — inherit frozen J_q,
   or register a burst/transition functional (new registration, not a rescue);
   and whether AT-1's sibling objective doubles as it.
5. **AT-1 scope — RESOLVED 2026-07-02.** `AT1_HARNESS_SIGNOFF.md` authorizes an
   additive AT-1 sample-emission/post-processing path in
   `pde_c1_kolmogorov_cell.py`, with existing no-export presets kept
   semantically stable. No AT-1 run is authorized until the frozen spec lands.
6. **AT-7 / compute posture** — stays parked; any future integrator spend is
   its own decision, not implied here.
7. **Whether F1 (the closed-pole observation) gets a line in the NSE ledger**
   (`SUNDOG_V_NAVIERSTOKES.md` is owner-run surface; this slate does not edit
   it).

## 5. Receipt / grep map for the inheritor

`Dev\sundog`: this file; `docs/chatv2/AT1_HARNESS_SIGNOFF.md`;
`docs/proof/PDE_C1_OBJECTIVE_OVERLAP_DISCRIMINATOR.md`
§12 (AT-1's foothold); `docs/proof/PDE_C1_SEPARATION_STATEMENT.md` §5 (the
unrun K-window companion AT-2 absorbs); `docs/proof/PDE_C1_MZ_ENERGY_BUDGET.md`
(AT-4's slice locator); `docs/RESIST_SIDE_CONSTRUCTION_ROADMAP.md` Pass 4
(AT-7's target); `results/proof/c1-paired-fiber-g{200,300}/manifest.json`
(the witness-pair ground truth quoted in §0); `docs/chatv2/
R2_INTERSECTION_HYPOTHESES.md` (the crossover form + H2/H5 results this
transplants); `docs/SUFFICIENT_STAT_ORDER_SLATE.md` + addenda (filtration
count). `Dev\sundogcert`: `SurfaceBag.lean` / `SurfaceBagGraded.lean` (AT-5's
pattern), `ShadowDecay{General,Cauchy,Lattice}.lean` (AT-6's core),
`OrderRelative.lean` + `ParityNoSufficientStat.lean` (AT-2's growth idiom),
`AxiomAudit.lean` (the guard any AT-5/AT-6 module must join). **Do not touch
owner WIP** (`AnalyticGate`, `AbstractionCert`, uncommitted OR-lane wiring).

---

*Slate v1, 2026-07-02, with same-day receipts. AT-6 ran and is Lean-clear;
AT-1 harness-scope sign-off is resolved; remaining entries require owner
review and frozen specs before any verdict-bearing run.*
