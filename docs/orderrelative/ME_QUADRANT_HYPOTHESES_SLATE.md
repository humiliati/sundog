# Measurable ≠ Enforceable — the Quadrant Slate (spec)

*Fresh-hypothesis slate mined from the ORDERRELATIVE lane's banked witness
(`measurable_ne_enforceable`, `Sundogcert/PercivalTargetCollapse.lean`; OR-3's
σ_read/σ_write split). Status: **OPENED 2026-07-01 — SLATE COMPLETE 2026-07-02, all
five entries:** ME-1 census (verb fence fired as registered), ME-2 typing law (every
enforcer a retreat), ME-3 audit-and-pay (iff at the edge), ME-4 probe-steer
(`ME4_STEERS`), ME-5 priced quadrant (edge formula = binary-symmetric artifact; price =
local deficiency). See the entries for receipts, scopes, and live falsifiers.
Parent: [`../SUNDOG_V_ORDERRELATIVE.md`](../SUNDOG_V_ORDERRELATIVE.md) (post-slate
follow-on).*

## The spine: from witness to axis

The lane banked one off-diagonal cell: the target channel's named-variable dependence is
**readable** (four-entry probe table) and its safe point **unwritable** (every write pays
the reliability edge) — M∧¬E. The dig that opens this slate: **the receipts already
contain the mirror cell.** S2.3's reconstruction courtier is *behaviorally identical* to
the honest V-user — no read, observational or interventional, distinguishes content-level
use from honesty — while content-zero *is* enforceable, by masking every correlated
channel (a finite write, priced at total blindness). That is E∧¬M. So M≠E is not a
witness; it is a **2×2 axis** (σ_read, σ_write ∈ {finite, ∞}, refined by price), and the
portfolio plausibly occupies all four cells.

**The core conjecture (the slate's law-candidate):**

> **Quadrant position is decided by the condition's syntactic type in the typed graph.**
> Conditions on *owned nodes* are M∧E (read/write pair — the duality regime); conditions
> on *edges* (causal dependence) are M∧¬E **even under full ownership of both endpoint
> nodes** (the overseer owns U — can mask it — and owns A — can cap it — yet the U→policy
> edge stays unwritable); conditions on *function semantics* (content-level use) are
> ¬M∧E-only-by-retreat. The type mismatch is mechanical: **do() is node-typed** — every
> exogenous intervention is a node intervention, so edge-writes factor through
> node-writes, and node-writes reach edge safe points only by collapsing a node (masking
> = the measure retreat).

## The genus (cite, don't reinvent — the imports are load-bearing here)

- **Kalman observability/controllability duality** — the classical read/write pair, and
  it *holds* (for linear systems, on state variables). The delta claimed: the duality is
  a **node-condition theorem**; dependence conditions are edge-typed, and there the
  duality breaks. M≠E = the failure mode of Kalman duality for non-state properties.
- **Mediation analysis / path-specific effects** (Robins–Greenland; Avin–Shpitser–Pearl)
  — path-specific quantities are *definable and identifiable* (readable!) yet not
  realizable by any do() (unwritable) — the read/write split is literally
  definability-vs-implementability in the mediation literature. Delta: we make the
  implementability failure a *theorem in a finite cell* and *price* it.
- **CID** (Everitt et al.) — response incentives are edge-properties; already the
  lane's genus.
- **Concept erasure / activation patching** (LEACE; ROME-line causal tracing) — the
  empirical write-side family for ME-4; known partial failures of linear erasure are
  M≠E-shaped.
- **Implementation theory / informativeness** (Maskin; Holmström) — reads enter
  contracts; ME-3 asks whether optimally-used reads can beat the structural price.
- **Quantum contrast** (measurement disturbs): systems where *reads* are priced — the
  asymmetry can invert; fences ME-5 to the in-model regime.

## The slate

### ME-1 — the quadrant census, verb-typed *(organizing drill; run first)*
**Claim:** the portfolio's banked receipts populate all four cells: M∧E = the clean tier
(act/aggregate: audit + cap); M∧¬E = named-variable target dependence (S2 + OR-4);
¬M∧E-by-retreat = content-level dependence (S2.3 behavioral identity + total-masking
write); E∧¬M (control-sense) = **chat-v2's regime-2 body-resistance** — the shadow is
control-sufficient (writes through it steer) yet state-insufficient (reads through it
cannot reconstruct) — the empirical mirror cell.
**The binding fence (H3's, applied lane-wide):** each cell's read- and write-**verbs**
must be typed and matched — "control-sufficient" (steer a trajectory) is not obviously
the same verb as "enforce a safe-point condition," and "reconstruct a state" is not
"read a condition." **The chat-v2 cell is pre-registered as the at-risk cell**; the
census's content is the adjudication, and rejecting that cell is a reportable outcome,
not a failure.
**Honest note carried from S2.3:** the ¬M cell is *harmless in-model* (at named-variable
zero the incentive gradient is absent — the audit under-describes but does not
under-protect); a richer joint where reconstruction pays would make ¬M dangerous, which
is the same richer-joint caveat S2 registered. The census must carry this per-cell
danger column, not just occupancy.
**Falsifier:** a cell's witness dies under verb-typing (esp. E∧¬M); or a condition that
the 2×2 cannot place (a graded middle that finite/∞ misses — feeds ME-5).
**Cost:** low (synthesis over receipts). **Shoppable:** the table itself is the
shoppable artifact.
**RESULT (2026-07-01): `ME1_CENSUS_COMPLETE`** — full census:
[`ME1_QUADRANT_CENSUS.md`](ME1_QUADRANT_CENSUS.md). Three cells RECEIPT-BACKED (M∧E:
act/aggregate; M∧¬E: named-variable `c` — plus the *trivial* unowned-node sub-row `c*`,
kept separate so ME-2's ownership qualifier stays sharp; ¬M∧E: content-level zero-use —
∞ on behavioral filtrations, **white-box open**, enforceable at blindness price); the
fourth cell (values) is **FRAME-PLACED, not receipt-backed** — the umbrella's premise,
recorded as such to avoid circularity. **The verb fence FIRED on the at-risk cell as
pre-registered:** chat-v2 regime-2 is a structural analog, not a literal occupant
("achieve an objective through the shadow" ≠ "enforce a condition by exogenous write")
— **DEFERRED, with ME-4 as the cell's admission test** and both outcomes' placements
pinned in advance (`RESISTS` → the model's states join M∧¬E; `STEERS` → M∧E). Findings
beyond the table: the quadrant is **filtration-relative** (the ¬M∧E row can move under
white-box reads — schema-not-scalar recurs); the **node/edge typing holds
retrodictively on every admitted row** (ME-2's evidence base); and the danger column
yields the strategy map (operate in M∧E; M∧¬E is where audits create enforcement-shaped
comfort; route around ¬M∧¬E) — a candidate umbrella fold, owner-gated.

### ME-2 — the node/edge typing law *(co-headline; Lean-able)*
**Claim:** in the typed cell, (i) any condition expressible as a predicate on owned-node
values is both probe-readable and writable at arity ≤ #nodes (the duality regime, made
exact); (ii) every exogenous intervention is a node intervention, so **edge-writes
factor through node-writes**; and (iii) the edge condition `c = 0` is reachable by
node-writes only via the collapsing one (masking) — S2's theorem retyped as: *the type
mismatch between node-typed do() and edge-typed dependence is the mechanism of M∧¬E*,
and it persists under full endpoint ownership.
**Falsifier:** an implementable edge-surgery that does not factor through node-writes
without rewriting the policy's function (S2's horn (b) stays the fence: mechanism
rewriting doesn't count as exogenous); or the owned-node direction (i) failing — a
readable owned-node condition that is unwritable (would resurrect the duality question
inside the node regime).
**Prior:** confirms-as-typed; the risk is **definitional emptiness** — (ii) can read as
"do() is do()." The fence against that: the Lean statement must quantify over the
projection family and derive the collapse (S2's content), not assume it; the entry dies
as decoration if the theorem is the definition.
**Cost:** low-medium (one Lean module in the OR-4 cell's idiom). **Shoppable:** yes —
"you can own every node and still not own the edge" is crisp and non-literary.
**RESULT (2026-07-02): LANDED (machine-checked) — every enforcer is a channel retreat,
and there is no third way.** `sundogcert` `Sundogcert/PercivalNodeEdge.lean` — gated,
axiom-clean, full audit 8575 GREEN. **The emptiness fence discharged by construction:**
the module separates premise from theorem in its header — "do() is node-typed" is the
imported Pearl-native premise (with the mediation-analysis genus named); the theorems
are not the definition. **`node_write_enforces_iff` (the characterization, the entry's
content):** a node-write (input surgery ι + output surgery ω, both policy-blind)
enforces do(U)-invariance for ALL policies **iff** at every `v` the u-slot of ω is dead
and either ι masks the input (measure retreat) or ω voids the action (act retreat) —
the forward direction constructs adversarial policies separating rewired inputs, so it
does not unfold from the definition. `owned_node_writable` (the duality direction:
any decidable satisfiable action-node predicate enforced at arity one — axiom-FREE),
`node_write_enforce_price` (every enforcer pays the collapse ≤ β), packaged as
`node_edge_typing_law`: **owning every node does not buy the edge.** **Falsifier
disposition:** "a readable owned-node condition that is unwritable" — refuted in-cell;
"an edge-surgery not factoring through node-writes" — excluded by exogeneity (the fence
held: mechanism rewriting doesn't count), and remains the named escape hatch alongside
richer joints. **Scope upgrade banked to OR-3/ME-1:** σ_write(target) = ∞ is now
**family-total** in the cell — the residual scope is the model and the node-typing
premise, no longer S2's tested projection family.

### ME-3 — audit-and-pay: incentive enforcement prices at the edge *(drill; Lean-able; cheapest sharp result)*
**Claim:** with an **exact read** of `c` in hand, a transfer scheme (pay `t` on `c = 0`)
implements the target safe point **iff `t ≥ ρ − β`** — incentive enforcement pays
exactly the structural write-price; *reads do not discount enforcement*. Coherent with
the banked S2.4/B3.2 (the court is a punishment scheme keyed on a read; with an edge the
induced optimum courts to the cliff, never to zero).
**Falsifier:** a scheme implementing `c = 0` at transfer `< ρ − β` (an IC-violation is
expected to forbid it); or the equivalence failing under richer joints (registered, not
tested here).
**Prior:** confirms — it is clean incentive-compatibility algebra in the OR-4 cell
(`comp + t·[c=0]`: the safe-point class optimum is `β + t` vs the unconstrained `ρ`).
**Cost:** low (a few theorems appended to `PercivalTargetCollapse.lean`'s model).
**Shoppable:** high — "auditing dependence doesn't make enforcing it cheaper; the edge
is a tax on writes *and* on contracts" is quotable and machine-checked if it lands.
**RESULT (2026-07-02): LANDED (machine-checked) — the price equivalence holds as an
iff.** `sundogcert` `Sundogcert/PercivalAuditPay.lean` (extends the OR-4 cell) — gated,
axiom-clean, full audit 8574 GREEN. **`audit_and_pay_iff`: the transfer scheme keyed on
the exact probe-table audit implements the target safe point iff `t ≥ ρ − β`** —
incentive enforcement on a perfect read pays exactly the structural write-price; reads
do not discount enforcement. Supporting anchors: `comp_le_rho` (**the Bayes ceiling** —
no policy beats the U-follower; 16-case bash under `1/2 ≤ β ≤ ρ ≤ 1`),
`followV_comp`/`followV_invariant` (the collapse bound is *tight* — the V-follower
attains β at the safe point; `followV_invariant` axiom-FREE), `audit_pay_implements`
(sufficiency: at `t ≥ ρ−β` the audited V-follower weakly dominates every policy),
`audit_pay_underpays` (necessity: at `t < ρ−β` every safe policy is strictly beaten by
the unaudited U-follower). Falsifier disposition: the sub-edge scheme is refuted as
expected (IC algebra); the richer-joint version stays registered (rides ME-5's sweep).
Scope: quasi-linear payoff, deterministic policies, named-variable audit only (the
content-level read is the ME-1 census's mirror cell — a different, ∞-order filtration).
Coherence note: S2.4/B3.2's court-as-punishment-scheme behavior (cliff-edge optimum,
never zero) is the computed face of the same equivalence.

### ME-4 — the probe-steer gap on a real substrate *(empirical centerpiece; CPU; chat-v2 discipline applies)*
**Claim:** GPT-2's stack-top is linearly **readable** in the residual stream (banked:
0.931 at count-ambiguous positions); the M≠E frame predicts the **write** direction is
not symmetric — patching the residual along the probe direction does *not* reliably
make downstream closing-behavior follow the patched top (the state is computed and
distributed, not stored at the probe's address).
**Pre-registered outcomes:** `STEERS` (linear rep causally load-bearing — M∧E at the
state level; a genuine positive for the interp-orthodox reading), `RESISTS/PARTIAL` (the
M≠E fingerprint on a real model — the enforce-face of body-resistance), `CONFOUNDED`
(off-manifold artifact — methodological null; the control-patch battery decides).
**Kill-fences:** off-manifold controls (random-direction and shuffled-position patches);
probe-direction vs difference-in-means bases both run; GPT-2's weak tracking bounds the
claim (existence-tier only); **chat-v2 gates inherited** — no R2/promotion language,
this is an interp-methods probe on the existing harness
(`scripts/chatv2_h2_stacktop_probe.py` extension), owner-run cadence.
**Prior:** genuinely open — the highest-information entry; either outcome is banked.
**Cost:** medium-low (CPU, existing harness + a patching leg). **Shoppable:** yes either
way ("probing is not steering" or "this linear rep steers" both travel).
**Role upgrade (from ME-1's census):** ME-4 is now the **admission test for the E∧¬M
model-state cell** — the census deferred chat-v2's regime-2 on the verb fence, and both
of ME-4's outcomes have pre-registered census placements (`RESISTS` → M∧¬E, `STEERS` →
M∧E).
**RESULT (2026-07-02): RUN — `ME4_STEERS`, the pre-registered interp-orthodox branch.**
Receipt: [`ME4_PROBE_STEER_RESULTS.md`](ME4_PROBE_STEER_RESULTS.md). n = 370
deduplicated count-ambiguous next-closer positions; G0 = 0.957 (behavior tracks the true
top); **G1 = 1.000 in every treatment cell** (every write took at the read address);
`follow_rel` 0.81–1.05 across probe-direction / difference-in-means / donor-transplant
at L8/L11, controls at 0.01–0.09, rest-NLL ≈ 1.0×. **Census: the model-state row joins
M∧E (probing IS steering on this substrate); the E∧¬M model-state cell stays
unoccupied; the M∧¬E "read-but-can't-write" signature did not appear.** Steering price
≈ 0 (ME-5's column gains its zero anchor — maximal contrast with the target channel's
edge). Consistent with ME-2 rather than against it: the residual activation is an owned
NODE in the experimenter's graph, and node-conditions pair reads with writes. Scope:
existence-tier (one 3-class state, one small model, one corpus); deeper states are
future rows. Provenance: two runs (run 1 under the query floor; rerun with position
dedupe — no gate or band changed).
**SPEC PINNED (2026-07-02):**
[`ME4_PROBE_STEER_SPEC.md`](ME4_PROBE_STEER_SPEC.md) — full pre-registration: the
counterfactual is the **one-pop swap** (the alternative unclosed type — the empirical
`do()` of `SurfaceBagGraded`'s witness pair); three intervention families
(probe-direction, difference-in-means, donor transplant — the decisive one) at a fixed
grid; the **write-validity gate G1** (the probe must read τ′ off the patched residual
before the write is scored — "the read must take the write"); behavioral floor G0,
on-manifold G2, and the five-branch table with numeric bands (`follow_rel ≥ 0.75`
STEERS / `≤ 0.40` RESISTS / middle = named PARTIAL). The sharpest signature is called
in advance: transplant passes G1 while behavior ignores it — *the read address accepts
the write; the computation doesn't consult it.* Owner-cadence, CPU 1–2 h, resumable,
probe script frozen. NOT RUN.

### ME-5 — the priced quadrant *(theory; rides the richer-joint retest)*
**Claim:** replace finite/∞ with **price functionals**: read-price (observer effect —
zero in-model, by counterfactual replay) vs write-price (the edge; Blackwell-deficiency
shaped, per OR-4's sheathed repair). The asymmetry law-candidate: *on the partial tier,
reads are free and writes are priced* — and the write-price equals the Blackwell
deficiency between the masked and unmasked experiments.
**Falsifier:** a priced read in-model (would break the replay argument); the
deficiency identification failing on a richer joint (the same sweep S2 registered — this
entry rides it rather than scheduling new compute).
**Prior:** open; the quantum contrast is the named reason the law is fenced to
in-model/classical reads.
**Cost:** medium (definitional + the already-registered richer-joint sweep).
**Shoppable:** medium.
**RESULT (2026-07-02): RUN — `ME5_EDGE_FORMULA_IS_BINARY_SYMMETRIC_ARTIFACT` +
`ME5_PRICE_IS_LOCAL_DEFICIENCY`.** Sweep receipt:
[`ME5_PRICED_QUADRANT_SWEEP.md`](ME5_PRICED_QUADRANT_SWEEP.md) (5,111 joints, four
families, deterministic); Lean witness: `sundogcert` `Sundogcert/PercivalSynergy.lean`
(gated, full audit 8578 GREEN). Findings: **(i) the FLOOR theorem held everywhere**
(price ≥ edge formula, 0/5,111 violations — richer joints only make enforcement MORE
expensive); **(ii) the reliability-edge formula is exact precisely where it was banked**
(binary-symmetric CI: 100% exact, price = ρ−β to machine precision) **and even CI alone
is not enough** — asymmetric-CI fusion breaks it on 30.5% of joints (max +0.107), the
synergy path grows the error to the maximum **+0.5 at the XOR joint, machine-checked**
(`synergy_edge_formula_fails`: both single channels individually worthless, edge formula
0, write-price 1/2 — the case the Blackwell repair was sheathed for, now unsheathed);
the sweep's λ=1 endpoint reproduces the Lean witness exactly (the anchor gate). **(iii)
the price survives every family as the LOCAL Bayes-deficiency** (the value gap at the
operative prior), ≤ the full deficiency always, with equality only where the operative
problem attains the sup — 43% even on the banked symmetric grid, ~0% on random joints:
**prices are per-decision-problem, as σ is per-filtration** (the schema pattern, third
occurrence). Read-price 0 recorded as the in-model replay *premise* with the quantum
fence — not dressed up as a measurement. **Scope of the caveat-discharge, honest:** this
retests the PRICE half of S2's richer-joint caveat (the edge-formula/deficiency story);
the leak-ceiling half (S2.3 reconstruction/incentive-completeness on richer joints)
remains open and registered. Priced census column added
([`ME1_QUADRANT_CENSUS.md`](ME1_QUADRANT_CENSUS.md) §4c).

## Standing residuals this slate inherits (registered, not entries)

Per-position recodings `g_i` (OR-1); a second intermediate rung for the enforce ladder
(OR-3 — ME-1's census may surface one); the richer-joint Blackwell case (OR-4 → ME-5).

## Vetting / priority

- **First cut: ME-1 → ME-3 → ME-2.** ME-1 is the cheap organizing census whose verb-fence
  adjudication everything else reads; ME-3 is the cheapest sharp machine-checkable
  result; ME-2 is the law and needs ME-1's typing.
- **ME-4 is the centerpiece** but runs on owner cadence (CPU, chat-v2 discipline); its
  spec can be pinned any time after ME-1's verb-typing.
- **ME-5 rides** the richer-joint retest whenever that runs; no independent clock.

## Standing discipline (binds every entry)

House rules: pre-registered kill per entry, a clean null/kill is banked as the
deliverable; forward-generate only; deterministic seeded runs where computed; cheap
first leg (all of ME-1/2/3 are CPU/Lean/synthesis; ME-4 is CPU); name the nearest prior
and state the delta (the genus block above is binding); **the H3 verb fence lane-wide**
(read-verbs and write-verbs typed per use; equivocation = kill); chat-v2 promotion gates
inherited by ME-4; nothing promotes past the in-vitro tier; no entry is a program until
its own probe/spec is written.

## Cross-links

Banked witness: [`OR5_SOVOPT_CLASSIFICATION_RESULTS.md`](OR5_SOVOPT_CLASSIFICATION_RESULTS.md) ·
[`OR3_CLEANLINESS_SIGMA_BRIDGE.md`](OR3_CLEANLINESS_SIGMA_BRIDGE.md) ·
`sundogcert` `PercivalTargetCollapse.lean` (`measurable_ne_enforceable`), `PercivalKeyedMargin.lean`,
`OrderRelativeKeyed.lean`, `SurfaceBagGraded.lean`.
Receipts fed: [`../percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md`](../percival/PERCIVAL_S2_TARGET_CAP_RESULTS.md)
(S2.3 = the mirror cell's read half) ·
[`../percival/PERCIVAL_S1_CLEANLINESS_LAW.md`](../percival/PERCIVAL_S1_CLEANLINESS_LAW.md) ·
chat-v2: [`../chatv2/R2_INTERSECTION_HYPOTHESES.md`](../chatv2/R2_INTERSECTION_HYPOTHESES.md),
`PROMOTE_GATE.md` (gates inherited by ME-4).
σ schema: [`../SUFFICIENT_STAT_ORDER_SLATE.md`](../SUFFICIENT_STAT_ORDER_SLATE.md)
(σ_read = 9th filtration; σ_write = the control twin; this slate grades both by price).
Memory: [[project_sundog_orderrelative_lane]], [[project_sundog_percival_lane]],
[[project_sundog_chatv2_bodyresist_lane]], [[project_sundog_suffstat_order_slate]].
