# The Steer-Arity Ladder — Hypothesis Slate (spec)

*ME-4's sequel: the enforce-order ladder (OR-3: 1 < Θ(M) < ∞, keying decides the growth
law) instantiated INSIDE the transformer. Parent:
[`../SUNDOG_V_ORDERRELATIVE.md`](../SUNDOG_V_ORDERRELATIVE.md);
banked anchor: [`ME4_PROBE_STEER_RESULTS.md`](ME4_PROBE_STEER_RESULTS.md) (`ME4_STEERS`
= the ladder's arity-1 rung on a real substrate). Chat-v2 discipline inherited in full.
Status: **OPENED 2026-07-02. SPEC FOR RED-TEAM, NOTHING RUN.***

## The spine

> **Activation-patch arity is the model-side σ_write, and it is graded like the
> enforce-order ladder.** A model-state's tier is decided by *where its computation
> realizes it*:
>
> - **Register (arity 1):** cached in the activation at the readout site — one node-write
>   enforces. ME-4 banked GPT-2's stack-top here.
> - **Distributed (arity Θ(k)):** carried across k positions' residuals/KV — readout-site
>   writes fail; a *coordinated* k-site write succeeds. The coalition rung.
> - **Parametric (∞ at the activation filtration):** realized in the weights — the
>   condition is a **∀-quantified property of the function** (a rule, not an instance),
>   unreachable by activation writes at any arity. Enforcement requires mechanism
>   rewriting — **ME-2's fenced escape hatch wearing a training loop** (fine-tuning /
>   weight surgery). The model-side edge condition.

The ∀-quantification is the load-bearing typing fact, exactly as it was for `c = 0`:
*instances* of a rule are node-values (patchable one at a time); the *rule* is a property
of the map. ME-2's theorem shape predicts the split; SA-5 makes it provable in a toy
graph; SA-1..4 test it on GPT-2.

**Verb discipline (inherited, binding):** writes are `do(activation = value)` at a
declared set of (position, layer) sites; arity = |sites|; conditions are operationalized
behaviorally with per-state floors and write-validity gates (the ME-4 G0/G1 pattern —
the read must take the write before the write is scored). Site sets are **declared from
mechanistic priors before running** — no post-hoc site mining.

## The slate

### SA-5 — the provable trichotomy *(Lean spine; run first)*
**Claim:** in a small explicit computation graph (the `PercivalNodeEdge` idiom with
nodes = activations): (i) a node-realized state is enforceable by an arity-1 node-write;
(ii) a state defined as an aggregate across k nodes is enforceable at arity k and at no
smaller arity (the exact intermediate rung — also discharging OR-3's registered
"second intermediate-rung instance" residual on the formal side); (iii) a ∀-quantified
property of the network's *function* (a rule over all inputs) is enforceable by NO set
of node-writes (activation interventions are per-input; the rule quantifies over
inputs) while a weight-write enforces it — the three tiers as theorems.
**Falsifier:** the arity-k lower bound fails (some clever single-node write reaches the
aggregate condition), or the ∀-condition is reachable by input-uniform node-writes
(would collapse parametric into register — the "knob" existing in the toy).
**Prior:** confirms; the content is pinning the exact statements the empirical rows
instantiate (the OR-1 lesson: state the law so the empirical claim is decidable).
**Cost:** low-medium (one Lean module). **Shoppable:** as the slate's spine.

### SA-2 — the distributed rung: the induction-source row *(sharpest empirical test; the tier's admission)*
**Claim:** the induction/copy state — "the token that followed the previous occurrence
of the current token" — is **distributed**: it lives at the *source* occurrence's
sites, not the readout site. Pre-registered: patching the readout position (arity 1,
all ME-4 families) fails to change the induction continuation; patching the **source
position(s)** succeeds; random-site controls fail.
**Why this state:** induction heads are the best-understood circuit (Olsson et al.) —
the mechanistic prior localizes the state *away* from the readout, so the
register-vs-distributed prediction is sharp, cheap (CPU, natural repeated-token
windows), and grounded in known structure rather than curve-fitting.
**Kill-fences:** the state may partially cache at the readout (GPT-2 copies into the
final residual) — the pre-registered middle band is "split state" (partial follow at
arity 1, full at source+readout arity 2), a named outcome; behavioral floor = unpatched
induction preference must be live; write-validity via probes at both sites.
**Prior:** genuinely open between distributed and split; either populates the middle
rung. **Cost:** medium-low (ME-4 harness + source-site patching). **Shoppable:** yes —
"steering needs a coalition of positions" is the coalition rung on a real substrate.

### SA-1 — the ladder battery *(the census rows)*
**Claim:** a pre-registered battery of states spans the tiers, each with a structural
tier-prediction declared in advance: **depth** (register predicted — linearly cached,
H2's control); **the second stack element** (top-after-one-pop; genuinely open —
register if GPT-2 caches the whole stack OthelloGPT-style, distributed if only the top
is cached and the rest recomputed); **induction target** (SA-2's row); plus one
optional high-risk row (entity-state) marked floor-prone on GPT-2-small.
**Falsifier per row:** the ME-4 gates; **falsifier for the battery:** a row whose
measured arity contradicts its declared prediction *and* survives the controls — which
feeds SA-4 rather than killing the slate (the law is what's on trial, per-row).
**Cost:** medium (each row is an ME-4-shaped run). **Shoppable:** the populated ladder.

### SA-3 — the parametric rung: the rule/instance split *(the escape hatch, made literal)*
**Claim:** the counterfactual grammar "`(` is closed by `]`" (swap the closer rule) is
**parametric**: per-*instance* writes succeed (banked — ME-4's one-pop swap is an
instance write), but no bounded-budget activation write enforces the *rule* across
contexts — neither probe-derived constant vectors nor a small **optimized** steering
vector (budget declared: one vector per layer, trained on the rule objective, fixed
optimization steps) — while a **weight-write** (small fine-tune on swapped-grammar
data) enforces it. The third tier witnessed, with ME-2's exogeneity fence made
empirical: activation writes are node-typed and per-input; the rule is a property of
the function.
**Pre-registered counter-outcome (named, big if true):** `SA3_KNOB` — an optimized
constant vector DOES implement the rule globally (the refusal-direction precedent says
some ∀-ish behaviors have knobs). Then the rule is register-typed, the parametric tier
needs a harder witness, and the census gains a surprising row instead.
**Kill-fences:** rule-enforcement metric declared (swapped-closer preference across a
held-out context battery, floors as ME-4); the FT leg is the gated/owner-cadence half
(CPU LoRA-scale); "fails at bounded budget" is honestly budget-relative — recorded as
such, not as an impossibility proof.
**Prior:** open — the knob outcome is live. **Cost:** medium (+ the gated FT leg).
**Shoppable:** high either way ("rules aren't steerable, only trainable" vs "grammar
has a knob").

### SA-4 — the law: read-localization predicts write-arity *(the verdict entry)*
**Claim:** across all rows, the **σ_read localization profile predicts the σ_write
arity**: readable-at-readout ⟹ arity-1 steerable; readable-only-at-distributed-sites ⟹
arity = the site count; not activation-readable anywhere while behaviorally live ⟹
parametric. The bridge hypothesis on the substrate — read structure determines write
structure.
**Falsifier (the census-splitting one):** a state *readable at the readout site* that
still resists arity-1 writes — M∧¬E inside the model, the row ME-4 didn't find; or
readable-only-distributed yet arity-1 steerable. Either is the higher-information
outcome and gets banked as the deliverable.
**Prior:** open-leaning-confirms on the battery's easy rows; SA-2/SA-3 carry the risk.
**Cost:** low (analysis over the rows). **Shoppable:** high — this is the slate's law.

## Vetting / priority

- **First cut: SA-5 → SA-2 → SA-1(depth + second-stack) → SA-3(knob leg) → SA-4.**
  SA-5 pins the statements; SA-2 is the sharpest single test and delivers the middle
  rung (or the split-state finding) cheaply; SA-3's FT leg is the one gated item and
  can trail.
- SA-3's knob search and SA-1's optional entity row are the budget-risk items; both
  carry named early-exit outcomes.
- Every empirical row inherits the ME-4 gate battery verbatim (G0 floor / G1
  write-validity / G2 on-manifold + controls / pre-registered bands).

## Standing discipline (binds every entry)

House rules in full: pre-registered kill per entry, clean nulls banked; deterministic
seeds; frozen instruments (the H2/ME-4 scripts untouched; new work in new scripts);
site sets and optimization budgets declared before running; the verb fence lane-wide;
chat-v2 promotion gates inherited (no R2 / world-model language — these are
existence-tier rows on a small model); name the nearest prior and the delta.

## The genus (cite, don't reinvent)

- **Induction heads** (Olsson et al.) — SA-2's mechanistic prior; the delta: we use the
  known circuit to *pre-register a write-arity*, not to re-derive the circuit.
- **ROME / MEMIT** (rank-one weight edits) — the parametric tier's known citizens:
  facts edited by *weight* surgery; the delta: the arity ladder places weight-writes as
  the tier *above* every activation arity, one graded frame.
- **Steering vectors / ITI / refusal-direction** — the knob precedent, and exactly why
  `SA3_KNOB` is a named outcome rather than an afterthought.
- **Activation patching / causal scrubbing** — the write machinery; the delta: arity as
  the graded invariant with the enforce-order law (OR-3) behind it, and the
  read-localization→write-arity predictor (SA-4) as a falsifiable law rather than a
  method.
- **LEACE / concept erasure** — the erase-side cousin of the parametric boundary.

## Cross-links

Banked rungs this ladder extends: [`ME4_PROBE_STEER_RESULTS.md`](ME4_PROBE_STEER_RESULTS.md)
(arity 1) · [`OR3_CLEANLINESS_SIGMA_BRIDGE.md`](OR3_CLEANLINESS_SIGMA_BRIDGE.md) (the
enforce-order ladder + growth law) · `sundogcert` `PercivalNodeEdge.lean` (the typing
law SA-5 lifts into a network) · [`ME1_QUADRANT_CENSUS.md`](ME1_QUADRANT_CENSUS.md)
(rows land there) · chat-v2 gates: `docs/chatv2/PROMOTE_GATE.md`.
Memory: [[project_sundog_orderrelative_lane]], [[project_sundog_chatv2_bodyresist_lane]].
