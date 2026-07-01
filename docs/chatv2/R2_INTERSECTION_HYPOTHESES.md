# Chat-v2 R2 — The Intersection Hypotheses (what label family is high-dim AND surface-undecodable?)

> 2026-07-01, hypotheses slate. **Non-promotional, nothing run here.** Seeded by the three R2
> admissions (`R2_REAL_SUBSTRATE_SPEC.md` MVP F3, `R2_V2_ADMISSION_RECEIPT.md` agreement
> F3-input, `R2_LM_DATA_AUDIT_RECEIPT.md` FewRel 0/64). Those triangulate the R2 gate's real
> obstacle: the intersection **{surface-undecodable ∧ model-computed ∧ ≥24 independent axes}**
> was empty for count-parity (undecodable, not computed), agreement (computed, decodable), and
> relations (high-dim, trigger/type-decodable). This slate asks the constructive question:
> **is any label family in the intersection, and which?**

## What the failures triangulate

Every surface probe used (bag-of-tokens, tf-idf n-gram, delexicalized, entity-type/metadata,
masked-trigger) is **order-blind (or local-order) and linear** — it reads a label iff the
label ≈ a linear function of token counts within an n-gram window. So a label is
**surface-undecodable** only if it is either **(a) nonlinear** in the counts (parity/XOR) or
**(b) long-range order-dependent** (depends on the sequence beyond any n-gram window). The dead
families: count-parity = (a) but *not model-computed*; agreement = a *local* cue (decodable);
relations = *trigger/type* cues (decodable). **None were long-range order-dependent states.**

## The core conjecture

> **The intersection is occupied by long-range order-dependent STATE: the current value of a
> variable updated across a long context, whose value depends on the order and cumulative
> updates — so no order-blind surface probe can read it, yet a capable sequence model computes
> and linearly represents it because tracking that state is useful for next-token prediction.**

**Existence proof (non-empty corner):** OthelloGPT (Li et al. 2022 → Nanda 2023). Board state
is high-dimensional (64 squares), **linearly represented** in the residual stream (computed),
and **not linearly readable** from the move-token sequence (undecodable — flips are
order-dependent). The corner is occupied — on a synthetic-game LM. R2's real question is
whether a capable pretrained LM carries such states over *real* text.

**Portfolio bridge:** a surface probe is an *order-blind, low-order sufficient statistic*; a
label is undecodable ⟺ its minimal sufficient statistic has **order > the probe's window** —
which is the **suffstat-order σ** of `docs/SUFFICIENT_STAT_ORDER_SLATE.md`. So
"undecodable ∧ computed" = **"a high-σ state the model computes,"** and the R1 toy worked
because it *trained* the model on running-parity — the canonical high-σ latent. R2 asks whether
a *pretrained* model already carries high-σ states. (See H5.)

---

### H1 — Emergent world / entity-state tracking (primary)

**Statement.** The intersection is occupied by the model's own *state variables*: entity
location/possession/status after a narrative or procedure; game-board state from move notation.
High-dimensional (entities × states); the actual "world-model / body-resistance" story.
- **Why from this:** OthelloGPT proves the corner is non-empty; it is the highest-σ family a
  capable LM plausibly carries.
- **Attack:** probe a capable LM's residual stream for a trackable state (game board from PGN;
  entity-state after a procedure / bAbI-TextWorld-style text) vs surface + random-init floor;
  run the R2 fingerprint on the state bank.
- **Kill if:** the state is bag-linearly-decodable (some are — check first), or the LM doesn't
  linearly carry it (F2), or annotation forces a synthetic corpus that violates the "real text"
  spirit without an amendment.
- **Promise high / risk medium (annotation/corpus).** The real reopening of the R2 gate.

### H2 — Stack-top / expected-closer in nested structure (cleanest, testable now)

**Statement.** In nested structure (code brackets, Dyck-k), the **top of the stack** — which
delimiter must close next — is order-dependent and undecodable, while a capable code-LM tracks
it. **Trap to avoid:** nesting *depth* = `#open − #close` is *linearly decodable* from prefix
counts; the stack-**top** (not depth) is the undecodable, high-σ quantity.
- **Why from this:** the cleanest instance of the conjecture and CPU-testable — brackets are
  real (code), high-cardinality (bracket type × context), and GPT-2 sees code in WebText.
- **Attack:** on real code, probe GPT-2 (CPU) for the stack-top at each position; compare a bag/
  n-gram surface probe (must fail) vs the residual-stream probe (should succeed) vs random-init
  floor. This is the cheapest empirical test of the whole slate.
- **Kill if:** the stack-top is bag/n-gram-readable after all (short stacks may be), or GPT-2
  doesn't carry it (then it needs a code model / bigger LM).
- **Promise high / risk low / runnable now.** The falsifier for the whole "state-tracking"
  answer. **→ hardening first (owner's call).**

### H3 — Long-range cumulative / toggled quantities

**Statement.** Order-and-reset-dependent running values a strong model tracks: score-leader
after updates, on/off after toggles, current value after +/−/reset operations, truth-value
after negations. "Count-parity as a discourse state a capable model actually computes."
- **Why from this:** it is the (a)-nonlinear family (parity-like) *plus* order/reset, in the
  regime where a capable model has reason to compute it.
- **Attack:** construct/annotate texts with a tracked quantity; probe the residual stream vs
  surface; needs a model strong enough to do the running update (GPT-2 likely too weak → bigger).
- **Kill if:** even a capable model doesn't track it (F2), or the quantity is bag-decodable
  (e.g., monotone counters).
- **Promise medium / risk medium (model capability).**

### H4 — Long-range binding / coreference under attractors

**Statement.** The antecedent the model resolves via discourse when the nearest-mention
heuristic is *wrong* — the reference generalization of the agreement "attractor" slice, but
higher-cardinality (many entities).
- **Why from this:** binding is order/discourse-dependent (high-σ) where the local surface cue
  fails; the attractor filter is exactly what killed decodability for agreement number.
- **Attack:** filter coreference data to non-nearest-antecedent cases; probe the model's
  resolved-referent representation vs a nearest-mention surface baseline.
- **Kill if:** low-dimensional (few binding sites per passage) or still surface-leaky after the
  attractor filter.
- **Promise medium / risk medium (dimensionality).**

### H5 — The characterization + the σ-bridge (drill-in / theory)

**Statement.** Formalize the intersection: undecodable ⟺ the label's minimal sufficient
statistic has **order > the surface probe's window** (bag = order-0/1); computed ⟺ linearly in
the residual stream. Then "undecodable ∧ computed" = **"a high-σ state the model computes,"**
unifying R2 with the parity lane and `SUFFICIENT_STAT_ORDER_SLATE.md`. Reframes R2 operationally:
**probe the model's KNOWN emergent state variables, do not hunt undecodable labels in arbitrary
text.**
- **Why from this:** it explains *why* the three families failed and *why* state-tracking is the
  answer, in one axis (σ), and connects to the machine-checked suffstat/Lean framework.
- **Attack:** a written note + a small formal statement (surface-probe = order-≤w sufficient
  statistic; undecodable-at-order-w ⟺ σ > w) in the suffstat idiom; possibly a Lean anchor.
- **Kill if:** the σ-order framing doesn't cleanly separate the decodable vs undecodable
  families on the actual data (i.e., it's post-hoc, not predictive).
- **Promise medium (synthesis) / risk low.** The organizing spine; makes H1–H4 one story.

---

## Routing & status

| hook | promise | risk | testable-now? | status |
| --- | --- | --- | --- | --- |
| H1 world/entity-state | high | medium | no (needs capable LM + annotation) | PROPOSED — the R2-v3 reopening |
| **H2 stack-top / Dyck** | **high** | **low** | **yes (CPU, GPT-2 + code)** | **CONFIRMED (existence) 2026-07-01** — see result below |
| H3 cumulative/toggled | medium | medium | partial (bigger model) | PROPOSED |
| H4 binding under attractors | medium | medium | partial | PROPOSED |
| H5 σ-bridge characterization | medium (synthesis) | low | yes (analysis/Lean) | PROPOSED — the spine |

## The payoff (why this isn't just theory)

LM-0 killed *relations* because they are low-σ (surface triggers). **State-tracking (H1/H2) is
high-σ by construction** — the family that defeats the surface probes *because* it is what a
sequence model computes. So a genuine **R2-v3** = a state-tracking label family (stack-top or
entity-state) on a capable model, and *there* the H200/larger-model route would buy something.
The intersection is not empty; the earlier admissions were fishing in surface-cued labels
instead of the model's own integration targets.

## H2 hardening result — CONFIRMED (existence), 2026-07-01

`scripts/chatv2_h2_stacktop_probe.py`, GPT-2-small (CPU), real Python code, ~12.4k valid
query positions (depth ≥ 1), class-balanced (chance 0.333). Decisive baseline = the **pure
order-blind sufficient statistic** (bracket counts), not the bag-of-tokens (whose GPT-2
subwords leak local order and confounded the first pass).

| positions | counts (order-blind) | GPT-2 residual (L11) | control: counts→depth |
| --- | --- | --- | --- |
| ALL (mostly single-type stacks) | **0.965** | 0.926 | 0.947 |
| count-ambiguous (≥2 types unclosed) | **0.770** | **0.931** | 1.000 |

**The crossover is the finding.** Where the stack is single-type, counts *determine* the top
and beat the residual (0.965 > 0.926) — stack-top is *count-decodable*, not in the intersection.
But at count-ambiguous positions (`([` vs `[(` — identical counts, order-determined top) the
order-blind statistic collapses to 0.770 while **GPT-2's residual holds at 0.931 (+0.161)**, with
the depth control fully live (counts→depth 1.000). The residual is robust to the ambiguity that
cripples the order-blind statistic — the fingerprint of a genuinely **order-dependent state the
model computes.**

**What this establishes / doesn't.** Establishes: the intersection is **non-empty on a general
small pretrained LM over real text** — an order-dependent label (bracket stack-top) that a
proper order-blind statistic cannot read is linearly present in GPT-2's residual. This validates
the slate's **state-tracking answer** and the σ-bridge (H5): the undecodable-∧-computed label is
exactly a *high-σ state the model already computes*. Does **not** establish the `d_dec ≥ 20`
R2 gate — stack-top is a 3-class (low-dim) quantity; H2 is an **existence proof**, not a full
R2 bank. Caveats: GPT-2-small is a weak tracker (its all-position residual even loses to counts);
single quantity, one corpus (Python), light string/comment lexer.

**Onward (owner's call).** The intersection being reachable, the high-dimensional route is H1
(world/entity-state — order genuinely matters, *not* count-determined like a stack) on a capable
model; a code-trained / larger model would also widen the H2 margin. This is the first **positive**
in the R2 arc: it says the earlier F3s were a *substrate/label-family* limitation, not a wall.

## Fences

Still **R2 exploration, not an R2 claim.** No promotion, no public surface, no R3 / world-model
language. `PROMOTE_GATE.md` R2 stays NOT STARTED until a state-tracking family clears the full
gate *and* external review. Cross-refs: the three admission receipts above,
`R2_LARGER_MODEL_ROUTE_CAMPAIGN.md`, `SUFFICIENT_STAT_ORDER_SLATE.md`.
