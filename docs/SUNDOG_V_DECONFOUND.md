# Sundog vs. the De-confound Wall (computed-state substrate)

Working hook:

> The toy's hidden cause was ours. A model that computes its own state hides one we
> didn't plant. Can the sundial read *that*?

Short version:

> The determining-shadow-set instrument needs an **input-undecodable** functional or its
> read is trivial ("the body computes `u`" collapses to "`u` was in the input"). The toy
> got one by construction (parity). The **de-confound wall** is getting one on a *real*
> substrate. The door: **computed-state models** (Othello-GPT-class), where the model
> computes a hidden state (the board) that is **input-undecodable** (not linearly in the
> moves) yet **representation-decodable** (Nanda's linear world-rep) — the de-confound
> satisfied by **task structure** on a real trained transformer, and a state **the model
> built, not us.** Phase 0 ports the de-confound pre-check + body-shadow read to such a
> model; closure requires a separately preflight-passing linear functional. **Kill-gated
> R&D; an R2 down-payment.**

**Status:** Scaffold opened 2026-06-04. No public page, `site-pages.json` entry, spec, run,
or external packet. **Gated on (a) a lit pass and (b) a Phase-0 spec freeze before any run.**
Generality R&D (2026-06-04 pivot demoted it) → carries an explicit **kill-gate**.

**2026-06-04 preflight update:** the model-free Othello functional screen fired
**`blocked_by_ill_posed_functional`** for the named Othello closure slate in 76 seconds,
with no model run. Legal moves, material parity, frontier, and mobility all fail to produce
a non-vacuous linear `k_func << k_state` bracket. Othello remains a candidate for the
de-confound pre-check and a weaker regime-2 body/shadow read, but it is **not** the closure
substrate on the current candidate slate. Receipt:
`results/deconfound/othello-functional-screen-2026-06-04.txt`. The closure prize now routes
through **Attack B, semi-synthetic injection** unless a new canonical linear Othello
functional is named before spec lock.

Not a claim about AGI, frontier-LLM cognition, or "understanding." It is a measurement
question: does the determining-shadow-set / body-shadow read survive onto a real model that
computes its own hidden state.

## 1. Why this lane exists — the wall

Every real-substrate body/shadow read is blocked by one thing: on real data the input is
*given*, so natural functionals (topic, class, sentiment) are sitting in the input → a
linear input-probe reads them → the de-confound fails → "the representation computes it" is
vacuous. The toy dodged this by *constructing* an input-undecodable functional. This lane
gets one on a real substrate, which **unlocks the whole instrument** (JEPA, LLMs, the lot).

## 2. Thesis — computed-state models satisfy the de-confound by task structure

A sequence-prediction model that must **track a latent state** computes a functional that
is input-undecodable but representation-decodable. **Othello-GPT** is the exemplar (Li et
al. 2022; Nanda's linear board probe): the board is computed from the move stream (you must
track flips), so it is not linearly in the input, yet it is linearly in the residual
stream. The de-confound holds **for free**, on a real trained transformer — and the state
is the model's, so this is the first place "the sundial reads structure we did not design"
is *earnable*. The board squares are also **coupled by the rules**, so a closure bracket can
*seem* possible there. The preflight screen tightened this: computed-state structure gives
the de-confound, but it does **not** automatically give a linear closure functional.

## 3. Claim Boundary

Does **not** claim:
- that Othello-GPT (or any model) has a "world model" in any cognition sense;
- anything about frontier LLMs;
- that toy (allelopathy / Phase-7) results transfer;
- that a closure / regime-2 read certifies understanding or planning.

## 4. Why this fits Sundog

It is the **gate for every real-substrate determining-shadow-set / body-shadow read.** The
de-confound door (computed-state) carries part of the calibrated instrument across the
toy/real line that JEPA could not. The closure bracket still needs a linear functional; the
substrate being real and model-built is not enough.

## 5. The core questions

- **(a) De-confound:** is the computed state ≤-chance from a linear probe on the *input*,
  and well-decodable from the residual stream? (The pre-check, ported.)
- **(b) Determining-set / closure:** does a shadow-set of the state determine a registered
  functional (or the omitted state) before the individuals — reflecting the coupling the
  model learned? **Current Othello status:** blocked for the named candidate slate; requires
  a new canonical linear computed functional or a pivot to Attack B.
- **(c) Body/shadow regime-2:** does the model's *output* (the control-sufficient shadow,
  e.g. legal next moves) fail to reconstruct the *computed state* (body) → state-insufficient
  / resisting, on a real substrate?

## 6. Phase 0 (KILL-GATED) — port the instrument to a computed-state model

- **Substrate:** Othello-GPT (public weights + board probe if usable, else a small ~1080
  reproduction). The lit pass confirms which.
- **De-confound pre-check (ports directly):** board ≤-chance from input move-tokens;
  well-decodable from the residual stream. **Expected to pass — that *is* the Othello-GPT
  result;** failing it would itself be a finding.
- **The read.** Run the regime-2 body/shadow read on Othello only after the de-confound
  pre-check; run the determining-shadow-set / closure read only if a registered board
  functional passes the model-free linear-functional screen. The carried-over `u_null`
  control and selection-corrected null remain required; LEACE/amnesic remains a
  corroborating causal axis.
- **Model-free functional screen (new hard preflight):** before any model run, a candidate
  board functional must be linearly readable from the full board and support a non-vacuous
  `k_func << k_state` bracket. The named Othello slate failed this screen on 2,500 generated
  legal positions (`python scripts/deconfound_othello_datacheck.py --functionals all`):

  | functional | high-k det_func | high-k det_state | best lead | verdict |
  | --- | ---: | ---: | ---: | --- |
  | legal moves | -0.068 | 0.142 | -0.004 | blocked |
  | material parity | 0.134 | 0.152 | 0.007 | blocked |
  | frontier | -0.020 | 0.146 | 0.001 | blocked |
  | mobility | 0.161 | 0.143 | 0.018 | blocked |

  This is not a model null. It is a measurement-geometry block: the obvious real Othello
  functionals are nonlinear or too weak for the linear determining-set instrument, and none
  gives a closure lead over board-state reconstruction.
- **KILL-GATE:** (i) the de-confound must hold (board input-undecodable + rep-decodable),
  else the premise fails → repair/shelve; (ii) a non-vacuous read — a determining/closure
  structure or an honest null. Distinguish **`blocked_by_unavailable_substrate`** (can't get
  weights / a faithful repro) and **`blocked_by_ill_posed_functional`** (de-confound passes
  but no non-vacuous board functional can be registered) from a **scientific `shelve`**
  (clean read, no structure).

> **HONEST OPEN — the methodological crux.** The toy's clean `u` (hidden source) vs `z`
> (noisy state) split does **not** trivially port to a board: the board *is* the state, and
> the "functional" must be a *derived* quantity. Candidates to register: the board's
> **internal determining-set** (game-reachability couples squares → a few determine others),
> the **legal-move functional**, **side-to-move**, or a **board aggregate** ("material /
> who's ahead"). **Choosing the right decomposition is the sub-lane's core work** — it is
> the first thing the lit pass + Phase-0 spec must settle, and it is where the read could
> turn out ill-posed. Flagged, not hand-waved.

**Target-sharpening rule before any Phase-0 spec:** split Phase 0 into **0A**
(board de-confound pre-check) and **0B** (registered functional choice). A board-decoding
pass alone does **not** license a closure read. The spec must nominate exactly one primary
functional before measuring closure/body-shadow, with all alternatives fixed as report-only
or explicit fallback branches.

**Named Othello candidate disposition:** legal moves are still useful for a regime-2 output
shadow read, but legal moves, material parity, frontier, and mobility are rejected as
primary **linear closure** functionals. `side-to-move` still risks being
position/parity-decodable from the input stream. Board-square determining sets are promising
but higher-risk; treat them as a separate sub-lane only if the lit pass finds a canonical
linear reachability basis. Otherwise the closure read should pivot to **Attack B**.

## 7. Attack hierarchy (from `DECONFOUND_REAL_DATA_MEMO.md`)

- **A. Computed-state (Othello-GPT)** — primary for de-confound and possible regime-2;
  closure remains blocked unless a linear computed functional is found.
- **B. Semi-synthetic injection (SYNLABEL-style)** — primary for the closure read after
  the legal-move block; functional still ours (weaker on "more than we know").
- **C. Amnesic / causal (LEACE)** — corroborating "computes vs decodes" axis, not a
  replacement.

## 7.1 Attack-B Phase 0B — closure result

**Status:** executed 2026-06-04 -> `attack_b_closure_confirmed`.

Phase 0B routes the closure prize through the de-confound-clean real-feature
cell validated by the Attack-B 0-pre screen: median-binarized sklearn digit
features, `D=8`, `u = XOR(b_0,b_1,b_2)`, primary 0-pre `det=+0.077` (`HOLD`).
It trained paired bodies on the same real features:

- **state-keeper:** reconstruct all `b_j`, so it must keep state;
- **functional-keeper:** predict `u`, so it is free to discard `b_j notin S`.

The read landed as a clean double dissociation under the inherited
selection-corrected null and independent `u_null` control:

| body | `k_func(u)` | `k_state(b_j notin S)` | bracket |
| --- | ---: | ---: | :--: |
| functional-keeper | 2 | none | 5/5 |
| state-keeper | none | 4 | 0/5 |

Median paired `keeper_gap = 7`; `u_null = none` everywhere; all interpreted
selection-corrected p-values are `0.001`. The state-keeper's `k_state=4` proves
the probe can find outside features when the body carries them, so the
functional-keeper's `k_state=none` is genuine discard, not probe weakness.

This is the JEPA principle in clean supervised form on real features; it is
**not** a JEPA implementation and not "more than we know." Evidence ceiling:
**R1.5**.

Spec: `docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_SPEC.md`.
Results: `docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_RESULTS.md`.

## 8. Lit-pass targets (resolve before the Phase-0 spec)

1. **Othello-GPT specifics** — public weights + the linear board probe (Nanda's
   mine/theirs relative basis), reproducibility on the 1080, and the exact board
   representation + decodability numbers.
2. **The functional/state decomposition (§6 crux)** — the named Othello slate has failed
   the linear closure preflight. The lit pass may reopen Othello closure only by naming a
   new canonical functional that is both nontrivial and linearly readable from the full
   board. Otherwise Othello is limited to de-confound/regime-2 and closure Phase 0 pivots
   to Attack B.
3. **The de-confound numbers** — is "board ≤-chance from input move-tokens" measured
   anywhere, or do we measure it ourselves (cheap).
4. **Closest neighbours** — emergent-world-model interpretability (Othello-GPT, chess-GPT,
   tracking models); does a determining-shadow-set / closure read add to what's known, or
   restate it.

## 9. Promote-gate / tier

- **Phase 0:** **R2 down-payment** — real transformer, model-computed state, free
  de-confound. **External review before any promotion.**
- **"World model" / "understanding" / "planning":** R3 vocabulary, **forbidden** until R2
  clears and a registered theory gate is defined — *even though the field calls Othello-GPT's
  board a "world model,"* we report observability structure, not cognition.

## 10. Allowed / Forbidden language

**Allowed (Phase 0, on a clean read):**
> On a computed-state model (Othello-GPT), with the de-confound satisfied by task structure,
> the instrument measured *[a regime-2 separation / an honest null]* of the model-computed
> board. A closure claim requires a separately registered linear functional; legal moves do
> not qualify.

**Forbidden:**
- "The model understands Othello" / "has a world model" (as cognition).
- "This transfers to / explains real LLMs."
- any frontier-model or AGI claim.

## 11. Open question (aspiration, not claim)

The first real test of whether the shadow tells us about an agent's **self-built** structure
we did not plant — the honest form of "a sundial tells an agent more than we know." Tagged
*Normative*; it motivates the lane, it is not a result.

---

*Sundog Research Lab — SUNDOG_V_DECONFOUND scaffold. Internal; kill-gated R&D. No spec, run,
or public surface until a lit pass and a Phase-0 spec freeze.*
