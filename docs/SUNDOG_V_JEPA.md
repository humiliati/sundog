# Sundog vs. JEPA

Working hook:

> JEPA is built to keep the predictable and discard the rest. That *is* the closure.
> Does it leave a cleaner shadow?

Short version:

> JEPA (Joint Embedding Predictive Architecture) trains a representation to predict a
> *target embedding* and discard unpredictable detail — i.e., to keep the **functional**
> and throw away the **state**. That is exactly the closure relation the
> determining-shadow-set instrument now measures (Phase 2/7/7b/7c, calibrated). So the
> in-scope question is sharp and falsifiable: **does a JEPA body show the closure bracket
> (`k_func ≪ k_state`) more cleanly than a generative body, because it was trained to
> keep the functional and drop the state?** First test reuses the frozen coupled toy and
> the calibrated instrument — one objective swap. **Kill-gated R&D.**

**Status:** Scaffold opened 2026-06-04. No public page, `site-pages.json` entry, spec,
run, or external packet exists. **Gated on (a) a lit pass and (b) a Phase-0 spec freeze
before any run.** This is generality R&D, which the 2026-06-04 strategy pivot demoted —
so it carries an explicit **kill-gate** (below), not an open-ended program.

This is not a claim about AGI, world models, real LLMs, or whether LeCun is right. It is
a measurement question about what a JEPA-trained representation keeps and discards, asked
first on a synthetic toy.

## 1. Thesis — JEPA *trains* the closure; Sundog *measures* it

- A generative model must reconstruct its input (tokens/pixels), so it carries the
  **state**, noise included.
- A JEPA model predicts the **target's representation** and is free to discard whatever
  is unpredictable. The encoder is incentivised to represent only the **predictable
  functional** (the shared/coupling structure) and to drop the irreducible per-sample
  noise (predicting it only adds loss).
- In our vocabulary that is the closure relation as an *objective*: keep `u`
  (functional), discard `x_i` (state). The determining-shadow-set instrument measures
  exactly that split. **The lane points the calibrated sundial at the architecture
  engineered to cast the shadow it reads.**

## 2. Claim Boundary

This roadmap does **not** claim:

- that JEPA has / lacks a world model, or "understands" anything;
- that JEPA is better or worse than generative modelling;
- that any toy result transfers to real I-JEPA / V-JEPA / LLMs;
- that the instrument's reading is anything but a measurement on a designed substrate
  until the promote-gate (real architecture + external review) clears.

## 3. Why this fits Sundog

The instrument is built and **calibrated across the full closure spectrum** — it returns
honest negatives (Phase 2), partials (Phase 7), a confirmed positive (7b), and a located
boundary (7c). The coupled toy already has the closure built in with the de-confound
intact. So the cheapest, most rigorous JEPA question available is: **swap the objective,
re-read the body.** Nothing else in the portfolio can test JEPA's central claim
("representation-space prediction keeps the abstract, drops the noise") as a measured
quantity rather than a slogan.

## 4. The core question

> Does a JEPA-trained body show `k_func ≪ k_state` **more cleanly** than a
> generative-trained body on the same data — i.e., does JEPA represent the hidden
> functional `u` while *discarding* the per-latent state `x_i` that generative must keep?

## 5. Phase 0 (KILL-GATED) — JEPA vs generative on the coupled toy

- **Substrate:** the frozen Phase-7 coupled toy (`z_i = parity(u,A_i) ⊕ x_i`, closure
  built in, input-probe de-confound intact). No new substrate risk.
- **Two heads, same data, matched init/budget:**
  - **GEN** (baseline, already built): generative objective — must carry `x_i`.
  - **JEPA**: encoder + predictor + EMA/stop-grad target encoder; mask a subset of latent
    channels; predict the masked channels' **target embeddings** from the context
    channels; loss in representation space (no token reconstruction). Collapse-avoidance
    per the lit pass.
- **Read both bodies with the determining-shadow-set instrument** (`k_func` on `u`,
  `k_state` on omitted `z_j`, selection-corrected null, `u_null` control, UNLEARNED
  guard — all carried from the frozen probe).
- **Pre-registered prediction:** JEPA → `k_func` small (keeps `u`) and `k_state`
  large/none (discarded `x_i`); GEN → muddier (carries the state). The headline is the
  **JEPA−GEN closure difference**, not either alone.
- **KILL-GATE:** Phase 0 must show a pre-registered, control-clean **JEPA-vs-GEN closure
  difference** (effect size to be frozen in the spec — e.g. a `k_state` or body-state-carry
  gap beyond the seed spread). **If JEPA and GEN read the same on the toy, the "JEPA
  discards state" claim has no signal here → shelve the lane.** No scaling to real JEPA
  without a toy signal.

## 6. Phases beyond 0 (gated on Phase-0 pass AND the lit pass)

- **Phase 1 — small real task.** A small JEPA vs generative on a controlled but realistic
  task; the de-confound gets materially harder (see §7).
- **Phase 2 — R2, real architecture.** A *pretrained* I-JEPA / V-JEPA residual stream vs a
  generative LLM body, read by the instrument. **External review before any "JEPA
  learns / doesn't learn X" statement.** This is the rung where "the sundial reads
  structure we didn't design" could finally be earned — the encoder, not us, decides what
  to keep.

## 7. The de-confound wall — the make-or-break (primary lit-pass target)

On the toy the de-confound is solved (parity latents are input-undecodable by
construction). On **real** JEPA data it is not obvious what plays the role of the hidden
functional `u`, nor whether it is input-undecodable. **If the de-confound discipline
cannot port to real JEPA representations, the lane stalls at the toy** — and that is an
acceptable, bounded outcome, but it must be known *before* Phase 1. The lit pass decides
whether there is a real path past the toy.

## 8. Lit-pass targets (resolve before the Phase-0 spec)

1. **JEPA mechanics** — I-JEPA / V-JEPA encoder–predictor–EMA-target structure, masking
   schemes, and the exact collapse-avoidance that makes a faithful *small* JEPA trainable
   on the toy backbone (the 1080).
2. **What's already measured** — prior probing of JEPA representations: is "JEPA discards
   unpredictable detail / keeps abstract structure" an empirical result or a design claim?
   What methods exist, and does the determining-shadow-set read add anything?
3. **The de-confound-on-real-data problem (§7)** — any prior art on input-undecodable
   probing targets for self-supervised representations.
4. **Closest neighbours** — linear-probing of SSL features, RCDM / representation
   inversion, "what does a JEPA encoder throw away" studies; position the lane against
   them honestly.

## 9. Promote-gate / tier

- **Phase 0:** toy tier (R1-adjacent) — a measured JEPA-vs-GEN difference on a designed
  substrate. No real-model word.
- **Phase 2:** R2 (real architecture) — external review required.
- **"World model" / "understanding" / "path to AGI":** R3 vocabulary, **forbidden** until
  R2 clears and a registered theory gate is defined.

## 10. Allowed / Forbidden language

**Allowed (Phase 0, on pass):**
> On a designed coupled toy, the JEPA objective produced a *[cleaner / same / muddier]*
> closure body than the generative objective, measured by the determining-shadow-set
> instrument against frozen thresholds and a clean independent control.

**Forbidden:**
- "JEPA learns a world model" / "JEPA understands."
- "This proves LeCun right / wrong."
- "JEPA is the path to AGI / real intelligence."
- any statement about real JEPA / LLM behaviour drawn from the toy.

## 11. Open question (aspiration, not claim — for the spec/lit pass)

If a JEPA body is closure-clean (functional kept, state discarded), is it **more**
shadow-determined than a generative body (the body ≈ the functional, less hidden state to
resist) — or is the discarded state exactly the un-shadowed body that always resisted?
The instrument is the one tool that could separate those. Tagged *Normative*; it
motivates the lane, it is not a result.

---

*Sundog Research Lab — SUNDOG_V_JEPA scaffold. Internal; kill-gated R&D. No spec, run, or
public surface until a lit pass and a Phase-0 spec freeze.*
