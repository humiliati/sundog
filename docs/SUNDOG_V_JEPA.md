# Sundog vs. JEPA

Working hook:

> JEPA is built to keep the predictable and discard the rest. That *is* the closure.
> Does it leave a cleaner shadow?

Short version:

> JEPA (Joint Embedding Predictive Architecture) trains a representation to predict a
> *target embedding* and discard unpredictable detail — i.e., to keep the **functional**
> and throw away the **state**. That is exactly the closure relation the
> determining-shadow-set instrument now measures (Phase 2/7/7b/7c, calibrated). So the
> in-scope question is sharp and falsifiable: **does a JEPA body retain less private
> unpredictable noise than a generative body, while still keeping the predictable shared
> source?** First test reuses the frozen coupled toy and the calibrated instrument — one
> objective swap. **Kill-gated R&D.**

**Status:** Scaffold opened 2026-06-04. Phase-0 noise-carry spec drafted at
`docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md`, but the first smoke moved it to **DEBUG
HOLD**. JEPA code exists at `scripts/jepa_phase0_noise_carry.py`; no full battery, public
page, `site-pages.json` entry, or external packet exists. The repaired GEN positive-control
has now passed (`z_flip_acc=0.7839` on the banked Phase-7b GEN body); the remaining lock
blocker is a JEPA training/collapse smoke. This is generality R&D, which the 2026-06-04
strategy pivot demoted — so it carries an explicit **kill-gate** (below), not an open-ended
program.

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

> Does a JEPA-trained body carry less directly probe-able private noise `x_i` than a
> generative-trained body on the same coupled toy, while both bodies still retain the shared
> predictable source `u`?

The original closure-bracket sketch (`k_func` on `u`, `k_state` on `z_j`) is now demoted to
report-only. On the coupled toy, both objectives can recover the shared `u`, and
`k_state(z_j)` is dominated by private noise that is unrecoverable from other latents
regardless of objective. The discriminating read is therefore **noise-carry**, repaired as a
flip-conditioned noisy-bit read: train `body -> z_i` on all rows, then evaluate on held-out
noise flips `{x_i=1}`.

**Debug caveat (2026-06-04):** direct linear `det(x_i | body)` failed as a GEN positive-control
on the already banked Phase-7b GEN body: `u_det=0.754`, but local linear `noise_det=-0.010`.
The repaired flip-conditioned `z_i` read passes that GEN preflight (`z_flip_acc=0.7839`,
held-out flip counts 118-165 per latent). Phase 0 remains on DEBUG HOLD only because JEPA
training/collapse smoke has not passed yet.

## 5. Phase 0 (KILL-GATED) — JEPA vs generative on the coupled toy

- **Substrate:** the frozen Phase-7 coupled toy (`z_i = parity(u,A_i) ⊕ x_i`, closure
  built in, input-probe de-confound intact). No new substrate risk.
- **Two heads, same data, matched init/budget:**
  - **GEN** (baseline, already built): generative objective — must carry `x_i`.
  - **JEPA**: encoder + predictor + EMA/stop-grad target encoder; mask a subset of latent
    channels; predict the masked channels' **target embeddings** from the context
    channels; loss in representation space (no token reconstruction). Collapse-avoidance
    per the lit pass.
- **Read both bodies with the determining-shadow-set machinery**, but make the repaired
  noise-carry statistic primary: `z_flip_acc = median_i Acc(body -> z_i | heldout x_i=1)`.
  The probe is trained on all rows; only the scoring slice is conditioned on flips.
  `k_func(u)` / `k_state(z_j)` and direct `det(x_i | body)` remain report-only controls.
- **Pre-registered prediction:** GEN predicts observed noisy `z_i` on flips; JEPA denoises and
  fails/sub-chance-predicts `z_i` on flips while retaining `u`.
  The headline is the paired **GEN−JEPA noise-carry gap**, not either body alone.
- **KILL-GATE:** Phase 0 must show a pre-registered, control-clean **JEPA-vs-GEN
  noise-carry gap** that survives to the non-bottleneck `d_model=256` rung where chatv2's
  `objective_excess` deflated. If the gap exists only at low capacity, the verdict is
  `blocked_by_capacity`, not "JEPA works." No scaling to real JEPA without a toy signal.
- **Blocked-vs-shelved distinction:** if the lit-pass-faithful small JEPA collapses or
  cannot be trained on the toy backbone, Phase 0 is `blocked_by_unfaithful_jepa`, not a
  scientific null. The shelf verdict is reserved for a trainable, control-clean JEPA
  whose noise-carry read matches GEN inside the pre-registered seed spread.

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

This is a hard wall, not a delay. If the lit pass cannot name a faithful real-data
de-confound, the lane may still run Phase 0 as a toy-tier measurement, but it cannot use
that result as permission to speak about real JEPA representations.

**RESOLVED — partial crossing (2026-06-04, via Attack-B in `SUNDOG_V_DECONFOUND.md`).** The
de-confound wall *does* have a real-feature crossing — but only for a **linear, constructed**
functional: on real digit features the determining-shadow closure read confirms a
functional-vs-state dissociation (Attack-B 0B), and the input-de-confound is shown load-bearing
(0C). So **the Phase-0 question — does a functional-targeting objective keep the functional and
discard the state? — is now answerable on a *real* substrate**, not only the toy; Attack-B
already answered it for a *supervised* functional objective. The JEPA-specific step is the
natural fork: **swap the supervised functional-keeper for an actual JEPA objective on the same
Attack-B substrate.** What stays blocked is the *nonlinear / computed-state* functional (the
Othello slate failed because legal moves are nonlinear) — the "more than we know" direction a
constructed linear functional cannot reach.

## 8. Lit-pass targets (resolve before the Phase-0 spec)

1. **JEPA mechanics** — I-JEPA / V-JEPA encoder–predictor–EMA-target structure, masking
   schemes, and the exact collapse-avoidance that makes a faithful *small* JEPA trainable
   on the toy backbone (the 1080).
   - Sub-question: can Phase 0 reuse the coupled toy directly, or does it need a
     JEPA-compatible context/target view of the same toy? A new JEPA-friendly toy is not
     licensed unless the lit pass says the frozen toy cannot host a faithful JEPA read.
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
> On a designed coupled toy, the JEPA objective discarded per-sample private noise that the
> generative objective retained, while both bodies retained the predictable shared source.
> The effect survived the capacity rung where chatv2's prior objective-excess contrast
> deflated, and passed the collapse / null controls.

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

*Sundog Research Lab — SUNDOG_V_JEPA scaffold. Internal; kill-gated R&D. Phase-0 spec
drafted; no run or public surface until operator lock.*
