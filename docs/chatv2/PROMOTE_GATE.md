# Chat-v2 Promote-Gate — the standard a result must clear before it becomes a claim

> 2026-06-01, DRAFT for ratification. Written *because* the Phase-1 results are
> seductive: a de-confounded, seed-robust, objective-driven "body that resists its
> shadow" invites a satisfying theory of **why AI works**. This document exists to
> make that promotion **hard and explicit** — to ground every tempting claim
> against a pre-registered bar before it is spoken, written, or pushed on.
>
> Standing rule (binding once ratified): **no rung's language may be used until
> that rung's gate is cleared.** Until then the lane speaks only in the language of
> the rung it has actually reached. A result is a *receipt* until promoted.

## 0. Why a gate, not a vibe

The failure mode is not bad data — the data is good. The failure mode is
**reading a toy measurement as a mechanism of intelligence.** Three temptations
make it acute here:

1. The vocabulary (*body*, *shadow*, *resistance*, *control-sufficient*) is
   evocative and pre-loaded with meaning it has not yet earned on this substrate.
2. The result is *robust* (seed-stable, decomposed) — and robustness of a *toy*
   measurement feels like robustness of a *truth*.
3. "Generative training builds more than the decision needs" pattern-matches onto
   real intuitions about LLMs — so the toy seems to *confirm* them.

A theory of why AI works must **predict something novel and then survive a test**.
A framework that *fits* a toy result predicts nothing. The gate's job is to keep
"fits our vocabulary" and "explains intelligence" from collapsing into each other.

## 1. The claim ladder (where we are, and how far the theory is)

| rung | claim form | current status |
| --- | --- | --- |
| **R0 — Receipt** | "On *this* synthetic toy with *this* fingerprint we measured *these* numbers." | **DONE** (`PHASE1_SEEDSTAB_RECEIPT.md`) |
| **R1 — Toy-substrate claim** | "Computed-latent transformers exhibit a robust, de-confounded, objective-driven body-resistance." | **MET 2026-06-29** — all §2 gates cleared (`PHASE1_R1_COMPLETION.md`). Scope: parity-family latents, toy from-scratch transformers, `d_dec<20`; pair-XOR to H=8/`d_dec≈7`, 3-parity to H≈2–4. |
| **R2 — Real-substrate claim** | "*Real pretrained* LLM residual streams exhibit this on a *non-synthetic* task." | **NOT STARTED** |
| **R3 — Theory of AI** | "This *explains* [generalization / capability / representation / why scaling works]." | **FAR. Not on the near horizon.** |

The temptation is to speak at R3 from an R0/R1 result. The distance R1→R3 is the
entire content of this document.

## 2. Per-rung promotion gates (pre-registered)

**R1 — Toy-substrate claim.** Promote from R0 only when *all*:
- seed-stability shown (≥2/3 frozen seeds, same branch) — **met**;
- contrast reported continuously + decomposed against the architectural baseline
  (not a binary at-the-bar gate) — **met** (`objective_excess 0.205±0.022`);
- **generality across the toy family** — the effect survives ≥2 *latent
  computations* (not just pair-XOR) and ≥2 *architectures* (depth/width/heads),
  pre-registered — **met 2026-06-29**: architectures A1 (d192) + A2 (d128) both SHARP;
  latent computations pair-XOR (SHARP to H=8) + 3-bit parity (SHARP at H=2, excess 0.20;
  it grok-walls **UNLEARNED** at H=8, a *learnability* ceiling, not a resistance negative
  — so shown at lower `H`, honestly scoped);
- the high-dimensional bar is honestly reported as **unmet** (`d_dec≈7.6 < 20`) —
  met, and R1 does **not** claim the high-dim regime.
- Falsifiers to rule out: the result is an artifact of (a) the fair-readout choice,
  (b) δ=0.45 making the task near-trivial, (c) one optimizer/schedule — **all cleared
  2026-06-29** (F-readout PASS 0.205/0.245; F-δ=0.30 excess 0.181; F-opt lr=1e-3
  3-seed mean excess 0.137 ≥ 0.10).

**R2 — Real-substrate claim.** Promote from R1 only when *all*:
- the fingerprint is ported to a **real pretrained LLM** on a **real task** with
  the same de-confounds (an input-leakage pre-check analogue, an information-basis
  dimensionality, a genuine objective contrast) — pre-registered;
- the body-resistance clears a **real high-dimensional bar** (`d_dec` ≫ the
  marginal substrates, target ≥20) on that substrate;
- **external mech-interp review** signs the measurement is not an artifact and the
  framing is not a category error (the portfolio's external-review-packet pattern —
  cf. NSE / Yang-Mills / Riemann / Hodge packets);
- the alternative explanation "this is just probing pre-existing input structure"
  is ruled out on the real substrate.

**R3 — Theory of AI.** Promote from R2 only when *all*:
- the framework makes a **novel, falsifiable, quantitative prediction** about a
  real model's behavior or capability that was **not** used to build it;
- that prediction is **tested and confirmed**, and a pre-registered version that
  would have **falsified** it is on record;
- the prediction is **not** reproducible by a simpler existing account (linear
  probing, superposition, the platonic-representation / world-model literature) —
  i.e., the theory earns its keep over the incumbents;
- it survives review by named external ML/interpretability researchers who are
  invited to **refute** it.
- Until every one of these holds, the words "why AI works", "world model",
  "understanding", "mechanism of intelligence" are **not licensed**.

## 3. The do-not-claim ledger (ground each temptation here)

| the satisfying theory | why it is NOT yet licensed (the grounding) |
| --- | --- |
| "AI builds rich internal **world models** — the body genuinely resists." | The latents are XOR-of-bit-pairs with **no semantics**; `d_dec≈7.6` is a handful of independent synthetic bits, not a world; one toy architecture; and ~0.09 of the "resistance" is **architectural** (an *untrained* net reads them via random features). A few decodable synthetic bits ≠ a world model. |
| "**Generative training is special** / uniquely builds capability." | `objective_excess = 0.205` is *modest*, on a toy where the control objective genuinely never needs the other latents — a clean separation the real generative/discriminative distinction does not have. Modest excess on a rigged-easy contrast is not "special." |
| "The **Sundog regime-2 is confirmed for AI**." | Confirmed for a **toy** at `d_dec≈7.6 < 20`, fair-readout-dependent, one seed-config family. "For AI" is **R2** — a real LLM, not started. |
| "This **explains generalization / scaling / emergence**." | **Zero** link established: the toy has no generalization split, no scaling law, no emergent capability. This is vocabulary pattern-matching, not explanation. **R3**, far off. |
| "The **body/shadow framework is the right theory of representation**." | It is a *measurement* that *fit* a toy. It has predicted **nothing novel** about a real model. A framework that fits is not a theory that predicts (§0). |

## 4. Adversarial pre-mortem (how a promotion would be *wrong*)

Before any rung, write the obituary: *"We promoted, and we were wrong, because…"*
The standing candidates for this result:
- **The toy was too easy.** δ=0.45 + 2-bit XOR may sit in a regime where the
  effect is generic and says nothing about hard computation.
- **The architectural floor is the real story.** Random features already read the
  latents at 0.59; the "objective-driven" 0.205 may shrink or vanish on substrates
  where the architectural baseline is different.
- **Single everything.** One architecture, one optimizer, one latent computation,
  one `H`. None of the variance that matters has been sampled.
- **Vocabulary capture.** The framing makes any positive result *feel* like the
  theory, because the theory was used to design the measurement.
Each must be actively *attacked* (ablation / generality run / red review), not
assumed away, before its rung promotes.

## 5. External grounding (no self-promotion past R1)

R2 and R3 require an external-review packet in the portfolio pattern (reading-only
ask: is the measurement an artifact? is the framing a category error? is the
claim reproducible by a simpler account?), sent to named ML/interpretability
reviewers **invited to refute**. The lane may not promote itself past R1 on
internal conviction alone — internal conviction is exactly what this gate distrusts.

## 6. How to use it

Before writing any sentence about chatv2 in a memo, page, ledger, or conversation:
1. Identify the rung the sentence is speaking at.
2. Confirm that rung's §2 gate is cleared. If not, **rewrite the sentence down to
   the rung we have actually reached** (usually R1-partial: "on a synthetic toy,
   the de-confounded fingerprint shows a robust, objective-driven, *below-bar*
   body-resistance").
3. If the sentence appears in §3, it is a known temptation — replace it with its
   grounding.

The reward for the discipline is real: when chatv2 *does* clear R2 with external
review, the claim will be **believable**, precisely because the lane refused to
make it early.
