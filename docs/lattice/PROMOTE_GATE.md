# Lattice-Deduction Promote-Gate — the standard a result must clear before it becomes a claim

> 2026-06-02, DRAFT for ratification. Ported from the chatv2 promote-gate
> ([`../chatv2/PROMOTE_GATE.md`](../chatv2/PROMOTE_GATE.md)) — its "prize" — because
> this lane's substrate is **more** seductive, not less: an *exact*, *certified*,
> *architecturally-given* control shadow on a real-ish reasoner invites an even
> stronger theory of **why reasoning works**. This document exists to make that
> promotion **hard and explicit** *before* any 20+h reimplementation run lands a
> `CERTIFIED_SHARP` in the moment when discipline is hardest to apply.
>
> Standing rule (binding once ratified): **no rung's language may be used until
> that rung's gate is cleared.** Until then the lane speaks only in the language of
> the rung it has actually reached. A result is a *receipt* until promoted.

## 0. Why a gate, not a vibe (the lattice-specific seduction)

The chatv2 failure mode — reading a *toy measurement* as a *mechanism of
intelligence* — is sharper here because two words do rhetorical work the
measurement may not earn:

1. **"Certified."** Only the **state-insufficiency leg** is certified — the fiber
   `γ(a)` is non-trivial *by construction* (Cousot & Cousot 1977; **Target A
   folklore**, [`LITPASS_MEMO.md`](LITPASS_MEMO.md) Track E). The
   control-sufficiency-with-soundness leg and the high-dim-body leg are **measured
   on one reimplemented model** and can fail (`UNSOUND`, `CERTIFIED_MARGINAL_BODY`).
   "Certified" describes the fiber, **not** the result.
2. **"Exact."** The fiber is exact; the *result* is not a proof. An exact fiber with
   a marginal neural body is `CERTIFIED_MARGINAL_BODY` — the **expected** outcome,
   not an exact regime-2 witness.
3. The shadow leg is **architecturally given**, so there is nothing to "discover" —
   the only open empirical question is whether the *learned body* is high-dim.

A theory of why reasoning works must **predict something novel and survive a test**.
A framework that *fits* a Sudoku reasoner predicts nothing. The gate keeps "fits our
vocabulary" and "explains reasoning" from collapsing into each other.

## 1. The claim ladder (where the lane can be, and how far the theory is)

| rung | claim form | gating phase |
| --- | --- | --- |
| **R0 — Receipt** | "On *this* reimplemented LDT (build-gate pinned) we measured *these* fiber-size / false-elimination / `d_dec` / `k_control` / cross-decode numbers." | Phase 1 + Phase 2 |
| **R1 — Reimplemented-substrate claim** | "The reimplemented LDT realizes a certified state-insufficient, control-sufficient regime-2, with verdict `CERTIFIED_SHARP` / `CERTIFIED_MARGINAL_BODY` / `UNSOUND`." | Phase 2 (+ B1/B3) |
| **R2 — Substrate-general / real-model claim** | "This holds **beyond one reimplementation** — across the LDT family / seeds, or on the authors' released model — and survives external refutation." | Phase 5 + Phase 7 |
| **R3 — Theory claim** | "Exact computational regime-2 **explains** [decision-vs-reconstruction structure of sound reasoning / why interpretable-bottleneck reasoners generalize]." | **FAR. Not on the near horizon.** |

The temptation is to speak at R3 from an R0/R1 receipt. The distance R1→R3 is the
content of this document.

## 2. Per-rung promotion gates (pre-registered)

**R0 — Receipt.** Earned when a Phase-2 run files a manifest + runner/dirty-git
provenance + the frozen B2 metrics. A receipt is **not** a claim — it is the input
to R1.

**R1 — Reimplemented-substrate claim.** Promote from R0 only when *all*:
- **build-gate passed** — the reimplementation reproduced 100% Sudoku-Extreme
  (`build_gate_pass`); no B-layer number is read off a model that missed the bar;
- **measured on the learned model, not Target A** — every leg except the certified
  fiber is a measurement of the trained network (false-elimination rate, `d_dec`),
  not the definitional α/γ pair;
- **cross-decode guard clean** — `g*` is **not** decodable from the body beyond the
  fiber-uncertainty floor (NeuroSAT caution, litpass Track C); a positive flags a
  leak or the "secretly solved" failure mode, not a result;
- **vacuity guards clear** — fiber is genuinely non-singleton (not the trivial
  "puzzle unfinished"); soundness is not achieved by abstention (the B3 progress
  floor);
- **generality across ≥ 2 tasks** — the verdict survives on **Sudoku-Extreme AND**
  (Snowflake or Maze), pre-registered — not one task, one seed;
- **the verdict is reported honestly** — `CERTIFIED_MARGINAL_BODY` and `UNSOUND` are
  *the result*, not failures to be reframed; R1 does **not** upgrade a marginal body
  to "sharp," and the high-dim bar (`d_dec ≥ 16`, `k_control/d_dec ≤ 0.25`, PHASE0
  §5) is reported met-or-unmet, never softened.

**R2 — Substrate-general / real-model claim.** Promote from R1 only when *all*:
- the verdict **replicates** across multiple reimplementations / seeds, or on the
  authors' model **if released** (with an access/fidelity audit) — one reimplemented
  model is **not** "the LDT";
- **external review** signs that (a) the α/γ → projection/fiber coupling is not
  mere vocabulary, (b) the twin-state certifies state-insufficiency in the claimed
  sense, (c) the body measurement is not an artifact — the portfolio
  external-review-packet pattern (Phase 7), **plus the chatv2 audit team invited to
  refute**;
- the alternative **"this is just sound abstract interpretation restated"** (Target
  A) is ruled out *on the measured model* — the result must turn on the *learned*
  representation, not the definitional operator;
- the result is **not reproducible by a simpler account** (a standard CP solver's
  candidate-set abstraction; a linear probe of the lattice input).

**R3 — Theory claim.** Promote from R2 only when *all*:
- the framework makes a **novel, falsifiable, quantitative prediction** about a
  reasoner's behavior that was **not** used to build it, and it is **tested and
  confirmed**, with a pre-registered falsifying version on record;
- it is **not** reproducible by an incumbent (abstract-interpretation theory; the
  HRM/TRM recurrent-reasoning literature; mechanistic-interpretability probing; the
  world-model / platonic-representation accounts);
- it survives review by named external researchers invited to **refute** it.
- Until every one holds, the words "explains reasoning", "world model",
  "understanding", "mechanism of reasoning" are **not licensed**.

## 3. The do-not-claim ledger (ground each temptation here)

| the satisfying theory | why it is NOT yet licensed (the grounding) |
| --- | --- |
| "The LDT **proves** Sundog regime-2 — it's **certified**." | Only the *fiber* (state-insufficiency leg) is certified, by construction (Cousot folklore, Target A). The sound-control and high-dim-body legs are measured and can fail. "Certified" describes the fiber, not the result. |
| "Sundog **explains the LDT** / explains neural reasoning." | The LDT's mechanism is sound abstract interpretation (Cousot 1977). Sundog measured a body/shadow *reading* of one reimplemented model. No theory of reasoning is established — **R3, far**. |
| "We **found the control-sufficient shadow** — the first sharp control substrate." | The shadow is **architecturally given**, not discovered. The open question is the *learned body*, whose honest prior is `CERTIFIED_MARGINAL_BODY` (collapse) — **not** a sharp control substrate. |
| "An **exact** regime-2 — better than the marginal substrates / AB." | Exactness of the *fiber* is not exactness of the *result*. Exact fiber + marginal body = `CERTIFIED_MARGINAL_BODY`, not an exact witness. AB's exactness is on the *topological* axis (a different corner); do not conflate. |
| "The reimplemented-model result is a claim about **the LDT paper**." | It measures *our* reimplementation; it neither validates nor refutes the authors' unreleased model (`LATTICE-REIMPL-LAUNDERING`). |

## 4. Adversarial pre-mortem (how a promotion would be *wrong*)

Before any rung, write the obituary — *"We promoted, and we were wrong, because…"*:
- **The fiber was trivial.** We sampled near-singleton lattices, so "certified
  state-insufficiency" was the vacuous "puzzle unfinished" (PHASE0 F5).
- **The body carried the answer.** The model secretly decoded `g*` (NeuroSAT-style)
  and we read "bounded sound deduction" anyway (cross-decode guard).
- **The reimplementation was unfaithful.** The build-gate slipped and every number
  is uninterpretable.
- **"Certified" did the work the measurement didn't.** The word carried a claim the
  body-dimensionality leg never earned (the sharpest word-capture risk here).
- **Single model / single task.** One reimplementation, one task, one seed; none of
  the variance that matters was sampled.
Each must be actively *attacked* (vacuity check / cross-decode / build-gate /
generality run / audit-team review), not assumed away, before its rung promotes.

## 5. External grounding (no self-promotion past R1)

R2 and R3 require the Phase-7 external-review packet (reading-only ask: is the
coupling vocabulary? does the twin-state certify what it claims? is the body
measure an artifact? is it reproducible by a simpler account?) sent to the
[`LITPASS_MEMO.md`](LITPASS_MEMO.md) reviewer shortlist **and** the **chatv2 audit
team invited to refute**. The lane may not promote past R1 on internal conviction —
internal conviction is exactly what this gate distrusts.

## 6. How to use it

Before writing any sentence about lattice in a memo, page, ledger, or conversation:
1. Identify the rung the sentence speaks at.
2. Confirm that rung's §2 gate is cleared. If not, **rewrite the sentence down to
   the rung actually reached** — usually R0 ("on one reimplemented LDT we measured
   …") or, post-Phase-2, an honest R1 verdict (most likely
   "`CERTIFIED_MARGINAL_BODY`: the regime-2 is certified at the abstraction, the
   learned body is low-dimensional").
3. If the sentence is in §3, it is a known temptation — replace it with its grounding.

The reward is real: if the lattice lane *does* reach `CERTIFIED_SHARP` and clears R2
with external review, the claim will be **believable**, precisely because the lane
refused to make it early.

## 7. Cross-references

- [`../chatv2/PROMOTE_GATE.md`](../chatv2/PROMOTE_GATE.md) — the source gate this
  ports; the R0→R3 discipline.
- [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) — §7 points here for the gate;
  §Phase 5/8 for the promotion ladder by verdict.
- [`PHASE0_MINIMUM_FALSIFIABLE.md`](PHASE0_MINIMUM_FALSIFIABLE.md) — the frozen
  B2 verdicts + thresholds the R1 gate references.
- [`LITPASS_MEMO.md`](LITPASS_MEMO.md) — Target-A fence (Cousot), the cross-decode
  caution (NeuroSAT), the reviewer shortlist for R2/R3 grounding.
