# Chat-v2 Phase 0.2 — Computed-Latent Cell (pre-registration)

> 2026-05-30, **draft / not yet run**. Phase 0 (Amendment 1) overturned the
> variance-masking artifact and gave the lane's first positive body-resistance
> read — but with two issues it also surfaced: (1) a **passive-decodability
> confound** (the latents were simple input functions, so any transformer's
> residual stream carried them — high-dimensionality partly "free", resistance
> partly trivial), and (2) a **mis-specified contrast metric** (`d_dec` measures
> rank, not representation strength, so the objective-driven contrast was
> invisible to the verdict). Phase 0.2 fixes both and re-poses the gate so a
> SHARP result would be non-trivial.

## 0. What Phase 0.2 must do

Make the body-resistance question *non-trivial*: the decision-relevant state must
be something the model has to **compute**, not echo from the input. Then a clean
SHARP requires that (a) the generative body carries a high-dimensional *computed*
state, (b) the decision-shadow is state-insufficient for the rest, and (c) the
**control-only twin fails to build the non-decision state** (it had no reason
to) — the objective-driven mechanism, now measured on the right axis.

## 1. Fix A — computed latents (kill the passive floor)

Replace the linear-aggregate latents (per-channel emission bias, linearly
decodable from the input) with latents that are **nonlinear functions of the
input, not linearly decodable from it.** The defining property, *verified before
any verdict run* (§4 pre-check): a linear probe on the **raw input bits**
recovers each latent at **≈ chance**. Then any linear decodability of the latent
from the residual stream reflects **active computation by the model**, not a
passive echo — so the twin's non-decision floor drops to chance and the contrast
is clean.

**Frozen realization (2026-05-30) — pair-XOR latents.** Each channel `i` carries
a per-sequence latent `z_i ∈ {0,1}` and emits `P = bits_per_channel/2` independent
**bit-pairs** `(u, v)`: `u` is fair, and the pair's XOR `x = u ⊕ v` is biased by
`z_i` (`x ~ Bernoulli(0.5 ± δ)`). Because `u` is fair and `v = u ⊕ x`, **neither
`u` nor `v` alone correlates with `z_i`** (`Cov(u,z)=Cov(v,z)=0`) — `z_i` lives
only in the *pair XOR*, a nonlinearity no linear input-probe can read. Predicting
`v` (given `u` and the inferred `z_i`) forces the model to compute `XOR(u,v)` and
aggregate, so `z_i` is a **maintained per-sequence latent present at the final
position** (the structure that worked in Phase 0). Decision = `z_1`; non-decision
state = `{z_2…z_H}`.

> **Design pivot — the pre-check earned its place.** The first realization
> (channel bit biased by the parity of its last `window` bits) **failed §4's
> pre-check at 0.85**: biasing each bit toward the running parity makes the
> stream an LFSR-like GF(2) recurrence (`bitₜ ≈ bitₜ₋₁ ⊕ bitₜ₋₂`), which
> *collapses* the window-parity back to a single earlier bit — linearly
> decodable. The pair-XOR design has no recurrence to collapse and is provably
> XOR-only; pre-check then passed at **0.498–0.516 ≈ chance** across all `H`.
> The de-confound gate caught a bad substrate before any training compute.
> Pair-XOR is also *local and learnable* (the smoke learned it in 400 steps), so
> the D5 capacity bump is held in reserve, not spent.

## 2. Fix B — strength-based contrast criterion (measure the right axis)

`d_dec` (read-out rank) is retained as the un-masked dimensionality measure, but
the **objective contrast** is re-posed on representation *strength*:

- `body_carry` = mean `z_recover` over the **non-decision** latents
  (`z_2…z_H`) — how strongly the body represents the rest of the state.
- The contrast is `body_carry_gen` vs `body_carry_twin`. With computed latents
  the twin's non-decision floor is ≈ chance, so a real objective effect shows as
  `body_carry_gen ≫ body_carry_twin ≈ 0.5`.

## 3. The fingerprint (per `H`, gen + twin)

Carried over from Amendment 1 (un-masked, information-basis), plus the
de-confound pre-check:

- `d_dec` — decodable dimensionality (read-out-direction effective rank).
- `body_carry` — mean `z_recover` over `z_2…z_H` (the new contrast axis).
- `z1_acc` — control-sufficiency of the decision.
- `cross_latent_leak` — z₁ shadow → other latents; ≈ chance ⇒ resists.
- `outlier_carries / survives` — the three-way medium test (sundog / atmosphere
  / weather), retained — owner's hypothesis, now on a computed substrate.
- `eff_dim_raw` / `eff_dim_robust` — kept to keep watching the masking.

## 4. Mandatory de-confound pre-check (gates the whole run)

Before computing any verdict, train a **linear probe on the raw input bits** to
predict each latent. Required: **mean input-probe accuracy ≈ 0.5** (chance). If
the latents are still linearly input-decodable (`> ~0.6`), Fix A failed — the
substrate is re-picked and re-registered, **no verdict is read.** (This is the
check that the original toy would have failed — its bias-latents were ~0.85
input-decodable.)

## 5. Revised sharpness verdict (pre-registered)

For the generative model, per `H`, **SHARP** iff *all*:
- `d_dec ≥ H/2` (high-dim computed state), **and**
- `z1_acc ≥ 0.70` (control-sufficient), **and**
- `cross_latent_leak ≤ 0.58` (state-insufficient ⇒ resists), **and**
- `body_carry_gen ≥ 0.70` **and** `body_carry_gen − body_carry_twin ≥ 0.20`
  (the body carries the non-decision state, and **the objective built it** —
  the twin did not).

`H*` = smallest passing `H`. **MARGINAL** if no `H` passes — and with the passive
floor removed, a MARGINAL here is a *clean* negative (the model didn't build a
high-dim resisting state even under generative pressure), not an artifact.

## 5b. Design decisions for sign-off

- **D4 — latent computation.** Default: parity-state channels (§1). Fallbacks if
  parity won't learn (§6): (a) **2-input XOR** of two designated bits per channel
  (minimal nonlinearity, easier); (b) **running-sum mod 3** threshold (nonlinear,
  often easier to learn than parity); (c) **AND/OR composition** of two
  sub-channel states. Pick the *easiest* realization that still passes the §4
  chance pre-check.
- **D5 — capacity / budget.** Parity-style tasks can need more steps/width than
  the bias task (grokking). Default bump: `max_steps` 2500 → 4000, `d_model`
  128 → 192; revisit if `gen` underfits (F3′).
- **D6 — `H` sweep.** Keep `{1,2,4,8,16}` (compute permitting; the §6 cost note).

## 6. Named failure modes (Phase 0.2)

- **F1′ — clean marginal.** Even generative training yields low `d_dec` or
  `leak` above chance. With no passive floor, this is a *real* negative: the
  substrate doesn't support a high-dim resisting computed state. (Honest, and
  the point of the cell.)
- **F2′ — de-confound failed.** Input-probe pre-check (§4) > chance ⇒ the
  "computed" latent is still input-linear. Re-pick the computation, re-register;
  no verdict.
- **F3′ — model can't compute.** `gen` `z_recover` is low *and* `eval_loss` is
  far from the Bayes floor ⇒ the model didn't learn the parity (capacity /
  steps), not a body-resistance result. Bump D5, distinguish from F1′ via the
  loss gap.
- **F4′ — twin learns it anyway.** `body_carry_twin` ≈ `body_carry_gen` despite
  the chance input-floor ⇒ the backbone computes the non-decision latents even
  unsupervised (e.g. they're forced by the generative-shaped *data* the twin
  also sees). Informative; would mean the contrast needs an even stricter
  control (e.g. twin trained on shuffled non-decision channels).

## 7. Cost / build / reuse

- **Reuse:** the Amendment-1 harness (`scripts/chatv2_phase0_bodyresist.py`) —
  swap `gen_batch` for the computed-latent generator, add the §4 input-probe
  pre-check, add `body_carry` + the revised verdict. Decoupled train/measure and
  body-saving already in place.
- **Compute:** likely **longer than Amendment 1** (parity is harder; D5 bump +
  the H=16/L=256 tail). Plan a clean CPU window or accept multi-hour; bodies are
  saved so the measure iterates for free. Run a smoke + the §4 pre-check first —
  if the pre-check fails, no expensive train.

## 8. What Phase 0.2 still does NOT do

Same boundary as Phase 0: no real LLM, no ledger, no multi-turn, no public
surface, no promotion. It hardens the *toy* gate; a positive result earns the
move to a real-model substrate (Phase 1), still review-gated.

## Result (2026-05-30) — first de-confounded SHARP (low-`H`); high-`H` blocked by a learnability wall

Ran `--mode full --stage all --latent computed` (pair-XOR; ~3.5 h;
`results/chatv2/phase02-full/`). The §4 pre-check passed at every `H` (input-probe
0.498–0.516 ≈ chance). **Verdict: `SHARP`, `H* = 2`** — the portfolio's first
de-confounded body-resistance read.

| `H` | learned? (`eval_loss` vs 0.693 = chance) | `d_dec` | `leak` | `body_carry` gen / twin | status |
| --- | --- | --- | --- | --- | --- |
| 2 | **yes** (0.599) | 2.0 / 2 | 0.48 | **0.97 / 0.65** | **SHARP** |
| 4 | **yes** (0.608) | 3.9 / 4 | 0.54 | **0.84 / 0.57** | **SHARP** |
| 8 | **no** (0.693 = log 2) | 7.0\* | 0.51 | 0.52 / 0.52 | UNLEARNED |
| 16 | **no** (0.693 = log 2) | 12.6\* | 0.51 | 0.51 / 0.50 | UNLEARNED |

**Where it is genuinely SHARP (`H = 2, 4`).** The generative model learns the
*computed* latent (`eval_loss` below the chance line, toward the 0.558 floor); the
body is full-rank (`d_dec` 2.0 / 3.9); it **resists** (`leak ≈ chance` — the z₁
shadow is state-insufficient for the rest); and the objective-driven contrast is
clean and strong (`body_carry` gen 0.97 / 0.84 vs twin 0.65 / 0.57). The **twin
floor near chance** is the de-confound working: control-only training does not
build the non-decision state, where in the Phase-0 (bias) toy it passively did
(~0.85 floor). This is the state-insufficient-yet-control-sufficient split,
objective-driven, on a non-trivial substrate — the thing the four prior
substrates could not show.

**Honest catch (`H = 8, 16`) — undertrained, NOT marginal.** `eval_loss = 0.693
= log 2 = pure chance`: the d=128 / 2500-step model failed to learn the pair-XOR
aggregation at high `H` (even z₁ alone collapses to 0.53), early-stopping because
the loss never left random. These are **F3′ (a learnability wall), uninformative**
— and the `d_dec` 7.0 / 12.6 there (\*) is **noise-rank** (effective rank of
chance-level read-out directions), **not** a high-dim body; it is not cited as one.

**Disposition.** Body-resistance is **proven, de-confounded, at low
dimensionality (2–4 dims)** — a genuine first for the portfolio — but the
genuinely high-dimensional resisting body (the real target) remains **unproven,
blocked by learnability, not refuted.** Not promotable (low-dim only; one toy).
→ **focused `H=8` probe** with the reserved D5 capacity bump (`d_model`→192,
`max_steps`→6000) + a *signal* bump (larger δ / more pairs so the same nonlinear
latent is easier to estimate; the XOR stays input-undecodable, so the de-confound
holds), and a curriculum (warm-start from saved low-`H` weights) if it still
grok-cliffs. **Methodology fix landed:** an `H` whose `eval_loss ≈ chance` is now
flagged **UNLEARNED**, never silently MARGINAL — "couldn't train" and "no
resistance" must not be conflated.

## 9. Cross-references

- [`PHASE0_MINIMUM_FALSIFIABLE.md`](PHASE0_MINIMUM_FALSIFIABLE.md) — Phase 0 + Amendment 1 (the result this refines).
- [`LANE_CHARTER.md`](LANE_CHARTER.md) — lane mission + the three-marginal mandate.
- [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) — body-resistance axis; chatv2 is the first non-flatly-marginal entry.
