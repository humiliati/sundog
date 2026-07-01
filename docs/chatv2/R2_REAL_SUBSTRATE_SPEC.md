# Chat-v2 R2 — Real-Substrate Cell (pre-registration)

> 2026-06-29, **draft for sign-off, not yet run.** R1 is MET (`PROMOTE_GATE.md`,
> `PHASE1_R1_COMPLETION.md`): computed-latent *toy* transformers show a robust,
> de-confounded, objective-driven body-resistance (parity-family, `d_dec<20`). R2 asks
> the gate's next question: **does a *real pretrained* LLM's residual stream show the same
> — state-insufficient yet control-sufficient — on a *non-synthetic* task, above the
> `d_dec≥20` high-dimensional bar the three physical substrates never cleared?**
>
> **Forks resolved (owner, 2026-06-29):** substrate = **GPT-2 small** (pretrained, CPU
> inference); objective contrast = **random-init architectural floor** (the twin
> substitute — a pretrained LLM has no control-only twin); shadow = **data-driven
> control-subspace** (Phase-0 style; the maintained-ledger shadow is deferred).

> **Fences (binding).** This is **R2, not R3.** A SHARP result would say a real LLM
> carries more in its residual stream than a narrow decision needs — **not** that it has a
> "world model" or that this "explains why AI works" (`PROMOTE_GATE.md` §3). No public
> surface; the NSE page's "resistant substrate" line stays "to be run" until R2 clears
> **including external review**.

## 0. What R2 must do (the gate, `PROMOTE_GATE.md` §2)

1. port the fingerprint to a **real pretrained LLM on a real task** with the same
   de-confounds (input-leakage pre-check, information-basis dimensionality, a genuine
   objective contrast);
2. clear a **real high-dimensional bar** `d_dec ≥ 20` (≫ NSE ~18 / Mesa ~2 / shell ~1.7);
3. **external mech-interp review** signs it is not an artifact / not a category error;
4. rule out **"just probing pre-existing input structure."**

## 1. Substrate

- **Body:** GPT-2 small (124M, 12 layers, d=768) residual-stream activations at the
  final token, extracted per passage. **Pretrained weights, CPU, inference only (no
  gradients).** Compute the fingerprint at every layer; report the layer with max `d_dec`
  as the body (Phase-0 D3 pattern).
- **Objective-contrast baseline (the twin substitute):** a **random-init GPT-2 small**
  (identical architecture, untrained) — the architectural / random-feature floor. The
  claim is `body_carry_pretrained ≫ body_carry_random` (pretraining *built* the
  non-decision state; random features didn't). This directly attacks the toy's own
  honest caveat (the ~0.59 random-feature floor), now the load-bearing contrast.
- **Shadow:** the `k_control`-dim data-driven subspace that reads the decision at ≈ full
  accuracy (accuracy-vs-`k` saturation sweep, Phase-0).

## 2. Task + latents — a FROZEN BANK of count-parities (input-undecodable)

Real text passages (a fixed public corpus slice, frozen split; fixed token length). The
**latent for attribute `A` = parity of the count of `A`-matches in the passage (mod 2)** —
the real-token analogue of the toy's XOR-parity: a *parity* is nonlinear, so a linear
raw-token probe **cannot** read it (the §4 de-confound is passed by the *construction* of
parity, not by luck), yet the model must integrate over the whole passage to compute it.

**Frozen attribute bank (~48–64, pinned in `scripts/chatv2_r2_real_substrate.py` before the
run; deterministic regex / lexicon only — reproducible, no NER/POS dependency):**
- **morphology:** `-ed -ing -ly -s(plural) -er -est -tion -ment -ness -able -ful -less
  -ize un- re- -ity -al -ic -ous -ive` …
- **function words (lexicon sets):** negators, modals, auxiliaries, wh-words,
  personal/possessive pronouns, demonstratives, connectives, prepositions, determiners,
  quantifiers, intensifiers;
- **entity/format proxies:** titlecase spans, all-caps tokens, digit tokens, number words,
  hyphenated tokens, apostrophe-s, parentheticals, long/short words;
- **punctuation/structure:** comma, semicolon, colon, double-quote, question, exclamation,
  period, sentence-count, dash, ellipsis.

**Two frozen selection gates (on a held-out split, before ANY verdict):**
1. **balance** — keep `A` only if its parity class balance ∈ **[0.40, 0.60]** (else the
   latent is near-constant / un-probeable);
2. **input-undecodability (§4)** — keep `A` only if a **linear raw-token probe**
   (bag-of-tokens counts) recovers its parity at **≤ 0.60** (≈ chance; parity should pass —
   this catches any that leak).

**F3-R2 (no rescue):** if **fewer than 24** attributes survive both gates, the bank is too
thin to legitimately test `d_dec ≥ 20` → file **F3-R2**; do **not** hand-add attributes.

**Decision selection (pre-registered, to avoid selection-on-test optimism):** among the
survivors, pick the **decision** as the attribute with the highest **validation** `z1_acc`
(control-sufficiency), then **test its `z1_acc` once** on a held-out split. **Non-decision
state = all other survivors.** All fingerprint numbers use the test split.

## 3. Fingerprint (information basis, un-masked — the Amendment-1 lesson)

Per candidate set, on GPT-2's body (and the random-init floor):

- **`d_dec`** — decodable dimensionality = effective rank of the stacked per-latent linear
  readout directions. Real high-dim bar: **`d_dec ≥ 20`.**
- **`z1_acc`** — control-sufficiency of the decision (compact readout at ≈ full accuracy).
- **`cross_latent_leak`** — the decision-shadow → other latents; ≈ chance ⇒ **resists**
  (recoverability, *the* axis per this session's resistant-body test).
- **`body_carry`** — mean recovery of the non-decision latents, **pretrained vs
  random-init floor**; the objective contrast is `body_carry_pretrained − body_carry_floor`.
- **Apparatus-liveness (H4):** a control latent *known to be in the shadow* (the decision
  itself, or a copy) must leak ≈ 1 — proving the leak-probe is live, so a null on the
  others means *resists*, not *dead probe*.
- **Compute-can't-cross (H5):** a larger (MLP) reconstructor must not beat the `leak`
  floor → the resistance is **information-loss / recoverability**, not a weak probe.

## 4. Verdict (pre-registered)

**SHARP (R2's bet wins the internal half)** iff *all*: `d_dec ≥ 20` **and** `z1_acc ≥ 0.70`
**and** `cross_latent_leak ≈ chance` (perm-controlled) **and**
`body_carry_pretrained − body_carry_floor ≥ 0.15` (objective-driven above the
architectural floor) **and** the H4 liveness control leaks. → proceed to §7 external review.

**MARGINAL / negative** iff *any*: `leak` high (the shadow reconstructs the rest — the
**net.7 trap**, a real "it doesn't resist" result) **or** `d_dec < 20` (not high-dim on
this substrate) **or** `body_carry_pretrained ≈ floor` (the richness is passive/
architectural, not objective-driven). A clean MARGINAL is an honest R2 negative — real
LLMs may not resist a narrow decision's shadow, and that is a legitimate finding.

## 5. Named failure modes (before the run)

- **F1-R2 — the net.7 trap (the expected risk).** GPT-2's state is effectively
  reconstructable from the control-subspace → control-sufficient **and** state-sufficient
  → marginal. The whole point of R2 is that this can happen.
- **F2-R2 — control-insufficient.** GPT-2 doesn't represent the decision compactly
  (`k_control` large / `z1` low) — high-dim body but no compact shadow, not the split.
- **F3-R2 — de-confound fails.** No candidate passes the raw-token-chance gate → the
  "computed" latents are input structure. Re-pick / construct harder attributes; no verdict.
- **F4-R2 — architectural-floor confound.** `body_carry_pretrained ≈ random-init floor` →
  the non-decision richness is passive random features, not built by pretraining. This is
  the contrast doing its job; report as a negative, do **not** drop the floor.

## 6. Cost / build / reuse

- **CPU-tractable** (unlike the toy *training*): GPT-2 small forward passes over a few
  thousand passages ≈ minutes–1 h on CPU; the random-init floor is the same, free; probes
  are cheap. **No GPU / no multi-hour training wall.** (Downloading GPT-2 weights: set
  `HF_TOKEN` first per `reference_local_model_keys` to dodge the HF rate limit; GPT-2 is
  public, un-gated.)
- **Reuse:** the un-masked fingerprint + `d_dec` + perm-control + the §4 input-probe gate
  from `scripts/chatv2_phase0_bodyresist.py`. **New:** a HF GPT-2 activation extractor +
  the real-text attribute suite → `scripts/chatv2_r2_real_substrate.py`. Dependency check:
  `transformers` availability (verify before the run; the box is CPU-only).

## 7. The gated second half — external review (R2 is not self-licensing)

A SHARP internal result does **not** license R2. Per `PROMOTE_GATE.md` §5, assemble the
external **mech-interp review packet** (the NSE / Yang-Mills / Riemann / Hodge pattern):
reading-only ask — *is the measurement an artifact? is the framing a category error? is
it reproducible by a simpler account (linear probing / superposition / platonic-rep)?* —
sent to named ML/interpretability reviewers **invited to refute.** Only on their sign-off
does R2 promote. Owner-driven.

## 8. Cross-references

- [`PROMOTE_GATE.md`](PROMOTE_GATE.md) — the R2 gate + the do-not-claim ledger this obeys.
- [`PHASE1_R1_COMPLETION.md`](PHASE1_R1_COMPLETION.md) — R1 MET (the rung this builds on).
- [`PHASE0_2_COMPUTED_LATENTS.md`](PHASE0_2_COMPUTED_LATENTS.md) — the fingerprint + the
  variance-masking lesson (why the information basis, not variance PR).

## MVP RUN 1 (2026-06-29): F3-R2 — bank too thin on the reachable corpus; + a design lesson

`scripts/chatv2_r2_real_substrate.py`, GPT-2 small, CPU. **Corpus:** the box has network only
to HuggingFace (Gutenberg/GitHub unreachable) and no `pandas`/`pyarrow` to read wikitext
parquet → the only readable real-prose corpus was **TinyStories** (GPT-4-written children's
stories — simple English). N=1500, seq=128, frozen 51-attribute bank.

**Result: `F3-R2` — 18/51 attributes survive** balance[0.40,0.60] + raw-token-probe ≤ 0.60
(< 24 → bank too thin to legitimately test `d_dec ≥ 20`). **No verdict, no rescue** (per the
pre-registration: do not hand-add attributes or loosen gates). The 18-way thinness is a
*corpus* effect — TinyStories' simple vocabulary makes nominalizers / comparatives /
semicolons / all-caps / digits / rare suffixes near-constant → they fail the balance gate.

**The deeper lesson this MVP surfaced (the real R2 design problem):** count-parities were
chosen to inherit the toy's clean input-undecodability (parity is nonlinear ⇒ a linear
raw-token probe can't read it). But the toy's model was *trained* to compute its parities;
**GPT-2 was not trained to count feature occurrences and take parity**, so even with ≥24
survivors the decision step would very likely file **F2-R2 (control-insufficient)** — GPT-2
does not represent count-parities. This is the genuine R1→R2 tension: **"input-undecodable"
(needs nonlinearity) conflicts with "actually computed by a *pretrained* model" (favours
linearly-present features).** Resolving it is the substance of R2 v2:
- a **richer human-text corpus** (wikitext: needs `pyarrow`, or another HF-hosted plain-text
  human corpus) — clears F3; **and**
- a **latent family GPT-2 demonstrably represents** yet is input-undecodable — e.g. computed
  *relational/positional* attributes (agreement, coreference-ish, in-context relations) that
  a linear token-probe still misses, rather than raw regex count-parities.

R2 stays **NOT STARTED-effectively** on the gate (this MVP filed F3, not a body-resistance
read). No R2 promotion, no public claim, no R3 / world-model language — the NSE page's
"resistant substrate" line stays "to be run."
