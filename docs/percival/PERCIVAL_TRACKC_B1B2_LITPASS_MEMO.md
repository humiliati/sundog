# Percival Track-C B1/B2 — Lit-Pass Memo (S0)

> Prior-art and claim-boundary record for the B1/B2 real-system bridge
> ([`PERCIVAL_TRACKC_B1B2_BRIDGE_SCOPE.md`](PERCIVAL_TRACKC_B1B2_BRIDGE_SCOPE.md) §S0).
> Gates the pre-registration freeze and any run, page, or outreach claim.

**Date:** 2026-07-03
**Status:** S0 CLOSED. **The scope's demotion rule FIRES** (detail below). Treat all gap claims as
time-stamped: "not found in this pass," not "does not exist."
**Citation check:** arXiv:2411.00640 (Miller) and arXiv:2605.30315 (Kotawala) PDFs pulled and
text-verified this pass. **Verify-before-file catch: the WebFetch summarizer HALLUCINATED
confirmations shaped by our leading questions** (claimed 2605.30315 covers Pythia ladders,
temperature, and a nondeterminism floor; direct text extraction: 0 hits for all three). All verdicts
below rest on extracted text or abstracts, flagged per item. arXiv:2605.28873 and arXiv:2506.09501
abstract-level only. Cormier et al. (churn) and Koehn 2004 (paired bootstrap) cited from training
knowledge, NOT re-verified this pass.
**Surfaces:** scope doc + this memo only. No pre-registration frozen, no code, no downloads yet.

## Purpose

The exact ratio law `R = (d_s − Δ²)/(p_A(1−p_A) + p_B(1−p_B))` is a one-line corollary of two
textbook variance formulas. The question S0 answers: which of the scope's five claimed deltas are
already owned, and what does the run honestly test after the accounting.

## Executive Verdict

**Owned (classical + 2024–2026 literature) — the demotion fires:**

- **Paired analysis for LLM eval comparisons:** Miller 2024 (arXiv:2411.00640) — question-level
  paired differences recommended outright (§4.2); variance reduction via score correlation
  ("agreement on which questions are easy/hard"); clustered SEs; power analysis; resampling.
- **The binary paired-variance form and its unpaired contrast, on LLM benchmarks:**
  Kotawala 2026 (arXiv:2605.30315, ICML 2026 HT workshop) — McNemar/Connor required-N on discordant
  pairs (`σ_D²` form, our numerator), resolution ratio `q = N/N★`, **paired-McNemar required-N
  median 2.15× smaller than Miller's unpaired formula on real leaderboard data (IQR [1.60, 2.75])**,
  plus a calculator-misuse lemma (unpaired-shortcut off by ~2× in the close-comparison regime).
  This is our B1-a comparison, executed on leaderboards. The classical lineage (McNemar 1947;
  Connor 1987; Agresti–Min 2005) is cited inside it.
- **Close-pair paired design with a disagreement rate:** arXiv:2605.28873 (abstract-verified) runs
  paired-MDE budgeting on FP16-vs-quantized pairs with `ρ_d` disagreement — the "nearly-identical
  models, margin lives on discordant items" design exists for quantization pairs.
- **Harness nondeterminism as variance contaminant:** arXiv:2506.09501 ("Give Me FP32 or Give Me
  Death?") — numerical nondeterminism inflates reported eval variance; BF16 >> FP16 >> FP32≈0;
  batch-size-dependent reduction kernels (Thinking Machines' batch-invariance work) are the
  mechanism. The floor is documented; FP32/fixed-batch mitigations known.
- **Sampling noise under pairing:** Miller §3 — temperature/conditional variance is handled by
  RESAMPLING (variance reduction ~(1+2/K)/3 at K resamples), and §3.3 "Don't touch the thermostat!"
  explicitly advises AGAINST lowering temperature (it changes the object being measured). Our B1-c
  arm must be positioned against this, not around it.

**Not found in this pass (the surviving deltas — all design/synthesis, none "new statistics"):**

1. **The checkpoint-distance dial:** sweeping ONE training run's adjacent checkpoints (Pythia
   ladder) so `d_s` becomes a controlled dial, and validating the ratio law as a CURVE R(d_s) rather
   than at scattered leaderboard pairs. Kotawala validates N★ prescriptions on unrelated model
   pairs; nobody found running the law along a training trajectory. (Adjacent-known, not verified
   this pass: prediction-churn literature, Cormier et al. 2016 — version-to-version disagreement as
   an object of study, but not tied to paired-eval variance laws.)
2. **The self-pair floor audit as a gate:** model-vs-itself paired margin ≡ 0 identically under
   deterministic scoring, so any measured deviation IS the harness floor, run FIRST to bound the
   smallest resolvable ladder rung. The floor phenomenon is documented (2506.09501); the audit
   design gating a paired ladder is not found as such.
3. **The T=0/T>0 pairing contrast, stated as structure:** at deterministic scoring pairing cancels
   item noise EXACTLY (zero-variance agreement branch); at T>0 generation noise is unpairable —
   only resamplable at √K cost. Components owned (Miller's resampling; loglik-ranking evals are
   standard T-free practice); the measured CONTRAST arm and its "structurally unpaired" framing not
   found. Miller's thermostat caution stands: our T=0 arm studies greedy/loglik behavior
   deliberately and says so.
4. **Machine-checked derivation:** the Lean anchors (`PercivalNoisyMargin.lean` T1–T3,
   `PercivalFixedPoint.lean` round bridge) — ours by construction; an artifact, not a discovery;
   never sold as more.
5. **The deception-margin interpretation** (margin between deceptive and robust models lives
   entirely on D → unpaired eval wastes power exactly where it matters) — interpretive, one fenced
   line only. Adjacent AI-control/sandbagging-eval literature NOT searched this pass; do not state
   even the interpretive form as unowned without that search.

**Sundog synthesis (what the run now honestly is):** a CALIBRATION of the classical paired-variance
law on a new design — the within-run checkpoint dial — plus the floor audit and the T-contrast arm,
anchored to a machine-checked derivation. Replication-grade statistics, design-grade novelty.

## The demotion, applied to the gates

Per the scope's registered rule ("if prior art owns the law, demote to replication + Lean-anchored
derivation note BEFORE any run"):

- **B1-a** is now a *calibration gate* (measured R within CI of the classical formula along the
  dial), not a novel-law confirmation. Cite Miller + Kotawala as the law's owners in any output.
- **B1-b (monotone dial), B1-c (T-contrast), B2-a (floor audit), B2-b (sign-stability)** survive
  unchanged as design contributions.
- **Headline language for any future surface:** "we verify the classical paired-comparison law along
  a training trajectory and package the floor audit" — NOT "we discovered the ratio law."
- **New practical consequence from the floor literature:** run the ladder in FP32 with fixed batch
  size on the 1080 (160M fits comfortably) — pushes the B2 floor toward zero so the audit VERIFIES
  rather than discovers; report the BF16/FP16 floor as a secondary observation if cheap.

## Corrections this pass forces on the scope doc

1. §S0 status → CLOSED, demotion fired (this memo is the record).
2. §1's "the pre-registrable number" language stands mathematically but must carry the attribution
   (classical; Kotawala's 2.15× is the same comparison on leaderboards).
3. §3 substrate: add FP32 + fixed-batch requirement (floor mitigation, from 2506.09501/batch-
   invariance findings).
4. §4 B1-c: cite Miller §3.3 and state the arm as a deliberate greedy/loglik-object study.

## Verdict

**S0 CLOSED — proceed is licensed, with the demoted claim set.** The run is still worth doing: the
dial curve, the floor audit, and the T-contrast are real gaps at 2026-07-03, the compute is ~2h
local, and a clean `B1B2_QUALITATIVE_ONLY` or `_REFUTED` on the dial would be genuinely informative
about where the toy's noise model misdescribes real evals. Next: S1 cold re-derivation audit →
freeze the pre-registration with the demoted language → owner sign-off → build (B2 floor first).

## Sources (this pass)

- Miller 2024, arXiv:2411.00640 — PDF text-verified.
- Kotawala 2026, arXiv:2605.30315 (ICML 2026 HT workshop) — PDF text-verified; summarizer
  hallucination caught and corrected against extracted text.
- arXiv:2605.28873 (paired-MDE quantization pre-registration) — abstract.
- arXiv:2506.09501 ("Give Me FP32 or Give Me Death?") + Thinking Machines batch-invariance post —
  search-level.
- McNemar 1947; Connor 1987; Agresti–Min 2005 — via Kotawala's related work (verified present in
  its text).
- Cormier et al. 2016 (churn); Koehn 2004 (paired bootstrap) — training knowledge, not re-verified.
