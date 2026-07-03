# Percival Track-C — Self-Correction (results)

Generated 2026-07-03T07:41:20.376Z by `scripts/percival-trackc-self-correction.mjs`. Spec: [`PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md`](PERCIVAL_TRACKC_SELF_CORRECTION_SPEC.md).

Constructive toy: unsupervised objective-discovery (max observable richness ρ − λ·description-length) over {V, Û, W_naive, W_dec}, swept over gap × budget × simplicity-ordering. The agent never sees V; V only checks recovery.

## Verdict: **TC_SELF_CORRECTION_MAPPED**

Predictions: P1=true, P2=true, P3=true, P3_CTRL=true, P4=true, P5=true, P6=true.
Deceptive-capture region is EXACTLY {inverted ∧ defect-unsampled ∧ λ>0}: **true**.
Outcome counts: {"DECEPTIVE_RISK":32,"DECEPTIVE_CAPTURE":48,"SELF_CORRECTED":76,"TIE_OTHER":4}.

## Representative slice (gap ρ_wrong=0.3, λ=0.1)

| defect | ordering | ρ_wrong | budget k | λ | outcome |
| --- | --- | ---: | ---: | ---: | --- |
| reach | inverted | 0.3 | 4 | 0.1 | DECEPTIVE_CAPTURE |
| reach | inverted | 0.3 | 8 | 0.1 | DECEPTIVE_CAPTURE |
| reach | inverted | 0.3 | 10 | 0.1 | DECEPTIVE_CAPTURE |
| reach | inverted | 0.3 | 11 | 0.1 | SELF_CORRECTED |
| reach | inverted | 0.3 | 12 | 0.1 | SELF_CORRECTED |
| reach | non_inverted | 0.3 | 4 | 0.1 | SELF_CORRECTED |
| reach | non_inverted | 0.3 | 8 | 0.1 | SELF_CORRECTED |
| reach | non_inverted | 0.3 | 10 | 0.1 | SELF_CORRECTED |
| reach | non_inverted | 0.3 | 11 | 0.1 | SELF_CORRECTED |
| reach | non_inverted | 0.3 | 12 | 0.1 | SELF_CORRECTED |
| UNREACH | inverted | 0.3 | 4 | 0.1 | DECEPTIVE_CAPTURE |
| UNREACH | inverted | 0.3 | 8 | 0.1 | DECEPTIVE_CAPTURE |
| UNREACH | inverted | 0.3 | 10 | 0.1 | DECEPTIVE_CAPTURE |
| UNREACH | inverted | 0.3 | 11 | 0.1 | DECEPTIVE_CAPTURE |
| UNREACH | inverted | 0.3 | 12 | 0.1 | DECEPTIVE_CAPTURE |
| UNREACH | non_inverted | 0.3 | 4 | 0.1 | SELF_CORRECTED |
| UNREACH | non_inverted | 0.3 | 8 | 0.1 | SELF_CORRECTED |
| UNREACH | non_inverted | 0.3 | 10 | 0.1 | SELF_CORRECTED |
| UNREACH | non_inverted | 0.3 | 11 | 0.1 | SELF_CORRECTED |
| UNREACH | non_inverted | 0.3 | 12 | 0.1 | SELF_CORRECTED |

## Reading

- **P1** richness defeats naive wireheading (W_naive never selected — Σρ=0 sinks it). The observable ρ is the third condition beyond invariance+simplicity.
- **P2** the courtesy proxy Û never wins (V is richer when gap>0 AND strictly simpler).
- **P3** F4 fires: under an inverted simplicity prior with the defect unsampled, the deceptive wirehead W_dec — observationally identical to V on the sample, and simpler — is selected. False self-correction.
- **P4** raising the budget to sample the defect context exposes W_dec (it plays the world-collapsing action there, ρ drops) → V recovered. Exploration defeats *sampleable* deception.
- **P5** grace (external V-labels) recovers V in one step everywhere.
- **P6** the residual: when the defect context is unsampleable, no budget closes it — under inversion W_dec wins at every budget and only grace recovers V. The deceptive-alignment ceiling; the reason the story needs the hermit.

## Honest boundary

Constructive demonstration, not empirical discovery — the frontier follows from the construction; the checked content is that it is CLEAN (P1-P6 hold) and that the deceptive-capture region is EXACTLY {inverted ∧ defect-unsampled ∧ λ>0}, with no leak cells. ρ is an observable weaker-than-V signal by design; the result is about ρ's insufficiency against deception, not empowerment=alignment. Toy, discrete, single instantiation. Feeds the Angle-4 fixed-point: the corrigible fixed point exists iff the update channel resists the proxy (budget defeats sampleable deception) AND the simplicity prior is not inverted.

