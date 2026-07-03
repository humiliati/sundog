# Percival Track-C v3 — The Noisy Margin (crispness is pairing) — pre-registration

*v1 mapped the frontier; v2 proved recovery = probe mass on the disagreement region D. The registered
bridge claim was: noisy empirical deceptive-vs-robust margins instantiate the same crisp inequality
under optimization noise. v3 asks whether that survives as a THEOREM. It does — but the theorem is
about evaluation structure, not about noise being small: the crisp margin survives arbitrary noise iff
the comparison is PAIRED, and the familiar "noisy uninterpretable margin" is derivable as unpaired
evaluation on a disagreement-sparse eval set.*

Status: **PRE-REGISTERED 2026-07-03.** Three parts: (T) theorems — target = machine-checked in
`sundogcert`; (N) Monte-Carlo verification of the noisy laws; (B) the residual empirical bridge
predictions the toys cannot establish, stated as falsifiable claims about real systems.

## Setup

Two hypotheses V, W′ agreeing everywhere except a disagreement region D (v2's deceiver). Evaluation
observes a noisy richness `ρ̂(a,t) = ρ(a,t) + ε` per (action, context) observation. Selection margin
`Δscore = score(V) − score(W′)`; prior margin `λ·ΔL = 0.3` toward W′ (inverted). `h = |S∩D|` = hits.
**Paired** evaluation: both hypotheses scored on the same contexts; where they take the same action
they receive the SAME observation. **Unpaired**: independent observations per (hypothesis, context).

## T — the theorems (Lean targets, ℚ, finite lists)

- **T1 (zero-coverage capture is noise-invariant / paired cancellation).** If the two policies agree on
  every sampled context, then for EVERY observation function (arbitrary noise, adversarial included)
  the paired margin equals exactly −λ·ΔL. No noise process can rescue zero coverage; v2's Q1/Q2/Q4
  inseparability is noise-robust, not a noiseless idealization.
- **T2 (bounded-noise recovery margin).** If every observed per-hit gap is ≥ g − 2β (true gap g, per-
  observation noise bound β), then `h·(g−2β) > λ·ΔL ⟹ recovery` — recovery is guaranteed under ANY
  bounded noise once coverage clears the noise-degraded threshold. Paired noise tolerance:
  `β*_paired = (h·g − λΔL)/(2h)` — per-hit, independent of eval size.
- **T3 (unpaired fragility witness).** In unpaired evaluation the shared (agreeing) contexts stop
  canceling: an explicit adversarial assignment (ε=−β on V's observations, +β on W′'s) shifts the
  margin by −2β·|S_shared|, flipping any true margin m once `β ≥ m/(2·|S|)`. So
  `β*_unpaired = (h·g − λΔL)/(2·|S|)`, and **tolerance ratio paired:unpaired = |S|/h — the crisp margin
  degrades exactly as the disagreement fraction h/|S| shrinks.** The owner's observation ("noisy,
  uninterpretable margin even at toy scale") is the h/|S| → 0 limit, derived.

## N — Monte-Carlo predictions (Gaussian ε, σ ∈ {0.25, 1.0, 5.0}; h ∈ {0,1,2,4}; |S|=24; 20k trials/cell)

- **N1 (paired crispness at h=0):** margin ≡ −0.3 with ZERO sample variance at every σ including 5.0;
  capture rate exactly 1. (The T1 cancellation, watched happening.)
- **N2 (paired probit law):** recovery rate matches `Φ((h − λΔL)/(σ√(2h)))` within
  `max(4σ_MC, 0.012)` — the v2 step function smears into a probit whose CENTER is the same crisp
  inequality and whose width is the per-hit noise.
- **N3 (unpaired smearing):** recovery matches `Φ((h − λΔL)/(σ√(2|S|)))`; empirical margin-variance
  ratio paired/unpaired ≈ h/|S| (±35%); at h=0 the unpaired comparison is a near-coin-flip
  (`Φ(λΔL/(σ√(2|S|)))` ≈ 0.50–0.57) — selection between behaviorally-identical models under unpaired
  noise is decided by luck, where the paired comparison is deterministic prior-driven capture.
- **N4 (bounded adversarial thresholds, deterministic):** the T3 witness flips at β 10% above
  `m/(2|S|)` and cannot flip 10% below; paired flips at β above `m/(2h)` and not below.

## B — the residual empirical bridge (falsifiable on real systems; NOT established by v3)

- **B1 (variance collapse):** for a real checkpoint pair evaluated on the same benchmark, the variance
  of the paired (per-prompt, matched-seed) margin vs the unpaired (independent-run) margin should scale
  as the behavioral disagreement fraction. Testable with two checkpoints and one benchmark.
- **B2 (h=0 crispness):** where two models agree behaviorally on the whole eval, their comparison
  outcome should be near-deterministic and prior-driven (size/simplicity/inductive bias), with variance
  collapsing toward zero in the paired design.

## Verdict

`TCV3_CRISPNESS_IS_PAIRING` iff T1–T3 machine-checked (or, failing Lean, proven on paper with the MC
agreeing) AND N1–N4 hold. Any leak → `TCV3_LEAK`, reported as-is.

## Fences

- v3 isolates the NOISE layer: h is given (v2 already owns the coverage law); reachability/placement
  not re-modeled. The Gaussian probit shape is MC-verified, not Lean-verified (Lean gets the exact
  cancellation + bounded-noise inequalities in ℚ).
- The theorems are about the toy's selection rule (sum-of-observations minus description-length
  prior). B1/B2 are where the claim meets real systems and can die; they are registered, not run.
- Connects: pairing = D-restriction is why differential/targeted evaluation works; interp (v2's
  whitebox) and paired eval are the same move at different layers — restrict the comparison to where
  the hypotheses disagree.
