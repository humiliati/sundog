# Percival Track-C — B1/B2 real-system bridge — SCOPE (pre-build)

*The one live empirical edge of the Track-C arc (v1 frontier, v2 coverage, v3 pairing theorem, v4
fixed-point gate): do the pairing laws hold on a REAL checkpoint pair? This is the place the program
can die on contact with reality. Scope only — no code, no downloads; build is gated on owner sign-off
and the S0 prior-art check.*

Status: **SCOPED 2026-07-03. NOT pre-registered yet** — S0 (prior-art) and S1 (derivation audit) must
close first; the exact gates below become the pre-registration after that.

## 0. What is actually being transported

v3 proved (Lean, `PercivalNoisyMargin.lean`): paired comparisons cancel noise on shared behavior
exactly, so margin variance lives only on the disagreement region D; unpaired comparisons carry noise
from every context; tolerance ratio = disagreement fraction. The real-system versions:

- **B1 (ratio law):** for two models evaluated on one benchmark, Var(paired margin) / Var(unpaired
  margin) equals a derived function of the score-disagreement fraction — not merely "paired is
  smaller" (textbook) but the EXACT ratio.
- **B2 (h=0 crispness):** where two models agree behaviorally on the whole eval, the paired margin is
  exactly zero — so any measured nonzero variance is a direct read of the harness's nondeterminism
  floor. B2 doubles as an eval-reproducibility audit.

**NOT claimed:** anything about deception detection on real models. The deception link (the margin
between a deceptive and robust model lives entirely on D, so unpaired eval wastes power at rate
|S|/h) is interpretive; the run tests the variance law on ordinary checkpoint pairs.

## 1. The exact statistics (derived here so the gate is a number, not a vibe)

Deterministic scoring (see §3): per item i, scores a_i, b_i ∈ {0,1}; margin m_i = a_i − b_i ∈
{−1,0,+1}. Population quantities: p_A, p_B (accuracies), Δ = p_A − p_B, and **d_s = P(a ≠ b)** the
score-disagreement fraction (⊆ behavioral disagreement d_b: items where outputs differ but both score
0 are in d_b, not d_s — both fractions reported; the law runs on d_s).

Noise source under deterministic scoring = **item subsampling** (which n items you evaluate — exactly
Miller-style eval sampling error). Protocols, pinned:

- **paired(n):** draw ONE n-item subset, evaluate both models on it, aggregate per-item differences.
  `Var_paired = (d_s − Δ²)/n`.
- **unpaired(n):** draw TWO independent n-item subsets, one per model, difference the means.
  `Var_unpaired = [p_A(1−p_A) + p_B(1−p_B)]/n`.

**The B1 ratio law (the pre-registrable number):**

```
R = Var_paired / Var_unpaired = (d_s − Δ²) / (p_A(1−p_A) + p_B(1−p_B))
```

Toy correspondence check: in the homogeneous limit (close models, p_A≈p_B=p, Δ≈0) R ≈ d_s/(2p(1−p)) —
the toy's h/|S| with the base-rate variance as the "per-context noise." The item-difficulty deviation
plays the toy's shared ε: identical on agreement items (cancels paired), independent draws unpaired.
Power corollary: paired needs R× the items of unpaired for equal power — for adjacent checkpoints
(d_s ~ 3%, p ~ 0.35) that's ~14× fewer items. This concrete multiplier is the practical payoff.

**Design discovery 1 (registrable as its own arm): temperature > 0 is structurally UNPAIRED.** With
sampling, generation noise is independent per model even on the same prompt — it never cancels, on
any item. Prediction B1-c: the T=0 paired ratio collapse does NOT occur at T>0 (variance floor ∝
generation entropy across ALL items). Practical form: greedy paired evals are ~1/R more powerful for
close models; sampled evals forfeit the collapse.

**Design discovery 2: B2 is a nondeterminism audit.** Deterministic scoring of a model against
ITSELF has margin ≡ 0 identically; anything else measures GPU/batching nondeterminism. B2's product
is the harness noise floor, which also bounds the smallest ladder signal B1 can resolve.

## 2. S0 — prior-art gate (must close before pre-registration)

Paired-comparison variance reduction is textbook (paired t-test; common random numbers). Known
adjacent work to check and position against, explicitly:

- Miller 2024, "Adding Error Bars to Evals" (Anthropic) — recommends paired analysis for model
  comparisons; establishes the qualitative direction. **The delta we'd claim: the exact ratio law R
  (disagreement-fraction form), the h=0 zero-variance branch, the T>0 structurally-unpaired
  observation, and the machine-checked derivation (Lean anchors) — verify none of these is already
  published there or in the CRN literature.**
- Eval-variance / benchmark-reproducibility literature (harness nondeterminism reports).

Gate: if the exact law is already published, the run demotes honestly to *replication + Lean-anchored
derivation note* — still worth one page, claim rewritten BEFORE any run.

## 3. Substrate + protocol (the thorough design)

- **Models: Pythia-160M public checkpoint ladder** (154 checkpoints, one training run — disagreement
  fraction becomes a DIAL). Pairs: final (143k) vs {142k, 140k, 130k, 110k, 70k, 30k} = 6 rungs of
  increasing d_s; plus **self-pair** (B2); plus one **wide pair** (160m vs 160m-deduped final) as the
  large-d_s anchor. Fits the 1080 (fp16, forward-only); ~8 checkpoint downloads ≈ 2.5 GB (HF_TOKEN
  keyring discipline per `Dev\AGENTS.md`).
- **Eval: multiple-choice by choice-loglikelihood ranking** (HellaSwag validation subset, frozen
  ~2,000 items; LAMBADA greedy last-word as a second scoring family). Loglik ranking = forward passes
  only, fully deterministic, no sampling — the T=0 arm by construction.
- **T>0 arm (B1-c):** small generative task (e.g., 300-item LAMBADA-style completion) at T=0.7,
  k=4 samples, scored per sample — one ladder rung only (cost control).
- **Estimation:** bootstrap over item subsets (B=2,000 replicates; n=500 per subset; paired = shared
  subset, unpaired = disjoint independent subsets). Report per rung: d_b, d_s, Δ, p_A, p_B, measured
  Var_paired, Var_unpaired, measured R vs derived R with bootstrap CI; sign-stability of the margin
  (fraction of replicates agreeing with the full-set sign).
- **Budget:** loglik, 2k items × 9 checkpoints ≈ minutes each on the 1080; full ladder well under
  ~2h. Long-run discipline: launched as owner/background jobs per house rule; decoupled
  score-caching (per-checkpoint per-item score files) so all statistics are instant re-analysis.

## 4. Pre-registered gates (to be frozen at pre-registration, after S0/S1)

- **B1-a (the ratio law):** measured R within the bootstrap 95% CI of derived R on ≥5 of 6 ladder
  rungs + the wide pair.
- **B1-b (monotone dial):** d_s and Var_paired both increase monotonically with checkpoint distance
  (ordering, not magnitudes).
- **B1-c (structurally unpaired):** at T>0, paired/unpaired variance ratio ≫ the T=0 R on the same
  rung (no collapse; floor visible on agreement items).
- **B2-a (noise floor):** self-pair paired margin ≡ 0 across all replicates, or the measured floor is
  reported and is < the smallest rung's signal (else the ladder is truncated to rungs above floor —
  reported, not hidden).
- **B2-b (near-agreement crispness):** on the closest rung, the paired comparison's SIGN is stable
  (≥99% of replicates) while the unpaired sign flips materially (<90% stability) — the "deterministic
  prior-driven selection vs coin-flip" contrast from v3's N1/N3, on real weights.

**Verdicts:** `B1B2_BRIDGE_CONFIRMED` (all) / `B1B2_QUALITATIVE_ONLY` (directions hold, exact R
misses — reported as the finding; the toy's noise model is wrong about real evals in a specific,
stated way) / `B1B2_REFUTED` (paired variance does NOT collapse per the law — the Track-C bridge
dies, banked honestly) / `B1B2_FLOOR_BLOCKED` (nondeterminism floor swamps the ladder).

## 5. Failure modes + kills (registered before build)

1. **Harness nondeterminism** (GPU reductions, batching) makes even loglik scores flicker → B2 floor
   nonzero → truncate ladder to resolvable rungs; if none resolvable, `B1B2_FLOOR_BLOCKED`.
2. **d_s too small at close rungs** (discrete 0/1 scores, 2k items → d_s below counting noise) →
   widen item set (up to full 10k HellaSwag val) before touching the gates; pre-set in
   pre-registration, not tuned after.
3. **Δ² term non-negligible** (wide pair) → the exact R formula already carries it; no adjustment.
4. **Prior art owns the law** (S0) → demote to replication note; run optional.
5. **Chatv2-adjacency discipline:** real-LLM substrate, but the claims are eval-statistics only — no
   body-resistance, no R2 vocabulary, PROMOTE_GATE language rule respected; no deception-detection
   claims (interpretive link only, stated once, fenced).

## 6. Sequence

S0 prior-art memo → S1 derivation audit (this doc §1 re-derived cold + reconciled) → freeze
pre-registration (gates above verbatim) → build harness (score-caching first, statistics second) →
self-pair B2 floor FIRST (it gates everything) → ladder → adjudicate → bank. Owner sign-off gates the
build; the whole run is local (1080) with no external surface.
