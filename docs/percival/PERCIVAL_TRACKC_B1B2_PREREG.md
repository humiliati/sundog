# Percival Track-C B1/B2 — FROZEN PRE-REGISTRATION

*Freeze of the scoped bridge test ([`PERCIVAL_TRACKC_B1B2_BRIDGE_SCOPE.md`](PERCIVAL_TRACKC_B1B2_BRIDGE_SCOPE.md))
after S0 ([`PERCIVAL_TRACKC_B1B2_LITPASS_MEMO.md`](PERCIVAL_TRACKC_B1B2_LITPASS_MEMO.md) — demotion
FIRED) and the S1 derivation audit (§1 below). Gates frozen before any download or score.*

Status: **FROZEN 2026-07-03.** Claim register (demoted, binding): this is a **CALIBRATION of the
classical paired-comparison variance law** (McNemar 1947 / Connor 1987 lineage; applied to LLM evals
by Miller 2024 arXiv:2411.00640 and Kotawala 2026 arXiv:2605.30315) **along a training trajectory**,
plus three design contributions: the checkpoint-distance dial, the self-pair floor audit, and the
T=0/T>0 pairing contrast. NOT a new statistical law. No deception-detection claims; the
Track-C/coverage link is interpretive and stated once. The Lean anchors
(`PercivalNoisyMargin.lean`) are the derivation's machine-checked form.

## 1. S1 — derivation audit (re-derived cold; corner-checked)

Per item i: scores `a_i, b_i ∈ {0,1}`; margin `m_i = a_i − b_i ∈ {−1,0,+1}`. Population:
`p_A = E[a]`, `p_B = E[b]`, `Δ = p_A − p_B`, `d_s = P(a ≠ b)` (score disagreement),
`d_b` = prediction disagreement (`d_s ≤ d_b`; both-wrong-different-answer items are in `d_b` only).

- `E[m] = Δ`; `E[m²] = d_s`; `Var(m) = d_s − Δ²`.
- **Paired(n)** (one index set, both models): `Var_p = (d_s − Δ²)/n`.
- **Unpaired(n)** (two independent index sets): `Var_u = [p_A(1−p_A) + p_B(1−p_B)]/n`.
- **Ratio: `R = (d_s − Δ²) / (p_A(1−p_A) + p_B(1−p_B))`.**

Corner checks (audit): identical models → `d_s = Δ = 0` → `R = 0` (the exact zero-variance branch —
v3's T1). Antithetical models (`a = 1−b`, `p_B = 1−p_A`) → numerator `1−(2p_A−1)² = 4p_A(1−p_A)` =
2× denominator → `R = 2`: pairing HURTS under negative correlation, matching classical theory —
the audit's non-trivial corner passes. Bootstrap protocol resamples WITH replacement from the frozen
item set (paired: one index vector shared; unpaired: two independent index vectors), matching the
iid derivation for both arms; no finite-population correction needed (same factor would cancel in R
anyway).

## 2. Frozen protocol

- **Models (HF, fp32, fixed batch):** `EleutherAI/pythia-160m` revisions `step143000` (main),
  `step142000`, `step140000`, `step130000`, `step110000`, `step70000`, `step30000`;
  `EleutherAI/pythia-160m-deduped` revision `step143000` (wide pair). **Self-pair = TWO independent
  eval passes of main `step143000`** (fresh model load per pass, same everything).
- **Determinism controls (from S0's floor literature):** dtype float32; fixed batch size 16; frozen
  item order; no compile; greedy/loglik only (T-arm excepted). The floor is whatever survives these
  — measured, not assumed.
- **Tasks:** (1) **HellaSwag** validation, frozen subset = 2,000 items (indices shuffled by numpy
  seed 20260703, first 2000). Scoring: per ending, sum of continuation-token log-probs given
  context; **primary score = raw-loglik argmax correctness ("acc")**; byte-normalized argmax
  ("acc_norm") cached as secondary/descriptive. (2) **LAMBADA-openai**, frozen subset = 1,000 items;
  greedy teacher-forced exact match of the target word's tokens (all argmax positions match).
- **T-arm (B1-c):** LAMBADA subset first 300 of the frozen 1,000; temperature 0.7, k=4 samples/item,
  seeded per (model, item, sample); per-item score = mean exact-match over k. Rung: main step143000
  vs step130000 only. Positioning (binding): this arm deliberately studies the greedy/sampled
  CONTRAST; Miller §3.3's thermostat caution acknowledged — we are not advising T=0 as a variance
  hack, we are measuring what pairing can and cannot cancel at each temperature.
- **Estimation:** per pair per task, from cached per-item scores: `d_b, d_s, Δ, p_A, p_B`; bootstrap
  `B = 2000` replicates at `n = 500` (seed 20260703): paired variance (shared index vector),
  unpaired variance (independent index vectors), `R_meas = Var_p/Var_u`; sign stability = fraction
  of paired (resp. unpaired) replicates whose margin sign matches the full-subset sign.
- **Pairs (frozen list):** self (passA vs passB); main-143000 vs each of the 6 ladder rungs; main
  vs deduped (wide). Primary task for gates = HellaSwag acc; LAMBADA = replication family
  (reported, gates B1-a/B1-b evaluated on it descriptively).

## 3. Frozen gates

- **B2-a (floor, runs FIRST):** self-pair `d_s = 0` exactly (fp32 + fixed batch expectation), OR
  measured floor `d_s_self < d_s(closest rung)/10` (ladder fully resolvable). Else truncate the
  ladder to rungs with `d_s > 10 × d_s_self` and report; if none survive → `B1B2_FLOOR_BLOCKED`.
- **B1-a (calibration):** `|R_meas − R_formula| / R_formula ≤ 0.15` on ≥ 5 of 6 ladder rungs AND the
  wide pair (HellaSwag acc). (Bootstrap estimator noise at B=2000 ≈ 4–5% relative — the 15% band is
  meaningfully tight above it.)
- **B1-b (monotone dial):** along rungs 142000→30000, `d_s` and bootstrap `Var_p` both nondecreasing
  with ≥ 4 strict increases out of 5 steps (HellaSwag acc).
- **B1-c (structurally unpaired):** on the T-arm rung: `R_meas(T=0.7) ≥ 3 × R_meas(T=0)` AND
  agreement-item margin variance is exactly 0 at T=0 while > 0 at T=0.7.
- **B2-b (crispness):** there exists a rung with paired sign-stability ≥ 0.99 while unpaired < 0.90
  (per-rung stability table reported in full).

**Verdicts:** `B1B2_BRIDGE_CONFIRMED` (all five) / `B1B2_QUALITATIVE_ONLY` (B1-b, B1-c, B2 gates
hold but B1-a misses — the classical law's noise model misdescribes this design in a specific way;
reported as the finding) / `B1B2_REFUTED` (B1-b or the paired collapse itself fails) /
`B1B2_FLOOR_BLOCKED` (per B2-a). No post-hoc gate edits; misses are findings.

## 4. Exact commands (gates = these commands, unchanged)

```powershell
# venv: C:/Users/hughe/.venvs/sundog-gpu  (HF_TOKEN loaded from the keyring per Dev\AGENTS.md, silent)
# Step 0 — self-pair floor (FIRST):
python scripts/percival_b1b2_cache.py --revision step143000 --tag main-143000-passA
python scripts/percival_b1b2_cache.py --revision step143000 --tag main-143000-passB
python scripts/percival_b1b2_stats.py --floor-only
# Step 1 — ladder + wide (skips existing tags):
python scripts/percival_b1b2_cache.py --revision step142000 --tag main-142000
python scripts/percival_b1b2_cache.py --revision step140000 --tag main-140000
python scripts/percival_b1b2_cache.py --revision step130000 --tag main-130000
python scripts/percival_b1b2_cache.py --revision step110000 --tag main-110000
python scripts/percival_b1b2_cache.py --revision step70000  --tag main-70000
python scripts/percival_b1b2_cache.py --revision step30000  --tag main-30000
python scripts/percival_b1b2_cache.py --model EleutherAI/pythia-160m-deduped --revision step143000 --tag dedup-143000
# Step 2 — T-arm (two models, sampled):
python scripts/percival_b1b2_cache.py --revision step143000 --tag main-143000-passA --t-arm
python scripts/percival_b1b2_cache.py --revision step130000 --tag main-130000 --t-arm
# Step 3 — adjudicate:
python scripts/percival_b1b2_stats.py
```

Outputs: `results/percival/b1b2/scores/<tag>/{hellaswag,lambada,tarm}.json` (per-item caches; all
statistics are instant re-analysis) → `results/percival/b1b2/summary.json` +
`docs/percival/PERCIVAL_TRACKC_B1B2_RESULTS.md`.

## 5. Fences

Eval-statistics claims only (no body-resistance, no R2 vocabulary — PROMOTE_GATE respected; no
deception-detection claims). One substrate family (Pythia-160M), one primary task — the calibration
is a point on a curve of possible designs, not "evals are solved." Budget ~3 GB downloads + well
under 2 h GPU; runs local; long steps backgrounded per house rule.
