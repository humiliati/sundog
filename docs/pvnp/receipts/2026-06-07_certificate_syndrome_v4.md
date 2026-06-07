# Sundog Certificate Problem — Syndrome Certificate v4 Receipt (Median-Calibration + R2′)

- Receipt id: `pvnp-certificate-syndrome-v4-2026-06-07`
- Phase / probe: Phase 4 / §5 — v4 closes the two flaws v3 named in itself: (1) the
  prediction method (mean→**median** lock), (2) the rung-2 (`R2′`) confound. **No new
  scientific claim** beyond v3; this hardens methodology and re-interprets one v3 point.
- Date run: 2026-06-06 → 2026-06-07 (overnight chain; wall ≈ 8 h; op-count the cost signal)
- Runner: `pvnp-certificate-syndrome-v4.py` (`--median-precal` → `--validate-v3` →
  `--frozen --regime r2prime` → `--summarize`)
- Result dir: `results/pvnp/certificate-syndrome-v4/` (transient, gitignored)
- Frozen slate: [`SUNDOG_CERTIFICATE_SYNDROME_V4_SLATE.md`](../SUNDOG_CERTIFICATE_SYNDROME_V4_SLATE.md)
  (stage-1 frozen 2026-06-06)
- Stage-2 lock: `SUNDOG_CERTIFICATE_SYNDROME_V4_PREDICTION_LOCK.json` (median-precal)
- v3 receipt (the ladder this re-reads): [`2026-06-06_certificate_syndrome_v3.md`](2026-06-06_certificate_syndrome_v3.md)
- **Adversarially audited** (5-dimension Workflow, 14 agents): sound to write up; the
  framing below incorporates all confirmed corrections.

## Verdict (dual, with caveats — neither half is a clean pass)

**(A) The median fix is validated only as an *in-sample retrofit*, not a universal
calibration.** On v3's already-measured rungs 1 & 3 the median lock predicts the frozen
`C@50%` within factor 2 at all four points (ratios **1.00 / 1.20 / 1.33 / 1.09**), genuinely
closing v3's mean-based 5.32× rung-2 miss. **But on the one *fresh*, never-before-measured
regime (`R2′`), the median lock missed Stern `l=8` by 2.77×** (`within_factor2=false`). By
the slate's own all-points definition (verdict branch: "every re-measured point within
factor 2 *incl. the R2′ retest*") the run-level verdict is therefore **`method_still_off`
/ `model_deviation`, NOT `method_validated`** — the harness emitted `method_validated` only
because `--validate-v3` is scoped to the two v3-ground-truth regimes. The median form is a
demonstrated improvement over v3's mean form on heavy-tailed Stern; it is **not** a proven
prospective predictor.

**(B) `R2′` is a NEAR-TIE that dissolves v3's `l=10` artifact *downward* — not a clean
Stern crossover.** Re-measuring `[160,80]w16` (same code as v3 rung-2 via a `code_digest`
gate, fresh targets) with the proper fixed `l∈{8,9}`: `C_LB = 2.279×10⁹`,
`C_Stern(l8) = 2.229×10⁹`, `C_Stern(l9) = 2.236×10⁹` — all three within a **2.2% band**,
`St/LB = 0.978`. The harness's pre-registered binary rule mechanically labels this
`Stern_wins`, **but 2.2% is not a defensible winner**: it sits inside (i) the regime's own
**seed-noise floor** (the same-regime median wanders 1.43–2.77× on a single target-seed
flip) and (ii) a **censoring asymmetry that biases Stern low** (below). So v3's dramatic
`St/LB = 1.96` LB-win is **resolved DOWN to a dead heat** — the `l=10` window (analytic
`l*=8`) was the handicap — but **`[160,80]w16` is best described as a near-tie, neither a
clean Stern win (unlike rung-1 0.75, rung-3 0.76) nor an LB win.** The strong reading
"crossover monotone in `w` / Stern wins at every `w≥16`" is **not** shown and is withdrawn.

The honest, load-bearing claim that survives: **v3's off-pattern `[160,80]w16` LB-win was an
`l`-selection artifact, not a genuine 2-D LB feature.** The v3 measured ladder is untouched.
Integrity is fully clean (below). `C_best ≈ 2.23×10⁹` is an upper bound vs LB/Stern.

## (A) Method validation — in-sample retrofit on v3 rungs 1/3 (walled off)

| Regime / variant | v4 median prediction | v3 *measured* `C@50%` | ratio | within 2× |
| --- | ---: | ---: | ---: | :---: |
| v3-R1 `[128,64]w16` LB | 1.351×10⁸ | 1.351×10⁸ | 1.00 | ✓ |
| v3-R1 Stern `l=7` | 1.215×10⁸ | 1.016×10⁸ | 1.20 | ✓ |
| v3-R3 `[192,96]w18` LB | 1.434×10¹⁰ | 1.076×10¹⁰ | 1.33 | ✓ |
| v3-R3 Stern `l=9` | 8.907×10⁹ | 8.136×10⁹ | 1.09 | ✓ |

This is a sound **retrofit / in-sample consistency check** against pre-existing ground
truth — it closes v3's mean-vs-median error (v3 rung-2 missed 5.32× on the mean; the median
form tracks). It is **walled off** from the R2′ result and must **not** be read as a
universal validation: the lone prospective test (R2′, fresh `target_seed`) failed.

## (B) R2′ rung-2 — the near-tie, reported as a compound statement

**Frozen verdict: `model_deviation` (median lock missed: Stern `l8` 2.77×,
`within_factor2=false`; `l9` 1.98×; LB 1.43×) — rung-2 mechanical outcome `Stern_wins` on a
2.2% margin.** These are inseparable in every artifact and are reported together here.

| Variant | measured `C@50%` | median-lock pred | ratio | censored | max_B |
| --- | ---: | ---: | ---: | ---: | ---: |
| LB | 2.279×10⁹ | 1.589×10⁹ | 1.43 (✓) | **0/64** | 6,888 |
| Stern `l=8` | 2.229×10⁹ | 8.048×10⁸ | **2.77 (✗)** | **10/64** | 2,975 |
| Stern `l=9` | 2.236×10⁹ | 1.132×10⁹ | 1.98 (✓) | **13/64** | 5,001 |

`C_Stern = min(l8,l9) = 2.229×10⁹` (best `l=8`); `St/LB = 0.978`.

**Why this is a near-tie, not a Stern win:**

- **Seed-noise floor.** The same `[160,80]w16` regime's median moves **1.43–2.77×** across a
  single target-seed change (precal seed `20264203` → frozen seed `20264202`, identical
  `code_digest`). A 2.2% inter-attacker gap is far inside that floor.
- **Censoring asymmetry (biases Stern low).** LB is fully observed (0/64 censored, observed
  worst case 6,888 iters); Stern's hardest targets are truncated at `max_B = 2,975` (l8) /
  `5,001` (l9) — *below* LB's observed worst case — with 10/64 and 13/64 censored. Right-
  censored Stern targets (treated as `>cap`) can only have biased the **Stern median
  downward**. A fairly-observed `C_Stern` would move **up**, which could erase or flip the
  2% Stern edge. (The median itself is unbiased — censoring is far above the 32/33 median
  rank — but the *cross-attacker comparison* is not fair.)
- The harness flags the regime `model_deviation` for exactly this point.

So the dramatic v3 non-monotone spike (St/LB `0.75 → 1.96 → 0.76`) collapses to
`0.75 → ~1.0 → 0.76`: **the 1.96 was the `l=10` handicap; the true `[160,80]w16` is a dead
heat.** This *dissolves* v3's off-pattern point; it does **not** convert it into a Stern win.

## The 2.77× precal-vs-frozen divergence — real heterogeneity, not a bug

Confirmed (audit dimension 4, passed): the divergence is **target-sample heterogeneity over
a heavy-tailed ops distribution**, not a censoring/seed/estimator defect. Decisive evidence:
**LB is a cap-immune control** — 0 censored on *both* precal (0/48) and frozen (0/64) sides,
so no cap can touch it, yet it still diverges **1.43×** purely from the differing seeds and
sample sizes over a heavy tail. The censored-median is unbiased (the median lives far below
the censoring boundary; re-censoring the precal sample at the frozen budget moves it <1%).
One design wrinkle, noted: frozen `max_B = FROZEN_MULT · precal median_first_B`, so an
under-estimating precal yields a slightly-too-tight frozen cap — but the re-censoring test
shows this contributes <1% to the median, so it does not explain the divergence. The honest
reading: **a single median form does not yet capture R2′ Stern's heterogeneity** — the
slate's pre-registered `method_still_off` branch ("a deeper distributional model … a further
slate").

## Find-vs-check gap + audits (integrity clean)

| Quantity | R2′ `[160,80]w16` |
| --- | ---: |
| Verifier (flat, `2mn+n+m`) | **25,840 ops** |
| `C_best` (near-tie, ≈ Stern `l8`) | ≈ 2.23×10⁹ |
| Find-vs-check gap | ≈ **86,000×** |

`code_valid (G Hᵀ=0)` = true; `labels_wt_ok` = true; **`code_identity` match = true**
(precal and frozen regenerated the same `(G,H)`, digest `7c058d24…`); **all witnesses valid**
(`He*=z ∧ wt≤τ`, re-checked from public `H,z,τ` over *all* successes for LB, Stern l8, l9);
attackers received **only `z`** (labels scoring-only); manifest emitted + hashed before any
attacker; verifier ≪ every attacker (no 6.1/6.4). Determinism: seed-pinned (`PYTHONHASHSEED=0`),
the identical pipeline byte-identical on the harness-test re-run.

## Methodology open problems (for a future slate)

1. **Heavy-tail prediction.** Neither mean (v3) nor median (v4) predicts pathologically-
   heterogeneous `[160,80]w16` Stern within factor 2 (mean 5.32×, median 2.77×). A deeper
   distributional model, or a much larger `T_pre` with a fair (shared, uncensored-to-a-common-
   `max_B`) cross-attacker budget, is the open `method_still_off` work.
2. **Slate internal inconsistency (disclosed).** The v4 slate's Gates row scopes "method
   validation" to rungs 1/3 only, while its verdict branch defines `method_validated` over
   *all* points incl. R2′. A future slate should fold the R2′ `within_factor2` into a single
   all-points verdict so `method_validated` cannot be emitted while a required point fails.
3. **Near-tie outcome bin.** The binary `Stern_wins`/`LB_wins` rule auto-promotes a sub-5%-
   margin, one-sided-censored result to a "win." Add a `near_tie / inconclusive` bin (record
   margin + censoring alongside the label).

## Adversarial audit

A 5-dimension Workflow audit (measurement-validity, resolution-honesty, method-validation-
soundness, divergence-diagnosis, integrity-boundary; 14 agents) returned **sound to write
up** with 6 confirmed *framing* findings (all in resolution-honesty + method-validation-
soundness; zero measurement/integrity/privilege defects). Every flaw was already self-flagged
in the machine-readable artifacts (`manifest.json` pairs `Stern_wins` with `model_deviation`;
`prediction_vs_measured.json` carries `within_factor2=false`; `valid_iteration_audit.json`
carries the censoring asymmetry). The corrections — near-tie framing, dual-scoped method
validation, real-heterogeneity divergence, full censoring/seed disclosure — are all applied
above. The auto-generated `V4_SUMMARY.md` / `scaling_summary_v4.json` headline a cleaner
`method_validated` + `Stern_wins` story than is warranted; **this receipt supersedes them.**

## Claim boundary

`C_best` is an upper bound vs the tested classes (LB, Stern); Prange not measured here.
No cryptographic one-wayness claim; verification not claimed "in P"; **no progress on
P vs NP**; op-count is the cost, wall-time diagnostic. v4 hardens methodology and
re-interprets one v3 point; it does **not** enlarge the scientific claim.

## What this earns

A scrupulously-bounded methodology result: the **median calibration closes v3's mean-vs-
median error as an in-sample retrofit** (rungs 1/3 within factor 2) but is **not yet a
universal predictor** (the fresh R2′ heavy-tail regime missed 2.77×). And v3's lone off-
pattern point is **explained**: the dramatic `[160,80]w16` LB-win was an `l=10` selection
artifact; with the proper window the regime is a **dead heat** — dissolving the apparent
non-monotonicity rather than confirming or reversing it. The certificate ladder's measured
science (v1–v3) stands unchanged; v4 sharpens how its predictions are made and read.
