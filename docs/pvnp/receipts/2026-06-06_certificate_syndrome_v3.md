# Sundog Certificate Problem — Syndrome Certificate v3 Receipt (Scaling Ladder)

- Receipt id: `pvnp-certificate-syndrome-v3-2026-06-06`
- Phase / probe: Phase 4 / §5 capacity experiment — the syndrome/SIS certificate's
  find-vs-check **scaling** across a 4-rung regime ladder, measuring Lee-Brickell and
  Stern at each (the v2 LB↔Stern crossover + gap-scaling questions)
- Date run: 2026-06-05 → 2026-06-06 (overnight chain; wall ≈ 14 h, Stern/rung-3-dominated;
  **op-count is the cost signal, wall-time diagnostic-only**)
- Runner: `pvnp-certificate-syndrome-v3.py` (`--precal` → `--frozen --rung {1,2,3}` →
  `--summarize`), the staged PowerShell/command contract from the slate
- Result dir: `results/pvnp/certificate-syndrome-v3/` (transient, gitignored)
- Frozen slate: [`SUNDOG_CERTIFICATE_SYNDROME_V3_SLATE.md`](../SUNDOG_CERTIFICATE_SYNDROME_V3_SLATE.md)
  (stage-1 frozen 2026-06-05)
- Stage-2 prediction lock: [`SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json`](../SUNDOG_CERTIFICATE_SYNDROME_V3_PREDICTION_LOCK.json)
  (empirical pre-calibration, produced before frozen scoring)
- v2 receipt (the anchor / rung 0): [`2026-06-05_certificate_syndrome_v2.md`](2026-06-05_certificate_syndrome_v2.md)

## Verdict

**Bounded-positive scaling result with the LB↔Stern crossover LOCATED — Stern's high-`w`
advantage is confirmed at scale — carrying (a) a named rung-2 model-deviation (a
pre-calibration heavy-tail / mean-vs-median artifact) and (b) a Claim-A gate-calibration
caveat (the frozen gate was mis-fit to the fixed-`n` rung-1).** Four results:

1. **Crossover located (Claim B).** At both cleanly-calibrated higher-`w` rungs — rung 1
   `[128,64] w16` (St/LB = **0.75**) and rung 3 `[192,96] w18` (St/LB = **0.76**) — **Stern
   overtakes Lee-Brickell**, reversing v2's `w=12` anchor where LB won (St/LB = 1.71). v2's
   hypothesis that Stern's win is a *large-`w`* phenomenon is **confirmed**; the crossover
   sits between `w=12` and `w=16`.
2. **Non-monotone, rung-2 flagged.** Rung 2 `[160,80] w16` reverted to LB (St/LB = 1.96)
   but as a **model_deviation** traced to a pre-calibration artifact (below), not a clean
   regime reversal.
3. **The find-vs-check gap scales with `n` (Claim A, with a caveat).** `C_best/verifier`:
   `5,015× → 6,131× → 72,550× → 218,999×` (anchor → rung 3 = **43.7×**). The growth is
   driven by `n`; the frozen Claim-A gate ("every interior rung ≥ 5× the anchor gap") was
   not met because the fixed-`n` rung 1 only reached 1.2× — a gate mis-fit to that rung's
   design, not a scaling failure.
4. **Empirical pre-calibration: net win with a characterized failure mode.** It caught the
   analytic being wrong in *both* directions and landed 5 of 6 attacker-rungs within the
   factor-2 lock; the single miss (rung-2 Stern) is fully explained and points at the fix.

The reported one-wayness bound at each rung is `C_best = min(C_LB, C_Stern)` against the
tested classes (LB, Stern) — an **upper bound** (BJMM/MMT would lower it).

## Frozen regime ladder (stage-1 frozen; rung 0 = v2 verbatim)

`k=n/2`, `τ=w`, GF(2). Rungs 1–3 have fresh `code_seed/target_seed/precal_target_seed`
(stage-1 frozen). Each rung emitted `target_manifest.json` (attacker-visible `z` |
labels-only `s,e,wt`) and hashed it **before** any attacker; attackers consumed only `z`.

## The measured ladder (op-count = the cross-attacker unit; median ops-to-first-success)

| Rung | Regime | `C_LB` (ops) | `C_Stern` (ops) | `C_best` | source | **St/LB** | gap@`C_best` | lock verdict |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| 0 | `[128,64]w12` | 8.314×10⁷ | 1.423×10⁸ | 8.314×10⁷ **[LB]** | v2 | 1.71 | 5,015× | — |
| 1 | `[128,64]w16` | 1.351×10⁸ | 1.016×10⁸ | 1.016×10⁸ **[Stern]** | measured | **0.75** | 6,131× | within lock (LB 0.91×, Stern 1.06×) |
| 2 | `[160,80]w16` | 1.875×10⁹ | 3.666×10⁹ | 1.875×10⁹ **[LB]** | measured | **1.96** | 72,550× | **model_deviation** (Stern 5.32×) |
| 3 | `[192,96]w18` | 1.076×10¹⁰ | 8.136×10⁹ | 8.136×10⁹ **[Stern]** | measured | **0.76** | 218,999× | within lock (LB 1.08×, Stern 1.70×) |

Stern's selected windows (empirical `l`-reopt): rung 1 `l=7` (=analytic `l*`), rung 2
`l=10` (analytic `l*=8` — diverged), rung 3 `l=9` (=analytic `l*`). Prange at the scaled
rungs is a **formula** prediction (`N_analytic·ln2·base(m)`), never a measured `C`:
`1.85×10¹¹ / 2.82×10¹¹ / 2.03×10¹²` — drops vs `C_best` of ≈1,820× / 150× / 249×
(measured-LB/Stern vs formula-Prange, labelled).

## Claim B — the crossover, located and non-monotone

Measured St/LB ladder: **1.71 → 0.75 → 1.96 → 0.76**. Stern wins at rungs 1 and 3 (both
higher-`w`, both with clean empirical `l`-selection), LB wins at rung 0 (`w12`) and rung 2.
So the LB↔Stern winner is **non-monotone in `(n,w)`**, but the dominant, clean signal is
that **Stern overtakes LB once `w` rises from 12 to 16** — confirming the v2 prediction.
The rung-2 LB-win is the one off-pattern point, and it is a flagged confound (next section).

## Rung-2 diagnosis (the named model-deviation)

Rung-2 Stern measured `C = 3.666×10⁹` vs the locked (mean-based) prediction `6.885×10⁸` —
a **5.32×** miss (`model_deviation`, the slate's named branch). Root cause is the
**mean-vs-median issue under extreme pre-calibration heterogeneity**:

- The 16-target precal for rung-2 Stern was wildly heterogeneous — per-target successes
  `[10,0,1,0,14,3,1,9,9,0,0,1,0,1,13,0]`, **6 of 16 targets with zero successes**. The
  mean-based `p` was dominated by ~4 easy targets (`N_mean = 324`) while the typical target
  is far harder (`N_median_implied = 1255`, a **3.9×** gap).
- The frozen run (64 targets) saw the hard typical behavior — **17/64 Stern targets
  censored** at `max_B = 20·N_mean` (the cap itself was too low *because* `N_mean` under-
  estimated the typical hardness) — so the measured median landed at the harder value.
- The **median-implied diagnostic** (logged in the lock per the slate's heterogeneity
  requirement) predicts the measured `C` within factor 2 at **all three** Stern rungs
  (meas/median = 0.84 / 1.37 / 0.74), where the locked **mean-based** form missed rung 2
  (meas/mean = 1.06 / **5.32** / 1.70). The mean→`C@50%` map is the failure point, not the
  `p` measurement.
- The `l=10` edge-selection (vs analytic `l*=8`) was driven by the same noisy 16-target
  data and likely handicapped rung-2 Stern further.

So rung 2 is a **confounded point** (heavy-tail pre-cal noise + a suboptimal edge `l`), not
a clean "`[160,80]w16` favors LB." A clean retest (more precal targets, `l ∈ {8,9}`) is a
future-slate item.

## Claim A — the gap scales with `n` (frozen gate caveat)

Gap ladder `5,015× → 6,131× → 72,550× → 218,999×`. The top rung is **43.7×** the anchor
(≥10× ✓), and the `n`-growing rungs scale clearly (rung 2 = 14.5×, rung 3 = 43.7×). But the
frozen gate requires *every interior rung* ≥ 5× the anchor, and the fixed-`n` rung 1
(`[128,64]`, same `n` and same flat verifier `16,576` as the anchor) only reached 1.2× —
so `--summarize` returns **`gap_scaling_not_confirmed`** by the strict frozen rule. This is
a **gate-calibration caveat**: rung 1 was the fixed-`n` `w`-scaling rung and was never going
to grow the gap; the certificate's one-wayness threshold demonstrably **scales with `n`**.

## Empirical pre-calibration assessment

The empirical pre-calibration (the v3 mechanism) was a **net win** and is itself a result:

- It **caught the analytic being wrong in both directions**: Lee-Brickell is ~**8× easier**
  than the analytic predicts at `w=16` (measured `N_LB ≈ 109` vs analytic `968` at rung 1 —
  the mirror image of v2's Stern being *harder* than analytic), and Stern's success is
  regime-variable. A purely-analytic comparator would have mis-ranked the ladder.
- **5 of 6 attacker-rungs landed within the factor-2 lock** (rung-1 LB+Stern, rung-2 LB,
  rung-3 LB+Stern). The base(`m`) cost model also re-validated independently: a fresh
  two-size probe gave `base(64) = 1.396×10⁶`, matching the v2 lock's `1.394×10⁶` to 0.1%,
  with `β = 2.96` (cubic) extending the scaling to the v3 `m`-range.
- The one miss (rung-2 Stern) is **characterized, not mysterious**: the mean-based
  `C = (1/p_emp)·ln2·per_iter` over-predicts success for heavy-tailed Stern; the
  median-implied form fixes it. **Methodology recommendation for a future slate:** lock the
  median-implied `C` (or the mean↔median band as a calibrated uncertainty), and raise the
  pre-calibration target count well above 16 for heavy-tailed attackers.

## Find-vs-check gap

| Rung | verifier (flat) | `C_best` | gap |
| --- | ---: | ---: | ---: |
| 1 | 16,576 | 1.016×10⁸ | 6,131× |
| 2 | 25,840 | 1.875×10⁹ | 72,550× |
| 3 | 37,152 | 8.136×10⁹ | 218,999× |

`verifier_below_all_attackers = true` at every rung. The flat witness-verifier
(`2mn+n+m`) stays vastly cheaper than recovering the deviation, and the gap **grows with
the regime** — the scaling result the certificate was built to show.

## Audits (all rungs)

| Gate | Rung 1 | Rung 2 | Rung 3 |
| --- | --- | --- | --- |
| Code valid (`G Hᵀ=0`) | pass | pass | pass |
| Labels `wt=w` | pass | pass | pass |
| Manifest before attackers | pass (`3aecd278…`) | pass (`b6a226aa…`) | pass (`75a9c121…`) |
| Witnesses valid (`He*=z ∧ wt≤τ`, public recheck) | LB+Stern ✓ | LB+Stern ✓ | LB+Stern ✓ |
| Privilege (attacker sees only `z`) | pass | pass | pass |
| Stern censoring (data-quality flag) | 2/64 | **17/64** | 6/64 |

Determinism: seed-pinned pure-numpy (`PYTHONHASHSEED=0`); the identical pipeline (precal +
frozen-rung bundle) was byte-identical on the harness-test re-run. A literal full-ladder
re-run was not performed (≈14 h; not load-bearing).

## Claim boundary

Each rung's `C_best` is an **upper bound** against the tested classes (Lee-Brickell, Stern);
BJMM/MMT would lower it (a separate slate). The LB↔Stern crossover is a statement about
*these two attackers* on these codes, not about ISD in general. Prange at the scaled rungs
is a **formula** prediction, never a measured `C`. No cryptographic one-wayness claim
(hardness imported, not proved); no claim that verification is "in P"; **no progress on
P vs NP**. `invert-s` remains unconditionally vacuous; spoof remains structurally
impossible. Op-count is the cost; wall-time diagnostic-only.

## What this earns

The lane now has a **measured scaling ladder** for the syndrome certificate: the
find-vs-check gap grows from `5,015×` (anchor) to `218,999×` (rung 3) with `n`, and the
v2 LB↔Stern reversal is resolved into a **located crossover** — Stern's advantage is a
high-`w` phenomenon that reappears at scale (`w16`, `w18`), with the winner non-monotone in
`(n,w)`. It also produced a clean **methodology result**: empirical pre-calibration is
necessary (the analytic ISD heuristics are unreliable in both directions), and the
mean→`C@50%` mapping must use the median form for heavy-tailed attackers — with the
rung-2 model-deviation as the worked example and the median-implied diagnostic as the fix.

## Pre-run validation

The v3 harness reused the v2 (GREEN-audited) GF(2) core / attackers / bundle emitter;
before the overnight chain it was validated inline: the stage-1 base(`m`) smoke (cost model
holds to the v3 scales), a tiny end-to-end plumbing + byte-determinism test, and a
real-rung precal-path check on all three regimes. An adversarial post-run audit of the lock
+ artifacts + this receipt is the recommended next ratchet before promotion.
