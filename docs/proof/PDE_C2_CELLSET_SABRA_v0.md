# PDE C2 Cell Set v0 — Sabra Shell-Model Burst-Detection Instance

> **Pre-registration / design proposal for sign-off**, filed 2026-05-29.
> First numerical artifact for Candidate 2, instantiating
> [`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
> (which pinned the research object, channel taxonomy, baselines, task,
> and two-sided negative). This file pins the concrete numerics the
> scoping deferred. **Status: proposed, not built, not run.** §9
> specifies the harness for build; §10 lists the sign-off decisions. No
> shell-model integrator, signature, or baseline code is written and no
> numerical run is launched until this design is signed off.
>
> **Carry-over from C1 (load-bearing).** The C1 campaign's central lesson
> was that a *fixed-percentile threshold on an intermittent quantity goes
> vacuous across regimes* (the v6 `DEFERRED_VACUITY` at G=300). Shell
> models are the canonical intermittent system, so `E_burst` is pinned
> using the C1 portable-objective fix: a **held-out quantile** targeting
> a pre-registered base rate, with leakage-safe splits, plus a
> **base-rate gate** (§8) that defers a cell whose burst rate is
> degenerate rather than reading a detector verdict off it.

## 1. Sabra model (pinned; reviewer-checkable)

Standard Sabra shell model on a geometric ladder `k_n = k_0 λ^n`:

```text
du_n/dt = i ( a k_n u*_{n+1} u_{n+2}
            + b k_{n-1} u*_{n-1} u_{n+1}
            + c k_{n-2} u_{n-1} u_{n-2} )
          − ν k_n^2 u_n + f_n
```

Pinned (canonical Sabra; flagged for reviewer confirmation in §10):

| parameter | value |
|---|---|
| shells `N_shells` | 22 |
| ladder base `λ` | 2 |
| `k_0` | `2^-4` |
| Sabra coefficients | `a = 1`, `b = −0.5`, `c = −0.5` (ε = 1/2) |
| viscosity `ν` (headline) | `1e-7` |
| forcing | constant `f = (1+i)·0.005` on shell `n = 1`, else 0 |
| integrator | 4th-order Runge–Kutta with integrating-factor on the `−ν k_n^2` linear term |
| time step `dt` | pinned in §9 to resolve the fastest shell (`dt ≲ 1/(ν k_N^2)` and `≲ 1/(k_N |u_N|)`) |
| warm-up | discard the first `T_warm` (pinned §9) to reach the stationary attractor |
| seed | `20260528` |

These are standard, not Sundog-original. The coefficients and
energy-conserving structure are reviewer-checkable against the
Mailybaev / L'vov–Procaccia Sabra literature (citations in the
lit-pass memo).

## 2. Burst label `E_burst` (held-out quantile — the C1 fix)

`B(t) = 1` iff `max_n |u_n(s)|^2 > E_burst` for some
`s ∈ (t, t + τ_burst]`.

**`E_burst` is NOT a fixed absolute level and NOT a percentile of the
training data.** It is the `q_burst`-quantile of the instantaneous
`max_n |u_n(s)|^2` over a **held-out calibration window** `C_burst`,
disjoint from train/val/test (see §7):

```text
E_burst = quantile_{q_burst}( { max_n |u_n(s)|^2 : s ∈ C_burst } ),
q_burst = 0.98.
```

`q_burst = 0.98` targets a rare-but-not-vanishing instantaneous
exceedance; the *query-time* base rate (fraction of `t` with `B(t)=1`
over the τ_burst lookahead) is the gated quantity (§8), expected in the
low tens of percent. `τ_burst` pinned in §9.

Calibrating on held-out data (not train) keeps the label leakage-safe
and regime-portable: the same construction gives a comparable base rate
across `ν` cells instead of going vacuous at higher intermittency.

## 3. Channels and signature representation

Per the scoping channel taxonomy:

- **Tier 0 (headline):** shell-wise log-energy
  `c0_n(t) = log(|u_n(t)|^2 + 1e-30)`, `n = 1..N_shells` → 22 channels.
- **Tier 1:** Tier 0 + inter-shell energy transfers `T_n(t)`.
- **Tier 2:** Tier 1 + hidden self-similarity coordinates `χ_n(t)`
  (Mailybaev 2022).

**Signature representation: log-signature to level `D = 2`** over a
rolling window of length `W` (pinned §9), computed on the (standardised)
channel vector. Log-signature, not full signature, to control
dimension: full signature to level 2 on 22 channels is 22 + 22^2 = 506
terms; the log-signature is much smaller and is the standard
multivariate-series choice. The detector is a logistic / shallow
classifier on the log-signature features. **Matched-budget rule (§4)
caps the detector's trainable parameter count so a richer tier or deeper
signature gets no free advantage.**

Channel standardisation (mean/scale) is fit on the **training split
only** and applied to val/test (leakage-safe).

## 4. Detector and matched-budget baselines

Detector: classifier on Tier-`t` log-signature features predicting
`B̂(t)`.

Baselines (scoping §Baselines), each producing a `B̂(t)` score on the
**same windows, same splits, same labels**:

1. DMD/Koopman spectral features on the window;
2. critical-slowing-down (variance + lag-1 autocorrelation of total
   energy `Σ_n|u_n|^2`);
3. recurrence lacunarity;
4. Rényi entropy (order-q) of a coarse-grained embedding.

**Matched-budget rule (enforced before any test-split evaluation):**
same data/splits; equal hyperparameter-search trial count `H` (pinned
§9); each baseline's trainable parameter count ≥ the signature
detector's at the evaluated tier.

## 5. Cells (pinned set; no post-hoc additions/removals)

1. **Headline:** Sabra, `ν = 1e-7`, Tier 0.
2. **Tier ablations:** same cell at Tier 1 and Tier 2.
3. **Re-extrapolation (held-out `ν`):** `ν = 3e-8` (higher Reynolds),
   detectors trained on the `1e-7` cell and evaluated here without
   retraining — overfit audit (`PDE-C2-NEG-B` surface).
4. **GOY cross-check:** the headline Tier-0 cell re-run on the GOY shell
   model, same channels, same matched budget.
5. **Operating-envelope grid:** `ν ∈ {3e-7, 1e-7, 3e-8}` — inside/edge
   cells; outside-envelope behaviour (detector ≈ baselines or worse)
   must be reported, not hidden.

## 6. Evaluation surface

Pareto frontier of **lead-time** (mean `s*−t` over true positives) vs.
**false-positive rate** (fraction of `B(t)=0` query times flagged).
Detector strictly inside the matched-budget baseline frontier → win;
on it → tie; outside → `PDE-C2-NEG-A`. Operating point pinned §9 before
test-split evaluation; sliding it post hoc → `PDE-C2-NEG-B`.

## 7. Splits (leakage discipline)

The stationary trajectory is partitioned into **contiguous, disjoint
blocks with decorrelation gaps** (not random per-sample splits, which
leak under autocorrelation):

```text
[ warm-up | C_burst (label calib) | gap | train | gap | val | gap | test ]
```

- `E_burst` from `C_burst` only (§2);
- channel standardisation, classifier fit, baseline fit, and
  hyperparameter search on `train`/`val` only;
- operating point chosen on `val`;
- `test` is read exactly once, after everything is pinned.

Block lengths and gap (≳ a few integral times) pinned §9.

## 8. Base-rate gate (new; the C1 vacuity lesson)

**Before** reading any detector-vs-baseline comparison, confirm the task
is non-degenerate on the `test` split:

```text
base_rate_test = mean_t B(t)  ∈  [0.05, 0.40].
```

- Outside the band → **`PDE-C2-DEFERRED-BASERATE`**: the burst objective
  is degenerate at this cell (too rare to detect, or near-trivial). This
  is a non-verdict, parallel to C1's `DEFERRED_VACUITY`. Do **not** retune
  `q_burst`, `τ_burst`, or `E_burst` to rescue a cell after reading
  (that is `PDE-C2-NEG-B`); re-posing requires a new pre-registered cell.

This gate is the explicit import of the C1 v6 finding: an intermittent
threshold objective can go vacuous, and that must be caught as a
deferral rather than mis-read as a detector result.

## 9. Harness spec (for build — not yet built)

A new, self-contained module (separate from the Kolmogorov harness):

- **Sabra/GOY integrator** (RK4 + integrating factor); `dt`, `T_warm`,
  block lengths, gap pinned here at build with a CFL-style check.
- **Label builder**: `max_n|u_n|^2`, `E_burst` from `C_burst`, `B(t)`
  over `τ_burst`.
- **Channels**: Tier 0/1/2 + standardisation fit on train.
- **Signature**: log-signature level 2 (library decision in §10).
- **Detector + 4 baselines** with the matched-budget harness.
- **Receipts**: `manifest.json`, per-cell Pareto points, a
  `PDE_C2_RESULTS.md` receipt with the base-rate gate, the
  detector-vs-frontier verdict, and the named branch.
- Smoke preset (tiny `N_shells`/short trajectory, manual-override →
  non-verdict) before any verdict-bearing cell, per house discipline.

Concrete `dt`, `T_warm`, `W`, `τ_burst`, block lengths, gap, `H`, and
operating point are pinned in a §9 table at build time **before** any
run, and listed in the receipt.

## 10. Open sign-off decisions

1. **Sabra parameters (§1).** Confirm the canonical coefficients,
   `N_shells = 22`, `ν = 1e-7`, forcing scheme, and integrator against
   the shell-model literature — this is the spot most needing a
   shell-model-literate check (a candidate criterion-(c) reviewer could
   confirm before the run).
2. **Signature library.** `iisignature` (light, numpy-friendly) vs a
   hand-rolled level-2 log-signature (no dependency, fully auditable) vs
   `signatory` (torch). Recommend hand-rolled level-2 log-signature for
   auditability and zero dependency, unless `iisignature` is already
   available.
3. **`q_burst = 0.98` / base-rate band `[0.05, 0.40]`.** Confirm the
   burst rarity and the gate band.
4. **Compute budget.** `N_shells = 22`, `ν = 1e-7` over a long stationary
   trajectory + 4 baselines + hyperparameter search is heavier than the
   Kolmogorov runs; confirm trajectory length / `H` so a full cell stays
   in the tens-of-minutes range.
5. **Build trigger.** On sign-off I build the self-contained C2 harness,
   smoke-test (integrator energy-balance + label + signature + baseline
   plumbing), then run the headline cell first.

## 12. Result (2026-05-29) — PDE-C2-DEFERRED-BASERATE (non-stationarity)

The objective-validity layer ran (`results/proof/c2-sabra-headline-baserate/`,
~32 min, 6.3M steps; integrator energy-conservation self-test passed
first). The base-rate gate **deferred**, with a diagnostic pattern:

| block | burst base rate |
|---|---:|
| train | **0.138** (in band) |
| val | 0.000 |
| test | 0.000 |

`E_burst = 4.43` (held-out q=0.98 of calib max-shell-energy; calib
median 3.12, p99 4.46).

**Diagnosis: block-dependent base rate ⇒ the trajectory is not
stationary / not representatively sampled across the labelled span.** A
simple degenerate objective would give a uniform rate; a rate that runs
0.138 → 0 → 0 across temporally-ordered blocks means either (a) slow
energy drift past the 2M-step warmup, or (b) — more likely for a shell
model — the **burst recurrence time is comparable to the block
lengths**, so `train` caught burst activity and `val`/`test` did not.
This is the C1 intermittency lesson in a new guise: shell-model bursts
are intermittent enough that *block representativeness* fails at these
trajectory lengths. The integrator is verified correct (energy
conservation), so this is a sampling/stationarity issue, not an
integration bug.

**Disposition.** `PDE-C2-DEFERRED-BASERATE`, a non-verdict. Per §8 / the
scoping, **no `q_burst` / `τ_burst` / `E_burst` rescue on this cell**
(that would be `PDE-C2-NEG-B`). The gate did its designed job: it caught
a non-stationary cell **before** any detector-vs-baseline comparison was
built, so no compute was wasted on a comparison over unrepresentative
labels.

**Re-pose direction (a new pre-registered v1 cell, not a v0 retune).**
A stationary, representatively-sampled Sabra cell needs:

1. a **per-block stationarity gate** — require the base rate to be
   consistent across `train`/`val`/`test` (e.g. each in band *and*
   pairwise within a tolerance), not just `test` in band; a
   block-dependent rate files `PDE-C2-DEFERRED-NONSTATIONARY`;
2. **much longer blocks** (many burst recurrence times each), and/or a
   longer warmup, chosen so each block contains O(10²) burst events;
3. **per-block diagnostics** (mean/median max-energy, burst count) in the
   receipt to distinguish drift (a) from rare-cluster intermittency (b);
4. possibly a forcing/viscosity pair verified to give a statistically
   steady cascade before labelling.

This is a v1 cell-set with its own pre-registration, mirroring the C1
v6 → v1 portable-objective move (deferral → diagnose → new pinned cell,
never a same-cell retune).

## 11. Cross-references

- [`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  — the scoping this instantiates; research object, channel taxonomy,
  baselines, two-sided negative, promotion path.
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md)
  — the C1 portable-objective fix whose lesson is imported into §2/§8
  (held-out quantile label + base-rate gate).
- [`../NAVIERSTOKES_LITPASS_MEMO.md`](../NAVIERSTOKES_LITPASS_MEMO.md)
  — Sabra / hidden-self-similarity / baseline citations.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — ledger;
  Candidate 2 promotion criteria and the `PDE-C1-NEG` re-framing rule.
