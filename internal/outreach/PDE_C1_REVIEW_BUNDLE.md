# PDE C1 Finite-Galerkin Separation — Self-Contained Review Bundle

> **This is one file containing the whole reviewer packet**, assembled by
> `scripts/build-c1-review-bundle.mjs` from the on-disk artifacts. It is the
> attachment the outreach email refers to as the packet. Send this (or its PDF
> rendering) plus the email from
> `docs/proof/PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`.

**Status: PROVISIONAL / UNPROMOTED, gated on external review.** This is a
structural separation in a finite-dimensional Galerkin truncation of 2D
Navier–Stokes. It is **not** a Navier–Stokes existence/smoothness claim, **not**
a new determining-modes theorem, **not** a statement about the
infinite-dimensional NSE attractor, and **not** a Clay-problem claim.

## The one-sentence ask

On the sampled invariant measure of a 2D Kolmogorov-flow Galerkin model
(`k_f = 2`, `K = 3`, `G ∈ {200, 300}`), the low-band signature `Φ_K` is
**state-insufficient** (certified non-injective) yet **control-sufficient** for a
registered decision (`Φ_K`-measurable up to a measure-`δ ≈ 0.037` boundary
layer). Is this a genuine finite-Galerkin separation between state-reconstruction
and action-sufficiency, or ordinary observer / data-assimilation / LES-closure
behavior that should not be made load-bearing?

## The four locked questions

1. **State-insufficiency language.** Given the twin-state certificate is
   sampled-SRB / finite-Galerkin (not an exact attractor theorem), is
   "`Φ_K` non-injective / state-insufficient on `supp μ`" reasonable, or should
   it be weakened to a finite-sample claim?
2. **Control-sufficiency language.** Is the kNN/disintegration reading of
   "`π*` factors through `Φ_K` up to `μ`-measure zero" (local action-mixing
   `mean_minority → 0`, plus paired action-constancy on the certified
   non-injective pairs) mathematically fair, or overstated?
3. **Objective legitimacy.** Is a held-out look-ahead-max quantile a defensible
   regime-portable proxy action, given the older burn-in-percentile trigger went
   vacuous at `G = 300` and was replaced (not retuned)?
4. **Real separation vs renaming.** A genuine "a sub-determining mode set can be
   control-sufficient without being state-reconstructive" separation, or ordinary
   functional-observability / data-assimilation / LES-closure behavior?

## The answer menu (a paragraph is plenty — a negative reply is the most valuable)

```text
The framing seems conservative / basically right.
Weaken or rename X (e.g. "state-insufficient" → finite-sample only).
The control-sufficiency / fiber language is too strong because Z.
This is standard observer / data-assimilation, cite B.
The objective is implicitly a function of Φ_K; the separation is vacuous because W.
```

## Anti-folklore guard (worth pre-empting)

The natural reviewer reflex — "this is just LES / closure: of course coarse
energy is predictable from coarse state" — is exactly what the non-injectivity
certificate is meant to foreclose: LES / AIM / closure assume or derive
**state-sufficiency** via slaving; C1 *certifies state-insufficiency* (twin
states) and shows the **decision** survives the genuinely unresolved fine state.
Please stress-test that distinction specifically.

---

*Read order below: separation statement → result → two adjudicators → three run
receipts (verbatim). Each section is the unmodified on-disk artifact.*


==============================================================================
## [ PRIMARY · separation statement ]
*source: `docs/proof/PDE_C1_SEPARATION_STATEMENT.md` · sha256:98eea9515f52*
==============================================================================

# PDE C1 — Finite-Galerkin Structural Separation (reviewer-facing statement)

> Proof-track consolidation, 2026-05-29. Lifts the Postulate-1 reading
> ([`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md))
> into a single self-contained **finite-Galerkin** separation statement with
> its empirical witness, an anti-vacuity ledger, and a determining-modes
> comparator. This is the artifact to put in front of a PDE reviewer.
>
> **Scope.** This is a structural separation in a finite-dimensional
> dynamical system that is a Galerkin truncation of 2D Navier-Stokes. It is
> **not** a Navier-Stokes existence/smoothness claim, **not** a new
> determining-modes theorem, and **not** a statement about the
> infinite-dimensional NSE attractor. Status: **PROVISIONAL, UNPROMOTED**,
> gated on external review.

## 1. The statement

Let `S_N` be the 2D Kolmogorov flow truncated to a fixed Galerkin cutoff
(`N = 16`, `32 x 32` grid, pseudo-spectral vorticity), forced at `k_f = 2`,
at Grashof number `G`. Let `mu` be the empirical (sampled SRB-like) invariant
measure on its attractor. Let

```text
Phi_K : state -> R^d      d = 18  (the K = 3 low-Fourier band, 9 complex modes)
```

be the low-band signature, and let `J_q` be the registered control objective
(Section 3). Write `pi*` for the `J_q`-optimal binary selector
(`{no_op, damp_low_band}`).

> **Claim (regime-2 separation).** On `(S_N, mu)` at `G in {200, 300}`,
> `k_f = 2`, `K = 3`:
>
> 1. **State-insufficient.** `Phi_K` is **not injective** on `supp mu`:
>    there is positive `mu`-mass of state pairs with `Phi_K`-distance
>    `<= epsilon_K` and complementary high-mode (`Q_K`) separation
>    `>= delta_H`. *(twin-state certificate, CERTIFIED both regimes)*
>
> 2. **Control-sufficient.** `pi*` factors through `Phi_K` up to
>    `mu`-measure zero: on `epsilon_K` signature-balls the proxy action is
>    near-constant (`mean_minority -> 0`), and on the **certified
>    non-injective pairs themselves** the action disagreement
>    `D_witness <= delta_action`. *(kNN convergence POSITIVE + paired
>    fiber-constancy exhibit)*
>
> The smallest control-sufficient signature for `J_q` is therefore strictly
> smaller than any state-reconstructive signature on this support: `Phi_K`
> collapses states that determination would still separate, yet those
> collapsed states share the `J_q`-optimal action.

This is precisely Reading-2 regime 2 of the Postulate-1 sidecar — "state
insufficient, control sufficient" — the non-vacuous target — now stated as a
finite-dimensional fact with a pre-registered numerical witness.

## 2. Decision domain, signature, measure

| Object | This cell |
| --- | --- |
| Decision domain `X` | Galerkin state of `S_N` (`d_state = 440` real DOF) |
| Signature `Phi_K` | low-Fourier band `K = 3`, `d = 18` real components |
| Null coordinate `Q_K` | complementary high modes, `d_high = 422` |
| Measure `mu` | empirical invariant measure, `50000` samples, interval `50` steps after `10^5` burn-in |
| Objective `J_q` | held-out quantile-calibrated low-band overshoot trigger (Section 3) |
| Selector `pi*` | `1[ lookahead_max(state) > E_max ]`, `E_max` = held-out `(1 - q)`-quantile |

The signature resolves **18 of 440** real degrees of freedom (~4%); **422**
are unresolved. That gross resolution gap is what makes non-injectivity
plausible and the separation non-trivial.

## 3. The objective is not a function of the signature (anti-tautology)

The decisive anti-vacuity property: `J_q` is defined on the **full state's
future**, not on `Phi_K`. The label is

```text
a(x) = 1[ max_{t <= tau} E_low(flow_t(x)) > E_max ],
```

a `tau`-step look-ahead of the *full nonlinear flow* `flow_t`, thresholded at
a quantile `E_max` calibrated on a **disjoint** held-out window. So
control-sufficiency — that this full-state look-ahead label is nonetheless
`Phi_K`-measurable a.e. — is a **dynamical fact about `S_N`**, not a
definitional consequence of how the objective was written. If the label were
a function of `Phi_K` directly (e.g. instantaneous `E_low = ||Phi_K||`-band
energy), the claim would be vacuous; it is not, because the label is a
forward-time excursion of the full state.

## 4. Anti-vacuity ledger

Each way the separation could be trivial, and why it is excluded here.

| Vacuity mode (from the sidecar) | Excluded because |
| --- | --- |
| Objective is full-state tracking / reconstruction | `J_q` is a scalar binary trigger, not state tracking. |
| Action directly actuates all unresolved components | Actions are `{no_op, damp_low_band}`; neither reads/penalizes `Q_K`. |
| Selector separates every state-distinct fiber | Falsified by the paired exhibit: certified `Q_K`-separated pairs share the action (`D_witness <= delta_action`). |
| Determining machinery already reconstructs at `<= K` modes | Falsified internally: `Phi_K` (`K=3`) is **certified non-injective** on this support (Section 5). |
| Strictness only appears after a post-hoc objective change | `E_max` is held-out quantile-calibrated and **frozen** before scoring; no post-hoc retune (would be `PDE-C1-NEG-B`). |
| Objective is constant on the whole support (degenerate) | `damp_fraction = 0.298 (G=200) / 0.269 (G=300)` — non-trivial; the harness `delta_proxy_min` / structural-constancy gates guard this. |

The harness encodes the last three as hard gates (`DEFERRED_VACUITY`,
structural-vacuity precedence, `PDE-C1-NEG-B` on post-hoc retune), so a
vacuous instance defers or files negative rather than passing.

## 5. Determining-modes comparator (internal, no borrowed constant)

The state-reconstruction reference is supplied **internally** by a fixed-
Galerkin bracket, not by importing an asymptotic literature constant (which
would be loose at moderate `G`):

- **Lower bracket (have it).** At `K = 3` the twin-state certificate is
  POSITIVE: signature-near pairs with `Q_K` separation `p50 = 0.0154` at
  high-mode-norm median `0.233` (~6.6% of the norm) within `epsilon_K`-balls
  of radius `~0.06`. So `K = 3` provably does **not** determine state on
  `supp mu` — a sampled-support determining-modes *negative*.
- **Upper bracket (registered next run).** Increase `K` until the twin-state
  witness vanishes (injectivity returns). The smallest such `K*` is an
  internal, geometry-specific upper bound on the determining count for this
  cell; the separation lives in the gap `K = 3 < K*`. Pre-registered as the
  K-window companion (also serves the robustness sub-goal).

**Literature backdrop (qualitative).** Foias-Prodi / Foias-Temam and
Constantin-Foias-Manley-Temam establish that finitely many modes determine
2D NSE long-time dynamics, with the determining count finite and growing with
`G`. We use this only as the reason a finite `K*` exists; the *value* here is
measured, not asserted.

## 6. Theorem vs. witness — what is and is not proved

- **Provable-as-witnessed (finite-dimensional).** On `(S_N, mu)` the three
  numerical predicates — non-injectivity, neighbourhood action-constancy,
  paired action-constancy — are pre-registered, gated, and reproduced
  deterministically (seed `20260528`). Within the sampled support they
  *certify* the regime-2 separation for this cell. This is an existence
  witness in a finite-dimensional dynamical system, not a theorem schema.
- **Not claimed.** (i) The infinite-dimensional NSE attractor — `S_N` is a
  truncation; the support is sampled, not the full attractor. (ii) Any
  Millennium-problem content. (iii) A new determining-modes bound. (iv)
  Generality beyond the tested objective family / forcing geometry (`k_f = 2`)
  — that is the robustness sub-goal, in progress.

## 7. Observer-theory restatement and prior-art positioning

The separation restates cleanly in control/observer language, which is
where its novelty sits (recon:
[`PDE_C1_MECHANISM_RECON.md`](PDE_C1_MECHANISM_RECON.md)). Take the Galerkin
flow as the system, `Phi_K` as the measurement/output map, `mu` as the
invariant measure. Then C1 says:

> `Phi_K` is **decision-observable but state-unobservable** on `supp mu`:
> the full state is *not* functionally observable from `Phi_K` (certified
> non-injectivity — the output map has positive-`mu`-mass fibers), yet the
> registered safety decision `a(x) = 1[ max_{t<=tau} E_low(flow_t x) >
> E_max ]` *is* `Phi_K`-measurable up to a measure-`delta` boundary layer
> (`delta ~ D_witness ~ 0.037`).

**Pivotal claim a reviewer must bless or reject (everything else is
support):** *on the sampled invariant measure of this Galerkin 2D-NSE
system, the low-band projection `Phi_K` is non-injective (state-
unobservable) yet the registered decision is `Phi_K`-measurable up to a
measure-`delta` boundary layer (decision-observable), with `K` below the
determining-modes count.*

**Prior-art delta** — why this is an extension, not a renaming:

| Established | C1's delta |
| --- | --- |
| Functional observability is **linear / finite-dim / network / estimation** (Montanari–Motter PNAS 2022); the nonlinear extension (arXiv 2301.04108) lists turbulent observables like energy on spatially-extended systems as an **open gap** | Instantiated on a (Galerkin) **NSE attractor**, for **energy** (the named-open observable), for a **decision** objective (not estimation), with a **measured** non-injectivity certificate |
| **Determining functionals** determine the **state** (Cockburn–Jones–Titi 1997) | `Phi_K` is **below** the determining threshold (non-determining) yet decision-sufficient — the **complement** of that theory |
| **Approximate inertial manifolds** slave high→low, i.e. state-**sufficient** | C1 is **below** the AIM threshold — state-**insufficient**, certified — the opposite regime |
| **Mori–Zwanzig closure**: the unresolved coupling into the resolved energy budget is *approximable* via memory (Parish–Duraisamy 2017) | C1 **measures** that the coupling `R` is ~99% **signature-determined** (`R²(R\|Φ_K)=0.998` (G=200) / `0.990` (G=300), at the exact-function ceiling, neg-control ~0) — a local energy-budget closure `R≈R(Φ_K)` that holds *precisely where state reconstruction fails*; this is the measured mechanism, not a fitted closure (see `PDE_C1_MZ_ENERGY_BUDGET.md` §10) |

**Anti-folklore guard.** The natural reviewer reflex — "this is just
LES / closure: of course coarse energy is predictable from coarse state" —
is exactly what the non-injectivity certificate forecloses. LES / AIM /
closure assume or derive **state-sufficiency** via slaving; C1 *certifies
state-insufficiency* (twin states) and shows the **decision** survives the
unreconstructable state anyway. That is the distinction between "coarse
predicts coarse because coarse slaves fine" and "the coarse decision is
robust to genuinely unresolved fine state."

## 8. Review path and cross-references

The three reviewer questions (sidecar Section "External Review Path") map to:
(1) Reading 1 faithful to determining modes — Section 5 + lit-pass; (2)
Reading 2 a real distinction — Sections 1, 3 (anti-tautology); (3) no hidden
reconstruction in `Phi_K` — Section 4 ledger + Section 5 internal
non-injectivity. The reviewer email is staged at
[`PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md`](PDE_C1_EXTERNAL_REVIEW_EMAIL_DRAFT.md).

- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md) — the reading note this consolidates.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md) — state-insufficiency half.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) — control-sufficiency half.
- [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) — the paired exhibit composing them on the same pairs.
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) — the two-regime portable objective.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — the ledger.


==============================================================================
## [ RESULT · two-regime portable objective ]
*source: `docs/proof/PDE_C1_REGIME_GENERALITY_v1.md` · sha256:23f17da8b0b6*
==============================================================================

# PDE C1 Regime Generality v1 — Portable Objective

> **Pre-registration and result**, filed 2026-05-29. Successor to
> [`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md),
> whose G=300 probe returned `PDE-C1-RG-DEFERRED_VACUITY`: the
> kNN/twin-state machinery stayed usable, but the *registered overshoot
> proxy went near-vacuous* (`damp_fraction ≈ 0.004`) under
> heavy-tailed/intermittent energy statistics. The regime-generality
> question was therefore **not adjudicated** — the failure was objective
> construction, not fiber-locality coverage or support-level
> non-injectivity. This artifact replaces the fixed-percentile objective
> with a **regime-portable** one and re-poses the Grashof-axis generality
> test cleanly.
>
> **Status: executed.** §12 records `PDE-C1-RG-POS`: the portable
> objective passes its portability gate at G=200 and G=300, the control
> half is POSITIVE at both regimes, and the G=300 twin-state companion
> is CERTIFIED. C1 remains unpromoted because proxy faithfulness and
> external PDE review remain open.

## 1. The fix in one line

Replace "exceed the **burn-in** 95th-percentile `E_K` in lookahead"
with "be in the top `(1−q)` fraction of **own-regime** τ-excursions,"
calibrated on a **held-out post-burn-in** window. This pins
`damp_fraction = 1−q` by construction at every regime, removing the
v0 vacuity failure mode.

## 2. Why the v0 objective was not portable

`E_max` was the 95th percentile of the (last-25%) burn-in `E_K`. Two
coupled defects under intermittency:

- **Calibration/sampling mismatch.** A single burst in the burn-in
  calibration window shifts the 95th percentile far out; the
  post-burn-in sampling window is comparatively quiescent → almost no
  look-ahead window reaches `E_max`.
- **Heavy-tailed shape-dependence.** At G=200 the energy distribution is
  tight, so the 95th percentile sits near the bulk (damp_fraction 0.30);
  at G=300 it is intermittent, so the 95th percentile sits on a
  rare-burst tail (damp_fraction 0.004). The *same* percentile rule
  means different things at different regimes.

Confirmed regime property, not artifact: v2 (G=300, full-burnin E_max)
also gave sub-1% damp. The objective, not the machinery, fails to
transfer.

## 3. Portable objective (pinned)

For a state `u` with no-op τ-lookahead trajectory, define the
**lookahead-max excursion**

```text
M(u) = max_{t in [0, tau]} E_K(u(t)),     tau = 5.0 time units.
```

Calibrate the threshold on a held-out post-burn-in calibration window
`C` (disjoint from the adjudication sample `A`, see §4):

```text
E_max = quantile_q( { M(u) : u in C } ),     q = 0.70.
```

The proxy selector, applied to the adjudication sample `A`:

```text
pi_hat(u) = damp_low_band   iff   M(u) > E_max
          = no_op           otherwise,        for u in A.
```

By construction `damp_fraction` on `C` is exactly `1 − q = 0.30`; on
the disjoint stationary `A` it is `≈ 0.30` (the **portability gate**,
§6, verifies this at both regimes). `q = 0.70` is chosen so the new
objective's damp scale matches v5's emergent `0.30`, making the G=200
re-run a clean comparison to the existing witness.

Everything else is inherited from v5/v6 unchanged: `k_f = 2`, `K = 3`,
`d_K = 18`, `dt = 0.01`, `tau = 500` steps, `epsilon_K = 0.05·sqrt(2
E_max)` (now with the portable `E_max`), kNN sweep `k ∈
{10,15,20,25,30,40,50}` with `a_mm` thresholds (≤0.005 POSITIVE,
≥0.015 NEG-A), twin-state `k_twin=50`, `delta_H = max(1e-6,
0.05·median‖Q_K‖)`, gates `0.01` / `100`.

## 4. Held-out split (pinned)

The post-burn-in trajectory is partitioned into two disjoint blocks
with a decorrelation gap:

```text
calibration window C : 50,000 samples at interval 50 steps
decorrelation gap    : 5,000 steps (~a few Lyapunov times)
adjudication sample A: 50,000 samples at interval 50 steps
```

`E_max` is computed from `C` only; labelling, kNN convergence, and
twin-state run on `A` only. `A` is held at 50,000 to match v5's
adjudication N for apples-to-apples comparison. Total post-burn-in
integration roughly doubles vs v5 (~40 min/run expected; see §7 cost).
Rationale for held-out rather than in-sample: although `E_max` is a
single global scalar (so in-sample calibration would not obviously leak
*fiber* structure into the local kNN test), held-out removes all doubt
at low cost and matches the agreed design.

*Cost-reduced alternative (not pinned; a sign-off option):* `C = A =
30,000` with the same gap, ~1.2× v5 cost. Decide at sign-off.

## 5. Harness objective-mode

A new objective mode, selected by a flag, leaves the v0–v6 fixed-
percentile path untouched:

```text
--objective {overshoot-burnin (default), portable-quantile}
```

`portable-quantile` config additions (pinned in the preset, pre-
registered): `objective_quantile q = 0.70`, `calibration_sample_count =
50000`, `calibration_gap_steps = 5000`. New presets `lock_v7_g200`
(`k_f=2, G=200, K=3`) and `lock_v7_g300` (`k_f=2, G=300, K=3`), both
with `e_max_burnin_fraction` irrelevant under this objective (E_max no
longer from burn-in) and `objective = portable-quantile`.

Implemented by extending `run_cell` to integrate the extra calibration
block + gap before the adjudication block; compute `M(u)` for
calibration samples; set `E_max = quantile(M_C, q)`; then
label/adjudicate the adjudication block exactly as today. All
adjudicators (`knn`, `knn-sweep`, `twin-state`) consume the adjudication
block unchanged. Smoke parity and overshoot-burnin regression were
checked before the lock runs.

## 6. Portability gate (new, pre-registered)

**Before** interpreting any control-sufficiency verdict, confirm the
objective is actually portable:

```text
damp_fraction(A) in [0.20, 0.40]   at BOTH G=200 and G=300.
```

- If both pass → the objective is portable; proceed to read the kNN
  verdicts as a genuine regime comparison.
- If either fails → **`PDE-C1-RG-PORTABILITY-FAIL`**: even the
  held-out-quantile objective does not transfer (e.g. severe
  non-stationarity between `C` and `A`). This is itself a finding;
  do **not** retune `q` or the split to rescue it (that would be
  `PDE-C1-RG-NEG-B`). A new construction (burst-onset, §9) would be a
  separate v2 pre-registration.

`[0.20, 0.40]` is `0.30 ± 0.10`; by construction `A` should sit very
near `0.30`, so this gate mainly catches stationarity/leakage failures.

## 7. Program and run order (pinned)

```text
1. lock_v7_g200 --adjudicator knn-sweep   (positive control + de-confound)
2. lock_v7_g300 --adjudicator knn-sweep   (the generality test)
   [portability gate checked on both before interpreting either]
3. IF both return STRICTNESS_WITNESS_POSITIVE:
      lock_v7_g300 --adjudicator twin-state   (support companion)
      (lock_v7_g200 twin-state optional, for a complete re-witness)
```

The **G=200 re-run is mandatory and runs first**: the portable objective
is a *new* objective, so the v5 witness must be re-established under it
before G=300 generality means anything. If G=200 flips (does **not**
return POSITIVE under the portable objective), that is a critical
finding — it would mean the v5 POSITIVE was partly an artifact of the
old fixed-percentile threshold — and the generality program pauses to
reconcile before G=300 is interpreted.

Expected wall-clock ~40 min per kNN-sweep run (≈2× v5 from the doubled
post-burn-in integration), ~35 min per twin-state run. Up to ~4 runs.
None inline under the ~10-minute rule.

## 8. Branches and interpretation

Per-regime kNN verdict feeds the existing
[`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md) §6
branch family, gated by §6 portability here:

| G=200 kNN | G=300 kNN | program outcome |
|---|---|---|
| POSITIVE | POSITIVE | `PDE-C1-RG-POS` (control-suff replicates across G under a portable objective); proceed to twin-state |
| POSITIVE | NEG-A | `PDE-C1-RG-NEG-A` (regime-2 is a localized window on the Grashof axis — the informative failure) |
| POSITIVE | deferral | `PDE-C1-RG-INCONCLUSIVE_CONTROL` |
| not POSITIVE | — | **program pauses**: portable objective overturns the v5 witness; reconcile before any generality claim |

`PDE-C1-RG-POS` here means: the v5 Reading-2 regime-2 witness has a
higher-Grashof replication **under an objective that is portable by
construction** — strictly stronger than a same-objective coincidence.
It still does **not** mean generality across all `G`, across `k_f`, an
infinite-dimensional NSE theorem, `J`-optimality, or promotion without
external review.

## 9. Pre-registration discipline

- All values in §3–§6 are fixed before any verdict-bearing run.
- Post-hoc change to `q`, the split, the gap, `epsilon_K`, the kNN/
  twin-state thresholds, or the regime after reading a receipt →
  `PDE-C1-RG-NEG-B`.
- The portability gate is a **precondition**, not a tunable: a
  `PORTABILITY-FAIL` is filed, not rescued.
- Burst-onset (a `dE_K/dt` or relative-jump onset criterion) is the
  **documented v2 alternative**, not part of this artifact; it carries
  more free parameters and would need its own pre-registration.

## 10. Build Decisions Closed

The sign-off choices landed as:

1. **Split sizes:** 50k calibration / 50k adjudication, preserving
   apples-to-apples comparison with v5.
2. **Quantile:** `q = 0.70`, targeting `damp_fraction ≈ 0.30` to match
   the v5 action scale.
3. **Build:** `--objective portable-quantile` plus `lock_v7_g200` /
   `lock_v7_g300` presets, with regression and smoke checks before the
   verdict-bearing runs.

## 12. Result (2026-05-29) — PDE-C1-RG-POS

The full program executed; all three verdict-bearing runs landed. The
portable objective fixed the v6 vacuity and the complete regime-2
witness **replicates at G=300**.

| run | portability gate (adj damp ∈ [0.20,0.40]) | verdict |
|---|---|---|
| `lock_v7_g200` kNN (positive control) | 0.3003 ✓ (calib 0.300) | `STRICTNESS_WITNESS_POSITIVE`, `a_mm = −0.00079`, slope 0.736 |
| `lock_v7_g300` kNN (generality test) | 0.2688 ✓ (calib 0.300) | `STRICTNESS_WITNESS_POSITIVE`, `a_mm = +0.00058`, slope 0.564 |
| `lock_v7_g300` twin-state | — | `TWIN_STATE_CERTIFIED` (100% witness coverage, 942,834 unique pairs, `δ_H = 0.0111` from real median ‖Q_K‖ = 0.222) |

Receipts: `results/proof/c1-rg-v1-g200-knn-sweep/`,
`results/proof/c1-rg-v1-g300-knn-sweep/`,
`results/proof/c1-rg-v1-g300-twin-state/`.

### What it establishes

- **Portability gate passed at both regimes.** Held-out
  `damp_fraction` = 0.300 (G=200) and 0.269 (G=300), both in
  [0.20,0.40]. The portable objective is genuinely regime-portable:
  v6's `damp_fraction = 0.004` vacuity at G=300 is gone (0.269 now). The
  small G=300 calib-vs-adj gap (0.300 → 0.269) is a mild, honest
  signature of intermittency non-stationarity, well inside the gate.
- **Positive control (G=200) re-establishes the v5 witness under a
  different, sounder objective.** `a_mm = −0.00079`, slope 0.736 vs v5's
  −0.00078 / 0.737 — near-identical despite a different `E_max`
  construction (held-out look-ahead-max quantile 0.7344 vs burn-in
  95th-percentile). The v5 control-sufficiency result is **not** an
  artifact of the old threshold rule.
- **Generality test (G=300) POSITIVE.** `mean_minority` extrapolates to
  ~zero (`a_mm = +0.00058 ≤ 0.005`), through-origin, 10–22× below the
  random-label floor → clean decision surface, control-sufficient.
- **Twin-state CERTIFIED at G=300**, non-degenerately, at the same
  `ε_K = 0.0664` as the control read, so the two halves **compose** into
  a complete regime-2 witness at G=300.

**Net: regime-2 (state-insufficient AND control-sufficient) is now a
two-regime result on the Grashof axis — (k_f=2, G=200) and
(k_f=2, G=300) — under a portable objective, no longer cell-local.**
The control half is also dimension-robust (v4/v5 across d=32/d=18) and
now objective-robust (fixed-percentile and portable-quantile both
POSITIVE at G=200).

### What it does NOT establish (scope held)

- **Two points on ONE axis.** Grashof only; `k_f = 2` fixed throughout.
  The objective family now has two members but both are
  energy-overshoot-type. This is not full substrate-generality.
- **Finite-Galerkin, sampled-support, numerical** — not a theorem about
  the infinite-dimensional NSE attractor.
- **Proxy faithfulness strengthened, not closed.** Two independent
  objective constructions agreeing helps, but `π̂` is still a proxy, not
  a derived `J`-optimal selector with an explicit action cost.
- **External PDE review (criterion c) open.** **C1 remains UNPROMOTED** —
  but this is the strongest the candidate has been: a two-regime,
  dimension-robust, objective-robust, end-to-end regime-2 witness.

### Next firm-up axes (none required, none urgent)

`k_f`-axis generality (vary forcing geometry at fixed G); a genuinely
different objective family (enstrophy / burst-onset, the documented v2
alternative); proxy-faithfulness via a derived `J`-optimal selector;
external review. Any of these would further harden C1; none is needed
to bank the present two-regime result.

## 11. Cross-References

- [`PDE_C1_REGIME_GENERALITY_v0.md`](PDE_C1_REGIME_GENERALITY_v0.md) —
  the G=300 fixed-percentile probe that deferred for objective vacuity;
  this artifact is its portable-objective successor.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) /
  [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md)
  — the adjudicators reused unchanged on the adjudication block.
- [`PDE_C1_CELLSET_KOLMOGOROV_v5.md`](PDE_C1_CELLSET_KOLMOGOROV_v5.md) —
  the completed `G=200, K=3` witness the G=200 re-run must re-establish
  under the portable objective.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — the
  ledger; §12's `PDE-C1-RG-POS` is reflected there as a two-regime,
  still-unpromoted C1 witness.


==============================================================================
## [ ADJUDICATOR · control-sufficiency (kNN convergence) ]
*source: `docs/proof/PDE_C1_KNN_CONVERGENCE_CHECK.md` · sha256:7429945451a4*
==============================================================================

# PDE C1 — kNN Convergence Check (Pre-Registration)

> Pre-registration of the scale-dependence test that adjudicates the
> provisional `PDE-C1-NEG-A` from the v4-regime kNN run
> (`results/proof/c1-kolmogorov-v4-knn/`, `incompat_fraction = 0.0716`).
> Filed 2026-05-28, **before** the convergence run is read. Purpose:
> distinguish a *genuine* fiber-incompatibility from a *finite-radius
> boundary-straddling artifact*. Classification thresholds below are
> fixed here and not tuned post-hoc (post-hoc change → `PDE-C1-NEG-B`).

## 1. The question

The v4 kNN run solved coverage (`fidelity_coverage = 1.0`) and fired a
mechanical `PDE-C1-NEG-A`: 7.16% of fidelity-passing neighbourhoods
have local minority fraction above `delta_action = 0.10`. Two competing
explanations:

- **Genuine fiber-incompatibility (true NEG-A).** Distinct microstates
  with the same `Phi_K` need different proxy actions — expected here,
  since the proxy label depends on the full-state `E_K` future while
  `Phi_K` sees only the low modes, so high modes vary freely within a
  fiber and drive different labels. Predicts `incompat_fraction` is
  **constant as the neighbourhood radius `r_k` shrinks** (shrinking the
  signature-ball never constrains the high modes).
- **Finite-radius boundary-straddling (true POSITIVE).** The proxy is
  locally a function of `Phi_K` (control-sufficient), with a clean
  decision surface; the 7.16% is just the attractor measure within
  `r_k` of that surface. Predicts `incompat_fraction → 0` as `r_k → 0`,
  scaling ~linearly with `r_k` (shell of thickness `r_k` around a
  codimension-1 surface).

This is exactly Reading 2's regime-2 (control-sufficient) vs. regime-3
(control-insufficient) distinction, made empirical.

## 2. The test

Re-run the v4 regime (`--preset lock_v4 --adjudicator knn-sweep`).
Query the `BallTree` once at `k = 100`; for each
`k ∈ {10, 20, 30, 50, 100}` sub-slice that query and compute:

- `r_k_median` (over all samples) — the neighbourhood scale;
- `fidelity_coverage` = fraction with `r_k ≤ epsilon_K`;
- `incompat_fraction` = fraction of fidelity-passing samples with local
  minority fraction `> delta_action`.

Smaller `k` → smaller `r_k`. The shape of `incompat_fraction(r_k)` is
the discriminator. Cost: one ~26-min integration; the sweep itself is a
single `k=100` query plus sub-slicing (negligible).

## 3. Pre-registered classification

Ordinary-least-squares fit of `incompat_fraction` on `r_k_median`
across the five swept `k`; let `a` be the intercept (extrapolation to
`r_k → 0`) and `min_incompat` the smallest `incompat_fraction` over the
sweep:

- **`PDE-C1-NEG-A` confirmed (PLATEAU_NONZERO)** iff `a > 0.02` **and**
  `min_incompat > delta_incompat` (= 0.01). The incompatibility
  survives extrapolation to zero radius → genuine.
- **`STRICTNESS_WITNESS_POSITIVE` (DECAYS_TO_ZERO)** iff `a < 0.01`.
  The incompatibility extrapolates to zero → finite-radius boundary
  artifact; the provisional NEG-A is overturned and the proxy is
  control-sufficient on fibers at this cell.
- **`INCONCLUSIVE_CONVERGENCE`** otherwise (`0.01 ≤ a ≤ 0.02`). Neither
  decisively; a larger `N` or a wider `k` range is needed. Non-verdict.

The vacuity gate (global `damp_fraction ∈ [delta_proxy_min,
1 - delta_proxy_min]`) still applies and takes precedence; v4's
`damp_fraction = 0.30` passes it.

## 4. Caveats recorded before the read

- **Small-`k` grain.** At `k = 10`, `minority > 0.10` means `≥ 2/10`
  disagree (effective threshold 0.20); at `k = 100`, `≥ 11/100` (0.11).
  The threshold grain differs across `k` — a known confound. The raw
  `incompat_fraction(k)` table is reported alongside the mechanical
  classification so the trend is visible irrespective of the OLS rule.
- **Linear extrapolation is first-order.** A smooth codim-1 boundary
  gives `incompat ∝ r_k` (linear through the origin); the OLS intercept
  is a first-order discriminator, not a proof. A clearly-positive or
  clearly-zero intercept is trustworthy; a marginal one files
  `INCONCLUSIVE_CONVERGENCE`, not a forced call.
- **High-`k` fidelity.** At `k = 100`, `r_k` grows and
  `fidelity_coverage` may fall below 1; `incompat_fraction` is always
  computed over the fidelity-passing set at that `k`.

## 6. First-run disposition and amended pre-registration (2026-05-28)

The first convergence run (`results/proof/c1-kolmogorov-v4-knn-sweep/`,
sweep `k ∈ {10,20,30,50,100}`) returned a **mechanical `PDE-C1-NEG-A`
that does not survive scrutiny**. The sweep:

| k | r_k median | fidelity coverage | incompat fraction |
|---:|---:|---:|---:|
| 10 | 0.0196 | 1.00 | 0.0349 |
| 20 | 0.0288 | 1.00 | 0.0645 |
| 30 | 0.0346 | 1.00 | 0.0716 |
| 50 | 0.0448 | 1.00 | 0.0974 |
| 100 | 0.0638 | **0.447** | 0.0583 |

The OLS intercept came out `+0.046` (→ NEG-A) **only because the k=100
point is included** — and that point fails its own fidelity-coverage
gate (`0.447 < S_pos = 0.50`). It has high `r_k` but lower
`incompat_fraction` (its population is half-excluded), which levers the
intercept up. Refit on the four full-coverage points (k≤50):
`incompat ≈ 2.4·r_k`, intercept `≈ −0.010` → DECAYS_TO_ZERO →
**boundary artifact, POSITIVE**. The verdict flips on one
coverage-failing point. Two pre-registration gaps caused this:

1. The OLS was not restricted to coverage-passing sweep points.
2. The thresholded `incompat_fraction` has a **grain confound**: the
   effective minority threshold is `0.20` at k=10 vs `0.125` at k=40
   (since `minority > 0.10` rounds to a different neighbour count at
   each k), biasing the thresholded statistic *toward* POSITIVE at
   small k — so it cannot be the trusted primary statistic.

**Amended pre-registration for the re-run** (fixed before re-reading):

- **Sweep** `k ∈ {10,15,20,25,30,40,50}` — all expected full-coverage
  at this regime (`r_k(k=50) = 0.045 < epsilon_K = 0.063`); a denser
  low-`k` curve, no coverage-failing point.
- **Exclusion.** Any sweep point with `fidelity_coverage < S_pos` is
  dropped from both fits.
- **Primary statistic: `mean_minority`** — the mean local minority
  fraction over fidelity-passing samples (threshold-free; the canonical
  nonparametric estimator of the conditional non-constancy
  `E[1 - max_a mu_sigma(a)]`, whose `r_k → 0` limit is the
  Blackwell-sufficiency-failure measure). Fit `mean_minority` vs
  `r_k_median` over coverage-passing points; intercept `a_mm`.
  - `a_mm ≤ 0.005` (consistent with zero) → **POSITIVE** (boundary
    artifact; proxy control-sufficient on fibers — Reading-2 regime 2).
  - `a_mm ≥ 0.015` → **PDE-C1-NEG-A** (genuine fiber-incompatibility).
  - else → **INCONCLUSIVE_CONVERGENCE**.
- **Secondary (diagnostic, not gated):** thresholded
  `incompat_fraction(k)` with the grain caveat, reported for
  continuity.
- The raw `mean_minority(r_k)` curve is reported so the trend is
  visible regardless of the threshold call.

Neither the contaminated NEG-A nor the clean-points POSITIVE recompute
is filed; the amended re-run adjudicates.

## 7. Amended-run result (2026-05-28) — provisional POSITIVE

The amended convergence check
(`results/proof/c1-kolmogorov-v4-knn-sweep2/`, sweep
`k ∈ {10,15,20,25,30,40,50}`, all full coverage) returned
**`STRICTNESS_WITNESS_POSITIVE`**. The provisional v4 `PDE-C1-NEG-A`
is **overturned**.

| k | r_k median | mean_minority | incompat fraction |
|---:|---:|---:|---:|
| 10 | 0.0196 | 0.01228 | 0.035 |
| 15 | 0.0237 | 0.01658 | 0.058 |
| 20 | 0.0288 | 0.02058 | 0.065 |
| 25 | 0.0322 | 0.02153 | 0.069 |
| 30 | 0.0346 | 0.02408 | 0.072 |
| 40 | 0.0393 | 0.02763 | 0.087 |
| 50 | 0.0448 | 0.03112 | 0.097 |

Primary fit `mean_minority = a_mm + b·r_k`: `a_mm = −0.00125`
(≤ 0.005 → POSITIVE), slope `0.729`. Diagnostic `incompat_fraction`
intercept `−0.0031` (agrees).

**Three robustness checks (all pass):**

1. **Through-origin, no plateau.** `mean_minority / r_k ≈ 0.70`
   constant across the sweep — a clean line through the origin, no
   flattening at small `r_k`. A genuine plateau would level to a
   positive constant; it does not. Decay-to-zero is robust.
2. **Not a random-label artifact.** Unstructured iid-30%-damp labels
   would give `mean_minority ≈ 0.30` flat in `r_k`. Observed 0.012–0.031
   is 10–25× smaller and scales with radius → labels are spatially
   organized into action-pure regions with mixing only in a thin
   boundary shell (a clean decision surface). This decisively rules out
   distance-concentration-randomness.
3. **Grain confound neutralized.** Threshold-free `mean_minority` and
   grain-prone `incompat_fraction` both extrapolate to ~0; they agree,
   so the verdict is not an artifact of either statistic.

**Interpretation (scoped).** On the v4 cell (`k_f=2, G=200, K=4`), the
low-band-energy safety proxy `\hat{pi}` is **control-sufficient on
`Phi_K`-fibers** up to a measure-zero decision surface, even though
`Phi_K` is provably non-injective (cell-set §4.1). This is the C1
sidecar's Reading-2 **regime 2** (state-insufficient, control-
sufficient) — the non-vacuous Sundog target. `damp_fraction = 0.30` so
the control-sufficiency is non-trivial (a real 30/70 action split
determined by `Phi_K`), not a degenerate all-`no_op`.

**What this does NOT establish (held against the result):**

- **One cell.** v4 regime / K=4 / this objective only. No claim of
  generality across regimes, signatures, or objectives.
- **State-insufficiency on the attractor is not yet airtight.** §4.1
  certifies `Phi_K` non-injective on `B_abs`; the full regime-2 claim
  needs the deferred attractor-support twin-state certificate on
  `supp(mu_SRB)`. The POSITIVE establishes control-sufficiency on the
  attractor; the state-insufficiency half is `B_abs`-level pending that
  certificate.
- **Proxy faithfulness.** `\hat{pi}` is a proxy; a reviewer may require
  a derived `J`-optimal selector (design §3 substitution path).
- **Resolution floor.** The test resolves to `r_k ≈ 0.02`; genuine
  incompatibility below that scale is unprobed (a larger `N` would
  push lower).
- **External review (criterion c) open.**

**Verdict status: provisional POSITIVE — strongly advances C1, does
not promote it out of the ledger.**

## 8. Replication at v5 / d=18 (2026-05-28)

To test whether the POSITIVE is a `d=32` distance-concentration
artifact, the convergence check was re-run at the v5 regime (K=3,
signature dim 18; `results/proof/c1-kolmogorov-v5-knn-sweep/`). Result:
**`STRICTNESS_WITNESS_POSITIVE`**, near-identical to v4:

| quantity | v4 (d=32) | v5 (d=18) |
|---|---:|---:|
| `a_mm` (intercept) | −0.00125 | −0.00078 |
| slope `b` | 0.729 | 0.737 |
| `damp_fraction` | 0.30014 | 0.2977 |
| `mean_minority` @ k=30 | 0.02408 | 0.02323 |

The control-sufficiency verdict, the boundary-shell slope (~0.73), and
the proxy split (~0.30) are invariant to halving the signature
dimension — the same dimension-robustness `damp_fraction` showed across
v4/v5 binning runs. This **rules out the d=32-specific
distance-concentration explanation** for the clean-boundary reading.
Together with the through-origin scaling and the random-label control,
the boundary-artifact → control-sufficiency reading is well-supported
at this regime.

**Still provisional / still scoped:** both v4 and v5 are the *same*
regime (`k_f=2, G=200`); dimension-robustness is not regime-generality.
The twin-state certificate, proxy faithfulness, resolution floor, and
external review (criterion c) remain open exactly as in §7.

## 5. Cross-references

- [`PDE_C1_KNN_ADJUDICATION_DESIGN.md`](PDE_C1_KNN_ADJUDICATION_DESIGN.md)
  — the adjudicator this check stress-tests; §8 flagged the
  boundary-straddling risk this resolves.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) §5b — the kNN
  adjudication method.
- [`PDE_C1_LOCK_EXECUTION_SYNTHESIS.md`](PDE_C1_LOCK_EXECUTION_SYNTHESIS.md)
  — the v0–v5 obstruction this campaign is working through.
- `results/proof/c1-kolmogorov-v4-knn/` — the provisional NEG-A this
  check adjudicates.


==============================================================================
## [ ADJUDICATOR · state-insufficiency (twin-state) ]
*source: `docs/proof/PDE_C1_TWIN_STATE_CERTIFICATE.md` · sha256:4254a81f610f*
==============================================================================

# PDE C1 Twin-State Certificate

> Pre-registration and execution harness for the C1 support-level
> state-insufficiency bridge. Filed 2026-05-28 after the v4/v5 kNN
> convergence checks delivered a provisional, dimension-robust
> control-sufficiency positive at `(k_f = 2, G = 200)`. This artifact
> tests the remaining state-insufficiency-on-attractor half: whether
> `Phi_K` is non-injective on the sampled SRB-like support, not merely
> on the absorbing ball `B_abs`.

## 1. Claim Boundary

This certificate does **not** adjudicate proxy-control sufficiency. That
was the role of the kNN convergence check
([`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md)).
It also does not prove a theorem about the exact infinite-dimensional
Navier-Stokes attractor.

It claims only this, if the positive branch fires:

```text
On the finite-Galerkin sampled support for the pinned Kolmogorov cell,
there is positive-mass numerical evidence of distinct states with the
same low signature Phi_K and separated complementary high-mode state Q_K.
```

That closes the C1 regime-2 state-insufficiency bridge at the sampled
support level for the tested cell. C1 still remains unpromoted until
proxy faithfulness and external PDE review are handled.

## 2. Target Cell

Primary target:

```text
--preset lock_v5
k_f = 2
G = 200
K = 3
d_K = 18
N_twin = 50,000
```

Why v5 first: v5 is the lower-dimensional replication of the v4
positive, with near-identical boundary-shell slope and intercept. A
support certificate at v5 is therefore the tightest direct companion to
the dimension-robust control-sufficiency read.

Optional cross-check:

```text
--preset lock_v4
k_f = 2
G = 200
K = 4
d_K = 32
N_twin = 50,000
```

The v4 run is a replication check, not a precondition for the v5
certificate.

## 3. State Coordinates

The low signature `Phi_K` is exactly the existing harness signature:
half-plane Fourier representatives, normalized as

```text
omega_hat(k) / (M^2 |k|)
```

with real and imaginary parts split into Euclidean coordinates.

The complementary high-mode projection `Q_K` is implemented by the same
normalization over all active de-aliased half-plane Galerkin modes not
included in `Phi_K`. This keeps the low-signature and high-mode
separation norms in the same coordinate convention.

For v5, the harness reports:

```text
high_mode_count = 211
high_mode_dimension = 422
```

for the active de-aliased complement. For v4, the complement is smaller
because `Phi_K` includes more low modes.

## 4. Pair Search

For each sampled state `u_i`, query a `BallTree` in signature space for
the `k_twin = 50` nearest neighbours, including self. Discard self. A
directed candidate edge `(i, j)` is signature-near iff:

```text
||Phi_K(u_i) - Phi_K(u_j)|| <= epsilon_K
```

where `epsilon_K` is inherited from the C1 cell:

```text
epsilon_K = 0.05 * sqrt(2 E_max)
```

and `E_max` is computed exactly as in the selected preset. Coverage is
measured at the sample level:

```text
candidate_sample_fraction =
  |{ i : i has at least one non-self signature-near neighbour }| / N_twin
```

The coverage gate is inherited:

```text
candidate_sample_fraction >= S_pos = 0.50
```

## 5. High-Mode Separation Threshold

Before reading the full lock result, pin:

```text
delta_H = max(1e-6, 0.05 * median_i ||Q_K(u_i)||)
```

Rationale: the certificate should not count machine-noise high-mode
differences as state separation, but the threshold should scale with the
actual high-mode amplitude of the sampled attractor. The formula uses
only the sample's marginal high-mode norm distribution, not the
candidate-pair separations.

A witness edge is a signature-near candidate edge satisfying:

```text
||Q_K(u_i) - Q_K(u_j)|| >= delta_H
```

## 6. Branches

Pre-registered branches, in order:

1. **`SMOKE_ONLY`** if the preset is not verdict-bearing or manual
   overrides are used.
2. **`TWIN_STATE_DEFERRED_HIGH_MODE_FLOOR`** if
   `median_i ||Q_K(u_i)|| <= 1e-6`. The sampled support is numerically
   flat in complementary modes at this scale.
3. **`TWIN_STATE_DEFERRED_COVERAGE`** if
   `candidate_sample_fraction < 0.50`. The sample does not provide enough
   signature-near candidate pairs.
4. **`TWIN_STATE_CERTIFIED`** iff both hold:

```text
witness_sample_fraction >= 0.01
unique_witness_pairs >= 100
```

where `witness_sample_fraction` is the fraction of all samples with at
least one witness edge, and `unique_witness_pairs` canonicalizes
directed edges to unordered pairs.

5. **`TWIN_STATE_NO_CERTIFICATE`** otherwise. This is explicitly a
   no-certificate receipt, not a proof of injectivity.

## 7. Harness

Implemented in
[`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py)
as:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v5 --adjudicator twin-state --out results\proof\c1-kolmogorov-v5-twin-state
```

Receipt files:

```text
manifest.json
PDE_C1_KOLMOGOROV_RESULTS.md
twin-state-witnesses.csv
```

The witness CSV records a capped set of witness examples for audit:
sample ids, signature distance, high-mode distance, high-distance over
`delta_H`, and the two high-mode norms.

## 8. Smoke Receipt

Smoke command, run 2026-05-28:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset smoke --adjudicator twin-state --allow-unregistered-overrides --sample-count 80 --burnin-steps 200 --sample-interval-steps 3 --lookahead-steps 10 --out results\proof\c1-twin-state-smoke
```

Receipt: `results/proof/c1-twin-state-smoke/`.

Readout:

```text
status = SMOKE_ONLY
elapsed_seconds = 2.030
candidate_sample_fraction = 1.0
witness_sample_fraction = 1.0
```

The smoke validates plumbing only: high-mode capture, BallTree query,
candidate/witness aggregation, manifest writing, receipt writing, and
witness CSV output. It files no support-level certificate.

## 9. Full-Run Commands

Do not run these inline under the repo's ~10-minute rule. Expected
wall-clock is approximately 20-30 minutes for v5, based on the v5 lock
and v5 kNN-sweep receipts.

Primary:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v5 --adjudicator twin-state --out results\proof\c1-kolmogorov-v5-twin-state
```

Optional replication:

```powershell
python scripts\pde_c1_kolmogorov_cell.py --preset lock_v4 --adjudicator twin-state --out results\proof\c1-kolmogorov-v4-twin-state
```

If v5 returns `TWIN_STATE_CERTIFIED`, the C1 ledger can mark the
state-insufficiency-on-attractor bridge closed for the tested regime.
If it returns any deferral or no-certificate receipt, C1 remains at the
current honest boundary: non-injectivity certified on `B_abs`, not yet
on `supp(mu_SRB)`.

## 11. Result (2026-05-28) — TWIN_STATE_CERTIFIED at v5

Primary v5 run (`results/proof/c1-kolmogorov-v5-twin-state/`, ~21 min):
**`TWIN_STATE_CERTIFIED`**.

| quantity | value | gate |
|---|---:|---|
| `delta_H` | 0.01167 | `0.05 × median‖Q_K‖`, floor `1e-6` |
| median ‖Q_K‖ / min / max | 0.2333 / 0.2202 / 0.2442 | — |
| signature-near coverage | 1.00 | ≥ `S_pos = 0.50` |
| candidate pairs (unique) | 1,263,121 | — |
| witness sample fraction | **1.00** | ≥ 0.01 |
| witness pairs (unique) | **693,795** | ≥ 100 |
| witness high-dist p50 / p95 | 0.0154 / 0.0210 | ≥ `delta_H` |

**Robustness / non-degeneracy checks:**

- **Not machine noise.** `delta_H` is set by the real high-mode
  amplitude (median ‖Q_K‖ = 0.23 ≈ 27% of the ‖Φ_K‖ scale ~0.85), far
  above the `1e-6` floor.
- **Not "any two points."** Candidates are within `ε_K = 0.0606`; the
  50-NN radius at this regime is ~0.045 (~5% of the signature norm), so
  signature-near is genuinely signature-local.
- **Overwhelmingly above the gates** (100% witness coverage; 694k
  unique witness pairs vs the 100 threshold), not marginal.

**What this closes.** `Phi_K` is non-injective on the sampled SRB-like
support for the v5 cell — the state-insufficiency-on-attractor half
that §4.1 of the cell-set had only established algebraically on
`B_abs`. Tested at the **same `ε_K`** as the kNN control-sufficiency
read, so the two halves compose into a complete Reading-2 **regime 2**
witness at this cell: within `ε_K` signature-balls, Q_K varies
(certified here) while the proxy action stays constant up to
measure-zero (the kNN POSITIVE). The signature collapses distinct full
states yet suffices for the control objective.

**Sharpened 2026-05-29 (paired fiber-constancy).** The composition above
is at a *matched radius* — two population statistics over the same
`ε_K`-balls. [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md)
tightens it to a **paired** test on the *same* witness pairs: among the
certified `Q_K`-separated pairs, the proxy-action disagreement is
`D_witness = 0.0367` (G=200) / `0.0382` (G=300), both well under the
`delta_action = 0.10` line and within ~1 point of the candidate-pair rate
(`0.0319` / `0.0290`). High-mode separation adds almost nothing to action
disagreement — the residual is a signature-space boundary layer, not
`Q_K`-driven. Both runs reproduced this certificate bit-for-bit
(`PAIRED_FIBER_CONSTANCY_POSITIVE`), composing state-insufficiency and
control-sufficiency on the same pairs in both regimes.

**Honest calibration.** This was the *expected-easy* half — on a
chaotic attractor with a 422-dim high-mode complement at ~0.23
amplitude, signature-local states with separated high modes are nearly
automatic. The certificate's contribution is that it is now *measured*
on the actual support with pre-registered thresholds, not argued.

**What it does NOT do.** Finite-Galerkin, sampled-support, numerical —
not a theorem about the infinite-dimensional NSE attractor. One cell
(k_f=2, G=200, K=3); not regime-generality. Does not touch proxy
faithfulness or external review. **C1 stays unpromoted.**

## 10. Cross-References

- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) section
  4.1 - the original absorbing-ball non-injectivity witness and
  attractor-support caveat.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) section 6 - the
  one-line deferred support-level certificate spec.
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) -
  the control-sufficiency result this state-insufficiency bridge
  complements.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) - the
  staging ledger that keeps this certificate as an open firm-up item
  until the full run returns.


==============================================================================
## [ RUN RECEIPT · G=200 kNN sweep (positive control) ]
*source: `results/proof/c1-rg-v1-g200-knn-sweep/PDE_C1_KOLMOGOROV_RESULTS.md` · sha256:7e7771baf5b0*
==============================================================================

# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v7_g200`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.018229 | 1 | 0.012264 | 0.03486 |
| 15 | 0.0226535 | 1 | 0.0163893 | 0.05664 |
| 20 | 0.0271156 | 1 | 0.019529 | 0.0622 |
| 25 | 0.0306346 | 1 | 0.0212952 | 0.0684 |
| 30 | 0.0325761 | 1 | 0.0232107 | 0.06868 |
| 40 | 0.0372058 | 1 | 0.0262915 | 0.0834 |
| 50 | 0.0423703 | 1 | 0.0306324 | 0.09616 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = -0.000792245`, slope `0.736096`
- secondary (diagnostic) `incompat_fraction` fit intercept: `-0.00227918` (grain-confounded; not gated)
- damp fraction (global): `0.3003`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `2418.494`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`


==============================================================================
## [ RUN RECEIPT · G=300 kNN sweep (generality test) ]
*source: `results/proof/c1-rg-v1-g300-knn-sweep/PDE_C1_KOLMOGOROV_RESULTS.md` · sha256:596a0f7de103*
==============================================================================

# PDE C1 Kolmogorov kNN Convergence-Check Receipt

**Status:** STRICTNESS_WITNESS_POSITIVE
**Preset:** `lock_v7_g300`
**Adjudicator:** `knn-sweep`
**Interpretable verdict:** `True`

## Sweep (vs neighbourhood radius; primary statistic = mean_minority)

| k | r_k median | fidelity coverage | mean_minority | incompat fraction |
|---:|---:|---:|---:|---:|
| 10 | 0.019769 | 1 | 0.011988 | 0.03652 |
| 15 | 0.0253967 | 1 | 0.014876 | 0.04816 |
| 20 | 0.0306851 | 1 | 0.016937 | 0.04936 |
| 25 | 0.0326464 | 1 | 0.0190904 | 0.06076 |
| 30 | 0.0355433 | 1 | 0.0213187 | 0.0667 |
| 40 | 0.042201 | 1 | 0.0243645 | 0.076 |
| 50 | 0.0469411 | 1 | 0.0270216 | 0.08498 |

## Readout

- PRIMARY fit `mean_minority = a_mm + b * r_k_median` over `7` coverage-passing points: intercept `a_mm = 0.000577142`, slope `0.564177`
- secondary (diagnostic) `incompat_fraction` fit intercept: `0.000704953` (grain-confounded; not gated)
- damp fraction (global): `0.26878`
- classification (pre-registered §6): `a_mm <= 0.005` => POSITIVE; `a_mm >= 0.015` => NEG-A; else INCONCLUSIVE
- elapsed seconds: `2610.636`

## Branch

`mean_minority` extrapolates to ~zero as `r_k -> 0` (`a_mm <= 0.005`): the observed mixing is a finite-radius boundary-straddling artifact around a clean decision surface. The proxy is control-sufficient on fibers at this cell (Reading-2 regime 2); the provisional v4 `PDE-C1-NEG-A` is **overturned**.

## Files

- `manifest.json`
- `knn-sweep.csv`


==============================================================================
## [ RUN RECEIPT · G=300 twin-state (support companion) ]
*source: `results/proof/c1-rg-v1-g300-twin-state/PDE_C1_KOLMOGOROV_RESULTS.md` · sha256:ad06ebf514a1*
==============================================================================

# PDE C1 Twin-State Certificate Receipt

**Status:** TWIN_STATE_CERTIFIED
**Preset:** `lock_v7_g300`
**Adjudicator:** `twin-state`
**Interpretable certificate:** `True`

## Readout

- samples: `50000`, k (effective): `50`
- `epsilon_K` (signature radius): `0.0664219`
- `delta_H`: `0.0111032` (`0.05` x median high-mode norm, floor `1e-06`)
- high-mode norm median / min / max: `0.222065` / `0.214416` / `0.233035`
- signature-near sample coverage: `1` vs `S_pos = 0.5` (`50000` of `50000`)
- candidate pairs unique / directed: `1318748` / `2450000`
- witness sample fraction: `1` vs `0.01` (`50000` samples)
- witness pairs unique / directed: `942834` / `1699022` vs min unique `100`
- witness high-distance p50 / p95: `0.0172896` / `0.0286686`
- elapsed seconds: `2491.092`

## Branch

A positive-mass fraction of sampled states has a signature-near twin with high-mode separation above `delta_H`. This certifies `Phi_K` non-injective on the sampled SRB-like support for this cell.

## Files

- `manifest.json`
- `twin-state-witnesses.csv`


==============================================================================
## [ BUNDLE PROVENANCE ]
==============================================================================

Each section above is a verbatim copy of an on-disk artifact. Hashes:

| File | Bytes | sha256 (first 12) |
| --- | ---: | --- |
| `docs/proof/PDE_C1_SEPARATION_STATEMENT.md` | 11926 | `98eea9515f52` |
| `docs/proof/PDE_C1_REGIME_GENERALITY_v1.md` | 13876 | `23f17da8b0b6` |
| `docs/proof/PDE_C1_KNN_CONVERGENCE_CHECK.md` | 12093 | `7429945451a4` |
| `docs/proof/PDE_C1_TWIN_STATE_CERTIFICATE.md` | 10479 | `4254a81f610f` |
| `results/proof/c1-rg-v1-g200-knn-sweep/PDE_C1_KOLMOGOROV_RESULTS.md` | 1492 | `7e7771baf5b0` |
| `results/proof/c1-rg-v1-g300-knn-sweep/PDE_C1_KOLMOGOROV_RESULTS.md` | 1490 | `596a0f7de103` |
| `results/proof/c1-rg-v1-g300-twin-state/PDE_C1_KOLMOGOROV_RESULTS.md` | 1070 | `ad06ebf514a1` |

Regenerate with `node scripts/build-c1-review-bundle.mjs`. If a hash here
does not match a fresh read of the file, the bundle is stale — rebuild before
sending.


==============================================================================
## [ CLOSING ]
==============================================================================

<!-- PDF_CLOSING_GRAPHIC -->

### Thank you for reading to the end.

That is the whole packet. The ask was narrow on purpose: not endorsement,
not a Navier–Stokes claim — just a framing check from someone who knows where the honest boundary of this problem sits.
If the language is too strong, the most useful thing you can tell us is
exactly which phrase to weaken and why. A negative answer is the most
valuable outcome we could get.

> **One last note, for a cover-to-cover read.** A sundog — the bright parhelion
> beside the sun that gives this lab its name — is the sky's own `Φ_K`: a
> low-information projection of an enormous hidden state (every ice crystal,
> every ray) that is nonetheless *enough to read the sun's altitude*, yet never
> enough to reconstruct the whole sky. State-insufficient, decision-sufficient.
> That is the separation this packet is asking you to check — and the reason we
> went looking for it in a 2D Navier–Stokes attractor in the first place.

*— Jeffery Hughes Jr., Sundog*
