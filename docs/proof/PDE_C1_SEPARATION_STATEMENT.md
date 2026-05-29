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

## 7. Review path and cross-references

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
