# PDE C1 — Proposition, Definitions, and Claim Ledger

> The crisp, theorem-shaped statement of the C1 result, 2026-05-29. This is
> the top-level reviewer-facing artifact; the detailed positioning,
> anti-vacuity ledger, and observability framing live in
> [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md), and the
> numerical certificates in the per-instance docs (cross-refs in §7).
>
> **Honest framing.** The main clauses below are a **certified-empirical
> proposition** on a finite-Galerkin sampled support — *not* a deductive PDE
> theorem. Two supporting statements are genuinely proved (Lemma, Reading 1);
> the rest are pre-registered numerical certificates; the ∞-dimensional NSE
> analogue is hypothesized. The claim ledger (§6) tags every clause.

## 1. Definitions

| Term | Definition (this cell) |
| --- | --- |
| **Decision domain** `𝒮_N` | the `N=16` Galerkin truncation of 2D Kolmogorov flow (`32×32` grid, pseudo-spectral vorticity), forcing wavenumber `k_f=2`, Grashof `G`. State = retained vorticity Fourier modes (440 real DOF). |
| **Observation map** `Φ_K` | the projection onto the `K=3` low-Fourier band — the 9 lowest half-plane modes, `d=18` real components (the "signature"). `Q_K` = the complementary projection onto the 211 unresolved high modes (422 real DOF). |
| **Sampled support** `supp μ_G` | the empirical (numerical SRB-like) invariant measure from `50000` samples taken every 50 steps after `10⁵` burn-in steps. *Not* the full theoretical attractor; finite, numerical. |
| **Objective** `J_q`, **selector** `π*` | `J_q` = keep the `τ`-step look-ahead of low-band energy `E_low` below `E_max`, with `E_max` the `(1−q)`-quantile of look-ahead-max on a disjoint held-out calibration window (so `damp_fraction = q` is pinned by construction). `π*: 𝒮_N → {no_op, damp}` is the induced optimal selector, `π*(x) = 1[max_{t≤τ} E_low(flow_t x) > E_max]`. |
| **State-insufficient** | `Φ_K` is **not injective** on `supp μ_G`: there is positive `μ_G`-mass of state pairs that `Φ_K` collapses but `Q_K` separates. Equivalently, `Φ_K` does not determine the state. |
| **Control-sufficient** | `π*` is `𝓕_{Φ_K}`-measurable up to `μ_G`-measure `δ`: `π*` factors through `Φ_K` off a set of measure `≤ δ`. Equivalently, `Φ_K` determines the action. |
| **Decision surface** `Γ` | the level set in signature space `Γ = {σ : 𝔼[max_{t≤τ} E_low | Φ_K = σ] = E_max}` separating `no_op` from `damp`. The **boundary set** `B_δ` is its `μ_G`-neighbourhood where the action label is fragile. |

## 2. Proposition (certified-empirical)

> Let `𝒮_N`, `μ_G`, `Φ_K`, `Q_K`, `J_q`, `π*` be as in §1. Then for
> **`G ∈ {200, 300}`**, on `supp μ_G`:
>
> **(i) State-insufficient.** `Φ_K` is non-injective: there is positive
> `μ_G`-mass of pairs `(x,y)` with `‖Φ_K x − Φ_K y‖ ≤ ε_K` and
> `‖Q_K x − Q_K y‖ ≥ δ_H`.
>
> **(ii) Control-sufficient.** `π*` is `Φ_K`-measurable up to a set of
> `μ_G`-measure `≤ δ`, `δ ≈ 0.037`: on the certified non-injective pairs of
> (i), `π*(x) = π*(y)` except on the boundary set `B_δ`.
>
> **(iii) Mechanism.** The low-band energy tendency decomposes as
> `dE_low/dt = g(Φ_K) + R`, where `g = D_low + F_low` (dissipation +
> forcing) is exactly `Φ_K`-measurable and the band-closed transfer is
> identically zero (Lemma, §6); and the unresolved coupling `R = T_low`
> satisfies `R = 𝔼[R|Φ_K] + ξ` with `Var(ξ)/Var(R) ≤ 0.01` — `R` is
> signature-determined off a ~1% residual.

So the smallest control-sufficient observation for `J_q` is strictly coarser
than any state-reconstructive one on this support: `Φ_K` collapses states
that reconstruction separates (i), yet the action (ii) and the
decision-relevant energy coupling (iii) are `Φ_K`-determined.

## 3. What this does / does not claim

**Does claim:** a measured, pre-registered, two-regime separation in a
finite-dimensional dynamical system derived from 2D NSE — `Φ_K` is
state-insufficient yet control-sufficient on the sampled invariant measure,
with the coupling mechanism (iii) explaining *why* — stated in standard
observer-theory terms (§5). **Does not claim:** any Navier–Stokes
existence/smoothness result; a theorem about the infinite-dimensional NSE
attractor (clauses hold on a finite truncation and a sampled measure); a new
determining-modes bound; or generality beyond the tested objective family and
the single forcing geometry `k_f=2`. C1 is **PROVISIONAL, UNPROMOTED**, gated
on external review.

## 4. Two-regime stability (item: portability)

A **certificate**, not a theorem: portability is *demonstrated across two
Grashof regimes* and *argued regime-agnostic by construction* (the objective
is quantile-calibrated, so `damp_fraction` is pinned at every `G`); it is not
proved deductively.

| quantity | G=200 (`lock_v5`) | G=300 (`lock_v7_g300`) | objective-dependent? |
| --- | --- | --- | --- |
| (i) non-injectivity | CERTIFIED (693,795 witness pairs) | CERTIFIED (942,834) | **no** (structural) |
| `ε_K` / `δ_H` | 0.0606 / 0.0117 | 0.0664 / 0.0111 | no |
| (ii) action disagreement `δ ≈ D_witness` | 0.0367 | 0.0382 | **yes** (via `J_q`) |
| `D_witness` vs candidate-pair rate | 0.0367 vs 0.0319 | 0.0382 vs 0.0290 | yes |
| `damp_fraction` | 0.298 | 0.269 | yes (pinned by `q`) |
| (iii) coupling slaving `R²(R\|Φ_K)` | 0.998 | 0.990 | **no** (structural) |
| `R²(g)` ceiling / `R²(perm)` floor | 0.999 / −0.001 | 0.980 / −0.001 | no |

**Objective-dependence split:** clauses **(i)** and **(iii)** are
*objective-free* structural facts about `Φ_K` and the energy budget; clause
**(ii)** is the *objective-dependent* bridge, shown robust across two
objective constructions (fixed-percentile and portable-quantile) and two
signature dimensions (`d=18/32`). The novelty rides on (i)+(iii) being
objective-free and (ii) being robust.

## 5. The decision surface / thin bad set (item: decision surface)

The boundary set `B_δ` is the **thin bad set** where (ii) can fail. Two facts
pin it down:

- **Measure.** `μ_G(B_δ) ≈ D_witness ≈ 0.037` (G=200) / `0.038` (G=300) —
  small and stable across regimes.
- **Disposition — genuine geometry, not artifact.** The paired fiber-
  constancy exhibit shows the disagreement rate on `Q_K`-*separated* witness
  pairs (`0.0367`) is within ~1 point of the rate on *all* signature-near
  pairs (`0.0319`). So `B_δ` is **not** driven by the unresolved high modes
  (not a state-reconstruction artifact) and **not** a model/truncation
  boundary — it is the genuine **signature-space decision surface** `Γ`:
  pairs whose look-ahead sits within the calibration band of `E_max` flip the
  label regardless of `Q_K`. The bad set is exactly the codimension-1 level
  set the controller is *supposed* to be sensitive to, thickened by the
  finite neighbourhood radius `ε_K`. Shrinking `ε_K` (finer sampling) is the
  registered test that `μ_G(B_δ) → 0` as the boundary layer thins.

## 6. Claim ledger (item: empirical vs theoretical)

| Statement | Tag | Basis |
| --- | --- | --- |
| **Lemma:** `T_LLL ≡ 0` (band cannot self-feed energy) | **PROVED** | detailed energy conservation of the advection nonlinearity |
| **Prop (Reading 1):** state-reconstruction ⟹ control-sufficiency | **PROVED** | measurable-selector factorization (sidecar) |
| (i) `Φ_K` non-injective on `supp μ_G` | **CERTIFIED-EMPIRICAL** | twin-state certificate, pre-registered, both regimes |
| (ii) control-sufficient up to `B_δ`, `δ≈0.037` | **CERTIFIED-EMPIRICAL** | kNN convergence POSITIVE + paired fiber-constancy POSITIVE |
| (iii) coupling slaving `R²(R\|Φ_K)≥0.99` | **CERTIFIED-EMPIRICAL** | held-out regression R², calibrated controls, both regimes |
| `K=3` below the determining count | **CERTIFIED (internal)** | non-injectivity at `K=3`; fixed-Galerkin `K*` upper bracket registered |
| `B_δ` is genuine geometry, not artifact | **DEMONSTRATED** | `D_witness ≈ D_candidate` (paired exhibit) |
| portability across `G` | **DEMONSTRATED** (2 points) + argued by construction | §4 table |
| ∞-dimensional NSE-attractor analogue | **HYPOTHESIZED** | Galerkin-attractor upper-semicontinuity cited, not applied |
| refinement-invariance (`N`, projection, objective family) | **OPEN** | the robustness wave (un-run) |

## 7. Glossary to accepted PDE / control language

| Internal term | Standard term | Reference |
| --- | --- | --- |
| `Φ_K` state-insufficient | `Φ_K` not **state-determining**; below the **determining-modes/functionals** count | Foias–Prodi; Cockburn–Jones–Titi 1997 |
| `Φ_K` control-sufficient | **functional observability** of the decision event; `π*` is `𝓕_{Φ_K}`-measurable (Blackwell-sufficient for `J_q`) | Montanari–Motter PNAS 2022; nonlinear 2301.04108 |
| coupling slaving `R≈R(Φ_K)` | a measured local **closure** / **approximate inertial manifold** for the energy observable; Mori–Zwanzig memory ≈ 0 | Foias–Manley–Temam; Parish–Duraisamy 2017 |
| `supp μ_G` | **attractor support** / invariant (SRB-like) measure | standard |
| `J_q` look-ahead trigger | a **data-assimilation / control** readout objective | standard |
| decision surface `Γ` | the optimal-policy **switching manifold** | standard control |

## 8. Cross-references

- [`PDE_C1_SEPARATION_STATEMENT.md`](PDE_C1_SEPARATION_STATEMENT.md) — detailed framing, anti-vacuity ledger, observability positioning (§7).
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md) — Reading 1 proof + the fiber criterion.
- [`PDE_C1_TWIN_STATE_CERTIFICATE.md`](PDE_C1_TWIN_STATE_CERTIFICATE.md) — clause (i).
- [`PDE_C1_KNN_CONVERGENCE_CHECK.md`](PDE_C1_KNN_CONVERGENCE_CHECK.md) + [`PDE_C1_PAIRED_FIBER_CONSTANCY.md`](PDE_C1_PAIRED_FIBER_CONSTANCY.md) — clause (ii) + the boundary set.
- [`PDE_C1_MZ_ENERGY_BUDGET.md`](PDE_C1_MZ_ENERGY_BUDGET.md) — clause (iii) + the Lemma.
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) — the two-regime portability run.
