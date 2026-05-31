# PDE C1 — Objective-Overlap Discriminator (control-sufficiency vs near-determination)

> **Pre-registration**, filed 2026-05-31. **Status: UNRUN.** Result section
> (§12) is appended only after the two verdict-bearing runs land.
>
> This probe stress-tests the C1 result's central framing. The proposition
> ([`PDE_C1_PROPOSITION.md`](PDE_C1_PROPOSITION.md)) presents a **separation**:
> `Φ_K` state-insufficient (i) yet control-sufficient (ii). The 2026-05-31
> target-validity re-examination argued (ii) may be a near-corollary of (iii)
> coupling-slaving **because the registered objective lives in `Φ_K`'s own
> closed low-band-energy subspace** — so "control-sufficient" would really mean
> "`Φ_K` is an approximate inertial manifold that near-determines all physical
> content (energy FVE 0.997 / enstrophy 0.993)," not a special low-dimensional
> control shadow. This discriminator decides between those readings.

## 1. The claim under test

Every C1 control-sufficiency read to date (`E_low`, `Z_low`) used a **low-band**
objective — the same `K=3` band `Φ_K` observes. The proposition's own §2(iii)
gives `dE_low/dt = g(Φ_K) + R` with `g` exactly `Φ_K`-measurable (Lemma
`T_LLL≡0`) and `R` 99% `Φ_K`-slaved, so a low-band-energy objective is
`Φ_K`-predictable nearly by construction. If control-sufficiency holds **only
because the objective is `Φ_K`-predictable**, then it is near-determination, not
control. The falsifiable test:

> Across a slate of objectives spanning the band/dissipation axis, does
> control-sufficiency (`a_mm → 0`) **track** objective-predictability
> `R²(M|Φ_K)`? If `a_mm` rises exactly as `R²(M|Φ_K)` falls, control-sufficiency
> **is** predictability (near-determination). If `a_mm` stays ≈ 0 even where
> `R²(M|Φ_K)` is low, `Φ_K` controls what it cannot predict — a genuine control
> shadow.

## 2. Objective slate (pinned)

Each objective is the registered construction — `τ=5.0` lookahead-max,
`E_max = quantile_{q=0.70}` on a disjoint held-out calibration window, label
`π(u)=1[M(u)>E_max]`, `damp_fraction = 0.30` by construction (§4 of
[`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) inherited
verbatim) — with **only the observable changed**. `Φ_K` = `K=3` low band
(`d=18`); `Q_K` = the 211 complementary high modes (422 real DOF).

| # | objective `M` | observable (per state `u`) | band / moment | predicted `R²(M\|Φ_K)` | predicted `a_mm` |
|---|---|---|---|---|---|
| 1 | `E_low` | `low_energy` (registered anchor) | low energy | highest | POSITIVE |
| 2 | `Z_low` | `low_enstrophy` | low enstrophy | high | POSITIVE |
| 3 | `E_high` | `low_energy` over `high_indices` (`‖Q_K u‖²`) | high-band energy | high (FVE 0.9994) | **POSITIVE** ← `Φ_K` controls the *complementary* band because it near-determines it |
| 4 | `Z_high` | `low_enstrophy` over `high_indices` | high-band enstrophy | lower | POSITIVE→marginal |
| 5 | `palinstrophy` | `∑_all |k|²·|ω̂/scale|²` | dissipation-rate (`k²`-weighted) | lower | marginal→NEG-A |
| 6 | `top_shell` | energy in the highest-`|k|` shell | the under-determined dissipation range (per-DOF ~0.71) | lowest | NEG-A or underpowered |

Each high-band / dissipation observable mirrors the existing `low_energy` /
`low_enstrophy` method with a different index set (`high_indices`, all-indices
with `k²` weight, top-shell indices) — see §5. The slate is ordered by **expected
`R²(M|Φ_K)`**; the discriminator is whether the measured `a_mm` ordering tracks
it.

## 3. Three pre-gated measurements per objective × regime

1. **Power gate (precondition).** Portability `damp_fraction(A) ∈ [0.20, 0.40]`
   (inherited from [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md)
   §6). An objective whose held-out `damp_fraction` drifts out of band is
   **degenerate/underpowered** — its `a_mm` is reported but **not interpreted as
   control-insufficiency** (the YM target-validity lesson: a NEG-A on a
   noise/degenerate objective is uninformative). Degenerate objectives count
   toward the `UNDERPOWERED` verdict (§7), not toward tracking.
2. **Control-sufficiency `a_mm`.** The existing kNN sweep (`k ∈
   {10,15,20,25,30,40,50}`), threshold-free `mean_minority` intercept `a_mm`:
   `≤ 0.005` POSITIVE / `≥ 0.015` NEG-A / between = ambiguous. Classifier
   unchanged from the regime-generality runs.
3. **Objective-predictability `R²(M|Φ_K)`.** Reuse `r2_from_signature` (the MZ
   slaving regressor: held-out 70/30, `HistGradientBoostingRegressor`) on the
   per-sample objective **value** `M(u)`, with its existing estimator-validity
   gate `r2_g > 0.90` AND `r2_perm < 0.10`. If the estimator gate fails, the
   regime's `R²` row is invalid and the run re-checks before interpreting.

## 4. Discriminating statistic (pre-registered)

Across the 6 objectives, at each regime:

```text
tracking := Spearman corr( a_mm , 1 − R²(M|Φ_K) )   over the powered objectives.
```

`tracking ≥ 0.70` (and the per-objective predictions of §2 holding) ⇒
control-sufficiency is predictability. `tracking < 0.70` with `a_mm` flat-near-0
across the predictability range ⇒ control is not predictability.

## 5. Harness extension (specified; implemented as the next step)

In `scripts/pde_c1_kolmogorov_cell.py`, additively (the `E_low`/`Z_low` path and
every existing preset stay **bit-for-bit unchanged**):

- new observables mirroring `low_energy`/`low_enstrophy`: `high_energy`,
  `high_enstrophy` (over `self.high_indices`), `palinstrophy`
  (`∑_all |k|²·|ω̂/scale|²`), `top_shell_energy` (highest-`|k|` shell indices);
- a **slate objective-mode** that, in **one** `τ`-lookahead pass per sample,
  records the lookahead-max of all 6 observables (shared trajectory → ~2 runs
  total, not 12);
- presets `lock_disc_g200` / `lock_disc_g300` reusing the `lock_v7_g200/g300`
  integration setup (`k_f=2, K=3, d_K=18, dt=0.01, τ=500 steps`, held-out split
  50k/5k/50k), `objective = portable-quantile`, `q = 0.70`;
- per objective: label → kNN sweep `a_mm`, and `r2_from_signature(M)` for
  `R²(M|Φ_K)`; emit a per-objective row `{damp_fraction, a_mm verdict, R², powered}`.

## 6. Run order (pinned)

```text
1. lock_disc_g200   (anchor self-check + full slate)
2. lock_disc_g300   (regime replication of the tracking)
   [power gate + estimator gate checked per objective before interpreting a_mm]
```

~40–50 min each; none inline under the 10-minute rule.

## 7. Branches and interpretation (fixed before any read)

| outcome | verdict | meaning |
|---|---|---|
| `tracking ≥ 0.70`; physical objectives (`E_low,Z_low,E_high`) POSITIVE *because* `R²` high; only under-determined (`palinstrophy,top_shell`) NEG-A | **`PDE-C1-DISC-CONFIRM`** | control-sufficiency = near-determination. Downgrade the proposition's "state-insufficient yet control-sufficient **separation**" to "`Φ_K` is an approximate inertial manifold; the objective reads only resolved scales." Clause (iii) (local closure / AIM) is the defensible core; (i)+(ii) as a *separation* oversells. |
| `a_mm ≈ 0` even where `R²(M|Φ_K)` is low (no tracking) | **`PDE-C1-DISC-REFUTE`** | `Φ_K` controls objectives it cannot predict → a genuine low-dimensional control shadow; the inroad is **stronger** than the target-validity critique allows. C1's separation framing is vindicated and strengthened. |
| every high-mode objective fails the power gate | **`PDE-C1-DISC-UNDERPOWERED`** | the cell is so marginal that no high-mode objective can pose a control question — confirms the proposition §3 ("missed modes read by no physical objective") from the other side; the separation is vacuous-for-physical-purposes at this `G`. |
| post-hoc change to slate / observables / gates / `q` after a read | **`PDE-C1-DISC-NEG-B`** | discipline breach; voids. |

## 8. What this does / does not establish

- **Does:** adjudicate whether the *existing* C1 control-sufficiency reads reflect
  a control shadow or near-determination, on the *same* cell and machinery.
- **Does not:** change clauses (i) state-insufficiency or (iii) coupling-slaving
  (both objective-free, untouched); produce any NSE theorem; promote or demote C1
  by itself — a `CONFIRM` recommends a framing downgrade for owner + external
  review to ratify, it does not unilaterally rewrite the proposition.

## 9. Pre-registration discipline

- All of §2–§7 fixed before any verdict-bearing run.
- The power gate and estimator gate are **preconditions**, not tunables; a
  failure is filed (`UNDERPOWERED` / re-check), not rescued by retuning.
- The `E_low` slate member is the **anchor self-check**: it MUST reproduce the
  registered v7 read (`a_mm ≈ −0.0008` POSITIVE at G=200, `damp ≈ 0.30`) and a
  high `R²(M|Φ_K)`; if it does not, the slate harness is mis-wired and both runs
  void (mirrors the YM `γ_held`-must-fail self-check).
- Post-hoc reinterpretation of a `CONFIRM` as anything other than a framing
  downgrade, or of a `REFUTE`/`UNDERPOWERED` to rescue the separation, is
  `PDE-C1-DISC-NEG-B`.

## 10. Cross-references

- [`PDE_C1_PROPOSITION.md`](PDE_C1_PROPOSITION.md) — the §2 clauses this probes;
  a `CONFIRM` recommends downgrading the (i)+(ii) "separation" framing.
- [`PDE_C1_MZ_ENERGY_BUDGET.md`](PDE_C1_MZ_ENERGY_BUDGET.md) — clause (iii); the
  `r2_from_signature` machinery reused for `R²(M|Φ_K)`.
- [`PDE_C1_REGIME_GENERALITY_v1.md`](PDE_C1_REGIME_GENERALITY_v1.md) — the
  objective construction, held-out split, portability gate, and kNN gates
  inherited verbatim.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — ledger.

## 12. Result (2026-05-31)

Both verdict-bearing runs executed (~45 min each). **Anchor self-check PASSED at
both regimes** — `E_low` reproduces the v7 read (`a_mm=−0.00079`/damp 0.300 at
G=200; `a_mm=+0.00058`/damp 0.269 at G=300; R²≈0.98) — so the slate is correctly
wired and the reads are interpretable. Negative controls clean (`R²(perm)≈0`,
`est_ok` all True). Receipts: `results/proof/c1-disc-g200/`, `.../c1-disc-g300/`.

**G=200 → `PDE-C1-DISC-INCONCLUSIVE`, tracking Spearman(a_mm, 1−R²) = −0.75.**

| objective | damp | a_mm | kNN | R²(M\|Φ_K) |
|---|---|---|---|---|
| E_low | 0.300 | −0.00079 | POSITIVE | 0.9955 |
| Z_low | 0.300 | 0.00097 | POSITIVE | 0.9998 |
| E_high | 0.300 | 0.00000 | INCONCLUSIVE | 0.9948 |
| Z_high | 0.300 | 0.00323 | POSITIVE | 0.9998 |
| palinstrophy | 0.300 | 0.19494 | **NEG-A** | 1.0000 |
| top_shell | 0.300 | 0.00000 | INCONCLUSIVE | 0.8770 |

**G=300 → `PDE-C1-DISC-REFUTE`, tracking = −1.0 (3 powered).**

| objective | damp | a_mm | kNN | R²(M\|Φ_K) |
|---|---|---|---|---|
| E_low | 0.269 | 0.00058 | POSITIVE | 0.9797 |
| Z_low | 0.000 | — | underpowered | 0.2485 |
| E_high | 0.000 | — | underpowered | 0.9582 |
| Z_high | 0.000 | — | underpowered | 0.9972 |
| palinstrophy | 0.265 | 0.00110 | POSITIVE | 0.9997 |
| top_shell | 0.296 | 0.00000 | INCONCLUSIVE | 0.7602 |

### Reading — the pre-registered target-validity critique is NOT confirmed

1. **Φ_K predicts essentially every objective well** (R² 0.76–1.00 at both
   regimes), including the high-band (`E_high` 0.99, `Z_high` 0.997) and the
   dissipation range (`top_shell` 0.88/0.76). **There is no powered objective Φ_K
   cannot predict at this cell** — the critique's premise (a low-R² high-mode
   objective where Φ_K would fail control-sufficiency) is empirically false here.
   This *strengthens* the approximate-inertial-manifold reading but removes the
   "objective lives in the easy closed subspace" basis of the critique.
2. **Control-sufficiency does NOT track predictability** — and where it varies it
   runs *opposite* to the prediction (tracking −0.75 / −1.0, not ≥ +0.70). The
   striking case: **`palinstrophy` at G=200 has R²=1.0000 (perfectly
   Φ_K-predictable) yet `a_mm`=0.195 (NEG-A, control-INsufficient)** — a
   predictable-but-not-control-sufficient objective, the opposite mechanism from
   the critique and unanticipated by both the critique and the proposition.

### Honest caveats

- **g300 REFUTE is weak.** Three of six objectives (`Z_low, E_high, Z_high`) went
  **degenerate** (damp→0, underpowered) at G=300 — the same intermittency
  non-stationarity the v0→v1 fix fought — so the high-band objectives that most
  directly test the critique are missing at G=300; the REFUTE rests on
  `E_low`+`palinstrophy` POSITIVE.
- **The verdict logic assumed objectives spanning R²; they don't** (all high), so
  the tracking statistic tests a relationship without its assumed structure, and
  `E_high`/`top_shell` returned INCONCLUSIVE kNN reads. The robust raw fact is #1
  (uniform high R²), not the tracking label.
- **`palinstrophy` flips** NEG-A (G=200) → POSITIVE (G=300); its control read is
  not regime-robust, consistent with a burst-dominated look-ahead-max (unstable
  decision surface) — possibly an artifact worth a separate look.

### Disposition

The probe did its adversarial job: the target-validity critique is **not
supported**. The C1 control-sufficiency reading is **not** downgraded by this
test, and Φ_K's predictability extends to the high-mode / dissipation objectives.
The "separation" framing stands as-is pending external review; the proposition is
unchanged. New open thread (separate probe, not pursued here): the
predictable-but-not-controllable `palinstrophy` behaviour — real phenomenon vs
burst artifact.
