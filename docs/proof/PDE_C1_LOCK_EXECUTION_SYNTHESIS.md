# PDE C1 Lock Execution Synthesis — v0 through v5

> Step-back consolidation of the Candidate-1 cell-set lock executions
> filed 2026-05-28, written after six runs and four protocol amendments
> all returned `DEFERRED_*` (no substantive verdict). This note exists
> because the right move after v5 is to characterize what we have
> learned and choose a fork — **not** to launch a seventh
> parameter knob. Status: synthesis, unreviewed.

## 1. What was run

All cells are 2D incompressible Kolmogorov flow on `T^2`, Galerkin
`N = 16`, observation a low-Fourier signature, objective a low-band
energy safety trigger with proxy selector `\hat{pi}`. Harness:
[`../../scripts/pde_c1_kolmogorov_cell.py`](../../scripts/pde_c1_kolmogorov_cell.py).

| Cell | Regime | Key param | Verdict | Why |
|---|---|---|---|---|
| v0 | `k_f=4, G=100` | — | `DEFERRED_VACUITY` | laminar fixed point; `damp_fraction = 0` |
| v1 | `k_f=2, G=100` | — | `DEFERRED_VACUITY` | non-trivial attractor but `E_K` conserved to 13 dp; `damp_fraction = 0` |
| v2 | `k_f=2, G=300` | — | `DEFERRED_COVERAGE` | chaotic (`damp_fraction = 0.0086`) but 45k bins, near-vacuity |
| v3 | `k_f=2, G=200` | — | `DEFERRED_VACUITY` | steady-state below transient-contaminated `E_max = 1.07`; `damp_fraction = 0` |
| v4 | `k_f=2, G=200` | `e_max_frac=0.25` | `DEFERRED_COVERAGE` | **`damp_fraction = 0.30`**; coverage fails at K=4 (45k bins) |
| v4-fb | `k_f=2, G=200` | `N=200k` | `DEFERRED_COVERAGE` | `damp_fraction = 0.300`; 139k bins, still 0 evaluated |
| v5 | `k_f=2, G=200` | `K=3` | `DEFERRED_COVERAGE` | `damp_fraction = 0.30`; coverage fails (38k bins) — K-cut hypothesis falsified |

## 2. Four protocol amendments (all pre-registered, none post-hoc)

1. **`delta_proxy_min` vacuity gate** + `DEFERRED_VACUITY` branch —
   a constant proxy can never separate fibers, so a constant-proxy run
   is a non-verdict, not a positive.
2. **Structural-vacuity precedence** — when `damp_fraction` is exactly
   `0` or `1`, vacuity overrides the coverage gate (no sample budget
   resolves a structurally zero proxy).
3. **`e_max_burnin_fraction`** — `E_max` from a steady-state burn-in
   sub-window, excluding transient peaks that biased the threshold
   above the attractor's typical excursions.
4. **`K` as cell-set parameter** — *hypothesis since falsified by v5*
   (see §3).

## 3. Two genuine findings

### Finding A (positive): a discriminative cell exists

At `(k_f = 2, G = 200)` with the steady-state `E_max` (v4/v5),
`damp_fraction ≈ 0.30` — stable across:

- K = 4 vs K = 3 (`0.30014` → `0.2977`),
- `N_sample` = 50,000 vs 200,000 (`0.30014` → `0.300125`).

Roughly 30% of attractor states would trigger `damp_low_band` within
the look-ahead horizon and 70% would not. **The safety objective is
non-trivially action-varying over this attractor.** This is exactly
the premise Reading 2 of the C1 sidecar needs: the objective is not
constant, so the question "is the optimal action constant on
`Phi_K`-fibers?" is non-vacuous here. The discrimination is robust
to both the signature dimension and the sample budget — a real,
reproducible feature of the regime, not a tuning artifact.

### Finding B (negative): hard-bin coverage is governed by attractor dimension

v5 falsified amendment 4's premise. The scaling
`bin_count ∝ (R / h_K)^{d_K}` assumed the attractor fills the
`d_K`-dimensional signature box. It does not — the attractor is a
low-dimensional invariant set, and occupied-bin count tracks its
**box-counting dimension `D_box` at scale `h_K`**, invariant to the
embedding dimension. Halving `d_K` (32 → 18) and coarsening `h_K`
29% cut occupied bins only 16.5% (45,827 → 38,281). The honest
scaling is `occupied_bins ≈ (R / h_K)^{D_box}`.

## 4. The core obstruction

Hard-bin fiber adjudication has a **tolerance-fidelity vs. coverage
tension**:

- `epsilon_K` (hence `h_K`) must be *small* to make a bin a faithful
  approximation of a continuous fiber (protocol §1).
- `h_K` must be *large* enough that bins collect `n_min` samples for
  the minority-fraction test to have power.

For the v4/v5 attractor, a rough estimate (`D_box ~ 3–4`) puts the
coverage-clearing `h_K ≈ 0.05`, i.e. `epsilon_K ≈ 0.21` — ~17% of the
signature-norm scale. At that tolerance, distinct signatures are
lumped into one "fiber" and a coarse bin straddling the safety
boundary can manufacture artificial incompatibility or mask real
structure. The tension is governed by the attractor's measure
concentration and may not be resolvable by binning-parameter tuning
at the `5×10^4`–`2×10^5` sample budget.

Six cells have triangulated this cleanly: the vacuity floor (too weak
a regime → constant proxy) and the coverage ceiling (strong enough
regime → attractor too spread for hard bins) bracket a window that
hard binning does not comfortably fit at feasible sample counts.

## 5. Forks (pick before any further compute)

### Fork A — pivot the adjudication method (principled)

Replace hard binning with a **k-nearest-neighbour / kernel
fiber-locality test**: for each sample, examine its `k` nearest
neighbours in signature space and test whether they share the proxy
action; report the neighbourhood *radius* `r` as a per-sample
fidelity measure. This decouples coverage (every sample gets `k`
neighbours by construction) from fidelity (flag samples whose `r`
exceeds `epsilon_K` as low-fidelity rather than silently lumping
them). It converts the §4 tension from a binary gate failure into a
measurable per-sample distribution — strictly more honest, and it
directly targets the obstruction v5 exposed. Cost: a new protocol
section, new harness path, fresh pre-registration. This is a genuine
methodology pivot, not another knob, and warrants explicit sign-off.

### Fork B — report Finding A as a partial C1 read

The robust `damp_fraction ≈ 0.30` result establishes Reading 2's
*premise* (non-trivial action variation on a known-smooth attractor)
without yet adjudicating fiber-constancy. This is a modest but real
contribution to the C1 sidecar: it confirms the strictness question
is non-vacuous in a registered regime, and documents the
adjudication obstruction as future work. Pairs naturally with the
external-reviewer path (criterion c).

### Fork C — bank C1 lessons, pivot to C2

We have a working harness, a robust discrimination signal, four
pre-registered amendments, and a clearly characterized obstruction.
C2 (shell-model signatures,
[`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md))
is **structurally insulated** from this exact tension — its
predicates are classifier lead-time/false-positive on the Pareto
frontier, not σ-algebra fiber binning, so the coverage-vs-fidelity
wall does not arise. Banking C1 at "discriminative cell found,
adjudication-method obstruction characterized" and opening the C2
empirical leg is defensible.

### Not a fork — `fallback_v5` or v6-by-knob

The v5 disposition explains why `fallback_v5` would re-confirm the
attractor-dimension mechanism at ~90 min cost, and why another
binning-parameter regime cell would re-enter the same bracketed
window. Neither is recommended.

## 6. Recommendation

Fork A is the principled resolution of the obstruction; Fork C is the
pragmatic bank-and-pivot. Either is defensible; both are better than a
seventh binning cell. The choice is a research-direction call for the
owner. Whichever is chosen, **Finding A is filed and durable**: a
low-dimensional signature is robustly control-discriminative on a
known-smooth 2D NSE attractor, independent of signature dimension
(K = 3 vs 4) and sample budget (50k vs 200k).

## 7. Cross-references

- Cell-set dispositions: [v0](PDE_C1_CELLSET_KOLMOGOROV.md),
  [v1](PDE_C1_CELLSET_KOLMOGOROV_v1.md),
  [v2](PDE_C1_CELLSET_KOLMOGOROV_v2.md),
  [v3](PDE_C1_CELLSET_KOLMOGOROV_v3.md),
  [v4](PDE_C1_CELLSET_KOLMOGOROV_v4.md),
  [v5](PDE_C1_CELLSET_KOLMOGOROV_v5.md).
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) — methodology;
  §4 amendment hypothesis falsified by v5 (annotated there).
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  — the C1 sidecar; Finding A supports Reading 2's premise.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — ledger.
