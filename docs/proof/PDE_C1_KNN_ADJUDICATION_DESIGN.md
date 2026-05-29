# PDE C1 — kNN / Kernel Fiber-Locality Adjudication (Design Proposal)

> **Design proposal for sign-off**, filed 2026-05-28 after the C1
> lock-execution synthesis selected Fork A. This document pins the
> *method, statistic, gates, parameters, and cost* of a
> nearest-neighbour fiber-locality adjudicator that targets the
> tolerance-fidelity vs. coverage obstruction characterized in
> [`PDE_C1_LOCK_EXECUTION_SYNTHESIS.md`](PDE_C1_LOCK_EXECUTION_SYNTHESIS.md)
> §4. **Status: proposed, not built, not run.** No harness path or
> protocol section is written until this design is signed off. §7
> lists the open choices that need the owner's call.

## 1. Why this is principled, not another knob

The fiber protocol §1 named two admissible realizations of the
Reading-2 fiber criterion: the **binning** branch (built, exhausted
across v0–v5) and the **disintegration** branch (deferred). This
proposal *is* the disintegration branch, realized nonparametrically.

The Reading-2 criterion asks whether the proxy `\hat{pi}` is constant
on `Phi_K`-fibers up to `mu`-null sets — equivalently, whether the
disintegration conditionals `mu_sigma` (the measure `mu` conditioned
on `Phi_K = sigma`) put all their mass on a single action for
`(mu ∘ Phi_K^{-1})`-a.e. `sigma`. The `k` nearest neighbours of a
sample `u_i` in signature space approximate a draw from `mu`
conditioned on `Phi_K ≈ Phi_K(u_i)` within radius `r_k(u_i)`. The
local action-disagreement among those neighbours estimates the
conditional non-constancy `1 - max_a mu_sigma(a)`. **kNN is a
consistent estimator of the disintegration criterion** — not an ad
hoc third method.

## 2. The obstruction this resolves

Hard binning failed because faithful (`h_K ≈ epsilon_K`) bins were
under-populated: tens of thousands of near-empty cells in the sparse
tail drove `S_eval = 0`, even though the dense core was well sampled
(Finding A: `damp_fraction` stable to 4 dp across sample budgets).

kNN inverts the construction. **Every sample gets exactly `k`
neighbours by construction**, so there are no empty cells. The cost
of sparsity reappears — honestly and measurably — as the
neighbourhood *radius* `r_k(u_i)`: in dense regions `r_k` is small
(faithful local fiber); in sparse regions `r_k` is large (the test is
being conducted at a coarser scale than the tolerance allows). We
adjudicate only where `r_k ≤ epsilon_K` and *report* the fraction
where it holds. Coverage and fidelity become per-sample and
co-measured, instead of a binary bin-population gate.

## 3. Per-sample statistic

For each sample `u_i` (i = 1..N):

- `N_k(u_i)` = the `k` nearest neighbours of `u_i` in signature space
  under the L2 metric (the same metric defining the `epsilon_K`
  tolerance ball), **including `u_i` itself** (so `|N_k| = k`).
- `r_k(u_i)` = L2 distance from `u_i` to its `k`-th nearest neighbour
  (the neighbourhood radius / fidelity measure).
- `a_maj(u_i)` = majority proxy action in `N_k(u_i)`; ties broken to
  `no_op` (inherits the cell-set action tie-break).
- `m_i` = local minority fraction = `1 - |{u ∈ N_k(u_i) : \hat{pi}(u)
  = a_maj}| / k`.

## 4. Gates and verdict

Pre-registered, in order — parallels the binning protocol §5 so the
two are auditable side by side:

1. **Vacuity gate (unchanged).** Global `damp_fraction` must lie in
   `[delta_proxy_min, 1 - delta_proxy_min]`; exact `0`/`1` →
   `DEFERRED_VACUITY` (structural precedence unchanged).
2. **Fidelity-coverage gate (replaces bin-coverage).** Let
   `F = {i : r_k(u_i) ≤ epsilon_K}` be the fidelity-passing set.
   If `|F| / N < S_pos`, record `DEFERRED_FIDELITY_COVERAGE` and stop
   — the attractor is too sparse at scale `epsilon_K` for a faithful
   local fiber test at this `k` and `N`. (Distinct receipt name from
   `DEFERRED_COVERAGE` so the binning and kNN deferrals are not
   conflated.)
3. **Verdict (on `F`).**
   - If a pre-registered positive fraction of `F` has `m_i >
     delta_action` → **`PDE-C1-NEG-A`** (fiber incompatibility:
     fibers exist where the proxy is not locally constant).
   - Else → **`STRICTNESS_WITNESS_POSITIVE`**: the proxy is locally
     constant on fibers across the fidelity-passing coverage, and the
     proxy is non-trivially discriminative (vacuity gate passed).

The "positive fraction" threshold for NEG-A needs a pre-registered
value (a single incompatible sample shouldn't flip the verdict; a
robust positive-mass region should). Proposed: NEG-A iff
`|{i ∈ F : m_i > delta_action}| / |F| > delta_incompat`, with
`delta_incompat` a new small pre-registered constant (proposed 0.01).

## 5. Parameters

| Param | Value | Source |
|---|---|---|
| `k` | 30 | parallels binning `n_min = 30` (binomial SE ~0.04 on `m ~ 0.05`, so `delta_action = 0.10` sits ~1.25 SE above noise) |
| `epsilon_K` | `0.05 sqrt(2 E_max)` | inherited from cell-set (fidelity radius threshold = the tolerance ball) |
| `delta_action` | 0.10 | inherited (per-neighbourhood minority threshold) |
| `delta_proxy_min` | 0.01 | inherited (vacuity gate) |
| `S_pos` | 0.50 | inherited semantics (now fidelity coverage, not bin coverage) |
| `delta_incompat` | 0.01 | **new** — positive-mass threshold for filing NEG-A |
| `e_max_burnin_fraction` | 0.25 | inherited from v4 (steady-state E_max) |

All pre-registered before any kNN run, same discipline as the binning
cells. Post-hoc change → `PDE-C1-NEG-B`.

## 6. Computational approach

- **Reuse the deterministic integration.** The Galerkin solve is
  seeded (`random_seed = 20260528`); re-running `lock_v5`'s regime
  reproduces the exact signature set. The kNN adjudicator is a new
  *backend* over the stored signature array — the ~20-min integration
  is unchanged; only the post-sampling aggregation swaps.
- **Signature storage.** The current harness retains
  `sample_signatures` in memory during `run_cell` but only writes
  per-sample rows for `sample_count ≤ 5000`. Add a `--write-signatures`
  path (or an in-process kNN backend) so the 50k–200k signature array
  is available to the adjudicator.
- **Neighbour search.** `scipy.spatial.cKDTree` for d ≤ ~20 (the
  K = 3 regime, d = 18, is ideal); for d = 32 (K = 4) KD-trees degrade
  and a `BallTree` or approximate-NN backend is preferable. Cost for
  50k points in 18-D: seconds-to-minutes — negligible beside the
  integration. This is why the **K = 3 / d = 18 regime (v5) is the
  natural first kNN target**: discrimination already confirmed robust
  there, and the neighbour search is cheap and exact.

## 7. Open choices for sign-off

1. **First kNN target cell.** Recommend re-using the **v5 regime**
   (`k_f = 2, G = 200, K = 3, e_max_burnin_fraction = 0.25`) — Finding
   A confirmed `damp_fraction ≈ 0.30` there and d = 18 keeps kNN exact
   and cheap. Alternative: the v4 regime (K = 4, d = 32) to test at the
   richer signature, accepting BallTree/approx-NN cost.
2. **`k` value.** Recommend `k = 30` (parallels `n_min`). Larger `k`
   (e.g. 50) gives tighter `m_i` estimates but larger `r_k` (worse
   fidelity coverage). Genuine tradeoff.
3. **`delta_incompat`.** Recommend 0.01 (a robust positive-mass region
   must flag, not a lone sample). This is the NEG-A sensitivity knob.
4. **Replace vs. augment binning.** Recommend kNN becomes the primary
   adjudicator for high-dimensional-attractor cells, with binning
   retained in the protocol for low-dim cases where it works. I.e.
   add a kNN protocol section, do not delete the binning section.
5. **Build location.** Recommend a new `--adjudicator knn` flag on the
   existing harness (shares integration, swaps backend) over a separate
   script (avoids signature-dump round-trip).

## 8. Pre-registered expectations (how this could still fail)

- **Best case.** A meaningful fraction of the dense core passes
  fidelity (`|F|/N ≥ 0.5`), `m_i ≈ 0` across `F` → first
  `STRICTNESS_WITNESS_POSITIVE`; or a positive-mass incompatible
  region → first `PDE-C1-NEG-A`. Either is the first substantive C1
  fiber-adjudication read.
- **Fidelity-coverage still fails.** If even the dense core is too
  sparse at `epsilon_K` for `k = 30` neighbours, `|F|/N < 0.5` →
  `DEFERRED_FIDELITY_COVERAGE`. This would mean the attractor measure
  is diffuse at the tolerance scale — a genuine finding (the fiber is
  not densely sampled anywhere), pointing to either a larger `N`, a
  larger `epsilon_K` (with the same fidelity caveat as binning), or a
  reconsideration of whether the continuous-fiber criterion is the
  right object for this attractor. Pre-registered, not a surprise.
- **The honest improvement either way.** Unlike the binning
  `S_eval = 0`, kNN reports the *distribution of `r_k`* — so even a
  deferral comes with a quantitative map of how far from fidelity the
  attractor sits, which the binning receipts could not provide.

## 9. Cross-references

- [`PDE_C1_LOCK_EXECUTION_SYNTHESIS.md`](PDE_C1_LOCK_EXECUTION_SYNTHESIS.md)
  — the v0–v5 synthesis; §5 Fork A is this proposal.
- [`PDE_C1_FIBER_PROTOCOL.md`](PDE_C1_FIBER_PROTOCOL.md) — §1 names the
  disintegration branch this realizes; the binning §5 is the structure
  this parallels. A new kNN adjudication section is added there *after*
  sign-off.
- [`PDE_C1_CELLSET_KOLMOGOROV_v5.md`](PDE_C1_CELLSET_KOLMOGOROV_v5.md)
  — the recommended first kNN target regime.
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  — the C1 sidecar; Reading 2's fiber criterion is what this
  adjudicates.
