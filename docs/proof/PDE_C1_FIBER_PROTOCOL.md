# PDE C1 Fiber Protocol — Tolerance and Binning

> Methodology artifact for the continuous-fiber / tolerance gate of
> [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md)
> section 6 ("Open measure item") and the provisional-selector gate of
> section 3 ("Open review item"). Status: drafted, unreviewed,
> 2026-05-28. This file pins parameter *roles* and the adjudication
> *procedure*; concrete parameter *values* belong in the cell-set v0 (or
> a v0 patch). Alternative protocols remain admissible if a reviewer
> prefers them; this is the v0 default.

## Entry And Gate

This protocol enters from
[`SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) Promotions:

> The v0 comparator has been tightened: ... Promotion criterion (d) is
> staged but still needs the continuous-fiber / tolerance protocol;
> criterion (b) is partially closed (procedure defined, not executed).

and from [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md)
section 6:

> In the continuous setting, exact `Phi_K`-fibers are typically
> `mu`-null. A later patch must replace the finite-MDP phrase
> "positive-measure fibers" with either a regular-conditional /
> disintegration statement or a pre-registered numerical binning /
> tolerance scheme.

The protocol below is the **binning + tolerance** branch of the
disjunction. The disintegration branch remains an admissible
alternative; if a reviewer prefers it, the cell-set's Reading-2 fiber
criterion can be ported to a regular-conditional statement without
discarding this artifact (the binning instance then serves as the
numerical approximation of the disintegration).

## Claim Boundary

This protocol does **not** claim:

- a measure-theoretic disintegration theorem for the Kolmogorov-flow
  attractor measure `mu_SRB`;
- that binning resolves all continuous-measure issues — only that it
  produces an executable adjudication predicate under a pre-registered
  parameter set;
- that the proxy selector `\hat{pi}` is `J`-optimal — it is named here
  as a proxy with explicit substitution conditions, not as a derived
  optimum;
- that the support-level twin-state certificate (cell-set v0 § 4.1
  attractor-support caveat) is closed — it is named here as a separate
  open item with a one-line spec, instantiation deferred.

It claims only a runnable predicate structure:

1. a tolerance object on signature space defines a continuous fiber
   relaxation;
2. a binning lattice defines an executable numerical approximation of
   the tolerance object;
3. a per-bin majority-action selector with a minority-fraction
   sub-predicate adjudicates the Reading-2 fiber criterion on bins
   with positive sample count;
4. a pre-registered coverage gate decides when the verdict is
   interpretable.

## Local Symbols

| Symbol | Definition |
| --- | --- |
| `Phi_K` | Signature map from `X` to `R^{d_K}` (cell-set v0 § 2). |
| `epsilon_K` | Tolerance radius on signature space `R^{d_K}`. |
| `B_K` | Pre-registered bin partition of `R^{d_K}` (lattice spacing, alignment, bounding box). |
| `b` | A bin in `B_K`. |
| `n(b)` | Number of sampled states from `mu_SRB` falling in bin `b`. |
| `n_min` | Pre-registered minimum sample count per bin required to evaluate that bin. |
| `\hat{pi}(u)` | Proxy action label per state, computed from the cell-set v0 § 3 deterministic `E_K` overshoot check. |
| `\hat{pi}_bin(b)` | Majority proxy action in bin `b` over states with `n(b) >= n_min`. |
| `delta_action` | Pre-registered minority-fraction threshold; a bin with minority fraction above this is flagged as fiber-incompatible. |
| `S_pos` | Pre-registered minimum fraction of evaluated bins required before the verdict is interpreted (coverage gate). |
| `delta_proxy_min` | Pre-registered lower bound on the proxy-action discrimination floor; if the global `damp_fraction` is outside `[delta_proxy_min, 1 - delta_proxy_min]`, the proxy is essentially constant and the predicate is non-discriminative (vacuity gate). |
| `N_sample` | Pre-registered number of post-burn-in samples drawn from `mu_SRB`. |

All symbols above are parameter **slots**. Their values are pinned in
the cell-set v0 (or a patch), not here.

## 1. Tolerance Object (Theoretical)

The continuous-fiber predicate of the Reading-2 fiber criterion is
relaxed to a tolerance ball on signature space:

```text
C_sigma^{epsilon_K} = { u in X : ||Phi_K(u) - sigma||_{R^{d_K}} <= epsilon_K }.
```

The Reading-2 condition becomes: the proxy selector is constant on
`epsilon_K`-tolerance balls, up to a pre-registered minority fraction
`delta_action`, on a pre-registered coverage fraction `S_pos` of the
support.

This is the object the C1 sidecar's fiber criterion now refers to in
the continuous setting. It is named here to give the criterion an
executable continuous-measure surface; section 2 below provides the
executable approximation.

## 2. Binning Instance (Runnable)

The runnable predicate replaces `epsilon_K`-tolerance balls with bins
from a pre-registered lattice partition `B_K` of `R^{d_K}`:

- **Lattice.** `B_K` is a uniform cubic lattice on `R^{d_K}` with
  per-coordinate spacing `h_K` chosen so that the typical bin diameter
  is approximately `epsilon_K`. The bounding box is the empirical
  range of `Phi_K(u)` over a pre-registered burn-in trajectory.
- **Alignment.** Lattice origin is pinned at the bounding-box corner
  computed during burn-in; alignment is *not* re-optimised after
  sampling.
- **Bin** `b in B_K` is the open lattice cell.

The bin lattice approximates the tolerance ball; for a state `u`, the
bin containing `Phi_K(u)` is an `O(epsilon_K)`-tolerance neighbourhood
of `Phi_K(u)` in signature space. Approximation faithfulness is itself
review-checkable: a reviewer who prefers exact tolerance balls can
replace `B_K` with a ball-cover at the cost of slower adjudication.

## 3. Proxy Selector

The cell-set v0 § 3 deterministic `E_K` overshoot check is renamed
here as a proxy label, with explicit substitution conditions:

```text
\hat{pi}(u) = damp_low_band  iff  the no_op evolution from u carries
                                  E_K above E_max at some t in [0, tau];
                = no_op       otherwise.
```

**Proxy status.** `\hat{pi}` is *not* derived from a `J`-optimal
calculation; it is a deterministic label computed from the truncated
dynamics under the registered objective. The protocol adjudicates the
Reading-2 fiber predicate under `\hat{pi}`. If an external reviewer
treats `\hat{pi}` as faithful to the intended objective, the protocol
result reads as the Postulate-1 predicate. If the reviewer requires a
derived `J`-optimal selector, this protocol's result reads as a proxy
adjudication and is replaced by re-running steps 4–6 with the
reviewer's selector substituted for `\hat{pi}`.

This resolves cell-set v0 § 3 "Open review item" by **naming the
proxy explicitly** and pinning the substitution procedure, rather
than by adding a cost-of-damping term. Both branches remain
admissible at review.

## 4. Per-Bin Majority Selector And Minority Fraction

For a bin `b` with `n(b) >= n_min`:

```text
\hat{pi}_bin(b) = argmax_{a in A_ctrl} |{u in b : \hat{pi}(u) = a}|.
m(b)            = 1 - |{u in b : \hat{pi}(u) = \hat{pi}_bin(b)}| / n(b).
```

`\hat{pi}_bin(b)` is the bin-local majority action; ties broken
deterministically by action ordering pinned in the cell-set v0.
`m(b) in [0, 1/2]` is the bin-local minority fraction.

**Fiber-incompatibility flag.** A bin is flagged fiber-incompatible
iff `m(b) > delta_action`.

## 5. Adjudication Procedure

Pre-registered, in this order:

1. **Pin parameters.** `epsilon_K`, `h_K`, `n_min`, `delta_action`,
   `S_pos`, `N_sample`, burn-in length, integration step, action
   tie-break order. Fixed in the cell-set v0 (or v0 patch) before
   any sampling.
2. **Burn-in.** Integrate the truncated Galerkin NSE from a fixed
   initial condition; discard the burn-in window.
3. **Sample.** Draw `N_sample` post-burn-in states approximating
   independent samples from `mu_SRB` (e.g. sub-sampling at intervals
   of one Lyapunov time, pinned in v0).
4. **Bin.** Assign each sample to its bin in `B_K`.
5. **Label.** Compute `\hat{pi}(u)` for each sample using the
   deterministic overshoot check.
6. **Per-bin aggregation.** For each bin with `n(b) >= n_min`, compute
   `\hat{pi}_bin(b)` and `m(b)`.
7. **Coverage gate.** Let `S_eval = |{ b : n(b) >= n_min }| /
   |{ b : n(b) >= 1 }|`. If `S_eval < S_pos`, the result is **deferred**
   for insufficient coverage; record and stop without verdict.
8. **Vacuity gate.** Compute the global proxy-action distribution:
   `damp_fraction = (sum over samples of [\hat{pi}(u) == damp_low_band])
   / N_sample`. If
   `damp_fraction < delta_proxy_min` or
   `damp_fraction > 1 - delta_proxy_min`,
   the proxy selector is essentially constant on the entire sampled
   support: every fiber is trivially action-consistent and the
   strictness predicate has no discriminative content. Record as
   **`DEFERRED_VACUITY`** and stop without verdict.
9. **Verdict.**
   - If any evaluated bin has `m(b) > delta_action`, **file
     `PDE-C1-NEG-A`** at this cell.
   - Otherwise, the cell is a **strictness-witness positive**: under
     the proxy `\hat{pi}` and the registered binning, the fiber
     criterion holds on the evaluated coverage *and* the proxy
     selector exhibits non-trivial action discrimination on the
     sampled support.

Both deferral branches in steps 7 and 8 are structurally distinct from
the filed-negative and filed-positive branches: they are the protocol's
named ways of refusing to interpret an under-sampled or
non-discriminative adjudication attempt, and they do **not** count as a
positive.

## 6. Support-Level Certificate (Bridge)

The cell-set v0 § 4.1 attractor-support caveat introduces a twin-state
search certificate to upgrade the non-injectivity claim from `B_abs`
to `supp(mu_SRB)`. That certificate is a separate kind of artifact —
a sampling-based numerical certificate, not a methodology
document — and is named here as a follow-up item, not instantiated.

**One-line spec for the deferred certificate.** Sample
`N_twin >= N_sample` states from `mu_SRB`; for each pair `(u_i, u_j)`
with `||Phi_K(u_i) - Phi_K(u_j)|| <= epsilon_K`, compute the
high-mode separation `||Q_K(u_i) - Q_K(u_j)||`; if a pre-registered
fraction of such pairs has high-mode separation at least `delta_H`,
non-injectivity on `supp(mu_SRB)` is certified. Parameters
`N_twin`, `delta_H`, sampling interval, acceptance count pinned at
file time of the certificate, not here.

This protocol's verdict in section 5 remains valid on `B_abs` without
the support-level certificate. If a reviewer requires
`supp(mu_SRB)`-level state insufficiency before the C1 strictness
witness is treated as load-bearing, the certificate is the precondition.

## 7. Parameter Pinning Discipline

The protocol commits to the following non-negotiable order of
operations, per the spec self-consistency rule
([`../SCIENTIFIC_CRITERIA.md`](../SCIENTIFIC_CRITERIA.md)):

- `epsilon_K`, `h_K`, `n_min`, `delta_action`, `S_pos`, `delta_proxy_min`,
  `N_sample`, burn-in length, integration step, action tie-break order,
  lattice bounding box, lattice alignment are pinned **before** step 3
  (sampling).
- Re-tuning any parameter after step 6 (aggregation) and re-running
  steps 7–8 counts as filing `PDE-C1-NEG-B` (cell-set drift /
  post-hoc tuning). Note: this introduces the negative-receipt label
  `PDE-C1-NEG-B` on the C1 side, parallel to the
  `PDE-C2-NEG-B` receipt on the C2 side
  ([`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  § Pre-Registered Negative). The original `PDE-C1-NEG` becomes
  `PDE-C1-NEG-A` retroactively for parallelism; existing references
  to `PDE-C1-NEG` should be read as `PDE-C1-NEG-A`.
- Refining the binning lattice after step 6 to make a marginal verdict
  flip is `PDE-C1-NEG-B`.
- Adding bins post hoc that move the verdict is `PDE-C1-NEG-B`.

The deferred-insufficient-coverage branch (step 7 of section 5) is
**not** `PDE-C1-NEG-B`. Increasing `N_sample` to escape a deferral and
re-running, with all other parameters held, is admissible iff the
increase is pre-registered as a fall-back path before sampling.

The deferred-vacuity branch (step 8 of section 5) is also **not**
`PDE-C1-NEG-B`. It indicates the cell is in a non-discriminative
regime — the proxy selector is constant on `mu_SRB`-support so the
strictness predicate has no operational content. **No fall-back is
admissible to escape a vacuity deferral on the same cell.** Relaxing
`delta_proxy_min` post hoc to re-interpret a constant-proxy run as a
strictness-witness positive files `PDE-C1-NEG-B` (the predicate was
moved, not the data). Re-pinning to a discriminative regime
(higher `G`, different `k_f`, alternative forcing) requires a *new
cell-set instance* (e.g. v1), filed as a separate artifact rather
than as a retune of the original cell.

## 8. What This Protocol Closes And Does Not Close

**Closes:**

- **Criterion (b)(ii) — continuous-fiber measure protocol.** The
  Reading-2 fiber criterion now has a runnable predicate on bins,
  with the tolerance object as its continuous-measure relaxation.
- **Criterion (b)(i) — provisional selector.** `\hat{pi}` is named
  explicitly as a proxy with a pinned substitution procedure under
  reviewer redefinition. The cost-of-damping branch remains
  admissible but is not required for adjudication.
- **Verdict-rule completeness gap (added 2026-05-28).** Section 5's
  vacuity gate distinguishes a constant-proxy run from a
  fiber-coherent run. Without it, a cell whose objective is trivially
  satisfied by a single action would file a vacuous strictness-witness
  positive.

**Bridges (named, not closed here):**

- **Criterion (b)(iii) — support-level certificate.** Named in
  section 6 with a one-line spec; instantiation deferred to a
  dedicated certificate artifact.

**Does not close:**

- Criterion (a) — Front-A vacuity rebuttal remains an external-review
  question on the C1 sidecar, not on this protocol.
- Criterion (c) — named external PDE reviewer remains open.
- Criterion (d) — pre-registered failure boundary moves from "staged"
  to **"closed pending external review"**. The protocol pins the parameter
  roles, and the Kolmogorov cell-set v0 section 7 now pins their concrete
  values.

## 9. Cross-File Consequences

The cell-set v0 patch requested by this protocol landed in
[`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) section 7.

If this protocol survives review:

- update [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md)
  Promotions to keep criterion (d) closed at the artifact level and to
  adjudicate criterion (b) once the registered lock receipt lands;
- introduce `PDE-C1-NEG-A` / `PDE-C1-NEG-B` as the parallel labels
  to `PDE-C2-NEG-A` / `PDE-C2-NEG-B`. Reading `PDE-C1-NEG` as
  `PDE-C1-NEG-A` is the retroactive convention.

If this protocol fails review:

- file the named negative against this artifact (not against the C1
  sidecar);
- explore the disintegration branch of the cell-set v0 § 6 disjunction
  as an alternative to binning + tolerance;
- consider whether a smaller observation than `Phi_K` (e.g. low-band
  energy `E_K` directly, treating it as a scalar fiber predicate)
  sidesteps the continuous-measure issue by reducing fiber dimension.

## 10. External Review Path

Promotion of this protocol requires a reviewer to confirm:

1. the tolerance object in section 1 is a faithful continuous-measure
   relaxation of the Reading-2 fiber criterion;
2. the binning instance in section 2 is a faithful numerical
   approximation of the tolerance object on `R^{d_K}`;
3. the proxy selector `\hat{pi}` is either an acceptable proxy for
   the intended objective or substitutable per section 3;
4. the parameter pinning discipline in section 7 closes the
   post-hoc-tuning attack surface.

The reviewer for this protocol can be the same PDE analyst named for
the C1 sidecar at promotion, or a separate methodologist (statistical
methodologist familiar with measure-theoretic discretisation, or a
data-assimilation researcher familiar with observer-design parameter
pre-registration). Reviewer named at promotion.

## Status

- **Drafted.** 2026-05-28.
- **Desk-auditable.** A reviewer can read this file end-to-end and
  audit the predicate structure without running code.
- **Unreviewed.** No external reviewer has signed off on the
  tolerance object, binning instance, proxy selector, or parameter
  pinning discipline.
- **Unrun.** No numerical execution of the section 5 procedure has
  been attempted as a verdict-bearing lock. A non-verdict smoke receipt
  exists at `results/proof/c1-kolmogorov-v0-smoke/manifest.json`.
- **v0 patch landed.** The cell-set v0 section 7 instantiates
  `epsilon_K`, `h_K`, `n_min`, `delta_action`, `S_pos`, `N_sample`,
  burn-in length, integration step, action tie-break order, and section 8
  stages the registered lock / fallback commands.

## Cross-references

- [`PDE_C1_CELLSET_KOLMOGOROV.md`](PDE_C1_CELLSET_KOLMOGOROV.md) —
  the cell-set v0 this protocol services. Sections 3 and 6 of the
  cell-set v0 are the gates this protocol addresses; section 4.1's
  attractor-support caveat is bridged here (section 6).
- [`PDE_DETERMINING_MODES_POSTULATE1.md`](PDE_DETERMINING_MODES_POSTULATE1.md)
  — the C1 sidecar. The Reading-2 fiber criterion gets its
  continuous-measure realisation through this protocol. The strictness
  witness's "independent non-injectivity" requirement is upstream of
  this protocol and unchanged.
- [`PHASE2_MDP.md`](PHASE2_MDP.md) — the finite-MDP fiber-selector
  Phase-2 result. This protocol is the PDE/Galerkin-instance
  translation, with binning replacing the finite state-action
  partition.
- [`PDE_C2_SHELL_SIGNATURE_SCOPING.md`](PDE_C2_SHELL_SIGNATURE_SCOPING.md)
  — C2 scoping. The `PDE-C1-NEG-B` / `PDE-C2-NEG-B` parallelism
  introduced in section 7 of this protocol aligns the two candidates'
  post-hoc-tuning receipt labels.
- [`../SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) — the
  ledger. Criterion (d) closure depends on this protocol plus the
  cell-set v0 patch.
- [`../SCIENTIFIC_CRITERIA.md`](../SCIENTIFIC_CRITERIA.md) — the
  parameter-pinning discipline in section 7 inherits the
  pre-registered audit-chain rule from this document.
