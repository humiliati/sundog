# Yang-Mills Phase 2 v3 Target-Redesign Probe Spec

Filed: **2026-05-29 (PT)**
Author triggers:
- Phase 2 v0 named null
  [`YM-P2-NEG-A no_rank_local_structure`](../receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)
- Phase 2 v1 (APE-smearing) named null
  [`YM-P2-NEG-A no_rank_local_structure`](../receipts/2026-05-29_SU2_3D_phase2_v1_no_rank_local_structure.md)
- Phase 2 v2 (connected-correlator) named null
  [`YM-P2-NEG-A no_rank_local_structure`](../receipts/2026-05-29_SU2_3D_phase2_v2_no_rank_local_structure.md)
- v2 probe spec
  [`2026-05-29_phase2_v2_correlator_probe.md`](2026-05-29_phase2_v2_correlator_probe.md)
  pre-stated this v3 fallback row

P0 lock: [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
P0 amendment 1 (APE smearing):
[`../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md)

Status: **probe spec**, not a binding pre-registration. This document
records the read of the three-NEG-A ladder, selects "different target
derived from same loops" from the v2 probe spec's pre-stated v3
fallback table, locks the new held-out target as the per-configuration
spatial variance of W33, and pre-states v4 fallback options.

The binding artifact triggered by this probe is
[`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md).

**No P0 amendment is required for v3.** The held-out vocabulary is
extended by a new version `v2` (per-configuration spatial variance of
the largest held-out loop `W33`), but the held-out loop set itself
remains `{W14, W23, W33}` and the underlying loop set is unchanged.
Per P0 §"Held-Out Observable Label", the held-out target may be
re-summarized as long as it stays gauge-invariant and disjoint from
the signature — both conditions are satisfied by `σ²_W33`.

## Three-NEG-A Ladder Read

| Probe | Signature vocab | Held-out summary | Primary bin-purity@5 | RAND margin | Verdict |
| --- | --- | --- | --- | --- | --- |
| v0 | v1 (bare 8-dim mean+var) | v1 γ_held (LS slope) | 0.31042 | +0.01042 | NEG-A |
| v1 | v4 (smeared 8-dim mean+var) | v1 γ_held (LS slope) | 0.29375 | -0.00208 | NEG-A |
| v2 | v5 (bare 20-dim correlator) | v1 γ_held (LS slope) | 0.30833 | +0.02083 | NEG-A |

All three primaries sit within ±0.04 of chance baseline `1/3`. The
small-loop signature space — across marginal moments (bare and
smeared) and 2-point spatial structure (bare) — does not resolve
γ_held tertile bins in within-β nearest-neighbor rank space at this
cell, lattice, and ensemble size. The small-loop **signature**
hypothesis is exhausted in its natural extensions inside P0.

What remains untested at this point is the **target** side: every
probe so far has scored against the same γ_held LS-slope tertile
label. The v2 probe spec's pre-stated v3 fallback table specifically
listed "variance of largest held-out loop" as a within-P0 target-side
redesign candidate; this probe selects exactly that option.

## Target Selection — Why σ²_W33

The held-out loop set `{W14, W23, W33}` is unchanged. What changes is
the per-configuration summary statistic used to assign a label.

| Held-out summary | Definition | Information class |
| --- | --- | --- |
| v1 `γ_held` (v0 / v1 / v2 baseline) | LS slope of `ln Re(W)` vs area on three held-out points `(W14, W23, W33)` | per-config area-law decay rate (large-loop area-law mean structure) |
| **v2 `σ²_W33` (this probe)** | sample variance of `(1/2) Re Tr U_loop_3x3` across all `12³ × 3 = 5184` (position × orientation) samples in one configuration | per-config spatial inhomogeneity of the largest held-out loop (non-area-law spatial structure) |

Why this target is informative as a v3 redesign:

1. **Different statistical moment.** γ_held depends on per-config
   *means* of held-out loops. σ²_W33 depends on per-config *variance*
   of the largest held-out loop's position-and-orientation samples.
   These are mathematically independent moments of the same
   underlying loop distribution.

2. **Different physical class.** γ_held is an area-law-decay proxy.
   σ²_W33 measures spatial inhomogeneity within a single
   configuration. Two configurations with identical mean `W33` (so
   identical γ_held contribution from the W33 anchor) can have very
   different σ²_W33.

3. **Same held-out loops.** The held-out loop set `{W14, W23, W33}`
   is unchanged; only the summary statistic differs. P0 disjointness
   with the signature (which uses `{W11, W12, W13, W22}`) is
   preserved.

4. **Gauge invariant.** `Re Tr U_loop_3x3` is gauge-invariant per
   loop, so the variance of these samples per configuration is
   gauge-invariant by construction. `CTRL_GAUGE_RAND` invariance
   gate applies the same way as for γ_held.

5. **No P0 amendment.** The held-out vocabulary version bumps from
   v1 to v2; the held-out loop class is unchanged from P0. Polyakov
   and topological observables remain deferred (no v3 amendment).

## Why Not σ²_W14 or σ²_W23

Only the largest held-out loop is used for the variance target.
Smaller loops (W11, W12, W13, W22) are in the signature; W14 and W23
are in the held-out set but at smaller area than W33. Using only
σ²_W33 keeps the scientific contrast sharp:

- the signature uses `{W11, W12, W13, W22}` *means* and *variances*
  (vocab v1) — that is, signature spatial structure at small loops;
- the new held-out target uses *only* `W33` spatial variance — that
  is, target spatial structure at the largest available loop.

The probe is: does small-loop signature spatial structure preserve
*large-loop spatial inhomogeneity* tertile bins in within-β rank
space? This is a sharper, more scientifically isolable question than
a multi-loop variance summary.

If σ²_W33 also fails (v3 NEG-A), σ²_W14 or σ²_W23 as alternative
v4 targets remain pre-stated below — but they are NOT a "shop the
target distribution" search, they are pre-registered next-step
candidates if and only if v3 lands NEG-A.

## v3 Design — Locked

Held-out target vocabulary v2 (this probe-spec freezes the convention;
the v3 binding spec consumes it):

- **Per-config target:** `σ²_W33(config)` = sample variance of
  `(1/2) Re Tr U_loop_3x3(x, orientation)` across all
  `12³ × 3 = 5184` (position × orientation) samples for that config.
- **Sample definition:**
  ```text
  S(x, o) = (1/2) Re Tr (product of 12 links around the 3×3
                         Wilson loop anchored at base position x
                         in plane orientation o)
  μ(config)  = (1 / 5184) Σ_{x, o} S(x, o)
  σ²_W33(config) = (1 / 5184) Σ_{x, o} (S(x, o) - μ(config))²
  ```
- **Per-β tertile bin edges:** linear-interpolation 33.3% and 66.7%
  percentiles of the 32-config σ²_W33 distribution within each β.
  Computed in the v3 aggregation runner; frozen before any NN
  scoring; recorded in
  `aggregation/per_beta_v3_bin_edges.json` with a write timestamp
  earlier than first scoring artifact.
- **Across-β tertile edges:** computed on the combined 96-config
  σ²_W33 distribution; same convention; recorded as
  `aggregation/global_v3_bin_edges.json`.

Signature vocabulary (this probe-spec uses the unchanged v0
baseline):

- **Signature vocab:** v1 (bare 8-dim mean+var, the same v0 baseline
  signature). Reused directly from v0's
  `signatures/signature_vectors.csv` per β. **Not recomputed in v3**
  — v0's emitted signature is the input.

Ensembles: v0 ensembles bit-for-bit, same as v1 and v2.

NN methodology: identical to v0 (Euclidean L2 in z-score-normalized
8-dim signature space; within-β primary + across-β cross-check; k
slate `{3, 5, 10}` with k=5 primary; bootstrap 95% CIs).

Controls battery: identical to v0 (six scored, `CTRL_FINITE_SIZE`
deferred). All control NN graphs are computed against the new v2
target labels; the underlying neighbor structure for primary,
`CTRL_RAW`, `CTRL_RAND`, etc. is computed the same way as v0; only
the label being scored against changes.

Promotion thresholds: identical to v0.

## v4 Fallback (Pre-Stated)

If v3 also lands `YM-P2-NEG-A`, the bounded-null framing strengthens
significantly: the small-loop signature space (vocab v1) does not
resolve either area-law-mean or spatial-variance tertile labels of
the largest held-out loop on this cell. The v4 design space is
intentionally narrow:

| v4 candidate | P0 impact | Rationale | Notes |
| --- | --- | --- | --- |
| **σ²_W14 or σ²_W23** held-out target | none | tests whether the spatial-variance issue is specifically about W33 or extends to all held-out loops | minimal delta; risk of "shopping the variance target distribution" |
| **Polyakov-loop expectation per config** | P0 amendment 2 (Polyakov is deferred at P0) | tests against a fundamentally different gauge-invariant observable class | biggest design delta still within finite-lattice envelope |
| **Smeared signature vocab v4 against σ²_W33** | none (smearing already admitted by P0 amendment 1) | tests whether smearing helps the σ²_W33 target even though it didn't help γ_held | compound change vs the v0 baseline; harder to attribute outcome |
| **PAUSE: file bounded-null synthesis receipt** | none | treats the four NEG-As as a substantive bounded null on this envelope; honors the lit-pass scaffold's "named nulls are first-class" principle | the legitimate end state if v3 NEG-As; admitted now to prevent later p-hacking |

Pre-stating these options is the discipline that prevents the v4
selection from being retrofitted to the v3 result. If v3 lands
P2-A, none of these v4 options is invoked. If v3 lands NEG-A, one
of these options is selected on documented reasoning, not
data-shopping.

## Anti-Scope-Creep

The σ²_W33 target definition is frozen at this probe spec: variance
over **all** `12³ × 3 = 5184` (position × orientation) samples,
biased variance estimator (divisor `5184`, not `5183`), no
per-orientation breakdown, no Bessel correction. The per-β bin
convention remains linear-interpolation tertile. The signature is
v1 vocab unchanged. The displacement-slate or smearing parameters
(used by v1/v2 binding specs) are not used by v3.

If v3 lands NEG-A and a future probe spec wants a different variance
estimator, a per-orientation variance, or a different held-out loop's
variance, that is a v4 probe spec with a new vocabulary version,
never an in-place revision of this probe or the v3 binding spec.

## Filing List

This probe spec triggers one binding document and one runner
implementation pair:

1. [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v3.md)
   — binding Phase 2 v3 spec with held-out vocab v2 (σ²_W33).
2. New aggregation runner
   `scripts/yang-mills-phase2-v3-su2-3d-aggregate.mjs` paired with the
   npm script `yang-mills:phase2:v3:su2-3d:aggregate`. The new runner
   reuses `scripts/lib/yang-mills-su2-3d-core.mjs` for per-position
   W33 evaluation. The v0, v1, and v2 aggregation runners + bare
   SU(2) 3D core + smearing module + correlator module all remain
   bit-for-bit unchanged.

No new per-β ensembles are required. The v0 ensemble dirs at
`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`
are the v3 input, exactly as for v1 and v2.

## What A v3 Outcome Tells Us

| v3 verdict | Diagnostic interpretation | Next allowed step |
| --- | --- | --- |
| `P2-A bounded_positive` | Small-loop signature space resolves a non-area-law observable (σ²_W33) within β; the v0/v1/v2 NEG-As were specifically about *which* target was scored, not signature vacuity per se | draft Phase 3 observable-certificate manifest per P0 §8 Phase 3; the certificate envelope is "vocab v1 signature ⇒ σ²_W33 tertile, registered envelope" |
| `YM-P2-NEG-A no_rank_local_structure` (again) | Four consistent NEG-As; signature space genuinely vacuous on every tested held-out summary; bounded null is the right frame | select one of v4 pre-stated options; the PAUSE option becomes the recommended call unless a Polyakov or σ²_W14/σ²_W23 probe has a clear scientific justification |
| `YM-P2-NEG-B metadata_only` | CTRL_META meets or beats primary on σ²_W33 too | bin convention itself may be the issue; reconsider per-β tertile vs binary or quintile |
| `YM-P2-NEG-D raw_dominates` | CTRL_RAW matches or beats primary on σ²_W33 | raw matrix entries carry more about σ²_W33 than the v1 invariant signature; signature class fundamentally too coarse |

Any of these is a publishable receipt under P0's named-null
convention.
