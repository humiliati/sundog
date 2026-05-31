# Yang-Mills Phase 2 v4 Powered-Target Probe Spec

Filed: **2026-05-31 (PT)**

Author triggers:
- The four-NEG-A bounded null
  [`2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md`](../receipts/2026-05-29_SU2_3D_phase2_bounded_null_synthesis.md)
  (v0/v1/v2 × γ_held, v3 × σ²_W33; all within ±0.04 of chance 1/3).
- The **2026-05-30 target-validity diagnostic**
  [`2026-05-30_cheap_a_controls_target_validity.md`](../receipts/2026-05-30_cheap_a_controls_target_validity.md):
  the held-out targets were noise-floored, so the four NEG-As were structurally
  guaranteed and **uninformative about the signature**.

P0 lock: [`../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`](../../prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md)
P0 amendment 1 (APE smearing): [`../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md`](../../prereg/yang-mills/P0_AMENDMENT_2026-05-29_ape_smearing.md)

Status: **probe spec**, not a binding pre-registration. It records the read of
the diagnostic, selects the "powered-target audit" move, locks the candidate
pool and the power / leakage gates, introduces the `YM-P2-UNDERPOWERED` branch,
and pre-states the v5 fallback. The binding artifact it triggers is
[`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md).

**No P0 amendment is required for v4.** The held-out loop set `{W14, W23, W33}`
is unchanged; only the per-config *summary* changes (a new held-out vocabulary
version `v3`). Per P0 §"Held-Out Observable Label" and the v3 precedent, a
held-out target may be re-summarized as long as it stays gauge-invariant and
disjoint from the signature; both are gated explicitly below.

## Motivation — why a powered-target probe, and why it is not p-hunting

The bounded-null synthesis pre-stated that any v4 move requires **fresh external
scientific motivation**, not a search for a positive. The 2026-05-30 diagnostic
*is* that motivation, and it is documented and dated:

- `γ_held` is **~99% determined by an ε-floor clamp** (R² = 0.99 at all three β)
  that fires when the 3×3 loop `W33` dips ≤ 0; `W33` here has **per-config SNR
  < 1** (0.38 / 0.79 / 1.99 at β = 2.0 / 2.4 / 2.8). At β = 2.0 the **top γ_held
  tertile is 100% the clamp set** — so the tertile labels v0/v1/v2 scored against
  were a noise coin-flip. v3's `σ²_W33` is a summary of the same noise loop.
- Therefore the four NEG-As were **structurally guaranteed**: no signature can
  predict a noise-driven label. The nulls do not bear on the signature at all.

The fix is not "try more targets until one scores." It is to **require the target
to carry signal before it is eligible to be scored** — a property of the target
measured *without the signature*, fixed before any rank-locality computation. The
v4 probe does exactly that. Its scientific value is that **both** Stage-2 outcomes
become informative for the first time on this lane.

## v4 design — locked

### Stage 1 — Target Power Audit (pre-scoring; no rank-locality score)

A new pre-scoring admission step, in the same spirit as bin-edge freezing. It runs
on the v0 ensembles and uses **no** nearest-neighbor / rank-locality score.

**Split-half reliability.** For each config, partition its `12³ × 3 = 5184`
(position × orientation) loop samples into two halves by **site-coordinate
parity** (sample assigned to half A if `(x+y+z)` of its base position is even,
else half B; orientation does not affect the split). This is spatially balanced
and fixed before any data is read. For each candidate summary `T`:

- compute `T1(c)` on half A and `T2(c)` on half B for every config `c`;
- **ICC** `:= Pearson corr( T1, T2 )` across the 32 configs (split-half
  reliability; the fraction of the per-config summary that reproduces on an
  independent half);
- **agreement** `:= mean_c 1[ tertile(T1)(c) == tertile(T2)(c) ]`, each tertile
  taken within its own half's 32-config distribution.

**Power gate (frozen):** a candidate is **POWERED** iff

```text
ICC >= 0.50   AND   agreement >= 0.50
```

(ICC ≥ 0.5 = majority of variance is reproducible config-level signal; agreement
≥ 0.50 = ≥ 16/32, one-sided binomial p < 0.05 above the 1/3 tertile-chance floor
at n = 32.)

**Leakage gate (frozen; reuses the 2026-05-30 Q1 method).** A candidate is
**DISJOINT** iff

```text
CV-R²( T | 8-dim v1 signature ) <= 0.25
```

(5-fold CV OLS R², same estimator as `scripts/yang-mills-q1q5-controls.py`). Above
0.25 the target's tertile is majority-predictable from the signature and a
positive Stage-2 read would be trivial leakage, so it is rejected.

### Candidate pool (amendment-free re-summaries of `{W14, W23, W33}`)

| Candidate | Definition | Rationale |
| --- | --- | --- |
| `mean_W14` | per-config mean of `(1/2)Re Tr U` over the 1×4 loop | largest held-out loop value (~0.046 at β=2.0); no log / no clamp — the most likely POWERED candidate |
| `mean_W23` | per-config mean over the 2×3 loop | intermediate area; no log |
| `σ²_W14` | per-config spatial variance of the 1×4 loop | variance class, better-measured than σ²_W33 |
| `σ²_W23` | per-config spatial variance of the 2×3 loop | variance class |
| `ratio_W23_W14` | `mean_W23 / mean_W14` per config | a clamp-free area-decay proxy (no log; the area-law content γ_held tried to capture, minus the noise-floored W33 anchor) |

**Prior targets re-audited (reported, not admissible as new):**

- `γ_held` (v1 target) — **expected to FAIL** the power gate, reproducing the
  diagnostic. This is the gate's built-in self-validation: if `γ_held` were to
  pass, the gate is mis-specified and the probe voids (see Verification).
- `σ²_W33` (v2 target / v3 probe) — power **genuinely untested** by the
  diagnostic. The audit re-adjudicates whether v3's NEG-A was already on a
  powered target (if `σ²_W33` is POWERED, v3's null was informative after all).

### Admission and primary selection (signature-blind)

Admit candidates that are **POWERED and DISJOINT in all three β** (so the primary
is uniformly powered across the slate, matching the within-β primary lane). The
**primary** target is the admitted candidate with the **highest mean-over-β ICC**
(tie-break: lowest mean leakage CV-R²).
This selection uses only the power and leakage metrics — both signature-blind to
the rank-locality score — and is written to `aggregation/admitted_target.json`
with a write timestamp strictly earlier than the first Stage-2 scoring artifact.
Other admitted candidates are reported as secondary context, never gated.

### Stage 2 — Relative-locality test (only on the admitted primary)

Identical to v0 in every respect except the label scored:

- signature: v1 bare 8-dim mean+var, re-read from v0's
  `signatures/signature_vectors.csv` (hash-asserted against v0; not recomputed);
- distance: Euclidean L2 in z-score-normalized 8-dim signature space; within-β
  primary lane + across-β cross-check;
- k slate `{3, 5, 10}`, k = 5 primary; bootstrap 95% CIs (B = 1000);
- six scored controls `CTRL_META / CTRL_RAW / CTRL_RAND / CTRL_RAND_STRAT /
  CTRL_PERM / CTRL_GAUGE_RAND`; `CTRL_FINITE_SIZE` deferred to Phase 4;
- **promotion gates unchanged from v0** (verbatim): primary bin-purity@5 ≥ 0.5;
  margin ≥ 0.10 over `CTRL_RAND` / `CTRL_META` / `CTRL_RAW`; across-β margin
  ≥ 0.05 over `CTRL_RAND_STRAT`; `CTRL_PERM` within 0.05 of chance 1/3;
  `CTRL_GAUGE_RAND` matches primary to ≤ 1e-12.

Scored against the admitted primary's per-β tertile labels (bin edges frozen on
the **full-sample** primary summary, before scoring, in
`aggregation/per_beta_v4_bin_edges.json` with the freeze-before-scoring timestamp
discipline).

## New outcome branch (introduced by this probe spec)

| Branch | Trigger | Disposition |
| --- | --- | --- |
| `YM-P2-UNDERPOWERED no_powered_target_in_envelope` | **no** candidate is both POWERED and DISJOINT | quarantine-class: the 12³ cell cannot supply a valid held-out target. **NOT** a `NEG-A` — it does not implicate the signature. Triggers the v5 fallback. |

If a powered+disjoint primary IS admitted, Stage 2 lands in the existing P0
branches (A / B / C / D / E / F / G / H) — but now a `YM-P2-NEG-A` is
**informative**: the signature is genuinely vacuous on a target that carried
signal.

### What a v4 outcome tells us

| v4 verdict | Interpretation | Next allowed step |
| --- | --- | --- |
| `P2-A bounded_positive` | signature encodes a **powered** held-out target; the prior NEG-As were about target power, not signature vacuity | draft Phase 3 observable-certificate manifest (P0 §8) |
| `YM-P2-NEG-A` (on a powered target) | signature **genuinely vacuous** on a target with real signal — the informative null the prior probes could not deliver | bounded-null synthesis **upgrade** from "uninformative" to "informative"; lane pause is now substantive |
| `YM-P2-UNDERPOWERED` | 12³ cell cannot supply a powered+disjoint target | v5 fallback (below) |
| `YM-P2-NEG-B` / `NEG-C` / quarantines | as in P0 branch table | as in P0 |

## v5 Fallback (pre-stated, not bound)

If v4 lands `YM-P2-UNDERPOWERED`, the documented next step is a **P0 amendment 2**
to a **powered regime**, selected on the audit evidence, not shopped:

| v5 candidate | P0 impact | Rationale |
| --- | --- | --- |
| **weaker-β slate** (e.g. {2.8, 3.2, 3.6}) | amendment 2 (β slate frozen at P0) | larger held-out loops retain area-law signal at weaker coupling — directly raises target SNR |
| **larger volume** (16³) | amendment 2 (lattice slate frozen) | more position samples per config → lower per-config measurement noise → higher ICC |
| **Polyakov-loop target** | amendment 2 (deferred at P0) | a non-self-averaging order-parameter-class observable with genuine config-to-config signal |
| **PAUSE — UNDERPOWERED is the receipt** | none | file the underpowered-envelope finding as the honest end state; the bounded null is then explicitly "no powered target available in this envelope," not "signature vacuous" |

Pre-stating these prevents the v5 selection from being retrofitted to the v4
result. If v4 lands `P2-A` or an informative `NEG-A`, no v5 option is invoked.

## Anti-Scope-Creep

Frozen at this probe spec: the split rule (site-coordinate parity), the candidate
pool (the five re-summaries above + the two re-audited priors), the power gate
(`ICC ≥ 0.50 AND agreement ≥ 0.50`), the leakage gate (`CV-R² ≤ 0.25`), the
primary-selection rule (highest ICC among powered+disjoint), and the unchanged v0
Stage-2 methodology and gates. The held-out loop set stays `{W14, W23, W33}`; the
signature stays v1; ensembles stay the v0 ensembles bit-for-bit. Any different
split rule, candidate, or gate threshold is a future probe spec with a new
vocabulary version — never an in-place revision of this probe or the v4 binding
spec.

## Filing List

1. [`../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md`](../../prereg/yang-mills/PHASE2_SU2_3D_relative_locality_v4.md)
   — binding Phase 2 v4 spec (held-out vocab v3 = the audited powered target).
2. New aggregation runner `scripts/yang-mills-phase2-v4-su2-3d-aggregate.mjs`
   + npm script `yang-mills:phase2:v4:su2-3d:aggregate`, **filed only after the
   binding spec is signed off** (P0 admission discipline). Reuses
   `scripts/lib/yang-mills-su2-3d-core.mjs` for per-position loop evaluation; v0–v3
   runners + cores stay bit-for-bit unchanged. No new ensembles
   (`results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/` is
   the input, as for v1/v2/v3).
