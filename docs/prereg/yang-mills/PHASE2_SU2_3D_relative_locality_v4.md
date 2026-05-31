# Phase 2 v4 — SU(2) 3D Relative-Locality on a Powered Held-Out Target (binding)

Status: **binding pre-registration**, filed 2026-05-31. Triggered by
[`../../yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md`](../../yang-mills/specs/2026-05-31_phase2_v4_powered_target_probe.md).
No runner code is admitted until this document is signed off and a runner
manifest fills every Admission Requirement below.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Diagnostic motivation: [`../../yang-mills/receipts/2026-05-30_cheap_a_controls_target_validity.md`](../../yang-mills/receipts/2026-05-30_cheap_a_controls_target_validity.md)

**No P0 amendment.** Held-out loop set `{W14, W23, W33}` unchanged; held-out
vocabulary version bumps `v2 → v3` (the audited powered summary). Signature vocab
stays `v1`. Ensembles are the v0 ensembles bit-for-bit.

## 1. Cell / ensemble (inherited from v0, hash-asserted)

- Cell `SU2_3D`, lattice `12×12×12`, periodic, Wilson action.
- β slate `{2.0, 2.4, 2.8}`, 32 configs/β = 96 total.
- Ensemble source dirs (read-only inputs, **not** regenerated):
  `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`.
- The runner MUST assert: each ensemble's `configs/su2_links.jsonl` SHA-256
  matches the v0 `hashes.json`; the re-read `signatures/signature_vectors.csv`
  agrees with v0 to ≤ 1e-12 per component. Mismatch → `Z void_run`.

## 2. Stage 1 — Target Power Audit (pre-scoring; frozen gates)

Runs before any nearest-neighbor computation. Uses **no** rank-locality score.

**Split.** Each config's `12³ × 3 = 5184` (position × orientation) loop samples
are split by base-site parity: half A = `(x+y+z) mod 2 == 0`, half B otherwise.
Fixed; not tunable.

**Per-candidate metrics** (across the 32 configs per β):
- `ICC := corr(T1, T2)` — Pearson split-half reliability of the per-config
  summary (`T1` on half A, `T2` on half B).
- `agreement := mean 1[tertile(T1) == tertile(T2)]` (tertiles within each half).
- `leakage := CV-R²(T | 8-dim v1 signature)` — 5-fold OLS CV R², the exact
  estimator in `scripts/yang-mills-q1q5-controls.py`.

**Frozen gates:**

```text
POWERED   :=  ICC >= 0.50  AND  agreement >= 0.50
DISJOINT  :=  leakage <= 0.25
ADMITTED  :=  POWERED AND DISJOINT
```

**Candidate pool (held-out vocab v3):** `mean_W14`, `mean_W23`, `σ²_W14`,
`σ²_W23`, `ratio_W23_W14 = mean_W23/mean_W14`.
**Re-audited prior targets (reported, not admissible):** `γ_held` (must FAIL the
power gate — self-validation), `σ²_W33`.

`σ²` uses the biased estimator (divisor 5184). Variance / ratio summaries that are
undefined for a config (e.g. `mean_W14 == 0` exactly) → that config flagged; if
> 5% of configs are flagged for a candidate, that candidate is `UNDEFINED` and
not admissible.

**Per-β handling.** Gates evaluated **per β**; a candidate is ADMITTED only if it
clears the gates in **all three** β (so the chosen primary is uniformly powered
across the slate, matching the within-β primary lane).

**Primary selection (signature-blind):** primary = ADMITTED candidate with the
highest **mean-over-β ICC** (tie-break: lowest mean leakage). Written to
`aggregation/admitted_target.json` with a write timestamp **strictly earlier**
than any Stage-2 scoring artifact (runner asserts the ordering). Secondary
admitted candidates recorded for context, never gated.

## 3. Stage 2 — Relative-locality test (admitted primary only)

Identical to v0 except the scored label.

- Signature: v1 bare 8-dim mean+var, re-read from v0 (hash-asserted, §1).
- Distance: Euclidean L2 in z-score-normalized 8-dim signature space; within-β
  primary + across-β cross-check (per-β vs combined-96 normalization).
- k slate `{3, 5, 10}`; **k = 5 primary**; gates at k = 5 only. Bootstrap 95% CI,
  B = 1000 within-β resamples (point estimates gate; CI for context).
- Primary label: per-β tertile of the **full-sample** primary summary; bin edges
  frozen in `aggregation/per_beta_v4_bin_edges.json` before scoring (timestamp
  asserted). Across-β global tertiles in `aggregation/global_v4_bin_edges.json`.
- Controls scored (6 of 7): `CTRL_META`, `CTRL_RAW`, `CTRL_RAND`,
  `CTRL_RAND_STRAT`, `CTRL_PERM`, `CTRL_GAUGE_RAND`. `CTRL_FINITE_SIZE` deferred
  to Phase 4 (`not_scored: phase4_reserved`).

**Promotion gates (verbatim from v0; point estimates at k = 5, within-β primary):**

```text
primary bin-purity@5 >= 0.5                       (discrimination ratio >= 1.5)
primary - CTRL_RAND  >= 0.10                       (also vs CTRL_META, CTRL_RAW)
across-β primary - CTRL_RAND_STRAT >= 0.05
CTRL_PERM within 0.05 of chance 1/3
CTRL_GAUGE_RAND matches primary to <= 1e-12
```

## 4. Outcome branches

| Branch | Trigger | Verdict |
| --- | --- | --- |
| **UNDERPOWERED** | no candidate ADMITTED (Stage 1) | `YM-P2-UNDERPOWERED no_powered_target_in_envelope` (quarantine-class; NOT NEG-A; → v5 fallback) |
| A | primary clears all promotion gates; GAUGE_RAND invariant | `P2-A bounded_positive` (→ Phase 3) |
| C | primary fails RAND / RAND_STRAT margin | `YM-P2-NEG-A no_rank_local_structure` — **now informative** (powered target) |
| B | CTRL_META ≥ primary | `YM-P2-NEG-B metadata_only` |
| G | tracks β bin only, not within-β tertile | `YM-P2-NEG-C coupling_triviality` |
| D / E / Z | GAUGE_RAND breach / target leakage / PERM ≠ chance / hash mismatch | quarantine per P0 |

## 5. Admission Requirements (runner manifest must state)

Inherited-from-v0 fields asserted by hash: generator + 1HB/4OR mix, per-β seeds
`202605290201/0202/0203`, burn-in ≥ 2000, pilot τ_int source + value, thinning
interval, signature vocab `v1`. v4-specific fields the manifest must add:

- held-out target vocab version `v3`; candidate pool (the five above); split rule
  (site parity);
- power gate `ICC ≥ 0.50 AND agreement ≥ 0.50`; leakage gate `CV-R² ≤ 0.25`;
  primary-selection rule (highest mean-β ICC among admitted);
- `admitted_target.json` and `per_beta_v4_bin_edges.json` freeze timestamps
  earlier than first scoring artifact;
- control set = the six scored + `CTRL_FINITE_SIZE` deferred;
- output dir, code commit hash, **exact command line** (gates = the exact
  unchanged command), compute-cap declaration (≤ 10 min wall, one invocation).

**Exact intended command (one invocation):**

```text
node scripts/yang-mills-phase2-v4-su2-3d-aggregate.mjs \
  --ensemble-root results/yang-mills/phase2/SU2_3D \
  --out results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v4 \
  --power-icc-gate 0.50 --power-agreement-gate 0.50 --leakage-cvr2-gate 0.25 \
  --k-primary 5 --bootstrap 1000 --seed 202605310204
```

## 6. Code organization

- New: `scripts/yang-mills-phase2-v4-su2-3d-aggregate.mjs` (Stage 1 audit +
  Stage 2 scoring); npm `yang-mills:phase2:v4:su2-3d:aggregate`.
- Reuses `scripts/lib/yang-mills-su2-3d-core.mjs` for per-position loop
  evaluation. v0/v1/v2/v3 aggregation runners + bare core + smearing + correlator
  modules remain bit-for-bit unchanged. No new ensembles.

## 7. Anti-scope-creep

The split rule, candidate pool, gate thresholds, primary-selection rule, and the
v0-identical Stage-2 methodology are frozen here. A `YM-P2-UNDERPOWERED` verdict
routes to the pre-stated v5 fallback (probe spec §"v5 Fallback"), never to a
silent in-place change of cell, β, lattice, signature, candidate, or gate.
