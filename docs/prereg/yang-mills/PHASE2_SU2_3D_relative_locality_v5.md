# Phase 2 v5 — SU(2) 3D Relative-Locality on a Symmetric Polyakov Target (binding)

Status: **binding pre-registration**, filed 2026-05-31. Triggered by
[`../../yang-mills/specs/2026-05-31_phase2_v5_polyakov_probe.md`](../../yang-mills/specs/2026-05-31_phase2_v5_polyakov_probe.md).
No runner code admitted until signed off and a runner manifest fills every
Admission Requirement.

P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md) ·
amendment 2 (Polyakov target class):
[`P0_AMENDMENT_2026-05-31_polyakov.md`](P0_AMENDMENT_2026-05-31_polyakov.md).

Held-out target vocabulary **v4 (Polyakov)**; signature vocabulary unchanged
**v1**; ensembles are the v0 12³ ensembles bit-for-bit (no new generation).

## 1. Cell / ensemble (inherited from v0, hash-asserted)

- `SU2_3D`, `12³`, periodic, Wilson action, β slate `{2.0, 2.4, 2.8}`, 32 configs/β.
- Source (read-only): `results/yang-mills/phase2/SU2_3D/2026-05-29_su2_3d_beta<β>_ensemble_v0/`.
- Runner asserts: each `configs/su2_links.jsonl` SHA-256 matches the v0
  `hashes.json`; the re-read v1 signature matches v0 to ≤ 1e-12. Mismatch → `Z void_run`.

## 2. Stage 1 — Polyakov target power audit (pre-scoring; frozen gates)

For each config, reconstruct the lattice from `su2_links.jsonl` and compute the
Polyakov loop `P(x_⊥; μ) = (½)Tr ∏_{t} U_μ(x_⊥ + t·μ̂)` (the `[0]` quaternion
component) for each of the 3 wrap directions μ and all transverse sites `x_⊥`.

**Candidate pool (held-out vocab v4), each averaged over the 3 wrap directions:**
`abs_mean_P`, `mean_abs_P`, `chi_P` (defined in amendment 2). **Prior re-audit
(reported, not admissible):** `γ_held` — **must FAIL** the power gate.

**Per-candidate metrics** (across the 32 configs per β):
- `ICC := corr(T1, T2)` from a transverse-**site-parity** split-half of the
  Polyakov values;
- `agreement := mean 1[tertile(T1)==tertile(T2)]`;
- `leakage := CV-R²(target | 8-dim v1 signature)`, the 5-fold-OLS estimator of
  `scripts/yang-mills-q1q5-controls.py`.

**Frozen gates (verbatim from v4):**

```text
POWERED   :=  ICC >= 0.50  AND  agreement >= 0.50   (in all three β)
DISJOINT  :=  leakage <= 0.25
ADMITTED  :=  POWERED AND DISJOINT
```

**Gauge-invariance health (amendment 2):** `CTRL_GAUGE_RAND` recomputes each
Polyakov summary after a random gauge transform (`applySU2GaugeTransform`); each
must match to ≤ 1e-12, else `YM-P1-NEG-A gauge_leakage`.

**Primary selection (signature-blind):** highest mean-over-β ICC among admitted
(tie-break lowest mean leakage), written to `aggregation/admitted_target.json`
with a write timestamp earlier than any Stage-2 artifact.

## 3. Stage 2 — relative-locality test (admitted primary only)

Identical to v0 except the scored label: 8-dim v1 signature (hash-asserted),
z-scored L2-NN, within-β + across-β lanes, k slate `{3,5,10}` (k=5 primary),
bootstrap 95% CI (B=1000), six scored controls (META/RAW/RAND/RAND_STRAT/PERM/
GAUGE_RAND; FINITE_SIZE deferred). Primary label = per-β tertile of the admitted
Polyakov summary, bin edges frozen in `aggregation/per_beta_v5_bin_edges.json`
before scoring.

**Promotion gates (verbatim from v0):** primary bin-purity@5 ≥ 0.5; margin ≥ 0.10
over RAND/META/RAW; across-β ≥ 0.05 over RAND_STRAT; PERM within 0.05 of 1/3;
GAUGE_RAND matches primary to ≤ 1e-12.

## 4. Outcome branches

| Branch | Trigger | Verdict |
| --- | --- | --- |
| UNDERPOWERED | no candidate ADMITTED | `YM-P2-UNDERPOWERED no_powered_target_in_envelope` → v6 finite-T (pre-stated) |
| A | admitted primary clears all gates; GAUGE_RAND invariant | `P2-A bounded_positive` (→ Phase 3) — first powered Stage-2 read |
| C | primary fails RAND/RAND_STRAT margin | `YM-P2-NEG-A` — informative (signature vacuous on a powered, disjoint target) |
| B / G / D / Z | per P0 branch table | per P0 |

## 5. Admission Requirements (runner manifest)

Inherited-from-v0 (hash-asserted): generator + mix, per-β seeds, burn-in,
τ_int/thinning, signature vocab v1. v5-specific: held-out vocab **v4 (Polyakov)**;
candidate pool (the 3 summaries); wrap = all 3 directions; split = transverse-site
parity; power gate `ICC≥0.50 ∧ agreement≥0.50`; leakage gate `≤0.25`; primary =
highest mean-β ICC; `admitted_target.json` + bin-edge freeze timestamps earlier
than first scoring artifact; `CTRL_GAUGE_RAND` Polyakov invariance ≤ 1e-12;
control set = six scored + FINITE_SIZE deferred; output dir; commit; **exact
command**; ≤ 10-min cap, one invocation.

**Exact intended command:**

```text
node scripts/yang-mills-phase2-v5-su2-3d-aggregate.mjs \
  --ensemble-root results/yang-mills/phase2/SU2_3D \
  --out results/yang-mills/phase2/SU2_3D/2026-05-31_su2_3d_relative_locality_v5 \
  --power-icc-gate 0.50 --power-agreement-gate 0.50 --leakage-cvr2-gate 0.25 \
  --k-primary 5 --bootstrap 1000 --seed 202605310205
```

## 6. Code organization

New `scripts/yang-mills-phase2-v5-su2-3d-aggregate.mjs` (+ npm
`yang-mills:phase2:v5:su2-3d:aggregate`); a new `polyakovLoop` function added to
`scripts/lib/yang-mills-su2-3d-core.mjs` (reusing `getLink`/`qMul`/`qConj`), with
all existing core functions and the v0–v4 runners bit-for-bit unchanged. No new
ensembles.

## 7. Anti-scope-creep

Pool, gates, wrap-averaging, primary rule, and v0-identical Stage-2 frozen here.
A `YM-P2-UNDERPOWERED` routes to the pre-stated v6 finite-T spec, never to a
silent change of cell/target/gate.
