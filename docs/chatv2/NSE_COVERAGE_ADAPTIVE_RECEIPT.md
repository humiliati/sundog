# Coverage-Adaptive Apparatus — Receipt (Approach A)

> 2026-07-07. Build + §3 regression of `NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md`.
> Non-promotional; apparatus-generality only. H3 stays closed at
> `NSE-H3-INCONCLUSIVE_COVERAGE` unless R1 (below) reads G=675 under this
> now-validated apparatus.

## Build + synthetic validation

Additive `--adjudicator twin-state-adaptive` (frozen read via the untouched
`aggregate_twin_state` + the density-stratified adaptive read in one pass; raw
`samples.npz` export). Harness self-test passed; smoke passed.
`scripts/nse_coverage_adaptive_regression.py --self-test` 3/3: T1 compact ⇒
adaptive **bit-identical** to frozen (reduction theorem), T2 core+halo ⇒ frozen
defers on coverage while adaptive reads the core, T3 tiny core ⇒ sliver.

## R0 — regression gate: PASS (agent-run; ~40 min/cell, N=50k)

| cell | frozen | adaptive | f | \|adaptive − frozen\| disagree | banked xref |
| --- | --- | --- | --- | --- | --- |
| G=200 (`lock_v7_g200`) | `TWIN_STATE_CERTIFIED` | `TWIN_STATE_ADAPTIVE_CERTIFIED` | **1.0000** | **0.00000** | 0.03678 vs 0.0367 (\|d\| 0.0001) |
| G=300 (`lock_v7_g300`) | `TWIN_STATE_CERTIFIED` | `TWIN_STATE_ADAPTIVE_CERTIFIED` | **1.0000** | **0.00000** | 0.03819 vs 0.0382 (\|d\| 0.0000) |

Every dense-count percentile is 49/49 (full admission), and the adaptive pair
sets are **bit-identical** to frozen (693,774 / 942,834 unique witness pairs — the
G=300 count reproduces the banked RG-v1 twin exactly). Both paired reads
`PAIRED_FIBER_CONSTANCY_POSITIVE`. The reduction theorem is confirmed numerically
on the real cells: on a compact attractor the apparatus *is* the frozen test.

**`REGRESSION: PASS → G=675 read unblocked.`** The apparatus is the same test
where the test already worked; it has earned the G=675 read. Not
`NSE-H3-APPARATUS-REJECTED`.

## R1 — application (owner-run; the reopened G=675 read)

Re-run the matched-Re fallback cell (`fallback_v7_g675_kf3`, N=200k) with the
validated adjudicator:

```
python scripts/pde_c1_kolmogorov_cell.py --preset fallback_v7_g675_kf3 --adjudicator twin-state-adaptive --out results/proof/c1-h3-kf3-g675-adaptive
```

~2–2.2 h; quiet box (200k sample arrays; the 2026-07-06 memory-squeeze precedent
applies). Branches (spec §4), read off `adaptive_covered_fraction` (`f`) and the
adaptive/paired verdicts:

| R1 outcome | verdict |
| --- | --- |
| `f < 0.10` | `NSE-H3-COVERAGE-SLIVER` |
| `f ≥ 0.10`, `ADAPTIVE_CERTIFIED` + paired POSITIVE | `NSE-H3-FORCING-GENERAL` (scoped to `f`) |
| `f ≥ 0.10`, `ADAPTIVE_CERTIFIED` + paired NEG | `NSE-H3-GRASHOF-LOCAL` |
| `f ≥ 0.10`, `ADAPTIVE_NO_CERTIFICATE` | `NSE-H3-COVERAGE-WALL-CONFIRMED` |

Evaluate on completion; no gate widening or constant retune after the read.

## R1 result (owner-run 2026-07-07, N=200k, ~88 min): `TWIN_STATE_ADAPTIVE_SLIVER`

| quantity | value | reading |
| --- | --- | --- |
| covered fraction `f` | **0.0364** (floor 0.10) | only 3.6% of samples sit in an ε_K-dense fiber |
| dense-count p05 / p50 / p95 | **0 / 0 / 8** | the **median sample has zero** of its 49 NN within ε_K; even p95 (8) is below `k_min=10` |
| frozen candidate fraction | 0.4692 | ≥1 near pair — chance adjacency, not fibers |
| admitted samples | 7,254 / 200,000 | |
| sliver witness disagree | 0.0295 | the tiny resolved sliver is anchor-like (cf. 0.033), but below floor ⇒ no claim |

**Verdict: `NSE-H3-COVERAGE-SLIVER` — and it upgrades the wall from "unresolved"
to "real".** The frozen deferral's 0.47 "coverage" was chance-adjacency: nearly
half the samples have *some* neighbor within ε_K, but the median sample has
**zero**, and only 3.6% have a genuine dense fiber (≥10 near). An apparatus that
*cannot manufacture coverage* (fidelity preserved — no pair beyond ε_K ever
compared) looked and found the ε_K-resolvable fiber structure is essentially
absent. The matched-Re G=675 attractor is genuinely higher-effective-dimensional
at the signature scale; the witness cannot be tested there because its geometric
precondition (dense fibers) does not exist, not because of sampling.

## H3 forcing axis — final disposition (four concordant probes)

1. v1 (fixed G): laminar (`INPUT-UNPOWERED`).
2. v1.1 matched-Re 50k: frozen defers (coverage 0.459).
3. v1.2 matched-Re 200k: frozen defers, N-flat (0.469).
4. **adaptive apparatus (regression-validated): `COVERAGE-SLIVER`** — dense-fiber
   fraction 0.036, dense-count median 0; the wall is a geometric fact, not
   bookkeeping.

The forcing axis stays closed. A future reopening would need a genuinely
different resolution mechanism (not coverage bookkeeping — that question is now
answered), which is its own registration. The anchor witness and the closed slate
ledger are untouched; H1's proxy-relative typing and H2's N48-stability stand.

Cross-refs: `NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md`,
`NSE_COVERAGE_ADAPTIVE_APPARATUS_SCOPE.md`, `NSE_H3_ADMISSION_RECEIPT.md`,
`results/proof/c1-adaptive-reg-g{200,300}/`,
`results/proof/c1-h3-kf3-g675-adaptive/` (this read),
`results/proof/c1-h3-kf3-g675-fb-twin/` (the frozen deferral this re-reads).
