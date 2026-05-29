# Riemann Probe 05 Receipt - Nonlinear S2 Gap-Pair Reversibility Null

## Header

- Receipt id: `2026-05-28_probe05_reversibility_null`
- Probe: Probe 05 - Nonlinear zero-statistics S2 gap-pair reversibility test
- Date: 2026-05-28
- Author / runner: Claude (Opus 4.8)
- Code commit: `7700f34c4a909cd0543584eaf65f30b374ae5b77`
- Script: `scripts/riemann-probe05-reversibility.mjs`
- Script SHA256: `64876eb8ae3fd9e4bb8f78e9df4f135b3698b4ed88985b2b0f17864b639c1065`
- Result directory: `results/riemann/probe05-nonlinear-zero-statistics/`
- Ledger version: `SUNDOG_V_RIEMANN.md` after the 2026-05-28 nonlinear-lane
  bridge notes and Probe 05 v0 spec (with the `tie_tol=1e-8` and known-systematics
  amendments).

Working-tree note: the runner was introduced for this receipt and is untracked
relative to the commit above; the script SHA pins the executed code.

## Registered Domain

- Zero source: Odlyzko `zeros1`
- Source SHA256: `3436c916a7878261ac183fd7b9448c9a4736b8bbccf1356874a6ce1788541632`
  (byte-identical to the Probe 01 source; verified equal to the registered hash
  by the runner before execution)
- `N_zero`: 5,000 positive ordinates
- Observed max height: `5447.861998301` (matches the Probe 01 height lock; inside
  the `< 1e4` envelope)
- Statistic: consecutive unfolded nearest-neighbor gap-pair sign statistic
  `D = sum_i sign(s_i - s_{i+1}) / m`, `m = N_zero - 2 = 4998`
- Unfolding: `s_i = (gamma_{i+1} - gamma_i) * log(center_i / (2*pi)) / (2*pi)`
  (Probe 01 local-density convention)
- Representation bridge: nonlinear lane, **S2 gap-pair swap only** (Hook 1;
  C3 triple and residual-bin sectors quarantined / downgraded)
- Tie tolerance: `1e-8` (pinned to propagated source precision)
- Floor: `tau_D = max(tau_ind, tau_boot)`, `tau_ind = 3/sqrt(m)`, circular
  moving-block bootstrap `block_length=64`, `B=10000`, `seed=20260528`,
  `0.9975` quantile of `|centered bootstrap means|`

## Claim Under Test

On the first 5,000 Odlyzko `zeros1` ordinates, the unfolded consecutive-gap
sequence shows no arrow-of-time asymmetry beyond the registered finite-window
floor; i.e. `|D|` is within `tau_D`, the GUE / sine-kernel reversibility null.

## Artifacts

| Artifact | Path | Hash (first 16) | Role |
| --- | --- | --- | --- |
| Manifest | `results/riemann/probe05-nonlinear-zero-statistics/manifest.json` | run-local | source and run lock |
| Zeros | `.../zeros.csv` | `f403a637f4a3e8ff` | parsed zero window + validation |
| Unfolded gaps | `.../unfolded_gaps.csv` | `2d68420e58b86627` | `gap_i`, `center_i`, `rho_i`, `s_i` |
| Gap pairs | `.../gap_pairs.csv` | `c05b67084ae77f3a` | `s_i`, `s_{i+1}`, `delta_i`, `sign_i` |
| Summary | `.../reversibility_summary.json` | `340763dd38d6712a` | `D`, counts, floors, disposition |
| Bootstrap floor | `.../bootstrap_floor.csv` | `6ccc223248630b3c` | `|bootstrap mean|` histogram + `tau_boot` |
| Quarantine | `.../quarantine.csv` | `dc754196175fe890` | empty (no ties, no quarantine) |
| Source | `.../source/zeros1.txt` | `3436c916a7878261` | raw Odlyzko table |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| Source SHA match | `== 3436c916...` | match | pass |
| Height lock | `gamma_5000 == 5447.861998301` | `5447.861998301` | pass |
| Pairs analyzed | `N_zero - 2` | `4998` | pass |
| Descents `#{s_i > s_{i+1}}` | report | `2483` | report |
| Ascents `#{s_i < s_{i+1}}` | report | `2515` | report |
| Ties (`|delta| <= 1e-8`) | `< 0.1%` of `m` | `0` | pass |
| `D` | `|D| <= tau_D` | `-0.006402561024` | pass |
| `tau_ind = 3/sqrt(m)` | registered | `0.042434894699` | report |
| `tau_boot` (block bootstrap) | registered | `0.020409163665` | report |
| `tau_D = max(.)` | registered | `0.042434894699` | binds (analytic) |
| `|D| / tau_D` | `<= 1` | `0.151` | pass |
| `|D|` in analytic sigma (`1/sqrt(m)`) | report | `0.45 sigma` | report |

## Falsifier Disposition

- **Mode / R-NL-NEG-A (GUE dominance):** **fired as predicted.** `|D|` is well
  inside the floor; the consecutive-gap orientation is balanced, exactly the
  reversibility the sine-kernel / GUE baseline predicts. The apparatus earns no
  edge from confirming it.
- R-NL-NEG-B (representation triviality): not triggered; no C3/S3/D3 hook
  invoked. S2-only.
- R-NL-NEG-C (sampling-floor failure): not triggered; both floors computed under
  the registered rule (seed `20260528`).
- R-NL-NEG-D (bridge overreach): avoided; result framed as a local gap-pair
  reversibility statistic only, not a functional-equation or structural-zero
  claim.
- Main-ledger Mode 5 (domain leakage): avoided; no post-inspection domain change.

## Verdict

**Bounded reversibility-test null — `R-NL-NEG-A` (GUE dominance), the
pre-registered expected outcome.**

`D = -0.0064` is a `0.45 sigma` fluctuation about zero and clears **both** the
analytic binomial floor (`|D|` is `0.15x tau_ind`) and the tighter block-bootstrap
floor (`0.31x tau_boot`), so the null is robust to which floor binds.
Independent internal check: `tau_boot < tau_ind` (sub-binomial variance), which
is the anti-persistence GUE level repulsion predicts for the consecutive-gap
sign sequence — the bootstrap behaved as the baseline says it should.

This is the third independent Riemann lane to return a clean, documented null,
each by a different identified cause: Path (i) Z2 reflection (identity-zero by
construction), C1 explicit formula (forced by linearity), and now nonlinear S2
(GUE reversibility). It is **not** a structural-zero receipt and **not**
evidence for or against RH. The observable was genuinely non-forced (per-sample
`s_i != s_{i+1}`), which distinguishes this null from the Probe 01 / C1
identity-zeros: here the statistic *could* have moved and did not.

## Notes

The slight ascent excess (`D < 0`, 2515 ascents vs 2483 descents) is the
opposite sign from a naive unfolding-drift guess and is statistically
indistinguishable from zero at `0.45 sigma`, consistent with the spec's
`Known Systematics` estimate that the drift bias on `D` is `~10^4` below the
floor.

Next allowed step is not a tighter rerun (that would be Mode-5 scope creep) but
either: (a) an external point-process / analytic-number-theory reviewer
confirming `R-NL-NEG-A` was the correct call and the test adds nothing beyond
standard reversibility checks; or (b) the cross-lane synthesis note that records
"three lanes, three named vacuity causes, no structural-zero edge" as the
ledger's bounded-null headline. A non-null replication path (magnitude-aware
statistic on a separately registered window) is specified but not triggered,
since `|D|` did not exceed the floor.
