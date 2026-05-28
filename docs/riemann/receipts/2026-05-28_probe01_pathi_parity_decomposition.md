# Riemann Probe 01 v1 Receipt - Path (i) Parity Decomposition

## Header

- Receipt id: `2026-05-28_probe01_pathi_parity_decomposition`
- Probe: Probe 01 - Isotropy v0.3 on low-lying zero pair data
- Date: 2026-05-28
- Author / runner: Codex
- Code commit: `da4051388f0fc438ed3aecbcb47b2fe773ffb72a`
- Script: `scripts/riemann-probe01-pathi.mjs`
- Script SHA256: `9d0a32fe1445c298f9e0bb93ab9039cbf765ff25ebaf7fb9425200ace99fddfc`
- Result directory: `results/riemann/probe01-isotropy-zero-pairs/`
- Ledger version: `SUNDOG_V_RIEMANN.md` after 2026-05-28 lit-pass and bridge notes

Working-tree note: the manifest records the script as untracked relative to the
commit above because the executable was introduced for this receipt. The script
hash pins the executed code.

## Registered Domain

- Zero source: Odlyzko `zeros1`
- Source URL: `https://www-users.cse.umn.edu/~odlyzko/zeta_tables/zeros1`
- Source SHA256: `3436c916a7878261ac183fd7b9448c9a4736b8bbccf1356874a6ce1788541632`
- Source declared content: first 100,000 positive ordinates of non-trivial
  Riemann zeta zeros, accuracy `3e-9`
- `N`: 5,000
- Observed max height: `5447.861998301`
- Statistic: nearest-neighbor unfolded spacings
- Pair rule: consecutive positive-zero pairs `(gamma_i, gamma_{i+1})`, mirrored
  to `(-gamma_{i+1}, -gamma_i)`
- Unfolding:
  `u_i = (gamma_{i+1} - gamma_i) * log(center_i / (2*pi)) / (2*pi)`
- Representation bridge: Path (i), Z2 descent under `s -> 1 - s`
- Thresholds:
  - reflection residual <= `1e-12`
  - spacing sign-component <= `1e-12`
- Structural-zero status: not reachable under Path (i)

## Claim Under Test

Under the registered Path (i) Z2 descent, nearest-neighbor zero-spacing features
remain reflection-even on the first 5,000 Odlyzko `zeros1` ordinates, yielding a
parity-decomposition receipt but not a v0.3h structural-zero receipt.

## Artifacts

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | `results/riemann/probe01-isotropy-zero-pairs/manifest.json` | run-local | source and run lock |
| Zeros | `results/riemann/probe01-isotropy-zero-pairs/zeros.csv` | `a926ff48ad62cabcea05aec544dbdf94377d2f307d40879800423b2bb1c6fc41` | parsed zero window |
| Spacings | `results/riemann/probe01-isotropy-zero-pairs/unfolded_spacings.csv` | `26da0167ac317c5ee8d136eecdcfcc66419dd1d51ae9aae267f3a2c792814223` | unfolded nearest-neighbor data |
| Pair features | `results/riemann/probe01-isotropy-zero-pairs/pair_features.csv` | `f815ecb0b35e54a7d5c9933e0fc6c36eb7adc6af1f727e45d2f9da516abbc3d3` | parity feature table |
| Isotropy records | `results/riemann/probe01-isotropy-zero-pairs/isotropy_records.csv` | `a75934400ad9b09396b1ea29d77c05d18007b7e5fa6482779bad8b5b5ab6b542` | per-pair Path (i) records |
| Summary | `results/riemann/probe01-isotropy-zero-pairs/structural_zero_summary.csv` | `fbbcc39d4be5da2f5cc25d50bc38c7cce67545662d783669b614bba9f93a19cd` | boundary summary |
| Quarantine | `results/riemann/probe01-isotropy-zero-pairs/quarantine.csv` | `dc754196175fe89094d92abcd8cfd0b7cfe4d1bb3757f9cdf4847e460f0fbb4f` | empty quarantine table |
| Source | `results/riemann/probe01-isotropy-zero-pairs/source/zeros1.txt` | `3436c916a7878261ac183fd7b9448c9a4736b8bbccf1356874a6ce1788541632` | raw Odlyzko table |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| Zero validation | finite, positive, ordered, max height < `1e4` | pass; max height `5447.861998301` | pass |
| Pairs analyzed | `N - 1` | `4999` | pass |
| Mean unfolded spacing | report-only | `1.0000184725911547` | report |
| Median unfolded spacing | report-only | `0.9604449773098614` | report |
| Max reflection residual | <= `1e-12` | `0` | pass |
| RMS spacing sign-component | <= `1e-12` | `0` | pass |
| RMS signed-height component | report-only | `0.6097931805142922` | report |
| Quarantine count | `0` for clean run | `0` | pass |

## Falsifier Disposition

- Mode 1 - invariant mismatch after alignment: not exercised; no alignment
  subpass admitted.
- Mode 2 - isotropy v0.3 structural failure: clean Path (i) parity catalog;
  no structural-zero category invoked.
- Mode 3 - projection residual breach: not exercised.
- Mode 4 - dynamical escape under stress-test: not exercised.
- Mode 5 - domain leakage / scope creep: avoided. Path (ii) S3-via-triple was
  not invoked.

## Verdict

Bounded positive receipt under Path (i): parity-decomposition receipt.

Spacing-derived features are reflection-even inside the registered window. The
nonzero sign sector is only the signed-height carrier created by including
height as a coordinate. Branch B structural-zero language is not reachable under
Path (i), so this receipt must not be described as a v0.3h structural-zero
result and must not be described as evidence for or against RH.

## Notes

This receipt is deliberately small. It establishes that Probe 01 v1 can produce
a clean, auditable artifact under the reduced Z2 bridge. The next escalation
should be either an external sanity check on whether this parity-decomposition
receipt has any Front-A edge beyond ordinary zero-statistics bookkeeping, or a
separate Front-A reading note on functional-equation reflection as a
structural-zero scaffold for smoothed explicit formulae.
