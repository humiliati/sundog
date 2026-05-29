# Probe 05 - Nonlinear Zero-Statistics Reversibility Test

Status: v0 spec filed. No execution. Filed 2026-05-28.

Bridge admission: per
[`NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md`](NONLINEAR_PAIR_CORRELATION_BRIDGE_NOTES.md),
the only admitted hook is the **S2 gap-pair swap** on consecutive unfolded
gaps. This probe is a reversibility test for the gap sequence. It is not a
structural-zero probe, not a D3/S3 descent, and not evidence for or against RH.

Expected disposition before execution: `R-NL-NEG-A` (GUE dominance). The
sine-kernel / GUE baseline predicts reversal symmetry of consecutive-gap
statistics, hence `E[D] = 0`. A clean null is therefore expected and should be
reported as a bounded null receipt, not as a Sundog edge.

## Purpose

Probe 05 tests whether the low-lying zero gap sequence shows a registered
arrow-of-time asymmetry in consecutive unfolded gaps:

```text
(s_i, s_{i+1}) versus (s_{i+1}, s_i)
```

The observable is non-forced: individual samples generally have
`s_i != s_{i+1}`. This distinguishes Probe 05 from Probe 01 Path (i) and the C1
explicit-formula cell set, where the relevant zero residuals were identity-zero
by construction or by linearity.

The admission is narrow. Even if the test returns `D ~= 0`, the result belongs
to standard GUE / sine-kernel reversibility, not to a new structural-zero
mechanism.

## Claim Boundary

Allowed outcome language:

- "Probe 05 produced a bounded reversibility-test receipt on the registered
  zero window."
- "Probe 05 returned the expected GUE-dominance null (`R-NL-NEG-A`)."
- "Probe 05 found a finite-window reversibility anomaly requiring independent
  replication."
- "Probe 05 failed by registered floor, bridge, or domain-leakage falsifier."

Forbidden outcome language:

- "Evidence for RH."
- "Evidence against RH."
- "A structural-zero receipt for Riemann zeros."
- "D3/S3 symmetry has been found in zero statistics."
- "The C3 triple hook rescues the S2 null."
- "Residual bins reveal a new representation sector."

## Registered Domain

Default v0 execution domain:

- **Zero source:** Odlyzko `zeros1`, first `N_zero = 5000` positive ordinates,
  unless changed by a new spec before execution.
- **Expected height lock if the same source is used:** `gamma_5000 =
  5447.861998301`, matching the Probe 01 Path (i) receipt.
- **Statistic:** consecutive unfolded nearest-neighbor gap pairs only.
- **Symmetry hook:** `S2 = {id, tau}`, where
  `tau(s_i, s_{i+1}) = (s_{i+1}, s_i)`.
- **Baseline:** sine-kernel / GUE bulk process; registered expectation
  `E[D] = 0`.
- **No higher hook:** no C3 triple, no S3/D3 upgrade, no residual-bin sector in
  this receipt.

Any change to the zero source, `N_zero`, height window, unfolding rule, or
observable after inspecting output files triggers domain-leakage quarantine.

## Unfolding Rule

Use positive ordinates:

```text
0 < gamma_1 < gamma_2 < ... < gamma_N
```

For `i = 1 .. N_zero - 1`, define:

```text
gap_i    = gamma_{i+1} - gamma_i
center_i = (gamma_i + gamma_{i+1}) / 2
rho_i    = log(center_i / (2*pi)) / (2*pi)
s_i      = gap_i * rho_i
```

This is the same local-density convention used by the Probe 01 Path (i)
runner. If a reviewer rejects this unfolding before execution, file a v0.1 spec
with the replacement rule. Do not change the unfolding after seeing `D`.

## Gap-Pair Observable

For `i = 1 .. N_zero - 2`, define:

```text
pair_i = (s_i, s_{i+1})
delta_i = s_i - s_{i+1}
sign_i =
  +1 if delta_i > tie_tol
   0 if abs(delta_i) <= tie_tol
  -1 if delta_i < -tie_tol
```

Registered tie tolerance:

```text
tie_tol = 1e-14
```

Ties contribute `0` to the numerator and remain in the denominator.

Primary statistic:

```text
m = N_zero - 2
D = sum_i sign_i / m
```

Interpretation:

- `D > 0`: more adjacent gap descents than ascents.
- `D < 0`: more adjacent gap ascents than descents.
- `D = 0`: balanced orientation at the registered resolution.

The statistic is a time-reversibility observable for the gap sequence. It is not
forced to vanish by stationarity alone.

## Sampling Floor

The floor must be computed by the registered method before the verdict is
assigned. The method may use the realized sign sequence, but the algorithm,
block size, seed, and acceptance rule are fixed here and may not be tuned after
inspection.

Let `m = N_zero - 2`.

Analytic floor:

```text
tau_ind = 3 / sqrt(m)
```

Block-bootstrap floor:

```text
block_length = 64
B = 10000
seed = 20260528
```

Procedure:

1. Build the sign sequence `(sign_1, ..., sign_m)`.
2. Center it by subtracting its observed mean `D`.
3. Generate `B` circular moving-block bootstrap samples of length `m`, using
   blocks of length `64` and the fixed seed.
4. For each sample, compute the mean centered sign.
5. Set `tau_boot` to the empirical `0.9975` quantile of the absolute bootstrap
   means.

Registered decision floor:

```text
tau_D = max(tau_ind, tau_boot)
```

This is a receipt-level finite-window floor, not a theorem about the zeros. If
the bootstrap floor cannot be computed exactly as registered, file
`R-NL-NEG-C` rather than changing the method after output inspection.

## Disposition Table

| Outcome | Disposition |
| --- | --- |
| `abs(D) <= tau_D` | bounded reversibility-test null; `R-NL-NEG-A` (expected) |
| `abs(D) > tau_D` | GUE-reversibility anomaly flag; not a Sundog structural-zero; requires independent replication |
| tie count is unexpectedly large or source precision cannot support `tie_tol` | source / precision quarantine before interpretation |
| floor cannot be computed under the registered rule | `R-NL-NEG-C` sampling-floor failure |
| C3 triple hook invoked to rescue or reinterpret the result | `R-NL-NEG-B` representation-triviality quarantine |
| residual sign-bins invoked as a representation sector | `R-NL-NEG-A` / `R-NL-NEG-D` downgrade; standard residual analysis only |
| zero source, window, unfolding, or floor changed after seeing output | domain-leakage quarantine |

## Required Data Products

Required outputs under
`results/riemann/probe05-nonlinear-zero-statistics/`:

- `manifest.json` - source, `N_zero`, height, unfolding rule, floor settings,
  command, runner, code commit.
- `zeros.csv` - zero index, ordinate, source, validation status.
- `unfolded_gaps.csv` - `gap_i`, `center_i`, `rho_i`, `s_i`.
- `gap_pairs.csv` - `s_i`, `s_{i+1}`, `delta_i`, `sign_i`.
- `reversibility_summary.json` - `D`, counts, `tau_ind`, `tau_boot`, `tau_D`,
  disposition.
- `bootstrap_floor.csv` - bootstrap replicate means or enough summary data to
  audit `tau_boot`.
- `quarantine.csv` - excluded or suspect rows with reason.
- `README.md` - human-readable run note and falsifier disposition.

Durable reviewed receipts should be summarized in
`docs/riemann/receipts/` using [`RECEIPT_TEMPLATE.md`](RECEIPT_TEMPLATE.md).

## Pipeline Sketch

1. Load the frozen zero source.
2. Validate ordering, uniqueness, and precision.
3. Compute unfolded gaps using the registered local density rule.
4. Build consecutive gap pairs.
5. Compute `D`, tie count, and orientation counts.
6. Compute `tau_ind`, `tau_boot`, and `tau_D`.
7. Assign the registered disposition without adding C3, S3/D3, or residual-bin
   rescue language.
8. File result artifacts and a dated receipt.

## Falsifier Coupling

Probe 05 exercises the nonlinear-lane negatives:

- **`R-NL-NEG-A: GUE dominance`** - predicted if `abs(D) <= tau_D`.
- **`R-NL-NEG-B: representation triviality`** - fires if C3/S3/D3 is imported
  into this S2-only receipt.
- **`R-NL-NEG-C: sampling-floor failure`** - fires if the finite-window floor is
  not fixed or cannot be computed as registered.
- **`R-NL-NEG-D: bridge overreach`** - fires if local gap-pair reversibility is
  rewritten as functional-equation symmetry or as a structural-zero claim.

It also exercises the main ledger's Mode 5 if the domain is expanded after
inspection.

## Review Gate

Probe 05 is not complete until:

- artifacts exist under the registered result path;
- the receipt template is filled;
- one maintainer pass verifies source / `N_zero` / height / unfolding / floor
  consistency;
- the result is cross-linked from the Riemann index and main ledger;
- any non-null anomaly is replicated on a separately registered window before
  being discussed as more than a finite-window flag.

External sanity check target: an analytic number theorist or random-matrix /
point-process reviewer familiar with Montgomery-Odlyzko statistics and
consecutive-gap tests.

## Current State

- 2026-05-28: Probe 05 v0 spec filed.
- Execution: none.
- Expected result: bounded null, `R-NL-NEG-A`.
- Only admitted hook: S2 gap-pair reversibility.
- Quarantined hooks: C3 triple; residual-bin representation sectors; S3/D3
  upgrades.
