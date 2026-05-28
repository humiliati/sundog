# Riemann Receipt Template

Use this template for every `sundog_v_riemann` probe result. Keep null and
negative results. Do not rewrite a failed receipt into a better-looking scope.

## Header

- Receipt id:
- Probe:
- Date:
- Author / runner:
- Code commit:
- Result directory:
- Ledger version:

## Registered Domain

- Zero source or periodic-data source:
- Source hash / generator version:
- `N` or height/cutoff:
- Statistic:
- Smoothing / binning:
- Representation bridge:
- Thresholds:
- Random seeds, if any:

## Claim Under Test

One sentence. It must be narrower than the ledger.

## Artifacts

List files and hashes:

| Artifact | Path | Hash / version | Role |
| --- | --- | --- | --- |
| Manifest | | | source and run lock |
| Primary CSV / JSON | | | data |
| Summary | | | receipt summary |

## Observed Values

| Quantity | Registered threshold | Observed value | Pass/fail/quarantine |
| --- | --- | --- | --- |
| Closure residual | | | |
| Structural-zero count | | | |
| Quarantine count | | | |
| Alignment score | | | |
| Variance / residual breach | | | |

## Falsifier Disposition

Choose one or more:

- Mode 1 - invariant mismatch after alignment:
- Mode 2 - isotropy v0.3 structural failure:
- Mode 3 - projection residual breach:
- Mode 4 - dynamical escape under stress-test:
- Mode 5 - domain leakage / scope creep:

## Verdict

Allowed values:

- bounded positive receipt;
- null receipt;
- structural-zero receipt;
- named quarantine;
- falsified in registered cell;
- void run.

## Notes

Record interpretation, caveats, and next allowed step. If a domain expansion is
desired, name it as a new probe rather than modifying this one.
