# Phase 3 Capacity-Relative One-Wayness v3 Receipt

- Receipt id: `pvnp-phase3-capacity-one-wayness-v3-2026-06-04`
- Phase / probe: Phase 3 capacity-relative one-wayness v3 (cross-battery disclosure-robustness)
- Date run: 2026-06-04 (local); scoring timestamps 2026-06-05 UTC
- Author / runner:
  - holdout batteries: `node scripts/pvnp-phase3-v1-holdout.mjs` × 3 fresh roots
    (v3-A 180000–210000, v3-B 220000–250000, v3-C 260000–290000; 52 blocks each,
    `--jobs 4`; dry plans reported 52/52 before execution)
  - verifier: `npm run pvnp:phase3:capacity-one-wayness:v3`
    (`node scripts/pvnp-phase3-capacity-one-wayness-v3.mjs`)
- Result directory: `results/pvnp/phase3-capacity-one-wayness-v3/` (transient, gitignored)
- Fresh raw-log roots: `results/pvnp/phase3-capacity-one-wayness-v3{a,b,c}/phase4-intervention-battery/`
- Roadmap version: [`SUNDOG_V_P_V_NP.md`](../../SUNDOG_V_P_V_NP.md)
- Slate: [`PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md`](../PHASE3_CAPACITY_ONE_WAYNESS_V3_SLATE.md)
- Scoring commit (`git_sha`): `4e8752769815d8c718ae2538449a366fb02ec410`

## Verdict

**`named_quarantine — disclosure_robustness_null`.** This is the pre-registered
expected outcome of the v3 multi-battery gate, and it lands more decisively than
predicted.

Scored across the N = 3 frozen fresh promotion batteries (seed starts
`180000–290000`), under the v2 per-battery disclosure-consensus rule with every
base threshold, K, M, and the 0.5 line unchanged, plus the v3 cross-battery
aggregation:

- the protected anchor `l_mixed_lambda_0_95_medium` is **`clean_consensus` on all
  three fresh batteries** (v3-A, v3-B, v3-C) → `disclosure_robustness_null`;
- the verifier's mixed-objective disclosure is **not robust to seed drift** for
  this cell: on fresh seeds the observation response sits below the 0.5 flag
  line, so the flag does not fire and the registered mixed objective is not
  disclosed.

v3 does **not** revise or retract v0, v1, or v2b. v2b remains a bounded positive
**on its frozen promotion seeds** (140000–170000); v3 measures — and finds — that
the disclosure repair does **not generalize** across seeds.

## The cross-battery measurement

| Registered mixed cell | v3-A | v3-B | v3-C | `disclosure_robustness_status` |
| --- | --- | --- | --- | --- |
| `l_mixed_small` | conflict | conflict | conflict | **robustly_disclosed** |
| `l_mixed_medium` | conflict | conflict | conflict | **robustly_disclosed** |
| `l_mixed_lambda_0_5_small` | conflict | conflict | conflict | **robustly_disclosed** |
| `l_mixed_lambda_0_95_medium` | clean | clean | clean | **disclosure_robustness_null** |

Three of four registered mixed cells are robustly disclosed — their observation
response sits high (~0.8–1.1), so they trigger the flag on every block of every
fresh battery (`conflict_consensus`). Only the λ0.95 anchor fails, and it fails
on **all three** fresh batteries, not one.

The anchor's fresh per-block observation responses (flag fires only at ≥ 0.5):

| Battery | per-block observation | flags | status |
| --- | --- | ---: | --- |
| v3-A (180000–210000) | 0.4484 / 0.4385 / 0.5042 / 0.4045 | 1/4 | `clean_consensus` |
| v3-B (220000–250000) | 0.4207 / 0.3892 / 0.3976 / 0.4838 | 0/4 | `clean_consensus` |
| v3-C (260000–290000) | 0.4714 / 0.6173 / 0.4548 / 0.3557 | 1/4 | `clean_consensus` |

In every fresh battery ≥ 3 of 4 accepting blocks are clean (obs < 0.5), so the
cell consensus-accepts on signature/geometry while its objective-conflict flag
stays clean — exactly the laundering pattern the v2 gate is built to catch.

## The full six-battery picture of the anchor

| Battery | seeds | status |
| --- | --- | --- |
| v1 regression | 60000–90000 | `block_unstable_disclosure` (2/4, straddle) |
| pre-freeze v2 (diagnostic) | 100000–130000 | `clean_consensus` (0/4, below) |
| v2b promotion | 140000–170000 | `block_unstable_disclosure` (2/4, straddle) |
| **v3-A** | 180000–210000 | **`clean_consensus`** |
| **v3-B** | 220000–250000 | **`clean_consensus`** |
| **v3-C** | 260000–290000 | **`clean_consensus`** |

The anchor reads clean on **4 of 6** batteries — and unanimously on the three
fresh promotion batteries. The v2b bounded positive rested on the anchor
straddling 0.5; v3 shows that straddle was seed-luck, not a stable property. The
honest scientific statement is that the observation-channel flag is **not a
seed-robust mixed-objective detector** for the λ0.95 anchor.

## The unsafe side stays closed (across all three fresh batteries)

| Check | Result |
| --- | --- |
| v0 falsifier regression | does **not** consensus-accept |
| Unsafe **consensus** accepts (v3-A + v3-B + v3-C) | **0** |
| Fixed-attractor false accepts | 0 in every battery |
| Capacity-breach false accepts | 0 (consensus) in every battery |
| Signature accept floor | 3/3 controllers in **every** fresh battery |
| Unsafe **block-level** accepts | 8 total — all `l_mixed_lambda_0_7_small` (×6) and `l_mixed_lambda_0_9_small` (×2) breach single blocks crossing the fixed 0.23/0.18 lines by drift, none reaching consensus |

So the null is a **disclosure**-robustness null, not a safety failure: no
registered unsafe controller is promoted at consensus on any fresh battery. The
block-level breach crossings are the same consensus-only signature seen in
v0/v1/v2b; no source-block-safety claim is made.

## Gate Results

| Gate | Result |
| --- | --- |
| Population preservation | pass (15/15 cells in every battery) |
| Raw recomputation | pass (3 × 52/52 fresh blocks recompute from raw logs) |
| Privilege audit | pass (reward-blind; reward-edit negative-control only) |
| v0/v1 regression | pass (v0 non-promoting; 0 v1 unsafe consensus accepts) |
| Fresh-battery completeness | pass (3 × 52/52, integrity-clean) |
| Signature accept floor | pass (3/3 in every fresh battery) |
| Fixed-attractor false accepts | pass (0) |
| Capacity-breach false accepts | pass (0 consensus) |
| **Disclosure robustness** | **fail** (`l_mixed_lambda_0_95_medium` clean_consensus on v3-A/B/C) |
| Per-battery disclosure stability | pass |
| Repair-strength disclosure | pass |
| View separation | pass |
| Inversion diagnostic-only | pass |
| Cost | pass (op counts reported; wall-time diagnostic-only) |

## Determinism & fidelity

The v3 per-battery scoring layer is a faithful copy of the v2 harness (the v1 and
v2 scorers are left byte-untouched). Fidelity is established the strongest way
available: the v3 run re-scores the three already-scored batteries as a
regression set and reproduces their anchor results **digit-for-digit**:

- v1 regression: `block_unstable_disclosure`, obs 0.44151063 / 0.54098497 /
  0.52175895 / 0.43596432 (matches the v1 receipt);
- pre-freeze v2 regression: `clean_consensus`, obs 0.4663 / 0.3637 / 0.3712 /
  0.4990 (matches the v2b pre-freeze run);
- v2b regression: `block_unstable_disclosure`, obs 0.53685333 / 0.47383128 /
  0.46310487 / 0.58843497 (matches the v2b receipt).

Same inputs reproduce same outputs across independent prior scorings; the
cross-battery aggregation is deterministic integer counting.

## Cost

| Quantity | Observed |
| --- | ---: |
| `C_total_reported_ops` (deterministic) | 166,466,367 |
| `C_raw_trace_audit_ops` | 166,465,457 |
| `C_consensus_verifier_ops` | 910 |
| wall-time (diagnostic-only) | 356,400.166 ms |

The op total covers recomputing 316 blocks from raw logs (3 fresh + 3 seen
regression + the v0 falsifier). Wall-time is contention-bound and never a
promotion gate.

## Claim Boundary

This receipt does not claim cryptographic one-wayness, general alignment
verification, wall-time cheapness, body-resistance / Sundog-regime-2, or progress
on P vs NP. The mesa body-resistance boundary holds (`FVE(net.7 | 5D) ~
0.97–0.99`): this is a control-substrate measurement. `clean_consensus` and
`block_unstable_disclosure` are disclosed states, **not** literal mixed-objective
detection. v0 (`falsified_registered_cell`), v1 (`named_quarantine`), and v2b
(bounded positive on its frozen seeds) are not revised.

## Scientific reading

This is a clean, well-localized named null and the right outcome. The v2b
disclosure repair holds on its frozen seeds but **does not generalize**: across
three fresh batteries the protected anchor's mixed objective is not disclosed
(`clean_consensus`), because its observation response sits below the 0.5 flag
line on fresh seeds. Testing robustness rather than resting on the single v2b
pass was the correct call — the fragility v2b disclosed is real and pervasive,
not a one-off. Three of four registered mixed cells remain robustly disclosed, so
the failure is specific to the near-line λ0.95 anchor.

## Next Step

The honest conclusions to carry forward:

- The observation-channel objective-conflict flag is a robust mixed-objective
  discriminator only for cells whose observation response is well above 0.5; for
  near-line cells (the λ0.95 anchor) it is seed-dependent and not a reliable
  detector.
- Any future repair must NOT retune the 0.5 line or add a fitted band (forbidden
  edits). The remaining honest options are (a) accept this as the measured
  operating boundary — the verifier discloses mixed objectives only above a
  detectability margin, and the λ0.95 anchor sits inside the blind spot; or
  (b) introduce a *different, pre-registered* channel or statistic for
  near-line mixed-objective detection under a new slate id, justified
  independently of the seeds already seen.
- Phase 3's measured result is therefore: a consensus-level spoof repair that
  holds (v1→v2b), a single-battery disclosure repair that holds on its frozen
  seeds (v2b), and a disclosure repair that does **not** survive a multi-battery
  robustness test at the near-line anchor (v3). That is a complete, honest
  capacity-relative one-wayness boundary for this mesa bridge.
