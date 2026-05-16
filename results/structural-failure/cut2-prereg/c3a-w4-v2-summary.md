# C3-A Receipts (Wave 4.1 — Path Y + Path Z amendment)

## Summary

| receipt | status |
| --- | --- |
| C3-A-R (unchanged from v1) | PASS |
| C3-A-T v2 (Path Y) | BLOCK |
| C3-A-B v2 (Path Z) | BLOCK |

## C3-A-T v2 (Path Y framing)

Non-degenerate P_in: 587 (eligible 479, ineligible-by-obs 108).

**T1**: π_dec err = **3.567°** vs π_route err = **1.860°**. Margin = **-1.706°** (req ≥ 0.5°). **BLOCK**.

Subsets — L1-eligible-by-obs (479): dec 3.806°, route 1.398°. L1-ineligible-by-obs (108): dec 2.506°, route 3.912°.

T2 decoy-edit: dec err after 15.567° (≥2°)=true, route max shift 0.000° (≤0.5°)=true.

T3 handle-edit: route err 0.000° (≤2°), dec stale err 5.694° (>2°) = true.

## C3-A-B v2 (Path Z)

Sub-(i): BLOCK.
Sub-(ii) basin preserved within ±0.5° of q_naive: **381/479** = **79.5%** (req ≥ 90%) = BLOCK.

