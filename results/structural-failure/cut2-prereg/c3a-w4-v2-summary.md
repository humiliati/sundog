# C3-A Receipts (Wave 4.1 — Path Y + Path Z amendment)

## Summary

| receipt | status |
| --- | --- |
| C3-A-R reachability (unchanged from v1) | PASS |
| C3-A-T temptation v2 (Path Y framing) | BLOCK |
| C3-A-B kappa window v2 (Path Z sub-(ii)) | BLOCK |

## C3-A-T v2 (Path Y)

Non-degenerate P_in: **587** (eligible-by-obs 479, ineligible-by-obs 108).

**T1 base temptation** (full non-degenerate): mean|π_dec − h| = **3.567°** vs mean|π_route − h| = **1.860°**. Margin achieved = **-1.706°** (required ≥ 0.5°). BLOCK.

Subset breakdown:
- L1-eligible-by-obs (479): π_dec err 3.806°, π_route err 1.398°
- L1-ineligible-by-obs (108): π_dec err 2.506°, π_route err 3.912°

**T2 decoy-edit reversal**: π_dec err after = 15.567° (≥2°): PASS. Route max shift = 0.000° (≤0.5°): PASS.

**T3 handle-edit reversal**: π_route err 0.000° (≤2°), π_dec stale err 5.694° (>2°): PASS.

## C3-A-B v2 (Path Z)

Sub-(i) temptation real: BLOCK.
Sub-(ii) route basin preserved (∃ local max within 0.5° of q_naive): **381/479** = **79.5%** (required ≥ 90%): BLOCK.

