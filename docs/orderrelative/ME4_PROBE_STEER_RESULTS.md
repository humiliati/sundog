# ME-4 — Probe-Steer Gap on GPT-2's Stack-Top (results)

*Run 2026-07-02 by `scripts/chatv2_me4_probe_steer.py --n 2400 --max-queries 600`
(deterministic seeds; per-cell checkpoints; summary
`results/orderrelative/me4-probe-steer/summary.json`).
Spec: [`ME4_PROBE_STEER_SPEC.md`](ME4_PROBE_STEER_SPEC.md) — all gates and bands
pre-registered. Chat-v2 discipline: non-promotional; no R2 / world-model claim.*

## Verdict

> **`ME4_STEERS` — the linear address is causally load-bearing.** Every treatment
> family passed the write-validity gate at 1.000 and moved behavior to the one-pop-swap
> target at `follow_rel` 0.81–1.05, while random-direction and null-swap controls sat at
> 0.01–0.09 and the manifold stayed intact (rest-NLL ratio ≈ 1.0). **Census placement
> (pre-registered): GPT-2's stack-top state joins M∧E** — on this substrate, for this
> state, *probing is steering*.

## Setup (frozen instrument + provenance)

Corpus, lexer, windows, and probe protocol replicated unchanged from the banked H2
probe (`chatv2_h2_stacktop_probe.py`, frozen). Probe bank: 13,659 rows; probes L8/L11
holdout 0.941/0.965 (consistent with the banked read). Donors (probe-verified,
next-closer, hard): `(` 103 / `[` 229 / `{` 14 at L11 (the `{` pool is thin —
recorded). Query set: 1,785 next-closer positions, **deduplicated by absolute corpus
position**; primary (count-ambiguous) slice **n = 370**.

**Provenance note (two runs):** run 1 (`--n 800`) produced only 136 primary queries —
under the 150 floor — and exited early with a filtered message (exit 0); the rerun
raised windows to 2400 and added the position dedupe (window overlap had been
double-counting code positions). No gate or band was changed between runs.

## Gates

- **G0 (behavioral floor):** unpatched closer preference agrees with the TRUE top at
  **0.957** on the primary slice (0.985 all-slice) — far above the 0.60 floor. GPT-2's
  closer behavior tracks the stack-top; the write test can read out.
- **G1 (write validity):** **1.000 in every treatment cell** — the probe reads τ′ off
  every patched residual; every write took at the read address.
- **G2 (on-manifold):** median rest-of-continuation NLL ratio 0.98–1.06 across all
  cells (bound 3.0); random-direction controls moved preference at < 0.1 of treatment
  levels. Clean.

## Cells (primary slice, n = 370 each)

| cell | G1 took | follow(τ′) | follow_rel | rest-NLL× |
| --- | ---: | ---: | ---: | ---: |
| probedir L8 α2 | 1.000 | 0.935 | 0.977 | 1.00 |
| probedir L8 α4 | 1.000 | 1.000 | **1.045** | 1.00 |
| probedir L11 α2 | 1.000 | 0.986 | 1.031 | 1.00 |
| probedir L11 α4 | 1.000 | 1.000 | 1.045 | 1.00 |
| diffmeans L8 α2 | 1.000 | 0.968 | 1.011 | 1.00 |
| diffmeans L8 α4 | 1.000 | 1.000 | 1.045 | 1.02 |
| diffmeans L11 α2 | 1.000 | 1.000 | 1.045 | 1.00 |
| diffmeans L11 α4 | 1.000 | 1.000 | 1.045 | 1.00 |
| **transplant L8** | 1.000 | 0.776 | 0.811 | 1.05 |
| **transplant L11** | 1.000 | 0.919 | 0.960 | 0.98 |
| randdir L8 α2 (ctrl) | 0.073 | 0.057 | 0.059 | 1.00 |
| randdir L8 α4 (ctrl) | 0.159 | 0.089 | 0.093 | 1.01 |
| randdir L11 α2 (ctrl) | 0.057 | 0.049 | 0.051 | 1.00 |
| randdir L11 α4 (ctrl) | 0.116 | 0.068 | 0.071 | 1.01 |
| nullswap L8 (ctrl) | 0.000 | 0.030 | 0.031 | 1.06 |
| nullswap L11 (ctrl) | 0.000 | 0.014 | 0.014 | 1.00 |

Decision cell: `probedir_L8_a4`, `follow_rel = 1.045`, controls clean. `follow_rel > 1`
means steering follows τ′ *more* reliably than unpatched behavior follows the true top
(α = 4 saturates the flip). The decisive family — the **donor transplant**, on-manifold
by construction — follows at 0.78/0.92: even the strictest write moves behavior.

## What this places, and what it does not

- **Census (pre-registered placement):** the model-state row lands in **M∧E** — the
  stack-top's linear address is readable *and* writable; reads and writes pair, as in
  the clean tier. The **E∧¬M model-state cell remains unoccupied**: chat-v2's regime-2
  analog failed the verb fence (ME-1) and its admission test has now gone the other
  way on this substrate.
- **The M∧¬E signature did NOT appear here.** "You can read the model's state but not
  write it" is *false* for GPT-2's stack-top at the probe's address. The interp-orthodox
  reading (linear representations are causally load-bearing) wins this row.
- **The steering price is ≈ 0** (ME-5's column): on this state, enforcement through the
  owned activation node costs nothing measurable (rest-NLL ≈ 1.0×) — the M∧E cell's
  write-price contrasts maximally with the target channel's edge.
- **Not claimed:** anything beyond existence-tier — one 3-class state, one small model,
  one corpus (Python), light lexer, restricted closer readout (BPE-merge caveat), thin
  `{` donor pool. High-dimensional or semantically deep states may behave differently —
  that is a future row, not this receipt. No R2 promotion; `PROMOTE_GATE.md` untouched.

## Cross-links

Spec: [`ME4_PROBE_STEER_SPEC.md`](ME4_PROBE_STEER_SPEC.md) · census:
[`ME1_QUADRANT_CENSUS.md`](ME1_QUADRANT_CENSUS.md) (adjudication addendum) · slate:
[`ME_QUADRANT_HYPOTHESES_SLATE.md`](ME_QUADRANT_HYPOTHESES_SLATE.md) · read-side
receipt: `docs/chatv2/R2_INTERSECTION_HYPOTHESES.md` (H2) · JSON:
`results/orderrelative/me4-probe-steer/summary.json` + per-cell checkpoints + `run.log`.
