# Chat-v2 H1/V3-0b — Ambiguity-Slice Bank Redesign (spec amendment)

> 2026-07-01, pre-registration. **Amends `H1_V3_STATEBANK_SCOPE.md`** (§2 surface criterion →
> slice-conditional; §4 chess bank construction; §6/§7 V3-0 gate and branches). Everything not
> amended here is inherited unchanged — ceilings, balance window, ≥24 axes, no-rescue, fences,
> the V3-0.5 → V3-1 → V3-2 ladder and its gating. **Non-promotional. No model, no GPU at this
> rung. Nothing here alters `PROMOTE_GATE.md`.**

## 1. What this answers

V3-0 (`H1_V3_0_DATA_ADMISSION_RECEIPT.md`) failed both corpora with one diagnosis: **state is
surface-undecodable only on the count-ambiguous slice, and in natural distributions that slice
is thin.** Witness pairs existed (the state is order-dependent *in principle*) while the probe
gate failed on the whole distribution (bag-dominated *in practice*). Every prior positive in
this lane lived on an ambiguity slice: H2's crossover was measured on count-ambiguous bracket
positions; the R1 toy's latents were ambiguous *by construction* (uniform bits). V3-0b makes
the slice the explicit object: **define each axis on its ambiguity slice, then run the
unchanged gate there.**

Theory anchor: a slice instance is a point on a **non-degenerate fiber of the bag statistic**
— the empirical image of `SurfaceBag`'s witness construction (`([` vs `[(`). On the slice,
σ_surface > 1 holds instance-wise, not just family-wise.

## 2. The slice (the new §2)

For an axis `a` and registered surface statistic = the **UCI move bag**:

- **Candidate criterion (mechanical, exact replay, no probe/model input):** an instance is
  slice-candidate for axis `a` = (pred, sq) iff the square's piece-code (∅/wP/bP/w/b) **changes
  ≥ 2 times** during the prefix. Board-diff per ply, so captures, en passant, and castling
  rook squares are covered automatically.
- **Witness certification (per axis):** ≥ 1 (target 3) interpreter/replay-verified pairs —
  two legal orderings of the same per-color move multisets with different truth for `a` —
  drawn by **bounded local search** (adjacent same-color swaps + random block shuffles;
  random full permutations rarely stay legal at depth). Budget 600 s total. Pairs are
  certificates, not corpus (inherited §2B/§3.5 conventions); stored as JSONL.
- **Coverage honesty metric:** on a ≥ 200-instance subsample, report the fraction of
  slice-candidates that get instance-level witnesses within budget — measures how loose the
  candidate criterion is. Reported, not gated.

Properties worth stating: the candidate slice may retain bag-*correlation* (ambiguous ≠
uniform), so **the probe gate on the slice is a real test, not a tautology** — it now measures
exactly the residual: how well natural move statistics pin state on formally-undetermined
fibers. Budget shortfalls shrink witness coverage; they never corrupt it (conservative).

## 3. Primary arm: chess (the new §4)

- **Corpus:** Lichess 2013-01 auto-parquet (as V3-0); games ≥ marker+4 plies, cap 2600;
  replay oracle python-chess.
- **Marker scan {24, 32, 40}, declared rule:** maximize the number of axes meeting the slice
  floor (mechanical; no probe peeking). Feasibility scan (2026-07-01, 1000 games — scoping
  data, not a gate run): squares with ≥ 120 candidate instances = 33 / 44 / **54** of 64 at
  ply 24 / 32 / 40; **co-ambiguity per instance at ply 40: median 22, p75 24, max 28.**
- **Axes:** piece-on-square family (occ/w/b/P/wP/bP × 64), evaluated **on each axis's slice**;
  cap at the 48 most-balanced (24 needed). Conditions: UCI = registered; SAN/sanitized
  reported as leak controls (cannot supply witnesses — reordering changes the SAN bag).
- **Gates per axis (all inherited numbers):** slice ≥ 120 instances; balance [0.40, 0.60] *on
  the slice*; ALL probes ≤ 0.60 held-out *on the slice* (LR-counts, LR-tfidf 1–2, MLP-on-counts
  w ∈ {1,2,3}, LR-meta on ECO); ≥ 1 witness pair. Group split by game, seed 0, probe cap 1600.
- **Liveness control (new):** one declared bag-determined axis — `e2e4`-token-present — probed
  on slice instances must read ≥ 0.95, else the apparatus is dead (F4-style abort, no verdict).
- **Admission:** ≥ 24 surviving axes → `H1-V3-0B-ADMIT`.

## 4. Optional arm: code (corpus-floor-gated)

V3-0b does **not** revive the code arm by itself — its F3 was corpus-structural
(`F3-V3/copy`: computed share 0.07), not distributional. The arm runs only if a corpus
expansion (HF code corpora, e.g. the-stack-smol via auto-parquet, mined identically) first
clears a declared floor: **≥ 1,500 executable prefixes AND computed-value share ≥ 0.25** —
else `F3-V3b/corpus` immediately, no probing. If cleared: slice = slots with ≥ 2 updates +
statement-permutation witnesses; gates as §3.

## 5. Branches

| branch | meaning |
| --- | --- |
| `H1-V3-0B-ADMIT` | ≥ 24 axes survive slice floor + balance + probes + witnesses |
| `F3-V3b/slice` | ambiguity slices too thin (axes under the 120 floor) |
| `F3-V3b/input` | probes still read the slice — **the strong negative:** natural move statistics pin state even on formally-undetermined fibers |
| `F3-V3b/witness` | axes lack witness pairs within budget |
| `F3-V3b/bank` | < 24 axes clear |
| `F3-V3b/corpus` | code-arm expansion floor unmet |

No rescue: if the slice gate fails, file the branch and stop.

## 6. What admission licenses (and doesn't)

An `H1-V3-0B-ADMIT` licenses **only** a *scoped* bank: "≥ 24 state axes, each
surface-undecodable **on its ambiguity slice**." Two honesty consequences carried forward:

- **d_dec re-scoping.** Downstream (V3-1/V3-2), each axis is evaluated on its slice. The bank
  d_dec is the count of axes with above-floor carry on their slices (**union form** — weaker
  than the toy's one-site joint form). The **co-ambiguity report** (per-instance candidate-axis
  count; median 22 at ply 40) must ship with any fingerprint so the joint-position form can be
  assessed against it. Whether the union form satisfies `PROMOTE_GATE.md` R2's `d_dec ≥ 20`
  line is an **explicit owner decision at V3-2 prereg** — this spec flags it, does not settle it.
- Claim language: "state tracking on ambiguity slices of real games," never unqualified
  "world model." All scope-doc fences inherited.

V3-0.5 (GPT-2 calibration) and V3-1 inherit the slice construction unchanged: same axes, same
slices, model representations swapped in for surface features; V3-0.5 runs only on admission,
per the scope's §11.

## 7. Deliverables & frozen defaults

- Script: `scripts/chatv2_h1_v3_0b_slice_admission.py` (chess arm; `--arm code` optional arm).
- Receipt: `docs/chatv2/H1_V3_0B_SLICE_ADMISSION_RECEIPT.md` — verdicts per arm, per-axis
  table (slice n, balance, per-probe, witness count), liveness row, coverage honesty metric,
  co-ambiguity distribution, ≥ 1 exact witness pair with labels, marker-scan table.
- Frozen: change-count ≥ 2; markers {24,32,40} by floor-count; slice floor 120; balance
  [0.40,0.60]; ceiling 0.60; axis cap 48; witness ≥ 1/axis (target 3), 600 s budget,
  local-swap search; liveness ≥ 0.95; games cap 2600; probe cap 1600; group split by game;
  seed 0; code-arm floor 1,500 prefixes / 0.25 computed share.

Cross-refs: `H1_V3_STATEBANK_SCOPE.md` (parent prereg), `H1_V3_0_DATA_ADMISSION_RECEIPT.md`
(the diagnosis), `R2_INTERSECTION_HYPOTHESES.md` (H2/H5 anchors), `sundogcert` `SurfaceBag.lean`
(the fiber theory).
