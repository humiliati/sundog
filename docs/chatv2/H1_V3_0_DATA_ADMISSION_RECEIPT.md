# Chat-v2 H1/V3 — V3-0 Data-Admission Receipt

> 2026-07-01. Run of `H1_V3_STATEBANK_SCOPE.md` rung V3-0 (CPU, no model).
> **Non-promotional. No model run, no GPU, no R2 claim.** Screens whether either scoped
> corpus yields a state bank of ≥24 balanced axes surviving the **H5-upgraded** surface
> criterion (MLP-on-counts suite + interpreter/replay-verified witness pairs).
> Script: `scripts/chatv2_h1_v3_data_admission.py`. Artifacts: `results/chatv2/h1_v3/`.

## Verdicts

| corpus | branch | one-line cause |
| --- | --- | --- |
| code variable-state (primary) | **`F3-V3/copy`** | the executable straight-line slice of real code is literal-dominated — computed-value share **0.07**, 28/32 value axes emptied by the §3.4 filter |
| chess PGN board-state (secondary) | **`F3-V3/input`** | all 39 balanced piece-on-square axes are surface-readable at **0.94–0.996** in *every* condition — including UCI, where each move token names its destination square |

**V3-0 does not admit a bank. Per §11: V3-0.5 and V3-1 do not run; no GPU/H200 work;
`PROMOTE_GATE.md` R2 unchanged.**

## Code corpus (primary) — `F3-V3/copy`

- **Mined:** 14,750 real Python files (local repos + stdlib + site-packages), 844
  straight-line runs (len ≥ 3), **1,373 executable canonical prefixes** (whitelist int
  interpreter, |v| ≤ 1e9, ≤8 slots, cap 4 prefixes/run).
- **The kill:** provenance-computed share across slots = **363/5541 = 0.07**. Real
  straight-line assignment runs are *config blocks* — literal constants — so the §3.4
  computed-value filter (correctly) empties 28/32 value axes below min-N 120; 4 more die on
  thin slots. Real code keeps its computed state inside loops and calls, outside the
  restricted interpreter's reach by construction.
- **Apparatus liveness:** the one axis reaching probes (`v3.defined`, balance 0.49) was read
  at **1.000** — as predicted, `defined` is a function of the statement *set* (bag-determined,
  provably: no witness pair can exist). The suite reads a bag-determined axis perfectly.
- **Witness generator validated** (not corpus data — generator certificate only):
  `v0 = -2 ; v0 = v0 % 2 ; v0 = v0 % 3` vs the swapped update order — identical token bag,
  final parity flips (`label_a=False, label_b=True`).

## Chess corpus (secondary) — `F3-V3/input`

- **Data:** Lichess `standard-chess-games` 2013-01 auto-parquet shard; 2,600 games ≥24
  plies; replay oracle = python-chess 1.10.0.
- **Bank design (declared, balance-only — no probe peeking):** marker-ply scan over
  {16, 24, 32, 40} on the piece-on-square family (occ/w/b/P/wP/bP × 64 = 384 candidates):
  39 / 38 / 31 / 39 balanced axes → **ply 16 chosen, 39 balanced axes**, 2,585 instances,
  capped at the 40 most-balanced. (Occupancy-only peaked at 20 — the piece-on-square family
  was needed just to clear the bank-size gate.)
- **Witness pairs (real transpositions, UCI):** 8/39 axes covered within the 240 s search
  budget — genuine legal-permutation pairs from real games, e.g. axis `P.d4`:
  `e2e4 e7e5 g1f3 b8c6 d2d4 e5d4 f3d4 c6d4` vs `d2d4 b8c6 e2e4 c6d4 g1f3 e7e5 f3d4 e5d4`
  — same 8-move bag, pawn-on-d4 differs. The state IS order-dependent in principle.
- **The kill:** the probe suite reads **every** axis far above the 0.60 ceiling, in all three
  conditions (SAN / sanitized / UCI): probe-max 0.94–0.996. The UCI condition — built to
  *remove* the SAN notation leak — is the most readable of all, because a UCI token carries
  its destination square: at any early marker, "piece on X" ≈ "some move lands on X, none
  leaves" — nearly a bag function on the natural distribution. ECO metadata alone reads many
  axes at 0.54–0.76 (the opening determines early structure).

## Reading — the ambiguity-slice diagnosis (the receipt's real content)

The two failures have one shape, and it is the H2 lesson at bank scale:

> **State is surface-undecodable only on the count-ambiguous slice, and in natural
> distributions that slice is thin.** H2's crossover was measured *on* the hard slice
> (counts collapse 0.770 there, GPT-2 holds 0.931) — but on ALL positions counts win
> (0.965). V3-0's gate runs on the whole natural distribution, where redundancy makes the
> bag nearly sufficient even for in-principle order-dependent state. The witness pairs
> (which exist!) certify the *in-principle* half; the probe ceiling measures the
> *distributional* half; natural data fails the second while passing the first.

So the intersection boundary is now measured at **five** families: count-parity
(undecodable/not computed), agreement (computed/decodable), FewRel relations
(high-dim/decodable), **code variable-state (real corpus too literal to even form the
bank)**, **chess board-state (high-dim, order-dependent in principle, bag-dominated in
distribution)**.

## Disposition & onward options (owner's call — none taken)

1. **V3-0b — ambiguity-slice bank redesign (the constructive read).** Define each axis ON
   its count-ambiguous slice (chess: squares touched ≥2× in the prefix; code: slots with ≥2
   updates) — the honest generalization of H2's hard slice, and arguably what the R1 toy did
   by construction (its latents were *made* ambiguous). Requires a spec amendment + new
   prereg: it changes the bank's meaning (per-axis conditional distributions, and `d_dec ≥ 20`
   must be re-defined at co-measurable positions). Not run here.
2. **Bank R1 and freeze** — the R2 boundary is now empirically dense: five families, three
   levels (toy/model/data), one theory anchor (`SurfaceBag`); the honest statement is that
   the natural-distribution intersection is thin *everywhere we have looked*.
3. Narrative entity-state stays parked per §5.

Cross-refs: `H1_V3_STATEBANK_SCOPE.md` (the prereg this runs), `R2_INTERSECTION_HYPOTHESES.md`
(H1 row; H2/H5 results), `R2_LM_DATA_AUDIT_RECEIPT.md` (the relation-extraction stop),
`PROMOTE_GATE.md` (unchanged).
