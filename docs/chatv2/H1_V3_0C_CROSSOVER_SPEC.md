# Chat-v2 H1/V3-0c — Crossover-Form Gate (spec; supersedes the absolute ceiling)

> 2026-07-01, pre-registration. **Amends `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md`:** the
> absolute surface-undecodability ceiling (≤ 0.60) is **replaced** by the crossover
> criterion, with the probe suite as the **matched baseline**. Bank construction (slices,
> floors, balance, caps, liveness) is inherited unchanged from V3-0b. **This is a gate
> redefinition adopted by owner decision after `H1_V3_0B_SLICE_ADMISSION_RECEIPT.md`, not
> a rescue of a failed run.** Non-promotional; nothing here alters `PROMOTE_GATE.md`.

## 1. Rationale

The absolute gate failed at six families, and the lane's actual positives never cleared an
absolute ceiling: H2's hard-slice count baseline was 0.770 (its result was the *crossover*,
residual 0.931 ≫ counts); R1's licensed metric was `objective_excess` (gen − twin). V3-0b
showed ambiguity slices collapse surface readability (~0.95 → ~0.65) but ambiguous fibers
are not uniform. The honest criterion is therefore **relative**: the model must read state
**better than the registered surface statistic allows, at matched evaluation**.

**Why witnesses remain required.** Under a relative gate, a bag-*determined* label with an
under-fit probe could fake a crossover (probe fails to learn the bag function from ~300
samples; the model reads it). A per-axis witness pair certifies the label is **not a bag
function**, closing that loophole. Witness coverage is now a *binding* data gate, so the
search is repaired: budget-exhausting loop + **interaction-directed** perturbations
(rotations of the same-color move at the square's change plies — V3-0b showed undirected
legal swaps mostly commute).

## 2. The gates

**V3-0c-data (CPU, this rung):** the bank admits iff **≥ 24 axes** each satisfy:
slice floor ≥ 120, balance [0.40, 0.60] on slice, **≥ 1 witness pair** (target 3; repaired
search, 900 s budget). Per-axis **surface baselines are frozen** here — the full probe
suite (LR-counts, LR-tfidf 1–2, MLP w ∈ {1,2,3}, LR-meta/ECO), same split (group by game,
seed 0), `surface_max` = max over probes — written to a manifest **before any model run**.
Liveness (`e2e4`-present ≥ 0.95) inherited.

**Model crossover gate (V3-0.5 = GPT-2 calibration, non-gate; V3-1 = 1B, gate):** per axis,
with `acc_model` = held-out accuracy of a linear readout on frozen residual-stream features
at the query position (final prefix token), same slice, same split:

- `acc_model ≥ surface_max + 0.15` (the crossover), **and**
- `acc_model ≥ acc_randinit + 0.15` (same-architecture random-init floor, same extraction —
  excludes random-feature-kernel artifacts).

Layer set {≈L/3, ≈2L/3, L}; the reported layer chosen on **validation only**. Bank-level:
**≥ 20 axes crossing** (union-form d_dec; the co-ambiguity report ships with any result —
ply-40 median 22 — so the joint-position form can be assessed). V3-0.5 anchors
expectations and validates extraction; only V3-1+ can claim the gate.

## 3. Branches

| branch | meaning |
| --- | --- |
| `H1-V3-0C-DATA-ADMIT` | ≥ 24 axes with floor + balance + witness; baselines frozen |
| `F3-V3c/witness` | repaired search still cannot certify ≥ 24 axes |
| `F3-V3c/bank` | < 24 axes reach the witness stage |
| `H1-V3-1-CROSS-ADMIT` | ≥ 20 axes cross on a 1B model (both margins) |
| `F2-V3c/carry` | model does not cross vs the frozen surface baseline |
| `F4-V3c/floor` | random-init floor explains the carry |

No rescue; margins and floors frozen as above.

## 4. Claim language (the point of the redefinition)

An eventual positive licenses: **"the model reads game state better than the registered
surface statistic allows, on the ambiguity slices of real games"** — NOT "the state is
surface-invisible," NOT world-model language. This is the H2 crossover at bank scale.
Whether union-form d_dec ≥ 20 satisfies `PROMOTE_GATE.md` R2 remains an explicit owner
decision at V3-2 prereg. All parent fences inherited.

## 5. Deliverables & frozen numbers

- **This rung:** `scripts/chatv2_h1_v3_0c_bank_freeze.py` → bank + witness panels +
  frozen baselines manifest (`results/chatv2/h1_v3/v3_0c_bank_manifest.json`) + receipt
  `docs/chatv2/H1_V3_0C_BANK_RECEIPT.md`.
- **Next (on data-admit):** `scripts/chatv2_h1_v3_gpt2_calibration.py` (V3-0.5, crossover
  form, non-gate) → `H1_V3_0_5_GPT2_CALIBRATION_RECEIPT.md`; then V3-1 per the parent scope
  (CPU-lite → GTX-1080 → H200).
- Frozen: marker ply 40; chg ≥ 2 slices; floor 120; balance [0.40,0.60]; axis cap 48;
  witness ≥ 1/axis (target 3), 900 s, directed rotations (shift ≤ 3) + optional adjacent
  swap; margins +0.15/+0.15; model bank gate ≥ 20; layers {L/3, 2L/3, L} validation-chosen;
  seed 0 splits; probe cap 1600; liveness ≥ 0.95.

Cross-refs: `H1_V3_0B_AMBIGUITY_SLICE_SPEC.md` + receipt (the negative this answers),
`H1_V3_STATEBANK_SCOPE.md` (parent), `R2_INTERSECTION_HYPOTHESES.md` (H2 crossover, H5),
`PHASE1_R1_COMPLETION.md` (`objective_excess` precedent).
