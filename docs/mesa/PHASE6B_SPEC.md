# Mesa Phase 6b - Large-Tier Activation Patching on the v3 Cliff Pair

This document is the implementation-grade spec for Phase 6b, which
extends Phase 6 v1 (Medium cliff-pair activation patching localized to
`net.7`) to a Large-tier cliff pair surfaced by
[`PHASE7_V3_RESULTS.md`](PHASE7_V3_RESULTS.md). Phase 6b is a *Large
sibling* of Phase 6 v1, not a successor; the v1 22-cell net.7 result on
Medium is unchanged.

Where this spec and [`PHASE6_SPEC.md`](PHASE6_SPEC.md) disagree, the v1
spec wins for patching semantics. Where both are silent, this spec is
authoritative for the Large extension.

## 1. Decision Lock

Six pinned calls:

- **One cliff pair, mirroring v1's design.** The primary cliff pair is
  **mixed_0_99 vs mixed_0_97** at Large with `--value-coef 0.25`,
  both at 10M env-steps, `seed_start = 10000`. mixed_0_99 is
  `field-coupled` (the "recovered" side); mixed_0_97 is
  `field-coupled, under-budget` (the v3 trough side). Adjacent in λ,
  one cell on each side of the v3 behavioral boundary — direct Phase 6
  v1 analog of the Medium 0.95 / 0.97 pair.
- **Inherit Phase 6 v1 patching semantics exactly.** Same activation-
  capture-and-inject mechanism via `register_forward_hook`, same
  per-step injection at the layer of interest, same
  `patch_success` metric, same clean / intervened conditions.
  Phase 6b is a v1 *application* at a new tier, not a v1 redesign.
- **Layer of interest is `net.9` (analog of Medium `net.7`).** Large
  actor architecture is depth=5, hidden_size=1024 (vs Medium depth=4,
  hidden_size=256). Final hidden Tanh activation index in the
  `Sequential` module list is **`net.9`** for Large (`0..9 = L,T,L,T,
  L,T,L,T,L,T`; `net.10` is the policy head Linear). All five Tanh
  layers — `net.1`, `net.3`, `net.5`, `net.7`, `net.9` — are run in
  the layer sweep, with `net.9` as the pre-registered terminal
  candidate matching v1's net.7 verdict.
- **Pre-registered mechanism hypothesis from v3 §7.** The Phase 7 v3
  side-finding (observation-channel response at trough cells ~3× the
  canonical signature controller, observation-channel `old_basin_pref`
  ~7× higher) gives Phase 6b a sharp entry hypothesis: **the
  field-coupled-under-budget pathway is over-leveraged on
  position-observation input.** If true, the patched cliff effect
  should depend on layers downstream of where position-observation is
  most strongly read. This is named explicitly in §6 (GG6b-mech) and
  is independent of the GG6b-localization prediction.
- **Single-seed slate, mirroring v1.** Both cliff-pair cells are
  single-seed at `seed_start = 10000` (carried over from v2). v3.1
  triangulation is a separate work item; Phase 6b does not own it.
  This is a tractability decision, not a methodology preference;
  Phase 6b v2 (or a v3.1 retrofit) would extend to multi-seed.
- **Harness extension is mandatory, scope is bounded.** The Phase 6 v1
  harness (`training/mesa/phase6_probes.py`) hard-codes Medium
  `CLIFF_PROTECTED` / `CLIFF_COLLAPSED` at module top-level. Phase 6b
  adds a Large pair as a parallel constant and exposes a
  `--cliff-pair {medium-v1,large-v3}` flag with `medium-v1` as the
  default. v1 code path is preserved unchanged. No other harness
  modifications are authorized in Phase 6b.

## 2. Purpose

[`PHASE7_V3_RESULTS.md`](PHASE7_V3_RESULTS.md) §5 introduced a new
traceability class — `field-coupled, under-budget` — and named the
behavioral boundary at Large between recovery (λ=0.99) and trough
(λ=0.97). The v3 receipt established that this is *not* the Medium
collapse class: the trough policies still read the signature.

The mechanistic question Phase 6b asks: **does the cliff between
field-coupled and field-coupled-under-budget at Large localize to a
specific layer, the way Medium's protected-vs-collapsed cliff localized
to `net.7`?**

Three named hypotheses (§6 below):

- **GG6b-localization** — the cliff localizes at Large net.9 with
  patch_success crossing the v1 P4 threshold (0.8) in at least one
  direction.
- **GG6b-mech** — the v3 §7 observation-sensitivity hypothesis holds
  mechanistically: the under-budget pathway is dominated by position-
  observation features at some layer downstream of input.
- **GG6b-substrate-shape** — the entangled-5D substrate finding from
  Phase 6 v3 (97.4% variance, 5 PCs, K=5 minimum) is *not* the same
  shape at Large net.9. Hidden dim is 4× larger; the compression ratio
  may differ.

The third hypothesis is pre-registered as a v2-style falsifier rather
than a confirm-or-fail dichotomy.

## 3. Target Cliff Pair

Two checkpoints (`.pt` paths relative to repo root):

| side | tier | λ | v3 traceability label | checkpoint |
| --- | --- | ---: | --- | --- |
| recovered | Large | 0.99 | `field-coupled` (probe-confirmed, basin-attractor avoidance) | `results/mesa/phase7v2-large-cliff-subset/mixed_0_99_vc0_25/checkpoints/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_99_vc0_25.pt` |
| trough | Large | 0.97 | `field-coupled, under-budget` | `results/mesa/phase7v2-large-cliff-subset/mixed_0_97_vc0_25/checkpoints/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_97_vc0_25.pt` |

Both checkpoints are verified present at spec-write time. Both were
trained with `--value-coef 0.25` at 10M env-steps.

The Phase 6 harness loads `.pt` checkpoints (not `.policy.json`) via
`policy_from_checkpoint(load_checkpoint(...))`; both target paths are
in the same on-disk format as the Medium v1 cliff pair.

## 4. Architecture Audit

`training/mesa/policy.py`:

```python
CAPACITY_CONFIGS: dict[str, PolicyConfig] = {
    "small":  PolicyConfig(tier="Small",  hidden_size=64,   depth=2),
    "medium": PolicyConfig(tier="Medium", hidden_size=256,  depth=4),
    "large":  PolicyConfig(tier="Large",  hidden_size=1024, depth=5),
}
```

Actor architecture is `nn.Sequential` of alternating `Linear` and
`Tanh` modules, with a final `Linear` projection to `act_dim=2`.
Named-module layout per tier (`policy.named_modules()` returns keys
under the `net` attribute):

| index | Medium (depth=4) | Large (depth=5) |
| ---: | --- | --- |
| net.0 | Linear (6 → 256) | Linear (6 → 1024) |
| net.1 | Tanh | Tanh |
| net.2 | Linear (256 → 256) | Linear (1024 → 1024) |
| net.3 | Tanh | Tanh |
| net.4 | Linear (256 → 256) | Linear (1024 → 1024) |
| net.5 | Tanh | Tanh |
| net.6 | Linear (256 → 256) | Linear (1024 → 1024) |
| net.7 | **Tanh** (final hidden, v1 cliff locus) | Tanh |
| net.8 | Linear (256 → 2) | Linear (1024 → 1024) |
| net.9 | — | **Tanh** (final hidden, Large analog) |
| net.10 | — | Linear (1024 → 2) |

Phase 6 v1 found the cliff localized to Medium **net.7** (final hidden
Tanh before the policy head). The Large structural analog is **net.9**
(final hidden Tanh, 1024-dim vs Medium's 256-dim).

Actor parameter counts (from the canonical `.policy.json` exports for
each tier):

| tier | hidden_size | depth | actor parameter_count |
| --- | ---: | ---: | ---: |
| Medium | 256 | 4 | 199,682 |
| Large | 1024 | 5 | 4,207,618 |

Large is **~21× Medium** in actor parameter count (not the rough 20×
implied by hidden_size² × depth alone — the depth=5 vs depth=4 step
adds another Linear layer on top of the 4× hidden-dim scaling).

## 5. Harness Extension

`training/mesa/phase6_probes.py` requires one bounded change:

- Add `CLIFF_PROTECTED_LARGE` and `CLIFF_COLLAPSED_LARGE` PolicySpec
  constants pointing to the §3 checkpoints.
- Add `--cliff-pair` to the `axis-b-smoke` argparser with choices
  `{medium-v1, large-v3}` and default `medium-v1`.
- In `run_axis_b_patch`, branch on `args.cliff_pair` to pick which
  pair to load. All downstream code (loops, metrics, CSV writing)
  remains identical.

No other changes. The harness's `register_forward_hook` mechanism is
already layer-agnostic — it accepts any name in
`dict(policy.named_modules())`.

A separate `axis-b-large-smoke` subcommand alias is not added; the
existing `axis-b-smoke` command takes the new flag.

## 6. Pre-Registered Predictions

### GG6b-localization — net.9 is the Large cliff locus

The v1 P4 threshold for patch_success was 0.8 in either direction
(protected → collapsed *or* collapsed → protected). Pre-registered:
**at least one of the two directions clears 0.8 at net.9**, and no
earlier Large layer (net.1, net.3, net.5, net.7) clears 0.8 in either
direction.

- **GG6b-loc-A (confirm)** — net.9 clears P4 in at least one
  direction, earlier layers do not. The Large cliff localizes
  structurally analogous to Medium net.7. Confirms Phase 6 v1's
  "final hidden activation as the basin-attractor locus" finding at
  a new capacity tier.
- **GG6b-loc-B (falsify — earlier locus)** — some earlier layer
  (net.1, net.3, net.5, or net.7) clears P4 and net.9 does not. The
  cliff at Large lives upstream of the final hidden — possibly closer
  to position-observation processing, consistent with the v3 §7 hint.
- **GG6b-loc-C (falsify — no locus)** — no layer clears P4 in either
  direction. The Large cliff between field-coupled and
  field-coupled-under-budget is distributed rather than localized; the
  Phase 6 v1 "single-layer locus" finding does not generalize to
  Large.

### GG6b-mech — observation-pathway dependence

From v3 §7: trough cells show ~3× the observation-channel response of
the canonical signature controller. Pre-registered: **at the layer
that clears GG6b-loc, the dominant patch direction is the one
restoring the trough's missing navigation capability — collapsed →
protected (mixed_0_97 → mixed_0_99) is the larger patch_success
direction.**

This is asymmetric with v1, where the Medium cliff was roughly
symmetric (both directions cleared P4). The v3 §7 hint predicts
asymmetry: the trough cells *have* signature responsiveness, so
restoring their navigation (C→P direction) should work better than
breaking the recovered cell's navigation (P→C direction).

- **GG6b-mech-A (confirm)** — C→P patch_success is decisively higher
  than P→C at the locus layer (mean delta ≥ 0.2). Confirms the
  observation-pathway / signature-pathway asymmetry hinted at by v3
  §7. The trough has a missing-piece that the locus layer carries.
- **GG6b-mech-B (falsify)** — symmetric patch behavior like Medium
  v1. The cliff is bidirectional, the v3 §7 observation-sensitivity
  finding does not translate to a directional patching asymmetry.

### GG6b-substrate-shape — 5D entangled subspace at Large?

Phase 6 v3 found that Medium net.7's basin-attractor circuit
compresses to **5 PCA components capturing 97.4% of variance** and
reproducing the full-layer patch effect (51× compression from 256
dims). At Large net.9, hidden_dim is 4× larger; the same compression
shape may or may not hold.

- **GG6b-shape-A (confirm — same shape)** — 5–10 PCA components at
  Large net.9 reproduce the patch effect from §3, at variance capture
  ≥ 90%. The "small handful of generators, irreducibly entangled"
  finding generalizes across capacity.
- **GG6b-shape-B (falsify — wider entanglement)** — a substantially
  larger PCA basis is required at Large net.9 (≥ 30 components to
  reach 90% variance, or ≥ 20 to reproduce the patch effect). The
  entangled-substrate generator count scales with capacity.
- **GG6b-shape-C (defer)** — net.9 fails GG6b-loc, so the
  substrate-shape question doesn't apply at this layer. Phase 6b v2
  picks up wherever the locus lives.

GG6b-shape is deferred to a second-pass after GG6b-loc is called.
Phase 6b v1 lands GG6b-loc and GG6b-mech; the PCA decomposition is a
v1.1 follow-on if (and only if) GG6b-loc finds a locus.

## 7. Acceptance Criteria

Phase 6b is complete when:

- The harness has the `--cliff-pair large-v3` flag implemented and a
  smoke (8 seeds, `--layer net.9`) lands clean.
- The full layer sweep (`--layers net.1,net.3,net.5,net.7,net.9` at
  64 seeds) has run and CSVs are written to
  `results/mesa/phase6b-large-cliff-pair/`.
- Each of GG6b-loc-{A,B,C} is called; if A or B confirms, the locus
  layer is named.
- GG6b-mech is called against the locus layer (if any).
- [`PHASE6B_RESULTS.md`](PHASE6B_RESULTS.md) is written with the
  three pre-registered verdicts and the locus / mechanism findings.

## 8. Compute Envelope

Reference point: Phase 6 v1 axis-b at Medium swept the four post-Tanh
hidden layers (`net.1`, `net.3`, `net.5`, `net.7`; the final Linear
head is excluded from the canonical v1 layer sweep). With the
harness's default `--conditions clean,intervened` and four rollouts
per inner iteration (clean_protected, clean_collapsed,
patched_protected_to_collapsed, patched_collapsed_to_protected), the
v1 step count is:

```
64 seeds × 4 layers × (2 patch directions × 2 conditions) × ~200 steps
  ≈ ~205k env-steps
```

at Medium hidden_size 256, bridge-bound, ~2–3 hours operator wall-
clock.

Phase 6b at Large differs from v1 along two axes:

- **Step count is ~1.25× v1** — 64 seeds × **5 layers** (net.1, net.3,
  net.5, net.7, net.9 — one more Tanh layer than v1's four, because
  Large is depth=5) × 2 patch directions × 2 conditions × 200 steps
  ≈ **~256k env-steps**.
- **Per-step cost is higher at Large** — actor parameter count is
  ~21× Medium (§4), so the policy forward pass is heavier even
  though the env-bridge round-trip is unchanged. The bridge round-
  trip dominates per-step cost at Medium (per Phase 7 v2 §14.4);
  whether it still dominates at Large during *inference-only*
  patching is not measured yet. The smoke step (§9) is the
  calibration measurement — wall-clock on smoke × ~32 gives the full
  estimate.

Pre-measurement estimates (to be confirmed by smoke):

- **Smoke** (8 seeds × 1 layer × 2 patch directions × 2 conditions ×
  200 steps ≈ 6.4k step-equivalents): **~5–10 minutes**.
- **Full** (~256k step-equivalents): **~2.5–4 hours** if bridge-
  bound holds at Large; longer if torch forward becomes a meaningful
  per-step share. Single-threaded, CPU-only, no GPU.

This is **larger** than a single Phase 7 v3 intervention battery
(~60–70 minutes) — Phase 6b's step count is ~5× v3's and the per-step
cost is comparable or slightly higher. Treat Phase 6b as a multi-
hour session, not a same-budget extension of v3.

## 9. Staged Commands (operator)

```powershell
# repo root: C:\Users\hughe\Dev\sundog

# SMOKE — Large cliff pair, single layer net.9, 8 seeds (~5-10 min;
# this run is the calibration measurement for the full sweep below)
python -m training.mesa.phase6_probes axis-b-smoke `
  --cliff-pair large-v3 `
  --layer net.9 `
  --seeds 8 `
  --out results/mesa/phase6b-large-cliff-pair/smoke

# FULL — Large cliff pair, layer sweep, 64 seeds (~2.5-4 hours
# pre-measurement; scale from smoke wall-clock × ~32 once smoke lands)
python -m training.mesa.phase6_probes axis-b-smoke `
  --cliff-pair large-v3 `
  --layers net.1,net.3,net.5,net.7,net.9 `
  --seeds 64 `
  --out results/mesa/phase6b-large-cliff-pair/full
```

(The subcommand keeps the `axis-b-smoke` name from Phase 6 v1 even
though the full run is not a smoke; renaming the subcommand is out of
Phase 6b scope.)

A `Show-PatchSummary` PowerShell helper will be staged in chat
alongside the operator commands, mirroring the `Show-IntervSummary`
helper from Phase 7 v3.

## 10. Outputs

```
results/mesa/phase6b-large-cliff-pair/
  smoke/
    axis-b-patch-smoke.csv
    axis-b-patch-smoke-aggregate.csv
  full/
    axis-b-patch-smoke.csv
    axis-b-patch-smoke-aggregate.csv
```

CSV column names inherit from Phase 6 v1:
`mean_patch_success_protected_to_collapsed`,
`mean_patch_success_collapsed_to_protected`,
`patch_success_ratio_of_means`, `median_patch_success`, etc.

## 11. Cross-References

- **v1 source spec:** [`PHASE6_SPEC.md`](PHASE6_SPEC.md) v1.6.
  Patching semantics, axis-b harness structure, P4 threshold,
  ratio-of-means metric inherited unchanged.
- **v1 reference receipt:** [`PHASE6_RESULTS.md`](PHASE6_RESULTS.md)
  v1 (Medium net.7 localization) and
  [`PHASE6_V31_RESULTS.md`](PHASE6_V31_RESULTS.md) (5D entangled
  subspace finding) for GG6b-shape calibration.
- **v3 driver:** [`PHASE7_V3_RESULTS.md`](PHASE7_V3_RESULTS.md) §5
  introduces the `field-coupled, under-budget` class; §7 hands off
  the observation-pathway hypothesis underlying GG6b-mech.
- **Cliff-pair lineage:**
  [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md) §2 envelope (target
  checkpoint provenance).
- **Architecture source of truth:** `training/mesa/policy.py`
  `CAPACITY_CONFIGS` constant.
- **Harness:** `training/mesa/phase6_probes.py` `axis-b-smoke`
  subcommand (extended with `--cliff-pair` flag per §5).

## 12. Non-Goals

Phase 6b does not own:

- Cross-tier patching (Medium → Large or vice versa). The cliff pair
  is within-tier.
- Held-out policy pair generalization studies analogous to Phase 6
  v3.5–v3.7 (J1/J2 work). Phase 6b is a single-pair localization
  pass; cross-policy generalization at Large is a future v2 follow-on
  once GG6b-loc is called.
- Probe-slate (Phase 3) at Large. Open edge from v3 §8; not in 6b
  scope.
- Seed-shift triangulation of the cliff-pair cells. Open edge for
  v3.1; not in 6b scope.

## 13. Versioning

- **v1 (2026-05-18, spec)** — initial Phase 6b spec. Single Large
  cliff pair (mixed_0_99 vs mixed_0_97 at vc=0.25, 10M, seed=10000),
  layer sweep across net.1/3/5/7/9 with net.9 pre-registered as the
  Medium-net.7 analog. Three named GG predictions: GG6b-loc
  (localization), GG6b-mech (observation-pathway asymmetry from v3
  §7), GG6b-shape (5D entanglement at Large net.9; deferred to v1.1).
  Compute envelope: ~1.5–2.5 hours full, ~3–5 minutes smoke.
