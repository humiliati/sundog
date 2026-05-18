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

> **v1.1 amendment (2026-05-18) — metric reframe.** The v1 Phase 6b
> spec inherited the v1 patch_success metric, which normalizes on
> the `old_basin_pref` gap between protected and collapsed cells.
> That metric is well-defined on the Medium v1 cliff pair (collapse
> gap ~5+), but on the Large v3 cliff pair the gap is ~1.0 and the
> normalized "fraction" is unbounded — patching swings of ±6
> produce `patch_success` values in the hundreds. The v1 P4
> threshold (0.8) is meaningless on this scale. The first full
> sweep at the broken v1 metric is preserved as a methodological
> receipt but **no GG verdict was called against the broken metric**
> (see §6.A below).
>
> v1.1 introduces a parallel `patch_success_align` metric
> normalized on the **terminal-alignment gap** between protected
> (mixed_0_99 align 0.912) and collapsed (mixed_0_97 align 0.486),
> ≈ 0.426. The alignment gap is the actual behavioral separation
> the v3 receipt established between recovered and trough cells.
> §6 predictions are rewritten against this v1.1 metric;
> alignment-patch_success is the canonical v1.1 reading. v1's
> basin-pref-based patch_success is kept as a transparency column
> in the CSV but is not used for verdict calls.

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

`training/mesa/phase6_probes.py` requires the following bounded
changes:

**v1 (landed 2026-05-18):**

- Added `CLIFF_PROTECTED_LARGE` and `CLIFF_COLLAPSED_LARGE` PolicySpec
  constants pointing to the §3 checkpoints.
- Added `--cliff-pair` to the `axis-b-smoke` argparser with choices
  `{medium-v1, large-v3}` and default `medium-v1`.
- In `run_axis_b_patch`, branched on `args.cliff_pair` to pick which
  pair to load. v1 medium-v1 code path preserved unchanged.
- Updated manifest emission to use the selected `protected_spec` /
  `collapsed_spec` rather than the hardcoded v1 constants, with a
  `cliff_pair` field recording the selection.

**v1.1 (queued, not yet landed):**

- Add `terminal_alignment` to `PatchRollout` (computed from rollout
  summary, mirroring `mesa-intervention-battery.mjs` semantics).
- Add `safe_patch_success_align` function paralleling
  `safe_patch_success` but normalized on alignment gap (denominator
  is `collapsed_align − protected_align` for P→C; symmetric for
  C→P). Same `nan` guard on small denominators.
- Compute both metrics per (layer, condition, seed); write both
  parallel column sets to the aggregate CSV. New columns:
  `mean_patch_success_align`, `median_patch_success_align`,
  `patch_success_align_ratio_of_means`, `mean_protected_alignment`,
  `mean_collapsed_alignment`, `mean_patched_alignment`. v1 columns
  (`mean_patch_success`, etc.) retained for transparency.
- Investigate the `clean` / `intervened` bit-identity observed in
  the §6.A first-sweep receipt before re-running. The intervention
  list is built in `run_patched_rollout` and forwarded to the bridge
  `make` request; the bridge's `make` handler is the likely
  inspection point. Until clean/intervened produces meaningful
  divergence, v1.1 GG6b verdicts use only the `clean` rows.

No other changes. The harness's `register_forward_hook` mechanism is
already layer-agnostic — it accepts any name in
`dict(policy.named_modules())`.

A separate `axis-b-large-smoke` subcommand alias is not added; the
existing `axis-b-smoke` command takes the new flag.

## 6. Pre-Registered Predictions (v1.1)

**Metric:** all GG6b predictions in v1.1 read against
`patch_success_align`, the alignment-normalized parallel metric:

```
patch_success_align (P→C direction)
  = (collapsed_align − patched_align) / (collapsed_align − protected_align)
patch_success_align (C→P direction)
  = (protected_align − patched_align) / (protected_align − collapsed_align)
```

where `*_align` is `mean_on_terminal_alignment` from the rollout
output. The denominator is the alignment gap (~0.426 for the Large
cliff pair, see preamble) — by construction the metric is bounded
[~0, 1] for patched outputs that interpolate between source and
target, > 1 if patching overshoots toward the target, < 0 if it
diverges away. P4 threshold of **0.8** carries the same semantic as
v1's basin-pref-based metric ("patching covers ≥ 80% of the gap"),
but is now applied to an outcome variable the pair actually
separates on.

### GG6b-localization (v1.1) — net.9 is the Large cliff locus

Pre-registered: **at least one of the two patch directions clears
0.8 in `patch_success_align` at net.9**, and no earlier Large layer
(net.1, net.3, net.5, net.7) clears 0.8 in either direction.

- **GG6b-loc-A (confirm)** — net.9 clears P4 in at least one
  direction, earlier layers do not. The Large recovery-vs-trough
  cliff localizes structurally analogous to Medium net.7's basin
  cliff.
- **GG6b-loc-B (falsify — earlier locus)** — some earlier layer
  (net.1, net.3, net.5, or net.7) clears P4 and net.9 does not. The
  Large cliff lives upstream of the final hidden, consistent with
  the v3 §7 position-observation pathway hint.
- **GG6b-loc-C (falsify — no locus)** — no layer clears P4 in either
  direction at `patch_success_align`. The Large recovery-vs-trough
  cliff is distributed rather than localized.
- **GG6b-loc-D (falsify — Phase 6 v1 protocol doesn't generalize)** —
  `patch_success_align` is also unbounded or chaotic at Large
  (e.g., values regularly outside [-2, 2], or median and mean
  disagree by orders of magnitude), indicating the activation-patching
  mechanism itself doesn't transfer to Large at this pair. File as a
  methodology limit, not a substrate finding.

### GG6b-mech (v1.1) — observation-pathway dependence

From v3 §7: trough cells show ~3× the observation-channel response of
the canonical signature controller. Pre-registered against
`patch_success_align`: **at the layer that clears GG6b-loc-A or
GG6b-loc-B, the dominant patch direction is the one restoring the
trough's missing navigation capability — collapsed → protected
(mixed_0_97 → mixed_0_99) is the larger `patch_success_align`
direction.**

- **GG6b-mech-A (confirm)** — C→P `patch_success_align` is decisively
  higher than P→C at the locus layer (mean delta ≥ 0.2). Confirms the
  observation-pathway / signature-pathway asymmetry hinted at by v3
  §7. The trough has a missing-piece that the locus layer carries
  and that's restorable by injecting recovered-side activations.
- **GG6b-mech-B (falsify)** — symmetric patch behavior. Both
  directions clear P4 at comparable magnitudes; the v3 §7
  observation-sensitivity finding does not translate to a directional
  patching asymmetry. The cliff is bidirectional in alignment, like
  Medium v1 was bidirectional in basin-pref.

### GG6b-substrate-shape — 5D entangled subspace at Large?

Phase 6 v3 found that Medium net.7's basin-attractor circuit
compresses to **5 PCA components capturing 97.4% of variance** and
reproducing the full-layer patch effect (51× compression from 256
dims). At Large net.9, hidden_dim is 4× larger; the same compression
shape may or may not hold. Reads against `patch_success_align`.

- **GG6b-shape-A (confirm — same shape)** — 5–10 PCA components at
  Large locus layer reproduce the patch effect from §3, at variance
  capture ≥ 90%. The "small handful of generators, irreducibly
  entangled" finding generalizes across capacity.
- **GG6b-shape-B (falsify — wider entanglement)** — a substantially
  larger PCA basis is required (≥ 30 components to reach 90%
  variance, or ≥ 20 to reproduce the patch effect). The
  entangled-substrate generator count scales with capacity.
- **GG6b-shape-C (defer)** — GG6b-loc-A/B did not find a locus, so
  the substrate-shape question doesn't apply at this layer. Phase
  6b v2 picks up wherever the locus lives.

GG6b-shape is deferred to a second-pass after GG6b-loc is called.
Phase 6b v1.1 lands GG6b-loc and GG6b-mech; the PCA decomposition is a
v1.2 follow-on if (and only if) GG6b-loc finds a locus.

## 6.A First-Sweep Receipt (v1 metric, no verdict)

For methodological transparency, the first full Phase 6b sweep
(2026-05-18, ~12 min wall-clock, 64 seeds × 5 layers, v1
basin-pref-based `patch_success`) is preserved at
`results/mesa/phase6b-large-cliff-pair/full/`. **No GG verdict was
called against this sweep** — the basin-pref-based metric is
unbounded for the Large cliff pair (basin-pref gap ~1.0, patching
swings ±6 → values in the hundreds) and the v1 P4 threshold is not
meaningful in that range.

Two findings from the first sweep ARE worth noting independent of
the metric question, because they shape v1.1 expectations:

- **`clean` and `intervened` rows are bit-identical to 16 sig figs
  across all 5 layers.** At Medium v1 these diverged because basin-
  position intervention shifted the collapsed-side policy. At Large,
  neither side of the cliff pair internalizes the basin (v3 GG4-A
  bp_obp 0.46 / 1.47), so the intervention is *expected* to have
  near-zero behavioral effect — but bit-identity is too clean,
  suggesting the intervention either isn't being applied at the env
  side or its effect is masked by something upstream of the policy.
  **This is a v1.1 prerequisite to investigate before the re-run**
  (see §5 v1.1 bullet).
- **First-sweep basin-pref `patch_success` mean values are in the
  hundreds with negative-direction medians.** Even accepting the
  metric is broken, this is preliminary evidence pointing away from
  GG6b-loc-A (net.9 as a clean Medium-net.7 analog) — patching
  net.9 swings the basin-pref outcome wildly rather than transferring
  cleanly. v1.1 will tell whether this is metric-specific noise
  (GG6b-loc-C or -D under v1's metric, possibly different under
  v1.1's) or substantive.

## 7. Acceptance Criteria (v1.1)

Phase 6b v1.1 is complete when:

- The harness `--cliff-pair large-v3` flag is implemented (v1, done)
  *and* the alignment-patch_success metric is computed and written
  to the aggregate CSV (v1.1, queued — see §5 v1.1 bullet).
- The `clean` / `intervened` bit-identity bug is investigated and
  either fixed or its non-effect explained mechanistically. v1.1
  verdicts default to `clean` rows only until the bug is resolved.
- A v1.1 smoke (8 seeds, `--layer net.9`) lands with sensible
  `patch_success_align` magnitudes (bounded, not in the hundreds).
- The v1.1 full layer sweep (`--layers net.1,net.3,net.5,net.7,net.9`
  at 64 seeds) has run and the parallel-column CSVs are written.
- Each of GG6b-loc-{A,B,C,D} is called against
  `patch_success_align`; if A or B confirms, the locus layer is
  named.
- GG6b-mech is called against the locus layer (if any).
- [`PHASE6B_RESULTS.md`](PHASE6B_RESULTS.md) is written with the
  v1.1 verdicts and a back-reference to the §6.A first-sweep
  receipt for methodological completeness.

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
- **v1 first sweep (2026-05-18, executed)** — harness extension
  landed, full sweep ran in ~12 min wall-clock (smoke pace was 15s,
  the §8 multi-hour estimate was conservative). **No GG verdict
  called.** Metric defect surfaced: v1's basin-pref-based
  `patch_success` is unbounded on the Large cliff pair (gap ~1.0 vs
  Medium's ~5.5), producing values in the hundreds. The first sweep
  is preserved at `results/mesa/phase6b-large-cliff-pair/full/` as a
  methodological receipt; see §6.A.
- **v1.1 (2026-05-18, amendment)** — metric reframe. Predictions
  rewritten against `patch_success_align`, an alignment-gap
  normalized parallel metric (denominator: collapsed_align −
  protected_align ≈ 0.426 for the Large pair). v1's basin-pref
  metric retained as a transparency column but not used for
  verdicts. New falsification branch added: GG6b-loc-D for "Phase 6
  v1 patching protocol doesn't generalize to Large in either
  metric." `clean` / `intervened` bit-identity flagged as v1.1
  prerequisite to investigate before re-run. Harness extension
  queued in §5 v1.1 bullet; re-run pending.
