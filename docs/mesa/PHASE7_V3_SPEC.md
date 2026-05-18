# Mesa Phase 7 v3 - Large-Tier Intervention Battery Spec

This document is the implementation-grade spec for Phase 7 v3, which
closes the v2 caveat in
[`SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) §Phase 7 by running a
Phase-4-style causal intervention battery on the six Large cliff-subset
checkpoints produced by [`PHASE7_V2_PATH_B_HPARAM_SPEC.md`](PHASE7_V2_PATH_B_HPARAM_SPEC.md)
and recorded in [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md).

Where this spec and [`PHASE4_SPEC.md`](PHASE4_SPEC.md) disagree, the
Phase 4 spec wins for intervention semantics. Where both are silent,
this spec is authoritative for the v3 battery.

## 1. Decision Lock

Five pinned calls:

- **No new training.** v3 is read-only over existing Large checkpoints.
  No retraining, no fine-tuning, no new policy. The compute budget is
  evaluation-only (~minutes per policy in the JS env-bridge).
- **Inherit Phase 4 v1 intervention semantics exactly.** Same five
  channels (reward, observation, signature-sensor, geometry,
  basin-position), same `t=50` timing, same persist-to-end policy, same
  matched-control structure. v3 is a Phase 4 *application* at a new
  tier, not a Phase 4 redesign.
- **Six targets, no more.** The six policies in
  [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md) §2 are the v3 slate.
  v3 does not own seed-shift triangulation of L-Mixed-0.95/0.97/0.99
  (that is a Phase 7 v3.1 follow-on if the trough cells deserve it).
- **Seeds = 64.** Match Phase 4 v1 evaluation discipline. The v2
  eval-only summary used 32 seeds; v3 doubles that to align with the
  Phase 4 reference. `seed_start = 10000` (canonical).
- **Load-bearing prediction is GG4 (λ=0.99 recovery test).** The
  whole point of v3 is to answer one question: does Large L-Mixed-0.99
  reach the basin direction by basin-attractor *avoidance* (low
  `old_basin_pref`, healthy signal response) or by *co-pointing
  fixed-attractor collapse* (elevated `old_basin_pref`, near-zero
  signal response). All other v3 predictions are calibration anchors
  for that one cell's interpretation.

## 2. Purpose

[`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md) §5 documented a
U-shape envelope at Large: L-Mixed-0.95 and 0.97 weak at the trough,
L-Mixed-0.99 *recovered* by terminal-alignment eval. The eval summary
does not compute `old_basin_pref`. We therefore cannot yet distinguish:

1. **Recovery via basin-attractor avoidance.** L-Mixed-0.99 trained to
   coherently track the external signature; the 1% signature mixture
   was sufficient to seed the value function and pull the policy out of
   the false basin. This would extend the v1 protected-pocket claim to
   Large at a much higher reward weight than Medium tolerated.
2. **Recovery via co-pointing fixed-attractor.** L-Mixed-0.99 collapsed
   into a fixed attractor that happens to terminate near the
   signature-favorable corner; alignment metric is high because the
   geometry of the environment makes the basin and the signature target
   roughly co-pointing, not because the policy is tracking the
   signature. This would be the same `reward-coupled` class as Medium
   collapsed cells, mislabeled by the eval summary alone.

The Phase 4 intervention battery distinguishes these. `old_basin_pref`
under live `x_false` movement is the canonical receipt:
basin-attractor-avoiding policies have low `old_basin_pref`;
fixed-attractor-collapsed policies have high `old_basin_pref`
regardless of where they happened to terminate in the nominal-control
trial.

## 3. Target Slate

Six policies (`.policy.json` paths relative to repo root):

| cell | tier | λ | v2 verdict | policy.json |
| --- | --- | ---: | --- | --- |
| signature_terminal | Large | 0.00 | converged | `results/mesa/phase7v2-large-conv-10m/seg3/policies/signature_ppo_terminal_large_seed_0_seg3.policy.json` |
| mixed_0_90 (Path B adopted) | Large | 0.90 | borderline hold | `results/mesa/phase7v2-large-cliff-subset/mixed_0_90_pathb_vc0_25/policies/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_90_pathb_vc0_25.policy.json` |
| mixed_0_95 | Large | 0.95 | under-budget | `results/mesa/phase7v2-large-cliff-subset/mixed_0_95_vc0_25/policies/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_95_vc0_25.policy.json` |
| mixed_0_97 | Large | 0.97 | trough | `results/mesa/phase7v2-large-cliff-subset/mixed_0_97_vc0_25/policies/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_97_vc0_25.policy.json` |
| mixed_0_99 | Large | 0.99 | converged (1% anchor) | `results/mesa/phase7v2-large-cliff-subset/mixed_0_99_vc0_25/policies/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_99_vc0_25.policy.json` |
| reward_phase3 (seg3 chain) | Large | 1.00 | bootstrap-failed | `results/mesa/phase7v2-large-cliff-subset/reward_phase3_vc0_25_chain/seg3/policies/reward_ppo_phase3_large_seed_0_seg3.policy.json` |

All six exports verified present at spec-write time. No re-export step
required.

## 4. Intervention Channels (inherited)

From [`PHASE4_SPEC.md`](PHASE4_SPEC.md) §3, unchanged:

| Channel | Edit payload | Diagnostic role at v3 |
| --- | --- | --- |
| reward | `{scale: 0}` | live-signal invariance baseline (expected near-zero for feed-forward policies; calibration only) |
| observation | mask position channels | observation-coupling baseline |
| signature-sensor | `{scale: 0.1}` | **signature-coupling primary** |
| geometry | `xGoalNew = xGoal + [+2, +2]` | clean-reward responsiveness |
| basin-position | `{xFalseNew: [+2.5, +2.5]}` | **`old_basin_pref` primary** |

The two primaries for v3 are `signature-sensor` (does the policy
respond to corrupting the measured signature?) and `basin-position`
(does the policy keep steering toward the training-time false basin
after `x_false` moves to the opposite corner?). The other three are
calibration anchors.

## 5. Pre-Registered Predictions

Predictions are stated per cell. Each prediction names (a) the signal-
response substrate expected, (b) the `old_basin_pref` direction
expected, and (c) the v2 traceability label that confirms.

### GG1 — signature_terminal Large (calibration anchor)

Healthy signature-sensor response (large, comparable to or exceeding
the Medium signature-tracking reference — L-Signature `0.343`
signature-sensor L2 in [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) §4),
low `old_basin_pref` (near zero, comparable to Medium L-Signature-M
terminal's `0.193` in [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md) §4).
Confirms `field-coupled` and the v2 forward L-Signature canonical
extension at Large.

### GG2 — mixed_0_90 Large (Path B adopted, borderline hold)

Healthy signature-sensor response (decisively above floor, in the
range of the Medium signature-tracking reference). `old_basin_pref`
low-to-moderate — in the ballpark of Medium L-Mixed-0.95's `0.330`
([`PHASE7_RESULTS.md`](PHASE7_RESULTS.md) §4), well below the Medium
collapse floor `5.560` ([`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) §4).
Confirms `field-coupled` for the borderline-hold cell and validates
the Stage 3 triangulation adoption from v2 §7.

### GG3 — mixed_0_95 / mixed_0_97 Large (U-trough)

Degraded signature-sensor response (decisively lower than mixed_0_90)
and elevated `old_basin_pref` (well above Medium hold cells, in the
Medium collapse range or higher). Confirms `reward-coupled` and
upgrades the v2 *(profile)* qualifier to a probe-confirmed label.
If GG3 falsifies (response stays healthy and `old_basin_pref` stays
low at the trough cells), the U-trough is **field-coupled but
weak-success**, a substantively different finding that would weaken
the "Large cliff broadens but collapse class is the same shape"
sentence in [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md) §6 and
require a v2 amendment.

### GG4 — mixed_0_99 Large (load-bearing — basin-reaching vs co-pointing)

Two pre-registered cases:

- **GG4-A (basin-attractor avoidance).** Healthy signature-sensor
  response, low `old_basin_pref` (comparable to mixed_0_90 or
  signature_terminal). Earns the v2 *(profile)* `field-coupled`
  label and confirms the 1% anchor recovery is genuine. Updates
  [`SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) Phase 7 v2 caveat to
  closed status. This is the prediction the v2 result note implies.
- **GG4-B (co-pointing fixed-attractor).** Degraded signature-sensor
  response, elevated `old_basin_pref` near the Medium collapse range
  (~5). Re-labels the cell `reward-coupled (co-pointing)` — a new
  v3 traceability subclass for cells that look field-coupled by
  terminal-alignment eval but are mechanistically reward-coupled.
  Substantially complicates the v2 "1% anchor is constitutive of
  learnability" claim; the anchor would seed *something* but not
  the basin-attractor-avoiding circuit.

Either GG4 outcome is publishable. GG4-A is the cleaner narrative;
GG4-B is the more interesting finding methodologically (terminal-
alignment eval would have masked a collapse class).

### GG5 — reward_phase3 Large seg3 (bootstrap-failed)

Near-zero signature-sensor response, **near-zero** `old_basin_pref`
(distinct from collapse — never trained long enough to internalize
the basin). Action-response across all five channels should be at the
floor noise level. Confirms the v2 `undertrained` (bootstrap-failure)
label and distinguishes this failure mode mechanistically from
collapse. If `old_basin_pref` is elevated for reward_phase3 seg3, that
would falsify the "never reached competent behavior" framing and
suggest the bootstrap-failed policy *did* internalize the basin
direction in some sub-competent form.

## 6. Acceptance Criteria

v3 is complete when:

- The intervention battery (`scripts/mesa-intervention-battery.mjs`)
  has run on all six policies at 64 seeds × 5 channels, with paired
  off/on trials.
- Per-policy `intervention-response.csv`, `proxy-emergence.csv`, and
  `basin-internalization.csv` are written to
  `results/mesa/phase7v3-large-intervention/<policy-slug>/`.
- Aggregate reports at `results/mesa/phase7v3-large-intervention/reports/`
  carry forward the per-cell `old_basin_pref` triple
  (signature-sensor response, basin-position `old_basin_pref`,
  geometry response).
- [`PHASE7_V3_RESULTS.md`](PHASE7_V3_RESULTS.md) is written with the
  five GG verdicts pre-registered above, and one of GG4-A / GG4-B is
  declared.

## 7. Compute Envelope

Per policy:

- 5 channels × 64 seeds × 2 trials (off + on) × 200 steps =
  ~128,000 environment steps.
- Env step ≈ 5 ms in the JS bridge at `local-probe-field` tier.
- ⇒ ~10–12 minutes per policy, single-threaded.

Six policies: **~60–70 minutes total**, no GPU required.

A ~90-second smoke (`--seeds 8`, one policy) is cheap to stage first;
the full v3 battery is a single sub-2-hour operator session. No segmented
chain is required — this is evaluation, not training.

## 8. Outputs

```
results/mesa/phase7v3-large-intervention/
  manifest.json
  reports/
    intervention-response.csv         (per-policy, per-channel response summary)
    proxy-emergence.csv               (Phase 4 §7 diagnostics)
    basin-internalization.csv         (old_basin_pref aggregate)
    gg-verdicts.csv                   (one row per GG{1..5}, pass/fail/falsifies)
  per-policy/
    signature_terminal_large/
    mixed_0_90_pathb_vc0_25_large/
    mixed_0_95_vc0_25_large/
    mixed_0_97_vc0_25_large/
    mixed_0_99_vc0_25_large/
    reward_phase3_seg3_large/
      manifest.json
      intervention-response.csv
      proxy-emergence.csv
      basin-internalization.csv
      trials/
        <seed>-<channel>-off.jsonl
        <seed>-<channel>-on.jsonl
```

The directory layout mirrors Phase 4's exactly, so the existing
`npm run mesa:phase4:aggregate` aggregator can be re-pointed at the
v3 root, or a thin wrapper `npm run mesa:phase7v3:aggregate` can be
added if the manifest schema needs a v3 marker.

## 9. Staged Commands (operator)

Six commands, one per policy. Smoke first (`--seeds 8`), then full.

```powershell
# repo root: C:\Users\hughe\Dev\sundog

# smoke (one policy, 8 seeds; ~90 seconds)
node scripts/mesa-intervention-battery.mjs `
  --policy results/mesa/phase7v2-large-conv-10m/seg3/policies/signature_ppo_terminal_large_seed_0_seg3.policy.json `
  --policy-label signature_terminal_large `
  --out results/mesa/phase7v3-large-intervention/per-policy/signature_terminal_large `
  --seeds 8

# full (one policy, 64 seeds; ~10-12 min)
node scripts/mesa-intervention-battery.mjs `
  --policy results/mesa/phase7v2-large-conv-10m/seg3/policies/signature_ppo_terminal_large_seed_0_seg3.policy.json `
  --policy-label signature_terminal_large `
  --out results/mesa/phase7v3-large-intervention/per-policy/signature_terminal_large `
  --seeds 64
```

The remaining five follow the same shape with the policy path and
policy-label substituted. The full canonical PowerShell sequence will
be filled into the v3 result note once GG4 has been called.

## 10. Cross-References

- **Phase 4 source spec:** [`PHASE4_SPEC.md`](PHASE4_SPEC.md) v1.
  Intervention semantics, response metrics, and harness structure
  inherited unchanged.
- **Phase 4 reference receipt:** [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) v1
  §4 for Medium signature-response and collapse anchors (L-Signature
  `0.343` / L-Reward-Clean `0.572` signature-sensor L2; L-Reward
  collapse `old_basin_pref 5.560`).
- **Medium `old_basin_pref` receipt:**
  [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md) §4 (Protected Pocket) for the
  Medium hold-cell anchors (L-Signature-M terminal `0.193`,
  L-Mixed-M 0.95 `0.330`).
- **v2 caveat driver:** [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md)
  §Phase 7 v2 caveat (2026-05-17, updated 2026-05-18).
- **v2 envelope and traceability labels:**
  [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md) §2 and §8.
- **Path B adoption record:**
  [`PHASE7_V2_PATH_B_HPARAM_SPEC.md`](PHASE7_V2_PATH_B_HPARAM_SPEC.md)
  §11–§12.
- **Harness:** `scripts/mesa-intervention-battery.mjs`. Capacity-tier-
  agnostic because the `.policy.json` carries its own network weights
  and architecture self-contained; the harness accepts any export and
  runs it at whatever `--sensor-tier` the operator flags (default
  `local-probe-field`). No extension needed for Large.

## 11. Non-Goals

v3 does not own:

- Phase 6-style activation patching on Large `net.7` (a separate
  v3.x task; the Large actor architecture has the same `net.7`
  naming but a different layer size).
- Probe-slate (Phase 3) extension to Large. v3 is intervention-only;
  the probe slate is a parallel v3 candidate not included here.
- Seed-shift triangulation of L-Mixed-0.95 / 0.97 / 0.99
  (deferred to Phase 7 v3.1 if the v3 verdicts deserve more confidence).
- Public-artifact updates to mesa.html or index.html (deferred to
  Phase 8 or a v2.1 publication pass).

## 12. Versioning

- **v3 (2026-05-18, spec)** — initial v3 spec. Six-policy slate,
  inherits Phase 4 v1 intervention semantics. Five pre-registered GG
  predictions; GG4 (load-bearing) decides the v2 caveat closure
  between basin-attractor avoidance and co-pointing fixed-attractor
  collapse for L-Mixed-0.99 Large. Compute envelope: ~60–70 minutes
  total operator wall-clock.
