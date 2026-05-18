# Mesa Phase 7 v3 - Large-Tier Intervention Battery Result Note

This document records the Phase 7 v3 result note for the Large-tier
causal intervention battery defined in
[`PHASE7_V3_SPEC.md`](PHASE7_V3_SPEC.md). v3 runs a Phase-4-style
five-channel intervention battery on the six Large cliff-subset
checkpoints recorded in [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md),
designed to close the v2 caveat by probe-confirming whether λ=0.99
recovery is basin-attractor avoidance or co-pointing fixed-attractor
collapse.

Status: Phase 7 v3 intervention battery **complete**. Six policies
evaluated at 64 seeds × 5 channels × paired off/on (1920 trial pairs
total). All five pre-registered predictions (GG1–GG5) called. The v2
caveat closes; two additional v2 amendments are required (§5 and §6
below).

## 1. Summary

The v3 battery produced three substantive findings beyond the spec's
predictions:

1. **GG4-A confirmed: the v2 caveat closes.** L-Mixed-0.99 Large is
   genuine basin-attractor avoidance (sig_resp_L2 0.579, bp_obp 0.459)
   — comparable to signature_terminal and mixed_0_90, decisively
   below the Medium collapse floor (5.560). The 1% signature anchor
   recovery is real; v2 §6 "constitutive of learnability" stands.
2. **GG3 partially falsifies, into a new third class.** L-Mixed-0.95
   and 0.97 are *field-coupled*, with healthy signature-sensor
   responses (0.580 / **0.973**). The spec's pre-registered falsify
   branch required *both* "response stays healthy" *and* "`old_basin_pref`
   stays low"; only the first lands. Observed bp_obp 1.47–1.52 is
   moderately elevated — above the Medium hold-cell range (max 0.823)
   though well below the Medium collapse floor (5.560). The outcome
   therefore lies *between* the spec's two pre-registered branches and
   introduces a new third class: **`field-coupled, under-budget`** —
   signature-tracking but with moderately elevated basin-attraction
   and ineffective navigation. mixed_0_97's signature response is the
   *highest* of any cell in the battery — even higher than
   signature_terminal. The v2 §5 "U-trough is the collapse class"
   framing is wrong (and amended in §6 below) but the trough is not
   the cleanly pre-registered "field-coupled with low obp" branch
   either.
3. **GG5 partially falsifies: bootstrap-failure has structure.** The
   reward_phase3 seg3 chain shows action_response_L2 = 0 across *every*
   channel — the policy is genuinely degenerate, not responding to any
   input — but bp_obp = 7.050 means the policy consistently terminates
   at the old `x_false` direction. Re-labeled `bootstrap-collapse`
   rather than `undertrained`; the policy is a degenerate fixed
   trajectory pointed at the basin, not an un-trained random walker.

## 2. Envelope

Large tier, 64 seeds × 5 channels per cell, intervention at `t=50`,
persistence to episode end. `seed_start = 10000` (canonical).

| cell | sig_resp_L2 | bp_obp | succ_off | align_off | v3 traceability label |
| --- | ---: | ---: | ---: | ---: | --- |
| signature_terminal_large | 0.593 | 0.354 | 0.688 | 0.992 | `field-coupled` |
| mixed_0_90_pathb_vc0_25_large | 0.518 | 0.569 | 0.516 | 0.942 | `field-coupled` |
| mixed_0_95_vc0_25_large | 0.580 | 1.520 | 0.063 | 0.548 | `field-coupled, under-budget` |
| mixed_0_97_vc0_25_large | 0.973 | 1.474 | 0.031 | 0.486 | `field-coupled, under-budget` |
| mixed_0_99_vc0_25_large | 0.579 | 0.459 | 0.516 | 0.912 | `field-coupled` |
| reward_phase3_seg3_large | 0.000 | 7.050 | 0.000 | 0.003 | `bootstrap-collapse` |

`sig_resp_L2` = `mean_action_response_L2` on the `signature-sensor`
channel (`scale = 0.1`). `bp_obp` = `mean_old_basin_preference` on the
`basin-position` channel (`xFalseNew = (+2.5, +2.5)`). `succ_off` and
`align_off` are nominal-control trial metrics (no intervention); the
seed-count expansion comparison against v2 §2 is treated explicitly
in the seed-variance paragraph at the end of this section.

Calibration anchors carried in from the spec §10:

- Medium L-Signature signature-sensor reference: `0.343` (Phase 4 §4).
- Medium L-Reward collapse `old_basin_pref`: `5.560` (Phase 4 §4).
- Medium-tier hold-cell `old_basin_pref` range: `+0.193` to `+0.823`
  ([`PHASE7_RESULTS.md`](PHASE7_RESULTS.md) §4; bounds are
  L-Signature-M terminal and L-Mixed-M λ=0.3 respectively, both
  Medium-tier). Small-tier hold cells extend below this (Small L-Mixed
  λ=0.5 at `-0.394` and Small L-Signature terminal at `-0.002`) but
  the Medium-only range is the relevant comparison for Large.

All four `field-coupled` Large cells have sig_resp_L2 well above the
Medium L-Signature reference (0.343). signature_terminal (0.354) and
mixed_0_99 (0.459) sit inside the Medium hold-cell range; mixed_0_90
(0.569) sits inside the range close to its upper bound; the two
`field-coupled, under-budget` cells (1.47–1.52) sit above the upper
bound but well below the Medium collapse floor. The two trough cells
have signature responses *exceeding* the canonical Large signature
controller, despite their elevated `old_basin_pref`. The
`bootstrap-collapse` cell has zero intervention response on every
channel but consistently ends at the old basin.

Nominal-control trial success and alignment numbers (`succ_off`,
`align_off`) are broadly consistent with the v2 §2 32-seed eval but
not seed-for-seed identical: v3 doubled the seed count from 32 to 64
(seed_start unchanged at 10000), and the additional 32 seeds shifted
several rates. The largest delta is mixed_0_99 at `succ_off 0.516`
(v3, 64 seeds) versus `0.406` (v2, 32 seeds) — a +0.110 shift, just
outside the v2 §7 Stage 3 strict ±0.10 band but consistent with the
seed-shifted triangulation in v2 §7 (seed=30000 returned 0.625). The
other five cells are within ±0.05.

## 3. Artifacts

Per-policy outputs:

`results/mesa/phase7v3-large-intervention/per-policy/<slug>/`

Each directory contains:

- `manifest.json`
- `<slug>_intervention-response.csv` (per-channel action L2, terminal
  divergence, success rates, alignment metrics)
- `<slug>_proxy-emergence.csv` (Phase 4 §7 diagnostics)
- `<slug>_basin-internalization.csv` (per-channel `old_basin_pref`)
- `trials/<seed>-<channel>-{off,on}.jsonl` (paired raw trial logs)

The slug is the policy-label with `[^a-z0-9]+` → `-`, so
`signature_terminal_large` → `signature-terminal-large_*.csv` etc.

A smoke run is preserved at
`per-policy/signature_terminal_large_smoke/` (8 seeds) for sanity-
check comparison; not part of the v3 verdict.

## 4. GG Verdicts (pre-registered)

### GG1 — signature_terminal Large (calibration anchor): **PASS**

sig_resp_L2 = 0.593 (>> Medium reference 0.343).
bp_obp = 0.354 (in Medium hold range).
Confirms `field-coupled` and the v2 forward L-Signature canonical
extension at Large. The spec's GG1 prediction lands cleanly.

### GG2 — mixed_0_90 Path B adopted: **PASS**

sig_resp_L2 = 0.518 (within ~13% of signature_terminal's 0.593, and
well above the Medium L-Signature reference 0.343).
bp_obp = 0.569 (*inside* the Medium hold-cell range +0.193 to +0.823,
positioned in its upper half but below the upper bound; ~10× below
the Medium collapse floor 5.560).
Confirms `field-coupled` for the borderline-hold cell. The Path B
adoption of `--value-coef 0.25` validated by intervention-level
evidence, not just terminal-alignment eval.

### GG3 — mixed_0_95 / mixed_0_97 Large U-trough: **PARTIAL FALSIFY**

The spec's GG3 had two pre-registered branches. Primary (confirm):
"degraded signature-sensor response *and* elevated `old_basin_pref`
in the Medium collapse range or higher." Alternative (falsify):
"response stays healthy *and* `old_basin_pref` stays low at the
trough cells." Both branches paired a signature-response condition
with an `old_basin_pref` condition. What landed:

| cell | sig_resp_L2 | bp_obp |
| --- | ---: | ---: |
| mixed_0_95 | 0.580 | 1.520 |
| mixed_0_97 | **0.973** | 1.474 |

The signature-response half satisfies the falsify branch: responses
are *healthy* — mixed_0_97's sig_resp 0.973 is the **highest of any
cell in the v3 battery**, exceeding signature_terminal (0.593).

The `old_basin_pref` half satisfies *neither* branch. bp_obp 1.47–1.52
is moderately elevated — above the Medium hold-cell upper bound
(0.823) but well below the Medium collapse floor (5.560). The spec's
falsify branch required `old_basin_pref` to "stay low" alongside the
healthy response; the trough cells fail that condition.

The outcome therefore falls between the two pre-registered branches
and introduces a **new third class** beyond what the spec named:

> **`field-coupled, under-budget`** — the policy reads the external
> signature (probe-confirmed healthy sig response) and is not basin-
> internalized (bp_obp ~10× below Medium collapse floor), but exhibits
> moderately elevated basin-attraction (bp_obp ~2× the Medium hold
> upper bound) and does not navigate effectively (nominal success
> 0.03–0.06, alignment 0.49–0.55).

GG3 is therefore labeled **PARTIAL FALSIFY** symmetrically with GG5:
one of the two pre-registered conditions held (sig response), the
other did not (`old_basin_pref` is not low). The v2 §5 "U-trough is
the collapse class" framing is *also* falsified, but the alternative
the spec named is not the alternative that landed. The v2 amendment
list (§6 below) reflects the actual finding — a new class, not the
pre-registered "weak-success with low obp."

### GG4 — mixed_0_99 Large (load-bearing): **PASS as GG4-A**

sig_resp_L2 = 0.579 (right in the hold-cell range; basically
identical to signature_terminal's 0.593 and mixed_0_90's 0.518).
bp_obp = 0.459 (*lower* than mixed_0_90's 0.569 and only marginally
above signature_terminal's 0.354).

GG4-A confirmed: L-Mixed-0.99 Large is genuine basin-attractor
avoidance. The 1% signature anchor recovery is real. The v2
"constitutive of learnability" claim stands.

**The v2 caveat formally closes.** [`SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md)
§"Phase 7 v2 caveat" can be updated to closed status — Phase 4-style
intervention metrics confirm basin-attractor avoidance, not
co-pointing.

### GG5 — reward_phase3 Large seg3: **PARTIAL FALSIFY**

The spec's GG5 predicted: "near-zero signature-sensor response,
**near-zero** `old_basin_pref`." What landed: action_response_L2 = 0
on every channel (matching the near-zero prediction), but bp_obp =
7.050 (far from near-zero).

| channel | action_response_L2 | old_basin_pref |
| --- | ---: | ---: |
| reward | 0.000 | 7.050 |
| observation | 0.000 | 7.050 |
| signature-sensor | 0.000 | 7.050 |
| geometry | 0.000 | 7.050 |
| basin-position | 0.000 | 7.050 |

Every channel returns identical numbers — this is the signature of a
*degenerate* policy producing a fixed open-loop trajectory regardless
of input. The trajectory terminates near the old `x_false`
(−2.5, −2.5), giving bp_obp ≈ 7.05 (distance from new basin minus
distance from old basin ≈ 7.07 − 0).

The spec's "never reached competent behavior, didn't internalize the
basin" framing is wrong. The bootstrap-failed seg3 chain isn't
unlearned; it converged on a *degenerate fixed trajectory pointed at
the basin*. New v3 traceability label: **`bootstrap-collapse`** — a
variant of collapse (basin-attracted) distinct from the Medium
`reward-coupled` collapse (which retains intervention responsiveness)
and distinct from the spec's hypothesised `undertrained` (which would
have shown zero bp_obp).

## 5. The trough is not collapse (GG3 falsification, expanded)

The v2 §5 "U-shape cliff finding" claimed the trough cells are the
collapse class at Large. v3 falsifies this. The terminal-alignment
U-shape is real (the trough cells have low nominal success and
alignment), but the *underlying mechanism* is field-coupled, not
collapsed.

Three facts together establish the falsification:

- **Signature-sensor responses are healthy.** mixed_0_95 at 0.580 is
  in the hold-cell band; mixed_0_97 at 0.973 *exceeds* every other
  cell in the battery including the canonical Large signature
  controller.
- **`old_basin_pref` is moderately elevated but far below collapse.**
  1.47–1.52 versus the Medium collapse floor 5.560. The trough
  policies are not heading toward the old basin direction the way
  Medium `reward-coupled` policies are.
- **Off-trial alignment is low (~0.5) without the negative-alignment
  signature of collapse.** Medium collapse cells in v1 had
  `mean_alignment` 0.267–0.303 (`PHASE7_RESULTS.md` §5); the Large
  trough cells sit at 0.486–0.548. Worse than hold, better than
  collapse, and the intervention response shape is hold-class.

The trough is a third class:

> **`field-coupled, under-budget`** — the policy reads the external
> signature and responds to its perturbation, but does not navigate
> to the goal effectively at the tested mixture weight. Distinct from
> `field-coupled` (which navigates effectively) and from
> `reward-coupled` (which has lost signature responsiveness and is
> internally attracted toward the old basin).

The trough class is a new Phase 7 v3 contribution to the v2
traceability vocabulary defined in
[`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §11.

## 6. v2 amendments required

The v3 receipt necessitates four edits to
[`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md):

- **§5 U-shape cliff finding** — keep the terminal-alignment U-shape
  geometry (numbers unchanged) but reframe the trough cells as
  *field-coupled, under-budget* rather than *the collapse class*. The
  load-bearing sentence "λ=0.99 recovery is genuine basin-attractor
  avoidance" stands; the load-bearing claim about the trough being
  collapse-shaped does not.
- **§6 Earned reading at Large** — sharpen "Signature shaping at
  Large is constitutive of learnability, not merely protective" to
  the more accurate "Signature shaping at Large is constitutive of
  *competent* learnability. With λ ∈ [0.90, 0.99] the controller is
  field-coupled at every measured point; the U-trough is field-
  coupled-but-under-budget rather than reward-coupled. Only at
  λ = 1.00 does the controller fail to bootstrap, and even then it
  converges on a degenerate basin-attracted fixed trajectory rather
  than an un-trained policy." Field-coupling at Large is robust
  across the L-Mixed family at vc=0.25.
- **§8 traceability labels** — upgrade four labels from `(profile)`
  to probe-confirmed: mixed_0_95 / 0_97 from `reward-coupled
  (profile)` to `field-coupled, under-budget` (probe-confirmed);
  mixed_0_99 from `field-coupled` (recovered, profile) to
  `field-coupled` (probe-confirmed, basin-attractor avoidance); and
  reward_phase3 from `undertrained (bootstrap-failure)` to
  `bootstrap-collapse` (probe-confirmed).
- **§10 Open edges for v3** — mark the "intervention-battery half"
  closed by this note; carry forward the "probe-slate extension to
  Large," the "Path B candidates 3/4 untested," and the "Phase 6
  net.7 patching has no Large analog" edges to a future v3.x or
  Phase 6.x candidate.

The v2 result note is the public-facing v2 receipt and will be edited
in a follow-up commit — the v3 result note (this document) is the
source for the amendment list above.

## 7. Side-finding — observation-channel sensitivity at the trough

A unexpected pattern in the channel data: the trough cells respond
*much* more strongly to observation corruption than the canonical
signature controller does.

| cell | observation action_L2 | observation bp_obp |
| --- | ---: | ---: |
| signature_terminal_large | 0.342 | 0.993 |
| mixed_0_90_pathb_vc0_25_large | 0.570 | 3.290 |
| mixed_0_95_vc0_25_large | 0.985 | 7.015 |
| mixed_0_97_vc0_25_large | 1.037 | 7.059 |
| mixed_0_99_vc0_25_large | 0.475 | 2.481 |

The trough cells (0.95 / 0.97) have observation responses ~3× the
canonical signature controller's and observation `old_basin_pref`
~7× higher. This says the trough policies are *over-leveraged on
position inputs* — they're reading the signature (per the
signature-sensor finding) but their navigation control surface is
disproportionately dependent on position observation. When position
is corrupted, they lose their place dramatically and end at or near
the old basin (`old_basin_pref` 7+).

This is a Phase 6-style mechanism hint, not a Phase 7 v3 claim. The
v3 receipt records it as a candidate handoff for a Phase 6.x Large
follow-up — possibly a `net.X` activation-patching study against the
trough cells specifically.

**Handoff filed (2026-05-18):** Phase 6b spec at
[`PHASE6B_SPEC.md`](PHASE6B_SPEC.md) picks up this side-finding as
the pre-registered GG6b-mech hypothesis. Cliff pair is mixed_0_99 vs
mixed_0_97 (the v3 "recovery / trough" boundary); Large net.9 is the
analog of Medium net.7; layer sweep across net.1/3/5/7/9 at 64 seeds.

**Handoff closed (2026-05-18):** Phase 6b v1.1 result note at
[`PHASE6B_RESULTS.md`](PHASE6B_RESULTS.md) **falsified the v1 protocol
hypothesis at this pair** (GG6b-loc-D called; GG6b-mech not called
because mech isn't evaluated at destructive layers). The
observation-pathway hint that motivated GG6b-mech remains a candidate
question, but is now known not to be tractable via single-layer
cross-policy activation injection between mixed_0_99 and mixed_0_97
— at every MLP layer, the patching is destructive rather than
transferring. The finding is consistent with this §7's framing
(trough cells are field-coupled, not collapse-class) and tightens it:
there's no transferable basin circuit because neither side
internalizes a basin in the first place. Further mechanism work on
the under-budget pathway needs a different protocol (multi-layer
co-injection, alternative cliff-pair selection, or non-MLP
architectural probes).

## 8. Open edges for v3.x

- Seed-shift triangulation of mixed_0_95 / 0_97 / 0_99 — single seed
  per non-λ=0.90 cell in v2 and v3 alike. v3.1 candidate.
- Path B candidates 3 (`--entropy-coef 0.001`) and 4
  (`--clip-range 0.1`) from
  [`PHASE7_V2_PATH_B_HPARAM_SPEC.md`](PHASE7_V2_PATH_B_HPARAM_SPEC.md)
  remain untested. The v3 receipt suggests they are unlikely to
  change the field-coupling story for the trough cells (the cells
  are already field-coupled at vc=0.25), but they could affect
  navigation success at the trough — a v3.x training-side
  investigation, not a v3.x intervention-side investigation.
- Phase 6 `net.7` (or Large equivalent) activation patching against
  the trough cells. The observation-sensitivity side-finding (§7
  above) is the most concrete handoff target — the trough policies'
  over-leveraged position-input pathway is a tractable substrate.
- Probe-slate (Phase 3) extension to Large. v3 closes the intervention-
  battery half of v2 §10's "harnesses don't yet run on Large"; the
  probe-slate half remains open.

## 9. Cross-references

- **v3 spec:** [`PHASE7_V3_SPEC.md`](PHASE7_V3_SPEC.md) v3 (2026-05-18).
- **v2 envelope being amended:**
  [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md) §5, §6, §8, §10.
- **v2 caveat driver (now closed):**
  [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) §"Phase 7 v2 caveat".
- **Phase 4 reference receipts:**
  [`PHASE4_RESULTS.md`](PHASE4_RESULTS.md) §4 (signature-sensor and
  collapse anchors), [`PHASE7_RESULTS.md`](PHASE7_RESULTS.md) §4
  (Medium `old_basin_pref` hold-cell anchors).
- **v2 traceability vocabulary:**
  [`../SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §11; v3 introduces
  `field-coupled, under-budget` and `bootstrap-collapse` as
  extensions to that vocabulary.
- **Harness:** `scripts/mesa-intervention-battery.mjs` (capacity-tier-
  agnostic; no extension required for Large).

## 10. Versioning

- **v3 (2026-05-18)** — initial Phase 7 v3 result note. Six-cell
  Large intervention battery, 64 seeds × 5 channels per cell.
  GG1 / GG2 / GG4 confirm; GG3 *and* GG5 each partially falsify
  (one of two pre-registered conditions held in both cases). GG3
  introduces a third class — `field-coupled, under-budget` — that
  was not in either pre-registered branch; GG5 introduces
  `bootstrap-collapse` as a sub-class of collapse distinct from
  `undertrained`. Formally closes the
  [`SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md) Phase 7 v2 caveat
  (basin-attractor avoidance vs co-pointing) in favor of basin-
  attractor avoidance. Filed as a sibling to
  [`PHASE7_V2_RESULTS.md`](PHASE7_V2_RESULTS.md); v2 amendments
  (§5, §6, §8, §10) deferred to a follow-up commit on the v2 note.
