# Chat-v2 Phase 1 - Residual Body Scaling Spec

> 2026-06-01, DRAFT ONLY. This is a red/blue/audit planning note, not a
> freeze-marker, not an execution-admitted pre-registration, and not a public
> claim. Do not wire a runner, amend the lane charter, or roll this into
> `CROSS_SUBSTRATE_NOTES.md` until the active Phase 0.2 `H=16` curriculum run is
> read back and classified.

## 0. Integration posture

The chatv2 lane is mid-experiment. Phase 1 should therefore stay lightweight
until the current Phase 0.2 run resolves the immediate question:

```text
Does the grok-aware H=8 -> H=16 curriculum learn, and if it learns, does the
state-insufficient / control-sufficient split survive?
```

This draft is meant to sharpen the next move without disturbing the active
red-team lane. Promotion requires a later Amendment A that freezes:

- the exact runner or runner patch,
- the exact `H` sweep and model sizes,
- the seed set,
- all branch thresholds,
- the receipt schema,
- the staged long-run commands.

Until then, this file is a private design scaffold: useful, but not binding.

## 1. Why Phase 1 exists

ARC Phase 4 v2 falsified the "maybe the body PR was sample-limited" hypothesis.
The raw-grid ARC body plateaus near PR 11 over the whole public-training corpus.
That makes ARC the least-marginal measured body, but not the >=20 high-dimensional
computational body the dimensional regime-2 program needs.

The chatv2 lane is the live candidate because Phase 0.2 already repaired the two
major artifacts:

- passive input-decodability was killed by pair-XOR computed latents;
- raw variance PR masking was replaced by an information-basis measure, `d_dec`.

The current best positive is `H=8`, `d_dec ~= 7.2`, with:

- learned generative objective (`eval_loss` far below chance),
- compact decision control (`z1_acc ~= 0.94`),
- state-insufficient shadow (`cross_latent_leak ~= chance`),
- objective-built non-decision body (`body_carry_gen - body_carry_twin ~= 0.26`).

That is the highest-dimensional sharp resisting body in the portfolio so far,
but it does not yet clear the >=20 dimensional bar. Phase 1 asks whether the same
deconfounded mechanism scales.

## 2. Core question

Does a generatively trained transformer residual stream support a computed,
state-insufficient yet control-sufficient body whose information-basis
dimensionality clears the unchanged high-dimensional bar?

Primary test:

```text
Can `d_dec` reach >=20 while all Phase 0.2 resistance controls still pass?
```

Secondary test:

```text
Does a robust participation-ratio read, after the already-known residual-stream
outlier masking is accounted for, agree directionally with `d_dec`?
```

Important: raw variance PR is still reported, but it is not the primary gate.
Phase 0 showed that transformer residual streams can hide decodable state in
low-variance directions while a few outlier directions dominate raw PR. Treating
raw PR as the only body-dimensionality measure would reintroduce the artifact
this lane already found.

## 3. Body, shadow, and primary quantities

**Body.** The residual-stream activation at the final position of the
generative transformer on the computed-latent sequence. The Phase 0.2 runner
currently extracts the saved body used for `d_dec`, `z_recover`, leak, and
medium tests. If Phase 1 adds layerwise extraction, the primary layer must be
frozen before the verdict run; otherwise the primary body is the existing final
body for continuity.

**Shadow.** The decision readout for `z_1`, treated as the compact
control-sufficient shadow. The shadow is not allowed to include labels,
non-decision latents, query outputs, or any oracle-selected subspace.

**Primary dimensionality.** `d_dec`, the participation ratio of the stacked
per-latent linear readout directions. This is the residual-stream analogue of a
body PR in the information basis.

**Corroborating dimensionality.**

- `eff_dim_raw`: raw variance PR, reported for continuity and outlier diagnosis.
- `eff_dim_robust`: variance PR after frozen outlier handling, if implemented.
- `outlier_carries` / `latents_survive_outlier_removal`: retained as the
  medium test, not as a nuisance deletion.

**Resistance / control.**

- `z1_acc`: decision control sufficiency.
- `cross_latent_leak`: whether the `z_1` shadow reconstructs other latents.
- `body_carry_gen`: non-decision latent recoverability in the generative body.
- `body_carry_twin`: same read from the control-only twin.
- `eval_loss`: learnability guard; chance-level models are `UNLEARNED`, never
  marginal.

## 4. Phase 0.2 read-back gate

Before Phase 1 can freeze, classify the active Phase 0.2 `H=16` curriculum run:

| result | Phase 1 implication |
| --- | --- |
| `H=16` learned and SHARP | Phase 1 becomes a scaling replication and extension to `H=32`. |
| `H=16` learned but MARGINAL | Phase 1 should first diagnose the high-H upper edge before attempting `H=32`. |
| `H=16` UNLEARNED | Phase 1 is a learnability/curriculum spec, not a body-resistance verdict spec. |
| run void / artifact | Repair Phase 0.2 before opening Phase 1. |

This prevents Phase 1 from papering over an unresolved midstream result.

## 5. Proposed scaling cell

Primary sweep after Phase 0.2 read-back:

```text
H = {8, 16, 32}
```

Rationale:

- `H=8` anchors the known positive.
- `H=16` tests continuity with the tabled curriculum.
- `H=32` is the first rung that can clear `d_dec >=20` while still leaving room
  for the `d_dec >= H/2` criterion.

Candidate model family:

```text
d_model: 192 baseline, 256 escalation if H=32 is UNLEARNED
layers: current runner default unless frozen runner exposes this knob
heads: current runner default unless frozen runner exposes this knob
latent: computed pair-XOR
delta: carry forward the H=8 winning value unless Phase 0.2 read-back voids it
bits_per_channel: carry forward the H=8 winning value unless learnability fails
curriculum: H=8 -> H=16 -> H=32, with fixed positional capacity
early stop: grok-aware min_steps + patience floor
```

The current runner does not expose every architecture knob. That is acceptable
for this draft. A binding Phase 1 freeze must either use the existing knobs only
or patch the runner first and freeze the patch.

## 6. Pre-registered gates

All gates below are proposed for the promoted spec. They are not binding until
an Amendment A freezes them.

### Gate A - deconfound

Pass iff:

- input linear-probe precheck worst accuracy <= 0.60 for every verdict-bearing
  `H`;
- the latent generator is unchanged from Phase 0.2 or a new generator is
  separately frozen before seeing Phase 1 results.

Fail branch: `phase1_void_input_leakage`.

### Gate B - learnability

For each `H`, classify as learned iff:

```text
eval_loss < log(2) - 0.02
```

Any unlearned `H` is excluded from marginality claims. If `H=32` is unlearned
under the frozen budget, the verdict is `phase1_learnability_block`, not
`phase1_marginal`.

### Gate C - sharpness at a learned H

For a learned generative model at `H`, SHARP iff all hold:

```text
d_dec >= H / 2
z1_acc >= 0.70
cross_latent_leak <= 0.58
body_carry_gen >= 0.70
body_carry_gen - body_carry_twin >= 0.20
```

These carry forward Phase 0.2. They are not to be relaxed after seeing `H=16` or
`H=32`.

### Gate D - high-dimensional body bar

Phase 1 clears the portfolio high-dimensional body target iff a learned SHARP
model satisfies:

```text
d_dec >= 20
```

Preferred stronger read, if robust PR tooling is added and frozen:

```text
d_dec >= 20
eff_dim_robust >= 20
```

Raw `eff_dim_raw >=20` is not required because it is known to be masked by
residual-stream outlier directions. It is still reported.

### Gate E - seed stability

The minimal binding version should require at least 2 of 3 frozen seeds to pass
the same branch at the primary `H=32` cell. A cheaper single-seed probe may be
used only to estimate wall time and diagnose learnability; it cannot adjudicate.

## 7. Branch taxonomy

| branch | condition | interpretation |
| --- | --- | --- |
| `phase1_highdim_sharp` | at least 2/3 seeds learned, SHARP, and `d_dec >=20` at a frozen H | The >=20 information-basis residual body exists in the toy LLM substrate. |
| `phase1_scaling_sharp_below_bar` | learned SHARP persists but `d_dec <20` | Regime-2 scales, but not yet to the high-dimensional bar. |
| `phase1_learnability_block` | high H remains `UNLEARNED` under the frozen curriculum | The bottleneck is optimization/capacity, not a resistance verdict. |
| `phase1_highH_marginal` | high H is learned but fails leak/body-carry/control gates | The toy has a real upper edge; body-resistance does not survive scaling. |
| `phase1_void_input_leakage` | input-probe precheck fails | The substrate is no longer deconfounded; no verdict. |
| `phase1_void_runner_or_receipt` | manifest/seed/hash/output contract fails | No result; repair tooling first. |

## 8. Red / blue / audit split

**Blue role.** Make the mechanism earn the stronger claim:

- preserve the Phase 0.2 deconfounds;
- scale `H` and model capacity only through frozen knobs;
- report the strongest honest positive and the first real upper edge.

**Red role.** Attack the result before it becomes a claim:

- Is `d_dec` noise-rank from an unlearned model?
- Is the latent still input-linear in disguise?
- Did the curriculum leak non-decision labels or reuse a verdict-selected
  checkpoint?
- Did a stronger `delta` make the task trivial rather than learnable?
- Is the twin unfairly undertrained or architecturally disadvantaged?
- Is `H=32` selected only because it is the first rung that can clear 20?

**Audit role.** Make the receipt boring:

- exact command(s), seed list, git commit, git dirty state;
- runner SHA or patch hash;
- manifest path and all per-H records;
- wall-clock by H and by gen/twin;
- precheck table;
- verdict table;
- raw and robust dimensionality table;
- explicit classification of every unlearned H.

## 9. Receipt contract

The promoted runner should write:

```text
results/chatv2/phase1-residual-body-scaling/
  manifest.json
  records.csv
  per_seed_summary.csv
  precheck.csv
  body_dimensionality.csv
  train_curves/
  bodies/                    # optional, can be gitignored
  branch_adjudication.md
  commands.md
```

Minimum `manifest.json` fields:

```json
{
  "lane": "chatv2",
  "phase": "1",
  "verdict": "draft_unrun",
  "gitCommit": null,
  "gitDirty": null,
  "runnerSha256": null,
  "phase02Readback": null,
  "seeds": [],
  "hSweep": [8, 16, 32],
  "primaryDimensionality": "d_dec",
  "highDimBar": 20,
  "records": []
}
```

## 10. Staged commands, not to run inline

The current H=8 probe was already multi-hour on CPU. Phase 1 full runs will
exceed the repo's roughly 10-minute inline rule, so the promoted spec should
stage commands for the operator instead of running them inside an agent turn.

Pre-freeze timing probe, if needed:

```powershell
python scripts/chatv2_phase0_bodyresist.py --mode smoke --stage all --latent computed `
  --h-sweep 8 --d-model 192 --delta 0.45 --bits-per-channel 24 `
  --max-steps 400 --min-steps 200 --patience 4 `
  --out results/chatv2/_phase1_timing_probe_h8
```

Current Phase 0.2 read-back command, carried forward from the existing doc:

```powershell
python scripts/chatv2_phase0_bodyresist.py --mode full --stage all --latent computed `
  --curriculum --h-sweep 8,16 --pos-h 16 --d-model 192 --delta 0.45 `
  --bits-per-channel 24 --max-steps 6000 --min-steps 3000 --patience 10 `
  --out results/chatv2/phase02-curriculum
```

Illustrative Phase 1 command shape after promotion:

```powershell
python scripts/chatv2_phase0_bodyresist.py --mode full --stage all --latent computed `
  --curriculum --h-sweep 8,16,32 --pos-h 32 --d-model 192 --delta 0.45 `
  --bits-per-channel 24 --max-steps 8000 --min-steps 3000 --patience 12 `
  --seed 0 --out results/chatv2/phase1-residual-body-scaling/seed0
```

The binding spec should stage one command per seed, plus a merge/adjudication
command if a wrapper is added.

## 11. What Phase 1 does not claim

Phase 1 does not claim:

- anything about real hosted LLMs;
- anything about the Ask Sundog product;
- a conversational ledger result;
- a public-facing theory result;
- literal raw-variance PR sufficiency for transformer residual streams.

It only tests whether the deconfounded toy residual body can cross the
dimensional bar that ARC could not cross.

## 12. Promotion checklist

Promote this draft only after:

- Phase 0.2 `H=16` is read back and classified;
- red team signs that the proposed high-dimensional bar is not a metric swap;
- blue team signs the scaling command is feasible enough to stage;
- audit signs the receipt schema and dirty-git discipline;
- Amendment A freezes the exact runner/hash/thresholds/commands.

Recommended stance right now: keep this as a quick, enlightening sidecar. If
the `H=16` read-back is SHARP, galvanize it into the main chatv2 lane as the
natural Phase 1 scaling gate. If `H=16` is learned-but-marginal or unlearned,
rewrite Phase 1 around that fact rather than pretending the next rung is already
licensed.
