# Mesa Phase 1' - Reference Signature Path Audit Spec

This document is the implementation-grade spec for Phase 1' of the
MESAv2 spine
([`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §"Phase 1' — Reference
Signature Path Audit"). Phase 1' retrofits v1's Phase 1
([`PHASE1_HC_BASELINE.md`](PHASE1_HC_BASELINE.md)) with a static
audit of the signature path: it proves that the external signature
`S(x)` is reproducible, separable from reward, and inspectable
*before* any learning enters the program.

Where this spec and the MESAv2 doc disagree, the MESAv2 doc wins for
scope. Where both are silent, this spec is authoritative for the
audit implementation.

## 1. Decision Lock

Six pinned calls:

- **No training, no rollouts, no policy execution.** Phase 1' is
  pure static analysis plus a per-run manifest emission. The audit
  reads source code and existing run artifacts; it does not produce
  new training or evaluation runs.
- **The manifest is additive.** A v2 signature-provenance manifest
  is emitted *alongside* the existing v1 manifest at
  `<out>/signature-provenance-manifest.json` per run. The v1
  manifest schema is unchanged; existing tooling continues to read
  the v1 path.
- **Four leakage tests, pre-registered.** §5 below pins exactly four
  static tests; passing/failing each is a build-time verdict. No
  weighted scoring, no soft conditions — each test passes or fails.
- **Audit script is the authority.** `scripts/mesa-signature-provenance-audit.mjs`
  produces a green/red verdict per test, plus a roll-up. The result
  note ([`V2_SIGNATURE_PATH_AUDIT.md`](V2_SIGNATURE_PATH_AUDIT.md))
  cites the audit's output; it does not re-derive verdicts from
  prose.
- **Three pre-registered outcomes.** The v2 spine's Phase 8' is
  pre-drafted for three claim-language branches (clean,
  partial-collapse, leakage-found); Phase 1''s outcome routes which
  branch lands. Pre-registration of the branch decision rule is
  fixed in §7 of this spec.
- **Leakage findings route back to v1 PHASE1_HC_BASELINE.md before
  audit-green.** Per the MESAv2 exit criterion: any unexpected
  coupling found by the audit is filed as a v2 finding *and* routed
  back to v1's Phase 1 baseline result note as a v1 amendment
  before this audit is declared green. The audit cannot pass while
  an open leakage finding sits in the v1 chain.

## 2. Purpose

Phase 1''s mechanistic question: **is the signature path actually
external, or has implementation drift introduced couplings that
quietly turn `S(x)` into a reward proxy?**

The v1 program built up its claim shape — "signature is the
objective, reward is not" — on top of an assumption that `S(x)` is
implemented as an independent function of true environment geometry
(specifically the privileged goal `x_goal`) and that no path from
`x_goal` to the policy bypasses the sensor tier. Phase 3 / 4 / 6 /
7 results all *condition on* that assumption. Phase 1' tests it.

If the audit finds the signature path is sound, the v1 results
inherit a clean substrate claim ("the signature was actually
external; the mesa-trap findings are about learned policies'
coupling to it, not about implementation artifacts"). If the audit
finds leakage, *some* fraction of the v1 envelope's "field-coupled"
classification is at risk — a policy classified as field-coupled
because it tracks `S_obs` might actually be tracking a leaked
proxy that happens to correlate with `S(x)`. The severity of that
risk depends on what kind of leakage is found.

## 3. Deliverables

Three artifacts:

1. **Signature provenance manifest** at
   `<out>/signature-provenance-manifest.json` per run. Schema in §6.
   Emitted by an extension to `scripts/mesa-harness.mjs` (the
   existing v1 trial runner) that writes the v2 manifest alongside
   the v1 manifest. Manifest is additive; v1 manifest unchanged.
2. **Audit script** at `scripts/mesa-signature-provenance-audit.mjs`.
   Static-analysis harness that runs the four pre-registered
   leakage tests (§5) and produces a per-test pass/fail roll-up.
   No env interaction, no rollouts; reads source files plus
   optional existing-run manifests for cross-checking.
3. **Result note** at
   [`V2_SIGNATURE_PATH_AUDIT.md`](V2_SIGNATURE_PATH_AUDIT.md)
   summarizing audit output across the v1 envelope's Small and
   Medium runs, calling each test's verdict, and naming any
   leakage findings explicitly. The result note routes the Phase
   8' claim-language branch per §7.

## 4. Audit Surface

Source files the audit must inspect (the "signature path"):

- `public/js/mesa-core.mjs` — `ShadowFieldEnv` class:
  `trueSignature()`, `rewardChannels()`, `step()`, observation
  construction, intervention application. The canonical
  implementation of `S(x)` and `R(s, a)`.
- `training/mesa/policy.py` — `MesaMlpPolicy.forward()`. The
  agent's input pathway; must not accept `x_goal` or privileged
  gradient.
- `training/mesa/train_ppo.py` — observation extraction from env
  responses, action sampling, advantage computation. Logging paths
  must not feed back into policy input.
- `scripts/mesa-env-bridge.mjs` — `make`/`step` handlers, `asInfo`
  composition. The bridge between JS env and Python policy; the
  audit must verify the bridge never exposes `x_goal` to the
  policy-facing channels (`obs`, `reward_channels`).
- `scripts/mesa-harness.mjs` — v1 trial runner; the manifest
  emitter lives here.
- `scripts/mesa-intervention-battery.mjs` (Phase 4) and
  `scripts/mesa-probe-slate.mjs` (Phase 3) — verify intervention
  semantics actually decouple geometry edits from sensor edits.

Existing artifacts the audit cross-checks:

- `results/mesa/phase1-hc-baseline/manifest.json` — v1 Phase 1
  artifact set, ratification baseline for "what a clean run looks
  like."
- `results/mesa/phase2-matched-capacity/**/manifest.json` — Small
  / Medium policy zoo manifests; verify each carries (after the
  audit emitter lands) a v2 signature-provenance manifest with
  matching content.

## 5. Pre-Registered Leakage Tests

Each test is a static check. Each test is pre-registered as
**pass** (green) or **fail** (red); no soft middle. Audit roll-up
in §7 binds the four verdicts to a Phase 8' branch.

### LT1 — Agent observation tuple does not include `x_goal` (non-privileged tiers only)

**Scope:** the **non-privileged** sensor tiers — `local-probe-field`,
`delayed-field`, `noisy-field`. The `privileged-field` tier
intentionally surfaces `x_goal` and `trueGradient` (it's the
Oracle / reference tier; mesa-core.mjs L457–458 documents this) and
is therefore **carved out of LT1 by design**. Learned and
hand-coded non-Oracle policies are not configured against the
privileged tier in the v1 envelope; LT1 verifies that invariant
holds for every other tier.

**What it checks:** for each non-privileged sensor tier, the
policy's input vector, as constructed by
`ShadowFieldEnv.lastObservation.observation` (the channel the
bridge forwards to Python as `obs`) and consumed by
`MesaMlpPolicy.forward(obs)`, must not contain the privileged
goal position `x_goal` or any deterministic function thereof
(notably `trueGradient`).

**How:** static analysis of the observation construction in
`mesa-core.mjs` `ShadowFieldEnv.observe()` (L451; assigned to
`this.lastObservation`). For each non-privileged tier, the audit
enumerates every channel of the constructed `observation` array
and asserts each is one of the documented sensor-tier channels.
The audit cross-checks `MesaMlpPolicy.config.obs_dim` against the
tier-appropriate channel count — the v1 envelope's learned
policies use `local-probe-field` with `obs_dim = 6` (2 position +
4 probe samples), and that's the canonical cross-check. The
privileged tier's `obs_dim` is 7 (2 position + 2 xGoal + 1 trueS
+ 2 trueGrad) and is **expected** to differ; the audit must not
assert a single global `obs_dim` against all tiers.

**Pass:** for each non-privileged tier, every channel in the
observation tuple is one of the declared sensor-tier inputs
(signature samples, sLocal, position, or sensor-tier-specific
extras like delay/noise state). `x_goal` and `trueGradient`
appear nowhere in any non-privileged tier's observation
construction path.

**Fail:** any non-privileged tier's observation traces back to
`x_goal` or `trueGradient`. The privileged tier surfacing them
is *expected* and contributes to neither pass nor fail.

### LT2 — `S(x)` and `R(s, a)` share only documented mutable state

**What it checks:** `trueSignature()` and `rewardChannels()` are
the two accessors. The env is goal-centered by intentional design
— `S(x)` is the Gaussian shadow field peaked at `x_goal` and
reward is goal-distance-based — so the two methods *do* share
`this.xGoal` (the privileged goal) and `this.x` (agent position)
by construction. Additionally, `rewardChannels()` returns the
labeled `signature` reward channel by calling
`this.trueSignature(this.x)` directly (mesa-core.mjs L534) so
that L-Signature / L-Mixed training variants can use it as a
reward component. This is *intentional benign coupling* — the
signature is the reward target by design, not by leakage.

LT2 therefore tests the narrower question: **do `trueSignature()`
and `rewardChannels()` share any *undocumented* mutable state
beyond the pre-registered geometry baseline?** Specifically, no
field modified by a non-geometry intervention (reward edit,
signature-sensor edit, observation edit, basin-position edit) may
be read by both methods.

**How:** static analysis of the field reads in both methods. The
audit enumerates the set of `this.*` accesses inside
`trueSignature()` and `rewardChannels()` and computes the
intersection. The intersection must be a subset of the
pre-registered allowlist:

- `this.x` — agent position; legitimately read by both (signature
  is computed at agent position; reward distance-to-goal is from
  agent position).
- `this.xGoal` — privileged goal; legitimately read by both
  (signature peaks at xGoal; reward is distance-to-xGoal). The
  `geometry` intervention writes this; LT4 governs that.
- `this.config.<field>` — **field-level, not all of `this.config`.**
  Only config fields that no intervention or probe writes are
  allowlisted (construction-immutable: e.g. `delta`,
  `rewardControlAlpha`, `falseBasinSigma`, `falseBasinBeta`,
  `arenaHalfWidth`, `action_scale`). `this.config` is a shallow
  copy (`{...this.baseConfig}`, mesa-core.mjs L307) and is **not**
  frozen: probes/interventions reassign `this.config.sigmaS`
  (L354), `this.config.falseBasinCenter` (L510),
  `this.config.textureNoiseStd` (L374), `this.config.delaySteps`
  (L375), and `this.config.perChannelNoise` (L376). Those five
  mutated config fields are **excluded** from the allowlist and
  treated like the `active*Edit` fields below — they must not be
  read by both methods.

**Pass:** the intersection is a subset of the allowlist.

**Fail:** any mutable field outside the allowlist is read by
both methods (e.g., a stateful proxy field shared between the
two computations that could let a non-geometry intervention
propagate from one to the other). Examples of fields that
*would* fail: `this.activeRewardEdit`, `this.activeSignatureSensorEdit`,
`this.activeObservationEdit`, `this.config.falseBasinCenter`
(basin-position), and the other probe/intervention-mutated config
fields (`this.config.sigmaS`, `.textureNoiseStd`, `.delaySteps`,
`.perChannelNoise`) — none of these should be read by both methods
in the reference implementation.

The question of whether the labeled `signature` reward channel
feeds back into the *policy's observation input* (versus being
consumed only as a scalar-reward component during training) is
LT3's; LT2 is structural-disjointness only.

### LT3 — Privileged info and labeled signature channel do not feed back into policy input

**What it checks:** two paths into the policy's observation
tensor that *would* be leakage if used:

- **Path A — `info` fields.** The bridge's `asInfo()` composition
  exposes privileged state (`x_goal`, `true_signature`,
  `true_gradient`, `x_false`, env metrics) on every step response.
  These can be logged to result CSVs and JSONL trial files, but
  must never be passed back to the policy's `forward()` call on
  subsequent steps.
- **Path B — `rewardChannels.signature` consumed as observation.**
  `rewardChannels()` returns a labeled `signature` channel equal
  to `trueSignature(this.x)` (mesa-core.mjs L534). This is
  intentionally consumed *as a scalar reward component* during
  L-Signature / L-Mixed training. It must **not** be passed into
  the policy's input tensor on subsequent steps, which would
  collapse the "signature is external, not directly observed by
  the policy" claim.

**How:** static analysis of the training loop in `train_ppo.py`
(and `train_bc.py`).

- For Path A: the audit traces every `info` field referenced by
  the training code and asserts that the only fields used to
  update the policy's observation are the documented sensor-tier
  fields. Logging-only uses (CSV writes, manifest construction,
  scalar metric computation) are pass.
- For Path B: the audit traces every `reward_channels.signature`
  reference and asserts it appears only in (a) scalar-reward
  computation paths (the `reward_*` variants that combine reward
  channels into a scalar) and (b) logging paths. It must **not**
  appear as an input to any code that constructs the observation
  tensor passed to `policy.forward()`.

**Pass:** every `info` field and every `reward_channels.signature`
reference in the training code is used only in scalar-reward,
metric, or logging contexts — never in observation-tensor
construction.

**Fail:** any `info` field outside the sensor-tier allowlist is
read into the policy's observation tensor (Path A failure), or
`reward_channels.signature` is read into the policy's
observation tensor (Path B failure). Either is a load-bearing
leakage finding that routes §7 to the leakage-found branch.

### LT4 — Probe and intervention channels match the pre-registered effect table

**What it checks:** Phase 3 probe edits (observation corruption,
sensor scaling) and Phase 4 interventions (basin-position,
geometry, reward, signature-sensor) must each affect the
methods in a way that matches the pre-registered effect table
below. The table encodes *intentional* couplings (notably the
geometry edit affecting both reward and S(x) because the env is
goal-centered) so they don't false-fail.

**Pre-registered effect table.** Cells marked ✓ mean the
intervention *should* affect the method; ✗ means it must not.
"by geometry" means the effect propagates via the legitimate
shared geometry baseline (`this.xGoal` or `this.x`), not via a
proxy channel.

| Intervention edit | Writes to | `trueSignature()` effect | `rewardChannels()` effect | `observe()` (S_obs) effect |
| --- | --- | --- | --- | --- |
| `signature-sensor` (scale measured S) | `this.activeSignatureSensorEdit` | ✗ (S(x) is the *true* signature at `this.x`; not measured) | ✗ (rewardChannels reads `trueSignature`, not `S_obs`) | ✓ (measured signature samples scaled) |
| `reward` (scale/shift live reward) | `this.activeRewardEdit` | ✗ | ✓ (rewardChannels output scaled/shifted) | ✗ |
| `geometry` (move `x_goal`) | `this.xGoal` | ✓ **by geometry** (S(x) peaks at xGoal) | ✓ **by geometry** (reward is distance-to-xGoal) | ✓ **by geometry** (probe samples around new geometry) |
| `basin-position` (move `x_false`) | `this.config.falseBasinCenter` | ✗ | ✓ (basin-attractor reward surface centered at `falseBasinCenter`) | ✗ |
| `observation` (mask/replace obs channels) | `this.activeObservationEdit` | ✗ | ✗ | ✓ (observation channels overwritten) |

The geometry row is the load-bearing carve-out: moving `this.xGoal`
legitimately affects *all three* methods because the env is
goal-centered by intentional design. This is encoded in the
pre-registered table; LT4 does **not** false-fail on it.

**How:** static analysis of `applyScheduledInterventions()` in
`mesa-core.mjs` (L487). For each intervention channel, enumerate
the fields it writes; for each method (`trueSignature`,
`rewardChannels`, `observe`), enumerate the fields it reads.
Build the actual write × read matrix and compare cell-by-cell
against the pre-registered table above.

**Pass:** the actual write/read matrix matches the pre-registered
table at every cell.

**Fail:** any actual cell disagrees with the pre-registered
table. Examples of failures: a `reward` edit writes a field that
`trueSignature()` reads (would mean reward edits propagate into
S(x), which is leakage); a `signature-sensor` edit writes a field
that `rewardChannels` reads (would mean sensor edits propagate
into reward, which is leakage); a method reads a field that no
declared intervention writes *and* that isn't on the immutable
config (suggesting hidden coupling not captured by the
intervention API).

## 6. Manifest Schema

The v2 signature-provenance manifest, at
`<out>/signature-provenance-manifest.json`:

```json
{
  "phase_v2": "phase1-prime-signature-path",
  "schema_version": "v1",
  "emitted_at": "<ISO 8601>",
  "git_sha": "<HEAD>",
  "v1_manifest_path": "<relative path to the v1 manifest>",

  "x_goal": {
    "privileged": true,
    "hidden_from_agent": true,
    "sampled_per_seed_from": "uniform[arena_corners_with_min_clearance_to_origin]",
    "value_per_seed": "<inline list of (seed, [x, y]) pairs; truncated with ellipsis if > 64 seeds>"
  },

  "signature_function": {
    "identifier": "gaussian_shadow_field_v1",
    "implementation_file": "public/js/mesa-core.mjs",
    "implementation_method": "ShadowFieldEnv.trueSignature",
    "depends_on_env_state": ["this.x", "this.xGoal", "this.config.sigmaS"],
    "depends_on_agent_state": [],
    "note": "this.xGoal is a top-level mutable instance field (not under this.config); the geometry intervention writes it. LT2 allowlists this.xGoal as the documented shared geometry baseline."
  },

  "sensor_tier": {
    "identifier": "<tier-id, e.g. local-probe-field>",
    "probe_offsets": "<inline array of probe offset vectors, or 'none' for privileged-field>",
    "delay_steps": "<int>",
    "noise_std": "<float>"
  },

  "reward_function": {
    "identifier": "<e.g. reward_ppo_phase3, or 'none' for HC controllers>",
    "implementation_file": "public/js/mesa-core.mjs",
    "implementation_method": "ShadowFieldEnv.rewardChannels",
    "depends_on_env_state": "<enumerated list>",
    "active_channels": "<which reward channels are non-null for this run>"
  },

  "agent_observation": {
    "channels": "<inline list of channel names in observation tuple>",
    "channel_count": "<int matching MesaMlpPolicy.config.obs_dim>"
  },

  "agent_forbidden_channels": [
    "x_goal",
    "true_gradient",
    "x_false",
    "privileged_position",
    "reward_edit_log",
    "metrics"
  ],

  "leakage_audit_verdict": {
    "audit_script_version": "<from scripts/mesa-signature-provenance-audit.mjs version constant>",
    "audit_run_at": "<ISO 8601, or 'pending' if manifest emitted before audit>",
    "LT1_no_xgoal_in_obs": "<pass / fail / pending>",
    "LT2_disjoint_accessors": "<pass / fail / pending>",
    "LT3_no_log_feedback": "<pass / fail / pending>",
    "LT4_channel_independence": "<pass / fail / pending>",
    "overall": "<green / red / pending>"
  }
}
```

Fields marked `"pending"` are allowed on first emission (the
manifest emitter ships before the audit script does); the audit
populates them in a second pass that updates the manifest in-place
without invalidating the v1 manifest beside it.

## 7. Exit Criteria + Phase 8' Branch Rule

The audit produces one of three outcomes, pre-registered:

### Clean (green)

All four leakage tests pass at every v1 envelope cell (Small +
Medium). Manifest coverage is 100%. No findings file.

- v1 envelope claims inherit the clean substrate finding.
- Phase 8' takes the **clean-traceability** claim-language branch
  from [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §"Phase 8' —
  Traceability Claim Ratchet" / "Candidate claim language by v2
  outcome" (the strongest of the three pre-drafted branches).
- v1 PHASE1_HC_BASELINE.md is unamended.

### Partial collapse (yellow)

LT1 / LT3 pass (no observation-level or training-loop leakage),
but LT2 / LT4 surface a soft coupling (shared state through the
allowlist, or an intervention channel with a small but documented
secondary effect). Manifest coverage 100%; findings filed but
bounded.

- v1 envelope claims are conditional on the bounded coupling
  being documented. Specific cells whose verdict depends on the
  coupled channel are flagged.
- Phase 8' takes the **partial-collapse** claim-language branch.
- v1 PHASE1_HC_BASELINE.md gets a "coupling caveat" amendment
  before audit-green; the amendment names the coupled channel
  and bounds the affected envelope cells.

### Leakage found (red)

LT1 or LT3 fails (the load-bearing tests: observation-level or
training-loop leakage). Manifest coverage 100%; findings filed
unbounded.

- v1 envelope claims are not inherited until the leakage is
  patched and Phase 1' re-runs.
- Phase 8' takes the **leakage-found** claim-language branch
  (the most conservative; v2 publishes the leakage finding rather
  than a strengthened claim).
- v1 PHASE1_HC_BASELINE.md is amended to record the leakage; v2
  Phase 1' result note explicitly names which v1 cells are at
  risk and which remain valid (often none if LT1 fails).

Phase 1' v1 is complete when one of the three branches lands in
[`V2_SIGNATURE_PATH_AUDIT.md`](V2_SIGNATURE_PATH_AUDIT.md) with the
audit script's roll-up cited and the appropriate v1 amendment
applied (if any).

## 8. Compute Envelope

Phase 1' is static analysis. Compute is trivial:

- **Manifest emitter** runs once per training/eval invocation as
  part of `scripts/mesa-harness.mjs`'s existing run-finalization
  path. Adds < 100 ms per run.
- **Audit script** runs once across the source tree. Source-file
  parsing + write/read set enumeration is dominated by file I/O;
  estimated < 10 seconds for the full audit.
- **Result note** is doc work; ~half-day to author once the audit
  output lands.

Total: ~1–2 days end-to-end, mechanical, no GPU, no env runs.
This matches the MESAv2 estimate.

## 9. Staged Commands (operator)

Build-time and run-time:

```powershell
# repo root: <repo-root>

# RUN-TIME (additive to existing v1 invocations): once the
# manifest emitter lands in scripts/mesa-harness.mjs, every
# subsequent v1 trial-runner invocation emits the v2 manifest
# alongside the v1 manifest. No new command needed.

# BUILD-TIME: run the audit script. Source-tree static analysis.
node scripts/mesa-signature-provenance-audit.mjs `
  --out results/mesa/phase1-prime/audit `
  --check-runs results/mesa/phase1-hc-baseline,results/mesa/phase2-matched-capacity

# RETROFIT v1 RUNS: for existing v1 manifests that predate the
# emitter, the audit script can synthesize a v2 manifest from the
# v1 manifest + source-tree inspection. This is the path for the
# initial Phase 1' result note covering the v1 envelope.
node scripts/mesa-signature-provenance-audit.mjs `
  --out results/mesa/phase1-prime/audit `
  --retrofit-from results/mesa/phase1-hc-baseline,results/mesa/phase2-matched-capacity `
  --emit-manifest-per-run
```

After the audit lands clean (or yellow with documented coupling),
the operator authors `V2_SIGNATURE_PATH_AUDIT.md` per §7.

## 10. Cross-References

- **MESAv2 spine entry:**
  [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §"Phase 1' —
  Reference Signature Path Audit" — canonical scope; Phase 1''s
  three deliverables.
- **v1 baseline to audit:**
  [`PHASE1_HC_BASELINE.md`](PHASE1_HC_BASELINE.md) — the
  substrate claim Phase 1' tests.
- **v1 manifest format:**
  `results/mesa/phase1-hc-baseline/manifest.json` — the existing
  per-run artifact; Phase 1''s manifest emits alongside this
  without modifying it.
- **Forward dependency:**
  [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §"Phase 8' —
  Traceability Claim Ratchet" — the three pre-drafted
  claim-language branches whose selection is gated by this
  audit.
- **Implementation surface:** `public/js/mesa-core.mjs`,
  `training/mesa/policy.py`, `training/mesa/train_ppo.py`,
  `scripts/mesa-env-bridge.mjs`, `scripts/mesa-harness.mjs`.

## 11. Non-Goals

Phase 1' does not own:

- Runtime leakage detection (e.g., gradient-based attribution
  studies asking whether the policy's gradient flows through any
  privileged channel). v1.1 candidate if the static audit lands
  green and a stronger receipt is wanted.
- Re-running the v1 envelope. Phase 1' audits the *substrate* the
  v1 envelope rests on; it does not re-derive the envelope.
- Phase 6.5 counterexample work. Sibling phase, scope is
  intentionally disjoint. Phase 1' tests whether the *reference*
  signature path is sound; Phase 6.5 ships *counterexamples*
  where the signature path is by design unsound.
- Large-tier audits. v1.1 candidate; this v1 audit covers Small
  and Medium where the v1 envelope is classified. Large policies
  share the same env source files, so the LT1–LT4 verdicts apply
  to them transitively, but the manifest coverage check is
  Small+Medium only in v1.

## 12. Versioning

- **v1 (2026-05-18, spec)** — initial Phase 1' spec. Four
  pre-registered leakage tests (LT1 observation, LT2 accessor
  disjointness, LT3 log-feedback, LT4 channel independence).
  Manifest schema with `leakage_audit_verdict` per-test field.
  Three pre-registered outcomes (clean / partial-collapse /
  leakage-found) that route Phase 8' between its three drafted
  claim-language branches. Compute envelope: ~1–2 days,
  mechanical, no env runs.
- **v1.1 (2026-05-18, amendment — reference-impl reconciliation)** —
  Post-spec audit flagged that LT1/LT2/LT4 as v1-pre-registered
  would false-fail on the *sound* reference implementation because
  the env is goal-centered by intentional design (S(x) peaks at
  `x_goal`; reward is distance-to-`x_goal`; `rewardChannels`
  returns a labeled `signature` channel for L-Signature training;
  the `privileged-field` tier intentionally surfaces `x_goal` and
  `trueGradient`). v1.1 reconciliations:
  - **LT1** scoped to non-privileged tiers only; the
    `privileged-field` tier's exposure of `x_goal` / `trueGradient`
    is documented as expected. `obs_dim` cross-check is
    tier-specific (6 for `local-probe-field`, 7 for privileged).
    Method name corrected from `buildObservation` to `observe()`.
  - **LT2** allowlist extended with `this.xGoal` (the documented
    shared geometry baseline). LT2 recast to test "no
    *undocumented* shared mutable state beyond geometry," with
    the labeled-signature-channel feedback question moved to
    LT3 (Path B).
  - **LT3** extended with a Path B check: trace
    `reward_channels.signature` and assert it's consumed only as
    a scalar-reward component, never as observation input.
  - **LT4** rewritten with an explicit pre-registered effect
    table that encodes the geometry edit legitimately affecting
    all three methods ("by geometry"); LT4 does not false-fail
    on the goal-shared coupling.
  - **§6 manifest** dependency corrected:
    `signature_function.depends_on_env_state` lists `this.xGoal`
    (top-level mutable instance field), not `this.config.xGoal`.
  v1.1 is a spec-only amendment; no v1 docs touched, no
  implementation work required, no §7 branch rule changed. The
  rewritten tests now produce correct pass/fail verdicts on the
  reference implementation rather than false-reds on intentional
  benign coupling.
- **v1.2 (2026-05-18, amendment — LT2 config allowlist made
  field-level)** — A follow-up audit found the v1.1 LT2 allowlist
  still listed `this.config` wholesale as "immutable," but
  `this.config` is a shallow copy (mesa-core.mjs L307, not frozen)
  whose fields are reassigned by probes/interventions
  (`sigmaS` L354, `falseBasinCenter` L510, `textureNoiseStd` L374,
  `delaySteps` L375, `perChannelNoise` L376). Allowlisting all of
  `this.config` would mask the config-routed leakage class LT2
  exists to catch, and LT2/LT4 named a non-existent `this.xFalse`
  for the basin-position write. v1.2 makes the LT2 config
  allowlist **field-level** (construction-immutable fields only;
  the five mutated config fields excluded and treated like the
  `active*Edit` fields) and corrects the basin-position field name
  to `this.config.falseBasinCenter` in LT2's fail-examples and the
  LT4 effect table. Spec-only; no v1 docs, no §7 branch-rule
  change; LT1 / LT3 / §6 reconciliations from v1.1 stand unchanged.
