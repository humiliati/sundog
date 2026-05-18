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

### LT1 — Agent observation tuple does not include `x_goal`

**What it checks:** the policy's input vector, as constructed by
`ShadowFieldEnv.lastObservation.observation` (the channel the
bridge forwards to Python as `obs`) and consumed by
`MesaMlpPolicy.forward(obs)`, must not contain the privileged
goal position `x_goal` or any deterministic function thereof
that would let the policy reconstruct it.

**How:** static analysis of the observation construction in
`mesa-core.mjs` (`buildObservation` and call sites), cross-checked
against `MesaMlpPolicy.config.obs_dim` (currently 6 — must match
the documented sensor-tier channels and not include 2 channels
for `x_goal`). The audit script enumerates every channel in the
observation tuple and asserts each is one of the documented
sensor-tier channels.

**Pass:** every channel in the observation tuple is one of the
declared sensor-tier inputs (signature samples, sLocal, position,
or sensor-tier-specific extras like delay/noise state). `x_goal`
appears nowhere in the observation construction path.

**Fail:** any channel of the observation traces back to `x_goal`
or `trueGradient` (the privileged gradient of `S` toward
`x_goal`).

### LT2 — `S(x)` and `R(s, a)` use separate accessors with no shared mutable state

**What it checks:** `trueSignature()` and `rewardChannels()` are
the two accessors. They must read from disjoint sets of
environment state, or from a documented shared read-only baseline.
Specifically: no field modified by a reward-side intervention can
be read by `trueSignature()`, and vice versa.

**How:** static analysis of the field reads in both methods. The
audit enumerates the set of `this.*` accesses inside
`trueSignature()` and `rewardChannels()` and computes their
intersection. The intersection must be empty *or* must contain
only fields whose reads are documented in a pre-registered
allowlist (currently: `this.x` (agent position) and `this.config`
(immutable env config); these are the legitimate shared baselines).

**Pass:** the intersection of mutable-state reads is empty modulo
the allowlist.

**Fail:** any mutable field is read by both methods and is not in
the allowlist (e.g., a stateful proxy field shared between the
two computations that could let a reward edit propagate into
`S(x)` or vice versa).

### LT3 — Logging does not feed back into policy input

**What it checks:** the bridge's `asInfo()` composition and the
training loop's `info` consumption. Fields in `info` that contain
privileged state (`x_goal`, `true_signature`, `true_gradient`,
`x_false`, env metrics) must not be passed back to the policy's
`forward()` call on subsequent steps. They can be logged to
result CSVs and JSONL trial files, but the agent's observation
must never include them on input.

**How:** static analysis of the training loop in `train_ppo.py`
(and `train_bc.py`). The audit traces every `info` field
referenced by the training code and asserts that the only fields
used to update the policy's observation are the documented
sensor-tier fields. Logging-only uses (CSV writes, manifest
construction) are pass.

**Pass:** every `info` field used in the policy forward path is
in the allowlist (the documented sensor-tier channels). All
privileged `info` fields are used only in logging, metric
computation, or manifest construction.

**Fail:** any `info` field outside the sensor-tier allowlist is
read into the policy's observation tensor.

### LT4 — Probe and intervention channels are independent

**What it checks:** Phase 3 probe edits (observation corruption,
sensor scaling) and Phase 4 interventions (basin-position,
geometry, reward, signature-sensor) must each alter exactly the
channel they declare, and only that channel. Specifically:

- A `signature-sensor` edit must change `S_obs` (measured
  signature) but not `S(x)` (true signature value at `this.x`)
  and not the reward channel.
- A `reward` edit must change `rewardChannels()` output but not
  `S(x)` or `S_obs`.
- A `geometry` edit (move `x_goal`) must change `S(x)` (via
  geometry change, since `S` depends on `x_goal`) but must not
  affect the reward unless reward is itself a function of `S(x)`
  in a documented way.
- A `basin-position` edit must change `x_false` and the reward
  surface but must not change `S(x)` or `S_obs`.

**How:** static analysis of `applyScheduledInterventions()` in
`mesa-core.mjs`. For each intervention channel, enumerate the
fields it writes; for each method (`trueSignature`,
`rewardChannels`, `lastObservation`), enumerate the fields it
reads. Build a write-set × read-set matrix and assert the matrix
matches the pre-registered channel-independence table.

**Pass:** the write/read matrix matches the documented
intervention semantics.

**Fail:** any intervention channel writes a field that an
"unrelated" method reads (e.g., a `reward` edit writes a field
that `trueSignature` reads), or any method reads a field that no
declared intervention writes (suggesting hidden coupling).

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
    "depends_on_env_state": ["this.x", "this.config.xGoal", "this.config.sigmaS"],
    "depends_on_agent_state": []
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
# repo root: C:\Users\hughe\Dev\sundog

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
