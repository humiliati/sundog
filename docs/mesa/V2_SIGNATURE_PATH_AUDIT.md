# Mesa Phase 1' - Reference Signature Path Audit Result Note

This document records the Phase 1' audit result for the MESAv2 spine
([`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §Phase 1'). The audit
is a static-analysis sweep of the signature-path implementation
defined in [`PHASE1_PRIME_SPEC.md`](PHASE1_PRIME_SPEC.md) v1.2;
this note records the verdict, evidence, and routing decision for
Phase 8'.

Status: Phase 1' v1.1 **complete — GREEN verdict (clean branch).**
All four pre-registered leakage tests (LT1–LT4) pass on the
reference implementation. No leakage findings; v1
[`PHASE1_HC_BASELINE.md`](PHASE1_HC_BASELINE.md) is **unamended**.
Phase 8' is routed to the **clean-traceability claim-language
branch** per [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §Phase 8'.

## 1. Summary

The Phase 1' static-analysis audit was executed against the current
source tree (git_sha `6e5533080904f719915b646193f2af7fcd3d27fc`).
All four leakage tests pass:

| Test | Verdict | Load-bearing? |
| --- | --- | --- |
| LT1 — Agent observation tuple does not include `x_goal` (non-privileged tiers only) | **PASS** | yes |
| LT2 — `S(x)` and `R(s, a)` share only documented mutable state | **PASS** | no |
| LT3 — Privileged `info` and labeled signature channel do not feed back into policy input | **PASS** | yes |
| LT4 — Probe and intervention channels match the pre-registered effect table | **PASS** | no |

**Overall: GREEN.** Per spec §7, all four passing → clean outcome →
Phase 8' routes to clean-traceability.

The substrate claim the v1 envelope rests on — "`S(x)` is externally
defined, reproducible, separable from reward, and inspectable" — is
**confirmed at the static-analysis layer** for the current source
tree. The mesa-trap findings in Phase 3 / 4 / 5 / 6 / 7 are about
learned policies' coupling to a sound external signature, not about
implementation artifacts that quietly turned `S(x)` into a reward
proxy.

## 2. Audit Execution

The audit script `scripts/mesa-signature-provenance-audit.mjs` ran
end-to-end against the source tree and the v1 envelope's existing
run manifests:

```powershell
node scripts/mesa-signature-provenance-audit.mjs `
  --out results/mesa/phase1-prime/audit `
  --retrofit-from results/mesa/phase1-hc-baseline,results/mesa/phase2-matched-capacity
```

Run metadata:

- Audit script version: **v1.0**
- Run timestamp: `2026-05-18T18:59:58.916Z`
- Git SHA: `6e5533080904f719915b646193f2af7fcd3d27fc`
- Wall-clock: ~1 second (pure static analysis)
- Retrofit manifests updated: **2** v2 manifests synthesized for the
  existing v1 envelope's Phase 1 + Phase 2 run directories.

Audit-report artifacts:

- `results/mesa/phase1-prime/audit/audit-report.json` — full
  verdict roll-up with per-test evidence.
- `results/mesa/phase1-prime/audit/audit-report.txt` — text
  summary for terminal-side reading.
- Two v2 manifests at the existing v1 run directories now carry
  `leakage_audit_verdict` blocks with `overall: green`.

## 3. LT Verdicts and Evidence

### LT1 — Agent observation tuple does not include `x_goal` (non-privileged tiers)

**Verdict: PASS.**

The audit scanned `ShadowFieldEnv.observe()` in
`public/js/mesa-core.mjs`. Findings:

- The privileged-tier branch is gated on
  `SENSOR_TIERS.PRIVILEGED_FIELD` (mesa-core.mjs L457). The carve-out
  in PHASE1_PRIME_SPEC v1.2 §5 LT1 applies cleanly — the privileged
  tier's intentional `[...this.x, ...this.xGoal, trueS, ...trueGrad]`
  exposure (L458) is the Oracle / reference path, not a learned-policy
  path.
- The non-privileged branch (the `else` clause at L459–461) builds
  `[...this.x, ...samples]`. No `this.xGoal` or `trueGrad` reference
  appears in the non-privileged construction path.
- Tiers checked: `local-probe-field`, `delayed-field`, `noisy-field`.
  All clear.

The `obs_dim = 6` from `MesaMlpPolicy.config` (training/mesa/policy.py)
matches the documented channel count for `local-probe-field` (2
position + 4 probe samples). The v1 envelope's learned policies are
configured against this tier, and the channel count is correct.

### LT2 — `S(x)` and `R(s, a)` share only documented mutable state

**Verdict: PASS.**

Static analysis of the field reads inside `trueSignature()` (L401–403)
and `rewardChannels()` (L519–578) yields:

- `trueSignature()` reads: `this.config`, `this.xGoal`.
- `rewardChannels()` reads: `this.activeRewardEdit.scale`,
  `this.activeRewardEdit.shift`, `this.config.delta`,
  `this.config.falseBasinBeta`, `this.config.falseBasinCenter`,
  `this.config.falseBasinSigma`, `this.config.rewardControlAlpha`,
  `this.trueSignature`, `this.x`, `this.xGoal`.
- Intersection: **`{this.xGoal}`** — the documented shared geometry
  baseline.

The intersection is a strict subset of the v1.2 LT2 allowlist
(`this.x`, `this.xGoal`, plus the construction-immutable config
fields). No intervention-written field (`this.activeRewardEdit`,
`this.activeSignatureSensorEdit`, `this.activeObservationEdit`,
`this.config.falseBasinCenter`, `this.config.sigmaS`,
`this.config.textureNoiseStd`, `this.config.delaySteps`,
`this.config.perChannelNoise`) appears in the intersection.

Note: `rewardChannels()` calls `this.trueSignature()` to return the
labeled `signature` reward channel (L534). This is intentional benign
coupling — the labeled channel is consumed only as a reward component
during L-Signature / L-Mixed training, not as observation input.
LT3 (Path B) checks the latter.

### LT3 — Privileged info and labeled signature channel do not feed back into policy input

**Verdict: PASS.**

Static analysis of `training/mesa/train_ppo.py`:

- **Path A** (privileged `info` fields): zero forbidden uses
  detected. The training loop reads `obs` directly from the bridge's
  `make`/`step` response into the policy; `info` fields
  (`x_goal`, `true_signature`, `true_gradient`, `x_false`,
  `metrics`) are not referenced in any line that assigns to
  `obs` / `observation` / `policy_input` or that passes into
  `policy.forward(...)` / `policy.act(...)`.
- **Path B** (labeled `signature` reward channel): zero
  observation-feedback uses detected. References to
  `reward_channels.signature` in the training loop are confined to
  scalar-reward computation paths (the `reward_*` variants that
  combine channels into a scalar) — none feed into the observation
  tensor.

Both load-bearing leakage paths are clean.

### LT4 — Probe and intervention channels match the pre-registered effect table

**Verdict: PASS.**

For each of the five intervention channels, the audit built the
actual write × read matrix from `applyScheduledInterventions()`
(L487+), the three method bodies (`trueSignature`,
`rewardChannels`, `observe`, transitively expanded to include
helpers like `sensorSamples` → `localProbeSamples` →
`measuredSignature`), and compared cell-by-cell against the v1.2
pre-registered effect table:

| Channel | Writes | trueSignature | rewardChannels | observe | Match |
| --- | --- | --- | --- | --- | --- |
| `signature-sensor` | `this.activeSignatureSensorEdit` | ✗ (expected ✗) | ✗ (expected ✗) | ✓ (expected ✓ via measuredSignature transitive read) | ✓ |
| `reward` | `this.activeRewardEdit` | ✗ | ✓ | ✗ | ✓ |
| `geometry` | `this.xGoal` | ✓ by-geometry | ✓ by-geometry | ✓ by-geometry | ✓ |
| `basin-position` | `this.config.falseBasinCenter` | ✗ | ✓ | ✗ | ✓ |
| `observation` | `this.activeObservationEdit` | ✗ | ✗ | ✓ | ✓ |

The geometry row is the load-bearing carve-out: moving `this.xGoal`
legitimately affects all three methods because the env is
goal-centered by intentional design. The pre-registered table
encodes this; LT4 does not false-fail on it.

Zero deviations.

## 4. Implementation Notes (audit-script tuning)

The audit script was iterated twice between first execution and the
final verdict, both for audit-script-implementation reasons that
did NOT change any LT pre-registration:

**Issue 1 — LT4 transitive-method-call blindness (first run).** The
initial `methodReadsAnyOf` implementation scanned only direct method
bodies. `observe()` reads `this.activeSignatureSensorEdit` via
`sensorSamples()` → `localProbeSamples()` → `measuredSignature()`
(L410), not directly. The first run reported a false negative for
`signature-sensor → observe`. **Fix:** added `expandMethodBody()` to
transitively expand method bodies via one-level `this.foo(...)`
call resolution. The audit now sees the chained reads. The fix is
implementation-side; the LT4 effect table itself is unchanged.

**Issue 2 — LT4 parent-fallback false positives (first run).** The
initial `methodReadsAnyOf` had a fallback rule: if the field was
`this.config.falseBasinCenter` and the method body matched
`this.config`, treat it as a hit. This over-matched —
`trueSignature()` reads `this.config` to pass to `signatureField`,
which uses `sigmaS` and `xGoal`, not `falseBasinCenter`. The first
run reported false positives for `basin-position → trueSignature`
and `basin-position → observe`. **Fix:** removed the parent-fallback;
use exact-path matching only. The audit now correctly distinguishes
"reads this.config wholesale" from "reads this.config.falseBasinCenter
specifically."

Both fixes are recorded in the audit-script `v1.0` source. Neither
required a spec amendment — the LT pre-registrations were correct;
the audit-script heuristics needed sharpening. The final verdict
is anchored in the post-fix audit run.

## 5. Routing Decision for Phase 8'

Per spec §7, all four LT passing routes to the **clean** outcome.
Phase 8' takes the clean-traceability claim-language branch from
[`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §Phase 8' — the
strongest of the three pre-drafted branches.

The v1 envelope's substrate claim inherits the clean finding:

- The signature path is externally defined and inspectable.
- `S(x)` is implemented as a documented function of true environment
  geometry (`this.x`, `this.xGoal`, `this.config.sigmaS`) with no
  hidden coupling to reward state.
- The agent's observation channel does not include `x_goal` or
  `trueGradient` in any tier the v1 envelope's learned policies are
  configured against.
- Probe and intervention channels affect the methods according to
  the pre-registered effect table; no intervention has out-of-band
  side effects.

[`PHASE1_HC_BASELINE.md`](PHASE1_HC_BASELINE.md) is **unamended**.
No "coupling caveat" required; no v1 cells are flagged as at-risk.

## 6. Open Items

- **clean / intervened bit-identity bug** (carried forward from
  Phase 6b v1.1 §6.A and §8). Unrelated to Phase 1' — that bug
  lives in the bridge `make` handler at the Large path. v1.1
  candidate for a separate Phase 6b v2 round.
- **Large-tier audit coverage.** Phase 1' v1 covers Small + Medium
  manifest coverage. The same LT1–LT4 verdicts apply transitively
  to Large because Large policies share the env source files, but
  a Large-specific manifest cover-check is a v1.1 candidate.
- **Runtime leakage detection** (non-goal in v1 per spec §11).
  Gradient-based attribution studies asking whether the policy's
  gradient flows through any privileged channel would tighten the
  static finding. v1.1 / v2 candidate.
- **Manifest emitter at the PPO training side.** The v1.1 emitter
  ships in `scripts/mesa-harness.mjs` (HC baseline) only. Extending
  to `training/mesa/train_ppo.py` so every PPO-trained policy's
  run directory carries a v2 manifest is mechanical and queued.

## 7. Cross-References

- **Spec source:** [`PHASE1_PRIME_SPEC.md`](PHASE1_PRIME_SPEC.md)
  v1.2 (initial spec v1, reference-impl reconciliation v1.1,
  field-level LT2 allowlist v1.2).
- **MESAv2 spine entry:**
  [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §Phase 1'.
- **v1 substrate baseline (unamended):**
  [`PHASE1_HC_BASELINE.md`](PHASE1_HC_BASELINE.md).
- **Audit script:** `scripts/mesa-signature-provenance-audit.mjs`
  v1.0.
- **Manifest emitter:** `scripts/mesa-harness.mjs`
  `buildSignatureProvenanceManifest()`.
- **Audit-report artifacts:**
  `results/mesa/phase1-prime/audit/audit-report.{json,txt}`.
- **Retrofit-updated v2 manifests:**
  `results/mesa/phase1-hc-baseline/signature-provenance-manifest.json`,
  `results/mesa/phase2-matched-capacity/signature-provenance-manifest.json`
  (and any sub-run manifests within those directories).
- **Forward dependency:**
  [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §Phase 8' —
  the clean-traceability claim-language branch is routed.
- **Sibling phase (unblocked):**
  [`SUNDOG_V_MESAV2.md`](../SUNDOG_V_MESAV2.md) §Phase 6.5 —
  Formal Separability Counterexample Pass. Per the v2 ratification
  doc, Phase 6.5 is sequenced after Phase 1' lands clean. Phase 1'
  is now clean; Phase 6.5 is the next substantive forward motion
  on the spine.

## 8. Versioning

- **v1.1 (2026-05-18)** — initial Phase 1' result note. Audit
  executed against source tree at git_sha
  `6e5533080904f719915b646193f2af7fcd3d27fc`. All four LT pass,
  overall GREEN. Phase 8' clean-traceability branch routed.
  PHASE1_HC_BASELINE.md unamended. Audit script tuned twice before
  final verdict (transitive method expansion, exact-path matching);
  both fixes are implementation-side, no spec amendment required.
  Two retrofit v2 manifests synthesized for the existing v1
  envelope's Phase 1 + Phase 2 run directories.
