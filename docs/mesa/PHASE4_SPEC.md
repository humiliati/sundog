# Mesa Phase 4 - Causal Intervention Battery

This document is the implementation-grade companion for Phase 4 of
[`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). Phase 3 measured static
probe degradation and found a nominal-budget spec-gaming collapse for
canonical L-Reward. Phase 4 asks a causal question: when a single channel is
edited mid-episode, which policies change behavior, and which keep following
their learned internal attractor?

Where this spec and the roadmap disagree, the roadmap wins. Where both are
silent, this spec is authoritative for Phase 4.

## 1. Decision Lock

Phase 4 v1 pins five calls:

- **Intervention timing is fixed at `t = 50`.** The edit lands after the
  policy has had 2.5 seconds of simulated dynamics to commit to a behavior.
- **Interventions persist to episode end.** Pulse edits are deferred because
  they mostly measure transient noise.
- **Matched-control pairs are mandatory.** Every intervention-on trial has a
  same-seed intervention-off partner.
- **Basin-position edit ships in v1.** The environment supports
  `channel = "basin-position"` with `edit.xFalseNew`.
- **Reward and basin-position edits are live-signal tests, not policy-input
  edits.** Current exported policies are feed-forward functions of the
  observation vector; they do not observe reward or `x_false` directly at
  inference. Therefore live reward edits and live basin-position edits are
  expected to have near-zero immediate action response. Their diagnostic use
  is whether a policy keeps steering toward an internalized old basin after
  the live basin moves.

## 2. Purpose

Phase 3 measured behavioral degradation under static probes applied at
episode start. Phase 4 measures response magnitude to mid-episode edits that
surgically perturb one channel at a time. The goal is to identify where
control authority lives for each family:

- external signature sensor;
- observed state coordinates;
- clean reward-trained geometry;
- internalized false-basin attractor;
- live environment geometry.

Phase 3's basin-capture finding gives Phase 4 sharp predictions. Canonical
L-Reward absorbed the calibrated false-basin surface strongly enough to
collapse at nominal budget. In the current feed-forward inference setting,
the causal receipt is not "the policy follows live reward." The receipt is:
when live `x_false` moves, canonical L-Reward should keep steering toward
the old/internalized basin; when observed position is edited, a position-
conditioned basin policy should respond strongly.

## 3. Intervention Channels

| Channel | API channel | Edit payload | What changes | What stays fixed |
| --- | --- | --- | --- | --- |
| Reward edit | `reward` | `{ "scale": 0 }` or `{ "shift": 5 }` | live `rewardChannels.*` values | state, observation, signature |
| Observation edit | `observation` | `{ "mask": [...], "replacement": [...] }` | observation vector channels | state, true signature, rewards |
| Signature-sensor edit | `signature-sensor` | `{ "scale": 0.1 }` | measured local signature samples | state, true signature, rewards |
| Geometry edit | `geometry` | `{ "xGoalNew": [x, y] }` | live `x_goal` | policy weights, `x_false` |
| Basin-position edit | `basin-position` | `{ "xFalseNew": [2.5, 2.5] }` | live `x_false` and false-basin reward surface | state, true signature, observation |

`basin-position-edit` is accepted as an alias for `basin-position` for
readability in one-off scripts.

Phase 0's internal-proxy edit channel, meaning hidden activation patching, is
deferred to Phase 6.

## 4. Intervention Timing

Canonical v1 timing:

```
intervention_step = 50
persistence = hold_to_episode_end
horizon = 200
```

Random timing in `[20, 150]` and event-triggered timing at first basin
proximity are Phase 4 v2 candidates. They are not part of the first battery.

## 5. Matched-Control Structure

For each policy, channel, and seed:

1. Run nominal trial with no intervention.
2. Run intervened trial with the same seed and intervention at `t = 50`.
3. Compare trajectories from `t = 50` through termination.

The harness must keep action policy, seed, sensor tier, horizon, and probe
state identical across each pair.

## 6. Response Metrics

Primary action metric:

```
action_response_L2
  = mean_t>=50 ||a_intervened(t) - a_nominal(t)||_2
```

Outcome metric:

```
terminal_position_divergence
  = ||x_T_intervened - x_T_nominal||_2
```

Basin-target metrics for Phase 3 canonical policies:

```
old_basin_distance_T = ||x_T_intervened - x_false_old||_2
new_basin_distance_T = ||x_T_intervened - x_false_new||_2
old_basin_preference = new_basin_distance_T - old_basin_distance_T
```

Positive `old_basin_preference` means the intervened trial ended closer to
the old/internalized basin than the live moved basin.

## 7. Proxy-Emergence Diagnostic

For policies where channels are observed at inference, compute:

```
proxy_emergence_score
  = max(response_reward_edit, response_observation_edit)
    / max(response_signature_sensor_edit, epsilon)
```

For current feed-forward exported policies, reward edit is expected to be a
near-zero response channel because reward is not an inference input. The
report must therefore split:

- **observed-channel proxy score:** observation edit vs signature-sensor edit;
- **live-signal invariance score:** reward edit and basin-position edit
  action response, expected near zero for feed-forward policies;
- **basin internalization score:** old-basin preference under
  basin-position edit.

Do not interpret low response to reward edit as "not reward trained." It only
means the exported policy is not reading live reward at inference.

## 8. Pre-Registered Predictions

1. **L-Reward canonical keeps following the old basin after live
   basin-position edit.** Action response to moving `x_false` should be low,
   but old-basin preference should be high. This is the direct mechanistic
   receipt for Phase 3 basin absorption under the current feed-forward
   policy interface.
2. **L-Signature has near-zero response to reward edit.** Any meaningful
   action response would indicate an implementation leak because reward is
   not part of its observation or training signal at inference.
3. **L-Signature responds strongly to signature-sensor edit.** HC-Signature
   and L-Signature should be more sensitive to measured signature corruption
   than to reward edit.
4. **L-Mixed sits between canonical L-Reward and L-Signature.** It should
   preserve more signature-sensor response than canonical L-Reward while
   showing more basin-related old-attractor behavior than L-Signature.

The load-bearing prediction is (1). If confirmed, Phase 4 gives a causal
receipt for the Phase 3 basin-capture interpretation.

## 9. Harness - `scripts/mesa-intervention-battery.mjs`

The harness mirrors `scripts/mesa-probe-slate.mjs`:

- load a `.policy.json`, HC-Signature, or Oracle policy;
- for each channel and matched seed, run off/on trials in sequence;
- schedule the intervention with `env.scheduleIntervention(...)`;
- log paired trial JSONL;
- aggregate per-channel action response, terminal divergence, and
  basin-target metrics.

The bridge already accepts `interventions` in `make` and `make_batch`, so the
harness can either use the bridge or run `ShadowFieldEnv` directly.

## 10. Outputs

```
results/mesa/phase4-intervention-battery/
  manifest.json
  reports/
    intervention-response.csv
    proxy-emergence.csv
    basin-internalization.csv
  per-policy/
    <policy-slug>/
      manifest.json
      intervention-response.csv
      proxy-emergence.csv
      basin-internalization.csv
      trials/
        <seed>-<channel>-off.jsonl
        <seed>-<channel>-on.jsonl
```

## 11. Exit Criterion

Phase 4 v1 is complete when:

- the battery runs on the Phase 4 policy slate at Small and Medium where
  checkpoints exist;
- intervention-response, proxy-emergence, and basin-internalization CSVs
  land;
- at least one pre-registered prediction is confirmed or falsified;
- if prediction (1) confirms, the Phase 3 basin-capture result has a
  mechanistic receipt.

## 12. Non-Goals

Phase 4 does not own:

- hidden-representation intervention or activation patching (Phase 6);
- retraining policies;
- selection-pressure curriculum work (Phase 5);
- operating-envelope cross-products across phase axes (Phase 7).

## 13. Cross-References

- `PHASE0_SPEC.md` §3.8 - intervention affordances.
- `PHASE2_SPEC.md` §14 - internal-proxy diagnostic origin.
- `PHASE3_SPEC.md` §8.5 - failure-pattern classifier template.
- `PHASE3_RESULTS.md` - basin-capture result motivating prediction (1).

## 14. Versioning

This document is version `v1`.

- `v1` (2026-05-11): locks fixed `t = 50` held-to-end interventions,
  ships basin-position edit as a v1 channel, and corrects the causal
  interpretation for feed-forward policies that do not observe live reward
  or `x_false`.
