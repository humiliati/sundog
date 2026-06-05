# Phase 3 v4 Path-A Verify-First Note (basin action-visible)

Status: **PAUSED — verify-first experiment run; Design 1 not justified; scaffolding kept (default-off).** This is a note, not a receipt: no slate was frozen and no promotion claim is made.

Date: 2026-06-04

## Context

The v4 basin-OR mechanism was falsified before freeze (see
[`PHASE3_CAPACITY_ONE_WAYNESS_V4_SLATE.md`](PHASE3_CAPACITY_ONE_WAYNESS_V4_SLATE.md)):
the reward-only `basin-position` channel is structurally inert (response 0; the
basin move feeds only `rewardChannels`, which reward-blind feed-forward policies
ignore at inference). The owner chose **Path A** — make the basin action-visible.
Scoping found the only non-privileged action-visible variant is **basin-in-observation
+ retrain** (Design 1), which re-scopes the certificate from "detect hidden
internalization" to "agent observes-and-reacts" and needs a ~45–76-policy retrain.
Per the verify-first discipline (do not pre-register an unverified mechanism), a
cheap experiment was run before any v4-A slate.

## Scaffolding built (KEPT, additive, default-off)

- `public/js/mesa-core.mjs`: optional `basinChannel` config (default false). When
  true, `observe()` appends the agent-relative `observedBasinCenter` vector (obs
  6→8). New `basin-observation` intervention channel edits `observedBasinCenter`,
  **decoupled** from the reward-only `falseBasinCenter` (which the legacy
  `basin-position` channel edits). Existing length-6 policies and batteries are
  unaffected (channel off by default).
- `training/mesa/train_ppo.py`: `--basin-channel` flag (sets env `basinChannel=true`
  and bumps `PolicyConfig.obs_dim` by 2). Exports an obs_dim-8 policy.
- Scratch harness: `scripts/_v4a_plumbing_check.mjs`, `scripts/_v4a_discrimination_eval.mjs`.
- Scratch trained policies (gitignored): `results/pvnp/_v4a_train{,_long}/`.

## Result 1 — plumbing + decoupling: PASS (mechanism is action-visible)

`scripts/_v4a_plumbing_check.mjs`:
- obs length 6 → 8 with `basinChannel`.
- `basin-observation` edit: observed-basin features change; `reward.false_basin`
  **unchanged**. An obs-reading policy's action moves `|ΔA| = 0.707`.
- legacy `basin-position` edit: `reward.false_basin` changes; observed-basin
  features **unchanged**; action `|ΔA| = 0.000` (inert, reproduces the v4 falsification).
- Decoupling clean. The basin can be made action-visible without touching reward.

## Result 2 — discrimination / specificity: FAIL (non-specific)

`scripts/_v4a_discrimination_eval.mjs` (32 seeds, basin-observation edit at step 50
to the opposite corner; mean post-intervention action divergence). Trained a
λ0 `signature_ppo_dense` (no basin reward → should be basin-flat) vs a λ1.0
`reward_ppo_phase3` (full basin reward → should be basin-responsive), both with
`--basin-channel`:

| basin-observation response | 50 updates | 120 updates (firmed, ~1M steps) |
| --- | ---: | ---: |
| signature (λ0) | 0.311 | **0.431** |
| reward (λ1.0) | 0.474 | 0.558 |
| **discrimination ratio** | 1.52× | **1.29×** |
| reward-only basin-position | 0.000 | 0.000 |

**Conclusion: the basin-observation channel is action-visible but NOT a specific
mixed-objective detector.** The pure-signature policy responds at 0.431 — far above
any 0.23-type line — and the gap to the reward policy is a noise-level 1.29×. With
more training the signature response **grew** (0.31→0.43) and the ratio **shrank**
(1.52→1.29): there is no gradient pressure to zero a signature policy's basin-input
weights, so it never becomes basin-flat. This is structural — putting the basin in
the observation means **any** policy wired to it responds to editing it; the
response measures "the feature is in my input," not "I internalized a basin
preference." A pure-signature cell would consensus-trigger the basin flag (the
specificity claim-cap), so no bounded positive could be earned.

**Therefore the full ~45–76-policy retrain (Design 1) is NOT justified.** The
verify-first experiment did its job: it killed Design 1 with two small trains
before the big spend.

## Net for Phase 3

No reward-blind behavioral channel rescues the near-line anchor: the reward-only
basin channel is inert (v4), and the basin-observation channel is confounded /
non-specific (v4-A). The observation channel (v3) remains the only reward-blind
mixed-objective signal, and it is seed-fragile at the anchor — a complete, honest
capacity-relative one-wayness boundary.

## Paused — possible future direction (NOT pursued; p-hack caveat)

The straightforward action-magnitude metric is confounded. A **goal-directed**
basin-response metric — does the policy move *toward* the new basin (a basin-seeker)
vs respond incidentally (any wired policy) — might separate the two. BUT designing
or tuning such a metric now, on data already seen, is p-hacking; it would require a
fresh pre-registration and fresh seeds, and it still tests "observes-and-reacts,"
not hidden internalization. Left as a paused option, scaffolding retained.
