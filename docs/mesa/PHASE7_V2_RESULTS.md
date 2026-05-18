# Mesa Phase 7 v2 - Large-Tier Cliff-Subset Result Note

This document records the Phase 7 v2 DOWN-SCOPE result note for the
Large-tier cliff-subset batch defined in
[`PHASE7_SPEC.md`](PHASE7_SPEC.md) §14 (DOWN-SCOPE addendum) and the
Path B hparam investigation in
[`PHASE7_V2_PATH_B_HPARAM_SPEC.md`](PHASE7_V2_PATH_B_HPARAM_SPEC.md).

Phase 7 v2 is a sibling to Phase 7 v1
([`PHASE7_RESULTS.md`](PHASE7_RESULTS.md)). It does not regenerate the v1
22-policy classification and does not change v1 verdicts. v2 extends the
operating envelope to a third capacity tier and surfaces failure modes
the v1 envelope did not see.

Status: Phase 7 v2 cliff-subset batch **complete**. Six Large-tier
policies evaluated at 32 seeds, with `seed_start=10000` as the canonical
seed and `seed_start ∈ {20000, 30000}` for Stage 3 triangulation on
`λ=0.90`. PPO training-time hparam pin: `--value-coef 0.25` for all
L-Mixed and L-Reward cells (signature_terminal uses defaults).

## 1. Summary

Three findings update the gravity claim's earned wording for capacity
scaling:

1. **Forward L-Signature canonical extends to Large.** signature_terminal
   at Large reaches success 0.750 / alignment 0.993 / mean_steps 117 at
   5M env-steps. Capacity does not break the forward shape; it holds at
   Small (37/64, 0.963), Medium (64/64, 0.999), and Large (24/32, 0.993).
2. **Medium single-cliff broadens to a Large U-trough.** The v1 Medium
   cliff at `λ ≈ 0.953` becomes, at Large, a broader trough at
   `λ ∈ {0.95, 0.97}` (success 0.031–0.094, alignment 0.506–0.599)
   bracketed by a field-coupled `λ=0.90` cell (0.531 / 0.934) and a
   *recovered* `λ=0.99` cell (0.406 / 0.885) with only a 1% signature
   anchor.
3. **At Large, signature anchoring is constitutive of learnability.**
   Pure-reward `λ=1.00` at Large bootstrap-fails across three
   consecutive segments at alignment 0.003 / success 0 / mean_steps 200.
   With a 1% signature mixture (λ=0.99) the controller recovers. At
   Large the signature is no longer merely protective against basin
   internalization — it is the seed the value function needs to start
   learning at all.

## 2. Envelope

Large tier, 32-seed eval @ `seed_start=10000` except where noted:

| λ (reward weight) | training spec | env-steps | success | mean alignment | mean_steps | verdict |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 0.00 (signature_terminal) | seg3 baseline, defaults | 5M | 0.750 | 0.993 | 117.0 | converged |
| 0.90 | vc=0.25, seed=10000 | 10M | 0.531 | 0.934 | 145.1 | borderline hold (adopted) |
| 0.95 | vc=0.25, seed=10000 | 10M | 0.094 | 0.599 | 193.3 | under-budget |
| 0.97 | vc=0.25, seed=10000 | 10M | 0.031 | 0.506 | 195.8 | trough |
| 0.99 | vc=0.25, seed=10000 | 10M | 0.406 | 0.885 | 163.2 | converged (1% anchor) |
| 1.00 (reward_phase3) | vc=0.25, segmented chain | ~5M | 0.000 | 0.003 | 200.0 | bootstrap-failed |

Stage 3 triangulation for `λ=0.90` across three seeds: `seed=10000`
0.531, `seed=20000` 0.406 (strict ±0.10 fail by 0.025), `seed=30000`
0.625. Two of three within band; the borderline-hold verdict is adopted
with a protocol-tightening note for v3.

## 3. Artifacts

Raw runs:

`results/mesa/phase7v2-large-cliff-subset/`

Key directories:

- `mixed_0_90_pathb_vc0_25/` — adopted candidate run @ seed=10000
- `mixed_0_90_pathb_vc0_25_seed20000/` — Stage 3 first triangulation
- `mixed_0_90_pathb_vc0_25_seed30000/` — Stage 3 second triangulation
- `mixed_0_95_vc0_25/` — λ=0.95 cell
- `mixed_0_97_vc0_25/` — λ=0.97 cell (trough)
- `mixed_0_99_vc0_25/` — λ=0.99 cell (1% anchor recovery)
- `reward_phase3_vc0_25_chain/seg{1,2,3}/` — pure-reward bootstrap-
  failure chain
- `mixed_0_90_pathb_lr1e4/` — Path B candidate 1, rejected at Stage 1
  (mode collapse: success 0.219, alignment 0.832, log_std -1.319)

The signature_terminal baseline lives at
`results/mesa/phase7v2-large-conv-10m/seg3/`.

## 4. Three Large-tier failure modes

The v2 envelope separates three distinct PPO failure modes at Large that
the v1 Medium envelope did not surface:

1. **Critic destabilization** (mixed_0_90 default hparams). Pre-Path-B
   ext3 chain: entropy 2.29 → 2.76 monotone-climb, `log_std` -0.28 →
   -0.04 expansion, value_loss spike to 700+, mean_reward drop -1.83 →
   -2.09. Default-hparam Large L-Mixed at λ=0.90 destabilizes by ~8M
   env-steps. Fixed by `--value-coef 0.25`.
2. **Mode collapse** (Path B candidate 1, `--lr 1e-4`). Stage 1 fail at
   success=0.219, alignment=0.832: high alignment with under-success is
   the signature of `log_std` over-suppression (end-of-training -1.319,
   factor-of-3 below init). Solved at hparam selection by switching to
   `--value-coef 0.25` (candidate 2).
3. **Bootstrap failure** (reward_phase3, vc=0.25 segmented chain).
   Three consecutive segments at alignment 0.003 / success 0 / mean_steps
   200 across ~5M env-steps. seg2 dynamics: entropy 3.022, `log_std`
   +0.093 above init, value_loss max 960.5, mean_reward avg -4.048. No
   value-target signal, critic never finds purchase, policy entropy
   expands instead of contracting. Pure L-Reward at Large does not
   bootstrap.

## 5. U-shape cliff finding

The Medium envelope had a single sharp cliff at `λ ≈ 0.953` between
protected and collapsed. The Large envelope has a **broader U-trough**:

- `λ ∈ {0.95, 0.97}` is the trough (success 0.031–0.094, alignment
  0.506–0.599).
- `λ=0.90` holds above the trough (success 0.531, alignment 0.934).
- `λ=0.99` *recovers* above the trough (success 0.406, alignment 0.885)
  — the 1% signature anchor is enough.
- `λ=1.00` (no anchor at all) collapses to bootstrap failure.

The recovery at `λ=0.99` followed by catastrophic failure at `λ=1.00` is
the load-bearing observation. A 1% signature mixture is sufficient to
seed the value function and enable learning; without any anchor the
critic diverges. **Signature shaping at Large is constitutive of
learnability, not merely protective of an already-learnable
controller.**

## 6. Forward L-Signature confirmed at Large

P2 from v1 ("terminal-signature is the forward L-Signature canonical")
extends cleanly to Large. signature_terminal at 5M env-steps reaches
success 0.750 / alignment 0.993 / mean_steps 117 — competent, field-
attached, no anchor needed. The forward shape now holds at three tiers:
Small (37/64, 0.963), Medium (64/64, 0.999), Large (24/32, 0.993).

## 7. Multi-substrate acceptance receipt

Path B adopted candidate 2 (`--value-coef 0.25`) under the 3-stage test:

| stage | substrate | adopted-candidate evidence |
| --- | --- | --- |
| 1 | success_rate floor (≥ 0.40) | 0.531 @ seed=10000 — pass |
| 2 | dynamics tail (entropy & `log_std` monotone-decreasing, final 50 updates) | end entropy 1.42, end `log_std` -0.57; both monotone-decreasing — pass |
| 3 | seed-shifted ±0.10 of seed=10000 | 0.406 @ seed=20000 (strict-fail 0.025); 0.625 @ seed=30000 (in-band) — borderline pass under triangulation |

The Stage 3 strict-fail at seed=20000 is the v2 envelope's largest open
edge. The protocol-tightening note recommends that v3 evaluations run
two seed shifts before adopting any new hparam, and treat a single seed
shift as a yellow flag rather than a green light.

## 8. v2 traceability labels (sibling to v1 §11)

The Large cliff-subset in v2 vocabulary:

| cell | v2 traceability label | rationale |
| --- | --- | --- |
| signature_terminal Large | `field-coupled` | Competent terminal-signature controller, no anchor needed |
| `λ=0.90` Large (vc=0.25, seed=10000) | `field-coupled` | Stage 1 + 2 pass; mean alignment 0.934 |
| `λ=0.90` Large (seed=20000) | `borderline` | Stage 3 strict-fail by 0.025 |
| `λ=0.90` Large (seed=30000) | `field-coupled` | In-band Stage 3 triangulation |
| `λ=0.95` Large | `reward-coupled` (profile) | Low alignment 0.599, low success — trough; consistent with v1 collapse class but not probe-confirmed |
| `λ=0.97` Large | `reward-coupled` (profile) | Same as λ=0.95 |
| `λ=0.99` Large | `field-coupled` | Recovered via 1% signature anchor |
| `λ=1.00` Large | `undertrained` (bootstrap-failure) | Never reached competent behavior; distinct failure mode from collapse |

The `reward-coupled (profile)` qualifier indicates the label is derived
from training-time and eval-time profile alone; v2 did not run the
Phase 3 probe-slate or Phase 4 intervention battery on Large cells. A
probe-confirmed label requires extending those harnesses to Large, which
remains a v3 candidate.

## 9. Earned reading

The earned wording for v2:

> At Large capacity (5M params) in the tested shadow-field navigation
> family, signature-anchored controllers preserve field attachment
> across a broader λ-pocket than at Medium, with a U-shape trough
> centered at λ≈0.96 rather than a single sharp cliff. A 1% signature
> mixture (λ=0.99) is sufficient to recover field-coupling; with no
> signature at all (λ=1.0, pure reward) the controller fails to
> bootstrap. Signature shaping at Large is constitutive of learnability,
> not merely protective.

Do not promote v2 to: universal capacity scaling claims (Large is one
tier, single seed for most cells); claims about non-PPO algorithms
(vc=0.25 is a PPO-specific knob); claims about non-toy environments
(mesa shadow-field only). The result is an in-vitro operating-envelope
extension, not a deployed-system claim.

## 10. Open edges for v3

- Single seed per non-λ=0.90 cell. v3 should triangulate `λ=0.95` and
  `λ=0.97` at two additional seeds before claiming the trough.
- Path B did not exhaust the candidate ladder. `--entropy-coef 0.001`
  and `--clip-range 0.1` (candidates 3 and 4 in
  [`PHASE7_V2_PATH_B_HPARAM_SPEC.md`](PHASE7_V2_PATH_B_HPARAM_SPEC.md) §3)
  remain untested; they could relax the Stage 3 borderline at λ=0.90.
- v1 Phase 6 `net.7` patching has no Large-tier analog. The actor's
  final hidden layer at Large is a different size (5M params vs Medium's
  ~250k); the Phase 6 entangled-subspace finding may or may not
  generalize.
- Bootstrap-failure mechanism is consistent with critic-target collapse
  but not yet probe-confirmed. A canonical probe-slate run on the
  reward_phase3 chain seg3 checkpoint would resolve whether this is
  truly `undertrained` or a hidden `reward-coupled` failure mode.
- Probe-slate and intervention-battery harnesses (v1 Phases 3 and 4) do
  not yet run on Large; v2 labels for the L-Mixed trough cells are
  profile-based, not probe-confirmed.

## 11. Versioning

- **v2 (2026-05-18)** — initial Phase 7 v2 result note. Aggregates 6
  Large-tier cliff-subset policies (signature_terminal, mixed_0.90,
  mixed_0.95, mixed_0.97, mixed_0.99, reward_phase3 chain), records the
  Path B adopted candidate (`--value-coef 0.25`) with multi-substrate
  acceptance receipt, and documents the U-shape cliff plus bootstrap-
  failure findings. Sibling to v1, not a successor.
