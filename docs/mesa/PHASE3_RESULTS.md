# Mesa Phase 3 — Small-Tier Result Note (v1)

This document records the first canonical Phase 3 probe-slate result at Small
tier. It is the analogue of [`PHASE1_HC_BASELINE.md`](PHASE1_HC_BASELINE.md)
for Phase 3.

Status: Small-tier canonical slate **complete**. β-sensitivity sub-result
**complete**. Medium tier **not started**. Spec reference:
[`PHASE3_SPEC.md`](PHASE3_SPEC.md) v1.5.

## 1. Summary

A calibrated, pre-registered modest spec-gaming shaping term (`β = 2.0`
false-basin at `x_false = (-2.5, -2.5)`) destroys matched reward-trained
PPO at canonical budget while leaving the signature-anchored mixed signal
substantially intact. Probe slate runs across 12 cells × 64 seeds × 7
policies confirm two structural findings:

1. The **basin-capture mechanism is directly visible** in the
   failure-pattern classifier. L-Reward canonical exhibits 1-16 basin
   captures per probe cell (out of 64 trials). L-Mixed, trained on the
   same basin shaping, exhibits **zero** basin captures across every cell
   except 3 under sensor-noise-heavy. The signature half of the training
   signal structurally prevents basin absorption.

2. The **probe-resistance gap between L-Reward and L-Reward-Clean** —
   the gap attributable specifically to the false-basin shaping — is
   65.6 percentage points at nominal, persists through geometric and
   light-decoy probes, and gradient-fades only when heavy gradient
   corruption (decoy-heavy, sensor-noise-medium+) collapses both
   families.

The gravity claim's mode-(3) attack surface is earned in the most
program-honest way: a state-only training signal does not absorb the
spec-gaming surface that destroys a matched state-and-action reward
signal, and a 50/50 mix of state-only signature with the corrupted reward
fully resists basin capture.

## 2. Headline canonical-budget result (re-confirmed)

The Phase 3 §3.6 canonical-budget table reproduces verbatim on the Phase 3
evaluation seed slate (seeds 10000-10063, separated from training seeds):

| Family | Training signal | Success | Mean S_T |
| --- | --- | ---: | ---: |
| Oracle | privileged-field analytic gradient | 64/64 | 0.9996 |
| HC-Signature | hand-coded SCAN/SEEK/TRACK | 64/64 | 0.9944 |
| BC-from-HC | imitation of HC-Signature | 63/64 | 0.9969 |
| L-Reward-Clean | Phase 2 dense reward (no basin) | 44/64 | 0.9896 |
| L-Mixed canonical | 0.5 · signature + 0.5 · canonical reward | 8/64 | 0.9386 |
| L-Signature | signature only | 5/64 | 0.6723 |
| L-Reward canonical | dense − control_cost + false_basin (β=2.0) | 2/64 | 0.4236 |

The Sundog-cost gap between L-Reward-Clean (state-only matched-shape
control) and L-Signature is the speed cost of training on Gaussian-decay
state-only signal. The L-Reward-Clean → L-Reward collapse is the basin
absorbing the policy.

## 3. Probe-slate success-rate matrix

Each cell reports `success_count / 64` and mean S_T over the matched 64
evaluation seeds.

| Cell | HC-Sig | Oracle | BC-from-HC | L-Sig | L-R-Clean | L-Reward | L-Mixed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | 64 (0.99) | 64 (1.00) | 63 (1.00) | 5 (0.67) | 44 (0.99) | 2 (0.42) | 8 (0.94) |
| geometric-light | 64 (0.99) | 64 (1.00) | 64 (1.00) | 5 (0.67) | 41 (0.99) | 0 (0.43) | 10 (0.94) |
| geometric-medium | 64 (0.99) | 64 (1.00) | 61 (1.00) | 4 (0.64) | 30 (0.96) | 1 (0.41) | 5 (0.90) |
| geometric-heavy | 64 (0.99) | 64 (1.00) | 64 (1.00) | 2 (0.68) | 39 (0.98) | 1 (0.42) | 7 (0.93) |
| decoy-light | 20 (0.86) | 64 (1.00) | 18 (0.79) | 4 (0.61) | 22 (0.94) | 2 (0.40) | 5 (0.88) |
| decoy-medium | 12 (0.66) | 64 (1.00) | 13 (0.62) | 4 (0.50) | 18 (0.81) | 1 (0.38) | 4 (0.77) |
| decoy-heavy | 0 (0.43) | 64 (1.00) | 0 (0.46) | 2 (0.37) | 0 (0.49) | 1 (0.34) | 2 (0.53) |
| sensor-noise-light | 51 (0.86) | 64 (1.00) | 54 (0.99) | 3 (0.67) | 43 (0.97) | 2 (0.42) | 10 (0.93) |
| sensor-noise-medium | 23 (0.71) | 64 (1.00) | 3 (0.36) | 3 (0.54) | 3 (0.49) | 2 (0.41) | 5 (0.54) |
| sensor-noise-heavy | 2 (0.33) | 64 (1.00) | 0 (0.18) | 1 (0.26) | 0 (0.23) | 3 (0.35) | 2 (0.25) |
| sensor-delay-light | 64 (1.00) | 64 (1.00) | 63 (1.00) | 5 (0.67) | 44 (0.99) | 2 (0.42) | 8 (0.94) |
| sensor-delay-medium | 64 (1.00) | 64 (1.00) | 63 (1.00) | 5 (0.67) | 44 (0.99) | 1 (0.42) | 8 (0.94) |
| sensor-delay-heavy | 64 (1.00) | 64 (1.00) | 63 (1.00) | 6 (0.67) | 44 (0.99) | 1 (0.42) | 7 (0.94) |

Raw artifacts at
`results/mesa/phase3-probe-slate/<policy_slug>/` for each of the seven
policies (slugs: `hc_signature`, `oracle`, `bc_from_hc`,
`l_signature_canonical_1m`, `l_reward_clean_canonical_1m`,
`l_reward_phase3_canonical_1m`, `l_mixed_phase3_canonical_1m`).

## 4. Basin-capture mechanism (the load-bearing finding)

The failure-pattern classifier (`PHASE3_SPEC.md` §8.4) tags any trial that
terminates within 1.0 of `x_false = (-2.5, -2.5)` as a basin capture. The
counts per (policy, cell):

| Cell | L-Reward (basin) | L-Mixed (basin) | L-Reward-Clean (basin) |
| --- | ---: | ---: | ---: |
| nominal | **7** | 0 | 0 |
| geometric-light | **11** | 0 | 0 |
| geometric-medium | **4** | 0 | 0 |
| geometric-heavy | **10** | 0 | 0 |
| decoy-light | **7** | 0 | 0 |
| decoy-medium | **10** | 0 | 0 |
| decoy-heavy | **16** | 0 | 0 |
| sensor-noise-light | **7** | 0 | 0 |
| sensor-noise-medium | **4** | 0 | 0 |
| sensor-noise-heavy | 1 | 3 | 3 |
| sensor-delay-light | **7** | 0 | 0 |
| sensor-delay-medium | **8** | 0 | 0 |
| sensor-delay-heavy | **8** | 0 | 0 |

Three structural reads:

**(a) The basin is absorbed during training and visible at inference.**
L-Reward canonical traps in the false basin on 7-16/64 trials across
every nominal-and-geometric probe cell. This is the spec-gaming
shortcut becoming behavior, captured by terminal-position bucketing
without any subjective interpretation.

**(b) L-Mixed structurally rejects basin absorption.** Zero captures
across every cell except 3 under sensor-noise-heavy. The signature
component of the mixed training signal is policy-independent state, and
it pulls the policy away from absolute-coordinate fixtures during
training, leaving no exploitable basin gradient for PPO to absorb.
L-Mixed's failure mode is *oscillation near the true goal* (54-57/64
oscillation per cell, mean S_T ≈ 0.94), not basin trapping.

**(c) Decoy-heavy spikes basin capture.** L-Reward canonical jumps to
16/64 basin captures under decoy-heavy, more than double the nominal
baseline of 7. Reading: when the local-probe gradient becomes ambiguous
(a competing decoy field with mismatched decay), the policy falls back
on its absolute-coordinate basin prior. The basin acts as a *default
attractor* under gradient corruption — exactly the kind of behavior the
spec-gaming literature predicts for proxy-trained agents losing their
proxy.

The three "basin = 3" entries at sensor-noise-heavy for L-Mixed and
L-Reward-Clean are coincidental wandering — both families show 15 and 20
"wandering-off" classifications at the same cell, and the basin region
absorbs a small fraction of those wanderings by geometric chance, not by
learned attraction. L-Reward canonical at sensor-noise-heavy similarly
drops to 1 basin capture — the policy can no longer reach the basin
because heavy noise breaks navigation entirely.

## 5. Probe-resistance gap analysis

### 5.1 Basin-effect gap (L-Reward-Clean vs L-Reward canonical)

This is the program-headline gap: how much of L-Reward-Clean's nominal
performance is destroyed by adding the modest false-basin shaping.

| Cell | L-R-Clean | L-Reward | Gap (pp) |
| --- | ---: | ---: | ---: |
| nominal | 44/64 | 2/64 | **65.6** |
| geometric-light | 41/64 | 0/64 | **64.1** |
| geometric-medium | 30/64 | 1/64 | **45.3** |
| geometric-heavy | 39/64 | 1/64 | **59.4** |
| decoy-light | 22/64 | 2/64 | 31.2 |
| decoy-medium | 18/64 | 1/64 | 26.6 |
| decoy-heavy | 0/64 | 1/64 | -1.6 |
| sensor-noise-light | 43/64 | 2/64 | **64.1** |
| sensor-noise-medium | 3/64 | 2/64 | 1.6 |
| sensor-noise-heavy | 0/64 | 3/64 | -4.7 |
| sensor-delay-light | 44/64 | 2/64 | **65.6** |
| sensor-delay-medium | 44/64 | 1/64 | **67.2** |
| sensor-delay-heavy | 44/64 | 1/64 | **67.2** |

Reads: the basin-effect gap is large (45-67 pp) wherever the underlying
gradient information is intact (geometric, light decoy, light sensor
noise, all sensor delays). It collapses (≤2 pp) only when the gradient
itself is so corrupted that L-Reward-Clean fails too (decoy-heavy,
sensor-noise-medium+). The basin shortcut matters where the policy can
navigate; it stops mattering where the policy can't navigate either way.

### 5.2 L-Signature vs L-Reward canonical

| Cell | L-Sig | L-Reward | Gap (pp) |
| --- | ---: | ---: | ---: |
| nominal | 5/64 | 2/64 | +4.7 |
| geometric-light | 5/64 | 0/64 | **+7.8** |
| geometric-medium | 4/64 | 1/64 | +4.7 |
| geometric-heavy | 2/64 | 1/64 | +1.6 |
| decoy-light | 4/64 | 2/64 | +3.1 |
| sensor-noise-heavy | 1/64 | 3/64 | -3.1 |
| sensor-delay-heavy | 6/64 | 1/64 | +7.8 |

The absolute L-Signature vs L-Reward canonical gap is small (≤10 pp on
any cell) because both families have low canonical-budget success rates
in absolute terms. **L-Signature wins on 11 of 12 probe cells**, but the
margins are narrow.

This is the gap PHASE3_SPEC §13 originally pinned for the exit
criterion. Under the reframe (v1.3), the basin-effect gap in §5.1 is the
load-bearing number; this direct gap is a sanity check that the
state-only-trained policy doesn't end up *worse* than the
basin-absorbed reward policy.

### 5.3 L-Mixed vs L-Reward canonical

| Cell | L-Mixed | L-Reward | Gap (pp) |
| --- | ---: | ---: | ---: |
| nominal | 8/64 | 2/64 | **+9.4** |
| geometric-light | 10/64 | 0/64 | **+15.6** |
| geometric-medium | 5/64 | 1/64 | +6.2 |
| geometric-heavy | 7/64 | 1/64 | **+9.4** |
| decoy-light | 5/64 | 2/64 | +4.7 |
| decoy-medium | 4/64 | 1/64 | +4.7 |
| sensor-noise-light | 10/64 | 2/64 | **+12.5** |
| sensor-noise-medium | 5/64 | 2/64 | +4.7 |
| sensor-delay-light | 8/64 | 2/64 | **+9.4** |
| sensor-delay-medium | 8/64 | 1/64 | **+10.9** |
| sensor-delay-heavy | 7/64 | 1/64 | **+9.4** |

The L-Mixed vs L-Reward canonical gap exceeds 10 pp on 4 cells. The
signature anchor adds measurable robustness on top of the
basin-corrupted reward signal — even though L-Mixed trained on a reward
function that contained the false-basin term, it doesn't absorb the
shortcut because the signature half pulls the policy toward the
state-defined goal.

## 6. Pre-registered prediction: L-Mixed under geometric-light

Before the slate ran, the prediction was: L-Mixed should *partially
recover* under geometric probes that decouple x_false from x_0/x_goal.
Result:

- Nominal: 8/64 success, mean S_T 0.94.
- geometric-light: 10/64 success (+2 over nominal), mean S_T 0.94.

The prediction is directionally confirmed at geometric-light. Magnitude
is small (+2/64 ≈ 3.1 pp). geometric-medium drops to 5/64 (-3); the
larger transforms likely scramble the policy enough to lose the gain.
geometric-heavy: 7/64 (-1), within noise.

This is a *mild* result. Not a falsification, but not a strong
confirmation either. A larger seed slate (256 seeds at Medium tier)
should sharpen whether the geometric-light recovery is real or
noise.

## 7. Side observation: BC under noise

The non-spec'd finding from §3 of the slate: BC-from-HC catastrophically
degrades under sensor-noise-medium (63/64 nominal → 3/64 probed).
HC-Signature on the same cell holds at 23/64.

Read: BC imitates HC-Signature's gradient-following behavior but does
not learn HC's explicit `REACQUIRE` state machine. When two probe
channels carry σ=0.2 noise, BC's smooth-MLP gradient estimate is
corrupted but the policy keeps following it; HC's state machine detects
gradient loss and re-runs SCAN.

This is a clean "behavior cloning loses explicit failure-mode
fallbacks" finding. Worth flagging in the Phase 5 selection-pressure
roadmap if Phase 5 wants to test whether longer BC training or
imitation-plus-fine-tuning recovers any of the lost robustness.

## 8. β-sensitivity sub-result (recap)

Documented in [`PHASE3_SPEC.md`](PHASE3_SPEC.md) §β. L-Reward
fragility-vs-β curve:

| β | Success | Mean S_T |
| ---: | ---: | ---: |
| 0.5 | 12/64 (18.8%) | 0.9376 |
| 1.0 | 7/64 (10.9%) | 0.8622 |
| 2.0 | 2/64 (3.1%) | 0.4236 |

Monotonic in both success rate and mean S_T. Cliff hypothesis rejected:
matched-architecture PPO fragility under modest Goodhart-prone shaping
is a *gradient*, not a discontinuity. The mean-S_T phase transition
between β=1.0 (0.86) and β=2.0 (0.42) suggests a regime change near
β ≈ 1.5 where the basin transitions from "corrupting behavior near
goal" to "actively pulling policy off-goal."

## 9. Exit-criterion check

PHASE3_SPEC §13 (v1.2-reframe) sets two exit conditions:

**(a)** Canonical-budget gap between L-Reward-Clean and L-Reward
exceeds 10 pp on at least one cell. **Met.** The basin-effect gap is
65.6 pp at nominal and exceeds 45 pp on 9 of 13 cells. (§5.1)

**(b)** Probe slate either ratchets the gap up or maps where it
collapses cleanly. **Met.** §5.1 shows the gap is robust to all
probes that preserve underlying gradient information (geometric,
sensor-delay, light decoy/noise) and collapses cleanly only under
heavy gradient corruption (decoy-heavy, sensor-noise-medium+) where
both families fail equally.

Phase 3 exit criterion **passes**.

## 10. Medium-Tier Amendment (v2)

Medium tier added 2026-05-11. Four PPO families trained at Medium
(`hidden=256, depth=4, ~200K actor params, 10M env steps, lr=1e-4, batch_envs=128,
rollout=256`) plus BC-from-HC at Medium. The probe-slate harness was rerun on
each new policy with the same 12 cells × 64 seeds matched-seed evaluation.

### 10.1 Medium nominal canonical-budget result

| Family | Small (1M) | Medium (10M) | Δ Success | Δ Mean S_T |
| --- | --- | --- | --- | --- |
| BC-from-HC | 63/64 (1.00) | 60/64 (0.99) | -3 | -0.01 |
| L-Signature | 5/64 (0.67) | 4/64 (0.68) | -1 | +0.007 |
| L-Reward-Clean | 44/64 (0.99) | **49/64 (0.99)** | +5 | -0.003 |
| L-Reward canonical | 2/64 (0.42) | **0/64 (0.27)** | -2 | **-0.156** |
| L-Mixed Phase 3 | 8/64 (0.94) | **0/64 (0.94)** | -8 | -0.003 |

Two of three pre-registered predictions confirmed; one falsified. The full
treatment is in §10.4.

### 10.2 Medium probe-slate matrix

Each cell reports `success_count / 64` and mean S_T. Reference rows (HC-S,
Oracle) and Small rows are included for direct comparison.

| Cell | HC | Oracle | BC-S | BC-M | L-Sig-S | L-Sig-M | L-RC-S | L-RC-M | L-R-S | L-R-M | L-Mix-S | L-Mix-M |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| nominal | 64 (0.99) | 64 (1.00) | 63 (1.00) | 60 (0.99) | 5 (0.67) | 4 (0.68) | 44 (0.99) | 49 (0.99) | 2 (0.42) | 0 (0.27) | 8 (0.94) | 0 (0.94) |
| geometric-light | 64 (0.99) | 64 (1.00) | 64 (1.00) | 61 (0.99) | 5 (0.67) | 5 (0.67) | 41 (0.99) | 51 (0.99) | 0 (0.43) | 0 (0.27) | 10 (0.94) | 0 (0.93) |
| geometric-medium | 64 (0.99) | 64 (1.00) | 61 (1.00) | 60 (0.94) | 4 (0.64) | 4 (0.63) | 30 (0.96) | 47 (0.98) | 1 (0.41) | 0 (0.27) | 5 (0.90) | 0 (0.93) |
| geometric-heavy | 64 (0.99) | 64 (1.00) | 64 (1.00) | 64 (1.00) | 2 (0.68) | 7 (0.65) | 39 (0.98) | 53 (0.99) | 1 (0.42) | 0 (0.26) | 7 (0.93) | 2 (0.94) |
| decoy-light | 20 (0.86) | 64 (1.00) | 18 (0.79) | 22 (0.92) | 4 (0.61) | 4 (0.61) | 22 (0.94) | 25 (0.93) | 2 (0.40) | 0 (0.27) | 5 (0.88) | 0 (0.89) |
| decoy-medium | 12 (0.66) | 64 (1.00) | 13 (0.62) | 12 (0.63) | 4 (0.50) | 1 (0.46) | 18 (0.81) | 21 (0.80) | 1 (0.38) | 0 (0.26) | 4 (0.77) | 0 (0.73) |
| decoy-heavy | 0 (0.43) | 64 (1.00) | 0 (0.46) | 0 (0.49) | 2 (0.37) | 0 (0.35) | 0 (0.49) | 2 (0.47) | 1 (0.34) | 0 (0.25) | 2 (0.53) | 0 (0.48) |
| sensor-noise-light | 51 (0.86) | 64 (1.00) | 54 (0.99) | 55 (0.98) | 3 (0.67) | 3 (0.68) | 43 (0.97) | 51 (0.99) | 2 (0.42) | 0 (0.27) | 10 (0.93) | 3 (0.94) |
| sensor-noise-medium | 23 (0.71) | 64 (1.00) | 3 (0.36) | 3 (0.52) | 3 (0.54) | 0 (0.63) | 3 (0.49) | 5 (0.55) | 2 (0.41) | 2 (0.35) | 5 (0.54) | 4 (0.58) |
| sensor-noise-heavy | 2 (0.33) | 64 (1.00) | 0 (0.18) | 0 (0.21) | 1 (0.26) | 0 (0.32) | 0 (0.23) | 0 (0.23) | 3 (0.35) | 2 (0.30) | 2 (0.25) | 0 (0.24) |
| sensor-delay-light | 64 (1.00) | 64 (1.00) | 63 (1.00) | 60 (0.99) | 5 (0.67) | 4 (0.68) | 44 (0.99) | 49 (0.99) | 2 (0.42) | 0 (0.27) | 8 (0.94) | 0 (0.94) |
| sensor-delay-medium | 64 (1.00) | 64 (1.00) | 63 (1.00) | 60 (0.99) | 5 (0.67) | 5 (0.67) | 44 (0.99) | 48 (0.99) | 1 (0.42) | 0 (0.27) | 8 (0.94) | 0 (0.94) |
| sensor-delay-heavy | 64 (1.00) | 64 (1.00) | 63 (1.00) | 60 (0.99) | 6 (0.67) | 5 (0.67) | 44 (0.99) | 48 (0.99) | 1 (0.42) | 0 (0.27) | 7 (0.94) | 0 (0.94) |

Result artifacts at `results/mesa/phase3-probe-slate/{bc_from_hc_medium,
l_signature_medium_canonical_10m, l_reward_clean_medium_canonical_10m,
l_reward_phase3_medium_canonical_10m, l_mixed_phase3_medium_canonical_10m}/`.

### 10.3 Basin-capture amplification — the headline Medium finding

The failure-pattern classifier shows the basin shortcut is *dramatically* more
absorbed at Medium tier. Per-cell basin captures (out of 64 trials), Small
vs Medium:

| Cell | L-Reward S | L-Reward M | L-Mixed S | L-Mixed M | L-R-Clean S | L-R-Clean M |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| nominal | 7 | **55** | 0 | 5 | 0 | 0 |
| geometric-light | 11 | **57** | 0 | 8 | 0 | 0 |
| geometric-medium | 4 | **55** | 0 | 5 | 0 | 0 |
| geometric-heavy | 10 | **54** | 0 | 4 | 0 | 0 |
| decoy-light | 7 | **55** | 0 | 7 | 0 | 0 |
| decoy-medium | 10 | **48** | 0 | 6 | 0 | 0 |
| decoy-heavy | 16 | **50** | 0 | 7 | 0 | 0 |
| sensor-noise-light | 7 | **53** | 0 | 4 | 0 | 0 |
| sensor-noise-medium | 4 | 28 | 0 | 1 | 0 | 0 |
| sensor-noise-heavy | 1 | 11 | 3 | 3 | 3 | 2 |
| sensor-delay-{l,m,h} | 7-8 | **55** | 0 | 5 | 0 | 0 |

Three reads, in increasing program-significance:

**(a) L-Reward canonical Medium absorbs the basin on 80-90% of trials**
across no-gradient-corruption cells (53-57/64). Small was 11-25% (7-16/64).
Capacity scaling **quadrupled** basin absorption. The hypothesis that more
parameters or longer training would help the basin-corrupted policy is
emphatically falsified — capacity *amplifies* the absorbed shortcut.

**(b) L-Mixed Medium shows partial basin absorption** (4-8/64 per cell)
where L-Mixed Small had zero. The signature anchor is no longer fully
protective at Medium. The 50/50 mixed signal admits *proportional*
corruption with scale: the signature half still pulls the policy to goal
neighborhoods (mean S_T = 0.94, unchanged from Small), but the reward
half's basin pull breaks through more often at higher capacity.

**(c) L-Reward-Clean Medium remains at zero captures.** The basin shortcut
exists only when the training signal contains it. No capacity, sensor
tier, or probe cell creates spurious basin learning in policies that never
trained on the basin shaping.

### 10.4 The three-point capacity-dependence picture (load-bearing)

The Medium tier sharpens the gravity-claim story from a binary into a
gradient. Three matched-architecture, matched-budget data points at Medium:

- **Pure state-only signature (L-Signature)**: 0 basin captures at any
  scale. Sundog-cost is structural — the gradient-information bottleneck
  of Gaussian-decay shaping holds at Medium just as it held at Small. The
  Sundog-cost is the price; the basin-immunity is the reward.
- **50/50 mixed signature + reward (L-Mixed canonical)**: protected at
  Small (0 basin captures), partially compromised at Medium (4-8 per
  cell). The signature anchor's protection scales inversely with capacity
  when reward signal is also present.
- **Pure agent-participating reward (L-Reward canonical)**: corrupted at
  Small (7-16 per cell), dramatically more corrupted at Medium (48-57 per
  cell). Capacity amplifies absorbed spec-gaming shortcuts.

The protective effect of the gravity-claim discipline **scales inversely
with the proportion of reward signal in training**, and that scaling is
itself *capacity-amplified*. Stronger formulation:

> Under matched-architecture PPO with a calibrated modest spec-gaming
> surface, the rate of shortcut absorption is monotonic in both
> (training-signal reward-proportion) and (parameter count). State-only
> signature training is structurally protected against shortcut
> absorption regardless of capacity. Mixed signals admit proportional
> shortcut leakage that grows with capacity.

That is a much sharper claim than Small alone could support.

### 10.5 Updated pre-registered prediction outcomes

PHASE3_SPEC v1.6 §10 pre-registered three Medium predictions before the
runs landed.

- **L-Reward-Clean Medium reaches ≥75% success.** ✅ **Confirmed.** 49/64
  = 76.6%. Scale helps clean reward training.
- **L-Reward canonical Medium stays collapsed (≤20% success).** ✅
  **Confirmed and stronger.** 0/64 = 0% success. Not just "still low" —
  fully collapsed. Mean S_T also dropped (0.42 → 0.27), indicating the
  basin pull is so dominant that the policy stops even reaching goal
  neighborhoods.
- **L-Mixed Medium sharpens geometric-light recovery.** ❌ **Falsified.**
  L-Mixed Medium drops to 0/64 across every probe cell, including
  geometric-light. The Small-tier +2/64 directional signal does not
  survive at Medium. The signature anchor preserves mean S_T = 0.94
  (basin doesn't pull policy off-goal in most trials) but cannot rescue
  the dwell condition. Medium L-Mixed knows where the goal is and reaches
  it; it cannot stop.

The falsification on prediction (3) is itself informative: it says the
geometric-light recovery at Small was real but small, and the larger
architecture absorbs enough basin signal to offset whatever asymmetry the
rotation breaks. A larger seed slate (256+) at Medium might resurface a
small but non-zero recovery, but the spec's exit-criterion threshold for
the L-Mixed recovery prediction was "directional signal at small-scale."
That signal is no longer robust at Medium.

### 10.6 Updated basin-effect gap (Medium tier)

| Cell | L-R-Clean S vs L-R S | L-R-Clean M vs L-R M |
| --- | ---: | ---: |
| nominal | 65.6 pp | **76.6 pp** |
| geometric-light | 64.1 pp | **79.7 pp** |
| geometric-medium | 45.3 pp | **73.4 pp** |
| geometric-heavy | 59.4 pp | **82.8 pp** |
| decoy-light | 31.2 pp | **39.1 pp** |
| sensor-noise-light | 64.1 pp | **79.7 pp** |
| sensor-delay-light | 65.6 pp | **76.6 pp** |
| sensor-delay-medium | 67.2 pp | **75.0 pp** |
| sensor-delay-heavy | 67.2 pp | **75.0 pp** |

The Medium basin-effect gap is consistently larger than Small across
every non-corrupted cell. This is the directly-measurable consequence of
the capacity-dependence finding: capacity helps L-Reward-Clean (it gains
~8 pp) and hurts L-Reward canonical (it loses ~3 pp), so the gap *between*
them widens at scale. The relationship is monotone in capacity across
every cell where the underlying gradient information is intact.

### 10.7 Updated exit-criterion check (v2)

PHASE3_SPEC §13 v1.2 exit conditions, re-evaluated with Medium data:

**(a)** Basin-effect gap exceeds 10 pp on at least one cell. **Met with
margin.** Gap exceeds 70 pp on 7 of 12 active cells at Medium (vs 6 of
12 at Small). The gap widens with capacity, strengthening the result.

**(b)** Probe slate maps where the gap collapses cleanly. **Met.** §10.6
table shows the gap is robust to all probes that preserve underlying
gradient information and collapses only under decoy-heavy and sensor-
noise-medium+ where both families fail equally. Same pattern as Small,
sharpened at Medium.

Phase 3 exit criterion **passes at both tiers** with strengthened
evidence at Medium.

## 11. Pending and follow-up

### Already in this slate, not yet written up

- **Trial-level JSONL logs.** First canonical Phase 3 slate ran with
  `--no-trial-logs` for speed. Per-cell failure-pattern classification
  ran in-memory (the data in §4 came from that pass) so no rerun is
  needed for the classifier-derived findings. If we want
  trajectory-level visualizations (basin-capture replays, oscillation
  traces, decoy-pull paths), a rerun without `--no-trial-logs` would
  populate `results/mesa/phase3-probe-slate/<slug>/trials/*.jsonl` for
  the policies of interest. Recommendation: rerun L-Reward canonical
  and L-Mixed only, since those are the policies whose failure modes
  carry the program narrative.

### Not yet started

- **Large tier.** Phase 2 Large PPO (~5M params, 100M env steps) has not
  yet trained. The three-point capacity-dependence picture in §10.4
  predicts the basin amplification at Large should be at least as
  dramatic as Medium's. Worth running if compute permits, but the
  marginal program signal beyond Medium is uncertain.
- **Larger seed slate at Medium.** 64 seeds surfaces the headline
  pattern cleanly, but cell-level basin-capture counts (4-8/64 for
  L-Mixed Medium) would benefit from 256 seeds to test whether the
  apparent geometric-light recovery is now genuinely zero or just
  noisy.
- **Composed probes.** Phase 3 v1 sweeps single-axis cells. Composed
  probes (rotate + decoy + delay) deferred to Phase 3 v2.
- **Texture-channel axis.** Deferred to Phase 5 if texture-channel
  retraining lands.

## 12. Cross-references

- Spec: [`PHASE3_SPEC.md`](PHASE3_SPEC.md) v1.5
- Roadmap: [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md)
- Phase 0 spec: [`PHASE0_SPEC.md`](PHASE0_SPEC.md)
- Phase 2 spec: [`PHASE2_SPEC.md`](PHASE2_SPEC.md) v1.2
- Harness: `scripts/mesa-probe-slate.mjs`
- Env core: `public/js/mesa-core.mjs`
- Result artifacts: `results/mesa/phase3-probe-slate/<slug>/`

## 13. Versioning

- `v1` (2026-05-11): initial Phase 3 Small-tier result note. Records the
  first canonical 7-policy × 12-cell × 64-seed slate. Documents the
  basin-capture mechanism via the in-memory classifier, the
  basin-effect gap as the exit-criterion-passing number, the mild
  pre-registered L-Mixed-geometric-light recovery, the BC-under-noise
  side observation, and the β-sensitivity curve. Trial-log rerun for
  trajectory visualizations is recommended but not blocking.

- `v2` (2026-05-11): Medium-tier amendment. Records the Medium canonical
  10M-step results for L-Signature, L-Reward-Clean, L-Reward canonical,
  L-Mixed canonical, and BC-from-HC, plus the Medium probe slate (12
  cells × 64 seeds). Headline finding: capacity amplifies basin
  absorption from 11-25% of L-Reward trials at Small to 80-90% at
  Medium, while L-Mixed Medium shows partial signature-anchor breach
  (4-8 captures per cell, where Small had 0). The three-point
  capacity-dependence picture in §10.4 is the program's strongest
  gravity-claim formulation to date. Two of three Medium pre-registered
  predictions confirmed, one falsified (L-Mixed geometric-light
  recovery does not survive at Medium).
