# H1.2c Reward-Asymmetric Cap — Binding Results

Status: **`H1_2C_NULL` — the pantheon tax was NOT a symmetric-cap artifact.**
Ran 2026-06-18. Spec: [`H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md`](H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md),
caps frozen (field 1.00 / reward 0.50 / guard 0.70) before the run. Reference
negative: [`H1_2B_RESULTS.md`](H1_2B_RESULTS.md).

Bounding the bull while leaving Sol uncapped did **not** recover the competence
H1.2b lost. The equal-budget monolith still wins, and the field-uncapped council
is, if anything, marginally *worse* than the H1.2b symmetric council.

## Configuration

- Same frozen heads, features, seeds, monolith baseline, and 13-cell slate as
  H1.2b. Only the action-blend cap geometry changed (symmetric 0.70 →
  reward-asymmetric 1.00/0.50/0.70), applied identically to the privileged
  best-mix **target** (dataset) and the arbiter projection (eval) via the shared
  `capSimplexProject`.
- Dataset 608 374 / 155 047 rows, 981 basin rollouts. Param budget matched
  (council 3428 vs M-Adapter 3432). Cap invariant held: max council
  `w_reward = 0.50` (= cap), `bull_breach = 0` (structurally bounded).
- H1.2c-a plumbing probe passed (cap recorded in all manifests, no leakage,
  gates computable) before this binding run.

## Result (vs H1.2b symmetric reference)

| controller | mean S_T | S_T (GI) | basin (GI) | field-relief | bull-breach |
| --- | --- | --- | --- | --- | --- |
| `P-Council-RA50` (field-uncapped) | 0.725 | 0.834 | 0.094 | **0.125** | 0.000 |
| `M-Adapter` (equal-budget monolith) | **0.795** | **0.911** | **0.056** | — | — |
| `Blind-Council-Sym70` (ref) | 0.702 | 0.804 | 0.074 | 0.000 | 0.203 |
| `P-Council-Sym70` (H1.2b ref) | 0.747 | 0.858 | 0.071 | (capped) | 0.039 |

Gates: 1 = **false** (cap-tax repair slate **−0.022**, GI **−0.024** — *negative*),
2 = **false** (0.725 < 0.795 − 0.05), 3 = **false** (M-Adapter has fewer GI
captures, 0.056 < 0.094), 4 = **true** (reward ≤ 0.50, no bull breach), 5 = **true**
(budget matched, no forbidden feature). Branch: **`H1_2C_NULL`**.

M-Adapter beats `P-Council-RA50` on **all 13 cells** (Δ −0.025 to −0.096),
uniformly — the same shape as H1.2b. Uncapping the field changed essentially
nothing in the competition.

## Why — the relief was available but the supervised arbiter did not use it

The cap geometry is not the bottleneck. The structural freedom is real (field
cap 1.00; the council *can* and sometimes does exceed 0.70 on field), but the
**supervised arbiter barely exercises it**, and not only on the corrupted cells:

| cell | GI | field-relief (w_field>0.70) | mean w_field | mean w_reward |
| --- | --- | --- | --- | --- |
| nominal | yes | 0.119 | 0.538 | 0.216 |
| geometric-light/med/heavy | yes | 0.11/0.08/0.11 | ~0.54 | ~0.21 |
| sensor-delay-light/med/heavy | yes | 0.12/0.11/0.10 | ~0.53 | ~0.22 |
| decoy-light/med/heavy | no | 0.13/0.16/0.16 | 0.58/0.59/0.63 | ~0.21 |
| sensor-noise-light/med/heavy | no | 0.15/0.04/0.005 | 0.54/0.46/0.35 | 0.22/0.26/0.32 |

Even on the **clean** gradient-intact cells — where the field is the correct
controller and the cap now permits `w_field → 1.0` — the arbiter puts >0.70 on
the field only ~12% of the time and sits at a cautious central blend
(~0.53 field / 0.22 reward / 0.25 guard). It correctly *reduces* field weight on
the genuinely corrupted sensor-noise cells (relief → 0.005), so the arbiter is
not broken — it is *hedging globally* because it cannot tell, from
non-privileged features alone, when the local field is trustworthy enough to
fully follow. The wider RA target is also harder to fit (arbiter CE 0.946 vs
H1.2b's 0.869), nudging the learned policy slightly *below* the symmetric
council.

**Conclusion:** uncapping the field is *necessary-not-sufficient*. It removes
the structural barrier the pantheon tax diagnosis identified, but the
**supervised** privileged-best-mix arbiter does not have the feature-based
discrimination to convert that freedom into competence. This is precisely the
`H1_2C_NULL` reading: *"the supervised arbiter cannot use the relief."*

## Consequence for the Tauroctony ledger

Two independent binding nulls now stand: H1.2b (symmetric cap) and H1.2c
(reward-asymmetric cap). At Small tier, with a **supervised** arbiter,
role-separation loses to an equal-budget monolith on competence and
proxy-capture **regardless of cap geometry**. The MESA-lane pantheon thesis
**remains [ORNAMENT]** (`SUNDOG_V_TAUROCTONY.md` §I falsifier).

What survived intact across both runs: the discipline keeps the **bull off the
throne** (H1.2c: reward authority structurally ≤ 0.50, zero bull breaches). The
unproven half remains governance: *we can bound the proxy; we have not shown the
bounded council out-governs a well-tuned monarch.*

Registered reopening rungs (each must be pre-registered before running; none is
a re-score):

- **H1.2d — RL-trained arbiter.** The supervised arbiter's inability to use the
  field relief is the named bottleneck. A reward-shaped (RL) arbiter that is
  optimized for terminal alignment directly — rather than regressed to a
  privileged target — is the spec-reserved amendment most likely to convert the
  relief into competence. Spec: [`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md).
- **Richer trust features / history** so the arbiter can identify a clean local
  field (the current 17 local features + short history were not enough).
- **Higher tier** (Medium/large), where the capacity gap to the monolith may
  differ.
