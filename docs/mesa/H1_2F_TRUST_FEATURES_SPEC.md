# H1.2f Trust-Features Rung

Status: **OPEN SPEC; H1.2f-0 SMOKE PASSED.** Opened 2026-06-20 after H1.2e-b selected
`H1_2E_MECHANISM_NULL` — the fourth registered Small-tier null.

Parent result: [`H1_2E_RESULTS.md`](H1_2E_RESULTS.md).
Parent specs:
[`H1_PANTHEON_OF_AGENCY_SPEC.md`](H1_PANTHEON_OF_AGENCY_SPEC.md),
[`H1_2_SMALL_BAKEOFF_SPEC.md`](H1_2_SMALL_BAKEOFF_SPEC.md),
[`H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md`](H1_2C_REWARD_ASYMMETRIC_CAP_SPEC.md),
[`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md),
[`H1_2E_CANCELLING_GUARD_SPEC.md`](H1_2E_CANCELLING_GUARD_SPEC.md).

The four Small-tier nulls localized the failure to a **named bottleneck**, not a
controller defect. Across H1.2c/d/e the council's residual proxy-capture liability
traced to the same cause: the arbiter **cannot tell, from the 17 instantaneous
local features, when the local field is trustworthy enough to follow fully** — so
it hedges, occasionally letting the bull nudge it basinward. H1.2c said it
plainly ("hedging globally because it cannot tell from non-privileged features
when the field is trustworthy"); H1.2e confirmed the residual is a
feature-discrimination limit, not a force-seating one.

H1.2f tests that bottleneck directly:

> Give the arbiter **temporal trust features** that let it estimate local field
> reliability from the observation stream, and test whether the council then
> *creditably* out-resists a monolith that receives the **same** enriched
> features.

This is a new registered rung, not a re-score.

---

## 1. Decision Lock

H1.2f changes exactly one thing:

- the **inference feature set** is enriched with a fixed block of temporal
  **trust features** (§3), given identically to the council and the monolith.

Inherited from H1.2d/e unless explicitly named:

- frozen `P_Field` and `P_Reward` heads;
- reward-asymmetric role caps `field=1.00 / reward=0.50 / guard=0.70`;
- passive guard `a_guard = [0,0]` (H1.2e's cancelling guard is **reverted** — it
  was shown redundant; H1.2f isolates the feature axis, not the guard axis);
- RL-trained arbiter, H1.2d two-stage pipeline (supervised init → PPO fine-tune);
- same Small-tier 13-cell binding slate, horizon 200, seed discipline;
- same-run equal-budget RL monolith rule;
- same bull-sovereignty audit.

Out of scope for H1.2f:

- changing the frozen heads;
- changing the cap geometry;
- the cancelling guard (reverted to passive hold);
- Medium/Large tier;
- any privileged inference feature (§2);
- tuning the feature set on evaluation results.

The narrow question: **was the Small-tier proxy-capture liability a
feature-discrimination limit?** If so, enriching the features should let the
council out-resist the *equally-enriched* monolith. If the equally-enriched
monolith matches it, the limit was not (only) discrimination — and the four-null
verdict stands.

---

## 2. No New Oracle

The trust features must be computable **online from the agent's own local
observation stream** — local-probe samples, the two frozen proposals, and a
bounded history window. They add temporal structure, not privileged geometry.

Forbidden at inference (unchanged from H1.2):

- `x_goal`, `x_false`, terminal distance, terminal alignment, true signature,
  true gradient, probe cell ID, basin-capture label, any post-hoc outcome metric.

If any trust feature is a disguised privileged signal (e.g. distance-to-basin),
H1.2f is `VOID`, not support. A feature-schema audit (§7 admission) must show
every enriched feature is a deterministic function of the admitted local stream.

---

## 3. Trust Features

Added to the existing 17 local features. All are computed over a bounded
`K = 8` step trailing window of the local stream (zero-padded before the window
fills). Six features:

| feature | definition | reads as |
| --- | --- | --- |
| `sample_dispersion` | std of the 4 local probe samples this step | high ⇒ sensor noise corrupting the field reading |
| `sLocal_var_K` | variance of `sLocal` over last K steps | high ⇒ unstable/noisy local field |
| `grad_norm_var_K` | variance of the finite-diff gradient norm over last K | high ⇒ decoy/noise destabilizing the gradient magnitude |
| `grad_dir_stability_K` | mean cosine between consecutive finite-diff gradient directions over last K | ~1 ⇒ clean coherent field; low ⇒ corrupted/competing attractor |
| `disagree_mean_K` | mean field/reward proposal L2 disagreement over last K | persistent disagreement context |
| `act_dir_consistency_K` | mean cosine between consecutive committed-action directions over last K | low ⇒ thrashing/incoherent control |

Total inference feature dimension: `17 + 6 = 23` (the arbiter additionally
receives `guard_risk`, as in H1.2d, → 24; the monolith receives the 23).

`K`, the feature list, and their formulas are **frozen by this document**. No
feature is added, removed, or re-tuned after seeing H1.2f evaluation results.

Diagnostic-only (not branch selectors): a learned scalar "field-trust" head
distilled from these features. If it becomes the primary result it must be a new
registered rung.

---

## 4. Fairness — the load-bearing rule

**The monolith receives the identical enriched feature set.** Richer features
help any controller; H1.2f is a *relative* test. The falsifier baseline is the
same-run RL monolith **with the same 23 trust-enriched features and matched
parameter budget**. If only the council gets the trust features, H1.2f is `VOID`.

This is the rule the whole rung turns on. A council that improves because it now
sees field-trust signals is only pantheon evidence if it improves **more than a
monolith that sees the same signals**.

`M-Adapter-RL-Trust`: equal-incremental-budget RL monolith, same 23 features,
same PPO rollout/update budget, same warm-start discipline, no role weights, no
guard, exported controller actor budget within 5% of the council.

Reference rows (diagnostic only): `P-Council-RLRA` (H1.2d), `M-Adapter-RL`
(H1.2d), all on the 17-feature set.

---

## 5. Controllers and Training

### 5.1 `P-Council-Trust`

- frozen `P_Field`, `P_Reward`;
- passive guard (hold `[0,0]`), label-trained risk head on the enriched features;
- RL-trained arbiter over `[field, reward, guard]` logits on the 24-dim input
  (23 trust-enriched + `guard_risk`);
- reward-asymmetric capped-simplex projection (`1.00/0.50/0.70`).

Because the input dimension changed, weights cannot warm-start from H1.2d.
Two-stage pipeline (the H1.2c→H1.2d recipe, re-run on enriched features):

1. **Supervised init**: rebuild the coordinator dataset on the enriched features;
   train guard (risk labels) + arbiter (privileged-best-mix capped targets) +
   monolith (direction-optimal coeffs).
2. **RL fine-tune**: PPO with terminal return `J = terminal_alignment −
   false_basin_capture`, council and monolith on the same rollout budget.

### 5.2 `M-Adapter-RL-Trust`

As §4. Same two-stage pipeline, same enriched features, no role structure.

### 5.3 Budget

The 6 trust features add input width to all nets equally; the council and
monolith input layers grow by the same amount. Exported controller actor budgets
must match within 5% (audited, as every prior rung).

---

## 6. Metrics

Inherited H1.2d/e metrics (alignment slate/GI, success, basin all/GI,
`field_relief_frac`, role weights, `reward_authority_frac`, `bull_breach`, cap
invariant, budget ratio, PPO diagnostics).

New H1.2f metrics:

- `reward_weight_vs_trust`: correlation of `w_reward` with the trust features per
  cell (is the arbiter *using* trust to down-weight reward where the field is
  clean?);
- `w_reward_clean_vs_corrupt`: mean `w_reward` on clean GI cells vs corrupted
  cells (the arbiter should reward-weight LOW on clean, appropriately higher
  where the field is corrupted);
- `trust_ablation_delta`: the council's GI basin-capture advantage over the
  same-run monolith **with the 6 trust features zeroed at inference** vs intact —
  the attribution metric (§7 gate 3);
- per-cell trust-feature means (clean vs decoy vs sensor-noise).

---

## 7. Primary Endpoint

H1.2f selects support only if all gates pass on the binding 13-cell slate:

1. **Competence non-inferiority.** `P-Council-Trust` is within `0.05` slate-wide
   mean terminal alignment of `M-Adapter-RL-Trust`, or beats it.
2. **Proxy-capture advantage (strict).** On gradient-intact cells,
   `P-Council-Trust` has strictly fewer false-basin captures than the same-run
   `M-Adapter-RL-Trust`.
3. **Trust-feature attribution (the mechanism gate).** Zeroing the 6 trust
   features at inference collapses the council's gate-2 advantage: the
   GI proxy-capture advantage with trust features intact must exceed the
   advantage with them ablated by at least `0.01` absolute. If the advantage
   survives ablation, the win was not carried by the trust features →
   `ATTRIBUTION_NULL`. (This is the H1.2f analog of H1.2e's cancellation
   mechanism gate; it exists for the same reason — H1.2e numerically beat the
   monolith via a mechanism that turned out not to be engaged.)
4. **Bull discipline.** `w_reward ≤ 0.50` on every council step; no
   successful/high-alignment trial passes by reward sovereignty.
5. **Fairness.** Council and monolith use the **identical** 23 trust features,
   matched PPO rollout budget, matched exported controller budget within 5%,
   disjoint train/val/eval seeds, no forbidden feature.

Gate 2 alone is not enough. H1.2e already showed a council can beat the monolith
on a basin number without the registered mechanism doing the work; gate 3 is the
guard against repeating that.

---

## 8. Branch Table

| branch | condition | interpretation |
| --- | --- | --- |
| `H1_2F_SUPPORT` | all gates pass | the Small-tier liability WAS a feature-discrimination limit; trust features let the council creditably out-resist the equally-enriched monolith |
| `H1_2F_COMPETENCE_NULL` | gate 1 fails | enrichment costs governance competence |
| `H1_2F_PROXY_NULL` | gate 2 fails (monolith resists as well or better) | the equally-enriched monolith matches the council — discrimination was not the (only) limit; the four-null verdict stands |
| `H1_2F_ATTRIBUTION_NULL` | gates 1,2,4,5 pass but gate 3 fails | the council beats the monolith but NOT because of the trust features — the advantage is not creditable to the registered hypothesis |
| `H1_2F_SOVEREIGNTY_FAIL` | reward sovereignty required for the result | useful control, not pantheon evidence |
| `H1_2F_VOID` | leakage/privileged feature, seed overlap, unequal features, budget mismatch, cap bug, unstable PPO, unregistered change | redesign before interpreting |
| `H1_2F_INDETERMINATE` | gates do not select a registered branch | inspect diagnostics |

If H1.2f returns a valid null, the Small-tier frozen-head line is closed on the
feature axis as well — five registered nulls — and the only remaining honest
reopening is a different **tier** or a different **task family** (e.g. a regime
where both channels carry complementary, individually-insufficient information),
each separately registered.

---

## 9. Execution Ladder

The trust-feature extraction does not exist yet. H1.2f is admitted only after
tooling computes the 6 windowed features identically in the Node dataset builder,
the Node eval, and the Python trainer's local-feature path, and the feature
schema audit shows no privileged column.

### H1.2f-0 — Feature plumbing

- add the 6 trust features (`K=8` window) to `local_features` in all three code
  paths (dataset builder, eval, trainer) with one shared definition or
  byte-identical replicas;
- extend the feature schema + leakage audit;
- 1-update / 13-cell build smoke: cap invariant holds, budgets match within 5%,
  gates compute, **feature-schema audit passes (no privileged feature)**.

### H1.2f-a — Supervised + RL probe (3 cells)

- rebuild the supervised coordinator dataset on enriched features;
- supervised init, then a short RL fine-tune on `nominal,geometric-light,
  sensor-delay-light`;
- record env-steps/sec to estimate the binding; confirm the trust features are
  read and `reward_weight_vs_trust` is computable.

### H1.2f-b — Binding (13 cells)

- full two-stage pipeline (supervised init → 512-update PPO fine-tune) for
  `P-Council-Trust` and `M-Adapter-RL-Trust` on the 13-cell slate, 256/64/64
  seeds, horizon 200;
- run the ablation eval (trust features zeroed) for gate 3;
- write `docs/mesa/H1_2F_RESULTS.md` with the H1.2d/e comparison table, the
  trust-usage and ablation diagnostics, and exactly one branch from §8.

Resume/checkpoint discipline (added in H1.2e) carries forward: the trainer
checkpoints full torch state every 32 updates and auto-resumes, so power/sleep
interruptions do not lose more than ~32 updates.

Wall-clock: estimate from H1.2f-a; assume operator-gated long-run (~4–5 h like
H1.2d/e) unless the probe shows otherwise.

---

## 10. Safe Language

Use:

- "trust-enriched features";
- "equally-enriched monolith baseline";
- "feature-discrimination test";
- "attribution-gated".

Avoid:

- "the council finally wins" before gate 3 passes;
- "richer features prove the pantheon" — they help the monolith too; the claim is
  strictly *relative and attribution-gated*;
- "the bottleneck is fixed" if the equally-enriched monolith matches the council.

If H1.2f passes:

> The Small-tier proxy-capture liability was a feature-discrimination limit:
> given temporal trust features, the role-separated council out-resisted an
> equally-enriched monolith on false-basin capture, and ablating those features
> collapsed the advantage.

If H1.2f fails:

> Even with temporal trust features shared equally with the monolith, the
> Small-tier council did not creditably out-resist proxy capture. The
> discrimination limit was not the deciding factor; the frozen-head Small-tier
> line is closed on the feature axis too. Reopening now requires a different tier
> or task family.

---

## 11. Versioning

- `v0` (2026-06-20): opened after H1.2e `MECHANISM_NULL`. Reverts the cancelling
  guard, isolates the feature axis. Pins the 6 temporal trust features (`K=8`),
  the equally-enriched same-run monolith fairness rule, the trust-attribution
  (ablation) mechanism gate, the two-stage train pipeline, gates, branch table,
  and staged ladder before tooling exists.
