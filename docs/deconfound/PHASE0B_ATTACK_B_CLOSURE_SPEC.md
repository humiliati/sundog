# Deconfound Phase 0B - Attack-B Real-Feature Closure Spec

> 2026-06-04, **LOCKED + EXECUTED** -> `attack_b_closure_confirmed` (runner +
> `PHASE0B_ATTACK_B_CLOSURE_RESULTS.md` filed). This spec follows the passed Attack-B 0-pre screen and
> defines the first binding closure read on a de-confound-clean real-feature
> substrate.
>
> **Locked amendments (2026-06-04):** (1) added a read-dim-3
> *functional-keeper-compressed* capacity-disambiguation diagnostic (§2/§3/§6) so a
> stateful functional-keeper splits into a real objective null vs the chatv2-style
> capacity-ambiguous null; (2) made `closure_margin` well-defined when `k_func = none`
> (§5: map none to `Kmax+1` for both terms; primary `none->9`, compressed diagnostic
> `none->4`; clamp at 0); (3) made §6 branches exhaustive + ordered (precedence note;
> `_capacity_unresolved` also covers a learned-but-inconclusive compressed control).
> **LOCKED 2026-06-04** (operator-approved); run executed.

## 1. Decision Lock

Phase 0B tests whether an objective that targets a constructed functional keeps
that functional while discarding irrelevant state, compared with an objective
that must keep the state.

The substrate is frozen to the passed 0-pre cell:

- data: `sklearn.datasets.load_digits`, no download;
- feature map: 8x8 digit images pooled into `D=8` spatial blocks on a 4x2 grid;
- binarization: per-feature median threshold on the full digits table;
- primary functional: `u = XOR(b_0, b_1, b_2)`;
- state targets: the non-functional feature bits `b_j` for `j notin S`,
  where `S = [0, 1, 2]` and `j in {3,4,5,6,7}`.

0-pre receipt:
`results/deconfound/attack-b-0pre-digits-2026-06-04.txt`.

Relevant 0-pre rows:

| row | bitcorr | acc/base | det | flag |
| --- | ---: | ---: | ---: | --- |
| digits D=8, s=3 primary | 0.116 | 0.567/0.530 | +0.077 | HOLD |
| digits D=4, s=3 boundary | 0.113 | 0.536/0.511 | +0.050 | HOLD |
| digits D=8, s=2 sensitivity | 0.116 | 0.564/0.528 | +0.075 | HOLD |
| digits D=4, s=2 sensitivity | 0.113 | 0.580/0.579 | +0.003 | HOLD |

The Othello escape hatch is closed for the named slate:
`results/deconfound/othello-functional-screen-2026-06-04.txt`.

## 2. Bodies

Train two primary paired bodies on the same `b` rows, same train/validation/test split,
same primary body architecture, same optimizer budget, and matched initial body weights
per seed. Train one additional compressed diagnostic body on the same split and seeds; it is
consulted only when the primary functional-keeper is stateful.

**State-keeper**:

- objective: reconstruct all `D=8` input bits `b`;
- loss: mean binary cross-entropy over all `b_j`;
- expected pressure: keep the full real-feature state.

**Functional-keeper**:

- objective: predict `u = XOR(b_S)`;
- loss: binary cross-entropy on `u`;
- expected pressure: keep the functional and discard `b_j notin S`.

**Functional-keeper-compressed** (capacity-disambiguation diagnostic; consulted only to
interpret a stateful functional-keeper, per Section 6 — NEVER part of the headline):

- identical to the functional-keeper in objective, loss, optimizer, budget, seeds, split, and
  matched init, EXCEPT the body read layer is narrowed to read-dim `3`:
  `Linear(8,32) -> GELU -> Linear(32,3) -> GELU`, functional head `Linear(3,1)`;
- purpose: force the representation below the 8-bit input so the body cannot retain
  `b_j notin S` for free. If the functional-keeper is stateful at read-dim 8 only because it
  had room, closure re-appears here; if the objective genuinely does not discard, it persists.
  This mirrors the chatv2 capacity sweep that separated capacity-driven from objective-driven
  body content. Its determining-shadow read (Section 4) sweeps `k in {1,2,3}` (its body dim).

Architecture is intentionally small and vector-native:

| parameter | value |
| --- | --- |
| input dim | 8 |
| body | MLP, `Linear(8,32) -> GELU -> Linear(32,8) -> GELU` |
| body read layer | final 8-dimensional body activation |
| state head | `Linear(8,8)` logits |
| functional head | `Linear(8,1)` logit |
| optimizer | AdamW |
| learning rate | 1e-3 |
| weight decay | 1e-3 |
| max epochs | 500 |
| early stop | validation loss patience 40 |
| seeds | `{0,1,2,3,4}` |
| split seed | 20260604 |
| split | stratified train/validation/test by `u`, 60/20/20 |

If the eventual implementation changes any architecture or optimization value,
the spec must be amended before training.

## 3. Learned-Body Gates

No primary closure read is interpreted until the state-keeper and functional-keeper clear
their learning gates. The compressed diagnostic is interpreted only if its own learned gate
clears; if it fails, the primary read is not voided, but any stateful functional-keeper
branch becomes capacity-unresolved.

| gate | pass condition |
| --- | --- |
| functional-keeper learned | held-out `u` det >= 0.70 and accuracy >= 0.80 |
| state-keeper learned | held-out mean bit det over all `b_j` >= 0.70 and mean bit accuracy >= 0.80 |
| functional-keeper-compressed learned (diagnostic) | held-out `u` det >= 0.70 and accuracy >= 0.80; if it fails, the capacity diagnostic is `control_void_unlearned` and a stateful functional-keeper is reported `attack_b_functional_keeper_stateful_capacity_unresolved` (Section 6) |
| split sanity | train/validation/test `u` base rates within 0.08 of each other |
| de-confound replay | primary 0-pre row remains `det <= 0.20` |

Failure of a primary learned-body gate yields `closure_void_unlearned`, not a
scientific negative. Failure of only the compressed diagnostic gate is not a primary void.

## 4. Determining-Shadow Read

Read the final body activations with the carried-over determining-shadow-set
instrument.

For each trained body and seed:

- for each allowed `k`, select a size-`k` subset of body dimensions on the
  probe-training split;
  - primary state-keeper and functional-keeper: `k in {1,2,3,4,5,6,7,8}`;
  - functional-keeper-compressed diagnostic: `k in {1,2,3}`;
- fit linear probes from the selected body dimensions to:
  - `u` for `k_func`;
  - each outside-state bit `b_j`, `j notin S`, for `k_state`;
  - `u_null`, an independent Bernoulli target with the same held-out base rate
    as `u`, generated by `default_rng(20260604 + 1000 * seed)`;
- evaluate on the held-out probe split only.

Determination score:

```text
det = (accuracy - base_accuracy) / max(1 - base_accuracy, 1e-9)
```

Thresholds:

| quantity | definition |
| --- | --- |
| `k_func` | smallest `k` with held-out `det(u) >= 0.70` and selection-corrected `p <= 0.01` |
| `k_state` | smallest `k` with mean held-out `det(b_j notin S) >= 0.70` and selection-corrected `p <= 0.01` |
| `k_null` | same as `k_func`, but on `u_null`; must be `none` |
| `none` mapping | for margin arithmetic only, map `none` to `Kmax + 1`; primary bodies use `Kmax=8 -> 9`, compressed diagnostic uses `Kmax=3 -> 4` |

Selection-corrected null:

- preserve body activations and labels;
- repeat the same subset-selection procedure on 1,000 random-label permutations;
- compute one-sided `p = (1 + count(null_score >= observed_score)) / 1001`;
- the observed subset must clear both the det threshold and `p <= 0.01`.

The read is invalid if the selected subset was chosen on held-out rows.

## 5. Headline Statistic

For each body:

```text
closure_margin = max(0, mapped(k_state) - mapped(k_func))
```

where `mapped(none) = Kmax + 1` for BOTH `k_func` and `k_state` (consistent with the
Section 4 mapping). For the headline primary bodies, `Kmax=8`, so `none -> 9`. A body with
`k_func = none` has no functional and scores `closure_margin = 0`. A body has a closure
bracket when:

```text
k_func exists
k_state is none OR k_state >= k_func + 2
k_null is none
```

The headline comparison is paired by seed:

```text
keeper_gap = closure_margin(functional_keeper) - closure_margin(state_keeper)
```

Primary pass requires:

- functional-keeper closure bracket in at least 4/5 seeds;
- state-keeper closure bracket in at most 1/5 seeds;
- median paired `keeper_gap >= 2`;
- `u_null` negative in both primary bodies for all interpreted seeds.

## 6. Branches

| branch | condition | reading |
| --- | --- | --- |
| `attack_b_closure_confirmed` | all gates pass and primary pass clears | objective-targeted functional keeping produces a closure bracket on real features, while state keeping does not |
| `attack_b_objective_gap_absent` | gates pass but median paired `keeper_gap < 2` | the objective contrast does not produce a measured closure gap on this substrate |
| `attack_b_functional_keeper_stateful_objective` | functional-keeper stateful (`k_state <= k_func + 1` in >=4/5 seeds) AND the read-dim-3 compressed control is ALSO stateful in >=4/5 seeds | the functional objective genuinely does not discard outside state even under forced compression — a real objective null, not capacity |
| `attack_b_functional_keeper_stateful_capacity` | functional-keeper stateful, but the compressed control brackets in >=4/5 seeds | the read-dim-8 body merely had room to keep `b_j notin S`; closure appears once compression is forced. A capacity-ambiguous null, NOT a clean objective null — re-pose at a smaller primary read-dim before any closure reading |
| `attack_b_functional_keeper_stateful_capacity_unresolved` | functional-keeper stateful, but the compressed diagnostic is `control_void_unlearned` OR learned-but-inconclusive (neither stateful nor bracketing in >=4/5 seeds) | cannot adjudicate objective null vs capacity artifact; repair/re-pose the compressed control before making the objective claim |
| `attack_b_state_keeper_closure_like` | state-keeper has a closure bracket in >=2/5 seeds | the contrast is contaminated; state keeping also yields closure-like structure |
| `closure_void_unlearned` | learned-body or split gate fails | no scientific result |
| `closure_void_deconfound_replay` | 0-pre replay exceeds `det > 0.20` | no result; substrate no longer de-confound-clean |
| `closure_void_control` | `u_null` clears in any interpreted body/seed | no result; instrument hallucinated a functional |

**Branch precedence (first match wins):** (1) void branches (`closure_void_unlearned`,
`closure_void_deconfound_replay`, `closure_void_control`); (2) `attack_b_state_keeper_closure_like`;
(3) the `attack_b_functional_keeper_stateful_*` family; (4) `attack_b_closure_confirmed` /
`attack_b_objective_gap_absent`. A run matching an earlier tier is not also reported under a
later one.

## 7. Integrity Boundaries

Allowed on pass:

> On median-binarized sklearn digit features, a supervised functional objective
> produced a closure bracket (`k_func << k_state`) that the state-reconstruction
> objective did not, under a passed input-deconfound precheck and an independent
> `u_null` control.

Forbidden:

- claiming the functional was model-discovered; it is constructed;
- claiming Othello closure was rescued;
- claiming real JEPA behavior; this is a JEPA-principle contrast, not a JEPA
  implementation;
- claiming R2/"more than we know"; ceiling is R1.5 until an external/reviewed
  real-model variant exists.

## 8. Run Order

1. Operator lock review.
2. Implement `scripts/deconfound_attack_b_phase0_closure.py` additively.
3. Add npm aliases only after the script exists.
4. Run:

```powershell
python -m py_compile scripts/deconfound_attack_b_phase0_closure.py
python scripts/deconfound_synth_b_datacheck.py
python scripts/deconfound_attack_b_phase0_closure.py --phase attack-b-phase0-closure --out results/deconfound/attack-b-phase0-closure
```

5. If a timing smoke estimates runtime above the repo's ~10-minute inline rule,
   stop and stage the exact PowerShell command for the operator instead of
   running the full measurement inline.
6. Record verdict in
   `docs/deconfound/PHASE0B_ATTACK_B_CLOSURE_RESULTS.md`.

## 9. Readback Checklist

- git SHA and script SHA;
- 0-pre replay row and receipt path;
- per-seed learned-body gates;
- per-seed `k_func`, `k_state`, `k_null`, `closure_margin`;
- functional-keeper-compressed (read-dim 3) gates + per-seed `k_func`, `k_state`, `k_null`, and bracket (capacity diagnostic, consulted only on a stateful functional-keeper);
- paired `keeper_gap` table;
- selection-corrected null summary;
- branch verdict;
- allowed/forbidden language copied from Section 7.

---

*Sundog Research Lab - Deconfound Phase 0B Attack-B closure spec. Locked + executed
2026-06-04 -> attack_b_closure_confirmed.*
