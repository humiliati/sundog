# H1.2b Binding Bake-Off — Results

Status: **NULL — pantheon thesis NOT supported at small-tier; the
equal-budget monolith wins across the operating envelope.** Ran 2026-06-18.
Spec: [`H1_2_SMALL_BAKEOFF_SPEC.md`](H1_2_SMALL_BAKEOFF_SPEC.md) §9 H1.2b, v0.1
gates frozen before this run.

This is the **binding** test. The indicative H1.2a capped probe (3 easy
gradient-intact cells) returned `H1_2_SUPPORT`; at full size across all 13 cells
that result **did not replicate**. The indicative→binding discipline did its
job.

## Configuration

- Dataset: 256 train / 64 val seeds × 13 cells (nominal + 12 active Phase 3
  single-axis cells), 608 374 train / 155 047 val rows, 981 basin-capture
  rollouts. 74.7 s, 10 220 rows/s.
- Trainer: param budget matched (council 3428 vs M-Adapter 3432, ratio 1.001);
  guard val AUC 0.825; arbiter CE 0.869; M-Adapter MSE 0.117.
- Eval: 64 seeds × 13 cells × 3 controllers = 2496 trials, 32.7 s; cap held on
  every step; no leakage.

## Result (3-cell-slate → full-envelope)

| controller | mean S_T | success | basin (all) | basin (GI) | bull-breach frac |
| --- | --- | --- | --- | --- | --- |
| Learned-P-Council | 0.747 | 19% | 0.102 | 0.071 | 0.039 |
| **M-Adapter** (equal-budget monolith) | **0.803** | 29% | **0.065** | **0.036** | — |
| Blind-Council | 0.702 | 6% | 0.123 | 0.074 | 0.203 |

Gates v0.1: 1 = **false** (gap-closure 0.152 < 0.40), 2 = **false**
(0.747 < 0.803 − 0.05), 3 = **false** (monolith has *fewer* basin captures on
gradient-intact cells, 0.036 < 0.071), 4 = **true** (bull restrained).

Mechanical branch (gate-1 precedence): `H1_2_ARBITER_NULL`. **Substantive
reading is stronger:** gates 2 and 3 *also* fail — the equal-budget monolith
**matches-or-beats the council on both competence and proxy-capture across the
operating envelope**, which is exactly the parent **H1 falsifier** condition
(`SUNDOG_V_TAUROCTONY.md` §I). The child branch table lacks an explicit
"monolith-wins" row; that is a spec-refinement note, not grounds to relabel
after the fact.

## Per-cell evidence (council − M-Adapter alignment)

The monolith wins on **12 of 13 cells** (decoy-heavy ties, +0.004); the gap is
remarkably uniform (~−0.05 to −0.09) on the gradient-intact cells where the
field head should dominate:

| cell | GI | council S_T | M-Adapter S_T | Δ |
| --- | --- | --- | --- | --- |
| nominal | yes | 0.865 | 0.931 | −0.066 |
| geometric-light/med/heavy | yes | 0.87/0.83/0.86 | 0.92/0.90/0.91 | −0.05/−0.08/−0.05 |
| sensor-delay-light/med/heavy | yes | 0.86/0.86/0.85 | 0.93/0.93/0.93 | −0.07/−0.07/−0.07 |
| decoy-light/med/heavy | no | 0.84/0.73/0.54 | 0.88/0.75/0.54 | −0.04/−0.02/+0.00 |
| sensor-noise-light/med/heavy | no | 0.82/0.47/0.30 | 0.91/0.55/0.36 | −0.09/−0.08/−0.06 |

## Diagnosis — the "pantheon tax" (structural, not a bug)

The council and M-Adapter regress to the **same** direction-optimal target
(`α*`, the field/reward split best matching the true gradient). The difference
is the role-separation discipline:

- **M-Adapter** outputs the blend coefficients directly and uncapped — when the
  field is right (`α* → 1`), it follows the field fully.
- **P-Council** must pass `α*` through softmax → **0.70 cap** → renormalize.
  When the field is the correct value, the cap forbids `w_field > 0.70`, forcing
  **~0.30 of contaminating reward pull** onto a field-dominant step. The
  guard-brake (`hold`) further shrinks the action. Both reduce convergence
  within horizon 200.

The ~0.07 alignment deficit on the easy gradient-intact cells (where field
*should* win cleanly) is the signature of this forced contamination. The thesis
cost is real and interpretable: **refusing to let any single value become
sovereign means you cannot fully follow even the *correct* value (the
field/Sol) when it is right** — and in this task that tax is *not* repaid by
proxy-resistance, because the unconstrained monolith resisted the false basin
*better*, not worse (0.036 vs 0.071 GI capture).

The one place the discipline shows value: the learned arbiter keeps the **bull
(reward) head** off the throne far better than the blind blend (bull-breach
0.039 vs 0.203). But sovereignty discipline without a competence or
proxy-resistance win is not pantheon support.

## Consequence for the Tauroctony ledger

Per the H1 falsifier in `SUNDOG_V_TAUROCTONY.md` (§I, Pantheon Thesis): *"If
matched-capacity monolithic controllers resist the MESA proxy-splitting battery
as well as, or better than, the pantheon-structured controller across the
operating envelope, the design stance is wrong and 'assemble a pantheon' is
demoted to [ORNAMENT]."* **That condition is now met at small-tier.** The
pantheon thesis is demoted toward **[ORNAMENT] for the MESA lane**, banked as a
negative — with these scoped caveats keeping it from a blanket kill:

1. **One arbiter design.** This is the supervised privileged-best-mix arbiter.
   The spec (§5.2) reserved an RL fine-tune as a registered amendment; it has
   not run. A differently-trained arbiter could, in principle, beat the
   monolith.
2. **The cap is a free hyperparameter.** The 0.70 symmetric cap is what imposes
   the tax. A reward-asymmetric *structural* cap (bound only the bull, leave the
   field uncapped) was adopted for the *audit* (v0.1) but not for the
   action-blend cap — applying it there is an untested H1.2c direction.
3. **Small-tier, frozen heads.** Medium/large tiers and a from-scratch matched
   pantheon are untested.

None of these are run here. The honest banked result stands: **at H1.2
small-tier, role-separation lost to an equal-budget monolith on both competence
and proxy-capture.** Reopening requires a registered H1.2c (asymmetric blend cap
or RL arbiter) or a higher tier — not a re-score of this run.
