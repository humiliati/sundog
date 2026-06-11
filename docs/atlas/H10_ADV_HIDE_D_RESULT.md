# H10 RESULT — ADV-HIDE-D: KILL (clean strengthening null, with a certified existence gap)

**Run:** 2026-06-11 (prereg frozen same day at `c964031c`, before any code; apparatus + non-verdict
smoke at `9494969e`). **Prereg:** `docs/atlas/H10_ADV_HIDE_D_PREREG.md`. Slate HS3.
**Outcome: KILL — the pre-registered clean strengthening null.** No deviation; all gates passed;
the full run (120 epochs, N_TRAIN 8000, N_PROBE 2000, eval λ ∈ {0, 1, 2, 3}, wall 6634 s) decided.

---

## Headline

**The determine half is robust to adversarial incentive within the GRL attack class — on a
substrate where hide-with-retention provably EXISTS.** Every guarded trained adversary (A1/A2 at
λ_adv ∈ {0.1, 1.0}, hide-enabled substrate) left best-of-3-probe pooled d-accuracy **≥ 0.996 at
every eval λ** (line: 0.90). Meanwhile the analytic encoder A3 — the trained-body instantiation of
the machine-checked `pooled_channel_blind` fiber construction (sundogcert HS7) — hid d essentially
perfectly on the same substrate: pooled d-acc **[0.502, 0.534]** across eval λ under the same
three probes, with per-unit retention **0.739** (single-unit Bayes ceiling ≈ 0.748) and an intact
c-pathway (R² 0.845, |Δ| from its baseline = 0.000). **Existence ≠ trainability**: gradient
training under explicit hide pressure never finds the antisymmetric blind subspace; what it does
instead is destroy competence — at λ_adv = 10 the primary task collapses (train fit −0.61, primary
R² 0.840 → 0.560) while d stays at 0.981.

H3's banked asymmetry therefore upgrades: determine was "robust to indifference" (banked), and is
now "robust to adversarial incentive **within the named attack class** (GRL vs per-epoch-retrained
pooled probe, λ_adv ≤ 10, this architecture and training budget)" — never more than that, per the
prereg scope clause; A3 proves unconditional concealment-impossibility claims would be FALSE.

## The frontier (the banked artifact)

Hide-enabled substrate (baselines: primary R² 0.840/0.841; retention 0.700/0.747; b0 pooled d
0.993–1.000):

| arm | λ_adv | primary R² | retention | guarded | min pooled d-acc |
|---|---|---|---|---|---|
| A1 | 0.1 | 0.842 | 0.721 | ✓ | **0.998** |
| A1 | 1.0 | 0.825 | 0.712 | ✓ | **1.000** |
| A1 | 10 | 0.560 | 0.690 | ✗ (primary) | 0.981 |
| A2 | 0.1 | 0.841 | 0.745 | ✓ | **1.000** |
| A2 | 1.0 | 0.832 | 0.744 | ✓ | **1.000** |
| A2 | 10 | 0.742 | 0.737 | ✗ (primary) | 1.000 |
| A3 analytic | — | 0.845 | 0.739 | (demo) | **0.502** |

Frozen-substrate control (symmetry prior: exchangeable φ cannot cancel the d-odd component
positionally): **prior CONFIRMED — no guarded hide.** The one large d-suppression seen anywhere
(frozen A1_1.0: pooled d 0.574) is **hide-by-destroying**: its retention collapsed to 0.605 and
its primary to 0.533 — both guards fail, and the prereg pre-named that case as deciding nothing.
At λ_adv = 10 on the frozen substrate the same pattern: guards fail, d still 0.83–0.91.

## Gates and demo checks (all pre-registered)

- **C0/C1 (hide substrate):** raw mean washes c at λ=2 (R² 0.000); single unit carries c (0.999).
- **DET:** β0 pooled d-acc ≥ 0.95 both substrates (min 0.991).
- **ABORT gate (frozen test, pre-run):** GRL training byte-reproducible (exact, max|diff| = 0);
  antisymmetric cancellation identity 1.4e-17; v2 module untouched (import-only; the concurrent
  H3-PC session's apparatus byte-equal — its own frozen test 10/10 re-verified on this machine).
- **A3 pinned predictions:** pooled ∈ [0.45, 0.55] → observed [0.502, 0.534] ✓; retention ≥ 0.70 →
  0.739 ✓; c-R² within 0.05 of β0_rffonly → Δ 0.000 ✓. The d-free c-pathway control:
  β0_rffonly's own pooled d-acc 0.510 (chance), so A3's concealment is carried entirely by the
  closed-form antisymmetric channel.
- **Smoke honesty note:** the non-verdict smoke had shown a guarded dip (A1_0.1 ≈ 0.87–0.88, both
  substrates). It did not survive full training (0.996–0.998) — an undertrained-probe-equilibrium
  artifact, which is exactly why smoke runs are pre-registered as non-verdict.

## What this banks (and what it does not)

Banked: the first adversarial-objective test in the tower, aimed at the robust half; the
λ_adv-resolved competence frontier (hide pressure buys NO concealment and costs primary R²
monotonically); the certified existence gap (A3 ≈ 0.50 vs trained ≥ 0.996 on the same substrate,
same probes); the frozen-substrate symmetry prior confirmed under attack. NOT banked: any claim
beyond the attack class (stronger attack families — e.g. initializing at A3 and fine-tuning,
distribution-matching adversaries, longer budgets — are the natural follow-on); any claim about
the determine THEOREM (untouched — it concerns the raw averaged shadow, not trained bodies).

Consistent with Elazar & Goldberg 2018 (adversarial removal leaks under retrained probes), now
with the sharper structure: the leak persists even where exact concealment is constructible, and
the failure mode under pressure is competence destruction, not hiding.

## Frozen test

`scripts/test_shadow_adv_hide_d.py` — **17/17** (12 apparatus/ABORT-gate checks + 5 banked pins
against the committed verdict JSON). Verdict artifact:
`results/atlas/h10/adv_hide_d_result.json` (full per-arm, per-λ, per-probe table); smoke record
preserved at `results/atlas/h10/adv_hide_d_result_smoke.json`.

## Follow-on (owner-gated, not this run)

Stronger-attack escalation, in order of bite: (1) warm-start a trained arm AT the A3 solution and
test stability under the joint objective (does training move OFF the blind subspace?); (2) a
distribution-matching adversary (MMD/critic on pooled reps) instead of a classification probe;
(3) multi-class d / asymmetric per-unit noise variants from the slate's optional list.
