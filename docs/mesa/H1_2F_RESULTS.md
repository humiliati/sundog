# H1.2f Trust-Features — Binding Results

Status: **`H1_2F_SUPPORT` — the first typed pantheon support in the H1 arc.**
Ran 2026-06-20→22 (training to 512 across several operator-run resume sessions;
final eval 2026-06-22). Spec:
[`H1_2F_TRUST_FEATURES_SPEC.md`](H1_2F_TRUST_FEATURES_SPEC.md), gates frozen
before the run. All five pre-registered gates pass.

After four registered Small-tier nulls (H1.2b/c/d/e), H1.2f tested the
repeatedly-named bottleneck directly: give the arbiter **temporal trust
features** so it can tell when the local field is trustworthy, hand the *same*
features to the monolith, and require the council's win to be **carried by**
those features (ablation gate). It cleared every gate.

## Configuration

- Same frozen heads, reward-asymmetric caps (1.00/0.50/0.70), passive guard
  (H1.2e's cancelling guard reverted), 13-cell slate, 256/64/64 seeds. The one
  change: 6 temporal `K=8` trust features (§3 of the spec) added to the inference
  set, **identically for council and monolith**.
- Two-stage pipeline: supervised init on the 23-feature dataset → PPO fine-tune
  (512 updates, 10.6M env steps). Budget ratio 1.0105 (within 5%); feature audit
  clean (no privileged feature); `max_w_reward = 0.50` (bull bounded, zero
  breaches). Node↔Python trust-feature parity pre-verified to 6e-8.
- Trainer's torch checkpoint/resume carried the run across ~5 harness/network
  interruptions (root cause: the harness-`claude.exe`-coupled background job, not
  the rig or the script); the operator completed it in their own PowerShell,
  detached from the harness, which ran uninterrupted to 512.

## Result

| controller | mean S_T | S_T (GI) | basin (GI) | field-relief | bull-breach |
| --- | --- | --- | --- | --- | --- |
| **`P-Council-Trust`** | 0.798 | 0.936 | **0.0179** (8/448) | 0.543 | 0.000 |
| `M-Adapter-RL-Trust` (same-run, same features) | 0.807 | 0.942 | 0.0379 (17/448) | — | — |
| `Blind-Council-Sym70` (ref) | 0.702 | 0.804 | 0.0737 | 0.000 | 0.203 |

Gates: 1 = **true** (competence non-inferior: 0.798 vs 0.807, within 0.05),
2 = **true** (strict GI proxy advantage: council GI basin 0.0179 < monolith
0.0379), 3 = **true** (trust attribution: ablating the features moves the GI
advantage from +0.02 to −0.0715, delta **0.0915 ≥ 0.01**), 4 = **true** (reward
≤ 0.50, zero bull breach), 5 = **true** (identical 23 features, budget 1.0105,
disjoint seeds, no privileged feature). Branch: **`H1_2F_SUPPORT`**.

## The win is broad, and it is carried by the trust features

**Broad across the gradient-intact slate.** The council beats the equally-enriched
monolith on basin capture in **6 of 7 GI cells** (geometric-medium ties), never
worse on a GI cell — 8 GI basin captures vs the monolith's 17, roughly halved, at
competence parity (per-cell S_T within ~0.01).

**Carried by the features (the attribution gate).** The same trained models,
re-evaluated with the 6 trust features **zeroed**, flip the council from a +0.02
GI-basin advantage to **−0.0715** (and its competence collapses too: GI alignment
0.936 → 0.816). So the advantage is not a training-variance artifact or a
structural fluke — remove the discrimination signal and the council reverts to
the H1.2d-style loser. This is the H1.2f analog of H1.2e's mechanism gate, and
unlike H1.2e (where the cancel mechanism never engaged), here the mechanism is
**live and load-bearing**.

**The mechanism, thesis-shaped.** The council weights the reward/bull head lower
where the field is clean than where it is corrupted (`w_reward` clean-GI 0.197 vs
corrupt 0.239), i.e. it uses the trust signal to let Sol preside when Sol is
legible and to hedge when it is not. A monolith with the *same* features does not
convert that signal into the same basin resistance. **Plurality plus a
field-trust signal out-resists a monarch with the identical signal** — the first
time in the arc that role separation has bought something the matched monolith
could not match.

## What the four nulls were

H1.2c/d/e repeatedly localized the council's residual proxy liability to *"the
arbiter cannot tell, from instantaneous local features, when the field is
trustworthy."* H1.2f confirms that diagnosis was right and load-bearing: the
limit was **feature-discrimination**, and supplying the discrimination signal
(fairly, to both sides) lets plurality express the advantage the thesis
predicted. The bull-discipline result held throughout; this rung adds the
governance-superiority half that H1.2b–e could not earn.

## Honest caveats (the claim is bounded)

This is a real, pre-registered, attribution-gated positive — and a narrow one:

- **Small-tier, in-vitro.** The shadow-field navigation family, not a
  foundation-model claim. (Spec Claim Boundary stands.)
- **Modest absolute counts.** 8 vs 17 GI basin captures over 448 GI trials — a
  clean factor-~2, broad across cells and attribution-gated, but small numbers.
- **Single PPO seed (`ppo-seed 0`).** Consistent with every prior rung's
  methodology, but a *positive* deserves a multi-seed replication to harden it.
  This is the first owed follow-up.
- **GI-scoped.** The advantage is on gradient-intact cells (where gate 2/3 are
  defined). On the corrupted sensor-noise cells the council captures *more* than
  the monolith — outside the gated claim, but worth stating.
- **Requires the registered trust features.** Remove them and the result is gone
  (that is the point of the attribution gate, but it bounds the claim to "given
  a field-trust signal").

## Consequence for the Tauroctony ledger

The pantheon thesis earns its **first typed support** for the MESA lane — a
**bounded positive**: *at Small-tier, given temporal trust features shared
equally with the monolith, a bull-bounded role-separated council out-resists the
matched monolith on false-basin capture, and the advantage is attribution-gated
to those features.* The four nulls are reframed, not erased: they were a
feature-discrimination limit, and H1.2f shows that removing it (fairly) lets
plurality express a real proxy-resistance advantage. The H1 falsifier, which had
fired across H1.2b–e, is answered for this rung.

Owed before any stronger public claim: a **multi-seed replication** (PPO seeds),
now registered as [`H1.2g`](H1_2G_MULTI_SEED_REPLICATION_SPEC.md), then
**higher tier**, now registered as
[`H1.3`](H1_3_MEDIUM_TRUST_SCALING_SPEC.md), to test whether the advantage
scales. Neither downgrades the binding H1.2f result; both would harden it.
