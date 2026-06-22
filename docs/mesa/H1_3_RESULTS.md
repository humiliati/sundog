# H1.3 Medium Trust-Scaling — Binding Results

Status: **`H1_3_ATTRIBUTION_NULL` — a Medium proxy advantage exists, but it is
NOT creditable to the registered trust features. The H1.2f trust-feature
mechanism does not transfer to Medium tier; the Small-tier positive is now
bounded.** Ran 2026-06-22 (512-update binding). Spec:
[`H1_3_MEDIUM_TRUST_SCALING_SPEC.md`](H1_3_MEDIUM_TRUST_SCALING_SPEC.md), gates
frozen before the run.

The one change from H1.2f: frozen role heads upgraded **Small → Medium**; trust
features, caps, slate, seeds, same-run equal-budget monolith, and the attribution
gate all inherited. The H1.3-a 3-cell probe had selected `H1_3_MEDIUM_SUPPORT`
(attribution delta +0.1042); the binding does not replicate that on the full
13-cell slate.

## Result

Intact eval (trust features live):

| controller | mean S_T | S_T (GI) | basin (GI) | field-relief |
| --- | --- | --- | --- | --- |
| `P-Council-Trust-M` | **0.766** | **0.890** | **0.069** | 0.248 |
| `M-Adapter-RL-Trust-M` (same-run) | 0.719 | 0.796 | 0.199 | — |
| `Blind-Council` (ref) | 0.702 | 0.804 | 0.074 | 0.000 |

Gates: validity/fairness **true** (budget 1.0105, 23 features matched, audit
clean, no leakage); competence non-inferior **true** (council 0.766 *beats*
monolith 0.719); GI proxy advantage **true** (council GI basin 0.069 < monolith
0.199); **trust attribution FALSE**; bull discipline **true** (reward ≤ 0.50,
zero breach). 512 updates, 8.87M env steps. Branch: **`H1_3_ATTRIBUTION_NULL`**.

## Why attribution fails — the features help the monolith more

The attribution gate asks whether *ablating the trust features collapses the
council's advantage*. At Medium it does the opposite:

| | council GI basin | monolith GI basin | council advantage |
| --- | --- | --- | --- |
| trust features **intact** | 0.069 | 0.199 | **+0.130** |
| trust features **ablated** (zeroed) | 0.183 | 0.375 | **+0.192** |

Attribution delta = `0.130 − 0.192 = −0.0625` (< the 0.01 threshold → fails).

The trust features help *both* controllers resist the basin (council 0.183→0.069,
monolith 0.375→0.199), but they help the **monolith more in absolute terms**
(0.176 vs 0.114 reduction), so the council's *relative* advantage actually
**shrinks** with the features. The council still uses them (intact field-relief
0.248 vs ablated 0.027; `w_reward` vs `disagree_mean_K` correlation −0.63), and
they raise its competence (intact GI 0.890 vs ablated 0.769) — but they are not
what produces its edge over the monarch at Medium. Per the H1.3 spec branch
table, this is the pre-registered `ATTRIBUTION_NULL` reading.

## What it does and does not mean

**Does mean (the registered verdict):** the H1.2f *mechanism* — trust features
carrying the pantheon's proxy-resistance advantage — is **Small-tier-bounded**.
It does not transfer to Medium frozen heads. The H1.2f result stands as a
Small-tier, in-vitro, trust-feature-attributed bounded positive; H1.3 bounds the
scaling claim exactly as the spec foresaw (§6: "valid null branches preserve
H1.2f as Small-tier support and bound the scaling claim").

**Does NOT mean plurality lost at Medium — the opposite, but unregistered.**
Both gates the spec scores for the *outcome* pass, and strongly: the council
**out-competes** (0.766 vs 0.719) **and** out-resists the monarch on GI basin
(0.069 vs 0.199, ~3×), with or without trust features. This is consistent with
the H1.2d prediction that *"the monolith's ignore-the-proxy advantage may not
scale"*: at Medium the monolith degrades hard (GI alignment 0.796, GI basin
0.199) while the role-separated council holds (GI 0.890 / 0.069). So a **Medium
plurality advantage appears to be carried by the role-separation structure
itself, not by the trust features.**

But that is a *diagnostic observation, not a gated result*. The attribution gate
was built to credit the trust-feature mechanism, and it correctly says this win
isn't that. The structural-advantage hypothesis needs its own registered rung
with its own attribution (e.g. ablate role-separation / a non-role-separated
equal-budget control; multi-seed) before any claim. We do not get to relabel an
attribution-null as a structural-support win after seeing it.

## Consequence for the Tauroctony ledger

- **H1.2f stays the bounded positive** — Small-tier, trust-feature-attributed.
  Unchanged and uncorrected (its attribution held at Small).
- **The scaling claim is bounded:** the trust-feature mechanism does **not**
  transfer to Medium (`H1_3_ATTRIBUTION_NULL`).
- **A new, unregistered observation:** a structural plurality advantage at Medium
  (council beats the monarch on both competence and proxy-capture, features
  ablated or not). Owed as a *separately registered* rung — **H1.4: structural
  attribution at Medium** (ablate role-separation, multi-seed) — before it can be
  claimed. Until then it is a diagnostic, not support. This owed rung is now
  registered as [`H1.4`](H1_4_MEDIUM_STRUCTURAL_ATTRIBUTION_SPEC.md).

The methodology held again: the attribution gate refused to let a real-looking
Medium win (gates 1–3 all green) be credited to a mechanism that the ablation
shows is not responsible — the same discipline that turned H1.2e into an honest
null, now applied to a positive.
