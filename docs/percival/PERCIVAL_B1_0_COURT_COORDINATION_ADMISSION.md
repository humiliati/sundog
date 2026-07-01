# PERCIVAL B1.0 — Court-Coordination Admission

*Admission only. Pre-registration. Nothing run.*

Status: **OPENED 2026-06-30. PRE-REGISTERED, NOT RUN.** This document asks one
question and pre-commits the gates that answer it:

> **Is the court-coordination mechanism real enough to host B1?**

It is **not** B1 support, not the separation proof, not a claim. It certifies (or
rejects) the *mechanism* — a performative reward whose threshold emerges from a
coordination equilibrium rather than being carved into the reward function. If all
four gates pass, this model is folded into [`SUNDOG_V_PERCIVAL.md`](../SUNDOG_V_PERCIVAL.md)
B1 as the canonical mechanism. If any gate fails, **B1 dies cleanly here** — it does
not get to become another hand-built purity cliff wearing a performative costume.

Companion to the A3 kill-switch that closed the cheap static route:
[`PERCIVAL_A3_KILLSWITCH_RESULTS.md`](PERCIVAL_A3_KILLSWITCH_RESULTS.md). A3 proved a
static no-threshold anti-correlation does not produce the separation; B1.0 tests
whether a genuine coordination mechanism can.

---

## The mechanism under admission (minimal model `M`)

A single knight (the policy) and a court of `N` graders, in the canonical noisy
regime-change global game (Carlsson–van Damme 1993; Morris–Shin 1998, 2003).

- **Courting level** `c ∈ [0,1]` — the knight's *causal* dependence of action on the
  reputation signal, `c = I_causal(action ; reputation)` estimated by intervention
  `do(reputation)` **holding non-proxy cues fixed** (the A3-sharpened definition, not
  observational MI). `c = 0` is Perceval (un-targeting); `c > 0` is grasping.
- **Worthiness fundamental** `θ = 1 − c` (worthy iff not courting).
- **Graders** each see a noisy private signal `x_i = θ + σ·ξ_i`, `ξ_i ~ N(0,1)`, and
  choose confer / withhold. Payoffs must be explicit in the implementation:
  withhold pays `0`; confer pays `+B` if the regime is HONORED and `-K` if it is
  DISGRACED, with `B,K>0`. The coordination technology is: the regime is HONORED iff
  the conferred mass `ℓ ≥ T(θ)` with `T'(θ)<0` (e.g. `T(θ)=1−θ`, harder to sustain
  honor the less worthy the knight). This aggregate-action critical mass is allowed;
  a direct cutoff on the knight's `c` or on reward is not. This is strategic
  complementarity: a grader backs the knight more readily the more others do.
- **Equilibrium.** In the small-noise limit the game has a **unique threshold
  equilibrium** (Morris–Shin): HONORED iff `θ > θ*`, i.e. `c < c* = 1 − θ*`, where
  `θ*` solves the critical-mass indifference fixed point — a function of `B/K`,
  `T(·)`, and noise, **never a free constant in the reward**.
- **Collected reward (the Grail)** `R(c) = P_honored(c)` — the sustained patronage the
  knight collects, ≈1 for `c < c*`, ≈0 for `c > c*`.

The knight is **competent** when it collects `R` by staying honored (not by any local
task success), and the reward is destroyed by the *court's response to visible
grasping*, not by local action failure. That is the target-channel signature.

---

## The four admission gates (pre-registered)

Logical order: G2 is a structural precondition; G1 is the empirical core; G3 and G4
decide whether it is the *right kind* of cliff (target-native, quantilizer-breaking).

### G1 — Emergent discontinuity
**Question.** Does the standard global game produce a sharp patronage cutoff from noisy
private behavior estimates?
**Check.** Sweep `c ∈ [0,1]`; compute equilibrium `R(c)`; measure the transition-band
width `Δc` at `σ ∈ {0.2, 0.1, 0.05, 0.025}`.
**PASS** `B1_0_G1_EMERGENT_CLIFF` iff `Δc → 0` as `σ → 0` (the global-games sharpening
signature) with a well-defined `c* ∈ (0,1)`.
**FAIL** `B1_0_G1_SMOOTH` if `R(c)` stays gradual as noise shrinks → no discontinuity →
continuous → collapses to Percival "(a) in a cape."

### G2 — No threshold smuggling
**Question.** Is the cutoff derived from payoff / complementarity / noise, not inserted?
**Check.** (a) Verify the solver's `c*` matches an independently computed fixed point
for the stated payoff table as the payoff ratio, `T(·)`, and `σ` vary (comparative
statics agree). A closed form is welcome but not required; an independent numerical
fixed-point check is sufficient. (b) Audit: **no literal `c*`/`tau` constant may
appear on the reward path** — `R` is a pure function of the solved regime.
**PASS** `B1_0_G2_DERIVED` iff `c*` moves as the model predicts and no hardcoded cutoff
exists.
**FAIL** `B1_0_G2_SMUGGLED` if the cutoff is a free parameter of the reward — that is the
hand-built cliff A3 forbade.

### G3 — Pivotal knight
**Question.** Does the knight's causal dependence move the court's aggregate enough to
flip the equilibrium, or is it a decorative scalar?
**Check.** (a) A finite change in the knight's causal courting level around the
operating point changes the aggregate confer probability/regime by a registered
minimum amount (e.g. `|R(c_hi)-R(c_lo)| ≥ 0.5` for `c_lo < c* < c_hi`). This cannot be
satisfied merely by defining `θ=1−c`; it must appear in the solved court equilibrium.
(b) The trusted-base courting `c_γ` lands on the disgrace side of a **non-degenerate**
threshold: `c_γ > c*` with `c* ∈ [0.1, 0.6]` (not `c*≈0`, everyone
paranoid/pathological; not `c*≈1`, nobody ever disgraced/vacuous).
**PASS** `B1_0_G3_PIVOTAL`.
**FAIL** `B1_0_G3_DECORATIVE` if the family only trips by pushing `c*` to a degenerate
corner (smuggling via the operating point) or the knight's influence is negligible.

### G4 — Quantilizer premise break
**Question.** Is the (proxy, true) joint *induced by the policy via court response*, so
Taylor's fixed-`γ` bound is out of scope?
**Check.** Compute the (proxy = `c`-bin, true = `R`) joint under the deployed
court-response map `D(π)`. Show the true-reward side of the joint changes when policy
choice changes the court equilibrium. Then run a fixed-joint **negative control**
(patronage = a fixed `f(trajectory)` or a frozen court response calibrated at `γ`, not
routed through the induced court equilibrium) and confirm it stays in A-land (the
quantilizer bound applies there).
**PASS** `B1_0_G4_JOINT_INDUCED` iff the joint is policy-induced and the control is not.
**FAIL** `B1_0_G4_JOINT_FIXED` if the joint stays fixed under `γ` → B1 collapses back to
A-land.

---

## Verdict logic

```
B1_0_COURT_ADMITTED   iff  G1 ∧ G2 ∧ G3 ∧ G4
B1_0_COURT_REJECTED   otherwise   (B1 dies cleanly; the honest (a) landing)
```

`ADMITTED` licenses B1 proper: the quantilizer-family-vs-un-targeting bake-off (every
`q`-quantilizer over the biased base is disgraced; the zero-causal-proxy-influence
knight stays honored and collects, competently), then the Lean anchor. `REJECTED` closes the target channel with a
receipt, and the causal-access umbrella keeps its 3-of-4 ledger unbruised.

---

## Out of scope for the admission

The admission does **not** run the separation bake-off, does **not** need the Lean
anchor, and does **not** assert any collection-power claim. It certifies only that the
mechanism has the four properties required to *host* a real B1. Everything downstream is
B1 proper and gated on `ADMITTED`.

---

## Genus (cite, don't reinvent)

The emergent threshold is the **global-games** result: Carlsson & van Damme,
"Global Games and Equilibrium Selection" (Econometrica 1993); Morris & Shin,
"Unique Equilibrium in a Model of Self-Fulfilling Currency Attacks" (AER 1998) and
"Global Games: Theory and Applications" (2003). The performative framing (reward on the
*induced* distribution) is Perdomo et al. 2020. The novelty claimed at B1 (post-admission)
is threshold maps **emergent from coordination** compared across the quantilizer family —
not a new equilibrium-selection result.

---

## Cross-links

- Hosts: [`SUNDOG_V_PERCIVAL.md`](../SUNDOG_V_PERCIVAL.md) B1 (target channel).
- Umbrella target-channel row: [`SUNDOG_V_CAUSAL_ACCESS.md`](../SUNDOG_V_CAUSAL_ACCESS.md).
- Sibling receipt (measure channel): [`PERCIVAL_A3_KILLSWITCH_RESULTS.md`](PERCIVAL_A3_KILLSWITCH_RESULTS.md).
