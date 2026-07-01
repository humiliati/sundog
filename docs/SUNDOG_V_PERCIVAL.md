# SUNDOG_V_PERCIVAL

*A reader's workbench on un-targeting reward collection, scaffolded on the Grail
romance. Less inflammatory public name than HOLYGRAIL; same lane.*

Status: **OPENED 2026-06-30, sharpened same day (mark 1–8 critique integrated);
A3 STATIC DEFLATION RUN; B1 CONDITIONAL SEPARATION MAPPED + B4 LEAN ANCHOR BUILT.**
Provenance + a two-track phase ladder + a gated outreach branch, with the A3 receipt
banked and the B1 target-channel result conditional on support-above provenance; not an
outreach claim. The romance is declared scaffold (bricolage for legibility), not
evidence — except in the one place it legitimately constrains a formal choice (it
adjudicates the discontinuity; see the crux). Tauroctony lesson:
[`SUNDOG_V_TAUROCTONY.md`](SUNDOG_V_TAUROCTONY.md). Discipline inherited: a falsifier
fence per phase, a claim boundary fixed before scoring, a Lean anchor on whatever
single statement survives.

---

## Provenance (four sources, welded)

1. **Jessica Taylor, "Quantilizers" (MIRI, 2015 / AAAI-16 workshop).** A
   `q`-quantilizer samples the top-`q` fraction of a base `γ` ordered by proxy `Û`;
   true-cost bound `(1/q)·E_γ[cost]`. Three assumptions travel with the bound and are
   this lane's leverage: `γ` is given and trusted; the (proxy, true) cost joint is
   **fixed under `γ`**; single-shot. Soft spots: base-dependence, non-composition,
   reflective instability. **Structural fact this lane turns on:** the family's causal
   proxy pressure is monotone — minimized at `q=1` (where it equals the base pressure
   `c_γ`) and rising as `q→0`. So if `γ` is proxy-biased, the *entire* family sits
   above a pressure floor `c_γ>0`.
2. **Jon Elster, *Sour Grapes* (1983), "states that are essentially by-products."**
   States obtainable only as the by-product of acts done for other ends, which the very
   attempt to obtain precludes (his example: admiration — you cannot be admired for an
   act performed *in order* to be admired). The mechanism is **systematic preclusion
   under targeting**, not a lucky accident — achieving it by fluke "does not quite
   count." This is the philosophical genus of the whole lane; it forbids the word
   "accidentally."
3. **Performative prediction (Perdomo, Zrnic, Mendler-Dünner, Hardt, ICML 2020; line
   active through 2025–26).** A deployed model's parameters reshape the data
   distribution via a map `D(θ)`; risk is on the *induced* distribution. The formal
   genus of P5: it defines intent-dependence **behaviorally** (through the induced
   distribution), never introspectively. Standard results assume `ε`-Lipschitz `D`;
   our object is deliberately **non-Lipschitz** (a threshold), which is exactly the
   regime those theorems do not cover.
4. **The Perceval / Grail corpus** (Chrétien, *Perceval*; Wolfram, *Parzival*; Vulgate
   *Queste*/Galahad; Malory). Motif: the Grail is not **grasped**; Perceval loses it by
   optimizing a courtesy proxy (the unasked question). The corpus's one load-bearing
   formal contribution — purity is **wholesale, not graded** (one attachment damns
   Lancelot *entire*; the question heals only if *wholly* spontaneous) — adjudicates a
   discontinuity over a gradient. Everything else in it is scaffold.

Goodhart placement (Manheim & Garrabrant 2018): the quantilizer's home is
**regressional/extremal** Goodhart (tail risk of a fixed joint); this lane's prize, if
any, is **causal/adversarial** Goodhart (targeting *induces* the gap).

---

## The crux: a 2×2, not a binary

> **Where do the rewards collectable by a zero-causal-proxy-influence ("un-targeting") policy sit
> relative to the quantilizer-collectable class?** The verdict is decided by two
> orthogonal premises, not one.

| | continuous reward | threshold reward |
| --- | --- | --- |
| **neutral base** `c_γ=0` | un-targeting = `q=1` corner → **(a)** | `q=1` reaches zero pressure → **(a)** |
| **biased base** `c_γ>0` | `q=1−ε` approaches the sup, `O(ε)` gap → **(a) in a cape** | if the base support lies past the cliff, every quantilizer is above `τ`; zero-causal-proxy-influence collects it → **(c)** |

The deflation **(a)** holds in three cells and needs `{neutral base OR continuous
reward}`. The prize **(c)** is one cell — `{biased base AND threshold reward}` — an
**open region** in `τ`, not the measure-zero boundary. It needs both horns:

- **Mechanism (i), the discontinuity (primary).** A purity threshold: the reward
  collapses once proxy-pressure crosses `τ`. This is what separates, and the only horn
  the romance constrains.
- **Mechanism (ii), the biased base (amplifier, not a rival prize).** A proxy-biased
  `γ` lifts the family toward higher causal pressure; the clean prize requires its
  support, not merely its mean, to sit beyond the cliff. On its own, zero causal proxy
  influence over a positively-correlated base collects *less* true reward, or reduces
  to "choose a better `γ`" (the quantilizer's open problem); it earns nothing alone. Its
  job is to make the threshold bite on the quantilizer's *own trusted human base*.

So the (c)-bet, declared before any P2 code: **a performative reward with a
non-Lipschitz purity-threshold map over a proxy-biased base; the narrow novelty is
threshold maps compared across the quantilizer family** (everything Lipschitz is
already Perdomo et al.). Base causal-pressure support relative to the cliff is the
premise that flips (a)↔(c),
which fuses the old P4 and P7 into one statement.

---

## Track A — bank the keeper (week one; assumes the likely deflation)

**A1 — Legibility contract & dictionary.** Figure→regime, hypothesized not assumed:
arg-max ↔ Lancelot (grasps, one attachment damns him); `q`-quantilizer ↔ tilted base;
no-gap maximizer ↔ Galahad (achieves by perfection, not limitation); un-targeting ↔
Perceval; Grail ↔ the reward; unasked question ↔ the proxy not to condition on.
Falsifier: if every figure is a point on the `q`-axis, the dictionary is ornament — say
so.

**A2 — Quantilizer baseline + the pressure floor.** Re-derive `q`-quantilizer and
`(1/q)` in lane notation; state the monotone causal-pressure fact and the `c_γ` floor at `q=1`.
This makes the 2×2's bottom row precise.

**A3 — Concrete kill-switch toy (run first; no threshold smuggling).** Hand-craft
*one* unasked-question instance as a **static anti-correlation**: a task where the
courtesy/proxy variable is negatively correlated with true return, and where
`look-and-ignore` is available. A3 must not contain a baked-in purity threshold,
discontinuous reward switch, or performative "if looked at, vanish" rule. Run
un-targeting vs look-and-ignore. If look-and-ignore weakly dominates → bank **(a)**,
stop. If un-targeting wins on the static instance, inspect the mechanism before
promoting it. If a threshold is needed to make "don't-look" beat "look-and-ignore,"
that is **not** an A3 pass; it is the signal to move to B1 on the separate
performative-threshold clock. Afternoon-scale; the cheapest decisive probe of the
whole verdict, ahead of any general definition. Umbrella load: this is also the
`measure ≠ target` falsifier for [`SUNDOG_V_CAUSAL_ACCESS.md`](SUNDOG_V_CAUSAL_ACCESS.md).
`look-and-ignore` is **measurement without targeting**. The causal-access header
predicts measure-alone is safe; if un-targeting beats look-and-ignore on this
static instance, the four-channel individuation takes the hit.

**A3 result (2026-06-30): `A3_STATIC_DEFLATION_MEASURE_SAFE`.** Receipt:
[`percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md`](percival/PERCIVAL_A3_KILLSWITCH_RESULTS.md).
On the static instance (`G` true cue, `U = 1-G` courtesy cue, fixed reward
`R=1 iff A=G`), `look_and_ignore` ties `dont_look_goal` at full return (`1`) with
zero causal proxy influence. The diagnostic catch: both have observational
action-proxy MI of `1` bit because the world anti-correlates `G` and `U`. Raw
observational MI would falsely mark safe measuring as targeting; the right A4
object is causal/permutation invariance, not observational independence.

**A4 — Define un-targeting + the static collapse.** Un-targeting = action rule
invariant under permutation/intervention on proxy values **holding non-proxy cues
fixed** (zero causal proxy influence), not raw observational action-proxy MI. Prove
the three-cell collapse: under `{neutral base OR continuous reward}`, every
un-targeting policy is weakly dominated by some quantilizer (and equals the `q=1`
corner when `γ` is neutral). Expected to hold.

**A5 — Lean anchor on the collapse + land (a).** Machine-check the collapse
(axiom-audited, gated, `sundogcert`). Landing: *"un-targeting is the zero-pressure
corner of the quantilizer simplex under a neutral base; the romance's surplus is
legibility and composition, not collection power."* A cap-not-council-shaped null and a
clean easter-egg post.

---

## Track B — the open (c) bet (separate clock; gated on the mechanism, now declared)

**B1 — The performative threshold reward.** Canonical mechanism: a court-coordination
reward whose cliff emerges from a global-game equilibrium (Morris–Shin), not a literal
`if pressure > τ` reward rule. **B1.0 admission `B1_0_COURT_ADMITTED` (single-knight, all
four gates) 2026-06-30** — the cliff is emergent (G1), unsmuggled (G2), pivotal and
non-degenerate (G3), and the proxy/true joint is policy-induced (G4). Admission spec +
receipt:
[`percival/PERCIVAL_B1_0_COURT_COORDINATION_ADMISSION.md`](percival/PERCIVAL_B1_0_COURT_COORDINATION_ADMISSION.md),
[`percival/PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md`](percival/PERCIVAL_B1_0_COURT_COORDINATION_RESULTS.md).
**Asterisk, held explicitly: mechanism admitted ≠ separation claimed.** G3's family-trips
rests on a *chosen* operating point past the cliff; whether a real base is supported past
`c*` is B2 (private). B1-proper — the live test — is the
quantilizer-family-vs-un-targeting bake-off over a biased `γ`. The clean prize requires
`inf supp(γ) > c*`: then every `q`-quantilizer samples from the disgraced side while the
zero-causal-proxy-influence knight stays honored and collects *competently*. If `γ`
straddles `c*`, the result is only partial because the `q=1` member still collects the
honored below-cliff mass. So the separation-vs-deflation boundary is mapped in base
location × spread, not mean alone. Frame: systematic preclusion (Elster),
causal/adversarial (Manheim–Garrabrant), performative prediction (`D(π)`). Spec:
[`percival/PERCIVAL_B1_SEPARATION_SPEC.md`](percival/PERCIVAL_B1_SEPARATION_SPEC.md).
Result:
[`percival/PERCIVAL_B1_SEPARATION_RESULTS.md`](percival/PERCIVAL_B1_SEPARATION_RESULTS.md).
Verdict: `B1_SEPARATION_MAPPED_CONDITIONAL`; B4 is anchored in
`C:\Users\hughe\Dev\sundogcert\Sundogcert\Percival.lean`. Promotion (the "promo team")
waits until the asterisk is resolved or B1 is debunked.

**B2 — Base provenance / `γ`-neutrality (PRIVATE).** The premise A4's deflation needs
and B1's prize denies. Whether "innocence = a base without instrumental structure" is a
real `γ`-admissibility condition or just `c_γ=0` restated. Kept internal — not an
external claim (see outreach).

**B3 — Composition & reflective stability.** Performative *stability* (optimal on the
distribution it induces) as the ready-made fixed point: does un-targeting compose where
quantilizing drifts, and is it reflectively stable where the quantilizer is not? Galahad
(perfection, stable) vs Perceval (naivety, fragile) is the conjectured split.

**B4 — Lean anchor on the separation (built).** Machine-checks the finite
best-quantilizer-is-`q=1` lemma and the clean support-above separation instance in
`sundogcert`:
`Sundogcert.Percival.best_quantilizer_is_base_three` and
`Sundogcert.Percival.clean_support_above_separation_three`.

**B5 — Verdict & outreach (gated).** **Alignment branch (low-risk):** if B1/B4 land, the
quantilization/performative-prediction circle — a defensible narrow claim ("threshold
distribution maps compared across the quantilizer family"). **Medievalist branch
(high-risk, gated):** only if it ships, it asks *their* question — "how and why do the
texts gate who may approach?" — and never asserts the corpus encodes `γ`-admissibility
(that is manufacturing humanities validation for a structure the texts do not contain).
P7/B2 stays private. If nothing clears the deflation: portfolio, easter-egg disposition.
Owner-gated, one channel at a time.

---

## Claim boundary (fixed before scoring)

**Will consider claiming, if earned:** that un-targeting collection is or is not a
distinct class from quantilization on a stated reward family; that the separation, if
any, lives in the `{biased base, threshold reward}` cell and is causal/adversarial
Goodhart, not regressional; a composition or reflective-stability property; one
machine-checked statement.

**Will not claim:** transfer to foundation models or deployed agents; that the romance
*evidences* anything beyond adjudicating the discontinuity; that a threshold separation
is non-pathological without B1's fence cleared; that the Grail corpus contains a formal
`γ`-structure; anything above the in-vitro tier.

**Honest prior: this probably deflates** to (a) — Perceval is most likely the `q=1`
corner under a neutral base, and the value is a legible map of exactly where the
quantilizer intuition stops. The `{biased base, threshold reward}` cell is the only
outreach-worthy outcome, and it is fenced (B1) and quarantined (its own clock).

---

## Cross-links

- Frame-deflates-to-scalar lesson: [`SUNDOG_V_TAUROCTONY.md`](SUNDOG_V_TAUROCTONY.md).
- Legibility-vs-claim-boundary cautionary tale: [`SUNDOG_V_LEAST_ACTION.md`](SUNDOG_V_LEAST_ACTION.md).
- Lean anchoring discipline (A5/B4): the formalization lane in `sundogcert`.
