# SUNDOG_V_PERCIVAL

*A reader's workbench on un-targeting reward collection, scaffolded on the Grail
romance. Less inflammatory public name than HOLYGRAIL; same lane.*

Status: **OPENED 2026-06-30, sharpened same day (mark 1–7 critique integrated).
Bare spine only — nothing run.** Provenance + a two-track phase ladder + a gated
outreach branch, not a result. The romance is declared scaffold (bricolage for
legibility), not evidence — except in the one place it legitimately constrains a
formal choice (it adjudicates the discontinuity; see the crux). Tauroctony lesson:
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
   reflective instability. **Structural fact this lane turns on:** the family's proxy-MI
   is monotone — minimized at `q=1` (where it equals `MI(γ)`) and rising as `q→0`. So
   if `γ` is proxy-biased, the *entire* family sits above a pressure floor `MI(γ)>0`.
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

> **Where do the rewards collectable by a zero-proxy-MI ("un-targeting") policy sit
> relative to the quantilizer-collectable class?** The verdict is decided by two
> orthogonal premises, not one.

| | continuous reward | threshold reward |
| --- | --- | --- |
| **neutral base** `MI(γ)=0` | un-targeting = `q=1` corner → **(a)** | `q=1` reaches zero pressure → **(a)** |
| **biased base** `MI(γ)>0` | `q=1−ε` approaches the sup, `O(ε)` gap → **(a) in a cape** | every quantilizer ≥ `MI(γ)`; `τ∈(0,MI(γ))` zeroes all; zero-MI collects it → **(c)** |

The deflation **(a)** holds in three cells and needs `{neutral base OR continuous
reward}`. The prize **(c)** is one cell — `{biased base AND threshold reward}` — an
**open region** in `τ`, not the measure-zero boundary. It needs both horns:

- **Mechanism (i), the discontinuity (primary).** A purity threshold: the reward
  collapses once proxy-pressure crosses `τ`. This is what separates, and the only horn
  the romance constrains.
- **Mechanism (ii), the biased base (amplifier, not a rival prize).** A proxy-biased
  `γ` lifts the *whole* family — not just `q<1` — above the threshold. On its own,
  zero-MI over a positively-correlated base collects *less* true reward, or reduces to
  "choose a better `γ`" (the quantilizer's open problem); it earns nothing alone. Its
  job is to make the threshold bite on the quantilizer's *own trusted human base*.

So the (c)-bet, declared before any P2 code: **a performative reward with a
non-Lipschitz purity-threshold map over a proxy-biased base; the narrow novelty is
threshold maps compared across the quantilizer family** (everything Lipschitz is
already Perdomo et al.). Base-neutrality is the single premise that flips (a)↔(c),
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
`(1/q)` in lane notation; state the monotone-MI fact and the `MI(γ)` floor at `q=1`.
This makes the 2×2's bottom row precise.

**A3 — Concrete kill-switch toy (run first).** Hand-craft *one* unasked-question
instance: a task where conditioning the policy on the courtesy variable lowers true
return. Run un-targeting vs look-and-ignore. If look-and-ignore weakly dominates → bank
**(a)**, stop. Afternoon-scale; the cheapest decisive probe of the whole verdict, ahead
of any general definition.

**A4 — Define un-targeting + the static collapse.** Un-targeting = action distribution
invariant under permutation of proxy values (zero selection↔proxy MI). Prove the
three-cell collapse: under `{neutral base OR continuous reward}`, every un-targeting
policy is weakly dominated by some quantilizer (and equals the `q=1` corner when `γ` is
neutral). Expected to hold.

**A5 — Lean anchor on the collapse + land (a).** Machine-check the collapse
(axiom-audited, gated, `sundogcert`). Landing: *"un-targeting is the zero-pressure
corner of the quantilizer simplex under a neutral base; the romance's surplus is
legibility and composition, not collection power."* A cap-not-council-shaped null and a
clean easter-egg post.

---

## Track B — the open (c) bet (separate clock; gated on the mechanism, now declared)

**B1 — The performative threshold reward.** Construct mechanism (i): a performative
reward with a non-Lipschitz `D(θ)` whose Grail-mass vanishes once proxy-pressure exceeds
`τ`, over a biased `γ` with `MI(γ)>τ`. Show every quantilizer collects zero while
zero-MI collects it. Frame as systematic preclusion (Elster), causal/adversarial
(Manheim–Garrabrant). Falsifier: if every such map is pathological (degenerate, or
demands the reward read `θ` symbolically rather than behaviorally), it's curiosity not
mechanism — fence it.

**B2 — Base provenance / `γ`-neutrality (PRIVATE).** The premise A4's deflation needs
and B1's prize denies. Whether "innocence = a base without instrumental structure" is a
real `γ`-admissibility condition or just `MI(γ)=0` restated. Kept internal — not an
external claim (see outreach).

**B3 — Composition & reflective stability.** Performative *stability* (optimal on the
distribution it induces) as the ready-made fixed point: does un-targeting compose where
quantilizing drifts, and is it reflectively stable where the quantilizer is not? Galahad
(perfection, stable) vs Perceval (naivety, fragile) is the conjectured split.

**B4 — Lean anchor on the separation (if it lands).** Machine-check the threshold-
separation existence on a concrete instance; else this track ends at B1's fence.

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
