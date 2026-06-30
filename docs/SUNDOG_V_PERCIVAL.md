# SUNDOG_V_PERCIVAL

*A reader's workbench on un-targeting reward collection, scaffolded on the Grail
romance. Less inflammatory public name than HOLYGRAIL; same lane.*

Status: **OPENED 2026-06-30. Bare spine only — nothing run.** This is provenance +
a phase ladder + an outreach branch, not a result. The romance is declared
scaffold (intellectual bricolage for legibility), not evidence; see the tauroctony
lesson in [`SUNDOG_V_TAUROCTONY.md`](SUNDOG_V_TAUROCTONY.md) for what happens when a
symbolic frame meets policies (it deflates, and the deflation is the finding). The
discipline this lane inherits: a falsifier fence on every phase, a claim boundary
fixed before scoring, and a Lean anchor for whatever single statement survives.

---

## Provenance (two sources, welded)

1. **Jessica Taylor, "Quantilizers: A Safer Alternative to Maximizers for Limited
   Optimization" (MIRI, 2015 / AAAI workshop 2016).** A `q`-quantilizer samples
   from the top-`q` fraction of a base distribution `γ` ordered by expected *proxy*
   utility `Û`, instead of arg-maxing. Its safety property: expected true cost is
   bounded by `(1/q) · E_γ[cost]` — you trade optimization power (`q → 0`) against a
   bounded multiple of the base distribution's own risk. Three load-bearing
   assumptions travel with the bound and are this lane's whole leverage: the base
   `γ` is given and trusted; the (proxy, true) cost joint is **fixed under `γ`**; and
   it is single-shot. Known soft spots in the literature: base-distribution
   dependence, **non-composition** under iteration, and **reflective instability**.

2. **The Perceval / Grail corpus.** Chrétien de Troyes, *Perceval, le Conte du
   Graal* (c. 1180s, unfinished) — the Grail procession and the **unasked
   question**; Wolfram von Eschenbach, *Parzival* (c. 1200s) — innocence → failure →
   redemption, the wound of Anfortas; the Vulgate *Queste del Saint Graal* — Galahad
   achieves the Grail by **perfection**, Lancelot fails by his **one attachment**,
   Bors and Percival partake; Malory's synthesis. The load-bearing motif: the Grail
   is not **grasped**. Perceval's canonical failure is *not asking* "Whom does the
   Grail serve?" — he optimized a social proxy (courtesy, "don't ask too many
   questions") and lost it. The best worldly knight cannot collect it; the innocent
   and the perfect can.

The weld: the Grail is a candidate reward that the **targeting** policy cannot
collect, and the lane exists to find out whether that picture is more than a
costume for something MIRI already wrote down in 2015.

---

## The crux (stated as a question, not a thesis)

> **Characterize the rewards collectable by a policy that does not condition its
> selection on a model of the reward.** Call such a policy *un-targeting*. Where does
> the class of un-targeting-collectable rewards sit relative to the
> quantilizer-collectable class — strictly inside, equal, or strictly outside?

Three verdicts are registered live, and the apparatus is built to decide between
them rather than to argue for one:

- **(a) Inside / null.** "Blind" cashes out as "sample from `γ`, drop the tilt" — i.e.
  the `q = 1` (zero-pressure) corner of the quantilizer simplex, which collects
  *less* expected proxy than any `q < 1`. If this is all there is, Percival is a
  quantilizer footnote and the romance bought only legibility. This is the **deflation**,
  and it would be a clean, honest result of the cap-not-council shape.
- **(b) Equal.** Un-targeting is a reparameterization of low-`q` quantilizing with no
  separating instance.
- **(c) Outside / prize.** There exist rewards an un-targeting policy collects that
  **no** `q < 1` quantilizer can, because the quantilizer's residual proxy-conditioned
  pressure is itself what destroys them. This is the only verdict that earns the lane
  a paper instead of a footnote.

The fence: (a) and (b) are the expected default; (c) is the thing to try to break the
lane against, not to assume.

---

## Phase ladder (a dozen, ascending)

**P1 — Legibility contract & dictionary.** Fix the figure→regime mapping as
*hypothesized*, to be earned: arg-max maximizer ↔ Lancelot (grasps, fails by one
attachment); `q`-quantilizer ↔ a tilted-base knight; the no-proxy-gap maximizer ↔
Galahad (achieves by perfection, not by limitation); the **un-targeting policy** ↔
Percival; the Grail ↔ the reward under study; the unasked question ↔ the proxy
Percival must *not* condition on. Falsifier: if every figure collapses to a single
point on the `q`-axis, the dictionary is ornament — flag it, don't smuggle it.

**P2 — Quantilizer baseline, restated exactly.** Re-derive the `q`-quantilizer and
its `(1/q)` bound in lane notation; pin the three assumptions (given `γ`; fixed cost
joint; single-shot). This is the wall Percival must clear to exist.

**P3 — Define "un-targeting" formally.** A policy whose action distribution is
invariant under permutation of proxy values (equivalently: zero mutual information
between selection and the proxy ranking). Locate it on the quantilizer simplex (first
guess: the `q = 1` / zero-pressure corner). Register which of P-crux (a/b/c) each
later phase would move.

**P4 — Static separation attempt (expected null).** Outcome-reward regime, cost joint
fixed under `γ`. Conjecture: every un-targeting policy is weakly dominated by some
quantilizer. If it holds — likely — then "blind = base minus tilt" is *proved*, the
deflation is banked, and any surplus must live off the static axis (→ P5). Toy +
candidate Lean.

**P5 — The dynamic / strong-Goodhart regime (where the prize would be).** Define
**intent-dependent rewards**: `R` that degrades as a function of the policy's
optimization *pressure* on the proxy — the measure-becomes-target law made literal,
the Grail that withdraws when grasped. In this regime a quantilizer with any tilt
`q < 1` *reduces* `R` by the tilt it still applies, while an un-targeting policy
preserves it. Candidate strict separation. Falsifier: if every such `R` is
pathological, measure-zero, or requires the agent to read its own source, the
separation is a curiosity, not a mechanism — fence it as such.

**P6 — The unasked-question instrument.** Operationalize Perceval's failure: a runnable
toy whose true return is *lowered* by conditioning the policy on a particular observed
variable (the courtesy proxy). A clean witness of P5 if "don't look" strictly beats
"look and ignore"; a clean null if it never does (look-and-ignore weakly dominates).

**P7 — Where does `γ` come from (the romance's contribution to the open problem).**
Quantilizers punt on the base distribution. Test whether the corpus's selection of
*which* knights may approach (Galahad/Percival/Bors yes; Lancelot/Gawain no) encodes
admissibility conditions on `γ` — innocence as a base defined by *absence of
instrumental structure*. Mostly a philology→formal bridge; expected soft.

**P8 — Composition under the quest (a clean positive even if P5 is null).**
Quantilizers degrade under iteration; the quest is long and yet purity is
maintained/restored. Test whether un-targeting composes where quantilizing drifts —
the zero-pressure corner as the one fixed point of iterated quantilization. Toy:
iterate both, measure distributional drift.

**P9 — Reflective stability.** A quantilizer need not build a quantilizer. Does an
un-targeting policy protect its own un-targeting — and is this where Galahad
(perfection, stable) and Percival (naivety, fragile) actually split? Expected
conceptual/negative; register honestly.

**P10 — Lean anchor.** Machine-check the single statement that survives P4–P8 — most
likely the static-collapse (`Percival ⊆ quantilizer` at the zero-pressure corner), or,
if it lands, a concrete existence witness for the dynamic separation. Axiom-audited,
gated, in `sundogcert`; cite [`project_sundog_lean_formalization`] discipline. Sold as
not-deep; its value is exactness and scope.

**P11 — Synthesis & verdict.** State which of P-crux (a/b/c) the apparatus supports.
Two honest landings, pre-committed: *deflation* — "un-targeting is the zero-pressure
corner of the quantilizer simplex; the romance's surplus was legibility and
composition, not collection power"; or *prize* — "un-targeting strictly collects
intent-dependent rewards quantilizers cannot, because quantilizing still applies
proxy-conditioned pressure." Report whichever lands, including the null.

**P12 — Outreach (branch on what lands).** If the structural/philology half carries
(the corpus genuinely encodes `γ`-admissibility or a composition rule): DM
medievalists / Grail-corpus historians (Chrétien & Wolfram scholarship). If the formal
half carries (a real intent-reward separation, or a clean Lean anchor and composition
result): DM alignment researchers and optimization/physics people working the
Goodhart/quantilization line. If nothing clears the deflation: deposit as portfolio,
easter-egg disposition. Outreach owner-gated, one channel at a time.

---

## Claim boundary (fixed before scoring)

**Will consider claiming, if earned:** that un-targeting reward collection is or is
not a distinct class from quantilization on a stated reward family; the regime
boundary (static vs intent-dependent) where any separation lives; a composition
property; one machine-checked statement.

**Will not claim:** that this transfers to foundation models or deployed agents; that
the Grail romance *evidences* anything (it is scaffold); that a separation found on a
toy intent-reward is non-pathological without the P5 fence cleared; or anything above
the in-vitro tier.

The one-line honest prior: **this probably deflates** — Percival is most likely the
`q = 1` corner Taylor already drew, and the lane's value is then a legible map of
*exactly where* the quantilizer intuition stops and a strictly different one would have
to begin. The small chance it does not deflate (the intent-dependent / strong-Goodhart
separation) is the only part worth an outreach DM, and it is fenced accordingly.

---

## Cross-links

- Lesson on symbolic frames deflating to a scalar: [`SUNDOG_V_TAUROCTONY.md`](SUNDOG_V_TAUROCTONY.md).
- Legibility-vs-claim-boundary cautionary tale: [`SUNDOG_V_LEAST_ACTION.md`](SUNDOG_V_LEAST_ACTION.md).
- Lean anchoring discipline for P10: the formalization lane in `sundogcert`.
