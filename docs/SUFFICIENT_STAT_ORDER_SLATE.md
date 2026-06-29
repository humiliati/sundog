# Sufficient-Statistic Order — a cross-lane invariant slate

**Opened 2026-06-29**, seeded by the parity-barrier lane (`docs/parity/PARITY_BARRIER_HOOKS_SLATE.md`,
P-1/P-2/P-3). That lane sharpened a single recurring coordinate — **how much context determines the
latent** — into a machine-checked extreme (parity = infinite/uncomputable order). This slate tests
whether that coordinate, **"sufficient-statistic order" σ**, is a genuine *cross-lane invariant*
unifying the lab's determine/resist program, rather than a coincidence of vocabulary.

> **Discipline.** Hooks are **PROPOSED**, not hardened. Imported walls are named inline. Public-eligible
> candidates only — the tooling hook (H4) and the unratified-reframe hook (H5) are routed to
> `internal/slates/SUFF_STAT_ORDER_INTERNAL_2026-06-29.md`. Nothing here claims a new theorem until a
> RESULT block says so.

---

## The definition under test

For a lab object with latent `z` and observable shadow `s`, let

> **σ(object)** = the least order `k` such that some order-`k` statistic of the shadow is a *sufficient
> statistic* for `z` (i.e. determines `z`); `σ = ∞` if no finite-order statistic suffices, and the
> object is **uncomputable-order** if even an unbounded but computable statistic fails.

The conjecture of the slate: **determine ⟺ finite σ, resist ⟺ σ = ∞**, and σ is one scalar that ranks
every worked example in the portfolio.

---

### H1 (lead) — σ IS the determine/resist axis

**Statement.** The lab's determine/resist examples are separated by σ alone: determine examples have
*finite* σ, resist examples have σ = ∞, and the existing machine-checked cores already pin each value —
so the unification assembles **proved facts under one definition**, not a new claim.

- **Anchors (each cites an existing proved core):**
  - parity **σ = ∞** — P-1 `Sundog.ParityNoSufficientStat.partial_not_sufficient` (no finite-order
    partial parity is sufficient for the total).
  - syndrome certificate **σ = 1** — `Sundog.Certificate.syndrome_independent_of_secret` (the secret
    drops under one linear map; the surviving coset label is an order-1 sufficient statistic).
  - Gaussian charFun **σ = 2** — `ShadowDecay`: the discrete mean survives averaging (`determination`),
    the continuous spread resists (`resistance`) — determined by ≤2 moments.
  - Cauchy **σ = ∞** — `ShadowDecayCauchy`: no finite moments, charFun resists; the finite-order
    separator that fails.
  - halo geometry **σ ≈ 1** — `HaloGeometry`: one minimum-deviation parameter + the ice lattice
    determine the ring.
- **Why from this:** P-1 supplies the σ = ∞ anchor in Lean; the others are already axiom-clean cores.
- **Attack:** (a) a Lean `def suffStatOrder` on the toy + `suffStatOrder_parity_eq_top` reusing
  `partial_not_sufficient`; (b) a classification table mapping each worked example to its σ with the
  citing theorem; (c) a one-paragraph statement of what σ *predicts* (which lanes are reachable).
- **Kill if:** σ is not well-defined cross-lane (the "order" means different things per lane and the
  unification is only a word); or it collapses to a bare "finite vs infinite" with no predictive
  content; or any anchor's σ is mis-assigned on inspection.
- **Promise high / risk low / buildable.** Public-eligible.

#### H1 RESULT (hardening 2026-06-29): strong form FALSIFIED, restricted form SURVIVES — σ is a unifying SCHEMA, not one comparable scalar

**Definition tested:** `σ(shadow → latent)` = least `k` such that some order-`k` statistic in the
shadow's natural filtration is sufficient for the latent; `∞` if none.

**Cross-lane stress test** (each σ checked against the cited axiom-clean core):

| object | latent | filtration ("order" = …) | σ | core | side |
| --- | --- | --- | --- | --- |
| parity toy | total parity | # coordinates read | ∞ | `partial_not_sufficient` (P-1) | resist |
| parity toy | parity over S | # coordinates read | \|S\| | trivial | determine |
| syndrome cert | secret | linear functionals of z | ∞ (info-theoretically lost) | `syndrome_independent_of_secret` | resist |
| syndrome cert | safe/unsafe verdict | coset weight (rank) | finite | `accept_sound`/`reject_sound` | determine |
| Gaussian charFun | discrete label | moments | 1 (mean) | `determination` | determine |
| Gaussian charFun | continuous spread | moments | ∞ (damped) | `resistance` | resist |
| Cauchy | location | moments | ∞ (no finite moments) | `ShadowDecayCauchy` | resist |
| halo | ring radius | refraction parameters | ≈1 | `HaloGeometry` | determine |

**Finding.** The determine/resist split = **finite-vs-∞ σ holds WITHIN each object's filtration** (every
row's side matches the finiteness of its σ). But the filtrations are genuinely *different* —
coordinate-subset size (parity), linear-functional rank (syndrome), moment count (Gaussian/Cauchy),
parameter dimension (halo). Parity's ∞ and Cauchy's ∞ are different ∞'s; Gaussian's `2` and a
Markov-order-2 source's `2` are not comparable. Therefore:

- **Strong H1 ("σ is ONE comparable scalar ranking every lane") — FALSIFIED.** There is no single
  filtration; the orders are not cross-comparable.
- **Restricted H1 ("σ is a unifying SCHEMA: every determine/resist pair carries a filtration of
  statistics with a least-sufficient order σ, and determine ⟺ finite σ, resist ⟺ σ = ∞") — SURVIVES.**
  The schema unifies the **form** (each lane has a filtration + a σ + a finite/∞ dichotomy), not the
  **scale** (the σ's live in incomparable filtrations).

**Kill conditions:**
- **KC1** (σ ill-defined cross-lane) — **partially fired**: σ is well-defined *per filtration*, not
  cross-filtration. This is what downgrades strong → restricted (a scoping, not a kill).
- **KC2** (collapses to "finite vs ∞" with no content) — **cleared**: the schema's content is the
  *filtration* — it names, per lane, the natural axis along which sufficiency is measured — and it
  retrodicts which latent of an object is recoverable/checkable (finite σ) vs which is the
  security/barrier (σ = ∞): it correctly separates the cert's cheap verdict from its lost secret, and
  isolates parity's barrier. Organizing/retrodictive, **not** a novel prediction (so labelled).
- **KC3** (anchor mis-assigned) — **cleared**: the sloppy initial "syndrome σ = 1" is really *two* σ's
  (verdict finite / secret ∞), which *strengthens* the schema — one object exhibiting both sides.

**Machine-checked content:** only the parity σ = ∞ anchor is in Lean (P-1 `partial_not_sufficient`). The
other rows cite existing axiom-clean cores for their *side*; the σ-*value* is analysis, not Lean.

**Honest label & disposition:** H1 is a **synthesis/organization** result (the schema), not a new
theorem. Lead promise downgraded **high → medium**. Spin-offs: **H2 is exactly the schema's
coordinate-locality (κ) instance** → promote H2 to "the schema's spatial instance." **Open frontier
(named, not built):** is there a *universal* filtration (Kolmogorov / description-length order) into
which the lane-filtrations embed, restoring a comparable scalar? That is the only route to rescue strong
H1 — parked. **Lean follow-up DONE 2026-06-29:** `Sundogcert/ParityNoSufficientStat.lean` now defines
`IsSufficient` + `suffStatOrder` and proves **`suffStatOrder_eq : suffStatOrder n = n`** — the total
parity's only sufficient subset-parity is the full tuple, so σ = `n` (grows without bound; no fixed
finite order suffices). Supporting: `isSufficient_univ`, `not_isSufficient_of_ne_univ`,
`isSufficient_iff_univ`. Axiom-clean (`[propext, Classical.choice, Quot.sound]`), in the `AxiomAudit`
`#guard_msgs` gate, full `lake build` green (8525 jobs). The σ-anchor is now a machine-checked *order*,
not just a non-sufficiency.

### H2 — recognizability radius ↔ σ (space vs time)

**Statement.** Ghost's **recognizability radius** `r` (a *spatial*, finite Mossé constant for aperiodic
tilings) and σ (a *temporal* context length) are the same coordinate in two geometries; σ is the
1-D-time specialization of `r`.

- **Why from this:** bridges the Ghost lane (Phase-4 metric probe, the 1-D substitution recognizability
  ladder) and the parity σ-ladder.
- **Attack:** run the existing 1-D substitution recognizability ladder (`ghost/metric-probe-core.js`)
  next to the order-`k` sufficient-statistic ladder; show they measure the same thing on Fibonacci /
  period-doubling (finite `r`, finite σ) vs Thue–Morse (the parity-like case — predict its σ behaviour).
- **Kill if:** the mapping stays metaphor (no shared formula), or Thue–Morse / parity don't line up.
- **Promise medium / risk low.** A sub-part of H1. Public-eligible.

### H3 — parity on the find-vs-check ledger (the check-hard pole)

**Statement.** The lab's find/check program (`CheckCost`, `StraightLineCost`: *cheap to check*) and
parity form a spectrum; parity is the **check-hard extreme** — even *verifying* the next symbol from
finite history is impossible (σ = ∞), independent of find-cost.

- **Why from this:** P-2's own-R² ladder is literally a check-at-finite-order measurement.
- **Attack:** place the lab's objects on a 2-D (find-cost, check-order) grid — syndrome =
  (hard-find, cheap-check), parity = (find n/a, no-finite-check) — as a table or a small Lean ordering.
- **Kill if:** "check" equivocates between *verify-a-witness* (the NP sense in CheckCost) and
  *predict-next* (the statistical sense in P-2) — the two must stay explicitly distinct.
- **Promise medium / risk medium.** Public-eligible.

---

## Routing & status

| hook | promise | risk | home | status |
| --- | --- | --- | --- | --- |
| H1 | high→**medium** | low | this file (public) | **HARDENED 2026-06-29** — strong form falsified, restricted (schema) form survives |
| H2 | medium | low | this file (public) | PROPOSED — **promoted**: the schema's coordinate-locality (spatial) instance |
| H3 | medium | medium | this file (public) | PROPOSED |
| H4 | utility | low | internal slate | PROPOSED (tooling) |
| H5 | high-concept | high | internal slate | PROPOSED (unratified reframe) |
