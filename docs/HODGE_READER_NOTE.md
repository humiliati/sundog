# Hodge Body/Shadow Reader — Front A (Phase 1)

- Artifact id: `HODGE_READER_NOTE`
- Type: Front-A reader note (Phase 1) — a reading of *known* mathematics and one
  *open* conjecture, not a proof, a claim, or an executable probe
- Date: 2026-06-01
- Ledger: [`SUNDOG_V_HODGE.md`](SUNDOG_V_HODGE.md) · Lit-pass:
  [`HODGE_LITPASS_MEMO.md`](HODGE_LITPASS_MEMO.md)
- Status: internal reader draft. No public page, `site-pages.json` entry, or
  executable probe is live. This note is the source the eventual `hodge.html`
  reader page would draw from, after the red-team checks below and external
  review.

> The form is visible. Is there a cycle behind it?

## 1. What this is, and what it is not

This is a **reader**: it explains the **Hodge conjecture** — a precise *open*
problem about *finished* mathematical objects — in Sundog's body/shadow
vocabulary, and places it on the portfolio's body/shadow family. It is **not**:

- a proof or disproof of the Hodge conjecture, in any case;
- a new criterion for the algebraicity of a Hodge class;
- a construction of any algebraic cycle;
- evidence for any open Hodge case;
- a transfer from electromagnetic Hodge-star / harmonic decomposition to
  algebraic cycles;
- a regime-2 / control-sufficiency or body-resistance *separation* claim;
- Sundog-original mathematics.

Every theorem cited below is due to others (Lefschetz; Deligne; Cattani–Deligne–
Kaplan; Voisin; Atiyah–Hirzebruch). Sundog supplies the **reading** and the
**placement**, fenced as above. The canonical statement is Deligne's Clay
formulation; all definitions defer to it (see the lit-pass Primary Anchors).

## 2. The Hodge conjecture

Let `X` be a smooth projective variety over `ℂ` of complex dimension `n`. Its
cohomology carries the **Hodge decomposition**

> `Hᵏ(X,ℂ) = ⊕_{p+q=k} H^{p,q}(X)`,  with  `H^{q,p} = conj(H^{p,q})`.

A **rational Hodge class** of type `(p,p)` is a class that is simultaneously
*rational* and *of pure type `(p,p)`*:

> `α ∈ H^{2p}(X,ℚ)`  whose image in `H^{2p}(X,ℂ)` lies in `H^{p,p}(X)`.

Write `Hdg^p(X) = H^{2p}(X,ℚ) ∩ H^{p,p}(X)` for the space of these classes.

Every codimension-`p` **algebraic cycle** `Z` (a `ℚ`-combination of closed
subvarieties) has a class `cl(Z) ∈ H^{2p}(X,ℚ)` that automatically lands in
`Hdg^p(X)`. The cycle classes span a subspace `Alg^p(X) ⊆ Hdg^p(X)`.

> **Hodge conjecture (Deligne / Clay).** For `X` smooth projective over `ℂ`,
> `Alg^p(X) = Hdg^p(X)`: every rational Hodge class is a `ℚ`-linear combination
> of algebraic cycle classes.

What is known: the case `p = 1` (divisor classes) is the **Lefschetz `(1,1)`
theorem**, and by the hard Lefschetz isomorphism the dual case `p = n−1`
follows. These facts are enough to settle the general conjecture for **every
smooth projective variety of dimension `≤ 3`**. The first dimension in which the
general statement is not known is **dimension four** (codimension-two classes on
a fourfold) — this is the boundary the Clay page names. Beyond the known cases
the conjecture is open; it is not known to be true or false in general.

## 3. The register ladder (the part to keep straight)

The single most common way to *sound* like progress on Hodge is to slide between
four different objects as if they were one. They are not. Naming them as a
ladder, with each arrow labeled, is the spine of this reader.

```
 (R1) closed differential form  ω            "a representative"
        │  Hodge theory + a Kähler metric        [THEOREM, metric-dependent]
        ▼
 (R2) harmonic representative of [ω]         "the canonical form"
        │  pass to the cohomology class          [THEOREM]
        ▼
 (R3) class [ω] ∈ H^{2p}(X,ℂ), with a type   ── rational + type (p,p) ⇒  SHADOW
        ▲                                          (a rational Hodge class)
        │  cycle class map  cl(Z)                [THEOREM: lands in R3, type (p,p)]
        │
 (R4) algebraic cycle  Z  of codimension p   ── the BODY
```

- **R1 → R2 → R3** is Hodge theory: given a Kähler metric, each class has a
  unique **harmonic representative**, and harmonic forms split by `(p,q)`-type.
  The *metric* enters at R2, and the harmonic representative depends on it. The
  `(p,q)` decomposition and the question whether a class is of pure type
  `(p,p)` are set by the complex structure, not by which Kähler metric is used to
  realize the harmonic representative.
- **R4 → R3** is the **cycle class map** `cl`: it is always defined and always
  produces a rational class of type `(p,p)`. This is the **body → shadow** map.
- **R3 → R4 is the conjecture, not a map.** "This rational `(p,p)` class is
  `cl(Z)` for some algebraic `Z`" is exactly what Hodge asserts and what is open.

Three category errors the ladder forbids (these are failure mode 3,
`HODGE-CATEGORY-ERROR`):

- **CE1 — harmonic ≠ rational.** The harmonic representative (R2) is a
  metric-dependent analytic object; rationality (the `H^{2p}(X,ℚ)` condition in
  R3) is a separate arithmetic structure. A harmonic form is not "the rational
  class."
- **CE2 — type `(p,p)` ≠ algebraic.** Lying in `H^{p,p}` (R3) is *necessary* for
  a class to be a cycle class, but asserting it is *sufficient* is the conjecture
  itself (R3 → R4). Erasing that gap is the central error.
- **CE3 — a picture of a form ≠ an exhibited cycle.** Drawing or animating a
  representative (R1/R2) does not display a body (R4).

## 4. The body and the shadow

In Sundog terms the ladder collapses to one clean translation:

| Hodge object | Sundog reading |
| --- | --- |
| algebraic cycle `Z` (codim `p`) | **body** |
| cycle class map `cl(Z)` | **body → shadow** map |
| rational `(p,p)` class `α ∈ Hdg^p(X)` | **shadow** (the right *kind* of shadow) |
| Hodge conjecture | does every admissible shadow have a body? |

The shadow is "admissible" precisely when it is rational and of type `(p,p)` —
the two conditions a cycle class must satisfy. The conjecture is the single
question:

> Is the body → shadow map `cl` **rationally onto** the admissible shadows —
> i.e. is `Alg^p(X) = Hdg^p(X)`?

This is a reading, not a method. It introduces no detector, algorithm, or
construction, and it does not make any open case more tractable.

## 5. Where Sundog earns its place — the existence pole

The portfolio reads many substrates on a **body/shadow** axis. Hodge occupies a
slot none of the others do, and saying *which* slot is the contribution here.

- **Faraday** is the *zero-resistance* pole: the plaquette-holonomy shadow
  reconstructs the body by the Bianchi identity. The body cannot hide.
- **Kakeya** (finite field) is the *maximal-resistance* pole: a complete
  direction-shadow reconstructs nothing about the body, yet forces it to a
  constant fraction of full size. The body cannot shrink.
- Both of those are **reconstruction / compression** statements about a body
  that is *known to exist*.
- **Hodge is a different axis entirely.** The shadow has the *right type*, and
  the body → shadow map is well-defined and safe in that direction; the open
  question is whether a body **exists at all** behind a given admissible shadow.
  Hodge is the **existence / algebraicity** pole — about surjectivity of `cl`,
  not about how lossy `cl` is.

> Faraday asks *can the shadow rebuild the body*; Kakeya asks *how small can the
> body be*; **Hodge asks whether the body is there at all.** That third question
> is the one the portfolio did not yet have a clean anchor for.

This placement is also the **binding fence against mis-filing Hodge**: because
the question is existence, not reconstruction, Hodge is **not** a body-resistance
measurement and **not** a regime-2 / control-sufficiency separation. Filing it as
either would be the category error this section exists to prevent.

### 5a. The shadow is structured, not arbitrary (but structure ≠ a body)

Cattani–Deligne–Kaplan proved that in a smooth projective family, for a fixed
transported cohomology class, the locus on the base where that class stays of
type `(p,p)` is an *algebraic subvariety*. (Ranging over all classes, the full
locus of Hodge classes is a *countable union* of such subvarieties.) So the
shadow is far from arbitrary; as classes move in families, the type condition
has algebraic structure. The fence: this is a
statement about **where the shadow lives**, not a construction of the **body**.
Structured shadows do not, by themselves, produce algebraic cycles.

### 5b. Faraday / Aharonov–Bohm bridge (companion — vocabulary and caution only)

The word "Hodge" appears in the Faraday/Maxwell work because Hodge theory and the
Hodge star organize cohomological and metric data there too. The transfer that is
*licensed* is one of **discipline**, not method:

> Faraday and Aharonov–Bohm taught Sundog not to confuse local closure with
> global structure. Hodge is the algebraic-geometry version of that caution: a
> class may have the right visible type, but the missing body is an *algebraic
> cycle*, and the physics vocabulary does not construct it.

What is **forbidden** (failure mode `HODGE-PHYSICS-TRANSFER`): treating the
electromagnetic Hodge-star or a harmonic survivor as a route to an algebraic
cycle, or as equivalent to a rational `(p,p)` class. A fuller standalone bridge
note is now opened in
[`HODGE_PHASE2_BRIDGE_NOTE.md`](HODGE_PHASE2_BRIDGE_NOTE.md); this paragraph is
the bounded companion.

## 6. The fences (binding)

- **Rational, not integral.** Everything above is about *rational* Hodge classes.
  The **integral** Hodge conjecture is false (Atiyah–Hirzebruch obstruction;
  Kollár-type examples). Unless a statement is explicitly marked "integral," it
  is rational.
- **Projective, not Kähler.** "Smooth projective over `ℂ`" cannot be weakened to
  "compact Kähler": Voisin's counterexample shows the Kähler analogue fails.
- **Type is necessary, not sufficient.** Being a rational `(p,p)` class is the
  *entry condition* for being a cycle class, never a certificate of one (CE2).
- **Hodge loci ≠ cycle construction.** Algebraicity of the Hodge locus (§5a) is
  not a construction of algebraic cycles.
- **Physics vocabulary does not transfer** (§5b).
- **Adjacent theories are out of scope.** Motives, Tate, generalized Hodge,
  absolute Hodge, and periods are adjacent context only, not part of this reader.
- **Sundog reads; it does not prove.** Every theorem here is someone else's.

## 7. Red-team / vacuity self-check (`HODGE-FRONT-A-VACUOUS`)

Red-team amendments from 2026-06-01 tightened three claims before Phase 2 was
opened: known-case language now says "the general conjecture below dimension
four" rather than implying a full case taxonomy; the cycle-class direction is
"well-defined" rather than "perfectly understood"; and the Hodge-locus paragraph
is scoped to the fixed-class family statement, not to cycle construction.

Honest test: does this reader say anything a careful standard exposition does not?

- The **statement and known cases** (§2): *no* — standard, and flagged as such.
- The **register ladder with labeled arrows and the three named category errors**
  (§3): *yes* — a projection-discipline exhibit that pins exactly where "visible
  form" stops being "known cycle," which standard expositions present implicitly
  at most.
- The **existence-pole placement** (§5) — Hodge as the existence/algebraicity
  slot, *distinct from* Faraday-zero and Kakeya-maximal, and explicitly **not**
  body-resistance and **not** regime-2: *yes* — portfolio-specific, in no Hodge
  exposition.
- The **fence set** (§6): *yes* — claim-boundary clarity a standard exposition
  does not carry.

The reader provisionally clears `HODGE-FRONT-A-VACUOUS` on §3, §5, and §6, not
on §2, as an internal red-team judgment only. If a future edit strips §3/§5/§6
down to a plain statement-and-known-cases retelling, it *fails* the check and
must not be promoted.

## 8. What this licenses

A clean Front-A reader unlocks, in order (per the lit-pass roadmap), and **not**
before:

1. a standalone **Faraday/AB/Hodge bridge note** (Phase 2): the shared
   vocabulary and the hard non-transfer, guarded by `HODGE-PHYSICS-TRANSFER`;
2. a **known-example gallery spec** (Phase 3): divisors / codimension-one first,
   every row labeled "known because…" and "does not imply…", guarded by
   `HODGE-TOY-LAUNDERING`;
3. a **public decision** (Phase 4): whether Hodge earns a page, a section of a
   math rail, or stays a docs note — and, if a page, only after clearing the repo
   SEO/social readiness rules;
4. **external review** by an algebraic geometer / Hodge-theory reader before any
   public launch, on the three questions in the lit-pass External Review Path.

Exit criterion for Phase 1: this note survives the §7 vacuity check
(provisionally, internally), commits no category error from §3 (the ladder is the
guard), and implies no progress on any open Hodge case. Public promotion still
requires external Hodge-theory review.
