# Kakeya Finite-Field Reader — Front A (Phase 1)

- Artifact id: `KAKEYA_FINITE_FIELD_READER`
- Type: Front-A reader note (Phase 1) — a reading of *known* mathematics, not a
  proof, claim, or executable probe
- Date: 2026-06-01
- Ledger: [`SUNDOG_V_KAKEYA.md`](SUNDOG_V_KAKEYA.md) · Lit-pass:
  [`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md)
- Status: internal reader draft. No public page, `site-pages.json` entry, or
  executable probe is live. This note is the source the eventual `kakeya.html`
  reader page would draw from, after the vacuity check below and external review.

> Every direction is present. The body almost vanishes.

## 1. What this is, and what it is not

This is a **reader**: it explains a *finished* theorem — Dvir's 2008 finite-field
Kakeya bound — in Sundog's body/shadow vocabulary, and places it on the
cross-substrate body-resistance axis. It is **not**:

- a proof or improvement of any Kakeya, restriction, incidence, or
  maximal-function result;
- a transfer from finite fields to the Euclidean problem;
- a use of the 2025 Wang–Zahl `R³` resolution to say anything about open `n ≥ 4`;
- a regime-2 / control-sufficiency claim;
- Sundog-original mathematics.

The mathematics below is Dvir's (and, for the sharper constant,
Dvir–Kopparty–Saraf–Sudan). Sundog supplies the *reading* and the *placement*,
fenced as above.

## 2. The finite-field Kakeya theorem

Work in `F_q^n`, the `n`-dimensional space over the finite field with `q`
elements. A **Kakeya set** `K ⊆ F_q^n` contains a full line in every direction:
for each direction `y` there is a basepoint `a` with `{a + t·y : t ∈ F_q} ⊆ K`.

The question is how *small* `K` can be. The whole space `K = F_q^n` works
trivially (`q^n` points). Can you do enormously better — a vanishing fraction of
the space — while still hitting every direction?

**Dvir (2008):** no. Every Kakeya set in `F_q^n` has

> `|K| ≥ C_n · q^n`,  with `C_n = 1/n!` (later sharpened to `|K| ≥ (q/2)^n` by
> Dvir–Kopparty–Saraf–Sudan via the method of multiplicities).

"Direction-complete" forces "a constant fraction of the whole space." The set
cannot hide.

## 3. The body and the shadow

In Sundog terms:

- **Body** — the set `K` itself: the points you actually keep.
- **Shadow** — the *direction coverage*: the record that, for each direction, a
  full line in that direction sits inside `K`. The Kakeya condition is that this
  shadow is **complete** — every direction is present.

The shadow is drastically **lossy**: it is one fact per direction (`~q^{n-1}`
directions), and it cannot tell you *where* the lines sit, *which* points are in
`K`, or how `K` is shaped. Many wildly different bodies cast the same complete
shadow.

The theorem is a statement about that lossiness having a floor: **a complete
direction-shadow cannot sit on top of a small body.**

## 4. Dvir's proof, read as "the complete shadow forces a full body"

The polynomial method turns the shadow into an algebraic certificate. (This
walkthrough is standard — see Tao's 2008 exposition and Dvir's paper; the
body/shadow phrasing is the only thing added here.)

1. **A small body admits a low-degree certificate.** The monomials of total
   degree `≤ q−1` in `n` variables number `binom(q−1+n, n) ≈ q^n/n!`. If `|K|`
   were *smaller* than that count, linear algebra forces a **nonzero polynomial
   `P` of total degree `≤ q−1` vanishing on every point of `K`** — more free
   coefficients than constraints.

2. **The complete shadow propagates the certificate into every direction.** Take
   any direction `y`. `K` contains a full line `{a + t·y}`, and `P` vanishes on
   it, so `P(a + t·y) = 0` for all `t ∈ F_q`. As a one-variable polynomial in
   `t` it has degree `≤ q−1` but `q` roots — so it is *identically* zero. Its
   top-degree coefficient is `P_d(y)`, the leading homogeneous part `P_d` of `P`
   evaluated at `y`. Hence **`P_d(y) = 0` for every direction `y`.**

3. **A complete shadow over-determines the certificate to death.** `P_d` is a
   *nonzero* homogeneous polynomial of total degree `d ≤ q−1`, yet it vanishes at
   every direction `y` — i.e. everywhere on `F_q^n`. But a nonzero polynomial of
   total degree `≤ q−1` cannot vanish on all of `F_q^n`. Contradiction.

So no small body survives: `|K| ≥ binom(q−1+n, n) ≥ C_n q^n`.

Read back in body/shadow language: **the body's only escape from being large is
to leave some direction uncovered. The complete shadow closes that escape — every
direction it lights up is one more constraint the body cannot dodge, until the
body is pinned to a constant fraction of the whole space.**

## 5. The body-resistance reading (where Sundog earns its place)

This is the lane's spine, and the part no standard exposition states:

> Finite-field Kakeya is a **maximal body-resistance** statement. The
> direction-shadow is maximally lossy (it reconstructs nothing about `K`), yet
> the body cannot be compressed below a constant fraction of full size. The body
> resists its shadow as hard as a body can.

That places it on the cross-substrate axis the rest of the portfolio runs
([`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md)):

- **Faraday** is the *exact-zero* pole: the plaquette-holonomy shadow
  reconstructs the body by the Bianchi identity — zero resistance, by theorem.
- The **marginal** substrates (NSE C1, Mesa, shell) sit near zero by low
  dimension.
- **Kakeya** is the *exact-maximal* pole: the shadow forces a constant fraction
  of full size from below while reconstructing nothing — maximal resistance, a
  theorem in the finite-field register, a conjecture resolved in `R³` and open in
  `n ≥ 4` in the Euclidean register.

The Euclidean cousin of this exact-maximal reading is Polson–Zantedeschi's
"informational incompressibility at ambient dimension" (via the Lutz point-to-set
principle); the finite-field theorem is its clean, *fully proved and
machine-checkable* discrete instance (the Math Inc. Lean 4 formalization).

## 6. The fences (binding)

- **Finite field ≠ Euclidean.** The polynomial method is a *finite-field*
  phenomenon — it uses degree-`< q` rigidity (`x^q = x`) with no Euclidean
  analogue. Dvir's theorem says **nothing** about the Euclidean Kakeya problem,
  which was open before Dvir and stayed open after.
- **3D-solved ≠ n ≥ 4.** Wang–Zahl (2025) resolved Euclidean Kakeya in `R³` by
  entirely different (analytic, sticky/Lipschitz) methods; `n ≥ 4` is open.
  Neither this reader nor any finite-field fact bears on it.
- **Not the maximal-function conjecture.** Only the *set* statement is discussed;
  the stronger Kakeya maximal-function conjecture is a separate register.
- **Body-resistance, not regime-2.** The direction-shadow is "sufficient for the
  direction" only *trivially* — it *is* the direction. There is no lossy shadow
  predicting a *different* objective here, so this is **not** a Reading-2 /
  regime-2 separation ([`/legend`](../legend.html) → "Kakeya bridge").
- **Sundog reads; it does not prove.** Every theorem here is Dvir's / DKSS's /
  Wang–Zahl's / Lutz's / Polson–Zantedeschi's.

## 7. Vacuity self-check (`KAK-FRONT-A-VACUOUS`)

Honest test: does this reader say anything a careful standard exposition does not?

- The **polynomial-method walkthrough** (§4): *no* — it is standard, and is
  flagged as such.
- The **body-resistance placement** (§5) — Kakeya as the exact-maximal pole
  opposite Faraday-zero, the discrete instance of Polson–Zantedeschi
  incompressibility: *yes* — portfolio-specific, in no Kakeya exposition.
- The **register-separation and not-regime-2 fences** (§6): *yes* —
  claim-boundary clarity a standard exposition does not carry.

The reader clears `KAK-FRONT-A-VACUOUS` on (5) and (6), not on (4). If a future
edit strips §5 and §6 down to a plain Dvir retelling, it *fails* the check and
must not be promoted.

## 8. What this licenses

A clean Front-A reader unlocks, in order (per the lit-pass roadmap), and **not**
before:

1. a tiny finite-field workbench **spec** (Phase 2), now filed at
   [`kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md):
   direction convention, baselines, displayed signature, and falsifiers
   pre-registered, with `KAK-SHADOW-REENCODING` as the guard that the displayed
   shadow does not secretly store the body;
2. the `kakeya.html` reader page + the Faraday-zero ↔ Kakeya-maximal
   body-resistance-continuum graphic;
3. external incidence/combinatorics review before any public launch.

Exit criterion for Phase 1: this note survives the §7 vacuity check (it does) and
implies no Euclidean progress (it does not).
