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
coordinate-locality (κ) instance** → promote H2 to "the schema's spatial instance." **Lean follow-up DONE 2026-06-29:** `Sundogcert/ParityNoSufficientStat.lean` now defines
`IsSufficient` + `suffStatOrder` and proves **`suffStatOrder_eq : suffStatOrder n = n`** — the total
parity's only sufficient subset-parity is the full tuple, so σ = `n` (grows without bound; no fixed
finite order suffices). Supporting: `isSufficient_univ`, `not_isSufficient_of_ne_univ`,
`isSufficient_iff_univ`. Axiom-clean (`[propext, Classical.choice, Quot.sound]`), in the `AxiomAudit`
`#guard_msgs` gate, full `lake build` green (8525 jobs). The σ-anchor is now a machine-checked *order*,
not just a non-sufficiency.

#### H1 FRONTIER RESOLVED (2026-06-29): the universal description-length filtration does NOT rescue strong H1

Receipt: `scripts/suffstat_h1b_kolmogorov_frontier.py`. The one candidate route to rescue strong H1 was a
*universal* filtration — order statistics by description length / Kolmogorov complexity `K`, a single
comparable scale (bits, up to `O(1)`) into which every lane-filtration embeds. The universal comparable
scalar **does exist** (`K`); the test is whether it **preserves** the determine/resist dichotomy. It does
not — and **parity is the witness**:

| object | K (unbounded) | locality (bounded) | finite-order σ (bounded) | dichotomy |
| --- | --- | --- | --- | --- |
| finite-Markov (period q) | finite | finite | finite | determine (all agree) |
| info-loss (GF(2) shadow w/ kernel) | ∞ (≥ lost dims) | ∞ | ∞ | resist (all agree) |
| **parity** | **finite** (≤212 bits; fixed program `XOR-all`, size independent of n) | **∞** (reads all n) | **∞** | **SPLIT — K says determine** |

Mechanism: parity's latent `T = XOR(all bits)` is a (short-program, whole-shadow) function of the shadow
— fiber = 1 — so `K(T | shadow) = O(1)`: `K` calls it **determine**, while the bounded axes (locality,
finite-order) call it **resist**. *Any* info-present latent (fiber = 1) is K-determine, so the only
bound-free filtration **collapses "resist" down to exactly the info-loss (fiber > 1) cases**, missing
computational/parity resistance entirely.

**Resolution: strong H1 is UNRESCUABLE — not merely unproven.** "Resist" is **bound-relative**: it is
defined against a resource/locality bound, and that bound *is* the filtration. The unique bound-free
scalar (`K`) exists but does not preserve the dichotomy, so **no single filtration is both universal and
dichotomy-preserving** — the **schema** (H1 restricted) is the genuine ceiling, not a failure of effort.
This is the same fact as H5's fiber × σ split (parity is info-present yet finite-order-inaccessible); the
H1 frontier and the H5 resistance-type split are one phenomenon seen twice.

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

#### H2 RESULT (hardening 2026-06-29): strong "same coordinate" form FALSIFIED, schema form SURVIVES — recognizability is the STRUCTURAL instance, never the resist pole

Receipt: `scripts/suffstat_h2_space_vs_time.mjs` (reuses `ghost/metric-probe-core.js`). Two ladders on
the same sequences — spatial `r` = `recognizabilityRadius1D` (least centered `(2L+1)`-window fixing the
**role**, i.e. the desubstitution / a *structural* latent); temporal `σ_predict` = least preceding-`k`
window fixing the **next symbol** (no right-special factor of length `k`).

| sequence | r (role) | RS(k), k=1..8 | σ_predict |
| --- | --- | --- | --- |
| fibonacci | 1 (stable) | [1,1,1,1,1,1,1,1] | ∞ (>18) |
| period-doubling | 1 (stable) | [1,2,1,2,2,1,1,2] | ∞ (>18) |
| thue-morse | 2 (stable) | [2,2,4,2,4,4,2,2] | ∞ (>18) |
| periodic `abc` | capture 4 | — | 1 |
| periodic `aab` | capture 4 | — | 2 |

**Finding.** On every aperiodic substitution `r` is **finite** (1–2, Mossé) but `σ_predict = ∞`: by
Morse–Hedlund, aperiodic ⟹ complexity `p(k)` strictly increasing ⟹ a right-special factor at *every*
length ⟹ the next symbol is never determined by any finite preceding window. The two ladders **agree
only on the periodic controls** (both finite ≈ period). Therefore:

- **Strong H2 ("r and σ are the same coordinate; σ is the time-specialization of r") — FALSIFIED.**
  Fibonacci/Thue–Morse separate them: `r = 1–2` vs `σ_predict = ∞`.
- **Schema H2 ("recognizability is the coordinate-locality instance of the schema 'least context
  determining A latent'") — SURVIVES**, but the *latent differs*: `r` fixes the **structural** latent
  (desubstitution), `σ_predict` the **surface** next symbol. Aperiodicity is exactly the gap — structure
  stays locally recoverable, prediction does not.
- **Structural payoff (honest):** recognizability is **always** in the finite-σ / *determine* regime
  (Mossé: every aperiodic primitive substitution is recognizable), so the spatial instance **cannot host
  the σ = ∞ resist pole** — parity's σ = ∞ lives on the *predictive* filtration, which `r` never touches.
  "Space" and "time" are **not symmetric**: the structural latent is locally legible, the predictive one
  is not.

**Kill condition:** "mapping stays metaphor / Thue–Morse doesn't line up" — the strong *same-number*
mapping is indeed only metaphor (Thue–Morse falsifies it), but the *schema* mapping (least context
determining a latent) is a real shared formula. So H2 survives as the schema's structural instance,
consistent with H1 (schema unifies form, not scale). **Imported facts named:** Mossé (finite
recognizability for aperiodic primitive substitutions); Morse–Hedlund (aperiodic ⟺ `p(k) > k` ⟹
right-special factor at every length). The lab contributes the side-by-side ladder receipt.

**Disposition: H2 HARDENED** — schema-confirming, strong-form-falsifying; a clean negative that sharpens
H1 (the spatial instance is determine-only; the resist pole is predictive-only). Promise medium held.

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

#### H3 RESULT (hardening 2026-06-29): KILLED as stated (category error); salvage = find/check ⊥ predict-σ

Receipt: `scripts/suffstat_h3_verify_vs_predict.py`. The hook's own kill condition fired. Testing
"check" on its two meanings:

- **VERIFY-a-witness** (the CheckCost / NP sense): `λ(n)` given `n`'s prime factorization is **poly to
  verify** — product check + per-factor primality + parity count (measured 2 / 6 / 8 / 12 ops for
  Ω = 1 / 3 / 4 / 6). So parity is **verify-EASY**, the same column as the syndrome cert; *finding* the
  factorization (factoring) is the hard direction. Parity is **(find-hard, verify-easy)**.
- **PREDICT-next** (the P-2 sense): `λ(n)` from the λ-**sequence** history is determined at *no* finite
  order — `σ_predict = ∞`, ambiguous already at `k = 1` (a context `λ(n−1) = +1` is followed by both `+1`
  and `−1`).

| object | find-cost | verify-witness | predict-order σ |
| --- | --- | --- | --- |
| syndrome cert | hard (NP) | easy (O(mn)) | finite (verdict) |
| parity λ | hard (factoring) | easy (factorization) | ∞ |
| halo | n/a | n/a | finite (~1 param) |

**Verdict.** "check" equivocates: **find/check (verify-a-witness) and predict-order σ are orthogonal
axes**. Parity is verify-EASY + predict-∞; calling it the "check-hard pole" of the find/check ledger
imports its predict-∞ into the verify column — the exact category error the kill warned of.

- **Strong H3 ("parity = the check-hard pole of the find/check ledger") — KILLED (category error).**
- **Salvage (the deliverable, in the lab's kill-as-deliverable discipline):** the two axes are distinct,
  demonstrated concretely (verify-easy receipt + predict-∞ receipt). This **sharpens H1** — "verify a
  witness" and "predict from finite context" are two more of the *distinct filtrations*, reinforcing that
  σ is a family of order-invariants, not one comparable scalar.

**Imported facts named:** factoring's hardness (find side; not NP-complete, no known poly); primality
∈ P (AKS) / NP (Pratt) so verify ∈ P; P-2 for predict-∞.

---

## Routing & status

| hook | promise | risk | home | status |
| --- | --- | --- | --- | --- |
| H1 | high→**medium** | low | this file (public) | **HARDENED 2026-06-29** — strong form falsified, restricted (schema) form survives; **frontier RESOLVED** — strong form *unrescuable* (Kolmogorov filtration doesn't preserve the dichotomy) |
| H2 | medium | low | this file (public) | **HARDENED 2026-06-29** — strong "same coordinate" falsified (Thue–Morse: r finite, σ_predict=∞); schema (structural-instance) form survives |
| H3 | medium | medium | this file (public) | **HARDENED 2026-06-29** — KILLED as stated (category error: "check" equivocates); salvage = find/check ⊥ predict-σ are orthogonal axes |
| H4 | utility | low | internal slate | PROPOSED (tooling) |
| H5 | high-concept | high | internal slate | PROPOSED (unratified reframe) |

## Through-line (public hooks H1–H3 hardened 2026-06-29)

All three public hardenings tell **one coherent story**: "sufficient-statistic order" σ is a real organizing
**schema** — every determine/resist pair carries a filtration of statistics with a least-sufficient order,
and determine ⟺ finite σ, resist ⟺ σ = ∞ — but it is **not one comparable scalar**. The filtrations are
genuinely distinct, and each strong "it's all one thing" over-claim was falsified while the restricted
finding survived:

- **H1:** the scalar form falsified (coordinate-count vs rank vs moments vs parametric are incomparable);
  the schema survives. The σ = ∞ parity anchor is machine-checked (`suffStatOrder_eq`).
- **H2:** recognizability `r` ≠ predictive σ (Thue–Morse: `r` finite, σ = ∞); `r` is the schema's
  *structural* instance and lives only in the determine regime — it cannot reach the resist pole.
- **H3:** verify-a-witness ≠ predict-from-context (parity is verify-easy + predict-∞); the find/check
  ledger and the predict-σ axis are orthogonal — conflating them is a category error.

Net: at least six distinct filtrations appear (coordinate-count, rank, moments, parametric, verify-witness,
predict-σ); σ is a *family* of order-invariants, one per filtration, unified in form, not in scale. **The
one universal candidate — Kolmogorov / description-length (a 7th axis) — was then tested directly and
*fails to preserve the dichotomy* (it makes parity determine), so the schema is the *provable* ceiling,
not merely the observed one: "resist" is bound-relative (H1 FRONTIER block).**
