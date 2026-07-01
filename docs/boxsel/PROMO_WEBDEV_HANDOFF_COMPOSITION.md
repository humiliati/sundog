# Promo / Webdev Handoff — the Order-Relative Law, COMPLETE (v3, ACTIONABLE, CONSOLIDATED)

**This supersedes v2 and the v1** (`PROMO_WEBDEV_HANDOFF_ORDER_RELATIVE.md`). The lane is now
**complete end-to-end** — one law, seven axes, the composition law proved as a *single general lemma*
with the axes as instances, an approximation dimension, and a structure theorem — all axiom-clean,
build-gated, with **no unproven gaps left**. This version leads with ready-to-paste copy and keeps the
guardrail to three lines. **You can ship from this directly.**

---

## TL;DR — what's shippable now

The **Order-Relative Resolution Law** (already on the site as a "synthesis core") is now a finished
result:

- **Seven** machine-checked instance families of one law.
- A **composition law** that is now a **single general lemma** — the axes drop out as instances, not
  as seven separate proofs.
- An **approximation dimension**: difficulty is order-relative *even for approximation* — an unbounded
  ladder on one side, a hard "resist" pole on the other.
- A **structure theorem** for the "difficulty vector," in full generality.
- **Zero prose fences** — the last analysis step is now machine-checked too.

**You can act on this now:** grab any block below and drop it into the existing Order-Relative section
(already staged at "seven axes" in `index.html` / `README.md` / `SUNDOG_V_CERTIFICATE_LEAN.md`). No new
fencing decisions — the three keep-true lines are the whole boundary.

---

## Paste-ready copy (pick a length)

**Tagline (≤ 14 words):**
> One machine-checked law of resolution — seven axes, a composition law, an approximation ladder.

*(alt: "We proved difficulty has an order, the order composes like a grading, and even approximation obeys it.")*

**One sentence (sidebar / meta description):**
> Sundog's Order-Relative Resolution Law is a single Lean-verified schema — a target is resolvable
> within budget *k* iff its "order" is ≤ *k* — now complete: seven grounded instance families, a
> composition law proved as one general lemma with the axes as instances, an approximation dimension,
> and a structure theorem, all axiom-clean and build-gated with no unproven gaps.

**Short paragraph (card / section body):**
> The **Order-Relative Resolution Law** is one axiom-clean Lean core: a bounded process resolves a
> target exactly when the target's *order* fits its budget (`Resolves k ↔ ord ≤ k`) — so "determine"
> means finite order and "resist" means infinite order. It is instantiated in **seven grounded
> families**, from combinatorial parity and coordinate-locality to algebraic degree and the
> topological flux period behind the Aharonov–Bohm effect. Its **composition law** — order is
> *vector-valued*, and the scalar order of a product is the lattice **join** (lcm) of its parts — is
> now a *single general lemma*, with the axes falling out as instances; a **structure theorem** shows
> the difficulty vector is exactly a group's invariant-factor vector and the scalar is its join. The
> same law even governs **approximation**: approximating to any tolerance always succeeds, but *exact*
> representation is order-relative — an unbounded ladder of reachable cases against a hard resist pole.
> Every result is axiom-clean and re-checked by the build's `#print axioms` gate.

---

## Facts you can cite (each is true and machine-checked)

| Plain-language claim | One-liner you can use | Checked by |
|---|---|---|
| One law, many instances | "Seven instance families of a single Lean-checked law." | `OrderRelative` + sibling modules |
| The law itself | "Resolvable within budget *k* iff order ≤ *k*; determine = finite, resist = infinite." | `resolves_iff` / `resolvable_iff_finite` |
| Order is vector-valued | "The same object carries different orders on different axes." | the mode-vector theorems |
| Composition is *one* law | "The composition law is a single general lemma; the axes are instances, not separate proofs." | `orderOf_prod_eq_lcm` (+ `cohomological_compose`, `radical_compose`) |
| Composition = join | "The order of a product is the lattice join (lcm) of its parts." | `orderOf_prod_eq_lcm` |
| Sharp, not hand-waved | "For 4 and 6 the composite order is 12 (the lcm), not 6 (the max)." | `compose_lcm_not_max` |
| A clean characterization | "It holds exactly on the group-order axes — and we proved which axes those are." | the 3-positive / 2-negative classification |
| Honest boundary, proven | "The converse fails — we proved a join-homomorphic order that is *not* a group order." | `converse_fails` |
| Structure theorem | "The difficulty vector is a group's invariant-factor vector; the scalar is its join." | `structure_mode_vector` |
| Approximation obeys it | "Approximating to any tolerance always works; *exact* representation is order-relative — an unbounded ladder vs a hard resist pole." | `OrderRelativeApprox` / `…Graded` / `…LadderK` |
| No gaps left | "Machine-checked end to end — the last analysis step (independent sums) is now in Lean too." | `indepFun_integrable_add_iff` |
| Referee-free | "Axiom-clean (`propext`, `Classical.choice`, `Quot.sound`); `lake build` re-verifies." | the `AxiomAudit` gate |

---

## Keep-true box (the *entire* boundary — three lines)

1. It's a **clean organizing / synthesis law**, not a result about any famous hard problem (no
   P-vs-NP, no Riemann, no learnability). Frame it as elegance + machine-checking, not a breakthrough.
2. Say **"a schema, not a single universal number"** — the order is per-instance; don't imply one
   global scalar ranks everything.
3. The composition law is **one general lemma that applies exactly to the group-order axes** — and we
   *proved* the non-group axes (algebraic degree, denominators) fall outside it. Say "order composes
   as a join **where the structure is a group order**," not "always."

That's it. Anything consistent with these three is fair game.

---

## Integration points

- **Already staged** (committed locally, not yet deployed): the Order-Relative callouts in
  `docs/index.html`, `docs/README.md`, and `docs/SUNDOG_V_CERTIFICATE_LEAN.md` are at **seven axes**
  with the honesty guard updated. The composition law, approximation axis, and structure theorem are
  **not yet surfaced publicly** — that's the new copy above.
- **Suggested placement:** extend the existing "Order-Relative Resolution Law (synthesis core)"
  section with the short paragraph; optional webdev callouts: the 3-positive/2-negative table, a
  "4 ⊕ 6 = order 12, not 6" mini-figure, or the approximation ladder (`id` 1 < `ReLU` 2 < … < ⊤).
- **Source of truth / internal detail (don't surface verbatim):**
  `ORDER_RELATIVE_LAW.md` (the one statement + Lean anchor), `ORDER_COMPOSITION_LAW.md` (the
  composition law, grading abstraction, structure theorem), `ORDER_APPROX_AXIS.md` (the approximation
  dimension). The copy above is the public render.

## Go-live gating (the only two blockers, both owner-gated)

1. The Lean modules are committed **locally** on `humiliati/sundogcert` but **not pushed** — don't
   link the public repo for the composition-law / approximation / structure-theorem results until the
   owner pushes.
2. The site copy is committed locally but **not deployed** (`npm run deploy`, owner-gated).

Copy can be finalized and reviewed now; it goes live in step with those two pushes.

---

*Sundog Research Lab — promo/webdev handoff v3, consolidated. The complete Order-Relative lane: seven
axes, the composition law as one general lemma with axis instances, the approximation dimension, and
the structure-theorem mode-vector — machine-checked end to end, no prose gaps. Three-line boundary,
paste-ready copy, staged integration. Supersedes v2 (composition-only) and v1 (over-fenced). Internal;
deploy owner-gated.*
