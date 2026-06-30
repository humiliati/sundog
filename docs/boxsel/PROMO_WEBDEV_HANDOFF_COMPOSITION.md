# Promo / Webdev Handoff — Order-Relative Law: seven axes + the composition law (v2, ACTIONABLE)

**This supersedes the v1 handoff** (`PROMO_WEBDEV_HANDOFF_ORDER_RELATIVE.md`), which fenced the claims
so hard the team couldn't tell what they *could* say. This version leads with ready-to-paste copy and
compresses the guardrail to three keep-true lines. **You can ship from this directly.**

---

## TL;DR — what's new and shippable

The **Order-Relative Resolution Law** (already on the site as a "synthesis core") has grown from 2 to
**seven** machine-checked instance families, and now has a **composition law**: a proof that
"difficulty" is *vector-valued* and composes as a lattice join. All axiom-clean, all build-gated.

**You can act on this now:** grab any block of copy below, drop it into the existing Order-Relative
section (it's already staged at "seven axes" in `index.html` / `README.md` /
`SUNDOG_V_CERTIFICATE_LEAN.md`). No new fencing decisions required — the keep-true box is the whole
boundary.

---

## Paste-ready copy (pick a length)

**Tagline (≤ 12 words):**
> One machine-checked law of resolution, now instantiated seven different ways.

*(alt: "Difficulty has an order — and we proved the order composes like a grading.")*

**One sentence (sidebar / meta description):**
> Sundog's Order-Relative Resolution Law is a single Lean-verified schema — a target is resolvable
> within budget *k* iff its "order" is ≤ *k* — now grounded in seven instance families and equipped
> with a machine-checked composition law showing that order is vector-valued and composes as a
> lattice join.

**Short paragraph (card / section body):**
> The **Order-Relative Resolution Law** is one axiom-clean Lean core: a bounded process resolves a
> target exactly when the target's *order* fits its budget (`Resolves k ↔ ord ≤ k`) — so "determine"
> means finite order and "resist" means infinite order. It is now instantiated in **seven grounded
> families**, from combinatorial parity and coordinate-locality to algebraic degree and the
> topological flux period behind the Aharonov–Bohm effect. A **composition law** shows the order is
> natively *vector-valued*: the scalar order of a product is the lattice **join** (lcm) of its parts —
> and this holds **exactly on "group-order" axes**, with a proven counterexample where the converse
> fails. Every result is axiom-clean and re-checked by the build's `#print axioms` gate.

---

## Facts you can cite (each is true and machine-checked)

| Plain-language claim | One-liner you can use | Checked by |
|---|---|---|
| One law, many instances | "Seven instance families of a single Lean-checked law." | `OrderRelative` + 6 sibling modules |
| The law itself | "Resolvable within budget *k* iff order ≤ *k*; determine = finite, resist = infinite." | `resolves_iff` / `resolvable_iff_finite` |
| Order is vector-valued | "The same object can carry different orders on different axes." | the mode-vector theorems |
| Composition = join | "The order of a product is the lattice join (lcm) of its parts." | `compose_order_eq_lcm` |
| Sharp, not hand-waved | "For 4 and 6 the composite order is 12 (the lcm), not 6 (the max)." | `compose_lcm_not_max` |
| A clean characterization | "This holds exactly on the group-order axes." | the 3-positive / 2-negative classification |
| Honest boundary, proven | "The converse fails — we proved a join-homomorphic order that is *not* a group order." | `converse_fails` |
| Referee-free | "Axiom-clean (`propext`, `Classical.choice`, `Quot.sound`); `lake build` re-verifies." | the `AxiomAudit` gate |

---

## Keep-true box (the *entire* boundary — three lines)

1. It's a **clean organizing/synthesis law**, not a result about any famous hard problem (no
   P-vs-NP, no Riemann, no learnability). Frame it as elegance + machine-checking, not a breakthrough.
2. Say **"a schema, not a single universal number"** — the order is per-instance; don't imply one
   global scalar ranks everything.
3. The composition law is **axis-internal** (the group-order axes), *not* a universal cross-axis
   identity — "order composes as a join *where the structure supports it*."

That's it. Anything consistent with these three is fair game.

---

## Integration points

- **Already staged** (committed locally, not yet deployed): the Order-Relative callouts in
  `docs/index.html`, `docs/README.md`, and `docs/SUNDOG_V_CERTIFICATE_LEAN.md` were bumped from
  "two grounded axes" to **seven**, with the honesty guard updated. The composition law is *not* yet
  surfaced publicly — that's the new copy above.
- **Suggested placement:** extend the existing "Order-Relative Resolution Law (synthesis core)"
  section with the short paragraph; optionally add a small visual (the 3-positive/2-negative table, or
  a "4 ⊕ 6 = order 12, not 6" mini-figure) as a webdev-friendly callout.
- **Source of truth / internal detail:** `docs/boxsel/ORDER_COMPOSITION_LAW.md` (the full ledger —
  don't surface its internal phrasing verbatim; the copy above is the public render).

## Go-live gating (the only two blockers, both owner-gated)

1. The new Lean modules are committed **locally** on `humiliati/sundogcert` but **not pushed** — so
   don't link the public repo for the *composition-law* theorems until the owner pushes.
2. The site copy is committed locally but **not deployed** (`npm run deploy`, owner-gated).

Copy can be finalized and reviewed now; it goes live in step with those two pushes.

---

*Sundog Research Lab — promo/webdev handoff v2. Actionable render of the Order-Relative composition
law (seven axes + the join-composition characterization). Three-line boundary, paste-ready copy,
staged integration. Supersedes the over-fenced v1. Internal; deploy owner-gated.*
