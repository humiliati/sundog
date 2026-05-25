# Shadow Faraday Zero-Out: Minimal Experiment Roadmap (v0.1)

**Status**: Phase 6 local site readiness complete; post-deploy validator pass pending
**Owner**: Sundog Research Lab (Humiliati + collaborators)  
**Goal**: Test whether the Sundog shadow physics framework (local gauge-invariant observables, σ₃-style detectors, tidal proxies) algebraically "zeroes out" on classical electromagnetism — specifically, whether local shadow projections are sufficient to recover Faraday's law of induction and the Lorentz invariants of the electromagnetic field tensor without global state reconstruction.  
**Style**: Low-compute, pre-registered, receipt-driven, falsification-first. Mirrors isotrophy v0.3 / three-body controller discipline.  
**Compute envelope**: Purely algebraic + symbolic (hand or SymPy) + tiny optional Python spot-check. No MuJoCo, no heavy simulation, no new ML.  
**Success framing**: Exact algebraic identity (structural zero) **or** named, physically interpretable quarantine term. Either outcome is a high-value receipt.

---

## Phase 1: Algebraic Foundations & Cross-Domain Mapping

**Objective**: Establish shared notation and map existing Sundog primitives onto the Faraday tensor without introducing new assumptions.

**Tasks**:
- Write canonical definitions:
  - Faraday tensor: \( F_{\mu\nu} = \partial_\mu A_\nu - \partial_\nu A_\mu \)
  - Electric & magnetic fields from \( F \)
  - Two Lorentz invariants: \( F_{\mu\nu}F^{\mu\nu} \) and \( F_{\mu\nu}\tilde{F}^{\mu\nu} \)
  - Integral form of Faraday's law: \( \oint \mathbf{E} \cdot d\mathbf{l} = -\frac{d\Phi_B}{dt} \)
- Map Sundog objects:
  - σ₃ detector → gauge-invariant 2-form contractions
  - Tidal tensor / local acceleration proxies → local field gradient operators
  - Existing gauge-cocycle / Procrustes machinery → EM gauge transformations
- Pre-register:
  - Full symbol table
  - Gauge-fixing conventions (e.g., Lorenz or Coulomb gauge where helpful)
  - Explicit assumption list (classical vacuum EM, flat spacetime or local inertial frame, no sources/monopoles initially)
  - Falsification priors (what would count as a structural failure)

**Receipts to produce**:
- Symbol glossary + assumption ledger (versioned)
- One-page "mapping table" (Sundog primitive ↔ EM object)

**Exit criteria**: Symbol table and mapping signed off; no new physics introduced.

---

## Phase 2: Local Shadow Projection Operator

**Objective**: Define a local shadow projection operator \( P_\text{shadow} \) that extracts gauge-invariant local differences from the EM field (or potential) without requiring global reconstruction of \( A_\mu \).

**Tasks**:
- Formalize \( P_\text{shadow} \) analogously to the three-body tidal tensor / local probe sampling.
- Prove (or audit) that \( P_\text{shadow} \) commutes with gauge transformations (preserves gauge invariance like σ₃).
- Explore whether isotropy/Floquet averaging or twist-operator ideas from v0.3 add robustness (optional, keep minimal).
- Identify the minimal locality radius or stencil needed.

**Pre-reg & Audit**:
- Three-stage audit style (structural zero receipts + named quarantine + intact chain)
- Explicit test: does \( P_\text{shadow}(A + d\lambda) = P_\text{shadow}(A) \)?

**Receipts**:
- Definition of \( P_\text{shadow} \) with gauge-invariance proof sketch
- First quarantine log (any non-local residuals)

**Exit criteria**: Operator defined and gauge-invariance audited.

---

## Phase 3: Core Zero-Out — Faraday Induction from Shadows

**Objective**: Algebraically derive the integral form of Faraday's law using only shadow-projected local E and B, showing exact cancellation (or controlled quarantine) of any global reconstruction terms.

**Tasks**:
- Compute line integral of shadow E around a local loop.
- Compute time derivative of shadow magnetic flux through the spanned surface.
- Show:
  \[
  \oint (P_\text{shadow} \mathbf{E}) \cdot d\mathbf{l} + \frac{d}{dt} \int (P_\text{shadow} \mathbf{B}) \cdot d\mathbf{A} \stackrel{?}{=} 0
  \]
  (or identify the precise residual term).
- Repeat for the two Lorentz invariants expressed via shadow data.
- Classify outcome: clean structural zero vs. named quarantine (e.g., topological term, boundary term at infinity, etc.).

**Pre-registration (critical)**:
- Exact success predicate before any calculation.
- Falsifier: "If reconstruction terms survive at order X, the claim is bounded."

**Receipts**:
- Full derivation (handwritten or SymPy notebook)
- Closure residual table
- Quarantine taxonomy (if any)

**Exit criteria**: Derivation complete with signed pre-reg outcome.

---

## Phase 4: Symbolic Verification & Minimal Falsification Battery

**Objective**: Verify the derivation on canonical cases and run a tiny falsification suite. Keep strictly minimal.

**Tasks**:
- Hand or SymPy verification on:
  - Uniform constant B field (trivial control)
  - Smooth source-free plane wave (nontrivial clean-domain pass candidate)
  - Optional: infinite solenoid or simple oscillating dipole only as a named
    quarantine / sourced-domain extension, not as a required clean-domain pass
- Design 2–3 minimal falsifiers:
  1. Deliberately non-local projection → should produce residual
  2. Add artificial monopole or source term → expected failure mode
  3. Gauge transformation applied after projection → must remain invariant
- Optional tiny Python spot-check (stdlib only or single numpy import) for numerical consistency on one example. No MuJoCo, no heavy deps.

**Receipts**:
- Verification notebook / annotated derivation
- Falsification battery results table (Pass / Partial / Fail with explanations)
- Updated closure residuals

**Exit criteria**: All planned verifications executed; falsifiers documented.

---

## Phase 5: Chapter Close, Internal Documentation & Handoff

**Objective**: Close the pre-registered chapter with durable receipts and prepare internal handoff (mirroring isotrophy v0.6 / three-body patterns).

**Tasks**:
- Write full chapter-close note:
  - Outcome (clean zero-out / named quarantine / bounded failure)
  - All receipts, residuals, and falsifiers
  - Limitations & scope (classical only, etc.)
  - Suggested next minimal extensions (e.g., with sources, curved spacetime, or plasma)
- Create or update `docs/SHADOW_FARADAY.md` (or append to existing isotrophy/threebody docs)
- Prepare any pair-ID / gauge-cocycle style catalog if multiple cases emerged
- Quick fidelity audit on the derivation chain

**Receipts**:
- Signed chapter-close document
- Updated docs/ with clear "pre-registered negative / partial / positive" header
- Handoff note for any future teammate

**Exit criteria**: Chapter closed with receipts; docs updated and audited.

---

## Phase 6: Public Site Integration — faraday.html (or shadow.html)

**Objective**: Expose the result (and the rigorous process) on the public site in the same transparent, receipt-heavy style as threebody.html and isotrophy.html.

**Tasks**:
- Create new root-level page: `faraday.html` (preferred) **or** `shadow.html` if we decide to broaden the shadow metaphor page.
  - Model structure after `threebody.html` + `isotrophy.html`:
    - Hero / claim boundary
    - Mathematical setup (KaTeX for all equations)
    - Shadow Projection definition
    - Derivation + key receipts (structural zeros, quarantine taxonomy)
    - Falsification battery summary
    - Limitations & scope
    - "Come break it" replication invitation with GitHub links
    - Links back to `docs/SHADOW_FARADAY.md` and raw receipts
- Update `site-pages.json` (or equivalent nav) so the new page appears in menus / applications rail if appropriate.
- Add a concise card or link from `index.html` (Applications / Legend / Load-Bearing Evidence rail) — keep language precise and non-hype.
- Optional: simple SVG or minimal diagram showing shadow projection acting on EM field lines (can be hand-drawn or generated later).
- Cross-link from `/legend` (shadow theory) and `/threebody` (local proxies) for coherence.

**Tone & Guardrails** (same as existing site):
- "Local shadows suffice for Faraday induction in the classical case — with these explicit receipts and falsifiers."
- Own any quarantine or scope limits immediately.
- Invite external replication and critique.

**Exit criteria**:
- `faraday.html` (or `shadow.html`) live at `https://sundog.cc/faraday`
- Navigation updated
- Page passes basic accessibility / consistency check with existing HTML pages
- Announcement language drafted (receipt-heavy, zero fluff)

---

## Overall Timeline (Minimal / Scrappy)

- **Phase 1–2**: 1–2 days (mostly notation + operator definition)
- **Phase 3**: 1–2 days (core algebra)
- **Phase 4**: 1 day (verification + falsifiers)
- **Phase 5**: 0.5–1 day (close + docs)
- **Phase 6**: 1 day (HTML page + nav)

**Total**: ~5–7 focused days for a complete, pre-registered, publicly documented minimal experiment.

---

## Review Questions Resolved (2026-05-25)

1. Canonical page stays `faraday.html`; a broader `shadow.html` umbrella is
   deferred until there is more than one shadow-substrate result.
2. Phase 4 must include uniform constant `B`, a smooth source-free plane wave,
   and an artificial monopole/source insertion falsifier. The time-varying
   solenoid is optional and quarantine-facing.
3. Phase 3 remains strictly source-free and contractible. Minimal sourced or
   topological cases are allowed only in Phase 4 as quarantine demonstrations.
4. Hand / exterior-calculus derivation is authoritative. SymPy or tiny Python
   checks may verify signs and examples, but cannot replace the algebraic
   receipt or add new outcome branches.

---

**Next step**: Deploy and record the post-deploy LinkedIn/Twitter validator
pass for `https://sundog.cc/faraday`. Local Phase 6 / Bucket 1 artifacts are
now present: designed `1200x630` `og:image`, JSON-LD `TechArticle`, tuned
title/description, inbound link from `index.html`, and sitemap entry.

**Progress note (2026-05-25)**: Phase 1 is now opened in
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md) with the first symbol table,
assumption ledger, primitive mapping, and pre-registered outcome branches.

**Progress note (2026-05-25, later)**: Phase 2 is now opened in
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md) ▸ "Phase 2: Local Shadow Projection
Operator". `P_shadow` is candidate-defined as the plaquette-holonomy operator
on `A` in two tiers (finite-stencil `oint A` and point-limit `F_{mu nu}`), with
an explicit locality receipt, a gauge-invariance proof under
`A -> A + d lambda` (via Stokes on a closed loop), an admissibility rule for
`(S, partial S)` pairs, and five named quarantine hooks (regularity, topology,
monopole, operator-stencil commutator, motional EMF). The four Phase 2 open
questions are now resolved in the ledger: coordinate plaquettes; point-limit
gate plus finite-stencil locality receipt; no Floquet/twist enrichment in
Phase 3; two-tier operator retained with roles locked.

**Progress note (2026-05-25, takeoff)**: Phase 3 is cleared for takeoff in
[`SHADOW_FARADAY.md`](SHADOW_FARADAY.md) ▸ "Phase 3: Takeoff Gate". The gate
locks allowed inputs, forbidden shortcuts, derivation work order, exact success
predicate, closure residual table, landing branches, and the dedicated receipt
target.

**Progress note (2026-05-25, Phase 3 landed)**: Phase 3 receipt landed in
[`FARADAY_PHASE3_DERIVATIONS.md`](FARADAY_PHASE3_DERIVATIONS.md), with
proof-hygiene corrections to the form-degree Stokes statement and
finite-stencil scaling. The registered clean-domain residuals all land as
structural zeros, so the Phase 3 branch is **A - clean structural zero**. At
this checkpoint, Phase 4 verification / falsification was still owed before
chapter close or external sharing.

**Progress note (2026-05-25, Phase 4 landed)**: Phase 4 verification and
minimal falsification landed in
[`FARADAY_PHASE4_VERIFICATION.md`](FARADAY_PHASE4_VERIFICATION.md), with
support artifacts from `npm run faraday:phase4`. The battery passed 5/5:
constant `B`, source-free plane wave, nonlocal projection residual, artificial
monopole quarantine, and finite-plaquette gauge invariance.

**Progress note (2026-05-25, Phase 5 closed)**: Phase 5 chapter close is now
recorded in [`SHADOW_FARADAY.md`](SHADOW_FARADAY.md) ▸ "Phase 5: Chapter
Close". The experiment lands **Branch A — clean structural zero** on the
registered classical-vacuum domain. The chapter close carries a full receipts
catalog, the consolidated closure-residual summary, seven explicit scope
limitations (sourced EM, non-contractible topology, distributional fields,
curved spacetime, quantum EM, plasma, moving surfaces), six suggested next
minimal extensions, and a fidelity audit that independently hand-verifies the
Phase 4 nonlocal-falsifier residual to twelve digits and the finite-stencil
leading O(epsilon) coefficient. Four soft hygiene notes are recorded
transparently with no Branch-A impact. The experiment chapter is **closed**;
only Phase 6 (page promotion / Bucket 1 readiness) remained for `/faraday` at
this checkpoint.

**Progress note (2026-05-25, Phase 6 local readiness)**: `/faraday` now has
full local site-readiness treatment: `public/og/faraday.png` (1200×630),
OG/Twitter metadata, JSON-LD `TechArticle`, tuned title/description, a homepage
pillar link, `site-pages.json` promotion to `evidence-page`, and
`public/sitemap.xml` coverage. The only remaining external action is the
post-deploy social validator pass.

This keeps the Sundog standard: traceable, falsifiable, receipt-driven, and public. The final artifact is both a mathematical result **and** a clean public page that demonstrates the method. 

Ready when you are.
