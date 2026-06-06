# Sundog vs. IUT / abc — a Verification-Crisis Telescope

> **STATUS: DRAFT SCAFFOLD, unpromoted, no public surface.** Opened 2026-06-05 as a
> flagship-demonstration + portfolio lane, NOT a mathematics lane. No public-facing copy, no
> `site-pages.json` entry, no gallery surface until owner sign-off and an evidence-tier review. House
> template: `SUNDOG_V_P_V_NP.md`. This lane is an application of the **certificate / claim-boundary
> verifier** discipline (`docs/pvnp/SUNDOG_CERTIFICATE_PROBLEM.md`) to the most famous real
> verification crisis in mathematics.

Working hook:

> When the community cannot agree whether a proof checks, the honest verdict is not *accept* or
> *reject* — it is a **named quarantine** with a capacity envelope. abc/IUT is the world's hardest
> test of that discipline, and the cleanest possible demo of the flagship.

Short version:

> Sundog has **no view** on whether the abc conjecture is proven or whether Inter-universal
> Teichmüller theory (IUT) is valid. Sundog has a verifier that can say, precisely and in a named
> grammar, *why no one can currently check* — and that named abstention is the demonstration.

---

## 0. Boundary first (the loudest section — do not bury it)

**This lane makes zero mathematical claims about abc or IUT.** It does not evaluate Corollary 3.12 of
IUT III, the log-theta-lattice, multiradiality, or any anabelian content. It does not take a side in
the Mochizuki ↔ Scholze–Stix dispute. Sundog has neither the competence nor the resources to, and the
lab's own `SCIENTIFIC_CRITERIA.md` evidence-tier discipline **forbids** it (this is R3-territory for
the mathematics; the lane lives at R1-meta).

What the lane studies is the **verification situation** — the publicly documented structure, cost,
and topology of the dispute — as a worked case study in the lab's claim-boundary verifier discipline.

**Self-consistency clause (load-bearing).** A lane whose entire thesis is *"the honest move is
principled abstention"* must itself abstain from the mathematics. If this lane ever opines on whether
abc is proven, it has falsified its own thesis and must be quarantined (§6). The lane's verdict on the
*mathematics* is, by construction, a permanent `named_quarantine`. That is not a weakness of the
lane — it **is** the lane.

---

## 1. Why abc/IUT is the right Tycho (four resonances)

The lab is Kepler with a finished instrument (the claim-boundary verifier) looking for a Tycho — a
rich, real, high-stakes dataset to demonstrate it on. abc/IUT qualifies on four independent axes:

1. **abc is itself a shadow/projection statement.** The radical `rad(n)` = product of the distinct
   primes of `n` — a map that *forgets multiplicity*, a maximally-lossy multiplicative projection.
   abc asserts that this lossy radical-shadow *nearly determines* additive structure
   (`c <_ε rad(abc)^{1+ε}`). That is the program's native question — *does a low-dimensional lossy
   projection control the body?* — transposed to ℤ. It is the **number-theory entry in the
   substrate-rhyme gallery** (cap-set polynomial rank; unit-distance Minkowski projection; the
   compander bottleneck; the chatv2 determining-shadow-set). Earning that card requires **no** claim
   that abc is true.

2. **It is the most legible verification-crisis dataset that exists.** Publicly documented record
   only: the IUT papers (I–IV) appeared in PRIMS (2021), of which Mochizuki is chief editor (a
   verification-*governance* datum in itself); Scholze & Stix's 2018 report ("Why abc is still a
   conjecture") locates a claimed fatal gap at IUT III **Corollary 3.12**; Mochizuki published
   rebuttals asserting a conflation of distinctions IUT keeps; no proof-assistant formalization
   exists; verification has stayed confined to a small, non-independent circle; monetary prizes have
   been publicly announced around IUT. The community has not produced a clean accept/reject in over a
   decade — the exact gap the widget's three-valued verdict fills.

3. **It is the single best refusal-demo prompt for the flagship.** The widget's killer property is
   *0 unsafe-accepts across 5,670 adversarial trials* — it knows when not to answer. The most
   seductive prompt to break that discipline is *"Is abc proven? Is IUT correct?"*, on which a naive
   LLM confabulates a confident (wrong-either-way) verdict. Ask Sundog, built right, returns a
   **structured quarantine** that names the capacity boundary instead.

4. **The dispute's internal logic rhymes with the lab's failure taxonomy — observable without
   entering it.** Stripped of anabelian content, the Scholze–Stix ↔ Mochizuki standoff is an argument
   about **whether a particular identification/projection is load-bearing or may be collapsed** —
   which is the lab's native §6.x question ("did the shadow silently upgrade / was a load-bearing
   distinction collapsed?"). The lane may *note this structural rhyme* precisely because its verdict
   is to **quarantine, not resolve**.

---

## 2. What is honest vs. what is reach

**Honest:**
- Modeling the abc/IUT situation as a `named_quarantine` under a registered verification-capacity
  envelope, in the lab's existing receipt grammar (Faraday Branch-C; certificate §6.4 verifier
  overhead).
- A **measured** proof-verification-cost study (§7 Phase 2) using only externally observable signals
  (independent-verifier count, time-to-consensus, existence of a formalization,
  reproducibility-without-the-author), with formalized proofs as the cheap-check positive control and
  abc/IUT as the measured outlier.
- The radical-as-lossy-projection **framing rhyme** as a portfolio resonance.
- A flagship demonstration of principled, named abstention on a famous question.

**Reach; do not claim:**
- "Sundog has a view on whether abc is proven / IUT is valid."
- "Sundog's instrument resolves (or could resolve) the dispute."
- "The radical-as-shadow rhyme is a mathematical insight into abc."
- "IUT is wrong / right." / "Scholze–Stix are right / wrong."
- "Sundog evaluated Corollary 3.12."
- Any statement that reads as a new route to, or opinion on, abc.

---

## 3. Ratified hook language

**Safe:**
> Sundog vs. IUT/abc demonstrates how a capacity-bounded claim-boundary verifier names a quarantine
> on a proof it cannot check — using the most famous unresolved verification question in mathematics
> as the worked example.

**Avoid:**
- "Sundog weighs in on abc." / "Sundog's take on IUT."
- "We show IUT is (in)valid."
- "The shadow framing cracks abc."

---

## 4. Core definitions

- **Proof-as-certificate.** A proof is a certificate for a theorem; the verification promise (the
  NP-shaped one) is that *checking the certificate is cheaper than finding it*, and that checking is
  *independent of the author*.
- **Social verification-capacity envelope `D`.** The promise domain: the set of proofs checkable by
  independent verifiers within bounded effort and without the originator's private framework. Inputs
  outside `D` must **quarantine**, never silently accept.
- **The quarantine verdict.** Three-valued, as everywhere in the lab: `accept` (independently
  re-checked / formalized), `reject` (a fatal gap is independently confirmed), `quarantine`
  (checkability not established within `D`). abc/IUT's correct verdict is `quarantine`.
- **Verification-cost curve.** Observable check-cost vs find-cost across a spectrum of famous proofs;
  the certificate framework's `capacity-relative` axis ported to human mathematics.

---

## 5. Pre-registered framing claims (about the verification situation, NOT the mathematics)

1. The abc/IUT situation is correctly modeled as a `named_quarantine` under `D`, with the locus of
   the unresolved indeterminacy nameable from public documents (the Cor 3.12 keep-distinct-vs-collapse
   disagreement is *itself* unadjudicated).
2. Formalized proofs (Lean/`mathlib`, Flyspeck/Kepler, Gonthier four-color & odd-order) are the
   **cheap-check positive control**: check-cost collapsed and verification became author-independent.
3. On the verification-cost curve, abc/IUT is a **measured outlier** where the check-cheaper-than-find
   promise visibly did not hold.
4. The quarantine is **capacity-relative and updates** (the falsifier): if IUT is independently
   verified or formalized, abc/IUT *moves on the curve* and the verdict changes — built-in, not
   hand-waved.

---

## 6. Falsification surface — how the LANE fails (scope-violation gates)

The lane is killed (`blocked_by_scope_violation`) the instant it does any of:

| gate | violation | disposition |
| --- | --- | --- |
| **adjudication** | states/implies a view on whether abc is proven or IUT valid | scope violation — kill; this is R3, forbidden |
| **math engagement** | engages the mathematics beyond citing third-party documents as facts | scope violation — kill |
| **resolution claim** | claims the instrument resolves/adjudicates the dispute | category error — the verdict is quarantine, not resolution |
| **rhyme-as-math** | over-reads the radical-as-shadow rhyme as a mathematical claim | over-claim — demote to framing-only |
| **reputational** | any public-adjacent move without owner sign-off + evidence-tier review | hold — internal only |

These gates are the lane's primary discipline. The seduction is real: the radical-as-projection
rhyme *will* tempt the lab into believing it has a mathematical angle. It does not. The discipline
that has repeatedly saved this lab — read-before-asserting, the "not a Millennium attack" guardrail,
the JEPA-0D and Direction-B kills — is what gates this lane.

---

## 7. Roadmap (phases)

### Phase 0 — Document spine (third-party facts only)
Compile the publicly documented record (PRIMS publication + chief-editor governance note;
Scholze–Stix 2018 report naming Cor 3.12; Mochizuki's published rebuttals; formalization status;
verification-independence status; announced prizes) as **cited fact**, with the boundary in §0
restated at the top. **Exit:** the doc can state the dispute without opining; a lit-pass memo
(`docs/iut/IUT_ABC_LITPASS_MEMO.md`) mirrors the P-vs-NP memo's discipline.

### Phase 1 — The quarantine receipt
Treat "is IUT a valid proof of abc?" as a verifier query; emit a `named_quarantine` receipt in the
house grammar (Faraday Branch-C / certificate §6.4), naming the capacity envelope `D` and the locus
of the unresolved indeterminacy. Asserts nothing about abc's truth — only that *checking did not
become cheap/independent*, an external fact. **Exit:** a receipt that a skeptic agrees is
scope-clean.

### Phase 2 — The verification-cost curve (the measurable artifact)
The real portfolio receipt. Place a spectrum of famous proofs on an observable check-cost-vs-find-cost
axis:
- **cheap-check pole (positive control):** Lean/`mathlib`, Flyspeck (Hales/Kepler), Gonthier
  four-color & odd-order — author-independent, machine-recheckable.
- **the body:** Fermat–Wiles (gap found *and closed*: quarantine→accept), Perelman/Poincaré
  (independent reteams collapsed the check), classification of finite simple groups (decades; a famous
  gap; second-generation proof).
- **the outlier:** abc/IUT — check-cost never fell below find-cost; verification stayed
  non-independent.
Signals are **observable only** (independent-verifier count, time-to-consensus,
formalization-exists, reproducibility-without-author). **Exit:** a measured curve + a named outlier +
the capacity-relative falsifier; zero correctness claims about any proof.

### Phase 3 — Flagship demo
Add the abc/IUT quarantine prompt to the chat gold slate (`chat/prompts/gold-boundary.jsonl` /
`gold-falsification.jsonl`); the widget (`public/js/sundog-chat-widget.mjs`) must **demonstrably**
return a structured quarantine, not a confabulated verdict; route it cleanly in
`chat/claim_map.json`. **Exit:** the widget passes the hardest boundary prompt in the slate.

### Phase 4 — Public surface (GATED)
The radical-as-shadow rhyme card on `/unit-distance`'s grid; an `/abc-quarantine` overlay (sibling to
`/unit-distance`) walking a lay reader through "how a claim-boundary verifier handles a proof it
cannot check." **Hard gate:** owner sign-off + `SCIENTIFIC_CRITERIA.md` evidence-tier review before
any public-adjacent move (reputational weight is real). Until then, internal only.

---

## 8. Promotion criteria

- **To Active (internal):** Phase 0–2 land scope-clean (a skeptic confirms zero math claims) and the
  verification-cost curve has a named outlier + a live falsifier.
- **To Flagship demo:** Phase 3 — the widget demonstrably quarantines the prompt in the slate.
- **To Public:** Phase 4 only, and only with owner sign-off + evidence-tier review. Do **not** promote
  to any "result about abc/IUT" — there is none, by construction.

## 9. Cross-references

- `docs/pvnp/SUNDOG_CERTIFICATE_PROBLEM.md` — the certificate / capacity-envelope framework this lane
  applies to mathematical proof.
- `docs/SUNDOG_V_P_V_NP.md` — finding-vs-checking, verifier-overhead (§6.4), 3-valued verifier.
- `faraday/SUNDOG_V_FARADAY.md` — the structural-zero / named-quarantine receipt grammar (Branch A–D).
- `docs/SCIENTIFIC_CRITERIA.md` — evidence tiers; the discipline that forbids the math lane.
- `internal/feedback/Human/REDDIT_ImOutOfIceCream_UNIT-DISTANCE.md` — the substrate-rhyme / projection
  vocabulary; abc/radical is the number-theory rhyme card.
- `docs/CROSS_SUBSTRATE_NOTES.md` — the cross-substrate generality map (where this lane's rhyme sits).
- `public/js/sundog-chat-widget.mjs`, `chat/claim_map.json`, `chat/prompts/gold-*.jsonl` — the
  flagship demo surface.

## 10. Forbidden language

- "Sundog proves / disproves / has a view on abc." / "Sundog evaluated IUT / Corollary 3.12."
- "IUT is (in)valid." / "Scholze–Stix / Mochizuki are right / wrong."
- "The shadow framing resolves abc." / "Sundog's instrument adjudicates the dispute."
- Any phrasing that reads as a new route to, or opinion on, the mathematics.

## 11. One-paragraph public summary (draft, DO NOT DEPLOY)

Sundog vs. IUT/abc does not try to decide whether the abc conjecture is proven. It uses the most
famous unresolved verification question in mathematics as a worked example of a discipline: when a
claimed proof cannot be checked independently within bounded effort, a healthy verifier does not
guess — it issues a **named quarantine** and says exactly why. The lane places that case on a
verification-cost curve next to proofs that *were* independently re-checked or machine-formalized, and
demonstrates the Sundog flagship doing the one thing a naive system won't: refusing to confabulate a
verdict on the question it most wants to answer.

---

*Sundog Research Lab — SUNDOG_V_IUT_ABC scaffold. A verification-crisis telescope, not a mathematics
lane. The verdict on the mathematics is a permanent named quarantine, by construction. Internal;
unpromoted; no public surface until owner sign-off.*
