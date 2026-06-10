# §5/§6 CPP Sharpening Proposals — 2026-06-10

**STATUS: P1–P4 APPROVED by owner and APPLIED 2026-06-10** (P4 in the prose-only
variant) to all three sources — MD master, LIPIcs tex, CPP tex. Both PDFs rebuilt
clean afterward: CPP 8 pp (limit 12 excl. bib), LIPIcs 13 pp; zero overfull/undefined;
new sections visually verified. **P5 (scoped anonymized artifact) remains OPEN** —
approved direction, not yet built.

Why these exist: the drafting notes gate CPP on "sharpening the proof-engineering /
experience-report angle (§5)". CPP's CFP names "mechanized (un)decidability and
computational complexity proofs" and "certified mathematical libraries" as topics, has an
artifact-evaluation track, and its audience rewards *portable technique* and *typed trust
stories*. Every sentence below is grounded in the artifact (file + line checked
2026-06-10, rev `d5d1223`); no new technical claims.

Page budget: CPP allows 12 pp excluding bibliography; the draft is 7 pp **including**
bibliography. All proposals together add ≈0.9 pp → lands ≈8 pp. No cuts needed.

---

## P1 — §5.1: generalize the axiom-audit gate into a portable pattern  (~0.15 pp)

**Why.** Right now §5.1 says what the gate does for *this* development. CPP readers want
the takeaway pattern they can adopt tomorrow, plus the non-obvious gotcha. Both are
already documented in `AxiomAudit.lean`'s header — the paper just doesn't surface them.

**Where.** Append to §5.1 (after "...can no longer silently regress.").

**Draft text.**

> The pattern is small and project-agnostic: one audit module imports every load-bearing
> module and pins one `#guard_msgs in #print axioms` block per exported theorem;
> extending the gate is one block per new result, and guarding the top-level
> `sat_iff_decodes` transitively protects the whole chain (its axiom set would change if
> any dependency's did). Before the gate, axiom-cleanliness was checked by a human
> reading `#print axioms` output; now the kernel's own output is the regression test,
> with no CI scripting outside the build. One practical subtlety: `#guard_msgs` captures
> *every* message the command emits, so the gate interacts with active linters — under a
> full `lake build` the mathlib whitespace linter also fires, and each `#print axioms`
> must sit on its own line at column~0, or the linter's message lands in the captured
> text and spuriously fails the gate.

**Grounding.** `AxiomAudit.lean:12–25` (human-ritual line, column-0/linter note,
extension recipe verbatim); `AxiomAudit.lean:233–235` ("Guarding the top-level
`sat_iff_decodes` transitively protects the whole chain").

---

## P2 — new §5.4 "What ports out of this development"  (~0.35 pp; renumber workflow note to §5.5)

**Why.** The conclusion's "Growing the corpus" paragraph *names* the reusable pieces in
one sentence; CPP reviewers will ask "how would I actually reuse them?" Putting the
substance in §5 makes C5 (the experience report) earn its contribution billing. The
conclusion sentence stays as the echo.

**Where.** New subsection after §5.3 ("The development at a glance"); current §5.4
("A note on the workflow") becomes §5.5.

**Draft text.**

> ### What ports out of this development
>
> Three pieces of this development are reduction-agnostic.
>
> **The reindexing bridge.** Decision problems are conventionally stated over an opaque
> index (`Fin s → W×X×Y`), but gadget arguments live over structured indexes (here, a
> four-way sum of products). The generic bridge
> `ThreeDM_I t ↔ ThreeDM (t ∘ e.symm)` (along `e := Fintype.equivFin _`) lets the entire
> correctness proof run over the natural index and transports the result to the opaque
> statement at the end. Any mechanized gadget reduction whose instance space has
> structure — which is to say, any gadget reduction — needs exactly this bridge, and it
> is independent of 3DM.
>
> **The counted-bijection pattern.** Gadget constructions routinely end with "absorb the
> leftovers with garbage" — an informal step that, mechanized, becomes: partition the
> resource (here, tips) into consumed / claimed / unclaimed classes, prove the three
> cardinalities, and obtain the absorbing bijection non-constructively from
> `Fintype.equivFinOfCardEq`. The shape (three-way partition count, then a counted
> bijection) is the reusable unit; the 543-line forward proof is mostly instances of it.
>
> **The encoding lock.** Before any gadget depends on the 3SAT encoding, two
> kernel-`decide` examples (one satisfiable, one unsatisfiable, exercising both literal
> signs) pin its semantics; the polarity bridge is additionally locked by a four-case
> $(\mathit{sign} \times a_i)$ `decide`. Locking an encoding with decidable concrete
> instances *before* building on it is cheap insurance against the silent sign error —
> the class of bug a reduction cannot recover from — and costs the kernel milliseconds.

**Grounding.** §3.4/§3.5 of the paper (bridge, partition, bijection — already claimed);
`SATNPHard.lean:15–18, 72–74, 95–97` (decide examples, kernel-reduce, no
`native_decide`); paper §3.3 (four-case decide). Line counts verified 2026-06-10.

---

## P3 — §5: artifact-evaluation paragraph  (~0.1 pp)

**Why.** CPP has an artifact-evaluation track; the paper never says "this is what the
artifact is and how long it takes". One paragraph converts the AE submission from an
afterthought into a claim the paper itself makes.

**Where.** End of §5.3 ("The development at a glance"), or a one-paragraph §5.x
"Artifact" — owner's preference.

**Draft text.**

> The artifact is the development itself: from a clean checkout, `lake exe cache get`
> followed by `lake build` re-elaborates every proof and re-runs the axiom gate atop the
> cached mathlib, in minutes on commodity hardware; we re-verified exactly this flow from
> a fresh clone for this submission. There is nothing else to trust: no vendored
> binaries, no extracted code, no `native_decide` — the trusted base is the Lean kernel,
> the three foundational axioms, and the pinned toolchain (Lean and mathlib v4.30.0).

**Grounding.** Clean-checkout verification run 2026-06-10 (clone → cache get → `lake
build`, 3,529 jobs, exit 0, AxiomAudit green; cache ≈5 min, total ≈35 min wall).
Deliberately number-free in the text because the job count covers the full repo — see P5;
if the scoped artifact is built, the numbers can be re-measured and stated.

---

## P4 — §6: surface the typed wall (CertWall's four theorems)  (~0.25–0.35 pp)

**Why.** §6 is currently all prose; its only machine-checked content is the `Decodes`
identification. But the artifact's `CertWall` module already *types* the consumer side of
the hardness import with four kernel-checked facts. Surfacing them gives §6 the
distinctly-CPP shape: the imported assumption pinned between a machine-checked lower
anchor (this paper's chain) and a machine-checked upper boundary (the typed wall). This
is the strongest single addition for this venue.

**Where.** New paragraph in §6, after "...that certificate's imported hardness wall to
the canonical NP-hard problem."

**Draft text.**

> The wall is typed, not only named. The artifact's `CertWall` module defines the
> semantic quantity behind the certificate — `minCosetWeight`, the least Hamming weight
> over a syndrome's coset, whose bounded decision form is exactly the `Decodes` our chain
> lands on — and proves the *shape* of the import from the consumer's side. The true
> quantity is a code invariant: a change of parity-check basis (left-multiplication by an
> invertible matrix) leaves it unchanged (`minCosetWeight_rowEquiv`). The verifier's
> cheap sound bound provably is not: on two bases of the *same* code it returns the true
> distance on one and collapses to zero on the other
> (`colWeightLb_not_basis_invariant`, a witnessed counterexample). And a conditional
> theorem (`tight_bound_decodes`) types the closing observation: a cheap bound that is
> *tight* on a syndrome computes the decoding distance there. Jointly: a cheap,
> basis-robust, tight bound would *be* a decoder, so the cheap bound's basis-dependence
> is the visible edge of the imported hardness — now machine-checked on both sides,
> anchored below by this paper's chain (decoding is at least as hard as
> \textsc{3sat}, modulo the named wrapping) and bounded above by the typed conditional
> (anything that closed the gap would discharge the import). No hardness is proved on
> either side; what is kernel-checked is where the assumption lives.

**Variant (owner choice).** Optionally add the verbatim conditional as a listing
(requires a half-sentence gloss of `Scheme`/`Verifier`: "a packaged code with weight
threshold, and a verifier carrying a sound lower bound `lb`"):

```
theorem tight_bound_decodes (S : Scheme F)
    (V : Verifier S) (y : Fin S.n → F)
    (hTight : TightAt S.H V.lb (S.H *ᵥ y)) :
    V.lb (S.H *ᵥ y)
      = minCosetWeight S.H (S.H *ᵥ y)
```

My recommendation: prose only (skip the listing) — it avoids importing two undefined
types into the paper and keeps §6 at one new paragraph.

**Grounding.** `CertWall.lean` in full (read 2026-06-10): `minCosetWeight` (l.62),
`safe_iff_minCosetWeight_le` (l.75), `minCosetWeight_rowEquiv` (l.108),
`colWeightLb_not_basis_invariant` (l.175, the m-vs-0 witnessed pair),
`tight_bound_decodes` (l.196, verbatim above), synthesis
`colWeightLb_cannot_be_tight_basisRobust` (l.207); `DecodingNPHard.lean:82–84`
("**The decoding question** (decision form of `minCosetWeight (Hmat c) allOnes ≤ q`)").
The module's own honesty ledger (header l.36–41) matches the claim boundary.

**Consistency note.** §4.3's existing sentence ("a `CertWall` module *types* the imported
quantity") stays; this paragraph is its §6 payoff. The MD master and the LIPIcs build
should receive the same paragraph to prevent version drift.

---

## P5 — artifact scoping for review (submission mechanics, NOT paper text)

**Why (two independent reasons).**
1. **Anonymity:** the repo contains non-paper modules (`ShadowDecay*`, `HaloGeometry`,
   `FaradayAB`, `RSCertificate`, `SortingCert`) whose distinctive subject matter mirrors
   the public sundog program — a reviewer who searches the topics can de-anonymize even
   if the repo name is scrubbed.
2. **AE hygiene:** evaluators should build exactly what the paper claims, not 18 extra
   modules; `AxiomAudit.lean` currently imports the whole repo and would need trimming
   anyway.

**Proposal.** Build the anonymized review artifact as a scoped export: the eleven chain
modules + `CertWall` and its dependency closure (`Certificate`, `Looseness`, and whatever
they pull — to be computed mechanically) + a trimmed `AxiomAudit` covering exactly the
included headline theorems + `lakefile`/`lean-toolchain`/manifest + a neutral README.
Verify `lake build` green on the export, re-measure the P3 numbers there, then upload to
an anonymized host (e.g. anonymous.4open.science) as the `supplementary material` link.
If P4 is adopted, `CertWall` is in the paper's story and must be in the export; if P4 is
rejected, it still must be (it is named in §4.3).

**Cost.** A few hours of mechanical work + one clean-build verification. I can do all of
it except the upload (owner-gated publishing action).

---

## Suggested adoption

P1 + P3 are low-risk, pure-win for this venue. P2 is the direct answer to the drafting
notes' "sharpen §5" gate. P4 is the strongest addition but adds one paragraph of new
technical surface — read it against the boundary discipline before approving (it was
written to stay inside it). P5 is required regardless of the others, for the anonymity
reason alone.

On approval: apply adopted texts to all three sources (MD master, LIPIcs tex, CPP tex),
rebuild both PDFs clean, and re-run the verbatim-listing check if the P4 listing variant
is chosen.
