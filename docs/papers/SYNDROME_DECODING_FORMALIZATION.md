# Mechanising the Hardness of Syndrome Decoding
## A Machine-Checked `3SAT ≤ 3DM ≤ X3C ≤ Decoding` Chain in Lean 4

*Working draft — venue-neutral Markdown. This file contains the **Abstract**, **§1 Introduction**, and
**§7 Related Work** only; the technical sections (§2 Background, §3 The reduction chain, §4 The GF(2)-native
decoding reduction & the imported wall, §5 Engineering & experience report, §6 The certificate application)
are TODO. Citations marked `[TODO-cite]` need their canonical bibliographic details confirmed before
submission; the four anchor citations (Gäher–Kunze, Balbach, Kreuzer–Nipkow, Affeldt et al.) are verified.*

---

## Abstract

The security of code-based cryptography — including McEliece-style systems, a long-standing post-quantum
reference point — rests on the conjectured intractability of *syndrome decoding*: given a parity-check matrix over
`GF(2)`, a target syndrome, and a weight bound, decide whether some error vector of bounded Hamming weight
produces that syndrome. Its NP-completeness, established by Berlekamp, McEliece, and van Tilborg in 1978, is a
textbook result; yet, to the best of our knowledge, the reduction-correctness core of that result has never
been mechanized in a proof assistant.

We present a complete, axiom-clean Lean 4 / mathlib formalization of the reduction-correctness spine of the
classical Karp chain
`3SAT ≤ 3DM ≤ X3C ≤ bounded-weight GF(2) syndrome decoding`, proving each link as a *both-directions*
equivalence. The development mechanizes the Garey–Johnson `3SAT ≤ 3-dimensional matching` gadget reduction —
its variable-wheel, clause, and garbage gadgets — which is, to our knowledge, also a first in any proof
assistant, together with the onward reductions to exact-cover-by-3-sets and to syndrome decoding. Following
the discipline of Kreuzer and Nipkow's lattice-hardness formalization, we mechanize *reduction correctness*
and **name, rather than prove**, the surrounding complexity-theoretic apparatus — the class NP, the
polynomial-time bound on the reductions, and 3SAT's own NP-hardness via Cook–Levin; mathlib supplies none of
this scaffolding.

A central lesson is a subtlety the textbook construction glosses over: the "obvious" route to bounded-weight
decoding via subset-sum is **unsound over `GF(2)`**, where the parity check is a sum modulo 2 that discards
carries (XOR-subset-sum is in P). We route instead through a covering / odd-cover construction native to
`GF(2)`. Every theorem depends only on Lean's three foundational axioms, build-enforced by a `#guard_msgs`
gate; the development — roughly 2,050 lines across eleven modules — is public and reproducible by `lake build`.

---

## 1. Introduction

Code-based cryptography is among the most mature foundations for post-quantum security: McEliece-style
cryptosystems have been studied since 1978 and have remained central reference points in standardization
efforts, and their security is tied to the difficulty of decoding random-looking linear codes. The decision form of
that problem — **syndrome decoding** — asks, given a parity-check matrix `H` over `GF(2)`, a target syndrome
`s`, and a weight bound `w`, whether some error vector `e` of Hamming weight at most `w` satisfies `H · e = s`.
Berlekamp, McEliece, and van Tilborg [BMvT78] proved this problem NP-complete, giving code-based cryptography
its standard worst-case complexity baseline. This is not an average-case security proof, but it is the textbook
hardness result that every account of the problem must route through.

NP-hardness reductions of this kind are the load-bearing structure of complexity theory, but they are seldom
mechanized. The Cook–Levin theorem — that SAT is NP-complete — has been formalized in Coq [GK21] and in
Isabelle/HOL [Bal23], each building the machine model, the classes P and NP, and polynomial-time many-one
reducibility from scratch. Beyond SAT itself the mechanized landscape thins quickly: the largest collection of
formalized Karp reductions, the Isabelle `poly-reductions` project [PolyRed], covers SAT to independent set,
vertex cover, clique, set cover, and Hamiltonian cycle; the Coq complexity library adds *k*SAT to clique. The
classical gadget reductions to *3-dimensional matching* and *exact cover*, and the reductions into *coding
theory*, have — to the best of our knowledge — not been mechanized at all. In Lean 4 the situation is starker
still: mathlib, the standard library, contains no notion of P, NP, SAT, or polynomial-time reduction, and no
coding theory beyond the Hamming metric and norm — no linear codes, syndromes, or parity-check matrices.

This paper closes part of that gap. We give a complete, machine-checked formalization, in Lean 4 with mathlib,
of the reduction-correctness chain

> `3SAT  ≤  3DM  ≤  X3C  ≤  bounded-weight GF(2) syndrome decoding`,

where **3DM** is 3-dimensional matching and **X3C** is exact-cover-by-3-sets. Each link is proved as a
*both-directions* equivalence — a satisfying assignment exists iff a perfect matching exists iff an exact
cover exists iff a bounded-weight error decodes — and the chain composes to a single headline theorem relating
3-CNF satisfiability to syndrome decoding. The first link is the Garey–Johnson [GJ79] truth-setting reduction,
with its variable-wheel, clause, and garbage gadgets, which we believe is mechanized here for the first time.

Following the discipline established by Kreuzer and Nipkow in their formalization of lattice-problem hardness
[KN23], we are precise about what is and is not proved. We mechanize **reduction correctness**: the many-one
equivalences `inst ∈ A ⟺ reduce(inst) ∈ B`, which are pure combinatorics and the substance of every gadget
argument. We do **not** mechanize the complexity-theoretic *wrapping*: the class NP, the polynomial-time bound
on the reduction functions, and the NP-hardness of 3SAT itself (Cook–Levin). These are *imported* — named
explicitly as the assumptions a hardness reading depends on, rather than smuggled into the development.
mathlib offers no machine model with which to formalize them. Consequently, the Lean theorem is an unconditional
reduction-correctness theorem, while the **NP-hardness label** imports the usual Cook–Levin and polynomial-time
wrapping. Any *intractability* reading is further conditional on `P ≠ NP`; we make no claim about P versus NP.
This is the same boundary Kreuzer and Nipkow draw for the lattice problems, where polynomial-time-ness is
"discussed but not formalized" and base-problem hardness is taken as given.

Mechanization repays the effort by surfacing a subtlety the textbook construction passes over. The standard
route from exact-cover-style problems to *bounded-weight decoding* runs through subset-sum; but **subset-sum
is the wrong target over `GF(2)`**, where the parity check `H · e` is a sum modulo 2 and the carries that make
integer subset-sum hard are discarded — XOR-subset-sum is solvable in polynomial time by Gaussian
elimination. We route instead through a *covering* (odd-cover) construction native to `GF(2)`, in which the
syndrome's parity bits encode an exact cover directly. Recognizing and repairing this mismatch — invisible at
the level of an informal "and now reduce to decoding" — is, in the spirit of [KN23], a contribution of the
formalization itself.

Our contributions are:

- **(C1)** To the best of our knowledge, the **first machine-checked reduction-correctness spine for the
  NP-hardness of bounded-weight `GF(2)` syndrome decoding** in any proof assistant — the decision problem shown
  NP-complete by [BMvT78], with the NP / polynomial-time wrapping imported as stated above.
- **(C2)** To our knowledge, the **first mechanization of the Garey–Johnson `3SAT ≤ 3DM` gadget reduction and
  of `3DM ≤ X3C`** in any proof assistant.
- **(C3)** A **`GF(2)`-native exact-cover-to-decoding reduction**, and the identification and repair of the
  subset-sum-over-`GF(2)` pitfall.
- **(C4)** The complete chain as a **single, axiom-clean Lean 4 / mathlib development** of roughly 2,050 lines
  across eleven modules, with axiom-cleanliness build-enforced — to our knowledge, the first Lean
  formalization of a classical coding-theory hardness reduction of this kind.
- **(C5)** An **experience report** on the *name-the-imported-wall* discipline, the proof-engineering walls we
  met, and a brief account of the AI-assisted workflow used to build the development.

The full development is public [Artifact] and reproducible by `lake build`; every headline theorem's axiom
footprint is pinned by a `#guard_msgs` gate, so a `sorry`, a `native_decide`, or any stray axiom fails the
build. The remainder of the paper is organized as follows. §2 fixes the four problems and the Lean/mathlib
setting. §3 develops the reduction chain — the wheel and clause gadgets, the global assembly, and the forward
and reverse correctness proofs. §4 presents the `GF(2)`-native decoding reduction and discusses precisely what
the development imports. §5 is the engineering and experience report. §6 relates the result to the worst-case
complexity background for code-based cryptography. §7 surveys related work and §8 concludes.

---

## 7. Related Work

**Cook–Levin and the complexity wrapping.** The foundational result we import — that SAT is NP-hard — has two
from-scratch mechanizations. Gäher and Kunze [GK21] formalized the Cook–Levin theorem in Coq over the weak
call-by-value λ-calculus *L*, building the classes P and NP, polynomial-time many-one (Karp) reductions, and
NP-completeness, as part of the Coq Library of Complexity Theory atop the Coq Library of Undecidability Proofs
[CLUP]. Balbach [Bal23] gave an Isabelle/HOL formalization over deterministic multi-tape Turing machines,
proving SAT NP-hard via an oblivious verifier and emphasizing that both the complexity-theoretic concepts and
the polynomiality of the reduction use the *same* model of computation. Our development depends on exactly this
result and treats it as the terminal imported wall; neither formalization addresses 3DM, X3C, or decoding.

**Mechanized Karp reductions.** Downstream of SAT, the Isabelle `poly-reductions` project [PolyRed]
formalizes a family of reductions — 3CNF-SAT to independent set, vertex cover, clique, and set cover, together
with Hamiltonian-cycle variants and vertex cover to feedback node set — with explicit polynomial running-time
bounds, going beyond reduction correctness to *poly-time* reductions. The Coq complexity library contributes
*k*SAT to clique. None of these touches 3-dimensional matching, exact cover, subset-sum, or any coding problem;
the gadget reductions we mechanize are disjoint from this corpus. As of [GK21] (2021), the downstream Karp
reductions — including 3DM and exact cover — were listed explicitly as enabled-but-not-yet-undertaken future
work.

**Lattice hardness — the methodological precedent.** Closest in genre is Kreuzer and Nipkow's [KN23]
Isabelle/HOL verification of NP-hardness reduction functions for the exact lattice problems CVP and SVP (in
the `ℓ∞` norm), via reductions from subset-sum and partition. The scope matches ours almost exactly: they
verify reduction correctness, take the base problems' NP-hardness *as given*, and explicitly discuss — but do
not formalize — polynomial-time-ness (their §7). Notably, their formalization *uncovered gaps in the textbook
proofs* it mechanized, lending the experience report a contribution independent of the underlying mathematics.
CVP is the lattice analogue of nearest-codeword decoding, making this work *coding-adjacent*; but it is a
distinct problem domain over a different algebraic structure, it does not subsume the syndrome-decoding result,
and we adopt its honest experience-report framing rather than competing with it.

**Formalized coding theory.** The most developed formalized coding theory in any assistant is the Coq/MathComp
`infotheo` library [AGS20], which formalizes linear error-correcting codes and Hamming, Reed–Solomon, BCH,
Reed–Muller, and cyclic codes, with LDPC sum-product decoding and an extracted verified decoder, alongside
Shannon's theorems. Crucially, this is decoding-*algorithm correctness*, not decoding-*hardness*: `infotheo`
proves that decoders compute correctly, not that decoding is hard — the two are orthogonal, and only the
latter is our subject. In Lean, mathlib provides only the Hamming metric and norm (`InformationTheory.Hamming`);
a search of mathlib for parity-check, syndrome, generator, or linear-code primitives returns nothing. Recent
external Lean work has begun to formalize coding-theory infrastructure in specialized domains — notably
Lean-QEC's stabilizer-code and verified distance-certificate pipeline [ELWT26] — but that work verifies
distance certificates rather than a syndrome-decoding NP-hardness reduction. Our development therefore
contributes, to our knowledge, the first coding-theoretic *hardness* formalization in any system and introduces
the bounded-weight `GF(2)` syndrome-decoding decision problem into a Lean setting.

**The gap.** No prior work mechanizes the hardness of syndrome decoding, the Garey–Johnson `3SAT ≤ 3DM`
reduction, or `3DM ≤ X3C`, in any system. Lean's complexity-theory ecosystem is young: recent work includes a
Lean 4 preprint formalizing complexity results and reduction infrastructure for a different family of decision
problems [Sim26], while other community efforts remain definitional, specialized, or unpublished. None of these
mechanizes any link of our chain or any syndrome-decoding hardness result. We will re-survey immediately before
submission.

---

## References

*(Verified anchors marked ✓; entries marked `[TODO-cite]` need canonical details confirmed.)*

- **[BMvT78]** E. R. Berlekamp, R. J. McEliece, H. C. A. van Tilborg. *On the Inherent Intractability of
  Certain Coding Problems.* IEEE Transactions on Information Theory, 24(3):384–386, 1978.
- **[GJ79]** M. R. Garey, D. S. Johnson. *Computers and Intractability: A Guide to the Theory of
  NP-Completeness.* W. H. Freeman, 1979. (Theorem 3.2, `3SAT ≤ 3DM`.)
- **[GK21]** ✓ L. Gäher, F. Kunze. *Mechanising Complexity Theory: The Cook–Levin Theorem in Coq.* ITP 2021,
  LIPIcs vol. 193, 20:1–20:18.
- **[Bal23]** ✓ F. J. Balbach. *The Cook–Levin Theorem.* Archive of Formal Proofs, 2023.
- **[KN23]** ✓ K. Kreuzer, T. Nipkow. *Verification of NP-Hardness Reduction Functions for Exact Lattice
  Problems.* CADE-29, 2023 (Springer LNCS); arXiv:2306.08375; AFP entry `CVP_Hardness`.
- **[AGS20]** ✓ R. Affeldt, J. Garrigue, T. Saikawa. *A Library for Formalization of Linear Error-Correcting
  Codes.* Journal of Automated Reasoning, 64:1123–1164, 2020. DOI 10.1007/s10817-019-09538-8. (`infotheo`.)
- **[ELWT26]** `[TODO-cite]` M. Ehatamm, Y. Lee, X. Wu, R. Tao. *End-to-End Formalization of Quantum Error
  Correction.* arXiv:2605.16523, 2026. (`Lean-QEC`; confirm author spellings and artifact details.)
- **[Sim26]** `[TODO-cite]` T. Simas. *Computational Complexity of Physical Counting.* arXiv:2601.15571,
  2026. (Lean 4 complexity/reduction infrastructure for an orthogonal decision-problem family.)
- **[CLUP]** `[TODO-cite]` Y. Forster et al. *A Coq Library of Undecidability Proofs.* (CPP 2020 / project; confirm authors + venue.)
- **[PolyRed]** `[TODO-cite]` The Isabelle `poly-reductions` project (`github.com/wimmers/poly-reductions`); confirm the canonical paper/authors (e.g., the associated Isabelle/HOL formalisation of poly-time Karp reductions).
- **[mathlib]** `[TODO-cite]` The mathlib Community. *The Lean Mathematical Library.* CPP 2020; module `Mathlib.InformationTheory.Hamming` (v4.30.0).
- **[Artifact]** The public Lean development: `github.com/humiliati/sundogcert` (confirm anonymization policy for double-blind submission).

---

*Drafting notes (not for submission):*
- *Headline claims are hedged "to the best of our knowledge / in any proof assistant" per the prior-art scout;
  re-run the negative searches near submission (the no-mechanization claims rest on repo snapshots + exhaustive
  search).*
- *Do not claim "first complexity-reduction work in Lean": current search finds at least one 2026 Lean 4
  complexity/reduction preprint. Keep the Lean first-claim narrowed to syndrome decoding / coding-hardness.*
- *Do not conflate decoding-algorithm correctness (which `infotheo` has) with decoding-hardness (which nobody
  has) — that distinction is load-bearing for (C1).*
- *Cite mathlib's coding-theory absence by actual file content (grep), never a paraphrased docstring.*
- *§5 will carry the brief, honest AI-assisted-workflow note (recon → build → adversarial), per the agreed
  framing — mentioned, not the spine.*
- *Double-blind venues (CPP/ITP) require anonymizing the artifact link and the "sundogcert / Sundog" framing;
  the code-based-crypto motivation (not the lab-specific narrative) is the application anchor in §6.*
