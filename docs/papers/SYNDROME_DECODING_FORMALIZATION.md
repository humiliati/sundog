# Mechanising the Hardness of Syndrome Decoding
## A Machine-Checked `3SAT ≤ 3DM ≤ X3C ≤ Decoding` Chain in Lean 4

*Working draft — venue-neutral Markdown. This file contains the full paper arc: **Abstract**, **§1
Introduction**, **§2 Background**, **§3 The reduction chain**, **§4 The GF(2)-native decoding reduction &
imported wall**, **§5 Engineering & experience report**, **§6 The certificate application**, **§7 Related Work**,
**§8 Conclusion**, and **References**. Citations marked `[TODO-cite]` need their canonical bibliographic details
confirmed before submission; verified anchors are marked ✓. Drafting notes at the end are not for submission.*

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

## 2. Background

### 2.1 The four decision problems

**3SAT.** A 3-CNF formula over `n` Boolean variables and `m` clauses, each clause an ordered triple of
literals; `Satisfiable φ` holds when some assignment satisfies every clause. We represent a literal as a
variable index with a sign (`Fin n × Bool`), a clause as `Fin 3 → Literal`, and a formula as `Fin m → Clause`.

**3-Dimensional Matching (3DM).** Given a family of triples over three ground sets `W, X, Y`, a *perfect
matching* is a selection covering each element of each ground set exactly once.

**Exact Cover by 3-Sets (X3C).** Given a family `c : Fin s → Finset X` of 3-element subsets of a ground set
`X`, an *exact cover* is a selection covering each point exactly once.

**Bounded-weight syndrome decoding.** Given a parity-check matrix `H` over `GF(2)`, a syndrome, and a weight
bound `q`, decide whether some error `e` of Hamming weight `≤ q` satisfies `H · e = s`. This is the decision
form of minimum-distance decoding — the problem [BMvT78] showed NP-complete.

### 2.2 The classical reductions

The chain composes three classical many-one reductions. **(1)** The Garey–Johnson [GJ79] *truth-setting*
reduction `3SAT ≤ 3DM` encodes each variable as a "wheel" gadget admitting exactly two matchings (its two
truth values), each clause as a gadget coverable iff one of its literals is true, with *garbage* elements
absorbing unused gadget tips. **(2)** `3DM ≤ X3C` is a relabelling: a triple `(w,x,y)` becomes the 3-set
`{w, x, y}` over the disjoint union `W ⊕ X ⊕ Y`. **(3)** `X3C ≤ bounded-weight GF(2) decoding` takes the
incidence matrix of the set family as the parity check and the all-ones vector as the syndrome (§4).

### 2.3 Lean 4, mathlib, and axiom-cleanliness

We work in Lean 4 with mathlib v4.30.0. A Lean proof is trusted exactly insofar as one trusts the kernel and
the axioms a result invokes; mathlib's developments rest on three foundational axioms — `propext`,
`Classical.choice`, `Quot.sound`. We call a result *axiom-clean* when `#print axioms` reports only these (or a
subset), with no `sorryAx` (which a `sorry` introduces) and no `Lean.ofReduceBool` (which `native_decide`
introduces, trusting the compiler's evaluator); kernel `decide` is axiom-clean. mathlib supplies finite types,
`Finset`s, matrices over `ZMod 2`, and the Hamming norm — but no complexity theory and no coding theory beyond
the Hamming metric.

---

## 3. The reduction chain

The chain is built bottom-up across nine modules, landing on the `X3C` and decoding endpoints. We describe the
`3SAT ≤ 3DM` reduction — the substantive part — and then the composition.

### 3.1 Encoding 3SAT (`SATNPHard`)

The 3SAT definitions are standard combinatorics over finite types, so satisfiability is decidable on concrete
instances. We lock the encoding *before* the gadgets depend on it with two kernel-`decide`-validated examples —
a satisfiable and an unsatisfiable formula — exercising both literal signs.

### 3.2 The variable-wheel gadget (`VarWheel`)

A variable's wheel of size `m` owns, per spoke `j : Fin m`, two internal nodes `a j, b j` and two tips, with a
positive triple `t_j = (posTip j, a j, b j)` and a negative triple `t̄_j = (negTip j, a (j+1), b j)`, where
`j+1` is cyclic in `Fin m`. Since `b j` lies only in `{t_j, t̄_j}`, any internal cover selects one triple per
spoke — a `Selection σ : Fin m → Bool`. Node `a j` is covered exactly once iff exactly one of
`{σ j = true, σ (j−1) = false}` holds; by the Boolean identity `xor a (¬b) = (a = b)`, that collapses to
`σ j = σ (j−1)`. The engine of the gadget is then:

```lean
theorem validCover_iff_const (hm : 0 < m) (σ : Selection m) :
    ValidCover σ ↔ (σ = fun _ => true) ∨ (σ = fun _ => false)
```

A valid internal cover is *constant* — and the two constants are the variable's two truth values. The
forward direction is a strong induction showing a cyclic equal-neighbour selection is constant. Geometrically
(Figure 1) the gadget *is* the even cycle `C₂ₘ`: its `2m` internal nodes are the cycle vertices, its `2m`
triples the cycle edges, and a valid internal cover is a perfect matching of that cycle — of which an even
cycle has exactly two.

**Figure 1.** The variable-wheel gadget. *ASCII preview (the production figure is the TikZ below):*

```
  ring (cyclic, even cycle C_2m; shown for m = 3):
       a0 --t0-- b0 --t~0-- a1 --t1-- b1 --t~1-- a2 --t2-- b2 --t~2-- (a0)

  positive triple  t_j  = (posTip_j, a_j,     b_j)     -- the a_j – b_j edge
  negative triple  t~_j = (negTip_j, a_(j+1), b_j)     -- the b_j – a_(j+1) edge  (+1 a-node shift)

  b_j is shared only by t_j and t~_j  =>  a cover picks ONE per spoke (a Selection sigma).
  a_j is shared by t_j and t~_(j-1)   =>  the cover must be one of the cycle's exactly TWO
                                          perfect matchings -- the variable's two truth values:
     sigma == true   : select all t_j   -> consumes the posTips, FREES the negTips (for clauses)
     sigma == false  : select all t~_j  -> consumes the negTips, FREES the posTips
```

```latex
% Production figure. Requires \usetikzlibrary{calc}.
\begin{figure}[t]
\centering
\begin{tikzpicture}[scale=1.7,
    anode/.style={circle, draw, fill=white, inner sep=1.3pt, font=\footnotesize},
    bnode/.style={circle, draw, fill=black!12, inner sep=1.3pt, font=\footnotesize},
    tip/.style={rounded corners=2pt, draw, fill=white, inner sep=2pt, font=\scriptsize},
    freetip/.style={rounded corners=2pt, draw=red!70!black, thick, fill=red!5,
                    inner sep=2pt, font=\scriptsize},
    sel/.style={line width=1.1pt},
    unsel/.style={gray!70, line width=0.5pt, densely dotted}]
  % internal nodes: a_j open, b_j shaded, alternating around the ring
  \node[anode] (a0) at (90:1.7)  {$a_0$};   \node[bnode] (b0) at (30:1.7)  {$b_0$};
  \node[anode] (a1) at (-30:1.7) {$a_1$};   \node[bnode] (b1) at (-90:1.7) {$b_1$};
  \node[anode] (a2) at (-150:1.7){$a_2$};   \node[bnode] (b2) at (150:1.7) {$b_2$};
  % positive triples t_j = (posTip_j, a_j, b_j): the selected matching (bold)
  \draw[sel] (a0)--(b0); \draw[sel] (a1)--(b1); \draw[sel] (a2)--(b2);
  % negative triples t-bar_j = (negTip_j, a_{j+1}, b_j): unselected (dotted)
  \draw[unsel] (b0)--(a1); \draw[unsel] (b1)--(a2); \draw[unsel] (b2)--(a0);
  % positive tips (consumed by the selected matching)
  \node[tip] (p0) at (60:2.75) {$\mathrm{posTip}_0$};
  \node[tip] (p1) at (-60:2.75){$\mathrm{posTip}_1$};
  \node[tip] (p2) at (180:2.75){$\mathrm{posTip}_2$};
  \draw[sel] (p0)--($(a0)!.5!(b0)$); \draw[sel] (p1)--($(a1)!.5!(b1)$);
  \draw[sel] (p2)--($(a2)!.5!(b2)$);
  % negative tips (FREE -> available to the clause gadgets)
  \node[freetip] (q0) at (0:3.0)    {$\mathrm{negTip}_0$};
  \node[freetip] (q1) at (-120:3.0) {$\mathrm{negTip}_1$};
  \node[freetip] (q2) at (120:3.0)  {$\mathrm{negTip}_2$};
  \draw[unsel] (q0)--($(b0)!.5!(a1)$); \draw[unsel] (q1)--($(b1)!.5!(a2)$);
  \draw[unsel] (q2)--($(b2)!.5!(a0)$);
  % edge labels, just inside the ring
  \node[font=\scriptsize] at (60:1.22)  {$t_0$}; \node[font=\scriptsize] at (-60:1.22){$t_1$};
  \node[font=\scriptsize] at (180:1.22) {$t_2$};
  \node[font=\scriptsize,gray] at (0:1.22)   {$\bar t_0$};
  \node[font=\scriptsize,gray] at (-120:1.22){$\bar t_1$};
  \node[font=\scriptsize,gray] at (120:1.22) {$\bar t_2$};
\end{tikzpicture}
\caption{The variable-wheel gadget for $m=3$ (general $m$: the even cycle $C_{2m}$). Internal nodes $a_j$
(open) and $b_j$ (shaded) alternate around the ring; each ring edge is a triple carrying a tip. The
positive triple $t_j=(\mathrm{posTip}_j,a_j,b_j)$ and the negative
$\bar t_j=(\mathrm{negTip}_j,a_{j+1},b_j)$ share $b_j$, so any internal cover selects exactly one per
spoke (a \emph{Selection} $\sigma$); the $a$-node sharing---$t_j$ contributes $a_j$ while $\bar t_{j-1}$
contributes $a_{(j-1)+1}=a_j$---then forces $\sigma$ to be constant, i.e.\ one of the cycle's exactly two
perfect matchings, the variable's two truth values. The selected matching (bold, all $t_j$) covers every
internal node once, consumes the positive tips, and \emph{frees} the negative tips (boxed) for the clause
gadgets; the other matching (dotted, all $\bar t_j$) does the reverse. The negative triple's $a$-node is
the cyclic successor $a_{j+1}$---the $+1$ shift responsible for the $a$-/$b$-node asymmetry in the
incidence lemmas (\S3.4).}
\label{fig:wheel}
\end{figure}
```

### 3.3 The clause gadget and the polarity bridge (`ClauseGadget`)

The clause gadget's internal pair is coverable iff some literal's tip is free (`clauseCoverable`). The
*polarity bridge* connects the wheel's raw free-tip predicate — under the convention that a variable assigned
`aᵢ` runs the constant selection `fun _ => ¬aᵢ` — to literal evaluation:

```lean
theorem litTipFree_iff_eval (a : Assignment n) (l : Literal n) {m} [NeZero m] (j : Fin m) :
    litTipFree a l j ↔ evalLiteral a l = true
```

A tip is free exactly when its literal is true. This discharges the truth-value labelling and is the one
place a sign error would silently corrupt the reduction; we additionally lock it with a four-case
`(sign × aᵢ)` `decide`.

### 3.4 Global assembly (`SATReduction`, `ThreeDMReindex`, `SATReductionIncidence`)

The three coordinate sets are the tips `Tip = Fin n × Fin m × Bool` (the `W`-part) and
`XNode = YNode = (Fin n × Fin m) ⊕ Fin m ⊕ Fin (m·(n−1))` — variable-internal nodes, clause-internal nodes,
and garbage. Each has cardinality `2mn`; the garbage count `m(n−1)` is exactly what balances the books, and
the equality `n·m + m + m(n−1) = 2mn` requires `n ≥ 1`. A triple family `tripleFn` emits each gadget triple,
and `reduce φ` re-indexes it along the canonical finite equivalence into the `Fin s → W×X×Y` shape the
matching problem expects. Cardinality lemmas feed `reduce_chain_connects`, wiring `reduce φ` onto the
already-formalized `3DM ≤ X3C ≤ decoding` tail.

To keep the heavy counting over the structured sum-type index rather than an opaque `Fin s`, we prove a
generic reindexing bridge `ThreeDM_I t ↔ ThreeDM (t ∘ e.symm)` (along `e := Fintype.equivFin _`) and carry the
gadget arguments over the natural index. Six *incidence* lemmas then characterize which triple-indices cover
each node. The subtle one is a cyclic-shift asymmetry: an `a`-node `inl(i,j)` is covered by `posT(i,j)` and
`negT(i, j−1)`, whereas a `b`-node is covered by `posT(i,j)` and `negT(i,j)` — no shift. A wrong shift here
corrupts the wheel-state extraction downstream.

### 3.5 Forward and reverse correctness (`SATReductionForward`, `SATReductionReverse`)

The **reverse** direction reads an assignment out of a matching: the `b`-node and `a`-node
"covered exactly once" facts force each wheel's `Selection` to be a `ValidCover`, hence constant by the engine
(giving each `aᵢ`); each clause's selected slot's tip is forced free, hence its literal true.

The **forward** direction *builds* the matching from a satisfying assignment: the constant wheel cover per
variable, one satisfied-slot triple per clause, and the leftover free tips absorbed by a counted *garbage
bijection* — `Fintype.equivFinOfCardEq` on the size-`m(n−1)` set of unclaimed free tips. The three cover
conditions split over the disjoint summands, and the tip count is a three-way partition (consumed /
free-and-claimed / free-and-unclaimed), each summing to one.

The forward/reverse asymmetry is striking — `reverse` is 184 lines, `forward` 543: constructing and counting a
perfect matching, garbage bijection included, is materially harder than reading an assignment out of a given
one.

### 3.6 Composition (`SATReductionMain`)

`sat_iff_threeDM_I = ⟨forward, reverse⟩` packages both directions; transported through the reindexing bridge to
the real `Fin s`-indexed `ThreeDM`, then through `reduce_chain_connects`, this yields the headline:

```lean
theorem sat_iff_decodes (φ : Formula n m) :
    Satisfiable φ ↔ Decodes (reduce3DM (reduce φ)) (2 * m * n)
```

— `3SAT ≤ 3DM ≤ X3C ≤ decoding`, closed end to end. (The `3DM ≤ X3C` and `X3C ≤ decoding` endpoints, §4, supply
the tail.)

---

## 4. The GF(2)-native decoding reduction and the imported wall

### 4.1 The subset-sum trap over GF(2)

One might expect the route from a covering problem to bounded-weight decoding to pass through subset-sum:
encode memberships as integers, ask for a low-weight subset summing to a target. **Over `GF(2)` this is
unsound.** The parity check `H · e` is a sum modulo 2 — the carries that make integer subset-sum NP-hard are
discarded, and XOR-subset-sum is solvable in polynomial time by Gaussian elimination. The correct source
structure is *covering*, where mod-2 addition is the native operation. (The development records this as the
reduction's failure mode: a wrong source problem yields a wrong reduction.)

### 4.2 The covering reduction (`DecodingNPHard`)

An `X3C` instance `c : Fin s → Finset X` becomes the parity-check matrix whose column `i` is the GF(2)
indicator of the 3-set `c i`, with the all-ones syndrome:

```lean
def Hmat : Matrix X (Fin s) (ZMod 2) := Matrix.of (fun x i => if x ∈ c i then 1 else 0)
def Decodes : Prop := ∃ e : Fin s → ZMod 2, Hmat c *ᵥ e = allOnes c ∧ hammingNorm e ≤ q
```

The connection lemma makes the encoding exact: for the 0/1 indicator `eOf T` of a selection `T`, the syndrome
coordinate at point `x` is the mod-2 cast of its cover count,

```lean
theorem syndrome_eq_coverCount_cast (T) (x) :
    (Hmat c *ᵥ eOf c T) x = ((coverCount c T x : ℕ) : ZMod 2)
```

so "syndrome = all-ones" says exactly "every point is covered an *odd* number of times". An exact cover (count
one) is odd; conversely a weight-`q` odd cover must be exact, because the weight bound `q = |X|/3` and the
3-uniformity, via the double-count `∑ₓ coverCount = 3·|T|`, pin `|T| = q` and then force every count to one.
Both directions are machine-checked (the iff carries the structural hypotheses the prose uses — that every
set has three elements and that `|X| = 3q`):

```lean
theorem reduction_iff (hc : ∀ i, (c i).card = 3) (hX : Fintype.card X = 3 * q) :
    X3C c ↔ Decodes c q
```

At the chain's head these hypotheses are discharged by the reduction's own cardinality lemmas, so
`sat_iff_decodes` is unconditional (modulo `n, m ≥ 1`). The landed `Decodes` is precisely the decision form of the minimum-coset-weight quantity whose hardness a
syndrome certificate's soundness assumes (§6).

### 4.3 Naming the imported wall

What we do *not* prove is the complexity-theoretic wrapping. The reduction functions are visibly local — each
gadget index maps to one triple, each triple to one 3-set, each 3-set to one matrix column — but
"polynomial-time" is unformalized, as mathlib offers no machine model. NP membership, and 3SAT's own
NP-hardness (Cook–Levin), are likewise imported. We name this boundary in the development: a `CertWall` module
*types* the imported quantity, and every module's documentation states the limit. The Lean output is therefore
an *unconditional reduction-correctness theorem*; the NP-hardness reading attaches the imported wrapping
(Cook–Levin and the poly-time bound), and only an intractability reading further assumes `P ≠ NP`.

---

## 5. Engineering and experience report

### 5.1 Axiom-cleanliness as a build invariant

Every headline theorem is axiom-clean. We make this *self-checking*: an `AxiomAudit` module pins each result's
`#print axioms` output with `#guard_msgs`, so a regression — a stray `sorry`, a `native_decide`, any extra
axiom — changes the captured message and fails `lake build`. The referee-free promise can no longer silently
regress.

### 5.2 Proof-engineering walls

Mechanizing the chain met several walls worth recording.

- **Decidable equality on a four-way sum.** The triple-index type is a four-way nested `Sum` of products;
  `DecidableEq` exceeds the default `synthInstance.maxSize` inside large proof contexts (where the
  filter-rewrites force the search). A scoped `set_option synthInstance.maxSize 512` clears it.
- **Rewriting a filter predicate.** Rewriting a `Finset.filter`'s predicate through an incidence lemma fails
  directly — `DecidablePred` cannot be synthesized at the rewrite site — so we name the rewritten Finset in an
  explicit `have heq := Finset.filter_congr …; rw [heq]`, forcing elaboration with the instance in scope.
- **The garbage bijection.** The forward construction needs a bijection from the index set `Fin (m(n−1))` onto
  the assignment-dependent set of unclaimed free tips; its existence follows from a cardinality equality
  supplied non-constructively by `Fintype.equivFinOfCardEq`, and the counting (`|free|=mn`, `|claimed|=m`,
  `|unclaimed|=m(n−1)`) is the real work.
- **Cyclic `Fin` arithmetic.** The wheel's cyclic predecessor uses `Fin m`'s additive-group structure under
  `[NeZero m]`; the defining `j+1 = j₀ ↔ j = j₀−1` is `eq_sub_iff_add_eq`.

### 5.3 The development at a glance

The chain is roughly 2,050 lines across eleven modules (1,638 in the nine new reduction modules; the `X3C` and
decoding endpoints supply the remainder), comprising on the order of fifty theorems and lemmas. Every headline
result depends only on `[propext, Classical.choice, Quot.sound]` or a subset, and the development rebuilds via
`lake build` atop a cached mathlib, reproducible from the public artifact.

### 5.4 A note on the workflow

The development was produced with substantial AI assistance, under a fixed discipline: each module was scoped
and reconnoitred against the live toolchain, drafted, adversarially reviewed for correctness and over-claim,
then reproduced independently by the author before commit. The axiom-clean `#guard_msgs` gate and the kernel
re-check make this provenance immaterial to the result's *validity* — the kernel certifies the proofs however
they were written — but we record it honestly as how the development was built.

---

## 6. The certificate application

Bounded-weight `GF(2)` syndrome decoding is not an arbitrary target: it is the hardness assumption beneath
code-based cryptography. McEliece-style cryptosystems encrypt by adding a bounded-weight error to a codeword,
and their security is tied to the difficulty of decoding. Our chain machine-checks the reduction-correctness
equivalence between this decoding problem and a canonical NP-hard source problem, anchoring the qualitative
"decoding is hard" intuition to 3SAT, and the landed `Decodes` instance is exactly the decision form of the
minimum-coset-weight quantity a syndrome certificate's soundness rests on — so the development connects, in one
formal artifact, that certificate's imported hardness wall to the canonical source problem.

We are careful about the boundary. [BMvT78] worst-case NP-hardness is the standard complexity baseline every
account of the problem routes through, but it is **not an average-case security proof**: McEliece's security
rests on the average-case hardness of decoding *random* codes — a strictly stronger assumption that worst-case
NP-hardness does not establish. Our contribution puts the worst-case baseline — its reduction-correctness spine
— on a machine-checked footing; it does not certify the cryptographic assumption, and we claim no such thing.

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

## 8. Conclusion and Future Work

We have given a complete, axiom-clean Lean 4 / mathlib formalization of the reduction-correctness spine of the
chain `3SAT ≤ 3DM ≤ X3C ≤ bounded-weight GF(2) syndrome decoding`. Both directions of each link are
machine-checked; the reduction-correctness spine for the Garey–Johnson truth-setting chain and for
bounded-weight `GF(2)` syndrome-decoding hardness is, to the best of our knowledge, mechanized here for the
first time in any proof assistant. The development brings this coding-theoretic hardness problem — in the
parity-check / syndrome / bounded-weight-error vocabulary — into Lean in a form that other formalization and
cryptography projects can inspect, replay, and extend. Throughout, we have been precise about the boundary:
what the kernel certifies is reduction *correctness*; the complexity wrapping — NP membership, the polynomial-time
bound on the reductions, and 3SAT's own Cook–Levin hardness — is named and imported, not proved, following
Kreuzer and Nipkow. The discipline repaid itself by forcing a subtlety into the open: bounded-weight decoding
over `GF(2)` is reached not through subset-sum (unsound over a carry-free field) but through a covering
construction native to mod-2 arithmetic.

**Pushing the wall inward.** The imported wall is not immovable; it is a sequence of further mechanization
targets of increasing difficulty. The polynomial-time-ness of our reductions is the nearest — the maps are
visibly local, and a cost model (a Lean cost monad, or the resource-bounded Turing machines now appearing in
community developments) would let it be *discharged* rather than asserted. NP membership of the four problems
is a similar exercise once such a model is in place. The terminal wall is 3SAT's own NP-hardness, the
Cook–Levin theorem: mechanized in Coq and Isabelle/HOL but not yet in Lean, it is the result that, once ported
or re-proved, would let the entire chain stand on formalized foundations within a single system. Each such step
converts a named assumption into a theorem without disturbing what is already proved.

**Growing the corpus.** The reduction-agnostic parts of this development — the generic reindexing bridge
between `Fintype`-indexed and `Fin s`-indexed matchings, the gadget-counting patterns, and the build-enforced
axiom-clean gate — could seed a wider Lean library of complexity reductions, an area where mathlib is currently
bare. On the coding side, the vocabulary introduced here is a starting point for further coding-theoretic
hardness results.

**The honest distance to cryptography.** Finally, we restate the gap of §6. Worst-case NP-hardness is the
standard complexity baseline for code-based cryptography, but the security of McEliece-style systems rests on
the *average-case* hardness of decoding random codes — a strictly stronger assumption that worst-case hardness
does not establish. Mechanizing an average-case reduction, or the worst-case-to-average-case connections
studied for related problems, would be a substantial undertaking; the worst-case spine formalized here is a
prerequisite for that programme, not a substitute for it.

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
