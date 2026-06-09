# Fresh hypotheses slate #2 (workflow `wm13nclfe`, 2026-06-08)

> Generate→adversarially-vet→rank, the second slate after `ww6koomb1` (which produced H1–H6, now all
> attacked). **24 candidates generated across 6 lab-asset lenses; 18 KILLED, 2 survived.** The vetting was
> brutal and substantive — several killers ran forward probes to refute candidates against their own kill
> criteria, and prior-art search killed most of the "Atlas for X" caustic ideas as founding catastrophe-
> optics (Berry/Nye/Upstill). NOT public-eligible.

## Survivors (ranked)

### Rank 1 — C1: the Reed–Solomon evaluation certificate (strength 6.5)
**A new axiom-clean Lean pillar (sundogcert): k low-degree evaluations DETERMINE the unique codeword while
LOSING which corruptions were applied** — the *evaluation-dual* of the parity-check/coset syndrome
certificate (the two classical RS views).
- **Core (cheap-CHECK soundness):** `rs_unique` — two polynomials of `natDegree < k` agreeing on `k`
  distinct nodes are EQUAL. A *thin specialization* of `Polynomial.eq_of_natDegree_lt_card_of_eval_eq`
  (**confirmed present** at mathlib `Roots.lean:709`; ZMod p gives CommRing+IsDomain). Cheap `Verifier`
  exhibits a degree-`<k` `f` matching `≥ n−τ` positions, O(n·k) forward; `accept_sound`, `no_double_fit`.
- **The NEW theorem (earns its place vs the syndrome pillar):** `corruption_fiber_nontrivial` — the set of
  received words decoding to the SAME `f` is non-trivial (evaluations determine the codeword but LOSE the
  corruption pattern) — the determine/lose split on a DIFFERENT mathlib core (polynomial root-counting).
  Plus a Scaling-style quantitative companion (reject bound from the unique-decoding radius `τ < (n−k+1)/2`)
  and `nodes_distinct_cert` via `det_vandermonde_ne_zero_iff` (**confirmed** `Vandermonde.lean:232`).
- **Imported wall (named, NOT proved):** list-decoding past the unique radius / RS-decoding NP-hardness
  (Guruswami–Sudan; Guruswami–Vardy 2005). The ISD analog.
- **Attack:** `sundogcert/Sundogcert/RSCertificate.lean` mirroring `Certificate.lean`/`Scaling.lean`; axiom
  audit; ~20-line numpy GF(7) sanity. **Kill if** the build needs a sorry, the check isn't O(n·k) forward,
  the find-hardness leaks into the core, or `corruption_fiber_nontrivial` collapses to a restatement.
- **Why it survived:** the cited cores are verified present (thin specialization, not a new proof); the
  cheap-check is genuinely forward; the wall is correctly imported. Most tractable + highest machinery-
  leverage on the slate. Capped at 6.5 by closeness to the syndrome pillar — the new determine/lose
  theorem + radius companion is what escapes "too incremental."

### Rank 2 — H8 (recast): regularization-removal of double descent as a Whitney A3 cusp (strength 5)
**The point in the (label-noise, ridge-λ) plane where double descent's bump ANNIHILATES is a Whitney A3
cusp (fold-pair annihilation)** — paralleling the H5 mirage 1→3 onset.
- **Recast fixes a category error:** the ridgeless (capacity, noise) peak is a variance **POLE** (R ~
  1/(1−γ)), NOT a fold caustic — so "the peak is an A2 fold" is ill-posed and is forbidden. The catastrophe
  framing applies only to the *regularized* risk, where λ smooths the pole into a finite bump with a real
  smooth critical line.
- **Headline test (could go either way):** at the double-descent-vanishes locus (Nakkiran 2020 "optimal
  regularization removes double descent"), is there a c2-sign-change cusp with |c3| BOUNDED (calibrate vs
  the Morin-A4 control + H5's |c3|~426), corank-1? **Clean null:** the bump shrinks monotonically with no
  fold-pair → not a catastrophe (informative).
- **Attack:** random-features ridge regression, CLOSED-FORM expected risk (numpy, no training loop); build a
  genuine generating potential (state = a 1-D order parameter, NOT the ad-hoc `(p, dR/dp)` map — that's a
  2nd-derivative test, not a Lagrangian map); feed the gradient map to `atlas_jet_classify.jet_from_chart`
  → `cusp_c3`. Novelty is narrow: WHICH germ + "regularization-removal = fold-pair annihilation" (the
  folklore "double descent is phase-transition-like" — Belkin/Mei-Montanari/Rocks-Mehta — is NOT the claim).

## The kill record (the discipline is part of the deliverable)
18 killed. The clusters:
- **Published prior art (catastrophe optics is founded ground):** binary-lens & pool D4 umbilics, the
  scintillation / ship-wake / rainbow / supernumerary caustic "atlases" (Berry 1977, Nye, Berry–Upstill,
  DLMF 36.13). The "Atlas for X" lane is largely mined out.
- **Category errors:** charts that aren't Lagrangian maps (logistic-iterate period-doubling; the ad-hoc
  double-descent map), "certificates" with NO cheap-check≫hard-find asymmetry (Euler parity & lottery-
  ticket — Eulerian-trail finding is *polynomial*; W1 caustic-corank).
- **Not-falsifiable / self-refuting:** several determining-shadow extensions whose own forward probe fired
  both kill criteria (correlated draws, charFun-curvature half-life, LMC-mod-permutation, neural-collapse,
  W2 mirage-shadow).
- **Notable repo finding:** `ShadowDecayCauchy.lean` and `ShadowDecayLattice.lean` ALREADY EXIST (the owner
  shipped the Cauchy-separator + lattice-recurrence Lean legs I'd scoped as future work in H2) — which
  killed the Rajchman/Cauchy/lattice salvage paths.

## Recommendation
**C1 (the RS evaluation certificate)** is the strongest, most buildable bet — a real axiom-clean Lean pillar
in bounded hours, with verified mathlib cores. H8 is a good CPU-bounded jet-classifier test with an honest
null possibility.
