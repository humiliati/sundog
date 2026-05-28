# Riemann Lit-Pass Memo

> Single-page record of the 2026-05-28 lit pass that precedes filing
> [`SUNDOG_V_RIEMANN.md`](SUNDOG_V_RIEMANN.md). Records what is in the indexed
> literature as of May 2026 and what is not, so the ledger's claim that the
> Probe 01–04 shortlist attacks non-vacuous gaps can be re-audited.

**Date:** 2026-05-28
**Status:** Filed as the prior-art reference for the four-probe shortlist in
the Riemann ledger. Treat the gap claims here as time-stamped:
"not indexed at lit-pass date," not "does not exist."

## Method

Four searches: (A) zero pair-correlation and higher n-point statistics
2024–2025 state-of-the-art, with Montgomery–Odlyzko as the baseline; (B)
trace formulae / explicit formulae and the periodic-orbit / quantum-chaos
reading (Berry–Keating, Bogomolny, Connes); (C) Hilbert–Pólya spectral
realization candidates and representation-theoretic / symmetry-group framings
on the critical line; bonus track on the explicit-formula → smoothed-trace
projection literature and on choreographic-orbit databases as the natural
substrate analog for Probe 04. Targeted fetches on the 2024
Connes–Consani–Moscovici prolate-wave-operator integration and on the
Baluyot–Goldston–Suriajaya–Turnage-Butterbaugh unconditional Montgomery
theorem. Two-pass: an initial sweep, then sharper queries on path-signature
/ sufficient-statistic methods applied to zero data and on
choreography-↔-zero-substrate work.

## Three tracks

### A — Zero pair-correlation and higher n-point statistics

Montgomery's 1973 pair-correlation conjecture (under RH, Fourier transform of
the pair correlation is `|x|` for `|x| < 1`) and Odlyzko's numerical program
(zeros computed at heights `10^20` and `10^22` with neighbor counts in the
tens of millions to one billion) are the foundational layer. Rudnick–Sarnak
extended the framework to general L-function families and showed that
n-point correlations of suitable test functions agree with the Gaussian
Unitary Ensemble (GUE) prediction; subsequent work pushed the agreement
through two-, three-, and four-point statistics. Keating–Snaith averages over
random unitary matrices reproduce moment-formula asymptotics, separating a
universal random-matrix factor from an arithmetic Euler product factor. This
is mature, heavily-tooled territory.

Recent (2024–2025) progress on the pair-correlation side:
- Baluyot–Goldston–Suriajaya–Turnage-Butterbaugh 2024 proved an
  **unconditional** form of Montgomery's pair-correlation theorem
  ([2306.04799](https://arxiv.org/abs/2306.04799)).
- A January 2025 extension established that at least 2/3 of zeros are simple
  and at least 2/3 lie on the critical line via the pair-correlation method
  ([2501.14545](https://arxiv.org/abs/2501.14545)).
- A March 2025 paper sharpened the pair-correlation conjecture itself
  ([2503.15449](https://arxiv.org/abs/2503.15449)).
- A November 2025 Goldston–Suriajaya note tightened the simple-zero /
  critical-line counts ([2511.20059](https://arxiv.org/abs/2511.20059)).

The agreement between Riemann-zero statistics and GUE / random matrix theory
extends well beyond Montgomery's original window; long-range structural
auto-correlations have been demonstrated numerically over `10^22` zeros (see
[nlin/0405058](https://arxiv.org/abs/nlin/0405058)).

**Gap:** no indexed work applies a Sundog-style structural-zero /
representation-theoretic discipline — the v0.3h K_facet idiom of "twenty
structural zeros plus one named quarantine" — to zero-pair data. The
literature reads zero statistics through random-matrix and analytic-number-
theory lenses, not through finite-catalog isotypic-decomposition lenses. The
gap is real but its load-bearing prerequisite is a **representation bridge**
from `(pair of zeros, unfolded spacing)` to a D3-representation or smaller-
group analog. Without that bridge the gap is not addressable — see the
disposition note on Probe 01 below.

### B — Trace formulae, explicit formulae, periodic-orbit / quantum-chaos reading

The Riemann–Weil explicit formula relates zero sums to prime sums and bears a
"striking and mysterious resemblance to the Selberg trace formula"
([Watkins/Weil](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/weilexplicitformula.htm),
[Wikipedia explicit formulae](https://en.wikipedia.org/wiki/Explicit_formulae_for_L-functions)).
The periodic-orbit reading — primes ↔ periodic orbits of an unknown chaotic
"Riemann dynamics," zeros ↔ semiclassical quantum eigenvalues — is the
Berry–Keating / Bogomolny program, formalized in
[Berry–Keating SIAM Review 1999](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/berry.htm)
and the [Riemann zeros and quantum chaos](http://www.scholarpedia.org/article/Riemann_zeros_and_quantum_chaos)
Scholarpedia article. Recent activity in the program:
- A June 2024 arXiv posting on "spectral flow for the Riemann zeros."
- A 2025 KKR-determinant approach
  ([2504.07928](https://arxiv.org/abs/2504.07928)).
- Connes–Consani–Moscovici 2024 "Zeta zeros and prolate wave operators"
  integrates two spectral realizations into the semilocal trace formula
  framework; Connes' broader adèle-class-space program continues
  ([math/9811068](https://arxiv.org/abs/math/9811068),
  [math/0703392](https://arxiv.org/abs/math/0703392),
  [Connes 2025 "Knots, Primes and the adele class space"](https://alainconnes.org/2025/07/knots-primes-and-the-adele-class-space/)).

Smoothed and explicit-formula approximations to `π(x)` are standard
machinery; smoothed Landau-type explicit formulae have a 2023 instance
([2311.04347](https://arxiv.org/abs/2311.04347)).

**Gap:** there is no indexed work that *publishes a pre-registered residual
threshold* for a specific projection from periodic data to a smoothed zero
count or trace, then runs it as an admit/falsify experiment in the
sundog-discipline shape (Probe 03 territory). This is a small gap — the
explicit-formula approximation literature has all the machinery — but the
pre-registration / falsifier discipline is itself the deliverable, not the
projection. Whether that deliverable is worth filing as a Sundog probe vs.
imported as "this is what a careful analyst already does informally" is the
Front-A-vacuity question for Probe 03.

### C — Hilbert–Pólya, representation-theoretic / symmetry framings, spectral candidates

The Hilbert–Pólya conjecture (zeros as eigenvalues of a self-adjoint
operator) remains the dominant attractor for spectral-realization proposals.
Recent 2024–2025 candidates:
- Yakaboylu's "On the Existence of the Hilbert–Pólya Hamiltonian"
  ([2408.15135](https://arxiv.org/abs/2408.15135), revised through 2026)
  argues for an essentially-self-adjoint construction.
- A 2025 Majorana-relativistic spectral approach in Rindler spacetime
  ([2503.09644](https://arxiv.org/abs/2503.09644)).
- A 2024 matrix formulation of the zero condition off the critical line
  ([2411.17701](https://arxiv.org/abs/2411.17701)).
- Supersymmetric quantum-mechanical models with energy eigenvalues
  corresponding to ζ in the critical strip
  ([1810.02204](https://arxiv.org/abs/1810.02204)).
- Multiple non-arXiv candidate constructions (entropy-geometry, modular-
  resonant, tick-time-fractal) of widely varying quality.

Family-of-L-function symmetry types are governed by classical compact groups
(unitary for ζ; orthogonal / symplectic for other families) per the
Katz–Sarnak framework; the natural symmetry on the critical line itself is
the **Z₂ reflection `s ↔ 1-s`** induced by the functional equation, not
S₃ or D3. This is the load-bearing observation for the Sundog representation
bridge: the v0.3h apparatus is built on the D3 representation of 21 strict
G.2 single-curve choreographies, and the Riemann substrate's natural symmetry
is one order of magnitude smaller in group-theoretic size.

**Gap:** no indexed work applies the v0.3h K_facet pattern — finite-catalog
isotypic decomposition + twist operator + induced-representation case
analysis + F_beta template + tau-flag + structural-zero classifier — to
Riemann-zero data. The gap is broad, but its addressability is gated on the
representation bridge. If the natural bridge is Z₂ (functional-equation
reflection), the v0.3h machinery has to descend from D3 to its Z₂ sub-
representations and the apparatus loses most of its structural reach. If a
candidate S₃ or larger action can be defended (e.g., on consecutive triples
of zeros, or on a zero-prime-reflection composite), the full apparatus might
admit — but no such bridge is in the literature.

## Bonus precedent — functional-equation reflection as the natural Z₂

The reflection `s ↔ 1-s` is the cleanest existing piece of "structural
symmetry on the critical line" and is the natural target for any
isotropy/parity-style analysis. The supersymmetric QM model of
[1810.02204](https://arxiv.org/abs/1810.02204) explicitly uses the
reflection-fixed line as the zero-locating condition (zeros on the critical
line emerge from a vanishing ground-state-energy condition). This is the
existing piece of mathematics that **makes a Z₂-descent of the v0.3h
apparatus not analogy but sibling**. The ledger preamble should cite this so
the framing is not read as imported metaphor.

The Katz–Sarnak symmetry-type taxonomy (unitary / orthogonal / symplectic)
is the existing piece of representation-theoretic discipline on
L-function families. It is *not* a finite-catalog discipline — it operates
at the family / Plancherel level — and its presence in the literature means
any Sundog claim of "we apply representation theory to zeros" needs to
declare what it adds beyond Katz–Sarnak.

## Bonus precedent — choreographic-orbit databases

The 2022 high-precision database of 462 trivial planar choreographies
([2210.00594](https://arxiv.org/abs/2210.00594)) and the 2025
Li–Liao discovery of **10,059 three-dimensional periodic orbits** of the
general three-body problem — including **twenty-one 3D choreographic
periodic orbits** for equal masses
([2508.08568](https://arxiv.org/abs/2508.08568)) — are the substrate-side
analog for Probe 04. The "21" coincidence with the v0.3h strict G.2 catalog
is *numerical only*; nothing in the literature pairs these two "21" objects
as the same family. Worth noting as a re-audit hook, not as a claim.

## Bonus competitor — standard statistical baselines

The baselines any Sundog-style zero-pair invariant has to beat — or explain
its different problem statement against — are extraordinarily well-tooled:
Montgomery pair correlation, Odlyzko numerical tables, Rudnick–Sarnak
n-point, Keating–Snaith CUE moment predictions, and the recent unconditional
Montgomery and 2/3-simple-zeros theorems
([2306.04799](https://arxiv.org/abs/2306.04799),
[2501.14545](https://arxiv.org/abs/2501.14545)). Long-range autocorrelation
work demonstrates invariance with respect to averaging window over `10^22`
zeros ([nlin/0405058](https://arxiv.org/abs/nlin/0405058)). The honest
position: Sundog's structural-zero discipline encodes a different question
(does a catalog row land in a privileged representation block?) than the
statistical-comparison literature does (do empirical spacings match GUE?).
They are likely complementary, not competing. The ledger should not claim
Sundog *replaces* the GUE-statistics framework.

## Bonus competitor — Connes' adèle-class-space program

The Connes / Bost–Connes / Connes–Consani–Moscovici program is the dominant
"global structural" alternative for reading the explicit formula as a trace
formula. It is operative, deep, and has continued activity through 2024
([Connes–Consani–Moscovici 2024 prolate wave](https://arxiv.org/abs/math/9811068)
parent thread; [2025 Knots-Primes-adèles](https://alainconnes.org/2025/07/knots-primes-and-the-adele-class-space/)).
Any Probe 02 / Probe 03 claim that "the explicit formula is a Sundog-style
traceable projection" is operating in territory where Connes already has a
formal trace-formula reduction. The Sundog deliverable in this lane is
operational pre-registration / residual-publication discipline, not a new
trace-formula candidate.

## Updated probe ranking

| Rank | Probe | Δ from ledger draft | Cost | Why this rank |
|---|---|---|---|---|
| 1 | P01 — Isotropy v0.3 on low-lying zero pair data | **conditionally admitted** | Low | Gap exists (structural-zero discipline on zero pairs is not indexed) but the representation bridge from `(zero pair)` to a D3-or-smaller catalog is the load-bearing prerequisite. Default first move: file the bridge notes resolving Z₂-descent vs S₃-via-triple, then admit the Z₂-descent version of P01 for a desk-auditable first run. |
| 2 | P03 — Projection residual on smoothed trace | unchanged | Low-Med | Smoothed explicit formula machinery is mature (Riemann, Weil, Burnol, smoothed Landau). Sundog adds pre-registered residual threshold + publish-the-falsifier discipline. Gap is small but the deliverable is the discipline, not the projection. Reframe explicitly as "operational receipt" not "new projection" to clear Front-A vacuity. |
| 3 | P04 — Three-body stress on abstract orbit analog | **downgraded** | Med | Quantum-chaos / periodic-orbit literature for trace formulae is very deep (Berry–Keating, Bogomolny, Cvitanović school). Adding three-body / choreography stability discipline to candidate orbit analogs is novel-ish but at high risk of being overshadowed by existing work. The numerical 21-vs-21 choreography coincidence is a re-audit hook, not a claim. Defer until P01 lands. |
| 4 | P02 — Choreography invariant alignment to explicit formula terms | **deferred** | High | Aligning n-body choreography invariants (`E`, `|L|`) to smoothed explicit-formula terms via Procrustes + gauge cocycle is structurally suspect: the choreography invariants and the explicit-formula terms live in different physical / algebraic categories, and no published framework couples them. Probably needs to be voided and replaced with a cleaner Front-A reading note. Hold until P01 + bridge notes land. |

## Disposition

Findings sufficient to file the ledger at the LEDGER tier (capset / NSE
pattern). P01 is the only probe with a clear gap that can plausibly be
attacked with the apparatus on hand, *and only* after the representation
bridge resolves. P03 is admissible as an "operational discipline" deliverable
if explicitly de-scoped from "new projection." P04 demoted pending P01.
P02 deferred and likely voided. No claim leaves this memo into a public
surface without an external-mathematician sanity check, per
[`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md) and [`SCIENTIFIC_CRITERIA.md`](SCIENTIFIC_CRITERIA.md)
discipline.

Re-audit point: gap claims here are time-stamped to 2026-05-28. The
unconditional-Montgomery and 2/3-simple-zeros literature is moving in 2024–
2025; any P01 admission older than ~6 months should be re-checked against
the current pair-correlation state of the art before a public-facing claim.

## Sources

Pair correlation, n-point statistics, GUE / random matrix:
- [Montgomery's pair correlation conjecture (Wikipedia)](https://en.wikipedia.org/wiki/Montgomery's_pair_correlation_conjecture)
- [An unconditional Montgomery theorem for pair correlation (2306.04799)](https://arxiv.org/abs/2306.04799)
- [Pair correlation of zeros: proportions of simple and critical zeros (2501.14545)](https://arxiv.org/abs/2501.14545)
- [Pair correlation conjecture for zeros (2503.15449)](https://arxiv.org/abs/2503.15449)
- [Zeta zeros on the critical line (2511.20059)](https://arxiv.org/abs/2511.20059)
- [Correlations among Riemann zeros: invariance, resurgence, self-duality (nlin/0405058)](https://arxiv.org/abs/nlin/0405058)
- [Statistics on Riemann zeros (1112.0346)](https://arxiv.org/abs/1112.0346)
- [L-functions and Random Matrix Theory (AIM)](https://www.aimath.org/WWN/lrmt/lrmt.pdf)

Berry–Keating, Bogomolny, quantum-chaos / trace formula reading:
- [Berry–Keating SIAM Review 1999 (mirror)](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/berry.htm)
- [Riemann zeros and quantum chaos (Scholarpedia)](http://www.scholarpedia.org/article/Riemann_zeros_and_quantum_chaos)
- [Quantum and Arithmetical Chaos (nlin/0312061)](https://arxiv.org/abs/nlin/0312061)
- [Berry–Keating operator on `L^2(ℝ_>, dx)` (0912.3183)](https://arxiv.org/abs/0912.3183)
- [Riemann zeros and KKR determinant (2504.07928)](https://arxiv.org/abs/2504.07928)

Connes / noncommutative-geometry / adèle program:
- [Trace formula in noncommutative geometry (math/9811068)](https://arxiv.org/abs/math/9811068)
- [The Weil proof and the geometry of the adèles class space (math/0703392)](https://arxiv.org/abs/math/0703392)
- [Connes 2025 — Knots, Primes and the adèle class space](https://alainconnes.org/2025/07/knots-primes-and-the-adele-class-space/)

Explicit formulae:
- [Riemann–Weil explicit formula (Watkins mirror)](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/weilexplicitformula.htm)
- [Burnol commentary on Riemann–Weil](https://empslocal.ex.ac.uk/people/staff/mrwatkin/zeta/burnol-weil.htm)
- [Explicit formulae for L-functions (Wikipedia)](https://en.wikipedia.org/wiki/Explicit_formulae_for_L-functions)
- [A smooth version of Landau's explicit formula (2311.04347)](https://arxiv.org/abs/2311.04347)
- [Connes — Sur les Formules Explicites I (math/0101068)](https://arxiv.org/abs/math/0101068)

Hilbert–Pólya, spectral / representation candidates:
- [On the Existence of the Hilbert–Pólya Hamiltonian (2408.15135)](https://arxiv.org/abs/2408.15135)
- [Majorana relativistic quantum approach in Rindler spacetime (2503.09644)](https://arxiv.org/abs/2503.09644)
- [Riemann zero condition off the critical line: matrix formulation (2411.17701)](https://arxiv.org/abs/2411.17701)
- [Supersymmetry and Riemann zeros on the critical line (1810.02204)](https://arxiv.org/abs/1810.02204)
- [Riemann zeros as spectrum (1601.01797)](https://arxiv.org/abs/1601.01797)

Choreography databases (Probe 04 substrate context):
- [Database of trivial choreographies, planar 3-body (2210.00594)](https://arxiv.org/abs/2210.00594)
- [10,059 new 3D periodic orbits of general three-body problem, with 21 3D choreographies for equal masses (2508.08568)](https://arxiv.org/abs/2508.08568)
- [Three body problem (Scholarpedia)](http://www.scholarpedia.org/article/Three_body_problem)
