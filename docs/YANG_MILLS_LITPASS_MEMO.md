# Yang-Mills Lit-Pass Memo

> Prior-art record for the Sundog vs. Yang-Mills finite-lattice certificate
> roadmap. This memo records what the indexed literature supports as of
> 2026-05-29 and what it does not, so the roadmap can use gauge-theory
> language without drifting into a Clay-problem claim.

**Date:** 2026-05-29
**Status:** Filed as the prior-art and citation spine for
[`SUNDOG_V_YANG_MILLS.md`](SUNDOG_V_YANG_MILLS.md). Treat all gap claims here
as time-stamped: "not found in this lit pass," not "does not exist."

## Method

Targeted searches covered six fronts:

1. the official Clay/Jaffe-Witten Yang-Mills existence and mass-gap statement;
2. foundational lattice gauge theory, Wilson loops, plaquettes, area law, and
   finite-lattice Monte Carlo practice;
3. recent 2025-2026 rigorous lattice Yang-Mills work on area law, loop
   equations, large-N / t'Hooft regimes, and 2D/3D constructive limits;
4. lattice positivity/bootstrap and Wilson-loop observable programs;
5. gauge-equivariant / gauge-invariant machine-learning work for lattice
   gauge theory;
6. recent public proof-claim attempts and claimed Clay-solution artifacts.

Preference was given to primary papers, arXiv records, publisher pages, DOI
records, the official Clay source, and author/institution-hosted pages. Broad
web search surfaced many 2025-2026 claimed solutions. Those are recorded only
as a quarantine/watchlist unless they are in a recognized source trail. They
are not evidence for Sundog claims.

## Track A - Clay Boundary And Constructive Status

The official target is the Jaffe-Witten Clay problem: construct nontrivial
quantum Yang-Mills theory on four-dimensional Euclidean space for every
compact simple gauge group, satisfying standard quantum-field-theory axioms,
and prove a positive mass gap. The Sundog roadmap must not dilute that target
into a finite-lattice experiment. A finite lattice can be a receipt substrate;
it is not the continuum theorem.

This lit pass found no Clay-recognized solution. It did find several recent
claim artifacts:

- D. C. Jacobsen's 2025 arXiv submission claimed a constructive SU(3) proof,
  but the arXiv record marks the submission withdrawn for research-content
  quality reasons.
- James Glimm's 2025 "The Pure Yang-Mills Field" page claims the Millennium
  conditions are fulfilled, while the abstract also states that the construction
  depends on an assumed maximum-entropy-production principle. That makes it a
  watchlist item, not a roadmap foundation.
- Oliver Odusanya's 2026 `yangmills.dev` series claims a constructive proof via
  Balaban-style RG and says verification is in progress. The source is useful
  only as a proof-claim watchlist item until external adjudication exists.
- SSRN, Zenodo, ResearchGate, Preprints.org, and similar sites contain many
  recent "solved" or "framework" claims. This memo does not use them as
  technical support unless a future specialist review promotes a specific item.

**Disposition for roadmap:** the Clay problem remains the boundary object, not
the claim object. Public language must say "finite-lattice gauge-invariant
certificate test," never "Yang-Mills progress," "mass gap result," or
"confinement proof."

## Track B - The Invariant Shadow Is Native Here

The roadmap hook is not imported metaphor. Lattice gauge theory already makes
the tension explicit:

- link variables carry gauge redundancy;
- closed loops and plaquettes are gauge-invariant observables;
- Wilson-loop area-law behavior is a central confinement diagnostic;
- loop equations and trace relations organize gauge-invariant observables.

Wilson's lattice formulation, Osterwalder-Seiler's lattice gauge-theory
framework, Creutz-style SU(2) Monte Carlo, and Kogut's review are the boring
spine. Giles's reconstruction theorem is the cautionary complement: a complete
Wilson-loop family can reconstruct gauge potentials up to gauge under strong
constraints, but Sundog is not proposing a complete loop family. A small
signature is intentionally lossy.

That is the core honesty constraint. If Sundog keeps only a finite vocabulary
of plaquette, Wilson-loop, correlator, or blocking summaries, the only allowed
claim is that this lossy invariant shadow preserves a registered finite-lattice
label better than controls. It cannot claim to encode "the field" or "the
theory."

**Gap:** no indexed source found in this pass asks Sundog's exact question:
given a small, frozen, gauge-invariant signature, does rank-neighborhood
structure preserve a pre-registered finite-lattice observable/regime label
beyond metadata, gauge-variant, random, and target-leak controls?

**Disposition for roadmap:** use the lattice observable vocabulary, but make
the lossiness explicit. "Do not reconstruct the gauge field" is a discipline,
not a novelty claim.

## Track C - Recent Rigorous Lattice / Probability Movement

The most relevant recent activity is not a Clay solution. It is a cluster of
finite-lattice, loop-observable, and lower-dimensional constructive advances:

- Cao, Nissim, and Sheffield extended regimes where area law is proven for pure
  U(N) lattice Yang-Mills, improving the Osterwalder-Seiler lineage.
- The same authors used a dynamical approach to prove Wilson area law in a
  t'Hooft parameter regime for groups with nontrivial center.
- Nissim established mass gap, unique infinite-volume limit, and large-N
  behavior for U(N) lattice Yang-Mills in a t'Hooft regime, explicitly as a
  lattice result.
- Shen, Zhu, and Zhu developed a stochastic-analysis approach to lattice
  Yang-Mills at strong coupling, with Langevin dynamics, functional
  inequalities, large-N limits, and mass gap in the registered regime.
- Lemoine's 2026 work identifies finite-N structures for Wilson-loop
  expectations through state-sum, gauge/string, spin-foam/channel, and master
  loop-equation views.
- Liu and Yang's 2026 loop-equation work systematizes direct and indirect loop
  equations in finite-N lattice Yang-Mills.
- Dang and Nohra's 2026 two-dimensional result treats the Yang-Mills measure on
  compact surfaces as a universal scaling limit of lattice models.
- Chevyrev and Shen's 2026 3D stochastic Yang-Mills-Higgs paper proves
  uniqueness of gauge-covariant renormalisation and underscores that lattice
  dynamics and Wilson loops remain active tools even below the Clay dimension.

These papers collectively say: Wilson loops, area-law proxies, loop equations,
and lower-dimensional/stochastic gauges are current, serious terrain. They do
not say that a small local signature is enough. They also do not invite a
finite-lattice-to-continuum leap.

**Disposition for roadmap:** the first Sundog domain should be intentionally
boring: small finite Wilson-action ensembles with frozen loop sets and a
receipt-level observable label. 2D and 3D are acceptable harness domains; 4D is
a later diagnostic only after the smoke and leakage controls are clean.

## Track D - ML / Bootstrap Competitors Are Close

This lane has real neighboring work. The closest competitors and baselines are:

- lattice gauge equivariant convolutional neural networks (L-CNNs), which build
  gauge equivariance into convolutional layers and form Wilson-loop-like
  features;
- finite-N lattice Yang-Mills bootstrap programs using loop equations,
  positivity, and Wilson-loop truncations;
- 2026 gauge-equivariant diffusion models for non-Abelian lattice gauge
  theory, trained on traditional Monte Carlo ensembles and checked against
  Wilson loops and topological susceptibility;
- 2026 neural-network Wilson-loop work that trains gauge-equivariant layers to
  improve static quark-antiquark interpolators while maintaining gauge
  invariance;
- 2026 gauge-equivariant graph neural networks and Abelian Wilson-loop GNNs,
  which explicitly use local gauge structure or Wilson-loop representations.

This is good news and bad news. Good: the field agrees that respecting gauge
symmetry matters. Bad: Sundog cannot claim novelty merely for using
gauge-invariant or gauge-equivariant features. The possible niche is narrower:
receipt discipline, compact invariant signatures, rank-local certificate
metrics, and hard leakage controls.

**Gap:** no indexed work found in this pass packages the question in the
Sundog receipt style: pre-register a lossy invariant signature, refuse
reconstruction, test rank-local preservation against metadata/raw/random/
permutation/gauge-randomized controls, and publish named nulls as first-class
outcomes.

**Disposition for roadmap:** L-CNNs, bootstrap margins, Wilson-loop neural
interpolators, and equivariant diffusion models are baseline/competitor
language. They are not to be caricatured as "raw ML" foils.

## Track E - Sundog Intuition After The Lit Pass

The useful intuition is:

> Gauge theory is a permission slip to stop worshipping coordinates.

But the next sentence has to be severe:

> A gauge-invariant shadow can be too small, too nonlocal, or too entangled
> with the target to certify anything.

So the Sundog move is not "solve Yang-Mills by compression." It is:

1. choose a finite lattice;
2. freeze an ensemble source and loop/signature vocabulary;
3. compute labels from observables held out from the signature;
4. check whether rank-neighborhoods in signature space preserve those labels
   beyond controls;
5. accept a named null if metadata, raw-link leakage, target leakage, or
   finite-size artifacts explain the signal.

The field itself is the wrong trophy. The trophy, if one exists, is a clean
receipt showing that a small quotient-readable shadow preserved some bounded
finite-lattice structure without smuggling the answer in.

## Updated Roadmap Scaffold

The lit pass suggests the following scaffold for
[`SUNDOG_V_YANG_MILLS.md`](SUNDOG_V_YANG_MILLS.md):

1. **Boundary first.** This is not a Clay lane. It is a finite-lattice
   certificate lane using Clay Yang-Mills as a stress-test direction.
2. **Domain before code.** Freeze theory, group, dimension, lattice size,
   action, boundary condition, coupling slate, ensemble source, thermalization
   rule, and output budget before writing a runner.
3. **Invariant-only primary.** Primary signatures must be gauge invariant by
   construction. Gauge-fixed or raw-link encodings are controls/diagnostics.
4. **Held-out observable labels.** Do not score a Wilson-loop label using the
   same loop embedded in the signature. Prefer larger loops, correlator
   classes, or blocked observables held out from the signature vocabulary.
5. **Leakage-first controls.** Metadata-only, raw/gauge-fixed, random,
   coupling-stratified random, gauge-randomized copies, and label permutations
   are not optional.
6. **Competitor awareness.** Bootstrap, L-CNN, equivariant diffusion, and
   Wilson-loop-neural methods are the live comparison language.
7. **2D/3D before 4D.** 2D/U(1) or 2D/SU(2) cells are instrumentation and
   leakage checks. 3D/SU(2) is the first plausible nontrivial smoke. 4D is a
   later diagnostic and still not a Clay claim.
8. **Rank-local first.** Absolute fibers are likely sparse. The ARC lesson says
   register rank-locality before fixed-radius sufficiency.
9. **Nulls are durable.** A clean "invariant shadow too coarse" result is a
   real outcome and should be filed as such.

## Probe Ranking

| Rank | Probe | Lit-pass disposition | Cost | Why this rank |
| --- | --- | --- | --- | --- |
| 1 | Phase 0 domain and receipt lock | **Admitted** | Low | The field is too mature and too overclaimed to proceed without a frozen finite-lattice envelope and public boundary. |
| 2 | Gauge-invariance smoke on deterministic fixtures | **Admitted** | Low | It tests the exact thing Sundog might accidentally violate: invariance under random gauge transformations. |
| 3 | Abelian or 2D instrumentation cell | **Admitted only as harness smoke** | Low | Useful for leakage controls and loop bookkeeping; never evidence for non-Abelian Yang-Mills. |
| 4 | Small SU(2) relative-locality certificate | **Conditionally admitted** | Med | The first real Sundog question: compact invariant signature vs controls on held-out finite-lattice labels. |
| 5 | Bootstrap / L-CNN comparison cell | **Deferred until a primary signal exists** | Med | Too costly and too easy to misframe until the simple certificate either passes or fails. |
| 6 | 4D finite-lattice diagnostic | **Deferred** | Med-High | May be useful later, but it invites overreach and does not by itself approach the Clay theorem. |
| 7 | Public essay / gallery promotion | **Blocked** | Low | Requires Phase 0 filing, at least one receipt, and external lattice-gauge sanity check. |

## Disposition

The roadmap can move from handoff draft to Phase 0 drafting after this memo,
but not to execution. The next artifact should be
`docs/prereg/yang-mills/P0_DOMAIN_AND_RECEIPT_LOCK.md`, and it should freeze
at least:

- gauge group and dimension;
- lattice sizes and boundary condition;
- action and coupling slate;
- ensemble source or generator algorithm;
- burn-in/thinning/autocorrelation handling if generated locally;
- primary signature vocabulary;
- held-out observable labels;
- all leakage controls;
- compute cap and staged commands under the repo's ten-minute rule;
- exact positive/null/metadata-only branch table.

Recommended first intuition:

> Start too small to impress anyone. If the invariant shadow cannot pass a
> tiny, boring, pre-registered finite-lattice test, it has no business near the
> profound version of the problem.

## Sources

Official problem boundary:

- Clay Mathematics Institute, Yang-Mills and the Mass Gap:
  <https://www.claymath.org/millennium/yang-mills-the-maths-gap/>
- Arthur Jaffe and Edward Witten, "Quantum Yang-Mills Theory" official problem
  description:
  <https://www.claymath.org/wp-content/uploads/2022/06/yangmills.pdf>

Foundational lattice / invariant-observable spine:

- Kenneth G. Wilson, "Confinement of quarks" (1974):
  <https://doi.org/10.1103/PhysRevD.10.2445>
- Konrad Osterwalder and Erhard Seiler, "Gauge Field Theories on a Lattice"
  (1978): <https://doi.org/10.1016/0003-4916(78)90039-8>
- John B. Kogut, "An Introduction to Lattice Gauge Theory and Spin Systems"
  (1979): <https://doi.org/10.1103/RevModPhys.51.659>
- Michael Creutz, "Monte Carlo Study of Quantized SU(2) Gauge Theory" (1980):
  <https://doi.org/10.1103/PhysRevD.21.2308>
- Robert Giles, "Reconstruction of gauge potentials from Wilson loops" (1981):
  <https://doi.org/10.1103/PhysRevD.24.2160>
- A. D. Kennedy and B. J. Pendleton, "Improved heatbath method for Monte Carlo
  calculations in lattice gauge theories" (1985):
  <https://www.osti.gov/etdeweb/biblio/6427299>
- F. R. Brown and T. J. Woch, "Overrelaxed heat-bath and Metropolis algorithms
  for accelerating pure gauge Monte Carlo calculations" (1987):
  <https://www.osti.gov/biblio/6510289>

Recent rigorous lattice/probability and loop-equation work:

- Hao Shen, Rongchan Zhu, Xiangchan Zhu, "A stochastic analysis approach to
  lattice Yang-Mills at strong coupling" (2022):
  <https://arxiv.org/abs/2204.12737>
- Sky Cao, Ron Nissim, Scott Sheffield, "Expanded regimes of area law for
  lattice Yang-Mills theories" (2025):
  <https://arxiv.org/abs/2505.16585>
- Sky Cao, Ron Nissim, Scott Sheffield, "Dynamical approach to area law for
  lattice Yang-Mills" (2025): <https://arxiv.org/abs/2509.04688>
- Ron Nissim, "U(N) lattice Yang-Mills in the t'Hooft regime" (2025):
  <https://arxiv.org/abs/2510.22788>
- Xizhe Liu and Gang Yang, "Direct and Indirect Loop Equations in Lattice
  Yang-Mills Theory" (2026): <https://arxiv.org/abs/2601.04316>
- Nguyen Viet Dang and Elias Nohra, "The Yang-Mills measure on compact
  surfaces as a universal scaling limit of lattice gauge models" (2026):
  <https://arxiv.org/abs/2602.08591>
- Ilya Chevyrev and Hao Shen, "Uniqueness of gauge covariant renormalisation
  of stochastic 3D Yang-Mills-Higgs" (2026):
  <https://arxiv.org/abs/2503.03060>
- Thibaut Lemoine, "Universal dualities for Wilson loops in lattice
  Yang-Mills" (2026): <https://arxiv.org/abs/2604.16252>

Bootstrap / ML-adjacent competitors:

- Matteo Favoni, Andreas Ipp, David I. Muller, Daniel Schuh, "Lattice gauge
  equivariant convolutional neural networks" (2020/2022):
  <https://arxiv.org/abs/2012.12901>
- Vladimir Kazakov and Zechuan Zheng, "Bootstrap for Finite N Lattice
  Yang-Mills Theory" (2024): <https://arxiv.org/abs/2404.16925>
- Gert Aarts et al., "Generalizable Equivariant Diffusion Models for
  Non-Abelian Lattice Gauge Theory" (2026):
  <https://arxiv.org/abs/2601.19552>
- Verena Bellscheidt, Nora Brambilla, Andreas S. Kronfeld, Julian
  Mayer-Steudte, "Wilson loops with neural networks" (2026):
  <https://arxiv.org/abs/2602.02436>
- Ali Rayat, Yaohang Li, Gia-Wei Chern, "Gauge-Equivariant Graph Neural
  Networks for Lattice Gauge Theories" (2026):
  <https://arxiv.org/abs/2604.20797>
- Ali Rayat and Gia-Wei Chern, "Graph Neural Networks in the Wilson Loop
  Representation of Abelian Lattice Gauge Theories" (2026):
  <https://arxiv.org/abs/2605.03901>

Claimed-proof watchlist / quarantine:

- D. C. Jacobsen, withdrawn arXiv claim, "A Constructive Proof of Existence
  and Mass Gap for Pure SU(3) Yang-Mills in Four-Dimensional Space-Time"
  (2025): <https://arxiv.org/abs/2506.00284>
- James Glimm, "The Pure Yang-Mills Field" (2025):
  <https://commons.library.stonybrook.edu/ams-articles/4/>
- Oliver Odusanya, "Yang-Mills Mass Gap - Constructive Proof" (2026):
  <https://yangmills.dev/>
