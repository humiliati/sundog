# P-vs-NP Lit-Pass Memo

> Prior-art record for the Sundog vs. P-vs-NP verification roadmap. This
> memo records what the indexed literature supports as of 2026-05-28 and what
> it does not, so the roadmap can use complexity vocabulary without drifting
> into a complexity-theoretic claim.

**Date:** 2026-05-28
**Status:** Filed as the prior-art and citation spine for
[`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md). Treat all gap claims here as
time-stamped: "not found in this lit pass," not "does not exist."

## Method

Targeted searches covered six fronts:

1. classical P, NP, NP-completeness, reductions, and proof barriers;
2. promise problems, parameterized complexity, and search-vs-decision;
3. proof systems, PCPs, interactive proofs, proof-carrying code, and
   certifying algorithms;
4. one-way functions, hard-core predicates, and cryptographic hardness;
5. verified AI, neural-network verification, certified robustness, and
   shielding;
6. alignment-specific failure modes: reward hacking, Goodhart, mesa-
   optimization, goal misgeneralization, ELK, scalable oversight, and
   mechanistic interpretability.

Preference was given to primary papers, author-hosted PDFs, official project
pages, arXiv records, proceedings pages, and publisher pages. Claimed
"solutions" to P vs. NP found in broad web search were excluded unless they
were part of a recognized complexity-theory source trail.

## Track A - Classical P/NP and Proof Barriers

The safe foundation is the Cook-Levin/Karp spine. Cook's theorem-proving
paper and Levin's independent universal-search framing established the modern
P-vs-NP question: nondeterministic polynomial-time acceptance is equivalent to
polynomial-time verification by a bounded certificate. Karp then made the
reduction discipline concrete by showing many natural combinatorial problems
equivalent under polynomial reductions.

Current posture remains unchanged: P vs. NP is open. Fortnow's CACM survey
and Cook's Clay problem description are the right public sanity checks. The
roadmap should not gesture toward resolving, reframing, or weakening the
Millennium problem. It should use the "finding versus checking" distinction
as vocabulary for a separate bounded alignment-verification program.

Barrier literature matters because it prevents accidental overreach. Baker,
Gill, and Solovay show relativization limits; Razborov and Rudich show the
natural-proofs obstruction under pseudorandomness assumptions; Aaronson and
Wigderson add algebrization. These are not directly used by Sundog, but they
are guardrails: any public line that sounds like "a new route to P != NP" is
too hot.

**Gap:** no indexed result found that maps low-dimensional alignment
signatures into a new P/NP complexity-class statement. The legitimate Sundog
move is not class separation. It is a promise-bounded verifier study.

**Disposition for roadmap:** keep the hook, but explicitly say the work is a
bounded verifier analogy and empirical/formal toy program, not a complexity
theory result.

## Track B - Promise, Parameter, and Capacity Envelopes

The strongest formal match for "inside a named operating envelope" is the
promise-problem tradition. Even, Selman, and Yacobi define promise problems
as partial decision problems, motivated in part by public-key cryptography.
This maps cleanly to Sundog: the verifier only claims correctness on inputs
that satisfy a registered envelope condition.

Parameterized complexity supplies the second scaffold. Instead of asking
whether verification is polynomial in full generality, the roadmap can ask
whether a verifier is tractable for fixed or bounded parameters such as
sensor tier, policy capacity, adversary class, horizon length, latent-field
dimension, noise, or envelope margin. Downey-Fellows and Flum-Grohe provide
the normal vocabulary here: a problem can be hard in general but tractable
for bounded structure.

Search-vs-decision and TFNP literature add useful caution. Search, decision,
and witness production are not automatically the same problem outside special
self-reducible cases. Megiddo-Papadimitriou's total-search framing is useful
background for "a witness exists but finding it is not known to be easy."

**Gap:** "capacity-relative one-wayness" is not a standard complexity class.
No indexed source was found using that phrase as a formal object. The phrase
is admissible only as Sundog terminology for an empirical or toy-formal
capacity ladder: easy to verify at capacity `C`, hard to invert/spoof below
`C`, measured failure above `C`.

**Disposition for roadmap:** define the operating envelope as a promise
class, and define the capacity ladder as an explicit parameterized family.
Avoid "polynomial" claims unless the toy model actually proves one.

## Track C - Certificates, Proof Systems, and Cheap Checkers

NP certificates are the root metaphor, but the richer citation spine is proof
systems and certifying computation. PCP theory shows that verifier design can
change what "checkable" means: a proof can be checked probabilistically with
limited queries under strong formal conditions. Interactive proofs show an
even sharper lesson: adding interaction and randomness changes verifier
power; Shamir's IP=PSPACE is the canonical result.

Engineering literature gives a more practical cousin. Proof-carrying code
and certifying algorithms put the checker in the foreground: the producer
ships an artifact plus a checkable witness, and the consumer runs a small
checker against a declared safety policy or output contract. This is close
to the Sundog question, except Sundog's certificate is not expected to be a
formal proof string.

AI safety via debate is a useful alignment-adjacent bridge because it
explicitly uses a complexity analogy: a limited judge may evaluate a
structured dispute that would be too hard to solve directly. It is not the
same model as a Sundog signature verifier, but it supports the broader
claim that "verifier design" is a live alignment problem.

**Gap:** no indexed work was found that treats a local geometric signature as
a proof-carrying certificate for an alignment predicate. The nearest existing
objects are formal proof certificates, neural-network safety certificates,
and debate/oversight structures.

**Disposition for roadmap:** require a certificate schema:
`source observations`, `signature transform`, `checker`, `cost accounting`,
`false-accept rule`, `quarantine rule`, and `privilege-leak audit`.

## Track D - One-Wayness, Inversion, and Spoofing

Diffie-Hellman introduced the practical cryptographic shape: easy operations
for honest parties and computationally infeasible recovery for adversaries.
Goldreich-Levin then shows that hard-core predicates can be extracted from
one-way functions. These are actual cryptographic claims and should not be
blurred into the Sundog setting.

For Sundog, the useful transfer is the adversarial game structure:

- verify safety from a signature;
- reconstruct the hidden target from the signature;
- spoof the signature while violating safety.

This is weaker than cryptographic one-wayness and must be measured against a
named adversary family. The natural-proofs barrier is a reminder that
hardness assumptions and pseudorandomness are delicate; a casual "one-way"
phrase can sound like a cryptographic theorem when none exists.

**Gap:** no indexed source was found formalizing low-dimensional alignment
signatures as cryptographic one-way functions. That would be too strong for
the current Sundog evidence tier.

**Disposition for roadmap:** use "capacity-relative one-wayness" only with a
measurement contract: capacity tier, adversary class, inversion/spoof task,
success metric, and failure threshold.

## Track E - Alignment Verification Pressure

The alignment literature supplies the problem pressure. Concrete Problems in
AI Safety separates reward hacking, scalable supervision, safe exploration,
and distribution shift. Goodhart variants formalize proxy collapse under
optimization pressure. Learned optimization and mesa-optimization explain why
a trained policy may develop internal objectives not equivalent to the outer
training target. Goal misgeneralization shows that a system can pursue a
wrong goal even when training behavior looked correct.

ELK is especially relevant: the central difficulty is eliciting what a model
knows about the world rather than what it reports under pressure. Sundog's
signature verifier should be framed as adjacent to this problem, not a
solution to it. A low-dimensional signature can be a verifier aid only when
the safety predicate is actually carried by the signature.

Mechanistic interpretability adds both support and warning. Circuit-style
analysis and model-editing work show that localized mechanisms can sometimes
be found and causally tested. Superposition warns that low-dimensional
features may hide or entangle many concepts, so a small signature is not
automatically simple or safe.

**Gap:** no indexed alignment paper found in this pass establishes a general
low-dimensional certificate for policy safety. Existing work motivates the
failure modes: proxy capture, false reporting, reward hacking, inner
objectives, and interpretability compression loss.

**Disposition for roadmap:** the falsification battery should lead with false
accepts, proxy capture, decoy signatures, and sensor-tier degradation. Utility
or reward should be secondary.

## Track F - Verified AI, Neural Verification, and Safety Monitors

Verified AI and neural-network verification provide the baseline world that
Sundog must respect. Reluplex and later DNN verification work attempt formal
property checks for neural networks. Certified robustness via randomized
smoothing produces a probabilistic certificate for a bounded perturbation
class. Shielding in safe reinforcement learning synthesizes a monitor that
intercepts unsafe actions with respect to a formal specification.

These approaches often require the network, formal specification, or system
model to be available in ways Sundog's indirect-signature program may not
assume. That is not a weakness of the existing literature; it is the precise
contrast. Sundog's possible contribution is a verifier with less direct state
access, not a replacement for formal verification.

**Gap:** no indexed work found here that replaces model/system access with a
Sundog-style local, gauge-invariant signature while preserving a named safety
envelope. Neural verification and shields are therefore baselines and
competitors, not merely related work.

**Disposition for roadmap:** every toy verifier should compare against at
least one full-state or rollout verifier and one formal/symbolic baseline
where feasible.

## Updated Roadmap Scaffold

The lit pass suggests the following scaffold for
[`SUNDOG_V_P_V_NP.md`](SUNDOG_V_P_V_NP.md):

1. **Boundary first.** This is not P-vs-NP work. It is a bounded verifier
   study that borrows the finding/checking distinction.
2. **Promise object.** Define the verifier domain as a promise problem:
   inputs outside the operating envelope must reject or quarantine.
3. **Parameterized capacity.** Treat capacity, noise, sensor tier, adversary
   class, and horizon as parameters, not footnotes.
4. **Certificate schema.** A Sundog certificate must specify source probes,
   signature transform, invariance checks, checker cost, failure modes, and
   privilege-leak audit.
5. **Three adversary tasks.** Verification, inversion, and spoofing are
   separate tasks with separate metrics.
6. **Baselines.** Compare signature verification to rollout, full-state,
   formal/symbolic, and neural-verification baselines where feasible.
7. **False accepts first.** False accept rate is the primary safety metric.
   Utility preservation comes later.
8. **Public-copy restraint.** Say "measured verifier advantage inside a
   registered envelope," not "polynomial certificate" or "proof of safety."

## Probe Ranking

| Rank | Probe | Lit-pass disposition | Cost | Why this rank |
| --- | --- | --- | --- | --- |
| 1 | Formal toy promise verifier | **Admitted** | Low | Promise problems and certifying algorithms give the right scaffold. The toy must define `Safe`, `H`, `sigma`, `V`, cost, and quarantine behavior before implementation. |
| 2 | Capacity-relative one-wayness battery | **Admitted with terminology guard** | Med | Cryptography motivates verify/invert/spoof structure, but the Sundog term is empirical. Must report threshold where one-wayness fails. |
| 3 | Baseline verifier comparison | **Admitted** | Med | Verified AI, Reluplex-style checks, shielding, and randomized smoothing are the competitor set. Signature verification only matters if it beats or complements these under restricted access. |
| 4 | Mesa verifier bridge | **Conditionally admitted** | Med | Alignment literature makes mesa/Goodhart pressure central. Admission requires causal interventions that distinguish behavior imitation from certificate sufficiency. |
| 5 | ARC discrete-abstraction port | **Conditionally admitted** | Med | Blackwell sufficiency and information-bottleneck ideas support the question, but ARC decoder failures mean this must be framed as a pocket search or quarantine. |
| 6 | Public essay/paper | **Deferred** | Low-Med | Publish only after a toy verifier and capacity battery produce receipts. The provocation is useful only after the boundary is impossible to miss. |

## Disposition

The roadmap can be promoted from conceptual lineage to active research only
after two things exist:

- a formal promise-verifier toy problem with a checkable certificate schema;
- a capacity battery that separately measures verification, inversion, and
  spoofing.

Without those, "P-vs-NP" remains a hook. With them, the project has a real
research object: bounded alignment verification from indirect signatures.

## Sources

Classical complexity and barriers:
- Stephen Cook, "The Complexity of Theorem-Proving Procedures" (1971):
  <https://www.cs.cmu.edu/~15455/resources/Cook1971-complx-thm-proof.pdf>
- Stephen Cook, Clay Mathematics Institute official P vs. NP description:
  <https://www.claymath.org/wp-content/uploads/2022/06/pvsnp.pdf>
- Richard Karp, "Reducibility among Combinatorial Problems" (1972):
  <https://link.springer.com/chapter/10.1007/978-1-4684-2001-2_9>
- Sanjeev Arora and Boaz Barak, *Computational Complexity: A Modern
  Approach*, draft:
  <https://theory.cs.princeton.edu/complexity/book.pdf>
- Lance Fortnow, "The Status of the P versus NP Problem" (CACM 2009):
  <https://lance.fortnow.com/papers/files/pnp-cacm.pdf>
- Theodore Baker, John Gill, Robert Solovay, "Relativizations of the
  P = ? NP Question" (1975): <https://doi.org/10.1137/0204037>
- Alexander Razborov and Steven Rudich, "Natural Proofs" (1997):
  <https://www.cs.toronto.edu/tss/files/papers/1-s2.0-S002200009791494X-main.pdf>
- Scott Aaronson and Avi Wigderson, "Algebrization: A New Barrier in
  Complexity Theory" (2008/2009):
  <https://www.scottaaronson.com/papers/alg.pdf>

Promise, search, and parameterization:
- Shimon Even, Alan L. Selman, Yacov Yacobi, "The complexity of promise
  problems with applications to public-key cryptography" (1984):
  <https://www.sciencedirect.com/science/article/pii/S001999588480056X>
- Nimrod Megiddo and Christos H. Papadimitriou, "On Total Functions,
  Existence Theorems and Computational Complexity" (1991):
  <https://theory.stanford.edu/~megiddo/pdf/papadimX.pdf>
- Rod Downey and Michael Fellows, "Fixed-Parameter Tractability and
  Completeness I: Basic Results" (1995):
  <https://doi.org/10.1137/S0097539792228228>
- Jorg Flum and Martin Grohe, *Parameterized Complexity Theory*:
  <https://link.springer.com/book/10.1007/3-540-29953-X>

Proof systems, certificates, and checkers:
- Sanjeev Arora, Carsten Lund, Rajeev Motwani, Madhu Sudan, Mario Szegedy,
  "Proof Verification and the Hardness of Approximation Problems":
  <https://people.seas.harvard.edu/~madhusudan/papers/1992/almss-conf.pdf>
- Adi Shamir, "IP = PSPACE" (1992):
  <https://weizmann.esploro.exlibrisgroup.com/esploro/outputs/journalArticle/IP--PSPACE/993265992703596>
- Shafi Goldwasser, Silvio Micali, Charles Rackoff, "The Knowledge
  Complexity of Interactive Proof Systems" (1989):
  <https://epubs.siam.org/doi/10.1137/0218012>
- George Necula, "Proof-Carrying Code" (1997):
  <https://dblp.org/rec/conf/popl/Necula97>
- R. M. McConnell, Kurt Mehlhorn, Stefan Naeher, Pascal Schweitzer,
  "Certifying Algorithms" (2011):
  <https://openresearch-repository.anu.edu.au/items/3a94e5d9-4237-41d2-95d0-7492c3af3781>

One-wayness and cryptographic hardness:
- Whitfield Diffie and Martin Hellman, "New Directions in Cryptography"
  (1976): <https://www-ee.stanford.edu/~hellman/publications/24.pdf>
- Oded Goldreich and Leonid Levin, "A Hard-Core Predicate for all One-Way
  Functions" (1989): <https://www.cs.bu.edu/fac/lnd/pdf/hard.pdf>
- Russell Impagliazzo and Michael Luby, "One-way functions are essential for
  complexity based cryptography" (1989):
  <https://doi.org/10.1109/SFCS.1989.63483>

Sufficiency, compression, and interpretability:
- David Blackwell, "Equivalent Comparisons of Experiments" (1953):
  <https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-24/issue-2/Equivalent-Comparisons-of-Experiments/10.1214/aoms/1177729032.pdf>
- Naftali Tishby, Fernando Pereira, William Bialek, "The Information
  Bottleneck Method" (1999/2000): <https://arxiv.org/abs/physics/0004057>
- Chris Olah et al., "Zoom In: An Introduction to Circuits" (2020):
  <https://distill.pub/2020/circuits/zoom-in/>
- Nelson Elhage et al., "Toy Models of Superposition" (2022):
  <https://arxiv.org/abs/2209.10652>
- Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov, "Locating and
  Editing Factual Associations in GPT" (2022):
  <https://arxiv.org/abs/2202.05262>

Alignment and safety pressure:
- Dario Amodei et al., "Concrete Problems in AI Safety" (2016):
  <https://arxiv.org/abs/1606.06565>
- David Manheim and Scott Garrabrant, "Categorizing Variants of Goodhart's
  Law" (2018/2019): <https://arxiv.org/abs/1803.04585>
- Evan Hubinger et al., "Risks from Learned Optimization in Advanced Machine
  Learning Systems" (2019): <https://arxiv.org/abs/1906.01820>
- Rohin Shah et al., "Goal Misgeneralization: Why Correct Specifications
  Aren't Enough For Correct Goals" (2022):
  <https://arxiv.org/abs/2210.01790>
- Paul Christiano and Mark Xu, ARC technical report announcement,
  "Eliciting Latent Knowledge" (2021):
  <https://www.alignment.org/blog/arcs-first-technical-report-eliciting-latent-knowledge/>
- Geoffrey Irving, Paul Christiano, Dario Amodei, "AI Safety via Debate"
  (2018): <https://arxiv.org/abs/1805.00899>
- Jan Leike et al., "Scalable agent alignment via reward modeling" (2018):
  <https://arxiv.org/abs/1811.07871>

Verified AI and certified safety:
- Sanjit Seshia, Dorsa Sadigh, Shankar Sastry, "Towards Verified Artificial
  Intelligence" (2016/2020): <https://arxiv.org/abs/1606.08514>
- Guy Katz et al., "Reluplex: An Efficient SMT Solver for Verifying Deep
  Neural Networks" (2017): <https://arxiv.org/abs/1702.01135>
- Xiaowei Huang, Marta Kwiatkowska, Sen Wang, Min Wu, "Safety Verification
  of Deep Neural Networks" (2017): <https://arxiv.org/abs/1610.06940>
- Mohammed Alshiekh et al., "Safe Reinforcement Learning via Shielding"
  (2018): <https://ojs.aaai.org/index.php/AAAI/article/view/11797>
- Jeremy Cohen, Elan Rosenfeld, Zico Kolter, "Certified Adversarial
  Robustness via Randomized Smoothing" (2019):
  <https://proceedings.mlr.press/v97/cohen19c.html>
