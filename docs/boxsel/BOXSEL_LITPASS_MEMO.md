# Sundog BoxSEL - Lit-Pass Memo

> Prior-art and claim-boundary record for the Sundog BoxSEL false-closure lane.
> This memo fills Phase 0.5 of [`../SUNDOG_V_BOXSEL.md`](../SUNDOG_V_BOXSEL.md)
> and gates any oracle, sampler, extremal optimizer, public page, or outreach
> claim.

**Date:** 2026-06-20  
**Status:** Starter lit-pass filled. Treat all gap claims here as time-stamped:
"not found / not locked in this pass," not "does not exist."  
**Citation check:** arXiv:2407.11821 v2 is current on arXiv as of this pass:
submitted 2024-07-16, last revised 2026-06-02, listed as accepted at UAI 2026.
The v1 and v2 arXiv sources were downloaded and checked for the PMP algorithm
artifact.  
**Surfaces:** `docs/SUNDOG_V_BOXSEL.md` and this memo only. No prereg, oracle,
sampler, optimizer, code result, public page, or external packet exists yet.

## Purpose

BoxSEL is not a Sundog invention. Statistical EL, entailment intervals, box
embeddings, zero-loss embedding soundness, and probabilistic-modus-ponens style
bounds all belong to the literature. The Sundog lane is narrower:

> Use small exact-oracle SEL fragments to separate logical concentration from
> concentration caused by search, representation, loss, or semantic/implementation
> mismatch; then test whether observable embedding traces can trigger an
> accept / widen / abstain rule without the oracle.

This pass sets the attribution boundary and corrects two places where the lane
would otherwise overclaim:

1. **Zero-loss soundness is an anchor-paper result, not a gap we may casually
   deny.** A "coherence gap" is only a watch item until the finite-counting oracle
   and the paper's geometric-volume semantics are reconciled.
2. **The PMP discrepancy is real in the paper artifacts, but not yet a result
   invalidation.** The paper body/source gives the `+ 1 - l1` upper slack, while
   Appendix Algorithm 2 prints `+ 1 - q2`. The blast radius depends on evaluation
   code or on how the authors instantiated the premises.

## Executive Verdict

**Banked:** The anchor paper directly supports the false-closure hook. It estimates
unknown probability intervals by min/max over independently trained box embeddings;
it proves point-estimate soundness at zero loss; it explicitly warns that interval
estimates can be too tight; and its toy example reports `[0.43, 0.83]` while the
exact interval is `[0.16, 0.96]`.

**Standard, not ours:** SEL semantics; ExpTime hardness; type-elimination / LP
exact reasoning; BoxEL-style boxes and affine roles; box-volume conditionals;
reject-option / selective prediction; calibration and conformal prediction for
KG embeddings.

**Sundog synthesis:** The named split between:

```text
search gap          I_box^n minus I_sample^{n,N}
representation gap  I* minus I_box^n
loss gap            estimates outside I* when L_T(w) > epsilon
```

and the attempt to learn a false-closure detector from traces rather than from
the exact oracle.

**Open / not banked:** Any claim that zero-loss BoxSEL can escape the exact
logical interval. The anchor paper's Theorem 3 points the other way under its
semantics. The lane may test semantic mismatch between finite-counting exact
models and geometric-volume interpretations, but must not present that as a
known failure.

## Method

Targeted pass covered six fronts:

1. arXiv:2407.11821 v1/v2 text and source;
2. SEL/statistical-DL origins and complexity;
3. BoxEL, Box2EL, probabilistic box embeddings, and uncertain KG box methods;
4. PMP / probabilistic-logic bounds;
5. selective prediction, calibration, and conformal KGE uncertainty;
6. Sundog claim-priority map.

Preference was given to arXiv records, ACL/ACM/proceedings pages, author pages,
and the arXiv TeX source. This is not a comprehensive survey of probabilistic
DLs or KG uncertainty; it is the claim-boundary pass needed to start Phase 1.

## Primary Anchors

| Source | URL | Supports | Boundary action |
| --- | --- | --- | --- |
| Zhu, Potyka, Xiong, Tran, Nayyeri, Kharlamov, Staab, "Approximating Probabilistic Inference in Statistical EL with Knowledge Graph Embeddings" | https://arxiv.org/abs/2407.11821 | Anchor BoxSEL paper. v2 accepted at UAI 2026. Defines box-volume approximate inference for SEL; gives zero-loss embedding and inference soundness; estimates intervals from multiple embeddings; reports the `[0.43, 0.83]` vs `[0.16, 0.96]` example; contains the PMP algorithm artifact. | Primary source for this lane. Cite prominently. Do not claim BoxSEL as ours. |
| Penaloza and Potyka, "Towards Statistical Reasoning in Description Logics over Finite Domains" | https://arxiv.org/abs/1706.03207 | Statistical description-logic semantics over finite domains; conditionals as proportions. | SEL/statistical-DL semantics are prior art. |
| Bednarczyk, "Statistical EL is ExpTime-complete" | https://arxiv.org/abs/1911.00696 | ExpTime-completeness of SEL consistency. | Exact oracle is small-fragment only; no scalability claim. |
| Lutz and Schroder, "Probabilistic Description Logics for Subjective Uncertainty" | https://www.informatik.uni-bremen.de/tdki/research/papers/2010/LutSchroe-KR10.pdf | Type-elimination/linear-inequality style exact reasoning in probabilistic DL context; cited by the anchor paper as the theoretical exact route. | Do not present type enumeration + LP as novel. Our implementation is a micro-oracle, not a new proof method. |
| Xiong, Potyka, Tran, Nayyeri, Staab, "Faithful Embeddings for EL++ Knowledge Bases" | https://arxiv.org/abs/2201.09919 | BoxEL: concepts as axis-parallel boxes, roles as affine transforms, zero-loss soundness for EL++ KBs. | Box body comes from BoxEL; Sundog only uses it as substrate. |
| Jackermeier, Chen, Horrocks, "Dual Box Embeddings for the Description Logic EL++" | https://arxiv.org/abs/2301.11118 | Later EL++ box-embedding family; represents both concepts and roles as boxes; proves soundness and evaluates deductive approximation. | Closest ontology-embedding neighbor; useful for "not just BoxSEL" context. |
| Patel, Dasgupta, Boratko, Li, Vilnis, McCallum, "Representing Joint Hierarchies with Box Embeddings" | https://openreview.net/forum?id=J246NSqR_l | Box volume as joint/conditional probability; smooth box training lineage. | Volume-ratio probability intuition is prior art. |
| Chen, Boratko, Chen, Dasgupta, Li, McCallum, "Probabilistic Box Embeddings for Uncertain Knowledge Graph Reasoning" | https://arxiv.org/abs/2104.04597 | BEUrRE: calibrated probabilistic semantics for uncertain KG box embeddings. | Closest KG-uncertainty box neighbor; not SEL entailment. |
| Frisch and Haddawy, "Anytime Deduction for Probabilistic Logic" | https://pure.york.ac.uk/portal/en/publications/anytime-deduction-for-probabilistic-logic/ | Rule-based deduction with conditional probabilities. | PMP-like bounds sit in probability-logic prior art. |
| Hunter and Potyka, "Syntactic Reasoning with Conditional Probabilities in Deductive Argumentation" | https://discovery.ucl.ac.uk/id/eprint/10170273/ | Recent conditional-probability inference rules. | Do not claim conditional-probability proof-rule novelty. |
| Wagner, "Modus Tollens Probabilized" | https://web.math.utk.edu/~cwagner/papers/modus.pdf | Gives the sharp conditional-PMP form `ab <= P(H) <= ab + 1 - b` for `P(H|E)=a`, `P(E)=b`. | Useful independent sanity check for the `+ 1 - q1` slack. |
| Chow, "On Optimum Recognition Error and Reject Tradeoff" | https://dl.acm.org/doi/10.1109/TIT.1970.1054406 | Classical reject-option framing. | Abstention/reject option is standard. |
| Geifman and El-Yaniv, "Selective Classification for Deep Neural Networks" | https://arxiv.org/abs/1705.08500 | Selective prediction/risk-coverage for DNNs. | Sundog detector must be positioned as trace-conditioned false-closure gating, not as inventing abstention. |
| Guo, Pleiss, Sun, Weinberger, "On Calibration of Modern Neural Networks" | https://arxiv.org/abs/1706.04599 | Calibration, temperature scaling, modern miscalibration context. | Calibration vocabulary is prior art; the lane does not provide a calibration guarantee. |
| Zhu et al., "Conformalized Answer Set Prediction for Knowledge Graph Embedding" | https://arxiv.org/abs/2408.08248 | KGE uncertainty via conformal answer sets with coverage guarantees. | Important closest-neighbor: statistically valid answer sets, but not SEL exact-interval false closure. |
| Zhu et al., "Predicate-Conditional Conformalized Answer Sets for Knowledge Graph Embeddings" | https://arxiv.org/abs/2505.16877 | Predicate-conditional conformal KGE uncertainty. | Stronger KG uncertainty neighbor; credit when discussing high-stakes per-query coverage. |
| Zhu et al., "Certainty in Uncertainty: Reasoning over Uncertain Knowledge Graphs with Statistical Guarantees" | https://arxiv.org/abs/2510.24754 | Prediction intervals for uncertain KGs with statistical guarantees. | Adjacent "uncertainty with guarantees" work by overlapping authors; distinguish from logical entailment intervals. |

## Track A - SEL Semantics and Exact Intervals

**READ 2026-06-20.**

SEL conditionals `(D | C)[l, u]` are statistical/proportion claims. In finite
semantics, an interpretation satisfies the conditional if `C` is empty or:

```text
|(D and C)^I| / |C^I| in [l, u]
```

The exact inference problem is to find the largest lower bound and smallest upper
bound entailed by the TBox:

```text
I* = [l*, u*]
```

where every model of the ontology satisfies `(D | C)[l*, u*]`.

Exact reasoning is standard and hard. The anchor paper says the only theoretical
exact route constructs an exponentially large LP via type elimination and that no
implementation existed to the authors' knowledge. Bednarczyk gives ExpTime
completeness for SEL consistency.

**Boundary action:** Phase 2 may implement a small, role-free exact oracle by
enumerating logical types and solving LPs. That is engineering for a testbed, not
a new reasoning method.

## Track B - BoxSEL and Box Embeddings

**READ 2026-06-20.**

The anchor paper extends BoxEL. Concepts are axis-aligned boxes; roles are affine
transformations; probabilities are read as volume ratios:

```text
q_w(D | C) = Vol(D_w intersect C_w) / Vol(C_w)
```

For arbitrary concepts, intersections and existential-role boxes are computed
recursively, yielding linear-time query evaluation in the concept sizes times
embedding dimension.

The key guarantees:

- **Embedding Soundness:** if `L_T(w) = 0`, the geometric interpretation
  satisfies the ontology.
- **Inference Soundness:** if `L_T(w) = 0` and the ontology entails
  `(D | C)[l, u]`, then the point estimate `p = q_w(D | C)` lies in `[l, u]`.

The interval estimator is not a theorem about endpoints. It samples `N` trained
embeddings and reports:

```text
I_sample^{n,N} = [min_i q_{w_i}(D | C), max_i q_{w_i}(D | C)]
```

The paper explicitly notes this can be too tight. The toy example:

```text
sampled estimate: [0.43, 0.83]
exact interval:   [0.16, 0.96]
```

**Boundary action:** The false-closure lane starts exactly here. A zero-loss
point may be sound while the sampled min/max interval is falsely narrow because
the sampled embeddings did not reach endpoint models.

## Track C - PMP Replication Gate

**READ 2026-06-20.**

The body/source statement of Proposition 2 is:

```text
If T |= (D | C)[l1, u1]
and T |= (E | C and D)[l2, u2],
then T |= (E | C)[l3, u3]

l3 = l1*l2
u3 = min(1, u1*u2 + 1 - l1)
```

For point premises `(A | Q1)[q1]` and `(Q2 | A and Q1)[q2]`, this specializes to:

```text
lower = q1*q2
upper = min(1, q1*q2 + 1 - q1)
```

That `1 - q1` slack is the probability mass in `Q1` outside `A`. It passes the
sanity check `q1 = 1 -> upper = q2`.

**Confirmed artifact mismatch:** Appendix Algorithm 2 in arXiv v2 source prints:

```text
Bound_max <- min(1, q1*q2 + 1 - q2)
```

under premises:

```text
(A | Q1)[q1] and (Q2 | A)[q2]
```

v1 source also prints `+ 1 - q2`, while its premise form is closer to Proposition
2:

```text
(A | Q1)[q1] and (Q2 | A and Q1)[q2]
```

So there are two replication-gate checks, not one:

1. **Upper-slack mismatch:** body Proposition 2 uses `+ 1 - l1`; Algorithm 2 uses
   `+ 1 - q2`.
2. **Premise-shape drift in v2:** Proposition 2 requires the second premise over
   `C and D`; v2 Algorithm 2 prints `(Q2 | A)` rather than `(Q2 | A and Q1)`.
   This is harmless only if construction guarantees `A subset Q1` or an equivalent
   deterministic relation.

**Numerical consequence:** at `q1 = 0.2`, `q2 = 0.8`, the body formula gives
`0.96`; Algorithm 2 gives `0.36`. At `q1 = 1`, Algorithm 2 returns an upper bound
of `1`, while the proposition returns `q2`.

**Blast radius not yet settled:** No official public evaluation repository surfaced
in this pass. The next task is not to accuse the paper; it is to implement the PMP
calculator, reproduce the toy cases, and determine whether any published metric
depends on the printed Algorithm 2 form.

## Track D - Exact-vs-Approximate Gap Map

**READ 2026-06-20.**

The safe formalization for the lane is:

```text
I*              exact SEL interval from the oracle
I_box^n         endpoint range over zero-loss n-dimensional BoxSEL embeddings
I_sample^{n,N}  endpoint range over N ordinary training runs
```

With zero-loss samples and under the paper's geometric-volume soundness:

```text
I_sample^{n,N} subset I_box^n
I_box^n subset I*
```

The first inclusion is definitional. The second is the anchor paper's Theorem 3
applied to every zero-loss embedding.

**Important correction to the lane charter:** A "coherence gap" is not banked as a
BoxSEL failure. The paper claims zero-loss inference soundness. The only honest
ways to keep a coherence watch item are:

1. **finite-vs-volume semantics audit:** our exact oracle uses finite counting
   models; the embedding uses geometric volume semantics. Phase 2 must verify that
   the micro-fragment oracle and the BoxSEL soundness target are aligned before
   asserting `I_box^n subset I*`;
2. **implementation/training audit:** a training run with near-zero numerical loss
   may satisfy sampled/normalized constraints but not the full intended theory;
3. **PMP/evaluation audit:** the published evaluation procedure may instantiate
   PMP differently from Proposition 2.

Until one of those audits finds a real mismatch, the lane should use only three
banked gaps:

```text
search gap          I_box^n minus I_sample^{n,N}
representation gap  I* minus I_box^n
loss gap            estimates outside I* when L_T(w) > epsilon
```

**Boundary action:** Do not say "BoxSEL is incoherent" or "zero-loss can escape the
true interval." Say: "we test whether the oracle, the geometric relaxation, and
the implemented training/evaluation pipeline coincide on small cases."

## Track E - Selective Prediction, Calibration, and KG Uncertainty

**READ 2026-06-20.**

The abstention part of the lane has many ancestors:

- Chow gives the classical reject-option framing.
- Geifman and El-Yaniv bring selective classification/risk-coverage to deep nets.
- SelectiveNet trains a reject option end-to-end.
- Guo et al. provide the standard modern calibration warning.
- The KGE/conformal line by Zhu/Potyka/Kharlamov/Staab and collaborators gives
  statistically valid answer sets or intervals for KG embeddings.

**What remains distinct:** Sundog's proposed guard is not "confidence calibration"
and not conformal coverage. It is a trace-based detector for one specific failure:
a narrow embedding interval that is narrow because the search or representation
pipeline missed models that the SEL ontology still permits.

Candidate trace signals:

```text
endpoint drift as restarts accumulate
endpoint movement as embedding dimension changes
ordinary-vs-extremal endpoint disagreement
constraint slack concentration
regularization sensitivity
loss/query-gradient conflict
low-loss basin instability
```

**Boundary action:** Any public "abstention" language must cite reject-option /
selective prediction. Any "KG uncertainty" language must cite conformal KGE
neighbors. The Sundog novelty claim is only the trace source and the false-closure
target.

## Track F - v1 to v2 Notes

**READ 2026-06-20.**

v1 and v2 are the same broad technical program, but v2 changes the public framing
and evaluation presentation.

Banked differences from source/html inspection:

- v1 has a DL/lung-disease style introduction; v2 reframes the motivation around
  Simpson's paradox, UC Berkeley admissions, medicine, public policy, and
  structured statistical reasoning.
- v2 adds a clearer "no implementation exists" / exact-LP discussion in the
  introduction.
- v2 names the method as BoxSEL in the experimental comparison and adds explicit
  fixed/random/KDE baselines.
- v2 adds Appendix B on resolving Simpson's paradox with SEL.
- v2 is formatted for UAI 2026 and arXiv lists it as accepted at UAI 2026.
- The toy interval example `[0.43, 0.83]` vs `[0.16, 0.96]` is already present in
  v1.
- The Algorithm 2 `+ 1 - q2` artifact is present in both v1 and v2; v2 additionally
  changes the printed second premise from `(Q2 | A and Q1)` to `(Q2 | A)`.

**Boundary action:** The operator's "v2 is looser" read should be rendered as:
"v2 is broader and more public-facing; it strengthens motivation and baseline
framing." Do not claim a mathematical weakening without a dedicated diff.

## Claim Ledger

| Lane claim | Placement | Source / action |
| --- | --- | --- |
| SEL conditionals express proportions over finite domains. | STANDARD | Penaloza/Potyka; anchor paper Sec. 3. |
| Exact SEL reasoning is ExpTime-hard / impractical at scale. | STANDARD | Bednarczyk; anchor paper intro. |
| Exact micro-oracle by type enumeration + LP. | STANDARD method used locally | Lutz/Schroder lineage; implement only for tiny fragments. |
| Concepts as boxes, roles as affine transforms. | STANDARD | BoxEL / anchor paper. |
| Query probabilities as box-volume ratios. | STANDARD | Box embeddings / anchor paper. |
| Zero-loss embedding/inference soundness. | STANDARD anchor result | Anchor paper Theorems 1 and 3. |
| Sampled min/max intervals can be too tight. | STANDARD anchor result | Anchor paper Example 3 and discussion. |
| `I_sample subset I_box`. | DEFINITIONAL | Use without novelty language. |
| `I_box subset I*`. | STANDARD under paper semantics; audit under our finite oracle | Theorem 3; Phase 2 semantic check. |
| Search gap vs representation gap vs loss gap. | SYNTHESIS | Sundog framing over standard objects. |
| Coherence gap at zero loss. | WATCH ITEM, not banked | Must be proven as finite-vs-volume or implementation mismatch before claiming. |
| Trace-based false-closure detector. | FIRST PREREG FAILED | Phase-7 run accepted stable PMP-shaped false closures and did not beat restart variance. |
| Phase-6b general trace schema/adapters. | STARTED | Converts Phase-3/Phase-7 diagnostics to `GeneralTrace`; adds PMP pressure producer; no v2 held-out claim. |
| Phase-7b preregistration. | LOCKED, NOT RUN | Boundaries/families/baselines seeded; corpus generator and evaluator built; v2 detector rule and thresholds frozen; held-out results still `NOT_RUN`. |
| Accept/widen/abstain rule. | SYNTHESIS over prior abstention | Cite Chow, Geifman/El-Yaniv, calibration, conformal KGE. |
| PMP `+ 1 - q1` upper slack. | STANDARD / body formula | Proposition 2 and Wagner-style conditional PMP sanity. |
| Algorithm 2 typo / drift. | BANKED ARTIFACT ISSUE | Verified in arXiv source; blast radius unresolved. |

## Phase-1 Replication Gate

Before any oracle or detector work:

1. Build a minimal PMP calculator:

```text
lower(q1, q2) = q1*q2
upper(q1, q2) = min(1, q1*q2 + 1 - q1)
```

2. Unit-test hand cases:

```text
q1 = 1, q2 = r      -> [r, r]
q2 = 1, q1 = r      -> [r, 1]
q1 = 0, q2 = r      -> [0, 1]
q1 = 0.2, q2 = 0.8  -> [0.16, 0.96]
```

3. Test the Algorithm 2 printed variant against the same cases and record the
   failures.
4. Audit premise shape: for each generated query, verify whether the second
   premise is `(Q2 | A and Q1)` or whether a deterministic/subsumption fact makes
   `(Q2 | A)` equivalent.
5. Search again for official code or request it from the authors before making
   any statement about reported metrics.

**Gate output:** `docs/boxsel/PHASE1_PMP_REPLICATION_NOTE.md` with a verdict:

```text
PMP_BODY_CONFIRMED
PMP_ALG2_TYPO_NO_METRIC_BLAST_RADIUS
PMP_ALG2_TYPO_AFFECTS_METRICS
PMP_PREMISE_SHAPE_UNRESOLVED
```

## Phase-2 Exact Oracle Boundary

The oracle must begin role-free and tiny. It should enumerate logical types over
the concept vocabulary and solve an LP for the target conditional. The goal is not
to compete with SEL reasoners; it is to have a ground truth on toy fragments.

Required checks:

- agreement with PMP hand cases;
- agreement with the anchor toy example if encoded cleanly;
- explicit handling of empty conditioning concept;
- exact finite-counting semantics documented separately from geometric-volume
  semantics;
- no web-scale or real-KG language.

## Falsifiers and Failure Modes

- `BOXSEL-LIT-PRIORITY-INFLATION`: claiming SEL, BoxEL, PMP, reject option, or
  conformal KG uncertainty as Sundog inventions.
- `BOXSEL-ZEROLOSS-OVERCLAIM`: claiming a zero-loss coherence gap despite the
  anchor paper's Theorem 3, without a finite-vs-volume or implementation mismatch
  receipt.
- `BOXSEL-PMP-TYPO-AS-ATTACK`: using Algorithm 2 as a rhetorical attack before
  code/metric blast radius is known.
- `BOXSEL-ORACLE-LAUNDERING`: presenting a tiny exact oracle as practical SEL
  reasoning.
- `BOXSEL-CALIBRATION-LAUNDERING`: calling the trace guard a calibration guarantee.
- `BOXSEL-CONFORMAL-CONFLATION`: confusing false-closure detection with conformal
  coverage.
- `BOXSEL-RESTART-VARIANCE-BASELINE-MISSED`: detector beats no trivial baseline
  because restart variance alone was not locked as the comparator.
- `BOXSEL-HELDOUT-TRAP-FAIL`: low-loss, stable, narrow sampled intervals pass the
  guard even though the exact oracle remains wide.

## Allowed / Forbidden Language

**Allowed now:**

> The BoxSEL paper gives a clean substrate for a false-closure experiment: zero-loss
> point estimates are sound, but sampled min/max intervals can be too tight. Sundog's
> question is whether search, representation, and loss-induced narrowing can be
> detected from embedding traces.

**Allowed after Phase 1 only if confirmed:**

> Appendix Algorithm 2 appears to contain a PMP upper-bound typo relative to
> Proposition 2. We verified the correct body formula on hand cases; the effect on
> reported metrics is [verdict].

**Allowed only after a future Phase 6/7 redesign passes:**

> On small SEL fragments with an exact oracle, observable embedding traces flagged
> falsely closed intervals better than restart variance alone.

**Allowed after the Phase-7 bounded null:**

> The first trace-only guard caught the Helly seed-variant traps but failed the preregistered
> falsifier: it accepted stable PMP-shaped false closures and matched, rather than beat, the
> restart-variance-only baseline.

**Forbidden:**

- "Sundog discovered BoxSEL uncertainty."
- "BoxSEL is unsafe / overconfident" as a general claim.
- "Zero-loss BoxSEL can escape the true interval" without a specific semantic or
  implementation mismatch receipt.
- "The paper's results are wrong" before the PMP blast-radius audit.
- "This calibrates KG reasoning."
- "Ask Sundog abstains correctly" before product integration and external review.

## Disposition

The lane is worth running because the anchor paper gives the rare combination of:

```text
hard exact semantics
geometric approximation
zero-loss point soundness
explicit sampled-interval narrowness
small examples with exact endpoints
```

That makes false closure measurable rather than rhetorical. The contribution to
earn is not an interval formula or a box embedding. It is a delineator: a trace
rule that can say, before seeing the oracle, "I do not have enough models to be
this sure."

---

*Sundog Research Lab - BoxSEL lit-pass memo. Phase 0.5 filled 2026-06-20. Internal
claim-boundary artifact; not a result report.*
