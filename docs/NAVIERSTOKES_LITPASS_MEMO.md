# Navier–Stokes Lit-Pass Memo

> Single-page record of the 2026-05-28 lit pass that precedes filing
> [`SUNDOG_V_NAVIERSTOKES.md`](SUNDOG_V_NAVIERSTOKES.md). Records what is
> in the indexed literature as of May 2026 and what is not, so the
> ledger's claim that several candidates address non-vacuous gaps can be
> re-audited.

**Date:** 2026-05-28
**Status:** Filed as the prior-art reference for the candidate ranking
in the Navier–Stokes ledger. Treat the gap claims here as time-stamped:
"not indexed at lit-pass date," not "does not exist."

## Method

Four searches: (A) path signatures and rough paths on NSE / turbulence;
(B) determining modes / data assimilation 2024–2025 state-of-the-art;
(C) PINN regularity / blow-up diagnostics on NSE; bonus track on the
Onsager conjecture as the existing PDE-side precedent for
coarse-graining-as-regularity-threshold. Targeted fetches on the
synchronization-framework paper and the hidden-self-similarity paper.
Two-pass: an initial sweep, then sharper queries on shell-model
intermittency and on signature-based precursor detection.

## Three tracks

### A — Path signatures on NSE / turbulence

Path-signature theory is mature. The signature is a *sufficient statistic
for solutions to rough/ordinary differential equations* (Lyons
universality), the generalized signature method is tooled
([2006.00873](https://arxiv.org/pdf/2006.00873)), randomized signature
layers exist ([2201.00384](https://arxiv.org/abs/2201.00384v1)), and
unsupervised path signatures are now a published anomaly-extraction
primitive
([IEEE 10889612](https://ieeexplore.ieee.org/document/10889612/)).

Rough-path machinery has been applied to NSE: a 2019 rough-perturbation
formulation ([1902.09348](https://arxiv.org/pdf/1902.09348)) and a 2026
fractional-transport-noise limit
([2601.21762](https://arxiv.org/pdf/2601.21762)). But these *define
solutions* under rough noise. They are not regime-diagnosis tools.

**Gap:** no indexed work applies path signatures to NSE-class flows as
**transit / blow-up / regime detectors**. The June 2025 introduction
paper ([2506.01815](https://arxiv.org/pdf/2506.01815)) does not address
NSE or turbulence. Shell-model intermittency in particular — Sabra,
GOY — has zero indexed path-signature usage.

### B — Determining modes / state-of-the-art

Foias–Temam 1984 and Constantin–Foias–Manley–Temam 1985 established
that finitely many modes / nodes determine NSE long-time dynamics.
Data-assimilation results extended this to continuous
nudging-style filters
([SIAM J. Math. Anal. 2020](https://epubs.siam.org/doi/10.1137/20M1323229);
[SIAM J. Appl. Dyn. Sys. local observables](https://epubs.siam.org/doi/10.1137/20M136058X);
3D NSE-α extension by Albanez–Nussenzveig Lopes–Titi).

The 2024 "Synchronization Framework" paper
([2408.01064](https://arxiv.org/pdf/2408.01064)) is the cleanest current
articulation: determining modes ↔ filter convergence ↔ **state
reconstruction**, with "intertwinement" as the formal name for the
two-way implication. Confirmed by direct fetch: the framing is
observer-design / synchronization, **not** Blackwell-sufficiency for a
control objective.

**Gap:** the control-side complement — re-reading determining modes
through [`COARSE_GRAINING_PROOF_ROADMAP.md`](COARSE_GRAINING_PROOF_ROADMAP.md)
Postulate 1 (`𝓕_σ`-measurability of `π*`) — is not in the indexed
literature. This is the Candidate-1 anchor.

### C — PINN regularity diagnostics

PINN work on NSE focuses on solving the equations or parameter ID
(e.g. [ScienceDirect 2024 multi-scale PINN](https://www.sciencedirect.com/science/article/pii/S2590037424000967)).
A confidence-score diagnostic head that scores PINN output for
**regularity / blow-up risk** is essentially absent from indexed work.

**Gap confirmed**, but the experimental cost includes standing up a
working PINN-NSE pipeline, which dominates the cost of Candidate 5.

## Bonus precedent — Onsager 2024/25

Onsager's conjecture is the existing PDE-side precedent for
"coarse-graining sets a regularity threshold": energy is conserved for
C^α with α > 1/3 and dissipation is admissible below. Recent results:
Giri–Radu 2024 settled the flexible part for 2D Euler at γ < 1/3
(via Newton–Nash iteration,
[Inv. Math. 2024](https://link.springer.com/article/10.1007/s00222-024-01291-z));
Du–Li–Ye 2025 strengthened the construction
([2506.15396](https://arxiv.org/abs/2506.15396)). The endpoint regularity
side has the 2024 *Anal. PDE* result
([msp.org 2024 endpoint](https://msp.org/apde/2024/17-6/apde-v17-n6-p09-p.pdf)).

These are the existing pieces of mathematics that **make
Postulate-1-style measurability-threshold framing not analogy but
sibling**. The ledger preamble should cite this so the framing is not
read as imported metaphor.

## Bonus competitor — regime-transition detection community

The baselines a signature-based transit detector has to beat — or
explain its different problem statement against — are well-tooled:
DMD/Koopman ([1904.09082](https://arxiv.org/pdf/1904.09082)),
critical-slowing-down ([1901.08084](https://arxiv.org/pdf/1901.08084)),
recurrence lacunarity ([2101.10136](https://arxiv.org/pdf/2101.10136)),
Rényi-entropy dynamical phase transitions
([2407.13452](https://arxiv.org/pdf/2407.13452)), and the 2024 critical
assessment of these methods ([2406.05195](https://arxiv.org/html/2406.05195v1)).

Honest position: signatures encode trajectory order; CSD encodes
timescale separation; DMD encodes spectral structure; recurrence
captures multiscale return statistics. They are likely complementary,
not competing. The ledger should not claim signatures *replace* these
methods.

## Updated candidate ranking

| Rank | Candidate | Δ from proposal | Cost | Why this rank |
|---|---|---|---|---|
| 1 | C1 — Postulate-1 reading of determining modes | unchanged | Low | 2024 synchronization paper confirms the gap is state-reconstruction-only. Reading note is cheapest, cleanest, lowest-overreach first move. |
| 2 | C3 — Signatures on shell-model intermittency (Sabra/GOY) | **promoted from #3** | Low-Med | Smallest empirical leg. Sabra/GOY runs on a laptop. Hidden self-similarity ([2201.04005](https://arxiv.org/pdf/2201.04005)) gives a target invariant. Instanton/importance-sampling ([2308.00687](https://arxiv.org/pdf/2308.00687)) provides labeled rare-event data. Baselines are explicit and tooled (DMD, CSD, lacunarity, Rényi). |
| 3 | C5 — PINN diagnostic head | unchanged | Med | Gap confirmed; PINN infrastructure cost dominates. Defer until C1 lands. |
| 4 | C2 — Full PDE signature transit detection | demoted | High | Subsumed in spirit by C3 at a fraction of the cost. Reconsider only if C3 returns positive. |
| 5 | C4 — Vorticity structural-zero | demoted | High | Algebraic ansatz needs clarification first; defer. |

## Disposition

Findings sufficient to file the ledger at the LEDGER tier (capset
pattern), with C1 staged as the first proof-track artifact and C3
staged as the first empirical leg. No claim leaves this memo into a
public surface without an external-mathematician sanity check, per
[`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md) discipline.

## Sources

Determining modes and data assimilation:
- [Continuous Data Assimilation for the 3D Navier–Stokes (SIAM J. Math. Anal.)](https://epubs.siam.org/doi/10.1137/20M1323229)
- [Data Assimilation for NSE Using Local Observables (SIAM J. Appl. Dyn. Sys.)](https://epubs.siam.org/doi/10.1137/20M136058X)
- [Determining Modes, State Reconstruction, and Intertwinement (2408.01064)](https://arxiv.org/abs/2408.01064)
- [Continuous data assimilation for 3D Navier–Stokes-α](https://journals.sagepub.com/doi/10.3233/ASY-151351)
- [Determining Map for the 3D Boussinesq System](https://link.springer.com/article/10.1007/s00245-022-09896-7)

Path signatures and rough paths:
- [Generalised Signature Method for Multivariate Time Series (2006.00873)](https://arxiv.org/pdf/2006.00873)
- [Randomized Signature Layers (2201.00384)](https://arxiv.org/abs/2201.00384v1)
- [Path Signatures for Feature Extraction (2506.01815)](https://arxiv.org/pdf/2506.01815)
- [The Signature of a Rough Path: Uniqueness (1406.7871)](https://arxiv.org/pdf/1406.7871)
- [Path Signatures as Unsupervised Anomaly Extractors (IEEE 10889612)](https://ieeexplore.ieee.org/document/10889612/)
- [Rough perturbation of NSE / vorticity formulation (1902.09348)](https://arxiv.org/pdf/1902.09348)
- [Navier–Stokes with fractional transport noise (2601.21762)](https://arxiv.org/pdf/2601.21762)

Shell-model intermittency and rare events:
- [Hidden self-similarity in shell-model intermittency (2201.04005)](https://arxiv.org/pdf/2201.04005)
- [Instanton importance sampling for extreme fluctuations in shell models (2308.00687)](https://arxiv.org/pdf/2308.00687)
- [Direct and inverse cascades scaling in real shell models (2409.11898)](https://arxiv.org/pdf/2409.11898)
- [Statistical ML tools for probabilistic turbulence closures (2502.17316)](https://arxiv.org/abs/2502.17316)

Onsager conjecture:
- [Admissible solutions of the 2D Onsager conjecture (2506.15396)](https://arxiv.org/abs/2506.15396)
- [The Onsager conjecture in 2D: a Newton–Nash iteration (Inv. Math. 2024)](https://link.springer.com/article/10.1007/s00222-024-01291-z)
- [Endpoint regularity in Onsager's conjecture (Anal. PDE 2024)](https://msp.org/apde/2024/17-6/apde-v17-n6-p09-p.pdf)

Regime-transition detection baselines:
- [Critical assessment of time-series-based detection of critical transitions (2406.05195)](https://arxiv.org/html/2406.05195v1)
- [Regime transitions via dynamic mode decomposition (1904.09082)](https://arxiv.org/pdf/1904.09082)
- [Critical slowing down as early warning signal (1901.08084)](https://arxiv.org/pdf/1901.08084)
- [Lacunarity as multiscale recurrence quantification (2101.10136)](https://arxiv.org/pdf/2101.10136)
- [Measuring dynamical phase transitions in time series (2407.13452)](https://arxiv.org/pdf/2407.13452)

PINN on NSE:
- [Multi-scale PINN for NSE (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S2590037424000967)
