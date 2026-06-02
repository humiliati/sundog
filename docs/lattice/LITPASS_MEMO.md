# Lattice-Deduction Lit-Pass Memo

> Prior-art and claim-boundary record for the Sundog vs. Lattice-Deduction lane
> (Phase -1 paper-design freeze). This memo records what the citation spine
> supports as of 2026-06-02, so the lane can use abstract-interpretation and
> neural-reasoning vocabulary without (a) claiming the definitional α/γ soundness
> as a Sundog result, or (b) treating a future-dated focal preprint's internal
> claims as established.

**Date:** 2026-06-02
**Status:** Phase -1 citation spine for [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md).
Treat all gap claims here as time-stamped: "not locked in this pass," not "does
not exist."
**Citation check (2026-06-02):** the eight **anchor-literature** works below were
resolved and verified by title / author / arXiv-ID / venue via web search this
pass. The **focal preprint** (LDT, arXiv 2605.08605) had its *architecture*
verified via the HTML on 2026-06-02; its bibliometrics (author list, venue, its
own reference list, the headline numbers) are **pending** and are quarantined as
to-verify (`LATTICE-FUTURE-CITE-UNVERIFIED`).
**Surfaces:** `docs/lattice/LANE_CHARTER.md`, `docs/lattice/PHASE0_MINIMUM_FALSIFIABLE.md`,
`docs/SUNDOG_V_LATTICE.md`. No public page, `site-pages.json` entry, executable
probe, build-gate, or receipt is live. `lattice.html` is Phase-6 / late-path.

## Purpose

Place the Lattice Deduction Transformer (LDT) in its real literature and fix the
claim boundary **before** any reimplementation, so Phase 0's B2 twin-state cell
rests on a verified spine. The useful one-line spine is:

> The LDT's α/γ Galois pair is the Sundog projection/fiber in direct form — so the
> *control-sufficient shadow leg is architecturally given*, and the only open
> question is whether the **learned** within-pass body is high-dimensional and
> sound across the certified fiber.

That spine justifies a measurement lane. It is **not** a claim that sound abstract
interpretation is a Sundog discovery, nor evidence about the original paper's
unreleased model.

## Claim Boundary

This lane explicitly does **not** claim:

1. that Sundog proves, validates, or extends the LDT paper;
2. that the α/γ ↔ projection/fiber coupling is a Sundog discovery (it is a reading
   of textbook abstract interpretation — Cousot & Cousot 1977);
3. that the *definitional* state-insufficient/control-sufficient separation (the
   ideal α/γ operator) is a new result — that is **Target A folklore**, named and
   walked past (the C1 Mori–Zwanzig posture);
4. that sound abstract interpretation, discrete diffusion, or neural-symbolic
   solving is a Sundog contribution;
5. that a reimplemented-model measurement is a statement about the authors'
   unreleased weights;
6. any Sudoku-solving, reasoning-SOTA, or "Sundog solves symbolic reasoning" claim;
7. that a public workbench is warranted before the build-gate and external review.

The admissible Phase-(-1) claim is narrower:

1. The LDT abstraction/concretization setup is *structurally coupled* to Sundog's
   projection/fiber vocabulary (reading, not theorem).
2. The lineage (HRM, TRM) the LDT positions against is real, and the LDT's move —
   replacing an opaque recurrent latent with an interpretable lattice — is the
   reason the shadow leg is architecturally given.
3. The body measurement must use an **information-basis** estimator, not variance
   PR, because the body is residual-stream-like (Massive Activations) — the chatv2
   lesson, with a literature anchor.

## Method

Targeted searches covered five fronts: (1) the abstract-interpretation foundation
(Galois connections, soundness); (2) the recurrent neural reasoners the LDT
positions against (HRM, TRM); (3) neural-symbolic / differentiable constraint
solving and learned Sudoku/SAT (SATNet, RRN, NeuroSAT); (4) the discrete-diffusion
reframing the LDT borrows (D3PM); (5) residual-stream interpretability that grounds
the body measurement (Massive Activations). Preference was given to arXiv records,
proceedings pages, and author/lab code repos. This is a starter pass, not a
comprehensive survey; the Phase-7 external review remains required.

## Primary Anchors (verified 2026-06-02)

| Source | URL | Supports | Boundary action |
| --- | --- | --- | --- |
| **LDT (focal preprint)** — "Lattice Deduction Transformer" | https://arxiv.org/abs/2605.08605 | The substrate: a per-cell candidate-set lattice as the *inter-step state*, α/γ Galois maps, a learned CLS conflict head (`λ_cls=0.1`, `θ=0.6`), `d=128` / 4 layers / `L=16`, 800K params, Sudoku-Extreme / Snowflake / Maze. | **Architecture verified via HTML (2026-06-02); bibliometrics + headline numbers PENDING.** Do not cite its internal claims as established; the lane reproduces and *measures*, it does not vouch for the paper. |
| **Cousot & Cousot 1977** — "Abstract Interpretation: A Unified Lattice Model…" (4th POPL) | https://dl.acm.org/doi/10.1145/512950.512973 | The foundation: Galois connections/insertions between concrete and abstract lattices; soundness by construction; fixpoint approximation. | **This is the α/γ + soundness folklore = Target A.** Cite as the *source the lane borrows from*, never as a Sundog result. The soundness guarantee is theirs. |
| **HRM** — Sapient Intelligence, "Hierarchical Reasoning Model" | https://arxiv.org/abs/2506.21734 | Two-timescale recurrent reasoner, 27M params, single-pass, 100% Sudoku-Extreme / 100% Maze-Hard / 40.3% ARC-AGI-1; passes an **opaque latent** between recurrent steps. | The opaque-latent reasoner the LDT replaces with an *interpretable lattice*. Sibling/lineage, **not** a Sundog baseline-to-beat. Model-builder reviewer source. |
| **TRM** — Jolicoeur-Martineau, "Less is More: Recursive Reasoning with Tiny Networks" | https://arxiv.org/abs/2510.04871 | 7M-param, 2-layer recursive reasoner beating HRM on ARC/Sudoku/Maze; the other opaque-latent recurrent reasoner. | Second lineage anchor; the LDT's framing ("latent passed opaquely between recurrent steps") refers to HRM/TRM. Reviewer/build-fidelity source (Samsung SAIL code public). |
| **SATNet** — Wang, Donti, Wilder, Kolter (ICML 2019) | https://arxiv.org/abs/1905.12149 | Differentiable MAXSAT (SDP relaxation) in a deep net; learns 9×9 Sudoku from examples. | Neural-symbolic precedent; the LDT's lattice is a *different, explicit, sound* shadow than SATNet's continuous SDP relaxation. Do not conflate the two shadows. |
| **Recurrent Relational Networks** — Palm, Paquet, Winther (NeurIPS 2018) | https://arxiv.org/abs/1711.08028 | Graph message-passing that solves the hardest Sudoku (96.6%) via 64+ steps of iterative relational reasoning. | The iterative-refinement-over-candidate-state precedent; supports "solver = iterative refinement of a partial-information state." |
| **NeuroSAT** — Selsam, Lamm, Bünz et al. (ICLR 2019) | https://arxiv.org/abs/1802.03685 | Message-passing SAT classifier from single-bit supervision; **"the satisfying assignment can almost always be decoded from its activations."** | **Direct precedent for the Phase-0 cross-decode guard.** Caution: if the *solution* is decodable from the LDT body, the "bounded sound deduction" story changes — NeuroSAT shows learned solvers *can* carry the answer in activations. Bake into F-modes. |
| **D3PM** — Austin, Johnson, Ho, Tarlow, van den Berg (NeurIPS 2021) | https://arxiv.org/abs/2107.03006 | Discrete-state denoising diffusion (absorbing/structured transition kernels). | Anchors the LDT's reframing "a logical solver is a special case of discrete diffusion targeting *uniform over valid solutions*." The reframing is the LDT's, borrowed from D3PM; not a Sundog claim. |
| **Massive Activations** — Sun et al. (2024) | https://arxiv.org/abs/2402.17762 | A few residual-stream features carry magnitudes up to ~10⁵× the median (e.g. LLaMA2-7B dims 1415/2533). | **Grounds the chatv2 masked-variance → information-basis lesson with a citation.** Why variance PR is the wrong body-dimensionality estimator here; use `d_dec` (Phase 0 §3.4). |

## Internal Spine (Sundog anchors, not external lit)

| Doc | Role |
| --- | --- |
| [`../proof/PDE_C1_PROPOSITION.md`](../proof/PDE_C1_PROPOSITION.md) | The twin-state adjudicator + the "decision-observable but state-unobservable" measure (`R²(R\|Φ_K)=0.99`) that B2 ports to an *exact* computational fiber. |
| [`../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md`](../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md) | The information-basis fingerprint (`d_dec`, cross-decode) and the masked-variance Amendment-1 lesson this lane inherits. |
| [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) | The body-resistance axis (§6.3) + the marginal control-substrate column this lane tries to beat with a *certified* (not measured) state-insufficiency. |

## Track A — The Structural Coupling (reading, not theorem)

| LDT object (Cousot α/γ) | Sundog object |
| --- | --- |
| `α`: concrete grids → candidate-set lattice | projection `π` / shadow map |
| `γ(a) = {grids consistent with a}` | fiber over the shadow |
| `α` many-to-one (`|γ(a) ∩ solutions| > 1`) | state-insufficiency (certified, not estimated) |
| `ded_p(a) = α(γ(a) ∩ ‖p‖)` is a function of `a` | decision factors through the shadow = control-sufficiency |
| abstract-interpretation **soundness** | control-sufficiency **with a guarantee** |
| learned within-pass activations | candidate high-dimensional body |

This is a **reader translation**. The α/γ structure and its soundness are Cousot &
Cousot 1977; the lane does not claim them. The novel content is entirely the
*measured* behavior of the **learned** model (Target B).

## Track B — The Recurrent-Reasoner Lineage (HRM, TRM)

The LDT's pitch is *against* opaque-latent recurrent reasoners: HRM (2506.21734)
and TRM (2510.04871) both pass an uninterpreted latent between recurrent steps and
hit ~100% Sudoku-Extreme. The LDT replaces that latent with the interpretable
lattice — which is exactly **why the Sundog shadow leg is architecturally given
here and discovered/learned there**. Boundary: HRM/TRM are *siblings/lineage*, not
Sundog baselines; the lane does not benchmark against them and makes no capability
claim relative to them.

## Track C — Neural-Symbolic / Learned Solvers (SATNet, RRN, NeuroSAT)

These establish that learned systems do constraint reasoning by iterative
refinement of a candidate state (RRN), via differentiable relaxations (SATNet), or
message passing (NeuroSAT). Two boundary actions:

1. **The LDT shadow is distinct.** SATNet's shadow is a continuous SDP relaxation;
   NeuroSAT's is a learned embedding; the LDT's is an *explicit, sound* lattice.
   Do not import their guarantees into the LDT or vice versa.
2. **NeuroSAT is a live caution.** "The satisfying assignment can be decoded from
   its activations" means a learned solver *can* secretly carry the answer. The
   Phase-0 **cross-decode guard** (decode `g*` from the body) is exactly the test
   that separates "bounded sound deduction" from "secretly solved, dribbling
   eliminations." This is folded into Phase-0 §3.4 and failure mode F-cross.

## Track D — The Discrete-Diffusion Reframing (D3PM)

The LDT's framing — a logical solver as a special case of discrete diffusion that
targets a *uniform* distribution over valid solutions rather than a
plausibility-weighted one — borrows D3PM (2107.03006). Boundary: this reframing is
the LDT's contribution (or D3PM's lineage), **not** a Sundog result; the lane uses
it only to explain why "iterative refinement of a partial-information state" is the
right altitude for the body/shadow reading.

## Track E — The Abstract-Interpretation Fence (Cousot, Target A)

The single most important fence for this lane:

> Sound abstract interpretation already gives a control-sufficient decision over a
> state-insufficient abstraction *for the ideal operator*. That is Cousot & Cousot
> 1977. The lane **acknowledges and walks past it** (Target A). Every claimed leg
> except the certified-fiber existence must be measured on the **trained network**
> (false-elimination rate, `d_dec`), or the result is `LATTICE-FOLKLORE-AS-RESULT`.

## Track F — The Body-Measurement Basis (Massive Activations)

chatv2 Amendment 1 found variance PR is *masked* by outlier residual directions on
LLM-like bodies. Massive Activations (2402.17762) is the literature anchor: a few
residual features dominate variance by ~10⁵×. Therefore the Phase-0 body measure is
**information-basis** (`d_dec` = effective rank of stacked decision-readout
directions; cross-decode), with variance PR reported only for continuity, never as
the gate. This is a *methods* inheritance, with a citation, not a new claim.

## Peer-Review Contact — Shortlist (owner-to-select; none contacted)

Mapped to the roadmap Phase-7 reviewer profiles. **No contact happens at Phase -1**
— this is a target list so the Phase-7 packet
([`SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) §Phase 7) has named candidates;
the email/packet pattern (owner fills `[Name]`/`[link]`) is the same as the
Hodge / NSE / Yang-Mills external-review drafts.

| Phase-7 profile | Candidate(s) | Best placed to answer |
| --- | --- | --- |
| Abstract interpretation / Galois | **Patrick Cousot** (NYU); or an AI-community successor (e.g. Antoine Miné, Xavier Rival, ENS/Inria) | "Is α/γ → projection/fiber a legitimate coupling or just vocabulary?" — the foundational author of the machinery the lane borrows. |
| Neural reasoner / LDT-style builder | **Alexia Jolicoeur-Martineau** (Samsung SAIL Montreal, TRM); or the Sapient/HRM team | Build-gate fidelity ("is the reimplementation faithful?") and whether the within-pass body measurement is fair for a tiny recursive reasoner. |
| Neural-symbolic / differentiable logic | **J. Zico Kolter** / **Po-Wei Wang** (CMU, SATNet) | Whether "lattice as control-shadow" is sound vs the SATNet/NeuroSAT precedents — esp. the cross-decode guard (NeuroSAT's decodable-assignment result). |
| Constraint programming / exact Sudoku | a CP / exact-enumeration specialist (profile; validate fiber enumeration against a standard exact solver, e.g. the Tdoku lineage) | "Does the twin-state construction certify state-insufficiency in the claimed sense?" — correctness of `γ(a) ∩ solutions` enumeration. |
| Interpretability / residual-stream outliers | **Neel Nanda** (mech interp); or the Massive-Activations authors (Mingjie Sun / locuslab) | "Is `d_dec` a defensible body-dimensionality proxy given outlier features?" |

Selection + contact is an **owner action at Phase 7**, gated on a build-gate-pass +
a non-vacuous B-layer result; a `CERTIFIED_MARGINAL_BODY` or `UNSOUND` outcome
changes which reviewer is asked first (per the roadmap §Phase 5/8 ladder).

## Local Corrections / To-Verify From This Pass

- The LDT focal preprint is **architecture-verified only**; its authors, venue, and
  headline numbers are **not** verified — do not cite them as established
  (`LATTICE-FUTURE-CITE-UNVERIFIED`). The web search surfaced a same-month
  future-dated cluster (e.g. a 2605.08504 "massive activations" follow-up); anchor
  on the *verified* 2024 Sun et al. (2402.17762), not the future-dated ID.
- HRM (2506.21734) and TRM (2510.04871) are **real and code-public** (Sapient;
  Samsung SAIL) — the LDT's lineage framing is grounded.
- Cousot & Cousot 1977 is the canonical α/γ source; the soundness guarantee is
  theirs. Any lane copy implying Sundog "found" the control-sufficient shadow on
  this substrate is wrong by construction (Target A).
- NeuroSAT's decodable-assignment result is a *caution*, not a comfort: it means the
  cross-decode guard can genuinely fire (the body may carry `g*`).
- Variance PR is a **known-masked** estimator here (Massive Activations) — the memo
  binds the lane to information-basis `d_dec`.

## Pre-Registered Negatives (lane-specific)

- `LATTICE-ANALOGY-INFLATION` — the lane claims credit for abstract-interpretation
  folklore rather than using it as the substrate (roadmap Phase -1 primary failure).
- `LATTICE-FOLKLORE-AS-RESULT` — the definitional α/γ (Target A) separation is
  presented as a measured finding.
- `LATTICE-VACUOUS-INTERSTEP` — the body is measured on the inter-step state
  (`FVE ≡ 1` trivially) instead of within-pass activations.
- `LATTICE-REIMPL-LAUNDERING` — a reimplemented-model result is framed as a claim
  about the authors' unreleased model.
- `LATTICE-FUTURE-CITE-UNVERIFIED` — leaning on a future-dated arXiv ID (incl. the
  focal LDT paper's own bibliometrics) as if fully verified.
- `LATTICE-LIT-MISMATCH` — the memo misstates the HRM/TRM/SATNet/RRN/NeuroSAT/Cousot
  positioning or a benchmark number.
- `LATTICE-CROSSDECODE-MISSED` — the body secretly carries `g*` (NeuroSAT-style) and
  the lane reads "bounded sound deduction" anyway.

## Cross-References

- [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) — parent roadmap + phase ladder.
- [`LANE_CHARTER.md`](LANE_CHARTER.md) — the α/γ reframe + Target A/B split.
- [`PHASE0_MINIMUM_FALSIFIABLE.md`](PHASE0_MINIMUM_FALSIFIABLE.md) — the frozen B2
  cell this spine supports (the cross-decode guard, the `d_dec` body measure).
- [`../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md`](../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md)
  — the masked-variance → information-basis lesson (with Massive-Activations anchor).
