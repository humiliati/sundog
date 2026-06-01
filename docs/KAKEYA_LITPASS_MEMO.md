# Kakeya Lit-Pass Memo

> Prior-art and claim-boundary record for the Sundog vs. Kakeya
> reader/workbench lane. This memo records what the current citation spine
> supports as of 2026-06-01, so the lane can use Kakeya vocabulary without
> drifting into a Euclidean, finite-field, maximal-function, or regime-2 claim.

**Date:** 2026-06-01
**Status:** Starter memo filed as the citation spine for
[`SUNDOG_V_KAKEYA.md`](SUNDOG_V_KAKEYA.md). Treat all gap claims here as
time-stamped: "not locked in this pass," not "does not exist."
**Surfaces:** `docs/SUNDOG_V_KAKEYA.md`,
`docs/KAKEYA_FINITE_FIELD_READER.md`,
`docs/kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`, `legend.html`. No
public page, `site-pages.json` entry, executable probe, or public launch claim
is live.

## Purpose

Move Kakeya from "lit-pass anchors landed" to a bounded Front-A reader lane.
The outside mathematics stays primary; Sundog's role is a claim-boundary and
body-resistance reading, plus a possible finite-field workbench after a domain
lock.

The useful spine is:

> Kakeya is the exact-maximal body-resistance anchor opposite Faraday-zero.

That spine is enough to justify a reader/workbench page later. It is not a
proof route, not a transfer principle, and not a regime-2/control-sufficiency
result.

## Claim Boundary

Sundog did not prove, verify, extend, or formalize any Kakeya result.

This lane explicitly does not claim:

1. a proof of the Euclidean Kakeya conjecture;
2. an improvement to any Kakeya, restriction, incidence, or maximal-function
   bound;
3. a new proof of the finite-field Kakeya theorem;
4. transfer from finite fields to Euclidean Kakeya sets;
5. transfer from the 2025 Wang-Zahl resolution in `R^3` to open `n >= 4`,
   to finite fields, or to any Sundog claim;
6. a regime-2 / control-sufficiency separation on Kakeya;
7. evidence that a browser needle-field visualization demonstrates a dimension
   lower bound.

The admissible claim is narrower:

1. Kakeya is a clean high-math anchor for the body/shadow or
   body-resistance axis.
2. The finite-field theorem is a safe reader/workbench substrate because it is
   known, algebraic, and formally checkable in a current Lean artifact.
3. The Euclidean 3D result, the open Euclidean `n >= 4` problem, the
   finite-field theorem, and toy pixel/tube visuals are separate registers.

## Method

Targeted searches covered six fronts:

1. the current Euclidean status after Wang-Zahl's 2025 `R^3` preprint;
2. Dvir's finite-field theorem and the polynomial-method proof spine;
3. finite-field refinements and adjacent maximal-function language;
4. algorithmic-information treatments of Kakeya, especially point-to-set and
   conditional Kolmogorov complexity;
5. the Math Inc. Lean 4 finite-field formalization;
6. Sundog-facing public-copy risks: finite-field laundering, 3D-to-`n >= 4`
   laundering, maximal-function drift, and regime-2 overclaim.

Preference was given to arXiv records, journal or publisher pages, official
repository pages, and author-hosted/expository material by domain experts.
This is a starter lit pass, not a comprehensive Kakeya survey; external
incidence-geometry review remains required before public inbound links,
external promotion, or treating a Kakeya page's launch intent as satisfied. An
unlinked `NOT PEER REVIEWED` page may exist as a reviewer surface.

## Primary Anchors

| Source | Public URL | Supports | Boundary action |
| --- | --- | --- | --- |
| Hong Wang and Joshua Zahl, "Volume estimates for unions of convex sets, and the Kakeya set conjecture in three dimensions", arXiv:2502.17655 (2025) | https://arxiv.org/abs/2502.17655 | Current Euclidean landscape: every Kakeya set in `R^3` has Hausdorff and Minkowski dimension 3. | Hard fence: only `R^3`; no transfer to `n >= 4`, finite fields, maximal function, or Sundog claims. |
| Terence Tao, "The three-dimensional Kakeya conjecture, after Wang and Zahl" (2025) | https://terrytao.wordpress.com/2025/02/25/the-three-dimensional-kakeya-conjecture-after-wang-and-zahl/ | Expository orientation for the 3D result, tube-discretized intuition, sticky/non-sticky language, and the stronger maximal-function caution. | Context only; not a replacement for the Wang-Zahl preprint. |
| Zeev Dvir, "On the size of Kakeya sets in finite fields", arXiv:0803.2336 / JAMS 22(4), 2009 | https://arxiv.org/abs/0803.2336 | Finite-field theorem: a Kakeya set in `F_q^n` contains a line in every direction and has size at least `C_n q^n`. Primary polynomial-method reader spine. | This is the finite-field workbench substrate, not evidence for Euclidean Kakeya. |
| Zeev Dvir, Swastik Kopparty, Shubhangi Saraf, and Madhu Sudan, "Extensions to the Method of Multiplicities, with Applications to Kakeya Sets and Mergers", SICOMP 42(6), 2013 | https://epubs.siam.org/doi/10.1137/100783704 | Sharper finite-field lower bound `q^n / 2^n` and method-of-multiplicities context. | Use only if the page discusses constants or near-optimal finite-field bounds. |
| Jack H. Lutz and Neil Lutz, "Algorithmic information, plane Kakeya sets, and conditional dimension", arXiv:1511.00442 | https://arxiv.org/abs/1511.00442 | Point-to-set principle and conditional Kolmogorov-complexity route to the known 2D Kakeya theorem. | Supports the algorithmic-information vocabulary; not a high-dimensional proof. |
| Nicholas G. Polson and Daniel Zantedeschi, "Kakeya Conjecture and Conditional Kolmogorov Complexity", arXiv:2603.25611 (2026) | https://arxiv.org/abs/2603.25611 | Body/shadow bridge: fiber label as direction shadow, chain-rule body/residual split, incompressibility as maximal body-resistance, adaptive-fibering obstruction. | Binding fence: body-resistance only, not regime-2/control sufficiency. |
| math-inc/KakeyaFiniteFields, Lean 4 repository | https://github.com/math-inc/KakeyaFiniteFields | AI-assisted Lean 4 formalization of the finite-field Kakeya theorem, produced by Math Inc.'s Gauss agent from a LaTeX blueprint. Front-A evaluator exhibit. | Read/cite/audit only unless licensing changes; no code copying into Sundog without a visible license grant. |

## Track A - Euclidean Boundary

Wang-Zahl is the current landscape anchor: the Euclidean Kakeya set conjecture
is resolved in three dimensions, with Hausdorff and Minkowski dimension equal
to 3 for Kakeya sets in `R^3`.

The Sundog boundary language should be severe:

> The 3D Euclidean theorem is a landmark result. It does not transfer to open
> `n >= 4`, to finite fields, to a browser workbench, or to any Sundog claim.

If the public copy says "Kakeya is solved" without "in `R^3`", it fails this
memo. If it suggests that finite-field or pixel/tube experiments support the
open Euclidean problem, it fails this memo.

## Track B - Finite-Field Reader Spine

The primary Sundog target is the finite-field theorem as a known-positive
reader/workbench substrate:

> In `F_q^n`, a Kakeya set contains a line in every direction; Dvir proves that
> its size is bounded below by a constant depending only on `n` times `q^n`.

The intended reader spine is:

1. assume a too-small Kakeya set;
2. build a nonzero low-degree polynomial vanishing on the set;
3. use the full-line-in-every-direction condition to force vanishing in all
   directions;
4. obtain the contradiction that the polynomial must be zero.

That proof shape is ideal for Sundog because the "shadow" is not a fuzzy
metaphor: direction coverage forces algebraic constraints. But it is still
outside mathematics. Sundog may explain it and test how legibly a workbench
teaches it; Sundog may not repackage it as a new theorem.

## Track C - Formal / Evaluator Exhibit

The Math Inc. repository is valuable because it is the same epistemic object
the capset/unit-distance lane wants to learn how to read: AI-assisted
mathematics passed through a formal checker.

Current admissible use:

- cite the repository as an external formalization of the finite-field theorem;
- audit its theorem statement, dependency graph, build instructions, and
  proof outline;
- if a later phase clones it, record the commit hash, Lean toolchain, build
  command, and outcome in a receipt.

Current inadmissible use:

- copying Lean code or blueprint assets into Sundog while no license file is
  visible;
- treating "machine-checked" as "Sundog-verified";
- using a finite-field formalization as evidence for Euclidean Kakeya.

## Track D - Body-Resistance Bridge

Polson-Zantedeschi supplies the real hook for the lane. Their setup maps cleanly
onto Sundog vocabulary:

| Polson-Zantedeschi object | Sundog reading |
| --- | --- |
| fiber label | direction shadow |
| point on the fiber | body/residual state |
| Kolmogorov chain rule at precision `r` | body/shadow split |
| incompressibility at ambient dimension | maximal body-resistance |
| adaptive fibering | identifiability boundary |

This justifies the phrase:

> Kakeya is the exact-maximal body-resistance anchor opposite Faraday-zero.

The bridge is bindingly limited. Polson-Zantedeschi discuss reconstruction
description length and dimensional incompressibility. They do not supply a
control objective, control sufficiency, or a lossy shadow predicting a different
task. Therefore Kakeya is not a Reading-2 / regime-2 witness.

## Track E - Public Workbench Safety

A later browser workbench is admissible only if it keeps three layers visibly
separate:

1. **Finite field:** discrete directions and exact finite-field lines.
2. **Toy grid / pixel model:** a visualization, not the theorem.
3. **Euclidean boundary:** open `n >= 4` and non-transfer explicitly labeled.

The first public-facing artifact should be Front A: a reader/boundary page. The
current review-surface exception is narrow: a live but unlinked `kakeya.html`
may be sent to reviewers if it carries a visible `NOT PEER REVIEWED` banner and
has no obvious public inbound path. That does not clear public promotion. Front
B visuals remain admissible only behind the finite-field workbench spec with
direction conventions, baselines, UI labels, and falsifiers pre-registered.

## Local Corrections From This Pass

- Do not call Wang-Zahl "Kakeya solved" without "in `R^3`".
- Do not state or imply that Wang-Zahl settles the Kakeya maximal-function
  conjecture.
- Do not treat Dvir's finite-field theorem as evidence for the Euclidean
  conjecture.
- Do not use a browser needle animation as evidence for any dimension lower
  bound.
- Do not call the Kakeya bridge regime-2. It is body-resistance only.
- Keep the source order citation-first: lit-pass memo, Wang-Zahl for Euclidean
  status, Dvir for finite fields, Polson-Zantedeschi/Lutz-Lutz for
  algorithmic-information framing, Math Inc. for the Lean artifact, Sundog
  ledger last.

## Candidate Ranking

| Rank | Candidate | Disposition | Why |
| --- | --- | --- | --- |
| 1 | Finite-field Kakeya reader | Promote to Phase 1 after this memo is linked. | Known theorem, clean body/shadow split, low overclaim if fenced. |
| 2 | Euclidean boundary note | Merge into the reader or file beside it. | Necessary to prevent finite-field and 3D laundering. |
| 3 | Lean formalization audit sidecar | Stage after reader draft. | Strong Front-A evaluator exhibit, but requires license/build hygiene. |
| 4 | Tiny finite-field workbench | Spec filed; implementation still gated. | Useful spectacle if the direction shadow does not re-encode the body. |
| 5 | Needle-field spectacle | Hold. | High visual value, high miscalibration risk. |

## Pre-Registered Negatives

- `KAK-FRONT-A-VACUOUS`: the reader says only what a standard Kakeya
  exposition already says, with no added claim-boundary or body-resistance
  clarity.
- `KAK-FINITE-FIELD-LAUNDERING`: any copy implies the finite-field theorem
  supports the Euclidean conjecture.
- `KAK-3D-LAUNDERING`: any copy implies the Wang-Zahl `R^3` result supports
  open `n >= 4`.
- `KAK-BRIDGE-OVERCLAIM`: body-resistance vocabulary is advertised as
  regime-2/control sufficiency.
- `KAK-SHADOW-REENCODING`: the proposed direction shadow secretly stores or
  reconstructs the full point/tube set.
- `KAK-WORKBENCH-MISCALIBRATED`: the visualization teaches pixel density,
  area, or overlap as if it were a dimension theorem.

## Page-Level Framing

Allowed lead:

> Every direction is present. The body almost vanishes.

Allowed support:

> Sundog reads Kakeya as the maximal-resistance pole of the body/shadow axis:
> direction coverage is visible, but the body cannot be compressed below full
> dimension in the known theorem registers.

Required fence:

> This is a reader/workbench around known mathematics. It is not a proof,
> finite-field-to-Euclidean bridge, or regime-2 claim.

Forbidden:

> Sundog is working toward a proof of Kakeya.

Forbidden:

> Finite-field Kakeya is evidence for the Euclidean conjecture.

Forbidden:

> The 2025 3D solution suggests Sundog can attack `n >= 4`.

Forbidden:

> Kakeya proves that a direction shadow is control-sufficient.

## Updated Roadmap Scaffold

The Kakeya ledger should now use this order:

1. **Front A reader first.** Write a bounded finite-field/body-resistance reader
   before any visual page or code.
2. **Boundary paragraph before spectacle.** Every public-facing artifact must
   separate finite-field, Euclidean `R^3`, open `n >= 4`, maximal-function, and
   toy-grid claims.
3. **Lean sidecar after source audit.** Clone/build only in a later receipt,
   with commit hash and license observation recorded.
4. **Workbench spec before implementation.** Phase 2 now freezes field size,
   dimension, direction convention, line convention, displayed signature,
   baselines, and falsifiers before writing UI.
5. **External review before public linking/promotion.** Ask a
   combinatorics/incidence or analysis reviewer whether the reader teaches the
   right boundary. An unlinked `NOT PEER REVIEWED` review surface can exist
   before this clears; a public launch claim cannot.

## Follow-Up Ratchets

- If Wang-Zahl receives a journal version, correction, or explicit maximal
  function sequel, update Track A before any public refresh.
- If the Math Inc. repository adds a license, update the audit/copy boundary.
- If a specialist reviewer says the Polson-Zantedeschi bridge is too soft for
  public framing, demote it to an internal vocabulary note and keep the public
  page finite-field-first.
- If a new Kakeya page is added to `site-pages.json`, it must clear Bucket 1 in
  `docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md` before external sharing or any
  `publicLaunchIntent` claim. Reviewer-only sharing of an unlinked page must use
  the `NOT PEER REVIEWED` banner.
