# Sundog vs. Kakeya

Working hook:

> Every direction is present. The body almost vanishes.

Short version:

> Kakeya asks how small a set can be while still containing a unit segment in
> every direction. Sundog asks a bounded reader/workbench question: when the
> body is visually tiny but direction-complete, what shadow certifies the
> hidden incidence structure, and where does that certificate stop transferring?

Status: Scaffold + lit-pass anchors landed (2026-05-31); the real-hook gate is
adjudicated, and the starter
[`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md) was filed 2026-06-01, and the
Phase-1 Front-A reader
[`KAKEYA_FINITE_FIELD_READER.md`](KAKEYA_FINITE_FIELD_READER.md) is drafted (it
clears its `KAK-FRONT-A-VACUOUS` self-check on the body-resistance placement and
the claim-boundary fences). The Phase-2 tiny finite-field workbench spec is
filed at
[`kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md).
The Phase-3 workbench is built as an **internal, non-deployed** artifact under the
repo-root `kakeya/` dir (`workbench.html` + `kakeya-core.js` + `kakeya-workbench.js`);
`npm run kakeya:test` passes all spec-§9 acceptance checks (33/33 across
`q ∈ {5, 7, 11}`), and
[`kakeya/PHASE3_WORKBENCH_QA.md`](kakeya/PHASE3_WORKBENCH_QA.md) records the
desktop/mobile visual QA pass. The **Front-A reward graphic** is started in the
same internal `kakeya/` dir (`gallery.html` + `kakeya-gallery.js`): a
body-resistance continuum (Faraday-zero ↔ Kakeya-maximal, with the marginal and
Aharonov–Bohm markers) plus a gallery of finite-field instances (each a body in
`F_q²` paired with its lossy direction-shadow fan, generated from the verified
core; render QA recorded in
[`kakeya/PHASE3_GALLERY_QA.md`](kakeya/PHASE3_GALLERY_QA.md)). It mirrors the
threebody isotrophy-gallery style and carries the same fences (finite-field,
body-resistance only). It is
deliberately outside the site build (vite only scans root-level `.html`), so it
carries no `site-pages.json` entry and is not launched. Public copy, a real
`kakeya.html` page, and a site-page entry are still pending.
The review-gate interpretation is now narrower: an unlinked `kakeya.html`
review surface may go live with a visible `NOT PEER REVIEWED` banner so a
specialist has something concrete to inspect, but public inbound links,
external promotion, and any `publicLaunchIntent` claim remain gated on the
external sanity check plus SEO/social readiness.
The H-K3 shadow-collision audit is also filed as an internal Phase-3B receipt:
[`kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md`](kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md)
records a reproduced bounded `q=5` enumeration where `245506` bodies collapse to
`7` registered direction-shadow signatures, plus deterministic line-extension
audits for `q=7` (`2408` states, max nonempty collision class `301`) and `q=11`
(`14652` states, max nonempty collision class `1221`). The exact structured
count `q(q^2 - q + 1)` passes across the supported rungs, the empirical
reencoding falsifier is clear, and `npm run kakeya:test` now pins
different-size same-shadow collisions across `q in {5, 7, 11}` (`42/42`).
Phase 3D turns this receipt into a finite-geometry lemma:
[`kakeya/PHASE3D_SHADOW_COLLISION_LEMMA.md`](kakeya/PHASE3D_SHADOW_COLLISION_LEMMA.md)
proves the generalized safe range `line + <= q-2 outside points`, the sharp
first break at `q-1`, and the per-direction count
`q * sum_{i=0}^{q-2} C(q^2 - q, i)` (`npm run kakeya:shadow-lemma`; core tests
`48/48`).
Phase 3E generalizes the threshold into a per-direction metric:
[`kakeya/PHASE3E_DIRECTION_ACTIVATION_SPECTRUM.md`](kakeya/PHASE3E_DIRECTION_ACTIVATION_SPECTRUM.md)
computes, for any body, the exact witnessed minimal number of added points
that lights each direction (`min` over that direction's intercept lines of
the missing-point count), with closed-form spectra for lines, stars, and the
pencil-minus-one closure (`npm run kakeya:activation-spectrum`; q=5 exhaustive
minimality cross-check; core tests `54/54`).
Phase 3F solves the joint problem:
[`kakeya/PHASE3F_JOINT_ACTIVATION_GAP.md`](kakeya/PHASE3F_JOINT_ACTIVATION_GAP.md)
computes the exact minimal addition lighting a *set* of directions, proves the
gap sandwich `0 <= sum-of-marginals - joint <= C(k,2)`, and locates the tax:
from structured bodies the deficit is zero for every proper direction subset
and exactly `(q-1)/2` for the full pencil, whose joint cost is the minimal
complete Kakeya set - exhaustively `17`/`31` at `q=5`/`7` (= the imported
Blokhuis-Mazzocca minimum) and witnessed `<= 71 < 77 = greedy` at `q=11` via
parabola tangents (`npm run kakeya:joint-gap`; core tests `60/60`).
Phase 3G probes embeddability in minimal sets:
[`kakeya/PHASE3G_EMBED_MINIMAL_PROBE.md`](kakeya/PHASE3G_EMBED_MINIMAL_PROBE.md)
proves the concurrency-budget identity `ex = sacrifice - (q-1)/2` (a `k`-star
embeds only if `(k-1)(k-2)/2 <= (q-1)/2`) and resolves the frontier at all
three fields: 3-stars and below embed everywhere, 4-stars nowhere (budget
necessary but not sufficient), with measured cross-ratio dependence
(`ex = 2` vs `3` at `q=7`) and greedy's `q=11` excess `6` = its sacrifice
overspend (`npm run kakeya:embed-probe`; core tests `66/66`).
Phase 3H closes the deficit-onset question:
[`kakeya/PHASE3H_DEFICIT_ONSET.md`](kakeya/PHASE3H_DEFICIT_ONSET.md)
dissolves the inf-dichotomy by axis avoidance - a parabola's tangent family
covers every direction except its axis, so any direction set with `k <= q`
achieves zero deficit via an axis-avoiding parabola, self-certified against
the proven pairwise bound (4094 sets at `q=11`, retiring the Phase-3F cap);
the onset is exactly `k = q+1` ("no axis left"; dually, arc saturation), tax
`(q-1)/2`, boundary axis-symmetric across all `q+1` axis choices
(`npm run kakeya:deficit-onset`; core tests `72/72`). The Blokhuis-Mazzocca
planar minimum is now bibliographically pinned in the litpass memo addendum
(2026-07-06).
Phase 3I sweeps the 4-star cross-ratio orbits:
[`kakeya/PHASE3I_CROSSRATIO_ORBIT_SWEEP.md`](kakeya/PHASE3I_CROSSRATIO_ORBIT_SWEEP.md)
solves all 580 direction quadruples exactly - embeddability excess is
constant on every PGL(2,q) orbit, with the split real at `q=7` (harmonic 2 vs
equianharmonic 3) and collapsed at `q=11` (harmonic = generic = 4, the new
measurement); reopen condition for the equianharmonic question is `q=13`,
outside the locked field range (`npm run kakeya:crossratio-sweep`; core tests
`78/78`).
Phase 3J runs that reopen condition as an out-of-register `q=13` sidecar
(register untouched):
[`kakeya/PHASE3J_EQUIANHARMONIC_PROBE.md`](kakeya/PHASE3J_EQUIANHARMONIC_PROBE.md)
**falsifies the equianharmonic conjecture** - at `q=13` the deviator is the
*harmonic* orbit (6 vs 5=5) - and replaces it with a sharper banked pattern:
all 8 measured orbit-field pairs have `ex in {(q-3)/2, (q-1)/2}` (never above
the concurrency budget, never more than one below), with the harmonic orbit
at full budget exactly when `q = 1 (mod 4)`; discriminating fields `q=17`/
`q=19` banked as new pre-registrations (`npm run kakeya:equianharmonic`).
Phase 3K runs the `q=17` sidecar:
[`kakeya/PHASE3K_TWOLEVEL_PROBE.md`](kakeya/PHASE3K_TWOLEVEL_PROBE.md)
certifies all three `q=17` orbits exactly at the LOW level (`7/7/7`) - the
**two-level law strengthens to 11/11** and generic orbits are uniform, but
the **harmonic mod-4 rule is falsified** (second consecutive first-test
pattern death); instrument amendments documented (star-pivot scaling
symmetry ~8x, controls revalidated); surviving mod-8 observation flagged as
third-iteration forking-paths suspect; `q=19` staged as an owner-fired run
with pre-registrations (`npm run kakeya:twolevel`).
Phase 3L takes the construction route (Track 1):
[`kakeya/PHASE3L_STAR_PARABOLA_CONSTRUCTION.md`](kakeya/PHASE3L_STAR_PARABOLA_CONSTRUCTION.md)
anatomizes the optimal completions (pivot always exactly mult-4, everything
else triples; parabola-tangent signature at `q=13`), extracts a parabola
family (axis in star directions + tangents in the non-star directions) that
matches the solver-certified optimum at 10/11 known field-orbits (gap:
`q=11` harmonic), and maps the excess to `q = 37` in seconds: two-level law
extends 37/37 in upper-bound form, mod-8 harmonic rule survives everywhere
decidable, chi-equianharmonic gets a definitive low at `q=37`, generic
uniformity breaks at `q in {23, 31}` (upper bounds), and the staged `q=19`
B&B shrinks to two load-bearing solves (`npm run kakeya:star-anatomy`,
`npm run kakeya:star-construction`).
Phase 3M executes the three banked moves:
[`kakeya/PHASE3M_INVARIANT_HUNT_AND_EXTENSION.md`](kakeya/PHASE3M_INVARIANT_HUNT_AND_EXTENSION.md)
mines the orbit invariants (bare chi-signature FAILS on exact data; two
survivors: `sig+type` and `sig+chi(j-1728)`, the latter subsuming mod-8 via
harmonic j = 1728 identically), closes the `q=11`-harmonic family gap by
descent (validation 11/11 MATCH-ALL), finds every construction high
descent-resistant, and isolates ONE rule-vs-instrument divergence -
19-harmonic (rules LOW, descent 9). The q=19 harmonic B&B solve
(12.2B nodes, exact) returns **9 = HIGH**: the descent was right, both
character rules die, and re-mining shows the entire character battery is
exhausted (the critical 11-h/19-h pair is separated only by `chi(j)`, which
is globally inconsistent) - the level is not orbit-character-determined, so a
deeper arithmetic invariant is forced (PHASE3N reopener). Two-level law
survives (`9 in {8,9}`); the equianharmonic solve returns **9 = HIGH**
(exact, EQ-3 confirmed), closing the PHASE3M empirical leg - descent exact at
13/13 known field-orbits, two-level exact-confirmed through `q=19` and
UB-verified to `q=37`, global character classifier falsified, one
within-equianharmonic sub-pattern still live.
Phase 3N resolves the deeper-invariant thread as a NULL:
[`kakeya/PHASE3N_DEEPER_INVARIANT.md`](kakeya/PHASE3N_DEEPER_INVARIANT.md)
reduces the whole classification to `sig+type` with a UNIQUE exception
(`11-harmonic`), isolates the question to `q=3 (mod 8)` harmonic (where the
governing quartic character degenerates), and extends by construction to find
**`q=11` is the only LOW in the class** (`19..131` all HIGH). No arithmetic
invariant separates `q=11` - it is a small-field anomaly. The level IS
`sig+type` (quadratic-character signature + orbit type), full stop; the
deeper-invariant hypothesis is retired (`npm run kakeya:deeper-invariant`).
Phase 3O turns the level into a geometric mechanism:
[`kakeya/PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md`](kakeya/PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md)
verifies that `sacrifice = 3 + T` where T counts triple points, each provably
`{1 star line, 2 tangents}` (a tangent-chord pole on a star line; no 3
tangents concurrent), so the level is a cross-ratio-controlled incidence count
(LOW = q-5, HIGH = q-4). The parabola completion is optimal for every 4-star
except the lone `q=11`-harmonic (a non-parabola optimum), which pins the
small-field anomaly geometrically; and the parabola prices `q=43` harmonic
HIGH in ms, subsuming the infeasible depth-40 exact solve (lever 1). Lever 2
reduces to two sharp finite-geometry lemmas (parabola-optimality for `q>=13`;
the cross-ratio pole-incidence count) (`npm run kakeya:triple-anatomy`).
Phase 3P attacks parabola-optimality (lever 2, lemma 1):
[`kakeya/PHASE3P_PARABOLA_OPTIMALITY.md`](kakeya/PHASE3P_PARABOLA_OPTIMALITY.md)
PROVES the concurrency identity `sacrifice = |K| - q(q+1)/2` (so minimizing
sacrifice = minimizing the completion's point count), the optimal profile
`sacrifice = 3 + T`, and the dual reformulation (completion -> `q+1` points
with the 4 star lines forced collinear on `O*`; multiplicity = rich-line
count), verified `D1+D2+D3` at every exact field (`npm run kakeya:dual-arc`).
It then REDUCES parabola-optimality to a *relative Segre* statement (min-
3-secant near-arc with a forced pivot line = conic on the free points),
cited to Segre 1955 + Blokhuis-Mazzocca, shown NOT a corollary of BM, with
the `q=11` exception forcing any proof to use `q>=13`. Honest status: the
reduction and scaffolding are proved/verified; the relative-Segre lemma
itself is OPEN (Segre pinned in the litpass memo addendum).
Phase 3Q attempts to close the relative-Segre lemma and reports the honest
outcome - it does NOT close:
[`kakeya/PHASE3Q_RELATIVE_SEGRE_STATUS.md`](kakeya/PHASE3Q_RELATIVE_SEGRE_STATUS.md)
PROVES parabola-optimality exhaustively as a finite theorem (`q<=19` exact,
recomputed from scratch at `q<=13`) with the SINGLE exception `q=11`
harmonic, which DISPROVES the universal form; locates why the general
`q>=13` case resists (non-arc dual config with a forced 4-secant, so Segre's
Lemma of Tangents is obstructed; not the Ball-Blokhuis-Domenzain result;
untested beyond `q=19`); and reframes to the cleaner exception-free FLOOR
conjecture `ex >= (q-3)/2` (holds at all tested `q` incl. 11) as the better
target. No proof of the general lemma is claimed (`npm run kakeya:parabola-opt`).
This file is a target-selection and claim-boundary scaffold opened from
[`SUNDOG_HIGH_STAKES_PROBLEM_MATRIX.md`](SUNDOG_HIGH_STAKES_PROBLEM_MATRIX.md).

This is not a claim to progress on the Kakeya conjecture, the Kakeya maximal
function problem, restriction theory, finite-field incidence geometry, or any
Euclidean dimension lower bound. It is a proposal for a bounded Sundog-facing
artifact: a finite-field-first spectacle and evaluator lane with a hard guard
against laundering a solved finite-field theorem into an open Euclidean claim.

## Claim Boundary

This document explicitly does **not** claim:

- a proof of the Euclidean Kakeya conjecture;
- an improvement to any Kakeya, restriction, or incidence bound;
- a new proof of the finite-field Kakeya theorem;
- transfer from finite fields to Euclidean sets;
- transfer of the 2025 Wang–Zahl resolution of *3D* Euclidean Kakeya to the open
  `n ≥ 4` case, to finite fields, or to any Sundog claim;
- a regime-2 / control-sufficiency separation on Kakeya — the body-resistance
  bridge below is the *state* half only (see the honest fence);
- evidence that a visual needle-field workbench says anything about the open
  problem.

What this scaffold may stage, after the literature pass:

- a defensible reader lens for the known finite-field Kakeya theorem and its
  polynomial-method proof shape;
- a browser-native workbench that makes "direction-complete but spatially
  small" legible without pretending the visualization is evidence;
- a pre-registered boundary note separating finite-field, discrete-grid,
  tube-discretized, and Euclidean claims;
- a candidate public spectacle that rhymes with `Pressure Mines`, `Cap-set`,
  and the body-resistance vocabulary in
  [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md).

## Lit-Pass Anchors and the Body-Resistance Bridge (2026-05-31)

Lit-pass anchors are identified and the starter
[`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md) is filed. The gate on a
*real intellectual hook* — prioritized before any page or graphics — has been
adjudicated.

**Recent landscape.**

- **Wang–Zahl (2025)** resolved the Euclidean Kakeya conjecture *in three
  dimensions*: every Kakeya set in `R³` has Hausdorff and Minkowski dimension 3
  ([arXiv:2502.17655](https://arxiv.org/abs/2502.17655)). `n ≥ 4` remains open.
  The lane reads the resolved 3D result, the open `n ≥ 4`, and the finite-field
  theorem as three *separate* registers; laundering one into another is the hard
  stop (failure mode 2).
- **Dvir (2008)** — finite-field Kakeya by the polynomial method, the root of the
  method that later cracked cap-set; the finite-field workbench substrate.
- **math-inc/KakeyaFiniteFields** — a *complete Lean 4 autoformalization* of
  Dvir's theorem, produced by Math Inc.'s "Gauss" agent from a LaTeX blueprint
  ([github](https://github.com/math-inc/KakeyaFiniteFields)). The Front-A evaluator
  exhibit: AI-produced, machine-verified math — the same epistemic object the
  cap-set lane is built to read. (No license stated → read/cite/audit, not copy.)

**The body-resistance bridge — the real hook, gate cleared (bounded).**
Polson–Zantedeschi (2026), ["Kakeya Conjecture and Conditional Kolmogorov
Complexity"](https://arxiv.org/abs/2603.25611), independently reconstruct the
Sundog body/shadow decomposition in algorithmic-information terms and apply it to
Kakeya:

- the **fiber label** = the line *direction* = the lossy **shadow**;
- their chain rule `K(x↾r) = K(z↾r) + K(u↾r | z) + O(log r)` (Prop. 1) is the
  body/shadow split made exact — direction + along-fiber residual;
- **"informationally incompressible at ambient dimension"** is maximal
  **body-resistance**: the direction-shadow cannot compress the set below full
  dimension (= the Kakeya conjecture), via the Lutz point-to-set principle;
- the **adaptive-fibering obstruction** (§5.2) — body-resistance is exact only when
  the shadow is *identifiable*; a point on many fibers lets an adversary pick the
  max-compression direction — is precisely why `n ≥ 4` is open while `R³`
  (sticky/Lipschitz) is not.

So **Kakeya is the exact-*maximal* body-resistance anchor** — the opposite pole
from Faraday's exact-*zero* anchor (Bianchi) on the axis the
[`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) failure map already runs.
A genuine, non-manufactured spine for the lane.

**Honest fence (binding).** This is a **body-resistance** bridge, **not** a
regime-2 / Reading-2 separation. The Polson–Zantedeschi framework has *no
control-sufficiency notion* — it is reconstruction description length throughout.
The direction-shadow is "control-sufficient for the direction" only *trivially*
(it *is* the direction), not the non-trivial "lossy shadow predicts a *different*
objective" that the real regime-2 substrates (NSE C1, Aharonov–Bohm) have.
Claiming a Kakeya regime-2 is forbidden overreach. Sundog supplies the
body-resistance *reading* and the cross-substrate placement; the mathematics is
Wang–Zahl / Dvir / Lutz / Polson–Zantedeschi — no Sundog-original theorem.

**Framing decision.** Front A (reader/workbench page + graphics) is justified on
*this* body-resistance spine — the Faraday-zero ↔ Kakeya-maximal continuum is a
real visual — not on a manufactured regime-2. Abstract vocabulary parallels are
tracked at [`/legend`](../legend.html).

## Why Kakeya Fits Sundog

Kakeya is a strong Sundog target because the hidden/body object and readable
shadow are almost painfully separated:

- **Body:** a small-looking set, grid subset, or tube union that secretly
  contains a line or needle in every direction.
- **Shadow:** direction coverage, incidence counts, polynomial constraints,
  projection data, or tube-overlap structure.
- **Tension:** the body can look visually negligible while the direction shadow
  is complete.
- **Spectacle:** a visitor can watch needles rotate through a sparse-looking
  field and see that "small" does not mean "direction-poor."

The conceptual fit is strongest next to three existing Sundog lanes:

- **Cap-set / unit-distance:** finite-field and incidence geometry as public
  reader workbenches, not Sundog-original mathematics.
- **Gimmicks / Pressure Mines:** hidden geometry leaking through a field-like
  proxy; Kakeya can become the high-math version of "the field bends around a
  hidden thing."
- **Cross-substrate body resistance:** Kakeya is a dimensional-resistance
  stress test in the reader/explainer class. The body is combinatorial or
  geometric; the shadow is directional/incidence-complete.

## The Coupling Claim

The coupling is staged on two fronts.

### Front A - Reader / Evidence-Tier Lens

Sundog can use its claim-boundary apparatus to read the finite-field Kakeya
story:

- What exactly is the body?
- What exactly is the shadow?
- Which step turns direction-completeness into a size lower bound?
- Which parts are finite-field-specific?
- Which public visual intuitions are helpful, and which are dangerous?

The product is a reader note or exhibit, not a theorem. The first useful
Sundog contribution would be evidence-tier hygiene: showing why a solved
finite-field result is a clean workbench for the projection discipline while
still not being evidence for the open Euclidean problem.

### Front B - Browser Workbench / Spectacle

A browser workbench can show three layers side by side:

1. **Finite grid:** directions are discrete; each direction requires a full
   line in the selected set.
2. **Tube / pixel model:** Euclidean-looking needles are thickened into pixels;
   overlap and resolution become visible.
3. **Claim boundary:** the page explicitly labels which statements are known
   finite-field facts, which are discretized analogies, and which are open
   Euclidean territory.

Front B should be promoted only if Front A is clean. A pretty needle animation
without the finite-field/Euclidean boundary is a liability, not an asset.

## Falsification Surface

The coupling claim can fail or be bounded in five named modes:

1. **Front-A vacuity.** The Sundog reader note says only what any careful
   exposition of Kakeya already says. The apparatus adds no edge.
2. **Finite-field laundering.** The artifact implies that the solved
   finite-field theorem supports the Euclidean conjecture without a separately
   registered bridge. This is a hard stop.
3. **Visualization miscalibration.** The workbench teaches the wrong lesson:
   visitors infer area, measure, or dimension claims from pixel density,
   animation overlap, or finite-grid artifacts.
4. **Shadow re-encoding.** The "direction shadow" simply stores the full body
   or an equivalent lookup table. The workbench has become reconstruction,
   not projection.
5. **Incidence-bound mismatch.** After the literature pass, the proposed
   reader/probe target does not align with the actual state of Kakeya
   literature, or misses the important obstruction named by specialists.

Nulls are admissible. A clean "this is only a good spectacle, not a research
instrument" result is still useful if the boundary is named before public
presentation.

## Initial Domain Map

The literature pass must lock these domains before any probe or page work:

- **Finite-field model:** field size(s), dimension(s), direction convention,
  line convention, and known theorem statement.
- **Discrete-grid visual model:** whether it is a finite-field model, an
  integer-grid toy, or a pixel/tube approximation. These are not the same.
- **Euclidean boundary:** exact wording for what remains open and what no
  finite workbench may imply.
- **Direction shadow:** the registered observable: direction coverage, line
  incidence, tube overlap, polynomial vanishing condition, or another named
  signature.
- **Baseline:** trivial construction, random grid subset, greedy cover, or
  known small examples, chosen after the lit pass.

Expansion requires a new phase sign-off. No scope creep from "finite-field
reader" to "Euclidean evidence" is allowed midstream.

## Sundog Expression

| Sundog object | Kakeya target | Expression / probe | Primary falsifier |
| --- | --- | --- | --- |
| Body/shadow split | Direction-complete set | Body = selected points/tubes; shadow = certified direction coverage | 2, 4 |
| Projection discipline | Direction/incidence summary | Can a compact signature certify direction-completeness without reconstructing the set? | 4 |
| Cap-set reader precedent | Polynomial-method proof shape | Explain known finite-field proof as a reader artifact with evidence tiers | 1, 5 |
| Gimmick spectacle | Needle-field visualization | Make "small but every direction" visible as a first-contact demo | 3 |
| Boundary ledger | Finite-field vs Euclidean | Separate known theorem, toy model, and open conjecture in page copy | 2 |

## Initial Probe / Artifact Shortlist

Each item below is scaffold-only until the lit-pass memo decides whether it is
admissible.

### Candidate 1 - Finite-Field Kakeya Reader

Working hook:

> The polynomial does not see a sparse set. It sees every direction.

- **Front:** A.
- **Cost:** Low.
- **Artifact:** a short reader note that explains the known finite-field proof
  structure in Sundog body/shadow vocabulary, with a strict "not Euclidean
  evidence" boundary.
- **Attacks failure modes:** 1, 2, 5.
- **Pre-registered negative:** if the note cannot say anything more precise
  than a standard exposition without adding misleading vocabulary, file
  `KAK-FRONT-A-VACUOUS` and do not promote.

### Candidate 2 - Tiny Finite-Field Workbench

Working hook:

> Click the points. The direction shadow tells you what is missing.

- **Front:** A/B bridge.
- **Cost:** Low to medium.
- **Artifact:** spec filed at
  [`kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md).
  The first implementation target is prime-field `F_q^2` only (`q in {5, 7,
  11}`): a user selects points, sees which directions are covered by full
  lines, and compares size against simple teaching baselines.
- **Attacks failure modes:** 3, 4.
- **Pre-registered negative:** if the direction shadow is only a disguised copy
  of the point set, or if the UI makes size lower bounds look like empirical
  discoveries, file `KAK-SHADOW-REENCODING` or
  `KAK-WORKBENCH-MISCALIBRATED`.

### Candidate 3 - Needle-Field Spectacle

Working hook:

> The set is almost invisible until the directions light up.

- **Front:** B.
- **Cost:** Medium.
- **Artifact:** a browser-native visual layer showing rotating needles/tubes,
  overlap, resolution, and direction coverage.
- **Attacks failure mode:** 3.
- **Pre-registered negative:** if the visual cannot display the finite
  resolution caveat clearly, hold it as internal art only. Do not publish.

### Candidate 4 - Euclidean Boundary Note

Working hook:

> The finite shadow is clean. The Euclidean boundary is where the trouble lives.

- **Front:** A.
- **Cost:** Low after lit pass.
- **Artifact:** a boundary note stating what finite-field Kakeya, discretized
  tube models, maximal-function formulations, and Euclidean dimension claims do
  and do not share.
- **Attacks failure modes:** 2, 5.
- **Pre-registered negative:** if the bridge cannot be stated without expert
  caveats dominating the claim, keep Kakeya as a finite-field reader/spectacle
  only.

## Phase Plan

### Phase 0 - Literature Pass and Domain Lock

Goal: keep [`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md) as the
citation/domain lock before any technical or public move.

The starter memo now locks:

- canonical problem statements and variants;
- finite-field theorem statement and proof references;
- Euclidean open-boundary wording;
- known relationships to restriction/maximal-function language, if included;
- which examples are safe for a public visual;
- external-review path.

Starter exit criterion: memo filed and linked here. Public-promotion exit
criterion: claim boundary and candidate ranking updated after the reader draft
and an external sanity-check path.

### Phase 1 - Reader Note

Goal: write Candidate 1 as a bounded Front-A artifact.

Exit criterion: the note survives an internal vacuity check and does not imply
Euclidean progress.

### Phase 2 - Tiny Workbench Spec

Goal: pre-register the finite-field toy model, direction convention, baselines,
UI labels, and falsifiers before implementation. Filed at
[`kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md).

Exit criterion: no hidden equivalence between the direction shadow and full
body reconstruction. The spec requires many-to-one shadow collisions and a
shadow export that omits point membership and witness intercept lists.

### Phase 3 - Browser Workbench

Goal: implement the finite-field workbench and optional needle-field spectacle
with claim-boundary labels baked into the interface.

Internal exit criterion: **met** for the finite-field workbench core and UI; see
[`kakeya/PHASE3_WORKBENCH_QA.md`](kakeya/PHASE3_WORKBENCH_QA.md). Public launch
exit criterion: page-copy audit, external sanity check, and Bucket 1 SEO/social
requirements if a page is added to `site-pages.json`. An unlinked
`NOT PEER REVIEWED` page may be used as a review surface before this gate clears,
but it is not public promotion.

### Phase 4 - External Sanity Check

Goal: ask a combinatorics/incidence-geometry reviewer whether the reader note
and workbench teach the right boundary.

Packet: [`kakeya/EXTERNAL_REVIEW_PACKET.md`](kakeya/EXTERNAL_REVIEW_PACKET.md).

Exit criterion: either promote to public educational workbench/public inbound
links or file the named boundary/null and keep the live page, if any, as an
unlinked `NOT PEER REVIEWED` review surface.

## Lit-Pass Checklist

The filed starter [`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md) answers
these at a first-pass level. Before any public page, recheck:

- Which Kakeya variant is the primary Sundog target?
- What is the exact finite-field statement, with dimension and field-size
  dependencies?
- Which proof is the intended reader spine?
- Which finite examples are small enough for a browser workbench?
- What is the cleanest Euclidean boundary language?
- Which adjacent topics are in scope: maximal function, restriction,
  sum-product/incidence, joints, cap-set/polynomial method?
- What should be explicitly excluded to prevent overclaim?
- Who is the external-review audience?

## Promotion Criteria

This scaffold becomes an active roadmap only when all are true:

- the literature pass is filed and linked here;
- the finite-field / Euclidean boundary is stated in one paragraph a
  non-specialist can understand;
- at least one candidate has a pre-registered negative;
- the artifact can be useful even if it only teaches a boundary;
- public copy can avoid all forbidden phrases below.

## Claim-Language Guardrails

Allowed:

> Sundog is drafting a finite-field-first Kakeya reader and workbench. It asks
> how direction-complete shadows certify hidden incidence structure, and where
> that certification stops.

Allowed:

> The Kakeya workbench is a spectacle for the boundary between finite,
> discretized, and Euclidean claims.

Forbidden:

> Sundog is working toward a proof of Kakeya.

Forbidden:

> The finite-field theorem is evidence for the Euclidean conjecture.

Forbidden:

> A browser needle animation demonstrates a dimension lower bound.

Forbidden:

> Direction coverage is a useful shadow if it secretly reconstructs the full
> set.

## Cross-References

- [`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md) - 2026-06-01 starter
  citation spine and claim-boundary lock for the Kakeya lane.
- [`KAKEYA_FINITE_FIELD_READER.md`](KAKEYA_FINITE_FIELD_READER.md) - Phase-1
  Front-A reader; clears `KAK-FRONT-A-VACUOUS` on body-resistance placement and
  claim-boundary fences, not on the standard proof retelling.
- [`kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md)
  - Phase-2 pre-implementation lock for the tiny prime-field workbench.
- [`kakeya/PHASE3_WORKBENCH_QA.md`](kakeya/PHASE3_WORKBENCH_QA.md) - internal
  Phase-3 QA pass for the non-deployed workbench.
- [`kakeya/PHASE3_GALLERY_QA.md`](kakeya/PHASE3_GALLERY_QA.md) - internal
  render QA pass for the body-resistance reward graphic and finite-field
  instance gallery.
- [`kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md`](kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md)
  - internal H-K3 audit showing the registered direction shadow is many-to-one
  on bounded `q=5` states, structured `q in {5, 7, 11}` line-extension states,
  and the Phase-2 guard witnesses.
- [`kakeya/PHASE3D_SHADOW_COLLISION_LEMMA.md`](kakeya/PHASE3D_SHADOW_COLLISION_LEMMA.md)
  - finite-geometry lemma upgrading H-K3 from a measured one-point receipt to a
  closed-form one-direction collision family with sharp `q-1` break threshold.
- [`kakeya/PHASE3E_DIRECTION_ACTIVATION_SPECTRUM.md`](kakeya/PHASE3E_DIRECTION_ACTIVATION_SPECTRUM.md)
  - direction activation spectrum: exact witnessed minimal added-point cost per
  unlit direction for any body, with closed-form line/star/pencil spectra and a
  `q=5` exhaustive minimality cross-check.
- [`kakeya/PHASE3F_JOINT_ACTIVATION_GAP.md`](kakeya/PHASE3F_JOINT_ACTIVATION_GAP.md)
  - joint-vs-marginal activation gap: exact joint lighting cost for direction
  sets, gap sandwich `[0, C(k,2)]`, full-pencil concurrence tax `(q-1)/2`, and
  machine-verified Blokhuis-Mazzocca instances at `q in {5, 7}` (imported
  anchor; bibliographic pin landed 2026-07-06 in the litpass memo addendum).
- [`kakeya/PHASE3G_EMBED_MINIMAL_PROBE.md`](kakeya/PHASE3G_EMBED_MINIMAL_PROBE.md)
  - embeddability-in-minimal-sets probe: concurrency-budget identity
  `ex = sacrifice - (q-1)/2`, star frontier (3-stars embed, 4-stars do not,
  cross-ratio-dependent excess), affine certificates plus exact completion
  solver, all bodies resolved at `q in {5, 7, 11}`.
- [`kakeya/PHASE3H_DEFICIT_ONSET.md`](kakeya/PHASE3H_DEFICIT_ONSET.md)
  - deficit onset by axis avoidance: zero deficit for every direction set with
  `k <= q` via axis-avoiding parabola tangents (self-certifying against the
  proven bound; full subset lattice at all fields), onset exactly at the full
  pencil with tax `(q-1)/2`, axis-symmetric boundary.
- [`kakeya/PHASE3I_CROSSRATIO_ORBIT_SWEEP.md`](kakeya/PHASE3I_CROSSRATIO_ORBIT_SWEEP.md)
  - 4-star cross-ratio orbit sweep: excess is a PGL-orbit invariant (580/580
  exact solves), split at `q=7` (harmonic 2, equianharmonic 3), collapsed at
  `q=11` (both orbits 4); equianharmonic reopen condition = `q=13`.
- [`kakeya/PHASE3J_EQUIANHARMONIC_PROBE.md`](kakeya/PHASE3J_EQUIANHARMONIC_PROBE.md)
  - `q=13` out-of-register sidecar: equianharmonic conjecture FALSIFIED
  (harmonic is the `q=13` deviator, 6 vs 5); banked two-level law
  `ex in {budget-1, budget}` (8/8) with harmonic at full budget iff
  `q = 1 (mod 4)` (4/4); register untouched, no new pins.
- [`kakeya/PHASE3K_TWOLEVEL_PROBE.md`](kakeya/PHASE3K_TWOLEVEL_PROBE.md)
  - `q=17` sidecar, certified exact: all orbits low (`7/7/7`), two-level law
  11/11, generic uniformity confirmed, harmonic mod-4 rule FALSIFIED;
  star-pivot symmetry solver amendment; `q=19` staged owner-fired with
  pre-registrations; mod-8 observation banked as explicitly suspect.
- [`kakeya/PHASE3L_STAR_PARABOLA_CONSTRUCTION.md`](kakeya/PHASE3L_STAR_PARABOLA_CONSTRUCTION.md)
  - Track-1 construction rung: optimal-completion anatomy (pivot exactly
  mult-4 + all-triples; parabola signature), parabola-tangent family matching
  10/11 solver exacts (gap `q=11` harmonic), excess map extended to `q=37` as
  verified upper bounds; two-level 37/37; low-side pattern confirmations
  definitive at 23/31/37; generic split observed; q=19 B&B reduced to two
  solves.
- [`kakeya/PHASE3M_INVARIANT_HUNT_AND_EXTENSION.md`](kakeya/PHASE3M_INVARIANT_HUNT_AND_EXTENSION.md)
  - invariant hunt (two surviving character rules; mod-8 subsumed), descent
  extension closing the `q=11`-harmonic gap (11/11 MATCH-ALL), all highs
  descent-resistant, 19-harmonic divergence isolated; two q=19 exact solves
  (both HIGH, character battery exhausted).
- [`kakeya/PHASE3N_DEEPER_INVARIANT.md`](kakeya/PHASE3N_DEEPER_INVARIANT.md)
  - deeper-invariant thread resolved NULL: classification reduces to
  `sig+type` with unique exception `11-harmonic`; `q=3 (mod 8)` harmonic is
  HIGH for `19..131`, LOW only at `q=11`; no arithmetic separator (quartic
  character degenerates in-class); `q=11` is a small-field anomaly, deeper
  invariant retired.
- [`kakeya/PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md`](kakeya/PHASE3O_TRIPLE_CONCURRENCE_MECHANISM.md)
  - geometric mechanism: level = triple count T of the optimal parabola
  completion (`sacrifice = 3 + T`, every triple `{1 star, 2 tangents}`, no 3
  tangents concurrent), verified G1-G3 at exact fields; parabola optimal except
  `q=11`-harmonic (non-parabola optimum = the anomaly); prices `q=43` HIGH in
  ms (lever 1); lever 2 reduced to two finite-geometry lemmas.
- [`kakeya/PHASE3P_PARABOLA_OPTIMALITY.md`](kakeya/PHASE3P_PARABOLA_OPTIMALITY.md)
  - parabola-optimality reduction: PROVES concurrency identity
  (`sacrifice = |K| - q(q+1)/2`), optimal profile (`3 + T`), and dual
  reformulation (D1 star-collinearity proved; D2/D3 verified); REDUCES the
  lemma to a relative-Segre statement (cited Segre 1955 + BM, not a BM
  corollary; `q=11` forces `q>=13`), left honestly OPEN.
- [`kakeya/PHASE3Q_RELATIVE_SEGRE_STATUS.md`](kakeya/PHASE3Q_RELATIVE_SEGRE_STATUS.md)
  - honest close-attempt: parabola-optimality PROVED exhaustively for `q<=19`
  4-stars with the single exception `q=11` harmonic (which disproves the
  universal form); general `q>=13` OPEN (non-arc dual config obstructs
  Segre's Lemma of Tangents; not in the literature; untested beyond `q=19`);
  cleaner exception-free FLOOR conjecture `ex>=(q-3)/2` recommended. Not
  closed.
- [`kakeya/EXTERNAL_REVIEW_PACKET.md`](kakeya/EXTERNAL_REVIEW_PACKET.md) -
  owner-pending external-review ask; frames Kakeya as a boundary/pedagogy review,
  not a result-review packet.
- [`SUNDOG_HIGH_STAKES_PROBLEM_MATRIX.md`](SUNDOG_HIGH_STAKES_PROBLEM_MATRIX.md)
  - target-selection matrix that promoted Kakeya as the strongest next public
  spectacle candidate.
- [`SUNDOG_V_CAPSET.md`](SUNDOG_V_CAPSET.md) - finite-field / polynomial-method
  reader-workbench precedent.
- [`SUNDOG_V_GIMMICKS.md`](SUNDOG_V_GIMMICKS.md) - hidden-state spectacle and
  anti-perfect-information framing.
- [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md) - body-resistance and
  failure-mode vocabulary.
