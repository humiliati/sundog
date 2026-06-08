# Phase 8 — Catastrophe-Stratum Classification of the Bifurcation Set (receipt ledger)

> **The honest form of "invisible-halo predictions"** (`docs/SUNDOG_V_ATLAS.md` Phase 8): not new-halo
> discovery but the **catastrophe CLASSIFICATION** of the ice-halo bifurcation set — each transition's
> stratum (A₂ fold / A₃ cusp / A₄ swallowtail / D₄ umbilic) **DERIVED** from the halo-function caustic
> (corank from the Jacobian singular values; codimension from the `∂δ` order), never asserted from arc
> shape (the §6 armchair gate). **NOT public-eligible.** Scripts: `scripts/atlas_strata_map.py` +
> `scripts/test_atlas_strata_map.py` (all pass).
>
> **PRIOR-ART RESOLUTION (lit-pass Track B, 2026-06-07) — the novelty is PARTIAL, and the claim is
> reworded accordingly.** Tape & Können 1999 ("A general setting for halo theory", Appl. Opt. 38:1552;
> full 74-pp text read) is **pure parameterization** — zero occurrences of catastrophe/cusp/caustic/
> Jacobian/bifurcation; it even *dissolves* the UTA/LTA→circumscribed transition ("there is only one
> halo, not two or three… the three names are unfortunate and misleading"). So no classified
> bifurcation diagram exists. **BUT the decisive prior art is M. V. Berry, "Supernumerary ice-crystal
> halos?", Appl. Opt. 33:4563 (1994):** Berry already *raised* the catastrophe-typing of ice halos,
> recorded that "some halos do indeed show cusps" (crediting Tape 1979/1983), argued the standard Thom
> A_n/D_n taxonomy **does not transfer wholesale because the halo orientation→deflection map is
> NON-GRADIENT**, **predicted no umbilics (D₄)** generically, and flagged the **swallowtail (A₄) as a
> stable-but-conspicuously-absent open question**. Consequences for this lane:
> - **Defensible claim (reworded):** *not* "first catastrophe classification of ice halos" (Berry
>   pre-empts the idea; Tape the cusp observation) — but "the first **systematic stratification** of the
>   hexagonal-ice halo bifurcation set, taking up Berry's (1994) question and addressing the non-gradient
>   map." Cite Tape 1980 + Tape & Können 1999 + Berry 1994 + Berry-Upstill/Nye.
> - **The A₂/A₃ labels survive the non-gradient objection** by **Whitney's theorem**: folds and cusps
>   are the generic stable singularities of *any* smooth 2-D map (gradient or not). The column's A₃
>   point-cusps are the **UTA/LTA apexes** (8-B locator); the **29.7° merge is the A₃-class
>   *metamorphosis*** (two A₂ folds coalesce — also a Whitney event, valid in the non-gradient map; Berry
>   confirms cusps occur). It is the *higher* strata (A₄/D₄) where gradient-vs-non-gradient bites.
> - **`corank-1` alone does NOT fix A₃ vs A₄** — that needs the 3-jet/4-jet determinacy conditions. The
>   29.7° merge is A₃ by the **two-fold-coalescence topology** (the 6.5-B gap-closure = Whitney cusp),
>   not by corank alone; an A₄ claim would require the higher-jet check.
> - **"No D₄ on the column" CONFIRMS Berry's no-umbilic prediction** — it is a cross-check of Berry, not
>   our discovery. Frame it that way.
> - **Expected outcome = classification, not discovery** (the standard-habit space is 40-yr-swept; the
>   repo's pyramidal program hit a P2 ceiling; the field's novelty mode is observation-first; Berry: no
>   umbilics, swallowtail absent) — **a clean null is honest.**

## Discipline (pre-registered)
- **Gate 1 — derive, never assert:** corank = #(near-zero singular values of `J=∂(sky-chart)/∂(γ,α)`)
  (corank-1 → A_k; corank-2 → D₄); the order (A₂/A₃/A₄) from the `∂δ` vanishing count + the
  caustic-coalescence topology. No `A₄`/`D₄` printed without the rank computation.
- **Gate 2 — catalog cross-check before any "prediction":** every higher stratum → look up in
  `HALO_PHENOMENA_ACCOUNTING.md` §C + the literature. (i) named arc → CLASSIFICATION (expected); (ii)
  named-nothing but renders in HaloSim → P1/P2 candidate, internal only; (iii) doesn't render →
  FALSIFIED. Default framing is "classified the catalog."
- **Gate 3 — bands not points (§0.2);** **bounded-novelty:** the A_n/D_n LABELS are SYNTHESIS
  (Berry-Upstill/Nye classification applied on **Tape 1980**'s caustic = Jacobian-kernel construction);
  cite Tape 1980 + Berry 1994 + **Tape & Können 1999** (the prior-art check below).

## 8-A — Column strata classification — LANDED 2026-06-07
Corank computed from the 2×2 Jacobian SVD over the `(γ,α)` torus; on the caustic the smaller singular
value `s2/scale → 0` (the fold) while the larger `s1/scale` distinguishes corank-1 (bounded away from 0)
from corank-2 (`s1` also → 0). Threshold: corank-2 flagged if `min(s1)/scale < 0.05` (boundary eroded 3
cells to exclude wing-tip artifacts).

| stratum | feature | corank | `s1/scale` on caustic | label |
| --- | --- | ---: | ---: | --- |
| **A₂ fold** | 22°/46° edges + tangent-arc folds (all h) | 1 | 0.14–0.40 | generic caustic |
| **A₃ point-cusp** | the **UTA/LTA apexes** (top/bottom, 8-B locator) | 1 | — | Whitney cusp, all h |
| **A₃-class metamorphosis** | the **29.7° UTA+LTA→circumscribed merge** | 1 | **0.26** | two A₂ folds coalesce |
| A₄ / D₄ | — none on the column — | — | — | honest null |

- **OPEN QUESTION CLOSED (PHASE65 §6.5-B), label SHARPENED by 8-B:** the 29.7° merge is **corank-1, NOT a
  D₄ umbilic** (`s1/scale = 0.26`, cleanly bounded away from 0) — so it is an A_k cuspoid, not an umbilic.
  **8-B's cusp locator sharpens *which* A₃ object it is:** the merge is a **caustic METAMORPHOSIS** (the
  two UTA/LTA arc components reconnect as the gap of 6.5-B closes), the codim-2 topology change of the
  elevation family — *A₃-class* (two A₂ folds coalesce, a Whitney event valid for any smooth 2-D map, so
  it survives Berry's non-gradient objection), but **not a point-cusp** (the locator finds no cusp at the
  side reconnection). The persistent **A₃ point-cusps are the UTA/LTA apexes** (top/bottom). The derived
  29.7° number is unaffected; only the label is refined (metamorphosis vs apex point-cusps). Consistent
  with Berry 1994 ("some halos do show cusps").
- **Honest null:** corank-1 **everywhere** on the column (min `s1/scale = 0.062` at low sun — a
  wing-tip-near-admissibility-boundary closest-approach, still corank-1). The 2-DOF→2-sky square map
  exposes **only corank-1 (A_k) strata**; **D₄ needs ≥2 control DOF** (the elevation × habit grid of
  8-B). This is the expected, honest result — not a failure.
- **Derive-not-assert verified:** the Jacobian/singular values recompute from n (scale 0.467→0.489 as
  n 1.31→1.40) while the corank label correctly stays structurally 1.

## 8-B — the swallowtail search (column LANDED 2026-06-07; other habits staged)
Added a **cusp locator** (`cusp_field`/`cusp_count` in `atlas_strata_map.py`): on the caustic the A₃
cusps are where the kernel direction `K` (small-singular-value eigenvector of `J`) is **tangent** to the
caustic, i.e. `g := K·∇(det J) = 0` (a fold has `K` transverse, `g≠0`). Cusps = `{det J = 0} ∩ {g = 0}`.

**Column result — and it sharpens the 8-A label:**
- **The A₃ point-cusps are the UTA/LTA APEXES** — exactly **2**, at (δ≈21.3°, ψ=0° top) and (21.3°,
  180° bottom), **stable across all h ≥ 22°**. These (not the merge) are the rigorously-located A₃ cusps.
- **LABEL SHARPENED (corrects 8-A / 6.5-B):** the **29.7° UTA+LTA merge is a caustic METAMORPHOSIS** —
  the two arc components reconnect (their admissibility-bounded wing-tips meet, closing the 6.5-B gap) as
  the sun-elevation control varies. It is the codim-2 **topology change** (A₃-*class*), **not a point-
  cusp** (the cusp locator finds NO cusp at the side reconnection). The **derived 29.7° number is
  unaffected and correct**; only the catastrophe *label* is refined: metamorphosis (merge) vs the
  persistent point-cusps (apexes).
- **NO A₄ SWALLOWTAIL — confirms Berry 1994.** The cusp count is stable at 2 across the robust regime
  (no pair born/annihilated → no A₄ event). The apparent low-sun (h<22°) proliferation (11–19 cusps) is a
  **numerical artifact**: grid-dependent (h=18 gives 11/13/19 at ngrid 240/300/400; h=20 gives 4/26/17),
  the signature of a fragmented caustic near the admissibility boundary — excluded. This **computationally
  confirms Berry's observation** that the swallowtail is "conspicuously absent from… numerous halo
  simulations" (for the column habit). Honest null, the intended Berry-engagement.

**90°-WEDGE FAMILY (46° / supralateral / infralateral arcs) — LANDED 2026-06-07.** `cm.sky_grid` gained
a `wedge` param: `'basal90'` swaps the exit face for a **basal (end) face** (normal = `c`, ⟂ the prism
side faces → a **90° refracting wedge**), keeping the **same 2 orientation DOF (γ,α)** — so the cusp
locator + corank classifier run unchanged on the second 2-DOF caustic family. Result:
- **The caustic is the 46° family** — cusps at δ≈47°→58° (h 10°→28°), i.e. the supralateral / 46°-tangent
  arcs (min-deviation 45.7° for the 90° wedge, n=1.31).
- **corank-1 throughout** (`s1/scale ≈ 0.69–0.82 ≫ 0.05`) → **no D₄ umbilic** on this family either.
- **2 cusps — a ψ-symmetric pair at the sides** (ψ≈±75–85°, the lateral-arc cusps), **stable across
  grids** (240/300/400) for h≲28°. **No A₄ swallowtail:** the cusp count never changes. The caustic
  **vanishes off-sky near h~30°** (the supralateral-arc elevation limit — a **component-B admissibility
  wall**, like the CZA at 32°, *not* a cusp-pair annihilation). (Robustness: the labeler can split one
  cusp into adjacent cells → spurious ψ-asymmetric odd counts; `cusp_count` merges centroids within 4° to
  fix it.)
- **Both 2-DOF column families (60°-wedge tangent arcs + 90°-wedge 46° arcs) now CONFIRM Berry 1994** — no
  swallowtail, no umbilic. The swallowtail search is **complete for the column habit**.

**STILL STAGED:** **plate** (parhelia/CZA) and **Parry** are **1-DOF** (azimuth only) — their caustics are
**folds only** (no cusps in a 1-DOF map), so they get a fold-classification, *not* a swallowtail search.
The remaining genuine 2-DOF case is **pyramidal** (Tape AH-CH10/SAX-CH11 odd-wedge families). **The A₄
target needs the 3-jet check** (`∂δ=∂²δ=∂³δ=0`, `∂⁴δ≠0`) where the cusp-count method is ambiguous; the
**D₄ search EXPECTS NONE** (Berry; the anthelic-X is generically two A₂ folds). Every higher stratum →
Gate-2 catalog cross-check; any bucket-(ii) candidate is the only "prediction," capped P1/P2 internal.
Expected: A₄ absent / coincident with a named locus; D₄ confirmed-absent → contribution = the systematic
stratification + the engagement of Berry's open questions, not a new halo.

## Lit-pass Track B — RESOLVED 2026-06-07
**Tape & Können 1999 (Appl. Opt. 38:1552, full 74-pp text read): pure parameterization, no caustic/
catastrophe/bifurcation classification** (it dissolves the merge into "one halo"). **Berry 1994 (Appl.
Opt. 33:4563): the decisive prior art** — raised catastrophe-typing of ice halos, recorded cusps occur,
flagged the **non-gradient** map (Thom taxonomy doesn't transfer wholesale), predicted **no umbilics**,
left the **swallowtail open**. Verdict: **novelty = PARTIAL** (the systematic *stratification* taking up
Berry's question, addressing non-gradient — defensible); NOT "first catastrophe classification" (Berry
pre-empts the idea, Tape the cusp). Mandatory: cite Berry 1994 + engage the non-gradient objection;
A_n/D_n labels need the jet-determinacy check, not corank alone; the no-D₄ result confirms Berry.

## Status
8-A + the full 8-B column habit (BOTH 2-DOF wedge families) are clean, defensible components: the
catastrophe-stratum corank is now a **computed** property of the halo caustic (the §6 armchair gate
cleared), the **A₃-vs-D₄ question is closed (corank-1, A₃-class)**, the cusp locator places the **A₃
point-cusps** (apexes for the 60°-wedge tangent arcs; lateral pairs for the 90°-wedge 46° arcs) and the
**29.7° merge as the A₃-class metamorphosis**, and the **A₄ swallowtail search is a clean NULL on both
families** (cusp counts stable — confirms Berry 1994), **no D₄** anywhere. 8-B's remaining leg = the
**pyramidal** habit (the other genuine 2-DOF case; plate/Parry are 1-DOF → folds only, a fold-
classification not a swallowtail search) + the optional A₄ 3-jet refinement. **NOT public-eligible**
(Phase 0.5 lit-pass, incl. the Tape & Können 1999 prior-art check, gates any claim).
