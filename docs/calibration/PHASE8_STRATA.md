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
>   are the generic stable singularities of *any* smooth 2-D map (gradient or not). So the 29.7° merge =
>   A₃ Whitney cusp is valid (and Berry confirms cusps occur). It is the *higher* strata where
>   gradient-vs-non-gradient bites.
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
| **A₃ cusp** | the **29.7° UTA+LTA→circumscribed merge** | 1 | **0.26** | two A₂ folds coalesce |
| A₄ / D₄ | — none on the column — | — | — | honest null |

- **OPEN QUESTION CLOSED (PHASE65 §6.5-B):** the 29.7° merge is **corank-1 → an A₃ (Whitney) cusp, NOT a
  D₄ umbilic** (`s1/scale = 0.26`, cleanly bounded away from 0). The two UTA/LTA fold branches coalesce
  (the gap-closure of 6.5-B) — the A₃ identification rests on the **two-fold-coalescence topology** (a
  Whitney cusp, valid for any smooth 2-D map per Whitney's theorem, gradient or not — so it survives
  Berry's non-gradient objection), not on corank alone (`corank-1` does not by itself separate A₃ from
  A₄). Consistent with Berry 1994 ("some halos do show cusps").
- **Honest null:** corank-1 **everywhere** on the column (min `s1/scale = 0.062` at low sun — a
  wing-tip-near-admissibility-boundary closest-approach, still corank-1). The 2-DOF→2-sky square map
  exposes **only corank-1 (A_k) strata**; **D₄ needs ≥2 control DOF** (the elevation × habit grid of
  8-B). This is the expected, honest result — not a failure.
- **Derive-not-assert verified:** the Jacobian/singular values recompute from n (scale 0.467→0.489 as
  n 1.31→1.40) while the corank label correctly stays structurally 1.

## 8-B — other habits + the SWALLOWTAIL search (STAGED; reframed by the Berry prior-art)
Extend `atlas_strata_map.py` (reusing the Snell kernel + `sky_grid`) to: **plate** (1-DOF azimuthal),
**Parry** (2-DOF tilt+azimuth), **pyramidal** (discrete wedge families, Tape AH-CH10/SAX-CH11). Run the
corank classifier across the `(elevation × habit)` grid. **The genuine target is the SWALLOWTAIL (A₄) —
Berry 1994's explicit open question** (stable in non-gradient R³→R³ maps, yet "conspicuously absent from
numerous halo simulations"). A₄ needs the 3-jet check (`∂δ=∂²δ=∂³δ=0` with `∂⁴δ≠0`), not just corank.
**The umbilic (D₄) search EXPECTS NONE** — Berry predicts no umbilics in the non-gradient halo map, so a
flagged corank-2 point is most likely a boundary artifact (or would need extraordinary justification
against Berry); the anthelic-point X-crossing is *generically two A₂ folds*, not an umbilic. Frame the
D₄ search as **testing/confirming Berry's no-umbilic prediction**, not finding one. Every higher stratum
→ Gate-2 catalog cross-check; any bucket-(ii) candidate is the only "prediction," capped P1/P2 internal.
Expected: A₄ absent or coincident with a named locus; D₄ confirmed-absent → contribution = the
systematic stratification + the engagement of Berry's open questions, not a new halo.

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
8-A is a clean, defensible component: the catastrophe-stratum corank is now a **computed** property of
the halo caustic (the §6 armchair gate cleared for the column), the **A₃-vs-D₄ question is closed (A₃)**,
and the column carries no higher strata (honest null). 8-B (other habits + the umbilic candidate) is the
remaining directed search. **NOT public-eligible** (Phase 0.5 lit-pass, incl. the Tape & Können 1999
prior-art check, gates any claim).
