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

**REMAINING GENUINE 2-DOF TARGETS** (a 30-agent enumeration + adversarial DOF-verification, 2026-06-07;
**corrects the earlier "pyramidal is the only/last 2-DOF case"**). The cusp/swallowtail search needs a 2-D
orientation manifold (`F: T²→sky`, a square map); THREE single-crystal targets remain, ranked:
1. **LOWITZ orientation — DONE 2026-06-07 → A₃-lips, confirms Berry (see §8-C).** `wedge='lowitz60'`, a
   geometrically distinct SO(3) square map (c tilts out of horizontal). It produced the search's **first
   cusp-CREATION** (a 2→4→2 interior pair-birth at h≈16.5) — a genuine codim-1 higher catastrophe — but the
   topological test shows it is an **A₃-lips metamorphosis, NOT Berry's A₄ swallowtail**, and it is
   parameterization-specific (α0≈60° only; plate-Lowitz negative). **Confirms Berry's no-A₄, no D₄.**
2. **Pyramidal-capped horizontal column — DONE 2026-06-07 → confirms Berry (§8-D).** `wedge='pyrcol'`,
   same (γ,α) column manifold, exit = pyramid {10-11} cap (x=62° from c); validated at the 9°/23.8° odd
   radii. The 9° arc is a clean flat ψ-symmetric 2-cusp pair; the 23.8° arc has no grid-stable interior
   A₄. **No A₄, no D₄.**
3. **Wegener-arc ray path.** Same column torus, **new path** (one internal reflection off a basal face);
   a workflow agent reported a *grid-stable cusp-count change 4→5* = a **CANDIDATE A₄ signature** —
   **UNVERIFIED** (needs the eroded robust-regime + 3-jet audit; may be an admissibility-boundary cusp,
   not a true A₄). The one column path whose A₄ is not a foregone null.

**NOT targets (corrected):** **RANDOM pyramidal** = the odd-radius CIRCULAR halos (9/18/20/23/24/35°) = a
**3-DOF RING** family (A₂ fold circles, the pyramidal analog of the 22°/46° halos), **NOT** a cusp target;
oriented pyramidal **plates / Parry** = 1-DOF folds. **Plate & Parry confirmed 1-DOF** (the Parry
roll-wobble is a *smearing* of the 1-DOF fold curve — 14–24° short of the column's cusp rolls — not a 2nd
DOF). **D₄ expected absent GENERICALLY** (Berry; the anthelic X-crossing = two transverse A₂ folds from
*distinct* populations, NOT a single-map D₄ umbilic — the anthelic point is not a D₄ candidate). **The A₄
target needs the 3-jet check** (`∂δ=∂²δ=∂³δ=0`, `∂⁴δ≠0`) where the cusp-count is ambiguous. Every higher
stratum → Gate-2 catalog cross-check; bucket-(ii) candidate capped P1/P2 internal. Expected: A₄ absent /
coincident with a named locus; D₄ confirmed-absent → contribution = the systematic stratification.

## 8-C — Lowitz + Wegener builds (2026-06-07)
Two of the three remaining 2-DOF targets were built (`sky_grid` gained `wedge='wegener'` and
`wedge='lowitz60'`) and searched.

**WEGENER (`wedge='wegener'`) — SEARCHED → candidate-A₄ REFUTED, confirms Berry.** Entry prism side →
internal TIR reflection off a basal face → exit other side face (same (γ,α) column torus). A prior
workflow agent had flagged a "grid-stable cusp-count change = candidate A₄." Reproduced with our own code
and **refuted it** by the same audit that the column passed:
- The extra cusp is a **single, on-meridian (ψ=0) cusp sitting AT the admissibility wall** —
  boundary-distance **1.0–2.6 cells** (vs the genuine Wegener cusps at bd 13–20, deep interior) — so it is
  a **component-B admissibility cusp** (the caustic meeting the basal-TIR wall), NOT an intrinsic A₄.
- The **corank-2 flag at h≈8–11 is marginal** (`s1/scale` floors at 0.024 then recovers; never the clean
  `s1→0` of a true umbilic) — a low-sun boundary effect, **not a D₄**.
- The genuine inventory is **6 interior A₃ cusps** (2 apex at 22° ⊕ 4 anthelic at 88°, ψ-symmetric),
  corank-1, **no A₄, no D₄ → confirms Berry** (the third 2-DOF column family to do so).
- Lesson: the corank-2 threshold (0.05) over-flags near a TIR boundary; the decisive test is
  interior-survival + clean `s1→0`, not the threshold alone.

**LOWITZ (`wedge='lowitz60'`) — CHARACTERIZED 2026-06-07 → an A₃-class cusp-pair-creation (lips-type),
DECISIVELY NOT Berry's A₄ swallowtail. BANKED.** (9-agent workflow `w6pwapt6l`: 4 characterization lines +
4 adversarial re-runs + synthesis; load-bearing facts independently reproduced.) The column frame rotated
by φ about the horizontal a-axis u (so the c-axis tilts out of horizontal, `c_z=sin φ` — a geometrically
DISTINCT 2-surface in SO(3); φ=0 reduces to `_column_normals` exactly). 2 DOF (γ, φ); roll fixed at
`LOWITZ_ALPHA0`. The caustic is the Lowitz arcs (δ≈23°, ψ≈±84°, the parhelia). **It is the FIRST non-null
cusp-CREATION in the whole halo search** (the column is flat at 2) — a genuine codim-1 higher catastrophe —
**but A₃, not A₄ → it CONFIRMS Berry's "swallowtail conspicuously absent," it does not answer his question.**
- **The event:** interior cusp count **2 (h≤16) → 4 (17–30) → 2 (h≥31)** (ng=320, bd=5); the born pair at
  δ≈32–33°, ψ≈±173° (straddling the bottom meridian), **deep interior** (boundary-distance ≈27 cells at
  h=18, scaling with grid → genuine, not a fixed-pixel artifact); **corank-1 throughout** (`s1/scale`≈0.63
  ≫ 0.05, `s2/scale`≈0.01 → **no D₄**). Column `prism60` flat at 2 in the same window.
- **Decisively NOT A₄ (the load-bearing call):** the image caustic stays **single-valued / non-self-
  intersecting through the entire birth window** (h=16.2–28; confirmed 3 coordinate-free ways: ψ-bin
  δ-multiplicity=1, segment-crossing=0, ng=1000 sheet-count=1) — a swallowtail *is* a self-crossing
  caustic. The **fold set stays ONE connected component** (no reconnection → not beak-to-beak, no A₄ oval).
  The pair nucleates symmetrically at the bottom-meridian mirror point (γ=90°, φ=180°, detJ=0 ∀h by the Z₂
  symmetry) and separates as **Δγ ~ √(h−h\*)**, h\*≈16.15 — the codim-1 A₃ normal-form opening.
- **Caveats (weighed heavily — same bar that refuted Wegener):** (1) **α0-specificity (corrects the earlier
  "α0=30/60/105"):** the clean 2→4→2 birth occurs **only at α0≈60°**, the **NON-canonical FACE-aligned**
  rotation axis; the canonical edge-Lowitz rolls FAIL (α0=90° flat at 2 like the column, α0=30° grid-
  unstable). Refined to a finite ~8° band on the m-axis by the rigor upgrade below. (2) **Doubly
  non-canonical:** column-Lowitz frame (c horizontal) + face axis; the textbook Lowitz crystal is the
  c-vertical PLATE. (3) **Plate-Lowitz cross-check NEGATIVE** (`scripts/plate_lowitz_check.py`): the classic
  c-vertical plate-Lowitz does NOT reproduce it (cusp count flat at 5, all at parhelion flanks, none on the
  bottom meridian) → **parameterization-specific, not habit-robust.** (4) The **"death" half is a grid-
  unstable boundary exit** — only the BIRTH carries catastrophe weight. (5) The prior reduced-jet A₄
  discriminator was retracted as unsound — **now superseded by a sound test (rigor upgrade below).**
- **RIGOR UPGRADE (2026-06-07) — a SOUND, CALIBRATED jet test independently confirms A₃-lips, and the
  α0-band is mapped.** (`scripts/atlas_jet_classify.py` + `test_atlas_jet_classify.py`.)
  - **The jet classifier rebuilt soundly:** the Morin/Thom-Boardman iterated-kernel discriminant
    `c₃ = K·∇(K·∇ det DF)` with the **smooth, sign-consistent row-kernel `K=(Xα,−Xγ)`** (the prior unsound
    test used the sign-ambiguous SVD eigenvector). A₄ ⟺ `c₃→0`; A₃ ⟺ `c₃` bounded away from 0.
  - **Calibrated on BOTH controls:** (+) the **Morin A₄ normal form** `F_h=(γ, α⁴+hα²+γα)` — `|c₃|` DIVES to
    0 as the two cusps merge into the A₄ (ratio 1.00→0.16, the √(−h) law) → the test *registers* an A₄;
    (−) the **column apex A₃ cusps** — `|c₃|≈9` bounded away from 0 → registers A₃.
  - **Lowitz verdict:** at the birth (h→16.5⁺) `|c₃|≈8` (ratio **0.82** of generic), comparable to the
    column-A₃ control and **NOT diving** — where a true A₄ gives ratio <0.25. So **the sound jet test
    independently confirms A₃-lips, agreeing with the topological self-intersection test** (two independent
    methods concur; the methodological gap is closed).
  - **α0-band mapped:** the clean 2→4 birth is a **finite ~8° band (α0≈58–66°)**, not a razor-thin value
    (good for a population spread) — but it sits on the **m-axis (face-aligned) rotation**, the boundary
    between a 2-cusp and a 4-cusp α0-regime. The **canonical Lowitz axis is the a-axis through opposite
    edges (α0=90°, which shows NO birth)**. So the metamorphosis requires the *non-canonical* m-axis
    rotation; whether a real Lowitz population realizes that orientation is the one **residual, honest
    caveat** (the rest is now closed). The catastrophe ORDER (A₃, not A₄) is unaffected.
- **Lit:** the Lowitz-arc transformation in the 15–30° window is documented QUALITATIVELY (Mueller & Greenler
  JOSA 69:1103 1979; Tape; Riikonen; Cowley/atoptics) but **never in cusp/lips/catastrophe terms** — the
  catastrophe LABELING is uncatalogued, the underlying arc physics known. **Berry's A₄ remains open.**

## 8-D — Pyramidal-capped column (the last enumerated 2-DOF target) — 2026-06-07 → confirms Berry
`sky_grid` gained `wedge='pyrcol'`: the singly-oriented horizontal column (same (γ,α) 2-torus) but the
EXIT face is a **pyramid {10-11} cap face** — normal at `PYR_X` = **61.99°** from the c-axis (computed from
ice's c/a=1.628, the textbook ice pyramid angle) and azimuthal facet offset `PYR_DPHI`. The odd "Galle"
wedge: `n2 = cos(x)·c + sin(x)·(cos(α+dφ)·u + sin(α+dφ)·w)`, giving `n1·n2 = sin(x)·cos(dφ)` = a constant
wedge over the whole (γ,α) sweep. **Validated against the documented odd radii:** dφ=120° → caustic
min-deviation **23.82°** (the 23° halo); dφ=180° → **8.96°** (the 9° halo); the other facet offsets are
correctly TIR-blocked. **corank-1 throughout → no D₄.**
- **9° arc (dφ=180°, the ψ-symmetric facet):** a clean **2-cusp ψ-symmetric pair at (δ≈9°, ψ≈±88°), FLAT
  across h=8–35°** (merge-corrected `cusp_count`), corank-1. **No A₄, no D₄.** The cleanest test.
- **23.8° arc (dφ=120°):** a single pyramid facet breaks the ψ→−ψ symmetry → a **one-sided** arc (cusps at
  ψ≈+150° only; the mirror is the dφ=240° facet). It carries **~1 robust deep-interior A₃ cusp per facet**
  (symmetric union ≈ 2); the apparent 1↔2 flicker is **grid-unstable cluster-splitting** (~1° separation),
  not a pair-birth, and the arc **vanishes off-sky at h≥28°** (a component-B admissibility wall). **No
  grid-stable interior A₄** (contrast Lowitz's clean 2→4→2 across every grid).
- **Verdict: confirms Berry** — no A₄ swallowtail, no D₄ umbilic, on both the 9° and 23.8° oriented
  odd-radius arcs.

**THE 2-DOF SWEEP IS COMPLETE.** Every enumerated genuine 2-DOF single-crystal map — column 60° (tangent
arcs), column 90° (46°/supralateral), Wegener (anthelic), Lowitz, pyramidal-capped column (9°/23.8°
odd-radius) — has been searched. **All confirm Berry 1994: no A₄ swallowtail, no D₄ umbilic.** The single
non-trivial higher catastrophe found anywhere is the **Lowitz A₃-lips** (the search's first cusp-creation,
§8-C) — and it is A₃, not A₄. Berry's "swallowtail conspicuously absent from halo simulations" is now
**computationally confirmed across the full 2-DOF target set** (and two candidate-A₄ false alarms —
Wegener's and the initial Lowitz read — were refuted by code).

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
families** (cusp counts stable — confirms Berry 1994), **no D₄** anywhere. Two further 2-DOF maps are now
searched (§8-C): **Wegener** (candidate-A₄ refuted — a component-B wall cusp) and **Lowitz** (the search's
first cusp-CREATION, but an **A₃-lips metamorphosis, not Berry's A₄** — parameterization-specific). **Every
2-DOF map searched so far CONFIRMS Berry: no A₄ swallowtail, no D₄ umbilic** — and the lab found + labelled
its first higher-catastrophe (the A₃-lips) and refuted its own two candidate-A₄ false alarms by code. **The
2-DOF target sweep is now COMPLETE (§8-D — the pyramidal-capped column also confirms Berry): all five
enumerated 2-DOF single-crystal maps show no A₄ swallowtail and no D₄ umbilic.** **NOT public-eligible**
(Phase 0.5 lit-pass, incl. the Tape & Können 1999 prior-art check, gates any claim).

## 8-E — the corank-2 / D₄ detector CALIBRATED (the never-fired branch, 2026-06-07)
A self-audit (the lab refereeing its own null): the "no D₄ anywhere" verdict above rested on a corank-2
detector (`corank_on_caustic`, flag `s1_min_rel < 0.05`) that had **only ever returned negatives** — unlike
the A₄ jet test, its corank-2 branch was **never demonstrated to fire**. Now given the missing positive
control (`scripts/atlas_jet_classify.py` §4 + `test_atlas_jet_classify.py`):
- **POSITIVE — a synthetic hyperbolic-umbilic D₄** (`synthetic_umbilic`: the gradient map
  `∇(x³/3+y³/3+w·xy)=(x²+wy, y²+wx)`, J vanishes at the origin at w=0, `det=4xy` sign-changing). The
  chart-based detector (`corank_from_chart`) **FIRES — `s1_min_rel = 0.0036`, corank-2** — and as the umbilic
  **unfolds** (w: 0.05→0.4) it climbs cleanly back to corank-1 (0.059→0.44), the catastrophe-theory
  expectation. So the corank-2 branch is **not blind**, and 0.05 sits cleanly in the gap.
- **NEGATIVE — the A₄ swallowtail** (`synthetic_swallowtail`, a corank-1 cuspoid): `s1_min_rel = 0.81`,
  corank-1 — the detector **distinguishes a D₄ umbilic from an A₄ swallowtail** (and from the ice halos at
  `s1_min_rel ≈ 0.2–0.3`).
- **Step 1 verdict: the "no-D₄" null is UPGRADED from CONFIRMED to VALIDATED** — the tool genuinely *can*
  see a D₄ (it fires at 0.0036 on the synthetic normal form); ice halos demonstrably lack one.
- **Step 2 — the REAL physical D₄ (DONE 2026-06-07; `scripts/drop_umbilic.py` + `test_drop_umbilic.py`):**
  an independent-referee test against a *literature-established* physical umbilic. A genuine **oblate-raindrop
  primary-rainbow** forward chart (oblate spheroid semi-axes a,a,c; horizontal +x beam; enter → ONE internal
  reflection → exit; n=1.333) was traced and fed to `corank_from_chart`. Sweeping the aspect ratio D/H = a/c,
  the detector **fires corank-2 precisely in the band D/H ∈ [1.295, 1.320]** (grid-stable across ng
  256/320/400; min `s1_min_rel`≈0.016) — essentially identical to the **Marston & Trinh / Nye 1984
  hyperbolic-umbilic value D/H = 1.305 ± 0.016**. The **sphere (D/H=1) is corank-1** (`s1_min_rel`=0.93, the
  axisymmetric rainbow fold) and the **over-flattened drop (D/H=1.5) is corank-1** (the umbilic unfolds).
  (Confounds excluded: the backscattering-**glory** axial focus and the grazing silhouette edge — both
  corank-2 at *every* aspect ratio — are removed by restricting to the rainbow annulus + eroding the
  admissibility mask; `corank_from_chart` gained a `good` param.) **The raindrop D₄ at D/H=1.305 is Marston &
  Trinh / Nye 1984's (Nature 312:529 + 312:531); the lab independently reproduces it from ray optics — NOT a
  new-physics claim — validating the corank-2 detector on a real physical D₄.**
- **Net: the "no-D₄" null is VALIDATED on BOTH a synthetic normal form AND a real physical D₄.** The detector
  fires on (i) the synthetic hyperbolic umbilic (`s1_min_rel`=0.0036) and (ii) the Marston/Nye raindrop
  umbilic at D/H≈1.305, and correctly stays corank-1 on the sphere, the A₄ swallowtail, and all five
  ice-halo families. The tool genuinely sees D₄; ice halos genuinely lack one.
