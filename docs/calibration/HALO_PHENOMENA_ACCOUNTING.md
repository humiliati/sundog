# Halo Phenomena Accounting Matrix

Filed: 2026-05-14  
Phase: Sundog V Geometry Phase 14  
Status: initiated

## Purpose

This is the Phase 14 ledger for the halo vocabulary around the Sundog atlas.
It exists to keep four things separate:

- what the public atlas renders,
- what the project has calibrated against photographs,
- what the literature or HaloSim names but Sundog does not yet model, and
- what the project might later prove, simulate, or reject.

A halo phenomenon is **not accounted for** just because its name appears on a
page. For Phase 14, "accounted for" means the row names the source, rough
geometry or crystal family, Sundog rendering status, evidence status, public
claim boundary, and next proof step.

## Status Vocabulary

| status | meaning |
| --- | --- |
| `rendered-core` | Rendered in the public atlas as part of the main workbench vocabulary. |
| `rendered-optional` | Rendered or named in the atlas as optional vocabulary, but not promoted as a core inverse handle. |
| `named-only` | Named for vocabulary completeness; not drawn as a separate Sundog primitive. |
| `not-modeled` | Outside the current atlas generator. |
| `halosim-reproducible` | Has or should have a HaloSim recipe/render receipt. |
| `analytic-candidate` | Has a plausible closed-form or brute-force proof path, but no accepted project receipt yet. |
| `speculative` | Proposed or unphotographed; must stay out of public proof language until promoted. |
| `observed` | Photographic or historical observation exists outside this project. |
| `rejected` | Tested and failed under the current protocol. |

## Evidence Vocabulary

| evidence | meaning |
| --- | --- |
| `calibrated-photo` | Measured against at least one calibration photograph with a documented residual. |
| `calibrated-inverse` | Promoted or evaluated as a route for recovering hidden state such as sun altitude. |
| `photo-partial` | Visually present or partially measured, but not promoted. |
| `literature` | Named in Greenler, Tape, Cowley, or equivalent atmospheric-optics reference material. |
| `halosim-candidate` | Known or suspected HaloSim asset exists; receipt still needs capture. |
| `unvalidated` | Named, but not yet grounded by this project's evidence chain. |

## Current Accounting Matrix

| phenomenon / family | aliases | current status | evidence | source basis | Sundog boundary | next Phase 14/15 step |
| --- | --- | --- | --- | --- | --- | --- |
| 22 deg halo | small halo | `rendered-core` | `calibrated-photo`, `literature` | Greenler / Tape / Cowley; scale-lock primitive | Public atlas primitive and scale reference; not an inverse handle by itself. | Keep as scale-lock anchor for future rows. |
| 46 deg halo | large halo | `rendered-core` | `photo-partial`, `literature` | Literature formula and post-A1b WB_R46 correction | Rendered vocabulary; CZA/supralateral context, not a promoted inverse route. | Add HaloSim geometry receipt and clarify tangent/crossing language per public legend. |
| Sundog / parhelion | parhelion, dagger | `rendered-core` | `calibrated-inverse`, `calibrated-photo` | Parhelion offset formula `offset = R22 / cos(h)` | Sole promoted inverse handle after Phase 10 audit: strict eligible set p2, p7, p13. | Preserve as promoted handle; expand only after new eligible photo anchors land. |
| Parhelic circle | horizontal belt | `rendered-core` | `photo-partial`, `literature` | Horizontal circle through the sun/parhelia | Rendered belt primitive; not itself a hidden-state proof. | Track belt-y residuals as primitive QA, not a demotion trigger unless thresholds fail. |
| Circumzenithal arc | CZA | `rendered-core` | `photo-partial`, `literature` | Literature formula from Pass A1a/A1b; cutoff h = 32.196 deg | Conditional visible vocabulary; CZA-apex inverse route is coverage-gated. | Re-anchor any pre-A1b CZA examples before using residuals in public-facing proof. |
| Supralateral arc | supralateral | `rendered-optional` | `photo-partial`, `literature` | 46 deg halo adjacency / high-arc literature | Optional vocabulary; inverse route failed coverage and structural-discrimination gates. | Find additional photos or HaloSim receipts that separate it from CZA/46 deg halo overlap. |
| Upper tangent arc | UTA | `rendered-optional` | `photo-partial`, `literature` | 22 deg halo tangent-family vocabulary; C5/C6/C7 audit tension | Logo/animation vocabulary only; not promoted under circle-fit or canonical opening-angle handle. | Specialist re-anchoring of p2 before any stronger tangent claim. |
| Suncave Parry arc | Parry cap | `rendered-optional` | `photo-partial`, `literature` | Parry-oriented column-crystal family | Weak evidence; optional atlas label, not promoted. | Move into Parry-family submatrix with HaloSim receipts. |
| Lower tangent arc | LTA | `rendered-optional` | `photo-partial`, `literature` | Tangent-family counterpart at 22 deg halo bottom | Single-photo/pose vocabulary; not core geometry. | Tie to low-altitude named poses and validate with HaloSim before promotion. |
| Parry supralateral arc | Parry supralateral | `rendered-optional` | `photo-partial`, `literature` | Parry-family / 46 deg halo top vocabulary | Coverage plus weak-evidence failure; optional vocabulary only. | Needs clearer photo or HaloSim separation from supralateral/46 deg halo. |
| Infralateral arcs | infralateral | `rendered-optional` | `photo-partial`, `literature` | 46 deg lower-side family | Periphery vocabulary; not an inverse route. | Add low-altitude/46 deg HaloSim receipt and note observer geometry. |
| Sun pillar | light pillar | `rendered-core` | `photo-partial`, `literature` | Plate-crystal reflection vocabulary | Rendered visual primitive; atlas construction is stylized, not a full crystal physics proof. | Mark exact generator boundary in the public legend. |
| Parry-family arcs | sunvex Parry, suncave Parry, Parry supralateral | `named-only`, `rendered-optional` | `literature`, `halosim-candidate` | Tape; HaloSim Parry assets; Cowley | Family name is broader than the rendered optional arcs. | Create subrows by named Parry member and assign HaloSim proof receipts. |
| Pyramidal / odd-radius halos | 9 deg, 18 deg, 20 deg, 23 deg, 24 deg, 35 deg halos | `named-only`, `not-modeled` | `literature`, `halosim-candidate` | Tape ch. 10; Tape & Moilanen; HaloSim pyramidal assets | Not modeled in the current atlas; do not imply Sundog covers all halo radii. | Phase 15 analytic/ray-trace candidate set. |
| Lowitz arcs | upper/middle/lower Lowitz | `named-only`, `not-modeled` | `literature`, `halosim-candidate`, `observed` | Lowitz orientation; historical/modern photo literature | Not modeled; should not be folded into sundog/parhelion geometry. | Add orientation row and HaloSim receipt before any public graphic. |
| Antisolar features | anthelion, anthelic arcs, paranthelia, 120 deg parhelia | `named-only`, `not-modeled` | `literature`, `halosim-candidate` | Tape; Cowley; HaloSim anthelic assets | Outside current front-facing atlas field. | Decide whether public legend should show rear-sky vocabulary or keep it as docs-only. |
| Sub-horizon halos | subsun, subparhelia | `named-only`, `not-modeled` | `literature`, `halosim-candidate` | Greenler / Tape / Cowley | Requires aircraft/mountain observer geometry; not part of the default ground-observer atlas. | Treat as observer-geometry extension. |
| Circumhorizon arc | CHA, fire rainbow | `named-only`, `not-modeled` | `literature`, `observed` | High-sun plate-crystal family | Not a sundog inverse handle; high-sun regime outside current explainer. | Add a high-sun observer row if Phase 15 broadens beyond sundog-facing optics. |

## Phase 14 Work Queue

1. **14A - Ledger seed:** land this file and keep it as the canonical tracking
   surface for vocabulary status.
2. **14B - Public legend:** publish a lightweight `legend.html` dictionary so
   the atlas can point to the broader vocabulary without burying
   `sundog.html`.
3. **14C - Machine-readable mirror:** add `public/data/halo-phenomena-status.json`
   only when the public UI needs dynamic filtering or Ask Sundog needs the
   same rows. This is intentionally deferred until promotion pressure exists.
4. **14D - Source pass:** attach exact Greenler / Tape / Cowley / HaloSim
   source receipts to every named-only family.
5. **14E - HaloSim receipt pass:** for every `halosim-candidate`, capture at
   least one render or mark the row as not reproducible under the local
   HaloSim library.
6. **14F - Proof routing:** move analytic or speculative rows into Phase 15
   only after their source and geometry fields are filled.

## Gate

Phase 14 closes only when the public legend and this ledger agree row by row,
and no public or chatbot surface can reasonably be read as "Sundog has accounted
for all halo geometry" without distinguishing rendered, optional, named-only,
not-modeled, and speculative entries.
