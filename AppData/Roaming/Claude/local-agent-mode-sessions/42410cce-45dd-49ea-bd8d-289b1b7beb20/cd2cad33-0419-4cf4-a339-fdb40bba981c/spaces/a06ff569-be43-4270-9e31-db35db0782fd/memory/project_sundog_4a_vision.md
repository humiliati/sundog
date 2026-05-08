---
name: Sundog 4a hero vision
description: Triple-sun parhelion "eye" composition for the index hero — nine-ring halo intersection, BoxForge rainbow orbs for suns, CSS/SVG for halos, 4-phase animation with idle window for embellishment
type: project
---

Sundog Migration Step 4a (index hero) is owner-codified 2026-05-08 as a "sundog eye" composition driven by a real halo-display photograph the owner shared.

**Geometry:**
- Three suns total: 1 primary + 2 virtual peripheral suns (sourced from peripheral-arc tips)
- Three halos per sun → 9 intersecting rings
- Composition reads as an eye: pupil = primary sun, iris = stacked halos around it, eyelids = peripheral arcs at top of frame
- Golden-ratio placement of the three suns is iterative — take the passes necessary

**Color rule:** misty translucent rainbow with meaningful overlaps. Rainbow is physical (ice-crystal dispersion physics: red-inside, blue-outside on each ring), not decoration. Stacked alpha so overlaps read as additive light, not muddied.

**Tooling split:**
- Suns → BoxForge rainbow-orb primitive (verify the BoxForge library actually has this; owner mentioned it exists)
- Halos → stacked CSS `radial-gradient` + optional SVG `feGaussianBlur`/`feColorMatrix` for dispersive edges. Explicitly NOT BoxForge — halos are atmosphere, not boxes
- Arcs (iris layers / eyelids) → SVG paths with stroke gradients

**Four-phase animation (BoxForge phase discipline: idle / hover / active / handoff·settle):**
- Phases 1–3 = active "expressing the math" — sequenced reveal of parhelion geometry
- Phase 4 = idle window — golden static composition holds, subtle scintillation only. Embellishment work (color refinement, layer ordering, edge dispersion tuning) happens HERE, before 3D animation muddies the geometry
- Phase 5+ = 3D handoff — BoxForge phase animation drives orbs into dimensional motion; halos stay planar (no 3D treatment, they're sky)

**Why:** Owner wants the static composition to do real visual work — not be a backdrop for animation. Phase 4's idle window is the design moment that earns the rest. Documented in `docs/UI_UX_THEME_FOUNDATION.md` Step 4a.

**How to apply:**
- Don't conflate halos with BoxForge — they need separate techniques. Halo "improvements" without a BoxForge analog is intentional, not a vision gap
- 4b (threebody) stays formal — no 3D, just frame/palette upgrades
- 4c (balance) is the exuberant outlier — Paper Mario 2.5D parallax with pixel-art cart, polished mobile/indie game feel; deliberate aesthetic departure
