# Halo Phenomena Accounting Matrix

Filed: 2026-05-14
Phase: Sundog V Geometry Phase 14
Status: 14A landed · 14D source pass landed · 14E receipt pass landed (11/11 reproduced) · 14F partially routed

## Purpose

This is the Phase 14 ledger for the halo vocabulary around the Sundog atlas.
It exists to keep four things separate:

- what the public atlas renders,
- what the project has calibrated against photographs,
- what the literature or HaloSim names but Sundog does not yet model, and
- what the project might later prove, simulate, or reject.

It also exists to correct a specific mistake. The project briefly believed its
own atlas / `mesa` work had *invented* geometry that, in fact, already exists
in W. Tape's *Atmospheric Halos*, the HaloSim `.sim` / `.xsh` / `.xng` asset
library, and Greenler / Cowley. The closed-form generator for essentially
every row below was sitting in plain text on disk the whole time. Phase 14
installs it.

A halo phenomenon is **not accounted for** just because its name appears on a
page. For Phase 14, "accounted for" means the row names the source, crystal
family / orientation, geometric generator, observer geometry, Sundog
rendering status, evidence status, public claim boundary, and next proof
step.

This ledger is organized into three lenses, matching the Phase 14 directive:

- **§A — Anchor: what we know.** The proven / calibrated core. Generator =
  the project's *own* validated closed form in
  [`public/js/parhelion-geometry.mjs`](../../public/js/parhelion-geometry.mjs),
  backed by a project receipt (Pass C7, Pass A1a/A1b, the parhelion offset
  law). Includes an honesty ratchet: rendered ≠ anchored.
- **§B — Install: what we should have known.** The canonical pre-existing
  generator for each family, read straight out of the Tape chapter scans and
  the HaloSim `.sim` recipes. This is the geometry the atlas placeholders
  must be reconciled against.
- **§C — Extrapolate: bounding the unknown.** For named-only / not-modeled /
  speculative families: what the installed geometry predicts, the candidate
  inverse handle (the project's indirect-inference method — recover hidden
  sun altitude / crystal population from an observed locus), the
  falsification criterion, and the current Phase 15 proof tier (P0–P5).

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

## Source key

- **Code** — [`public/js/parhelion-geometry.mjs`](../../public/js/parhelion-geometry.mjs).
- **Tape AH** — *Atmospheric Halos* (Antarctic Research Series, vol. 64),
  on-disk chapter scans: [`AH-CH06/`](AH-CH06) (sun-elevation role),
  [`AH-CH10/`](AH-CH10) (pyramidal / odd-radius).
- **Tape SAX** — *Halos: A Search for Angle X*, on-disk chapter scan:
  [`AH-SAX-CH11/`](AH-SAX-CH11).
- **HaloSim `.sim`** — bundled recipe library at `C:\Users\hughe\*.sim`
  (cited by file name; the recipe's description block states the ray path).
- **Receipts** — [`PASS_C7_OUTPUT.txt`](PASS_C7_OUTPUT.txt),
  [`HALOSIM_VALIDATION_PROTOCOL.md`](HALOSIM_VALIDATION_PROTOCOL.md).
- **Cowley** — atoptics atmospheric-optics reference (cited at root only; no
  guessed deep URLs, consistent with `sundog.html` §7).

> The underlying minimum-deviation refraction equation itself is in Tape
> *Atmospheric Halos* Ch. 4, which is **not on disk**. The on-disk chapters
> deliver the generator as **interfacial(wedge)-angle → minimum-deviation
> radius tables** (AH-CH10 p6; SAX-CH11 Tables 11.1–11.2) plus the Galle
> wedge relation. The ledger cites those tables as the generator; it does
> not reconstruct or guess the deviation formula.

## Status-summary table

| phenomenon / family | lens | Sundog status | generator source | proof tier |
| --- | --- | --- | --- | --- |
| 22° halo | §A/§B | `rendered-core` | Tape AH-CH10 p6 (3–5 wedge 62°→21.8°); `22deg halo.sim` | proven core |
| Sundog / parhelion | §A | `rendered-core` | Code `phase3.daggerOffset`; `Parhelia.sim`; Pass A1 | proven core · sole inverse handle |
| Upper tangent arc | §A/§B | `rendered-optional` | Code `tangentArcLocus` (Pass C7); Tape AH-CH06 p62; `Tangent arcs.sim` | bounded (single-cell) |
| Circumzenithal arc | §A/§B | `rendered-core` | Code `czaVisible` h≤32°; Tape AH-CH06 p63; `Circumzenithal arc.sim` | proven core (coverage-gated inverse) |
| Parhelic circle | §A/§B | `rendered-core` | Constant-altitude small circle; `Parhelic circ and more.sim` (code smile = empirical fit) | proven core (atlas curve = placeholder) |
| 46° halo | §A/§B | `rendered-core` | Tape AH-CH10 p6 (1–5 wedge 90°→45.7°); `46halo.sim` | proven core |
| Sun pillar | §A/§B | `rendered-core` | Reflection off near-horizontal basal faces; `Sun pillar.sim` (atlas vesica = placeholder) | proven core (mechanism ≠ atlas construction) |
| Supralateral arc | §B | `rendered-optional` | 90° side→end face, singly-oriented columns, h<32°; `Supralateral arc.sim` | failed inverse gate; install only |
| Infralateral arcs | §B | `rendered-optional` | 90° end→side face, horizontal columns; `Infralateral arc.sim` | install only |
| Suncave / sunvex Parry; Parry supralateral | §B/§C | `rendered-optional`, `named-only` | Parry orientation; Tape AH-CH06 Fig 6-14/6-17; `Parry arcs.sim` | install + route to §C |
| Circumscribed halo | §B | `not-modeled` (= UTA/LTA merged) | 60° prism, singly-oriented columns, h≳29°; `Circumscribed halo.sim` | install only |
| Pyramidal / odd-radius halos | §C | `named-only`, `not-modeled`, `halosim-reproducible` | Tape AH-CH10 p6 table; SAX-CH11 Galle α=180−2x; `Pyramidal *.sim`; phase15 ray-filter receipts | P2 simulated candidate; P3 not met |
| Lowitz arcs | §C | `named-only`, `not-modeled` | 60° prism, Lowitz orientation; `Lowitz arcs.sim` | P0/P1 |
| Antisolar features | §C | `named-only`, `not-modeled` | multi-internal-reflection, rear sky; `Anthelic Point display.sim` | P0 |
| Subhorizon halos | §C | `named-only`, `not-modeled` | reflection off horizontal plates, observer-above | P0 |
| Circumhorizon arc | §C | `named-only`, `not-modeled`, `halosim-reproducible` | 90° side→bottom face, plate/Parry, h>58°; `Circumhorizon arc.sim` | P2 simulated candidate; P3 not met |

## §A — Anchor: what we know

The proven / calibrated core. Each block names the project's own validated
closed form, the receipt that validates it, and the public claim boundary.
Generator detail (crystal / wedge / source) is in the matching §B block;
§A does not restate it.

### 22° halo · small halo
- **Status / evidence:** `rendered-core` · `calibrated-photo`, `literature`.
- **Project anchor:** the scale-lock primitive. `HALO_22_RADIUS = 220 px`
  ⇒ 22° at 10 px/° (Code, used as `phase3.HALO_22_RADIUS`). Every other
  radius in this ledger is locked to it.
- **Inverse handle:** none alone — it is the scale reference for all rows.
- **Boundary / next step:** public scale anchor and `literature` primitive;
  not itself a hidden-state proof. Keep as the lock for §A/§B radii.

### Sundog / parhelion · parhelion, mock sun, dagger
- **Status / evidence:** `rendered-core` · `calibrated-inverse`,
  `calibrated-photo`.
- **Project anchor:** `phase3.daggerOffset(h) = HALO_22_RADIUS / cos(h)`
  (Code line ~1060). Offset of the parhelion from the sun grows as
  `R22 / cos h` — confirmed by `Parhelia.sim` ("as the solar altitude
  increases, the parhelia pull away from the 22° halo").
- **Inverse handle:** the **sole promoted** hidden-state route after the
  Phase 10 audit — strict eligible photo set p2, p7, p13.
- **Boundary / next step:** preserve as the only promoted inverse handle;
  expand only when new eligible photo anchors land.

### Upper tangent arc · UTA
- **Status / evidence:** `rendered-optional` · `photo-partial`, `literature`.
- **Project anchor:** `tangentArcLocus(h)` ⇒ ρ(ψ) = 22 + A(h)·|ψ|^1.5,
  A(h) = 0.031·(29−18.6)/(29−h) (Code lines 813–838). **Pass C7-validated
  at the single cell h = 18.6°, 0.1° column tilt**; intermediate-h is
  PROVISIONAL. Returns `null` at h ≥ 29° (merges to circumscribed halo).
- **Inverse handle:** not promoted — Pass C7 falsified the earlier circle-fit
  handle ([`PASS_C7_OUTPUT.txt`](PASS_C7_OUTPUT.txt)).
- **Boundary / next step:** logo / animation vocabulary and a bounded
  forward model only; the Phase 12B increment-2 (h, tilt) render grid is
  the gate to de-provisionalize A(h).

### Circumzenithal arc · CZA
- **Status / evidence:** `rendered-core` · `photo-partial`, `literature`.
- **Project anchor:** `phase3.czaVisible(h)` ⇒ visible only for h ≤ 32°
  (Code lines 965–969). Independently confirmed by Tape AH-CH06 p63 ("for
  sun elevations greater than about 32°, the internal reflection becomes
  total, the circumzenith arc disappears") and `Circumzenithal arc.sim`.
- **Inverse handle:** CZA-apex route is **coverage-gated** — not promoted.
- **Boundary / next step:** conditional visible vocabulary; re-anchor any
  pre-A1b CZA example before any residual enters public-facing proof.

### Parhelic circle · horizontal belt
- **Status / evidence:** `rendered-core` · `photo-partial`, `literature`.
- **Project anchor:** `parhelicCurvatureFromAltitude(h)` is an **empirical
  smile fit** pinned to the 25° Troels Nielsen calibration (Code lines
  971–981) — *not* the canonical constant-altitude locus (see §B). It is
  inside the rendered core but is a **placeholder-class** generator.
- **Inverse handle:** none — belt-y residual is primitive QA only.
- **Boundary / next step:** track belt-y residuals as QA; reconcile the
  empirical smile against the §B constant-altitude circle before any
  belt-based claim.

### 46° halo · large halo
- **Status / evidence:** `rendered-core` · `photo-partial`, `literature`.
- **Project anchor:** rendered with the post-A1b `WB_R46` correction;
  `HALO_46_RADIUS = 440 px` (Code). Tape AH-CH06 p60 is the discipline
  anchor: a sharp 46° halo from *random* crystals is **not** a combination
  of supralateral + infralateral arcs — keep the rows distinct.
- **Inverse handle:** none promoted (CZA / supralateral context only).
- **Boundary / next step:** rendered vocabulary; clarify tangent/crossing
  language per the public legend (parallel `legend.html` workstream).

### Sun pillar · light pillar
- **Status / evidence:** `rendered-core` · `photo-partial`, `literature`.
- **Project anchor:** `applyPillarFromTwoHalos` builds the pillar as a
  vesica of two dagger-centered halos (Code ~487). This is an **atlas
  construction tied to dagger geometry**, geometrically unrelated to the
  physical mechanism (reflection off near-horizontal basal faces — see §B).
  Placeholder-class within the rendered core.
- **Inverse handle:** none.
- **Boundary / next step:** mark the exact generator boundary in the public
  legend; the atlas pillar is stylized, not a crystal-physics proof.

### Honesty ratchet — rendered ≠ anchored
These are drawn or labelled by the atlas but have **no derivation cited** in
the project code; they are hardcoded circles (`applySupralateralArc`,
`applySuncaveParryArc`, `applyParrySupralateralArc`,
`applyInfralateralArc`). They are **not** anchored. They are accounted for
only via the canonical generator in §B and must be reconciled against it
before any promotion:

- Supralateral arc, infralateral arcs → §B (singly-oriented columns, 90°).
- Suncave Parry, sunvex Parry, Parry supralateral → §B / §C (Parry).
- The parhelic-circle smile and the sun-pillar vesica are flagged in their
  §A blocks above as placeholder-class generators inside the rendered core.

## §B — Install: what we should have known

The canonical pre-existing generator for each family, read straight from the
Tape scans and the HaloSim recipe descriptions. **No geometry is invented
here** — every line is a citation.

### Ordinary halos — 22° and 46°
- **Crystal / orientation:** hexagonal prism · random (`random.xng`).
- **Generator:** minimum-deviation refraction. **22°** = 60° prism wedge
  (prism side face → side face inclined 60°), deviation floor ≈ 21.8° at
  n = 1.31 (Tape AH-CH10 p6, faces 3–5; `22deg halo.sim`: "rays … leave
  through a face inclined 60 degrees to the first … none … less than 22
  degrees"). **46°** = 90° wedge (prism side ↔ end face), floor ≈ 45.7°
  (Tape AH-CH10 p6, faces 1–5; `46halo.sim`).
- **Observer:** ground; any sun altitude.
- **Source:** Tape AH-CH10 p6 wedge→radius table; HaloSim `22deg halo.sim`,
  `46halo.sim`; Greenler / Cowley.

### Parhelion / sundog
- **Crystal / orientation:** hexagonal plate · plate (`plate .Ndeg
  disp.xng`), large faces near-horizontal.
- **Generator:** 60° prism-wedge minimum deviation (same faces as the 22°
  halo) under plate orientation; the parhelion sits at the sun's altitude
  and offsets from the sun by `R22 / cos h` as h rises (`Parhelia.sim`;
  Tape parhelion family). This is the canonical law the §A code anchor
  implements.
- **Observer:** ground; sharpest at low sun, smears toward the 22° halo at
  high sun.
- **Source:** `Parhelia.sim`, `Sundogs.sim`; Tape AH; Cowley.

### Tangent arcs / circumscribed halo
- **Crystal / orientation:** hexagonal column · singly-oriented horizontal
  columns (`Horiz column .Ndeg disp.xng`).
- **Generator:** the **same 60° prism wedge as the 22° halo**, differing
  only by orientation (`Tangent arcs.sim`: "the very same faces … but …
  the crystals have no favoured orientation"). Upper + lower tangent arcs
  bend together and **merge into the circumscribed halo at sun elevation
  ≈ 29°** (Tape AH-CH06 p62: "at a sun elevation of 29° the two halos merge
  … the value 29° … is theoretical"; `Circumscribed halo.sim`). This is
  exactly the boundary condition the §A `tangentArcLocus` `null`-guard and
  the A(h) → ∞ limit encode — the project code comment already cites
  "Tape AH Ch 6 / Pass C7".
- **Observer:** ground; arc shape strongly h-dependent.
- **Source:** Tape AH-CH06 p61–62; `Tangent arcs.sim`,
  `Circumscribed halo.sim`.

### Circumzenithal arc
- **Crystal / orientation:** hexagonal plate · plate, large prism end faces
  horizontal.
- **Generator:** light enters the **top horizontal face** and exits a
  **vertical side face** (≈ 90° effective refraction); forms only for sun
  altitude ≲ 32° (above that the relevant internal reflection goes total).
  `Circumzenithal arc.sim`; Tape AH-CH06 p63.
- **Observer:** ground; h ≲ 32°.
- **Source:** `Circumzenithal arc.sim`; Tape AH-CH06 p63; Cowley.

### Supralateral & infralateral arcs
- **Crystal / orientation:** hexagonal column · singly-oriented horizontal
  columns (`Horiz column .1 deg disp.xng`).
- **Generator:** 90° refraction. **Supralateral** = ray enters a side face,
  exits a perpendicular end face; forms only for h < ~32° (like the CZA)
  (`Supralateral arc.sim`). **Infralateral** = the reciprocal, ray enters a
  vertical end face, exits a side face (`Infralateral arc.sim`); shape
  changes with sun elevation and resembles the circumhorizon arc at high
  sun. Tape AH-CH06 p60 warns these are *not* the same as a random-crystal
  46° halo — keep distinct.
- **Observer:** ground; supralateral h<32°, infralateral broad h-range.
- **Reconciliation target:** this is the canonical generator the §A
  placeholder arcs (`applySupralateralArc`, `applyInfralateralArc`) must be
  measured against before any promotion.
- **Source:** `Supralateral arc.sim`, `Infralateral arc.sim`; Tape AH-CH06
  p60.

### Parry-family arcs
- **Crystal / orientation:** hexagonal column · Parry orientation
  (`Parry .N - .N deg disp.xng`) — columns with two prism faces horizontal.
- **Generator:** refraction through Parry-oriented prism faces. Lower
  **sunvex** Parry: ray enters one sloping prism face and exits the other
  (Tape AH-CH06 Fig 6-14). Lower **suncave** Parry: ray enters an upper
  sloping prism face and exits the bottom prism face; occurs only for
  fairly high sun (Tape AH-CH06 Fig 6-17, p67). `Parry arcs.sim`,
  `Parry 1820 display.sim`.
- **Observer:** ground; suncave members need high sun.
- **Reconciliation target:** the §A `applySuncaveParryArc` /
  `applyParrySupralateralArc` placeholders → measure against this.
- **Source:** Tape AH-CH06 Fig 6-14/6-17 p66–67; `Parry arcs.sim`.

### Circumhorizon arc
- **Crystal / orientation:** hexagonal plate (and Parry) · plate / Parry.
- **Generator:** ray enters an almost-vertical side face and exits the
  **lower horizontal face** (≈ 90° refraction); a close relation of the
  CZA, visible only for sun **> ~58°**, always low and parallel to the
  horizon (`Circumhorizon arc.sim`; Tape AH-CH06 p65 Fig 6-13 ray path).
- **Observer:** ground; high sun (> 58°) — outside the sundog-facing
  explainer's default regime.
- **Source:** `Circumhorizon arc.sim`; Tape AH-CH06 p65.

### Sun pillar & subsun
- **Crystal / orientation:** plate (also column / Parry) · near-horizontal
  basal faces.
- **Generator:** **reflection, not refraction.** The pillar is a vertical
  streak through the sun from reflection off the slightly-tilted horizontal
  basal faces of plate crystals; "no net dispersion … so they are
  colourless" (`Sun pillar.sim`). The subsun is the direct mirror
  reflection seen from above the crystal cloud. There is **no
  minimum-deviation halo radius** — it is a reflection locus, which is why
  the §A atlas vesica construction is geometrically unrelated.
- **Observer:** pillar from ground; subsun needs observer above the cloud.
- **Source:** `Sun pillar.sim`, `Subhorizon arcs.sim`; Tape AH-CH06 p58.

### Parhelic circle
- **Crystal / orientation:** plate / column / Parry, vertical-axis crystals.
- **Generator:** the locus of constant altitude equal to the sun's — a
  horizontal small circle through the sun. Mechanism is mixed: a single
  external reflection dominates near the sun, with refraction + one or more
  internal reflections farther away ("formed via a greater variety of ray
  paths than any other halo", `Parhelic circ and more.sim`). The §A code
  smile is an empirical fit to this, not the constant-altitude circle.
- **Observer:** ground; any sun altitude.
- **Source:** `Parhelic circ and more.sim`; Tape AH; Cowley.

### Pyramidal / odd-radius halos
- **Crystal / orientation:** **pyramidal** ice with {10-11}/{10-1-1} pyramid
  faces (`Pyr_*.xsh`, e.g. `Pyr_.30_.30_.30_27.98.xsh` encodes the apex) ·
  random for the circular odd-radius rings; plate/Parry for pyramidal
  parhelia (`Pyramidal *.sim`).
- **Generator (closed-form, from Tape):** halo radius is a function of the
  interfacial (wedge) angle between entry and exit faces. **Tape AH-CH10 p6
  (Fig 10-8)** gives the canonical table for random pyramidal crystals at
  n = 1.31:

  | entry–exit faces | wedge angle | halo radius | observed |
  | --- | --- | --- | --- |
  | 3–26 | 28.0° | **9.0°** | 9.1° |
  | 13–25 | 52.4° | **18.3°** | 18.0° |
  | 23–26 | 56.0° | **19.9°** | 20.1° |
  | 3–5 (prism) | 62.0° | **21.8°** | 22.0° |
  | 1–25 | 60.0° | **22.9°** | — |
  | 3–25 | 63.8° | **23.8°** | 23.6° |
  | 23–25 | 80.2° | **34.9°** | 35.5° |
  | 1–5 (basal+prism) | 90.0° | **45.7°** | — |

  The pyramid-face inclination *x* to the c-axis sets the wedge angles via
  the **Galle relation α = 180 − 2x** (SAX-CH11 p2); the **Rational
  Tangents Principle** `tan x / tan x₀ = v/u` (SAX-CH11 p6, x₀ = 54.7°
  Bravais) enumerates crystallographically likely *x*. SAX Tables 11.1
  (x = 54.7°) and 11.2 (x = 43.3°) give worked wedge→Δ_min sets.
- **Observer:** ground; random orientation for the circular family.
- **Source:** Tape AH-CH10 p6; Tape SAX-CH11 p2/p4/p6, Tables 11.1–11.2;
  `Pyramidal 9d/18d/20-35d halo.sim`, `Pyramidal parhelia.sim`. *Underlying
  Δ_min equation: Tape Ch 4 (not on disk) — table cited as the generator.*

## §C — Extrapolate: bounding the unknown

For families Sundog does not anchor. Each block states what the installed
geometry predicts, the candidate inverse handle (the project's
indirect-inference method — recover hidden state from an observed locus),
the falsification criterion, and the Phase 15 proof tier. **No result
language**: these are simulated / speculative candidates, never "discovered
halos".

> Full per-candidate proof records (generator, crystal population, window,
> signature, HaloSim config + receipt, atlas-projection status,
> falsification, pre-registration, tier) live in the Phase 15 ledger
> [`SPECULATIVE_HALO_PROOFS.md`](SPECULATIVE_HALO_PROOFS.md). As of
> 2026-05-15, pyramidal/odd-radius and circumhorizon are **P2** (analytic
> + 14E HaloSim receipt); Lowitz/antisolar/subhorizon are P0 stubs. The
> tiers below are the summary; that doc is canonical for proof status.

### Pyramidal / odd-radius halos
- **Predicts:** discrete rings at 9°, 18°, 20°, 23°, 35° (plus 22°/46°)
  from the §B wedge table, for random pyramidal crystals; the ~18° and
  ~23° "halos" are blends (18.3°+19.9°, 22.9°+23.8°).
- **Inverse-handle candidate:** the observed ring-radius *set* over-
  determines the pyramid-face inclination *x* (via α = 180 − 2x) — a
  hidden crystal-population parameter recoverable from a single
  multi-ring photo, analogous to recovering sun altitude from the
  parhelion offset.
- **Falsification:** a measured ring that does not match any
  wedge→Δ_min row for any crystallographically rational *x* (Rational
  Tangents) falsifies the pyramidal attribution.
- **Tier:** **P2 simulated candidate.** The analytic table and HaloSim
  reproduction stand, but the 2026-05-15 1D residual campaign is a clean
  negative for P3: 1M gave 0 clean rings, 6M gave 1 marginal ring, and the
  8-frame 4M ray-filter isolation run gave 1 strong ring (4.6σ), still
  below the ≥3 azimuthally-separable rings required for a residual table.
  A fourth lever — HaloSim's own `Tools → Scale` instrument stamped on
  all 8 wedge frames (`phase15_pyrfilter/pyr_w*_scale.png`) — likewise
  could not anchor-validate (the stamped ruler's span is shorter than the
  ring field; 6-fold symmetry still prevents per-wedge isolation), and
  the read script refused to tabulate. **Disposition (closed strategy
  decision, not a deferred task): P2 is this evidence chain's ceiling.**
  Four fundamentally different methods (1M / 6M / 8×ray-filter /
  8×HaloSim-Scale) were exhausted with the no-fabrication gate holding
  every time; P3 needs a fundamentally different setup and is out of
  scope here. Canonical detail: [`SPECULATIVE_HALO_PROOFS.md`](SPECULATIVE_HALO_PROOFS.md).

### Lowitz arcs · upper / middle / lower Lowitz
- **Predicts:** 60° prism refraction under **Lowitz orientation** (crystal
  rotating about a horizontal axis through opposed prism edges); upper arc
  above the sun joining the 22° parhelia, lower arc extending from the
  parhelia (`Lowitz arcs.sim`). Confusable with the 23° pyramidal and
  Parry arcs.
- **Inverse-handle candidate:** none promoted; the arc family would only
  ever co-constrain orientation, not sun altitude.
- **Falsification:** position coincidence with the 23° pyramidal / Parry
  family unless the full Lowitz arc continuity is present.
- **Tier:** **P0 named / P1** (historical: Lowitz 1790 St Petersburg;
  modern photos exist — `observed` outside the project).

### Antisolar / anthelic features · anthelion, anthelic arcs, Wegener, Tricker, subhelic, 120° parhelia
- **Predicts:** rear-sky features from multi-internal-reflection ray paths
  in column + Parry crystals near the anthelic point (direction opposite
  the sun, same altitude); Tricker "Ankh", twin Wegener arcs, tangential
  subhelic arc (`Anthelic Point display.sim`, camera az ≈ 155°).
- **Inverse-handle candidate:** none — outside the front-facing atlas
  field; observer must look away from the sun.
- **Falsification:** N/A at this tier (catalogue only).
- **Tier:** **P0 named / catalogue**. Decide later whether the public
  legend shows rear-sky vocabulary or keeps it docs-only.

### Subhorizon halos · subsun, subparhelia
- **Predicts:** subsun = direct reflection off horizontal plate basal
  faces; subparhelion = an extra internal reflection within the plate
  (`Subhorizon arcs.sim`). Requires the observer **above** the crystal
  cloud (aircraft / mountain).
- **Inverse-handle candidate:** subsun depression = − sun altitude (a
  trivial mirror relation) — only meaningful for an above-cloud observer.
- **Falsification:** absence of the mirror-symmetric subsun at the
  predicted negative altitude.
- **Tier:** **P0 / observer-geometry extension** — not the default
  ground-observer atlas.

### Circumhorizon arc · CHA, "fire rainbow"
- **Predicts:** §B generator (90° side→bottom face, plate/Parry) — a long
  horizontal band low in the sky, only for sun > 58°.
- **Inverse-handle candidate:** presence alone bounds sun altitude to
  > 58° — a weak one-sided altitude constraint.
- **Falsification:** appearance at sun < 58° falsifies the plate-CHA
  attribution (likely an infralateral arc instead — Tape AH-CH06 p65).
- **Tier:** **P1 analytic-feasible**; high-sun regime outside the current
  sundog-facing explainer.

### Named-only catalogue entries (P0)
Present in the HaloSim library and the literature, retained for vocabulary
completeness, no project geometry yet: **Kern arc** (`Kern arc.sim`),
**Liljequist parhelia** (`Liljequist parhelia.sim`), **Wegener / Hastings
arcs** (`Wegener and other arcs.sim`), **Bishop's ring**
(`Bishop display.sim`), and the historical display poses **Parry 1820** /
**St Petersburg 1790** (`Parry 1820 display.sim`,
`St Petersburg display.sim`). All **P0 named / catalogue**; promote into a
dedicated §C block only when a specific proof or user need appears.

## What counts as "accounted for"

A phenomenon is **not** accounted for because its name appears on a page or
in the status-summary table. It is accounted for only when its block names:
source, crystal family / orientation, geometric generator, observer
geometry, Sundog rendering status, evidence status, public claim boundary,
and next proof step — and, for §C, additionally a falsification criterion
and a Phase 15 proof tier.

## Phase 14 Work Queue

1. **14A — Ledger seed:** *done* (2026-05-14). This file is the canonical
   tracking surface.
2. **14B — Public legend:** `legend.html` is a **parallel workstream the
   owner runs separately**; this ledger is kept row-consistent with it but
   does not edit it.
3. **14C — Machine-readable mirror:** `public/data/halo-phenomena-status.json`
   remains intentionally deferred until the public UI needs dynamic
   filtering or Ask Sundog needs the same rows.
4. **14D — Source pass:** *substantially landed (this revision).* Every
   family carries an exact Tape chapter / page or HaloSim `.sim` citation
   and the crystal-orientation family; the underlying Δ_min equation is
   cited to Tape Ch 4 (off-disk) rather than reconstructed.
5. **14E — HaloSim receipt pass:** *landed 2026-05-15.* All 11 un-receipted
   `.sim`-backed rows (supralateral, infralateral, circumscribed, parhelic
   circle, sun pillar, Parry, pyramidal/odd-radius, Lowitz, antisolar,
   subhorizon, circumhorizon) rendered B&W at 1M rays via the proven HS-0
   mechanism and confirmed to reproduce the named phenomenon —
   `halosim-candidate` → **`halosim-reproducible`** for every one (0 marked
   not-reproducible). Receipts + provenance table:
   [`halosim_outputs/phase14e/`](halosim_outputs/phase14e)
   (`_PHASE14E_RECEIPTS.md`); rays-pinned source frames in
   [`halosim_outputs/phase14e_frames/`](halosim_outputs/phase14e_frames),
   generated byte-safe by
   [`scripts/halosim_pin_rays.py`](../../scripts/halosim_pin_rays.py).
   22°/46°/CZA/parhelion (`hs0_spike/`) and the upper tangent arc (Pass C7)
   were already receipted. Empirical finding: ~300k b&w is unreliable —
   ~1M b&w is the floor (AGENTS.md + HALOSIM_VALIDATION_PROTOCOL.md updated).
6. **14F — Proof routing:** *partially landed; Phase 15 seeded
   2026-05-15.* §C rows carry P0–P5 tiers and falsification lines, and the
   full Phase 15 proof records now live in
   [`SPECULATIVE_HALO_PROOFS.md`](SPECULATIVE_HALO_PROOFS.md): pyramidal /
   odd-radius and circumhorizon reached **P2** (analytic + a 14E HaloSim
   reproduction receipt); Lowitz / antisolar / subhorizon are P0 stubs.
   P3+ and the public-language gate remain downstream.

## Gate

Phase 14 closes only when this ledger and the parallel public `legend.html`
agree row by row, and no public or chatbot surface can reasonably be read as
"Sundog has accounted for all halo geometry" without distinguishing
rendered-core, rendered-optional, named-only, not-modeled, and speculative
entries. **This revision does not close the gate**: it lands 14D and routes
14F, and records that ledger ↔ legend row-consistency is cross-checked
against the in-parallel `legend.html`, not closed here.
