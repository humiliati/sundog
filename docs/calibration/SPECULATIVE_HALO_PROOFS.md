# Speculative / Unphotographed Halo Proof Program

Filed: 2026-05-15
Phase: Sundog V Geometry Phase 15
Status: **seeded** — pyramidal & circumhorizon at **P2**; Lowitz / antisolar
/ subhorizon **P0** stubs; gate **not** reached.

## Purpose

This is the Phase 15 ledger for halo forms that are unseen, rarely
photographed, or only speculated. It exists to investigate "can the math, a
brute-force ray trace, the atlas, and HaloSim together predict a halo before
photographs or specialist consensus catch up?" — **without turning
speculation into public proof**.

One proof record per candidate. A record names the proposed generator, the
necessary crystal population, the sun-altitude / observer-geometry window,
the expected visual signature, the HaloSim configuration, the atlas
projection status, the falsification criteria, the pre-registration, and the
current proof tier. A candidate is **not** "proven" because it has a name or
a render; it is only as strong as the tier its evidence supports.

This document depends on the Phase 14 accounting matrix
([`HALO_PHENOMENA_ACCOUNTING.md`](HALO_PHENOMENA_ACCOUNTING.md)) and the
Phase 12A HaloSim validation protocol
([`HALOSIM_VALIDATION_PROTOCOL.md`](HALOSIM_VALIDATION_PROTOCOL.md)).

## Proof tiers

| tier | meaning | public boundary |
| --- | --- | --- |
| P0 named / rumored | name appears in literature, historical report, or HaloSim catalogue | catalogue only |
| P1 analytic-feasible | ray geometry or projection math says the locus is possible | speculative math candidate |
| P2 brute-force reproducible | independent ray tracer or HaloSim produces the feature under declared crystal / observer conditions | simulated candidate |
| P3 atlas-representable | the locus can be expressed in the atlas parameter grammar without ad hoc placement | model candidate |
| P4 predictive target | pre-registered conditions and photo-search / controlled-observation plan exist | prediction |
| P5 observed / specialist-confirmed | photograph, simulation, and specialist taxonomy agree | promotable evidence |

## Gate

A speculative halo **cannot move into public-facing result language** until
it reaches at least **P4** with pre-registration and independent simulation
receipts; **P5** requires real observation or specialist-confirmed
classification. Negative results are kept as useful map entries, not buried.

**This document does not cross the gate.** The highest tier reached here is
**P2** (analytic + HaloSim-reproduced). Every P1–P4 entry below is a
*simulated / speculative candidate* — never a "discovered halo," never a
public Sundog result. Ask Sundog and any public surface must say the same.

---

## P2 — Pyramidal / odd-radius halos

**Tier: P2 (brute-force reproducible).** P1 analytic-feasible **and**
HaloSim-reproduced under declared conditions. **Not** P3 (see below).
Public boundary: *simulated candidate* — not a discovered halo, not a
Sundog result.

- **Proposed generator (P1 analytic).** Minimum-deviation refraction
  through the wedges of pyramidal {10-11}/{10-1-1} ice crystals. The
  closed-form generator is the interfacial(wedge)-angle → minimum-deviation
  radius mapping in **Tape AH-CH10 p6 (Fig 10-8)** at n = 1.31:

  | entry–exit faces | wedge | halo radius | observed (Neiman) |
  | --- | --- | --- | --- |
  | 3–26 | 28.0° | **9.0°** | 9.1° |
  | 13–25 | 52.4° | **18.3°** | 18.0° |
  | 23–26 | 56.0° | **19.9°** | 20.1° |
  | 3–5 (prism) | 62.0° | **21.8°** | 22.0° |
  | 1–25 | 60.0° | **22.9°** | — |
  | 3–25 | 63.8° | **23.8°** | 23.6° |
  | 23–25 | 80.2° | **34.9°** | 35.5° |
  | 1–5 (basal+prism) | 90.0° | **45.7°** | — |

  Pyramid-face inclination *x* to the c-axis sets the wedge angles via the
  **Galle relation α = 180 − 2x**; the **Rational Tangents Principle**
  `tan x / tan x₀ = v/u` (x₀ = 54.7° Bravais) enumerates
  crystallographically likely *x* (Tape SAX-CH11 p2/p4/p6, Tables
  11.1–11.2). Underlying Δ_min equation: Tape Ch 4 (off-disk) — the table
  is cited as the generator, not reconstructed. Source: ledger §B.
- **Necessary crystal population.** Randomly-oriented pyramidal crystals,
  `Pyr_*.xsh` (e.g. `Pyr_.30_.30_.30_27.98.xsh`, apex ≈ 27.98°). Oriented
  pyramidal crystals produce pyramidal parhelia/arcs — qualitatively
  described by Tape, no closed form on the on-disk pages; out of scope
  here.
- **Sun-altitude / observer-geometry window.** Ground observer; broad
  sun-altitude range (the rings are sun-centred). 14E receipt at h = 18.3°.
- **Expected visual signature.** Discrete concentric rings at ≈ 9°, 18°,
  20°, 23°, 35° in addition to the ordinary 22°/46°. The visible ~18° and
  ~23° "halos" are blends (18.3°+19.9°, 22.9°+23.8°).
- **HaloSim configuration + P2 receipt.** `Pyramidal 20-35d halos.sim`
  (also `Pyramidal 9d halo.sim`, `Pyramidal parhelia.sim`), random
  `Pyr_*.xsh`, B&W ("Grey shades on white") at 1M rays via the proven
  HS-0 mechanism. Receipt:
  [`halosim_outputs/phase14e/pyramidal_h18_1M_bw.png`](halosim_outputs/phase14e/pyramidal_h18_1M_bw.png)
  — concentric odd-radius rings clearly reproduced
  ([`phase14e/_PHASE14E_RECEIPTS.md`](halosim_outputs/phase14e/_PHASE14E_RECEIPTS.md)).
  **⇒ P2 met.**
- **Atlas projection.** The odd-radius ring family is **not** expressed in
  the `public/js/parhelion-geometry.mjs` parameter grammar — there is no
  pyramidal-wedge primitive in the atlas. **P3 is NOT met** and is not
  claimed.
- **Falsification.** A measured ring that matches **no** wedge→Δ_min row
  for any crystallographically rational *x* (via Galle + Rational Tangents)
  falsifies the pyramidal attribution for that ring.
- **Pre-registration (for any future promotion).**
  - *Positive:* in an independent render or photograph, ≥ 3 of the
    predicted rings (9 / 18 / 20 / 23 / 35°) are present within the
    measurement tolerance, at radii matching the AH-CH10 table.
  - *Null:* rings absent, or present only at radii that do not match any
    rational-*x* wedge row.
  - *Tier ceiling without it:* stays P2. May enter **P3 only** when the
    ring set is expressible in the atlas parameter grammar without ad-hoc
    placement; **P4 only** with a pre-registered photo-search /
    controlled-observation plan; **P5** requires real observation or
    specialist-confirmed classification.
- **Quantitative probe (follow-up #1) — attempted 2026-05-15, NEGATIVE.**
  [`scripts/pyramidal_ring_residual.py`](../../scripts/pyramidal_ring_residual.py)
  masked the blue plot, found the ring centre by radial-profile
  peak-sharpness, and applied a detrend + 3σ resolvability gate to the
  14E receipt. **Result: no resolvable discrete ring loci.** The
  azimuthally-averaged radial profile is a smooth, core-dominated diffuse
  glow; only the inner bright feature (r ≲ 62 px — the ~9° ring blended
  with the sun core) exceeds 3σ. No discrete 18–35° rings rise above
  Monte-Carlo noise. This is a **data** limit (1M rays is too few for the
  intrinsically-faint pyramidal paths spread over an all-sky plan
  projection), **not** a method limit — the detrend+3σ test would surface
  rings if present. **No predicted-vs-measured residual table was
  produced; the script refuses to fabricate one off a smooth profile.**
  *Consequence:* pyramidal **remains P2** — this evidence does **not**
  advance it toward P3. Evidence retained:
  [`halosim_outputs/phase14e/pyramidal_h18_1M_bw_radialprofile.txt`](halosim_outputs/phase14e/pyramidal_h18_1M_bw_radialprofile.txt)
  (radial profile + detrended SNR) and the script. Negative result kept
  as a map entry per the gate, not buried.
- **Higher-fidelity retry (follow-up #1′, ray-count lever) — 6M render,
  2026-05-15, VERIFIED NEGATIVE.** A dedicated 6M-ray B&W "Sun centered
  Plan" render
  ([`halosim_outputs/phase14e/pyramidal_h18_6M_bw.png`](halosim_outputs/phase14e/pyramidal_h18_6M_bw.png),
  source frame `phase14e_frames/pyramidal_6M.sim`, HS-0 mechanism,
  `Startup.sim` backed up + restored byte-identical) re-run through the
  gated script. 6× the rays gave only a **marginal** gain: **one** outer
  ring now resolves (r ≈ 132 px, 4.0σ) versus zero at 1M — still short of
  the ≥3 distinct rings a scale-locked residual table + 46° linearity
  cross-check require. Critically, this was **not** a centring artifact: a
  wide center-grid brute search confirmed (944,393) is the
  peak-sharpness-optimal centre, and **no** centre anywhere in the plot
  yields >1 resolvable outer ring. Conclusion: the **pure ray-count lever
  is insufficient** for the all-sky plan projection — the closely-spaced
  odd-radius rings (18.3/19.9°, 22.9/23.8°) stay blended under the diffuse
  glow and the dominant 22° prism halo. Pyramidal **still P2**, no
  residual table, nothing fabricated. The 6M PNG is kept as a
  *higher-fidelity qualitative* reproduction receipt (it does strengthen
  the P2 visual reproduction); evidence:
  `phase14e/pyramidal_h18_6M_bw_radialprofile.txt`. The definitive
  remaining path is now follow-up #1′ below — **ray-filter isolation**,
  not more rays.
- **Ray-filter-isolation campaign (follow-up #1′) — 2026-05-15,
  MECHANICALLY SUCCESSFUL · QUANTITATIVELY NEGATIVE.** Built
  [`scripts/halosim_filter_frames.py`](../../scripts/halosim_filter_frames.py)
  (byte-safe `.sim` edit of the ray-filter Entrance T+8 / Exit T+10
  fields + rays — a clean text-edit extension of the proven HS-0 loop,
  *not* GUI face-typing), generated **8 wedge frames** (entry→exit faces
  per the Tape AH-CH10 table: 3→26, 13→25, 23→26, 3→5, 1→25, 3→25,
  23→25, 1→5) at 4M, and ran all 8 through HS-0 (`Startup.sim` backed up
  + restored byte-identical). Receipts:
  [`halosim_outputs/phase15_pyrfilter/`](halosim_outputs/phase15_pyrfilter)
  (`pyr_w*_4M.png`). **Finding 1 (mechanism):** the face-pair filters do
  **not** single-isolate one ring — the crystal's 6-fold symmetry admits
  many symmetry-equivalent paths, so each filtered render still shows the
  *family*; but the filter (+ 2-face-traversal cap) **markedly crispened**
  the rings vs the diffuse 1M/6M. **Finding 2 (quantitative):** under the
  rigorous detrend + 3σ test with a background-agnostic contrast signal
  and an edge-excluded ring-SNR centre (the `--filtered` mode of
  `pyramidal_ring_residual.py`), the crispest receipt (`pyr_w9`,
  ring-SNR centre 783,441) resolves **1 strong ring** (r ≈ 211 px,
  4.6σ) — the clear data-quality progression is **1M = 0 clean rings →
  6M = 1 marginal → 4M-filtered = 1 strong** — but **still < the ≥3
  azimuthally-separable rings** a predicted-vs-measured table + the
  22°/46° linearity cross-check require. Four independent centring
  methods (peak-sharpness, contrast-coherence, gradient-Hough, ring-SNR)
  were applied; the family visible *by eye* does not separate into ≥3
  clean loci on these split-sky "Sun centered Plan" renders.
  **No residual table fabricated. Pyramidal stays P2.** Evidence:
  `phase14e/pyr_w9_e3_x26_4M_radialprofile.txt`; 8 receipts +
  byte-safe frame sims retained. Refined remaining path is #1′ below.
- **HaloSim-native angular-Scale method (follow-up #1′ final lever) —
  2026-05-15, METHOD IMPLEMENTED · ANCHORS FAIL · P2 UNCHANGED.** Recon
  found HaloSim's own measurement instrument (`Tools → Scale`, FIX —
  help h10/h1/h5: a degree-graduated ruler drawn *from the sun*,
  stampable into the render). It dissolves the px↔° scale-lock and
  centre-finding in principle: the ruler's row gives the exact sun-Y and
  its ticks encode HaloSim's own non-linear projection mapping. All 8
  Tape-wedge frames were re-rendered with the scale FIXed mid-render and
  harvested (`phase15_pyrfilter/pyr_w*_scale.png` — 8 scale-stamped
  artifacts; sun-Y robustly locked at y≈393 by the ruler; new tool
  [`scripts/pyramidal_scale_read.py`](../../scripts/pyramidal_scale_read.py)
  with a hard anchor gate). **Outcome:** the anchor gate **failed and the
  script refused to tabulate** — two compounding limits: (a) the stamped
  ruler's angular span is *shorter than the ring field* (the ordinary
  22°/46° anchor rings at 21.8°/45.7° fall **beyond** the calibrated
  ruler tip, so they cannot be read against it), and (b) per-wedge
  isolation is still structurally impossible (6-fold symmetry → the same
  dense family in every wedge frame). The Scale tool itself works and the
  artifacts are a genuine asset (HaloSim's own calibrated instrument
  stamped per frame), but as executed it does not yield an
  anchor-validated measured-vs-Tape set. **No fabrication; pyramidal
  stays P2.** Artifacts + script retained as precise map entries per the
  gate.
- **Status of the P3 probe — four methods exhausted.** 1M (0 rings) →
  6M (1 marginal) → 8× ray-filter isolation (crisp family, no
  per-wedge isolation, 1 ring by 1-D profile) → HaloSim-native Scale
  (instrument works but span < ring field; anchors un-validatable).
  The project's no-fabrication gate held at **every** step. The
  bottleneck is structural (6-fold-symmetry non-isolation + the
  Plan-projection/scale-span), not a tooling tweak away. **Recommended
  disposition: accept P2 as this evidence chain's ceiling.** Pyramidal
  is solidly P2 — P1 analytic (Tape AH-CH10 closed-form table) + P2
  qualitative HaloSim reproduction across many independent renders. P3
  (atlas-grammar representability / clean quantitative residual) requires
  a fundamentally different setup (a non-symmetric single-ring isolation,
  or a 2-D ring-template fit on a non-split equidistant render with a
  full-span scale) and is out of scope for this chain.

---

## P2 — Circumhorizon arc ("fire rainbow")

**Tier: P2 (brute-force reproducible).** P1 analytic-feasible **and**
HaloSim-reproduced. **Not** P3. Public boundary: *simulated candidate*.

- **Proposed generator (P1 analytic).** 90° refraction: the ray enters an
  almost-vertical side prism face and exits the **lower horizontal basal
  face** of a plate-oriented (and Parry-oriented) hexagonal crystal — a
  close relation of the circumzenithal arc. Tape AH-CH06 p65 Fig 6-13 ray
  path; ledger §B.
- **Necessary crystal population.** Plate-oriented (and Parry-oriented)
  hexagonal crystals.
- **Sun-altitude / observer-geometry window.** Ground observer; **sun
  > ~58°** only (high-sun regime). The arc is always low in the sky and
  parallel to the horizon. 14E receipt at h = 62°.
- **Expected visual signature.** A long, bright, spectrally-coloured
  horizontal band low in the sky, parallel to the horizon.
- **HaloSim configuration + P2 receipt.** `Circumhorizon arc.sim`,
  plate/Parry, B&W at 1M rays via HS-0. Receipt:
  [`halosim_outputs/phase14e/circumhorizon_h62_1M_bw.png`](halosim_outputs/phase14e/circumhorizon_h62_1M_bw.png)
  — the CHA band reproduced low and parallel to the horizon at h = 62°.
  **⇒ P2 met.**
- **Atlas projection.** Not parameterized in `parhelion-geometry.mjs`.
  **P3 NOT met**, not claimed.
- **Falsification.** Appearance at sun < ~58° falsifies the plate-CHA
  attribution — at lower sun the analogous low band is an infralateral
  arc, not the CHA (Tape AH-CH06 p65).
- **Pre-registration (for any future promotion).**
  - *Positive:* the band is present **only** for sun > ~58°, at the
    predicted elevation and orientation (parallel to horizon).
  - *Null:* a comparable low band appears at sun < ~58° (⇒ infralateral),
    or no band appears at high sun under plate populations.
  - *Tier ceiling without it:* stays P2. P3 needs an atlas-grammar
    expression; P4 a pre-registered observation plan; P5 real /
    specialist-confirmed observation.

---

## P0 — Catalogue stubs (not yet worked to P1+)

Named and present in the HaloSim library; recorded so the map is complete.
Each names its generator pointer, current tier, why it is not higher, and
what would promote it.

- **Lowitz arcs** — *P1-pending-isolation.* Generator: 60° prism
  refraction under **Lowitz orientation** (crystal rotating about a
  horizontal axis through opposed prism edges); `Lowitz arcs.sim`; ledger
  §C. A 14E receipt exists
  ([`phase14e/lowitz_h30_1M_bw.png`](halosim_outputs/phase14e/lowitz_h30_1M_bw.png))
  but the Lowitz arcs are intertwined with the 22° halo / parhelia **by
  their physics** (they emanate from the parhelia), so the reproduction is
  qualitative, not isolated. *Promotion to a clean P2:* a
  crystal-block-reduced isolation render (tracked follow-up #3) that shows
  the Lowitz arc system separated from the 22° halo.
- **Antisolar / anthelic features** (anthelion, anthelic arcs, Wegener,
  Tricker, subhelic, 120° parhelia) — *P0.* Generator: multi-internal-
  reflection ray paths in column + Parry crystals in the rear sky near the
  anthelic point; `Anthelic Point display.sim`; ledger §C. 14E receipt
  ([`phase14e/antisolar_h20_1M_bw.png`](halosim_outputs/phase14e/antisolar_h20_1M_bw.png))
  is a qualitative camera-view reproduction. *Promotion:* a per-feature
  analytic generator (P1) per named member before any P2 claim — the
  family is currently catalogued as one render, not resolved per arc.
- **Subhorizon halos** (subsun, subparhelia) — *P0.* Generator: direct
  reflection off horizontal plate basal faces (subsun) + an internal
  reflection (subparhelia); requires an **observer above** the crystal
  cloud (aircraft / mountain); `Subhorizon arcs.sim`; ledger §C. 14E
  receipt
  ([`phase14e/subhorizon_h18-5_1M_bw.png`](halosim_outputs/phase14e/subhorizon_h18-5_1M_bw.png)).
  *Promotion:* the subsun depression = − sun altitude relation is a
  trivial mirror law (near-P1) but only meaningful for the above-cloud
  geometry; resolve the observer-geometry framing before tiering up.

---

## Tracked follow-ups (deferred, not dropped)

1. ~~Quantitative pyramidal residual artifact off the 14E receipt.~~
   **Done 2026-05-15 → NEGATIVE** (see the Pyramidal record's
   "Quantitative probe" entry). The 14E 1M receipt has no resolvable
   discrete rings; no residual table was produced. The tooling
   ([`scripts/pyramidal_ring_residual.py`](../../scripts/pyramidal_ring_residual.py))
   is built and resolvability-gated — it will emit the residual table +
   46° linearity cross-check automatically once a sufficient receipt
   exists. Superseded by **#1′**.
1′. ~~Pyramidal residual via ray-filter isolation.~~ **Done 2026-05-15 →
   mechanically successful, quantitatively negative** (see the Pyramidal
   record's "Ray-filter-isolation campaign" entry). All three levers have
   now been exhausted on the 1D-azimuthal-profile method: ray count
   (1M→0, 6M→1) and per-wedge ray-filtering (8 frames, 4M → 1 strong
   ring). The face-pair filters crispen but do **not** single-isolate
   (6-fold crystal symmetry), and ≥3 azimuthally-separable rings still do
   not extract from the split-sky "Sun centered Plan" renders. The
   bottleneck is no longer rays/filtering — it is the extraction method
   + projection. Superseded by **#1″**.
1″. ~~Pyramidal residual via HaloSim-native angular Scale.~~ **Done
   2026-05-15 → method implemented, anchors fail, P2 unchanged** (see
   the Pyramidal record's "HaloSim-native angular-Scale method" entry).
   The fourth and final lever: HaloSim's own `Tools → Scale` instrument
   (degree ruler from the sun, FIX-stamped) was applied to all 8 wedge
   frames (`phase15_pyrfilter/pyr_w*_scale.png`,
   [`scripts/pyramidal_scale_read.py`](../../scripts/pyramidal_scale_read.py),
   hard anchor gate). The scale solves sun-Y/centring but its stamped
   span is shorter than the ring field (the 22°/46° anchors fall beyond
   the calibrated ruler tip) and 6-fold symmetry still prevents per-wedge
   isolation — anchors un-validatable, no table, **no fabrication**.
   **Disposition (recommended & adopted as the standing position):
   accept P2 as this evidence chain's ceiling.** Pyramidal is solidly
   P2 (P1 analytic Tape table + P2 qualitative HaloSim reproduction
   across 1M/6M/8×ray-filter/8×scale renders). P3 (atlas-grammar
   representability / a clean quantitative residual) requires a
   fundamentally different setup — a non-symmetric single-ring isolation
   or a 2-D ring-template fit on a non-split equidistant render with a
   full-span scale — and is **out of scope for this chain**. No further
   mechanical levers; this is a closed strategy decision, not a deferred
   task.
2. **Chat boundary refinement.** `chat/claim_map.json`
   `halo_atlas_vocabulary_status` currently frames pyramidal/CHA as
   "named-only / not modeled." It should be refined so Ask Sundog says
   "simulated / HaloSim-reproduced **candidate**, not a discovered halo,
   not a public result" for the P2 entries. **Dependency:** any
   `claim_map.json` change triggers the Phase 13 chat:eval gate
   (`chat:eval:static` / `:phase3` / `:phase3:adversarial` /
   `:phase3:differential` / `:phase4`) before deploy — do **not** edit
   `claim_map.json` until that gate is run. Tracked, not done here.
3. **Lowitz isolation render.** A crystal-block-reduced `Lowitz arcs.sim`
   variant (disable the random/22°-halo block, keep the Lowitz-orientation
   block + a `random.xng` scale reference) rendered B&W @ ≥1M, to move
   Lowitz from P1-pending-isolation to a clean P2.
