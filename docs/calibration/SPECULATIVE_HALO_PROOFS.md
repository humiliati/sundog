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
  advance it toward P3. Refined next step is now follow-up #1′ below.
  Evidence retained:
  [`halosim_outputs/phase14e/pyramidal_h18_radialprofile.txt`](halosim_outputs/phase14e/pyramidal_h18_radialprofile.txt)
  (radial profile + detrended SNR) and the script. Negative result kept
  as a map entry per the gate, not buried.

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
1′. **Higher-fidelity pyramidal render for the residual table.** A
   dedicated render that resolves the odd-radius family: **≥4–10M rays**
   and/or **per-ring HaloSim ray-filter isolation** (filter by entry/exit
   faces per the Tape AH-CH10 wedge table — 3–26 for 9°, 13–25 for 18.3°,
   etc.) and/or a tighter sun-centred FOV than the all-sky plan. Then
   re-run `scripts/pyramidal_ring_residual.py` (the gate passes → residual
   table + linearity cross-check + overlay). This is the real concrete
   probe toward pyramidal P3; the 1M 14E receipt was insufficient.
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
