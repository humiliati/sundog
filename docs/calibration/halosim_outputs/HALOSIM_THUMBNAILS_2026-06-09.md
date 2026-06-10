# HaloSim cross-render thumbnails (public assets) — 2026-06-09

Eight high-ray-count colour halo renders maintained as **public thumbnail / hero assets**: the five
cross-validation renders below, plus three pyramidal novelty variants (see the later section). Each of
the five was chosen to **cross-validate** a forward-model result from the in-house polarized halo raytracer
(`scripts/s2_halo_raytracer.py`) against the independent apparatus, and rendered at press-grade ray
counts (`~10M`, the "spectacular thumbnail" tier in AGENTS.md ▸ HaloSim Ray-count guidance).

## Attribution (load-bearing)
> **The apparatus is not ours.** These images were produced with **HaloSim3** (`HalSim361.exe`, 2004),
> a Monte-Carlo ice-halo ray-tracer by **Les Cowley & Michael Schroeder** (Atmospheric Optics,
> atoptics.co.uk). We *ran* it on its own bundled crystal/sun recipe library; we did not build it. The
> ice physics (refractive index n = 1.31, hexagonal-ice habits, pyramidal {10-11} faces) is standard
> atmospheric optics (Greenler; Tape; Können; Cowley). No new physical claim is made by these renders.

## The five renders
| PNG | source recipe | sun alt | rays | shows / cross-validates |
|---|---|---|---|---|
| `halosim_pyramidal_odd_radius_10Mcolour.png` | `pyramidal crystal halos.sim` | 18.3° | **10M** (100%) | the **9/18/20/23/24/35° odd-radius (Galle) halos** — the raytracer's pyramidal result + the DoP(R) polarization ladder |
| `halosim_parhelic_circle_6Mcolour.png` | `Parhelic circ and more.sim` | 22° | 6.1M | the **parhelic circle + 120° parhelia + sundogs + tangent arcs** — the K=2 multi-bounce result |
| `halosim_circumzenithal_arc_10Mcolour.png` | `Circumzenithal arc.sim` | 26° | **10M** (100%) | the **CZA** — the 32.196° TIR admissibility wall (raytracer-confirmed) |
| `halosim_circumhorizon_arc_10Mcolour.png` | `Circumhorizon arc.sim` | 62° | **10M** (100%) | the **CHA** — the 57.804° complement TIR wall (CZA+CHA = 90°) |
| `halosim_st_petersburg_display_10Mcolour.png` | `St Petersburg display.sim` | 51° | **10M** (100%) | a **complete multi-halo display** — the whole classified Atlas in one sky (hero/logo material) |

All are 1536×864 colour ("Color shades on color"), zenith- or sun-centred fisheye / camera view as the
recipe specifies. The parhelic render auto-stopped at 6.1M rays (still well above the ~3M confirm tier;
the camera-view + multi-population recipe caps early); the other four reached the full 10M.

## Pyramidal novelty variants (second batch, same day)

Three additional **pyramidal odd-radius** renders at different settings, maintained as public thumbnails
alongside the zoomed hero above (`halosim_pyramidal_odd_radius_10Mcolour.png`). All show the
**9 / 18 / 20 / 23 / 24 / 35° odd-radius (Galle) halo family** produced by {10‑11} pyramidal ice faces.

| PNG | projection | sun alt | rays | character |
|---|---|---|---|---|
| `halosim_pyramidal_display_wide_10Mcolour.png` | sun‑centred **fisheye**, 1× | 20° | **10M** (100%) | the whole family in a wide all‑sky dome (horizon context; canvas field is HaloSim's plot background) |
| `halosim_pyramidal_plan_10Mcolour.png` | sun‑centred **plan** (flat disk) | 18.3° | **10M** (100%) | the same family as clean true‑circle concentric rings, full‑frame |
| `halosim_pyramidal_parhelia_2p2Mcolour.png` | sun‑centred **camera view** | 28° | **2.2M** (sweet spot) | the pyramidal **parhelia** — bright ring + parhelion nodes flanking the sun |

### Ray‑count is feature‑dependent (a real lesson, not a bug)
- **Camera‑view colour washes out at high ray counts.** The tight sun‑centred camera view concentrates the
  bright near‑sun features, so by ~3–5M the centre clips to white and bleaches the detail. The parhelia
  render is therefore taken at the **~2.2M sweet spot** (auto‑stop), *not* 10M — verified by eye against
  earlier washed frames. (The `.sim` ray token is **absolute**, e.g. `2200000`; the GUI "Rays '000" shows
  it ÷1000.)
- **Wide / plan views need ~10M.** The faint *outer* odd‑radius rings (20–35°) only rise above MC noise at
  high ray counts; the intensity is spread over the dome/disk so the centre does **not** wash. Dropping
  these to the parhelia sweet spot under‑exposes the outer family (collapses to just the inner ring).
- The **sun‑centred camera/fisheye view does not change with sun altitude** (the sun stays centred), so
  altitude variants of the full display are redundant — only the *projection* (zoom/fisheye/plan) varies
  the image. The 3.46× zoom + pan framing is baked into the hero recipe's view line.
- The intended *isolated* 20–35° view (ray‑filter on faces 13–18) renders in HaloSim **Split‑Sky** mode
  (a hard centre seam, left=full / right=filtered). HaloSim's **Ray Filters** panel was greyed/unavailable
  after a clean relaunch; the in‑`.sim` Full‑Sky toggle that removes the seam also disables the filter, so
  the clean substitute shipped here is the **plan‑view full family** rather than outer‑rings‑only.

## Provenance / reproducibility
- The exact 10M-ray `.sim` configs used are kept beside this file in `_thumbs_staging/*.sim` (each is a
  bundled HaloSim recipe with only the ray count bumped to 10,000,000 via `scripts/halosim_gen_frames.py`
  token logic; St Petersburg additionally switched to colour plot-style).
- Rendered via the proven **HS-0** mechanism (AGENTS.md): swap each `.sim` over `Startup.sim`, Reset/Start,
  poll `autosave.bmp` to settled completion, harvest, restore the user's `Startup.sim`.
- Cross-validation context: `docs/atlas/S2_HALO_RAYTRACER_RESULT.md`,
  `docs/atlas/ATLAS_HALO_POLARIZATION_OBSERVABLE.md`, `docs/atlas/ATLAS_ORIENTATION_BOUNDARIES.md`.
