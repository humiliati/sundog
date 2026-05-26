# Phase 14E — HaloSim Receipt Pass

Filed: 2026-05-15
Phase: Sundog V Geometry Phase 14E
Status: **11/11 reproduced** — every targeted `halosim-candidate` row now has a render receipt.

## What this is

For every halo-phenomena row that had a HaloSim `.sim` recipe but no
captured render receipt, this pass renders the recipe and confirms the
local HaloSim library reproduces the phenomenon. Receipts for
22°/46°/CZA/parhelion (`../hs0_spike/`) and the upper tangent arc (Pass C7,
`../halosim_tangent_p2_*`) already existed and were **not** re-rendered.

## Method

- **Rays-only pin.** Each library `.sim` at `%USERPROFILE%\` was copied
  through [`scripts/halosim_pin_rays.py`](../../../../scripts/halosim_pin_rays.py)
  into [`../phase14e_frames/`](../phase14e_frames), which edits **only** the
  T-3 ray-count token (300k–400k → **1,000,000**) and preserves every other
  byte — sun altitude, camera aim/azimuth, crystal blocks, view, filters —
  exactly as the library authored it. This is the faithful "shipped recipe,
  as-is" configuration. `gen_frames` was deliberately **not** used: several
  targets are Camera-View with the camera intentionally decoupled from the
  sun (e.g. subhorizon: sun 18.5°, camera aimed −11°; parhelic-circle: sun
  22°, camera-aim 25°), where its camera-sync would corrupt the framing.
- **Render.** Proven HS-0 mechanism: swap → `Startup.sim`, **Reset** (auto-
  reload), set plot style to **Grey shades on white** (B&W), **Start**,
  poll `autosave.bmp` mtime-advance + size-stable (never a fixed sleep),
  harvest → PNG. `Startup.sim` was backed up before the pass and restored
  byte-for-byte (sha256-verified) after.
- **Ray count: 1M, not 300k.** Per the user call (300k b&w is not reliable
  for clean geometry — Monte-Carlo noise); see AGENTS.md ▸ Ray-count
  guidance and `../../HALOSIM_VALIDATION_PROTOCOL.md`.
- **B&W note.** "Grey shades on white" is the B&W plot style. HaloSim's
  Camera-View / Fisheye projections render a simulated **sky background**
  (white for zenith-fisheye; blue for camera/plan views) over which the
  grey-shaded halo plot is drawn — this is inherent to the recipe's
  projection, not a setting error. The named feature is clearly visible in
  every receipt; that is the receipt criterion.
- **Crystal-block reduction: not needed.** No target was noise-drowned by
  unrelated halos. Multi-block displays (Parry, Lowitz) show the target
  feature clearly; Parry/Lowitz arcs are intertwined with the 22° halo /
  parhelia *by their physical nature*, not by render noise, so no block
  reduction was applied. The technique remains documented for future use.

## Receipts

All PNGs in this directory; 1536×864; converted from the harvested
`autosave.bmp` via PIL (BMP discarded — `*.bmp` is gitignored).
`L-std` = greyscale standard deviation (blank-render guard; all ≫ 1).

| receipt PNG | source library `.sim` | sun alt | view | rays | L-std | feature confirmed |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `supralateral_h22_1M_bw.png` | `Supralateral arc.sim` | 22° | camera (split-sky filtered) | 1M | 22.6 | supralateral arc band + 46° halo segment |
| `infralateral_h15_1M_bw.png` | `Infralateral arc.sim` | 15° | camera (split-sky filtered) | 1M | 21.8 | infralateral arc rising from horizon |
| `circumscribed_h45_1M_bw.png` | `Circumscribed halo.sim` | 45° | zenith fisheye | 1M | 32.2 | circumscribed halo (oval, high-sun tangent-arc merge) |
| `parhelic-circle_h22_1M_bw.png` | `Parhelic circ and more.sim` | 22° | camera | 1M | 66.9 | full parhelic circle through sun + parhelia + CZA |
| `sun-pillar_hneg3_1M_bw.png` | `Sun pillar.sim` | −3° | camera | 1M | 24.8 | vertical sun pillar (set sun) + subparhelia |
| `parry_h15_1M_bw.png` | `Parry arcs.sim` | 15° | camera | 1M | 56.6 | suncave/sunvex Parry arcs above the sun |
| `pyramidal_h18_1M_bw.png` | `Pyramidal 20-35d halos.sim` | 18.3° | sun-centred plan | 1M | 65.8 | concentric odd-radius rings (9/18/20/23/35°) |
| `lowitz_h30_1M_bw.png` | `Lowitz arcs.sim` | 30° | zenith fisheye | 1M | 54.0 | Lowitz upper/lower arcs threading the parhelia |
| `antisolar_h20_1M_bw.png` | `Anthelic Point display.sim` | 20° | camera (az 155, anthelic) | 1M | 50.1 | anthelic-point arc crossing + 120° parhelion |
| `subhorizon_h18-5_1M_bw.png` | `Subhorizon arcs.sim` | 18.5° | camera (aimed −11°) | 1M | 12.3 | subsun + subparhelia below the horizon |
| `circumhorizon_h62_1M_bw.png` | `Circumhorizon arc.sim` | 62° | camera | 1M | 19.2 | circumhorizon arc band low & parallel to horizon |

## Notes / honest caveats

- **Older `.sim` solar-diameter prompt.** `Parry arcs.sim` raised HaloSim's
  "Peculiar solar diameter" warning (1° vs the modern 0.5° earth value, a
  pre-3.6 file). Answered **Yes** (keep the recipe as authored) — solar
  diameter only sizes the sun disk and does not affect whether the Parry
  arcs reproduce.
- **Pre-Start verification is load-bearing.** During an active render
  HaloSim's control panel transiently displays stale defaults
  (sun 22 / random.xng); the *loaded* sim was verified by zoom after
  Reset/splash and before Start every frame. The rendered output (not the
  panel) is the receipt of record.
- These receipts confirm **reproduction** (the library renders the named
  phenomenon), not measurement. Quantitative locus extraction remains the
  Pass-C7 / HALOSIM_VALIDATION_PROTOCOL pathway.
