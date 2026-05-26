# Hero Rail Artwork Inventory

Status: active, 2026-05-16.
Scope: public homepage application rail, logo/favicon system, and near-term
anniversary screenshot needs.

## Summary

The rail behavior is in good shape for the anniversary pass: center-focus
carousel, stamp-cued auto-cycle, manual controls, keyboard stepping,
drag/wheel takeover, replay, and reduced-motion fallback all have live code in
`public/js/motion-rail.mjs`. The visual evidence gap is now smaller: Balance,
Photometric Alignment, EyesOnly, Dungeon Gleaner, and Money Bags have real
poster images in `public/media`; Three-Body and Pressure Mines still rely on
bespoke CSS visuals.

The logo roadmap is ready to promote: the characterized halo mark already has
SVG and PNG proofs. Production favicons were still using the older mark until
this pass.

## Current Rail Cards

| card | current visual | asset state | next artwork action |
| --- | --- | --- | --- |
| Sundog Balance | Real poster image. | `public/media/balance-phase10-rail-poster.jpg` exists. | Keep for launch; later rename to non-phase asset path if we do a broader media cleanup. |
| Three-Body Dynamics | Bespoke CSS local-probe/orbit visual. | No poster or clip. | Capture a near-escape pocket screenshot or short clip from the later-trials workbench. Candidate path: `public/media/threebody-later-trials-rail-poster.jpg`. |
| Photometric Alignment | Real MuJoCo laser screenshot. | `public/media/photometric-alignment-rail-poster.png` exists. | Keep for launch; optional later crop/compress pass can tighten framing. |
| Pushable Occluder | Commented-out interrupt card. | Planned poster path does not exist and is no longer embedded as a commented image URL. | Do not activate until the boundary clip/poster exists and the owning roadmap has evidence. Candidate path remains `public/media/pushable-occluder-poster.jpg`. |
| Pressure Mines | Bespoke CSS pressure heatmap/grid visual. | No poster or clip. | Capture a board screenshot showing the confirmed pocket plus visible failure/boundary language. Candidate path: `public/media/pressure-mines-rail-poster.jpg`. |
| EyesOnly / Gone Rogue | Real constellation/card screenshot. | `public/media/eyesonly-rail-poster.png` exists. | Keep stamp at PLAUSIBLE until matched baselines exist; screenshot QA still owed. |
| Dungeon Gleaner | Real gameplay/UI screenshot. | `public/media/dungeon-gleaner-rail-poster.png` exists. | Keep for launch; screenshot QA still owed. |
| Money Bags | Real softbody telemetry plot. | `public/media/money-bags-rail-poster.png` exists. | Keep for launch; replace with a gameplay-plus-telemetry composite if one becomes available. |

## Screenshot QA Needed

Before the anniversary publish, take screenshots at:

- 390 px wide mobile.
- 520 px wide large mobile.
- 1280 px desktop.
- Reduced-motion enabled.

Check:

- The active card is visually dominant and neighbors read as peeks.
- Stamps do not cover the CTA or make status text unreadable.
- Replay, previous, and next controls are reachable on mobile.
- No text wraps outside card or button bounds.
- CSS placeholders read as intentional placeholders, not broken media.

Latest smoke pass, 2026-05-17:

- `npm run build` passed, including the dist link check.
- Headless Chrome checks at 390 px, 520 px, and 1280 px found seven active rail
  cards, no horizontal page overflow, reachable controls, injected verdict
  stamps, and no text escaping card bounds.
- Keyboard `ArrowRight` moves the rail from Sundog Balance to Three-Body
  Dynamics and sets `data-rail-state="user"`.
- Wheel interaction on the rail sets `data-rail-state="user"` and arms the
  current card's stamp, so the auto-cycle does not fight manual exploration.
- Reduced-motion emulation at 520 px sets `data-rail-state="settled"` and arms
  all seven stamps at load.
- Offscreen lazy poster images can remain unloaded until their cards approach
  the viewport; this is expected and separate from broken-link checks.

## Rail Implementation Notes

- `public/js/motion-rail.mjs` injects accessible verdict stamps and keeps the
  visible stamps decorative with screen-reader verdict text.
- Keyboard navigation now supports `ArrowLeft`, `ArrowRight`, `Home`, and
  `End`. Wheel or pointer takeover also switches the rail to user-driven mode
  for the rest of the page session.
- The roadmap says `data-media` duration should override `data-clip-ms`, but
  the current script still uses `data-clip-ms` or the static default. This is
  harmless while all active cards are static, but it must be fixed before
  adding video clips.
- The active Pushable Occluder markup is intentionally commented out. The
  poster path is now named only in this inventory, not embedded as a commented
  `<img>` in `index.html`, so local asset searches do not report it as a false
  broken image.

## Newly Supplied Source Images

Source folder: `assets/images/`.

| source image | rail use |
| --- | --- |
| `mujocov1 file-TbTB7TLJTfLXnmkQPrVruK-sundog laser popcorn working.png` | Promoted to `public/media/photometric-alignment-rail-poster.png`. |
| `eyesonly constellation.png` | Promoted to `public/media/eyesonly-rail-poster.png`. |
| `dungeon gleaner background.png` | Promoted to `public/media/dungeon-gleaner-rail-poster.png`. |
| `mujoco results graph file-LhZiGRrd24SGUKnsmMse6P-sundog_signature_stringfield.png` | Held as supporting evidence art; not used as the rail poster because graph labels are likely too small in card crop. |
| `aquarium controller.png` | Held unassigned; visually strong, but not yet mapped cleanly to a specific rail card. |

## Additional Local Source Images

| source image | rail use |
| --- | --- |
| `../Money Bags/Money Bags/playtest/2026-04-30_130419_03_gmtk_2024_port_v_tumble__Foxtrot_Slope45__hf74276c8/plot_4up.png` | Promoted to `public/media/money-bags-rail-poster.png`. |

## Logo State

The existing logo roadmap's next step is promotion, not invention. The
characterized mark has already passed a quick visual check at 512 px and 32 px:
it is more distinctive than the older mark and still reads at favicon size.

Promotion rules:

- Regenerate proofs with `npm run logo:toolkit`.
- Promote live favicons and app icons with `npm run logo:promote`.
- Keep `public/icons/sundog-character-mark.layers.json` as the design handoff
  source of truth.
- Use the static SVG for production favicon; reserve the animated SVG for
  deliberate page surfaces only.
