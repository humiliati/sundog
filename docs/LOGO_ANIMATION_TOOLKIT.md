# Sundog Logo Animation Toolkit

This is the Phase 11 handoff kit for a characterized Sundog mark. It gives a
designer or motion pass a small source of truth without requiring the full
geometry roadmap.

The toolkit is deliberately bounded: the mark can feel alive, but it should
still trace back to the calibrated Halo Atlas overlay work.

## Regenerate

From the repository root:

```bash
npm run logo:toolkit
```

The command writes all current assets from:

```text
scripts/generate-sundog-logo-toolkit.mjs
```

## Outputs

| File | Use |
| --- | --- |
| `public/icons/sundog-character-mark.svg` | Static full-color source mark with background. |
| `public/icons/sundog-character-mark.transparent.svg` | Transparent SVG for lockups, slides, and dark surfaces. |
| `public/icons/sundog-character-mark.animated.svg` | Self-contained SVG/CSS idle animation with reduced-motion fallback. |
| `public/icons/sundog-character-mark.layers.json` | Layer, geometry, color, and motion manifest for design tools. |
| `public/icons/sundog-character-favicon-16.png` | Tiny favicon proof. |
| `public/icons/sundog-character-favicon-32.png` | Browser favicon proof. |
| `public/icons/sundog-character-favicon-48.png` | Windows/browser icon proof. |
| `public/icons/sundog-character-icon-192.png` | App/PWA icon proof. |
| `public/icons/sundog-character-icon-512.png` | Large app/social icon proof. |
| `public/icons/sundog-character-transparent-512.png` | Transparent PNG proof for decks and overlays. |

These files are not wired into `index.html` yet. They are Phase 11 design
prototypes that can replace the production favicon set after visual review.

## Character Sheet

Protected core:

- `core.sun`: observable source, centered at `(512, 512)`.
- `core.halo-22`: iris ring, radius `258`.
- `core.parhelia`: left and right glints at `x = 512 +/- 271.3`.
- `core.parhelic-belt`: sweep through the glints at `y = 499.1`, using the
  Phase 10 `-0.05 R22` belt-height correction.
- `core.upper-tangent`: the eyelid gesture; it may blink or reveal but should
  not detach from the halo system.

Optional, large-size only:

- `optional.circumzenithal-arc`: good for animated or presentation marks.
- `optional.suncave-parry`: annotation and education only for now.
- `optional.parry-supralateral`: annotation only for now.
- `optional.infralateral`: rich-display variant only; omit from favicons.

## Motion Rules

Idle:

- Slow sun breathing.
- Low-amplitude parhelic belt shimmer.
- Gentle parhelion glint pulse.

Active reveal:

- Draw the parhelic belt first.
- Light the left and right parhelia second.
- Bring in the upper tangent as the eyelid.

Hover:

- Raise parhelion opacity.
- Slide each glint outward by about `6-10 px`.
- Do not move the sun or the 22 deg halo.

Label callout:

- Pulse one `data-layer` group at a time.
- Use the IDs in `sundog-character-mark.layers.json`.

Reduced motion:

- Use the static SVG or freeze the animated SVG.
- No opacity cycling, shimmer, or glint translation.

## Small-Size Rules

| Size | Keep | Remove or simplify |
| --- | --- | --- |
| `16-32 px` | sun, 22 deg halo, parhelia | 46 deg halo, CZA, rich vocabulary arcs |
| `48-180 px` | sun, 22 deg halo, parhelia, simplified upper tangent | CZA unless contrast survives |
| `192 px+` | full default character mark | optional arcs only if they remain legible |

## Calibration Boundary

The default mark borrows from the Phase 10 p2/p7/p13 tuning pass:

- p2 supplies the strongest rich-display vocabulary.
- p7 keeps the mark honest under high-sun and one-sided visibility.
- p13 gives the square social-crop constraint.

The default logo should stay with stable layers: sun, 22 deg halo, parhelia,
parhelic belt, upper tangent, and optional CZA. Suncave Parry, Parry
supralateral, and infralateral arcs remain annotation or rich-display variants
until later calibration promotes them.
