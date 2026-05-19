# Sundog Icon Assets

The Sundog site icon system now uses the characterized halo/parhelion mark
from the logo toolkit:

- deep blue field for the research site background;
- gold sun for observable signal;
- outer halo and tangent arcs for the geometry animation;
- left/right parhelion glints for indirect alignment without direct sight.

## Files

| File | Use |
| --- | --- |
| `public/favicon.svg` | Modern browser SVG favicon. |
| `public/favicon.ico` | Legacy multi-size favicon. |
| `public/apple-touch-icon.png` | iOS home-screen icon. |
| `public/site.webmanifest` | PWA/app metadata and Android icon links. |
| `public/icons/sundog-icon.svg` | Source vector mark. |
| `public/icons/icon-16.png` | Small favicon PNG. |
| `public/icons/icon-32.png` | Small favicon PNG. |
| `public/icons/icon-48.png` | Windows/browser icon size. |
| `public/icons/icon-180.png` | Apple touch source size. |
| `public/icons/icon-192.png` | Android/PWA icon. |
| `public/icons/icon-512.png` | Large Android/PWA icon. |
| `public/icons/maskable-192.png` | Android maskable icon. |
| `public/icons/maskable-512.png` | Android maskable icon. |
| `public/icons/sundog-character-mark.svg` | Characterized static SVG source/proof. |
| `public/icons/sundog-character-mark.transparent.svg` | Transparent SVG proof for lockups and slides. |
| `public/icons/sundog-character-mark.animated.svg` | SVG/CSS animation prototype. |
| `public/icons/sundog-character-mark.layers.json` | Layer, geometry, color, and motion manifest. |
| `public/icons/sundog-character-favicon-16.png` | Tiny favicon proof. |
| `public/icons/sundog-character-favicon-32.png` | Favicon proof. |
| `public/icons/sundog-character-favicon-48.png` | Browser icon proof. |
| `public/icons/sundog-character-icon-180.png` | Apple touch icon proof. |
| `public/icons/sundog-character-icon-192.png` | App/PWA icon proof. |
| `public/icons/sundog-character-icon-512.png` | Large app/social icon proof. |
| `public/icons/sundog-character-transparent-512.png` | Transparent PNG proof. |
| `public/icons/sundog-pixel-chibi-favicon-16.png` | Optional pixel-chibi favicon proof. |
| `public/icons/sundog-pixel-chibi-favicon-32.png` | Optional pixel-chibi favicon proof. |
| `public/icons/sundog-pixel-chibi-favicon-48.png` | Optional pixel-chibi browser icon proof. |
| `public/icons/sundog-pixel-chibi-preview-512.png` | Nearest-neighbor review preview for the pixel-chibi proof. |
| `public/icons/sundog-pixel-chibi-favicon.ico` | Optional multi-size ICO proof. |
| `public/icons/sundog-pixel-chibi.layers.json` | Pixel-chibi layer, palette, and source-mapping manifest. |

## HTML Integration

These tags are wired into `index.html`:

```html
<link rel="icon" href="/favicon.ico" sizes="any">
<link rel="icon" type="image/svg+xml" href="/favicon.svg">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="manifest" href="/site.webmanifest">
<meta name="theme-color" content="#1A3A52">
```

## Regeneration

The toolkit proof set is regenerated with:

```bash
npm run logo:toolkit
```

The optional pixel-chibi favicon proof set is regenerated with:

```bash
npm run logo:pixel-chibi
```

The production favicon/app-icon set is promoted from the same geometry with:

```bash
npm run logo:promote
```

Both commands use `scripts/generate-sundog-logo-toolkit.mjs`, which writes SVG,
SVG/CSS animation, manifest JSON, dependency-free PNG proofs, and
PNG-in-ICO favicon output without ImageMagick or Pillow.

`logo:pixel-chibi` uses the same dependency-free PNG and ICO encoder, but keeps
its files under `public/icons/sundog-pixel-chibi-*`. It is a review lane for an
original early-web forum-avatar pixel treatment of the Sundog mark, not a
production replacement.

## Characterized Logo Toolkit

`SUNDOG_V_GEOMETRY.md` turns this asset inventory into a graphic-design
handoff kit after the richer overlay tuning pass against calibration images
2, 7, and 13. The first tuning notes live in
`docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md`; the design handoff lives in
[`LOGO_ANIMATION_TOOLKIT.md`](LOGO_ANIMATION_TOOLKIT.md).

The toolkit gives designers a small bounded surface:

- source SVG anatomy tied to Halo Atlas primitives;
- protected geometry rules for the sun, 22° halo, parhelia, and eyelid arcs;
- optional vocabulary layers for suncave Parry, Parry supralateral, and
  infralateral arcs;
- static export targets for favicon, app icon, social/avatar, and transparent
  logo marks;
- animation states for idle shimmer, active reveal, hover shimmer, and
  reduced-motion fallback.

The first generated character mark uses the Phase 10 belt-height correction:
the parhelic belt and parhelia sit at `-0.05 R22` relative to the sun, while
the 22Â° and 46Â° halo registrations stay centered on the sun.

The design rule is: characterize the Sundog mark from calibrated sky
morphology first, decorative invention second.
