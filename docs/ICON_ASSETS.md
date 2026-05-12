# Sundog Icon Assets

The Sundog site icon system uses a halo/parhelion mark:

- deep blue field for the research site background;
- gold sun for observable signal;
- outer halo and tangent arcs for the theorem animation;
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
| `public/icons/sundog-character-mark.svg` | Phase 11 characterized static SVG prototype. |
| `public/icons/sundog-character-mark.transparent.svg` | Phase 11 transparent SVG prototype. |
| `public/icons/sundog-character-mark.animated.svg` | Phase 11 SVG/CSS animation prototype. |
| `public/icons/sundog-character-mark.layers.json` | Phase 11 layer, geometry, color, and motion manifest. |
| `public/icons/sundog-character-favicon-16.png` | Phase 11 tiny favicon proof. |
| `public/icons/sundog-character-favicon-32.png` | Phase 11 favicon proof. |
| `public/icons/sundog-character-favicon-48.png` | Phase 11 browser icon proof. |
| `public/icons/sundog-character-icon-192.png` | Phase 11 app/PWA icon proof. |
| `public/icons/sundog-character-icon-512.png` | Phase 11 large app/social icon proof. |
| `public/icons/sundog-character-transparent-512.png` | Phase 11 transparent PNG proof. |

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

The PNG and ICO files were generated from the same geometry as
`public/icons/sundog-icon.svg` using a small standard-library Python renderer,
because the repo does not require ImageMagick or Pillow.

The characterized Phase 11 prototype set is regenerated with:

```bash
npm run logo:toolkit
```

That command uses `scripts/generate-sundog-logo-toolkit.mjs`, which writes
SVG, SVG/CSS animation, manifest JSON, and dependency-free PNG proofs.

## Characterized Logo Toolkit

`SUNDOG_V_GEOMETRY.md` Phase 11 turns this asset inventory into a
graphic-design handoff kit after the richer overlay tuning pass against
calibration images 2, 7, and 13. The first tuning notes live in
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
