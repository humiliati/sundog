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
