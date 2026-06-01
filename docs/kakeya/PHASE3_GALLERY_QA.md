# Kakeya Phase 3 - Reward Graphic QA Note

- Artifact id: `KAK-PHASE3-REWARD-GRAPHIC-QA`
- Date: 2026-06-01
- Status: internal render QA pass for `kakeya/gallery.html` and
  `kakeya/kakeya-gallery.js`.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Core QA: [`PHASE3_WORKBENCH_QA.md`](PHASE3_WORKBENCH_QA.md)

## Verdict

**Internal render QA pass.** The reward graphic renders the body-resistance
continuum and the finite-field instance gallery from the verified
`kakeya-core.js` logic. Click-to-inspect updates the selected stage. The page
remains internal and non-deployed: no root `kakeya.html`, no `site-pages.json`
entry, and no public launch claim.

## What Was Checked

Commands:

```powershell
node --check kakeya\kakeya-gallery.js
node --check kakeya\kakeya-core.js
npm run kakeya:test
```

`npm run kakeya:test` reported:

```text
KAKEYA_WORKBENCH_TESTS q={5,7,11} pass=33 fail=0
```

Gallery data generated from the verified core:

| Panel | q | \|K\| | Directions | Complete | Dvir-consistent |
| --- | ---: | ---: | ---: | --- | --- |
| Single line | 7 | 7 | 1 / 8 | no | n/a |
| Random half | 7 | 25 | 1 / 8 | no | n/a |
| Greedy cover | 7 | 31 | 8 / 8 | yes | yes |
| Whole plane minus one | 7 | 48 | 8 / 8 | yes | yes |
| Greedy cover | 5 | 17 | 6 / 6 | yes | yes |
| Greedy cover | 11 | 77 | 12 / 12 | yes | yes |

## Browser Render QA

The page was checked through a one-off local static server and headless Chrome:

```text
http://127.0.0.1:5181/kakeya/gallery.html
```

Viewport checks:

| Viewport | Result |
| --- | --- |
| Desktop `1366 x 900` | Pass |
| Mobile `390 x 844` | Pass |

Observed browser facts in both viewports:

- page title: `Kakeya Body-Resistance Gallery (internal)`;
- body-resistance continuum SVG rendered with Faraday, marginal substrates,
  Aharonov-Bohm, and Kakeya markers;
- default stage is `Greedy cover - q=7`, with `|K| = 31 / 49`,
  `directions covered = 8 / 8`, and Dvir consistency marked true;
- gallery contains six finite-field panels;
- clicking `Random half` updates the stage to `|K| = 25 / 49`,
  `directions covered = 1 / 8`, verdict `near miss`;
- required warning is present: finite-field reader graphic, not Euclidean
  Kakeya evidence, not maximal-function, not regime-2;
- footer labels the direction fan as schematic and not literal Euclidean
  angles;
- no console errors;
- no page-level horizontal overflow.

Responsive adjustment made during QA: the body-resistance continuum keeps a
fixed readable SVG width inside a horizontally scrollable card on narrow
screens, avoiding page-wide overflow while preserving the axis labels.

Temporary screenshots were written under `tmp/kakeya-qa/` during this local QA
pass:

- `tmp/kakeya-qa/gallery-desktop-full.png`
- `tmp/kakeya-qa/gallery-mobile-full.png`

Those screenshots are local QA scratch artifacts, not public assets.

## Public Gate

This QA note does not authorize public promotion. A live but unlinked
`kakeya.html` review surface is allowed only if it carries a visible
`NOT PEER REVIEWED` header and has no obvious public inbound links. Public
promotion still requires:

- external incidence/combinatorics sanity review;
- a real root `kakeya.html` page-copy audit against the reader fences;
- Bucket 1 SEO/social readiness if a page is added to `site-pages.json`.
