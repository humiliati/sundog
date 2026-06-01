# Kakeya Phase 3 - Workbench QA Note

- Artifact id: `KAK-PHASE3-WORKBENCH-QA`
- Date: 2026-06-01
- Status: internal QA pass for the non-deployed workbench under repo-root
  `kakeya/`.
- Ledger: [`../SUNDOG_V_KAKEYA.md`](../SUNDOG_V_KAKEYA.md)
- Spec:
  [`PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md)

## Verdict

**Internal Phase 3 QA pass.** The core acceptance tests pass, the site build is
clean, and local headless browser QA shows the internal workbench loads at
desktop and mobile widths with the required warning, body/shadow separation,
and no console errors.

This does **not** launch Kakeya publicly. There is still no root `kakeya.html`,
no `site-pages.json` entry, and no public claim. A future live but unlinked
`kakeya.html` may be used as a review surface only with a visible
`NOT PEER REVIEWED` banner; public inbound links and launch claims remain gated
on external incidence/combinatorics sanity review plus page-copy and SEO/social
readiness if a public page is later added.

## Commands Run

```powershell
npm run kakeya:test
npm run build
```

`npm run kakeya:test` reported:

```text
KAKEYA_WORKBENCH_TESTS q={5,7,11} pass=33 fail=0
```

`npm run build` completed and the postbuild `dist` link check passed.

## Browser QA

The workbench was verified with a one-off local static server and headless
Chrome because the local Vite dev server hung during dependency scanning in
this session. This affects the dev-server route used for QA, not the
workbench's static files or core logic.

Checked URL:

```text
http://127.0.0.1:5180/kakeya/workbench.html
```

Viewport checks:

| Viewport | Result |
| --- | --- |
| Desktop `1366 x 900` | Pass |
| Mobile `390 x 844` | Pass |

Observed browser facts in both viewports:

- page title: `Kakeya Finite-Field Workbench (internal)`;
- default field size: `q = 7`;
- point grid: `49` cells;
- direction list: `8` rows;
- required warning present: finite-field toy, not Euclidean Kakeya evidence,
  not maximal-function, not regime-2;
- body panel and direction-shadow panel present;
- no horizontal overflow;
- no console errors;
- metric-label scan did not find forbidden theorem-facing labels outside the
  boundary warning.

Temporary screenshots were written under `tmp/kakeya-qa/` during this local QA
pass:

- `tmp/kakeya-qa/kakeya-desktop-full.png`
- `tmp/kakeya-qa/kakeya-mobile-full.png`

Those screenshots are local QA scratch artifacts, not public assets.

## Exit-Criterion Readback

Per the Phase 2 spec, Phase 3 may be considered internally complete when:

1. all acceptance tests pass for `q in {5, 7, 11}` - **pass**;
2. the primary shadow export passes the no-reencoding guard - **pass via
   acceptance test T10 and core inspection**;
3. the UI includes the required finite-field boundary warning - **pass**;
4. baseline labels avoid extremality and Euclidean language - **pass**;
5. screenshot or local QA records desktop/mobile visibility without overlap -
   **pass via this note**.

Public promotion remains blocked on the separately named public gates.
