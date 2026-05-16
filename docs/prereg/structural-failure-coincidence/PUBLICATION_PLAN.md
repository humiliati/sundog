# Structural Failure Boundary Map Publication Plan

Sources:
[`README.md`](README.md),
[`BOUNDARY_MAP.md`](BOUNDARY_MAP.md),
[`P2_CUT2_WAVE42_DISPOSITION.md`](P2_CUT2_WAVE42_DISPOSITION.md),
[`P2_CUT3_ADMISSION.md`](P2_CUT3_ADMISSION.md),
[`P2_CUT3_H0_CALIBRATION.md`](P2_CUT3_H0_CALIBRATION.md)

Public data contract: `public/data/structural-failure-boundary-map.json`
in the source tree, served as `/data/structural-failure-boundary-map.json`
after build.

Filed: **2026-05-16 (PT)**. Status: **publication plan, not a result**.
This document explains how to publish the structural-failure boundary map
without changing the frozen preregistration artifacts and without upgrading
the current scientific status.

## Core public claim

Lead with the apparatus:

> A traceable system should fail where the closed-form inverse loses
> identifiability.

That is the publishable object. The boundary map is not a theorem proof and
not an agent-pass claim. It is the falsifier that was written before the agent:
where the route is allowed to work, where it must fail/abstain/switch, and
where a mere correlate would keep reporting confidence.

## Current status to publish

- P0 boundary map: **passed**. The five loci L1-L5 are frozen with source
  receipts.
- P1 admission: **passed**. The falsifier is admitted as the right apparatus.
- P2 first cut: **reclassified** as machinery-live route-test-vacuous.
- Cut 2 closed-form line: **regime-separability finding**. The closed-form
  correlate cannot compete where the parhelion route is eligible; it only helps
  in the route's abstain region.
- Cut 3 rendered escalation: **opened, execution held**. The run is gated on H0
  angular calibration, corpus, baselines, edit operators, and renewed
  admission.
- H0 calibration: **open**. The checker design exists, but the real-frame
  negative-side measurement remains open after the Wave H0-1 correction.

This status should be visible anywhere the map appears. The safe label is:
**Frozen falsifier + closed-form regime-separability; rendered Cut-3 held.**

## Publication surfaces

1. **Canonical document.** Keep `BOUNDARY_MAP.md` as the source of truth. It is
   frozen and receipt-cited.
2. **Generated data.** Build
   `public/data/structural-failure-boundary-map.json` from
   `scripts/build-structural-boundary-data.mjs`. The JSON is the contract for
   charts, page cards, and future SVG exports.
3. **Post-rail evidence panel.** Replace the generic app-card placeholder with
   a compact five-locus map. The panel should say "Pre-registered falsifier,"
   not "confirmed."
4. **Expanded page, later.** Add a `structural-failure.html` page only after
   the first visual panel is stable. The page can hold the five-locus map, the
   status ladder, links to the prereg trail, and the public-language guard.
5. **About page, later.** Link the map as an example of Sundog's research
   posture: write the failure boundary first, then test against it.

## Visual spec

The first map should be a static SVG under `public/media/` generated from the
JSON contract. Avoid live canvas until mobile screenshots prove it behaves.

Rows:

- L1 Parhelion offset route.
- L2 CZA visibility cutoff.
- L3 Tangent arc merge.
- L4 Supralateral permanent fail.
- L5 Rendered is not anchored.

Columns:

- Handle.
- Eligible window.
- Must break.
- Correlate tell.

Treatment:

- Eligible route: restrained green.
- Abstain/switch/merge: amber.
- Permanent fail/invalid: red.
- Admissibility rule: neutral gray.

L5 should be visually separated from L1-L4 because P1 re-scoped it as an
evidence-admissibility rule, not a standalone behavioral locus.

## Safe copy

Use:

- "The falsifier before the agent."
- "A traceable system should fail where the closed-form inverse loses
  identifiability."
- "Frozen boundary map."
- "Failure-boundary coincidence."
- "Closed-form regime-separability finding."
- "Rendered Cut-3 remains held on H0/corpus/admission artifacts."

Avoid:

- "The theorem is proven."
- "Traceability is confirmed."
- "The agent passed."
- "The rendered-signal test passed."
- "A probe decoded it, therefore the route was used."

## Implementation steps

1. Land the generated JSON contract and wire it into `npm run build`.
2. Export `public/media/structural-boundary-five-locus-map.svg` from the JSON
   with `npm run structural:boundary-svg`.
3. Replace the relevant post-rail card placeholder with the SVG and a source
   trail to this prereg folder.
4. Add a small status ladder panel only if the card needs more context.
5. Consider `structural-failure.html` after the card is readable on mobile.

## Checks

Run:

```powershell
npm run structural:public-data
npm run structural:boundary-svg
npm run build
```

Before any homepage/panel publish, also run local screenshots at:

- 390 px mobile after the rail;
- 520 px mobile after the rail;
- 1280 px desktop grid;
- reduced-motion mode.

Phrase check before deploy:

```powershell
rg -n "theorem proved|traceability is confirmed|agent passed|rendered Cut-3 passed|probe decoded" index.html about.html applications-gallery.html mesa.html sundog.html
```

The expected result is no root-page claim that upgrades the current status.

## Exit criteria for first publish

- `BOUNDARY_MAP.md` remains the canonical frozen source.
- The public JSON exists and builds deterministically.
- The visual maps all five loci and separates L5 as an admissibility rule.
- The visible status says P0/P1 passed, first P2 was vacuous, closed-form Cut 2
  found regime-separability, and rendered Cut 3 is held.
- No public surface says theorem, confirmed traceability, agent pass, or
  rendered-signal pass.
