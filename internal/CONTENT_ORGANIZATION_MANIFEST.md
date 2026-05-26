# Content Organization Manifest

Date: 2026-05-26

This manifest records the cleanup passes for the Sundog documentation tree. It
is intentionally internal: it can name staging folders, public/private boundary
questions, and next-pass candidates without turning those notes into a public
site artifact.

## Landed Passes

### Pass 0 - Migration Manifest And Script

- Added `scripts/organize-docs.mjs`.
- The script records the Faraday, Isotrophy, K_facet, and historical appendix
  move map.
- Public root-level document paths are kept as compatibility stubs when moved.
- The script is now safe to rerun after the first application: expected public
  stubs are treated as already moved.

### Pass 1 - Faraday Folder

Canonical location: `docs/faraday/`

Moved the Shadow Faraday roadmap, ledger, Phase 3 derivation receipt, Phase 4
verification receipt, Phase 7 spec, and small Phase 3 derivation pointer into a
single public folder.

Root compatibility stubs remain for:

- `docs/SHADOW_FARADAY.md`
- `docs/SUNDOG_V_FARADAY.md`
- `docs/FARADAY_PHASE3_DERIVATIONS.md`
- `docs/FARADAY_PHASE4_VERIFICATION.md`
- `docs/FARADAY_PHASE7_SPEC.md`
- `docs/SHADOW_FARADAY_PHASE3_DERIVATION.md`

### Pass 2 - Isotrophy And K_facet Folder

Canonical locations:

- `docs/isotrophy/`
- `docs/isotrophy/kfacet/`
- `docs/isotrophy/archive/`

Moved the Isotrophy roadmap, K_facet public ledger, promo handoff, K_facet
appendix set, and historical appendix notes out of the anniversary catch-all.

Root compatibility stubs remain for:

- `docs/sundog_v_isotrophy.md`
- `docs/SUNDOG_V_ISOTROPHY_KFACET.md`
- `docs/ISOTROPHY_PROMO_HANDOFF_2026-05-24.md`

### Pass 3 - Reference And TOC Repair

- Updated public page links, SVG metadata links, JSON data links, generated
  chat/public data, and script-held form-lock references.
- Added folder README files for Faraday, Isotrophy, and K_facet.
- Added Faraday and Isotrophy entries to `docs/README.md` and `docs/index.html`.

### Pass 4 - Verification

- `npm run build` succeeded.
- The build's dist link check passed.
- A moved-Markdown link probe over `docs/faraday` and `docs/isotrophy`
  reported `missing=0`.

### Pass 5 - Brand, Site, And Promo Folders

Canonical locations:

- `docs/brand/`
- `docs/site/`
- `docs/promo/`

Moved the public brand/legal/stress-test docs, site operations/assets/policy
docs, and promo/outreach docs out of the root docs shelf. Root compatibility
stubs remain for each moved public path.

Brand:

- `docs/brand/BRAND_POSITIONING.md`
- `docs/brand/BRAND_ROADMAP.md`
- `docs/brand/LEGAL_STANDING.md`
- `docs/brand/Mythos-Benchmark.md`
- `docs/brand/gemini-benchmark.md`

Site:

- `docs/site/WEBSITE_DEVELOPMENT.md`
- `docs/site/SEO_AND_SOCIAL_READINESS_ROADMAP.md`
- `docs/site/UI_UX_THEME_FOUNDATION.md`
- `docs/site/HIGHLIGHTS_RAIL_ROADMAP.md`
- `docs/site/ICON_ASSETS.md`
- `docs/site/LOGO_ANIMATION_TOOLKIT.md`
- `docs/site/PHOTO_DATA_POLICY.md`
- `docs/site/THIRD_PARTY_REUSE.md`

Promo:

- `docs/promo/PROMO_HIGHLIGHTS.md`
- `docs/promo/SUNDOG_OUTREACH_PACKET.md`

## Current Audit Notes

- Remaining `internal/anniversary` references in public docs are not left over
  from the Faraday/Isotrophy move. They point at the older postulation,
  analogy, attack-vector, fix-roadmap, and anniversary planning layer.
- `SCIENTIFIC_CRITERIA.md` needed a refresh after Faraday and `/safety-method`;
  that refresh belongs to the check pass, not a new folder migration.
- The brand and benchmark documents are still public, now under `docs/brand/`.
  A later pass can decide whether some of that material belongs under
  `internal/feedback/` instead.

## Next-Pass Candidates

### Feedback Corpus

Likely target shape:

- `internal/feedback/Human/` - current human critique and Reddit feedback.
- `internal/feedback/model/` - possible future private home for the public
  stress-test reports now living at `docs/brand/Mythos-Benchmark.md` and
  `docs/brand/gemini-benchmark.md`, if those reports should stop shipping with
  `docs/**`.
- `internal/feedback/README.md` - orientation over human/model critique,
  accepted changes, and rejected or quarantined interpretations.

### Brand And Legal/IP

Likely split:

- Public posture stays public only when it helps readers understand claims and
  boundaries.
- Legal/IP execution, contractor/CLA plans, and entity-formation details move
  toward `internal/brand/` or `internal/legal/`.
- Public pages that still need a brand anchor can point at a short public
  posture note instead of the full execution roadmap.

Candidate files have been moved into the public category folders. The remaining
decision is whether any of the public category docs should instead become
internal-only.

### Anniversary Catch-All

Likely split:

- `internal/theory/` for `postulations.md` and `analogies.md`.
- `internal/feedback/` for `attack_vectors.md`.
- `internal/roadmaps/` or `internal/outreach/` for `fix_roadmap.md`,
  `anni_spam_roadmap.md`, and `first_public_statement.md`.
- Keep `scratchpad_brainstorm_notes.md` quarantined unless a specific public
  artifact is extracted from it.

### Root Docs Category Folders

After public/private decisions, consider moving root-level public docs into
folders such as:

- `docs/brand/`
- `docs/site/`
- `docs/legal/`
- `docs/promo/`

Use public compatibility stubs for any page already linked from the site,
search, or social surfaces.

## Guardrails For Future Passes

- Move by manifest, not by vibes.
- Keep public compatibility stubs for public `docs/*.md` moves.
- Run `npm run build` after each applied migration.
- Re-run a Markdown link probe on moved folders.
- Treat generated `public/data/sundog-*` changes as expected after build, but
  inspect any unrelated generated-data churn before staging.
