# Sundog SEO and Social-Readiness Roadmap

Working hook:

> A LinkedIn preview is a Sundog UI surface. It either honours the
> claim-hygiene posture or it doesn't.

Status: 2026-05-21. Filed in response to the OpenAI unit-distance
disproof wave (`/capset` shipped same day) and the unsurfaced fact that
most public pages had no OG metadata, no JSON-LD, and no shareable
preview image. This roadmap tracks the per-page work needed to make
every shareable URL render as a Sundog-discipline preview rather than a
bare title-and-favicon fallback.

This is a roadmap, not a marketing playbook. It does not say what to
post; it says what each page needs to render correctly when *someone
else* posts it.

## Story Shape

LinkedIn and the rest of the OG-consuming web cache previews
aggressively — typically 7 days, sometimes longer. The cost of an
unprepared first share is the cost of that share's preview being shown
to everyone who sees it in feed for the following week. Anecdotally
the engagement gap between a no-image preview and a designed
`summary_large_image` preview is ~3×. The discipline gap is even
larger: a Sundog page rendering as a bare text card reads as a casual
project, not as a research lab.

This work was triggered by the 2026-05-20 OpenAI unit-distance
disproof. Within hours, `/capset` shipped with full OG metadata, a
designed 1200×630 card, and a JSON-LD `TechArticle` schema. By the
end of 2026-05-21, the seven other most-shareable pages (`/`, `/about`,
`/alignment`, `/balance`, `/threebody`, `/mines`, `/sundog`) had
matching treatments. This roadmap exists to (a) document the per-page
state of that work, (b) finish what's left at a calmer pace, and (c)
prevent the next wave from catching the site flat-footed.

## Claim Boundary

This roadmap does **not**:

- prescribe content for any LinkedIn, Twitter/X, or Bluesky post;
- promise traffic, engagement, or ranking outcomes;
- license title or H1 changes that drift away from the
  claim-hygiene language in
  [`presentation/claims-and-scope.md`](../presentation/claims-and-scope.md);
- supersede [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) on
  visual-language decisions — the OG card series inherits from the
  paper-theme tokens documented there;
- supersede [`SUNDOG_OUTREACH_PACKET.md`](../promo/SUNDOG_OUTREACH_PACKET.md)
  on specialist-tier review channels — that is a different audience.

What this roadmap *does* track:

1. per-page presence of OG / Twitter / JSON-LD metadata,
2. per-page existence of a designed 1200×630 preview image,
3. per-page tuning of `<title>` and `<meta name="description">` for
   search-intent fit (within claim-hygiene limits),
4. internal-link equity from `/` and positioning pages,
5. a small standing checklist for any new public-share-class page added
   to `site-pages.json` after this filing.

## Page Classes

Public HTML pages on sundog.cc fall into four classes for this work.
The class determines whether a page needs an OG card at all, and how
much title/description tuning is permitted.

| Class | Definition | OG treatment |
| --- | --- | --- |
| **A. Public-share** | Pages an external reader is likely to paste into LinkedIn or a slide. Home, positioning, workbenches with public claims, atlas. | Full OG block + designed 1200×630 image + JSON-LD. |
| **B. Public-share, planned** | Pages with public-share value but not yet treated. Each entry below in this class has a named promotion gate. | Same target as Class A; tracked in this roadmap until done. |
| **C. Defer** | Pages that are public but unlikely to drive standalone shares (reference pages, redirect-style pages, interactive widgets that read poorly out of context). | OG block with a generic Sundog card + minimal JSON-LD; no per-page designed image. |
| **D. Internal-only** | Pages whose existence is operational, not promotional (design demos, repo navigation maps, aliases). | No OG metadata; `<meta name="robots" content="noindex">` recommended if discoverable. |

A page can be promoted from Class C to Class A if its share value
changes (e.g., gets featured in an external citation). The
classification is editorial, not technical.

## Readiness Matrix

State as of 2026-05-21 PM. Read across each row; columns are listed
under the legend below.

Legend: ✓ = done · ◐ = partial · ✗ = missing · — = N/A for the page's class

| Page | Class | OG block | OG image | JSON-LD | Title tuned | Desc tuned | Rail-linked | Sitemap | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `/` (index.html) | A | ✓ | home.png | WebSite | ✓ | ✓ | — (is rail) | ✓ | |
| `/about` | A | ✓ | about.png | AboutPage | ✓ | ✓ | header nav | ✓ | |
| `/alignment` | A | ✓ | alignment.png | Article | ✓ | ✓ | header nav | ✓ | |
| `/sundog` | A | ✓ | atlas.png | TechArticle | ✓ | ✓ | header nav | ✓ | replaced legacy `p0.overlay.png` OG image |
| `/balance` | A | ✓ | balance.png | TechArticle | ✓ | ✓ | rail card | ✓ | |
| `/threebody` | A | ✓ | threebody.png | TechArticle | ✓ | ✓ | rail card | ✓ | |
| `/mines` | A | ✓ | mines.png | TechArticle | ✓ | ✓ | rail card | ✓ | |
| `/capset` | A | ✓ | capset.png | TechArticle | ✓ | ✓ | rail card + contextual links | ✓ | first to ship; Post-Inspector confirmed; B2.3 added links from `/about`, `/alignment`, and `/applications-gallery` 2026-05-22 |
| `/applications-gallery` | A | ✓ | applications-gallery.png | Article | ✓ | ✓ | header nav | ✓ | promoted from B 2026-05-21 PM; viz is a 3×2 tier-card grid plus Cap-Set + "rest of the gallery" placeholder |
| `/h-of-x` | A | ✓ | h-of-x.png | TechArticle | ✓ | ✓ | pillar card | ✓ | promoted from B 2026-05-21 PM; pillar card added to `Load-Bearing Evidence` 2026-05-21 PM (late). Per-node clickable equation + parhelion sketch. |
| `/mesa` | A | ✓ | mesa.png | TechArticle | ✓ | ✓ | pillar card | ✓ | promoted from B 2026-05-21 PM; pillar card refactored to per-region clicks (hold/cliff/breach/locus) 2026-05-21 PM (late). |
| `/structural-failure` | A | ✓ | structural-failure.png | TechArticle | ✓ | ✓ | pillar card | ✓ | promoted from B 2026-05-21 PM; pillar card refactored to per-node clicks (P0/P1/Cut2/Cut3/apparatus) 2026-05-21 PM (late). |
| `/isotrophy` | A | ✓ | isotrophy.png | TechArticle | ✓ | ✓ | sitemap + homepage pillar | ✓ | filed 2026-05-22 with full Class A treatment; broadened 2026-05-25 from the v0.3h public companion to the v0.3-v0.9 isotrophy audit-chain pause page. Viz remains the v0.3h 21-cell catalog grid (20 gold structural-zero receipts + 1 dashed-red quarantine for O_617), which is still the first load-bearing pillar. New copy also carries the restricted-domain v0.7 Floquet velocity-fraction positive and pause boundary. Sister ledger at [`SUNDOG_V_ISOTROPHY_KFACET.md`](../isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md). |
| `/geometry` | A | ✓ | geometry.png | TechArticle | ✓ | ✓ | sitemap | ✓ | added to matrix 2026-05-21 PM (final); page exists at 774 lines but wasn't tracked. Viz is a 3-tile workbench shelf (cap-set + halo + h(x)). No homepage pillar yet — geometry is a hub for the existing pillars rather than a peer. |
| `/sundog-workbench` | C | ✓ (generic) | sundog-generic.png | TechArticle | ✓ | ✓ | — | ✓ | shared generic OG card 2026-05-21 PM (final); being absorbed into index hero per UI_UX 4a so per-page bespoke deferred indefinitely. |
| `/chat` | C | ✓ (generic) | sundog-generic.png | Article | ✓ | ✓ | header nav | ✓ | shared generic OG card 2026-05-21 PM (final); interactive widget, page-specific title/desc cover the trace-conditioned chat experiment hook. |
| `/legend` | C | ✓ (generic) | sundog-generic.png | Article | ✓ | ✓ | footer | ✓ | shared generic OG card 2026-05-21 PM (final). |
| `/origin` | C | ✓ (generic) | sundog-generic.png | Article | ✓ | ✓ | footer | ✓ | shared generic OG card 2026-05-21 PM (final); promote to bespoke if origin story gets shared externally. |
| `/paper-theme-demo` | D | — | — | — | — | — | — | — | internal design-system demo; `noindex, follow` added 2026-05-21 PM (final). Not in sitemap. |
| `/repo-map` | D | — | — | — | — | — | — | — | internal navigation map; `noindex, follow` added 2026-05-21 PM (final). Removed from sitemap 2026-05-21 PM (final). |
| `/atlas` | D | — | — | — | — | — | — | ✗ | alias / redirect to `/sundog`; `noindex, follow` added 2026-05-21 PM (final). |
| `/safety-method` | B | ✓ | safety-method.png | TechArticle | ✓ | ✓ | `/index` → `/faraday` → `/safety-method.html` | ✓ | SKETCH method-essay filed 2026-05-25 as the bridge between the `/faraday` Branch A receipt and three load-bearing translations to AI behavioral guarantees (local gauge-invariant readout, structural zero, named quarantine). Indexable (no `noindex`) but page carries a visible SKETCH banner so claim hygiene is preserved while the translation is evaluated. Local Bucket 1 artifacts are now present: bespoke 1200×630 `og:image`, full OG/Twitter metadata, JSON-LD `TechArticle`, tuned title/description, sitemap entry, clean-url redirect, and the inbound thread from `index.html` through `/faraday` to `/safety-method.html`. Post-deploy validator pass and the staged compander citation ratchet remain owed before an external push. Anchored to [`SHADOW_FARADAY.md#phase-5-chapter-close`](../faraday/SHADOW_FARADAY.md#phase-5-chapter-close) as the worked example. The three numbered claims are each paired with a falsifiable next experiment, and the new good-faith hooks keep credit, proof scope, and falsifier discipline separated. |
| `/faraday` | A | ✓ | faraday.png | TechArticle | ✓ | ✓ | homepage pillar + sitemap | ✓ | Public Shadow Faraday Zero-Out evidence page promoted 2026-05-25 after [`SUNDOG_V_FARADAY.md`](../faraday/SUNDOG_V_FARADAY.md) v0.1 closed Phases 1–5. Receipts live at [`SHADOW_FARADAY.md`](../faraday/SHADOW_FARADAY.md): Phase 3 landed Branch A in [`FARADAY_PHASE3_DERIVATIONS.md`](../faraday/FARADAY_PHASE3_DERIVATIONS.md), Phase 4 landed 5/5 in [`FARADAY_PHASE4_VERIFICATION.md`](../faraday/FARADAY_PHASE4_VERIFICATION.md) via `npm run faraday:phase4`, and Phase 5 chapter close landed in [`SHADOW_FARADAY.md#phase-5-chapter-close`](../faraday/SHADOW_FARADAY.md#phase-5-chapter-close) with receipts catalog, seven scope limitations, six suggested next minimal extensions, and a fidelity audit. Local Bucket 1 artifacts are present: designed 1200×630 `og:image`, full OG/Twitter metadata, JSON-LD `TechArticle`, tuned title/description, homepage internal link, sitemap entry, clean-url redirect, and onward bridge to `/safety-method`. Post-deploy LinkedIn/Twitter validator pass remains to record after deployment. |
| `/navierstokes` | A | ✓ | navierstokes.png | TechArticle | ✓ | ✓ | seventh load-bearing pillar + generality umbrella + sitemap | ✓ | Filed 2026-05-30 alongside [`/generality`](generality). The Reading-2 witness deep dive for the Kolmogorov-flow C1 cell: G=200 + G=300 replication under a portable objective, 942,834 G=300 witness pairs, composed at matching `ε_K = 0.0664`. Bespoke 1200×630 `og:image` shows two regime boxes + 12 witness-pair beads + a prominent `REVIEW-GATED · C1 UNPROMOTED` badge so the Standing-Rule discipline survives social cards. Page carries five named falsification modes (Front-A vacuity/miscalibration, Front-B vacuity/reach, coupling overreach) and four claim-boundary cards (not Clay, not NSE smoothness, not energy-method replacement, determining modes are not Sundog's idea). Standing claim boundary: this is a witness, not a Navier-Stokes existence or smoothness claim. Cross-ref from [`SUNDOG_V_NAVIERSTOKES.md`](../SUNDOG_V_NAVIERSTOKES.md) and from the seventh load-bearing pillar on the homepage. Post-deploy LinkedIn/Twitter validator pass remains to record after deployment. |
| `/generality` | A | ✓ | generality.png | TechArticle | ✓ | ✓ | seventh pillar + `/h-of-x` + geometry/alignment/legend/repo-map + sitemap | ✓ | Filed 2026-05-30 as the umbrella over six high-stakes transfer lanes: NSE C1 (review-gated witness, strongest current lane — links out to [`/navierstokes`](navierstokes) for the deep dive), Riemann (3 lanes, 3 bounded nulls, review-gated), Yang-Mills (4 probes, 1 bounded null, Phase 2 NEG-A across the slate), P-vs-NP (safety green, cost-hold on v5), ARC-AGI (execution-hold after sparse absolute fibers and oracle-leakage findings), Three-Body 15C (running, 6/12 shards logged). Bespoke 1200×630 `og:image` shows a 6-tile mini-matrix with NSE prominent (bolder border) and P-vs-NP red (cost-hold differentiated). Carries the asymmetric framing explicitly — NSE has its own page, the other five are honest cards on the matrix — and now includes a site-route bridge from `/h-of-x` first-equation discipline to harder transfer-test substrates. Four claim-boundary disavowals: not a Clay-problem claim, not a theorem-claims page, not symmetric weight, not a launch surface for the underlying ledgers. Post-deploy LinkedIn/Twitter validator pass remains to record after deployment. |

### Promoted from B to A (2026-05-21 PM)

All four original Class B pages cleared Bucket 1 in the same session
the Class A pages did. Promotion notes retained here for posterity.

| Page | Promotion outcome |
| --- | --- |
| `/applications-gallery` | Promoted. Custom 6-card grid viz matches the page's role as "every working system and its evidence tier"; tier dots colour-match the existing applications-gallery card style. |
| `/h-of-x` | Promoted ahead of full content-stability. If `SUNDOG_V_GEOMETRY.md` later shifts the page substantially, re-render the OG card and re-run Post Inspector. The equation viz is rendered as flat text (cairosvg mangles `baseline-shift` tspans), so the subscript on R₂₂ uses Unicode `₂₂` rather than SVG tspan markup. |
| `/mesa` | Promoted. The λ-axis viz visually anchors the 0.953 cliff that the description leans on. |
| `/structural-failure` | Promoted. The OG card uses a stylized five-locus quincunx rather than embedding the existing `/media/structural-boundary-five-locus-map.svg` — that asset is too dense for OG-thumbnail legibility. |

## Phase Status

**Phase 1 (Bucket 1) — CLEARED 2026-05-21; Class A page additions through 2026-05-30.**
Seventeen Class A pages ship with OG block + designed 1200×630 image +
JSON-LD + tuned title/description + sitemap entry + homepage internal link.
The load-bearing pillar pages (`/h-of-x`, `/mesa`, `/structural-failure`,
`/isotrophy`, `/faraday`, `/navierstokes`) moved from header/footer
references to dedicated pillar cards in the `Load-Bearing Evidence`
section on `index.html`, with per-node clickable SVGs and hover
tooltips routing readers to specific phase documents rather than the
roadmap overview. The four Class C pages each received a
sentinel-wrapped minimal OG block pointing at a shared generic Sundog
card. The three Class D pages received `noindex, follow` robots meta
and `/repo-map` was removed from the sitemap. `/geometry` was
discovered mid-sweep (existed but wasn't in the matrix) and brought to
full Class A treatment. `/navierstokes` and `/generality` were filed
2026-05-30 as the asymmetric high-stakes-math architecture: NSE C1
earns its own deep-dive page (strongest current witness, the only
"probably fruitful" lane); the generality umbrella covers all six
lanes (NSE + Riemann + Yang-Mills + P-vs-NP + ARC-AGI + Three-Body 15C)
with the discipline-framing carried explicitly into the layout
("bounded-null and review-gated results are receipts, not failures";
asymmetric weight is the discipline, not a flaw).

Outstanding from Phase 1: the *one* item that needs your hands and not
the codebase — running [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)
and [Twitter Card Validator](https://cards-dev.twitter.com/validator)
on each Class A URL *after* the next deploy. Until that's done, the
caches won't refresh on the corrected metadata. Treat this as the hard
prerequisite before any external share of a previously-cached URL.

**Phase 2 (Bucket 2) — READY TO START.** Compounding work, no
deadlines. See Bucket 2 section below for scoped items.

## Buckets of Work

### Bucket 1 — Do before any new public share (deadline-bound) — CLEARED

These are the moves with a hard deadline because LinkedIn caches
previews. Each row below maps to a discrete change.

1. **Add OG + Twitter card meta tags** to the page, mirroring the
   pattern in [`public/og/_patch_meta.py`](../../public/og/_patch_meta.py)
   (sentinel-wrapped, idempotent on re-runs).
2. **Create a real `og:image`** at 1200×630, using
   [`public/og/_generate.py`](../../public/og/_generate.py)'s template.
   Without an image the preview falls back to no-image card.
3. **Tune the title and description**: title ≤ ~60 chars for Google
   SERP, description ≤ ~155 chars. Include the actual searched terms
   the page should answer. Stay within claim-hygiene limits — see
   anti-pattern (i) below.
4. **Add JSON-LD `Article` / `TechArticle` / `WebSite` / `AboutPage`**
   schema. Less LinkedIn-relevant but helps Google rich results and
   can pull in a "Published on" line.
5. **Link the page from `index.html`'s research rail** if it's a
   workbench, or from the header nav if it's positioning. Without an
   inbound link from the homepage, the page has weak internal
   link-equity and Google's deep crawl is slower.
6. **Run [LinkedIn Post Inspector](https://www.linkedin.com/post-inspector/)
   and [Twitter Card Validator](https://cards-dev.twitter.com/validator)**
   on the deployed URL *before* the first external share. This forces
   the cache to refresh on the corrected metadata. After this is done
   once, the page is in steady-state.

For Class B pages, Bucket 1 is owed end-to-end. For Class A pages
already in the matrix as ✓, Bucket 1 is done.

### Bucket 2 — Compounding (Phase 2, ready to start)

Phase 2 is not deadline-bound. Items below are scoped enough to pick up
without re-deriving them. Each carries an owner, a concrete first step,
and a success criterion.

**B2.1 — Three-gate reading note for the unit-distance result**
- **Owner:** human author (Sundog research voice); Claude can draft from
  ledger Cand. 1 if delegated.
- **Why this matters:** the workbench is the engagement hook; the note
  is what ranks for the analytical queries ("what does the OpenAI
  unit-distance result mean", "AI math autonomy", "discrete-geometry
  breakthrough explained"). Interactive grids don't match those search
  intents — analytical prose does.
- **First concrete step:** create `docs/notes/CAPSET_THREEGATE_READING.md`
  per the scope in
  [`SUNDOG_V_CAPSET.md`](../SUNDOG_V_CAPSET.md) Cand. 1 §Sundog variant
  (taxonomy restatement, application to the 78-year stalemate, explicit
  detection-primary attribution, counterexamples that would force
  residual-primary). Promote to a `/notes/capset-threegate` URL only
  after one combinatorial-geometry working mathematician sanity-checks
  the reading.
- **Success criterion:** note published as Class A with full Bucket 1
  treatment (OG card, JSON-LD `Article`, designed image, matrix row);
  external sanity-check on file; ledger Cand. 1 graduates to a roadmap.

**B2.2 — Submit sitemap to Google Search Console**
- **Owner:** human with sundog.cc GSC access (one-off; cannot be done
  from the sandbox).
- **First concrete step:** Google Search Console → Settings → Sitemaps
  → submit `https://sundog.cc/sitemap.xml`. Confirm coverage report
  picks up the thirteen Class A URLs.
- **Success criterion:** GSC reports "Success" on the sitemap submission
  and all thirteen Class A URLs appear in the coverage report within ~7 days.

**B2.3 — Cross-link `/capset` from positioning pages** — completed 2026-05-22
- **Owner:** Claude (codebase work).
- **Status:** completed 2026-05-22. `/about`, `/alignment`, and
  `/applications-gallery` each now carry one contextual inline link to
  `/capset`.
- **Why this matters:** `/capset` started Bucket 2 with *two* inbound links
  from `index.html` (rail card + footer rail CTA) but zero from
  `/about`, `/alignment`, `/applications-gallery`. Cross-linking
  accumulates internal link weight before Google's next deep crawl.
- **First concrete step:** locate a content-appropriate paragraph in
  each of the three pages (e.g., the "AI mathematics under
  traceability" angle on `/alignment`; the "evidence tiers across
  applications" angle on `/applications-gallery`; the "research-program
  scope" angle on `/about`) and add a single inline link to `/capset`
  with anchor text that names what the reader will find ("cap-set
  workbench", not "click here").
- **Success criterion:** met 2026-05-22. Three new inbound internal
  links live in the build output; matrix annotation on `/capset`
  updated.

**B2.4 — Standing checklist for new public pages** — rule added 2026-05-22
- **Owner:** Claude when invoked on new pages; humans during code
  review for non-Claude-authored pages.
- **Status:** standing rule added to `AGENTS.md` on 2026-05-22. Final
  observation remains pending until the next new `site-pages.json` entry
  exercises the rule.
- **Why this matters:** the matrix is the gate. Without a standing
  check, new pages will ship without Bucket 1 treatment and the
  discipline degrades.
- **First concrete step:** completed 2026-05-22. `AGENTS.md` now says
  new entries in `site-pages.json` must update the SEO matrix and clear
  Bucket 1 before `publicLaunchIntent` is treated as satisfied.
- **Success criterion:** pending. The rule must be observed at least
  once on a new page added after this filing; matrix row exists for
  that page.

**B2.5 — Linter / file-watcher investigation (new in Phase 2)**
- **Owner:** human with editor/IDE config access.
- **Why this matters:** during Phase 1 work, `index.html`,
  `public/og/_generate.py`, `public/og/_patch_meta.py`, and
  `capset.html` all had their tails silently truncated at edit time.
  The current workaround is to splice from git HEAD when corruption is
  detected. This is fragile and will eventually bite a file with no
  recoverable history.
- **First concrete step:** check the editor's Prettier / ESLint / save
  hook configuration. Most likely cause: a save-action with a file-size
  or line-count limit. Less likely: a watcher syncing partial writes.
- **Success criterion:** a known-large file (e.g., `index.html` >2000
  lines) survives multiple edit cycles in the affected editor without
  truncation.

### Bucket 3 — Anti-patterns (things to specifically NOT do)

These are not aesthetic preferences; each one has a Sundog-brand cost
attached.

i. **No clickbait titles or H1s.** "AI Just Solved an Unsolvable
   Problem!" reads against Sundog's whole claim-hygiene posture —
   the same posture
   [`SUNDOG_V_CHAT.md`](../SUNDOG_V_CHAT.md) defends in the chat widget,
   the same posture the
   [`SUNDOG_OUTREACH_PACKET.md`](../promo/SUNDOG_OUTREACH_PACKET.md) enforces
   with specialist reviewers. A clickbait title is an unforced
   inconsistency.

ii. **No keyword stuffing the description.** One mention each of the
    relevant search terms is enough. Description is for humans
    deciding whether to click; the Google ranking signal from
    description text is weak, and over-stuffing reads as low-quality.

iii. **No public sharing before OG block + image are deployed and the
     Post Inspector has been run.** LinkedIn caches the broken preview
     for the cache window (typically 7 days). The cost of being early
     is being shown a broken preview to everyone who sees the post in
     feed for a week.

iv. **No `og:image` reuse across pages with different content.** A
    single generic Sundog card across `/balance` and `/threebody`
    erases the per-page identity that makes the series scan-readable in
    feed.

v. **No promoting Class C pages to Class A without a content reason.**
   Hooking `/legend` up with a designed OG card because it's "easy"
   produces a card no one will share. Promotion follows share value,
   not effort.

## Renderer Notes

Two pieces of infrastructure made this work tractable; future authors
should be aware of both.

1. **CairoSVG, not ImageMagick.** The system's ImageMagick was
   compiled `--without-rsvg`, so its internal MSVG parser silently
   ignores `stop-opacity` and `fill-opacity`. Gradients render at full
   opacity. Use the `cairosvg`-based renderer in `_generate.py`.

2. **Linter / file-watcher in `public/og/`.** Long Python files in
   that directory have had their tails silently truncated mid-statement
   during this work session. The current workaround is to run the
   generator from `/tmp` by copying the script there first. If the
   `public/og/` scripts become a permanent part of the build, they
   should probably move to `scripts/` or `tools/` outside the path the
   linter watches. Tracked as a follow-up; see
   [Inspection Trail](#inspection-trail).

## Promotion / Done Criteria

The roadmap is "done" when:

1. every Class A page row in the matrix is fully ✓ across OG block,
   image, JSON-LD, title, description, rail-link, and sitemap;
2. every Class B page has either been promoted into Class A and
   completed Bucket 1, or has been re-classified to C with a recorded
   reason;
3. Bucket 2 item (4) — the standing checklist for new pages — has
   been observed at least once on a new `site-pages.json` entry, so
   the discipline is exercised, not just written down;
4. Bucket 3 anti-patterns have been declared in
   [`presentation/claims-and-scope.md`](../presentation/claims-and-scope.md)
   or `AGENTS.md` as standing rules, so future authors don't need to
   re-derive them from this roadmap.

Until all four are true, this is a living roadmap.

## Cross-references

- [`UI_UX_THEME_FOUNDATION.md`](UI_UX_THEME_FOUNDATION.md) — the OG
  card series inherits its paper-theme tokens (`--sd-*`), serif
  display type, and warm-paper background from the theme foundation.
  A Migration Steps cross-reference pointer was added there pointing
  back to this roadmap.
- [`SUNDOG_V_CAPSET.md`](../SUNDOG_V_CAPSET.md) — the cap-set ledger; its
  Cand. 1 (three-gate reading note) is Bucket 2 item (1) of this
  roadmap.
- [`SUNDOG_V_GEOMETRY.md`](../SUNDOG_V_GEOMETRY.md) — the geometry
  roadmap; `/geometry.html` is now the Class A geometry hub, and future
  geometry pages inherit the standing checklist.
- [`SUNDOG_OUTREACH_PACKET.md`](../promo/SUNDOG_OUTREACH_PACKET.md) — the
  specialist-review-tier packet; different audience, different surface.
  The two documents are complementary, not overlapping.
- [`presentation/claims-and-scope.md`](../presentation/claims-and-scope.md)
  — the claim-hygiene reference; Bucket 3 anti-patterns derive from
  the discipline this doc names.
- [`public/og/_generate.py`](../../public/og/_generate.py),
  [`public/og/_patch_meta.py`](../../public/og/_patch_meta.py) — the
  current renderer and patcher. See Renderer Notes above.

## Inspection Trail

- 2026-05-20 — OpenAI unit-distance disproof published
- 2026-05-21 AM — `/capset` shipped with full OG / Twitter / JSON-LD
  and a designed `og:image` (initially ImageMagick-rendered with
  broken gradient; later re-rendered through CairoSVG).
- 2026-05-21 PM — Class A pages (7 additional) brought to parity via
  `_generate.py` (cards) and `_patch_meta.py` (HTML).
- 2026-05-21 PM — this roadmap filed.
- 2026-05-21 PM (late) — all four Class B pages promoted to Class A and
  cleared Bucket 1 in the same session. The then-current twelve Class A
  pages ✓ on OG
  block / image / JSON-LD / title / description / sitemap. Internal
  link equity (rail-link column) for `/h-of-x`, `/mesa`,
  `/structural-failure` was at ◐ — they had header / footer / contextual
  links but no homepage card.
- 2026-05-21 PM (later) — `Load-Bearing Evidence` section on
  `index.html` reworked: `/h-of-x` added as a new fourth pillar; the
  three pre-existing pillar cards (`/structural-failure`, `/mesa`,
  `/coarse-graining`) had their dense 1280-wide SVGs replaced with
  mobile-friendly inline SVGs that mirror the OG card visual vocabulary.
  Pillar SVGs were then refactored from a single whole-SVG hyperlink to
  per-node clicks: 18 individual nodes across the four pillars, each
  with its own `<a href>` to a substantive phase document and a
  `<title>` tooltip naming the destination. Card-lift hover disabled
  for pillar items. Result: matrix rail-link column for `/h-of-x`,
  `/mesa`, `/structural-failure` flips ◐ → ✓.
- 2026-05-21 PM (final) — capset.html's OG block re-wrapped in
  `OG-BLOCK-START`/`OG-BLOCK-END` sentinels (had been stripped by the
  file-watcher truncation bug at an earlier point), making the
  then-current twelve Class A pages re-runnable through `_patch_meta.py` without
  duplication. Phase 1 cleared.
- 2026-05-21 PM (matrix sweep) — easy items off the matrix:
  (a) discovered `/geometry` existed but wasn't tracked; treated as
  Class A with 3-tile workbench-shelf OG card + full Bucket 1 meta.
  (b) shared generic Sundog OG card (`sundog-generic.png`) built and
  applied to all four Class C pages (`/sundog-workbench`, `/chat`,
  `/legend`, `/origin`) with sentinel-wrapped minimal blocks and
  page-specific title/description.
  (c) `noindex, follow` added to all three Class D pages
  (`/paper-theme-demo`, `/repo-map`, `/atlas`).
  (d) `/repo-map` removed from `public/sitemap.xml` (was Class D in
  the sitemap by accident).
  Net: thirteen Class A, four Class C (minimal-OG), three Class D
  (no-index). Every page in `site-pages.json` now has a defensible
  matrix row.
- 2026-05-22 — `/isotrophy` filed as a new Class A page. Public-facing
  companion to `docs/isotrophy/kfacet/kfacet_v03h_writeup.md`: a static
  narrative page rendering the 20/21 structural-zero verdict + 1 named
  quarantine (O_617) with explicit "audit chain is intact; theorem-facing
  result is not closed" boundary. Full Bucket 1 treatment landed (OG
  card with paper-ruled aesthetic + 21-cell catalog grid + sticky-note
  verdict tiles; JSON-LD TechArticle; sitemap entry; site-pages.json
  manifest entry; sister ledger at
  [`SUNDOG_V_ISOTROPHY_KFACET.md`](../isotrophy/SUNDOG_V_ISOTROPHY_KFACET.md)). No
  homepage pillar yet — open follow-up. Class A count: 14.

**Phase 1 closeout summary:** thirteen Class A pages, thirteen OG cards, thirteen
JSON-LD blocks, all internal links present, all sitemap entries
present, all sentinels present. Only outstanding Phase 1 item is
running LinkedIn Post Inspector + Twitter Card Validator on each URL
post-deploy — that's a one-off human action, not a codebase change.

Open follow-ups (carried into Phase 2):

- All Bucket 2 (Phase 2) items are scoped in the Bucket 2 section
  above with owner, first step, and success criterion. Pick up there.
- The linter-truncation bug (Renderer Notes §2 above, and B2.5 in
  Bucket 2) is the highest-risk un-owned item — a single failed
  truncation on a less-recoverable file would cost real work.
- Class C / D pages remain explicitly deferred. Promote a Class C only
  with a content reason per Bucket 3 anti-pattern (v).
