# Anniversary Fix Roadmap

Internal readiness checklist for the Sundog Year 1 public statement.

Status: active, 2026-05-16.
Target publication date: 2026-05-19.
Companion packet: [`anni_spam_roadmap.md`](anni_spam_roadmap.md).

## Purpose

This file keeps the unglamorous fixes separate from the public copy. The
rollout is only credible if the site and docs show the same discipline the
statement claims: narrow claims, visible boundaries, and no hidden overreach.

Each item below now has:

- **Scope:** what files or surfaces are in play.
- **Steps:** the work sequence.
- **Checks:** commands or review probes.
- **Exit:** the condition for calling it done.
- **Solo status:** whether one person can finish it without a specialist or
  long experiment.

## P0 - Publication-Gating Before First Post

These block the broad public push if unresolved.

### 1. Terminal-accuracy wording

**Solo status:** mostly soloable. Text fix landed 2026-05-16; remaining work is
one final search before launch.

Problem:

The phrase "statistically indistinguishable from the oracle" is too easy to
attack. A non-significant Mann-Whitney result at `n=30` is failure to detect a
difference, not an equivalence result.

Scope:

- `index.html`
- `about.html`
- `docs/presentation/claims-and-scope.md`
- `docs/SCIENTIFIC_CRITERIA.md`
- `docs/PAPER_v1_draft.md`
- any anniversary copy that mentions terminal accuracy

Steps:

1. Replace broad equivalence language with "no detected terminal-intensity
   difference at n=30" or "comparable terminal accuracy in the tested setting."
2. Pair every terminal-accuracy sentence with the acquisition-cost caveat:
   roughly `16x` slower in the reported core run.
3. Keep oracle language precise: the target-aware analytic baseline is a
   comparison reference, not a ground-truth optimum.
4. Do not claim equivalence unless a TOST/equivalence test with a
   pre-registered margin is actually added later.

Checks:

```powershell
rg -n "statistically indistinguishable|indistinguishable|matches the oracle|reaches the same terminal|equality with cost|matching oracle baseline accuracy" index.html about.html docs\presentation\claims-and-scope.md docs\SCIENTIFIC_CRITERIA.md docs\PAPER_v1_draft.md internal\anniversary
rg -n "terminal accuracy|terminal-intensity|16x|16×|oracle" index.html about.html docs\presentation\claims-and-scope.md docs\SCIENTIFIC_CRITERIA.md docs\PAPER_v1_draft.md internal\anniversary
```

Exit:

- The first search returns no launch-facing overclaim hits.
- The second search shows terminal-accuracy claims visibly paired with slower
  acquisition or bounded wording.

Solo pass, 2026-05-16:

- Updated `index.html`, `about.html`,
  `docs/presentation/claims-and-scope.md`, `docs/SCIENTIFIC_CRITERIA.md`, and
  `docs/PAPER_v1_draft.md`.
- Exact overclaim search returned no hits on the edited launch-facing set.

### 2. Oracle-leakage audit

**Solo status:** audit can be soloed; any discovered leakage becomes a code or
paper fix and may need a longer pass.

Problem:

The cleanest skeptic attack is implementation leakage: the agent may be denied
target coordinates in its observation vector but still receive target-aware
training, termination, shaping, or success signals.

Scope:

- core MuJoCo environment / task assembly
- controller observation construction
- reward, success, and termination code
- experiment runner and metric computation
- paper wording that distinguishes controller-visible from metric-only state

Steps:

1. Locate core experiment files with:

   ```powershell
   rg -n "doa_direct|doa_noisy|target|laser|detector|photometric|termination|success|reward|observation|obs" .
   ```

2. Build a four-column visibility table:

   | quantity | controller-visible | training-visible | metric-only | logging-only |
   | --- | --- | --- | --- | --- |

3. Trace target coordinates through:

   - reset/seed generation;
   - observation vector;
   - controller update;
   - reward/success;
   - episode termination;
   - saved result metrics.

4. Write a receipt under `docs/` or `results/` with exact file/line references.
5. If any leakage is controller-visible or training-visible, demote the
   anniversary claim until repaired.

Checks:

- A reviewer can reproduce the visibility table from file references.
- The paper and public copy use "without target-position access" only for the
  controller path, not for metric computation.

Exit:

- Receipt exists.
- Either "no leakage into controller/training path" is supported, or the public
  statement is revised downward.

### 3. Mesa lambda-confound caveat

**Solo status:** wording pass is soloable; confound experiment is not.

Problem:

The Mesa cliff at `lambda ~= 0.952588` is powerful and suspicious for the same
reason. If lambda is collinear with effective learning rate, reward-scale
normalization, or gradient-norm ratio, the cliff could be an optimizer artifact.

Scope:

- `mesa.html`
- `docs/presentation/claims-and-scope.md`
- `docs/SUNDOG_V_MESA.md`
- `docs/COARSE_GRAINING_PROOF_ROADMAP.md`
- anniversary posts

Steps:

1. Treat the cliff as a mapped threshold / operating-envelope boundary.
2. Avoid using it as standalone evidence for universal Goodhart immunity.
3. Keep the confound experiment named:
   reward-gradient rescale plus no-op transform invariance.
4. Do not promote Postulate 2 / Postulate 4 language out of speculative until
   the confound test clears.

Checks:

```powershell
rg -n "sidesteps Goodhart|cannot be reward-hacked|Goodhart immunity|universal immunity|gravity.*proof|law" mesa.html docs\presentation\claims-and-scope.md docs\SUNDOG_V_MESA.md docs\COARSE_GRAINING_PROOF_ROADMAP.md internal\anniversary
```

Exit:

- Anniversary copy uses the cliff as a failure-boundary receipt.
- Any gravity-frame sentence carries the bounded caveat or links the gravity
  ledger / proof roadmap.

### 4. Geometry rendered-vs-anchored surface check

**Solo status:** audit is soloable. Visual/copy patch is soloable unless it
requires new HaloSim renders.

Problem:

The geometry lane is now honest in the docs, but the public surface must not
visually imply that every rendered primitive is an anchored inverse route.

Scope:

- `sundog.html`
- `legend.html`
- `docs/SUNDOG_V_GEOMETRY.md`
- `docs/calibration/HALO_PHENOMENA_ACCOUNTING.md`
- any anniversary post using halo geometry as evidence

Steps:

1. Confirm the public page distinguishes:

   - rendered core;
   - rendered optional;
   - named-only literature;
   - not-modeled / speculative.

2. Confirm the public page exposes or links:

   - strict parhelion inverse eligibility;
   - `29 deg` tangent/circumscribed merge;
   - `32.2 deg` CZA cutoff;
   - rendered-not-anchored warning;
   - parhelic-belt-y caveat if the visual layer still draws it.

3. Patch copy before adding new visuals. Do not invent new status labels.
4. If a visual primitive is known placeholder-class, label it as display /
   vocabulary, not inverse evidence.

Checks:

```powershell
rg -n "rendered|anchored|named-only|not-modeled|CZA|32\.2|29|circumscribed|parhelic-belt|belt-y" sundog.html legend.html docs\SUNDOG_V_GEOMETRY.md docs\calibration\HALO_PHENOMENA_ACCOUNTING.md
```

Exit:

- A reader can tell what is rendered, what is anchored, and what is only named
  or speculative without opening an internal memo.

Solo pass, 2026-05-17:

- `legend.html` already separated rendered core, rendered optional, named-only,
  and not-modeled families, with only parhelion offset promoted as the inverse
  handle.
- Patched `sundog.html` so the hero, sun-altitude demo, and full-atlas lead
  say explicitly that rendered does not mean promoted. The page now names the
  current promoted inverse handle, the strict eligible-photo posture, the
  29 degree tangent/circumscribed merge, the ~32.2 degree CZA cutoff, and the
  parhelic-belt / pillar visual-QA caveat.
- Remaining work is visual: future atlas cards can carry explicit status chips,
  but the launch-blocking copy distinction is now public.

## P1 - Same-Day Public Coherence

These do not block the first post, but should land before wider technical
outreach.

### 5. Remove internal phase shorthand from public HTML

**Solo status:** inventory is soloable; copy cleanup should be done in small
page batches.

Problem:

Public cards say "Phase NN" in ways that collide across workstreams. Two
different "Phase 11" references can mean unrelated roadmaps.

Scope:

- launch-facing root HTML: `index.html`, `about.html`,
  `applications-gallery.html`, `sundog.html`, `mesa.html`, `chat.html`,
  `threebody.html`, `balance.html`, `mines.html`, `legend.html`
- `docs/index.html` is allowed to use phase names because it is a document
  index, but its notes should still be readable.
- developer/test pages such as `public/phase3-tests.html` are not launch copy.

Steps:

1. Inventory visible launch-facing hits:

   ```powershell
   rg -n "Phase [0-9]+|phase [0-9]+|Step [0-9]+|step [0-9]+|roadmap" --glob "*.html" --glob "!dist/**"
   ```

2. Classify each hit:

   - **Visible public copy:** rewrite or link exact doc.
   - **Code comment / data attribute:** leave unless confusing in generated UI.
   - **Docs index label:** okay if it names the document.
   - **Test/dev page:** leave.

3. Rewrite visible copy by replacing "Phase NN" with:

   - "the bounded three-body result";
   - "the repaired Balance operating-envelope result";
   - "the Pressure Mines confirmed pocket";
   - "the chat safety-floor sweep";
   - "the Mesa operating-envelope map";
   - or a direct link to the exact doc when the phase label is the artifact.

4. Re-run the inventory and check that remaining hits are intentional.

Initial visible hit list, 2026-05-16:

- `index.html`: public copy references Phase 10 attack roadmap in the
  substrate-coincidence section. Rewrite or hyperlink with non-phase wording.
- `applications-gallery.html`: visible Three-Body / Balance / Mines status
  cards use Phase 10 / Phase 11. Rewrite to result names.
- `balance.html`: visible scaffold/status copy uses Phase 1-10. Rewrite to
  "diagnostic scaffold" and "operating-envelope check."
- `threebody.html`: controls and status panels use Phase 2 / Phase 3 /
  Phase 11. Rewrite UI labels to "Sensor-limited view," "Controller," and
  "Operating-envelope result."
- `mines.html`: boundary and replay labels use Phase 9 / Phase 10. Rewrite to
  "boundary warning," "confirmed pocket," and "replay shortcuts."
- `mesa.html` and `chat.html`: many phase labels are evidence-artifact labels.
  Keep if linked/inspectable, but add a non-phase lead sentence where needed.
- `sundog.html` / `sundog-workbench.html`: visible phase labels around math
  binding and upload. Rewrite in public page; comments can stay.

Solo pass, 2026-05-16:

- Visible `Phase 11` shorthand was removed from public HTML. The three-body,
  applications-gallery, chat, and docs-index surfaces now use "later trials"
  / "later-trials" wording with inspectable links instead of relying on the
  internal phase label.

Exit:

- A non-repo reader does not need the internal phase calendar to understand a
  public page.

### 6. Ask Sundog polish

**Solo status:** prompt/data audit is soloable; hosted-model regression is a
separate smoke pass.

Problem:

The chat widget should act like a claim-boundary guide during rollout, not a
generic hype bot.

Scope:

- `chat/claim_map.json`
- `chat/prompts/*.jsonl`
- `public/data/sundog-*.json`
- `chat.html`
- `docs/SUNDOG_V_CHAT.md`

Steps:

1. Confirm claim-map entries point to current docs for:

   - traceability harness;
   - structural failure coincidence;
   - photometric result;
   - Mesa bounded envelope;
   - Atari / crypto / Mythos corrections.

2. Add or adjust answer templates for launch-day prompts:

   - "what is Sundog?"
   - "did Sundog solve alignment?"
   - "is this just a theorem?"
   - "does the agent actually use the inferred variable?"
   - "is this related to SunDog Frozen Legacy?"

3. Rebuild generated chat data with `npm run build` or the chat index script.
4. Smoke test deterministic answers first; only then run hosted/backend checks.

Checks:

```powershell
rg -n "solves alignment|universal theorem|SunDog: Frozen Legacy|crypto|traceability harness|structural failure|probe" chat public\data docs\SUNDOG_V_CHAT.md chat.html
npm run build
```

Exit:

- Ask Sundog answers "what is Sundog?" with apparatus language and links
  sources.
- Ask Sundog answers adversarial prompts with boundaries instead of escalation.

### 7. Docs index and inspection path

**Solo status:** soloable. Reopened by new proof/prereg documents, then
re-closed on 2026-05-16.

Problem:

The public statement asks for review. Reviewers need a path that does not feel
like spelunking.

Scope:

- `docs/index.html`
- `docs/README.md`
- `docs/proof/README.md`
- proof and prereg docs added during anniversary prep

Steps:

1. Run the missing-doc audit:

   ```powershell
   $html = Get-Content -Raw docs\index.html
   $hrefs = [regex]::Matches($html, 'href="([^"]+)"') | ForEach-Object { [System.Uri]::UnescapeDataString(($_.Groups[1].Value -replace '\\','/')) }
   $docs = rg --files docs | ForEach-Object { $_ -replace '\\','/' } | Where-Object { $_ -match '\.(md|tex|pdf)$' } | ForEach-Object { $_ -replace '^docs/','' } | Sort-Object
   $docs | Where-Object { $hrefs -notcontains $_ }
   ```

2. Add missing documents to themed sections, not a flat dump.
3. Keep the inspection path near the top:

   - Scientific Criteria;
   - Claims and Scope;
   - Coarse-Graining Proof Roadmap;
   - Structural Failure Coincidence prereg;
   - Ask Sundog / Chat claim-boundary docs.

4. Re-run `npm run build` so `dist` link checking catches broken URLs.

Audit result, 2026-05-16 pre-fix:

The docs index had reopened. Missing count: 26. Missing families:

- proof trunk: `COARSE_GRAINING_PROOF_ROADMAP.md`,
  `proof/README.md`, `proof/POSTULATE1_DEFINITIONS.md`,
  `proof/PHASE1_LQG.md`;
- geometry audit: `geometry_agent_audit.md`;
- Mesa path-B spec: `mesa/PHASE7_V2_PATH_B_HPARAM_SPEC.md`;
- structural-failure prereg expansion:
  `BOUNDARY_MAP.md`, `P1_ADMISSION.md`, `P2_*`, and related cut files under
  `prereg/structural-failure-coincidence/`.

Solo pass, 2026-05-16:

- Added the proof trunk, proof artifacts, geometry agent audit, Mesa Path-B
  hparam spec, and structural-failure expansion docs to `docs/index.html`.
- Re-ran the missing-doc audit: `docs=150`, `missing=0`.

Exit:

- Missing-doc audit returns zero.
- `npm run build` passes with `dist link check passed`.

### 7a. Hero rail artwork and logo readiness

**Solo status:** inventory and logo promotion are soloable; new screenshots and
clips depend on each workbench being opened and framed deliberately.

Problem:

The homepage rail behavior is stronger than its visual evidence. Balance,
Photometric Alignment, EyesOnly, Dungeon Gleaner, and Money Bags now have real
posters; Three-Body and Pressure Mines still rely on CSS visuals until we have
clean screenshots. The logo toolkit is ready, but production favicons must be
promoted explicitly.

Scope:

- `index.html`
- `public/js/motion-rail.mjs`
- `public/media/`
- `public/favicon.*`
- `public/apple-touch-icon.png`
- `public/icons/`
- `docs/ICON_ASSETS.md`
- `docs/LOGO_ANIMATION_TOOLKIT.md`
- [`hero_rail_artwork_inventory.md`](hero_rail_artwork_inventory.md)

Steps:

1. Keep the Pushable Occluder interrupt disabled until its poster/clip and
   boundary receipt exist.
2. Promote the characterized logo set with `npm run logo:promote`.
3. Capture rail screenshots at 390 px, 520 px, 1280 px, and reduced-motion.
4. Replace CSS placeholders with real posters one card at a time, keeping the
   stamp tier unchanged unless the owning evidence doc changes.
5. Before adding video clips, update the rail script so `data-media` duration
   actually drives the stamp handoff.

Checks:

```powershell
rg -n /media index.html
npm run logo:promote
npm run build
```

Exit:

- Logo production assets are regenerated from the characterized mark.
- The artwork inventory names every missing rail poster and the screenshot QA
  matrix.
- Build passes after the promoted assets are in place.

Solo pass, 2026-05-16:

- Added real homepage rail posters for Photometric Alignment, EyesOnly, and
  Dungeon Gleaner from `assets/images/`.
- Left the MuJoCo graph and aquarium image unassigned pending a cleaner card
  mapping.
- Added a Money Bags telemetry poster from the local Money Bags playtest
  output.
- Upgraded the remaining Three-Body and Pressure Mines CSS visuals so the
  placeholders are card-specific rather than generic grid marks.
- Added keyboard stepping plus wheel/pointer takeover to the motion rail script,
  then smoke-checked 390 px, 520 px, 1280 px, keyboard takeover, wheel takeover,
  and reduced-motion behavior.
- Removed the commented Pushable Occluder image URL from `index.html` to avoid
  false broken-asset hits before the interrupt card is ready.

### 7b. Post-rail Working Systems evidence panels

**Solo status:** roadmap and placeholder polish are soloable; chart generation
should be done one evidence panel at a time from the owning result docs.

Problem:

On phone-width layouts, the Working Systems grid after the motion rail stacks
seven 200px placeholder bands. The bands are intentional "no image yet"
slots, but they read as broken empty sections because the label duplicates the
card heading, the label is not centered, and the dark pinstripe background does
not communicate "chart pending." This is also a missed opportunity: the site
now has heavily documented interpretation surfaces that would benefit from
small charts more than more generic screenshots.

Scope:

- `index.html`
- `public/css/sundog-theme.css`
- `public/media/`
- `docs/WEBSITE_DEVELOPMENT.md`
- `docs/UI_UX_THEME_FOUNDATION.md`
- `docs/SUNDOG_V_MESA.md`
- `docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md`
- `docs/COARSE_GRAINING_PROOF_ROADMAP.md`

Panel contract:

- Replace placeholder labels with an actual visual or a deliberately styled
  "chart pending" treatment; do not duplicate the card `h3`.
- Every visual names its evidence source in the card link or caption trail.
- Static PNG/SVG exports under `public/media/` are preferred for launch; live
  canvas is allowed only after mobile screenshots prove it does not jank.
- The visual must interpret a claim boundary, not decorate the card.
- Mobile height should be responsive, not a fixed 200px slab.

First chart candidates:

| panel | source | visual to create | why it belongs after the rail |
| --- | --- | --- | --- |
| Mesa Optimization | `docs/SUNDOG_V_MESA.md`, `mesa.html`, `results/mesa/operating-envelope/` | Lambda-cliff mini chart plus class-balance strip. | Shows the boundary between protected and collapsed proxy behavior without universalizing the result. |
| Structural Failure Boundary Map | `docs/prereg/structural-failure-coincidence/BOUNDARY_MAP.md`, `docs/prereg/structural-failure-coincidence/PUBLICATION_PLAN.md`, `public/data/structural-failure-boundary-map.json` | Five-locus eligibility / abstain / switch / fail map. | Makes the traceability harness legible: a real inverse must fail where the inverse is ill-posed. |
| Coarse-Graining Proof Trunk | `docs/COARSE_GRAINING_PROOF_ROADMAP.md`, `docs/proof/*` | Phase ladder from definitions through LQG, MDP, boundary theorem, and open empirical controls. | Gives the theorem-track work a disciplined public shape without turning it into an overclaim. |
| Three-Body Dynamics | `docs/threebody/PHASE13_RESULTS.md`, `PHASE14_RESULTS.md`, `PHASE15_RESULTS.md` | Pocket-vs-boundary heatmap with low-velocity warning cells explicit. | Replaces a generic orbit slab with the actual operating-envelope result. |
| Photometric Alignment | paper/result artifacts and current MuJoCo poster | Acquisition-time vs terminal-accuracy comparison or R22/cos(h) route sketch. | Keeps the confirmed result narrow and inspectable. |
| Money Bags | local playtest telemetry already promoted to rail poster | Softbody telemetry thumbnail or graph-metric mini panel. | Uses the strongest current evidence shape: telemetry before verdict. |

Steps:

1. Keep the temporary placeholder treatment centered and responsive so mobile
   no longer reads as a chrome bug while real visuals are staged.
2. Add a small shared visual primitive, likely `.sd-evidence-panel-visual`,
   that accepts `<img>` first and can later host inline SVG/canvas.
3. Build the Mesa panel first because `mesa.html` already carries the chart
   model and claim language.
4. Build the structural-failure boundary-map panel second because it is the
   cleanest public expression of the "apparatus, not theorem" pivot.
5. Build the coarse-graining proof ladder third, with status language that
   distinguishes closed proofs from open empirical controls.
6. Revisit the existing Working Systems cards: either replace their
   `.app-card-img` blocks with evidence visuals or split the section into
   "Working Systems" and "Interpretation Surfaces" if the research-track cards
   make the app grid too broad.

Checks:

```powershell
npm run build
npm run dev -- --port 5173
```

Then screenshot:

- `390px` width after the rail;
- `520px` width after the rail;
- `1280px` desktop grid;
- reduced-motion mode.

Exit:

- No mobile card has a large undifferentiated blank visual slab.
- No visual repeats the heading as its only content.
- Mesa, structural-failure boundary map, and coarse-graining each have at
  least one chart/diagram slot with a source trail.
- Build passes and the mobile screenshots show no text overlap.

Solo pass, 2026-05-16:

- Centered the current placeholder labels, made the pinstripe/grid treatment
  more visibly intentional, and changed the app-card visual height from fixed
  `200px` plus shared `12rem` min-height to a responsive clamp so the mobile
  stack is less slabby while the real evidence panels are created.

### 7c. Homepage elevator pitch and hero explanation pass

**2026-05-17 reality check:** this pass improved mechanics but moved the page
in the wrong content direction. The compressed v1.2 pitch, standalone static
hero guide, and added explanatory cards made the page more redundant and less
serious. Treat this section as historical context only. The corrective roadmap
is **7d. Homepage coherence rollback and evidence-first reorder** below.

**Solo status:** roadmap, copy compression, stable glossary IDs, and HaloSim
thumbnail promotion are soloable. Using third-party calibration photographs as
public margin art requires an attribution / reuse check first.

Problem:

The homepage elevator pitch is doing too much at once. It is a four-paragraph
claim block, it leans on terms like "circumzenithal" before the reader has a
visual grip on them, and it uses punctuation that makes the prose feel more
academic than readable. The hero also now renders as a static atlas snapshot;
it no longer teaches the old phase sequence directly, so readers need nearby
cards that explain what the hero means optically, physically, and
application-wise.

Scope:

- `index.html`
- `legend.html`
- `sundog.html`
- `docs/WEBSITE_DEVELOPMENT.md`
- `public/media/`
- `public/data/`
- `docs/calibration/33.webp`
- `docs/calibration/halosim_outputs/hs0_spike/`
- `docs/calibration/halosim_outputs/phase14e/`

Pitch v2 contract:

- Compress the pitch into a hook plus three scannable blocks: optics,
  hidden-state inference, and applications.
- Keep each public paragraph short enough to scan on mobile; no large
  uninterrupted prose wall.
- Strip most em dash characters from the public pitch. Use shorter sentences,
  commas, colons, or parentheses instead.
- Put hard halo vocabulary behind plain-language phrasing. The reader should
  understand "high smile-shaped arc" before seeing "circumzenithal arc."
- Keep the audit hedge and claim-map discipline intact; do not strengthen the
  substrate-coincidence claim just because the copy gets shorter.

Glossary / hover contract:

- Add stable IDs to the relevant `legend.html` phenomenon cards so terms can
  link somewhere exact.
- Add a small glossary source, likely `public/data/halo-glossary.json`, with
  term, plain definition, and target URL.
- Implement definitions as accessible links or focusable popovers. Hover can
  help desktop readers, but mobile tap and keyboard focus must work.
- First terms to wire: `parhelion`, `circumzenithal arc`,
  `circumhorizon arc`, `tangent arc`, `parhelic circle`, `22 deg halo`, and
  `46 deg halo`.

Margin image contract:

- Promote curated images into `public/media/` with launch-safe names such as
  `elevator-halosim-cza.png`, `elevator-halosim-parhelia.png`, or
  `elevator-sundog-photo-33.webp`.
- Prefer HaloSim-generated receipts for the first pass because they are
  project-created and already tied to the geometry audit trail.
- Treat `docs/calibration/33.webp` as visually promising but attribution-gated.
  Do not make it homepage art until credit and reuse status are clear.
- On desktop, images can sit in the pitch margins. On mobile, collapse them
  into a short inline strip or one selected image so the pitch does not become
  a gallery.

Hero phase-card contract:

- Do not imply the hero is drawing in real time unless the real-time animation
  is rebuilt.
- Add a compact card trio near the hero or immediately after it:

  | card | reader question | link trail |
  | --- | --- | --- |
  | Optics | What am I seeing in the sky? | `sundog.html`, `legend.html` |
  | Physics | What geometry fixes the arcs? | calibration/accounting docs |
  | Applications | Why does this matter beyond halos? | motion rail / workbenches |

- These cards should explain the static hero scene in three registers, not
  duplicate the theorem grid or the motion rail.

Steps:

1. Add stable `id` anchors to the core `legend.html` phenomenon cards.
2. Draft `public/data/halo-glossary.json` and one small accessible term-link
   primitive.
3. Promote two HaloSim thumbnails into `public/media/` and wire them into a
   margin/inline visual treatment for the pitch.
4. Rewrite the pitch to v1.2: shorter, mostly no em dash characters, simpler
   vocabulary first, same audit boundary.
5. Add the three hero explanation cards and link them to inspection surfaces.
6. Run the chat claim-map checks because this is a positioning edit.

Checks:

```powershell
rg -n "$([char]0x2014)|&mdash;|circumzenithal|circumhorizon|Phase [0-9]+|phase [0-9]+" index.html
npm run build
npm run chat:eval:static
npm run chat:eval:phase3
npm run chat:eval:phase3:adversarial
npm run chat:eval:phase3:differential
npm run chat:eval:phase4
```

Then screenshot:

- 390 px homepage top through the elevator pitch;
- 520 px homepage top through the elevator pitch;
- 1280 px desktop top through the elevator pitch;
- one keyboard-focus pass over glossary terms.

Exit:

- The pitch is no longer a large block of text.
- Specialist halo words are either avoided, defined inline, or linked to a
  relevant atlas/legend target.
- The homepage has at least one real sundog / HaloSim image near the pitch with
  a clear provenance trail.
- The static hero has explanatory cards that say what it means optically, what
  it means physically, and why it matters for applications.

### 7d. Homepage coherence rollback and evidence-first reorder

**Solo status:** roadmap and first implementation passes are soloable. Rebuilding
the original animated hero is soloable only as a visual/interaction pass if it
can be verified locally at mobile and desktop widths. Any stronger scientific
copy still needs the usual chat/evidence checks.

Problem:

The homepage is becoming additive instead of coherent. Recent passes improved
hover definitions, image handling, and mobile spacing, but also created content
duplication:

- the v1.2 elevator pitch became a second lightweight explanation block;
- the new "Static hero guide" repeats the theorem cards and sounds weaker than
  both;
- the existing Step I-Step VI theorem cards still describe the old animated
  hero sequence, but the hero is now a static SVG snapshot;
- the application rail appears before the load-bearing evidence graphics, so
  product/workbench breadth arrives before the evidence spine;
- nav links are inconsistent across pages, with Mesa promoted as a top-level
  item while Legend and Chat are harder to reach;
- `applications-gallery.html` has a bottom "Next Gallery Pass" block that reads
  internal and should not be public;
- the hero motto/tagline/value prop need a deliberate brand copy pass, not
  incremental phrase accretion.

Recovered source material:

- Old elevator pitch source: `git show d529b68:index.html`, section
  `#elevator-pitch`, `data-version="v1.1"`, headline
  "Field-not-reward, in plain language."
- The old body is the four-paragraph field-not-reward pitch beginning
  "A sundog is an optical phenomenon beside the sun..." and ending
  "the agent, robot, and game engineering that follow from that distinction are
  different, more economical, and more traceable."
- Hero animation history:
  - `f11fdf0` added the animated parhelion canvas sequence.
  - `c7e6174` replaced it with the current static atlas snapshot.
  - `74d5333` added the separate static hero guide.

Order of operations:

1. **Freeze additive content work.**
   - Do not add another homepage explainer block.
   - Do not create more cards to solve a copy hierarchy problem.
   - Keep the glossary/hover primitive because it is useful.

2. **Reinstate the old elevator pitch body with glossary links.**
   - Restore the v1.1 four-paragraph pitch from `d529b68`.
   - Preserve the modern accessible glossary behavior on terms such as
     `sundog`, `22 deg`, `circumzenithal`, `circumhorizon`, `tangent`,
     `mesa-optimization`, `5-dimensional subspace`, and `net.7`.
   - Keep the audit hedge from v1.1. Do not strengthen the
     substrate-coincidence claim while restoring the older prose.
   - Defer content rewriting until the restored version is readable and linked.

3. **Remove the standalone static hero guide.**
   - Delete or hide `.hero-phase-section`; it is currently a weaker duplicate
     of both the pitch and theorem cards.
   - Move any genuinely useful link targets into the existing Step I-Step VI
     cards or the hero CTA cluster.

4. **Decide hero animation vs. static hero explicitly.**
   - Audit whether the old canvas animation can be safely restored from
     `f11fdf0` without regressing performance or clashing with the current
     atlas/brand direction.
   - If restored, the Step I-Step VI cards should roll with or point to the
     animated phases: left parhelion, right parhelion, upper tangent, lower
     tangent, inner iris, outer halo.
   - If the static SVG stays, rewrite Step I-Step VI so they describe the
     current single atlas pose honestly and stop implying a live sequence.

5. **Promote the existing Step I-Step VI cards to the hero explanation layer.**
   - These cards are the right place for "what this means optically, what this
     means physically, and what this could mean for applications."
   - Rewrite them around the new traceability posture: apparatus first,
     boundary-visible, theorem language held as research-program language.
   - Each step should have one job: observed trace, geometric constraint,
     hidden-state inference, controlled experiment, measured result, boundary.

6. **Reorder homepage sections evidence-first.**
   - Proposed order:
     1. Hero.
     2. Restored elevator pitch.
     3. Step I-Step VI / theorem explanation.
     4. Load-Bearing Evidence graphics: photometric metrics, structural-failure
        boundary map, Mesa envelope, coarse-graining proof ladder.
     5. Application motion rail.
     6. Working systems / applications grid or CTA.
   - Rationale: the page should establish claim, apparatus, and evidence before
     it asks the reader to scan every workbench and product expression.

7. **Clean public/internal boundaries.**
   - Move the `applications-gallery.html` bottom "Next Gallery Pass" block to
     an internal roadmap doc or remove it from the public page.
   - Public applications copy should show current tiers and links, not the
     maintenance checklist.

8. **Normalize top navigation.**
   - Remove `Mesa` as a persistent top-nav item unless the whole nav becomes an
     evidence menu.
   - Candidate public nav: `About`, `Atlas`, `Legend`, `Applications`, `Chat`,
     `Docs`, `GitHub`.
   - Keep mobile density in mind; if this is too wide, use `About`, `Atlas`,
     `Applications`, `Docs`, `GitHub` in the header and surface `Legend` /
     `Chat` in a secondary homepage evidence/inspection strip.
   - Apply the same nav decision across `index.html`, `about.html`,
     `applications-gallery.html`, `sundog.html`, `legend.html`, `mesa.html`,
     `chat.html`, and `docs/index.html`.

9. **Hero motto and leading statement pass.**
   - Replace `The math draws the sky.` with `Math draws the sky.`
   - Replace `An atlas for sundogs, calibrated against the photographs.` with
     `Halo alignment` or a slightly more legible variant such as
     `Halo alignment from indirect traces.`
   - Replace the current value prop with a disciplined version of:
     `Explore our discovery of the mathematical formula h(x) that describes
     solar phenomena and builds robots.`
   - Before shipping that exact line, check whether "builds robots" overclaims
     the current evidence. Safer candidate:
     `Explore the h(x) geometry behind solar halos, then follow the same
     traceability pattern into robots, agents, and workbenches.`

Checks:

```powershell
rg -n "Static hero guide|The hero is one scene|Next Gallery Pass|Mesa</a>" index.html applications-gallery.html about.html sundog.html legend.html mesa.html chat.html docs/index.html
rg -n "data-version=\"v1.1\"|Field-not-reward|data-definition|glossary-link" index.html
npm run build
npm run chat:eval:static
npm run chat:eval:phase3
npm run chat:eval:phase3:adversarial
npm run chat:eval:phase3:differential
npm run chat:eval:phase4
```

Browser screenshots:

- 390 px: hero through restored pitch.
- 390 px: evidence graphics before rail.
- 760 px: nav/header plus first two sections.
- 1280 px: full top-down section order through the rail.
- Reduced motion mode if the animated hero returns.

Exit:

- No standalone static hero guide remains.
- The old elevator pitch body is restored and glossary-linked.
- Step I-Step VI cards are the only hero explanation cards.
- Evidence pillars with graphics appear before working applications.
- Public application gallery no longer exposes internal maintenance notes.
- Header nav is consistent across top-level pages.
- Hero copy uses the new motto direction without adding an unsupported claim.

## P2 - After The First Wave

These are good follow-through once the initial public statement is out.

### 8. Young-reader / high-legibility path

**Solo status:** first sketch is soloable; reader simulation should be a later
separate pass.

Problem:

There are approachable demos, but no deliberate "for young readers" layer.

Steps:

1. Decide the surface:

   - a standalone `plain-language.html`; or
   - a docs page plus short callouts on About / Home.

2. Draft the first explanation:

   > Sundog is about using clues when you cannot see the answer directly.

3. Define three examples:

   - shadow as clue;
   - pressure as clue;
   - field reading as clue.

4. Add one boundary example:

   - a clue can be too weak, or point to the wrong thing.

Checks:

- Can a nontechnical reader define target, clue, and failure boundary after one
  page?
- Does the page avoid baby talk and keep the science honest?

Exit:

- One plain-language draft exists.
- No public claim is strengthened by the simplified version.

### 9. Bayesian floor across workbenches

**Solo status:** roadmap insertion is soloable; actual baselines are not.

Problem:

The project should not spin up a `bayes_v_sundog.md` culture-war doc. The
right move is a Bayesian-optimal or information-theoretic baseline inside each
workbench.

Steps:

1. Add a "Bayesian floor / information baseline" row to each relevant roadmap:

   - core photometric;
   - three-body;
   - Balance;
   - Pressure Mines;
   - future vortex/wishing-well toy.

2. For each row, specify:

   - hidden state;
   - observation channel;
   - optimal estimator/controller reference;
   - metric;
   - what would falsify the Sundog advantage.

3. Keep this as baseline discipline, not a new Bayes-vs-Sundog identity track.

Checks:

```powershell
rg -n "Bayesian floor|Bayes-optimal|information baseline" docs internal\anniversary
```

Exit:

- Public claims can compare Sundog to a positive information-theoretic baseline,
  not only to oracle/random/naive references.

### 10. Post-launch critique capture

**Solo status:** soloable if critiques are collected manually.

Problem:

The rollout will produce attacks. They should become roadmap entries, not
comment-section fog.

Steps:

1. Create a post-launch addendum in [`attack_vectors.md`](attack_vectors.md).
2. For each serious critique, record:

   - source/channel;
   - exact claim attacked;
   - strongest version of the attack;
   - cost if true;
   - current defense;
   - next action.

3. Classify:

   - fatal if true;
   - scope reduction;
   - wording fix;
   - future experiment;
   - misunderstanding already handled by docs.

Checks:

- Every serious critique has either a doc link or a new work item.
- No critique is dismissed because of tone alone.

Exit:

- Within 72 hours of launch, `attack_vectors.md` has a post-launch addendum
  and at least one next-action owner per serious critique.

## Quick Search List

Run these before final launch copy:

```powershell
rg -n "statistically indistinguishable|indistinguishable|solves alignment|universal theorem|sidesteps Goodhart|cannot be reward-hacked|Phase [0-9]+|phase [0-9]+|oracle" *.html docs internal
```

## Launch Gate

The public statement may ship when:

- [x] P0.1 terminal-accuracy wording is safe on the edited launch-facing set;
      rerun the final search before posting.
- [ ] P0.2 oracle-leakage receipt exists or the claim is demoted.
- [ ] P0.3 Mesa cliff language is bounded.
- [x] P0.4 geometry rendered-vs-anchored distinction is visible.
- [ ] `first_public_statement.md` passes the public guardrails in
      `anni_spam_roadmap.md`.

The broader social rollout may continue while P1/P2 work proceeds, as long as
posts keep linking to the boundary language.
