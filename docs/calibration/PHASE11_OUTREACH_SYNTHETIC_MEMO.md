# Phase 11 Outreach Synthetic Dispatch Memo

Filed: 2026-05-14  
Status: **wave-1 dispatched + consolidated + 15 patches applied 2026-05-14;
wave-2 independent re-dispatch + consolidation 2026-05-14 (see §8); wave-2
surfaces 3 send-blocking + 14 recommended findings, none of which have been
auto-applied pending user review.**  
Purpose: pre-deployment bounce test for the Phase 11 outreach surfaces after
the Phase 10 attack campaign, re-audit, specialist handoff rewrite, public
ratchet wave, and outreach-packet rewrite.

This memo is the Phase 11 analogue of
[`PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`](PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md),
but the persona roles are external-reviewer impersonators rather than
internal optical-claim auditors. The goal is to find places where the
outreach packet, cover materials, or tiering could bounce off real reviewers
before anything is sent externally.

## 0. Controlling Inputs

Load-bearing status:

- [`PHASE10_OPTICAL_REAUDIT_MEMO.md`](PHASE10_OPTICAL_REAUDIT_MEMO.md) -
  governs the post-pass route taxonomy and re-audit clearance.
- [`PHASE10_OPTICAL_AUDIT_HANDOFF.md`](PHASE10_OPTICAL_AUDIT_HANDOFF.md) -
  specialist-tier handoff, rewritten and cleared for external specialist use.
- [`PHASE11_OUTREACH_BRIEF.md`](PHASE11_OUTREACH_BRIEF.md) - Phase 11
  audience order, deployment methodology, and outreach posture.
- [`../SUNDOG_OUTREACH_PACKET.md`](../SUNDOG_OUTREACH_PACKET.md) - shared
  outreach packet for science-communications and Wikipedia-adjacent tiers.
- [`../PHASE10_ATTACK_ROADMAP.md`](../PHASE10_ATTACK_ROADMAP.md) -
  campaign-provenance source. Required for Persona A claim-verification:
  any "did A1b actually fix the formula?" or "was C1 actually run?"
  question routes here. The eight-pass status stamps + the §6 hedge
  inventory are the canonical record of what was repaired and what
  remains an open question.

Held back from Phase 11 lead framing:

- [`../MESA_CROSSOVER_NOTE.md`](../MESA_CROSSOVER_NOTE.md)
- [`../SUNDOG_V_GRAVITY.md`](../SUNDOG_V_GRAVITY.md)

Those documents record the ratcheted two-substrate / forward-inverse
asymmetry receipt, but Phase 11 outreach does not lead with that framing. It
may be discussed only if a specialist asks.

## 1. Current Route Taxonomy To Preserve

| route | outreach status | allowed wording |
| --- | --- | --- |
| Parhelion offset -> h | promoted, post-hedged | strict 3-photo subset: p2, p7, p13; p13 carries a low-lever caveat |
| CZA apex -> h | not promoted | coverage-gated; only p2 is in-window and measured after A1b/A2 |
| Supralateral -> h | not promoted | structural-discrimination gated; h-spread below measurement noise |
| Tangent-arc curvature -> h | not promoted | coverage + detector/anchoring tension: C5 manual samples recover p2, but C6 matched-filter falsifies the natural extension on the same b* substrate; specialist re-anchoring is the verify gate |

Retired phrases remain retired:

- "passes residual gate on every eligible photo"
- "three independent failure layers"
- class-level tangent-route failure
- C5 manual tangent recovery as a promoted route before specialist
  re-anchoring or alternative-substrate confirmation
- CZA residual-gate failure as the live issue
- broad inverse recovery across the full rich-display slate

## 2. Persona Dispatch

### Persona A - Atmospheric-Optics Specialist

Reviewer-impersonator role: read the specialist handoff and skim the outreach
packet as a halo specialist on a 30-minute time budget.

Primary surfaces:

- [`PHASE10_OPTICAL_AUDIT_HANDOFF.md`](PHASE10_OPTICAL_AUDIT_HANDOFF.md)
- [`../SUNDOG_OUTREACH_PACKET.md`](../SUNDOG_OUTREACH_PACKET.md)
- [`PHASE10_OPTICAL_REAUDIT_MEMO.md`](PHASE10_OPTICAL_REAUDIT_MEMO.md)

Surface:

- wrong primitive or wrong regime calls;
- formula or cutoff mistakes;
- places where textbook optics are framed as project contribution;
- literature-fidelity gaps;
- whether the audit-survived parhelion wording reads as honest.

Output format:

| item | verdict | note |
| --- | --- | --- |
| primitive taxonomy | sound / caveat / pushback / out-of-area | _pending_ |
| formula fidelity | sound / caveat / pushback / out-of-area | _pending_ |
| parhelion claim boundary | sound / caveat / pushback / out-of-area | _pending_ |
| tangent C5/C6 tension framing | sound / caveat / pushback / out-of-area | _pending_ |
| external-send readiness | sound / caveat / pushback / out-of-area | _pending_ |

### Persona B - Science-Communications Editor

Reviewer-impersonator role: read the brief, packet, and workbench as a
technically literate editor evaluating whether the artifact is understandable
and worth linking.

Primary surfaces:

- [`PHASE11_OUTREACH_BRIEF.md`](PHASE11_OUTREACH_BRIEF.md)
- [`../SUNDOG_OUTREACH_PACKET.md`](../SUNDOG_OUTREACH_PACKET.md)
- live workbench: `https://sundog.cc/sundog.html`

Surface:

- whether the claim license is obvious;
- whether the originality boundary is legible;
- whether the math summary is too dense or too thin;
- whether the disclaimers are strong enough without burying the artifact;
- whether a non-specialist reader can tell what is worth clicking.

Output format:

| item | verdict | note |
| --- | --- | --- |
| claim license clarity | sound / caveat / pushback | _pending_ |
| educational framing | sound / caveat / pushback | _pending_ |
| jargon load | sound / caveat / pushback | _pending_ |
| cover-note readiness | sound / caveat / pushback | _pending_ |
| public-copy revision needed | yes / no | _pending_ |

### Persona C - Wikipedia-Adjacent Reviewer

Reviewer-impersonator role: read the packet and live workbench as an
external-link reviewer applying conservative Wikipedia-style discipline.

Primary surfaces:

- [`../SUNDOG_OUTREACH_PACKET.md`](../SUNDOG_OUTREACH_PACKET.md), especially
  section 4;
- live workbench: `https://sundog.cc/sundog.html`;
- current literature attribution in the packet and geometry roadmap.

Surface:

- whether the artifact is useful as an explanatory adjunct rather than a
  primary-source claim;
- whether literature attribution is visible enough;
- whether any wording risks original-research interpretation;
- whether section 4's Wikipedia-adjacent suggestions are appropriately
  tiered after specialist/editorial review;
- what would need to change before an external-link proposal is defensible.

Output format:

| item | verdict | note |
| --- | --- | --- |
| external-link usefulness | sound / caveat / pushback | _pending_ |
| original-research risk | low / medium / high | _pending_ |
| literature attribution | sound / caveat / pushback | _pending_ |
| tier-3 timing | sound / caveat / pushback | _pending_ |
| send readiness | yes / no | _pending_ |

## 3. Verified Findings That Should Change Outreach

Persona dispatch executed and consolidated 2026-05-14. Findings below survived
the §5 verification gate (cross-checked against the controlling inputs, source
docs, executable code where applicable, and WP:EL / WP:NOR discipline where
applicable). Findings are grouped by send-blocking severity.

### 3a. Send-blocking findings (must fix before any external send) — *all 8 applied 2026-05-14*

| # | finding | surfaced by | verified against | required change |
| ---: | --- | --- | --- | --- |
| 1 | **46° halo "plate-oriented" mislabeling.** Packet §1 row 2 says "from the 90° prism path through plate-oriented hexagonal ice." The 46° halo is from *random-orientation* hexagonal columns; *plate-orientation* belongs to CZA / supralateral. | Persona A | Greenler 1980 ch. 2 standard text; conflates two crystal-orientation families | Rewrite packet §1 row 2 source/origin column to: "from the 90° face-pair (basal + prism) refraction in randomly oriented hexagonal ice columns" |
| 2 | **WP:NOR risk in §4 CZA proposal.** "Computed exactly as 32.196° in `scripts/cza_formula.py` per Pass A1a of the Phase 10 attack campaign" reads as project correcting Wikipedia from its own four-significant-figure precision against the article's stated ~32°. | Persona C | WP:NOR / WP:OR discipline | Replace with "matches the literature ~32° visibility cutoff (Greenler ch. 4)"; drop the 32.196° precision and the campaign nomenclature from any external-link-supporting copy |
| 3 | **WP:ELNO primary-research trap in §4 Sun dog link blurb.** "Photograph calibration evidence" advertises the project's own empirical work as link content. | Persona C | WP:ELNO #19 (links primarily promoting a viewpoint); WP:ELNO primary-research | Replace link blurb with literature-anchored single-claim hook: "Interactive demonstration of the `R₂₂ / cos(h)` parhelion-offset relation (Greenler 1980, ch. 3)" |
| 4 | **§4 proposals 3–6 fail WP:ELNO #1 ("unique resource" test).** The 22° halo, 46° halo, Vädersoltavlan, and Halo (optical phenomenon) entries point to the same atlas URL with project-framing blurbs ("single calibrated parametric model," "all four 46°-tangent features toggleable," "derivable consequences of a small set of circles") and would be reverted as project marketing across multiple articles. | Persona C | WP:ELNO #1 (unique resource), WP:ELNO #19 (promotional); per-article verdicts all "pushback" | Either (a) drop these four proposals from §4 and keep only Sun dog + CZA, or (b) rewrite each blurb to a literature-anchored single-claim hook AND explicitly acknowledge the WP:ELNO #1 risk / `atoptics.co.uk` overlap in the §"Editorial caveats" subsection. **Recommendation: (a).** |
| 5 | **Campaign-internal vocabulary on editorial-tier surfaces.** "Pass A1b/B1/B2/C1," "lever 5.52%," "post-Pass-B2," "post-A1b," "anchor-noise-bounded," "post-pass failure-mode taxonomy" appear in the brief and packet — surfaces meant to serve a non-specialist editorial tier. | Persona B (with Persona A independently flagging same pattern) | Brief §5; packet §1 row 5b, §2.3, §3 Note on R₄₆ | Add a one-line glossary near top of packet §0 OR rewrite editorial-tier surfaces to drop internal-pass nomenclature (keep it in handoff + roadmap for the specialist tier) |
| 6 | **No editorial-tier cover note.** Brief §10 is explicitly the specialist cover note. The editorial tier (Persona B audience) has no cover artifact, and the editorial-tier reader can't be cold-pitched without one. | Persona B (with explicit draft in B4) | Brief §10 inspection; brief §14 item 3 status PARTIALLY DRAFTED | Add an editorial-tier cover note as brief §10b or as a new "§16. Editorial cover note" using Persona B's B4 draft as the starting text |
| 7 | **No "what you'll see when you click" workbench description.** Packet §3 Path A says how to use the live page but never describes its landing state, so an editor or WP reviewer who hasn't yet clicked can't predict what they'd encounter. | Persona B | Packet §3 Path A inspection | Add a single-paragraph workbench landing description to packet §0 or as a new sub-bullet at the top of §3 Path A: "When you load the page you see X, Y, Z" |
| 8 | **`scripts/cza_formula.py` docstring conflates "sodium-D line" with "visible-band centroid."** Sodium-D = 589 nm (yellow); visible-band centroid ≈ 550 nm; ice n varies from 1.308 (yellow) to 1.317 (violet); the disappearance altitude varies by ~0.1° across the visible chromatic spread. A specialist would flag the two references in one sentence. | Persona A | Docstring inspection vs. standard ice-n wavelength dependence | Rewrite the n=1.31 commentary in the docstring to Persona A's suggested phrasing: "n = 1.31 is the visible-band centroid for ice; the CZA disappearance altitude varies by ~0.1° across the visible chromatic spread, which is below this atlas's measurement precision" |

### 3b. Recommended improvements (not send-blocking; should land before tier-2 / tier-3 deployment) — *7 of 7 applied 2026-05-14 where actionable; #13 (provenance overcounting) partially addressed via §0 compression*

| # | finding | surfaced by | required change |
| ---: | --- | --- | --- |
| 9 | **"Equivalent to the standard great-circle ↔ parhelic-circle projection geometry"** (packet §1 row 3) overstates the exactness of `R₂₂ / cos(h)` — it's the small-angle / planar screen-projection, not exact great-circle. At p7 (h = 59.4°) the small-angle error is non-trivial. | Persona A | Soften packet §1 row 3 to: "the standard small-angle screen-projection of the parhelion's azimuthal offset from the sun; see Tape (1994) §3 for the exact great-circle treatment." Add a one-line note to packet §3 Path B clarifying which one `parhelion-geometry.mjs` implements. |
| 10 | **"Three structurally different failure modes" reads as taxonomic claim** rather than post-hoc description. Appears in packet §0, handoff §1 / §2.7, gravity ledger receipt, mesa crossover note. A specialist asks "are these failure modes pre-registered?" | Persona A | Soften the wording on the most-public surfaces (packet §0, handoff §2.7 audit-survived sentence) to "fail for three different reasons (dataset coverage, h-discrimination physics, and detector tooling respectively)." Leave the in-house surfaces (dispatch memo §1, re-audit memo §94, attack roadmap) using the original phrasing. |
| 11 | **"Ratchet" / "ratcheted" used as buzzword on editorial-tier surface.** Editors won't know if it means tightened, loosened, or logged. | Persona B | In editorial-tier text (brief, packet), replace "ratcheted" with "tightened" or "revised". Keep "ratchet" as a technical verb in the internal-process surfaces (re-audit memo, attack roadmap, dispatch memo). |
| 12 | **Packet §0 tier-placement framing clutters reading.** Editors at the tier shouldn't have to read meta-framing about which tier they are. | Persona B | Soften or relocate packet §0's tier-placement block to a footer / appendix; lead with the substantive packet content. |
| 13 | **Provenance / meta-framing overcounting across packet + brief + handoff.** Campaign provenance narrated multiple times (handoff §1 + §4.2; packet §0 + §6; brief §0 + §12 + §14 + §15). Reads as defensive at editorial-tier density. | Persona A (narrative) | Compress provenance language to one paragraph per surface; let the audit-survived sentence do its own work. Editorial-tier surfaces especially. |
| 14 | **§3 R₄₆ note long enough to read as "audit theatre."** The historical paragraph re-explains the pre-audit reasoning before naming Pass A1b's retirement. | Persona A | Compress §3 R₄₆ note: lead with "Pre-audit, the workbench had a known-wrong WB_R₄₆ value matched to a photo-mean ratio; this was a confound, not a measurement, and was retired in Pass A1b." Drop the longer historical narration. |
| 15 | **Brief §6 "two-substrate field-shape pattern claim held out" wording** is jargon-loaded enough to invite curiosity in the direction the brief is trying to avoid. | Persona B | Soften the held-back description to: "Held out of Phase 11 outreach (conservative-first, not retired): a broader cross-substrate framing the team has filed in separate research surfaces; available on request once tier-1 + tier-2 clear." |

## 4. Unverified / Contradicted Persona Claims

Persona outputs that did not survive consolidator verification, or that fall
outside the dispatch's scope, are filed here rather than into §3.

| # | claim | persona | why deferred or dropped |
| ---: | --- | --- | --- |
| 1 | "We are not aware of an existing public resource that surfaces the full vocabulary as one parametric model" line (packet §2 original-contribution item 1) is self-asserted novelty and risks WP:NOR if quoted on a WP talk page. | Persona C | **Deferred, not dropped.** The wording itself is in the packet's claim-license context (internal-facing), not in any external-link-supporting copy. The risk Persona C names is future-state (talk-page defense, not current). Filed for the talk-page reply discipline: do not lean on this line in any tier-3 / WP-side reply. |
| 2 | Workbench attribution must be above-the-fold; Greenler / Tape / Cowley named on `sundog.cc/sundog.html` before "Halo Atlas" project terminology. | Persona C | **Deferred to a workbench-audit follow-up step.** The persona explicitly cannot open the workbench in this dispatch; the finding is verifiable but not verified here. Filed as a pre-tier-3 requirement: before any §4 external-link proposal is defensible, audit the workbench HTML for above-the-fold literature attribution. |
| 3 | Wavelength-dependent n choices for CZA (1.308 at 589 nm vs 1.317 at violet) may shift the 32.196° disappearance altitude by ~0.1°. Whether this matters for the atlas's measurement precision is an open question. | Persona A (A4 out-of-area) | **Deferred.** Persona A flags but punts as out of 30-min budget. Already partially addressed by finding #8 in §3a (docstring wording fix); the deeper precision question is filed for a future literature pass and is not send-blocking. |
| 4 | Helic / sub-parhelic features as additional inverse-route candidates. | Persona A (A4 out-of-area) | **Already in scope as an open question.** Handoff §2.1 already asks specialists about additional candidate routes; this is not a new finding requiring patch. |
| 5 | Whether the upper-tangent route's canonical inverse handle in the literature is opening-angle, curvature, or apex-height. | Persona A (A4 out-of-area) | **Already in scope as an open question.** Handoff §2.3 already asks specialists about the literature-standard tangent detector. Not a new finding. |
| 6 | Cross-substrate "two-substrate field-shape pattern" claim review. | Persona A (A4 out-of-area) | **Explicitly out of dispatch scope.** Held out of Phase 11 per design; the framing is recorded in the mesa crossover note + gravity ledger and is not part of tier-1 / tier-2 / tier-3 outreach. |

## 4a. Consolidator notes on the persona dispatch itself

Three cross-persona convergences worth flagging at meta-level:

1. **Personas A and B independently converged on the "campaign-vocabulary leak" pattern** (Persona A: "Pass A1b/B1/B2/C1 used as in-group code"; Persona B: "lever / ratchet / post-pass / Pass A1b reads as noise to a non-specialist editor"). This is the highest-confidence finding in the dispatch — two independent reviewers from different tiers surfaced the same pattern in the same wording instances. Reflected in §3a finding #5 and §3b findings #10, #11, #12, #13.

2. **All three personas read the campaign discipline as a positive signal** (Persona A: "more honest than most peer-reviewed work in this corner"; Persona B: "the product of disciplined internal review — which is exactly what an editor at my tier wants to see"; Persona C: "the project is being appropriately conservative *on paper*"). The verify-gate discipline is itself a load-bearing piece of the outreach case; the §3a wording changes preserve that signal while fixing specific overreach risks.

3. **No persona surfaced a CZA formula correctness issue** post-A1b. Persona A's spot-check confirmed the formula is canonical; Persona C's WP:NOR concern is about the *wording precision* in the link blurb (32.196° vs ~32°), not the formula itself. The Phase 10 attack campaign's verify-gate-as-load-bearing principle holds in this Phase 11 dispatch: the formula is sound; the bounces are about how the result is framed for external readers.

## 5. Consolidator Verification Gate

The consolidator must re-check every proposed change against:

- the live artifact and workbench routes;
- `scripts/cza_formula.py` and `scripts/test_cza_formula.py` for CZA formula
  claims;
- `scripts/overlay_calibrate.py` for overlay behavior;
- [`PHASE10_OPTICAL_REAUDIT_MEMO.md`](PHASE10_OPTICAL_REAUDIT_MEMO.md) for the
  route taxonomy;
- [`../SUNDOG_OUTREACH_PACKET.md`](../SUNDOG_OUTREACH_PACKET.md) for the current
  public/editorial surface;
- [`PHASE10_OPTICAL_AUDIT_HANDOFF.md`](PHASE10_OPTICAL_AUDIT_HANDOFF.md) for
  specialist-facing claims.

Verification standards:

- Persona claims about formulas must be backed by executable code or a
  specific cited literature line in the packet/handoff.
- Persona claims about outreach overstatement must quote the exact wording
  and name the destination surface.
- Persona claims about Wikipedia-adjacent risk must identify whether the risk
  is originality, self-promotion, sourcing, external-link suitability, or
  reader utility.
- Claims that are plausible but not verified stay in section 4.

## 6. Recommended Execution Order After Dispatch

1. ~~Run Persona A/B/C passes against the controlling inputs.~~ **DONE
   2026-05-14.** Three independent persona drafts produced via
   subagent dispatch; outputs preserved in agent return values.
2. ~~Consolidate verified findings into section 3 and dropped claims
   into section 4.~~ **DONE 2026-05-14.** All persona findings routed
   through the §5 verification gate; 15 verified findings to §3 (8
   send-blocking + 7 recommended); 6 deferred to §4 (out-of-scope or
   future-state). Three cross-persona convergences recorded in §4a.
3. ~~Patch the packet, handoff, brief, or cover-note artifacts only
   where §3 requires it.~~ **DONE 2026-05-14.** All 8 send-blocking
   patches and 7 recommended patches applied: cza_formula.py docstring
   (#8); packet §0 vocabulary glossary + tier-placement compression +
   ratchet→revised vocab + post-pass-taxonomy table softening (#5,
   #10, #11, #12); packet §1 rows 2 + 3 (#1, #9); packet §3 Path A
   workbench landing description (#7); packet §3 R₄₆ note compression
   (#14); packet §4 wholesale rewrite — dropped 4 of 6 proposals,
   rewrote Sun dog + CZA blurbs with literature-anchored hooks (#2, #3,
   #4); brief §6 cross-substrate held-back wording softened (#15);
   brief §10b editorial cover note added from Persona B's B4 draft
   (#6); handoff §1 audit-survived sentence softened (#10). Finding
   #13 (provenance overcounting) partially landed via §0 compression
   in the packet; further compression deferred since editorial-tier
   testing of the current state cleared.
4. ~~Re-run a final stale-phrase scan for the retired phrases in
   section 1.~~ **DONE 2026-05-14.** All public-surface stale phrases
   ("plate-oriented hexagonal ice", "photograph calibration evidence",
   "computed exactly as 32.196", "single calibrated parametric model",
   "every eligible photo") are zero in the packet outside retract-
   contexts. Two retract-context hits remain in packet §4 editorial-
   caveats where dropped phrasings are quoted to explain the pruning
   — correct usage. All §1 retired-phrase list hits are in
   retract-language contexts (Brief §5 retired-phrases list; dispatch
   memo §1 same; re-audit memo's explicit retirement).
5. **Now ready:** assemble tier-specific outreach packet + small named
   outreach list per tier. The four-doc + dispatch-memo Phase 11
   surface is internally coherent at 2026-05-14.

## 7. Open Deployment Gates

- Phase 11 synthetic-persona pass: ~~**pending**~~ **CLEARED 2026-05-14.**
- Tier-1 specialist cover artifact (handoff): ✅ landed 2026-05-14
  (with §2.7 audit-survived sentence softened per finding #10).
- Tier-2 editorial cover artifact (brief §10b): ✅ landed 2026-05-14
  (using Persona B's B4 draft).
- Tier-3 Wikipedia-adjacent artifacts (packet §4): ✅ pruned 2026-05-14
  to Sun dog + CZA only, with literature-anchored link blurbs.
- Workbench above-the-fold attribution audit: **pending** — Persona C
  raised this as a §4 deferred finding; before tier-3 external-link
  proposals are defensible, audit `sundog.cc/sundog.html` for Greenler /
  Tape / Cowley attribution visibility above-the-fold.
- Named outreach list per tier: **pending**.
- Homepage elevator-pitch v1.2: **held pending explicit approval**
  (per re-audit memo §107 item 3).
- `chat/claim_map.json` ratchet: **held pending explicit approval**
  (per re-audit memo §107 item 3).
- Workbench above-the-fold attribution audit: ~~**pending**~~
  **AUDITED 2026-05-14 (wave-2 §8.4); GATE OPEN — verified blocker
  for tier-3.** Two independent WebFetches (Persona B + Persona C
  in wave-2 dispatch) confirm Greenler / Tape / Cowley appear only
  in §7 "History & reading" below the fold of `sundog.cc/sundog.html`.
  Wave-2 §8.6a finding W2 specifies the required credit-strip change.
- Phase 11 synthetic-persona pass: ~~**CLEARED 2026-05-14**~~
  → **wave-1 cleared 2026-05-14; wave-2 independent re-dispatch
  2026-05-14 (§8) surfaces 3 send-blocking + 14 recommended findings.**
  Wave-2 reading is more conservative than wave-1.

## 8. Wave-2 Independent Re-Dispatch

Filed: 2026-05-14 (same day as wave-1, separate run).
Status: **wave-2 dispatch executed; consolidator verification gate applied; findings filed below.**

This section records a fresh independent three-persona dispatch run
after the wave-1 patches landed, at user request. Methodology: three
parallel general-purpose subagents, one per persona, each prompted
with the persona role + primary surfaces and instructed to skip
§3-§7 of this memo so they could form independent verdicts. Personas
B and C also performed WebFetch against the live workbench.

Wave-2 purpose: do the wave-1 §3 patches actually resolve the bounces
they were meant to resolve, and does an independent pass surface
new bounces?

### 8.1 Persona A wave-2 — Atmospheric-Optics Specialist

| item | verdict | note |
| --- | --- | --- |
| primitive taxonomy | caveat | Persona A flagged row-7 supralateral/infralateral wording; consolidator hedge in §8.4 — not promoted as a verified finding pending further specialist review |
| formula fidelity | pushback | "n = 1.31" with 32.196° precision survives in packet §1 row 5 without chromatic-dispersion hedge; 46/22 ratio cited as "2.091" with precision-to-3-figures unbacked by physics derivation |
| parhelion claim boundary | sound with caveat | strict-3-photo wording is honest, but p7 at h = 59.4° is in the regime where column-plate parhelion mechanism is suspect — possible production-mechanism mis-attribution |
| tangent C5/C6 tension framing | sound with caveat | "C5↔C6 substrate tension" framing reads correctly as route-open; caveat is that `R_uta/R₂₂` is not the canonical literature inverse handle for tangent arcs |
| external-send readiness | pushback | wording overclaims survive: p7 parhelion claim, "non-obvious use" in packet §2 item 2, 2.091 precision, 32.196° precision, upper-tangent-curvature implied canonicalness |

**Significant wave-2 findings from Persona A:**

1. **A-N1.** p7 (h = 59.4°) parhelion claim may mis-attribute production
   mechanism. At h ≈ 60° the column-plate parhelion mechanism is suspect —
   Tape (1994) §3 and Greenler ch. 3 both treat parhelia as
   low-to-moderate-h plate-refraction features; circumscribed-halo
   brightness on the parhelic circle competes. Either p7 has been
   mis-identified as a parhelion display, or — if plate-population
   is real at h = 59.4° (unusual) — `R₂₂/cos h` is not the right
   kinematic for that lobe.
2. **A-N2.** Upper-tangent-arc curvature `R_uta/R₂₂` is not a canonical
   literature inverse handle for h. Tape 1994 §6 + Cowley tangent-arcs
   page treat tangent arcs as one continuous family with circumscribed
   halo; the literature shape parameter is opening angle / azimuthal
   extent, not circle-fit radius. Handoff §2.3 (v) hedges — downstream
   verdict tables and packet text treat it as settled.
3. **A-N3.** "2.091" ratio in packet §1 row 5b + §3 R₄₆ note is
   presented with precision the underlying physics doesn't carry. The
   integer-label ratio 46/22 ≈ 2.091 happens to coincide with the
   minimum-deviation ratio (≈ 45.7/21.84 ≈ 2.092) but the packet doesn't
   name which derivation is being cited; the three-figure precision
   reads as literature-derived when it is integer-label division.
4. **A-N4.** "Non-obvious use of the formula" wording in packet §2 item 2.
   The inverse application of `parhelion_offset = R₂₂/cos(h)` is ≥ a
   century old (Pernter & Exner; Greenler ch. 3; Cowley atoptics
   parhelia page; Tape 1994 §3 has the exact great-circle treatment).
   Calling it "non-obvious" frames standard atmospheric optics as
   project contribution.
5. **A-N5.** CZA disappearance "~32.196°" persists in packet §1 row 5
   with no chromatic-dispersion hedge. Wave-1 §3a #8 patched the
   cza_formula.py docstring; the dispersion note did not propagate
   to the packet.
6. **A-N6.** CZA "tangent to 46° halo at top" framing in §1 row 4 is
   geometrically correct only in a narrow h-window. The §4 editorial
   caveats subsection has the right hedge; pull it up into §1 row 4
   so the framing is consistent.
7. **A-N7.** Supralateral "~0.5° variation across h = 0-22°" claim in
   handoff §2.2 is cited to internal audit memo §2 item 12 rather
   than to primary literature (Tape 1994 §6.4 or the actual derivation
   source).

### 8.2 Persona B wave-2 — Science-Communications Editor

| item | verdict | note |
| --- | --- | --- |
| claim license clarity | caveat | §2's standard-vs-original split is genuinely clean — but lives at line ~140 behind ~140 lines of audit-internal preamble |
| educational framing | pushback | packet §0 is internal-ops framing handed to an editor; the actual educational lede had to be dug for |
| jargon load | pushback | "Pass A1b/C2/C4/C5/C6," "audit-survived," "verify gate," "anchor-noise-bounded," "lever," "post-hedged," "matched-filter on halo-subtracted b* substrate" leak into editorial-tier surface. Glossary helps but signals the problem. |
| cover-note readiness | caveat | §10b cover note broadly sendable but trailing sentence undermines |
| public-copy revision needed | yes | (a) packet §0 non-audit-vocabulary opener; (b) §10b trailing-sentence cut; (c) workbench above-the-fold attribution gap |

**Significant wave-2 findings from Persona B:**

1. **B-N1.** Packet opens to wrong audience. First substantive paragraph
   is meta-process ("This packet is the shared support artifact across
   three outreach tiers"). Cold editor on a 15-minute clock wants the
   artifact in sentence one.
2. **B-N2.** §10b cover note trailing sentence undermines: "A specialist
   audit pass already narrowed our claims; this is the editorial-tier
   follow-up." Reveals tier-2 pipeline position; "already narrowed our
   claims" prompts "what got walked back?" rather than "what should I
   look at?".
3. **B-N3.** Packet §1 row 4 CZA cell contains internal release-note
   prose. "the Phase 10 attack campaign's Pass A1b replaced the
   approximation with the literature formula" is release-note text
   inside a math summary row; belongs in §6 Maintenance.
4. **B-N4.** §4 Wikipedia content doesn't belong in editor-tier packet.
   ~80 lines of WP external-link strategy with WP:ELNO discussion
   makes the packet feel like a multi-pronged outreach campaign
   rather than a halo explainer.
5. **B-N5.** Workbench above-the-fold attribution gap (WebFetch
   verified). `sundog.cc/sundog.html` H1 is "A sun dog is geometry
   doing optics"; Greenler / Tape / Cowley appear only in §7
   "History & reading," below the fold. Packet's own §4 Editorial
   caveats names this as constraint (i) for tier-3 defensibility.
   The constraint is named but not met. Cross-confirmed by Persona C.

### 8.3 Persona C wave-2 — Wikipedia-Adjacent Reviewer

| item | verdict | note |
| --- | --- | --- |
| external-link usefulness | caveat | interactive slider for R₂₂/cos(h) is genuinely unique vs atoptics — but buried under project-branded full-atlas landing state |
| original-research risk | medium | §4 link blurbs are restrained; risk is in destination page H1 ("A sun dog is geometry doing optics") and §2 items 1-3 if quoted on a talk page |
| literature attribution | pushback | Greenler / Tape / Cowley NOT above-the-fold on the live page; in §7 only. Single biggest blocker. |
| tier-3 timing | sound | packet correctly defers tier-3 behind specialist + editorial review |
| send readiness | no | above-the-fold literature attribution must land on destination page first (packet's own §4 caveat names this as precondition; gate is not closed) |

**Significant wave-2 findings from Persona C:**

1. **C-N1.** Above-the-fold attribution gap on live workbench (WebFetch
   verified, independent of Persona B). Same finding as B-N5; two
   independent WebFetches converge.
2. **C-N2.** Per-article verdict on §4 surviving proposals.
   **Sun dog:** link blurb itself defensible, but landing state is
   project-branded full-atlas — would be reverted on WP:ELNO #19
   (promotional) and WP:ELNO #11 (no named author) grounds.
   **CZA:** oppose; single-threshold supplement is pedagogically
   thin; deep-link `#sun-altitude-binding` fragile.
3. **C-N3.** "Stellar Aqua LLC" corporate authorship is WP:SELFPUB
   yellow flag. Citation template in §4 names "Stellar Aqua LLC,
   2026"; a corporate-entity author with no named human attracts
   scrutiny atoptics.co.uk doesn't face.
4. **C-N4.** `phase3-tests.html` Quick-link is an OR-trap reachable
   from packet. "Interactive math-binding tests" reads as
   primary-research artifact; if a talk-page defender links it as
   evidence the atlas "works," that's WP:OR.
5. **C-N5.** §2 item 3 calibration evidence (strict 3-photo subset,
   lever percentages) is WP:OR if quoted on talk page. Internally
   a key audit-survived finding; quoted in WP defense, it becomes
   unpublished primary research on photographs.
6. **C-N6.** §3 reproducibility paths are WP:V-inversion if used as
   link defense. "Trust our results because you can rerun them" is
   the verifiability-not-truth inversion; WP reviewers want
   third-party sources, not reproducible project code.
7. **C-N7.** OR-risk delta principle. §4 link blurbs are defensible
   *in isolation*; risk is in the packet behind them. Packet §4
   Editorial caveats constraint (iii) is the right policy; extend
   it explicitly to §2 items 2 and 3.

### 8.4 Consolidator Verification Gate (wave-2)

Each wave-2 persona claim re-checked against the §5 standards
(formulas / executable code / exact wording / WP discipline naming).
Verification notes:

- **A-N1 (p7 parhelion mis-attribution).** Plausible specialist
  concern grounded in Tape 1994 §3 + Greenler ch. 3 framing of
  parhelia as low-to-moderate-h plate-refraction features. The
  specific claim about p7 has not been verified against the actual
  photograph (would require image inspection by a specialist). The
  persona's concern is the type of question the tier-1 outreach is
  meant to elicit. **Promoted as a tier-1 dispatch question rather
  than a unilateral patch.**
- **A-N3 (2.091 ratio precision).** Verified. The integer-label
  ratio 46/22 = 2.0909... is the literal derivation; the underlying
  minimum-deviation physics (22° halo inner edge ≈ 21.84°; 46° halo
  inner edge ≈ 45.7°) gives a ratio that *happens* to be ≈ 2.092,
  but presenting "2.091" to three figures without naming which
  derivation is being used is the wording concern.
- **A-N5 (CZA chromatic dispersion).** Verified against the formula:
  at threshold `cos²h = n² − 1`. For n(yellow, 589 nm) ≈ 1.308:
  h ≈ 32.49°. For n(violet) ≈ 1.317: h ≈ 31.04°. For n(red, ~670 nm)
  ≈ 1.306: h ≈ 32.77°. **Visible-band spread is ~1.5–1.8°, not
  ~0.1° as wave-1 §3a #8 docstring patch states.** Wave-1's
  docstring numerical claim appears off by ~10×. Open numerical
  reconciliation gate; see §8.4 hedge below.
- **A-N4 (non-obvious use).** Verified. The inverse application of
  `parhelion_offset = R₂₂/cos(h)` is ≥ a century old; calling it
  "non-obvious" in a project-contribution claim is overclaim.
- **B-N5 / C-N1 (workbench above-the-fold attribution gap).**
  Verified via two independent WebFetches (Persona B and Persona C
  both retrieved the page; both confirmed Greenler / Tape / Cowley
  appear only in §7 "History & reading" below the interactive
  atlas). Cross-references wave-1 §4 deferred finding #2 (filed
  as a pre-tier-3 requirement). Wave-2 has now performed the
  deferred audit. **Promotes the wave-1 deferred finding to
  verified blocker for tier-3.**
- **C-N3 (Stellar Aqua LLC corporate authorship).** Verified against
  packet §4 citation template; corporate-entity author with no
  named human is a WP:SELFPUB consideration.
- **A-N (primitive-taxonomy / row-7 supralateral wording).**
  **Not promoted.** Persona A's verdict-table note on row-7
  supralateral/infralateral "column-orientation 90° path" wording
  appears to be either a misreading or a subtle distinction; the
  packet's current row-7 wording is consistent with Greenler ch. 4
  standard usage. Re-checking with the specialist tier is
  appropriate but the consolidator does not patch.
- **Other findings.** Verified against quoted wordings + destination
  surfaces; all surface real text in the named files at time of
  dispatch.

**Resolution — chromatic-spread magnitude.** Wave-1 §3a finding #8
docstring patch quotes "~0.1° across the visible chromatic spread …
below this atlas's measurement precision." Consolidator re-derivation
from the script's own formula (lines 56-58: `h_disappear =
arccos(√(n² − 1))`):

| wavelength | n (ice) | h_disappear |
| --- | --- | --- |
| red (~670 nm) | 1.306 | 32.86° |
| yellow (~589 nm, Na-D) | 1.308 | 32.50° |
| visible centroid | 1.31 | 32.19° |
| violet (~404 nm) | 1.317 | 31.05° |

**Visible-band spread: ~1.8°.** Analytic check via implicit
differentiation: `dh/dn = −n / [√(2−n²) · √(n²−1)]`; at n = 1.31
this is ≈ −166°/unit-n, so for Δn = 0.011 (visible-band range),
Δh ≈ 1.83°. Wave-1 §3a #8 magnitude claim is off by a factor of
~18× and the "below this atlas's measurement precision" claim
is also wrong: 1.8° is well above the ~0.1° altitude precision
the photo-anchor JSONs work at. Practical consequence: the CZA
disappearance is gradual between ~31° (violet edge drops out first)
and ~33° (red edge drops out last), not a sharp 32.2° cutoff.

This is a wave-1 verification-gate miss: the §5 standard "persona
claims about formulas must be backed by executable code or a
specific cited literature line" should have caught the ~0.1° number
during wave-1 consolidation. Two corrections owed:
1. `scripts/cza_formula.py` docstring (lines 46-52) needs the
   ~0.1° / "below measurement precision" claim replaced with the
   ~1.8° / "gradual disappearance between ~31° and ~33°" framing.
2. Wave-2 W7 packet hedge wording in §8.6b below uses the corrected
   ~1.8° magnitude rather than the wave-1 ~0.1° magnitude.

### 8.5 Reconciliation Diff vs Wave-1

**Wave-1 findings re-confirmed (patches hold; not re-attacked by wave-2):**

| wave-1 # | finding | wave-2 status |
| ---: | --- | --- |
| 1 | 46° "plate-oriented" mislabeling | patch holds; wave-2 didn't re-surface |
| 3 | "Photograph calibration evidence" link blurb | patch holds; §4 rewrite survives |
| 4 | §4 proposals 3-6 fail WP:ELNO #1 | patch holds (pruned to 2); wave-2 §8.3 C-N2 finds even the 2 surviving on the edge |
| 6 | No editorial-tier cover note | patch holds (§10b landed); wave-2 §8.2 B-N2 refines (trailing-sentence cut) |
| 7 | No "what you'll see when you click" workbench description | patch holds; wave-2 didn't re-attack |
| 3b #9 | R₂₂/cos(h) exact great-circle | patch holds; wave-2 A-N4 attacks a different aspect ("non-obvious use") |
| 3b #10 | "three structurally different failure modes" | patch holds |
| 3b #11 | "ratchet"/"ratcheted" buzzword | patch holds |
| 3b #15 | brief §6 two-substrate held-back wording | patch holds |

**Wave-1 findings wave-2 STRENGTHENS / WIDENS:**

| wave-1 # | wave-2 extension |
| ---: | --- |
| 2 | wave-1 dropped 32.196° from external-link copy only; wave-2 A-N5 propagates the hedge to packet §1 row 5 with chromatic-dispersion footnote |
| 5 | wave-1 added §0 glossary; wave-2 B-N1 + jargon-load pushback says glossary itself is a tell and packet §0 should open with the artifact |
| 8 | wave-1 patched cza_formula.py docstring claiming ~0.1° spread; wave-2 consolidator re-derivation finds ~1.5–1.8° spread — docstring numerical claim may need second revision |
| 4 deferred #2 | wave-1 filed workbench attribution audit as pre-tier-3 requirement; wave-2 §8.4 promotes to verified blocker |

**Wave-1 findings wave-2 doesn't re-attack (residual status — patches stand):**

3a #1, #3, #6, #7; 3b #12, #13, #14. (§3b #12 packet §0 tier-placement
clutter softening is strengthened by wave-2 B-N1, but the wave-1
patch itself stands.)

### 8.6 Wave-2 New Findings (verified, not in wave-1)

#### 8.6a Send-blocking

Status legend: `pending — batch N` = queued for the named application
batch (see §8.6 application-order note below the table). `applied
2026-05-14 (batch N)` = patch landed. `deferred` = not in the planned
application sequence per user-approved workflow.

| # | finding | surfaced by | required change | status |
| ---: | --- | --- | --- | --- |
| W1 | **p7 (h = 59.4°) parhelion production-mechanism question.** Possible mis-attribution of circumscribed-halo brightness as parhelion in the strict 3-photo set. | Persona A | Add to the tier-1 specialist dispatch as an explicit question (handoff §2 or §7). Do NOT unilaterally drop p7 — that is the specialist's call. If specialist confirms mis-attribution, strict-subset rewrites cascade through handoff §1, packet §2 item 3, reaudit memo §94, brief §5. | **applied 2026-05-14 (batch 3)** |
| W2 | **Workbench above-the-fold attribution gap (WebFetch verified, two personas).** Greenler / Tape / Cowley only in §7 "History & reading" of `sundog.cc/sundog.html`. Packet §4 caveat names this as constraint (i) for tier-3 defensibility. | Persona B + C | Add a one-line credit strip immediately under the H1 on `sundog.cc/sundog.html`: e.g. "Geometry follows Greenler 1980 and Tape 1994; see History & reading for citations." Blocks tier-3 send until landed. | **applied 2026-05-14 (batch 4)** |
| W3 | **§10b cover note trailing sentence undermines editorial-tier framing.** | Persona B | Cut "A specialist audit pass already narrowed our claims; this is the editorial-tier follow-up." from brief §10b. End the note on "...we'd value your read on whether the page earns a link or a short write-up." | **applied 2026-05-14 (batch 1)** |

#### 8.6b Recommended (not send-blocking)

| # | finding | surfaced by | required change | status |
| ---: | --- | --- | --- | --- |
| W4 | **Upper-tangent-arc `R_uta/R₂₂` is not canonical literature inverse handle.** | Persona A | Pull handoff §2.3 (v)'s framing-question hedge into the verdict tables across handoff §1 and packet §0. Add a one-line note to packet §1 row 6 + §0 tangent row: "Note: the literature standard inverse for tangent arcs uses opening angle / arc extent (Tape 1994 §6), not circle-fit radius." | **applied 2026-05-14 (batch 3)** |
| W5 | **"2.091" ratio precision overclaim** in packet §1 row 5b + §3 R₄₆ note. | Persona A | Soften to: "matches the integer-label angular ratio 46/22 ≈ 2.09" OR cite a specific minimum-deviation derivation. Drop the three-figure precision. | **applied 2026-05-14 (batch 2)** |
| W6 | **"Non-obvious use of the formula" wording in packet §2 item 2.** | Persona A | Replace "is a non-obvious use of the formula" with "is applied as a measurement instrument; the inverse direction is treated in Tape 1994 §3 and on Cowley's atoptics parhelia page." | **applied 2026-05-14 (batch 2)** |
| W7 | **CZA "~32.196°" precision in packet §1 row 5 with no dispersion hedge.** Wave-1 §3a #8 magnitude claim is also off by ~18× (see §8.4 resolution). | Persona A + consolidator re-derivation | Replace "CZA disappears at h > ~32.196°" with "CZA disappears gradually between ~31° (violet edge) and ~33° (red edge) as the chromatic discriminant exceeds 1 at progressively longer wavelengths; visible-band centroid (n = 1.31) crosses at 32.19°." This is the dispersion-corrected wording; supersedes wave-1 §3a #8 docstring magnitude. The `scripts/cza_formula.py` docstring at lines 46-52 also needs the same correction. | docstring half **applied 2026-05-14 (batch 1)**; packet-hedge half **applied 2026-05-14 (batch 2)** |
| W8 | **CZA "tangent to 46° halo at top" framing in §1 row 4 lacks h-window scoping.** | Persona A | Pull the §4 editorial caveat ("any external-link copy that says 'the CZA is tangent to the 46° halo' should be hedged or scoped to the near-disappearance regime") up to §1 row 4 source-column. | **applied 2026-05-14 (batch 2)** |
| W9 | **Supralateral "~0.5° across h=0-22°" claim cited to internal audit memo.** | Persona A | Cite to primary literature (Tape 1994 §6.4 or the actual computation source) in handoff §2.2. | **applied 2026-05-14 (batch 2)** |
| W10 | **Packet §0 opens with meta-process; lift artifact framing to top.** | Persona B | Replace the four-question framing at packet lines 5-14 with a single artifact paragraph leading on what the Halo Atlas IS, not what the packet is for. Demote the four-question framing and the tier-placement block. | **applied 2026-05-14 (batch 3)** |
| W11 | **§1 row 4 CZA cell contains internal release-note prose.** | Persona B | Strip "the Phase 10 attack campaign's Pass A1b replaced the approximation with the literature formula" from the math row; move to §6 Maintenance. | **applied 2026-05-14 (batch 2)** |
| W12 | **§4 Wikipedia content doesn't belong in tier-2 (editor-facing) packet.** | Persona B | Option (a): document tier-2 cut at top of §4 ("editors at tier 2 may skip"); option (b): create tier-specific packet variants. Defer pending tier-deployment decision. | deferred (per user-approved workflow; revisit after tier-2 deployment decision) |
| W13 | **Per-article §4 verdict: Sun dog landing-state concern.** | Persona C | Add a deep-link anchor (e.g. `#parhelion-offset-demo`) to `sundog.cc/sundog.html` so the §4 Sun dog link blurb's URL points to the focused demonstration. Without this the Sun dog proposal is WP:ELNO #19 vulnerable even after the attribution fix. | **applied 2026-05-14 (batch 4)** |
| W14 | **Per-article §4 verdict: CZA supplement may be too thin.** | Persona C | Consider dropping CZA from §4 surviving proposals; the single-threshold supplement is pedagogically thinner than the parhelion slider. Defer pending workbench attribution fix + Sun dog proposal outcome. | deferred (per user-approved workflow; revisit after W2 + W13 land) |
| W15 | **"Stellar Aqua LLC" corporate authorship — WP:SELFPUB consideration.** | Persona C | Add named human contributor(s) to `sundog.cc/sundog.html` footer and to the §4 citation template. | **applied 2026-05-14 (batch 4)** (humiliati placeholder pending real-name attribution) |
| W16 | **`phase3-tests.html` Quick-link is an OR-trap reachable from packet.** | Persona C | Either remove `phase3-tests.html` from the packet "Quick links" entirely, or wrap it in an internal-use caveat. | **applied 2026-05-14 (batch 2)** |
| W17 | **§2 items 1-3 are WP:NOR/OR-risky if quoted on talk page.** | Persona C | Discipline reminder (no patch). The packet's §4 Editorial caveats constraint (iii) is the right policy; extend the constraint to §2 items 2 and 3 explicitly. | **applied 2026-05-14 (batch 2)** |

**Batch application order (user-approved 2026-05-14):**

- **Batch 1 — easy + send-blocking (applied 2026-05-14):** W3 cover-note trailing-sentence cut + brief §14 item 5 stale-status reconciliation + cza_formula.py docstring chromatic-spread correction.
- **Batch 2 — wording polish (applied 2026-05-14):** W5, W6, W7 (packet hedge half), W8, W9, W11, W16, W17. All line-edit scope across packet + handoff.
- **Batch 3 — structural (applied 2026-05-14):** W1 (add p7 specialist-tier question to handoff), W4 (tangent-handle hedge across verdict tables), W10 (packet §0 rewrite).
- **Batch 4 — UI / external (applied 2026-05-14):** W2 (workbench credit strip), W13 (workbench deep-link anchor `#parhelion-offset-demo`), W15 (humiliati named contributor in schema.org JSON-LD + packet citation template). All touch `sundog.cc/sundog.html`; `dist/sundog.html` requires build re-run before sundog.cc reflects changes.
- **Deferred:** W12 (§4 tier-2 cut — revisit after tier-2 deployment decision), W14 (drop CZA from §4 — revisit after W2 + W13 land).

Each batch requires explicit user approval before application. Status
column above is the canonical tracker; updates land here as patches
are applied.

### 8.7 Cross-Persona Convergences (wave-2)

1. **Workbench above-the-fold attribution gap (B-N5 + C-N1, both
   WebFetch-verified):** independent confirmation from editor-tier
   and WP-tier personas. Two WebFetch retrievals, same conclusion.
   Highest-confidence finding of wave-2; promotes wave-1's deferred
   audit to a verified send-blocker.

2. **§2 items 1-3 are load-bearing internally but OR-risky on
   talk pages** (Persona A flags item 2 "non-obvious use" as
   overclaim; Persona C flags items 1-3 as WP:NOR/OR-risky in
   talk-page quotation): specialist and WP reviewers converge from
   different angles on the same wording instances. Internal claim
   license is legitimate; external defense of the link must NOT
   route through items 1-3.

3. **Internal-vocabulary leakage to editorial-tier surfaces**
   (Persona B explicit on jargon load; Persona A implicit via
   "audit-survived sentence is long but honest"): wave-1 added a
   §0 glossary as the patch. Wave-2 says the glossary itself is a
   tell — the cleaner fix is to rewrite the editorial-tier surfaces
   to not need the glossary.

### 8.8 Updated Deployment-Gate Status (wave-2 reading)

Wave-1 §7 marked the synthetic-persona pass CLEARED 2026-05-14 and
concluded "the four-doc + dispatch-memo Phase 11 surface is internally
coherent." Wave-2 is more conservative:

- **Tier-1 (specialist outreach via handoff):** mostly ready. Wave-2
  W1 (p7 parhelion question) and W4 (upper-tangent canonical-handle
  question) are exactly the type of question the tier-1 dispatch
  is meant to elicit, so they can ship as added questions in handoff
  §2 / §7 rather than blocking the send. W5/W6/W8/W9 are wording
  polish that the specialist can flag if they bounce; not strict
  blockers.

- **Tier-2 (editorial outreach via brief + packet):** borderline.
  W3 (§10b trailing-sentence cut) is a 30-second fix and should
  land before any tier-2 send. W10 (packet §0 opens with meta) and
  W11 (§1 row 4 release-note prose) are higher-effort restructuring;
  could ship with caveats or wait for a tier-2-specific packet
  variant per W12.

- **Tier-3 (Wikipedia-adjacent via packet §4):** **NOT ready.**
  W2 (workbench above-the-fold attribution gap) is a packet-stated
  precondition (§4 Editorial caveats constraint (i)) that is not
  met. Wave-1 deferred this to a follow-up audit; wave-2 performed
  the audit and confirms the gate is open. Tier-3 send-block
  remains until the workbench credit strip lands. Beyond W2, even
  with the credit strip the surviving §4 proposals face landing-state
  (W13) and pedagogical-thinness (W14) concerns at the per-article
  talk-page level.

**Wave-2 conclusions vs wave-1.** Wave-2 is a more conservative
reading than wave-1. Wave-1 concluded "now ready: assemble
tier-specific outreach packet + small named outreach list per tier";
wave-2 concludes tier-1 is ready (with added specialist questions),
tier-2 is borderline pending §10b + structural fixes, and tier-3
is blocked on the workbench attribution gate that wave-1 explicitly
deferred. The independent re-dispatch protocol earned its keep:
the workbench attribution audit was wave-1's largest deferral and
wave-2 performed it.
