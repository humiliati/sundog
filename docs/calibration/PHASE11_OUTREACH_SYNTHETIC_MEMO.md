# Phase 11 Outreach Synthetic Dispatch Memo

Filed: 2026-05-14  
Status: **scaffolded; persona dispatch not yet executed**  
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
| Tangent-arc curvature -> h | unresolved | column-peak detector fails on p2/p13/p27; C2 detector unbuilt |

Retired phrases remain retired:

- "passes residual gate on every eligible photo"
- "three independent failure layers"
- class-level tangent-route failure
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
| tangent unresolved framing | sound / caveat / pushback / out-of-area | _pending_ |
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
