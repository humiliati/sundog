Phase 11 Outreach Brief
Sundog Geometry / Halo Atlas
Drafted: 2026-05-14
Purpose: specialist-facing and editor-facing outreach brief after the Phase 10 attack campaign and re-audit.

1. What this brief is for
This brief stages external outreach for the Sundog Geometry / Halo Atlas work. Its purpose is to let a technically literate reviewer answer four questions quickly:

What the artifact is.
What the artifact claims.
What survived internal audit and re-audit.
What kind of feedback we are actually asking for.
This is not a new research memo, and it is not a claim of novel atmospheric-optics theory. The goal is conservative external positioning after the post-audit rewrite clearance in docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md.

2. Executive summary
The Sundog project has built an interactive geometric atlas for sundog / parhelion phenomena. The atlas renders named halo-display features from a compact set of geometric primitives and standard atmospheric-optics relations, then exposes those relations in a public interactive workbench.

The project's contribution is integration, calibration workflow, and interactive presentation. The core equations and primitive relationships are not original to this project; they come from the standard atmospheric-optics literature.

The Phase 10 attack campaign and the 2026-05-14 re-audit narrowed the public claim surface. After that work, the strongest outreach position is:

the atlas is a literature-grounded geometric explainer and interactive workbench;
the parhelion-offset route remains the only promoted inverse handle, and only on a strict three-photo subset;
CZA, supralateral, and tangent routes are not promoted inverse handles in the current state;
outreach should ask for review of the atlas's fidelity, calibration framing, and claim discipline, not endorsement of a broad new theorem.
3. Current project position
What the atlas is
The Halo Atlas is a browser-based geometric workbench that renders a sundog display as a set of named arc families anchored to environmental state and represented as geometric primitives. In the current public-facing framing, it serves three roles:

Explainer — a visual and manipulable account of standard sundog geometry.
Atlas — a single surface that names and displays multiple halo families in one coherent vocabulary.
Calibration instrument — a bounded overlay workflow that compares rendered geometry to photographs.
What the atlas is not
The project does not currently claim:

novel atmospheric-optics equations;
a new physical explanation of sundogs;
a general-purpose inverse-recovery system for rich halo displays;
a validated tangent-arc inverse handle;
a validated CZA inverse handle beyond the current single in-window measured case;
ray-traced optical simulation or full photometric realism.
4. Claim license
Standard atmospheric-optics content
The following belong to the literature, not to this project:

the 22° halo angular relation;
the 46° halo angular relation;
parhelion offset as a function of sun altitude;
CZA tangency and visibility cutoff behavior;
tangent-arc families and their relation to the 22° / 46° halo systems;
Parry-family and related named arc vocabulary.
These are implemented and referenced in the atlas; they are not introduced by it.

Project-specific contribution
The project's current contribution is the artifact layer:

Integrated atlas presentation
A single interactive workbench that presents multiple named halo features in one manipulable geometric surface.

Calibration workflow
A reproducible overlay pipeline for comparing atlas predictions against photographs.

Interactive inverse-use demonstration
A bounded use of standard geometry in reverse, especially the parhelion-offset route on the strict subset that survived re-audit.

Engineering surface
Pose schema, control model, snapshotting, and browser-facing manipulation infrastructure.

Public educational packaging
A literature-grounded explainer page and supporting docs aimed at specialist review, editorial review, and general comprehension.

5. Post-audit / post-reaudit status
Governing memo
The load-bearing status document for outreach is:

docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md
The re-audit result is a clearance to rewrite, not a clearance to send older handoff or outreach materials unchanged.

Audit-survived route taxonomy
As of 2026-05-14, the post-reaudit route taxonomy is:

route	current status	limiting reason
Parhelion offset -> sun altitude	promoted, post-hedged	survives only on strict three-photo eligibility
CZA apex -> sun altitude	not promoted	fails coverage; only one in-window measured case
Supralateral -> sun altitude	not promoted	fails structural discrimination; h-sensitivity below visual-edge noise
Tangent-arc curvature -> sun altitude	unresolved open question	current detector is inadequate; C2 unrun
Strongest surviving bounded claim
The strongest surviving inverse-use claim is:

The parhelion-offset route remains promotable in bounded form on a strict three-photo subset: photos with unambiguous bilateral peaks, valid geometry, an independently fittable 22° halo, and non-trivial discrimination (where the geometric lever `sec(h) − 1` is materially above the typical anchor noise floor).

That subset is currently p2, p7, and p13 per the re-audit memo, with one nuance worth flagging in specialist outreach: p2 (h = 18.6°, lever 5.52%) and p7 (h = 59.4°, lever 96.5%) carry the route on discrimination grounds; p13 (h = 6.83°, lever 0.71%) sits below the 2% lever threshold and contributes on the "unambiguous bilateral peaks + ring-fit halo" eligibility criterion per Pass B2's audit-survived wording, not on independent residual-gate discrimination. The Pass B1 per-photo eligibility sub-table in `docs/calibration/RICH_DISPLAY_OVERLAY_NOTES.md` is the canonical source for the per-photo breakdown; specialist readers will see this and ought not be surprised by p13's caveat.

Explicitly retired phrases
The following pre-audit framings should remain retired in outreach unless quoted historically or retractively:

“passes residual gate on every eligible photo”
“three independent failure layers”
broad tangent-route failure wording
any wording that implies CZA-route residual failure is still the live issue
any wording that implies broad inverse recovery across the full rich-display slate
6. What survived review
Survived strongly
These are the safest outreach points:

the atlas is a literature-grounded geometric explainer;
the workbench integrates multiple named arc families into one coherent interactive artifact;
the project is careful about separating standard literature from project-specific implementation;
the parhelion route survives in narrowed, hedged form;
internal audit discipline found and corrected overclaiming and a load-bearing formula bug;
the current framing is intentionally conservative.
Survived with limits
These can be discussed, but only with explicit bounds:

inverse use of the parhelion-offset relation;
calibration evidence from the photo set;
route comparison across different halo families;
public-facing linkage to broader field/signature ideas.
Did not survive as broad claims
These should not be used as outreach headlines:

“single-handle verdict” as a broad decisive slogan;
a strong CZA inverse route claim;
a strong tangent-route negative claim;
an overextended “field-not-reward” empirical claim based on geometry alone.

Held out of Phase 11 outreach (conservative-first, not retired)
A broader cross-substrate framing — comparing the atlas work to a separate in-vitro research result the team has on the agent-research side — is filed in the team's internal research surfaces (`docs/MESA_CROSSOVER_NOTE.md` and `docs/SUNDOG_V_GRAVITY.md`, both revised 2026-05-14). **Phase 11 outreach intentionally holds it back.** The Phase 11 ask is narrow: a literature-fidelity / claim-discipline check on the geometry workbench. The cross-substrate framing is a separate later conversation, available on request once tier-1 and tier-2 reviews clear; the Phase 11 packet does not lead with it.
7. What feedback we are asking for
This outreach is not asking reviewers to endorse a grand theory. It is asking for bounded technical and editorial feedback on a specific artifact.

Primary specialist questions
Literature fidelity
Does the atlas represent the standard named halo families conservatively and recognizably?

Primitive taxonomy
Are the current classifications and route distinctions reasonable after the post-audit corrections?

Calibration framing
Is the surviving parhelion-route claim stated narrowly enough?

Open-question handling
Is tangent properly framed as an unresolved detector question rather than a settled negative?

Outreach discipline
Does the packet distinguish clearly enough between textbook optics and project-specific implementation?

Secondary editorial questions
Is the artifact useful as a runnable explanatory supplement?
Is the claim license clear enough for Wikipedia-style or educational linking?
Does the packet avoid overstating originality?
8. Suggested audience order
First pass
Atmospheric-optics specialists / halo experts

Goal:

taxonomy sanity check;
claim discipline check;
feedback on whether the artifact is useful and honestly framed.
Second pass
Technically literate editors / science communicators

Goal:

assess whether the atlas is a useful educational demonstration;
refine wording for external, non-specialist audiences.
Third pass
Wikipedia-adjacent outreach or external-link proposals

Goal:

only after specialist-facing wording is stable;
position the atlas as an explanatory adjunct, not as a source of primary physics.
9. Recommended outreach posture
Lead with
interactive geometric halo atlas
standard literature, carefully integrated
reproducible and inspectable implementation
conservative post-audit framing
Do not lead with
“our theorem explains sundogs”
“new atmospheric-optics result”
“empirical proof” language
broad inverse-handle rhetoric
broad field-theory crossover rhetoric
Preferred one-sentence positioning
We built a literature-grounded interactive geometric atlas for sundog phenomena; the contribution is not new optics, but an integrated, calibrated, and publicly inspectable workbench for rendering and testing standard halo geometry.

10. Suggested short specialist cover note
We’ve built an interactive geometric halo atlas for sundog / parhelion phenomena and have just finished an internal attack pass plus re-audit on its calibration and claim surface. We are not claiming new atmospheric optics; the goal is an integrated, literature-grounded atlas and a bounded calibration workflow. We’d value feedback on whether the current primitive taxonomy, route framing, and surviving parhelion claim are conservative and technically fair.

10b. Suggested short editorial / science-communications cover note
Tier-2 (technically-literate editor) cover note, drafted 2026-05-14 by the Phase 11 synthetic-persona dispatch (Persona B output B4). For editors at outlets like Quanta, Ars Technica science desk, Nature commentary, or Atlas Obscura science contributors with a physics undergrad:

> We've built an interactive geometric atlas for sundogs and related halo arcs — a single web page where you drag the sun across the sky and watch every named halo feature (22° halo, parhelia, circumzenithal arc, tangent arcs) update from the same underlying geometry. The math is standard atmospheric optics dating to Greenler 1980; our contribution is the integrated, manipulable presentation and a calibration workflow that compares the rendered geometry against real sundog photographs. We're not claiming new physics. We think it's a useful runnable adjunct for readers who've encountered halo terminology and want to *see* how the pieces fit together, and we'd value your read on whether the page earns a link or a short write-up. A specialist audit pass already narrowed our claims; this is the editorial-tier follow-up.

Use the §10 specialist cover for atmospheric-optics specialists (Tier 1); use this §10b cover for science-communications editors (Tier 2). The Wikipedia-adjacent reviewer tier (Tier 3) is served by the §4 of the packet directly, not by a separate cover note — per the dispatch memo §3a finding #4, that tier needs the workbench attribution audit landed first.

11. Suggested explicit disclaimers
Include some version of the following in external specialist-facing material:

The project does not claim novel halo physics.
The project’s formulas are drawn from standard atmospheric-optics sources.
The parhelion inverse-use claim is bounded to the re-audited strict subset.
CZA and supralateral are not currently promoted inverse handles.
Tangent remains an unresolved detector question in the current state.
Earlier stronger framings were internally attacked and narrowed before outreach.
12. Outreach-ready artifact list
The following are the right supporting materials to send or link (updated 2026-05-14 to reflect what is already revised):

`docs/SUNDOG_V_GEOMETRY.md` — Phase 10 closeout table re-derived from the post-pass / post-reaudit state.
`docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md` — load-bearing governing memo.
`docs/PHASE10_ATTACK_ROADMAP.md` — eight required passes plus the optional C2; provenance for what was repaired and what remains an open question.
`docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` — **rewritten 2026-05-14 and cleared for external handoff**; serves the specialist tier of §8. Replaces the pre-reaudit stop-bannered version.
`docs/MESA_CROSSOVER_NOTE.md` — ratcheted 2026-05-14; the cross-substrate framing the brief intentionally holds back from outreach but that specialists may surface in conversation.
`docs/SUNDOG_V_GRAVITY.md` forward/inverse asymmetry receipt — ratcheted 2026-05-14; the load-bearing public-framing surface for the broader claim, also held back from Phase 11 outreach.
`docs/PROMO_HIGHLIGHTS.md` Atlas-side single-handle receipt paragraph — ratcheted 2026-05-14 to use the post-pass failure-mode taxonomy.
`docs/calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md` — **scaffolded 2026-05-14**; the dispatch memo for §15's deployment methodology. Persona A/B/C scaffolds, retired-phrase guardrails, verified-findings and dropped-claims bins, consolidator verification gate, and open deployment gates. Persona dispatch not yet executed.
`docs/SUNDOG_OUTREACH_PACKET.md` — **rewritten 2026-05-14**; now aligned with the post-reaudit route taxonomy and the three-tier outreach order.
live explainer / workbench surface;
calibration script (`scripts/overlay_calibrate.py`, `scripts/cza_formula.py`) and selected overlay examples.
Do not send stale pre-attack or pre-reaudit framing surfaces without explicit revision. The §14 immediate-next-actions list below tracks what is and is not revised.

13. Phase 11 objective
Phase 11 outreach should achieve one thing:

Move the Sundog Geometry project from internally calibrated and re-audited status to externally reviewable status under conservative, technically bounded language.

Success is not public ratification of a large theory. Success is:

specialists do not bounce off the framing,
the artifact reads as honest and inspectable,
the surviving claim surface is clear,
and future outreach can proceed from a stable, reviewable base.
14. Immediate next actions
Status reflects the 2026-05-14 specialist-handoff-rewrite + public-framing-ratchet wave.

1. ~~Rewrite `docs/SUNDOG_OUTREACH_PACKET.md` to align with the post-reaudit taxonomy.~~ **Status: LANDED 2026-05-14.** Packet now carries the §0 tier-placement block, the post-pass route taxonomy, the strict three-photo parhelion wording, the CZA formula correction, and the Phase 11 provenance chain.
2. ~~Rewrite or replace the stale specialist handoff from the post-pass state.~~ **Status: LANDED 2026-05-14.** `docs/calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md` rewritten end-to-end; previously stop-bannered, now ✅ Cleared for External Handoff. Provenance chain preserved in §1 of the handoff.
3. **Prepare specialist + editorial outreach artifacts:** one short specialist cover note (§10 of this brief), one short editorial-tier cover note (§10b of this brief, added 2026-05-14 from Persona B dispatch output), and one longer packet (built from the artifact list in §12). **Status: COVER NOTES LANDED 2026-05-14.** `SUNDOG_OUTREACH_PACKET.md` revision also landed; longer packet assembly remaining.
4. ~~Check `docs/PROMO_HIGHLIGHTS.md` and any other reusable public-copy surfaces for stale pre-audit wording before external use.~~ **Status: LANDED 2026-05-14.** Atlas-side single-handle-receipt paragraph rewritten to the post-pass failure-mode taxonomy (dataset / physics / tooling); "all four eligible photos" and "three independent failure layers" framings retired. `docs/MESA_CROSSOVER_NOTE.md` and `docs/SUNDOG_V_GRAVITY.md` forward/inverse receipt also ratcheted in the same wave.
5. ~~Choose deployment methodology before any external outreach happens.~~ **Status: SCAFFOLD LANDED 2026-05-14.** Methodology documented in §15 below; dispatch scaffold filed at [`PHASE11_OUTREACH_SYNTHETIC_MEMO.md`](PHASE11_OUTREACH_SYNTHETIC_MEMO.md). The dispatch memo itself is scaffolded but the persona passes have not yet executed. **Status: SCAFFOLDED 2026-05-14** in `docs/calibration/PHASE11_OUTREACH_SYNTHETIC_MEMO.md`; persona execution pending.

Conservative framing is part of the deliverable. The outreach brief should make it easy for an external reviewer to see both the artifact's value and the boundaries the project is intentionally keeping around its claims.

15. Recommended deployment methodology
*Dispatch scaffold landed 2026-05-14 at [`PHASE11_OUTREACH_SYNTHETIC_MEMO.md`](PHASE11_OUTREACH_SYNTHETIC_MEMO.md); refer there for the executable persona dispatch, verification gate, and execution order. The text below is the design rationale.*

This is the answer to "same specialist synthesis or a more refined, contextually aware method?" The recommendation: **same synthesis machinery, swapped personas + tiered output.** The Phase 10 attack campaign's three-persona protocol earned its keep three times (Persona 3's p22 over-statement caught by the consolidator; the audit memo's CZA-formula transcription error caught by Pass A1a's verify gate; the re-audit pass clearing against the consolidator's numerical re-runs). The synthesis machinery is well-tested. What needs to change for outreach is the *persona roles*: the Phase 10 personas were internal-audit personas (tangent / CZA + supralateral / parhelion + forward), each testing a specific claim against the literature. Phase 11 outreach is the inverse problem — testing whether the packet bounces off real external reviewers — so the personas should be **external-reviewer impersonators**, each carrying a different tier from §8.

Recommended Phase 11 personas
**Persona A — Atmospheric-optics specialist.** Reviewer-impersonator role: read the specialist-handoff packet (per §12) as if you were a halo specialist on a 30-minute time budget. Surface: (i) any "they tested the wrong primitive" calls; (ii) any "their formula is wrong" calls; (iii) any "this is well-known textbook material they're claiming as a contribution" calls; (iv) any literature-fidelity issues; (v) whether the audit-survived wording reads as honest. Output: a short bounce-test memo of the form §5 of the existing handoff requests (sound / sound-with-caveat / pushback / out-of-area).

**Persona B — Science-communications editor.** Reviewer-impersonator role: read the brief, the explainer, and the workbench as if you were an editor at a science-communications outlet evaluating whether to link the artifact, on a 15-minute time budget. Surface: (i) is the claim license obvious (what's standard literature vs project-specific contribution); (ii) does the packet over- or under-claim originality; (iii) is the educational framing clear; (iv) what wording is too jargon-laden for non-specialist readers; (v) are the disclaimers in §11 sufficient. Output: an editorial revision pass on the public-facing copy.

**Persona C — Wikipedia-adjacent editor / external-link reviewer.** Reviewer-impersonator role: read the brief and the live workbench as if you were considering whether the artifact qualifies as an external-link adjunct in halo-display articles. Wikipedia's WP:EL discipline is strict; this is the most conservative tier. Surface: (i) does the artifact pass the "is this useful and not a primary-source claim" test; (ii) is the literature attribution clear; (iii) does any wording risk being read as original research; (iv) where is the artifact's specific contribution distinct from textbook content; (v) what would have to change for an external-link proposal to be defensible.

Consolidator verification gate
The same verify-gate discipline that earned its keep in the Phase 10 campaign carries through to Phase 11. The consolidator re-checks each persona's output against:
- the live artifact (the workbench renders + the calibration scripts run)
- the load-bearing docs (re-audit memo + ratcheted gravity / crossover / promo-highlights surfaces)
- the audit-survived wording

If a persona surfaces a finding that doesn't survive the verify gate (e.g., a "their formula is wrong" call that actually misreads the post-A1b literature formula), the finding goes to a §3-style "dropped claims" bin rather than the load-bearing recommendations bin. The discipline that caught Persona 3's p22 over-statement and the memo's CZA transcription error stays mandatory.

Output shape
A Phase 11 consolidated memo of the same shape as `docs/calibration/PHASE10_OPTICAL_AUDIT_SYNTHETIC_MEMO.md`, except the §2 "verified findings that should change the outreach packet" bin contains *reviewer-bounce-test* findings (the packet over-claims X, the framing reads as Y, the cover note needs revision Z) rather than claim-audit findings (the formula is wrong, the route is mis-identified). The §3 "unverified / contradicted claims" bin keeps the same disqualification discipline.

After the Phase 11 synthetic pass clears, the same recommended-execution-order discipline from Phase 10 §3.0 applies: hedge / ratchet the outreach packet, *then* deploy to real external reviewers, not the other way around. Real-reviewer outreach is the closing step after the synthetic pass + revision wave.

Deployment milestone
Phase 11 outreach is ready to deploy to real reviewers when:
- the SUNDOG_OUTREACH_PACKET rewrite (§14 item 1) is landed;
- a Phase 11 synthetic-persona pass on the packet has cleared its verify gate;
- each of the three audience tiers (§8) has a tier-specific cover artifact (specialist cover note, editorial framing, Wikipedia-a