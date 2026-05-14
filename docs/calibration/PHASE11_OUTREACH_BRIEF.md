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

The parhelion-offset route remains promotable in bounded form on a strict three-photo subset: photos with unambiguous bilateral peaks, valid geometry, non-trivial discrimination, and an independently fittable 22° halo.

That subset is currently p2, p7, and p13 per the re-audit memo.

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

11. Suggested explicit disclaimers
Include some version of the following in external specialist-facing material:

The project does not claim novel halo physics.
The project’s formulas are drawn from standard atmospheric-optics sources.
The parhelion inverse-use claim is bounded to the re-audited strict subset.
CZA and supralateral are not currently promoted inverse handles.
Tangent remains an unresolved detector question in the current state.
Earlier stronger framings were internally attacked and narrowed before outreach.
12. Outreach-ready artifact list
The following are the right supporting materials to send or link:

docs/SUNDOG_V_GEOMETRY.md
docs/calibration/PHASE10_OPTICAL_REAUDIT_MEMO.md
docs/SUNDOG_OUTREACH_PACKET.md after revision to post-reaudit wording
live explainer / workbench surface
calibration script and selected overlay examples
any revised specialist handoff replacing the stale pre-reaudit version
Do not send stale pre-attack or pre-reaudit framing surfaces without explicit revision.

13. Phase 11 objective
Phase 11 outreach should achieve one thing:

Move the Sundog Geometry project from internally calibrated and re-audited status to externally reviewable status under conservative, technically bounded language.

Success is not public ratification of a large theory. Success is:

specialists do not bounce off the framing,
the artifact reads as honest and inspectable,
the surviving claim surface is clear,
and future outreach can proceed from a stable, reviewable base.
14. Immediate next actions
Rewrite docs/SUNDOG_OUTREACH_PACKET.md to align with the post-reaudit taxonomy.
Rewrite or replace the stale specialist handoff from the post-pass state.
Prepare one short specialist email / DM template and one longer packet.
Check docs/PROMO_HIGHLIGHTS.md and any other reusable public-copy surfaces for stale pre-audit wording before external use.
Conservative framing is part of the deliverable. The outreach brief should make it easy for an external reviewer to see both the artifact’s value and the boundaries the project is intentionally keeping around its claims.