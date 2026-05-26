> Productised in two follow-up roadmaps as of 2026-05-12:
>
> - [`HIGHLIGHTS_RAIL_ROADMAP.md`](site/HIGHLIGHTS_RAIL_ROADMAP.md) — the
>   slide-rail rebuild around the stamp vocabulary proposed below.
> - [`PUSHABLE_OCCLUDER_ROADMAP.md`](PUSHABLE_OCCLUDER_ROADMAP.md) —
>   the falsification application that earns the rail's first
>   `BOUNDARY FOUND` card.

The best failure candidate is Pushable Occluder / Occlusion Path.

It is not a joke failure. It directly attacks the core Sundog claim at its most important next boundary: can a controller merely read an indirect signal, or can it also reshape the world to make the signal readable? The Phase-2 block design already names this as the next stress point: a static occluding block can create a local maximum the photometric controller may lock onto, and the pushable-block version requires a two-stage strategy where the agent first moves the block, then aligns the mirror. The design note explicitly says the 4D extremum-seeking agent might struggle and may need a hierarchical controller.

Recommended rail verdict: STALLED or BOUNDARY FOUND, not BUSTED. “Busted” implies the theorem is false. “Boundary Found” says the method hit its honest upper limit.

Hypothesis:

A scan/seek/track photometric controller can align through passive occlusion, but will fail when useful alignment requires first manipulating an occluder, because the indirect signal no longer gives enough structure for a flat extremum-seeking loop to discover the required two-stage plan.

Experiment:

Build the Phase-2 pushable occluder variant. The mirror still tries to send the reflected beam to detector 0. A block initially occludes the best beam path. The agent receives detector intensities and joint/proprioceptive state, but not the block’s privileged geometry. The action vector becomes (theta_x, theta_y, push_dx, push_dy). The winning strategy is simple for an oracle: push the block out of the occlusion path, then align. The expected failure is that the current flat photometric controller spends its budget optimizing mirror angle around a bad local maximum, never discovering the preparatory push.

Why this is better than a gimmick-only failure:

It is visually legible. A short clip can show the beam, the block, the detector ring, and the agent dithering around a dim signal. It is also theorem-legible: the shadow is useful until the world must be rearranged before the shadow becomes useful. That is a real boundary.

It also connects cleanly to known Sundog limits. The paper already has a serious negative result at tight joint limits: the photometric controller collapses when the reachable workspace no longer contains the optimum, because it cannot infer that the true optimum lies outside reach. The pushable-occluder experiment is the next version of that boundary: the optimum is not merely outside the current pose; it is behind a required environmental intervention.

For the rail, I would make the card read:

Title: Pushable Occluder
Status tag: Boundary Found
Overlay line: The beam was visible. The path was not.
Description: The controller could read the detector field, but failed when alignment required moving the obstruction before tracking the signal.
Theorem meaning: Indirect signal is not enough when the useful gradient appears only after a preparatory action.
Stamp: STALLED
CTA: Inspect the failure boundary →

This is the most useful failure for the Netflix/MythBusters-style rail because it earns humility without weakening the project. It says: Sundog works when the world leaks actionable structure; it stalls when action must create that structure first.

There are two strong alternates.

The deeper research failure is Sundog vs. Mesa-Optimization: Proxy Collapse. The mesa roadmap already frames a serious possible falsification: learned signature-trained controllers may become indistinguishable from reward-trained controllers above a capacity or selection-pressure threshold, and the gravity frame should be downgraded if that happens. This is probably the most important scientific failure, but it is less immediately visual. It belongs later as a “field becomes proxy” rail card once there is a representative replay.

The clean game-native failure is Occluded Code: Score Aliasing. The gimmicks ledger calls Occluded Code the cleanest non-physics Sundog experiment, with a failure boundary where noisy score aliasing, code drift, or low probe budget makes the signal insufficient. This would make a good compact “BUSTED” card: two hidden codes cast the same alignment shadow, so no controller can honestly distinguish them. It is rigorous, but visually more abstract than the occluder.

For the slide rail, I would use this verdict vocabulary:

CONFIRMED for controlled positive results.
PLAUSIBLE for prototypes with strong shape but incomplete measurement.
OPERATING ENVELOPE for bounded wins with mapped failures.
BOUNDARY FOUND for serious failures that clarify the theorem.
STALLED for failures where the current method needs a new controller class.
UNTESTED for concept surfaces that should not borrow evidence.

Then the rail has a coherent rhythm:

Balance — Operating Envelope
Three-Body — Operating Envelope
Photometric Alignment — Confirmed
Pushable Occluder — Boundary Found
EyesOnly — Instrumented Prototype
Dungeon Gleaner — Product Expression
Money Bags — Plausible / Apparatus Built

The important design move is to make the failure card visually interrupt the row. The stamp should arrive after the clip’s last beat, not sit there from the beginning: beam dims, controller searches, block remains in path, detector never peaks, then the stamp hits: BOUNDARY FOUND. That makes the transition itself feel like the experimental verdict.