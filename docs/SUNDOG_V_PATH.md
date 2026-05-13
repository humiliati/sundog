# Pushable Occluder Roadmap

Working title: **Sundog Occluded Path**

Working hook:

> The shadow is enough for alignment, until the world has to be
> rearranged before the shadow becomes useful.

Sundog Occluded Path is the program's first deliberately-named
falsification slate on the photometric controller's *visible* failure
register. It is paired with `SUNDOG_V_MESA.md`, which attacks the
*deep* register (proxy collapse under capacity and selection pressure).
Path is the visible failure; Mesa is the deep failure. The two
roadmaps are companions, not substitutes.

The public question is small enough to defend:

> Can a flat scan/seek/track photometric controller discover the
> two-stage plan "push the block, then align" from the indirect
> photometric signal alone — or does the indirect signal stop being
> enough at the point where the world has to be rearranged?

Owner cross-references:

- [`debunked.md`](debunked.md) — origin brief; this roadmap is the
  productisation of its "best failure candidate" recommendation.
- [`PHASE2_BLOCKS_DESIGN.md`](PHASE2_BLOCKS_DESIGN.md) — the upstream
  experimental design (Options A / B / C). This roadmap builds **Option
  B (pushable block as auxiliary action)** as the falsification slate;
  Option A is retained as a control, Option C is out of scope.
- [`HIGHLIGHTS_RAIL_ROADMAP.md`](HIGHLIGHTS_RAIL_ROADMAP.md) — the
  consumer of this work. The rail's first `BOUNDARY FOUND` card is
  blocked on Phase 2 of this roadmap.
- [`PAPER_OUTLINE_v0.md`](PAPER_OUTLINE_v0.md) and
  [`SCIENTIFIC_CRITERIA.md`](SCIENTIFIC_CRITERIA.md) — the bar this
  experiment is held to. This is a falsification result; it must be
  reported with the same discipline as a positive result.
- [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) — sibling falsification slate
  on the deep register. The two are intentionally adjacent: Path is the
  visible failure, Mesa is the structural one.
- [`STANDALONE_APP_ROADMAP.md`](STANDALONE_APP_ROADMAP.md) — the public
  app that eventually hosts the Boundary Replay mode (see Post-Verdict
  Roadmap below).
- [`prereg/path-hypothesis.md`](prereg/path-hypothesis.md) — the Phase 0
  pre-registration artifact. Append-only after its frozen body; the
  body locks the H1/H2/H3 statements, the expected failure mode, and
  the non-confound enforcement pointers before any code lands.

## Sundog Expression *(canonical for the APPLICATIONS.md row)*

- **Hidden target:** photometric optimum at detector 0 in a scene where
  an occluder lies on the beam-to-detector line of sight.
- **Indirect signal:** the 8 detector intensities, mirror joint state,
  and push-effector proprioception. The block's geometry is not in the
  observation channel.
- **Transformation:** scan/seek/track over the photometric signal in a
  4D action space (`theta_x`, `theta_y`, `push_dx`, `push_dy`),
  matching the Phase-1 controller architecture with no per-action
  prior on pushing.
- **Actionable output:** terminal mirror alignment when the photometric
  signal exposes the preparatory action; a clean, attributable failure
  mode (low push utilisation, dithering at a scattering-induced local
  maximum) when it does not.

This block is here so the cross-application comparison row in
`APPLICATIONS.md` and any gallery card's
indirect-signal/transformation/output triplet do not drift between the
roadmap and the public writeup. The Phase 4 broadcast surfaces must
stay aligned to this claim shape, whether the verdict is
`BOUNDARY FOUND` or `CONFIRMED`.

## Current State

The roadmap is in **falsification-slate scoping**: no code, no
controlled claim, no rail card. Pre-registration discipline is
landing first; implementation follows.

- The Phase 0 pre-registration artifact has landed at
  [`prereg/path-hypothesis.md`](prereg/path-hypothesis.md). Its body is
  frozen at 2026-05-13 (PT); future edits are amendments below the
  rule.
- The upstream experimental design is committed in
  `PHASE2_BLOCKS_DESIGN.md`. Option B is the build target.
- The Phase-1 photometric mirror-alignment scene exists and remains
  the substrate this roadmap extends. The relevant pre-existing files
  are inventoried in the prereg note's §Honest Non-Confounds —
  Enforcement Pointers table.
- No occluder_block code, oracle controller, or harness exists yet.
  Phase 1 builds them.

Tier note: this roadmap is not a workbench. The deliverable is a
**falsification result** — a measured failure boundary reported with
the same discipline as a controlled positive result. The Evidence Tier
under which it lands (a new `Falsification result` row vs. a footnoted
cell vs. the `n/a — falsification slate` convention already committed
in `HIGHLIGHTS_RAIL_ROADMAP.md`) is an open question carried into
Phase 4. The prereg note records that no commitment was made at Phase
0 land time.

## Why This Experiment

The Sundog program is currently strongest where it should be: a single
controlled positive result (photometric mirror alignment without
target-position access) plus operating-envelope studies that report
their failure regions. The next attack surface is **not** another
positive demo. It is the smallest experiment that can clarify the
theorem by hitting a deliberate, named, theorem-relevant failure.

`debunked.md` argues — and this roadmap accepts — that the right
falsification slate to run *first* is the pushable-occluder variant:

> A scan/seek/track photometric controller can align through passive
> occlusion, but will fail when useful alignment requires first
> manipulating an occluder, because the indirect signal no longer gives
> enough structure for a flat extremum-seeking loop to discover the
> required two-stage plan.

Three reasons this is the right falsification slate to ship first:

1. **Visually legible.** A short clip shows the beam, the block, the
   detector ring, and the agent dithering around a dim signal. The
   public can see the failure without a chart.
2. **Theorem-legible.** The shadow remains useful until the world must
   be *rearranged* before the shadow becomes useful. That is a real
   boundary, not a gimmick failure.
3. **Continuous with known limits.** The Sundog paper already reports a
   serious negative result at tight joint limits: the photometric
   controller collapses when the reachable workspace no longer contains
   the optimum. The pushable-occluder boundary is the next version of
   that: the optimum is not merely outside the current pose; it is
   behind a required environmental intervention.

The deliverable of this roadmap is a falsification artefact: a measured
failure mode reported with the same discipline as the positive
photometric result.

## Ratified Hook Language

Safe hook:

> Sundog Occluded Path asks whether a flat photometric controller can
> discover the two-stage plan "push the block, then align" from the
> indirect photometric signal alone, when the block must be moved
> before the beam reaches the detector.

Short version:

> The shadow is enough, until the world has to be rearranged first.

Avoid:

- "Sundog cannot do planning." (the program does not claim it can; the
  boundary is *the* finding, not the symptom of a broken claim)
- "The photometric controller fails because Sundog is a toy." (the
  controller succeeds on the Phase-1 task; the boundary is structural,
  not capacity-limited)
- "Pushable Occluder proves Sundog needs hierarchical control." (Phase
  3 is optional and discussion-only; the headline is the failure
  boundary, not the hierarchical baseline's behavior)
- "A failed flat controller is evidence the field is wrong." (the
  field is unchanged by this result; only the operating envelope of
  the photometric controller is.)
- "A gif of the failure is evidence." (the rail card's clip is
  illustration; the evidence is the Phase 2 metric tables and verdict
  artifact.)

## Hypothesis

Stated tightly enough to attack (frozen verbatim in the prereg note):

> **H1.** A flat scan/seek/track photometric controller — the same
> architecture that succeeds on the Phase-1 mirror-alignment task and
> the Phase-2 Option-A static-occluder task — fails to discover a
> two-stage plan when alignment requires first pushing an occluder out
> of the beam path, because the indirect photometric signal does not
> expose the preparatory action as a usable gradient.

Companion hypotheses, both required for the result to be honest:

- **H2 (oracle reachability).** An oracle controller with access to
  block geometry can solve the task: push the block out of the beam
  cone, then run the standard photometric alignment loop.
- **H3 (continuity with prior limit).** The failure mode of the flat
  controller on the pushable-occluder task is structurally similar to
  its failure mode at tight joint limits — both involve an optimum that
  lies beyond a non-photometric prerequisite, not beyond the controller's
  expressivity in a vacuum.

If H1 fails (the flat controller solves the task) the verdict is not
`BOUNDARY FOUND`. It is `CONFIRMED` for a stronger claim than the
program currently makes, and the rail card is rewritten accordingly.
This roadmap is structured so either outcome ratchets the program
forward — see the Pre-Registered Verdict Template below.

## Experimental Setup

This builds on `PHASE2_BLOCKS_DESIGN.md` Option B with minimal additions.

**Scene:**

- Existing mirror-alignment scene (Phase-1 MuJoCo setup).
- Add an `occluder_block` with `block_xy` initial position drawn from a
  bounded distribution that guarantees the block is in the beam-to-best-
  detector line of sight at episode start.
- Block geometry is a vertical bounding cylinder of fixed radius and
  height. Block has overdamped first-order dynamics:
  `block_xy += alpha * (push_dx, push_dy)`, clipped to a workspace bound.

**Observations:**

- All 8 detector intensities (Phase-1 standard).
- Mirror joint state (`theta_x`, `theta_y`).
- Push-effector state (proprioceptive). The agent knows where its
  pusher is.
- **Not** `block_xy`. The block's privileged geometry is denied — same
  discipline as denying target position in Phase 1.

**Actions:**

- `(theta_x, theta_y, push_dx, push_dy)`. 4D continuous.
- Mirror velocity limits and push velocity limits are matched so the
  controller cannot brute-force one channel.

**Reward / signal:**

- The photometric agent receives the same photometric loss as in
  Phase 1: maximise intensity at detector 0. There is no
  reward-shaping bonus for moving the block. **This is load-bearing.**
  The whole point is that the indirect signal must reveal the
  preparatory action without a per-action prior.

**Episode budget:**

- Match Phase-1 episode length. If the flat controller cannot solve in
  that budget, that is part of the result.

### Signal/Tier Table

| Signal | Tier | Use |
| --- | --- | --- |
| `block_xy` (true position) | Privileged | oracle baseline only; never visible to flat/hierarchical/random |
| beam cone geometry | Privileged | oracle baseline only |
| 8 detector intensities | Sensor-available | all controllers; the photometric signal |
| `theta_x`, `theta_y` mirror joint state | Proprioceptive | all controllers |
| push-effector state | Proprioceptive | all controllers |
| terminal photometric error | Metric | post-hoc only; not in control loop |
| push utilisation | Metric / diagnostic | post-hoc only; flat-controller failure signature |
| failure-mode classifier label | Metric / diagnostic | post-hoc manual labelling on a sample |

The table is here so the same audit discipline as Phase-1 (privileged
inputs labelled, sensor-available inputs labelled, metrics held out of
the control loop) is enforceable file-by-file. The prereg note's
non-confound table names which source file enforces each row.

### Baselines

Four controllers, all matched on observations and action space:

1. **Flat photometric** — the Phase-1 scan/seek/track controller
   extended to the 4D action space. Probe schedule re-tuned but
   architecture unchanged.
2. **Oracle** — has access to `block_xy` and to the beam-cone geometry.
   Executes a hand-coded two-stage plan: push the block to a known
   clear region, then run the standard photometric alignment loop.
   This is the "yes the task is solvable" witness.
3. **Hierarchical photometric (stretch)** — same architecture as the
   flat photometric controller but with a learned or scripted high-
   level "push then align" mode switch driven by a coarse photometric
   feature (e.g., detector-ring entropy). Included only if Phase 3
   below has time; its presence is for the discussion section, not the
   headline.
4. **Random** — uniform random action. Sanity floor.

### Metrics

- **Terminal photometric error** at end-of-episode (matches Phase 1).
- **Time-to-acquisition** — first frame after which detector 0 stays
  above a threshold for N consecutive frames.
- **Push utilisation** — total push displacement applied per episode.
  Diagnostic: a flat controller that never pushes is the expected
  failure signature.
- **Failure-mode classifier** (post-hoc, manual on a sample): "never
  pushed", "pushed away from clearance", "pushed into clearance but
  did not re-acquire alignment". The classifier is a small lookup, not
  a learned model.

### Honest non-confounds

The experiment is honest only if these are controlled. Each is mapped
to an enforcement site in the prereg note's table; the summary here is:

- **Beam cone geometry is not adversarial.** The block must be
  pushable out of the cone in fewer push steps than the controller has
  budget for. If the geometry forces the task to be unsolvable in
  principle, the failure is uninformative.
- **Push effector is not photometric.** The pusher must not change the
  photometric signal by occluding the detector itself or reflecting
  light. If it does, the failure is a confound (the agent might learn
  to push *because* it perturbs the signal, not *because* it clears
  the beam).
- **Probe schedule is comparable.** The flat controller's probe
  frequency is re-tuned for the 4D action space using the same
  procedure as Phase 1, not pessimised to guarantee failure.
- **Seeds are matched across controllers.** Block initial position,
  mirror initial pose, and any stochastic optics noise are shared
  per-seed across all four controllers.

If any non-confound is violated, the result is unreportable until it
is fixed. The fix lands as a prereg amendment.

## Expected Failure Mode

The advance prediction (recorded in the prereg note so a future-us
cannot retro-fit it):

- Flat photometric controller: **fails**. Push utilisation low or zero.
  Terminal photometric error high. Time-to-acquisition undefined for
  most seeds. The agent spends its budget optimising mirror angle
  around a local maximum produced by the occluded beam reaching the
  detector through scattering or edge effects.
- Oracle: **succeeds**. Terminal photometric error matches Phase 1.
- Hierarchical photometric: **plausibly succeeds** if the high-level
  mode switch fires. Included for discussion only.
- Random: fails noisily.

If the flat photometric controller succeeds on a non-trivial fraction
of seeds, **the prediction is wrong** and the verdict changes. See the
Pre-Registered Verdict Template below.

## Roadmap

### Phase 0 — Confounds checked (target: 1 sitting)

Pure thinking before any code:

1. Re-read `PHASE2_BLOCKS_DESIGN.md` Option B carefully.
2. Enumerate the "honest non-confounds" above against the existing
   Phase-1 MuJoCo scene. For each, name the file or commit where the
   constraint will be enforced.
3. Write the expected-failure-mode paragraph above into the experiment
   notebook *before* implementation. Time-stamp it.

Acceptance: a short pre-registration note (one page) committed to
[`prereg/path-hypothesis.md`](prereg/path-hypothesis.md). The note is
append-only after its frozen body — any later edits land as
amendments below the rule, with a timestamp and a written
justification.

*Status — Phase 0 acceptance landed (2026-05-13).* The prereg note's
frozen body locks H1/H2/H3, the expected failure mode, and the
non-confound enforcement pointers. Phase 1 is unblocked.

### Phase 1 — Build the scene + oracle (target: 2 sittings)

1. Add `occluder_block` to the Phase-1 scene with the dynamics from
   `PHASE2_BLOCKS_DESIGN.md` Option B.
2. Extend the action space to 4D. Update the random and oracle
   controllers.
3. Build the oracle controller: scripted two-stage plan (push to known
   clear region using `block_xy`, then run Phase-1 photometric loop).
4. Run the oracle on N=64 seeds. Verify H2 (oracle reachability) holds.

Acceptance:

- Oracle terminal photometric error matches Phase 1 within tolerance on
  ≥ 90% of seeds.
- A 6–12 second silent clip is captured of one representative oracle
  run for the still poster: beam visible, block visible, push, clear,
  align, detector ring peaks. This is **not** the rail card clip;
  this is the witness that the task is solvable. The clip is filed
  under `public/media/pushable-occluder-oracle.{mp4,webm}` and the
  still under `public/media/pushable-occluder-poster.jpg`.
- The push-effector-is-not-photometric unit test passes: with mirror
  and block frozen, sweeping the pusher across its workspace produces
  detector readings constant to within the sensor noise floor.
- The two open decisions deferred from prereg §Open Decisions
  (push-effector embodiment; block dynamics parameters) land as
  prereg amendments before Phase 2 begins.

### Phase 2 — Run the flat photometric controller (target: 2 sittings)

1. Re-tune the flat controller's probe schedule for the 4D action
   space, matching the Phase-1 tuning procedure step-for-step. Document
   the tuning in the same notebook.
2. Run flat + random + oracle on N≥128 matched seeds.
3. Compute the four metrics. Classify failure modes manually on the
   first 32 failed flat-controller episodes.
4. **Land a verdict** per the Pre-Registered Verdict Template below.

Acceptance: an experiment-result note committed under
`docs/_results/pushable_occluder_2026-05-12.md` with:

- The hypothesis statement (carried from the prereg note).
- The data file references.
- The four metric tables.
- The failure-mode classifier counts.
- A verdict (`BOUNDARY FOUND` / `CONFIRMED` / halt).
- A 6–12 second silent clip of one representative *flat-controller*
  failure for the rail card: beam visible, mirror dithering, block in
  path, detector ring never peaks, controller gives up. Filed under
  `public/media/pushable-occluder-failure.{mp4,webm}` and used for the
  rail card's poster.

### Phase 3 — Hierarchical baseline (target: optional)

Only if Phase 2 lands on `BOUNDARY FOUND` and the team wants to spend
the budget. Build the hierarchical photometric controller from
"Baselines" above and re-run on the same seeds. The result earns one
discussion paragraph; it does not change the rail card.

### Phase 4 — Productisation: rail card + apps gallery entry (target: 1 sitting, blocks on Phase 2)

1. Add the Pushable Occluder card to the rail markup per the data
   contract in `HIGHLIGHTS_RAIL_ROADMAP.md` §"Card 4: Pushable Occluder
   (detail)".
2. Add a `applications-gallery.html#pushable-occluder` anchor section
   with the failure clip, the metric table, and a link back to this
   roadmap and to `PHASE2_BLOCKS_DESIGN.md`.
3. Add a one-paragraph mention in `APPLICATIONS.md` under a new
   "Falsification slate" sub-heading. Resolve the Evidence-Tier naming
   open question here (a new `Falsification result` row vs. a
   footnoted cell vs. `HIGHLIGHTS_RAIL_ROADMAP`'s already-committed
   `n/a — falsification slate`). The land-time decision is recorded as
   a prereg amendment.
4. Cross-link from `PROMO_HIGHLIGHTS.md` §"Product Highlights" with a
   short block that names the failure honestly and links to this
   roadmap.

Acceptance: the rail's first `BOUNDARY FOUND` card ships. Three
non-author readers can describe the failure correctly after one
viewing. The Cross-Application Comparison Row below lands in
`APPLICATIONS.md` in the same pass.

## Pre-Registered Verdict Template

Phase 2 produces matched-seed metric tables and a failure-mode
classifier. The verdict template below pre-commits to *what those
artifacts have to say* before the rail card can ship or the gallery
can broadcast. This is the same Money-Bags-style discipline that
ratifies Stage 1 verdicts before captures, and the same shape Balance
Phase 10 uses for its CONFIRM/REFUTE/AMBIGUOUS gates. Verdict is
data-driven; disposition is locked.

**Cell classes and P1 denominator.**

Trials are classified before the Phase 2 verdict pass:

- *Diagnostic-positive* seeds: oracle succeeds; block is reachable
  within episode budget; non-confounds 1–4 verified for that seed.
- *Failure-regime* seeds: oracle fails (used only for setup audit, see
  P0 below).
- *Borderline* seeds: oracle succeeds but margin is within sensor
  noise; flagged for review.

The P1 denominator is the *diagnostic-positive* seeds only.

**Pre-registered predictions.**

*P0 — apparatus check.* Oracle reaches the Phase-1 terminal
photometric error tolerance on ≥ 90% of N=64 Phase-1 seeds. P0 is the
smoke-gate equivalent — below this threshold the apparatus is
insufficient, not the prediction falsified. Failure here halts;
disposition is "fix the setup and re-run" (push step budget, beam cone
reachability, block dynamics parameters).

*P1 — central effect.* Inside diagnostic-positive seeds, the flat
photometric controller fails to reach the Phase-1 terminal-error
tolerance on a strict majority (≥ 50%) of seeds, with push utilisation
in the bottom quartile (i.e., "never pushed" dominates the
failure-mode classifier). P1 holds iff both the failure-rate condition
and the push-utilisation condition land. P1 holding is the
`BOUNDARY FOUND` precondition.

*P2 — failure-mode shape.* Of the failed flat-controller seeds, the
failure-mode classifier returns "never pushed" or "pushed away from
clearance" on ≥ 80% of the first-32 manually-labelled sample. P2
distinguishes the canonical `BOUNDARY FOUND` reading (flat controller
never finds the preparatory action) from the sharper-gloss reading
("preparatory action discovered; alignment still not reached"). Both
P2 satisfactions ship a `BOUNDARY FOUND` card; only the card copy
changes.

*P3 — random floor.* The random controller's terminal photometric
error distribution is strictly worse than the flat controller's on
matched seeds. P3 failure indicates a broken flat controller (probe
schedule pessimised, action-scale mismatch); the slate is invalidated
and the controller is re-tuned. Disposition is halt + re-tune.

*P4 — continuity with prior limit.* The failure-mode signature of the
flat controller (low push utilisation; dithering at a scattering
local-maximum) is qualitatively comparable to the joint-limit failure
mode reported in the Phase-1 paper. P4 is reported as a discussion
finding, not a gate — it is the H3 attestation. Non-comparability is
noted but does not move the verdict.

**Verdict template.**

After the Phase 2 result note lands, the writeup asserts *one* of three
verdicts:

*BOUNDARY FOUND.* P0 holds. P1 holds. P3 holds. P2's
"never pushed / pushed away from clearance" majority is the headline
gloss; a "pushed but did not re-acquire alignment" majority ships the
sharper gloss. P4 is reported as discussion. The rail card ships; the
applications-gallery card lands; the `APPLICATIONS.md` row enters
under whichever Evidence Tier convention Phase 4 ratifies. The
Cross-Application Comparison Row enters in the same pass.

*CONFIRMED (stronger claim).* P0 holds. P1 *fails* — the flat
photometric controller succeeds on a non-trivial fraction of seeds.
The hypothesis statement above is wrong; the failure boundary is
somewhere else. A new roadmap attacks the surprise. The rail card is
rewritten to report the stronger Sundog claim; the prereg note's H1
prediction is appended-amended to record what the data showed and why
the prediction was wrong. The Sundog program is in a better epistemic
position than expected — the deliverable becomes the controlled-claim
strengthening, not the boundary card.

*AMBIGUOUS / halt.* P0 fails (oracle cannot solve the task on enough
seeds — setup confound), or P3 fails (flat controller and random
controller are indistinguishable — broken controller), or a non-confound
audit (push-effector-is-not-photometric; matched seeds) fails at audit
time. The slate is invalidated. No rail card, no gallery entry, no
APPLICATIONS.md row until a fix lands and Phase 2 re-runs. Disposition
is locked: ambiguous results do not ship under a softening footnote.

**Disposition is locked.** The verdict is filed in the same directory
as the data: `docs/_results/pushable_occluder_<datetime>.md`. The
choice of verdict is from the predictions, not author-discretionary. A
`CONFIRMED` verdict means the broadcast reports a stronger Sundog
claim — it does not become an asterisked `BOUNDARY FOUND`. A halt
means the broadcast says nothing until the apparatus is fixed; the
rail does not get a "we ran the experiment and it was unclear"
placeholder card.

## Post-Verdict Roadmap

The phases below were pre-registered against the assumption that
Phase 2 returns a `BOUNDARY FOUND` verdict and Phase 4 lands the rail
card. They are gated on Phase 2 + Phase 4, but the *shape* of the
post-verdict work is committed here with the same discipline as the
verdict template — not authored after a verdict has changed what
counts as load-bearing.

### Phase 5 — Companion experiment hook (deferred, depends on `SUNDOG_V_MESA.md`)

The deeper falsification slate is `SUNDOG_V_MESA.md` — proxy collapse
under capacity and selection pressure. Pushable Occluder and Mesa are
companions:

- Pushable Occluder attacks the **shape** of the indirect signal: it
  asks whether the signal exposes the preparatory action.
- Mesa attacks the **content** of the signature: it asks whether
  signature-trained agents secretly reconstruct an internal reward
  proxy.

A combined "Falsification" view on the rail or in the apps gallery is
the right place to host both once Mesa has Phase-1 results. The
applications-gallery section anchor (`#pushable-occluder` plus a new
`#mesa-proxy-collapse`) is named at Phase 4 land time so Mesa's
section can attach cleanly without a markup rewrite.

**Gating:** requires Phase 4 land (rail card shipped) and `SUNDOG_V_MESA.md`
Phase 1 land (HC-Signature canonical task verified). On `CONFIRMED`
instead of `BOUNDARY FOUND`, Phase 5 still ships — the
"combined Falsification view" becomes a "combined operating-envelope
strengthening view", with section copy rewritten accordingly.

### Phase 6 — Standalone App Integration (deferred, depends on `STANDALONE_APP_ROADMAP.md`)

`STANDALONE_APP_ROADMAP.md` describes a guided in-browser experience
with Guided Story, Experiment Replay, and stress-test modes. Once
Phase 4 lands, the Pushable Occluder experiment becomes a **Boundary
Replay** mode in that app:

- Replay the matched-seed flat-controller failure alongside the oracle
  success.
- Annotate the moment the controller "gives up" (terminal probe
  schedule exhausted).
- Let the visitor inspect the detector-ring signal at the local
  maximum and verify the maximum is not the true target.

This is the strongest version of the Boundary Replay because the user
can switch between the flat controller and the oracle on the same seed
and *see* the difference is not random.

**Gating:** requires Phase 4 land and `STANDALONE_APP_ROADMAP.md`
guided-mode scaffolding. On `CONFIRMED`, the Boundary Replay becomes a
"two-stage discovery replay" — same UI, different annotation: the
controller is shown *finding* the preparatory action rather than
failing to.

### Phase 7 — Multi-block stretch (deferred, design-only)

A two-block variant where the agent must sequence two preparatory
actions is a natural extension of the boundary. Out of scope until
Phase 4 lands and the program has rail-card budget to broadcast a
second falsification slate. The design surface is named here so it
does not get reinvented later. Open question carried: does the
boundary deepen monotonically with required preparatory actions, or
does it bifurcate (one-block fails cleanly, two-block fails noisily)?

## Scientific Honesty Notes

- This is a falsification result, not a gotcha. The Sundog program does
  not claim that flat photometric control solves every alignment task.
  The claim is narrower: photometric mirror alignment without
  target-position access in the tested MuJoCo setting. Pushable
  Occluder shows where that narrow claim ends.
- The result should be reported in the same register as the Phase-1
  result: matched baselines, seeded trials, metric tables, failure-
  mode classifier. The fact that the verdict is `BOUNDARY FOUND` does
  not lower the bar for evidence.
- Do not retroactively widen the claim that this result attacks. If
  Phase 2 hands us `CONFIRMED` instead of `BOUNDARY FOUND`, the
  hypothesis statement above is what changes — not the bar.
- The notes-block of the prior draft flagged three program-wide
  consistency issues (smoke-gate vocab uniformity, prereg amendment
  mechanism, Evidence-Tier naming). The first two are resolved in the
  prereg note. The third is carried as an open question into Phase 4.

## Claim Boundary

Safe claim after Phase 2 returns `BOUNDARY FOUND`:

> In the tested MuJoCo photometric setup, a flat scan/seek/track
> photometric controller failed to discover the two-stage plan
> ("push the block out of the beam cone, then align") on the
> diagnostic-positive seed slate, while an oracle controller with
> access to block geometry solved the task. The failure mode is
> dominated by `never pushed`: the indirect photometric signal does
> not expose the preparatory action as a usable gradient in this
> setting.

Safe claim after Phase 2 returns `CONFIRMED`:

> In the tested MuJoCo photometric setup, a flat scan/seek/track
> photometric controller discovered the two-stage plan from the
> indirect photometric signal alone, on a non-trivial fraction of
> diagnostic-positive seeds. The boundary predicted in the prereg
> note is rejected; the Sundog program's controlled claim is
> strengthened to include preparatory-action discovery in this
> setting, and a follow-up roadmap is opened to locate the actual
> failure boundary.

What this would strengthen:

- the theorem's framing of the indirect signal as a function of
  environmental state;
- the public intuition that flat extremum-seeking has a structural
  limit where the world must be rearranged before the signal is
  useful;
- the application portfolio's evidence balance between positive
  workbenches (Three-Body, Balance, Mines) and falsification slates
  (Path, Mesa).

What it would not prove:

- general planning limits of any photometric controller (the result
  is in the tested MuJoCo setting);
- structural limits of hierarchical or learned controllers (Phase 3 is
  optional discussion only);
- limits of the broader Sundog signature program (this attacks the
  visible-shape register; Mesa attacks the deep register);
- that a `CONFIRMED` flat-controller success means the controller
  generalises to multi-block scenarios — Phase 7 stretch carries that
  open question.

## Pre-Committed Cross-Application Comparison Row

When Phase 4 lands, the row below enters `APPLICATIONS.md` under the
cross-application comparison. If the Phase 4 Evidence-Tier convention
decision is `Falsification result`, the row enters under that
sub-heading; otherwise it enters under a footnoted cell per
`HIGHLIGHTS_RAIL_ROADMAP.md`'s already-committed
`n/a — falsification slate` precedent.

| Application | Domain | Indirect signal | Transformation | Actionable output |
| --- | --- | --- | --- | --- |
| Sundog Occluded Path | Photometric control under required preparatory action | 8 detector intensities, mirror joint state, push-effector proprioception in the presence of a beam-blocking occluder | Flat scan/seek/track in a 4D action space (mirror + push) with no per-action prior on pushing | Falsification result: a measured failure mode where the indirect photometric signal does not expose the preparatory action as a usable gradient, with `never pushed` dominating the failure-mode classifier |

On `CONFIRMED`, the row's "Actionable output" cell becomes:

> Controlled result: the flat controller discovered the preparatory
> action from the indirect photometric signal alone on a non-trivial
> fraction of diagnostic-positive seeds, strengthening the Phase-1
> photometric-alignment claim to include two-stage plan discovery in
> the tested setting.

Keeping the row here means later writeup passes cannot drift the
language between the roadmap and the broadcast.

## Initial Build Slice

The first implementation pass is deliberately small:

1. Land the prereg note (frozen body + amendment footer). *(landed
   2026-05-13)*
2. Add `occluder_block` to the Phase-1 scene with Option B dynamics.
3. Extend the action space to 4D and update random + oracle.
4. Build the oracle's scripted two-stage plan and run on N=64 seeds.
5. Verify the push-effector-is-not-photometric unit test passes.

That slice is enough to decide whether the apparatus is sound before
the flat controller's probe schedule is re-tuned. Below the
oracle-reachability floor (≥ 90% of N=64 Phase-1 seeds), the setup is
unsafe to broadcast against — fix the apparatus and re-run before
spending budget on Phase 2.

**Exit criterion for the first slice:** P0 holds (oracle reaches
Phase-1 terminal-error tolerance on ≥ 90% of N=64 seeds). Below this
floor — oracle does not solve the matched-Phase-1 setup with block
geometry available — the slate is unreportable until the
non-confound enforcement pointers in the prereg note are re-audited.

## References

- `debunked.md` — origin brief. The "best failure candidate" argument
  and the rail card draft.
- `PHASE2_BLOCKS_DESIGN.md` — Options A / B / C; this roadmap is the
  productisation of Option B.
- `HIGHLIGHTS_RAIL_ROADMAP.md` §"Card 4: Pushable Occluder (detail)" —
  the rail card markup and the choreography for the interrupt.
- `SUNDOG_V_MESA.md` — sibling falsification slate, deferred companion.
- `SUNDOG_V_GIMMICKS.md` Candidate 2 (Occluded Code) — alternate
  cleaner-but-more-abstract falsification path, not chosen for this
  pass.
- `APPLICATIONS.md` Evidence Tiers — the canonical bookkeeping; this
  result will land outside the existing tiers and may motivate a new
  `Falsification result` row.
- `PAPER_OUTLINE_v0.md` / `SCIENTIFIC_CRITERIA.md` — the evidence bar.
- `STANDALONE_APP_ROADMAP.md` — the public app that eventually hosts
  the Boundary Replay mode.
- [`prereg/path-hypothesis.md`](prereg/path-hypothesis.md) — the
  Phase 0 pre-registration artifact. Frozen body + append-only
  amendments.
