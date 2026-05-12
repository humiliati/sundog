# Pushable Occluder Roadmap

Working title: **Sundog Occluded Path**

Date: 2026-05-12
Status: falsification-slate scoping doc. No code yet. No claim yet.
Verdict target: **BOUNDARY FOUND** on the highlights rail.

Owner cross-references:

- [`debunked.md`](debunked.md) — origin brief; this roadmap is the
  productisation of its "best failure candidate" recommendation.
- [`PHASE2_BLOCKS_DESIGN.md`](PHASE2_BLOCKS_DESIGN.md) — the upstream
  experimental design (Options A / B / C). This roadmap builds **Option
  B (pushable block as auxiliary action)** as the falsification slate;
  Option A is retained as a control, Option C is out of scope.
- [`HIGHLIGHTS_RAIL_ROADMAP.md`](HIGHLIGHTS_RAIL_ROADMAP.md) — the
  consumer of this work. The rail's first `BOUNDARY FOUND` card is
  blocked on Phase 1 of this roadmap.
- [`PAPER_OUTLINE_v0.md`](PAPER_OUTLINE_v0.md) and
  [`SCIENTIFIC_CRITERIA.md`](SCIENTIFIC_CRITERIA.md) — the bar this
  experiment is held to. This is a falsification result; it must be
  reported the same way a positive result would be.
- [`SUNDOG_V_MESA.md`](SUNDOG_V_MESA.md) — sibling falsification slate
  on a deeper register (mesa proxy collapse). Pushable Occluder is the
  *visible* failure; Mesa is the *deep* failure. They are companions,
  not substitutes.
- [`STANDALONE_APP_ROADMAP.md`](STANDALONE_APP_ROADMAP.md) — the public
  app this experiment eventually feeds into, as an interactive
  "Boundary Replay" mode alongside Guided Story and Experiment Replay.

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

## Hypothesis

Stated tightly enough to attack:

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
`BOUNDARY FOUND`. It is `CONFIRMED` for a stronger claim than we
currently make, and the rail card is rewritten accordingly. This roadmap
is structured so either outcome ratchets the program forward.

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
   feature (e.g., detector-ring entropy). Included only if Phase 2
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

The experiment is honest only if these are controlled:

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
is fixed.

## Expected Failure Mode

The advance prediction (recorded here so a future-us cannot
retro-fit it):

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
of seeds, **the prediction is wrong** and the verdict changes. See
"Outcome Branching" below.

## Outcome Branching

The roadmap commits to acting on every outcome, not just the one we
expect:

| Observed outcome | Rail stamp | Action |
| --- | --- | --- |
| Flat fails on majority of seeds; oracle succeeds; failure-mode classifier dominated by "never pushed". | `BOUNDARY FOUND` | Ship the card. Cite this roadmap as `data-stamp-source`. |
| Flat fails but failure mode is "pushed but did not re-acquire alignment" — i.e., the agent stumbles into pushing and still cannot finish. | `BOUNDARY FOUND` with a sharper gloss: "Preparatory action discovered; alignment still not reached." Card content is rewritten; the boundary is real but located differently. |
| Flat succeeds on a meaningful fraction of seeds. | `CONFIRMED` for a *stronger* claim than the program currently makes. New roadmap to attack this surprise; the rail card is rewritten. The hypothesis statement above is wrong; the failure boundary is somewhere else. |
| Oracle fails. | Halt. The setup has a confound (beam cone unreachable, push dynamics too slow, etc.). Fix the setup and re-run. Do not ship any card until the oracle succeeds. |
| Random performs comparably to flat. | The flat controller is broken. Re-tune the probe schedule. Do not ship until random is below flat. |

The verdict is *not* "we predicted the failure correctly". The verdict
is whichever cell of the table the data lands in.

## Phases

### Phase 0 — Confounds checked (target: 1 sitting)

Pure thinking before any code:

1. Re-read `PHASE2_BLOCKS_DESIGN.md` Option B carefully.
2. Enumerate the "honest non-confounds" above against the existing
   Phase-1 MuJoCo scene. For each, name the file or commit where the
   constraint will be enforced.
3. Write the expected-failure-mode paragraph above into the experiment
   notebook *before* implementation. Time-stamp it.

Acceptance: a short pre-registration note (one page) committed to
`docs/_prereg/pushable_occluder_2026-05-12.md`. The note is appended
to, never edited, after this point.

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
- A 6-12 second silent clip is captured of one representative oracle
  run for the still poster: beam visible, block visible, push, clear,
  align, detector ring peaks. This is **not** the rail card clip;
  this is the witness that the task is solvable. The clip is filed
  under `public/media/pushable-occluder-oracle.{mp4,webm}` and the
  still under `public/media/pushable-occluder-poster.jpg`.

### Phase 2 — Run the flat photometric controller (target: 2 sittings)

1. Re-tune the flat controller's probe schedule for the 4D action
   space, matching the Phase-1 tuning procedure step-for-step. Document
   the tuning in the same notebook.
2. Run flat + random + oracle on N≥128 matched seeds.
3. Compute the four metrics. Classify failure modes manually on the
   first 32 failed flat-controller episodes.
4. **Branch on outcome** per the Outcome Branching table.

Acceptance: an experiment-result note committed under
`docs/_results/pushable_occluder_2026-05-12.md` with:

- The hypothesis statement.
- The data file references.
- The four metric tables.
- The failure-mode classifier counts.
- A verdict (`BOUNDARY FOUND` / `CONFIRMED` / halt).
- A 6-12 second silent clip of one representative *flat-controller*
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
3. Add a one-paragraph mention in `APPLICATIONS.md` under a new "Falsification
   slate" sub-heading. The Evidence Tier is **not** "Research result";
   it is a new informal tier "Falsification result" or it sits outside
   the table with a footnote. Decide at land-time.
4. Cross-link from `PROMO_HIGHLIGHTS.md` §"Product Highlights" with a
   short block that names the failure honestly and links to this
   roadmap.

Acceptance: the rail's first `BOUNDARY FOUND` card ships. Three
non-author readers can describe the failure correctly after one
viewing.

### Phase 5 — Companion experiment hook (deferred)

The deeper falsification slate is `SUNDOG_V_MESA.md` — proxy collapse
under capacity and selection pressure. Pushable Occluder and Mesa are
companions:

- Pushable Occluder attacks the **shape** of the indirect signal: it
  asks whether the signal exposes the preparatory action.
- Mesa attacks the **content** of the signature: it asks whether
  signature-trained agents secretly reconstruct an internal reward
  proxy.

A combined "Falsification" view on the rail or in the apps gallery is
the right place to host both once Mesa has Phase-1 results. This is
out of scope for the current roadmap but should be considered when
naming the apps gallery section.

## Standalone App Integration (later)

`STANDALONE_APP_ROADMAP.md` describes a guided in-browser experience
with Guided Story, Experiment Replay, and stress-test modes. Once Phase
4 lands, the Pushable Occluder experiment becomes a **Boundary Replay**
mode in that app:

- Replay the matched-seed flat-controller failure alongside the oracle
  success.
- Annotate the moment the controller "gives up" (terminal probe
  schedule exhausted).
- Let the visitor inspect the detector-ring signal at the local
  maximum and verify the maximum is not the true target.

This is the strongest version of the Boundary Replay because the user
can switch between the flat controller and the oracle on the same seed
and *see* the difference is not random.

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

## Open Questions

1. **Push effector embodiment.** The cheapest realisation is a second
   joint chain with its own mirror-free tip. The cleaner realisation is
   a shared kinematic chain where the pusher and the mirror are end-
   effectors on the same arm. Defer the decision to Phase 1; capture
   it in the pre-registration note.
2. **Block dynamics realism.** First-order overdamped dynamics are
   chosen for clean failure attribution. If the result is sensitive to
   block inertia or friction, a second study with richer dynamics
   becomes a downstream task.
3. **Multi-block stretch.** A two-block variant where the agent must
   sequence two preparatory actions is a natural extension. Out of
   scope for the current roadmap; mentioned here so it doesn't get
   reinvented later.
4. **Naming.** "Pushable Occluder" is the descriptive name. "Sundog
   Occluded Path" is the working public name. The rail card uses
   "Pushable Occluder" (it survives the
   one-glance test better). Reconsider at Phase 4.

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
  "Falsification result" row.
- `PAPER_OUTLINE_v0.md` / `SCIENTIFIC_CRITERIA.md` — the evidence bar.
- `STANDALONE_APP_ROADMAP.md` — the public app that eventually hosts
  the Boundary Replay mode.
