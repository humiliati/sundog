# RUNNING.md — Dispatch report 2026-07-03

**Nothing launched.** Dispatcher's briefing was based on stale state. Owner
should read this before triaging.

## Bottom line

- The Phase 15 saga the dispatcher briefed me on is **complete and closed**
  (Phase 15 → 15B → 15C all landed with verdicts, latest 15C = "Multi-step
  steering REJECTED", 2026-05-29).
- The version of the TODO I quoted in chat earlier this session was
  `docs/TODO.md` last assembled 2026-05-18. It has since been rewritten
  (2026-05-29) and even that revision is now ~5 weeks stale. Reality has moved
  well past it in ways the dispatcher and I both missed.
- None of the dispatcher's four preferred pops actually apply:
  - Shard 0 canary → obsolete (whole lock done May 24-27)
  - Amended smoke → obsolete (ran May 16, T_window locked, in
    `PHASE15_RESULTS.md`)
  - Phase 13/14 regression gates → those were gates *for* Phase 15; obsolete
  - Longview list → `docs/Longview` does not exist. The only "Longview" files
    are `internal/outreach/LONGVIEW_*` — a **funder concept note** (Longview
    Philanthropy grant, deadline ~2026-07-10), not a compute queue. The
    dispatcher was probably conflating "Longview" the outreach doc with
    "long view" i.e. the operator TODO.

## Why nothing was launched

1. **Briefing was stale.** The dispatcher's preferences are all for Phase 15
   work that finished five weeks ago.
2. **The remaining compute-blocked items in the current TODO don't have
   runnable specs at the "just kick it off" level.** They are design-blocked at
   the runner-command layer:
   - **Coarse-Graining Proof Trunk Phase 4** (`compute-blocked`): next action
     per TODO is *"Add the exact runnable PowerShell commands to
     PHASE4_THREEBODY.md, including module or npm form, output directory,
     resume behavior, and read-back manifest path."* That's a design task, not
     a launch. Note: `scripts/threebody-phase4-iad-*.mjs`
     (shard/merge/concurrent/regret/sat-slate/bayes-floor) *exist* — someone
     wrote them since the TODO was last assembled — but the TODO doesn't
     record which npm command is the canonical entry point or a validated rate
     probe, so blindly launching one of them without owner sign-off is a
     violation of the TODO's own discipline.
   - **Mesa Phase 7 v3** (`compute-blocked`): next action is *"Draft Phase 7
     v3 with old_basin_pref…"* → spec drafting, not compute.
   - **Three-Body Phase 15C** (listed `compute-blocked` in the TODO): actually
     already complete per `PHASE15C_RESULTS.md`. TODO is stale on this.
3. **Environment concern.** This session runs in a sandboxed Linux VM that is
   likely to terminate with the session. `nohup` in that sandbox is not a
   guarantee against session teardown. A multi-hour job the operator wanted
   backgrounded from a real machine should not be started from here.
4. **TODO discipline is explicit.** Every compute-blocked entry repeats: *"Do
   not use an interactive coding-agent session for a multi-hour/multi-day
   lock."* I am an interactive coding-agent session.

## What owner needs to decide first (60-second version)

1. **Refresh the TODO.** It's been ~5 weeks. Confirmed state changes since:
   - Phase 15C landed → verdict "Multi-step steering REJECTED" (2026-05-29,
     recorded in `docs/threebody/PHASE15C_RESULTS.md`).
   - `scripts/threebody-phase16-*`, `threebody-phase17-shard.mjs`,
     `threebody-phase18-calibrate.mjs` now exist — threebody has moved on past
     Phase 15.
   - ARC Branch E3 learned ranker CLOSED 2026-07-01 (Amendment D). Already in
     the current TODO's "Current Holds At A Glance."
   - HS9 introspection-onset study PARKED (per Longview delta, 2026-07-02).
   - Chat-v2 body-resistance R1 gate ran 2026-06-29. Not in the TODO.
   - Many docs modified after the TODO — chatv2/*, boxsel/ORDER_*, algo-approx/*.
2. **Longview funding deadline is unresolved.** Third-party sources conflict
   (Cambridge Neuroscience says 2026-07-10; Future Impact Group says
   2026-07-24). Owner needs to open
   `longview.org/request-for-proposals-research-and-applied-work-on-digital-minds/`
   in a real browser and read the applied-work deadline directly. This is
   already tracked in `internal/outreach/LONGVIEW_NOTE_DELTA_2026-07-02.md` §A
   as OPEN. If the applied deadline is actually Jul 24, the runway doubles.
3. **What to actually run.** If owner wants a compute-blocked lane launched
   next, the two candidates are Coarse-Graining Phase 4 IAD (scripts now
   exist; needs a rate probe and canonical command chosen) and Mesa Phase 7 v3
   (needs spec draft first). Neither is a "flip switch" job.

## Optional low-risk ~8h things that don't need a compute pop

If owner just wants forward motion:

- **V0.3h K_facet Tooling Polish** (`deferred`, ~10-hour budget of *coding*,
  not compute; explicit acceptance criteria in TODO §"Onboarding / Polish").
  Runs against existing receipts in seconds. No compute lock, no
  operator gating, and the exit deliverable is well-defined. Would need owner
  sign-off since it's marked deferred rather than active.
- **Ask Sundog claim-map freshness** (`public-surface`): inventory new claim
  phrases → update `chat/claim_map.json` → `npm run chat:index` → chat evals
  (`chat:eval:static`, `chat:eval:phase3`, `chat:eval:phase3:adversarial`,
  `chat:eval:phase3:differential`, `chat:eval:phase4`). Well under 8 h. Also
  needs owner sign-off — it's launch-sensitive and could contaminate an eval
  if run casually.

Both of these should be *offered to owner*, not started unilaterally. Neither
matches "pop a shell now" cleanly.

## Files consulted

- `docs/TODO.md` (2026-05-29 assembly)
- `docs/threebody/PHASE15_SPEC.md`, `PHASE15_RESULTS.md`, `PHASE15C_RESULTS.md`
- `internal/outreach/LONGVIEW_NOTE_DELTA_2026-07-02.md`
- `package.json` (npm scripts)
- `scripts/threebody-*.mjs` inventory
- `results/threebody/` directory listing

## What owner will not find here

- No PID, no log path, no ETA. No job was started.
- No mutations to owner code or configs.
- No writes outside `RUNNING.md`.
