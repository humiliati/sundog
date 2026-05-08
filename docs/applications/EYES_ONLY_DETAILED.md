# EyesOnly / Gone Rogue Detailed Writeup

Status: Ratified scaffold. This file records the research-team response to
`docs/EYES_ONLY_WRITEUP_ROADMAP.md` before the full detailed writeup is
executed.

## Ratified Decisions

**Tier badge.** Keep the card-level tier as `Instrumented Prototype`.

The headless turn-envelope runner is the load-bearing Sundog surface: it exists,
runs against the real Gone Rogue JavaScript engine through Playwright, and is
seedable and policy-pluggable. The matched-seed policy comparison has not run,
so the tier should not be strengthened.

**Three-surface presentation.** Adopt the Three Surfaces table.

This is the right voice move. EyesOnly contains one instrumented Sundog runner,
one sibling UX automation surface, and one forward-looking design. Naming those
tiers separately prevents the runner's credibility from leaking into unbuilt or
non-Sundog surfaces.

**Playtest-agent re-scope.** Re-scope `playtest-agent.js` as sibling UX
automation, not Sundog evidence.

The UI-bound agent is useful product automation, but its scope is regression
hunting for edge cases. It should not be described as a policy-bound Sundog
runner constrained to visible UI affordances.

**LAGM treatment.** Cite Live Agentic Game Moderation as Conceptual Lineage /
Forward-Looking Application Design.

Omitting LAGM would leave the "level manipulation" language ungrounded.
Including it with explicit tier discipline is better: the design has a clear
Sundog shape on paper, but it is not shipping code and not current evidence.

**Detailed doc location.** Use this sibling file rather than expanding
`docs/APPLICATIONS.md`.

`APPLICATIONS.md` should remain a cross-application map. This file can carry the
runner architecture deep dive, matched-seed protocol, playtest-agent scope
clarification, and LAGM intent-spec walk-through.

**Above-vs-below-game framing.** Keep this scoped to EyesOnly until LAGM ships.

The vertical-composition story is promising: under-game runner now, above-game
moderator later. It should not be promoted to a cross-program claim until the
above-game surface exists as code.

## Voice Check

Approved. The roadmap reads disciplined rather than defensive.

The strongest sentence shape is:

> The harness exists; the study does not yet. The playtest agent is sibling UX
> automation. LAGM is the planned above-game Sundog application, currently a
> design document.

That framing is calm, inspectable, and aligned with the Money Bags and Dungeon
Gleaner correction pattern. It avoids both overclaiming and apology. The posture
is not "we have less than we thought"; it is "we know exactly which tier each
surface holds."

## Detailed Writeup Outline

1. **Runner Architecture**
   Explain `PERCEIVE / PLAN / EXECUTE_BATCH`, the Playwright bridge, and the
   `GoneRogue.headless` API surface.

2. **Perception Payload**
   Document the compressed fields: floor, biome, HP ratio, alert level, visible
   enemies, combat state, inventory, gates, and other policy-relevant signals.

3. **Policy Contract**
   Explain the existing greedy policy, planned `random_legal` baseline, and
   target-aware/debug-state upper-bound policy.

4. **Matched-Seed Study Protocol**
   Define seeds, run counts, metrics, JSONL output shape, aggregate tables, and
   acceptance criteria.

5. **Compressed-Perception Ablation**
   Define full versus reduced payload conditions and what failures would weaken
   the Sundog interpretation.

6. **Volatility Sweep**
   Sweep stop-condition thresholds and report survival, steps per floor, and
   batch length.

7. **UI-Bound Playtest Agent Scope**
   Name what `playtest-agent.js` is good for, and explicitly exclude it from the
   Sundog evidence column.

8. **Forward-Looking LAGM**
   Walk through telemetry to competence indices to next-floor intent, with the
   "not built yet" framing visible in the section header.

9. **Claim Boundary**
   Safe claims, avoid claims, and what evidence is needed to promote the runner
   from apparatus to controlled product result.
