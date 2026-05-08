# EyesOnly / Gone Rogue Detailed Writeup

Status: Pass 4 — full deep-dive content executed against the ratified scaffold
on 2026-05-08. The Ratified Decisions and Voice Check sections below are
preserved from the team response to `docs/EYES_ONLY_WRITEUP_ROADMAP.md`. The
nine numbered sections that follow execute against the scaffold's outline.

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

---

## 1. Runner Architecture

The headless turn-envelope runner is the load-bearing Sundog surface in
EyesOnly. It is a Python module set rooted at `<sundog>/runners/` that drives
the real Gone Rogue JavaScript engine through Playwright against the
`GoneRogue.headless` API exposed by `<EyesOnly>/public/js/gone-rogue.js`.

The runtime topology, restated from `docs/runners.md` for self-containment of
this writeup:

```
sundog TurnEnvelopeRunner   (Python, sundog/runners/)
  └── GoneRogueAdapter       (sundog/runners/adapters/gone_rogue.py)
          └── Playwright (Chromium, headless or visible)
                  └── about:blank page with injected DOM stubs and game scripts
                          └── GoneRogue.headless.{getState,
                                                  getLegalActions,
                                                  applyAction,
                                                  getGrid,
                                                  resetToState}
                                  └── Real Gone Rogue JS engine
```

The adapter's load sequence is fixed and idempotent. On first `reset(seed)`
the adapter opens `about:blank`, injects DOM stubs (`requestAnimationFrame`,
canvas `getContext` fallback, `localStorage` shim), creates a stub canvas at
id `gone-rogue-canvas`, and adds the EyesOnly script tags in dependency
order: `seeded-random.js`, `card-system.js`, `lighting-system.js`,
`ground-effects.js`, `boss-encounters.js`, `secret-floors.js`,
`gone-rogue-canvas.js`, `gone-rogue.js`, then the headless adapter
`agent-headless-adapter.js` from `<EyesOnly>/public/tests/`. After loading,
the adapter verifies that `GoneRogue`, `GoneRogue.headless`, and
`HeadlessAdapter` are all globally defined before allowing any action call.

The verification check is the load-bearing apparatus-trust signal. A run can
fail loudly at load time if any of the JS surfaces have shifted, rather than
silently mis-perceiving partway through a 1000-step run. The gate matters
for matched-seed reproducibility: a partially-loaded engine would produce
plausible-looking-but-wrong telemetry.

The turn-envelope loop is implemented in
`GoneRogueAdapter.run_turn_envelope(policy, max_steps, max_batch,
volatility_threshold)`:

```
PERCEIVE  ← state = self.get_state()
            compressed = self.compress_perception(state)
            perception = policy.perceive(compressed)

PLAN      ← plan = policy.plan(perception)

EXECUTE_BATCH
          ← batch_depth =
                min(2, len(plan.action_batch))
                  if plan.volatility > volatility_threshold
                else min(len(plan.action_batch), max_batch)

            for action in plan.action_batch[:batch_depth]:
                result = self.apply_action(action)
                steps += 1
                if hp_after_action <= 0: outcome = "died"; break
                if any(stop in stop_conditions for stop in result.events): break

            repeat until terminal or max_steps
```

Three properties of this loop matter for the Sundog claim:

The batch-depth selector is the *operational* expression of the alignment
theorem's stop-conditioned envelope. It is a one-line guard that distinguishes
volatile from stable plans. When `volatility > threshold`, batch depth is
hard-clamped to 2 regardless of how long the policy's `action_batch` is. When
volatility is below threshold, the runner takes the full batch up to
`max_batch`. The threshold is exposed as a CLI flag (`--volatility F`,
default `0.7`) so the runner can be run as both Sundog policy executor and
sensitivity-sweep substrate.

Stop conditions are *checked against the engine's response*, not predicted
ahead of time. Each `applyAction` returns a result dict with a list of
`events` that the engine itself emits — `combat_start`, `floor_changed`,
`new_enemy`, `gate_encountered`, etc. The runner intersects those events
with `plan.stop_conditions` and breaks on any match. This means stop
conditions are verifiable claims about runner behaviour rather than
hand-waved policy intent.

`died` is a special-case stop condition checked between every action by
re-reading the player HP from `get_state()` rather than waiting for the
engine to emit a death event. This is intentional. It is the one stop
condition the runner cannot defer to engine semantics, because policy choice
between "die now" and "do one more thing" is the kind of edge case that
distinguishes a Sundog policy from a flat one.

Two CLI runners ship on top of the adapter:

`<sundog>/runners/gone_rogue_headless.py` is the batch CLI for matched-seed
runs and CI. Flags include `--runs N`, `--seed N`, `--policy NAME`,
`--max-batch N`, `--volatility F`, `--max-steps N`, `--out PATH`. Run *i*
uses `seed + i`, which is the seed convention the matched-seed protocol in
§4 builds on.

`<sundog>/runners/gone_rogue_ui.py` runs the same turn-envelope loop in a
visible Chromium window with a `--slow-mo MS` delay between actions
(default 200 ms). It supports `--screenshot-dir DIR` for end-of-run PNGs
and `--keep-open` for inspection. This is the runner the behaviour-clip
work in §4 depends on; the headless variant cannot produce visible
recordings.

The adapter's lifecycle is context-manager-bound. Tests under
`<sundog>/tests/runners/test_gone_rogue.py` exercise both adapter unit
behaviours (without browser, mocked page) and integration behaviours
(against a running EyesOnly dev server, gated on
`EYESONLY_BASE_URL`). The integration tests are tagged `@pytest.mark.integration`
so they only run when the env var is set.

What this section does *not* claim: that the architecture is novel, that
Playwright is the optimal bridge, or that the JS-side `GoneRogue.headless`
is API-stable across EyesOnly releases. The architecture is the apparatus.
Its merits are inspectable; its limits are real.

## 2. Perception Payload

`GoneRogueAdapter.compress_perception(state)` is the PERCEIVE step of the
turn envelope. It maps the raw `GoneRogue.headless.getState()` return —
which is a deeply nested JS object containing the entire run, the floor
grid, all entities, all timers, the deck, the hand, the discard, and the
combat sub-state — into a small flat dict the policies can read without
needing to understand the engine's internals.

The compressed payload is the *indirect signal* in the Sundog formulation.
The full game state is the world; the payload is the projection of the
world onto the policy's decision-relevant axes.

**Fields and policy relevance:**

`floor_index` (int, 0-based) — derived from `state.floor` minus one to match
the sundog convention. The floor is the long-horizon progression axis;
`floors_remaining = max(0, 32 - (floor - 1))` is exposed as a sibling field
so policies can act differently on F1 vs F30 without recomputing.

`biome` (str) — the active floor's biome tag (`forest`, `industrial`,
`aerospace_museum`, etc.). Policies do not currently branch on biome, but
the field is exposed because biome-aware ablations are a likely future
study direction.

`hp` / `max_hp` / `hp_ratio` (int / int / float) — the survival axis. The
greedy policy uses `hp_ratio < 0.25` as the survival threshold and
`hp_ratio < 0.45` as a caution threshold that elevates volatility without
switching axis. Per `gone_rogue_greedy.py`:

```python
SURVIVAL_HP = 0.25   # below this → survival axis
CAUTION_HP  = 0.45   # below this → elevated volatility
```

`keys_inventory` (dict, currently `{"owned": int}`) and `keys_on_floor`
(int) — the resource axis. The greedy policy switches to `resource` axis
when `keys_on_floor > 0 and not in_combat`.

`gates_visible` / `gate_requires_key` / `gate_key_color` — the progression
axis trigger. The runner currently flags `gates_visible` from a combination
of `runState.gatesSpawnedThisRun > 0` and `state.hasExit`. The
key-requiring fields are present but always default values for Gone Rogue
specifically because the game does not gate exits behind keys; the fields
exist to keep the payload shape consistent with the parent `PerceptionPayload`
dataclass in `<sundog>/runners/policy.py`.

`enemies_visible` (int) — counted from `state.enemies` filtered to non-dead
entries. This is the volatility primary input: the greedy policy sets
`volatility = 0.8` when any enemy is visible outside combat, and
`volatility = 0.9` inside STR combat.

`in_combat` (bool, mirrored as `str_combat_active`) — the STR (Single-Turn
Resolve) combat sub-state flag. When true, action selection collapses to
combat-card play and flee, and the policy's stop conditions become
extremely conservative.

`alert_level` (str: `safe` / `caution` / `danger`) — the engine's own
coarse threat classification. The greedy policy does not currently branch
on this directly, but the field is exposed for the ablation study (§5)
because `alert_level` is one of the three perception fields most
plausibly redundant with `enemies_visible`.

`steps_on_floor` (int) — the within-floor horizon counter. Policies that
want to time-bound exploration use this; the greedy policy does not.

`stealth` / `detection` / `energy` / `max_energy` (ints) — Gone Rogue-specific
fields that don't appear in the base `PerceptionPayload` dataclass. The
adapter passes them through anyway because future policies that lean into
the stealth axis will need them.

`advantage` (str: `neutral` / `player` / `enemy`) — the STR combat
advantage marker. Shipped through the payload; not consumed by greedy.

What the payload does *not* include, intentionally:

The full grid (`getGrid()` output) is available through a separate adapter
method but is not pushed into the payload. A 50×50 tile grid would dominate
the perception size and force every policy to do its own pathfinding
ingest. Policies that want grid context can call `adapter.get_grid()`
directly; this is a deliberate separation of "small stable perception" from
"on-demand structural context."

The full deck / hand / discard. The greedy policy plays cards by hand
*index*, not by card identity, because the card-identity surface is
unstable across EyesOnly releases. This is a real limitation of the
greedy baseline that the §3 Policy Contract calls out.

The full enemy list with positions, intents, HP. Policies that want
combat-target reasoning will need a more enriched perception. The §5
ablation explicitly probes whether the current minimal payload is
sufficient for the greedy axis-selection logic to outperform random.

**Stability commitment:**

The payload field names are stable across `GoneRogueAdapter` versions in
the sense that a new policy written against today's payload will continue
to work as long as the field names listed above exist. New fields may be
added; existing fields will not be renamed or repurposed without a major
version bump on the adapter. This is the contract a matched-seed study
needs in order to be re-run six months from now without losing comparability.

## 3. Policy Contract

The runner's policy contract is intentionally narrow. Any object with a
`perceive(compressed_dict)` method that returns a `PerceptionPayload`-shaped
object and a `plan(perception)` method that returns a `TurnPlan` can be
plugged into `run_turn_envelope`. The contract is pinned in
`<sundog>/runners/policy.py`:

```python
@dataclass
class TurnPlan:
    axis_priority: AxisPriority   # "progression" | "resource" | "survival"
    short_strategy: str
    action_batch: List[str]
    stop_conditions: List[str]
    volatility: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)
```

The `axis_priority` field is the explicit framing of the policy's choice.
Logging it at every PLAN step is what makes the matched-seed comparison
analyzable — runs can be sliced by which axis was active at what step
density without re-deriving the framing from action history.

`stop_conditions` are *named*, not anonymous. The runner intersects them
against engine event tags. A new policy that wants to break on a new event
type adds a string to `stop_conditions` and the runner picks it up
automatically; the policy does not have to re-implement the loop control.

`volatility` is a single float on `[0.0, 1.0]` with a soft-pinned semantics:
above `volatility_threshold` (default 0.7), the runner clamps batch depth
to 2; below threshold, full batch up to `max_batch`. The §6 sensitivity
sweep is the empirical handle on whether the soft-pinned semantics are
the right ones.

**Three policies the matched-seed study (§4) requires:**

`greedy` (shipped, `<sundog>/runners/policies/gone_rogue_greedy.py`):
the existing `GoneRogueGreedyPolicy`. Axis selection by HP and
key-on-floor; volatility by combat status; stop conditions including
`new_enemy`, `floor_changed`, `died`, `combat_start`, `gate_encountered`
plus axis-specific extensions. The policy does *not* use `alert_level`,
`stealth`, `detection`, `energy`, `biome`, `floor_index`, or
`floors_remaining` — its axis logic is HP-and-key-and-combat only. This
matters for the §5 ablation: the greedy baseline's tolerance for missing
fields can be measured directly by ablating the fields it ignores
(should be no effect) versus the fields it consumes (should degrade).

`random_legal` (not yet shipped, blocking Priority 1 of §4):
a baseline that calls `adapter.get_legal_actions()` and selects one
uniformly at random. Stop conditions: `died`, `floor_changed`. Axis
priority: hard-coded to `progression`. Volatility: 0.0. This is the
"no Sundog turn-envelope structure at all" baseline — it doesn't batch,
doesn't differentiate by HP, doesn't recognise combat. Comparing
`greedy` to `random_legal` measures the value of the turn-envelope
shape at all.

`target_aware` (not yet shipped, blocking Priority 1 of §4):
a debug-state policy that uses authorial knowledge — exit position from
`adapter.get_grid()`, gate solutions, and floor layout — to upper-bound
performance. It is *not* a Sundog policy; it has access to information
the indirect-signal payload deliberately omits. Comparing `greedy` to
`target_aware` measures how much performance the indirect-signal
discipline costs versus a privileged-information ceiling.

The three-policy structure is the falsification frame. If `greedy`
performs no better than `random_legal` on matched seeds, the turn-envelope
shape is not contributing structure. If `greedy` performs nearly as well
as `target_aware`, the indirect-signal discipline is not costly. Either
extreme is reportable evidence for or against the Sundog hypothesis as
applied to procedural roguelikes.

**Out of scope for the apparatus:**

The contract does not include policy training, gradient-based
optimisation, or RL-style reward shaping. Sundog is an alignment program,
not an RL benchmark, and the runner is built to support comparison
studies between hand-authored policies. A learned policy is an obvious
future extension; it is not part of the current apparatus and will not
be claimed as part of the §4 study.

## 4. Matched-Seed Study Protocol

This section pre-registers the matched-seed multi-policy study against
the Gone Rogue runner. Pre-registering before the study runs is the same
discipline pattern Money Bags applied to its Stage 1 capture work: the
slate, predictions, metrics, verdict template, and disposition rule are
committed before the data exists.

**Hypothesis under test.**

The Sundog turn-envelope architecture (compressed perception, axis
selection, volatility-modulated batched action with named stop conditions)
produces materially better procedural-roguelike play than uniform random
selection over legal actions, at indirect-signal cost compared to a
target-aware debug-state policy.

The hypothesis is two-sided. A match between `greedy` and `random_legal`
falsifies the Sundog claim for this domain. A near-match between `greedy`
and `target_aware` corroborates the indirect-signal-discipline claim. A
result between the two is the expected outcome and is the finding the
study is designed to characterise.

**Slate.**

Three policies × matched seeds. Initial slate:

| Cell | Policy | Seeds | Runs per cell |
| --- | --- | --- | --- |
| (a) | `greedy` | `[42, 43, 44, …, 141]` | 100 |
| (b) | `random_legal` | `[42, 43, 44, …, 141]` | 100 |
| (c) | `target_aware` | `[42, 43, 44, …, 141]` | 100 |

Total: 300 runs across 100 distinct seeds. Same seed across cells means
the engine emits identical floor layouts, enemy spawns, card draws, and
RNG sequences. Cross-cell variance is therefore attributable to policy
choice, not to engine sampling.

**Configuration.**

`--max-steps 1000`, `--max-batch 8`, `--volatility 0.7`, `--policy NAME`,
headless mode. Same `EYESONLY_BASE_URL` across all cells, recorded in the
bundle metadata to defend against silent EyesOnly drift between runs.

**Filed deliverable.**

`<sundog>/results/eyesonly/matched_seed_<datetime>/` containing:

- One JSONL bundle per cell: `greedy.jsonl`, `random_legal.jsonl`,
  `target_aware.jsonl`. Each line is the per-run record specified in
  `runners.md` § Output Format (`run_index`, `seed`, `policy`, `outcome`,
  `floor`, `biome`, `steps`, `player`, `events`).
- One aggregate summary table: `aggregate.md`. Per-policy outcome
  distributions, mean / median / max floor reached, survival rate by
  floor, mean steps, mean steps-per-floor, axis-priority dwell
  fractions.
- One environment record: `env.json`. EyesOnly commit hash at run time,
  Sundog commit hash, Playwright version, Chromium version,
  Python version, runner CLI invocation, and the `volatility` and
  `max_batch` flags as actually used.
- One reproduction script: `reproduce.sh`. The exact CLI invocation
  for each cell with the same `--seed` base and `--runs` count, so a
  reader can re-run any cell without re-deriving flags.

**Metric definitions.**

| Metric | Operational definition |
| --- | --- |
| Floor reached | `final_state.floor` at run end. |
| Survival rate by floor F | Fraction of runs where final floor ≥ F. |
| Outcome distribution | Counts of `{died, completed, game_ended, ongoing}`. |
| Mean steps | Mean of `steps` across runs in the cell. |
| Steps per floor | `steps / floor` per run, averaged across cell. |
| Axis dwell | Per-axis fraction of PLAN calls (logged in the events stream). Only applies to `greedy`; `random_legal` is always `progression`; `target_aware` is fixed by its own logic. |
| Volatility-clamped fraction | Fraction of EXECUTE_BATCH steps where `volatility > threshold` clamped batch depth to 2. |

**Pre-registered predictions.**

P1: `greedy` exceeds `random_legal` on mean floor reached by at least
one floor on 100-run aggregates (effect-size threshold). If
`greedy − random_legal < 1.0` on mean floor across the slate, the
Sundog claim for this domain is REFUTED.

P2: `target_aware` exceeds `greedy` on mean floor reached by at least
two floors. Smaller margin would suggest the indirect-signal discipline
is nearly free; larger margin would suggest indirect-signal cost is
material. The exact margin is the indirect-signal-cost reading, not a
pass/fail.

P3: Survival rate at F5 is monotonic across `random_legal < greedy <
target_aware`. Non-monotonic survival rate is a red flag for cell
contamination (e.g., one cell ran on a different EyesOnly commit) and
triggers a re-run after `env.json` reconciliation.

P4: `greedy`'s axis dwell shows non-trivial use of `survival` and
`resource` axes (each > 5% of PLAN calls). Near-zero dwell on either
axis means the axis-selection logic is not firing under the slate's
seed range, and the result is not a fair test of the Sundog framing —
it is a test of `progression`-only play. Triggers a slate extension
to seeds with more aggressive enemy spawns.

**Verdict template.**

After the bundle lands, the writeup against this protocol asserts
*one* of three verdicts:

CONFIRM: P1 holds; the `greedy − random_legal` margin is materially
positive; the `target_aware − greedy` margin is reportable but does
not invalidate the indirect-signal discipline. Bundle promoted to
Research Result tier (with caveats: one game, one policy family).

REFUTE: P1 fails. `greedy` does not beat `random_legal` by the
pre-registered margin. The under-game runner remains an Instrumented
Prototype but the *current* greedy policy does not corroborate the
Sundog-shape-helps hypothesis for Gone Rogue. The Sundog framing
remains plausible at the architecture level (the turn envelope can
drive a real engine) but is not corroborated at the policy level
on this game.

AMBIGUOUS: P3 fails (non-monotonic survival rate) or P4 fails
(axis-dwell collapse on `progression`). The slate is invalidated;
the next-pass plan extends seeds and re-runs.

The verdict is filed in the same bundle as the data. The choice of
verdict is data-driven, not author-discretionary. Disposition is
locked: REFUTE means the runner section in `APPLICATIONS.md` is
edited to reflect the negative finding before the writeup is
broadcast.

**Acceptance criteria for running the study.**

The study is gated on three apparatus prerequisites:

(i) `random_legal` policy implemented in
`<sundog>/runners/policies/gone_rogue_random.py`, registered in the
CLI runner's `POLICIES` dict, and unit-tested.

(ii) `target_aware` policy implemented in
`<sundog>/runners/policies/gone_rogue_target_aware.py`, registered
similarly, and unit-tested.

(iii) Hosted EyesOnly accessible at the URL the bundles will record.
For Pass-4-time execution this is local (`http://localhost:8787/public/js`).
For deploy-time execution against `flapsandseals.com` the URL must be
recorded in `env.json` so a reader knows whether the bundle measured
the dev build or the live build.

Until all three prerequisites land, the study is correctly described
as PENDING APPARATUS, not as PENDING DATA. This distinction matters
because it tells skeptic-observers exactly which gate is currently
closed.

## 5. Compressed-Perception Ablation

The Sundog claim that the indirect-signal payload is sufficient for
coherent play is empirically weak unless we can show that *removing*
fields from the payload degrades performance. This section pre-registers
the ablation study that produces that read.

**Ablation conditions.**

`A0 — full payload` (control): the current `compress_perception` output.
This is the baseline; identical to the `greedy` cell of §4.

`A1 — drop alert_level`: the field is set to a constant `"safe"` before
the policy sees the payload. The greedy policy does not currently consume
`alert_level`, so this is the *negative control*: ablation should produce
no measurable difference. If A1 differs from A0 by more than the
seed-noise floor, the experiment has a confound (seed noise
underestimated, or the greedy policy is consuming the field in a way
the source doesn't show).

`A2 — drop in_combat`: the field is set to `False` regardless of engine
state. The greedy policy gates STR-combat behaviour on this flag, so
ablation should degrade survival in floors with active combat. Expected
strong negative effect on `mean floor reached` and `survival rate at F5`.

`A3 — drop hp_ratio`: the field is set to `1.0` (full HP). The greedy
policy uses `hp_ratio` as the primary axis-selection input. Ablation
should collapse axis dwell to `progression` and `resource` (no
`survival`), and degrade survival materially. Expected strongest negative
effect among ablations.

`A4 — drop keys_on_floor`: set to `0`. The `resource` axis trigger goes
away. Expected: axis dwell collapses to `progression` and `survival`
(no `resource`), and floor reached should be near-baseline because the
gameplay impact of missing keys is small in Gone Rogue.

`A5 — drop enemies_visible`: set to `0`. Volatility scoring loses its
primary input. Expected: volatility-clamped fraction drops to near zero;
batches become longer; collisions with newly-spawned enemies should rise.

**Slate.**

Same 100 seeds as §4. Per condition, 100 runs. Total: 600 runs (`A0`
through `A5`). The control condition `A0` is already produced by §4's
`greedy` cell; the ablation slate adds 500 runs.

**Reporting shape.**

Per-condition outcome distribution, mean floor reached, survival rate
at F5 and F10, mean steps, axis dwell, volatility-clamped fraction.
Plus a delta table: each ablation condition's metrics minus `A0`'s
metrics, with seed-paired bootstrap confidence intervals.

**Pre-registered expected pattern.**

| Condition | Predicted ∆(mean floor) vs A0 | Predicted ∆(survival rate at F5) vs A0 |
| --- | --- | --- |
| A1 (drop alert_level) | ≈ 0 (within seed noise) | ≈ 0 |
| A2 (drop in_combat) | strongly negative | strongly negative |
| A3 (drop hp_ratio) | most strongly negative | most strongly negative |
| A4 (drop keys_on_floor) | ≈ 0 to slightly negative | ≈ 0 |
| A5 (drop enemies_visible) | moderately negative | moderately negative |

A1 acting like A2 or A3 is the failure pattern that triggers a re-run
after greedy-policy code re-inspection. A3 acting like A1 is the failure
pattern that suggests `hp_ratio` is not actually load-bearing in the
greedy policy under this slate's seeds — possible if seeds happen to
avoid prolonged low-HP states.

**What the ablation does not test.**

The ablation tests sufficiency of the *current* payload, not optimality.
A field that is *missing* from the payload entirely (e.g., grid layout,
enemy positions, full hand state) cannot be ablated by definition. The
ablation answers "are the fields we expose load-bearing?" and not
"would adding X improve play?" The latter is a separate study.

## 6. Volatility-Threshold Sensitivity Sweep

The runner's batch-depth selector is gated on a single scalar: when
`plan.volatility > volatility_threshold`, batch depth is hard-clamped
to 2. The default threshold is 0.7. This section pre-registers the
sensitivity sweep that produces the empirical reading on whether the
default is right.

**Sweep grid.**

`volatility_threshold ∈ {0.3, 0.5, 0.7, 0.9, 1.01}` × the `greedy` policy
× the same 100 seeds as §4.

`1.01` is a guard value: `plan.volatility ∈ [0.0, 1.0]`, so a threshold
above 1.0 means "never clamp batch depth." This is the "no volatility
discipline" condition — equivalent to letting the policy run full
batches always.

`0.3` is a guard value at the other end: most plans the greedy policy
emits will be clamped, since `volatility = 0.65` triggers on
`hp_ratio < 0.45`. This is "always clamp batch depth."

**Predicted shape of the curve.**

If the volatility discipline is doing useful work, mean floor reached
should be highest near the default 0.7 and degrade at both ends — too
low (always clamping) starves the policy of long stable batches; too
high (never clamping) lets the policy walk into combat without replanning.

If the curve is monotonic — for example, mean floor monotonically
increases as threshold increases — the volatility discipline is not
helping; the runner is paying for replan cost without benefit, and the
default should be raised or the discipline removed.

If the curve is flat — no meaningful difference across threshold values —
the volatility discipline is also not helping; it is a free no-op. This
is a weaker negative finding than monotonic, but it still falsifies the
"volatility discipline matters here" claim.

The expected curve shape (peaked near 0.7) is the structural Sundog
prediction for this domain. The sweep is the test.

**Reporting shape.**

A 5-row table of `volatility_threshold` × `mean floor reached` × `survival rate at F5` × `mean steps` × `volatility-clamped fraction`. Plus a single chart with `volatility_threshold` on the x-axis and `mean floor reached` on the y-axis with seed-paired bootstrap confidence bands.

The chart goes into the gallery card's behaviour-clip Need by proxy: a
single curve summarising 500 runs is a more compact piece of broadcast
content than five clips. Pass 4 of the gallery refresh can land the
chart as a static SVG.

## 7. UI-Bound Playtest Agent Scope

`<EyesOnly>/public/js/playtest-agent.js` ships alongside the headless
runner and is sometimes mistaken for a Sundog surface. This section
documents what the agent actually is, so the broadcast can cite it
honestly and so future work doesn't re-import the misreading.

**What the agent is.**

A self-activating regression bot that runs in the EyesOnly browser tab
when a `?playtest=1` URL parameter is set. It binds to the same DOM
elements a human player uses, observes UI animations and viewport state,
and attempts to provoke the kinds of UX failures that show up in
production but not in unit tests.

The agent's edge-case taxonomy is named explicitly in the source:

```
EDGE_CASES = {
  FAN_BREAKOUT_CLIP:        'fan-breakout-clip',
  FAN_CARD_UNTAPPABLE:      'fan-card-untappable',
  FAN_OCCLUDE_GRID:         'fan-occlude-grid',
  STR_OCCLUDE_FAN:          'str-occlude-fan',
  CARD_DEPLOY_LOST:         'card-deploy-lost',
  GRID_OVERFLOW:            'grid-overflow',
  SCROLL_JUMP:              'scroll-jump',
  ORIENTATION_LAYOUT_BREAK: 'orientation-layout-break',
  TOUCH_DEAD_ZONE:          'touch-dead-zone',
  Z_INDEX_INVERSION:        'z-index-inversion'
}
```

These are UX failure modes: hand-fan card popup clipping outside the
viewport, card touch targets falling below 44 px, fan covering the game
grid entirely, STR combat window covering the fan, card play animation
losing its target on deploy, terminal grid extending beyond its
container, scroll position jumping when a card is played, layout
breakage on orientation change, tappable areas with no event handler,
and z-index ordering inversions. The agent's tick rate is 600 ms
(approximately human-pace) and it caps at 2000 ticks. It rotates
orientation tests on a 15-second cycle.

**What the agent is good for.**

UX regression hunting in the kinds of failure modes that depend on the
browser layout engine, the touch event subsystem, animation timing, and
viewport changes. Things that headless Playwright tests cannot easily
catch because headless Playwright does not lay out for visible humans.

**What the agent is not.**

A Sundog policy. The agent does not compress perception, select an axis,
emit a typed plan, or operate a stop-conditioned action batch. It runs
direct DOM probes and click sequences scoped to UX-failure detection.
Calling it "Sundog under tighter UI constraints" misreads the source —
the constraint isn't the policy's information surface, it's the click
path's reliance on visible DOM. That is QA discipline, not Sundog
discipline.

**Why it ships alongside.**

The same product hosts both surfaces. UX regression and Sundog policy
research benefit from being run on the same engine — a regression in
either bot's expected behaviour is an early signal that EyesOnly's
external surfaces have shifted. Co-location is an apparatus convenience,
not an evidence-tier alignment. The Three Surfaces table in
`APPLICATIONS.md` § EyesOnly / Gone Rogue documents this explicitly.

**One forward path that might bridge the two.**

A future surface that *is* arguably Sundog-shaped: a UI-bound runner
that wraps the headless turn envelope in DOM-visible play (click
sequences, real animations, real wait-for-render). This would test
whether the Sundog turn envelope's claims hold up when the action
substrate is the *visible* DOM rather than the headless API. It does
not exist today. If built, it would belong in the Three Surfaces table
as a fourth row at Instrumented Prototype tier, distinct from the
playtest agent, which would remain a sibling UX automation surface.

## 8. Forward-Looking LAGM

The Live Agentic Game Moderation (LAGM) system is the planned above-game
Sundog application in EyesOnly. It is currently a design document at
`<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md`. No LAGM code
ships in the current EyesOnly tree. This section walks through the
design with the "not built" framing visible, so the broadcast can name
LAGM honestly without overselling it.

**The Sundog shape, on paper.**

LAGM's design specifies four components: a Player Competence Model, a
Live Agentic Moderator, a Floor Synthesis Engine, and a Human-in-the-Loop
Console.

The indirect-signal-to-action pattern is in the first two:

```
Per-floor telemetry  (completion time, damage taken, ability usage,
                      rope sophistication, puzzle solve time,
                      backtracking, secret discovery rate,
                      exploit pattern attempts, hoarding behaviour)
        │
        ▼
Player Competence Model
    compresses telemetry into:
        competence       ∈ [0, 1]
        confidence       ∈ [0, 1]
        frustration      ∈ [0, 1]
        mastery.{rope, stealth, puzzles, …}
        │
        ▼
Live Agentic Moderator
    emits next-floor intent:
        goal:                   "Increase stealth pressure"
        difficulty_delta:        +0.15
        introduce_new_synergy:   true
        counter_exploit:        "rope-cheese"
        narrative_tone:         "tense"
        │
        ▼
Floor Synthesis Engine
    consumes the intent and authors the next floor.
```

The per-floor telemetry is the *indirect signal*. The competence model
is the *transformation* — collapsing nine raw inputs into four indices.
The moderator's next-floor intent is the *action* — a small structured
output that the floor synthesiser can act on without needing to read
the player's intent or the full telemetry stream.

That is exactly the indirect-signal-to-action shape the Sundog research
program names. The same shape as the photometric experiment, the
verb-field NPC system, the softbody graph telemetry.

**The integrity safeguards in the design.**

LAGM specifies four run integrity classes: Class A (Static, no live
manipulation, only class eligible for the global leaderboard); Class B
(Human-Moderated, intervened by a designer through the Human-in-the-Loop
Console); Class C (Agent-Moderated, intervened by the Live Agentic
Moderator); Class D (Hybrid).

The integrity classification matters for the Sundog framing because it
makes the agent's effect on the run *legible*. A Class C run cannot
quietly become a Class A run; the run header records the class
permanently. This is the same kind of pre-registration discipline Money
Bags applied to its falsification rubric, applied here to the broadcast
shape of agent-moderated play.

The "BIG BROTHER" mode in LAGM's Section 5.2 is a design-time consent
gate: live manipulation is disabled by default, requires a global
toggle, and players must acknowledge a ping to enter monitored ascent
mode. This is not just a UX nicety; it is the design's answer to the
tension that any above-game agent reading player telemetry creates a
surveillance surface. The Sundog research-program framing should
acknowledge this rather than gloss it.

**What is not built.**

No LAGM code exists in `<EyesOnly>/public/js/` today. The Player
Competence Model is not implemented. The Live Agentic Moderator is not
implemented. The Floor Synthesis Engine, beyond the existing Unified
Designer that LAGM would call into, is not implemented. The
Human-in-the-Loop Console is not implemented. The run integrity
classification is not implemented. The "BIG BROTHER" toggle is not
implemented.

The design document is the inspectable surface. That is the Conceptual
Lineage / Forward-Looking Application Design tier in the Three Surfaces
table.

**The bridge between the under-game runner and LAGM.**

The matched-seed JSONL bundles produced by §4 are the natural
*synthetic-player* training and evaluation data for a future LAGM
moderator. The same per-floor metrics LAGM's competence model is
designed to consume — completion time, damage taken, axis dwell,
volatility-clamped fraction, outcome distributions — are exactly the
fields the runner emits today. The runner is therefore not just an
under-game Sundog application; it is also the apparatus that makes
the above-game Sundog application *trainable* against a stable,
seedable substrate.

This sequencing is intentional. Build the under-game runner first.
Generate matched-seed bundles. Use the bundles to scaffold the
above-game moderator's competence model on synthetic players before
hooking it to live ones. Defer the BIG BROTHER consent gate to
implementation time, when the design's commitments meet the actual
deploy surface at flapsandseals.com.

When LAGM ships, the EyesOnly Three Surfaces table grows from one
Sundog row + one sibling automation row + one design row to two
Sundog rows + one sibling automation row + one no-longer-design
row. At that point the cross-application comparison table in
`APPLICATIONS.md` § Cross-Application Comparison can split EyesOnly
into two rows — under-game runner and above-game moderator — to
surface the vertical-composition contribution. Until then, one row.

## 9. Claim Boundary

The Three Surfaces in EyesOnly each carry their own safe and avoid
claims. Restated here in compact form, with each surface's tier and the
load-bearing apparatus citation.

**Surface 1 — Headless turn-envelope runner. Tier: Instrumented Prototype.**

Safe claim:

> The Sundog turn-envelope architecture (PERCEIVE → PLAN → EXECUTE_BATCH
> with named stop conditions and volatility-modulated batch depth) drives
> the real Gone Rogue JavaScript engine through Playwright against the
> `GoneRogue.headless` API. Perception compression is implemented in
> `GoneRogueAdapter.compress_perception` and produces a typed payload of
> floor / biome / HP / alert / inventory / gates / combat fields. The
> greedy policy in `gone_rogue_greedy.py` selects between progression /
> resource / survival axes by HP ratio and key-on-floor presence. The
> harness is policy-pluggable, seedable, and supports matched-seed
> multi-policy comparison. CLI entry points are
> `runners/gone_rogue_headless.py` and `runners/gone_rogue_ui.py`.

Avoid:

> EyesOnly proves the Sundog theorem for procedural games.

The matched-seed multi-policy study is pre-registered in §4 and not yet
run. Pending the `random_legal` and `target_aware` policy
implementations and a hosted EyesOnly URL recorded in `env.json`. The
status is correctly described as PENDING APPARATUS, not PENDING DATA.

**Surface 2 — UI-bound playtest agent. Tier: Product Expression — sibling UX automation, not Sundog evidence.**

Safe claim:

> `<EyesOnly>/public/js/playtest-agent.js` is a UI-bound regression bot
> for hunting UX failure modes (fan clipping, card touch-target
> sizing, animation loss, scroll jumps, orientation breakage,
> z-index inversion, etc.) under human DOM constraints. It ships
> alongside the headless runner because the same product hosts both,
> not because the two share a Sundog architecture.

Avoid:

> The playtest agent demonstrates Sundog under tighter UI constraints.

The agent's failure-mode taxonomy (the `EDGE_CASES` constant) is the
direct check on this boundary. The agent does not compress perception,
emit a typed plan, or operate a stop-conditioned batch. Its constraint
surface is the visible DOM, not a policy-information limit.

**Surface 3 — Live Agentic Game Moderation. Tier: Conceptual Lineage / Forward-Looking Application Design.**

Safe claim:

> `<EyesOnly>/docs/MANIPULATION_LAYER_AGENT_MODERATION.md` specifies an
> above-game agent that compresses per-floor player telemetry
> (completion time, damage, ability usage, rope sophistication, puzzle
> solve time, backtracking, secret discovery, exploit attempts,
> hoarding) into competence / confidence / frustration / mastery
> indices, and emits structured next-floor intent (goal, difficulty
> delta, synergy injection, exploit counter, narrative tone) that a
> floor-synthesis engine consumes. This is the indirect-signal-to-action
> shape the Sundog research program names, applied above the game
> rather than below it. The design includes integrity classification
> (Class A/B/C/D), a Human-in-the-Loop console, and a "BIG BROTHER"
> consent gate. No LAGM code ships in EyesOnly today; the design is
> the inspectable surface.

Avoid:

> EyesOnly performs Sundog-driven level manipulation.

The placeholder phrase "level manipulation" in earlier broadcasts has
been a gesture at LAGM. LAGM is design surface, not shipping evidence.
Promoting it to evidence requires implementing the Player Competence
Model, the Live Agentic Moderator, the Floor Synthesis Engine
integration, and the integrity classification — none of which exist in
the current `<EyesOnly>/public/js/` tree.

**What promotes the runner from Instrumented Prototype to Research Result.**

A clean execution of the §4 matched-seed study that produces a CONFIRM
verdict, plus the §5 ablation study with the predicted A1 negative-control
result, plus the §6 volatility sweep with a non-flat non-monotonic curve.
Three studies, three verdicts, all data-driven, all with verdict templates
pre-registered in this document.

**What promotes LAGM from Conceptual Lineage to Instrumented Prototype.**

Implementation of at least the Player Competence Model and the Live
Agentic Moderator stubs against the runner's matched-seed bundles
(synthetic players first). The "BIG BROTHER" consent gate and Class A/B/C/D
integrity classification ship before any live-player exposure at
flapsandseals.com. At Instrumented Prototype tier, LAGM enters the Three
Surfaces table as Surface 4; the cross-application comparison can split
EyesOnly into under-game and above-game rows.

**Summary of the broadcast posture.**

EyesOnly hosts one instrumented Sundog runner, one sibling UX
automation surface that is not Sundog evidence, and one forward-looking
Sundog design that is not yet built. The runner is apparatus pending
study; the playtest agent is QA discipline; LAGM is intent pre-registered
against the same indirect-signal pattern, awaiting implementation.

The vertical-composition story — under-game runner and above-game
moderator at the same product engine — is reserved for the broadcast
once both surfaces ship as code. Until then, one row in the
cross-application table; the discipline holds.

---

## Appendix: Open Questions for the Team

These are the questions the Pass-4 deep-dive surfaced that defer to the
research-team and the EyesOnly maintainer rather than committing in this
document.

**A1. Slate seed range.** The §4 protocol uses seeds `[42, 43, …, 141]`
(100 seeds). Is 100 enough to read a one-floor effect-size threshold,
or should the slate extend to 250 or 500? Recommend running 100 first
and reading the per-cell variance; extend if the bootstrap CI is wider
than the predicted effect.

**A2. EyesOnly URL recorded in `env.json`.** For the Pass-4-time study,
`http://localhost:8787/public/js` is the obvious choice. For deploy-time
re-runs, should the protocol also commit to a `flapsandseals.com` URL
so the production build is measured? Recommend yes, but as a separate
follow-up cell rather than the primary slate; production-build drift
is a real confound and deserves its own row.

**A3. Whether to ship `random_legal` first or `target_aware` first.**
`random_legal` is the load-bearing baseline for the §4 hypothesis; it
is also simpler. `target_aware` is more involved (needs grid access,
needs gate-solution logic) but produces the indirect-signal-cost
reading. Recommend `random_legal` first, then run the partial study
(2-cell), then add `target_aware` for the full 3-cell study.

**A4. Whether the §6 volatility sweep deserves its own slate or can
piggyback on §4's seed set.** Recommend piggyback: same 100 seeds, only
the `greedy` cell with five threshold values. Cheaper, comparable, and
the seed-paired comparison is the right statistical frame.

**A5. Whether to publish behaviour clips before or after the §4
study lands.** Behaviour clips are a gallery-card Need but they
broadcast a story. Recommend after the study: clips that depict
greedy-success, greedy-failure, and random-legal-failure on the *same*
seed are far more useful as evidence than clips of success only.

**A6. Whether LAGM's "BIG BROTHER" consent gate becomes a discussion
point in the Sundog research-program voice or stays scoped to EyesOnly's
product voice.** Recommend the latter for now; the consent surface is
a product-design decision and shouldn't borrow the research program's
voice. If LAGM ships and the integrity-class system becomes a
generalisable Sundog-research pattern, revisit.

— EyesOnly maintainer / Sundog application correction pass, 2026-05-08
