# Sundog Runner Framework — Gone Rogue Integration

This document covers the sundog runner framework as applied to the
**EyesOnly – Gone Rogue** minigame.

---

## Overview

The sundog framework drives external game engines using the
**Sundog Alignment Theorem turn envelope**:

```
TURN_ENVELOPE {
  PERCEIVE  (compress salient state)
  PLAN      (axis/utility framing)
  EXECUTE_BATCH (N actions until stop condition)
}
```

For **Gone Rogue** (the ASCII stealth roguelike in [EyesOnly](https://github.com/humiliati/EyesOnly)),
the framework connects via **Playwright** to the real JavaScript game engine,
calling the `GoneRogue.headless` API directly.

```
sundog TurnEnvelopeRunner  (Python)
  └── GoneRogueAdapter
          └── Playwright (headless or visible browser)
                  └── GoneRogue.headless.{getState, getLegalActions, applyAction}
                          └── Actual Gone Rogue JS engine
```

---

## Quick Start

### Prerequisites

1. **EyesOnly** must be running and its `public/js/` directory accessible via HTTP
   (or served from a local checkout).

   ```bash
   # Local dev (from EyesOnly repo):
   npm run dev   # starts Cloudflare Workers dev server at http://localhost:8787
   ```

2. **Playwright** must be installed:

   ```bash
   pip install playwright
   playwright install chromium
   ```

3. Set the `EYESONLY_BASE_URL` environment variable:

   ```bash
   export EYESONLY_BASE_URL=http://localhost:8787/public/js
   ```

---

## Headless Batch Runner

Runs Gone Rogue in a **headless** (no visible window) Playwright browser.
Suitable for CI, large batch runs, and automated balance testing.

```bash
python -m sundog.runners.gone_rogue_headless \
  --runs 100 \
  --seed 42 \
  --eyesonly-url http://localhost:8787/public/js \
  --out gr_headless.jsonl \
  --policy greedy \
  --max-batch 8 \
  --max-steps 1000
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--runs N` | 10 | Number of simulation runs |
| `--seed N` | 0 | Base seed; run *i* uses `seed+i` |
| `--eyesonly-url URL` | `EYESONLY_BASE_URL` | URL of `public/js/` directory |
| `--eyesonly-tests-url URL` | sibling `tests/` | URL of `public/tests/` (headless adapter) |
| `--out PATH` | `gr_headless.jsonl` | Output JSONL path |
| `--policy NAME` | `greedy` | Policy to use |
| `--max-batch N` | 8 | Max batch depth per turn envelope |
| `--volatility F` | 0.7 | Volatility threshold for shallow batches |
| `--max-steps N` | 1000 | Hard cap on actions per run |
| `--slow-mo MS` | 0 | Playwright action delay (ms) |
| `--quiet` | — | Suppress per-run progress |

---

## UI-Bound Runner

Runs Gone Rogue in a **visible** browser window. Intended for manual
inspection, screenshot capture, and debugging.

```bash
python -m sundog.runners.gone_rogue_ui \
  --runs 3 \
  --seed 42 \
  --eyesonly-url http://localhost:8787/public/js \
  --slow-mo 300 \
  --screenshot-dir screenshots/ \
  --keep-open
```

### Additional Options (UI mode)

| Flag | Default | Description |
|------|---------|-------------|
| `--slow-mo MS` | 200 | ms between Playwright actions (higher = more visible) |
| `--screenshot-dir DIR` | — | Save end-of-run PNG screenshots |
| `--keep-open` | — | Pause before closing browser (for inspection) |
| `--max-batch N` | 6 | Shorter default than headless for visibility |

---

## Output Format

Both runners write one JSON record per run to the output JSONL file:

```json
{
  "run_index": 0,
  "seed": 42,
  "policy": "greedy",
  "outcome": "died",
  "floor": 7,
  "biome": "industrial",
  "steps": 312,
  "player": {
    "hp": 0,
    "maxHp": 10,
    "x": 18,
    "y": 9
  },
  "events": [
    {
      "step": 1,
      "action": {"type": "move", "dx": 1, "dy": 0},
      "floor": 1,
      "biome": "forest"
    }
  ]
}
```

---

## Turn-Envelope Architecture

### PERCEIVE

`GoneRogueAdapter.compress_perception(state)` maps the full JS game state
returned by `GoneRogue.headless.getState()` to a compact dict:

```python
{
  "floor_index":    int,     # 0-based floor number
  "biome":          str,     # e.g. "industrial", "aerospace_museum"
  "hp":             int,
  "max_hp":         int,
  "hp_ratio":       float,   # hp / max_hp
  "keys_inventory": dict,
  "keys_on_floor":  int,
  "gates_visible":  int,
  "enemies_visible":int,     # live enemies only (dead excluded)
  "in_combat":      bool,    # STR combat active
  "alert_level":    str,     # "safe" | "caution" | "danger"
  "stealth":        int,
  "detection":      int,
  "floors_remaining": int,
  # … more GoneRogue-specific fields
}
```

### PLAN

The policy's `plan(perception)` returns a `TurnPlan`:

```python
TurnPlan(
  axis_priority   = "progression",   # or "resource" | "survival"
  short_strategy  = "Explore floor — move to reveal map and locate exit.",
  action_batch    = [
    {"type": "move", "dx": 1, "dy": 0},
    {"type": "move", "dx": 0, "dy": 1},
    # …
  ],
  stop_conditions = ["new_enemy", "floor_changed", "died", "combat_start"],
  volatility      = 0.1,
)
```

### EXECUTE_BATCH

`GoneRogueAdapter.run_turn_envelope()` applies actions via
`GoneRogue.headless.applyAction(action)`, checking stop conditions after each:

- **Volatile** plans (`volatility > threshold`): max 2 actions per batch
- **Stable** plans: up to `max_batch` actions per batch

**Stop conditions** that interrupt the batch:
- `died` — player HP reached 0
- `floor_changed` — player used the exit tile
- `new_enemy` / `combat_start` — enemy engaged
- `gate_encountered` — gate entity on the map

---

## Policies

### Built-in: `greedy`

`sundog.runners.policies.gone_rogue_greedy.GoneRogueGreedyPolicy`

**Axis selection:**
- `survival` — HP < 25%: flee/rest
- `resource` — key visible on floor, safe: collect items
- `progression` — default: find exit / fight

**Volatility:**
- In STR combat → 0.9 (very shallow batches)
- HP < 45% → 0.65
- Enemy visible → 0.8

### Adding a New Policy

1. Create `sundog/runners/policies/my_policy.py`:

```python
from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy
from sundog.runners.policy import PerceptionPayload, TurnPlan

class MyPolicy(GoneRogueGreedyPolicy):
    def plan(self, perception: PerceptionPayload) -> TurnPlan:
        # … your logic
        return TurnPlan(
            axis_priority="progression",
            short_strategy="My custom strategy.",
            action_batch=[{"type": "wait"}],
            stop_conditions=["died", "new_enemy"],
        )
```

2. Register it in the CLI runner:

```python
# in gone_rogue_headless.py
from sundog.runners.policies.my_policy import MyPolicy
POLICIES = {
    "greedy": GoneRogueGreedyPolicy,
    "my_policy": MyPolicy,          # ← add here
}
```

3. Use it:

```bash
python -m sundog.runners.gone_rogue_headless \
  --policy my_policy \
  --eyesonly-url http://localhost:8787/public/js
```

---

## Python API

### Direct Usage

```python
from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy

with GoneRogueAdapter(
    base_url="http://localhost:8787/public/js",
    headless=True,
) as adapter:
    # Start game with seed
    adapter.reset(seed=42)

    # PERCEIVE
    state = adapter.get_state()
    compressed = adapter.compress_perception(state)

    # PLAN
    policy = GoneRogueGreedyPolicy()
    perception = policy.perceive(compressed)
    plan = policy.plan(perception)

    # EXECUTE_BATCH — run until game ends
    summary = adapter.run_turn_envelope(policy=policy, max_steps=1000)
    print(summary)
```

### GoneRogueAdapter Methods

| Method | Description |
|--------|-------------|
| `reset(seed=None)` | Start a new game (optionally seeded) |
| `get_state()` | `GoneRogue.headless.getState()` |
| `get_legal_actions()` | `GoneRogue.headless.getLegalActions()` |
| `apply_action(action)` | `GoneRogue.headless.applyAction(action)` |
| `get_grid()` | `GoneRogue.headless.getGrid()` |
| `reset_to_state(state)` | Restore a state snapshot |
| `is_game_active()` | `GoneRogue.isActive()` |
| `compress_perception(state)` | Map raw state → compact perception dict |
| `run_turn_envelope(policy, ...)` | Run complete PERCEIVE→PLAN→EXECUTE loop |

---

## Running Tests

```bash
# Unit tests (no browser needed):
python -m pytest sundog/tests/runners/test_gone_rogue.py -v

# Integration tests (needs EyesOnly running):
EYESONLY_BASE_URL=http://localhost:8787/public/js \
  python -m pytest sundog/tests/runners/test_gone_rogue.py -v -m integration
```

---

## GoneRogue.headless API Reference

The underlying JavaScript API (defined in `EyesOnly/public/js/gone-rogue.js`):

| JS Method | Description |
|-----------|-------------|
| `GoneRogue.headless.getState()` | Full game state (player, enemies, floor, biome, runState, …) |
| `GoneRogue.headless.getLegalActions()` | List of valid action objects for current state |
| `GoneRogue.headless.applyAction(action)` | Execute one action through the real game engine |
| `GoneRogue.headless.getGrid()` | Map grid data (tiles, dimensions) |
| `GoneRogue.headless.resetToState(state)` | Restore a previously exported state snapshot |
| `GoneRogue.setSeed(seed)` | Set seed before `GoneRogue.start()` for deterministic runs |

**Action types** (from `getLegalActions()`):

| Type | Description |
|------|-------------|
| `move` | Move one tile; fields: `dx`, `dy`, `targetX`, `targetY` |
| `useCard` | Play a card from hand; field: `cardIndex` |
| `flee` | Flee from STR combat |
| `pickup` | Pick up item at player's position |
| `pickupCurrency` | Pick up currency (¢) at player's position |
| `exit` | Use exit tile to descend to next floor |
| `useActiveItem` | Trigger active item |
| `wait` | Pass turn |

---

## Browser Harness

`sundog/runners/adapters/gone_rogue_harness.html` is a self-contained HTML
page that the `GoneRogueAdapter` can open directly via `page.goto()`.  It:

- Loads all game dependencies sequentially from a configurable `?base=` URL
- Injects DOM stubs (canvas, terminal div, requestAnimationFrame)
- Exposes `window.__harness` for direct Playwright access

Open manually for debugging:

```
http://localhost:8787/sundog/runners/adapters/gone_rogue_harness.html?base=http://localhost:8787/public/js
```
