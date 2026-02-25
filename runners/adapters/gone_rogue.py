"""
sundog.runners.adapters.gone_rogue
===================================
Playwright adapter that drives the EyesOnly *Gone Rogue* browser minigame
through the sundog turn-envelope runner framework.

Architecture
------------

    sundog TurnEnvelopeRunner (Python)
        └── GoneRogueAdapter                 (this module)
                └── Playwright browser
                        └── GoneRogue.headless API   (EyesOnly JS engine)
                                └── Real Gone Rogue game logic

The adapter:
- Manages a Playwright browser session (headless or visible).
- Loads the Gone Rogue game dependencies into a blank page via ``page.add_script_tag``.
- Wraps ``GoneRogue.headless.getState()``, ``getLegalActions()``,
  ``applyAction()``, ``resetToState()``.
- Provides a ``compress_perception()`` method that maps the raw JS game state
  to a sundog ``PerceptionPayload``.

Usage
-----
::

    from sundog.runners.adapters.gone_rogue import GoneRogueAdapter

    adapter = GoneRogueAdapter(
        base_url="http://localhost:8787/public/js",   # EyesOnly local dev
        headless=True,
    )
    with adapter:
        adapter.reset(seed=42)
        state = adapter.get_state()
        actions = adapter.get_legal_actions()
        result = adapter.apply_action(actions[0])

``base_url`` may also be a ``file://`` path or any URL serving the EyesOnly
``public/js/`` directory.  For CI, set ``EYESONLY_BASE_URL`` env variable.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ordered list of script dependencies from EyesOnly public/js/
# Mirror of the order used by public/tests/test-headless-integration.html
# ---------------------------------------------------------------------------
GONE_ROGUE_SCRIPTS = [
    "seeded-random.js",
    "card-system.js",
    "lighting-system.js",
    "ground-effects.js",
    "boss-encounters.js",
    "secret-floors.js",
    "gone-rogue-canvas.js",
    "gone-rogue.js",
]

# The headless adapter bundled with EyesOnly (in public/tests/)
HEADLESS_ADAPTER_SCRIPT = "agent-headless-adapter.js"

# Minimal DOM stubs injected before any game script so the game doesn't
# crash on missing browser APIs when a plain blank page is used.
_DOM_STUBS_JS = """
(function() {
  // Stub requestAnimationFrame if absent (blank pages may lack it)
  if (typeof window.requestAnimationFrame === 'undefined') {
    window.requestAnimationFrame = function(cb) { return setTimeout(cb, 16); };
    window.cancelAnimationFrame = function(id) { clearTimeout(id); };
  }
  // Stub canvas getContext if needed
  if (typeof HTMLCanvasElement !== 'undefined') {
    var _origGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type) {
      try { return _origGetContext.apply(this, arguments); } catch(e) { return null; }
    };
  }
  // localStorage shim (Playwright pages have it, but be safe)
  if (typeof window.localStorage === 'undefined') {
    var _store = {};
    window.localStorage = {
      getItem: function(k) { return Object.prototype.hasOwnProperty.call(_store, k) ? _store[k] : null; },
      setItem: function(k, v) { _store[k] = String(v); },
      removeItem: function(k) { delete _store[k]; },
      clear: function() { _store = {}; }
    };
  }
})();
"""

# ---------------------------------------------------------------------------
# GoneRogueAdapter
# ---------------------------------------------------------------------------

class GoneRogueAdapter:
    """
    Drives the EyesOnly *Gone Rogue* game via a Playwright browser session.

    Parameters
    ----------
    base_url:
        URL of the EyesOnly ``public/js/`` directory, e.g.
        ``http://localhost:8787/public/js`` or a local ``file:///`` URL.
        Falls back to the ``EYESONLY_BASE_URL`` environment variable.
    headless_adapter_base_url:
        URL of the directory containing ``agent-headless-adapter.js``
        (usually the EyesOnly ``public/tests/`` directory).  Defaults to
        ``<base_url>/../tests`` if not set.
    headless:
        ``True`` (default) for headless Playwright mode; ``False`` for a
        visible browser window (UI-bound runner).
    slow_mo:
        Milliseconds between Playwright actions (useful for UI mode to make
        actions visible).
    timeout_ms:
        Default navigation/evaluation timeout in milliseconds.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        headless_adapter_base_url: Optional[str] = None,
        headless: bool = True,
        slow_mo: int = 0,
        timeout_ms: int = 10_000,
    ):
        self.base_url = (
            base_url
            or os.environ.get("EYESONLY_BASE_URL", "")
        ).rstrip("/")
        if not self.base_url:
            raise ValueError(
                "GoneRogueAdapter requires a base_url pointing to the "
                "EyesOnly public/js/ directory (or set EYESONLY_BASE_URL)."
            )
        # Default adapter URL: sibling 'tests' directory
        if headless_adapter_base_url:
            self._adapter_base = headless_adapter_base_url.rstrip("/")
        else:
            # e.g. http://host/public/js -> http://host/public/tests
            parts = self.base_url.rsplit("/", 1)
            self._adapter_base = parts[0] + "/tests" if len(parts) == 2 else self.base_url

        self._headless = headless
        self._slow_mo = slow_mo
        self._timeout_ms = timeout_ms

        self._playwright = None
        self._browser = None
        self._page = None
        self._game_loaded = False

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "GoneRogueAdapter":
        self._start_browser()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _start_browser(self):
        """Launch Playwright browser and open a blank page."""
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "playwright is required for GoneRogueAdapter. "
                "Install it with: pip install playwright && playwright install chromium"
            ) from exc

        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(
            headless=self._headless,
            slow_mo=self._slow_mo,
        )
        self._page = self._browser.new_page()
        self._page.set_default_timeout(self._timeout_ms)

    def _load_game(self):
        """Navigate to a blank page and inject all Gone Rogue scripts."""
        if self._game_loaded:
            return

        # Open a blank page so we have a proper browser context
        self._page.goto("about:blank")

        # Inject DOM stubs first
        self._page.evaluate(_DOM_STUBS_JS)

        # Add a minimal HTML structure the game might depend on
        self._page.evaluate("""
            () => {
                var canvas = document.createElement('canvas');
                canvas.id = 'gone-rogue-canvas';
                canvas.width = 800;
                canvas.height = 400;
                document.body.appendChild(canvas);

                var terminal = document.createElement('div');
                terminal.id = 'terminal-output';
                document.body.appendChild(terminal);
            }
        """)

        # Load each game script in dependency order
        for script_name in GONE_ROGUE_SCRIPTS:
            url = f"{self.base_url}/{script_name}"
            self._page.add_script_tag(url=url)

        # Load the headless adapter (from tests/ directory)
        adapter_url = f"{self._adapter_base}/{HEADLESS_ADAPTER_SCRIPT}"
        self._page.add_script_tag(url=adapter_url)

        # Verify critical globals loaded
        ok = self._page.evaluate("""
            () => (
                typeof GoneRogue !== 'undefined' &&
                typeof GoneRogue.headless !== 'undefined' &&
                typeof HeadlessAdapter !== 'undefined'
            )
        """)
        if not ok:
            raise RuntimeError(
                "Gone Rogue scripts did not load correctly. "
                f"Verify base_url is accessible: {self.base_url}"
            )

        self._game_loaded = True

    def reset(self, seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Start (or restart) a Gone Rogue game with an optional seed.

        Returns the initial compressed state dict.
        """
        if not self._page:
            self._start_browser()
        self._load_game()

        # Initialize the game engine
        self._page.evaluate("() => GoneRogue.init()")

        # Apply seed if given (before start)
        if seed is not None:
            self._page.evaluate(f"() => GoneRogue.setSeed({seed})")

        # Start the game
        self._page.evaluate("() => GoneRogue.start({})")

        # Give the game loop one tick to settle
        time.sleep(0.1)

        return self.get_state()

    def close(self):
        """Tear down the browser session."""
        try:
            if self._browser:
                self._browser.close()
        finally:
            if self._playwright:
                self._playwright.stop()
            self._playwright = None
            self._browser = None
            self._page = None
            self._game_loaded = False

    # ------------------------------------------------------------------
    # GoneRogue.headless API wrappers
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return the current full game state (via ``GoneRogue.headless.getState()``)."""
        return self._page.evaluate("() => GoneRogue.headless.getState()")

    def get_legal_actions(self) -> List[Dict[str, Any]]:
        """Return legal actions for the current state."""
        return self._page.evaluate("() => GoneRogue.headless.getLegalActions()")

    def apply_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply a single action through the game engine.

        Parameters
        ----------
        action:
            Action dict as returned by ``get_legal_actions()``, e.g.
            ``{"type": "move", "dx": 1, "dy": 0, ...}``.

        Returns
        -------
        dict with keys: ``success``, ``state``, ``messages``, ``events``
        """
        result = self._page.evaluate(
            "(action) => GoneRogue.headless.applyAction(action)",
            action,
        )
        return result if result else {"success": False, "messages": [], "events": []}

    def get_grid(self) -> Dict[str, Any]:
        """Return the current floor's map grid data."""
        return self._page.evaluate("() => GoneRogue.headless.getGrid()")

    def reset_to_state(self, state: Dict[str, Any]) -> bool:
        """Restore a specific game state snapshot (for replay)."""
        return bool(
            self._page.evaluate(
                "(state) => GoneRogue.headless.resetToState(state)",
                state,
            )
        )

    def is_game_active(self) -> bool:
        """Return True if the game is currently running."""
        return bool(self._page.evaluate("() => GoneRogue.isActive()"))

    # ------------------------------------------------------------------
    # Turn-envelope helpers
    # ------------------------------------------------------------------

    def compress_perception(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compress the raw Gone Rogue game state into a small, stable
        perception payload for the turn-envelope PERCEIVE step.

        Maps the JS ``getState()`` return value to a dict matching the
        ``PerceptionPayload`` field names used by sundog policies.
        """
        if state is None:
            return {}

        player = state.get("player", {})
        hp = player.get("hp", 0)
        max_hp = player.get("maxHp", 10)
        hp_ratio = hp / max_hp if max_hp > 0 else 0.0

        # Alert / combat status
        alert = state.get("alertLevel", "safe")
        in_combat = bool(state.get("strCombatActive", False))

        # Enemies
        enemies = state.get("enemies", []) or []
        enemies_visible = len([e for e in enemies if not e.get("dead", False)])

        # Items / keys on floor
        items = state.get("items", []) or []
        keys_on_floor = len([i for i in items if i.get("type") == "key"])

        # Run state (keys, gates)
        run_state = state.get("runState", {}) or {}
        keys_owned: List[Any] = run_state.get("keysOwned", []) or []
        gates_spawned: int = run_state.get("gatesSpawnedThisRun", 0)

        # Floor / biome
        floor = state.get("floor", 1)
        biome = state.get("biome", "unknown")

        # Check if there's a gate entity nearby (exit tile on map)
        grid_data = state.get("gridData")  # may be None; only present if adapter fetches it
        has_exit = bool(state.get("hasExit", False))

        return {
            "floor_index": floor - 1,          # 0-based to match sundog convention
            "biome": biome,
            "hp": hp,
            "max_hp": max_hp,
            "hp_ratio": hp_ratio,
            "keys_inventory": {"owned": len(keys_owned)},
            "keys_on_floor": keys_on_floor,
            "gates_visible": int(gates_spawned > 0 or has_exit),
            "gate_requires_key": False,         # GoneRogue gates don't use a key system; adjusted below
            "gate_key_color": None,
            "enemies_visible": enemies_visible,
            "in_combat": in_combat,
            "alert_level": alert,
            "steps_on_floor": player.get("combatEntries", 0),
            "pity_triggered": False,
            "sidequest_entrance": None,
            "floors_remaining": max(0, 32 - (floor - 1)),
            # GoneRogue-specific extras (not in base PerceptionPayload)
            "stealth": player.get("stealth", 3),
            "detection": player.get("detection", 0),
            "energy": player.get("energy", 5),
            "max_energy": player.get("maxEnergy", 5),
            "str_combat_active": in_combat,
            "advantage": state.get("strCombatAdvantage", "neutral"),
        }

    # ------------------------------------------------------------------
    # Batch-run helpers (used by the headless runner CLI)
    # ------------------------------------------------------------------

    def run_turn_envelope(
        self,
        policy,
        max_steps: int = 2000,
        max_batch: int = 10,
        volatility_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """
        Drive a complete Gone Rogue run using the turn-envelope pattern.

        PERCEIVE → PLAN → EXECUTE_BATCH  (repeat until game over or max_steps)

        Parameters
        ----------
        policy:
            A ``GoneRoguePolicy`` (or any object with a ``plan(perception)``
            method returning a ``TurnPlan``).
        max_steps:
            Hard cap on total actions executed per run.
        max_batch:
            Maximum batch depth per turn envelope.
        volatility_threshold:
            Batches are shallow when ``plan.volatility > threshold``.

        Returns
        -------
        dict with summary: floor reached, outcome, steps, events log.
        """
        steps = 0
        events_log: List[Dict[str, Any]] = []
        outcome = "ongoing"

        while steps < max_steps:
            state = self.get_state()
            if state is None or not self.is_game_active():
                outcome = "game_ended"
                break

            # ── PERCEIVE ──────────────────────────────────────────────
            compressed = self.compress_perception(state)
            perception = policy.perceive(compressed)

            # ── PLAN ──────────────────────────────────────────────────
            plan = policy.plan(perception)

            # ── EXECUTE_BATCH ─────────────────────────────────────────
            batch_depth = (
                min(2, len(plan.action_batch))
                if plan.volatility > volatility_threshold
                else min(len(plan.action_batch), max_batch)
            )

            for i, action in enumerate(plan.action_batch[:batch_depth]):
                if steps >= max_steps:
                    break

                result = self.apply_action(action)
                steps += 1

                event_entry: Dict[str, Any] = {
                    "step": steps,
                    "action": action,
                    "result": result,
                    "floor": state.get("floor", 1),
                    "biome": state.get("biome", "unknown"),
                }
                events_log.append(event_entry)

                # Check terminal conditions
                player = (self.get_state() or {}).get("player", {})
                if player.get("hp", 1) <= 0:
                    outcome = "died"
                    break

                # Check stop conditions from plan
                result_events = result.get("events", []) or []
                fired = set(e.get("type") for e in result_events if e)
                stop_fired = fired & set(plan.stop_conditions)
                if stop_fired:
                    event_entry["stop_condition"] = list(stop_fired)[0]
                    break

            if outcome in ("died", "completed", "game_ended"):
                break

            # Check if game ended after batch
            state_after = self.get_state()
            if state_after is None or not self.is_game_active():
                outcome = "completed"
                break

        final_state = self.get_state() or {}
        return {
            "outcome": outcome,
            "floor": final_state.get("floor", 1),
            "biome": final_state.get("biome", "unknown"),
            "steps": steps,
            "events": events_log,
            "player": final_state.get("player", {}),
        }
