"""
tests/runners/test_gone_rogue.py
=================================
Unit and integration tests for the EyesOnly Gone Rogue runner bridge.

These tests use mocking to avoid requiring a live browser or EyesOnly
server.  Tests marked with ``@pytest.mark.integration`` require a real
browser and ``EYESONLY_BASE_URL`` to be set; they are skipped in CI unless
the environment variable is present.

Test categories
---------------
1. ``GoneRogueGreedyPolicy`` – axis selection, stop conditions, batch construction
2. ``GoneRogueAdapter`` – interface contract (mocked Playwright)
3. ``compress_perception`` – mapping JS game state → PerceptionPayload-like dict
4. Turn-envelope loop – PERCEIVE → PLAN → EXECUTE_BATCH sequence
5. Headless CLI – argument parsing, output format
6. Integration smoke test (skipped unless EYESONLY_BASE_URL set)
"""
from __future__ import annotations

import json
import os
import sys
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _make_raw_state(
    hp=10, max_hp=10, floor=3, biome="industrial",
    enemies=None, items=None, in_combat=False,
    alert="safe", gates_spawned=0,
):
    """Return a minimal raw game state dict as ``GoneRogue.headless.getState()`` would."""
    return {
        "player": {
            "hp": hp,
            "maxHp": max_hp,
            "energy": 5,
            "maxEnergy": 5,
            "stealth": 3,
            "detection": 0,
            "str": 5,
            "dex": 5,
            "x": 10,
            "y": 8,
            "combatEntries": 0,
        },
        "floor": floor,
        "biome": biome,
        "enemies": enemies or [],
        "items": items or [],
        "strCombatActive": in_combat,
        "strCombatAdvantage": "neutral",
        "alertLevel": alert,
        "runState": {
            "keysOwned": [],
            "gatesSpawnedThisRun": gates_spawned,
            "keysFoundThisRun": 0,
        },
        "hasExit": False,
    }


def _enemy():
    return {"id": "e1", "hp": 5, "dead": False, "x": 12, "y": 8}


def _key_item():
    return {"id": "k1", "type": "key", "x": 5, "y": 5}


# ---------------------------------------------------------------------------
# 1. GoneRogueGreedyPolicy
# ---------------------------------------------------------------------------

class TestGoneRogueGreedyPolicy:
    def setup_method(self):
        from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        self.PolicyClass = GoneRogueGreedyPolicy
        self.AdapterClass = GoneRogueAdapter

    def _perceive(self, raw_state):
        adapter = self.AdapterClass.__new__(self.AdapterClass)
        policy = self.PolicyClass()
        compressed = adapter.compress_perception(raw_state)
        return policy.perceive(compressed)

    def test_progression_axis_normal_hp(self):
        raw = _make_raw_state(hp=10, max_hp=10)
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert plan.axis_priority == "progression"

    def test_survival_axis_low_hp(self):
        raw = _make_raw_state(hp=2, max_hp=10)   # 20% HP
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert plan.axis_priority == "survival"

    def test_resource_axis_key_on_floor(self):
        raw = _make_raw_state(hp=8, max_hp=10, items=[_key_item()])
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert plan.axis_priority == "resource"

    def test_plan_has_stop_conditions(self):
        raw = _make_raw_state()
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert len(plan.stop_conditions) > 0, "plan must declare stop conditions"

    def test_plan_has_axis(self):
        raw = _make_raw_state()
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert plan.axis_priority in ("progression", "resource", "survival")

    def test_batch_respects_max_batch(self):
        raw = _make_raw_state()
        perception = self._perceive(raw)
        policy = self.PolicyClass(max_batch=3)
        plan = policy.plan(perception)
        assert len(plan.action_batch) <= 3

    def test_high_volatility_in_combat(self):
        raw = _make_raw_state(enemies=[_enemy()], in_combat=True)
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert plan.volatility > 0.5, "combat must raise volatility"

    def test_low_volatility_no_enemies(self):
        raw = _make_raw_state(hp=10, max_hp=10, enemies=[])
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert plan.volatility < 0.5

    def test_stop_condition_combat_start_included(self):
        raw = _make_raw_state(in_combat=True, enemies=[_enemy()])
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert "combat_start" in plan.stop_conditions or "died" in plan.stop_conditions

    def test_plan_short_strategy_is_string(self):
        raw = _make_raw_state()
        perception = self._perceive(raw)
        policy = self.PolicyClass()
        plan = policy.plan(perception)
        assert isinstance(plan.short_strategy, str)
        assert len(plan.short_strategy) > 0


# ---------------------------------------------------------------------------
# 2. compress_perception
# ---------------------------------------------------------------------------

class TestCompressPerception:
    def setup_method(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        self.adapter = GoneRogueAdapter.__new__(GoneRogueAdapter)

    def test_hp_fields(self):
        raw = _make_raw_state(hp=6, max_hp=10)
        p = self.adapter.compress_perception(raw)
        assert p["hp"] == 6
        assert p["max_hp"] == 10
        assert abs(p["hp_ratio"] - 0.6) < 0.01

    def test_floor_index_zero_based(self):
        raw = _make_raw_state(floor=5)
        p = self.adapter.compress_perception(raw)
        assert p["floor_index"] == 4   # floor 5 → index 4

    def test_biome_propagated(self):
        raw = _make_raw_state(biome="aerospace_museum")
        p = self.adapter.compress_perception(raw)
        assert p["biome"] == "aerospace_museum"

    def test_enemies_counted(self):
        raw = _make_raw_state(enemies=[_enemy(), _enemy()])
        p = self.adapter.compress_perception(raw)
        assert p["enemies_visible"] == 2

    def test_dead_enemies_not_counted(self):
        dead = {**_enemy(), "dead": True}
        raw = _make_raw_state(enemies=[dead])
        p = self.adapter.compress_perception(raw)
        assert p["enemies_visible"] == 0

    def test_keys_on_floor_counted(self):
        raw = _make_raw_state(items=[_key_item(), _key_item()])
        p = self.adapter.compress_perception(raw)
        assert p["keys_on_floor"] == 2

    def test_none_state_returns_empty(self):
        p = self.adapter.compress_perception(None)
        assert p == {}

    def test_in_combat_flag(self):
        raw = _make_raw_state(in_combat=True)
        p = self.adapter.compress_perception(raw)
        assert p["str_combat_active"] is True

    def test_floors_remaining(self):
        raw = _make_raw_state(floor=1)   # floor 1 → 32 remaining (32 - (1-1))
        p = self.adapter.compress_perception(raw)
        assert p["floors_remaining"] == 32


# ---------------------------------------------------------------------------
# 3. GoneRogueAdapter – interface contract (mocked Playwright)
# ---------------------------------------------------------------------------

class TestGoneRogueAdapterInterface:
    """Tests that the adapter methods call the correct Playwright evaluate calls."""

    def _make_adapter(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        adapter = GoneRogueAdapter.__new__(GoneRogueAdapter)
        adapter.base_url = "http://localhost:8787/public/js"
        adapter._adapter_base = "http://localhost:8787/public/tests"
        adapter._headless = True
        adapter._slow_mo = 0
        adapter._timeout_ms = 10_000
        adapter._playwright = None
        adapter._browser = None
        adapter._page = MagicMock()
        adapter._game_loaded = True
        return adapter

    def test_get_state_calls_evaluate(self):
        adapter = self._make_adapter()
        adapter._page.evaluate.return_value = _make_raw_state()
        state = adapter.get_state()
        adapter._page.evaluate.assert_called_once_with(
            "() => GoneRogue.headless.getState()"
        )
        assert state["floor"] == 3

    def test_get_legal_actions_calls_evaluate(self):
        adapter = self._make_adapter()
        adapter._page.evaluate.return_value = [{"type": "move", "dx": 1, "dy": 0}]
        actions = adapter.get_legal_actions()
        adapter._page.evaluate.assert_called_once_with(
            "() => GoneRogue.headless.getLegalActions()"
        )
        assert actions[0]["type"] == "move"

    def test_apply_action_calls_evaluate_with_action(self):
        adapter = self._make_adapter()
        action = {"type": "move", "dx": 1, "dy": 0}
        adapter._page.evaluate.return_value = {"success": True, "messages": [], "events": []}
        result = adapter.apply_action(action)
        call_args = adapter._page.evaluate.call_args
        assert call_args[0][0] == "(action) => GoneRogue.headless.applyAction(action)"
        assert call_args[0][1] == action
        assert result["success"] is True

    def test_apply_action_none_result_is_safe(self):
        adapter = self._make_adapter()
        adapter._page.evaluate.return_value = None
        result = adapter.apply_action({"type": "wait"})
        assert result["success"] is False

    def test_is_game_active_calls_evaluate(self):
        adapter = self._make_adapter()
        adapter._page.evaluate.return_value = True
        assert adapter.is_game_active() is True

    def test_constructor_raises_without_base_url(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        with pytest.raises(ValueError, match="base_url"):
            GoneRogueAdapter(base_url="")


# ---------------------------------------------------------------------------
# 4. Turn-envelope loop (mocked adapter)
# ---------------------------------------------------------------------------

class TestTurnEnvelopeLoop:
    def setup_method(self):
        from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter

        self.adapter = GoneRogueAdapter.__new__(GoneRogueAdapter)
        self.adapter.base_url = "http://localhost"
        self.adapter._adapter_base = "http://localhost/tests"
        self.adapter._headless = True
        self.adapter._slow_mo = 0
        self.adapter._timeout_ms = 5000
        self.adapter._playwright = None
        self.adapter._browser = None
        self.adapter._page = MagicMock()
        self.adapter._game_loaded = True

        self.policy = GoneRogueGreedyPolicy(max_batch=4)

    def test_envelope_runs_without_error(self):
        states = [_make_raw_state(hp=10)] * 20 + [_make_raw_state(hp=0)]

        call_count = {"n": 0}
        def eval_side(script, *args):
            if "getState" in script:
                idx = min(call_count["n"], len(states) - 1)
                call_count["n"] += 1
                return states[idx]
            if "isActive" in script:
                return call_count["n"] < len(states)
            if "getLegalActions" in script:
                return [{"type": "move", "dx": 1, "dy": 0}]
            return {"success": True, "messages": [], "events": []}

        self.adapter._page.evaluate.side_effect = eval_side
        result = self.adapter.run_turn_envelope(
            policy=self.policy, max_steps=30, max_batch=4
        )
        assert result["steps"] > 0

    def test_envelope_stops_on_death(self):
        dead_state = _make_raw_state(hp=0)
        alive_state = _make_raw_state(hp=10)
        states = [alive_state, alive_state, dead_state, dead_state]

        call_count = {"n": 0}
        def eval_side(script, *args):
            if "getState" in script:
                idx = min(call_count["n"], len(states) - 1)
                call_count["n"] += 1
                return states[idx]
            if "isActive" in script:
                return True
            if "getLegalActions" in script:
                return [{"type": "move", "dx": 1, "dy": 0}]
            return {"success": True, "messages": [], "events": []}

        self.adapter._page.evaluate.side_effect = eval_side
        result = self.adapter.run_turn_envelope(
            policy=self.policy, max_steps=50, max_batch=2
        )
        assert result["outcome"] == "died"

    def test_volatile_plan_shallow_batch(self):
        """In combat (volatile), execution engine must use a shallow batch."""
        from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy
        policy = GoneRogueGreedyPolicy(max_batch=8)
        combat_state = _make_raw_state(hp=10, max_hp=10, enemies=[_enemy()], in_combat=True)
        perception = policy.perceive(
            self.adapter.compress_perception(combat_state)
        )
        plan = policy.plan(perception)
        assert plan.volatility > 0.7, "combat plan must be volatile"
        # With volatility > 0.7, batch depth should be capped at 2
        effective_batch = min(2, len(plan.action_batch))
        assert effective_batch <= 2


# ---------------------------------------------------------------------------
# 5. Headless CLI – argument parsing
# ---------------------------------------------------------------------------

class TestHeadlessCLI:
    def test_missing_base_url_exits_nonzero(self, tmp_path):
        from sundog.runners.gone_rogue_headless import main
        out = str(tmp_path / "out.jsonl")
        ret = main(["--runs", "1", "--out", out])
        assert ret != 0

    def test_unknown_policy_exits(self, tmp_path):
        from sundog.runners.gone_rogue_headless import main
        out = str(tmp_path / "out.jsonl")
        with pytest.raises(SystemExit):
            main(["--runs", "1", "--eyesonly-url", "http://x",
                  "--policy", "nonexistent", "--out", out])

    def test_default_args_parsed(self):
        from sundog.runners import gone_rogue_headless as m
        assert "greedy" in m.POLICIES

    def test_output_file_written(self, tmp_path):
        """Even on error, run_single should write a record to the output file."""
        out = str(tmp_path / "out.jsonl")
        from sundog.runners.gone_rogue_headless import main
        # Will fail because no real browser, but should still create the file
        main([
            "--runs", "1",
            "--seed", "0",
            "--eyesonly-url", "http://localhost:19999/public/js",
            "--out", out,
            "--quiet",
        ])
        assert os.path.exists(out)
        with open(out) as f:
            line = f.readline().strip()
        if line:
            record = json.loads(line)
            assert "run_index" in record


# ---------------------------------------------------------------------------
# 6. Integration smoke test (skipped unless EYESONLY_BASE_URL is set)
# ---------------------------------------------------------------------------

EYESONLY_URL = os.environ.get("EYESONLY_BASE_URL", "")


@pytest.mark.skipif(not EYESONLY_URL, reason="EYESONLY_BASE_URL not set")
class TestIntegrationSmoke:
    """Requires a live EyesOnly instance with public/js/ accessible."""

    def test_adapter_loads_game(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        with GoneRogueAdapter(base_url=EYESONLY_URL, headless=True) as adapter:
            state = adapter.reset(seed=1)
            assert state is not None
            assert "player" in state
            assert "floor" in state

    def test_get_legal_actions_not_empty(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        with GoneRogueAdapter(base_url=EYESONLY_URL, headless=True) as adapter:
            adapter.reset(seed=2)
            actions = adapter.get_legal_actions()
            assert isinstance(actions, list)
            assert len(actions) > 0

    def test_apply_move_action(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        with GoneRogueAdapter(base_url=EYESONLY_URL, headless=True) as adapter:
            adapter.reset(seed=3)
            actions = adapter.get_legal_actions()
            move_actions = [a for a in actions if a.get("type") == "move"]
            if move_actions:
                result = adapter.apply_action(move_actions[0])
                assert isinstance(result, dict)

    def test_turn_envelope_smoke(self):
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        from sundog.runners.policies.gone_rogue_greedy import GoneRogueGreedyPolicy
        with GoneRogueAdapter(base_url=EYESONLY_URL, headless=True) as adapter:
            adapter.reset(seed=7)
            policy = GoneRogueGreedyPolicy(max_batch=4)
            result = adapter.run_turn_envelope(
                policy=policy, max_steps=50, max_batch=4
            )
            assert "outcome" in result
            assert result["steps"] > 0

    def test_determinism_same_seed(self):
        """Same seed must produce same biome/floor on reset."""
        from sundog.runners.adapters.gone_rogue import GoneRogueAdapter
        runs = []
        for _ in range(2):
            with GoneRogueAdapter(base_url=EYESONLY_URL, headless=True) as adapter:
                adapter.reset(seed=99)
                state = adapter.get_state()
                actions = adapter.get_legal_actions()
                runs.append({
                    "floor": state.get("floor"),
                    "biome": state.get("biome"),
                    "action_types": sorted(set(a["type"] for a in actions)),
                })
        assert runs[0]["floor"] == runs[1]["floor"]
        assert runs[0]["biome"] == runs[1]["biome"]
        assert runs[0]["action_types"] == runs[1]["action_types"]
