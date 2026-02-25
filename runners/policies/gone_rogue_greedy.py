"""
sundog.runners.policies.gone_rogue_greedy
==========================================
A simple but intentional greedy policy for the EyesOnly *Gone Rogue* game.

Axes
----
- **progression** – head toward exit, use cards offensively, explore map.
- **resource**    – collect currency/items, stay out of sight, build deck.
- **survival**    – flee or rest when HP is critically low.

Stop conditions
---------------
The policy sets conservative stop conditions for volatile states (combat,
low HP, new enemy detected) so the execution engine replan frequently.
"""
from __future__ import annotations

from typing import Any, Dict, List

from sundog.runners.policy import AgentPolicy, PerceptionPayload, TurnPlan

# Maximum batch depth for this policy
MAX_BATCH = 8

# HP ratio thresholds
SURVIVAL_HP = 0.25   # below this → survival axis
CAUTION_HP  = 0.45   # below this → elevated volatility


class GoneRogueGreedyPolicy(AgentPolicy):
    """
    Greedy policy adapted for Gone Rogue's actual game actions.

    ``perceive()`` is overridden to accept the raw compressed dict returned by
    ``GoneRogueAdapter.compress_perception()``.  ``plan()`` produces a
    ``TurnPlan`` using GoneRogue action primitives.
    """

    def __init__(self, max_batch: int = MAX_BATCH):
        self.max_batch = max_batch

    # ------------------------------------------------------------------
    # PERCEIVE – map GoneRogue compressed state → PerceptionPayload
    # ------------------------------------------------------------------

    def perceive(self, game_state: Any) -> PerceptionPayload:
        """
        Accept either a raw ``GoneRogueAdapter.compress_perception()`` dict or
        the standard sundog ``GameState`` dataclass.
        """
        # GoneRogueAdapter.compress_perception returns a plain dict
        if isinstance(game_state, dict):
            return self._perceive_from_dict(game_state)
        # Fall back to base-class behaviour (sundog GameState dataclass)
        return super().perceive(game_state)

    def _perceive_from_dict(self, d: Dict[str, Any]) -> PerceptionPayload:
        hp      = d.get("hp", 10)
        max_hp  = d.get("max_hp", 10)
        return PerceptionPayload(
            floor_index      = d.get("floor_index", 0),
            biome            = d.get("biome", "unknown"),
            hp               = hp,
            max_hp           = max_hp,
            hp_ratio         = d.get("hp_ratio", hp / max_hp if max_hp else 0.0),
            keys_inventory   = d.get("keys_inventory", {}),
            gates_visible    = d.get("gates_visible", 0),
            gate_requires_key= d.get("gate_requires_key", False),
            gate_key_color   = d.get("gate_key_color"),
            keys_on_floor    = d.get("keys_on_floor", 0),
            enemies_visible  = d.get("enemies_visible", 0),
            steps_on_floor   = d.get("steps_on_floor", 0),
            pity_triggered   = d.get("pity_triggered", False),
            sidequest_entrance=d.get("sidequest_entrance"),
            floors_remaining = d.get("floors_remaining", 32),
        )

    # ------------------------------------------------------------------
    # PLAN – produce TurnPlan with GoneRogue action primitives
    # ------------------------------------------------------------------

    def plan(self, perception: PerceptionPayload) -> TurnPlan:
        hp_ratio    = perception.hp_ratio
        enemies     = perception.enemies_visible
        in_combat   = getattr(perception, "in_combat", False) or enemies > 0

        # ── Axis selection ─────────────────────────────────────────────
        if hp_ratio < SURVIVAL_HP:
            axis = "survival"
        elif perception.keys_on_floor > 0 and not in_combat:
            axis = "resource"
        else:
            axis = "progression"

        # ── Volatility ────────────────────────────────────────────────
        volatility = 0.0
        if in_combat:
            volatility = 0.9
        elif hp_ratio < CAUTION_HP:
            volatility = 0.65
        elif perception.sidequest_entrance:
            volatility = 0.7
        elif enemies > 0:
            volatility = 0.8

        # ── Base stop conditions ──────────────────────────────────────
        stop_conditions: List[str] = [
            "new_enemy",
            "floor_changed",
            "died",
            "combat_start",
            "gate_encountered",
        ]

        # ── Build action batch ────────────────────────────────────────
        actions: List[Dict[str, Any]]
        strategy: str

        if axis == "survival":
            strategy = "HP critical — flee combat and conserve energy."
            if in_combat:
                actions = [
                    {"type": "flee"},
                    {"type": "wait"},
                ]
            else:
                # Try to use a heal card then move cautiously
                actions = [
                    {"type": "useCard", "cardIndex": 0},  # hope for a heal card
                    {"type": "move", "dx": 0, "dy": -1},
                    {"type": "wait"},
                ]
            stop_conditions += ["low_hp", "key_acquired"]

        elif axis == "resource":
            strategy = "Collect items and currency before pushing forward."
            actions = [
                {"type": "pickup"},
                {"type": "pickupCurrency"},
                {"type": "move", "dx": 1, "dy": 0},
                {"type": "move", "dx": 0, "dy": 1},
                {"type": "move", "dx": -1, "dy": 0},
                {"type": "move", "dx": 0, "dy": -1},
                {"type": "pickup"},
                {"type": "pickupCurrency"},
            ]
            stop_conditions += ["key_acquired", "new_enemy", "combat_start"]

        else:  # progression
            if in_combat:
                strategy = "In STR combat — play attack card and manage advantage."
                actions = [
                    {"type": "useCard", "cardIndex": 0},
                    {"type": "useCard", "cardIndex": 1},
                    {"type": "flee"},
                ]
                stop_conditions += ["low_hp"]
            elif perception.gates_visible:
                strategy = "Exit visible — move toward gate and descend."
                actions = [
                    {"type": "exit"},
                    {"type": "move", "dx": 1, "dy": 0},
                    {"type": "move", "dx": 0, "dy": 1},
                    {"type": "exit"},
                ]
                stop_conditions += ["floor_changed", "gate_encountered"]
            else:
                strategy = "Explore floor — move to reveal map and locate exit."
                actions = self._exploration_batch()
                stop_conditions += ["new_enemy", "gate_encountered"]

        # Trim to max_batch
        action_batch = actions[: self.max_batch]

        return TurnPlan(
            axis_priority   = axis,
            short_strategy  = strategy,
            action_batch    = action_batch,
            stop_conditions = stop_conditions,
            volatility      = volatility,
            debug           = {
                "hp_ratio":      hp_ratio,
                "enemies":       enemies,
                "in_combat":     in_combat,
                "gates_visible": perception.gates_visible,
            },
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _exploration_batch(self) -> List[Dict[str, Any]]:
        """
        A short clockwise exploration sweep pattern.
        The game's BFS pathfinding will resolve actual walkable targets.
        """
        return [
            {"type": "move", "dx": 1,  "dy": 0},
            {"type": "move", "dx": 1,  "dy": 0},
            {"type": "move", "dx": 0,  "dy": 1},
            {"type": "move", "dx": 0,  "dy": 1},
            {"type": "move", "dx": -1, "dy": 0},
            {"type": "move", "dx": 0,  "dy": -1},
            {"type": "wait"},
            {"type": "pickup"},
        ]
