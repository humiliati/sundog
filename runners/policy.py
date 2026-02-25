from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Any, Optional

AxisPriority = Literal["progression", "resource", "survival"]


@dataclass
class PerceptionPayload:
    """Compressed perception - small and stable."""
    floor_index: int
    biome: str
    hp: int
    max_hp: int
    hp_ratio: float
    keys_inventory: Dict[str, int]
    gates_visible: int
    gate_requires_key: bool
    gate_key_color: Optional[str]
    keys_on_floor: int
    enemies_visible: int
    steps_on_floor: int
    pity_triggered: bool
    sidequest_entrance: Optional[str]
    floors_remaining: int


@dataclass
class TurnPlan:
    axis_priority: AxisPriority
    short_strategy: str
    action_batch: List[str]
    stop_conditions: List[str]
    volatility: float = 0.0
    debug: Dict[str, Any] = field(default_factory=dict)


class AgentPolicy:
    """Abstract base class for agent policies."""

    def perceive(self, game_state) -> PerceptionPayload:
        """Compress game state into a small perception payload."""
        s = game_state
        fs = s.floor_state
        gate_requires_key = False
        gate_key_color = None
        if fs.gates_visible:
            # Prefer an unlocked gate or a gate whose key we already have
            chosen = fs.gates_visible[0]
            for g in fs.gates_visible:
                if not g["locked"] or not g["key_color"]:
                    chosen = g
                    break
                if s.keys_inventory.get(g["key_color"], 0) > 0:
                    chosen = g
                    break
            if chosen["locked"] and chosen["key_color"]:
                gate_requires_key = True
                gate_key_color = chosen["key_color"]
        return PerceptionPayload(
            floor_index=s.floor_index,
            biome=s.biome,
            hp=s.hp,
            max_hp=s.max_hp,
            hp_ratio=s.hp / s.max_hp if s.max_hp > 0 else 0.0,
            keys_inventory=dict(s.keys_inventory),
            gates_visible=len(fs.gates_visible),
            gate_requires_key=gate_requires_key,
            gate_key_color=gate_key_color,
            keys_on_floor=len(fs.keys_on_floor),
            enemies_visible=len(fs.enemies),
            steps_on_floor=s.steps_on_floor,
            pity_triggered=fs.pity_triggered,
            sidequest_entrance=fs.sidequest_entrance,
            floors_remaining=32 - s.floor_index,
        )

    def plan(self, perception: PerceptionPayload) -> TurnPlan:
        """Produce axis/utility framing and action batch."""
        raise NotImplementedError

    def turn_envelope(self, game_state) -> TurnPlan:
        """Execute the full PERCEIVE → PLAN turn envelope."""
        perception = self.perceive(game_state)
        return self.plan(perception)
