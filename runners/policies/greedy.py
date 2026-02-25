from __future__ import annotations
from sundog.runners.policy import AgentPolicy, PerceptionPayload, TurnPlan


class GreedyPolicy(AgentPolicy):
    """
    Greedy policy: axis = progression first, then resource if stuck, survival if low HP.
    Always tries to path toward next gate/exit.
    """
    MAX_BATCH = 8

    def plan(self, perception: PerceptionPayload) -> TurnPlan:
        # Determine axis
        if perception.hp_ratio < 0.3:
            axis = "survival"
        elif (perception.keys_on_floor > 0 and perception.gate_requires_key
              and not any(perception.keys_inventory.values())):
            axis = "resource"
        else:
            axis = "progression"

        # Determine volatility
        volatility = 0.0
        if perception.enemies_visible > 0:
            volatility = 0.9
        elif perception.gate_requires_key and not any(perception.keys_inventory.values()):
            volatility = 0.5
        elif perception.hp_ratio < 0.4:
            volatility = 0.6
        elif perception.sidequest_entrance:
            volatility = 0.7

        # Build action batch based on axis
        actions = []
        stop_conditions = ["new_enemy", "gate_encountered", "died", "floor_changed", "gate_spawned"]
        strategy = ""

        if axis == "survival":
            strategy = "Restore HP before engaging."
            actions = ["rest", "rest", "flee"] if perception.enemies_visible else ["rest", "rest", "move"]
            stop_conditions += ["low_hp", "combat_start"]
        elif axis == "resource":
            strategy = "Collect keys needed for the gate."
            actions = ["take_key"] + ["move"] * (self.MAX_BATCH - 1)
            stop_conditions += ["key_acquired"]
        else:  # progression
            has_required_key = (perception.gate_requires_key and perception.gate_key_color and
                                perception.keys_inventory.get(perception.gate_key_color, 0) > 0)
            if has_required_key:
                strategy = "Use key on gate and descend."
                actions = ["use_gate"]
                stop_conditions += ["floor_changed", "key_used"]
            elif perception.keys_on_floor > 0 and perception.gate_requires_key:
                strategy = "Pick up key then proceed to gate."
                actions = ["take_key", "use_gate"]
                stop_conditions += ["key_acquired", "floor_changed"]
            elif (perception.gate_requires_key and
                  perception.keys_inventory.get(perception.gate_key_color or "", 0) == 0):
                # No matching key, none on floor — move to accumulate steps for pity spawn
                strategy = "Move to trigger pity key spawn."
                actions = ["move"] * self.MAX_BATCH
                stop_conditions += ["pity_key_spawned", "key_acquired"]
            else:
                strategy = "Move toward exit gate and descend."
                actions = ["use_gate"] + ["move"] * (self.MAX_BATCH - 1)

        return TurnPlan(
            axis_priority=axis,
            short_strategy=strategy,
            action_batch=actions[: self.MAX_BATCH],
            stop_conditions=stop_conditions,
            volatility=volatility,
        )
