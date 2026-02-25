from __future__ import annotations
from typing import Optional, List, Dict, Any
from sundog.runners.policy import TurnPlan

STOP_CONDITIONS = {
    "new_enemy",
    "gate_encountered",
    "gate_spawned",
    "biome_change",
    "combat_start",
    "died",
    "completed",
    "key_acquired",
    "key_used",
    "low_hp",
    "sidequest_entrance",
    "floor_changed",
    "invalid_action",
    "rng_variance",
}


class ExecutionEngine:
    def __init__(self, max_batch_depth: int = 10, volatility_threshold: float = 0.7):
        self.max_batch_depth = max_batch_depth
        self.volatility_threshold = volatility_threshold

    def execute_batch(self, game_sim, plan: TurnPlan) -> dict:
        """
        Execute action_batch safely.
        Returns: {
          "actions_executed": list of executed actions,
          "stop_reason": str|None,
          "events": list of event dicts,
          "steps": int,
        }
        """
        actions_executed = []
        events = []
        stop_reason = None
        max_depth = self._compute_max_depth(plan)

        for i, action in enumerate(plan.action_batch):
            if i >= max_depth:
                stop_reason = "batch_depth_exhausted"
                break

            result = game_sim.apply_action(action)
            actions_executed.append(action)
            if result.get("event"):
                events.append({"action": action, "event": result["event"], "result": result})

            stop = self._check_stop_condition(result, plan)
            if stop:
                stop_reason = stop
                break

        return {
            "actions_executed": actions_executed,
            "stop_reason": stop_reason,
            "events": events,
            "steps": len(actions_executed),
        }

    def _compute_max_depth(self, plan: TurnPlan) -> int:
        """Dynamic depth: safe=full batch, volatile=shallow."""
        if plan.volatility > self.volatility_threshold:
            return min(2, len(plan.action_batch))
        return min(len(plan.action_batch), self.max_batch_depth)

    def _check_stop_condition(self, result: dict, plan: TurnPlan) -> Optional[str]:
        """Check if a stop condition fires based on action result."""
        event = result.get("event")
        if not event:
            return None
        # Always stop on terminal events
        if event in ("died", "completed", "invalid_action"):
            return event
        # Check plan's stop conditions
        if event in plan.stop_conditions:
            return event
        return None
