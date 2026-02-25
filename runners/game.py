"""
Self-contained 32-floor turn-based game simulation (no MuJoCo dependency).
"""
from __future__ import annotations
import random
import uuid
from dataclasses import dataclass, field
from typing import Optional

BIOMES = ["forest", "mall", "office", "industrial", "aerospace_museum"]
SIDEQUEST_BIOMES = ["church_catacombs", "tavern_basement"]
KEY_COLORS = ["red", "blue", "green", "yellow"]
PITY_THRESHOLD = 15
MAX_FLOORS = 32

@dataclass
class FloorState:
    floor_index: int
    biome: str
    gates_visible: list
    enemies: list
    keys_on_floor: list
    pity_triggered: bool = False
    sidequest_entrance: Optional[str] = None

@dataclass
class GameState:
    seed: int
    run_id: str
    floor_index: int
    biome: str
    hp: int
    max_hp: int
    keys_inventory: dict
    score: int
    steps_on_floor: int
    total_steps: int
    floors_completed: int
    outcome: str   # "ongoing", "completed", "died"
    rng: object    # random.Random instance, not serialized
    floor_state: FloorState

class GameSimulation:
    def __init__(self, seed: int, run_id: Optional[str] = None):
        self.seed = seed
        self.run_id = run_id or str(uuid.uuid4())
        self._state: Optional[GameState] = None

    def reset(self) -> GameState:
        rng = random.Random(self.seed)
        floor_state = self._generate_floor_with_rng(0, rng)
        self._state = GameState(
            seed=self.seed,
            run_id=self.run_id,
            floor_index=0,
            biome=floor_state.biome,
            hp=100,
            max_hp=100,
            keys_inventory={c: 0 for c in KEY_COLORS},
            score=0,
            steps_on_floor=0,
            total_steps=0,
            floors_completed=0,
            outcome="ongoing",
            rng=rng,
            floor_state=floor_state,
        )
        return self._state

    def get_current_floor_state(self) -> FloorState:
        return self._state.floor_state

    def apply_action(self, action: str) -> dict:
        s = self._state
        result = {"ok": False, "event": None, "state_changed": False, "new_floor": False, "died": False, "completed": False}

        if s.outcome != "ongoing":
            result["event"] = "run_ended"
            return result

        if action == "move":
            s.steps_on_floor += 1
            s.total_steps += 1
            s.score += 1
            result["ok"] = True
            result["state_changed"] = True
            # Random enemy encounter
            if s.floor_state.enemies and s.rng.random() < 0.3:
                result["event"] = "combat_start"
            elif s.rng.random() < 0.1 and len(s.floor_state.gates_visible) == 0:
                # Spawn a gate
                gate = {"gate_id": f"gate_{s.floor_index}_{s.steps_on_floor}", "locked": s.rng.random() < 0.5, "key_color": s.rng.choice(KEY_COLORS) if s.rng.random() < 0.5 else None, "position": (s.rng.randint(0,9), s.rng.randint(0,9))}
                s.floor_state.gates_visible.append(gate)
                result["event"] = "gate_spawned"
            else:
                # Check pity
                if self._check_pity():
                    color = s.rng.choice(KEY_COLORS)
                    key = {"key_id": f"pity_key_{s.total_steps}", "color": color, "position": (s.rng.randint(0,9), s.rng.randint(0,9))}
                    s.floor_state.keys_on_floor.append(key)
                    s.floor_state.pity_triggered = True
                    result["event"] = "pity_key_spawned"

        elif action == "take_key":
            if s.floor_state.keys_on_floor:
                key = s.floor_state.keys_on_floor.pop(0)
                s.keys_inventory[key["color"]] = s.keys_inventory.get(key["color"], 0) + 1
                result["ok"] = True
                result["state_changed"] = True
                result["event"] = "key_acquired"
            else:
                result["event"] = "no_key_available"

        elif action in ("use_gate", "descend"):
            if s.floor_state.gates_visible:
                # Find best usable gate: unlocked first, then locked with matching key
                gate = None
                for g in s.floor_state.gates_visible:
                    if not g["locked"] or not g["key_color"]:
                        gate = g
                        break
                    if s.keys_inventory.get(g["key_color"], 0) > 0:
                        gate = g
                        break
                if gate is None:
                    gate = s.floor_state.gates_visible[0]
                if gate["locked"] and gate["key_color"]:
                    color = gate["key_color"]
                    if s.keys_inventory.get(color, 0) > 0:
                        s.keys_inventory[color] -= 1
                        result["event"] = "key_used"
                    else:
                        result["event"] = "gate_encountered"
                        result["ok"] = False
                        return result
                # Advance floor
                s.floors_completed += 1
                s.floor_index += 1
                s.steps_on_floor = 0
                result["ok"] = True
                result["state_changed"] = True
                result["new_floor"] = True
                if s.floor_index >= MAX_FLOORS:
                    s.outcome = "completed"
                    result["completed"] = True
                    result["event"] = "completed"
                else:
                    new_floor = self._generate_floor_with_rng(s.floor_index, s.rng)
                    s.floor_state = new_floor
                    s.biome = new_floor.biome
                    result["event"] = "floor_changed"
            else:
                result["event"] = "no_gate_visible"

        elif action == "attack":
            if s.floor_state.enemies:
                enemy = s.floor_state.enemies[0]
                dmg_dealt = s.rng.randint(10, 25)
                dmg_taken = s.rng.randint(0, 15)
                enemy["hp"] -= dmg_dealt
                s.hp -= dmg_taken
                result["ok"] = True
                result["state_changed"] = True
                if enemy["hp"] <= 0:
                    s.floor_state.enemies.pop(0)
                    s.score += 10
                    result["event"] = "enemy_defeated"
                else:
                    result["event"] = "combat_ongoing"
                if s.hp <= 0:
                    s.hp = 0
                    s.outcome = "died"
                    result["died"] = True
                    result["event"] = "died"
                elif dmg_taken > 0 and s.hp / s.max_hp < 0.3:
                    result["event"] = "low_hp"
            else:
                result["event"] = "no_enemy"

        elif action == "flee":
            s.hp = max(0, s.hp - s.rng.randint(0, 5))
            result["ok"] = True
            result["state_changed"] = True
            result["event"] = "fled"
            if s.hp <= 0:
                s.outcome = "died"
                result["died"] = True
                result["event"] = "died"

        elif action == "rest":
            restore = s.rng.randint(5, 15)
            s.hp = min(s.max_hp, s.hp + restore)
            result["ok"] = True
            result["state_changed"] = True
            result["event"] = "rested"

        else:
            result["event"] = "invalid_action"

        return result

    def _generate_floor(self, floor_index: int) -> FloorState:
        rng = random.Random(self.seed + floor_index * 1000)
        return self._generate_floor_with_rng(floor_index, rng)

    def _generate_floor_with_rng(self, floor_index: int, rng: random.Random) -> FloorState:
        biome_index = floor_index % len(BIOMES)
        biome = BIOMES[biome_index]

        # Sidequest entrance (rare)
        sidequest = None
        if rng.random() < 0.1:
            sidequest = rng.choice(SIDEQUEST_BIOMES)

        # Generate gates (1-2)
        num_gates = rng.randint(1, 2)
        gates = []
        for i in range(num_gates):
            locked = rng.random() < 0.6
            key_color = rng.choice(KEY_COLORS) if locked else None
            gates.append({
                "gate_id": f"gate_f{floor_index}_g{i}",
                "locked": locked,
                "key_color": key_color,
                "position": (rng.randint(0, 9), rng.randint(0, 9)),
            })

        # Generate keys (0-2)
        num_keys = rng.randint(0, 2)
        keys = []
        for i in range(num_keys):
            color = rng.choice(KEY_COLORS)
            keys.append({
                "key_id": f"key_f{floor_index}_k{i}",
                "color": color,
                "position": (rng.randint(0, 9), rng.randint(0, 9)),
            })

        # Generate enemies (more in later floors)
        num_enemies = rng.randint(0, min(1 + floor_index // 8, 4))
        enemies = []
        for i in range(num_enemies):
            base_hp = 20 + floor_index * 3
            enemies.append({
                "enemy_id": f"enemy_f{floor_index}_e{i}",
                "hp": base_hp + rng.randint(-5, 5),
                "position": (rng.randint(0, 9), rng.randint(0, 9)),
            })

        return FloorState(
            floor_index=floor_index,
            biome=biome,
            gates_visible=gates,
            enemies=enemies,
            keys_on_floor=keys,
            pity_triggered=False,
            sidequest_entrance=sidequest,
        )

    def _check_pity(self) -> bool:
        s = self._state
        # Trigger pity if player is stuck (no keys but gate requires one)
        if s.steps_on_floor < PITY_THRESHOLD:
            return False
        if not s.floor_state.gates_visible:
            return False
        gate = s.floor_state.gates_visible[0]
        if not gate["locked"] or not gate["key_color"]:
            return False
        color = gate["key_color"]
        if s.keys_inventory.get(color, 0) > 0:
            return False
        if s.floor_state.keys_on_floor:
            return False
        # Pity: spawn a key if steps_on_floor is a multiple of PITY_THRESHOLD to avoid spam
        return s.steps_on_floor % PITY_THRESHOLD == 0
