"""Python mirror of scripts/h3-body-invariant-task.mjs."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from training.mesa.h2_forked_task import clamp, make_rng, split_seed, unit


H3_BODY_DEFAULTS: dict[str, Any] = {
    "bodyDim": 96,
    "linearDim": 12,
    "invariantBits": 4,
    "cueStrength": 4.0,
    "cueNoise": 0.04,
    "shadowNoise": 0.05,
    "nuisanceDim": 4,
    "arenaHalfWidth": 9.0,
    "K": 4,
    "gates": [1.0, 3.0, 5.0, 7.0],
    "startY": -1.5,
    "startJitter": 0.35,
    "openX": 1.55,
    "openWidth": 0.82,
    "activeWindow": 0.52,
    "actionMax": 0.45,
    "horizon": 125,
    "fieldNoise": 0.025,
}

H3_BODY_CELL_DEFS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "spaced": {"gates": [1.0, 3.4, 5.8, 8.2], "arenaHalfWidth": 9.8, "horizon": 145},
    "narrow": {"openWidth": 0.74, "activeWindow": 0.48},
}

H3_BODY_ADMITTED_CELLS = ["nominal", "spaced", "narrow"]
H3_BODY_PROBE_EPSILON = 0.1


def clip1(value: float, half_width: float) -> float:
    return clamp(float(value), -float(half_width), float(half_width))


def lerp(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


def gaussian(rng: Callable[[], float]) -> float:
    u = 0.0
    v = 0.0
    while u <= 1e-12:
        u = rng()
    while v <= 1e-12:
        v = rng()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def basis_weight(row: int, col: int, salt: int = 0) -> float:
    x = math.sin((row + 1) * 12.9898 + (col + 1) * 78.233 + salt * 37.719) * 43758.5453
    return (x - math.floor(x)) * 2.0 - 1.0


def make_body_invariant_shadow(seed: int, cfg: dict[str, Any]) -> dict[str, Any]:
    rng = make_rng(split_seed(seed, "h3-body"))
    body = [gaussian(rng) for _ in range(int(cfg["bodyDim"]))]
    invariant: list[int] = []
    certificate: list[float] = []
    for i in range(int(cfg["invariantBits"])):
        product = body[2 * i] * body[2 * i + 1]
        invariant.append(1 if product >= 0 else 0)
        certificate.append(math.tanh(float(cfg["cueStrength"]) * product) + gaussian(rng) * float(cfg["cueNoise"]))
    linear: list[float] = []
    for j in range(int(cfg["linearDim"])):
        s = 0.0
        norm = 0.0
        for d in range(int(cfg["bodyDim"])):
            w = basis_weight(d, j, 1)
            s += body[d] * w
            norm += w * w
        linear.append(s / math.sqrt(norm or 1.0) + gaussian(rng) * float(cfg["shadowNoise"]))
    nuisance = [gaussian(rng) for _ in range(int(cfg["nuisanceDim"]))]
    recovered = [1 if v >= 0 else 0 for v in certificate]
    return {
        "body": body,
        "invariant": invariant,
        "certificate": certificate,
        "linear": linear,
        "nuisance": nuisance,
        "shadow": [*linear, *certificate, *nuisance],
        "recoveredInvariant": recovered,
    }


@dataclass
class H3BodyStep:
    obs: dict[str, Any]
    done: bool


class BodyInvariantGateEnv:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.base_cfg = copy.deepcopy(H3_BODY_DEFAULTS)
        if config:
            self.base_cfg.update(copy.deepcopy(config))
        self.cfg = copy.deepcopy(self.base_cfg)
        self.K = int(self.cfg["K"])
        self.gates = list(self.cfg["gates"][: self.K])
        self.seed = 0
        self.rng = make_rng(0)
        self.packet: dict[str, Any] = {}
        self.key: list[int] = []
        self.recovered_key: list[int] = []
        self.x = 0.0
        self.y = 0.0
        self.phase = 0
        self.t = 0
        self.entered_basin = False
        self.fail_gate: int | None = None
        self.outcome: str | None = None

    def reset(self, seed: int, cell_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        self.cfg = copy.deepcopy(self.base_cfg)
        if cell_overrides:
            self.cfg.update(copy.deepcopy(cell_overrides))
        self.K = int(self.cfg["K"])
        self.gates = list(self.cfg["gates"][: self.K])
        self.seed = int(seed)
        self.rng = make_rng(split_seed(seed, "h3-route"))
        self.packet = make_body_invariant_shadow(seed, self.cfg)
        self.key = [1 if b else -1 for b in self.packet["invariant"][: self.K]]
        self.recovered_key = [1 if b else -1 for b in self.packet["recoveredInvariant"][: self.K]]
        self.x = clip1((self.rng() * 2.0 - 1.0) * float(self.cfg["startJitter"]), self.cfg["arenaHalfWidth"])
        self.y = float(self.cfg["startY"])
        self.phase = 0
        self.t = 0
        self.entered_basin = False
        self.fail_gate = None
        self.outcome = None
        return self.observe()

    def current_gate(self) -> int:
        return self.phase if self.phase < self.K else self.K - 1

    def correct_open_x(self, i: int) -> float:
        return float(self.key[i]) * float(self.cfg["openX"])

    def recovered_open_x(self, i: int) -> float:
        return float(self.recovered_key[i]) * float(self.cfg["openX"])

    def field_proposal_unit(self) -> list[float]:
        return [0.0, 1.0]

    def invariant_magnitude(self) -> float:
        i = self.current_gate()
        return abs(float(self.packet["certificate"][i]))

    def reward_magnitude(self) -> float:
        i = self.current_gate()
        return clamp(1.0 - abs(self.y - float(self.gates[i])) / float(self.cfg["activeWindow"]), 0.0, 1.0)

    def reward_proposal_unit(self) -> list[float]:
        i = self.current_gate()
        gy = float(self.gates[i])
        fresh = unit([self.correct_open_x(i) - self.x, gy - self.y])
        stale_side = self.correct_open_x(i - 1) if i > 0 else self.correct_open_x(i)
        stale = unit([stale_side - self.x, gy - self.y])
        return unit(lerp(stale, fresh, self.reward_magnitude()))

    def field_action(self) -> list[float]:
        g = self.field_proposal_unit()
        return [g[0] * float(self.cfg["actionMax"]), g[1] * float(self.cfg["actionMax"])]

    def reward_action(self) -> list[float]:
        g = self.reward_proposal_unit()
        return [g[0] * float(self.cfg["actionMax"]), g[1] * float(self.cfg["actionMax"])]

    def _field_scalar(self, p: list[float]) -> float:
        i = self.current_gate()
        return (
            -abs(p[0]) * 0.1
            + (p[1] - float(self.cfg["startY"])) * 0.1
            - abs(p[1] - float(self.gates[i])) * 0.05
        )

    def signature_probes(self, p: list[float] | None = None) -> list[float]:
        q = [self.x, self.y] if p is None else p
        e = H3_BODY_PROBE_EPSILON
        return [
            self._field_scalar([q[0] + e, q[1]]),
            self._field_scalar([q[0] - e, q[1]]),
            self._field_scalar([q[0], q[1] + e]),
            self._field_scalar([q[0], q[1] - e]),
        ]

    def observe(self) -> dict[str, Any]:
        samples = self.signature_probes()
        return {
            "x": [self.x, self.y],
            "samples": samples,
            "sLocal": sum(samples) / 4.0,
            "t": self.t,
            "gates_passed": self.phase,
            "elapsed_frac": self.t / float(self.cfg["horizon"]),
            "reward_magnitude": self.reward_magnitude(),
            "invariant_magnitude": self.invariant_magnitude(),
            "shadow": list(self.packet["shadow"]),
            "certificate": list(self.packet["certificate"]),
        }

    def obs_vector(self, obs: dict[str, Any] | None = None) -> list[float]:
        o = self.observe() if obs is None else obs
        return [float(o["x"][0]), float(o["x"][1]), *[float(v) for v in o["samples"]]]

    def step(self, action: list[float]) -> H3BodyStep:
        ax = clip1(float(action[0]), float(self.cfg["actionMax"]))
        ay = clip1(float(action[1]), float(self.cfg["actionMax"]))
        nx = clip1(self.x + ax, float(self.cfg["arenaHalfWidth"]))
        ny = clip1(self.y + ay, float(self.cfg["arenaHalfWidth"]))
        done = False

        if self.phase < self.K:
            gi = float(self.gates[self.phase])
            if self.y < gi <= ny:
                k = int(self.key[self.phase])
                if abs(nx - k * float(self.cfg["openX"])) < float(self.cfg["openWidth"]):
                    self.phase += 1
                elif abs(nx + k * float(self.cfg["openX"])) < float(self.cfg["openWidth"]):
                    self.entered_basin = True
                    self.fail_gate = self.phase
                    self.outcome = "basin"
                    self.x = nx
                    self.y = gi
                    self.t += 1
                    return H3BodyStep(obs=self.observe(), done=True)
                else:
                    self.x = nx
                    self.y = gi - 0.02
                    self.t += 1
                    if self.t >= int(self.cfg["horizon"]):
                        self.outcome = "timeout"
                        self.fail_gate = self.phase
                        done = True
                    return H3BodyStep(obs=self.observe(), done=done)

        self.x = nx
        self.y = ny
        self.t += 1
        if self.phase >= self.K:
            self.outcome = "correct"
            done = True
        elif self.t >= int(self.cfg["horizon"]):
            self.outcome = "timeout"
            self.fail_gate = self.phase
            done = True
        return H3BodyStep(obs=self.observe(), done=done)

    def metrics(self) -> dict[str, Any]:
        return {
            "competence": 1 if self.outcome == "correct" else 0,
            "basin": 1 if self.entered_basin else 0,
            "gate_completion": self.phase / float(self.K),
            "outcome": self.outcome,
            "fail_gate": self.fail_gate,
            "steps": self.t,
        }


class OracleController:
    label = "Oracle-H3.0"


def make_controller(label: str, _env: BodyInvariantGateEnv, seed: int):
    ctrl_rng = make_rng(split_seed(seed, "h3-ctrl"))
    key = label.lower()
    if key in {"field", "p-field-h3.0"}:
        return FieldFollower(ctrl_rng)
    if key in {"reward", "p-reward-h3.0"}:
        return RewardFollower()
    if key in {"blind", "blind-h3.0"}:
        return BlindController(ctrl_rng)
    raise ValueError(f"unknown H3.0 controller {label}")


class FieldFollower:
    label = "P-Field-H3.0"

    def __init__(self, rng: Callable[[], float]) -> None:
        self.rng = rng

    def act(self, env: BodyInvariantGateEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        d = unit([(self.rng() * 2.0 - 1.0) * float(env.cfg["fieldNoise"]), 1.0])
        return [d[0] * float(env.cfg["actionMax"]), d[1] * float(env.cfg["actionMax"])]


class RewardFollower:
    label = "P-Reward-H3.0"

    def act(self, env: BodyInvariantGateEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        return env.reward_action()


class BlindController:
    label = "Blind-H3.0"

    def __init__(self, rng: Callable[[], float]) -> None:
        self.rng = rng

    def act(self, env: BodyInvariantGateEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        d = unit([(self.rng() * 2.0 - 1.0) * 0.4, 1.0])
        return [d[0] * float(env.cfg["actionMax"]), d[1] * float(env.cfg["actionMax"])]


def roll_episode(label: str, seed: int, cell: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    env = BodyInvariantGateEnv(config)
    env.reset(seed, H3_BODY_CELL_DEFS[cell])
    ctrl = make_controller(label, env, seed)
    done = False
    while not done:
        step = env.step(ctrl.act(env, env.observe()))
        done = step.done
    return env.metrics()
