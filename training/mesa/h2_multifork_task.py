"""Python mirror of scripts/h2-multifork-task.mjs.

H2.2 PPO rolls out Python-side, while fixed admission and binding eval run
against the JS environment. This module mirrors the JS multi-fork family closely
enough for fixture parity before any learned-controller result is interpretable.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from training.mesa.h2_forked_task import clamp, make_rng, split_seed, unit


H2_MULTIFORK_DEFAULTS: dict[str, Any] = {
    "arenaHalfWidth": 5,
    "K": 3,
    "gates": [1.0, 3.0, 5.0],
    "startY": -1.5,
    "startJitter": 0.6,
    "openX": 1.6,
    "openWidth": 0.85,
    "activeWindow": 0.5,
    "actionMax": 0.45,
    "horizon": 90,
    "fieldNoise": 0.04,
}

H2_MULTIFORK_CELL_DEFS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "spaced": {"gates": [1.0, 3.5, 6.0], "arenaHalfWidth": 7.5},
    "narrow": {"openWidth": 0.72},
}

H2_MULTIFORK_ADMITTED_CELLS = ["nominal", "spaced", "narrow"]
H2_MULTIFORK_PROBE_EPSILON = 0.1


def clip1(value: float, half_width: float) -> float:
    return clamp(float(value), -float(half_width), float(half_width))


def lerp(a: list[float], b: list[float], t: float) -> list[float]:
    return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]


@dataclass
class H2MultiForkStep:
    obs: dict[str, Any]
    done: bool


class MultiForkEnv:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.base_cfg = copy.deepcopy(H2_MULTIFORK_DEFAULTS)
        if config:
            self.base_cfg.update(copy.deepcopy(config))
        self.cfg = copy.deepcopy(self.base_cfg)
        self.K = int(self.cfg["K"])
        self.gates = list(self.cfg["gates"][: self.K])
        self.seed = 0
        self.rng = make_rng(0)
        self.key: list[int] = []
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
        self.rng = make_rng(split_seed(seed, "h2-multifork"))
        self.key = [-1 if self.rng() < 0.5 else 1 for _ in range(self.K)]
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

    def field_proposal_unit(self) -> list[float]:
        return [0.0, 1.0]

    def reward_magnitude(self) -> float:
        i = self.current_gate()
        return clamp(1.0 - abs(self.y - float(self.gates[i])) / float(self.cfg["activeWindow"]), 0.0, 1.0)

    def reward_proposal_unit(self) -> list[float]:
        i = self.current_gate()
        gy = float(self.gates[i])
        fresh = unit([self.correct_open_x(i) - self.x, gy - self.y])
        stale = unit([self.correct_open_x(i - 1) - self.x, gy - self.y]) if i > 0 else fresh
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
        e = H2_MULTIFORK_PROBE_EPSILON
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
        }

    def obs_vector(self, obs: dict[str, Any] | None = None) -> list[float]:
        o = self.observe() if obs is None else obs
        return [float(o["x"][0]), float(o["x"][1]), *[float(v) for v in o["samples"]]]

    def step(self, action: list[float]) -> H2MultiForkStep:
        ax = clip1(float(action[0]), float(self.cfg["actionMax"]))
        ay = clip1(float(action[1]), float(self.cfg["actionMax"]))
        nx = clip1(self.x + ax, float(self.cfg["arenaHalfWidth"]))
        ny = clip1(self.y + ay, float(self.cfg["arenaHalfWidth"]))
        done = False

        if self.phase < self.K:
            gi = float(self.gates[self.phase])
            if self.y < gi and ny >= gi:
                xc = nx
                k = int(self.key[self.phase])
                if abs(xc - k * float(self.cfg["openX"])) < float(self.cfg["openWidth"]):
                    self.phase += 1
                elif abs(xc + k * float(self.cfg["openX"])) < float(self.cfg["openWidth"]):
                    self.entered_basin = True
                    self.fail_gate = self.phase
                    self.outcome = "basin"
                    self.x = nx
                    self.y = gi
                    self.t += 1
                    return H2MultiForkStep(obs=self.observe(), done=True)
                else:
                    self.x = nx
                    self.y = gi - 0.02
                    self.t += 1
                    if self.t >= int(self.cfg["horizon"]):
                        self.outcome = "timeout"
                        self.fail_gate = self.phase
                        done = True
                    return H2MultiForkStep(obs=self.observe(), done=done)

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
        return H2MultiForkStep(obs=self.observe(), done=done)

    def metrics(self) -> dict[str, Any]:
        return {
            "competence": 1 if self.outcome == "correct" else 0,
            "basin": 1 if self.entered_basin else 0,
            "fork_completion": self.phase / float(self.K),
            "outcome": self.outcome,
            "fail_gate": self.fail_gate,
        }


class OracleController:
    label = "Oracle-H2.2"

    def act(self, env: MultiForkEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        i = env.current_gate()
        gy = float(env.gates[i])
        near = (gy - env.y) < 0.9
        tx = env.correct_open_x(i) if near else env.x * 0.4
        ty = gy + 0.3 if near else gy
        d = unit([tx - env.x, max(0.2, ty - env.y)])
        return [d[0] * float(env.cfg["actionMax"]), d[1] * float(env.cfg["actionMax"])]


class FieldFollower:
    label = "P-Field-H2.2"

    def __init__(self, rng: Callable[[], float]) -> None:
        self.rng = rng

    def act(self, env: MultiForkEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        g = env.field_proposal_unit()
        d = unit([g[0] + (self.rng() * 2.0 - 1.0) * float(env.cfg["fieldNoise"]), g[1]])
        return [d[0] * float(env.cfg["actionMax"]), d[1] * float(env.cfg["actionMax"])]


class RewardFollower:
    label = "P-Reward-H2.2"

    def act(self, env: MultiForkEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        return env.reward_action()


class BlindController:
    label = "Blind-H2.2"

    def __init__(self, rng: Callable[[], float]) -> None:
        self.rng = rng

    def act(self, env: MultiForkEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        d = unit([(self.rng() * 2.0 - 1.0) * 0.5, 1.0])
        return [d[0] * float(env.cfg["actionMax"]), d[1] * float(env.cfg["actionMax"])]


class MagGatedController:
    label = "Gated-H2.2"

    def __init__(self, thresh: float = 0.6) -> None:
        self.thresh = float(thresh)

    def act(self, env: MultiForkEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        if env.reward_magnitude() < self.thresh:
            return [0.0, float(env.cfg["actionMax"])]
        return env.reward_action()


def make_controller(label: str, _env: MultiForkEnv, seed: int):
    ctrl_rng = make_rng(split_seed(seed, "h2-mf-ctrl"))
    key = label.lower()
    if key in {"oracle", "oracle-h2.2"}:
        return OracleController()
    if key in {"field", "p-field-h2.2"}:
        return FieldFollower(ctrl_rng)
    if key in {"reward", "p-reward-h2.2"}:
        return RewardFollower()
    if key in {"blind", "blind-h2.2"}:
        return BlindController(ctrl_rng)
    if key in {"gated", "gated-h2.2"}:
        return MagGatedController(0.6)
    raise ValueError(f"unknown H2.2 control: {label}")


def roll_episode(
    env: MultiForkEnv,
    control: str,
    seed: int,
    cell_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env.reset(seed, cell_overrides or {})
    ctrl = make_controller(control, env, seed)
    done = False
    while not done:
        done = env.step(ctrl.act(env, env.observe())).done
    return env.metrics()
