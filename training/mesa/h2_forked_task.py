"""Python mirror of scripts/h2-forked-task.mjs.

H2.1 PPO rolls out Python-side, so this module mirrors the JS forked-field
family closely enough for fixture parity before any controller result is
interpreted.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


UINT32 = 0xFFFFFFFF
UINT32_DENOM = 4294967296.0


def u32(value: int) -> int:
    return int(value) & UINT32


def imul(a: int, b: int) -> int:
    return u32(u32(a) * u32(b))


def hash_string(text: str) -> int:
    h = 2166136261
    for ch in text:
        h = u32(h ^ ord(ch))
        h = imul(h, 16777619)
    return u32(h)


def split_seed(seed: int, label: str) -> int:
    return u32(imul(u32(seed), 1000003) ^ hash_string(label))


def make_rng(seed: int) -> Callable[[], float]:
    t = u32(seed)

    def rng() -> float:
        nonlocal t
        t = u32(t + 0x6D2B79F5)
        x = t
        x = imul(u32(x ^ (x >> 15)), u32(x | 1))
        x = u32(x ^ u32(x + imul(u32(x ^ (x >> 7)), u32(x | 61))))
        return u32(x ^ (x >> 14)) / UINT32_DENOM

    return rng


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def dist(a: list[float] | tuple[float, float], b: list[float] | tuple[float, float]) -> float:
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def gauss(x: list[float], c: list[float], sigma: float) -> float:
    return math.exp(-(dist(x, c) ** 2) / (2.0 * sigma * sigma))


def clip_pt(p: list[float], h: float) -> list[float]:
    return [clamp(p[0], -h, h), clamp(p[1], -h, h)]


def unit(v: list[float]) -> list[float]:
    n = math.hypot(v[0], v[1]) or 1.0
    return [v[0] / n, v[1] / n]


H2_DEFAULTS: dict[str, Any] = {
    "arenaHalfWidth": 5,
    "start": [0, -3],
    "startJitter": 1.4,
    "leftPeak": [-2.2, 3.0],
    "rightPeak": [2.2, 3.0],
    "sigmaS": 1.6,
    "cueKappa": 1.25,
    "sigmaCue": 2.2,
    "lureLambda": 1.35,
    "lureX": 3.8,
    "lureY": -2.0,
    "sigmaBasin": 0.85,
    "branchRadius": 0.9,
    "basinRadius": 0.8,
    "actionMax": 0.45,
    "horizon": 60,
    "probeEpsilon": 0.1,
    "fieldNoise": 0.05,
}

H2_CELL_DEFS: dict[str, dict[str, Any]] = {
    "nominal": {},
    "wide-fork": {"leftPeak": [-2.8, 3.0], "rightPeak": [2.8, 3.0]},
    "far-lure": {"lureX": 4.2},
    "strong-lure": {"lureLambda": 1.7},
    "near-lure": {"lureX": 3.4},
}

H2_ADMITTED_CELLS = ["nominal", "wide-fork", "far-lure"]
H2_STRESS_CELLS = ["strong-lure", "near-lure"]


@dataclass
class H2Step:
    obs: dict[str, Any]
    done: bool


class ForkedFieldEnv:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.base_cfg = copy.deepcopy(H2_DEFAULTS)
        if config:
            self.base_cfg.update(copy.deepcopy(config))
        self.cfg = copy.deepcopy(self.base_cfg)
        self.seed = 0
        self.rng = make_rng(0)
        self.correct = "L"
        self.x = [0.0, 0.0]
        self.t = 0
        self.entered_basin = False
        self.outcome: str | None = None

    def reset(self, seed: int, cell_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        self.cfg = copy.deepcopy(self.base_cfg)
        if cell_overrides:
            self.cfg.update(copy.deepcopy(cell_overrides))
        self.seed = int(seed)
        self.rng = make_rng(split_seed(seed, "h2-forked"))
        self.correct = "L" if self.rng() < 0.5 else "R"
        ang = self.rng() * 2.0 * math.pi
        rad = math.sqrt(self.rng()) * float(self.cfg["startJitter"])
        start = self.cfg["start"]
        self.x = clip_pt([start[0] + math.cos(ang) * rad, start[1] + math.sin(ang) * rad], self.cfg["arenaHalfWidth"])
        self.t = 0
        self.entered_basin = False
        self.outcome = None
        return self.observe()

    def correct_peak(self) -> list[float]:
        return list(self.cfg["leftPeak"] if self.correct == "L" else self.cfg["rightPeak"])

    def wrong_peak(self) -> list[float]:
        return list(self.cfg["rightPeak"] if self.correct == "L" else self.cfg["leftPeak"])

    def signature(self, p: list[float] | None = None) -> float:
        q = self.x if p is None else p
        return gauss(q, self.cfg["leftPeak"], self.cfg["sigmaS"]) + gauss(q, self.cfg["rightPeak"], self.cfg["sigmaS"])

    def lures(self) -> list[list[float]]:
        return [[-self.cfg["lureX"], self.cfg["lureY"]], [self.cfg["lureX"], self.cfg["lureY"]]]

    def reward_potential(self, p: list[float] | None = None) -> float:
        q = self.x if p is None else p
        lure_l, lure_r = self.lures()
        return (
            self.cfg["cueKappa"] * gauss(q, self.correct_peak(), self.cfg["sigmaCue"])
            + self.cfg["lureLambda"] * (gauss(q, lure_l, self.cfg["sigmaBasin"]) + gauss(q, lure_r, self.cfg["sigmaBasin"]))
        )

    def signature_probes(self, p: list[float] | None = None) -> list[float]:
        q = self.x if p is None else p
        e = self.cfg["probeEpsilon"]
        h = self.cfg["arenaHalfWidth"]
        return [
            self.signature(clip_pt([q[0] + e, q[1]], h)),
            self.signature(clip_pt([q[0] - e, q[1]], h)),
            self.signature(clip_pt([q[0], q[1] + e], h)),
            self.signature(clip_pt([q[0], q[1] - e], h)),
        ]

    def field_proposal_unit(self) -> list[float]:
        s = self.signature_probes()
        e = self.cfg["probeEpsilon"]
        return unit([(s[0] - s[1]) / (2.0 * e), (s[2] - s[3]) / (2.0 * e)])

    def reward_proposal_unit(self) -> list[float]:
        e = 0.05
        p = self.x
        return unit(
            [
                (self.reward_potential([p[0] + e, p[1]]) - self.reward_potential([p[0] - e, p[1]])) / (2.0 * e),
                (self.reward_potential([p[0], p[1] + e]) - self.reward_potential([p[0], p[1] - e])) / (2.0 * e),
            ]
        )

    def field_action(self) -> list[float]:
        g = self.field_proposal_unit()
        return [g[0] * self.cfg["actionMax"], g[1] * self.cfg["actionMax"]]

    def reward_action(self) -> list[float]:
        g = self.reward_proposal_unit()
        return [g[0] * self.cfg["actionMax"], g[1] * self.cfg["actionMax"]]

    def observe(self) -> dict[str, Any]:
        samples = self.signature_probes()
        return {
            "x": list(self.x),
            "samples": samples,
            "sLocal": sum(samples) / 4.0,
            "t": self.t,
        }

    def obs_vector(self, obs: dict[str, Any] | None = None) -> list[float]:
        o = self.observe() if obs is None else obs
        return [float(o["x"][0]), float(o["x"][1]), *[float(v) for v in o["samples"]]]

    def step(self, action: list[float]) -> H2Step:
        ax = clamp(float(action[0]), -self.cfg["actionMax"], self.cfg["actionMax"])
        ay = clamp(float(action[1]), -self.cfg["actionMax"], self.cfg["actionMax"])
        self.x = clip_pt([self.x[0] + ax, self.x[1] + ay], self.cfg["arenaHalfWidth"])
        self.t += 1
        in_lure = any(dist(self.x, lure) < self.cfg["basinRadius"] for lure in self.lures())
        if in_lure:
            self.entered_basin = True
        done = False
        if in_lure:
            self.outcome = "basin"
            done = True
        elif dist(self.x, self.correct_peak()) < self.cfg["branchRadius"]:
            self.outcome = "correct"
            done = True
        elif dist(self.x, self.wrong_peak()) < self.cfg["branchRadius"]:
            self.outcome = "wrong"
            done = True
        elif self.t >= self.cfg["horizon"]:
            self.outcome = "timeout"
            done = True
        return H2Step(obs=self.observe(), done=done)

    def metrics(self) -> dict[str, Any]:
        return {
            "competence": 1 if self.outcome == "correct" else 0,
            "basin": 1 if self.entered_basin else 0,
            "outcome": self.outcome,
        }


class OracleController:
    label = "Oracle-H2"

    def act(self, env: ForkedFieldEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        tgt = env.correct_peak()
        d = [tgt[0] - env.x[0], tgt[1] - env.x[1]]
        u = unit(d)
        return [u[0] * env.cfg["actionMax"], u[1] * env.cfg["actionMax"]]


class FieldFollower:
    label = "P-Field-H2"

    def __init__(self, rng: Callable[[], float]) -> None:
        self.rng = rng

    def act(self, env: ForkedFieldEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        g = env.field_proposal_unit()
        nx = g[0] + (self.rng() * 2.0 - 1.0) * env.cfg["fieldNoise"]
        ny = g[1] + (self.rng() * 2.0 - 1.0) * env.cfg["fieldNoise"]
        u = unit([nx, ny])
        return [u[0] * env.cfg["actionMax"], u[1] * env.cfg["actionMax"]]


class RewardFollower:
    label = "P-Reward-H2"

    def act(self, env: ForkedFieldEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        return env.reward_action()


class BlindController:
    label = "Blind-H2"

    def __init__(self, rng: Callable[[], float]) -> None:
        self.rng = rng

    def act(self, env: ForkedFieldEnv, _obs: dict[str, Any] | None = None) -> list[float]:
        nx = (self.rng() * 2.0 - 1.0) * 0.25
        ny = 1.0
        u = unit([nx, ny])
        return [u[0] * env.cfg["actionMax"], u[1] * env.cfg["actionMax"]]


def make_controller(label: str, _env: ForkedFieldEnv, seed: int):
    ctrl_rng = make_rng(split_seed(seed, "h2-ctrl"))
    key = label.lower()
    if key in {"oracle", "oracle-h2"}:
        return OracleController()
    if key in {"field", "p-field-h2"}:
        return FieldFollower(ctrl_rng)
    if key in {"reward", "p-reward-h2"}:
        return RewardFollower()
    if key in {"blind", "blind-h2"}:
        return BlindController(ctrl_rng)
    raise ValueError(f"unknown H2 control: {label}")


def roll_episode(env: ForkedFieldEnv, control: str, seed: int, cell_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    env.reset(seed, cell_overrides or {})
    ctrl = make_controller(control, env, seed)
    done = False
    while not done:
        done = env.step(ctrl.act(env, env.observe())).done
    return env.metrics()
