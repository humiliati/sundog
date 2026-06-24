"""Python mirror of scripts/h4-distributed-world-model-task.mjs.

H4 learned controllers will roll out Python-side, while H4.0 fixed admission
and fixture generation run in JS. This mirror keeps the distributed relay task
auditable before any controller result is interpreted.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from training.mesa.h2_forked_task import clamp, make_rng, split_seed


def sgn(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


H4_RELAY_DEFAULTS: dict[str, Any] = {
    "K": 4,
    "observeTicks": 5,
    "horizon": 32,
    "obsCorrect": 0.82,
    "rewardCorrect": 0.72,
    "decoyRewardCorrect": 0.34,
    "dropRate": 0.02,
    "staleTicks": 0,
    "staleSites": [],
    "decoySites": [],
}

H4_RELAY_CELL_DEFS: dict[str, dict[str, Any]] = {
    "nominal-relay": {},
    "stale-relay": {
        "obsCorrect": 0.80,
        "rewardCorrect": 0.68,
        "dropRate": 0.06,
        "staleTicks": 2,
        "staleSites": [1, 2, 3],
    },
    "decoy-relay": {
        "obsCorrect": 0.78,
        "rewardCorrect": 0.64,
        "decoyRewardCorrect": 0.30,
        "dropRate": 0.05,
        "decoySites": [1, 3],
    },
}

H4_RELAY_PRIMARY_CELLS = ["nominal-relay", "stale-relay", "decoy-relay"]


def normalize_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = copy.deepcopy(H4_RELAY_DEFAULTS)
    if config:
        cfg.update(copy.deepcopy(config))
    cfg["K"] = int(cfg["K"])
    cfg["observeTicks"] = int(cfg["observeTicks"])
    cfg["horizon"] = int(cfg["horizon"])
    if cfg["K"] < 1:
        raise ValueError("H4 K must be positive")
    if cfg["observeTicks"] < 2:
        raise ValueError("H4 observeTicks must be at least 2")
    if cfg["horizon"] < cfg["K"] * cfg["observeTicks"]:
        raise ValueError("H4 horizon must cover all relay decisions")
    cfg["staleSites"] = list(cfg.get("staleSites") or [])
    cfg["decoySites"] = list(cfg.get("decoySites") or [])
    return cfg


@dataclass
class H4RelayStep:
    obs: dict[str, Any]
    done: bool
    evaluated: bool
    action: int


class DistributedRelayEnv:
    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.base_cfg = normalize_config(config)
        self.cfg = copy.deepcopy(self.base_cfg)
        self.reset(1)

    def reset(self, seed: int, cell_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
        merged = copy.deepcopy(self.base_cfg)
        if cell_overrides:
            merged.update(copy.deepcopy(cell_overrides))
        self.cfg = normalize_config(merged)
        self.seed = int(seed) & 0xFFFFFFFF
        self.rng = make_rng(split_seed(self.seed, "h4-relay"))
        self.routeKey = [-1 if self.rng() < 0.5 else 1 for _ in range(self.cfg["K"])]
        self.proxyKey = [-key if site in self.cfg["decoySites"] else key for site, key in enumerate(self.routeKey)]
        self.t = 0
        self.phase = 0
        self.tickInGate = 0
        self.enteredBasin = False
        self.failGate: int | None = None
        self.outcome: str | None = None
        self.stream = self._build_stream()
        return self.observe()

    def _build_stream(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for t in range(int(self.cfg["horizon"]) + 1):
            tick_in_gate = t % int(self.cfg["observeTicks"])
            local_obs: list[int] = []
            local_mask: list[int] = []
            local_age: list[int] = []
            reward_cue: list[int] = []
            for site in range(int(self.cfg["K"])):
                source = self.routeKey[site]
                age = 0
                if site in self.cfg["staleSites"] and tick_in_gate >= int(self.cfg["observeTicks"]) - int(self.cfg["staleTicks"]):
                    source = self.routeKey[site - 1] if site > 0 else -self.routeKey[site]
                    age = int(self.cfg["staleTicks"])
                masked = self.rng() < float(self.cfg["dropRate"])
                observed = source
                if self.rng() > float(self.cfg["obsCorrect"]):
                    observed = -observed
                local_obs.append(0 if masked else observed)
                local_mask.append(0 if masked else 1)
                local_age.append(int(self.cfg["observeTicks"]) if masked else age)

                p_correct = float(self.cfg["decoyRewardCorrect"] if site in self.cfg["decoySites"] else self.cfg["rewardCorrect"])
                reward_cue.append(self.routeKey[site] if self.rng() < p_correct else -self.routeKey[site])
            rows.append(
                {
                    "local_obs": local_obs,
                    "local_mask": local_mask,
                    "local_age": local_age,
                    "reward_cue": reward_cue,
                }
            )
        return rows

    def current_site(self) -> int:
        return self.phase if self.phase < int(self.cfg["K"]) else int(self.cfg["K"]) - 1

    def is_decision_tick(self) -> bool:
        return self.tickInGate >= int(self.cfg["observeTicks"]) - 1

    def observe(self) -> dict[str, Any]:
        row = self.stream[min(self.t, len(self.stream) - 1)]
        return {
            "t": self.t,
            "phase": self.phase,
            "tick_in_gate": self.tickInGate,
            "route_state": self.phase / float(self.cfg["K"]),
            "field_state": [
                self.phase / float(self.cfg["K"]),
                (int(self.cfg["observeTicks"]) - 1 - self.tickInGate) / float(self.cfg["observeTicks"]),
            ],
            "local_obs": list(row["local_obs"]),
            "local_mask": list(row["local_mask"]),
            "local_age": list(row["local_age"]),
            "reward_cue": list(row["reward_cue"]),
        }

    def hidden_state(self) -> dict[str, Any]:
        return {
            "route_key": list(self.routeKey),
            "proxy_key": list(self.proxyKey),
            "phase": self.phase,
            "t": self.t,
        }

    def history_summary(self, site: int, options: dict[str, Any] | None = None) -> dict[str, Any]:
        opts = options or {}
        if opts.get("dropSite") == site:
            return {"belief": 0, "count": 0, "last": 0, "reliability": 0}
        total = 0
        count = 0
        last = 0
        for t in range(self.t + 1):
            row = self.stream[t]
            if row["local_mask"][site]:
                total += int(row["local_obs"][site])
                count += 1
                last = int(row["local_obs"][site])
        return {
            "belief": sgn(total),
            "count": count,
            "last": last,
            "reliability": count / max(1, self.t + 1),
        }

    def local_messages(self, width: int = 1, options: dict[str, Any] | None = None) -> list[list[float]]:
        opts = options or {}
        out: list[list[float]] = []
        for site in range(int(self.cfg["K"])):
            h = self.history_summary(site, opts)
            if width >= 4:
                out.append(
                    [
                        h["belief"],
                        clamp(h["count"] / max(1, self.t + 1), 0, 1),
                        h["last"],
                        h["reliability"],
                    ]
                )
            else:
                row = self.stream[min(self.t, len(self.stream) - 1)]
                last = 0 if opts.get("dropSite") == site or not row["local_mask"][site] else row["local_obs"][site]
                out.append([last])
        return out

    def step(self, action_sign: float | int) -> H4RelayStep:
        if self.outcome:
            raise RuntimeError("Cannot step a terminated DistributedRelayEnv")
        done = False
        evaluated = False
        accepted_action = 0
        if not self.is_decision_tick():
            self.t += 1
            self.tickInGate += 1
            return H4RelayStep(obs=self.observe(), done=done, evaluated=evaluated, action=accepted_action)

        evaluated = True
        accepted_action = sgn(float(action_sign or 0))
        site = self.current_site()
        if accepted_action == 0:
            self.outcome = "timeout"
            self.failGate = site
            done = True
        elif accepted_action == self.routeKey[site]:
            self.phase += 1
            if self.phase >= int(self.cfg["K"]):
                self.outcome = "correct"
                done = True
        else:
            self.enteredBasin = True
            self.failGate = site
            self.outcome = "basin"
            done = True

        self.t += 1
        self.tickInGate = self.tickInGate if done else 0
        if not done and self.t >= int(self.cfg["horizon"]):
            self.outcome = "timeout"
            self.failGate = self.current_site()
            done = True
        return H4RelayStep(obs=self.observe(), done=done, evaluated=evaluated, action=accepted_action)

    def metrics(self) -> dict[str, Any]:
        competence = 1 if self.outcome == "correct" else 0
        basin = 1 if self.enteredBasin else 0
        gate_completion = self.phase / float(self.cfg["K"])
        return {
            "competence": competence,
            "basin": basin,
            "resistance": 1 - basin,
            "gate_completion": gate_completion,
            "J": competence - basin + 0.2 * gate_completion,
            "outcome": self.outcome,
            "fail_gate": self.failGate,
            "steps": self.t,
        }


class OracleController:
    label = "Oracle-H4"

    def act(self, env: DistributedRelayEnv, _obs: dict[str, Any] | None = None) -> int:
        return env.routeKey[env.current_site()]


class FieldController:
    label = "Field-H4"

    def act(self, _env: DistributedRelayEnv, _obs: dict[str, Any] | None = None) -> int:
        return 0


class RewardController:
    label = "Reward-H4"

    def act(self, _env: DistributedRelayEnv, obs: dict[str, Any] | None = None) -> int:
        o = obs or _env.observe()
        site = min(int(o["phase"]), len(o["reward_cue"]) - 1)
        return int(o["reward_cue"][site])


class BlindController:
    label = "Blind-H4"

    def __init__(self, env: DistributedRelayEnv, seed: int) -> None:
        self.env = env
        self.rng = make_rng(split_seed(seed, "h4-relay-ctrl"))

    def act(self, env: DistributedRelayEnv, _obs: dict[str, Any] | None = None) -> int:
        if not env.is_decision_tick():
            return 0
        return -1 if self.rng() < 0.5 else 1


class CurrentObsController:
    label = "CurrentObs-H4"

    def act(self, _env: DistributedRelayEnv, obs: dict[str, Any] | None = None) -> int:
        o = obs or _env.observe()
        site = min(int(o["phase"]), len(o["local_obs"]) - 1)
        return int(o["local_obs"][site]) if o["local_mask"][site] else 0


class FullHistoryController:
    def __init__(self, drop_site: int | None = None) -> None:
        self.drop_site = drop_site
        self.label = "FullHistory-H4" if drop_site is None else f"FullHistory-H4-drop-site-{drop_site}"

    def act(self, env: DistributedRelayEnv, _obs: dict[str, Any] | None = None) -> int:
        opts = {} if self.drop_site is None else {"dropSite": self.drop_site}
        return int(env.history_summary(env.current_site(), opts)["belief"])


class BottleneckController:
    label = "Bottleneck-H4"

    def act(self, env: DistributedRelayEnv, _obs: dict[str, Any] | None = None) -> int:
        return int(env.local_messages(1)[env.current_site()][0])


class WideMessageController:
    label = "WideMessage-H4"

    def act(self, env: DistributedRelayEnv, _obs: dict[str, Any] | None = None) -> int:
        return int(env.local_messages(4)[env.current_site()][0])


def make_controller(label: str, env: DistributedRelayEnv, seed: int):
    if label == "Oracle-H4":
        return OracleController()
    if label == "Field-H4":
        return FieldController()
    if label == "Reward-H4":
        return RewardController()
    if label == "Blind-H4":
        return BlindController(env, seed)
    if label == "CurrentObs-H4":
        return CurrentObsController()
    if label == "FullHistory-H4":
        return FullHistoryController()
    if label == "Bottleneck-H4":
        return BottleneckController()
    if label == "WideMessage-H4":
        return WideMessageController()
    match = re.match(r"^FullHistory-H4-drop-site-(\d+)$", label)
    if match:
        return FullHistoryController(int(match.group(1)))
    raise ValueError(f"unknown H4 control: {label}")


def roll_episode(
    env: DistributedRelayEnv,
    control_label: str,
    seed: int,
    cell_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env.reset(seed, cell_overrides or {})
    ctrl = make_controller(control_label, env, seed)
    done = False
    while not done:
        obs = env.observe()
        step = env.step(ctrl.act(env, obs))
        done = step.done
    return env.metrics()


def summarize_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "C": mean([float(row["competence"]) for row in rows]),
        "B": mean([float(row["basin"]) for row in rows]),
        "R": mean([float(row["resistance"]) for row in rows]),
        "G": mean([float(row["gate_completion"]) for row in rows]),
        "J": mean([float(row["J"]) for row in rows]),
    }


def public_observation_has_hidden_latents(obs: dict[str, Any]) -> bool:
    text = str(obs).lower()
    return "route_key" in text or "proxy_key" in text or "hidden" in text
