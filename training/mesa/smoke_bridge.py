"""Smoke test for the Phase 2 JS environment bridge.

This deliberately uses only the Python standard library. The goal is to prove
the process boundary and JSONL protocol before bringing in Gymnasium, PyTorch,
or PPO.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = REPO_ROOT / "scripts" / "mesa-env-bridge.mjs"
THROUGHPUT_BATCH_SIZE = 256
THROUGHPUT_BATCH_STEPS = 200
THROUGHPUT_FLOOR = 10_000


class BridgeClient:
    def __init__(self) -> None:
        self.proc = subprocess.Popen(
            ["node", str(BRIDGE)],
            cwd=REPO_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.next_id = 1

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("bridge process pipes are closed")
        payload = {"id": self.next_id, **payload}
        self.next_id += 1
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read() if self.proc.stderr else ""
            raise RuntimeError(f"bridge closed without response. stderr={stderr!r}")
        response = json.loads(line)
        if not response.get("ok"):
            raise RuntimeError(f"bridge error: {response.get('error')}")
        return response

    def close(self) -> None:
        if self.proc.poll() is None:
            try:
                self.request({"cmd": "close"})
            finally:
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()


def assert_close(a: float, b: float, *, tol: float = 1e-9, label: str = "value") -> None:
    if not math.isclose(a, b, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {b}, got {a}")


def assert_vec_close(a: list[float], b: list[float], *, tol: float = 1e-9, label: str = "vector") -> None:
    if len(a) != len(b):
        raise AssertionError(f"{label}: length mismatch {len(a)} != {len(b)}")
    for index, (left, right) in enumerate(zip(a, b)):
        assert_close(left, right, tol=tol, label=f"{label}[{index}]")


def smoke_single_env(client: BridgeClient) -> None:
    made = client.request(
        {
            "cmd": "make",
            "env_id": "env-0",
            "seed": 123,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": 20},
        }
    )
    if len(made["obs"]) != 6:
        raise AssertionError(f"expected local-probe obs dim 6, got {len(made['obs'])}")
    if made["done"] is not False:
        raise AssertionError("new env should not be done")
    required_reward_channels = {"dense", "sparse", "signature"}
    if not required_reward_channels.issubset(set(made["reward_channels"])):
        raise AssertionError(f"unexpected reward channels: {made['reward_channels']}")

    initial_obs = made["obs"]
    stepped = client.request({"cmd": "step", "env_id": "env-0", "action": [0.5, -0.25]})
    if len(stepped["obs"]) != 6:
        raise AssertionError(f"expected stepped obs dim 6, got {len(stepped['obs'])}")
    assert_close(stepped["obs"][0], initial_obs[0] + 0.025, label="x after step")
    assert_close(stepped["obs"][1], initial_obs[1] - 0.0125, label="y after step")
    if stepped["info"]["step_index"] != 1:
        raise AssertionError(f"expected step_index 1, got {stepped['info']['step_index']}")

    reset = client.request({"cmd": "reset", "env_id": "env-0"})
    assert_vec_close(reset["obs"], initial_obs, label="reset obs")
    if reset["info"]["step_index"] != 0:
        raise AssertionError(f"expected reset step_index 0, got {reset['info']['step_index']}")


def smoke_batch_env(client: BridgeClient) -> None:
    made = client.request(
        {
            "cmd": "make_batch",
            "batch_id": "batch-0",
            "count": 4,
            "seed_start": 200,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": 20},
        }
    )
    if made["count"] != 4 or len(made["obs"]) != 4:
        raise AssertionError(f"bad make_batch shape: count={made.get('count')} obs={len(made.get('obs', []))}")
    initial_obs = [obs[:] for obs in made["obs"]]

    actions = [[0.0, 0.0], [0.25, 0.0], [0.0, -0.25], [0.25, -0.25]]
    stepped = client.request({"cmd": "step_batch", "batch_id": "batch-0", "actions": actions})
    if len(stepped["obs"]) != 4 or len(stepped["reward_channels"]) != 4 or len(stepped["done"]) != 4:
        raise AssertionError("bad step_batch response shape")
    assert_close(stepped["obs"][1][0], initial_obs[1][0] + 0.0125, label="batch x after step")
    assert_close(stepped["obs"][2][1], initial_obs[2][1] - 0.0125, label="batch y after step")

    reset = client.request({"cmd": "reset_batch", "batch_id": "batch-0"})
    for index, obs in enumerate(reset["obs"]):
        assert_vec_close(obs, initial_obs[index], label=f"batch reset obs {index}")


def smoke_batch_auto_reset(client: BridgeClient) -> None:
    made = client.request(
        {
            "cmd": "make_batch",
            "batch_id": "batch-autoreset",
            "count": 2,
            "seed_start": 900,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": 1},
        }
    )
    initial_obs = [obs[:] for obs in made["obs"]]
    terminal = client.request(
        {
            "cmd": "step_batch",
            "batch_id": "batch-autoreset",
            "actions": [[0.0, 0.0], [0.0, 0.0]],
        }
    )
    if terminal["done"] != [True, True]:
        raise AssertionError(f"expected terminal batch after horizon=1, got {terminal['done']}")
    reset = client.request(
        {
            "cmd": "step_batch",
            "batch_id": "batch-autoreset",
            "actions": [[1.0, 0.0], [1.0, 0.0]],
        }
    )
    if reset["done"] != [False, False]:
        raise AssertionError(f"expected auto-reset response to be non-terminal, got {reset['done']}")
    for index, info in enumerate(reset["info"]):
        if info.get("auto_reset") is not True:
            raise AssertionError(f"missing auto_reset marker for env {index}: {info}")
        if info["step_index"] != 0:
            raise AssertionError(f"auto-reset env {index} should be at step 0, got {info['step_index']}")
        assert_vec_close(reset["obs"][index], initial_obs[index], label=f"auto-reset obs {index}")


def deterministic_trace() -> list[dict[str, Any]]:
    client = BridgeClient()
    try:
        trace = []
        trace.append(client.request({"cmd": "ping"}))
        trace.append(
            client.request(
                {
                    "cmd": "make_batch",
                    "batch_id": "determinism",
                    "count": 8,
                    "seed_start": 1200,
                    "sensor_tier": "local-probe-field",
                    "env_config": {"horizon": 50},
                }
            )
        )
        action_sets = [
            [[0.1, -0.2] for _ in range(8)],
            [[0.0, 0.0] for _ in range(8)],
            [[-0.25, 0.15] for _ in range(8)],
        ]
        for actions in action_sets:
            trace.append(client.request({"cmd": "step_batch", "batch_id": "determinism", "actions": actions}))
        return trace
    finally:
        client.close()


def smoke_restart_determinism() -> None:
    first = deterministic_trace()
    second = deterministic_trace()
    if json.dumps(first, sort_keys=True) != json.dumps(second, sort_keys=True):
        raise AssertionError("bridge restart determinism failed")


def measure_batch_throughput(client: BridgeClient) -> float:
    client.request(
        {
            "cmd": "make_batch",
            "batch_id": "throughput",
            "count": THROUGHPUT_BATCH_SIZE,
            "seed_start": 5000,
            "sensor_tier": "local-probe-field",
            "env_config": {"horizon": 100_000},
        }
    )
    actions = [[0.05, -0.025] for _ in range(THROUGHPUT_BATCH_SIZE)]
    for _ in range(10):
        client.request({"cmd": "step_batch", "batch_id": "throughput", "actions": actions})

    start = time.perf_counter()
    for _ in range(THROUGHPUT_BATCH_STEPS):
        client.request({"cmd": "step_batch", "batch_id": "throughput", "actions": actions})
    elapsed = time.perf_counter() - start
    env_steps = THROUGHPUT_BATCH_SIZE * THROUGHPUT_BATCH_STEPS
    return env_steps / elapsed


def main() -> int:
    client = BridgeClient()
    try:
        ping = client.request({"cmd": "ping"})
        if ping.get("protocol_version") != "mesa-env-bridge-v1":
            raise AssertionError(f"unexpected protocol version: {ping.get('protocol_version')}")
        smoke_single_env(client)
        smoke_batch_env(client)
        smoke_batch_auto_reset(client)
        smoke_restart_determinism()
        throughput = measure_batch_throughput(client)
        if throughput < THROUGHPUT_FLOOR:
            raise AssertionError(
                f"bridge throughput below floor: {throughput:.0f} env-steps/sec < {THROUGHPUT_FLOOR}"
            )
        print(
            "mesa bridge smoke passed: "
            f"auto_reset=pass determinism_restart=pass throughput_env_steps_per_sec={throughput:.0f}"
        )
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
