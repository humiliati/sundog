"""Small Python client for the Phase 2 JS environment bridge."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE = REPO_ROOT / "scripts" / "mesa-env-bridge.mjs"


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
            if self.proc.stdin is not None:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=5)

    def __enter__(self) -> "BridgeClient":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
