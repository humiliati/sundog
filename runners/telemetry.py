import json
import time
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, IO, Union


@dataclass
class FloorSnapshot:
    """Per-floor snapshot for balancing analysis."""
    seed: int
    run_id: str
    build: str = "default"
    floor_index: int = 0
    biome: str = ""
    keys_inventory: Dict[str, int] = field(default_factory=dict)
    gates_visible: int = 0
    keys_acquired: int = 0
    keys_used: int = 0
    gate_spawned: bool = False
    gate_encountered: bool = False
    pity_triggered: bool = False
    outcome: str = "ongoing"
    score: int = 0
    steps_on_floor: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass
class EventRecord:
    """Append-only event log entry."""
    seed: int
    run_id: str
    event: str
    floor_index: int
    biome: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class TelemetryLogger:
    def __init__(self, output_file: Union[IO, str, None] = None):
        self._owned = False
        if output_file is None:
            self._file = sys.stdout
        elif isinstance(output_file, str):
            self._file = open(output_file, "w", encoding="utf-8")
            self._owned = True
        else:
            self._file = output_file

    def log_event(self, record: EventRecord):
        d = asdict(record)
        self._file.write(json.dumps({"type": "event", **d}) + "\n")
        self._file.flush()

    def log_snapshot(self, snapshot: FloorSnapshot):
        d = asdict(snapshot)
        self._file.write(json.dumps({"type": "snapshot", **d}) + "\n")
        self._file.flush()

    def close(self):
        if self._owned and self._file:
            self._file.close()
            self._file = None

    @staticmethod
    def open_file(path: str) -> "TelemetryLogger":
        return TelemetryLogger(output_file=path)
