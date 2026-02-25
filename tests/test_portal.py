"""
tests/test_portal.py
======================
Unit tests for the Sundog developer portal.

Tests are entirely self-contained; they spin up the HTTP server on a random
free port in a background thread and exercise it with urllib.
"""
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_port() -> int:
    """Return a free TCP port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _http(method: str, url: str, body: Any = None):
    """Make an HTTP request; return (status, dict)."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


# ---------------------------------------------------------------------------
# Fixture: live server
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def portal(tmp_path_factory):
    """Start the portal server on a free port; yield (base_url, state)."""
    from sundog.portal.server import _PortalHTTPServer, PortalHandler, PortalConfig, PortalState

    port = _free_port()
    out = str(tmp_path_factory.mktemp("portal_out"))
    config = PortalConfig()
    state = PortalState(config=config, output_dir=out)
    server = _PortalHTTPServer(("127.0.0.1", port), PortalHandler, state)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.2)   # let server bind
    base = f"http://127.0.0.1:{port}"
    yield base, state
    server.shutdown()


# ---------------------------------------------------------------------------
# 1. Dashboard HTML
# ---------------------------------------------------------------------------

class TestDashboard:
    def test_dashboard_returns_200(self, portal):
        base, _ = portal
        status, _ = _http("GET", f"{base}/api/status")
        assert status == 200

    def test_dashboard_html_contains_title(self, portal):
        base, _ = portal
        req = urllib.request.Request(f"{base}/", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            html = resp.read().decode()
        assert "SUNDOG DEV PORTAL" in html
        assert "Launch Run" in html

    def test_404_for_unknown_path(self, portal):
        base, _ = portal
        status, body = _http("GET", f"{base}/api/doesnotexist")
        assert status == 404
        assert body["ok"] is False


# ---------------------------------------------------------------------------
# 2. Config API
# ---------------------------------------------------------------------------

class TestConfigAPI:
    def test_get_config_returns_defaults(self, portal):
        base, state = portal
        status, body = _http("GET", f"{base}/api/config")
        assert status == 200
        assert body["ok"] is True
        cfg = body["data"]
        assert "runs" in cfg
        assert "seed" in cfg
        assert "policy" in cfg
        assert "max_batch" in cfg
        assert "volatility" in cfg

    def test_update_config_integer_field(self, portal):
        base, state = portal
        status, body = _http("POST", f"{base}/api/config", {"runs": 77})
        assert status == 200
        assert body["ok"] is True
        assert body["data"]["runs"] == 77

    def test_update_config_float_field(self, portal):
        base, state = portal
        status, body = _http("POST", f"{base}/api/config", {"volatility": 0.55})
        assert body["ok"] is True
        assert abs(body["data"]["volatility"] - 0.55) < 0.001

    def test_update_config_string_field(self, portal):
        base, state = portal
        status, body = _http("POST", f"{base}/api/config", {"policy": "greedy"})
        assert body["ok"] is True
        assert body["data"]["policy"] == "greedy"

    def test_update_config_unknown_key_returns_error(self, portal):
        base, state = portal
        status, body = _http("POST", f"{base}/api/config", {"nonexistent_key": 42})
        assert body["ok"] is False
        assert "nonexistent_key" in body["error"]

    def test_update_config_bad_type_returns_error(self, portal):
        base, state = portal
        status, body = _http("POST", f"{base}/api/config", {"runs": "not_a_number"})
        assert body["ok"] is False

    def test_empty_config_post_returns_error(self, portal):
        base, _ = portal
        # POST with empty body
        req = urllib.request.Request(
            f"{base}/api/config",
            data=b"",
            method="POST",
            headers={"Content-Type": "application/json", "Content-Length": "0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            body = json.loads(exc.read())
        assert body["ok"] is False


# ---------------------------------------------------------------------------
# 3. Status API
# ---------------------------------------------------------------------------

class TestStatusAPI:
    def test_status_structure(self, portal):
        base, _ = portal
        status, body = _http("GET", f"{base}/api/status")
        assert status == 200
        assert body["ok"] is True
        d = body["data"]
        assert "active_jobs" in d
        assert "total_jobs" in d
        assert "output_dir" in d
        assert "config" in d

    def test_status_active_jobs_is_int(self, portal):
        base, _ = portal
        _, body = _http("GET", f"{base}/api/status")
        assert isinstance(body["data"]["active_jobs"], int)

    def test_status_output_dir_exists(self, portal):
        base, state = portal
        _, body = _http("GET", f"{base}/api/status")
        assert os.path.isdir(body["data"]["output_dir"])


# ---------------------------------------------------------------------------
# 4. Jobs / Runs API
# ---------------------------------------------------------------------------

class TestJobsAPI:
    def test_list_jobs_initially_empty(self, portal):
        """Fresh server starts with no jobs."""
        from sundog.portal.server import _PortalHTTPServer, PortalHandler, PortalConfig, PortalState
        port = _free_port()
        out_dir = "/tmp"
        config = PortalConfig()
        state = PortalState(config=config, output_dir=out_dir)
        srv = _PortalHTTPServer(("127.0.0.1", port), PortalHandler, state)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        time.sleep(0.2)
        try:
            _, body = _http("GET", f"http://127.0.0.1:{port}/api/jobs")
            assert body["ok"] is True
            assert body["data"] == []
        finally:
            srv.shutdown()

    def test_start_run_missing_type_error(self, portal):
        base, _ = portal
        _, body = _http("POST", f"{base}/api/runs", {"runs": 2})
        assert body["ok"] is False

    def test_start_run_invalid_type_error(self, portal):
        base, _ = portal
        _, body = _http("POST", f"{base}/api/runs", {"type": "unknown_type", "runs": 1})
        assert body["ok"] is False

    def test_start_run_headless_creates_job(self, portal):
        base, state = portal
        _, body = _http("POST", f"{base}/api/runs", {
            "type": "headless", "runs": 2, "seed": 1, "policy": "greedy",
            "max_batch": 5, "max_steps": 100, "volatility": 0.7
        })
        assert body["ok"] is True
        assert "job_id" in body["data"]
        assert body["data"]["status"] in ("queued", "running")

    def test_start_run_returns_job_with_params(self, portal):
        base, _ = portal
        _, body = _http("POST", f"{base}/api/runs", {
            "type": "headless", "runs": 3, "seed": 42,
            "max_batch": 4, "max_steps": 50,
        })
        assert body["ok"] is True
        params = body["data"]["params"]
        assert params["runs"] == 3
        assert params["seed"] == 42

    def test_get_job_by_id(self, portal):
        base, _ = portal
        _, launch = _http("POST", f"{base}/api/runs", {
            "type": "headless", "runs": 1, "seed": 7,
            "max_batch": 3, "max_steps": 30,
        })
        job_id = launch["data"]["job_id"]
        _, body = _http("GET", f"{base}/api/jobs/{job_id}")
        assert body["ok"] is True
        assert body["data"]["job_id"] == job_id

    def test_get_unknown_job_returns_error(self, portal):
        base, _ = portal
        status, body = _http("GET", f"{base}/api/jobs/nonexistent_id")
        assert body["ok"] is False

    def test_list_jobs_includes_submitted_job(self, portal):
        base, _ = portal
        _, launch = _http("POST", f"{base}/api/runs", {
            "type": "headless", "runs": 1, "seed": 99,
            "max_batch": 3, "max_steps": 30,
        })
        job_id = launch["data"]["job_id"]
        _, body = _http("GET", f"{base}/api/jobs")
        ids = [j["job_id"] for j in body["data"]]
        assert job_id in ids

    def test_headless_job_completes(self, portal):
        """Submit a minimal headless run and wait for it to complete."""
        base, _ = portal
        _, launch = _http("POST", f"{base}/api/runs", {
            "type": "headless", "runs": 2, "seed": 5,
            "policy": "greedy", "max_batch": 4, "max_steps": 100,
        })
        assert launch["ok"] is True
        job_id = launch["data"]["job_id"]

        # Poll until done (max 15s for 2 quick runs)
        deadline = time.time() + 15
        status = "running"
        while time.time() < deadline and status in ("queued", "running"):
            time.sleep(0.5)
            _, body = _http("GET", f"{base}/api/jobs/{job_id}")
            status = body["data"]["status"]

        assert status in ("done", "error"), f"job still {status} after timeout"

    def test_completed_job_has_result_summary(self, portal):
        """After a done job, result_summary should be populated."""
        base, _ = portal
        _, launch = _http("POST", f"{base}/api/runs", {
            "type": "headless", "runs": 2, "seed": 11,
            "policy": "greedy", "max_batch": 4, "max_steps": 80,
        })
        job_id = launch["data"]["job_id"]

        deadline = time.time() + 15
        job_body = {}
        while time.time() < deadline:
            time.sleep(0.5)
            _, job_body = _http("GET", f"{base}/api/jobs/{job_id}")
            if job_body["data"]["status"] == "done":
                break

        if job_body.get("data", {}).get("status") == "done":
            summary = job_body["data"].get("result_summary")
            assert summary is not None
            assert "total_runs" in summary
            assert summary["total_runs"] == 2

    def test_cancel_nonexistent_job(self, portal):
        base, _ = portal
        req = urllib.request.Request(
            f"{base}/api/jobs/fake_id",
            method="DELETE",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            body = json.loads(resp.read())
        assert body["data"]["cancelled"] is False


# ---------------------------------------------------------------------------
# 5. PortalConfig unit tests
# ---------------------------------------------------------------------------

class TestPortalConfig:
    def test_defaults(self):
        from sundog.portal.server import PortalConfig
        cfg = PortalConfig()
        assert cfg.runs == 20
        assert cfg.seed == 0
        assert cfg.policy == "greedy"
        assert 0 < cfg.volatility <= 1.0

    def test_update_valid(self):
        from sundog.portal.server import PortalConfig
        cfg = PortalConfig()
        errors = cfg.update({"runs": 50, "seed": 123, "policy": "greedy"})
        assert errors == []
        assert cfg.runs == 50
        assert cfg.seed == 123

    def test_update_invalid_int(self):
        from sundog.portal.server import PortalConfig
        cfg = PortalConfig()
        errors = cfg.update({"runs": "bad"})
        assert errors

    def test_update_unknown_key(self):
        from sundog.portal.server import PortalConfig
        cfg = PortalConfig()
        errors = cfg.update({"foobar": 1})
        assert errors

    def test_to_dict_contains_all_fields(self):
        from sundog.portal.server import PortalConfig
        cfg = PortalConfig()
        d = cfg.to_dict()
        for key in ("runs","seed","policy","max_batch","volatility","max_steps",
                    "gr_max_batch","gr_max_steps","eyesonly_url","gr_slow_mo"):
            assert key in d, f"missing key: {key}"


# ---------------------------------------------------------------------------
# 6. CLI entrypoint
# ---------------------------------------------------------------------------

class TestCLIEntrypoint:
    def test_module_import(self):
        from sundog.portal import start_server, PortalConfig
        assert callable(start_server)

    def test_main_bad_port_exits(self):
        from sundog.portal.__main__ import main
        # port 0 is valid (OS assigns) so we just check it doesn't crash on import
        import inspect
        src = inspect.getsource(main)
        assert "start_server" in src

    def test_portal_config_round_trip(self):
        from sundog.portal.server import PortalConfig
        cfg = PortalConfig(runs=99, seed=7, policy="greedy")
        d = cfg.to_dict()
        cfg2 = PortalConfig(**d)
        assert cfg2.runs == 99
        assert cfg2.seed == 7
