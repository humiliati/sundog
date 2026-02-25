"""
sundog.portal.server
=====================
Self-contained HTTP server for the Sundog developer portal.

Uses only Python stdlib — no external web framework required.

Routes
------
GET  /                  Dashboard HTML
GET  /api/status        Server status, config, active job count
GET  /api/config        Current default run config (JSON)
POST /api/config        Update default run config (JSON body)
GET  /api/jobs          List all jobs (active + completed)
GET  /api/jobs/{id}     Single job detail + result
POST /api/runs          Launch a new runner job
     Body (JSON):
       type        : "headless" | "gone_rogue"   (required)
       runs        : int                          (default from config)
       seed        : int                          (default from config)
       policy      : str                          (default from config)
       max_batch   : int                          (default from config)
       volatility  : float                        (default from config)
       max_steps   : int                          (default from config)
       eyesonly_url: str                          (gone_rogue only)
DELETE /api/jobs/{id}   Cancel a running job (best-effort)

All JSON responses follow:  {"ok": bool, "data": ..., "error": str|null}
"""
from __future__ import annotations

import http.server
import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger("sundog.portal")

# ---------------------------------------------------------------------------
# Config / state dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PortalConfig:
    """Adjustable defaults for new runner jobs."""
    # Headless runner defaults
    runs: int = 20
    seed: int = 0
    policy: str = "greedy"
    max_batch: int = 10
    volatility: float = 0.7
    max_steps: int = 2000
    # Gone Rogue extras
    eyesonly_url: str = ""
    gr_max_batch: int = 8
    gr_max_steps: int = 1000
    gr_slow_mo: int = 0

    def update(self, data: dict) -> List[str]:
        """Apply a dict of updates; return list of validation errors."""
        errors = []
        int_fields = {"runs", "seed", "max_batch", "max_steps", "gr_max_batch", "gr_max_steps", "gr_slow_mo"}
        float_fields = {"volatility"}
        str_fields = {"policy", "eyesonly_url"}
        for k, v in data.items():
            if k in int_fields:
                try:
                    setattr(self, k, int(v))
                except (TypeError, ValueError):
                    errors.append(f"{k} must be an integer")
            elif k in float_fields:
                try:
                    setattr(self, k, float(v))
                except (TypeError, ValueError):
                    errors.append(f"{k} must be a float")
            elif k in str_fields:
                setattr(self, k, str(v))
            else:
                errors.append(f"Unknown config key: {k}")
        return errors

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class JobRecord:
    """Tracks a single runner job."""
    job_id: str
    job_type: str        # "headless" | "gone_rogue"
    params: Dict[str, Any]
    status: str = "queued"   # queued | running | done | error | cancelled
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    output_file: Optional[str] = None
    error: Optional[str] = None
    result_summary: Optional[Dict[str, Any]] = None
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    _cancel: bool = field(default=False, repr=False)

    def to_dict(self, include_result: bool = False) -> dict:
        d = {
            "job_id": self.job_id,
            "job_type": self.job_type,
            "params": self.params,
            "status": self.status,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_file": self.output_file,
            "error": self.error,
        }
        if include_result:
            d["result_summary"] = self.result_summary
        return d


# ---------------------------------------------------------------------------
# Portal state (singleton)
# ---------------------------------------------------------------------------

class PortalState:
    """Global server state shared across all request handlers."""

    def __init__(self, config: PortalConfig, output_dir: str):
        self.config = config
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        self._jobs: Dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    # ── Job management ────────────────────────────────────────────────

    def submit_job(self, job_type: str, params: dict) -> JobRecord:
        job_id = str(uuid.uuid4())[:8]
        out_file = os.path.join(self.output_dir, f"{job_type}_{job_id}.jsonl")
        job = JobRecord(
            job_id=job_id,
            job_type=job_type,
            params=params,
            output_file=out_file,
        )
        with self._lock:
            self._jobs[job_id] = job
        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        job._thread = t
        t.start()
        return job

    def _run_job(self, job: JobRecord):
        job.status = "running"
        job.started_at = time.time()
        try:
            if job.job_type == "headless":
                self._run_headless(job)
            elif job.job_type == "gone_rogue":
                self._run_gone_rogue(job)
            else:
                raise ValueError(f"Unknown job type: {job.job_type!r}")
            job.status = "done"
        except Exception as exc:  # noqa: BLE001
            job.status = "error"
            job.error = str(exc)
            logger.exception("Job %s failed", job.job_id)
        finally:
            job.finished_at = time.time()
            self._summarise_job(job)

    def _run_headless(self, job: JobRecord):
        """Run sundog.runners.headless in-process."""
        from sundog.runners.headless import run_single
        p = job.params
        results = []
        runs = p.get("runs", 20)
        seed = p.get("seed", 0)
        with open(job.output_file, "w", encoding="utf-8") as fh:
            for i in range(runs):
                if job._cancel:
                    job.status = "cancelled"
                    return
                res = run_single(
                    seed=seed,
                    run_index=i,
                    policy_name=p.get("policy", "greedy"),
                    max_batch=p.get("max_batch", 10),
                    volatility=p.get("volatility", 0.7),
                    max_steps=p.get("max_steps", 2000),
                )
                fh.write(json.dumps(res) + "\n")
                fh.flush()
                results.append(res)

    def _run_gone_rogue(self, job: JobRecord):
        """Launch gone_rogue_headless as a subprocess (requires Playwright)."""
        p = job.params
        cmd = [
            sys.executable, "-m", "sundog.runners.gone_rogue_headless",
            "--runs", str(p.get("runs", 10)),
            "--seed", str(p.get("seed", 0)),
            "--policy", str(p.get("policy", "greedy")),
            "--max-batch", str(p.get("max_batch", 8)),
            "--max-steps", str(p.get("max_steps", 1000)),
            "--out", job.output_file,
            "--quiet",
        ]
        url = p.get("eyesonly_url", "")
        if url:
            cmd += ["--eyesonly-url", url]
        env = os.environ.copy()
        if url:
            env["EYESONLY_BASE_URL"] = url
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        while proc.poll() is None:
            if job._cancel:
                proc.terminate()
                job.status = "cancelled"
                return
            time.sleep(0.5)
        rc = proc.returncode
        if rc != 0:
            stderr = proc.stderr.read().decode(errors="replace")
            raise RuntimeError(f"gone_rogue_headless exited {rc}: {stderr[:500]}")

    def _summarise_job(self, job: JobRecord):
        """Read the output JSONL and build a quick summary."""
        if not job.output_file or not os.path.exists(job.output_file):
            return
        try:
            results = []
            with open(job.output_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results.append(json.loads(line))
            if not results:
                return
            completed = sum(1 for r in results if r.get("outcome") == "completed")
            died = sum(1 for r in results if r.get("outcome") == "died")
            scores = [r.get("score", 0) for r in results if "score" in r]
            floors = [r.get("floors_completed", r.get("floor", 0)) for r in results]
            job.result_summary = {
                "total_runs": len(results),
                "completed": completed,
                "died": died,
                "completion_rate": round(completed / len(results) * 100, 1) if results else 0,
                "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
                "avg_floors": round(sum(floors) / len(floors), 1) if floors else None,
            }
        except Exception:  # noqa: BLE001
            pass

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job and job.status == "running":
            job._cancel = True
            return True
        return False

    def get_job(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[JobRecord]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for j in self._jobs.values() if j.status in ("queued", "running"))


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class PortalHandler(http.server.BaseHTTPRequestHandler):
    """Single-class request handler; ``server.state`` is a PortalState."""

    # Quiet the access log — noisy during auto-refresh
    def log_message(self, fmt, *args):
        pass

    # ── Dispatch ──────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path == "/":
            self._serve_dashboard()
        elif path == "/api/status":
            self._json(self._api_status())
        elif path == "/api/config":
            self._json({"ok": True, "data": self.server.state.config.to_dict()})
        elif path == "/api/jobs":
            self._json(self._api_list_jobs())
        elif path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/"):]
            self._json(self._api_get_job(job_id))
        else:
            self._json({"ok": False, "error": "not found"}, status=404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        body = self._read_json_body()
        if path == "/api/runs":
            self._json(self._api_start_run(body))
        elif path == "/api/config":
            self._json(self._api_update_config(body))
        else:
            self._json({"ok": False, "error": "not found"}, status=404)

    def do_DELETE(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/") or "/"
        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/"):]
            ok = self.server.state.cancel_job(job_id)
            self._json({"ok": ok, "data": {"cancelled": ok}})
        else:
            self._json({"ok": False, "error": "not found"}, status=404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── API handlers ──────────────────────────────────────────────────

    def _api_status(self) -> dict:
        st = self.server.state
        return {
            "ok": True,
            "data": {
                "active_jobs": st.active_count(),
                "total_jobs": len(st.list_jobs()),
                "output_dir": st.output_dir,
                "config": st.config.to_dict(),
            },
        }

    def _api_list_jobs(self) -> dict:
        jobs = self.server.state.list_jobs()
        return {
            "ok": True,
            "data": [j.to_dict(include_result=True) for j in jobs],
        }

    def _api_get_job(self, job_id: str) -> dict:
        job = self.server.state.get_job(job_id)
        if not job:
            return {"ok": False, "error": f"job {job_id!r} not found"}
        return {"ok": True, "data": job.to_dict(include_result=True)}

    def _api_start_run(self, body: dict) -> dict:
        if not body:
            return {"ok": False, "error": "request body required"}
        job_type = body.get("type", "")
        if job_type not in ("headless", "gone_rogue"):
            return {"ok": False, "error": "type must be 'headless' or 'gone_rogue'"}

        cfg = self.server.state.config
        if job_type == "headless":
            params = {
                "runs":       int(body.get("runs",       cfg.runs)),
                "seed":       int(body.get("seed",       cfg.seed)),
                "policy":     str(body.get("policy",     cfg.policy)),
                "max_batch":  int(body.get("max_batch",  cfg.max_batch)),
                "volatility": float(body.get("volatility", cfg.volatility)),
                "max_steps":  int(body.get("max_steps",  cfg.max_steps)),
            }
        else:
            params = {
                "runs":         int(body.get("runs",         cfg.runs)),
                "seed":         int(body.get("seed",         cfg.seed)),
                "policy":       str(body.get("policy",       cfg.policy)),
                "max_batch":    int(body.get("max_batch",    cfg.gr_max_batch)),
                "max_steps":    int(body.get("max_steps",    cfg.gr_max_steps)),
                "eyesonly_url": str(body.get("eyesonly_url", cfg.eyesonly_url)),
            }

        job = self.server.state.submit_job(job_type, params)
        return {"ok": True, "data": job.to_dict()}

    def _api_update_config(self, body: dict) -> dict:
        if not body:
            return {"ok": False, "error": "request body required"}
        errors = self.server.state.config.update(body)
        if errors:
            return {"ok": False, "error": "; ".join(errors)}
        return {"ok": True, "data": self.server.state.config.to_dict()}

    # ── HTML dashboard ────────────────────────────────────────────────

    def _serve_dashboard(self):
        html = _DASHBOARD_HTML
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(html.encode())))
        self.end_headers()
        self.wfile.write(html.encode())

    # ── Helpers ───────────────────────────────────────────────────────

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length > 0:
                return json.loads(self.rfile.read(length))
        except Exception:  # noqa: BLE001
            pass
        return {}

    def _json(self, data: dict, status: int = 200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Dashboard HTML (single-page, no external dependencies)
# ---------------------------------------------------------------------------

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sundog Dev Portal</title>
<style>
:root {
  --bg: #0d0d0d; --surface: #181818; --border: #2a2a2a;
  --accent: #1cff9b; --accent2: #ffcc00; --red: #ff4455;
  --text: #d4d4d4; --dim: #666; --font: 'Courier New', monospace;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: var(--bg); color: var(--text); font-family: var(--font); font-size: 13px; }
a { color: var(--accent); text-decoration: none; }
header {
  background: var(--surface); border-bottom: 1px solid var(--border);
  padding: 12px 24px; display: flex; align-items: center; gap: 16px;
}
header h1 { font-size: 16px; color: var(--accent); letter-spacing: 2px; }
header .meta { margin-left: auto; color: var(--dim); font-size: 11px; }
#status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dim); display: inline-block; }
#status-dot.live { background: var(--accent); box-shadow: 0 0 6px var(--accent); }
.layout { display: grid; grid-template-columns: 340px 1fr; gap: 0; height: calc(100vh - 45px); overflow: hidden; }
.sidebar {
  background: var(--surface); border-right: 1px solid var(--border);
  overflow-y: auto; padding: 16px;
}
.main { overflow-y: auto; padding: 16px; }
.card {
  background: var(--bg); border: 1px solid var(--border);
  border-radius: 4px; padding: 14px; margin-bottom: 14px;
}
.card h2 { font-size: 11px; letter-spacing: 1px; color: var(--dim); text-transform: uppercase; margin-bottom: 12px; }
label { display: block; font-size: 11px; color: var(--dim); margin-bottom: 3px; margin-top: 8px; }
label:first-of-type { margin-top: 0; }
input[type=text], input[type=number], select {
  width: 100%; background: var(--surface); border: 1px solid var(--border);
  color: var(--text); font-family: var(--font); font-size: 12px;
  padding: 5px 8px; border-radius: 3px; outline: none;
}
input:focus, select:focus { border-color: var(--accent); }
.row { display: flex; gap: 8px; }
.row > * { flex: 1; }
button {
  background: none; border: 1px solid var(--accent); color: var(--accent);
  font-family: var(--font); font-size: 12px; padding: 7px 14px;
  border-radius: 3px; cursor: pointer; transition: all .15s;
}
button:hover { background: var(--accent); color: var(--bg); }
button.danger { border-color: var(--red); color: var(--red); }
button.danger:hover { background: var(--red); color: var(--bg); }
button.secondary { border-color: var(--border); color: var(--dim); }
button.secondary:hover { border-color: var(--text); color: var(--text); background: none; }
.btn-row { display: flex; gap: 8px; margin-top: 12px; }
.badge {
  display: inline-block; font-size: 10px; padding: 1px 6px;
  border-radius: 2px; border: 1px solid;
}
.badge.running { color: var(--accent2); border-color: var(--accent2); }
.badge.done { color: var(--accent); border-color: var(--accent); }
.badge.error, .badge.cancelled { color: var(--red); border-color: var(--red); }
.badge.queued { color: var(--dim); border-color: var(--border); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th { text-align: left; padding: 6px 8px; font-size: 10px; color: var(--dim);
     letter-spacing: 1px; border-bottom: 1px solid var(--border); }
td { padding: 7px 8px; border-bottom: 1px solid var(--border)22; vertical-align: top; }
tr:hover td { background: var(--surface); }
.mono { font-family: var(--font); }
.dim { color: var(--dim); }
.accent { color: var(--accent); }
.warn { color: var(--accent2); }
.err { color: var(--red); }
#toast {
  position: fixed; bottom: 20px; right: 20px;
  background: var(--surface); border: 1px solid var(--accent);
  color: var(--accent); padding: 10px 16px; border-radius: 4px;
  font-size: 12px; display: none; z-index: 99;
}
.config-save-row { display: flex; justify-content: flex-end; margin-top: 10px; }
.section-title { font-size: 11px; color: var(--dim); letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid var(--border); padding-bottom: 6px; }
.summary-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.chip { font-size: 10px; padding: 2px 7px; border: 1px solid var(--border); border-radius: 10px; color: var(--dim); }
.chip.green { border-color: var(--accent)66; color: var(--accent); }
.chip.yellow { border-color: var(--accent2)66; color: var(--accent2); }
</style>
</head>
<body>
<header>
  <span id="status-dot"></span>
  <h1>⬡ SUNDOG DEV PORTAL</h1>
  <div class="meta">
    <span id="header-jobs">— jobs</span> &nbsp;|&nbsp;
    <span id="header-active">—</span> active &nbsp;|&nbsp;
    <span id="header-dir">—</span>
  </div>
</header>

<div class="layout">
  <!-- SIDEBAR: Launch + Config -->
  <div class="sidebar">

    <div class="card">
      <h2>Launch Run</h2>
      <label>Runner type</label>
      <select id="run-type" onchange="onTypeChange()">
        <option value="headless">Headless (simulation)</option>
        <option value="gone_rogue">Gone Rogue (Playwright)</option>
      </select>

      <div class="row">
        <div>
          <label>Runs</label>
          <input type="number" id="param-runs" value="20" min="1">
        </div>
        <div>
          <label>Base seed</label>
          <input type="number" id="param-seed" value="0">
        </div>
      </div>

      <label>Policy</label>
      <select id="param-policy">
        <option value="greedy">greedy</option>
      </select>

      <div class="row">
        <div>
          <label>Max batch</label>
          <input type="number" id="param-max-batch" value="10" min="1">
        </div>
        <div>
          <label>Max steps</label>
          <input type="number" id="param-max-steps" value="2000" min="1">
        </div>
      </div>

      <div id="volatility-row">
        <label>Volatility threshold</label>
        <input type="number" id="param-volatility" value="0.7" min="0" max="1" step="0.05">
      </div>

      <div id="gr-url-row" style="display:none">
        <label>EyesOnly URL (public/js)</label>
        <input type="text" id="param-eyesonly-url" placeholder="http://localhost:8787/public/js">
      </div>

      <div class="btn-row">
        <button onclick="launchRun()" style="flex:1">▶ Launch</button>
      </div>
      <div id="launch-error" style="color:var(--red);font-size:11px;margin-top:6px;display:none"></div>
    </div>

    <div class="card">
      <h2>Default Config</h2>
      <p style="font-size:11px;color:var(--dim);margin-bottom:10px;">
        Saved as defaults for new launch forms.
      </p>
      <div id="config-form"></div>
      <div class="config-save-row">
        <button onclick="saveConfig()" class="secondary" style="font-size:11px;padding:5px 12px">Save defaults</button>
      </div>
      <div id="config-error" style="color:var(--red);font-size:11px;margin-top:6px;display:none"></div>
    </div>

  </div>

  <!-- MAIN: Jobs table -->
  <div class="main">
    <div class="section-title">JOBS &nbsp;<span id="refresh-countdown" style="font-size:10px;color:var(--dim)"></span></div>
    <div id="jobs-container">
      <p class="dim" style="padding:20px">Loading…</p>
    </div>
  </div>
</div>

<div id="toast"></div>

<script>
// ── State ────────────────────────────────────────────────────────────────
var REFRESH_INTERVAL = 4000;
var refreshTimer = null;
var countdown = REFRESH_INTERVAL / 1000;

// ── Boot ────────────────────────────────────────────────────────────────
window.addEventListener('load', function () {
  loadStatus();
  loadConfig();
  loadJobs();
  scheduleRefresh();
});

function scheduleRefresh() {
  countdown = REFRESH_INTERVAL / 1000;
  clearInterval(refreshTimer);
  refreshTimer = setInterval(function () {
    countdown -= 1;
    var el = document.getElementById('refresh-countdown');
    if (el) el.textContent = '(refresh in ' + countdown + 's)';
    if (countdown <= 0) {
      loadStatus();
      loadJobs();
      countdown = REFRESH_INTERVAL / 1000;
    }
  }, 1000);
}

// ── API helpers ──────────────────────────────────────────────────────────
function api(method, path, body, cb) {
  var opts = { method: method, headers: { 'Content-Type': 'application/json' } };
  if (body) opts.body = JSON.stringify(body);
  fetch(path, opts)
    .then(function (r) { return r.json(); })
    .then(cb)
    .catch(function (e) { toast('Network error: ' + e.message, true); });
}

function toast(msg, err) {
  var el = document.getElementById('toast');
  el.textContent = msg;
  el.style.display = 'block';
  el.style.borderColor = err ? 'var(--red)' : 'var(--accent)';
  el.style.color = err ? 'var(--red)' : 'var(--accent)';
  setTimeout(function () { el.style.display = 'none'; }, 3000);
}

// ── Status ───────────────────────────────────────────────────────────────
function loadStatus() {
  api('GET', '/api/status', null, function (r) {
    if (!r.ok) return;
    var d = r.data;
    document.getElementById('header-jobs').textContent = d.total_jobs + ' jobs';
    document.getElementById('header-active').textContent = d.active_jobs;
    document.getElementById('header-dir').textContent = d.output_dir;
    var dot = document.getElementById('status-dot');
    dot.className = d.active_jobs > 0 ? 'live' : '';
  });
}

// ── Config ───────────────────────────────────────────────────────────────
var currentConfig = {};

function loadConfig() {
  api('GET', '/api/config', null, function (r) {
    if (!r.ok) return;
    currentConfig = r.data;
    renderConfigForm(r.data);
    applyConfigToForm(r.data);
  });
}

function renderConfigForm(cfg) {
  var rows = [
    ['runs',         'Default runs',          'number'],
    ['seed',         'Default seed',          'number'],
    ['policy',       'Default policy',        'text'],
    ['max_batch',    'Max batch (headless)',   'number'],
    ['volatility',   'Volatility threshold',  'number'],
    ['max_steps',    'Max steps (headless)',   'number'],
    ['gr_max_batch', 'Max batch (gone rogue)', 'number'],
    ['gr_max_steps', 'Max steps (gone rogue)', 'number'],
    ['eyesonly_url', 'EyesOnly URL',           'text'],
  ];
  var html = '';
  for (var i = 0; i < rows.length; i++) {
    var key = rows[i][0], lbl = rows[i][1], type = rows[i][2];
    var val = cfg[key] !== undefined ? cfg[key] : '';
    html += '<label>' + lbl + '</label>';
    html += '<input type="' + type + '" id="cfg-' + key + '" value="' + esc(String(val)) + '">';
  }
  document.getElementById('config-form').innerHTML = html;
}

function applyConfigToForm(cfg) {
  setVal('param-runs',          cfg.runs);
  setVal('param-seed',          cfg.seed);
  setVal('param-policy',        cfg.policy);
  setVal('param-max-batch',     cfg.max_batch);
  setVal('param-max-steps',     cfg.max_steps);
  setVal('param-volatility',    cfg.volatility);
  setVal('param-eyesonly-url',  cfg.eyesonly_url || '');
}

function setVal(id, v) {
  var el = document.getElementById(id);
  if (el && v !== undefined) el.value = v;
}

function saveConfig() {
  var keys = ['runs','seed','policy','max_batch','volatility','max_steps','gr_max_batch','gr_max_steps','eyesonly_url'];
  var data = {};
  for (var i = 0; i < keys.length; i++) {
    var el = document.getElementById('cfg-' + keys[i]);
    if (el) data[keys[i]] = el.value;
  }
  var errEl = document.getElementById('config-error');
  errEl.style.display = 'none';
  api('POST', '/api/config', data, function (r) {
    if (r.ok) {
      currentConfig = r.data;
      toast('Config saved.');
    } else {
      errEl.textContent = r.error || 'Error saving config';
      errEl.style.display = 'block';
    }
  });
}

// ── Run form ─────────────────────────────────────────────────────────────
function onTypeChange() {
  var type = document.getElementById('run-type').value;
  document.getElementById('gr-url-row').style.display    = type === 'gone_rogue' ? '' : 'none';
  document.getElementById('volatility-row').style.display = type === 'headless' ? '' : 'none';
  if (type === 'gone_rogue') {
    setVal('param-max-batch', currentConfig.gr_max_batch || 8);
    setVal('param-max-steps', currentConfig.gr_max_steps || 1000);
  } else {
    setVal('param-max-batch', currentConfig.max_batch || 10);
    setVal('param-max-steps', currentConfig.max_steps || 2000);
  }
}

function launchRun() {
  var type = document.getElementById('run-type').value;
  var errEl = document.getElementById('launch-error');
  errEl.style.display = 'none';
  var body = {
    type:       type,
    runs:       parseInt(document.getElementById('param-runs').value, 10),
    seed:       parseInt(document.getElementById('param-seed').value, 10),
    policy:     document.getElementById('param-policy').value,
    max_batch:  parseInt(document.getElementById('param-max-batch').value, 10),
    max_steps:  parseInt(document.getElementById('param-max-steps').value, 10),
  };
  if (type === 'headless') {
    body.volatility = parseFloat(document.getElementById('param-volatility').value);
  }
  if (type === 'gone_rogue') {
    body.eyesonly_url = document.getElementById('param-eyesonly-url').value;
  }
  api('POST', '/api/runs', body, function (r) {
    if (r.ok) {
      toast('Job ' + r.data.job_id + ' queued.');
      loadJobs();
      loadStatus();
    } else {
      errEl.textContent = r.error || 'Failed to launch';
      errEl.style.display = 'block';
    }
  });
}

// ── Jobs table ───────────────────────────────────────────────────────────
function loadJobs() {
  api('GET', '/api/jobs', null, function (r) {
    if (!r.ok) return;
    renderJobs(r.data);
  });
}

function renderJobs(jobs) {
  var el = document.getElementById('jobs-container');
  if (!jobs.length) {
    el.innerHTML = '<p class="dim" style="padding:20px">No jobs yet. Launch one from the sidebar.</p>';
    return;
  }
  var html = '<table><thead><tr>';
  html += '<th>ID</th><th>TYPE</th><th>STATUS</th><th>PARAMS</th><th>RESULT</th><th>TIME</th><th></th>';
  html += '</tr></thead><tbody>';
  for (var i = 0; i < jobs.length; i++) {
    html += jobRow(jobs[i]);
  }
  html += '</tbody></table>';
  el.innerHTML = html;
}

function jobRow(j) {
  var statusBadge = '<span class="badge ' + j.status + '">' + j.status.toUpperCase() + '</span>';
  var params = j.params || {};
  var paramStr = 'runs=' + (params.runs||'?') + ' seed=' + (params.seed||0) + ' policy=' + (params.policy||'?');

  var result = '';
  if (j.result_summary) {
    var s = j.result_summary;
    result = '<div class="summary-chips">';
    if (s.total_runs)        result += '<span class="chip">n=' + s.total_runs + '</span>';
    if (s.completion_rate !== undefined) result += '<span class="chip green">✓ ' + s.completion_rate + '%</span>';
    if (s.avg_score !== null && s.avg_score !== undefined) result += '<span class="chip yellow">score ' + s.avg_score + '</span>';
    if (s.avg_floors !== null && s.avg_floors !== undefined) result += '<span class="chip">floors ' + s.avg_floors + '</span>';
    result += '</div>';
  } else if (j.error) {
    result = '<span class="err" style="font-size:11px">' + esc(j.error.substring(0,80)) + '</span>';
  } else if (j.status === 'running') {
    result = '<span class="dim">running…</span>';
  }

  var duration = '';
  if (j.started_at) {
    var end = j.finished_at || (Date.now()/1000);
    var secs = Math.round(end - j.started_at);
    duration = secs + 's';
    if (j.output_file) duration += '<br><span class="dim" style="font-size:10px">' + esc(j.output_file.split('/').pop()) + '</span>';
  } else {
    var wait = Math.round(Date.now()/1000 - j.created_at);
    duration = '<span class="dim">+' + wait + 's</span>';
  }

  var actions = '';
  if (j.status === 'running') {
    actions = '<button class="danger" onclick="cancelJob(\'' + j.job_id + '\')" style="font-size:10px;padding:3px 8px">✕ Cancel</button>';
  }

  return '<tr>' +
    '<td class="mono accent">' + j.job_id + '</td>' +
    '<td>' + j.job_type + '</td>' +
    '<td>' + statusBadge + '</td>' +
    '<td class="dim" style="font-size:11px">' + esc(paramStr) + '</td>' +
    '<td>' + result + '</td>' +
    '<td class="dim" style="font-size:11px">' + duration + '</td>' +
    '<td>' + actions + '</td>' +
    '</tr>';
}

function cancelJob(id) {
  fetch('/api/jobs/' + id, { method: 'DELETE' })
    .then(function (r) { return r.json(); })
    .then(function (r) {
      toast(r.data && r.data.cancelled ? 'Job ' + id + ' cancelled.' : 'Could not cancel.', !r.ok);
      loadJobs();
    });
}

// ── Utility ──────────────────────────────────────────────────────────────
function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Server factory
# ---------------------------------------------------------------------------

class _PortalHTTPServer(http.server.HTTPServer):
    """HTTPServer subclass that carries PortalState."""
    def __init__(self, server_address, handler_class, state: PortalState):
        super().__init__(server_address, handler_class)
        self.state = state


def start_server(
    host: str = "127.0.0.1",
    port: int = 7860,
    output_dir: str = ".",
    eyesonly_url: str = "",
) -> None:
    """
    Start the Sundog developer portal HTTP server (blocking).

    Parameters
    ----------
    host:
        Bind address (default ``127.0.0.1``).
    port:
        Bind port (default ``7860``).
    output_dir:
        Directory where run output JSONL files are written.
    eyesonly_url:
        Optional default URL for Gone Rogue runs.
    """
    config = PortalConfig(eyesonly_url=eyesonly_url)
    state = PortalState(config=config, output_dir=output_dir)
    server = _PortalHTTPServer((host, port), PortalHandler, state)
    url = f"http://{host}:{port}"
    print(f"Sundog Dev Portal running at {url}")
    print(f"  Output dir : {state.output_dir}")
    print(f"  Press Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down portal.")
    finally:
        server.server_close()
