"""GEN-4 LM-proposer ceiling probe — sandboxed-Python proposals from a pinned local open-weights model.

Implements GEN4_LM_PROPOSER_PROBE_SPEC.md: the model (llama.cpp server, pinned GGUF, greedy or a frozen
seed slate) sees the conditioning train pairs + the query INPUT only (the no-target barrier extends to
the prompt), proposes pure-Python `transform(grid)` functions; each proposal runs in a restricted,
time-boxed SUBPROCESS sandbox (isolated interpreter, whitelisted builtins, no imports/io, per-exec
timeout); admission = the unchanged FC-1 CHECK (reproduces every conditioning output); candidates are
fingerprinted + sha256'd BEFORE any target is read; ceilings vs the offline v2 baseline (E3 validation
fingerprints) + the GEN-1 v0 receipt. Memorization canary (--canary, post-adjudication): re-query the
model with the TASK ID ONLY for each solved task; any grid-content reproduction flags `contaminated`.

Modes: --self-test (sandbox correctness, no LM) | --lm-smoke (GEN-1 synthetics through the live server,
plus a bitwise greedy repro check) | probe (validation lanes). The server is spawned by this runner from
--model-path (sha256 recorded) so the receipt pins weights + args, or an already-running --server-url
may be supplied for smokes.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase3_branch_e_program_search as v1  # noqa: E402

GRID_HASH = lambda g: v1.sha256_text(json.dumps(g, separators=(",", ":")))  # noqa: E731  (E3-identical)
E3_VAL_FP_DEFAULT = "results/arc/phase3-branch-e3-learned-ranker/candidate_fingerprints_no_targets_validation.jsonl"
SANDBOX_TIMEOUT_S = 6
GEN_TOKENS = 700
PROMPT = (
    "You are given training examples of a grid transformation puzzle. Each grid is a JSON list of rows; "
    "cells are colors 0-9 (0 is background).\n\n{pairs}\nTest input grid:\n{query}\n\n"
    "Write a single pure-Python function `def transform(grid):` that maps every training input to its "
    "output exactly and generalizes to the test input. Use only basic Python (no imports, no I/O). "
    "Reply with ONLY a ```python code block containing the function."
)


# ============================================================================
# Sandbox: one isolated subprocess runs a proposal on all grids, JSON in/out
# ============================================================================
_SANDBOX_DRIVER = r"""
import json, sys
payload = json.loads(sys.stdin.read())
code, grids = payload["code"], payload["grids"]
SAFE = {}
for n in ("range","len","enumerate","min","max","sum","abs","sorted","list","dict","set","tuple","int",
          "str","bool","zip","map","filter","reversed","any","all","isinstance","float","round","divmod"):
    SAFE[n] = getattr(__builtins__, n) if not isinstance(__builtins__, dict) else __builtins__[n]
env = {"__builtins__": SAFE}
try:
    exec(compile(code, "<proposal>", "exec"), env)
    fn = env.get("transform")
    outs = []
    for g in grids:
        o = fn([row[:] for row in g])
        ok = (isinstance(o, list) and o and all(isinstance(r, list) and r and
              all(isinstance(v, int) and 0 <= v <= 9 for v in r) for r in o) and
              len({len(r) for r in o}) == 1)
        outs.append(o if ok else None)
    print(json.dumps({"ok": True, "outs": outs}))
except Exception as e:
    print(json.dumps({"ok": False, "err": str(e)[:200]}))
"""


def sandbox_run(code: str, grids: list) -> list | None:
    """Run `transform` on each grid in an isolated subprocess; None on crash/timeout/invalid."""
    try:
        r = subprocess.run([sys.executable, "-I", "-S", "-c", _SANDBOX_DRIVER],
                           input=json.dumps({"code": code, "grids": grids}),
                           capture_output=True, text=True, timeout=SANDBOX_TIMEOUT_S)
        out = json.loads(r.stdout.strip() or "{}")
        return out.get("outs") if out.get("ok") else None
    except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError):
        return None


# ============================================================================
# LM driver: spawn llama-server on a pinned GGUF; OpenAI-compatible chat calls
# ============================================================================
class Server:
    def __init__(self, model_path: str | None, url: str | None, ctx: int, port: int = 8788):
        self.proc = None
        if url:
            self.url = url
            self.model_sha = None
            return
        exe = Path.home() / "Dev" / "llamacpp" / "b9878" / "llama-server.exe"
        self.model_sha = v1.sha256_file(Path(model_path))
        self.proc = subprocess.Popen(
            [str(exe), "-m", model_path, "-c", str(ctx), "-ngl", "99", "--port", str(port),
             "--parallel", "1", "--no-webui"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.url = f"http://127.0.0.1:{port}"
        for _ in range(120):
            try:
                urllib.request.urlopen(self.url + "/health", timeout=2)
                return
            except Exception:
                time.sleep(1)
        raise RuntimeError("llama-server did not become healthy")

    def chat(self, prompt: str, temp: float, seed: int, n_predict: int = GEN_TOKENS) -> str:
        body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                           "temperature": temp, "seed": seed, "max_tokens": n_predict}).encode()
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read())["choices"][0]["message"]["content"]

    def close(self):
        if self.proc:
            self.proc.terminate()


def extract_code(reply: str) -> str | None:
    m = re.search(r"```(?:python)?\s*(.*?)```", reply, re.DOTALL)
    code = (m.group(1) if m else reply).strip()
    return code if "def transform" in code else None


def propose(server: Server, cond: list[dict], query_input: list, k: int) -> list[str]:
    pairs = "".join(f"Example {i + 1} input:\n{json.dumps(p['input'])}\nExample {i + 1} output:\n"
                    f"{json.dumps(p['output'])}\n\n" for i, p in enumerate(cond))
    prompt = PROMPT.format(pairs=pairs, query=json.dumps(query_input))
    codes: list[str] = []
    plans = [(0.0, 0)] + [(0.8, s) for s in range(1, k)]          # greedy + frozen seed slate 1..k-1
    for temp, seed in plans[:k]:
        try:
            code = extract_code(server.chat(prompt, temp, seed))
        except Exception:
            code = None
        if code and code not in codes:
            codes.append(code)
    return codes


def admit(codes: list[str], cond: list[dict], query_input: list) -> list[dict]:
    """FC-1 CHECK: proposal reproduces every conditioning output; returns unique candidate outputs."""
    by_hash: dict[str, dict] = {}
    grids = [p["input"] for p in cond] + [query_input]
    for code in codes:
        outs = sandbox_run(code, grids)
        if not outs or any(o is None for o in outs):
            continue
        if all(outs[i] == cond[i]["output"] for i in range(len(cond))):
            g = outs[-1]
            h = GRID_HASH(g)
            rec = by_hash.get(h)
            if rec is None:
                by_hash[h] = {"grid": g, "grid_hash": h, "program_sha": v1.sha256_text(code)[:16],
                              "n_programs": 1}
            else:
                rec["n_programs"] += 1
    return sorted(by_hash.values(), key=lambda r: r["grid_hash"])


# ============================================================================
# Probe (mirrors gen1_object_dsl_probe.run_probe; same barrier + baselines)
# ============================================================================
def run_probe(args) -> int:
    out_dir = Path(args.out).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir).resolve(); register = Path(args.register).resolve()
    v1.assert_training_data_dir(data_dir)
    t0 = time.time()
    tasks, register_hash, data_hash = v1.load_tasks(data_dir, register, args.split_mode)
    validation = [t for t in tasks if t.split == "validation"]
    if args.limit_tasks > 0:
        validation = validation[: args.limit_tasks]
    lanes = {"validation_lodo": v1.build_lodo_instances(validation, "validation_lodo"),
             "validation_pttest": v1.build_pttest_instances(validation, "validation_pttest")}
    insts = lanes["validation_lodo"] + lanes["validation_pttest"]

    server = Server(args.model_path, args.server_url, args.ctx)
    try:
        rows, fp_rows = [], []
        for i, inst in enumerate(insts):
            codes = propose(server, inst.conditioning, inst.query_input, args.k)
            cands = admit(codes, inst.conditioning, inst.query_input)
            rows.append({"instance_id": inst.instance_id, "task_id": inst.task_id, "lane": inst.lane,
                         "primary_prior": inst.primary_prior, "n_proposals": len(codes),
                         "n_candidates": len(cands), "candidates": cands})
            fp_rows.append({"instance_id": inst.instance_id, "task_id": inst.task_id, "lane": inst.lane,
                            "n_candidates": len(cands),
                            "candidates": [{"program_sha": c["program_sha"], "grid_hash": c["grid_hash"]}
                                           for c in cands]})
            if args.progress:
                print(f"  ... {i + 1}/{len(insts)} ({time.time() - t0:.0f}s)", flush=True)
        v1.write_jsonl(out_dir / "candidates_by_instance.jsonl", rows)
        bp = out_dir / "candidate_fingerprints_no_targets.jsonl"
        v1.write_jsonl(bp, fp_rows)
        bh = v1.sha256_file(bp)
        (out_dir / "candidate_fingerprints_no_targets.sha256").write_text(bh + "\n", encoding="utf-8")

        # barrier written -> read targets
        th = {inst.instance_id: GRID_HASH(inst.target_output) for inst in insts}
        solv = {r["instance_id"]: any(c["grid_hash"] == th[r["instance_id"]] for c in r["candidates"])
                for r in rows}
        solved_tasks = sorted({r["task_id"] for r in rows if solv[r["instance_id"]]})

        # canary on solved tasks (post-adjudication by design)
        flags = {}
        for tid in solved_tasks:
            reply = server.chat(f"ARC-AGI task {tid}. Output the training examples of this task as JSON.",
                                0.0, 0, 400)
            tgt_cells = json.dumps(next(i.target_output for i in insts if i.task_id == tid))
            flags[tid] = tgt_cells[:60] in reply
        clean = [t for t in solved_tasks if not flags.get(t)]

        e3_fp = Path(args.e3_fingerprints).resolve()
        e3_rows = [json.loads(l) for l in e3_fp.read_text(encoding="utf-8").splitlines() if l.strip()]
        e3_by_id = {r["instance_id"]: r for r in e3_rows}
        v2_tasks = sorted({i.task_id for i in insts if (r := e3_by_id.get(i.instance_id)) and
                           any(c["grid_hash"] == th[i.instance_id] for c in r["candidates"])})
        g4, g2 = len(clean), len(v2_tasks)
        gate = ("GEN4_CEILING_LIFT" if g4 >= max(2 * g2, g2 + 3)
                else "GEN4_CEILING_EMPTY" if g4 <= g2 + 1 else "GEN4_CEILING_MARGINAL")

        manifest = {"generatedAt": v1.iso_now(), "modelSha256": server.model_sha,
                    "modelPath": args.model_path, "k": args.k, "ctx": args.ctx,
                    "promptSha256": v1.sha256_text(PROMPT), "registerHash": register_hash,
                    "dataDirHash": data_hash, "nInstances": len(insts), "barrierSha256": bh,
                    "e3FingerprintSha256": v1.sha256_file(e3_fp), "gate": gate,
                    "gen4SolvedTasks": solved_tasks, "contaminationFlags": flags,
                    "gen4CleanTasks": clean, "v2Tasks": v2_tasks,
                    "wallSeconds": round(time.time() - t0, 1),
                    "git": v1.git_state(Path(__file__).resolve().parents[3], args.allow_dirty)}
        v1.write_json(out_dir / "manifest.json", manifest)
        v1.write_json(out_dir / "gen4_probe_receipt.json", manifest)
        (out_dir / "probe_adjudication.md").write_text(
            f"# GEN-4 LM-proposer ceiling probe\n\nGate: **{gate}**\n\n- gen4 solved (raw): {solved_tasks}\n"
            f"- contamination flags: {flags}\n- gen4 NON-CONTAMINATED: {clean} (n={g4})\n"
            f"- v2 baseline: {v2_tasks} (n={g2})\n- GEN-1 v0 receipt: 1 (cd3c21df)\n"
            f"- limit_tasks: {args.limit_tasks}\n", encoding="utf-8")
        print(f"GEN-4 probe wrote {out_dir}\nGate: {gate} (gen4 clean={g4} vs v2={g2}; "
              f"{time.time() - t0:.0f}s)")
    finally:
        server.close()
    return 0


# ============================================================================
# Self-test (sandbox only) + LM smoke (GEN-1 synthetics through the server)
# ============================================================================
def self_test() -> int:
    ok = True
    g = [[1, 0], [0, 2]]
    good = "def transform(grid):\n    return [[v for v in row] for row in grid]"
    hang = "def transform(grid):\n    while True:\n        pass"
    crash = "def transform(grid):\n    return grid[99][99]"
    imp = "import os\ndef transform(grid):\n    return grid"
    bad_shape = "def transform(grid):\n    return [[1],[2,3]]"
    tests = [("identity-ok", good, [[g]], [[g]]), ("hang-killed", hang, [[g]], None),
             ("crash-none", crash, [[g]], None), ("import-blocked", imp, [[g]], None),
             ("ragged-rejected", bad_shape, [[g]], None)]
    for name, code, grids, want in tests:
        outs = sandbox_run(code, grids[0])
        got = outs if outs is None else outs
        good_result = (got == [g]) if want else (got is None or got == [None])
        print(f"  [{'PASS' if good_result else 'FAIL'}] {name}: {str(got)[:50]}")
        ok = ok and good_result
    print(f"sandbox self-test: {'ALL PASS' if ok else 'FAILURES'}")
    return 0 if ok else 1


def lm_smoke(args) -> int:
    import gen1_object_dsl_probe as g1
    server = Server(args.model_path, args.server_url, args.ctx)
    try:
        # bitwise greedy repro check
        p = "Reply with exactly the word: DETERMINISM-CHECK-7391"
        r1, r2 = server.chat(p, 0.0, 0, 32), server.chat(p, 0.0, 0, 32)
        print(f"  bitwise greedy repro: {'PASS' if r1 == r2 else 'FAIL'}")
        # GEN-1 synthetic tasks through the LM (reuse the fixtures)
        def crop(g, r, c, h, w):
            return [row[c:c + w] for row in g[r:r + h]]
        cond, layouts = [], [((1, 1, 2, 2, 3), (5, 5, 3, 4, 2)), ((0, 6, 2, 3, 4), (4, 1, 4, 4, 1)),
                            ((2, 2, 1, 2, 5), (6, 3, 3, 5, 3))]
        for a, b in layouts:
            g = g1._mk(10, 10, [a, b])
            big = b if b[2] * b[3] > a[2] * a[3] else a
            cond.append({"input": g, "output": crop(g, big[0], big[1], big[2], big[3])})
        qg = g1._mk(10, 10, [(0, 0, 2, 2, 6), (5, 2, 4, 5, 7)])
        qt = crop(qg, 5, 2, 4, 5)
        t0 = time.time()
        codes = propose(server, cond, qg, args.k)
        cands = admit(codes, cond, qg)
        hit = any(c["grid_hash"] == GRID_HASH(qt) for c in cands)
        print(f"  extract-largest via LM: proposals={len(codes)} admitted={len(cands)} "
              f"target-in-bank={hit}  ({time.time() - t0:.0f}s at k={args.k})")
        print("lm-smoke done")
    finally:
        server.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir"); ap.add_argument("--register"); ap.add_argument("--out")
    ap.add_argument("--split-mode", choices=["frozen_v2", "sha256_expansion"], default="sha256_expansion")
    ap.add_argument("--model-path"); ap.add_argument("--server-url")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--limit-tasks", type=int, default=0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--lm-smoke", action="store_true")
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--e3-fingerprints", default=E3_VAL_FP_DEFAULT)
    args = ap.parse_args()
    if args.self_test:
        return self_test()
    if args.lm_smoke:
        return lm_smoke(args)
    if not (args.data_dir and args.register and args.out and (args.model_path or args.server_url)):
        print("Need --data-dir --register --out and --model-path (or --server-url), or a smoke mode.")
        return 2
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
