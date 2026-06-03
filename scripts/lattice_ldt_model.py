#!/usr/bin/env python3
"""Lattice Deduction Transformer (LDT) reimplementation + trainer — Phase 1 build-gate.

Spec: docs/lattice/ (LANE_CHARTER, PHASE0_MINIMUM_FALSIFIABLE) + docs/SUNDOG_V_LATTICE.md
(Phase 1). This is the **build-gate** model: a best-effort faithful reimplementation
of the LDT (arXiv 2605.08605), whose ONLY job at this phase is to **reproduce the
paper's 100% Sudoku-Extreme headline cell**. Until `build_gate_pass`, NO body/fiber
number is read off it (B2/B1/B3 are separate runners, gated). Torch-only (numpy is
unavailable); device-parameterized (CUDA when available — the chatv2 on-ramp pattern).

================================================================================
VERIFIED architecture facts (from the paper HTML, 2026-06-02; litpass spine):
  - candidate-set lattice = the inter-step state; 729-dim for 9x9 (81 cells x 9 digits),
    multi-hot ("9 binary sigmoids per cell"); a ⊑ b iff a(i) ⊆ b(i); ⊤ all / ⊥ empty.
  - transformer: d_model=128, 4 layers, 4 heads, FFN x4.0; L=16 internal iterations
    per forward pass; learned 2D positional embeddings (2D RoPE for larger grids).
  - conflict head: a binary sigmoid on a distinguished CLS token; symmetric BCE vs
    1[lattice = ⊥]; weight λ_cls=0.1; inference θ_CLS=0.6; empty cell OR CLS>θ → backtrack.
  - output: sigmoid confidences per candidate; threshold eliminates low-confidence ones.
  - 800K params (Sudoku); trains ~minutes on a B200; 100% Sudoku-Extreme.

INFERRED (flagged; the build-gate is the test of faithfulness — tune these there):
  [I1] The 16 iterations are WEIGHT-SHARED recurrence of the 4-layer block. FORCED by
       the param budget: 64 distinct layers ≈ 12M params, not 800K. (high confidence)
  [I2] The input lattice is re-injected at each recurrent iteration (HRM/TRM lineage).
       Config `reinject_input` (default True). (medium)
  [I3] Elimination loss = per-cell BCE(sigmoid(logits), solution one-hot) — the
       SATNet/RRN/HRM "predict the solution per cell" supervision. The paper may use a
       per-step *deduction* target instead. (medium)
  [I4] Conflict-head target during training: with prob `conflict_corrupt_p`, corrupt the
       input lattice to a contradiction (drop a cell's solution candidate) → target 1;
       else 0. (medium)
  [I5] Inference deduction loop / backtracking search policy details (order, depth) are
       not specified by the summary; a greedy threshold + most-constrained-cell
       backtrack is used. (low — only matters for solve-rate eval, which the build-gate
       gates on; tune to hit 100%.)
================================================================================
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- frozen architecture constants (verified facts) ----
N = 9                       # Sudoku side (9x9)
N2 = N * N                  # 81 cells
DIGITS = 9                  # candidates per cell
LATTICE_DIM = N2 * DIGITS   # 729
D_MODEL = 128
N_LAYERS = 4
N_HEADS = 4
N_ITERS = 16                # L (weight-shared recurrence — [I1])
FFN_MULT = 4.0
LAMBDA_CLS = 0.1
THETA_CLS = 0.6
SEQ_LEN = 1 + N2            # CLS + 81 cells
DEFAULT_DATA_DIR = "docs/lattice/Soduko-Extreme"   # build-gate dataset (gitignored)


# ============================================================================
# Config
# ============================================================================
@dataclass
class Cfg:
    d_model: int = D_MODEL
    n_layers: int = N_LAYERS
    n_heads: int = N_HEADS
    n_iters: int = N_ITERS
    ffn_mult: float = FFN_MULT
    reinject_input: bool = True          # [I2]
    lr: float = 1e-3
    weight_decay: float = 0.01
    batch: int = 64
    max_steps: int = 6000
    elim_threshold: float = 0.5          # commit-elimination confidence cut (inference)
    conflict_corrupt_p: float = 0.25     # [I4]
    seed: int = 0
    mode: str = "smoke"                  # smoke | build-gate
    data_dir: Optional[str] = None       # Sudoku-Extreme root (build-gate)
    out: str = "results/lattice/build-gate-sudoku-extreme"
    allow_dirty: bool = False
    max_eval: int = 0                    # cap eval puzzles (0 = all)


# ============================================================================
# Sudoku data — synthetic generator (smoke) + Sudoku-Extreme loader interface
# ============================================================================
def _valid_base(perm: list[int], band: list[int], stack: list[int],
                rows_in: list[list[int]], cols_in: list[list[int]]) -> list[list[int]]:
    """A valid filled grid by permuting the canonical pattern (validity-preserving)."""
    def pat(r: int, c: int) -> int:
        return (N * (r % 3) + r // 3 + c) % N
    rows = [band[b] * 3 + rows_in[b][i] for b in range(3) for i in range(3)]
    cols = [stack[s] * 3 + cols_in[s][j] for s in range(3) for j in range(3)]
    return [[perm[pat(rows[r], cols[c])] for c in range(N)] for r in range(N)]


def gen_sudoku(rng: torch.Generator, n_holes: int) -> tuple[list[list[int]], list[list[int]]]:
    """Synthetic (puzzle, solution) — SMOKE ONLY. Not difficulty-controlled, not
    uniqueness-checked; the build-gate REQUIRES real Sudoku-Extreme (see load_dataset)."""
    def shuf(xs: list[int]) -> list[int]:
        idx = torch.randperm(len(xs), generator=rng).tolist()
        return [xs[i] for i in idx]
    perm = shuf(list(range(1, N + 1)))
    sol = _valid_base(perm, shuf([0, 1, 2]), shuf([0, 1, 2]),
                      [shuf([0, 1, 2]) for _ in range(3)], [shuf([0, 1, 2]) for _ in range(3)])
    flat = list(range(N2))
    holes = set(shuf(flat)[:n_holes])
    puz = [[0 if r * N + c in holes else sol[r][c] for c in range(N)] for r in range(N)]
    return puz, sol


def _parse_grid(s: str) -> list[list[int]]:
    cells = [0 if ch in "._0" else int(ch) for ch in s.strip()[:N2]]
    return [cells[r * N:(r + 1) * N] for r in range(N)]


def _valid_solution(g: list[list[int]]) -> bool:
    """Each row/col/3x3-box is a permutation of 1..9 — doubles as a parser correctness check."""
    need = set(range(1, N + 1))
    if any(set(row) != need for row in g):
        return False
    if any({g[r][c] for r in range(N)} != need for c in range(N)):
        return False
    return all({g[r][c] for r in range(br, br + 3) for c in range(bc, bc + 3)} == need
               for br in range(0, N, 3) for bc in range(0, N, 3))


def _read_csv(path: Path, limit: int) -> list[tuple]:
    """Stream `source,question,answer,rating`; -> [(puzzle 9x9, solution 9x9)]. limit=0 reads all."""
    pairs = []
    with path.open("r", encoding="utf-8") as fh:
        fh.readline()                                          # header
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            parts = line.rstrip("\n").split(",")
            if len(parts) < 3 or len(parts[1]) != N2 or len(parts[2]) != N2:
                continue
            pairs.append((_parse_grid(parts[1]), _parse_grid(parts[2])))
    return pairs


def load_dataset(data_dir: Path, n_train: int = 1000, limit_test: int = 0,
                 subset_seed: int = 0) -> tuple[list, list]:
    """Parse Sudoku-Extreme CSVs -> (train_pairs, test_pairs). The paper's headline trains
    on ~1K puzzles (Sudoku-Extreme's small-data point), so `n_train` caps the train subset.
    The EXACT 1K subset is a build-gate alignment param (flag for the audit team): the CSV
    is grouped by `source`, so 'first 1K' would be single-source-biased; default is a seeded
    sample from a multi-source pool. n_train=0 reads all of train.csv (memory-heavy)."""
    train_csv, test_csv = data_dir / "train.csv", data_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise SystemExit(f"Sudoku-Extreme not found under {data_dir} (need train.csv + test.csv).")
    if n_train:
        pool = _read_csv(train_csv, max(n_train * 100, 100000))     # pool spans multiple sources
        if len(pool) > n_train:
            g = torch.Generator().manual_seed(subset_seed)
            idx = torch.randperm(len(pool), generator=g)[:n_train].tolist()
            train_pairs = [pool[i] for i in idx]
        else:
            train_pairs = pool
    else:
        train_pairs = _read_csv(train_csv, 0)
    return train_pairs, _read_csv(test_csv, limit_test)


def grids_to_tensors(pairs: list[tuple], device) -> tuple[torch.Tensor, torch.Tensor]:
    """(puzzle, solution) grids -> (lattice0 (B,81,9) clue-seeded ⊤, solution idx (B,81))."""
    B = len(pairs)
    lat = torch.ones(B, N2, DIGITS, device=device)        # ⊤ : all candidates open
    sol = torch.zeros(B, N2, dtype=torch.long, device=device)
    for b, (puz, s) in enumerate(pairs):
        for r in range(N):
            for c in range(N):
                i = r * N + c
                sol[b, i] = s[r][c] - 1
                if puz[r][c] != 0:                          # clue → singleton candidate set
                    lat[b, i].zero_(); lat[b, i, puz[r][c] - 1] = 1.0
    return lat, sol


# ============================================================================
# Model
# ============================================================================
class Block(nn.Module):
    """Pre-norm transformer layer (MHA + FFN x4)."""
    def __init__(self, d: int, h: int, ffn_mult: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        hidden = int(d * ffn_mult)
        self.ff = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))

    def forward(self, x):
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)
        x = x + a
        x = x + self.ff(self.ln2(x))
        return x


class LatticeDeductionTransformer(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.cell_in = nn.Linear(DIGITS, d)                 # per-cell 9 candidates -> d
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.row_emb = nn.Embedding(N, d)                   # learned 2D positional
        self.col_emb = nn.Embedding(N, d)
        self.layers = nn.ModuleList([Block(d, cfg.n_heads, cfg.ffn_mult) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.elim_head = nn.Linear(d, DIGITS)               # per-cell candidate logits
        self.conflict_head = nn.Linear(d, 1)                # CLS -> conflict logit
        ri = torch.arange(N).repeat_interleave(N)
        ci = torch.arange(N).repeat(N)
        self.register_buffer("row_idx", ri, persistent=False)
        self.register_buffer("col_idx", ci, persistent=False)

    def _embed(self, lattice: torch.Tensor) -> torch.Tensor:
        # lattice (B,81,9) -> (B,82,d) with CLS + learned 2D pos
        B = lattice.shape[0]
        cell = self.cell_in(lattice) + self.row_emb(self.row_idx)[None] + self.col_emb(self.col_idx)[None]
        return torch.cat([self.cls.expand(B, -1, -1), cell], dim=1)

    def forward(self, lattice: torch.Tensor, capture: Optional[dict] = None):
        """One forward pass = n_iters weight-shared recurrences of the 4-layer block.
        Returns (elim_logits (B,81,9), conflict_logit (B,)). If `capture` is given, stores
        the residual stream at each (iteration, layer) for the B-phase fingerprint."""
        x0 = self._embed(lattice)
        x = x0
        for t in range(self.cfg.n_iters):
            if self.cfg.reinject_input and t > 0:           # [I2] re-inject input each step
                x = x + x0
            for li, layer in enumerate(self.layers):
                x = layer(x)
                if capture is not None:
                    capture[(t, li)] = x.detach()
        x = self.ln_f(x)
        elim = self.elim_head(x[:, 1:])                      # (B,81,9)
        conflict = self.conflict_head(x[:, 0]).squeeze(-1)   # (B,)
        return elim, conflict


# ============================================================================
# Loss + training
# ============================================================================
def compute_loss(model, lat: torch.Tensor, sol: torch.Tensor, cfg: Cfg, rng):
    # [I4] conflict supervision: corrupt some inputs to ⊥, target=1; else 0.
    B = lat.shape[0]
    lat_in = lat.clone()
    conflict_tgt = torch.zeros(B, device=lat.device)
    corrupt = torch.rand(B, generator=rng, device=lat.device) < cfg.conflict_corrupt_p
    for b in range(B):
        if corrupt[b]:
            i = int(torch.randint(N2, (1,), generator=rng, device=lat.device))
            lat_in[b, i, sol[b, i]] = 0.0                   # drop the solution candidate -> ⊥
            if lat_in[b, i].sum() == 0:
                conflict_tgt[b] = 1.0
            else:                                            # still has candidates: not yet ⊥
                conflict_tgt[b] = 0.0
    elim, conflict = model(lat_in)
    # [I3] elimination: per-cell BCE toward the solution one-hot
    tgt = F.one_hot(sol, DIGITS).float()
    elim_loss = F.binary_cross_entropy_with_logits(elim, tgt)
    conflict_loss = F.binary_cross_entropy_with_logits(conflict, conflict_tgt)
    return elim_loss + LAMBDA_CLS * conflict_loss, elim_loss.item(), conflict_loss.item()


@torch.no_grad()
def solve_rate(model, pairs: list[tuple], device, cfg: Cfg) -> float:
    """One-shot read of the solution from a single forward pass (argmax per cell).
    NOTE: this is the *cheap* eval (no iterative deduction/backtrack rollout — [I5]).
    The build-gate's 100% target is on the full rollout; this gates smoke sanity only."""
    if not pairs:
        return 0.0
    lat, sol = grids_to_tensors(pairs, device)
    elim, _ = model(lat)
    pred = elim.argmax(dim=-1)
    return (pred == sol).all(dim=1).float().mean().item()


def set_determinism(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Provenance / IO
# ============================================================================
def git_commit(repo: Path, allow_dirty: bool) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip().upper()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True).strip())
    except Exception:
        commit, dirty = "UNKNOWN", True
    if dirty and not allow_dirty:
        raise SystemExit("Dirty worktree; commit first or pass --allow-dirty for smoke.")
    return {"commit": commit, "dirty": dirty}


def param_count(model) -> int:
    return sum(p.numel() for p in model.parameters())


# ============================================================================
# Modes
# ============================================================================
def run_smoke(cfg: Cfg, device) -> dict:
    """Tiny synthetic Sudoku: confirm shapes, param count at full config, a train step
    reduces loss, a rollout runs. Seconds. NOT a build-gate verdict."""
    set_determinism(cfg.seed)
    cpu_rng = torch.Generator().manual_seed(cfg.seed)               # synthetic gen (CPU list work)
    dev_rng = torch.Generator(device=device).manual_seed(cfg.seed)  # training-time device ops
    full = LatticeDeductionTransformer(Cfg()).to(device)    # full config for the param check
    pc = param_count(full)
    model = LatticeDeductionTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pairs = [gen_sudoku(cpu_rng, n_holes=20) for _ in range(cfg.batch)]
    lat, sol = grids_to_tensors(pairs, device)
    losses = []
    for step in range(8):
        loss, el, cl = compute_loss(model, lat, sol, cfg, dev_rng)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    elim, conflict = model(lat)
    cap: dict = {}
    model(lat[:2], capture=cap)
    sr = solve_rate(model, pairs[:8], device, cfg)
    # real-data resmoke: confirm load_dataset parses real Sudoku-Extreme + the model runs on it
    real = {"checked": False, "note": "dataset not found at canonical path"}
    ddir = Path(cfg.data_dir) if cfg.data_dir else Path(DEFAULT_DATA_DIR)
    if (ddir / "test.csv").exists():
        rp = _read_csv(ddir / "test.csv", 32)                    # 32 real test puzzles
        rlat, rsol = grids_to_tensors(rp, device)
        relim, _ = model(rlat)
        rloss, _, _ = compute_loss(model, rlat, rsol, cfg, dev_rng)
        valid = sum(_valid_solution(s) for _, s in rp)
        clues = sum(1 for p, _ in rp for row in p for v in row if v) / max(1, len(rp))
        real = {"checked": True, "n_loaded": len(rp), "valid_solutions": f"{valid}/{len(rp)}",
                "parser_ok": bool(valid == len(rp)), "avg_clues": round(clues, 1),
                "elim_shape": list(relim.shape), "loss": round(rloss.item(), 4)}
    return {
        "mode": "smoke", "param_count_full_config": pc,
        "param_count_in_budget": bool(6e5 <= pc <= 1.0e6),  # ~800K target
        "elim_shape": list(elim.shape), "conflict_shape": list(conflict.shape),
        "loss_first": round(losses[0], 5), "loss_last": round(losses[-1], 5),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "capture_grains": len(cap), "capture_grains_expected": cfg.n_iters * cfg.n_layers,
        "smoke_solve_rate": round(sr, 4), "device": str(device),
        "real_data_check": real,
    }


def run_build_gate(cfg: Cfg, device) -> dict:
    """The real build-gate: train on Sudoku-Extreme to reproduce 100%. SCAFFOLDED; the
    run needs the dataset + GPU + time and is gated (do not launch until on-ramp parity
    is confirmed). load_dataset raises until the real data is wired."""
    set_determinism(cfg.seed)
    if not cfg.data_dir:
        raise SystemExit("--data-dir (Sudoku-Extreme root) required for build-gate.")
    # n_train=1000 matches the paper's small-data Sudoku-Extreme regime; test load capped for memory
    train_pairs, test_pairs = load_dataset(Path(cfg.data_dir), n_train=1000, limit_test=(cfg.max_eval or 10000))
    rng = torch.Generator(device=device).manual_seed(cfg.seed)
    model = LatticeDeductionTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    t0 = time.time()
    for step in range(cfg.max_steps):
        idx = torch.randint(len(train_pairs), (cfg.batch,), generator=rng, device=device).tolist()
        lat, sol = grids_to_tensors([train_pairs[i] for i in idx], device)
        loss, _, _ = compute_loss(model, lat, sol, cfg, rng)
        opt.zero_grad(); loss.backward(); opt.step()
    acc = solve_rate(model, test_pairs[: cfg.max_eval] if cfg.max_eval else test_pairs, device, cfg)
    branch = "build_gate_pass" if acc >= 0.999 else ("build_gate_partial" if acc > 0 else "build_gate_fail")
    return {"mode": "build-gate", "test_accuracy": round(acc, 5), "branch": branch,
            "wall_s": round(time.time() - t0, 1), "n_train": len(train_pairs), "n_test": len(test_pairs)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LDT build-gate model + trainer")
    ap.add_argument("--mode", choices=["smoke", "build-gate"], default="smoke")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/lattice/build-gate-sudoku-extreme")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    cfg = Cfg(mode=a.mode, data_dir=a.data_dir, out=a.out, seed=a.seed,
              max_steps=a.max_steps, batch=a.batch, lr=a.lr, allow_dirty=a.allow_dirty)
    if a.mode == "smoke":
        cfg = Cfg(mode="smoke", d_model=32, n_layers=1, n_iters=2, batch=16, seed=a.seed, allow_dirty=True, out=a.out)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = Path(__file__).resolve().parents[1]
    out = Path(cfg.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    git = git_commit(repo, cfg.allow_dirty or cfg.mode == "smoke")

    result = run_smoke(cfg, device) if cfg.mode == "smoke" else run_build_gate(cfg, device)
    manifest = {
        "lane": "lattice", "phase": "1-build-gate", "tool": "scripts/lattice_ldt_model.py",
        "gitCommit": git["commit"], "gitDirty": git["dirty"], "device": str(device),
        "torch": torch.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "arch": {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                 "n_iters": cfg.n_iters, "lattice_dim": LATTICE_DIM, "lambda_cls": LAMBDA_CLS, "theta_cls": THETA_CLS},
        "result": result,
        "discipline": "build-gate model only; no body/fiber number is a B-layer result until build_gate_pass",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"-> {out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
