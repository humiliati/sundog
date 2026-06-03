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
import random
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
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
    conflict_corrupt_p: float = 0.25     # [I4]
    aug_factor: int = 50                 # augmented pool = aug_factor x 1000 (symmetry aug; breaks 1K overfit)
    compile_model: bool = False          # torch.compile (reduce-overhead/CUDA graphs); needs GPU CC>=7.0 (A100/H100)
    # I5 rollout (contract: docs/lattice/PHASE1_I5_ROLLOUT_CONTRACT.md)
    theta_drop: float = 0.5              # eliminate when drop_conf = 1 - sigmoid(logit) >= theta_drop
    theta_cls: float = THETA_CLS         # conflict when sigmoid(conflict_logit) > theta_cls (0.6)
    max_deduction_steps: int = 64        # per-node deduction cap
    max_search_nodes: int = 4096         # per-puzzle DFS node cap
    seed: int = 0
    mode: str = "smoke"                  # smoke | build-gate
    stage: str = "all"                   # train | eval | all
    data_dir: Optional[str] = None       # Sudoku-Extreme root (build-gate)
    out: str = "results/lattice/build-gate-sudoku-extreme"
    allow_dirty: bool = False
    max_eval: int = 0                    # cap eval puzzles (0 = all)
    resume: str = ""                     # "" | latest | checkpoint path
    checkpoint_every: int = 0            # save every N train steps (0 = final only)
    log_every: int = 100                 # train_log.jsonl cadence
    eval_every: int = 0                  # cheap one-shot diagnostic cadence (0 = off)
    eval_sample: int = 64                # first N test puzzles for eval_every diagnostic


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


def augment_grid(puz, sol, rng):
    """Validity-preserving Sudoku symmetry augmentation (digit permute + band/within row+col
    permutes + transpose). rng: random.Random. Breaks 1K-set memorization (HRM/TRM approach);
    the augmented grid stays a valid Sudoku, so the clue/solution relation is preserved."""
    dperm = list(range(DIGITS)); rng.shuffle(dperm)                       # digit v(1-9) -> dperm[v-1]+1
    def band_order():                                                     # band-respecting row/col perm
        bands = [0, 1, 2]; rng.shuffle(bands)
        within = [[0, 1, 2] for _ in range(3)]
        for w in within:
            rng.shuffle(w)
        return [bands[p // 3] * 3 + within[bands[p // 3]][p % 3] for p in range(N)]
    rows, cols = band_order(), band_order()
    transpose = rng.random() < 0.5
    def xf(g):
        out = [[g[rows[r]][cols[c]] for c in range(N)] for r in range(N)]
        if transpose:
            out = [list(z) for z in zip(*out)]
        return [[(dperm[v - 1] + 1 if v else 0) for v in row] for row in out]
    return xf(puz), xf(sol)


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
    """(puzzle, solution) grids -> (lattice0 (B,81,9) clue-seeded ⊤, solution idx (B,81)).
    Vectorized: build (B,81) puzzle/solution tensors in one shot, then the lattice with tensor
    ops (clue cells -> singleton, blanks -> all open) - no per-cell GPU scalar assignment."""
    puz = torch.tensor([[v for row in p for v in row] for p, _ in pairs], dtype=torch.long, device=device)  # (B,81)
    sln = torch.tensor([[v for row in s for v in row] for _, s in pairs], dtype=torch.long, device=device)  # (B,81)
    sol = sln - 1                                                          # (B,81) 0..8
    clue = (puz != 0)                                                      # (B,81)
    clue_digit = (puz - 1).clamp(min=0)                                    # (B,81); 0 for blanks (unused)
    lat = torch.ones(puz.shape[0], N2, DIGITS, device=device)             # ⊤ : all open
    lat = torch.where(clue[:, :, None], torch.zeros_like(lat), lat)       # zero clue cells
    lat = lat + F.one_hot(clue_digit, DIGITS).to(lat.dtype) * clue[:, :, None].to(lat.dtype)   # set clue digit
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
    # [I4] conflict supervision (vectorized — no per-item Python loop / GPU syncs): with prob
    # conflict_corrupt_p drop the true digit at a random cell; target=1 iff that empties the cell.
    B = lat.shape[0]
    dev = lat.device
    lat_in = lat.clone()
    b_idx = torch.arange(B, device=dev)
    corrupt = torch.rand(B, generator=rng, device=dev) < cfg.conflict_corrupt_p   # (B,)
    cells = torch.randint(N2, (B,), generator=rng, device=dev)                     # (B,) random cell/sample
    true_digit = sol[b_idx, cells]                                                 # (B,) solution digit there
    cb = corrupt.nonzero(as_tuple=True)[0]
    lat_in[b_idx[cb], cells[cb], true_digit[cb]] = 0.0                             # drop true digit (corrupted only)
    cell_sums = lat_in[b_idx, cells].sum(dim=1)                                    # candidates left in chosen cell
    conflict_tgt = (corrupt & (cell_sums == 0)).float()                           # (B,) ⊥ iff emptied
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


def _cfg_json(cfg: Cfg) -> dict:
    return {k: v for k, v in vars(cfg).items() if isinstance(v, (str, int, float, bool)) or v is None}


def _checkpoint_path(out: Path, resume: str) -> Path:
    if resume == "latest":
        return out / "checkpoint_latest.pt"
    return Path(resume)


def _save_checkpoint(out: Path, model, opt, step: int, cfg: Cfg, rng, *, keep_step_copy: bool) -> Path:
    ckpt = {
        "rolloutVersion": ROLLOUT_VERSION,
        "step": step,
        "cfg": _cfg_json(cfg),
        "model": getattr(model, "_orig_mod", model).state_dict(),   # unwrap torch.compile
        "optimizer": opt.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "train_rng_state": rng.get_state(),
    }
    latest = out / "checkpoint_latest.pt"
    torch.save(ckpt, latest)
    if keep_step_copy:
        step_path = out / f"checkpoint_step_{step:08d}.pt"
        torch.save(ckpt, step_path)
    return latest


def _load_checkpoint(out: Path, resume: str, model, opt, rng, device) -> dict:
    path = _checkpoint_path(out, resume)
    if not path.exists():
        raise SystemExit(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    getattr(model, "_orig_mod", model).load_state_dict(ckpt["model"])   # unwrap torch.compile
    opt.load_state_dict(ckpt["optimizer"])
    if ckpt.get("torch_rng_state") is not None:
        torch.set_rng_state(ckpt["torch_rng_state"].cpu())
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state_all") is not None:
        # map_location=device moved these onto cuda; cuda RNG state must be CPU ByteTensors
        torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state_all"]])
    if ckpt.get("train_rng_state") is not None:
        rng.set_state(ckpt["train_rng_state"].cpu())
    return {"path": str(path), "step": int(ckpt.get("step", 0)),
            "rolloutVersion": ckpt.get("rolloutVersion")}


def _append_jsonl(path: Path, row: dict):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


# ============================================================================
# I5 iterative-deduction rollout (contract: docs/lattice/PHASE1_I5_ROLLOUT_CONTRACT.md)
# ============================================================================
ROLLOUT_VERSION = "phase1_i5_v1"
PUZZLE_STOP_REASONS = ["solved_exact", "solved_valid_wrong", "node_cap_exceeded",
                       "unsolved_no_branches", "unsolved_conflict_exhausted"]


def _valid_grid_flat(g1to9: list[int]) -> bool:
    return _valid_solution([g1to9[r * N:(r + 1) * N] for r in range(N)])


def _lattice_key(lat: torch.Tensor) -> bytes:
    return bytes(lat.to(torch.uint8).flatten().tolist())


@torch.no_grad()
def _deduce_node(model, lat, clue_t, cfg, diag, ans_t=None):
    """Model-driven monotone narrowing to fixpoint within one node (contract §4) with the
    vectorized last-candidate guard. Returns (lattice, status, last_keep)."""
    last_keep = None
    for _ in range(cfg.max_deduction_steps):
        elim_logit, conflict_logit = model(lat.unsqueeze(0))
        keep = torch.sigmoid(elim_logit[0]); last_keep = keep
        if bool((lat.sum(dim=1) == 0).any()):
            return lat, "empty_cell_conflict", keep
        if torch.sigmoid(conflict_logit[0]).item() > cfg.theta_cls:
            return lat, "model_conflict", keep
        drop = 1.0 - keep
        open_mask = lat == 1.0
        flagged = open_mask & (~clue_t[:, None]) & (drop >= cfg.theta_drop)
        n_open = open_mask.sum(dim=1)
        n_flag = flagged.sum(dim=1)
        must_spare = (n_flag == n_open) & (n_flag > 0)   # all open flagged -> keep lowest-drop (incl. singletons)
        removed = flagged.clone()
        if bool(must_spare.any()):
            spare_idx = torch.where(open_mask, drop, torch.full_like(drop, 1e9)).argmin(dim=1)
            sp = must_spare.nonzero(as_tuple=True)[0]
            removed[sp, spare_idx[sp]] = False
            diag["blocked_last_candidate"] += int(must_spare.sum().item())
        committed = int(removed.sum().item())
        if ans_t is not None and committed:
            diag["false_eliminations"] += int(removed[torch.arange(N2, device=lat.device), ans_t].sum().item())
        diag["committed_eliminations"] += committed
        if committed == 0:
            return lat, ("solved" if bool((lat.sum(dim=1) == 1).all()) else "stall"), keep
        lat = lat.clone(); lat[removed] = 0.0
    return lat, "step_cap_exceeded", last_keep


@torch.no_grad()
def rollout(model, lat0, clue_t, cfg, ans_t):
    """Deterministic DFS over model deductions (contract §4-7). ans_t is used ONLY for
    scoring + the false-elim audit, never to guide search. -> (stop_reason, grid|None, diag)."""
    diag = {"committed_eliminations": 0, "blocked_last_candidate": 0, "false_eliminations": 0,
            "branches": 0, "nodes": 0, "model_conflict": 0, "empty_cell_conflict": 0,
            "terminal_invalid_grid": 0, "step_cap": 0, "stalled": 0}
    ans = ans_t.tolist()
    stack, seen = [lat0.clone()], set()
    while stack:
        if diag["nodes"] >= cfg.max_search_nodes:
            return "node_cap_exceeded", None, diag
        lat = stack.pop()
        key = _lattice_key(lat)
        if key in seen:
            continue
        seen.add(key); diag["nodes"] += 1
        final_lat, status, keep = _deduce_node(model, lat, clue_t, cfg, diag, ans_t)
        if status == "step_cap_exceeded":
            diag["step_cap"] += 1; continue
        if status in ("empty_cell_conflict", "model_conflict"):
            diag[status] += 1; continue
        if status == "solved":
            grid0 = final_lat.argmax(dim=1).tolist()
            if _valid_grid_flat([g + 1 for g in grid0]):
                return ("solved_exact" if grid0 == ans else "solved_valid_wrong"), final_lat.argmax(dim=1), diag
            diag["terminal_invalid_grid"] += 1; continue
        # stall -> branch (contract §5): most-constrained non-clue cell, candidates by keep desc
        diag["stalled"] += 1
        counts = final_lat.sum(dim=1)
        cand_counts = torch.where((counts > 1) & (~clue_t), counts, torch.full_like(counts, DIGITS + 1.0))
        best = int(cand_counts.argmin().item())
        if int(cand_counts[best].item()) > DIGITS:
            return "unsolved_no_branches", None, diag
        cands = [d for d in range(DIGITS) if final_lat[best, d].item() == 1.0]
        cands.sort(key=lambda d: (-keep[best, d].item(), d))     # latest keep_prob desc, digit asc
        diag["branches"] += len(cands)
        for d in reversed(cands):                                # reverse push -> LIFO preserves order
            child = final_lat.clone(); child[best].zero_(); child[best, d] = 1.0
            stack.append(child)
    return "unsolved_conflict_exhausted", None, diag


@torch.no_grad()
def rollout_eval(model, pairs, device, cfg):
    """Run the rollout per puzzle; aggregate rollout_exact_rate + contract diagnostics + records."""
    model.eval()
    counts = {s: 0 for s in PUZZLE_STOP_REASONS}
    agg = {k: 0 for k in ("nodes", "branches", "committed", "false_elim", "valid", "node_cap",
                          "step_cap", "model_conflict", "empty", "blocked", "stalled")}
    per_puzzle, n = [], len(pairs)
    for idx, (puz, sol) in enumerate(pairs):
        lat, _ = grids_to_tensors([(puz, sol)], device)
        clue_t = torch.tensor([puz[r][c] != 0 for r in range(N) for c in range(N)], device=device)
        ans_t = torch.tensor([sol[r][c] - 1 for r in range(N) for c in range(N)], device=device)
        sr, _, d = rollout(model, lat[0], clue_t, cfg, ans_t)
        counts[sr] = counts.get(sr, 0) + 1
        valid = sr in ("solved_exact", "solved_valid_wrong")
        agg["nodes"] += d["nodes"]; agg["branches"] += d["branches"]; agg["committed"] += d["committed_eliminations"]
        agg["false_elim"] += d["false_eliminations"]; agg["valid"] += int(valid)
        agg["node_cap"] += int(sr == "node_cap_exceeded"); agg["step_cap"] += d["step_cap"]
        agg["model_conflict"] += d["model_conflict"]; agg["empty"] += d["empty_cell_conflict"]
        agg["blocked"] += d["blocked_last_candidate"]; agg["stalled"] += d["stalled"]
        per_puzzle.append({"idx": idx, "clues": int(clue_t.sum().item()), "stop_reason": sr,
                           "exact": sr == "solved_exact", "valid": valid, "nodes": d["nodes"],
                           "branches": d["branches"], "committed": d["committed_eliminations"],
                           "false_elim": d["false_eliminations"]})
    m = max(1, n)
    diagnostics = {
        "rollout_valid_rate": round(agg["valid"] / m, 5),
        "avg_committed_eliminations": round(agg["committed"] / m, 2),
        "false_elimination_rate_answer_key_audit": round(agg["false_elim"] / m, 4),
        "avg_nodes_expanded": round(agg["nodes"] / m, 2),
        "avg_branches_created": round(agg["branches"] / m, 2),
        "node_cap_fraction": round(agg["node_cap"] / m, 4),
        "step_cap_fraction": round(agg["step_cap"] / m, 4),
        "model_conflict_count": agg["model_conflict"], "empty_cell_conflict_count": agg["empty"],
        "blocked_last_candidate_count": agg["blocked"], "stalled_branch_count": agg["stalled"],
    }
    return counts["solved_exact"] / m, counts, diagnostics, per_puzzle


def _rollout_smoke(model, real_pairs, cpu_rng, device, cfg):
    """Contract §10 smoke: rollout runs on real/synthetic; clue cells unchanged; no empty
    cell created by a step; stop reasons populated. Small caps so an untrained model bounds fast."""
    smcfg = replace(cfg, max_search_nodes=64, max_deduction_steps=16)
    sample = (list(real_pairs[:4]) if real_pairs else [gen_sudoku(cpu_rng, 30) for _ in range(4)])
    p0, s0 = sample[0]
    lat1, _ = grids_to_tensors([(p0, s0)], device)
    clue_t = torch.tensor([p0[r][c] != 0 for r in range(N) for c in range(N)], device=device)
    out_lat, _, _ = _deduce_node(model, lat1[0], clue_t, smcfg,
                                 {"committed_eliminations": 0, "blocked_last_candidate": 0, "false_eliminations": 0})
    clues_unchanged = bool((out_lat[clue_t] == lat1[0][clue_t]).all())
    no_empty = bool((out_lat.sum(dim=1) >= 1).all())
    er, counts, diag, _ = rollout_eval(model, sample, device, smcfg)
    model.train()
    return {"ran": True, "n": len(sample), "source": "real" if real_pairs else "synthetic",
            "clues_unchanged": clues_unchanged, "no_empty_cell_after_step": no_empty,
            "stop_reasons": counts, "exact_rate": round(er, 4), "diagnostics_present": bool(diag)}


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
    # I5 rollout smoke (contract §10): runs on real/synthetic, guards clues + no empty cell
    rcheck = _rollout_smoke(model, rp if real["checked"] else None, cpu_rng, device, cfg)
    return {
        "mode": "smoke", "param_count_full_config": pc,
        "param_count_in_budget": bool(6e5 <= pc <= 1.0e6),  # ~800K target
        "elim_shape": list(elim.shape), "conflict_shape": list(conflict.shape),
        "loss_first": round(losses[0], 5), "loss_last": round(losses[-1], 5),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "capture_grains": len(cap), "capture_grains_expected": cfg.n_iters * cfg.n_layers,
        "smoke_solve_rate": round(sr, 4), "device": str(device),
        "real_data_check": real, "i5_rollout_check": rcheck,
    }


def run_build_gate(cfg: Cfg, device, out: Path) -> dict:
    """The real build-gate: train on Sudoku-Extreme, eval via the I5 rollout to reproduce
    100% (contract PHASE1_I5_ROLLOUT_CONTRACT.md). Needs dataset + GPU + time; gated."""
    set_determinism(cfg.seed)
    if not cfg.data_dir:
        raise SystemExit("--data-dir (Sudoku-Extreme root) required for build-gate.")
    # n_train=1000 matches the paper's small-data Sudoku-Extreme regime; test load capped for memory
    train_pairs, test_pairs = load_dataset(Path(cfg.data_dir), n_train=1000, limit_test=(cfg.max_eval or 10000))
    rng = torch.Generator(device=device).manual_seed(cfg.seed)
    model = LatticeDeductionTransformer(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    resume_info = None
    start_step = 0
    if cfg.resume:
        resume_info = _load_checkpoint(out, cfg.resume, model, opt, rng, device)
        start_step = resume_info["step"]
    if cfg.compile_model:
        try:
            model = torch.compile(model, mode="reduce-overhead")   # CUDA graphs: kills launch overhead on CC>=7.0
            print(json.dumps({"torch_compile": "reduce-overhead"}), flush=True)
        except Exception as e:
            print(json.dumps({"torch_compile_failed": str(e)[:200]}), flush=True)
    t0 = time.time()
    train_summary = None
    if cfg.stage in ("train", "all"):
        model.train()
        # symmetry-augmented pool (breaks 1K-set overfitting, HRM/TRM approach); built ONCE on
        # device, then the train loop is a pure GPU gather (no per-step Python -> GPU-bound)
        aug_rng = random.Random(cfg.seed)
        aug_pool = [augment_grid(p, s, aug_rng) for (p, s) in train_pairs for _ in range(cfg.aug_factor)]
        pool_lat, pool_sol = grids_to_tensors(aug_pool, device)
        print(json.dumps({"aug_pool_size": len(aug_pool), "aug_factor": cfg.aug_factor}), flush=True)
        log_path = out / "train_log.jsonl"
        last_loss = last_elim = last_conflict = None
        for step0 in range(start_step, cfg.max_steps):
            step = step0 + 1
            idx = torch.randint(pool_lat.shape[0], (cfg.batch,), generator=rng, device=device)
            lat, sol = pool_lat[idx], pool_sol[idx]            # pure GPU gather from the augmented pool
            loss, elim_loss, conflict_loss = compute_loss(model, lat, sol, cfg, rng)
            opt.zero_grad(); loss.backward(); opt.step()
            last_loss, last_elim, last_conflict = float(loss.item()), float(elim_loss), float(conflict_loss)
            if cfg.log_every and (step == 1 or step % cfg.log_every == 0 or step == cfg.max_steps):
                row = {"step": step, "loss": round(last_loss, 6), "elim_loss": round(last_elim, 6),
                       "conflict_loss": round(last_conflict, 6), "wall_s": round(time.time() - t0, 1)}
                if cfg.eval_every and step % cfg.eval_every == 0:
                    sample = test_pairs[: min(cfg.eval_sample, len(test_pairs))]
                    row["diag_one_shot_exact_rate"] = round(solve_rate(model, sample, device, cfg), 5)
                _append_jsonl(log_path, row)
                print(json.dumps(row), flush=True)
            if cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
                _save_checkpoint(out, model, opt, step, cfg, rng, keep_step_copy=True)
        _save_checkpoint(out, model, opt, cfg.max_steps, cfg, rng, keep_step_copy=False)
        train_summary = {"start_step": start_step, "end_step": cfg.max_steps,
                         "last_loss": round(last_loss, 6) if last_loss is not None else None,
                         "last_elim_loss": round(last_elim, 6) if last_elim is not None else None,
                         "last_conflict_loss": round(last_conflict, 6) if last_conflict is not None else None,
                         "checkpoint_latest": str(out / "checkpoint_latest.pt")}

    if cfg.stage == "train":
        return {"mode": "build-gate", "stage": "train", "branch": "build_gate_train_checkpointed",
                "resume": resume_info, "train": train_summary, "wall_s": round(time.time() - t0, 1),
                "n_train": len(train_pairs), "n_test_loaded": len(test_pairs)}

    if cfg.stage == "eval" and not cfg.resume:
        raise SystemExit("--stage eval requires --resume latest or --resume <checkpoint.pt>")
    one_shot = solve_rate(model, test_pairs, device, cfg)            # diagnostic only
    exact_rate, counts, diagnostics, per_puzzle = rollout_eval(model, test_pairs, device, cfg)
    branch = ("build_gate_pass" if exact_rate >= 0.999 else
              "build_gate_partial" if exact_rate > 0 else "build_gate_fail")
    (out / "rollout_per_puzzle.jsonl").write_text("\n".join(json.dumps(r) for r in per_puzzle), encoding="utf-8")
    return {"mode": "build-gate", "stage": cfg.stage, "rollout_exact_rate": round(exact_rate, 5), "branch": branch,
            "one_shot_exact_rate": round(one_shot, 5), "stop_reason_counts": counts,
            "diagnostics": diagnostics, "resume": resume_info, "train": train_summary,
            "wall_s": round(time.time() - t0, 1),
            "n_train": len(train_pairs), "n_test": len(test_pairs)}


def main() -> int:
    ap = argparse.ArgumentParser(description="LDT build-gate model + trainer")
    ap.add_argument("--mode", choices=["smoke", "build-gate"], default="smoke")
    ap.add_argument("--stage", choices=["train", "eval", "all"], default="all")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="results/lattice/build-gate-sudoku-extreme")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--aug-factor", type=int, default=50, help="augmented pool = aug_factor x 1000")
    ap.add_argument("--compile", action="store_true", help="torch.compile reduce-overhead; needs GPU CC>=7.0 (A100/H100)")
    ap.add_argument("--resume", default="", help="latest or checkpoint path")
    ap.add_argument("--checkpoint-every", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=0, help="cheap one-shot diagnostic cadence during train")
    ap.add_argument("--eval-sample", type=int, default=64)
    ap.add_argument("--theta-drop", type=float, default=0.5)
    ap.add_argument("--theta-cls", type=float, default=THETA_CLS)
    ap.add_argument("--max-deduction-steps", type=int, default=64)
    ap.add_argument("--max-search-nodes", type=int, default=4096)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    cfg = Cfg(mode=a.mode, stage=a.stage, data_dir=a.data_dir, out=a.out, seed=a.seed,
              max_steps=a.max_steps, max_eval=a.max_eval, batch=a.batch, lr=a.lr, aug_factor=a.aug_factor,
              compile_model=a.compile,
              resume=a.resume, checkpoint_every=a.checkpoint_every, log_every=a.log_every,
              eval_every=a.eval_every, eval_sample=a.eval_sample, theta_drop=a.theta_drop,
              theta_cls=a.theta_cls, max_deduction_steps=a.max_deduction_steps,
              max_search_nodes=a.max_search_nodes, allow_dirty=a.allow_dirty)
    if a.mode == "smoke":
        cfg = Cfg(mode="smoke", d_model=32, n_layers=1, n_iters=2, batch=16, seed=a.seed,
                  allow_dirty=True, out=a.out, data_dir=a.data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    repo = Path(__file__).resolve().parents[1]
    out = Path(cfg.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    git = git_commit(repo, cfg.allow_dirty or cfg.mode == "smoke")

    result = run_smoke(cfg, device) if cfg.mode == "smoke" else run_build_gate(cfg, device, out)
    manifest = {
        "lane": "lattice", "phase": "1-build-gate", "tool": "scripts/lattice_ldt_model.py",
        "gitCommit": git["commit"], "gitDirty": git["dirty"], "device": str(device),
        "torch": torch.__version__, "python": sys.version.split()[0], "platform": platform.platform(),
        "arch": {"d_model": cfg.d_model, "n_layers": cfg.n_layers, "n_heads": cfg.n_heads,
                 "n_iters": cfg.n_iters, "lattice_dim": LATTICE_DIM, "lambda_cls": LAMBDA_CLS, "theta_cls": THETA_CLS},
        "stage": cfg.stage, "seed": cfg.seed, "maxSteps": cfg.max_steps, "maxEval": cfg.max_eval,
        "resume": cfg.resume, "checkpointEvery": cfg.checkpoint_every,
        "rolloutContract": "docs/lattice/PHASE1_I5_ROLLOUT_CONTRACT.md", "rolloutVersion": ROLLOUT_VERSION,
        "thetaDrop": cfg.theta_drop, "thetaCls": cfg.theta_cls, "logitSemantics": "candidate_keep_logit",
        "dropConfFormula": "1 - sigmoid(candidate_logit)",
        "maxDeductionStepsPerNode": cfg.max_deduction_steps, "maxSearchNodesPerPuzzle": cfg.max_search_nodes,
        "result": result,
        "discipline": "build-gate model only; no body/fiber number is a B-layer result until build_gate_pass",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"-> {out/'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
