#!/usr/bin/env python
"""JEPA Phase 0 - noise-carry test (coupled toy).

Implements docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md (locked). Imports the frozen GEN
machinery from chatv2_phase0_bodyresist (TinyGPT, gen_batch, train_generative, extract_body,
_lastpos, _COUPLE_A) UNEDITED; adds the JEPA objective (50% latent-channel mask + EMA/stop-grad
target encoder + 2-layer predictor + VICReg) and the noise-carry read.

FROZEN pins (spec 3.1): 50% mask, predictor 2-layer @ d_model, EMA tau=0.99,
VICReg lam_inv/lam_var/lam_cov = 25/25/1, gamma=1.0, embed_dim=d_model.
"""
import argparse
import copy
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from chatv2_phase0_bodyresist import (
    Cfg, TinyGPT, gen_batch, train_generative, extract_body, _lastpos, _COUPLE_A, _std,
)

MASK_TOKEN = 2                      # vocab 0,1 + MASK
LAM_INV, LAM_VAR, LAM_COV, GAMMA = 25.0, 25.0, 1.0, 1.0
EMA_TAU = 0.99
MASK_FRAC = 0.5


def cfg_for(d_model, p_noise, smoke=False):
    c = Cfg(latent="coupled", p_noise=p_noise, m=3, arity=2, h_sweep=[8], d_model=d_model,
            pos_h=8, n_fingerprint=2000)
    if smoke:
        c = replace(c, max_steps=300, eval_every=100, n_fingerprint=800)
    return c


def latent_of_pos(L, H, A):
    return (np.arange(L) // A) % H                          # position -> owning latent


# --------------------------------------------------------------------------- #
# JEPA
# --------------------------------------------------------------------------- #
class Predictor(nn.Module):
    def __init__(self, d, H):
        super().__init__()
        self.q = nn.Embedding(H, d)                         # per-latent query
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, ctx):                                 # ctx (B,d) -> (B,H,d)
        return self.net(ctx[:, None, :] + self.q.weight[None, :, :])


def vicreg(emb):                                            # emb (B,d)
    std = torch.sqrt(emb.var(dim=0) + 1e-4)
    var_loss = torch.relu(GAMMA - std).mean()
    z = emb - emb.mean(0)
    cov = (z.T @ z) / (emb.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off.pow(2).sum() / emb.shape[1]
    return var_loss, cov_loss


def train_jepa(H, cfg, device, seed):
    torch.manual_seed(seed + 31)
    rng = default_rng(seed + 31)
    L, A = cfg.max_len, max(2, cfg.arity)
    lat = latent_of_pos(L, H, A)
    lastpos = _lastpos(H, cfg.bits_per_channel, cfg.arity)
    ctx_enc = TinyGPT(3, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len).to(device)
    tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    predictor = Predictor(cfg.d_model, H).to(device)
    opt = torch.optim.AdamW(list(ctx_enc.parameters()) + list(predictor.parameters()),
                            lr=cfg.lr, weight_decay=0.01)
    nmask = int(round(MASK_FRAC * H))
    last_std_ok, last_eff = 0.0, 0.0
    for step in range(cfg.max_steps):
        bits, z, u = gen_batch(H, cfg.batch, cfg, rng)
        B = bits.shape[0]
        mask_lat = np.zeros((B, H), dtype=bool)
        for b in range(B):
            mask_lat[b, rng.choice(H, nmask, replace=False)] = True
        masked_bits = np.where(mask_lat[:, lat], MASK_TOKEN, bits)
        idx_ctx = torch.tensor(masked_bits, device=device)
        idx_full = torch.tensor(bits, device=device)
        ctx = ctx_enc(idx_ctx, return_hidden=True)[1][-1][:, -1, :]          # (B,d) context summary
        with torch.no_grad():
            tgt_all = tgt_enc(idx_full, return_hidden=True)[1][-1][:, lastpos, :]  # (B,H,d)
        pred_all = predictor(ctx)                                            # (B,H,d)
        m = torch.tensor(mask_lat, device=device)
        diff = ((pred_all - tgt_all) ** 2).mean(-1)                          # (B,H)
        inv = (diff * m).sum() / (m.sum() + 1e-9)
        var_loss, cov_loss = vicreg(ctx)
        loss = LAM_INV * inv + LAM_VAR * var_loss + LAM_COV * cov_loss
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, pc in zip(tgt_enc.parameters(), ctx_enc.parameters()):
                pt.mul_(EMA_TAU).add_(pc, alpha=1 - EMA_TAU)
            if step == cfg.max_steps - 1:
                s = ctx.std(0)
                last_std_ok = float((s >= 0.10).float().mean())
    return ctx_enc, {"steps": cfg.max_steps, "train_ctx_frac_std_ok": round(last_std_ok, 3)}


# --------------------------------------------------------------------------- #
# read
# --------------------------------------------------------------------------- #
def _cv(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=500), X, y, cv=4).mean())


def _det(acc, base):
    return (acc - base) / max(1 - base, 1e-9)


def eff_rank(X):
    c = np.cov(X.T)
    ev = np.linalg.eigvalsh(c); ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))


def read_body(model, H, cfg, device, seed):
    bodies, z, u = extract_body(model, H, cfg, device, seed)                 # (N,layers,d)
    Ln = bodies.shape[1]
    zr = [np.nanmean([_cv(_std(bodies[:, l, :]), z[:, i]) for i in range(H)]) for l in range(Ln)]
    l = int(np.nanargmax(zr))
    Xb = _std(bodies[:, l, :])
    clean = (u @ _COUPLE_A[:H].T) % 2
    x = (z ^ clean).astype(int)                                             # private noise x_i
    nd, base, minc = [], [], []
    for i in range(H):
        b = max(x[:, i].mean(), 1 - x[:, i].mean())
        base.append(round(float(x[:, i].mean()), 3))
        minc.append(int(min(int(x[:, i].sum()), len(x) - int(x[:, i].sum()))))
        nd.append(_det(_cv(Xb, x[:, i]), b))
    u_det = float(np.nanmean([_det(_cv(Xb, u[:, k]), max(u[:, k].mean(), 1 - u[:, k].mean()))
                              for k in range(u.shape[1])]))
    unull = (default_rng(seed + 777).random(len(x)) < 0.5).astype(int)
    raw = bodies[:, l, :]
    std = raw.std(0)
    return {
        "noise_det": round(float(np.nanmedian(nd)), 4),
        "noise_det_each": [round(float(v), 3) for v in nd],
        "x_base": base, "x_minority": minc,
        "u_det": round(u_det, 4),
        "u_null": round(_det(_cv(Xb, unull), max(unull.mean(), 1 - unull.mean())), 4),
        "body_layer": l,
        "collapse": {"frac_std_ok": round(float((std >= 0.10).mean()), 3),
                     "eff_rank": round(eff_rank(raw), 2), "d": int(raw.shape[1]),
                     "min_std": round(float(std.min()), 4)},
    }


def support_starved(read):
    return sum(1 for c in read["x_minority"] if c < 50) > 2


def collapsed(read, d):
    c = read["collapse"]
    return c["frac_std_ok"] < 0.90 or c["eff_rank"] < 0.5 * d


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chatv2/jepa-phase0-noise-carry")
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--p-noise", type=float, default=0.10)
    ap.add_argument("--smoke", action="store_true", help="1 seed, d=128, 300 steps")
    args = ap.parse_args()
    if args.smoke:
        args.dims, args.seeds = [128], [0]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    H = 8
    print(f"[cfg] device={device} dims={args.dims} seeds={args.seeds} p_noise={args.p_noise} "
          f"smoke={args.smoke}", flush=True)

    rows = []
    t0 = time.time()
    for d in args.dims:
        for seed in args.seeds:
            cfg = cfg_for(d, args.p_noise, args.smoke)
            gen, gmeta = train_generative(H, cfg, device, seed)
            gr = read_body(gen, H, cfg, device, seed)
            jepa, jmeta = train_jepa(H, cfg, device, seed)
            jr = read_body(jepa, H, cfg, device, seed)
            gap = round(gr["noise_det"] - jr["noise_det"], 4)
            row = {"d": d, "seed": seed, "gap": gap,
                   "gen_noise_det": gr["noise_det"], "jepa_noise_det": jr["noise_det"],
                   "gen_u_det": gr["u_det"], "jepa_u_det": jr["u_det"],
                   "gen_u_null": gr["u_null"], "jepa_u_null": jr["u_null"],
                   "jepa_collapse": jr["collapse"], "jepa_collapsed": collapsed(jr, d),
                   "support_starved": support_starved(gr) or support_starved(jr),
                   "x_minority": gr["x_minority"], "train_ctx_std_ok": jmeta["train_ctx_frac_std_ok"]}
            rows.append(row)
            print(f"[d={d} s={seed}] GEN noise_det={gr['noise_det']:.3f} u_det={gr['u_det']:.3f} | "
                  f"JEPA noise_det={jr['noise_det']:.3f} u_det={jr['u_det']:.3f} | gap={gap:+.3f} | "
                  f"jepa_collapse={collapsed(jr, d)} (std_ok={jr['collapse']['frac_std_ok']}, "
                  f"eff_rank={jr['collapse']['eff_rank']}/{d}) ({round(time.time()-t0,1)}s)", flush=True)

    (out / ("smoke.json" if args.smoke else "summary.json")).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}", flush=True)
    if args.smoke:
        r = rows[0]
        print("\n==== SMOKE READ ====")
        print(f"  JEPA collapsed = {r['jepa_collapsed']}  (guard: frac_std_ok>=0.90, eff_rank>=0.5d)")
        print(f"  GEN noise_det = {r['gen_noise_det']}  JEPA noise_det = {r['jepa_noise_det']}  gap = {r['gap']:+.3f}")
        print(f"  support_starved = {r['support_starved']}  (x minority counts: {r['x_minority']})")
        print(f"  u_det GEN/JEPA = {r['gen_u_det']}/{r['jepa_u_det']}")
        print("  GATE: smoke passes if JEPA not collapsed AND read runs; direction (gap>0) is a bonus.")


if __name__ == "__main__":
    main()
