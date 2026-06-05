#!/usr/bin/env python
"""JEPA Phase 0 - noise-carry test (coupled toy).

Implements docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md. Imports the frozen GEN
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from chatv2_phase0_bodyresist import (
    Cfg, TinyGPT, gen_batch, train_generative, extract_body, _lastpos, _COUPLE_A, _std,
)

MASK_TOKEN = 2                      # vocab 0,1 + MASK
LAM_INV, LAM_VAR, LAM_COV, GAMMA = 25.0, 25.0, 1.0, 1.0
EMA_TAU = 0.99
MASK_FRAC = 0.5
EFF_RANK_MIN_FRAC = 0.05


def cfg_for(d_model, p_noise, smoke=False):
    # Inherit the Phase-7b positive-control cell except for the d_model capacity axis.
    c = Cfg(latent="coupled", p_noise=p_noise, m=3, arity=2, h_sweep=[8], d_model=d_model,
            bits_per_channel=24, delta=0.45, max_steps=6000, min_steps=3000,
            patience=10, fair_readout=True, n_fingerprint=3000)
    if smoke:
        c = replace(c, n_fingerprint=1000)       # full training; smaller read sample for smoke
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
    return float(cross_val_score(LogisticRegression(max_iter=2000), X, y, cv=4).mean())


def _cv_nl(X, y):                                  # nonlinear (MLP) probe — for the noise-CONTAINMENT read
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(cross_val_score(MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                               random_state=0), X, y, cv=3).mean())


def _det(acc, base):
    return (acc - base) / max(1 - base, 1e-9)


def eff_rank(X):
    c = np.cov(X.T)
    ev = np.linalg.eigvalsh(c); ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))


def safe_nanmedian(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0 or np.isnan(arr).all():
        return float("nan")
    return float(np.nanmedian(arr))


def json_clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: json_clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_clean(x) for x in v]
    return v


def split_half(n, seed=0):
    perm = default_rng(seed).permutation(n)
    cut = n // 2
    return perm[:cut], perm[cut:]


def read_arrays(bodies, z, u, H, seed, with_nonlinear=False, read_protocol="full"):
    Ln = bodies.shape[1]
    fair = bodies.ndim == 4

    def view(layer, i):
        return bodies[:, layer, i, :] if fair else bodies[:, layer, :]

    def all_views(layer):
        if fair:
            return bodies[:, layer, :, :].reshape(bodies.shape[0], H * bodies.shape[-1])
        return bodies[:, layer, :]

    zr = [np.nanmean([_cv(_std(view(l, i)), z[:, i]) for i in range(H)]) for l in range(Ln)]
    l = int(np.nanargmax(zr))
    Xg = _std(all_views(l))
    clean = (u @ _COUPLE_A[:H].T) % 2
    x = (z ^ clean).astype(int)                                             # private noise x_i
    nd, nd_nl, nd_global, nd_global_nl, base, minc = [], [], [], [], [], []
    z_all_acc, z_flip_acc, z_clean_acc, z_flip_n = [], [], [], []
    tr, he = split_half(len(z))
    for i in range(H):
        b = max(x[:, i].mean(), 1 - x[:, i].mean())
        base.append(round(float(x[:, i].mean()), 3))
        minc.append(int(min(int(x[:, i].sum()), len(x) - int(x[:, i].sum()))))
        Xi = _std(view(l, i))
        nd.append(_det(_cv(Xi, x[:, i]), b))
        nd_nl.append(_det(_cv_nl(Xi, x[:, i]), b) if with_nonlinear else float("nan"))
        nd_global.append(_det(_cv(Xg, x[:, i]), b))
        nd_global_nl.append(_det(_cv_nl(Xg, x[:, i]), b) if with_nonlinear else float("nan"))

        # Repaired noise-carry read: train z_i probe on all training rows, then audit the
        # heldout x_i=1 flips. Training on flips would let a clean-only body learn the negation.
        Xtr, Xhe = view(l, i)[tr], view(l, i)[he]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        clf = LogisticRegression(max_iter=1000).fit((Xtr - mu) / sd, z[tr, i])
        pred = clf.predict((Xhe - mu) / sd)
        flip = x[he, i] == 1
        clean_rows = x[he, i] == 0
        z_all_acc.append(float((pred == z[he, i]).mean()))
        z_flip_acc.append(float((pred[flip] == z[he, i][flip]).mean()) if flip.any() else float("nan"))
        z_clean_acc.append(float((pred[clean_rows] == z[he, i][clean_rows]).mean()) if clean_rows.any() else float("nan"))
        z_flip_n.append(int(flip.sum()))
    u_det = float(np.nanmean([_det(_cv(Xg, u[:, k]), max(u[:, k].mean(), 1 - u[:, k].mean()))
                              for k in range(u.shape[1])]))
    unull = (default_rng(seed + 777).random(len(x)) < 0.5).astype(int)
    raw = view(l, H - 1)
    std = raw.std(0)
    return {
        "noise_det": round(safe_nanmedian(nd), 4),
        "noise_det_nl": round(safe_nanmedian(nd_nl), 4),
        "noise_det_global": round(safe_nanmedian(nd_global), 4),
        "noise_det_global_nl": round(safe_nanmedian(nd_global_nl), 4),
        "noise_det_each": [round(float(v), 3) for v in nd],
        "noise_det_nl_each": [round(float(v), 3) for v in nd_nl],
        "noise_det_global_each": [round(float(v), 3) for v in nd_global],
        "noise_det_global_nl_each": [round(float(v), 3) for v in nd_global_nl],
        "x_base": base, "x_minority": minc,
        "z_all_acc": round(float(np.nanmedian(z_all_acc)), 4),
        "z_flip_acc": round(float(np.nanmedian(z_flip_acc)), 4),
        "z_clean_acc": round(float(np.nanmedian(z_clean_acc)), 4),
        "z_all_acc_each": [round(float(v), 3) for v in z_all_acc],
        "z_flip_acc_each": [round(float(v), 3) for v in z_flip_acc],
        "z_clean_acc_each": [round(float(v), 3) for v in z_clean_acc],
        "z_flip_n": z_flip_n,
        "u_det": round(u_det, 4),
        "u_null": round(_det(_cv(Xg, unull), max(unull.mean(), 1 - unull.mean())), 4),
        "body_layer": l,
        "fair_readout": bool(fair),
        "read_protocol": read_protocol,
        "zr_by_layer": [round(float(v), 3) for v in zr],
        "collapse": {"frac_std_ok": round(float((std >= 0.10).mean()), 3),
                     "eff_rank": round(eff_rank(raw), 2), "d": int(raw.shape[1]),
                     "min_std": round(float(std.min()), 4)},
    }


def read_body(model, H, cfg, device, seed, with_nonlinear=False):
    bodies, z, u = extract_body(model, H, cfg, device, seed)
    return read_arrays(bodies, z, u, H, seed, with_nonlinear=with_nonlinear,
                       read_protocol="full_input_fair" if cfg.fair_readout else "full_input_final")


def extract_jepa_body_masked(model, H, cfg, device, seed, mask_reads=8):
    rng = default_rng(seed + 123)
    mask_rng = default_rng(seed + 456)
    bits, z, u = gen_batch(H, cfg.n_fingerprint, cfg, rng)
    lat = latent_of_pos(cfg.max_len, H, max(2, cfg.arity))
    nmask = int(round(MASK_FRAC * H))
    accum = None
    model.eval()
    with torch.no_grad():
        for _ in range(mask_reads):
            mask_lat = np.zeros((bits.shape[0], H), dtype=bool)
            for b in range(bits.shape[0]):
                mask_lat[b, mask_rng.choice(H, nmask, replace=False)] = True
            masked_bits = np.where(mask_lat[:, lat], MASK_TOKEN, bits)
            idx = torch.tensor(masked_bits, device=device)
            _, hiddens = model(idx, return_hidden=True)
            arr = np.stack([h[:, -1, :].cpu().numpy() for h in hiddens], axis=1)
            accum = arr if accum is None else accum + arr
    bodies = accum / float(mask_reads)
    return bodies, z, u


def read_jepa_body(model, H, cfg, device, seed, with_nonlinear=False, mask_reads=8):
    bodies, z, u = extract_jepa_body_masked(model, H, cfg, device, seed, mask_reads=mask_reads)
    out = read_arrays(bodies, z, u, H, seed, with_nonlinear=with_nonlinear,
                      read_protocol="masked_context_avg")
    out["mask_reads"] = int(mask_reads)
    return out


def support_starved(read):
    return sum(1 for c in read["z_flip_n"] if c < 50) > 2


def collapsed(read, d):
    c = read["collapse"]
    return c["frac_std_ok"] < 0.90 or c["eff_rank"] < max(8.0, EFF_RANK_MIN_FRAC * d)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chatv2/jepa-phase0-noise-carry")
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--p-noise", type=float, default=0.10)
    ap.add_argument("--smoke", action="store_true", help="1 seed, d=128, 300 steps")
    ap.add_argument("--gen-only", action="store_true", help="debug: train/read GEN only")
    ap.add_argument("--read-phase7b-gen", action="store_true",
                    help="debug: read existing Phase-7b GEN bodies instead of training")
    ap.add_argument("--with-nonlinear-probe", action="store_true",
                    help="debug: also run expensive MLP sidecar probes")
    ap.add_argument("--jepa-mask-reads", type=int, default=8,
                    help="number of random 50%%-mask read passes to average for JEPA")
    ap.add_argument("--read-n-fingerprint", type=int, default=0,
                    help="override cfg.n_fingerprint for the readout sample")
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
            if args.read_n_fingerprint:
                cfg = replace(cfg, n_fingerprint=args.read_n_fingerprint)
            run_seed = seed + 1000 * H                         # matches Phase-7b train_stage
            if args.read_phase7b_gen:
                npz = Path(f"results/chatv2/phase7b-coupled-lownoise/seed{seed}/bodies/H8_gen.npz")
                d0 = np.load(npz, allow_pickle=True)
                gr = read_arrays(d0["bodies"], d0["z"].astype(int), d0["u"].astype(int),
                                 H, run_seed, with_nonlinear=args.with_nonlinear_probe,
                                 read_protocol="phase7b_saved_full_input_fair")
                gmeta = {"source": str(npz), "trained_elsewhere": True}
            else:
                gen, gmeta = train_generative(H, cfg, device, run_seed)
                gr = read_body(gen, H, cfg, device, run_seed,
                               with_nonlinear=args.with_nonlinear_probe)
            row = {"d": d, "seed": seed, "run_seed": run_seed,
                   "gen_noise_det": gr["noise_det"], "gen_noise_det_nl": gr["noise_det_nl"],
                   "gen_noise_det_global": gr["noise_det_global"],
                   "gen_noise_det_global_nl": gr["noise_det_global_nl"],
                   "gen_z_flip_acc": gr["z_flip_acc"],
                   "gen_u_det": gr["u_det"], "gen_u_null": gr["u_null"],
                   "gen_train": gmeta, "gen_read": gr,
                   "support_starved": support_starved(gr), "x_minority": gr["x_minority"],
                   "z_flip_n": gr["z_flip_n"]}
            if args.gen_only:
                rows.append(row)
                print(f"[d={d} s={seed}] GEN nd_local={gr['noise_det']:.3f} "
                      f"nd_local_nl={gr['noise_det_nl']:.3f} nd_global={gr['noise_det_global']:.3f} "
                      f"nd_global_nl={gr['noise_det_global_nl']:.3f} zflip={gr['z_flip_acc']:.3f} "
                      f"u={gr['u_det']:.3f} "
                      f"zr={gr['zr_by_layer']} ({round(time.time()-t0,1)}s)", flush=True)
                continue

            jepa, jmeta = train_jepa(H, cfg, device, run_seed)
            jr = read_jepa_body(jepa, H, cfg, device, run_seed,
                                with_nonlinear=args.with_nonlinear_probe,
                                mask_reads=args.jepa_mask_reads)
            gap = round(gr["noise_det"] - jr["noise_det"], 4)
            z_flip_gap = round(gr["z_flip_acc"] - jr["z_flip_acc"], 4)
            row.update({"gap": gap,
                   "jepa_noise_det": jr["noise_det"], "jepa_noise_det_nl": jr["noise_det_nl"],
                   "jepa_noise_det_global": jr["noise_det_global"],
                   "jepa_noise_det_global_nl": jr["noise_det_global_nl"],
                   "jepa_z_flip_acc": jr["z_flip_acc"],
                   "z_flip_gap": z_flip_gap,
                   "jepa_u_det": jr["u_det"],
                   "jepa_u_null": jr["u_null"],
                   "jepa_collapse": jr["collapse"], "jepa_collapsed": collapsed(jr, d),
                   "jepa_read": jr,
                   "support_starved": support_starved(gr) or support_starved(jr),
                   "train_ctx_std_ok": jmeta["train_ctx_frac_std_ok"],
                   "jepa_read_protocol": jr["read_protocol"],
                   "jepa_mask_reads": jr["mask_reads"],
                   "n_fingerprint": cfg.n_fingerprint})
            rows.append(row)
            print(f"[d={d} s={seed}] GEN nd_lin={gr['noise_det']:.3f} nd_nl={gr['noise_det_nl']:.3f} "
                  f"zflip={gr['z_flip_acc']:.3f} u={gr['u_det']:.3f} effR={gr['collapse']['eff_rank']} | "
                  f"JEPA nd_lin={jr['noise_det']:.3f} nd_nl={jr['noise_det_nl']:.3f} u={jr['u_det']:.3f} "
                  f"zflip={jr['z_flip_acc']:.3f} effR={jr['collapse']['eff_rank']} std_ok={jr['collapse']['frac_std_ok']} "
                  f"read={jr['read_protocol']}x{jr['mask_reads']} | "
                  f"global_gap_nl={gr['noise_det_global_nl']-jr['noise_det_global_nl']:+.3f} "
                  f"zflip_gap={z_flip_gap:+.3f} "
                  f"({round(time.time()-t0,1)}s)",
                  flush=True)

    (out / ("smoke.json" if args.smoke else "summary.json")).write_text(
        json.dumps(json_clean(rows), indent=2)
    )
    print(f"\nwrote {out}", flush=True)
    if args.smoke:
        r = rows[0]
        print("\n==== SMOKE READ ====")
        if not args.gen_only:
            print(f"  JEPA collapsed = {r['jepa_collapsed']}  "
                  f"(guard: frac_std_ok>=0.90, eff_rank>=max(8,{EFF_RANK_MIN_FRAC}d))")
            print(f"  JEPA read = {r['jepa_read_protocol']} x {r['jepa_mask_reads']}  "
                  f"n_fingerprint = {r['n_fingerprint']}")
            print(f"  GEN z_flip_acc = {r['gen_z_flip_acc']}  JEPA z_flip_acc = {r['jepa_z_flip_acc']}  "
                  f"gap = {r['z_flip_gap']:+.3f}")
            print(f"  direct x sidecar GEN/JEPA = {r['gen_noise_det']}/{r['jepa_noise_det']}  "
                  f"gap = {r['gap']:+.3f}")
        else:
            print("  GEN-only debug smoke (JEPA skipped)")
            print(f"  GEN noise_det local/global = {r['gen_noise_det']}/{r['gen_noise_det_global']} "
                  f"nl local/global = {r['gen_noise_det_nl']}/{r['gen_noise_det_global_nl']}")
            print(f"  GEN z_flip_acc = {r['gen_z_flip_acc']}")
        print(f"  support_starved = {r['support_starved']}  "
              f"(heldout flip counts: {r['z_flip_n']}; x minority counts: {r['x_minority']})")
        if not args.gen_only:
            print(f"  u_det GEN/JEPA = {r['gen_u_det']}/{r['jepa_u_det']}")
        else:
            print(f"  u_det GEN = {r['gen_u_det']}")
        print("  GATE: smoke passes if JEPA not collapsed AND read runs; direction (gap>0) is a bonus.")


if __name__ == "__main__":
    main()
