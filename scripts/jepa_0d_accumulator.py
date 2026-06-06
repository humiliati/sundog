#!/usr/bin/env python
"""JEPA-0D accumulator runner - GEN vs JEPA on the accumulator/count substrate.

Implements docs/chatv2/JEPA_0D_ACCUMULATOR_SPEC.md (gates 3-6). Imports the FROZEN substrate from
jepa_0d_accumulator_preflight (gen_accumulator, AccCfg, ...) so the runner and the model-free
preflight cannot drift, and the body machinery (TinyGPT, _std) from chatv2_phase0_bodyresist
UNEDITED. New vs the parity runner (jepa_phase0_noise_carry):

  * WHOLE-CHECKPOINT mask (the locked re-pose): mask all of one checkpoint's readout tokens as a
    unit; predict their target embeddings from context -> events->u_t is the only non-trivial route.
  * checkpoint u_det read at the event-integration position (last event token of tick c, which is
    CAUSALLY BEFORE c's readouts, so masking c is invariant); per-checkpoint majority baseline.
  * pooled flip read at the readout position, read under masks that keep the target checkpoint
    VISIBLE (mask some OTHER checkpoint), so the body has actually seen the readout it is probed for.

FROZEN JEPA pins (spec 3.1): EMA tau=0.99, VICReg 25/25/1 (gamma=1.0), 2-layer predictor @ d_model,
embed_dim=d_model, mask_reads=8, collapse guard (std>=0.10 on >=90% dims, eff-rank>=max(8,0.05d)).
"""
import argparse
import copy
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from chatv2_phase0_bodyresist import TinyGPT, _std
from jepa_0d_accumulator_preflight import AccCfg, gen_accumulator
from jepa_0d_accumulator_preflight import _cv_acc as pf_cv_acc, _det as pf_det

MASK_TOKEN = 2                         # vocab 0,1 + MASK (JEPA context encoder)
# 2026-06-05 objective fix (operator-approved, experimental; spec deviation pending re-smoke):
#   LAM_COV 1->10 to break the low-rank summary (diag showed ctx effR ~1 with lam_cov=1);
#   targets standardized per masked-checkpoint group (see train_jepa) to kill the position shortcut.
LAM_INV, LAM_VAR, LAM_COV, GAMMA = 25.0, 25.0, 10.0, 1.0
EMA_TAU = 0.99
EFF_RANK_MIN_FRAC = 0.05
U_DET_BAR = 0.70                       # gates 3 (GEN) & 4 (JEPA)
FROZEN_DELTA = 0.15                    # gate 5


@dataclass
class TCfg:
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    lr: float = 3e-4
    batch: int = 128
    eval_batch: int = 512
    max_steps: int = 6000
    min_steps: int = 3000
    patience: int = 10
    eval_every: int = 250
    mask_reads: int = 8


def acc_cfg(n_fingerprint):
    """The locked accumulator substrate (preflight defaults), with the read sample size set."""
    return replace(AccCfg(), n=n_fingerprint)


# --------------------------------------------------------------------------- #
# layout / batches
# --------------------------------------------------------------------------- #
def read_layout(acc):
    """Deterministic token positions (same for every batch). Returns:
    event_pos[ci]            last event token of checkpoint tick c (u_t integration position),
    readout_end[ci][j]       last token of readout block (c,j) (where z_{c,j} is freshest),
    readout_tok[ci]          all token indices of checkpoint c's readout blocks (the mask unit)."""
    data = gen_accumulator(replace(acc, n=1), default_rng(0))
    lay, ckpts = data["layout"], data["ckpts"]
    tick_ev = {tk["tick"]: tk["event_slice"] for tk in lay["ticks"]}
    event_pos, readout_end, readout_tok = [], [], []
    for c in ckpts:
        event_pos.append(tick_ev[c][1] - 1)
        ends, toks = [], []
        for j in range(acc.n_U):
            s, e = lay["checkpoints"][c][j]
            ends.append(e - 1); toks.extend(range(s, e))
        readout_end.append(ends); readout_tok.append(toks)
    return {"ckpts": ckpts, "event_pos": event_pos, "readout_end": readout_end,
            "readout_tok": readout_tok, "L": data["L"]}


def gen_batch(acc, batch, rng):
    d = gen_accumulator(replace(acc, n=batch), rng)
    return d["X"].astype(np.int64), d["e"], d["u"], d["z_obs"], d["x_flip"], d["clean"]


def u_at_ckpts(u, ckpts):
    return u[:, [c - 1 for c in ckpts]]                    # (n, n_ckpt) count at each checkpoint


# --------------------------------------------------------------------------- #
# GEN (generative next-token baseline)
# --------------------------------------------------------------------------- #
def train_gen(acc, tcfg, device, seed):
    torch.manual_seed(seed); rng = default_rng(seed)
    model = TinyGPT(2, tcfg.d_model, tcfg.n_layers, tcfg.n_heads, read_layout(acc)["L"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=0.01)
    eb, _, _, _, _, _ = gen_batch(acc, tcfg.eval_batch, default_rng(seed + 999))
    et = torch.tensor(eb, device=device)
    best, bad, steps = float("inf"), 0, 0
    for step in range(tcfg.max_steps):
        bits, *_ = gen_batch(acc, tcfg.batch, rng)
        idx = torch.tensor(bits, device=device)
        loss = F.cross_entropy(model(idx)[:, :-1].reshape(-1, 2), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        steps = step + 1
        if steps % tcfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                el = F.cross_entropy(model(et)[:, :-1].reshape(-1, 2), et[:, 1:].reshape(-1)).item()
            model.train()
            if el < best - 1e-4:
                best, bad = el, 0
            else:
                bad += 1
                if bad >= tcfg.patience and steps >= tcfg.min_steps:
                    break
    return model, {"eval_loss": float(best), "steps": steps}


# --------------------------------------------------------------------------- #
# JEPA (whole-checkpoint mask)
# --------------------------------------------------------------------------- #
class Predictor(nn.Module):
    def __init__(self, d, n_U):
        super().__init__()
        self.q = nn.Embedding(n_U, d)                       # per-readout-channel query
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, ctx):                                 # ctx (B,d) -> (B,n_U,d)
        return self.net(ctx[:, None, :] + self.q.weight[None, :, :])


def vicreg(emb):
    std = torch.sqrt(emb.var(dim=0) + 1e-4)
    var_loss = torch.relu(GAMMA - std).mean()
    z = emb - emb.mean(0)
    cov = (z.T @ z) / (emb.shape[0] - 1)
    off = cov - torch.diag(torch.diagonal(cov))
    return var_loss, off.pow(2).sum() / emb.shape[1]


def train_jepa(acc, tcfg, device, seed):
    torch.manual_seed(seed + 31); rng = default_rng(seed + 31)
    lay = read_layout(acc)
    ckpts, n_ckpt, n_U = lay["ckpts"], len(lay["ckpts"]), acc.n_U
    readout_tok = [np.asarray(t) for t in lay["readout_tok"]]
    readout_end = [np.asarray(e) for e in lay["readout_end"]]
    ctx_enc = TinyGPT(3, tcfg.d_model, tcfg.n_layers, tcfg.n_heads, lay["L"]).to(device)
    tgt_enc = copy.deepcopy(ctx_enc).to(device)
    for p in tgt_enc.parameters():
        p.requires_grad_(False)
    predictor = Predictor(tcfg.d_model, n_U).to(device)
    opt = torch.optim.AdamW(list(ctx_enc.parameters()) + list(predictor.parameters()),
                            lr=tcfg.lr, weight_decay=0.01)
    last_std_ok = 0.0
    for step in range(tcfg.max_steps):
        bits, *_ = gen_batch(acc, tcfg.batch, rng)
        B = bits.shape[0]
        mc = rng.integers(0, n_ckpt, B)                    # per-sample masked checkpoint
        masked = bits.copy()
        for ci in range(n_ckpt):
            sel = np.where(mc == ci)[0]
            if sel.size:
                masked[np.ix_(sel, readout_tok[ci])] = MASK_TOKEN
        idx_ctx = torch.tensor(masked, device=device)
        idx_full = torch.tensor(bits, device=device)
        ctx = ctx_enc(idx_ctx, return_hidden=True)[1][-1][:, -1, :]          # (B,d) context summary
        with torch.no_grad():
            tgt_full = tgt_enc(idx_full, return_hidden=True)[1][-1]          # (B,L,d)
        tgt = torch.zeros(B, n_U, tcfg.d_model, device=device)
        for ci in range(n_ckpt):
            sel = mc == ci
            if sel.any():
                tgt[sel] = tgt_full[sel][:, readout_end[ci], :]
        pred = predictor(ctx)                                                # (B,n_U,d)
        # FIX (2026-06-05): standardize targets per masked-checkpoint group so the predictor must
        # match the COUNT-DEPENDENT variation, not the positional mean. The diagnostic showed JEPA
        # trivially matched the position-dominated embedding and never learned u_c (parity failure
        # mode). Centering+normalizing the target per (group, channel, dim) removes that shortcut.
        inv_terms = []
        for ci in range(n_ckpt):
            m = torch.tensor(mc == ci, device=device)
            if int(m.sum()) < 4:
                continue
            t = tgt[m]
            t = (t - t.mean(0, keepdim=True)) / (t.std(0, keepdim=True) + 1e-4)
            inv_terms.append(((pred[m] - t) ** 2).mean())
        inv = torch.stack(inv_terms).mean() if inv_terms else (pred * 0.0).mean()
        var_loss, cov_loss = vicreg(ctx)
        loss = LAM_INV * inv + LAM_VAR * var_loss + LAM_COV * cov_loss
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            for pt, pc in zip(tgt_enc.parameters(), ctx_enc.parameters()):
                pt.mul_(EMA_TAU).add_(pc, alpha=1 - EMA_TAU)
            if step == tcfg.max_steps - 1:
                last_std_ok = float((ctx.std(0) >= 0.10).float().mean())
    return ctx_enc, {"steps": tcfg.max_steps, "train_ctx_frac_std_ok": round(last_std_ok, 3)}


# --------------------------------------------------------------------------- #
# reads
# --------------------------------------------------------------------------- #
def _forward_positions(model, bits, device, positions, chunk=512):
    """hidden at the given token positions, all layers -> (n, len(positions), Ln, d)."""
    outs = []
    model.eval()
    pos = list(positions)
    with torch.no_grad():
        for i in range(0, bits.shape[0], chunk):
            idx = torch.tensor(bits[i:i + chunk], device=device)
            _, hiddens = model(idx, return_hidden=True)
            arr = np.stack([h[:, pos, :].cpu().numpy() for h in hiddens], axis=2)  # (b,P,Ln,d)
            outs.append(arr)
    return np.concatenate(outs, axis=0)


def extract_gen(model, acc, device, seed):
    rng = default_rng(seed + 123)
    bits, e, u, z_obs, x_flip, clean = gen_batch(acc, acc.n, rng)
    lay = read_layout(acc); ckpts, n_U = lay["ckpts"], acc.n_U
    u_bodies = _forward_positions(model, bits, device, lay["event_pos"])      # (n,n_ckpt,Ln,d)
    flat = [p for ends in lay["readout_end"] for p in ends]
    zb = _forward_positions(model, bits, device, flat)                        # (n,n_ckpt*n_U,Ln,d)
    z_bodies = zb.reshape(zb.shape[0], len(ckpts), n_U, zb.shape[2], zb.shape[3])
    return u_bodies, z_bodies, u, z_obs, x_flip, ckpts


def extract_jepa(model, acc, tcfg, device, seed):
    rng = default_rng(seed + 123)
    bits, e, u, z_obs, x_flip, clean = gen_batch(acc, acc.n, rng)
    lay = read_layout(acc); ckpts, n_ckpt, n_U = lay["ckpts"], len(lay["ckpts"]), acc.n_U
    readout_tok = [np.asarray(t) for t in lay["readout_tok"]]
    flat = [p for ends in lay["readout_end"] for p in ends]
    n = bits.shape[0]
    acc_u, acc_z, cu = None, None, np.zeros(n_ckpt)
    for p in range(tcfg.mask_reads):
        mc = p % n_ckpt                                    # batch-wide masked checkpoint this pass
        masked = bits.copy(); masked[:, readout_tok[mc]] = MASK_TOKEN
        hu = _forward_positions(model, masked, device, lay["event_pos"])       # (n,n_ckpt,Ln,d)
        hz = _forward_positions(model, masked, device, flat)
        hz = hz.reshape(n, n_ckpt, n_U, hz.shape[2], hz.shape[3])
        acc_u = hu if acc_u is None else acc_u + hu                            # u-read: avg over all passes
        if acc_z is None:
            acc_z = np.zeros_like(hz)
        for ci in range(n_ckpt):                                               # z-read of c: only passes NOT masking c
            if ci != mc:
                acc_z[:, ci] += hz[:, ci]; cu[ci] += 1
    u_bodies = acc_u / tcfg.mask_reads
    z_bodies = acc_z / cu[None, :, None, None, None]
    return u_bodies, z_bodies, u, z_obs, x_flip, ckpts


def eff_rank(X):
    c = np.cov(X.T)
    ev = np.linalg.eigvalsh(c); ev = ev[ev > 1e-12]
    return float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))


def probe_u_det(u_bodies, u_ck):
    """u_det = median over checkpoints of (acc_c - maj_c)/(1-maj_c), at the best (by mean u-recovery)
    layer; per-checkpoint majority baseline strips the monotone-count position prior."""
    n, n_ckpt, Ln, d = u_bodies.shape
    layer_score = []
    for l in range(Ln):
        accs = [pf_cv_acc(_std(u_bodies[:, ci, l, :]), u_ck[:, ci])[0] for ci in range(n_ckpt)]
        layer_score.append(np.nanmean(accs))
    lstar = int(np.nanargmax(layer_score))
    per = []
    for ci in range(n_ckpt):
        a, m, _ = pf_cv_acc(_std(u_bodies[:, ci, lstar, :]), u_ck[:, ci])
        per.append({"acc": round(a, 4), "maj": round(m, 4), "u_det": round(pf_det(a, m), 4)})
    udet = float(np.nanmedian([p["u_det"] for p in per]))
    body = u_bodies[:, -1, lstar, :]                       # last-checkpoint body for collapse stats
    std = body.std(0)
    return {"u_det": round(udet, 4), "body_layer": lstar, "per_ckpt": per,
            "layer_score": [round(float(v), 3) for v in layer_score],
            "collapse": {"frac_std_ok": round(float((std >= 0.10).mean()), 3),
                         "eff_rank": round(eff_rank(body), 2), "d": int(d),
                         "min_std": round(float(std.min()), 4)}}


def probe_z_flip(z_bodies, z_obs, x_flip, lstar, seed=0):
    """Flip-conditioned read: train body@c -> z_{c,j} on all-row train split, score held-out flips
    {x=1}; POOLED across the n_U channels per checkpoint (spec 6). Probe NOT trained on flips only."""
    n, n_ckpt, n_U, Ln, d = z_bodies.shape
    perm = default_rng(seed).permutation(n); cut = n // 2
    tr, he = perm[:cut], perm[cut:]
    per_ck, all_acc, clean_acc, flipn = [], [], [], []
    for ci in range(n_ckpt):
        fc, fn = 0, 0; ac, an = 0, 0; cc, cn = 0, 0
        for j in range(n_U):
            X = z_bodies[:, ci, j, lstar, :]
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
            clf = LogisticRegression(max_iter=1000).fit((X[tr] - mu) / sd, z_obs[tr, ci, j])
            pred = clf.predict((X[he] - mu) / sd)
            y = z_obs[he, ci, j]; flip = x_flip[he, ci, j] == 1; cln = ~flip
            ac += int((pred == y).sum()); an += y.size
            fc += int((pred[flip] == y[flip]).sum()); fn += int(flip.sum())
            cc += int((pred[cln] == y[cln]).sum()); cn += int(cln.sum())
        per_ck.append(round(fc / max(fn, 1), 4)); flipn.append(fn)
        all_acc.append(round(ac / max(an, 1), 4)); clean_acc.append(round(cc / max(cn, 1), 4))
    return {"z_flip_acc": round(float(np.median(per_ck)), 4), "z_flip_per_ckpt": per_ck,
            "z_all_per_ckpt": all_acc, "z_clean_per_ckpt": clean_acc, "flip_n_per_ckpt": flipn}


def collapsed(coll, d):
    return coll["frac_std_ok"] < 0.90 or coll["eff_rank"] < max(8.0, EFF_RANK_MIN_FRAC * d)


def json_clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {k: json_clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


# --------------------------------------------------------------------------- #
# DIAGNOSTIC: where does each body store u_c? probe multiple read surfaces.
# --------------------------------------------------------------------------- #
def _diag_positions(acc):
    lay = read_layout(acc)
    rre = [max(es) for es in lay["readout_end"]]            # last token of each ckpt's readout region
    pos = [lay["L"] - 1] + list(lay["event_pos"]) + rre     # [final] + event-end[c] + readout-end[c]
    return lay, pos, rre


def diag_extract_gen(model, acc, device, seed):
    bits, e, u, *_ = gen_batch(acc, acc.n, default_rng(seed + 123))
    lay, pos, _ = _diag_positions(acc); nck = len(lay["ckpts"])
    H = _forward_positions(model, bits, device, pos)         # (n, 1+2*nck, Ln, d)
    final = H[:, 0]; ev = H[:, 1:1 + nck]; ro = H[:, 1 + nck:1 + 2 * nck]
    surf = {"final": np.repeat(final[:, None], nck, axis=1), "event": ev, "readout": ro}
    return surf, u_at_ckpts(u, lay["ckpts"])


def diag_extract_jepa(model, acc, tcfg, device, seed):
    bits, e, u, *_ = gen_batch(acc, acc.n, default_rng(seed + 123))
    lay, pos, _ = _diag_positions(acc); ckpts = lay["ckpts"]; nck = len(ckpts)
    readout_tok = [np.asarray(t) for t in lay["readout_tok"]]
    fin_all, ev_all = None, None
    fin_mc = [None] * nck; cfm = np.zeros(nck)
    ro_vis = [None] * nck; cro = np.zeros(nck)
    for p in range(tcfg.mask_reads):
        mc = p % nck
        masked = bits.copy(); masked[:, readout_tok[mc]] = MASK_TOKEN
        H = _forward_positions(model, masked, device, pos)
        fin = H[:, 0]; ev = H[:, 1:1 + nck]; ro = H[:, 1 + nck:1 + 2 * nck]
        fin_all = fin if fin_all is None else fin_all + fin
        ev_all = ev if ev_all is None else ev_all + ev
        fin_mc[mc] = fin if fin_mc[mc] is None else fin_mc[mc] + fin; cfm[mc] += 1
        for ci in range(nck):
            if ci != mc:
                ro_vis[ci] = ro[:, ci] if ro_vis[ci] is None else ro_vis[ci] + ro[:, ci]; cro[ci] += 1
    surf = {"final_all": np.repeat((fin_all / tcfg.mask_reads)[:, None], nck, axis=1),
            "final_maskC": np.stack([fin_mc[ci] / max(cfm[ci], 1) for ci in range(nck)], axis=1),
            "event": ev_all / tcfg.mask_reads,
            "readout_vis": np.stack([ro_vis[ci] / max(cro[ci], 1) for ci in range(nck)], axis=1)}
    return surf, u_at_ckpts(u, ckpts)


def run_diag(acc, tcfg, device, seed):
    t0 = time.time()
    gen, gm = train_gen(acc, tcfg, device, seed)
    gs, uck = diag_extract_gen(gen, acc, device, seed)
    gen_res = {k: probe_u_det(v, uck) for k, v in gs.items()}
    print(f"[diag] GEN trained ({gm['steps']} steps, eval_loss={gm['eval_loss']:.4f}); "
          f"surfaces read  ({round(time.time()-t0,1)}s)", flush=True)
    jepa, jm = train_jepa(acc, tcfg, device, seed)
    js, uck2 = diag_extract_jepa(jepa, acc, tcfg, device, seed)
    jepa_res = {k: probe_u_det(v, uck2) for k, v in js.items()}
    out = {"gen": {"train": gm, "surfaces": gen_res}, "jepa": {"train": jm, "surfaces": jepa_res},
           "wall_s": round(time.time() - t0, 1)}
    print("\n==== DIAGNOSTIC: u_det by read surface (per-ckpt majority baseline) ====")
    print("  GEN:")
    for k, r in gen_res.items():
        print(f"    {k:12s} u_det={r['u_det']:+.3f}  per_ckpt={[p['u_det'] for p in r['per_ckpt']]}  "
              f"effR={r['collapse']['eff_rank']}  layer={r['body_layer']}")
    print("  JEPA:")
    for k, r in jepa_res.items():
        print(f"    {k:12s} u_det={r['u_det']:+.3f}  per_ckpt={[p['u_det'] for p in r['per_ckpt']]}  "
              f"effR={r['collapse']['eff_rank']}  layer={r['body_layer']}")
    print(f"  (wall {out['wall_s']}s)  Reading: which JEPA surface (final_all/final_maskC/event/"
          f"readout_vis) carries u_c tells us the read fix; GEN surfaces show if the count is "
          f"present-but-decaying vs read-misplaced.", flush=True)
    return out


# --------------------------------------------------------------------------- #
def run_cell(d, seed, acc, tcfg, device):
    t0 = time.time()
    gen, gmeta = train_gen(acc, tcfg, device, seed)
    gu, gz, u, z_obs, x_flip, ckpts = extract_gen(gen, acc, device, seed)
    g_u = probe_u_det(gu, u_at_ckpts(u, ckpts))
    g_z = probe_z_flip(gz, z_obs, x_flip, g_u["body_layer"], seed)

    jepa, jmeta = train_jepa(acc, tcfg, device, seed)
    ju, jz, u2, z2, x2, _ = extract_jepa(jepa, acc, tcfg, device, seed)
    j_u = probe_u_det(ju, u_at_ckpts(u2, ckpts))
    j_z = probe_z_flip(jz, z2, x2, j_u["body_layer"], seed)

    z_flip_gap = round(g_z["z_flip_acc"] - j_z["z_flip_acc"], 4)
    row = {"d": d, "seed": seed,
           "gen_u_det": g_u["u_det"], "jepa_u_det": j_u["u_det"],
           "gen_z_flip_acc": g_z["z_flip_acc"], "jepa_z_flip_acc": j_z["z_flip_acc"],
           "z_flip_gap": z_flip_gap,
           "jepa_collapsed": collapsed(j_u["collapse"], tcfg.d_model),
           "jepa_train_std_ok": jmeta["train_ctx_frac_std_ok"],
           "gen": {"u": g_u, "z": g_z, "train": gmeta},
           "jepa": {"u": j_u, "z": j_z, "train": jmeta},
           "wall_s": round(time.time() - t0, 1)}
    print(f"[d={d} s={seed}] GEN u_det={g_u['u_det']:.3f} z_flip={g_z['z_flip_acc']:.3f} | "
          f"JEPA u_det={j_u['u_det']:.3f} z_flip={j_z['z_flip_acc']:.3f} effR={j_u['collapse']['eff_rank']} "
          f"collapsed={row['jepa_collapsed']} | gap={z_flip_gap:+.3f}  ({row['wall_s']}s)", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chatv2/jepa-0d-accumulator-lock")
    ap.add_argument("--dims", type=int, nargs="+", default=[128, 256])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--smoke", action="store_true", help="d=128, 1 seed, full training, n_fp=1000")
    ap.add_argument("--diag", action="store_true",
                    help="diagnostic re-smoke: full training, probe u at final/event/readout surfaces")
    ap.add_argument("--dev", action="store_true", help="tiny self-test: d=64, 40 steps, n_fp=300")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit CPU execution (default: refuse, since the CPU path is hours-slow)")
    args = ap.parse_args()

    if args.diag:
        args.dims, args.seeds = [128], [0]
        acc, tcfg = acc_cfg(1000), TCfg(d_model=128, mask_reads=9)   # 9 -> balanced 3 masks/ckpt
    elif args.smoke:
        args.dims, args.seeds = [128], [0]
        acc, tcfg = acc_cfg(1000), TCfg(d_model=128)
    elif args.dev:
        args.dims, args.seeds = [64], [0]
        acc = acc_cfg(300)
        tcfg = TCfg(d_model=64, n_layers=2, max_steps=40, min_steps=0, eval_every=20,
                    patience=2, mask_reads=2)
    else:
        acc, tcfg = acc_cfg(3000), TCfg()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and not (args.allow_cpu or args.dev):
        print(f"[ABORT] CUDA not available (torch {torch.__version__}). You are almost certainly on "
              f"the wrong interpreter — use the GPU venv:\n"
              f"  C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe scripts/jepa_0d_accumulator.py "
              f"{'--smoke ' if args.smoke else ''}--out {args.out}\n"
              f"The CPU path is hours-slow (handoff footgun). Pass --allow-cpu to override.", flush=True)
        raise SystemExit(2)
    print(f"[cfg] device={device} dims={args.dims} seeds={args.seeds} "
          f"smoke={args.smoke} diag={args.diag} dev={args.dev} n_fp={acc.n} steps={tcfg.max_steps} "
          f"mask=whole_checkpoint mask_reads={tcfg.mask_reads}", flush=True)

    if args.diag:
        res = run_diag(acc, replace(tcfg, d_model=args.dims[0]), device, args.seeds[0])
        (out / "diag.json").write_text(json.dumps(json_clean(res), indent=2))
        print(f"\nwrote {out/'diag.json'}", flush=True)
        return

    rows = []
    for d in args.dims:
        for seed in args.seeds:
            rows.append(run_cell(d, seed, acc, replace(tcfg, d_model=d), device))

    (out / ("smoke.json" if args.smoke else "dev.json" if args.dev else "summary.json")).write_text(
        json.dumps(json_clean(rows), indent=2))
    print(f"\nwrote {out}", flush=True)
    if args.smoke or args.dev:
        r = rows[0]
        print("\n==== SMOKE READ ====")
        print(f"  JEPA collapsed = {r['jepa_collapsed']} (guard: std_ok>=0.90, effR>=max(8,{EFF_RANK_MIN_FRAC}d))")
        print(f"  GEN  u_det = {r['gen_u_det']}  (gate-3 positive control, bar {U_DET_BAR})")
        print(f"  JEPA u_det = {r['jepa_u_det']}  (GATE 4, bar {U_DET_BAR}) -> "
              f"{'PASS' if r['jepa_u_det'] >= U_DET_BAR else 'FAIL (blocked_by_unfaithful_jepa)'}")
        print(f"  z_flip GEN/JEPA = {r['gen_z_flip_acc']}/{r['jepa_z_flip_acc']}  gap={r['z_flip_gap']:+.3f} "
              f"(delta {FROZEN_DELTA})")
        print(f"  GATE: smoke proceeds to the lock battery ONLY if JEPA u_det >= {U_DET_BAR}.")


if __name__ == "__main__":
    main()
