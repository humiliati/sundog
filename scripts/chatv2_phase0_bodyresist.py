#!/usr/bin/env python
"""Chat-v2 Phase 0 / 0.2 - body-resistance cell.

Synthetic H-factor channel process; a small from-scratch transformer (generative
/ next-token) + a control-only twin (predicts z_1). Information-basis fingerprint
robust to transformer outlier features (Amendment 1).

--latent bias      : channel emission biased by a per-sequence z_i (Phase 0 /
                     Amendment 1).  z_i is LINEARLY decodable from the input
                     (passive floor) -> confounds the twin contrast.
--latent computed  : channel's next bit biased by the parity of its last
                     `window` bits (Phase 0.2).  z_i = that parity -> NONLINEAR,
                     not linearly decodable from the input (passive floor -> chance),
                     so the twin contrast is clean.  Gated by a mandatory
                     linear-input-probe pre-check (must be ~chance, else abort).

Fingerprint (per H, gen + twin): d_dec (read-out effective rank, un-masked),
cross_latent_leak (z1 shadow -> other latents; ~chance => resists), body_carry
(mean z_recover over the NON-decision latents = the strength-contrast axis),
z_recover, and the outlier carry/survive medium test.

SHARP (computed, per H, generative) iff d_dec>=H/2 && z1_acc>=0.70 &&
    cross_latent_leak<=0.58 && body_carry_gen>=0.70 &&
    (body_carry_gen - body_carry_twin)>=0.20.

Pre-reg: docs/chatv2/PHASE0_MINIMUM_FALSIFIABLE.md (+ Amendment 1) and
PHASE0_2_COMPUTED_LATENTS.md.  NOT a public surface; no promotion.
"""
import argparse
import json
import pathlib
import time
from dataclasses import dataclass, asdict, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    mode: str = "full"
    stage: str = "all"               # train | measure | all
    latent: str = "computed"         # bias | computed
    window: int = 2                  # parity window for the computed latent
    h_sweep: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16])
    bits_per_channel: int = 16
    delta: float = 0.35
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    lr: float = 3e-4
    batch: int = 128
    eval_batch: int = 512
    max_steps: int = 2500
    eval_every: int = 250
    patience: int = 4
    n_fingerprint: int = 3000
    k_outlier: int = 2
    seed: int = 0
    # curriculum / grok-aware training
    curriculum: bool = False         # each H warm-starts from the previous H's checkpoint
    warm_start: str = ""             # explicit gen checkpoint to warm-start the first H
    pos_h: int = 0                   # size the positional embedding for this H (0 = max(h_sweep));
                                     # fix across a curriculum so checkpoints transfer
    min_steps: int = 0               # no early-stop before this many steps (grok flat phase)
    twin_patience: int = 3           # twin early-stops on z_1 acc plateau
    save_ckpt: bool = True
    fair_readout: bool = False       # read each latent at its channel's freshest (last) position
                                     # instead of the global final position (computed layout only) --
                                     # a readout-FAIRNESS fix, not a threshold/metric change
    arity: int = 2                   # computed-latent: k-tuple parity arity (2 = pair-XOR baseline)
    m: int = 3                       # coupled-latent (Phase 7): hidden-source bit count (frozen)
    p_noise: float = 0.25            # coupled-latent (Phase 7): per-latent coupling noise
    # sharpness thresholds
    d_dec_frac_min: float = 0.5
    z1_ctrl_min: float = 0.70
    leak_max: float = 0.58
    body_carry_min: float = 0.70     # gen carries the non-decision state
    body_carry_gap_min: float = 0.20 # ... and the objective built it (vs twin)
    precheck_max: float = 0.60       # input-probe must be <= this (~chance)

    @property
    def max_len(self) -> int:
        return self.bits_per_channel * (self.pos_h if self.pos_h else max(self.h_sweep))


def smoke_cfg() -> Cfg:
    return Cfg(mode="smoke", h_sweep=[1, 2], bits_per_channel=12, d_model=64,
               n_layers=2, n_heads=4, max_steps=400, eval_every=100, patience=4,
               n_fingerprint=800)


# --------------------------------------------------------------------------- #
# synthetic data
# --------------------------------------------------------------------------- #
def _gen_bias(H, batch, bpc, delta, rng):
    """Phase 0: channel i's bit ~ Bernoulli(0.5 +/- delta) by per-sequence z_i."""
    L = bpc * H
    z = rng.integers(0, 2, size=(batch, H)).astype(np.int64)
    p_chan = 0.5 + delta * (2.0 * z - 1.0)
    bits = (rng.random((batch, L)) < p_chan[:, np.arange(L) % H]).astype(np.int64)
    return bits, z


def _gen_computed(H, batch, bpc, delta, rng, arity=2):
    """Phase 0.2 / R1: per-channel latent z_i encoded in the PARITY of independent
    k-tuples (k = arity). Channel i emits P = bpc//k tuples; the first k-1 bits are
    fair and the last = (parity of the first k-1) XOR x, with x ~ Bernoulli(0.5 +/-
    delta) by z_i, so each tuple's k-bit parity = x and NO bit or sub-tuple correlates
    with z_i (input-undecodable). Predicting the last bit forces the model to compute
    the k-parity and aggregate -> z_i is a maintained per-sequence latent. arity=2 is
    bit-identical to the original pair-XOR; arity=3 is 3-bit parity (harder)."""
    A = max(2, arity)
    P = max(1, bpc // A)
    z = rng.integers(0, 2, size=(batch, H)).astype(np.int64)
    chan = np.arange(P * H) % H
    px = (0.5 + delta * (2.0 * z - 1.0))[:, chan]            # (B, P*H) P(tuple parity = 1)
    x = (rng.random((batch, P * H)) < px).astype(np.int64)
    tup = rng.integers(0, 2, size=(batch, P * H, A)).astype(np.int64)
    tup[:, :, A - 1] = (tup[:, :, :A - 1].sum(2) % 2) ^ x    # last bit -> tuple parity = x
    bits = tup.reshape(batch, P * H * A)
    return bits, z


_COUPLE_A = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1],
                      [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.int64)


def _gen_coupled(H, batch, bpc, delta, rng, arity=2, m=3, p_noise=0.25):
    """Phase 7: H state latents are NOISY PARITIES of a shared m-bit hidden source u:
    z_i = (A_i . u mod 2) XOR x_i, x_i ~ Bernoulli(p_noise); A = the frozen coupling
    graph (`_COUPLE_A`, m=3, H<=8). Each z_i is then parity-channel encoded EXACTLY as
    _gen_computed (input-undecodable). Returns (bits, z, u) so the probe can target the
    hidden source u (the closure functional). Coupling lives in how z is GENERATED (shared
    u), not in the per-channel encoding -> the de-confound pre-check still applies per z_i."""
    assert m == 3 and H <= 8, "coupled latent frozen at m=3, H<=8 (Phase 7 spec)"
    A = _COUPLE_A[:H]
    u = rng.integers(0, 2, size=(batch, m)).astype(np.int64)
    sig = (u @ A.T) % 2                                          # (B, H) parity(u, A_i)
    z = (sig ^ (rng.random((batch, H)) < p_noise).astype(np.int64)).astype(np.int64)
    a = max(2, arity); P = max(1, bpc // a)
    chan = np.arange(P * H) % H
    px = (0.5 + delta * (2.0 * z - 1.0))[:, chan]
    x = (rng.random((batch, P * H)) < px).astype(np.int64)
    tup = rng.integers(0, 2, size=(batch, P * H, a)).astype(np.int64)
    tup[:, :, a - 1] = (tup[:, :, :a - 1].sum(2) % 2) ^ x
    return tup.reshape(batch, P * H * a), z, u


def gen_batch(H, batch, cfg, rng):
    if cfg.latent == "bias":
        b, z = _gen_bias(H, batch, cfg.bits_per_channel, cfg.delta, rng)
        return b, z, None
    if cfg.latent == "coupled":
        return _gen_coupled(H, batch, cfg.bits_per_channel, cfg.delta, rng,
                            cfg.arity, cfg.m, cfg.p_noise)
    b, z = _gen_computed(H, batch, cfg.bits_per_channel, cfg.delta, rng, cfg.arity)
    return b, z, None


# --------------------------------------------------------------------------- #
# tiny GPT
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, d, nh):
        super().__init__()
        self.nh, self.hd = nh, d // nh
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.qkv(self.ln1(x)).view(B, L, 3, self.nh, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, L, D))
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, nh, max_len):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList([Block(d, nh) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab)

    def forward(self, idx, return_hidden=False):
        B, L = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(L, device=idx.device))[None]
        hiddens = [x]
        for blk in self.blocks:
            x = blk(x)
            if return_hidden:
                hiddens.append(x)
        logits = self.head(self.ln_f(x))
        return (logits, hiddens) if return_hidden else logits


class TwinHead(nn.Module):
    def __init__(self, backbone, d):
        super().__init__()
        self.backbone = backbone
        self.clf = nn.Linear(d, 1)

    def forward(self, idx, return_hidden=False):
        _, hiddens = self.backbone(idx, return_hidden=True)
        z1 = self.clf(hiddens[-1][:, -1, :]).squeeze(-1)
        return (z1, hiddens) if return_hidden else z1


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_generative(H, cfg, device, seed, warm_start=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyGPT(2, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len).to(device)
    if warm_start:
        model.load_state_dict(torch.load(warm_start, map_location=device))
        print(f"[H={H}] warm-started gen from {warm_start}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    eb, _, _ = gen_batch(H, cfg.eval_batch, cfg, np.random.default_rng(seed + 999))
    et = torch.tensor(eb, device=device)
    best, bad, steps = float("inf"), 0, 0
    for step in range(cfg.max_steps):
        bits, _, _ = gen_batch(H, cfg.batch, cfg, rng)
        idx = torch.tensor(bits, device=device)
        loss = F.cross_entropy(model(idx)[:, :-1].reshape(-1, 2), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        steps = step + 1
        if steps % cfg.eval_every == 0:
            model.eval()
            with torch.no_grad():
                el = F.cross_entropy(model(et)[:, :-1].reshape(-1, 2),
                                     et[:, 1:].reshape(-1)).item()
            model.train()
            if el < best - 1e-4:
                best, bad = el, 0
            else:
                bad += 1
                # grok-aware: never quit during the flat pre-grok phase
                if bad >= cfg.patience and steps >= cfg.min_steps:
                    break
    p = 0.5 + cfg.delta
    bayes_pred = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    # computed: ~half the tokens (fair `u`) are unpredictable -> mixed floor
    bayes = 0.5 * np.log(2.0) + 0.5 * bayes_pred if cfg.latent in ("computed", "coupled") else bayes_pred
    return model, {"eval_loss": float(best), "bayes_floor": float(bayes), "steps": steps}


def train_twin(H, cfg, device, seed):
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 7)
    backbone = TinyGPT(2, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len).to(device)
    twin = TwinHead(backbone, cfg.d_model).to(device)
    opt = torch.optim.AdamW(twin.parameters(), lr=cfg.lr, weight_decay=0.01)
    eb, ez, _ = gen_batch(H, cfg.eval_batch, cfg, np.random.default_rng(seed + 888))
    et = torch.tensor(eb, device=device)
    ey = torch.tensor(ez[:, 0], dtype=torch.float32, device=device)
    best_acc, bad, steps = 0.0, 0, 0
    for step in range(cfg.max_steps):
        bits, z, _ = gen_batch(H, cfg.batch, cfg, rng)
        idx = torch.tensor(bits, device=device)
        y = torch.tensor(z[:, 0], dtype=torch.float32, device=device)
        loss = F.binary_cross_entropy_with_logits(twin(idx), y)
        opt.zero_grad(); loss.backward(); opt.step()
        steps = step + 1
        if steps % cfg.eval_every == 0:
            twin.eval()
            with torch.no_grad():
                acc = float(((twin(et) > 0).float() == ey).float().mean())
            twin.train()
            # twin learns z_1 fast and does not grok -> early-stop freely (no min_steps)
            if acc > best_acc + 1e-3:
                best_acc, bad = acc, 0
            else:
                bad += 1
                if bad >= cfg.twin_patience:
                    break
    return twin, {"train_acc_z1": float(best_acc), "steps": steps}


def _lastpos(H, bpc, arity=2):
    """Computed k-tuple layout (k=arity): P=bpc//k tuples/channel; channel i's last
    token at k*((P-1)*H + i) + (k-1). Each latent's freshest read position."""
    A = max(2, arity)
    P = max(1, bpc // A)
    return [A * ((P - 1) * H + i) + (A - 1) for i in range(H)]


def extract_body(model, H, cfg, device, seed):
    rng = np.random.default_rng(seed + 123)
    bits, z, u = gen_batch(H, cfg.n_fingerprint, cfg, rng)
    idx = torch.tensor(bits, device=device)
    model.eval()
    with torch.no_grad():
        _, hiddens = model(idx, return_hidden=True)
    if cfg.fair_readout:
        pos = _lastpos(H, cfg.bits_per_channel, cfg.arity)   # H per-channel freshest positions
        bodies = np.stack([np.stack([h[:, p, :].cpu().numpy() for p in pos], axis=1)
                           for h in hiddens], axis=1)   # (N, layers, H, d)
    else:
        bodies = np.stack([h[:, -1, :].cpu().numpy() for h in hiddens], axis=1)  # (N, layers, d)
    return bodies, z, u


# --------------------------------------------------------------------------- #
# information-basis fingerprint
# --------------------------------------------------------------------------- #
def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def _pr(vals):
    vals = np.asarray(vals, dtype=float)
    return float((vals.sum() ** 2) / (np.square(vals).sum() + 1e-12))


def _cv(X, y):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=4).mean())


def z_recover(X, z):
    Xs = _std(X)
    accs = [_cv(Xs, z[:, i]) for i in range(z.shape[1])]
    return float(np.nanmean(accs)), accs


def latent_readout_dirs(X, z):
    Xs = _std(X)
    W, accs = [], []
    for i in range(z.shape[1]):
        if len(np.unique(z[:, i])) < 2:
            W.append(np.zeros(Xs.shape[1])); accs.append(float("nan")); continue
        lr = LogisticRegression(max_iter=1000).fit(Xs, z[:, i])
        w = lr.coef_.ravel()
        W.append(w / (np.linalg.norm(w) + 1e-12))
        accs.append(_cv(Xs, z[:, i]))
    return np.array(W), accs


def decodable_dim(W):
    return _pr(np.linalg.svd(W, compute_uv=False) ** 2)


def cross_latent_leak(X, z, shadow=0):
    Xs = _std(X)
    if len(np.unique(z[:, shadow])) < 2:
        return float("nan")
    lr = LogisticRegression(max_iter=1000).fit(Xs, z[:, shadow])
    s = (Xs @ lr.coef_.ravel())[:, None]
    accs = [_cv(s, z[:, j]) for j in range(z.shape[1]) if j != shadow]
    return float(np.nanmean(accs)) if accs else float("nan")


def outlier_analysis(X, z, k_outlier):
    Xs = _std(X)
    V = PCA(n_components=min(k_outlier, Xs.shape[1])).fit(Xs).components_
    top = Xs @ V.T
    carry = [_cv(top, z[:, i]) for i in range(z.shape[1])]
    Xr = Xs - top @ V
    survive = [_cv(Xr, z[:, i]) for i in range(z.shape[1])]
    return float(np.nanmean(carry)), float(np.nanmean(survive))


def fingerprint(bodies, z, cfg):
    """bodies (N, layers, d) = global final-position body; or (N, layers, H, d) =
    per-latent freshest-position views (fair_readout). Each latent i is read from
    view(l, i); for the non-fair body that view is the same final body for all i,
    so behavior is identical to before. Corroborating variance/outlier reads use
    the final-token view (= channel H-1's freshest position)."""
    fair = bodies.ndim == 4
    Ln, H = bodies.shape[1], z.shape[1]

    def view(l, i):
        return bodies[:, l, i, :] if fair else bodies[:, l, :]

    zr_layers = [float(np.nanmean([_cv(_std(view(l, i)), z[:, i]) for i in range(H)]))
                 for l in range(Ln)]
    lstar = int(np.nanargmax(zr_layers))

    W, accs = [], []
    for i in range(H):
        Xi = _std(view(lstar, i))
        if len(np.unique(z[:, i])) < 2:
            W.append(np.zeros(Xi.shape[1])); accs.append(float("nan")); continue
        lr = LogisticRegression(max_iter=1000).fit(Xi, z[:, i])
        w = lr.coef_.ravel()
        W.append(w / (np.linalg.norm(w) + 1e-12)); accs.append(_cv(Xi, z[:, i]))
    W = np.array(W)

    X0 = _std(view(lstar, 0))
    if len(np.unique(z[:, 0])) > 1 and H > 1:
        s = (X0 @ LogisticRegression(max_iter=1000).fit(X0, z[:, 0]).coef_.ravel())[:, None]
        leak = float(np.nanmean([_cv(s, z[:, j]) for j in range(1, H)]))
    else:
        leak = float("nan")
    body_carry = float(np.nanmean(accs[1:])) if H > 1 else float("nan")

    Xf = _std(view(lstar, H - 1))                          # final-token view for variance/outlier
    Vk = PCA(n_components=min(cfg.k_outlier, Xf.shape[1])).fit(Xf).components_
    ed_rob = _pr(PCA().fit(Xf - (Xf @ Vk.T) @ Vk).explained_variance_)
    carry, survive = outlier_analysis(view(lstar, H - 1), z, cfg.k_outlier)
    return {
        "fair_readout": fair,
        "body_layer": lstar,
        "zr_by_layer": [round(v, 3) for v in zr_layers],
        "d_dec": round(decodable_dim(W), 3),
        "eff_dim_raw": round(_pr(PCA().fit(Xf).explained_variance_), 3),
        "eff_dim_robust": round(ed_rob, 3),
        "z1_acc": round(accs[0], 4),
        "z_recover_mean": round(float(np.nanmean(accs)), 4),
        "body_carry": round(body_carry, 4),
        "z_recover_each": [round(a, 4) for a in accs],
        "cross_latent_leak": round(leak, 4),
        "outlier_carries_latents": round(carry, 4),
        "latents_survive_outlier_removal": round(survive, 4),
    }


# --------------------------------------------------------------------------- #
# de-confound pre-check (gates the run for computed latents)
# --------------------------------------------------------------------------- #
def input_probe_precheck(cfg):
    res = {}
    for H in cfg.h_sweep:
        rng = np.random.default_rng(cfg.seed + 555 + H)
        bits, z, _ = gen_batch(H, cfg.n_fingerprint, cfg, rng)
        X = bits.astype(float)
        res[H] = float(np.nanmean([_cv(X, z[:, i]) for i in range(H)]))
    return res


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def train_stage(cfg, out, device):
    bdir = out / "bodies"; bdir.mkdir(parents=True, exist_ok=True)
    cdir = out / "ckpt"; cdir.mkdir(parents=True, exist_ok=True)
    chance = float(np.log(2.0))
    prev_ckpt = cfg.warm_start or None
    for H in cfg.h_sweep:
        hs = cfg.seed + 1000 * H
        t = time.time()
        ws = prev_ckpt if cfg.curriculum else (cfg.warm_start or None)
        print(f"[H={H}] training generative (L={cfg.bits_per_channel*H})"
              f"{' [warm]' if ws else ''}...", flush=True)
        gen, gmeta = train_generative(H, cfg, device, hs, warm_start=ws)
        gb, gz, gu = extract_body(gen, H, cfg, device, hs)
        np.savez(bdir / f"H{H}_gen.npz", bodies=gb, z=gz, u=gu, meta=json.dumps(gmeta))
        learned = gmeta["eval_loss"] < chance - 0.02
        if cfg.save_ckpt:
            torch.save(gen.state_dict(), cdir / f"H{H}_gen.pt")
            if learned:
                prev_ckpt = str(cdir / f"H{H}_gen.pt")   # curriculum warm-starts from last LEARNED rung
        print(f"[H={H}] gen saved: eval_loss={gmeta['eval_loss']:.4f} vs bayes "
              f"{gmeta['bayes_floor']:.4f} ({gmeta['steps']} steps) learned={learned}", flush=True)
        if learned:
            twin, tmeta = train_twin(H, cfg, device, hs)
            tb, tz, tu = extract_body(twin, H, cfg, device, hs)
            np.savez(bdir / f"H{H}_twin.npz", bodies=tb, z=tz, u=tu, meta=json.dumps(tmeta))
            print(f"[H={H}] twin saved ({tmeta['steps']} steps). ({round(time.time()-t,1)}s)", flush=True)
        else:
            print(f"[H={H}] gen UNLEARNED -> skipping twin (no contrast vs an unlearned gen). "
                  f"({round(time.time()-t,1)}s)", flush=True)


def measure_stage(cfg, out, precheck=None):
    bdir = out / "bodies"
    records = []
    chance = float(np.log(2.0))
    for H in cfg.h_sweep:
        g = np.load(bdir / f"H{H}_gen.npz", allow_pickle=True)
        gfp = fingerprint(g["bodies"], g["z"], cfg)
        gmeta = json.loads(str(g["meta"]))
        # UNLEARNED guard: an H whose gen never left chance is not a body-resistance
        # read at all (F3'), not a "marginal" -- never conflate the two.
        learned = gmeta["eval_loss"] < chance - 0.02
        tpath = bdir / f"H{H}_twin.npz"
        if tpath.exists():
            t = np.load(tpath, allow_pickle=True)
            tfp, tmeta = fingerprint(t["bodies"], t["z"], cfg), json.loads(str(t["meta"]))
        else:
            tfp, tmeta = None, None                 # gen UNLEARNED -> twin was skipped
        tbc = tfp["body_carry"] if tfp else float("nan")
        bc_ok = (tfp is not None and H > 1 and gfp["body_carry"] >= cfg.body_carry_min
                 and (gfp["body_carry"] - tbc) >= cfg.body_carry_gap_min)
        sharp = bool(learned and gfp["d_dec"] >= cfg.d_dec_frac_min * H
                     and gfp["z1_acc"] >= cfg.z1_ctrl_min
                     and gfp["cross_latent_leak"] <= cfg.leak_max and bc_ok)
        status = "UNLEARNED" if not learned else ("SHARP" if sharp else "MARGINAL")
        records.append({"H": H, "generative": gfp, "twin": tfp, "gen_train": gmeta,
                        "twin_train": tmeta, "sharp": sharp, "learned": learned, "status": status})
        print(f"[H={H:>2}] {status:<9} gen: d_dec={gfp['d_dec']:.1f}/{H} z1={gfp['z1_acc']:.2f} "
              f"leak={gfp['cross_latent_leak']:.2f} body_carry={gfp['body_carry']:.2f} "
              f"(twin {(f'{tbc:.2f}' if tfp else 'skip')}) eval_loss={gmeta['eval_loss']:.3f}", flush=True)
    sharp_Hs = [r["H"] for r in records if r["sharp"]]
    H_star = min(sharp_Hs) if sharp_Hs else None
    verdict = "SHARP" if H_star is not None else "MARGINAL"
    manifest = {"lane": "chatv2",
                "phase": "7" if cfg.latent == "coupled" else "0.2" if cfg.latent == "computed" else "0",
                "verdict": verdict, "H_star": H_star, "precheck": precheck,
                "cfg": asdict(cfg), "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nVERDICT: {verdict}  H*={H_star}  -> {out/'manifest.json'}", flush=True)
    return manifest


def run(cfg, out):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # GPU when available (LATTICE)
    print(f"[start] device={device} mode={cfg.mode} stage={cfg.stage} latent={cfg.latent} "
          f"arity={cfg.arity} H={cfg.h_sweep} d_model={cfg.d_model}/{cfg.n_layers}L/{cfg.n_heads}H "
          f"lr={cfg.lr} delta={cfg.delta} steps<={cfg.max_steps} fair={cfg.fair_readout}", flush=True)
    t0 = time.time()
    precheck = None
    if cfg.latent in ("computed", "coupled"):
        print("[precheck] linear input-probe (must be ~chance)...", flush=True)
        precheck = input_probe_precheck(cfg)
        for H, a in precheck.items():
            print(f"  H={H}: input-probe mean acc = {a:.3f}", flush=True)
        worst = max(precheck.values())
        if worst > cfg.precheck_max:
            print(f"[ABORT] de-confound FAILED (F2'): input-probe {worst:.3f} > "
                  f"{cfg.precheck_max}; latents are linearly input-decodable.", flush=True)
            out.mkdir(parents=True, exist_ok=True)
            (out / "precheck_failed.json").write_text(json.dumps(precheck, indent=2))
            return
        print(f"[precheck] PASS (worst {worst:.3f} <= {cfg.precheck_max})", flush=True)
    if cfg.stage in ("train", "all"):
        train_stage(cfg, out, device)
    if cfg.stage in ("measure", "all"):
        measure_stage(cfg, out, precheck)
    print(f"[done] {round(time.time()-t0,1)}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--stage", choices=["train", "measure", "all"], default="all")
    ap.add_argument("--latent", choices=["bias", "computed", "coupled"], default=None)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--h-sweep", default=None, help="comma-sep H values, e.g. 8 or 2,4,8")
    ap.add_argument("--d-model", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--delta", type=float, default=None)
    ap.add_argument("--bits-per-channel", type=int, default=None)
    ap.add_argument("--curriculum", action="store_true", help="each H warm-starts from prev H ckpt")
    ap.add_argument("--warm-start", default=None, help="gen checkpoint to warm-start the first H")
    ap.add_argument("--pos-h", type=int, default=None, help="size pos-emb for this H (fix across curriculum)")
    ap.add_argument("--min-steps", type=int, default=None, help="no early-stop before this (grok floor)")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--fair-readout", action="store_true", help="read each latent at its channel's freshest position")
    ap.add_argument("--arity", type=int, default=None, help="computed-latent k-tuple parity arity (2=pair-XOR)")
    ap.add_argument("--p-noise", type=float, default=None, help="coupled-latent per-latent coupling noise (Phase 7)")
    ap.add_argument("--n-layers", type=int, default=None)
    ap.add_argument("--n-heads", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = smoke_cfg() if args.mode == "smoke" else Cfg()
    cfg.seed, cfg.stage = args.seed, args.stage
    if args.latent:
        cfg.latent = args.latent
    if args.window:
        cfg.window = args.window
    if args.h_sweep:
        cfg.h_sweep = [int(x) for x in args.h_sweep.split(",")]
    if args.d_model:
        cfg.d_model = args.d_model
    if args.max_steps:
        cfg.max_steps = args.max_steps
    if args.delta is not None:
        cfg.delta = args.delta
    if args.bits_per_channel:
        cfg.bits_per_channel = args.bits_per_channel
    if args.curriculum:
        cfg.curriculum = True
    if args.warm_start:
        cfg.warm_start = args.warm_start
    if args.pos_h:
        cfg.pos_h = args.pos_h
    if args.min_steps is not None:
        cfg.min_steps = args.min_steps
    if args.patience:
        cfg.patience = args.patience
    if args.fair_readout:
        cfg.fair_readout = True
    if args.arity:
        cfg.arity = args.arity
    if args.p_noise is not None:
        cfg.p_noise = args.p_noise
    if args.n_layers:
        cfg.n_layers = args.n_layers
    if args.n_heads:
        cfg.n_heads = args.n_heads
    if args.lr is not None:
        cfg.lr = args.lr
    torch.set_num_threads(4)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(f"results/chatv2/phase0-{args.mode}")
    run(cfg, out)


if __name__ == "__main__":
    main()
