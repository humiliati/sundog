#!/usr/bin/env python
"""Chat-v2 Phase 0 - minimum falsifiable body-resistance cell (Amendment 1).

Synthetic H-factor channel process. Trains a small from-scratch transformer
(generative / next-token) and a control-only twin (predicts z_1 directly). For
each complexity H, measures an INFORMATION-BASIS regime-2 fingerprint that is
robust to transformer "massive-activation" outlier features (which masked the
variance-based measures in the first run -- see docs Amendment 1):

    d_dec               decodable dimensionality = effective rank of the per-latent
                        linear read-out directions  (un-masked replacement for eff_dim)
    cross_latent_leak   can the 1-D z_1 shadow predict the OTHER latents?
                        ~chance => state-insufficient = body RESISTS  (replaces FVE)
    z_recover           per-latent linear decodability (variance-agnostic; trustworthy)
    outlier_*           three-way test of the high-variance directions:
                        sundog (carry latents) / atmosphere (latents die on removal)
                        / separate weather (latents survive removal)

SHARP (per H, generative) iff  d_dec >= H/2  &&  z1_acc >= 0.70  &&
    cross_latent_leak <= 0.58  &&  d_dec >= 1.5 * twin_d_dec.

Train and measure are decoupled: the train stage saves bodies to
<out>/bodies/*.npz so re-measurement is instant.  Pre-reg:
docs/chatv2/PHASE0_MINIMUM_FALSIFIABLE.md.  NOT a public surface; no promotion.
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
    k_outlier: int = 2               # # top-variance dirs treated as "outlier medium"
    seed: int = 0
    # sharpness thresholds (information-basis, Amendment 1)
    d_dec_frac_min: float = 0.5      # gen d_dec >= H/2
    z1_ctrl_min: float = 0.70        # control-sufficient
    leak_max: float = 0.58           # ~chance => resists
    twin_contrast_ratio: float = 1.5 # gen d_dec >= 1.5 * twin d_dec

    @property
    def max_H(self) -> int:
        return max(self.h_sweep)

    @property
    def max_len(self) -> int:
        return self.bits_per_channel * self.max_H


def smoke_cfg() -> Cfg:
    return Cfg(
        mode="smoke", h_sweep=[1, 2], bits_per_channel=12, d_model=64,
        n_layers=2, n_heads=4, max_steps=300, eval_every=100, patience=3,
        n_fingerprint=600,
    )


# --------------------------------------------------------------------------- #
# synthetic H-factor channel data
# --------------------------------------------------------------------------- #
def gen_batch(H, batch, bits_per_channel, delta, rng):
    """Interleaved round-robin channels; channel i's bit ~ Bernoulli(0.5 +/- delta) by z_i."""
    L = bits_per_channel * H
    z = rng.integers(0, 2, size=(batch, H)).astype(np.int64)
    p_chan = 0.5 + delta * (2.0 * z - 1.0)
    chan = np.arange(L) % H
    p_pos = p_chan[:, chan]
    bits = (rng.random((batch, L)) < p_pos).astype(np.int64)
    return bits, z


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
        pos = torch.arange(L, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
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
def train_generative(H, cfg, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyGPT(2, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.01)
    eb, _ = gen_batch(H, cfg.eval_batch, cfg.bits_per_channel, cfg.delta,
                      np.random.default_rng(seed + 999))
    et = torch.tensor(eb, device=device)
    best, bad, steps = float("inf"), 0, 0
    for step in range(cfg.max_steps):
        bits, _ = gen_batch(H, cfg.batch, cfg.bits_per_channel, cfg.delta, rng)
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
                if bad >= cfg.patience:
                    break
    p = 0.5 + cfg.delta
    bayes = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return model, {"eval_loss": float(best), "bayes_floor": float(bayes), "steps": steps}


def train_twin(H, cfg, device, seed):
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 7)
    backbone = TinyGPT(2, cfg.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len).to(device)
    twin = TwinHead(backbone, cfg.d_model).to(device)
    opt = torch.optim.AdamW(twin.parameters(), lr=cfg.lr, weight_decay=0.01)
    acc = 0.0
    for step in range(cfg.max_steps):
        bits, z = gen_batch(H, cfg.batch, cfg.bits_per_channel, cfg.delta, rng)
        idx = torch.tensor(bits, device=device)
        y = torch.tensor(z[:, 0], dtype=torch.float32, device=device)
        z1 = twin(idx)
        loss = F.binary_cross_entropy_with_logits(z1, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % cfg.eval_every == 0:
            acc = float(((z1.detach() > 0).float() == y).float().mean())
    return twin, {"train_acc_z1": acc}


def extract_body(model, H, cfg, device, seed):
    rng = np.random.default_rng(seed + 123)
    bits, z = gen_batch(H, cfg.n_fingerprint, cfg.bits_per_channel, cfg.delta, rng)
    idx = torch.tensor(bits, device=device)
    model.eval()
    with torch.no_grad():
        _, hiddens = model(idx, return_hidden=True)
    bodies = np.stack([h[:, -1, :].cpu().numpy() for h in hiddens], axis=1)
    return bodies, z


# --------------------------------------------------------------------------- #
# information-basis fingerprint (Amendment 1)
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
    """Per-latent logistic read-direction (unit norm) + held-out per-latent acc."""
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
    """Effective rank (singular-value participation ratio) of the read-out set."""
    s = np.linalg.svd(W, compute_uv=False)
    return _pr(s ** 2)


def cross_latent_leak(X, z, shadow=0):
    """Predict the OTHER latents from the 1-D z_shadow read-direction projection."""
    Xs = _std(X)
    if len(np.unique(z[:, shadow])) < 2:
        return float("nan"), []
    lr = LogisticRegression(max_iter=1000).fit(Xs, z[:, shadow])
    s = (Xs @ lr.coef_.ravel())[:, None]
    accs = [_cv(s, z[:, j]) for j in range(z.shape[1]) if j != shadow]
    return (float(np.nanmean(accs)) if accs else float("nan")), accs


def outlier_analysis(X, z, k_outlier):
    """sundog (outliers carry latents) / atmosphere (latents die on removal) / weather."""
    Xs = _std(X)
    V = PCA(n_components=min(k_outlier, Xs.shape[1])).fit(Xs).components_
    top = Xs @ V.T                                   # (N,k) outlier-subspace coords
    carry = [_cv(top, z[:, i]) for i in range(z.shape[1])]
    Xr = Xs - top @ V                                # remove the outlier medium
    survive = [_cv(Xr, z[:, i]) for i in range(z.shape[1])]
    return float(np.nanmean(carry)), float(np.nanmean(survive))


def fingerprint(bodies, z, cfg):
    L = bodies.shape[1]
    zr_layers = [z_recover(bodies[:, l, :], z)[0] for l in range(L)]
    lstar = int(np.nanargmax(zr_layers))             # most latent-rich layer
    X = bodies[:, lstar, :]
    W, accs = latent_readout_dirs(X, z)
    leak_mean, _ = cross_latent_leak(X, z, 0)
    carry, survive = outlier_analysis(X, z, cfg.k_outlier)
    ed_raw = _pr(PCA().fit(_std(X)).explained_variance_)
    Xs = _std(X)
    V = PCA(n_components=cfg.k_outlier).fit(Xs).components_
    ed_rob = _pr(PCA().fit(Xs - (Xs @ V.T) @ V).explained_variance_)
    return {
        "body_layer": lstar,
        "zr_by_layer": [round(v, 3) for v in zr_layers],
        "d_dec": round(decodable_dim(W), 3),
        "eff_dim_raw": round(ed_raw, 3),
        "eff_dim_robust": round(ed_rob, 3),
        "z1_acc": round(accs[0], 4),
        "z_recover_mean": round(float(np.nanmean(accs)), 4),
        "z_recover_each": [round(a, 4) for a in accs],
        "cross_latent_leak": round(leak_mean, 4),
        "outlier_carries_latents": round(carry, 4),
        "latents_survive_outlier_removal": round(survive, 4),
    }


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #
def train_stage(cfg, out, device):
    bdir = out / "bodies"
    bdir.mkdir(parents=True, exist_ok=True)
    for H in cfg.h_sweep:
        hs = cfg.seed + 1000 * H
        t = time.time()
        print(f"[H={H}] training generative (L={cfg.bits_per_channel*H})...", flush=True)
        gen, gmeta = train_generative(H, cfg, device, hs)
        gb, gz = extract_body(gen, H, cfg, device, hs)
        np.savez(bdir / f"H{H}_gen.npz", bodies=gb, z=gz, meta=json.dumps(gmeta))
        print(f"[H={H}] gen saved: eval_loss={gmeta['eval_loss']:.4f} vs bayes "
              f"{gmeta['bayes_floor']:.4f} ({gmeta['steps']} steps); twin...", flush=True)
        twin, tmeta = train_twin(H, cfg, device, hs)
        tb, tz = extract_body(twin, H, cfg, device, hs)
        np.savez(bdir / f"H{H}_twin.npz", bodies=tb, z=tz, meta=json.dumps(tmeta))
        print(f"[H={H}] twin saved. ({round(time.time()-t,1)}s)", flush=True)


def measure_stage(cfg, out):
    bdir = out / "bodies"
    records = []
    for H in cfg.h_sweep:
        g = np.load(bdir / f"H{H}_gen.npz", allow_pickle=True)
        t = np.load(bdir / f"H{H}_twin.npz", allow_pickle=True)
        gfp = fingerprint(g["bodies"], g["z"], cfg)
        tfp = fingerprint(t["bodies"], t["z"], cfg)
        sharp = (gfp["d_dec"] >= cfg.d_dec_frac_min * H
                 and gfp["z1_acc"] >= cfg.z1_ctrl_min
                 and gfp["cross_latent_leak"] <= cfg.leak_max
                 and gfp["d_dec"] >= cfg.twin_contrast_ratio * max(tfp["d_dec"], 1e-9))
        records.append({"H": H, "generative": gfp, "twin": tfp,
                        "gen_train": json.loads(str(g["meta"])),
                        "twin_train": json.loads(str(t["meta"])), "sharp": bool(sharp)})
        print(f"[H={H:>2}] gen: d_dec={gfp['d_dec']:.1f}/{H} "
              f"eff(raw{gfp['eff_dim_raw']:.1f}/rob{gfp['eff_dim_robust']:.1f}) "
              f"z1={gfp['z1_acc']:.2f} zrec={gfp['z_recover_mean']:.2f} "
              f"leak={gfp['cross_latent_leak']:.2f} | outlier(carry={gfp['outlier_carries_latents']:.2f} "
              f"survive={gfp['latents_survive_outlier_removal']:.2f}) | "
              f"twin d_dec={tfp['d_dec']:.1f} zrec={tfp['z_recover_mean']:.2f} | SHARP={sharp}",
              flush=True)
    sharp_Hs = [r["H"] for r in records if r["sharp"]]
    H_star = min(sharp_Hs) if sharp_Hs else None
    verdict = "SHARP" if H_star is not None else "MARGINAL"
    manifest = {"lane": "chatv2", "phase": 0, "amendment": 1, "verdict": verdict,
                "H_star": H_star, "cfg": asdict(cfg), "records": records}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nVERDICT: {verdict}  H*={H_star}  -> {out/'manifest.json'}", flush=True)
    return manifest


def run(cfg, out):
    device = torch.device("cpu")
    print(f"[start] mode={cfg.mode} stage={cfg.stage} H={cfg.h_sweep} "
          f"threads={torch.get_num_threads()} d_model={cfg.d_model} "
          f"layers={cfg.n_layers} steps<={cfg.max_steps}", flush=True)
    t0 = time.time()
    if cfg.stage in ("train", "all"):
        train_stage(cfg, out, device)
    if cfg.stage in ("measure", "all"):
        measure_stage(cfg, out)
    print(f"[done] {round(time.time()-t0,1)}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--stage", choices=["train", "measure", "all"], default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = smoke_cfg() if args.mode == "smoke" else Cfg()
    cfg.seed = args.seed
    cfg.stage = args.stage
    torch.set_num_threads(4)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(f"results/chatv2/phase0-{args.mode}")
    run(cfg, out)


if __name__ == "__main__":
    main()
