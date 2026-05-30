#!/usr/bin/env python
"""Chat-v2 Phase 0 - minimum falsifiable body-resistance cell.

Synthetic H-factor channel process. Trains a small from-scratch transformer
(generative / next-token) and a control-only twin (predicts z_1 directly). For
each complexity H, measures the three-number regime-2 fingerprint ported from
the C1/Mesa work:

    eff_dim(body)          participation ratio of the residual stream
    k_control              smallest PCA-subspace dim that reads z_1 at ~full acc
    FVE(body | shadow)     how much of the body the z_1 control-shadow rebuilds
                           (nonlinear HGB R^2, perm-controlled) -- LOW = resists

SHARP iff there is an H* where the body is high-dim, the z_1 shadow is compact
(k_control << eff_dim), and FVE is low (body not slaved to the decision) -- and
the control-only twin is marginal at that H (objective, not data, drives it).

Pre-registration: docs/chatv2/PHASE0_MINIMUM_FALSIFIABLE.md
NOT a public surface; no promotion.
"""
import argparse
import json
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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
@dataclass
class Cfg:
    mode: str = "full"
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
    fve_targets: int = 20            # top-M body PCs to reconstruct
    ks: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    seed: int = 0
    # sharpness thresholds (toy uses ratio-based; absolute floor is a sanity gate)
    eff_dim_floor: float = 6.0
    ratio_max: float = 0.20          # k_control / eff_dim
    fve_max: float = 0.60            # body resists
    fve_perm_max: float = 0.12       # perm control must be clean
    z_recover_min: float = 0.70      # factors really present (anti-F3)

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
        n_fingerprint=600, fve_targets=12, ks=[1, 2, 4, 8, 16],
    )


# --------------------------------------------------------------------------- #
# synthetic H-factor channel data
# --------------------------------------------------------------------------- #
def gen_batch(H, batch, bits_per_channel, delta, rng):
    """Interleaved round-robin channels; channel i's bit ~ Bernoulli(0.5 +/- delta) by z_i."""
    L = bits_per_channel * H
    z = rng.integers(0, 2, size=(batch, H)).astype(np.int64)       # (B,H) latents
    p_chan = 0.5 + delta * (2.0 * z - 1.0)                          # (B,H) in {.5-d,.5+d}
    chan = np.arange(L) % H                                         # (L,) channel of each pos
    p_pos = p_chan[:, chan]                                         # (B,L)
    bits = (rng.random((batch, L)) < p_pos).astype(np.int64)       # (B,L)
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
    """Same backbone, control-only objective: predict z_1 at the final position."""
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
        logits = model(idx)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, 2), idx[:, 1:].reshape(-1))
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
    # bayes-optimal next-token loss floor (per biased bit), for context
    p = 0.5 + cfg.delta
    bayes = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return model, {"eval_loss": best, "bayes_floor": float(bayes), "steps": steps}


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


# --------------------------------------------------------------------------- #
# fingerprint
# --------------------------------------------------------------------------- #
def _std(X):
    return (X - X.mean(0)) / (X.std(0) + 1e-8)


def eff_dim(X):
    ev = PCA().fit(_std(X)).explained_variance_
    return float((ev.sum() ** 2) / (np.square(ev).sum() + 1e-12))


def k_control(X, y, ks):
    Xs = _std(X)
    scores = PCA().fit_transform(Xs)
    accs = {}
    for k in ks:
        if k > scores.shape[1]:
            continue
        accs[k] = float(cross_val_score(LogisticRegression(max_iter=1000),
                                        scores[:, :k], y, cv=4).mean())
    full = accs[max(accs)]
    kc = min((k for k in accs if accs[k] >= full - 0.02), default=max(accs))
    return int(kc), accs, float(full)


def z_recover(X, z):
    Xs = _std(X)
    accs = [float(cross_val_score(LogisticRegression(max_iter=1000), Xs, z[:, i], cv=4).mean())
            for i in range(z.shape[1])]
    return float(np.mean(accs)), accs


def fve_body_given_shadow(X, y, M, seed):
    """Shadow = body projected onto the z_1 logistic read-direction (1-D control summary).
    FVE = energy-weighted held-out HGB R^2 reconstructing the top-M body PCs from it."""
    Xs = _std(X)
    lr = LogisticRegression(max_iter=1000).fit(Xs, y)
    shadow = (Xs @ lr.coef_.ravel())[:, None]                       # (N,1)
    pca = PCA(n_components=min(M, Xs.shape[1])).fit(Xs)
    Y, ev = pca.transform(Xs), pca.explained_variance_
    n = len(shadow); cut = int(n * 0.7)

    def _fve(sv):
        r2 = []
        for m in range(Y.shape[1]):
            reg = HistGradientBoostingRegressor(max_iter=150, random_state=seed)
            reg.fit(sv[:cut], Y[:cut, m])
            r2.append(max(0.0, r2_score(Y[cut:, m], reg.predict(sv[cut:]))))
        return float((np.array(r2) * ev).sum() / (ev.sum() + 1e-12))

    real = _fve(shadow)
    perm = _fve(shadow[np.random.default_rng(seed).permutation(n)])
    return real, perm


def fingerprint(bodies, z, cfg, seed):
    effs = [eff_dim(bodies[:, l, :]) for l in range(bodies.shape[1])]
    lstar = int(np.argmax(effs))
    Xb, y = bodies[:, lstar, :], z[:, 0]
    kc, accs, full = k_control(Xb, y, cfg.ks)
    fve, fve_perm = fve_body_given_shadow(Xb, y, cfg.fve_targets, seed)
    zr, zr_each = z_recover(Xb, z)
    return {
        "eff_dim_by_layer": [round(e, 3) for e in effs],
        "body_layer": lstar,
        "eff_dim": round(effs[lstar], 3),
        "k_control": kc,
        "k_control_accs": {str(k): round(a, 4) for k, a in accs.items()},
        "z1_full_acc": round(full, 4),
        "fve_body_given_z1": round(fve, 4),
        "fve_perm": round(fve_perm, 4),
        "z_recover_mean": round(zr, 4),
        "z_recover_each": [round(a, 4) for a in zr_each],
    }


# --------------------------------------------------------------------------- #
# sweep
# --------------------------------------------------------------------------- #
def run(cfg, out_dir):
    device = torch.device("cpu")
    t0 = time.time()
    records = []
    print(f"[start] mode={cfg.mode} H_sweep={cfg.h_sweep} threads={torch.get_num_threads()} "
          f"d_model={cfg.d_model} layers={cfg.n_layers} steps<={cfg.max_steps}", flush=True)
    for H in cfg.h_sweep:
        hs = cfg.seed + 1000 * H
        tg = time.time()
        print(f"[H={H}] training generative (L={cfg.bits_per_channel*H})...", flush=True)
        gen, gmeta = train_generative(H, cfg, device, hs)
        print(f"[H={H}] gen done: eval_loss={gmeta['eval_loss']:.4f} vs bayes "
              f"{gmeta['bayes_floor']:.4f} ({gmeta['steps']} steps); fingerprint+twin...", flush=True)
        gbodies, gz = extract_body(gen, H, cfg, device, hs, twin=False)
        gfp = fingerprint(gbodies, gz, cfg, hs)

        twin, tmeta = train_twin(H, cfg, device, hs)
        tbodies, tz = extract_body(twin, H, cfg, device, hs, twin=True)
        tfp = fingerprint(tbodies, tz, cfg, hs)

        sharp = (gfp["eff_dim"] >= cfg.eff_dim_floor
                 and gfp["k_control"] / max(gfp["eff_dim"], 1e-9) <= cfg.ratio_max
                 and gfp["fve_body_given_z1"] <= cfg.fve_max
                 and gfp["fve_perm"] <= cfg.fve_perm_max
                 and gfp["z_recover_mean"] >= cfg.z_recover_min)
        rec = {"H": H, "generative": gfp, "gen_train": gmeta,
               "twin": tfp, "twin_train": tmeta, "sharp": bool(sharp),
               "secs": round(time.time() - tg, 1)}
        records.append(rec)
        print(f"[H={H:>2}] gen: effdim={gfp['eff_dim']:.1f} kctrl={gfp['k_control']} "
              f"FVE={gfp['fve_body_given_z1']:.3f}(perm {gfp['fve_perm']:.3f}) "
              f"zrec={gfp['z_recover_mean']:.2f} z1acc={gfp['z1_full_acc']:.2f} "
              f"| twin: effdim={tfp['eff_dim']:.1f} FVE={tfp['fve_body_given_z1']:.3f} "
              f"| SHARP={sharp} ({rec['secs']}s)", flush=True)

    sharp_Hs = [r["H"] for r in records if r["sharp"]]
    H_star = min(sharp_Hs) if sharp_Hs else None
    verdict = "SHARP" if H_star is not None else "MARGINAL"
    manifest = {
        "lane": "chatv2", "phase": 0, "verdict": verdict, "H_star": H_star,
        "cfg": asdict(cfg), "records": records,
        "total_secs": round(time.time() - t0, 1),
        "sharp_criteria": {
            "eff_dim_floor": cfg.eff_dim_floor, "ratio_max": cfg.ratio_max,
            "fve_max": cfg.fve_max, "fve_perm_max": cfg.fve_perm_max,
            "z_recover_min": cfg.z_recover_min,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nVERDICT: {verdict}  H*={H_star}  ({manifest['total_secs']}s)  -> {out_dir/'manifest.json'}")
    return manifest


def extract_body(model, H, cfg, device, seed, twin=False):
    rng = np.random.default_rng(seed + 123)
    bits, z = gen_batch(H, cfg.n_fingerprint, cfg.bits_per_channel, cfg.delta, rng)
    idx = torch.tensor(bits, device=device)
    model.eval()
    with torch.no_grad():
        _, hiddens = model(idx, return_hidden=True)
    bodies = np.stack([h[:, -1, :].cpu().numpy() for h in hiddens], axis=1)
    return bodies, z


def main():
    import pathlib
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["smoke", "full"], default="full")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = smoke_cfg() if args.mode == "smoke" else Cfg()
    cfg.seed = args.seed
    torch.set_num_threads(4)
    out = pathlib.Path(args.out) if args.out else pathlib.Path(
        f"results/chatv2/phase0-{args.mode}")
    run(cfg, out)


if __name__ == "__main__":
    main()
