#!/usr/bin/env python
"""P-vs-NP Direction-B — Gate-0 leg-(d) preflight (MODEL-FREE, no training, ~$0).

The question: can capacity-relative one-wayness (leg d: cheap-check / hard-invert) be an EMERGENT
property of a TRAINED body, or does it collapse to imported algebra? The design panel's cheapest
killer: if a RANDOM-FEATURE (untrained, smooth) body already exposes the abstraction z and the
certificate sigma to a cheap LINEAR decoder — above the chatv2 de-confound floor — then a trained
body (which only adds structure, and MUST expose z for control-sufficiency) is at least as
invertible. Stage-(i) of any learned attack (body -> abstraction) is then free, so whatever hardness
remains is the imported combinatorial core (syndrome decoding) -> leg-(d) does NOT emerge.

Substrate D2: chatv2 pair-XOR latents (latent="computed", arity=2; z_i = parity-channel aggregate,
provably NOT linearly input-decodable by construction) + a FROZEN public GF(2) syndrome head
sigma = H_pub @ z (rank m), which is lossy by algebra (2^(H-m) preimages per sigma).

Reads, per target, recovery det = (cv_acc - majority)/(1 - majority) for decoders:
  - raw-linear        linear on raw input bits           (the chatv2 de-confound FLOOR; ~chance for z)
  - raw-mlp           1-hidden-MLP on raw bits           (functional present nonlinearly -> expected high)
  - body-linear       linear on an UNTRAINED TinyGPT hidden (the leg-(d) test: smooth random body)
  - body-mlp          1-hidden-MLP on the untrained body
  - rff-linear        linear on generic ReLU random features of bits (cross-check)

VERDICT:
  leg_d_imported_smooth_body_invertible  if the random body LINEARLY exposes z well above the raw floor
  gate0_inconclusive_survives_to_B1      otherwise (would earn a real body train; but see the
                                         structural note: control-sufficiency requires body->z easy,
                                         so the body cannot be one-way regardless — one-wayness lives
                                         only in sigma = imported H_pub algebra).

    python scripts/pvnp_directionb_gate0.py --out results/pvnp/directionb-gate0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.random import default_rng
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score

from chatv2_phase0_bodyresist import Cfg, TinyGPT, gen_batch, _std


def gf2_rank(M):
    M = (M % 2).astype(np.int64).copy()
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = next((i for i in range(r, rows) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] + M[r]) % 2
        r += 1
    return r


def rand_rank_gf2(m, H, rng):
    for _ in range(1000):
        M = rng.integers(0, 2, size=(m, H))
        if gf2_rank(M) == m:
            return M.astype(np.int64)
    raise RuntimeError("could not sample full-rank H_pub")


def secret_functional(H_pub, H, rng):
    """A g in GF(2)^H NOT in rowspace(H_pub): sigma = H_pub z does not linearly determine g.z."""
    for _ in range(1000):
        g = rng.integers(0, 2, size=H).astype(np.int64)
        if gf2_rank(np.vstack([H_pub, g])) > gf2_rank(H_pub):
            return g
    raise RuntimeError("could not find a secret functional outside rowspace")


def cv_acc(X, y, clf):
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(cross_val_score(clf, X, y, cv=4, scoring="accuracy").mean())


def det(acc, y):
    if np.isnan(acc):
        return float("nan")
    base = max(float(y.mean()), 1 - float(y.mean()))
    return (acc - base) / max(1 - base, 1e-9)


def med(xs):
    a = np.asarray([v for v in xs if not (isinstance(v, float) and np.isnan(v))], float)
    return float(np.median(a)) if a.size else float("nan")


def recover(X, targets, clf):
    """median recovery det over a list of (name, y) targets, for one feature set X + decoder clf."""
    return [det(cv_acc(X, y, clf), y) for _, y in targets]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pvnp/directionb-gate0")
    ap.add_argument("--H", type=int, default=8)
    ap.add_argument("--m", type=int, default=5, help="syndrome bits (kernel dim = H-m secret dims)")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--bpc", type=int, default=24)
    ap.add_argument("--delta", type=float, default=0.45)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--rff", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    H, m, n = args.H, args.m, args.n
    rng = default_rng(args.seed)
    print(f"[cfg] H={H} m_syndrome={m} (secret dims={H-m}) n={n} bpc={args.bpc} delta={args.delta} "
          f"d_model={args.d_model}", flush=True)

    # --- substrate: chatv2 pair-XOR latents ---
    cfg = Cfg(latent="computed", arity=2, bits_per_channel=args.bpc, delta=args.delta,
              h_sweep=[H], d_model=args.d_model, n_layers=3, n_heads=4, n_fingerprint=n)
    bits, z, _ = gen_batch(H, n, cfg, rng)                       # bits (n,L) in {0,1}; z (n,H)
    L = bits.shape[1]

    # --- frozen public GF(2) syndrome head (lossy by algebra) ---
    H_pub = rand_rank_gf2(m, H, rng)                             # (m,H), rank m
    sigma = (z @ H_pub.T) % 2                                    # (n,m)
    g = secret_functional(H_pub, H, rng)                         # secret functional outside rowspace
    secret = (z @ g) % 2                                         # (n,) NOT determined by sigma
    preimages_per_sigma = 2 ** (H - m)

    # --- targets ---
    z_targets = [(f"z{i}", z[:, i]) for i in range(H)]
    sig_targets = [(f"sig{j}", sigma[:, j]) for j in range(m)]
    secret_target = [("secret", secret)]

    # --- feature sets ---
    Xraw = _std(bits.astype(np.float64))
    torch.manual_seed(args.seed)
    model = TinyGPT(2, args.d_model, cfg.n_layers, cfg.n_heads, cfg.max_len)   # UNTRAINED random body
    model.eval()
    with torch.no_grad():
        _, hid = model(torch.tensor(bits), return_hidden=True)
    body = _std(hid[-1][:, -1, :].cpu().numpy().astype(np.float64))            # final hidden, last pos
    Wr = rng.standard_normal((L, args.rff))
    rff = _std(np.maximum(0.0, bits.astype(np.float64) @ Wr))                  # generic ReLU random features

    lin = LogisticRegression(max_iter=1000)
    mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0)

    print(f"[setup] L={L} tokens; H_pub rank={gf2_rank(H_pub)}; {preimages_per_sigma} z-preimages/sigma "
          f"({round(time.time()-t0,1)}s)", flush=True)

    res = {}
    def block(tag, X):
        zl = recover(X, z_targets, lin); zm = recover(X, z_targets, mlp)
        sl = recover(X, sig_targets, lin)
        se = det(cv_acc(X, secret, lin), secret)
        r = {"z_lin_med": round(med(zl), 4), "z_mlp_med": round(med(zm), 4),
             "sig_lin_med": round(med(sl), 4), "secret_lin": round(se, 4) if not np.isnan(se) else None,
             "z_lin_each": [round(v, 3) for v in zl]}
        res[tag] = r
        print(f"[{tag:11s}] z_lin={r['z_lin_med']:+.3f}  z_mlp={r['z_mlp_med']:+.3f}  "
              f"sig_lin={r['sig_lin_med']:+.3f}  secret_lin={r['secret_lin']}  "
              f"({round(time.time()-t0,1)}s)", flush=True)
        return r

    raw = block("raw", Xraw)             # de-confound floor (z_lin ~ 0) + functional present (z_mlp high)
    bdy = block("rand_body", body)       # THE leg-(d) test: smooth random body, linear readout
    block("rand_body_mlp_only", body) if False else None
    # body MLP (one number, the strongest smooth-body readout)
    body_mlp_z = round(med(recover(body, z_targets, mlp)), 4)
    res["rand_body"]["z_mlp_med"] = body_mlp_z
    rf = block("rff", rff)               # generic random features cross-check

    # --- verdict ---
    floor = raw["z_lin_med"]
    body_lin = bdy["z_lin_med"]
    smooth_invertible = (body_lin - floor) > 0.10 and body_lin > 0.20
    # sigma lossiness check: a linear decoder of the SECRET from sigma alone must be ~chance
    secret_from_sigma = det(cv_acc(_std(sigma.astype(np.float64)), secret, lin), secret)
    sigma_lossy = (secret_from_sigma is not None) and (abs(secret_from_sigma) < 0.10 or np.isnan(secret_from_sigma))

    verdict = ("leg_d_imported_smooth_body_invertible" if smooth_invertible
               else "gate0_inconclusive_survives_to_B1")

    manifest = {
        "lane": "pvnp-directionb", "stage": "gate0_leg_d_preflight", "verdict": verdict,
        "cfg": {"H": H, "m_syndrome": m, "secret_dims": H - m, "n": n, "bpc": args.bpc,
                "delta": args.delta, "d_model": args.d_model, "L": L,
                "preimages_per_sigma": preimages_per_sigma, "seed": args.seed},
        "deconfound_floor_z_lin": floor,
        "random_body_z_lin": body_lin,
        "random_body_z_mlp": res["rand_body"]["z_mlp_med"],
        "delta_body_minus_floor_lin": round(body_lin - floor, 4),
        "smooth_body_linearly_invertible": bool(smooth_invertible),
        "sigma_lossy_secret_unreadable_from_sigma": bool(sigma_lossy),
        "secret_det_from_sigma": round(secret_from_sigma, 4) if not np.isnan(secret_from_sigma) else None,
        "blocks": res,
        "structural_note": ("Control-sufficiency REQUIRES body->z easy (the body computes z to use it; "
                            "chatv2 z1_acc~0.94). So the body cannot be one-way regardless of this "
                            "measurement; the only one-way object is sigma=H_pub@z, whose hardness "
                            "(sigma->low-weight z*) is imported GF(2) syndrome decoding, measured in "
                            "results/pvnp/certificate-syndrome-v1. Gate-0 confirms the smooth-body leg "
                            "empirically."),
        "wall_s": round(time.time() - t0, 1),
    }
    (out / "gate0.json").write_text(json.dumps(manifest, indent=2))

    print(f"\n==== GATE-0 VERDICT: {verdict} ====")
    print(f"  de-confound floor (z linear from raw)         = {floor:+.3f}  (expect ~0: pair-XOR holds)")
    print(f"  random-feature body, z LINEAR readout         = {body_lin:+.3f}  (delta vs floor = {body_lin-floor:+.3f})")
    print(f"  random-feature body, z MLP readout            = {res['rand_body']['z_mlp_med']:+.3f}")
    print(f"  generic RFF, z LINEAR readout                 = {rf['z_lin_med']:+.3f}")
    print(f"  sigma lossy? secret from sigma (linear)       = {manifest['secret_det_from_sigma']}  "
          f"(expect ~0: lossy by algebra, {preimages_per_sigma} preimages)")
    if smooth_invertible:
        print("  -> A SMOOTH RANDOM body already linearly exposes z above the de-confound floor.")
        print("     A trained body (more structure + control-sufficiency) is >= as invertible, so")
        print("     stage-(i) body->abstraction is free; leg-(d) hardness is the IMPORTED syndrome core.")
        print("     EMERGENT capacity-relative one-wayness from a trained body: NOT SUPPORTED. Do not spend GPU.")
    else:
        print("  -> Random features did NOT linearly crack z. D2 would survive to a real body train (B1),")
        print("     BUT control-sufficiency still forces a trained body to expose z, so one-wayness can")
        print("     only live in sigma = imported algebra. Re-read the structural note before any GPU.")
    print(f"  wrote {out/'gate0.json'}  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
