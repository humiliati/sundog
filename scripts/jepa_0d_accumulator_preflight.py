#!/usr/bin/env python
"""JEPA-0D accumulator substrate - MODEL-FREE de-confound preflight.

Gate 1 (de-confound) + Gate 2 (information-present oracle) of the JEPA-0D mandatory
gate order (docs/chatv2/JEPA_0D_HANDOFF.md). NO model training. numpy + sklearn only.

Substrate (generalises the Phase-7 coupled toy: a static m-bit source u -> a running
bounded count u_t, with event channels that feed the count):

    hidden event    e_t ~ Bernoulli(p_event),  t = 1..T
    hidden count    u_t = min(sum_{s<=t} e_s, K)         (BOUNDED sum, not modulo)
    event channel   parity-channel encoding of e_t       (CLEAN -> u_t is a clean functional,
                                                           and the oracle can sum events -> u_t)
    count-readout   at checkpoints c, n_U channels emit an OBSERVED NOISY parity-readout
       channels        z_{c,j} = parity(PSI_j, bits(u_c)) XOR x_{c,j},  x ~ Bernoulli(p_noise)
                       then z_{c,j} is itself parity-channel encoded (input-undecodable).
                       clean part depends on u_c  -> forces a tracker to keep the count;
                       private flip x             -> the eventual JEPA-vs-GEN discard target.

Every emitted token is marginally fair (the last tuple bit = parity(fair bits) XOR payload,
and XOR with a fair bit randomises the marginal), so NO single bit, sub-tuple, or bit-count
linearly reveals e_t or u_t. This preflight verifies that empirically rather than asserting it.

Pre-registered bars (handoff "Suggested preflight bars"):
  * raw linear u_det <= 0.10   (and within 0.05 of the per-position majority baseline)
  * oracle u_t recovery >= 0.95
  * no class / support starvation for the planned probe targets

Multiclass u_det (handoff): u_det = (heldout_acc - majority_base) / max(1 - majority_base, eps),
measured per read position against that position's OWN majority baseline (so a position-only
predictor that just guesses E[u_t | t] scores ~0 and gets no credit for the monotone-count prior),
then aggregated by median over positions.

    python scripts/jepa_0d_accumulator_preflight.py --out results/chatv2/jepa-0d-accumulator-preflight
"""
import argparse
import csv
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List

import numpy as np
from numpy.random import default_rng
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# Count-readout coupling graph. Reused verbatim from chatv2_phase0_bodyresist._COUPLE_A
# (8 readouts x 3 count-bits, GF(2)-rank 3 so the clean readouts collectively determine u_t).
PSI = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1],
                [1, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1]], dtype=np.int64)

EPS = 1e-9


@dataclass
class AccCfg:
    K: int = 6                       # count cap (bounded, not modulo); 3 count bits cover 0..7
    T: int = 12                      # ticks per sequence
    p_event: float = 0.34            # P(e_t = 1)
    arity: int = 2                   # parity-channel tuple arity (2 = pair-XOR baseline)
    delta: float = 0.45              # parity-channel payload softness (carried from Phase-7b)
    p_noise: float = 0.10            # private flip rate on count-readouts (clean-u regime)
    P_e: int = 5                     # event tuples per tick (odd -> clean majority vote)
    P_u: int = 3                     # readout tuples per channel
    n_U: int = 8                     # count-readout channels per checkpoint (<= len(PSI))
    emit_checkpoints: List[int] = field(default_factory=lambda: [4, 8, 12])  # ticks that emit readouts
    n: int = 3000                    # sample size (matches Phase-7b n_fingerprint)
    seed: int = 0
    # pre-registered bars
    deconfound_max: float = 0.10
    oracle_min: float = 0.95
    min_class_support: int = 30
    min_flip_support: int = 50

    @property
    def count_bits(self) -> int:
        return max(1, int(np.ceil(np.log2(self.K + 1))))

    @property
    def Le(self) -> int:               # event-channel tokens per tick
        return self.P_e * self.arity

    @property
    def Lu(self) -> int:               # readout-channel tokens (per channel)
        return self.P_u * self.arity


# --------------------------------------------------------------------------- #
# parity-channel codec (matches chatv2_phase0_bodyresist._gen_computed exactly)
# --------------------------------------------------------------------------- #
def parity_encode(payload, P, A, delta, rng):
    """payload (n,) in {0,1} -> bits (n, P*A). Each of P tuples has A-parity = a
    payload-biased Bernoulli(0.5 +/- delta) draw; first A-1 bits fair, last = parity XOR draw."""
    n = payload.shape[0]
    px = 0.5 + delta * (2.0 * payload - 1.0)                      # (n,) P(tuple parity = 1)
    x = (rng.random((n, P)) < px[:, None]).astype(np.int64)      # (n,P) per-tuple parity = payload-biased
    tup = rng.integers(0, 2, size=(n, P, A)).astype(np.int64)
    tup[:, :, A - 1] = (tup[:, :, :A - 1].sum(2) % 2) ^ x
    return tup.reshape(n, P * A)


def parity_decode(bits, P, A):
    """Inverse estimator: per-tuple A-parity, mean over P tuples, threshold at 0.5 -> payload hat."""
    n = bits.shape[0]
    tup = bits.reshape(n, P, A)
    par = tup.sum(2) % 2                                          # (n,P) estimate of each tuple payload
    return (par.mean(1) > 0.5).astype(np.int64)


def int_to_bits(vals, nbits):
    """vals (...,) -> (..., nbits) big-endian GF(2) bits."""
    out = ((vals[..., None] >> np.arange(nbits - 1, -1, -1)) & 1).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def gen_accumulator(cfg: AccCfg, rng):
    """Returns a dict with raw tokens, hidden state, observed readouts, flips, and a token layout."""
    n, T, A = cfg.n, cfg.T, cfg.arity
    assert cfg.n_U <= len(PSI), "n_U exceeds the PSI coupling graph"
    assert max(cfg.emit_checkpoints) <= T and min(cfg.emit_checkpoints) >= 1
    Psi = PSI[:cfg.n_U]                                            # (n_U, 3) but we use count_bits cols
    b = cfg.count_bits
    Psi = Psi[:, -b:] if b <= PSI.shape[1] else np.pad(Psi, ((0, 0), (PSI.shape[1] - b, 0)))

    e = (rng.random((n, T)) < cfg.p_event).astype(np.int64)       # (n,T) hidden events
    u = np.minimum(np.cumsum(e, axis=1), cfg.K)                   # (n,T) bounded count u_t (t=1..T)

    ckpts = list(cfg.emit_checkpoints)
    ubits = int_to_bits(u, b)                                     # (n,T,b)

    cols, layout = [], {"ticks": [], "checkpoints": {}}
    cursor = 0
    # observed readouts + flips, indexed by checkpoint then channel
    z_obs = np.zeros((n, len(ckpts), cfg.n_U), dtype=np.int64)
    x_flip = np.zeros((n, len(ckpts), cfg.n_U), dtype=np.int64)
    clean = np.zeros((n, len(ckpts), cfg.n_U), dtype=np.int64)

    for t in range(1, T + 1):
        # --- event channel (clean encoding of e_t) ---
        blk = parity_encode(e[:, t - 1], cfg.P_e, A, cfg.delta, rng)
        cols.append(blk)
        ev_slice = [cursor, cursor + blk.shape[1]]
        cursor += blk.shape[1]
        ck_slices = {}
        # --- count-readout channels at emit checkpoints ---
        if t in ckpts:
            ci = ckpts.index(t)
            cbits = ubits[:, t - 1, :]                            # (n,b) bits of u_t
            for j in range(cfg.n_U):
                c = (cbits @ Psi[j]) % 2                          # (n,) clean parity readout
                flip = (rng.random(n) < cfg.p_noise).astype(np.int64)
                z = (c ^ flip).astype(np.int64)                  # observed noisy bit
                clean[:, ci, j], x_flip[:, ci, j], z_obs[:, ci, j] = c, flip, z
                blk = parity_encode(z, cfg.P_u, A, cfg.delta, rng)
                cols.append(blk)
                ck_slices[j] = [cursor, cursor + blk.shape[1]]
                cursor += blk.shape[1]
            layout["checkpoints"][t] = ck_slices
        layout["ticks"].append({"tick": t, "event_slice": ev_slice})

    X = np.concatenate(cols, axis=1).astype(np.int8)             # (n, L) raw tokens
    return {"X": X, "e": e, "u": u, "ubits": ubits, "z_obs": z_obs, "x_flip": x_flip,
            "clean": clean, "ckpts": ckpts, "layout": layout, "L": X.shape[1], "Psi": Psi}


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _cv_acc(X, y, cv=4, multiclass=False):
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan"), 1.0
    maj = float(counts.max() / counts.sum())
    folds = int(min(cv, counts.min()))
    if folds < 2:
        return float("nan"), maj
    clf = LogisticRegression(max_iter=2000, C=1.0)
    acc = float(cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean())
    return acc, maj


def _det(acc, maj):
    if np.isnan(acc):
        return float("nan")
    return (acc - maj) / max(1.0 - maj, EPS)


def safe_median(xs):
    arr = np.asarray([v for v in xs if not (isinstance(v, float) and np.isnan(v))], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def raw_linear_probes(data, cfg: AccCfg):
    """Gate 1: raw input must NOT linearly read u_t, e_t, or the count-dependent emission."""
    X = data["X"].astype(np.float64)
    u, e = data["u"], data["e"]
    out = {"u_per_tick": [], "e_per_tick": [], "emission": {}}

    # u_t at every tick (per-position majority baseline removes the monotone-count prior)
    for t in range(cfg.T):
        acc, maj = _cv_acc(X, u[:, t], multiclass=True)
        out["u_per_tick"].append({"tick": t + 1, "acc": round(acc, 4) if not np.isnan(acc) else None,
                                  "maj": round(maj, 4), "u_det": round(_det(acc, maj), 4) if not np.isnan(acc) else None})
    # e_t at every tick
    for t in range(cfg.T):
        acc, maj = _cv_acc(X, e[:, t])
        out["e_per_tick"].append({"tick": t + 1, "acc": round(acc, 4) if not np.isnan(acc) else None,
                                  "maj": round(maj, 4), "e_det": round(_det(acc, maj), 4) if not np.isnan(acc) else None})
    # count-dependent emission: clean readout bit c_{c,j} AND observed z_{c,j}
    clean, z = data["clean"], data["z_obs"]
    cdet, zdet = [], []
    for ci, c in enumerate(data["ckpts"]):
        for j in range(cfg.n_U):
            a1, m1 = _cv_acc(X, clean[:, ci, j]); cdet.append(_det(a1, m1))
            a2, m2 = _cv_acc(X, z[:, ci, j]);     zdet.append(_det(a2, m2))
    out["emission"]["clean_det_median"] = round(safe_median(cdet), 4)
    out["emission"]["obs_det_median"] = round(safe_median(zdet), 4)

    # paranoid count-leak probe: per-tick event-block bit-SUMS -> u_T (linear count feature)
    ev_slices = [tk["event_slice"] for tk in data["layout"]["ticks"]]
    sums = np.stack([X[:, s:e2].sum(1) for s, e2 in ev_slices], axis=1)   # (n,T) block bit-counts
    accs, majs = _cv_acc(sums, u[:, -1], multiclass=True)
    out["bitcount_uT_det"] = round(_det(accs, majs), 4) if not np.isnan(accs) else None

    udet = [r["u_det"] for r in out["u_per_tick"] if r["u_det"] is not None]
    edet = [r["e_det"] for r in out["e_per_tick"] if r["e_det"] is not None]
    out["u_det_median"] = round(safe_median(udet), 4)
    out["u_det_max"] = round(float(np.max(udet)), 4) if udet else None
    out["e_det_median"] = round(safe_median(edet), 4)
    out["e_det_max"] = round(float(np.max(edet)), 4) if edet else None
    return out


def oracle_recover(data, cfg: AccCfg):
    """Gate 2: a deterministic parser that decodes parity ticks and sums them must recover u_t."""
    X = data["X"]
    u = data["u"]
    # event route: decode each tick's event channel -> e_hat -> bounded cumsum -> u_hat
    ev_slices = [tk["event_slice"] for tk in data["layout"]["ticks"]]
    e_hat = np.stack([parity_decode(X[:, s:e2].astype(np.int64), cfg.P_e, cfg.arity)
                      for s, e2 in ev_slices], axis=1)                     # (n,T)
    u_hat = np.minimum(np.cumsum(e_hat, axis=1), cfg.K)
    e_acc = float((e_hat == data["e"]).mean())
    u_acc_per_tick = [float((u_hat[:, t] == u[:, t]).mean()) for t in range(cfg.T)]
    u_acc_overall = float((u_hat == u).mean())
    u_exact_seq = float((u_hat == u).all(axis=1).mean())

    # readout route (corroboration): decode z_hat per readout, majority-vote bits via Psi, recover u_c
    Psi = data["Psi"]
    b = cfg.count_bits
    all_codes = int_to_bits(np.arange(cfg.K + 1), b)                       # (K+1, b)
    code_clean = (all_codes @ Psi.T) % 2                                  # (K+1, n_U) clean readout per value
    ro_acc = {}
    for ci, c in enumerate(data["ckpts"]):
        slices = data["layout"]["checkpoints"][c]
        z_hat = np.stack([parity_decode(X[:, slices[j][0]:slices[j][1]].astype(np.int64),
                                        cfg.P_u, cfg.arity) for j in range(cfg.n_U)], axis=1)  # (n,n_U)
        # nearest codeword by Hamming distance to the clean readout table
        dist = (z_hat[:, None, :] != code_clean[None, :, :]).sum(2)        # (n, K+1)
        u_ro = dist.argmin(1)
        ro_acc[c] = round(float((u_ro == u[:, c - 1]).mean()), 4)
    return {"event_route_e_acc": round(e_acc, 4),
            "event_route_u_acc_overall": round(u_acc_overall, 4),
            "event_route_u_exact_seq": round(u_exact_seq, 4),
            "event_route_u_acc_per_tick": [round(v, 4) for v in u_acc_per_tick],
            "readout_route_u_acc_per_ckpt": ro_acc}


def base_rates(data, cfg: AccCfg):
    u, e = data["u"], data["e"]
    per_tick = []
    starved_ticks = []
    for t in range(cfg.T):
        vals, cnts = np.unique(u[:, t], return_counts=True)
        hist = {int(v): int(c) for v, c in zip(vals, cnts)}
        usable = sum(1 for c in cnts if c >= cfg.min_class_support)
        per_tick.append({"tick": t + 1, "hist": hist, "n_classes": int(len(vals)),
                         "n_usable_classes": int(usable), "maj": round(float(cnts.max() / cnts.sum()), 4)})
        if usable < 2:
            starved_ticks.append(t + 1)
    vals, cnts = np.unique(u, return_counts=True)
    pooled = {int(v): int(c) for v, c in zip(vals, cnts)}
    pooled_starved = [int(v) for v in range(cfg.K + 1) if pooled.get(v, 0) < cfg.min_class_support]
    # flip support per (checkpoint, channel) for the later gate-5 discard read
    flip_counts = data["x_flip"].sum(axis=0)                               # (n_ckpt, n_U)
    flip_starved = int((flip_counts < cfg.min_flip_support).sum())
    return {"u_per_tick": per_tick, "u_pooled": pooled,
            "u_pooled_classes_starved": pooled_starved,
            "u_ticks_starved": starved_ticks,
            "e_base_rate": round(float(e.mean()), 4),
            "flip_counts": flip_counts.astype(int).tolist(),
            "flip_min": int(flip_counts.min()), "flip_starved_cells": flip_starved}


# --------------------------------------------------------------------------- #
def json_clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {(int(k) if isinstance(k, (np.integer,)) else k): json_clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_clean(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chatv2/jepa-0d-accumulator-preflight")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--p-event", type=float, default=None)
    ap.add_argument("--p-noise", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--checkpoints", default=None, help="comma-sep ticks, e.g. 4,8,12")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = AccCfg()
    if args.K is not None: cfg.K = args.K
    if args.T is not None: cfg.T = args.T
    if args.p_event is not None: cfg.p_event = args.p_event
    if args.p_noise is not None: cfg.p_noise = args.p_noise
    if args.n is not None: cfg.n = args.n
    if args.checkpoints: cfg.emit_checkpoints = [int(x) for x in args.checkpoints.split(",")]
    cfg.seed = args.seed

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = default_rng(cfg.seed)
    print(f"[cfg] K={cfg.K} T={cfg.T} p_event={cfg.p_event} p_noise={cfg.p_noise} "
          f"n={cfg.n} ckpts={cfg.emit_checkpoints} count_bits={cfg.count_bits}", flush=True)

    data = gen_accumulator(cfg, rng)
    print(f"[gen] L={data['L']} tokens/seq  ({round(time.time()-t0,1)}s)", flush=True)

    raw = raw_linear_probes(data, cfg)
    print(f"[deconfound] raw u_det median={raw['u_det_median']} max={raw['u_det_max']}  "
          f"e_det median={raw['e_det_median']} max={raw['e_det_max']}  "
          f"emission clean/obs={raw['emission']['clean_det_median']}/{raw['emission']['obs_det_median']}  "
          f"bitcount->uT={raw['bitcount_uT_det']}  ({round(time.time()-t0,1)}s)", flush=True)

    orc = oracle_recover(data, cfg)
    print(f"[oracle] event-route u_acc={orc['event_route_u_acc_overall']} "
          f"e_acc={orc['event_route_e_acc']} exact-seq={orc['event_route_u_exact_seq']}  "
          f"readout-route per-ckpt={orc['readout_route_u_acc_per_ckpt']}  ({round(time.time()-t0,1)}s)", flush=True)

    br = base_rates(data, cfg)
    print(f"[support] e_base={br['e_base_rate']}  u_pooled_starved={br['u_pooled_classes_starved']}  "
          f"ticks_starved={br['u_ticks_starved']}  flip_min={br['flip_min']} "
          f"flip_starved_cells={br['flip_starved_cells']}", flush=True)

    # ---- gate evaluation (pre-registered bars) ----
    g1 = (raw["u_det_max"] is not None and raw["u_det_max"] <= cfg.deconfound_max
          and raw["e_det_max"] is not None and raw["e_det_max"] <= cfg.deconfound_max
          and raw["emission"]["clean_det_median"] <= cfg.deconfound_max
          and raw["emission"]["obs_det_median"] <= cfg.deconfound_max
          and (raw["bitcount_uT_det"] is None or raw["bitcount_uT_det"] <= cfg.deconfound_max))
    g2 = orc["event_route_u_acc_overall"] >= cfg.oracle_min
    g3 = (len(br["u_ticks_starved"]) == 0 and br["flip_starved_cells"] == 0)
    if not g1:
        verdict = "blocked_by_deconfound_leak"
    elif not g2:
        verdict = "blocked_by_absent_functional"
    elif not g3:
        verdict = "support_starved"
    else:
        verdict = "preflight_pass_ready_for_spec"

    manifest = {"lane": "jepa-0d-accumulator", "stage": "model_free_preflight",
                "verdict": verdict, "gates": {"deconfound": g1, "oracle": g2, "support": g3},
                "bars": {"deconfound_max": cfg.deconfound_max, "oracle_min": cfg.oracle_min,
                         "min_class_support": cfg.min_class_support, "min_flip_support": cfg.min_flip_support},
                "cfg": asdict(cfg), "L": data["L"],
                "raw_linear": raw, "oracle": orc, "base_rates": br,
                "wall_s": round(time.time() - t0, 1)}
    (out / "preflight.json").write_text(json.dumps(json_clean(manifest), indent=2))

    # per-tick CSV
    with (out / "u_per_tick.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "raw_u_acc", "raw_u_maj", "raw_u_det", "oracle_u_acc",
                    "n_classes", "n_usable_classes"])
        for t in range(cfg.T):
            rr = raw["u_per_tick"][t]; bb = br["u_per_tick"][t]
            w.writerow([t + 1, rr["acc"], rr["maj"], rr["u_det"],
                        orc["event_route_u_acc_per_tick"][t], bb["n_classes"], bb["n_usable_classes"]])

    print(f"\n==== PREFLIGHT VERDICT: {verdict} ====")
    print(f"  gate1 de-confound (raw u_det<= {cfg.deconfound_max}): {'PASS' if g1 else 'FAIL'}")
    print(f"  gate2 oracle (u recovery>= {cfg.oracle_min}):        {'PASS' if g2 else 'FAIL'}")
    print(f"  gate3 support (no starvation):              {'PASS' if g3 else 'FAIL'}")
    print(f"  wrote {out/'preflight.json'} and u_per_tick.csv  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
