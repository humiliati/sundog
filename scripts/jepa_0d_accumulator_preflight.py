#!/usr/bin/env python
"""JEPA-0D accumulator substrate - MODEL-FREE preflight (gates 1, 2, support, mask-necessity).

Gate 1 (de-confound) + Gate 2 (information-present oracle) + support + a model-free
mask-necessity gate, from the JEPA-0D mandatory gate order (docs/chatv2/JEPA_0D_HANDOFF.md).
NO model training. numpy + sklearn only.

Substrate (generalises the Phase-7 coupled toy: a static m-bit source u -> a running
bounded count u_t, with event channels that feed the count):

    hidden event    e_t ~ Bernoulli(p_event),  t = 1..T
    hidden count    u_t = min(sum_{s<=t} e_s, K)         (BOUNDED sum, not modulo)
    event channel   parity-channel encoding of e_t       (CLEAN -> u_t is a clean functional,
                                                           and the oracle sums events -> u_t)
    count-readout   at checkpoints c, n_U channels emit an OBSERVED NOISY parity-readout
       channels        z_{c,j} = parity(PSI_j, bits(u_c)) XOR x_{c,j},  x ~ Bernoulli(p_noise)
                       then z_{c,j} is itself parity-channel encoded (input-undecodable).
                       clean part depends on u_c  -> forces a tracker to keep the count;
                       private flip x             -> the eventual JEPA-vs-GEN discard target.

Every emitted token is marginally fair (last tuple bit = parity(fair bits) XOR payload, and
XOR with a fair bit randomises the marginal), so NO single bit, sub-tuple, or bit-count linearly
reveals e_t or u_t. This preflight verifies that empirically rather than asserting it.

MASK-NECESSITY (added after adversarial review). All count-readouts at a checkpoint are functions
of the SAME u_c, so a masked readout is recoverable from its same-checkpoint peers by pure XOR
(det~1.0, ZERO count integration) -- the exact parity Phase-0 failure if a generic random latent
mask is used. This preflight measures that shortcut AND verifies that WHOLE-CHECKPOINT masking
(mask all of a checkpoint's readouts as a unit) removes it, leaving the event->u_t path as the
only route. The chosen mask policy is therefore whole-checkpoint, not the handoff's carried-over
random 50% latent mask; the spec must adopt this (with operator sign-off).

Pre-registered bars (handoff "Suggested preflight bars" + the mask-necessity addendum):
  * raw linear u_det <= 0.10   (and within 0.05 of the per-position majority baseline)
  * oracle u_t recovery >= 0.95
  * no class / support starvation for the planned probe targets
  * mask-necessity: whole-checkpoint shortcut det <= 0.15 AND intended event->u_t path det >= 0.70

u_det metric (PINNED, one definition for ALL gates incl. the later gate-4 u_det>=0.70):
  u_det = (heldout_acc - majority_base) / max(1 - majority_base, eps), accuracy-based multiclass,
  measured per read position against that position's OWN majority baseline (so a position-only
  predictor that guesses E[u_t|t] scores ~0), aggregated by median over positions. An ordinal
  MAE-reduction det is reported as a REPORT-ONLY sidecar (within-1 accuracy-det is rejected: its
  baseline -> 1 at early checkpoints causes a div-by-~0 blowup). Mid-lane metric substitution is
  forbidden; if the gate-4 bar is ever restated under MAE-det it must be re-derived and re-pinned.

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
from sklearn.model_selection import cross_val_score, train_test_split

EPS = 1e-9


def make_psi(count_bits, n_U):
    """n_U distinct nonzero GF(2) functionals of `count_bits` count-bits, basis vectors first so
    any prefix of length >= count_bits spans (rank = count_bits). Replaces the reused _COUPLE_A,
    which had a literal duplicate row (0==7) and only 3 usable bits. n_U <= 2**count_bits - 1."""
    b = count_bits
    assert n_U <= (2 ** b) - 1, f"n_U={n_U} exceeds {2**b - 1} distinct nonzero {b}-bit functionals"
    basis = [1 << k for k in range(b)]                                # powers of two (big-endian below)
    rest = [v for v in range(1, 2 ** b) if v not in basis]
    order = basis + rest
    vals = order[:n_U]
    rows = [[(v >> (b - 1 - k)) & 1 for k in range(b)] for v in vals]  # big-endian bit rows
    Psi = np.array(rows, dtype=np.int64)
    # GF(2) rank check
    M = Psi.copy() % 2
    rank, r = 0, 0
    cols = M.shape[1]
    for c in range(cols):
        piv = None
        for i in range(r, M.shape[0]):
            if M[i, c]:
                piv = i; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] = (M[i] + M[r]) % 2
        r += 1; rank += 1
    assert rank == b, f"PSI rank {rank} != count_bits {b}; readouts do not determine u_t"
    return Psi


@dataclass
class AccCfg:
    K: int = 6                       # count cap (bounded, not modulo)
    T: int = 12                      # ticks per sequence
    p_event: float = 0.34            # P(e_t = 1)
    arity: int = 2                   # parity-channel tuple arity (2 = pair-XOR baseline)
    delta: float = 0.45              # parity-channel payload softness (carried from Phase-7b)
    p_noise: float = 0.10            # private flip rate on count-readouts (clean-u regime)
    P_e: int = 5                     # event tuples per tick (odd -> clean majority vote)
    P_u: int = 3                     # readout tuples per channel
    n_U: int = 7                     # count-readout channels per checkpoint (7 distinct 3-bit functionals)
    emit_checkpoints: List[int] = field(default_factory=lambda: [4, 8, 12])
    n: int = 3000                    # sample size (matches Phase-7b n_fingerprint)
    seed: int = 0
    # pre-registered bars
    deconfound_max: float = 0.10
    oracle_min: float = 0.95
    min_class_support: int = 30
    min_flip_support: int = 50
    mask_shortcut_max: float = 0.15  # whole-checkpoint shortcut det must be <= this
    intended_path_min: float = 0.70  # event->u_t path det must be >= this

    @property
    def count_bits(self) -> int:
        return max(1, int(np.ceil(np.log2(self.K + 1))))


# --------------------------------------------------------------------------- #
# parity-channel codec (matches chatv2_phase0_bodyresist._gen_computed exactly)
# --------------------------------------------------------------------------- #
def parity_encode(payload, P, A, delta, rng):
    """payload (n,) in {0,1} -> bits (n, P*A). Each of P tuples has A-parity = a payload-biased
    Bernoulli(0.5 +/- delta) draw; first A-1 bits fair, last = parity(fair) XOR draw."""
    n = payload.shape[0]
    px = 0.5 + delta * (2.0 * payload - 1.0)
    x = (rng.random((n, P)) < px[:, None]).astype(np.int64)
    tup = rng.integers(0, 2, size=(n, P, A)).astype(np.int64)
    tup[:, :, A - 1] = (tup[:, :, :A - 1].sum(2) % 2) ^ x
    return tup.reshape(n, P * A)


def parity_decode(bits, P, A):
    """Inverse estimator: per-tuple A-parity, mean over P tuples, threshold 0.5 -> payload hat."""
    n = bits.shape[0]
    par = bits.reshape(n, P, A).sum(2) % 2
    return (par.mean(1) > 0.5).astype(np.int64)


def int_to_bits(vals, nbits):
    return ((vals[..., None] >> np.arange(nbits - 1, -1, -1)) & 1).astype(np.int64)


# --------------------------------------------------------------------------- #
# generator
# --------------------------------------------------------------------------- #
def gen_accumulator(cfg: AccCfg, rng):
    n, T, A = cfg.n, cfg.T, cfg.arity
    assert max(cfg.emit_checkpoints) <= T and min(cfg.emit_checkpoints) >= 1
    b = cfg.count_bits
    Psi = make_psi(b, cfg.n_U)                                        # (n_U, b) full-width, rank b

    e = (rng.random((n, T)) < cfg.p_event).astype(np.int64)
    u = np.minimum(np.cumsum(e, axis=1), cfg.K)                       # (n,T) bounded count u_t
    ckpts = list(cfg.emit_checkpoints)
    ubits = int_to_bits(u, b)                                         # (n,T,b)

    cols, layout = [], {"ticks": [], "checkpoints": {}}
    cursor = 0
    z_obs = np.zeros((n, len(ckpts), cfg.n_U), dtype=np.int64)
    x_flip = np.zeros((n, len(ckpts), cfg.n_U), dtype=np.int64)
    clean = np.zeros((n, len(ckpts), cfg.n_U), dtype=np.int64)

    for t in range(1, T + 1):
        blk = parity_encode(e[:, t - 1], cfg.P_e, A, cfg.delta, rng)
        cols.append(blk)
        ev_slice = [cursor, cursor + blk.shape[1]]; cursor += blk.shape[1]
        if t in ckpts:
            ci = ckpts.index(t)
            cbits = ubits[:, t - 1, :]                                # (n,b)
            ck_slices = {}
            for j in range(cfg.n_U):
                c = (cbits @ Psi[j]) % 2
                flip = (rng.random(n) < cfg.p_noise).astype(np.int64)
                z = (c ^ flip).astype(np.int64)
                clean[:, ci, j], x_flip[:, ci, j], z_obs[:, ci, j] = c, flip, z
                blk = parity_encode(z, cfg.P_u, A, cfg.delta, rng)
                cols.append(blk)
                ck_slices[j] = [cursor, cursor + blk.shape[1]]; cursor += blk.shape[1]
            layout["checkpoints"][t] = ck_slices
        layout["ticks"].append({"tick": t, "event_slice": ev_slice})

    X = np.concatenate(cols, axis=1).astype(np.int8)
    return {"X": X, "e": e, "u": u, "ubits": ubits, "z_obs": z_obs, "x_flip": x_flip,
            "clean": clean, "ckpts": ckpts, "layout": layout, "L": X.shape[1], "Psi": Psi}


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def _cv_acc(X, y, cv=4):
    """(acc, majority, method). Stratified-holdout fallback when the rarest class is too small for
    k>=2 folds, so a sparse class is never SILENTLY dropped to nan (adversarial-review fix)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan"), 1.0, "single_class"
    maj = float(counts.max() / counts.sum())
    folds = int(min(cv, counts.min()))
    clf = LogisticRegression(max_iter=2000)
    if folds >= 2:
        acc = float(cross_val_score(clf, X, y, cv=folds, scoring="accuracy").mean())
        return acc, maj, f"cv{folds}"
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
        clf.fit(Xtr, ytr)
        return float((clf.predict(Xte) == yte).mean()), maj, "holdout"
    except Exception:
        return float("nan"), maj, "failed"


def _det(acc, maj):
    return float("nan") if np.isnan(acc) else (acc - maj) / max(1.0 - maj, EPS)


def _mae_det(X, y, cv=4):
    """Report-only ordinal sidecar: MAE-reduction det = 1 - MAE(probe)/MAE(median baseline)."""
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) < 2:
        return float("nan")
    base_pred = int(np.median(y))
    base_mae = float(np.abs(y - base_pred).mean())
    if base_mae < EPS:
        return float("nan")
    folds = int(min(cv, counts.min()))
    if folds < 2:
        return float("nan")
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    maes = []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
        maes.append(np.abs(clf.predict(X[te]) - y[te]).mean())
    return float(1.0 - (np.mean(maes) / base_mae))


def safe_median(xs):
    arr = np.asarray([v for v in xs if not (isinstance(v, float) and np.isnan(v))], dtype=float)
    return float(np.median(arr)) if arr.size else float("nan")


def raw_linear_probes(data, cfg: AccCfg):
    """Gate 1: raw input must NOT linearly read u_t, e_t, or the count-dependent emission."""
    X = data["X"].astype(np.float64)
    u, e = data["u"], data["e"]
    out = {"u_per_tick": [], "e_per_tick": [], "emission": {}, "dropped": []}

    for t in range(cfg.T):
        acc, maj, meth = _cv_acc(X, u[:, t])
        mae = _mae_det(X, u[:, t])
        rec = {"tick": t + 1, "acc": None if np.isnan(acc) else round(acc, 4),
               "maj": round(maj, 4), "u_det": None if np.isnan(acc) else round(_det(acc, maj), 4),
               "mae_det": None if np.isnan(mae) else round(mae, 4), "method": meth}
        out["u_per_tick"].append(rec)
        if rec["u_det"] is None:
            out["dropped"].append(f"u_t tick{t+1} ({meth})")
    for t in range(cfg.T):
        acc, maj, meth = _cv_acc(X, e[:, t])
        rec = {"tick": t + 1, "acc": None if np.isnan(acc) else round(acc, 4), "maj": round(maj, 4),
               "e_det": None if np.isnan(acc) else round(_det(acc, maj), 4), "method": meth}
        out["e_per_tick"].append(rec)
        if rec["e_det"] is None:
            out["dropped"].append(f"e_t tick{t+1} ({meth})")

    clean, z = data["clean"], data["z_obs"]
    cdet, zdet = [], []
    for ci, c in enumerate(data["ckpts"]):
        for j in range(cfg.n_U):
            a1, m1, me1 = _cv_acc(X, clean[:, ci, j])
            a2, m2, me2 = _cv_acc(X, z[:, ci, j])
            d1, d2 = _det(a1, m1), _det(a2, m2)
            cdet.append(d1); zdet.append(d2)
            if np.isnan(d1): out["dropped"].append(f"clean ckpt{c} ch{j} ({me1})")
            if np.isnan(d2): out["dropped"].append(f"obs ckpt{c} ch{j} ({me2})")
    out["emission"]["clean_det_median"] = round(safe_median(cdet), 4)
    out["emission"]["obs_det_median"] = round(safe_median(zdet), 4)

    # paranoid count-leak probe: per-tick event-block bit-SUMS -> u_T (a linear count feature)
    ev_slices = [tk["event_slice"] for tk in data["layout"]["ticks"]]
    sums = np.stack([X[:, s:e2].sum(1) for s, e2 in ev_slices], axis=1)
    a, m, _ = _cv_acc(sums, u[:, -1])
    out["bitcount_uT_det"] = None if np.isnan(a) else round(_det(a, m), 4)

    udet = [r["u_det"] for r in out["u_per_tick"] if r["u_det"] is not None]
    edet = [r["e_det"] for r in out["e_per_tick"] if r["e_det"] is not None]
    out["u_det_median"] = round(safe_median(udet), 4)
    out["u_det_max"] = round(float(np.max(udet)), 4) if udet else None
    out["u_mae_det_median"] = round(safe_median([r["mae_det"] for r in out["u_per_tick"]
                                                 if r["mae_det"] is not None]), 4)
    out["e_det_median"] = round(safe_median(edet), 4)
    out["e_det_max"] = round(float(np.max(edet)), 4) if edet else None
    out["eval_incomplete"] = len(out["dropped"]) > 0
    return out


def oracle_recover(data, cfg: AccCfg):
    """Gate 2: a deterministic parser that decodes parity ticks and bounded-sums them recovers u_t.
    NOTE: this is a STRUCTURE-AWARE parser (parity-decode + bounded cumsum), NOT a generic learner;
    a flat MLP on raw tokens scores u_det~0 (recovery requires sequential parity + integration)."""
    X, u = data["X"], data["u"]
    ev_slices = [tk["event_slice"] for tk in data["layout"]["ticks"]]
    e_hat = np.stack([parity_decode(X[:, s:e2].astype(np.int64), cfg.P_e, cfg.arity)
                      for s, e2 in ev_slices], axis=1)
    u_hat = np.minimum(np.cumsum(e_hat, axis=1), cfg.K)
    e_acc = float((e_hat == data["e"]).mean())
    u_acc_per_tick = [float((u_hat[:, t] == u[:, t]).mean()) for t in range(cfg.T)]

    Psi, b = data["Psi"], cfg.count_bits
    all_codes = int_to_bits(np.arange(cfg.K + 1), b)
    code_clean = (all_codes @ Psi.T) % 2
    ro_acc = {}
    for ci, c in enumerate(data["ckpts"]):
        slices = data["layout"]["checkpoints"][c]
        z_hat = np.stack([parity_decode(X[:, slices[j][0]:slices[j][1]].astype(np.int64),
                                        cfg.P_u, cfg.arity) for j in range(cfg.n_U)], axis=1)
        dist = (z_hat[:, None, :] != code_clean[None, :, :]).sum(2)
        ro_acc[c] = round(float((dist.argmin(1) == u[:, c - 1]).mean()), 4)
    return {"event_route_e_acc": round(e_acc, 4),
            "event_route_u_acc_overall": round(float((u_hat == u).mean()), 4),
            "event_route_u_exact_seq": round(float((u_hat == u).all(axis=1).mean()), 4),
            "event_route_u_acc_per_tick": [round(v, 4) for v in u_acc_per_tick],
            "readout_route_u_acc_per_ckpt": ro_acc,
            "u_hat": u_hat}


def mask_necessity(data, cfg: AccCfg, u_hat):
    """Model-free mask-necessity gate (added after adversarial review).

    The shortcut A5 found lives at the DECODED level: a JEPA encoder computes each visible readout's
    parity, then same-checkpoint readouts (all functions of the SAME u_c) XOR-predict a masked one
    with NO count integration. A linear probe on raw parity-encoded tokens cannot see this (that is
    just the de-confound), so the shortcut features must be the parity-DECODED peer bits -- the
    representation the encoder forms. For each checkpoint c and readout target, predict the masked
    clean/observed readout from:
      * decoded same-ckpt peers (the shortcut available under a RANDOM channel mask -- a masked
        readout keeps >=1 same-ckpt peer ~99% of the time -- expected HIGH = footgun confirmed);
      * decoded OTHER-checkpoint readouts (whole-checkpoint mask: NO same-ckpt peers; any residual is
        count-based cross-ckpt inference, not the local XOR shortcut -- expected low);
      * one-hot(u_hat) from the EVENT route (the intended event->u_t path -- expected HIGH).
    Whole-checkpoint masking PASSES iff the residual shortcut is floored AND the intended path lives."""
    def decode_block(c, j):
        s, e = data["layout"]["checkpoints"][c][j]
        return parity_decode(data["X"][:, s:e].astype(np.int64), cfg.P_u, cfg.arity)

    clean, z, ckpts = data["clean"], data["z_obs"], data["ckpts"]
    dec = {c: np.stack([decode_block(c, j) for j in range(cfg.n_U)], axis=1) for c in ckpts}  # (n,n_U) observed
    uoh = np.eye(cfg.K + 1)[u_hat[:, [c - 1 for c in ckpts]]]            # (n, n_ckpt, K+1) one-hot u_hat_c

    same, other, intended_c, intended_z = [], [], [], []
    for ci, c in enumerate(ckpts):
        other_dec = np.concatenate([dec[cc] for cc in ckpts if cc != c], axis=1) \
            if len(ckpts) > 1 else np.zeros((clean.shape[0], 1))
        for j in range(cfg.n_U):
            peers = np.delete(dec[c], j, axis=1)                         # decoded same-ckpt peers minus self
            a, m, _ = _cv_acc(peers, clean[:, ci, j], cv=3); same.append(_det(a, m))
            a, m, _ = _cv_acc(other_dec, clean[:, ci, j], cv=3); other.append(_det(a, m))
            a, m, _ = _cv_acc(uoh[:, ci, :], clean[:, ci, j], cv=3); intended_c.append(_det(a, m))
            a, m, _ = _cv_acc(uoh[:, ci, :], z[:, ci, j], cv=3); intended_z.append(_det(a, m))
    return {"shortcut_same_ckpt_clean_det": round(safe_median(same), 4),
            "shortcut_whole_ckpt_clean_det": round(safe_median(other), 4),
            "intended_path_clean_det": round(safe_median(intended_c), 4),
            "intended_path_obs_z_det": round(safe_median(intended_z), 4),
            "mask_policy": "whole_checkpoint",
            "note": "shortcut features = parity-decoded peer bits (encoder-level); random-mask keeps a same-ckpt peer ~99%"}


def base_rates(data, cfg: AccCfg):
    u, e = data["u"], data["e"]
    per_tick, starved_ticks = [], []
    for t in range(cfg.T):
        vals, cnts = np.unique(u[:, t], return_counts=True)
        usable = int(sum(1 for c in cnts if c >= cfg.min_class_support))
        per_tick.append({"tick": t + 1, "hist": {int(v): int(c) for v, c in zip(vals, cnts)},
                         "n_classes": int(len(vals)), "n_usable_classes": usable,
                         "min_class": int(cnts.min()),
                         "maj": round(float(cnts.max() / cnts.sum()), 4)})
        if usable < 2 or int(cnts.min()) < 2:        # unified with probe-ability (rare class -> nan risk)
            starved_ticks.append(t + 1)
    vals, cnts = np.unique(u, return_counts=True)
    pooled = {int(v): int(c) for v, c in zip(vals, cnts)}
    flip_counts = data["x_flip"].sum(axis=0)
    return {"u_per_tick": per_tick, "u_pooled": pooled,
            "u_pooled_classes_starved": [int(v) for v in range(cfg.K + 1)
                                         if pooled.get(v, 0) < cfg.min_class_support],
            "u_ticks_starved": starved_ticks, "e_base_rate": round(float(e.mean()), 4),
            "flip_counts": flip_counts.astype(int).tolist(),
            "flip_min": int(flip_counts.min()),
            "flip_starved_cells": int((flip_counts < cfg.min_flip_support).sum())}


# --------------------------------------------------------------------------- #
def json_clean(v):
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, dict):
        return {(int(k) if isinstance(k, np.integer) else k): json_clean(x) for k, x in v.items()}
    if isinstance(v, list):
        return [json_clean(x) for x in v]
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return json_clean(v.tolist())
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/chatv2/jepa-0d-accumulator-preflight")
    ap.add_argument("--K", type=int, default=None)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--p-event", type=float, default=None)
    ap.add_argument("--p-noise", type=float, default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n-U", type=int, default=None)
    ap.add_argument("--checkpoints", default=None, help="comma-sep ticks, e.g. 4,8,12")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = AccCfg()
    for k, a in [("K", args.K), ("T", args.T), ("p_event", args.p_event), ("p_noise", args.p_noise),
                 ("n", args.n), ("n_U", args.n_U)]:
        if a is not None:
            setattr(cfg, k, a)
    if args.checkpoints:
        cfg.emit_checkpoints = [int(x) for x in args.checkpoints.split(",")]
    cfg.seed = args.seed

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = default_rng(cfg.seed)
    print(f"[cfg] K={cfg.K} T={cfg.T} p_event={cfg.p_event} p_noise={cfg.p_noise} n={cfg.n} "
          f"n_U={cfg.n_U} ckpts={cfg.emit_checkpoints} count_bits={cfg.count_bits}", flush=True)

    data = gen_accumulator(cfg, rng)
    print(f"[gen] L={data['L']} tokens/seq  PSI rows={data['Psi'].tolist()}  "
          f"({round(time.time()-t0,1)}s)", flush=True)

    raw = raw_linear_probes(data, cfg)
    print(f"[deconfound] raw u_det median={raw['u_det_median']} max={raw['u_det_max']}  "
          f"e_det max={raw['e_det_max']}  emission clean/obs="
          f"{raw['emission']['clean_det_median']}/{raw['emission']['obs_det_median']}  "
          f"bitcount->uT={raw['bitcount_uT_det']}  mae_det(sidecar)={raw['u_mae_det_median']}  "
          f"incomplete={raw['eval_incomplete']}  ({round(time.time()-t0,1)}s)", flush=True)

    orc = oracle_recover(data, cfg)
    print(f"[oracle] event-route u_acc={orc['event_route_u_acc_overall']} "
          f"e_acc={orc['event_route_e_acc']} exact-seq={orc['event_route_u_exact_seq']}  "
          f"readout-route={orc['readout_route_u_acc_per_ckpt']}  ({round(time.time()-t0,1)}s)", flush=True)

    msk = mask_necessity(data, cfg, orc["u_hat"])
    print(f"[mask] same-ckpt shortcut(clean)={msk['shortcut_same_ckpt_clean_det']}  "
          f"whole-ckpt shortcut(clean)={msk['shortcut_whole_ckpt_clean_det']}  "
          f"intended path clean/obs={msk['intended_path_clean_det']}/{msk['intended_path_obs_z_det']}  "
          f"({round(time.time()-t0,1)}s)", flush=True)

    br = base_rates(data, cfg)
    print(f"[support] e_base={br['e_base_rate']}  u_pooled_starved={br['u_pooled_classes_starved']}  "
          f"ticks_starved={br['u_ticks_starved']}  flip_min={br['flip_min']} "
          f"flip_starved={br['flip_starved_cells']}", flush=True)

    # ---- gate evaluation (pre-registered bars) ----
    g1 = (not raw["eval_incomplete"]
          and raw["u_det_max"] is not None and raw["u_det_max"] <= cfg.deconfound_max
          and raw["e_det_max"] is not None and raw["e_det_max"] <= cfg.deconfound_max
          and raw["emission"]["clean_det_median"] <= cfg.deconfound_max
          and raw["emission"]["obs_det_median"] <= cfg.deconfound_max
          and (raw["bitcount_uT_det"] is None or raw["bitcount_uT_det"] <= cfg.deconfound_max))
    g2 = orc["event_route_u_acc_overall"] >= cfg.oracle_min
    g3 = (len(br["u_ticks_starved"]) == 0 and br["flip_starved_cells"] == 0)
    g4 = (msk["shortcut_whole_ckpt_clean_det"] <= cfg.mask_shortcut_max
          and msk["intended_path_clean_det"] >= cfg.intended_path_min)

    verdict = ("blocked_by_deconfound_leak" if not g1 else
               "blocked_by_absent_functional" if not g2 else
               "support_starved" if not g3 else
               "blocked_by_mask_shortcut" if not g4 else
               "preflight_pass_ready_for_spec")

    manifest = {"lane": "jepa-0d-accumulator", "stage": "model_free_preflight",
                "verdict": verdict,
                "gates": {"deconfound": g1, "oracle": g2, "support": g3, "mask_necessity": g4},
                "bars": {"deconfound_max": cfg.deconfound_max, "oracle_min": cfg.oracle_min,
                         "min_class_support": cfg.min_class_support, "min_flip_support": cfg.min_flip_support,
                         "mask_shortcut_max": cfg.mask_shortcut_max, "intended_path_min": cfg.intended_path_min},
                "cfg": asdict(cfg), "count_bits": cfg.count_bits, "L": data["L"],
                "psi": data["Psi"].tolist(),
                "raw_linear": raw, "oracle": {k: v for k, v in orc.items() if k != "u_hat"},
                "mask_necessity": msk, "base_rates": br, "wall_s": round(time.time() - t0, 1)}
    (out / "preflight.json").write_text(json.dumps(json_clean(manifest), indent=2))

    with (out / "u_per_tick.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "raw_u_acc", "raw_u_maj", "raw_u_det", "raw_u_mae_det",
                    "oracle_u_acc", "n_classes", "n_usable_classes", "min_class", "probe_method"])
        for t in range(cfg.T):
            rr, bb = raw["u_per_tick"][t], br["u_per_tick"][t]
            w.writerow([t + 1, rr["acc"], rr["maj"], rr["u_det"], rr["mae_det"],
                        orc["event_route_u_acc_per_tick"][t], bb["n_classes"],
                        bb["n_usable_classes"], bb["min_class"], rr["method"]])

    print(f"\n==== PREFLIGHT VERDICT: {verdict} ====")
    print(f"  gate1 de-confound (raw u_det<= {cfg.deconfound_max}, complete):  {'PASS' if g1 else 'FAIL'}")
    print(f"  gate2 oracle (u recovery>= {cfg.oracle_min}):                  {'PASS' if g2 else 'FAIL'}")
    print(f"  gate3 support (no starvation):                       {'PASS' if g3 else 'FAIL'}")
    print(f"  gate4 mask-necessity (whole-ckpt shortcut<= {cfg.mask_shortcut_max},"
          f" path>= {cfg.intended_path_min}): {'PASS' if g4 else 'FAIL'}")
    print(f"        same-ckpt shortcut(clean)={msk['shortcut_same_ckpt_clean_det']} "
          f"-> whole-ckpt={msk['shortcut_whole_ckpt_clean_det']} | path={msk['intended_path_clean_det']}")
    print(f"  wrote {out/'preflight.json'} and u_per_tick.csv  ({round(time.time()-t0,1)}s)", flush=True)


if __name__ == "__main__":
    main()
