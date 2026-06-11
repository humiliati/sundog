#!/usr/bin/env python
"""H10 ADV-HIDE-D -- can an incentivized body conceal the determine latent, and at what cost?

Prereg: docs/atlas/H10_ADV_HIDE_D_PREREG.md (frozen 2026-06-11, commit c964031c, BEFORE this file
existed). Slate HS3. Attacks the DETERMINE half of banked H3 v2: the asymmetry (continuous-resist
defeatable, discrete-determine robust) was established under INDIFFERENT objectives only. Here the
encoder is given an explicit hide-d incentive (gradient reversal vs a pooled d-probe), on:

  * the FROZEN v2 substrate (exchangeable phi) -- symmetry-prior CONTROL: per-unit d-info is
    d-odd, K=64 pooling amplifies it; an exchangeable encoder cannot cancel it positionally ->
    prediction: NO hide-with-retention (falsifiable; a guarded hide here breaks the prior);
  * the HIDE-ENABLED substrate (per-unit sinusoidal positional channel, P=8) -- the LIVE case:
    the analytic arm A3 proves hide-with-retention EXISTS here (s_i*zhat_i with alternating
    s_i: pooled d-component = d*(sum s_i)/K + noise = EXACTLY 0; per-unit index-aware retention
    at the Bayes ceiling Phi(1/1.5) ~ 0.748). A3 is an EXISTENCE DEMO, not a kill arm (prereg
    section 0 spec correction). The falsifiable question is TRAINABILITY.

Pre-registered bands (TRAINED arms, hide-enabled substrate, guards intact):
  BREAK: some guarded arm-run drives best-of-3-probe pooled d-acc < 0.90 at some eval lambda.
  KILL : every guarded arm stays >= 0.90 at every eval lambda (incl. the no-guarded-arm case)
         -> determine upgrades to "robust to adversarial incentive within the attack class".
Guards: primary c-R2 >= arm-baseline - 0.05; per-unit retention >= max(baseline - 0.05, 0.65).

ZERO-CONFLICT: imports shadow_pooled_synthetic_v2 (frozen apparatus) and modifies NOTHING in it
(the H3-PC session runs off that module concurrently).

NOT public-eligible. Attribution: Ganin & Lempitsky 2015 (GRL/DANN); Elazar & Goldberg 2018;
Edwards & Storkey 2015; LEACE 2023; DeepSets (Zaheer 2017).
Run:  python scripts/shadow_adv_hide_d.py            (full verdict run, ~2-4 h CPU)
      python scripts/shadow_adv_hide_d.py --smoke    (plumbing check, no verdict)
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
import torch                                    # noqa: E402
import torch.nn as nn                           # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier   # noqa: E402
from sklearn.neural_network import MLPClassifier     # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import shadow_pooled_synthetic_v2 as v2         # noqa: E402  (FROZEN apparatus -- import only)

torch.set_num_threads(1)
SEED = 5150                                     # H10 seed family (distinct from v2's 1234)
K, M, D, H = v2.K, v2.M, v2.D, v2.H
P = 8                                           # positional dims (4 freqs x sin/cos)
F_FROZEN, F_HIDE = v2.F, v2.F + P
TRAIN_LAM = v2.TRAIN_LAM
EVAL_LAMS = [0.0, 1.0, 2.0, 3.0]
LADV = [0.1, 1.0, 10.0]
DACC_LINE = 0.90
GUARD_DELTA = 0.05
RET_FLOOR = 0.65
RET_ROWS = 50_000
EPOCHS, BS, LR, PROBE_LR, PROBE_STEPS = 120, 256, 1e-3, 1e-2, 100
N_TRAIN, N_PROBE = v2.N_TRAIN, v2.N_PROBE
ARM_SEED = {"b0_regc": 101, "b0_ret": 102, "b0_rffonly": 103,
            "A1_0.1": 111, "A1_1.0": 112, "A1_10.0": 113,
            "A2_0.1": 121, "A2_1.0": 122, "A2_10.0": 123}

# positional channel: slot constants, sample/c/d-independent
_i = np.arange(K)
POS = np.stack([f(2 * np.pi * fr * _i / K) for fr in (1, 2, 4, 8) for f in (np.sin, np.cos)],
               axis=1).astype(np.float32)        # (K, P)
S_SIGNS = ((-1.0) ** _i).astype(np.float32)      # alternating signs, sum = 0 (K even)


def gen_hide(n, lam, seed):
    """v2.gen + positional features appended -> units (n, K, F_HIDE)."""
    u, c, d = v2.gen(n, lam, seed)
    pos = np.broadcast_to(POS[None], (n, K, P))
    return np.concatenate([u, pos], axis=2), c, d


def gen_sub(substrate, n, lam, seed):
    return gen_hide(n, lam, seed) if substrate == "hide" else v2.gen(n, lam, seed)


class PhiH(nn.Module):
    """Banked architecture family: per-unit MLP -> mean-pool."""

    def __init__(self, fin):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(fin, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, H))

    def units(self, u):
        return self.net(u)                       # (B, K, H)

    def forward(self, u):
        return self.net(u).mean(dim=1)           # (B, H)


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def train_arm(arm, substrate, units, c, d, ladv=0.0, retention=False, mask_disc=False,
              epochs=EPOCHS):
    """One training run. arm in ARM_SEED; substrate in {frozen, hide}. Deterministic."""
    base = SEED + ARM_SEED[arm] + (1000 if substrate == "frozen" else 0)
    torch.manual_seed(base)
    np.random.seed(base)
    fin = units.shape[2]
    phi = PhiH(fin)
    reg_head = nn.Linear(H, 1)
    ret_head = nn.Linear(H, 2) if retention else None
    probe = nn.Linear(H, 2) if ladv > 0 else None
    U = torch.tensor(units)
    if mask_disc:                                # A3 c-pathway: disc dims zeroed at train AND eval
        U = U.clone()
        U[:, :, M:M + D] = 0.0
    Ct = torch.tensor(c).float()[:, None]
    Dt = torch.tensor((d > 0).astype(np.int64))
    mse, ce = nn.MSELoss(), nn.CrossEntropyLoss()
    enc_params = list(phi.parameters()) + list(reg_head.parameters()) + (
        list(ret_head.parameters()) if retention else [])
    opt = torch.optim.Adam(enc_params, lr=LR)
    opt_p = torch.optim.Adam(probe.parameters(), lr=PROBE_LR) if probe is not None else None
    n = U.shape[0]
    for epoch in range(epochs):
        if probe is not None:
            # per-epoch probe refresh: re-init, retrain on CACHED current reps (encoder frozen)
            torch.manual_seed(base + 10_000 + epoch)
            probe = nn.Linear(H, 2)
            opt_p = torch.optim.Adam(probe.parameters(), lr=PROBE_LR)
            with torch.no_grad():
                reps = torch.cat([phi(U[b:b + 1024]) for b in range(0, n, 1024)])
            pperm = torch.randperm(n)
            for s in range(PROBE_STEPS):
                idx = pperm[(s * BS) % n:(s * BS) % n + BS]
                if len(idx) < 8:
                    continue
                opt_p.zero_grad()
                ce(probe(reps[idx]), Dt[idx]).backward()
                opt_p.step()
        perm = torch.randperm(n)
        for b in range(0, n, BS):
            idx = perm[b:b + BS]
            opt.zero_grad()
            if opt_p is not None:
                opt_p.zero_grad()
            uin = U[idx]
            per_unit = phi.units(uin)
            pooled = per_unit.mean(dim=1)
            loss = mse(reg_head(pooled), Ct[idx])
            if retention:
                loss = loss + ce(ret_head(per_unit).reshape(-1, 2),
                                 Dt[idx].repeat_interleave(uin.shape[1]))
            if probe is not None:
                loss = loss + ce(probe(GradReverse.apply(pooled, float(ladv))), Dt[idx])
            loss.backward()
            opt.step()
            if opt_p is not None:
                opt_p.step()                     # probe learns at full strength (normal grads)
    phi.eval()
    with torch.no_grad():
        pred = reg_head(torch.cat([phi(U[b:b + 1024]) for b in range(0, n, 1024)]))[:, 0].numpy()
    fit = float(1 - np.var(c - pred) / np.var(c))
    return phi, fit


def pool_feats(phi, units, mask_disc=False):
    U = torch.tensor(units)
    if mask_disc:
        U = U.clone()
        U[:, :, M:M + D] = 0.0
    with torch.no_grad():
        return torch.cat([phi(U[b:b + 1024]) for b in range(0, U.shape[0], 1024)]).numpy()


def unit_feats(phi, units, mask_disc=False):
    U = torch.tensor(units)
    if mask_disc:
        U = U.clone()
        U[:, :, M:M + D] = 0.0
    with torch.no_grad():
        return torch.cat([phi.units(U[b:b + 1024]) for b in range(0, U.shape[0], 1024)]).numpy()


def a3_pool(phi_rff, units):
    """A3 analytic encoder: [phi_rffonly(u, disc zeroed) , s_i * zhat_i] mean-pooled."""
    zhat = units[:, :, M:M + D] @ v2.A_DISC                          # (n, K)
    anti = (zhat * S_SIGNS[None, :]).mean(axis=1, keepdims=True)     # (n, 1) -- exactly d-free
    return np.concatenate([pool_feats(phi_rff, units, mask_disc=True), anti], axis=1)


def a3_units(phi_rff, units):
    zhat = units[:, :, M:M + D] @ v2.A_DISC
    anti = (zhat * S_SIGNS[None, :])[:, :, None]                     # (n, K, 1)
    return np.concatenate([unit_feats(phi_rff, units, mask_disc=True), anti], axis=2)


def d_probe_best(X, y):
    """Best-of-3 independent retrained probes (the H3 probe-robustness protocol)."""
    Xs = StandardScaler().fit_transform(X)
    yb = (y > 0).astype(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    out = {}
    out["logistic"] = float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=skf,
                                            scoring="balanced_accuracy").mean())
    out["mlp"] = float(cross_val_score(MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                                                     random_state=0), Xs, yb, cv=skf,
                                       scoring="balanced_accuracy").mean())
    out["knn"] = float(cross_val_score(KNeighborsClassifier(n_neighbors=15), Xs, yb, cv=skf,
                                       scoring="balanced_accuracy").mean())
    out["best"] = max(out["logistic"], out["mlp"], out["knn"])
    return out


def retention_acc(units_h, d, with_pos=True):
    """Per-unit retention: best-of-{logistic, MLP(64,)} d-acc on concat[phi_i, pos_i], 50k rows."""
    n, k, hh = units_h.shape
    rows = units_h.reshape(n * k, hh)
    if with_pos:
        rows = np.concatenate([rows, np.tile(POS, (n, 1))], axis=1)
    y = np.repeat((d > 0).astype(int), k)
    rng = np.random.default_rng(SEED + 77)
    sel = rng.choice(n * k, size=min(RET_ROWS, n * k), replace=False)
    Xs = StandardScaler().fit_transform(rows[sel])
    yb = y[sel]
    skf = StratifiedKFold(3, shuffle=True, random_state=0)
    lo = float(cross_val_score(LogisticRegression(max_iter=2000), Xs, yb, cv=skf,
                               scoring="balanced_accuracy").mean())
    ml = float(cross_val_score(MLPClassifier(hidden_layer_sizes=(64,), max_iter=300,
                                             random_state=0), Xs, yb, cv=skf,
                               scoring="balanced_accuracy").mean())
    return max(lo, ml)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="plumbing check (reduced; NO verdict)")
    ap.add_argument("--out", default="results/atlas/h10/adv_hide_d_result.json")
    args = ap.parse_args()
    t0 = time.time()
    global EPOCHS
    epochs = 30 if args.smoke else EPOCHS
    n_train = 2000 if args.smoke else N_TRAIN
    n_probe = 800 if args.smoke else N_PROBE
    eval_lams = [1.0, 2.0] if args.smoke else EVAL_LAMS
    mode = "SMOKE (no verdict)" if args.smoke else "FULL (verdict run)"

    print("=" * 92)
    print(f"H10 ADV-HIDE-D -- adversarial concealment of the determine latent   [{mode}]")
    print(f"  arms: b0_regc, b0_ret, A1(reg_c+GRL), A2(reg_c+ret-aux+GRL) x ladv {LADV};")
    print(f"  A3 analytic existence demo (hide-enabled only). guards: primary -{GUARD_DELTA}, "
          f"retention max(b0-{GUARD_DELTA}, {RET_FLOOR}).")
    print(f"  bands: BREAK if guarded arm < {DACC_LINE} pooled d-acc at some lam in {eval_lams}; "
          f"else KILL.")
    print("=" * 92)

    results = {"smoke": bool(args.smoke), "arms": {}}
    eval_sets = {}
    for lam in eval_lams:
        eval_sets[("frozen", lam)] = v2.gen(n_probe, lam, SEED + 7 + int(lam * 1000))
        eval_sets[("hide", lam)] = gen_hide(n_probe, lam, SEED + 7 + int(lam * 1000))

    # ---- substrate gates on gen_hide (C0/C1/DET analogs; ABORT if broken) ---- #
    uh2, ch2, dh2 = eval_sets[("hide", 2.0)] if ("hide", 2.0) in eval_sets else gen_hide(
        n_probe, 2.0, SEED + 7 + 2000)
    raw_c2 = v2.c_r2(uh2.mean(axis=1), ch2)
    uh0, ch0, dh0 = gen_hide(n_probe, 0.0, SEED + 7)
    unit_c0 = v2.c_r2(uh0[:, 0, :], ch0)
    gate_c0 = raw_c2 <= 0.08
    gate_c1 = unit_c0 >= 0.5
    print(f"\n[GATE] hide-substrate C0 raw-wash @lam=2: c-R2={raw_c2:.3f} (<=0.08) "
          f"-> {'PASS' if gate_c0 else 'ABORT'}")
    print(f"[GATE] hide-substrate C1 unit-carries-c @lam=0: c-R2={unit_c0:.3f} (>=0.5) "
          f"-> {'PASS' if gate_c1 else 'ABORT'}")

    def run_arm(name, substrate, ladv=0.0, retention=False, mask_disc=False, seed_key=None):
        seed_key = seed_key or name
        tr_u, tr_c, tr_d = gen_sub(substrate, n_train, TRAIN_LAM, SEED + 1)
        t1 = time.time()
        phi, fit = train_arm(seed_key, substrate, tr_u, tr_c, tr_d, ladv=ladv,
                             retention=retention, mask_disc=mask_disc, epochs=epochs)
        row = {"substrate": substrate, "ladv": ladv, "train_fit": fit, "d": {}, "c": {}}
        for lam in eval_lams:
            u, c, d = eval_sets[(substrate, lam)]
            Xp = pool_feats(phi, u, mask_disc=mask_disc)
            row["d"][str(lam)] = d_probe_best(Xp, d)
            row["c"][str(lam)] = v2.c_r2(Xp, c)
        u, c, d = eval_sets[(substrate, TRAIN_LAM)]
        row["retention"] = retention_acc(unit_feats(phi, u, mask_disc=mask_disc), d,
                                         with_pos=(substrate == "hide"))
        row["primary_r2"] = row["c"][str(TRAIN_LAM)]
        row["min_d_best"] = min(row["d"][str(l)]["best"] for l in eval_lams)
        print(f"  [{substrate:>6}] {name:<10} ladv={ladv:<5} fit={fit:.3f} "
              f"primary_c_R2={row['primary_r2']:.3f} ret={row['retention']:.3f} "
              f"min_d_best={row['min_d_best']:.3f}  ({time.time() - t1:.0f}s)", flush=True)
        return phi, row

    # ---- baselines ---- #
    print("\nBASELINES:")
    arms = {}
    for sub in ["hide", "frozen"]:
        _, arms[f"b0_regc[{sub}]"] = run_arm("b0_regc", sub)
        _, arms[f"b0_ret[{sub}]"] = run_arm("b0_ret", sub, retention=True)
    phi_rff, row_rff = run_arm("b0_rffonly", "hide", mask_disc=True)
    arms["b0_rffonly[hide]"] = row_rff

    # ---- trained attack arms ---- #
    print("\nTRAINED ATTACK ARMS:")
    for sub in ["hide", "frozen"]:
        for ladv in LADV:
            _, arms[f"A1_{ladv}[{sub}]"] = run_arm(f"A1_{ladv}", sub, ladv=ladv)
            _, arms[f"A2_{ladv}[{sub}]"] = run_arm(f"A2_{ladv}", sub, ladv=ladv, retention=True)

    # ---- A3 analytic existence demo (hide only) ---- #
    print("\nA3 ANALYTIC (existence demo, not a kill arm):")
    a3 = {"substrate": "hide", "d": {}, "c": {}}
    for lam in eval_lams:
        u, c, d = eval_sets[("hide", lam)]
        Xp = a3_pool(phi_rff, u)
        a3["d"][str(lam)] = d_probe_best(Xp, d)
        a3["c"][str(lam)] = v2.c_r2(Xp, c)
    u, c, d = eval_sets[("hide", TRAIN_LAM)]
    a3["retention"] = retention_acc(a3_units(phi_rff, u), d, with_pos=True)
    a3["primary_r2"] = a3["c"][str(TRAIN_LAM)]
    a3["max_d_best"] = max(a3["d"][str(l)]["best"] for l in eval_lams)
    a3["min_d_best"] = min(a3["d"][str(l)]["best"] for l in eval_lams)
    arms["A3[hide]"] = a3
    print(f"  [  hide] A3 analytic: pooled d-acc range [{a3['min_d_best']:.3f}, "
          f"{a3['max_d_best']:.3f}]  ret={a3['retention']:.3f}  c_R2={a3['primary_r2']:.3f}")

    # ---- guards + verdict ---- #
    base_primary = {"A1": {s: arms[f"b0_regc[{s}]"]["primary_r2"] for s in ("hide", "frozen")},
                    "A2": {s: arms[f"b0_ret[{s}]"]["primary_r2"] for s in ("hide", "frozen")}}
    base_ret = {"A1": {s: arms[f"b0_regc[{s}]"]["retention"] for s in ("hide", "frozen")},
                "A2": {s: arms[f"b0_ret[{s}]"]["retention"] for s in ("hide", "frozen")}}
    det_min_b0 = min(arms[f"b0_regc[{s}]"]["min_d_best"] for s in ("hide", "frozen"))
    gate_det = det_min_b0 >= 0.95
    print(f"\n[GATE] DET: b0 pooled d-acc >= 0.95 on both substrates (min={det_min_b0:.3f}) "
          f"-> {'PASS' if gate_det else 'ABORT'}")

    print(f"\n{'arm':<12} {'sub':>6} {'primary':>8} {'(>=)':>7} {'ret':>6} {'(>=)':>7} "
          f"{'guard':>6} {'min_d':>6}")
    verdict_rows, frozen_ctrl_rows = [], []
    for fam in ("A1", "A2"):
        for sub in ("hide", "frozen"):
            for ladv in LADV:
                r = arms[f"{fam}_{ladv}[{sub}]"]
                p_need = base_primary[fam][sub] - GUARD_DELTA
                r_need = max(base_ret[fam][sub] - GUARD_DELTA, RET_FLOOR)
                guarded = (r["primary_r2"] >= p_need) and (r["retention"] >= r_need)
                r["guarded"] = bool(guarded)
                r["primary_need"], r["ret_need"] = p_need, r_need
                (verdict_rows if sub == "hide" else frozen_ctrl_rows).append(
                    (f"{fam}_{ladv}", r))
                print(f"{fam}_{ladv:<9} {sub:>6} {r['primary_r2']:>8.3f} {p_need:>7.3f} "
                      f"{r['retention']:>6.3f} {r_need:>7.3f} {str(guarded):>6} "
                      f"{r['min_d_best']:>6.3f}")

    guarded_hide = [(nm, r) for nm, r in verdict_rows if r["guarded"]]
    break_rows = [(nm, r) for nm, r in guarded_hide if r["min_d_best"] < DACC_LINE]
    killed = len(break_rows) == 0                      # KILL = clean strengthening null
    frozen_guarded_breaks = [(nm, r) for nm, r in frozen_ctrl_rows
                             if r["guarded"] and r["min_d_best"] < DACC_LINE]

    # A3 demo vs pinned predictions
    a3_ok = (0.45 <= a3["min_d_best"] and a3["max_d_best"] <= 0.55
             and a3["retention"] >= 0.70
             and abs(a3["primary_r2"] - row_rff["primary_r2"]) <= 0.05)

    print("\n" + "=" * 92)
    print(f"A3 existence demo vs pinned predictions: pooled in [0.45,0.55]? "
          f"[{a3['min_d_best']:.3f},{a3['max_d_best']:.3f}]  ret>=0.70? {a3['retention']:.3f}  "
          f"|c_R2 - b0_rffonly|<=0.05? {abs(a3['primary_r2'] - row_rff['primary_r2']):.3f}  "
          f"-> {'CONFIRMED' if a3_ok else 'FAILED (abort-grade investigation)'}")
    print(f"Frozen-substrate symmetry-prior control: guarded hides = "
          f"{[nm for nm, _ in frozen_guarded_breaks] or 'NONE (prior confirmed)'}")
    gates_ok = gate_c0 and gate_c1 and gate_det
    if args.smoke:
        verdict = "smoke"
        print("\nSMOKE complete -- plumbing only, NO verdict.")
    elif not gates_ok:
        verdict = "gate_abort"
        print("\nRESULT: GATE ABORT (substrate/apparatus; fix and rerun -- NOT a verdict).")
    elif not a3_ok:
        verdict = "a3_anomaly_abort"
        print("\nRESULT: A3 ANOMALY -- the exactness demo failed its pinned predictions; "
              "abort-grade investigation before any verdict.")
    elif not killed:
        verdict = "break"
        nm, r = break_rows[0]
        print(f"\nRESULT: BREAK. Guarded trained adversary hides the determine latent: e.g. {nm} "
              f"pooled d-acc {r['min_d_best']:.3f} < {DACC_LINE} with primary "
              f"{r['primary_r2']:.3f} and retention {r['retention']:.3f}. H3's determine-half "
              "asymmetry collapses to objective-dependence; the frontier is the banked artifact.")
    else:
        verdict = "kill"
        print(f"\nRESULT: KILL (clean strengthening null). No guarded trained arm gets below "
              f"{DACC_LINE} at any eval lambda (guarded arms: "
              f"{[nm for nm, _ in guarded_hide] or 'none passed guards'}). Determine upgrades to "
              "'robust to adversarial incentive WITHIN THE GRL ATTACK CLASS' -- while A3 proves "
              "hide-with-retention exists: the gap between existence and trainability is the "
              "banked sentence.")
    print("=" * 92)

    out = Path(args.out if not args.smoke else args.out.replace(".json", "_smoke.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    results.update(dict(
        verdict=verdict, gates=dict(c0=bool(gate_c0), c1=bool(gate_c1), det=bool(gate_det)),
        a3_ok=bool(a3_ok), eval_lams=eval_lams, epochs=epochs, n_train=n_train, n_probe=n_probe,
        seed=SEED, dacc_line=DACC_LINE, guard_delta=GUARD_DELTA, ret_floor=RET_FLOOR,
        base_primary=base_primary, base_ret=base_ret,
        frozen_guarded_breaks=[nm for nm, _ in frozen_guarded_breaks],
        break_arms=[nm for nm, _ in break_rows], arms=arms,
        wall_s=round(time.time() - t0, 1)))
    out.write_text(json.dumps(results, indent=2,
                              default=lambda o: float(o) if isinstance(o, np.floating) else o))
    print(f"\nwrote {out}  ({round(time.time() - t0, 1)}s)")
    return 0 if (args.smoke or (gates_ok and a3_ok)) else 1


if __name__ == "__main__":
    sys.exit(main())
