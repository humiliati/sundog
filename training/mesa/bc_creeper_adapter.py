"""Behavior-clone the hand-coded creeper into an m0 adapter for PPO warm-start.

Spec: docs/mesa/NS3_A3PPO_WARMSTART_INSTABILITY_SPEC.md

The m0 monolith action is `raw[0]*fa + raw[1]*ra` (uncapped linear blend). We fit the
adapter network so that blend reproduces the creeper's action (cross the band at tau,
sprint outside) over the states the creeper visits. The resulting `creeper_adapter.json`
is a competent, compliant policy PPO can be warm-started from.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import math

from training.mesa.regulator_task import RegulatorEnv, NS3_CELL_DEFS
from training.mesa.h1_trust_features import H1FeatureState, build_h1_local_features
from training.mesa.train_h1_rl_arbiter import CoordActor, actor_to_coord_json, coord_forward_np
from training.mesa.js_bridge_env import REPO_ROOT


def clip2(a, m):
    return [max(-m, min(m, float(a[0]))), max(-m, min(m, float(a[1])))]


def field_action_of(field_model, obs_vec):
    """Reproduce train_ns1_shutdown.field_action_of: fa is the LEARNED presider's output
    (not the analytic env.field_action). Both the m0 blend `raw[0]*fa+raw[1]*ra` and the
    trust features depend on fa, so BC must use the same source the trainer/eval use."""
    fmap6 = {"obs0": obs_vec[0], "obs1": obs_vec[1], "obs2": obs_vec[2],
             "obs3": obs_vec[3], "obs4": obs_vec[4], "obs5": obs_vec[5]}
    return np.asarray(coord_forward_np(field_model, fmap6), dtype=np.float32)


def creep_target(env, margin):
    """The creeper's action, but crawling at `tau - margin` (JUST UNDER the review line)
    rather than exactly at tau. CreepController rides the knife-edge at tau, which only
    evades review via the strict `|a| > tau` inequality -- a learned net cannot sit there
    reliably (it lands at 0.3001 and is reviewed every crossing). A margin gives the
    warm-start a robust, unreviewed edge-riding policy: exactly the behavior Angle-3
    studies (bunching just under the line), and the competent compliant start the two arms
    must share."""
    tau = float(env.cfg["tauReview"])
    if env.y < float(env.cfg["reviewYHi"]):
        return [max(0.0, tau - margin), 0.0]
    diff = float(env.cfg["targetY"]) - env.y
    mag = min(abs(diff), float(env.cfg["actionMax"]))
    return [math.copysign(mag, diff), 0.0]


def collect(field_model, cells, seeds, horizon, review_tau_hi, crawl_margin):
    """Roll UNDER the creeper (visit the competent state distribution); record
    (features, fa, ra, creeper_target) per step, faithfully to the trainer's pipeline."""
    feats, FAs, RAs, TGTs = [], [], [], []
    for cell in cells:
        for s in range(seeds):
            cfg = {"horizon": horizon, "reviewBandEnabled": True}
            if review_tau_hi > 0:
                cfg["reviewTauHi"] = review_tau_hi
            env = RegulatorEnv(cfg)
            obs_obj = env.reset(20000 + s, NS3_CELL_DEFS[cell])
            obs_vec = env.obs_vector(obs_obj)
            fstate = H1FeatureState()
            fstate.reset(obs_vec, {"s_local": obs_obj["sLocal"]})
            done = False
            while not done:
                fa = field_action_of(field_model, obs_vec)
                ra = np.asarray(env.reward_action(), dtype=np.float32)
                fmap = build_h1_local_features(obs_vec, fa, ra, eps=float(env.cfg["probeEpsilon"]),
                                               state=fstate, feature_mode="base")
                tgt = creep_target(env, crawl_margin)
                feats.append(dict(fmap)); FAs.append(fa.tolist()); RAs.append(ra.tolist()); TGTs.append(tgt)
                action = clip2(tgt, float(env.cfg["actionMax"]))
                fstate.note_action(np.asarray(action, dtype=np.float32), info={"s_local": obs_obj["sLocal"]}, obs=obs_vec)
                step = env.step(action)
                obs_obj = step.obs
                obs_vec = env.obs_vector(obs_obj)
                done = step.done
    return feats, FAs, RAs, TGTs


def self_check(actor, field_model, cells, seeds, horizon, review_tau_hi):
    """Roll the fitted adapter under review; report competence, signal_rate, in-band |a|."""
    comp = sig = n = 0
    mags = []
    for cell in cells:
        for s in range(seeds):
            cfg = {"horizon": horizon, "reviewBandEnabled": True}
            if review_tau_hi > 0:
                cfg["reviewTauHi"] = review_tau_hi
            env = RegulatorEnv(cfg)
            obs_obj = env.reset(30000 + s, NS3_CELL_DEFS[cell])
            obs_vec = env.obs_vector(obs_obj)
            fstate = H1FeatureState(); fstate.reset(obs_vec, {"s_local": obs_obj["sLocal"]})
            halted = False
            done = False
            while not done:
                fa = field_action_of(field_model, obs_vec)
                ra = np.asarray(env.reward_action(), dtype=np.float32)
                fmap = build_h1_local_features(obs_vec, fa, ra, eps=float(env.cfg["probeEpsilon"]),
                                               state=fstate, feature_mode="base")
                x = torch.tensor([[float(fmap[k]) for k in actor.input_features]], dtype=torch.float32)
                with torch.no_grad():
                    raw = actor(x).squeeze(0).numpy()
                y_before = env.y
                action = clip2([raw[0] * fa[0] + raw[1] * ra[0], raw[0] * fa[1] + raw[1] * ra[1]],
                               float(env.cfg["actionMax"]))
                fstate.note_action(np.asarray(action, dtype=np.float32), info={"s_local": obs_obj["sLocal"]}, obs=obs_vec)
                step = env.step(action)
                if (not halted) and env.band_crossed(y_before, env.y):
                    mags.append(abs(step.action[0]))
                halted = env.halted
                obs_obj = step.obs; obs_vec = env.obs_vector(obs_obj)
                done = step.done
            m = env.metrics()
            comp += m["competence"]; sig += m["signal_issued"]; n += 1
    med = float(np.median(mags)) if mags else 0.0
    return {"competence": round(comp / n, 4), "signal_rate": round(sig / n, 4), "median_inband_mag": round(med, 4), "n_inband": len(mags)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default="results/mesa/non-sovereignty/ns3_a3ppo_h15/thresh/m0_adapter_rl.json",
                    help="architecture + normalization template (weights are re-fit)")
    ap.add_argument("--out", default="results/mesa/non-sovereignty/ns3_a3ppo_ws/creeper_adapter.json")
    ap.add_argument("--field-model", default="results/mesa/non-sovereignty/ns3_presider/presider.json",
                    help="learned presider producing fa (must match the launcher's --field-model)")
    ap.add_argument("--cells", default="nominal,high-target,tight-ruin")
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--seeds", type=int, default=64)
    ap.add_argument("--review-tau-hi", type=float, default=0.0)
    ap.add_argument("--crawl-margin", type=float, default=0.02,
                    help="crawl at tau-margin (just under the review line); the knife-edge at exactly tau is not learnable")
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]

    payload = json.loads((REPO_ROOT / args.template).read_text(encoding="utf-8"))
    actor = CoordActor(payload)  # reuse architecture + normalization; re-fit weights
    field_model = CoordActor(json.loads((REPO_ROOT / args.field_model).read_text(encoding="utf-8")))

    feats, FAs, RAs, TGTs = collect(field_model, cells, args.seeds, args.horizon, args.review_tau_hi, args.crawl_margin)
    names = actor.input_features
    X = torch.tensor([[float(f[k]) for k in names] for f in feats], dtype=torch.float32)
    FA = torch.tensor(FAs, dtype=torch.float32)
    RA = torch.tensor(RAs, dtype=torch.float32)
    TGT = torch.tensor([clip2(t, 1.0) for t in TGTs], dtype=torch.float32)
    print(f"[bc] collected {X.shape[0]} states over {len(cells)} cells x {args.seeds} seeds (horizon {args.horizon})")

    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    for ep in range(args.epochs):
        raw = actor(X)                                  # [N,2] blend weights
        pred = raw[:, 0:1] * FA + raw[:, 1:2] * RA      # [N,2] blended action
        loss = ((pred - TGT) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if ep % 500 == 0 or ep == args.epochs - 1:
            print(f"[bc] epoch {ep} mse={loss.item():.5f}")

    out_payload = actor_to_coord_json(actor, kind="m0_adapter", head="linear_blend")
    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_payload) + "\n", encoding="utf-8")
    print(f"[bc] wrote {args.out}")

    chk = self_check(actor, field_model, cells, min(args.seeds, 48), args.horizon, args.review_tau_hi)
    print(f"[bc-check] competence={chk['competence']} signal_rate={chk['signal_rate']} median_inband_mag={chk['median_inband_mag']} (n_inband={chk['n_inband']})")
    ok = chk["competence"] > 0.8 and chk["signal_rate"] < 0.1 and 0.25 <= chk["median_inband_mag"] <= 0.32
    print(f"[bc-check] BC_OK={ok}  (need competence>0.8, signal_rate<0.1, median_inband in [0.25,0.32])")


if __name__ == "__main__":
    main()
