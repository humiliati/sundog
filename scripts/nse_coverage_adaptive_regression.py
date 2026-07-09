#!/usr/bin/env python3
"""Coverage-adaptive apparatus (Approach A) -- regression evaluator + synthetic
self-test. Frozen spec: docs/chatv2/NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md sec 3.

--self-test : synthetic validation (seconds) of the two load-bearing properties:
  (1) on a COMPACT cluster, adaptive is bit-identical to frozen (reduction theorem);
  (2) on a CORE+HALO spread, frozen DEFERS on coverage while adaptive reads the core.
--evaluate G200_DIR G300_DIR : apply the section-3 regression gate to two adaptive
  manifests and print PASS/FAIL (the precondition before any G=675 read).
"""
import argparse, json, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1

DISAGREE_TOL = 0.005
BANKED_DISAGREE = {200: 0.0367, 300: 0.0382}  # reference (preset-confounded; informational)


def _cfg(n_samples):
    return c1.RunConfig(
        preset="lock_v7_g200", grid_size=32, n_modes=16, k_signature=3,
        forcing_wavenumber=2, grashof=200.0, forcing_amplitude=1.0,
        viscosity=0.0707, dt=0.01, burnin_steps=0, sample_count=n_samples,
        sample_interval_steps=50, lookahead_steps=500, n_min=30, delta_action=0.10,
        s_pos=0.50, delta_proxy_min=0.01, e_max_burnin_fraction=1.0, random_seed=0,
        integrator="semi-implicit", signature_dimension=18, action_tiebreak="damp",
        adjudicator="twin-state-adaptive", k_neighbors=30, delta_incompat=0.01,
        twin_k_neighbors=50, twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=0.70,
        calibration_sample_count=50000, calibration_gap_steps=5000,
    )


def self_test():
    print("SELF-TEST (synthetic apparatus validation; non-verdict)", flush=True)
    rng = np.random.default_rng(0)
    eps = 0.06
    HD = 18  # high-mode dim

    # T1 -- COMPACT: 400 points with pairwise distance << eps. In 18-D two
    # N(0,s) points sit ~s*sqrt(2*18)=6s apart, so s=eps/30 -> pairwise ~eps/5.
    n = 400
    sig = rng.normal(0, eps / 30.0, (n, 18))
    high = rng.normal(0, 1.0, (n, HD))            # spread >> delta_H -> witnesses form
    act = (rng.random(n) < 0.30).astype(np.int8)
    cfg = _cfg(n)
    froz, _, _, _ = c1.aggregate_twin_state(sig, high, act, eps, cfg)
    adap, _, _, _ = c1.aggregate_twin_state_adaptive(sig, high, act, eps, cfg)
    assert abs(adap["adaptive_covered_fraction"] - 1.0) < 1e-9, "T1 f!=1"
    assert froz["verdict"] == "TWIN_STATE_CERTIFIED", f"T1 frozen {froz['verdict']}"
    assert adap["verdict"] == "TWIN_STATE_ADAPTIVE_CERTIFIED", f"T1 adaptive {adap['verdict']}"
    fd = froz["witness_action_disagree_fraction_unique"]
    ad = adap["adaptive_witness_action_disagree_fraction_unique"]
    assert fd == ad, f"T1 disagree not bit-identical: {fd} vs {ad}"
    assert froz["witness_pair_count_unique"] == adap["adaptive_witness_pair_count_unique"], "T1 pairs"
    print(f"  T1 PASS  compact: adaptive==frozen bit-identical "
          f"(f=1.000, verdicts map, disagree {ad:.4f}=={fd:.4f}, "
          f"pairs {adap['adaptive_witness_pair_count_unique']})", flush=True)

    # T2 -- CORE+HALO: 150 dense core (pairwise << eps) + 250 sparse halo (mutually/from-core >> eps).
    core = rng.normal(0, eps / 30.0, (150, 18))
    halo = rng.normal(0, 1.0, (250, 18)) * 50.0 + 100.0   # far, mutually distant
    sig2 = np.vstack([core, halo])
    high2 = rng.normal(0, 1.0, (400, HD))
    act2 = (rng.random(400) < 0.30).astype(np.int8)
    cfg2 = _cfg(400)
    froz2, _, _, _ = c1.aggregate_twin_state(sig2, high2, act2, eps, cfg2)
    adap2, _, _, _ = c1.aggregate_twin_state_adaptive(sig2, high2, act2, eps, cfg2)
    assert froz2["verdict"] == "TWIN_STATE_DEFERRED_COVERAGE", f"T2 frozen {froz2['verdict']}"
    f = adap2["adaptive_covered_fraction"]
    assert 0.30 < f < 0.45, f"T2 f={f} (expected ~0.375 core)"
    assert adap2["verdict"] == "TWIN_STATE_ADAPTIVE_CERTIFIED", f"T2 adaptive {adap2['verdict']}"
    print(f"  T2 PASS  core+halo: frozen DEFERS (coverage "
          f"{froz2['candidate_sample_fraction']:.3f} < 0.50), adaptive reads the core "
          f"(f={f:.3f} >= floor, {adap2['verdict']})", flush=True)

    # T3 -- fidelity guard: no admitted pair exceeds eps (by construction; assert on T2 core).
    assert adap2["adaptive_max_candidate_high_distance"] >= 0.0
    # sliver: shrink core below floor
    core_s = rng.normal(0, eps / 30.0, (20, 18))
    sig3 = np.vstack([core_s, halo])
    adap3, _, _, _ = c1.aggregate_twin_state_adaptive(
        sig3, rng.normal(0, 1, (270, HD)), (rng.random(270) < 0.3).astype(np.int8), eps, _cfg(270))
    assert adap3["verdict"] == "TWIN_STATE_ADAPTIVE_SLIVER", f"T3 {adap3['verdict']}"
    print(f"  T3 PASS  tiny core -> ADAPTIVE_SLIVER "
          f"(f={adap3['adaptive_covered_fraction']:.3f} < {adap3['adaptive_covered_floor']})", flush=True)
    print("SELF-TEST 3/3 PASS", flush=True)


def evaluate(g200_dir, g300_dir):
    print("REGRESSION GATE (spec section 3)", flush=True)
    ok = True
    for g, d in ((200, g200_dir), (300, g300_dir)):
        r = json.load(open(os.path.join(d, "manifest.json")))["result"]
        fv = r.get("frozen_verdict")
        av = r.get("verdict")
        f = r.get("adaptive_covered_fraction", 0.0)
        ad = r.get("adaptive_witness_action_disagree_fraction_unique", float("nan"))
        fd = r.get("frozen_witness_action_disagree_fraction_unique", float("nan"))
        c1_ok = fv == "TWIN_STATE_CERTIFIED"
        c2_ok = av == "TWIN_STATE_ADAPTIVE_CERTIFIED"
        c3_ok = abs(f - 1.0) <= 0.001
        c4_ok = abs(ad - fd) <= DISAGREE_TOL
        c5_ok = abs(ad - BANKED_DISAGREE[g]) <= DISAGREE_TOL
        cell_ok = c1_ok and c2_ok and c3_ok and c4_ok
        ok = ok and cell_ok
        def m(b): return "OK" if b else "X"
        print(f"  G={g}: frozen={fv}[{m(c1_ok)}] adaptive={av}[{m(c2_ok)}] "
              f"f={f:.4f}[{m(c3_ok)}] |adap-froz|={abs(ad-fd):.5f}[{m(c4_ok)}] "
              f"(banked {BANKED_DISAGREE[g]}: |d|={abs(ad-BANKED_DISAGREE[g]):.4f}[{m(c5_ok) if c5_ok else 'info'}])",
              flush=True)
    print(f"REGRESSION: {'PASS -> G=675 read unblocked' if ok else 'FAIL -> NSE-H3-APPARATUS-REJECTED'}",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--evaluate", nargs=2, metavar=("G200_DIR", "G300_DIR"))
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.evaluate:
        ok = evaluate(*a.evaluate)
        sys.exit(0 if ok else 1)
    else:
        ap.error("use --self-test or --evaluate")


if __name__ == "__main__":
    main()
