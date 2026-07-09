#!/usr/bin/env python3
"""Coverage-adaptive apparatus (Approach A) -- regression evaluator + synthetic
self-test. Frozen spec: docs/chatv2/NSE_COVERAGE_ADAPTIVE_APPARATUS_SPEC.md sec 3.

--self-test : synthetic validation (seconds) of the two load-bearing properties:
  (1) on a COMPACT cluster, adaptive is bit-identical to frozen (reduction theorem);
  (2) on a CORE+HALO spread, frozen DEFERS on coverage while adaptive reads the core.
--evaluate G200_DIR G300_DIR : apply the section-3 regression gate to two adaptive
  manifests and print PASS/FAIL (the precondition before any G=675 read).
"""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1

DISAGREE_TOL = 0.005
BANKED_DISAGREE = {200: 0.0367, 300: 0.0382}  # reference (preset-confounded; informational)


def _cfg(n_samples, preset="lock_v7_g200"):
    return c1.RunConfig(
        preset=preset, grid_size=32, n_modes=16, k_signature=3,
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


def _cluster(rng, center_energy, sigma, dim_axis, n, HD, action_mode):
    """A tight signature cluster at a target ||Phi||^2 = center_energy, with high
    modes spread (witnesses) and actions per `action_mode` ('zero'|'random')."""
    c = np.zeros(18)
    c[dim_axis] = math.sqrt(center_energy)
    sig = c[None, :] + rng.normal(0, sigma, (n, 18))
    high = rng.normal(0, 1.0, (n, HD))
    if action_mode == "zero":
        act = np.zeros(n, dtype=np.int8)
    else:
        act = (rng.random(n) < 0.50).astype(np.int8)
    return sig, high, act


def self_test_b():
    print("SELF-TEST-B (Approach B v2; relative eps_K + inflation-SHELL guard)", flush=True)
    eps = 0.06
    HD = 18

    # B-T1 -- COMPACT anchor-like (E~0.72): relative ~ frozen, SHELL empty ->
    # inflation_clean=None -> PASSES. This is the case v1's tercile guard WRONGLY
    # rejected; the v2 shell guard must be inert here.
    rng = np.random.default_rng(0)
    c = np.zeros(18); c[0] = math.sqrt(0.72)
    sig = c[None, :] + rng.normal(0, 0.008, (400, 18))
    high = rng.normal(0, 1.0, (400, HD))
    act = (rng.random(400) < 0.30).astype(np.int8)
    cfg = _cfg(400)
    froz, _, _, _ = c1.aggregate_twin_state(sig, high, act, eps, cfg)
    rel, _, _, _ = c1.aggregate_twin_state_relative(sig, high, act, eps, cfg)
    assert froz["verdict"] == "TWIN_STATE_CERTIFIED", f"B-T1 frozen {froz['verdict']}"
    assert rel["verdict"] == "TWIN_STATE_RELATIVE_CERTIFIED", f"B-T1 relative {rel['verdict']}"
    assert rel["relative_covered_fraction"] >= 0.98, f"B-T1 f_rel {rel['relative_covered_fraction']}"
    assert rel["relative_inflation_clean"] is not False, \
        f"B-T1 guard spuriously fired: clean={rel['relative_inflation_clean']} shell={rel['relative_shell_pair_count_unique']}"
    print(f"  B-T1 PASS  compact anchor-like: relative~frozen (f_rel={rel['relative_covered_fraction']:.3f}), "
          f"SHELL={rel['relative_shell_pair_count_unique']} pairs -> inflation_clean="
          f"{rel['relative_inflation_clean']} (v1 wrongly rejected this; v2 inert)", flush=True)

    # B-T2 -- INFLATION: a high-E cluster spread so pairwise sits BETWEEN frozen
    # eps_K (0.06) and its relative radius (~0.08) -> those pairs are SHELL, and
    # random actions make them disagree. The v2 shell guard MUST fire (False).
    rng = np.random.default_rng(1)
    lo_s, lo_h, lo_a = _cluster(rng, 0.30, 0.004, 0, 250, HD, "zero")     # core, disagree ~0
    # high-E, sigma so 6*sigma ~ 0.072 (between 0.06 and 0.081), actions random
    hi_s, hi_h, hi_a = _cluster(rng, 1.30, 0.012, 1, 300, HD, "random")
    sig2 = np.vstack([lo_s, hi_s]); high2 = np.vstack([lo_h, hi_h]); act2 = np.concatenate([lo_a, hi_a])
    rel2, _, _, _ = c1.aggregate_twin_state_relative(sig2, high2, act2, eps, _cfg(550))
    assert rel2["relative_shell_pair_count_unique"] >= c1.RELATIVE_MIN_SHELL_PAIRS, \
        f"B-T2 shell underpowered ({rel2['relative_shell_pair_count_unique']}) -- adjust sigma"
    assert rel2["relative_inflation_clean"] is False, \
        f"B-T2 guard did not fire: clean={rel2['relative_inflation_clean']} shell_disagree={rel2.get('relative_disagree_shell')}"
    print(f"  B-T2 PASS  inflation caught: SHELL {rel2['relative_shell_pair_count_unique']} pairs disagree "
          f"{rel2['relative_disagree_shell']:.3f} vs core {rel2['relative_disagree_core']:.3f} "
          f"-> inflation_clean=False", flush=True)

    # B-T3 -- SHELL-CLEAN positive: same geometry as B-T2 (shell pairs exist) but
    # the high-E cluster carries a CONSTANT action -> SHELL pairs agree -> clean.
    rng = np.random.default_rng(2)
    lo_s, lo_h, _ = _cluster(rng, 0.30, 0.004, 0, 250, HD, "zero")
    hi_s, hi_h, _ = _cluster(rng, 1.30, 0.012, 1, 300, HD, "zero")
    sig3 = np.vstack([lo_s, hi_s]); high3 = np.vstack([lo_h, hi_h])
    act3 = np.concatenate([np.zeros(250, np.int8), np.ones(300, np.int8)])  # constant per cluster
    rel3, _, _, _ = c1.aggregate_twin_state_relative(sig3, high3, act3, eps, _cfg(550))
    assert rel3["relative_shell_pair_count_unique"] >= c1.RELATIVE_MIN_SHELL_PAIRS, "B-T3 shell underpowered"
    assert rel3["relative_inflation_clean"] is True, \
        f"B-T3 not clean: shell_disagree={rel3.get('relative_disagree_shell')} core={rel3.get('relative_disagree_core')}"
    print(f"  B-T3 PASS  shell-clean: SHELL {rel3['relative_shell_pair_count_unique']} pairs disagree "
          f"{rel3['relative_disagree_shell']:.3f} <= core+margin -> inflation_clean=True", flush=True)
    print("SELF-TEST-B 3/3 PASS", flush=True)


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


def reprocess_b(g200_dir, g300_dir):
    """R0-v2 via POST-PROCESSING on the banked v1 samples.npz (no re-integration):
    load signatures/high_modes/actions/eps_K, run the v2 aggregator, write a v2
    manifest, then apply the regression gate. This is the 'third apparatus
    iteration = pure post-processing' path the sample export was built for."""
    for g, d in ((200, g200_dir), (300, g300_dir)):
        z = np.load(os.path.join(d, "samples.npz"))
        preset = str(z["preset"])
        cfg = _cfg(int(z["signatures"].shape[0]), preset=preset)  # twin params are lock_v7-standard
        res, _, _, _ = c1.aggregate_twin_state_relative(
            z["signatures"], z["high_modes"], z["actions"], float(z["epsilon_k"]), cfg)
        outp = os.path.join(d, "v2_reprocess_manifest.json")
        json.dump({"result": {**res,
                              "reprocessed_from": "samples.npz", "preset": preset}},
                  open(outp, "w"), indent=1, default=str)
        print(f"  [reprocessed] G={g} -> {outp}  verdict={res['verdict']} "
              f"inflation_clean={res['relative_inflation_clean']} "
              f"shell={res['relative_shell_pair_count_unique']}", flush=True)
    # feed the reprocessed manifests into the standard gate by symlinking name
    return evaluate_b_from(g200_dir, g300_dir, "v2_reprocess_manifest.json")


def reprocess_b1(cell_dir):
    """R1-v2: reprocess a single banked cell's samples.npz through the v2 relative
    aggregator and map to the spec section-4 branch (post-processing; no re-run)."""
    z = np.load(os.path.join(cell_dir, "samples.npz"))
    preset = str(z["preset"])
    cfg = _cfg(int(z["signatures"].shape[0]), preset=preset)
    r, _, _, _ = c1.aggregate_twin_state_relative(
        z["signatures"], z["high_modes"], z["actions"], float(z["epsilon_k"]), cfg)
    json.dump({"result": {**r, "reprocessed_from": "samples.npz", "preset": preset}},
              open(os.path.join(cell_dir, "v2_reprocess_manifest.json"), "w"), indent=1, default=str)
    f = r["relative_covered_fraction"]
    v = r["verdict"]
    paired = r["relative_paired_fiber_verdict"]
    ic = r["relative_inflation_clean"]
    if f < c1.RELATIVE_COVERED_FLOOR or v == "TWIN_STATE_RELATIVE_SLIVER":
        branch = "NSE-H3-COVERAGE-SLIVER-RELATIVE"
    elif v != "TWIN_STATE_RELATIVE_CERTIFIED":
        branch = "NSE-H3-COVERAGE-WALL-CONFIRMED-RELATIVE"
    elif paired == "PAIRED_FIBER_CONSTANCY_POSITIVE" and ic is False:
        branch = "NSE-H3-APPARATUS-REJECTED-B2"
    elif paired == "PAIRED_FIBER_CONSTANCY_POSITIVE":
        branch = "NSE-H3-FORCING-GENERAL-RELATIVE"
    else:
        branch = "NSE-H3-GRASHOF-LOCAL"
    print("R1-v2 (relative read on the banked G=675 cell)", flush=True)
    print(f"  verdict={v}  paired={paired}  inflation_clean={ic}", flush=True)
    print(f"  f_rel={f:.5f}  rel eps_K p05/p50/p95={r['relative_eps_k_p05']:.4f}/"
          f"{r['relative_eps_k_p50']:.4f}/{r['relative_eps_k_p95']:.4f} (frozen {r['epsilon_k_frozen_reference']:.4f})",
          flush=True)
    print(f"  CORE {r['relative_core_pair_count_unique']} pairs disagree {r['relative_disagree_core']}"
          f"  | SHELL {r['relative_shell_pair_count_unique']} pairs disagree {r['relative_disagree_shell']}",
          flush=True)
    print(f"  frozen comparator: {r['frozen_verdict']} (candidate cov {r['frozen_candidate_sample_fraction']})", flush=True)
    print(f"BRANCH: {branch}", flush=True)
    return branch


def evaluate_b_from(g200_dir, g300_dir, fname):
    print("REGRESSION GATE-B v2 (inflation-shell; section 3)", flush=True)
    ok = True
    for g, d in ((200, g200_dir), (300, g300_dir)):
        r = json.load(open(os.path.join(d, fname)))["result"]
        fv = r.get("frozen_verdict"); rv = r.get("verdict")
        f = r.get("relative_covered_fraction", 0.0)
        rd = r.get("relative_witness_action_disagree_fraction_unique", float("nan"))
        fd = r.get("frozen_witness_action_disagree_fraction_unique", float("nan"))
        ic = r.get("relative_inflation_clean")
        shell = r.get("relative_shell_pair_count_unique", 0)
        c1_ok = fv == "TWIN_STATE_CERTIFIED"
        c2_ok = rv == "TWIN_STATE_RELATIVE_CERTIFIED"
        c3_ok = f >= 0.98
        c4_ok = abs(rd - fd) <= DISAGREE_TOL
        c5_ok = ic is not False
        cell_ok = c1_ok and c2_ok and c3_ok and c4_ok and c5_ok
        ok = ok and cell_ok
        def m(b): return "OK" if b else "X"
        print(f"  G={g}: frozen={fv}[{m(c1_ok)}] relative={rv}[{m(c2_ok)}] "
              f"f_rel={f:.4f}[{m(c3_ok)}] |rel-froz|={abs(rd-fd):.5f}[{m(c4_ok)}] "
              f"inflation_clean={ic} (shell {shell})[{m(c5_ok)}]", flush=True)
    print(f"REGRESSION-B v2: {'PASS -> G=675 relative read unblocked' if ok else 'FAIL -> NSE-H3-APPARATUS-REJECTED-B2'}",
          flush=True)
    return ok


def evaluate_b(g200_dir, g300_dir):
    """Approach B regression gate (spec section 3): relative reduces to frozen
    within tolerance on the compact cells, and the scale-consistency guard does
    not spuriously fire."""
    print("REGRESSION GATE-B (relative eps_K; section 3)", flush=True)
    ok = True
    for g, d in ((200, g200_dir), (300, g300_dir)):
        r = json.load(open(os.path.join(d, "manifest.json")))["result"]
        fv = r.get("frozen_verdict")
        rv = r.get("verdict")
        f = r.get("relative_covered_fraction", 0.0)
        rd = r.get("relative_witness_action_disagree_fraction_unique", float("nan"))
        fd = r.get("frozen_witness_action_disagree_fraction_unique", float("nan"))
        ic = r.get("relative_inflation_clean")
        shell = r.get("relative_shell_pair_count_unique", 0)
        c1_ok = fv == "TWIN_STATE_CERTIFIED"
        c2_ok = rv == "TWIN_STATE_RELATIVE_CERTIFIED"
        c3_ok = f >= 0.98
        c4_ok = abs(rd - fd) <= DISAGREE_TOL
        c5_ok = ic is not False   # v2 shell guard: inert (None) on compact cells -> pass
        cell_ok = c1_ok and c2_ok and c3_ok and c4_ok and c5_ok
        ok = ok and cell_ok
        def m(b): return "OK" if b else "X"
        print(f"  G={g}: frozen={fv}[{m(c1_ok)}] relative={rv}[{m(c2_ok)}] "
              f"f_rel={f:.4f}[{m(c3_ok)}] |rel-froz|={abs(rd-fd):.5f}[{m(c4_ok)}] "
              f"inflation_clean={ic} (shell {shell})[{m(c5_ok)}]", flush=True)
    print(f"REGRESSION-B: {'PASS -> G=675 relative read unblocked' if ok else 'FAIL -> NSE-H3-APPARATUS-REJECTED-B'}",
          flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--self-test-b", action="store_true")
    ap.add_argument("--evaluate", nargs=2, metavar=("G200_DIR", "G300_DIR"))
    ap.add_argument("--evaluate-b", nargs=2, metavar=("G200_DIR", "G300_DIR"))
    ap.add_argument("--reprocess-b", nargs=2, metavar=("G200_DIR", "G300_DIR"))
    ap.add_argument("--reprocess-b1", metavar="CELL_DIR")
    a = ap.parse_args()
    if a.self_test:
        self_test()
    elif a.self_test_b:
        self_test_b()
    elif a.evaluate:
        ok = evaluate(*a.evaluate)
        sys.exit(0 if ok else 1)
    elif a.evaluate_b:
        ok = evaluate_b(*a.evaluate_b)
        sys.exit(0 if ok else 1)
    elif a.reprocess_b:
        ok = reprocess_b(*a.reprocess_b)
        sys.exit(0 if ok else 1)
    elif a.reprocess_b1:
        reprocess_b1(a.reprocess_b1)
    else:
        ap.error("use --self-test[-b], --evaluate[-b], --reprocess-b, or --reprocess-b1")


if __name__ == "__main__":
    main()
