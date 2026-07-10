#!/usr/bin/env python3
"""NSE-H3 mid-energy sub-regime restriction (frozen spec:
docs/chatv2/NSE_H3_MIDE_SUBREGIME_SPEC.md). Post-processing on banked samples.npz:
restrict to the CENTRAL energy tercile (band fixed by rule) and run the UNCHANGED
frozen twin-state certificate on the self-contained sub-attractor. Non-promotional.

  --self-test              synthetic restriction check (seconds)
  --regress G200 G300      section-3 anchor regression (restriction must reproduce CERTIFIED)
  --run CELL_DIR           the mid-E read on a cell's samples.npz + branch
"""
import argparse, json, math, os, sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pde_c1_kolmogorov_cell as c1


def _cfg(n, preset):
    return c1.RunConfig(
        preset=preset, grid_size=32, n_modes=16, k_signature=3, forcing_wavenumber=2,
        grashof=200.0, forcing_amplitude=1.0, viscosity=0.0707, dt=0.01, burnin_steps=0,
        sample_count=n, sample_interval_steps=50, lookahead_steps=500, n_min=30,
        delta_action=0.10, s_pos=0.50, delta_proxy_min=0.01, e_max_burnin_fraction=1.0,
        random_seed=0, integrator="semi-implicit", signature_dimension=18,
        action_tiebreak="damp", adjudicator="twin-state", k_neighbors=30, delta_incompat=0.01,
        twin_k_neighbors=50, twin_delta_high_fraction=0.05, twin_high_norm_floor=1e-6,
        twin_min_witness_fraction=0.01, twin_min_unique_pairs=100,
        objective="portable-quantile", objective_quantile=0.70,
        calibration_sample_count=50000, calibration_gap_steps=5000,
    )


def restrict(sig, band):
    """Mask for the central energy tercile (band='mid') or all (band='full')."""
    E = np.sum(sig.astype(np.float64) ** 2, axis=1)
    if band == "full":
        return np.ones(len(sig), dtype=bool), E
    q1, q2 = np.quantile(E, [1.0 / 3.0, 2.0 / 3.0])
    return (E > q1) & (E <= q2), E


def certify(cell_dir, band):
    z = np.load(os.path.join(cell_dir, "samples.npz"))
    preset = str(z["preset"])
    sig, high, act, eps = z["signatures"], z["high_modes"], z["actions"], float(z["epsilon_k"])
    mask, E = restrict(sig, band)
    sub_sig, sub_high, sub_act = sig[mask], high[mask], act[mask]
    cfg = _cfg(int(mask.sum()), preset)
    res, _, _, _ = c1.aggregate_twin_state(sub_sig, sub_high, sub_act, eps, cfg)
    res["_band"] = band
    res["_n_restricted"] = int(mask.sum())
    res["_n_full"] = int(len(sig))
    res["_preset"] = preset
    return res


def branch_of(r):
    cov = r.get("candidate_sample_fraction", 0.0)
    v = r.get("verdict")
    paired = r.get("paired_fiber_verdict")
    if v == "TWIN_STATE_DEFERRED_COVERAGE" or cov < 0.50:
        return "NSE-H3-MIDE-UNDERCOVERED"
    if v == "TWIN_STATE_CERTIFIED" and paired == "PAIRED_FIBER_CONSTANCY_POSITIVE":
        return "NSE-H3-FORCING-GENERAL-MIDE"
    if v == "TWIN_STATE_CERTIFIED":
        return "NSE-H3-GRASHOF-LOCAL-MIDE"
    return f"NSE-H3-MIDE-{v}"


def self_test():
    print("SELF-TEST (restriction machinery; synthetic)", flush=True)
    rng = np.random.default_rng(0)
    # three energy bands; central tercile must select the mid one.
    lo = rng.normal(0, 0.006, (300, 18)); lo[:, 0] += math.sqrt(0.3)
    mid = rng.normal(0, 0.006, (300, 18)); mid[:, 1] += math.sqrt(0.7)
    hi = rng.normal(0, 0.006, (300, 18)); hi[:, 2] += math.sqrt(1.3)
    sig = np.vstack([lo, mid, hi])
    mask, E = restrict(sig, "mid")
    # the mid tercile should be dominated by the E~0.7 cluster
    picked_mid = mask[300:600].mean()
    assert picked_mid > 0.9, f"T1 central tercile mostly the mid cluster: {picked_mid}"
    assert abs(mask.mean() - 1 / 3) < 0.02, f"T1 tercile ~1/3: {mask.mean()}"
    print(f"  T1 PASS  central tercile selects mid-E cluster ({picked_mid:.2f}) and ~1/3 of samples "
          f"({mask.mean():.3f})", flush=True)
    full, _ = restrict(sig, "full")
    assert full.all(), "T2 full = all"
    print("  T2 PASS  band='full' selects all", flush=True)
    print("SELF-TEST 2/2 PASS", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--regress", nargs=2, metavar=("G200_DIR", "G300_DIR"))
    ap.add_argument("--run", metavar="CELL_DIR")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return
    if a.regress:
        print("REGRESSION (section 3): restriction must reproduce banked CERTIFIED", flush=True)
        ok = True
        for g, d in zip((200, 300), a.regress):
            r = certify(d, "full")
            rm = certify(d, "mid")
            c_full = r["verdict"] == "TWIN_STATE_CERTIFIED"
            c_mid = rm["verdict"] == "TWIN_STATE_CERTIFIED"
            ok = ok and c_full and c_mid
            print(f"  G={g}: full={r['verdict']}[{'OK' if c_full else 'X'}] "
                  f"mid={rm['verdict']} cov={rm['candidate_sample_fraction']:.3f} "
                  f"paired={rm['paired_fiber_verdict']}[{'OK' if c_mid else 'X'}] "
                  f"(n {rm['_n_restricted']}/{rm['_n_full']})", flush=True)
        print(f"REGRESSION: {'PASS' if ok else 'FAIL -- restriction machinery broken'}", flush=True)
        sys.exit(0 if ok else 1)
    if a.run:
        r = certify(a.run, "mid")
        branch = branch_of(r)
        json.dump({"result": r, "branch": branch},
                  open(os.path.join(a.run, "mide_subregime_manifest.json"), "w"), indent=1, default=str)
        print("NSE-H3 mid-energy sub-regime read", flush=True)
        print(f"  n_mid={r['_n_restricted']}/{r['_n_full']}  eps_K={r['epsilon_k_radius_threshold']:.5f}", flush=True)
        print(f"  candidate coverage={r['candidate_sample_fraction']:.4f} (gate >= 0.50)", flush=True)
        print(f"  verdict={r['verdict']}  witness_pairs={r.get('witness_pair_count_unique')}  "
              f"paired={r['paired_fiber_verdict']}  disagree={r.get('witness_action_disagree_fraction_unique')}",
              flush=True)
        print(f"BRANCH: {branch}", flush=True)
        return
    ap.error("use --self-test, --regress, or --run")


if __name__ == "__main__":
    main()
