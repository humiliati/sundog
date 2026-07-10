"""FROZEN test for H3-PL v3 — locks the re-registered spec BEFORE the run script runs.

Literals below copy the FROZEN v3 prereg (H3_PRIVILEGE_LOCALIZATION_V3_PREREG.md); the tests assert
the shared config matches (drift-lock), the verdict logic satisfies the structural invariants (no
dead zone, both kills reachable, B2 exclusion), and — NEW in v3 — the ACHIEVABILITY asserts: gate
arithmetic in code plus the dry-run receipt's pinned values (the regression lock on the process fix
that voided v2). v2's test is preserved in git at 9bdfb255. Runs standalone or under pytest.
"""
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import hs1_privilege_config as cfg  # noqa: E402

DRYRUN_RECEIPT = os.path.join(os.path.dirname(__file__), "..", "results", "atlas", "hs1",
                              "v3_dryrun.json")


# ---- 1. constants match the frozen v3 prereg literals (drift-lock) ---- #
def test_constants_match_frozen_prereg():
    assert cfg.SIGMA_D == 9.0
    assert (cfg.K_UNITS, cfg.H_DIM) == (64, 32)
    assert cfg.LAMBDA == 2.0
    assert (cfg.N_TRAIN, cfg.N_SPLIT) == (20000, 10000)
    assert cfg.K_COL == 128 and cfg.NYSTROEM_N == 100
    assert cfg.NYSTROEM_N < cfg.K_COL
    assert cfg.INJ_TARGET_R2 == 0.10 and cfg.DETECT_FLOOR == 0.05        # D9b fixed-R2 policy
    assert cfg.P6_CONTROL_MIN == 0.30
    assert cfg.COUNTED_BATTERY == ("P1_ridge", "P4_nystroem", "P5_gbt", "P6_demod")  # D9a
    assert (cfg.M_ENC_LOCALIZED_MIN, cfg.M_POOL_LOCALIZED_MIN) == (-0.10, 0.15)
    assert cfg.M_ENC_ENCODER_DEEP_MAX == -0.25
    assert cfg.REC_MARGIN == 0.20
    assert cfg.D_ACC_GATE == 0.75                                        # D8 band floor
    assert cfg.K_NULL == 20 and cfg.NULL_PCTILE == 95
    assert cfg.BETAS == (0.3, 1.0, 3.0)
    assert (cfg.C0_MAX, cfg.C1_MIN) == (0.10, 0.50)
    assert cfg.DESAT_BAND == (0.75, 0.85)


# ---- 2. ACHIEVABILITY asserts (the v2-voiding defect class, now regression-locked) ---- #
def test_gate_arithmetic_reachable():
    # D8: the gate must sit inside the substrate's own band — v2 had 0.90 > 0.85 and voided.
    assert cfg.D_ACC_GATE == cfg.DESAT_BAND[0] <= cfg.DESAT_BAND[1]


def test_dryrun_receipt_freeze_clear():
    with open(DRYRUN_RECEIPT) as f:
        d = json.load(f)
    assert d["verdict"] == "FREEZE_CLEAR"
    assert d["a1_body"]["in_band"] and d["a1_body"]["C0_pass"] and d["a1_body"]["C1_pass"]
    assert d["a2_p6_raw_control"]["cv_r2"] >= cfg.P6_CONTROL_MIN         # 0.5322 pinned
    assert all(v["converged"] for v in d["a3_injection"].values())
    assert d["a4_beta0_gate"]["d_acc"] >= cfg.D_ACC_GATE


# ---- 3. B2 lock (carried): recovery readout is NOT a null-gate metric ---- #
def test_b2_null_gate_excludes_recovery_readout():
    assert cfg.NULL_GATE_METRICS == ("linear_cka", "dacc_drift")
    assert "delta_c_decodability" not in cfg.NULL_GATE_METRICS


# ---- 4. ACCESS partition: named cases + exhaustive/disjoint ---- #
def test_access_named_cases():
    assert cfg.classify_access(0.50, 0.48, 0.13) == "a_localized_at_pool"
    assert cfg.classify_access(0.50, 0.20, 0.13) == "b_encoder_deep"
    assert cfg.classify_access(0.50, 0.35, 0.13) == "c_partial_mixed"
    assert cfg.classify_access(0.20, 0.20, 0.13) == "c_partial_mixed"


def test_access_partition_exhaustive_and_disjoint():
    valid = {"a_localized_at_pool", "b_encoder_deep", "c_partial_mixed"}
    grid = [round(x, 2) for x in [i * 0.05 for i in range(0, 21)]]
    for raw, tr, pl in itertools.product(grid, grid, grid):
        assert cfg.classify_access(raw, tr, pl) in valid


# ---- 5. kill fireability (anti-foreordination locks) ---- #
def test_access_kill_b_is_reachable():
    assert any(cfg.classify_access(raw, tr, 0.13) == "b_encoder_deep"
               for raw in (0.4, 0.5, 0.6) for tr in (0.1, 0.2))


def test_backaction_kill_e_is_reachable():
    assert cfg.classify_backaction(0.45, 0.80, 0.13, False, False) == "e_recovery_without_disturbance"


# ---- 6. BACK-ACTION partition on named cases (v3 gate = 0.75) ---- #
def test_backaction_named_cases():
    # genuine competence collapse -> named non-verdict
    assert cfg.classify_backaction(0.99, 0.70, 0.13, True, True) == "objective_abandoned"
    # v2's void case is now IN-GATE: d-acc 0.79 evaluates (this exact value voided v2)
    assert cfg.classify_backaction(0.41, 0.79, 0.049, True, False) == "d_restructuring_priced"
    assert cfg.classify_backaction(0.13 + 0.10, 0.80, 0.13, True, True) == "f_no_recovery"
    assert cfg.classify_backaction(0.13 + 0.25, 0.80, 0.13, False, False) == \
        "e_recovery_without_disturbance"


# ---- 7. verdict-beta selection (gate 0.75) ---- #
def test_verdict_beta_is_smallest_over_gate():
    assert cfg.select_verdict_beta({0.3: 0.72, 1.0: 0.78, 3.0: 0.81}) == 1.0
    assert cfg.select_verdict_beta({0.3: 0.79, 1.0: 0.78, 3.0: 0.81}) == 0.3
    assert cfg.select_verdict_beta({0.3: 0.70, 1.0: 0.71, 3.0: 0.74}) is None


# ---- 8. seed ledger: fresh 52235-family, no overlap with v2's 51235-family ---- #
def test_seed_ledger():
    for k in ("body", "cv_split", "battery", "joint", "null", "dryrun", "reserve"):
        assert k in cfg.SEEDS
    assert cfg.SEEDS["null"] == tuple(range(82235, 82255))
    assert len(set(cfg.SEEDS["null"])) == cfg.K_NULL == 20
    scalars = [cfg.SEEDS[k] for k in ("body", "cv_split", "battery", "joint", "dryrun", "reserve")]
    assert len(set(scalars)) == len(scalars)
    v2_family = {51235, 61235, 1789, 71235, 86235, 91235} | set(range(81235, 81255))
    v3_all = set(scalars) | set(cfg.SEEDS["null"])
    assert v3_all.isdisjoint(v2_family)            # contamination firewall: every verdict seed fresh


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} frozen-test checks pass — H3-PL v3 spec locked.")


if __name__ == "__main__":
    main()
