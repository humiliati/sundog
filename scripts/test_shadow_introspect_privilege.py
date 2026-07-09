"""FROZEN test for H3-PL (HS1) — locks the reviewed spec BEFORE the run script exists.

Every threshold/seed here is a LITERAL copy of the FROZEN prereg (v2, commit 815ea529); the tests
assert the shared config hs1_privilege_config.py matches those literals (drift-lock) and that the
verdict logic satisfies the review's structural invariants — no dead zone, both kills reachable,
matched dimensionality, and the B2 null-gate exclusion. Runs as `python test_...py` (no pytest
needed) or under pytest. numpy-free, deterministic.
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import hs1_privilege_config as cfg  # noqa: E402


# ---- 1. constants match the frozen prereg literals (self-consistency drift-lock) ---- #
def test_constants_match_frozen_prereg():
    assert cfg.SIGMA_D == 9.0                       # §3/§7 pinned (tuning db1c52b2)
    assert (cfg.K_UNITS, cfg.H_DIM) == (64, 32)
    assert cfg.LAMBDA == 2.0
    assert (cfg.N_TRAIN, cfg.N_SPLIT) == (20000, 10000)
    assert cfg.K_COL == 128 and cfg.NYSTROEM_N == 100
    assert cfg.NYSTROEM_N < cfg.K_COL              # B4: no interpolation of the per-unit dims
    assert cfg.INJ_LEVELS == (0.10, 0.20)
    assert cfg.COUNTED_BATTERY == ("P1_ridge", "P4_nystroem", "P5_gbt")
    assert (cfg.M_ENC_LOCALIZED_MIN, cfg.M_POOL_LOCALIZED_MIN) == (-0.10, 0.15)
    assert cfg.M_ENC_ENCODER_DEEP_MAX == -0.25
    assert cfg.REC_MARGIN == 0.20 and cfg.D_ACC_GATE == 0.90
    assert cfg.K_NULL == 20 and cfg.NULL_PCTILE == 95
    assert cfg.BETAS == (0.3, 1.0, 3.0)
    assert (cfg.C0_MAX, cfg.C1_MIN) == (0.10, 0.50)
    assert cfg.DESAT_BAND == (0.75, 0.85)


# ---- 2. B2 lock: the recovery readout is NOT a null-gate metric ---- #
def test_b2_null_gate_excludes_recovery_readout():
    assert cfg.NULL_GATE_METRICS == ("linear_cka", "dacc_drift")
    assert "delta_c_decodability" not in cfg.NULL_GATE_METRICS  # would make kill (e) unfireable


# ---- 3. ACCESS partition: exhaustive, disjoint, matches §2 on named cases ---- #
def test_access_named_cases():
    # (a): encoder preserved (trained ~ raw), pool killed it (trained >> pooled)
    assert cfg.classify_access(0.50, 0.48, 0.13) == "a_localized_at_pool"
    # (b) KILL: encoder attenuated c per-unit BEFORE the pool (trained << raw)
    assert cfg.classify_access(0.50, 0.20, 0.13) == "b_encoder_deep"
    # (c): partial encoder attenuation (m_enc in [-0.25,-0.10))
    assert cfg.classify_access(0.50, 0.35, 0.13) == "c_partial_mixed"
    # (c): encoder fine but pool loses little (m_pool < 0.15)
    assert cfg.classify_access(0.20, 0.20, 0.13) == "c_partial_mixed"


def test_access_partition_exhaustive_and_disjoint():
    valid = {"a_localized_at_pool", "b_encoder_deep", "c_partial_mixed"}
    grid = [round(x, 2) for x in [i * 0.05 for i in range(0, 21)]]  # 0.00..1.00
    for raw, tr, pl in itertools.product(grid, grid, grid):
        out = cfg.classify_access(raw, tr, pl)
        assert out in valid                        # exactly one name, no dead zone


# ---- 4. kill fireability (the anti-foreordination lock — both kills must be reachable) ---- #
def test_access_kill_b_is_reachable():
    assert any(cfg.classify_access(raw, tr, 0.13) == "b_encoder_deep"
               for raw in (0.4, 0.5, 0.6) for tr in (0.1, 0.2))


def test_backaction_kill_e_is_reachable():
    # recovery clears, both disturbance metrics INSIDE the null -> recovery-for-free kill
    assert cfg.classify_backaction(0.45, 0.95, 0.13, False, False) == "e_recovery_without_disturbance"


# ---- 5. BACK-ACTION partition on named cases ---- #
def test_backaction_named_cases():
    # d-acc below gate -> named non-verdict regardless of recovery
    assert cfg.classify_backaction(0.99, 0.85, 0.13, True, True) == "objective_abandoned"
    # recovery below threshold -> no recovery
    assert cfg.classify_backaction(0.13 + 0.10, 0.95, 0.13, True, True) == "f_no_recovery"
    # recovery clears + a disturbance metric outside null -> priced
    assert cfg.classify_backaction(0.13 + 0.25, 0.95, 0.13, True, False) == "d_restructuring_priced"
    # recovery clears + both inside null -> free (kill e)
    assert cfg.classify_backaction(0.13 + 0.25, 0.95, 0.13, False, False) == \
        "e_recovery_without_disturbance"


# ---- 6. verdict-beta selection rule (B6) ---- #
def test_verdict_beta_is_smallest_over_gate():
    assert cfg.select_verdict_beta({0.3: 0.88, 1.0: 0.92, 3.0: 0.95}) == 1.0
    assert cfg.select_verdict_beta({0.3: 0.91, 1.0: 0.92, 3.0: 0.95}) == 0.3
    assert cfg.select_verdict_beta({0.3: 0.85, 1.0: 0.87, 3.0: 0.89}) is None


# ---- 7. seed ledger completeness (B5: k=20 distinct null seeds) ---- #
def test_seed_ledger():
    for k in ("body", "cv_split", "battery", "joint", "null", "tuning", "reserve"):
        assert k in cfg.SEEDS
    assert cfg.SEEDS["null"] == tuple(range(81235, 81255))
    assert len(set(cfg.SEEDS["null"])) == cfg.K_NULL == 20
    scalars = [cfg.SEEDS[k] for k in ("body", "cv_split", "battery", "joint", "tuning", "reserve")]
    assert len(set(scalars)) == len(scalars)       # no seed collisions across arms


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  [PASS] {t.__name__}")
    print(f"\n{len(tests)}/{len(tests)} frozen-test checks pass — H3-PL spec locked.")


if __name__ == "__main__":
    main()
