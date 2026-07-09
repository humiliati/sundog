"""H3-PL (HS1) FROZEN spec — single source of truth for the run script + frozen test.

Pure constants + verdict logic ONLY. No torch, no sklearn, no data, no compute: importing this
module runs no experiment. Every number here is pinned by the FROZEN prereg
docs/atlas/H3_PRIVILEGE_LOCALIZATION_PREREG.md (v2, commit 815ea529) and must not change; the frozen
test test_shadow_introspect_privilege.py re-asserts each against the doc's literals. The run script
shadow_introspect_privilege.py (to be written) imports the classifiers below so the verdict is
computed by this frozen, tested code — the run cannot fudge it.
"""

# ---- substrate (§3, §7) ---- #
SIGMA_D = 9.0            # PINNED de-saturation (tuning db1c52b2: pooled d-acc 0.8004)
K_UNITS = 64
H_DIM = 32
LAMBDA = 2.0
N_TRAIN = 20000
N_SPLIT = 10000         # once-touched verdict split, scored once per ceiling (§4)

# ---- battery, matched dimensionality (§4; B1/B4/B7) ---- #
K_COL = 128             # common PCA column budget for EVERY ceiling arm (no arm wins on dim)
NYSTROEM_N = 100        # < K_COL (re-picked from the 2000 that interpolated 2048 per-unit dims)
INJ_LEVELS = (0.10, 0.20)
COUNTED_BATTERY = ("P1_ridge", "P4_nystroem", "P5_gbt")   # + KSG-MI confirmatory (no verdict weight)

# ---- ACCESS thresholds (§2; B1) — margins m_enc = trained-raw, m_pool = trained-pooled ---- #
M_ENC_LOCALIZED_MIN = -0.10     # (a): encoder preserved the per-unit c (trained ~ raw)
M_POOL_LOCALIZED_MIN = 0.15     # (a): the pool is where c dies (trained >> pooled)
M_ENC_ENCODER_DEEP_MAX = -0.25  # (b) KILL: encoder attenuated c per-unit before any pool

# ---- BACK-ACTION thresholds (§5; B2/B5/B6) ---- #
REC_MARGIN = 0.20               # rec_joint >= ceil_pooled + REC_MARGIN
D_ACC_GATE = 0.90               # below -> OBJECTIVE-ABANDONED (named non-verdict)
K_NULL = 20                     # seed-retrain null size (B5: raised from 10)
NULL_PCTILE = 95
BETAS = (0.3, 1.0, 3.0)         # verdict beta = smallest with d-acc >= gate (B6)
# B2 LOCK: the null-gate disturbance metrics are exactly these two. Delta-pooled-c-decodability is
# the recovery READOUT and is deliberately NOT here (including it made kill (e) unfireable).
NULL_GATE_METRICS = ("linear_cka", "dacc_drift")

# ---- gates (§3) ---- #
C0_MAX = 0.10           # raw mean-of-u c-R2 <= C0_MAX (raw averaging washes c); tuning value 0.0
C1_MIN = 0.50           # single raw unit c-R2 >= C1_MIN (c present per-unit); tuning value 0.999
DESAT_BAND = (0.75, 0.85)   # run body pooled d-acc must land here (else substrate-regen abort)

# ---- seeds ledger (§8) ---- #
SEEDS = {
    "body": 51235, "cv_split": 61235, "battery": 1789, "joint": 71235,
    "null": tuple(81235 + i for i in range(K_NULL)),   # 81235..81254
    "tuning": 86235, "reserve": 91235,
}


def classify_access(ceil_raw: float, ceil_trained: float, ceil_pooled: float) -> str:
    """§2 partition (exhaustive, disjoint). Contrast is trained-vs-RAW per-unit (encoder effect),
    not per-unit-vs-pooled (which is ~ data-processing — the B1 defect)."""
    m_enc = ceil_trained - ceil_raw
    m_pool = ceil_trained - ceil_pooled
    if m_enc < M_ENC_ENCODER_DEEP_MAX:
        return "b_encoder_deep"                     # KILL (informative null)
    if m_enc >= M_ENC_LOCALIZED_MIN and m_pool >= M_POOL_LOCALIZED_MIN:
        return "a_localized_at_pool"
    return "c_partial_mixed"


def classify_backaction(rec_joint: float, d_acc_joint: float, ceil_pooled: float,
                        cka_outside_null: bool, dacc_drift_outside_null: bool) -> str:
    """§2 back-action partition. Disturbance = {CKA, d-acc drift} only (B2); the recovery readout
    (Delta pooled c-decodability) is NOT a gate metric."""
    if d_acc_joint < D_ACC_GATE:
        return "objective_abandoned"                # named non-verdict
    if rec_joint < ceil_pooled + REC_MARGIN:
        return "f_no_recovery"
    if cka_outside_null or dacc_drift_outside_null:
        return "d_restructuring_priced"
    return "e_recovery_without_disturbance"          # KILL (recovery for free)


def select_verdict_beta(dacc_by_beta: dict) -> float | None:
    """§5 B6: verdict beta = the smallest beta in BETAS whose joint d-acc clears the gate."""
    for b in sorted(dacc_by_beta):
        if dacc_by_beta[b] >= D_ACC_GATE:
            return b
    return None
