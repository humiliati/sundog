"""H3-PL FROZEN spec v3 — single source of truth for the run script + frozen test.

Pure constants + verdict logic ONLY (no torch/sklearn/data; importing runs no experiment). Every
number is pinned by the FROZEN v3 prereg docs/atlas/H3_PRIVILEGE_LOCALIZATION_V3_PREREG.md and must
not change; the frozen test re-asserts each. v2 constants remain in git at 9bdfb255 (v2 spec
815ea529, voided by run 2ace4df8). v3 deltas: back-action gate = the de-saturation BAND FLOOR (D8);
P6 demodulating probe added to the counted battery (D9a); injection = fixed-R2 TARGET (D9b); all
verdict seeds fresh (52235-family, contamination firewall).
"""

# ---- substrate (v3 §5; SIGMA_D from tuning db1c52b2, carried) ---- #
SIGMA_D = 9.0
K_UNITS = 64
H_DIM = 32
LAMBDA = 2.0
N_TRAIN = 20000
N_SPLIT = 10000         # once-touched verdict split, scored once per ceiling arm

# ---- battery, matched dimensionality (v3 §4) ---- #
K_COL = 128             # common PCA column budget for the linear/kernel members
NYSTROEM_N = 100        # < K_COL
INJ_TARGET_R2 = 0.10    # D9b: injection calibrated per arm to this ridge-CV target (bisection)
INJ_OK_BAND = (0.05, 0.20)   # calibration counts as converged if achieved R2 lands here (x2 slack hi)
DETECT_FLOOR = 0.05     # a member is live on an arm if it reads >= this on the calibrated injection
P6_CONTROL_MIN = 0.30   # P6 liveness = native raw-arm positive control (dry-run: 0.5322)
COUNTED_BATTERY = ("P1_ridge", "P4_nystroem", "P5_gbt", "P6_demod")  # + KSG-MI confirmatory

# ---- ACCESS thresholds (v3 §3; UNCHANGED from v2 — untainted by run-1, see prereg §1) ---- #
M_ENC_LOCALIZED_MIN = -0.10     # (a): encoder preserved the per-unit c (trained ~ raw)
M_POOL_LOCALIZED_MIN = 0.15     # (a): the pool is where c dies (trained >> pooled)
M_ENC_ENCODER_DEEP_MAX = -0.25  # (b) KILL: encoder attenuated c per-unit before any pool

# ---- BACK-ACTION thresholds (v3 §3; D8 gate = band floor) ---- #
REC_MARGIN = 0.20               # rec_joint >= ceil_pooled + REC_MARGIN (unchanged, pre-run-1)
DESAT_BAND = (0.75, 0.85)
D_ACC_GATE = DESAT_BAND[0]      # 0.75 — "still a d-body by the substrate's own standard" (D8)
K_NULL = 20
NULL_PCTILE = 95
BETAS = (0.3, 1.0, 3.0)         # verdict beta = smallest with d-acc >= gate
# B2 LOCK (carried from v2): disturbance metrics exactly these two; the recovery readout
# (Delta pooled c-decodability) is deliberately NOT a gate metric.
NULL_GATE_METRICS = ("linear_cka", "dacc_drift")

# ---- gates (v3 §5) ---- #
C0_MAX = 0.10
C1_MIN = 0.50

# ---- seeds ledger (v3 §6 — ALL fresh vs v2's 51235-family) ---- #
SEEDS = {
    "body": 52235, "cv_split": 62235, "battery": 2789, "joint": 72235,
    "null": tuple(82235 + i for i in range(K_NULL)),   # 82235..82254
    "dryrun": 87235, "reserve": 92235,
}


def classify_access(ceil_raw: float, ceil_trained: float, ceil_pooled: float) -> str:
    """v3 §3 partition (exhaustive, disjoint). Contrast is trained-vs-RAW per-unit (encoder
    effect), not per-unit-vs-pooled (the B1 data-processing defect)."""
    m_enc = ceil_trained - ceil_raw
    m_pool = ceil_trained - ceil_pooled
    if m_enc < M_ENC_ENCODER_DEEP_MAX:
        return "b_encoder_deep"                     # KILL (informative null)
    if m_enc >= M_ENC_LOCALIZED_MIN and m_pool >= M_POOL_LOCALIZED_MIN:
        return "a_localized_at_pool"
    return "c_partial_mixed"


def classify_backaction(rec_joint: float, d_acc_joint: float, ceil_pooled: float,
                        cka_outside_null: bool, dacc_drift_outside_null: bool) -> str:
    """v3 §3 back-action partition. Gate = band floor (D8); disturbance = {CKA, d-acc drift} only."""
    if d_acc_joint < D_ACC_GATE:
        return "objective_abandoned"                # named non-verdict (genuine collapse only)
    if rec_joint < ceil_pooled + REC_MARGIN:
        return "f_no_recovery"
    if cka_outside_null or dacc_drift_outside_null:
        return "d_restructuring_priced"
    return "e_recovery_without_disturbance"          # KILL (recovery for free)


def select_verdict_beta(dacc_by_beta: dict) -> float | None:
    """v3 §3: verdict beta = the smallest beta in BETAS whose joint d-acc clears the gate."""
    for b in sorted(dacc_by_beta):
        if dacc_by_beta[b] >= D_ACC_GATE:
            return b
    return None
