"""A4 part B: oracle internal consistency, PSI slicing, off-by-one, checkpoint indexing."""
import numpy as np
from numpy.random import default_rng

from jepa_0d_accumulator_preflight import (
    AccCfg, gen_accumulator, parity_encode, parity_decode, int_to_bits, PSI,
    raw_linear_probes, oracle_recover, base_rates, _cv_acc, _det, safe_median,
)

print("=" * 70)
print("TEST 5: generation Psi vs oracle Psi vs codeword table consistency")
print("=" * 70)
cfg = AccCfg(n=400, seed=0)
rng = default_rng(0)
data = gen_accumulator(cfg, rng)
Psi = data["Psi"]
b = cfg.count_bits
print("Psi (generation, sliced):")
print(Psi)
print("Psi shape:", Psi.shape, " count_bits:", b)

# In generation: clean[ci,j] = (cbits @ Psi[j]) % 2, cbits = int_to_bits(u_t, b)
# In oracle: code_clean = (all_codes @ Psi.T) % 2, all_codes = int_to_bits(arange(K+1), b)
# Verify the oracle's clean codeword table matches the generator's clean bits exactly.
all_codes = int_to_bits(np.arange(cfg.K + 1), b)
code_clean = (all_codes @ Psi.T) % 2
# For each sample at each ckpt, regenerate clean from u and compare to data["clean"]
ckpts = data["ckpts"]
u = data["u"]
mismatch = 0
for ci, c in enumerate(ckpts):
    uc = u[:, c - 1]                  # value at checkpoint c (t=1..T, 0-based index c-1)
    expect_clean = code_clean[uc]    # (n, n_U) via table lookup
    actual_clean = data["clean"][:, ci, :]
    m = (expect_clean != actual_clean).sum()
    mismatch += m
    print(f"  ckpt {c}: table-lookup clean vs stored clean mismatches = {m}")
print("TOTAL clean-table mismatch:", mismatch, "(0 = oracle table agrees with generator)")

print()
print("=" * 70)
print("TEST 6: off-by-one — does checkpoint c read u_c (t=c, value u[:,c-1])?")
print("=" * 70)
# The substrate spec: at checkpoint t, emit readout of u_t. Generation uses
# ubits[:, t-1, :] inside the `if t in ckpts` block. Oracle compares u_ro to u[:, c-1].
# Confirm these refer to the SAME tick value.
for ci, c in enumerate(ckpts):
    # stored clean was computed from ubits[:, c-1]; check it equals readout of u[:,c-1]
    cbits = int_to_bits(u[:, c - 1], b)
    recomputed = np.stack([(cbits @ Psi[j]) % 2 for j in range(cfg.n_U)], axis=1)
    agree = np.array_equal(recomputed, data["clean"][:, ci, :])
    print(f"  ckpt {c}: clean == readout(u[:,{c-1}]):", agree)

print()
print("=" * 70)
print("TEST 7: GF(2) rank of sliced Psi — can readout uniquely identify u in 0..K?")
print("=" * 70)
def gf2_rank(M):
    M = M.copy() % 2
    rows, cols = M.shape
    r = 0
    for col in range(cols):
        piv = None
        for row in range(r, rows):
            if M[row, col]:
                piv = row; break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for row in range(rows):
            if row != r and M[row, col]:
                M[row] = (M[row] + M[r]) % 2
        r += 1
    return r
print("GF(2) rank of Psi (need >= count_bits for unique decode):", gf2_rank(Psi), "/", b)
# distinct codewords?
print("distinct clean codewords among 0..K:", len(set(map(tuple, code_clean.tolist()))), "of", cfg.K + 1)

print()
print("=" * 70)
print("TEST 8: PSI[:, -b:] slicing when count_bits != 3 (K=2 -> b=2, K=8 -> b=4)")
print("=" * 70)
for K in [2, 8]:
    c2 = AccCfg(K=K, n=300, seed=1)
    # n_U default 8, but PSI has 3 cols; for b=4 the pad branch triggers
    try:
        d2 = gen_accumulator(c2, default_rng(1))
        print(f"  K={K} b={c2.count_bits}: Psi.shape={d2['Psi'].shape}")
        orc = oracle_recover(d2, c2)
        print(f"    oracle u_acc_overall={orc['event_route_u_acc_overall']} "
              f"readout per-ckpt={orc['readout_route_u_acc_per_ckpt']}")
        # check rank for unique decode
        print(f"    distinct codewords 0..K: "
              f"{len(set(map(tuple, ((int_to_bits(np.arange(K+1), c2.count_bits) @ d2['Psi'].T)%2).tolist())))} of {K+1}")
    except Exception as ex:
        print(f"  K={K}: EXCEPTION {type(ex).__name__}: {ex}")
