"""A4 code/bug review scratch tests for jepa_0d_accumulator_preflight.py."""
import numpy as np
from numpy.random import default_rng

from jepa_0d_accumulator_preflight import (
    AccCfg, gen_accumulator, parity_encode, parity_decode, int_to_bits, PSI,
    raw_linear_probes, oracle_recover, base_rates, _cv_acc, _det, safe_median,
)
import chatv2_phase0_bodyresist as bp

print("=" * 70)
print("TEST 1: parity_encode faithfulness vs _gen_computed (arity=2, H=1)")
print("=" * 70)
# _gen_computed with H=1 uses chan = arange(P)%1 = all-0; px shape (B,P). For a
# single-channel comparison, generate with same payload by fixing z.
# We compare the codec MECHANICS: same RNG draw sequence -> same bits?
rng1 = default_rng(123)
payload = (default_rng(7).random(2000) < 0.4).astype(np.int64)
P, A, delta = 5, 2, 0.45
# preflight encoder
bits_pf = parity_encode(payload, P, A, delta, rng1)
# replicate _gen_computed internals for a single channel with the SAME draw order
rng2 = default_rng(123)
n = payload.shape[0]
px = 0.5 + delta * (2.0 * payload - 1.0)
x = (rng2.random((n, P)) < px[:, None]).astype(np.int64)
tup = rng2.integers(0, 2, size=(n, P, A)).astype(np.int64)
tup[:, :, A - 1] = (tup[:, :, :A - 1].sum(2) % 2) ^ x
bits_ref = tup.reshape(n, P * A)
print("draw-order-identical bits match:", np.array_equal(bits_pf, bits_ref))
# tuple parity should equal x
tp = bits_pf.reshape(n, P, A).sum(2) % 2
print("tuple parity == payload-biased draw x:", np.array_equal(tp, x))

print()
print("=" * 70)
print("TEST 2: marginal fairness of every token (no single bit reveals payload)")
print("=" * 70)
# correlation of each emitted bit with payload should be ~0
corrs = []
for col in range(bits_pf.shape[1]):
    c = np.corrcoef(bits_pf[:, col], payload)[0, 1]
    corrs.append(c)
print("max |corr(bit, payload)| over all tokens:", round(max(abs(np.array(corrs))), 4))

print()
print("=" * 70)
print("TEST 3: parity_decode round-trips a known payload")
print("=" * 70)
rng3 = default_rng(55)
pay = (rng3.random(5000) < 0.5).astype(np.int64)
enc = parity_encode(pay, 5, 2, 0.45, rng3)
dec = parity_decode(enc, 5, 2)
print("decode accuracy (P=5, delta=0.45):", round((dec == pay).mean(), 4))

print()
print("=" * 70)
print("TEST 4: int_to_bits endianness + PSI slicing for count_bits != 3")
print("=" * 70)
print("int_to_bits(5, 3) =", int_to_bits(np.array(5), 3), "(expect big-endian [1,0,1])")
print("int_to_bits(6, 3) =", int_to_bits(np.array(6), 3), "(expect [1,1,0])")
# count_bits for default K=6
cfg = AccCfg(n=200)
print("count_bits for K=6:", cfg.count_bits)
# Now probe a config where K forces count_bits != 3
for K in [1, 2, 3, 4, 7, 8, 15]:
    c = AccCfg(K=K, n=10)
    print(f"  K={K}: count_bits={c.count_bits}, ceil(log2(K+1))={int(np.ceil(np.log2(K+1)))}")
