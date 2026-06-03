# chatv2 → LATTICE handoff: warm-start asymmetry + parity confirm

> 2026-06-01. The two things the LATTICE/GPU model build asked chatv2 to pin down,
> stated implementation-grade. Runner of record: `scripts/chatv2_phase0_bodyresist.py`
> (GPU-ready — `run()` auto-selects CUDA; models/tensors already `.to(device)`).
> Cells + read-out: `PHASE1_R1_COMPLETION.md`. Promotion discipline: `PROMOTE_GATE.md`
> (we are R1-partial; no R2/R3 language).

## 1. Warm-start asymmetry (the curriculum contract)

Warm-start transfers a **learned generative backbone** to a **harder** rung so it
bypasses the grok flat-phase that kills the hard rung cold. Four asymmetries the
build must honor — get any wrong and either `load_state_dict` throws or the
contrast is contaminated:

**A. Direction: easier → harder, never the reverse.** The easy rung learns cold;
the hard rung loads the easy rung's `state_dict`, then continues training. Rung
orders: `H: 8 → 16 → 32` and `arity: 2 → 3`. You never warm-start the easy rung
from the hard one.

**B. Shared positional capacity (the load-bearing one).** Source and target must
instantiate the **same architecture including the positional-embedding size**, or
the checkpoint's `pos.weight` shape mismatches and the load fails. So size the
positional embedding for the **target (largest) rung in both**, via `--pos-h <maxH>`:
- *H-scaling:* to warm-start `H=16` from `H=8`, train `H=8` with `--pos-h 16`. Its
  `pos.weight` is `(bpc·16, d)`; `H=8` uses only the first `bpc·8` rows — **the rest
  are reserved, untrained, spare capacity dedicated to the target.** That spare-row
  reservation *is* the asymmetry.
- *arity-scaling:* **no pos-h change needed** — at `bpc=24`, arity-2 (`2·12·8`) and
  arity-3 (`3·8·8`) both give `L=192` at `H=8`, so the architectures already match.
  Warm-starting L2 (arity-3) from an arity-2 checkpoint is a plain same-shape load.

**C. Chain from the last *learned* rung.** Each rung warm-starts from the most
recent rung whose gen actually learned (`eval_loss < log2 − 0.02`). If a rung fails
to learn, the next one warm-starts from the last **success**, not the failure.
(Harness: `prev_ckpt` only advances when `learned == True`.)

**D. Gen-only transfer; twin always fresh; twin skipped if gen UNLEARNED.** Only the
**generative** backbone is checkpointed and warm-started. The control-only **twin is
trained from scratch at every rung** — it is the contrast baseline, and transferring
it would contaminate `objective_excess`. The twin is **skipped entirely** when the
gen is UNLEARNED (no contrast against a model that didn't learn).

> Immediate use: **L2 (arity-3) is the learnability risk** — it did not learn at
> smoke scale. If cold L2 returns UNLEARNED at full scale, warm-start it from a
> learned **arity-2** checkpoint (asymmetry A + the no-pos-change arity case in B).

## 2. Parity confirm (the latent generator + the non-negotiable de-confound)

Per-channel latent `z_i ∈ {0,1}` (one per channel, iid per sequence), **encoded in
the parity of independent k-tuples** (`k = arity`). Channel `i` emits `P = bpc//k`
tuples. For each tuple:

- the first `k−1` bits are **fair** (iid uniform);
- the last bit `= (XOR of the first k−1) ⊕ x`, where `x ~ Bernoulli(0.5 ± δ)` chosen
  by `z_i` (leans to 1 if `z_i=1`, to 0 if `z_i=0`, by margin δ).
- ⇒ each tuple's **k-bit parity = x**, biased by `z_i`.

**The de-confound guarantee (mandatory, non-negotiable):** no individual bit and no
sub-tuple correlates with `z_i` (`Cov = 0`), so a **linear probe on the raw input
bits reads `z_i` at ≈ chance**. **Confirmed:** arity-2 ≈ 0.50, **arity-3 = 0.484**.
The GPU re-implementation **must run this input-probe pre-check and require
`≤ 0.60` before any verdict** — it is the gate that makes the substrate non-trivial,
and it catches any re-implementation bug that accidentally leaks the latent.

- **Arities in play:** `2` = pair-XOR (R1 baseline, L1); `3` = 3-bit parity (R1 L2).
  `arity=2` in the runner is bit-identical to the original pair-XOR.
- **Layout:** tuples concatenated, channels round-robin (tuple-slot `p` → channel
  `p % H`); `L = k · (bpc//k) · H`. At `bpc=24`: both arities give `L=192` at `H=8`.
- **Task:** next-token prediction over the bit stream; predicting a tuple's last bit
  requires computing its k-parity → forces the model to build `z_i`.
- **Decision / readout:** decision `= z_1` (channel 0). Fingerprint (`d_dec`,
  `cross_latent_leak`, `body_carry`, …) on the residual stream; fair-readout reads
  each `z_i` at its channel's freshest token, `_lastpos(H, bpc, arity)`.
- **Scope honesty (carry it forward):** L1/L2 are **parity-family** — the only clean
  input-undecodable functions we could construct. Cross-family generality is an
  **R1+ limitation, not claimed**.

## 3. GPU-port contract (minimal)

- `run()` already selects CUDA when available; the model + train/eval tensors are
  device-correct. The **only** likely hot-spot is `gen_batch` (numpy/CPU) — moving
  tuple generation to torch-on-GPU is a correctness-equivalent speedup, not a
  semantics change. **Do not alter the parity construction or the pre-check.**
- Frozen R1 baseline (one knob varied per cell): `--latent computed --fair-readout
  --h-sweep 8 --d-model 192 --delta 0.45 --bits-per-channel 24 --max-steps 6000
  --min-steps 3000 --patience 10 --seed 0`. Cells: `--arity 3` (L2), `--d-model 128
  --n-layers 4` (A2), `--delta 0.30` (F-δ), `--lr 1e-3` (F-opt).
- **Pass/fail per cell (frozen):** pre-check ≈ chance, gen learned, `d_dec ≥ H/2`,
  `z1_acc ≥ 0.70`, `cross_latent_leak ≤ 0.58`, **`objective_excess ≥ 0.10`**
  (`= body_carry_gen − body_carry_twin`). Read-out: `chatv2_phase1_adjudicate.py` /
  `chatv2_phase1_contrast.py`. F-readout is already **PASS** (free).
- **Receipt:** per-cell `manifest.json` (records with `d_dec`, `z1_acc`,
  `cross_latent_leak`, `body_carry`, `eval_loss`, `status`) + the contrast JSON.
