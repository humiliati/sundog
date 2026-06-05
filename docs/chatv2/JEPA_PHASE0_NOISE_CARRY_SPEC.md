# JEPA Phase 0 — Noise-Carry Test (coupled toy)

> 2026-06-04, DRAFT for operator lock review. No JEPA code, no run. Implements
> `SUNDOG_V_JEPA.md` Phase 0, **re-posed** to the noise-carry read: the determining-shadow
> closure does *not* distinguish JEPA from GEN (both recover the shared `u`; `k_state(z_j)` is
> noise-driven for both), so the distinguishing read is **whether the body retains the
> unpredictable private noise `x_i`**. Tier **R1 (toy)**.

## 1. Primary claim

> **JEPA reduces private-noise carry relative to GEN at matched substrate, budget, and read
> capacity** — the JEPA body discards the per-sample unpredictable noise `x_i` that the
> generative body must retain.

This sharpens, rather than repeats, the chatv2 result: noise-carry *is* the chatv2
`objective_excess` measure, which **deflated to architectural** by `d=256`. So the real question
is whether JEPA's *principled* discarding (embedding prediction) **survives the capacity confound
that killed chatv2's gen-vs-twin**.

## 2. Substrate (frozen)

The Phase-7b coupled toy (`chatv2_phase0_bodyresist.py --latent coupled`):
`z_i = parity(u,A_i) ⊕ x_i`; `m=3` source; `A = _COUPLE_A` (8×3); `H=8` latents;
**`p_noise = 0.10`** (the lane's confirmed-closure regime — clean `u`, characterized bodies).
Input = the parity-channel encoding (input-undecodable; the de-confound holds by construction).

The private noise is ground truth: **`x_i = z_i ⊕ parity(u,A_i)`**, computed from the saved
`(u, z)` — a directly probe-able target.

## 3. Bodies (matched pair)

- **GEN** (baseline): the generative objective (reconstruct the parity-channel input) — must
  carry `x_i`.
- **JEPA**: context encoder (matched to GEN's body) + frozen predictor + **EMA / stop-grad target
  encoder**. Mask a fixed fraction of input channels; predict the **masked channels' target
  embeddings** from context; loss in embedding space (no input reconstruction). Collapse
  avoidance = **VICReg variance/covariance term + explicit collapse check** (per-dim embedding
  variance floor; rank; no constant embedding) — per the JEPA lit pass.
- **Matched control:** same body width/depth, same train budget (steps), same substrate, same
  augment/noise slate; matched body init on the shared encoder where the architecture allows.
- **Body read** = the context encoder's final body activation (the same residual/body read used
  for GEN).

## 3.1 JEPA mechanics — FROZEN implementation pin

Frozen **before** implementation; the smoke records the collapse metrics at exactly these
values, and any change requires a spec amendment.

| knob | frozen value |
| --- | --- |
| masking | **50% latent-channel mask** — 4 of the `H=8` latents masked per sample (random set, seed-derived); context = the other 4 latents' tokens, targets = the 4 masked latents |
| context encoder (= body) | the TinyGPT body, **matched to GEN** (width `d_model`, same depth/heads, matched init); body read = final hidden, identical pooling to GEN |
| target encoder | EMA copy of the context encoder, **tau = 0.99**, stop-grad; encodes the masked latents' tokens → target embeddings |
| predictor | **2-layer MLP, width `d_model`**, GELU; input = context body rep + a learned per-target channel query; output = predicted embedding per masked channel |
| embedding dim | **= `d_model`** (matched body/embedding) |
| loss | `L = λ_inv·MSE(pred, stopgrad(target)) + λ_var·var + λ_cov·cov`; VICReg var/cov applied to the **context-encoder body embeddings** (the read surface) |
| VICReg weights | **λ_inv = 25, λ_var = 25, λ_cov = 1** (original VICReg defaults); variance target γ = 1.0, hinge `mean(relu(1 − std_d))` |
| optimizer / budget | **matched to GEN** (same AdamW, lr, total steps) |

**Collapse guard** (gates `blocked_by_unfaithful_jepa`; the smoke records and asserts these):
- per-dim body-embedding std **≥ 0.10** (batch-averaged) on **≥ 90%** of dims;
- effective rank of the body-embedding covariance **≥ 0.5·`d_model`**;
- body output not constant (total variance > floor).

Collapse = any threshold violated at smoke end → `blocked_by_unfaithful_jepa` (repair, not a
null). **Pre-locked contingency:** if the smoke collapses at `λ_var=25`, raise to `λ_var=50` and
re-smoke once as a **global pre-measurement setting**; a second collapse is
`blocked_by_unfaithful_jepa` for the locked design. No per-seed, per-width, or post-hoc
measurement retry is allowed.

**Lock-review pins:** VICReg `λ_inv/λ_var/λ_cov = 25/25/1`; the exact collapse thresholds
(std ≥ 0.10 on ≥90% of dims, eff-rank ≥ 0.5·`d_model`); embedding dim = `d_model`; and the
`λ_var=50` single global smoke-retry contingency.

## 4. Primary statistic — `noise_det`

For each trained body:

```text
noise_det = median over i in {1..H} of  det(x_i | body)
det(x_i | body) = (held-out acc - base) / max(1 - base, 1e-9),  linear probe body -> x_i
```

`median` over the H private bits is the primary read. The private bits are imbalanced at
`p_noise=0.10`, so the readback must report each `x_i` base rate and minority count. A
balanced-subset score may be reported as a **sidecar only**; it cannot promote a verdict. If
the full-distribution read is support-starved (minority count `< 50` for more than two of the
eight `x_i` in the held-out probe split), the phase returns `blocked_by_noise_readout`, not a
rescued balanced-subset pass.

Expected: **GEN `noise_det` HIGH** (carries `x`), **JEPA `noise_det` LOW** (discards `x`).

## 5. Capacity sweep (the chatv2 hook)

Vary body width **`d_model ∈ {128, 256}`** — `256` is the point where chatv2's `objective_excess`
collapsed. For each `(d, seed)`, compute the paired gap
`gap_seed = GEN_noise_det − JEPA_noise_det`. The per-`d` headline is
`gap_d = median_seed(gap_seed)`. **Non-bottleneck = `d=256`.** (Optional middle rung
`d=192` if compute allows, report-only unless pre-locked separately.)

## 6. Kill-gate

`jepa_noise_discard_confirmed` requires **all** of:
- `gap_256 >= frozen_delta = 0.15`, **and**
- the GEN positive-control is live at `d=256`: `median_seed(GEN_noise_det) >= 0.30`, **and**
- `gap_256 > std_seed(gap_seed at d=256)`.

**`blocked_by_capacity`** if the gap `>= frozen_delta` at `d=128` but **`< frozen_delta` at
`d=256`** — the apparent discard is capacity-limited, echoing chatv2's `objective_excess`
deflation. This is **not** "JEPA works."

**`blocked_by_noise_readout`** if the GEN positive-control fails at `d=256` or the primary
full-distribution `x_i` read is support-starved. Then the linear noise-carry instrument did not
establish that GEN visibly carries `x`, so a JEPA/GEN gap cannot be interpreted.

## 7. Required controls

- **`u_det`:** both bodies retain the predictable functional — `det(u | body) >= 0.70` for GEN
  **and** JEPA. (If JEPA fails to keep `u`, it did not learn the functional → not a clean test →
  fold into `blocked_by_unfaithful_jepa`.)
- **`u_null`:** independent Bernoulli target at the same base rate — its determining read must
  stay clean (no significant determination); else `control_void`.
- **`z_det` / old closure read:** the determining-shadow `k_func(u)` / `k_state(z_j)` —
  **report-only**, expected **not** to distinguish GEN from JEPA (confirms noise-carry is the
  right read, not closure).
- **JEPA collapse guard:** target/predictor variance floor; VICReg variance/covariance receipt;
  no constant embedding. Collapse despite VICReg → `blocked_by_unfaithful_jepa`.
- **Matched GEN:** same width / budget / augment / noise slate.

## 8. Verdict tree (precedence: first match wins)

| branch | condition | reading |
| --- | --- | --- |
| `control_void` | `u_null` clears, or the selection-corrected null misbehaves | instrument unreliable; no result |
| `blocked_by_unfaithful_jepa` | JEPA collapses / objective fails (collapse guard trips, or `u_det(JEPA) < 0.70`) | not a faithful JEPA; repair, not a scientific null |
| `blocked_by_noise_readout` | GEN positive-control fails at `d=256` (`median_seed(GEN_noise_det) < 0.30`), or the primary `x_i` read is support-starved | the noise-carry instrument is not live enough to test JEPA-vs-GEN |
| `blocked_by_capacity` | gap `>= delta` at `d=128` but `< delta` at `d=256` | apparent discard is capacity-limited — the chatv2 echo, not an objective win |
| `jepa_noise_discard_confirmed` | gap `>= delta` survives at `d=256`, controls clean | GEN carries `x`, JEPA drops it; principled discarding survives capacity |
| `jepa_not_distinguished_from_gen` | no robust noise-carry gap (gap `< delta` at all `d`, not capacity-explained) | the JEPA objective does not discard noise more than GEN on this toy |

## 9. Tier / integrity

**R1 (toy)** — synthetic coupled substrate (per `SUNDOG_V_JEPA §9`). Does not license real-JEPA
language.

**Allowed (on `jepa_noise_discard_confirmed`):**
> On the coupled toy, the JEPA objective discarded the per-sample private noise that the
> generative objective retained (by noise-carry), and the gap survived a body-capacity sweep to
> the dimension where chatv2's objective-excess had deflated — measured against a matched GEN
> control and a passed JEPA collapse guard.

**Forbidden:** "JEPA learns a world model / understands"; "proves LeCun right/wrong"; "JEPA is the
path to AGI"; any statement about real I-JEPA/V-JEPA/LLM behaviour; R2 / "more than we know".

## 10. Run order

1. Operator lock review.
2. Implement the JEPA mode additively (extend `chatv2_phase0_bodyresist.py` or a new runner
   importing it); add the `noise_det` read (`x_i = z_i ⊕ parity(u,A_i)`) + the collapse guard.
3. **Smoke** (1 seed, `d=128`, few steps): JEPA trains without collapse (VICReg receipt + variance
   floor) AND the `noise_det` read runs.
4. **Training is operator-staged (background, hours on the 1080)** — GEN + JEPA × `{128,256}` × 3
   seeds. Stage the exact PowerShell commands; do not run the full battery inline.
5. Read + verdict; record in `docs/chatv2/JEPA_PHASE0_NOISE_CARRY_RESULTS.md`.

## 11. Readback checklist

- git + script SHA; substrate config (`p_noise=0.10`, `m=3`, `_COUPLE_A`);
- per-`(d, seed)` GEN/JEPA `noise_det` + paired gap; `gap_d = median_seed(gap_seed)`;
  `std_seed(gap_seed)`; the capacity-sweep table;
- `x_i` base rates + minority counts; balanced-subset sidecar if computed (non-gating);
- `u_det` (both bodies), `u_null`, `z_det` (report-only);
- JEPA collapse guard: per-dim variance floor, VICReg variance/covariance receipt, rank;
- `frozen_delta`, per-seed spread; branch verdict; allowed/forbidden from §9.

## Open lock-review knobs

- `p_noise = 0.10` (clean `u`) vs a higher-noise rung (more to discard, weaker `u_det`).
- `frozen_delta = 0.15`; capacity points `{128, 256}`; **3** seeds.

---

*Sundog Research Lab — JEPA Phase-0 noise-carry spec. Draft pending operator lock. No model run
until lock. R1 toy; kill-gated R&D.*
