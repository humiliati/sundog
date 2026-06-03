# Chat-v2 Phase 7 — Coupled-Latent Closure Spec (pre-registration)

> 2026-06-03, DRAFT for sign-off — **not run.** Phase 7 of the frozen
> `docs/SUNDOG_V_ALLELOPATHY.md` roadmap: the coupled-latent toy, **the first place
> in this lane where a positive `k_func << k_state` closure can exist.** Phase 2
> showed the uncoupled pair-XOR toy has none (independent latents → no coupling for
> a determining set to exploit). "Closure is a coupling phenomenon" — so build a
> substrate with coupling by construction and test whether the closure survives
> training + body-readout. Tighten-not-loosen binds. **No verdict run until sign-off.**

Reserved: `scripts/chatv2_phase7_coupled.py` (generator + train) reusing the
Phase-0 backbone; probe extends `scripts/chatv2_phase2_shadowset.py`.
Out: `results/chatv2/phase7-coupled-latent/`.

## 1. The coupled design (data-level closure VALIDATED 2026-06-03)

- **Hidden source:** `m = 3` independent bits `u = (u_0,u_1,u_2) ~ Bernoulli(0.5)`.
- **Coupling graph** (fixed, known → `k_func` pre-registerable): `A` is the 8×3
  binary matrix whose rows are the parity masks
  `[001, 010, 100, 011, 101, 110, 111, 001]`.
- **State latents:** `z_i = (A_i · u mod 2) ⊕ x_i`, `x_i ~ Bernoulli(p_noise)`,
  **`p_noise = 0.25`** (independent per-latent coupling noise). `H = 8`.
- **Encoding (de-confound, unchanged from Phase 2):** each `z_i` is parity-channel
  encoded into its own input channel (first `A−1` bits fair, last `= channel parity
  ⊕ bias keyed by z_i`), so `z_i` is **input-undecodable** by a linear probe. The
  coupling lives only in how the `z_i` *values* are generated (shared `u`), not in
  the per-channel encoding — so the de-confound pre-check still applies per latent.

**Pre-registered closure (from the clean-`z` data check, this spec's prediction):**

```
 k   func_u_acc   state_z_acc
 3   0.757        0.530        <- k_func = 3 (u determined), k_state never reaches 0.70
 7   0.757        0.630        <- individual z states resist to the end
```

i.e. **`k_func = 3 ≪ k_state = none≤7`** on the ideal latents. The trained-body
probe is the real test: readout noise (~0.8 per-latent) may tighten or wash it.

## 2. Training (gen + twin)

- Backbone + objective identical to Phase 1 seed-stability: `H=8`, `d_model=192`,
  fair readout, `bits_per_channel=24`, `max_steps=6000`, grok-aware floor,
  seeds `{0,1,2}`, GTX-1080 GPU venv. **gen** = full generative objective; **twin**
  = control-only (decision latent `z_0`). UNLEARNED guard binds (a body whose gen
  did not learn is never probed).
- The generator (`--latent coupled`) saves `bodies (N,4,8,192)`, `z (N,8)`, **and
  `u (N,3)`** (the hidden source) into the npz so the probe can target `u`.
- **De-confound pre-check (gate):** linear input-probe on each `z_i` ≤ 0.60 ≈
  chance, else ABORT (same `F2'` as Phase 2). Smoke-verify before any train run.

## 3. The closure probe (extends Phase 2)

Reuse the Phase-2 shadow-set machinery (train/held split, train-selected subsets,
selection-corrected null, controls) with **one new registered functional**:

- **`k_func` target = the hidden source `u`** (the coupling-relevant quantity): for
  shadow set `S` of `z`-readouts, predict each `u_l` from `{s_i : i∈S}`; `func(S)`
  = mean held accuracy over `l`. `k_func` = smallest `|S|` with `func(S*) ≥ 0.70`,
  selection-corrected against the random-direction null (Amendment A4).
- **`k_state` target = the omitted individual latents `z_j`**: mean held label
  accuracy over `j∉S`; `k_state` = smallest `|S|` with `state(S*) ≥ 0.70`
  (expected `none≤7`). This is deliberately **not** the Phase-2 omitted-score FVE
  target; the Phase-7 clean-latent bracket is defined in label space so the
  validated data-level table remains the binding prediction.
- **Bracket:** the headline is `k_func` vs `k_state`. The NSE-like positive is
  `k_func ≪ k_state` (small shadow set closes `u` while individual states resist).

## 4. Headline branches (pre-registered)

| branch | condition | reading |
| --- | --- | --- |
| `closure_confirmed` | `k_func` exists and `k_func < k_state` (or `k_state` none) in the **trained body**, controls pass | **first positive closure** — a coupled substrate has `k_func ≪ k_state`; the determining-shadow-set instrument measures it |
| `closure_washed_by_readout` | data-level bracket holds but the trained-body `func(u)` never clears the selection-corrected null | the closure exists in the data but not in the recoverable body representation (readout noise) — honest partial |
| `closure_absent` | no `k_func` even at data level on the realised cells | design/learnability failure; repair before reading |
| `closure_void` | de-confound, UNLEARNED, split, or control contract fails | no result |

## 5. Negative control (mandatory)

Re-run the same `u`-target probe on the Phase-2 **uncoupled pair-XOR** bodies
(`phase1-seedstab`) using a frozen synthetic control target
`u_null ~ Bernoulli(0.5)^(N×3)`, generated from `default_rng(7007 + seed)` and
saved in the control receipt. Expected: `k_func = none` — confirming the runner
does not hallucinate hidden-source closure when the target is independent of the
body. A coupled `closure_confirmed` is only credible if this uncoupled control
shows `closure_absent` on the same runner.

Secondary negative: the Phase-2 omitted-`z` same-seed probe remains the expected
uncoupled null. It is reported as supporting context, not substituted for the
primary `u_null` control.

## 6. Frozen parameters

| parameter | value |
| --- | --- |
| `m` (hidden source bits) | 3 |
| coupling graph `A` | `[001,010,100,011,101,110,111,001]` |
| `p_noise` | 0.25 |
| `H`, `d_model`, steps, seeds | 8, 192, 6000, {0,1,2} |
| `k_func` target | hidden source `u`, mean held acc `≥ 0.70`, selection-corrected null |
| `k_state` target | omitted individual `z_j`, mean held label accuracy `≥ 0.70`; not Phase-2 score FVE |
| pre-registered prediction | `k_func = 3`, `k_state = none≤7` (data-level) |
| null / controls | inherited from Phase 2 spec + Amendment 1 (A4 null, perm, random-dir, twin floor, A3 subspace ceiling carried as diagnostics) |

## 7. Receipt schema

`docs/chatv2/PHASE7_COUPLED_LATENT_RECEIPT.md`: provenance (spec sha, runner sha,
git, wall-clock); data-level validation command/output; de-confound pre-check;
per-seed gen learned + `k_func` / `k_state` + branch; the bracket table
(func_u_acc and state_z_acc vs `k`); selection-corrected null; negative-control
(`u_null` on uncoupled bodies) `k_func`; secondary Phase-2 omitted-`z` null;
controls table; allowed/forbidden language; honest tier. Promotion stays gated —
a positive here is a **toy** closure on a *designed* coupled substrate, not a
real-LLM or NSE claim.

## 8. Run order

1. Sign-off (freeze this spec).
2. Implement `--latent coupled` generator + de-confound smoke (no train).
3. Run a timing smoke before training. If the measured or extrapolated train is
   > 10 minutes wall-clock, do **not** run it inline; write the exact PowerShell
   command(s), wall-clock estimate, resume-safety notes, and read-back paths into
   the receipt/staging note for the operator or long-budget runner.
4. Train gen+twin (3 seeds, GPU) only under the timing rule above.
5. Probe (coupled bodies + uncoupled negative control).
6. Receipt + branch adjudication.

---

*Sundog Research Lab — chatv2 Phase 7 spec, pre-registration draft. Internal; gated
on the frozen `SUNDOG_V_ALLELOPATHY.md` roadmap. No verdict run until sign-off.*
