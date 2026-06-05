# JEPA-0D — Accumulator / Count Substrate (spec)

> 2026-06-05. **STATUS: mask policy LOCKED (whole-checkpoint, operator 2026-06-05);
> self-consistency review PASSED; runner built + dev-validated; smoke operator-staged.** Model-free preflight is COMPLETE
> and PASSED (`preflight_pass_ready_for_spec`); no model battery has been run. Implements the
> JEPA-0D fork of `docs/SUNDOG_V_JEPA.md` per `docs/chatv2/JEPA_0D_HANDOFF.md`. Tier **R1 (toy)**.
> Kill-gated R&D (2026-06-04 strategy pivot).
>
> **DEVIATION FROM THE HANDOFF — APPROVED AT LOCK (2026-06-05, §3.1):** the handoff says "carry over
> the 50% latent-channel mask." The preflight proved that a random latent mask reproduces the parity
> Phase-0 failure on this substrate (a masked count-readout is XOR-recoverable from its
> same-checkpoint peers at det≈0.56 with **zero** count integration). This spec therefore pins
> **whole-checkpoint masking** instead, which the preflight verified floors that shortcut to
> det 0.131 (≤ 0.15) while keeping the event→`u_t` path at det 0.99. This is the
> "Pins to preserve **unless the preflight forces a re-pose**" clause firing as designed.

## 0. Why this lane / what parity proved

Parity Phase-0 (`JEPA_PHASE0_NOISE_CARRY_SPEC.md`) closed `blocked_by_unfaithful_jepa`: JEPA
trained without collapse (eff-rank ~68) and showed the predicted lower flip-conditioned read
(`z_flip_gap=+0.208`), but did **not** keep the shared source — masked-context `u_det=0.368 ≪ 0.70`.
Root cause (handoff): the parity source `u` was *arbitrary* from JEPA's view; the encoder could
match masked target embeddings without isolating `u`. The accumulator is the fairer substrate:
masked structure is made to **depend on a running count `u_t`**, so a predictive JEPA objective has
a native reason to keep it.

## 1. Primary claim

> **On the accumulator toy, a JEPA body retains the running bounded count `u_t` (the predictable
> functional) while discarding the per-readout private flip `x` that a generative body must carry.**

The kill question is the same one parity failed: does a faithful small JEPA **keep the functional**
(`u_det ≥ 0.70`)? If yes, the discard contrast (gate 5) and the capacity sweep (gate 6) become
interpretable. If no, the lane shelves — the substrate change did not rescue functional retention.

## 2. Substrate (FROZEN — reconciled to the preflight receipt)

Generator: `scripts/jepa_0d_accumulator_preflight.py :: gen_accumulator`. One sequence = `T` ticks.

```text
hidden event    e_t ~ Bernoulli(p_event),                         t = 1..T
hidden count    u_t = min(sum_{s<=t} e_s, K)                      BOUNDED sum (not modulo)
event channel   parity-channel encoding of e_t   (CLEAN)         -> u_t is a clean functional;
                                                                    oracle sums decoded events -> u_t
count-readout   at checkpoints c in C, n_U channels emit
   channels        z_{c,j} = parity(PSI_j, bits(u_c)) XOR x_{c,j},  x_{c,j} ~ Bernoulli(p_noise)
                   then z_{c,j} is itself parity-channel encoded (input-undecodable).
                   clean part depends on u_c  -> forces a tracker to keep the count
                   private flip x_{c,j}       -> the JEPA-vs-GEN discard target (gate 5)
```

Parity-channel codec = `chatv2_phase0_bodyresist._gen_computed`, **bit-identical** (adversarially
verified: `parity_encode` matches draw-order and tuple-parity payload; every token is marginally
fair, max |corr(bit, payload)| = 0.004 at n=200k). De-confound holds **by construction** (last
tuple bit = parity(fair bits) XOR payload → marginal randomised) and is verified empirically (§5).

| knob | locked value | note |
| --- | --- | --- |
| `K` (count cap) | **6** | `count_bits = 3`; bounded, not modulo (handoff pin) |
| `T` (ticks) | **12** | |
| `p_event` | **0.34** | gives `e_base ≈ 0.335`, spreads `u_t` across {0..6} |
| `arity` `A` | **2** | pair-XOR baseline (bit-identical to original parity codec) |
| `delta` | **0.45** | parity-channel payload softness (carried from Phase-7b) |
| `p_noise` | **0.10** | private flip rate on readouts (clean-`u` regime, carried) |
| `P_e` (event tuples/tick) | **5** | odd → clean majority vote → oracle `e_acc = 0.999` |
| `P_u` (readout tuples/ch) | **3** | |
| `n_U` (readout channels/ckpt) | **7** | the 7 **distinct** nonzero GF(2) functionals of 3 count-bits |
| `C` (emit checkpoints) | **{4, 8, 12}** | readouts emitted here; `u_t` read here at the body gates |
| `n` (fingerprint) | **3000** | pinned floor; an n=2000 regression run showed tail-class starvation (`support_starved`, not banked) |
| `PSI` | rows `[001,010,100,011,101,110,111]` | rank-3, min-codeword-Hamming 4, **no duplicate** |
| `L` (tokens/seq) | **246** | `12·(P_e·A) + |C|·n_U·(P_u·A) = 120 + 126` |

`PSI` replaces the reused `_COUPLE_A` (which had a literal duplicate row `0==7`); `make_psi`
generates a rank-`count_bits`, duplicate-free coupling for any `K` (the `K≥8` crash is fixed).

## 2.1 Ground-truth targets

- **Functional to keep:** `u_t` (per-tick bounded count), read at checkpoints `C`. Multiclass {0..K}.
- **Discard target:** `x_{c,j} = z_{c,j} XOR parity(PSI_j, bits(u_c))` — the private readout flip.
- **Observed noisy channel:** `z_{c,j}` (the readout payload), audited by the flip-conditioned read.
- **Events:** `e_t` (clean) — report-only de-confound target; the oracle's input.

## 3. Bodies (matched pair)

- **GEN** (baseline): generative next-token objective on the accumulator tokens (reconstruct the
  parity-channel input). Must carry the private flips `x`. Reuse `train_generative` / `TinyGPT`
  from `chatv2_phase0_bodyresist` UNEDITED. Body read = the context encoder's final/checkpoint
  hidden (the same read used for JEPA).
- **JEPA**: context encoder (= GEN's `TinyGPT` body, matched width/depth/heads/init) + EMA/stop-grad
  target encoder + 2-layer predictor + VICReg (§3.1).
- **Matched control:** same width/depth, same train budget (steps), same substrate, same optimizer.

## 3.1 JEPA mechanics — FROZEN implementation pin

Carried over from the parity spec **except the mask policy** (the load-bearing re-pose, §0).

| knob | frozen value |
| --- | --- |
| **masking** | **WHOLE-CHECKPOINT** (re-pose). Per sample, pick one checkpoint `c ∈ C` uniformly and mask **all `n_U` of its count-readout token-blocks as a unit**; context = every other token (all event channels + the other checkpoints' readouts); targets = the `n_U` masked readout embeddings of `c`. Removes the same-checkpoint XOR shortcut so events→`u_t` integration is the only non-trivial route. |
| context encoder (= body) | the `TinyGPT` body, matched to GEN (width `d_model`, same depth/heads, matched init); body read = checkpoint/final hidden, identical pooling to GEN |
| target encoder | EMA copy of the context encoder, **tau = 0.99**, stop-grad; encodes the **unmasked** sequence; targets = the masked checkpoint's readout-position embeddings |
| predictor | **2-layer MLP, width `d_model`**, GELU; input = context body rep + a learned per-readout-channel query; output = predicted embedding per masked readout channel |
| embedding dim | **= `d_model`** |
| loss | `L = λ_inv·MSE(pred, stopgrad(target)) + λ_var·var + λ_cov·cov`; VICReg var/cov on the **context-encoder body embeddings** (the read surface) |
| VICReg weights | **λ_inv = 25, λ_var = 25, λ_cov = 1**; variance target γ = 1.0, hinge `mean(relu(1 − std_d))` |
| optimizer / budget | **matched to GEN** (same AdamW, lr, total steps); inherit the Phase-7b training cell except the `d_model` capacity axis |
| JEPA read protocol | **masked-context average**: read the context encoder under the same whole-checkpoint mask distribution used in training, take the checkpoint/final body activation, average over **`mask_reads = 8`** independently sampled mask patterns before probing |

**Mask policy LOCKED to whole-checkpoint** (operator, 2026-06-05). A **predict-ahead** variant
(mask only the last checkpoint, tick 12, predict from earlier ticks) was considered and **not
selected** — it is the most JEPA-native option (causal, no future leakage) but narrower (one mask
target, fewer `u_t` read positions); it is recorded here as the sole sanctioned future re-pose if
whole-checkpoint masking later proves insufficient. No mid-lane switching without a spec amendment.

**Collapse guard** (gates `blocked_by_unfaithful_jepa`; smoke records & asserts):
- per-dim body-embedding std **≥ 0.10** on **≥ 90%** of dims;
- effective rank of the body-embedding covariance **≥ max(8, 0.05·`d_model`)**;
- body output not constant.

**Pre-locked collapse contingency:** if the smoke collapses at `λ_var = 25`, raise to `λ_var = 50`
and re-smoke **once** as a global setting; a second collapse is `blocked_by_unfaithful_jepa`.

## 3.2 `u_det` metric — PINNED (one definition, all gates)

```text
u_det = median_{c in C} ( acc_c - maj_c ) / max(1 - maj_c, 1e-9)
```
accuracy-based multiclass, where `acc_c` = held-out accuracy of a linear probe `body@c -> u_c` and
`maj_c` = the majority-class frequency **at checkpoint c** (per-position baseline → a position-only
predictor scores ~0; verified: a constant `E[u|t]` feature scores det = 0.000). The **same
per-position det formula** is used everywhere; the two gates differ only in read set and aggregator:
gate-1 de-confound applies it to **every tick and takes the max** (strictest floor, `≤ 0.10`), while
the gate-4 retention bar uses the **median over checkpoints C** (`≥ 0.70`). An **MAE-reduction det**
is reported as a **report-only** ordinal sidecar; within-1 accuracy-det is **rejected** (its baseline
→ 1 at early checkpoints causes a div-by-~0 blowup; not computed in the banked preflight).
**Mid-lane metric substitution is forbidden**; restating the 0.70 bar under MAE-det would require
re-deriving and re-pinning it (acc-det and ordinal-det differ by ~0.15).

## 4. Mandatory gate order (do not reorder)

| # | gate | bar | status |
| --- | --- | --- | --- |
| 1 | **model-free de-confound** | raw linear `u_det / e_det / emission det ≤ 0.10`, complete (no dropped probe) | **PASS** (§5) |
| 2 | **information-present oracle** | deterministic parser `u_t` recovery `≥ 0.95` | **PASS** (§5) |
| — | **support** | no class/tick starvation; flip support adequate | **PASS** (§5) |
| — | **mask-necessity (model-free)** | whole-ckpt shortcut det `≤ 0.15` AND intended path det `≥ 0.70` | **PASS** (§5) |
| 3 | **GEN positive control** | trained GEN body carries observed `z` (`z_flip_acc` high) **and** keeps the functional (`u_det ≥ 0.70`) | staged |
| 4 | **JEPA functional-retention** | JEPA `u_det ≥ 0.70` (**the kill gate; where parity died**) | staged |
| 5 | **noise-discard contrast** | paired `gap = GEN_z_flip_acc − JEPA_z_flip_acc ≥ frozen_delta = 0.15` | staged, gated on 4 |
| 6 | **capacity sweep** | gap survives `d=256` | staged, gated on 5 |

Gates 1–2 + support + mask-necessity are the model-free preflight (COMPLETE). Gates 3–6 are
model training, **operator-staged**, and run only in order.

## 5. Preflight result (COMPLETE — receipt `results/chatv2/jepa-0d-accumulator-preflight/preflight.json`)

`preflight_pass_ready_for_spec` at the locked config (n=3000, seed 0, L=246, 82s):

| read | value | bar |
| --- | ---: | --- |
| de-confound — raw linear `u_det` (max / median) | **−0.050 / −0.087** | ≤ 0.10 |
| de-confound — raw linear `e_det` (max) | **−0.123** | ≤ 0.10 |
| de-confound — emission clean / obs det (median) | **−0.137 / −0.123** | ≤ 0.10 |
| de-confound — paranoid bit-count → `u_T` det | **−0.021** | ≤ 0.10 |
| de-confound — eval complete (no silent drop) | **True** | required |
| oracle — event-route `u` recovery / `e_acc` / exact-seq | **0.9941 / 0.999 / 0.988** | ≥ 0.95 |
| oracle — readout-route per ckpt {4,8,12} | 0.931 / 0.902 / 0.856 | corroboration only |
| support — `e_base`; pooled `u` starved; ticks starved; flip-min | 0.335; none; none; **250** | no starvation |
| mask-necessity — same-ckpt shortcut (clean) | **0.560** | documents footgun |
| mask-necessity — **whole-ckpt** shortcut (clean) | **0.131** | ≤ 0.15 |
| mask-necessity — intended event→`u_t` path (clean / obs) | **0.990 / 0.753** | ≥ 0.70 |

**Adversarial verification (5 independent agents + synthesis, 2026-06-05; agent-reported scratch
probes banked to `adversarial_verification.json` — NOT a deterministic receipt).** The de-confound,
oracle, and support claims survived hard pressure: linear `u_det` floored across seeds 1/2,
`p_noise=0.25`, and n up to 16k (converges to ~0, never creeps to 0.10); the small-MLP-on-raw 0.095
was overfit noise (a larger net did worse); the oracle is honest (zeroing readouts leaves
event-route `u_acc` intact; decoding a fresh dataset cross-matches at chance 0.274); the codec is
bit-identical; the position-prior is neutralised. Four findings were fixed before this spec: (1) the mask shortcut
→ whole-checkpoint masking + non-degenerate `PSI`; (2) a silent nan-drop → stratified-holdout
fallback + gate-1 fails on incomplete eval + unified support; (3) metric ambiguity → §3.2 pin;
(4) `K≥8` crash → `make_psi`.

**Caveat for downstream (do not mis-cite):** the `≥0.95` oracle is a **structure-aware deterministic
parser** (parity-decode + bounded cumsum), **not** a generic learner — a flat MLP on raw tokens
scores `u_det ~ 0`. Recovery requires sequential parity + integration; that is exactly the bar
gate-4 sets for the JEPA body.

## 6. Primary statistic — flip-conditioned `z_flip_acc` (gate 5)

For each trained body, readout channel `(c, j)`:
```text
train:  linear probe body@c -> z_{c,j} on the all-row train split
score:  held-out accuracy on the subset {x_{c,j} = 1}
z_flip_acc = median over (c, j) of Acc[z_{c,j} | heldout x_{c,j}=1]
```
The probe is **not** trained on flip rows only (training on flips lets a clean-only body learn the
negation). **Support (gate-5 fix):** score flips **pooled across the `n_U` channels per checkpoint**
(~2000 flips total per checkpoint in the receipt; the held-out slice scales with the train/score
split) — per-cell scoring is too thin to resolve a sub-0.15 gap, so pooling is mandatory. Expected:
GEN carries observed `z` → high on flips; JEPA carries `clean = parity(PSI_j, bits(u_c))` → on
flips `clean = not z` → a `z` probe trained on all rows is systematically weak/sub-chance.

## 7. Capacity sweep (gate 6, the chatv2 hook)

`d_model ∈ {128, 256}` (256 = where chatv2's `objective_excess` deflated), 3 seeds. Per `(d,seed)`:
`gap_seed = GEN_z_flip_acc − JEPA_z_flip_acc`; per-`d` headline `gap_d = median_seed`. Non-bottleneck
= `d=256`.

## 8. Kill-gate

`jepa_accumulator_discard_confirmed` requires **all** of:
- gate 4: JEPA `u_det ≥ 0.70` (median over checkpoints), **and**
- gate 5: `gap_256 ≥ frozen_delta = 0.15`, **and**
- GEN positive control live at `d=256`: `median_seed(GEN_z_flip_acc) ≥ 0.65`, **and**
- `gap_256 > std_seed(gap_seed at d=256)`.

**Kill / shelve:** if gate-4 `u_det < 0.70` (as in parity) → `blocked_by_unfaithful_jepa`; the
substrate re-pose did not rescue functional retention → **shelve the lane** (do not scale to real
JEPA). No capacity battery if gate 4 fails.

## 9. Verdict tree (precedence: first match wins)

| branch | condition | reading |
| --- | --- | --- |
| `blocked_by_deconfound_leak` | raw input linearly reads `u_t` (gate 1) | confounded substrate — **cleared by preflight** |
| `blocked_by_absent_functional` | oracle cannot recover `u_t` (gate 2) | functional absent — **cleared by preflight** |
| `blocked_by_mask_shortcut` | masked readout recoverable without `u_t` (mask-necessity) | shortcut substrate — **cleared by preflight (whole-ckpt mask)** |
| `control_void` | `u_null` clears (a Bernoulli null reads as determined) | instrument unreliable; no result |
| `blocked_by_gen_control` | trained GEN fails the positive control (gate 3): `z_flip_acc` low or `u_det < 0.70` | reads/training not live |
| `blocked_by_unfaithful_jepa` | JEPA collapses or `u_det(JEPA) < 0.70` (gate 4) | JEPA did not keep the functional → **shelve** |
| `blocked_by_flip_readout` | gate-5 flip read support-starved or GEN flip control fails at `d=256` | discard read uninterpretable |
| `blocked_by_capacity` | gap `≥ delta` at `d=128` but `< delta` at `d=256` | capacity artifact (chatv2 echo), not a JEPA win |
| `jepa_accumulator_discard_confirmed` | gates 4 + 5 hold and survive `d=256`, controls clean | toy-tier JEPA-native positive |
| `jepa_not_distinguished_from_gen` | controls pass, gate-4 holds, but gap `< delta` (not capacity-explained) | no selective discard on this substrate |

## 10. Required controls

- **`u_det`:** both bodies keep the functional — `u_det ≥ 0.70` for GEN (gate 3) and JEPA (gate 4).
- **`u_null`:** independent Bernoulli target at the count base rate — its determining read must stay
  clean; else `control_void`.
- **Direct `x` read:** report-only failed sidecar (the XOR-derived flip is not a linear body read).
- **JEPA collapse guard:** §3.1.
- **Matched GEN:** same width / budget / optimizer / substrate.

## 11. Tier / integrity

**R1 (toy)** — synthetic accumulator substrate. Does not license real-JEPA language.

**Allowed (only on `jepa_accumulator_discard_confirmed`):**
> On a designed accumulator toy, a JEPA body kept the running bounded count while a generative body
> additionally carried the per-readout private flip; the paired flip-conditioned gap survived the
> capacity rung where chatv2's objective-excess had deflated — against a matched GEN control and a
> passed collapse guard.

**Forbidden:** "JEPA learns a world model / understands"; "proves LeCun right/wrong"; "JEPA is the
path to AGI"; any statement about real I-JEPA/V-JEPA/LLM behaviour; R2/R3 / "more than we know".

## 12. Run order

1. **Operator lock review** (this doc) — especially the §3.1 mask re-pose and the predict-ahead
   alternative.
2. **Model-free preflight — DONE.** `preflight_pass_ready_for_spec` (§5).
   ```powershell
   python scripts/jepa_0d_accumulator_preflight.py --out results/chatv2/jepa-0d-accumulator-preflight
   ```
3. **Runner — BUILT + dev-validated** at `scripts/jepa_0d_accumulator.py` (reuses `TinyGPT` / `_std`
   from `chatv2_phase0_bodyresist`, imports the frozen substrate from
   `jepa_0d_accumulator_preflight` so runner/preflight cannot drift; new: whole-checkpoint mask +
   per-sample target gather, checkpoint `u_det` read at the event-integration position,
   masked-context flip read that keeps the target checkpoint visible, pooled across channels). A
   `--dev` self-test (d=64, 40 steps, ~13s) exercised every path end-to-end (no science, plumbing
   only).
   ```powershell
   python scripts/jepa_0d_accumulator.py --dev --out results/chatv2/jepa-0d-accumulator-dev
   ```
4. **Smoke (operator-staged, ~20 min est.; L=246 ≈ 1.3× the parity L=192 ⇒ ~1.3× the ~15.5-min
   parity smoke).** Clean launch (no `2>&1 | Tee-Object`; verify the process is alive). Stops after:
   de-confound replay, GEN positive control, JEPA collapse guard, JEPA `u_det` gate, one flip-read
   sanity.
   ```powershell
   python scripts/jepa_0d_accumulator.py --smoke --out results/chatv2/jepa-0d-accumulator-smoke
   ```
5. **Only if smoke clears `u_det ≥ 0.70`:** lock battery `{128,256} × 3 seeds` (operator-staged,
   ~2 h est.).
   ```powershell
   python scripts/jepa_0d_accumulator.py --out results/chatv2/jepa-0d-accumulator-lock
   ```
   **Do not run the lock battery if `u_det` fails.**

## 13. Readback checklist

- git + script SHA; substrate config (`K=6, T=12, p_event=0.34, p_noise=0.10, n_U=7, C={4,8,12}, PSI`);
- preflight receipt (gates 1/2/support/mask-necessity) + adversarial-verification note;
- per-`(d, seed)` GEN/JEPA `z_flip_acc` + paired gap; `gap_d`, `std_seed`; capacity table;
- held-out flip counts (pooled-per-checkpoint) + all-row/clean-row `z` accuracies;
- `u_det` (both bodies, accuracy-det, per-checkpoint baseline) + MAE-det sidecar; `u_null`;
- JEPA mask policy = `whole_checkpoint` (locked; `predict_ahead` only via spec amendment),
  `mask_reads`, collapse guard (per-dim std floor, VICReg var/cov receipt, eff-rank);
- `frozen_delta = 0.15`, per-seed spread; branch verdict; allowed/forbidden from §11.

## Locked pins used

- Substrate: `K=6, T=12, p_event=0.34, arity=2, delta=0.45, p_noise=0.10, P_e=5, P_u=3, n_U=7,
  C={4,8,12}, n=3000, L=246`, `PSI` = 7 distinct nonzero 3-bit functionals (rank 3, min-Hamming 4).
- JEPA: whole-checkpoint mask (re-pose), EMA tau=0.99, VICReg 25/25/1 (γ=1.0), 2-layer predictor
  @ `d_model`, embed_dim=`d_model`, `mask_reads=8`, collapse guard (std≥0.10 on ≥90% dims, eff-rank
  ≥ max(8, 0.05·d)), `λ_var=50` single-retry contingency.
- Metric: accuracy-`u_det`, per-checkpoint majority baseline, median over checkpoints; MAE-det
  sidecar (report-only); within-1 det rejected.
- Gates: `u_det ≥ 0.70` (gates 3 & 4); `frozen_delta = 0.15` (gate 5); capacity `{128,256}`, 3 seeds.

---

*Sundog Research Lab — JEPA-0D accumulator spec. DRAFT for operator lock. Preflight passed and
adversarially verified; the decisive gate-4 `u_det ≥ 0.70` is where parity died. R1 toy; kill-gated R&D.*
