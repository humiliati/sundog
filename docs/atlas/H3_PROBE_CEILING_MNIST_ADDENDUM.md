# H3-PC-B Pre-registration ADDENDUM — the probe-ceiling battery on the real-body MNIST-rotation shadow

**STATUS: FROZEN 2026-06-11** (adversarially reviewed pre-freeze, agent `ab916f33277d04c2d`: blocking
BD-1–BD-7 applied — blindness-excuses-silence-never-speech, both-bars ceiling counting, two-instance
RNG pinning, loader-kwargs + data-hash identity receipts, joint-separation gate, best-effort-stream /
gates-are-binding split, linearly-readable-anchor scoping — plus N-1–N-9. No edits below this line
after this stamp — addenda only.)

> Second leg of HS4 (slate 2026-06-10), named-but-fenced in `H3_PROBE_CEILING_PREREG.md` §7. Audits
> the banked Substrate-B claim of `H3_POOLED_SHADOW_RESULT.md` ("GAP only *partially* attenuates θ")
> with the leg-1 calibrated battery. **BINDING once frozen.** NOT public-eligible. A clean null is a
> SUCCESS; forward-generate only. Language rule: probe-access asymmetries in a ground-truth substrate —
> no claims about introspection as a mental phenomenon. Attribution as leg 1 (amnesic probing;
> V-information; LEACE; KSG) + the H3 Substrate-B leg (`scripts/shadow_pooled_mnist.py`).

## 1. The question

Banked (`results/atlas/h3/mnist_result.json`, exact command
`python scripts/shadow_pooled_mnist.py --n 12000 --epochs 20 --probe-n 1800` — note `--probe-n 1800`
is NON-default; the default 2000 would have given a different split): a TinyCNN trained to classify
digit y under rotation θ~U[−30°,30°] has GAP-shadow θ-R² = **0.342** (plain scaler+ridge — the 32-dim
GAP is below the substrate's PCA cap; PCA-128 applies only to the 1568-dim pre-pool) vs pre-GAP
**0.623** (dim-fair PCA-128 ridge) — read as "GAP partially attenuates θ." The result doc also cites "~0.47 under a strong nonlinear probe" — **that number is NOT
in the banked JSON** (it came from the v1 workflow's verification pass); this addendum anchors to the
JSON and treats 0.47 as an un-receipted doc-level claim to be superseded by the battery's measurement.
Leg 1 showed the lab's weak-battery silences certify little (its banked MLP was MEMBER-BLIND; ridge at
N=2000 under-read by ~10×). The open question here: **what is θ's true recoverability ceiling from the
GAP shadow, and does the attenuation claim survive a calibrated battery?**

## 2. Fixed substrate (re-derived, with continuity gates — NOT assumed byte-identical)

No model/reps were saved from the banked run; the CNN is **re-derived** by the banked protocol with
determinism hardening (`torch.set_num_threads(1)`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1` before
numeric imports). The re-derived body is THIS leg's object; byte-match with the banked run is NOT
asserted — continuity gates stand in for it:

- **Data**: `fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False,
  parser="liac-arff")` — kwargs VERBATIM (newer sklearn defaults to the pandas parser; row order must
  come from the same cached-ARFF parse) — served from the existing local cache
  (`~/scikit_learn_data/.../mnist_784.arff.gz`). Apparatus gate: `X.shape == (70000, 784)` else VOID.
  **The 8×8 `load_digits` fallback is FORBIDDEN** — if the cached fetch fails, VOID (different
  substrate, not a result). The result JSON records `sha256(X.tobytes())` + the cached `.arff.gz`
  hash as the forward-going image-identity receipt (the banked run recorded no hash; identity
  relative to it rests on cache + verbatim loader + the continuity gates below, and is claimed only
  to that strength).
- **RNG pinning (two FRESH `RandomState(1234)` instances, as in the banked script — NOT one
  continuing stream):** instance #1 (loader) → `permutation(70000)[:12000]` = the subsample;
  instance #2 (main) → `uniform(−30, 30, 12000)` = the banked θs (attached in subsample order), then
  → `shuffle(y_perm)` → `shuffle(θ_perm)` (the banked control consumption, mirrored exactly).
  Split = `torch.randperm(12000, generator=manual_seed(1234))` (dedicated generator,
  byte-reproducible): probe 1800 / train 10200.
- **Training (best-effort stream pin; continuity gates are the SOLE binding acceptance):**
  TinyCNN(**ch=16**), Adam lr=1e-3, batch 128, **epochs 20**; global-torch consumption order pinned as
  `manual_seed(1234)` → TinyCNN init → training-loop `randperm`s, with NO other global-torch consumers
  in between. The banked run was BLAS-unpinned, so its float trajectory is unreachable in principle —
  "exactly as banked" is NOT claimed for the trained weights; the gates below accept the body.
- **Rep extraction (pinned):** `model.eval()` (BatchNorm running stats) + `torch.no_grad()` for pool,
  test, AND continuity reps; eval-mode extraction is batch-size invariant, so batched forward passes
  are free.
- **Continuity gates (VOID if any fails)**: held-out CNN acc ∈ 0.83 ± 0.05; GAP θ-R² under the
  substrate's own frozen probe on the banked 1800 probe set ∈ 0.342 ± 0.10; permuted-θ control
  ≤ 0.05 in |R²|; **joint-separation gate: PRE_pool − BASE ≥ 0.15** (BASE per §5; if the re-derived
  body's pre/post gap collapses, the (a)/(b)/(c) bars cross — apparatus failure, not a result).
- **NEW disjoint evaluation draws** (the battery never touches the banked 1800 probe set): pool =
  images at banked-permutation indices **[12000:32000)** (20,000 unseen images), test = indices
  **[32000:42000)** (10,000) — disjoint from training and from each other by construction. Pool θ
  seed **101235** (SEED+100001), test θ seed **111235** (SEED+110001), both U[−30,30], rotation via
  the substrate's `rotate_batch`. Reps = the re-derived model's 32-dim GAP on these images.
- **PRE anchor (apparatus quantity, measured before any battery readout)**: pre-GAP θ-R² on the POOL
  under the substrate's frozen dim-fair probe (flattened pre-pool → PCA-128 ridge) = **PRE_pool**;
  banked analog 0.623; continuity expectation, not a gate.

## 3. The frozen battery + selection/replication protocol

Identical to leg 1 §3 (P1 ridge α=1.0; P2 MLP(128,64, mi=600, rs=0); P3 kNN k∈{5,10,20,50,100};
P4 Nyström(m=2000, rs=71235, γ∈{0.001,0.01,0.1,1.0})×Ridge(α∈{0.1,1,10}); P5 HistGBT(rs=0,
lr∈{0.05,0.1}, mi∈{200,500})), target = θ (standardized for injections; raw for R²). Same
conventions: StandardScaler refit per dataset; 5-fold KFold(rs=0) pool-CV selection (argmax per
family, ties first-in-grid); replication = fit on full pool, scored ONCE on the frozen test split;
**unclipped R² everywhere**. Learning-curve leg (non-gating): best member + P1 at n ∈ {2000, 5000,
10000, 20000} (subsample rng **91235**).

## 4. Delta-injection calibration (floors on a NONZERO baseline)

Leg 1's floor machinery assumed baseline ≈ 0; here the GAP carries real θ-signal (banked ridge 0.342).
Floors are therefore **deltas above the real baseline**: `z′ = z + α · g(θ) · v`, `g(θ)` = θ
standardized by pool stats, `v` = fixed random unit vector in R³² (`default_rng(SEED+120001=121235)`).

- α set by bisection (bracket [0,3], 28 iters) so that **Δridge ≡ ridge-CV(z′) − ridge-CV(z)** ≈
  {0.05, 0.10}, each within ±0.01. Calibration gate = bisection converges; else VOID.
- **Per-member liveness at level δ**: ANY grid config with CV_m(z′_δ) − CV_m(z, same config) ≥ 0.5·δ
  (same config = identical grid cell, both CVs under the battery convention). Member live at neither
  δ ⇒ **MEMBER-BLIND** (to linear deltas). **Blindness excuses SILENCE, never speech:** a blind label
  only voids the evidential weight of that member's low readings; it does NOT exempt the member from
  outcome (b)'s band (a high-reading member mislabeled blind — possible here via R² concavity on a
  signal-bearing baseline — must still block (b)). Floor = weakest live δ. P1's floor is 0.05 by
  construction. The test split is never injected.
- **The MI leg is DESCRIPTIVE-ONLY in this addendum** (non-gating): with a real nonzero-MI baseline,
  "within the shuffled null" can never fire, and a gate would be vacuous. KSG (k=5, subsample 5000,
  rng 86235; shuffle seeds from 81235, **49 shuffles** — trimmed, it gates nothing; rank rule) is
  reported for the real reps and the **0.10-δ injection only**, at PCA-k ∈ {2,4,8,16,32}, as context
  for the ceiling.
- Positive control: P1 on the real pool reps must read CV θ-R² ≥ 0.20 (banked 0.342 − generous
  drift tolerance; this is also caught by the §2 continuity gate on the banked probe set). Else VOID.

## 5. Pre-registered outcomes (precedence top-down; thresholds anchored, not CV-anchored)

Let **PRE_pool** = the §2 pre-GAP anchor (measured first; banked analog 0.623) and
**BASE** = P1 pool-CV on the real GAP reps (banked analog 0.342).

| # | Outcome | Condition (battery on the real GAP reps) | Reading |
|---|---------|------------------------------------------|---------|
| V | **VOID** | data-source/fetch fails (fallback forbidden), any §2 continuity gate fails, calibration fails, or the positive control fails | apparatus; fix and re-run; not a result |
| a | **ATTENUATION ILLUSORY** | some member: pool-CV ≥ **PRE_pool − 0.05** AND split ≥ **0.8 × (PRE_pool − 0.05)** | the banked P-B1 "partial resist" was wholly probe-relative **against its own anchor**: the GAP attenuates nothing of the **linearly-readable pre-GAP content** (PRE_pool is a PCA-128 ridge reading; nonlinearly-encoded pre-GAP content above it is out of scope) — correction owed to the result doc's Substrate-B section |
| b | **ATTENUATION BAND CERTIFIED** (clean-null SUCCESS) | **EVERY member, MEMBER-BLIND included**: pool-CV ≤ **BASE + 0.05** AND split ≤ BASE + 0.05 (blindness never exempts a reading from the band — §4) | the banked partial-attenuation is REAL down to each live member's δ-floor (battery- and linear-direction-relative): no probe family recovers meaningfully more θ than the banked linear readout; the un-receipted "~0.47 strong-probe" number is superseded downward |
| c | **BOUNDED-PARTIAL (ceiling raised)** | anything else | named outcome; counted set = members with **pool-CV ≥ BASE + 0.05 AND split ≥ BASE + 0.05** (both, per leg-1 fluctuation semantics); **ceiling = max split R² over the counted set**; if the counted set is empty, sub-outcome **CV-ONLY-EXCEEDANCE** with ceiling = measured BASE. Any member with pool-CV ≥ PRE_pool − 0.05 but split below its bar is UNREPLICATED-POSITIVE → mandatory re-run addendum. Reading: attenuation real but the resist band is **[ceiling, PRE_pool]** (upper end = linearly-readable anchor, per (a)'s caveat); result-doc numbers re-scoped. **Adjudication of the un-receipted 0.47:** ceiling < 0.42 ⇒ "0.47 superseded downward"; ∈ [0.42, 0.52] ⇒ "0.47 confirmed by a calibrated battery"; > 0.52 ⇒ "0.47 superseded upward" |

Multiplicity: thresholds frozen here; the split touched once; (a) anchored to the pre-registered
formula, not to observed CV (leg-1 reviewer finding B4 carried over). **Expected-branch disclosure
(reviewer N-1):** given leg 1's Nyström-beats-ridge margin and the in-hand v1 number ~0.47, outcome
(c) is the expected branch; the partition is kept because the (b)/(c) joint IS the falsification
joint for the banked claim, and the deliverable either way is the measured ceiling. Convention note
(reviewer N-4): battery pool-CV uses scale-once-per-dataset KFold(rs=0); PRE_pool uses the
substrate's per-fold pipeline KFold(rs=1234) — an order-0.01 mismatch inside the 0.05 margins,
acknowledged.

## 6. Determinism, files, commands

- Thread pinning + seed ledger: banked substrate streams reproduced as pinned in §2 (np instance #1 +
  #2 both 1234, with instance-#2 consumption order uniform → y-shuffle → θ-shuffle; torch split
  generator 1234; global-torch order manual_seed(1234) → init → randperms). NEW seeds {101235,
  111235, 121235} + carried {71235, 81235, 86235, 91235} — **the disjointness claim is scoped to the
  new seeds** (1234 is intentionally reused across np/torch engines by the banked substrate).
  Auxiliary fixed states, enumerated: battery CV KFold(rs=0); substrate-probe KFold + PCA rs=1234
  (inside `probe_theta_r2`/`_maybe_pca`); MI-leg PCA rs=0; sklearn member rs=0 (P2/P5) and
  Nyström rs=71235.
- Script: `scripts/h3_probe_ceiling_mnist.py` → `results/atlas/h3/probe_ceiling_mnist_result.json`
  (records the data hashes (§2), PRE_pool, BASE, floors, MEMBER-BLIND list, readouts, descriptive MI,
  verdict letter, sub-outcomes incl. the 0.47-adjudication band; unclipped R²). Runtime estimate:
  retrain ~2 min; rotations ~1 min; PRE_pool probe ~5 min; calibration ~3 min; liveness + battery
  40–90 min (MLP + the 12-config Nyström grid dominate); learning curve 10–30 min; descriptive MI
  ~30 min (trimmed). **Total ≈ 1.5–3 h CPU.** Pre-pool extraction is BATCHED (eval-mode invariant;
  a single 20k forward would transiently allocate ~1.6 GB at conv1).
- Frozen test: `scripts/test_h3_probe_ceiling_mnist.py` — reduced size = **the FIRST 4000 / 2000 rows
  of the full pool/test index ranges** (subset rule, binding); retrains the body ONCE per process and
  reuses its reps for the in-process battery rerun; pins continuity gates, calibration convergence,
  liveness machinery, byte-identical readouts + battery-only letter (= the §5 letter evaluated
  without the MI block, which is descriptive anyway) across the rerun; full-run letter equality NOT
  asserted.
- Commands (exact, unchanged): `python scripts/h3_probe_ceiling_mnist.py` ·
  `python scripts/test_h3_probe_ceiling_mnist.py`. Existing suites stay green
  (`test_shadow_pooled_synthetic_v2.py`, `test_h3_probe_ceiling.py`).

## 7. Honest boundaries (pre-stated)

- The re-derived CNN is a NEW body passing continuity gates, not the banked bytes; all certificates
  attach to it (and the banked numbers are quoted as the continuity anchor, nothing more).
- Floors are linear-direction deltas in the 32-dim GAP space; nonlinear residues below every live
  member's sensitivity remain logically possible (leg-1 boundary carried verbatim).
- One substrate, one body, one nuisance (rotation); no claim beyond this CNN/MNIST configuration.
- Verdict (a) would NOT touch the determine half (y-acc 0.87 banked) — only the θ-resist clause.
- The non-monotone banked sweep (0.342→0.394→0.146) is out of scope here; this addendum audits the
  STATIC post-GAP ceiling only.
