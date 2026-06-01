# ARC Phase 4 — Body-Resistance / Low-Dimensional Collapse Probe

Parent / context specs:

- [`../../CROSS_SUBSTRATE_NOTES.md`](../../CROSS_SUBSTRATE_NOTES.md) (body-resistance axis; §8)
- [`../../SUNDOG_V_ARC.md`](../../SUNDOG_V_ARC.md) (Phase 4 = "5D / low-dimensional collapse check")
- [`PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md`](PHASE3E_RELATIVE_LOCALITY_CERTIFICATE_SPEC.md)
- [`PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md`](PHASE3_BRANCH_E_PROGRAM_SEARCH_SPEC.md)
- [`PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md`](PHASE0_CONTEXT_EXPANSION_FOR_FIBERS_SPEC.md)

Drafted: **2026-05-29 (PT)**

Status: **DRAFT ONLY; EXECUTION NOT ADMITTED; TOOLING NOT FROZEN.** This proposes
the body-resistance reading of the ARC substrate (the roadmap's Phase 4
low-dimensional-collapse check), sharpened by the cross-substrate body-resistance
axis. A later freeze-marker amendment must add runner, wrapper, npm wiring,
leak-check receipt, smoke fingerprint, and timing before any binding run.

## Cross-Substrate Context (why this lane, now)

`CROSS_SUBSTRATE_NOTES.md` §8 reframed the portfolio around one load-bearing axis,
**body-resistance**: a high-dimensional body genuinely resists its low-dim shadow
when `dim(body) >> dim(shadow)` and the shadow *cannot* reconstruct the state.
Three-for-three, every *measurable control* substrate is **marginal** on this axis
— 2D Navier-Stokes C1 (`FVE(body|shadow) ~ 0.99`), Mesa `net.7` (participation
ratio ≈ 2.0 of 256), and the Sabra shell (effective rank ≈ 1.7 of 30) — because
each has an intrinsically low-dimensional state. The note's explicit conclusion is
that a sharp control regime-2 needs a **genuinely high-dimensional computational
body** (high-Re turbulence behind the numerical wall, or high-dim RL/LLM agents) —
**not currently in the portfolio**.

ARC is a candidate for that gap: the task body is a grid (up to 30×30 over 10
colors), a far larger state than `net.7` or a low-Fourier attractor. The Phase 3E
certificate program already showed the `signature_palette` *shadow* has no usable
fiber or rank locality (four negatives), and Branch E showed the *body* is modestly
solvable by deterministic program search (capability floor-clear). What has **not**
been measured is the body's own **intrinsic dimensionality** — whether the ARC
grid-state collapses to a low-dim manifold like Mesa `net.7`, or is genuinely
high-dimensional. That is this probe (and the roadmap's long-standing Phase 4).

## Core Question

Across the registered public-training ARC contexts, is the **raw-grid body**
representation **genuinely high-dimensional** (effective rank / participation ratio
far above the marginal substrates' ≈ 2), and does it **fail to be reconstructed**
from a matched-dimension shadow (low held-out `FVE`)? If yes, ARC is the
portfolio's first high-dimensional *computational* body on the body-resistance
axis. If it collapses to low dimension, ARC joins the three-for-three marginal
column.

## Honest Scope (pre-registered caveats)

This probe is read carefully, not over-read:

1. **Read-off, not control.** This measures the body's intrinsic dimensionality
   (a read-off property, like cap-set / unit-distance). It is **not** a
   control-regime-2 witness — that needs a low-dim shadow demonstrated
   *control-sufficient yet state-insufficient*. The Phase 3E certificate program
   already tested `signature_palette` as that control-shadow and it failed
   (no usable locality; Branch E could not select with it). So a positive here
   establishes the high-dim **body**; it does not supply the missing
   *control*-sufficient shadow.
2. **The naive `FVE(grid|signature_palette)` is a trivial baseline.** The
   `signature_palette` shadow is a deliberately coarse summary (palette, shape,
   density) that discards spatial layout, so its reconstruction `FVE` is expected
   low **by construction** — reported as a baseline, never the verdict driver. The
   verdict is driven by the body's effective rank and the **matched-dimension**
   reconstruction curve.
3. **Sample-bounded.** Participation ratio is bounded by the number of contexts
   (`U_all`, a few hundred). The probe reports the bound and reads "high-dim" only
   relative to it and to the marginal substrates, never as an absolute claim.
4. Not a Blackwell-sufficiency proof, ARC solve, public-evaluation result, or
   Kaggle claim. Phase 6 remains the only public-evaluation gate.

## Frozen Inputs

```text
primary_register   = docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv
primary_split      = sha256_expansion
diagnostic_register = docs/prereg/arc/P0_TASK_REGISTER.csv
U_all              = validation_lodo ∪ validation_pttest ∪ test_lodo ∪ pttest
```

Body and shadow representations are inherited unchanged from the Phase 3E runners
(`feature_vector` / `represent_grid`), training-split only:

- **body** = `raw_grid` representation of the query grid (the high-dim state).
- **registered shadows** = `signature_palette`, `signature_only`, `metadata_only`
  (the coarse summaries already in the program).
- **matched-dim shadow** = the top-`k` principal components of the body itself,
  with `k = dim(signature_palette)` (the body's own best `k`-dim summary — the
  fair "can ANY k-dim shadow reconstruct it" test).

## Frozen Measurements (ported from C1 / Mesa)

For the body representation matrix over `U_all` (rows = contexts, columns =
features), all on a frozen train/held-out split:

1. **Effective dimensionality** — participation ratio
   `PR = (Σλ_i)² / Σλ_i²` and the 90%/95%/99% energy ranks of the body covariance
   (the Mesa `net.7` PR port). Reported against the marginal substrates' PR ≈ 2.
2. **Shadow reconstruction `FVE(body | shadow)`** — held-out coefficient of
   determination of a ridge regression predicting the body from each registered
   shadow (the C1 / Mesa `FVE` estimator). The `signature_palette` row is the
   coarse baseline (caveat 2).
3. **Matched-dim reconstruction curve** — held-out `FVE(body | top-k PCA of body)`
   for a frozen grid of `k` (including `k = dim(signature_palette)`): how much of
   the body a `k`-dim summary recovers. The sharp-vs-marginal driver.

All estimators, the ridge penalty, the PCA, and the train/held-out split are
frozen here; none is tuned after seeing the numbers.

## Branches

| branch | condition | interpretation |
| --- | --- | --- |
| `arc_body_high_dim` | Body PR ≫ the marginal-substrate band (PR ≥ a frozen multiple of 2 **and** ≥ a frozen fraction of the sample bound) **and** matched-dim `FVE(body | top-k=dim(signature) PCA)` ≤ a frozen reconstruction ceiling. | The ARC grid body is genuinely high-dimensional — the portfolio's first high-dim *computational* body; the missing-control-shadow question stays open. |
| `arc_body_marginal` | Body PR within / below the marginal band (≈ 2), **or** matched-dim `FVE` above the ceiling (a `k`-dim summary reconstructs the body). | ARC's body collapses to low dimension like Mesa `net.7` — it joins the three-for-three marginal column. |
| `arc_body_inconclusive` | Neither cleanly holds (e.g., PR moderate, reconstruction borderline, or sample-bound dominates). | The dimensionality reading is ambiguous at this register size. |

The exact PR multiple, sample fraction, and `FVE` ceiling are frozen in the
freeze-marker amendment before any binding run, calibrated only to the marginal
substrates' published numbers (PR ≈ 2; `FVE ≈ 0.99`), never to ARC output.

## Required Artifacts

Binding output path: `results/arc/phase4-body-resistance/`

- `manifest.json`; `split.csv`; `body_spectrum.csv` (eigenvalues, PR, energy
  ranks); `shadow_fve.csv` (FVE per registered shadow); `matched_dim_fve_curve.csv`
  (FVE vs k); `per_lane_dimensionality.csv`; `phase4_body_resistance_receipt.json`;
  `branch_adjudication.md`; `commands.md`; `hashes.json`.

The manifest records register hash, split, body/shadow representation hashes, the
frozen estimator parameters + thresholds, source-code hashes, and the branch.

## Reserved Implementation Names

- Python runner: `docs/prereg/arc/phase4_body_resistance.py`
- Node wrapper: `scripts/arc-phase4-body-resistance.mjs`
- npm script: `arc:phase4:body-resistance`
- receipt path: `results/arc/phase4-body-resistance/`

This probe is a spectral + ridge-regression measurement over a few hundred
contexts — expected well under the ten-minute rule (no model training), so the
binding run should run inline once the freeze marker is filed.

## Public Language

Allowed before a binding receipt:

> "ARC Phase 4 has a draft body-resistance probe: does the ARC grid body collapse
> to low dimension like the marginal control substrates, or is it genuinely
> high-dimensional? It ports the C1/Mesa participation-ratio + FVE estimators. No
> tooling freeze marker or receipt exists yet."

Allowed if `arc_body_high_dim`:

> "The ARC grid body is genuinely high-dimensional on the cross-substrate
> body-resistance axis — the first high-dim computational body in the portfolio.
> This is a read-off dimensionality result; it does not supply a control-sufficient
> low-dim shadow, prove Blackwell sufficiency, solve ARC, or make any
> public-evaluation / Kaggle claim."

Allowed if `arc_body_marginal`:

> "The ARC grid body collapses to low dimension like Mesa net.7 and the marginal
> control substrates — it does not provide the high-dim body the cross-substrate
> program needs."

Forbidden:

- reading a high-dim body as a control-regime-2 witness, Blackwell sufficiency, or
  an ARC solve;
- reading the coarse `signature_palette` reconstruction FVE as the verdict;
- any public-evaluation or Kaggle claim;
- tuning the PR multiple, sample fraction, FVE ceiling, ridge penalty, or split
  after seeing the body spectrum.

---

## Amendment 1 — Freeze Marker (2026-06-01 PT)

Status: **EXECUTION ADMITTED; TOOLING FROZEN.** Tooling, thresholds, leak
receipt, smoke fingerprint, and timing are pinned below. Nothing here is tuned
to ARC output (the binding run has not been read at freeze time).

### Frozen tooling

| component | path | sha256 |
| --- | --- | --- |
| runner | `docs/prereg/arc/phase4_body_resistance.py` | `5ca151e11f654ac4fef1fadb60ce1b766c3ed7ea9ff9c23822049c421e639149` |
| wrapper | `scripts/arc-phase4-body-resistance.mjs` | `de6514e570f6258073bca3a47ef32f3f85b2c1e510871f9ea652054a5b1c5d1a` |
| representations (imported, unmodified) | `docs/prereg/arc/phase3d_mask_target_v3.py` | `9f6b4ba7931a08f0a7d971e8758a04c63840b3bb92433d41b89b523955415329` |
| loaders/IO (imported, unmodified) | `docs/prereg/arc/phase3_branch_e_program_search.py` | `036ea6a99d5b3d23845ecab3cbaf2370e9b88f42f39c6c7d717e1db62946d32a` |

npm: `arc:phase4:body-resistance`. Result path (gitignored):
`results/arc/phase4-body-resistance/`. All linear algebra is `torch.linalg`
(numpy is not installed). Body = `raw_grid_onehot` (9900-dim); low-dim shadow =
`metadata_vector` (28-dim); coarse baseline shadow = `feature_vector(...,
signature_palette_edit_mask_v3)` (4124-dim nominal — caveat 2). All inherited
byte-for-byte from the Phase 3E runners.

### Frozen thresholds (calibrated only to the marginal substrates, never to ARC)

```text
PR_HIGH_MIN              = 20.0     # >= 10x the marginal-substrate PR ~ 2 (Mesa net.7 / C1 / Sabra)
PR_MARGINAL_MAX          = 5.0      # ~ 2.5x the marginal band
FVE_RECON_CEILING        = 0.90     # high_dim requires matched-dim FVE <= this
FVE_MARGINAL_MIN         = 0.95     # marginal if matched-dim FVE >= this (marginal substrates ~ 0.99)
PR_BOUND_SATURATION_MAX  = 0.90     # PR/bound above this -> inconclusive (sample-saturated)
RIDGE_LAMBDA             = 1.0
SHADOW_DIM_K             = 28       # matched-dim PCA cut (see operationalization note)
heldout split            = sha256(instance_id) % 10 < 3   (~30% held out)
ENERGY_LEVELS            = [0.90, 0.95, 0.99]
PCA_K_GRID               = [1, 2, 5, 10, 28, 50, 100, 200]
```

Adjudication (frozen): **inconclusive** if `PR / min(n_features, n_contexts-1) >
0.90` (sample-saturated guard fires first); else **high_dim** if `PR >= 20.0`
**and** matched-dim `FVE(body | top-28 PCA) <= 0.90`; else **marginal** if `PR <=
5.0` **or** matched-dim `FVE >= 0.95`; else **inconclusive**.

### Two pre-registration operationalizations (decided before reading ARC output)

1. **`SHADOW_DIM_K = 28`, anchored to the low-dim metadata shadow.** The spec body
   text (Frozen Inputs) names the matched-dim cut `k = dim(signature_palette)`. The
   `signature_palette` vector's *nominal* dimension is 4124, but that is a sparse
   SHA-bucket hash embedding of low-information object signatures; its top-`k` PCA
   at `k = 4124` over `<= 491` contexts is a degenerate **full-rank** reconstruction
   (`FVE = 1` by construction, rank `<= n-1`), which would make the matched-dim test
   vacuous. The cut is therefore anchored to the **genuine low-dim shadow's
   dimension** — the 28-dim `metadata_vector` (palette/shape/density), the direct
   analogue of the 5-D Mesa shadow and the low-Fourier C1 shadow. `k = 28` asks the
   fair question "can the body's *own* optimal 28-dim linear summary reconstruct it,
   and does it beat the hand-built 28-dim metadata shadow?" The full `PCA_K_GRID`
   curve (through `k = 200`) is reported regardless, so nothing is hidden.
2. **Sample-bound clause operationalized as a strict saturation guard.** The branch
   table lists `... or sample-bound dominates -> inconclusive` (caveat 3). This is
   frozen as: if `PR` exceeds 90% of its sample/feature bound `min(n_features,
   n_contexts-1)`, the covariance is near-isotropic at this register size and the
   reading is unreliable -> `inconclusive`. The guard only **removes** high-dim
   verdicts (makes high_dim *harder*), never adds them; `PR / bound` is reported.

### Leak receipt + smoke fingerprint

- `npm run arc:phase0:leak-check`: **0 fail / 0 warn**; register 36 training / 0
  evaluation-blind / 0 evaluation; 26 ARC scripts scanned for eval literals (incl.
  the new `arc-phase4-body-resistance.mjs`); no Kaggle scaffolding.
- `py_compile`: clean. Dry-run (`--dry-run --allow-dirty`) wrote all **10**
  artifacts (`manifest.json`, `split.csv`, `body_spectrum.csv`, `shadow_fve.csv`,
  `matched_dim_fve_curve.csv`, `per_lane_dimensionality.csv`,
  `phase4_body_resistance_receipt.json`, `branch_adjudication.md`, `commands.md`,
  `hashes.json`).

### Timing

Pure spectral + ridge measurement over `U_all` (~491 contexts): one centered SVD
of a `~491 x 9900` matrix, two ridge solves (`28-` and `4124-`dim shadows), and an
8-point PCA reconstruction curve. No model training — expected **well under the
ten-minute rule**, so the binding run executes **inline** on a clean worktree (no
shard/stage needed).

### Staged binding command

```powershell
cd C:\Users\hughe\Dev\sundog
node scripts/arc-phase4-body-resistance.mjs `
  --data-dir "$env:USERPROFILE\Datasets\ARC-AGI-2\data" `
  --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv `
  --split-mode sha256_expansion `
  --out results/arc/phase4-body-resistance
```

(`SUNDOG_PYTHON` may pin the interpreter; the runner refuses a dirty worktree
unless `--allow-dirty`, so the freeze marker is committed before the binding run.)

---

## Amendment 2 — Binding Verdict (2026-06-01 PT)

**Branch: `arc_body_inconclusive` — directionally non-marginal, below the frozen
high-dim bar at this register.**

Binding run: `gitCommit 1892CBD4` (the Amendment-1 freeze commit; ARC subtree
clean — the manifest `gitDirty=true` reflects only concurrent **non-ARC** sibling
lanes, not any ARC-lane file), `runnerSha256 5CA151E1…` (matches the frozen
runner), `splitMode sha256_expansion`, `registerHash 1EA51F10…`, `dataDirHash
73347BCD…`. `U_all = 491` contexts (val_lodo 118 / val_pttest 37 / test_lodo 259 /
pttest 77), 156 held out.

### Measured (frozen estimators, read once)

| measure | value | marginal-substrate reference |
| --- | --- | --- |
| body participation ratio (PR) | **9.15** | Mesa `net.7` ≈ 2.0; NSE-C1 ≈ low; Sabra ≈ 1.7 |
| top-1 energy fraction | 0.287 | marginal substrates: one mode dominates |
| energy ranks 90 / 95 / 99 % | 96 / 159 / 293 | of ≤ 490 |
| matched-dim `FVE(body | top-28 PCA)` (held-out) | **0.659** | Mesa: 5-D reconstructs ≈ 0.97–0.99 |
| `FVE(body | top-k PCA)` curve | k=1→0.299, 10→0.575, 28→0.659, 50→0.702, 100→0.749, 200→0.789 | — |
| `FVE(body | metadata 28d)` (baseline, caveat 2) | 0.399 | — |
| `FVE(body | signature_palette 4124d)` (baseline, caveat 2) | 0.431 | — |
| PR / sample bound | 0.019 (not saturated) | guard 0.90 |
| per-lane PR | 6.55 / 5.99 / 8.70 / 7.90 | each lane individually ≫ marginal |

### Why inconclusive (the frozen gate, not retuned)

- `PR 9.15` sits **between** `PR_MARGINAL_MAX 5.0` and `PR_HIGH_MIN 20.0`: it
  clearly clears the marginal band (not `arc_body_marginal`) but does **not** reach
  the pre-registered 10×-marginal high-dim bar (not `arc_body_high_dim`).
- The matched-dim FVE clause *is* satisfied for high-dim (0.659 ≤ 0.90 ceiling; and
  no `k`-dim summary through k=200 reconstructs the body past 0.79) — but the gate
  is an **AND**, and the PR clause fails. So the verdict falls through to
  inconclusive. **The thresholds are not retuned after seeing the spectrum** (that
  would be the p-hack the freeze marker exists to prevent).

### What this does / does not establish

- **Does:** ARC's raw-grid body is the **least-marginal substrate measured to date**
  — PR ≈ 4.6× the three control substrates, no dominant mode (top-1 energy 0.287),
  and **reconstruction-resistant**: a 28-dim summary recovers only 66 % and even a
  200-dim summary only 79 % of the held-out body (cf. Mesa `net.7`, where a 5-D
  shadow reconstructs ≈ 0.97–0.99). On the *reconstruction* axis the ARC body does
  resist low-dim collapse. So the portfolio updates from "three-for-three marginal"
  to "three **control** substrates marginal; the ARC **computational** body is
  materially more dimensional, but below the frozen high-dim bar at 491 contexts."
- **Does not:** clear the high-dim threshold, so it is **not** a claim that ARC is a
  high-dim body; and it is **not** a control-regime-2 witness, Blackwell
  sufficiency, ARC solve, or any public-evaluation / Kaggle claim (Phase 6 only).
- **Sample-limited (caveat 3), direction noted:** 99 % of body energy needs 293 of
  ≤ 490 components — a broad spectrum whose PR is plausibly an **under**-estimate of
  the population PR at a larger register. The sample bound caps PR, so the
  shortfall-from-20 points toward "register too small to clear the bar," not toward
  "marginal." This is recorded as a direction, **not** a verdict.

### Honest next move (if reopened)

A frozen **Phase 4 v2** could test whether PR clears the (unchanged) `PR_HIGH_MIN =
20` bar on a **larger context register** (more registered training tasks → more
contexts → a less sample-limited PR), with the per-lane PR + the matched-dim curve
as the same frozen readouts. The bar stays 20; only the sample size grows. This is
the disciplined escalation — never lowering the threshold to the observed 9.15.

### Public language (inconclusive — addition to the spec list)

Allowed:

> "ARC Phase 4 (body-resistance) returned **inconclusive but directionally
> non-marginal**: the raw-grid body's participation ratio (≈ 9.1) is several times
> the three marginal control substrates' (≈ 2), with no dominant mode and a
> reconstruction-resistant spectrum (a 28-dim summary recovers only ~66 %), but it
> does not clear the pre-registered 10×-marginal high-dim threshold at the
> 491-context register. It is the least-marginal substrate measured; it is not a
> high-dim claim, a control witness, an ARC solve, or any eval/Kaggle result."

Still forbidden: reading 9.15 as high-dim by lowering the bar; any control-witness /
Blackwell / solve / eval / Kaggle claim.
