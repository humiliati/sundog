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
