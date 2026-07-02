# Chat-v2 H1/V3-0c — Bank-Freeze Receipt (data rung)

> 2026-07-01. Run of `H1_V3_0C_CROSSOVER_SPEC.md` data rung (chess arm; CPU, no model).
> **Non-promotional. No model run, no GPU, no R2 claim.**
> Script: `scripts/chatv2_h1_v3_0c_bank_freeze.py`; log, manifest + witness JSONL in
> `results/chatv2/h1_v3/`.

## Verdict: `F3-V3c/witness` — but the failure is a certificate-availability fact, not a bank fact

The bank itself forms cleanly (29 balanced, floored axes at ply 40; liveness 1.000;
baselines frozen to the manifest). What fails is **per-axis witness certification at bank
scale**, and three escalating searches show it is not a search-effort problem:

| search | strategy | coverage |
| --- | --- | --- |
| V3-0b run | random adjacent swaps, single pass (41 s) | 5/29 |
| V3-0c run 1 | interaction-directed random rotations, full 900 s | **10/29** (22 pairs) |
| V3-0c run 2 | **systematic enumeration** (all change-plies × ±3 rotations, incl. opposite-color partner) | **7/29 — candidate lists EXHAUSTED at 425 s** |

Run 2 is the decisive one: for 22 of 29 axes, sweeping *every* slice instance × *every*
systematic rotation found **no** legal same-bag reordering that flips the axis. At ply 40,
legality chains are long, so the space of legal same-multiset alternates per instance is
tiny, and almost all of them commute.

**Witnessability scan across markers (mechanical, probe-free; 1,400 games):**

| ply | floored+balanced sliced axes | witnessed (150 s, systematic) |
| --- | --- | --- |
| 16 | 1 | 0 |
| 24 | 0 | 0 |
| 32 | 16 | 3 |
| 40 (full-scale runs) | 29 | 7–10 |

Shallow markers cannot form the bank on slices (slices too thin / balance skewed); deep
markers form it but cannot witness it. **No marker supports ≥ 24 witnessed axes.** The
per-axis witness-pair certificate is unattainable at bank scale on real chess under bounded
search — at every marker.

## What survives intact

- **The frozen bank:** 29 axes at ply 40 (floor ≥ 120, balance [0.40, 0.60] on slices),
  surface baselines per axis (probe suite max 0.59–0.90, bulk 0.61–0.72) in
  `v3_0c_bank_manifest.json` — the matched baseline the crossover gate needs.
- **Witness pairs where found** (union of searches, 10+ axes incl. `b.a7`, `w.d2`, `b.d7`,
  `b.f7`, `occ.d7`): genuine marker-ply certificates, retained as evidence-where-available.
- Liveness 1.000; co-ambiguity median 22 / p75 24.

## Amendment proposal A1 (owner adopted after filing)

The witness gate's *role* in V3-0c is to close one loophole: a bag-determined label with an
under-fit probe could fake a crossover. There is a control that closes the same loophole
**at the model level**, with no search-availability problem, evaluated on *every* crossing
axis at the exact marker:

> **Order-shuffle control (H4b).** For each crossing axis, re-extract model features on
> order-shuffled versions of the same instances (same bag, fixed shuffle seed) and re-run
> the readout against the original labels. If the label were a bag function readable by the
> model, accuracy survives shuffling; if the model's representation is order-dependent
> state, it does not. Gate: `acc_model(original) − acc_model(shuffled) ≥ 0.10` required per
> crossing axis.

Under A1: the data rung admits on **floor + balance + frozen baselines** (29 ≥ 24 — met),
witnesses are **reported where found** (not gated), and the model rungs gate on **three**
conditions per axis: crossover ≥ +0.15 vs `surface_max`, ≥ +0.15 vs random-init floor, and
the ≥ 0.10 shuffle drop. Rationale: the shuffle control tests order-sensitivity *directly on
the model representation*, at the marker, on all axes — strictly wider coverage than the
witnessable subset, closing the same inferential gap the witness was for.

**Owner decision after this receipt:** A1 is adopted in
`H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md`. Under A1, the existing 29-axis bank admits as
`H1-V3-A1-DATA-ADMIT`; witnesses remain reported evidence where available, and the
order-shuffle drop becomes a binding model-time control before any V3-1 claim.

If A1 had been declined: `F3-V3c/witness` would have stood as final; the lane's remaining
option would have been bank-R1-and-freeze.

Cross-refs: `H1_V3_0C_CROSSOVER_SPEC.md` (the prereg; §2 witness role),
`H1_V3_0B_SLICE_ADMISSION_RECEIPT.md` (the commuting-swaps finding),
`H1_V3_A1_ORDER_SHUFFLE_CONTROL_SPEC.md` (owner-adopted amendment),
`results/chatv2/h1_v3/witnessability_scan.log`, `PROMOTE_GATE.md` (unchanged).
