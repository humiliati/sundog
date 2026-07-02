# AT-1 — Palinstrophy Boundary-Layer Probe (frozen spec)

> 2026-07-02. Lift of slate entry AT-1 (`NSE_ATTRACTOR_TAIL_HYPOTHESES.md`) into its
> frozen pre-registration, under the harness scope granted by `AT1_HARNESS_SIGNOFF.md`.
> **Everything in §2–§5 is frozen before any margin curve is read.** Non-promotional;
> no ∞-dim NSE claim; the C1 separation and banked receipts are not reopened.
> Anomaly under test (banked, 2026-05-31 discriminator §12): palinstrophy at G=200 is
> Φ_K-predictable (R² = 1.0000) yet control-insufficient (a_mm = 0.195, NEG-A),
> flipping POSITIVE at G=300.

## 1. Rungs & commands

- **Rung 1 (owner-run lock, ~45 min):** re-run the frozen discriminator preset with the
  additive AT-1 export path (harness change per sign-off; no-export semantics untouched):
  `python scripts/pde_c1_kolmogorov_cell.py --preset lock_disc_g200 --at1-export results/proof/at1-g200/at1-samples.npz --out results/proof/at1-g200`
- **Rung 2 (CPU minutes, this spec's verdict):**
  `python scripts/at1_boundary_layer.py --npz results/proof/at1-g200/at1-samples.npz`
- **Capped smoke + regression gate (pre-lock, agent-run):** `--self-test`; then
  `lock_disc_g200` with `--burnin-steps 2000 --sample-count 300 --calibration-sample-count 300 --allow-unregistered-overrides`
  run **with and without** `--at1-export`, comparing manifests on pre-existing fields
  (ignoring timestamps/elapsed/file lists). Drift ⇒ `AT1_HARNESS_VOID`.
- **Rung 3 (conditional):** only if rung 2 files TWO_POLE — twin-pair palinstrophy
  composition, to be frozen in its own spec then. Not part of this verdict.
- **G=300:** no rerun; the banked flip (NEG-A → POSITIVE) is the replication read, cited.

## 2. Export artifact (schema v1, frozen)

`at1-samples.npz`: `schema_version=1`, `preset`, `objective_names` (the 6 slate names,
frozen order), `adj_starts`, `calib_starts`, `look`, `quantile`;
`phi_k` (n_adj × 18); `m_adj` (n_adj × 6, lookahead-**max** per objective);
`e_max` (6, calibration-quantile thresholds); `margin` = m_adj − e_max;
`actions` (n_adj × 6); **sibling** (registered exactly, per sign-off):
**lookahead-MEAN palinstrophy** — `m_mean_pal` (n_adj,), `e_max_mean_pal` (same q = 0.70
on the same calibration starts), `margin_mean_pal`, `actions_mean_pal`. Same horizon,
same calibration protocol; only max → mean. This is the burst-artifact control: the
lookahead-max was the flagged burst-unstable statistic.

## 3. Margin-band excision (frozen)

Bands excise the smallest-|margin| fraction β of adjudication samples for the
palinstrophy objective: **β ∈ {0, 0.02, 0.05, 0.10, 0.20}** (quantiles of |margin| —
scale-free). Per band: `damp_fraction(β)` on retained samples, powered iff ∈ [0.20, 0.40];
`a_mm(β)` via the harness's own `aggregate_knn_sweep` on retained (Φ_K, action) with
**ε fixed from the unbanded e_max** (the registered 0.05·√(2·e_max) formula — not
re-derived per band); held-out R²(Φ_K → m_adj) per band, reported.
Gates evaluate on **powered bands only**; unpowered bands are reported and excluded.

## 4. Controls (frozen)

- **Liveness:** E_low unbanded a_mm must reproduce the banked POSITIVE (≤ 0.005) from the
  same artifact — else the export path is questioned: void, fix, re-run (no verdict).
- **Sibling row:** lookahead-mean palinstrophy scored unbanded (a_mm, damp, R²) by the
  identical machinery.
- **Reproduction check:** unbanded (β = 0) palinstrophy a_mm should reproduce ≈ 0.195
  (rerun of the same frozen preset, same seed — drift beyond ±0.02 is reported and the
  run is treated as a fresh measurement, not a contradiction of the banked one).

## 5. Branch table (frozen)

| branch | fires iff |
| --- | --- |
| `AT1_BOUNDARY_LAYER_ARTIFACT` | ∃ powered band β ≤ 0.20 with a_mm(β) ≤ 0.005 **AND** sibling unbanded a_mm ≤ 0.005 |
| `AT1_TWO_POLE_CONFIRMED` | a_mm(β) ≥ 0.015 at **every** powered band **AND** sibling unbanded is powered with a_mm ≥ 0.015 |
| `AT1_UNDERPOWERED` | unbanded damp_fraction ∉ [0.20, 0.40], or no powered band, or sibling unpowered |
| `AT1_INCONCLUSIVE_MIXED` | none of the above (e.g., a_mm lands in (0.005, 0.015), or band and sibling disagree) — recorded, no verdict forced |
| `AT1_NEG_B` | any post-read change to bands, sibling, thresholds, or gates — voids the entry |
| `AT1_HARNESS_VOID` | regression gate fails (no-export drift) — do not interpret |

Thresholds inherited from the discriminator: POSITIVE line 0.005, NEG-A line 0.015,
power window [0.20, 0.40], q = 0.70. Nothing else is tunable.

## 6. Does not claim (inherited)

`AT1_TWO_POLE_CONFIRMED` does not demote the C1 separation (E_low's regime-2 read is
untouched); palinstrophy is a registered objective, not asserted safety-relevant; no new
PDE result; slate do-not-say list binds in full.
