# NSE-H3 — Forcing-Axis Scope (`k_f: 2 → 3` at G=200; rung 0 frozen here)

> 2026-07-07. Lift of H3 in `NSE_POST_AT_HYPOTHESES_SLATE.md` §5 (unparked by
> owner after H1/H2 closed: `NSE-H1-PROXY-ONLY` + `NSE-H2-TWO-REGIME-N48-STABLE`).
> **The rung-0 admission in §3 is frozen before any k_f=3 number exists.** Rung 1
> needs one owner sign-off (§4). Non-promotional; the slate's claim boundary binds:
> a positive broadens regime geometry by **one axis only** — no substrate
> generality, no universality, no theorem.

## 1. The one registered move (slate mandate: do not combine axes)

**`k_f: 2 → 3` at fixed `G = 200`, grid 32, n_modes 16, K = 3, dt = 0.01, seed
20260528, v7 portable objective — nothing else moves.**

- **Why k_f=3, not 4:** the forcing-scale Reynolds falls roughly with k_f³
  (velocity scale ∝ 1/(ν·k_f²), length ∝ 1/k_f). k_f=4 at fixed G cuts Re_f
  ~8× below the known-chaotic anchor and risks a laminar cell, which would force
  a simultaneous G move — the axis-combining the slate forbids in v0. k_f=3
  (Re_f ~3.4× below anchor) is the minimal genuine forcing-geometry move.
- **Signature membership changes by force-insert (registered, reported):** the
  forced mode (0,3) is outside the natural top-9 low-mode selection, so the
  harness's registered force-insert path includes it, displacing one natural
  shell-2 mode. `d = 18` and K = 3 are unchanged; the forced mode is observed in
  both the anchor and the new cell (the design's intent). The signature *set*
  therefore differs from the G-axis cells — reported in every receipt.
- If the k_f=3 cell turns out non-chaotic, that is `NSE-H3-INPUT-UNPOWERED`
  (regime-typed), and any G-adjusted retry is a **new registration**, not an
  amendment of this one.

## 2. What the anchor provides (comparability targets, receipts-true)

(k_f=2, G=200) v7: portability 0.3003; kNN a_mm −0.00079 / slope 0.736; twin
693,795 pairs @ ε_K 0.060598; E_max 0.7344 — plus the H2 result that these
numbers are N-refinement-stable. ε_K / δ_H re-derive by rule at the new cell,
never copied.

## 3. Rung 0 — formation admission (FROZEN; agent-run, ~10 min, truth-only)

`scripts/nse_h3_kf3_admission.py` → `results/proof/nse-h3-kf3-g200-adm/` →
`NSE_H3_ADMISSION_RECEIPT.md`. No kNN, twin, or fiber number is computed.

- **Stream:** k_f=3 cell as registered above, lock seed 20260528; burn-in 100k;
  window 500,000 steps + 501 tail; instants every 50.
- **Formation (pi_hat's own functional, house inclusive window):** M(s) = max
  E_low over [s, s+500]; threshold = q0.70 of M over the first 100,000 window
  steps (2,000 instants, calibration-first); held-out damp on the remaining
  400,000 steps (7,990 instants).
- **Gates:** G1 held-out damp ∈ **[0.20, 0.80]** (the slate's H3 window); G2
  blockwise — 8 blocks of 50k, every block in [0.10, 0.90] (H4 collapse check);
  G3 atom clearance — mass of M within ±1e-6 of the threshold ≤ 0.05 (this gate
  also types steady/periodic regimes automatically — a non-chaotic cell puts an
  atom at the threshold); G4 liveness — IQR(M) ≥ 1e-9. Regime character
  (envelope mean/std/min/max, detrended autocorrelation first zero) reported,
  not gated.
- **Relationship to the lock:** probe gates are necessary-condition screening at
  probe tier; the binding portability gate remains the in-harness RG-v1 §6 gate
  ([0.20, 0.40]) inside the rung-1 locks.
- **Branch:** `H3_CELL_ADMITTED_PROBE_TIER`, else `NSE-H3-INPUT-UNPOWERED`
  (stage-typed, final for this registration — no rescue, no retune).

## 4. Rung 1 — locks (owner-gated; needs one 3-site harness sign-off)

New verdict-bearing preset **`lock_v7_g200_kf3`**: the `lock_v7_g200` block with
`kf = 3` — same three edit sites as the H2 presets (`VERDICT_BEARING_PRESETS`,
argparse choices, `build_config` elif). On sign-off: apply, harness self-test,
non-verdict smoke, config echo (existing presets unchanged + new preset echoes
kf=3/grid 32/portable-quantile), then owner-run:

```text
python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g200_kf3 --adjudicator knn-sweep --out results/proof/c1-h3-kf3-knn-sweep   (~40 min)
python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g200_kf3 --adjudicator twin-state --out results/proof/c1-h3-kf3-twin        (~35 min)
```

## 5. Decision gate (slate §5 verbatim, sharpened)

| outcome | verdict |
| --- | --- |
| portability + kNN `STRICTNESS_WITNESS_POSITIVE` + twin `TWIN_STATE_CERTIFIED`, composed at the same-rule ε_K | `NSE-H3-FORCING-GENERAL` |
| powered objective + clean diagnostics, either half fails | `NSE-H3-GRASHOF-LOCAL` (the informative failure: the witness is local to the tested forcing geometry) |
| rung-0 gates fail, or the lock portability gate fails | `NSE-H3-INPUT-UNPOWERED` (stage-typed; no adjudicator read) |

## 6. Does not claim

One axis, one new point. A positive does not claim k_f-generality (one new k_f),
does not touch H1's proxy-relative typing, does not promote C1, and makes no
infinite-dimensional statement. `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_POST_AT_HYPOTHESES_SLATE.md` §5, `PDE_C1_REGIME_GENERALITY_v1.md`
(objective + gates inherited), `NSE_H2_N48_RECEIPT.md` (anchor hardening),
`NSE_STATIONARITY_GATE_CHECKLIST.md` (H4, imported at G2),
`results/proof/c1-rg-v1-g200-knn-sweep/` + `c1-paired-fiber-g200/` (targets).

---

## 7. v1.1 — The Reynolds-Matched Forcing Move (commissioned 2026-07-07; frozen pre-run)

> Owner selected the matched-Re registration after v1's rung-0
> `NSE-H3-INPUT-UNPOWERED` (steady state at k_f=3/G=200;
> `NSE_H3_ADMISSION_RECEIPT.md`). **Frozen here before any G=675 number exists.**

- **Cell:** `(k_f = 3, G = 675)` — the forcing-scale Reynolds `Re_f = G/k_f³` is
  held at the anchor value (200/8 = **25** = 675/27) while the one physical axis,
  forcing geometry, moves. Two harness knobs change; one physical quantity is
  held; the registration is named accordingly. Everything else identical to §1
  (grid 32, n_modes 16, K=3, d=18 with (0,3) force-inserted, dt=0.01, v7
  portable objective, lock seed).
- **One registered deviation:** burn-in `100k → 200k` steps — the
  `lock_hidim_g1000` precedent (deeper attractor at higher G needs the longer
  settle; precedented, not a tune).
- **Pre-registered caveats:** G=675 lies beyond the tested Grashof range (the
  attractor character there is unverified — that is what the probe measures);
  CFL at grid 32 projects safe (U₀ ≈ 2.9 vs the anchor's 3.5) and a non-finite
  probe series types as a diagnostics failure, not a rescue prompt. A pass at
  the matched-Re cell reads "forcing geometry at matched Re_f", never
  "k_f-generality at fixed G" — v1 measured that the fixed-G axis is confounded
  with distance-to-laminarization.
- **Rung 0:** §3 gates verbatim (probe script gains registered `--grashof` /
  `--burnin` arguments; nothing else changes);
  out `results/proof/nse-h3-kf3-g675-adm/`.
- **Pre-committed stop:** if the matched-Re cell also fails rung-0 formation,
  **H3 closes at `NSE-H3-INPUT-UNPOWERED` final — no v1.2, no further
  formation attempts on this axis under this scope.**
- **Rung 1 (only on admission; owner sign-off then owner-run):** preset
  **`lock_v7_g675_kf3`** — the `lock_v7` block with `kf=3`, `grashof=675`,
  `burnin_steps=200_000`; same two locks, same §5 decision gate.

## 8. v1.2 — Coverage Power Move (commissioned 2026-07-07; frozen pre-run)

> Owner registered `fallback_v7_g675_kf3` after the v1.1 locks deferred on the
> registered coverage gates (`NSE-H3-INCONCLUSIVE_COVERAGE`, receipt locks
> section: sweep 0/7 fit points; twin candidate coverage 0.4588 < s_pos 0.50;
> mechanism = wider attractor at shrunken rule-ε_K). **Frozen here before any
> 200k number exists.**

- **One change, house-precedented:** adjudication `sample_count 50k → 200k` —
  the `fallback_v5` idiom (the repo's registered bigger-N variant size, imported
  rather than guessed). Calibration block (50k samples), gap, burn-in 200k,
  seed, objective, ε_K/δ_H rules, both adjudicators' gates: all unchanged.
  Coverage is the only quantity this move addresses (r_k shrinks ~N^(1/d_eff)
  at fixed protocol).
- **Cost:** ≈ 12.71M steps ≈ ~2–2.2 h per run × 2, owner-run; twin
  post-processing grows ~16× in pairs (manageable; run on a quiet box — the
  2026-07-06 memory-squeeze precedent applies at 200k sample arrays).
- **Decision gate:** §5 unchanged for interpretable outcomes
  (`FORCING-GENERAL` / `GRASHOF-LOCAL`), **plus the previously missing deferral
  row, now enumerated:** any registered deferral verdict at 200k ⇒
  **`NSE-H3-INCONCLUSIVE_COVERAGE` final — coverage-walled at house scale; no
  v1.3, no ε_K-rule or s_pos change ever** (that would be gate-widening). A
  200k deferral is itself a measurement: the cell's effective dimension
  outruns the house apparatus.
- Commands (owner-run):
  `python scripts/pde_c1_kolmogorov_cell.py --preset fallback_v7_g675_kf3 --adjudicator knn-sweep --out results/proof/c1-h3-kf3-g675-fb-knn-sweep`
  then `--adjudicator twin-state --out results/proof/c1-h3-kf3-g675-fb-twin`.
