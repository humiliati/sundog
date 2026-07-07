# NSE-H2 — v7 N-Refinement Spec (frozen; `lock_v7_g{200,300}_n48`)

> 2026-07-06. Lift of `NSE_H2_V7_N48_SCOPE.md` into its frozen pre-registration —
> **all gates, criteria, and branch labels fixed here before any n48 number under
> the v7 objective exists.** Owner signed off the 3-site additive harness edit;
> applied same day (presets `lock_v7_g200_n48` / `lock_v7_g300_n48`); harness
> self-test PASSED, non-verdict smoke PASSED
> (`results/proof/c1-h2-preflight-smoke/`), and the `lock_v7_g300` config echo
> matches its banked manifest field-for-field (no existing preset touched).
> Non-promotional; the scope's claim boundary binds verbatim. H2 uses the **v7
> portable selector** (H1 closed `NSE-H1-PROXY-ONLY`; slate §4 item 4).

## 1. Registration

- **Perturbation (exact, one axis):** `grid_size 32 → 48`, `n_modes 16 → 24`.
  Everything else bit-identical to `lock_v7_g{200,300}` (verified by config
  echo): k_f=2, K=3 (d=18), dt=0.01, seed 20260528, burn-in 100k,
  portable-quantile q=0.70, calibration 50k / gap 5k / adjudication 50k at
  stride 50, τ=500, kNN sweep k∈{10,15,20,25,30,40,50} with a_mm ≤0.005
  POSITIVE / ≥0.015 NEG-A, twin k=50 with gates (witness fraction ≥0.01, unique
  pairs ≥100). **No dt/objective/K/q retune after any read.**
- **Derived-by-rule, never copied:** ε_K = 0.05·√(2·E_max) at the n48 E_max;
  δ_H = max(1e-6, 0.05·median‖Q_K‖) over the n48 high-mode set. Reported
  against grid-32 values.
- **Regression targets (banked; comparability report only, never gated):**
  g200 kNN a_mm −0.00079 / slope 0.736; g300 kNN a_mm +0.00058 / 0.564; g300
  twin 942,834 pairs @ ε_K 0.0664 / δ_H 0.0111; pair-count drops of the
  `lock_v5_n48` magnitude (693,795 → 25,979) are precedented and acceptable.

## 2. Program (pinned order; owner-run; hard stop)

| run | command | est. |
| --- | --- | --- |
| R1 | `python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g200_n48 --adjudicator knn-sweep --out results/proof/c1-h2-g200-n48-knn-sweep` | ~1.5–2 h |
| R2 | `python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g200_n48 --adjudicator twin-state --out results/proof/c1-h2-g200-n48-twin` | ~1.5 h |
| — | **HARD STOP: R1+R2 must both pass §3 before any G=300 run is interpreted** | |
| R3 | `python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g300_n48 --adjudicator knn-sweep --out results/proof/c1-h2-g300-n48-knn-sweep` | recal. from R1 |
| R4 | `python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g300_n48 --adjudicator twin-state --out results/proof/c1-h2-g300-n48-twin` | recal. from R1 |

R1's measured wall-clock recalibrates R3/R4 estimates before they fire.

## 3. Gates (per cell, in order; H4 checklist imported)

1. **Solver diagnostics (pre-read):** no NaN/Inf; bounded burn-in envelope;
   dealias cutoff 16; step rate within ~2× projection. Fail ⇒
   `NSE-H2-NUMERIC-WALL`, no verdict field read.
2. **Portability gate:** held-out adjudication damp ∈ [0.20, 0.40]. Fail with
   clean diagnostics ⇒ `NSE-H2-RES-SENSITIVE(objective)`, typed, no retune.
3. **kNN half:** `STRICTNESS_WITNESS_POSITIVE` under the identical sweep.
4. **Twin half:** `TWIN_STATE_CERTIFIED` under the identical protocol gates, at
   the same-rule ε_K as the kNN read (composition requirement).

"Survives refinement" = same verdict labels under identical protocol gates —
never numeric parity with the targets.

## 4. Verdict table (frozen)

| outcome | verdict |
| --- | --- |
| R1–R4 all pass | `NSE-H2-TWO-REGIME-N48-STABLE` |
| R1+R2 pass; R3 or R4 fails gates 2–4 with clean diagnostics | `NSE-H2-RES-SENSITIVE(g300)`; G=200 half banks as `NSE-H2-V7-N48-STABLE` |
| R1 or R2 fails gates 2–4 with clean diagnostics | `NSE-H2-RES-SENSITIVE(g200)` — program pauses; no G=300 spend |
| diagnostics or runtime prevent a fair read at either cell | `NSE-H2-NUMERIC-WALL` (cell-typed) |

No gate widening, no constant retune, failures typed and final. Receipt:
`NSE_H2_N48_RECEIPT.md` (branch + comparability report + measured wall-clock).

## 5. Does not claim

One refinement rung on one axis — not grid convergence, not a continuum limit,
not promotion, no infinite-dimensional NSE statement. H1's proxy-relative typing
stands untouched. A full pass licenses exactly the slate's phrase: *"two-regime,
current-selector, N-refinement-stable witness."* `docs/chatv2/` stays no-publish.

Cross-refs: `NSE_H2_V7_N48_SCOPE.md` (design rationale),
`NSE_POST_AT_HYPOTHESES_SLATE.md` §4, `PDE_C1_REGIME_GENERALITY_v1.md` (§3–§6
inherited verbatim), `PDE_C1_ROBUSTNESS_WAVE.md` + `results/proof/c1-n48-twin/`,
`NSE_H1_FIBER_RECEIPT.md`, `NSE_STATIONARITY_GATE_CHECKLIST.md`.

---

## 6. v1.2 Solver-Stability Amendment (commissioned 2026-07-06; frozen pre-run)

> Owner green-lit `lock_v7_g300_n48_dt5` after the R3/R4 diagnostics failure and
> the CFL probe (dt=0.01 non-finite at step 300; dt=0.005 stable through the full
> 600k-step / 3,000-time-unit soak). **Frozen here before any dt5 number exists.**

- **One physical change:** `dt 0.01 → 0.005` at the G=300/n48 cell only — a
  CFL-consistent solver-stability accommodation (dt_crit scales with dx), not a
  free dial. **Step-count constants rescale ×2 so every physical quantity is
  held fixed:** burn-in 200k steps (1,000 tu), sample interval 100 steps
  (0.5 tu), lookahead 1,000 steps (τ = 5.0 tu), calibration gap 10k steps
  (50 tu). Sample counts (50k cal / 50k adj), objective, q, K, seed, and all
  adjudicator constants unchanged. Total ≈ 10.21M steps ≈ 2–2.3 h per run.
- **Gates:** §3 verbatim (diagnostics → portability → kNN → twin), targets and
  criteria unchanged.
- **Typed caveat (pre-registered):** a dt5 pass makes the two-regime claim
  "N-refinement-stable with CFL-consistent dt at the refined grid"; a dt5
  gates-2–4 failure is `NSE-H2-RES-SENSITIVE(g300)` with a dt-vs-grid
  disambiguation NOT pre-committed (a G=200/n48/dt5 run would be a future
  registration, not a rescue of this one).
- **Pre-committed stop:** if dt5 also fails gate 1 (diagnostics),
  `NSE-H2-NUMERIC-WALL(g300)` stands **final — no v1.3, no further dt moves.**
- Commands (owner-run):
  `python scripts/pde_c1_kolmogorov_cell.py --preset lock_v7_g300_n48_dt5 --adjudicator knn-sweep --out results/proof/c1-h2-g300-n48-dt5-knn-sweep`
  then `--adjudicator twin-state --out results/proof/c1-h2-g300-n48-dt5-twin`.

---

> **Post-run status (2026-07-06): R1–R4 owner-run; receipt filed
> (`NSE_H2_N48_RECEIPT.md`).** R1 `STRICTNESS_WITNESS_POSITIVE` + R2
> `TWIN_STATE_CERTIFIED` at G=200/n48 with near-invariant comparability (E_max /
> ε_K to 4–5 decimals, δ_H to 6, twin pairs within 0.7%, high-mode complement
> 2.5×) ⇒ **`NSE-H2-V7-N48-STABLE` banked.** R3/R4 hit a solver blow-up
> (overflow → NaN in the early transient; no verdict field produced) ⇒
> **`NSE-H2-NUMERIC-WALL(g300)`**, probe-confirmed as the explicit-advection
> CFL boundary (dt=0.01 non-finite at step 300; dt=0.005 stable). Live owner
> fork in receipt §5: bounded v1.2 dt-amendment (`lock_v7_g300_n48_dt5`,
> physical quantities held fixed, ~2–2.3 h × 2 runs) or close at the wall.
