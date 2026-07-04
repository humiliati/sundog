# AT-2b — Growth Law, In-Band Re-Registration (frozen; mean-form primary)

> 2026-07-03. **New registration, not a rescue** (adopted by owner after
> `AT2_GROWTH_LAW_RECEIPT.md`; the AT-2 verdict `NO_GATE_READ` stands untouched).
> **No harness change and no new simulation:** AT-2b is pure post-processing on the
> banked `results/proof/at2-g{200,300}/at2-samples.npz` artifacts — the wide export was
> built for exactly this. Analysis frozen here before any AT-2b curve is read.
> Non-promotional; 32×32 truncation; no ∞-dim language; σ_modes registration stays
> deferred (on receipts, not intentions).

## 1. What changes vs AT-2 (and why)

AT-2 established that the **max-form** objective degenerates above τ ≈ 500 on this cell
(G=200: horizon-dependent threshold-atom; G=300: damp saturation). AT-2b therefore:

- **Primary objective → mean-form:** J_mean(τ) = lookahead-**mean** of the frozen
  E_low_K3 over [s, s+τ]; threshold = q = 0.70 calibration quantile of the same;
  action = (mean > threshold). AT-1's clean sibling — no atom, no max-saturation.
- **τ-grid → in-band:** **{100, 250, 400, 500, 750}** (5 cells; the ≥3-included rule is
  satisfiable with two losses).
- **Max-form retained as a reported sibling** on its usable band τ ∈ {100, 250, 400, 500}
  — continuity with AT-2, no gate role.
- **The "most-resistance" lens, registered (reported tier):** per included cell,
  **Δ(τ, G) = K_state − K_min(J_mean)** — the mode-count form of the C1 separation
  (Δ > 0 = decision readable *below* the state-reconstruction budget = the regime-2
  shape). AT-2 measured Δ = 0 everywhere it could see; AT-2b asks the same question on a
  grid fine enough to answer. Feeds AT-3's design either way; cannot rescue or veto §3.

Unchanged: shadows Φ_K (K ∈ {1..6}, nested slices), value control (E_low_K3 > calib
median), a_mm via `aggregate_knn_sweep` with ε(τ) = 0.05·√(2·threshold(τ)), K_min = least
K with a_mm ≤ 0.005 (censor > 6), inclusion checks (power [0.20, 0.40] + threshold-atom
clearance ≤ 0.05 within ±10⁻⁶), state proxy R²(Φ_K → high-mode norm) ≥ 0.5, event flag
(backward max over [s−500, s] > e_max^max(500) — same registered definition), seeds.

## 2. Mechanism registration (honest revision for the mean-form)

- **M1 (error cascade) is registered for the max-form sibling only** (nondecreasing
  K_min in τ, in-band). For the **mean-form the direction is genuinely ambiguous** —
  error infection pushes K_min up with τ; averaging concentration pushes decisions easier
  — so the mean-form direction is **registered as a question, not a prediction**: either
  monotone outcome is informative, neither is gated beyond §3's own arithmetic.
- **M2 (K↑ in G)** unchanged: direction-only for the matched-τ cross-regime arm.

## 3. Branch table (frozen; AT2B_ namespace)

| branch | fires iff |
| --- | --- |
| `AT2B_GROWTH_CONFIRMED` | K_min(J_mean; τ) increases by ≥ 2 across the included τ-grid at fixed G (or by ≥ 1 at matched τ, G=200→300, over ≥ 3 matched included cells) **AND** value-control K_min flat within ±1 |
| `AT2B_FLAT_NULL` | both curves flat within ±1 |
| `AT2B_COLLAPSE_VACUOUS` | K_min(J_mean) = K_state at **every** included cell (both regimes read) — the §3.6 negative; **the registered honest prior** |
| `AT2B_CENSORED` | K_min censored at ≥ half the included cells |
| `AT2B_NEG_B` | any post-read change to grid, objective, thresholds, or inclusion — voids |

Sub-reads (reported tier): Δ(τ, G); max-form sibling K_min on its band;
event-conditioned K_min (`AT2B_EVENT_CARRIED` / `_FLAT` / `_UNPOWERED`, same definitions);
M1 spacing if any increments appear.

## 4. Command & deliverables

`python scripts/at2b_growth_law.py --npz results/proof/at2-g200/at2-samples.npz
--npz300 results/proof/at2-g300/at2-samples.npz` (CPU minutes; agent-run).
Receipt: `AT2B_GROWTH_LAW_RECEIPT.md` — the K_min table (mean-form 5 τ × 2 G + sibling),
Δ table, inclusion causes, vacuity read, event tokens.

## 5. Does not claim

All AT-2 §6 fences inherited verbatim. Additionally: the mean-form objective is a *new
registration* whose relation to the frozen J_q (max-form) is measured, not assumed; a
vacuous-collapse verdict completes the anchor doc's own §3.6 negative and licenses no
resistance language; Δ > 0, if seen, is a reported observation for AT-3's prereg, not a
separation claim.
