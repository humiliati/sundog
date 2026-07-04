# AT-2b — Growth-Law Receipt (in-band, mean-form)

> 2026-07-03. Run of the frozen `AT2B_GROWTH_LAW_SPEC.md` — pure post-processing on the
> banked AT-2 exports (no simulation, no harness change). Artifact
> `results/proof/at2-g200/at2b_growth_law.json`. **Non-promotional; 32×32 truncation;
> no ∞-dim NSE claim.**

## Verdict: `AT2B_COLLAPSE_VACUOUS` at BOTH regimes — the §3.6 negative, run to completion at full strength

The in-band grid worked perfectly: **10/10 cells included** (damp pinned 0.300/0.236,
atom 0.000 everywhere — both instrument pathologies from AT-2 eliminated by design), and
at **every** cell:

> **K_min(decision) = K_min(value) = K_state** — all 1 at G=200, all 2 at G=300,
> perfectly flat in τ, for the mean-form primary *and* the max-form sibling.

| read | G=200 | G=300 |
|---|---|---|
| K_min(J_mean), τ ∈ {100..750} | 1,1,1,1,1 | 2,2,2,2,2 |
| K_min(max sibling), τ ∈ {100..500} | 1,1,1,1 | 2,2,2,2 |
| value control / K_state | 1 / 1 | 2 / 2 |
| **Δ = K_state − K_min** | **0 at every cell** | **0 at every cell** |

## The cross-regime adjudication (transparent, per the frozen table)

The cross-regime arm's *mechanical condition* is satisfied: K_min gains +1 at all five
matched τ, value control moves by exactly 1 ("flat within ±1"). **But K_state also gains
+1 at every matched τ** — decision, value, and state budgets all move 1 → 2 *together*.
Two frozen branches are simultaneously satisfied; the vacuity branch takes precedence
(the AT-2 implementation's registered ordering, CENSORED → VACUOUS → GROWTH), and the
physics reading is unambiguous: **the +1 is not decision-specific growth — the whole
cell gets one shell richer at G=300, and the decision budget simply rides the state
budget.** M2's direction (budgets ↑ in G) is confirmed — for the *state* budget, with
the decision budget showing no independent motion whatsoever.

## Sub-reads (reported tier)

- **Δ lens (the "most-resistance" tending): zero everywhere.** No cell where the
  decision reads below the reconstruction budget — the regime-2 mode-count shape is
  absent on this cell under every registered lens.
- **M1's mean-form question, answered:** flat K_min with the one-mode-short margin
  (a_mm at K = K_min − 1) **shrinking** in τ at G=300 (0.171 → 0.060) — averaging
  concentration beats error infection at this cell's horizons. The max-form sibling
  shows the same (0.168 → 0.067 on its band).
- **Event sub-read: `AT2B_EVENT_FLAT` at both regimes** (all evaluable gaps = 0). The
  reconnection 2D-analog finds no signal for the third time; the 3D parked lead stays
  parked with no motion from this arc.

## What the AT-2 arc establishes (AT-2 + AT-2b together)

**On the C1 truncation, control sufficiency collapses to state sufficiency under every
registered lens** — two functionals (max, mean), five in-band horizons, two regimes, a
value control, event slices, and the Δ separation read. The anchor doc's pre-registered
§3.6 vacuity negative is thereby **completed as a first-class measured result**, not a
cited caveat: at this cell, "enough modes to decide" and "enough modes to reconstruct"
are the same budget. Per the do-not-say list (2c), this is a publishable-quality negative
and is recorded plainly.

**Consequences.**
1. **AT-3 is now the sharpest question in the slate:** the *static* read cannot separate
   decision from state budgets — AT-3's maintained ledger (nudging at sub-determining
   budget) is precisely the instrument that could decouple them dynamically, and its
   §3.6 vacuity branch now has a measured static baseline to compare against.
2. **σ_modes registration: defer, definitively** — no growth measured, vacuity confirmed;
   registration-on-receipts says no.
3. The growth-form ∞-analogue (F1's consequence ii) found no purchase on this cell; the
   honest σ=∞-analogue inventory for the PDE side stands at zero measured instances.

Cross-refs: `AT2B_GROWTH_LAW_SPEC.md` (frozen prereg), `AT2_GROWTH_LAW_RECEIPT.md`
(the instrument findings this grid was built from), `NSE_ATTRACTOR_TAIL_HYPOTHESES.md`
AT-2/AT-3/F1, `PDE_DETERMINING_MODES_POSTULATE1.md` (whose §3.6 negative this completes).
