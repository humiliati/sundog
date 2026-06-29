# C4 — The Order-Relative Law on a 4th *Kind* of Axis: find/check

**Date:** 2026-06-29  
**Status:** `EXTENDS_mode_relative` — the order-relative law extends to the find/check axis, but as a
**mode-vector, not a scalar.** Extends the cross-lane conjecture slate
([BOXSEL_CONJECTURE_SLATE.md](BOXSEL_CONJECTURE_SLATE.md)) past C1/C2/Phase-7g. Builds on the σ-order
slate's H3; toy / parity substrate; frozen-as-portfolio.

## The question

The slate's law held on three axes that each assign **one** order to a target — a bounded process with
budget `k` resolves iff target-order ≤ `k`: determination (σ-order), search reach (C1), pressure
reach/repertoire (C2 / Phase 7g). The find/check axis (σ-order slate H3) is a different *kind*: the same
target carries **two** orders at once. Does the law extend, or break?

```text
verify-order   = the witness-check budget — ops to CONFIRM the answer given a structural witness
predict-order  = the history budget that DETERMINES the next value (the σ_predict from the σ-slate)
```

## Result (exact / deterministic, N = 200,000)

```text
target            verify-order (witness)        predict-order (history)
parity λ          17   (max Ω; mean 3.5)         inf   (σ_predict)        → modes DIVERGE
LFSR(5) control    5   (state window)              6   (finite)           → modes AGREE  (meter live)
```

- **The extension (mode-relative).** For parity the verify-order is **finite** (≤ log₂n via Ω(n) given
  the factorization) while the predict-order is **infinite**. The same target carries two divergent
  orders ⟹ **no single scalar budget governs resolution; the law holds *per mode* only.** The law
  itself holds within each mode: a finite verify-budget confirms parity; no finite predict-budget does.
- **Non-vacuity guard.** A finite-σ control (LFSR) has **both** orders finite — verify 5, predict 6 —
  the modes agree and the predict-meter detects the finite order. So parity's divergence is a property
  of the *target*, not a dead meter.

## The 4th *kind*

On find/check the "order" is a **mode-vector**, not a scalar — the find/check analog of the σ-schema's
finding that σ is "≥6 distinct filtrations, not one comparable scalar." The first three axes were each
single-mode (one process-order per target); find/check is the axis where the order provably *splits*
into independent mode-budgets.

| axis | the "order" | shape |
|---|---|---|
| determination (σ-schema) | sufficient-statistic order | scalar (per filtration) |
| search reach (C1) | grid denominator | scalar |
| pressure reach (C2) | probe order | scalar |
| pressure repertoire (Phase 7g) | failure-shape set | scalar |
| **find/check (C4)** | **(verify-order, predict-order)** | **mode-vector** |

## It explains C2

The payoff is the connection back: **C2's break *is* a mode-confusion.** A pressure detector's signal —
"the answer is stable under the pressures I applied" — is a **verify-mode** check (confirm the closure
against a finite witness-set of pressures; finite budget). But *soundness* — "the closure is genuinely
closed" — is a **predict-mode** property (would *any* pressure move it, including unapplied ones; order
= predict-order = ∞ for σ=∞ closures). C2's break is the detector silently substituting its finite
verify-budget for the infinite predict-order — exactly the find/check category error H3 named. So the
4th axis doesn't just add an instance; it **diagnoses why the order-relative law has soundness teeth**:
resolution-soundness lives in the predict/find mode, but bounded oracle-free detectors operate in the
verify mode.

## Honest boundary

H3 already established that verify-a-witness and predict-σ are orthogonal axes (the category error). C4
adds the **order-relative-law framing** (the law holds per-mode, the order is a mode-vector, with a
liveness guard) and the **C2-explanation** — it does not re-derive the orthogonality. Parity substrate;
the verify-order is the worst-case Ω over the range (finite, ~log₂N); not a real-KG or product claim.

## Files

- `scripts/c4_find_check_order.py` — the two-mode order measurement + verdict (reuses
  `suffstat_h3_verify_vs_predict` and `order_meter`).
- `scripts/test_c4_find_check_order.py` — frozen test: **12/12 PASS, exit 0**.

---

*Sundog Research Lab — C4. The order-relative law extends to find/check as a mode-vector (parity:
verify-order 17, predict-order ∞); the law holds per-mode, no single budget; and the mode-split
explains C2's soundness break as a verify-vs-predict confusion. Internal; frozen-as-portfolio.*
