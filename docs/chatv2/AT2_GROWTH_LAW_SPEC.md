# AT-2 — Growth-Law Spec (frozen analysis; pending harness sign-off)

> 2026-07-03, **v1.1** (pre-run amendment, owner-requested: §1.5 mechanism registration +
> §2.5 event-conditioned sub-read added **before any curve exists**; the §3 branch table
> is byte-unchanged from v1 — the additions are subordinate reads that can neither rescue
> nor veto the main gates). Lift of slate entry AT-2 into its frozen pre-registration.
> **Implementation gated on `AT2_HARNESS_SIGNOFF_REQUEST.md` being stamped.**
> Non-promotional; no new determining-modes bound (K\* and K_min are *measured on the
> 32×32 truncation*, never asserted for NSE); no ∞-dim language; σ_modes schema
> registration remains an owner decision regardless of outcome.

## 1. Objects (all frozen)

- **Decision objective (fixed across ALL cells):** J_q(τ) = lookahead-max of the frozen
  **K=3 E_low** (the registered v5/v7 band — a fixed 9-mode set, independent of the
  emitted K=6 signature) over [s, s+τ]; threshold e_max(τ) = q = 0.70 quantile on the
  calibration block; action = (m > e_max(τ)).
- **Matched energy-value control:** action_val = (E_low_K3(t) > calibration median) at
  sample instants — same observable, no lookahead, no max.
- **Shadows:** Φ_K for K ∈ {1,…,6} = column slices of the emitted K=6 signature
  (verified nested: select(1) ⊂ … ⊂ select(6), sizes 1/4/9/16/25/36).
- **τ-grid:** {250, 500, 1000, 2000} steps. **Regimes:** G ∈ {200, 300}.
- **Sufficiency read:** a_mm(K, τ) via the harness's own `aggregate_knn_sweep`
  (read-only import) with ε(τ) = 0.05·√(2·e_max(τ)) — the registered formula, fixed
  across K within a τ-cell.
- **K_min(obj; τ, G):** least K ∈ {1..6} with a_mm ≤ 0.005 (the POSITIVE line);
  censored as "> 6" if none.

## 1.5 Registered mechanism predictions (v1.1; direction-only, named imports)

Two imports turn the K_min table into a mechanism test. Both are registered as
**direction/ordering predictions only** — no exponent is gated (a 32×32 cell at
G ∈ {200, 300} has no inertial range; asymptotic scalings are named, not applied).

- **(M1) The error cascade (Lorenz 1969; Leith–Kraichnan predictability).** Uncertainty
  at unobserved high wavenumbers infects the decision scale after a finite infection
  time (per-octave eddy-turnover spacing; in 2D enstrophy-range reasoning, roughly
  uniform per octave). Longer decision horizons give the infection more time ⇒
  **K_min(J_q; τ) is predicted nondecreasing in τ**, while the value objective (no
  horizon) has no infection channel ⇒ **flat**. This is the mechanism behind
  `AT2_GROWTH_CONFIRMED`, stated before the data. *Side-read (reported, not gated):*
  the τ-spacing between successive K_min increments estimates the per-shell infection
  time — one number per regime, comparable to the cell's eddy-turnover/burst timescales
  (AT-6 measured E_low autocorr ≈ 261 steps at G=200).
- **(M2) Determining-mode counts grow with G (Foias–Temam; Jones–Titi).** The imported
  bounds give K\*↑ in G — registered here only as the *direction* for the G=200→300
  matched-τ comparison and as a weak prior for the §4 bracket (the Jones–Titi-type
  G^(2/3)-flavored ratio ≈ 1.3 for G 200→300 is **noted, not gated**).

## 2. Cell inclusion (frozen; both checks inherited from receipts)

A (τ, G) cell enters the gates iff:
- **power:** damp(τ) ∈ [0.20, 0.40] (τ-level; K-independent), and
- **threshold-atom clearance (AT-1's lesson):** the m(τ) mass within ±10⁻⁶ of e_max(τ)
  is ≤ 0.05 — the failure mode the power gate provably misses.

Excluded cells are reported with cause. The growth read requires ≥ 3 included τ-cells
per regime; fewer ⇒ that regime contributes no gate read (reported).

## 2.5 Event-conditioned sub-read (v1.1; the parked reconnection lead's 2D analog)

The NSE ledger carries a **parked lead: vortex reconnection → determining-mode jumps**
(3D, owner-gated). 2D has **no** vortex reconnection (vorticity is materially conserved)
— the honest 2D analog of "topological event scrambles small scales and briefly raises
the determining budget" is the cell's **burst/merger events**. Registered sub-read, on
the same export (zero extra compute or fields):

- **Event flag (backward-looking, label-independent):** sample s is *post-event* iff
  max E_low_K3 over [s−500, s] > e_max(τ=500) (the same calibrated threshold); else
  *quiescent*. Backward-looking so the flag cannot leak the forward action label.
- **Sub-read:** K_min(J_q; τ) computed separately on the post-event and quiescent
  slices (each slice subject to the §2 inclusion checks, reported per slice).
- **Prediction (the parked lead, transplanted):** K_min(post-event) > K_min(quiescent)
  at matched τ — the growth is *carried by the event slice* (the AT-1/H2 lesson again:
  sufficiency is decided at events/margins, not the bulk).
- **Sub-branch tokens (reported tier — can neither rescue nor veto §3):**
  `AT2_EVENT_CARRIED` (gap ≥ 2 at ≥ 2 included τ-cells in a regime) /
  `AT2_EVENT_FLAT` (gap ≤ 1 everywhere included) / `AT2_EVENT_UNPOWERED`.
- If `AT2_EVENT_CARRIED` fires, the 3D parked lead gains its first instrument-grade
  motion (still parked; a 2D-analog receipt, not a reconnection result).

## 3. Branch table (frozen)

| branch | fires iff |
| --- | --- |
| `AT2_GROWTH_CONFIRMED` | K_min(J_q; τ) increases by ≥ 2 across the included τ-grid at fixed G (or by ≥ 1 at matched τ from G=200→300) **AND** K_min(value control) stays flat within ±1 over the same cells |
| `AT2_FLAT_NULL` | both K_min curves flat within ±1 — the graded structure is absent at this cell; recorded as the bounding negative |
| `AT2_COLLAPSE_VACUOUS` | at **every** included cell, the K where decision a_mm crosses POSITIVE equals the K where the state proxy crosses its line (held-out R²(Φ_K → high-mode norm) ≥ 0.5) — the §3.6 vacuity negative, verbatim, run as a first-class branch |
| `AT2_CENSORED` | K_min(J_q) censored (> 6) at ≥ half the included cells — growth unresolvable in this budget range (consistent-with, never confirmation) |
| `AT2_NEG_B` | any post-read change to τ-grid, objective, thresholds, or inclusion rules — voids the entry |
| `AT2_HARNESS_VOID` | regression gate fails |

## 4. Side deliverable (same batch, no harness change)

**K\* upper bracket** (`PDE_C1_SEPARATION_STATEMENT.md` §5, pre-registered, never run):
`lock_v5_k5` / `lock_v5_k6` twin runs. Bracket = smallest K whose twin-state witness set
vanishes; K = 2/3/4 are banked `TWIN_STATE_CERTIFIED`, so the honest outcomes are
"K\* ∈ {5, 6}" or "K\* > 6 at this cell." Folded into the receipt; no gate role.

## 5. Commands & deliverables

Owner batch (after sign-off + regression gate): the four commands in
`AT2_HARNESS_SIGNOFF_REQUEST.md`. Post: `python scripts/at2_growth_law.py --npz
results/proof/at2-g200/at2-samples.npz --npz300 results/proof/at2-g300/at2-samples.npz`.
Receipt: `AT2_GROWTH_LAW_RECEIPT.md` — the K_min table (2 objectives × 4 τ × 2 G), the
excision causes, the vacuity comparison, the K\* bracket, and the σ_modes registration
question restated for the owner (registration on receipts, not intentions).

## 6. Does not claim (inherited + entry-specific)

Growth on a 32×32 Galerkin cell licenses zero ∞-dim language; K_min values are
cell-specific; the state-proxy vacuity read is a declared gauge, not "the" data-
assimilation gauge (AT-3 owns that); a confirmed growth is a *measurement* instantiating
the `suffStatOrder_eq` idiom, not a theorem. **v1.1 additions:** no vortex-reconnection
claim in 2D (the event sub-read is the *analog*, and the 3D parked lead stays parked);
the error-cascade and Jones–Titi imports are direction-only (no exponent, no inertial-
range assertion on this cell); intermittency/structure-function content stays with AT-7
(compute-gated) — nothing here boards it.
