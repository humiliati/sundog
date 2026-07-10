# NSE-H3 Global-Gauge (Non-Fiber) Probe — Spec (frozen)

> 2026-07-07. The generalization push after `NSE-H3-FORCING-GENERAL-MIDE`: does
> regime-2 hold on the **full** G=675 attractor — tails included — in the
> coverage-free global gauge? Every fiber apparatus (frozen / A / B) was
> coverage-walled there; this probe has no fibers to wall. Pure post-processing
> on banked `samples.npz` (no integration). **Frozen before any G=675 global
> number exists.** Non-promotional; finite, sampled-support, proxy-relative,
> no C1 promotion, `docs/chatv2/` no-publish.

## 0. Why this and not the richer-K shadow (the other candidate)

- **K=4 fibers are receipts-predicted to fail:** the harness's own v5 note
  records the "curse-of-dimensionality coverage failure observed in v2/v4 at
  K = 4" — at the *compact* G=200 cell with 50k samples. On the wider G=675
  attractor a K=4 fiber read is a guaranteed `UNPOWERED`; and the banked
  signatures are K=3 (18-dim), so any K=4 read requires new integration.
- **The global gauge answers the open question with banked data:** the mid-E
  positive certified the mode band; the tails stayed unread because fibers
  can't reach them. The global gauge reads the full support by construction.
- **Parked successor:** a state-recon K-sweep at G=675 (the m_det bracket at the
  deeper cell) is the natural K-move *after* this probe, and needs its own
  registration + new runs.

## 1. The two halves, both coverage-free (frozen machinery, reused verbatim)

**State half — `aggregate_state_recon` (untouched):** FVE(Q_K | Φ_K) via the
validated HGB estimator (block 70/30 split; permutation gate
`|R²(E_high permuted)| < 0.10` or no read), energy-weighted primary +
enstrophy-weighted + equal-weight median. **The marginal/non-marginal line is
the frozen receipt threshold, not a new constant: `FVE ≥ 0.99` ⇒ "approximately
a graph over Φ_K" (marginal); `< 0.99` ⇒ genuinely under-determined
(non-marginal).** Anchor context: the banked G=200 read was marginal (~0.99
energy-weighted — the "real but marginal" separation).

**Control half — global action read (new, registered here):**
`HistGradientBoostingClassifier` (same validated family), action from Φ_K.
Block split: first 70% train, **400-sample guard gap** (= 20k steps, beyond any
measured correlation scale on these cells), rest test. Gates:
- powered: `acc − majority ≥ 0.10` (the `delta_action` idiom);
- estimator control: permuted-label `acc − majority < 0.05`.

**Stratification (reported, never gated):** one model per half, test-set
metrics stratified by energy tercile (edges `q⅓, q⅔` by rule, as in the mid-E
spec) — does the state become *more* under-determined in the tails, and does
control hold there?

## 2. Data (banked; registered subsample)

- **G=675:** `c1-h3-kf3-g675-adaptive/samples.npz` (200k), **subsampled by
  stride 4 to 50k** (registered: matches the banked estimator envelope and the
  anchor N; preserves time order for the block split).
- **Anchors:** `c1-relative-reg-g{200,300}/samples.npz` (50k each), unmodified.
- `comp_k` (enstrophy weights) rebuilt from the cell's stepper geometry —
  **with the cell's own k_f** (3 for G=675: the force-inserted (0,3) changes the
  low/high mode split). Assert `len(comp_k) == high dim` before any fit.

## 3. Regression gate (precondition; agent-run)

Both anchors must read: estimator valid (state + control permutation gates) ∧
control powered (`acc − majority ≥ 0.10` — control-sufficiency is banked fact on
these cells). FVE is **reported** against the banked G=200 ~0.99 comparator
(comparability, not gated — different objective era). Any failure ⇒
`NSE-H3-GLOBAL-APPARATUS-REJECTED`, no G=675 read.

## 4. G=675 branches (frozen)

| outcome | verdict |
| --- | --- |
| control powered ∧ `FVE_varweighted < 0.99` | **`NSE-H3-GLOBAL-REGIME2-NONMARGINAL`** — full-attractor, coverage-free regime-2 with a genuinely under-determined state |
| control powered ∧ `FVE ≥ 0.99` | `NSE-H3-GLOBAL-REGIME2-MARGINAL` (anchor-like marginality persists at matched-Re) |
| control unpowered (gates pass, acc fails) | `NSE-H3-GLOBAL-CONTROL-FAILS` (the mode-band scope was real; control does not extend globally) |
| either permutation gate fails | `NSE-H3-GLOBAL-ESTIMATOR-INVALID` (no read) |

No threshold moves after any read. Tercile tables are interpretation aids only.

## 5. Claim boundary

A `NONMARGINAL` positive says: on the full sampled matched-Re G=675 attractor,
in the global (coverage-free) gauge, Φ_K genuinely under-determines the state
(`FVE < 0.99`) while determining the action (powered classifier) — regime-2
without the marginality caveat, at one cell, one forcing move. It does not
overturn the fiber verdicts (different gauge), claim k_f-generality, promote
C1, or make any infinite-dimensional statement. All reads remain proxy-relative
(`NSE-H1-PROXY-ONLY` typing carried).

## 6. Deliverable

`scripts/nse_h3_global_gauge.py`: `--self-test` (synthetic: determined ⇒ FVE≈1,
independent ⇒ FVE≈0, control + permutation behavior), `--regress`, `--run`.
Receipt: `NSE_H3_GLOBAL_GAUGE_RECEIPT.md`. Agent-run; the HGB fits are the cost
(~10–30 min/cell at 50k on a quiet box).

Cross-refs: `NSE_H3_MIDE_SUBREGIME_RECEIPT.md` (the mode-band positive this
extends), `NSE_H3_G675_STRUCTURE_DIGIN.md`, `PDE_C1_NONMARGINAL_PROBE.md` (the
three-norm FVE doctrine), `PDE_C1_ROBUSTNESS_WAVE.md` (state-recon + the K=4
coverage-failure receipt), `results/proof/c1-recon-k3/` (banked G=200 global
comparator), banked `samples.npz` under `c1-h3-kf3-g675-adaptive/` +
`c1-relative-reg-g{200,300}/`.

---

> **FINAL (2026-07-07): `NSE-H3-GLOBAL-REGIME2-NONMARGINAL`**
> (`NSE_H3_GLOBAL_GAUGE_RECEIPT.md`). Regression PASS (G=200 FVE 0.9994
> reproduces the banked marginality; G=300 first global read 0.9916; both
> control-powered, all permutation gates clean). G=675: **FVE 0.6322**
> (residual 0.368; eq-weight median per-component R² **0.0039** — the typical
> high-mode component is free), control powered (0.887 vs 0.692, margin 0.194,
> holds in every energy tercile incl. the tails, strongest in highE). The
> matched-Re cell is the **non-marginal regime-2 cell** — the attractor width
> that coverage-walled the fiber gauge is the genuine state freedom in the
> global gauge. Estimator-relative FVE + proxy-relative labels + one cell:
> fences carried.
