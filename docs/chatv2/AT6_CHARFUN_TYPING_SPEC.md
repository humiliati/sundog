# AT-6 — charFun Typing of Compact Shadows (frozen spec)

> 2026-07-02. Lift of slate entry AT-6 (`NSE_ATTRACTOR_TAIL_HYPOTHESES.md`) into its frozen
> pre-registration, per house rule. **Class assignments below are frozen BEFORE any decay
> curve is computed** (the entry's kill condition). Non-promotional; no control claim; no
> PDE theorem; inherits the slate's do-not-say list in full.
> Script: `scripts/at6_charfun_typing.py`. Receipt: `AT6_CHARFUN_TYPING_RECEIPT.md`.
> Run: `python scripts/at6_charfun_typing.py` (CPU; generates its own streams by read-only
> import of `KolmogorovStepper` — the frozen harness file is not modified).

## 1. Cell & streams (inherited, pinned)

Kolmogorov cell exactly as `lock_v7_g200` / `lock_v7_g300`: grid 32, dt 0.01, k_f = 2,
forcing amplitude 1.0, ν = √(1/G), K = 3 (signature dim 18), G ∈ {200, 300}, seed 0.
Per regime: burn-in 100,000 steps, then a **100,000-step recorded stream** (1,000 time
units) of the per-step observation vector and label primitives.

**Observation vector (24-dim, frozen):** the 18-dim Φ_K signature + the 6 discriminator-
slate observables (`E_low, Z_low, E_high, Z_high, palinstrophy, top_shell`).

**Shadow at window T:** the running mean of the observation vector over [t−T, t] —
the trajectory-bag. T-grid (steps, frozen): **{1, 10, 50, 250, 1000, 5000}**
(0.01 → 50 time units; the registered lookahead horizon is 5 t.u. = 500 steps).

## 2. Registered rows (classes FROZEN here)

All labels are median/quantile-balanced by construction (chance = 0.5).

| row | label at time t | class (frozen prediction) |
| --- | --- | --- |
| R1 `regime_Elow` | E_low(t) > running-median | **component — SURVIVES** |
| R2 `regime_Zlow` | Z_low(t) > running-median | **component — SURVIVES** |
| R3 `mode_phase` | Re ω̂(1,0)(t) > 0 (orbit phase quadrant) | **phase — WASHES** |
| R4 `rising` | E_low(t) − E_low(t−50) > 0 (timing in cycle) | **phase — WASHES** |
| R5 `imminence` | max E_low over (t, t+500] > its median (transition imminence) | **phase — WASHES** |

Mechanism check (reported, not gate-bearing): |⟨e^{i·arg ω̂(1,0)}⟩_T| decay curve — the
literal charFun-under-averaging signature (`ShadowDecayGeneral` pattern).

## 3. Readout & evaluation (frozen)

Evaluation points: every 50 steps (the C1 sample interval), excluding the first
max(T-grid) steps. Readout: logistic regression (max_iter 500) on the z-scored 24-dim
T-averaged shadow. Split: **contiguous** first 70% train / last 30% test with a 5,000-step
gap (no window overlap across the split). Seed 0 throughout.

## 4. Gates (frozen)

Per regime, with acc(row, T) = held-out accuracy:

- **`AT6_TYPING_CONFIRMED`**: ∃ T in the grid with **all** phase rows ≤ 0.55 AND **all**
  component rows ≥ 0.65 (matched T — the eventual decay of everything as T → ∞ is
  expected and does not count against typing).
- **`AT6_TYPING_BROKEN(row)`**: at every such candidate T some frozen class assignment
  fails — record *which row* crossed class. A surviving phase row is the interesting
  outcome (a discrete invariant hiding in a nominally continuous observable), not a
  failure to rescue.
- **`AT6_DEAD_APPARATUS`**: at T = 1 (no averaging) any row reads < 0.55 — the shadow
  can't even read the unaveraged label; void, fix, re-run. (Component rows at T = 1 are
  near-tautologies — E_low is *in* the observation vector — which is intended: they are
  the liveness anchor.)
- **Kill (`AT6_NEG_B`)**: any change to rows, classes, T-grid, or thresholds after the
  first curve is read voids the run. This spec is the freeze.

Verdict = per-regime; the slate-level read requires the same verdict at both G.

## 5. Does not claim (inherited + entry-specific)

Time-averages are not claimed to be the only compact shadows; the typing types *shadows*,
not resistance (that stays with AT-3/AT-4); no ergodicity theorem — mixing behavior on
this cell is imported empirics; the bridge "physical time-average = the in-tree averaging
map" is a **named import**, not a proof; nothing here edits the C1 claim chain,
`PROMOTE_GATE.md`, or any public surface.
