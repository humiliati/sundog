# NSE-H3 Rung 0 — Admission Receipt (k_f=3, G=200)

> 2026-07-07. Lock-seed formation probe per the frozen `NSE_H3_KF3_SCOPE.md` §3
> (artifact `results/proof/nse-h3-kf3-g200-adm/h3_admission.json`; self-test 4/4
> preceded it). **Truth-only; no kNN, twin, or fiber number was computed. No
> rung-1 preset was ever added to the harness.** Non-promotional.

## Result: the cell is a steady state

| gate | measured | gate window |
| --- | --- | --- |
| G1 held-out damp | **0.000** | [0.20, 0.80] |
| G2 blockwise | 0.000 × 8 | [0.10, 0.90] |
| G3 atom mass at threshold | **1.000** | ≤ 0.05 |
| G4 liveness IQR(M) | **0.000e+00** | ≥ 1e-9 |

E_low envelope over the 500k-step window: **[0.321, 0.321]** — constant. After
the 100k burn-in the k_f=3, G=200 flow sits on (or numerically at) a fixed
point: at forcing wavenumber 3 the forcing-scale Reynolds number is ~k_f³ ≈ 3.4×
below the chaotic anchor (Re_f ∝ G/k_f³: 25 at the anchor, ≈ 7.4 here), and the
laminarization boundary falls between k_f=2 and k_f=3 at this G. The scope named
exactly this risk; the atom gate (AT-1 mandate) typed it automatically — a
steady cell puts all threshold mass on one atom.

## Verdict (per the frozen scope): `NSE-H3-INPUT-UNPOWERED` (regime: steady state)

**Final for this registration.** No rescue, no G-adjustment inside it. This is
**not** `NSE-H3-GRASHOF-LOCAL`: the witness did not fail — the cell poses no
decision problem (a steady state has a trivial shadow; regime-2's preconditions
fail before the witness can be tested). Total cost of the typed answer: ~5 min
of truth integration. Rung 1 was never unlocked; no harness change was made.

## Owner fork (staged, not spent)

1. **New registration — the Reynolds-matched forcing move (recommended):**
   `(k_f=3, G=675)`, holding the forcing-scale Reynolds at the anchor value
   (Re_f = G/k_f³: 200/8 = 25 = 675/27) while moving the one physical axis
   (forcing geometry). Two harness knobs move, one physical quantity is held —
   the registration would be named as such (`matched-Re forcing move`), which is
   the physically honest version of "one forcing-axis move": at fixed G the axis
   is confounded with distance-to-laminarization, as this probe just measured.
   Residual caveats to pre-register: G=675 sits beyond the tested Grashof range
   (attractor character unverified), and CFL at grid 32 projects safe
   (U₀ ≈ 2.9 vs the anchor's 3.5) but the probe gates decide. First step is the
   same ~6-min agent-run rung-0 probe at the new cell; locks and the preset
   sign-off only exist if it forms.
2. **Close H3 at `NSE-H3-INPUT-UNPOWERED`** — a first-class typed branch per the
   slate; the forcing axis stays open for any future registration.

Cross-refs: `NSE_H3_KF3_SCOPE.md`, `NSE_POST_AT_HYPOTHESES_SLATE.md` §5,
`NSE_H1_ADMISSION_RECEIPT.md` (the two-rung idiom), `AT1_BOUNDARY_LAYER_RECEIPT.md`
(the atom mandate that fired here), `results/proof/nse-h3-kf3-g200-adm/`.
