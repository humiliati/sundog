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

---

# v1.1 Reynolds-Matched Move — Rung 0 Receipt (2026-07-07)

> Probe run of scope §7 (frozen pre-run): `(k_f=3, G=675)`, Re_f held at the
> anchor's 25, burn-in 200k (the `lock_hidim` precedent). Artifact
> `results/proof/nse-h3-kf3-g675-adm/h3_admission.json`.

## Result: `H3_CELL_ADMITTED_PROBE_TIER` — the matched-Re cell is chaotic and forms

| gate | measured | window |
| --- | --- | --- |
| G1 held-out damp | **0.333** | [0.20, 0.80] (also inside the lock's binding [0.20, 0.40]) |
| G2 blockwise (8 × 50k) | 0.245–0.398, no collapse | [0.10, 0.90] |
| G3 atom mass | **0.000** | ≤ 0.05 |
| G4 liveness IQR(M) | 8.503e-2 | ≥ 1e-9 |

Regime character (reported): E_low envelope [0.404, 1.151], detrended
autocorrelation first zero ≈ 6,421 steps — a genuinely aperiodic cell, in sharp
contrast to v1's fixed point at the same k_f. **Re_f-matching is what made the
forcing-geometry move testable**; the v1/v1.1 pair is itself a measured result:
at fixed G the k_f axis exits the chaotic window, at matched Re_f it does not.

## Disposition

Per scope §7: **rung 1 unblocked, pending the one owner sign-off** — 3-site
additive preset `lock_v7_g675_kf3` (`lock_v7` block with `kf=3`, `grashof=675`,
`burnin_steps=200_000`), then the two owner-run locks (§4 commands with the
g675 preset/out-dirs; ~40 + 35 min). Decision gate §5 unchanged:
`NSE-H3-FORCING-GENERAL` / `NSE-H3-GRASHOF-LOCAL` / `NSE-H3-INPUT-UNPOWERED`
(the lock portability gate [0.20, 0.40] remains the binding formation gate).

---

# v1.1 Rung 1 — Locks Receipt (owner-run 2026-07-07): both adjudicators DEFER on coverage

> Preset applied under sign-off (self-test, config echo incl. Re_f 25.0 = 25.0,
> smoke — all clean). Artifacts `results/proof/c1-h3-kf3-g675-{knn-sweep,twin}/`.
> Integration clean both runs (~43 + 45 min; 1,958 steps/s). **Non-promotional.**

## What the locks measured

- **Formation PASSED (the binding gate):** damp 0.30856 (calib 0.29992), inside
  [0.20, 0.40] — the portable objective is powered and portable at the new cell.
  Not `INPUT-UNPOWERED`.
- **kNN-sweep: `INCONCLUSIVE_CONVERGENCE`** (`insufficient_coverage_passing_
  sweep_points`, `interpretable: false`): **zero** of the seven sweep points had
  fidelity coverage (r_k ≤ ε_K) sufficient to enter the fit — the a_mm read
  never formed.
- **Twin-state: `TWIN_STATE_DEFERRED_COVERAGE`** (`insufficient_signature_near_
  pair_coverage`, `PAIRED_FIBER_UNDEFINED`): candidate coverage **0.4588 <
  s_pos = 0.50** (registered gate, pre-dating this registration; every G-axis
  cell sat at 1.0). Witness pairs that do exist (38,577) behave anchor-like
  (disagree 0.0329), but the certificate correctly refuses a support-level claim
  from a minority of the sample.
- **Mechanism (measured):** matched Re_f preserved chaos but not attractor
  compactness. The G=675 cell's sampled energy range is **[0.386, 1.352]**
  (3.5:1) vs the anchor's [0.715, 0.735] (1.03:1), while the rule-derived ε_K
  *shrank* to 0.0589 (E_max 0.6946) — the same-rule fiber ball covers a far
  smaller fraction of a far wider signature distribution. High-mode norms
  spread likewise ([0.207, 0.760] vs [0.220, 0.244]).

## Verdict: `NSE-H3-INCONCLUSIVE_COVERAGE` (typed by precedent; spec-gap noted)

The scope §5 table did not enumerate a deferral row — an honest spec gap. Both
harness verdicts are the registered **deferral** category (neither positive nor
diagnosed failure), matching the RG-family precedent row
(`PDE-C1-RG-INCONCLUSIVE_CONTROL`, RG-v0 §6). The witness at (k_f=3, G=675) is
**neither established nor refuted**: the apparatus at its registered sample
density (50k) cannot resolve ε_K-fibers on this wider attractor. Not
`GRASHOF-LOCAL` (nothing failed while powered); not `FORCING-GENERAL`. No gate
was widened; no number reinterpreted.

**The forcing axis has now resisted in two distinct, measured ways:** laminar at
fixed G (v1), coverage-walled at matched Re_f (v1.1). Both are apparatus/regime
facts, not witness facts — the anchor witness stands exactly as receipted.

## Owner fork (staged, not spent)

1. **Coverage power move (new registration):** `fallback_v7_g675_kf3` —
   `sample_count 50k → 200k`, the house `fallback_v5` idiom (a registered
   bigger-N variant with precedent), everything else identical. ≈ 12.7M steps
   ≈ ~2–2.2 h per run × 2, owner-run, one more 3-site sign-off. Honest odds
   note: coverage grows as r_k shrinks ~N^(1/d_eff); with d_eff unknown at this
   cell the lift from 0.459 past 0.50 is plausible but not guaranteed — and a
   200k deferral would itself measure the cell as
   coverage-walled-at-house-scale. Pre-committed stop: if 200k also defers, H3
   closes final on this axis.
2. **Close H3 at `NSE-H3-INCONCLUSIVE_COVERAGE`** — two typed walls, axis open
   for future registrations, anchor untouched.

Cross-refs: `results/proof/c1-h3-kf3-g675-{knn-sweep,twin}/manifest.json`,
`PDE_C1_REGIME_GENERALITY_v0.md` §6 (the deferral precedent row),
`pde_c1_kolmogorov_cell.py` (`s_pos` coverage gate; `fit_rows < 2` sweep gate),
`NSE_H3_KF3_SCOPE.md` §5+§7.

---

# v1.2 Coverage Power Move — Receipt (owner-run 2026-07-07): the stop fires. H3 FINAL.

> `fallback_v7_g675_kf3` locks per the frozen scope §8 (adjudication N 50k → 200k,
> nothing else). Artifacts `results/proof/c1-h3-kf3-g675-fb-{knn-sweep,twin}/`.
> Integration clean, ~98 + 96 min. **Non-promotional.**

## The measurement: coverage is nearly N-flat

| quantity | 50k (v1.1) | 200k (v1.2) |
| --- | --- | --- |
| twin candidate coverage (gate ≥ 0.50) | 0.4588 | **0.4692** |
| kNN sweep coverage-passing fit points (need ≥ 2) | 0 | **0** |
| held-out damp (formation) | 0.30856 | 0.30674 |
| fiber pairs that exist / their disagree | 38,577 / 0.0329 | 179,516 / 0.0339 |

Quadrupling the adjudication sample moved candidate coverage **+0.010**. The
shortfall is not sample density: the r₅₀ radii shrink as N^(1/d_eff), and a 4×
lift that barely moves the ≤ε_K fraction reads as a large local effective
dimension over most of the sampled support. Linear-in-log-N extrapolation puts
the 0.50 gate ~two more quadruplings away (≈3.2M samples ≈ 160M steps ≈ 34 h per
run) — outside house scale, and clearing the twin gate would still leave the
sweep at zero fit points. Meanwhile everything that *can* be read is stable
across N: formation (damp ~0.307), and the existing fiber pairs stay
anchor-like (disagree ~0.033) at both sample sizes.

## Verdict (pre-committed, fires as frozen): `NSE-H3-INCONCLUSIVE_COVERAGE` — FINAL

Per scope §8: any registered deferral at 200k closes H3 **final** —
coverage-walled at house scale. No v1.3; the ε_K rule and `s_pos` stay untouched
forever under this scope (changing either after these reads would be
gate-widening). The witness at (k_f=3, G=675) is neither established nor
refuted, and the anchor witness is untouched.

## H3 close-out: the forcing axis, mapped

Three measured walls in one day, none of them witness failures:

1. **v1 (k_f=3, G=200):** the fixed-G axis exits the chaotic window — steady
   state, typed by the atom gate in ~5 min.
2. **v1.1 (k_f=3, G=675, matched Re_f):** chaos restored, objective portable
   (damp 0.307–0.309 at every N), but the attractor widens ~3.5× in signature
   range while rule-ε_K shrinks — both adjudicators defer.
3. **v1.2 (200k):** the deferral is N-flat — an attractor-geometry fact, not a
   sampling accident.

What a future reopening would need (future scope, not this one): a
coverage-adaptive apparatus registration — e.g., density-stratified fibers or a
regime-conditioned ε_K — which is a *protocol* change requiring its own
pre-registration and comparability argument against the banked G-axis cells.

## Post-AT slate ledger — FINAL (all entries dispositioned)

| entry | verdict |
| --- | --- |
| H1 proxy-faithfulness | `NSE-H1-PROXY-ONLY` (the witness is a proxy-relative fact) |
| H2 resolution stability | `NSE-H2-TWO-REGIME-N48-STABLE` (CFL-dt caveat carried) |
| H3 forcing-axis | `NSE-H3-INCONCLUSIVE_COVERAGE` (final; axis mapped, three typed walls) |
| H4 stationarity doctrine | `NSE-H4-STATIONARITY-GATE-LANDED` |
| H5 AT symbolic closeout | `AT5_FORMAL_CORE_ALREADY_LANDED` |

Cross-refs: `results/proof/c1-h3-kf3-g675-fb-{knn-sweep,twin}/manifest.json`,
`NSE_H3_KF3_SCOPE.md` §8, `NSE_H1_FIBER_RECEIPT.md`, `NSE_H2_N48_RECEIPT.md`,
`NSE_POST_AT_HYPOTHESES_SLATE.md`.
