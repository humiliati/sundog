# Structural Failure Coincidence — Cut 2 C2-A Numeric Freeze + Behavioral-Effectiveness Receipts

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Parent: [`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md) (C2-A)
Inputs: [`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md) · [`P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md`](P2_CUT2_C2CD_LEVERAGE_GATE_AND_INVALID.md)
Controller: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md) (`agents/photometric.py`)
Audited by: [`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) (D3 / C4-D)
Filed: **2026-05-16 (PT)**. Status: **C2-A FILED FOR AUDIT — HOLD FOR
EXECUTION**. Freezes the complete C2 numeric set and pre-registers three
behavioral receipts; **the receipts are computed and frozen before any
controller run and may legitimately fail (which blocks, never tunes)**.
Cut-2 execution remains HELD on these receipts, C3-A/B/C/D, C4-A/B/C/D,
C5, and a fresh P2-spec admission re-check. No harness written; nothing
run.

## Purpose

C2-A is the complete C2 numeric freeze in one pass and must resolve the
two C2-C/D audit holds plus the package-gating clarification:

1. a scalar `C_L1(s_obs)` is constant in the controller's carrier `q`,
   so it cannot move `argmax_q I` — it must be proven to change the
   *actual* `PhotometricAgent` confidence/abstain behavior, not rescale
   an unchanged argmax;
2. `f_par_obs < R22` needs a frozen *objective-level* abstain criterion,
   never an `if`-branch;
3. rule whether multiplying the whole `[P+T_cza+T_tan]` bracket by
   `C_L1` masks the L2/L3 consistency-term tests.

## 1. C2-A-1 — `C_L1` behavioral-effectiveness receipt (resolves hold 1)

`C_L1(s_obs)` is **not** behaviorally inert, because the bound
controller is not a pure argmax oracle. `agents/photometric.py` has two
absolute-magnitude-sensitive paths:

- **Re-acquire / lock-fail:** `reacquire_threshold = 0.05`,
  `reacquire_hold_steps = 30` — if the target intensity stays below the
  threshold it abandons TRACK and re-SCANs (never emits a confident
  converged `q̂`).
- **Curvature-limited TRACK estimator:** perturb-and-observe ESC; the
  demodulated gradient SNR scales with peak height/curvature, which
  `C_L1` multiplies.

`s_obs = f_par_obs/R22 − 1` (observable, no `h`). The frozen `C_L1`
ramp drives the **full-field target intensity** (the bridge maps `I` to
`detector_intensities[target]` under the §5 frozen scale) **below the
controller's own `reacquire_threshold` in the L1-ineligible band** — so
PhotometricAgent's *own* logic re-scans and never locks (emergent
abstain) — while it stays **above** it inside the eligible band (the
controller locks). 

**Receipt (pre-run, computed, no Cut-2 score):** tabulate, against the
documented controller constants and the §5 frozen bridge scale, the
full-field target intensity and TRACK gradient-SNR vs `h` across the L1
line; show the eligible band locks and the ineligible band trips the
controller's own reacquire/lock-fail. **This is a landscape-vs-
controller-threshold characterization, not a Cut-2 run.** If the frozen
`C_L1` does not cross the controller's real threshold at the L1 line,
**C2-A is not closed** (the gate would be cosmetic → boundary test
vacuous). No post-results tuning (A3).

## 2. C2-A-2 — C2-D objective-level abstain criterion (resolves hold 2)

`f_par_obs < R22` abstain is read from a **frozen property of the
objective `O`**, never a branch:

> abstain ⟺ `max_q O < O_floor` **or** no `|r| ≤ r_tol` solution exists
> in the `q` domain **or** the peak condition number exceeds
> `κ_cond_max`.

By C2-B geometry these rows have no `r=0` manifold and a low flat
penalty-edge max, so the criterion fires there and the controller's own
reacquire/lock-fail path (the §1 mechanism) yields the emergent abstain.
`O_floor`, `r_tol`, `κ_cond_max` are frozen in §4. **Receipt:** show the
degenerate (`f_par_obs<R22`) rows trip the criterion and eligible rows
do not — with **no `if f_par_obs<R22` branch anywhere** (C4-D's D3
taint test cross-checks the controller code).

## 3. C2-A-3 — package-gating separation (resolves the clarification)

Ruling: **`C_L1` is not whole-route-package masking; the L1 ramp and the
L2/L3 term-loss are disjoint in `h` by geometry.** Leverage
`s_obs ≈ sec(h)−1` is monotone increasing in `h`: the L1-ineligible band
is **low `h`** (small leverage); the L2 (`h>32°`) and L3 (`h≥29°`) loci
are **high `h`**, where leverage is large and `C_L1 ≈ 1`. **Receipt:**
show `C_L1(s_obs(h)) ≥ 1 − ε_C` for all `h ≥ h*` with a frozen
`h* < 29°`, so multiplying the bracket by `C_L1` leaves the L2/L3
consistency-term tests intact (`C_L1 ≈ 1` throughout that region).

## 4. The complete C2 numeric freeze (provenance-tagged, A3)

Every C2 number is fixed here, before any run. Provenance class:
**[G]** immutable geometry/receipt boundary (a change is a geometry
re-spec, forbidden as a goalpost move); **[E]** pre-registered
engineering tolerance (amend-only, justified, **never** post-results).

| symbol | role | provenance |
| --- | --- | --- |
| L1 line `sec(h)−1 = 2%·R22` | eligibility boundary | **[G]** BOUNDARY_MAP L1 |
| L2 `h=32°`, L3 `h=29°` | handle-vanish loci | **[G]** P1 §B-1 / Pass C7 |
| `q_a ∈ [−A,+A]`, `A = ρ·R22` | anchor-correction support | **[G]** C2-B (form); `ρ` **[E]** |
| `ρ`, `σ`, RNG `seed` | anchor-noise scale, ridge width, determinism | **[E]** |
| `h`-grid, `q_h` domain | sweep / search domains | **[E]** |
| `λ`, C2-B(i) conditioning floor, C2-B(ii) P-A tol | degeneracy-break + "doesn't move the optimum" | **[E]** |
| `C_L1` form (smooth monotone ramp), steepness, centre `=` L1 line | leverage gate | centre **[G]**; steepness/form **[E]** |
| `T_cza`,`T_tan` closed forms + magnitudes | consistency terms | **[E]** magnitudes; gating on observed `f_cza/f_tan` **[G]** |
| C2-C(i)/(ii) detectable+discriminating params | boundary calibration window | **[E]** |
| `O_floor`, `r_tol`, `κ_cond_max` | C2-D objective-level abstain | **[E]** |
| bridge scale (eligible-band peak ≡ 1.0) | `I → detector_intensities[target]` map | **[E]** convention, see §5 |
| `reacquire_threshold = 0.05`, `reacquire_hold_steps = 30` | controller's own abstain path | **[G]** by C1 binding (the controller's constants, not ours to set) |
| C4-D1 / P-A floor + must-differ region | D1 anti-Cut-1 target (vs true `h`) | **[E]** floor; region **[G]** (L1 band) |

The actual numeric values are fixed in this freeze and are A3-immutable
from this point; the admission re-check audits them, it does not let
them move after results.

## 5. C2-A load-bearing trap (surfaced adversarially)

The single biggest self-seal hazard in the whole numeric layer is the
**bridge `I → detector_intensities[target]` scale**: if that scale is
chosen *after* seeing whether `C_L1` crosses `reacquire_threshold`, the
§1 receipt is rigged. Therefore the bridge scale is frozen by an
**independent principle, before the receipts are computed**: the
**eligible-band route peak is normalized to `1.0`** (a pre-registered
convention, not fit to the controller). With the scale fixed
independently, C2-A-1/2 become genuine pass/fail with **no scale freedom
left**. If the receipts fail under that frozen scale, **C2-A fails and
the gate design must change by an append-only, justified amendment** —
never a post-hoc scale tweak (A3). This is the anti-self-seal that keeps
"prove C_L1 bites the controller" from degenerating into "pick a scale
that makes it bite."

## 6. Honest couplings

- **C2-A-1 ↔ C4-D.** C2-A-1 proves the *landscape* crosses the
  controller's own thresholds (landscape side); C4-D's D3 taint test
  proves the controller code has **no branch** on
  `f_cza/f_tan/f_par_obs<R22/h` (code side). Both ⇒ genuinely emergent.
- **C2-A is the upstream freeze.** C3-A (`P_in, κ, σ_D, M, τ_pc`) and
  C4-A (D1/P-A floor; D2 floor **=** the C3-C receipt floor, one number;
  fixtures; probe set) **inherit** C2-A's frozen scale/grid/seed and
  must not contradict them.
- **C2-A-3 feeds q3 scoring:** L1 (low-`h`) and L2/L3 (high-`h`) are
  scored as disjoint loci; a single controller behavior change cannot
  satisfy two loci at once.

## Cut-2 C2-A binding rules

1. Every §4 number is frozen here; **[G]** rows immutable, **[E]** rows
   amend-only/justified/never-post-results (A3).
2. The bridge scale is the §5 independent convention, frozen before the
   receipts; no post-hoc scale tuning ⇒ such a run is **void**.
3. C2-A-1/2/3 receipts are produced and frozen **before** any controller
   instantiation; a failing receipt **blocks** (append-only redesign),
   never a tuned pass.
4. Reading true `h` anywhere in `C_L1`/`O`/the abstain criterion ⇒ run
   **VOID** (A1).

## Explicit non-bindings (cannot satisfy C2-A)

- Choosing/adjusting the bridge scale after seeing the §1 receipt.
- A `C_L1` that only rescales an unchanged argmax (argmax-inert ⇒ hold 1
  unresolved).
- An `if f_par_obs<R22: abstain` (or `if f_cza==0: …`) branch.
- Computing the receipts against a strawman instead of the documented
  `PhotometricAgent` constants / the bound controller.
- Any [E] value tuned after a controller result.

## Open items

C2-A files the complete freeze + the three receipt obligations for
audit. The receipts (C2-A-1 effectiveness, C2-A-2 objective-abstain,
C2-A-3 separation) are pre-run artifacts to be **computed and frozen
before the joint admission re-run**; a negative receipt blocks Cut-2.
Still-open siblings: **C3-A/B/C/D**, **C4-A/B/C/D**, **C5**.

After C2-A (incl. its receipts), C3-A/B/C/D, C4-A/B/C/D, and C5 are all
filed, the P2-spec admission check **must be re-run** as one audit of
the whole discriminating cut; only on **ADMIT** may a Cut-2 harness be
built or run. Public-Language Constraint remains fully in force: no
`CONFIRMED` / traceability-success / theorem language anywhere
(including the rail).

## Honest prior (unchanged)

Even with the gate proven behaviorally load-bearing, a real inverse-free
ESC controller against a biased signal with a tempting reachable decoy
keeps the likely honest outcome at **D / BOUNDARY FOUND** — the
Proxy-Collapse confirmation avenue (`debunked.md`, P1 §C). **B** is
earned *only* by a measured refusal of the tempting decoy at the
quantified in-sample cost **and** emergent failure coincident with
L1/L2/L3. Either is a clean result; the in-between is not.

## Audit Notes

*(reviewer space — append-only below)*

**2026-05-16 (PT) — Codex audit.** Direction accepted; C2-A is **not
yet execution-closing**. The key C2-C/D objection is resolved in the
right way: the L1 ramp is tied to `PhotometricAgent`'s actual
absolute-intensity reacquire path, and the bridge-scale anti-self-seal
(eligible-band peak `= 1.0` before receipts) is load-bearing. However,
this file currently freezes the **provenance slots and receipt
obligations**, not all numeric values: `rho`, `sigma`, `lambda`, the
grid/domains, `C_L1` steepness/form parameters, `T_cza`/`T_tan`
magnitudes, `O_floor`, `r_tol`, `kappa_cond_max`, floors, and receipt
tables still need concrete values before C2-A can be counted closed.
Two receipt-level holds remain: (1) the reacquire argument must follow
the actual phase semantics in `agents/photometric.py` (SCAN/SEEK always
proceed; the threshold is enforced only during TRACK after
`reacquire_hold_steps`), so the confidence/lock readout must be frozen
as a sustained-TRACK criterion, not inferred from peak intensity alone;
(2) C2-A-2 must include a reproducible objective scan proving invalid
rows trip the abstain criterion while eligible rows do not. No
controller/harness run.
