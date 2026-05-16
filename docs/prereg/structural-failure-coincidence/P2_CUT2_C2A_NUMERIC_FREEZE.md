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

**2026-05-16 (PT) — maintainer. Wave-2 [E] values + bridge-scale
convention frozen.** Append-only; the frozen body above is unchanged.
This populates the §4 provenance slots with concrete `[E]` values,
pins the §5 anti-self-seal bridge-scale convention, corrects a
maintainer-side arithmetic error on `h_L1`, and freezes the comparator
semantics for `D1_min_bias`. **All values are A3-immutable from this
point**: amend-only / justified / never post-results. The Wave-3
receipts (C2-A-1 sustained-TRACK landscape table, C2-A-2 objective scan,
C2-A-3 separation table) remain as pre-run obligations to be computed
against these frozen values; the Codex audit's two receipt-level holds
will be addressed in Wave 3, not here.

*Maintainer-side correction (sloppy cosine in v1 proposal).* The L1
boundary `h_L1 = arccos(1/1.02)` is **`≈ 11.37°`** (or `11.366°` to
three decimals), **not** `11.48°` as the draft proposal stated.
`cos(11.366°) ≈ 0.98039` matches `1/1.02` to four decimals; the v1
`11.48°` was an arithmetic error caught at audit. The correct value
propagates to the `h`-grid framing, the `C_L1` calibration below, and
the coincidence-window endpoints. No frozen body was edited; the
correction is recorded once here and applies to all downstream
receipts.

*§5 bridge-scale convention — pinned first, before any value below.*
The bridge `I → detector_intensities[target]` is the transform
normalising the eligible-band `I_route` landscape peak to `1.0`. With
the route Gaussian `I_route ∝ exp(-(f_par_obs - R22/cos(q_h) - q_a)² /
(2σ²))`, the per-bundle landscape peak is already `1.0` at the joint
optimum (the exponent vanishes there), so the bridge is identity in
the route channel. **The convention is therefore: bridge = identity;
eligible-band route peak ≡ 1.0 by construction.** Frozen here before
any C2-A-1/2/3 receipt is computed; A3-immutable; never tuned
post-results.

*§4 freeze (concrete `[E]` values, A3-immutable).* `[G]` rows
inherited from prior freezes are unchanged; the table below fills the
`[E]` slots.

| symbol | value | provenance | one-line defense |
| --- | --- | --- | --- |
| L1 line `sec(h)-1 = 2%·R22` (i.e. `s_obs = 0.02`, `h_L1 ≈ 11.37°`) | (inherited) | `[G]` BOUNDARY_MAP L1 | unchanged; the `11.37°` figure replaces the v1 proposal's incorrect `11.48°`. |
| L2 `h = 32°`, L3 `h = 29°` | (inherited) | `[G]` P1 §B-1 / Pass C7 | unchanged. |
| `ρ` (anchor-noise scale, fraction of R22) | **`0.02`** | `[E]` | Pinned at the L1 receipt line. `A = ρ·R22 ≈ 0.44°`. The L1-eligible/ineligible transition is then the noise-dominated→signal-dominated transition by construction, not an arbitrary tolerance. |
| `σ` (route Gaussian width in `q_h` degrees) | **`0.5°`** | `[E]` | Below `τ1 = 1.5°` (so the ridge is discriminating vs the PASS tolerance) and above the typical visual-edge `~5–10 px → q_h` mapping (so a competent ESC can climb on the eligible band — P-B). |
| RNG `seed` | **`20260516`** | `[E]` | Date integer; arbitrary, pinned only for reproducibility. |
| `h`-grid | **`[0°, 40°]` step `0.5°`** (81 points) | `[E]` | Brackets `h_L1 ≈ 11.37°`, `L3 = 29°`, `L2 = 32°` with ≥ 5° margin on both sides; 0.5° step is 3× finer than the ±1.5° coincidence window. |
| `q_h` domain | **`[0°, 60°]`** | `[E]` | Brackets `arccos(R22/f_par)` over all eligible h plus search margin; tighter than Cut-1's `Q_MAX = 80°` since L2 caps real h at 32°. |
| `q_a` domain | **`[-A, +A]`** with `A = ρ·R22 ≈ 0.44°` | `[G]` form (C2-B); `A` derived from `[E]` ρ | C2-B unique-max construction; A pinned by ρ above. |
| `λ` (convex penalty strength on `(q_a/A)²`) | **`1.0`** | `[E]` | In bridge-normalised units (eligible peak ≡ 1.0). Penalty q_a-curvature `2λ/A² ≈ 10.3 /deg²` dominates route q_a-curvature `1/σ² = 4.0 /deg²` by **2.6×** at the joint optimum — unique max at `q_a = 0` with the C2-B(i) conditioning floor cleared; penalty equals route peak at `|q_a| = A` (1× safety factor); route signal not crushed. |
| `τ_C2-B-ii` (argmax-stability tol: `|argmax_{q_h} I - q_naive(h, ε)|`) | **`0.05°`** | `[E]` | 1/30 of τ1; well above 0.5° q_h-grid quantisation (so it's measurable as an actual delta on the grid), well below any τ1-relevant shift. Asserts adding the q_a axis + penalty does **not** move the q_h optimum. **Distinct from `D1_min_bias` below** — same letter "P-A" in the design heritage but a different quantity (this is argmax-stability of the inverse against the penalty addition; `D1_min_bias` is route-vs-true-`h` separation on the must-differ band). |
| `C_L1` form | **sigmoid `1 / (1 + exp(-k·(s_obs - 0.02)))`** | `[G]` centre at L1 line; `[E]` functional form | Smooth monotone ramp; sigmoid inflection at s = 0.02 (the [G] L1 boundary) where `C_L1 = 0.5`; one free parameter `k`. |
| `C_L1` steepness `k` | **`600`** | `[E]` | Calibrated to the boundary receipt. With sigmoid centre at `s = 0.02` and `reacquire_threshold = 0.05`, the 5%/95% crossings are at `s = 0.02 ± ln(19)/k = 0.02 ± 0.00491`. For both to fit inside the ±1.5° coincidence window in `s_obs` space — `[s(h_L1 - 1.5°), s(h_L1 + 1.5°)] = [0.01501, 0.02576]` (asymmetric because `sec h` is convex) — the binding constraint is the lower edge, requiring `k ≥ 590`. **k = 600** gives 5% crossing at `s ≈ 0.01509` → `h ≈ 9.89°` (inside lower window edge by 0.02°) and 95% crossing at `s ≈ 0.02491` → `h ≈ 12.66°` (inside upper window edge by 0.21°). The full 5–95% sigmoid transition therefore fits **inside** the ±1.5° window around `h_L1 ≈ 11.37°` by construction. By the §5 bridge convention (eligible-band route peak ≡ 1.0), bridge-mapped target intensity = `C_L1(s) · 1.0`, so PhotometricAgent's reacquire/lock-fail trips below `h ≈ 9.89°` and clears above `h ≈ 12.66°` — q3 L1 boundary coincidence within ±1.5° by construction, **not** by post-hoc k-tuning. |
| `T_cza` magnitude (additive when observed `f_cza = 1`) | **`0.3`** | `[E]` magnitude; `[G]` gating on observed `f_cza` | 30% of eligible-band route peak — meaningful enough that absence at `h > 32°` shows in landscape curvature, small enough not to swamp the route ridge. |
| `T_tan` magnitude (additive when observed `f_tan ≠ null`) | **`0.3`** | `[E]` magnitude; `[G]` gating on observed `f_tan` | Same reasoning. |
| C2-C(i) `detect_threshold_T` (Δ-curvature across coincidence window) | **`0.2`** (bridge-normalised) | `[E]` | Above bridge-scale numerical noise; below the 0.3 term magnitudes — a present↔absent transition reliably trips the detect, while a non-transition does not. |
| C2-C(ii) `separation_min` (`h`-distance between detected step centres) | **`2.0°`** | `[E]` | Larger than the ±1.5° coincidence window and smaller than the actual `32° - 29° = 3°` L2/L3 separation — a single ambiguous step cannot satisfy both loci. |
| `O_floor` (objective abstain floor in bridge-normalised units) | **`0.1`** | `[E]` | 10× above bridge-scale numerical noise; below the eligible-band landscape minimum after `C_L1` (which never dips below the upper-window 95% level inside the L1-eligible region). Degenerate rows trip; eligible rows do not. |
| `r_tol` (residual tolerance for "no valid solution exists") | **`0.66°`** | `[E]` | `1.5 · A = 1.5 · 0.44° = 0.66°`. Eligible rows always have a real residual `|r| ≤ A` (anchor noise is bounded by A); degenerate (`f_par_obs < R22`) rows have no real root because `arccos(R22/f_par_obs)` is undefined. |
| `κ_cond_max` (Hessian condition-number ceiling at the peak) | **`100`** | `[E]` | Order-of-magnitude tolerance; well-conditioned eligible-band maxima sit at ~10–30 (computed from `2λ/A² : 1/σ² = 10.3 : 4.0` plus C_L1 cross-coupling); flat-ridge degeneracies easily exceed 100. |
| Bridge scale (eligible-band route peak ≡ 1.0) | **`1.0` (convention)** | `[E]` convention, §5 anti-self-seal | Frozen *before* any receipt is computed; bridge is identity in the route channel. Anti-self-seal: prevents post-hoc scale tweaks that would make `C_L1` happen to cross `reacquire_threshold` at a convenient h. |
| `reacquire_threshold = 0.05`, `reacquire_hold_steps = 30` | (inherited) | `[G]` by C1 binding (`agents/photometric.py:70-71`) | Controller's own constants; not ours to set. |
| `D1_min_bias` (= C4 D1 / P-A floor on the must-differ band) | **`1.5°`** | `[E]` floor; `[G]` region (L1 band) | Pinned at `τ1` so a naive-equivalent route **cannot** accidentally pass τ1 in the must-differ region. With ρ=0.02 the analytic estimate `\|q_naive - h\| ≈ A / (\|sin h\| · f_par_obs)` predicts a bias `≈ 5°` at the L1 boundary and growing into the L1-ineligible band — floor comfortably clearable. **Distinct from `τ_C2-B-ii` above.** |

*Comparator semantics for `D1_min_bias`, frozen here (per Wave-2
sign-off).* The bias quantity is `b(h, ε) = |q_naive(h, ε) − h|` in
degrees (absolute value of signed q_naive minus signed true h). The
must-differ region is the open L1-ineligible band `{h : s_obs(h) <
0.02}` intersected with the frozen `h`-grid (so a finite enumerable
set). The floor passes iff `min_ε b(h, ε) ≥ 1.5°` for **every** `h` in
that set, where `ε` is sampled deterministically from
`Uniform[-A, +A]` under the frozen `seed`. **Not** compared against
`arccos(R22 / f_par_obs)` (that's the C2-B-equal-by-design analytic
inverse, identity-shaped — a non-bias by construction, not a vacuity
test).

*Scope: what this amendment does NOT close.* Wave 2 freezes the `[E]`
slots and the bridge-scale convention. It does **not** compute the
C2-A-1 (sustained-TRACK landscape vs `reacquire_threshold`), C2-A-2
(reproducible objective scan), or C2-A-3 (`C_L1` separation receipt
showing `C_L1(s_obs(h)) ≥ 1 - ε_C` for `h ≥ h*`) receipt tables.
Those are Wave 3, computed *against* the frozen `[E]` values above
under the §5 bridge convention. A failing receipt under these frozen
values **blocks** (append-only redesign), never silently tunes a
value above (A3).

*Honest tightnesses surfaced.* The k=600 calibration places the 5%
sigmoid crossing inside the lower window edge by only **0.02°** in h
— a tight but real margin under the `[G]` L1 boundary. The 95%
crossing is more comfortable at 0.21° inside the upper edge. If the
Wave-3 C2-A-1 receipt shows the controller's *actual* sustained-TRACK
behaviour transitions at the 5% crossing rather than at C_L1 = 0.5,
the q3 L1 coincidence pass margin is 0.02°. This is honest: the
boundary-receipt construction is real but not luxurious. A receipt
failure under k=600 may indicate the controller's transition is not
where the sigmoid 5% crossing predicts; that's a Wave-3 finding to
file as an append-only redesign, not a knob to tune k post-hoc.

Justification: closes the Wave-2 concrete-fill obligations for C2-A
(`[E]` values + bridge-scale convention + arithmetic correction +
`D1_min_bias` comparator semantics). No frozen body edited; no `[G]`
boundary moved; no Wave-3 receipt computed. Public-Language Constraint
remains fully in force. Cut-2-execute remains HELD on Wave 3 (C2-A-1/2/3
receipts), Wave 4 (C3-A `P_in` + receipts), Wave 6 (C4-A remaining
artifacts), Wave 7 (C4-B self-test), and the joint admission re-run.
