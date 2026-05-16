# Structural Failure Coincidence — Cut 2 C2-C (leverage-confidence gate) + C2-D (`f_par_obs < R22` invalid handling)

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Parent condition: [`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md) (C2-C, C2-D)
Depends on: [`P2_CUT2_C2B_PEN_AND_QA.md`](P2_CUT2_C2B_PEN_AND_QA.md) (route optimum geometry)
Audited by: [`P2_CUT2_C4_DERIVED_AUDIT.md`](P2_CUT2_C4_DERIVED_AUDIT.md) (D3 / C4-D taint test)
Filed: **2026-05-16 (PT)**. Status: **C2-C / C2-D FILED FOR AUDIT —
HOLD FOR EXECUTION**. Design only; numerics fold into the C2-A freeze.
Cut-2 execution remains **HELD** on the load-bearing items below, the
still-open C2-A, C3-A/B/C/D, C4-A/B/C/D, and C5, and a fresh P2-spec
admission re-check. No harness written; nothing run.

## Purpose

The last two C2 design sub-blockers. C2-C specifies the
leverage-confidence gate that makes the L1/L2/L3 boundary degradation
**emergent in the landscape the controller climbs** (not a flag it
reads). C2-D pre-registers the classification of the
`arccos(R22/f_par_obs)`-undefined rows. Closing both means C2-A can
later freeze a *complete* C2 in one numeric pass.

## 1. C2-C — the leverage-confidence gate

The C2-B-resolved route optimum is sharpened/limited by an
observable-only envelope. The gated route field:

```
I_route_full(q;bundle) = C_L1(s_obs) · [ P(q;bundle)
                          + 1[f_cza_obs] · T_cza(q;bundle)
                          + 1[f_tan_obs] · T_tan(q;bundle) ]
```

- `P` = the C2/C2-B parhelion ridge
  `exp(−[f_par_obs − R22/cos(q_h) − q_a]² / 2σ²)`.
- `s_obs = f_par_obs / R22 − 1` = **observed** parhelion leverage —
  a function of the observable `f_par_obs` and the constant `R22`,
  **no `h`** (A1-safe).
- `C_L1(s_obs)` ∈ [0,1] = a **smooth monotone** confidence ramp:
  → 0 well below the L1 receipt line, → 1 well above, transition
  centred on the L1 `2%·R22` line. Smooth (not a step) because L1 is a
  *noise-bounded gradual loss of informativeness* (anchor noise), so
  graded confidence is the honest model and the controller's L1
  degradation is a gradual flattening (emergent, matching the L1
  prediction "reports low leverage").
- `T_cza`, `T_tan` = closed-form CZA / tangent **consistency terms**
  that add ridge identifiability, each gated by the **observed**
  presence indicator (`f_cza_obs ∈ {0,1}`, `f_tan_obs ∈ {present,
  null}`; A1: observable bundle state, no `h`). When `f_cza_obs==0`
  (`h>32°`, L2) the CZA term is **absent** — the handle genuinely
  vanished, exactly the documented singularity ("a real inverse carries
  its singularities with it"); likewise `f_tan` null at `h≥29°` (L3).

**Emergent vs flag-read (honest tension, surfaced).** The
`1[f_cza]`/`1[f_tan]` indicators are discontinuous at the L2/L3 loci.
That discontinuity is the **world's real singularity** (the receipt: the
CZA disappears, the tangent merges), not an artifact. The controller
does **not branch** on a bit or on `h`; it climbs whatever `I` is, and
experiences the term's removal as a sudden loss of a consistency
constraint → ridge flattens → it must degrade / fall back / abstain.
From the controller's side that is emergent. **Whether it is genuinely
emergent (no code branch on `f_cza`/`f_tan`/`h`) is exactly what C4-D's
D3 taint + boundary-perturbation test must mechanically verify.** C2-C
defines the structural source; C4-D audits the controller's response.
Coupling recorded, not papered over.

**C2-C load-bearing calibration window** (the C2-C analog of
C2-B/C3-B): the magnitudes of `T_cza`/`T_tan` relative to `P` and the
`C_L1` steepness must thread a window —

- **Too small ⇒ boundary invisible.** Removing `T_cza`/`T_tan` barely
  changes the landscape; no measurable degradation at L2/L3; q3
  boundary-coincidence undetectable (a vacuity).
- **Too large ⇒ rigged-to-fail past the boundary.** The consistency
  terms dominate `P`; past L2/L3 the landscape collapses so hard that
  *every* controller fails for reasons unrelated to route-use.

C2-C is not closed until a pre-run numeric demonstration (folding into
C2-A) shows the frozen magnitudes/steepness make the L2/L3
identifiability loss **(i) detectable** — a route-using construction
measurably degrades/abstains/switches within the ±1.5° window — **and
(ii) discriminating** — a decoy-rider can still sail through, so the
boundary separates route from correlate rather than failing everyone.

## 2. C2-D — `f_par_obs < R22` invalid handling

When the anchor noise drives `f_par_obs = R22/cos(h) + ε < R22`, then
`R22/f_par_obs > 1` and `arccos(R22/f_par_obs)` is **undefined**. C2-B
already showed the geometry: no zero-bracket manifold exists
(`R22/cos(q_h) ≥ R22 > f_par_obs` ∀ `q_h∈[0,80]`), so the unique
`O`-max sits at a low-value penalty-best-fit domain edge with `r≠0`
everywhere.

**Pre-registered classification (no-fabrication discipline).** These
rows are **invalid / ineligible** — the documented inverse is *literally
undefined*, the sharpest possible structural-failure-coincidence point.
The controller must **abstain** (no confident `q̂`). **Clipping is
forbidden** (it would fabricate a number, violating the prereg's L1
prediction "does not count invalid photos as independent inverse
evidence").

**Scoring:**

- **Not counted toward q1** (no confident estimate expected on invalid
  rows).
- **Scored under q3-L1 boundary-coincidence:** a route-using controller
  abstains here (its objective is degenerate per C2-B); a controller
  that emits a confident `q̂` where the inverse is undefined is
  **provably** riding a correlate ⇒ q3 fail. This makes `f_par_obs<R22`
  a built-in, zero-ambiguity correlate detector — it **strengthens** the
  falsifier.
- Classification is on the **observed** per-sample `f_par_obs`
  (`s_obs < 0`; A1-observable, no `h`), pre-registered and deterministic
  given the frozen seed (the boundary is stochastic in `ε` but fixed by
  the seed).

**Emergent-abstain (honest coupling).** The abstain must be the
controller's **emergent** behavior — it abstains because its objective
`O` is degenerate there (low, no `r=0`, flat penalty-edge max, per
C2-B), **not** an injected `if f_par_obs < R22: abstain` branch (that
would be a flag-read ⇒ C4-D / D3 fail, and a hidden-derived gate).
C2-D therefore couples to **C2-B** (the degenerate-objective geometry
that *causes* the emergent abstain) and to **C4-D** (the taint test must
confirm no `f_par_obs<R22` branch). The abstain detector threshold
(what counts as "abstained": `O*` below a confidence floor / no `r=0`
within tolerance) is a pre-registered number folding into C2-A.

## 3. Shared coupling (stated once)

C2-C and C2-D specify *the world's structural boundaries* (the gated
landscape and the undefined-inverse rows). **C4-D's D3 taint /
boundary-perturbation test is the audit** that the controller's response
to those boundaries is emergent, not flag-read. The two reference each
other by role (source vs audit); this is not circular.

## 4. Cut-2 C2-C / C2-D binding rules

1. `C_L1`, `T_cza`, `T_tan`, and the invalid-row gate read only
   observable bundle fields + carrier `q`; any `h`-dependence ⇒ run
   **VOID** (A1).
2. `f_par_obs < R22` rows are **abstain/invalid**, never clipped; a
   confident `q̂` there is a q3 correlate failure.
3. The boundary/abstain behavior must arise from the landscape `I`, not
   a controller branch on `f_cza`/`f_tan`/`f_par_obs<R22`/`h` (C4-D
   verifies).
4. All C2-C/C2-D numerics (the `C_L1` steepness, `T_*` magnitudes, the
   abstain-detector threshold, the C2-C(i)/(ii) demonstrations) are
   pre-registered with the C2-A freeze, never edited post-results (A3).
   Immutable geometry/receipt boundaries unchanged.

## Explicit non-bindings (cannot satisfy C2-C / C2-D)

- A discrete L1 step gate (L1 is noise-bounded; the ramp must be smooth)
  or an `h`-dependent `C_L1`.
- `T_cza`/`T_tan` magnitudes outside the C2-C(i)/(ii) window.
- Clipping `f_par_obs<R22` to a number, or counting it toward q1.
- An injected `if f_par_obs<R22: abstain` (or `if f_cza==0: …`) branch —
  the abstain/degradation must be emergent from `I`.
- Tuning any C2-C/C2-D number after seeing a controller result.

## Open items

C2-C and C2-D file their **functional forms, classification rule,
load-bearing windows, and couplings** for audit. Still open before any
Cut-2 run:

- **C2-A** now absorbs the complete C2 numeric freeze: `ρ, A, σ, seed`,
  the grid + `q_h/q_a` domains, the C2-B `λ`/conditioning-floor/P-A
  tolerance, **and** the C2-C `C_L1` steepness + `T_cza`/`T_tan`
  magnitudes + C2-C(i)/(ii) demonstrations + the C2-D abstain-detector
  threshold. The C2 design layer is now complete; C2-A can freeze it in
  one pass.
- **C2-C ↔ C4-D** and **C2-D ↔ C2-B / C4-D** couplings must be
  discharged at the joint admission re-run.
- Still-open siblings: **C3-A/B/C/D**, **C4-A/B/C/D**, **C5**.

After C2-A, C3-A/B/C/D, C4-A/B/C/D, and C5 are all filed, the P2-spec
admission check **must be re-run** as one audit of the whole
discriminating cut; only on **ADMIT** may a Cut-2 harness be built or
run. Public-Language Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

A gated honest route (C2-C) with a sharp built-in undefined-inverse
correlate detector (C2-D), climbed by a real inverse-free ESC controller
(C1) against a biased signal (C2/C2-B) with a tempting reachable decoy
(C3), keeps the likely honest outcome at **D / BOUNDARY FOUND** — the
Proxy-Collapse confirmation avenue (`debunked.md`, P1 §C). **B** is
earned *only* by a measured refusal of the tempting decoy at the
quantified in-sample cost **and** emergent failure coincident with
L1/L2/L3. Either is a clean result; the in-between is not.

## Audit Notes

*(reviewer space — append-only below)*
