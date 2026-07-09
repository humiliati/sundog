# H3-PL Result — BOTH CLAUSES VOID: the frozen spec had two unfireable gates (caught at run)

**Verdict: NO SCIENTIFIC CLAIM.** The run (`scripts/shadow_introspect_privilege.py`, result
`results/atlas/h3/privilege_result.json`) executed the FROZEN prereg (v2, commit `815ea529`)
faithfully and, in doing so, proved that **two of its clauses cannot fire on the substrate it
pins.** This is a pre-registration error surfaced by running — not a null, not a finding. The
corrections require re-registration (§ Disposition); nothing here is patched-and-rerun.

> commands: `python scripts/shadow_introspect_privilege.py` · `python scripts/test_shadow_introspect_privilege.py` (9/9)

## What is valid

The de-saturated body and its gates are sound: **C0 raw-mean c-R² = 0.0** (≤ 0.10 ✓), **C1
single-unit c-R² = 0.999** (≥ 0.50 ✓), **pooled d-acc = 0.805** (in [0.75, 0.85] ✓), train-fit 0.865.
The pinned `SIGMA_D = 9.0` reproduced. The verdict LOGIC (`classify_access` / `classify_backaction`)
ran exactly as the frozen test locked it — the defects are in the THRESHOLDS' reachability, not the
code.

## Defect 1 (BACK-ACTION) — the d-acc gate is above the body's ceiling

§5 gates the back-action clause on `d-acc(joint) ≥ 0.90`. §7 de-saturates the body to pooled d-acc
**0.75–0.85 by design** (that is the whole point of the tuning pass). A joint body built on that
substrate cannot reach 0.90: measured `d-acc = 0.789 / 0.789 / 0.794` at β = 0.3 / 1.0 / 3.0. So the
clause is `objective_abandoned` for EVERY β — structurally, before any science. **`D_ACC_GATE (0.90)
> DESAT_BAND[1] (0.85)` is a self-consistency contradiction** between §5 and §7 that the review and
the frozen test both missed (they checked the gate's LOGIC, not whether it has a reachable region on
the pinned body).

*Buried signal (not a verdict — the clause is void):* the joint body did recover c substantially —
`rec_joint ≈ 0.41` (β = 3.0) up from the pooled ceiling 0.049 — with d-acc drift **outside** the k=20
null (0.012–0.017 vs null 0.0067) at all β, i.e. the encoder restructured beyond same-objective
seed variation. Under a *reachable* gate this would read as (d) RESTRUCTURING-PRICED RECOVERY. It is
recorded, unclaimed.

## Defect 2 (ACCESS) — the per-unit arms are measured where the substrate washes c

§3 reads all three ceilings at **λ = 2.0**. But the substrate spreads per-unit c as `c_i = c + λ·ξ`,
so at λ = 2.0 per-unit c is washed **by design** — C1 (the ACCESS reference the B1 fix leans on) is
measured at **λ = 0**. Result: both per-unit arms come back **all-battery-blind** (ridge, Nyström,
GBT all blind even to *injected* c; ceilings 0.0), so `m_enc = 0.0` and `m_pool = −0.049` are
differences of near-zero numbers and the `c_partial_mixed` verdict is **vacuous**. Two coupled
causes:

- **λ mismatch (D9a):** the arms must be read where per-unit c exists (λ ≈ 0, matched to the C1
  reference), not at λ = 2.0 where the substrate washes it per-unit. The B1 fix named the right
  contrast (trained-vs-raw per-unit) but §3 pinned it at the wrong λ.
- **injection too weak for high-dim arms (D9b):** {0.10, 0.20}-strength injection along one direction
  in a 128-PCA space contributes R² ≈ s²/128 ≪ 0.05, so the battery reads MEMBER-BLIND on the
  per-unit arms even where c is present. Injection must target a fixed detectable-R², not a fixed
  strength (the B4 fix matched dimensionality but not injection sensitivity to it). Pooled (32-dim)
  is barely live (P1 floor 0.10, cv-R² 0.0498), which is itself a warning the calibration is marginal.

## Root cause + the honest meta-pattern

All three defects (D8 gate-vs-band, D9a arm-λ-vs-C1-λ, D9b injection-vs-dim) are **achievability /
substrate-interaction** failures: none is a within-section logic error, so the adversarial review
(which hunts logic/foreordination) and the frozen test (which locks logic) both passed them. **This
is the THIRD consecutive spec defect caught only by running** — HS9's C1 (a three-pass functional
mislabeled single-pass, caught by the pilot), HS1's B1 (a data-processing tautology, caught by the
review), and now HS1's D8/D9 (unreachable gates, caught by the run). The pattern is consistent: the
defects that survive are the ones that only appear when constants meet the substrate.

**Process fix (proposed):** add an **achievability dry-run** to the pre-freeze checklist — a
tiny-data pass that checks every gate/threshold has a reachable region GIVEN the pinned substrate
constants (here: assert `D_ACC_GATE ≤ DESAT_BAND[1]`; assert each arm's battery is live at the arm's
λ before freezing; assert the injection yields a detectable R² at the arm's dimensionality). A dozen
seconds of dry-run would have caught all three.

## Disposition — re-registration required (NOT a silent re-run)

HS1 is **not resolved**. The corrections change the measurement, so they re-enter through a v3
re-freeze, not an edit-and-rerun:

1. **Back-action gate → body-relative:** `d-acc(joint) ≥ base_d_acc − δ` (e.g. δ = 0.03) instead of a
   fixed 0.90; the clause is about *retaining* d-competence, which is 0.80 on this body.
2. **Access arms → matched-λ:** read the three ceilings at λ = 0 (the C1 reference), or add a low-λ
   rung; the trained-vs-raw per-unit contrast is only meaningful where per-unit c exists.
3. **Injection → fixed-R² target:** calibrate strength per arm to a detectable-R² floor, not a fixed
   {0.10, 0.20}; verify battery liveness on every arm before freeze.
4. **Frozen test → add the achievability assertions** above so these cannot recur.

Owner call: invest a v3 cycle (re-register + re-run, ~1 day with the fixes), or bank HS1 as
spec-defective and move to HS5. Either way the run script, config, tuning, and this receipt stand as
the record; no claim is made from a void spec.
