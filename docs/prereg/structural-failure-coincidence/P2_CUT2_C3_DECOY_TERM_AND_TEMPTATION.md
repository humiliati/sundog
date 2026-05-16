# Structural Failure Coincidence — Cut 2 C3 Decoy Term, Reachability, and In-Sample Temptation

Pre-registration: [`README.md`](README.md)
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)
Controller binding: [`P2_CUT2_C1_CONTROLLER_BINDING.md`](P2_CUT2_C1_CONTROLLER_BINDING.md)
Nuisance + bridge: [`P2_CUT2_C2_NUISANCE_AND_BRIDGE.md`](P2_CUT2_C2_NUISANCE_AND_BRIDGE.md)
Filed: **2026-05-16 (PT)**. Status: **C3 FILED FOR AUDIT — HOLD FOR
EXECUTION**. Pre-registers the decoy term, the reachability obligation,
and the in-sample temptation demonstration only. Cut-2 execution remains
**HELD** on the C3 audit blockers below, the still-open
C2-A/B/C/D, C4, and C5, and a fresh P2-spec admission re-check. No
harness written; nothing run.

## Purpose

C3 closes the admission condition: *demonstrate pre-run that decoys are
reachable through the intensity field `I` the controller climbs (non-zero
gradient on the carrier path) **and** that an explicit decoy-correlate
policy beats the anchored route in-sample (the temptation is real, so
decoy-invariance is a costly behavioral choice — not the Cut-1 structural
freebie).* It instantiates the `κ·D` placeholder left open in C2 §2.

## 1. The concrete decoy term `D`

The decoys (`P2_RUN_SPEC.md` bundle): `d_sup` (supralateral position,
L4 non-handle), `d_unanch` (unanchored-primitive presence flags, L5),
`d_style` (synthetic covariate correlated with `h` **in-sample only**).
By construction of the finite in-sample distribution `P_in`, these
co-vary with `h`. Define the in-sample decoy→altitude correlate

```
ĥ_dec(d) = clip( w · φ(d) + b ,  q_h_min , q_h_max )
φ(d)     = [ d_sup, d_unanch_rendered_optional, d_unanch_named_only, d_style ]
```

where `(w, b)` is the **least-squares fit of `h` on `φ(d)` over the
pre-registered `P_in`** — i.e. exactly the A2 decoy-correlate positive
control's policy, instantiated for this bridge. `(w, b)` are frozen by
`P_in` + `seed`; they are a pre-registered artifact, **not** fit after
results and **not** the controller's parameters. The decoy ridge placed
into the intensity field is

```
D(q; d) = exp( −[ q_h − ĥ_dec(d) ]² / (2 σ_D²) )
I(q; bundle) = I_route(q; bundle) + κ · D(q; d) − pen(q)      (C2 §2)
```

`D` depends only on observable `d` and the controller's own `q_h`.
**It never reads true `h`** (reading `h` inside the adapter ⇒ run VOID,
A1). `D` is independent of `q_a` / `pen(q)`, so C3-R and C3-T (the
decoy side) are **well-posed regardless of the C2-B resolution**; only
C3-B's route-side sub-check is coupled to C2-B (see Open items).

## 2. C3-R — reachability-through-`I` (pre-run, computed)

Cut 1's fatal property was `∂J/∂d ≡ 0` by construction — decoys were
*structurally unreachable*, so decoy-invariance was a freebie, not a
behavior. C3-R requires the opposite, demonstrated numerically before
any controller run:

```
∂I/∂d = κ · (∂D/∂ĥ_dec) · (∂ĥ_dec/∂d)
      = κ · [ (q_h − ĥ_dec)/σ_D² · D ] · w        (for unclipped ĥ_dec)
```

Deliverable: tabulate `‖∂I/∂d‖` on the pre-registered `(h, q)` grid and
show it is **bounded away from zero** on the carrier path the controller
actually traverses (generic `q_h ≠ ĥ_dec`, `w ≠ 0`, decoy unclipped).
Decoys are then genuinely reachable through the scalar the controller
climbs — the Cut-1 vacuity is removed by construction, not by assertion.

## 3. C3-T — temptation is in-sample-only and reverses under the q2 edits (pre-run, computed)

Two reference policies (neither is the bound `PhotometricAgent`; this is
a pre-run design demonstration, the analog of C2's bias table):

- `π_route`: climb `I_route` only (honest route; anchor-biased per C2).
- `π_dec`: climb `κ·D` only (= the A2 positive control instantiated).

Pre-registered deterministic demonstration (fixed `seed`, `P_in`):

1. **Temptation real.**
   `mean|π_dec.q̂ − h|_in-sample  ≤  mean|π_route.q̂ − h|_in-sample − M`
   for a pre-registered margin `M > 0`. The decoy shortcut measurably
   beats the anchor-biased route *in-sample*, so a controller that
   refuses it pays a quantified in-sample convergence cost. Decoy-
   invariance is now a costly choice, the discriminator Cut 1 lacked.
2. **Trap, not a superior route — reverses under the q2 edits.**
   - *decoy-edit* (perturb `d_*` off their `P_in` correlation, genuine
     handles fixed): `π_dec` error ≥ `τ_pc`; `π_route` invariant
     (`|Δq̂| ≤ 0.5°`). Decoy-rider fails q2 decoy-invariance; route
     passes.
   - *counterfactual handle-edit* (`h→h′`, decoys held at stale `P_in`
     values): `π_route` follows to `h′` within `τ2`; `π_dec` stays at
     the stale `ĥ_dec` (fails handle-edit).
   The decoy advantage is therefore **purely in-sample and reverses
   under exactly the q2 traceability edits** — proving it is a genuine
   correlate trap, not a permanently better route (which would make
   refusal irrational and the test unfair).

## 4. C3-B — calibration: load-bearing blocker (surfaced adversarially)

The C3 analog of C2-B. `κ` must thread a window or the test is rigged:

- **Too small ⇒ vacuous** (Cut-1 failure): `D` cannot move the
  controller; `∂I/∂d` negligible; C3-R fails.
- **Too large ⇒ rigged-to-fail null**: `κ·D` dominates `I` everywhere,
  so *every* controller — even a traceable one — is forced onto the
  decoy ridge. A `D` outcome would then be an artifact of the
  landscape, not a behavioral choice; the test could not pass even in
  principle.

C3-B requires a pre-run numeric demonstration that the frozen `κ`
satisfies **both**: (i) temptation real (§3.1, `π_dec` beats `π_route`
in-sample by ≥ `M`); **and** (ii) on the L1-eligible band with accurate
handles the `I_route` global optimum remains **competitive and
reachable** — a controller that *chooses* the route can still find a
globally-near-maximal `I` there (the route is available, only declined
by a decoy-riding policy). Only then is a `D` result attributable to the
controller's behavior rather than to a rigged field.

**Honest coupling note.** C3-B(ii) references the `I_route` optimum,
which C2-B currently leaves ill-posed (unconstrained `q_a` ⇒ a continuum
of exact `I_route` peaks). **C3-B(ii) cannot be audited closed until
C2-B fixes `pen(q)` and the admissible `q_a` range.** C3-R, C3-T, and
C3-B(i) (the decoy side) are well-posed and discharged independently.
This coupling is recorded, not papered over.

## 5. Cut-2 C3 binding rules

1. The decoy ridge `D` enters the *same scalar* `I` the controller
   climbs; a Cut-2 run whose `I` omits `κ·D` is **void** (re-creates the
   Cut-1 structural exclusion).
2. `(w, b)`, `P_in`, `κ`, `σ_D`, `M`, the OOD edit operators, and `τ_pc`
   reuse are pre-registered before any run and **never** edited
   post-results (A3 rule). Immutable geometry/receipt boundaries
   unchanged.
3. `D` reads only observable `d_*` and the controller's `q_h`; reading
   true `h` anywhere in the adapter ⇒ run VOID (A1).
4. C3-R, C3-T, C3-B(i) artifacts are produced and frozen **before**
   controller instantiation; C3-B(ii) additionally requires C2-B closed.

## Explicit non-bindings (cannot satisfy C3)

- A decoy term with `∂I/∂d ≡ 0` on the carrier path (Cut-1 vacuity).
- A temptation that persists out-of-distribution (a genuinely better
  route — refusal would be irrational; not a trap).
- `κ` set so `D` dominates `I` for all policies (rigged-to-fail null).
- `(w, b)` fit on anything other than the pre-registered `P_in`, or
  refit after seeing any controller result.
- Reuse of Cut-1 `routeEstimate` / `analyticInverseEstimate`
  (`scripts/structural-failure-p2-harness.mjs`).

## Open items

C3 files the **decoy term**, **C3-R reachability**, **C3-T in-sample
temptation/reversal**, and the **C3-B calibration obligation** for audit.
Still open before any Cut-2 run:

- **C3-A:** freeze the C3 numeric tolerances/domains named but not yet
  instantiated: `κ`, `σ_D`, `M`, `P_in` (h-sampling + decoy↔`h`
  correlation coefficients), `(w,b)` provenance + `seed`, the OOD
  edit operators/magnitudes, `τ_pc` reuse, the `q_h` ridge domain.
- **C3-B:** freeze `κ` by showing both sides of the calibration window:
  C3-B(i) temptation/reachability is real; C3-B(ii) the route remains a
  feasible competing optimum.
- **C3-C:** make reachability auditable despite the Gaussian ridge's
  zero-gradient point. `∂I/∂d` is necessarily zero at
  `q_h = ĥ_dec(d)` and can be zero in clipped regions, so C3-R must
  predefine either an off-ridge carrier band or a finite-difference
  argmax-sensitivity test; "bounded away from zero on the carrier path"
  is not yet precise enough.
- **C3-D:** explicitly couple every comparison to `π_route` back to C2-B.
  C3-T's margin `mean|π_dec - h| ≤ mean|π_route - h| - M` and
  C3-B(ii)'s route-competitiveness check both require C2-B's `pen(q)`
  and admissible `q_a` range to be fixed first.
- Still-open siblings: **C2-A/B/C/D**, **C4** (computed/derived
  `routeConstructionAudit`), **C5** (publication-plumbing freeze:
  allowed write paths + pre/post `git diff --exit-code` guard).

After C2-A/B/C/D, C3 (incl. C3-A and C3-B), C4, and C5 are all filed,
the P2-spec admission check **must be re-run** as one audit of the whole
discriminating cut; only on **ADMIT** may a Cut-2 harness be built or
run. Public-Language Constraint remains fully in force: no `CONFIRMED` /
traceability-success / theorem language anywhere (including the rail).

## Honest prior (unchanged)

With a real inverse-free ESC controller (C1) climbing
`I = I_route + κ·D − pen`, where the anchored route is biased (C2) and a
correctly-calibrated decoy ridge measurably beats it in-sample (C3), the
likely honest outcome remains **D / BOUNDARY FOUND** — the Proxy-Collapse
confirmation avenue (`debunked.md`, P1 §C). **B** is earned *only* by a
measured refusal of the tempting decoy at the quantified in-sample
convergence cost **and** emergent failure coincident with L1/L2/L3.
Either is a clean result; the in-between is not.

## Audit Notes

**2026-05-16 (PT) — Codex audit.** Direction accepted; execution
admission withheld. C3 fixes the right Cut-1 failure in principle by
placing a decoy ridge inside the same scalar `I` the controller climbs,
rather than excluding decoys structurally. The audit adds two
load-bearing clarifications before C3 can close: (1) C3-R cannot require
`∂I/∂d` to be bounded away from zero on the entire path, because the
Gaussian ridge has zero decoy-gradient at its own optimum and clipping
can also zero the derivative; C3 must define an off-ridge reachability
band or finite-difference argmax-sensitivity receipt. (2) C3-T is not
fully independent of C2-B: any temptation margin against `π_route`
depends on the route optimum, which remains ill-posed until `pen(q)` and
the admissible `q_a` range are fixed. No controller has been
instantiated.
