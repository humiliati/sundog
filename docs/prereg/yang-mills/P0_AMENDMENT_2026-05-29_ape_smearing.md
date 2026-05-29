# P0 Amendment 1 - APE Smearing as Admissible Primary-Signature Class

Filed: **2026-05-29 (PT)**
Amendment id: `P0_AMD_001_APE_SMEARING_2026-05-29`
Parent P0 lock: [`P0_DOMAIN_AND_RECEIPT_LOCK.md`](P0_DOMAIN_AND_RECEIPT_LOCK.md)
Triggering probe spec:
[`../../yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md`](../../yang-mills/specs/2026-05-29_phase2_v1_smearing_probe.md)
Triggering null receipt:
[`../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md`](../../yang-mills/receipts/2026-05-29_SU2_3D_phase2_no_rank_local_structure.md)

Status: **P0 amendment, design lock**. This document modifies the
parent P0 lock as specified below. It admits no claim by itself and
admits no runner code by itself. The binding consumer of this amendment
is
[`PHASE2_SU2_3D_relative_locality_v1.md`](PHASE2_SU2_3D_relative_locality_v1.md).

## Purpose

> Promote APE smearing from "explicitly deferred at P0" to "admissible
> primary-signature class with frozen smearing parameters," scoped to
> a new signature vocabulary version v4. The bare-link vocabulary v1
> remains valid. The held-out target vocabulary v1 is unchanged. The
> rest of the P0 lock (cell ladder, lattice slate, action, β slate,
> generator, burn-in / thinning / autocorrelation rules, leakage
> controls battery, admission requirements, outcome-branch table,
> anti-scope-creep rule, public-language boundary, and reviewer
> category) is unchanged.

## Scope

This amendment modifies, in the parent P0 lock:

- §"Scope" — under "Out of scope at this lock," remove the bullet that
  defers smearing/blocking, and replace it with a narrower bullet that
  defers blocking only;
- §"Primary Signature Vocabulary (locked)" — add a new vocab v4
  alongside the existing v1; clarify v1's frozen status;
- §"Held-Out Observable Label (locked)" — unchanged in its locked
  values; add a one-line clarification that smearing is not applied to
  the held-out target;
- §"Forbidden as primary signatures" — narrow the smearing prohibition
  to "smearing parameters not in the locked set of any filed P0
  amendment," with this amendment being the first such filing;
- §"Open Decisions Closed By This Lock" — add a row noting open
  decision #4 ("Phase 2 fixed loop sets only, or include
  smearing/blocking levels?") is now resolved by this amendment for
  smearing only; blocking remains deferred.

Every other section of the parent P0 lock is unchanged.

## Locked Smearing Parameters

The following parameters are frozen at this amendment and **may not
be retuned** after any receipt is filed using them. A future probe
spec proposing different smearing parameters or a different smearing
algorithm requires a new dated P0 amendment.

| Field | Registered value |
| --- | --- |
| Smearing algorithm | APE smearing (Albanese, Falcioni, et al. 1987, lattice-QCD standard); see § "Algorithm" below |
| Smearing fraction `α` | `0.5` (lattice-QCD standard fraction; in the centre of the typical 0.3–0.8 published range) |
| Smearing iteration count `N_sm` | `10` (lattice-QCD standard depth for 3D SU(2); large enough to suppress UV noise on small Wilson loops, small enough to preserve area-law structure) |
| Re-unitarization | exact SU(2) projection after **every** iteration (see § "Projection" below) |
| Cells admitted | all P0 SU(N) cells (`SU2_2D`, `SU2_3D`); U(1) cells excluded because the U(1) smearing analogue is a separate algorithm and is not in scope here |
| Held-out target | unchanged; smearing **not** applied to held-out loops |

The `α = 0.5`, `N_sm = 10` choice is documented as a frozen
lattice-QCD-standard default, not the result of a parameter scan. No
parameter scan over `(α, N_sm)` is admissible at v1; if v1 lands a
null, the next probe spec must propose a different design knob entirely
or a fresh P0 amendment for a new `(α, N_sm)`, not a silent retune.

## Algorithm

For each link variable `U_μ(x) ∈ SU(2)` and each smearing iteration
`s = 1 ... N_sm`:

```text
staple_sum_μ(x) = sum over the 2·(D-1) plaquettes touching the link
                  of the staple matrices, where a staple is the
                  ordered product of the three other links in that
                  plaquette traversed so as to share endpoints with
                  U_μ(x).

M = (1 - α) · U_μ(x) + (α / (2·(D-1))) · staple_sum_μ(x)

U_μ(x) ← project_SU2(M)
```

`D` is the spacetime dimension of the cell (`D = 2` for `SU2_2D`,
`D = 3` for `SU2_3D`), so the staple normalization is `α/2` in 2D and
`α/4` in 3D, matching the standard APE convention. All iterations are
applied simultaneously across links (synchronous update); the update
must read from the snapshot at the start of the iteration so that
within an iteration no link uses its already-smeared value.

## Projection

The closest-SU(2) projection of a 2×2 complex matrix `M` close to SU(2)
is:

```text
project_SU2(M) = M / sqrt(det(M))
```

where the complex square root branch is chosen so that
`Re(Tr(M / sqrt(det(M)))) >= 0`. For SU(2) elements `det(M) = 1` and
the projection is the identity; for APE-smeared linear combinations
`M` is close enough to SU(2) that this projection is well-defined and
unique. The runtime manifest must record the maximum
`|det(M) - 1|` observed across all link updates as a smearing-health
gate (see § "Health Gates").

## Why The Held-Out Target Stays Bare

Smearing the held-out loops `W14`, `W23`, `W33` would mix the
relative-locality test with the smearing operation itself. The
honest test of "does the smeared-loop signature preserve bare-loop
area-law decay structure" requires the target to remain in bare-link
space; if the target were also smeared, a positive read could be
explained as "smearing makes everything correlated with itself."
Keeping the target bare gives a stricter test: the signature has been
moved to a different (smeared) representation, but the truth label it
is being scored against still lives in the original bare representation.

This choice is locked at this amendment and is not revisable in any
downstream Phase 2 vN spec without a fresh P0 amendment.

## Health Gates Added By This Amendment

Any Phase 2 (or future Phase 3 / Phase 4) receipt using vocab v4 must
report and gate:

| Quantity | Registered threshold | Branch if missed |
| --- | --- | --- |
| Max `|det(M) - 1|` over all post-smearing-step matrices `M` after projection | `<= 1e-10` Frobenius equivalent on `M_SU2 · M_SU2^† − I` | `YM-P*-QUAR-E smearing_drift` |
| Smeared-link unitarity max Frobenius residual | `<= 1e-10` after each iteration | `YM-P*-QUAR-E smearing_drift` |
| Smeared mean plaquette per orientation | reported, no gate | reported only |
| Per-orientation smeared mean plaquette spread | `<= 5e-2` | `YM-P*-QUAR-C orientation_anisotropy` (existing branch, extended to smeared signature) |

`YM-P*-QUAR-E smearing_drift` is a new named quarantine introduced by
this amendment. `P*` is the phase index of the consuming receipt
(`P1`, `P2`, `P3`, or `P4`).

## Compute Cost Impact

Smearing N_sm = 10 iterations on a 12³ × 3 link directions = 5184
links involves 10 · 5184 = 51,840 link updates per configuration, each
requiring a staple sum (4 plaquettes touching the link in 3D) and a
2×2 complex matrix-matrix add + projection. Per the Phase 1 SU(2) 3D
baseline (~0.27 µs per matmul), this is well under one second per
configuration. The full Phase 2 v1 aggregation pass on 96 configurations
therefore adds well under two minutes of smearing time, with the rest
of the aggregation (signature recomputation, NN graph, scoring) being
unchanged from v0. The P0 ten-minute-per-invocation cap remains in
force.

## Forbidden, Restated

This amendment forbids:

- using `α` or `N_sm` values other than `0.5` and `10` in any receipt
  citing this amendment;
- smearing the held-out target loops `W14`, `W23`, `W33`;
- applying smearing inside the per-β ensemble generation (the bare
  ensembles must remain identical to v0 for the head-to-head
  comparison; smearing is applied only inside the v1 aggregation
  runner reading those ensembles);
- promoting any smearing-based result to vocab v1 status (vocab v4
  is its own pre-registered class and labelled as such in every
  receipt);
- using this amendment to promote 4D Yang-Mills work (4D remains
  explicitly deferred per the parent P0 lock).

## Open-Decision Resolution

Parent P0 lock §"Open Decisions Closed By This Lock" item 4 was:

> 4. Phase 2 fixed loop sets only, or smearing/blocking? — **Fixed loop
>    set only at P0. Smearing/blocking deferred to a later P0
>    amendment after the fixed-loop signal either passes or fails.**

That row is now amended as follows:

> 4. Phase 2 fixed loop sets only, or smearing/blocking? — Fixed loop
>    set only at P0 (v1 vocabulary). Smearing admitted at the
>    `P0_AMD_001_APE_SMEARING_2026-05-29` amendment as vocab v4 with
>    frozen `(α, N_sm) = (0.5, 10)` after the v0 fixed-loop signal
>    landed as `YM-P2-NEG-A`. Blocking remains explicitly deferred.

## Current State

- 2026-05-29: amendment filed against parent P0 lock; vocab v4
  admitted; smearing parameters frozen at `(α, N_sm) = (0.5, 10)`.
  Triggers the binding Phase 2 v1 spec at
  [`PHASE2_SU2_3D_relative_locality_v1.md`](PHASE2_SU2_3D_relative_locality_v1.md).
