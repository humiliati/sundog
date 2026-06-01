# Shadow Faraday Phase 7 Receipt: Source / Topology Boundary Audit

**Status**: Landed 2026-05-31. Execution of [`FARADAY_PHASE7_SPEC.md`](FARADAY_PHASE7_SPEC.md).
**Parent ledger**: [`SHADOW_FARADAY.md`](SHADOW_FARADAY.md)
**Roadmap**: [`SUNDOG_V_FARADAY.md`](SUNDOG_V_FARADAY.md)
**Support battery**: `scripts/faraday-phase7-battery.mjs` -> `results/faraday/phase7-battery/manifest.json` (`npm run faraday:phase7`)
**Cross-substrate**: [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) §8 (Case 3 earns §8.2)

## Outcome

Three registered in-scope cases, three pre-registered branches, all landed as
expected. No `C7` bounded failure: no unregistered residual appeared in any
magnetically-clean contractible computation. The hand derivation is
authoritative; the support battery mirrors it numerically.

| Case | Domain condition (stated before computing) | Branch | Named result |
| --- | --- | --- | --- |
| 1. Electric-source control | `J^mu != 0`, `dF = 0` | **A7-clean-control** | `R_F(S) = 0` (electric sources invisible to Faraday) |
| 2. Magnetic-source quarantine | registered `dF = K_m != 0` | **B7-magnetic-source** | survivor `= Q_m` (registered magnetic flux) |
| 3. Topological quarantine | non-contractible `U`, `H^1 != 0`, `F = 0` on loop | **B7-topology** | survivor `= oint A = Phi`; exact regime-2 separation |

## Locked inputs (inherited, unchanged)

- Phase 1 sign convention: mostly-plus, `c = 1`, `F_{0i} = -E_i`,
  `F_{ij} = epsilon_{ijk} B_k`.
- Phase 2 operator: `P_shadow^stencil[A] = oint_{partial omega} A`,
  `P_shadow^point[A] = F_{mu nu}`.
- Phase 3 success predicate and residual-table shape; Phase 4 tiny-case support
  style; Phase 5 quarantine vocabulary.
- Branches per the spec's Outcome table. No operator change, no Floquet/twist
  rescue, no new quarantine class.

## Case 1 - Electric-source control -> A7-clean-control

**Domain (pre-registered).** Smooth `A` on a contractible patch; electric
4-current `J^mu != 0`; magnetically clean (`F = dA`, so `dF = 0`).

**Hand derivation (authoritative).** `F = dA` gives `dF = d(dA) = 0` identically.
The Faraday components of `dF = 0` are `curl E + partial_t B = 0`, so for any
registered surface-loop pair

```
R_F(S) = oint_{partial S} E dot dl + d/dt int_S B dot dA
       = int_S (curl E + partial_t B) dot dA = 0.
```

The electric current enters only the inhomogeneous equation `d*F = J` and never
appears in `dF`. Faraday closure is therefore tolerant of electric sources **only
in the narrow Bianchi sense** - this is not, and must not be read as, a
full-Maxwell sourced claim.

**Witness.** `A = (0, 0, x^2 cos t)` -> `E = (0, 0, x^2 sin t)`,
`B = (0, -2x cos t, 0)`. Then `curl E + partial_t B = (0, -2x sin t + 2x sin t, 0) = 0`
(mixed partials cancel). Source: `J_z = curl B - partial_t E = -(2 + x^2) cos t != 0`,
with `rho = div E = 0`, `div B = 0`.

**Support battery (mirror).** Sample `(x,y,z,t) = (0.7, -0.4, 0.3, 0.6)`:
`R_F = 0`, `|J| = 2.0551` (source present), `div E = 0`, `div B = 0`. Matches the
hand value `|J_z| = (2 + 0.49)|cos 0.6| = 2.055`.

**Branch: A7-clean-control.**

## Case 2 - Magnetic-source quarantine -> B7-magnetic-source

**Domain (pre-registered).** A registered magnetic source inserted by hand,
`dF = K_m != 0`, smooth/regularized.

**Hand derivation (authoritative).** With `dF = K_m`, Stokes gives, for any
registered region `W`,

```
oint_{partial W} F = int_W dF = int_W K_m  =  registered source flux.
```

The plaquette-holonomy sum over `partial W` returns exactly the registered
source. For a static magnetic charge the spatial component is the survivor:

```
oint.oint_{partial V} B dot dA = Q_m = int_V rho_m dV.
```

The Faraday-loop component stays `0` (static, `E = 0`); the magnetic-Gauss flux
is the named survivor. This is a **named quarantine, registered before
computation** - not a clean-domain failure.

**Witness.** `B = (x, y, z)` (uniformly magnetically-charged ball).
`div B = 3 = rho_m` (continuous with the Phase 4 `dF_xyz = 3` monopole reading).
On the unit sphere `B dot n = 1`, so `oint.oint B dot dA = 4 pi = Q_m`.

**Support battery (mirror).** `div B = 3.0000`,
`oint.oint B dot dA = 12.5665 = 4 pi = Q_m` (`12.56637`); Faraday-loop residual `0`.

**Honest note.** A true point monopole has no global `A` (the Dirac-string
obstruction): `B = curl A` would force `div B = 0`. That is why the source is
*registered as `dF = K_m`* rather than produced by the holonomy operator from a
global potential. That no-global-`A` obstruction is the topological theme made
exact in Case 3.

**Branch: B7-magnetic-source.**

## Case 3 - Topological quarantine (Aharonov-Bohm) -> B7-topology

**Domain (pre-registered).** Non-contractible `U = R^3 minus solenoid`,
`H^1(U) = Z`; `F = 0` everywhere the loop lives; flux `Phi` threads the excluded
hole.

**Hand derivation (authoritative).** Outside the ideal solenoid
`A = (Phi / 2 pi) grad(theta) = (Phi / 2 pi)(-y, x, 0)/r^2`. Then

```
B = curl A = 0  on U     (curl of a gradient, r > 0)   =>   P_shadow^point = F = 0,
```

but `A` is closed (`dA = 0`) and **not exact** (`U` non-contractible), so the
loop tier

```
oint_{gamma} A dot dl = (Phi / 2 pi) oint_{gamma} grad(theta) dot dl
                      = (Phi / 2 pi)(2 pi) = Phi   != 0,
```

even though `F = 0` on `gamma`. The holonomy is gauge-invariant:
`oint_{gamma} d lambda = 0` for single-valued `lambda`. Faraday itself holds
(`curl E + partial_t B = 0`, static), so this is a **survivor, not a violation**:
the contractible-loop residual is `0`; the non-contractible holonomy `Phi` is the
named topology survivor, equal to the `H^1` period.

**Support battery (mirror).** Solenoid `a = 1`, loop `r0 = 2`, registered
`Phi = 2 pi`: point tier `F_on_loop = 1.6e-11 ~ 0`; loop tier
`oint A = 6.283185 = Phi` (to `1e-12`). Two-tier separation gap `= Phi`.

**The regime-2 reading (what this earns).** The two registered operator tiers
diverge by exactly the topological period:

| tier | reads | verdict |
| --- | --- | --- |
| `P_shadow^point = F` | `0` on every accessible path | **control-blind** - predicts no AB phase (wrong) |
| `P_shadow^stencil = oint A` | `Phi` | **state-insufficient** (one flux number != interior `B(x)`) yet **control-sufficient** (AB phase `= q Phi / hbar`) |

The loop-holonomy shadow is state-insufficient yet control-sufficient, and the
gap **is** `Phi = ` the `H^1` period - **exact and topological**, not a measured
near-coincidence. This is the portfolio's first *exact* regime-2 witness on a
real physical substrate. It promotes the cross-substrate Aharonov-Bohm row
([`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) §8.2) from
HYPOTHESIZED to **earned**.

**Honest bounds (carried, not weakened).** The resisting body is "small": one
integer per `H^1` generator, a *topological*, not *dimensional*, resistance. AB
does **not** close the open frontier of a high-dimensional resisting *control*
body on a real substrate (§6.3) - it is a different *kind* of resistance, not a
bigger one. Classical only: single-valued `lambda` keeps `Phi` exactly
gauge-invariant; quantum flux quantization / large gauge transformations are out
of scope (spec: no quantum EM).

**Branch: B7-topology.**

## Forbidden-moves compliance

1. Sourced/topological cases are **not** used to broaden the Phase 3 Branch A
   clean-domain claim. Branch A stays scoped to smooth, source-free, contractible.
2. The electric-current case (Case 1) is **not** called a Faraday failure:
   `dF = 0`, `R_F = 0`.
3. No gauge choice or global potential reconstruction is used as evidence; the
   only `A`-level object is the gauge-invariant single-plaquette / single-loop
   holonomy.
4. No residual class was invented after the algebra: all three were registered in
   the spec.
5. The hand derivation is the proof; the battery is a seconds-scale mirror, not
   the proof.

## Exit criteria (met)

- [x] All three registered cases classified under the Phase 7 branch table.
- [x] Electric-source Faraday closure (`dF = 0`) explicitly separated from
  full-Maxwell sourced dynamics (`d*F = J`).
- [x] Magnetic-source and topological survivors named (`Q_m`, `Phi`) before any
  public copy change.
- [x] Support script reproducible (`npm run faraday:phase7`) and linked.
- [x] Parent roadmap + ledger record the branches selected.

## Claim boundary

Phase 7 sharpens the *boundary* around the Phase 3 Branch A result; it does not
change it. Faraday's public claim remains the registered classical-vacuum,
magnetically-clean, contractible Branch A clean structural zero. New and
receipt-backed:

- electric sources leave Faraday closure intact (`A7`, narrow Bianchi sense only);
- registered magnetic sources produce a named flux survivor (`B7`), not a
  clean-domain failure;
- the Aharonov-Bohm topology survivor is an exact `H^1` regime-2 separation
  (`B7`) - the first exact regime-2 witness in the portfolio, bounded to
  topological (not dimensional) resistance.

No public `faraday.html` copy is changed by this receipt; that step is held for
explicit go-ahead and a gated build/deploy.
