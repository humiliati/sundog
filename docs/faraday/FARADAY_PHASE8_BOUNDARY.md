# Shadow Faraday Phase 8 Receipt: Inhomogeneous (Sourced) Maxwell — Dual-Shadow Closure & Hodge Localization

**Status**: Landed 2026-05-31. Execution of [`FARADAY_PHASE8_SPEC.md`](FARADAY_PHASE8_SPEC.md).
**Parent ledger**: [`SHADOW_FARADAY.md`](SHADOW_FARADAY.md)
**Roadmap**: [`SUNDOG_V_FARADAY.md`](SUNDOG_V_FARADAY.md)
**Support battery**: `scripts/faraday-phase8-battery.mjs` -> `results/faraday/phase8-battery/manifest.json` (`npm run faraday:phase8`)
**Cross-substrate**: [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) §8.4.

## Outcome

The inhomogeneous half of Maxwell `d*F = J` closes through a metric-dependent dual
shadow, and the Hodge decomposition localizes **all** of the shadow's exact
regime-2 content to the harmonic / Aharonov-Bohm sector already named in Phase 7.
Maxwell-proper adds **no new sharp regime-2** — the pre-registered A8 + B8 landing,
now earned. No `C8` bounded failure. Hand derivation authoritative; the battery
mirrors it.

| Case | Domain condition (stated before computing) | Branch | Named result |
| --- | --- | --- | --- |
| 1. Electrostatic Gauss closure | `rho != 0`, static | **A8-dual-closure** | `oint.oint *F = Q_enc` (Gauss) |
| 2. Ampere-Maxwell + displacement current | `J`, time-varying `E` | **A8-dual-closure** | `oint B.dl = I_enc + d/dt flux_E` |
| 3. Hodge-split localization | non-contractible `U`, source + flux | **B8-harmonic-survivor** | field = sourced (determined) (+) harmonic (= Phase 7 AB); no new regime-2 |

## Locked inputs (inherited, unchanged)

- Phase 1 sign convention (`F_{0i} = -E_i`, `F_{ij} = epsilon_{ijk} B_k`, `c = 1`).
- Phase 2 `P_shadow`; Phase 3 homogeneous closure; Phase 7 boundary branches and
  the AB harmonic-sector result.
- The dual operator `P_shadow^dual = oint *F` is the **metric Hodge-dual** of the
  Phase 2 stencil — a new, metric-dependent operator, kept distinct from the
  metric-free `P_shadow`.

## The structural spine: where the metric enters

The homogeneous side was metric-free: `P_shadow = oint A = int F` closes `dF = 0`
using only `d` (Phase 3). The inhomogeneous side needs the **Hodge star**:

```
d*F = J     ->     oint.oint_{partial V} *F = int_V d*F = int_V J = Q_enc.
```

`*F` exchanges the roles of `E` and `B` through the metric, so the spatial part of
`*F` is the electric flux. The dual-shadow closure is the dual of Phase 3's
Stokes identity — structurally analogous — and the **new content** is (i) the
metric enters exactly here, (ii) the electric / magnetic source duality, and
(iii) the determinacy + Hodge localization (Case 3, the headline).

## Case 1 - Electrostatic Gauss closure -> A8-dual-closure

**Domain (pre-registered).** Static charge `rho != 0`, `J = 0`, `B = 0`.

**Hand derivation (authoritative).** The spatial component of `d*F = J` is
`div E = rho`; integrate over `V` and apply Stokes:

```
oint.oint_{partial V} *F = oint.oint_{partial V} E.dA = int_V rho dV = Q_enc.
```

**Witness — the dual of the Phase 7 monopole.** `E = (x, y, z)` (uniformly charged
ball). `div E = 3 = rho`; on the unit sphere `E.n = 1`, so
`oint.oint E.dA = 4 pi = Q_enc`. The arithmetic is identical to the Phase 7
magnetic-monopole witness — but **electric** charge lives in `d*F = J` (the normal
inhomogeneous law), where **magnetic** charge lived in `dF = K_m` (the forbidden
homogeneous break). The `*` is exactly the swap between them.

**Support battery (mirror).** `div E = 3.0000`, `oint.oint E.dA = 12.5665 = 4 pi = Q`.

**Branch: A8-dual-closure.**

## Case 2 - Ampere-Maxwell with displacement current -> A8-dual-closure

**Domain (pre-registered).** Current and/or time-varying `E`, so the displacement
current `partial_t E` is live.

**Hand derivation (authoritative).** The mixed component of `d*F = J` is
`curl B - partial_t E = J`; integrate over `S` and apply Stokes:

```
oint_{partial S} B.dl = int_S (J + partial_t E).dA = I_enc + d/dt flux_E.
```

The dual-shadow over a spacetime tube returns the conduction current **plus** the
displacement current — the distinctively-Maxwell term that the homogeneous side
never sees.

**Witness — pure displacement current.** `E = t z-hat`, `B_phi = r/2` (i.e.
`B = (-y, x, 0)/2`). Then `curl B = (0,0,1) = partial_t E`, so `J = 0`: a
source-free region where `partial_t E` alone drives `B` (the between-the-plates
region of a charging capacitor). Around a loop of radius `r0 = 2`:
`oint B.dl = pi r0^2 = 4 pi`, and `d/dt flux_E = (partial_t E_z)(pi r0^2) = 4 pi` —
equal, with no conduction current.

**Support battery (mirror).** `J = 2.8e-12 ~ 0`; `oint B.dl = 12.566371 = 4 pi`;
`d/dt flux_E = 12.566371 = 4 pi`; match residual `5.0e-13`. The displacement
current closes the loop to thirteen digits.

**Branch: A8-dual-closure.**

## Case 3 - Hodge-split localization -> B8-harmonic-survivor (the headline)

**Domain (pre-registered).** Non-contractible `U` (solenoid exterior, `H^1 != 0`)
carrying **both** an electric source `q` and the AB flux `Phi`.

**Hand derivation (authoritative).** The Hodge / Helmholtz decomposition splits the
potential into exact (gauge, unphysical) (+) co-exact (sourced) (+) harmonic
(topological). The two physical pieces are read by **orthogonal shadows**:

- **Dual-shadow reads the source.**
  `oint.oint *F = q` (Gauss). The AB piece is a magnetic flux tube with `E_AB = 0`
  outside, so it contributes nothing to the electric flux.
- **Loop-shadow reads the harmonic period.**
  `oint A = Phi` (the AB holonomy, Phase 7). The Coulomb piece is a pure scalar
  potential with vanishing vector potential, so `oint A_charge = 0`.

The readouts decouple exactly: `oint.oint *F` is blind to `Phi`, `oint A` is blind
to `q`. Therefore the **only** state-insufficient-yet-control-sufficient (exact
regime-2) content is the harmonic / AB sector from Phase 7; the sourced sector is a
lossy-but-determined Gauss summary, not a sharp separation. **Maxwell adds no new
regime-2.**

**Support battery (mirror).** Dual-shadow `chargeFlux = 12.5665 = 4 pi = q`;
loop-shadow `holonomy = 6.283185 = 2 pi = Phi`. Two orthogonal readouts, two
independent numbers.

**Branch: B8-harmonic-survivor.**

## Determinacy verdict (registered)

By the EM uniqueness theorem the sourced field is fixed by (sources, boundary data,
harmonic periods). The low-dimensional dual-shadow `Q_enc` is state-insufficient
(it does not recover the charge distribution) but furnishes **no exact,
objective-free regime-2** of the AB type. So the sourced sector is **determined**,
not resisting.

**Body-resistance verdict: four-for-four marginal, honestly disaggregated.**
Sourced classical EM joins NSE-C1 / Mesa / shell in the marginal column, but for a
**distinct reason** — not low-dimensional dynamics, but exact *determinacy by
sources*, with all sharp content topological and already counted in Phase 7. The
cross-substrate row is added at [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md)
§6.3, and §8.4 is promoted from pre-registered prediction to earned verdict.

## Forbidden-moves compliance

1. Phase 8 does **not** re-derive Maxwell; it tests dual-shadow closure of an
   already-given law.
2. The metric-dependent `P_shadow^dual` is kept explicitly distinct from the
   metric-free `P_shadow`; the metric role is named.
3. The sourced-sector determinacy is **not** presented as a new sharp regime-2;
   the only exact regime-2 is the Phase 7 harmonic / AB sector.
4. No residual or sector was invented after the algebra: the three cases and the
   determinacy verdict were registered in the spec.
5. The hand derivation is the proof; the battery is a seconds-scale mirror.
6. Free radiation, retarded Green's function, and curved spacetime were not
   touched — they remain the spec's named deferrals.

## Exit criteria (met)

- [x] Three registered cases classified under the A8/B8/C8 table.
- [x] Dual-shadow closure derived by hand; the metric role named.
- [x] Determinacy verdict and Hodge localization (regime-2 = harmonic / AB only)
  stated.
- [x] Support battery reproducible (`npm run faraday:phase8`) and linked.
- [x] Parent roadmap + ledger record the branches; §8.4 promoted to earned; §6.3
  gains the sourced-EM row.

## Claim boundary

Phase 8 completes the Maxwell picture without changing Branch A. The closure is the
**dual-Stokes** recovery of the integral form of the inhomogeneous law — not a
re-derivation of Maxwell from shadow primitives. The result is structural +
seconds-scale numerics on the registered static / given-source domain; free
radiation, retarded dynamics, curved spacetime, and quantum EM are named
deferrals. The unifying statement now on the record: **the entire classical EM
field is (sources, boundary, harmonic periods); the shadow's exact regime-2
separation lives entirely in the harmonic periods — which is why Aharonov-Bohm
(Phase 7) was the unique exact witness.** No public `faraday.html` copy is changed
by this receipt beyond the already-landed Phase 7 section; further public copy is
held for explicit go-ahead.
