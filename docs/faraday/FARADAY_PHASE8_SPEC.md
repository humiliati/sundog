# Shadow Faraday Phase 8 Spec: Inhomogeneous (Sourced) Maxwell — Dual-Shadow Closure & Hodge Localization

**Status**: Opened 2026-05-31; spec registered, execution not started.
**Parent ledger**: [`SHADOW_FARADAY.md`](SHADOW_FARADAY.md)
**Roadmap**: [`SUNDOG_V_FARADAY.md`](SUNDOG_V_FARADAY.md)
**Prior**: Phase 3 (homogeneous `dF = 0`), Phase 7 (source/topology boundary,
[`FARADAY_PHASE7_BOUNDARY.md`](FARADAY_PHASE7_BOUNDARY.md)).
**Cross-substrate**: [`../CROSS_SUBSTRATE_NOTES.md`](../CROSS_SUBSTRATE_NOTES.md) §8.4.
**Purpose**: Complete the Maxwell picture. Phase 3 closed the homogeneous half
`dF = 0` (the Bianchi identity, metric-free); Phase 7 bounded it and found the
lone exact regime-2 in the harmonic / Aharonov-Bohm sector. Phase 8 audits the
inhomogeneous half `d*F = J` — the metric-and-source side — through a
**dual-shadow** operator, and locates where the shadow's regime-2 content lives.

## Phase 8 Thesis

The inhomogeneous Maxwell law is the *other* half of the field equations:

```
d*F = J     ->   div E = rho   and   curl B - partial_t E = J     (c = 1).
```

Unlike `dF = 0`, this is **not** an identity — it is the sourced, metric-bearing
law. Phase 8 does **not** re-derive it from shadow primitives. It asks whether the
same operator discipline *closes* the sourced law through a metric-dependent dual
shadow, and where the shadow's regime-2 content sits once sources and the metric
enter.

The pre-registered answer, to be earned by hand: the sourced sector is
**determined** by its sources (the EM uniqueness theorem), so it furnishes no new
sharp separation; the **Hodge decomposition** localizes ALL exact regime-2 content
to the harmonic sector — exactly the Phase 7 Aharonov-Bohm witness. Maxwell-proper
adds **no new sharp regime-2**.

Load-bearing corrections for this phase:

- The Hodge star `*` (hence the **metric**) enters here, and only here; the
  homogeneous side (Phase 3/7) was metric-free / topological.
- "Closure" means the dual shadow recovers the **integral form** of the
  inhomogeneous law (Gauss / Ampere-Maxwell), **not** a re-derivation of Maxwell
  from shadow primitives.
- Time-varying sources (displacement current `partial_t E`) are in scope and are
  "dynamical" in the time-varying sense. Free radiation, the retarded Green's
  function, and curved spacetime are named deferrals (below).

## Registered Question

Given the Phase 2 plaquette-holonomy operator `P_shadow` and its Phase 3
homogeneous closure, does a metric-dependent dual operator

```
P_shadow^dual := oint_{partial V} *F
```

close the inhomogeneous Maxwell law as

```
oint_{partial V} *F = int_V J = Q_enc        (Gauss / Ampere-Maxwell),
```

gauge-invariantly and locally; and does the Hodge decomposition confirm that the
only state-insufficient-yet-control-sufficient (regime-2) content is the
harmonic / topological sector already named in Phase 7?

## In Scope

Three registered cases (hand-derivation authoritative; tiny battery mirrors). For
each, state the domain condition **before** computing.

1. **Electrostatic Gauss closure.** Static charge `rho != 0`. The dual-shadow flux
   through a closed surface equals the enclosed charge:
   `oint.oint_{partial V} *F = Q_enc`. Expected landing: **A8** — recovers Gauss's
   law; the metric enters via `*`. Witness: a smeared Coulomb charge.
2. **Ampere-Maxwell closure with displacement current.** Current `J != 0` with
   time-varying `E`. The dual shadow recovers `curl B - partial_t E = J`. Expected
   landing: **A8** — recovers Ampere-Maxwell including the displacement-current
   term. Witness: a charging-capacitor / quasistatic current region.
3. **Hodge-split localization.** A domain with topology (solenoid exterior /
   2-torus patch) carrying both a source and a flux. Decompose the field into
   co-exact (sourced, determined by `Q`) and harmonic (topological). Expected
   landing: **B8** — the harmonic piece equals the Phase 7 AB flux; the sourced
   piece is determined; **no new regime-2**. Witness: solenoid-exterior plus an
   electric source, field = (determined sourced part) (+) (AB harmonic part).

**Determinacy verdict (registered).** By the EM uniqueness theorem, the sourced
field is fixed by (sources, boundary data, harmonic periods). The low-dimensional
dual-shadow readout (`Q_enc`) is state-insufficient but furnishes **no exact,
objective-free regime-2** of the Aharonov-Bohm type.

## Out Of Scope

- Re-deriving Maxwell, gauge fixing, or the full field equations from shadow
  primitives.
- Free radiation / retarded Green's function / radiation reaction (deferred to a
  later Phase 8b or beyond).
- Curved spacetime / geodesic-disc dual operator (Phase 2 already deferred the
  curved case).
- Quantum EM, lattice QED, plasma / matter response.
- Treating the dual-shadow determinacy as a control claim about any non-EM
  substrate.

## Locked Inputs (inherited, unchanged)

- Phase 1 sign convention (`F_{0i} = -E_i`, `F_{ij} = epsilon_{ijk} B_k`, `c = 1`).
- Phase 2 operator `P_shadow` (two-tier plaquette holonomy); Phase 3 homogeneous
  closure; Phase 7 boundary branches and the AB harmonic-sector result.
- The dual operator `P_shadow^dual` is the **metric Hodge-dual** of the Phase 2
  stencil. It is a **new, metric-dependent** operator and must be named as such —
  never conflated with the metric-free `P_shadow`.
- Tiny-battery support style (Phase 4 / Phase 7): explicit cases, seconds-scale,
  mirrors the hand calculation.

## Forbidden Moves

1. Do not claim Phase 8 re-derives Maxwell or "shadow-generates" the field
   equations.
2. Do not conflate the metric-dependent dual shadow `P_shadow^dual` with the
   metric-free `P_shadow`; the metric role must be explicit.
3. Do not present the sourced-sector determinacy as a **new** sharp regime-2; the
   only exact regime-2 is the Phase 7 harmonic / AB sector.
4. Do not add an unregistered residual or sector after seeing the algebra.
5. Do not use the numerical battery as the proof; the hand derivation is
   authoritative.
6. Do not silently extend to radiation or curved spacetime; those are named
   deferrals.

## Work Order

1. Fill the three-case table; state each domain condition before computing.
2. Define `P_shadow^dual = oint *F`; prove the dual-Stokes closure
   `oint_{partial V} *F = int_V J` by hand; **name where the metric enters**.
3. State and apply the determinacy verdict (uniqueness theorem) to the sourced
   sector.
4. Carry out the Hodge decomposition on the registered topological witness; show
   field = sourced (determined) (+) harmonic (= Phase 7 AB); confirm no new
   regime-2.
5. Add an optional tiny `faraday:phase8` battery that mirrors the hand calc:
   Gauss flux `= Q`; Ampere-Maxwell including displacement current; a toy Hodge
   split showing field = sourced (+) AB-harmonic.
6. File `docs/faraday/FARADAY_PHASE8_BOUNDARY.md`. Update `SHADOW_FARADAY.md`,
   `SUNDOG_V_FARADAY.md`, and `CROSS_SUBSTRATE_NOTES.md` §8.4 **only after** the
   receipt lands. Public copy (`faraday.html`) is held for explicit go-ahead.

## Outcome Branches

| Branch | Meaning | Public posture |
| --- | --- | --- |
| A8-dual-closure | `P_shadow^dual` recovers the inhomogeneous law (`oint *F = Q_enc`); metric named; sourced sector determined | Maxwell's sourced side closes via a metric-dependent dual shadow; no new regime-2. |
| B8-harmonic-survivor | The only state-insufficiency is the harmonic / topological sector = Phase 7 AB | Maxwell adds no new sharp separation; all of it is the already-named AB witness. |
| C8-bounded-failure | An unregistered residual appears, or the dual shadow fails to close in the clean sourced domain | Re-open the relevant prior phase; the public claim narrows. |

**Pre-registered landing:** A8 + B8 (no new regime-2). **Body-resistance
verdict:** sourced classical EM joins the marginal column for a *distinct* reason
— **determined-by-sources**, with all sharp content topological and already
counted (four-for-four, honestly disaggregated: not low-dimensional dynamics like
NSE-C1 / Mesa / shell, but exact determinacy plus the harmonic AB sector).

## Exit Criteria

Phase 8 exits only when:

- the three registered cases are classified under the A8/B8/C8 table;
- the dual-shadow closure is derived by hand and the metric role is named;
- the determinacy verdict and the Hodge localization (regime-2 = harmonic / AB
  only) are stated;
- any support battery is reproducible (`npm run faraday:phase8`) and linked from
  the receipt;
- the parent roadmap and ledger record the branch selected, and
  `CROSS_SUBSTRATE_NOTES.md` §8.4 is updated from "pre-registered prediction" to
  the earned verdict;
- no public `faraday.html` copy changes until explicit go-ahead.
