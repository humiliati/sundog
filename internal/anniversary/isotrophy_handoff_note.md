# Isotrophy Handoff Note

Status: current, 2026-05-19.
Audience: internal operators, anniversary promo team, future coding agents.
Source docs: `docs/sundog_v_isotrophy.md`, `docs/SUNDOG_V_THREEBODY.md`.

## One-Line Read

The isotrophy thread produced one real technical win and one clean negative:
G.2 reconciled the 21-choreography literature count with a hardened detector;
K1 showed the v0.2 daughter-count operator collapses to the equivariance-only
null and should not be promoted or extended into K2-K4.

## What Happened

G.2 asked whether the workbench could recover the catalog's 21 equal-mass 3D
choreographies from the Li-Liao supplementary-A data without hand-counting.
After fixing the inverse generator and the cyclic-group gate, the detector
found:

- 21 strict single-inertial-curve choreographies, matching the catalog's 21;
- 4 additional relative/rotating choreographies admitted only by the SO(3)
  gauge-min;
- a clean rotation-angle separator: the 4 relative cases require a nontrivial
  `2*pi/3` global rotation.

That is a durable result. It validates the detector and explains why a
gauge-invariant gate sees 25 while the literature says 21.

K1 then froze the v0.2 `K_facet` prediction on the 21 strict rows before any
supplementary-B piano-trio count. The strict result was:

```text
K_facet = 0
```

All 21 rows pass only the structural `F_beta` generator strictly. The broader
SO(3)-gauged diagnostic is 21 because `beta_I` appears only when the free
alignment absorbs the missing `Rpi`; that is a gauge-collapse shadow, not an
emergent daughter count.

## What It Means

The v0.2 operator was not a Sundog theorem test. It was the equivariance-only
null in operational form.

For a choreography seed, the genuine symmetry is cyclic `Z3`. The piano-trio
target symmetry is transposition-class `Z2`. Generically:

```text
Z3 cap Z2 = {e}
```

So a static equal-mass containment scan for already-present transposition
symmetry is forced to return `d_i = 0` after quotienting the structural
`F_beta`. K1 caught that before the multi-hour K2-K4 sweep. That is a success
of the process, not a result to route around.

## Public-Safe Language

Use:

- "The three-body isotrophy sidecar resolved a detector and literature-count
  question, but it did not produce theorem evidence."
- "The workbench cleanly separates strict single-curve choreographies from
  rotating relatives using the same rotation angle the gate already computes."
- "A proposed daughter-count test was retired after its cheap K1 precheck
  reduced to the equivariance-only null."
- "This is an example of the project refusing to promote a plausible-looking
  theorem path when the operational test becomes tautological."

Short copy:

> The isotrophy sidecar is a useful negative. It reconciled the 21
> equal-mass choreography count with a hardened detector, then stopped a
> proposed theorem test at K1 when the prediction collapsed to the
> equivariance-only null. That is not a proof of Sundog. It is a receipt for
> the project's discipline: run the cheap falsifier first, publish the boundary,
> and do not patch the operator after the fact.

## Do Not Say

- Do not say isotrophy proves the theorem.
- Do not say `K_facet = 0` means "no piano-trios exist."
- Do not say the 4 rotating choreographies are mistakes in the literature.
  They are outside the strict "single closed trajectory" convention.
- Do not restart K2-K4 under the v0.2 scope.
- Do not patch `d_i` to avoid zero. A v0.3 must be a fresh derivation with a
  fresh pre-registration.

## v0.3 Design Decision

Decision: open v0.3 in principle, but do not code or freeze it yet.

Primary route: compute `d_i` from the `m3=1` choreography alone using the
monodromy/variational spectrum, not by continuing toward the supplementary-B
catalog.

Corrected foundation to derive on paper:

Do not treat `alpha_I = ((12), T/2)` as an isotropy of the parent choreography.
That would repeat the v0.2 failure one level up. At `m3=1`, `(12)` is a
symmetry of the equal-mass equations, not necessarily of the individual orbit.
For rows where `(12)` maps `C_i` to itself up to phase/spatial gauge, define:

```text
G_i := rho((12)) o Phi_{T/2}
V_PT,i := ker(G_i - I)
```

and prove `[M_i, G_i] = 0` before block-diagonalizing the monodromy. For rows
where `(12)` maps `C_i` to a different equal-mass orbit, use an induced
representation over the `S3` group orbit instead.

The load-bearing deliverable is not a run. After the v0.3d typed response, it
is specifically the pair-orbit / dihedral-representation lemma: choose the
based/free/shifted-partner loop convention, correct the neutral quotient to
`span{X_H,u_E}` with `(M-I)u_E=cX_H`, prove the alpha-fixed graph descends
through `N_i`, derive the `<sigma3,F_beta>` real representation on
`K_i^{fib}`, then define the branch candidate sector and multiplicity gate. If
that derivation is fuzzy, v0.3 is not ready.

The first allowed pre-derivation run is only the case-split receipt. It tests
Condition 3, not strict `alpha_I` and not the K1 SO(3)-absorbed shadow. It must
carry both explicit spatial parity candidates:

```text
tau12_I := ((12), free phi, no time reversal, spatial I, SO(3) gauge-min)
tau12_Z := ((12), free phi, no time reversal, spatial Z, SO(3) gauge-min)
```

For each of the 21 strict rows, persist `R_i`, `phi_i`, `phi_i/(T/2)`,
spatial parity, closure-relative residual, and rotation angle. Closure-tight
rows are endomorphism candidates; `O(1)` rows are induced-representation
candidates. If the split is not bimodal, stop and write the marginal category
before any monodromy work.

Proper-parity receipt: the first `tau12_I`-only run completed on 2026-05-19 in
`120.61 s` and found **0 proper-endomorphism cases, 21 induced-representation
cases, 0 marginal reviews**. Receipt:
`results/isotrophy/k-facet-v03-tau12-case-split-21strict/`.

Parity-union receipt: `npm run isotrophy:tau12:cases` completed on 2026-05-20
in `177.72 s` and found **0 endomorphism cases, 21 induced-representation
cases, 0 marginal reviews** across `{tau12_I, tau12_Z}`. `tau12_Z` won the
residual for 6 rows, but the best non-tight residual remained `2.815e7` times
closure. So v0.3, if continued, is induced-representation-only across all 21
strict choreographies. The parity-union receipt is protected at
`results/isotrophy/k-facet-v03-tau12-parity-union-21strict/`.

F_beta pair-ID receipt: `npm run isotrophy:fbeta:pair-id` completed on
2026-05-20 in `99.69 s` and confirmed the structural chain the new teammate
flagged: the strict 21 are 21 singleton `(E, |L|)` groups; bare `(12)`
preserves those invariants, but has **0** inside-catalog matches excluding the
self row; the completed all-induced case split rules out the self row as a
true endomorphism. Therefore all 21 strict rows are catalog-asymmetric under
bare `(12)`. The same receipt confirms `F_beta` closure-tight for all 21 rows
(`F_beta_to_closure` range `0.283..0.804`) and records the structural cocycle
at the manifest level:

```text
F_beta = ((12), tau-active, Rpi)
tau component = schema-constant active
per-row tau flag = false
partner-orbit IVP = false
M_(12*C_i) = rho(Rpi) * M_i^-1 * rho(Rpi)^-1
```

This is a correction to the draft note2 language: no row-by-row missing-partner
integration is required, and tau is not a row variable. Per-row variability
belongs only to any additional gauge fit layered on top of the structural
F_beta cocycle.

Rejected as primary:

- Full numerical continuation in `m3`. It is too circular with
  supplementary-B as a primary prediction, though it may become a validation
  step after the spectral prediction is frozen.
- Bragg/Floquet coherence as the primary. It is promising and Sundog-shaped,
  but it should be a named cross-check unless the projector derivation shows
  the kernel count is the wrong spectral readout.

Remaining honest fork:

- If the projector can be derived cleanly, pre-register v0.3 with the negative
  stated first.
- If it cannot, accept isotrophy as a clean negative plus the G.2 detector win.
- Prior gate: the v0.3c typed transport review blocked monodromy code until
  neutral quotient, fiber typing, `G_i^2`, structural quotient/reduction, and
  candidate multiplicity were resolved.
- The first typed transport response probably kills the canonical single-fiber
  `G_i` and proposes a pair-orbit alpha-fixed kernel, but this is not locked.
  New blockers: `A_F y(-t)=y(t)` does not mean pointwise fixedness; alpha lands
  in a shifted partner fiber; `M^{-1/2}` should be typed half-flow; the graph
  parameterization must descend through `N_i`; `<sigma3,F_beta>` is likely a
  real dihedral representation, not commuting projector masks; and the
  multiplicity factor is pending again because `A_F`-even sectors need not be
  symplectic.
- Pair-orbit / dihedral draft 2 is the best candidate shape so far:
  `K_i^{fib}` decomposes into real `D3` irreps `a_i*T + b_i*S + c_i*E`, with
  `d_i_candidate=c_i`. Still not locked. Remaining blockers: choose the
  based/free/shifted loop convention, certify the `F_beta` and conjugate partner
  anchors, fix the typed half-flow reduction, prove the neutral quotient is
  `D3`-equivariant, and validate `c_i` through the crossing form.
- Neutral-block refinement: the quotient is `D3`-equivariant as
  `N_C = T*u_E + S*X_H`, not wholly trivial. Thus
  `K_i^{fib*} ~= a_i*T + b_i*S + c_i*E` becomes
  `K_i^{fib} ~= (a_i-1)*T + (b_i-1)*S + c_i*E`, preserving the standard-irrep
  count `c_i`.
- Crossing-form gate review: the branch-validity gate should split the mass
  perturbation as `Delta H=Delta H_T+Delta H_E`, with no sign-irrep component
  because `Delta H` is `F_beta`-even. Do not freeze the scalar
  `gamma_i^(k)` recipe yet. If `c_i>1`, the load-bearing object is likely a
  quotient Floquet crossing matrix on the `E` multiplicity space, with a
  closure-relative rank gate, not independent diagonal scalars until proved.
- Current next paper-only gate: write v0.3g as a crossing-form definition on
  the neutral-quotiented `E` multiplicity space before any monodromy code.

## Operator Guardrail

Before any future isotrophy run:

1. Confirm whether the work is G.2 detector maintenance, v0.3 derivation, or
   accidental K2-K4 restart.
2. If it is K2-K4 under v0.2, stop.
3. If it is v0.3, use the completed all-induced case split, then write the
   anchored `D3` and v0.3g crossing-form gates: based-loop convention at
   `p_i^F`, anchor certification, typed half-flow, neutral quotient
   `N_C=T*u_E+S*X_H` with no `E` leakage, `d_i_candidate=c_i`, quotient
   crossing form on the `E` multiplicity space, reduced-coordinate `Delta H`,
   no-`S` proof, matrix/rank versus scalar-gamma decision, negative, and
   go/no-go branch before running any monodromy or supplementary-B
   classification.
