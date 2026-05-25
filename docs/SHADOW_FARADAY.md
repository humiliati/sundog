# Shadow Faraday Phase 1 Ledger

Working hook:

> If the shadow projection is honest, Faraday's loop closes without borrowing
> the global potential. If it does not close, the leftover term gets a name.

Status: 2026-05-25, Phases 1-6 local readiness signed off. Phase 3 landed Branch A
(clean structural zero) in [`FARADAY_PHASE3_DERIVATIONS.md`](FARADAY_PHASE3_DERIVATIONS.md);
Phase 4 verification/falsification passed 5/5 in
[`FARADAY_PHASE4_VERIFICATION.md`](FARADAY_PHASE4_VERIFICATION.md). Phase 5
chapter close is signed, and Phase 6 local site-readiness artifacts are in
place. The only remaining external action is the post-deploy social validator
pass.

This ledger is the working receipt surface for
[`SUNDOG_V_FARADAY.md`](SUNDOG_V_FARADAY.md). The roadmap stays the narrative
spine; this file records the first algebraic lock: symbols, assumptions,
mapping table, and pre-registered branches before the shadow operator is
defined in Phase 2.

## Claim Boundary

The current claim is only:

> A pre-registered algebraic test has been opened for whether local,
> gauge-invariant shadow data is sufficient to recover Faraday induction in the
> classical vacuum case, or else produce a named residual. On the registered
> classical vacuum domain, Phase 3 lands Branch A and Phase 4 support checks
> pass.

This ledger does not derive Maxwell's equations. It does not address quantum
electrodynamics, plasma, curved-spacetime global topology, or source-bearing
domains except as named future extensions or quarantines.

## Phase 1 Exit Contract

Phase 1 exits only when these four items are stable enough for Phase 2 to define
`P_shadow` against them:

1. Symbol table and sign conventions are frozen.
2. Assumption ledger is explicit.
3. Sundog primitive -> EM object mapping is written down without adding new
   physics.
4. Phase 3 outcome branches are pre-registered before any zero-out derivation is
   interpreted.

Current disposition: **signed off for Phase 2 use**. Re-open Phase 1 only if
the sign convention, assumption ledger, mapping table, or Phase 3 outcome
branches need to change.

## Coordinate And Sign Convention Candidate

First-pass convention:

- Work on a local spacetime patch `U` in a flat or locally inertial chart.
- Natural units, `c = 1`.
- Metric signature `eta = diag(-1, +1, +1, +1)`.
- Coordinates `x^mu = (t, x, y, z)`.
- Levi-Civita orientation `epsilon^{0123} = +1`.
- Potential one-form components: `A_mu = (-Phi, A_x, A_y, A_z)`.
- Faraday tensor:

```text
F_{mu nu} = partial_mu A_nu - partial_nu A_mu
```

With the convention above:

```text
F_{0i} = -E_i
F_{ij} = epsilon_{ijk} B_k
E = -grad Phi - partial_t A
B = curl A
```

Dual tensor:

```text
tilde F^{mu nu} = (1/2) epsilon^{mu nu rho sigma} F_{rho sigma}
```

Lorentz invariants under this convention:

```text
I_1 = F_{mu nu} F^{mu nu} = 2(|B|^2 - |E|^2)
I_2 = F_{mu nu} tilde F^{mu nu} = -4 E dot B
```

The homogeneous Maxwell equation is the exterior-derivative identity:

```text
dF = 0
```

In 3-vector form, the Faraday component is:

```text
curl E + partial_t B = 0
```

and the integral form is:

```text
oint_{partial S} E dot dl = - d/dt int_S B dot dA
```

## Symbol Table

| Symbol | Meaning | Phase 1 status |
| --- | --- | --- |
| `U` | Local spacetime patch under audit | candidate |
| `S` | Oriented spatial surface inside the audited patch | candidate |
| `partial S` | Oriented loop bounding `S` | candidate |
| `A_mu` | Electromagnetic potential one-form | convention candidate |
| `lambda` | Gauge scalar for `A -> A + d lambda` | candidate |
| `F` | Faraday 2-form / field tensor | candidate |
| `tilde F` | Hodge dual of `F` under the chosen orientation | candidate |
| `E`, `B` | Electric and magnetic fields induced by `F` | candidate |
| `I_1`, `I_2` | Lorentz invariants listed above | candidate |
| `P_shadow` | Local shadow projection operator to be defined in Phase 2 | placeholder |
| `R_F(S)` | Faraday closure residual on a surface-loop pair | candidate |
| `R_I` | Invariant reconstruction residual from projected data | candidate |
| `Q` | Named quarantine term if closure is not clean | candidate |

Proposed residual names:

```text
R_F(S) = oint_{partial S} P_shadow(E) dot dl
       + d/dt int_S P_shadow(B) dot dA

R_I = (I_1, I_2)_from_shadow - (I_1, I_2)_from_F
```

Phase 2 must define `P_shadow` strongly enough that these residuals are
well-typed before Phase 3 tries to evaluate them.

## Assumption Ledger

First-pass assumptions:

1. Classical electromagnetism only.
2. Fields are smooth enough on `U` for Stokes' theorem and time/surface
   derivative exchange to be valid.
3. `U` is contractible unless a topological quarantine is explicitly being
   tested.
4. No magnetic monopoles inside `U`; equivalently `dF = 0` on the audited patch.
5. Electric sources/currents are absent from the first-pass patch. If a Phase 4
   example uses a solenoid or dipole, source regions must be outside the audited
   local patch or named as sourced-domain extensions.
6. The shadow operator may read local gauge-invariant data derived from `F` and
   finite local differences. It may not silently read a globally reconstructed
   potential.
7. Gauge transformations are active only on `A`; `F`, `E`, `B`, `I_1`, and `I_2`
   remain the gauge-invariant target objects.
8. Numerical or symbolic spot-checks are supporting receipts only. The Phase 3
   branch is algebraic.

Void conditions:

- The sign convention changes after Phase 3 begins without re-opening Phase 1.
- `P_shadow` is allowed to inspect a global potential, hidden boundary condition,
  or nonlocal reconstruction not registered in Phase 2.
- A residual category is invented after seeing the Phase 3 algebra.
- A sourced, singular, or topological case is used to advertise a source-free
  clean zero.

## Sundog Primitive Mapping

| Sundog primitive | EM-side object in this experiment | Boundary |
| --- | --- | --- |
| `sigma_3`-style detector | Gauge-invariant local 2-form contractions, loop/flux closure residuals, and orientation-sensitive sign checks | Analogy only; no `Z_3` symmetry is assumed for EM. |
| Local tidal proxy | Local samples of `F`, `dF`, and finite loop/surface residuals | Must stay local to `U`. |
| Gauge cocycle | Transition data for `A -> A + d lambda` across overlapping local patches | Used to audit invariance, not to reconstruct global `A`. |
| Procrustes alignment | Choice of local frame/convention before comparing projected observables | Frame lock must not alter gauge content. |
| Structural-zero receipt | `R_F(S) = 0` or `R_I = 0` by algebraic identity in the registered domain | Not a tolerance threshold. |
| Named quarantine | A residual with a pre-registered physical name: boundary, topology, source, singularity, or operator-stencil defect | Quarantine is excluded from clean-zero evidence. |
| Bounded failure | A surviving reconstruction/gauge/global term in the registered clean domain | The claim is narrowed or fails. |

## Phase 3 Branches Pre-Registered Before Derivation

The Phase 3 derivation must land in exactly one of these branches.

### A. Clean structural zero

For smooth, source-free, contractible `U`, the registered `P_shadow` returns
enough local gauge-invariant data that:

```text
R_F(S) == 0
```

for every admitted surface-loop pair `S`, and the Lorentz invariants computed
from shadow data match `I_1` and `I_2` with no dependence on `A`, `lambda`, or a
global reconstruction.

Allowed public phrasing after this branch:

> Local shadow data suffices for Faraday induction in the registered classical
> vacuum domain.

### B. Named quarantine

The closure residual is nonzero, but the surviving term is exactly attributable
to a pre-registered category:

- boundary term,
- topological obstruction,
- sourced-domain term,
- singularity/regularity failure,
- operator-stencil commutator.

Allowed public phrasing after this branch:

> The Faraday shadow test produced a named quarantine: `<term>`. The clean-zero
> claim is not earned outside the remaining registered domain.

### C. Bounded failure

A dependence on global reconstruction, gauge choice, or nonlocal potential data
survives in the registered clean domain.

Allowed public phrasing after this branch:

> The proposed shadow projection is insufficient for the registered Faraday
> zero-out. The residual bounds the claim.

## Phase 2 Handoff

Phase 2 must define `P_shadow` with the following minimum checks:

1. Locality: list the radius/stencil or differential order it may inspect.
2. Gauge invariance: prove or falsify `P_shadow(A + d lambda) = P_shadow(A)` on
   admitted observables.
3. Typing: state whether `P_shadow` acts on `A`, `F`, `(E, B)`, contractions of
   `F`, or local samples thereof.
4. Admissibility: define what makes a loop/surface pair valid for `R_F(S)`.
5. Quarantine hooks: name which residual classes are detectable before Phase 3.

## Phase 2: Local Shadow Projection Operator

Phase 2 opened 2026-05-25 against the Phase 1 conventions above. Phase 2 inherits
the sign convention, symbol table, assumption ledger, and Sundog-to-EM mapping
without modification; if any of those four locks needs to move, Phase 1 must be
re-opened first (Void condition 1 of the Assumption Ledger).

### Phase 2 Exit Contract

Phase 2 exits only when the five items the Phase 1 Handoff named are all stable
enough for the Phase 3 zero-out to be expressed in terms of `P_shadow` without
adding new physics:

1. Locality: stencil shape and radius are stated.
2. Gauge invariance: `P_shadow(A + d lambda) = P_shadow(A)` is proved or
   falsified on admitted observables.
3. Typing: the input class (`A`, `F`, `(E, B)`, contractions of `F`, local
   samples) is fixed.
4. Admissibility: what makes a `(S, partial S)` pair valid for `R_F(S)` is
   written down.
5. Quarantine hooks: the residual classes the operator can flag *before*
   Phase 3 are enumerated.

Current disposition: **signed off for Phase 3 setup**. The definition below is
the registered Phase 2 operator unless a dated amendment re-opens Phase 2.

### Operator Definition Candidate

`P_shadow` is the local plaquette-holonomy operator on the potential one-form
`A`. It is defined in two coupled tiers, one finite-stencil and one point-limit,
which agree on smooth `A`.

#### Stencil typing

Pick a point `x in U` and an ordered pair of coordinate directions `(mu, nu)`
with `mu != nu`. For an edge length `epsilon > 0` small enough that the closed
square

```text
omega_{mu nu}(x, epsilon)
  = { x + s e_mu + t e_nu : 0 <= s, t <= epsilon }
```

lies inside `U`, the directed boundary `partial omega_{mu nu}(x, epsilon)` is
the loop traversed in the order

```text
x -> x + epsilon e_mu -> x + epsilon e_mu + epsilon e_nu
    -> x + epsilon e_nu -> x
```

This four-edge loop is the canonical **plaquette stencil** at `x`. The stencil
inspects `A` at exactly four points and on the four edges connecting them.

#### Finite-stencil tier

```text
P_shadow^stencil [A]_{mu nu}(x, epsilon)
  := oint_{partial omega_{mu nu}(x, epsilon)} A
```

By the assumption ledger (smooth `A` on `U`, `dF = 0`), Stokes' theorem gives

```text
P_shadow^stencil [A]_{mu nu}(x, epsilon)
  = int_{omega_{mu nu}(x, epsilon)} F
  = epsilon^2 * F_{mu nu}(x) + O(epsilon^3)
```

This tier is the operator that Phase 3 will plug into the integral-form
identity. It is well-typed even when `A` is only `C^1`; the leading correction
is `O(epsilon^3)` for `C^2` fields.

#### Point-limit tier

```text
P_shadow^point [A]_{mu nu}(x)
  := lim_{epsilon -> 0+} (1 / epsilon^2)
       * P_shadow^stencil [A]_{mu nu}(x, epsilon)
  =  F_{mu nu}(x)
```

This tier returns the Faraday tensor pointwise. It exists exactly when `A` is
differentiable at `x`. It is *not* what the Phase 3 closure receipt uses
(`R_F(S)` is an integral residual); it is the receipt that ties
`P_shadow^stencil` back to `F` in the smooth limit.

Pyramid summary (replaces the `P_shadow` placeholder in the Phase 1 symbol
table):

```text
P_shadow:    A     ->  oint_{partial omega} A    (gauge-invariant scalar per plaquette)
             A     ->  F_{mu nu}(x)              (smooth limit, on differentiable A)
             F     ->  identity on F             (pre-composed: trivially gauge-invariant)
             E, B  ->  inherited via the F_{0i}, F_{ij} relations of Phase 1
```

Phase 2 commits the input class as **`A` first**, with the `F` and `(E, B)`
actions inherited through the Phase 1 sign convention. The operator never
inspects a globally consistent choice of `A`; it only reads `A` along the
boundary of a single plaquette.

### Locality Receipt

The locality data the operator depends on is exhausted by:

| Quantity | Locality content | Phase 2 status |
| --- | --- | --- |
| Stencil shape | Closed coordinate plaquette `omega_{mu nu}(x, epsilon)` | candidate |
| Stencil radius | `epsilon` (free parameter, `> 0`) | candidate |
| Number of inspected `A` values | Four vertices + four edges of one plaquette | candidate |
| Differential order | At most one derivative of `A` per edge (zero for the holonomy form) | candidate |
| Global data inspected | None. No transition functions, no boundary of `U`, no field at infinity. | candidate |

The operator is therefore **strictly local** in the stencil sense: its output
at `(x, mu, nu, epsilon)` depends only on `A` restricted to the closed
plaquette. This is the load-bearing locality claim; Phase 3 will test whether
it is enough to close Faraday's law.

### Gauge Invariance Audit

#### Active gauge transformation

The Phase 1 ledger registers gauge transformations as active on `A`, with
`F`, `E`, `B`, `I_1`, `I_2` as the target invariants. For a scalar
`lambda` smooth on `U`:

```text
A   ->  A' = A + d lambda
A_mu  ->  A_mu + partial_mu lambda
```

#### Identity to audit

The Phase 1 Handoff predicate, restated for the candidate operator:

```text
P_shadow^stencil [A + d lambda]_{mu nu}(x, epsilon)
  ==
P_shadow^stencil [A]_{mu nu}(x, epsilon)
```

#### Proof sketch on contractible `U`

```text
P_shadow^stencil [A + d lambda]_{mu nu}(x, epsilon)
  = oint_{partial omega} (A + d lambda)
  = oint_{partial omega} A + oint_{partial omega} d lambda
```

The boundary `partial omega` is a closed loop. By the Poincare lemma applied to
the exact one-form `d lambda` (or directly by the fundamental theorem of
calculus along the closed loop):

```text
oint_{partial omega} d lambda = 0
```

Therefore:

```text
P_shadow^stencil [A + d lambda]_{mu nu}(x, epsilon)
  = oint_{partial omega} A
  = P_shadow^stencil [A]_{mu nu}(x, epsilon)
```

The same identity transfers to the point-limit tier by linearity of
`(1 / epsilon^2) lim_{epsilon -> 0+}`. The audit passes on the registered
domain (smooth `A`, contractible `U`, no monopoles).

#### Audit failure modes

The proof is **conditional** on three Phase 1 assumptions:

| Failure mode | Which Phase 1 assumption it breaks | Goes to |
| --- | --- | --- |
| `lambda` not differentiable on `omega` | Assumption 2 (smoothness for Stokes) | Quarantine hook 1 |
| `partial omega` non-contractible in `U` | Assumption 3 (contractible patch) | Quarantine hook 2 |
| Monopole world-line through `omega` | Assumption 4 (no monopoles) | Quarantine hook 3 |

The gauge-invariance audit therefore *passes* on the registered domain and
*fails by name* outside it. This is the receipt format Phase 1 demanded.

### Admissibility Rule For `(S, partial S)`

A surface-loop pair `(S, partial S)` is admissible for evaluating `R_F(S)`
under `P_shadow^stencil` when:

1. `S` is an oriented `C^2` two-surface inside `U` with `partial S = partial S`
   (the boundary is honest, not a self-intersection).
2. `S` is contractible in `U` (rules out non-trivial cycles up front).
3. `S` is static in the working frame (no Reynolds-transport motional terms;
   see Quarantine hook 5).
4. The plaquette mesh `{omega_{mu nu}(x_i, epsilon)}` used to evaluate
   `int_S P_shadow^stencil(B) dA` covers `S` without overlap or gap, and the
   limit `epsilon -> 0+` exists.
5. The mesh respects the orientation of `S` so that the induced orientation on
   `partial S` matches the orientation used in `oint_{partial S} ... dl`.

A pair that satisfies 1-5 is called a **registered surface-loop pair**. The
clean-zero claim in Branch A applies only to registered pairs; Branch B's
named-quarantine claims apply to pairs that fail one of 1-5 by a named class.

### Quarantine Hooks Detectable Before Phase 3

Five residual classes are nameable now, before any zero-out algebra runs.
Each is paired with the Phase 1 outcome branch it forces.

| # | Quarantine | Trigger | Forces |
| --- | --- | --- | --- |
| 1 | **Regularity** | `A` or `lambda` not `C^1` on `omega` (delta-string, distributional sources) | Branch B if domain is otherwise registered; Branch C if it appears in the smooth admitted domain |
| 2 | **Topology** | `partial omega` or `partial S` non-contractible in `U` (Aharonov-Bohm-type winding) | Branch B (boundary / topological obstruction) |
| 3 | **Monopole** | World-line of a magnetic monopole crosses `omega` or `S` (`dF != 0`) | Branch B (sourced-domain term); excluded from first-pass clean domain |
| 4 | **Operator-stencil commutator** | `epsilon -> 0+` limit and `partial_t` fail to commute on the registered class (e.g., stencil deforms with time) | Branch B if named (e.g., moving boundary); Branch C if unnamed |
| 5 | **Motional EMF** | `S` moves in the working frame, introducing Reynolds-transport terms beyond `partial_t int_S B dA` | Branch B if registered as a motional-term extension; excluded from first-pass admissibility |

Hooks 1, 2, 3 are excluded by the Phase 1 Assumption Ledger inside the
first-pass patch. Hooks 4 and 5 are excluded by the Admissibility Rule above.
Anything that escapes both filters in Phase 3 is, by construction, a Phase 3
quarantine and not a Phase 2 oversight.

### Three-Stage Audit Mapping

Phase 2 is structured as a three-stage receipt chain, mirroring the K_facet
shape so that future readers can compare audit discipline across experiments.
The names and content differ; the chain structure is the precedent.

| Stage | K_facet name (precedent) | Faraday Phase 2 name | What it checks |
| --- | --- | --- | --- |
| 1 | Sentinel / Gamma runner | **Stencil sentinel** | Plaquette stencil is well-typed; the four edges close into one loop; the Stokes identity `oint A = int F` holds at `O(epsilon^3)` on smooth `A`. |
| 2 | Adaptive-floor reprocessor | **Locality floor** | The smallest pre-registered `epsilon` for which the gauge-invariance identity and the Stokes identity both hold within a *named* tolerance (or by exact algebraic identity on smooth `A`, in which case the floor collapses to `epsilon -> 0+`). |
| 3 | Bridge audit | **Topology / regularity audit** | For every admitted `(S, partial S)` pair, no plaquette in the mesh traverses a non-contractible cycle and no `omega` straddles a regularity failure of `A` or `lambda`. |

Phase 2 closes when all three stages produce signed receipts for the registered
domain (smooth `A`, contractible `U`, source-free, static `S`). The stages do
not require numerical work in Phase 2; they require explicit *statements* of
what each stage will check in Phase 3 algebra.

### Phase 2 -> Phase 3 Handoff

Phase 3 inherits the following objects in their Phase 2-locked form, and must
not redefine them once the zero-out derivation begins:

1. `P_shadow^stencil`, `P_shadow^point`, and the relation between them.
2. The Locality Receipt table.
3. The Gauge Invariance audit identity and its proof.
4. The Admissibility Rule for `(S, partial S)`.
5. The five named Quarantine hooks.
6. The Three-Stage Audit Mapping (used to *file* Phase 3 receipts, not to
   re-derive Phase 2).

If Phase 3 needs to add a new admissibility clause or quarantine class, Phase 2
must be re-opened with a dated amendment block (this is the Sundog precedent
for Echo's appended-addendum handoff pattern).

### Phase 2 Sign-Off Decisions

These decisions answer the four Phase 2 open questions and are binding for
Phase 3 unless Phase 2 is re-opened with a dated amendment.

1. **Stencil shape: use coordinate plaquettes.** Phase 3 stays in the local
   flat or locally inertial chart registered in Phase 1. Coordinate plaquettes
   make Stokes' theorem and edge cancellation explicit. A geodesic-disc version
   is deferred to a later curved-spacetime extension or robustness appendix; it
   is not part of the first-pass operator.
2. **Finite vs. point limit: keep two named readouts.** The Phase 3 gate is the
   continuum readout

   ```text
   R_F^0(S) := lim_{epsilon -> 0+} R_F^epsilon(S)
   ```

   using `P_shadow^point = F`. The finite-stencil readout `R_F^epsilon(S)` is
   retained as a locality/discretisation receipt and must take the
   `epsilon -> 0+` limit at the end. A fixed-`epsilon` truncation term is not a
   clean-domain failure unless it survives the registered limit.
3. **Floquet / twist enrichment: defer.** The Phase 3 operator is the bare
   plaquette holonomy only. Averaging over rotated plaquettes, Floquet-style
   robustness, or twist-operator enrichment may be used later as a Phase 4
   falsifier or robustness variant, but it cannot rescue or alter the Phase 3
   result.
4. **Two-tier operator: carry both, with roles locked.** `P_shadow^point` is the
   mathematical object used for the algebraic zero-out. `P_shadow^stencil` is
   the locality receipt showing how the point object is obtained without a
   global reconstruction of `A`. Public claims must not blur the two.

Net sign-off: Phase 3 may begin with `P_shadow` as the two-tier coordinate
plaquette-holonomy operator.

## Phase 3: Takeoff Gate

Phase 3 is cleared to begin from the locked Phase 1/2 ledger. This section is
the preflight checklist: it says what the derivation may use, what it may not
use, what must be shown, and how the result lands.

### Locked Inputs

Phase 3 may use only these registered inputs:

1. `A_mu = (-Phi, A_x, A_y, A_z)` and
   `F_{mu nu} = partial_mu A_nu - partial_nu A_mu`.
2. The sign convention `F_{0i} = -E_i`, `F_{ij} = epsilon_{ijk} B_k`.
3. The smooth, source-free, contractible, static-surface domain from the
   Assumption Ledger and Admissibility Rule.
4. The two-tier coordinate plaquette operator:

   ```text
   P_shadow^stencil[A]_{mu nu}(x, epsilon) = oint_{partial omega_{mu nu}} A
   P_shadow^point[A]_{mu nu}(x) = F_{mu nu}(x)
   ```

5. Exterior calculus identities for smooth forms: linearity, Stokes' theorem,
   and `d(dA) = 0`.
6. The Lorentz-invariant definitions:

   ```text
   I_1 = F_{mu nu} F^{mu nu}
   I_2 = F_{mu nu} tilde F^{mu nu}
   ```

### Forbidden Shortcuts

Phase 3 must not:

1. Invoke Faraday's law (`curl E + partial_t B = 0`) as a premise.
2. Invoke the full Maxwell system as a premise.
3. Change the sign convention, admissibility rule, or branch taxonomy.
4. Add a new quarantine class after seeing the algebra.
5. Use a global reconstruction of `A` beyond the single-plaquette holonomy and
   registered point-limit relation.
6. Use a sourced, singular, topological, or moving-surface case as clean-domain
   evidence.

Any need for one of these moves re-opens Phase 1 or Phase 2 before Phase 3 may
continue.

### Derivation Work Order

The Phase 3 receipt should proceed in this order:

1. **Point-limit reduction:** replace `P_shadow^point[A]` with `F` and then
   with `(E, B)` using the locked sign convention. This is the only permitted
   bridge from shadow data to fields.
2. **Homogeneous identity:** compute `dF = d(dA)` and reduce it to zero by
   antisymmetry of mixed partials / nilpotence of `d`.
3. **Faraday component extraction:** extract the spatial three-index component
   of `dF = 0` and show it is exactly

   ```text
   curl E + partial_t B = 0
   ```

   under the locked sign convention.
4. **Integral closure:** apply Stokes to every registered surface-loop pair:

   ```text
   R_F^0(S) =
     oint_{partial S} P_shadow^point(E) dot dl
     + d/dt int_S P_shadow^point(B) dot dA
   ```

   and show `R_F^0(S) = 0`.
5. **Finite-stencil receipt:** record the finite-stencil readout
   `R_F^epsilon(S)` and its truncation class. The clean-domain gate is passed
   only by the registered limit `lim_{epsilon -> 0+} R_F^epsilon(S) = 0`.
6. **Invariant receipt:** show that `I_1` and `I_2` computed from
   `P_shadow^point[A] = F` match the Phase 1 invariant definitions and are
   independent of `A -> A + d lambda`.
7. **Landing classification:** choose exactly one of Branch A/B/C below and
   write the reason in the closure residual table.

### Exact Success Predicate

Phase 3 earns Branch A only if all of these are true on the registered clean
domain:

```text
For every registered surface-loop pair S:
  R_F^0(S) == 0

For every admitted point x:
  (I_1, I_2)_from_shadow(x) == (I_1, I_2)_from_F(x)

For every smooth lambda:
  both predicates are unchanged by A -> A + d lambda
```

If the finite-stencil residual is nonzero only at fixed `epsilon` and vanishes
in the registered point limit, that is a discretisation receipt, not a failure.
If any residual survives the registered limit in the clean domain, Branch A is
not earned.

### Closure Residual Table Template

Phase 3 must fill this table before any public summary changes.

| Receipt | Registered expression | Expected clean-domain value | Observed algebraic value | Branch impact |
| --- | --- | --- | --- | --- |
| Point-limit Faraday residual | `R_F^0(S)` | `0` | pending | pending |
| Finite-stencil residual | `R_F^epsilon(S)` | `O(epsilon)` or named truncation before limit; `0` after limit | pending | pending |
| Gauge invariance | `delta_lambda R_F^0(S)` | `0` | pending | pending |
| Lorentz invariant 1 | `I_1_from_shadow - I_1_from_F` | `0` | pending | pending |
| Lorentz invariant 2 | `I_2_from_shadow - I_2_from_F` | `0` | pending | pending |

### Landing Branches

Phase 3 must land in exactly one branch:

- **Branch A - clean structural zero:** all success predicates above are exact
  algebraic zeros on the registered clean domain.
- **Branch B - named quarantine:** a nonzero term appears only in a
  pre-registered non-clean case or one of the five named quarantine hooks.
- **Branch C - bounded failure:** a global-reconstruction, gauge-choice,
  nonlocal, or unregistered residual survives inside the clean domain.

### Phase 3 Receipt File

The derivation may be appended below this section, but the preferred receipt is
a dedicated file:

```text
docs/FARADAY_PHASE3_DERIVATIONS.md
```

That file should include the hand/exterior-calculus derivation first. Any SymPy
or tiny Python sign check belongs in a clearly marked supporting appendix and
does not decide the branch.

Takeoff disposition: **clear**. Phase 3 may begin.

## Phase 3 Result

Phase 3 receipt:
[`FARADAY_PHASE3_DERIVATIONS.md`](FARADAY_PHASE3_DERIVATIONS.md).

Disposition: **Branch A - clean structural zero** on the registered classical
vacuum domain.

Result summary:

- `R_F^0(S) = 0` for every registered static surface-loop pair.
- `R_F^epsilon(S) = O(epsilon)` before the registered point limit and `0` after
  the limit.
- `delta_lambda R_F^0(S) = 0`.
- `I_1_from_shadow - I_1_from_F = 0`.
- `I_2_from_shadow - I_2_from_F = 0`.

Claim boundary:

This closed Phase 3 only at the time it landed. Phase 4, Phase 5, and local
Bucket 1 site-readiness are now recorded below as later receipts.

## Phase 4 Result

Phase 4 receipt:
[`FARADAY_PHASE4_VERIFICATION.md`](FARADAY_PHASE4_VERIFICATION.md).

Support command:

```powershell
npm run faraday:phase4
```

Support artifacts:

- `scripts/faraday-phase4-battery.mjs`
- `results/faraday/phase4-battery/manifest.json`
- `results/faraday/phase4-battery/cases.csv`
- `results/faraday/phase4-battery/finite-stencil.csv`

Disposition: **pass - 5/5 predicates satisfied**.

Phase 4 confirms the required clean-domain cases and falsifiers:

- Uniform constant `B`: `maxFaradayResidual=0`, `I1=12.5`, `I2=0`.
- Source-free plane wave: `maxFaradayResidual=0`, `I1=0`, `I2=0`.
- Nonlocal projection falsifier: residual `0.787734891504`, as expected.
- Artificial monopole quarantine: `dF_xyz=3`, as expected.
- Gauge-after-projection check: finite plaquette holonomy delta `0`.

Phase 4 does not create a new branch. It supports the Phase 3 Branch A receipt
on the registered clean domain and demonstrates that the pre-registered
quarantine/falsifier hooks actually trip outside that domain.

### Roadmap Open Questions Resolved

The original roadmap questions are resolved as follows.

1. **Page name:** keep `faraday.html` as the canonical public page. A broader
   `shadow.html` umbrella can come later only after there is more than one
   shadow-substrate result to host.
2. **Phase 4 must-include examples:** use three required cases:
   uniform constant `B` as the trivial control; a smooth source-free plane wave
   as the nontrivial clean-domain pass candidate; and an artificial monopole or
   source insertion as the named-failure falsifier. The time-varying solenoid
   becomes an optional quarantine/topology case, not a required clean pass.
3. **Sourced cases:** Phase 3 remains strictly source-free and contractible.
   Phase 4 may include a minimal sourced or topological case only to demonstrate
   the quarantine machinery.
4. **Hand vs. SymPy receipts:** the hand/exterior-calculus derivation is
   authoritative. SymPy or a tiny Python spot-check may verify signs and example
   cases, but it cannot replace the algebraic proof or introduce new branches.

## Phase 5: Chapter Close

Phase 5 opened 2026-05-25 after Phases 3 and 4 landed. This section is the
chapter-close note for the Shadow Faraday experiment: it does not introduce
new physics, recompute the derivation, or change the branch. It rolls up the
receipts, fixes the outcome statement, names the limitations, lists the
extensions Phase 3 explicitly did not address, and runs a fidelity audit
over the Phase 1 -> Phase 4 chain.

### Outcome

The Shadow Faraday Zero-Out experiment lands in **Branch A - clean
structural zero** on the registered classical-vacuum domain.

The allowed public phrasing from the Phase 1 outcome ledger is now earned:

> Local shadow data suffices for Faraday induction in the registered
> classical vacuum domain.

Three things make that sentence honest rather than oversold. First, the
registered domain is narrow: smooth `A`, contractible `U`, source-free,
static surface-loop pair. Second, the operator is local in the stencil
sense and gauge-invariant by construction, not by appeal to a global
reconstruction. Third, every alternative outcome was named in advance:
five quarantine hooks (regularity, topology, monopole, operator-stencil
commutator, motional EMF), each paired with the branch it would force.

### Receipts Catalog

| Phase | Receipt | Disposition |
| --- | --- | --- |
| 1 | This document, sections "Coordinate And Sign Convention Candidate" through "Phase 3 Branches Pre-Registered Before Derivation" | Signed; mostly-plus metric, `c = 1`, `F_{0i} = -E_i`, `F_{ij} = epsilon_{ijk} B_k`. |
| 2 | This document, "Phase 2: Local Shadow Projection Operator" section through "Phase 2 Sign-Off Decisions" | Signed; `P_shadow` = two-tier coordinate plaquette holonomy, with locality receipt, gauge-invariance audit, admissibility rule, and five named quarantine hooks. |
| 3 | [`FARADAY_PHASE3_DERIVATIONS.md`](FARADAY_PHASE3_DERIVATIONS.md) | Branch A on the registered clean domain. All five closure-residual table entries are exact algebraic zeros. |
| 4 | [`FARADAY_PHASE4_VERIFICATION.md`](FARADAY_PHASE4_VERIFICATION.md) + `scripts/faraday-phase4-battery.mjs` + `results/faraday/phase4-battery/{manifest.json,cases.csv,finite-stencil.csv}` | 5/5 predicates passed via `npm run faraday:phase4`. Falsifier hooks trip when expected; clean-domain checks return structural zeros. |

The Branch A claim is carried by the Phase 3 hand exterior-calculus
derivation. Phase 4 is supporting evidence, not the proof.

### Closure Residual Summary

The five Phase 3 residual entries, plus the locality boundary and quarantine
machinery, with the supporting Phase 4 evidence:

- **Point-limit Faraday residual `R_F^0(S)`.** Phase 3 algebra: `0`. Phase 4
  support: constant `B` and source-free plane wave both pass at the
  registered sample point. Final disposition: closed at `0`.
- **Finite-stencil residual `R_F^epsilon(S)`.** Phase 3 algebra: `O(epsilon)`
  before the registered point limit and `0` after. Phase 4 support:
  plane-wave normalized-holonomy error decreases with `epsilon` and the
  `error / epsilon` column converges to the corner-anchored Taylor
  coefficient. Final disposition: closed at `0` in the registered limit.
- **Gauge invariance `delta_lambda R_F^0(S)`.** Phase 3 algebra: `0` by
  `oint d lambda = 0`. Phase 4 support: finite-plaquette gauge delta `0`
  for `lambda = 0.31 t x - 0.17 y z + 0.07 x z`. Final disposition: closed
  at `0`.
- **Lorentz invariant 1 reconstruction.** Phase 3 algebra: `0` exactly.
  Phase 4 support: constant `B` reproduces `I1 = 12.5`; plane wave
  reproduces `I1 = 0`. Final disposition: closed at `0`.
- **Lorentz invariant 2 reconstruction.** Phase 3 algebra: `0` exactly.
  Phase 4 support: constant `B` and plane wave both reproduce `I2 = 0`.
  Final disposition: closed at `0`.
- **Locality boundary.** Phase 3 algebra: nonlocal probe forbidden by
  construction. Phase 4 support: nonlocal projection deliberately violates
  the rule and returns `0.787734891504`. Locality is load-bearing, not
  decorative.
- **Quarantine machinery.** Phase 3 algebra: five named hooks registered
  before Phase 3. Phase 4 support: artificial monopole insertion trips
  `dF_xyz = 3`. Branch B machinery demonstrated to fire outside the clean
  domain.

### Limitations And Scope

The Branch A claim closes on the registered domain. It does not extend to:

1. **Sourced electromagnetism.** Phase 3 assumed source-free. A clean-domain
   claim under nonzero charge/current density is not earned by this work.
   The integral-form residual `R_F^0(S)` is still well-defined with sources,
   but the algebraic structural zero would need a separate experiment.
2. **Topology beyond contractible patches.** Non-contractible loops
   (Aharonov-Bohm setups, magnetic monopoles, solenoids that thread the
   surface) trip the registered topology quarantine. Branch A does not
   cover them.
3. **Distributional or singular fields.** `A` was assumed smooth enough for
   Stokes theorem and the `epsilon -> 0+` limit. Delta-string sources,
   shock fronts, and distributional fields are quarantined out.
4. **Curved spacetime.** Phase 3 used a flat or locally-inertial chart. A
   geodesic-disc variant of the operator is on the deferred list and would
   need its own takeoff gate.
5. **Quantum electrodynamics.** Classical fields only. The plaquette
   holonomy is suggestive of lattice gauge theory but Phase 3 did not
   pursue the Wilson-loop / lattice QED reading.
6. **Plasma and matter coupling.** Free vacuum case only. Polarisation and
   magnetisation are not modelled.
7. **Moving or deforming surfaces.** Phase 2 admissibility rule fixed `S`
   static in the working frame. Reynolds-transport (motional EMF) terms
   are outside the clean-domain claim.

Phase 4 demonstrated machinery for (1) and (2) via the monopole insertion
and the nonlocal probe. Phase 4 did not exhibit a worked example for
(3)-(7); those would be future quarantine receipts, not Branch A failures.

### Suggested Next Minimal Extensions

The roadmap explicitly invited a "suggested next minimal extensions" list.
Each item below is a self-contained next experiment with the same
receipt-driven posture as this one. None is required for the present
chapter close.

1. **Sourced Branch B receipt.** A minimal sourced patch with smooth
   charge density and `J_mu` registered in the assumption ledger. Expected
   landing: a named nonzero `R_F^0(S)` term equal to the displacement
   current / Ampere-Maxwell complement, not a Branch C failure. Completes
   the Maxwell story along the same shadow line.
2. **Aharonov-Bohm topology receipt.** A solenoid threading a
   non-contractible loop, evaluated against the registered topology
   quarantine. Expected landing: Branch B with the AB phase as the named
   survivor.
3. **Curved-spacetime takeoff.** A geodesic-disc variant of
   `P_shadow^stencil` on a locally-inertial chart with nonzero Riemann
   tensor. Phase 2 already deferred this; the takeoff gate would need a
   new admissibility rule.
4. **Lattice / Wilson-loop bridge.** Re-cast `P_shadow^stencil` explicitly
   as a U(1) Wilson plaquette and translate Phase 3 into the lattice gauge
   theory dictionary. Translation, not a new experiment; may produce a
   one-page primer rather than a full ledger.
5. **Floquet / twist robustness.** The Phase 2 sign-off deferred this; it
   would now run as a Phase 4-style robustness variant against the
   existing Branch A receipt.
6. **Shadow-substrate catalog.** If a second shadow-substrate experiment
   lands (shadow Gauss law, shadow Stokes theorem in 3D fluids), start a
   catalog under `docs/shadow/` with one row per substrate. Until there
   is a second row, there is no catalog to build; the roadmap
   gauge-cocycle catalog hint stays deferred.

### Fidelity Audit

A fidelity pass over the Phase 1 -> Phase 4 chain, recording what holds
and what is soft.

**What holds.**

- The Phase 1 sign convention is consistent across the ledger, the
  Phase 3 receipt, the Phase 4 support script, and `faraday.html`. No
  sign drift.
- Phase 3 invokes only the six Locked Inputs from the takeoff gate.
  None of the six Forbidden Shortcuts appears (Faraday law is derived,
  not assumed; no global `A`; no new quarantine class).
- The Phase 4 nonlocal-falsifier residual hand-verifies to twelve digits
  against an independent computation:
  `k A (sin(k(z-t)) - sin(k(z+delta-t))) = -0.787734891504`
  at the registered sample.
- The Phase 4 finite-stencil normalized error converges with `epsilon`
  to the corner-anchored Taylor-expansion leading coefficient
  `(partial_x F_xz + partial_z F_xz) / 2 = k A sin(k(z-t)) / 2`. At the
  registered sample this magnitude is `0.307272`. The Phase 4 table
  `error / epsilon` column converges toward that value from below,
  exactly as theory predicts.
- The receipt files cross-link consistently. `SHADOW_FARADAY.md`,
  `FARADAY_PHASE3_DERIVATIONS.md`, `FARADAY_PHASE4_VERIFICATION.md`,
  and `faraday.html` all carry the same outcome statement.

**Soft spots, recorded transparently.**

- *Constant-`B` control asserts rather than tests.* In the Phase 4
  support script, the constant-`B` case has `maxFaradayResidual = 0`
  hardcoded rather than computed from `curl(0) + partial_t (const)`.
  The answer is correct (the case is algebraically trivial), but the
  case is a sign / invariant control, not an algorithmic check.
- *"maxFaradayResidual" is single-point on the plane wave.* The
  registered domain claim is everywhere; the support script evaluates
  one sample point and labels it `max`. The single-point check is
  sufficient by smoothness on a free plane wave, but the label is
  slightly loose. Reads cleaner as `samplePointFaradayResidual` in any
  future battery.
- *Finite-stencil `error / epsilon` converges to about `0.307`, not to
  zero.* This is the leading O(epsilon) coefficient by design and
  matches theory, but a reader skimming the table may expect the ratio
  to go to zero. A one-line annotation ("ratios converge to the leading
  O(epsilon) coefficient, not to zero") would pre-empt the misread.
- *Corner-anchored vs centered plaquette.* Phase 2 sign-off picked
  coordinate plaquettes; the implementation uses corner-anchored
  squares. A centered plaquette would change the leading scaling from
  O(epsilon) to O(epsilon^2). Worth a one-line callout in the Phase 4
  finite-stencil receipt for future readers.

None of these soft spots changes the Branch A landing. They are filed
here so a future audit pass starts from full transparency.

### Chapter-Close Disposition

Phase 5 exits **closed**.

- Phase 1, Phase 2, Phase 3, and Phase 4 receipts are signed and
  cross-linked.
- The Branch A outcome is recorded with explicit scope, named
  limitations, and a registered extension menu.
- The fidelity audit is on the record with no Branch-A-blocking issues.
- The roadmap gauge-cocycle catalog hint stays deferred until a second
  shadow-substrate experiment lands.

Phase 6 local page readiness is now in place. Designed `1200x630` `og:image`,
JSON-LD `TechArticle`, tuned title and description, an inbound link from
`index.html`, `site-pages.json` promotion, and a sitemap entry have all landed.
The chapter is closed and the local site artifact is promoted. The remaining
external action is a post-deploy LinkedIn/Twitter validator pass.

## Inspection Trail

- 2026-05-25 - `faraday.html` filed as a noindex draft staging page.
- 2026-05-25 - `site-pages.json` and
  [`SEO_AND_SOCIAL_READINESS_ROADMAP.md`](SEO_AND_SOCIAL_READINESS_ROADMAP.md)
  keep `/faraday` Class D until Phase 3 result and Bucket 1 readiness
  exist.
- 2026-05-25 - Phase 1 ledger opened with symbol, assumption, mapping,
  and outcome-branch candidates.
- 2026-05-25 - Phase 2 candidate definition of `P_shadow` filed as the
  plaquette-holonomy two-tier operator, with locality receipt,
  gauge-invariance audit, admissibility rule, and five named quarantine
  hooks.
- 2026-05-25 - Phase 2 sign-off decisions recorded: coordinate
  plaquettes, point-limit gate plus finite-stencil locality receipt,
  bare plaquette only in Phase 3, and two-tier operator retained with
  roles locked.
- 2026-05-25 - Phase 3 takeoff gate recorded: locked inputs, forbidden
  shortcuts, derivation work order, exact success predicate, residual
  table template, landing branches, and receipt-file target.
- 2026-05-25 - Phase 3 receipt landed in
  [`FARADAY_PHASE3_DERIVATIONS.md`](FARADAY_PHASE3_DERIVATIONS.md),
  with proof-hygiene corrections to the form-degree Stokes statement
  and finite-stencil scaling. Branch A selected for the registered
  clean domain.
- 2026-05-25 - Phase 4 verification/falsification battery landed in
  [`FARADAY_PHASE4_VERIFICATION.md`](FARADAY_PHASE4_VERIFICATION.md).
  The support command `npm run faraday:phase4` passed 5/5 predicates.
- 2026-05-25 - Phase 5 chapter-close section recorded above. Outcome:
  Branch A on the registered classical-vacuum domain. Receipts
  catalog, closure-residual summary, limitations and scope, six
  suggested next minimal extensions, and a fidelity audit (including
  independent hand-verification of the Phase 4 nonlocal-falsifier
  residual to twelve digits) are on the record. Four soft hygiene
  notes recorded transparently with no Branch-A impact. The
  experiment chapter is **closed**.
- 2026-05-25 - Phase 6 local site-readiness artifacts landed for `/faraday`:
  `public/og/faraday.png`, full OG/Twitter metadata, JSON-LD `TechArticle`,
  homepage pillar link, `site-pages.json` evidence-page promotion, and
  `public/sitemap.xml` coverage. This supersedes the earlier noindex/Class D
  staging state. Post-deploy validator pass remains external.
