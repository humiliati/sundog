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

Working definition to derive on paper:

1. Integrate the variational equation along each of the 21 strict
   choreographies at `m3=1` to get the monodromy matrix `M_i`.
2. Project `M_i` onto the piano-trio twisted sector associated with
   `alpha_I = ((12), T/2)`.
3. Define `d_i` as the count of `+1` Floquet eigenvalues in that sector after
   excluding the structural `F_beta` sector.
4. Freeze the 21 integers and their sum before any supplementary-B clustering.

The load-bearing deliverable is not a run. It is the projector derivation:
which finite-dimensional twisted subspace corresponds to `alpha_I`, how the
structural `F_beta` sector is removed, and which neutral modes are quotiented
before counting `+1`. If that derivation is fuzzy, v0.3 is not ready.

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

## Operator Guardrail

Before any future isotrophy run:

1. Confirm whether the work is G.2 detector maintenance, v0.3 derivation, or
   accidental K2-K4 restart.
2. If it is K2-K4 under v0.2, stop.
3. If it is v0.3, write the `alpha_I` twisted-sector projector derivation,
   the neutral-mode quotient, the negative, and the go/no-go branch before
   running any monodromy or supplementary-B classification.
