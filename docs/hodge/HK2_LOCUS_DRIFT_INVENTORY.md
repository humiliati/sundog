# H-K2 - Hodge-Locus Drift Harness: Candidate Inventory + Kill Decision

- Artifact id: `HODGE-HK2-LOCUS-DRIFT-INVENTORY`
- Date: 2026-06-29
- Status: internal inventory + kill decision (the deliverable the slate specifies - not a page,
  not a harness build). **Falsifier `HODGE_LOCUS_TOO_SUBTLE` FIRES - do not promote the hook.**
- Slate hook: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md) (H-K2)
- Related: [`PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md`](PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md)
  (the static generator this decision falls back to), card `HODGE-RG-009` (the CDK fence, already
  live and model-tested).

## The hook being evaluated

Track when a class stays of type `(p,p)` across a family ("shadow drift/stability"), and teach
the persistence profile without constructing cycles. The falsifier fires if every honest example
either (a) needs enough specialist machinery that the harness is a disguised survey, or (b) has
visual language that makes type-persistence look like cycle construction.

## Candidate inventory (three examples, per the slate's first move)

### 1. Divisor-level safe case: Noether-Lefschetz loci for surfaces in P^3

Family: smooth degree-`d` surfaces in `P^3` (`d >= 4`). Generic member: Picard rank 1 (only the
hyperplane class). On the Noether-Lefschetz locus (a countable union of proper subvarieties of
the parameter space) an extra rational `(1,1)` class persists. K3 variant: in the K3 period
domain, rational `(1,1)` classes survive only on countably many NL divisors.

- *Legibility:* good - "the extra class survives only on a thin special locus" is drawable and
  classical.
- **The problem (falsifier clause b, and worse):** at `p = 1`, persistence **is** construction -
  wherever the rational `(1,1)` class persists, Lefschetz (1,1) *constructs* the divisor. A
  drift harness built on this case does not merely *risk* making persistence look like
  construction; in the only regime the case exhibits, persistence **literally implies**
  construction. It teaches the exact opposite of the lesson the harness exists to teach
  ("persists" != "constructed").

### 2. Special-locus case with the real split: abelian fourfolds of Weil type

Weil classes: rational `(2,2)` classes on abelian fourfolds with suitable complex
multiplication; the Weil locus is a classical special locus (consistent with Cattani-Deligne-
Kaplan algebraicity), the classes persist along it, and the corresponding cycles are unknown in
general (constructed only in scattered special cases, e.g. Schoen). This is the genuine
"persists but not constructed" exhibit.

- *Correctness:* the split is real here - exactly what the harness wants to display.
- **The problem (falsifier clause a):** stating the example honestly requires abelian
  varieties, endomorphism algebras, period lattices, and the Weil-class construction. Every
  legible simplification either quietly asserts the class exists "by symmetry" (untrue as
  stated) or reduces to an unverifiable cartoon. The harness becomes a disguised survey.

### 3. Hard-excluded boundary case: general fourfold codim-2 drift / compact Kaehler families

The general `(2,2)` Hodge-locus story on fourfolds (CDK guarantees the locus is algebraic;
nothing constructs the cycle - the content of card `HODGE-RG-009`), and Kaehler non-projective
families (complex tori), which sit outside the roster entirely (`HODGE-RG-008` fence).

- *Role:* stays labeled as boundary, per the slate. Not a harness candidate; it is the fence
  the static cards already encode.

## Kill decision: `HODGE_LOCUS_TOO_SUBTLE` FIRES

The three candidates form a pincer that hits both falsifier clauses with nothing in between:

- Every **teachable** example lives at `p = 1`, where persistence implies construction
  (Lefschetz (1,1)) - so the harness's visuals would teach "persistence => cycle," the
  *opposite* of its purpose (clause b, in its strongest form).
- Every example exhibiting the **true split** (`p >= 2`: Weil type and relatives) requires
  specialist machinery that turns the harness into a survey (clause a).
- The boundary case is already a fence, not a harness.

Per the slate's own instruction on firing: **do not promote the hook; keep the static register
problem generator.** The static generator already carries the one safe drift lesson as
`HODGE-RG-009` ("Hodge-locus algebraicity does not construct the cycle") - and the PHASE4D/4F
model runs show that card doing live work (models overclaim it unprompted; the judge catches
it). The drift lesson is thus already delivered, statically, at the right register.

## What would reopen this

A future concrete family where the persist-vs-construct split occurs at divisor-adjacent
legibility (no candidate currently known to this inventory), or a specialist collaborator
willing to co-sign a Weil-type exhibit at cartoon precision. Until one exists, H-K2 is closed
as falsified-by-inventory.

## Interpretation Boundary

This is a scoping decision about a *teaching harness*, not a mathematical claim about Hodge
loci. CDK algebraicity, Noether-Lefschetz theory, and the Weil-class literature are untouched;
no public claim is made or licensed.
