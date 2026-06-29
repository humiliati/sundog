# Hodge Phase 4 - Register Problem Generator Spec

- Artifact id: `HODGE_PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC`
- Type: internal Phase 4 problem-generator spec and seed card set
- Date: 2026-06-29
- Parent slate: [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Hodge ledger: [`../SUNDOG_V_HODGE.md`](../SUNDOG_V_HODGE.md)
- Reader source: [`../HODGE_READER_NOTE.md`](../HODGE_READER_NOTE.md)
- Known-example roster: [`PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md`](PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md)
- Status: H-K1 spec landed. No UI, no model eval, no public page, no claim about
  any open Hodge case.

> The card is not "state Hodge." The card is "which register did you just
> confuse?"

## 1. What this is

This spec turns the Hodge reader's R1-R4 register ladder into a reusable problem
generator. The generator makes small cards that force a learner or evaluator to
separate:

```text
R1: closed differential form / representative
R2: harmonic representative
R3: rational (p,p) cohomology class / shadow
R4: algebraic cycle / body
```

The generator is useful only if the answer key is checkable against known
mathematics already in the Hodge lane: the reader ladder, the Phase 3 known
roster, and the hard-excluded boundary rows.

## 2. What this is not

- Not a public Hodge page.
- Not a proof, detector, or construction of algebraic cycles.
- Not a model benchmark yet.
- Not a gallery rendering.
- Not a replacement for source citation when public copy is written.

## 3. Card schema

Each generated card must carry these fields:

| Field | Requirement |
| --- | --- |
| `id` | Stable id, `HODGE-RG-###`. |
| `source_row` | Reader register / CE tag and, where applicable, gallery row `G1`-`G5` or hard-excluded row. |
| `prompt` | One short question or repair task. |
| `target_register` | One of `R1`, `R2`, `R3`, `R4`, or a named arrow such as `R4->R3` / `R3->R4`. |
| `body` | The algebraic cycle body if known; otherwise explicitly `none licensed`. |
| `shadow` | The rational `(p,p)` class if in scope; otherwise the nearest visible representative/class and why it is not enough. |
| `known_because` | The theorem or roster rule that makes the answer checkable; use `none - boundary card` for open/excluded cases. |
| `tempting_wrong_answer` | The plausible mistake the card is designed to catch. |
| `correct_answer` | The answer key. |
| `falsifier_tags` | One or more tags from the Hodge lane or this spec. |

Card generation fails if a card has no source row, no tempting wrong answer, or
an answer that depends on unstated expert judgment.

## 4. Falsifier

`REGISTER_PROBLEMS_VACUOUS` fires if either condition holds:

1. The cards reduce to definition lookup and do not test a register transition,
   category error, known-case boundary, or source-row fence.
2. The answer key cannot be audited from `HODGE_READER_NOTE.md`,
   `PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md`, or the lit-pass fences.

Secondary failure tags inherited from the lane:

- `HODGE-CATEGORY-ERROR`: the card itself conflates form, harmonic
  representative, rational class, and cycle.
- `HODGE-LIT-MISMATCH`: the "known because" field cites the wrong theorem or
  treats an open/excluded case as known.
- `HODGE-TOY-LAUNDERING`: a known example is made to look like evidence for an
  open case.

## 5. Seed cards

### HODGE-RG-001 - Harmonic is not rational

- `source_row`: reader `CE1`, gallery `G3`
- `prompt`: A diagram labels a harmonic `(1,1)` representative on a smooth
  projective variety `X`. May the caption call that harmonic form "the rational
  Hodge class"?
- `target_register`: `R2` vs `R3`
- `body`: a divisor `D`, but only when a rational `(1,1)` class is supplied.
- `shadow`: a class in `H^2(X,Q) cap H^{1,1}(X)`, not the harmonic form by
  itself.
- `known_because`: Lefschetz `(1,1)` realizes rational `(1,1)` classes by
  divisors; the harmonic representative is metric-dependent and is not the
  rationality datum.
- `tempting_wrong_answer`: Yes; the harmonic `(1,1)` representative is the
  rational class.
- `correct_answer`: No. A harmonic representative sits at `R2`; rationality is a
  property of the cohomology class at `R3`. The caption must say "a harmonic
  representative of a class" unless the rational class has been specified.
- `falsifier_tags`: `HODGE-CATEGORY-ERROR`, `REGISTER_PROBLEMS_VACUOUS`

### HODGE-RG-002 - Type is not algebraicity

- `source_row`: reader `CE2`, gallery hard-excluded `general fourfold codim-two`
- `prompt`: A rational class `alpha in H^4(X,Q) cap H^{2,2}(X)` is given on a
  smooth projective fourfold. Does the register ladder let the card supply an
  algebraic cycle body?
- `target_register`: `R3->R4`
- `body`: none licensed.
- `shadow`: the rational `(2,2)` class `alpha`.
- `known_because`: none - boundary card. The general fourfold codimension-two
  case is the first general open Hodge range in the roster.
- `tempting_wrong_answer`: Yes; type `(2,2)` is exactly the condition for being a
  cycle class.
- `correct_answer`: No. Rational type `(p,p)` is the admissible shadow. The
  reverse direction from shadow to body is the Hodge conjecture, not a generator
  rule.
- `falsifier_tags`: `HODGE-CATEGORY-ERROR`, `HODGE-LIT-MISMATCH`

### HODGE-RG-003 - Projective-space warm-up

- `source_row`: gallery `G1`
- `prompt`: In `P^n`, a codimension-`k` linear subspace `P^{n-k}` is selected.
  Which object is the body, which object is the shadow, and which direction is
  theorem-defined?
- `target_register`: `R4->R3`
- `body`: the linear subspace `P^{n-k}` and `Q`-linear combinations of such
  subspaces.
- `shadow`: the generator of `H^{2k}(P^n,Q)`, of type `(k,k)`.
- `known_because`: the cohomology ring of projective space is generated by the
  hyperplane class, so each even cohomology group is spanned by an algebraic
  class.
- `tempting_wrong_answer`: The cohomology generator constructs the linear
  subspace by reading the shadow backward.
- `correct_answer`: The body is the linear subspace. The cycle class map sends
  body to shadow. Projective space is known because its cohomology is already
  generated by algebraic hyperplane powers, not because the general reverse map
  has been solved.
- `falsifier_tags`: `HODGE-TOY-LAUNDERING`

### HODGE-RG-004 - Curves do not test the middle

- `source_row`: gallery `G2`
- `prompt`: A smooth projective curve has rich `H^1`. Should a first Hodge card
  use `H^1` as the shadow whose algebraic body is being tested?
- `target_register`: roster admission / `R3`
- `body`: a point, for the admitted `H^2` row.
- `shadow`: the generator of `H^2(C,Q)`, of type `(1,1)`.
- `known_because`: `H^2` of a curve is spanned by the point class; the interesting
  `H^1` decomposes as `(1,0)+(0,1)` and is not a Hodge `(p,p)` target.
- `tempting_wrong_answer`: Yes; use the curve's rich `H^1` as the hidden-body
  card because it is where the geometry lives.
- `correct_answer`: No. The admitted row is the point class in `H^2`, which is
  a trivial known case. `H^1` is not the `(p,p)` shadow for a Hodge-cycle card.
- `falsifier_tags`: `HODGE-LIT-MISMATCH`, `HODGE-CATEGORY-ERROR`

### HODGE-RG-005 - Divisors are the flagship, not the whole conjecture

- `source_row`: gallery `G3`
- `prompt`: A rational `(1,1)` class on a smooth projective variety is supplied.
  What body can the answer key name, and what must it not generalize to?
- `target_register`: `R3->R4` in the known `p=1` case
- `body`: a `Q`-combination of divisors.
- `shadow`: a rational `(1,1)` class in `H^2(X,Q) cap H^{1,1}(X)`.
- `known_because`: Lefschetz `(1,1)` theorem.
- `tempting_wrong_answer`: Since divisors work, rational `(p,p)` classes in
  higher codimension work by the same mechanism.
- `correct_answer`: The body is a divisor class, rationally. The mechanism is
  codimension one and does not extend automatically to codimension `>= 2`.
- `falsifier_tags`: `HODGE-TOY-LAUNDERING`, `HODGE-LIT-MISMATCH`

### HODGE-RG-006 - Surface curves are still divisors

- `source_row`: gallery `G4`
- `prompt`: On a smooth projective surface `S`, a rational `(1,1)` class is
  presented as a curve class. Is this a new Hodge phenomenon beyond divisors?
- `target_register`: `R4->R3`, known `p=1`
- `body`: a curve `C subset S`, equivalently a divisor on the surface.
- `shadow`: a rational `(1,1)` class, i.e. an element of `NS(S) tensor Q`.
- `known_because`: Lefschetz `(1,1)`, specialized to surfaces.
- `tempting_wrong_answer`: Yes; curves on a surface are a separate
  higher-codimension case.
- `correct_answer`: No. A curve on a surface has codimension one. This is the
  divisor theorem again, in concrete clothing.
- `falsifier_tags`: `HODGE-LIT-MISMATCH`

### HODGE-RG-007 - Threefold rational is not integral

- `source_row`: gallery `G5`
- `prompt`: On a smooth projective threefold, a rational `(2,2)` class is known
  to be represented by a curve class. May the card silently upgrade this to an
  integral statement?
- `target_register`: `R3->R4`, rational/integral fence
- `body`: a `Q`-combination of curves.
- `shadow`: a rational `(2,2)` class in `H^4(X,Q) cap H^{2,2}(X)`.
- `known_because`: hard Lefschetz reduces codimension `n-1` rational Hodge to
  codimension one.
- `tempting_wrong_answer`: Yes; if the rational class is algebraic, every
  integral class in the same range is algebraic too.
- `correct_answer`: No. The row is rational only. Integral Hodge is a separate
  false-in-general variant and must be labeled separately.
- `falsifier_tags`: `HODGE-LIT-MISMATCH`, `HODGE-TOY-LAUNDERING`

### HODGE-RG-008 - Projective is not just Kaehler

- `source_row`: reader fence `projective, not Kaehler`; gallery hard-excluded
  `compact Kaehler`
- `prompt`: A compact Kaehler manifold has harmonic representatives and a
  rational-looking `(p,p)` class. Can the generator reuse the smooth-projective
  Hodge card without changing the source row?
- `target_register`: domain fence before `R3->R4`
- `body`: none licensed.
- `shadow`: a `(p,p)` class outside the smooth-projective roster.
- `known_because`: none - boundary card. The reader fence excludes compact
  Kaehler as a substitute for smooth projective.
- `tempting_wrong_answer`: Yes; Hodge theory and harmonic representatives exist,
  so the same algebraic-cycle question applies unchanged.
- `correct_answer`: No. The rational Hodge conjecture lane is smooth projective.
  Kaehler analogues are not safe substitutes and must not be admitted by the card
  generator.
- `falsifier_tags`: `HODGE-LIT-MISMATCH`, `HODGE-CATEGORY-ERROR`

### HODGE-RG-009 - Hodge loci are not cycle construction

- `source_row`: reader fence `Hodge loci != cycle construction`
- `prompt`: In a family, the locus where a fixed class stays of type `(p,p)` is
  algebraic. Does that algebraicity construct the algebraic cycle body?
- `target_register`: shadow stability vs `R3->R4`
- `body`: none licensed by this fact alone.
- `shadow`: a fixed cohomology class whose type `(p,p)` persists on a locus.
- `known_because`: Cattani-Deligne-Kaplan supports algebraicity of Hodge loci,
  not construction of cycles.
- `tempting_wrong_answer`: Yes; if the Hodge locus is algebraic, the cycle behind
  the class has been found.
- `correct_answer`: No. A structured shadow-locus is not a body construction.
  The card may teach shadow stability, but it cannot claim algebraicity of the
  class.
- `falsifier_tags`: `HODGE-CATEGORY-ERROR`, `HODGE-TOY-LAUNDERING`

### HODGE-RG-010 - A picture is not a body

- `source_row`: reader `CE3`, gallery `G3`
- `prompt`: A visual shows a smooth representative of a `(1,1)` class and labels
  a highlighted region as "the cycle." What repair should the answer key make?
- `target_register`: `R1/R2` vs `R4`
- `body`: a divisor `D`, if separately certified by Lefschetz `(1,1)`.
- `shadow`: the rational `(1,1)` class represented by the displayed form, if the
  rational class is specified.
- `known_because`: the visual does not certify anything; Lefschetz `(1,1)` is the
  known-case source for the divisor body.
- `tempting_wrong_answer`: The rendered form displays the algebraic cycle.
- `correct_answer`: The visual can display a representative, not the body. Repair
  the caption to "representative of the class"; name the divisor body only through
  the theorem/source row, not through the picture.
- `falsifier_tags`: `HODGE-CATEGORY-ERROR`, `HODGE-VISUAL-MISCALIBRATED`

## 6. Answer-key audit

Before any generated expansion, audit every card with this checklist:

1. The card names a target register or arrow.
2. The source row is present and points to the reader ladder, a gallery row, or a
   hard-excluded boundary row.
3. The body field is either a known algebraic cycle or explicitly `none
   licensed`.
4. The shadow field is not a harmonic representative unless the card is testing
   that confusion.
5. The tempting wrong answer maps to one named category error or boundary.
6. The correct answer can be checked without an unstated specialist judgment.

If a card fails any item, remove it from the seed set or mark it
`answer_key_blocked`.

## 7. What this licenses

This spec licenses only:

- hand-authored seed-card expansion;
- a future JSONL seed file using this schema;
- a future route/fence-fidelity eval over the cards.

It does not license a public Hodge page, visual gallery, external promotion, or
claim that the problem generator has mathematical reach beyond register
discipline.
