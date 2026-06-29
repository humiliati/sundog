# Hodge/Kakeya - Fresh Hypotheses Slate (opened 2026-06-29)

> **What this is.** A working slate for lanes whose public promotion is waiting
> on outside sanity checks, but whose internal idea-generation surface is already
> rich enough to probe. The first two substrates are Hodge and Kakeya because
> both have a clean body/shadow dictionary, known-safe examples, and a visible
> gap between "the shadow has the right form" and "the body is recovered."
>
> **Working rule.** The fences are falsifiers, not the idea generator. A hook can
> be speculative at birth if its first receipt can come back NULL, MISLABELED,
> VACUOUS, or SHADOW-REENCODING without damaging the lane. No hook here is a
> public claim or a promotion path by itself.
>
> **Provenance.** Authored from the live Hodge and Kakeya ledgers:
> [`SUNDOG_V_HODGE.md`](SUNDOG_V_HODGE.md),
> [`HODGE_READER_NOTE.md`](HODGE_READER_NOTE.md),
> [`hodge/PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md`](hodge/PHASE3_KNOWN_EXAMPLE_GALLERY_SPEC.md),
> [`SUNDOG_V_KAKEYA.md`](SUNDOG_V_KAKEYA.md), and
> [`kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md`](kakeya/PHASE2_TINY_FINITE_FIELD_WORKBENCH_SPEC.md).

## Tier legend

- **[PROBLEM-GENERATOR]** - produces a reusable exercise/evaluator family whose
  correctness is checked against known mathematics.
- **[EMPIRICAL]** - a runnable finite/toy experiment with a falsifiable statistic.
- **[IMPORT-AND-CHECK]** - the mathematical theorem exists elsewhere; the work is
  checking that the Sundog translation is faithful and finding the break point.
- **[FORMALIZABLE]** - a small Lean or typed-core target, probably on a toy
  abstraction rather than the full external theorem.
- **[SYNTHESIS]** - a cross-lane organizing claim that earns status only after
  at least one concrete hook below lands.

## The shared object

Both lanes are blocked as public "claims" but open as hypothesis factories:

```text
Hodge:  rational (p,p) class       -> asks for an algebraic cycle body
Kakeya: direction-complete shadow  -> asks how small / constrained the body can be
```

The useful shared question is not "can Sundog solve the famous problem?" It is:

```text
What shadow data is necessary, what body data is hidden, and which toy substrates
let us measure the loss without pretending the toy result transfers?
```

---

## H-K1 - The register-ladder problem generator. [PROBLEM-GENERATOR]

*Status - SPEC + JSONL + AUDIT LANDED 2026-06-29.* The spec is at
[`hodge/PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md`](hodge/PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md)
(card schema, falsifier, answer-key audit, 10 seed cards across the reader's R1-R4
ladder, gallery rows G1-G5, and hard-excluded boundary rows). The 10 cards are now
emitted as [`hodge/register-problem-cards.jsonl`](hodge/register-problem-cards.jsonl)
and pass the mechanized section-3 + section-6 audit (`npm run hodge:register-audit`;
`scripts/hodge-register-card-audit.mjs`): **10/10 pass, 0 blocked, 7 cycle / 3
none-licensed boundary, `REGISTER_PROBLEMS_VACUOUS` clear**, receipt at
[`hodge/PHASE4B_REGISTER_CARD_AUDIT.md`](hodge/PHASE4B_REGISTER_CARD_AUDIT.md). The
audit is structural (auditability against the named sources), not a math
certification; a specialist spot-check stays the promotion gate.

A route/fence-fidelity eval (`npm run hodge:register-eval`;
`scripts/hodge-register-eval.mjs`) then scored each card's gold vs trap answer with a
fixed route/fence rubric: **10/10 discriminating, gold-route 10/10, trap-route 0/10,
mean separation 4.0, eval-ready**, receipt at
[`hodge/PHASE4C_REGISTER_FIDELITY_EVAL.md`](hodge/PHASE4C_REGISTER_FIDELITY_EVAL.md) -
the cards are a valid route/fence eval set and the scorer is established.

The model-in-the-loop run then closed the K1 loop (`scripts/hodge-register-modeleval.mjs`;
secret-safe keyring): OpenAI `gpt-4o-mini` answered each card's prompt, scored by the same
rubric, in two modes. **Neutral (no register cue): 5 fenced / 3 overclaimed / 2 ambiguous
- the cards catch real overclaims** (RG-007 rational->integral upgrade = the integral
Hodge conjecture; RG-009 Hodge-locus/CDK -> "cycle found"; RG-006). **Primed (register
cue): 7 fenced / 0 overclaimed / 3 ambiguous - priming removes every overclaim and lifts
fencing 5->7.** Receipt
[`hodge/PHASE4D_REGISTER_MODELEVAL.md`](hodge/PHASE4D_REGISTER_MODELEVAL.md). The lexical
rubric cannot classify the constructive-body cards (RG-003/004/005, ambiguous); a
`target_register` route check + semantic judge is the next refinement. Route/fence
fidelity only (per H-K6), one model / one phrasing, exploratory - not a math
certification, no UI / no public page. **H-K1 generator->audit->fidelity->model loop is
complete.**

PHASE4E then (i) added the named **route check** (per-card, derived from `correct_answer`
stance + `body` field; verdict space fenced/routed/overclaimed/hedged/off) that resolves
the constructive-card blind spot, and (ii) **swept it across models x phrasings**
(`--sweep`; receipt [`hodge/PHASE4E_REGISTER_SWEEP.md`](hodge/PHASE4E_REGISTER_SWEEP.md)).
Live: openai gpt-4o-mini, groq llama-3.3-70b, mistral-small (anthropic key dead, skipped).
**Unprompted route-correct is model-dependent: llama-70b 9/10 (0 overclaim) > gpt-4o-mini
6/10 (3) > mistral-small 4/10 (3); RG-007 (integral Hodge) + RG-009 (Hodge-locus/CDK) are
the consistently-hardest traps (both proprietary models fall, llama resists). Priming ->
0 overclaims for all three** (mistral 4->10, gpt-4o-mini 6->9; groq dips 9->8 via
over-hedging). Still exploratory, route/fence only, no public claim; specialist
spot-check (PHASE4B) stays the promotion gate.

PHASE4F **sharpened with two independent semantic judges** (`scripts/hodge-register-judge.mjs`;
receipt [`hodge/PHASE4F_REGISTER_SEMANTIC_JUDGE.md`](hodge/PHASE4F_REGISTER_SEMANTIC_JUDGE.md))
- and the sharper instrument **corrected PHASE4E**. Judge instrument is reliable
(inter-judge verdict agreement 0.80) but the **lexical route check over-credited fidelity:
judge-consensus vs lexical agreement only 0.60** (it credited surface fence-words on answers
that semantically commit the trap - RG-002/RG-008 boundary cards). **The PHASE4E "llama-70b
resists unprompted (0 overclaim)" claim does NOT survive - judges find ~3 overclaims it
missed; under semantic grading all three models are poor unprompted (independent-judge
route-correct 2-4/10).** Surviving: priming still lifts route-correct (consensus openai
0->5, mistral 3->7, groq 3->4); hardest cards = the real false-in-general fences
(RG-002/007/008/009). The lexical proxy is a shadow that can't see where the overclaim lives
(meaning) - the cross-lane determine/resist invariant in miniature.

*Claim.* Hodge can deliver a meaningful Sundog artifact before it delivers a
technical probe: a generator of small problems that force a learner or model to
distinguish the four registers in the reader note:

```text
form -> harmonic representative -> rational (p,p) class -> algebraic cycle
```

The generator should produce paired prompts where the same visible phrase is
correct in one register and wrong in another. Example types: "is this a form, a
class, or a cycle?", "which arrow is a theorem?", "which reverse arrow is exactly
the conjectural one?", and "which known case makes this body available?"

*Traction.* This uses the strongest part of the Hodge lane: the register ladder
in `HODGE_READER_NOTE.md` and the known-example roster in the Phase 3 gallery
spec. It converts the SEO/reader surface into a testable teaching/evaluator
surface.

*Classical hook.* Hodge decomposition, Lefschetz `(1,1)`, hard Lefschetz dual
case, rational-vs-integral boundary.

*Falsifier* (`REGISTER_PROBLEMS_VACUOUS`): the generated questions reduce to
plain definition lookup, or the answer key cannot be checked against the known
roster without expert intervention. If this fires, Hodge stays a reader lane and
not a problem-generator lane.

*First move.* Write `docs/hodge/PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md` with
10 seed cards, each carrying: body, shadow, register, known-because, tempting
wrong answer, and source row. No UI. No model eval. Just a card schema and an
answer-key audit.

---

## H-K2 - Hodge-locus drift as a shadow-stability harness. [IMPORT-AND-CHECK]

*Claim.* The Hodge lane can probe something more alive than static examples by
tracking when a class stays of type `(p,p)` across a family. The Sundog reading:
the shadow is not merely visible at one point; it has a deformation-stability
profile. That profile can be taught and audited without constructing new cycles.

*Traction.* `HODGE_LITPASS_MEMO.md` already cites Cattani-Deligne-Kaplan as a
structured-shadow fact. This hook turns that citation into a concrete harness:
choose a known family where the special locus is classical, label what remains
type `(p,p)`, and ask whether the body/shadow vocabulary helps separate
"type persists" from "cycle constructed."

*Classical hook.* Hodge loci, Noether-Lefschetz-style special loci, variation of
Hodge structure.

*Falsifier* (`HODGE_LOCUS_TOO_SUBTLE`): every honest example requires enough
specialist machinery that the harness becomes a disguised survey, or the visual
language makes type-persistence look like cycle construction. If this fires, do
not promote the hook; keep the static register problem generator.

*First move.* File a one-page candidate inventory with three possible examples:
one divisor-level safe case, one surface/special-locus case, and one hard-excluded
case that stays labeled as a boundary. The deliverable is the inventory plus
kill decision, not a page.

---

## H-K3 - Direction-shadow collision audit for finite-field Kakeya. [EMPIRICAL]

*Status - RECEIPT EXTENDED 2026-06-29.* The audit is implemented as
`npm run kakeya:shadow-collision:all` and recorded at
[`kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md`](kakeya/PHASE3B_SHADOW_COLLISION_AUDIT.md).
For `q=5` with body size `<=6`, the bounded enumeration checked `245506` bodies,
collapsed them to `7` direction-shadow signatures, found collisions in every
observed signature, and cleared `KAK_SHADOW_REENCODING_EMPIRICAL`. The strongest
nonempty teaching witness is a size-5 line and the same line plus one extra
point sharing the same one-direction shadow. The `q=7` extension uses a
deterministic line-extension family (`2408` states, `8` one-direction signatures,
max nonempty collision class `301`) and the core regression suite now pins
different-size same-shadow collisions across `q in {5, 7, 11}`.

*q=11 + scaling law (PHASE3C, 2026-06-29).* Extended to the largest supported
field (`npm run kakeya:shadow-collision:q11`; receipt
[`kakeya/PHASE3C_SHADOW_COLLISION_Q11.md`](kakeya/PHASE3C_SHADOW_COLLISION_Q11.md)):
`12` structured signatures, max nonempty collision class `1221`, guard pass,
falsifier clear, regression `39/39`. Across all three rungs the largest nonempty
collision class is **exactly `q(q^2 - q + 1)`** (`5->105, 7->301, 11->1221`),
i.e. the shadow's lossiness grows **cubically in `q`** - a closed-form derived
from the `q` slope-0 lines plus their `q^2-q` one-point extensions. Note the
bounded brute force is line-free (vacuous) for `q>=7` since a line needs `q>6`
points, so the structured family is the ceiling of this approach.

*Claim.* A finite-field Kakeya workbench becomes more than a spectacle if its
displayed direction shadow is provably many-to-one: different point sets can
produce the same coverage signature. The many-to-one collision count is the
workbench's proof that the shadow is lossy rather than a disguised body encoding.

*Traction.* The Kakeya Phase 2 spec already demands a shadow export that omits
point membership and witness intercepts. This hook makes that requirement a
measurement: enumerate or sample point sets in small `F_q^2`, group by direction
coverage signature, and report collision classes with different bodies and sizes.

*Classical hook.* Finite-field lines, direction coverage, Dvir finite-field
Kakeya theorem as the known background.

*Falsifier* (`KAK_SHADOW_REENCODING_EMPIRICAL`): for the small displayed domains,
the chosen shadow is almost injective on the admissible workbench states, or the
UI cannot explain collisions without showing the omitted body data. Then the
shadow is pedagogically unsafe as a projection.

*First move.* Add a cheap Node script over the existing Kakeya core:
`scripts/kakeya-shadow-collision-audit.mjs`, starting with `q=5`, with a cap on
body size or a sampling mode if full enumeration is too large. Receipt fields:
`q`, state count, signature count, max collision size, two witness bodies with the
same signature, and whether the bodies differ in size.

---

## H-K4 - Adaptive-fibering ambiguity is the Kakeya obstruction Sundog can show. [EMPIRICAL]

*Claim.* The Kakeya body-resistance bridge becomes testable in toy form by
measuring ambiguity at points that lie on many candidate lines. A fixed direction
shadow is clean; an adaptive choice of direction can compress differently. The
gap between fixed and adaptive fiber labels is a toy proxy for why the Euclidean
boundary is hard to communicate.

*Traction.* `SUNDOG_V_KAKEYA.md` already names the Polson-Zantedeschi adaptive
fibering obstruction as the real body-resistance hook. The workbench can expose
the obstruction by counting, for each selected point, how many direction/intercept
pairs can explain it, then comparing fixed-label residuals against best-adaptive
residuals.

*Classical hook.* Point-to-set / conditional Kolmogorov-complexity reading of
Kakeya, fiber labels, direction shadows.

*Falsifier* (`ADAPTIVE_FIBERING_NO_SIGNAL`): adaptive choice does not change the
toy compression/residual statistic, or the statistic is entirely an artifact of
finite-grid enumeration. Then the body-resistance bridge remains prose, not a
workbench metric.

*First move.* Extend the collision audit with a report-only ambiguity panel:
for each body, compute average line-incidence multiplicity per point and compare
it to direction-coverage size. Keep it report-only until a preregistered statistic
is chosen.

---

## H-K5 - The Hodge/Kakeya shadow-collision table. [SYNTHESIS]

*Claim.* Hodge and Kakeya share a useful table shape:

| Lane | Shadow | Body | Collision question |
| --- | --- | --- | --- |
| Hodge | rational `(p,p)` class | algebraic cycle | how many bodies can cast the same class, and when is existence known? |
| Kakeya | direction coverage signature | point/tube set | how many bodies can cast the same direction shadow, and when is size forced? |

The table does not identify the theorems. It identifies the workbench move:
prove or measure that the shadow loses body information, then ask which extra
structure restores existence, size, or uniqueness.

*Traction.* This is the simplest cross-lane object that can spawn problem cards,
UI panels, and evaluator prompts. It also gives Hodge a non-SEO job: Hodge is the
existence pole; Kakeya is the size/resistance pole.

*Classical hook.* Cycle class map for Hodge; direction incidence for Kakeya;
fiber-body decomposition in the algorithmic-information Kakeya literature.

*Falsifier* (`COLLISION_TABLE_ONLY_RHYME`): the two rows cannot share any
operational field beyond vocabulary. If Hodge's "collision" is pure algebraic
equivalence while Kakeya's is finite-state enumeration, the synthesis stays a
reader metaphor and cannot guide probes.

*First move.* Build the table as a two-page internal note after H-K1 or H-K3
lands. Do not write it first; require one concrete receipt so the synthesis is
anchored.

---

## H-K6 - Known-example cards can be adversarial prompts, not just exhibits. [PROBLEM-GENERATOR]

*Claim.* The known-example rosters in Hodge and Kakeya can generate adversarial
prompts for Sundog Chat and future evaluator lanes. The useful prompts are not
"state the theorem"; they are boundary-preserving transformations:

```text
turn this known case into a tempting overclaim;
repair it without losing the body/shadow hook;
classify which register changed;
name the falsifier that would fire.
```

*Traction.* Both lanes already have forbidden phrases and named negatives. This
hook turns those fences into data rather than static prose. It also gives the
LinkedIn/academic feedback problem a lower-friction substitute: if a domain
expert only gives a quick thumbs-up or a one-line correction, that correction can
be converted into a new adversarial card.

*Classical hook.* Known Hodge cases; finite-field Kakeya theorem; Euclidean
boundary after Wang-Zahl in `R^3`.

*Falsifier* (`ADVERSARIAL_CARDS_DO_NOT_STICK`): generated cards either train the
model to recite disclaimers without preserving the mathematical distinction, or
they fail to catch seeded overclaims that the lane ledgers already name.

*First move.* Add a JSONL seed file with 12 cards:
6 Hodge register cards, 4 Kakeya finite-field/Euclidean boundary cards, and
2 cross-lane body/shadow collision cards. Score only route/fence fidelity at
first; no public claim quality metric.

---

## What this slate is not

- It is not a Hodge or Kakeya result.
- It is not a replacement for a specialist review when a public theorem-facing
  claim is made.
- It is not a page-launch checklist.
- It is not limited to "safe" reader copy: hypotheses are allowed to be strange
  if the first receipt can kill them cleanly.

## Recommended next attacks

1. **H-K3 - direction-shadow collision audit. [EMPIRICAL]** This is the cheapest
   runnable hook and directly hardens the Kakeya workbench's load-bearing
   projection claim.
2. **H-K1 - register-ladder problem generator. [PROBLEM-GENERATOR]** This is the
   best Hodge move because it makes the reader useful without waiting for a public
   promotion gate.
3. **H-K6 - adversarial known-example cards. [PROBLEM-GENERATOR]** Best bridge
   into the existing chat/evaluator corpus once H-K1 has seed cards.
4. **H-K2 - Hodge-locus drift harness. [IMPORT-AND-CHECK]** Higher upside, higher
   subtlety. Do only after the static Hodge cards prove they can stay accurate.
5. **H-K5 - shadow-collision table. [SYNTHESIS]** Write after H-K1 or H-K3 lands,
   not before.

> Cross-links: [`SUNDOG_V_HODGE.md`](SUNDOG_V_HODGE.md) -
> [`SUNDOG_V_KAKEYA.md`](SUNDOG_V_KAKEYA.md) -
> [`HODGE_LITPASS_MEMO.md`](HODGE_LITPASS_MEMO.md) -
> [`KAKEYA_LITPASS_MEMO.md`](KAKEYA_LITPASS_MEMO.md) -
> [`CROSS_SUBSTRATE_NOTES.md`](CROSS_SUBSTRATE_NOTES.md).
