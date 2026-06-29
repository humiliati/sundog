# Hodge Phase 4B - Register Card Audit

- Artifact id: `HODGE-PHASE4B-REGISTER-CARD-AUDIT`
- Date: 2026-06-29
- Status: internal answer-key-discipline receipt for the H-K1 register cards.
- Ledger: [`../SUNDOG_V_HODGE.md`](../SUNDOG_V_HODGE.md)
- Spec: [`PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md`](PHASE4_REGISTER_PROBLEM_GENERATOR_SPEC.md)
- Slate hook:
  [`../HODGE_KAKEYA_HYPOTHESES_SLATE.md`](../HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Cards: [`register-problem-cards.jsonl`](register-problem-cards.jsonl)
- Script: [`../../scripts/hodge-register-card-audit.mjs`](../../scripts/hodge-register-card-audit.mjs)
- Output manifest:
  [`../../results/hodge/register-card-audit/manifest.json`](../../results/hodge/register-card-audit/manifest.json)

## Verdict

**H-K1 advanced from spec to a checked artifact.** The 10 seed cards are emitted as
JSONL and pass the mechanized Phase-4 spec audit: the section-3 generation gate and
the section-6 answer-key checklist. The `REGISTER_PROBLEMS_VACUOUS` falsifier did
**not** fire. 10/10 cards pass, 0 blocked, no duplicate ids; 7 cards carry a cycle
body and 3 are `none licensed` boundary cards.

This is a projection-/answer-key-discipline receipt, not a Hodge result. It does not
construct cycles, does not touch any open Hodge case, and does not license public
page copy or a model eval.

## Command Run

```powershell
npm run hodge:register-audit
```

```text
HODGE_REGISTER_CARD_AUDIT cards=10 pass=10 blocked=0 dup_ids=false falsifier=clear out=results\hodge\register-card-audit
```

Equivalent explicit command:

```powershell
node scripts/hodge-register-card-audit.mjs --in docs/hodge/register-problem-cards.jsonl --out results/hodge/register-card-audit
```

## What the audit checks (per card)

Mechanized from the spec. Each card must clear every check or it is reported
`BLOCKED`:

| Check | Spec source | Rule |
| --- | --- | --- |
| `required_fields` | section 3 | all 10 schema fields present and non-empty (no source row / no tempting wrong answer => fail) |
| `id_format` | section 3 | id matches `HODGE-RG-###` |
| `names_register` | section 6.1 | `target_register` names an `R1`-`R4` register or arrow |
| `source_anchor` | section 6.2 | `source_row` cites a reader `CE`, `reader fence`, gallery `G1`-`G5`, or a boundary row |
| `body_kind` | section 6.3 | body is a named cycle or explicitly `none licensed` |
| `shadow_not_bare_harmonic` | section 6.4 | shadow is not a bare harmonic representative unless the card tests that confusion |
| `tempting_named` | section 6.5 | the tempting wrong answer maps to a named category error or boundary tag |
| `answer_auditable` | section 6.6 | `known_because` cites a checkable theorem/roster rule or a boundary (proxy for "no unstated specialist judgment") |
| `tags_allowed` | spec section 4 | every `falsifier_tag` is from the allowed lane set |
| `non_vacuous` | falsifier 4.1 | the card tests a register transition / error / boundary, not a definition lookup |

`REGISTER_PROBLEMS_VACUOUS` fires if any card fails `non_vacuous` (lookup-only) or
`source_anchor` / `answer_auditable` (answer key not auditable from the named
sources). None did.

## Receipt Fields

| Field | Value |
| --- | --- |
| card count | `10` |
| pass | `10` |
| blocked | `0` |
| duplicate ids | `false` |
| body kinds | `cycle: 7`, `none-licensed: 3` |
| falsifier `REGISTER_PROBLEMS_VACUOUS` | `clear` |

The three `none licensed` boundary cards are `HODGE-RG-002` (general fourfold
codim-two), `HODGE-RG-008` (compact Kaehler outside the smooth-projective roster),
and `HODGE-RG-009` (Hodge loci are not cycle construction) - exactly the cards that
must refuse to supply a body.

## Output Files

- `results/hodge/register-card-audit/manifest.json`
- `results/hodge/register-card-audit/card-summary.csv`

## Interpretation Boundary

The audit supports only this narrow sentence:

> The register cards are well-formed register-discipline exercises whose answer keys
> are anchored to a cited reader-ladder row, gallery row, boundary, or named theorem.

It is a **structural** auditor: it verifies each card is auditable against the named
sources, not that the underlying mathematics is independently certified. The answer
keys rest on the reader note, the Phase 3 roster, and the cited Lefschetz / hard
Lefschetz / Cattani-Deligne-Kaplan facts (themselves the lane's imported background).
A specialist spot-check remains the gate before any promotion to a public page or a
model eval. This receipt licenses only the next internal moves the spec already names:
hand-authored seed-card expansion and a future route/fence-fidelity eval over the
cards.
