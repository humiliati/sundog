# H-K5 - The Hodge/Kakeya Shadow-Collision Table

- Artifact id: `HK5-SHADOW-COLLISION-TABLE`
- Date: 2026-06-29
- Status: internal synthesis note (report-only). **Falsifier `COLLISION_TABLE_ONLY_RHYME` CLEAR** (machine-checked).
- Slate: [`HODGE_KAKEYA_HYPOTHESES_SLATE.md`](HODGE_KAKEYA_HYPOTHESES_SLATE.md)
- Builder: [`../scripts/hodge-kakeya-collision-table.mjs`](../scripts/hodge-kakeya-collision-table.mjs)
- Output: [`../results/synthesis/shadow-collision-table/collision-table.json`](../results/synthesis/shadow-collision-table/collision-table.json)

This note follows the slate's discipline: it was written **after** concrete receipts landed
(H-K1, H-K3, H-K4), and a builder script populates the table from those committed receipts so
the synthesis is anchored, not asserted.

## The table (filled from receipts)

| Lane | Shadow | Body | Lossiness measure (from receipts) | Restoring structure | determine / resist |
| --- | --- | --- | --- | --- | --- |
| **Kakeya** | direction-coverage signature | point/tube set in `F_q^2` | **finite count of bodies per signature, exactly `q(q^2-q+1)`** (`q5=105, q7=301, q11=1221`) | Dvir polynomial method - complete shadow forces size `>= q(q+1)/2` | **finite resist**: size determined, exact body not |
| **Hodge** | rational `(p,p)` class | algebraic cycle | **decoder overclaim rate** (shadow does not determine existence): unprompted models overclaim `0.2` of cards; judge-consensus route-correct `0-3/10` | Lefschetz `(1,1)` / hard Lefschetz / Cattani-Deligne-Kaplan in special cases; general case open (the Hodge conjecture) | **infinite resist**: existence undecidable by the shadow (`sigma = infinity`) |

## Why this is not only a rhyme (the falsifier check)

`COLLISION_TABLE_ONLY_RHYME` fires unless both rows share an **operational field beyond
vocabulary**. The builder requires each lane to populate, *from its committed receipts*, the
five operational slots: `{shadow, body, numeric lossiness measure, control, restoring
structure}`. Both do (`operational=true` for both rows), so:

```text
HK5_COLLISION_TABLE shared_operational_field=true typed_mismatch=true falsifier=clear
```

The shared operational field is concrete, not metaphorical:

1. **Both have a numeric, control-backed lossiness measurement.** Kakeya: the exact count
   `q(q^2-q+1)`, controlled against random same-size bodies (H-K4 showed the count is a
   density artifact unless taken as excess over that control). Hodge: an overclaim rate
   measured by a decoder, controlled by two independent semantic judges (H-K1 PHASE4F:
   lexical-vs-judge agreement `0.60`, inter-judge `0.80`).
2. **Both sit on one axis** - the portfolio's determine/resist (sufficient-statistic-order)
   law: a lossy shadow, plus a *named structure that restores the body*. Kakeya restores
   **size** (Dvir floor); Hodge restores **existence** (Lefschetz/CDK, in special cases).
3. **The same falsify-first apparatus already ran in both lanes** - card/audit/control
   receipts (Hodge: generator -> audit -> fidelity -> model -> judge; Kakeya: enumeration ->
   structured family -> q-scaling -> adaptive-fibering control). That the same instrument
   produces receipts in both is itself the operational, non-vocabulary link.

## The honest typed mismatch (not an identity)

The lossiness measures are **typed differently**: Kakeya's is a finite **count** (enumerable
bodies per signature); Hodge's is a **rate** (existence-undecidability, surfaced as a
decoder's overclaim frequency). They share the *frame* (lossy shadow) and the *measurement
idea* (a shadow-only decoder cannot invert to the body) but not the *units*. So the table is
a **calibrated analogy with a typed mismatch**, not an isomorphism. Stated plainly: Kakeya is
finite resist (the shadow pins size, leaving a finite body-count ambiguity); Hodge is infinite
resist (the shadow does not even decide existence). That difference is the content, not a flaw.

## What H-K4 contributes to the table

H-K4 (the adaptive-fibering null) earns a standing entry: **a collision/lossiness number is
only meaningful against a control.** The raw Kakeya gap was a finite-grid density artifact
until measured as excess over a random same-size baseline; the raw Hodge lexical route check
over-credited until checked against semantic judges. "Control the artifact before reporting a
collision number" is now a shared discipline both lanes enforce - the table's methodological row.

## Interpretation Boundary

The table does **not** identify the theorems, does not touch Euclidean Kakeya or resolve the
Hodge conjecture, and licenses no public claim. It identifies the **workbench move**: find the
shadow, measure its lossiness against a control, name the structure that restores body/size/
existence, and place the lane on the determine/resist axis.

## Generative payoff (next probes the table spawns)

1. **Cross-lane confirmation - DONE 2026-06-29, CONFIRMED**
   ([`kakeya/PHASE5_REGISTER_TRANSFER.md`](kakeya/PHASE5_REGISTER_TRANSFER.md)): the *same*
   card -> model -> two-judge scripts ran on 8 Kakeya register cards (shadow = direction
   signature; trap = "this signature forces a unique/extremal/minimal body"). The instrument
   transferred - inter-judge agreement `0.833` (Hodge `0.80`) - and the same shadow-non-
   invertibility overclaim appeared (hardest traps KAK-RG-003 and KAK-RG-006, both 6/6
   overclaim), with the same model ordering (groq llama-70b resists best). The shared
   operational field is confirmed by transfer, not just by schema.
2. **New-lane recipe:** for any future Sundog lane, populate the same five slots; a lane that
   cannot is not ready for a shadow->body workbench.
3. **H-K6:** the boundary cards from both rosters become adversarial evaluator prompts.
