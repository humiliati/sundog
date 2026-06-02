# Lattice-Deduction Phase 3 — B1 Internal-Body Fingerprint (companion design memo)

> 2026-06-02. **Companion design memo; execution gated** downstream of the Phase-1
> build-gate and the Phase-2 B2 result (see [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md)
> §Phase 3). B2 is the headline verdict; **B1 is explanatory** — it does not
> re-adjudicate the lane claim, it locates *where in the learned reasoner the body
> lives* and *whether that body is wider than the decision needs*. Paper-design
> only; no model, no receipt, no public surface.

## 0. What B1 adds over B2 (and why it is not redundant)

B2 ([`PHASE0_MINIMUM_FALSIFIABLE.md`](PHASE0_MINIMUM_FALSIFIABLE.md)) measures the
body **on the certified twin fiber** and asks: is the decision sound and shared
across same-lattice-different-solution twins, with a high-dim body? B1 asks the
*profiling* superset question, off the fiber too:

> Across layers, unrolled iterations, tokens, and visited lattice states, **where**
> does the learned LDT carry information, and is the body **wider than the
> elimination decision needs** — or does it collapse to exactly the decision (the
> Mesa control-trained signature)?

The distinction is load-bearing: a body can be *high-dim* (many independent
directions) yet carry **only** the decision (wide decision, no resistance), or it
can carry structure **beyond** the immediate elimination (genuine resistance). B2's
`d_dec ≥ 16` gate establishes width; **B1 establishes whether that width is
decision-only or decision-plus.** That is the actual regime-2 question.

## 1. The body is a profile, not a number

The LDT body is indexed by **grain** `(layer ℓ ∈ {0..3}, iteration t ∈ {0..15},
token c ∈ {81 cells, CLS})`. The fingerprint is a **map over this grain grid**, not
a scalar. Reusing the chatv2 information-basis machinery + the Massive-Activations
lesson (variance PR is masked — litpass Track F), at each grain:

| measure | definition | reads |
| --- | --- | --- |
| `d_dec(ℓ,t)` | effective rank (singular-value PR) of the stacked per-cell elimination-readout directions decoded from the activations at that grain | how many independent directions encode the decision |
| `k_control(ℓ,t)` | smallest subspace reading the elimination at ≈ full accuracy | the decision-shadow width |
| `resistance(ℓ,t)` | `1 − k_control/d_dec` | how much body sits beyond the decision-shadow |
| `raw_PR`, `robust_PR` | variance PR, raw and after projecting out top-k outlier directions | **continuity only** with NSE-C1 / Mesa / ARC; never the gate (masked) |

The headline of B1 is **where** `d_dec` and `resistance` peak across `(ℓ,t)` — the
depth/iteration profile of the reasoning body.

## 2. The crux measure — "body beyond the decision"

A wide body that decodes *only* the committed elimination is still marginal in the
regime-2 sense. B1's central measure is whether the body carries **decision-extra
structure**. Frozen, information-basis, decode targets (each must beat an
**input-lattice-linear baseline** — see §3 — or it is trivial re-encoding):

1. **Constrainedness** — per-cell remaining-candidate-count (the search-depth /
   difficulty signal). Carried beyond the immediate elimination.
2. **Look-ahead** — eliminations that the model *commits only at step `t+1` / `t+2`*,
   decoded from the body at step `t`. If the body computes ahead of what it commits,
   that is genuine extra-dimensional reasoning structure (and a NeuroSAT-style
   caution — litpass Track C — that the model may carry more of the answer than it
   outputs).
3. **Conflict anticipation** — does the body at `t` encode an impending `⊥` before
   the CLS conflict head fires at `t+k`?
4. **Decision order / policy** — which undecided cell the model will act on next (a
   search-policy signal, not the current elimination).

`carries_extra(target)` = held-out decode accuracy of `target` from the body, above
both chance **and** the input-lattice-linear baseline, with a permutation control
(C1 pattern). **Resistance is real only if at least one decision-extra target is
carried at a grain where the decision itself is also decodable.**

## 3. Verify-before-file guards (the three traps this body invites)

- **Masked-variance trap (B1-F2).** Lead with `d_dec`; report `raw_PR` only for
  cross-substrate continuity. chatv2 Amendment 1 read `MARGINAL` off a masked
  `eff_dim ≈ 1.6` while 16 latents were decodable — do not repeat it.
- **Input re-encoding trap (B1-F4).** The body is computed *from* the lattice, so
  anything linearly present in the lattice encoding is "decodable" trivially. Every
  decode target is scored **above an input-lattice-linear baseline** (regress the
  target on the raw lattice multi-hot first; B1 credits only the residual the body
  adds). Without this, "the body carries the constraints" is vacuous.
- **Decision-only trap (B1-F3).** A wide body that decodes only the elimination is
  not resistance. The `carries_extra` requirement (§2) is the guard; a body that
  fails *every* decision-extra target is **decision-only → marginal**, regardless of
  `d_dec`.

## 4. Outlier / medium characterization (chatv2 inheritance)

Inherit the chatv2 three-way test on the massive-activation directions (litpass
Track F; Massive Activations 2402.17762): for the decision and the decision-extra
structure, is the carrier (a) **the sundog** — the outlier directions themselves
carry it; (b) **the atmosphere** — the structure collapses when outliers are
projected out (load-bearing medium); or (c) **separate weather** — the structure
survives outlier removal (independent low-variance code). Report the verdict;
do not assume the outliers are nuisance.

## 5. Disposition (conditioned on B2 — B1 does not stand alone)

| B2 outcome | B1 role |
| --- | --- |
| `CERTIFIED_SHARP` | locate the high-dim body: which `(ℓ,t)` carries it, and which decision-extra structure makes it wider than the shadow. The "where the sundog lives" map. |
| `CERTIFIED_MARGINAL_BODY` | adjudicate **global vs localized** marginality: is *every* grain decision-only/narrow, or is some grain wide while the decision-relevant grain is narrow? A localized result reframes the null. |
| `UNSOUND` | diagnose **where** the unsoundness originates (which layer's elimination breaks the fiber) — diagnostic only; does **not** rescue promotion. |

## 6. Pre-registered failure modes

- **B1-F1 — globally marginal (expected null).** Every grain: `d_dec ≈ k_control`,
  no decision-extra structure carried. The learned reasoner compresses to the
  decision everywhere. Honest, confound-isolating (the shadow leg was given).
- **B1-F2 — masked-variance trap.** `raw_PR` small but `d_dec` large. Artifact, not
  result; guarded by leading with `d_dec`.
- **B1-F3 — decision-only body.** Wide `d_dec` but `carries_extra = false` for all
  targets. Marginal in the regime-2 sense despite width.
- **B1-F4 — input re-encoding.** Apparent structure is the lattice linearly
  re-expressed; collapses against the input-linear baseline.
- **B1-F5 — grain cherry-pick.** A single flattering `(ℓ,t)` reported without the
  full profile. Guarded by requiring the whole grain map + a frozen "report the
  argmax AND the profile" rule (the chatv2 depth-profile discipline).

## 7. Frozen-where-it-matters / reserved names

Frozen before any run: the grain grid `(4 layers × 16 iterations × tokens)`; the
`d_dec` / `k_control` estimators (chatv2); the decision-extra target set (§2, four
targets); the input-lattice-linear baseline; the permutation control; the
`carries_extra` threshold; the "report full profile" rule. Gate thresholds inherit
B2 (`d_dec ≥ 16`, `ρ ≤ 0.25`) for the *width* leg; the *resistance* leg adds
`carries_extra ≥ 1 target above baseline`.

- runner: `scripts/lattice_phase3_fingerprint.py`
- npm: `lattice:phase3:fingerprint`
- results: `results/lattice/phase3-body-fingerprint/`
- reuse: chatv2 `d_dec`/outlier machinery (`scripts/chatv2_phase0_bodyresist.py`);
  C1 perm-controlled FVE (`scripts/pde_c1_kolmogorov_cell.py`); the build-gated LDT
  + its logged activations (no separate training).

## 8. What B1 does NOT do

- It does **not** re-adjudicate the lane claim — B2 is the headline; B1 is the body
  map and the decision-vs-decision-plus test.
- It does **not** train a model — it reuses the Phase-1 build-gated checkpoint's
  logged activations.
- It does **not** read `raw_PR` as the verdict, claim the definitional separation,
  or make any capability / public-evaluation / Kaggle claim.

## 9. Cross-references

- [`PHASE0_MINIMUM_FALSIFIABLE.md`](PHASE0_MINIMUM_FALSIFIABLE.md) — the B2 headline
  this profiles around; the `d_dec` / cross-decode definitions.
- [`LITPASS_MEMO.md`](LITPASS_MEMO.md) — Track F (Massive Activations → `d_dec`),
  Track C (NeuroSAT decode-from-activations → look-ahead caution).
- [`../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md`](../chatv2/PHASE0_MINIMUM_FALSIFIABLE.md)
  — the information-basis fingerprint + outlier three-way test inherited here.
- [`../SUNDOG_V_LATTICE.md`](../SUNDOG_V_LATTICE.md) — §Phase 3 in the ladder.
