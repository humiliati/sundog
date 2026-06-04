# SUNDOG_V_JEPA — Focused Lit-Pass Memo

> 2026-06-04. Focused pass against `SUNDOG_V_JEPA.md` §8 (not the heavy deep-research
> workflow). Purpose: resolve feasibility + novelty + the de-confound wall **before** a
> Phase-0 spec, then re-pose the Phase-0 / `blocked` / `shelve` decision.

## §8.1 — JEPA mechanics + small-scale feasibility: **FEASIBLE WITH CARE**

I-JEPA (Assran et al. 2023, arXiv:2301.08243): context-encoder (ViT) on visible patches +
narrow predictor + **EMA/stop-grad target encoder**; predict masked target *embeddings*
from context. Collapse is avoided by the EMA target + architectural asymmetry "in most
settings" — **but small-scale is the catch:** secondary analyses report I-JEPA is
susceptible to total / dimension-wise collapse when regularization is omitted or configs
are poor; **EMA alone does not guarantee non-trivial solutions, and explicit
variance-covariance regularization (VICReg-style) is the documented fix.**

→ **Implication for Phase 0:** a faithful small JEPA on the toy backbone is *trainable*,
but the spec must include a **VICReg-style variance/covariance term + an explicit
collapse check** (per-dim embedding variance / rank). The `blocked_by_unfaithful_jepa`
branch then fires only if it collapses *even with* that regularizer — a real but
mitigable risk, not an open problem.

## §8.2 — Is "JEPA discards unpredictable detail" measured or designed? **DESIGN + DEEP-LINEAR THEORY, not a nonlinear de-confounded measurement**

The claim is well-established as a **design motivation** and now has **theory**: Apple's
*"How JEPA Avoids Noisy Features"* (arXiv:2407.03475) proves JEPA prioritises
**influential features** (high regression coefficient ρ — informative for prediction)
while MAE chases high-covariance features "even if noisy." **But the proof is on
deep-linear networks + synthetic Gaussian data + ODE simulations — explicitly *not*
measured on real data, and the authors flag their diagonalizable-covariance assumptions
deviate from realistic distributions.**

→ **This is the honest novelty read.** Our determining-shadow-set Phase 0 would be the
first **nonlinear (transformer), de-confounded, ground-truth-anchored *measurement*** of
the keep-functional / discard-state split (`k_func` vs `k_state`) against a generative
baseline — complementing the deep-linear theory and the qualitative RCDM picture below.
**It is not redundant. But its *outcome is theory-predicted*** (the implicit-bias theory
says JEPA should read cleaner than GEN), so Phase 0 is primarily **instrument-validation +
a nonlinear extension of known theory — calibration, not discovery.** A *null* (JEPA = GEN)
would be the surprising, flag-worthy result.

## §8.3 — De-confound prior art: **well-grounded; the real-data wall is field-acknowledged**

Our input-probe pre-check + `u_null` control are in the lineage of **Hewitt & Liang 2019
control tasks / selectivity** (arXiv:1909.03368): a probe must beat a randomized-label
control, or its accuracy is memorization, not representation content. Good — our
de-confound discipline has a citable pedigree. **Crucially, that literature also flags the
exact §7 wall:** control tasks are clean for word-level / designed targets and "less clear
how to apply more broadly." So porting the de-confound to **real** JEPA data is a genuine,
acknowledged open problem — vindicating the **toy-or-nothing** posture.

## §8.4 — Closest neighbour: **RCDM (qualitative); our read is quantitative**

RCDM (Bordes et al. 2021, arXiv:2112.09164) visualizes what SSL reps keep/discard by
decoding to pixels: SSL "overlooks low-level detail/background, keeps high-level parts,"
and MAE (pixel target) is pixel-useful but less high-level vs JEPA (embedding target).
That's the qualitative version of our thesis. **Our determining-shadow-set bracket is the
*quantitative, de-confounded, ground-truth* version** — distinct, but adjacent.

## Re-posed decision

| make-or-break | verdict |
| --- | --- |
| Faithful small JEPA trainable on the toy? | **Yes, with VICReg + collapse check** (else `blocked`). |
| Is the read novel? | **As a measurement, yes** (nonlinear, de-confounded, ground-truth) — **but theory-predicted**, so validation > discovery. |
| Does the de-confound port to real data (R2)? | **Not yet — a real open problem** (Hewitt-acknowledged). Toy result cannot license real-JEPA language. |

**Recommendation — `phase0_spec_as_validation_cell`, low priority.** The lane is feasible
and coherent, and Phase 0 is cheap (reuses the frozen toy + calibrated instrument; one
train + one probe). But the lit pass is honest that **Phase 0 confirms known theory
nonlinearly + validates the instrument on a JEPA body — it is not a discovery.** The
discovery lives at **R2 (real JEPA)**, which is **gated on the unsolved real-data
de-confound** (§7) — *that* is the lane's true research problem, not the toy.

So three live options for the owner:
1. **Spec the cheap validation cell** (Phase 0, with VICReg + collapse check + the frozen
   kill-gate) — bank the instrument-on-JEPA validation + a nonlinear test of the
   implicit-bias prediction. Modest, sound, an afternoon.
2. **Redirect to the real blocker** — open a *de-confound-on-real-data* design sub-lane
   (the §7 wall) as the highest-leverage JEPA work, since that's what gates any real claim.
3. **Park the lane** — the calibrated instrument is on the shelf; revisit if/when the
   real-data de-confound is cracked or a JEPA substrate becomes cheap to access.

Given the 2026-06-04 pivot (kill-gated R&D, product-first), my read is **(1) only if the
instrument-on-JEPA validation is independently wanted, else (3) park** — and treat the
**de-confound wall as the real prize** to chip at opportunistically, not a thrust.

---

*Sources:* I-JEPA [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) ·
"How JEPA Avoids Noisy Features" [arXiv:2407.03475](https://arxiv.org/html/2407.03475v1) ·
RCDM [arXiv:2112.09164](https://arxiv.org/abs/2112.09164) ·
Hewitt & Liang control tasks [arXiv:1909.03368](https://ar5iv.labs.arxiv.org/html/1909.03368) ·
Probing-classifiers survey [arXiv:2102.12452](https://arxiv.org/pdf/2102.12452)

*Sundog Research Lab — JEPA lit-pass memo, internal. Gates the Phase-0 decision; no run.*
