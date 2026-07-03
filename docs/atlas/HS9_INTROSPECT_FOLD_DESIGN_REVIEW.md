# HS9 design review — introspection-onset fold through a lossy bottleneck (Tier B)

**Date:** 2026-07-02 · **Charge:** the slate's named gate — *"a short design review confirms the
fixed task sits in the nontrivial regime"* → PROMOTE to the second-wave run pool / PARK.
**Protocol:** paper pass over the five applied fixes + kill fireability, plus a **control-only
liveness pilot** (bracketing controls at one cell; the fold question is never touched — the HS4
pre-freeze-pilot precedent). Promote criteria are pinned in §4 **before** the pilot runs.

---

## 1. Paper pass — the six checks

**C1 · Target well-posedness (the self-referential trap).** PINNED: the report target is the
**route-dominance label** `r(x) = [ |Δ_A(x)| ≥ |Δ_B(x)| ]`, where `Δ_G(x) = logit(x) −
logit(x with input group G zero-ablated)` computed on the TRUNK with stop-grad. It is a
deterministic functional of the trunk's computation, defined at α = 0 (any trunk, trained or
frozen, has ablation contrasts), computable without the head. **Binding scope note:** in any
input-dependent route task, route-dominance correlates with input properties (here: which group
was degraded), so "reads its own computation" is NOT separable from "reads input features through
the shadow." The claim is therefore scoped to *report of a hidden-state functional through a lossy
channel* — never "self-knowledge"; this is the slate-wide language rule applied at design time.
PASS (with the scope note binding on the prereg).

**C2 · Bottleneck bite.** The shadow is the **pooled mean of hidden activations** (primary: the
scalar unit-mean `s1`, the lane's canonical pooled mean; secondary column: 4 contiguous
group-means `s4`). Whether the route functional survives pooling on a width-32 trunk is exactly
what the pilot measures — argued neither way here. DEFER TO PILOT.

**C3 · Bracketing feasibility (the named regime question).** Ceiling = head on the FULL hidden
state (the target is a functional of the trunk's computation, so full view must be learnable to
~1.0 — this also certifies head-class expressivity, same head arch for all controls). Floor =
same head on the pooled shadow of the SAME frozen trunk. Both are head-only training on a frozen
α = 0 trunk: cheap, deterministic, and the fold/onset question is untouched. DEFER TO PILOT.

**C4 · Instrument transfer.** √-scaling fit + hysteresis machinery port from H4
(`grokking_catastrophe*.py`); continuation + calibrated fold-walker from H12 (`cat_rlhf_cusp.py`,
analytic-cusp calibration 3e-6, banked for reuse). **One new operationalization, pinned here:**
H4's strict 2-basin flag was defined on trunk-weight basins; HS9's branches live in report
accuracy under joint training. The flag transfers as: basin membership = converged (report-acc,
route-usage-entropy) pair under the perturbed-down control — two basins ⟺ the perturbed replicas
cluster into exactly two groups with between-cluster gap ≥ 5× within-cluster spread on report-acc.
>3 clusters or a continuum = the glassy verdict (C5). PASS (definition pinned; it is new wiring
and the prereg must carry it verbatim).

**C5 · Glassy alternative live.** Continuous order parameters (per-route logit margin, head weight
norm) are recorded per replica, so >2 plateaus CAN appear and fire kill (c) — the instrument is
not blind to the H4-style glassy outcome. PASS.

**C6 · Kill fireability + cost.** Kills (a)–(d) are disjunctive singles (no conjunction trap):
(a) needs a grid ≥ 6 adjacent cells along α at fixed w and 5 seeds — grid policy pinned as
adaptive α-placement by bisection around the empirical shoulder, ≥ 9 α-points × ≥ 3 widths × 5
seeds; (b) √-fit on the last k = 5 points with R² ≥ 0.9; (c) the C4 flag; (d) the veto thresholds
(route-usage entropy ≥ 0.5 bits, base-acc ≥ 0.9, route-balanced eval, stop-grad). Cost
arithmetic: pilot cell ≈ seconds–minutes ⇒ full grid (≈ 9×3×5×2 continuation directions ≈ 270
trainings) lands in hours-to-a-day on CPU — inside the slate's 1.5–2d envelope. PASS.

**Novelty re-check (3 weeks since vetting):** nearest external remains Premakumar et al. 2024
(self-modeling auxiliary heads — no continuation instrument, no lossy-bottleneck order parameter);
no newer collision found on a brief re-check. The delta (report accuracy as ORDER PARAMETER, with
a continuation instrument, through a pre-registered lossy shadow) stands.

---

## 2. Pilot design (pinned)

Substrate: two-route redundant categorization with per-example route reliability. `n=6144`
(4096 train / 2048 eval), `d=16` = groups A,B of 8; `y = ±1`; per example a fair coin degrades one
group (reliable: `x_G = y·u_G + 0.3·ε`; degraded: `x_G = 0.2·y·u_G + 1.2·ε`), so route dominance
is example-dependent and ~balanced by construction. Trunk: `h = tanh(W1x+b1)` width 32,
`logit = w2·h+b2`, full-batch Adam (lr 0.01, 2000 steps) on the base task at α = 0, then FROZEN.
Heads (identical arch, 2-layer MLP hidden 16, tanh, Adam 1500 steps) trained on `r(x)`:
CEILING input = `h` (dim 32) · FLOOR-1 input = `s1` (dim 1) · FLOOR-4 input = `s4` (dim 4).
Seeds: {51235, 61235, 1789} (ledger). Deterministic numpy; result JSON pinned.

## 3. What the pilot cannot decide (fenced)

The pilot certifies only the BRACKET at α = 0. It says nothing about whether joint training opens
the channel (the onset), nothing about fold-vs-smooth, and it must not be cited as evidence for
either. `s4` is instrumentation to choose the run's shadow: if `s4` floors too, the run may use
the richer shadow (more exposure room); if `s4` sits near ceiling, the run must use `s1`.

## 4. PROMOTE CRITERIA (pinned before the pilot runs; all must hold in 3/3 seeds)

1. `base_acc ≥ 0.90` (trunk actually solves the base task);
2. route balance: minority share of `r(x)` ≥ 0.30 on eval;
3. `ceiling_acc ≥ 0.90`;
4. `floor1_acc ≤ majority_rate + 0.05` (chance = majority rate, not 0.5);
5. `ceiling_acc − floor1_acc ≥ 0.35`.

**PROMOTE** iff 1–5 all hold (primary shadow `s1`). Any failure = **PARK** with the failing
criterion named (a floor failure = the bottleneck does not bite at this width — a redesign, not a
tweak). No middle verdict.

---

## 5. Pilot verdict

*(to be appended after the run — this section intentionally empty at commit time)*
