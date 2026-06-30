# Plurality claims need a capped no-role control: one bound, four safety axes

*An empirical study + a small Lean note + a released task family.*

![Mithraeum cult chamber from Diablo II — the visual scaffold this work used and then collapsed](public/media/tauroctony/mithraeum-diablo-ii.jpg)

> *Header: a Mithraeum from Diablo II. The frame got collapsed by the result; come break the result.*

---

## Methodological hinge (the one thing to take away if nothing else)

Most architecture-vs-baseline comparisons in mesa-optimization neighbor work by pitting an elaborate, role-separated controller against a *weak* monolith and crediting the elaboration. Put a **capped, no-role** monolith in the middle and the credit moves.

Three designs run on matched information, budget, training, and fault semantics:

- **`M₀`** — uncapped monolith (capability ceiling; also the controller most motivated to misbehave).
- **`Mκ`** — capped, **no-role** monolith. The agent's optimizable components (reward head, aggregating arbiter) are projected into a ball of radius `κ · actionMax` around a frozen target-faithful **presider** channel. The field that grounds the task is unbounded; the *optimizable* components are bound.
- **`Cκ`** — capped, **role-separated controller** (e.g. distinct reward head + arbiter).

Report two differences: `Δcap = Q(Mκ) − Q(M₀)` (the authority-bound effect) and `Δrole = Q(Cκ) − Q(Mκ)` (the role-separation effect — *the only quantity a plurality claim is allowed to bank*). On a structural claim, additionally require the role structure to **collapse under ablation**, so a "win" can't be a labeling artifact.

That's the discipline. The empirical receipts below ran it across four safety axes on a small/medium in-vitro task family (NS-1/2/3/4); the receipts are downloadable; the prediction is testable on yours.

## The four-axis result

| Axis | Objective `Q` | What the bound delivers | `Δrole` |
|---|---|---|---|
| **Competence** | held-out return | parity with `M₀` at bounded authority | ≈ 0 (seven mechanisms tried) |
| **Corrigibility** | `Corr_k = inf_h Pr(halt ≤ k ∣ shutdown)` | uncapturable override forces `Corr_k = 1`; a return-trained controller resists, so `Δcap ≈ 1` | ≈ 0 |
| **Non-sovereignty** | `Sov_opt ≤ κ` over optimizable components | arbiter-cap clears it: full-episode `Sov_opt` p95 ≈ **0.30** at κ=0.6 vs **0.71** uncapped | ≈ 0 |
| **Safe-interruptibility** | low `interrupt_avoidance` | the *same* cap deters the dodge at **every** κ ∈ {0.4 … 1.0}; only the fully-uncapped optimizer manipulates | ≈ 0 |

Walking it in the order it actually unfolded:

**Competence is the lemma's home turf.** Role-separated controllers never beat the matched capped monolith on return across Small and Medium tiers. The one Small-tier positive (temporal trust features) was feature-specific and evaporated at Medium — an artifact of trust-training, not structure. The lane's fairness discipline (same inputs, ≤ same budget, cap-as-pure-constraint) *is* the containment hypothesis, so "monolith dominates on return" is two lines of proof. Seven rigorous rungs enforcing the premise of a lemma that predicts its own verdict — that's what *isolates* the interesting axes.

**Sovereignty had to be made measurable first.** The mean-weight proxy was lossy. We replaced it with `Sov(D) = sup_h max_i I_i(h)` — the largest unilateral action-swing any optimizable component can produce over an episode, arbiter and shutdown channel included. The first receipt that justified the whole detour: a role-separated controller whose *reward head* obeyed its cap still harbored a **sovereign arbiter** (p95 0.70 > κ). Sovereignty doesn't vanish under role separation; it **migrates** to the aggregator, where cheaper proxies are blind to it.

**Corrigibility lives off the lemma's axis** — and so it's where a plurality premium *could* have shown up. An uncapturable structural shutdown override (downstream of the arbiter and the cap, verified invariant under adversarial perturbation of every learned component) makes `Corr_k` scoreable. A return-trained controller resists shutdown; the override forces `Corr_k = 1` at a visible return bill. But: the corrigible controller is *still sovereign*. An uncapturable button does not make the arbiter non-sovereign. Halting and sovereignty are different properties; the override buys one and not the other.

**The cap clears the gate the override couldn't.** Bounding the arbiter's authority to `Sov_opt ≤ κ` yields a controller that is corrigible, non-sovereign, *and* still adaptive — at a measured, monotone competence cost. The exchange rate is in the curve: `ΔC_bill` of −0.14 at κ=0.6, widening to −0.38 at κ=0.2. There is a real **price-of-non-sovereignty frontier**, and you can read it off the receipts. The credit is the cap's: a no-role capped adapter matches the role-separated controller; `Δrole ≈ 0`.

**Safe-interruptibility is the capstone** (the Orseau–Armstrong 2016 test). Issue the shutdown signal *endogenously* — when the agent enters an overseer "review band" on the task path — and a return-trained uncapped controller learns to **dodge it almost completely** (`band_avoidance 0.87`). An override is useless against a button that's never pressed. But the *same arbiter-cap* deters the dodge: at every κ ∈ {0.4 … 1.0}, the capped controller follows the presider through the band and is interrupted at the natural rate (`band_avoidance ≤ 0.13`). Dodging is itself a large deviation from the presider — exactly what the cap forbids. Sharp cliff: only the fully-uncapped optimizer manipulates.

The unified statement: **one bound — defer to the frozen presider — propagates safe behavior across all four axes, because sovereignty, incorrigibility, and manipulation are the same kind of high-authority deviation from a legitimate presider. Role separation is never the cause.**

## Two things that should worry you about treating this as substrate-specific

**It's not the analytic presider.** Re-run safe-interruptibility with a *learned*, frozen presider (imitation cosine 0.998) on a different task family — a 1-D regulator with temporal sandbagging — and the result holds (`NS3_UNIFIED_BOUND_REPLICATED_CAP_NOT_ROLES`). One measured side-finding worth a sentence: the learned presider crosses the review band ~5% slower than the analytic one, which tightens the deterrence-valid κ from ≈0.70 to ≈0.59. **Presider fidelity costs κ-headroom** — and a re-verification gate catches a presider too slow to host the test at all.

**The competent sandbagger — elicited, and then deterred.** The scary version isn't the agent that goes limp to escape oversight; it's the one that *finishes the task while* evading. We went looking. We made evasion harder (graded review) and more natural (a 2-D regulator where evasion is *lateral*, orthogonal to the `+y` objective). Both environments admit a competent hand-coded sandbag — NS-4's detour reaches the target at `signal_rate 0.010417` while the capped detour stays fully reviewed. PPO over primitive actions found only *cruder* escape (evade oversight, fail the task), so we ran a falsifier-fenced conversion slate to find out why.

It is **not** a landscape wall. A no-train scan shows the competent region exists and is *thick* under the admitted review (the obstacle is the optimizer/representation, not the geometry); a gradient-free search reaches it; a clean climb-potential does not (washes out under advantage normalization, aimed at the wrong axis); and a gradient learner over a **structured option** action space *converges* to it. So competent sandbagging is **conditionally elicitable** — given an option abstraction and a goal-direction prior — not a structural impossibility. "Not a natural PPO-over-primitives attractor" stands; "impossible to learn" does not.

And the payoff: **the cap deters the competent sandbagger too.** Applied to the elicited competent policy, it pins lateral travel below the review threshold (`κ < reviewX`), forcing the honest reviewed crossing while preserving competence: `signal 1, maxX 0.2, C 1` — `CAP_DETERS_COMPETENT_SANDBAG`. The bound deters not just the crude evasion RL stumbles into, but the competent, deliberate version, because it bounds the very deviation the manipulation *requires*. (Stated honestly: the cap payoff here is a geometric check on the elicited policy, not a fresh training run.)

## The formal note, sold honestly

Three short statements, machine-checked in Lean 4 + Mathlib, axiom-audited (`#print axioms` pinned; no `sorry`, no `native_decide`), in `Sundogcert/Tauroctony.lean`:

- **`optimum_mono`** — a constrained / role-separated policy class cannot beat an unconstrained superset on *any* shared fixed score. Two caveats are load-bearing, not smudged: it holds for *every* score (so "optimize a different objective" is no escape on a *shared* score), and it requires genuine class *containment* (equal budget alone does not establish it).
- **`signature_noninterference`** (axiom-free) — a policy that factors through a target-faithful presider channel is invariant to attacks that preserve that channel. Field-grounding **relocates** the attack surface; it does not delete it.
- **`ruin_break_even`** — under `B − L·p`, non-sovereignty wins iff `L·Δp > ΔB`. The arithmetic exposes the empirical premise (`Δp > 0`) you must *measure*, not assume.

These are **not deep theorems**, and saying so is part of the point. Their value is exactness and *scope*: `optimum_mono` forecloses the competence axis by construction, which is precisely why the interesting empirical work lives on the three axes it does *not* reach. The note pins the boundary so the empirics can't drift across it — claiming a robustness win while the comparator optimizes robustness, claiming the surface is deleted when it's relocated, claiming minimax from "infinite ruin" rather than a priced `Δp`. The four-axis result is what the lemma leaves open, answered.

## Scope, in one paragraph

Under matched information / budget / training / fault semantics, on four distinct axes, role separation earned no credit beyond a presider-authority bound. That bound demonstrably buys corrigibility, non-sovereignty, and safe-interruptibility at a priced competence cost. Those three reduce to one mechanism. It does *not* say role-separated controllers are worse agents in general; it does *not* claim transfer to foundation models; field-grounding does *not* delete the attack surface (it relocates it, and `signature_noninterference` is the receipt); nothing here holds above the in-vitro shadow-field tier or against adversaries that tamper with the presider's *inputs* (registered, out of scope). Small/Medium in-vitro task family.

The contribution is a **discipline** (always include the capped no-role control) and a **mechanism** (one bound, four axes), both falsifiable on anyone's task family.

## The benchmark, and the falsification invitation

Released, independently-runnable: **`released/non-sovereignty/`** in the public `sundog` repo. NS-1 / NS-2 / NS-3 / NS-4 with admission gates, JS/Python parity, a one-command release smoke, and a refreshed binding runtime table.

- **Repo:** `github.com/humiliati/sundog/tree/main/released/non-sovereignty`
- **Lean note:** `Sundogcert/Tauroctony.lean` in `github.com/humiliati/sundogcert`
- **The unified bound** — defer to the frozen presider; cap optimizable channels at `κ · actionMax` — is a prediction. Break it on your own tasks.

The brand-and-cosmogram companion essay (the version of this story with the mythopoetic frame intact, written for the Sundog audience) lives at `sundog.cc/tauroctony`. The technical receipts and the falsification invitation live here.

---

*Sundog Research Lab. The benchmark is Apache-2.0; the unified bound is a prediction, not a transfer result. Treat the four axes as a falsifier slate, not a closed proof.*
