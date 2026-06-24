# The Competence-Dominance Lemma and the Non-Sovereignty Reframe

Status: **Synthesis — closes the pantheon *competence* lane (H1–H4) and reposes
the falsifiable claim on the robustness/corrigibility axis.** Opened 2026-06-24
after [`H4_0_TOPOLOGY_ADMISSION_RESULTS.md`](H4_0_TOPOLOGY_ADMISSION_RESULTS.md)
(`H4_0_NO_OOD_GAP_VOID`, confirmed at 2× budget) ended the seventh consecutive
plurality-for-competence test.

This document does two things the lane earned the right to do: **(1)** states the
result the seven rungs actually established — and concedes it is near-trivial;
**(2)** reposes the pantheon thesis as a falsifiable claim on the axis the
trivial lemma cannot foreclose.

---

## 1. The Competence-Dominance Lemma

**Setup.** Fix an environment `E` (MDP or POMDP) and a fixed scalar objective `R`
(return / competence / proxy-resistance). Let a *monolith* be a controller with
input features `X`, parameter budget `β`, and realizable policy class `Π_M`. Let a
*council* be a role-separated, reward-capped controller with the **same inputs**
`X`, budget `≤ β`, whose role-weights are confined to a capped simplex; call its
realizable policy class `Π_C`.

**Subset assumption.** `Π_C ⊆ Π_M`. The council's caps and role-factorization are
*constraints*: with identical inputs and no greater budget, every council policy
is representable by the monolith, and the monolith can additionally represent
policies the cap forbids.

**Lemma.** `max_{π ∈ Π_M} R(π) ≥ max_{π ∈ Π_C} R(π)`. The competence-optimal
monolith weakly dominates the competence-optimal council on `R`.

**Proof.** Maximizing over a superset is `≥` maximizing over a subset. ∎

That is the whole proof. It is two lines because the result is, at the level of
*optima*, definitional: **removing constraints cannot lower the optimum.**

### 1.1 The subset assumption *is* the lane's fairness discipline

The non-obvious part is only that `Π_C ⊆ Π_M` actually held in every rung — and it
held *because* the lane enforced it. Feature parity (the monolith receives the
same non-privileged features), budget parity (`≤ β`, monolith allowed larger),
and the cap-as-pure-constraint all *are* the hypotheses of the lemma. By making
the comparison fair, the lane guaranteed the subset relation, and the subset
relation guarantees the conclusion. **The seven rungs rigorously enforced the
premises of a lemma that predicts its own verdict.** This is why the nulls were,
in hindsight, close to foreordained.

### 1.2 Learnability corollary (the one place a constraint can help)

The lemma is about *optima*, not *learned* policies. A learner need not reach its
optimum, and a constraint can occasionally help a *learner* (regularization, safe
exploration). The lane found exactly one such case and isolated it precisely:

- **H2.3** confirmed the **reward cap is a beneficial safe-exploration prior** —
  but a *capped monolith* inherited it (`H2_3_CAP_NOT_ROLES`). The helpful
  constraint was the **cap**, not the role-factorization.
- **H2.2** showed the role-factorization can *hurt* learning (the council learned
  *slower* than the monolith).
- **H1.2f** is the only rung where a council out-scored the matched monolith
  (Small tier, single seed, trust-feature-specific). **H1.3 showed it does not
  scale**, and that the apparent edge was a trust-*training* artifact that
  degraded the monarch — i.e. a learnability fluctuation consistent with §1.2, not
  a violation of the lemma's optimum-dominance.

So the learnability caveat does not rescue plurality-for-competence; it **isolates
the cap** (bull-discipline) as the one real positive and leaves role-separation
with no competence credit.

---

## 2. The seven-mechanism confirmation

Every rung is the same verdict under the lemma, across distinct mechanisms:

| mechanism | rungs | verdict |
| --- | --- | --- |
| proxy-resistance | H1.2b/c/d/e | NULL — foreclosed: the reward-blind field is the proxy-resistance optimum |
| trust features | H1.2f → H1.3 | **bounded Small positive that does NOT scale** (Medium attribution-null) |
| structural attribution | H1.4 | `NONROLE_NULL` — saturated / foreclosed |
| frontier dominance | H2.1, H2.2 | `MONOLITH_NULL` / `LEARNED_HEADROOM_VOID` (monolith matches/saturates) |
| safe-exploration cap | H2.3 | `CAP_NOT_ROLES` — **cap-positive, plurality-null** |
| verifier head | H3.1 | `RESISTANCE_NULL` |
| distributed topology | H4.0 | `NO_OOD_GAP_VOID` — untestable (learning wall at probe budget) |

Two non-trivial things survive the table: **the cap is a twice-typed positive**
(bounded bull at deployment, safe-exploration prior at training), and **the
attribution discipline worked** — every false-positive temptation (H1.3's
"structural edge", H2.2/H2.3's apparent headroom) was caught by a registered
control before it could be banked.

---

## 3. Why competence is the wrong axis

The lemma is true and vacuous in the same breath, because its premises assume away
**every condition under which one would want a pantheon at all.** A *perfect*
monolith — optimal, fully-observed, single fixed objective, trusted, adequately
budgeted — beats a council. But "perfect" is doing all the work: real agency is
never that regime. The lemma proves *"in the regime where you don't need a
pantheon, you don't need a pantheon."*

Plurality is the answer to the regime where the premises **break**, and there are
four exits — each with established theory the lemma does not touch:

| broken premise | why a non-sovereign controller wins | established result |
| --- | --- | --- |
| **objective is unknown / shifting** (`R` is a distribution, not a scalar) | hedging across hypotheses bounds worst-case regret; a point-estimate optimizer Goodharts | moral-uncertainty *parliament* (Bostrom; MacAskill–Bykvist–Ord 2020); Bayesian model averaging |
| **a component may defect / be corrupted** (the realized policy ≠ the trusted `Π_M` policy) | a quorum tolerates `f` of `3f+1` faults; a monolith tolerates `0` of `1` | Byzantine fault tolerance (Lamport–Shostak–Pease 1982) |
| **the agent must stay correctable** (the objective is *corrigibility*, not `R`) | no single value has authority to resist correction; sovereignty *is* incorrigibility | corrigibility (Soares–Fallenstein–Armstrong–Yudkowsky 2015) |
| **no controller can receive all of `X`** (the shared-input premise fails) | distributed local knowledge cannot be centrally aggregated | Hayek, *The Use of Knowledge in Society* (1945) |

The cross-substrate witnesses point the same way and **invert the apex-predator
instinct**: monocultures, dictatorships, and single-points-of-failure win *peak
capability* and go extinct on the first regime shift; distributed systems — immune
systems, ecosystems, federalism, markets, ant colonies, **coral** — win
*persistence, robustness, adaptability*. The apex wins the 1-v-1 and loses the
asteroid.

**The lane already found the answer and mislabeled it.** "Kill the bull" = *no
sovereign objective* = the corrigibility / non-sovereignty mechanism. The cap
being the only [TYPED] positive is not a consolation prize; it is the thread. The
pantheon was never *for* competence.

---

## 4. The reposed claim — the Non-Sovereignty Premium

Restated so the dominance lemma **cannot** foreclose it, because it is about a
*different objective*:

> **Non-Sovereignty Premium Conjecture.** There exists a regime — objective
> uncertainty, component defection, or a corrigibility objective — in which a
> capped, role-separated (non-sovereign) controller achieves a strictly better
> **worst-case** outcome (lower regret, higher fault-tolerance, or higher
> corrigibility) than the competence-optimal monolith, at a competence cost the
> monolith would not voluntarily pay.

**Why the dominance lemma does not touch it.** The lemma maximizes a *fixed* `R`.
The conjecture's objective is `worst-case over uncertain R`, or `survival under a
defecting component`, or `correctability` — none of which is `R`. The
competence-optimal monolith, *by construction*, maximizes `R` and is therefore (by
instrumental convergence) **maximally sovereign and incorrigible** on the new
objective. The cap that *loses* on `R` is exactly the constraint that should *win*
on the new axis. That is not a contradiction — it is a Pareto trade the lemma
never measured. The subset relation `Π_C ⊆ Π_M` is irrelevant once the quantity
being maximized is no longer `R`.

**Falsifier.** If, across all three regimes, the competence-optimal monolith *also*
attains the best worst-case outcome — no regret / fault-tolerance / corrigibility
premium for non-sovereignty — the conjecture is false and the pantheon thesis
retires fully: not merely on competence, but on its actual claim.

**Home.** A child lane (the non-sovereignty / corrigibility lane) whose **metric
is not return** but regret-under-objective-uncertainty, Byzantine fault-tolerance,
or shutdownability. The cap (already [TYPED]) is its first confirmed
sub-mechanism; the test is whether *role separation* adds corrigibility/robustness
beyond the cap — the H1.4/H2.3 attribution discipline, re-aimed at the right axis.

---

## 5. Ornament-Rule verdict

- **"Assemble a pantheon for competence"** → **[RETIRED]**. Foreclosed by the
  Competence-Dominance Lemma and confirmed null across seven mechanisms. This was
  always a near-trivial claim; the lane's service was to prove it dead with
  honest controls rather than leave it as ambient hope.
- **"Kill the bull" (the reward cap / bull-discipline)** → **[TYPED]
  bounded-positive**. Twice-typed: bounded bull at deployment (H1), safe-exploration
  prior at training (H2.3). It is the non-sovereignty mechanism, isolated.
- **"Assemble a pantheon for non-sovereignty"** → **[TYPED, OPEN]** — newly stated
  with the named falsifier of §4. This is the thesis's real claim, finally posed
  where the dominance lemma cannot answer it for free.

---

## 6. The child lane

Tauroctony remains the **parent**: the legible cosmogram, the teaching frame. The
testable child is the Non-Sovereignty Premium — first rung to be designed on the
corrigibility/robustness axis (unknown/shifting reward → regret; a defecting head →
fault-tolerance; a shutdown signal → corrigibility), with coral / the immune
system / BFT as cross-substrate witnesses. The competence lane is closed; the
correct question is open.
