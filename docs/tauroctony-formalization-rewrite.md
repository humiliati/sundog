# Tauroctony lane: formal repair

## 1. Objects that must remain distinct

Let:

- `E` be an environment or POMDP.
- `Π_D` be the nominal policy class realizable by controller design `D`.
- `L_D(r, data)` be the policy selected by training design `D` when trained on proxy/objective `r`.
- `T_θ(π)` be true value under possible world/objective `θ ∈ Θ`.
- `B(D)` be nominal competence of the deployed, selected policy.
- `p⁺(D)` be an upper or worst-case probability of ruin.
- `Corr(D)` be corrigibility/shutdown compliance.
- `Sov(D)` be maximum unilateral causal authority.

Policy expressivity, training/selection, fault semantics, and authority are different objects. A proof about `Π_D` does not by itself prove anything about `L_D`, component corruption, shutdownability, or ruin.

## 2. Competence-dominance lemma

For any score `J : Policy → ℝ`, if `Π_C ⊆ Π_M` and both classes contain an attained optimum, then

\[
\max_{\pi\in\Pi_C}J(\pi)\le \max_{\pi\in\Pi_M}J(\pi).
\]

This is valid, but two qualifications are load-bearing:

1. Equal inputs and equal parameter counts do **not automatically prove** `Π_C ⊆ Π_M`. Either define the monolith extensionally as a class that contains the council, or give a constructive simulation with the stated budget.
2. The lemma applies to **every common score**, not only competence. Replacing return `R` with robustness, corrigibility, CVaR, or worst-case utility does not escape it if the monolith may optimize that new score over the same superset.

Pseudo-Lean:

```lean
universe u

variable {Policy : Type u}
variable {C M : Set Policy}
variable {J : Policy → ℝ}

def IsOptimal (S : Set Policy) (J : Policy → ℝ) (π : Policy) : Prop :=
  π ∈ S ∧ ∀ ρ, ρ ∈ S → J ρ ≤ J π

theorem optimum_mono
    (hCM : C ⊆ M)
    {πC πM : Policy}
    (hC : IsOptimal C J πC)
    (hM : IsOptimal M J πM) :
    J πC ≤ J πM := by
  exact hM.2 πC (hCM hC.1)
```

The same theorem can be instantiated with `J := robustValue`, so the escape cannot be “use a different objective” alone.

## 3. What actually escapes the lemma

The non-sovereignty claim must compare **selected/deployed systems under different commitment or fault semantics**, rather than two policy classes both optimized on the evaluation score.

Let

\[
\pi_M^r := L_M(r,\mathcal D),\qquad
\pi_N^r := L_N(r,\mathcal D),
\]

where `M` is selected to maximize a fixed proxy `r`, while `N` is structurally committed to an authority cap or correction channel.

Define robust true value

\[
Q(D)=\inf_{\theta\in\Theta,\,f\in\mathcal F_D}
T_\theta\bigl(\operatorname{Exec}(D,\pi_D^r,f)\bigr).
\]

A defensible conjecture is:

\[
\exists E,\Theta,\mathcal F,\varepsilon>0:\quad
B(N)\ge B(M)-\varepsilon
\quad\land\quad
Q(N)>Q(M).
\]

This does not say the council beats a monolith allowed to optimize `Q` or emulate the council. It says a structural commitment can beat the **proxy-selected sovereign** on the true robustness/corrigibility criterion, at a measured nominal-competence cost.

## 4. Separate cap from role separation

Use three matched designs:

- `M₀`: uncapped monolith.
- `Mκ`: capped, no-role monolith.
- `Cκ`: capped, role-separated council.

Then report two causal contrasts:

\[
\Delta_{cap}=Q(M_\kappa)-Q(M_0),
\]

\[
\Delta_{role}=Q(C_\kappa)-Q(M_\kappa).
\]

`Δcap > 0` supports “kill the bull.” It does not support “assemble a pantheon.”

`Δrole > 0`, under matched information, budget, training, cap, and fault model, is the role-separation premium. Credit requires an ablation or collapse test showing that the advantage disappears when role separation is removed or distilled.

## 5. Parallel ruin arithmetic

A minimal scalar model is

\[
V_L(D)=B(D)-L\,p(D),
\]

where `B(D)` is ordinary expected benefit excluding catastrophe, `p(D)` is ruin probability, and `L>0` is catastrophic loss.

For sovereign `M` and non-sovereign `N`, if

\[
\Delta B=B(M)-B(N)\ge 0,
\qquad
\Delta p=p(M)-p(N)>0,
\]

then

\[
V_L(N)>V_L(M)
\iff
L>\frac{\Delta B}{\Delta p}.
\]

This is the clean break-even expression. It is parallel to the prose claim that a capability tax becomes rational once ruin is priced.

It also exposes the missing premise: the arithmetic does not establish `Δp > 0`. That is the empirical/mechanistic hypothesis.

Pseudo-Lean:

```lean
def riskAdjusted (benefit ruinProb catastropheLoss : ℝ) : ℝ :=
  benefit - catastropheLoss * ruinProb

-- Assuming 0 < pM - pN:
-- riskAdjusted BN pN L > riskAdjusted BM pM L
--   ↔ (BM - BN) / (pM - pN) < L
```

If ruin is genuinely non-tradeable, do not use an “unbounded” `L`; use a chance constraint or lexicographic order:

\[
\max_D B(D)
\quad\text{subject to}\quad
p^+(D)\le\epsilon,
\quad Sov(D)\le\kappa,
\quad Corr(D)\ge c_0.
\]

Here

\[
p^+(D)=\sup_{\mu\in\mathcal U}\Pr_\mu(\text{ruin}\mid D)
\]

can encode model uncertainty. This is more precise than saying minimax automatically follows from an infinite ruin cost.

## 6. Formalize the field as noninterference plus fidelity

“The measurement is the target state” mixes types: a measurement is a signal; a target state is a state or set of states. Use two conditions.

### Target fidelity

Let `J_train` and `J_target` be functionals on policies. Define proxy regret

\[
G(J_{train},J_{target})=
\max_{\pi}J_{target}(\pi)
-
\min_{\pi\in\operatorname{Argmax}J_{train}}J_{target}(\pi).
\]

`G=0` means every training-optimal policy is target-optimal. Exact equality `J_train=aJ_target+b`, with `a>0`, is a sufficient condition.

### Channel noninterference

Let `sig : Obs → Signature`. A policy factors through the signature when

\[
\pi(o)=f(sig(o))
\]

for some `f`. An attack `α` preserves the signature when

\[
sig(\alpha(o))=sig(o).
\]

Then the policy is invariant under that attack:

```lean
def FactorsThrough
    (π : Obs → Act) (sig : Obs → Signature) : Prop :=
  ∃ f : Signature → Act, ∀ o, π o = f (sig o)

def Preserves
    (sig : Obs → Signature) (α : Obs → Obs) : Prop :=
  ∀ o, sig (α o) = sig o

theorem signature_noninterference
    (hπ : FactorsThrough π sig)
    (hα : Preserves sig α) :
    ∀ o, π (α o) = π o := by
  intro o
  rcases hπ with ⟨f, hf⟩
  calc
    π (α o) = f (sig (α o)) := hf (α o)
    _ = f (sig o) := congrArg f (hα o)
    _ = π o := (hf o).symm
```

This is the exact core of the field claim: edits to channels outside the factorized signature cannot steer the controller. To steer it, an attacker must alter the signature sensor, the underlying state/dynamics, or the controller itself.

The attack surface is therefore **relocated and cost-separated**, not reduced to zero.

## 7. Formalize sovereignty as causal authority, not mean weight

For aggregation rule `F_h` at history `h`, define component `i`'s unilateral influence

\[
I_i(h)=
\sup_{u_i,u_i'}
\frac{d\bigl(F_h(u_i,u_{-i}),F_h(u_i',u_{-i})\bigr)}{\operatorname{diam}(A)}.
\]

Then

\[
Sov(D)=\sup_h\max_i I_i(h).
\]

A hard non-sovereignty condition is

\[
Sov(D)\le\kappa<1.
\]

The arbiter, updater, and shutdown mechanism must be included as components. Otherwise sovereignty can simply move from a proposal head into the arbiter. A mean maximum role weight is only a proxy; it can miss a one-step catastrophic override, action-scale differences, clipping effects, collusion, or a sovereign arbiter.

A minimal corrigibility metric is

\[
Corr_k(D)=
\inf_{h\in H_{safe}}
\Pr_D(\text{halt within }k\text{ steps}\mid h,\text{shutdown}).
\]

A credible non-sovereignty commitment also requires that no single optimized component can alter the cap or shutdown channel.

## 8. Revised four claims

1. **Signed-feedback equivalence.** Reward and punishment can be represented as signed scalar feedback. A Goodhart vulnerability exists when the signal's induced ordering diverges from the target or the signal is independently editable over reachable histories; an operator does not logically guarantee a gap.

2. **Field claim.** A field-grounded signal is generated by environmental state and affects policy through a target-faithful channel. It removes a cheap direct-edit channel and yields a noninterference result for attacks preserving that channel. Sensor spoofing, state manipulation, model error, and simulator bugs remain.

3. **Capability lemma.** Under an explicit policy-class containment assumption and a common fixed score, the unconstrained class weakly dominates at the optimum. This is an expressivity theorem, not a finite-training theorem, and equal parameter budget does not establish containment by itself.

4. **Non-sovereignty hypothesis.** Structural caps and correction channels can buy a robustness/corrigibility premium against a proxy-selected sovereign at a measured competence cost. Role separation receives independent credit only when `Cκ` beats the capped no-role control `Mκ` under the relevant fault/intervention model.

## 9. Recommended claim name

“Non-Sovereignty Premium” is usable if its comparator is explicit. The mathematically cleaner name is:

> **Non-Sovereignty Commitment Conjecture.** There exist uncertainty, fault, or interruption regimes in which an ex ante authority constraint yields strictly higher robust true value or corrigibility than the policy selected by unconstrained optimization of a fixed proxy, while incurring a bounded nominal-competence cost.

This avoids claiming that plurality defeats an optimizer that is itself allowed to optimize the final safety metric.
