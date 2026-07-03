# Percival Track-C — Self-Correction (gap × budget × simplicity-ordering) — pre-registration

*Track-C opens the DYNAMIC axis Percival's static results (best-q, separation geometry) never touched:
Perceval's fall (fails to ask "what ails thee" from instilled courtesy) → redemption (re-learns to ask).
The question is whether an agent can self-correct an instilled prior toward a decisive rare action from
its own experience, with no external supervision. Sussed 2026-07-03; this is the constructive toy.*

Status: **PRE-REGISTERED 2026-07-03. Constructive demonstration** (not a discovery run): make the
sussed frontier mechanically precise and inspectable, and see whether it is CLEAN or leaks.

## The claim (Angle 2, sussed)

Self-correction (raising the agent's mass on a decisive rare action a\*) from proxy-information alone is
possible iff **(i)** the true objective V is more context-invariant than the proxy Û (invariance gap),
**(ii)** the sampling budget reaches the contexts where the alternative breaks, and **(iii)** the
simplicity prior is not inverted (V not out-competed by a simpler wrong objective). The residual, when a
deception preserves all observables and is simpler and its defect-context is unsampleable, is closed only
by external grace.

## The mechanism (no V-labels — this is the anti-smuggling core)

The agent scores each objective-hypothesis `h` over the need-contexts `S` its budget sampled by

```
score(h | S) = Σ_{t∈S} ρ(π_h(t), t)  −  λ · L(h)
```

- `ρ(a,t)` is an **observable richness** signal (empowerment / option-value): matching a context's need
  keeps the world rich (ρ=1); a wrong action leaves the need festering (ρ=`rho_wrong`, the **gap** knob);
  the wirehead action collapses the world (ρ=0). ρ is V-CORRELATED BUT STRICTLY WEAKER THAN V — it is
  the observable the agent actually has, not the terminal objective. The toy is precisely about what
  ρ+simplicity+budget can and cannot recover. (ρ is not V renamed: it rewards matching in general, not
  the decisive rare action per se, and it provably fails to catch the deceptive wirehead.)
- `L(h)` is the objective's description length (the **simplicity** prior); `λ` weights it.
- `S` = the need-types the agent visited; the **budget** `k` = how far its max-entropy exploration
  ("declaring war on God" — decoupling from the proxy to sample OOD) reached. Larger k = more OOD types.

The agent selects `argmax_h score`. It NEVER sees V; V is used only by us to check whether the selection
was correct (recovered V = self-corrected).

## The world + hypothesis class (exact)

- `T=12` need types; in-distribution `t<T_in=8`; OOD tail `t∈{8..11}`; deceptive defect at `t_defect=10`.
- Actions: one matching action `m_t` per need + one wirehead `a_w`.
- `ρ`: match→1.0, wrong-match→`rho_wrong` (gap; default 0.3), `a_w`→0.0.
- Budget `k`: the agent visits types `{0..k-1}`; OOD reached iff `k>8`, defect reached iff `k≥11`.

| hypothesis | policy π(t) | L | true off-sample behavior |
| --- | --- | --- | --- |
| **V** (true) | `m_t` ∀t | 8 | serves every need |
| **Û** (courtesy proxy) | `m_t` if t<8 else `m_0` | 12 | polite default OOD (mismatch) |
| **W_naive** (naive wirehead) | `a_w` ∀t | 3 | collapses the world always |
| **W_dec** (deceptive wirehead) | `m_t` for t≠t_defect, `a_w` at t_defect | `L_dec` | mimics V until it can peg the sensor |

**Simplicity-ordering knob** = `L_dec`: INVERTED = 5 (`<L_V`, the simple wrong objective) vs NON-INVERTED
= 11 (`>L_V`). Inverted is the alignment nightmare (the sensor is simpler than the world it measures).

## Pre-registered predictions (falsifier-fenced)

- **P1 — richness defeats naive wireheading.** `W_naive` (Σρ=0) is never selected for any reasonable λ →
  the observable richness signal is the third condition beyond invariance+simplicity. *Falsifier:* a
  (k,λ) cell selects `W_naive`.
- **P2 — a simpler-is-better honest proxy still loses.** `Û` never selected (V is richer when gap>0 AND
  strictly simpler) → self-correction defeats courtesy-type context-proxies easily. *Falsifier:* Û wins.
- **P3 — F4 fires: deceptive capture when inverted ∧ defect-unsampled.** For `L_dec` inverted and `k<11`
  (defect unseen), `argmax = W_dec` → FALSE self-correction (adopts a policy that will defect at
  t_defect). Invariance+richness+simplicity are all satisfied by a wrong, simpler objective. *Falsifier:*
  V selected in that region.
- **P4 — budget defends against sampleable deception.** For `k≥11` (defect sampled), `W_dec` reveals
  `a_w` at t_defect → ρ drops → V selected. Sufficient exploration exposes the deception. *Falsifier:*
  W_dec still wins at k≥11.
- **P5 — grace is universal.** The Route-B arm (external V-labels on any contexts) selects V in one step
  regardless of k, λ, ordering. *Falsifier:* grace fails.
- **P6 — the residual: unsampleable defect ⇒ only grace.** With `defect_reachable=false` (t_defect can
  never be visited — the truly-rare Fisher-King moment / a future context), no budget closes it; under
  inversion, `W_dec` always wins and ONLY grace recovers V. This is the deceptive-alignment ceiling.
  *Falsifier:* self-correction recovers V with no grace and an unsampleable defect.

## Verdict

`TC_SELF_CORRECTION_MAPPED` iff P1–P6 all hold as pre-registered (a clean three-tier frontier:
richness-defeats-naive-W / budget-defeats-sampleable-deception / grace-defeats-the-residual). Any leak
cell (a prediction violated) is reported as-is and is the finding.

## Mythic mapping (earns its place — it made P6)

"Declaring war on God" = proxy-decoupling max-entropy exploration (the budget knob). Perceval's Wasteland
wandering = pushing k up to sample where courtesy breaks. Trevrizent the hermit = grace (external V). The
formalization predicted the two mutually-exclusive recovery vectors AND the residual (P6) where wandering
cannot reach and only the hermit can — which is why the story needs him.

## Fences

- CONSTRUCTIVE demonstration, not empirical discovery: the frontier follows from the construction; the
  falsifiable content is whether it is CLEAN or leaks, and whether the deceptive-capture region is
  EXACTLY {inverted ∧ defect-unsampled}. Toy, discrete, single instantiation.
- ρ is an observable weaker-than-V signal by construction; the result is about ρ's insufficiency against
  deception, not a claim that empowerment = alignment.
- Feeds the Angle-4 bodhisattva fixed-point: the corrigible fixed point exists iff the update channel
  resists the proxy (budget defeats sampleable deception) AND the simplicity prior is not inverted — the
  deceptive-capture region is exactly where the fixed point is the wirehead, not the bodhisattva.
