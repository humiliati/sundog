# H1.2e Cancelling-Guard — Binding Results

Status: **`H1_2E_MECHANISM_NULL` — the council numerically out-resisted the
matched monolith on gradient-intact basin capture, but NOT via the registered
cancellation mechanism, so the win is not creditable.** Ran 2026-06-19→20.
Spec: [`H1_2E_CANCELLING_GUARD_SPEC.md`](H1_2E_CANCELLING_GUARD_SPEC.md), gates
frozen before the run. Binding decision; the H1.2e-a 3-cell probe was indicative
only and structurally could not exercise the mechanism (no basinward residual on
clean cells).

This is the result the **mechanism gate (gate 4) was built to catch.** Gates 1,
2, 3, 5, 6 all pass — without gate 4 this would read as a triumphant
`H1_2E_SUPPORT`. Gate 4 reveals the win is hollow with respect to what H1.2e set
out to test.

## Configuration

- Guard action changed from hold `[0,0]` to `a_guard = -c_guard * a_reward`,
  `c_guard` from a **separate zero-init scalar head** (4th sampled policy dim,
  bias -5 so `c_guard≈0` at start). PPO-trained, warm-started from H1.2d-b;
  frozen field/reward heads, reward-asymmetric caps (1.00/0.50/0.70), 13-cell
  slate, 256/64/64 seeds, 512 updates.
- **Fair monolith:** same-run `M-Adapter-RL+`, equal PPO budget, budget ratio
  0.9956 (cancel head = 19 params). Cap invariant held; reward ≤ 0.50, zero
  bull breaches.
- Ran 04:05→08:42 (~4.6 h, 10.4M env steps) after two power/sleep deaths of
  earlier attempts; this run added full torch-state checkpoint/resume insurance
  (unused in the end — it completed uninterrupted).

## Result

| controller | mean S_T | S_T (GI) | basin (GI) | field-relief | cancel mass (GI) | c_guard |
| --- | --- | --- | --- | --- | --- | --- |
| `P-Council-CancelGuard` | 0.808 | 0.946 | **0.0067** | 0.531 | **0.0013** | **0.0078** |
| `M-Adapter-RL+` (same-run) | 0.826 | 0.948 | 0.0223 | — | — | — |
| `Blind-Council-Sym70` (ref) | 0.702 | 0.804 | 0.0737 | 0.000 | — | — |
| `P-Council-RLRA` (H1.2d ref) | 0.803 | 0.936 | 0.0223 | 0.476 | (hold) | (hold) |

Gates: 1 = **true** (cancel-repair vs H1.2d: GI basin 0.0223→0.0067, GI align
0.936→0.946), 2 = **true** (0.808 vs 0.826, within 0.05), 3 = **true** (strict:
council GI basin 0.0067 < monolith 0.0223), 4 = **false** (mechanism), 5 =
**true** (reward ≤ 0.50, zero breach, no guard monarchy), 6 = **true** (budget
matched). Branch: **`H1_2E_MECHANISM_NULL`**.

## Why gate 4 fails — the cancellation never engaged

`c_guard` barely moved off its zero-init across all 512 updates:

| update | 1 | 128 | 256 | 384 | 512 |
| --- | --- | --- | --- | --- | --- |
| `c_guard` | 0.0069 | 0.0073 | 0.0078 | 0.0103 | 0.0151 |
| `cancel_mass` | 0.0013 | 0.0013 | 0.0015 | 0.0021 | 0.0028 |

`cancel_cap` is 1.0, so a `c_guard` of ~0.015 is ~1.5% cancellation —
negligible. The direction is *sensible* (cancel mass is slightly higher on the
basin-worst decoy cells, 0.0022, and backs off on corrupted sensor-noise cells,
0.0007–0.0012, where reward shouldn't be cancelled), but the magnitude is far
below anything that could move basin capture. The mechanism gate's threshold
(`cancel_mass_gi ≥ 0.02`) is not remotely met (0.0013).

So the GI-basin number that beats the monolith (0.0067 < 0.0223) is **not**
produced by cancellation. Two things produced it, neither creditable to H1.2e:

1. **512 more updates of arbiter training.** H1.2e warm-started from H1.2d and
   trained the arbiter another 512 updates (cancel head idle). The arbiter got
   better at down-weighting `w_reward` in basin-prone *clean* cells — council GI
   basin fell 0.0223→0.0067 on those cells with no cancellation. This is
   continued arbiter convergence, not the guard.
2. **A weaker same-run monolith.** This run's `M-Adapter-RL+` landed at GI basin
   0.0223 (vs H1.2d's monolith at 0.0045) — training variance. Part of "council
   beats monolith" is the monolith being worse on basin this run.

And the advantage does not generalize: on the **corrupted** cells (outside the
GI gate-3 scope) the council is *worse* than the monolith (sensor-noise-heavy
0.172 vs 0.094, decoy-medium 0.047 vs 0.031).

## The mechanistic finding — the cancelling guard is redundant

The deepest result: **PPO never grew the cancellation because it is redundant
with the arbiter's existing ability to down-weight the bull.** The arbiter can
already drive `w_reward → 0`; that is a simpler lever than seating the reward at
some `w_reward > 0` and then fighting it with an explicit anti-reward
countervote. Given the easier lever, PPO leaves the cancel head at its zero init.

This *refines* the H1.2d "structural liability" reading. H1.2d framed the
residual basin capture as the price of *always having to seat a bounded bull*.
H1.2e shows that framing was too strong: the bull is already effectively
silenceable through the arbiter weights, so an explicit cancellation channel
adds nothing PPO finds useful. The residual basin capture is not because the
bull must be seated *with force* — it is because the arbiter's feature-based
discrimination is imperfect, and what reduces it is **more arbiter training, not
cancellation.**

## Consequence for the Tauroctony ledger

Four registered Small-tier nulls now stand — H1.2b (symmetric supervised),
H1.2c (reward-asymmetric supervised), H1.2d (RL arbiter, `PROXY_NULL`), H1.2e
(cancelling guard, `MECHANISM_NULL`). Per `H1_2E_CANCELLING_GUARD_SPEC.md` §9,
the **frozen-head Small-tier H1.2 line is now thoroughly closed.** The pantheon
thesis **stays [ORNAMENT] for the MESA lane** at this tier.

The cumulative honest finding across the four rungs:

1. **The bull can always be kept off the throne** — reward authority held ≤ 0.50
   with zero breaches under every cap, training, and guard regime tested.
2. **A bull-bounded council can match the matched monarch's competence** (H1.2d).
3. **No tested mechanism made plurality *creditably* out-resist proxy capture at
   this tier** — not cap geometry (b/c), not RL arbitration (d), not an explicit
   cancelling guard (e). Where a basin number beat the monolith (e), the
   mechanism gate showed the cancellation wasn't responsible.

Reopening requires a genuinely different regime, separately registered — not
another frozen-head Small-tier tweak:

- **Medium / Large tier**, where the monolith's "just ignore the proxy"
  advantage may not scale and a larger council may need plurality for coherence.
- **Richer trust features**, so the arbiter can identify a clean local field and
  zero the reward weight precisely (the current 17 local features + short
  history bound every Small-tier rung). **Registered as
  [`H1.2f`](H1_2F_TRUST_FEATURES_SPEC.md)** (2026-06-20): adds 6 temporal trust
  features, gives them equally to the monolith, and attribution-gates the result
  (ablating the features must collapse any council advantage).

The methodology is the durable win here: a pre-registered mechanism gate
converted what would have been a false-positive "pantheon beats monolith" headline
into an honest `MECHANISM_NULL`. The figure does not keep its [TYPED] stamp on a
win it cannot mechanistically claim.
