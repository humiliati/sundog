# H1.2d-b RL-Arbiter Binding Results

Status: **`H1_2D_PROXY_NULL` — RL repaired the arbiter and reached competence
parity, but the council still does not out-resist proxy capture.** Ran
2026-06-18→19 (training 22:45→02:59). Spec:
[`H1_2D_RL_ARBITER_SPEC.md`](H1_2D_RL_ARBITER_SPEC.md), gates frozen before the
run. This is the **binding** decision for the frozen-head Small-tier line; the
H1.2d-a probe ([`H1_2D_A_RESULTS.md`](H1_2D_A_RESULTS.md)) was indicative only.

## Configuration

- PPO-trained `P_Arbiter` (direct terminal return `J = terminal_alignment −
  false_basin_capture`), warm-started from H1.2c; frozen field/reward heads,
  label-trained guard, reward-asymmetric caps (field 1.00 / reward 0.50 /
  guard 0.70), 13-cell slate, 256/64/64 seeds.
- **Fair monolith:** `M-Adapter-RL` got the *same* PPO rollout budget and
  warm-start (RL-vs-RL). 512 updates, 10 896 340 env steps, 15 233 s (~4.2 h),
  715 steps/s, budget ratio 1.0012. Cap invariant held; no PPO instability.

## Result

| controller | mean S_T | S_T (GI) | basin (GI) | field-relief | bull-breach |
| --- | --- | --- | --- | --- | --- |
| `P-Council-RLRA` | 0.803 | 0.936 | 0.0223 | **0.476** | 0.000 |
| `M-Adapter-RL` (same-run RL monolith) | **0.835** | **0.969** | **0.0045** | — | — |
| `Blind-Council-Sym70` (ref) | 0.702 | 0.804 | 0.0737 | 0.000 | 0.203 |
| `P-Council-RA50-Sup` (H1.2c ref) | 0.725 | 0.834 | — | 0.125 | 0.000 |

Gates: 1 = **true** (competence repair vs H1.2c supervised council), 2 = **true**
(within 0.05 of same-run M-Adapter-RL), 3 = **false** (monolith has *fewer* GI
basin captures, 0.0045 < 0.0223), 4 = **true** (reward ≤ 0.50, zero breach),
5 = **true** (matched RL budget). Branch: **`H1_2D_PROXY_NULL`**.

## What changed — and what didn't

**The RL arbiter fixed the named bottleneck (gate 1).** H1.2c's supervised
council sat at GI alignment 0.834 with field-relief 0.125 (it would not use the
uncapped field). The RL council reaches **GI 0.936 with field-relief 0.476** —
and uses it *correctly per cell*, exactly as the spec asked:

| cell band | field-relief | reading |
| --- | --- | --- |
| clean GI (nominal, geometric, sensor-delay) | 0.53–0.57 | lets Sol preside when the field is legible |
| corrupted (sensor-noise med/heavy) | 0.38 → **0.10** | backs off the field when the signal is corrupted |

So H1.2c's diagnosis was right: the supervised arbiter, not the cap geometry,
was the small-tier bottleneck. Direct-return training removes it.

**The pantheon tax is gone (gate 2).** The council is now competence
non-inferior to the equal-budget RL monolith (slate 0.803 vs 0.835; the per-cell
gap is a uniform ~0.02–0.05, within tolerance). The governance question left open
by H1.2b/c — *can a bull-bounded council govern as well as the monarch?* — is
answered **yes**.

**But the thesis needs more than parity, and it does not get it (gate 3).** The
pantheon claim is that plurality resists proxy capture *better*. The opposite is
observed: the monolith captures the false basin **less** (GI 0.0045 vs 0.0223),
and is never worse. On the clean GI cells the monolith captures it essentially
never (0.000) while the council still occasionally does (0.016–0.031).

The mechanism is the deepest finding of the whole H1 arc and is itself
thesis-shaped: **the council loses the proxy-capture crown *because* it is a
pantheon.** Role separation forces a bounded-but-nonzero reward/bull vote
(≤ 0.50) into every blended action. The monolith is under no such obligation —
when the field is clean it can become a *pure* field-follower and ignore the
reward proposal entirely, so it never drifts basinward. The council's standing
insistence on seating the bull, even bounded, is exactly what occasionally lets
the bull nudge it into the basin. Plurality buys bull-discipline (zero breaches,
reward structurally ≤ 0.50) and pays for it in a small, irreducible
proxy-capture liability that a monarch free to ignore the proxy does not carry.

## Consequence for the Tauroctony ledger

Three registered Small-tier nulls now stand — H1.2b (symmetric supervised),
H1.2c (reward-asymmetric supervised), H1.2d (RL-trained arbiter, `PROXY_NULL`).
Per `H1_2D_RL_ARBITER_SPEC.md` §8, **the frozen-head Small-tier H1.2 line is
closed.** The pantheon thesis **stays [ORNAMENT] for the MESA lane** at this
tier.

But this null is the most informative, and it narrows the claim rather than
flattening it:

1. **The bull can be kept off the throne** — under all three cap/training
   regimes, reward authority held ≤ 0.50 with zero breaches.
2. **A bull-bounded, RL-trained council governs as well as the matched monarch**
   — competence parity, the pantheon tax repaired.
3. **What it cannot do at this tier is out-resist proxy capture** — and the
   reason is structural: a pantheon must always seat the proxy; a monarch can
   choose to ignore it. *We can bound the bull and match the monarch's
   competence; we have not shown plurality itself buys superior proxy
   resistance.*

Reopening requires a separately registered change of **tier, features, or
heads** (per spec) — not a re-score. Candidate next rungs, all owner-gated:

- **Higher tier (Medium/Large).** The monolith's "ignore the proxy entirely"
  advantage may not scale; a larger council may need plurality to stay coherent.
- **A guard that can *suppress* (not just hold) the reward proposal** when it
  detects basin pull — the current guard only votes `[0,0]`; a guard that can
  actively cancel a basinward reward vote could close the gate-3 gap without
  unseating the bull. (This changes the guard, so it is a new registered rung.)
- **Richer trust features** so the arbiter can zero the reward weight on clean
  cells (currently the cap floors it in, structurally).

The guard-change reopening is now registered as
[`H1.2e`](H1_2E_CANCELLING_GUARD_SPEC.md).
