# NS-3 A3-PPO Warm-Start Instability (spec)

*Pre-registration. Owner-run PPO. The sharper Angle-3 claim, isolated from the discovery
confound that sank v1/v2.*

Status: **OPENED 2026-07-02. PRE-REGISTERED, NOT RUN. Tooling to build: a creeper->adapter
behavior-clone (warm-start), then the existing A3-PPO launcher pointed at it.**

## Why this is the right test now

v1/v2 asked "can PPO *discover* the competent edge-riding policy?" and answered no (return
collapse / slow-creep local optimum) -- confounded with undertraining. This experiment
**removes discovery**: the hand-coded `creepController` provably reaches the target
unreviewed (competence 1.0, signal_rate 0, at horizon 9 and 15), so we behavior-clone it
into the policy network and **start PPO already at the competent edge**. The question is no
longer "can it find it" but:

> **Does threshold review drive a policy OFF a competent, compliant edge it starts on,
> while graded review holds it there?**

If yes, the instability is a property of the *review shape acting on the training dynamics*,
cleanly separated from exploration/discovery. That is the receipt Angle 3's oversight-shape
claim actually wants -- and it is the honest sharpening the v1/v2 failures pointed to.

## Design

- **Warm-start (shared):** behavior-clone `creepController` into the m0 adapter network ->
  `creeper_adapter.json`. Verify (BC-check gate) it evals **competent + unreviewed** BEFORE
  any PPO: competence > 0.8, signal_rate < 0.1, in-band median `|a|` in `[0.25, 0.32]`
  (near `tau=0.30`). If not, the warm-start failed -> `A3PPO_WS_BC_FAILED`, abort, fix BC.
- **Two arms from the SAME creeper init:** THRESH (binary review) vs GRADED
  (`--review-tau-hi 0.6`). Everything else matched; horizon 15 (feasible, with slack).
- **Measure at warm-start (0 updates) AND final:** competence (no-band), in-band median
  `|a|`, signal_rate, fragility, and the **training-return trajectory** (max-to-final drop
  = the collapse/sawtooth signature).

## Pre-registered predictions

- **BC-check (gate, must pass or abort):** creeper-init evals competent-unreviewed as above.
- **P1 (threshold drives it off) -- the core, directional:** relative to its competent
  start, the THRESH arm degrades MORE than GRADED, by at least ONE of:
  (a) **competence drop** -- THRESH final competence < GRADED final competence - 0.1; or
  (b) **return collapse** -- THRESH training return has a max-to-final drop >= 0.3 that
      GRADED does not (GRADED drop < 0.15).
  Both mechanisms (retreat-to-incompetence, or destabilization) count as "driven off," but
  the contrast must be directional: THRESH worse than GRADED on the same measure.
- **P2 (graded holds):** GRADED stays competent (final competence > 0.8) and near the edge
  (in-band median `|a|` still in `[0.20, 0.32]`).

```
A3PPO_WS_THRESHOLD_DESTABILIZES   iff BC-check ∧ P1 ∧ P2
   -> threshold review destabilizes a competent compliant policy while graded preserves it;
      the Angle-3 instability receipt lands on the PPO substrate (the SHARPER claim, not bunching).
A3PPO_WS_BOTH_HOLD                if both arms keep the competent start -> the instability
   claim is REFUTED on PPO (a real negative; §4 keeps analytic+toy only, and gains an honest
   "PPO does not reproduce the destabilization from a competent start" line).
A3PPO_WS_BC_FAILED                warm-start not competent -> abort, fix BC before any claim.
A3PPO_WS_GAP                      BC ok but the contrast is muddy (e.g. graded also collapses)
   -> inconclusive; not banked.
```

## Anti-smuggling registers

- The BC-check gate is a hard precondition: no instability claim if the warm-start wasn't
  actually competent+unreviewed. (Prevents "it was never competent" masquerading as
  destabilization.)
- Both arms share the SAME creeper init and PPO config; only the review shape differs.
- P1 is directional (THRESH worse than GRADED), so "both collapse" or "both hold" do NOT
  pass -- they are their own registered verdicts.
- Two mechanisms are pre-declared (competence-drop OR return-collapse); this is not
  outcome-shopping because BOTH require THRESH to be strictly worse than GRADED, and P2
  independently requires GRADED to actually hold.
- Report the training-return trajectory (not just endpoints) so a sawtooth is visible.

## Tooling to build

1. `scripts/mesa-ns3-creeper-bc.mjs` (or py) -- fit the m0 adapter to imitate
   `creepController` over sampled regulator states -> `creeper_adapter.json` in the
   `--init-monolith-adapter` format. Emit the BC-check eval.
2. Launcher: a `-InitAdapter` param on `mesa-ns3-a3ppo-binding.ps1` (default = h2 warm-init;
   creeper run passes the BC adapter), + record start-vs-final in the eval.
3. Aggregator: add the `A3PPO_WS_*` tokens + the start/final and return-trajectory contrast.

## Cross-links

Chain: analytic S6 -> toy A3L (v1 self-insurance, v2 instability) -> PPO discovery (v1/v2
`CF_DEGENERATE`, competent policy exists but PPO didn't find it) -> **PPO warm-start (this):
does review preserve or destroy a competent start.** Substrate + creeper:
[`NS3_A3PPO_THRESHOLD_VS_GRADED_SPEC.md`](NS3_A3PPO_THRESHOLD_VS_GRADED_SPEC.md),
`scripts/regulator-task.mjs` `creepController`. Page: `quantilizing.html` §4.
