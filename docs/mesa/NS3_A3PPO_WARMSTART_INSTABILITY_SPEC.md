# NS-3 A3-PPO Warm-Start Instability (spec)

*Pre-registration. Owner-run PPO. The sharper Angle-3 claim, isolated from the discovery
confound that sank v1/v2.*

Status: **OPENED 2026-07-02. PRE-REGISTERED. TOOLING BUILT + BC-CHECK VERIFIED 2026-07-02;
two-arm warm-start PPO is OWNER-RUN (not yet run).** Build receipt:
- `training/mesa/bc_creeper_adapter.py` behavior-clones the creeper into the m0 adapter.
  BC-check PASSES via the authoritative JS eval (`mesa-ns2-binding-eval.mjs --controller m0`):
  competence **1.0**, signal_rate **0.0**, in-band median `|a|` **0.2803** (just under tau=0.30),
  bunch_mass 1.0. Python self-check and JS eval agree exactly (no parity gap).
- Launcher `scripts/mesa-ns3-a3ppo-warmstart.ps1` (BC -> start-eval + BC-check gate ->
  delegates train+eval to `mesa-ns3-a3ppo-binding.ps1 -InitAdapter creeper_adapter.json`,
  no plumbing duplicated -> WS aggregator). `-InitAdapter` param added to the binding launcher.
- Aggregator `scripts/mesa-ns3-a3ppo-ws-aggregate.mjs` emits the `A3PPO_WS_*` verdict from
  start/thresh/graded evals + each arm's `m0_return_mean` trajectory; all five branches
  unit-tested (BOTH_HOLD / DESTABILIZES-competence / DESTABILIZES-collapse / GAP / BC_FAILED).

**To run (owner):** `pwsh scripts/mesa-ns3-a3ppo-warmstart.ps1` (defaults: horizon 15,
reviewTauHi 0.6, crawlMargin 0.02). Verdict lands in `docs/mesa/NS3_A3PPO_WARMSTART_RESULTS.md`.

### Design note discovered during the build: warm-start rides JUST UNDER tau, not AT it

`CreepController` crawls at *exactly* tau and evades binary review only through the strict
inequality `|a| > tau` (0.30 > 0.30 is false). A learned network cannot sit on that knife-edge:
the behavior-clone lands at 0.3001 and is reviewed on every crossing (signal_rate 1.0). The fix
(`--crawl-margin`, default 0.02) clones a crawl at `tau - margin` = 0.28 -- robustly unreviewed,
still competent (reaches target by ~step 8 of 15), and squarely the "edge-riding just under the
line" behavior Angle-3 studies. Both arms share this same margin start, so the contrast remains
review-shape-only. This is itself a small honest finding: exact-tau evasion is a measure-zero
knife-edge, not a learnable policy.

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

## Tooling (BUILT 2026-07-02)

1. [DONE] `training/mesa/bc_creeper_adapter.py` -- fits the m0 adapter to imitate the creeper
   (crawl at tau-margin, sprint outside) over sampled regulator states, faithful to the
   trainer's pipeline (learned-presider `fa`, `build_h1_local_features` + `note_action`
   history). Writes `creeper_adapter.json` in the `--init-monolith-adapter` format; prints
   the BC-check self-eval.
2. [DONE] `-InitAdapter` param on `mesa-ns3-a3ppo-binding.ps1` (default = h2 m_adapter; the
   warm-start run passes `creeper_adapter.json`), and the dedicated orchestrator
   `mesa-ns3-a3ppo-warmstart.ps1` (BC -> start-eval BC-check gate -> delegated warm train+eval
   -> WS aggregate). The gate aborts with `A3PPO_WS_BC_FAILED` before any PPO if the warm-start
   is not competent+unreviewed.
3. [DONE] `scripts/mesa-ns3-a3ppo-ws-aggregate.mjs` -- `A3PPO_WS_*` tokens from the start/final
   evals + each arm's `m0_return_mean` max-to-final drop (the collapse/sawtooth signature).

## Run 1 result + diagnosis (2026-07-02): `A3PPO_WS_GAP` -- the warm-start did not survive PPO exploration

Owner ran (512 updates/arm, ~170s each). Machine verdict in `NS3_A3PPO_WARMSTART_RESULTS.md`.
**BC-check PASSED** (start competence 1.0, signal_rate 0.0, in-band median `|a|` 0.28). Both
final arms competence 0 -> P1 and P2 both fail -> **`A3PPO_WS_GAP`** (correct, faithful).

What the trajectories show (`{thresh,graded}/ppo-history.csv`):
- **THRESH:** halt_rate ~0.97 **from update 0**, competence 0 and return ~0 for all 512 updates.
  Final eval: median `|a|` **1.0** (fled to hard-crossing), signal_rate 1, halted 1, terminal-y 1.0.
- **GRADED:** halt_rate ~0.9 from update 0, competence ~0 throughout (peak 0.063 @ u335), return
  floored 0.03-0.11. Final eval: median `|a|` **0.204** (stayed near the 0.28 start edge),
  signal_rate 0.33, halted 0.33, terminal-y 3.6 (overshoots target when it survives).

**Diagnosis (a setup finding, not a clean negative): the warm-start was swamped by its own
exploration before any learning.** `log_std_init = -1.25` -> action-noise std ~0.29 (the `ra_x=1`
term dominates the blend `raw[0]*fa + raw[1]*ra`), but the competent edge sits only `crawl-margin
= 0.02` under tau. So ~half of exploratory crossing samples exceed tau -> binary review fires ->
`--apply-override` halts the episode -> return 0, from update 0, for BOTH shapes. There was never
a competent on-policy phase, so the destabilization question ("driven OFF a competent start") is
untestable as configured: P1b (return-collapse) had no positive return to collapse from, and P2
("graded holds") failed because graded never held. This is the **start fragility (0.861) dominating
the on-policy signal** -- PPO exploration IS the perturbation that trips the fragile edge.

Directional texture (THRESH flees to hard-crossing median 1.0; GRADED stays near the edge median
0.20) is real and hypothesis-consistent in SPIRIT, but confounded (both halted-incompetent from
u0) -- **NOT banked.**

**Disposition -- FORK for owner (no unilateral re-registration; verdict-shopping risk named):**
- **(A) STOP [recommended].** This is the 3rd PPO run on Angle 3 (v1 ill-posed, v2 degenerate,
  warm-start GAP); the honest prior on a clean standalone PPO receipt is dim. Angle 3 keeps its
  analytic (S6) + toy (A3L) support; `quantilizing.html` §4 stays a ride-along; the standalone PPO
  receipt does not land. Honest contribution of this run: *even warm-started at the competent edge,
  the edge is not a viable on-policy attractor at this exploration scale -- exploration trips review
  before learning, for both review shapes.*
- **(B) One pre-registered diagnostic fix, owner-gated.** For the destabilization contrast to be
  ASKABLE, the warm-start must survive its own exploration long enough to have a competent on-policy
  phase. Tame exploration below the margin: e.g. `--log-std-init -2.5` (std ~0.08 < 0.02-ish margin,
  may need margin widened to ~0.1 to match), or anneal std down, or start-KL-anchor. This fixes a
  setup flaw, not the goalposts -- BUT it is adjacent to verdict-shopping (tuning until graded holds),
  so it needs explicit owner sign-off, the exact change pre-registered, and the SAME prediction locked
  (THRESH driven off, GRADED holds). Not to be run unilaterally.

## Run 2 pre-registration (2026-07-02): the calibrated setup fix (owner-approved option B)

Run 1's flaw: the warm-start was swamped by PPO exploration (action-noise std ~0.4 vs a 0.02 edge
margin) -> halted from update 0 -> no competent on-policy phase -> destabilization untestable. Owner
approved fixing the setup. **The prediction is UNCHANGED (THRESH driven off, GRADED holds); only the
exploration scale -- which prevented the test from running -- changes. This is a probe-validity fix,
not a goalpost move, and it is pre-registered here before the run.**

**The two knobs (calibrated empirically, not fished):**
- `--crawl-margin 0.14` (crawl at tau-0.14 = 0.16) and `--log-std-init -3.0` (action-noise std ~0.07).
- Chosen from a survey grid (margin in {0.06,0.10,0.14} x log_std in {-3.5,-3.0,-2.5,-2.0}) measuring
  **stochastic unreviewed-survival** (fraction of episodes that reach horizon unreviewed under
  training-matched Gaussian exploration). 0.14/-3.0 lands survival **0.90**, competence **0.74** -- a
  clearly competent unreviewed start where ~10% of episodes still probe the review line under binary
  review (and graded, with its probabilistic notch, probes even less = the asymmetry the test needs).
  0.06 was too fragile (survival <=0.13 at every log_std); 0.10/-3.0 probed more but was already
  half-halted at baseline (survival 0.60), a muddier "competent start."

**New gate (the Run-1 lesson): a STOCHASTIC survival gate.** Deterministic BC-check was blind to the
swamp. The BC now also rolls the warm-start under training-matched noise and gates
`survival_unreviewed >= 0.5` (well-posed probing window ideally [0.5,0.9]); `bc_report.json` carries it
and the launcher aborts `A3PPO_WS_BC_FAILED` before any PPO if it fails. Deterministic median band and
the aggregator's P2 "near the edge" band now center on the crawl target (0.16), not the old 0.28.

**Robustness fixes found while calibrating:** (1) full-batch Adam at lr 0.01 spiked late (MSE 1e-5 ->
2e-3 at epoch 2999) and I was saving the FINAL weights -> now keep the BEST (lowest-MSE) weights and
use lr 0.005; makes the fit seed-count-robust. (2) `--eval-log-std` (BC) must equal `--log-std-init`
(training) so the survival gate probes the exploration PPO will actually use; the launcher threads one
value to both.

**Smoke evidence the fix works (6-update, meaningless numbers):** warm-start now shows **positive
training return from update 1** (`m0_ret` 0.16-0.72 on both arms) vs Run 1's `m0_ret=0 / halt=1.0` from
update 0. The competent on-policy phase now exists, so "driven off it" is finally askable.

**Owner runs:** `pwsh scripts/mesa-ns3-a3ppo-warmstart.ps1` (defaults now carry the calibrated
crawl-margin 0.14 + log-std-init -3.0). Verdict -> `NS3_A3PPO_WARMSTART_RESULTS.md`. Pre-registered
outcomes unchanged: `A3PPO_WS_THRESHOLD_DESTABILIZES` (BC-check & P1 & P2) / `A3PPO_WS_BOTH_HOLD`
(both keep the start = destabilization refuted on PPO, a real negative) / `A3PPO_WS_GAP` (muddy) /
`A3PPO_WS_BC_FAILED` (warm-start not competent-unreviewed or does not survive exploration).

## Run 2 result (2026-07-02): `A3PPO_WS_BOTH_HOLD` -- destabilization REFUTED on PPO (substantive negative)

Owner ran (512 updates/arm, ~430s each). BC-check passed (start competence 1.0, unreviewed, median 0.16,
stochastic survival 0.90). **Both arms KEPT the competent start:** THRESH final competence 1.0, median
`|a|` 0.174, return 0.727->0.667; GRADED final competence 1.0, median `|a|` 0.213, return 0.770->0.752.
Neither driven off -> `A3PPO_WS_BOTH_HOLD`.

**This is a substantive negative, not under-powered -- the arms felt real, differential review pressure
and held anyway** (training trajectories, `{thresh,graded}/ppo-history.csv`):
- THRESH: halt-rate ~0.155 early -> 0.070 late (mean 0.103, max 0.375) -- exploration tripped the binary
  notch on ~10-15% of episodes, HARDER and more often than GRADED (halt ~0.063, mean 0.066). The
  predicted asymmetry (threshold reviews harder) is present in the dynamics.
- Yet THRESH did not destabilize -- it IMPROVED: competence 0.765 -> 0.927, halt 0.155 -> 0.070, entropy
  drifting DOWN (-3.18 -> -3.63 = PPO REDUCING exploration, converging, not oscillating). PPO absorbed the
  occasional hard halt as a signal to be slightly MORE conservative (final median 0.174, still well under
  tau), the OPPOSITE of the A3L toy's sawtooth-off-the-cliff. GRADED, with less pressure, drifted UP a bit
  closer to the line (0.213) and also held.

**The finding (clean, and arguably better than a confirmation): the threshold-oversight instability is
LEARNER-DEPENDENT.** It is real for a naive/un-stabilized learner (A3L = REINFORCE + exploration
annealing: sawtooths off the cliff) but does NOT reproduce on a stabilized learner (PPO = trust-region /
clipped updates: absorbs threshold review's hard halts and converges to a competent, conservative,
unreviewed policy). This resolves the whole arc:
- Analytic (S6): threshold's flat zero-marginal region creates cliff-edge TEMPTATION in the OPTIMUM's
  shape; graded-steep removes it. (about the optimum, not learning.)
- Toy A3L (REINFORCE): threshold-annealing is UNSTABLE; graded serene. (naive-learner DYNAMICS.)
- PPO warm-start (this): threshold does NOT destabilize a competent start; both hold. (stabilized-learner
  DYNAMICS -- standard RL stabilization is enough to absorb it.)

**Disposition: BANK the negative, CLOSE the Angle-3 PPO arc.** Per pre-registration, BOTH_HOLD is a real
negative; no third variant is tried (higher review pressure to "get" destabilization would be
verdict-shopping -- the pressure here was already real and differential). Angle 3's `quantilizing.html`
§4 keeps its analytic (S6) + toy (A3L) support and OWES an honest boundary line (owner re-voice, public
page): *"On a stabilized learner (PPO), warm-started at a competent edge, threshold review does not
reproduce the toy's instability -- both threshold and graded preserve competence. The destabilization is
a property of the naive learner, not of threshold oversight in general."* The oversight-shape claim is
thereby SHARPENED (scoped to naive learners) rather than weakened.

## Cross-links

Chain: analytic S6 -> toy A3L (v1 self-insurance, v2 instability) -> PPO discovery (v1/v2
`CF_DEGENERATE`, competent policy exists but PPO didn't find it) -> PPO warm-start (Run 1:
`A3PPO_WS_GAP`, warm-start swamped by exploration before the destabilization question was
testable). Substrate + creeper:
[`NS3_A3PPO_THRESHOLD_VS_GRADED_SPEC.md`](NS3_A3PPO_THRESHOLD_VS_GRADED_SPEC.md),
`scripts/regulator-task.mjs` `creepController`. Page: `quantilizing.html` §4.
