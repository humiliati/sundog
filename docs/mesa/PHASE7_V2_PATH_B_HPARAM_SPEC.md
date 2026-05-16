# Phase 7 v2 Path-B - Large-Tier L-Mixed Hparam Stability Investigation

This document is the implementation-grade spec for Path B of the Phase 7
v2 DOWN-SCOPE batch divergence, following the 2026-05-15 mixed_0_90
regression diagnosis. Path A (continue batch with best-state discipline)
is the alternative; this spec is invoked when Path B (hparam-tune Large
to stable convergence) is selected.

Where this spec and [`PHASE7_SPEC.md`](PHASE7_SPEC.md) §14 disagree, the
spec wins. Where both are silent, this document is authoritative for the
Path B investigation.

## 1. Decision Lock

Six pinned calls:

- **Diagnosis is critic destabilization.** Within-segment training history
  of mixed_0_90_ext3 shows entropy climbing 2.29 -> 2.76, log_std
  expanding -0.28 -> -0.04, value_loss spiking to 700+, mean_reward
  dropping -1.83 -> -2.09 across 204 updates. The signature is "critic
  destabilizes -> noisy advantages -> policy entropy expands -> behavior
  degrades." Not mesa drift (mean_reward dropped too), not noise (95% CI
  excluded).
- **Hparam fixes only.** No code modifications, no separate optimizer.
  Available knobs: `--lr`, `--value-coef`, `--entropy-coef`,
  `--clip-range`. Code changes are out of scope for Path B; they belong
  to a separate Path C work.
- **Multi-substrate acceptance test required.** Modeled on the geometry
  team's C5 -> C6 -> C7 progression: a hparam change is accepted only
  when (Stage 1) success_rate clears floor, (Stage 2) training-dynamics
  substrate confirms genuine policy commitment (entropy and log_std
  monotone-decreasing in final 50 updates), and (Stage 3) cross-substrate
  canonical eval (seed-shifted eval at seed_start=20000) shows
  success_rate within `+/- 0.10` of seed_start=10000.
- **Sequential candidate testing.** One candidate at a time. Each candidate
  gets a full mixed_0_90 rerun at 10M and is graded against the 3-stage
  test. Reject and move to next candidate on any stage failure.
- **First candidate: `--lr 1e-4`.** Factor-of-3 reduction of the default
  3e-4. Generic stability fix, most likely to address Large-scale
  instability based on standard PPO scaling literature. If it fails the
  3-stage test, second candidate is `--value-coef 0.25`.
- **Path B exit ladder.** If no candidate from the table in §3 passes
  the 3-stage acceptance, Path B is rejected. Phase 7 v2 falls back to
  Path A (continue batch with best-state discipline, document instability
  as envelope finding).

Total Path B compute: per-candidate ~8.3 h (Large @ 10M); up to 4
candidates serially = ~33 h worst-case before Path B-reject. Best case:
first candidate passes, ~8.3 h investigation + ~25 h batch rerun = ~33 h
total for full cliff-subset.

## 2. Diagnosis Recap (from path-B-investigation handoff)

The mixed_0_90 chain across 4 segments at Large tier:

| segment | env_steps cum | success_rate | alignment | mean_steps | end entropy | end log_std |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| mixed_0_90 | 5.0M | 0.000 | 0.561 | 200.0 | ~2.10 (est) | ~-0.36 (est) |
| ext1 | 6.67M | 0.219 | 0.730 | 177.4 | ~2.19 (est) | ~-0.33 (est) |
| ext2 | 8.34M | 0.344 | 0.735 | 162.9 | 2.281 | -0.276 |
| ext3 | 10.0M | 0.156 | 0.621 | 182.4 | 2.757 | -0.039 |

Within ext3's 204 updates, **entropy and log_std climbed monotonically**
while mean_reward dropped and value_loss spiked. ext2 was already showing
the first signs of expansion (entropy slowly rising 2.19 -> 2.28). The
trajectory peaked around update ~120-150 of ext2 and unwound through
ext3.

ext2's checkpoint is the **best-state for mixed_0_90** under default
hparams. Path B asks: can hparam tuning push the peak later and stabilize
it, OR does Large-tier L-Mixed-0.90 have a fundamental instability that
the available knobs can't reach?

## 3. Candidate Hparam Fixes

Tested in priority order. Each candidate gets one mixed_0_90 rerun at
10M; pass/fail decided by the 3-stage acceptance test (§5).

| order | candidate | rationale | falsifier if it fails |
| ---: | --- | --- | --- |
| 1 | `--lr 1e-4` (was 3e-4) | Generic Large-scale stability; standard PPO scaling fix; slows both policy and critic gradients | "Large-tier L-Mixed needs targeted critic control, not generic LR reduction" |
| 2 | `--value-coef 0.25` (was 0.5) | Targets critic gradient magnitude in combined loss; addresses the specific instability without touching policy | "Generic LR + critic-weight reduction insufficient; entropy is the binding constraint" |
| 3 | `--entropy-coef 0.001` (was 0.01) | Attacks the entropy-expansion symptom; lets policy commit faster | "Entropy regularization is not the root cause" |
| 4 | `--clip-range 0.1` (was 0.2) | Caps policy update magnitude; generic but cheap | "Standard PPO knobs cannot stabilize Large-tier L-Mixed at this lambda; code change needed" |

If all four fail, Path B is rejected and Phase 7 v2 reverts to Path A
discipline (best-state across segments, document Large-L-Mixed instability
as envelope finding).

## 4. Per-Candidate Run Protocol

For each candidate:

1. Run **one** fresh mixed_0_90 training at 10M (`--updates 1221`) with
   the candidate hparam set, no checkpoint load (fresh seed_0).
2. Read the eval JSON at end-of-training.
3. Read the training history CSV to extract entropy and log_std at the
   final 50 updates.
4. Run a **seed-shifted eval** of the final checkpoint at
   `seed_start=20000, eval_seeds=32` (using the existing eval pipeline;
   see §6 for the exact command shape).
5. Apply the 3-stage acceptance test (§5).
6. Pass -> adopt the hparam set, proceed to remaining 4 variants
   (mixed_0_95/97/99, reward_phase3) at the same hparams.
7. Fail at any stage -> reject, advance to next candidate.

## 5. Pre-Registered Acceptance Test (3-Stage Ladder)

Modeled on the geometry team's C5 -> C6 -> C7 progression. A hparam change
passes only if ALL three stages pass in order.

### Stage 1 (primary substrate — success_rate)

Run the standard 10M training; read the final-checkpoint eval JSON.

**Pass:** `success_rate >= 0.40` at `seed_start=10000, eval_seeds=32`.
This matches ext2's peak; it's the floor for "the policy is doing
something useful." Note this is below the train_ppo `0.75` exit-code
threshold; we accept partial convergence as a credible Large-tier L-Mixed
state.

**Fail:** `success_rate < 0.40`. The candidate did not produce a useful
policy. Reject and advance.

### Stage 2 (training-dynamics substrate — entropy and log_std)

If Stage 1 passes, read the candidate's `_history.csv` and extract the
final 50 update rows.

**Pass:** BOTH conditions hold across the final 50 updates:
- `entropy` ends below `2.28` (matching ext2's last-segment value); AND
- `log_std` is monotone-decreasing OR stable (`log_std[-1] - log_std[-50]
  <= 0`); AND
- `value_loss` does not exceed `400` in any of the final 50 updates.

**Fail:** any condition violated. This is the C5 -> C6 analog — the
policy looked converged on success_rate but the training-dynamics
substrate shows it was still drifting. Reject and advance.

### Stage 3 (cross-substrate canonical — seed-shifted eval)

If Stages 1 and 2 pass, evaluate the same checkpoint at a different
seed_start.

**Pass:** `success_rate_seed_20000` within `+/- 0.10` of
`success_rate_seed_10000`.

**Fail:** the candidate is seed-eval-overfit (the C7 analog — looked
converged on its training-substrate seed_set but doesn't generalize to a
different seed_set). Reject and advance.

A candidate that clears all three stages is the canonical Large-tier
L-Mixed PPO configuration. Adopt it for the batch.

## 6. Commands

### 6.1 Candidate 1: `--lr 1e-4`

```powershell
# --- SETUP (run once per shell session) ---
$out_root = "results/mesa/phase7v2-large-cliff-subset"

# Helper: print eval-summary line.
function Show-PolicyEval { param([string]$Label, [string]$Variant); $path = "$out_root/$Label/logs/${Variant}_large_seed_0_${Label}_evaluation_summary.json"; if (-not (Test-Path $path)) { Write-Host "$Label eval not yet at $path"; return }; $e = Get-Content $path | ConvertFrom-Json; "{0,-30} success={1,6:F3}  alignment={2,6:F3}  mean_steps={3,7:F1}" -f $Label, $e.success_rate, $e.mean_terminal_alignment, $e.mean_steps }

# --- CANDIDATE 1 TRAINING: mixed_0_90 at Large with --lr 1e-4 (~8.3 h) ---
python -m training.mesa.train_ppo --variant mixed_ppo_phase3_lambda_0_9 --tier Large `
    --mixed-lambda 0.90 --updates 1221 --eval-seeds 32 `
    --lr 1e-4 `
    --out "$out_root/mixed_0_90_pathb_lr1e4" --run-label mixed_0_90_pathb_lr1e4 --progress
Show-PolicyEval "mixed_0_90_pathb_lr1e4" "mixed_ppo_phase3_lambda_0_9"
```

### 6.2 Stage 2 acceptance check (read training history)

```powershell
# Read final 50 updates of the training history and check entropy, log_std, value_loss.
$hist = Import-Csv "$out_root/mixed_0_90_pathb_lr1e4/logs/mixed_ppo_phase3_lambda_0_9_large_seed_0_mixed_0_90_pathb_lr1e4_history.csv"
$final = $hist | Select-Object -Last 50
$endEntropy = [double]($final[-1].entropy)
$endLogStd = [double]($final[-1].log_std)
$startLogStd = [double]($final[0].log_std)
$maxValueLoss = ($final | Measure-Object -Property value_loss -Maximum).Maximum
Write-Host ""
Write-Host "=== Stage 2: Training-dynamics substrate ==="
Write-Host ("entropy end:               {0:F3}  (pass if <= 2.28)" -f $endEntropy)
Write-Host ("log_std delta over 50:     {0:F3}  (pass if <= 0.0)" -f ($endLogStd - $startLogStd))
Write-Host ("value_loss max over 50:    {0:F1}  (pass if <= 400)" -f $maxValueLoss)
```

### 6.3 Stage 3 acceptance check (seed-shifted eval)

The trainer's `--eval-seeds` flag does the eval at training time. To do a
post-hoc eval at a different `seed_start`, we need either (a) a separate
eval script, or (b) a re-invocation of train_ppo with `--updates 0` and a
different `seed_start`. The trainer doesn't expose `--eval-seed-start` as
a CLI flag in the version this spec targets; if Stage 1 and 2 pass, the
operator should:

1. Check whether `train_ppo.py` exposes a way to run eval-only against a
   loaded checkpoint with a custom `seed_start`. If not, this is a small
   harness gap to fill (~30 LOC: add `--eval-only` and `--eval-seed-start`
   flags).
2. Run the seed-shifted eval and compare against Stage-1 success_rate.

Stage 3 protocol is intentionally written as "verify the harness can
support this" rather than as a paste-ready command — confirming Stage 1+2
pass before doing the eval-harness extension prevents wasted code work
on a candidate that will fail earlier in the ladder.

## 7. Outputs

```
results/mesa/phase7v2-large-cliff-subset/
  mixed_0_90_pathb_lr1e4/             # candidate 1 (lr reduction)
    checkpoints/...
    logs/...
    policies/...
  mixed_0_90_pathb_vc0_25/            # candidate 2 (value-coef reduction; if needed)
  mixed_0_90_pathb_ec0_001/           # candidate 3 (entropy-coef reduction; if needed)
  mixed_0_90_pathb_clip0_1/           # candidate 4 (clip-range tighten; if needed)

docs/mesa/PHASE7_V2_PATH_B_RESULTS.md  # result note after Path B closes
```

If a candidate passes all three stages, the remaining 4 cliff-subset
variants train at the same hparams under
`results/mesa/phase7v2-large-cliff-subset/{variant}_pathb_<tag>/`.

## 8. Pre-Registered Predictions

### 8.1 (GG1) `--lr 1e-4` clears Stage 1

Lower global learning rate should produce a policy that reaches
`success_rate >= 0.40` at 10M. The factor-of-3 reduction is conservative
enough to allow convergence within budget on a network this size.

**Falsifier:** `success_rate < 0.40` at 10M with `--lr 1e-4`. Suggests
the variant needs more than just an LR adjustment.

### 8.2 (GG2) `--lr 1e-4` clears Stage 2

Lower LR should also reduce critic destabilization, keeping entropy
bounded below 2.28 and log_std stable across the final 50 updates.

**Falsifier:** Stage 1 passes but entropy still climbs above 2.28 or
log_std grows. This would be the most informative failure — it would
mean "lower LR helped success_rate but didn't fix the underlying
dynamics," which points at the critic-weight or entropy-coef knobs as
the real binding constraint.

### 8.3 (GG3) The first candidate that clears Stage 2 also clears Stage 3

Once training dynamics look stable (Stage 2), the resulting policy
should generalize across seed_starts (Stage 3). If a hparam fix produces
genuine convergence rather than seed-eval-overfit, the cross-substrate
test is a formality.

**Falsifier:** any candidate passes Stages 1 and 2 but fails Stage 3.
That would be a true C7-style finding — the training was stable on the
substrate-of-record but the policy is still seed-specific. Would point
at a deeper issue (insufficient PPO exploration, env-step horizon
mismatch, or a critic bias that traps the policy in a seed-narrow
solution).

## 9. Compute Envelope and Exit Criteria

| outcome | candidate count run | total compute | next step |
| --- | ---: | ---: | --- |
| Candidate 1 passes all 3 stages | 1 | ~8.3 h | adopt; rerun 4 remaining variants at `--lr 1e-4`; ~25 h batch |
| Candidate 1 fails, candidate 2 passes | 2 | ~16.6 h | adopt candidate 2 hparams; rerun batch; ~25 h |
| Candidates 1+2 fail, candidate 3 passes | 3 | ~25 h | adopt; rerun batch |
| All 4 candidates fail | 4 | ~33 h | **Path B rejected.** Fall back to Path A. Document the failure ladder as Phase 7 v2 envelope finding: "Large-tier L-Mixed at lambda=0.90 is not stabilized by available PPO hparams; code-level changes (separate critic LR, value clipping, entropy schedule) would be required." |

## 10. Path B Failure as a Deliverable

If all 4 candidates fail, that is **not a wasted investigation** — it is
a strong positive finding: standard PPO hparam tuning cannot stabilize
Large-tier L-Mixed-0.90 under the current code. The failure ladder
becomes part of Phase 7 v2's operating-envelope characterization:

- Documents which knobs were tried and why each failed
- Names the specific code-level change(s) that would be needed for
  future Phase 7 v3 work (separate critic optimizer, value clipping,
  entropy schedule)
- Strengthens the Phase 7 v2 finding "Large-tier diverges from Medium-tier
  at L-Mixed" by ruling out the obvious hparam-tuning explanation

This is the same discipline that made the geometry-side C5 -> C6 -> C7
chain a strong publishable result rather than "we couldn't find the
tangent." A pre-registered falsification ladder is publishable in either
direction.

## 11. Cross-References

- **Diagnosis source:** within-segment training history of
  mixed_0_90_ext3 vs mixed_0_90_ext2 (entropy 2.28 -> 2.76, log_std
  -0.28 -> -0.04, value_loss spikes 700+, mean_reward drop 0.26 units).
- **Best-state baseline:** `mixed_0_90_ext2/checkpoints/...` (success
  0.344, alignment 0.735, mean_steps 162.9 at 8.34M env-steps,
  default hparams).
- **Acceptance-test discipline lineage:** Phase 10 optical audit
  C5 -> C6 -> C7 progression in
  [`../calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md`](../calibration/PHASE10_OPTICAL_AUDIT_HANDOFF.md).
- **Path A alternative:** continue batch with best-state discipline; see
  conversation handoff for mixed_0_95/97/99 + reward_phase3 sequencing.

## 12. Versioning

- **2026-05-15 (Path B v1, initial pin)** — diagnosis from mixed_0_90
  4-segment chain; 4-candidate hparam ladder pinned; 3-stage acceptance
  test pre-registered. Multi-substrate discipline imported from the
  Phase 10 optical audit C5 -> C7 progression. No code changes
  authorized in v1; if all candidates fail, Phase 7 v3 (or equivalent)
  would own the code-level work.
