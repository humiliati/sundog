# Copilot Cloud-Agent Test-Run Brief — threebody:phaseNN

Purpose: measure whether a GitHub cloud sandbox can run the long Threebody
experiment harness materially faster (or just usefully offloaded) versus the
local 2–5 h (now measured: up to ~75 h) waits, so more elaborate
claims/ratchets do not bottleneck on the project machine.

This is an **experiment about runtime**, not a license to interpret results.
Read "Discipline boundary" before running anything.

## Operating rule — report before session expiry (READ THIS FIRST)

Observed: a chat-pool agent hit a **~1 h session/context cap** and timed out
*after* finishing phase14 but *before* delivering any report — all of that
work was nearly lost. Therefore:

- **Write results to disk and to the "Cloud sandbox results" section of this
  file incrementally, after EACH phase completes** — never batch the report to
  the end. Commit (or otherwise persist) per phase.
- **Budget for a ~1 h hard cap.** phase13 (~45 min on the measured cloud box)
  alone nearly fills it. Plan to report phase13 immediately on completion, then
  start phase14, rather than chaining all three and reporting once.
- If you are an **assigned Copilot coding agent on an issue**, you may be on a
  different (longer-lived) resource pool than the chat assistant — still report
  incrementally; do not assume a longer budget.
- A final report that lands is worth more than a complete run that never
  reports. Prioritize the report.

## Measured local baselines (this machine, single-thread Node)

From the run `manifest.json` `startedAt`/`completedAt` (exact wall-clock, not
estimates):

| run | trials | wall-clock | s/trial | dominant cost |
| --- | ---: | ---: | ---: | --- |
| `npm run threebody:phase13` | 3,456 | 48.7 min | 0.845 | `oracle` mode (9 cand × 16-step lookahead) |
| `npm run threebody:phase14` | 6,048 | 4.9 min | 0.049 | no oracle; cheap ablations |
| `npm run threebody:phase15:smoke` | 144 | 57.3 min | 23.86 | `forward_oracle_strict` (9 × 32 × 8 substeps) + per-step counterfactual strict-oracle call + dt=0.004 (4,000 steps/trial) |

Takeaway: cost is dominated by oracle-class lookahead, not trial count
(phase14 has 1.75× the trials of phase13 but ran in 1/10 the time because it
has no oracle).

## Local estimate for the Phase 15 full lock (NOT yet run)

`npm run threebody:phase15` = 12,960 trials over the 5-step ladder
{0.004,0.006,0.008,0.01,0.012}, 9 modes, 8 seeds, duration 16,
`--track-action-coupling 1 --precision-receipts 1`.

Extrapolation from the smoke (the representative heavy path: same 9 modes,
precision receipts on), scaled by the average steps/trial ratio
(`avg(1/dt)` full 145.0 vs smoke 166.7 ⇒ ×0.87):

- ≈ 23.86 s/trial × 0.87 × 12,960 ≈ **~75 hours** (range ~65–90 h given the
  mode/dt cost spread). Naive un-adjusted upper bound ≈ 86 h.
- Method and rates recorded here so a later agent can re-extrapolate without
  re-measuring (per `AGENTS.md` workflow rule).

Dominant drivers (each recurs every step, so cost is ~linear in steps/trial):
`forward_oracle_strict` does 9 × 32 × 8 = 2,304 `integrateStep` evals per
controller step; the per-step counterfactual injects a full
`computeStrictOracleThrust` (same 2,304 evals) on every eligible thrusting step
of the other thrusting arms; dt=0.004 is 4,000 steps/trial.

## What the cloud agent should run, in order

```bash
npm ci   # or npm install; pure Node ESM harness, no native build step
npm run threebody:phase13        # hard-void gate A  (local ~49 min)
npm run threebody:phase14        # hard-void gate B  (local ~5 min)
npm run threebody:phase15:smoke  # amended Richardson smoke (local ~57 min)
```

## What to report back (append to this file under "Cloud sandbox results")

1. **Wall-clock per command** (use the `manifest.json` `startedAt`/`completedAt`
   in each `results/threebody/<phase>*/manifest.json`, not a stopwatch).
2. **Gate reproduction numbers**, verbatim from each run's console
   `[threebody] …` lines + promising-best-cell count:
   - phase13 must be: 3,456 trials; candidate envelope rows 88/324; 81
     promising best cells; outcomes 1,154 bounded / 2,030 escape / 272 close.
   - phase14 must be: 6,048 trials; 5,184 paired; candidate envelope rows
     130/648; outcomes 1,269 bounded / 4,616 escape / 163 close.
3. **Environment**: Node version (`node -v`), OS, CPU model, core count.
4. Whether `richardson-order-map.csv` was emitted by the smoke and the
   smoke's `earlyTrajectoryPointCount` on `off` trials (non-zero).

Do NOT run `npm run threebody:phase15` (the ~75 h full lock) — see boundary.

## Two caveats that decide whether this is worth it

1. **Cross-platform bit-for-bit risk (the gates).** The hard-void gates demand
   *bit-identical* reproduction (3,456 / 88·324 / 81 / 1154·2030·272 and
   6,048 / 5,184 / 130·648 / 1269·4616·163). The harness PRNG (`makeRng`,
   integer ops) is deterministic, but `Math.sqrt/sin/cos/log` (used in
   `computeAcceleration`, `seededInitialParticle`, etc.) can differ by ~1 ULP
   across libm / CPU / Node version, and RK4 amplifies tiny differences in the
   chaotic regime. **A gate "deviation" on a different machine may be a
   platform-fp difference, not a code regression.** If a cloud gate deviates,
   do NOT declare Phase 15 void — report the exact numbers and environment so
   it can be triaged against the committed local
   `results/threebody/<phase>*/` artifacts first.
   **First cloud run (2026-05-16, Linux x86_64 / AMD EPYC 7763 / Node
   v20.20.2): both gates reproduced bit-for-bit** (phase13 3,456 / 88·324 / 81
   / 1154·2030·272; phase14 6,048 / 130·648 / 1269·4616·163). The cross-platform
   fp risk did NOT materialize on this stack — a strong de-risk for the whole
   offload approach. The triage rule still stands for any future deviation.
2. **Single-thread: more cores ≠ faster.** The `envelopeCases` loop in
   `scripts/threebody-operating-envelope.mjs` is strictly sequential; the
   scripts spawn no worker threads. A many-core sandbox does **not** speed a
   single run — only a faster single core or pure offload helps. The realistic
   win is *offload* (it runs off the project machine, unattended), not a
   speedup, unless the harness is parallelized first (a separate task: it is
   embarrassingly parallel across `cases` if sharded by `--mass-ratios` /
   `--timesteps` and the per-cell outputs are merged).

## Discipline boundary (pre-registration is binding)

`PHASE15_SPEC.md` is locked. The full lock (`npm run threebody:phase15`) stays
**operator-gated**: it may only run after both hard-void gates reproduce
bit-for-bit, the smoke runs, and `T_window` + Richardson-order evidence are
recorded in `PHASE15_RESULTS.md` and the operator signs off (see
`PHASE15_RESULTS.md` "pending" list). This cloud experiment measures
*feasibility and speed of the gates + smoke only*. It does not run, interpret,
or unblock the full lock, and it does not alter any pinned spec parameter.

## Cloud sandbox results

### Run 1 — 2026-05-16 (chat-pool assistant, ran directly; not an issue-assigned agent)

**Environment:** Node v20.20.2 · AMD EPYC 7763 64-Core (only **4 cores
visible**) · Linux 6.17.0-1010-azure x86_64.

| run | cloud wall-clock | local baseline | ratio |
| --- | ---: | ---: | --- |
| `threebody:phase13` | ~45 min | 48.7 min | ~1.08× faster (marginal) |
| `threebody:phase14` | ~10 min | 4.9 min | **~2× slower** |
| gate pair total | ~55 min | ~53.6 min | **≈ parity** |
| `threebody:phase15:smoke` | not reached (session expired) | 57.3 min | — |

**Gate reproduction — both PASS bit-for-bit on this Linux/EPYC/Node20 stack:**
phase13 = 3,456 / 88·324 / 81 promising / outcomes 1,154 b / 2,030 e / 272 ca;
phase14 = 6,048 / 130·648 / outcomes 1,269 b / 4,616 e / 163 ca. (Resolves the
top caveat in the favorable direction — see caveat 1.)

**Session:** the agent timed out at ~1 h, *after* phase14 finished but before
delivering a report — confirms the ~1 h chat-pool cap and motivates the
report-before-expiry operating rule above. In-run throughput was wildly
inconsistent (158 → ~63 trials/min as oracle modes engaged), consistent with
the local finding that oracle-class lookahead dominates cost.

**Honest read:** *feasible and de-risking, but not a raw speedup on this box.*
The gates reproduce cross-platform and the harness is pure Node (no build
step), so offload works. But raw compute was ≈ parity (phase13 a touch
faster, phase14 ~2× slower) on a 4-core, shared-tenant Azure host with high
variance — a mid-run extrapolation that guessed ~2× faster did not hold once
oracle modes engaged. The value delivered is **(a) offload** (the project
machine is freed; the ~75 h phase15 full lock can run elsewhere unattended)
and **(b) confirmed cross-platform bit-for-bit reproducibility**, not faster
wall-clock. A genuine speedup needs a less-contended / higher-core box **and**
sharding the single-threaded harness (caveat 2), or a faster single core.

### Run 2 — 2026-05-16 (issue-assigned cloud agent; incremental report after phase13)

**Environment:** Node v20.20.2 · AMD EPYC 7763 64-Core Processor (**4 visible
CPUs**) · Linux 6.17.0-1010-azure x86_64.

**Phase completed:** `threebody:phase13` only (reported immediately after phase
completion).

- manifest: `results/threebody/phase13-long-horizon-lock/manifest.json`
  - `startedAt`: `2026-05-16T03:35:03.926Z`
  - `completedAt`: `2026-05-16T04:22:17.010Z`
  - wall-clock: **47m 13.084s**

**Gate reproduction numbers (exact run outputs):**

- phase13 trials: **3,456**
- candidate envelope rows: **88/324**
- promising best cells: **81** (`best-by-cell.csv` with `bestRegionClass=promising`)
- outcomes: **1,154 bounded / 2,030 escape / 272 close_approach**

**Runtime-gated skip decisions for this run:**

- `threebody:phase14`: **skipped** because phase13 wall-clock exceeded the
  issue threshold (~30 min).
- `threebody:phase15:smoke`: **skipped** because phase14 was skipped by the
  prior gate; full lock remains explicitly out-of-scope.

**Caveat-aware interpretation (incremental):**

- This run supports the caveat that cloud hard-void comparisons must include
  exact environment metadata + exact emitted numbers for triage (platform-fp
  effects remain a possible explanation for any future deviation).
- This run also supports the harness caveat: phase13 alone consumed most of
  a ~1 h budget, consistent with offload/reproducibility value rather than
  guaranteed speedup from cloud hardware.

### Synthesis (current — after Run 3) + next-issue guidance

Four independent reproductions now agree (Run 3 raw block below):

| run | platform | phase13 wall-clock | gates bit-for-bit |
| --- | --- | ---: | --- |
| local | Windows / project box | 48.7 min | (reference) |
| Run 1 | Linux / **AMD EPYC 7763** 4-core, chat pool | ~45 min | ✓ |
| Run 2 | Linux / **AMD EPYC 7763** 4-core, assigned agent | 47m13s | ✓ |
| Run 3 | Linux / **Intel Xeon 8370C** 4-core, assigned agent | 45m52s | ✓ |

Conclusions (high confidence — 4 reproductions, 2 distinct cloud CPU ISAs):

- **Caveat 1 retired (hard).** phase13 reproduced bit-for-bit across Windows,
  AMD EPYC (×2), and Intel Xeon — different ISAs/microarch, identical
  3,456 / 88·324 / 81 / 1154·2030·272. Cross-platform fp is not a blocker;
  the triage rule remains only as a formality.
- **Offload, not speedup — settled.** phase13 is 45–49 min on every platform
  and both cloud CPU types; cloud hardware does not accelerate the
  single-threaded harness.
- **Incremental-commit + PR-as-deliverable works.** Runs 2 and 3 both landed
  phase13 cleanly via per-phase PR commit before session end — no data lost.
  That problem is solved; prefer issue-assigned agents over the chat pool.

**Structural blocker (the real finding).** A single assigned-agent session is
~1 h, but phase13 (~46 min) + phase14 (~5–10 min) + phase15:smoke (~57 min) ≈
**~110 min sequential**. phase13 alone eats the session, so Runs 2 and 3 each
landed phase13 only — even Run 3, under the corrected issue that explicitly
authorized all three phases. Authorizing phases cannot create session time.

**Revised next-issue guidance (supersedes the prior list):**

1. **Do NOT re-run phase13 on the cloud.** Reproduced 4× (all PASS, ~46 min);
   re-running it only burns the whole session.
2. Target the two still-unmeasured assigned-pool runs by **fitting each issue
   under ~1 h**. Recommended: **two issues** — (A) `threebody:phase14` only
   (~5–10 min, trivially fits, guaranteed); (B) `threebody:phase15:smoke` only
   (~57 min, fits ~1 h with margin; also verifies the amended Richardson
   sampler emits `richardson-order-map.csv` + non-zero off-trial
   `earlyTrajectoryPointCount` on the cloud). Lighter single-issue
   alternative: phase14-then-smoke cheap-first (~67 min; phase14 always
   lands, smoke may not).
3. Keep: per-phase incremental commit (proven), PR-as-deliverable, the
   determinism triage rule, and the full lock explicitly out of scope and
   operator-gated.

### Run 3 — 2026-05-16 (issue-assigned agent; incremental per-phase report)

**Environment:** Node v20.20.2 · Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
(**4 visible CPUs**) · Linux 6.17.0-1010-azure x86_64.

**Phase completed:** `threebody:phase13` (reported immediately after completion).

- manifest: `results/threebody/phase13-long-horizon-lock/manifest.json`
  - `startedAt`: `2026-05-16T05:31:06.365Z`
  - `completedAt`: `2026-05-16T06:16:58.658Z`
  - wall-clock: **45m 52.293s**

**Gate reproduction numbers (exact run outputs):**

- phase13 trials: **3,456** (**PASS**)
- candidate envelope rows: **88/324** (**PASS**)
- promising best cells: **81** (`grep -c ',promising,' best-by-cell.csv`) (**PASS**)
- outcomes: **1,154 bounded / 2,030 escape / 272 close_approach** (**PASS**)

### Run 4 — 2026-05-16 (issue-assigned agent; phase14+smoke cheap-first, incremental)

**Environment:** Node v20.20.2 · AMD EPYC 9V74 80-Core Processor (**4 visible
CPUs**) · Linux 6.17.0-1010-azure x86_64.

#### Phase 14 — `threebody:phase14` (hard-void gate B)

- manifest: `results/threebody/phase14-mechanism-decomposition-lock/manifest.json`
  - `startedAt`: `2026-05-16T07:31:28.158Z`
  - `completedAt`: `2026-05-16T07:37:08.028Z`
  - wall-clock: **5m 39.870s**

**Gate reproduction numbers (exact run outputs):**

```
[threebody] wrote 6048 phase14-mechanism-decomposition-lock trials to results/threebody/phase14-mechanism-decomposition-lock
[threebody] wrote trial-outcomes.csv, paired.csv, envelope-map.csv, aggregate-envelope.csv, best-by-cell.csv, cell-class-map.csv, cell-delta-map.csv, and candidate-envelope.csv
[threebody] candidate envelope rows 130/648
[threebody] outcomes {"escape":4616,"close_approach":163,"bounded":1269}
```

- trials: **6,048** ✓
- paired non-passive rows: **5,184** (paired.csv has 5185 lines incl. header) ✓
- candidate envelope rows: **130/648** ✓
- outcomes: **1,269 bounded / 4,616 escape / 163 close_approach** ✓

**Gate verdict: PASS** — all four numbers match the pre-registered expected values
bit-for-bit.

#### Phase 15 smoke — `threebody:phase15:smoke`

_(smoke started after phase14 commit; results appended below on completion)_

### Run N — _(next cloud agent: append here, per phase, before session expiry)_
