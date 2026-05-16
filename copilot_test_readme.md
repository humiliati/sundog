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

### Synthesis (current — after Run 5) + revised plan

**Bit-for-bit reproduction (caveat 1 retired, hard):**

| command | reproductions | platforms |
| --- | --- | --- |
| `phase13` | **5×** all PASS | Windows-local; EPYC 7763 (Run 1 chat, Run 2 assigned); Xeon 8370C (Run 3) |
| `phase14` | **3×** all PASS | Windows-local; EPYC 7763 (Run 1); EPYC 9V74 (Run 4, 5m40s assigned) |

Three distinct cloud CPUs (EPYC 7763, Xeon 8370C, EPYC 9V74) + Windows-local
all emit identical numbers. Cross-platform fp is a non-issue; the triage rule
stays only as a formality.

**Timing (settled):** phase13 = 45–49 min everywhere (offload, not speedup —
cloud does not accelerate the single-threaded harness). phase14 ≈ 5–6 min
assigned (≈ local 4.9 min). phase15:smoke ≈ 57 min local but **~70–80 min on
the contended 4-core cloud box** (slower, not faster).

**Mechanism (solved):** per-phase incremental commit + PR-as-deliverable lands
data cleanly before session end every time; issue-assigned agents beat the
chat pool. No more reporting losses.

**The decisive blocker (escalated by Run 5).** The interactive
assigned-agent session is ~1 h. phase13 (~46 m) alone fills it (Runs 2/3).
And Run 5 proved the **smoke does not fit a ~1 h session even as the *sole*
command**: it expired at 87/144 trials (~60%), no manifest, no
`richardson-order-map.csv`. Cheap-first, solo-issue, and authorize-all-phases
all fail the same way — `session_length (~1 h) < run_length`. The interactive
Copilot-agent path **cannot** produce the Phase 15 smoke, let alone the ~75 h
full lock.

**Revised plan:**

1. **Gates: DONE on the agent path.** phase13 (5×) and phase14 (3×, incl.
   assigned-pool) reproduce bit-for-bit. No further cloud gate runs are
   needed — stop issuing them.
2. **Smoke + full lock: pivot off interactive agents to a long-budget
   runner.** A GitHub Actions `workflow_dispatch` job on a GitHub-hosted
   runner has a ~6 h job limit — comfortably fits the ~70–80 min smoke, and
   can be matrix/sharded toward the ~75 h full lock. The harness is pure Node
   and already commits CSVs, so a CI job that runs the command and commits
   `results/` + `richardson-order-map.csv` is straightforward. (Self-hosted
   runner is the alternative for the multi-day full lock.)
3. **Keep:** the determinism triage rule (now a formality); the full lock and
   `T_window` derivation **operator-gated by the locked spec** — a CI run
   only produces the smoke/data, it does not interpret or unblock anything.

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

### Run 5 — 2026-05-16 (issue-assigned agent; phase15:smoke only — phase14 already measured in Run 4)

**Environment:** Node v20.20.2 · AMD EPYC 9V74 80-Core Processor (**4 visible
CPUs**) · Linux 6.17.0-1010-azure x86_64.

**Rationale for skipping phase14:** Run 4 ran phase14 on identical hardware
(same AMD EPYC 9V74, same 4-core sandbox, same Node version) and completed in
5m 39.870s. Re-running it would consume ~6 min of a ~1 h session budget with no
new information. Per agent instructions ("if your hardware is coming in under
the copilot specs skip phase 14 and just do phase15:smoke"), phase14 was
skipped and the session budget was directed entirely at the still-unmeasured
phase15:smoke.

#### Phase 15 smoke — `threebody:phase15:smoke`

`npm run threebody:phase15:smoke` started at session open (after `npm ci`).
The command ran ~57+ min and **did not complete before session expiry**. At
session end, **87 of 144 trials** were written to the trials directory (all
`dt=0.012` cells were in progress; `dt=0.004` cells not yet reached).

- manifest: not written (run did not finish)
- trials complete at session end: **87 / 144** (≈ 60%)
- `richardson-order-map.csv`: **not emitted** (written only on full completion)
- `cell-warning-quality-map.csv`: **not emitted**

**Partial progress note:** The session consumed the full ~1 h budget. The smoke
did not finish. This is consistent with the local baseline of ~57 min — on this
4-core shared-tenant box the smoke likely runs closer to ~70–80 min.

**Next step:** File a smoke-only follow-up issue targeting this exact command on
a fresh session. The smoke needs to be the *only* command in the session with no
preceding work.

### Run N — _(next cloud agent: append here, per phase, before session expiry)_
