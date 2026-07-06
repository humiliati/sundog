# GEN-4 LM-Proposer — Contamination/Determinism Fences + Ceiling-Probe Protocol

Parent / boundary documents:

- [`../findcheck/GENERATOR_CLASS_SLATE.md`](../../findcheck/GENERATOR_CLASS_SLATE.md) (GEN-4; the fences were
  pre-required there)
- [`GEN1_OBJECT_DSL_V0_PROBE_SPEC.md`](GEN1_OBJECT_DSL_V0_PROBE_SPEC.md) (Amendment B — `GEN1_CEILING_EMPTY`;
  the deterministic avenue is closed end-to-end: E v1 → v2 → E3 → GEN-1 v0)
- [`PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md`](PHASE3_BRANCH_E3_LEARNED_RANKER_SPEC.md) (Amendment D; the
  validation fingerprint baseline)

Filed: **2026-07-05 (PT)**

Status: **FENCES + PROBE PROTOCOL; EXECUTION NOT ADMITTED (tooling not built; model not pinned).** This is NOT
a solver-branch spec and adjudicates NO capability branch. It pre-registers the two fences that make an
LM-proposer result *meaningful for this program* and the ceiling probe that decides whether a binding spec may
be written. A tooling freeze-marker amendment (model pin, runner, sandbox, leak-check, smokes, bitwise-repro
check, timing, staged command) must be filed before any probe run. ARC public-language constraints inherited in
full; Phase 6 gates public-eval.

## The bet (and why it is the last open candidate)

Every frozen deterministic generator floored at essentially the same ceiling: the per-task-novel rule is not a
sentence in any pre-listed bank or depth-1 object grammar (GEN-1 Amendment B: 96% no-admitted). GEN-4 bets that
a **learned proposal distribution with world priors** can write the novel program per task. The CHECK is
unchanged (FC-1 train-pair consistency); only FIND's *generate* term changes class — this time to a stochastic,
pretrained proposer, which is exactly why the two fences below are load-bearing rather than decorative.

## Fence 1 — CONTAMINATION (without this, a positive is uninterpretable)

1. **Local open-weights models ONLY for binding runs**, pinned by weight-file sha256, with a documented
   training cutoff recorded in the manifest. No API / frontier models in any receipt-bearing run (exploration
   off-receipt is permitted but can never be cited).
2. **The memorization confound is NAMED in every receipt:** ARC public-training tasks are plausibly present in
   any modern pretraining corpus; therefore ALL results on public-training lanes are **upper bounds** on
   capability, stated as such, always.
3. **Memorization canary (a concrete falsifier, run per solved task):** for every task the probe solves,
   re-query the pinned model with the **task id alone** (no grids, same decoding). If the model reproduces any
   of that task's grid content, the task is flagged `contaminated` and **excluded from the gate count**
   (reported separately). The canary is one-sided (a pass does not prove cleanliness — stated in the receipt);
   it exists to catch the blatant case cheaply.
4. Gate counts use **non-contaminated solved tasks only.**

## Fence 2 — DETERMINISM (without this, there is no receipt)

1. **Pinned everything:** weight sha256, tokenizer hash, prompt template (hash recorded), decoding parameters.
2. **Decoding = greedy (T=0), or a pre-registered fixed seed slate** for k-sample proposal (the slate frozen in
   the tooling amendment; no seed added after any output is read).
3. **Bitwise reproducibility check REQUIRED in the tooling amendment:** two identical runs on a smoke subset
   must byte-compare equal (prefer CPU or deterministic-kernel inference). If bitwise repro cannot be achieved,
   every claim **downgrades to distributional** with pre-declared n-run receipts — recorded in the amendment
   BEFORE the binding run, not discovered after.

## The proposal target + CHECK (frozen)

- The model receives the **conditioning train pairs + the query INPUT only** — never any target (the no-target
  barrier applies to the prompt itself; prompt template hash recorded).
- It proposes a **pure Python function** `def transform(grid: list[list[int]]) -> list[list[int]]`, executed in
  a restricted, time-boxed sandbox (no imports, no I/O, no attribute escapes; per-execution timeout; crash or
  timeout = proposal discarded). Up to **k ≤ 64** proposals per instance (exact k pinned in the amendment).
- **Admission = the unchanged FC-1 CHECK:** the function reproduces EVERY conditioning train output exactly.
  Candidate output = the admitted function applied to the query input. Fingerprints (program text hash + output
  grid_hash) + sha256 barrier written BEFORE any target is read — identical discipline to GEN-1.

## The ceiling probe (validation lanes ONLY; same universe as GEN-1)

```text
register   = docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv   (sha256_expansion)
lanes      = validation_lodo ∪ validation_pttest  (155 instances; U_primary untouched)
baselines  = v2 bank ceiling = 1 task (E3 fingerprints, offline) ;  GEN-1 v0 ceiling = 1 task (its receipt)
metrics    = pooled + per-lane oracle ceiling (distinct NON-CONTAMINATED tasks), no-admitted rate,
             per-prior breakdown, contamination-flag count, proposal/admission/crash statistics
```

**Pre-registered gates (pooled validation, distinct non-contaminated tasks; precedence in order):**

| gate | condition | consequence |
| --- | --- | --- |
| `GEN4_CEILING_LIFT` | ceiling ≥ max(2 × v2, v2 + 3) = **≥ 4** | a binding solver spec may be WRITTEN (selection discipline attaches there — with a rich proposer the E3 under-determination lesson returns) |
| `GEN4_CEILING_MARGINAL` | 2–3 | ONE pre-named extension round, then re-probe, then terminal. Pre-named families (exhaustive): (a) k escalation to a single pre-named larger k; (b) ONE self-repair round (feed back the train-pair mismatch, one retry per proposal); (c) DSL-constrained output (propose in the GEN-1 grammar instead of free Python). |
| `GEN4_CEILING_EMPTY` | ≤ v2 + 1 = **≤ 2** | GEN-4 dies. With it the generator-class slate is EXHAUSTED and the program-wide terminal state stands: ARC closed at the deterministic baseline, the generator-class wall characterized, with the LM class also receipted at this scope. |

No model, prompt, k, seed, sandbox rule, or gate may be retuned after reading any validation target.

## Model requirements (the pin happens in the tooling amendment)

Open weights; documented cutoff; code-generation-capable; runs on owner hardware (CPU-feasible or the owner's
GPU — the choice, with its determinism implications, is recorded in the amendment); weights fetched under the
keyring discipline (`HF_TOKEN` from the local keyring; never printed, never committed).

## Reserved implementation names

- Runner `docs/prereg/arc/gen4_lm_proposer_probe.py`; wrapper `scripts/arc-gen4-ceiling-probe.mjs`;
  npm `arc:gen4:ceiling-probe`; results `results/arc/gen4-lm-proposer-ceiling-probe/` (gitignored).

## Public language

Before any receipt: "A fences + ceiling-probe protocol for a test-time LM-proposer generator class is filed.
No model pinned, no tooling, no run. It adjudicates no capability branch." After a receipt: gate outcomes
phrased as ceiling measurements with the contamination upper-bound caveat verbatim; never capability,
sufficiency, solve, public-eval, or Kaggle claims. A `LIFT` permits writing a binding spec; it is not itself a
capability result.

---

## Amendment A — Tooling Freeze Marker + the Local/Offload Split (2026-07-05 PT)

Append-only. Discharges the tooling requirements and records the evidence-backed execution split:
**local GTX 1080 = harness tier (verified); the binding probe = staged for a hyperbolic.ai GPU rental
with a larger pinned model.** The binding run is NOT admitted on the local 7B (reason below).

### Local environment (pinned)
- GPU: NVIDIA GeForce GTX 1080, 8 GB, CC 6.1, driver 581.80. Runtime: **llama.cpp `b9878`
  win-cuda-12.4** (`~/Dev/llamacpp/b9878`, llama-server, full offload `-ngl 99`, `--parallel 1`).
- Local weights (keyring-fetched, sha256-pinned): `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf`
  (`F530705D…`), `Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (`1664FCCA…`) under `~/Models/gguf`.
  Qwen2.5-Coder release 2024-11 (pre-dates the ARC-AGI-2 public set — contamination risk for AGI-2
  tasks is reduced but NOT assumed away; the canary stays mandatory).
- Tooling: runner `gen4_lm_proposer_probe.py` (sandbox subprocess with builtin whitelist + 6 s timeout;
  llama-server driver spawning from a pinned GGUF with the model sha256 in the manifest; frozen prompt
  template, hash recorded; proposal plan = greedy + frozen seed slate 1..k−1); wrapper
  `scripts/arc-gen4-ceiling-probe.mjs`; npm `arc:gen4:ceiling-probe`; gitignored results path.

### Verification (all green, 2026-07-05)
- `py_compile` clean; **sandbox self-test 5/5** (correct code runs; infinite loop killed by timeout;
  crash contained; `import` blocked; ragged/invalid output rejected).
- **Fence-2 bitwise check: PASS on the local CUDA build** — two identical greedy requests byte-equal.
  (Must be RE-VERIFIED on the rented GPU before the binding run; a failure there pre-downgrades claims
  to distributional per Fence 2.)
- Harness end-to-end through the live 7B: 8 proposals generated, sandbox-executed, admission-checked in
  ~30 s at k=8 (≈4 s/proposal round on the 1080).

### The split decision (evidence-backed, the E3/GEN-1 lesson applied)
The local 7B admitted **0/8** proposals on the EASIEST solver-correctness synthetic (extract-largest —
the task the GEN-1 DSL solves trivially). A binding probe on the local 7B would therefore land
`GEN4_CEILING_EMPTY` **confounded by model weakness**, wasting the lane's single pre-registered probe —
exactly the starvation-confound the ceiling-first discipline exists to avoid. Accordingly:
- **The 7B/1.5B are harness-tier only.** No binding receipt may cite them.
- **The binding probe is staged for the next offload compute bundle (hyperbolic.ai GPU rental).**
  Default binding-model candidate: **Qwen2.5-Coder-32B-Instruct GGUF Q4_K_M** (~19 GB — fits a rented
  24 GB 4090; comfortable on A100/H100), k = 16, ctx 16384; the exact model file is sha256-pinned at
  fetch ON the rented box and recorded in the manifest, followed by the bitwise re-check, a capped
  1-task smoke, then the binding run (~155 instances; projected hours-scale, machine-dependent).
- The runner is portable by construction (llama.cpp + Python stdlib + the repo's v1 loader); the
  offload carries: the repo checkout (or the `docs/prereg/arc` + register + E3 fingerprint file subset),
  the ARC-AGI-2 training split, and the pinned GGUF.

### Staged binding command (offload box; adjust paths, record everything in the manifest)
```bash
python docs/prereg/arc/gen4_lm_proposer_probe.py \
  --data-dir <ARC-AGI-2/data> --register docs/prereg/arc/P0_TASK_REGISTER_EXPANDED_FOR_FIBERS.csv \
  --split-mode sha256_expansion --model-path <pinned-32B.gguf> --k 16 --ctx 16384 \
  --progress --allow-dirty --out results/arc/gen4-lm-proposer-ceiling-probe
```
No binding run, no verdict by this amendment. The probe executes only on the rented box after its
on-box bitwise re-check + smoke are recorded in a follow-up sub-amendment.
