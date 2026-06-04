# Chat-v2 Phase 7 — Coupled-Latent Closure Receipt (PARTIAL)

> 2026-06-03. Execution of the frozen `PHASE7_COUPLED_LATENT_SPEC.md`. **Status:
> PARTIAL** — generator built + validated, de-confound PASS, `u_null` negative
> control banked; the 3-seed **training is STAGED (operator runs it)** per the
> >10-minute timing rule, and the coupled probe runs once the bodies land. Toy;
> unpromoted; promote-gate R1 not met regardless of outcome.

## 1. Build + pipeline validation (DONE)

- `--latent coupled` generator added to `scripts/chatv2_phase0_bodyresist.py`
  (`_gen_coupled`, `_COUPLE_A`, Cfg `m=3` / `p_noise=0.25`, `u` threaded through
  `extract_body` → npz). Probe `scripts/chatv2_phase7_probe.py`.
- **De-confound pre-check PASS:** input-probe `0.502 ≤ 0.60` at `H=8` — the coupled
  latents stay input-undecodable (coupling is among latents via shared `u`, not in
  the per-channel parity encoding).
- **Pipeline smoke (300 steps, seed 0):** train → `extract_body` → npz save with
  keys `bodies (3000,4,8,192)`, `z (3000,8)`, **`u (3000,3)`**, `meta` — confirmed.
  Gen UNLEARNED at 300 steps (expected; seedstab needs 6000) → UNLEARNED guard
  correctly skipped the twin. Wall-clock 154 s incl. measure.

## 2. Training — RUNNING (background task `bwxr1y4lo`, started 2026-06-03)

300-step smoke ⇒ ~0.25–0.3 s/step net ⇒ **~25–40 min gen + twin per seed ⇒ ~1.5–3 h
for 3 seeds.** Over the 10-minute inline bar → launched as a **background task**
(3 seeds sequential on the one GPU; `rm -rf` each seed dir first for idempotency).
Equivalent PowerShell for an operator re-run:

```powershell
$PY = "C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe"
$S  = "C:/Users/hughe/Dev/sundog/scripts/chatv2_phase0_bodyresist.py"
foreach ($seed in 0,1,2) {
  & $PY $S --mode full --stage all --latent coupled --fair-readout --h-sweep 8 `
    --d-model 192 --bits-per-channel 24 --delta 0.45 --arity 2 --p-noise 0.25 `
    --max-steps 6000 --min-steps 3000 --patience 10 --seed $seed `
    --out "C:/Users/hughe/Dev/sundog/results/chatv2/phase7-coupled-latent/seed$seed"
}
```

- **Read-back paths:** `results/chatv2/phase7-coupled-latent/seed{0,1,2}/bodies/H8_gen.npz`
  (+ `H8_twin.npz`, `manifest.json`). Each gen npz must carry the `u` key and
  `manifest.json` `eval_loss < 0.673` (learned) before the probe is meaningful.
- **Resume safety:** each seed writes its own `--out` dir and is independent; on
  interruption, re-run only the missing seed (idempotent — overwrites its dir).
  `save_ckpt=True` also writes `ckpt/H8_gen.pt`.
- **De-confound gate is automatic:** each run re-runs the input-probe pre-check and
  ABORTs to `precheck_failed.json` if any `z_i` becomes input-decodable.

## 3. Negative control — `u_null` on uncoupled Phase-2 bodies

Frozen independent target `u_null ~ Bernoulli(0.5)^(N×3)`, `default_rng(7007+seed)`,
probed on the Phase-2 pair-XOR bodies with the **same runner**. Expected
`k_func = none` (the runner must not manufacture hidden-source closure on a target
independent of the body). Command:
`python scripts/chatv2_phase7_probe.py --mode control`.

> **Result (PASS, 2026-06-03):** `k_func = None` on all three seeds — the runner
> does **not** manufacture hidden-source closure on an independent target.
> `func(u_null) ≈ 0.49–0.51` across every `k` (chance), and the selection-corrected
> floor `nf_func ≈ 0.51` is itself at chance — correctly far below the 0.73 real-`z`
> floor, because `u_null` is independent of the body so best-of-254 selection earns
> nothing. `k_state = None` too (omitted-`z` label-accuracy, consistent with Phase 2).
> `control_probe.json`; wall-clock 16.5 min. **The load-bearing control passes** —
> a future coupled `closure_confirmed` is credible only against this clean
> `k_func = none`, and that condition is now met.

## 4. Coupled probe — runs when bodies land

`python scripts/chatv2_phase7_probe.py --mode coupled` →
`coupled_probe.json` (**~16 min, background** — same machinery as the control).
Targets: `k_func` = hidden source `u` (mean held acc ≥ 0.70, selection-corrected);
`k_state` = omitted `z_j` mean held **label accuracy** ≥ 0.70.

**Pre-registered prediction (data-level):** `k_func = 3 ≪ k_state = none≤7`.

| branch | condition |
| --- | --- |
| `closure_confirmed` | `k_func` exists and `< k_state` (or `k_state` none) in the trained body; `u_null` control = none |
| `closure_washed_by_readout` | data bracket holds but trained-body `func(u)` never clears the null |
| `closure_absent` | no `k_func` even in the realised cells |
| `closure_void` | de-confound / UNLEARNED / control contract fails |

A `closure_confirmed` is only credible **with** the §3 `u_null` control showing
`k_func = none` on the same runner.

## 5. Tier

Toy closure on a *designed* coupled substrate — **not** a real-LLM or NSE claim.
Promote-gate R1 not met. This is the lane's first phase pointed at a *positive*; the
honest outcome is whatever the trained-body probe returns against the pre-registered
bracket and the `u_null` control.
