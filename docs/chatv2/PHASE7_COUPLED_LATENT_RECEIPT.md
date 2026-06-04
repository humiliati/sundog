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

## 4. Coupled probe — RESULT (2026-06-03): `closure_washed_by_readout` (partial)

Training landed clean — all 3 gen learned (`eval_loss` 0.476 / 0.481 / 0.488 < 0.673),
twins saved, no `closure_void`. Probe `coupled_probe.json` (~22 min). Per seed:

| seed | func(u) k=3→k=7 | nf_func | **k_func** | state(z) @k=7 | **k_state** |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.616 → 0.620 | 0.528 | **None** | 0.721 | 7 |
| 1 | 0.650 → 0.656 | 0.547 | **None** | 0.743 | 7 |
| 2 | 0.642 → 0.648 | 0.536 | **None** | 0.604 | None |

**Verdict: `closure_washed_by_readout`.** The pre-registered data-level bracket
(`k_func=3 ≪ k_state=none`, u at 0.757) did **not** survive the trained-body readout
to the frozen 0.70 bar: `k_func = None` on all three seeds.

**Honest nuance — the wash is PARTIAL, not total.** `func(u)` caps at **0.62–0.66,
clearly above the selection-corrected null (~0.53)** — so the hidden source IS
partially represented in the body; the coupling is genuinely there. The clean
`u_null` control (func ≈ 0.50, §3) certifies this ~0.63 as **real above-null closure
signal**, not selection optimism. Readout noise (~0.8/latent) compounding with the
coupling noise (0.25) drops `u` recovery from 0.757 (clean data) to ~0.63 — above the
null, **below the registered 0.70 threshold.**

**Partial inversion:** at high `k` the individual states reach their noise ceiling
(`1−p = 0.75`) by direct multi-shadow readout (`k_state = 7` on seeds 0/1), so the
trained-body bracket did not merely wash — it partially **inverted** (state crosses
the bar before `u` does). Seed 2 crosses neither.

**No retune (goal-post discipline).** The 0.70 threshold and `p_noise = 0.25` were
frozen pre-run; the wash is read honestly against them. A *separate, newly
pre-registered* lower-noise cell (Phase 7b, e.g. `p_noise ≈ 0.15`) could let `func(u)`
clear 0.70 under readout noise — but that is a new cell, **not** a loosening of this
one. The frozen Phase-7 verdict stands: **`closure_washed_by_readout`.**

## What the lane learned

The coupled toy did what it was built to do: it **located a real closure** (data-level
`k_func=3`, and partial above-null recovery in the trained body) where the uncoupled
Phase-2 toy had none — confirming "closure is a coupling phenomenon." But the registered
NSE-like bracket **did not clear the bar through a trained body's readout** at this noise
level. So the lane's first shot at a positive lands as an **honest partial**: the
mechanism is present and measurable, the registered threshold is not met, and the
instrument correctly reported both — including the clean `u_null` control that makes the
above-null signal trustworthy. No promotion; the only path to a clean positive is a new
lower-noise pre-registered cell, kept explicitly separate from this frozen one.

## 5. Tier

Toy closure on a *designed* coupled substrate — **not** a real-LLM or NSE claim.
Promote-gate R1 not met. This is the lane's first phase pointed at a *positive*; the
honest outcome is whatever the trained-body probe returns against the pre-registered
bracket and the `u_null` control.
