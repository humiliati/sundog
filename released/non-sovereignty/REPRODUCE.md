# Reproduce The Non-Sovereignty Task Family

All commands are run from the repository root.

## Smoke

Expected runtime: under 10 minutes on the project CPU-only machine. Broadened
(NS-1/2/3/4) local run: **PASS** in 6.659 s.

```powershell
npm install
npm run mesa:ns:release-smoke
```

Readback:

- `results/mesa/non-sovereignty/release_smoke/summary.json`
- `results/mesa/non-sovereignty/release_smoke/parity.json`
- `results/mesa/non-sovereignty/release_smoke/ns1_admission.json`
- `results/mesa/non-sovereignty/release_smoke/ns1c_cap_validity.json`
- `results/mesa/non-sovereignty/release_smoke/ns2_unified_bound.json`
- `results/mesa/non-sovereignty/release_smoke/ns3_admission.json`
- `results/mesa/non-sovereignty/release_smoke/ns4_admission.json`

## Manifest

Regenerate after intentional release-surface edits:

```powershell
npm run mesa:ns:release-manifest
```

Check without rewriting:

```powershell
npm run mesa:ns:release-manifest -- --check
```

## Binding Reproduction

These commands are operator-run and resumable. They are not part of the smoke.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns1-b-binding.ps1
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns1c-binding.ps1
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns2-0-admission.ps1
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns2-b-binding.ps1
```

NS-3 learned-presider replication (train the frozen presider, then the binding):

```powershell
python -m training.mesa.train_ns3_presider --out results/mesa/non-sovereignty/ns3_presider/presider.json --kappa-max 0.4
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns3-b-binding.ps1
```

NS-4 conversion slate (no training; SB-0 scan, SB-1 planner, SB-4 options + cap payoff):

```powershell
node scripts/mesa-ns4-sb0-landscape-scan.mjs --review-temperature 0.03
node scripts/mesa-ns4-sb1-planner.mjs --review-temperature 0.03
node scripts/mesa-ns4-sb4-options.mjs --review-temperature 0.03
```

Primary readbacks:

- `docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md`
- `docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md`
- `docs/mesa/NS2_0_ADMISSION_RESULTS.md`
- `docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md`
- `docs/mesa/NS3_B_BINDING_RESULTS.md`
- `docs/mesa/NS3_NS4_MANIPULATION_WALL_SYNTHESIS.md`
- `docs/mesa/NS4_SB4_OPTIONS_RESULTS.md` (the `CAP_DETERS_COMPETENT_SANDBAG` payoff)

Pooled JSON artifacts live under `results/mesa/non-sovereignty/`.

## Runtime Notes

All runtimes below were measured on the project machine (Intel Core i7-7820HK
@ 2.9 GHz, 8 threads, CPU-only; PyTorch CPU). Times scale with that budget.

### Smoke (CI-tier)

The broadened (NS-1/2/3/4) local release smoke measured (2026-06-30):

| Step | Runtime |
| --- | ---: |
| manifest check | 0.242 s |
| JS/Python parity | 0.440 s |
| NS-1 admission smoke | 0.243 s |
| NS-1-c cap-validity smoke | 4.678 s |
| NS-2 table regeneration | 0.176 s |
| NS-3 admission smoke | 0.447 s |
| NS-4 admission smoke | 0.432 s |
| total | 6.659 s |

The release smoke records fresh step runtimes in
`results/mesa/non-sovereignty/release_smoke/summary.json`.

### Binding (operator-run, resumable)

Wall-clock from the recorded `train-report.json` timing of the last local PPO
bindings (2026-06-30). Each launcher runs its child trainings sequentially, so
the launcher total is the sum; the per-run column is the median single training.
PPO is the dominant cost.

| Launcher | PPO updates | runs | per-run (median) | launcher total |
| --- | ---: | ---: | ---: | ---: |
| `mesa-ns1-b-binding.ps1` | 256 | 3 seeds | ~6.9 min | ~21 min |
| `mesa-ns1c-binding.ps1` | 256 | 9 (3 seed x 3 kappa) | ~6.8 min | ~61 min |
| `mesa-ns2-0-admission.ps1` | 512 | 3 | ~17 min | ~47 min |
| `mesa-ns2-b-binding.ps1` | 512 | 7 | ~9.9 min | ~68 min |
| `mesa-ns3-b-binding.ps1` | 512 | 5 | ~3.1 min | ~25 min |

The NS-3 learned presider trains in ~23 s before its binding
(`train_ns3_presider`, 250 epochs / 24k samples). The NS-4 conversion slate is
not PPO and runs in seconds: SB-0 ~0.4 s, SB-1 ~0.9 s, SB-4 ~4.5 s.

Total operator wall-clock to reproduce every PPO binding end-to-end is ~3.7 h on
this machine; runs are resumable from checkpoints, so they can be done in
batches.
