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
`results/mesa/non-sovereignty/release_smoke/summary.json`. Binding wall-clock
estimates should be refreshed from the launcher logs before external release.
