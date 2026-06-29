# Reproduce The Non-Sovereignty Task Family

All commands are run from the repository root.

## Smoke

Expected runtime: under 10 minutes on the project CPU-only machine. First green
local run: **PASS** in 3.129 s.

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

Primary readbacks:

- `docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md`
- `docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md`
- `docs/mesa/NS2_0_ADMISSION_RESULTS.md`
- `docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md`

Pooled JSON artifacts live under `results/mesa/non-sovereignty/`.

## Runtime Notes

The first green local release smoke measured:

| Step | Runtime |
| --- | ---: |
| manifest check | 0.102 s |
| JS/Python parity | 0.263 s |
| NS-1 admission smoke | 0.151 s |
| NS-1-c cap-validity smoke | 2.534 s |
| NS-2 table regeneration | 0.077 s |
| total | 3.129 s |

The release smoke records fresh step runtimes in
`results/mesa/non-sovereignty/release_smoke/summary.json`. Binding wall-clock
estimates should be refreshed from the launcher logs before external release.
