# Non-Sovereignty Released Task Family Spec

Status: **DRAFT - packaging/release gate, not a new result.** Opened 2026-06-28
from [`../NON_SOVEREIGNTY_PAPER_SPINE.md`](../NON_SOVEREIGNTY_PAPER_SPINE.md)
gap 1. The NS/H2 harness already exists; this spec defines what must be frozen,
documented, and smoke-tested before the task family can be cited as an
independently runnable public artifact.

The release is intentionally narrow: a small continuous-control benchmark for
testing whether an authority bound, rather than role separation, accounts for
competence, corrigibility, non-sovereignty, and safe-interruptibility.

---

## 0. Release Claim

> A clean checkout can reproduce the task definitions, admission gates, fixed
> controls, authority/corrigibility metrics, and table-generation path for the
> four-axis non-sovereignty result without local secrets, private paths, or
> manual notebook state.

This is a **task-family release**, not a pretrained-model zoo. Long PPO bindings
may remain operator-run, but the exact commands, seeds, gates, and readback files
must be public and deterministic.

---

## 1. Public Surface

The released artifact should expose only the following surfaces.

### Core environment

- `scripts/h2-forked-task.mjs` - forked field/reward task, analytic controls.
- `scripts/ns1-shutdown-task.mjs` - shutdown wrapper, arbiter-authority cap,
  review-band endogenous signal, partial-unavoidability floor.
- `training/mesa/h2_forked_task.py` - Python mirror.
- `training/mesa/ns1_shutdown_task.py` - Python shutdown mirror.

### Runners

- `scripts/mesa-ns1-shutdown-admission.mjs`
- `scripts/mesa-ns1c-cap-validity.mjs`
- `scripts/mesa-ns2-admission.mjs`
- `scripts/mesa-ns1c-binding-eval.mjs`
- `scripts/mesa-ns2-binding-eval.mjs`
- `scripts/mesa-ns1c-aggregate.mjs`
- `scripts/mesa-ns2-aggregate.mjs`
- `training/mesa/train_ns1_shutdown.py`

### Operator launchers

- `scripts/mesa-ns1-b-binding.ps1`
- `scripts/mesa-ns1c-binding.ps1`
- `scripts/mesa-ns2-0-admission.ps1`
- `scripts/mesa-ns2-b-binding.ps1`

### Result docs to regenerate

- `docs/mesa/NS1_0_SHUTDOWN_CHANNEL_ADMISSION_RESULTS.md`
- `docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md`
- `docs/mesa/NS1_C0_CAP_VALIDITY_RESULTS.md`
- `docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md`
- `docs/mesa/NS2_0_ADMISSION_RESULTS.md`
- `docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md`

Anything outside this list is internal provenance unless explicitly promoted.

---

## 2. Package Layout

The release should add a public entry directory:

```text
released/non-sovereignty/
  README.md
  REPRODUCE.md
  TASK_SPEC.md
  METRICS.md
  MANIFEST.json
```

`README.md` gives the one-screen claim and install path.
`REPRODUCE.md` lists smoke, probe, and binding commands.
`TASK_SPEC.md` freezes the environment and cells.
`METRICS.md` freezes `Corr_k`, `Sov_opt`, `band_avoidance`,
`interrupt_avoidance`, `Delta_cap`, and `Delta_role`.
`MANIFEST.json` records file hashes for the public scripts/docs above.

No large result directories or local machine paths belong in the release
directory. Binding outputs stay under `results/` and are referenced by command.

---

## 3. Admission Gates

Release status is `RELEASE_READY` only if all gates pass.

1. **Clean checkout.** `npm install` and the Python module imports work on a clean
   clone without local credential files or hidden model paths.
2. **JS/Python parity.** For the registered seeds and cells, JS and Python mirrors
   match on:
   - initial state after reset;
   - field/reward proposals;
   - shutdown/review-band/floor signal timing;
   - terminal metrics for fixed controls.
   Tolerance: exact for discrete fields, `<= 1e-9` for floats after rounding.
3. **Fixed-control admission.** The public admission commands reproduce the
   fixed-control gates for H2/NS1/NS2.
4. **Cap-validity smoke.** `Sov_opt <= kappa` for the action-ball cap and
   `Corr_k | triggered = 1` for the structural override on the public seed slate.
5. **Table regeneration.** Aggregators regenerate the four-axis result tables
   from JSON/CSV artifacts without notebooks or manual edits.
6. **Runtime declaration.** Every command is labelled as one of:
   - `smoke` (<10 min);
   - `probe` (<1 h expected, operator-run);
   - `binding` (multi-hour, operator-run, resumable).
7. **License decision.** The released directory has an explicit license statement.
   The repo root is currently `UNLICENSED`; a public benchmark release needs an
   intentional license choice before external citation.

Failure of gates 1-5 is `RELEASE_BLOCKED`. Gate 7 is `LEGAL_BLOCKED`, not a
scientific failure.

---

## 4. Smoke Commands

These are the intended CI/local smoke commands after packaging. They must stay
under the repository's ~10-minute rule.

```powershell
npm install
npm run mesa:ns1:admission
npm run mesa:ns1c:cap-validity
npm run mesa:ns2:admission -- --smoke
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns2-b-binding.ps1 -PreflightOnly
```

If a command does not support the shown smoke/preflight mode, add the mode before
release. The release smoke is allowed to verify the path and fixed controls; it
is not required to rerun 512-update PPO.

---

## 5. Binding Reproduction Commands

Binding commands are staged, not CI. They must be copied into `REPRODUCE.md`
with wall-clock estimates from the last local run.

```powershell
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns1-b-binding.ps1
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns1c-binding.ps1
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns2-0-admission.ps1
powershell -ExecutionPolicy Bypass -File scripts/mesa-ns2-b-binding.ps1
```

Required readback:

- `docs/mesa/NS1_B_CORRIGIBILITY_BINDING_RESULTS.md`
- `docs/mesa/NS1_C_ARBITER_AUTHORITY_CAP_RESULTS.md`
- `docs/mesa/NS2_0_ADMISSION_RESULTS.md`
- `docs/mesa/NS2_B_UNIFIED_BOUND_RESULTS.md`
- pooled JSON summaries under `results/mesa/non-sovereignty/`

---

## 6. Public Result Boundary

The release may claim:

- the task family is reproducible from source;
- the admission gates are runnable without private state;
- the binding commands are exact and resumable;
- the four-axis tables regenerate from recorded artifacts.

The release may not claim:

- foundation-model transfer;
- deployed-agent safe interruptibility;
- safety under presider-input tampering;
- role-separation support.

---

## 7. Work Items

1. Add `released/non-sovereignty/` docs and manifest.
2. Add or repair `--smoke` / `-PreflightOnly` paths where the release smoke needs
   them.
3. Add a JS/Python parity smoke for `ShutdownForkedFieldEnv`.
4. Add `npm run mesa:ns:release-smoke` once the smoke path is stable.
5. Record measured runtimes in `REPRODUCE.md`.
6. Pick the release license.

---

## 8. Versioning

- `v0` (2026-06-28): defines release boundary, public files, gates, smoke
  commands, binding commands, and remaining packaging work.
