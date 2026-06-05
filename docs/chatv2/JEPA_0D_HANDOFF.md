# JEPA-0D Accumulator Unblock Handoff

> 2026-06-05. Internal/withheld handoff for the next JEPA lane builder.
> Resume target: build the smallest JEPA-native accumulator/count substrate, starting with
> a model-free de-confound preflight. Do not run a full model battery until the preflight
> and spec are locked.

## TL;DR

The coupled-parity JEPA Phase 0 is closed as `blocked_by_unfaithful_jepa`.

The important result was not a collapse bug. JEPA trained, stayed healthy, and showed the
predicted lower flip-conditioned noisy-bit read, but it did not keep the shared source
strongly enough:

```text
full-input read:      JEPA u_det = 0.256
masked-context read:  JEPA u_det = 0.368
bar:                  u_det >= 0.70
z_flip_gap:           +0.208 directional
collapse:             false, effR ~= 68
```

So the next move is not "richer dataset" in the broad sense. The next move is a richer
functional: an accumulator/count-like hidden state that is naturally useful to a predictive
JEPA objective.

## Current Branch

Use this as the source of truth:

- Parent: `docs/SUNDOG_V_JEPA.md`
- Executed parity spec: `docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md`
- Parity result log: `docs/chatv2/JEPA_PHASE0_NOISE_CARRY_RESULTS.md`
- Runner to reuse: `scripts/jepa_phase0_noise_carry.py`
- Smoke receipt: `results/chatv2/jepa-phase0-noise-carry/smoke.json`

Verdict to carry forward: `blocked_by_unfaithful_jepa`.

Do not reopen the parity toy by loosening `u_det`, adding nonlinear probes, or running the
capacity battery. The phase branched before capacity because the functional-retention control
failed.

## Fork Decision

The three old options were:

- continue JEPA on the parity toy;
- jump to a richer substrate;
- build an emergent/accumulator functional.

Take the third path first.

Reason: a broad richer substrate can add complexity without fixing the failure mode. The
parity source `u` was arbitrary from JEPA's point of view; the encoder could match target
embeddings without isolating it. An accumulator/count state is different because masked or
future structure can be made to depend on the running state. That gives JEPA a fairer,
more native reason to keep the functional.

## JEPA-0D Question

> Can a small JEPA keep a hidden accumulator state `u_t` while dropping per-step private noise?

This is still R1 toy work. It does not speak about real I-JEPA/V-JEPA, world models, or
"more than we know."

## Candidate Substrate

Build a sequential generator with three properties:

1. Raw observations do not linearly reveal the accumulator.
2. A model that computes local parity ticks and integrates context can linearly read the
   accumulator from its body.
3. Predicting masked/future structure genuinely benefits from tracking the accumulator.

Proposed shape:

```text
hidden event:       e_t in {0,1}
hidden count:       u_t = bounded_sum(e_1 ... e_t), u_t in {0 ... K}
observed tick:      parity-channel encoding of e_t, plus private noise x_t
count emission:     parity-channel encoding of g(u_t) or g(u_{t-1})
JEPA objective:     masked latent-channel embedding prediction
noise target:       observed noisy channel, audited by flip-conditioned z/read analog
```

Pins to preserve unless the preflight forces a re-pose:

- Use bounded count, not modulo count. A cyclic count adds circular-embedding ambiguity and
  muddies the linear `u_det` read.
- Encode both event and count-dependent emissions through parity-channel style encodings,
  reusing the Phase-0/Phase-7 de-confound idea.
- Include count-dependent emissions so JEPA cannot succeed by ignoring `u_t`.
- Keep per-step private noise as the eventual discard target.

Open design knobs for the preflight/spec:

- count cap `K`;
- sequence length / tick count;
- event probability;
- count-dependent function `g`;
- number of parity tuples per event/emission;
- whether `u_t` is read at every step or only at selected terminal/checkpoint positions.

## Mandatory Gate Order

Do not reorder these.

1. **Model-free de-confound preflight.**
   Raw observation linear probe to `u_t` must be near floor.

2. **Information-present oracle.**
   A non-learned sufficient parser that decodes parity ticks and sums them must recover
   `u_t`. This proves the functional is present but not linearly leaked.

3. **GEN positive control.**
   A trained GEN body must visibly carry observed noisy state. This catches dead reads.

4. **JEPA functional-retention gate.**
   JEPA `u_det(accumulator) >= 0.70`. This is where parity died. If it fails again, shelve.

5. **Only after gate 4: noise-discard contrast.**
   Carry over the repaired flip-conditioned read shape:
   train `body -> observed noisy channel` on all rows, then score on held-out flip/noise
   subset. GEN should be high; JEPA should be lower if it denoises.

6. **Only after gate 5: capacity sweep.**
   Reuse `{128,256} x 3 seeds`, with `d=256` as the chatv2 deflation point.

## Suggested Verdict Tree

Use exact names later in the spec, but this is the intended shape:

| branch | condition | meaning |
| --- | --- | --- |
| `blocked_by_deconfound_leak` | raw input linearly reads `u_t` | substrate is confounded; re-encode before any model |
| `blocked_by_absent_functional` | oracle cannot recover `u_t` | task does not actually contain the functional |
| `blocked_by_gen_control` | GEN body fails observed-state positive control | read or training not live |
| `blocked_by_unfaithful_jepa` | JEPA collapses or `u_det < 0.70` | JEPA did not keep the functional; shelve |
| `blocked_by_flip_readout` | flip/noise read support or GEN positive control fails | discard read not interpretable |
| `blocked_by_capacity` | gap exists at low width but dies at `d=256` | capacity artifact, not JEPA principle |
| `jepa_accumulator_discard_confirmed` | `u_det` and noise-discard gap survive | toy-tier JEPA-native positive |
| `jepa_not_distinguished_from_gen` | controls pass but gap absent | no selective discard on this substrate |

## First Build: Model-Free Preflight

Create a small script first, before model training:

```text
scripts/jepa_0d_accumulator_preflight.py
```

It should:

- generate the accumulator dataset;
- write a manifest with all frozen generator knobs;
- run raw-input linear probes to `u_t`, `e_t`, and count-dependent emissions;
- run a deterministic oracle parser to recover `u_t`;
- report class base rates and support;
- write JSON/CSV receipts under `results/chatv2/jepa-0d-accumulator-preflight/`.

Expected command:

```powershell
python scripts/jepa_0d_accumulator_preflight.py --out results/chatv2/jepa-0d-accumulator-preflight
```

This should be cheap enough to run inline. If it is not cheap, measure and stage per the
repo's ~10-minute rule.

Suggested preflight bars:

- raw linear `u_det <= 0.10` or accuracy within 0.05 of majority baseline;
- oracle `u_t` recovery >= 0.95;
- no class/support starvation for the planned probe targets.

Adjust exact bars in the preflight/spec if the generated target is ordinal rather than
categorical, but pin them before running.

## Second Build: JEPA-0D Spec

Only after preflight passes, draft:

```text
docs/chatv2/JEPA_0D_ACCUMULATOR_SPEC.md
```

Follow the house shape from:

```text
docs/chatv2/JEPA_PHASE0_NOISE_CARRY_SPEC.md
```

Carry over these locked JEPA mechanics:

- 50% latent-channel mask;
- 2-layer predictor at `d_model`;
- EMA target encoder, tau = 0.99;
- VICReg 25/25/1, gamma = 1.0;
- collapse guard: std >= 0.10 on >=90% dims, eff-rank >= max(8, 0.05*d);
- JEPA masked-context average read, `mask_reads=8`;
- repaired flip-conditioned noisy-bit read as the discard metric;
- `u_det >= 0.70` as the functional-retention control.

Tier: R1 toy. Forbidden language: real JEPA, world models, AGI, R2/R3, "more than we know."

## Third Build: Runner

Likely script:

```text
scripts/jepa_0d_accumulator.py
```

Reuse from `scripts/jepa_phase0_noise_carry.py`:

- `TinyGPT` / GEN training machinery via `chatv2_phase0_bodyresist.py`;
- JEPA train loop;
- masked-context average read;
- collapse guard;
- JSON cleaning;
- flip-conditioned read pattern.

New work:

- accumulator generator;
- multiclass or ordinal `u_t` probe;
- raw-input de-confound probes;
- oracle parser receipts;
- count-dependent emission target;
- mapping from sequence positions to latent/channel read positions.

Implementation caution: if `u_t` is multiclass, define `u_det` as

```text
u_det = (heldout_acc - majority_base) / max(1 - majority_base, 1e-9)
```

If using scalar regression/R2 instead, explicitly justify and pin it in the spec before any run.

## Smoke and Battery

Smoke command should be staged after spec lock if it exceeds the inline rule:

```powershell
python scripts/jepa_0d_accumulator.py --smoke --out results/chatv2/jepa-0d-accumulator-smoke
```

Smoke must stop after:

- de-confound replay;
- GEN positive control;
- JEPA collapse guard;
- JEPA `u_det` gate;
- one directional/noise-read sanity.

Only if smoke clears `u_det >= 0.70`:

```powershell
python scripts/jepa_0d_accumulator.py --out results/chatv2/jepa-0d-accumulator-lock
```

Do not run the lock battery if `u_det` fails.

## Known Footguns

- Do not train/evaluate direct `x_i` probes. The parity lane proved XOR-derived `x_i`
  is the wrong linear target.
- Do not train the flip read on flip rows only. Train on all rows, score on flips.
- Do not use a modulo/cyclic accumulator unless the spec defines a circular read. Prefer
  bounded count.
- Do not let the count-dependent emission leak `u_t` linearly in raw input.
- Do not soften `u_det >= 0.70` after seeing a directional noise gap.
- Do not run the capacity sweep before JEPA passes the `u_det` control.
- Avoid the background PowerShell `2>&1 | Tee-Object` pattern that killed a JEPA smoke.
  Use a clean launch and verify the process is alive.

## Environment Notes

GPU Python, if needed:

```text
C:/Users/hughe/.venvs/sundog-gpu/Scripts/python.exe
```

Repo rule: anything expected over ~10 minutes is operator-staged with exact commands and
wall-clock estimates. The previous matched JEPA smoke was about 15.5 minutes.

## Handoff Definition Of Done

This handoff is complete when the next builder can answer:

- what failed in parity JEPA;
- why accumulator is the next fair substrate;
- what to build first;
- what gates can shelve the lane;
- which existing runner/spec pieces are safe to reuse;
- what not to run before `u_det` clears.

---

*Sundog Research Lab - JEPA-0D handoff. Internal; withheld in `docs/chatv2/`.*
