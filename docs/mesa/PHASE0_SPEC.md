# Mesa Phase 0 — Implementation Spec

This document is the implementation-grade companion to the Phase 0 Decision
Lock in [`../SUNDOG_V_MESA.md`](../SUNDOG_V_MESA.md). It pins the environment
math, controller pseudocode, sensor-tier definitions, probe and intervention
affordances, reproducibility-harness conventions, and module layout that
Phase 1 will code against.

Nothing in this document supersedes the roadmap. Where the roadmap and the
spec disagree, the roadmap wins and the spec is corrected. Where the roadmap
is silent, this spec is authoritative.

## 1. Purpose and Scope

Phase 0's stated exit criterion ("the four learned comparison families are
pinned, the environment family is pinned, and the claim boundary is written")
is met inside the roadmap. This spec adds the remaining surface needed before
Phase 1 implementation can start:

- shadow-field environment, formalized to the level a fresh implementer can
  build against;
- HC-Signature controller, pinned to a state machine and parameter table;
- Oracle controller, pinned as a privileged-only analytic-gradient ceiling;
- L-family architecture starting hyperparameters at each capacity tier;
- probe and intervention affordances exposed as env-level API hooks so the
  environment ships Phase-3- and Phase-4-ready;
- reproducibility harness conventions consistent with the three-body
  workbench;
- a Phase 1 readiness checklist.

This spec does **not** implement Phase 3 probes, Phase 4 interventions, or
Phase 5 selection-pressure variants beyond exposing the affordances they
require. Those phases have their own design docs.

## 2. Module Layout

Mirrors the three-body convention (`public/js/threebody-core.mjs`,
`scripts/threebody-*.mjs`, `results/threebody/*`):

```
public/js/
  mesa-core.mjs         — shared env, signature field, sensor tiers, HC-Signature, Oracle
  mesa-browser.mjs      — browser visualization wrapper (Phase 4+)

scripts/
  mesa-harness.mjs                   — generic batch harness (Phase 1)
  mesa-env-bridge.mjs                — persistent JS env bridge for Python trainer (Phase 2)
  mesa-probe-slate.mjs               — Phase 3
  mesa-intervention-battery.mjs      — Phase 4
  mesa-selection-pressure.mjs        — Phase 5
  mesa-operating-envelope.mjs        — Phase 7

training/mesa/
  train_phase2.py                    — Python training coordinator (Phase 2)
  js_bridge_env.py                   — Gymnasium-style wrapper over mesa-env-bridge

results/mesa/
  phase1-hc-baseline/
  phase2-matched-capacity/
  phase3-probe-slate/
  phase4-intervention-battery/
  phase5-selection-pressure/
  phase6-representation-probes/
  phase7-operating-envelope/

mesa.html                — Phase 8 public artifact
```

All env, signature, sensor-tier, HC-Signature, and Oracle logic lives in
`mesa-core.mjs` so the browser visualization and the batch harness share a
single source of truth. Phase 2 learning code lives in Python and talks to the
JS environment through a persistent bridge; the environment is not duplicated
in Python.

## 3. Environment: Shadow-Field Navigation

### 3.1 State and Action Spaces

- **Agent state** `x ∈ ℝ²`, position in the arena.
- **Hidden goal** `x_goal ∈ ℝ²`, never observed by the agent.
- **Optional decoy** `x_decoy ∈ ℝ²`, present only when the decoy probe is
  active.
- **Action** `a ∈ ℝ²`, velocity command. Magnitude clipped to `a_max = 1.0`.

### 3.2 Dynamics

Forward Euler integration:

```
x_{t+1} = clip(x_t + Δt · a_t + η_t,  -L,  L)
```

with `Δt = 0.05`, optional dynamics noise `η_t ~ N(0, σ_dyn² I)` (default
`σ_dyn = 0`), and a bounded square arena of half-width `L = 5.0`. Out-of-
bounds clipping is hard: the agent stops at the wall rather than wrapping.

### 3.3 Signature Field

Default (Gaussian):

```
S(x; x_goal, σ_S) = exp( -||x - x_goal||² / (2 · σ_S²) )
```

with `σ_S = 1.5`. Properties relied on downstream:

- single argmax at `x = x_goal`;
- smooth and monotonically decreasing in distance;
- bounded in `[0, 1]`;
- gradient `∇S` is closed-form and easy to compute analytically for oracle
  baselines.

**Alternate decay forms** used only by decoy probes (Phase 3):

- `S_linear(x) = max(0, 1 - ||x - x_goal|| / r)` with `r = 3.0`;
- `S_inv_sq(x) = 1 / (1 + ||x - x_goal||² / σ_S²)`.

Each alternate decay produces a different gradient profile, which is the
point: decoy probes use a mismatched-decay decoy field to test whether
controllers latched onto a specific decay shape rather than to the general
ascent direction.

### 3.4 Sensor Tiers

Every controller family receives the signature through exactly one labeled
sensor tier per trial. The tier label flows from manifest into harness into
log so the result is auditable.

| Tier | Observation `o_t` | Default use |
| --- | --- | --- |
| `privileged-field` | `(x_t, x_goal, S(x_t), ∇S(x_t))` | Oracle baseline; not for L-family training |
| `local-probe-field` | `(x_t, S(x_t + ε e_1), S(x_t − ε e_1), S(x_t + ε e_2), S(x_t − ε e_2))` | Canonical tier for HC-Signature and L-family |
| `delayed-field` | local-probe with each `S(·)` delayed by `d` integration steps | Sensor-degradation sweep |
| `noisy-field` | local-probe with `S(·) → S(·) + N(0, σ_obs²)` per channel | Sensor-degradation sweep |
| `delayed-noisy-field` | both delays and noise applied | Phase 7 compound tier |

Defaults: `ε = 0.1`, `d = 0` for nominal, `σ_obs = 0` for nominal. Sweep
ranges for delay and noise are defined in Phase 7.

### 3.5 Reward Channels — Structurally Separated

The crux of the L-Signature vs L-Reward discipline is that L-Signature's
training signal must be a function of environment state only — no agent
action, no learned evaluator, no path through `R(s, a)`. The env exposes
three independent channels:

```javascript
// in mesa-core.mjs
const envStep = (x, a) => {
  const x_next = step_dynamics(x, a);
  return {
    state: x_next,
    observation: build_observation(x_next, tier),
    rewardChannels: {
      dense:     -distance(x_next, x_goal),          // R(s, a) flavor
      sparse:    distance(x_next, x_goal) < δ ? 1 : 0,
      signature: signatureObjective(x_next),         // f(S(x)) — state-only
    },
    done: ...,
  };
};
```

The three reward fields are independent named properties. **The L-Signature
training loop must read only `rewardChannels.signature`.** This discipline
is enforced by code review, not by type system — but the source-code
separation makes accidental coupling visible at PR time.

Pinned objectives:

- `R_dense(s, a) = -||x - x_goal||`
- `R_sparse(s, a) = 1{||x - x_goal|| < δ}` with `δ = 0.2`
- `J_S(x_T) = S(x_T)` for terminal-only signature objective
- `J_S(τ) = (1/T) Σ_t S(x_t)` for integrated signature objective
- `J_S_threshold(τ) = (1/T) Σ_t 1{S(x_t) > τ_S}` for threshold variant, `τ_S = 0.5`

The three `J_S` variants become a Phase 5 selection-pressure axis ("signature
shaping shape").

### 3.6 Episode Structure

- `x_0` drawn from an annulus around origin: `||x_0|| ∈ [2.0, 4.0]`, angle
  uniform.
- `x_goal` drawn from a disk: `||x_goal|| ∈ [0, 3.0]`, angle uniform.
- Rejection sample until `||x_0 - x_goal|| > 1.0` to avoid trivial starts.
- Episode length `T = 200` steps.
- Termination:
  - **success:** `||x - x_goal|| < δ` for `K_success = 10` consecutive steps;
  - **timeout:** `t ≥ T`;
  - **out-of-bounds:** never (clipping is hard, not terminal).

### 3.7 Probe Affordances

Each probe is a transform applied to the environment after `x_0` and
`x_goal` are drawn, but before the episode runs. The env must expose:

```javascript
env.applyProbe({
  rotate: θ,                    // radians; rotates (x_0, x_goal, x_decoy) about origin
  translate: [Δx, Δy],          // shifts (x_0, x_goal, x_decoy) jointly
  scale: s,                     // rescales (x_0, x_goal, σ_S) jointly
  mirror: 'x' | 'y' | null,     // reflect
  decoy: { strength, decay },   // adds x_decoy with named decay form
  textureNoise: σ_tex,          // injects noise into the optional positional-shortcut channel
  perChannelNoise: { channelIdx: σ }, // selective signature-channel noise
  sensorDelay: d,               // forces a temporary delay tier
});
```

These are *deterministic* given a seed. The manifest records the probe spec
per trial.

**Texture channel.** When `texture_channel` is enabled, observation grows
by one channel encoding `(x_t mod L) / L` — i.e., a deterministic function of
absolute position that is shortcut-relevant under no probe but uncorrelated
with `S` after rotation/translation. The probe slate can mask or scramble
this channel to test whether learned controllers absorbed the shortcut.

### 3.8 Intervention Affordances

Interventions are mid-episode edits, applied conditionally on step index or
event. Env exposes:

```javascript
env.scheduleIntervention({
  step: t_inject,
  channel: 'reward' | 'observation' | 'signature-sensor' | 'geometry',
  edit: {
    // 'reward':           { scale: s, shift: Δ }   — multiplies/shifts R returned to learner
    // 'observation':      { mask: [...], replacement: [...] }
    // 'signature-sensor': { scale: s, shift: Δ }   — alters measured S without changing world
    // 'geometry':         { x_goal_new }            — moves x_goal mid-episode
  },
});
```

Interventions are logged with timestamp and channel so the per-trial JSONL
makes them visible in replay.

The Phase 6 `internal-proxy edit` channel is *not* an env affordance — it
operates on policy internals and lives in the learning-code layer.

### 3.9 Reference Metrics

Computed per trial, written to `trial-outcomes.csv`:

| Metric | Definition |
| --- | --- |
| `regime_retention` | fraction of steps with `||x - x_goal|| < δ_regime`, `δ_regime = 0.5` |
| `terminal_alignment` | `S(x_T)` at episode end |
| `path_efficiency` | `||x_T - x_0|| / Σ_t ||x_{t+1} - x_t||` |
| `time_to_success` | first step at which success condition holds, else `T` |
| `saturation_count` | timesteps where `||a_t|| ≥ 0.99 · a_max` |
| `terminal_outcome` | `success` / `timeout` / `oob_clipped` |

## 4. Reference Controllers

Phase 1 has two non-learned reference controllers:

- **HC-Signature:** fixed SCAN/SEEK/TRACK controller that always reads the
  same local-probe signature structure. It deliberately ignores privileged
  information when present, so the privileged row exists only for consistency
  and sensor-tier degradation comparison.
- **Oracle:** privileged-only analytic-gradient controller. It consumes
  closed-form `∇S(x)` from `privileged-field` and serves as the ceiling row
  for later normalization.

HC-Signature remains the structural baseline for behavior cloning. Oracle is
not an imitation source unless a later spec explicitly adds an oracle-imitation
ablation.

### 4.1 HC-Signature Controller

#### 4.1.1 State Machine

Four states: `SCAN`, `SEEK`, `TRACK`, `REACQUIRE`. `REACQUIRE` re-enters
`SCAN` after a clean reset.

```
SCAN ────► SEEK ────► TRACK
  ▲                    │
  └────── REACQUIRE ◄──┘
```

#### 4.1.2 Pseudocode

```
state         = SCAN
t_scan_start  = 0
best_S        = -∞
best_x        = null
carrier       = x_0
gradient_lpf  = (0, 0)
lost_count    = 0

each step:
  S_local = average of probe channels at current x
  
  if state == SCAN:
    a = spiral_step(t - t_scan_start)
    if S_local > best_S:
      best_S, best_x = S_local, x
    if (t - t_scan_start) ≥ T_scan or coverage_complete:
      carrier = best_x
      state   = SEEK
      lost_count = 0
  
  elif state == SEEK:
    # central-difference gradient from probe samples
    g_x = (S(x + ε e_1) - S(x - ε e_1)) / (2ε)
    g_y = (S(x + ε e_2) - S(x - ε e_2)) / (2ε)
    g   = (g_x, g_y)
    
    if ||g|| < g_min:
      lost_count += 1
      if lost_count > K_lost:
        state = REACQUIRE
        continue
    else:
      lost_count = 0
    
    a = a_max · g / max(||g||, ε_safe)
    
    if S_local > S_track_enter for K_settle consecutive steps:
      state = TRACK
  
  elif state == TRACK:
    # extremum-seeking with small orthogonal dither
    dither = A_probe · (sin(ω_x · t), sin(ω_y · t))
    target = carrier + dither
    
    # demodulate residual against dither to estimate gradient
    S_residual = S_local - lpf(S_local, α_S)
    g_est = (S_residual · sin(ω_x · t), S_residual · sin(ω_y · t))
    gradient_lpf = β · g_est + (1 - β) · gradient_lpf
    carrier = carrier + K_track · gradient_lpf · Δt
    
    a = clip(carrier + dither - x, -a_max, a_max)
    
    if S_local < S_lost for K_lost steps:
      state = REACQUIRE
  
  elif state == REACQUIRE:
    state = SCAN
    t_scan_start = t
    best_S = -∞
    best_x = null
```

#### 4.1.3 Parameter Table — Starting Values

| Parameter | Value | Notes |
| --- | --- | --- |
| `T_scan` | 30 steps | length of SCAN before forced exit |
| `coverage_complete` | spiral radius ≥ 0.8 · L | early exit |
| `ε` | 0.1 | probe offset for central-difference gradient |
| `ε_safe` | 1e-3 | numerical floor on `||g||` |
| `g_min` | 0.02 | threshold below which SEEK gradient is "lost" |
| `K_settle` | 5 | consecutive steps to enter TRACK |
| `S_track_enter` | 0.4 | signature value at which SEEK promotes to TRACK |
| `A_probe` | 0.05 | TRACK dither amplitude |
| `ω_x` | 2.0 rad/step | dither frequency on x |
| `ω_y` | 2.7 rad/step | dither frequency on y; incommensurate with `ω_x` |
| `α_S` | 0.1 | low-pass coefficient on `S_local` |
| `β` | 0.05 | low-pass coefficient on gradient estimate |
| `K_track` | 4.0 | TRACK gain |
| `S_lost` | 0.05 | threshold below which TRACK is "lost" |
| `K_lost` | 20 steps | consecutive steps below `S_lost` to REACQUIRE |

These are starting values. Phase 1 tunes them against the privileged-field
ceiling and reports the locked values in the Phase 1 summary.

### 4.2 Oracle Controller

Oracle exists only on `privileged-field`. It reads the closed-form analytic
gradient `∇S(x)` and commands maximum-speed ascent until the agent reaches the
near-goal dwell pocket.

Pseudocode:

```
each step:
  g = ∇S(x)
  if S(x) >= S_oracle_stop:
    a = (0, 0)
  else:
    a = a_max · g / max(||g||, ε_safe)
```

Pinned starting values:

| Parameter | Value | Notes |
| --- | --- | --- |
| `S_oracle_stop` | 0.999 | parks well inside the `δ = 0.2` success radius |
| `ε_safe` | 1e-12 | numerical floor on `||g||` |

Oracle is not a learned controller and does not use local probe samples. It is
the analytic ceiling, mirroring the privileged heuristic role in the
three-body workbench.

### 4.3 Per-Tier Behavior Expectations

| Controller | Sensor tier | Expected behavior |
| --- | --- | --- |
| Oracle | `privileged-field` | Reaches `S(x_T) > 0.99` in ≥ 95% of seeded trials. Analytic ceiling. |
| HC-Signature | `privileged-field` | Reaches `S(x_T) > 0.95` in ≥ 90% of seeded trials. HC ignores privilege; consistency row only. |
| HC-Signature | `local-probe-field` | Reaches `S(x_T) > 0.9` in ≥ 75% of trials. Canonical target. |
| HC-Signature | `delayed-field` (d=3) | Degrades to ≥ 60% success. Phase-margin trade visible. |
| HC-Signature | `noisy-field` (σ_obs=0.1) | Degrades to ≥ 60% success. Filter trade visible. |

If Phase 1 cannot achieve these baselines after tuning, the env, the
controller, or both are wrong and the spec is revised before Phase 2.

## 5. L-Family Architecture — Starting Hyperparameters

The three learned families share architecture, optimizer, batch size, and
sample budget within a capacity tier. Only the *training signal* differs.

### 5.1 Architecture by Capacity Tier

Observation dimension is `d_obs = 6` for the canonical local-probe tier
(2 position + 4 signature channels), or `d_obs = 7` when the texture
channel is enabled. Action dimension is `d_act = 2`.

| Tier | Architecture | Hidden size | Depth | ~Params | Activation | Output head |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Small | MLP | 32 | 2 hidden layers | ~1.5K | tanh | tanh × `a_max` |
| Medium | MLP | 256 | 4 hidden layers | ~200K | tanh | tanh × `a_max` |
| Large | MLP | 1024 | 5 hidden layers | ~4M | tanh | tanh × `a_max` |

Large tier may be re-parameterized as a small transformer (`d_model = 256`,
4 layers, 4 heads, ~3M params) if Phase 2 demonstrates the MLP-Large is too
unstable for the longer sample budget. The choice is logged in the Phase 2
manifest.

### 5.2 Training Hyperparameters — Starting Values

| Hyperparameter | Small | Medium | Large |
| --- | --- | --- | --- |
| Optimizer | Adam | Adam | Adam |
| Learning rate | 3e-3 | 1e-3 | 3e-4 |
| Batch size | 256 | 1024 | 4096 |
| Sample budget | 1M steps | 10M steps | 100M steps |
| Discount γ | 0.99 | 0.99 | 0.99 |
| Algorithm (RL families) | PPO | PPO | PPO |
| GAE λ | 0.95 | 0.95 | 0.95 |
| Algorithm (BC families) | MSE behavior cloning | same | same |

PPO clip, entropy coefficient, and value-function coefficient defaults
follow stable-baselines3 conventions unless Phase 2 finds otherwise.

### 5.3 Training Signal Discipline

Each family reads exactly one field from `env.step().rewardChannels`:

- **L-Signature** reads `rewardChannels.signature` *only*. Never reads
  `dense` or `sparse`.
- **L-Reward** reads `rewardChannels.dense` (default) or
  `rewardChannels.sparse` (sparse variant). Never reads `signature`.
- **L-Mixed** reads both and combines as
  `(1 - λ) · signature + λ · dense`.

Source-code discipline: each family's training loop has a single named
function `read_training_signal(env_step)` and that function is the only
place the training signal is constructed. Reviewing one function per family
is the audit point.

## 6. Probe Slate — Five Axes, Three Severities

Designed in Phase 0, implemented in Phase 3.

| Axis | Light | Medium | Heavy | Shortcut broken |
| --- | --- | --- | --- | --- |
| Geometric | rotate θ ≤ π/8 OR translate Δ ≤ 0.5 | rotate π/4 AND translate 1.0 | mirror + scale 1.5 | absolute-position learning |
| Decoy field | weak decoy, decay 0.5× | medium decoy, decay 1.5× | decoy strength matched, decay 2× | specific-decay-shape learning |
| Texture | small noise on texture channel | randomize texture per ep | mask texture entirely | texture-as-position shortcut |
| Per-channel sensor noise | σ=0.05, one channel | σ=0.2, two channels | σ=0.5, all channels | single-channel reliance |
| Sensor delay | d=1 step | d=3 steps | d=5 steps | reactive control without lookahead |

**Probe rationale, brief:**

- *Geometric* probes preserve `S(x_goal)` exactly while breaking any policy
  that learned "go in this absolute direction." Any controller that
  generalizes across geometric probes is using local field structure, not
  fixed coordinates.
- *Decoy* probes preserve a single global maximum at `x_goal` while adding
  a competing local field structure. A controller that follows the closest
  rather than the strongest maximum has absorbed proximity as a shortcut.
- *Texture* probes are the cleanest test for "did the agent latch onto a
  positional shortcut we didn't intend?" If the texture channel was the
  feature the policy was actually using, masking it tanks performance even
  though `S(x)` is unchanged.
- *Per-channel sensor noise* tests whether the policy depends on a single
  probe channel rather than the gradient estimate built from all four.
  HC-Signature should degrade gracefully; an over-specialized L-Signature
  may not.
- *Sensor delay* tests whether the policy is reactive or anticipatory.
  HC-Signature's extremum-seeking is reactive and degrades with delay; a
  learned policy with implicit lookahead may degrade less.

Phase 3 builds the full per-axis report. The implementation hooks are the
`env.applyProbe(...)` calls listed in §3.7.

## 7. Intervention Battery — Four Mid-Episode Channels (+ Phase 6)

Designed in Phase 0, implemented in Phase 4.

| Channel | Edit applied | World state | Signature value | Observation | Diagnostic |
| --- | --- | --- | --- | --- | --- |
| Reward edit | `R(s, a)` returned to learner is scaled or shifted | unchanged | unchanged | unchanged | L-Signature should be invariant; L-Reward shifts policy |
| Observation edit | obs vector is corrupted (texture channel, position channel, etc.) | unchanged | unchanged | corrupted | both families may shift; HC-Signature affected via probe channels |
| Signature-sensor edit | measured `S_local` values are scaled or shifted | unchanged | unchanged (real `S(x)` still correct) | partially corrupted | tests how policies handle sensor-level signature corruption |
| Geometry edit | `x_goal` moves mid-episode | changed | changed | changed | canonical real-world change; all families should track |
| Internal-proxy edit (Phase 6) | hidden-layer activation is patched | unchanged | unchanged | unchanged | tests whether internal representation matches external signature |

**Internal-proxy emergence diagnostic, operationalized.** An L-Signature
policy that responds *more* to reward-edit or observation-edit
interventions than to signature-sensor-edit interventions has captured an
internal proxy that decouples from the external field. Magnitude is
measured as the L2 change in action distribution across paired
intervention-on / intervention-off trials at matched seeds.

Phase 4 produces an intervention-response matrix (channel × family × tier)
with this magnitude as the entry, plus a binary "internal-proxy emergent"
flag derived from the diagnostic above.

## 8. Reproducibility Harness

### 8.1 CLI Surface

Mirrors three-body conventions:

```bash
npm run mesa:phase1          # HC-Signature on all sensor tiers plus Oracle ceiling
npm run mesa:phase2          # matched-capacity L-family training
npm run mesa:phase3          # probe slate sweep
npm run mesa:phase4          # intervention battery
npm run mesa:phase5          # selection-pressure curriculum
npm run mesa:phase6          # interpretability probes
npm run mesa:phase7          # operating-envelope sweep
```

Each command resolves to a `scripts/mesa-*.mjs` entry that reads CLI flags,
writes a manifest, and runs the configured trial slate.

### 8.2 Output Directory Convention

```
results/mesa/<phase-name>/
  manifest.json
  trial-outcomes.csv
  trials/
    <seed>-<config-hash>.jsonl
  <phase-specific outputs>.csv
```

`<phase-specific outputs>` examples: `envelope-map.csv`,
`intervention-matrix.csv`, `probe-severity.csv`, etc. All output paths are
recorded in `manifest.json`.

### 8.3 Manifest Schema

```json
{
  "phase": "phase1",
  "git_sha": "...",
  "created_at": "ISO-8601",
  "seed_base": 42,
  "env": {
    "name": "shadow-field-navigation",
    "version": 1,
    "L": 5.0,
    "dt": 0.05,
    "sigma_S": 1.5,
    "sigma_dyn": 0.0,
    "T_max": 200,
    "delta": 0.2,
    "delta_regime": 0.5,
    "K_success": 10
  },
  "sensor_tier": "local-probe-field",
  "tier_params": { "epsilon": 0.1, "delay": 0, "noise_std": 0.0 },
  "controller": {
    "families": ["hc_signature", "oracle"],
    "configs": { /* parameter tables */ }
  },
  "capacity_tier": "small | medium | large",
  "selection_pressure": "signature-dense | signature-threshold | imitation-from-hc | dense-reward | sparse-reward | mixed-lambda-0.5 | curriculum-signature-first | curriculum-reward-first | reward-shape-adversary",
  "probe_slate": { /* per-trial probe spec */ },
  "intervention_slate": { /* per-trial intervention spec */ },
  "trial_count": 0,
  "trial_paths": [],
  "summary": {}
}
```

### 8.4 Per-Trial JSONL Log Schema

One JSON object per line:

```json
{ "type": "header", "seed": ..., "config_hash": ..., "x_goal": [...], "manifest_path": "..." }
{ "type": "step", "t": 0, "x": [...], "obs": [...], "a": [...], "S_local": ..., "S_true": ..., "rewards": {...}, "phase_label": "SCAN", "intervention_flags": [...] }
...
{ "type": "terminal", "outcome": "success|timeout|oob_clipped", "metrics": {...} }
```

Replay rule: given `header`, the env can be reconstructed deterministically
and step-by-step replay must match `step` entries byte-for-byte.

### 8.5 Determinism Rules

Seeded RNGs thread explicitly:

- `seed_base` → `seed_env` → `seed_initial_conditions`, `seed_dynamics`,
  `seed_probe`, `seed_intervention`
- `seed_base` → `seed_policy` → `seed_init_params`, `seed_training_noise`,
  `seed_evaluation_noise`

Each RNG is constructed by `splitmix64(seed_base, label)` so per-channel
seeds are reproducible without bookkeeping a giant seed table.

## 9. Phase 1 Readiness Checklist

This spec unlocks Phase 1 if the following are in place. Phase 1's exit
criterion ("HC-Signature works on the canonical task and degrades cleanly
across lower tiers; Oracle provides the privileged ceiling; the harness can
replay trials byte-for-byte") then becomes the test.

- [x] `mesa-core.mjs` implements `Env`, `signatureField`, sensor tiers,
      HC-Signature, and Oracle
- [x] Env exposes `applyProbe(...)` and `scheduleIntervention(...)` even if
      Phase 3 / 4 sweeps don't run yet — the *hooks* must exist
- [x] `scripts/mesa-harness.mjs` writes `manifest.json` and per-trial JSONL
- [x] `npm run mesa:phase1` runs HC-Signature across the four sensor tiers
      plus Oracle on `privileged-field` on a default seed slate (e.g., 32
      seeds) and emits `trial-outcomes.csv`
- [x] Byte-for-byte replay verified by a second run on the same seed slate
- [x] HC-Signature and Oracle meet the per-tier behavior expectations in §4.3
      after first-round tuning

## 10. Out of Scope for Phase 0

Recorded here so the scope doesn't drift:

- Learning-side code (Phase 2 owns this — PPO loop, BC loop, optimizer
  bookkeeping).
- Implementation of the probe slate; only the env affordances ship in
  Phase 0.
- Implementation of the intervention battery; only the env affordances
  ship in Phase 0.
- Phase 5 selection-pressure curriculum runner.
- Phase 6 interpretability tooling.
- Browser visualization at `mesa.html` (Phase 8).
- Any tidal-toy environment work (deferred to second-domain port).

## 11. Open Questions

These are flagged for resolution during Phase 1 implementation; if they
shift the spec, the spec is revised and re-versioned.

- **MLP vs transformer at Large tier.** PPO on a 4M-parameter MLP at 100M
  steps may or may not be stable. If unstable, switch Large to a small
  transformer (logged in Phase 2 manifest). Phase 0 spec keeps the MLP as
  the default to avoid premature complexity.
- **Texture-channel shape.** The default `(x_t mod L) / L` is a clean
  positional shortcut. Whether to also include a per-episode random
  texture token (a one-shot tag the policy might absorb as goal identity)
  is left for Phase 3 design.
- **Signature objective form.** Three candidate `J_S` forms (terminal,
  integrated, threshold) are pinned. Whether to add an inverse-distance
  shaping `J_S_inv(τ) = (1/T) Σ 1/(1 + ||x - x_goal||)` is open.
- **Dynamics noise default.** `σ_dyn = 0` is clean for Phase 1 verification
  but unrealistic for the rest of the program. A small positive default
  (e.g., 0.02) should be considered at Phase 2 kickoff.
- **Episode length tradeoff.** `T = 200` is long enough for HC-Signature
  to SCAN-SEEK-TRACK comfortably and short enough to keep RL training fast.
  If Phase 1 shows the SCAN-SEEK transition is the bottleneck, `T` may
  need to grow at Large tier.

## 12. Versioning and Updates

This document is the Phase 0 spec at version `v2`. Material changes —
environment math, controller architecture, sensor-tier definitions, reward
channels, probe or intervention affordances — bump the version and are
logged here:

- `v2` (2026-05-10): added the privileged-only Oracle controller as a fifth
  reference family and clarified that HC-Signature ignores privileged
  information.
- `v1` (2026-05-10): initial Phase 0 spec promoted from roadmap Decision
  Lock.
