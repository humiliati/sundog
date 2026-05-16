# Structural Failure Coincidence - Cut 2 C1 Controller Binding

Pre-registration: [`README.md`](README.md)  
Run spec: [`P2_RUN_SPEC.md`](P2_RUN_SPEC.md)  
Admission check: [`P2_SPEC_ADMISSION.md`](P2_SPEC_ADMISSION.md)  
Filed: **2026-05-15 (PT)**. Status: **C1 CLOSED ONLY**; Cut-2 execution
remains held on C2-C4 and a fresh admission re-check.

## Purpose

C1 requires Cut 2 to bind the project's actual existing Sundog
extremum-seeking / photometric controller, not an inlined proxy inverter. This
record names that controller and the exact repo evidence for the binding.

## Binding

The actual existing controller is:

```text
sundog.agents.photometric.PhotometricAgent
```

Source file:

```text
agents/photometric.py
```

Load form for any Cut-2 Python harness:

```python
from sundog.agents.photometric import PhotometricAgent
```

The controller class is defined in `agents/photometric.py` at line 48. Its
public interface for the current experiment family is:

- Constructor: `PhotometricAgent(...)`, lines 51-94.
- Reset: `reset(carrier_init=(0.0, 0.0))`, lines 96-108.
- Policy step: `act(obs: Observation) -> np.ndarray`, lines 120-200.

The controller receives `sundog.env_v2.Observation` and reads detector
intensities plus proprioceptive state. The source docstring explicitly says it
does **not** receive `laser_pos` or target detector position; it does know the
target detector index.

## Existing Usage Evidence

`experiments/run_baseline_comparison.py` is the canonical baseline runner:

- The condition list names `photometric : PhotometricAgent
  (extremum-seeking, no target access)` at lines 3-8.
- It imports `PhotometricAgent` from `sundog.agents.photometric` at line 49.
- `_make_photometric` instantiates `PhotometricAgent()`, deletes the oracle,
  and calls `reset(carrier_init=tuple(initial_obs.joint_angles))` at lines
  157-160.
- `CONDITION_FACTORIES` binds the `photometric` condition to that factory at
  lines 189-193.

`experiments/stress_tests.py` is corroborating runner evidence:

- It imports the same class at line 58.
- `make_agent(... condition == "photometric")` instantiates
  `PhotometricAgent(...)` with stressor-controlled scan duration and joint
  limit, then calls `reset(carrier_init=tuple(initial_obs.joint_angles))` at
  lines 140-146.

## Cut-2 Binding Rule

Any Cut-2 harness claiming to test the existing Sundog controller must:

1. Instantiate `sundog.agents.photometric.PhotometricAgent`.
2. Preserve the controller's `reset(...)` / `act(obs)` interface.
3. Feed it an `Observation`-compatible signal path specified by C2/C3.
4. Leave the controller internals untouched unless a later amendment explicitly
   declares a new controller under test.

Any inline route inverter, grid-search proxy, analytic inverse, or reimplemented
extremum seeker is **not** the existing controller. A Cut-2 run using such a
substitute is void under C1.

## Explicit Non-Bindings

These are not the C1 controller:

- `scripts/structural-failure-p2-harness.mjs` `routeEstimate(...)`.
- `analyticInverseEstimate(...)` in the same harness.
- `agents.baselines.DOADirectAgent`, `DOANoisyAgent`, or `RandomAgent`.
- `agents.tsa.TorqueShadowAgent`.

## Open Items

C1 only names the controller. It does **not** admit Cut-2 execution.

Remaining holds:

- **C2:** concrete non-invertible nuisance and pre-run bias demonstration.
- **C3:** pre-run decoy reachability through `J` plus in-sample temptation.
- **C4:** computed, derived vacuity audit.

After C2-C4 are filed, the P2 admission check must be re-run before any Cut-2
controller execution.
