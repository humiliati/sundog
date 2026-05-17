# Oracle Leakage Audit

Status: P0 launch receipt, 2026-05-17.

Scope: the canonical photometric mirror-alignment runner and controller path.
This receipt supports the public wording "without target-position access" for
the controller path. It does not claim that every possible caller of the
environment API is incapable of requesting oracle state.

## Bottom Line

No Cartesian target-position leakage was found in the canonical photometric
controller path.

The one correction made during this audit is wording, not behavior:
`env_v2.get_oracle()` exists and returns ground-truth state for the
target-aware baselines. Older paper-outline text implied that oracle access was
impossible through the environment API. That was too strong and is now corrected. The
actual guarantee is narrower: the canonical photometric factory discards the
oracle before constructing `PhotometricAgent`, and `PhotometricAgent.act(...)`
only receives an `Observation`.

## Visibility Table

| quantity | controller-visible | training-visible | metric-only | logging-only |
| --- | --- | --- | --- | --- |
| Detector intensity vector | yes, all 8 detector intensities via `Observation.detector_intensities` | no separate training path | yes, target detector intensity summarized | yes |
| Target detector index | yes, as the index to maximize (`TARGET_DETECTOR_INDEX = 0`) | no separate training path | yes | yes |
| Laser Cartesian position | no for photometric controller | no separate training path | available through oracle for baselines and scene setup | saved as `laser_xy` |
| Target detector Cartesian position | no for photometric controller | no separate training path | available through oracle for target-aware baselines | not saved in the canonical episode log |
| Joint angles / velocities / torques | yes | no separate training path | yes | yes |
| Reward | none returned to the controller | none; controller is hand-coded extremum seeking | terminal/trajectory intensity metrics only | yes |
| Success / termination | no auto-termination in env; runner uses fixed step budget | no separate training path | computed after rollout | yes |

## Trace

1. Environment observation excludes target coordinates.
   `env_v2.py:48-56` defines `Observation` with detector intensities, joint
   angles, joint velocities, and joint torques only. `env_v2.py:165-170`
   constructs that observation from `_compute_intensities()` and proprioception.

2. The environment still computes hidden geometry internally.
   `env_v2.py:173-184` reads the laser site and mirror state to compute detector
   intensities. This is the world simulator, not controller input.

3. Oracle state is explicit and separated.
   `env_v2.py:191-199` returns `Oracle(laser_pos, target_detector_pos,
   target_detector_index)`. `agents/baselines.py:44-57` and
   `agents/baselines.py:76-93` use that oracle in the target-aware analytic and
   noisy baselines.

4. The photometric controller reads intensity and proprioception only.
   `agents/photometric.py:5-12` states the access boundary, including the
   target-detector-index caveat. `agents/photometric.py:120-200` implements
   `act(obs)`: it reads `obs.detector_intensities[self.target_detector_index]`,
   `obs.joint_angles`, and internal controller state. It does not call
   `get_oracle()` and has no `Oracle` parameter.

5. The canonical runner discards oracle for the photometric condition.
   `experiments/run_baseline_comparison.py:124-125` obtains the oracle once per
   episode because all condition factories share one interface.
   `experiments/run_baseline_comparison.py:157-160` immediately `del oracle` in
   `_make_photometric`, constructs `PhotometricAgent()`, and resets it from
   initial joint angles. `experiments/run_baseline_comparison.py:134-138` then
   loops as `agent.act(obs)` followed by `env.step(action)`, logging target
   intensity after the action.

6. The stress-test runner follows the same split.
   `experiments/stress_tests.py:136-146` constructs `PhotometricAgent(...)` for
   the photometric condition without using `oracle`;
   `experiments/stress_tests.py:194-207` obtains oracle for the shared runner
   interface and baseline
   conditions, then drives the controller from perturbed observations.

## Public Wording Rule

Allowed:

> A controller without Cartesian target-position access aligns from sparse
> photometric feedback and proprioception in the canonical MuJoCo runner.

Avoid:

> The environment makes oracle access impossible.

> The agent has no target information of any kind.

The controller is told which detector/intensity channel to maximize. It is not
given the detector's Cartesian position or the laser-source coordinates.
