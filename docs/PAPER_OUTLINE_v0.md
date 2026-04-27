# Paper outline v0: photometric mirror alignment without target access

Working title (TBD by team):
**Scan-Then-Extremum-Seek for Photometric Mirror Alignment Without Target
Position Access**

Alternative shorter titles to consider:
- *Photometric Pointing: A Scan-Then-Extremum-Seek Controller*
- *Mirror Alignment from Detector Intensities Alone*

## Target venue

**Primary candidate: IEEE RA-L (Robotics and Automation Letters).**
Six-page max, peer-reviewed, fast turnaround. The single-claim, single-
experiment shape of this work is exactly what RA-L was designed for.

**Backup: CoRL 2026 workshop track** (e.g. "Robot Learning Workshop" or a
workshop on indirect feedback / sensor-limited control). Workshop submission
takes the same content with looser length.

**Do not target:** main-track CoRL or ICRA — the contribution is too narrow,
and reviewers will (rightly) ask for more comprehensive empirics.

**Length budget for RA-L (6 pages two-column):**
- Abstract: 150 words
- Introduction: 3/4 page
- Related work: 1/2 page
- Method: 1.5 pages
- Experiments: 1.5 pages
- Results & discussion: 1 page
- Limitations & future work: 1/4 page
- References: 1/4 page

## Abstract (~150 words)

We study mirror alignment as an indirect-control task: an articulated
2-DoF pole with a mirrored end-effector must steer a reflected beam from
a fixed laser onto a designated photodetector, given only the per-step
intensities of an eight-detector floor ring and joint proprioception.
The agent has no Cartesian access to the laser or target detector
position. We propose a scan-then-extremum-seek controller: a Lissajous
joint-space scan locates the empirical maximum of the target intensity,
then perturb-and-observe extremum-seeking refines the lock. On 30 matched
scenes in a MuJoCo simulation, the photometric controller reaches terminal
target intensity statistically indistinguishable from a target-aware
analytic baseline (Mann-Whitney U=526, p=0.26), and significantly
outperforms a noisy-oracle baseline with 5 cm of position perception
error. The photometric controller pays for its lack of target access in
convergence time: median 188 steps versus the oracle's 12. We discuss
limitations and a planned occlusion extension.

## 1. Introduction (~3/4 page)

**Hook (1 paragraph).** Many practical alignment tasks deny the
controller direct knowledge of the target's Cartesian position: laser
interferometer alignment, beam steering in photonic packaging, antenna
pointing under occlusion, soft-robot end-effector positioning with
worn or absent extrinsic markers. Direct-observation control is an
upper-bound baseline, but the realistic case is the indirect one.

**The classical answer (1 paragraph).** Extremum-seeking control [Krstic
& Wang 2000] solves a single-input single-output version of this problem:
a small dither perturbs the input, demodulation recovers a gradient
estimate, the carrier moves up the gradient. ESC has known guarantees but
its standard formulation requires the controller to start within the
basin of attraction of the optimum. In multi-DoF embodied tasks, the
basin is small relative to the workspace, and the agent has no a priori
way to seed itself there.

**Our contribution (1 paragraph).** We present a scan-then-extremum-seek
architecture: a brief joint-space Lissajous scan acquires the empirical
maximum of the target signal, then ESC refines the lock. We instantiate
the architecture on a mirror-alignment task in MuJoCo and compare it
against a target-aware analytic baseline. The empirical claim is narrow
and quantitative: terminal alignment accuracy is statistically
indistinguishable between the photometric controller and the oracle, at
the cost of ~16x slower convergence.

**Roadmap (1 sentence).**

## 2. Related work (~1/2 page)

**Extremum-seeking control (~3 sentences).** Krstic & Wang 2000;
follow-up work on multi-input ESC [Ariyur & Krstic 2003]; sliding-mode
ESC variants. Most theory targets stability analysis around a known
optimum; less attention to global acquisition.

**Indirect feedback in optics (~3 sentences).** Lock-in detection
(Dicke), dithering for laser stabilization (PDH), beam-pointing
servos. Engineering practice for decades; cite the textbook treatment
[e.g., Hobbs *Building Electro-Optical Systems*].

**Sensor-driven robotics (~3 sentences).** Visual servoing
[Chaumette & Hutchinson]; image-based control with low-dimensional
features. Our task differs in that the feature is a sparse intensity
vector over a fixed detector array, not an image.

**Prior framing of this task (~3 sentences).** An earlier paper [Hughes
2024] proposed a "Sundog Alignment Theorem" framing for indirect-feedback
agent alignment, with qualitative claims about emergent resonance from
shadow geometry. The present work provides a concrete algorithmic
instantiation of that framing, an oracle baseline against which
performance can be measured, and empirical comparison; we make smaller
claims with quantitative support.

## 3. Method (~1.5 pages)

### 3.1 Task formulation (~1/3 page)

Articulated 2-DoF pole, base at origin, length 1.2 m, two perpendicular
hinges (range ±1.5 rad). Mirrored disc at the tip; mirror normal aligned
with the pole's local +z axis. Ceiling-mounted point laser source at
fixed (lx, ly, 2.5) with lx, ly varying per scene. Floor ring of eight
photodetectors at radius 1.2 m, evenly spaced; one designated as the
alignment target.

The agent's control objective: steer the joint angles such that the
beam — emitted from the laser, reflected by the mirror, propagating to
the floor — lands on or near the target detector.

### 3.2 Optics (~1/3 page)

**Reflection.** Standard mirror law: r = d - 2(d·n)n where d is the
incident propagation direction (laser to mirror) and n is the unit
mirror normal.

**Floor hit.** Ray-plane intersection with z = 0. Returns None if the
reflected beam goes upward (no floor hit, all detectors register zero).

**Detector intensity.** Gaussian-spot model:
I_i = exp(-‖H - D_i‖² / (2σ²))
with σ = 0.15 m. Detector intensities are computed analytically inside
the env's `_compute_intensities` so the agent's perceptual signal is
deterministic given joint state and laser position.

### 3.3 Observation and action (~1/3 page)

**Observation:** an 8-vector of detector intensities, the 2-vector of
joint angles, the 2-vector of joint velocities, and the 2-vector of
joint torque proxies. *Crucially, no Cartesian target position*.

**Action:** target joint angles in radians, clipped to ±1.45 rad (joint
limits with margin). Position-controlled actuators in MuJoCo with kp=80,
kv=8 servo to the target.

**Oracle (used by baselines only):** laser position and target detector
position. The photometric agent's API is constructed so that calling
get_oracle() raises by design rather than silently leaking ground
truth into agent code.

### 3.4 Photometric controller (~1/3 page)

Three phases:

1. **SCAN.** Lissajous trajectory over the joint workspace at incommensurate
   frequencies (ω_x = 1.7, ω_y = 2.3 rad/s, amplitude 1.4 rad). Run for a
   fixed wall-clock duration (4 s = 200 steps at 50 Hz). During SCAN,
   record the (joint_angles, intensity) pair with the highest target
   intensity observed.

2. **SEEK.** Jump the carrier to the best-seen joint configuration; dwell
   for 10 steps (200 ms) to let the joint position settle.

3. **TRACK.** Classical perturb-and-observe ESC [Krstic & Wang 2000].
   Carrier joint targets are augmented with small sinusoidal probes
   (ω_x = 6.0, ω_y = 8.5 rad/s, amplitude 0.05 rad). The DC-removed
   target intensity is demodulated against each probe, low-pass filtered
   to recover ∂I/∂θ_i, and the carrier moves along the gradient.

A re-acquire condition reverts to SCAN if the target intensity falls
below 0.05 for 30 consecutive steps during TRACK.

### 3.5 Baselines (~1/4 page)

- **DOA-direct.** Full oracle access. At episode start, computes
  optimal joint angles via grid-search seed + Nelder-Mead refinement
  on the analytic intensity at the target. Commands those angles every
  step. Upper bound on convergence speed.

- **DOA-noisy.** Same as DOA-direct but the laser xy and target xy
  in the oracle are corrupted by zero-mean Gaussian noise (σ = 5 cm),
  drawn once per episode. Tests sensitivity of the analytic solve to
  perception error.

- **Random.** Uniform random joint targets each step. Lower bound.

## 4. Experiments (~1.5 pages)

### 4.1 Setup (~1/3 page)

30 matched seeds. Per seed s, the laser xy is a deterministic function
of s (uniform [-0.4, 0.4]² with seed-keyed RNG); the initial joint
perturbation is a deterministic function of s (Gaussian σ = 0.05 rad).
All four conditions run on the same scene per seed.

500 steps per episode at 50 Hz control rate (10 s simulation time per
episode). MuJoCo timestep 0.005 s, frame_skip 4.

### 4.2 Metrics (~1/4 page)

- **Terminal target intensity.** Mean of detector_0 reading over the
  last 50 steps. Headline accuracy metric.
- **Time-to-threshold.** First step where target intensity > 0.9.
  Right-censored at 500 if never reached.
- **Terminal joint stability.** Std of each joint angle over the last
  100 steps, averaged across the two joints. Measures lock quality.

### 4.3 Statistical methodology (~1/4 page)

- Bootstrap percentile 95% CIs on per-condition means (5000 resamples).
- Mann-Whitney U two-sided test for between-condition comparisons
  on terminal target intensity. Reported U statistic and p-value.

### 4.4 Headline numbers table (~1/3 page)

| Condition    | n  | Terminal I (mean) | 95% CI         | Time-to-0.9 (median) | n_failed |
|---|---|---|---|---|---|
| photometric  | 30 | 0.945             | [0.936, 0.954] | 188                  | 0/30     |
| doa_direct   | 30 | 0.936             | [0.925, 0.947] | 11.5                 | 0/30     |
| doa_noisy    | 30 | 0.911             | [0.894, 0.927] | 14                   | 4/30     |
| random       | 30 | ~0                | ~[0, 0]        | 500 (censored)       | 30/30    |

**Pairwise tests on terminal target intensity:**
- photometric vs doa_direct: U=526, p=0.264 (not significantly different)
- photometric vs doa_noisy: U=649, p=0.003 (photometric better)
- photometric vs random: p=2×10⁻¹¹

### 4.5 Convergence plot

Figure: mean ± 95% bootstrap CI of target intensity vs step, per
condition. Shows the SCAN phase as two intensity bumps in the
photometric trace (around steps 30 and 175 where the Lissajous trajectory
crosses the optimum), the lock-in starting around step 220, and the
flat plateaus for DOA-direct and DOA-noisy. (Plot already generated:
sundog/results/analysis/convergence_curves.png)

## 5. Discussion (~3/4 page)

### Headline interpretation
On the noiseless scenes, **a controller that has no Cartesian access to
the target and only an eight-detector floor ring as photometric feedback
reaches the same terminal alignment as a target-aware controller with
analytic geometry**. The difference is in convergence speed: photometric
takes ~16x longer because it must scan to find the optimum before it can
refine. This is the trade-off the paper documents.

### A caveat that deserves call-out
Photometric's mean terminal (0.945) is fractionally higher than DOA-direct's
(0.936) — not statistically significant but notable. The reason is that
DOA-direct's analytic solve uses a half-vector formula seeded by grid
search; the joint-mirror-position coupling is approximated. Photometric
does naive global optimization via SCAN and locates the empirical
maximum directly. So in this geometry, "not knowing the target" is not
a disadvantage — the SCAN phase makes the agent into an effective
zeroth-order optimizer.

### Why this matters
The result is small but it inverts the standard reading of indirect
feedback: instead of "indirect feedback degrades performance vs. direct
feedback," we observe equality on terminal accuracy, with the cost paid
in time. For applications where the controller can afford a few seconds
of acquisition (laser alignment in a lab, beam steering at startup,
calibration routines), the indirect-feedback approach is effectively
free, and it removes the perception-system requirement entirely.

## 6. Limitations and future work (~1/4 page)

- **Geometry is fixed.** Single laser, single target detector index,
  static detector ring. Real applications would have time-varying
  geometry; the SCAN phase would need to repeat.
- **No detector noise.** Real photodiodes have shot noise, dark current,
  amplifier 1/f noise. Adding noise would degrade ESC's gradient estimate
  and bias the SCAN's argmax. We sketch this in the planned occlusion
  extension (Phase-2).
- **Hand-tuned parameters.** Probe frequencies, scan duration, gradient
  gain are hand-tuned. Open question whether these can be learned.
- **2-DoF only.** Embodied agents have many more degrees of freedom; the
  Lissajous SCAN scales poorly with action dimensionality.
- **Geometric optics only.** No diffraction, no spectrum, no polarization.

**Phase-2: occlusion robustness.** A static block on the floor between
mirror and detector ring partially occludes the reflected beam. The
photometric agent must align around the occlusion. Design memo:
[supplementary, sundog/docs/PHASE2_BLOCKS_DESIGN.md].

## 7. References (~1/4 page, ~6-8 entries)

1. Krstic, M., Wang, H.-H. (2000). *Stability of extremum seeking
   feedback for general nonlinear dynamic systems.* Automatica 36(4).
2. Ariyur, K. B., Krstic, M. (2003). *Real-time optimization by
   extremum-seeking control.* Wiley.
3. Todorov, E., Erez, T., Tassa, Y. (2012). *MuJoCo: A physics engine
   for model-based control.* IROS.
4. Chaumette, F., Hutchinson, S. (2006). *Visual servo control I: Basic
   approaches.* IEEE Robotics & Automation Magazine 13(4).
5. Hobbs, P. C. D. (2009). *Building Electro-Optical Systems.* Wiley
   (chapter on lock-in detection).
6. Hughes, J. W. (2024). *The Sundog Alignment Theorem: Shadow Physics
   and Emergent Resonance for A.I.* Available at admin@stellaraqua.com.
   (Acknowledged as the qualitative framing this paper instantiates.)

## Honest internal review notes

**Strengths.** The claim is small and precise. The experiment is
reproducible from this artifact (env XML, env wrappers, agent code,
runner, analysis all committed). The statistical comparison is
appropriate (non-parametric, bootstrapped). The result is interpretable
and the trade-off (16x slower for equal accuracy) is the point of the paper.

**Weaknesses a reviewer will probe.**
- Why a Lissajous scan and not a more efficient covering trajectory?
- Why these probe frequencies and not others? Sensitivity analysis missing.
- Single task. No demonstration on a second alignment problem.
- No real-hardware experiment.
- The DOA-direct baseline uses an analytic solve with known approximation
  error; a stronger baseline would be a target-aware iterative controller.

**Risk register.** In review, the most likely rejection grounds are:
(1) "incremental application of ESC" — counter by emphasizing the
SCAN-SEEK-TRACK architecture and the explicit comparison to oracle;
(2) "no real-world validation" — counter by positioning as a controlled
study in simulation, with hardware in future work; (3) "claim is too
narrow" — embrace it. The paper is small *on purpose* given the scope
of last cycle's reception.

**Pre-submission checklist.**
- [ ] One ablation: vary scan duration (1, 2, 4, 8 s) and report terminal
      accuracy. Adds a half-page but makes the choice defensible.
- [ ] Detector-noise stress test: add Gaussian noise (σ = 0.05) to
      observations and re-run. Mentioned in limitations but better as
      a result.
- [ ] Hardware-relevance section: explicitly cite a real beam-alignment
      task whose constraints match our setup.
- [ ] Authors and affiliations.
- [ ] Cite Hughes 2024 honestly (not buried).
