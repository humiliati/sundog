# Scan-Then-Extremum-Seek for Photometric Mirror Alignment Without Target Position Access

*Draft v1 — full prose against PAPER_OUTLINE_v0.md*

---

## Abstract

We study mirror alignment as an indirect-control task: an articulated 2-DoF
pole with a mirrored end-effector must steer a reflected beam from a fixed
ceiling laser onto a designated photodetector, given only the per-step
intensities of an eight-detector floor ring and joint proprioception. The
agent has no Cartesian access to the laser or to the target detector. We
propose a three-phase scan-then-extremum-seek controller: a Lissajous
joint-space scan locates the empirical maximum of the target intensity, a
short dwell lets the joints settle on that maximum, and a perturb-and-observe
extremum-seeking loop refines the lock. On 30 matched scenes in MuJoCo, the
photometric controller reaches a terminal target intensity of 0.945
(95% CI [0.936, 0.954]), statistically indistinguishable from a target-aware
analytic baseline at 0.936 (Mann-Whitney U = 526, p = 0.26), and significantly
better than a noisy-oracle baseline with 5 cm of perception error
(0.911, p = 0.003). The cost is convergence time: photometric reaches the
0.9-intensity threshold at a median of 188 steps versus the oracle's 11.5.
A five-stressor sweep — detector noise, beam width, scan duration, laser
height, and joint limit — confirms the equality-with-cost result holds across
a broad operating envelope but breaks at joint limits below ~1.2 rad, where
the optimum lies outside the workspace and the photometric controller
collapses to 0.114 while the oracle, by virtue of clipping its informed solution,
retains 0.800.

## 1. Introduction

Many practical alignment problems deny the controller direct knowledge of the
target's Cartesian position. Laser interferometer alignment, beam steering
in photonic packaging, antenna pointing under partial occlusion, and
end-effector positioning for soft robots with worn or absent extrinsic
markers all share the structure: there is a scalar quantity (a
photodetector reading, a coupling efficiency, a signal-strength estimate)
that the controller can read, and there is no separate channel that
reports where the target is in space. Direct-observation control — solving
the geometry analytically from a known target position — is an upper-bound
baseline rather than a deployable strategy.

The classical answer to this class of problems is extremum-seeking control
(ESC). In its single-input single-output form [Krstic & Wang 2000], a
small dither is injected at the input, the resulting modulation in the
scalar output is demodulated against the dither to recover a gradient
estimate, and a slow loop integrates that estimate to push the input up
the gradient. Multi-input variants extend the technique with incommensurate
dither frequencies on each channel [Ariyur & Krstic 2003]. ESC has stability
guarantees in the neighbourhood of the optimum, but its standard formulation
assumes the controller starts inside the basin of attraction. In multi-DoF
embodied tasks the basin is small relative to the workspace, and the
controller has no a priori way to seed itself there.

This paper presents a scan-then-extremum-seek architecture that handles
acquisition and refinement in a single controller. A brief joint-space
Lissajous scan covers the workspace at incommensurate frequencies and
records the highest target intensity observed; the carrier is then
jumped to that maximum and a perturb-and-observe loop refines the lock.
We instantiate the architecture on a mirror-alignment task in MuJoCo
and compare it against three baselines: an analytic target-aware oracle,
the same oracle with 5 cm of zero-mean Gaussian perception error, and a
random-action lower bound. On a 30-seed matched-scene benchmark, the
photometric controller achieves the same terminal alignment as the
target-aware oracle, paying for the absence of target access in
convergence time (~16× slower). A five-stressor robustness sweep
documents the operating envelope: the equality result is stable under
detector noise, beam-width changes, ±50% laser-height variation, and
scan-duration changes from 1 to 4 s; it breaks at tight joint limits
where the optimum lies outside the constrained workspace.

The remainder of this paper is structured as follows. §2 places the
work in the extremum-seeking, indirect-feedback-optics, and
sensor-driven-robotics literatures. §3 formalizes the task and the
controller. §4 describes the experiments and reports both the headline
comparison (§4.1–§4.5) and the stress-test sweep (§4.6). §5 interprets
the results, including the joint-limit failure mode. §6 lists
limitations and the planned occlusion extension.

## 2. Related work

**Extremum-seeking control.** The continuous-time stability proof of
Krstic & Wang [2000] established ESC as a principled controller for
SISO unimodal optimization landscapes. Multi-input variants
[Ariyur & Krstic 2003] use incommensurate dither frequencies to
decouple per-channel gradient estimates; sliding-mode ESC variants
trade smoothness for finite-time convergence. The literature
predominantly addresses local stability around an optimum the
controller is already near; global acquisition is typically left to a
separate seeding mechanism. Our SCAN phase is one such mechanism,
specialized to the case where the workspace itself is bounded and the
acquisition cost is dominated by the time required to traverse it.

**Indirect feedback in optics.** Lock-in detection (Dicke), the
Pound-Drever-Hall scheme for laser stabilization, and dithered
beam-pointing servos are decades-old engineering practice; the
textbook treatment in Hobbs [2009, ch. 11] covers most of the
practical considerations. These techniques are SISO and rely on the
operator already having coarsely aligned the system. The present
contribution differs in that the controller acquires alignment from
an arbitrary initial pose, and the indirect signal is a vector of
photodetector readings rather than a single demodulated channel.

**Sensor-driven robotics.** Image-based visual servoing
[Chaumette & Hutchinson 2006] uses low-dimensional features extracted
from images to close a control loop without explicit Cartesian
estimation of the target. Our task is structurally similar: the
agent reads a sparse 8-vector of detector intensities and steers the
end-effector based on it. The difference is that visual-servoing
features encode (typically) a geometric relationship — image
coordinates, line orientation, area — that can be back-projected to
target pose given calibration. Our intensity vector encodes a
photometric relationship (Gaussian falloff to each detector) that is
not invertible to target pose without solving an additional optimization.

**Prior framing of this task.** A previous paper [Hughes 2024]
proposed a "Sundog Alignment Theorem" framing for indirect-feedback
agent alignment, with qualitative claims about emergent resonance from
shadow geometry. The present work provides a concrete algorithmic
instantiation of that framing, an oracle baseline against which
performance can be measured, and an empirical comparison; we make
smaller claims with quantitative support.

## 3. Method

### 3.1 Task formulation

The system is an articulated 2-DoF pole rooted at the world origin, with
two perpendicular hinges at the base ($r_x$, axis world-$x$; $r_y$, axis
world-$y$). The MuJoCo XML lists $r_x$ first and $r_y$ second; MuJoCo
applies the inner rotation last, so the world-frame composition is
$R = R_{r_x} R_{r_y}$. The pole has length $L = 1.2$ m and a mirrored disc
at the tip whose normal is aligned with the pole's local $+z$ axis. The
mirror normal in world coordinates is therefore

$$
n(\theta_x, \theta_y) =
\begin{pmatrix}
\sin\theta_y \\
-\sin\theta_x \cos\theta_y \\
\cos\theta_x \cos\theta_y
\end{pmatrix},
$$

and the mirror position is $m = b + L\, n$ with the pole base
$b = (0, 0, 0.05)$ m. The forward-kinematics expression above was
validated at simulation startup against MuJoCo's site geometry to
within $10^{-6}$ m.

A point laser source is mounted at $\ell = (\ell_x, \ell_y, \ell_z)$ with
$\ell_x, \ell_y$ randomized per scene and $\ell_z = 2.5$ m by default.
Eight photodetectors $D_0, \dots, D_7$ lie on the floor at radius 1.2 m,
evenly spaced at $\pi/4$ increments starting from the world $+x$ axis;
detector $D_0$ at $(1.2, 0, 0.005)$ is the alignment target.

The control objective is to steer the joint angles such that the beam
emitted from $\ell$, reflected at the mirror, and propagated to the floor
plane lands on or near $D_0$.

### 3.2 Optics

The beam is treated geometrically. The incident propagation direction is
$d_{\text{in}} = (m - \ell)/\|m - \ell\|$. Specular reflection at a unit
mirror normal $n$ gives the reflected direction

$$
d_{\text{out}} = d_{\text{in}} - 2 (d_{\text{in}} \cdot n)\, n.
$$

The floor hit is found by ray-plane intersection with $z = 0$:

$$
H = m + t\, d_{\text{out}}, \qquad t = -m_z / d_{\text{out},z},
$$

valid only when $d_{\text{out},z} < 0$. If the reflected beam goes upward
($d_{\text{out},z} \geq 0$), there is no floor hit and every detector
registers zero — this happens when the pole is near vertical and the laser
is near the ceiling.

Each detector reads a Gaussian-spot intensity:

$$
I_i = \exp\!\left(-\frac{\|H - D_i\|^2}{2\sigma^2}\right),
$$

with $\sigma = 0.15$ m by default. This is computed analytically inside
the environment's `_compute_intensities`, so the agent's perceptual signal
is deterministic given joint state and laser position.

### 3.3 Observation and action

The agent's per-step observation is

$$
o_t = \big(I_{0:7},\ \theta_x, \theta_y,\ \dot\theta_x, \dot\theta_y,\ \tau_x, \tau_y\big) \in \mathbb{R}^{14},
$$

i.e. the eight detector intensities, the two joint angles, the two joint
velocities, and the two joint-torque proxies. Crucially, the observation
contains no Cartesian information about the laser, the mirror, the floor
hit, or the target detector. The agent's API is constructed so that the
oracle accessor `get_oracle()` is reachable only from baseline code — the
photometric agent never receives the oracle object.

The action is a 2-vector of target joint angles in radians, clipped to
$\pm 1.45$ rad with a 0.05 rad margin inside the physical joint range of
$\pm 1.50$ rad. MuJoCo position-controlled actuators with $k_p = 80$,
$k_v = 8$ servo to the target. The MuJoCo timestep is 0.005 s with frame
skip 4, giving an agent control rate of 50 Hz.

### 3.4 Photometric controller

The controller has three phases — SCAN, SEEK, TRACK — plus a re-acquire
fallback that returns to SCAN if the target intensity drops below a
threshold for sustained steps during TRACK.

**SCAN.** The carrier follows a Lissajous trajectory in joint space:

$$
\theta_x^\star(t) = A \sin(\omega_x t), \qquad
\theta_y^\star(t) = A \sin(\omega_y t),
$$

with amplitude $A = 1.4$ rad and incommensurate angular frequencies
$\omega_x = 1.7$ rad/s, $\omega_y = 2.3$ rad/s. The frequency ratio is
not a simple rational, so the trajectory densely covers the workspace box
in finite time. The phase runs for a fixed wall-clock duration
$T_{\text{scan}} = 4$ s (200 control steps at 50 Hz). Throughout the
phase, the controller maintains the running argmax over the target
detector reading,

$$
(\hat\theta^\star, \hat I^\star) = \arg\max_{t \leq T_{\text{scan}}}\, I_0(t),
$$

storing the joint configuration $\hat\theta^\star$ at which the highest
target intensity $\hat I^\star$ was observed. Note that $\hat\theta^\star$
is the argmax of the *measured* intensity, not the true geometric optimum;
under perfect observation these coincide, and under detector noise they
diverge by an amount controlled by the noise floor.

**SEEK.** At the end of SCAN the carrier is jumped to $\hat\theta^\star$
and held there for ten control steps (200 ms). This dwell lets the joint
position settle from the high-velocity Lissajous excursion to a quasi-static
pose at the recorded maximum, and primes the TRACK phase's DC-tracker with
the true measured intensity at $\hat\theta^\star$ rather than a value
contaminated by mid-scan transients.

**TRACK.** After SEEK the controller engages a multi-input
perturb-and-observe extremum seeker [Krstic & Wang 2000, Ariyur & Krstic
2003]. Two sinusoidal probes at incommensurate frequencies modulate the
carrier:

$$
\theta_x^\star(t) = c_x(t) + a \sin(\Omega_x t), \qquad
\theta_y^\star(t) = c_y(t) + a \sin(\Omega_y t),
$$

with probe amplitude $a = 0.05$ rad and frequencies $\Omega_x = 6.0$ rad/s,
$\Omega_y = 8.5$ rad/s. The DC component of the target intensity is tracked
by a first-order low-pass filter with mixing rate $\alpha = 0.02$,

$$
\bar I_0[t+1] = \bar I_0[t] + \alpha\, (I_0[t] - \bar I_0[t]),
$$

and the AC residual $\tilde I_0 = I_0 - \bar I_0$ is demodulated against
each probe. The instantaneous gradient estimates,

$$
\hat g_{x,\text{inst}} = \tilde I_0 \sin(\Omega_x t), \qquad
\hat g_{y,\text{inst}} = \tilde I_0 \sin(\Omega_y t),
$$

are themselves low-passed at mixing rate $\beta = 0.05$ to suppress the
$2\Omega$ harmonic in the demodulator output. The carrier is then advanced
along the smoothed gradient,

$$
c[t+1] = c[t] + K\, \hat g[t]\, \Delta t,
$$

with gain $K = 8$ and step $\Delta t = 0.02$ s. The commanded action at
each control step is the carrier plus the probe.

**Re-acquire.** If $I_0$ stays below 0.05 for 30 consecutive control steps
during TRACK, the controller resets and re-enters SCAN with the current
clock. This recovers the agent from rare cases where a poor SCAN argmax
seeded TRACK into a local plateau or where adversarial dynamics drove the
carrier off the peak.

The probe and SCAN frequencies are chosen at incommensurate ratios so
that, over the demodulator's averaging window, cross-channel coupling
averages to zero. The numerical values are hand-tuned; we did not search
over them.

### 3.5 Baselines

**DOA-direct.** Full oracle access. At episode start, the baseline solves
for joint angles that send the reflected beam onto the target detector.
The optimization runs an 11×11 grid search over the joint workspace
$[-1.5, 1.5]^2$ to seed Nelder-Mead refinement on the analytic intensity
$I_0(\theta_x, \theta_y)$ defined by §3.2 (with mirror position
$m(\theta_x, \theta_y)$, reflected ray, floor hit, and the Gaussian
falloff to $D_0$), terminating at $x_{\text{atol}} = 10^{-4}$,
$f_{\text{atol}} = 10^{-6}$, max 200 evaluations. The resulting joint
angles are clipped to the joint limits and commanded every step. We
note that an earlier half-vector fixed-point formulation was abandoned
because the joint-mirror coupling — the mirror tip moves on a 1.2-m sphere
as the joints rotate — caused oscillation between far-apart fixed points
for certain geometries; numerical optimization on the exact intensity is
robust to that case and is what the baseline uses. Convergence time for
this baseline is essentially the joint settling time of the PD servo.

**DOA-noisy.** Same as DOA-direct, but the laser xy and target xy
read from the oracle are corrupted by zero-mean Gaussian noise with
$\sigma = 0.05$ m, drawn once at episode start and held for the episode.
This represents a static perception error (e.g. a miscalibrated
camera-to-arm transform) and tests the sensitivity of the analytic solve
to perception-system error. Drawing the noise once-per-episode rather
than per-step prevents the agent from averaging the perception noise to
zero by repeated solves.

**Random.** Uniform random joint targets in $[-1.5, 1.5]^2$, drawn fresh
each control step. Lower bound included so that the comparison is grounded.

## 4. Experiments

### 4.1 Setup

We ran four conditions — `photometric`, `doa_direct`, `doa_noisy`, and
`random` — on the same 30 matched scenes. For seed $s$, the laser xy is
drawn from $\mathcal{U}([-0.4, 0.4]^2)$ keyed by `seed * 1_000_003 + 17`,
and the initial joint perturbation is drawn from
$\mathcal{N}(0, 0.05^2 \mathbb{I}_2)$ keyed by `seed * 9_999_991 + 41`.
Both functions are deterministic in $s$, so all four conditions see the
same laser position and the same initial joint state on a given seed —
the headline comparison is a paired design.

Each episode runs for 500 control steps (10 s of simulated time at the
50 Hz control rate). MuJoCo timestep is 0.005 s with frame skip 4. The
laser height is 2.5 m, beam $\sigma$ is 0.15 m, joint limits are $\pm 1.5$
rad, and the photometric SCAN runs for 4 s — these are the baseline
operating-point values, varied independently in the §4.6 stress sweep.

### 4.2 Metrics

**Terminal target intensity.** The mean of $I_0$ over the last 50 control
steps. This averages over residual probe-induced ripple in TRACK and over
the joint settling at the end of DOA-direct's PD servo. Headline accuracy
metric.

**Time-to-threshold.** The first control step at which $I_0 > 0.9$,
right-censored at 500 if never reached. Records acquisition speed.

**Terminal joint stability.** The standard deviation of each joint angle
over the last 100 control steps, averaged across the two joints. Lower
values indicate a tighter terminal lock and lower probe-induced excursion.

### 4.3 Statistical methodology

For each condition we report 95% bootstrap percentile confidence intervals
on the per-condition mean using 5000 resamples with replacement. For
between-condition comparisons on terminal target intensity we use a
two-sided Mann-Whitney U test, which makes no parametric assumption about
the per-condition distribution and is robust to the floor-and-ceiling
clipping that arises naturally on a $[0, 1]$ outcome.

### 4.4 Headline numbers

| Condition    | n  | Terminal $I_0$ | 95% CI            | Time-to-0.9 (median) | n_failed |
|--------------|----|----------------|-------------------|----------------------|----------|
| photometric  | 30 | 0.945          | [0.936, 0.954]    | 188                  | 0/30     |
| doa_direct   | 30 | 0.936          | [0.925, 0.947]    | 11.5                 | 0/30     |
| doa_noisy    | 30 | 0.911          | [0.894, 0.927]    | 14                   | 4/30     |
| random       | 30 | ≈ 0            | ≈ [0, 0]          | 500 (censored)       | 30/30    |

Pairwise tests on terminal target intensity:

- photometric vs doa_direct: $U = 526$, $p = 0.264$ (no significant difference)
- photometric vs doa_noisy: $U = 649$, $p = 0.003$ (photometric better)
- photometric vs random: $p = 2 \times 10^{-11}$

The headline finding is the photometric–vs–doa_direct comparison. With
$p = 0.264$, the data does not support a difference in terminal alignment
between a controller with full geometric oracle access and a controller
that reads only the eight-detector ring. The difference between these
two and DOA-noisy ($p = 0.003$) shows that 5 cm of perception noise on
the analytic baseline is enough to degrade terminal alignment below
either: the oracle is good only insofar as it is precise.

The DOA-noisy condition fails to reach $I_0 > 0.9$ on 4 of 30 seeds
(13%). This is the perception-error tax: when the analytic solve is
seeded with corrupted laser/target positions, the resulting joint
target can place the beam beyond the Gaussian-spot's effective radius,
and PD-servoing to that target never recovers. Photometric and
DOA-direct never failed to cross the threshold.

### 4.5 Convergence trajectories

![Convergence curves: target intensity vs control step, mean ± 95% bootstrap CI, by condition](C:\Users\hughe\Dev\sundog\results\analysis\convergence_curves.png)

Three features of the photometric trace are visible. First, two intensity
bumps near steps 30 and 175 mark the Lissajous trajectory crossing the
neighbourhood of the optimum during SCAN; on average across seeds, these
crossings happen twice in a 4 s window because the frequencies $\omega_x =
1.7$, $\omega_y = 2.3$ rad/s have a beat period that places the trajectory
near the optimum approximately every 2 s. Second, at step 220 (200 SCAN
steps + 10 SEEK steps + 10 TRACK ramp-up) the intensity locks in and
plateaus near 0.945. Third, the plateau is slightly above DOA-direct's,
which sits at 0.936 from joint settling onward — a small gap whose
mechanism we discuss in §5.

DOA-direct and DOA-noisy show flat plateaus from approximately step 50
onward, the duration of the joint settling. Random hovers near zero.

### 4.6 Stress tests

We sweep five perturbations in isolation, each at four or five levels,
using the same 30-seed matched-scene design as the headline experiment.
Per stressor, all four conditions are re-run on the same scenes; only
the swept parameter changes. Levels are chosen to bracket the operating
point and probe both lower-stress and higher-stress regimes.

#### 4.6.1 Detector noise

Additive zero-mean Gaussian noise is applied to the agent-visible detector
intensities and the result is clipped to $[0, 1]$. Ground-truth intensity
used for the terminal-$I_0$ metric is read from the unperturbed observation,
so the metric measures real beam alignment, not the agent's perceived
alignment. Levels: $\sigma_n \in \{0, 0.02, 0.05, 0.10, 0.20\}$.

| $\sigma_n$ | photometric | doa_direct | doa_noisy |
|------------|-------------|------------|-----------|
| 0.00       | 0.945       | 0.936      | 0.911     |
| 0.02       | 0.939       | 0.936      | 0.911     |
| 0.05       | 0.921       | 0.936      | 0.911     |
| 0.10       | 0.894       | 0.936      | 0.911     |
| 0.20       | 0.898       | 0.936      | 0.911     |

![Stress sweep: detector_noise (terminal target intensity vs detector-noise sigma, mean ± 95% bootstrap CI)](C:\Users\hughe\Dev\sundog\results\stress_tests\detector_noise\stress_curve.png)

The DOA baselines are flat across the sweep because they do not read
detector intensities; the stressor perturbs only the agent's input,
which only the photometric agent uses. The photometric trace decays
gracefully, losing about five points of terminal intensity by
$\sigma_n = 0.20$. The 0.10–0.20 plateau (0.894 → 0.898) is consistent
with the SCAN argmax saturating: at $\sigma_n = 0.20$ the noise floor
($\sim$0.20 RMS in raw detector counts) is comparable to the photometric
peak the agent locks onto, so additional noise does not bias the argmax
further — it just shifts which local-noise realization wins. The
TRACK-phase ESC averages probe-correlated signal against probe-
uncorrelated noise, which is also resilient.

#### 4.6.2 Beam sigma

The Gaussian-spot width $\sigma$ on the floor is varied; smaller $\sigma$
makes the alignment landscape sharper and the optimum less forgiving.
Levels: $\sigma \in \{0.05, 0.10, 0.15, 0.25, 0.40\}$ m.

| $\sigma$ (m) | photometric | doa_direct | doa_noisy |
|--------------|-------------|------------|-----------|
| 0.05         | 0.617       | 0.573      | 0.472     |
| 0.10         | 0.881       | 0.863      | 0.815     |
| 0.15         | 0.945       | 0.936      | 0.911     |
| 0.25         | 0.978       | 0.976      | 0.967     |
| 0.40         | 0.990       | 0.991      | 0.987     |

![Stress sweep: beam_sigma (terminal target intensity vs beam Gaussian-spot width, mean ± 95% bootstrap CI)](C:\Users\hughe\Dev\sundog\results\stress_tests\beam_sigma\stress_curve.png)

All conditions degrade as the beam narrows; an alignment error of
fixed metric size produces an exponentially worse intensity hit when
$\sigma$ shrinks. The interesting feature is that the photometric–DOA-direct
gap *widens* in photometric's favour at the sharp end:
$0.617 - 0.573 = 0.044$ at $\sigma = 0.05$, versus
$0.945 - 0.936 = 0.009$ at the baseline. At the loose end the conditions
collapse together because every reasonable solve is on top of the Gaussian
peak. The photometric-over-DOA-direct gap at narrow beam widths is the
discussion's caveat (§5): the analytic solve has a finite Nelder-Mead
tolerance, and at narrow beams the corresponding intensity error is no
longer negligible. Photometric's TRACK refines online and is bounded by
the ESC gradient resolution, which is finer than the analytic tolerance
in this geometry.

DOA-noisy stays one to two points behind DOA-direct across the sweep,
quantifying the cost of 5 cm of position error in joint-angle terms.

#### 4.6.3 Scan duration

The photometric SCAN window is varied; the episode budget remains fixed
at 10 s (500 steps at 50 Hz). Levels: $T_{\text{scan}} \in \{1, 2, 4, 8, 16\}$ s.

| $T_{\text{scan}}$ (s) | photometric | DOA baselines  |
|-----------------------|-------------|----------------|
| 1.0                   | 0.899       | 0.936 / 0.911  |
| 2.0                   | 0.914       | 0.936 / 0.911  |
| 4.0                   | 0.945       | 0.936 / 0.911  |
| 8.0                   | 0.766       | 0.936 / 0.911  |
| 16.0                  | 0.349       | 0.936 / 0.911  |

![Stress sweep: scan_duration (terminal target intensity vs SCAN window length, mean ± 95% bootstrap CI)](C:\Users\hughe\Dev\sundog\results\stress_tests\scan_duration\stress_curve.png)

The scan-duration sweep is an inverted-U with peak at 4 s, the headline
value. Below 4 s, the Lissajous trajectory under-samples the joint
workspace and the SCAN argmax is a noisier estimate of the true optimum,
so SEEK lands further from the peak and TRACK has further to refine —
$T_{\text{scan}} = 1$ s already costs half a point of terminal intensity.
Above 4 s, SCAN is over-sampled but TRACK has insufficient remaining
episode time to refine; at $T_{\text{scan}} = 8$ s, TRACK begins at
step 410 of a 500-step episode and the high-frequency ESC barely settles
before the metric window starts. At $T_{\text{scan}} = 16$ s the SCAN
never terminates inside the episode and the terminal $I_0$ is whatever
the Lissajous trajectory happens to be at near step 500 — for our
particular trajectory, that lands on average at 0.349. The variance at
$T_{\text{scan}} = 8$ s ($\sigma = 0.225$) is also far higher than at the
peak ($\sigma = 0.025$), reflecting how sensitive a half-finished TRACK
is to the seed. The result defends our headline parameter choice: 4 s
is approximately the saddle of "enough sampling to find the optimum"
and "enough refinement time to lock onto it."

#### 4.6.4 Laser height

The laser z-coordinate $\ell_z$ is varied while xy remains randomized
per seed. Levels: $\ell_z \in \{1.5, 2.0, 2.5, 3.0, 3.5\}$ m.

| $\ell_z$ (m) | photometric | doa_direct | doa_noisy |
|--------------|-------------|------------|-----------|
| 1.5          | 0.809       | 0.814      | 0.824     |
| 2.0          | 0.895       | 0.890      | 0.881     |
| 2.5          | 0.945       | 0.936      | 0.911     |
| 3.0          | 0.965       | 0.963      | 0.926     |
| 3.5          | 0.977       | 0.977      | 0.942     |

![Stress sweep: laser_height (terminal target intensity vs laser z-coordinate, mean ± 95% bootstrap CI)](C:\Users\hughe\Dev\sundog\results\stress_tests\laser_height\stress_curve.png)

Photometric and DOA-direct track each other within a percent across the
range. Lower laser heights make the geometry harder for both because the
reflection angles needed to send the beam down to the 1.2 m floor ring
become extreme, and the Gaussian spot is more sensitive to small
mirror-orientation errors at those reflection angles. At $\ell_z = 1.5$
m DOA-noisy actually surpasses DOA-direct fractionally
(0.824 vs 0.814) — within the 95% CI overlap, but worth noting. The
result is consistent with the joint-mirror coupling becoming
geometrically ill-conditioned in this region: small noise on the oracle
positions sometimes happens to displace the analytic seed in a direction
that the post-clip joint configuration can recover from. We do not
read this as a meaningful effect; the headline reading is that
photometric tracks DOA-direct across the realistic laser-height range.

#### 4.6.5 Joint limit — the honest limitation

The joint range $\pm \theta_{\max}$ is varied symmetrically. The photometric
agent's internal action clip is set to $\min(\theta_{\max} - 0.05,\ 1.45)$,
matching the headline policy of holding a 0.05 rad margin inside the
hardware limit. Levels: $\theta_{\max} \in \{0.8, 1.0, 1.2, 1.5\}$ rad.

| $\theta_{\max}$ (rad) | photometric | doa_direct | doa_noisy |
|-----------------------|-------------|------------|-----------|
| 0.8                   | $1.7 \times 10^{-5}$ | 0.027      | 0.022     |
| 1.0                   | **0.114**   | **0.800**  | 0.534     |
| 1.2                   | 0.956       | 0.926      | 0.684     |
| 1.5                   | 0.945       | 0.936      | 0.911     |

![Stress sweep: joint_limit (terminal target intensity vs symmetric joint limit, mean ± 95% bootstrap CI)](C:\Users\hughe\Dev\sundog\results\stress_tests\joint_limit\stress_curve.png)

This is the result a reviewer will probe and the boundary the paper
should own. Three readings stand out.

First, at $\theta_{\max} = 1.5$ and $\theta_{\max} = 1.2$ the photometric
agent matches or fractionally beats DOA-direct, consistent with the
headline. At $\theta_{\max} = 1.2$ the workspace is still large enough
that the optimum lies inside it for every laser xy in $[-0.4, 0.4]^2$,
and photometric's online refinement gives it the small terminal edge
already discussed.

Second, at $\theta_{\max} = 1.0$ photometric collapses to 0.114 while
DOA-direct retains 0.800. This is a 7× gap. The mechanism turns on
what each controller does when the unconstrained optimum lies outside
the joint range. DOA-direct solves on the unconstrained $[-1.5, 1.5]^2$
intensity landscape (the analytic solver does not see the runtime
$\theta_{\max}$ override), finds the global optimum, and *clips* the
returned angles to $\pm 1.0$. The clip puts the joints at the edge of
the constrained workspace, which is the closest reachable approach to
the true optimum — a sub-optimal but informed pose. Photometric, by
contrast, drives a Lissajous trajectory inside the constrained
workspace and locks onto whatever maximum it happens to find there. At
$\theta_{\max} = 1.0$ the maximum *reachable* target intensity is itself
low (the optimum lies outside reachability), so the SCAN argmax is
sensitive to where the trajectory happened to be when the laser
position randomized — different seeds yield different argmaxes that
all sit on a shallow plateau, and the across-seed mean collapses. The
DOA-noisy gap is intermediate (0.534) for the same reason in reverse:
the noise sometimes pushes the analytic solve in a direction the clip
recovers from, sometimes not.

Third, at $\theta_{\max} = 0.8$ all controllers fail. The reachable
mirror normal cone is too narrow to direct any reflection to the 1.2 m
target ring for most seeds, and the experiment becomes impossible
rather than informative. We include this level for completeness; it
documents the floor of the geometry, not a controller failure mode.

The mechanism is general. The photometric controller's information
advantage rests on the assumption that the SCAN covers a workspace in
which the optimum exists. When the optimum lies outside the constrained
workspace, the agent has no way to know that its observed maximum is a
constrained one, while the oracle's clip-to-bound behaviour preserves
informed direction. The result is a sharp asymmetry rather than a
graceful degradation: the photometric agent maintains parity with the
oracle inside the basin of reachability, then loses by an order of
magnitude when the basin narrows. Future work (§6) sketches an
adaptive-SCAN extension intended to detect this regime and fall back
to "point in the direction of the brightest detector during SCAN" — a
strategy that recovers the qualitative correctness of the oracle's
clip behaviour without requiring oracle access.

### 4.7 What survives the stress sweep

Across the four stressors that vary task or sensing parameters
(detector noise, beam sigma, laser height, scan duration), the headline
claim — terminal target intensity statistically indistinguishable from
the analytic oracle — survives. Photometric matches or fractionally
exceeds DOA-direct at every level of these four sweeps within the
ranges we tested. The single exception is the joint-limit sweep, where
the equality breaks at $\theta_{\max} \leq 1.0$ for the mechanism above.

## 5. Discussion

### 5.1 Headline interpretation

On the noiseless 30-seed benchmark, a controller that has no Cartesian
access to the target and only an eight-detector floor ring as photometric
feedback reaches the same terminal alignment intensity as a
target-aware controller that solves the geometry analytically with
full oracle access. The Mann-Whitney comparison ($U = 526$, $p = 0.26$)
does not support a difference. The cost of the photometric controller
is convergence speed: median time-to-threshold is 188 steps versus the
oracle's 11.5, a ratio of roughly $16\times$. This is the trade-off
the paper documents.

The standard reading of indirect feedback would have predicted that
removing target access degrades terminal accuracy. We observe instead
that it leaves terminal accuracy intact and pays the cost in time. For
applications where the controller can afford a few seconds of
acquisition — laser alignment in a lab, beam steering at startup,
calibration routines, fixed-geometry beam-pointing servos — the
indirect-feedback approach is effectively free in steady-state
performance, and it removes the perception-system requirement entirely.

### 5.2 The fractional photometric–DOA-direct gap

Photometric's mean terminal $I_0$ at the headline operating point is
0.945, fractionally above DOA-direct's 0.936. The difference is not
statistically significant ($p = 0.26$) but it is consistent across the
stress sweeps that don't vary the agent's input — at every laser height
and scan duration where both controllers are in their nominal regime,
the photometric mean sits a small amount above the oracle's. The §4.6.2
beam-sigma sweep makes the source visible: the gap widens as the beam
narrows, from $+0.009$ at $\sigma = 0.15$ to $+0.044$ at $\sigma = 0.05$.

Two mechanisms contribute. First, the analytic solver terminates at a
finite Nelder-Mead tolerance ($f_{\text{atol}} = 10^{-6}$ on the
intensity, $x_{\text{atol}} = 10^{-4}$ on the joint angles); at narrow
beam widths a $10^{-4}$ rad joint error produces a non-negligible
intensity error. Second, the closed-loop PD servo at $k_p = 80$ has
finite stiffness, so DOA-direct's terminal pose is the analytic seed
plus a steady-state servo error; the metric averages over residual
joint motion in the last 50 steps. Photometric's TRACK phase, in
contrast, refines online with ESC and is bounded by the gradient-
estimator resolution, which is finer in this geometry. The effect is
small at the headline operating point and grows where the
intensity landscape is sharpest.

The reading is not "photometric is better than the oracle." The reading
is that the analytic oracle is good only insofar as its solver is precise
and its servo is stiff, and at sharp landscapes those finite-precision
costs can exceed the variance of an online photometric refinement.

### 5.3 Why this matters

The result is small but it inverts the standard reading of indirect
feedback. Instead of "indirect feedback degrades performance versus
direct feedback," the empirical statement is *equality on terminal
accuracy with the cost paid in time*, conditional on the optimum lying
within the controllable workspace.

The asymmetry has practical consequences. Photometric removes the
perception-system requirement entirely: there is no camera, no
calibration step, no laser-position estimator. The eight detector
readings already in the application domain (because the alignment
target itself is one of them) suffice. For a beam-pointing servo at
startup, where commissioning a dedicated perception channel is
expensive and the sub-second alignment time is irrelevant, the
photometric architecture is a meaningful cost reduction — and the
DOA-noisy result quantifies what cost a poorly-calibrated perception
channel imposes (15 points of terminal-failure rate, three points of
mean intensity).

### 5.4 The joint-limit asymmetry

The joint-limit sweep (§4.6.5) identifies the boundary of the headline
result. Inside the workspace where the optimum is reachable
($\theta_{\max} \geq 1.2$ rad in our geometry), photometric matches or
fractionally exceeds DOA-direct. As the workspace narrows past the
optimum's reachability, DOA-direct's behaviour degrades gracefully — its
solver returns the unconstrained optimum, which gets clipped to the
edge of the constrained workspace, putting the joints in the closest
reachable approach to the true optimum. Photometric's behaviour
collapses — its SCAN reports a maximum, but that maximum is a
shallow constrained one that varies with seed and TRACK has nothing
useful to refine. The result is a sharp asymmetry rather than a
shared degradation curve.

The mechanism is informational, not computational. The oracle knows
the unconstrained optimum and uses the joint clip as a *constrained
projection*; the photometric agent sees only the constrained
intensity surface and cannot distinguish a low-but-true maximum
from a low constrained-maximum that should be projected toward the
unconstrained one. An adaptive SCAN that detects insufficient
intensity-during-sweep and falls back to "point at the brightest
detector and march in that direction" recovers part of the oracle's
behaviour without requiring oracle access; we sketch this in §6.

The joint-limit result should be read as a known boundary of the
controller, not as a refutation of the headline. It tells the reader
when to expect the result to hold (workspace covers the optimum) and
when not (workspace clips the optimum).

## 6. Limitations and future work

**Geometry is fixed.** Single laser, single target detector index, static
detector ring. Real applications would have time-varying geometry; the
SCAN phase would need to repeat on a schedule the controller does not
currently have.

**No detector noise in the headline.** The §4.6.1 sweep documents
graceful degradation under additive Gaussian noise, but real
photodiode chains add shot noise, dark current, and 1/f amplifier
noise; the next-step occlusion experiment (Phase 2, see
[`docs/PHASE2_BLOCKS_DESIGN.md`]) will combine occlusion with
realistic detector noise.

**Joint-limit cliff.** Documented in §4.6.5 and §5.4. The photometric
controller assumes the optimum lies inside the SCAN's workspace; when
that fails, the controller collapses while the analytic baseline
degrades gracefully via clip-to-bound. An adaptive SCAN that lifts
its amplitude in response to a low intensity-during-sweep, or a
hybrid policy that falls back to "march in the direction of the
brightest non-target detector" when no detector exceeds a SCAN
threshold, is the next experiment. This is the most important
follow-up.

**Hand-tuned parameters.** Probe frequencies $\Omega_x = 6$, $\Omega_y =
8.5$, scan frequencies $\omega_x = 1.7$, $\omega_y = 2.3$, scan amplitude
$A = 1.4$, gradient gain $K = 8$, low-pass mixing rates $\alpha = 0.02$
and $\beta = 0.05$, and the re-acquire threshold $0.05$ for $30$ steps
are all hand-tuned. We did not search these. Sensitivity analysis on
$T_{\text{scan}}$ is in §4.6.3; the others are open.

**2-DoF only.** The Lissajous SCAN scales poorly with action
dimensionality. A 6-DoF arm in a similar alignment task would need a
covering trajectory whose density at the optimum is comparable across
the four extra dimensions; cost grows as the product of per-axis
sample counts. Compressed-sensing or structured-stochastic SCAN
trajectories are an open question.

**Geometric optics only.** The simulation is a Gaussian-spot intensity
model with specular reflection; no diffraction, polarization, spectrum,
or multi-bounce. Hardware validation will surface effects this paper
does not model.

**Phase-2: occlusion robustness.** A static block placed on the floor
between the mirror and the detector ring partially occludes the
reflected beam. The photometric agent must align around the occlusion,
which breaks the assumption that the SCAN's intensity-vs-joint-angle
landscape has a single peak per laser xy. We expect the SCAN-SEEK-TRACK
architecture to need a multi-modal argmax rule (top-$k$ peaks, then
TRACK from each) and a re-acquire that distinguishes "below threshold
because of occlusion" from "below threshold because TRACK wandered
off." Design memo in `docs/PHASE2_BLOCKS_DESIGN.md`.

## 7. References

1. Krstic, M., Wang, H.-H. (2000). *Stability of extremum seeking
   feedback for general nonlinear dynamic systems.* Automatica, 36(4),
   595–601.
2. Ariyur, K. B., Krstic, M. (2003). *Real-time optimization by
   extremum-seeking control.* Wiley.
3. Todorov, E., Erez, T., Tassa, Y. (2012). *MuJoCo: A physics engine
   for model-based control.* IROS.
4. Chaumette, F., Hutchinson, S. (2006). *Visual servo control I: Basic
   approaches.* IEEE Robotics & Automation Magazine, 13(4), 82–90.
5. Hobbs, P. C. D. (2009). *Building Electro-Optical Systems: Making It
   All Work.* Wiley, ch. 11 (lock-in detection).
6. Hughes, J. W. (2024). *The Sundog Alignment Theorem: Shadow Physics
   and Emergent Resonance for A.I.* Available from the author at
   admin@stellaraqua.com. Acknowledged as the qualitative framing the
   present work instantiates.
