"""Photometric agent: scan-best-then-track on the target detector.

This agent defends the manuscript's central claim. It receives only the
Observation object from env_v2 - detector intensities, joint angles, joint
velocities, joint torques. It does NOT receive laser_pos or target detector
position.

It does, by construction, know the *index* of the target detector
(TARGET_DETECTOR_INDEX = 0): in the alignment task, the agent is told
which photodetector to maximize, just as a human operator aligning a
laser knows which power meter they are trying to peak. It does not know
where that detector is in space.

Algorithm
---------
Three phases:

  SCAN: drive the joints along a Lissajous trajectory that covers the
    workspace (amplitude near joint limit, incommensurate frequencies).
    Track the best (joint_angles, intensity) pair seen so far. Run for
    a fixed wall-clock duration. With a vertical pole, the reflected
    beam goes back up and never hits the floor, so we have to tilt
    before any photometric signal exists.

  SEEK: jump the carrier to the best joints seen during SCAN and dwell
    there for a few control steps to let the joint position settle and
    the intensity stabilise. This avoids the failure mode where TRACK's
    DC tracker is contaminated by mid-scan transients.

  TRACK: classical perturb-and-observe extremum-seeking control (Krstic
    & Wang 2000). Small sinusoidal probes at incommensurate frequencies
    modulate the target intensity; demodulation + low-pass filtering
    recovers per-joint gradient estimates; carrier moves along the
    gradient.

If during TRACK the target intensity drops below a re-acquire threshold
for sustained steps (carrier wandered off the peak), the agent falls
back to SCAN. This handles bad initial seeds and adversarial scenes.
"""

from __future__ import annotations

import numpy as np

from sundog.env_v2 import Observation, TARGET_DETECTOR_INDEX


class PhotometricAgent:
    """Scan-best-then-track on detector_TARGET intensity. No target position access."""

    def __init__(
        self,
        target_detector_index: int = TARGET_DETECTOR_INDEX,
        dt: float = 0.02,
        # SCAN phase.
        scan_duration_s: float = 4.0,
        scan_amplitude: float = 1.4,
        scan_omega_x: float = 1.7,
        scan_omega_y: float = 2.3,
        # SEEK phase (settle after jump).
        seek_steps: int = 10,
        # TRACK phase (extremum-seeking).
        probe_amplitude: float = 0.05,
        omega_x: float = 6.0,
        omega_y: float = 8.5,
        intensity_lowpass_alpha: float = 0.02,
        gradient_lowpass_alpha: float = 0.05,
        gradient_gain: float = 8.0,
        # Re-acquire if track loses signal.
        reacquire_threshold: float = 0.05,
        reacquire_hold_steps: int = 30,
        joint_limit: float = 1.45,
    ):
        self.target_detector_index = int(target_detector_index)
        self.dt = float(dt)

        self.scan_duration_s = float(scan_duration_s)
        self.scan_amplitude = float(scan_amplitude)
        self.scan_omega_x = float(scan_omega_x)
        self.scan_omega_y = float(scan_omega_y)
        self.seek_steps = int(seek_steps)

        self.probe_amplitude = float(probe_amplitude)
        self.omega_x = float(omega_x)
        self.omega_y = float(omega_y)
        self.intensity_lowpass_alpha = float(intensity_lowpass_alpha)
        self.gradient_lowpass_alpha = float(gradient_lowpass_alpha)
        self.gradient_gain = float(gradient_gain)

        self.reacquire_threshold = float(reacquire_threshold)
        self.reacquire_hold_steps = int(reacquire_hold_steps)
        self.joint_limit = float(joint_limit)

        self.reset()

    def reset(self, carrier_init: tuple[float, float] = (0.0, 0.0)) -> None:
        self.t = 0.0
        self.phase = "scan"
        self.carrier = np.array(carrier_init, dtype=float)

        self._best_joints = np.zeros(2, dtype=float)
        self._best_intensity = -1.0
        self._seek_remaining = 0

        self._intensity_dc = 0.0
        self._gradient_estimate = np.zeros(2, dtype=float)
        self._dc_initialized = False
        self._below_threshold_count = 0

    def _enter_scan(self) -> None:
        self.phase = "scan"
        self._best_joints = np.zeros(2, dtype=float)
        self._best_intensity = -1.0
        self._intensity_dc = 0.0
        self._gradient_estimate[:] = 0.0
        self._dc_initialized = False
        self._below_threshold_count = 0
        self._scan_t0 = self.t  # remember when scan started

    def act(self, obs: Observation) -> np.ndarray:
        i_now = float(obs.detector_intensities[self.target_detector_index])

        # ------------------------------------------------------------------
        # SCAN
        # ------------------------------------------------------------------
        if self.phase == "scan":
            # Track best position over the entire scan window.
            if i_now > self._best_intensity:
                self._best_intensity = i_now
                self._best_joints = obs.joint_angles.copy()

            scan_t = self.t - getattr(self, "_scan_t0", 0.0)
            if scan_t < self.scan_duration_s:
                sx = np.sin(self.scan_omega_x * self.t)
                sy = np.sin(self.scan_omega_y * self.t)
                action = np.array([self.scan_amplitude * sx,
                                   self.scan_amplitude * sy])
                self.t += self.dt
                return np.clip(action, -self.joint_limit, self.joint_limit)

            # End of scan: jump to best seen.
            self.phase = "seek"
            self._seek_remaining = self.seek_steps
            # Seed track-phase state with the best snapshot.
            self.carrier = self._best_joints.copy()
            self._intensity_dc = self._best_intensity
            self._dc_initialized = True

        # ------------------------------------------------------------------
        # SEEK (dwell at best, let joints settle)
        # ------------------------------------------------------------------
        if self.phase == "seek":
            self._seek_remaining -= 1
            if self._seek_remaining <= 0:
                self.phase = "track"
            self.t += self.dt
            return np.clip(self.carrier, -self.joint_limit, self.joint_limit)

        # ------------------------------------------------------------------
        # TRACK (extremum-seeking control)
        # ------------------------------------------------------------------
        # Re-acquire if intensity has been below threshold for sustained steps.
        if i_now < self.reacquire_threshold:
            self._below_threshold_count += 1
            if self._below_threshold_count >= self.reacquire_hold_steps:
                self._enter_scan()
                # Fall through to scan path with current self.t.
                sx = np.sin(self.scan_omega_x * self.t)
                sy = np.sin(self.scan_omega_y * self.t)
                action = np.array([self.scan_amplitude * sx,
                                   self.scan_amplitude * sy])
                self.t += self.dt
                return np.clip(action, -self.joint_limit, self.joint_limit)
        else:
            self._below_threshold_count = 0

        if not self._dc_initialized:
            self._intensity_dc = i_now
            self._dc_initialized = True

        self._intensity_dc += self.intensity_lowpass_alpha * (i_now - self._intensity_dc)
        i_ac = i_now - self._intensity_dc

        sx = np.sin(self.omega_x * self.t)
        sy = np.sin(self.omega_y * self.t)
        gx_inst = i_ac * sx
        gy_inst = i_ac * sy

        a = self.gradient_lowpass_alpha
        self._gradient_estimate[0] += a * (gx_inst - self._gradient_estimate[0])
        self._gradient_estimate[1] += a * (gy_inst - self._gradient_estimate[1])

        self.carrier += self.gradient_gain * self._gradient_estimate * self.dt

        action = np.array([
            self.carrier[0] + self.probe_amplitude * sx,
            self.carrier[1] + self.probe_amplitude * sy,
        ])
        self.t += self.dt
        return np.clip(action, -self.joint_limit, self.joint_limit)
