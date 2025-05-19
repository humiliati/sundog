"""
sundog_env.py

MuJoCo simulation wrapper for the Sundog Theorem prototype.
Simulates an articulated pole aligning to an overhead target via indirect cues
(torque and shadow alignment).

Author: Humiliati
License: unlicence 
"""

import mujoco
import mujoco.viewer
import numpy as np

class SundogEnv:
    def __init__(self, model_path: str):
        """Initialize the environment with a given .xml model path."""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None
        self.ctrl_range = [(-1, 1)] * self.model.nu

    def reset(self):
        """Reset simulation to initial state."""
        mujoco.mj_resetData(self.model, self.data)
        return self._get_obs()

    def step(self, ctrl):
        """Apply control and advance simulation."""
        ctrl = np.clip(ctrl, -1, 1)
        self.data.ctrl[:] = ctrl
        mujoco.mj_step(self.model, self.data)
        return self._get_obs(), self._compute_reward(), False, {}

    def _get_obs(self):
        """Return key observations."""
        return {
            "tip_pos": self.data.site("tip").xpos.copy(),
            "laser_pos": self.data.site("laser_dot").xpos.copy(),
            "torque": self.data.qfrc_actuator.copy()
        }

    def _compute_reward(self):
        """Reward = -alignment error + torque bonus."""
        tip = self.data.site("tip").xpos
        laser = self.data.site("laser_dot").xpos
        torque = self.data.qfrc_actuator
        return -np.linalg.norm(tip - laser) + 0.1 * np.linalg.norm(torque)

    def render(self):
        """Launch passive viewer (one-time only)."""
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        """Close viewer if open."""
        if self.viewer:
            self.viewer.close()


if __name__ == "__main__":
    # Test harness for local development
    env = SundogEnv("env/sundog_alignment.xml")
    obs = env.reset()
    for _ in range(500):
        ctrl = np.random.uniform(-0.5, 0.5, size=env.model.nu)
        obs, reward, _, _ = env.step(ctrl)
        print(f"Reward: {reward:.3f} | Tip: {obs['tip_pos']} | Torque: {obs['torque']}")
        env.render()
    env.close()
